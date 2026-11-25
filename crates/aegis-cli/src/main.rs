//! AegisLoop-RS unified binary.

use aegis_perception::{peak_rss_bytes, SpatialConfig, SpatialPipeline};
use aegis_proto::{VlmAction, VlmResponse};
use aegis_recorder::{replay_ledger, verify_ledger};
use aegis_safety::SafetyConfig;
use aegis_sim::scenarios::{Scenario, WorkcellSimulator};
use aegis_sim::RuntimeConfig;
use aegis_vlm::{HttpVlmBackend, VlmBackend, VlmConfig, VlmError, VlmRequest};
use clap::{Parser, Subcommand};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(name = "aegis-cli", version, about = "Deterministic spatial edge-safety runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Live loop: consume camera frames from Zenoh, gate every action through System 1 + Safety.
    Run {
        #[arg(long, default_value = "http://127.0.0.1:8081/v1/chat/completions")]
        endpoint: String,
        #[arg(long, default_value = "smolvlm2-256m-q4_k_m")]
        model: String,
        #[arg(long, default_value = "aegisloop-audit.db")]
        db: PathBuf,
        #[arg(long, default_value_t = 1500)]
        timeout_ms: u64,
    },
    /// Headless four-scenario simulation over real Zenoh topics.
    Sim {
        #[arg(long, default_value_t = 90)]
        frames: u64,
        #[arg(long, default_value_t = 5)]
        interval_ms: u64,
        #[arg(long, default_value = "aegisloop-audit.db")]
        db: PathBuf,
        #[arg(long, default_value_t = 25)]
        dashboard_every: u64,
        #[arg(long)]
        real_vlm: bool,
        #[arg(long)]
        real_model: bool,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = "http://127.0.0.1:8081/v1/chat/completions")]
        endpoint: String,
        #[arg(long, default_value = "smolvlm2-256m-q4_k_m")]
        model: String,
        #[arg(long, default_value_t = 1500)]
        timeout_ms: u64,
    },
    /// CPU-only latency benchmark (synthetic or real ONNX weights).
    Bench {
        #[arg(long, default_value_t = 2000)]
        frames: u64,
        #[arg(long, default_value_t = 100)]
        warmup: u64,
        #[arg(long)]
        real_model: bool,
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
    /// Verify the SHA-256 chain of a flight recorder database.
    VerifyAudit { path: PathBuf },
    /// Replay a recorded session through a fresh safety engine.
    Replay { path: PathBuf },
}

fn init_tracing() {
    let filter = std::env::var("AEGIS_LOG").unwrap_or_else(|_| "warn".to_owned());
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn resolve_model_dir(explicit: Option<&Path>) -> PathBuf {
    if let Some(dir) = explicit {
        return dir.to_path_buf();
    }
    if let Ok(dir) = std::env::var("AEGIS_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    let mnt = PathBuf::from("/mnt/data/aegisloop-models");
    if mnt.is_dir() {
        return mnt;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_owned());
    PathBuf::from(home).join(".cache/aegisloop-models")
}

/// Mock System 2 used by `sim`. A hallucinated offset and a human intrusion
/// are deliberately injected to exercise the safety envelope.
struct ScenarioMockBackend;

impl VlmBackend for ScenarioMockBackend {
    fn propose<'a>(
        &'a self,
        request: &'a VlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<VlmResponse, VlmError>> + Send + 'a>> {
        Box::pin(async move {
            let action = if request.spatial.human_intrusion {
                VlmAction::Continue
            } else if request.spatial.anomaly_score >= 0.70 {
                VlmAction::DivertPart { bin_id: 3 }
            } else {
                VlmAction::RegraspOffset { dx_mm: 900.0, dy_mm: -40.0, dz_mm: 0.0 }
            };
            Ok(VlmResponse {
                action,
                confidence: 0.87,
                rationale: "scenario mock triage".to_owned(),
            })
        })
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Sim {
            frames,
            interval_ms,
            db,
            dashboard_every,
            real_vlm,
            real_model,
            model_dir,
            endpoint,
            model,
            timeout_ms,
        } => {
            let mut spatial = SpatialConfig::default();
            let mut real_path = None;
            if real_model {
                spatial.allow_scene_hints = false;
                spatial.depth_offset_mm = 300.0;
                spatial.depth_gain_mm = 1500.0;
                let dir = resolve_model_dir(model_dir.as_deref());
                let path = dir.join("depth-anything-v2-small.onnx");
                if !path.is_file() {
                    eprintln!("ONNX weights not found at {}", path.display());
                    std::process::exit(1);
                }
                real_path = Some(path);
            }
            let vlm = VlmConfig {
                endpoint: endpoint.clone(),
                model: model.clone(),
                timeout_ms,
                ..VlmConfig::default()
            };
            let config = RuntimeConfig {
                total_frames: frames,
                frame_interval_ms: interval_ms,
                database: db,
                dashboard_every,
                scenarios: aegis_sim::scenarios::default_sequence(frames),
                real_model: real_path,
                spatial,
                vlm: vlm.clone(),
                ..RuntimeConfig::default()
            };
            let label = match (real_model, real_vlm) {
                (true, true) => "END-TO-END REAL WEIGHTS (Depth Anything V2 + SmolVLM2)",
                (true, false) => "REAL DEPTH ONNX + mock VLM",
                (false, true) => "SYNTHETIC DEPTH + HTTP SmolVLM2",
                (false, false) => "SIMULATED (synthetic depth + mock VLM)",
            };
            let result = if real_vlm {
                match HttpVlmBackend::new(vlm) {
                    Ok(backend) => aegis_sim::run(config, backend).await,
                    Err(e) => {
                        eprintln!("VLM client init failed: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                aegis_sim::run(config, ScenarioMockBackend).await
            };
            match result {
                Ok(report) => print_runtime_report(&report, label),
                Err(e) => {
                    eprintln!("sim failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Command::Run { endpoint, model, db, timeout_ms } => {
            let vlm = VlmConfig { endpoint, model, timeout_ms, ..VlmConfig::default() };
            let backend = match HttpVlmBackend::new(vlm.clone()) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("VLM client init failed: {e}");
                    std::process::exit(1);
                }
            };
            let config = RuntimeConfig {
                generate_frames: false,
                total_frames: u64::MAX,
                database: db,
                vlm,
                ..RuntimeConfig::default()
            };
            match aegis_sim::run(config, backend).await {
                Ok(report) => print_runtime_report(&report, "LIVE (external camera + HTTP VLM)"),
                Err(e) => {
                    eprintln!("run failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Command::Bench { frames, warmup, real_model, model_dir } => {
            bench(frames, warmup, real_model, model_dir.as_deref())
        }
        Command::VerifyAudit { path } => match verify_ledger(&path) {
            Ok(report) => {
                println!(
                    "entries={} valid={} first_bad={:?} detail={}",
                    report.entries, report.valid, report.first_bad_index, report.detail
                );
                if !report.valid {
                    std::process::exit(2);
                }
            }
            Err(e) => {
                eprintln!("verify failed: {e}");
                std::process::exit(1);
            }
        },
        Command::Replay { path } => match replay_ledger(&path, SafetyConfig::default()) {
            Ok(report) => {
                println!(
                    "entries={} approved={} clamped={} vetoed={} mismatches={} deterministic={} first_mismatch={:?}",
                    report.entries,
                    report.approved,
                    report.clamped,
                    report.vetoed,
                    report.mismatches,
                    report.deterministic,
                    report.first_mismatch
                );
                if !report.deterministic {
                    std::process::exit(2);
                }
            }
            Err(e) => {
                eprintln!("replay failed: {e}");
                std::process::exit(1);
            }
        },
    }
}

fn print_runtime_report(report: &aegis_sim::RuntimeReport, label: &str) {
    println!("=== {label} ===");
    println!("frames processed      : {}", report.frames);
    println!("spatial states        : {}", report.spatial_states);
    println!("VLM results           : {}", report.vlm_results);
    println!("VLM accepted/fallback : {}/{}", report.vlm_accepted, report.vlm_fallbacks);
    println!("VLM bypass rate       : {:.2}%", report.vlm_bypass_rate * 100.0);
    println!("safety vetoes         : {}", report.vetoes);
    println!("safety clamps         : {}", report.clamps);
    println!(
        "S1 latency p50/p95/p99: {:.3}/{:.3}/{:.3} ms",
        report.p50_ms, report.p95_ms, report.p99_ms
    );
    println!("E-stop latency        : {:?} ms", report.estop_latency_ms);
    println!("peak RSS              : {:.1} MiB", report.peak_rss_mib);
    println!(
        "audit entries         : {} (chain valid: {})",
        report.audit_entries, report.audit_valid
    );
}

fn bench(frames: u64, warmup: u64, real_model: bool, model_dir: Option<&Path>) {
    let cfg = SpatialConfig::default();
    let pipeline = build_bench_pipeline(&cfg, real_model, model_dir);
    let mut pipeline = match pipeline {
        Ok(p) => p,
        Err(e) => {
            eprintln!("benchmark backend unavailable: {e}");
            std::process::exit(1);
        }
    };
    let mut sim = WorkcellSimulator::new(64, 64);
    let total = frames.saturating_add(warmup).max(1);
    let mut above_threshold = 0_u64;
    let started = Instant::now();
    for i in 0..total {
        let scenario = if real_model {
            Scenario::Nominal
        } else {
            let seq = i.saturating_sub(warmup);
            if seq == frames.saturating_sub(1) {
                Scenario::HumanIntrusion
            } else if seq == frames.saturating_mul(2) / 3 {
                Scenario::DefectivePart
            } else {
                Scenario::Nominal
            }
        };
        let frame = sim.next_frame(i + 1, scenario);
        match pipeline.process(&frame) {
            Ok(spatial) => {
                if spatial.escalation_score() >= VlmConfig::default().escalation_threshold {
                    above_threshold = above_threshold.saturating_add(1);
                }
            }
            Err(e) => {
                eprintln!("frame {} failed: {e}", i + 1);
                std::process::exit(1);
            }
        }
    }
    let elapsed = started.elapsed();
    let summary = pipeline.stats().summary();
    let bypass_pct = if frames == 0 {
        0.0
    } else {
        (frames.saturating_sub(above_threshold.min(frames))) as f64 / frames as f64 * 100.0
    };
    let fps = if elapsed.as_secs_f64() > 0.0 { frames as f64 / elapsed.as_secs_f64() } else { 0.0 };
    let rss = peak_rss_bytes().unwrap_or(0) as f64 / (1024.0 * 1024.0);
    let label = if real_model {
        "REAL MODEL (Depth Anything V2 Small ONNX, CPU)"
    } else {
        "SIMULATED (deterministic synthetic depth)"
    };
    println!("=== BENCH: {label} ===");
    println!("frames                 : {}", frames);
    println!("warmup                 : {}", warmup);
    println!("latency p50            : {:.3} ms", summary.p50_ms);
    println!("latency p95            : {:.3} ms", summary.p95_ms);
    println!("latency p99            : {:.3} ms", summary.p99_ms);
    println!("latency mean           : {:.3} ms", summary.mean_ms);
    println!("throughput             : {:.1} FPS", fps);
    println!("VLM bypass rate        : {:.2}%", bypass_pct);
    println!("peak RSS               : {:.1} MiB", rss);
    println!("engine                 : {}", pipeline.config().engine_name());
    println!("model dir              : {}", resolve_model_dir(model_dir).display());
    println!("budget (<2048 MiB)     : {}", if rss < 2048.0 { "PASS" } else { "FAIL" });
}

fn build_bench_pipeline(
    cfg: &SpatialConfig,
    real_model: bool,
    model_dir: Option<&Path>,
) -> Result<SpatialPipeline, String> {
    if !real_model {
        return SpatialPipeline::synthetic(cfg.clone()).map_err(|e| e.to_string());
    }
    #[cfg(feature = "onnx")]
    {
        let mut cfg = cfg.clone();
        cfg.allow_scene_hints = false;
        cfg.depth_offset_mm = 300.0;
        cfg.depth_gain_mm = 1500.0;
        let dir = resolve_model_dir(model_dir);
        let path = dir.join("depth-anything-v2-small.onnx");
        if !path.is_file() {
            return Err(format!(
                "ONNX weights not found at {} (run scripts/download_models.sh)",
                path.display()
            ));
        }
        SpatialPipeline::onnx(cfg, &path).map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (model_dir, cfg);
        Err("binary compiled without --features onnx; rebuild with `cargo build --release --features onnx`".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_proto::FrameMeta;

    #[test]
    fn model_dir_prefers_explicit_then_env_then_fallback() {
        let explicit = resolve_model_dir(Some(Path::new("/tmp/custom")));
        assert_eq!(explicit, PathBuf::from("/tmp/custom"));
        let resolved = resolve_model_dir(None);
        assert!(!resolved.as_os_str().is_empty());
    }

    #[tokio::test]
    async fn scenario_mock_emits_expected_actions() {
        let mut sim = WorkcellSimulator::new(8, 8);
        let mut levels = Vec::new();
        for (i, scenario) in [
            Scenario::Nominal,
            Scenario::DefectivePart,
            Scenario::VlmHallucination,
            Scenario::HumanIntrusion,
        ]
        .into_iter()
        .enumerate()
        {
            let frame = sim.next_frame(i as u64 + 1, scenario);
            let mut pipeline =
                SpatialPipeline::synthetic(SpatialConfig::default()).expect("pipeline");
            let spatial = pipeline.process(&frame).expect("spatial");
            let request = VlmRequest {
                seq: frame.meta.seq,
                timestamp_ns: frame.meta.monotonic_ns,
                frame_hash: frame.meta.frame_hash,
                spatial,
                image_rgb: Some(frame.pixels.clone()),
                image_width: frame.meta.width,
                image_height: frame.meta.height,
            };
            let response = ScenarioMockBackend.propose(&request).await.expect("mock response");
            levels.push(response.action);
        }
        assert!(matches!(levels[1], VlmAction::DivertPart { .. }));
        assert!(matches!(levels[2], VlmAction::RegraspOffset { .. }));
        assert_eq!(
            levels[3],
            VlmAction::Continue,
            "adversarial VLM must propose Continue during intrusion"
        );
    }

    #[test]
    fn frame_hash_is_stable_for_identical_pixels() {
        let a = FrameMeta::new(1, 1, "c", 2, 2, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "x");
        let b = FrameMeta::new(1, 1, "c", 2, 2, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "x");
        assert_eq!(a.frame_hash, b.frame_hash);
    }
}

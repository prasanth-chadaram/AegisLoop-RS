//! Headless robotic workcell simulator and live runtime.
//!
//! Frames are published on real Zenoh topics. The same code path handles
//! generated simulator frames and externally produced camera frames.

pub mod rss;
pub mod scenarios;

use aegis_perception::{peak_rss_bytes, SpatialConfig, SpatialPipeline};
use aegis_proto::{
    AuditEvent, AuditRecord, Disposition, FrameMeta, RgbFrame, SpatialState, VlmAction,
    VlmProposal, VlmResponse, TOPIC_ACTUATOR_CMD, TOPIC_AUDIT_EVENT, TOPIC_CAM_RGB, TOPIC_SPATIAL,
    TOPIC_VLM_INTENT,
};
use aegis_recorder::{verify_ledger, Recorder, VerifyReport};
use aegis_safety::{DecidedCommand, SafetyConfig, SafetyEngine};
use aegis_vlm::{
    OutcomeKind, SharedVlmStats, VlmBackend, VlmConfig, VlmRequest, VlmResult, VlmWorker,
};
use scenarios::{Scenario, WorkcellSimulator};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, warn};
use zenoh::pubsub::Publisher;
use zenoh::sample::Sample;

static CLOCK_START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Process-local monotonic nanoseconds used by every decision timestamp.
pub fn now_ns() -> u64 {
    CLOCK_START.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error("zenoh error: {0}")]
    Zenoh(String),
    #[error("perception error: {0}")]
    Perception(String),
    #[error("recorder error: {0}")]
    Recorder(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("configuration error: {0}")]
    Config(String),
}

impl From<aegis_perception::PerceptionError> for SimError {
    fn from(value: aegis_perception::PerceptionError) -> Self {
        Self::Perception(value.to_string())
    }
}

impl From<aegis_recorder::RecorderError> for SimError {
    fn from(value: aegis_recorder::RecorderError) -> Self {
        Self::Recorder(value.to_string())
    }
}

impl From<serde_json::Error> for SimError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    /// Generate synthetic frames inside this process. `false` = consume
    /// externally published camera frames from Zenoh.
    pub generate_frames: bool,
    pub total_frames: u64,
    pub frame_interval_ms: u64,
    pub scenarios: Vec<Scenario>,
    pub database: PathBuf,
    pub dashboard_every: u64,
    pub safety: SafetyConfig,
    pub vlm: VlmConfig,
    pub spatial: SpatialConfig,
    /// Optional real ONNX weights path. `None` keeps the offline estimator.
    pub real_model: Option<PathBuf>,
    /// Print every safety verdict with rule IDs (benchmarks / incident review).
    pub log_verdicts: bool,
    /// Benchmark-only: keep deterministic scene tags as anomaly hints while
    /// using real ONNX weights, so System 2 escalations can be exercised.
    /// Never enable this in a production cell.
    pub benchmark_scene_hints: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let total_frames = 90;
        let scenarios = scenarios::default_sequence(total_frames);
        Self {
            generate_frames: true,
            total_frames,
            frame_interval_ms: 5,
            scenarios,
            database: PathBuf::from("aegisloop-audit.db"),
            dashboard_every: 25,
            safety: SafetyConfig::default(),
            vlm: VlmConfig::default(),
            spatial: SpatialConfig::default(),
            real_model: None,
            log_verdicts: false,
            benchmark_scene_hints: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RuntimeReport {
    pub frames: u64,
    pub spatial_states: u64,
    pub vlm_results: u64,
    pub vlm_accepted: u64,
    pub vlm_fallbacks: u64,
    pub bypassed: u64,
    pub approved: u64,
    pub vetoes: u64,
    pub clamps: u64,
    pub drops: u64,
    pub vlm_bypass_rate: f64,
    pub estop_latency_ms: Option<f32>,
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    /// Rust process peak RSS (VmHWM of this process).
    pub peak_rss_mib: f64,
    pub rust_process_peak_rss_mib: f64,
    /// Peak VmHWM observed for any llama-server process during this run.
    pub llama_server_peak_rss_mib: f64,
    /// Peak of sampled (Rust VmRSS + llama-server VmRSS) during this run.
    pub combined_peak_rss_mib: f64,
    pub audit_entries: u64,
    pub audit_valid: bool,
}

fn sample_json<T: DeserializeOwned>(sample: &Sample) -> Result<T, SimError> {
    let text =
        sample.payload().try_to_string().map_err(|e| SimError::Serialization(e.to_string()))?;
    serde_json::from_str(&text).map_err(|e| SimError::Serialization(e.to_string()))
}

async fn publish<T: Serialize>(publisher: &Publisher<'_>, value: &T) -> Result<(), SimError> {
    let json = serde_json::to_string(value)?;
    publisher.put(json).await.map_err(|e| SimError::Zenoh(e.to_string()))
}

fn system1_halt_proposal(seq: u64, reason: &str, timestamp_ns: u64) -> VlmProposal {
    VlmProposal {
        seq,
        request_id: seq,
        timestamp_ns,
        trace_id: format!("system1-halt-{seq}"),
        model: "system1-direct".to_owned(),
        latency_ms: 0.0,
        response: VlmResponse {
            action: VlmAction::HaltCell { reason: reason.to_owned() },
            confidence: 1.0,
            rationale: "System 1 hard E-stop; System 2 bypassed".to_owned(),
        },
    }
}

/// Run the complete closed loop. Stops after `total_frames` (simulation) or on
/// Ctrl-C (live), whichever comes first.
pub async fn run<B: VlmBackend + Send + 'static>(
    config: RuntimeConfig,
    backend: B,
) -> Result<RuntimeReport, SimError> {
    config.safety.validate().map_err(|e| SimError::Config(e.to_string()))?;
    config.spatial.validate().map_err(|e| SimError::Config(e.to_string()))?;

    let session =
        zenoh::open(zenoh::Config::default()).await.map_err(|e| SimError::Zenoh(e.to_string()))?;
    let spatial_pub = session
        .declare_publisher(TOPIC_SPATIAL)
        .await
        .map_err(|e| SimError::Zenoh(e.to_string()))?;
    let intent_pub = session
        .declare_publisher(TOPIC_VLM_INTENT)
        .await
        .map_err(|e| SimError::Zenoh(e.to_string()))?;
    let cmd_pub = session
        .declare_publisher(TOPIC_ACTUATOR_CMD)
        .await
        .map_err(|e| SimError::Zenoh(e.to_string()))?;
    let audit_pub = session
        .declare_publisher(TOPIC_AUDIT_EVENT)
        .await
        .map_err(|e| SimError::Zenoh(e.to_string()))?;
    let cam_sub = session
        .declare_subscriber(TOPIC_CAM_RGB)
        .await
        .map_err(|e| SimError::Zenoh(e.to_string()))?;

    let mut pipeline = match config.real_model.as_deref() {
        Some(path) => {
            #[cfg(feature = "onnx")]
            {
                SpatialPipeline::onnx(config.spatial.clone(), path)?
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(SimError::Config(
                    "real ONNX backend requires --features onnx".to_owned(),
                ));
            }
        }
        None => SpatialPipeline::synthetic(config.spatial.clone())?,
    };
    let mut safety = SafetyEngine::new(config.safety);
    let mut recorder = Recorder::open(&config.database)?;
    let stats = Arc::new(SharedVlmStats::default());
    let (vlm_tx, vlm_rx) = mpsc::channel::<VlmRequest>(config.vlm.queue_capacity.max(1));
    let mut vlm_tx = Some(vlm_tx);
    let (result_tx, mut result_rx) = mpsc::channel::<VlmResult>(16);
    let worker = VlmWorker::new(backend, config.vlm.clone(), vlm_rx, result_tx, stats.clone());
    let worker_handle = tokio::spawn(worker.run());

    let mut sim = WorkcellSimulator::new(64, 64);
    let generator = config.generate_frames;
    let total = config.total_frames;
    let interval = Duration::from_millis(config.frame_interval_ms);
    let generator_handle = if generator {
        let scenarios = config.scenarios.clone();
        let session2 = session.clone();
        Some(tokio::spawn(async move {
            for (i, scenario) in scenarios.iter().enumerate().take(total as usize) {
                let frame = sim.next_frame(i as u64 + 1, *scenario);
                let payload = match serde_json::to_string(&frame) {
                    Ok(p) => p,
                    Err(e) => {
                        warn!("frame serialization failed: {e}");
                        return;
                    }
                };
                if let Err(e) = session2.put(TOPIC_CAM_RGB, payload).await {
                    warn!("frame publish failed: {e}");
                    return;
                }
                tokio::time::sleep(interval).await;
            }
        }))
    } else {
        None
    };

    let mut report = RuntimeReport::default();
    let mut frames = 0_u64;
    let mut pending_vlm = 0_u64;
    let mut stop = false;
    let mut combined_peak_bytes = 0_u64;
    let mut llama_hwm_bytes = 0_u64;
    loop {
        if stop && pending_vlm == 0 {
            break;
        }
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl-C received; draining in-flight decisions");
                stop = true;
                let _ = vlm_tx.take();
            }
            maybe_sample = cam_sub.recv_async(), if !stop => {
                let sample = match maybe_sample {
                    Ok(s) => s,
                    Err(e) => {
                        warn!("camera subscription closed: {e}");
                        stop = true;
                        let _ = vlm_tx.take();
                        continue;
                    }
                };
                let frame: RgbFrame = match sample_json(&sample) {
                    Ok(f) => f,
                    Err(e) => {
                        warn!("bad camera payload: {e}");
                        continue;
                    }
                };
                frames = frames.saturating_add(1);
                let spatial = pipeline.process(&frame)?;
                safety.observe_spatial(&spatial, now_ns());
                publish(&spatial_pub, &spatial).await?;
                report.spatial_states = report.spatial_states.saturating_add(1);
                let ts = now_ns();

                let direct_estop = spatial.human_intrusion || spatial.obstacle_intrusion;
                let escalate = spatial.escalation_score() >= config.vlm.escalation_threshold;
                if direct_estop {
                    let proposal = system1_halt_proposal(
                        spatial.seq,
                        "System 1 critical clearance / intrusion",
                        spatial.timestamp_ns,
                    );
                    let decided = safety.evaluate(&proposal, ts);
                    publish(&intent_pub, &proposal).await?;
                    publish(&cmd_pub, &decided.command).await?;
                    record(&mut recorder, &audit_pub, &frame, &spatial, Some(&proposal), &decided, ts).await?;
                    tally_decision(&mut report, &decided, config.log_verdicts);
                    observe_rss(&mut combined_peak_bytes, &mut llama_hwm_bytes);
                    if decided.command.vetoed {
                        let latency = ts.saturating_sub(spatial.timestamp_ns) as f32 / 1_000_000.0;
                        report.estop_latency_ms = Some(latency);
                    }
                } else if escalate {
                    let fallback_spatial = spatial.clone();
                    let request = VlmRequest {
                        seq: frame.meta.seq,
                        timestamp_ns: frame.meta.monotonic_ns,
                        frame_hash: frame.meta.frame_hash,
                        spatial,
                        image_rgb: Some(frame.pixels.clone()),
                        image_width: frame.meta.width,
                        image_height: frame.meta.height,
                    };
                    let accepted = match vlm_tx.as_ref() {
                        Some(tx) => tx.send(request).await.is_ok(),
                        None => false,
                    };
                    if accepted {
                        pending_vlm = pending_vlm.saturating_add(1);
                    } else {
                        warn!("VLM queue closed; applying safe fallback");
                        let proposal =
                            aegis_vlm::safe_fallback(&fallback_spatial, frame.meta.seq);
                        let fallback_ns = now_ns();
                        let decided = safety.evaluate(&proposal, fallback_ns);
                        publish(&cmd_pub, &decided.command).await?;
                        record(
                            &mut recorder,
                            &audit_pub,
                            &frame,
                            &fallback_spatial,
                            Some(&proposal),
                            &decided,
                            fallback_ns,
                        )
                        .await?;
                    }
                } else {
                    let proposal = VlmProposal::synthetic_continue(frame.meta.seq, frame.meta.monotonic_ns);
                    let decided = safety.evaluate(&proposal, ts);
                    publish(&cmd_pub, &decided.command).await?;
                    record(&mut recorder, &audit_pub, &frame, &spatial, None, &decided, ts).await?;
                    report.bypassed = report.bypassed.saturating_add(1);
                    tally_decision(&mut report, &decided, config.log_verdicts);
                    observe_rss(&mut combined_peak_bytes, &mut llama_hwm_bytes);
                }

                if generator && frames >= total {
                    stop = true;
                    let _ = vlm_tx.take();
                }
                if config.dashboard_every > 0 && frames.is_multiple_of(config.dashboard_every) {
                    print_dashboard(frames, &pipeline, &stats, &safety);
                }
            }
            Some(result) = result_rx.recv() => {
                pending_vlm = pending_vlm.saturating_sub(1);
                report.vlm_results = report.vlm_results.saturating_add(1);
                if result.kind == OutcomeKind::Bypassed {
                    report.bypassed = report.bypassed.saturating_add(1);
                }
                let decision_ns = now_ns();
                let decided = safety.evaluate(&result.proposal, decision_ns);
                publish(&intent_pub, &result.proposal).await?;
                publish(&cmd_pub, &decided.command).await?;
                let frame = synthetic_frame_for(&result.proposal, &result.spatial);
                record(
                    &mut recorder,
                    &audit_pub,
                    &frame,
                    &result.spatial,
                    Some(&result.proposal),
                    &decided,
                    decision_ns,
                )
                .await?;
                tally_decision(&mut report, &decided, config.log_verdicts);
                observe_rss(&mut combined_peak_bytes, &mut llama_hwm_bytes);
            }
        }
    }

    worker_handle.abort();
    if let Some(handle) = generator_handle {
        handle.abort();
    }
    let summary = pipeline.stats().summary();
    let counters = stats.snapshot();
    report.frames = frames;
    report.vlm_accepted = counters.accepted;
    report.vlm_fallbacks = counters.fallbacks;
    report.drops = pipeline.drops();
    report.p50_ms = summary.p50_ms;
    report.p95_ms = summary.p95_ms;
    report.p99_ms = summary.p99_ms;
    let final_sample = rss::snapshot();
    let rust_peak = peak_rss_bytes()
        .unwrap_or(0)
        .max(final_sample.rust_hwm_bytes)
        .max(final_sample.rust_rss_bytes);
    llama_hwm_bytes = llama_hwm_bytes.max(final_sample.llama_hwm_bytes);
    combined_peak_bytes = combined_peak_bytes.max(rust_peak.saturating_add(llama_hwm_bytes));
    report.peak_rss_mib = rss::mib(rust_peak);
    report.rust_process_peak_rss_mib = report.peak_rss_mib;
    report.llama_server_peak_rss_mib = rss::mib(llama_hwm_bytes);
    report.combined_peak_rss_mib = rss::mib(combined_peak_bytes);
    report.vlm_bypass_rate = if frames == 0 { 0.0 } else { report.bypassed as f64 / frames as f64 };
    report.audit_entries = recorder.len();
    let verification: VerifyReport = verify_ledger(&config.database)?;
    report.audit_valid = verification.valid;
    Ok(report)
}

/// Rebuild the frame metadata that corresponds to a VLM decision. The sensor
/// hash is taken from the spatial snapshot carried with the result, so no
/// sensor data is invented.
fn synthetic_frame_for(proposal: &VlmProposal, spatial: &SpatialState) -> RgbFrame {
    let meta = FrameMeta {
        seq: proposal.seq,
        monotonic_ns: spatial.timestamp_ns,
        wall_clock_ms: 0,
        sensor_id: "audit-bridge".to_owned(),
        width: 1,
        height: 1,
        frame_hash: spatial.frame_hash,
        scene_tag: "vlm-intent".to_owned(),
    };
    RgbFrame { meta, pixels: vec![0, 0, 0] }
}

#[allow(clippy::too_many_arguments)]
async fn record(
    recorder: &mut Recorder,
    publisher: &Publisher<'_>,
    frame: &RgbFrame,
    spatial: &SpatialState,
    proposal: Option<&VlmProposal>,
    decided: &aegis_safety::DecidedCommand,
    ts: u64,
) -> Result<AuditEvent, SimError> {
    let record = AuditRecord {
        sensor_hash: frame.meta.frame_hash,
        spatial_state: spatial.clone(),
        vlm_proposal: proposal.cloned(),
        safety_decision: decided.decision.clone(),
        actuator_cmd: decided.command.clone(),
    };
    let event = recorder.append(&record, ts)?;
    publish(publisher, &event).await?;
    Ok(event)
}

fn tally_decision(report: &mut RuntimeReport, decided: &DecidedCommand, log: bool) {
    match decided.decision.disposition {
        Disposition::Approved => report.approved = report.approved.saturating_add(1),
        Disposition::Clamped => report.clamps = report.clamps.saturating_add(1),
        Disposition::Vetoed => report.vetoes = report.vetoes.saturating_add(1),
    }
    if log {
        let rules = decided.decision.rules.iter().map(|r| r.as_str()).collect::<Vec<_>>().join(",");
        let verdict = match decided.decision.disposition {
            Disposition::Approved => "APPROVED",
            Disposition::Clamped => "CLAMPED",
            Disposition::Vetoed => "VETOED",
        };
        println!(
            "[aegis] verdict seq={} {verdict} rules=[{rules}] cmd={:?}",
            decided.command.seq, decided.command.action
        );
    }
}

fn observe_rss(combined_peak_bytes: &mut u64, llama_hwm_bytes: &mut u64) {
    let sample = rss::snapshot();
    *combined_peak_bytes =
        (*combined_peak_bytes).max(sample.rust_rss_bytes.saturating_add(sample.llama_rss_bytes));
    *llama_hwm_bytes = (*llama_hwm_bytes).max(sample.llama_hwm_bytes);
}

fn print_dashboard(
    frames: u64,
    pipeline: &SpatialPipeline,
    vlm: &SharedVlmStats,
    safety: &SafetyEngine,
) {
    let summary = pipeline.stats().summary();
    let counters = vlm.snapshot();
    let sample = rss::snapshot();
    let rss = rss::mib(sample.rust_hwm_bytes);
    let llama_rss = rss::mib(sample.llama_rss_bytes);
    let bypass = if frames == 0 {
        0.0
    } else {
        frames.saturating_sub(counters.requests) as f64 / frames as f64 * 100.0
    };
    println!(
        "[aegis] frames={frames:5} s1 p50/p95/p99={:.2}/{:.2}/{:.2}ms vlm_bypass={bypass:5.1}% vetoes={} clamps={} rss={rss:6.1}MiB llama={llama_rss:6.1}MiB mode={}",
        summary.p50_ms,
        summary.p95_ms,
        summary.p99_ms,
        safety.vetoes(),
        safety.clamps(),
        safety.mode(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_sequence_covers_all_four_scenarios() {
        let seq = scenarios::default_sequence(90);
        assert!(seq.contains(&Scenario::Nominal));
        assert!(seq.contains(&Scenario::DefectivePart));
        assert!(seq.contains(&Scenario::VlmHallucination));
        assert!(seq.contains(&Scenario::HumanIntrusion));
        assert_eq!(seq.last(), Some(&Scenario::HumanIntrusion));
    }

    #[test]
    fn generator_frames_are_unique_and_scene_tagged() {
        let mut sim = WorkcellSimulator::new(64, 64);
        let a = sim.next_frame(1, Scenario::Nominal);
        let b = sim.next_frame(2, Scenario::Nominal);
        assert_ne!(a.meta.frame_hash, b.meta.frame_hash);
        assert_eq!(a.meta.scene_tag, "nominal");
        assert_eq!(sim.next_frame(3, Scenario::HumanIntrusion).meta.scene_tag, "human_intrusion");
    }
}

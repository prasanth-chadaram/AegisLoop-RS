//! Cross-crate integration tests: perception -> triage -> safety -> ledger.
//! No model weights, network or Zenoh broker required.

use aegis_perception::{SpatialConfig, SpatialPipeline};
use aegis_proto::{
    AuditRecord, Disposition, FrameMeta, RuleId, VlmAction, VlmProposal, VlmResponse, ZoneId,
};
use aegis_recorder::{replay_ledger, verify_ledger, Recorder};
use aegis_safety::{SafetyConfig, SafetyEngine};
use aegis_sim::scenarios::{Scenario, WorkcellSimulator};
use aegis_vlm::{safe_fallback, MockBackend, VlmBackend, VlmConfig, VlmError, VlmRequest};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDb(PathBuf);

impl TempDb {
    fn new() -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "aegis-integration-{}-{}.db",
            std::process::id(),
            n
        )))
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let mut name = self.0.clone().into_os_string();
            name.push(suffix);
            let _ = std::fs::remove_file(PathBuf::from(name));
        }
    }
}

fn pipeline() -> SpatialPipeline {
    SpatialPipeline::synthetic(SpatialConfig::default()).expect("pipeline")
}

#[test]
fn intrusion_chain_end_to_end() {
    let db = TempDb::new();
    let mut sim = WorkcellSimulator::new(64, 64);
    let mut perception = pipeline();
    let mut safety = SafetyEngine::new(SafetyConfig::default());
    let mut recorder = Recorder::open(&db.0).expect("ledger");

    let frame = sim.next_frame(1, Scenario::HumanIntrusion);
    let spatial = perception.process(&frame).expect("spatial");
    assert_eq!(spatial.clearance_for(ZoneId::Critical), spatial.min_clearance_mm());
    safety.observe_spatial(&spatial, spatial.timestamp_ns + 1_000_000);

    let proposal = VlmProposal {
        seq: frame.meta.seq,
        request_id: frame.meta.seq,
        timestamp_ns: frame.meta.monotonic_ns,
        trace_id: "integration".to_owned(),
        model: "mock".to_owned(),
        latency_ms: 1.0,
        response: VlmResponse {
            action: VlmAction::Continue,
            confidence: 0.99,
            rationale: "hallucinated continue".to_owned(),
        },
    };
    let decided = safety.evaluate(&proposal, spatial.timestamp_ns + 1_000_000);
    assert_eq!(decided.decision.disposition, Disposition::Vetoed);
    assert!(decided.decision.rules.contains(&RuleId::SafProximity));

    let decision_ns = spatial.timestamp_ns + 1_000_000;
    let event = recorder
        .append(
            &AuditRecord {
                sensor_hash: frame.meta.frame_hash,
                spatial_state: spatial,
                vlm_proposal: Some(proposal),
                safety_decision: decided.decision,
                actuator_cmd: decided.command,
            },
            decision_ns,
        )
        .expect("append");
    assert_eq!(event.index, 1);
    assert!(verify_ledger(&db.0).expect("verify").valid);
    let replay = replay_ledger(&db.0, SafetyConfig::default()).expect("replay");
    assert!(replay.deterministic);
    assert_eq!(replay.vetoed, 1);
}

#[test]
fn timeout_fallback_is_gated_before_actuation() {
    let mut sim = WorkcellSimulator::new(64, 64);
    let mut perception = pipeline();
    let frame = sim.next_frame(7, Scenario::DefectivePart);
    let spatial = perception.process(&frame).expect("spatial");
    let request = VlmRequest {
        seq: frame.meta.seq,
        timestamp_ns: frame.meta.monotonic_ns,
        frame_hash: frame.meta.frame_hash,
        spatial: spatial.clone(),
    };
    let backend = MockBackend::failing(VlmError::Timeout(1500));
    future::block_on(async move {
        let response = backend.propose(&request).await;
        assert!(response.is_err());
    });
    let fallback = safe_fallback(&spatial, frame.meta.seq);
    assert_eq!(fallback.response.action, VlmAction::SlowConveyor { speed_pct: 20 });
    let mut safety = SafetyEngine::new(SafetyConfig::default());
    safety.observe_spatial(&spatial, spatial.timestamp_ns + 1_000_000);
    let decided = safety.evaluate(&fallback, spatial.timestamp_ns + 1_000_000);
    assert_eq!(decided.decision.disposition, Disposition::Clamped);
    assert_eq!(decided.command.action, VlmAction::SlowConveyor { speed_pct: 6 });
}

#[test]
fn replay_detects_tampered_audit() {
    use rusqlite_for_tests::Connection;
    let db = TempDb::new();
    let mut recorder = Recorder::open(&db.0).expect("ledger");
    let mut sim = WorkcellSimulator::new(64, 64);
    let frame = sim.next_frame(1, Scenario::Nominal);
    let mut perception = pipeline();
    let spatial = perception.process(&frame).expect("spatial");
    let proposal = VlmProposal::synthetic_continue(frame.meta.seq, frame.meta.monotonic_ns);
    let mut safety = SafetyEngine::new(SafetyConfig::default());
    safety.observe_spatial(&spatial, spatial.timestamp_ns + 1_000_000);
    let decided = safety.evaluate(&proposal, spatial.timestamp_ns + 1_000_000);
    let decision_ns = spatial.timestamp_ns + 1_000_000;
    recorder
        .append(
            &AuditRecord {
                sensor_hash: frame.meta.frame_hash,
                spatial_state: spatial,
                vlm_proposal: Some(proposal),
                safety_decision: decided.decision,
                actuator_cmd: decided.command,
            },
            decision_ns,
        )
        .expect("append");
    let conn = Connection::open(&db.0).expect("reopen");
    conn.execute(
        "UPDATE events SET sensor_hash = ?1 WHERE idx = 1",
        rusqlite_for_tests::params!["00".repeat(32)],
    )
    .expect("tamper");
    drop(conn);
    let report = verify_ledger(&db.0).expect("verify");
    assert!(!report.valid);
    assert_eq!(report.first_bad_index, Some(1));
}

mod future {
    pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(f)
    }
}

// Integration tests are allowed to name rusqlite directly for deliberate
// database tampering; production code never does this.
mod rusqlite_for_tests {
    pub use rusqlite::{params, Connection};
}

#[test]
fn strict_vlm_schema_rejects_unknown_fields() {
    let json = r#"{"action":{"type":"continue"},"confidence":0.9,"extra":true}"#;
    let parsed: Result<VlmResponse, _> = serde_json::from_str(json);
    assert!(parsed.is_err());
    let _ = VlmConfig::default();
    let _ = FrameMeta::new(1, 1, "cam", 1, 1, &[0, 0, 0], "nominal");
}

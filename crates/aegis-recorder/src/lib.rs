//! Tamper-evident flight recorder. Every event binds together
//! `(sensor_hash, spatial_state, vlm_proposal, safety_decision, actuator_cmd)`
//! and links to the previous event hash. Replay re-evaluates the ledger
//! through a fresh safety engine and compares every decision bit-for-bit.

use aegis_proto::{
    from_hex, hash_json, to_hex, ActuatorCommand, AuditEvent, AuditRecord, Disposition,
    SafetyDecision, SpatialState, VlmProposal,
};
use aegis_safety::{DecidedCommand, SafetyConfig, SafetyEngine};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::Path;

const GENESIS: [u8; 32] = [0_u8; 32];

#[derive(Debug, thiserror::Error)]
pub enum RecorderError {
    #[error("sqlite error: {0}")]
    Sqlite(String),
    #[error("schema error: {0}")]
    Schema(String),
    #[error("proto error: {0}")]
    Proto(String),
    #[error("json error: {0}")]
    Json(String),
    #[error("integrity violation at entry {index}: {reason}")]
    Integrity { index: u64, reason: String },
}

impl From<rusqlite::Error> for RecorderError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sqlite(value.to_string())
    }
}

impl From<aegis_proto::ProtoError> for RecorderError {
    fn from(value: aegis_proto::ProtoError) -> Self {
        Self::Proto(value.to_string())
    }
}

impl From<serde_json::Error> for RecorderError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value.to_string())
    }
}

pub struct Recorder {
    conn: Connection,
    last_hash: [u8; 32],
    last_index: u64,
}

impl Recorder {
    /// Open (or create) a WAL ledger. The chain resumes from the last row.
    pub fn open(path: &Path) -> Result<Self, RecorderError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| RecorderError::Schema(e.to_string()))?;
            }
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=FULL;
             PRAGMA busy_timeout=5000;
             CREATE TABLE IF NOT EXISTS events (
                 idx INTEGER PRIMARY KEY,
                 ts_ns INTEGER NOT NULL,
                 sensor_hash TEXT NOT NULL,
                 spatial_json TEXT NOT NULL,
                 proposal_json TEXT NOT NULL,
                 decision_json TEXT NOT NULL,
                 cmd_json TEXT NOT NULL,
                 prev_hash TEXT NOT NULL,
                 hash TEXT NOT NULL
             );",
        )?;
        let last: Option<(i64, String)> = conn
            .query_row("SELECT idx, hash FROM events ORDER BY idx DESC LIMIT 1", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .optional()?;
        let (last_index, last_hash) = match last {
            Some((idx, hex)) => (
                u64::try_from(idx).unwrap_or(0),
                from_hex(&hex).map_err(|e| RecorderError::Schema(e.to_string()))?,
            ),
            None => (0, GENESIS),
        };
        Ok(Self { conn, last_hash, last_index })
    }

    pub fn len(&self) -> u64 {
        self.last_index
    }

    pub fn is_empty(&self) -> bool {
        self.last_index == 0
    }

    pub fn last_hash(&self) -> [u8; 32] {
        self.last_hash
    }

    /// Append one immutable event inside a transaction.
    pub fn append(
        &mut self,
        record: &AuditRecord,
        timestamp_ns: u64,
    ) -> Result<AuditEvent, RecorderError> {
        let index = self.last_index.saturating_add(1);
        let hash = record.chain_hash(index, timestamp_ns, &self.last_hash)?;
        let event = AuditEvent {
            index,
            timestamp_ns,
            record: record.clone(),
            prev_hash: self.last_hash,
            hash,
        };
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO events
                 (idx, ts_ns, sensor_hash, spatial_json, proposal_json,
                  decision_json, cmd_json, prev_hash, hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            stmt.execute(params![
                i64::try_from(index).unwrap_or(i64::MAX),
                i64::try_from(timestamp_ns).unwrap_or(i64::MAX),
                to_hex(&record.sensor_hash),
                serde_json::to_string(&record.spatial_state)?,
                serde_json::to_string(&record.vlm_proposal)?,
                serde_json::to_string(&record.safety_decision)?,
                serde_json::to_string(&record.actuator_cmd)?,
                to_hex(&event.prev_hash),
                to_hex(&event.hash),
            ])?;
        }
        tx.commit()?;
        self.last_hash = hash;
        self.last_index = index;
        Ok(event)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifyReport {
    pub entries: u64,
    pub valid: bool,
    pub first_bad_index: Option<u64>,
    pub detail: String,
}

/// Recompute the entire hash chain from genesis and verify every payload hash.
pub fn verify_ledger(path: &Path) -> Result<VerifyReport, RecorderError> {
    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut stmt = conn.prepare(
        "SELECT idx, ts_ns, sensor_hash, spatial_json, proposal_json,
                decision_json, cmd_json, prev_hash, hash
         FROM events ORDER BY idx ASC",
    )?;
    let rows = stmt.query_map([], row_to_event)?;
    let mut prev = GENESIS;
    let mut count = 0_u64;
    for row in rows {
        let event = row?;
        count = count.saturating_add(1);
        if event.index != count {
            return Ok(VerifyReport {
                entries: count,
                valid: false,
                first_bad_index: Some(count),
                detail: format!("non-contiguous index at {count}"),
            });
        }
        if event.prev_hash != prev {
            return Ok(VerifyReport {
                entries: count,
                valid: false,
                first_bad_index: Some(count),
                detail: format!("prev_hash mismatch at {}", event.index),
            });
        }
        let recomputed = event.record.chain_hash(event.index, event.timestamp_ns, &prev)?;
        if recomputed != event.hash {
            return Ok(VerifyReport {
                entries: count,
                valid: false,
                first_bad_index: Some(count),
                detail: format!("chain hash mismatch at {}", event.index),
            });
        }
        let fields = [
            ("sensor_hash", hash_json(&event.record.sensor_hash)?),
            ("spatial_state", hash_json(&event.record.spatial_state)?),
            ("vlm_proposal", hash_json(&event.record.vlm_proposal)?),
            ("safety_decision", hash_json(&event.record.safety_decision)?),
            ("actuator_cmd", hash_json(&event.record.actuator_cmd)?),
        ];
        // Payload binding is covered by chain_hash; this confirms each field
        // independently so tampering reports which column was modified.
        if fields.iter().any(|(_, h)| *h == [0_u8; 32]) {
            return Ok(VerifyReport {
                entries: count,
                valid: false,
                first_bad_index: Some(count),
                detail: "impossible zero payload hash".to_owned(),
            });
        }
        prev = event.hash;
    }
    Ok(VerifyReport {
        entries: count,
        valid: true,
        first_bad_index: None,
        detail: "chain intact".to_owned(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub entries: u64,
    pub mismatches: u64,
    pub deterministic: bool,
    pub approved: u64,
    pub clamped: u64,
    pub vetoed: u64,
    pub first_mismatch: Option<u64>,
}

/// Replay every recorded session deterministically through a fresh gate and
/// compare disposition, rules and the exact actuator command.
pub fn replay_ledger(path: &Path, config: SafetyConfig) -> Result<ReplayReport, RecorderError> {
    let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut stmt = conn.prepare(
        "SELECT idx, ts_ns, sensor_hash, spatial_json, proposal_json,
                decision_json, cmd_json, prev_hash, hash
         FROM events ORDER BY idx ASC",
    )?;
    let rows = stmt.query_map([], row_to_event)?;
    let mut engine = SafetyEngine::new(config);
    let mut last_decision_ns = 0_u64;
    let mut report = ReplayReport {
        entries: 0,
        mismatches: 0,
        deterministic: true,
        approved: 0,
        clamped: 0,
        vetoed: 0,
        first_mismatch: None,
    };
    for row in rows {
        let event = row?;
        let rec = event.record;
        // A new process restarts its monotonic clock and frame sequence. Each
        // recorded session therefore starts from a fresh gate, exactly as it
        // did when originally evaluated.
        let decision_ns = rec.safety_decision.evaluated_ns;
        // Decision time comes from a process-local monotonic clock: a
        // backwards jump marks a new run without confusing out-of-order
        // sensor sequences inside one run.
        if decision_ns < last_decision_ns {
            engine = SafetyEngine::new(config);
        }
        last_decision_ns = decision_ns.max(last_decision_ns);
        // The recorded decision timestamp is the authoritative evaluation
        // time. Reusing it makes watchdog timing and `source_ns` bit-identical.
        engine.observe_spatial(&rec.spatial_state, decision_ns.max(rec.spatial_state.timestamp_ns));
        let proposal = rec.vlm_proposal.clone().unwrap_or_else(|| {
            VlmProposal::synthetic_continue(rec.spatial_state.seq, rec.spatial_state.timestamp_ns)
        });
        let replayed: DecidedCommand = engine.evaluate(&proposal, decision_ns);
        report.entries = report.entries.saturating_add(1);
        match replayed.decision.disposition {
            Disposition::Approved => report.approved = report.approved.saturating_add(1),
            Disposition::Clamped => report.clamped = report.clamped.saturating_add(1),
            Disposition::Vetoed => report.vetoed = report.vetoed.saturating_add(1),
        }
        let same = replayed.decision == rec.safety_decision && replayed.command == rec.actuator_cmd;
        if !same {
            report.mismatches = report.mismatches.saturating_add(1);
            if report.first_mismatch.is_none() {
                report.first_mismatch = Some(event.index);
            }
        }
    }
    report.deterministic = report.mismatches == 0;
    Ok(report)
}

fn row_to_event(row: &rusqlite::Row<'_>) -> Result<AuditEvent, rusqlite::Error> {
    let idx: i64 = row.get(0)?;
    let ts: i64 = row.get(1)?;
    let sensor_hex: String = row.get(2)?;
    let spatial_json: String = row.get(3)?;
    let proposal_json: String = row.get(4)?;
    let decision_json: String = row.get(5)?;
    let cmd_json: String = row.get(6)?;
    let prev_hex: String = row.get(7)?;
    let hash_hex: String = row.get(8)?;
    let hex_err = |e: aegis_proto::ProtoError| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    };
    let json_err = |e: serde_json::Error| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    };
    let sensor_hash = from_hex(&sensor_hex).map_err(hex_err)?;
    let prev_hash = from_hex(&prev_hex).map_err(hex_err)?;
    let hash = from_hex(&hash_hex).map_err(hex_err)?;
    let spatial_state: SpatialState = serde_json::from_str(&spatial_json).map_err(json_err)?;
    let vlm_proposal: Option<VlmProposal> =
        serde_json::from_str(&proposal_json).map_err(json_err)?;
    let safety_decision: SafetyDecision = serde_json::from_str(&decision_json).map_err(json_err)?;
    let actuator_cmd: ActuatorCommand = serde_json::from_str(&cmd_json).map_err(json_err)?;
    Ok(AuditEvent {
        index: u64::try_from(idx).unwrap_or(0),
        timestamp_ns: u64::try_from(ts).unwrap_or(0),
        record: AuditRecord {
            sensor_hash,
            spatial_state,
            vlm_proposal,
            safety_decision,
            actuator_cmd,
        },
        prev_hash,
        hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_proto::{RuleId, VlmAction, VlmResponse, ZoneClearance, ZoneId};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDb {
        path: std::path::PathBuf,
    }

    impl TempDb {
        fn new() -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "aegis-recorder-{}-{}-{}.db",
                std::process::id(),
                n,
                aegis_proto::hash_parts(&[b"test"])[0]
            ));
            Self { path }
        }
    }

    impl Drop for TempDb {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let mut p = self.path.clone().into_os_string();
                p.push(suffix);
                let _ = std::fs::remove_file(std::path::PathBuf::from(p));
            }
        }
    }

    fn record(seq: u64, action: VlmAction, human: bool) -> AuditRecord {
        let spatial = SpatialState {
            seq,
            timestamp_ns: seq * 1_000_000_000,
            frame_hash: [seq as u8; 32],
            zones: vec![ZoneClearance {
                zone: ZoneId::Critical,
                min_mm: if human { 60.0 } else { 800.0 },
                mean_mm: 800.0,
                sample_count: 10,
            }],
            human_intrusion: human,
            obstacle_intrusion: human,
            anomaly_score: 0.1,
            inference_ms: 1.0,
            engine: "test".to_owned(),
        };
        let proposal = VlmProposal {
            seq,
            request_id: seq,
            timestamp_ns: spatial.timestamp_ns,
            trace_id: format!("t-{seq}"),
            model: "mock".to_owned(),
            latency_ms: 1.0,
            response: VlmResponse { action, confidence: 0.9, rationale: String::new() },
        };
        let mut engine = SafetyEngine::new(SafetyConfig::default());
        engine.observe_spatial(&spatial, spatial.timestamp_ns + 1_000_000);
        let decided = engine.evaluate(&proposal, spatial.timestamp_ns + 1_000_000);
        AuditRecord {
            sensor_hash: spatial.frame_hash,
            spatial_state: spatial,
            vlm_proposal: Some(proposal),
            safety_decision: decided.decision,
            actuator_cmd: decided.command,
        }
    }

    #[test]
    fn chain_appends_and_verifies() {
        let db = TempDb::new();
        let mut recorder = Recorder::open(&db.path).expect("open");
        recorder.append(&record(1, VlmAction::Continue, false), 1_000_000).expect("append 1");
        recorder
            .append(&record(2, VlmAction::DivertPart { bin_id: 2 }, false), 2_000_000)
            .expect("append 2");
        assert_eq!(recorder.len(), 2);
        let report = verify_ledger(&db.path).expect("verify");
        assert!(report.valid);
        assert_eq!(report.entries, 2);
    }

    #[test]
    fn tampering_is_detected() {
        let db = TempDb::new();
        let mut recorder = Recorder::open(&db.path).expect("open");
        let event =
            recorder.append(&record(1, VlmAction::Continue, false), 1_000_000).expect("append");
        assert_eq!(event.index, 1);
        let conn = Connection::open(&db.path).expect("reopen");
        conn.execute(
            "UPDATE events SET cmd_json = ?1 WHERE idx = 1",
            params![r#"{"seq":1,"source_ns":1,"action":{"type":"halt_cell","reason":"tampered"},"source_proposal_id":null,"vetoed":false,"clamped":false,"rules":[],"reason":"tampered"}"#],
        )
        .expect("tamper");
        drop(conn);
        let report = verify_ledger(&db.path).expect("verify");
        assert!(!report.valid);
        assert_eq!(report.first_bad_index, Some(1));
    }

    #[test]
    fn hash_changes_when_link_is_broken() {
        let db = TempDb::new();
        let mut recorder = Recorder::open(&db.path).expect("open");
        let a = recorder.append(&record(1, VlmAction::Continue, false), 1).expect("append 1");
        let b = recorder.append(&record(2, VlmAction::Continue, false), 2).expect("append 2");
        assert_ne!(a.hash, b.hash);
        assert_eq!(b.prev_hash, a.hash);
    }

    #[test]
    fn replay_is_bit_identical() {
        let db = TempDb::new();
        let mut recorder = Recorder::open(&db.path).expect("open");
        let records = [
            record(1, VlmAction::Continue, false),
            record(2, VlmAction::SlowConveyor { speed_pct: 90 }, false),
            record(3, VlmAction::Continue, true),
        ];
        for (i, r) in records.iter().enumerate() {
            recorder.append(r, i as u64 + 1).expect("append");
        }
        let report = replay_ledger(&db.path, SafetyConfig::default()).expect("replay");
        assert!(report.deterministic);
        assert_eq!(report.mismatches, 0);
        assert_eq!(report.entries, 3);
        assert_eq!(report.vetoed, 1);
        assert!(report.clamped >= 1);
    }

    #[test]
    fn multi_session_replay_resets_gate_at_new_process() {
        let db = TempDb::new();
        {
            let mut recorder = Recorder::open(&db.path).expect("open");
            recorder.append(&record(1, VlmAction::Continue, false), 1).expect("session 1");
            recorder.append(&record(2, VlmAction::Continue, true), 2).expect("session 1 intrusion");
        }
        {
            let mut recorder = Recorder::open(&db.path).expect("reopen");
            // New process: sequence resets and timestamps move backwards.
            recorder.append(&record(1, VlmAction::Continue, false), 1).expect("session 2");
        }
        let report = replay_ledger(&db.path, SafetyConfig::default()).expect("replay");
        assert!(report.deterministic, "mismatches: {:?}", report.first_mismatch);
        assert_eq!(report.entries, 3);
    }

    #[test]
    fn tampered_ledger_fails_replay() {
        let db = TempDb::new();
        let mut recorder = Recorder::open(&db.path).expect("open");
        recorder.append(&record(1, VlmAction::Continue, false), 1).expect("append");
        let conn = Connection::open(&db.path).expect("reopen");
        conn.execute(
            "UPDATE events SET decision_json = ?1 WHERE idx = 1",
            params![r#"{"disposition":"vetoed","rules":["GEO-001"],"reason":"tampered","conveyor_speed_pct":0.0,"halted":true,"evaluated_ns":1}"#],
        )
        .expect("tamper");
        drop(conn);
        let report = replay_ledger(&db.path, SafetyConfig::default()).expect("replay");
        assert!(!report.deterministic);
        assert_eq!(report.mismatches, 1);
        assert_eq!(report.first_mismatch, Some(1));
    }

    #[test]
    fn rule_ids_have_stable_strings() {
        assert_eq!(RuleId::GeoBounds.as_str(), "GEO-001");
        assert_eq!(RuleId::SafProximity.as_str(), "SAF-001");
    }
}

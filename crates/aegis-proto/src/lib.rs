//! Strongly typed schemas for every Zenoh topic in the AegisLoop-RS workcell.
//!
//! Every payload is `serde`-serializable (JSON on the wire, bincode for local
//! zero-copy-ish caching) and every audit record carries a deterministic
//! SHA-256 chain hash. No runtime path in this crate panics.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const TOPIC_CAM_RGB: &str = "workcell/cam/rgb";
pub const TOPIC_SPATIAL: &str = "workcell/perception/spatial";
pub const TOPIC_VLM_INTENT: &str = "workcell/vlm/intent";
pub const TOPIC_ACTUATOR_CMD: &str = "workcell/actuator/cmd";
pub const TOPIC_AUDIT_EVENT: &str = "workcell/audit/event";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProtoError {
    #[error("json serialization failed: {0}")]
    Json(String),
    #[error("invalid hash hex string: {0}")]
    Hex(String),
}

// ---------------------------------------------------------------------------
// workcell/cam/rgb
// ---------------------------------------------------------------------------

/// Metadata for one RGB frame. Pixel payloads stay out of the ledger; only the
/// hash is retained, which is enough to bind the audit chain to sensor data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameMeta {
    pub seq: u64,
    /// Monotonic clock, nanoseconds (CLOCK_MONOTONIC or an equivalent source).
    pub monotonic_ns: u64,
    /// Wall clock, Unix milliseconds (observability only, never used for safety).
    pub wall_clock_ms: i64,
    pub sensor_id: String,
    pub width: u32,
    pub height: u32,
    pub frame_hash: [u8; 32],
    /// Scenario hint consumed by the deterministic synthetic estimator.
    /// Real perception backends ignore this field.
    pub scene_tag: String,
}

impl FrameMeta {
    pub fn hash_pixels(pixels: &[u8]) -> [u8; 32] {
        hash_parts(&[b"aegis/frame/v1", pixels])
    }

    pub fn new(
        seq: u64,
        monotonic_ns: u64,
        sensor_id: &str,
        width: u32,
        height: u32,
        pixels: &[u8],
        scene_tag: &str,
    ) -> Self {
        Self {
            seq,
            monotonic_ns,
            wall_clock_ms: wall_clock_unix_ms(),
            sensor_id: sensor_id.to_owned(),
            width,
            height,
            frame_hash: Self::hash_pixels(pixels),
            scene_tag: scene_tag.to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RgbFrame {
    pub meta: FrameMeta,
    pub pixels: Vec<u8>,
}

fn wall_clock_unix_ms() -> i64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => i64::try_from(d.as_millis()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// workcell/perception/spatial
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ZoneId {
    Near,
    Warning,
    Critical,
}

impl fmt::Display for ZoneId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Near => write!(f, "near"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZoneClearance {
    pub zone: ZoneId,
    /// Millimetres. `f32` is deterministic here: all producers use the same
    /// fixed operation order and the same input quantization.
    pub min_mm: f32,
    pub mean_mm: f32,
    pub sample_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EscalationLevel {
    Normal,
    Elevated,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialState {
    pub seq: u64,
    pub timestamp_ns: u64,
    pub frame_hash: [u8; 32],
    pub zones: Vec<ZoneClearance>,
    pub human_intrusion: bool,
    pub obstacle_intrusion: bool,
    pub anomaly_score: f32,
    pub inference_ms: f32,
    pub engine: String,
}

impl SpatialState {
    pub fn min_clearance_mm(&self) -> f32 {
        let mut min = f32::MAX;
        for zone in &self.zones {
            if zone.min_mm < min {
                min = zone.min_mm;
            }
        }
        if min == f32::MAX {
            0.0
        } else {
            min
        }
    }

    pub fn clearance_for(&self, zone: ZoneId) -> f32 {
        self.zones.iter().find(|z| z.zone == zone).map_or(0.0, |z| z.min_mm)
    }

    /// Deterministic triage score in `[0, 1]`. System 2 is invoked only when
    /// this value crosses the configured escalation threshold.
    pub fn escalation_score(&self) -> f32 {
        let mut score = self.anomaly_score.clamp(0.0, 1.0);
        if self.obstacle_intrusion {
            score += 0.25;
        }
        if self.human_intrusion {
            score += 0.50;
        }
        if self.min_clearance_mm() < 300.0 {
            score += 0.10;
        }
        score.clamp(0.0, 1.0)
    }

    pub fn level(&self) -> EscalationLevel {
        if self.human_intrusion || self.obstacle_intrusion {
            EscalationLevel::Critical
        } else if self.escalation_score() >= 0.45 {
            EscalationLevel::Elevated
        } else {
            EscalationLevel::Normal
        }
    }
}

// ---------------------------------------------------------------------------
// workcell/vlm/intent
// ---------------------------------------------------------------------------

/// The complete action vocabulary System 2 is allowed to propose. Anything
/// outside this enum is rejected during JSON deserialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VlmAction {
    Continue,
    SlowConveyor { speed_pct: u8 },
    DivertPart { bin_id: u8 },
    RegraspOffset { dx_mm: f32, dy_mm: f32, dz_mm: f32 },
    HaltCell { reason: String },
}

impl VlmAction {
    /// Schema-level validation. Safety bounds are checked separately by
    /// `aegis-safety`; this only rejects structurally nonsensical proposals.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Continue => Ok(()),
            Self::SlowConveyor { speed_pct } if *speed_pct <= 100 => Ok(()),
            Self::SlowConveyor { .. } => Err("speed_pct outside 0..=100".to_owned()),
            Self::DivertPart { bin_id } if (1..=8).contains(bin_id) => Ok(()),
            Self::DivertPart { .. } => Err("bin_id outside 1..=8".to_owned()),
            Self::RegraspOffset { dx_mm, dy_mm, dz_mm }
                if dx_mm.is_finite() && dy_mm.is_finite() && dz_mm.is_finite() =>
            {
                Ok(())
            }
            Self::RegraspOffset { .. } => Err("non-finite regrasp offset".to_owned()),
            Self::HaltCell { reason } if !reason.trim().is_empty() => Ok(()),
            Self::HaltCell { .. } => Err("halt reason must not be empty".to_owned()),
        }
    }
}

/// Body returned by the OpenAI-compatible VLM endpoint. Strict schema: no
/// unknown fields, finite confidence, validated action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VlmResponse {
    pub action: VlmAction,
    pub confidence: f32,
    #[serde(default)]
    pub rationale: String,
}

impl VlmResponse {
    pub fn validate(&self) -> Result<(), String> {
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err("confidence outside 0..=1".to_owned());
        }
        self.action.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VlmProposal {
    pub seq: u64,
    pub request_id: u64,
    pub timestamp_ns: u64,
    pub trace_id: String,
    pub model: String,
    pub latency_ms: f32,
    pub response: VlmResponse,
}

impl VlmProposal {
    pub fn synthetic_continue(seq: u64, timestamp_ns: u64) -> Self {
        Self {
            seq,
            request_id: seq,
            timestamp_ns,
            trace_id: format!("bypass-{seq}"),
            model: "system1-fallback".to_owned(),
            latency_ms: 0.0,
            response: VlmResponse {
                action: VlmAction::Continue,
                confidence: 1.0,
                rationale: "System 2 bypassed by spatial triage".to_owned(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// workcell/actuator/cmd
// ---------------------------------------------------------------------------

/// Rule identifiers are stable strings in the audit ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuleId {
    #[serde(rename = "GEO-001")]
    GeoBounds,
    #[serde(rename = "GEO-002")]
    GeoBin,
    #[serde(rename = "SAF-001")]
    SafProximity,
    #[serde(rename = "WD-001")]
    WdStall,
    #[serde(rename = "WD-002")]
    WdLatency,
    #[serde(rename = "SPD-001")]
    SpdRange,
    #[serde(rename = "SPD-002")]
    SpdDelta,
    #[serde(rename = "SPD-003")]
    SpdJerk,
    #[serde(rename = "LATCH-001")]
    LatEstop,
}

impl RuleId {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GeoBounds => "GEO-001",
            Self::GeoBin => "GEO-002",
            Self::SafProximity => "SAF-001",
            Self::WdStall => "WD-001",
            Self::WdLatency => "WD-002",
            Self::SpdRange => "SPD-001",
            Self::SpdDelta => "SPD-002",
            Self::SpdJerk => "SPD-003",
            Self::LatEstop => "LATCH-001",
        }
    }
}

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Disposition {
    Approved,
    Clamped,
    Vetoed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SafetyDecision {
    pub disposition: Disposition,
    pub rules: Vec<RuleId>,
    pub reason: String,
    pub conveyor_speed_pct: f32,
    pub halted: bool,
    pub evaluated_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActuatorCommand {
    pub seq: u64,
    pub source_ns: u64,
    pub action: VlmAction,
    pub source_proposal_id: Option<u64>,
    pub vetoed: bool,
    pub clamped: bool,
    pub rules: Vec<RuleId>,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// workcell/audit/event
// ---------------------------------------------------------------------------

/// One immutable flight-recorder entry. `AuditEvent` adds the chain links.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditRecord {
    pub sensor_hash: [u8; 32],
    pub spatial_state: SpatialState,
    pub vlm_proposal: Option<VlmProposal>,
    pub safety_decision: SafetyDecision,
    pub actuator_cmd: ActuatorCommand,
}

impl AuditRecord {
    /// Canonical, domain-separated chain hash. Field order is fixed here so
    /// replay on any machine reproduces bit-identical hashes.
    pub fn chain_hash(
        &self,
        index: u64,
        timestamp_ns: u64,
        prev_hash: &[u8; 32],
    ) -> Result<[u8; 32], ProtoError> {
        let spatial = hash_json(&self.spatial_state)?;
        let proposal = hash_json(&self.vlm_proposal)?;
        let decision = hash_json(&self.safety_decision)?;
        let cmd = hash_json(&self.actuator_cmd)?;
        Ok(hash_parts(&[
            b"aegis/audit-chain/v1",
            &index.to_be_bytes(),
            &timestamp_ns.to_be_bytes(),
            prev_hash,
            &self.sensor_hash,
            &spatial,
            &proposal,
            &decision,
            &cmd,
        ]))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditEvent {
    pub index: u64,
    pub timestamp_ns: u64,
    pub record: AuditRecord,
    pub prev_hash: [u8; 32],
    pub hash: [u8; 32],
}

// ---------------------------------------------------------------------------
// hashing helpers
// ---------------------------------------------------------------------------

/// Length-prefixed, domain-separated SHA-256. Never panics.
pub fn hash_parts(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut out = [0_u8; 32];
    out.copy_from_slice(&digest);
    out
}

pub fn hash_json<T: Serialize>(value: &T) -> Result<[u8; 32], ProtoError> {
    let bytes = serde_json::to_vec(value).map_err(|e| ProtoError::Json(e.to_string()))?;
    Ok(hash_parts(&[b"aegis/json/v1", &bytes]))
}

pub fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len().saturating_mul(2));
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        s.push(char::from_digit((b & 0x0f) as u32, 16).unwrap_or('0'));
    }
    s
}

pub fn from_hex(s: &str) -> Result<[u8; 32], ProtoError> {
    let bytes = s.as_bytes();
    if bytes.len() != 64 {
        return Err(ProtoError::Hex(format!("expected 64 hex chars, got {}", bytes.len())));
    }
    let mut out = [0_u8; 32];
    let mut i: usize = 0;
    while i < 32 {
        let hi = hex_nibble(bytes[i.saturating_mul(2)])?;
        let lo = hex_nibble(bytes[i.saturating_mul(2).saturating_add(1)])?;
        out[i] = (hi << 4) | lo;
        i += 1;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, ProtoError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(ProtoError::Hex(format!("invalid nibble 0x{b:02x}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_domain_separated_and_stable() {
        assert_eq!(hash_parts(&[b"abc"]), hash_parts(&[b"abc"]));
        assert_ne!(hash_parts(&[b"abc"]), hash_parts(&[b"abd"]));
        let h = hash_parts(&[b"a", b"bc"]);
        assert_ne!(h, hash_parts(&[b"ab", b"c"]));
    }

    #[test]
    fn hex_roundtrip_and_rejects_bad_input() {
        let h = hash_parts(&[b"aegis"]);
        assert_eq!(from_hex(&to_hex(&h)), Ok(h));
        assert!(from_hex("deadbeef").is_err());
        assert!(from_hex(&"z".repeat(64)).is_err());
    }

    #[test]
    fn vlm_action_schema_is_strict() {
        let good = r#"{"action":{"type":"slow_conveyor","speed_pct":35},"confidence":0.9,"rationale":"ok"}"#;
        let parsed: VlmResponse = serde_json::from_str(good).expect("valid schema");
        assert!(parsed.validate().is_ok());
        let unknown = r#"{"action":{"type":"continue"},"confidence":0.9,"surprise":1}"#;
        assert!(serde_json::from_str::<VlmResponse>(unknown).is_err());
        let bad_speed = r#"{"action":{"type":"slow_conveyor","speed_pct":140},"confidence":0.9}"#;
        let parsed: VlmResponse = serde_json::from_str(bad_speed).expect("parse");
        assert!(parsed.validate().is_err());
    }

    #[test]
    fn spatial_escalation_math_is_deterministic() {
        let spatial = SpatialState {
            seq: 1,
            timestamp_ns: 10,
            frame_hash: [0; 32],
            zones: vec![ZoneClearance {
                zone: ZoneId::Critical,
                min_mm: 50.0,
                mean_mm: 60.0,
                sample_count: 10,
            }],
            human_intrusion: true,
            obstacle_intrusion: true,
            anomaly_score: 0.2,
            inference_ms: 1.0,
            engine: "test".to_owned(),
        };
        assert_eq!(spatial.level(), EscalationLevel::Critical);
        assert!((spatial.escalation_score() - 1.0).abs() < 1e-6);
    }
}

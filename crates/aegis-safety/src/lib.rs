//! Deterministic safety contract. This is the only component allowed to emit
//! an actuator command. It is pure with respect to its inputs: given the same
//! config, state, proposal and timestamps it always produces the same decision.
//!
//! Invariants enforced here:
//! * `GEO-001/002` - workspace geofence and kinematic offset bounds.
//! * `SAF-001` - proximity hard E-stop; overrides System 2 semantics.
//! * `WD-001/002` - spatial watchdog and latency SLA.
//! * `SPD-001/002/003` - conveyor range, rate and jerk clamps.
//! * `LATCH-001` - E-stop remains latched until an explicit operator reset.

use aegis_proto::{
    ActuatorCommand, Disposition, RuleId, SafetyDecision, SpatialState, VlmAction, VlmProposal,
};
use serde::{Deserialize, Serialize};
use std::fmt;

const NS_PER_MS: u64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SafetyConfig {
    /// Below this clearance the cell must stop, regardless of VLM intent.
    pub critical_clearance_mm: f32,
    /// Soft regrasp limit. Values inside the hard limit are clamped; values
    /// beyond it are vetoed outright.
    pub max_regrasp_mm: f32,
    pub hard_regrasp_mm: f32,
    pub max_speed_pct: f32,
    pub max_speed_delta_pct: f32,
    pub max_jerk_pct: f32,
    pub watchdog_ms: u64,
    pub latency_sla_ms: u64,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            critical_clearance_mm: 120.0,
            max_regrasp_mm: 25.0,
            hard_regrasp_mm: 60.0,
            max_speed_pct: 100.0,
            max_speed_delta_pct: 12.0,
            max_jerk_pct: 6.0,
            watchdog_ms: 100,
            latency_sla_ms: 100,
        }
    }
}

impl SafetyConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let finite = self.critical_clearance_mm.is_finite()
            && self.max_regrasp_mm.is_finite()
            && self.hard_regrasp_mm.is_finite()
            && self.max_speed_pct.is_finite()
            && self.max_speed_delta_pct.is_finite()
            && self.max_jerk_pct.is_finite();
        if !finite {
            return Err(ConfigError::NonFinite);
        }
        if self.max_regrasp_mm < 0.0 || self.hard_regrasp_mm < self.max_regrasp_mm {
            return Err(ConfigError::InvalidBounds);
        }
        if self.max_speed_pct < 0.0 || self.max_speed_delta_pct < 0.0 || self.max_jerk_pct < 0.0 {
            return Err(ConfigError::InvalidBounds);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    #[error("safety config contains a non-finite bound")]
    NonFinite,
    #[error("safety config bounds are inconsistent")]
    InvalidBounds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyMode {
    Armed,
    Estopped,
}

impl fmt::Display for SafetyMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Armed => write!(f, "armed"),
            Self::Estopped => write!(f, "estopped"),
        }
    }
}

/// Decision together with the one command that is allowed to reach hardware.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecidedCommand {
    pub decision: SafetyDecision,
    pub command: ActuatorCommand,
}

#[derive(Debug, Clone)]
pub struct SafetyEngine {
    config: SafetyConfig,
    mode: SafetyMode,
    last_spatial_ns: u64,
    last_spatial_latency_ms: f32,
    last_spatial_human: bool,
    last_spatial_min_mm: f32,
    conveyor_speed_pct: f32,
    last_speed_delta_pct: f32,
    halt_reason: String,
    evaluations: u64,
    vetoes: u64,
    clamps: u64,
}

impl SafetyEngine {
    pub fn new(config: SafetyConfig) -> Self {
        Self {
            config,
            mode: SafetyMode::Armed,
            last_spatial_ns: 0,
            last_spatial_latency_ms: 0.0,
            last_spatial_human: false,
            last_spatial_min_mm: f32::MAX,
            conveyor_speed_pct: 0.0,
            last_speed_delta_pct: 0.0,
            halt_reason: String::new(),
            evaluations: 0,
            vetoes: 0,
            clamps: 0,
        }
    }

    pub fn config(&self) -> SafetyConfig {
        self.config
    }

    pub fn mode(&self) -> SafetyMode {
        self.mode
    }

    pub fn is_halted(&self) -> bool {
        self.mode == SafetyMode::Estopped
    }

    pub fn halt_reason(&self) -> &str {
        &self.halt_reason
    }

    pub fn conveyor_speed_pct(&self) -> f32 {
        self.conveyor_speed_pct
    }

    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    pub fn vetoes(&self) -> u64 {
        self.vetoes
    }

    pub fn clamps(&self) -> u64 {
        self.clamps
    }

    /// Feed the latest System 1 heartbeat. Called for *every* frame, including
    /// bypassed ones, so watchdog timing does not depend on System 2.
    pub fn observe_spatial(&mut self, spatial: &SpatialState, received_ns: u64) {
        if spatial.timestamp_ns > self.last_spatial_ns {
            self.last_spatial_ns = spatial.timestamp_ns;
        }
        self.last_spatial_human = spatial.human_intrusion;
        self.last_spatial_min_mm = spatial.min_clearance_mm();
        if received_ns >= spatial.timestamp_ns {
            let observed_ms = elapsed_ms(spatial.timestamp_ns, received_ns);
            self.last_spatial_latency_ms = spatial.inference_ms.max(observed_ms);
        } else if spatial.inference_ms.is_finite() {
            self.last_spatial_latency_ms = spatial.inference_ms.max(0.0);
        }
    }

    /// Explicit operator reset. Never called automatically by perception or
    /// VLM paths.
    pub fn clear_estop(&mut self) {
        self.mode = SafetyMode::Armed;
        self.halt_reason.clear();
        self.last_speed_delta_pct = 0.0;
    }

    /// Evaluate one System 2 proposal and emit the only legal actuator command.
    pub fn evaluate(&mut self, proposal: &VlmProposal, now_ns: u64) -> DecidedCommand {
        self.evaluations = self.evaluations.saturating_add(1);
        if proposal.response.validate().is_err() {
            return self.veto(proposal, &[RuleId::GeoBounds], "invalid VLM schema", now_ns);
        }

        if self.mode == SafetyMode::Estopped {
            return self.veto(
                proposal,
                &[RuleId::LatEstop],
                "E-stop latched; operator reset required",
                now_ns,
            );
        }

        let spatial_age_ms = elapsed_ms(self.last_spatial_ns, now_ns);
        if self.last_spatial_ns == 0 || spatial_age_ms > self.config.watchdog_ms as f32 {
            self.enter_estop("System 1 watchdog expired");
            return self.veto(proposal, &[RuleId::WdStall], "System 1 watchdog expired", now_ns);
        }
        if self.last_spatial_latency_ms > self.config.latency_sla_ms as f32 {
            self.enter_estop("System 1 inference exceeded latency SLA");
            return self.veto(
                proposal,
                &[RuleId::WdLatency],
                "System 1 inference exceeded latency SLA",
                now_ns,
            );
        }

        // Proximity hard-stop is evaluated from the last observed spatial
        // snapshot. It deliberately overrides every semantic proposal.
        if self.last_spatial_human || self.last_spatial_min_mm < self.config.critical_clearance_mm {
            self.enter_estop("critical depth clearance / human intrusion");
            return self.veto(
                proposal,
                &[RuleId::SafProximity],
                "critical proximity: hard E-stop overrides System 2",
                now_ns,
            );
        }

        match &proposal.response.action {
            VlmAction::HaltCell { reason } => {
                self.enter_estop(reason);
                self.approve(
                    proposal,
                    VlmAction::HaltCell { reason: reason.clone() },
                    &[],
                    "VLM halt accepted",
                    now_ns,
                )
            }
            VlmAction::Continue => {
                self.approve(proposal, VlmAction::Continue, &[], "nominal continuation", now_ns)
            }
            VlmAction::DivertPart { bin_id } => {
                if (1..=8).contains(bin_id) {
                    self.approve(
                        proposal,
                        VlmAction::DivertPart { bin_id: *bin_id },
                        &[],
                        "divert target inside workspace",
                        now_ns,
                    )
                } else {
                    self.veto(proposal, &[RuleId::GeoBin], "divert bin outside workspace", now_ns)
                }
            }
            VlmAction::RegraspOffset { dx_mm, dy_mm, dz_mm } => {
                let (x, y, z) = (*dx_mm, *dy_mm, *dz_mm);
                let hard = self.config.hard_regrasp_mm;
                if x.abs() > hard || y.abs() > hard || z.abs() > hard {
                    return self.veto(
                        proposal,
                        &[RuleId::GeoBounds],
                        "regrasp offset exceeds hard kinematic limit",
                        now_ns,
                    );
                }
                let soft = self.config.max_regrasp_mm;
                let cx = clamp_finite(x, -soft, soft);
                let cy = clamp_finite(y, -soft, soft);
                let cz = clamp_finite(z, -soft, soft);
                if (cx, cy, cz) != (x, y, z) {
                    self.clamps = self.clamps.saturating_add(1);
                    let mut out = self.approve(
                        proposal,
                        VlmAction::RegraspOffset { dx_mm: cx, dy_mm: cy, dz_mm: cz },
                        &[RuleId::GeoBounds],
                        "regrasp offset clamped to workspace envelope",
                        now_ns,
                    );
                    out.decision.disposition = Disposition::Clamped;
                    out
                } else {
                    self.approve(
                        proposal,
                        VlmAction::RegraspOffset { dx_mm: x, dy_mm: y, dz_mm: z },
                        &[],
                        "regrasp offset inside envelope",
                        now_ns,
                    )
                }
            }
            VlmAction::SlowConveyor { speed_pct } => {
                let mut rules = Vec::with_capacity(3);
                let requested = f32::from(*speed_pct);
                let mut target = clamp_finite(requested, 0.0, self.config.max_speed_pct);
                if (target - requested).abs() > 0.0 {
                    rules.push(RuleId::SpdRange);
                }
                let mut delta = target - self.conveyor_speed_pct;
                let max_delta = self.config.max_speed_delta_pct;
                if delta > max_delta {
                    delta = max_delta;
                    target = self.conveyor_speed_pct + delta;
                    rules.push(RuleId::SpdDelta);
                } else if delta < -max_delta {
                    delta = -max_delta;
                    target = self.conveyor_speed_pct + delta;
                    rules.push(RuleId::SpdDelta);
                }
                let jerk = delta - self.last_speed_delta_pct;
                if jerk > self.config.max_jerk_pct {
                    delta = self.last_speed_delta_pct + self.config.max_jerk_pct;
                    target = self.conveyor_speed_pct + delta;
                    rules.push(RuleId::SpdJerk);
                } else if jerk < -self.config.max_jerk_pct {
                    delta = self.last_speed_delta_pct - self.config.max_jerk_pct;
                    target = self.conveyor_speed_pct + delta;
                    rules.push(RuleId::SpdJerk);
                }
                target = clamp_finite(target, 0.0, self.config.max_speed_pct);
                self.conveyor_speed_pct = target;
                self.last_speed_delta_pct = delta;
                let clamped_speed = target as u8;
                if rules.is_empty() {
                    self.approve(
                        proposal,
                        VlmAction::SlowConveyor { speed_pct: clamped_speed },
                        &[],
                        "speed command inside rate envelope",
                        now_ns,
                    )
                } else {
                    self.clamps = self.clamps.saturating_add(1);
                    let mut out = self.approve(
                        proposal,
                        VlmAction::SlowConveyor { speed_pct: clamped_speed },
                        &rules,
                        "speed command clamped by rate/jerk envelope",
                        now_ns,
                    );
                    out.decision.disposition = Disposition::Clamped;
                    out
                }
            }
        }
    }

    fn enter_estop(&mut self, reason: &str) {
        self.mode = SafetyMode::Estopped;
        self.halt_reason = reason.to_owned();
        self.last_speed_delta_pct = 0.0;
    }

    fn approve(
        &mut self,
        proposal: &VlmProposal,
        action: VlmAction,
        rules: &[RuleId],
        reason: &str,
        now_ns: u64,
    ) -> DecidedCommand {
        let halted = matches!(action, VlmAction::HaltCell { .. });
        let cmd = ActuatorCommand {
            seq: proposal.seq,
            source_ns: now_ns,
            action,
            source_proposal_id: Some(proposal.request_id),
            vetoed: false,
            clamped: !rules.is_empty(),
            rules: rules.to_vec(),
            reason: reason.to_owned(),
        };
        DecidedCommand {
            decision: SafetyDecision {
                disposition: Disposition::Approved,
                rules: rules.to_vec(),
                reason: reason.to_owned(),
                conveyor_speed_pct: self.conveyor_speed_pct,
                halted,
                evaluated_ns: now_ns,
            },
            command: cmd,
        }
    }

    fn veto(
        &mut self,
        proposal: &VlmProposal,
        rules: &[RuleId],
        reason: &str,
        now_ns: u64,
    ) -> DecidedCommand {
        self.vetoes = self.vetoes.saturating_add(1);
        let cmd = ActuatorCommand {
            seq: proposal.seq,
            source_ns: now_ns,
            action: VlmAction::HaltCell { reason: reason.to_owned() },
            source_proposal_id: Some(proposal.request_id),
            vetoed: true,
            clamped: false,
            rules: rules.to_vec(),
            reason: reason.to_owned(),
        };
        DecidedCommand {
            decision: SafetyDecision {
                disposition: Disposition::Vetoed,
                rules: rules.to_vec(),
                reason: reason.to_owned(),
                conveyor_speed_pct: 0.0,
                halted: true,
                evaluated_ns: now_ns,
            },
            command: cmd,
        }
    }
}

fn elapsed_ms(from_ns: u64, to_ns: u64) -> f32 {
    let delta = to_ns.saturating_sub(from_ns);
    delta as f32 / NS_PER_MS as f32
}

fn clamp_finite(value: f32, min: f32, max: f32) -> f32 {
    if !value.is_finite() || value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_proto::{ZoneClearance, ZoneId};
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(1);

    fn spatial(min_mm: f32, human: bool, ts: u64) -> SpatialState {
        SpatialState {
            seq: 1,
            timestamp_ns: ts,
            frame_hash: [0; 32],
            zones: vec![ZoneClearance {
                zone: ZoneId::Critical,
                min_mm,
                mean_mm: min_mm,
                sample_count: 1,
            }],
            human_intrusion: human,
            obstacle_intrusion: min_mm < 120.0,
            anomaly_score: 0.0,
            inference_ms: 1.0,
            engine: "test".to_owned(),
        }
    }

    fn proposal(action: VlmAction) -> VlmProposal {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        VlmProposal {
            seq,
            request_id: seq,
            timestamp_ns: 0,
            trace_id: format!("t-{seq}"),
            model: "mock".to_owned(),
            latency_ms: 1.0,
            response: aegis_proto::VlmResponse {
                action,
                confidence: 0.9,
                rationale: String::new(),
            },
        }
    }

    fn armed_engine() -> SafetyEngine {
        let mut e = SafetyEngine::new(SafetyConfig::default());
        e.observe_spatial(&spatial(500.0, false, 1_000_000_000), 1_001_000_000);
        e
    }

    #[test]
    fn approve_nominal_continue() {
        let mut e = armed_engine();
        let out = e.evaluate(&proposal(VlmAction::Continue), 1_010_000_000);
        assert_eq!(out.decision.disposition, Disposition::Approved);
        assert_eq!(out.command.action, VlmAction::Continue);
        assert!(!out.command.vetoed);
    }

    #[test]
    fn proximity_overrides_continue_and_slow() {
        for action in [VlmAction::Continue, VlmAction::SlowConveyor { speed_pct: 80 }] {
            let mut e = armed_engine();
            e.observe_spatial(&spatial(40.0, false, 1_001_000_000), 1_002_000_000);
            let out = e.evaluate(&proposal(action), 1_003_000_000);
            assert_eq!(out.decision.disposition, Disposition::Vetoed);
            assert!(out.decision.rules.contains(&RuleId::SafProximity));
            assert!(out.command.vetoed);
            assert!(e.is_halted());
        }
    }

    #[test]
    fn human_intrusion_is_hard_estop_even_without_low_clearance() {
        let mut e = armed_engine();
        e.observe_spatial(&spatial(900.0, true, 1_001_000_000), 1_002_000_000);
        let out = e.evaluate(&proposal(VlmAction::Continue), 1_003_000_000);
        assert_eq!(out.decision.disposition, Disposition::Vetoed);
        assert!(out.decision.rules.contains(&RuleId::SafProximity));
    }

    #[test]
    fn watchdog_expiry_forces_halt() {
        let mut e = SafetyEngine::new(SafetyConfig::default());
        e.observe_spatial(&spatial(500.0, false, 1_000_000_000), 1_001_000_000);
        let out = e.evaluate(&proposal(VlmAction::Continue), 1_500_000_000);
        assert_eq!(out.decision.disposition, Disposition::Vetoed);
        assert!(out.decision.rules.contains(&RuleId::WdStall));
    }

    #[test]
    fn latency_sla_violation_forces_halt() {
        let cfg = SafetyConfig { latency_sla_ms: 5, ..SafetyConfig::default() };
        let mut e = SafetyEngine::new(cfg);
        e.observe_spatial(&spatial(500.0, false, 1_000_000_000), 1_040_000_000);
        let out = e.evaluate(&proposal(VlmAction::Continue), 1_041_000_000);
        assert!(out.decision.rules.contains(&RuleId::WdLatency));
    }

    #[test]
    fn out_of_bounds_offset_is_vetoed() {
        let mut e = armed_engine();
        let out = e.evaluate(
            &proposal(VlmAction::RegraspOffset { dx_mm: 900.0, dy_mm: 0.0, dz_mm: 0.0 }),
            1_010_000_000,
        );
        assert_eq!(out.decision.disposition, Disposition::Vetoed);
        assert!(out.decision.rules.contains(&RuleId::GeoBounds));
        assert!(out.command.vetoed);
    }

    #[test]
    fn soft_offset_is_clamped_and_logged() {
        let mut e = armed_engine();
        let out = e.evaluate(
            &proposal(VlmAction::RegraspOffset { dx_mm: 40.0, dy_mm: -1.0, dz_mm: 0.0 }),
            1_010_000_000,
        );
        assert_eq!(out.decision.disposition, Disposition::Clamped);
        match out.command.action {
            VlmAction::RegraspOffset { dx_mm, .. } => assert_eq!(dx_mm, 25.0),
            other => panic!("unexpected action {other:?}"),
        }
        assert!(out.decision.rules.contains(&RuleId::GeoBounds));
    }

    #[test]
    fn speed_delta_and_jerk_are_clamped() {
        let mut e = armed_engine();
        let first = e.evaluate(&proposal(VlmAction::SlowConveyor { speed_pct: 90 }), 1_010_000_000);
        assert_eq!(first.decision.disposition, Disposition::Clamped);
        assert!(first.decision.rules.contains(&RuleId::SpdDelta));
        assert!(first.decision.rules.contains(&RuleId::SpdJerk));
        assert_eq!(e.conveyor_speed_pct(), 6.0);
        let second = e.evaluate(&proposal(VlmAction::SlowConveyor { speed_pct: 0 }), 1_020_000_000);
        assert!(second.decision.rules.contains(&RuleId::SpdJerk));
        assert_eq!(e.conveyor_speed_pct(), 6.0);
    }

    #[test]
    fn estop_latches_until_operator_reset() {
        let mut e = armed_engine();
        e.observe_spatial(&spatial(10.0, true, 1_001_000_000), 1_002_000_000);
        let _ = e.evaluate(&proposal(VlmAction::Continue), 1_003_000_000);
        e.observe_spatial(&spatial(900.0, false, 1_004_000_000), 1_005_000_000);
        let out = e.evaluate(&proposal(VlmAction::Continue), 1_006_000_000);
        assert!(out.decision.rules.contains(&RuleId::LatEstop));
        e.clear_estop();
        let out = e.evaluate(&proposal(VlmAction::Continue), 1_007_000_000);
        assert_eq!(out.decision.disposition, Disposition::Approved);
    }

    #[test]
    fn nan_proposals_cannot_panic_or_pass() {
        let mut e = armed_engine();
        let out = e.evaluate(
            &proposal(VlmAction::RegraspOffset { dx_mm: f32::NAN, dy_mm: 0.0, dz_mm: 0.0 }),
            1_010_000_000,
        );
        assert_eq!(out.decision.disposition, Disposition::Vetoed);
    }

    #[test]
    fn decisions_are_deterministic_across_engines() {
        let p = proposal(VlmAction::SlowConveyor { speed_pct: 70 });
        let mut a = armed_engine();
        let mut b = armed_engine();
        let da = a.evaluate(&p, 1_010_000_000);
        let db = b.evaluate(&p, 1_010_000_000);
        assert_eq!(da, db);
    }
}

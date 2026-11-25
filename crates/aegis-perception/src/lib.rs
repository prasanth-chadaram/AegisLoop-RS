//! System 1: deterministic, bounded-latency spatial perception.
//!
//! The hot path runs a reusable depth buffer, a fixed set of safety zones and
//! a bounded latency ring. No queue can grow without bound and no runtime path
//! panics. A deterministic synthetic estimator keeps tests and CI fully
//! offline; the `onnx` feature swaps in Depth Anything V2 Small on CPU.

pub mod stats;

#[cfg(feature = "onnx")]
mod onnx;

use aegis_proto::{FrameMeta, RgbFrame, SpatialState, ZoneClearance, ZoneId};
use serde::{Deserialize, Serialize};
use stats::LatencyStats;
use std::time::Instant;
use tokio::sync::mpsc;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PerceptionError {
    #[error("empty or mismatched frame: {0}")]
    InvalidFrame(String),
    #[error("ONNX backend error: {0}")]
    Model(String),
    #[error("frame channel closed")]
    ChannelClosed,
}

/// Normalized rectangle in image coordinates (`x`/`y` in `[0, 1]`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZoneSpec {
    pub zone: ZoneId,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialConfig {
    pub depth_width: u32,
    pub depth_height: u32,
    /// Converts normalized depth to millimetres.
    pub depth_scale_mm: f32,
    pub critical_mm: f32,
    pub zones: Vec<ZoneSpec>,
    /// Enable deterministic scene hints (simulator only). Real models set this
    /// to `false` so untrusted frame metadata can never create an intrusion.
    pub allow_scene_hints: bool,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            depth_width: 64,
            depth_height: 64,
            depth_scale_mm: 2000.0,
            critical_mm: 120.0,
            zones: vec![
                ZoneSpec { zone: ZoneId::Near, x0: 0.10, y0: 0.05, x1: 0.90, y1: 0.35 },
                ZoneSpec { zone: ZoneId::Warning, x0: 0.15, y0: 0.35, x1: 0.85, y1: 0.65 },
                ZoneSpec { zone: ZoneId::Critical, x0: 0.30, y0: 0.65, x1: 0.70, y1: 0.98 },
            ],
            allow_scene_hints: true,
        }
    }
}

impl SpatialConfig {
    pub fn validate(&self) -> Result<(), PerceptionError> {
        if self.depth_width == 0 || self.depth_height == 0 {
            return Err(PerceptionError::InvalidFrame(
                "depth dimensions must be non-zero".to_owned(),
            ));
        }
        if !self.depth_scale_mm.is_finite() || !self.critical_mm.is_finite() {
            return Err(PerceptionError::InvalidFrame(
                "depth scale / critical threshold must be finite".to_owned(),
            ));
        }
        if self.zones.is_empty() {
            return Err(PerceptionError::InvalidFrame(
                "at least one safety zone is required".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn engine_name(&self) -> &'static str {
        #[cfg(feature = "onnx")]
        {
            "depth-anything-v2-small-onnx"
        }
        #[cfg(not(feature = "onnx"))]
        {
            "synthetic-deterministic-v1"
        }
    }
}

/// Depth-estimation backend. Implementations write normalized depth into a
/// caller-owned scratch buffer so the hot path never allocates a depth map.
pub trait DepthEstimator {
    fn estimate_into(
        &mut self,
        frame: &RgbFrame,
        width: usize,
        height: usize,
        out: &mut [f32],
    ) -> Result<(), PerceptionError>;
    fn anomaly_hint(&self) -> f32 {
        0.0
    }
}

/// Fully deterministic fallback estimator. Same frame sequence always yields
/// bit-identical depth values, without model weights, GPU or network.
#[derive(Debug, Clone)]
pub struct SyntheticDepthEstimator {
    seed: u64,
    anomaly: f32,
}

impl Default for SyntheticDepthEstimator {
    fn default() -> Self {
        Self::new(0xAE71_5C0F)
    }
}

impl SyntheticDepthEstimator {
    pub fn new(seed: u64) -> Self {
        Self { seed, anomaly: 0.0 }
    }
}

#[inline]
fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl DepthEstimator for SyntheticDepthEstimator {
    fn estimate_into(
        &mut self,
        frame: &RgbFrame,
        width: usize,
        height: usize,
        out: &mut [f32],
    ) -> Result<(), PerceptionError> {
        if frame.meta.width == 0 || frame.meta.height == 0 || width == 0 || height == 0 {
            return Err(PerceptionError::InvalidFrame("zero-sized frame".to_owned()));
        }
        let len = width.saturating_mul(height).min(out.len());
        if len == 0 {
            return Err(PerceptionError::InvalidFrame("zero-sized depth buffer".to_owned()));
        }
        let scene = frame.meta.scene_tag.as_str();
        self.anomaly = if scene.eq_ignore_ascii_case("defective_part") {
            0.72
        } else if scene.eq_ignore_ascii_case("vlm_hallucination") {
            0.50
        } else {
            0.0
        };
        let seq_seed = mix64(self.seed ^ frame.meta.seq);
        for (i, value) in out.iter_mut().enumerate().take(len) {
            let y = i / width;
            let x = i % width;
            let ny = y as f32 / height as f32;
            let noise = (mix64(seq_seed.wrapping_add(i as u64)) & 0xffff) as f32 / 65535.0;
            let mut depth = 0.34 + 0.16 * ny + 0.02 * noise;
            if scene.eq_ignore_ascii_case("human_intrusion") {
                let cx = (x as f32 / width as f32 - 0.5).powi(2);
                let cy = (ny - 0.82).powi(2);
                if cx + cy < 0.012 && ny > 0.65 && ny < 0.98 {
                    depth = 0.035;
                }
            } else if scene.eq_ignore_ascii_case("obstacle_intrusion") {
                let cx = (x as f32 / width as f32 - 0.5).powi(2);
                let cy = (ny - 0.50).powi(2);
                if cx + cy < 0.010 {
                    depth = 0.10;
                }
            }
            *value = depth;
        }
        Ok(())
    }

    fn anomaly_hint(&self) -> f32 {
        self.anomaly
    }
}

/// Compute per-zone clearance from a normalized depth buffer.
pub fn compute_spatial(
    cfg: &SpatialConfig,
    meta: &FrameMeta,
    depth: &[f32],
    anomaly_hint: f32,
    engine: &str,
) -> SpatialState {
    let width = usize::try_from(cfg.depth_width).unwrap_or(0).max(1);
    let height = usize::try_from(cfg.depth_height).unwrap_or(0).max(1);
    let total = width.saturating_mul(height).min(depth.len());
    let mut zones = Vec::with_capacity(cfg.zones.len());
    let mut obstacle = false;
    for spec in &cfg.zones {
        let x0 = rect_index(spec.x0, width);
        let x1 = rect_index(spec.x1, width).max(x0 + 1).min(width);
        let y0 = rect_index(spec.y0, height);
        let y1 = rect_index(spec.y1, height).max(y0 + 1).min(height);
        let mut min = f32::MAX;
        let mut sum = 0.0_f64;
        let mut count = 0_u32;
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = y.saturating_mul(width).saturating_add(x);
                if idx >= total {
                    continue;
                }
                let d = depth[idx];
                if !d.is_finite() {
                    continue;
                }
                let mm = d * cfg.depth_scale_mm;
                if mm < min {
                    min = mm;
                }
                sum += f64::from(mm);
                count = count.saturating_add(1);
            }
        }
        let min_mm = if count == 0 { 0.0 } else { min };
        let mean_mm = if count == 0 { 0.0 } else { (sum / f64::from(count)) as f32 };
        if count > 0 && min_mm < cfg.critical_mm {
            obstacle = true;
        }
        zones.push(ZoneClearance { zone: spec.zone, min_mm, mean_mm, sample_count: count });
    }
    let human =
        cfg.allow_scene_hints && meta.scene_tag.eq_ignore_ascii_case("human_intrusion") && obstacle;
    SpatialState {
        seq: meta.seq,
        timestamp_ns: meta.monotonic_ns,
        frame_hash: meta.frame_hash,
        zones,
        human_intrusion: human,
        obstacle_intrusion: obstacle,
        anomaly_score: anomaly_hint.clamp(0.0, 1.0),
        inference_ms: 0.0,
        engine: engine.to_owned(),
    }
}

fn rect_index(normalized: f32, extent: usize) -> usize {
    let clamped = normalized.clamp(0.0, 1.0);
    let index = (clamped * extent as f32) as usize;
    index.min(extent.saturating_sub(1))
}

/// System 1 pipeline: one reusable depth buffer plus bounded statistics.
pub struct SpatialPipeline {
    config: SpatialConfig,
    estimator: Box<dyn DepthEstimator>,
    depth: Vec<f32>,
    stats: LatencyStats,
    frames: u64,
    drops: u64,
}

impl SpatialPipeline {
    pub fn new(
        config: SpatialConfig,
        estimator: Box<dyn DepthEstimator>,
    ) -> Result<Self, PerceptionError> {
        config.validate()?;
        let len = (config.depth_width as usize).saturating_mul(config.depth_height as usize);
        if len == 0 {
            return Err(PerceptionError::InvalidFrame("empty depth buffer".to_owned()));
        }
        Ok(Self {
            config,
            estimator,
            depth: vec![0.0; len],
            stats: LatencyStats::new(4096),
            frames: 0,
            drops: 0,
        })
    }

    pub fn synthetic(config: SpatialConfig) -> Result<Self, PerceptionError> {
        Self::new(config, Box::new(SyntheticDepthEstimator::default()))
    }

    #[cfg(feature = "onnx")]
    pub fn onnx(
        config: SpatialConfig,
        model_path: &std::path::Path,
    ) -> Result<Self, PerceptionError> {
        let estimator = onnx::OnnxDepthEstimator::new(
            model_path,
            config.depth_width as usize,
            config.depth_height as usize,
        )?;
        Self::new(config, Box::new(estimator))
    }

    pub fn config(&self) -> &SpatialConfig {
        &self.config
    }

    pub fn stats(&self) -> &LatencyStats {
        &self.stats
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    pub fn drops(&self) -> u64 {
        self.drops
    }

    /// Non-blocking submit into a bounded channel. Drops are counted, never
    /// converted into an unbounded queue.
    pub fn try_submit(&mut self, tx: &mpsc::Sender<RgbFrame>, frame: RgbFrame) -> bool {
        match tx.try_send(frame) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.drops = self.drops.saturating_add(1);
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.drops = self.drops.saturating_add(1);
                false
            }
        }
    }

    /// Single-frame hot path. No heap allocation in estimation or zone scan.
    pub fn process(&mut self, frame: &RgbFrame) -> Result<SpatialState, PerceptionError> {
        if frame.meta.width == 0 || frame.meta.height == 0 {
            return Err(PerceptionError::InvalidFrame("zero-sized frame".to_owned()));
        }
        let started = Instant::now();
        self.estimator.estimate_into(
            frame,
            self.config.depth_width as usize,
            self.config.depth_height as usize,
            &mut self.depth,
        )?;
        let mut spatial = compute_spatial(
            &self.config,
            &frame.meta,
            &self.depth,
            self.estimator.anomaly_hint(),
            self.config.engine_name(),
        );
        spatial.inference_ms = (started.elapsed().as_secs_f64() * 1000.0) as f32;
        self.stats.push(spatial.inference_ms);
        self.frames = self.frames.saturating_add(1);
        Ok(spatial)
    }
}

/// Snapshot of system memory high-water mark from Linux `/proc`.
pub fn peak_rss_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return kib.checked_mul(1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(seq: u64, tag: &str) -> RgbFrame {
        let pixels = vec![128_u8; 64 * 64 * 3];
        let meta = FrameMeta::new(seq, seq * 1_000_000, "cam-0", 64, 64, &pixels, tag);
        RgbFrame { meta, pixels }
    }

    #[test]
    fn synthetic_pipeline_is_deterministic() {
        let mut a = SpatialPipeline::synthetic(SpatialConfig::default()).expect("pipeline");
        let mut b = SpatialPipeline::synthetic(SpatialConfig::default()).expect("pipeline");
        let f = frame(7, "nominal");
        let sa = a.process(&f).expect("spatial");
        let sb = b.process(&f).expect("spatial");
        assert_eq!(sa.zones, sb.zones);
        assert_eq!(sa.frame_hash, sb.frame_hash);
        assert!(!sa.obstacle_intrusion);
    }

    #[test]
    fn intrusion_is_detected_in_critical_zone() {
        let mut p = SpatialPipeline::synthetic(SpatialConfig::default()).expect("pipeline");
        let s = p.process(&frame(9, "human_intrusion")).expect("spatial");
        assert!(s.obstacle_intrusion);
        assert!(s.human_intrusion);
        assert!(s.clearance_for(ZoneId::Critical) < 120.0);
    }

    #[test]
    fn scene_hints_can_be_disabled_for_real_models() {
        let cfg = SpatialConfig { allow_scene_hints: false, ..SpatialConfig::default() };
        let mut p = SpatialPipeline::synthetic(cfg).expect("pipeline");
        let s = p.process(&frame(9, "human_intrusion")).expect("spatial");
        assert!(s.obstacle_intrusion);
        assert!(!s.human_intrusion);
    }

    #[test]
    fn different_sequences_produce_different_depth() {
        let mut p = SpatialPipeline::synthetic(SpatialConfig::default()).expect("pipeline");
        let a = p.process(&frame(1, "nominal")).expect("spatial");
        let b = p.process(&frame(2, "nominal")).expect("spatial");
        assert_ne!(a.zones[0].mean_mm, b.zones[0].mean_mm);
    }

    #[test]
    fn bounded_channel_never_grows() {
        let (tx, _rx) = mpsc::channel::<RgbFrame>(2);
        let mut p = SpatialPipeline::synthetic(SpatialConfig::default()).expect("pipeline");
        for i in 0..10 {
            p.try_submit(&tx, frame(i, "nominal"));
        }
        assert!(p.drops() >= 8);
    }
}

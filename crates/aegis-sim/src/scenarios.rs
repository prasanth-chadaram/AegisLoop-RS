//! Deterministic workcell scenarios. Scene tags are consumed only by the
//! synthetic System 1 estimator; a real ONNX backend ignores them.

use aegis_proto::{FrameMeta, RgbFrame};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scenario {
    Nominal,
    DefectivePart,
    VlmHallucination,
    HumanIntrusion,
}

impl Scenario {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::DefectivePart => "defective_part",
            Self::VlmHallucination => "vlm_hallucination",
            Self::HumanIntrusion => "human_intrusion",
        }
    }
}

impl fmt::Display for Scenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tag())
    }
}

/// Nominal flow, then defect, hallucination and finally intrusion. The
/// intrusion is last so its latched E-stop terminates the run cleanly.
pub fn default_sequence(total_frames: u64) -> Vec<Scenario> {
    let mut out = Vec::with_capacity(total_frames as usize);
    let intrusion_at = total_frames.saturating_sub(1);
    let defect_at = total_frames.saturating_mul(2) / 3;
    let hallucination_at = total_frames.saturating_mul(4) / 5;
    for i in 0..total_frames {
        let scenario = if i == intrusion_at {
            Scenario::HumanIntrusion
        } else if i == hallucination_at {
            Scenario::VlmHallucination
        } else if i == defect_at {
            Scenario::DefectivePart
        } else {
            Scenario::Nominal
        };
        out.push(scenario);
    }
    out
}

/// Reusable frame buffer. Pixel data is regenerated deterministically and the
/// tag drives the synthetic depth estimator.
pub struct WorkcellSimulator {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl WorkcellSimulator {
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width as usize).saturating_mul(height as usize).saturating_mul(3);
        Self { width, height, pixels: vec![0; len.max(3)] }
    }

    pub fn next_frame(&mut self, seq: u64, scenario: Scenario) -> RgbFrame {
        let mut seed = seq.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for (i, pixel) in self.pixels.iter_mut().enumerate() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *pixel = ((seed >> 24) as u8).wrapping_add((i % 3) as u8 * 7);
        }
        let meta = FrameMeta::new(
            seq,
            super::now_ns(),
            "workcell-cam-0",
            self.width,
            self.height,
            &self.pixels,
            scenario.tag(),
        );
        RgbFrame { meta, pixels: self.pixels.clone() }
    }
}

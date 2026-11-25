//! Optional real-weight backend: Depth Anything V2 Small ONNX on CPU.
//!
//! Only compiled with `--features onnx`. Weights are never stored in git;
//! `scripts/download_models.sh` places them under `$AEGIS_MODEL_DIR`.
//! Intra-op parallelism is capped at 2 threads to respect the i3-1005G1 budget.

use crate::{DepthEstimator, PerceptionError};
use aegis_proto::RgbFrame;
use ort::{
    environment::Environment,
    session::{builder::GraphOptimizationLevel, Session},
    value::{Tensor, TensorRef},
};
use std::path::Path;

pub struct OnnxDepthEstimator {
    #[allow(dead_code)]
    env: Environment,
    session: Session,
    width: usize,
    height: usize,
    /// Reused NCHW input tensor storage (1 * 3 * H * W floats).
    input: Vec<f32>,
}

impl OnnxDepthEstimator {
    pub fn new(model_path: &Path, width: usize, height: usize) -> Result<Self, PerceptionError> {
        if width == 0 || height == 0 {
            return Err(PerceptionError::InvalidFrame("zero-sized model input".to_owned()));
        }
        let env = ort::init()
            .with_name("aegisloop-system1")
            .with_telemetry(false)
            .build()
            .map_err(|e| PerceptionError::Model(e.to_string()))?;
        let session = Session::builder(&env)
            .map_err(|e| PerceptionError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| PerceptionError::Model(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| PerceptionError::Model(e.to_string()))?
            .with_execution_providers([ort::ep::CPU::default().build()])
            .map_err(|e| PerceptionError::Model(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| PerceptionError::Model(e.to_string()))?;
        let count = width.saturating_mul(height).saturating_mul(3).max(1);
        Ok(Self { env, session, width, height, input: vec![0.0; count] })
    }
}

impl DepthEstimator for OnnxDepthEstimator {
    fn name(&self) -> &'static str {
        "depth-anything-v2-small-onnx"
    }

    fn estimate_into(
        &mut self,
        frame: &RgbFrame,
        width: usize,
        height: usize,
        out: &mut [f32],
    ) -> Result<(), PerceptionError> {
        let width = width.min(self.width);
        let height = height.min(self.height);
        let src_w = usize::try_from(frame.meta.width).unwrap_or(0);
        let src_h = usize::try_from(frame.meta.height).unwrap_or(0);
        if src_w == 0
            || src_h == 0
            || frame.pixels.len() < src_w.saturating_mul(src_h).saturating_mul(3)
        {
            return Err(PerceptionError::InvalidFrame("incomplete RGB frame".to_owned()));
        }
        // Nearest-neighbour resample into reused NCHW scratch (no allocation).
        for y in 0..height {
            let sy = y.saturating_mul(src_h).saturating_div(height).min(src_h - 1);
            for x in 0..width {
                let sx = x.saturating_mul(src_w).saturating_div(width).min(src_w - 1);
                let src = sy.saturating_mul(src_w).saturating_add(sx).saturating_mul(3);
                for c in 0usize..3 {
                    let dst = c
                        .saturating_mul(height)
                        .saturating_mul(width)
                        .saturating_add(y.saturating_mul(width))
                        .saturating_add(x);
                    if dst < self.input.len() && src + c < frame.pixels.len() {
                        self.input[dst] = f32::from(frame.pixels[src + c]) / 255.0;
                    }
                }
            }
        }
        let view = ndarray::ArrayView4::from_shape((1, 3, height, width), &self.input)
            .map_err(|e| PerceptionError::Model(e.to_string()))?;
        let tensor =
            TensorRef::from_array_view(view).map_err(|e| PerceptionError::Model(e.to_string()))?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(|e| PerceptionError::Model(e.to_string()))?;
        let (_, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| PerceptionError::Model(e.to_string()))?;
        let n = out.len().min(values.len());
        for (dst, src) in out.iter_mut().zip(values.iter().take(n)) {
            *dst = if src.is_finite() { *src } else { 0.5 };
        }
        // Depth Anything emits relative/inverse depth; calibrate to metres in
        // the model-specific preprocessor before multiplying by depth_scale_mm.
        Ok(())
    }
}

#[allow(dead_code)]
fn _assert_tensor_path_compiles(t: Tensor<f32>) -> Tensor<f32> {
    t
}

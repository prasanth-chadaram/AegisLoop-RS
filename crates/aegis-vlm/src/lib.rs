//! System 2: asynchronous semantic triage. It never blocks System 1.
//!
//! The worker is triggered only by escalations from spatial perception. HTTP
//! timeouts, malformed JSON and unreachable endpoints all collapse to a safe
//! fallback proposal; a failure here can never stall the deterministic loop.

use aegis_proto::{SpatialState, VlmAction, VlmProposal, VlmResponse};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::warn;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VlmConfig {
    /// OpenAI-compatible endpoint, e.g. llama-server on localhost.
    pub endpoint: String,
    pub model: String,
    /// Hard SLA. Any slower response is discarded.
    pub timeout_ms: u64,
    pub queue_capacity: usize,
    /// Escalation score at or above which System 2 is invoked.
    pub escalation_threshold: f32,
}

impl Default for VlmConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8081/v1/chat/completions".to_owned(),
            model: "smolvlm2-256m-q4_k_m".to_owned(),
            timeout_ms: 1500,
            queue_capacity: 8,
            escalation_threshold: 0.45,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VlmError {
    #[error("VLM request timed out after {0} ms")]
    Timeout(u64),
    #[error("VLM transport failure: {0}")]
    Transport(String),
    #[error("VLM returned malformed JSON: {0}")]
    Malformed(String),
    #[error("VLM response failed schema validation: {0}")]
    Schema(String),
    #[error("VLM endpoint unavailable")]
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VlmRequest {
    pub seq: u64,
    pub timestamp_ns: u64,
    pub frame_hash: [u8; 32],
    pub spatial: SpatialState,
    /// Raw RGB pixels, sent to the VLM as a data-URL PNG when available.
    pub image_rgb: Option<Vec<u8>>,
    pub image_width: u32,
    pub image_height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutcomeKind {
    Bypassed,
    Accepted,
    FallbackTimeout,
    FallbackMalformed,
    FallbackUnreachable,
}

impl OutcomeKind {
    pub fn is_bypass(self) -> bool {
        self == Self::Bypassed
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VlmResult {
    pub proposal: VlmProposal,
    pub kind: OutcomeKind,
    /// Spatial snapshot that triggered the request; carried into the ledger so
    /// VLM decisions are bound to the exact perception state.
    pub spatial: SpatialState,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct VlmCounters {
    pub requests: u64,
    pub bypassed: u64,
    pub accepted: u64,
    pub fallbacks: u64,
}

/// Counters shared with the control-plane dashboard. Atomic, allocation free.
#[derive(Debug, Default)]
pub struct SharedVlmStats {
    requests: AtomicU64,
    bypassed: AtomicU64,
    accepted: AtomicU64,
    fallbacks: AtomicU64,
}

impl SharedVlmStats {
    pub fn snapshot(&self) -> VlmCounters {
        VlmCounters {
            requests: self.requests.load(Ordering::Relaxed),
            bypassed: self.bypassed.load(Ordering::Relaxed),
            accepted: self.accepted.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
        }
    }

    fn record(&self, kind: OutcomeKind) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        match kind {
            OutcomeKind::Bypassed => {
                self.bypassed.fetch_add(1, Ordering::Relaxed);
            }
            OutcomeKind::Accepted => {
                self.accepted.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.fallbacks.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Backend abstraction. Trait uses an async method; implementations are
/// monomorphized so no dynamic dispatch occurs in the worker loop.
pub trait VlmBackend {
    fn propose<'a>(
        &'a self,
        request: &'a VlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<VlmResponse, VlmError>> + Send + 'a>>;
}

pub struct MockBackend {
    response: Result<VlmResponse, VlmError>,
    delay: Duration,
}

impl MockBackend {
    pub fn valid(action: VlmAction, delay: Duration) -> Self {
        Self {
            response: Ok(VlmResponse {
                action,
                confidence: 0.91,
                rationale: "mock semantic triage".to_owned(),
            }),
            delay,
        }
    }

    pub fn failing(error: VlmError) -> Self {
        Self { response: Err(error), delay: Duration::ZERO }
    }
}

impl VlmBackend for MockBackend {
    fn propose<'a>(
        &'a self,
        _request: &'a VlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<VlmResponse, VlmError>> + Send + 'a>> {
        let response = self.response.clone();
        let delay = self.delay;
        Box::pin(async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            response
        })
    }
}

#[derive(Clone)]
pub struct HttpVlmBackend {
    client: reqwest::Client,
    config: VlmConfig,
}

impl HttpVlmBackend {
    pub fn new(config: VlmConfig) -> Result<Self, VlmError> {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(2)
            .build()
            .map_err(|e| VlmError::Transport(e.to_string()))?;
        Ok(Self { client, config })
    }
}

/// Constrained decoding schema sent to llama-server. The response is still
/// re-validated client-side; the grammar only reduces malformed outputs.
static VLM_JSON_SCHEMA: std::sync::LazyLock<serde_json::Value> = std::sync::LazyLock::new(|| {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["action", "confidence"],
        "properties": {
            "action": {
                "oneOf": [
                    {"type": "object", "additionalProperties": false, "required": ["type"], "properties": {"type": {"const": "continue"}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "speed_pct"], "properties": {"type": {"const": "slow_conveyor"}, "speed_pct": {"type": "integer", "minimum": 0, "maximum": 100}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "bin_id"], "properties": {"type": {"const": "divert_part"}, "bin_id": {"type": "integer", "minimum": 1, "maximum": 8}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "dx_mm", "dy_mm", "dz_mm"], "properties": {"type": {"const": "regrasp_offset"}, "dx_mm": {"type": "number"}, "dy_mm": {"type": "number"}, "dz_mm": {"type": "number"}}},
                    {"type": "object", "additionalProperties": false, "required": ["type", "reason"], "properties": {"type": {"const": "halt_cell"}, "reason": {"type": "string"}}}
                ]
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"}
        }
    })
});

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
struct Message {
    content: String,
}

impl VlmBackend for HttpVlmBackend {
    fn propose<'a>(
        &'a self,
        request: &'a VlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<VlmResponse, VlmError>> + Send + 'a>> {
        Box::pin(async move {
            let spatial = &request.spatial;
            let prompt = format!(
                "Spatial: min_clearance_mm={:.1}, human={}, obstacle={}, anomaly={:.2}. \
                 Propose one safe action as strict JSON only: \
                 {{\"action\":{{\"type\":\"continue\"|\"slow_conveyor\"|\"divert_part\"|\"regrasp_offset\"|\"halt_cell\", ...}},\"confidence\":0..1,\"rationale\":\"...\"}}",
                spatial.min_clearance_mm(),
                spatial.human_intrusion,
                spatial.obstacle_intrusion,
                spatial.anomaly_score
            );
            let mut user_content = serde_json::json!([
                {"type": "text", "text": prompt}
            ]);
            if let (Some(pixels), w, h) =
                (request.image_rgb.as_ref(), request.image_width, request.image_height)
            {
                if w > 0 && h > 0 && !pixels.is_empty() {
                    match encode_png(pixels, w, h) {
                        Ok(png_bytes) => {
                            let url = format!(
                                "data:image/png;base64,{}",
                                base64::Engine::encode(
                                    &base64::engine::general_purpose::STANDARD,
                                    &png_bytes
                                )
                            );
                            if let Some(array) = user_content.as_array_mut() {
                                array.push(serde_json::json!({
                                    "type": "image_url",
                                    "image_url": {"url": url}
                                }));
                            }
                        }
                        Err(e) => warn!("PNG encoding failed: {e}"),
                    }
                }
            }
            let body = serde_json::json!({
                "model": self.config.model,
                "temperature": 0.0,
                "max_tokens": 96,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "aegis_vlm_intent",
                        "strict": true,
                        "schema": VLM_JSON_SCHEMA.clone()
                    }
                },
                "messages": [
                    {"role": "system", "content": "Output strict JSON only. No markdown."},
                    {"role": "user", "content": user_content}
                ]
            });
            let response = self
                .client
                .post(&self.config.endpoint)
                .timeout(Duration::from_millis(self.config.timeout_ms))
                .json(&body)
                .send()
                .await
                .map_err(|e| VlmError::Transport(e.to_string()))?;
            if !response.status().is_success() {
                return Err(VlmError::Unavailable);
            }
            let chat: ChatResponse =
                response.json().await.map_err(|e| VlmError::Malformed(e.to_string()))?;
            let content = chat
                .choices
                .first()
                .map(|c| c.message.content.as_str())
                .ok_or_else(|| VlmError::Malformed("empty choices".to_owned()))?;
            let parsed: VlmResponse = match serde_json::from_str(content) {
                Ok(parsed) => parsed,
                Err(e) => {
                    warn!(
                        "VLM returned malformed JSON ({}): {}",
                        e,
                        content.chars().take(240).collect::<String>()
                    );
                    return Err(VlmError::Malformed(e.to_string()));
                }
            };
            if let Err(e) = parsed.validate() {
                warn!("VLM response failed schema validation: {e}");
                return Err(VlmError::Schema(e));
            }
            Ok(parsed)
        })
    }
}

pub struct VlmWorker<B: VlmBackend> {
    backend: B,
    config: VlmConfig,
    receiver: mpsc::Receiver<VlmRequest>,
    sender: mpsc::Sender<VlmResult>,
    stats: Arc<SharedVlmStats>,
}

impl<B: VlmBackend> VlmWorker<B> {
    pub fn new(
        backend: B,
        config: VlmConfig,
        receiver: mpsc::Receiver<VlmRequest>,
        sender: mpsc::Sender<VlmResult>,
        stats: Arc<SharedVlmStats>,
    ) -> Self {
        Self { backend, config, receiver, sender, stats }
    }

    /// Run until the request channel closes. Dropping the result receiver does
    /// not stop System 1: results are simply dropped.
    pub async fn run(mut self) {
        while let Some(request) = self.receiver.recv().await {
            let (kind, response, latency_ms) =
                if request.spatial.escalation_score() < self.config.escalation_threshold {
                    (OutcomeKind::Bypassed, None, 0.0_f32)
                } else {
                    let started = Instant::now();
                    let budget = Duration::from_millis(self.config.timeout_ms);
                    let (kind, response) =
                        match tokio::time::timeout(budget, self.backend.propose(&request)).await {
                            Ok(Ok(response)) => (OutcomeKind::Accepted, Some(response)),
                            Ok(Err(VlmError::Malformed(_))) | Ok(Err(VlmError::Schema(_))) => {
                                (OutcomeKind::FallbackMalformed, None)
                            }
                            Ok(Err(VlmError::Unavailable)) | Ok(Err(VlmError::Transport(_))) => {
                                (OutcomeKind::FallbackUnreachable, None)
                            }
                            Ok(Err(VlmError::Timeout(_))) | Err(_) => {
                                (OutcomeKind::FallbackTimeout, None)
                            }
                        };
                    let elapsed_ms = (started.elapsed().as_secs_f64() * 1000.0) as f32;
                    (kind, response, elapsed_ms)
                };
            self.stats.record(kind);
            let proposal = build_proposal(&request, response, &self.config.model, kind, latency_ms);
            let result = VlmResult { proposal, kind, spatial: request.spatial };
            if self.sender.send(result).await.is_err() {
                break;
            }
        }
    }
}

fn encode_png(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(pixels.len().saturating_add(128));
    let mut encoder = png::Encoder::new(&mut out, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(pixels).map_err(|e| e.to_string())?;
    drop(writer);
    Ok(out)
}

pub fn safe_fallback(spatial: &SpatialState, seq: u64) -> VlmProposal {
    let action = if spatial.human_intrusion || spatial.obstacle_intrusion {
        VlmAction::HaltCell {
            reason: "System 2 unavailable; deterministic safe fallback".to_owned(),
        }
    } else {
        VlmAction::SlowConveyor { speed_pct: 20 }
    };
    VlmProposal {
        seq,
        request_id: seq,
        timestamp_ns: spatial.timestamp_ns,
        trace_id: format!("fallback-{seq}"),
        model: "safe-fallback-v1".to_owned(),
        latency_ms: 0.0,
        response: VlmResponse {
            action,
            confidence: 1.0,
            rationale: "System 2 timeout / malformed / unreachable".to_owned(),
        },
    }
}

fn build_proposal(
    request: &VlmRequest,
    response: Option<VlmResponse>,
    model: &str,
    kind: OutcomeKind,
    latency_ms: f32,
) -> VlmProposal {
    match response {
        Some(response) => VlmProposal {
            seq: request.seq,
            request_id: request.seq,
            timestamp_ns: request.timestamp_ns,
            trace_id: format!(
                "vlm-{}-{}",
                request.seq,
                aegis_proto::to_hex(&request.frame_hash[0..4])
            ),
            model: model.to_owned(),
            latency_ms,
            response,
        },
        None if kind == OutcomeKind::Bypassed => {
            VlmProposal::synthetic_continue(request.seq, request.timestamp_ns)
        }
        None => safe_fallback(&request.spatial, request.seq),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegis_proto::{FrameMeta, SpatialState, ZoneClearance, ZoneId};
    use std::time::Instant;

    fn request(score: f32, human: bool) -> VlmRequest {
        let spatial = SpatialState {
            seq: 1,
            timestamp_ns: 1_000_000_000,
            frame_hash: [7; 32],
            zones: vec![ZoneClearance {
                zone: ZoneId::Critical,
                min_mm: if human { 40.0 } else { 800.0 },
                mean_mm: 800.0,
                sample_count: 100,
            }],
            human_intrusion: human,
            obstacle_intrusion: human,
            anomaly_score: score,
            inference_ms: 1.0,
            engine: "test".to_owned(),
        };
        let meta = FrameMeta::new(1, 1_000_000_000, "cam-0", 8, 8, &[0; 192], "nominal");
        VlmRequest {
            seq: 1,
            timestamp_ns: meta.monotonic_ns,
            frame_hash: meta.frame_hash,
            spatial,
            image_rgb: None,
            image_width: 0,
            image_height: 0,
        }
    }

    async fn run_one<B: VlmBackend + Send + 'static>(
        backend: B,
        req: VlmRequest,
        timeout_ms: u64,
    ) -> (VlmResult, VlmCounters) {
        let (tx, rx) = mpsc::channel(4);
        let (out_tx, mut out_rx) = mpsc::channel(4);
        let stats = Arc::new(SharedVlmStats::default());
        let cfg = VlmConfig { timeout_ms, escalation_threshold: 0.45, ..VlmConfig::default() };
        tx.send(req).await.expect("send request");
        drop(tx);
        let worker = VlmWorker::new(backend, cfg, rx, out_tx, stats.clone());
        tokio::spawn(worker.run());
        let result = out_rx.recv().await.expect("result");
        (result, stats.snapshot())
    }

    #[tokio::test]
    async fn bypass_avoids_backend_call() {
        let (result, counters) = run_one(
            MockBackend::valid(VlmAction::HaltCell { reason: "x".to_owned() }, Duration::ZERO),
            request(0.0, false),
            200,
        )
        .await;
        assert_eq!(result.kind, OutcomeKind::Bypassed);
        assert_eq!(counters.bypassed, 1);
        assert_eq!(counters.accepted, 0);
        assert_eq!(result.proposal.response.action, VlmAction::Continue);
    }

    #[tokio::test]
    async fn valid_response_is_accepted() {
        let (result, counters) = run_one(
            MockBackend::valid(VlmAction::DivertPart { bin_id: 3 }, Duration::ZERO),
            request(1.0, false),
            200,
        )
        .await;
        assert_eq!(result.kind, OutcomeKind::Accepted);
        assert_eq!(counters.accepted, 1);
        assert_eq!(result.proposal.response.action, VlmAction::DivertPart { bin_id: 3 });
    }

    #[tokio::test]
    async fn timeout_falls_back_without_blocking() {
        let started = Instant::now();
        let (result, counters) = run_one(
            MockBackend::valid(VlmAction::Continue, Duration::from_secs(5)),
            request(1.0, false),
            30,
        )
        .await;
        assert_eq!(result.kind, OutcomeKind::FallbackTimeout);
        assert_eq!(counters.fallbacks, 1);
        assert_eq!(result.proposal.response.action, VlmAction::SlowConveyor { speed_pct: 20 });
        assert!(started.elapsed() < Duration::from_millis(800));
    }

    #[tokio::test]
    async fn malformed_response_falls_back_to_halt_on_intrusion() {
        let (result, _) = run_one(
            MockBackend::failing(VlmError::Malformed("bad json".to_owned())),
            request(1.0, true),
            200,
        )
        .await;
        assert_eq!(result.kind, OutcomeKind::FallbackMalformed);
        assert!(matches!(result.proposal.response.action, VlmAction::HaltCell { .. }));
    }

    #[test]
    fn png_encoder_accepts_rgb_frames() {
        let pixels = vec![90_u8; 8 * 8 * 3];
        let encoded = encode_png(&pixels, 8, 8).expect("png");
        assert!(encoded.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn fallback_is_safe_and_deterministic() {
        let spatial = request(1.0, false).spatial;
        let a = safe_fallback(&spatial, 42);
        let b = safe_fallback(&spatial, 42);
        assert_eq!(a, b);
        assert_eq!(a.model, "safe-fallback-v1");
    }
}

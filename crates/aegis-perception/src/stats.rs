//! Bounded latency statistics. Allocation happens only when percentiles are
//! requested (control plane), never inside the per-frame hot path.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencyStats {
    samples: Vec<f32>,
    capacity: usize,
    next_evict: usize,
}

impl LatencyStats {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            next_evict: 0,
        }
    }

    /// Constant-time amortized append to a fixed-capacity ring. Oldest sample
    /// is evicted once full; memory never grows past `capacity`.
    pub fn push(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        if self.samples.len() < self.capacity {
            self.samples.push(value);
        } else {
            let slot = self.next_evict.min(self.capacity.saturating_sub(1));
            self.samples[slot] = value;
            self.next_evict = (slot + 1) % self.capacity;
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn mean(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0_f64;
        for s in &self.samples {
            sum += f64::from(*s);
        }
        (sum / self.samples.len() as f64) as f32
    }

    pub fn max(&self) -> f32 {
        let mut max = 0.0_f32;
        for s in &self.samples {
            if *s > max {
                max = *s;
            }
        }
        max
    }

    /// Nearest-rank percentile in `[0, 100]`. Returns 0 for empty stats.
    pub fn percentile(&self, pct: f32) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rank = ((pct.clamp(0.0, 100.0) / 100.0) * sorted.len() as f32).ceil() as usize;
        let idx = rank.saturating_sub(1).min(sorted.len().saturating_sub(1));
        sorted[idx]
    }

    pub fn p50(&self) -> f32 {
        self.percentile(50.0)
    }

    pub fn p95(&self) -> f32 {
        self.percentile(95.0)
    }

    pub fn p99(&self) -> f32 {
        self.percentile(99.0)
    }

    pub fn summary(&self) -> LatencySummary {
        LatencySummary {
            count: self.len() as u64,
            mean_ms: self.mean(),
            p50_ms: self.p50(),
            p95_ms: self.p95(),
            p99_ms: self.p99(),
            max_ms: self.max(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencySummary {
    pub count: u64,
    pub mean_ms: f32,
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    pub max_ms: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_evicts_and_bounds_memory() {
        let mut s = LatencyStats::new(4);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            s.push(v);
        }
        assert_eq!(s.len(), 4);
        assert_eq!(s.max(), 6.0);
        assert_eq!(s.p50(), 4.0);
    }

    #[test]
    fn percentiles_ignore_non_finite_samples() {
        let mut s = LatencyStats::new(8);
        s.push(f32::NAN);
        s.push(3.0);
        assert_eq!(s.len(), 1);
        assert_eq!(s.p99(), 3.0);
    }
}

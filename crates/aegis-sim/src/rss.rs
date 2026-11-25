//! Process RSS measurement for the benchmark control plane.
//!
//! Rust RSS comes from `/proc/self/status`. llama-server RSS is discovered by
//! scanning `/proc/<pid>/comm` and reading that process's `VmRSS` / `VmHWM`.
//! No external tools, samples or escalation are needed.

/// Current and peak resident set sizes in bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RssSample {
    pub rust_rss_bytes: u64,
    pub rust_hwm_bytes: u64,
    pub llama_rss_bytes: u64,
    pub llama_hwm_bytes: u64,
}

pub fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

pub fn snapshot() -> RssSample {
    let (rust_rss, rust_hwm) = process_rss(std::process::id()).unwrap_or((0, 0));
    let mut llama_rss = 0_u64;
    let mut llama_hwm = 0_u64;
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return RssSample {
            rust_rss_bytes: rust_rss,
            rust_hwm_bytes: rust_hwm,
            llama_rss_bytes: 0,
            llama_hwm_bytes: 0,
        };
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid_text) = name.to_str() else { continue };
        if !pid_text.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let comm_path = entry.path().join("comm");
        let Ok(comm) = std::fs::read_to_string(comm_path) else { continue };
        if !comm.trim().eq_ignore_ascii_case("llama-server") {
            continue;
        }
        if let Some((rss, hwm)) = process_rss(pid_text.parse().unwrap_or(0)) {
            llama_rss = llama_rss.saturating_add(rss);
            llama_hwm = llama_hwm.max(hwm);
        }
    }
    RssSample {
        rust_rss_bytes: rust_rss,
        rust_hwm_bytes: rust_hwm,
        llama_rss_bytes: llama_rss,
        llama_hwm_bytes: llama_hwm,
    }
}

fn process_rss(pid: u32) -> Option<(u64, u64)> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    Some(parse_status(&text))
}

fn parse_status(text: &str) -> (u64, u64) {
    let mut rss = 0_u64;
    let mut hwm = 0_u64;
    for line in text.lines() {
        let (key, value) = match line.split_once(':') {
            Some(pair) => pair,
            None => continue,
        };
        let kib = value.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        match key {
            "VmRSS" => rss = kib.saturating_mul(1024),
            "VmHWM" => hwm = kib.saturating_mul(1024),
            _ => {}
        }
    }
    (rss, hwm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_status_fields() {
        let text = "Name:\taegis-cli\nVmRSS:\t  12345 kB\nVmHWM:\t  67890 kB\n";
        assert_eq!(parse_status(text), (12345 * 1024, 67890 * 1024));
    }

    #[test]
    fn snapshot_never_panics_and_converts_units() {
        let s = snapshot();
        assert!(s.rust_hwm_bytes >= s.rust_rss_bytes);
        assert_eq!(mib(1024 * 1024), 1.0);
    }
}

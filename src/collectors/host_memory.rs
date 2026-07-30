//! Host RAM and container cgroup memory cap for kv-offload sizing context.
//! Read once per snapshot assembly; no shell commands.

use std::fs;
use std::path::Path;

use super::types::HostMemoryFacts;

const MEMINFO_PATH: &str = "/proc/meminfo";
const CGROUP_V2_LIMIT: &str = "/sys/fs/cgroup/memory.max";
const CGROUP_V1_LIMIT: &str = "/sys/fs/cgroup/memory/memory.limit_in_bytes";
/// cgroup v1 reports page-rounded i64::MAX when no cap is set (e.g. 9223372036854771712).
const CGROUP_V1_UNLIMITED_MIN: u64 = 1 << 62;

/// Both facts readable, or `None` when either read fails.
pub fn read_host_memory_facts() -> Option<HostMemoryFacts> {
    read_host_memory_facts_from_paths(
        Path::new(MEMINFO_PATH),
        Path::new(CGROUP_V2_LIMIT),
        Path::new(CGROUP_V1_LIMIT),
    )
}

pub(crate) fn read_host_memory_facts_from_paths(
    meminfo_path: &Path,
    cgroup_v2_path: &Path,
    cgroup_v1_path: &Path,
) -> Option<HostMemoryFacts> {
    let available_bytes = parse_mem_available(meminfo_path)?;
    let container_limit_bytes = read_container_limit_bytes(cgroup_v2_path, cgroup_v1_path)?;
    Some(HostMemoryFacts {
        available_bytes,
        container_limit_bytes,
    })
}

fn parse_mem_available(path: &Path) -> Option<u64> {
    let content = fs::read_to_string(path).ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb_str = rest.trim().strip_suffix(" kB")?;
            let kb: u64 = kb_str.parse().ok()?;
            return kb.checked_mul(1024);
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CgroupLimitRead {
    Missing,
    Unlimited,
    Capped(u64),
    Failed,
}

fn read_container_limit_bytes(v2_path: &Path, v1_path: &Path) -> Option<Option<u64>> {
    match read_cgroup_limit_file(v2_path) {
        CgroupLimitRead::Capped(n) => return Some(Some(n)),
        CgroupLimitRead::Unlimited => return Some(None),
        CgroupLimitRead::Failed => return None,
        CgroupLimitRead::Missing => {}
    }
    match read_cgroup_limit_file(v1_path) {
        CgroupLimitRead::Capped(n) => Some(Some(n)),
        CgroupLimitRead::Unlimited | CgroupLimitRead::Missing => Some(None),
        CgroupLimitRead::Failed => None,
    }
}

fn read_cgroup_limit_file(path: &Path) -> CgroupLimitRead {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return CgroupLimitRead::Missing,
        Err(_) => return CgroupLimitRead::Failed,
    };
    let trimmed = content.trim();
    if trimmed.eq_ignore_ascii_case("max") {
        return CgroupLimitRead::Unlimited;
    }
    match trimmed.parse::<u64>() {
        Ok(n) if n >= CGROUP_V1_UNLIMITED_MIN => CgroupLimitRead::Unlimited,
        Ok(n) => CgroupLimitRead::Capped(n),
        Err(_) => CgroupLimitRead::Failed,
    }
}

/// Whole GiB for display (`Host RAM available: N GiB`).
pub(crate) fn bytes_to_display_gib(bytes: u64) -> u64 {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    #[allow(clippy::cast_precision_loss)]
    let gib = bytes as f64 / GIB;
    gib.round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_ID: AtomicU64 = AtomicU64::new(0);

    fn test_dir() -> PathBuf {
        let id = TEST_DIR_ID.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("profile_host_mem_{}_{id}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_meminfo(dir: &std::path::Path, avail_kb: u64) -> std::path::PathBuf {
        let path = dir.join("meminfo");
        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "MemAvailable:     {avail_kb} kB").unwrap();
        path
    }

    fn gib_bytes(gib: u64) -> u64 {
        gib << 30
    }

    #[test]
    fn both_readable_renders_rounded_gib() {
        let dir = test_dir();
        let meminfo = write_meminfo(&dir, gib_bytes(1921) / 1024);
        let v2 = dir.join("memory.max");
        fs::write(&v2, format!("{}", gib_bytes(234))).unwrap();
        let v1 = dir.join("memory.limit_in_bytes");

        let facts = read_host_memory_facts_from_paths(&meminfo, &v2, &v1).expect("both readable");
        assert_eq!(facts.available_bytes, gib_bytes(1921));
        assert_eq!(facts.container_limit_bytes, Some(gib_bytes(234)));
        assert_eq!(bytes_to_display_gib(facts.available_bytes), 1921);
        assert_eq!(
            bytes_to_display_gib(facts.container_limit_bytes.unwrap()),
            234
        );
    }

    #[test]
    fn v2_max_is_unlimited() {
        let dir = test_dir();
        let meminfo = write_meminfo(&dir, gib_bytes(64) / 1024);
        let v2 = dir.join("memory.max");
        fs::write(&v2, "max").unwrap();
        let v1 = dir.join("memory.limit_in_bytes");

        let facts = read_host_memory_facts_from_paths(&meminfo, &v2, &v1).expect("readable");
        assert!(facts.container_limit_bytes.is_none());
    }

    #[test]
    fn v1_fallback_when_v2_missing() {
        let dir = test_dir();
        let meminfo = write_meminfo(&dir, gib_bytes(128) / 1024);
        let v2 = dir.join("memory.max");
        let v1 = dir.join("memory.limit_in_bytes");
        fs::write(&v1, format!("{}", gib_bytes(32))).unwrap();

        let facts = read_host_memory_facts_from_paths(&meminfo, &v2, &v1).expect("readable");
        assert_eq!(facts.container_limit_bytes, Some(gib_bytes(32)));
    }

    #[test]
    fn v1_page_rounded_max_is_unlimited() {
        const V1_UNLIMITED_SENTINEL: u64 = 9_223_372_036_854_771_712;
        let dir = test_dir();
        let meminfo = write_meminfo(&dir, gib_bytes(64) / 1024);
        let v2 = dir.join("memory.max");
        let v1 = dir.join("memory.limit_in_bytes");
        fs::write(&v1, format!("{V1_UNLIMITED_SENTINEL}")).unwrap();

        let facts = read_host_memory_facts_from_paths(&meminfo, &v2, &v1).expect("readable");
        assert!(facts.container_limit_bytes.is_none());
    }

    #[test]
    fn both_cgroup_files_missing_is_unlimited() {
        let dir = test_dir();
        let meminfo = write_meminfo(&dir, gib_bytes(256) / 1024);
        let v2 = dir.join("memory.max");
        let v1 = dir.join("memory.limit_in_bytes");

        let facts = read_host_memory_facts_from_paths(&meminfo, &v2, &v1).expect("readable");
        assert!(facts.container_limit_bytes.is_none());
    }

    #[test]
    fn mem_available_missing_fails_entire_read() {
        let dir = test_dir();
        let meminfo = dir.join("meminfo");
        fs::write(&meminfo, "MemTotal: 1000 kB\n").unwrap();
        let v2 = dir.join("memory.max");
        fs::write(&v2, "max").unwrap();
        let v1 = dir.join("memory.limit_in_bytes");

        assert!(read_host_memory_facts_from_paths(&meminfo, &v2, &v1).is_none());
    }

    #[test]
    fn cgroup_parse_failure_fails_entire_read() {
        let dir = test_dir();
        let meminfo = write_meminfo(&dir, gib_bytes(8) / 1024);
        let v2 = dir.join("memory.max");
        fs::write(&v2, "not-a-number").unwrap();
        let v1 = dir.join("memory.limit_in_bytes");

        assert!(read_host_memory_facts_from_paths(&meminfo, &v2, &v1).is_none());
    }
}

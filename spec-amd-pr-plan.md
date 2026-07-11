# AMD support: PR plan

Linear dependency chain. Each PR is testable on its own.

## PR #115 (current, in flight): AMD foundation

Already in: AMD GPU catalog (11 entries), price catalog (MI210, MI250), CI pipeline optimization (22m to 7m), multi-target Dockerfile, ENTRYPOINT fix.

Add: build libdrm 2.4.123 from source in AMD Dockerfile stage. Root cause: `libdrm_amdgpu_sys` v0.8.16 binds all symbols at runtime via `dlopen`/`dlsym` with `require_all=true`. The vLLM ROCm base image ships libdrm 2.4.113 (Ubuntu 22.04), which is missing `drmSyncobjEventfd` (added in 2.4.116). One missing symbol fails the entire load, causing a panic.

Remove `libdrm2` and `libdrm-amdgpu1` from apt-get (no longer needed once libdrm is built from source). See `spec-libdrm-fix.md` for exact Dockerfile diff.

**What this PR achieves:** AMD containers build in CI, start on RunPod, and profile runs without panicking on library load.

**Validation:** CI build-amd job passes. On RunPod MI300X: `python3 -c "import ctypes; lib = ctypes.CDLL('libdrm.so.2'); lib.drmSyncobjEventfd; print('OK')"` prints OK. `./profile diagnose --duration 2m -v` gets past library loading (will still show phantom GPUs, that's #116).

---

## PR #116: Container device discovery

**Problem:** `libamdgpu_top` scans `/sys/bus/pci/drivers/amdgpu/` which exposes all host GPUs (8 on a typical RunPod MI300X node), not just the ones allocated to the container. The container only has one render device (`/dev/dri/renderD128`). NVIDIA's NVML handles this at the driver level (`NVIDIA_VISIBLE_DEVICES`); AMD has no equivalent.

Consequences of unfixed:
- Profile reports 8 GPUs when only 1 is allocated
- TP inferred as 8 (should be 1)
- vRAM aggregated to 1536GB (should be 192GB)
- Physics baseline inflated (decode ceiling 2650 tok/s instead of ~331 tok/s)
- Cost inflated ($15.99/hr waste instead of ~$2/hr)

**Changes in `src/collectors/gpu/amd.rs`:**

1. Filter device paths to only render nodes that actually exist in `/dev/dri/` before iterating. After `amdgpu_device_paths()` returns, discard any `DevicePath` whose `render` field points to a file that doesn't exist.

2. Wrap `device_path.init()` in `catch_unwind` as a safety net. Currently `libamdgpu_top`'s `get_fd()` panics (with `unwrap_or_else(|err| { panic!(); })`) instead of returning an error when the render device can't be opened. Profile's `else { continue; }` at line 35 never runs because the panic bypasses it. Wrapping in `catch_unwind` makes inaccessible devices skip gracefully.

Both changes are needed: the filter is the primary fix, `catch_unwind` is defense-in-depth against other failure modes in `libamdgpu_top`.

**What this PR achieves:** Profile reports only the GPUs actually allocated to the container. TP, vRAM, physics baseline, and cost are all correct.

**Validation:** On RunPod MI300X (1 GPU allocated): profile reports 1 GPU, TP=1, vRAM 154/192GB.

---

## PR #117: AMD sensor collection

**Problem:** All per-GPU metrics show `power 0W`, `temp 0°C`, `limit 0W`. This was observed even for the real GPU's PCI slot (not just phantom symlinks). Clock speed and mem_util do read correctly.

**Investigation needed:** How `libamdgpu_top`'s `Sensors` struct reads power/temp. Likely paths:
- `hwmon` sysfs interface (may not be exposed in containers, or may require different permissions)
- `gpu_metrics` sysfs file (AMD-specific binary blob, more reliable in containers)
- `ioctl` via the render device fd

Check which path `libamdgpu_top` uses, whether it falls back, and whether the data is available in the container at all. RunPod may not expose hwmon to containers.

If the data is unavailable at the container level, profile should show `-` instead of `0W`/`0°C` (per output rules: "if a metric is unavailable, show `-` and move on").

**What this PR achieves:** Power, temperature, and power limit either display correctly or show `-` when unavailable.

**Validation:** On RunPod MI300X: power, temp, limit show real values or `-`. No fabricated zeros.

---

## PR #118 (if needed): UI and minor observations

Depends on clean output from #116 and #117. May include:
- Formatting adjustments for AMD GPU names (longer than NVIDIA names)
- Per-GPU line layout if different field set from NVIDIA
- Any other cosmetic issues visible once the data is correct

**Validation:** Visual review of output on both NVIDIA and AMD.

---

## Observation: mem_util and clock inconsistency across symlinked GPUs

With 8 symlinks to the same render device, mem_util showed 31%, 8%, 0%, 0%, 0%, 0%, 0%, 11% and clocks showed 1613/1620 MHz variation. This confirms that sysfs sensor reads are per-PCI-slot (reading from different physical host GPUs), while vRAM reads go through the render device fd (same GPU, hence identical 154/192GB). This inconsistency goes away once #116 filters to real devices only, so no separate fix needed.

# WSL2 Setup for VortexSTARK

VortexSTARK's Cairo prover hits `Wsl/Service/E_UNEXPECTED` catastrophic
failure on some WSL2 configurations at `log_n ≥ 24`. This guide documents
the `.wslconfig` values that have been observed to avoid that crash plus
the WSL kernel caveat found on 2026-04-22.

## Minimum `.wslconfig`

Create (or merge into) `C:\Users\<YourUsername>\.wslconfig`:

```ini
[wsl2]
# Match host RAM minus ~8 GB for Windows. At 64 GB host, allocate 56 GB.
# Default (half of host) is often too tight for Cairo proofs at log_n>=24.
memory=56GB

# Swap sized generously — pinned allocations during prove can push memory
# briefly beyond `memory=` and swap prevents an OOM-kill of the VM.
swap=32GB

# Mirrored networking so WSL2 sees the same IP as Windows — needed if you
# plan to reach the Starknet RPC endpoints from inside WSL.
networkingMode=mirrored

# Default processor count is fine; the prover is GPU-bound, not CPU-bound.
```

Apply via `wsl --shutdown` and re-open a WSL terminal.

## GPU Access

No `.wslconfig` GPU settings are required — WSL2 on Windows 11 exposes the
GPU automatically when NVIDIA drivers ≥ 470 are installed. Verify with:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

If `nvidia-smi` is not found inside WSL2, reinstall the Windows-side NVIDIA
driver (the GPU-PV runtime ships with the Windows driver, not a WSL-side
package).

## WSL Kernel Version — important

Observed 2026-04-22 on two otherwise-identical RTX 5090 machines:

| Machine | WSL kernel | log_n=30 Cairo prove |
|---------|------------|----------------------|
| BigDaddy | `5.15.167.4-microsoft-standard-WSL2` | **works**: 19.68s verified |
| GreenDragon | `6.6.87.2-microsoft-standard-WSL2` | **fails**: `cudaMemcpy H2D error 2` on first pinned→device chunk |

### Bisect (2026-04-22 late)

Cairo prove at `log_n=22` and `log_n=23` both complete cleanly on BigDaddy
(13.25s and 29.65s respectively, proofs verify). At `log_n=24` WSL2 dies with
`Wsl/Service/E_UNEXPECTED` — the VM is killed, requiring `wsl --shutdown` to
recover. This happens deterministically, on a freshly-restarted WSL instance,
even immediately after a successful `log_n=23` run in the same session.

So:
- **log_n ≤ 23: works** inside WSL2 on BigDaddy.
- **log_n ≥ 24: crashes WSL2 VM** (not just the prover process).

This is a WSL-level GPU-PV / kernel issue, not fixable from user space.
Workarounds if you need `log_n ≥ 24` measurements:
- Boot native Linux and run the prover outside WSL.
- Downgrade the WSL kernel (potentially back to 5.15.x).
- Allocate more to `.wslconfig` (tried memory=56G + swap=32G — no effect at
  log_n=24; memory pressure isn't the trigger, GPU-PV call is).

The newer WSL kernel (6.6) has a GPU-PV staging-area limit that is hit by
the 8 GB pinned transfer used at log_n=30 — even when 30 GB of VRAM is free.
Chunking the transfer into 512 MB pieces (shipped in `src/device/buffer.rs`)
does **not** fix this — the first chunk alone still fails.

**Recommendations:**
- If you need the headline log_n=30 number: use a WSL kernel in the
  5.15.x line. `wsl --update --target-kernel-version 5.15` if available,
  or freeze your WSL kernel after a successful install.
- For log_n ≤ 29, both kernels work.

## Install CUDA Toolkit in WSL2 Ubuntu-24.04

The WSL2 NVIDIA driver provides the CUDA runtime, but `nvcc` (needed for
build) is not installed by default. Use NVIDIA's official apt repo:

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
sudo dpkg -i /tmp/ck.deb
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-2 build-essential pkg-config
```

Add to your shell init (`~/.bashrc` or `~/.profile`):

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH"
```

## Install Rust Nightly

```bash
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain nightly --profile minimal
```

`rustc 1.97.0-nightly` or newer has been tested; earlier nightlies may
have bugs that affect the build.

## Verify the Build

```bash
cd /mnt/c/Users/<YourUsername>/VortexSTARK   # or wherever you cloned
cargo build --release --bin stark_cli
./target/release/stark_cli bench 20
# Expect: "prove: <1000ms, verified: YES"
```

## Known Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `Wsl/Service/E_UNEXPECTED` mid-prove | WSL VM OOM or GPU-PV driver crash | Increase `memory=`, verify `swap=32GB`, restart with `wsl --shutdown` |
| `cudaMemcpy H2D failed: error 2` on first chunk | WSL kernel 6.6 GPU-PV staging limit | Downgrade to 5.15 kernel; use `log_n ≤ 29` as workaround |
| `nvcc fatal : Host compiler targets unsupported OS` | Trying to build Windows-native instead of WSL2 | Build inside WSL2 Ubuntu-24.04, not Windows PowerShell |
| `error: no matching package found: rayon` | `vendor/` dir stale or missing | Re-sync `vendor/` from a host with stwo-fork access |
| `failed to authenticate when downloading repository` | `.cargo/config.toml` references private stwo-fork and cargo has no creds | Use vendored build path; private dep is pre-vendored for this reason |

## Further Reading

- `PERF_ROADMAP.md` — measured per-phase timings at log_n=20 and 22
- `PROFILING.md` — `VORTEXSTARK_PROFILE=1` quickstart
- `FELT252_DESIGN.md` — scope of the pending Felt252 rework

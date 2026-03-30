# Benchmark Results and Codegen Notes

## AArch64 (Apple M-series, branch `mkitti-no-arch-simd`)

Measured with Criterion (`--warm-up-time 1 --measurement-time 3`),
kernel: `f64_to_u8`, clamp mode, `NearestEven`, N = 64 000 elements.

| Feature flags       | Dispatch path          | Throughput     | Notes |
|---------------------|------------------------|----------------|-------|
| *(default)*         | NEON hand-written       | **3.09 Gelem/s** | `frintn.2d`×8, `fmax/fmin.2d`, `fcvtzu.2d` + `uzp1`/`sqxtn`/`sqxtun` narrowing chain |
| `--features scalar-simd`  | two-pass scalar   | **1.42 Gelem/s** | 2.7× faster than `WithSimd`; NaN scan → hoisted-match round/clamp/narrow loops; `as u32 as u8` trick avoids redundant post-cast clamp |
| `--features no-arch-simd` | `pulp::WithSimd` (Neon, 2 lanes) | **0.51 Gelem/s** | `tmp` round-trip + per-element `v as u8` → ~6 insns/double |

### Key codegen observations (AArch64)

**`scalar-simd` inner loop (NearestEven arm):**
```asm
; auto-vectorized: 2-wide f64 vectors
frintn.2d  v0, v0          ; round-to-nearest-even
fmaxnm.2d  v0, v0, v31    ; clamp lo (0.0)
fminnm.2d  v0, v0, v30    ; clamp hi (255.0)
fcvtzu.2d  v0, v0          ; f64 → u64 (saturates at 0 and 2^64−1)
; narrowing: u64×2 → u32×2 (uzp1), u32×4 → u16×4 (sqxtn), u16×8 → u8×8 (sqxtun)
```
The `as u32 as u8` trick eliminated the redundant `cmp`/`csel` clamp that
the direct `as u8` path generated but did not yet produce the full
`uzp1.4s`/`sqxtn.4h`/`sqxtun.8b` narrowing chain from the hand-written
NEON kernel. LLVM emits `tbl.16b` for byte extraction instead.

**`no-arch-simd` (WithSimd) bottleneck:**
The `tmp: [f64; 8]` round-trip between SIMD registers and the stack is an
aliasing barrier LLVM cannot eliminate, serialising the pipeline.
Additionally, the `GOT` load + atomic feature-flag check in
`pulp::Arch::new().dispatch()` adds per-call overhead on AArch64.

---

## x86_64 — TODO (needs real hardware)

The branch `mkitti-no-arch-simd` is designed for testing on x86_64.
Expected paths to exercise:

| Feature flags             | Expected dispatch        |
|---------------------------|--------------------------|
| *(default, AVX2 CPU)*     | `avx2` module (`pulp::x86::V3`) |
| `--features no-arch-simd` | `pulp::WithSimd` → `x86-64-v3` when compiled with `-C target-cpu=x86-64-v3`, otherwise `Scalar` |
| `--features scalar-simd`  | two-pass scalar auto-vectorized |

### Suggested commands on x86_64 Linux/macOS

```bash
# Default (AVX2 if CPU supports it)
cargo bench -p zarr-cast-value --bench conversions -- f64_to_u8

# WithSimd generic path only (disable AVX2/AVX hand-written kernels)
cargo bench -p zarr-cast-value --features no-arch-simd \
    --bench conversions -- f64_to_u8

# Two-pass scalar auto-vectorized
cargo bench -p zarr-cast-value --features scalar-simd \
    --bench conversions -- f64_to_u8

# Emit assembly for codegen inspection (scalar-simd, AVX2 target-cpu)
RUSTFLAGS="-C target-cpu=x86-64-v3" \
    cargo rustc -p zarr-cast-value --features scalar-simd \
    --lib --release -- --emit=asm 2>/dev/null
# assembly: target/release/deps/zarr_cast_value-*.s
```

> **Note on Rosetta 2 (Apple Silicon):** Compiling with
> `-C target-cpu=x86-64-v3` produces AVX2 instructions that Rosetta
> cannot execute (SIGILL). Use `-C target-cpu=x86-64` (SSE2 baseline)
> or omit the flag for Rosetta testing. Throughput under Rosetta is
> roughly 10–24× lower than native AArch64, so Rosetta numbers are not
> representative of real x86_64 performance.

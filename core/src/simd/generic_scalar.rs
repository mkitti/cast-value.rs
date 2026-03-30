//! Two-pass scalar fallback kernels for codegen comparison.
//!
//! These functions are structurally identical to the [`super::generic`] kernels
//! but replace the [`pulp::WithSimd`] dispatch entirely. The goal is to
//! maximise LLVM auto-vectorization by removing the obstacles the `WithSimd`
//! design introduces:
//!
//! 1. **No `tmp` round-trip.** The original kernel stores each `lanes`-wide
//!    chunk to `tmp`, rounds per-element, loads back, clamps via SIMD, stores
//!    again, then converts element-by-element. Each store/load pair is an
//!    aliasing barrier LLVM cannot see through, so it serialises the pipeline.
//!
//! 2. **Rounding hoisted outside the loop.** With `WithSimd`, `rounding` is a
//!    captured field evaluated inside the generic `with_simd<S>` body. LLVM can
//!    see through the `match` when it mono-specialises `S = Scalar`, but the
//!    branch still sits inside the chunk loop. Hoisting it out gives each arm a
//!    single, unconditional inner loop body.
//!
//! 3. **Two-pass NaN/conversion split.** Scanning for NaN in a dedicated loop
//!    that only does `v.is_nan()` lets LLVM lower it to `fcmeq` comparisons
//!    reduced by `uminv`. The conversion loop that follows contains no branches
//!    at all — LLVM can auto-vectorize the full round → clamp → narrow pipeline.
//!
//! ## Expected codegen on AArch64
//!
//! - NaN scan: `fcmeq.2d` × N/2 + `uminv.16b` + `cbnz` for early exit
//! - Each rounding arm:
//!   ```asm
//!   frintn.2d / frintz.2d / frintp.2d / frintm.2d
//!   fmax.2d   v,  v, #0.0
//!   fmin.2d   v,  v, #255.0
//!   fcvtzs.2d v,  v
//!   uzp1.4s + sqxtn.4h + sqxtun.8b   ; 3-instruction narrowing chain
//!   str  d, [x]
//!   ```
//!   — matching the hand-written NEON kernel's inner loop.

use crate::RoundingMode;

// ---------------------------------------------------------------------------
// f64 → u8
// ---------------------------------------------------------------------------

/// Convert f64 slice to u8 with two-pass scalar auto-vectorizable loops.
///
/// This is the comparison counterpart to [`super::generic::f64_to_u8_clamp`].
/// Both are accessible in the `check_simd` example for assembly inspection.
pub(super) fn f64_to_u8_clamp(
    src: &[f64],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    // Pass 1: NaN scan — no conversion work, just comparisons.
    // LLVM vectorizes this as `fcmeq.2d` pairs followed by `uminv.16b`.
    for &v in src {
        if v.is_nan() {
            return Err(crate::CastError::NanOrInf { value: v });
        }
    }

    // Pass 2: round → clamp → narrow.
    // Hoisting the match outside the loop gives LLVM a branch-free inner body
    // per mode. With values already known to be non-NaN, `max`/`min`/`as u8`
    // can be vectorized as `fmax.2d`, `fmin.2d`, `fcvtzs.2d` + narrowing chain.
    match rounding {
        RoundingMode::NearestEven => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.round_ties_even().clamp(0.0, 255.0) as u32 as u8;
            }
        }
        RoundingMode::TowardsZero => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.trunc().clamp(0.0, 255.0) as u32 as u8;
            }
        }
        RoundingMode::TowardsPositive => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.ceil().clamp(0.0, 255.0) as u32 as u8;
            }
        }
        RoundingMode::TowardsNegative => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.floor().clamp(0.0, 255.0) as u32 as u8;
            }
        }
        RoundingMode::NearestAway => unreachable!(),
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// f64 → i32
// ---------------------------------------------------------------------------

/// Convert f64 slice to i32 with two-pass scalar auto-vectorizable loops.
pub(super) fn f64_to_i32_clamp(
    src: &[f64],
    dst: &mut [i32],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    for &v in src {
        if v.is_nan() {
            return Err(crate::CastError::NanOrInf { value: v });
        }
    }

    let lo = i32::MIN as f64;
    let hi = i32::MAX as f64;

    match rounding {
        RoundingMode::NearestEven => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.round_ties_even().clamp(lo, hi) as i32;
            }
        }
        RoundingMode::TowardsZero => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.trunc().clamp(lo, hi) as i32;
            }
        }
        RoundingMode::TowardsPositive => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.ceil().clamp(lo, hi) as i32;
            }
        }
        RoundingMode::TowardsNegative => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.floor().clamp(lo, hi) as i32;
            }
        }
        RoundingMode::NearestAway => unreachable!(),
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// f32 → u8
// ---------------------------------------------------------------------------

/// Convert f32 slice to u8 with two-pass scalar auto-vectorizable loops.
pub(super) fn f32_to_u8_clamp(
    src: &[f32],
    dst: &mut [u8],
    rounding: RoundingMode,
) -> Result<bool, crate::CastError> {
    for &v in src {
        if v.is_nan() {
            return Err(crate::CastError::NanOrInf { value: v as f64 });
        }
    }

    match rounding {
        RoundingMode::NearestEven => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.round_ties_even().clamp(0.0, 255.0) as u32 as u8;
            }
        }
        RoundingMode::TowardsZero => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.trunc().clamp(0.0, 255.0) as u32 as u8;
            }
        }
        RoundingMode::TowardsPositive => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.ceil().clamp(0.0, 255.0) as u32 as u8;
            }
        }
        RoundingMode::TowardsNegative => {
            for (&s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.floor().clamp(0.0, 255.0) as u32 as u8;
            }
        }
        RoundingMode::NearestAway => unreachable!(),
    }
    Ok(true)
}

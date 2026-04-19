//! `std::simd` fast paths — gated behind the `nightly` feature flag.
//!
//! Uses `f32x8` / `i32x8` (256 bits) as the primary lane width. AVX2
//! maps these to one YMM register; AArch64 NEON lowers to two Q-reg
//! pairs. In every case this is at least as fast as the stable
//! `chunked` path and often measurably quicker because the
//! portable_simd lowering is tighter than what LLVM derives from
//! `for lane in 0..8` on a `[f32; 8]`.
//!
//! Signatures and semantics match `super::scalar`.

use std::simd::cmp::SimdOrd;
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::{f32x8, i32x8, u8x8, Simd, StdFloat};

use super::scalar::cos_table;

/// 8-lane broadcast of a single f32.
#[inline(always)]
fn splat_f(x: f32) -> f32x8 {
    f32x8::splat(x)
}

/// Load an 8-element f32 row from a contiguous slice.
#[inline(always)]
fn load_f(slice: &[f32]) -> f32x8 {
    f32x8::from_slice(slice)
}

/// Inverse DCT — same broadcast-FMA shape as the chunked kernel but
/// expressed directly as `f32x8` FMA chains.
#[inline]
pub fn idct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];

    // Row pass: tmp[y][0..8] = Σ_k cos[k] * broadcast(block[y][k]).
    for y in 0..8 {
        let mut acc = f32x8::splat(0.0);
        let row = &block[y * 8..y * 8 + 8];
        for k in 0..8 {
            let tk = load_f(&t[k]);
            acc = tk.mul_add(splat_f(row[k]), acc);
        }
        acc.copy_to_slice(&mut tmp[y * 8..y * 8 + 8]);
    }

    // Column pass: block[m][0..8] = Σ_k tmp[k][0..8] * t[k][m].
    for m in 0..8 {
        let mut acc = f32x8::splat(0.0);
        for k in 0..8 {
            let row = load_f(&tmp[k * 8..k * 8 + 8]);
            acc = row.mul_add(splat_f(t[k][m]), acc);
        }
        acc.copy_to_slice(&mut block[m * 8..m * 8 + 8]);
    }
}

/// Forward DCT — structurally identical to the IDCT (same cosine matrix
/// under our normalisation).
#[inline]
pub fn fdct8x8(block: &mut [f32; 64]) {
    let t = cos_table();
    let mut tmp = [0.0f32; 64];

    // Row pass: tmp[y][0..8] (indexed by k) = Σ_n cos_col[n] *
    // broadcast(block[y][n]). `cos_col[n]` is the n-th column of t —
    // build it once per outer-loop iteration (hoisted by LLVM across
    // the y loop because `t` is &'static).
    let cos_cols: [f32x8; 8] = std::array::from_fn(|n| {
        let col = [
            t[0][n], t[1][n], t[2][n], t[3][n], t[4][n], t[5][n], t[6][n], t[7][n],
        ];
        f32x8::from_array(col)
    });

    for y in 0..8 {
        let mut acc = f32x8::splat(0.0);
        let row = &block[y * 8..y * 8 + 8];
        for n in 0..8 {
            acc = cos_cols[n].mul_add(splat_f(row[n]), acc);
        }
        acc.copy_to_slice(&mut tmp[y * 8..y * 8 + 8]);
    }

    // Column pass: block[k][0..8] = Σ_n tmp[n][0..8] * t[k][n].
    for k in 0..8 {
        let mut acc = f32x8::splat(0.0);
        for n in 0..8 {
            let row = load_f(&tmp[n * 8..n * 8 + 8]);
            acc = row.mul_add(splat_f(t[k][n]), acc);
        }
        acc.copy_to_slice(&mut block[k * 8..k * 8 + 8]);
    }
}

/// Dequantise 64 coefficients with the H.263 rule in-place.
#[inline]
pub fn dequant_h263(coeffs: &mut [i32; 64], q: i32, q_plus: i32, start: usize) {
    let two_q = q.saturating_mul(2);

    // Scalar-handle any prefix that doesn't align to 8 lanes.
    let mut i = start;
    while i < 64 && i & 7 != 0 {
        dequant_one(&mut coeffs[i], two_q, q_plus);
        i += 1;
    }

    let two_q_v = i32x8::splat(two_q);
    let q_plus_v = i32x8::splat(q_plus);
    let zero = i32x8::splat(0);
    let min_v = i32x8::splat(-2048);
    let max_v = i32x8::splat(2047);

    while i + 8 <= 64 {
        let l: i32x8 = i32x8::from_slice(&coeffs[i..i + 8]);
        // abs(l), then sign-restore:
        //   val = 2Q * |l| + q_plus;  if l < 0 { val = -val };  clamp.
        // Branchless zero-mask: mask = (l != 0); out = where(mask, val, 0).
        let abs = l.abs();
        let mag = two_q_v * abs + q_plus_v;
        let is_neg = l.simd_lt(zero);
        let signed = is_neg.select(-mag, mag);
        let clamped = signed.simd_max(min_v).simd_min(max_v);
        let is_zero = l.simd_eq(zero);
        let out = is_zero.select(zero, clamped);
        out.copy_to_slice(&mut coeffs[i..i + 8]);
        i += 8;
    }

    while i < 64 {
        dequant_one(&mut coeffs[i], two_q, q_plus);
        i += 1;
    }
}

#[inline]
fn dequant_one(slot: &mut i32, two_q: i32, q_plus: i32) {
    let l = *slot;
    if l == 0 {
        return;
    }
    let abs = l.abs();
    let mut val = two_q * abs + q_plus;
    if l < 0 {
        val = -val;
    }
    *slot = val.clamp(-2048, 2047);
}

/// Clip an 8×8 signed tile to `u8` and store into a strided plane.
#[inline]
pub fn clip_block_to_u8(src: &[i32; 64], dst: &mut [u8], dst_off: usize, stride: usize) {
    let min_v = i32x8::splat(0);
    let max_v = i32x8::splat(255);
    for j in 0..8 {
        let row = i32x8::from_slice(&src[j * 8..j * 8 + 8]);
        let clamped = row.simd_max(min_v).simd_min(max_v);
        // Narrow 8×i32 → 8×u8. `cast()` on a non-saturating type would
        // truncate, but we already clamped to 0..=255 so a plain
        // `as u8` is safe.
        let arr = clamped.to_array();
        let packed: [u8; 8] = std::array::from_fn(|i| arr[i] as u8);
        let base = dst_off + j * stride;
        dst[base..base + 8].copy_from_slice(&packed);
    }
}

/// Add predictor + residual, clip to `u8`, store strided.
#[inline]
pub fn add_residual_clip_block(
    pred: &[u8; 64],
    residual: &[i32; 64],
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
) {
    let min_v = i32x8::splat(0);
    let max_v = i32x8::splat(255);
    for j in 0..8 {
        // Load 8 u8 predictors and extend to i32×8.
        let p_bytes = u8x8::from_slice(&pred[j * 8..j * 8 + 8]);
        let p = p_bytes.cast::<i32>();
        let r = i32x8::from_slice(&residual[j * 8..j * 8 + 8]);
        let sum = p + r;
        let clamped = sum.simd_max(min_v).simd_min(max_v);
        let arr = clamped.to_array();
        let packed: [u8; 8] = std::array::from_fn(|i| arr[i] as u8);
        let base = dst_off + j * stride;
        dst[base..base + 8].copy_from_slice(&packed);
    }
}

/// Copy an 8×8 `u8` block into a row-strided destination.
#[inline]
pub fn copy_block_u8(src: &[u8; 64], dst: &mut [u8], dst_off: usize, stride: usize) {
    for j in 0..8 {
        let base = dst_off + j * stride;
        let row = &src[j * 8..j * 8 + 8];
        dst[base..base + 8].copy_from_slice(row);
    }
}

/// Copy a 16×16 luma macroblock. Each row is a 16-byte `copy_from_slice`
/// which the compiler lowers to a 128-bit vector load+store.
#[inline]
pub fn copy_mb_luma(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    for j in 0..16 {
        let s = src_off + j * src_stride;
        let d = dst_off + j * dst_stride;
        dst[d..d + 16].copy_from_slice(&src[s..s + 16]);
    }
}

/// Copy an 8×8 chroma block.
#[inline]
pub fn copy_mb_chroma(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    for j in 0..8 {
        let s = src_off + j * src_stride;
        let d = dst_off + j * dst_stride;
        dst[d..d + 8].copy_from_slice(&src[s..s + 8]);
    }
}

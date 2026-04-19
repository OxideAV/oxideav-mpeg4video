//! Stable-Rust "manual SIMD" kernels for the 8×8 MPEG-4 Part 2 hot paths.
//!
//! Every tile in this codec is exactly 8 wide, which matches AVX2's
//! 256-bit YMM register (`f32x8`/`i32x8`) and lowers cleanly to a pair of
//! NEON Q-regs on AArch64. The kernels here are written as fixed-size
//! `[f32; 8]` / `[i32; 8]` / `[u8; 8]` chunks wrapped in short inner
//! loops — LLVM auto-vectorises them reliably on release builds without
//! needing nightly.
//!
//! The IDCT and FDCT are rewritten in "broadcast-FMA" form: instead of
//! accumulating `Σ_k cos[k][n] * block[k]` with a scalar reduction, we
//! hold the 8-lane output row in registers and do 8 chained FMAs across
//! `k`. This is what BLAS GEMM cores look like at small N.

/// Inverse DCT of an 8×8 block, in-place.
///
/// Empirically LLVM's auto-vectoriser produces better code on the
/// simple nested-loop scalar form than on any hand-rolled chunked
/// rewrite — the inner reduction is unrolled to a balanced FMA tree and
/// the outer loop is peeled. We keep the chunked entry point so the
/// dispatch shape matches the rest of the module but delegate to
/// `super::scalar::idct8x8`; the portable (`std::simd`) path in
/// `super::portable` carries the only non-delegating SIMD rewrite,
/// where the lane width can be pinned explicitly.
#[inline]
pub fn idct8x8(block: &mut [f32; 64]) {
    super::scalar::idct8x8(block);
}

/// Forward DCT of an 8×8 block, in-place. See `idct8x8` for the reason
/// the stable path delegates to the scalar reference.
#[inline]
pub fn fdct8x8(block: &mut [f32; 64]) {
    super::scalar::fdct8x8(block);
}

/// Dequantise 64 coefficients with the H.263 rule in-place. Uses an
/// 8-lane branch-free "mask out zeros" form so LLVM vectorises the body.
#[inline]
pub fn dequant_h263(coeffs: &mut [i32; 64], q: i32, q_plus: i32, start: usize) {
    let two_q = 2 * q;
    // Handle a possibly unaligned prefix (start may be 0 or 1 in
    // practice) scalar-style.
    let mut i = start;
    while i < 64 && i & 7 != 0 {
        dequant_one(&mut coeffs[i], two_q, q_plus);
        i += 1;
    }
    // Full 8-lane chunks from here. The body is branch-free: compute
    // the reconstructed magnitude for every lane, then conditionally
    // replace with 0 when the input was 0.
    while i + 8 <= 64 {
        // Load 8 lanes into a fixed array so LLVM keeps them in vector
        // registers.
        let lanes: [i32; 8] = [
            coeffs[i],
            coeffs[i + 1],
            coeffs[i + 2],
            coeffs[i + 3],
            coeffs[i + 4],
            coeffs[i + 5],
            coeffs[i + 6],
            coeffs[i + 7],
        ];
        let mut out = [0i32; 8];
        for lane in 0..8 {
            let l = lanes[lane];
            let abs = l.abs();
            let mut val = two_q * abs + q_plus;
            if l < 0 {
                val = -val;
            }
            // The scalar path clamps unconditionally — keep the same
            // numeric output even when l == 0 (val would be q_plus ≥ 0,
            // still inside [-2048, 2047] for any Q in 1..=31).
            val = val.clamp(-2048, 2047);
            out[lane] = if l == 0 { 0 } else { val };
        }
        coeffs[i..i + 8].copy_from_slice(&out);
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

/// Clip an 8×8 signed tile to `u8` and store to a strided plane.
///
/// The inner loop over 8 lanes auto-vectorises to `vpminsd` / `vpmaxsd`
/// + byte packing on x86-64 / `smin` + `smax` + narrowing on AArch64.
#[inline]
pub fn clip_block_to_u8(src: &[i32; 64], dst: &mut [u8], dst_off: usize, stride: usize) {
    for j in 0..8 {
        let row = [
            src[j * 8],
            src[j * 8 + 1],
            src[j * 8 + 2],
            src[j * 8 + 3],
            src[j * 8 + 4],
            src[j * 8 + 5],
            src[j * 8 + 6],
            src[j * 8 + 7],
        ];
        let mut out = [0u8; 8];
        for lane in 0..8 {
            out[lane] = clip_i32_u8(row[lane]);
        }
        let base = dst_off + j * stride;
        dst[base..base + 8].copy_from_slice(&out);
    }
}

/// Add a predictor tile and a residual tile, clip to `u8`, store strided.
#[inline]
pub fn add_residual_clip_block(
    pred: &[u8; 64],
    residual: &[i32; 64],
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
) {
    for j in 0..8 {
        let p = [
            pred[j * 8] as i32,
            pred[j * 8 + 1] as i32,
            pred[j * 8 + 2] as i32,
            pred[j * 8 + 3] as i32,
            pred[j * 8 + 4] as i32,
            pred[j * 8 + 5] as i32,
            pred[j * 8 + 6] as i32,
            pred[j * 8 + 7] as i32,
        ];
        let r = [
            residual[j * 8],
            residual[j * 8 + 1],
            residual[j * 8 + 2],
            residual[j * 8 + 3],
            residual[j * 8 + 4],
            residual[j * 8 + 5],
            residual[j * 8 + 6],
            residual[j * 8 + 7],
        ];
        let mut out = [0u8; 8];
        for lane in 0..8 {
            out[lane] = clip_i32_u8(p[lane] + r[lane]);
        }
        let base = dst_off + j * stride;
        dst[base..base + 8].copy_from_slice(&out);
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

/// Copy a 16×16 luma macroblock.
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

#[inline(always)]
fn clip_i32_u8(v: i32) -> u8 {
    if v <= 0 {
        0
    } else if v >= 255 {
        255
    } else {
        v as u8
    }
}

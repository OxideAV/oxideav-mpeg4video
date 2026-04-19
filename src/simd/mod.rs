//! SIMD fast paths for the MPEG-4 Part 2 hot spots.
//!
//! The decoder and encoder both spend the majority of their time in four
//! kernels, each running once per 8×8 block (so 6× per macroblock, ~1500×
//! per CIF frame, ~6000× per SD frame):
//!
//! * [`idct8x8`] / [`fdct8x8`] — row-then-column cosine matrix-multiply
//!   with an 8-lane inner sum. Natural SIMD width: `f32x8`.
//! * [`dequant_h263`] — linear per-coefficient rescale with skip-on-zero.
//! * [`add_residual_clip_block`] / [`clip_block_to_u8`] — residual add
//!   and `u8` saturation on 8×8 tiles at the end of each block decode.
//! * [`copy_mb_luma`] / [`copy_mb_chroma`] — skipped-macroblock plane
//!   copies; small but called on every skipped MB.
//!
//! Layout (matches `oxideav-vorbis`):
//!
//! * [`scalar`] — reference implementations, always compiled, used as the
//!   oracle for the bit-exactness tests.
//! * [`chunked`] — stable-Rust "manual SIMD" built from fixed-size
//!   `[f32; 8]` / `[i32; 8]` chunks that LLVM lowers to AVX2 / NEON / SSE
//!   on release builds. This is the default path on stable.
//! * [`portable`] — `std::simd` paths behind the `nightly` feature flag.
//!
//! Public entry points dispatch at compile time via `cfg`. The chunked
//! and portable implementations must agree with the scalar path up to
//! FP-reassociation (validated by `tests::*_matches_scalar`).

pub mod chunked;
pub mod scalar;

#[cfg(feature = "nightly")]
pub mod portable;

/// Inverse DCT of an 8×8 natural-order block, in-place.
#[inline]
pub fn idct8x8(block: &mut [f32; 64]) {
    #[cfg(feature = "nightly")]
    {
        portable::idct8x8(block);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::idct8x8(block);
    }
}

/// Forward DCT of an 8×8 natural-order block, in-place.
///
/// Uses the same cosine basis as [`idct8x8`] so that `idct(fdct(x)) == x`
/// to within float rounding.
#[inline]
pub fn fdct8x8(block: &mut [f32; 64]) {
    #[cfg(feature = "nightly")]
    {
        portable::fdct8x8(block);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::fdct8x8(block);
    }
}

/// Dequantise 64 coefficients with the H.263 rule in-place:
/// `l != 0 → sign(l) * (2*Q*|l| + Q_plus)`, then clamp to [-2048, 2047].
/// `start` is 0 for inter (all 64), 1 for intra (DC handled separately).
#[inline]
pub fn dequant_h263(coeffs: &mut [i32; 64], q: i32, q_plus: i32, start: usize) {
    #[cfg(feature = "nightly")]
    {
        portable::dequant_h263(coeffs, q, q_plus, start);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::dequant_h263(coeffs, q, q_plus, start);
    }
}

/// Clip an 8×8 signed sample tile to `[0, 255]` and store as `u8` into
/// `dst` at row stride `stride`, starting at offset `dst_off`.
#[inline]
pub fn clip_block_to_u8(src: &[i32; 64], dst: &mut [u8], dst_off: usize, stride: usize) {
    #[cfg(feature = "nightly")]
    {
        portable::clip_block_to_u8(src, dst, dst_off, stride);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::clip_block_to_u8(src, dst, dst_off, stride);
    }
}

/// Add an 8×8 residual tile to an 8×8 predictor tile, clip to `[0, 255]`
/// and store as `u8` into `dst` at row stride `stride`, starting at
/// offset `dst_off`. Equivalent to:
/// ```text
/// for j in 0..8 { for i in 0..8 {
///     dst[dst_off + j*stride + i] = clamp(pred[j*8+i] + residual[j*8+i], 0, 255)
/// } }
/// ```
#[inline]
pub fn add_residual_clip_block(
    pred: &[u8; 64],
    residual: &[i32; 64],
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
) {
    #[cfg(feature = "nightly")]
    {
        portable::add_residual_clip_block(pred, residual, dst, dst_off, stride);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::add_residual_clip_block(pred, residual, dst, dst_off, stride);
    }
}

/// Copy an 8×8 `u8` block into a row-strided destination.
#[inline]
pub fn copy_block_u8(
    src: &[u8; 64],
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
) {
    #[cfg(feature = "nightly")]
    {
        portable::copy_block_u8(src, dst, dst_off, stride);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::copy_block_u8(src, dst, dst_off, stride);
    }
}

/// Copy a 16×16 luma macroblock from one strided plane to another.
#[inline]
pub fn copy_mb_luma(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    #[cfg(feature = "nightly")]
    {
        portable::copy_mb_luma(src, src_off, src_stride, dst, dst_off, dst_stride);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::copy_mb_luma(src, src_off, src_stride, dst, dst_off, dst_stride);
    }
}

/// Copy an 8×8 chroma block from one strided plane to another.
#[inline]
pub fn copy_mb_chroma(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
) {
    #[cfg(feature = "nightly")]
    {
        portable::copy_mb_chroma(src, src_off, src_stride, dst, dst_off, dst_stride);
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::copy_mb_chroma(src, src_off, src_stride, dst, dst_off, dst_stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_dct_block(seed: u32) -> [f32; 64] {
        let mut b = [0.0f32; 64];
        b[0] = 800.0 + (seed as f32 * 0.37) % 200.0;
        for i in 1..16 {
            let v = (((seed.wrapping_mul(i as u32 + 1)) % 401) as i32 - 200) as f32;
            b[(i * 3) % 64] = v;
        }
        b
    }

    fn mk_sample_block(seed: u32) -> [f32; 64] {
        let mut b = [0.0f32; 64];
        for j in 0..8 {
            for i in 0..8 {
                let noise =
                    ((seed.wrapping_mul(17 + i as u32).wrapping_add(j as u32 * 13)) % 16) as f32;
                b[j * 8 + i] = 64.0 + (i + j) as f32 * 10.0 + noise;
            }
        }
        b
    }

    fn approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= eps)
    }

    #[test]
    fn idct_matches_scalar() {
        for seed in [1u32, 42, 7777, 0xdeadbeef] {
            let mut a = mk_dct_block(seed);
            let mut b = a;
            scalar::idct8x8(&mut a);
            idct8x8(&mut b);
            assert!(approx_eq(&a, &b, 1e-2), "seed {seed}: idct mismatch");
        }
    }

    #[test]
    fn fdct_matches_scalar() {
        for seed in [1u32, 42, 7777, 0xdeadbeef] {
            let mut a = mk_sample_block(seed);
            let mut b = a;
            scalar::fdct8x8(&mut a);
            fdct8x8(&mut b);
            assert!(approx_eq(&a, &b, 1e-2), "seed {seed}: fdct mismatch");
        }
    }

    #[test]
    fn fdct_idct_round_trip() {
        let original = mk_sample_block(99);
        let mut b = original;
        fdct8x8(&mut b);
        idct8x8(&mut b);
        assert!(approx_eq(&b, &original, 1e-2), "round-trip failed");
    }

    #[test]
    fn dequant_matches_scalar() {
        // Mixed zero / positive / negative coefficients.
        let mut coeffs = [0i32; 64];
        for i in 0..64 {
            coeffs[i] = (((i as i32) * 17 + 3) % 41) - 20; // in -20..=20
        }
        // Force DC to a value that exists too.
        coeffs[0] = 31;

        for &(q, start) in &[(1i32, 0usize), (4, 0), (5, 0), (12, 1), (31, 1)] {
            let q_plus = if q & 1 == 1 { q } else { q - 1 };
            let mut a = coeffs;
            let mut b = coeffs;
            scalar::dequant_h263(&mut a, q, q_plus, start);
            dequant_h263(&mut b, q, q_plus, start);
            assert_eq!(a, b, "q={q} start={start}");
        }
    }

    #[test]
    fn clip_block_matches_scalar() {
        // dst layout: 16×16 plane; 8×8 block written at offset (2,1).
        let src: [i32; 64] = std::array::from_fn(|i| (i as i32 * 7) - 200);
        let mut a = [0u8; 16 * 16];
        let mut b = [0u8; 16 * 16];
        let off = 16 + 2;
        scalar::clip_block_to_u8(&src, &mut a, off, 16);
        clip_block_to_u8(&src, &mut b, off, 16);
        assert_eq!(a, b);
    }

    #[test]
    fn add_residual_matches_scalar() {
        let pred: [u8; 64] = std::array::from_fn(|i| ((i * 3) % 256) as u8);
        let residual: [i32; 64] = std::array::from_fn(|i| (i as i32 * 5) - 150);
        let mut a = [0u8; 16 * 16];
        let mut b = [0u8; 16 * 16];
        let off = 16 + 2;
        scalar::add_residual_clip_block(&pred, &residual, &mut a, off, 16);
        add_residual_clip_block(&pred, &residual, &mut b, off, 16);
        assert_eq!(a, b);
    }

    #[test]
    fn copy_block_matches_scalar() {
        let src: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(3));
        let mut a = [0u8; 16 * 16];
        let mut b = [0u8; 16 * 16];
        let off = 16 + 2;
        scalar::copy_block_u8(&src, &mut a, off, 16);
        copy_block_u8(&src, &mut b, off, 16);
        assert_eq!(a, b);
    }

    #[test]
    fn copy_mb_luma_matches_scalar() {
        // 32×32 source, 32×32 dest. 16×16 copy offsets at (4,4)→(8,8).
        let src: [u8; 32 * 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(5));
        let mut a = [0u8; 32 * 32];
        let mut b = [0u8; 32 * 32];
        scalar::copy_mb_luma(&src, 4 * 32 + 4, 32, &mut a, 8 * 32 + 8, 32);
        copy_mb_luma(&src, 4 * 32 + 4, 32, &mut b, 8 * 32 + 8, 32);
        assert_eq!(a, b);
    }

    #[test]
    fn copy_mb_chroma_matches_scalar() {
        let src: [u8; 16 * 16] = std::array::from_fn(|i| (i as u8).wrapping_mul(7));
        let mut a = [0u8; 16 * 16];
        let mut b = [0u8; 16 * 16];
        scalar::copy_mb_chroma(&src, 2 * 16 + 2, 16, &mut a, 4 * 16 + 4, 16);
        copy_mb_chroma(&src, 2 * 16 + 2, 16, &mut b, 4 * 16 + 4, 16);
        assert_eq!(a, b);
    }
}

//! §7.3 VOP reconstruction — the per-pixel
//! `d[y][x] = p[y][x] + f[y][x]` step-2 sum plus the step-3
//! `[0, 2^bits_per_pixel - 1]` display-range saturation that closes the
//! decoder pipeline.
//!
//! Each prior round produced one half of the equation:
//!
//! * The §7.4.x texture chain (rounds 9..15 for intra, round 21 for inter)
//!   delivers `f[y][x]` — either the reconstructed sample (intra) or the
//!   signed residual (inter), in the §7.4.5 saturation range
//!   `[-2^bpp, 2^bpp - 1]`.
//! * The §7.6 motion-compensation chain (rounds 19, 22..30) delivers
//!   `p[y][x]` — the §7.6.2.1 half-sample / §7.6.2.2 quarter-sample
//!   interpolated prediction macroblock from one or two anchor VOPs,
//!   in the display range `[0, 2^bpp - 1]`.
//!
//! §7.3 of ISO/IEC 14496-2:2004 specifies the three reconstruction
//! steps verbatim:
//!
//! 1. **Intra macroblocks.** "The luminance and chrominance values
//!    `f[y][x]` from the decoded texture data form the luminance and
//!    chrominance values of the VOP: `d[y][x] = f[y][x]`." Intra blocks
//!    carry no `p[y][x]`; the texture chain's output is the sample.
//! 2. **Inter macroblocks.** "First the prediction values `p[y][x]` are
//!    calculated using the decoded motion vector information and the
//!    texture information of the respective reference VOPs. Then, the
//!    decoded texture data `f[y][x]` is added to the prediction values,
//!    resulting in the final luminance and chrominance values of the
//!    VOP: `d[y][x] = p[y][x] + f[y][x]`."
//! 3. **Final clip.** "Finally, the calculated luminance and chrominance
//!    values of the reconstructed VOP are saturated so that
//!    `d[y][x] = 2^bits_per_pixel - 1` when `d[y][x] > 2^bits_per_pixel
//!    - 1`, `d[y][x]` when `0 <= d[y][x] <= 2^bits_per_pixel - 1`, and
//!    `0` when `d[y][x] < 0`."
//!
//! The §7.3 step-3 clip is the same shape as the §7.4.5 saturation
//! used inside the IDCT, but anchored at `[0, 2^bpp - 1]` (the display
//! range) rather than `[-2^bpp, 2^bpp - 1]` (the signed-residual
//! range). [`clip_display_sample`] is the per-sample primitive.
//!
//! ## Block / macroblock shapes
//!
//! The four entry points operate at three granularities:
//!
//! * [`reconstruct_inter_block_8x8`] / [`reconstruct_inter_block_8x8_into`]
//!   — one 8×8 block, the per-§6.2.7 `block(i)` shape. Useful as a
//!   primitive for callers that already iterate blocks themselves
//!   (e.g. a future encoder rate-distortion loop).
//! * [`reconstruct_inter_macroblock`] /
//!   [`reconstruct_inter_macroblock_into`] — one 4:2:0 inter
//!   macroblock: a 16×16 luminance plane plus 8×8 Cb / Cr planes,
//!   matching the [`crate::block::InterMacroblock`] residual shape
//!   produced by [`crate::decode_inter_macroblock`] and the
//!   [`crate::generate_b_vop_luma_prediction`] /
//!   [`crate::bvop_prediction::generate_b_vop_chroma_prediction`]
//!   prediction shape produced by the round-26 / round-29 B-VOP
//!   prediction module.
//!
//! Intra macroblocks bypass the predictor and clip to the display
//! range directly. The §7.4.5 IDCT already clamps to
//! `[-2^bpp, 2^bpp - 1]`, so the §7.3 step-3 clip is the only
//! remaining transformation that applies to an intra-block sample.
//! [`reconstruct_intra_block_8x8`] / [`reconstruct_intra_macroblock`]
//! make the no-predictor case explicit.

#![allow(clippy::needless_range_loop)]

use crate::block::{InterMacroblock, IntraMacroblock};

/// Side length of an 8×8 block in samples.
pub const BLOCK_SIDE: usize = 8;

/// Side length of a 4:2:0 luminance macroblock in samples.
pub const MACROBLOCK_LUMA_SIDE: usize = 16;

/// Side length of a 4:2:0 chrominance macroblock in samples (Cb / Cr).
pub const MACROBLOCK_CHROMA_SIDE: usize = 8;

/// §7.3 step-3 per-sample display-range clip.
///
/// Saturates `value` to the display range `[0, 2^bits_per_pixel - 1]`,
/// returning `0` for negative inputs and `2^bits_per_pixel - 1` for
/// inputs at or above the upper bound.
///
/// # Panics
///
/// Panics when `bits_per_pixel >= 31` (the upper bound `2^bpp - 1`
/// would overflow `i32`). The §6.3.3 `bits_per_pixel` field is at most
/// 12 in practice.
#[inline]
pub fn clip_display_sample(value: i32, bits_per_pixel: u32) -> i32 {
    assert!(
        bits_per_pixel < 31,
        "bits_per_pixel = {bits_per_pixel} is out of range"
    );
    let hi = (1i32 << bits_per_pixel) - 1;
    value.clamp(0, hi)
}

/// §7.3 step-2 + step-3 reconstruction of one 8×8 inter block.
///
/// `prediction[y][x]` is the motion-compensated `p[y][x]` from §7.6
/// (range `[0, 2^bpp - 1]`); `residual[y][x]` is the §7.4.5-saturated
/// `f[y][x]` from [`crate::decode_inter_block`] (range
/// `[-2^bpp, 2^bpp - 1]`). The returned block holds `d[y][x] =
/// clip(p[y][x] + f[y][x], 0, 2^bpp - 1)`.
///
/// Arithmetic uses `i32` so the worst-case sum
/// `(2^bpp - 1) + (2^bpp - 1) = 2^(bpp + 1) - 2` stays well inside the
/// type for any practical `bits_per_pixel`. The clip catches both the
/// over-range and under-range cases without further sign analysis.
#[inline]
pub fn reconstruct_inter_block_8x8(
    prediction: &[[i32; BLOCK_SIDE]; BLOCK_SIDE],
    residual: &[[i32; BLOCK_SIDE]; BLOCK_SIDE],
    bits_per_pixel: u32,
) -> [[i32; BLOCK_SIDE]; BLOCK_SIDE] {
    let mut out = [[0i32; BLOCK_SIDE]; BLOCK_SIDE];
    reconstruct_inter_block_8x8_into(prediction, residual, bits_per_pixel, &mut out);
    out
}

/// In-place variant of [`reconstruct_inter_block_8x8`].
///
/// Writes the reconstructed `d[y][x]` to `out[y][x]`. Useful when
/// reconstructing a macroblock by iterating over its four luma
/// sub-blocks into a 16×16 destination buffer one block at a time.
#[inline]
pub fn reconstruct_inter_block_8x8_into(
    prediction: &[[i32; BLOCK_SIDE]; BLOCK_SIDE],
    residual: &[[i32; BLOCK_SIDE]; BLOCK_SIDE],
    bits_per_pixel: u32,
    out: &mut [[i32; BLOCK_SIDE]; BLOCK_SIDE],
) {
    let hi = (1i32 << bits_per_pixel) - 1;
    for y in 0..BLOCK_SIDE {
        for x in 0..BLOCK_SIDE {
            // §7.3 step-2: d[y][x] = p[y][x] + f[y][x].
            let d = prediction[y][x] + residual[y][x];
            // §7.3 step-3: saturate to [0, 2^bpp - 1].
            out[y][x] = d.clamp(0, hi);
        }
    }
}

/// §7.3 step-1 + step-3 reconstruction of one 8×8 intra block.
///
/// The §7.3 step-1 branch sets `d[y][x] = f[y][x]` (no prediction add);
/// the §7.3 step-3 clip then bounds the value to the display range.
/// `decode_intra_block` already applies the display clip internally,
/// so this entry point is the per-sample identity-plus-clip for
/// callers that prefer to keep the §7.3 step-3 invariant explicit.
#[inline]
pub fn reconstruct_intra_block_8x8(
    sample: &[[i32; BLOCK_SIDE]; BLOCK_SIDE],
    bits_per_pixel: u32,
) -> [[i32; BLOCK_SIDE]; BLOCK_SIDE] {
    let hi = (1i32 << bits_per_pixel) - 1;
    let mut out = [[0i32; BLOCK_SIDE]; BLOCK_SIDE];
    for y in 0..BLOCK_SIDE {
        for x in 0..BLOCK_SIDE {
            out[y][x] = sample[y][x].clamp(0, hi);
        }
    }
    out
}

/// §7.3 step-2 + step-3 reconstruction of one 4:2:0 inter macroblock.
///
/// `prediction.luma` is the motion-compensated 16×16 luminance plane
/// produced by §7.6.2.1 / §7.6.2.2 interpolation (one of
/// [`crate::half_sample::interpolate_block_into`],
/// [`crate::quarter_sample::interpolate_block_qpel_into`], or
/// [`crate::generate_b_vop_luma_prediction`]); `prediction.cb` /
/// `prediction.cr` are the matching 8×8 chrominance planes (the §7.6.5
/// chroma-MV-derived [`crate::half_sample::interpolate_block_into`]
/// outputs). `residual` is the [`InterMacroblock`] returned by
/// [`crate::decode_inter_macroblock`]. The returned macroblock holds
/// the §7.3 step-3-clipped `d[y][x]` for each of the three planes,
/// ready to be written into the VOP frame buffer.
#[inline]
pub fn reconstruct_inter_macroblock(
    prediction: &InterPredictionMacroblock,
    residual: &InterMacroblock,
    bits_per_pixel: u32,
) -> ReconstructedMacroblock {
    let mut out = ReconstructedMacroblock {
        luma: [[0i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
        cb: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        cr: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
    };
    reconstruct_inter_macroblock_into(prediction, residual, bits_per_pixel, &mut out);
    out
}

/// In-place variant of [`reconstruct_inter_macroblock`].
pub fn reconstruct_inter_macroblock_into(
    prediction: &InterPredictionMacroblock,
    residual: &InterMacroblock,
    bits_per_pixel: u32,
    out: &mut ReconstructedMacroblock,
) {
    let hi = (1i32 << bits_per_pixel) - 1;
    for y in 0..MACROBLOCK_LUMA_SIDE {
        for x in 0..MACROBLOCK_LUMA_SIDE {
            let d = prediction.luma[y][x] + residual.luma[y][x];
            out.luma[y][x] = d.clamp(0, hi);
        }
    }
    for y in 0..MACROBLOCK_CHROMA_SIDE {
        for x in 0..MACROBLOCK_CHROMA_SIDE {
            let d_cb = prediction.cb[y][x] + residual.cb[y][x];
            out.cb[y][x] = d_cb.clamp(0, hi);
            let d_cr = prediction.cr[y][x] + residual.cr[y][x];
            out.cr[y][x] = d_cr.clamp(0, hi);
        }
    }
}

/// §7.3 step-1 + step-3 reconstruction of one 4:2:0 intra macroblock.
///
/// Step-1 sets `d[y][x] = f[y][x]` plane-by-plane; step-3 clips to
/// `[0, 2^bpp - 1]`. The clip is a no-op against
/// [`crate::decode_intra_macroblock`]'s output (which already clamps),
/// but the entry point exposes the §7.3 invariant for the intra path
/// alongside the inter path.
#[inline]
pub fn reconstruct_intra_macroblock(
    sample: &IntraMacroblock,
    bits_per_pixel: u32,
) -> ReconstructedMacroblock {
    let hi = (1i32 << bits_per_pixel) - 1;
    let mut out = ReconstructedMacroblock {
        luma: [[0i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
        cb: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        cr: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
    };
    for y in 0..MACROBLOCK_LUMA_SIDE {
        for x in 0..MACROBLOCK_LUMA_SIDE {
            out.luma[y][x] = sample.luma[y][x].clamp(0, hi);
        }
    }
    for y in 0..MACROBLOCK_CHROMA_SIDE {
        for x in 0..MACROBLOCK_CHROMA_SIDE {
            out.cb[y][x] = sample.cb[y][x].clamp(0, hi);
            out.cr[y][x] = sample.cr[y][x].clamp(0, hi);
        }
    }
    out
}

/// A §7.3 step-2 motion-compensated prediction macroblock — the
/// `p[y][x]` input to [`reconstruct_inter_macroblock`].
///
/// The luminance plane is 16×16, the two chrominance planes 8×8 each
/// per §6.1.3.4 4:2:0 sampling. Each sample is in the display range
/// `[0, 2^bpp - 1]`. The §7.6.2.1 half-sample / §7.6.2.2 quarter-
/// sample interpolation routines write directly into this shape (the
/// caller assembles four 8×8 sub-blocks into the 16×16 luma plane and
/// runs one 8×8 interpolation per chroma component).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterPredictionMacroblock {
    /// `p[y][x]` luminance samples, `luma[row][col]`, 16×16.
    pub luma: [[i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
    /// `p[y][x]` Cb samples, `cb[row][col]`, 8×8.
    pub cb: [[i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
    /// `p[y][x]` Cr samples, `cr[row][col]`, 8×8.
    pub cr: [[i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
}

impl InterPredictionMacroblock {
    /// Construct an all-zero prediction macroblock — the §7.6.9.6 / §7.6
    /// trivial case where the prediction contributes no signal (e.g. a
    /// boundary-substitution fallback). Adding this prediction to a
    /// residual reduces the §7.3 step-2 sum to the residual itself,
    /// which then takes the §7.3 step-3 clip.
    pub fn zero() -> Self {
        Self {
            luma: [[0i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        }
    }
}

/// One fully-reconstructed 4:2:0 macroblock — the `d[y][x]` output of
/// the §7.3 pipeline, ready to be blitted into the VOP frame buffer.
///
/// Every sample is in the §7.3 step-3 display range `[0, 2^bpp - 1]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructedMacroblock {
    /// `d[y][x]` luminance samples, `luma[row][col]`, 16×16.
    pub luma: [[i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
    /// `d[y][x]` Cb samples, `cb[row][col]`, 8×8.
    pub cb: [[i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
    /// `d[y][x]` Cr samples, `cr[row][col]`, 8×8.
    pub cr: [[i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_block(value: i32) -> [[i32; BLOCK_SIDE]; BLOCK_SIDE] {
        [[value; BLOCK_SIDE]; BLOCK_SIDE]
    }

    #[test]
    fn clip_display_sample_in_range_passthrough() {
        // §7.3 step-3 middle branch: 0 <= d <= 2^bpp - 1.
        assert_eq!(clip_display_sample(0, 8), 0);
        assert_eq!(clip_display_sample(128, 8), 128);
        assert_eq!(clip_display_sample(255, 8), 255);
    }

    #[test]
    fn clip_display_sample_upper_saturates() {
        // §7.3 step-3 top branch: d > 2^bpp - 1 saturates to 2^bpp - 1.
        assert_eq!(clip_display_sample(256, 8), 255);
        assert_eq!(clip_display_sample(1000, 8), 255);
        assert_eq!(clip_display_sample(i32::MAX, 8), 255);
    }

    #[test]
    fn clip_display_sample_lower_saturates() {
        // §7.3 step-3 bottom branch: d < 0 saturates to 0.
        assert_eq!(clip_display_sample(-1, 8), 0);
        assert_eq!(clip_display_sample(-500, 8), 0);
        assert_eq!(clip_display_sample(i32::MIN + 1, 8), 0);
    }

    #[test]
    fn clip_display_sample_10_bit() {
        // §6.3.3 supports `bits_per_pixel != 8` via the `not_8_bit` VOL
        // field; verify the clip tracks the bound rather than hard-
        // coding 8 bits.
        assert_eq!(clip_display_sample(1023, 10), 1023);
        assert_eq!(clip_display_sample(1024, 10), 1023);
        assert_eq!(clip_display_sample(-1, 10), 0);
    }

    #[test]
    fn inter_block_8x8_zero_residual_returns_prediction() {
        // §7.3 step-2 degenerate case: a zero residual leaves d = p.
        let prediction = flat_block(128);
        let residual = flat_block(0);
        let out = reconstruct_inter_block_8x8(&prediction, &residual, 8);
        assert_eq!(out, prediction);
    }

    #[test]
    fn inter_block_8x8_zero_prediction_returns_clipped_residual() {
        // §7.3 step-2 degenerate case: a zero prediction leaves d = f,
        // and §7.3 step-3 then clips. A negative residual maps to 0.
        let prediction = flat_block(0);
        let mut residual = [[0i32; BLOCK_SIDE]; BLOCK_SIDE];
        for y in 0..BLOCK_SIDE {
            for x in 0..BLOCK_SIDE {
                residual[y][x] = if (y + x) % 2 == 0 { -100 } else { 60 };
            }
        }
        let out = reconstruct_inter_block_8x8(&prediction, &residual, 8);
        for y in 0..BLOCK_SIDE {
            for x in 0..BLOCK_SIDE {
                let expect = if (y + x) % 2 == 0 { 0 } else { 60 };
                assert_eq!(out[y][x], expect, "y={y} x={x}");
            }
        }
    }

    #[test]
    fn inter_block_8x8_clip_upper_bound() {
        // §7.3 step-3 top-branch coverage: a prediction near 2^bpp - 1
        // plus a positive residual must saturate at 2^bpp - 1.
        let prediction = flat_block(250);
        let residual = flat_block(20);
        let out = reconstruct_inter_block_8x8(&prediction, &residual, 8);
        assert_eq!(out, flat_block(255));
    }

    #[test]
    fn inter_block_8x8_clip_lower_bound() {
        // §7.3 step-3 bottom-branch coverage: a small prediction plus a
        // large negative residual must saturate at 0.
        let prediction = flat_block(10);
        let residual = flat_block(-50);
        let out = reconstruct_inter_block_8x8(&prediction, &residual, 8);
        assert_eq!(out, flat_block(0));
    }

    #[test]
    fn inter_block_8x8_in_range_sum() {
        // §7.3 step-2 middle case: an in-range sum stays untouched by
        // §7.3 step-3.
        let prediction = flat_block(100);
        let residual = flat_block(50);
        let out = reconstruct_inter_block_8x8(&prediction, &residual, 8);
        assert_eq!(out, flat_block(150));
    }

    #[test]
    fn inter_block_8x8_into_matches_owned() {
        // Buffer-out and owned-return variants must produce identical
        // results on the same input.
        let mut prediction = [[0i32; BLOCK_SIDE]; BLOCK_SIDE];
        let mut residual = [[0i32; BLOCK_SIDE]; BLOCK_SIDE];
        for y in 0..BLOCK_SIDE {
            for x in 0..BLOCK_SIDE {
                prediction[y][x] = (y * BLOCK_SIDE + x) as i32 * 3;
                residual[y][x] = ((y + x) as i32) - 4;
            }
        }
        let owned = reconstruct_inter_block_8x8(&prediction, &residual, 8);
        let mut into = [[0i32; BLOCK_SIDE]; BLOCK_SIDE];
        reconstruct_inter_block_8x8_into(&prediction, &residual, 8, &mut into);
        assert_eq!(owned, into);
    }

    #[test]
    fn intra_block_8x8_clip_only() {
        // §7.3 step-1: intra blocks set d = f. §7.3 step-3 clips. An
        // in-range f is passed through unchanged.
        let sample = flat_block(123);
        let out = reconstruct_intra_block_8x8(&sample, 8);
        assert_eq!(out, sample);
    }

    #[test]
    fn intra_block_8x8_clip_out_of_range() {
        // §7.3 step-3 clip applied to an intra sample below / above the
        // display range. §7.4.5 already clamps to
        // `[-2^bpp, 2^bpp - 1]` so the lower input matches that bound.
        let mut sample = [[0i32; BLOCK_SIDE]; BLOCK_SIDE];
        for y in 0..BLOCK_SIDE {
            for x in 0..BLOCK_SIDE {
                sample[y][x] = if y < 4 { -256 } else { 300 };
            }
        }
        let out = reconstruct_intra_block_8x8(&sample, 8);
        for y in 0..BLOCK_SIDE {
            for x in 0..BLOCK_SIDE {
                let expect = if y < 4 { 0 } else { 255 };
                assert_eq!(out[y][x], expect);
            }
        }
    }

    fn inter_pred_filled(value: i32) -> InterPredictionMacroblock {
        InterPredictionMacroblock {
            luma: [[value; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[value; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[value; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        }
    }

    fn inter_residual_filled(value: i32) -> InterMacroblock {
        InterMacroblock {
            luma: [[value; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[value; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[value; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        }
    }

    #[test]
    fn inter_macroblock_sum_in_range() {
        // §7.3 step-2 over a full 4:2:0 macroblock: in-range sums are
        // passed straight through §7.3 step-3.
        let prediction = inter_pred_filled(120);
        let residual = inter_residual_filled(30);
        let out = reconstruct_inter_macroblock(&prediction, &residual, 8);
        for y in 0..MACROBLOCK_LUMA_SIDE {
            for x in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(out.luma[y][x], 150);
            }
        }
        for y in 0..MACROBLOCK_CHROMA_SIDE {
            for x in 0..MACROBLOCK_CHROMA_SIDE {
                assert_eq!(out.cb[y][x], 150);
                assert_eq!(out.cr[y][x], 150);
            }
        }
    }

    #[test]
    fn inter_macroblock_upper_clip() {
        // §7.3 step-3 upper-branch coverage over a full 4:2:0
        // macroblock.
        let prediction = inter_pred_filled(240);
        let residual = inter_residual_filled(30);
        let out = reconstruct_inter_macroblock(&prediction, &residual, 8);
        for y in 0..MACROBLOCK_LUMA_SIDE {
            for x in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(out.luma[y][x], 255);
            }
        }
    }

    #[test]
    fn inter_macroblock_lower_clip() {
        // §7.3 step-3 lower-branch coverage over a full 4:2:0
        // macroblock.
        let prediction = inter_pred_filled(20);
        let residual = inter_residual_filled(-50);
        let out = reconstruct_inter_macroblock(&prediction, &residual, 8);
        for y in 0..MACROBLOCK_LUMA_SIDE {
            for x in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(out.luma[y][x], 0);
            }
        }
    }

    #[test]
    fn inter_macroblock_zero_prediction_yields_clipped_residual() {
        // `InterPredictionMacroblock::zero()` plus the §7.3 step-2 sum
        // reduces to `d = clip(f, 0, 2^bpp - 1)`.
        let prediction = InterPredictionMacroblock::zero();
        let mut residual = inter_residual_filled(0);
        residual.luma[0][0] = -10;
        residual.luma[0][1] = 100;
        residual.luma[0][2] = 300;
        residual.cb[0][0] = -1;
        residual.cr[0][0] = 256;
        let out = reconstruct_inter_macroblock(&prediction, &residual, 8);
        assert_eq!(out.luma[0][0], 0);
        assert_eq!(out.luma[0][1], 100);
        assert_eq!(out.luma[0][2], 255);
        assert_eq!(out.cb[0][0], 0);
        assert_eq!(out.cr[0][0], 255);
    }

    #[test]
    fn inter_macroblock_into_matches_owned() {
        // Buffer-out variant agrees with the owned-return variant.
        let mut prediction = inter_pred_filled(0);
        let mut residual = inter_residual_filled(0);
        for y in 0..MACROBLOCK_LUMA_SIDE {
            for x in 0..MACROBLOCK_LUMA_SIDE {
                prediction.luma[y][x] = ((y * 16 + x) as i32) % 256;
                residual.luma[y][x] = ((y as i32) - (x as i32)) * 5;
            }
        }
        for y in 0..MACROBLOCK_CHROMA_SIDE {
            for x in 0..MACROBLOCK_CHROMA_SIDE {
                prediction.cb[y][x] = ((y * 8 + x) as i32) * 2;
                prediction.cr[y][x] = ((y + x) as i32) * 7;
                residual.cb[y][x] = -3 * (y as i32);
                residual.cr[y][x] = 4 * (x as i32);
            }
        }
        let owned = reconstruct_inter_macroblock(&prediction, &residual, 8);
        let mut into = ReconstructedMacroblock {
            luma: [[0i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        };
        reconstruct_inter_macroblock_into(&prediction, &residual, 8, &mut into);
        assert_eq!(owned, into);
    }

    #[test]
    fn intra_macroblock_clip() {
        // §7.3 step-1 + step-3 over a full 4:2:0 macroblock: f passes
        // through, then the clip enforces the display range.
        let mut sample = IntraMacroblock {
            luma: [[0i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        };
        sample.luma[0][0] = -1;
        sample.luma[0][1] = 200;
        sample.luma[0][2] = 256;
        sample.cb[1][1] = -5;
        sample.cr[1][1] = 257;
        let out = reconstruct_intra_macroblock(&sample, 8);
        assert_eq!(out.luma[0][0], 0);
        assert_eq!(out.luma[0][1], 200);
        assert_eq!(out.luma[0][2], 255);
        assert_eq!(out.cb[1][1], 0);
        assert_eq!(out.cr[1][1], 255);
    }

    #[test]
    fn inter_pred_zero_constructor() {
        // `InterPredictionMacroblock::zero` must be a true all-zero
        // macroblock so the §7.3 step-2 sum simplifies to d = f.
        let z = InterPredictionMacroblock::zero();
        for y in 0..MACROBLOCK_LUMA_SIDE {
            for x in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(z.luma[y][x], 0);
            }
        }
        for y in 0..MACROBLOCK_CHROMA_SIDE {
            for x in 0..MACROBLOCK_CHROMA_SIDE {
                assert_eq!(z.cb[y][x], 0);
                assert_eq!(z.cr[y][x], 0);
            }
        }
    }

    #[test]
    fn inter_macroblock_per_plane_independence() {
        // The luma / Cb / Cr planes must be processed independently —
        // a §7.3 step-3 clip on one plane must not affect the others.
        let mut prediction = inter_pred_filled(0);
        let mut residual = inter_residual_filled(0);
        // Saturate Cr at the top, leave Cb and luma in-range.
        prediction.cr[3][3] = 250;
        residual.cr[3][3] = 30;
        prediction.cb[3][3] = 100;
        residual.cb[3][3] = 20;
        prediction.luma[5][5] = 80;
        residual.luma[5][5] = -10;
        let out = reconstruct_inter_macroblock(&prediction, &residual, 8);
        assert_eq!(out.cr[3][3], 255);
        assert_eq!(out.cb[3][3], 120);
        assert_eq!(out.luma[5][5], 70);
    }

    #[test]
    fn reconstruct_inter_macroblock_10_bit() {
        // The §6.3.3 `not_8_bit` path: verify the clip uses the
        // supplied `bits_per_pixel` rather than 8.
        let prediction = inter_pred_filled(1000);
        let residual = inter_residual_filled(50);
        let out = reconstruct_inter_macroblock(&prediction, &residual, 10);
        for y in 0..MACROBLOCK_LUMA_SIDE {
            for x in 0..MACROBLOCK_LUMA_SIDE {
                // 1000 + 50 = 1050, but 2^10 - 1 = 1023.
                assert_eq!(out.luma[y][x], 1023);
            }
        }
    }
}

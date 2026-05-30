//! §7.6.2.2 Quarter-sample mode interpolation (Figures 7-31 / 7-32).
//!
//! Quarter-sample motion compensation is selected at the VOL level by
//! `quarter_sample == 1`. It applies **only** to the luminance
//! component of a P- / S(GMC)- / B-VOP; chrominance motion vectors are
//! reduced from quarter-pel units to half-pel units via Table 7-13
//! ([`reduce_qpel_to_half_pel_chroma`]) and then processed by the
//! §7.6.2.1 half-sample interpolation in [`crate::half_sample`].
//!
//! ## Luminance interpolation
//!
//! The §7.6.2.2.1 half-sample positions `b`, `c`, `d` (the cardinal
//! and diagonal half-pels relative to integer-pel `a = A_{-1,-1}`)
//! are computed by an 8-tap symmetric FIR filter with coefficients
//!
//! ```text
//!   (C[4], C[3], C[2], C[1], C[1], C[2], C[3], C[4]) / 256
//! ```
//!
//! where `C = [160, -48, 24, -8]` for `j = 1..=4`. The denominator
//! is `256` and the rounding offset is `128 - rounding_control`, so
//! the per-sample equations from Figure 7-31 are
//!
//! ```text
//!   b_i = ( Σ_{j=1..4} C[j] * (A_{-j, i} + A_{+j, i}) + 128 - rc ) / 256
//!   c   = ( Σ_{j=1..4} C[j] * (A_{-1, -j} + A_{-1, +j}) + 128 - rc ) / 256
//!   d   = ( Σ_{j=1..4} C[j] * (b_{-j} + b_{+j}) + 128 - rc ) / 256
//! ```
//!
//! Each FIR result is clipped to `[0, 2^bpp - 1]` (§7.6.2.2.1 last
//! sentence) **before** being used as an input to a downstream FIR
//! or bilinear stage — `d` consumes already-clipped `b_i`, and the
//! §7.6.2.2.2 quarter samples consume already-clipped `b`, `c`,
//! `d`. This is the ordering required to make the cascade match the
//! spec's two-step description.
//!
//! ## Quarter-sample positions
//!
//! Figure 7-32 names the quarter-pel positions:
//!
//! ```text
//!   A_{-1,-1}   e_{-1}   b_{-1}   f_{-1}   A_{1,-1}
//!     g           h        i        j
//!     c           k        d        l
//!     m           n        o        p
//!   A_{-1, 1}   e_{1}    b_{1}    f_{1}    A_{1, 1}
//! ```
//!
//! Per §7.6.2.2.2 the quarter-pel samples are bilinear (`(X + Y + 1 -
//! rc) / 2`) blends of the surrounding half/integer samples — except
//! `k` and `l` which apply the 8-tap FIR vertically to the
//! `e` / `f` columns:
//!
//! ```text
//!   e_i = (A_{-1, i} + b_i + 1 - rc) / 2
//!   f_i = (b_i + A_{1, i} + 1 - rc) / 2
//!   k   = ( Σ C[j] * (e_{-j} + e_{+j}) + 128 - rc ) / 256
//!   l   = ( Σ C[j] * (f_{-j} + f_{+j}) + 128 - rc ) / 256
//!   g   = (A_{-1,-1} + c + 1 - rc) / 2
//!   h   = (e_{-1} + k + 1 - rc) / 2
//!   i   = (b_{-1} + d + 1 - rc) / 2
//!   j   = (l + f_{-1} + 1 - rc) / 2
//!   m   = (c + A_{-1, 1} + 1 - rc) / 2
//!   n   = (k + e_{1} + 1 - rc) / 2
//!   o   = (d + b_{1} + 1 - rc) / 2
//!   p   = (l + f_{1} + 1 - rc) / 2
//! ```
//!
//! All quarter-sample values are clipped to `[0, 2^bpp - 1]`
//! (§7.6.2.2.2 last sentence).
//!
//! ## Sub-pel index convention
//!
//! Quarter-pel motion-vector components are decoded into a signed
//! integer in §7.6.3 *quarter-sample units*. Two LSBs encode the
//! fractional position:
//!
//! | `(qfrac_x, qfrac_y)` | Position             |
//! | -------------------- | -------------------- |
//! | `(0, 0)`             | `a` (integer-pel `A`)|
//! | `(1, 0)`             | `e_{-1}`             |
//! | `(2, 0)`             | `b_{-1}` (half-pel)  |
//! | `(3, 0)`             | `f_{-1}`             |
//! | `(0, 1)`             | `g`                  |
//! | `(1, 1)`             | `h`                  |
//! | `(2, 1)`             | `i`                  |
//! | `(3, 1)`             | `j`                  |
//! | `(0, 2)`             | `c` (half-pel)       |
//! | `(1, 2)`             | `k`                  |
//! | `(2, 2)`             | `d` (half-pel)       |
//! | `(3, 2)`             | `l`                  |
//! | `(0, 3)`             | `m`                  |
//! | `(1, 3)`             | `n`                  |
//! | `(2, 3)`             | `o`                  |
//! | `(3, 3)`             | `p`                  |
//!
//! [`split_quarter_pel`] decomposes one MV component into
//! `(integer_part, qfrac ∈ 0..=3)`. The integer part uses arithmetic
//! shift by 2, so negative MVs round toward `-∞` and the fractional
//! pair is non-negative — same `floor`-of-`/4` convention §3.4 uses
//! for division.
//!
//! ## Out of scope (this round)
//!
//! * §7.6.1 reference-VOP padding — the caller hands us a fully
//!   reconstructed and padded reference plane.
//! * Interlaced field-based motion compensation. The §7.6.2.2 note
//!   for half-sample mode ("vertically interpolated between two
//!   successive lines of the same field") extends here; needs a
//!   field-aware sample fetcher, deferred.

use crate::half_sample::ReferenceVop;

/// 8-tap FIR coefficients `C[1..=4] = [160, -48, 24, -8]`. The full
/// symmetric kernel is `(C[4], C[3], C[2], C[1], C[1], C[2], C[3],
/// C[4]) = (-8, 24, -48, 160, 160, -48, 24, -8)`, divided by 256.
pub const QPEL_FIR_C: [i32; 4] = [160, -48, 24, -8];

/// Split a §7.6.3 quarter-pel motion-vector component into the
/// integer-pel offset and the quarter-pel fractional bits.
///
/// `mv` is in quarter-sample units (so `mv = 1` is a quarter-pel
/// motion, `mv = 2` is a half-pel motion, `mv = 4` is a full-pel
/// motion). The integer part is `mv >> 2` (arithmetic shift, rounds
/// toward `-∞`); the fractional position is `mv & 3` in `0..=3`.
///
/// The decomposition satisfies `mv == (integer << 2) + (qfrac as i32)`
/// for all signed inputs because the arithmetic shift on a negative
/// number floors the integer part, leaving the low two bits to encode
/// the fractional position non-negatively. For example
/// `split_quarter_pel(-1) == (-1, 3)` since `-1 = (-1 << 2) + 3`.
///
/// # Examples
///
/// ```
/// use oxideav_mpeg4video::quarter_sample::split_quarter_pel;
/// assert_eq!(split_quarter_pel(0), (0, 0));
/// assert_eq!(split_quarter_pel(1), (0, 1));
/// assert_eq!(split_quarter_pel(2), (0, 2));
/// assert_eq!(split_quarter_pel(3), (0, 3));
/// assert_eq!(split_quarter_pel(4), (1, 0));
/// assert_eq!(split_quarter_pel(-1), (-1, 3));
/// assert_eq!(split_quarter_pel(-4), (-1, 0));
/// assert_eq!(split_quarter_pel(-5), (-2, 3));
/// ```
#[inline]
pub const fn split_quarter_pel(mv: i32) -> (i32, u8) {
    let integer = mv >> 2;
    let qfrac = (mv & 3) as u8;
    (integer, qfrac)
}

/// Reduce a §7.6.3 quarter-pel luma motion-vector component to a
/// half-pel chrominance motion-vector component using Table 7-13.
///
/// The luma MV component `c` is in quarter-pel units. The
/// chrominance MV component is in half-pel units (because §6.1.3.4's
/// 4:2:0 chroma subsampling halves the spatial resolution). Per
/// §7.6.5's preamble, the luma → chroma reduction in quarter-sample
/// mode divides by 2 (giving "fourth sample resolution" — quarter
/// luma pixels equal half chroma pixels, after the 2× sub-sampling)
/// and applies Table 7-13's modification toward the nearest half-pel
/// chrominance position.
///
/// Table 7-13 maps the quarter-pel fractional position
/// `{0, 1, 2, 3}` to the half-pel fractional position
/// `{0, 1, 1, 1}` — any non-zero quarter-pel offset rounds toward the
/// half-pel position `1` (= 0.5 chroma-pel). The integer part is
/// preserved.
///
/// For negative MVs we use the `floor`-of-`/4` decomposition from
/// [`split_quarter_pel`], so e.g. `-1` (= -0.25 chroma-pel) maps to
/// `-1` (= -0.5 chroma-pel); `-4` (= -1 chroma-pel) maps to `-2` (=
/// -1 chroma-pel). The mapping is anti-symmetric about zero.
///
/// # Examples
///
/// ```
/// use oxideav_mpeg4video::quarter_sample::reduce_qpel_to_half_pel_chroma;
/// // Table 7-13 fractional column.
/// assert_eq!(reduce_qpel_to_half_pel_chroma(0), 0);
/// assert_eq!(reduce_qpel_to_half_pel_chroma(1), 1);
/// assert_eq!(reduce_qpel_to_half_pel_chroma(2), 1);
/// assert_eq!(reduce_qpel_to_half_pel_chroma(3), 1);
/// // Integer-pel offsets (multiples of 4) double into half-pel
/// // integers (multiples of 2).
/// assert_eq!(reduce_qpel_to_half_pel_chroma(4), 2);
/// assert_eq!(reduce_qpel_to_half_pel_chroma(8), 4);
/// // Mixed: 5 quarter-pels == 1 full-pel + 1 quarter → 2 half-pels
/// // + 1 half-pel = 3 half-pels.
/// assert_eq!(reduce_qpel_to_half_pel_chroma(5), 3);
/// // Negative MVs floor toward -∞: -1 quarter → -1 half-pel.
/// assert_eq!(reduce_qpel_to_half_pel_chroma(-1), -1);
/// assert_eq!(reduce_qpel_to_half_pel_chroma(-4), -2);
/// assert_eq!(reduce_qpel_to_half_pel_chroma(-5), -3);
/// ```
#[inline]
pub const fn reduce_qpel_to_half_pel_chroma(c: i32) -> i32 {
    // Table 7-13 fractional mapping: {0 → 0, 1 → 1, 2 → 1, 3 → 1}.
    // Implementable as `min(qfrac, 1)`. Combined with the integer
    // part doubled, we get the half-pel-units result.
    let (int_part, qfrac) = split_quarter_pel(c);
    let half_pel_frac = if qfrac == 0 { 0i32 } else { 1i32 };
    (int_part << 1) | half_pel_frac
}

/// Apply the §7.6.2.2.1 8-tap symmetric FIR to the eight input
/// samples `s[0..8]` (centred on the half-pel position between
/// `s[3]` and `s[4]`) with rounding control `rc ∈ {0, 1}` and clip
/// the result to `[0, 2^bpp - 1]`.
///
/// The kernel is `(C[4], C[3], C[2], C[1], C[1], C[2], C[3], C[4])
/// / 256` with `C = [160, -48, 24, -8]`. The integer division by
/// `256` is the `/` operator per §3.4 (truncate-toward-zero). For
/// `bits_per_pixel == 8` the result is clipped to `[0, 255]`.
///
/// Used as the building block for the §7.6.2.2.1 `b`, `c`, `d` and
/// §7.6.2.2.2 `k`, `l` formulas. Each FIR result is independently
/// clipped before being used downstream (so the cascade matches the
/// spec's two-step description).
///
/// # Examples
///
/// ```
/// use oxideav_mpeg4video::quarter_sample::fir_8tap_clip;
/// // Flat input → output equals the constant (the FIR coefficients
/// // sum to 256 — symmetric pair sums: 160+160 - 48-48 + 24+24 - 8-8
/// // = 320 - 96 + 48 - 16 = 256).
/// assert_eq!(fir_8tap_clip(&[100; 8], 0, 8), 100);
/// assert_eq!(fir_8tap_clip(&[100; 8], 1, 8), 100);
/// ```
#[inline]
pub fn fir_8tap_clip(s: &[u8; 8], rounding_control: u8, bits_per_pixel: u32) -> u8 {
    let rc = (rounding_control & 1) as i32;
    let c = QPEL_FIR_C;
    let acc: i32 = c[0] * (s[3] as i32 + s[4] as i32)
        + c[1] * (s[2] as i32 + s[5] as i32)
        + c[2] * (s[1] as i32 + s[6] as i32)
        + c[3] * (s[0] as i32 + s[7] as i32);
    let v = (acc + 128 - rc) / 256;
    let max = (1i32 << bits_per_pixel) - 1;
    v.clamp(0, max) as u8
}

/// Read eight horizontal samples centred on the half-pel position
/// between integer cells `(int_x, int_y)` and `(int_x + 1, int_y)`
/// from a §7.6.4-clamped reference plane.
///
/// Returns `[A_{-3,0}, A_{-2,0}, A_{-1,0}, A_{0,0}, A_{1,0},
/// A_{2,0}, A_{3,0}, A_{4,0}]` (relative to the target half-pel
/// position; in plane coordinates the eight `x` indices are
/// `int_x - 3 .. int_x + 5`).
#[inline]
fn horiz_taps(vop: &ReferenceVop<'_>, int_x: i32, int_y: i32) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (k, dx) in (-3i32..=4).enumerate() {
        out[k] = vop.fetch_clamped(int_x + dx, int_y);
    }
    out
}

/// Read eight vertical samples centred on the half-pel position
/// between integer cells `(int_x, int_y)` and `(int_x, int_y + 1)`.
#[inline]
fn vert_taps(vop: &ReferenceVop<'_>, int_x: i32, int_y: i32) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (k, dy) in (-3i32..=4).enumerate() {
        out[k] = vop.fetch_clamped(int_x, int_y + dy);
    }
    out
}

/// §7.6.2.2.1 horizontal half-pel `b` at integer `(int_x, int_y)`.
///
/// Half-pel position is between `(int_x, int_y)` and `(int_x + 1,
/// int_y)`. Equivalent to `(Σ C[j] * (A_{-j, 0} + A_{+j, 0}) + 128 -
/// rc) / 256`, then clipped to `[0, 2^bpp - 1]`.
#[inline]
pub fn half_pel_b(
    vop: &ReferenceVop<'_>,
    int_x: i32,
    int_y: i32,
    rounding_control: u8,
    bits_per_pixel: u32,
) -> u8 {
    let taps = horiz_taps(vop, int_x, int_y);
    fir_8tap_clip(&taps, rounding_control, bits_per_pixel)
}

/// §7.6.2.2.1 vertical half-pel `c` at integer `(int_x, int_y)`.
///
/// Half-pel position is between `(int_x, int_y)` and `(int_x, int_y
/// + 1)`. Equivalent to `(Σ C[j] * (A_{0, -j} + A_{0, +j}) + 128 -
///   rc) / 256`, then clipped.
#[inline]
pub fn half_pel_c(
    vop: &ReferenceVop<'_>,
    int_x: i32,
    int_y: i32,
    rounding_control: u8,
    bits_per_pixel: u32,
) -> u8 {
    let taps = vert_taps(vop, int_x, int_y);
    fir_8tap_clip(&taps, rounding_control, bits_per_pixel)
}

/// §7.6.2.2.1 diagonal half-pel `d` at integer `(int_x, int_y)`.
///
/// `d` is the vertical FIR applied to the eight horizontal half-pel
/// `b` values `b_{-3} .. b_4`. Each `b_i` is independently computed
/// and clipped before entering the second-stage FIR.
#[inline]
pub fn half_pel_d(
    vop: &ReferenceVop<'_>,
    int_x: i32,
    int_y: i32,
    rounding_control: u8,
    bits_per_pixel: u32,
) -> u8 {
    let mut col = [0u8; 8];
    for (k, dy) in (-3i32..=4).enumerate() {
        col[k] = half_pel_b(vop, int_x, int_y + dy, rounding_control, bits_per_pixel);
    }
    fir_8tap_clip(&col, rounding_control, bits_per_pixel)
}

/// One bilinear blend `(x + y + 1 - rc) / 2`, clipped to `[0, 2^bpp -
/// 1]`. The §3.4 `/` operator truncates toward zero; for the legal
/// `(x, y)` input range `[0, 2^bpp - 1]` the numerator is in
/// `[0, 2^(bpp+1)]` so truncation is equivalent to `floor`.
#[inline]
fn bilinear(x: u8, y: u8, rounding_control: u8, bits_per_pixel: u32) -> u8 {
    let rc = (rounding_control & 1) as u32;
    let v = (x as u32 + y as u32 + 1 - rc) / 2;
    let max = (1u32 << bits_per_pixel) - 1;
    v.min(max) as u8
}

/// Interpolate one luma quarter-sample-mode pixel at sub-pel position
/// `(qfrac_x, qfrac_y) ∈ {0, 1, 2, 3}²` relative to the integer-pel
/// anchor `(int_x, int_y) = A_{-1,-1}` (the top-left of the 4×4
/// quarter-pel grid that surrounds it).
///
/// Returns the §7.6.2.2 sample at the chosen sub-pel position. The
/// sixteen positions correspond to Figure 7-32 in the order listed in
/// the module-level [`Sub-pel index convention`](self).
///
/// `rounding_control` is the VOP-header `vop_rounding_type` bit; any
/// other value is masked to its low bit. `bits_per_pixel` is taken
/// from the VOL header (`bits_per_pixel = 8` for `not_8_bit == 0`).
///
/// §7.6.4 last-full-pel clamping is applied to every reference-plane
/// fetch via [`ReferenceVop::fetch_clamped`].
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn interpolate_quarter_pixel(
    vop: &ReferenceVop<'_>,
    int_x: i32,
    int_y: i32,
    qfrac_x: u8,
    qfrac_y: u8,
    rounding_control: u8,
    bits_per_pixel: u32,
) -> u8 {
    // Figure 7-32 origin: A_{-1,-1} is the integer-pel anchor and
    // sits at plane coordinate (int_x, int_y); A_{1,-1} is at
    // (int_x + 1, int_y); A_{-1, 1} is at (int_x, int_y + 1); and
    // A_{1, 1} is at (int_x + 1, int_y + 1).
    let rc = rounding_control & 1;
    let bpp = bits_per_pixel;
    let qx = qfrac_x & 3;
    let qy = qfrac_y & 3;

    match (qx, qy) {
        // Row 0 (qy == 0): integer / e_{-1} / b_{-1} / f_{-1}.
        (0, 0) => vop.fetch_clamped(int_x, int_y),
        (1, 0) => {
            // e_{-1} = (A_{-1, -1} + b_{-1} + 1 - rc) / 2.
            // Implemented at i = 0 row of Figure 7-32; the spec uses
            // `e_{-1}` to denote the upper-row quarter-pel column.
            let a = vop.fetch_clamped(int_x, int_y);
            let b = half_pel_b(vop, int_x, int_y, rc, bpp);
            bilinear(a, b, rc, bpp)
        }
        (2, 0) => half_pel_b(vop, int_x, int_y, rc, bpp),
        (3, 0) => {
            // f_{-1} = (b_{-1} + A_{1, -1} + 1 - rc) / 2.
            let a = vop.fetch_clamped(int_x + 1, int_y);
            let b = half_pel_b(vop, int_x, int_y, rc, bpp);
            bilinear(b, a, rc, bpp)
        }
        // Row 1 (qy == 1): g / h / i / j.
        (0, 1) => {
            // g = (A_{-1, -1} + c + 1 - rc) / 2.
            let a = vop.fetch_clamped(int_x, int_y);
            let c = half_pel_c(vop, int_x, int_y, rc, bpp);
            bilinear(a, c, rc, bpp)
        }
        (1, 1) => {
            // h = (e_{-1} + k + 1 - rc) / 2.
            // e_{-1} is the row-0 quarter at qx=1, qy=0.
            // k is the vertical FIR over the e column (positions
            // e_{-3} .. e_{4}); that needs the eight `e` values
            // each computed independently.
            let e_top = {
                let a = vop.fetch_clamped(int_x, int_y);
                let b = half_pel_b(vop, int_x, int_y, rc, bpp);
                bilinear(a, b, rc, bpp)
            };
            let k = compute_k(vop, int_x, int_y, rc, bpp);
            bilinear(e_top, k, rc, bpp)
        }
        (2, 1) => {
            // i = (b_{-1} + d + 1 - rc) / 2.
            let b = half_pel_b(vop, int_x, int_y, rc, bpp);
            let d = half_pel_d(vop, int_x, int_y, rc, bpp);
            bilinear(b, d, rc, bpp)
        }
        (3, 1) => {
            // j = (l + f_{-1} + 1 - rc) / 2.
            let f_top = {
                let a = vop.fetch_clamped(int_x + 1, int_y);
                let b = half_pel_b(vop, int_x, int_y, rc, bpp);
                bilinear(b, a, rc, bpp)
            };
            let l = compute_l(vop, int_x, int_y, rc, bpp);
            bilinear(l, f_top, rc, bpp)
        }
        // Row 2 (qy == 2): c / k / d / l.
        (0, 2) => half_pel_c(vop, int_x, int_y, rc, bpp),
        (1, 2) => compute_k(vop, int_x, int_y, rc, bpp),
        (2, 2) => half_pel_d(vop, int_x, int_y, rc, bpp),
        (3, 2) => compute_l(vop, int_x, int_y, rc, bpp),
        // Row 3 (qy == 3): m / n / o / p.
        (0, 3) => {
            // m = (c + A_{-1, 1} + 1 - rc) / 2.
            let a = vop.fetch_clamped(int_x, int_y + 1);
            let c = half_pel_c(vop, int_x, int_y, rc, bpp);
            bilinear(c, a, rc, bpp)
        }
        (1, 3) => {
            // n = (k + e_{1} + 1 - rc) / 2.
            let e_bot = {
                let a = vop.fetch_clamped(int_x, int_y + 1);
                let b = half_pel_b(vop, int_x, int_y + 1, rc, bpp);
                bilinear(a, b, rc, bpp)
            };
            let k = compute_k(vop, int_x, int_y, rc, bpp);
            bilinear(k, e_bot, rc, bpp)
        }
        (2, 3) => {
            // o = (d + b_{1} + 1 - rc) / 2.
            let b_bot = half_pel_b(vop, int_x, int_y + 1, rc, bpp);
            let d = half_pel_d(vop, int_x, int_y, rc, bpp);
            bilinear(d, b_bot, rc, bpp)
        }
        (3, 3) => {
            // p = (l + f_{1} + 1 - rc) / 2.
            let f_bot = {
                let a = vop.fetch_clamped(int_x + 1, int_y + 1);
                let b = half_pel_b(vop, int_x, int_y + 1, rc, bpp);
                bilinear(b, a, rc, bpp)
            };
            let l = compute_l(vop, int_x, int_y, rc, bpp);
            bilinear(l, f_bot, rc, bpp)
        }
        _ => unreachable!("qfrac masked to 2 bits"),
    }
}

/// §7.6.2.2.2 quarter-sample `k` at integer-anchor `(int_x, int_y)`.
///
/// `k = (Σ C[j] * (e_{-j} + e_{+j}) + 128 - rc) / 256`, where each
/// `e_i = (A_{-1, i} + b_i + 1 - rc) / 2` is the left-column quarter
/// at row `i ∈ {-3..=4}` relative to the integer anchor.
fn compute_k(vop: &ReferenceVop<'_>, int_x: i32, int_y: i32, rc: u8, bpp: u32) -> u8 {
    let mut col = [0u8; 8];
    for (k, dy) in (-3i32..=4).enumerate() {
        let a = vop.fetch_clamped(int_x, int_y + dy);
        let b = half_pel_b(vop, int_x, int_y + dy, rc, bpp);
        col[k] = bilinear(a, b, rc, bpp);
    }
    fir_8tap_clip(&col, rc, bpp)
}

/// §7.6.2.2.2 quarter-sample `l` at integer-anchor `(int_x, int_y)`.
///
/// `l = (Σ C[j] * (f_{-j} + f_{+j}) + 128 - rc) / 256`, where each
/// `f_i = (b_i + A_{1, i} + 1 - rc) / 2` is the right-column quarter
/// at row `i ∈ {-3..=4}`.
fn compute_l(vop: &ReferenceVop<'_>, int_x: i32, int_y: i32, rc: u8, bpp: u32) -> u8 {
    let mut col = [0u8; 8];
    for (k, dy) in (-3i32..=4).enumerate() {
        let b = half_pel_b(vop, int_x, int_y + dy, rc, bpp);
        let a = vop.fetch_clamped(int_x + 1, int_y + dy);
        col[k] = bilinear(b, a, rc, bpp);
    }
    fir_8tap_clip(&col, rc, bpp)
}

/// Quarter-sample-interpolate a `block_w × block_h` luminance
/// prediction block from a reference plane, given an `(mv_x, mv_y)`
/// motion vector in §7.6.3 quarter-pel units and the block's
/// top-left pixel origin `(origin_x, origin_y)` in the *current*
/// (predicted) VOP.
///
/// The output is laid out row-major as `block[j][i] = block_w * j + i`
/// and has length `block_w * block_h`.
///
/// `mv_x` / `mv_y` are signed quarter-pel motion-vector components;
/// `vop_rounding_type ∈ {0, 1}` is the VOP-header field;
/// `bits_per_pixel` is from the VOL header. §7.6.4 edge clamping is
/// applied per fetch.
//
// The signature reflects the §7.6.2.2 inputs one-for-one (motion
// vector x/y, block origin x/y, block w/h, rounding control, bpp).
#[allow(clippy::too_many_arguments)]
pub fn interpolate_block_qpel(
    vop: &ReferenceVop<'_>,
    mv_x: i32,
    mv_y: i32,
    origin_x: i32,
    origin_y: i32,
    block_w: usize,
    block_h: usize,
    vop_rounding_type: u8,
    bits_per_pixel: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; block_w * block_h];
    interpolate_block_qpel_into(
        vop,
        mv_x,
        mv_y,
        origin_x,
        origin_y,
        block_w,
        block_h,
        vop_rounding_type,
        bits_per_pixel,
        &mut out,
    );
    out
}

/// Quarter-sample-interpolate a `block_w × block_h` luminance
/// prediction block into a caller-supplied buffer of length
/// `block_w * block_h`.
///
/// See [`interpolate_block_qpel`] for parameter semantics.
///
/// # Panics
///
/// Panics if `out.len() < block_w * block_h`.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_block_qpel_into(
    vop: &ReferenceVop<'_>,
    mv_x: i32,
    mv_y: i32,
    origin_x: i32,
    origin_y: i32,
    block_w: usize,
    block_h: usize,
    vop_rounding_type: u8,
    bits_per_pixel: u32,
    out: &mut [u8],
) {
    assert!(
        out.len() >= block_w * block_h,
        "interpolate_block_qpel_into: output buffer too small ({} < {} * {})",
        out.len(),
        block_w,
        block_h,
    );
    let (mvx_int, qfx) = split_quarter_pel(mv_x);
    let (mvy_int, qfy) = split_quarter_pel(mv_y);
    let rc = vop_rounding_type & 1;
    for j in 0..block_h {
        let row_y = origin_y + (j as i32) + mvy_int;
        for i in 0..block_w {
            let col_x = origin_x + (i as i32) + mvx_int;
            out[j * block_w + i] =
                interpolate_quarter_pixel(vop, col_x, row_y, qfx, qfy, rc, bits_per_pixel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────── split_quarter_pel ───────────────────────

    #[test]
    fn split_quarter_pel_round_trips_positive_and_negative() {
        for mv in -16i32..=16 {
            let (i, q) = split_quarter_pel(mv);
            assert!(q < 4, "qfrac out of range");
            assert_eq!((i << 2) + (q as i32), mv, "mv = {mv}");
        }
    }

    #[test]
    fn split_quarter_pel_zero() {
        assert_eq!(split_quarter_pel(0), (0, 0));
    }

    #[test]
    fn split_quarter_pel_full_pel_step() {
        // mv = 4 is exactly one full pel.
        assert_eq!(split_quarter_pel(4), (1, 0));
        assert_eq!(split_quarter_pel(-4), (-1, 0));
    }

    #[test]
    fn split_quarter_pel_negative_below_one_pel() {
        // mv = -5 is -1 1/4 pel: integer -2, qfrac 3 (since
        // -5 = (-2 << 2) + 3 = -8 + 3 = -5).
        assert_eq!(split_quarter_pel(-5), (-2, 3));
    }

    // ─────────────────── reduce_qpel_to_half_pel_chroma ──────────────

    #[test]
    fn reduce_qpel_chroma_table_7_13_fractional_column() {
        // Table 7-13: fourth pel position {0,1,2,3} → resulting
        // position {0,1,1,1} (//2 = half-pel units).
        let expected = [0i32, 1, 1, 1];
        for (qfrac, &exp) in expected.iter().enumerate() {
            assert_eq!(
                reduce_qpel_to_half_pel_chroma(qfrac as i32),
                exp,
                "qfrac = {qfrac}",
            );
        }
    }

    #[test]
    fn reduce_qpel_chroma_integer_pel_offsets_double() {
        for n in -8i32..=8 {
            // n full quarter-pels with frac=0 → 2n half-pel units.
            let q = n * 4;
            assert_eq!(reduce_qpel_to_half_pel_chroma(q), 2 * n, "n = {n}");
        }
    }

    #[test]
    fn reduce_qpel_chroma_negative_floors() {
        // Spec's table is anti-symmetric about zero via the `floor`
        // decomposition. -1 (= -0.25 chroma-pel) → -1 (= -0.5).
        assert_eq!(reduce_qpel_to_half_pel_chroma(-1), -1);
        // -2 → -1 (same row of Table 7-13, just int part -1).
        assert_eq!(reduce_qpel_to_half_pel_chroma(-2), -1);
        // -3 → -1 (one quarter past -0.5 toward -0.75 still rounds
        // to -0.5 in chroma half-pel).
        assert_eq!(reduce_qpel_to_half_pel_chroma(-3), -1);
        // -4 → -2 (one full pel).
        assert_eq!(reduce_qpel_to_half_pel_chroma(-4), -2);
    }

    // ────────────────────── fir_8tap_clip ────────────────────────────

    #[test]
    fn fir_8tap_clip_flat_reproduces_constant() {
        for &v in &[0u8, 1, 100, 200, 255] {
            assert_eq!(fir_8tap_clip(&[v; 8], 0, 8), v);
            assert_eq!(fir_8tap_clip(&[v; 8], 1, 8), v);
        }
    }

    #[test]
    fn fir_8tap_clip_symmetric_kernel_sum() {
        // C = [160, -48, 24, -8]; symmetric kernel sum:
        // 2*(160 + (-48) + 24 + (-8)) = 2 * 128 = 256.
        let sum_pairs = 2 * (160 + -48 + 24 + -8);
        assert_eq!(sum_pairs, 256);
    }

    #[test]
    fn fir_8tap_clip_step_edge_response() {
        // Step input [0,0,0,0,255,255,255,255]: FIR response at the
        // half-pel between the 4th and 5th sample. By symmetry the
        // result is around 127–128 with rc influencing the sub-LSB.
        // Compute exactly:
        //   acc = 160*(0+255) + -48*(0+255) + 24*(0+255) + -8*(0+255)
        //       = 255 * (160 - 48 + 24 - 8) = 255 * 128 = 32640.
        //   (32640 + 128 - 0)/256 = 32768/256 = 128.
        //   (32640 + 128 - 1)/256 = 32767/256 = 127.99... → 127.
        let s = [0, 0, 0, 0, 255, 255, 255, 255];
        assert_eq!(fir_8tap_clip(&s, 0, 8), 128);
        assert_eq!(fir_8tap_clip(&s, 1, 8), 127);
    }

    #[test]
    fn fir_8tap_clip_clips_below_zero() {
        // s = [255, 0, 0, 0, 0, 0, 0, 255]; with C=[160,-48,24,-8]
        // the only non-zero terms come from the far-tap pair:
        // acc = -8 * (255 + 255) = -4080
        // (-4080 + 128) / 256 = -3952 / 256 = -15 (Rust truncates
        // toward zero, but for negative dividend / positive divisor
        // truncation toward zero equals -ceil(|...|) which equals
        // -16; verify with the runtime kernel below).
        let s = [255, 0, 0, 0, 0, 0, 0, 255];
        let pre_clip = (-4080_i32 + 128) / 256;
        assert!(pre_clip < 0, "expected negative pre-clip, got {pre_clip}");
        assert_eq!(fir_8tap_clip(&s, 0, 8), 0);
    }

    #[test]
    fn fir_8tap_clip_clips_above_max_at_bpp_8() {
        // s = [0, 0, 0, 255, 255, 0, 0, 0]; only the centre tap
        // pair contributes: acc = 160 * (255 + 255) = 81600.
        // (81600 + 128) / 256 = 81728 / 256 = 319 → clip to 255.
        let s = [0, 0, 0, 255, 255, 0, 0, 0];
        let pre_clip = (81_600_i32 + 128) / 256;
        assert!(pre_clip > 255, "expected > 255 pre-clip, got {pre_clip}");
        assert_eq!(fir_8tap_clip(&s, 0, 8), 255);
    }

    // ─────────────────── half_pel_b / c / d ──────────────────────────

    #[test]
    fn half_pel_b_flat_plane_is_constant() {
        let buf = [77u8; 8 * 8];
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        for x in 0..8 {
            for y in 0..8 {
                assert_eq!(half_pel_b(&vop, x, y, 0, 8), 77);
                assert_eq!(half_pel_b(&vop, x, y, 1, 8), 77);
            }
        }
    }

    #[test]
    fn half_pel_c_flat_plane_is_constant() {
        let buf = [33u8; 8 * 8];
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        assert_eq!(half_pel_c(&vop, 4, 4, 0, 8), 33);
    }

    #[test]
    fn half_pel_d_flat_plane_is_constant() {
        let buf = [200u8; 8 * 8];
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        assert_eq!(half_pel_d(&vop, 4, 4, 0, 8), 200);
        assert_eq!(half_pel_d(&vop, 4, 4, 1, 8), 200);
    }

    #[test]
    fn half_pel_b_horizontal_step_matches_fir_centre() {
        // Make plane horizontally constant per column so the b
        // computation reduces to a 1-D step response we can verify.
        // 8-wide plane: [0, 0, 0, 0, 255, 255, 255, 255].
        let row = [0u8, 0, 0, 0, 255, 255, 255, 255];
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&row);
        let vop = ReferenceVop::new(&buf, 8, 1).unwrap();
        // half-pel b at int_x = 3, int_y = 0: half between row[3]
        // and row[4]; taps are row[0..8] which matches the
        // step-edge case.
        assert_eq!(half_pel_b(&vop, 3, 0, 0, 8), 128);
        assert_eq!(half_pel_b(&vop, 3, 0, 1, 8), 127);
    }

    // ───────────── interpolate_quarter_pixel ────────────────────────

    #[test]
    fn quarter_pixel_integer_position_matches_plane_sample() {
        // qfrac == (0, 0) returns the plane sample at (int_x,
        // int_y) directly.
        let mut buf = [0u8; 8 * 8];
        for j in 0..8 {
            for i in 0..8 {
                buf[j * 8 + i] = (i + j * 16) as u8;
            }
        }
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        for j in 0i32..8 {
            for i in 0i32..8 {
                assert_eq!(
                    interpolate_quarter_pixel(&vop, i, j, 0, 0, 0, 8),
                    buf[(j as usize) * 8 + (i as usize)],
                );
            }
        }
    }

    #[test]
    fn quarter_pixel_half_pel_b_matches_half_pel_b_helper() {
        // qfrac == (2, 0) is the half-pel `b`.
        let buf = [42u8; 8 * 8];
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        for j in 0i32..8 {
            for i in 0i32..7 {
                assert_eq!(
                    interpolate_quarter_pixel(&vop, i, j, 2, 0, 0, 8),
                    half_pel_b(&vop, i, j, 0, 8),
                );
            }
        }
    }

    #[test]
    fn quarter_pixel_half_pel_c_matches_half_pel_c_helper() {
        // qfrac == (0, 2) is the half-pel `c`.
        let buf = [42u8; 8 * 8];
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        assert_eq!(
            interpolate_quarter_pixel(&vop, 3, 3, 0, 2, 0, 8),
            half_pel_c(&vop, 3, 3, 0, 8),
        );
    }

    #[test]
    fn quarter_pixel_half_pel_d_matches_half_pel_d_helper() {
        // qfrac == (2, 2) is the half-pel `d`.
        let buf = [42u8; 8 * 8];
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        assert_eq!(
            interpolate_quarter_pixel(&vop, 3, 3, 2, 2, 0, 8),
            half_pel_d(&vop, 3, 3, 0, 8),
        );
    }

    #[test]
    fn quarter_pixel_flat_plane_reproduces_constant_for_all_subpel() {
        // Property: every sub-pel position on a flat reference
        // plane returns the constant value. The FIR coefficients
        // sum to 256, so (256*v + 128 - rc)/256 = v for any
        // rc ∈ {0,1} and v ∈ [0, 254] (and v = 255 with rc = 0
        // gives 255 exactly; rc = 1 gives 254).
        let buf = [100u8; 16 * 16];
        let vop = ReferenceVop::new(&buf, 16, 16).unwrap();
        for qy in 0u8..=3 {
            for qx in 0u8..=3 {
                for &rc in &[0u8, 1] {
                    let v = interpolate_quarter_pixel(&vop, 5, 5, qx, qy, rc, 8);
                    assert_eq!(v, 100, "qx={qx} qy={qy} rc={rc}");
                }
            }
        }
    }

    #[test]
    fn quarter_pixel_clamping_at_negative_origin() {
        // §7.6.4 clamp: motion vector landing the anchor at (-10,
        // -10) should still produce a defined value.
        let buf = [77u8; 8 * 8];
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        for qx in 0u8..=3 {
            for qy in 0u8..=3 {
                let v = interpolate_quarter_pixel(&vop, -10, -10, qx, qy, 0, 8);
                assert_eq!(v, 77, "qx={qx} qy={qy}");
            }
        }
    }

    #[test]
    fn quarter_pixel_rc_changes_subpel_tie() {
        // 2-wide plane (a=100, b=101) at int=0, int_y=0; the
        // horizontal-half-pel b averages to 100.5; rc=0 rounds up
        // (101) and rc=1 rounds down (100). The 8-tap FIR over
        // clamped neighbours gives the same answer because all
        // far taps are clamped to one of the two values, and the
        // symmetric coefficient sum is 256.
        let buf = [100u8, 101];
        let vop = ReferenceVop::new(&buf, 2, 1).unwrap();
        // FIR with rc = 0: each tap is clamped to either 100 or
        // 101; left half (4 taps) sees 100, right half sees 101.
        // acc = 160*(100+101) + (-48)*(100+101) + 24*(100+101)
        //     + (-8)*(100+101) = 201 * 128 = 25728
        // (25728 + 128 - 0)/256 = 25856/256 = 101
        // (25728 + 128 - 1)/256 = 25855/256 = 100.99... → 100.
        let rc0 = interpolate_quarter_pixel(&vop, 0, 0, 2, 0, 0, 8);
        let rc1 = interpolate_quarter_pixel(&vop, 0, 0, 2, 0, 1, 8);
        assert_eq!(rc0, 101);
        assert_eq!(rc1, 100);
    }

    // ────────────────── interpolate_block_qpel ───────────────────────

    #[test]
    fn block_qpel_integer_mv_copies_reference() {
        // (mv_x, mv_y) = (4, 4) is +1, +1 full pel — i.e. shift
        // the reference by (1, 1). Compare a 4×4 block at origin
        // (0, 0) to the reference [(0..4) × (0..4)] but shifted.
        let mut buf = [0u8; 8 * 8];
        for j in 0..8 {
            for i in 0..8 {
                buf[j * 8 + i] = (10 * j + i) as u8;
            }
        }
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        let block = interpolate_block_qpel(&vop, 4, 4, 0, 0, 4, 4, 0, 8);
        for j in 0..4 {
            for i in 0..4 {
                // The pixel at (i, j) in the predicted block reads
                // from reference (origin + (i, j) + integer-mv).
                // integer-mv from mv = 4 is (1, 1) per
                // split_quarter_pel(4) = (1, 0).
                assert_eq!(block[j * 4 + i], buf[(j + 1) * 8 + (i + 1)]);
            }
        }
    }

    #[test]
    fn block_qpel_zero_mv_reproduces_reference_rectangle() {
        let mut buf = [0u8; 8 * 8];
        for j in 0..8 {
            for i in 0..8 {
                buf[j * 8 + i] = (i * 31 + j * 7) as u8;
            }
        }
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        let block = interpolate_block_qpel(&vop, 0, 0, 0, 0, 4, 4, 0, 8);
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(block[j * 4 + i], buf[j * 8 + i]);
            }
        }
    }

    #[test]
    fn block_qpel_into_panics_on_short_buffer() {
        let buf = [0u8; 4];
        let vop = ReferenceVop::new(&buf, 2, 2).unwrap();
        let mut out = [0u8; 3];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            interpolate_block_qpel_into(&vop, 0, 0, 0, 0, 2, 2, 0, 8, &mut out);
        }));
        assert!(r.is_err());
    }

    #[test]
    fn block_qpel_flat_reference_reproduces_constant() {
        let buf = [55u8; 64];
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        for &rc in &[0u8, 1] {
            for &mvx in &[0i32, 1, 2, 3, -1, -2] {
                for &mvy in &[0i32, 1, 2, 3, -1, -2] {
                    let out = interpolate_block_qpel(&vop, mvx, mvy, 0, 0, 4, 4, rc, 8);
                    assert!(out.iter().all(|&v| v == 55), "mv=({mvx},{mvy}) rc={rc}");
                }
            }
        }
    }

    #[test]
    fn block_qpel_8x8_block_inside_reference_no_clamp() {
        // Bigger block (the OBMC block size) — 8×8 from a 16×16
        // flat reference, mv = (1, 1) quarter-pel. The quarter-pel
        // result on a flat reference is still flat.
        let buf = [123u8; 16 * 16];
        let vop = ReferenceVop::new(&buf, 16, 16).unwrap();
        let block = interpolate_block_qpel(&vop, 1, 1, 4, 4, 8, 8, 0, 8);
        assert!(block.iter().all(|&v| v == 123));
    }

    // ───────────────── h-only/v-only consistency ────────────────────

    #[test]
    fn horizontal_quarter_collapses_to_half_at_qx2() {
        // qy = 0, qx = 2 must equal half_pel_b — already tested
        // above, but also verify that qx = 1 sits between integer
        // and half (i.e. less than half and not equal to integer)
        // on a horizontal gradient.
        let row = [0u8, 32, 64, 96, 128, 160, 192, 224];
        let mut buf = [0u8; 8 * 8];
        for r in 0..8 {
            buf[r * 8..(r + 1) * 8].copy_from_slice(&row);
        }
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        let int_x = 3;
        let int_y = 3;
        let integer = interpolate_quarter_pixel(&vop, int_x, int_y, 0, 0, 0, 8);
        let q1 = interpolate_quarter_pixel(&vop, int_x, int_y, 1, 0, 0, 8);
        let q2 = interpolate_quarter_pixel(&vop, int_x, int_y, 2, 0, 0, 8);
        let q3 = interpolate_quarter_pixel(&vop, int_x, int_y, 3, 0, 0, 8);
        // Monotonic on a monotone-increasing horizontal ramp.
        assert!(
            integer <= q1 && q1 <= q2 && q2 <= q3,
            "monotonic property: ({integer}, {q1}, {q2}, {q3})",
        );
    }
}

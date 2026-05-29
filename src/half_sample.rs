//! §7.6.2.1 Half-sample mode bilinear interpolation (Figure 7-29).
//!
//! In half-sample motion-compensation mode (`quarter_sample == 0`),
//! sub-integer reference samples are computed by bilinear interpolation
//! of the four integer-pel neighbours. With the integer-pel grid
//! labelled
//!
//! ```text
//!       A   B
//!         a b
//!       C   D
//!         c d
//! ```
//!
//! ISO/IEC 14496-2:2004 (3rd edition) §7.6.2.1 Figure 7-29 defines:
//!
//! ```text
//!   a = A,
//!   b = (A + B + 1 - rounding_control) / 2,
//!   c = (A + C + 1 - rounding_control) / 2,
//!   d = (A + B + C + D + 2 - rounding_control) / 4
//! ```
//!
//! where the divisions are integer (`/` per the spec's §3.4 division
//! convention, which for non-negative integer operands matches
//! `floor`), and `rounding_control ∈ {0, 1}` is supplied by the VOP
//! header field `vop_rounding_type` (§6.3.5; defaults to `0` when
//! absent — i.e. for I-VOPs and B-VOPs the header doesn't carry the
//! bit and the value is `0`).
//!
//! ## Per-pixel positions
//!
//! Every reference sample requested by a motion-compensated block fetch
//! lands on one of four sub-pel positions per integer cell, indexed by
//! the half-pel fraction of the motion vector:
//!
//! * `(half_x, half_y) == (0, 0)` — `a`, the integer-pel sample `A`.
//! * `(1, 0)` — `b`, horizontal half-pel.
//! * `(0, 1)` — `c`, vertical half-pel.
//! * `(1, 1)` — `d`, diagonal half-pel.
//!
//! The §7.6.2 motion-vector representation is "half-sample units" —
//! one MV component value `mv_h` decomposes into integer part
//! `mv_h >> 1` and half-pel fraction `mv_h & 1`. A negative MV uses
//! Rust's arithmetic shift to keep the integer part rounded toward
//! `-∞` (the spec's `floor` convention) while the fractional bit is
//! taken from the two's-complement low bit, exactly matching the
//! §3.4 division. (See [`split_half_pel`].)
//!
//! ## §7.6.4 unrestricted-MC edge clamping
//!
//! When a reference-sample fetch lands outside the decoded VOP area,
//! §7.6.4 dictates that the last full pel inside the area is used
//! ("edge sample"). [`fetch_clamped`] applies this rule against a
//! caller-supplied `ReferenceVop` rectangle: any integer sample
//! coordinate `(x, y)` is clipped to `[0, width-1] × [0, height-1]`
//! before the fetch. The clipping is performed *per component*
//! (Figure 7-33), which matches the natural `x.clamp() / y.clamp()`
//! behaviour. Short-video-header streams forbid out-of-area motion
//! vectors per §7.6.4, but the clamp is unconditional — it's a no-op
//! when the MV stays inside, and there's no §7.6.4-conformant way for
//! the half-sample stage to observe a short-video-header violation
//! anyway.
//!
//! ## Out of scope (this round)
//!
//! * §7.6.2.2 quarter-sample mode — the 8-tap FIR filter with
//!   `C = [160, -48, 24, -8]` and the §7.6.2.2.2 quarter-pel bilinear
//!   step. Quarter-sample mode is a deeper round on its own.
//! * §7.6.1 reference-VOP padding — the caller hands us a fully
//!   reconstructed and padded reference plane. Padding is an upstream
//!   reconstruction-pipeline concern.
//! * §7.6.3 motion-vector decoding and §7.6.5 predictor — already
//!   landed in `motion.rs`. This module assumes a finalised
//!   `MotionVector` in half-pel units.
//! * Interlaced field-based motion compensation. §7.6.2's prose notes
//!   "for interlaced video, the half and quarter sample values are
//!   vertically interpolated between two successive lines of the same
//!   field" — that needs a field-aware sample fetcher, deferred.

/// Split a half-pel motion-vector component into the integer-pel
/// offset and the half-pel fractional bit.
///
/// `mv` is in §7.6.3 half-sample units (so `mv = 1` means a half-pel
/// motion). The integer part is `mv >> 1` (arithmetic shift, rounds
/// toward `-∞`); the half-pel fraction is the bit `mv & 1` viewed as
/// a boolean. The split satisfies `mv == (integer << 1) | fraction`
/// for non-negative MVs and `mv == (integer << 1) + fraction`
/// generally (with the integer adjusted to keep the fraction in
/// `{0, 1}`).
///
/// # Examples
///
/// ```
/// use oxideav_mpeg4video::half_sample::split_half_pel;
/// assert_eq!(split_half_pel(0), (0, false));
/// assert_eq!(split_half_pel(1), (0, true));
/// assert_eq!(split_half_pel(2), (1, false));
/// assert_eq!(split_half_pel(-1), (-1, true));
/// assert_eq!(split_half_pel(-2), (-1, false));
/// ```
#[inline]
pub const fn split_half_pel(mv: i32) -> (i32, bool) {
    // Arithmetic shift right: -1 >> 1 == -1, -2 >> 1 == -1, etc.
    // The fractional bit is the LSB viewed as two's complement:
    // -1 = 0xFFFFFFFF → LSB = 1 → fraction true.
    let integer = mv >> 1;
    let fraction = (mv & 1) != 0;
    (integer, fraction)
}

/// Evaluate the §7.6.2.1 / Figure 7-29 half-sample interpolation for
/// one pixel given the four integer-pel neighbours `A`, `B`, `C`, `D`
/// (top-left, top-right, bottom-left, bottom-right respectively), the
/// half-pel selection `(half_x, half_y)`, and `rounding_control`.
///
/// `rounding_control` is the VOP-header `vop_rounding_type` bit
/// (§6.3.5; `0` or `1` only). Any other value is masked to its low bit.
///
/// The four cases:
///
/// | `(half_x, half_y)` | Returns | Spec equation                                        |
/// | ------------------ | ------- | ---------------------------------------------------- |
/// | `(false, false)`   | `A`     | `a = A`                                              |
/// | `(true,  false)`   | `b`     | `(A + B + 1 - rc) / 2`                               |
/// | `(false, true)`    | `c`     | `(A + C + 1 - rc) / 2`                               |
/// | `(true,  true)`    | `d`     | `(A + B + C + D + 2 - rc) / 4`                       |
///
/// All arithmetic fits in `u32` for the legal `[0, 255]` sample range:
/// `255 * 4 + 2 = 1022 < 2^11`. The integer divisions are exact —
/// `floor`-equivalent for non-negative operands — and the returned
/// value is in `[0, 255]` (bounded by max neighbour, never exceeds it).
///
/// # Examples
///
/// ```
/// use oxideav_mpeg4video::half_sample::interpolate_pixel;
/// // Flat reference — all four samples = 100 — reproduces 100 at
/// // every sub-pel position, regardless of rounding_control.
/// for &rc in &[0u8, 1] {
///     for &hx in &[false, true] {
///         for &hy in &[false, true] {
///             assert_eq!(interpolate_pixel(100, 100, 100, 100, hx, hy, rc), 100);
///         }
///     }
/// }
/// // Horizontal half-pel between 100 and 102 — averages to 101 (no
/// // rounding tie).
/// assert_eq!(interpolate_pixel(100, 102, 0, 0, true, false, 0), 101);
/// assert_eq!(interpolate_pixel(100, 102, 0, 0, true, false, 1), 101);
/// // Tie between 100 and 101 — rc=0 rounds up to 101, rc=1 rounds
/// // down to 100 (the §7.6.2.1 rounding-control rule).
/// assert_eq!(interpolate_pixel(100, 101, 0, 0, true, false, 0), 101);
/// assert_eq!(interpolate_pixel(100, 101, 0, 0, true, false, 1), 100);
/// ```
#[inline]
pub fn interpolate_pixel(
    a: u8,
    b: u8,
    c: u8,
    d: u8,
    half_x: bool,
    half_y: bool,
    rounding_control: u8,
) -> u8 {
    let rc = (rounding_control & 1) as u32;
    let a = a as u32;
    let b = b as u32;
    let c = c as u32;
    let d = d as u32;
    let value = match (half_x, half_y) {
        (false, false) => a,
        (true, false) => (a + b + 1 - rc) / 2,
        (false, true) => (a + c + 1 - rc) / 2,
        (true, true) => (a + b + c + d + 2 - rc) / 4,
    };
    // Sample values fit in `[0, 255]` by construction (max of four
    // u8 values plus a +2 / -1 adjustment, divided by 4 / 2). The
    // upper bound: (255 * 4 + 2) / 4 == 255. The lower bound: rc=1
    // takes the (-1) from a +0 sum but `(0 - 1) / 2` in our u32 land
    // would wrap; the smallest legal sum is therefore `b + 0 ≥ 0`
    // and we have `1 - rc ∈ {0, 1}` always non-negative.
    debug_assert!(value <= 255);
    value as u8
}

/// A rectangular reference-VOP plane viewed as a clamped sample
/// fetcher.
///
/// The plane is in raster order, `width * height` `u8` samples, with
/// the stride matching `width` (callers with a row-padded buffer can
/// use [`ReferenceVop::with_stride`]). All sample reads via
/// [`ReferenceVop::fetch_clamped`] apply the §7.6.4 last-full-pel
/// clamp: integer coordinates outside the rectangle are clipped to
/// the nearest in-rectangle coordinate per component, matching
/// Figure 7-33.
///
/// `width` and `height` must be `>= 1` and small enough that the
/// product fits in `samples.len()` with the stride taken into account.
/// [`ReferenceVop::new`] checks the invariant.
#[derive(Debug, Clone, Copy)]
pub struct ReferenceVop<'a> {
    samples: &'a [u8],
    width: i32,
    height: i32,
    stride: i32,
}

impl<'a> ReferenceVop<'a> {
    /// Create a reference plane with stride equal to width.
    ///
    /// Returns `None` if `width == 0`, `height == 0`, or
    /// `samples.len() < width * height`.
    pub fn new(samples: &'a [u8], width: usize, height: usize) -> Option<Self> {
        Self::with_stride(samples, width, height, width)
    }

    /// Create a reference plane with an explicit row stride. The
    /// stride must be `>= width` and the buffer must contain at least
    /// `(height - 1) * stride + width` samples.
    ///
    /// Returns `None` on any invariant violation.
    pub fn with_stride(
        samples: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Option<Self> {
        if width == 0 || height == 0 || stride < width {
            return None;
        }
        // Bounds: last byte read is `(height-1)*stride + (width-1)`.
        let last = height.checked_sub(1)?.checked_mul(stride)?;
        let last = last.checked_add(width.checked_sub(1)?)?;
        if last >= samples.len() {
            return None;
        }
        // Bound the cast to i32 — the half-sample arithmetic uses i32
        // to permit negative MVs. We disallow planes larger than
        // `i32::MAX / 4` to keep the clamp arithmetic comfortably
        // away from overflow (in practice MPEG-4 Part 2 maxes at
        // 4096 × 4096 — well under).
        if width > (i32::MAX as usize) / 4
            || height > (i32::MAX as usize) / 4
            || stride > (i32::MAX as usize) / 4
        {
            return None;
        }
        Some(Self {
            samples,
            width: width as i32,
            height: height as i32,
            stride: stride as i32,
        })
    }

    /// Plane width in samples.
    #[inline]
    pub const fn width(&self) -> usize {
        self.width as usize
    }

    /// Plane height in samples.
    #[inline]
    pub const fn height(&self) -> usize {
        self.height as usize
    }

    /// Read one integer-pel sample at `(x, y)` with §7.6.4 last-full-
    /// pel clamping. `x` and `y` are signed so callers can pass MV
    /// components directly.
    #[inline]
    pub fn fetch_clamped(&self, x: i32, y: i32) -> u8 {
        let cx = x.clamp(0, self.width - 1);
        let cy = y.clamp(0, self.height - 1);
        let idx = (cy as usize) * (self.stride as usize) + (cx as usize);
        self.samples[idx]
    }
}

/// Interpolate one sub-pel sample at `(int_x + half_x/2, int_y +
/// half_y/2)` from a reference plane with §7.6.4 edge clamping.
///
/// `int_x` and `int_y` are the integer-pel position of the `A`
/// neighbour (top-left); `half_x` and `half_y` are the half-pel
/// fraction bits (typically obtained from [`split_half_pel`]).
/// `rounding_control` is the VOP-header `vop_rounding_type`.
///
/// Calls [`ReferenceVop::fetch_clamped`] for each of the up to four
/// neighbour fetches needed by the §7.6.2.1 equation: just `A` for
/// the integer-pel case, `A` + `B` (or `A` + `C`) for the
/// half-pel-cardinal case, and all four for the diagonal case.
pub fn fetch_clamped_sample(
    vop: &ReferenceVop<'_>,
    int_x: i32,
    int_y: i32,
    half_x: bool,
    half_y: bool,
    rounding_control: u8,
) -> u8 {
    let a = vop.fetch_clamped(int_x, int_y);
    let b = if half_x {
        vop.fetch_clamped(int_x + 1, int_y)
    } else {
        a
    };
    let c = if half_y {
        vop.fetch_clamped(int_x, int_y + 1)
    } else {
        a
    };
    let d = if half_x && half_y {
        vop.fetch_clamped(int_x + 1, int_y + 1)
    } else {
        a
    };
    interpolate_pixel(a, b, c, d, half_x, half_y, rounding_control)
}

/// Half-sample-interpolate an entire `block_w × block_h` prediction
/// block from a reference plane, given an `(mv_x, mv_y)` motion
/// vector in §7.6.3 half-sample units and the block's top-left
/// pixel origin `(origin_x, origin_y)` in the *current* (predicted)
/// VOP.
///
/// The output is laid out row-major as `block[j][i] = block_w * j +
/// i` and has length `block_w * block_h`.
///
/// `mv_x` / `mv_y` are signed; `vop_rounding_type ∈ {0, 1}` is the
/// VOP-header field. §7.6.4 edge clamping is applied per sample fetch
/// via [`ReferenceVop::fetch_clamped`].
///
/// Returns a freshly allocated `Vec<u8>`. Callers that wish to reuse
/// a buffer should prefer [`interpolate_block_into`].
//
// The signature reflects the §7.6.2.1 inputs one-for-one (motion
// vector x/y, block origin x/y, block w/h, plus the VOP-level
// rounding_control bit). Bundling them into a struct would add a
// shim layer without simplifying the call site.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_block(
    vop: &ReferenceVop<'_>,
    mv_x: i32,
    mv_y: i32,
    origin_x: i32,
    origin_y: i32,
    block_w: usize,
    block_h: usize,
    vop_rounding_type: u8,
) -> Vec<u8> {
    let mut out = vec![0u8; block_w * block_h];
    interpolate_block_into(
        vop,
        mv_x,
        mv_y,
        origin_x,
        origin_y,
        block_w,
        block_h,
        vop_rounding_type,
        &mut out,
    );
    out
}

/// Half-sample-interpolate an entire `block_w × block_h` prediction
/// block into a caller-supplied buffer of length `block_w * block_h`.
///
/// See [`interpolate_block`] for the parameter semantics.
///
/// # Panics
///
/// Panics if `out.len() < block_w * block_h`.
//
// Same justification as [`interpolate_block`]: each argument maps
// directly to a §7.6.2.1 input. The buffer-out variant adds one
// `&mut [u8]` parameter for the caller-supplied output.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_block_into(
    vop: &ReferenceVop<'_>,
    mv_x: i32,
    mv_y: i32,
    origin_x: i32,
    origin_y: i32,
    block_w: usize,
    block_h: usize,
    vop_rounding_type: u8,
    out: &mut [u8],
) {
    assert!(
        out.len() >= block_w * block_h,
        "interpolate_block_into: output buffer too small ({} < {} * {})",
        out.len(),
        block_w,
        block_h,
    );
    let (mvx_int, half_x) = split_half_pel(mv_x);
    let (mvy_int, half_y) = split_half_pel(mv_y);
    let rc = vop_rounding_type & 1;
    for j in 0..block_h {
        for i in 0..block_w {
            let int_x = origin_x + (i as i32) + mvx_int;
            let int_y = origin_y + (j as i32) + mvy_int;
            out[j * block_w + i] = fetch_clamped_sample(vop, int_x, int_y, half_x, half_y, rc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_half_pel_basic_cases() {
        assert_eq!(split_half_pel(0), (0, false));
        assert_eq!(split_half_pel(1), (0, true));
        assert_eq!(split_half_pel(2), (1, false));
        assert_eq!(split_half_pel(3), (1, true));
        assert_eq!(split_half_pel(-1), (-1, true));
        assert_eq!(split_half_pel(-2), (-1, false));
        assert_eq!(split_half_pel(-3), (-2, true));
        assert_eq!(split_half_pel(-4), (-2, false));
    }

    #[test]
    fn split_half_pel_reconstructs_mv() {
        // For every MV in [-32, 32], (integer * 2) + fraction == mv.
        for mv in -32..=32i32 {
            let (int_part, frac) = split_half_pel(mv);
            let recon = int_part * 2 + (frac as i32);
            assert_eq!(recon, mv, "mv = {mv}");
        }
    }

    #[test]
    fn integer_pel_returns_a() {
        // half_x = false, half_y = false → returns A unchanged.
        for rc in [0u8, 1] {
            for a in (0..=255u8).step_by(17) {
                assert_eq!(interpolate_pixel(a, 13, 27, 41, false, false, rc), a);
            }
        }
    }

    #[test]
    fn horizontal_half_pel_no_tie() {
        // (100 + 102 + 1 - rc) / 2 = 101 / 101 for rc = 1 / 0.
        assert_eq!(interpolate_pixel(100, 102, 0, 0, true, false, 0), 101);
        assert_eq!(interpolate_pixel(100, 102, 0, 0, true, false, 1), 101);
    }

    #[test]
    fn horizontal_half_pel_tie_rounding() {
        // (100 + 101 + 1 - rc) / 2 = 101 (rc=0), 100 (rc=1).
        assert_eq!(interpolate_pixel(100, 101, 0, 0, true, false, 0), 101);
        assert_eq!(interpolate_pixel(100, 101, 0, 0, true, false, 1), 100);
    }

    #[test]
    fn vertical_half_pel_tie_rounding() {
        // c = (A + C + 1 - rc) / 2 with A=50, C=51.
        assert_eq!(interpolate_pixel(50, 0, 51, 0, false, true, 0), 51);
        assert_eq!(interpolate_pixel(50, 0, 51, 0, false, true, 1), 50);
    }

    #[test]
    fn diagonal_half_pel_rounding() {
        // d = (A + B + C + D + 2 - rc) / 4.
        // 10 + 10 + 11 + 11 + 2 - 0 = 44 → 11
        // 10 + 10 + 11 + 11 + 2 - 1 = 43 → 10 (floor)
        assert_eq!(interpolate_pixel(10, 10, 11, 11, true, true, 0), 11);
        assert_eq!(interpolate_pixel(10, 10, 11, 11, true, true, 1), 10);
    }

    #[test]
    fn diagonal_half_pel_saturates_at_255() {
        // (255 * 4 + 2 - rc) / 4 = 255 for either rc.
        assert_eq!(interpolate_pixel(255, 255, 255, 255, true, true, 0), 255);
        assert_eq!(interpolate_pixel(255, 255, 255, 255, true, true, 1), 255);
    }

    #[test]
    fn diagonal_half_pel_zero_floor() {
        // (0 + 0 + 0 + 0 + 2 - rc) / 4 = 0 for either rc (floor).
        assert_eq!(interpolate_pixel(0, 0, 0, 0, true, true, 0), 0);
        assert_eq!(interpolate_pixel(0, 0, 0, 0, true, true, 1), 0);
    }

    #[test]
    fn rounding_control_only_affects_subpel() {
        // (false, false) ignores rc entirely.
        assert_eq!(interpolate_pixel(7, 0, 0, 0, false, false, 0), 7);
        assert_eq!(interpolate_pixel(7, 0, 0, 0, false, false, 1), 7);
    }

    #[test]
    fn ref_vop_new_rejects_zero_dims() {
        let buf = [0u8; 16];
        assert!(ReferenceVop::new(&buf, 0, 4).is_none());
        assert!(ReferenceVop::new(&buf, 4, 0).is_none());
    }

    #[test]
    fn ref_vop_new_rejects_short_buffer() {
        let buf = [0u8; 8];
        // Needs 4*4 = 16 samples; only 8 supplied.
        assert!(ReferenceVop::new(&buf, 4, 4).is_none());
    }

    #[test]
    fn ref_vop_stride_must_be_ge_width() {
        let buf = [0u8; 16];
        assert!(ReferenceVop::with_stride(&buf, 4, 4, 3).is_none());
    }

    #[test]
    fn ref_vop_fetch_clamped_in_range() {
        // 4x4 plane filled with row * 16 + col.
        let mut buf = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                buf[r * 4 + c] = (r * 16 + c) as u8;
            }
        }
        let vop = ReferenceVop::new(&buf, 4, 4).unwrap();
        assert_eq!(vop.fetch_clamped(0, 0), 0);
        assert_eq!(vop.fetch_clamped(3, 3), 3 * 16 + 3);
        assert_eq!(vop.fetch_clamped(1, 2), 2 * 16 + 1);
    }

    #[test]
    fn ref_vop_fetch_clamped_outside_clamps_per_component() {
        let mut buf = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                buf[r * 4 + c] = (r * 16 + c) as u8;
            }
        }
        let vop = ReferenceVop::new(&buf, 4, 4).unwrap();
        // x < 0 → x = 0; y < 0 → y = 0.
        assert_eq!(vop.fetch_clamped(-5, -10), 0);
        // x > width-1 → x = width-1; y unchanged.
        assert_eq!(vop.fetch_clamped(10, 2), 2 * 16 + 3);
        // y > height-1 → y = height-1; x unchanged.
        assert_eq!(vop.fetch_clamped(1, 10), 3 * 16 + 1);
        // Both out of range, same corner.
        assert_eq!(vop.fetch_clamped(10, 10), 3 * 16 + 3);
    }

    #[test]
    fn ref_vop_with_stride_padded_rows() {
        // 3x2 plane with stride 5 — rows padded with garbage.
        let buf = [
            10, 11, 12, 99, 99, // row 0
            20, 21, 22, 99, 99, // row 1
        ];
        let vop = ReferenceVop::with_stride(&buf, 3, 2, 5).unwrap();
        assert_eq!(vop.fetch_clamped(0, 0), 10);
        assert_eq!(vop.fetch_clamped(2, 1), 22);
        // Clamps past width to in-bounds column 2 (not the pad byte).
        assert_eq!(vop.fetch_clamped(7, 0), 12);
    }

    #[test]
    fn fetch_clamped_sample_integer_pel() {
        let buf: Vec<u8> = (0..16).collect();
        let vop = ReferenceVop::new(&buf, 4, 4).unwrap();
        // Integer-pel == direct fetch_clamped.
        for y in 0..4i32 {
            for x in 0..4i32 {
                assert_eq!(
                    fetch_clamped_sample(&vop, x, y, false, false, 0),
                    vop.fetch_clamped(x, y),
                );
            }
        }
    }

    #[test]
    fn fetch_clamped_sample_diagonal_inside_block() {
        // 2x2 plane:
        //   10 12
        //   14 16
        let buf = [10, 12, 14, 16];
        let vop = ReferenceVop::new(&buf, 2, 2).unwrap();
        // Diagonal at (0,0) integer position: (10+12+14+16+2-rc)/4.
        // rc=0: 54/4 = 13. rc=1: 53/4 = 13.
        assert_eq!(fetch_clamped_sample(&vop, 0, 0, true, true, 0), 13);
        assert_eq!(fetch_clamped_sample(&vop, 0, 0, true, true, 1), 13);
    }

    #[test]
    fn interpolate_block_integer_pel_zero_mv() {
        // 8x8 ramp.
        let mut buf = [0u8; 64];
        for r in 0..8 {
            for c in 0..8 {
                buf[r * 8 + c] = (r * 8 + c) as u8;
            }
        }
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        let out = interpolate_block(&vop, 0, 0, 0, 0, 8, 8, 0);
        assert_eq!(out, buf);
    }

    #[test]
    fn interpolate_block_horizontal_half_pel() {
        // 4-wide plane: 0, 2, 4, 6. Horizontal half-pel between
        // columns shifts the average column by 0.5. With rc=0:
        // (0+2+1)/2=1, (2+4+1)/2=3, (4+6+1)/2=5 — and the last
        // sample clamps to col=3.
        let buf = [0u8, 2, 4, 6];
        let vop = ReferenceVop::new(&buf, 4, 1).unwrap();
        // MV (1, 0) — one half-pel right. block 4 wide, origin (0, 0).
        let out = interpolate_block(&vop, 1, 0, 0, 0, 4, 1, 0);
        // Col 0: A=buf[0]=0, B=buf[1]=2 → 1.
        // Col 1: A=buf[1]=2, B=buf[2]=4 → 3.
        // Col 2: A=buf[2]=4, B=buf[3]=6 → 5.
        // Col 3: A=buf[3]=6, B=clamp(buf[4])=buf[3]=6 → (6+6+1)/2=6.
        assert_eq!(out, vec![1, 3, 5, 6]);
    }

    #[test]
    fn interpolate_block_negative_mv_clamps_to_edge() {
        let buf = [10u8, 20, 30, 40];
        let vop = ReferenceVop::new(&buf, 4, 1).unwrap();
        // MV (-2, 0) — one integer pel left. Origin (0, 0).
        // Col 0: int_x = -1 → clamps to 0 → 10.
        // Col 1: int_x = 0 → 10.
        // Col 2: int_x = 1 → 20.
        // Col 3: int_x = 2 → 30.
        let out = interpolate_block(&vop, -2, 0, 0, 0, 4, 1, 0);
        assert_eq!(out, vec![10, 10, 20, 30]);
    }

    #[test]
    fn interpolate_block_rounding_control_changes_subpel_tie() {
        // 2-wide plane 100, 101. Origin (0, 0). MV (1, 0).
        let buf = [100u8, 101];
        let vop = ReferenceVop::new(&buf, 2, 1).unwrap();
        let rc0 = interpolate_block(&vop, 1, 0, 0, 0, 1, 1, 0);
        let rc1 = interpolate_block(&vop, 1, 0, 0, 0, 1, 1, 1);
        assert_eq!(rc0, vec![101]);
        assert_eq!(rc1, vec![100]);
    }

    #[test]
    fn interpolate_block_into_panics_on_short_buffer() {
        let buf = [0u8; 4];
        let vop = ReferenceVop::new(&buf, 2, 2).unwrap();
        let mut out = [0u8; 3];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            interpolate_block_into(&vop, 0, 0, 0, 0, 2, 2, 0, &mut out);
        }));
        assert!(r.is_err());
    }

    #[test]
    fn interpolate_block_into_reuses_buffer() {
        let buf = [5u8, 5, 5, 5];
        let vop = ReferenceVop::new(&buf, 2, 2).unwrap();
        let mut out = [99u8; 4];
        interpolate_block_into(&vop, 0, 0, 0, 0, 2, 2, 0, &mut out);
        assert_eq!(out, [5, 5, 5, 5]);
    }

    #[test]
    fn interpolate_block_flat_reference_reproduces_constant() {
        // §7.6.2.1 property: a flat reference plane reproduces the
        // constant value at every sub-pel position, for every
        // rounding_control. The four formulas all evaluate to C
        // when A=B=C=D=C (`a=C`, `b=(2C+1-rc)/2≤C`, `c` similarly,
        // `d=(4C+2-rc)/4=C`). The diagonal case is the tightest:
        // (4C+2-rc)/4 = C + (2-rc)/4 — for C ≤ 254 the result is C.
        let buf = [42u8; 64];
        let vop = ReferenceVop::new(&buf, 8, 8).unwrap();
        for &rc in &[0u8, 1] {
            for &mvx in &[0i32, 1, -1, 3] {
                for &mvy in &[0i32, 1, -1, 3] {
                    let out = interpolate_block(&vop, mvx, mvy, 0, 0, 8, 8, rc);
                    assert!(out.iter().all(|&v| v == 42), "mv=({mvx},{mvy}) rc={rc}");
                }
            }
        }
    }
}

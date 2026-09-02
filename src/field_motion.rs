//! §7.7.2.1 field-based motion-compensated reconstruction for
//! interlaced P-VOPs.
//!
//! When a P-VOP macroblock is field predicted (`field_prediction == 1`,
//! §6.2.6.3), its 16×16 luminance area is reconstructed from **two**
//! motion vectors — one per output field. The top output field (even
//! lines 0, 2, 4, …) is predicted by the top-field MV, the bottom output
//! field (odd lines 1, 3, 5, …) by the bottom-field MV. Each field MV
//! independently selects which **reference** field (top or bottom) it
//! draws from, via the `forward_top_field_reference` /
//! `forward_bottom_field_reference` flags (§6.3.7.2). The reference VOP
//! is a single progressive plane in which the even lines are the top
//! field and the odd lines are the bottom field (§7.7.2.1 final
//! paragraph).
//!
//! ## §7.7.2.1 final motion-vector reconstruction
//!
//! The decoded differentials `(MVDx f1, MVDy f1)` / `(MVDx f2, MVDy f2)`
//! (the [`crate::motion::FieldMvPair`] top/bottom bodies) and the single
//! shared predictor `(Px, Py)` reconstruct the two field motion vectors:
//!
//! ```text
//!   MVx f1 = MVDx f1 + Px
//!   MVy f1 = 2 * (MVDy f1 + (Py / 2))
//!   MVx f2 = MVDx f2 + Px
//!   MVy f2 = 2 * (MVDy f2 + (Py / 2))
//! ```
//!
//! where `/` is integer division with truncation toward 0 (§7.7.2.1).
//! The vertical component of a field motion vector is therefore always
//! even in half-pel frame coordinates: vertical half-pel interpolation
//! happens between adjacent lines **of the same field**, which are two
//! frame lines apart. [`reconstruct_field_motion_vectors`] performs this
//! reconstruction.
//!
//! ## §7.7.2.1 `field_motion_compensate_one_reference`
//!
//! The prediction macroblock is then assembled by the spec's
//! `field_motion_compensate_one_reference` pseudo code, which issues six
//! [`mc`] calls: top-field luma, bottom-field luma, then top/bottom Cb
//! and Cr. The chrominance motion vectors are `Div2Round` of the luma
//! field MVs (§7.7.2.1 — [`div2_round`]). Each [`mc`] call uses
//! `y_incr = 2` so it writes only every other destination line, the
//! `pred_y0` field offset selecting top (0) or bottom (1).
//!
//! ## `mc` — the §7.7.2.1 / §7.6.2 half-sample reference routine
//!
//! [`mc`] reproduces the spec's `mc(pred, ref, x, y, width, height,
//! dx_halfpel, dy_halfpel, rounding, pred_y0, ref_y0, y_incr)` verbatim.
//! With `y_incr = 1` and `pred_y0 = ref_y0 = 0` it is the ordinary frame
//! half-sample fetch (equivalent to
//! [`crate::half_sample::interpolate_block_into`]); with `y_incr = 2` it
//! is the field fetch where the two vertically-averaged reference lines
//! are `y_ref` and `y_ref + 2` (the next line of the same reference
//! field).
//!
//! The §7.6.4 last-full-pel edge clamp is applied through
//! [`crate::half_sample::ReferenceVop`].
//!
//! ## §7.6.2.2 quarter-sample field MC
//!
//! When `quarter_sample == 1` the spec replaces the two luma [`mc`]
//! calls with the §7.6.2.2 quarter-pel interpolation "accordingly"
//! (§7.7.2.1, `field_motion_compensate_one_reference`), still on the
//! field reference grid (vertical neighbours are same-field lines).
//! [`field_motion_compensate_one_reference_qpel`] interpolates each
//! 16-wide field block as **two 8×8 blocks** (per-sub-block Figure
//! 7-30 mirroring — black-box-arbitrated, see the function docs) via
//! [`crate::quarter_sample::interpolate_block_qpel_field_into`] and
//! keeps the four [`mc`] chroma calls. Per §7.7.2.2 the quarter-sample
//! chroma vectors are `Div2Round` of the luma field MV **divided by
//! 2** (quarter → half): truncating on the horizontal component
//! ([`div2_round`]`(`[`half_pel_chroma_mv_from_qpel`]`)`), flooring on
//! the field-grid vertical component ([`field_chroma_dy_qpel`] —
//! probe-arbitrated), with chroma interpolated in half-sample mode
//! exactly as in the half-sample driver.
//!
//! ## Scope
//!
//! * The predictor `(Px, Py)` is supplied by the caller — the §7.7.2.1
//!   CASE 1 / 2 / 3 median selection over field-aware neighbour vectors
//!   is the motion-vector-prediction module's responsibility; this
//!   module consumes the finished predictor and the decoded
//!   differentials.

use crate::half_sample::ReferenceVop;
use crate::motion::{FieldMvPair, MotionVector};
use crate::quarter_sample::interpolate_block_qpel_field_into;
use crate::reconstruct::{InterPredictionMacroblock, MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};

/// `Div2Round(x) = (x >> 1) | (x & 1)` (§7.7.2.1).
///
/// Halves a motion-vector component while keeping any half-pel residue —
/// the `| (x & 1)` term forces an odd input to round its fractional bit
/// up into the result's LSB rather than truncating it. Used both to
/// reconstruct the §7.7.2.1 chrominance field MVs from the luminance
/// field MVs and (in the CASE 2 / 3 predictor derivation, elsewhere) to
/// average two field MVs into a frame predictor candidate.
///
/// The shift is arithmetic (`>>` on `i32`), so negative inputs round
/// toward `-∞` before the `| (x & 1)` adjustment, matching the spec's
/// bit-level definition.
///
/// # Examples
///
/// ```
/// use oxideav_mpeg4video::field_motion::div2_round;
/// assert_eq!(div2_round(0), 0);
/// assert_eq!(div2_round(4), 2);
/// assert_eq!(div2_round(3), 1); // (3 >> 1) | (3 & 1) = 1 | 1 = 1
/// assert_eq!(div2_round(-3), -1); // (-2) | (1) = -1
/// ```
#[inline]
pub const fn div2_round(x: i32) -> i32 {
    (x >> 1) | (x & 1)
}

/// The two reconstructed field motion vectors of one prediction
/// direction (§7.7.2.1) — top-field `MV f1` and bottom-field `MV f2`,
/// both in half-pel frame coordinates with even vertical components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldMotionVectors {
    /// Top-field motion vector (`MVx f1`, `MVy f1`).
    pub top: MotionVector,
    /// Bottom-field motion vector (`MVx f2`, `MVy f2`).
    pub bottom: MotionVector,
}

/// Reconstruct the §7.7.2.1 field motion vectors from a decoded
/// [`FieldMvPair`] differential and the shared predictor `(px, py)`.
///
/// Both fields share the single predictor; the vertical component is
/// reconstructed as `2 * (MVDy + (Py / 2))` so it is always even in
/// half-pel frame coordinates (the predictor's vertical half is taken
/// in field coordinates, the differential added there, then doubled
/// back to frame coordinates). `/` is integer division truncating
/// toward 0, matching `i32::div_euclid` only for non-negative `py`; the
/// spec's truncation-toward-0 is reproduced with Rust's `/` operator
/// directly (which truncates toward 0).
pub fn reconstruct_field_motion_vectors(pair: FieldMvPair, px: i32, py: i32) -> FieldMotionVectors {
    // Py / 2 with truncation toward 0 — Rust's `/` already truncates
    // toward 0, matching §7.7.2.1's stated convention.
    let py_half = py / 2;
    let top = MotionVector {
        x: pair.top.dx + px,
        y: 2 * (pair.top.dy + py_half),
    };
    let bottom = MotionVector {
        x: pair.bottom.dx + px,
        y: 2 * (pair.bottom.dy + py_half),
    };
    FieldMotionVectors { top, bottom }
}

/// [`reconstruct_field_motion_vectors`] with the §7.6.3 modulo wrap
/// applied to each reconstructed component on its own grid — the
/// horizontal in frame units, the vertical in **field** units before
/// the doubling (`MVy = 2 * wrap(MVDy + Py / 2)`). §7.6.3 declares its
/// general decoding process (which includes the `[low:high]` wrap)
/// "valid for the motion vector decoding in interlaced/progressive
/// P-, S(GMC)- and B-VOPs except that the generation of the predictor
/// (Px, Py) may be different"; §7.7.2.1 supplies that predictor and
/// the field-grid halving/doubling. For a conformant stream every
/// component is already inside `[low:high]` and the wrap is a no-op.
pub fn reconstruct_field_motion_vectors_wrapped(
    pair: FieldMvPair,
    px: i32,
    py: i32,
    vop_fcode: u8,
) -> FieldMotionVectors {
    let f = 1i32 << (vop_fcode.clamp(1, 7) - 1);
    let (low, high, range) = (-32 * f, 32 * f - 1, 64 * f);
    let wrap = |mut v: i32| {
        if v < low {
            v += range;
        }
        if v > high {
            v -= range;
        }
        v
    };
    let py_half = py / 2;
    let top = MotionVector {
        x: wrap(pair.top.dx + px),
        y: 2 * wrap(pair.top.dy + py_half),
    };
    let bottom = MotionVector {
        x: wrap(pair.bottom.dx + px),
        y: 2 * wrap(pair.bottom.dy + py_half),
    };
    FieldMotionVectors { top, bottom }
}

/// The §7.7.2.1 / §7.6.2 half-sample motion-compensation routine `mc`.
///
/// Fills the `width × height` block of `pred` (a row-major plane of
/// `stride` width) anchored at destination column `x` with the
/// half-sample-interpolated reference samples, stepping the destination
/// rows by `y_incr` and offsetting by `pred_y0`. The reference is read
/// through `reference` with the §7.6.4 last-full-pel clamp.
///
/// Parameters map one-for-one to the spec's `mc(...)`:
///
/// * `x`, `y` — destination/reference block coordinates for `MV = (0,0)`.
/// * `width`, `height` — block dimensions (16×16 luma, 8×8 chroma).
/// * `dx_halfpel`, `dy_halfpel` — the half-pel-resolution motion vector.
/// * `rounding` — `vop_rounding_type` (0 or 1).
/// * `pred_y0` — destination field offset (0 = top, 1 = bottom).
/// * `ref_y0` — reference field offset (0 = top ref, 1 = bottom ref).
/// * `y_incr` — vertical increment (1 = frame, 2 = field).
///
/// The vertical half-pel test is `dy_halfpel & y_incr`: for the frame
/// case (`y_incr = 1`) the LSB; for the field case (`y_incr = 2`) the
/// bit-1 weight, so the two averaged reference lines (`y_ref`,
/// `y_ref + y_incr`) are adjacent lines of the same field.
///
/// # Panics
///
/// Panics if a written sample index would fall outside `pred`.
#[allow(clippy::too_many_arguments)]
pub fn mc(
    pred: &mut [u8],
    pred_stride: usize,
    reference: &ReferenceVop<'_>,
    x: i32,
    y: i32,
    width: usize,
    height: usize,
    dx_halfpel: i32,
    dy_halfpel: i32,
    rounding: u8,
    pred_y0: usize,
    ref_y0: i32,
    y_incr: i32,
) {
    let rc = (rounding & 1) as i32;
    // dx = dx_halfpel >> 1 (arithmetic shift → floor for negatives).
    let dx = dx_halfpel >> 1;
    // dy = y_incr * (dy_halfpel >> y_incr).
    let dy = y_incr * (dy_halfpel >> y_incr);
    let half_x = (dx_halfpel & 1) != 0;
    let half_y = (dy_halfpel & y_incr) != 0;
    let y_incr_u = y_incr as usize;
    // §7.6.4 edge clamp: the last full-pel position inside the decoded
    // VOP area, on the *frame* grid for both field parities (the
    // §7.6.1.5 per-field rule governs arbitrary-shape padding inside
    // the bounding rectangle, not the rectangular-VOP edge read) —
    // black-box-confirmed on the interlaced conformance corpus.
    let fetch = |xr: i32, yr: i32| -> i32 { i32::from(reference.fetch_clamped(xr, yr)) };

    let mut iy: usize = 0;
    while iy < height {
        for ix in 0..width {
            let x_ref = x + dx + ix as i32;
            let y_ref = y + dy + iy as i32 + ref_y0;
            let value: i32 = match (half_y, half_x) {
                (true, true) => {
                    (fetch(x_ref, y_ref)
                        + fetch(x_ref + 1, y_ref)
                        + fetch(x_ref, y_ref + y_incr)
                        + fetch(x_ref + 1, y_ref + y_incr)
                        + 2
                        - rc)
                        >> 2
                }
                (true, false) => (fetch(x_ref, y_ref) + fetch(x_ref, y_ref + y_incr) + 1 - rc) >> 1,
                (false, true) => (fetch(x_ref, y_ref) + fetch(x_ref + 1, y_ref) + 1 - rc) >> 1,
                (false, false) => fetch(x_ref, y_ref),
            };
            let dst_row = iy + pred_y0;
            pred[dst_row * pred_stride + ix] = value as u8;
        }
        iy += y_incr_u;
    }
}

/// §7.7.2.1 `field_motion_compensate_one_reference` — assemble the
/// field-predicted 4:2:0 prediction macroblock from a single reference
/// VOP and the two reconstructed field motion vectors.
///
/// `luma_ref` / `cb_ref` / `cr_ref` are the reference VOP planes
/// (even lines = top field, odd lines = bottom field). `(x, y)` is the
/// luma top-left coordinate of the current macroblock; chroma uses
/// `(x/2, y/2)`. `top_field_ref` / `bottom_field_ref` are the
/// §6.3.7.2 `forward_top_field_reference` /
/// `forward_bottom_field_reference` flags (`false` = top ref field,
/// `true` = bottom ref field), wired into `mc`'s `ref_y0`.
///
/// Per §7.7.2.1 the chroma field MVs are `Div2Round` of the
/// corresponding luma field MV components.
///
/// Returns the assembled [`InterPredictionMacroblock`] (`p[y][x]` in
/// the display range), ready for the §7.3 residual add in
/// [`crate::reconstruct::reconstruct_inter_macroblock_into`].
#[allow(clippy::too_many_arguments)]
pub fn field_motion_compensate_one_reference(
    luma_ref: &ReferenceVop<'_>,
    cb_ref: &ReferenceVop<'_>,
    cr_ref: &ReferenceVop<'_>,
    mvs: FieldMotionVectors,
    top_field_ref: bool,
    bottom_field_ref: bool,
    x: i32,
    y: i32,
    rounding_type: u8,
) -> InterPredictionMacroblock {
    let top_ref_y0 = top_field_ref as i32;
    let bot_ref_y0 = bottom_field_ref as i32;

    let mut luma = [0u8; MACROBLOCK_LUMA_SIDE * MACROBLOCK_LUMA_SIDE];
    let mut cb = [0u8; MACROBLOCK_CHROMA_SIDE * MACROBLOCK_CHROMA_SIDE];
    let mut cr = [0u8; MACROBLOCK_CHROMA_SIDE * MACROBLOCK_CHROMA_SIDE];

    // Luma: top field (pred_y0 = 0) then bottom field (pred_y0 = 1).
    mc(
        &mut luma,
        MACROBLOCK_LUMA_SIDE,
        luma_ref,
        x,
        y,
        MACROBLOCK_LUMA_SIDE,
        MACROBLOCK_LUMA_SIDE,
        mvs.top.x,
        mvs.top.y,
        rounding_type,
        0,
        top_ref_y0,
        2,
    );
    mc(
        &mut luma,
        MACROBLOCK_LUMA_SIDE,
        luma_ref,
        x,
        y,
        MACROBLOCK_LUMA_SIDE,
        MACROBLOCK_LUMA_SIDE,
        mvs.bottom.x,
        mvs.bottom.y,
        rounding_type,
        1,
        bot_ref_y0,
        2,
    );

    let cx = x / 2;
    let cy = y / 2;
    let top_cx = div2_round(mvs.top.x);
    let top_cy = field_chroma_dy(mvs.top.y);
    let bot_cx = div2_round(mvs.bottom.x);
    let bot_cy = field_chroma_dy(mvs.bottom.y);

    for (plane, reference) in [(&mut cb, cb_ref), (&mut cr, cr_ref)] {
        mc(
            plane,
            MACROBLOCK_CHROMA_SIDE,
            reference,
            cx,
            cy,
            MACROBLOCK_CHROMA_SIDE,
            MACROBLOCK_CHROMA_SIDE,
            top_cx,
            top_cy,
            rounding_type,
            0,
            top_ref_y0,
            2,
        );
        mc(
            plane,
            MACROBLOCK_CHROMA_SIDE,
            reference,
            cx,
            cy,
            MACROBLOCK_CHROMA_SIDE,
            MACROBLOCK_CHROMA_SIDE,
            bot_cx,
            bot_cy,
            rounding_type,
            1,
            bot_ref_y0,
            2,
        );
    }

    let mut out = InterPredictionMacroblock::zero();
    for row in 0..MACROBLOCK_LUMA_SIDE {
        for col in 0..MACROBLOCK_LUMA_SIDE {
            out.luma[row][col] = luma[row * MACROBLOCK_LUMA_SIDE + col] as i32;
        }
    }
    for row in 0..MACROBLOCK_CHROMA_SIDE {
        for col in 0..MACROBLOCK_CHROMA_SIDE {
            out.cb[row][col] = cb[row * MACROBLOCK_CHROMA_SIDE + col] as i32;
            out.cr[row][col] = cr[row * MACROBLOCK_CHROMA_SIDE + col] as i32;
        }
    }
    out
}

/// The §7.7.2.1 chrominance **vertical** component for the field-mode
/// [`mc`] calls, from the luma field MV's vertical component (frame
/// half-pel units, always even).
///
/// The field-mode `mc` encodes the vertical sub-position with **bit 1**
/// as the same-field half-sample flag (`dy = y_incr * (dy_halfpel >>
/// y_incr)`, `half = dy_halfpel & y_incr` with `y_incr == 2`), so the
/// 4:2:0 halving must land on that grid: `2 * Div2Round(MVy / 2)` —
/// halve to field units (exact — the component is even), then the
/// §7.7.2.1 `Div2Round` snap of any fractional chroma offset onto the
/// half-sample position, re-encoded with the half flag on bit 1.
///
/// Feeding `Div2Round(MVy)` straight into the field-mode `mc` (a
/// literal composition of the two §7.7.2.1 pseudo-code fragments)
/// silently drops bit 0, collapsing the `MVy ≡ +2 (mod 8)` cases to a
/// pure copy while their negative mirrors keep the interpolation —
/// the black-box reference decode confirms the symmetric snap.
#[inline]
pub const fn field_chroma_dy(luma_field_mvy: i32) -> i32 {
    2 * div2_round(luma_field_mvy / 2)
}

/// Half-pel-units equivalent of a quarter-pel luma field motion-vector
/// component (§7.7.2.2): the luma quarter-pel value "divided by 2 in
/// case of quarter_sample mode" before the [`div2_round`] chroma
/// scaling. `/` truncates toward 0 per §3.4, which for an even input
/// (the always-even vertical field component, §7.7.2.1) is exact.
///
/// For the horizontal component this rounds an odd quarter-pel value
/// toward zero into half-pel units; the subsequent [`div2_round`] then
/// performs the §6.1.3.4 4:2:0 chroma sub-sampling halve-with-residue.
#[inline]
pub const fn half_pel_chroma_mv_from_qpel(qpel_component: i32) -> i32 {
    qpel_component / 2
}

/// Chroma **vertical** field motion-vector derivation in
/// quarter-sample mode: `2 * Div2Round(mv_y >> 2)`, where `mv_y` is
/// the luma field MV's vertical component in quarter-pel frame
/// coordinates (always even, §7.7.2.1) and the result is the chroma
/// field MV in the same frame-half-pel representation the [`mc`]
/// chroma calls consume (4 units per field line, bit 1 = half sample).
///
/// §7.7.2.2 prints the derivation as "applying Div2Round to the
/// luminance motion vectors, divided by 2 in case of quarter_sample
/// mode", which pins neither the order of the two halvings on the
/// field grid nor the rounding direction of the quarter → half step
/// (the printed `field_motion_compensate_one_reference` chroma calls
/// are already defective in half-sample mode — see [`field_chroma_dy`]'s
/// symmetric-snap correction). Black-box pixel arbitration over
/// constructed single-macroblock field-prediction probe streams
/// (`tests/fixtures/NOTES.md`, field-qpel probe set) uniquely
/// determines the reference behaviour: the field-grid quarter-pel
/// value `mv_y / 2` is halved **toward minus infinity** (`>> 1`, i.e.
/// `mv_y >> 2` overall) and the result `Div2Round`-snapped — the
/// truncating alternative `Div2Round((mv_y / 2) / 2)` mispredicts
/// every probe with a negative odd field-grid component while this
/// form reproduces all of them bit-exactly. The horizontal component
/// keeps the truncating §7.6.5 quarter-mode derivation
/// (`Div2Round(mv_x / 2)`), likewise probe-confirmed for both signs.
#[inline]
pub const fn field_chroma_dy_qpel(luma_field_mvy_qpel: i32) -> i32 {
    2 * div2_round(luma_field_mvy_qpel >> 2)
}

/// §7.7.2.1 `field_motion_compensate_one_reference` in **quarter-sample
/// mode** (`quarter_sample == 1`).
///
/// Identical to [`field_motion_compensate_one_reference`] except the two
/// luma field blocks are interpolated with the §7.6.2.2 quarter-pel
/// cascade on the field reference grid
/// ([`interpolate_block_qpel_field_into`]) rather than the half-sample
/// [`mc`] routine. The luma field MVs (`mvs.top` / `mvs.bottom`) are in
/// quarter-pel frame coordinates with even vertical components
/// (§7.7.2.1).
///
/// ## Luma sub-block granularity (black-box-arbitrated)
///
/// Each 16-wide field block is interpolated as **two 8×8 blocks**
/// (8 columns × 8 field lines), each with its own §7.6.2.2 /
/// Figure 7-30 `(M+1)×(N+1)` read + boundary mirroring. §7.6.2.2
/// defines the process "for each block of size MxN" without pinning
/// M×N for a field-predicted macroblock (the §7.7.2.1 pseudo code
/// compensates a field per [`mc`] call, but the quarter-sample text
/// only says the macroblock is calculated "as described in subclause
/// 7.6.2.2, accordingly"); the spec's block-level unit is the 8×8
/// block (§3). Probe-stream arbitration (constructed field-prediction
/// P-VOPs over a conformant interlaced+qpel anchor, see
/// `tests/fixtures/NOTES.md`) determines the 8×8 reading uniquely: a
/// single 16-wide interpolation mispredicts isolated samples in the
/// columns whose FIR taps span the centre seam, while the two-8×8
/// split reproduces every probe bit-exactly (the progressive 16×16
/// 1-MV macroblock keeps its single-block interpolation, pinned by
/// the bit-exact progressive qpel conformance streams).
///
/// ## Chroma
///
/// The four [`mc`] calls run in half-sample field mode; the horizontal
/// chroma MV is `Div2Round(mv_x / 2)` (truncating quarter → half, then
/// the 4:2:0 halve) and the vertical is [`field_chroma_dy_qpel`]
/// (floor-halving on the field grid — see its docs for the
/// arbitration).
///
/// `bits_per_pixel` is the VOL `bits_per_pixel` (8 for `not_8_bit ==
/// 0`); it drives the §7.6.2.2.1 FIR clip.
#[allow(clippy::too_many_arguments)]
pub fn field_motion_compensate_one_reference_qpel(
    luma_ref: &ReferenceVop<'_>,
    cb_ref: &ReferenceVop<'_>,
    cr_ref: &ReferenceVop<'_>,
    mvs: FieldMotionVectors,
    top_field_ref: bool,
    bottom_field_ref: bool,
    x: i32,
    y: i32,
    rounding_type: u8,
    bits_per_pixel: u32,
) -> InterPredictionMacroblock {
    let top_ref_y0 = top_field_ref as i32;
    let bot_ref_y0 = bottom_field_ref as i32;

    // --- Luma: each 16-wide field block as two 8×8 §7.6.2.2 blocks
    // (per-sub-block Figure 7-30 mirroring — see the function docs for
    // the probe arbitration). The field interpolator returns the 8
    // lines of one field; the top field's lines land on even
    // destination rows, the bottom field's on odd rows.
    let half_h = MACROBLOCK_LUMA_SIDE / 2;
    let sub_w = MACROBLOCK_LUMA_SIDE / 2;
    let mut top_field = [0u8; MACROBLOCK_LUMA_SIDE * (MACROBLOCK_LUMA_SIDE / 2)];
    let mut bot_field = [0u8; MACROBLOCK_LUMA_SIDE * (MACROBLOCK_LUMA_SIDE / 2)];
    for (field_buf, mv, ref_y0) in [
        (&mut top_field, mvs.top, top_ref_y0),
        (&mut bot_field, mvs.bottom, bot_ref_y0),
    ] {
        for sub in 0..2 {
            let x_off = sub * sub_w;
            let mut sub_buf = [0u8; (MACROBLOCK_LUMA_SIDE / 2) * (MACROBLOCK_LUMA_SIDE / 2)];
            interpolate_block_qpel_field_into(
                luma_ref,
                mv.x,
                mv.y,
                x + x_off as i32,
                y,
                sub_w,
                half_h,
                ref_y0,
                rounding_type,
                bits_per_pixel,
                &mut sub_buf,
            );
            for (line, sub_line) in sub_buf.chunks_exact(sub_w).enumerate() {
                let dst = line * MACROBLOCK_LUMA_SIDE + x_off;
                field_buf[dst..dst + sub_w].copy_from_slice(sub_line);
            }
        }
    }

    // --- Chroma: half-sample field mc. Horizontal Div2Round(qpel / 2)
    // (truncating), vertical `field_chroma_dy_qpel` (floor on the
    // field grid — black-box-arbitrated, see its docs). ---
    let top_cx = div2_round(half_pel_chroma_mv_from_qpel(mvs.top.x));
    let top_cy = field_chroma_dy_qpel(mvs.top.y);
    let bot_cx = div2_round(half_pel_chroma_mv_from_qpel(mvs.bottom.x));
    let bot_cy = field_chroma_dy_qpel(mvs.bottom.y);
    let cx = x / 2;
    let cy = y / 2;
    let mut cb = [0u8; MACROBLOCK_CHROMA_SIDE * MACROBLOCK_CHROMA_SIDE];
    let mut cr = [0u8; MACROBLOCK_CHROMA_SIDE * MACROBLOCK_CHROMA_SIDE];
    for (plane, reference) in [(&mut cb, cb_ref), (&mut cr, cr_ref)] {
        mc(
            plane,
            MACROBLOCK_CHROMA_SIDE,
            reference,
            cx,
            cy,
            MACROBLOCK_CHROMA_SIDE,
            MACROBLOCK_CHROMA_SIDE,
            top_cx,
            top_cy,
            rounding_type,
            0,
            top_ref_y0,
            2,
        );
        mc(
            plane,
            MACROBLOCK_CHROMA_SIDE,
            reference,
            cx,
            cy,
            MACROBLOCK_CHROMA_SIDE,
            MACROBLOCK_CHROMA_SIDE,
            bot_cx,
            bot_cy,
            rounding_type,
            1,
            bot_ref_y0,
            2,
        );
    }

    let mut out = InterPredictionMacroblock::zero();
    for r in 0..half_h {
        for col in 0..MACROBLOCK_LUMA_SIDE {
            // Top field's line r → destination even row 2r.
            out.luma[2 * r][col] = top_field[r * MACROBLOCK_LUMA_SIDE + col] as i32;
            // Bottom field's line r → destination odd row 2r + 1.
            out.luma[2 * r + 1][col] = bot_field[r * MACROBLOCK_LUMA_SIDE + col] as i32;
        }
    }
    for row in 0..MACROBLOCK_CHROMA_SIDE {
        for col in 0..MACROBLOCK_CHROMA_SIDE {
            out.cb[row][col] = cb[row * MACROBLOCK_CHROMA_SIDE + col] as i32;
            out.cr[row][col] = cr[row * MACROBLOCK_CHROMA_SIDE + col] as i32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::MotionVectorDelta;

    fn pair(top: (i32, i32), bottom: (i32, i32)) -> FieldMvPair {
        FieldMvPair {
            top: MotionVectorDelta {
                dx: top.0,
                dy: top.1,
            },
            bottom: MotionVectorDelta {
                dx: bottom.0,
                dy: bottom.1,
            },
        }
    }

    #[test]
    fn div2_round_matches_spec_bit_definition() {
        // Div2Round(x) = (x >> 1) | (x & 1).
        assert_eq!(div2_round(0), 0);
        assert_eq!(div2_round(1), 1); // (0) | (1)
        assert_eq!(div2_round(2), 1); // (1) | (0)
        assert_eq!(div2_round(3), 1); // (1) | (1) = 1
        assert_eq!(div2_round(4), 2);
        assert_eq!(div2_round(5), 3); // (2) | (1) = 3
        assert_eq!(div2_round(-1), -1); // (-1) | (1) = -1
        assert_eq!(div2_round(-2), -1); // (-1) | (0) = -1
        assert_eq!(div2_round(-3), -1); // (-2) | (1) = -1
        assert_eq!(div2_round(-4), -2);
    }

    #[test]
    fn reconstruct_field_mvs_even_vertical_zero_predictor() {
        // Zero predictor → MVy fi = 2 * MVDy fi, always even.
        let mvs = reconstruct_field_motion_vectors(pair((3, 1), (-2, -1)), 0, 0);
        assert_eq!(mvs.top, MotionVector { x: 3, y: 2 });
        assert_eq!(mvs.bottom, MotionVector { x: -2, y: -2 });
        // Both vertical components are even.
        assert_eq!(mvs.top.y % 2, 0);
        assert_eq!(mvs.bottom.y % 2, 0);
    }

    #[test]
    fn reconstruct_field_mvs_shared_predictor() {
        // §7.7.2.1: both fields use the same predictor (px, py).
        // MVx fi = MVDx fi + Px; MVy fi = 2 * (MVDy fi + Py/2).
        let mvs = reconstruct_field_motion_vectors(pair((1, 2), (0, -1)), 4, 6);
        // py/2 = 3.
        assert_eq!(
            mvs.top,
            MotionVector {
                x: 5,
                y: 2 * (2 + 3)
            }
        ); // (5, 10)
        assert_eq!(
            mvs.bottom,
            MotionVector {
                x: 4,
                y: 2 * (-1 + 3)
            }
        ); // (4, 4)
    }

    #[test]
    fn reconstruct_field_mvs_negative_predictor_truncates_toward_zero() {
        // Py = -3 → Py/2 = -1 (truncation toward 0, not floor → -2).
        let mvs = reconstruct_field_motion_vectors(pair((0, 0), (0, 0)), 0, -3);
        // MVDy = 0, Py/2 = -1 → 2 * (0 + -1) = -2 (not -4 from floor).
        assert_eq!(mvs.top.y, -2);
        assert_eq!(mvs.bottom.y, -2);
    }

    // ---------------------------------------------------------------
    // `mc` field/frame equivalence + interleave behaviour.
    // ---------------------------------------------------------------

    /// A 16×16 reference whose sample at (x, y) is a deterministic
    /// function of its coordinates, so the fetched value pins which
    /// reference line a field MC read.
    fn ramp_plane(side: usize) -> Vec<u8> {
        let mut v = vec![0u8; side * side];
        for y in 0..side {
            for x in 0..side {
                v[y * side + x] = ((y * 7 + x * 3) & 0xff) as u8;
            }
        }
        v
    }

    #[test]
    fn mc_frame_integer_mv_copies_reference() {
        // y_incr = 1, integer MV (0,0): pred[ix][iy] = ref[x+ix][y+iy].
        let plane = ramp_plane(16);
        let reference = ReferenceVop::new(&plane, 16, 16).unwrap();
        let mut pred = vec![0u8; 16 * 16];
        mc(&mut pred, 16, &reference, 0, 0, 16, 16, 0, 0, 0, 0, 0, 1);
        assert_eq!(pred, plane);
    }

    #[test]
    fn mc_field_top_reads_even_lines_only_with_zero_mv() {
        // Field top MC (pred_y0 = 0, ref_y0 = 0, y_incr = 2, MV = 0):
        // pred top-field line iy (iy even) = ref line iy.
        let side = 16;
        let plane = ramp_plane(side);
        let reference = ReferenceVop::new(&plane, side, side).unwrap();
        let mut pred = vec![0u8; side * side];
        mc(
            &mut pred, side, &reference, 0, 0, side, side, 0, 0, 0, 0, 0, 2,
        );
        // Only even destination rows written; pred[even][x] == ref[even][x].
        for y in (0..side).step_by(2) {
            for x in 0..side {
                assert_eq!(pred[y * side + x], plane[y * side + x]);
            }
        }
        // Odd destination rows untouched (still 0).
        for y in (1..side).step_by(2) {
            for x in 0..side {
                assert_eq!(pred[y * side + x], 0);
            }
        }
    }

    #[test]
    fn mc_field_bottom_ref_offset_reads_odd_reference_lines() {
        // Field bottom MC with bottom reference (ref_y0 = 1), MV = 0,
        // y_incr = 2, pred_y0 = 1: destination odd line iy+1 reads
        // ref line (iy + 1) — the odd (bottom) reference field.
        let side = 16;
        let plane = ramp_plane(side);
        let reference = ReferenceVop::new(&plane, side, side).unwrap();
        let mut pred = vec![0u8; side * side];
        mc(
            &mut pred, side, &reference, 0, 0, side, side, 0, 0, 0, 1, 1, 2,
        );
        for iy in (0..side).step_by(2) {
            let dst_row = iy + 1;
            if dst_row >= side {
                break;
            }
            for x in 0..side {
                // y_ref = y + dy + iy + ref_y0 = 0 + 0 + iy + 1.
                assert_eq!(pred[dst_row * side + x], plane[(iy + 1) * side + x]);
            }
        }
    }

    #[test]
    fn mc_field_vertical_half_pel_averages_same_field_lines() {
        // dy_halfpel = 2 sets the field vertical half-pel bit
        // (dy_halfpel & y_incr, y_incr = 2). The two averaged
        // reference lines are y_ref and y_ref + 2 — adjacent lines of
        // the same field. dy = 2 * (2 >> 2) = 0.
        let side = 16;
        // Plane where each line is a constant equal to its row index,
        // so the same-field average of lines L and L+2 is L+1.
        let mut plane = vec![0u8; side * side];
        for y in 0..side {
            for x in 0..side {
                plane[y * side + x] = y as u8;
            }
        }
        let reference = ReferenceVop::new(&plane, side, side).unwrap();
        let mut pred = vec![0u8; side * side];
        // Top field, top reference, rounding 0.
        mc(
            &mut pred, side, &reference, 0, 0, side, side, 0, 2, 0, 0, 0, 2,
        );
        for iy in (0..side - 2).step_by(2) {
            for x in 0..side {
                // (line iy + line iy+2 + 1 - 0) >> 1 = (iy + iy+2 +1)>>1
                let expected = ((iy + (iy + 2) + 1) >> 1) as u8;
                assert_eq!(pred[iy * side + x], expected, "iy={iy} x={x}");
            }
        }
    }

    #[test]
    fn field_mc_macroblock_flat_reference_reproduces_flat_prediction() {
        // A flat reference predicts a flat macroblock regardless of MV /
        // field references / rounding.
        let side = 48;
        let plane = vec![123u8; side * side];
        let cplane = vec![200u8; (side / 2) * (side / 2)];
        let luma_ref = ReferenceVop::new(&plane, side, side).unwrap();
        let cb_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let cr_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let mvs = reconstruct_field_motion_vectors(pair((3, 1), (-4, -2)), 2, 4);
        let pred = field_motion_compensate_one_reference(
            &luma_ref, &cb_ref, &cr_ref, mvs, true, false, 16, 16, 0,
        );
        for row in 0..MACROBLOCK_LUMA_SIDE {
            for col in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(pred.luma[row][col], 123);
            }
        }
        for row in 0..MACROBLOCK_CHROMA_SIDE {
            for col in 0..MACROBLOCK_CHROMA_SIDE {
                assert_eq!(pred.cb[row][col], 200);
                assert_eq!(pred.cr[row][col], 200);
            }
        }
    }

    #[test]
    fn field_mc_macroblock_top_and_bottom_select_independent_fields() {
        // Reference whose top field (even lines) is all 10 and bottom
        // field (odd lines) is all 200. With zero MVs and top→top,
        // bottom→bottom references, the predicted macroblock's even
        // lines must be 10 and odd lines 200.
        let side = 48;
        let mut plane = vec![0u8; side * side];
        for y in 0..side {
            let val = if y % 2 == 0 { 10u8 } else { 200u8 };
            for x in 0..side {
                plane[y * side + x] = val;
            }
        }
        // Chroma plane: flat (chroma field interleave tested via luma).
        let cplane = vec![128u8; (side / 2) * (side / 2)];
        let luma_ref = ReferenceVop::new(&plane, side, side).unwrap();
        let cb_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let cr_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        // Zero differentials, zero predictor → zero field MVs. With
        // MV = 0 the reference line read is `iy + ref_y0`, so the
        // chosen reference field is purely the *_field_ref flag.
        let mvs = reconstruct_field_motion_vectors(pair((0, 0), (0, 0)), 0, 0);
        // top output field → top reference (false → even ref lines = 10),
        // bottom output field → bottom reference (true → odd ref lines
        // = 200). The destination top field (even dst rows) gets 10, the
        // destination bottom field (odd dst rows) gets 200.
        let pred = field_motion_compensate_one_reference(
            &luma_ref, &cb_ref, &cr_ref, mvs, false, true, 0, 0, 0,
        );
        for row in 0..MACROBLOCK_LUMA_SIDE {
            let expected = if row % 2 == 0 { 10 } else { 200 };
            for col in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(pred.luma[row][col], expected, "row={row}");
            }
        }
    }

    #[test]
    fn field_mc_bottom_field_ref_flag_selects_bottom_reference_field() {
        // Same striped reference, but now force BOTH output fields to
        // read the bottom reference field via the field-reference flags.
        // top_field_ref = true, bottom_field_ref = true → every output
        // line reads an odd reference line → all 200.
        let side = 48;
        let mut plane = vec![0u8; side * side];
        for y in 0..side {
            let val = if y % 2 == 0 { 10u8 } else { 200u8 };
            for x in 0..side {
                plane[y * side + x] = val;
            }
        }
        let cplane = vec![128u8; (side / 2) * (side / 2)];
        let luma_ref = ReferenceVop::new(&plane, side, side).unwrap();
        let cb_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let cr_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let mvs = reconstruct_field_motion_vectors(pair((0, 0), (0, 0)), 0, 0);
        let pred = field_motion_compensate_one_reference(
            &luma_ref, &cb_ref, &cr_ref, mvs, true, true, 0, 0, 0,
        );
        for row in 0..MACROBLOCK_LUMA_SIDE {
            for col in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(pred.luma[row][col], 200, "row={row}");
            }
        }
    }

    #[test]
    fn field_mc_then_residual_add_reconstructs_interlaced_macroblock() {
        // End-to-end §7.7.2.1 → §7.3: a field-predicted P-VOP
        // macroblock reconstructs pixels by adding the decoded texture
        // residual to the field-motion-compensated prediction.
        use crate::block::InterMacroblock;
        use crate::reconstruct::reconstruct_inter_macroblock;

        // Striped reference: even (top) ref lines 60, odd (bottom) ref
        // lines 90. Flat chroma at 128.
        let side = 48;
        let mut plane = vec![0u8; side * side];
        for y in 0..side {
            let val = if y % 2 == 0 { 60u8 } else { 90u8 };
            for x in 0..side {
                plane[y * side + x] = val;
            }
        }
        let cplane = vec![128u8; (side / 2) * (side / 2)];
        let luma_ref = ReferenceVop::new(&plane, side, side).unwrap();
        let cb_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let cr_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();

        // Zero field MVs; top output → top ref (60), bottom output →
        // bottom ref (90).
        let mvs = reconstruct_field_motion_vectors(pair((0, 0), (0, 0)), 0, 0);
        let pred = field_motion_compensate_one_reference(
            &luma_ref, &cb_ref, &cr_ref, mvs, false, true, 0, 0, 0,
        );

        // Residual: +5 luma everywhere, +1 Cb / -1 Cr.
        let mut residual = InterMacroblock {
            luma: [[5i32; 16]; 16],
            cb: [[1i32; 8]; 8],
            cr: [[-1i32; 8]; 8],
        };
        // One sample driven past the display range to exercise the §7.3
        // step-3 clip: top-field sample 60 + 250 = 310 → 255.
        residual.luma[0][0] = 250;

        let recon = reconstruct_inter_macroblock(&pred, &residual, 8);

        // Top (even) lines: 60 + 5 = 65 (except the clipped sample).
        // Bottom (odd) lines: 90 + 5 = 95.
        assert_eq!(recon.luma[0][0], 255); // clipped
        assert_eq!(recon.luma[0][1], 65);
        assert_eq!(recon.luma[2][3], 65);
        assert_eq!(recon.luma[1][0], 95);
        assert_eq!(recon.luma[3][7], 95);
        for row in 0..MACROBLOCK_CHROMA_SIDE {
            for col in 0..MACROBLOCK_CHROMA_SIDE {
                assert_eq!(recon.cb[row][col], 129);
                assert_eq!(recon.cr[row][col], 127);
            }
        }
    }

    // ───────────────── quarter-sample field MC ───────────────────────

    #[test]
    fn half_pel_chroma_mv_from_qpel_truncates_toward_zero() {
        // §7.7.2.2: luma qpel divided by 2 (quarter → half), `/`
        // truncating toward 0. Even inputs (the field vertical
        // component) are exact.
        assert_eq!(half_pel_chroma_mv_from_qpel(0), 0);
        assert_eq!(half_pel_chroma_mv_from_qpel(2), 1);
        assert_eq!(half_pel_chroma_mv_from_qpel(4), 2);
        assert_eq!(half_pel_chroma_mv_from_qpel(3), 1); // toward 0
        assert_eq!(half_pel_chroma_mv_from_qpel(-3), -1); // toward 0
        assert_eq!(half_pel_chroma_mv_from_qpel(-4), -2);
    }

    #[test]
    fn field_mc_qpel_flat_reference_reproduces_flat_prediction() {
        // A flat reference predicts a flat macroblock regardless of MV /
        // field references / rounding, even through the 8-tap FIR
        // (coefficients sum 256).
        let side = 48;
        let plane = vec![123u8; side * side];
        let cplane = vec![200u8; (side / 2) * (side / 2)];
        let luma_ref = ReferenceVop::new(&plane, side, side).unwrap();
        let cb_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let cr_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        // Field MVs with even vertical components (§7.7.2.1 invariant).
        let mvs = reconstruct_field_motion_vectors(pair((5, 1), (-6, -2)), 2, 4);
        for &rc in &[0u8, 1] {
            let pred = field_motion_compensate_one_reference_qpel(
                &luma_ref, &cb_ref, &cr_ref, mvs, true, false, 16, 16, rc, 8,
            );
            for row in 0..MACROBLOCK_LUMA_SIDE {
                for col in 0..MACROBLOCK_LUMA_SIDE {
                    assert_eq!(pred.luma[row][col], 123, "rc={rc} row={row} col={col}");
                }
            }
            for row in 0..MACROBLOCK_CHROMA_SIDE {
                for col in 0..MACROBLOCK_CHROMA_SIDE {
                    assert_eq!(pred.cb[row][col], 200);
                    assert_eq!(pred.cr[row][col], 200);
                }
            }
        }
    }

    #[test]
    fn field_mc_qpel_top_and_bottom_select_independent_fields() {
        // Reference whose top field (even lines) is 10 and bottom field
        // (odd lines) is 200. Zero field MVs (full-pel), top→top and
        // bottom→bottom reference fields: even output rows = 10, odd
        // output rows = 200 — confirming the luma field interleave and
        // that the 8-tap FIR never crosses field parity at full pel.
        let side = 48;
        let mut plane = vec![0u8; side * side];
        for y in 0..side {
            let val = if y % 2 == 0 { 10u8 } else { 200u8 };
            for x in 0..side {
                plane[y * side + x] = val;
            }
        }
        let cplane = vec![128u8; (side / 2) * (side / 2)];
        let luma_ref = ReferenceVop::new(&plane, side, side).unwrap();
        let cb_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let cr_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let mvs = reconstruct_field_motion_vectors(pair((0, 0), (0, 0)), 0, 0);
        let pred = field_motion_compensate_one_reference_qpel(
            &luma_ref, &cb_ref, &cr_ref, mvs, false, true, 0, 0, 0, 8,
        );
        for row in 0..MACROBLOCK_LUMA_SIDE {
            let expected = if row % 2 == 0 { 10 } else { 200 };
            for col in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(pred.luma[row][col], expected, "row={row}");
            }
        }
    }

    #[test]
    fn field_mc_qpel_full_pel_mv_equals_half_sample_path() {
        // At a full-pel (integer) field MV there is no sub-pel
        // interpolation in either mode, so the quarter-sample driver
        // must produce the exact same prediction as the half-sample
        // driver. Use a non-trivial striped reference and a full-pel
        // field MV (MVx multiple of 4, MVy multiple of 8 in qpel; but
        // since the half-sample path takes half-pel MVs, compare with
        // the same *integer* displacement expressed in each unit).
        let side = 48;
        let mut plane = vec![0u8; side * side];
        for y in 0..side {
            for x in 0..side {
                plane[y * side + x] = ((y * 5 + x * 3) & 0xff) as u8;
            }
        }
        let mut cplane = vec![0u8; (side / 2) * (side / 2)];
        for y in 0..side / 2 {
            for x in 0..side / 2 {
                cplane[y * (side / 2) + x] = ((y * 9 + x * 4) & 0xff) as u8;
            }
        }
        let luma_ref = ReferenceVop::new(&plane, side, side).unwrap();
        let cb_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let cr_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();

        // Half-sample MVs: top = (+2, +4 half-pel), i.e. +1 full pel
        // horizontally, +2 full field lines (MVy / 2 = 2 → even). The
        // equivalent quarter-pel MVs are double: (+4, +8).
        let mvs_half = FieldMotionVectors {
            top: MotionVector { x: 2, y: 4 },
            bottom: MotionVector { x: -2, y: -8 },
        };
        let mvs_qpel = FieldMotionVectors {
            top: MotionVector { x: 4, y: 8 },
            bottom: MotionVector { x: -4, y: -16 },
        };
        let half = field_motion_compensate_one_reference(
            &luma_ref, &cb_ref, &cr_ref, mvs_half, true, false, 16, 16, 0,
        );
        let qpel = field_motion_compensate_one_reference_qpel(
            &luma_ref, &cb_ref, &cr_ref, mvs_qpel, true, false, 16, 16, 0, 8,
        );
        for row in 0..MACROBLOCK_LUMA_SIDE {
            for col in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(
                    qpel.luma[row][col], half.luma[row][col],
                    "luma mismatch at ({row},{col})"
                );
            }
        }
        for row in 0..MACROBLOCK_CHROMA_SIDE {
            for col in 0..MACROBLOCK_CHROMA_SIDE {
                assert_eq!(qpel.cb[row][col], half.cb[row][col], "cb ({row},{col})");
                assert_eq!(qpel.cr[row][col], half.cr[row][col], "cr ({row},{col})");
            }
        }
    }

    #[test]
    fn field_mc_qpel_then_residual_add_reconstructs_macroblock() {
        // End-to-end §7.6.2.2 field-qpel MC → §7.3 residual add, with
        // the §7.3 step-3 display clip exercised.
        use crate::block::InterMacroblock;
        use crate::reconstruct::reconstruct_inter_macroblock;

        let side = 48;
        let mut plane = vec![0u8; side * side];
        for y in 0..side {
            let val = if y % 2 == 0 { 60u8 } else { 90u8 };
            for x in 0..side {
                plane[y * side + x] = val;
            }
        }
        let cplane = vec![128u8; (side / 2) * (side / 2)];
        let luma_ref = ReferenceVop::new(&plane, side, side).unwrap();
        let cb_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();
        let cr_ref = ReferenceVop::new(&cplane, side / 2, side / 2).unwrap();

        // Zero field MVs; top → top ref (60), bottom → bottom ref (90).
        let mvs = reconstruct_field_motion_vectors(pair((0, 0), (0, 0)), 0, 0);
        let pred = field_motion_compensate_one_reference_qpel(
            &luma_ref, &cb_ref, &cr_ref, mvs, false, true, 0, 0, 0, 8,
        );

        let mut residual = InterMacroblock {
            luma: [[5i32; 16]; 16],
            cb: [[1i32; 8]; 8],
            cr: [[-1i32; 8]; 8],
        };
        residual.luma[0][0] = 250; // 60 + 250 = 310 → clip to 255.

        let recon = reconstruct_inter_macroblock(&pred, &residual, 8);
        assert_eq!(recon.luma[0][0], 255); // clipped
        assert_eq!(recon.luma[0][1], 65); // top field 60 + 5
        assert_eq!(recon.luma[2][3], 65);
        assert_eq!(recon.luma[1][0], 95); // bottom field 90 + 5
        assert_eq!(recon.luma[3][7], 95);
        for row in 0..MACROBLOCK_CHROMA_SIDE {
            for col in 0..MACROBLOCK_CHROMA_SIDE {
                assert_eq!(recon.cb[row][col], 129);
                assert_eq!(recon.cr[row][col], 127);
            }
        }
    }
}

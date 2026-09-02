//! §7.7.2.1 **field motion estimation** for the interlaced encoder.
//!
//! A field-predicted macroblock predicts each of its two output
//! fields (the even and the odd luminance lines) from one reference
//! field with its own motion vector; the vector's vertical component
//! lives on the *field* line grid (§7.7.2.1: `MVy fi` is always even
//! in frame half-/quarter-sample coordinates, an odd multiple of 2
//! meaning a same-field half-sample position). The estimator here
//! searches, per output field and per candidate reference field
//! parity, a full-pel window on the field grid followed by the
//! sub-pel ring refinement of the progressive estimator, scoring every
//! candidate through the decoder's own field interpolators
//! ([`crate::field_motion::mc`] in half-sample mode, the §7.6.2.2
//! per-8×8 field cascade in quarter-sample mode) so the chosen vector
//! is exactly the prediction a conformant decoder will form.
//!
//! The decoder reconstructs a field vector as `MVx = MVDx + Px`,
//! `MVy = 2 * (MVDy + Py / 2)` (§7.7.2.1), each component under the
//! §7.6.3 `[low:high]` modulo wrap on its own grid (the vertical one
//! in field units). A conformant stream keeps every reconstructed
//! component inside that Table 7-9 range and every differential
//! representable under the VOP's `fcode`, so the estimator takes the
//! predictor `(Px, Py)` the differentials will be coded against and
//! confines its candidates to that codable window (a field vector the
//! encoder left outside the range would be wrapped by the decoder —
//! black-box-confirmed on the reference decoder).
//!
//! Provenance: §7.7.2.1 (field MV reconstruction,
//! `field_motion_compensate_one_reference`, the `mc` routine) of
//! ISO/IEC 14496-2:2004 (3rd edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`,
//! via the crate's decoder transcriptions. No third-party source was
//! consulted.

use crate::bvop_prediction::BVopSampleMode;
use crate::field_motion::mc;
use crate::half_sample::ReferenceVop;
use crate::motion::MotionVector;
use crate::pvop_encode::{mv_range, SEARCH_RANGE};
use crate::quarter_sample::interpolate_block_qpel_field_into;

/// The estimate for one output field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldEstimate {
    /// The field motion vector in frame coordinates (VOL units,
    /// vertical component even).
    pub mv: MotionVector,
    /// The reference field parity (`false` = top, `true` = bottom) —
    /// the §6.3.7.2 `forward_*_field_reference` bit.
    pub ref_field: bool,
    /// Luma SAD of the 16×8 field block.
    pub sad: u32,
}

/// One 16×8 output field of the source macroblock: `field` selects
/// the even (`false`) or odd (`true`) lines of the 16×16 source.
pub(crate) fn source_field(src: &[[i32; 16]; 16], field: bool) -> [[i32; 16]; 8] {
    let mut out = [[0i32; 16]; 8];
    for (j, row) in out.iter_mut().enumerate() {
        *row = src[2 * j + usize::from(field)];
    }
    out
}

/// The decoder's luminance prediction of one 16×8 field block for the
/// field vector `mv` (frame coordinates, even vertical component)
/// drawn from reference field `ref_field` of `reference`, at luma
/// macroblock origin `(mb_x, mb_y)`. Half-sample mode runs the
/// §7.7.2.1 `mc` routine with `y_incr = 2`; quarter-sample mode runs
/// the §7.6.2.2 field cascade as two 8×8 blocks exactly as
/// [`crate::field_motion::field_motion_compensate_one_reference_qpel`]
/// does. (The output field parity does not enter the reference read —
/// only the destination line offset differs — so one prediction serves
/// both output fields.)
pub(crate) fn field_luma_prediction(
    reference: &ReferenceVop<'_>,
    mv: MotionVector,
    ref_field: bool,
    mb_x: i32,
    mb_y: i32,
    mode: BVopSampleMode,
) -> [[i32; 16]; 8] {
    debug_assert_eq!(mv.y & 1, 0, "field MV vertical components are even");
    let mut out = [[0i32; 16]; 8];
    match mode {
        BVopSampleMode::HalfPel => {
            let mut buf = [0u8; 16 * 16];
            mc(
                &mut buf,
                16,
                reference,
                mb_x,
                mb_y,
                16,
                16,
                mv.x,
                mv.y,
                0,
                0,
                i32::from(ref_field),
                2,
            );
            for (j, row) in out.iter_mut().enumerate() {
                for (i, cell) in row.iter_mut().enumerate() {
                    *cell = i32::from(buf[2 * j * 16 + i]);
                }
            }
        }
        BVopSampleMode::QuarterPel { bits_per_pixel } => {
            for sub in 0..2usize {
                let mut sub_buf = [0u8; 64];
                interpolate_block_qpel_field_into(
                    reference,
                    mv.x,
                    mv.y,
                    mb_x + (sub * 8) as i32,
                    mb_y,
                    8,
                    8,
                    i32::from(ref_field),
                    0,
                    bits_per_pixel,
                    &mut sub_buf,
                );
                for (j, row) in out.iter_mut().enumerate() {
                    for i in 0..8 {
                        row[sub * 8 + i] = i32::from(sub_buf[j * 8 + i]);
                    }
                }
            }
        }
    }
    out
}

/// SAD of a source field against a decoder-formed field prediction.
fn field_sad(src: &[[i32; 16]; 8], pred: &[[i32; 16]; 8]) -> u32 {
    let mut sad = 0u32;
    for (s_row, p_row) in src.iter().zip(pred.iter()) {
        for (&s, &p) in s_row.iter().zip(p_row.iter()) {
            sad += (s - p).unsigned_abs();
        }
    }
    sad
}

/// Full-pel SAD of a source field against reference field `ref_field`
/// displaced by `(dx, dy)` **field** pels (§7.6.4 edge clamp).
fn field_sad_full_pel(
    src: &[[i32; 16]; 8],
    reference: &ReferenceVop<'_>,
    ref_field: bool,
    mb_x: i32,
    mb_y: i32,
    dx: i32,
    dy: i32,
) -> u32 {
    let mut sad = 0u32;
    let y0 = mb_y + i32::from(ref_field);
    for (j, row) in src.iter().enumerate() {
        let ry = y0 + 2 * (j as i32 + dy);
        for (i, &s) in row.iter().enumerate() {
            let r = reference.fetch_clamped(mb_x + i as i32 + dx, ry);
            sad += (s - i32::from(r)).unsigned_abs();
        }
    }
    sad
}

/// The number of MV units per pel under `mode`.
fn units_per_pel(mode: BVopSampleMode) -> i32 {
    match mode {
        BVopSampleMode::HalfPel => 2,
        BVopSampleMode::QuarterPel { .. } => 4,
    }
}

/// Estimate the field motion vector of one output field (`field`:
/// `false` = top / even lines, `true` = bottom / odd lines) of the
/// source macroblock `src` against both reference field parities of
/// `reference`, returning the better parity's estimate.
///
/// `predictor` is the shared §7.7.2.1 `(Px, Py)` the two field
/// differentials will be coded against; candidates are confined to
/// the window whose differential is representable under `fcode`
/// (`MVx - Px` and `MVy / 2 - Py / 2` inside the Table 7-9 range).
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_field_motion(
    src: &[[i32; 16]; 16],
    reference: &ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    field: bool,
    predictor: MotionVector,
    mode: BVopSampleMode,
    fcode: u8,
) -> FieldEstimate {
    let src_field = source_field(src, field);
    let unit = units_per_pel(mode);
    let (low, high) = mv_range(fcode);
    // Codable window in field MV units (horizontal in VOL units,
    // vertical in VOL units of *field* lines): the reconstructed
    // component must lie inside the Table 7-9 `[low, high]` range on
    // its own grid (§7.6.3 — the general process, wrap included,
    // applies to interlaced VOPs, so a component the encoder leaves
    // outside would be wrapped by the decoder) *and* its differential
    // must be representable under `fcode`.
    let px = predictor.x;
    let py_half = predictor.y / 2;
    let (x_lo, x_hi) = ((px + low).max(low), (px + high).min(high));
    let (y_lo, y_hi) = ((py_half + low).max(low), (py_half + high).min(high));
    let clamp_field =
        |fx: i32, fy: i32| -> (i32, i32) { (fx.clamp(x_lo, x_hi), fy.clamp(y_lo, y_hi)) };

    let mut best: Option<FieldEstimate> = None;
    for ref_field in [false, true] {
        // Full-pel search on the field grid; the zero vector keeps the
        // progressive estimator's small favouring bias.
        let mut best_full = (0i32, 0i32);
        let mut best_full_sad =
            field_sad_full_pel(&src_field, reference, ref_field, mb_x, mb_y, 0, 0)
                .saturating_sub(64);
        for dy in -SEARCH_RANGE..=SEARCH_RANGE {
            for dx in -SEARCH_RANGE..=SEARCH_RANGE {
                if (dx, dy) == (0, 0) {
                    continue;
                }
                // Skip full-pel points whose differential cannot be coded.
                let (fx, fy) = (dx * unit, dy * unit);
                if fx < x_lo || fx > x_hi || fy < y_lo || fy > y_hi {
                    continue;
                }
                let sad = field_sad_full_pel(&src_field, reference, ref_field, mb_x, mb_y, dx, dy);
                if sad < best_full_sad {
                    best_full_sad = sad;
                    best_full = (dx, dy);
                }
            }
        }
        // Sub-pel ring refinement on the field grid (half, then quarter
        // in quarter-sample mode), scored through the decoder's field
        // interpolator.
        let (mut fx, mut fy) = clamp_field(best_full.0 * unit, best_full.1 * unit);
        let sad_of = |fx: i32, fy: i32| -> u32 {
            let mv = MotionVector { x: fx, y: 2 * fy };
            let pred = field_luma_prediction(reference, mv, ref_field, mb_x, mb_y, mode);
            field_sad(&src_field, &pred)
        };
        let mut best_sad = sad_of(fx, fy);
        let mut steps = vec![unit / 2];
        if matches!(mode, BVopSampleMode::QuarterPel { .. }) {
            steps.push(1);
        }
        for step in steps {
            let (cx, cy) = (fx, fy);
            for hy in -1..=1 {
                for hx in -1..=1 {
                    if (hx, hy) == (0, 0) {
                        continue;
                    }
                    let (nx, ny) = clamp_field(cx + hx * step, cy + hy * step);
                    if (nx, ny) == (fx, fy) {
                        continue;
                    }
                    let sad = sad_of(nx, ny);
                    if sad < best_sad {
                        best_sad = sad;
                        fx = nx;
                        fy = ny;
                    }
                }
            }
        }
        let est = FieldEstimate {
            mv: MotionVector { x: fx, y: 2 * fy },
            ref_field,
            sad: best_sad,
        };
        // Ties keep the top reference field (the all-zero default).
        if best.map_or(true, |b| est.sad < b.sad) {
            best = Some(est);
        }
    }
    best.expect("two reference parities were scored")
}

/// The field-MV differential pair the §7.7.2.1 reconstruction inverts:
/// `MVDx = MVx - Px`, `MVDy = MVy / 2 - Py / 2` (both `/` truncating
/// toward zero; `MVy` is even so its halving is exact).
pub(crate) fn field_mv_differential(mv: MotionVector, predictor: MotionVector) -> (i32, i32) {
    debug_assert_eq!(mv.y & 1, 0);
    (mv.x - predictor.x, mv.y / 2 - predictor.y / 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_motion::{reconstruct_field_motion_vectors, FieldMotionVectors};
    use crate::motion::{FieldMvPair, MotionVectorDelta};

    /// A textured 32×32 plane whose two fields carry unrelated
    /// textures (no displacement of one reproduces the other).
    fn plane() -> Vec<u8> {
        let mut p = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                p[y * 32 + x] = if y % 2 == 1 {
                    ((x * 53 + (y / 2) * 7) % 150) as u8 + 100
                } else {
                    ((x * 37 + (y / 2) * 11) % 120) as u8 + 10
                };
            }
        }
        p
    }

    #[test]
    fn differential_inverts_the_decoder_reconstruction() {
        for (mv, pred) in [
            (MotionVector { x: 5, y: 6 }, MotionVector { x: -3, y: 7 }),
            (MotionVector { x: -9, y: -4 }, MotionVector { x: 2, y: -5 }),
            (MotionVector { x: 0, y: 0 }, MotionVector { x: 1, y: 1 }),
        ] {
            let (dx, dy) = field_mv_differential(mv, pred);
            let pair = FieldMvPair {
                top: MotionVectorDelta { dx, dy },
                bottom: MotionVectorDelta { dx, dy },
            };
            let recon: FieldMotionVectors = reconstruct_field_motion_vectors(pair, pred.x, pred.y);
            assert_eq!(recon.top, mv);
            assert_eq!(recon.bottom, mv);
        }
    }

    #[test]
    fn estimator_finds_the_cross_field_shift() {
        let p = plane();
        let reference = ReferenceVop::new(&p, 32, 32).unwrap();
        // Source: the reference's *bottom* field placed on both output
        // fields (so the top output field is best predicted from the
        // bottom reference field with a zero vector).
        let mut src = [[0i32; 16]; 16];
        for y in 0..16 {
            for x in 0..16 {
                src[y][x] = i32::from(p[(8 + 2 * (y / 2) + 1) * 32 + 8 + x]);
            }
        }
        let est = estimate_field_motion(
            &src,
            &reference,
            8,
            8,
            false,
            MotionVector { x: 0, y: 0 },
            BVopSampleMode::HalfPel,
            1,
        );
        assert!(est.ref_field, "bottom reference field expected: {est:?}");
        assert_eq!(est.mv, MotionVector { x: 0, y: 0 });
        assert_eq!(est.sad, 0);
        // The prediction through the decoder's mc matches the source.
        let pred = field_luma_prediction(&reference, est.mv, true, 8, 8, BVopSampleMode::HalfPel);
        assert_eq!(pred, source_field(&src, false));
    }

    #[test]
    fn candidates_stay_inside_the_codable_window() {
        let p = plane();
        let reference = ReferenceVop::new(&p, 32, 32).unwrap();
        let src = [[128i32; 16]; 16];
        // A far predictor forces every candidate to the window edge.
        let pred = MotionVector { x: 60, y: 60 };
        let est = estimate_field_motion(
            &src,
            &reference,
            8,
            8,
            true,
            pred,
            BVopSampleMode::HalfPel,
            1,
        );
        let (dx, dy) = field_mv_differential(est.mv, pred);
        assert!(
            (-32..=31).contains(&dx) && (-32..=31).contains(&dy),
            "{est:?}"
        );
    }
}

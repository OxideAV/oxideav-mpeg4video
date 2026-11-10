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
//! component inside that Table 7-9 range — the vertical one in field
//! units, so a field vector reaches twice the frame-line distance of
//! a frame vector (`mpeg4-visual-errata.md` erratum E4, pinned by the
//! `interlaced-field-mv-range` fixture) — and codes the differential
//! `MV - P` reduced into the same range. The estimator therefore
//! searches the full field-unit range whatever the predictor is.
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
use crate::pvop_encode::{SEARCH_RANGE, mv_range, search_window_pels};
use crate::quarter_sample::interpolate_block_qpel_field_into;

/// The estimate for one output field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
/// Candidates span the whole
/// Table 7-9 range of `fcode` on the field grid (`MVx` and `MVy / 2`
/// inside `[low, high]`): the §7.6.3 reduction makes every such
/// vector codable against any predictor (erratum E4 — the vertical
/// range is a field-unit range), so the predictor the differentials
/// are coded against does not narrow the window.
#[allow(clippy::too_many_arguments)]
pub(crate) fn estimate_field_motion(
    src: &[[i32; 16]; 16],
    reference: &ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    field: bool,
    mode: BVopSampleMode,
    fcode: u8,
) -> FieldEstimate {
    let src_field = source_field(src, field);
    let unit = units_per_pel(mode);
    let (low, high) = mv_range(fcode);
    // Codable window in field MV units (horizontal in VOL units,
    // vertical in VOL units of *field* lines): the whole Table 7-9
    // `[low, high]` range on each component's own grid. §7.6.3 reduces
    // the reconstructed component — the vertical one in field units,
    // before the §7.7.2.1 doubling (erratum E4) — so the differential
    // `MV - P` is coded modulo the range width and every vector of the
    // range is reachable from every predictor.
    let clamp_field =
        |fx: i32, fy: i32| -> (i32, i32) { (fx.clamp(low, high), fy.clamp(low, high)) };
    let codable = |dx: i32, dy: i32| -> bool {
        let (fx, fy) = (dx * unit, dy * unit);
        (low..=high).contains(&fx) && (low..=high).contains(&fy)
    };
    // Widest full-pel displacement (pels horizontally, field lines
    // vertically) whose sub-pel neighbours stay inside the range.
    let window = search_window_pels(fcode, mode);

    let mut best: Option<FieldEstimate> = None;
    for ref_field in [false, true] {
        // Full-pel search on the field grid; the zero vector keeps the
        // progressive estimator's small favouring bias.
        let mut best_full = (0i32, 0i32);
        let mut best_full_sad =
            field_sad_full_pel(&src_field, reference, ref_field, mb_x, mb_y, 0, 0)
                .saturating_sub(64);
        let consider = |dx: i32, dy: i32, best: &mut (i32, i32), best_sad: &mut u32| {
            if (dx, dy) == (0, 0) || !codable(dx, dy) {
                return;
            }
            let sad = field_sad_full_pel(&src_field, reference, ref_field, mb_x, mb_y, dx, dy);
            if sad < *best_sad {
                *best_sad = sad;
                *best = (dx, dy);
            }
        };
        for dy in -SEARCH_RANGE..=SEARCH_RANGE {
            for dx in -SEARCH_RANGE..=SEARCH_RANGE {
                consider(dx, dy, &mut best_full, &mut best_full_sad);
            }
        }
        if window > SEARCH_RANGE {
            // Coarse 4-pel lattice over the rest of the window (the
            // vertical axis reaches the full field-unit range, i.e.
            // twice the frame-line reach of a frame vector), clipped
            // so the field block keeps a sample inside the reference,
            // then a dense pass around the lattice winner.
            let (ref_w, ref_h) = (reference.width() as i32, reference.height() as i32);
            let dx_lo = (-window).max(-mb_x - 15);
            let dx_hi = window.min(ref_w - 1 - mb_x);
            let dy_lo = (-window).max((-mb_y - 15) / 2);
            let dy_hi = window.min((ref_h - 1 - mb_y) / 2);
            let mut coarse = best_full;
            let mut coarse_sad = best_full_sad;
            let mut dy = dy_lo;
            while dy <= dy_hi {
                let mut dx = dx_lo;
                while dx <= dx_hi {
                    if dx.abs() > SEARCH_RANGE || dy.abs() > SEARCH_RANGE {
                        consider(dx, dy, &mut coarse, &mut coarse_sad);
                    }
                    dx += 4;
                }
                dy += 4;
            }
            if coarse != best_full {
                let centre = coarse;
                for dy in (centre.1 - 3)..=(centre.1 + 3) {
                    for dx in (centre.0 - 3)..=(centre.0 + 3) {
                        if dx.abs() <= window && dy.abs() <= window {
                            consider(dx, dy, &mut coarse, &mut coarse_sad);
                        }
                    }
                }
                best_full = coarse;
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
        if best.is_none_or(|b| est.sad < b.sad) {
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
    use crate::field_motion::{FieldMotionVectors, reconstruct_field_motion_vectors};
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
        let est = estimate_field_motion(&src, &reference, 8, 8, false, BVopSampleMode::HalfPel, 1);
        assert!(est.ref_field, "bottom reference field expected: {est:?}");
        assert_eq!(est.mv, MotionVector { x: 0, y: 0 });
        assert_eq!(est.sad, 0);
        // The prediction through the decoder's mc matches the source.
        let pred = field_luma_prediction(&reference, est.mv, true, 8, 8, BVopSampleMode::HalfPel);
        assert_eq!(pred, source_field(&src, false));
    }

    #[test]
    fn far_predictor_does_not_narrow_the_window() {
        // Erratum E4: the vertical range is a field-unit range and the
        // differential is coded modulo its width, so a vector at the
        // far side of the range from the predictor stays reachable.
        // Source = the reference's top field 7 field lines up.
        let (w, h) = (48usize, 64usize);
        let mut p = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                p[y * w + x] = ((x * 29 + (y / 2) * 53 + (x * (y / 2)) % 13 * 17) % 251) as u8;
            }
        }
        let reference = ReferenceVop::new(&p, w, h).unwrap();
        let (mb_x, mb_y) = (16i32, 32i32);
        let mut src = [[0i32; 16]; 16];
        for (y, row) in src.iter_mut().enumerate() {
            for (x, s) in row.iter_mut().enumerate() {
                let fy = mb_y as usize + 2 * (y / 2) - 14;
                *s = i32::from(p[fy * w + mb_x as usize + x]);
            }
        }
        let est = estimate_field_motion(
            &src,
            &reference,
            mb_x,
            mb_y,
            false,
            BVopSampleMode::HalfPel,
            1,
        );
        assert_eq!(est.mv, MotionVector { x: 0, y: -28 });
        assert!(!est.ref_field);
        assert_eq!(est.sad, 0);
        // Against a predictor at the opposite end of the range
        // (Py / 2 = 31) the differential leaves [low:high]; it is
        // emitted reduced and the decoder's field-unit reduction
        // restores the vector.
        let pred = MotionVector { x: 30, y: 62 };
        let (dx, dy) = field_mv_differential(est.mv, pred);
        assert_eq!((dx, dy), (-30, -45));
        let reduced = crate::vlc_encode::wrap_mvd_component(dy, 1);
        assert_eq!(reduced, 19);
        let pair = FieldMvPair {
            top: MotionVectorDelta { dx, dy: reduced },
            bottom: MotionVectorDelta { dx, dy: reduced },
        };
        let recon =
            crate::field_motion::reconstruct_field_motion_vectors_wrapped(pair, pred.x, pred.y, 1);
        assert_eq!(recon.top, est.mv);
    }

    #[test]
    fn wide_window_reaches_the_field_unit_range() {
        // fcode 1, half-sample: the field-unit range [-32, 31] spans
        // 15 field lines = 30 frame lines, twice the reach of a frame
        // vector. Source = the top reference field 13 field lines down.
        let (w, h) = (48usize, 96usize);
        let mut p = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                // Smooth, aperiodic texture (the coarse lattice needs
                // a SAD surface with a basin, as natural content has).
                let (fx, ff) = (x as f64, (y / 2) as f64);
                let v = 128.0
                    + 60.0 * (fx / 9.0).sin() * (ff / 6.3).cos()
                    + 40.0 * ((fx + 2.0 * ff) / 17.0).sin()
                    + 0.9 * ff;
                p[y * w + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        let reference = ReferenceVop::new(&p, w, h).unwrap();
        let (mb_x, mb_y) = (16i32, 32i32);
        let mut src = [[0i32; 16]; 16];
        for (y, row) in src.iter_mut().enumerate() {
            for (x, s) in row.iter_mut().enumerate() {
                let fy = mb_y as usize + 2 * (y / 2) + 26;
                *s = i32::from(p[fy * w + mb_x as usize + x]);
            }
        }
        let est = estimate_field_motion(
            &src,
            &reference,
            mb_x,
            mb_y,
            false,
            BVopSampleMode::HalfPel,
            1,
        );
        // 13 field lines = 26 field half-samples = 52 frame units.
        assert_eq!(est.mv, MotionVector { x: 0, y: 52 });
        assert_eq!(est.sad, 0);
    }
}

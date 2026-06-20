//! §6.3.6 / §7.8.7.1 S(GMC)-VOP macroblock prediction routing.
//!
//! In an S(GMC)-VOP every inter macroblock carries an `mcsel` flag
//! (§6.3.6). Per §7.8.7.1:
//!
//! * `mcsel == 1` — the macroblock (including its grayscale alpha) is
//!   predicted by **global motion compensation**: the §7.8.4/§7.8.5/
//!   §7.8.6 warp is applied to the reference VOP. No local block motion
//!   vectors are transmitted.
//! * `mcsel == 0` — the macroblock is predicted exactly as in a P-VOP,
//!   using the transmitted local motion vector(s) and the §7.6.2 half-
//!   or §7.6.3 quarter-sample interpolation.
//!
//! This module supplies [`s_gmc_prediction_macroblock`], the gate that
//! selects between those two prediction stages and produces a single
//! [`InterPredictionMacroblock`] ready for the §7.3 step-2 residual add.
//! The GMC branch is generated here from the warp geometry + reference
//! planes; the translational branch is built by the existing
//! half/quarter-sample machinery (the caller owns the per-block MVs and
//! passes the finished translational prediction in).
//!
//! Per §7.8.7.1's note, GMC prediction is performed using **frame**
//! prediction even when `interlaced == 1` (the warp is identical to the
//! progressive case), so this gate is field-agnostic.

use crate::gmc::{gmc_chroma_prediction, gmc_luma_prediction};
use crate::half_sample::ReferenceVop;
use crate::reconstruct::{InterPredictionMacroblock, MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};
use crate::warp::WarpGeometry;

/// The reference VOP planes (luma + the two chroma components) an
/// S(GMC)-VOP warps from. Each plane is a reconstructed reference frame
/// with §7.6.4 edge clamping handled by [`ReferenceVop`].
#[derive(Debug)]
pub struct GmcReferencePlanes<'a> {
    /// Luminance reference plane (full resolution).
    pub luma: ReferenceVop<'a>,
    /// Cb reference plane (half resolution, 4:2:0).
    pub cb: ReferenceVop<'a>,
    /// Cr reference plane (half resolution, 4:2:0).
    pub cr: ReferenceVop<'a>,
}

/// Build the §7.8.7.1 GMC prediction macroblock whose top-left luminance
/// pixel is at VOP coordinate `(mb_x, mb_y)` (luma pixels; the chroma
/// origin is `(mb_x / 2, mb_y / 2)` per 4:2:0).
///
/// `rounding_control` is the VOP-header `vop_rounding_type` (§7.6.2);
/// `bits_per_pixel` is from the VOL header.
pub fn gmc_prediction_macroblock(
    geometry: &WarpGeometry,
    reference: &GmcReferencePlanes<'_>,
    mb_x: i64,
    mb_y: i64,
    rounding_control: u8,
    bits_per_pixel: u32,
) -> InterPredictionMacroblock {
    let luma_pred = gmc_luma_prediction(
        geometry,
        &reference.luma,
        mb_x,
        mb_y,
        rounding_control,
        bits_per_pixel,
    );
    let cb_x = mb_x / 2;
    let cb_y = mb_y / 2;
    let cb_pred = gmc_chroma_prediction(
        geometry,
        &reference.cb,
        cb_x,
        cb_y,
        rounding_control,
        bits_per_pixel,
    );
    let cr_pred = gmc_chroma_prediction(
        geometry,
        &reference.cr,
        cb_x,
        cb_y,
        rounding_control,
        bits_per_pixel,
    );

    let mut out = InterPredictionMacroblock::zero();
    for y in 0..MACROBLOCK_LUMA_SIDE {
        for x in 0..MACROBLOCK_LUMA_SIDE {
            out.luma[y][x] = i32::from(luma_pred[y * MACROBLOCK_LUMA_SIDE + x]);
        }
    }
    for y in 0..MACROBLOCK_CHROMA_SIDE {
        for x in 0..MACROBLOCK_CHROMA_SIDE {
            out.cb[y][x] = i32::from(cb_pred[y * MACROBLOCK_CHROMA_SIDE + x]);
            out.cr[y][x] = i32::from(cr_pred[y * MACROBLOCK_CHROMA_SIDE + x]);
        }
    }
    out
}

/// §7.8.7.1 prediction-stage selection for one S(GMC)-VOP inter
/// macroblock.
///
/// * `mcsel == true` → return the GMC-warped prediction (built here from
///   `geometry` + `reference`).
/// * `mcsel == false` → return `translational`, the local-MC prediction
///   the caller already produced via the §7.6.2/§7.6.3 interpolation
///   path (the warp parameters are not used).
///
/// `mb_x`/`mb_y` are the luma top-left of the macroblock;
/// `rounding_control` / `bits_per_pixel` as in
/// [`gmc_prediction_macroblock`].
#[allow(clippy::too_many_arguments)]
pub fn s_gmc_prediction_macroblock(
    mcsel: bool,
    geometry: &WarpGeometry,
    reference: &GmcReferencePlanes<'_>,
    mb_x: i64,
    mb_y: i64,
    rounding_control: u8,
    bits_per_pixel: u32,
    translational: InterPredictionMacroblock,
) -> InterPredictionMacroblock {
    if mcsel {
        gmc_prediction_macroblock(
            geometry,
            reference,
            mb_x,
            mb_y,
            rounding_control,
            bits_per_pixel,
        )
    } else {
        translational
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite::SpriteTrajectory;
    use crate::vol::SpriteWarpingAccuracy;

    fn flat_planes(value: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // 64×64 luma, 32×32 chroma planes, all flat.
        (
            vec![value; 64 * 64],
            vec![value; 32 * 32],
            vec![value; 32 * 32],
        )
    }

    fn ramp_luma(w: usize, h: usize) -> Vec<u8> {
        let mut d = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                d[r * w + c] = ((r * 3 + c) & 0xff) as u8;
            }
        }
        d
    }

    #[test]
    fn mcsel_false_returns_translational_unchanged() {
        // The GMC branch must be entirely bypassed when mcsel == 0.
        let (ly, cb, cr) = flat_planes(0);
        let reference = GmcReferencePlanes {
            luma: ReferenceVop::new(&ly, 64, 64).unwrap(),
            cb: ReferenceVop::new(&cb, 32, 32).unwrap(),
            cr: ReferenceVop::new(&cr, 32, 32).unwrap(),
        };
        let traj = SpriteTrajectory::stationary();
        let geo = WarpGeometry::decode(&traj, 64, 64, SpriteWarpingAccuracy::HalfPel);

        let mut translational = InterPredictionMacroblock::zero();
        translational.luma[0][0] = 200;
        translational.cb[1][2] = 111;
        let expected = translational.clone();

        let out = s_gmc_prediction_macroblock(false, &geo, &reference, 0, 0, 0, 8, translational);
        assert_eq!(out, expected, "mcsel==0 must pass translational through");
    }

    #[test]
    fn mcsel_true_uses_gmc_warp_not_translational() {
        // Flat reference value 90 ⇒ GMC prediction is flat 90 everywhere,
        // independent of the (bogus) translational prediction passed in.
        let (ly, cb, cr) = flat_planes(90);
        let reference = GmcReferencePlanes {
            luma: ReferenceVop::new(&ly, 64, 64).unwrap(),
            cb: ReferenceVop::new(&cb, 32, 32).unwrap(),
            cr: ReferenceVop::new(&cr, 32, 32).unwrap(),
        };
        let traj = SpriteTrajectory::stationary();
        let geo = WarpGeometry::decode(&traj, 64, 64, SpriteWarpingAccuracy::HalfPel);

        let mut bogus = InterPredictionMacroblock::zero();
        bogus.luma[0][0] = 7; // must be ignored

        let out = s_gmc_prediction_macroblock(true, &geo, &reference, 0, 0, 0, 8, bogus);
        for row in &out.luma {
            assert!(row.iter().all(|&p| p == 90));
        }
        for row in &out.cb {
            assert!(row.iter().all(|&p| p == 90));
        }
        for row in &out.cr {
            assert!(row.iter().all(|&p| p == 90));
        }
    }

    #[test]
    fn gmc_macroblock_matches_underlying_gmc_prediction() {
        // The router's GMC branch must agree sample-for-sample with the
        // direct gmc_luma_prediction / gmc_chroma_prediction calls.
        let ly = ramp_luma(64, 64);
        let cb = ramp_luma(32, 32);
        let cr = ramp_luma(32, 32);
        let reference = GmcReferencePlanes {
            luma: ReferenceVop::new(&ly, 64, 64).unwrap(),
            cb: ReferenceVop::new(&cb, 32, 32).unwrap(),
            cr: ReferenceVop::new(&cr, 32, 32).unwrap(),
        };
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 2;
        traj.points = [[3, -2], [1, 4], [0, 0]];
        let geo = WarpGeometry::decode(&traj, 48, 48, SpriteWarpingAccuracy::QuarterPel);

        let mb_x = 16i64;
        let mb_y = 16i64;
        let out = gmc_prediction_macroblock(&geo, &reference, mb_x, mb_y, 0, 8);

        let direct_luma = gmc_luma_prediction(&geo, &reference.luma, mb_x, mb_y, 0, 8);
        let direct_cb = gmc_chroma_prediction(&geo, &reference.cb, mb_x / 2, mb_y / 2, 0, 8);
        let direct_cr = gmc_chroma_prediction(&geo, &reference.cr, mb_x / 2, mb_y / 2, 0, 8);

        for y in 0..MACROBLOCK_LUMA_SIDE {
            for x in 0..MACROBLOCK_LUMA_SIDE {
                assert_eq!(
                    out.luma[y][x],
                    i32::from(direct_luma[y * MACROBLOCK_LUMA_SIDE + x]),
                    "luma ({x},{y})"
                );
            }
        }
        for y in 0..MACROBLOCK_CHROMA_SIDE {
            for x in 0..MACROBLOCK_CHROMA_SIDE {
                assert_eq!(
                    out.cb[y][x],
                    i32::from(direct_cb[y * MACROBLOCK_CHROMA_SIDE + x]),
                    "cb ({x},{y})"
                );
                assert_eq!(
                    out.cr[y][x],
                    i32::from(direct_cr[y * MACROBLOCK_CHROMA_SIDE + x]),
                    "cr ({x},{y})"
                );
            }
        }
    }
}

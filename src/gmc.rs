//! §7.8.6 sample reconstruction for GMC (global motion compensation).
//!
//! Given a [`crate::warp::WarpGeometry`] (the §7.8.4/§7.8.5 warp) and a
//! reference VOP plane, this module produces the GMC prediction for a
//! macroblock by, for every destination pixel, evaluating `(F, G)`
//! (luma) or `(Fc, Gc)` (chroma), splitting it into an integer
//! reference position plus the `1/s`-pel residuals `ri` / `rj`, and
//! bilinearly blending the four surrounding reference samples per the
//! §7.8.6 `sprite_enable == "GMC"` formula:
//!
//! ```text
//! Y = ((s - rj)((s - ri) Y00 + ri Y01)
//!      + rj ((s - ri) Y10 + ri Y11)
//!      + s²/2 - rounding_control) / s²
//! ```
//!
//! where `Y00 = ref(F////s, G////s)`, `Y01 = ref(F////s + 1, G////s)`,
//! `Y10 = ref(F////s, G////s + 1)`, `Y11 = ref(F////s + 1, G////s + 1)`,
//! `ri = F - (F////s) s`, `rj = G - (G////s) s`, and `////` is integer
//! division with truncation toward negative infinity (§3.4).
//! `rounding_control` is the VOP-header `vop_rounding_type` (§7.6.2).
//! Out-of-bounds reference fetches use the §7.6.4 last-full-pel clamp
//! (the rectangular-S(GMC)-VOP bounding-rectangle case).

use crate::half_sample::ReferenceVop;
use crate::warp::WarpGeometry;

/// One 16×16 luminance macroblock of GMC prediction samples.
pub const MB_LUMA_SIDE: usize = 16;
/// One 8×8 chrominance block side (4:2:0).
pub const MB_CHROMA_SIDE: usize = 8;

/// `n //// d` (§3.4): integer division with truncation toward negative
/// infinity. `d` must be positive.
#[inline]
fn div_floor(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    let q = n / d;
    let r = n % d;
    if r != 0 && (r < 0) {
        q - 1
    } else {
        q
    }
}

/// Reconstruct one §7.8.6 GMC sample at warp coordinate `(cap_f, cap_g)`
/// (in `1/s`-pel units) from the reference plane.
///
/// `s` is the sprite-warping denominator, `rc` the `vop_rounding_type`.
/// Returns the clamped 8-bit (well, `[0, 2^bpp - 1]`-ranged) sample.
#[inline]
fn reconstruct_sample(
    reference: &ReferenceVop<'_>,
    cap_f: i64,
    cap_g: i64,
    s: i64,
    rc: i64,
    bpp: u32,
) -> u16 {
    // Integer reference position (toward -inf) and 1/s-pel residuals.
    let fx = div_floor(cap_f, s);
    let fy = div_floor(cap_g, s);
    let ri = cap_f - fx * s;
    let rj = cap_g - fy * s;

    // The four neighbour samples (with §7.6.4 last-full-pel clamping).
    let y00 = i64::from(reference.fetch_clamped(fx as i32, fy as i32));
    let y01 = i64::from(reference.fetch_clamped((fx + 1) as i32, fy as i32));
    let y10 = i64::from(reference.fetch_clamped(fx as i32, (fy + 1) as i32));
    let y11 = i64::from(reference.fetch_clamped((fx + 1) as i32, (fy + 1) as i32));

    // GMC bilinear blend.
    let numer =
        (s - rj) * ((s - ri) * y00 + ri * y01) + rj * ((s - ri) * y10 + ri * y11) + s * s / 2 - rc;
    let value = numer / (s * s);

    // Clip to [0, 2^bpp - 1].
    let max = (1i64 << bpp) - 1;
    value.clamp(0, max) as u16
}

/// Generate the §7.8.6 GMC luminance prediction for a macroblock whose
/// top-left pixel is at VOP coordinate `(mb_x, mb_y)`.
///
/// Fills a `MB_LUMA_SIDE × MB_LUMA_SIDE` block in row-major order. Each
/// output pixel `(x, y)` (`0 <= x,y < 16`) is the GMC prediction for VOP
/// pixel `(mb_x + x, mb_y + y)`. `rounding_control` is the VOP-header
/// `vop_rounding_type`; `bits_per_pixel` is from the VOL header
/// (typically 8).
pub fn gmc_luma_prediction(
    geometry: &WarpGeometry,
    reference: &ReferenceVop<'_>,
    mb_x: i64,
    mb_y: i64,
    rounding_control: u8,
    bits_per_pixel: u32,
) -> [u8; MB_LUMA_SIDE * MB_LUMA_SIDE] {
    let mut out = [0u8; MB_LUMA_SIDE * MB_LUMA_SIDE];
    let s = geometry.s;
    let rc = i64::from(rounding_control);
    for y in 0..MB_LUMA_SIDE {
        for x in 0..MB_LUMA_SIDE {
            let i = mb_x + x as i64;
            let j = mb_y + y as i64;
            let [cap_f, cap_g] = geometry.luma_fg(i, j);
            let sample = reconstruct_sample(reference, cap_f, cap_g, s, rc, bits_per_pixel);
            out[y * MB_LUMA_SIDE + x] = sample as u8;
        }
    }
    out
}

/// Generate the §7.8.6 GMC chrominance prediction for an 8×8 chroma
/// block whose top-left chroma pixel is at chroma coordinate
/// `(cb_x, cb_y)` (luma `/ 2` for 4:2:0).
///
/// Fills a `MB_CHROMA_SIDE × MB_CHROMA_SIDE` block in row-major order
/// using the §7.8.5 chroma warp `(Fc, Gc)`.
pub fn gmc_chroma_prediction(
    geometry: &WarpGeometry,
    reference: &ReferenceVop<'_>,
    cb_x: i64,
    cb_y: i64,
    rounding_control: u8,
    bits_per_pixel: u32,
) -> [u8; MB_CHROMA_SIDE * MB_CHROMA_SIDE] {
    let mut out = [0u8; MB_CHROMA_SIDE * MB_CHROMA_SIDE];
    let s = geometry.s;
    let rc = i64::from(rounding_control);
    for y in 0..MB_CHROMA_SIDE {
        for x in 0..MB_CHROMA_SIDE {
            let ic = cb_x + x as i64;
            let jc = cb_y + y as i64;
            let [cap_fc, cap_gc] = geometry.chroma_fg(ic, jc);
            let sample = reconstruct_sample(reference, cap_fc, cap_gc, s, rc, bits_per_pixel);
            out[y * MB_CHROMA_SIDE + x] = sample as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite::SpriteTrajectory;
    use crate::vol::SpriteWarpingAccuracy;

    fn flat_reference(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    #[test]
    fn div_floor_truncates_toward_negative_infinity() {
        assert_eq!(div_floor(7, 4), 1);
        assert_eq!(div_floor(-7, 4), -2);
        assert_eq!(div_floor(8, 4), 2);
        assert_eq!(div_floor(-8, 4), -2);
        assert_eq!(div_floor(-1, 4), -1);
    }

    #[test]
    fn flat_reference_yields_flat_prediction() {
        // A constant reference plane must warp to that same constant
        // everywhere, regardless of the (well-formed) warp.
        let w = 64;
        let h = 64;
        let refbuf = flat_reference(w, h, 137);
        let reference = ReferenceVop::new(&refbuf, w, h).unwrap();
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 2;
        traj.points = [[2, -2], [4, 1], [0, 0]];
        let geo = WarpGeometry::decode(&traj, w as u32, h as u32, SpriteWarpingAccuracy::HalfPel);
        let pred = gmc_luma_prediction(&geo, &reference, 0, 0, 0, 8);
        assert!(pred.iter().all(|&p| p == 137), "flat prediction expected");
    }

    #[test]
    fn zero_point_identity_copies_reference() {
        // 0 warping points (stationary): F = s i, G = s j ⇒ ri = rj = 0
        // ⇒ the prediction is a straight integer-pel copy of the
        // reference (Y == Y00).
        let w = 32;
        let h = 32;
        // A gradient so a copy is distinguishable from a blur.
        let mut refbuf = vec![0u8; w * h];
        for (idx, p) in refbuf.iter_mut().enumerate() {
            *p = ((idx % w) as u8).wrapping_mul(3);
        }
        let reference = ReferenceVop::new(&refbuf, w, h).unwrap();
        let traj = SpriteTrajectory::stationary();
        let geo = WarpGeometry::decode(&traj, w as u32, h as u32, SpriteWarpingAccuracy::HalfPel);
        let pred = gmc_luma_prediction(&geo, &reference, 0, 0, 0, 8);
        for y in 0..MB_LUMA_SIDE {
            for x in 0..MB_LUMA_SIDE {
                let expected = refbuf[y * w + x];
                assert_eq!(pred[y * MB_LUMA_SIDE + x], expected, "pixel ({x},{y})");
            }
        }
    }

    #[test]
    fn one_point_integer_translation_shifts_reference() {
        // Single warping point with du0 = +2·s/(s/2)... choose s=2 and
        // du0 = +2 ⇒ i0' = (s/2)·du0 = 1·2 = 2 (1/s-pel = 1/2 pel).
        // That's a 1-pel shift in 1/s units? i0'=2, s=2 ⇒ shift of
        // 2/2 = 1 integer pel. F = i0' + s·i = 2 + 2 i ⇒ fx = (2+2i)/2 =
        // i+1 ⇒ the prediction samples ref(x+1, y).
        let w = 32;
        let h = 32;
        let mut refbuf = vec![0u8; w * h];
        for (idx, p) in refbuf.iter_mut().enumerate() {
            *p = (idx % w) as u8; // column ramp 0,1,2,...
        }
        let reference = ReferenceVop::new(&refbuf, w, h).unwrap();
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 1;
        traj.points[0] = [2, 0];
        let geo = WarpGeometry::decode(&traj, w as u32, h as u32, SpriteWarpingAccuracy::HalfPel);
        let pred = gmc_luma_prediction(&geo, &reference, 0, 0, 0, 8);
        // pred(x,y) should equal ref(x+1, y) = (x+1) for x+1 < w.
        for y in 0..MB_LUMA_SIDE {
            for x in 0..MB_LUMA_SIDE {
                assert_eq!(
                    pred[y * MB_LUMA_SIDE + x],
                    refbuf[y * w + (x + 1)],
                    "pixel ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn half_pel_shift_averages_two_columns() {
        // s=2, du0=+1 ⇒ i0'=(s/2)·1=1 ⇒ F = 1 + 2 i ⇒ fx = (1+2i)/2 = i
        // (floor), ri = (1+2i) - i·2 = 1. With ri=1, s=2: the blend is
        // ((s-rj)((s-ri)Y00 + ri Y01) + ... )/s², rj=0 ⇒
        // Y = ((s)((s-1)Y00 + 1·Y01) + s²/2 - rc)/s²
        //   = (2(Y00 + Y01) + 2 - rc)/4. With rc=0 that's
        //   (2 Y00 + 2 Y01 + 2)/4 = (Y00 + Y01 + 1)/2 rounded.
        let w = 32;
        let h = 32;
        let mut refbuf = vec![0u8; w * h];
        for (idx, p) in refbuf.iter_mut().enumerate() {
            *p = ((idx % w) * 4) as u8; // 0,4,8,... so averages are exact
        }
        let reference = ReferenceVop::new(&refbuf, w, h).unwrap();
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 1;
        traj.points[0] = [1, 0];
        let geo = WarpGeometry::decode(&traj, w as u32, h as u32, SpriteWarpingAccuracy::HalfPel);
        let pred = gmc_luma_prediction(&geo, &reference, 0, 0, 0, 8);
        for y in 0..MB_LUMA_SIDE {
            for x in 0..MB_LUMA_SIDE {
                let y00 = i64::from(refbuf[y * w + x]);
                let y01 = i64::from(refbuf[y * w + x + 1]);
                let expected = ((2 * y00 + 2 * y01 + 2) / 4) as u8;
                assert_eq!(pred[y * MB_LUMA_SIDE + x], expected, "pixel ({x},{y})");
            }
        }
    }

    #[test]
    fn chroma_flat_reference_is_flat() {
        let w = 32;
        let h = 32;
        let refbuf = flat_reference(w, h, 200);
        let reference = ReferenceVop::new(&refbuf, w, h).unwrap();
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 2;
        traj.points = [[1, 1], [3, -2], [0, 0]];
        let geo = WarpGeometry::decode(&traj, 64, 64, SpriteWarpingAccuracy::QuarterPel);
        let pred = gmc_chroma_prediction(&geo, &reference, 0, 0, 0, 8);
        assert!(pred.iter().all(|&p| p == 200));
    }

    #[test]
    fn rounding_control_biases_half_cases_down() {
        // With rc=1 the s²/2 - rc term shifts a tie downward. Build a
        // pure half-pel average where the unrounded value is exactly a
        // half-integer.
        let w = 32;
        let h = 32;
        let mut refbuf = vec![0u8; w * h];
        for (idx, p) in refbuf.iter_mut().enumerate() {
            // Adjacent columns 10,11 give an odd sum Y00+Y01, so the
            // half-pel blend lands on a tie that the rc bias rounds down.
            *p = ((idx % w) % 2 + 10) as u8; // 10,11,10,11,...
        }
        let reference = ReferenceVop::new(&refbuf, w, h).unwrap();
        let mut traj = SpriteTrajectory::stationary();
        traj.count = 1;
        traj.points[0] = [1, 0]; // half-pel shift, ri=1
        let geo = WarpGeometry::decode(&traj, w as u32, h as u32, SpriteWarpingAccuracy::HalfPel);
        let pred0 = gmc_luma_prediction(&geo, &reference, 0, 0, 0, 8);
        let pred1 = gmc_luma_prediction(&geo, &reference, 0, 0, 1, 8);
        // rc=1 must never produce a larger sample than rc=0 (it only
        // ever rounds a tie downward).
        for (a, b) in pred0.iter().zip(pred1.iter()) {
            assert!(b <= a, "rc=1 sample {b} should be <= rc=0 sample {a}");
        }
        // And at least one pixel differs (the half-case bias fires).
        assert!(pred0.iter().zip(pred1.iter()).any(|(a, b)| a != b));
    }
}

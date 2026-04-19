//! Motion compensation for MPEG-4 Part 2 P-VOPs (§7.6.2).
//!
//! Half-pel resolution with bilinear filter, optional unrestricted-MV
//! domain (UMV — clamped to picture boundaries via edge replication).
//!
//! Quarter-pel motion (§7.6.2.2) is NOT enabled in this build because the
//! VOL parser currently rejects `quarter_sample == 1` up front. When
//! quarter-pel arrives in a follow-up the chroma and luma filters here will
//! need to switch to the 8-tap interpolator for the half-pel sub-positions.

/// Predict an `n × n` block from `ref_plane` into `dst`. `mv_x_half` and
/// `mv_y_half` are in half-pel units relative to the block's natural
/// position `(blk_px, blk_py)` in the reference picture.
///
/// `rounding` is the `vop_rounding_type` flag from the VOP header — when
/// set, the half-pel filter rounds to floor instead of nearest (§7.6.2.1
/// equation (105)).
#[allow(clippy::too_many_arguments)]
pub fn predict_block(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    blk_px: i32,
    blk_py: i32,
    mv_x_half: i32,
    mv_y_half: i32,
    n: i32,
    rounding: bool,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let int_x = mv_x_half >> 1;
    let int_y = mv_y_half >> 1;
    let hx = (mv_x_half & 1) != 0;
    let hy = (mv_y_half & 1) != 0;

    let src_x = blk_px + int_x;
    let src_y = blk_py + int_y;

    // §7.6.2.1 half-pel filter — bilinear with rounding offset 1 normally,
    // 0 when `rounding` is set (vop_rounding_type=1).
    let round = if rounding { 0 } else { 1 };
    let round2 = if rounding { 1 } else { 2 };

    // Fast path: the block (plus the 1-pel half-pel tap right/below) lies
    // entirely inside the reference plane, so no edge-replication clamp
    // is ever triggered. This is overwhelmingly the common case for
    // typical encoders: MVs stay inside the picture and only the rare
    // edge-adjacent MB needs the clamped walk.
    let tap_x = if hx { 1 } else { 0 };
    let tap_y = if hy { 1 } else { 0 };
    if src_x >= 0 && src_y >= 0 && src_x + n + tap_x <= ref_w && src_y + n + tap_y <= ref_h {
        predict_block_interior(
            ref_plane, ref_stride, src_x as usize, src_y as usize, hx, hy, n as usize, round,
            round2, dst, dst_stride,
        );
        return;
    }

    // Clamp helpers — replicate edges (unrestricted MV domain §7.6.4).
    let sample = |x: i32, y: i32| -> u32 {
        let xc = x.clamp(0, ref_w - 1) as usize;
        let yc = y.clamp(0, ref_h - 1) as usize;
        ref_plane[yc * ref_stride + xc] as u32
    };

    for j in 0..n {
        for i in 0..n {
            let v = match (hx, hy) {
                (false, false) => sample(src_x + i, src_y + j),
                (true, false) => {
                    let a = sample(src_x + i, src_y + j);
                    let b = sample(src_x + i + 1, src_y + j);
                    (a + b + round) >> 1
                }
                (false, true) => {
                    let a = sample(src_x + i, src_y + j);
                    let b = sample(src_x + i, src_y + j + 1);
                    (a + b + round) >> 1
                }
                (true, true) => {
                    let a = sample(src_x + i, src_y + j);
                    let b = sample(src_x + i + 1, src_y + j);
                    let c = sample(src_x + i, src_y + j + 1);
                    let d = sample(src_x + i + 1, src_y + j + 1);
                    (a + b + c + d + round2) >> 2
                }
            };
            dst[(j as usize) * dst_stride + (i as usize)] = v as u8;
        }
    }
}

/// Interior-only half-pel predictor: whole footprint guaranteed in
/// bounds, so no per-pel clamping. The three sub-pel branches unroll
/// into straight memory reads + add/shifts that the auto-vectoriser
/// lowers to `vpmovzxbw` / `vpaddw` / `vpsrlw` sequences.
#[allow(clippy::too_many_arguments)]
#[inline]
fn predict_block_interior(
    ref_plane: &[u8],
    ref_stride: usize,
    src_x: usize,
    src_y: usize,
    hx: bool,
    hy: bool,
    n: usize,
    round: u32,
    round2: u32,
    dst: &mut [u8],
    dst_stride: usize,
) {
    match (hx, hy) {
        (false, false) => {
            // Straight integer-pel copy.
            for j in 0..n {
                let s = (src_y + j) * ref_stride + src_x;
                let d = j * dst_stride;
                dst[d..d + n].copy_from_slice(&ref_plane[s..s + n]);
            }
        }
        (true, false) => {
            for j in 0..n {
                let s = (src_y + j) * ref_stride + src_x;
                let a = &ref_plane[s..s + n];
                let b = &ref_plane[s + 1..s + 1 + n];
                let d = j * dst_stride;
                for i in 0..n {
                    dst[d + i] = ((a[i] as u32 + b[i] as u32 + round) >> 1) as u8;
                }
            }
        }
        (false, true) => {
            for j in 0..n {
                let s0 = (src_y + j) * ref_stride + src_x;
                let s1 = (src_y + j + 1) * ref_stride + src_x;
                let a = &ref_plane[s0..s0 + n];
                let b = &ref_plane[s1..s1 + n];
                let d = j * dst_stride;
                for i in 0..n {
                    dst[d + i] = ((a[i] as u32 + b[i] as u32 + round) >> 1) as u8;
                }
            }
        }
        (true, true) => {
            for j in 0..n {
                let s0 = (src_y + j) * ref_stride + src_x;
                let s1 = (src_y + j + 1) * ref_stride + src_x;
                let a = &ref_plane[s0..s0 + n];
                let b = &ref_plane[s0 + 1..s0 + 1 + n];
                let c = &ref_plane[s1..s1 + n];
                let e = &ref_plane[s1 + 1..s1 + 1 + n];
                let d = j * dst_stride;
                for i in 0..n {
                    dst[d + i] =
                        ((a[i] as u32 + b[i] as u32 + c[i] as u32 + e[i] as u32 + round2) >> 2)
                            as u8;
                }
            }
        }
    }
}

/// Compute the chroma motion vector from the luma vector per §7.6.2.1.
/// MPEG-4 uses a "round to nearest half-pel" rule: the chroma component is
/// the luma component divided by 2 with the resulting fractional part
/// requantised to the half-pel grid.
///
/// Implementation per FFmpeg `chroma_4mv_motion_lowres`-style logic:
///   chroma = (luma >> 1) | (luma & 1)
/// equivalently `(luma + sign(luma)) / 2` with halfpel preserved.
///
/// We work in luma half-pel units throughout. Returned value is in chroma
/// half-pel units.
pub fn luma_mv_to_chroma(luma_mv_half: i32) -> i32 {
    // Derivation from FFmpeg `mpeg_motion_internal` (1MV H.263 path):
    //   chroma_int_offset = luma_mv >> 2          (signed, floor)
    //   chroma_half_bit   = 1 iff (luma_mv & 3) != 0
    //   chroma_mv_half    = chroma_int_offset * 2 + chroma_half_bit
    //
    // Worked examples (luma_mv → chroma_mv, both in their respective half-pel units):
    //   0 → 0,  1 → 1,  2 → 1,  3 → 1,  4 → 2,  5 → 3,  6 → 3,  7 → 3,  8 → 4
    //   −1 → −1, −2 → −1, −3 → −1, −4 → −2, −5 → −3, −6 → −3, −7 → −3, −8 → −4
    //
    // For non-negative luma the values match Table 7-15 of the spec.
    let int_part = luma_mv_half >> 2;
    let half_bit = if luma_mv_half & 3 != 0 { 1 } else { 0 };
    int_part * 2 + half_bit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_integer_copy() {
        // 4x4 ref plane.
        let refp: [u8; 16] = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33];
        let mut dst = [0u8; 4];
        predict_block(&refp, 4, 4, 4, 0, 0, 0, 0, 2, false, &mut dst, 2);
        assert_eq!(dst, [0, 1, 10, 11]);
        // MV (2,0) half = +1 pel.
        predict_block(&refp, 4, 4, 4, 0, 0, 2, 0, 2, false, &mut dst, 2);
        assert_eq!(dst, [1, 2, 11, 12]);
    }

    #[test]
    fn predict_half_pel_h() {
        let refp: [u8; 16] = [0, 10, 20, 30, 0, 10, 20, 30, 0, 10, 20, 30, 0, 10, 20, 30];
        let mut dst = [0u8; 4];
        predict_block(&refp, 4, 4, 4, 0, 0, 1, 0, 2, false, &mut dst, 2);
        // (0+10+1)/2=5, (10+20+1)/2=15, ...
        assert_eq!(dst, [5, 15, 5, 15]);
    }

    #[test]
    fn rounding_flag_floors() {
        // With rounding=true, +0 instead of +1 → (0+10)/2 = 5, (10+20)/2 = 15
        // (no change for these but test the (1,1) case).
        let refp: [u8; 16] = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut dst1 = [0u8; 1];
        predict_block(&refp, 4, 4, 4, 0, 0, 1, 1, 1, false, &mut dst1, 1);
        // (0+1+1+1+2)/4 = 5/4 = 1 (rounding off -> +2 offset)
        assert_eq!(dst1[0], 1);
        let mut dst2 = [0u8; 1];
        predict_block(&refp, 4, 4, 4, 0, 0, 1, 1, 1, true, &mut dst2, 1);
        // (0+1+1+1+1)/4 = 4/4 = 1 (rounding on -> +1 offset)
        assert_eq!(dst2[0], 1);
    }

    #[test]
    fn chroma_mv_mapping() {
        // Table per FFmpeg `mpeg_motion_internal` 1MV H.263 path (above).
        let expected: &[(i32, i32)] = &[
            (-8, -4),
            (-7, -3),
            (-6, -3),
            (-5, -3),
            (-4, -2),
            (-3, -1),
            (-2, -1),
            (-1, -1),
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 2),
            (5, 3),
            (6, 3),
            (7, 3),
            (8, 4),
        ];
        for &(luma, chroma) in expected {
            assert_eq!(
                luma_mv_to_chroma(luma),
                chroma,
                "luma {luma} -> expected chroma {chroma}, got {}",
                luma_mv_to_chroma(luma)
            );
        }
    }
}

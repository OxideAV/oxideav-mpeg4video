//! Motion compensation for MPEG-4 Part 2 P-VOPs (§7.6.2).
//!
//! Half-pel resolution with bilinear filter, optional unrestricted-MV
//! domain (UMV — clamped to picture boundaries via edge replication).
//!
//! Quarter-pel motion (§7.6.2.2) — 8-tap filter for the half-sample
//! positions, then bilinear averaging to reach quarter-sample positions.
//! Selected by the VOL `quarter_sample` flag; chroma remains on the
//! half-pel grid regardless (chroma MV is derived from the luma MV via
//! `luma_qmv_to_chroma` when QPel is active).

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
            ref_plane,
            ref_stride,
            src_x as usize,
            src_y as usize,
            hx,
            hy,
            n as usize,
            round,
            round2,
            dst,
            dst_stride,
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
                    dst[d + i] = ((a[i] as u32 + b[i] as u32 + c[i] as u32 + e[i] as u32 + round2)
                        >> 2) as u8;
                }
            }
        }
    }
}

/// Compute the chroma motion vector from a single luma vector (1MV mode)
/// per ISO/IEC 14496-2 §7.6.2.1 / §7.6.5 Table 7-13. The chroma component
/// is the luma component divided by 2 with the fractional part requantised
/// to the half-pel grid.
///
/// We work in luma half-pel units throughout. Returned value is in chroma
/// half-pel units.
///
/// Worked examples (luma_mv → chroma_mv, both in their respective half-pel units):
///   0 → 0,  1 → 1,  2 → 1,  3 → 1,  4 → 2,  5 → 3,  6 → 3,  7 → 3,  8 → 4
///   −1 → −1, −2 → −1, −3 → −1, −4 → −2, −5 → −3, −6 → −3, −7 → −3, −8 → −4
pub fn luma_mv_to_chroma(luma_mv_half: i32) -> i32 {
    let int_part = luma_mv_half >> 2;
    let half_bit = if luma_mv_half & 3 != 0 { 1 } else { 0 };
    int_part * 2 + half_bit
}

/// Compute the chroma motion vector for the 4MV mode (Inter4MV) per
/// ISO/IEC 14496-2 §7.6.5 + Table 7-10. Takes the SUM of the four
/// luma motion vector components (in luma half-pel units) and returns
/// the chroma motion vector (in chroma half-pel units).
///
/// Algorithm (§7.6.5):
/// 1. `MVDCHR_sixteenth = sum * 4 / K = sum` for K=4 (in 1/16 chroma
///    sample units; the integer-arithmetic path falls out because each
///    luma half-pel = 1/4 chroma full-pel = 4/16 chroma sample, and
///    `(sum * 4 / 16) / (2K=8)` = `sum * 4 / (16 * 8)` per luma block,
///    multiplied by K=4 luma blocks summed = `sum * 4 / 32 * 16/K` —
///    after the K cancellation `MVDCHR_sixteenth = sum` for K=4).
/// 2. Split `MVDCHR_sixteenth` into integer + fractional parts in 1/2
///    chroma sample units: `int = MVDCHR_sixteenth / 16`, `frac =
///    MVDCHR_sixteenth mod 16` (sign-aware).
/// 3. Map `frac` through Table 7-10 to a half-sample modifier in
///    {0, 1, 2}.
/// 4. Final chroma MV (in chroma half-pel) = `int * 2 + sign(int) *
///    table_modifier`. Sign handling: work on `abs(MVDCHR_sixteenth)`,
///    then re-apply the sign at the end. Spec text says the modifier
///    pulls toward the nearest half-sample, which is symmetric about
///    zero, so the sign-on-abs convention matches Table 7-10.
///
/// Worked examples (sum → chroma_mv_half, K=4):
///   sum=0 → 0,  sum=4 → 1,  sum=8 → 1,  sum=12 → 1,  sum=14 → 2,
///   sum=16 → 2,  sum=20 → 3,  sum=24 → 3,  sum=32 → 4.
///   Negatives: sum=-4 → -1,  sum=-14 → -2,  sum=-16 → -2.
pub fn luma_4mv_sum_to_chroma(sum_luma_half: i32) -> i32 {
    // Table 7-10: sixteenth pixel position → resulting position (in 1/2
    // chroma sample units). Index by (abs(sum) % 16).
    const TABLE_7_10: [i32; 16] = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2];
    let abs_sum = sum_luma_half.unsigned_abs() as i32;
    let int_part = abs_sum / 16;
    let frac_idx = (abs_sum % 16) as usize;
    let modifier = TABLE_7_10[frac_idx];
    let abs_chroma_half = int_part * 2 + modifier;
    if sum_luma_half < 0 {
        -abs_chroma_half
    } else {
        abs_chroma_half
    }
}

#[cfg(test)]
mod chroma_mv_tests {
    use super::{luma_4mv_sum_to_chroma, luma_mv_to_chroma};

    #[test]
    fn luma_4mv_sum_to_chroma_table_7_10() {
        // Verify the worked examples from the docstring + symmetry.
        assert_eq!(luma_4mv_sum_to_chroma(0), 0);
        assert_eq!(luma_4mv_sum_to_chroma(4), 1);
        assert_eq!(luma_4mv_sum_to_chroma(8), 1);
        assert_eq!(luma_4mv_sum_to_chroma(12), 1);
        assert_eq!(luma_4mv_sum_to_chroma(14), 2);
        assert_eq!(luma_4mv_sum_to_chroma(15), 2);
        assert_eq!(luma_4mv_sum_to_chroma(16), 2);
        assert_eq!(luma_4mv_sum_to_chroma(20), 3);
        assert_eq!(luma_4mv_sum_to_chroma(32), 4);
        // Negative symmetry.
        assert_eq!(luma_4mv_sum_to_chroma(-4), -1);
        assert_eq!(luma_4mv_sum_to_chroma(-14), -2);
        assert_eq!(luma_4mv_sum_to_chroma(-16), -2);
    }

    #[test]
    fn luma_4mv_4x_uniform_matches_1mv_chroma() {
        // When all 4 luma MVs are equal, chroma should match the 1MV
        // path applied to that single value. Spec §7.6.5 / Table 7-10
        // is the K=4 generalisation of Table 7-13 (K=1) — uniform 4MV
        // should reduce to 1MV. Verify on a few values.
        for mv in [-4, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12, 16, 20].iter() {
            let sum = mv * 4;
            let chroma_4mv = luma_4mv_sum_to_chroma(sum);
            let chroma_1mv = luma_mv_to_chroma(*mv);
            assert_eq!(
                chroma_4mv, chroma_1mv,
                "mv={mv} sum={sum}: 4MV={chroma_4mv} != 1MV={chroma_1mv}"
            );
        }
    }
}

/// Quarter-pel 8-tap filter coefficients (§7.6.2.2).
///
/// The MPEG-4 QPel design first builds a half-sample grid using an 8-tap
/// symmetric filter, then derives the quarter-sample positions by bilinear
/// averaging between the integer / half / integer pair that straddles the
/// target.
///
/// Coefficients: `{ -1, 3, -6, 20, 20, -6, 3, -1 } / 32`, rounded.
/// Symmetric around the midpoint between samples 3 and 4. Uncopyrightable
/// spec table — matches ISO 14496-2 §7.6.2.2.
const QPEL_TAPS: [i32; 8] = [-1, 3, -6, 20, 20, -6, 3, -1];

/// Apply the 8-tap QPel filter to 8 samples centred between indices 3 and 4.
/// The result is rounded, clipped to `[0, 255]`, and returned as `u8`.
#[inline]
fn qpel_filter8(s: [i32; 8], rounding: bool) -> u8 {
    let mut acc = 0i32;
    for i in 0..8 {
        acc += QPEL_TAPS[i] * s[i];
    }
    // §7.6.2.2 rounding — `round=16` (/32 nearest) when rounding flag is
    // clear; `round=15` (floor) when set. Divisor is 32.
    let round = if rounding { 15 } else { 16 };
    let v = (acc + round) >> 5;
    v.clamp(0, 255) as u8
}

/// Read a sample with edge-replication clamp.
#[inline]
fn clamp_sample(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    x: i32,
    y: i32,
) -> i32 {
    let xc = x.clamp(0, ref_w - 1) as usize;
    let yc = y.clamp(0, ref_h - 1) as usize;
    ref_plane[yc * ref_stride + xc] as i32
}

/// Predict an `n × n` block using quarter-pel motion (§7.6.2.2).
///
/// `mv_x_q` and `mv_y_q` are in quarter-pel units. The luma MV for QPel
/// blocks is decoded directly on the quarter-pel grid; this helper runs
/// the 8-tap filter along the sub-pel axis (or axes) then bilinearly
/// mixes to reach the true quarter-sample position when needed.
#[allow(clippy::too_many_arguments)]
pub fn predict_block_qpel(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    blk_px: i32,
    blk_py: i32,
    mv_x_q: i32,
    mv_y_q: i32,
    n: i32,
    rounding: bool,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let int_x = mv_x_q >> 2;
    let int_y = mv_y_q >> 2;
    let sx = mv_x_q & 3;
    let sy = mv_y_q & 3;

    let src_x = blk_px + int_x;
    let src_y = blk_py + int_y;

    // Pure integer sample (0/4, 0/4) — straight copy.
    if sx == 0 && sy == 0 {
        for j in 0..n as usize {
            for i in 0..n as usize {
                let v = clamp_sample(
                    ref_plane,
                    ref_stride,
                    ref_w,
                    ref_h,
                    src_x + i as i32,
                    src_y + j as i32,
                ) as u8;
                dst[j * dst_stride + i] = v;
            }
        }
        return;
    }

    // Strategy (§7.6.2.2):
    //
    //   (a) For a pure half-sample position in a single axis (sx=2, sy=0 or
    //       sx=0, sy=2) apply the 8-tap filter along that axis.
    //   (b) For the double-half position (sx=2, sy=2) filter horizontally
    //       then vertically (or vice versa — spec yields identical results
    //       after rounding).
    //   (c) Quarter-sample positions (sx or sy in {1,3}) bilinearly average
    //       the adjacent integer and half samples.
    //
    // We implement this compactly by first building a 16-bit "half-grid"
    // plane covering the block footprint plus the filter taps, then
    // bilinearly mixing for the final quarter offset.

    // Local fn: build one row's worth of horizontally-filtered half samples.
    let hh = |x: i32, y: i32| -> u8 {
        let samples: [i32; 8] = core::array::from_fn(|i| {
            clamp_sample(ref_plane, ref_stride, ref_w, ref_h, x + i as i32 - 3, y)
        });
        qpel_filter8(samples, rounding)
    };
    // Local fn: vertically-filtered half sample (integer x).
    let hv = |x: i32, y: i32| -> u8 {
        let samples: [i32; 8] = core::array::from_fn(|j| {
            clamp_sample(ref_plane, ref_stride, ref_w, ref_h, x, y + j as i32 - 3)
        });
        qpel_filter8(samples, rounding)
    };
    // Local fn: diagonal half-sample (filter horizontally into a row
    // buffer, then vertically through that).
    let hd = |x: i32, y: i32| -> u8 {
        // Vertical filter over 8 rows of horizontally-filtered halves.
        let rows: [i32; 8] = core::array::from_fn(|j| hh(x, y + j as i32 - 3) as i32);
        qpel_filter8(rows, rounding)
    };

    // Integer sample from the reference.
    let int_s =
        |x: i32, y: i32| -> u8 { clamp_sample(ref_plane, ref_stride, ref_w, ref_h, x, y) as u8 };

    // Pick the two "anchor" samplers and the mixing weights per sub-pel.
    // We compute sample at position (px, py) where px, py are measured in
    // quarter-pels relative to the block's integer origin. Positions are
    // derived from `(src_x, src_y, sx, sy)`.
    //
    // Quarter-pel interpolation rules (§7.6.2.2):
    //   (sx=1): mix integer-x and half-x, bilinear average.
    //   (sx=3): mix half-x and integer-(x+1), bilinear average.
    //   Similarly for sy.
    //   (sx=1, sy=1): average int(x,y) with hd(x,y).
    //   …four combinations in total when both sx and sy are odd.
    //
    // We encode the rules as a table of two source selectors and
    // coordinate offsets.

    enum Src {
        Int,
        Hh,
        Hv,
        Hd,
    }

    // Map (sx, sy) to up to two (source, dx, dy) contributors.
    // dx/dy are integer offsets (0 or 1) relative to (src_x, src_y).
    // A single contributor → straight half-sample filter.
    // Two contributors → bilinear mix (rounded to nearest per spec).
    let contribs: &[(Src, i32, i32)] = match (sx, sy) {
        (2, 0) => &[(Src::Hh, 0, 0)],
        (0, 2) => &[(Src::Hv, 0, 0)],
        (2, 2) => &[(Src::Hd, 0, 0)],
        (1, 0) => &[(Src::Int, 0, 0), (Src::Hh, 0, 0)],
        (3, 0) => &[(Src::Hh, 0, 0), (Src::Int, 1, 0)],
        (0, 1) => &[(Src::Int, 0, 0), (Src::Hv, 0, 0)],
        (0, 3) => &[(Src::Hv, 0, 0), (Src::Int, 0, 1)],
        (1, 1) => &[(Src::Int, 0, 0), (Src::Hd, 0, 0)],
        (3, 1) => &[(Src::Int, 1, 0), (Src::Hd, 0, 0)],
        (1, 3) => &[(Src::Int, 0, 1), (Src::Hd, 0, 0)],
        (3, 3) => &[(Src::Int, 1, 1), (Src::Hd, 0, 0)],
        (2, 1) => &[(Src::Hh, 0, 0), (Src::Hd, 0, 0)],
        (2, 3) => &[(Src::Hd, 0, 0), (Src::Hh, 0, 1)],
        (1, 2) => &[(Src::Hv, 0, 0), (Src::Hd, 0, 0)],
        (3, 2) => &[(Src::Hd, 0, 0), (Src::Hv, 1, 0)],
        _ => unreachable!("quarter-pel sub-position out of 0..=3"),
    };

    let round = if rounding { 0 } else { 1 };
    for j in 0..n {
        for i in 0..n {
            let x = src_x + i;
            let y = src_y + j;
            let sample = |c: &(Src, i32, i32)| -> u32 {
                let xx = x + c.1;
                let yy = y + c.2;
                (match c.0 {
                    Src::Int => int_s(xx, yy),
                    Src::Hh => hh(xx, yy),
                    Src::Hv => hv(xx, yy),
                    Src::Hd => hd(xx, yy),
                }) as u32
            };
            let v = if contribs.len() == 1 {
                sample(&contribs[0])
            } else {
                let a = sample(&contribs[0]);
                let b = sample(&contribs[1]);
                (a + b + round) >> 1
            };
            dst[(j as usize) * dst_stride + (i as usize)] = v as u8;
        }
    }
}

/// Chroma MV derivation when the luma MV is expressed in quarter-pel
/// units (§7.6.2.2). The chroma MV lives on the half-pel grid.
///
///   chroma_int_offset = luma_qmv >> 3                  (signed floor)
///   chroma_half_bit   = `table_q[luma_qmv & 7]`
///   chroma_mv_half    = chroma_int_offset * 2 + chroma_half_bit
///
/// The lookup table per §7.6.2.2 eq. (107):
///
///   rem (0..=7): 0, 1, 1, 1, 1, 1, 1, 2
///
/// Negative `luma_qmv` inherits the floor / remainder behaviour of the
/// signed arithmetic shift + AND(7) on a two's-complement integer.
pub fn luma_qmv_to_chroma(luma_qmv: i32) -> i32 {
    // Table from ISO 14496-2 §7.6.2.2 eq. (107).
    const Q_TABLE: [i32; 8] = [0, 1, 1, 1, 1, 1, 1, 2];
    let int_part = luma_qmv >> 3;
    let rem = (luma_qmv & 7) as usize;
    int_part * 2 + Q_TABLE[rem]
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
    fn qpel_integer_copy_equals_bilinear_integer() {
        // Integer QPel MV matches the integer half-pel MV path byte-for-byte.
        let refp: [u8; 64] = core::array::from_fn(|i| (i * 3) as u8);
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        // MV (4, 0) q = 1 int pel to the right
        predict_block_qpel(&refp, 8, 8, 8, 0, 0, 4, 0, 4, false, &mut a, 4);
        predict_block(&refp, 8, 8, 8, 0, 0, 2, 0, 4, false, &mut b, 4);
        assert_eq!(a, b, "QPel int-pel path should match half-pel int-pel path");
    }

    #[test]
    fn qpel_half_h_symmetric_gradient() {
        // 8-tap filter is symmetric around its centre — for a linear ramp,
        // the half-sample output equals the bilinear average of the adjacent
        // integer samples (within rounding).
        let refp: [u8; 64] = core::array::from_fn(|i| ((i % 8) * 30) as u8);
        let mut q = [0u8; 4];
        // sx=2 — pure horizontal half at offset (2/4, 0/4).
        predict_block_qpel(&refp, 8, 8, 8, 3, 3, 2, 0, 2, false, &mut q, 2);
        // Ramp is `x * 30` on each row. Half-sample at pixel 3.5 should be
        // approximately 3.5*30 = 105. The 8-tap filter rounds to nearest.
        assert!(
            (q[0] as i32 - 105).abs() <= 1,
            "half-pel ramp gave {}",
            q[0]
        );
    }

    #[test]
    fn qpel_rounding_flag_reduces_offset() {
        // With rounding=true (vop_rounding_type=1) the filter rounds to
        // floor instead of nearest → output is ≤ the rounding=false output.
        let refp: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(5));
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        predict_block_qpel(&refp, 8, 8, 8, 3, 3, 1, 0, 2, false, &mut a, 2);
        predict_block_qpel(&refp, 8, 8, 8, 3, 3, 1, 0, 2, true, &mut b, 2);
        // Each b[i] <= a[i] since rounding-on uses floor.
        for (av, bv) in a.iter().zip(b.iter()) {
            assert!(
                bv <= av,
                "rounding-on output {bv} should be ≤ rounding-off output {av}"
            );
        }
    }

    #[test]
    fn luma_qmv_to_chroma_table() {
        // Positive remainders per spec table.
        assert_eq!(luma_qmv_to_chroma(0), 0);
        assert_eq!(luma_qmv_to_chroma(1), 1);
        assert_eq!(luma_qmv_to_chroma(2), 1);
        assert_eq!(luma_qmv_to_chroma(3), 1);
        assert_eq!(luma_qmv_to_chroma(4), 1);
        assert_eq!(luma_qmv_to_chroma(5), 1);
        assert_eq!(luma_qmv_to_chroma(6), 1);
        assert_eq!(luma_qmv_to_chroma(7), 2);
        assert_eq!(luma_qmv_to_chroma(8), 2);
        // Symmetry around zero — the arithmetic shift floors negatives:
        //   -1 >> 3 = -1,  -1 & 7 = 7 → -1*2 + 2 = 0.
        assert_eq!(luma_qmv_to_chroma(-1), 0);
        assert_eq!(luma_qmv_to_chroma(-8), -2);
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

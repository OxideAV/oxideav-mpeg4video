//! §7.6.5 chrominance motion-vector derivation from `K` luminance
//! block vectors (4:2:0 rectangular VOP).
//!
//! Given `K ∈ {1, 2, 3, 4}` luminance motion vectors that correspond
//! to the `K` non-transparent 8×8 luminance sub-blocks of a 16×16
//! macroblock, this module derives the chrominance MV used for the
//! single 8×8 Cb and 8×8 Cr prediction blocks.
//!
//! ## Algorithm (§7.6.5 paragraph above the tables + §7.6.9.5.3 last
//! paragraph for the B-VOP direct-mode mirror form)
//!
//! 1. Sum the `K` luminance vectors component-wise.
//! 2. Divide each component sum by `2 * K` (using §3.4 integer
//!    division — truncation toward zero / floor as discussed below).
//! 3. Modify the resulting fractional component toward the nearest
//!    half-sample position using one of:
//!    * [`TABLE_7_13`] for `K = 1` ("fourth sample resolution" — same
//!      4-entry mapping that already powers
//!      [`crate::quarter_sample::reduce_qpel_to_half_pel_chroma`]);
//!    * [`TABLE_7_12`] for `K = 2` ("eighth sample resolution");
//!    * [`TABLE_7_11`] for `K = 3` ("twelfth sample resolution");
//!    * [`TABLE_7_10`] for `K = 4` ("sixteenth sample resolution").
//!
//! Each table maps an integer in `0..= (entries - 1)` (the fine-grid
//! residue per integer chroma position) to a value in `{0, 1, 2}` (the
//! count of half-sample offsets to add). An output of `2` means the
//! rounding carries into the next integer chroma position (an output
//! of `2 * 0.5 = 1` full chroma-pel).
//!
//! ## Sign convention
//!
//! The §3.4 spec division `/` truncates toward zero, but the existing
//! [`crate::quarter_sample::reduce_qpel_to_half_pel_chroma`] uses
//! floor (toward `-∞`) so the table-index residue is always
//! non-negative on negative inputs. This module follows the same
//! floor convention (via [`i32::div_euclid`] and [`i32::rem_euclid`])
//! so the K = 1 component-wise result is anti-symmetric around zero
//! (modulo the asymmetry Table 7-13 introduces by rounding every
//! non-zero quarter-residue to `+1` half-sample).
//!
//! ## Pre-reduction in quarter-sample mode
//!
//! §7.6.5 also says: *"in quarter sample mode the vectors are divided
//! by 2 before summation."* The MPEG-4 Part 2 spec text does not
//! specify the rounding for this pre-divide step; this module
//! therefore exposes only the **half-sample-mode** entry points and
//! the verbatim table transcriptions. Callers that need the
//! quarter-sample pre-divide should componentwise apply
//! [`crate::quarter_sample::reduce_qpel_to_half_pel_chroma`] (the
//! existing §7.6.9.2 Table 7-13 rounding) to each input MV first,
//! then pass the resulting half-sample-grid vectors into
//! [`chroma_mv_from_luma_blocks`]. A future round can introduce a
//! convenience wrapper once the docs collaborator confirms the
//! pre-divide rounding rule.
//!
//! ## What this module does **not** do
//!
//! * Gather the `K` luminance vectors from the §7.6.5 four-MV layout
//!   (i.e. resolve transparency / non-rectangular VOP shape). The
//!   caller decides which sub-blocks contribute. This module accepts
//!   the already-filtered set.
//! * Implement the §7.6.1.6 vector padding. The caller pads first.
//! * Apply the Table 7-9 `[low:high]` modulo wrap. The chroma MV
//!   inherits its valid range from the luminance MVs, all of which
//!   have already passed the Table 7-9 wrap in
//!   [`crate::motion::reconstruct_motion_vector`].
//! * Run the chrominance motion-compensation interpolation. That is
//!   the responsibility of the §7.6.2.1 / §7.6.2.2 routines once a
//!   chroma reference plane wrapper exists.

use crate::motion::MotionVector;

/// Table 7-13 — Modification of fourth sample resolution chrominance
/// vector components.
///
/// Input: fractional position on a `1/4`-sample grid in `0..=3`.
/// Output: number of half-sample offsets in `{0, 1}` to add to the
/// integer chroma position.
///
/// Verbatim transcription from ISO/IEC 14496-2:2004 §7.6.5 Table
/// 7-13.
pub const TABLE_7_13: [u8; 4] = [0, 1, 1, 1];

/// Table 7-12 — Modification of eighth sample resolution chrominance
/// vector components.
///
/// Input: fractional position on a `1/8`-sample grid in `0..=7`.
/// Output: number of half-sample offsets in `{0, 1, 2}` to add to the
/// integer chroma position. An output of `2` carries one full
/// chroma-pel into the integer part (handled by the caller).
///
/// Verbatim transcription from ISO/IEC 14496-2:2004 §7.6.5 Table
/// 7-12.
pub const TABLE_7_12: [u8; 8] = [0, 0, 1, 1, 1, 1, 1, 2];

/// Table 7-11 — Modification of twelfth sample resolution chrominance
/// vector components.
///
/// Input: fractional position on a `1/12`-sample grid in `0..=11`.
/// Output: number of half-sample offsets in `{0, 1, 2}` to add to the
/// integer chroma position.
///
/// Verbatim transcription from ISO/IEC 14496-2:2004 §7.6.5 Table
/// 7-11.
pub const TABLE_7_11: [u8; 12] = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2];

/// Table 7-10 — Modification of sixteenth sample resolution
/// chrominance vector components.
///
/// Input: fractional position on a `1/16`-sample grid in `0..=15`.
/// Output: number of half-sample offsets in `{0, 1, 2}` to add to the
/// integer chroma position.
///
/// Verbatim transcription from ISO/IEC 14496-2:2004 §7.6.5 Table
/// 7-10.
pub const TABLE_7_10: [u8; 16] = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2];

/// Errors raised by [`chroma_mv_from_luma_blocks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaMvError {
    /// `K` (the number of contributing luminance sub-blocks) was zero
    /// or greater than 4 — §7.6.5 requires `1 <= K <= 4` because the
    /// 4:2:0 macroblock contains four 8×8 luminance sub-blocks.
    InvalidBlockCount(usize),
}

impl core::fmt::Display for ChromaMvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ChromaMvError::InvalidBlockCount(k) => write!(
                f,
                "chroma-MV derivation requires K in 1..=4 luminance blocks (§7.6.5); got K = {k}"
            ),
        }
    }
}

impl std::error::Error for ChromaMvError {}

/// Per-component §7.6.5 reduction.
///
/// `sum` is the integer component-sum of the `K` luminance MV
/// components in **half-sample units** (i.e. the bitstream half-pel
/// representation). `k` is `K ∈ {1, 2, 3, 4}`.
///
/// Returns the chroma MV component in half-sample units — the §7.6.5
/// `sum / (2 * K)` integer half-sample part plus the Table-7-{13,12,
/// 11,10} half-sample fractional offset for the residue.
///
/// The integer division and modulo use the [`i32::div_euclid`] /
/// [`i32::rem_euclid`] flavour (floor-toward-`-∞`) so the residue is
/// always non-negative and the function is anti-symmetric around zero
/// (modulo the Table 7-13/12/11/10 rounding).
#[inline]
fn reduce_component_half_pel(sum: i32, k: usize) -> i32 {
    // §7.6.5: the chroma displacement is `sum / (2K)` in half-sample
    // units, i.e. `sum / (4K)` in full chroma pels. Decompose against
    // the *whole-pel* boundary: the quotient `sum div 4K` is the
    // integer chroma-pel part (worth two half-sample units each), and
    // the residue `sum mod 4K` is exactly the Table 7-10/7-11/7-12/
    // 7-13 "sixteenth / twelfth / eighth / fourth sample" position —
    // each table has `4K` entries mapping that position to its
    // nearest-half-sample offset in `{0, 1, 2}` (an output of `2`
    // carries into the next whole pel).
    //
    // Floor division (`div_euclid` / `rem_euclid`) keeps the residue
    // non-negative for negative sums, matching the tables' one-sided
    // domain.
    let four_k = 4 * (k as i32);
    let whole_pels = sum.div_euclid(four_k);
    let table_index = sum.rem_euclid(four_k) as usize;
    2 * whole_pels + i32::from(lookup_table(k, table_index))
}

#[inline]
fn lookup_table(k: usize, index: usize) -> u8 {
    match k {
        1 => TABLE_7_13[index],
        2 => TABLE_7_12[index],
        3 => TABLE_7_11[index],
        4 => TABLE_7_10[index],
        _ => 0, // Unreachable — `chroma_mv_from_luma_blocks` validates K first.
    }
}

/// Derive the §7.6.5 chrominance motion vector `MVDCHR` from the
/// `K` contributing luminance sub-block motion vectors of one
/// macroblock (`K ∈ {1, 2, 3, 4}`).
///
/// `luma_mvs` are the K luminance MVs **in half-sample units** (the
/// natural bitstream representation in half-pel mode). Each MV
/// corresponds to one 8×8 luminance sub-block that is non-transparent
/// (the spec excludes transparent / out-of-shape sub-blocks from the
/// sum). The order of the slice does not affect the result — only the
/// componentwise sum and count matter.
///
/// The returned chroma MV is also in half-sample units, ready to feed
/// the §7.6.2.1 bilinear chrominance interpolation primitives once
/// they wire up.
///
/// Quarter-sample mode callers must pre-reduce each luminance MV via
/// [`crate::quarter_sample::reduce_qpel_to_half_pel_chroma`] before
/// invoking this function — the §7.6.5 "divided by 2 before
/// summation" step. See the module-level docs for the rationale.
///
/// # Examples
///
/// ```
/// use oxideav_mpeg4video::chroma_mv::chroma_mv_from_luma_blocks;
/// use oxideav_mpeg4video::motion::MotionVector;
///
/// // K = 1: a single MV `(4, -2)` half-sample units (= (2.0, -1.0)
/// // luma-pel) yields chroma MV `(2, -1)` half-sample units (= (1.0,
/// // -0.5) luma-pel) per the §7.6.5 `sum / 2K` rule.
/// let chroma = chroma_mv_from_luma_blocks(&[MotionVector { x: 4, y: -2 }]).unwrap();
/// assert_eq!(chroma, MotionVector { x: 2, y: -1 });
///
/// // K = 4: four identical MVs `(4, 4)` half-sample sum to `(16,
/// // 16)`, divided by 2*4 = 8 yields the chroma MV `(2, 2)`
/// // half-sample (= (1.0, 1.0) luma-pel — half of the average luma
/// // displacement after subsampling).
/// let chroma = chroma_mv_from_luma_blocks(&[
///     MotionVector { x: 4, y: 4 },
///     MotionVector { x: 4, y: 4 },
///     MotionVector { x: 4, y: 4 },
///     MotionVector { x: 4, y: 4 },
/// ]).unwrap();
/// assert_eq!(chroma, MotionVector { x: 2, y: 2 });
/// ```
pub fn chroma_mv_from_luma_blocks(
    luma_mvs: &[MotionVector],
) -> Result<MotionVector, ChromaMvError> {
    let k = luma_mvs.len();
    if !(1..=4).contains(&k) {
        return Err(ChromaMvError::InvalidBlockCount(k));
    }
    let sum_x: i32 = luma_mvs.iter().map(|m| m.x).sum();
    let sum_y: i32 = luma_mvs.iter().map(|m| m.y).sum();
    Ok(MotionVector {
        x: reduce_component_half_pel(sum_x, k),
        y: reduce_component_half_pel(sum_y, k),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- table-shape sanity ----------

    #[test]
    fn table_7_13_matches_existing_quarter_sample_reduction() {
        // For K=1, Table 7-13 lookup with table_index = residue * 2
        // and residue ∈ {0, 1} produces the same half-pel rounding bit
        // as the existing `reduce_qpel_to_half_pel_chroma` for the
        // residue-equivalent quarter-pel input.
        // residue 0 → table index 0 → output 0 → half_pel bit 0
        // residue 1 → table index 2 → output 1 → half_pel bit 1
        assert_eq!(TABLE_7_13[0], 0);
        assert_eq!(TABLE_7_13[2], 1);
    }

    #[test]
    fn table_7_10_carries_to_next_integer_at_the_top() {
        // The §7.6.5 spec text places the "carry" outputs (= 2) at
        // the high end of each table: positions 14 and 15 in Table
        // 7-10. A carry-out of 2 means `+1 full chroma-pel` (= 2
        // half-sample units) added to the integer part.
        assert_eq!(TABLE_7_10[14], 2);
        assert_eq!(TABLE_7_10[15], 2);
    }

    #[test]
    fn table_7_12_has_correct_length_and_carry() {
        assert_eq!(TABLE_7_12.len(), 8);
        assert_eq!(TABLE_7_12[7], 2);
        // No carry below index 7.
        for &v in TABLE_7_12.iter().take(7) {
            assert!(v < 2);
        }
    }

    #[test]
    fn table_7_11_has_correct_length_and_carry() {
        assert_eq!(TABLE_7_11.len(), 12);
        assert_eq!(TABLE_7_11[10], 2);
        assert_eq!(TABLE_7_11[11], 2);
    }

    #[test]
    fn table_lengths_match_grid_denominators() {
        assert_eq!(TABLE_7_13.len(), 4);
        assert_eq!(TABLE_7_12.len(), 8);
        assert_eq!(TABLE_7_11.len(), 12);
        assert_eq!(TABLE_7_10.len(), 16);
    }

    // ---------- error handling ----------

    #[test]
    fn rejects_zero_blocks() {
        assert_eq!(
            chroma_mv_from_luma_blocks(&[]),
            Err(ChromaMvError::InvalidBlockCount(0))
        );
    }

    #[test]
    fn rejects_five_blocks() {
        let mvs = [MotionVector { x: 0, y: 0 }; 5];
        assert_eq!(
            chroma_mv_from_luma_blocks(&mvs),
            Err(ChromaMvError::InvalidBlockCount(5))
        );
    }

    // ---------- K = 1 (Table 7-13) ----------

    #[test]
    fn k1_zero_mv_is_zero() {
        let chroma = chroma_mv_from_luma_blocks(&[MotionVector { x: 0, y: 0 }]).unwrap();
        assert_eq!(chroma, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn k1_integer_full_pel_halves() {
        // MV = (4, -4) half-sample = (2, -2) luma-pel. chroma_mv =
        // sum/(2K) = (2, -2) half-sample = (1, -1) luma-pel.
        let chroma = chroma_mv_from_luma_blocks(&[MotionVector { x: 4, y: -4 }]).unwrap();
        assert_eq!(chroma, MotionVector { x: 2, y: -2 });
    }

    #[test]
    fn k1_odd_half_pel_rounds_via_table_7_13() {
        // MV = (1, 3) half-sample. The chroma displacement is sum/4
        // chroma-pels; the residue mod 4 is the Table 7-13 fourth-pel
        // position.
        // x: 1 → whole 0, TABLE_7_13[1] = 1 → 1 half-sample.
        // y: 3 → whole 0, TABLE_7_13[3] = 1 → 1 half-sample (3/4 pel
        //   rounds DOWN to the half-sample position, not up to 1 pel).
        let chroma = chroma_mv_from_luma_blocks(&[MotionVector { x: 1, y: 3 }]).unwrap();
        assert_eq!(chroma, MotionVector { x: 1, y: 1 });
    }

    #[test]
    fn k1_div_2_with_floor_for_negatives() {
        // §7.6.5 says `chroma_mv = sum / (2 * K)` rounded to the
        // nearest half-sample per Table 7-13. For K = 1 the sum is
        // the single MV; the whole-chroma-pel part is
        // `sum.div_euclid(4)` (worth 2 half-samples) and
        // `sum.rem_euclid(4)` indexes Table 7-13 = [0, 1, 1, 1], so
        // every fractional position rounds to the half-sample.
        //
        // mv = -1: div_euclid(4) = -1 → -2; rem_euclid = 3 →
        // TABLE_7_13[3] = 1; result = -2 + 1 = -1 half-sample (the
        // -0.25-pel displacement rounds to -0.5, matching the
        // positive side's 0.25 → 0.5).
        let cases = [
            (-4i32, -2i32),
            (-3, -1),
            (-2, -1),
            (-1, -1),
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 2),
        ];
        for (mv, expected) in cases {
            let got = chroma_mv_from_luma_blocks(&[MotionVector { x: mv, y: 0 }]).unwrap();
            assert_eq!(got.x, expected, "mv = {mv}");
        }
    }

    // ---------- K = 2 (Table 7-12) ----------

    #[test]
    fn k2_zero_sum_is_zero() {
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 3, y: -1 },
            MotionVector { x: -3, y: 1 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn k2_uniform_full_pel() {
        // 2 MVs of (4, 4) half-sample sum to (8, 8); / (2*2) = (2,
        // 2) half-sample (= (1, 1) luma-pel).
        let chroma =
            chroma_mv_from_luma_blocks(&[MotionVector { x: 4, y: 4 }, MotionVector { x: 4, y: 4 }])
                .unwrap();
        assert_eq!(chroma, MotionVector { x: 2, y: 2 });
    }

    #[test]
    fn k2_residue_rounds_via_table_7_12() {
        // K = 2 → the residue mod 8 indexes Table 7-12 =
        // [0, 0, 1, 1, 1, 1, 1, 2].
        // Sum (half-sample units) = (5, 3): x → whole 0 +
        // TABLE_7_12[5] = 1; y → whole 0 + TABLE_7_12[3] = 1.
        let chroma =
            chroma_mv_from_luma_blocks(&[MotionVector { x: 3, y: 1 }, MotionVector { x: 2, y: 2 }])
                .unwrap();
        assert_eq!(chroma, MotionVector { x: 1, y: 1 });
        // Sum = (7, 1): x → TABLE_7_12[7] = 2 (carry to the next
        // whole chroma pel); y → TABLE_7_12[1] = 0.
        let chroma =
            chroma_mv_from_luma_blocks(&[MotionVector { x: 4, y: 1 }, MotionVector { x: 3, y: 0 }])
                .unwrap();
        assert_eq!(chroma, MotionVector { x: 2, y: 0 });
    }

    #[test]
    fn k2_residue_3_picks_carry_entry_in_table_7_12() {
        // Sum = (3, 3). div_euclid(4) = (0, 0); residue = (3, 3).
        // table_index = 6 → TABLE_7_12[6] = 1 (NOT 2 — the carry
        // entry sits at index 7 only). Result = (1, 1) half-sample.
        let chroma =
            chroma_mv_from_luma_blocks(&[MotionVector { x: 2, y: 2 }, MotionVector { x: 1, y: 1 }])
                .unwrap();
        assert_eq!(chroma, MotionVector { x: 1, y: 1 });
    }

    // ---------- K = 3 (Table 7-11) ----------

    #[test]
    fn k3_uniform_full_pel() {
        // 3 MVs of (4, 4) half-sample sum to (12, 12); / 6 = (2, 2)
        // half-sample.
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 4, y: 4 },
            MotionVector { x: 4, y: 4 },
            MotionVector { x: 4, y: 4 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: 2, y: 2 });
    }

    #[test]
    fn k3_residue_uses_table_7_11() {
        // K = 3 → the residue mod 12 indexes Table 7-11 =
        // [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2].
        // Sum = (3, 5): x → whole 0 + TABLE_7_11[3] = 1;
        // y → whole 0 + TABLE_7_11[5] = 1.
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 1, y: 2 },
            MotionVector { x: 1, y: 2 },
            MotionVector { x: 1, y: 1 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: 1, y: 1 });
        // Sum = (11, 0): x → TABLE_7_11[11] = 2 (carry).
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 4, y: 0 },
            MotionVector { x: 4, y: 0 },
            MotionVector { x: 3, y: 0 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: 2, y: 0 });
    }

    // ---------- K = 4 (Table 7-10) ----------

    #[test]
    fn k4_zero_sum_is_zero() {
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 2, y: -3 },
            MotionVector { x: -2, y: 3 },
            MotionVector { x: 4, y: 0 },
            MotionVector { x: -4, y: 0 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn k4_uniform_full_pel() {
        // 4 MVs of (4, 4) half-sample sum to (16, 16); / 8 = (2, 2)
        // half-sample.
        let chroma = chroma_mv_from_luma_blocks(&[MotionVector { x: 4, y: 4 }; 4]).unwrap();
        assert_eq!(chroma, MotionVector { x: 2, y: 2 });
    }

    #[test]
    fn k4_residue_uses_table_7_10() {
        // K = 4 → the residue mod 16 indexes Table 7-10 =
        // [0, 0, 0, 1×11, 2, 2].
        // Sum = (7, 1): x → whole 0 + TABLE_7_10[7] = 1;
        // y → whole 0 + TABLE_7_10[1] = 0.
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 2, y: 1 },
            MotionVector { x: 2, y: 0 },
            MotionVector { x: 2, y: 0 },
            MotionVector { x: 1, y: 0 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: 1, y: 0 });
        // Sum = (15, 0): x → TABLE_7_10[15] = 2 (carry to one whole
        // chroma pel).
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 4, y: 0 },
            MotionVector { x: 4, y: 0 },
            MotionVector { x: 4, y: 0 },
            MotionVector { x: 3, y: 0 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: 2, y: 0 });
    }

    #[test]
    fn k4_residue_3_uses_table_7_10_entry_6() {
        // Sum = (3, 3). div_euclid(8) = (0, 0); residue = (3, 3).
        // table_index = 6 → TABLE_7_10[6] = 1.
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 1, y: 1 },
            MotionVector { x: 1, y: 1 },
            MotionVector { x: 1, y: 1 },
            MotionVector { x: 0, y: 0 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: 1, y: 1 });
    }

    // ---------- negative-input symmetry ----------

    #[test]
    fn negative_inputs_floor_division_is_symmetric_for_k4() {
        // Sum = (-7, -1). div_euclid(16) = (-1, -1) → -2 half-samples;
        // rem_euclid(16) = (9, 15).
        // x: TABLE_7_10[9] = 1 → -2 + 1 = -1 half-sample (the
        //   -7/16-pel displacement rounds to -0.5, mirroring +7/16 →
        //   +0.5).
        // y: TABLE_7_10[15] = 2 → -2 + 2 = 0 (the -1/16-pel
        //   displacement rounds to zero, mirroring +1/16 → 0).
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: -2, y: -1 },
            MotionVector { x: -2, y: 0 },
            MotionVector { x: -2, y: 0 },
            MotionVector { x: -1, y: 0 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: -1, y: 0 });
    }

    #[test]
    fn negative_full_pel_k1_matches_positive_negated() {
        // Sanity: chroma_mv(-MV) should equal -chroma_mv(MV) for
        // multiples of 2 (where Table 7-13's rounding doesn't break
        // the symmetry).
        for mv_x in [-8, -4, 0, 4, 8] {
            for mv_y in [-8, -4, 0, 4, 8] {
                let pos = chroma_mv_from_luma_blocks(&[MotionVector { x: mv_x, y: mv_y }]).unwrap();
                let neg =
                    chroma_mv_from_luma_blocks(&[MotionVector { x: -mv_x, y: -mv_y }]).unwrap();
                assert_eq!(
                    pos,
                    MotionVector {
                        x: -neg.x,
                        y: -neg.y,
                    },
                    "mv = ({mv_x}, {mv_y})"
                );
            }
        }
    }

    // ---------- error display ----------

    #[test]
    fn error_display_includes_k() {
        let err = ChromaMvError::InvalidBlockCount(0);
        let s = format!("{err}");
        assert!(s.contains("K = 0"), "got: {s}");
        let err = ChromaMvError::InvalidBlockCount(5);
        let s = format!("{err}");
        assert!(s.contains("K = 5"), "got: {s}");
    }
}

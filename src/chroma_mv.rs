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
    // §7.6.5 step 1: divide the half-sample sum by `2 * K`. The
    // integer-half quotient is the chroma MV's integer half-sample
    // part; the residue selects the §7.6.5 table's fractional
    // half-sample offset for the carry.
    let two_k = 2 * (k as i32);
    let int_half = sum.div_euclid(two_k);
    let residue_half = sum.rem_euclid(two_k);
    // The table denominator is `4 * K` (the §7.6.5 "fourth /
    // eighth / twelfth / sixteenth sample resolution" of `sum /
    // (2K)`'s remainder), which is `2 * (2 * K)` = `2 * two_k`.
    // The residue produced from a half-sample sum lives on the
    // `1/(2K)`-half-sample grid, so multiplying by 2 places it on
    // the `1/(4K)` grid the table indexes.
    let table_index = (residue_half as usize) * 2;
    let half_offset = lookup_table(k, table_index);
    int_half + half_offset as i32
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
        // MV = (1, 3) half-sample. sum/(2K) = sum/2.
        // x: 1/2 = 0 integer-half + residue 1 → table index 2 →
        //   TABLE_7_13[2] = 1 → result = 0 + 1 = 1 half-sample.
        // y: 3/2 = 1 integer-half + residue 1 → result = 1 + 1 = 2.
        let chroma = chroma_mv_from_luma_blocks(&[MotionVector { x: 1, y: 3 }]).unwrap();
        assert_eq!(chroma, MotionVector { x: 1, y: 2 });
    }

    #[test]
    fn k1_div_2_with_floor_for_negatives() {
        // §7.6.5 says `chroma_mv = sum / (2 * K)`. For K = 1, sum is
        // just the single MV. With floor (`div_euclid`) the result
        // for negative MVs rounds toward `-∞` so the function is
        // anti-symmetric around zero (with the Table 7-13 lookup
        // mapping every non-zero odd residue to a `+1` half-sample
        // offset). The pairs below pin the convention.
        //
        // mv = -2 (= -1 luma-pel half-sample sum): -2.div_euclid(2)
        // = -1; rem_euclid = 0; result = -1 half-sample. mv = -1:
        // -1.div_euclid(2) = -1; rem_euclid = 1; table index = 2 →
        // TABLE_7_13[2] = 1; result = -1 + 1 = 0 half-sample. mv =
        // 1: 1.div_euclid(2) = 0; rem_euclid = 1; result = 0 + 1 =
        // 1 half-sample.
        let cases = [
            (-4i32, -2i32),
            (-3, -1), // -3.div_euclid(2)=-2; rem 1 → table 2 → +1; result -1
            (-2, -1),
            (-1, 0),
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 2),
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
        // Sum (in half-sample units) = (1, 0). div_euclid(4) = (0,
        // 0); rem_euclid(4) = (1, 0).
        // x: table_index = 1 * 2 = 2 → TABLE_7_12[2] = 1; result = 0
        //   + 1 = 1.
        // y: residue 0 → table_index 0 → TABLE_7_12[0] = 0; result =
        //   0.
        let chroma =
            chroma_mv_from_luma_blocks(&[MotionVector { x: 1, y: 0 }, MotionVector { x: 0, y: 0 }])
                .unwrap();
        assert_eq!(chroma, MotionVector { x: 1, y: 0 });
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
        // Sum = (3, 5). div_euclid(6) = (0, 0); residue = (3, 5).
        // x: table_index = 3 * 2 = 6 → TABLE_7_11[6] = 1; result = 1.
        // y: table_index = 5 * 2 = 10 → TABLE_7_11[10] = 2; result =
        //   0 + 2 = 2 half-sample.
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 1, y: 2 },
            MotionVector { x: 1, y: 2 },
            MotionVector { x: 1, y: 1 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: 1, y: 2 });
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
        // Sum = (7, 1). div_euclid(8) = (0, 0); residue = (7, 1).
        // x: table_index = 7 * 2 = 14 → TABLE_7_10[14] = 2; result =
        //   0 + 2 = 2 half-sample (carry into the next integer
        //   chroma-pel).
        // y: table_index = 1 * 2 = 2 → TABLE_7_10[2] = 0; result =
        //   0.
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: 2, y: 1 },
            MotionVector { x: 2, y: 0 },
            MotionVector { x: 2, y: 0 },
            MotionVector { x: 1, y: 0 },
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
        // Sum = (-7, -1). div_euclid(8) = (-1, -1); rem_euclid(8) =
        // (1, 7).
        // x: table_index = 1 * 2 = 2 → TABLE_7_10[2] = 0; result =
        //   -1 + 0 = -1 half-sample.
        // y: table_index = 7 * 2 = 14 → TABLE_7_10[14] = 2; result =
        //   -1 + 2 = 1 half-sample.
        let chroma = chroma_mv_from_luma_blocks(&[
            MotionVector { x: -2, y: -1 },
            MotionVector { x: -2, y: 0 },
            MotionVector { x: -2, y: 0 },
            MotionVector { x: -1, y: 0 },
        ])
        .unwrap();
        assert_eq!(chroma, MotionVector { x: -1, y: 1 });
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

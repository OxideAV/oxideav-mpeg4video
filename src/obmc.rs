//! §7.6.6 Overlapped Motion Compensation (OBMC) for progressive
//! P- and S(GMC)-VOPs.
//!
//! Each pixel `p(i, j)` in an 8×8 luminance prediction block is a
//! weighted sum of three reference-frame fetches, divided by 8 with
//! rounding (§7.6.6 equation):
//!
//! ```text
//! p(i, j) = (q(i, j) * H0(i, j) + r(i, j) * H1(i, j) + s(i, j) * H2(i, j) + 4) // 8
//! ```
//!
//! where
//!
//! * `q(i, j) = p(i + MVx0, j + MVy0)` — the current block's MV (MV0);
//! * `r(i, j) = p(i + MVx1, j + MVy1)` — the "top-or-bottom" remote MV
//!   (MV1, selected per pixel by quadrant);
//! * `s(i, j) = p(i + MVx2, j + MVy2)` — the "left-or-right" remote MV
//!   (MV2, selected per pixel by quadrant).
//!
//! Per the §7.6.6 paragraph following the equation, MV1 is the motion
//! vector of the block above the current block for pixels in the
//! upper half (`j < 4`) and the motion vector of the block below for
//! pixels in the lower half (`j >= 4`); MV2 is the motion vector of
//! the block to the left for pixels in the left half (`i < 4`) and to
//! the right for pixels in the right half (`i >= 4`).
//!
//! The §7.6.6 prose lists four remote-MV substitution rules; see
//! [`RemoteBlockKind`] and [`ObmcNeighbourhood::resolved_remote_mv`].
//!
//! H0/H1/H2 are the §7.6.6 weighting matrices transcribed from
//! Figures 7-35, 7-36 and 7-37 of ISO/IEC 14496-2:2004 (3rd edition).
//! Each cell sums `H0 + H1 + H2 = 8`, so flat (`q == r == s == C`)
//! reference samples produce `(8 * C + 4) / 8 == C`.
//!
//! ## Out of scope (this round)
//!
//! * The reference-frame sample fetch itself — the bilinear /
//!   half-sample interpolation of §7.6.2 is the caller's job. This
//!   module receives a sample-provider callback that yields one
//!   `u8` luma sample per `(MV, i, j)` triple.
//! * The §7.6.5 spatial-neighbourhood walk that gathers the four
//!   surrounding 8×8 blocks (above / below / left / right) from
//!   already-decoded MB grid. Figure 7-34 / Figure 6-8 mapping is
//!   the caller's responsibility (and remains a docs gap for the
//!   median-filter predictor — see crate docs).
//! * The 8×8 chrominance prediction. §7.6.6 only specifies OBMC for
//!   the luminance plane; the chrominance MVDCHR derivation and the
//!   Table 7-10..7-13 sub-pel modification belong in a later round.
//! * The S(GMC)-VOP mcsel-boundary disablement note — the caller
//!   passes [`ObmcConfig::enabled = false`] when crossing such a
//!   boundary and [`obmc_predict_block`] returns the bare MV0-only
//!   prediction unchanged.

// `OBMC_H0` / `OBMC_H1` / `OBMC_H2` are matched indexing-wise over
// the four-corner / four-edge / interior structure of an 8×8
// luminance block; index-into-multiple-arrays-by-`(j, i)` is clearer
// than `enumerate()` for the per-cell composition and the matrix
// shape-checks in the unit tests. (Same justification as `idct.rs`.)
#![allow(clippy::needless_range_loop)]

/// Weighting matrix `H0` of Figure 7-35 — weights for the current
/// block's motion vector. Indexed as `H0[row][col]` = `H0(j, i)`.
///
/// Note the §7.6.6 equation parameterises the matrices as `H(i, j)`
/// "where `i, j` denote the column and row, respectively" — i.e. the
/// row index is `j` and the column index is `i`. The rust array is
/// laid out row-major in `j` and column-major within each row in
/// `i`, matching the visual layout of Figure 7-35.
pub const OBMC_H0: [[u8; 8]; 8] = [
    [4, 5, 5, 5, 5, 5, 5, 4],
    [5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 6, 6, 6, 6, 5, 5],
    [5, 5, 6, 6, 6, 6, 5, 5],
    [5, 5, 6, 6, 6, 6, 5, 5],
    [5, 5, 6, 6, 6, 6, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5],
    [4, 5, 5, 5, 5, 5, 5, 4],
];

/// Weighting matrix `H1` of Figure 7-36 — weights for the
/// above-or-below remote MV (`MV1`). Same `(j, i)` indexing as
/// [`OBMC_H0`].
pub const OBMC_H1: [[u8; 8]; 8] = [
    [2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 2, 2, 2, 2, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 2, 2, 2, 2, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2],
];

/// Weighting matrix `H2` of Figure 7-37 — weights for the
/// left-or-right remote MV (`MV2`). Same `(j, i)` indexing as
/// [`OBMC_H0`].
pub const OBMC_H2: [[u8; 8]; 8] = [
    [2, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 1, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 1, 2, 2],
    [2, 1, 1, 1, 1, 1, 1, 2],
];

/// Motion-vector pair used as input to the §7.6.6 reference-sample
/// fetch. The components are in the same units as the caller's
/// sample provider expects (typically half-pel or quarter-pel
/// integers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ObmcMv {
    /// Horizontal component of the motion vector.
    pub x: i32,
    /// Vertical component of the motion vector.
    pub y: i32,
}

impl ObmcMv {
    /// Construct a motion-vector pair.
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// The zero motion vector — produced by the §7.6.6
    /// "not-coded surrounding block" substitution rule.
    pub const ZERO: Self = Self { x: 0, y: 0 };
}

/// Status of a remote 8×8 neighbour block as seen from the current
/// block's perspective, governing the §7.6.6 remote-MV substitution
/// rules.
///
/// The four rules in §7.6.6 (paragraph following the equation) are:
///
/// 1. *Not coded*: remote MV → 0.
/// 2. *Intra-coded*: remote MV → current MV.
/// 3. *Absent at VOP/packet border*: remote MV → current MV.
/// 4. *Below-current-MB bottom row*: when the current 8×8 block sits
///    on the bottom row of its macroblock, the below-MB remote MV is
///    replaced by the current MV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteBlockKind {
    /// A regular inter-coded surrounding block; its motion vector is
    /// used verbatim.
    Inter(ObmcMv),
    /// The surrounding block was not coded (skipped / `not_coded`).
    /// Rule 1 → remote MV = 0.
    NotCoded,
    /// The surrounding block was intra-coded. Rule 2 → remote MV =
    /// current MV.
    Intra,
    /// The surrounding block is absent because the current 8×8 block
    /// sits on a VOP / video-packet / GOB border in the relevant
    /// direction. Rule 3 → remote MV = current MV.
    Absent,
}

impl RemoteBlockKind {
    /// Resolve this remote-block kind into the concrete motion vector
    /// that the §7.6.6 reference-sample fetch should use, given the
    /// current block's own motion vector.
    pub fn resolved(self, current_mv: ObmcMv) -> ObmcMv {
        match self {
            Self::Inter(mv) => mv,
            Self::NotCoded => ObmcMv::ZERO,
            Self::Intra => current_mv,
            Self::Absent => current_mv,
        }
    }
}

/// The four remote-block neighbours of an 8×8 luminance block plus
/// the §7.6.6 rule-4 "bottom-of-MB" marker.
///
/// `above` / `below` participate in MV1 selection (per-pixel by
/// `j < 4` vs `j >= 4`); `left` / `right` participate in MV2
/// selection (per-pixel by `i < 4` vs `i >= 4`).
///
/// `current_block_at_mb_bottom` is set by the caller when this 8×8
/// block sits on the bottom row of its 16×16 macroblock (i.e. when
/// the §7.6.6 rule-4 substitution applies to the `below` neighbour
/// because it would otherwise reach into the MB below the current
/// MB).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObmcNeighbourhood {
    /// Status of the 8×8 luminance block above the current block.
    pub above: RemoteBlockKind,
    /// Status of the 8×8 luminance block below the current block.
    pub below: RemoteBlockKind,
    /// Status of the 8×8 luminance block to the left of the current
    /// block.
    pub left: RemoteBlockKind,
    /// Status of the 8×8 luminance block to the right of the current
    /// block.
    pub right: RemoteBlockKind,
    /// Whether the current 8×8 block sits on the bottom row of its
    /// 16×16 macroblock — triggers the §7.6.6 rule-4 below-MB
    /// substitution: the `below` neighbour's MV is replaced by the
    /// current block's MV regardless of `below`'s actual kind.
    pub current_block_at_mb_bottom: bool,
}

impl ObmcNeighbourhood {
    /// Resolve the per-pixel MV1 (above-or-below remote) for the
    /// given column / row. Pixels in the upper half (`row < 4`) take
    /// the above neighbour; pixels in the lower half (`row >= 4`)
    /// take the below neighbour, with the §7.6.6 rule-4
    /// "bottom-of-MB" substitution applied when relevant.
    fn mv1_at(&self, row: usize, current_mv: ObmcMv) -> ObmcMv {
        if row < 4 {
            self.above.resolved(current_mv)
        } else if self.current_block_at_mb_bottom {
            // §7.6.6 rule 4: the remote MV for the 8×8 luminance
            // block in the macroblock below the current macroblock
            // is replaced by the current block's MV.
            current_mv
        } else {
            self.below.resolved(current_mv)
        }
    }

    /// Resolve the per-pixel MV2 (left-or-right remote) for the given
    /// column / row. Pixels in the left half (`col < 4`) take the
    /// left neighbour; pixels in the right half (`col >= 4`) take the
    /// right neighbour.
    fn mv2_at(&self, col: usize, current_mv: ObmcMv) -> ObmcMv {
        if col < 4 {
            self.left.resolved(current_mv)
        } else {
            self.right.resolved(current_mv)
        }
    }
}

/// Caller-side knobs for the §7.6.6 process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObmcConfig {
    /// `false` when the caller signals that OBMC is disabled for this
    /// 8×8 block — either because the VOL header carried
    /// `obmc_disable == 1`, or because the §7.6.6 opening paragraph's
    /// S(GMC)-VOP mcsel-boundary rule applies. With `enabled = false`
    /// [`obmc_predict_block`] returns the bare MV0-only prediction
    /// `p(i, j) = q(i, j)` for every pixel.
    pub enabled: bool,
}

impl ObmcConfig {
    /// Default configuration — OBMC enabled.
    pub const fn enabled() -> Self {
        Self { enabled: true }
    }

    /// Convenience constructor for the disabled case.
    pub const fn disabled() -> Self {
        Self { enabled: false }
    }
}

/// Compose the §7.6.6 OBMC prediction for one 8×8 luminance block.
///
/// `current_mv` is the current block's own motion vector (MV0);
/// `neighbours` is the four-neighbour bundle of [`ObmcNeighbourhood`]
/// plus the rule-4 "bottom-of-MB" flag. `cfg.enabled = false`
/// short-circuits to the MV0-only prediction.
///
/// The caller supplies `sample`, a closure that takes
/// `(mv, col, row)` — a motion vector and an in-block pixel
/// coordinate — and returns the corresponding `u8` luminance sample
/// from the reference frame. The closure owns the reference-frame
/// buffer and any §7.6.2 half-sample / §7.6.4 quarter-sample
/// interpolation; OBMC just composes the three samples per the
/// §7.6.6 equation.
///
/// Returns the rounded 8×8 prediction block as
/// `[[u8; 8]; 8]` indexed `block[j][i]` (row-major in `j`, matching
/// the H0/H1/H2 indexing).
///
/// # Panics
///
/// Does not panic on its own; if `sample` panics that propagates.
pub fn obmc_predict_block<S>(
    current_mv: ObmcMv,
    neighbours: ObmcNeighbourhood,
    cfg: ObmcConfig,
    mut sample: S,
) -> [[u8; 8]; 8]
where
    S: FnMut(ObmcMv, usize, usize) -> u8,
{
    let mut block = [[0u8; 8]; 8];
    for j in 0..8 {
        for i in 0..8 {
            let q = sample(current_mv, i, j);
            if !cfg.enabled {
                block[j][i] = q;
                continue;
            }
            let mv1 = neighbours.mv1_at(j, current_mv);
            let mv2 = neighbours.mv2_at(i, current_mv);
            let r = sample(mv1, i, j);
            let s = sample(mv2, i, j);
            let h0 = OBMC_H0[j][i] as u32;
            let h1 = OBMC_H1[j][i] as u32;
            let h2 = OBMC_H2[j][i] as u32;
            // §7.6.6 equation: (q*H0 + r*H1 + s*H2 + 4) // 8
            // §3.4 `//` is integer division with rounding to the
            // nearest, ties away from zero; for non-negative
            // numerators that is `(num + denom/2) / denom`. The
            // §7.6.6 equation pre-supplies the `+ 4` rounding term,
            // so plain truncating division on the non-negative
            // weighted sum yields the spec result.
            let weighted = (q as u32) * h0 + (r as u32) * h1 + (s as u32) * h2 + 4;
            block[j][i] = (weighted / 8) as u8;
        }
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h_matrices_sum_to_eight_per_cell() {
        // §7.6.6 guarantees H0 + H1 + H2 = 8 per cell so that flat
        // reference samples reproduce the input. Spot-check every
        // cell.
        for j in 0..8 {
            for i in 0..8 {
                let sum = OBMC_H0[j][i] as u32 + OBMC_H1[j][i] as u32 + OBMC_H2[j][i] as u32;
                assert_eq!(sum, 8, "H0+H1+H2 mismatch at ({j},{i}): {sum}");
            }
        }
    }

    #[test]
    fn h0_matches_figure_7_35_corners_and_centre() {
        // Spot-check the Figure 7-35 transcription at the four corners
        // (the 4-valued cells) and the 4×4 interior (the 6-valued
        // cells).
        assert_eq!(OBMC_H0[0][0], 4);
        assert_eq!(OBMC_H0[0][7], 4);
        assert_eq!(OBMC_H0[7][0], 4);
        assert_eq!(OBMC_H0[7][7], 4);
        for j in 2..=5 {
            for i in 2..=5 {
                assert_eq!(OBMC_H0[j][i], 6, "interior cell at ({j},{i})");
            }
        }
    }

    #[test]
    fn h1_matches_figure_7_36_edges() {
        // Figure 7-36 has 2 along the full top and bottom rows.
        for i in 0..8 {
            assert_eq!(OBMC_H1[0][i], 2);
            assert_eq!(OBMC_H1[7][i], 2);
        }
        // The interior 4×8 strip (rows 2..=5) is all ones.
        for j in 2..=5 {
            for i in 0..8 {
                assert_eq!(OBMC_H1[j][i], 1);
            }
        }
        // Row 1 (and row 6) has the "2,2 in the middle four columns"
        // shape: 1 1 2 2 2 2 1 1.
        for j in [1, 6] {
            assert_eq!(OBMC_H1[j][0], 1);
            assert_eq!(OBMC_H1[j][1], 1);
            for i in 2..=5 {
                assert_eq!(OBMC_H1[j][i], 2);
            }
            assert_eq!(OBMC_H1[j][6], 1);
            assert_eq!(OBMC_H1[j][7], 1);
        }
    }

    #[test]
    fn h2_matches_figure_7_37_edges() {
        // Figure 7-37 has 2 along the full left and right columns.
        for j in 0..8 {
            assert_eq!(OBMC_H2[j][0], 2);
            assert_eq!(OBMC_H2[j][7], 2);
        }
        // The interior 8×4 strip (cols 2..=5) is all ones.
        for j in 0..8 {
            for i in 2..=5 {
                assert_eq!(OBMC_H2[j][i], 1);
            }
        }
        // Column 1 (and column 6) is 1 2 2 2 2 2 2 1 (rows 0 and 7
        // are 1; rows 1..=6 are 2).
        for i in [1, 6] {
            assert_eq!(OBMC_H2[0][i], 1);
            for j in 1..=6 {
                assert_eq!(OBMC_H2[j][i], 2);
            }
            assert_eq!(OBMC_H2[7][i], 1);
        }
    }

    #[test]
    fn flat_reference_reproduces_input() {
        // When q == r == s == C for every pixel, (8C + 4) / 8 = C
        // exactly, for any 0 ≤ C ≤ 255. Exercise both ends of the
        // luma range plus a mid-grey.
        for c in [0u8, 1, 42, 127, 128, 200, 254, 255] {
            let block = obmc_predict_block(
                ObmcMv::new(3, -2),
                ObmcNeighbourhood {
                    above: RemoteBlockKind::Inter(ObmcMv::new(1, 1)),
                    below: RemoteBlockKind::Inter(ObmcMv::new(-1, 1)),
                    left: RemoteBlockKind::Inter(ObmcMv::new(2, 0)),
                    right: RemoteBlockKind::Inter(ObmcMv::new(0, 2)),
                    current_block_at_mb_bottom: false,
                },
                ObmcConfig::enabled(),
                |_mv, _i, _j| c,
            );
            for row in &block {
                for &px in row {
                    assert_eq!(px, c);
                }
            }
        }
    }

    #[test]
    fn quadrant_remote_selection() {
        // Build a sample provider that returns a different sentinel
        // for each of the four candidate MVs, so we can confirm
        // which remote MV is reached for each (col, row) pair.
        let mv0 = ObmcMv::new(0, 0);
        let mv_above = ObmcMv::new(10, 0);
        let mv_below = ObmcMv::new(20, 0);
        let mv_left = ObmcMv::new(0, 10);
        let mv_right = ObmcMv::new(0, 20);
        let neighbours = ObmcNeighbourhood {
            above: RemoteBlockKind::Inter(mv_above),
            below: RemoteBlockKind::Inter(mv_below),
            left: RemoteBlockKind::Inter(mv_left),
            right: RemoteBlockKind::Inter(mv_right),
            current_block_at_mb_bottom: false,
        };
        // Track which (mv1, mv2) pair was passed to `sample` at each
        // pixel coordinate by recording the MV1/MV2 selections.
        // We do this by routing through `obmc_predict_block` with a
        // closure that captures the per-pixel queries. Because the
        // closure runs (q, r, s) per pixel and `q` is always `mv0`,
        // we recover the per-pixel `(MV1, MV2)` selection by checking
        // each pixel's three calls.
        let mut calls: Vec<(usize, usize, ObmcMv)> = Vec::new();
        let _ = obmc_predict_block(mv0, neighbours, ObmcConfig::enabled(), |mv, i, j| {
            calls.push((i, j, mv));
            0
        });
        // Three calls per pixel, in order: (mv0, mv1, mv2).
        assert_eq!(calls.len(), 8 * 8 * 3);
        for j in 0..8 {
            for i in 0..8 {
                let base = (j * 8 + i) * 3;
                assert_eq!(calls[base].2, mv0, "q at ({i},{j})");
                let expected_mv1 = if j < 4 { mv_above } else { mv_below };
                let expected_mv2 = if i < 4 { mv_left } else { mv_right };
                assert_eq!(calls[base + 1].2, expected_mv1, "MV1 at ({i},{j})");
                assert_eq!(calls[base + 2].2, expected_mv2, "MV2 at ({i},{j})");
            }
        }
    }

    #[test]
    fn rule_1_not_coded_uses_zero_remote_mv() {
        let mv0 = ObmcMv::new(7, -3);
        let neighbours = ObmcNeighbourhood {
            // Above is not coded → MV1 (upper half) = (0, 0).
            above: RemoteBlockKind::NotCoded,
            below: RemoteBlockKind::Inter(ObmcMv::new(1, 1)),
            // Left is not coded → MV2 (left half) = (0, 0).
            left: RemoteBlockKind::NotCoded,
            right: RemoteBlockKind::Inter(ObmcMv::new(2, 2)),
            current_block_at_mb_bottom: false,
        };
        let mut mv1_at_0_0 = ObmcMv::new(99, 99);
        let mut mv2_at_0_0 = ObmcMv::new(99, 99);
        let _ = obmc_predict_block(mv0, neighbours, ObmcConfig::enabled(), |mv, i, j| {
            if (i, j) == (0, 0) {
                // Three calls land at (0,0): q (mv0), r (mv1), s (mv2).
                // Capture mv1 / mv2 by sentinel comparison.
                if mv == mv0 {
                    // first call — current MV
                } else if mv1_at_0_0 == ObmcMv::new(99, 99) {
                    mv1_at_0_0 = mv;
                } else {
                    mv2_at_0_0 = mv;
                }
            }
            0
        });
        assert_eq!(mv1_at_0_0, ObmcMv::ZERO);
        assert_eq!(mv2_at_0_0, ObmcMv::ZERO);
    }

    #[test]
    fn rule_2_intra_neighbour_replays_current_mv() {
        let mv0 = ObmcMv::new(5, 6);
        let neighbours = ObmcNeighbourhood {
            above: RemoteBlockKind::Intra,
            below: RemoteBlockKind::Inter(ObmcMv::new(1, 1)),
            left: RemoteBlockKind::Inter(ObmcMv::new(2, 2)),
            right: RemoteBlockKind::Intra,
            current_block_at_mb_bottom: false,
        };
        // Upper-left pixel (0, 0): MV1 from above (Intra → mv0), MV2
        // from left (Inter (2, 2)).
        // Upper-right pixel (7, 0): MV1 from above (Intra → mv0), MV2
        // from right (Intra → mv0).
        let mut captured: Vec<(usize, usize, ObmcMv)> = Vec::new();
        let _ = obmc_predict_block(mv0, neighbours, ObmcConfig::enabled(), |mv, i, j| {
            captured.push((i, j, mv));
            0
        });
        // pixel (0, 0): calls 0,1,2 → q, r, s
        assert_eq!(captured[1].2, mv0); // MV1 substituted Intra→mv0
        assert_eq!(captured[2].2, ObmcMv::new(2, 2));
        // pixel (7, 0): three calls starting at index (0*8 + 7) * 3 = 21
        assert_eq!(captured[21 + 1].2, mv0); // MV1 above intra
        assert_eq!(captured[21 + 2].2, mv0); // MV2 right intra
    }

    #[test]
    fn rule_3_absent_neighbour_replays_current_mv() {
        let mv0 = ObmcMv::new(-4, 9);
        let neighbours = ObmcNeighbourhood {
            // Block sits on the top edge of the VOP → above absent.
            above: RemoteBlockKind::Absent,
            below: RemoteBlockKind::Inter(ObmcMv::new(1, 1)),
            // Block sits on the left edge of the VOP → left absent.
            left: RemoteBlockKind::Absent,
            right: RemoteBlockKind::Inter(ObmcMv::new(2, 2)),
            current_block_at_mb_bottom: false,
        };
        let mut captured: Vec<(usize, usize, ObmcMv)> = Vec::new();
        let _ = obmc_predict_block(mv0, neighbours, ObmcConfig::enabled(), |mv, i, j| {
            captured.push((i, j, mv));
            0
        });
        // pixel (0, 0): MV1 above absent → mv0; MV2 left absent → mv0.
        assert_eq!(captured[1].2, mv0);
        assert_eq!(captured[2].2, mv0);
    }

    #[test]
    fn rule_4_bottom_of_mb_below_remote_replaced_by_current() {
        let mv0 = ObmcMv::new(11, -7);
        let neighbours = ObmcNeighbourhood {
            above: RemoteBlockKind::Inter(ObmcMv::new(3, 4)),
            // `below` would have reached into the next MB — but
            // current_block_at_mb_bottom = true overrides it.
            below: RemoteBlockKind::Inter(ObmcMv::new(99, 99)),
            left: RemoteBlockKind::Inter(ObmcMv::new(5, 6)),
            right: RemoteBlockKind::Inter(ObmcMv::new(7, 8)),
            current_block_at_mb_bottom: true,
        };
        let mut captured: Vec<(usize, usize, ObmcMv)> = Vec::new();
        let _ = obmc_predict_block(mv0, neighbours, ObmcConfig::enabled(), |mv, i, j| {
            captured.push((i, j, mv));
            0
        });
        // pixel (0, 4): three calls at index (4*8 + 0) * 3 = 96.
        // MV1 should be mv0 (rule 4 substitution), NOT (99, 99).
        assert_eq!(captured[96 + 1].2, mv0);
        // Upper-half pixel (0, 3) at index (3*8 + 0) * 3 = 72 still
        // takes `above` because the rule only governs the "below"
        // remote.
        assert_eq!(captured[72 + 1].2, ObmcMv::new(3, 4));
    }

    #[test]
    fn disabled_config_returns_bare_mv0_prediction() {
        // When cfg.enabled = false the sample provider should only be
        // called once per pixel (for q at MV0), and the per-pixel
        // result equals that single sample.
        let mv0 = ObmcMv::new(2, 2);
        let neighbours = ObmcNeighbourhood {
            above: RemoteBlockKind::Inter(ObmcMv::new(99, 99)),
            below: RemoteBlockKind::Inter(ObmcMv::new(99, 99)),
            left: RemoteBlockKind::Inter(ObmcMv::new(99, 99)),
            right: RemoteBlockKind::Inter(ObmcMv::new(99, 99)),
            current_block_at_mb_bottom: false,
        };
        let mut call_count = 0u32;
        let block = obmc_predict_block(mv0, neighbours, ObmcConfig::disabled(), |mv, i, j| {
            call_count += 1;
            // Sentinel: encode the pixel coordinates in the result.
            // mv must be mv0 — never a remote.
            assert_eq!(mv, mv0);
            ((i * 8 + j) as u8).wrapping_add(50)
        });
        assert_eq!(call_count, 64);
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(block[j][i], ((i * 8 + j) as u8).wrapping_add(50));
            }
        }
    }

    #[test]
    fn rounding_term_is_plus_four() {
        // q*H0 + r*H1 + s*H2 + 4 with truncating /8 must match the
        // spec's // operator on non-negative integers. Exercise a
        // worked case at pixel (3, 3) where H0=6, H1=1, H2=1, sum=8.
        //  q=10, r=20, s=30 → weighted = 60 + 20 + 30 + 4 = 114;
        //   114 / 8 = 14 (truncating); 114 // 8 (round half away
        //   from zero) = 14 (since 14*8 = 112, residue 2 < 4). ✓
        //  q=10, r=20, s=31 → weighted = 60 + 20 + 31 + 4 = 115;
        //   115 / 8 = 14; 115 // 8 = 14 (residue 3 < 4). ✓
        //  q=10, r=20, s=33 → weighted = 60 + 20 + 33 + 4 = 117;
        //   117 / 8 = 14; 117 // 8 = 15 (residue 5 >= 4) — but the
        //   §7.6.6 equation pre-rounds by adding 4 BEFORE division,
        //   so the truncating result is the canonical answer. The
        //   spec's `+ 4` term subsumes the rounding bias for
        //   non-negative numerators.
        let mv0 = ObmcMv::ZERO;
        let mv_r = ObmcMv::new(1, 0);
        let mv_s = ObmcMv::new(0, 1);
        let neighbours = ObmcNeighbourhood {
            above: RemoteBlockKind::Inter(mv_r),
            below: RemoteBlockKind::Inter(ObmcMv::ZERO),
            left: RemoteBlockKind::Inter(mv_s),
            right: RemoteBlockKind::Inter(ObmcMv::ZERO),
            current_block_at_mb_bottom: false,
        };
        for &(q, r, s, expect) in &[
            (10u32, 20u32, 30u32, 14u8),
            (10, 20, 31, 14),
            (10, 20, 33, 14),
            (100, 100, 100, 100),
            (255, 255, 255, 255),
            (0, 0, 0, 0),
        ] {
            let block = obmc_predict_block(mv0, neighbours, ObmcConfig::enabled(), |mv, _i, _j| {
                if mv == mv0 {
                    q as u8
                } else if mv == mv_r {
                    r as u8
                } else if mv == mv_s {
                    s as u8
                } else {
                    0
                }
            });
            // pixel (3, 3) at H0=6, H1=1, H2=1.
            // (Verify Figure 7-35/36/37 transcription says so.)
            assert_eq!(OBMC_H0[3][3], 6);
            assert_eq!(OBMC_H1[3][3], 1);
            assert_eq!(OBMC_H2[3][3], 1);
            assert_eq!(
                block[3][3], expect,
                "q={q} r={r} s={s} → expect {expect}, got {}",
                block[3][3]
            );
        }
    }

    #[test]
    fn remote_block_kind_resolution() {
        let cur = ObmcMv::new(5, 6);
        assert_eq!(
            RemoteBlockKind::Inter(ObmcMv::new(1, 2)).resolved(cur),
            ObmcMv::new(1, 2),
        );
        assert_eq!(RemoteBlockKind::NotCoded.resolved(cur), ObmcMv::ZERO);
        assert_eq!(RemoteBlockKind::Intra.resolved(cur), cur);
        assert_eq!(RemoteBlockKind::Absent.resolved(cur), cur);
    }
}

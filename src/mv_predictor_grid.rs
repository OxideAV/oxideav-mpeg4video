//! §7.6.5 / Figure 7-34 spatial motion-vector predictor candidate
//! gathering.
//!
//! [`crate::motion::predict_motion_vector`] computes the median of three
//! caller-supplied candidate predictors (`MV1`, `MV2`, `MV3`). This
//! module gathers those three candidates from the already-decoded
//! spatial neighbourhood per the per-current-block layout of
//! Figure 7-34.
//!
//! ## Spec references
//!
//! All numeric values, block-position rules, and validity-substitution
//! rules come from ISO/IEC 14496-2:2004 (3rd edition) and the in-repo
//! ASCII transcription of Figure 7-34
//! (`docs/video/mpeg4-visual/figure-7-34-mv-predictor-layout.md`):
//!
//! * §7.6.5 (printed page 283) — the four candidate-validity
//!   substitution rules and the boundary-substitution rule that treats
//!   any neighbour outside the current VOP / video packet / GOB as
//!   transparent.
//! * Figure 7-34 (printed page 284) — the spatial position of `MV1`,
//!   `MV2`, `MV3` for each of the four 8x8 luminance blocks of the
//!   current macroblock.
//! * §6.1.3.4 / Figure 6-8 — the 4-block luminance numbering
//!   (`block 1 = top-left`, `block 2 = top-right`, `block 3 =
//!   bottom-left`, `block 4 = bottom-right`).
//! * §7.6.5 (printed page 283, paragraph above Figure 7-34) — "when
//!   only one motion vector is present for the whole macroblock
//!   ... the top-left case is used for the single MV of the
//!   macroblock."
//!
//! ## Block-position lookup table (Figure 7-34)
//!
//! For each current 8x8 block in macroblock `(R, C)`, the three
//! candidates resolve to the following luminance-sub-block positions in
//! the `(2 * mb_rows) x (2 * mb_cols)` per-block sub-grid:
//!
//! ```text
//! current block  (0-indexed i ∈ 0..=3, Figure 6-8)
//!   |  current sub-grid pos
//!   |    | MV1                | MV2                | MV3                |
//!   +----+--------------------+--------------------+--------------------+
//!   |  0 | (2R   , 2C-1)      | (2R-1, 2C  )       | (2R-1, 2C+2)       |
//!   |  1 | (2R   , 2C  )      | (2R-1, 2C+1)       | (2R-1, 2C+2)       |
//!   |  2 | (2R+1, 2C-1)       | (2R  , 2C  )       | (2R  , 2C+1)       |
//!   |  3 | (2R+1, 2C  )       | (2R  , 2C  )       | (2R  , 2C+1)       |
//! ```
//!
//! Read against the ASCII transcription of Figure 7-34:
//!
//! * Block 1 (`i = 0`, TL of current MB) — MV1 is the top-right
//!   sub-block of the MB to the left, MV2 is the bottom-left sub-block
//!   of the MB above, MV3 is the bottom-left sub-block of the MB
//!   above-right (the second column-step right of MV2 in the figure
//!   moves into the next macroblock).
//! * Block 2 (`i = 1`, TR of current MB) — MV1 is block 1 of the
//!   current MB (the sub-block to its immediate left), MV2 is the
//!   bottom-right sub-block of the MB above (directly above block 2),
//!   MV3 is the bottom-left sub-block of the MB above-right.
//! * Block 3 (`i = 2`, BL of current MB) — MV1 is the bottom-right
//!   sub-block of the MB to the left, MV2 is block 1 of the current MB
//!   (directly above block 3), MV3 is block 2 of the current MB
//!   (above-right of block 3).
//! * Block 4 (`i = 3`, BR of current MB) — all three candidates are
//!   inside the current MB: MV1 = block 3 (left of block 4), MV2 =
//!   block 1, MV3 = block 2. The figure places MV2 at the TL position
//!   and MV3 at the TR position of the current MB; together with MV1
//!   at the BL position they tile the current MB around the BR-corner
//!   current block.
//!
//! ## Validity / substitution
//!
//! After resolving the three sub-grid positions a candidate is
//! [`Some(MotionVector)`] when its luminance sub-block carries a
//! valid (non-transparent) motion vector inside the current VOP +
//! video packet + GOB, and [`None`] otherwise. The boundary cases
//! collapse into [`None`] via:
//!
//! * a negative sub-grid coordinate (off the top / left of the VOP);
//! * a sub-grid coordinate beyond `2 * mb_rows` / `2 * mb_cols` (off
//!   the bottom / right of the VOP); or
//! * a recorded [`MbMv::Absent`] for the containing macroblock (the
//!   caller's signal for "MB outside the current video packet /
//!   GOB" or "transparent MB"); or
//! * a per-block transparency mask bit set.
//!
//! The §7.6.5 four substitution rules (one invalid → zero, two
//! invalid → the third's value, three invalid → all zero) are applied
//! by [`crate::motion::predict_motion_vector`] on the gathered
//! `[Option<MotionVector>; 3]` triple, so this module's contract is
//! "supply `None` where appropriate, supply `Some(MV)` otherwise."

use crate::motion::MotionVector;

/// The motion-vector content of one macroblock recorded in
/// [`MvGrid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbMv {
    /// The macroblock is outside the current VOP / video packet / GOB
    /// or is wholly transparent / not-yet-decoded. Every sub-block
    /// query against the four contained positions returns [`None`].
    Absent,
    /// The macroblock has a single 16x16 motion vector — Figure 7-34's
    /// "1-MV" path. The same vector is reported for all four sub-block
    /// positions inside the macroblock.
    OneMv(MotionVector),
    /// The macroblock has four 8x8 motion vectors (inter4v / 4-MV
    /// mode). Indexing follows Figure 6-8: `[block 1 (TL, i=0), block 2
    /// (TR, i=1), block 3 (BL, i=2), block 4 (BR, i=3)]`.
    FourMv([MotionVector; 4]),
}

/// 1-MV vs 4-MV mode plus the per-luma-block transparency mask of a
/// macroblock — what [`MvGrid::record`] needs from the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MbMvRecord {
    /// The macroblock-level motion-vector content.
    pub content: MbMv,
    /// Per-luma-sub-block transparency mask. A `true` entry marks the
    /// 8x8 block as transparent (and therefore a [`None`] candidate
    /// when queried as MV1 / MV2 / MV3 by a later block's predictor).
    /// Indexing follows Figure 6-8 luminance ordering, matching
    /// [`MbMv::FourMv`].
    pub transparent: [bool; 4],
}

impl MbMvRecord {
    /// Convenience constructor for a 1-MV macroblock with all four
    /// blocks opaque.
    pub fn one_mv(mv: MotionVector) -> Self {
        Self {
            content: MbMv::OneMv(mv),
            transparent: [false; 4],
        }
    }

    /// Convenience constructor for a 4-MV macroblock with all four
    /// blocks opaque.
    pub fn four_mv(mvs: [MotionVector; 4]) -> Self {
        Self {
            content: MbMv::FourMv(mvs),
            transparent: [false; 4],
        }
    }

    /// Convenience constructor for an [`MbMv::Absent`] macroblock —
    /// e.g. an out-of-VOP boundary fill or a transparent macroblock.
    pub fn absent() -> Self {
        Self {
            content: MbMv::Absent,
            transparent: [true; 4],
        }
    }
}

/// Per-VOP storage of decoded luminance-block motion vectors for the
/// §7.6.5 spatial predictor.
///
/// Sized `mb_rows x mb_cols` macroblocks; each macroblock holds an
/// [`MbMvRecord`]. The `record_*` setters update one macroblock; the
/// `predictor_candidates` query resolves the three Figure 7-34
/// positions for a given current block back into a
/// `[Option<MotionVector>; 3]` ready to feed into
/// [`crate::motion::predict_motion_vector`].
#[derive(Debug, Clone)]
pub struct MvGrid {
    mb_rows: usize,
    mb_cols: usize,
    /// Per-macroblock record. Cell `r * mb_cols + c` is MB at
    /// `(r, c)`. Every cell starts [`MbMv::Absent`] so blocks decoded
    /// before any neighbour was recorded see [`None`] across the
    /// board.
    cells: Vec<MbMvRecord>,
}

/// The four §6.1.3.4 / Figure 6-8 luminance sub-blocks per macroblock.
pub const LUMA_BLOCKS_PER_MB: usize = 4;

/// `MvGrid` errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvGridError {
    /// `mb_row` was `>= mb_rows`.
    RowOutOfBounds {
        /// The supplied `mb_row`.
        mb_row: usize,
        /// The grid's row count.
        mb_rows: usize,
    },
    /// `mb_col` was `>= mb_cols`.
    ColOutOfBounds {
        /// The supplied `mb_col`.
        mb_col: usize,
        /// The grid's column count.
        mb_cols: usize,
    },
    /// `block_index` was `>= 4`. Only the four Figure 6-8 luminance
    /// sub-block indices `0..=3` are supported.
    InvalidBlockIndex(usize),
}

impl core::fmt::Display for MvGridError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RowOutOfBounds { mb_row, mb_rows } => write!(
                f,
                "mv-grid row {mb_row} is out of bounds (grid has {mb_rows} rows)"
            ),
            Self::ColOutOfBounds { mb_col, mb_cols } => write!(
                f,
                "mv-grid col {mb_col} is out of bounds (grid has {mb_cols} cols)"
            ),
            Self::InvalidBlockIndex(i) => write!(
                f,
                "block_index {i} is not a Figure 6-8 luminance sub-block (0..=3)"
            ),
        }
    }
}

impl std::error::Error for MvGridError {}

impl MvGrid {
    /// Allocate an empty grid of `mb_rows x mb_cols` macroblocks.
    /// Every cell starts [`MbMv::Absent`].
    pub fn new(mb_rows: usize, mb_cols: usize) -> Self {
        Self {
            mb_rows,
            mb_cols,
            cells: vec![MbMvRecord::absent(); mb_rows.saturating_mul(mb_cols)],
        }
    }

    /// Number of macroblock rows in the VOP.
    #[inline]
    pub fn mb_rows(&self) -> usize {
        self.mb_rows
    }

    /// Number of macroblock columns in the VOP.
    #[inline]
    pub fn mb_cols(&self) -> usize {
        self.mb_cols
    }

    #[inline]
    fn cell_index(&self, mb_row: usize, mb_col: usize) -> usize {
        mb_row * self.mb_cols + mb_col
    }

    /// Borrow the macroblock record at `(mb_row, mb_col)`. Returns
    /// `None` for out-of-grid coordinates.
    pub fn get(&self, mb_row: usize, mb_col: usize) -> Option<&MbMvRecord> {
        if mb_row >= self.mb_rows || mb_col >= self.mb_cols {
            return None;
        }
        Some(&self.cells[self.cell_index(mb_row, mb_col)])
    }

    /// Record a macroblock's MV content + per-block transparency mask
    /// at `(mb_row, mb_col)`. Subsequent [`predictor_candidates`][Self::predictor_candidates]
    /// queries see this MB as the appropriate `MV1` / `MV2` / `MV3`
    /// neighbour for blocks decoded later in raster order.
    pub fn record(
        &mut self,
        mb_row: usize,
        mb_col: usize,
        record: MbMvRecord,
    ) -> Result<(), MvGridError> {
        if mb_row >= self.mb_rows {
            return Err(MvGridError::RowOutOfBounds {
                mb_row,
                mb_rows: self.mb_rows,
            });
        }
        if mb_col >= self.mb_cols {
            return Err(MvGridError::ColOutOfBounds {
                mb_col,
                mb_cols: self.mb_cols,
            });
        }
        let idx = self.cell_index(mb_row, mb_col);
        self.cells[idx] = record;
        Ok(())
    }

    /// Convenience wrapper for [`Self::record`] with a fully-opaque
    /// 1-MV macroblock.
    pub fn record_one_mv(
        &mut self,
        mb_row: usize,
        mb_col: usize,
        mv: MotionVector,
    ) -> Result<(), MvGridError> {
        self.record(mb_row, mb_col, MbMvRecord::one_mv(mv))
    }

    /// Convenience wrapper for [`Self::record`] with a fully-opaque
    /// 4-MV macroblock. The four sub-block MVs are indexed per
    /// Figure 6-8 (`[TL, TR, BL, BR]`).
    pub fn record_four_mv(
        &mut self,
        mb_row: usize,
        mb_col: usize,
        mvs: [MotionVector; LUMA_BLOCKS_PER_MB],
    ) -> Result<(), MvGridError> {
        self.record(mb_row, mb_col, MbMvRecord::four_mv(mvs))
    }

    /// Convenience wrapper for [`Self::record`] writing an
    /// [`MbMv::Absent`] entry — the §7.6.5 "treated as transparent"
    /// boundary substitution for macroblocks outside the current
    /// video packet / GOB.
    pub fn record_absent(&mut self, mb_row: usize, mb_col: usize) -> Result<(), MvGridError> {
        self.record(mb_row, mb_col, MbMvRecord::absent())
    }

    /// Resolve the motion vector at the luminance sub-grid position
    /// `(sub_row, sub_col)` if any.
    ///
    /// Returns [`None`] when the position falls outside the VOP, when
    /// the containing macroblock is [`MbMv::Absent`], or when the
    /// per-block transparency mask marks the position transparent.
    /// The sub-grid is `(2 * mb_rows) x (2 * mb_cols)`; sub-grid
    /// `(0, 0)` is block 1 (TL) of MB `(0, 0)`.
    fn sub_block_mv(&self, sub_row: isize, sub_col: isize) -> Option<MotionVector> {
        if sub_row < 0 || sub_col < 0 {
            return None;
        }
        let sub_row = sub_row as usize;
        let sub_col = sub_col as usize;
        if sub_row >= 2 * self.mb_rows || sub_col >= 2 * self.mb_cols {
            return None;
        }
        let mb_row = sub_row / 2;
        let mb_col = sub_col / 2;
        // sub_row / sub_col bit-0 picks the row / col within the MB.
        // Figure 6-8 numbering: index = 2 * row_bit + col_bit.
        let row_bit = sub_row & 1;
        let col_bit = sub_col & 1;
        let block_idx = 2 * row_bit + col_bit;
        let cell = &self.cells[self.cell_index(mb_row, mb_col)];
        if cell.transparent[block_idx] {
            return None;
        }
        match cell.content {
            MbMv::Absent => None,
            MbMv::OneMv(mv) => Some(mv),
            MbMv::FourMv(mvs) => Some(mvs[block_idx]),
        }
    }

    /// Gather the three §7.6.5 / Figure 7-34 candidate predictors for
    /// the current 8x8 block at macroblock `(mb_row, mb_col)` and
    /// luminance-block index `block_index ∈ 0..=3` (Figure 6-8
    /// ordering). Returns a `[Option<MotionVector>; 3]` triple ready
    /// to feed into [`crate::motion::predict_motion_vector`], which
    /// applies the §7.6.5 four substitution rules.
    ///
    /// The "1-MV mode (and always when `short_video_header == 1`)
    /// uses the top-left case for the single MV of the macroblock"
    /// rule is the caller's responsibility — pass
    /// `block_index = 0` for a 1-MV current macroblock. The mapping
    /// is symmetric: a 1-MV neighbouring macroblock contributes the
    /// same MV to every sub-grid query inside it, so no special-casing
    /// is needed on the neighbour side.
    ///
    /// Returns [`MvGridError`] only when `mb_row` / `mb_col` /
    /// `block_index` is out of range; never reads outside the grid.
    pub fn predictor_candidates(
        &self,
        mb_row: usize,
        mb_col: usize,
        block_index: usize,
    ) -> Result<[Option<MotionVector>; 3], MvGridError> {
        if mb_row >= self.mb_rows {
            return Err(MvGridError::RowOutOfBounds {
                mb_row,
                mb_rows: self.mb_rows,
            });
        }
        if mb_col >= self.mb_cols {
            return Err(MvGridError::ColOutOfBounds {
                mb_col,
                mb_cols: self.mb_cols,
            });
        }
        if block_index >= LUMA_BLOCKS_PER_MB {
            return Err(MvGridError::InvalidBlockIndex(block_index));
        }
        let r = mb_row as isize;
        let c = mb_col as isize;
        let (p1, p2, p3) = match block_index {
            // Block 1 (TL, i = 0): MV1 = left-MB's TR sub-block;
            // MV2 = above-MB's BL sub-block; MV3 = above-right MB's
            // BL sub-block.
            0 => (
                (2 * r, 2 * c - 1),
                (2 * r - 1, 2 * c),
                (2 * r - 1, 2 * c + 2),
            ),
            // Block 2 (TR, i = 1): MV1 = current MB's block 1
            // (sub_row = 2r, sub_col = 2c); MV2 = above-MB's BR
            // sub-block (sub_row = 2r-1, sub_col = 2c+1); MV3 =
            // above-right MB's BL sub-block (sub_row = 2r-1,
            // sub_col = 2c+2).
            1 => (
                (2 * r, 2 * c),
                (2 * r - 1, 2 * c + 1),
                (2 * r - 1, 2 * c + 2),
            ),
            // Block 3 (BL, i = 2): MV1 = left-MB's BR sub-block
            // (sub_row = 2r+1, sub_col = 2c-1); MV2 = current MB's
            // block 1 (sub_row = 2r, sub_col = 2c); MV3 = current MB's
            // block 2 (sub_row = 2r, sub_col = 2c+1).
            2 => ((2 * r + 1, 2 * c - 1), (2 * r, 2 * c), (2 * r, 2 * c + 1)),
            // Block 4 (BR, i = 3): MV1 = current MB's block 3
            // (sub_row = 2r+1, sub_col = 2c); MV2 = current MB's
            // block 1 (sub_row = 2r, sub_col = 2c); MV3 = current
            // MB's block 2 (sub_row = 2r, sub_col = 2c+1).
            3 => ((2 * r + 1, 2 * c), (2 * r, 2 * c), (2 * r, 2 * c + 1)),
            _ => unreachable!(),
        };
        Ok([
            self.sub_block_mv(p1.0, p1.1),
            self.sub_block_mv(p2.0, p2.1),
            self.sub_block_mv(p3.0, p3.1),
        ])
    }
}

/// Standalone helper: gather the Figure 7-34 candidate predictors for
/// the current block at `(mb_row, mb_col, block_index)` against the
/// supplied [`MvGrid`]. Equivalent to
/// [`MvGrid::predictor_candidates`]; provided for symmetry with the
/// other `*_motion_vector` / `*_motion_vectors` free functions of
/// [`crate::motion`].
///
/// ```
/// use oxideav_mpeg4video::{
///     gather_mv_predictor_candidates, predict_motion_vector, MotionVector, MvGrid,
/// };
///
/// // 2 x 2 macroblocks, all opaque, three 1-MV neighbours around MB (1, 1).
/// let mut grid = MvGrid::new(2, 2);
/// grid.record_one_mv(0, 0, MotionVector { x: 1, y: 2 }).unwrap();   // above-left
/// grid.record_one_mv(0, 1, MotionVector { x: 3, y: 4 }).unwrap();   // above
/// grid.record_one_mv(1, 0, MotionVector { x: 5, y: 6 }).unwrap();   // left
///
/// // We're decoding block 1 (TL) of MB (1, 1). MV3 here is the MB
/// // above-right of (1, 1) — out of the 2x2 grid, so None.
/// let candidates = gather_mv_predictor_candidates(&grid, 1, 1, 0).unwrap();
/// let predictor = predict_motion_vector(candidates);
/// // With one invalid candidate (MV3 = None) the §7.6.5 rule sets it
/// // to zero, and the median over {(5, 6), (3, 4), (0, 0)} → (3, 4).
/// assert_eq!(predictor, MotionVector { x: 3, y: 4 });
/// ```
pub fn gather_mv_predictor_candidates(
    grid: &MvGrid,
    mb_row: usize,
    mb_col: usize,
    block_index: usize,
) -> Result<[Option<MotionVector>; 3], MvGridError> {
    grid.predictor_candidates(mb_row, mb_col, block_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::predict_motion_vector;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    // ---------------------------------------------------------------------
    // Constructors + basic invariants.
    // ---------------------------------------------------------------------

    #[test]
    fn mb_mv_record_constructors() {
        let one = MbMvRecord::one_mv(mv(1, 2));
        assert_eq!(one.content, MbMv::OneMv(mv(1, 2)));
        assert_eq!(one.transparent, [false; 4]);

        let four = MbMvRecord::four_mv([mv(1, 1), mv(2, 2), mv(3, 3), mv(4, 4)]);
        assert!(matches!(four.content, MbMv::FourMv(_)));
        assert_eq!(four.transparent, [false; 4]);

        let abs = MbMvRecord::absent();
        assert_eq!(abs.content, MbMv::Absent);
        assert_eq!(abs.transparent, [true; 4]);
    }

    #[test]
    fn new_grid_starts_absent_everywhere() {
        let grid = MvGrid::new(3, 4);
        assert_eq!(grid.mb_rows(), 3);
        assert_eq!(grid.mb_cols(), 4);
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(grid.get(r, c).unwrap().content, MbMv::Absent);
            }
        }
        assert!(grid.get(3, 0).is_none());
        assert!(grid.get(0, 4).is_none());
    }

    #[test]
    fn record_rejects_out_of_bounds() {
        let mut grid = MvGrid::new(2, 2);
        assert!(matches!(
            grid.record_one_mv(2, 0, mv(0, 0)),
            Err(MvGridError::RowOutOfBounds {
                mb_row: 2,
                mb_rows: 2
            })
        ));
        assert!(matches!(
            grid.record_one_mv(0, 2, mv(0, 0)),
            Err(MvGridError::ColOutOfBounds {
                mb_col: 2,
                mb_cols: 2
            })
        ));
    }

    #[test]
    fn predictor_candidates_rejects_out_of_bounds() {
        let grid = MvGrid::new(2, 2);
        assert!(matches!(
            grid.predictor_candidates(2, 0, 0),
            Err(MvGridError::RowOutOfBounds {
                mb_row: 2,
                mb_rows: 2
            })
        ));
        assert!(matches!(
            grid.predictor_candidates(0, 2, 0),
            Err(MvGridError::ColOutOfBounds {
                mb_col: 2,
                mb_cols: 2
            })
        ));
        assert!(matches!(
            grid.predictor_candidates(0, 0, 4),
            Err(MvGridError::InvalidBlockIndex(4))
        ));
    }

    // ---------------------------------------------------------------------
    // Top-left-of-VOP cases (`MB (0, 0)`, any block).
    // ---------------------------------------------------------------------

    #[test]
    fn block_0_of_mb_0_0_has_all_invalid_candidates() {
        let grid = MvGrid::new(3, 3);
        // MV1 sub-grid (0, -1), MV2 (-1, 0), MV3 (-1, 2). All
        // outside the VOP → None.
        let cs = grid.predictor_candidates(0, 0, 0).unwrap();
        assert_eq!(cs, [None, None, None]);
    }

    #[test]
    fn block_1_of_mb_0_0_only_mv1_in_vop() {
        // For block 1 (i = 1) at MB (0, 0): MV1 = current MB's
        // block 0 at sub-grid (0, 0); MV2 = (-1, 1) outside;
        // MV3 = (-1, 2) outside.
        let mut grid = MvGrid::new(3, 3);
        grid.record_one_mv(0, 0, mv(7, 11)).unwrap();
        let cs = grid.predictor_candidates(0, 0, 1).unwrap();
        assert_eq!(cs, [Some(mv(7, 11)), None, None]);
    }

    #[test]
    fn block_2_of_mb_0_0_only_mv2_mv3_in_vop() {
        // For block 2 (i = 2) at MB (0, 0): MV1 = sub-grid (1, -1)
        // outside; MV2 = (0, 0) = current MB's block 0; MV3 =
        // (0, 1) = current MB's block 1.
        let mut grid = MvGrid::new(3, 3);
        grid.record_four_mv(0, 0, [mv(1, 1), mv(2, 2), mv(3, 3), mv(4, 4)])
            .unwrap();
        let cs = grid.predictor_candidates(0, 0, 2).unwrap();
        assert_eq!(cs, [None, Some(mv(1, 1)), Some(mv(2, 2))]);
    }

    #[test]
    fn block_3_of_mb_0_0_all_three_candidates_in_vop() {
        // Block 3 (i = 3) at MB (0, 0) has all three candidates
        // inside the current MB.
        let mut grid = MvGrid::new(3, 3);
        grid.record_four_mv(0, 0, [mv(1, 1), mv(2, 2), mv(3, 3), mv(4, 4)])
            .unwrap();
        let cs = grid.predictor_candidates(0, 0, 3).unwrap();
        assert_eq!(cs, [Some(mv(3, 3)), Some(mv(1, 1)), Some(mv(2, 2))]);
    }

    // ---------------------------------------------------------------------
    // Block-by-block Figure 7-34 verification, all neighbours opaque,
    // all 4-MV mode so each sub-block carries a distinct MV.
    // ---------------------------------------------------------------------

    fn populated_3x3_grid() -> MvGrid {
        // Layout: 3x3 macroblocks; MB (r, c) holds 4-MV macroblock
        // [(10*r + c, 0), (10*r + c, 1), (10*r + c, 2), (10*r + c, 3)]
        // so the candidate at any sub-block uniquely identifies its
        // owning MB and the block index inside it.
        let mut grid = MvGrid::new(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                let tag = (10 * r as i32) + c as i32;
                grid.record_four_mv(r, c, [mv(tag, 0), mv(tag, 1), mv(tag, 2), mv(tag, 3)])
                    .unwrap();
            }
        }
        grid
    }

    #[test]
    fn block_0_picks_left_mb_block_1_above_mb_block_2_above_right_mb_block_2() {
        let grid = populated_3x3_grid();
        // Current MB (1, 1), block 1 (i = 0).
        let cs = grid.predictor_candidates(1, 1, 0).unwrap();
        // MV1 = (2*1, 2*1 - 1) = (2, 1). MB row = 1, col = 0;
        // sub_row & 1 = 0, sub_col & 1 = 1 → block_idx = 1 (TR
        // of left MB, tag 10).
        assert_eq!(cs[0], Some(mv(10, 1)));
        // MV2 = (1, 2). MB row = 0, col = 1; sub_row & 1 = 1,
        // sub_col & 1 = 0 → block_idx = 2 (BL of above MB, tag 1).
        assert_eq!(cs[1], Some(mv(1, 2)));
        // MV3 = (1, 4). MB row = 0, col = 2; sub_row & 1 = 1,
        // sub_col & 1 = 0 → block_idx = 2 (BL of above-right MB,
        // tag 2).
        assert_eq!(cs[2], Some(mv(2, 2)));
    }

    #[test]
    fn block_1_picks_current_mb_block_0_above_mb_block_3_above_right_mb_block_2() {
        let grid = populated_3x3_grid();
        // Current MB (1, 1), block 2 (i = 1).
        let cs = grid.predictor_candidates(1, 1, 1).unwrap();
        // MV1 = (2, 2). MB row = 1, col = 1; sub_row & 1 = 0,
        // sub_col & 1 = 0 → block_idx = 0 (TL of current MB,
        // tag 11).
        assert_eq!(cs[0], Some(mv(11, 0)));
        // MV2 = (1, 3). MB row = 0, col = 1; sub_row & 1 = 1,
        // sub_col & 1 = 1 → block_idx = 3 (BR of above MB, tag 1).
        assert_eq!(cs[1], Some(mv(1, 3)));
        // MV3 = (1, 4) — same as block 0's MV3 (tag 2 BL of MB
        // (0, 2)).
        assert_eq!(cs[2], Some(mv(2, 2)));
    }

    #[test]
    fn block_2_picks_left_mb_block_3_current_mb_blocks_0_and_1() {
        let grid = populated_3x3_grid();
        // Current MB (1, 1), block 3 (i = 2).
        let cs = grid.predictor_candidates(1, 1, 2).unwrap();
        // MV1 = (3, 1). MB row = 1, col = 0; sub_row & 1 = 1,
        // sub_col & 1 = 1 → block_idx = 3 (BR of left MB, tag 10).
        assert_eq!(cs[0], Some(mv(10, 3)));
        // MV2 = (2, 2) = current MB's block 0 (tag 11).
        assert_eq!(cs[1], Some(mv(11, 0)));
        // MV3 = (2, 3) = current MB's block 1 (tag 11).
        assert_eq!(cs[2], Some(mv(11, 1)));
    }

    #[test]
    fn block_3_picks_current_mb_blocks_2_and_0_and_1() {
        let grid = populated_3x3_grid();
        // Current MB (1, 1), block 4 (i = 3).
        let cs = grid.predictor_candidates(1, 1, 3).unwrap();
        // MV1 = (3, 2) = current MB's block 2 (tag 11, idx 2).
        assert_eq!(cs[0], Some(mv(11, 2)));
        // MV2 = (2, 2) = current MB's block 0.
        assert_eq!(cs[1], Some(mv(11, 0)));
        // MV3 = (2, 3) = current MB's block 1.
        assert_eq!(cs[2], Some(mv(11, 1)));
    }

    // ---------------------------------------------------------------------
    // 1-MV neighbour MBs report the same MV across all four sub-block
    // queries inside them.
    // ---------------------------------------------------------------------

    #[test]
    fn one_mv_neighbour_replicates_across_all_sub_blocks() {
        let mut grid = MvGrid::new(3, 3);
        // Left + above + above-right neighbours all 1-MV.
        grid.record_one_mv(1, 0, mv(100, 200)).unwrap();
        grid.record_one_mv(0, 1, mv(300, 400)).unwrap();
        grid.record_one_mv(0, 2, mv(500, 600)).unwrap();
        // 1-MV current MB → query block 0 only (per the
        // §7.6.5 "top-left case used for the single MV of the
        // macroblock" rule).
        let cs = grid.predictor_candidates(1, 1, 0).unwrap();
        assert_eq!(
            cs,
            [Some(mv(100, 200)), Some(mv(300, 400)), Some(mv(500, 600))]
        );
    }

    // ---------------------------------------------------------------------
    // Absent / transparent neighbour mapping.
    // ---------------------------------------------------------------------

    #[test]
    fn absent_neighbour_yields_none_candidate() {
        let mut grid = MvGrid::new(3, 3);
        // Above neighbour absent (e.g. across a video-packet
        // boundary).
        grid.record_one_mv(1, 0, mv(5, 7)).unwrap();
        grid.record_absent(0, 1).unwrap();
        grid.record_one_mv(0, 2, mv(8, 9)).unwrap();
        let cs = grid.predictor_candidates(1, 1, 0).unwrap();
        assert_eq!(cs, [Some(mv(5, 7)), None, Some(mv(8, 9))]);
    }

    #[test]
    fn per_block_transparent_mask_yields_none() {
        // Above neighbour 4-MV but its BL sub-block (the MV2 cell
        // for block 1) is transparent.
        let mut grid = MvGrid::new(3, 3);
        grid.record(
            0,
            1,
            MbMvRecord {
                content: MbMv::FourMv([mv(1, 1), mv(2, 2), mv(3, 3), mv(4, 4)]),
                // Block 2 (BL, i = 2) is transparent.
                transparent: [false, false, true, false],
            },
        )
        .unwrap();
        grid.record_one_mv(1, 0, mv(9, 9)).unwrap();
        let cs = grid.predictor_candidates(1, 1, 0).unwrap();
        // MV2 lookup falls on the transparent BL sub-block → None.
        assert_eq!(cs[1], None);
        // MV1 still valid (left neighbour's TR sub-block from a
        // 1-MV record).
        assert_eq!(cs[0], Some(mv(9, 9)));
    }

    // ---------------------------------------------------------------------
    // End-to-end: feed the gathered candidates into the §7.6.5
    // median + the four substitution rules.
    // ---------------------------------------------------------------------

    #[test]
    fn end_to_end_median_with_three_valid_candidates() {
        // Match the spec's worked example: MV1=(-2,3), MV2=(1,5),
        // MV3=(-1,7) → predictor (-1, 5).
        let mut grid = MvGrid::new(3, 3);
        grid.record_one_mv(1, 0, mv(-2, 3)).unwrap();
        grid.record_one_mv(0, 1, mv(1, 5)).unwrap();
        grid.record_one_mv(0, 2, mv(-1, 7)).unwrap();
        let cs = grid.predictor_candidates(1, 1, 0).unwrap();
        let p = predict_motion_vector(cs);
        assert_eq!(p, mv(-1, 5));
    }

    #[test]
    fn end_to_end_one_invalid_candidate_becomes_zero() {
        let mut grid = MvGrid::new(3, 3);
        grid.record_one_mv(1, 0, mv(4, 4)).unwrap();
        grid.record_one_mv(0, 1, mv(6, 6)).unwrap();
        // Above-right neighbour absent → MV3 None → §7.6.5 rule 2
        // sets it to zero. Median over {(4,4), (6,6), (0,0)} →
        // (4, 4).
        grid.record_absent(0, 2).unwrap();
        let cs = grid.predictor_candidates(1, 1, 0).unwrap();
        let p = predict_motion_vector(cs);
        assert_eq!(p, mv(4, 4));
    }

    #[test]
    fn end_to_end_two_invalid_candidates_take_third() {
        let mut grid = MvGrid::new(3, 3);
        // Only the left neighbour is valid.
        grid.record_one_mv(1, 0, mv(7, 8)).unwrap();
        // Other two stay Absent (default state).
        let cs = grid.predictor_candidates(1, 1, 0).unwrap();
        let p = predict_motion_vector(cs);
        assert_eq!(p, mv(7, 8));
    }

    #[test]
    fn end_to_end_three_invalid_candidates_all_zero() {
        let grid = MvGrid::new(3, 3);
        // No neighbour recorded; block 0 of MB (1, 1) — all three
        // candidates None → predictor (0, 0).
        let cs = grid.predictor_candidates(1, 1, 0).unwrap();
        let p = predict_motion_vector(cs);
        assert_eq!(p, mv(0, 0));
    }

    // ---------------------------------------------------------------------
    // Right-edge VOP: MV3 (above-right) falls outside the grid even
    // when above-MB is present.
    // ---------------------------------------------------------------------

    #[test]
    fn right_edge_mb_loses_mv3_to_vop_boundary() {
        // Decoding MB (1, 2), block 0. MV3 sub-grid coordinate is
        // (1, 6) — beyond `2 * mb_cols = 6` → outside → None.
        let mut grid = MvGrid::new(3, 3);
        // Above MB (for current MB (1, 2)) at row 0 col 2.
        grid.record_one_mv(0, 2, mv(1, 1)).unwrap();
        // Left MB (relative to MB (1, 2)) at row 1 col 1.
        grid.record_one_mv(1, 1, mv(9, 9)).unwrap();
        let cs = grid.predictor_candidates(1, 2, 0).unwrap();
        assert_eq!(cs[2], None);
        assert_eq!(cs[0], Some(mv(9, 9)));
        assert_eq!(cs[1], Some(mv(1, 1)));
    }

    // ---------------------------------------------------------------------
    // The standalone `gather_mv_predictor_candidates` free function
    // is equivalent to the method.
    // ---------------------------------------------------------------------

    #[test]
    fn free_function_matches_method() {
        let grid = populated_3x3_grid();
        for r in 0..3 {
            for c in 0..3 {
                for i in 0..LUMA_BLOCKS_PER_MB {
                    let via_method = grid.predictor_candidates(r, c, i).unwrap();
                    let via_free = gather_mv_predictor_candidates(&grid, r, c, i).unwrap();
                    assert_eq!(via_method, via_free);
                }
            }
        }
    }
}

//! §7.4.3 / Figure 7-5 predictor candidate gathering for intra
//! macroblocks.
//!
//! The §7.4.3 spatial DC/AC predictor needs three neighbouring blocks
//! per block-to-decode `X`:
//!
//! ```text
//!     B  C       (above-left, above)
//!     A  X       (left, current)
//! ```
//!
//! Round 12 supplied the predictor *math* — [`crate::predictor`] —
//! given concrete `(FA, FB, FC)` and `(QpA, QpC)` inputs. Round 15
//! wired one isolated macroblock's blocks through the full §7.4.x chain
//! by passing [`crate::block::BlockPredictors::outside`] (every
//! neighbour treated as outside the VOP). This module owns the
//! intermediate step: given the already-decoded blocks of the
//! surrounding macroblocks (and the already-decoded earlier blocks of
//! the current macroblock), it resolves Figure 7-5's `A`, `B`, `C`
//! positions and builds the [`crate::block::BlockPredictors`] argument
//! that [`crate::block::decode_intra_block`] consumes.
//!
//! ## Block-grid layout (4:2:0)
//!
//! Each 4:2:0 macroblock contains six §6.2.7 blocks per Figure 6-8:
//!
//! ```text
//!     luma 0  luma 1         Cb 4         Cr 5
//!     luma 2  luma 3
//! ```
//!
//! Across a 2-D macroblock grid of `mb_rows × mb_cols` macroblocks,
//! the luma blocks form a `(2*mb_rows) × (2*mb_cols)` sub-grid and the
//! Cb / Cr blocks each form an `mb_rows × mb_cols` sub-grid.
//!
//! Given a block at component-grid position `(r, c)`:
//!
//! * `A` (left)  = `(r, c-1)`
//! * `B` (above-left) = `(r-1, c-1)`
//! * `C` (above) = `(r-1, c)`
//!
//! Any of `A` / `B` / `C` that falls outside its component sub-grid, or
//! belongs to a non-intra macroblock, triggers the §7.4.3.1 default:
//!
//! * DC value: `F[0][0] = 2^(bits_per_pixel + 2)`
//!   ([`crate::predictor::default_neighbour_dc`]).
//! * §7.4.3.3 AC prediction coefficients (first row for predictor `C`,
//!   first column for predictor `A`): taken as zero, which is encoded as
//!   `None` in [`crate::block::BlockPredictors`].
//!
//! ## What this round implements
//!
//! * [`BlockNeighbour`] — the per-block reconstructed state retained for
//!   the predictor (DC value, quantiser scale, optional first row /
//!   first column AC coefficients, and an "is-intra" flag).
//! * [`IntraBlockGrid`] — a `(2*mb_rows) × (2*mb_cols)` luma + two
//!   `mb_rows × mb_cols` chroma sub-grids of `Option<BlockNeighbour>`,
//!   with `None` meaning the macroblock at that position has not been
//!   decoded yet (or was non-intra).
//! * [`IntraBlockGrid::predictors_for`] — given an MB coordinate
//!   `(mb_row, mb_col)`, the §6.2.7 block index `i ∈ 0..6`, and the
//!   current block's `bits_per_pixel` + `quantiser_scale`, returns the
//!   [`BlockPredictors`] (the `A` / `B` / `C` resolved per Figure 7-5)
//!   that [`crate::block::decode_intra_block`] expects.
//! * [`IntraBlockGrid::record`] — store one decoded block back into the
//!   grid (so subsequent blocks in raster order pick it up as a
//!   neighbour).
//! * [`block_grid_position`] — the static mapping from `(mb_row,
//!   mb_col, i)` to component-grid coordinates, factored out so callers
//!   that walk the same grid can compute positions without touching the
//!   storage.
//!
//! ## Out of scope
//!
//! * Inter / B-VOP reconstruction. The predictor is only invoked for
//!   intra macroblocks; the grid stores `None` for non-intra positions
//!   and the predictor automatically falls back to the §7.4.3.1 default.
//! * Video-packet boundaries (§7.4.3.1 — neighbours across a video
//!   packet are treated as outside). This module's storage is per-VOP
//!   and the caller is responsible for clearing across-packet entries
//!   (or, equivalently, recording `None` at those positions).
//! * The §7.4.3.3 first row / first column extraction from a
//!   reconstructed block; the caller threads these in when recording a
//!   block (see [`BlockNeighbour::from_qf`]).
//! * 4:2:2 / 4:4:4 chroma layouts (Figures 6-9 / 6-10) and the
//!   non-rectangular SA-DCT modified scan.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition), read by
//! the agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §6.1.3 / Figure 6-8 — the 4:2:0 macroblock block layout (luma
//!   blocks 0, 1 / 2, 3; Cb block 4; Cr block 5).
//! * §7.4.3.1 page 242 — Figure 7-5 (the `A` / `B` / `C` neighbour
//!   layout) and the default-neighbour rule
//!   "If any of the blocks A, B or C are outside of the VOP boundary,
//!   or the video packet boundary, or they do not belong to an intra
//!   coded macroblock, their `F[0][0]` values are assumed to take a
//!   value of `2^(bits_per_pixel + 2)`."
//! * §7.4.3.3 page 243 — "If the prediction block (block 'A' or block
//!   'C') is outside of the boundary of the VOP or video packet, then
//!   all the prediction coefficients of that block are assumed to be
//!   zero."

use crate::block::BlockPredictors;
use crate::predictor::default_neighbour_dc;
use crate::texture::DcComponent;

/// One block's state retained in the [`IntraBlockGrid`] for the §7.4.3
/// predictor pass.
///
/// The §7.4.3.2 DC-coefficient prediction needs the neighbour's
/// inverse-quantised `F[0][0]`; the §7.4.3.3 AC prediction needs either
/// the neighbour's first row (`F[0][1..=7]`) or first column
/// (`F[1..=7][0]`) plus the neighbour's quantiser scale `Qp`. We store
/// the first row and first column of the *quantised* `QF` coefficients
/// directly so the §7.4.3.3 `(QFA[v][0] * QpA) // QpX` formula maps
/// straight onto the values without re-deriving them.
///
/// `is_intra` records whether this block belongs to an intra-coded
/// macroblock; the §7.4.3.1 default rule forces neighbours from
/// non-intra macroblocks back to `F[0][0] = 2^(bits_per_pixel + 2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockNeighbour {
    /// The inverse-quantised DC value `F[0][0]` of this block. Used by
    /// the §7.4.3.1 direction selection and the §7.4.3.2 DC predictor
    /// add.
    pub dc: i32,
    /// The quantiser scale `Qp` under which this block's AC coefficients
    /// were quantised. Used by the §7.4.3.3 `(QF[v][0] * Qp) // QpX`
    /// scaling.
    pub qp: u32,
    /// The block's quantised first row `QF[0][1..=7]`, used when this
    /// block is selected as the `C` (above) predictor.
    pub first_row: [i32; 7],
    /// The block's quantised first column `QF[1..=7][0]`, used when this
    /// block is selected as the `A` (left) predictor.
    pub first_column: [i32; 7],
    /// `true` if the macroblock that contains this block was intra-
    /// coded. §7.4.3.1: neighbours that are not intra-coded are forced
    /// to the default `F[0][0]` value (and AC prediction zeroed).
    pub is_intra: bool,
}

impl BlockNeighbour {
    /// Build a [`BlockNeighbour`] from the full quantised 8×8 `QF`
    /// block of an intra macroblock. The DC value `qf[0][0]` should be
    /// the *inverse-quantised* `F[0][0]` produced by §7.4.4.1.1
    /// ([`crate::inverse_quant::inverse_quant_intra_dc`]) — that is the
    /// value §7.4.3.1 compares with `|FA - FB|` etc.
    ///
    /// `qp` is the quantiser scale that the §7.4.4 inverse-quant pass
    /// used for this block's AC coefficients.
    pub fn from_qf(qf: &[[i32; 8]; 8], dc: i32, qp: u32) -> Self {
        let mut first_row = [0i32; 7];
        let mut first_column = [0i32; 7];
        for k in 0..7 {
            first_row[k] = qf[0][k + 1];
            first_column[k] = qf[k + 1][0];
        }
        Self {
            dc,
            qp,
            first_row,
            first_column,
            is_intra: true,
        }
    }
}

/// The Figure 6-8 sub-grid coordinate of a block within its component
/// plane.
///
/// For a macroblock at `(mb_row, mb_col)` and block index `i ∈ 0..6`:
///
/// * Luma (i = 0..=3): `(row, col) = (2*mb_row + top_bit,
///   2*mb_col + left_bit)`, where `top_bit` is 0 for i ∈ {0, 1} and 1
///   for i ∈ {2, 3}; `left_bit` is 0 for i ∈ {0, 2} and 1 for i ∈
///   {1, 3}.
/// * Cb (i = 4): `(mb_row, mb_col)`.
/// * Cr (i = 5): `(mb_row, mb_col)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGridPosition {
    /// The block's DC component (luminance / chrominance), per §6.2.7.
    pub component: DcComponent,
    /// For chroma blocks, which plane this block belongs to (Cb vs Cr).
    /// `None` for luma blocks.
    pub chroma_plane: Option<ChromaPlane>,
    /// Row index in the per-component sub-grid.
    pub row: usize,
    /// Column index in the per-component sub-grid.
    pub col: usize,
}

/// Which 4:2:0 chrominance plane a block belongs to (block 4 = Cb,
/// block 5 = Cr per Figure 6-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaPlane {
    /// Cb (Figure 6-8 block 4).
    Cb,
    /// Cr (Figure 6-8 block 5).
    Cr,
}

/// Map a `(mb_row, mb_col, i)` triple to its component sub-grid
/// position per Figure 6-8.
///
/// # Panics
///
/// Panics if `i >= 6`; only the 4:2:0 layout's 0..=5 block indices are
/// supported here.
pub fn block_grid_position(mb_row: usize, mb_col: usize, i: usize) -> BlockGridPosition {
    match i {
        0 => BlockGridPosition {
            component: DcComponent::Luminance,
            chroma_plane: None,
            row: 2 * mb_row,
            col: 2 * mb_col,
        },
        1 => BlockGridPosition {
            component: DcComponent::Luminance,
            chroma_plane: None,
            row: 2 * mb_row,
            col: 2 * mb_col + 1,
        },
        2 => BlockGridPosition {
            component: DcComponent::Luminance,
            chroma_plane: None,
            row: 2 * mb_row + 1,
            col: 2 * mb_col,
        },
        3 => BlockGridPosition {
            component: DcComponent::Luminance,
            chroma_plane: None,
            row: 2 * mb_row + 1,
            col: 2 * mb_col + 1,
        },
        4 => BlockGridPosition {
            component: DcComponent::Chrominance,
            chroma_plane: Some(ChromaPlane::Cb),
            row: mb_row,
            col: mb_col,
        },
        5 => BlockGridPosition {
            component: DcComponent::Chrominance,
            chroma_plane: Some(ChromaPlane::Cr),
            row: mb_row,
            col: mb_col,
        },
        _ => panic!("block_grid_position: i = {i} is not a 4:2:0 block index (0..6)"),
    }
}

/// The per-VOP storage of decoded intra blocks used by the §7.4.3
/// predictor.
///
/// Three 2-D `Vec<Option<BlockNeighbour>>` sub-grids — one for the
/// luma plane (`2 * mb_rows × 2 * mb_cols`) and one each for the Cb and
/// Cr planes (`mb_rows × mb_cols`). `None` entries are treated as
/// "outside the VOP / video packet" by the §7.4.3 predictor.
#[derive(Debug, Clone)]
pub struct IntraBlockGrid {
    mb_rows: usize,
    mb_cols: usize,
    luma: Vec<Option<BlockNeighbour>>,
    cb: Vec<Option<BlockNeighbour>>,
    cr: Vec<Option<BlockNeighbour>>,
}

impl IntraBlockGrid {
    /// Allocate an empty grid for a VOP of `mb_rows × mb_cols`
    /// macroblocks. Every cell starts `None` — i.e. the §7.4.3.1
    /// "outside the VOP" default applies until the cell is filled by
    /// [`record`][Self::record].
    pub fn new(mb_rows: usize, mb_cols: usize) -> Self {
        let luma_len = 2 * mb_rows * 2 * mb_cols;
        let chroma_len = mb_rows * mb_cols;
        Self {
            mb_rows,
            mb_cols,
            luma: vec![None; luma_len],
            cb: vec![None; chroma_len],
            cr: vec![None; chroma_len],
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

    fn luma_index(&self, row: usize, col: usize) -> usize {
        row * (2 * self.mb_cols) + col
    }

    fn chroma_index(&self, row: usize, col: usize) -> usize {
        row * self.mb_cols + col
    }

    fn get(&self, pos: BlockGridPosition) -> Option<BlockNeighbour> {
        match pos.component {
            DcComponent::Luminance => {
                if pos.row >= 2 * self.mb_rows || pos.col >= 2 * self.mb_cols {
                    return None;
                }
                self.luma[self.luma_index(pos.row, pos.col)]
            }
            DcComponent::Chrominance => {
                if pos.row >= self.mb_rows || pos.col >= self.mb_cols {
                    return None;
                }
                match pos.chroma_plane.expect("chroma block needs a plane") {
                    ChromaPlane::Cb => self.cb[self.chroma_index(pos.row, pos.col)],
                    ChromaPlane::Cr => self.cr[self.chroma_index(pos.row, pos.col)],
                }
            }
        }
    }

    /// Record one decoded block back into the grid at
    /// `(mb_row, mb_col, i)`. Subsequent calls to
    /// [`predictors_for`][Self::predictors_for] will see this block as
    /// the appropriate `A` / `B` / `C` neighbour for blocks decoded
    /// later in raster order.
    ///
    /// Pass `None` to mark a position as non-intra (or never decoded);
    /// the §7.4.3 predictor will treat it as "outside" per §7.4.3.1.
    pub fn record(
        &mut self,
        mb_row: usize,
        mb_col: usize,
        i: usize,
        neighbour: Option<BlockNeighbour>,
    ) {
        let pos = block_grid_position(mb_row, mb_col, i);
        match pos.component {
            DcComponent::Luminance => {
                assert!(pos.row < 2 * self.mb_rows && pos.col < 2 * self.mb_cols);
                let idx = self.luma_index(pos.row, pos.col);
                self.luma[idx] = neighbour;
            }
            DcComponent::Chrominance => {
                assert!(pos.row < self.mb_rows && pos.col < self.mb_cols);
                let idx = self.chroma_index(pos.row, pos.col);
                match pos.chroma_plane.expect("chroma block needs a plane") {
                    ChromaPlane::Cb => self.cb[idx] = neighbour,
                    ChromaPlane::Cr => self.cr[idx] = neighbour,
                }
            }
        }
    }

    /// Fetch the neighbour at position `(row, col)` of the same
    /// component sub-grid as `here`. Returns `None` when the position
    /// is outside the sub-grid OR is recorded `None`. The §7.4.3.1
    /// "non-intra macroblock" rule is folded in: if the neighbour was
    /// recorded with `is_intra == false`, this returns `None`.
    fn neighbour_at(
        &self,
        here: BlockGridPosition,
        row: isize,
        col: isize,
    ) -> Option<BlockNeighbour> {
        if row < 0 || col < 0 {
            return None;
        }
        let pos = BlockGridPosition {
            component: here.component,
            chroma_plane: here.chroma_plane,
            row: row as usize,
            col: col as usize,
        };
        match self.get(pos) {
            Some(nb) if nb.is_intra => Some(nb),
            _ => None,
        }
    }

    /// Build the [`BlockPredictors`] for block `(mb_row, mb_col, i)`,
    /// resolving the Figure 7-5 `A` (left), `B` (above-left), and `C`
    /// (above) neighbours against the recorded grid.
    ///
    /// `bits_per_pixel` is the §6.3.3 VOL field — the §7.4.3.1 default
    /// DC value for an unresolved neighbour is `2^(bpp + 2)`.
    /// `quantiser_scale` is the current block's `Qp` (used as a fallback
    /// `qp_a` / `qp_c` when the neighbour is unavailable; the
    /// §7.4.3.3 path is then short-circuited by the `None`
    /// `first_row` / `first_column` arrays).
    pub fn predictors_for(
        &self,
        mb_row: usize,
        mb_col: usize,
        i: usize,
        bits_per_pixel: u32,
        quantiser_scale: u32,
    ) -> BlockPredictors {
        let here = block_grid_position(mb_row, mb_col, i);
        let row = here.row as isize;
        let col = here.col as isize;
        let a = self.neighbour_at(here, row, col - 1);
        let b = self.neighbour_at(here, row - 1, col - 1);
        let c = self.neighbour_at(here, row - 1, col);

        let default_dc = default_neighbour_dc(bits_per_pixel);

        BlockPredictors {
            fa_dc: a.map(|n| n.dc).unwrap_or(default_dc),
            fb_dc: b.map(|n| n.dc).unwrap_or(default_dc),
            fc_dc: c.map(|n| n.dc).unwrap_or(default_dc),
            qp_a: a.map(|n| n.qp).unwrap_or(quantiser_scale),
            qp_c: c.map(|n| n.qp).unwrap_or(quantiser_scale),
            a_first_column: a.map(|n| n.first_column),
            c_first_row: c.map(|n| n.first_row),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- block_grid_position — Figure 6-8 layout -----

    #[test]
    fn luma_block_positions_within_one_mb() {
        // For a macroblock at (0, 0) the four luma blocks occupy a 2×2
        // sub-grid at (0..=1, 0..=1).
        let p0 = block_grid_position(0, 0, 0);
        assert_eq!(p0.component, DcComponent::Luminance);
        assert_eq!((p0.row, p0.col), (0, 0));
        let p1 = block_grid_position(0, 0, 1);
        assert_eq!((p1.row, p1.col), (0, 1));
        let p2 = block_grid_position(0, 0, 2);
        assert_eq!((p2.row, p2.col), (1, 0));
        let p3 = block_grid_position(0, 0, 3);
        assert_eq!((p3.row, p3.col), (1, 1));
    }

    #[test]
    fn luma_block_positions_in_neighbouring_mb() {
        // MB (1, 2) → luma sub-grid origin (2, 4).
        let p = block_grid_position(1, 2, 0);
        assert_eq!((p.row, p.col), (2, 4));
        let p = block_grid_position(1, 2, 3);
        assert_eq!((p.row, p.col), (3, 5));
    }

    #[test]
    fn chroma_block_positions_share_mb_coords() {
        // Cb and Cr both share the MB coordinate as their sub-grid
        // coordinate (one chroma block per MB in 4:2:0).
        let p4 = block_grid_position(3, 5, 4);
        assert_eq!(p4.component, DcComponent::Chrominance);
        assert_eq!(p4.chroma_plane, Some(ChromaPlane::Cb));
        assert_eq!((p4.row, p4.col), (3, 5));
        let p5 = block_grid_position(3, 5, 5);
        assert_eq!(p5.chroma_plane, Some(ChromaPlane::Cr));
        assert_eq!((p5.row, p5.col), (3, 5));
    }

    #[test]
    #[should_panic(expected = "not a 4:2:0 block index")]
    fn block_grid_position_rejects_oob_index() {
        let _ = block_grid_position(0, 0, 6);
    }

    // ----- IntraBlockGrid: empty grid → outside-of-VOP predictors -----

    #[test]
    fn empty_grid_yields_outside_predictors() {
        let grid = IntraBlockGrid::new(2, 2);
        let p = grid.predictors_for(0, 0, 0, 8, 8);
        let outside = BlockPredictors::outside(8, 8);
        // All three neighbours fall back to default DC, both AC arrays
        // are None, and qp_a/qp_c default to the current block's qs.
        assert_eq!(p, outside);
    }

    #[test]
    fn first_block_of_first_mb_has_no_neighbours() {
        // MB (0, 0), block 0 is at (0, 0) in the luma sub-grid; all
        // three of (row, col-1), (row-1, col-1), (row-1, col) are
        // negative -> None.
        let grid = IntraBlockGrid::new(4, 4);
        let p = grid.predictors_for(0, 0, 0, 8, 5);
        assert_eq!(p.fa_dc, default_neighbour_dc(8));
        assert_eq!(p.fb_dc, default_neighbour_dc(8));
        assert_eq!(p.fc_dc, default_neighbour_dc(8));
        assert_eq!(p.qp_a, 5);
        assert_eq!(p.qp_c, 5);
        assert!(p.a_first_column.is_none());
        assert!(p.c_first_row.is_none());
    }

    // ----- IntraBlockGrid: in-MB neighbours -----

    /// Build a sample intra block QF where every cell is the linear
    /// index, so first_row = [1, 2, 3, 4, 5, 6, 7] and
    /// first_column = [8, 16, 24, 32, 40, 48, 56].
    fn sample_qf(dc: i32) -> [[i32; 8]; 8] {
        let mut qf = [[0i32; 8]; 8];
        for (v, row) in qf.iter_mut().enumerate() {
            for (u, cell) in row.iter_mut().enumerate() {
                *cell = (v * 8 + u) as i32;
            }
        }
        qf[0][0] = dc;
        qf
    }

    #[test]
    fn block1_picks_block0_as_left_neighbour() {
        // Within MB (0, 0): block 1 is at luma (0, 1), so its A
        // neighbour is luma (0, 0) — block 0.
        let mut grid = IntraBlockGrid::new(2, 2);
        let qf0 = sample_qf(64);
        grid.record(0, 0, 0, Some(BlockNeighbour::from_qf(&qf0, 100, 7)));

        let p = grid.predictors_for(0, 0, 1, 8, 5);
        assert_eq!(p.fa_dc, 100);
        assert_eq!(p.qp_a, 7);
        // first_column is QF[1..=7][0]: 8, 16, 24, 32, 40, 48, 56.
        assert_eq!(p.a_first_column.unwrap(), [8, 16, 24, 32, 40, 48, 56]);
        // B and C are still outside (row -1).
        assert_eq!(p.fb_dc, default_neighbour_dc(8));
        assert_eq!(p.fc_dc, default_neighbour_dc(8));
        assert!(p.c_first_row.is_none());
    }

    #[test]
    fn block2_picks_block0_as_above_neighbour() {
        // Within MB (0, 0): block 2 is at luma (1, 0), so its C
        // neighbour is luma (0, 0) — block 0.
        let mut grid = IntraBlockGrid::new(2, 2);
        let qf0 = sample_qf(99);
        grid.record(0, 0, 0, Some(BlockNeighbour::from_qf(&qf0, 200, 9)));

        let p = grid.predictors_for(0, 0, 2, 8, 5);
        assert_eq!(p.fc_dc, 200);
        assert_eq!(p.qp_c, 9);
        // first_row is QF[0][1..=7]: 1, 2, 3, 4, 5, 6, 7.
        assert_eq!(p.c_first_row.unwrap(), [1, 2, 3, 4, 5, 6, 7]);
        // A and B are at col -1 / col -1, row -1 — both None.
        assert_eq!(p.fa_dc, default_neighbour_dc(8));
        assert_eq!(p.fb_dc, default_neighbour_dc(8));
        assert!(p.a_first_column.is_none());
    }

    #[test]
    fn block3_sees_all_three_inmb_neighbours() {
        // Within MB (0, 0): block 3 is at luma (1, 1).
        //   A (left)       = (1, 0) = block 2.
        //   B (above-left) = (0, 0) = block 0.
        //   C (above)      = (0, 1) = block 1.
        let mut grid = IntraBlockGrid::new(2, 2);
        grid.record(
            0,
            0,
            0,
            Some(BlockNeighbour::from_qf(&sample_qf(10), 10, 5)),
        );
        grid.record(
            0,
            0,
            1,
            Some(BlockNeighbour::from_qf(&sample_qf(20), 20, 5)),
        );
        grid.record(
            0,
            0,
            2,
            Some(BlockNeighbour::from_qf(&sample_qf(30), 30, 5)),
        );

        let p = grid.predictors_for(0, 0, 3, 8, 5);
        assert_eq!(p.fa_dc, 30); // block 2 -> A
        assert_eq!(p.fb_dc, 10); // block 0 -> B
        assert_eq!(p.fc_dc, 20); // block 1 -> C
        assert_eq!(p.qp_a, 5);
        assert_eq!(p.qp_c, 5);
        assert!(p.a_first_column.is_some());
        assert!(p.c_first_row.is_some());
    }

    // ----- IntraBlockGrid: cross-MB neighbours -----

    #[test]
    fn block0_of_second_mb_picks_block1_of_left_mb_as_a() {
        // MB (0, 1) block 0 is at luma (0, 2). Its A (left) is
        // luma (0, 1) — block 1 of MB (0, 0).
        let mut grid = IntraBlockGrid::new(1, 2);
        grid.record(
            0,
            0,
            1,
            Some(BlockNeighbour::from_qf(&sample_qf(77), 77, 4)),
        );
        let p = grid.predictors_for(0, 1, 0, 8, 5);
        assert_eq!(p.fa_dc, 77);
        assert_eq!(p.qp_a, 4);
        // B at (-1, 1), C at (-1, 2): None.
        assert_eq!(p.fb_dc, default_neighbour_dc(8));
        assert_eq!(p.fc_dc, default_neighbour_dc(8));
    }

    #[test]
    fn block0_of_second_mb_row_picks_block2_of_above_mb_as_c() {
        // MB (1, 0) block 0 is at luma (2, 0). Its C (above) is
        // luma (1, 0) — block 2 of MB (0, 0).
        let mut grid = IntraBlockGrid::new(2, 1);
        grid.record(
            0,
            0,
            2,
            Some(BlockNeighbour::from_qf(&sample_qf(55), 55, 6)),
        );
        let p = grid.predictors_for(1, 0, 0, 8, 5);
        assert_eq!(p.fc_dc, 55);
        assert_eq!(p.qp_c, 6);
        // A at (2, -1), B at (1, -1): None.
        assert_eq!(p.fa_dc, default_neighbour_dc(8));
        assert_eq!(p.fb_dc, default_neighbour_dc(8));
    }

    #[test]
    fn block0_of_inner_mb_picks_block3_of_diagonal_mb_as_b() {
        // MB (1, 1) block 0 is at luma (2, 2). The Figure 7-5 layout:
        //   A (left)       = (2, 1) = block 1 of MB (1, 0).
        //   B (above-left) = (1, 1) = block 3 of MB (0, 0).
        //   C (above)      = (1, 2) = block 2 of MB (0, 1).
        let mut grid = IntraBlockGrid::new(2, 2);
        grid.record(
            0,
            0,
            3,
            Some(BlockNeighbour::from_qf(&sample_qf(111), 111, 1)),
        );
        grid.record(
            0,
            1,
            2,
            Some(BlockNeighbour::from_qf(&sample_qf(222), 222, 2)),
        );
        grid.record(
            1,
            0,
            1,
            Some(BlockNeighbour::from_qf(&sample_qf(444), 444, 3)),
        );

        let p = grid.predictors_for(1, 1, 0, 8, 5);
        assert_eq!(p.fa_dc, 444);
        assert_eq!(p.fb_dc, 111);
        assert_eq!(p.fc_dc, 222);
        assert_eq!(p.qp_a, 3);
        assert_eq!(p.qp_c, 2);
    }

    // ----- chroma sub-grids are independent of luma -----

    #[test]
    fn chroma_neighbours_use_chroma_subgrid() {
        // MB (0, 1) block 4 is Cb at (0, 1). Its A neighbour is
        // Cb at (0, 0) — MB (0, 0) block 4.
        let mut grid = IntraBlockGrid::new(1, 2);
        let mut qf = sample_qf(150);
        // Distinct first_row / first_column so we can tell it apart
        // from a luma block.
        qf[0][1] = -1;
        qf[1][0] = -2;
        grid.record(0, 0, 4, Some(BlockNeighbour::from_qf(&qf, 150, 8)));
        // Cr at the same MB is *not* a neighbour of Cb.
        grid.record(
            0,
            0,
            5,
            Some(BlockNeighbour::from_qf(&sample_qf(999), 999, 9)),
        );

        let p = grid.predictors_for(0, 1, 4, 8, 5);
        assert_eq!(p.fa_dc, 150);
        assert_eq!(p.qp_a, 8);
        // first_column[0] == QF[1][0] == -2.
        assert_eq!(p.a_first_column.unwrap()[0], -2);
        // B and C are above the chroma sub-grid -> None.
        assert_eq!(p.fb_dc, default_neighbour_dc(8));
        assert_eq!(p.fc_dc, default_neighbour_dc(8));
        // The luma sub-grid (and the Cr sub-grid) must not leak into the
        // Cb predictor lookup.
    }

    #[test]
    fn cb_and_cr_sub_grids_are_isolated() {
        // Block 5 (Cr) of MB (0, 1) should NOT see block 4 (Cb) of
        // MB (0, 0) as a neighbour.
        let mut grid = IntraBlockGrid::new(1, 2);
        grid.record(
            0,
            0,
            4,
            Some(BlockNeighbour::from_qf(&sample_qf(100), 100, 3)),
        );
        let p = grid.predictors_for(0, 1, 5, 8, 5);
        // A would be Cr at (0, 0) — never recorded -> None -> default.
        assert_eq!(p.fa_dc, default_neighbour_dc(8));
        assert!(p.a_first_column.is_none());
    }

    // ----- non-intra neighbour → §7.4.3.1 default fallback -----

    #[test]
    fn non_intra_neighbour_treated_as_outside() {
        // Block 1 of MB (0, 0): A is block 0 of the same MB. Record
        // block 0 as a non-intra block (is_intra = false) and confirm
        // the predictor falls back to default.
        let mut grid = IntraBlockGrid::new(1, 1);
        grid.record(
            0,
            0,
            0,
            Some(BlockNeighbour {
                dc: 999,
                qp: 9,
                first_row: [99; 7],
                first_column: [99; 7],
                is_intra: false,
            }),
        );
        let p = grid.predictors_for(0, 0, 1, 8, 5);
        // §7.4.3.1: non-intra neighbour -> F[0][0] = 2^(bpp + 2).
        assert_eq!(p.fa_dc, default_neighbour_dc(8));
        // §7.4.3.3: AC predictor coefficients zero -> None.
        assert!(p.a_first_column.is_none());
        // qp_a falls back to current block's quantiser scale.
        assert_eq!(p.qp_a, 5);
    }

    #[test]
    fn explicit_none_record_is_outside() {
        // Record None explicitly (e.g. across a video-packet boundary):
        // the predictor sees outside.
        let mut grid = IntraBlockGrid::new(1, 1);
        grid.record(0, 0, 0, None);
        let p = grid.predictors_for(0, 0, 1, 8, 5);
        assert_eq!(p.fa_dc, default_neighbour_dc(8));
    }

    // ----- BlockNeighbour::from_qf -----

    #[test]
    fn from_qf_extracts_first_row_and_column() {
        let mut qf = [[0i32; 8]; 8];
        for (v, row) in qf.iter_mut().enumerate() {
            for (u, cell) in row.iter_mut().enumerate() {
                *cell = (v * 100 + u) as i32;
            }
        }
        let nb = BlockNeighbour::from_qf(&qf, 12345, 7);
        assert_eq!(nb.dc, 12345);
        assert_eq!(nb.qp, 7);
        assert!(nb.is_intra);
        // first_row = QF[0][1..=7]: 1, 2, 3, 4, 5, 6, 7.
        assert_eq!(nb.first_row, [1, 2, 3, 4, 5, 6, 7]);
        // first_column = QF[1..=7][0]: 100, 200, 300, 400, 500, 600, 700.
        assert_eq!(nb.first_column, [100, 200, 300, 400, 500, 600, 700]);
    }

    // ----- integration: full MB walk fills the grid sensibly -----

    /// Walk a 2×2 MB grid in raster order and record a deterministic QF
    /// for every block; then re-walk it and verify the predictor seen
    /// at each block matches what we expect from the Figure 7-5 layout.
    #[test]
    fn raster_walk_records_and_reads_back_consistently() {
        // Encoding scheme: dc = mb_row * 100 + mb_col * 10 + i.
        fn dc_of(mb_row: usize, mb_col: usize, i: usize) -> i32 {
            (mb_row * 100 + mb_col * 10 + i) as i32
        }
        let mut grid = IntraBlockGrid::new(2, 2);
        // Record every block.
        for mb_row in 0..2 {
            for mb_col in 0..2 {
                for i in 0..6 {
                    let dc = dc_of(mb_row, mb_col, i);
                    let qf = sample_qf(dc);
                    grid.record(mb_row, mb_col, i, Some(BlockNeighbour::from_qf(&qf, dc, 5)));
                }
            }
        }
        // Now query block 3 of MB (1, 1): its neighbours are blocks 2,
        // 0, 1 of MB (1, 1) — see block3_sees_all_three_inmb_neighbours
        // for the layout proof.
        let p = grid.predictors_for(1, 1, 3, 8, 5);
        assert_eq!(p.fa_dc, dc_of(1, 1, 2)); // block 2 of MB (1, 1)
        assert_eq!(p.fb_dc, dc_of(1, 1, 0)); // block 0 of MB (1, 1)
        assert_eq!(p.fc_dc, dc_of(1, 1, 1)); // block 1 of MB (1, 1)

        // And block 0 of MB (1, 1): A = block 1 of MB (1, 0),
        // B = block 3 of MB (0, 0), C = block 2 of MB (0, 1).
        let p = grid.predictors_for(1, 1, 0, 8, 5);
        assert_eq!(p.fa_dc, dc_of(1, 0, 1));
        assert_eq!(p.fb_dc, dc_of(0, 0, 3));
        assert_eq!(p.fc_dc, dc_of(0, 1, 2));
    }

    // ----- chroma neighbours across MBs -----

    #[test]
    fn cr_neighbour_at_mb_diagonal() {
        // Cr at MB (1, 1): row (1, 1). B = (0, 0) = Cr of MB (0, 0).
        let mut grid = IntraBlockGrid::new(2, 2);
        grid.record(
            0,
            0,
            5,
            Some(BlockNeighbour::from_qf(&sample_qf(77), 77, 7)),
        );
        let p = grid.predictors_for(1, 1, 5, 8, 5);
        assert_eq!(p.fb_dc, 77);
    }
}

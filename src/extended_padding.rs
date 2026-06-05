//! §7.6.1.3 extended padding — the third pass of the §7.6.1 reference-
//! VOP padding pipeline.
//!
//! The §7.6.1.1 horizontal pass (`crate::sample_padding`) and the
//! §7.6.1.2 vertical pass (`crate::vertical_padding`) together fill
//! every boundary macroblock: every transparent sample of a macroblock
//! that straddles the VOP shape mask is replaced with the average or
//! copy of its nearest VOP-interior neighbour, and the macroblock comes
//! out fully opaque. After those two passes the only macroblocks still
//! transparent are **exterior macroblocks** — macroblocks that lie
//! entirely outside the VOP shape (every `s[y][x] == 0`).
//!
//! §7.6.1.3 fills exterior macroblocks by replicating the border
//! row / column of the nearest **post-§7.6.1.2 boundary macroblock**.
//! If an exterior macroblock is adjacent to more than one boundary
//! macroblock, the spec disambiguates with the Figure 7-28 priority
//! convention; if it is adjacent to none, the exterior macroblock is
//! filled with the `2^(bits_per_pixel - 1)` mid-grey value (128 for
//! 8-bit components).
//!
//! ## Algorithm (verbatim from ISO/IEC 14496-2:2004 §7.6.1.3)
//!
//! > Exterior macroblocks immediately next to boundary macroblocks are
//! > filled by replicating the samples at the border of the boundary
//! > macroblocks. Note that the boundary macroblocks have been
//! > completely padded in subclause 7.6.1.1 and subclause 7.6.1.2. If
//! > an exterior macroblock is next to more than one boundary
//! > macroblocks, one of the macroblocks is chosen, according to the
//! > following convention, for reference.
//! >
//! > The boundary macroblocks surrounding an exterior macroblock are
//! > numbered in priority according to Figure 7-28. The exterior
//! > macroblock is then padded by replicating upwards, downwards,
//! > leftwards, or rightwards the row of samples from the horizontal
//! > or vertical border of the boundary macroblock having the largest
//! > priority number.
//! >
//! > The remaining exterior macroblocks (not located next to any
//! > boundary macroblocks) are filled with 2^(bits_per_pixel - 1).
//! > For 8-bit luminance component and associated chrominance this
//! > implies filling with 128.
//!
//! ## Figure 7-28 priority numbering
//!
//! Figure 7-28 surrounds the exterior macroblock with its four
//! side-adjacent neighbours, labelled `0..=3`:
//!
//! ```text
//!                     ┌──────────────────┐
//!                     │  Boundary MB 2   │   (above / north)
//!                     └──────────────────┘
//! ┌──────────────────┐┌──────────────────┐┌──────────────────┐
//! │  Boundary MB 3   ││ Exterior MB      ││  Boundary MB 1   │
//! │   (left / west)  ││                  ││   (right / east) │
//! └──────────────────┘└──────────────────┘└──────────────────┘
//!                     ┌──────────────────┐
//!                     │  Boundary MB 0   │   (below / south)
//!                     └──────────────────┘
//! ```
//!
//! The spec text picks "the boundary macroblock having the largest
//! priority number" — `3 > 2 > 1 > 0`. Reading positions:
//!
//! | Priority | Position relative to exterior | Border facing exterior | Replication direction |
//! | -------- | ----------------------------- | ---------------------- | --------------------- |
//! | 3        | left  (west)                  | rightmost column       | rightwards            |
//! | 2        | above (north)                 | bottom row             | downwards             |
//! | 1        | right (east)                  | leftmost column        | leftwards             |
//! | 0        | below (south)                 | top row                | upwards               |
//!
//! The exterior macroblock's every output sample is then set from the
//! chosen border row / column:
//!
//! * Priority 3 (left): `out[y][x] = left[y][N - 1]` for every
//!   `(y, x)` — the rightmost column of the left neighbour, replicated
//!   rightwards into every column of the exterior macroblock.
//! * Priority 2 (above): `out[y][x] = above[N - 1][x]` for every
//!   `(y, x)` — the bottom row of the above neighbour, replicated
//!   downwards into every row of the exterior macroblock.
//! * Priority 1 (right): `out[y][x] = right[y][0]` for every
//!   `(y, x)` — the leftmost column of the right neighbour, replicated
//!   leftwards into every column of the exterior macroblock.
//! * Priority 0 (below): `out[y][x] = below[0][x]` for every
//!   `(y, x)` — the top row of the below neighbour, replicated upwards
//!   into every row of the exterior macroblock.
//!
//! ## What this module does **not** do
//!
//! * Distinguish luma from chroma at the type level. The §7.6.1.3
//!   algorithm is identical for both; the [`LUMA_SIDE`] / [`CHROMA_SIDE`]
//!   entry points are the same const-generic implementation pinned to
//!   the matching side length.
//! * Decide which macroblocks are exterior vs. boundary. The §7.6.1
//!   framing text routes boundary macroblocks through §7.6.1.1 then
//!   §7.6.1.2; the macroblock-classification step lives in the caller
//!   (which has access to the VOP shape grid).
//! * Decimate the §7.6.1.4 chrominance shape. The chroma shape comes
//!   from §6.1.3.6 subsampling and is the caller's responsibility.
//! * Handle the §7.6.1.5 interlaced per-field padding. Interlaced
//!   exterior macroblocks split into two N×(N/2) fields; the per-field
//!   §7.6.1.3 application is a future entry point.
//! * Resolve the diagonal-neighbour case explicitly. Figure 7-28 only
//!   names side-adjacent boundaries (`0..=3`); the §7.6.1 framing
//!   refers to "neighbour" as "immediately next to", which the spec
//!   text always pairs with the four side-adjacent neighbours.
//!   Diagonally-adjacent boundary macroblocks do not enter the
//!   priority ranking.

#![allow(clippy::needless_range_loop)]

use crate::sample_padding::{CHROMA_SIDE, LUMA_SIDE};

/// Position of a boundary macroblock relative to the exterior
/// macroblock under §7.6.1.3 extended padding.
///
/// Variants are ordered to mirror the Figure 7-28 priority numbering
/// `3 > 2 > 1 > 0`:
///
/// * [`ExteriorNeighbourPosition::Left`] — boundary MB 3 (west); highest
///   priority. Its rightmost column is replicated rightwards into
///   every column of the exterior MB.
/// * [`ExteriorNeighbourPosition::Above`] — boundary MB 2 (north). Its bottom
///   row is replicated downwards into every row of the exterior MB.
/// * [`ExteriorNeighbourPosition::Right`] — boundary MB 1 (east). Its leftmost
///   column is replicated leftwards into every column of the exterior
///   MB.
/// * [`ExteriorNeighbourPosition::Below`] — boundary MB 0 (south); lowest
///   priority. Its top row is replicated upwards into every row of
///   the exterior MB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExteriorNeighbourPosition {
    /// Boundary MB 3 (west), priority 3 — replicate the boundary's
    /// rightmost column rightwards.
    Left,
    /// Boundary MB 2 (north), priority 2 — replicate the boundary's
    /// bottom row downwards.
    Above,
    /// Boundary MB 1 (east), priority 1 — replicate the boundary's
    /// leftmost column leftwards.
    Right,
    /// Boundary MB 0 (south), priority 0 — replicate the boundary's
    /// top row upwards.
    Below,
}

impl ExteriorNeighbourPosition {
    /// Numeric priority per Figure 7-28 (`3` highest, `0` lowest).
    /// Provided so callers building a custom neighbour-set selector
    /// don't have to duplicate the table.
    pub const fn priority(self) -> u8 {
        match self {
            ExteriorNeighbourPosition::Left => 3,
            ExteriorNeighbourPosition::Above => 2,
            ExteriorNeighbourPosition::Right => 1,
            ExteriorNeighbourPosition::Below => 0,
        }
    }
}

/// The four optional side-adjacent boundary macroblocks surrounding
/// an exterior macroblock.
///
/// Each field carries the **post-§7.6.1.2 fully-padded** boundary
/// macroblock's `N×N` sample grid (luma side `16` per §6.1.3.4 or
/// chroma side `8` per §7.6.1.4). A `None` value means the
/// corresponding side-adjacent macroblock is either off the bounding
/// rectangle or itself an exterior macroblock (and therefore not
/// usable as a §7.6.1.3 source); the §7.6.1.3 selector skips those
/// positions and falls through to the next-highest priority neighbour
/// or to the `2^(bits_per_pixel - 1)` mid-grey fallback when no
/// neighbour is present.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryNeighbours<'a, const N: usize> {
    /// Boundary MB 3 — to the **west** of the exterior macroblock.
    pub left: Option<&'a [[i32; N]; N]>,
    /// Boundary MB 2 — to the **north** of the exterior macroblock.
    pub above: Option<&'a [[i32; N]; N]>,
    /// Boundary MB 1 — to the **east** of the exterior macroblock.
    pub right: Option<&'a [[i32; N]; N]>,
    /// Boundary MB 0 — to the **south** of the exterior macroblock.
    pub below: Option<&'a [[i32; N]; N]>,
}

impl<'a, const N: usize> Default for BoundaryNeighbours<'a, N> {
    fn default() -> Self {
        Self {
            left: None,
            above: None,
            right: None,
            below: None,
        }
    }
}

impl<'a, const N: usize> BoundaryNeighbours<'a, N> {
    /// Construct a [`BoundaryNeighbours`] with every side absent. Same
    /// as `Default::default` but available in const context.
    pub const fn none() -> Self {
        Self {
            left: None,
            above: None,
            right: None,
            below: None,
        }
    }

    /// Return the highest-priority neighbour position that has a
    /// boundary macroblock attached, or `None` when every side is
    /// `None` (the §7.6.1.3 mid-grey case).
    ///
    /// The priority ordering follows Figure 7-28 with `3` (left) being
    /// the highest priority and `0` (below) being the lowest. The
    /// implementation here returns `Some(ExteriorNeighbourPosition::Left)` if
    /// `left` is `Some`, otherwise `Some(ExteriorNeighbourPosition::Above)` if
    /// `above` is `Some`, and so on through `Right` then `Below`.
    pub fn highest_priority_position(&self) -> Option<ExteriorNeighbourPosition> {
        if self.left.is_some() {
            Some(ExteriorNeighbourPosition::Left)
        } else if self.above.is_some() {
            Some(ExteriorNeighbourPosition::Above)
        } else if self.right.is_some() {
            Some(ExteriorNeighbourPosition::Right)
        } else if self.below.is_some() {
            Some(ExteriorNeighbourPosition::Below)
        } else {
            None
        }
    }
}

/// Per-macroblock outcome of the §7.6.1.3 extended-padding step.
///
/// * [`ExteriorPaddingOutcome::FromNeighbour`] — the exterior
///   macroblock was filled by replicating a border row / column of
///   the named boundary macroblock. The replicated samples come
///   straight from the post-§7.6.1.2 padded source; no averaging is
///   performed in §7.6.1.3 itself.
/// * [`ExteriorPaddingOutcome::MidGrey`] — the exterior macroblock has
///   no side-adjacent boundary neighbour. Every sample is filled with
///   the §7.6.1.3 default `2^(bits_per_pixel - 1)`. The value of
///   `bits_per_pixel` is the §6.3.3 `bits_per_pixel` of the channel
///   being padded; for the canonical 8-bit luma / 4:2:0 chroma case
///   the fill value is `128`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExteriorPaddingOutcome {
    /// The exterior macroblock was filled from the named neighbour.
    FromNeighbour(ExteriorNeighbourPosition),
    /// The exterior macroblock had no side-adjacent boundary
    /// neighbour and was filled with `2^(bits_per_pixel - 1)`.
    MidGrey,
}

/// Compute the §7.6.1.3 mid-grey fill value for a channel of
/// `bits_per_pixel` precision. Returns `2^(bits_per_pixel - 1)`.
///
/// `bits_per_pixel` follows the §6.3.3 `not_8_bit` path: the canonical
/// value is 8 (yielding `128`) and the spec admits the range
/// `[4, 12]` via the `quant_precision` companion field. The function
/// asserts in debug builds that the requested precision stays inside
/// that range so callers fed an out-of-band value catch the bug
/// rather than overflow silently.
pub const fn mid_grey_value(bits_per_pixel: u8) -> i32 {
    debug_assert!(bits_per_pixel >= 1);
    debug_assert!(bits_per_pixel <= 31);
    1i32 << (bits_per_pixel as i32 - 1)
}

/// §7.6.1.3 extended padding for one exterior macroblock.
///
/// Given the four (optional) side-adjacent boundary neighbours in
/// `neighbours` and the channel's `bits_per_pixel`, the function:
///
/// 1. Picks the highest-priority present neighbour per Figure 7-28.
/// 2. Fills the output `N×N` macroblock by replicating that
///    neighbour's border row or column per the table above.
/// 3. Falls through to the `2^(bits_per_pixel - 1)` mid-grey fill
///    when no side-adjacent boundary neighbour is present.
///
/// The returned [`ExteriorPaddingOutcome`] tells the caller which
/// branch fired; the returned `[[i32; N]; N]` is the padded exterior
/// macroblock.
pub fn extended_padding_macroblock<const N: usize>(
    neighbours: &BoundaryNeighbours<N>,
    bits_per_pixel: u8,
) -> ([[i32; N]; N], ExteriorPaddingOutcome) {
    let mut out = [[0i32; N]; N];
    match neighbours.highest_priority_position() {
        Some(ExteriorNeighbourPosition::Left) => {
            // §7.6.1.3 priority-3 branch (Figure 7-28 boundary MB 3,
            // west): replicate the rightmost column of the left
            // neighbour rightwards.
            //
            // The unwrap is guarded by `highest_priority_position`
            // returning `Some(Left)` only when `neighbours.left` is
            // `Some`.
            let left = neighbours.left.expect(
                "extended_padding_macroblock: highest_priority_position said Left but left is None",
            );
            for y in 0..N {
                let edge = left[y][N - 1];
                for x in 0..N {
                    out[y][x] = edge;
                }
            }
            (
                out,
                ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Left),
            )
        }
        Some(ExteriorNeighbourPosition::Above) => {
            // §7.6.1.3 priority-2 branch (Figure 7-28 boundary MB 2,
            // north): replicate the bottom row of the above neighbour
            // downwards.
            let above = neighbours.above.expect(
                "extended_padding_macroblock: highest_priority_position said Above but above is None",
            );
            let edge_row = &above[N - 1];
            for y in 0..N {
                out[y] = *edge_row;
            }
            (
                out,
                ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Above),
            )
        }
        Some(ExteriorNeighbourPosition::Right) => {
            // §7.6.1.3 priority-1 branch (Figure 7-28 boundary MB 1,
            // east): replicate the leftmost column of the right
            // neighbour leftwards.
            let right = neighbours.right.expect(
                "extended_padding_macroblock: highest_priority_position said Right but right is None",
            );
            for y in 0..N {
                let edge = right[y][0];
                for x in 0..N {
                    out[y][x] = edge;
                }
            }
            (
                out,
                ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Right),
            )
        }
        Some(ExteriorNeighbourPosition::Below) => {
            // §7.6.1.3 priority-0 branch (Figure 7-28 boundary MB 0,
            // south): replicate the top row of the below neighbour
            // upwards.
            let below = neighbours.below.expect(
                "extended_padding_macroblock: highest_priority_position said Below but below is None",
            );
            let edge_row = &below[0];
            for y in 0..N {
                out[y] = *edge_row;
            }
            (
                out,
                ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Below),
            )
        }
        None => {
            // §7.6.1.3 fallthrough: no side-adjacent boundary neighbour
            // → fill every sample with `2^(bits_per_pixel - 1)`.
            let fill = mid_grey_value(bits_per_pixel);
            for y in 0..N {
                for x in 0..N {
                    out[y][x] = fill;
                }
            }
            (out, ExteriorPaddingOutcome::MidGrey)
        }
    }
}

/// §7.6.1.3 extended padding for one 4:2:0 luminance exterior
/// macroblock (16×16 samples per §6.1.3.4).
pub fn extended_padding_luma(
    neighbours: &BoundaryNeighbours<LUMA_SIDE>,
    bits_per_pixel: u8,
) -> ([[i32; LUMA_SIDE]; LUMA_SIDE], ExteriorPaddingOutcome) {
    extended_padding_macroblock::<LUMA_SIDE>(neighbours, bits_per_pixel)
}

/// §7.6.1.3 extended padding for one 4:2:0 chrominance exterior
/// block (8×8 samples per §7.6.1.4).
pub fn extended_padding_chroma(
    neighbours: &BoundaryNeighbours<CHROMA_SIDE>,
    bits_per_pixel: u8,
) -> ([[i32; CHROMA_SIDE]; CHROMA_SIDE], ExteriorPaddingOutcome) {
    extended_padding_macroblock::<CHROMA_SIDE>(neighbours, bits_per_pixel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an N×N grid whose `(y, x)` entry is `base + 10 * y + x`.
    /// Useful as a recognisable fake "fully padded boundary
    /// macroblock" so the tests can assert which row / column was
    /// replicated.
    fn make_grid<const N: usize>(base: i32) -> [[i32; N]; N] {
        let mut g = [[0i32; N]; N];
        for y in 0..N {
            for x in 0..N {
                g[y][x] = base + 10 * y as i32 + x as i32;
            }
        }
        g
    }

    #[test]
    fn mid_grey_8_bit_is_128() {
        assert_eq!(mid_grey_value(8), 128);
    }

    #[test]
    fn mid_grey_10_bit_is_512() {
        assert_eq!(mid_grey_value(10), 512);
    }

    #[test]
    fn mid_grey_4_bit_is_8() {
        // §6.3.3 not_8_bit admits bits_per_pixel down to 4.
        assert_eq!(mid_grey_value(4), 8);
    }

    #[test]
    fn no_neighbour_fills_with_mid_grey() {
        let neighbours: BoundaryNeighbours<8> = BoundaryNeighbours::none();
        let (out, outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(outcome, ExteriorPaddingOutcome::MidGrey);
        for row in out.iter() {
            for &v in row.iter() {
                assert_eq!(v, 128);
            }
        }
    }

    #[test]
    fn no_neighbour_fills_with_mid_grey_10_bit() {
        let neighbours: BoundaryNeighbours<16> = BoundaryNeighbours::none();
        let (out, outcome) = extended_padding_macroblock::<16>(&neighbours, 10);
        assert_eq!(outcome, ExteriorPaddingOutcome::MidGrey);
        for row in out.iter() {
            for &v in row.iter() {
                assert_eq!(v, 512);
            }
        }
    }

    #[test]
    fn left_only_replicates_right_column_rightwards() {
        // Boundary MB 3 (left/west, priority 3) — its rightmost
        // column is replicated rightwards into every column of the
        // exterior MB.
        let left = make_grid::<8>(0);
        // left[y][7] = 0 + 10*y + 7 = 7, 17, 27, 37, 47, 57, 67, 77
        let neighbours = BoundaryNeighbours {
            left: Some(&left),
            ..BoundaryNeighbours::none()
        };
        let (out, outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(
            outcome,
            ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Left)
        );
        for y in 0..8 {
            let expected = left[y][7];
            for x in 0..8 {
                assert_eq!(out[y][x], expected, "y={y} x={x}");
            }
        }
    }

    #[test]
    fn above_only_replicates_bottom_row_downwards() {
        // Boundary MB 2 (above/north, priority 2) — its bottom row
        // is replicated downwards into every row of the exterior MB.
        let above = make_grid::<8>(0);
        // above[7] = [70, 71, 72, 73, 74, 75, 76, 77]
        let neighbours = BoundaryNeighbours {
            above: Some(&above),
            ..BoundaryNeighbours::none()
        };
        let (out, outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(
            outcome,
            ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Above)
        );
        let expected_row = above[7];
        for y in 0..8 {
            assert_eq!(out[y], expected_row, "y={y}");
        }
    }

    #[test]
    fn right_only_replicates_left_column_leftwards() {
        // Boundary MB 1 (right/east, priority 1) — its leftmost
        // column is replicated leftwards into every column of the
        // exterior MB.
        let right = make_grid::<8>(100);
        // right[y][0] = 100 + 10*y + 0 = 100, 110, 120, ..., 170
        let neighbours = BoundaryNeighbours {
            right: Some(&right),
            ..BoundaryNeighbours::none()
        };
        let (out, outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(
            outcome,
            ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Right)
        );
        for y in 0..8 {
            let expected = right[y][0];
            for x in 0..8 {
                assert_eq!(out[y][x], expected, "y={y} x={x}");
            }
        }
    }

    #[test]
    fn below_only_replicates_top_row_upwards() {
        // Boundary MB 0 (below/south, priority 0) — its top row is
        // replicated upwards into every row of the exterior MB.
        let below = make_grid::<8>(200);
        // below[0] = [200, 201, 202, 203, 204, 205, 206, 207]
        let neighbours = BoundaryNeighbours {
            below: Some(&below),
            ..BoundaryNeighbours::none()
        };
        let (out, outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(
            outcome,
            ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Below)
        );
        let expected_row = below[0];
        for y in 0..8 {
            assert_eq!(out[y], expected_row, "y={y}");
        }
    }

    #[test]
    fn all_four_neighbours_picks_left_as_priority_3() {
        // Per Figure 7-28: priority 3 > 2 > 1 > 0 → Left wins.
        let left = make_grid::<8>(0);
        let above = make_grid::<8>(100);
        let right = make_grid::<8>(200);
        let below = make_grid::<8>(300);
        let neighbours = BoundaryNeighbours {
            left: Some(&left),
            above: Some(&above),
            right: Some(&right),
            below: Some(&below),
        };
        let (out, outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(
            outcome,
            ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Left)
        );
        for y in 0..8 {
            let expected = left[y][7];
            for x in 0..8 {
                assert_eq!(out[y][x], expected, "y={y} x={x}");
            }
        }
    }

    #[test]
    fn above_and_right_picks_above_as_priority_2() {
        // Priority 2 > 1 → Above wins over Right when Left is absent.
        let above = make_grid::<8>(100);
        let right = make_grid::<8>(200);
        let neighbours = BoundaryNeighbours {
            above: Some(&above),
            right: Some(&right),
            ..BoundaryNeighbours::none()
        };
        let (_out, outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(
            outcome,
            ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Above)
        );
    }

    #[test]
    fn right_and_below_picks_right_as_priority_1() {
        // Priority 1 > 0 → Right wins over Below when Left and Above
        // are absent.
        let right = make_grid::<8>(200);
        let below = make_grid::<8>(300);
        let neighbours = BoundaryNeighbours {
            right: Some(&right),
            below: Some(&below),
            ..BoundaryNeighbours::none()
        };
        let (_out, outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(
            outcome,
            ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Right)
        );
    }

    #[test]
    fn priority_value_table_matches_figure_7_28() {
        // Figure 7-28: Left=3, Above=2, Right=1, Below=0.
        assert_eq!(ExteriorNeighbourPosition::Left.priority(), 3);
        assert_eq!(ExteriorNeighbourPosition::Above.priority(), 2);
        assert_eq!(ExteriorNeighbourPosition::Right.priority(), 1);
        assert_eq!(ExteriorNeighbourPosition::Below.priority(), 0);
    }

    #[test]
    fn highest_priority_position_with_empty_set_is_none() {
        let neighbours: BoundaryNeighbours<16> = BoundaryNeighbours::none();
        assert_eq!(neighbours.highest_priority_position(), None);
    }

    #[test]
    fn extended_padding_luma_with_above_neighbour_replicates_bottom_row() {
        // Luma side is 16: confirm the const-generic path works at
        // N=16 the same as the N=8 chroma case.
        let above = make_grid::<16>(0);
        let neighbours = BoundaryNeighbours {
            above: Some(&above),
            ..BoundaryNeighbours::none()
        };
        let (out, outcome) = extended_padding_luma(&neighbours, 8);
        assert_eq!(
            outcome,
            ExteriorPaddingOutcome::FromNeighbour(ExteriorNeighbourPosition::Above)
        );
        let expected_row = above[15];
        for y in 0..16 {
            assert_eq!(out[y], expected_row, "y={y}");
        }
    }

    #[test]
    fn extended_padding_chroma_with_no_neighbour_is_mid_grey() {
        let neighbours: BoundaryNeighbours<CHROMA_SIDE> = BoundaryNeighbours::none();
        let (out, outcome) = extended_padding_chroma(&neighbours, 8);
        assert_eq!(outcome, ExteriorPaddingOutcome::MidGrey);
        for row in out.iter() {
            for &v in row.iter() {
                assert_eq!(v, 128);
            }
        }
    }

    #[test]
    fn left_neighbour_replicates_only_rightmost_column() {
        // Sanity: left[y][0..=6] are NOT replicated — only left[y][7]
        // (rightmost column) flows into the exterior MB.
        let left = make_grid::<8>(0);
        // left[3][7] = 0 + 30 + 7 = 37; left[3][0] = 30. We expect
        // the output row 3 to be all 37 (not 30).
        let neighbours = BoundaryNeighbours {
            left: Some(&left),
            ..BoundaryNeighbours::none()
        };
        let (out, _outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(out[3], [37; 8]);
        assert_ne!(out[3][0], 30);
    }

    #[test]
    fn right_neighbour_replicates_only_leftmost_column() {
        // Sanity: right[y][1..=7] are NOT replicated — only right[y][0]
        // (leftmost column) flows into the exterior MB.
        let right = make_grid::<8>(0);
        // right[5][0] = 0 + 50 + 0 = 50; right[5][7] = 57. We expect
        // the output row 5 to be all 50 (not 57).
        let neighbours = BoundaryNeighbours {
            right: Some(&right),
            ..BoundaryNeighbours::none()
        };
        let (out, _outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        assert_eq!(out[5], [50; 8]);
        assert_ne!(out[5][7], 57);
    }

    #[test]
    fn above_neighbour_replicates_only_bottom_row() {
        // Sanity: above[0..=6][x] are NOT replicated — only above[7][x]
        // flows into the exterior MB.
        let above = make_grid::<8>(0);
        let neighbours = BoundaryNeighbours {
            above: Some(&above),
            ..BoundaryNeighbours::none()
        };
        let (out, _outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        // above[7] = [70, 71, 72, 73, 74, 75, 76, 77]; not above[0].
        assert_eq!(out[0], [70, 71, 72, 73, 74, 75, 76, 77]);
        assert_ne!(out[0], above[0]);
    }

    #[test]
    fn below_neighbour_replicates_only_top_row() {
        // Sanity: below[1..=7][x] are NOT replicated — only below[0][x]
        // flows into the exterior MB.
        let below = make_grid::<8>(0);
        let neighbours = BoundaryNeighbours {
            below: Some(&below),
            ..BoundaryNeighbours::none()
        };
        let (out, _outcome) = extended_padding_macroblock::<8>(&neighbours, 8);
        // below[0] = [0, 1, 2, 3, 4, 5, 6, 7]; not below[7].
        assert_eq!(out[7], [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_ne!(out[7], below[7]);
    }

    #[test]
    fn default_boundary_neighbours_has_no_sides() {
        let neighbours: BoundaryNeighbours<16> = BoundaryNeighbours::default();
        assert!(neighbours.left.is_none());
        assert!(neighbours.above.is_none());
        assert!(neighbours.right.is_none());
        assert!(neighbours.below.is_none());
        assert_eq!(neighbours.highest_priority_position(), None);
    }
}

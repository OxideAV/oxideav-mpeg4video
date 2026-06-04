//! §7.6.1.2 vertical repetitive padding — the second pass of the
//! §7.6.1 reference-VOP padding pipeline.
//!
//! The §7.6.1.1 horizontal pass (`crate::sample_padding`) fills the
//! transparent samples on every row that contains at least one opaque
//! sample, leaving rows with no opaque sample untouched (their
//! `s'[y][x]` stays at 0). The §7.6.1.2 vertical pass takes the
//! `(hor_pad, s')` output and operates column-by-column, filling the
//! remaining transparent samples by replicating the nearest sample
//! above or below — treating samples already filled by §7.6.1.1 as if
//! they were inside the VOP for the purpose of this pass.
//!
//! ## Algorithm (verbatim from ISO/IEC 14496-2:2004 §7.6.1.2)
//!
//! ```text
//! for (y = 0; y < M; y++) {
//!     if (s'[y][x] == 1) {
//!         hv_pad[y][x] = hor_pad[y][x];
//!     } else {
//!         if (s'[y'][x] == 1 && s'[y''][x] == 1)
//!             hv_pad[y][x] = (hor_pad[y'][x] + hor_pad[y''][x]) // 2;
//!         else if (s'[y'][x] == 1)
//!             hv_pad[y][x] = hor_pad[y'][x];
//!         else if (s'[y''][x] == 1)
//!             hv_pad[y][x] = hor_pad[y''][x];
//!     }
//! }
//! ```
//!
//! where `y'` is the nearest valid sample (`s'[y'][x] == 1`) above the
//! current location `y`, `y''` is the nearest boundary sample below
//! `y`, and `M` is the number of samples of a column in a macroblock
//! (16 for luminance, 8 for each chrominance block per §7.6.1.4). The
//! `//` operator is §3.4 integer division with truncation toward zero.
//!
//! ## Column-level outcome
//!
//! Three cases arise per column once both passes complete:
//!
//! * **At least one `s'[y][x] == 1` in the column.** Every transparent
//!   position is filled from the nearest above/below opaque sample (or
//!   the average of the two when both exist); every output position is
//!   marked opaque in the §7.6.1.2 sentinel `s''[y][x]`.
//! * **The column has no `s'[y][x] == 1`.** The §7.6.1.2 algorithm's
//!   inner branches all fail and `hv_pad[y][x]` is left undefined by
//!   the spec for the column. The §7.6.1 framing places such columns
//!   inside fully-transparent macroblocks; the §7.6.1.3 extended
//!   padding pass routes those macroblocks differently and the
//!   §7.6.1.2 pass is not the one that fills them. This module
//!   surfaces the case via [`ColumnState::FullyTransparent`] so the
//!   caller can route the macroblock to §7.6.1.3 instead.
//!
//! ## What this module does **not** do
//!
//! * Decide whether the macroblock is a boundary or exterior
//!   macroblock — that branch selector lives in the caller (which has
//!   access to the full VOP shape grid) and is the same caller-side
//!   routing as §7.6.1.1.
//! * Apply the §7.6.1.3 extended-padding pass for exterior
//!   macroblocks — exterior macroblocks immediately next to boundary
//!   macroblocks copy the boundary's edge row/column, and the
//!   remaining exterior macroblocks are filled with the
//!   `2^(bits_per_pixel - 1)` mid-grey value. Both are later-round
//!   work.
//! * Decimate the luminance shape mask into the §7.6.1.4 chroma shape.
//!   The chroma shape comes from §6.1.3.6 subsampling and is the
//!   caller's responsibility; the [`vertical_repetitive_padding_luma`]
//!   / [`vertical_repetitive_padding_chroma`] entry points operate on
//!   whichever side length the caller supplies (via the per-side
//!   constants exported by the §7.6.1.1 module).
//! * Cover the §7.6.1.5 interlaced per-field vertical padding.
//!   Interlaced padding splits the macroblock into two 16×8 fields and
//!   applies §7.6.1.1..§7.6.1.3 to each independently; the vertical
//!   pass is structurally identical to the progressive case so a
//!   future interlaced entry point can re-use this module against
//!   per-field slices.

#![allow(clippy::needless_range_loop)]

use crate::sample_padding::{SamplePresence, ShapeRowState, CHROMA_SIDE, LUMA_SIDE};

/// Per-column §7.6.1.2 outcome, returned alongside the padded column.
///
/// `FullyFilled` indicates the column had at least one `s'[y][x] == 1`
/// sample and the §7.6.1.2 pass filled every remaining transparent
/// sample from an above- or below-neighbour. `FullyTransparent`
/// indicates the column had no opaque sample after the §7.6.1.1 pass
/// (the §7.6.1.2 inner branches all fail); the §7.6.1 framing routes
/// such columns to §7.6.1.3 extended padding instead, so the caller
/// should not rely on the corresponding output samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnState {
    /// The column had at least one `s'[y][x] == 1` sample and every
    /// position is filled (`s''[y][x] == 1` for every `y`).
    FullyFilled,
    /// The column had no `s'[y][x] == 1` sample after §7.6.1.1; the
    /// §7.6.1.2 pass leaves the column unchanged and the caller must
    /// route the macroblock to §7.6.1.3 extended padding.
    FullyTransparent,
}

/// §7.6.1.2 vertical repetitive padding for one column.
///
/// `hor_pad` is the column of `hor_pad[y][x]` produced by the §7.6.1.1
/// horizontal pass; `s_prime` is the matching column of `s'[y][x]`.
/// On return:
///
/// * `out` holds `hv_pad[y][x]`. Positions whose `s'[y][x] == Opaque`
///   are copied straight from `hor_pad`; positions whose
///   `s'[y][x] == Transparent` are filled by replicating the
///   above/below neighbour or by averaging them when both exist.
///   Transparent positions in a fully-transparent column are left at
///   their input value (the caller routes the macroblock to §7.6.1.3).
/// * `s_double_prime` holds the §7.6.1.2 fill sentinel `s''[y][x]`:
///   `Opaque` for every position that was either already opaque
///   (`s'[y][x] == 1`) or filled by this pass, `Transparent` only when
///   the entire column had no opaque sample (the §7.6.1.2 inner
///   branches all fail; the caller routes the column to §7.6.1.3).
/// * The returned [`ColumnState`] reports whether the column was
///   filled or skipped.
///
/// Arithmetic note: the spec's `(hor_pad[y'][x] + hor_pad[y''][x])
/// // 2` is the §3.4 `//` operator (truncation toward zero). The
/// implementation uses `i32::div_euclid` against `2`, which matches
/// `//` for the non-negative `hor_pad` values in the §7.3 step-3
/// display range `[0, 2^bits_per_pixel - 1]`.
pub fn vertical_repetitive_padding_column<const M: usize>(
    hor_pad: &[i32; M],
    s_prime: &[SamplePresence; M],
    out: &mut [i32; M],
    s_double_prime: &mut [SamplePresence; M],
) -> ColumnState {
    // §7.6.1.2: every position whose s'[y][x] == 1 maps unchanged into
    // hv_pad and s''[y][x]; transparent positions await fill.
    *s_double_prime = [SamplePresence::Transparent; M];

    // Column-guard: a column with no §7.6.1.1-opaque sample falls
    // through every inner branch of §7.6.1.2. The §7.6.1 framing
    // places such columns inside exterior macroblocks (or — for a
    // boundary macroblock — inside a column-wide all-transparent
    // strip the §7.6.1.3 pass would normally fill from a neighbour
    // boundary MB). Either way the §7.6.1.2 pass is not the one that
    // fills them; report the case so the caller can route the
    // macroblock.
    if s_prime.iter().all(|s| s.is_transparent()) {
        // Copy hor_pad into out unchanged so the column is a defined
        // tensor.
        *out = *hor_pad;
        return ColumnState::FullyTransparent;
    }

    // Precompute the per-position `y'` (nearest opaque at-or-above y
    // — both passes treat the current position as a valid neighbour
    // when it is itself opaque, but the spec's outer-if has already
    // short-circuited that case, so for the fill branches `y' <= y`
    // with strict inequality when y itself is transparent) and `y''`
    // (nearest opaque strictly below y). The "strictly below" on `y''`
    // matches the spec text "the nearest boundary sample below" and
    // yields the (y', y'') ordering the averaging branch requires
    // when the current sample is itself transparent.
    let mut nearest_above: [Option<usize>; M] = [None; M];
    let mut nearest_below: [Option<usize>; M] = [None; M];
    let mut last_opaque: Option<usize> = None;
    for y in 0..M {
        if s_prime[y].is_opaque() {
            last_opaque = Some(y);
        }
        nearest_above[y] = last_opaque;
    }
    let mut next_opaque: Option<usize> = None;
    for y in (0..M).rev() {
        nearest_below[y] = next_opaque;
        if s_prime[y].is_opaque() {
            next_opaque = Some(y);
        }
    }

    for y in 0..M {
        if s_prime[y].is_opaque() {
            out[y] = hor_pad[y];
            s_double_prime[y] = SamplePresence::Opaque;
        } else {
            match (nearest_above[y], nearest_below[y]) {
                (Some(y_above), Some(y_below)) => {
                    // §7.6.1.2 averaging branch: (hor_pad[y'] + hor_pad[y'']) // 2.
                    let sum = hor_pad[y_above] + hor_pad[y_below];
                    out[y] = sum.div_euclid(2);
                    s_double_prime[y] = SamplePresence::Opaque;
                }
                (Some(y_above), None) => {
                    // Only the above-boundary sample exists in this column.
                    out[y] = hor_pad[y_above];
                    s_double_prime[y] = SamplePresence::Opaque;
                }
                (None, Some(y_below)) => {
                    // Only the below-boundary sample exists in this column.
                    out[y] = hor_pad[y_below];
                    s_double_prime[y] = SamplePresence::Opaque;
                }
                (None, None) => {
                    // Unreachable: the column-guard above has already
                    // returned `FullyTransparent` for columns with no
                    // opaque sample.
                    unreachable!(
                        "vertical_repetitive_padding_column: column with no opaque sample reached the inner loop"
                    );
                }
            }
        }
    }

    ColumnState::FullyFilled
}

/// §7.6.1.2 vertical repetitive padding for one 4:2:0 luminance
/// macroblock.
///
/// `hor_pad[y][x]` is the row-major 16×16 output of §7.6.1.1;
/// `s_prime[y][x]` is the matching `s'[y][x]` sentinel; `row_states`
/// is the per-row outcome from §7.6.1.1 (currently used only for the
/// post-condition that every fully-transparent row's column will get
/// filled by §7.6.1.2 — kept as a parameter so the caller can wire
/// the §7.6.1.1 output triple through unchanged). The returned
/// `(hv_pad, s_double_prime, column_states)` triple is the §7.6.1.2
/// output: per-pixel padded macroblock, per-pixel `s''[y][x]` sentinel,
/// and per-column [`ColumnState`] indicating whether each column was
/// filled or left for §7.6.1.3.
pub fn vertical_repetitive_padding_luma(
    hor_pad: &[[i32; LUMA_SIDE]; LUMA_SIDE],
    s_prime: &[[SamplePresence; LUMA_SIDE]; LUMA_SIDE],
    _row_states: &[ShapeRowState; LUMA_SIDE],
) -> (
    [[i32; LUMA_SIDE]; LUMA_SIDE],
    [[SamplePresence; LUMA_SIDE]; LUMA_SIDE],
    [ColumnState; LUMA_SIDE],
) {
    vertical_repetitive_padding_impl::<LUMA_SIDE>(hor_pad, s_prime)
}

/// §7.6.1.2 vertical repetitive padding for one 4:2:0 chrominance
/// block (Cb or Cr, 8×8 samples per §7.6.1.4).
///
/// Same shape as [`vertical_repetitive_padding_luma`] but the side
/// length is 8 to match the §6.1.3.4 chroma format. The decimated
/// chroma shape comes from §6.1.3.6 — out of scope for this module.
pub fn vertical_repetitive_padding_chroma(
    hor_pad: &[[i32; CHROMA_SIDE]; CHROMA_SIDE],
    s_prime: &[[SamplePresence; CHROMA_SIDE]; CHROMA_SIDE],
    _row_states: &[ShapeRowState; CHROMA_SIDE],
) -> (
    [[i32; CHROMA_SIDE]; CHROMA_SIDE],
    [[SamplePresence; CHROMA_SIDE]; CHROMA_SIDE],
    [ColumnState; CHROMA_SIDE],
) {
    vertical_repetitive_padding_impl::<CHROMA_SIDE>(hor_pad, s_prime)
}

/// Inner const-generic implementation shared by the luma / chroma
/// entry points. The §7.6.1.2 column loop reads one row-major column
/// at a time, so we materialise the column slices on the stack before
/// calling [`vertical_repetitive_padding_column`].
fn vertical_repetitive_padding_impl<const N: usize>(
    hor_pad: &[[i32; N]; N],
    s_prime: &[[SamplePresence; N]; N],
) -> ([[i32; N]; N], [[SamplePresence; N]; N], [ColumnState; N]) {
    let mut out = [[0i32; N]; N];
    let mut s_double_prime = [[SamplePresence::Transparent; N]; N];
    let mut column_states = [ColumnState::FullyTransparent; N];
    for x in 0..N {
        let mut column_in = [0i32; N];
        let mut column_s = [SamplePresence::Transparent; N];
        for y in 0..N {
            column_in[y] = hor_pad[y][x];
            column_s[y] = s_prime[y][x];
        }
        let mut column_out = [0i32; N];
        let mut column_sdp = [SamplePresence::Transparent; N];
        column_states[x] = vertical_repetitive_padding_column(
            &column_in,
            &column_s,
            &mut column_out,
            &mut column_sdp,
        );
        for y in 0..N {
            out[y][x] = column_out[y];
            s_double_prime[y][x] = column_sdp[y];
        }
    }
    (out, s_double_prime, column_states)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_padding::{
        horizontal_repetitive_padding_chroma, horizontal_repetitive_padding_luma,
    };

    fn op() -> SamplePresence {
        SamplePresence::Opaque
    }
    fn tr() -> SamplePresence {
        SamplePresence::Transparent
    }

    #[test]
    fn fully_opaque_column_is_identity() {
        let hor_pad = [10, 20, 30, 40, 50, 60, 70, 80];
        let s_prime = [op(); 8];
        let mut out = [0; 8];
        let mut s_dp = [tr(); 8];
        let state = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(state, ColumnState::FullyFilled);
        assert_eq!(out, hor_pad);
        assert_eq!(s_dp, [op(); 8]);
    }

    #[test]
    fn fully_transparent_column_is_skipped() {
        let hor_pad = [10, 20, 30, 40, 50, 60, 70, 80];
        let s_prime = [tr(); 8];
        let mut out = [0; 8];
        let mut s_dp = [op(); 8];
        let state = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(state, ColumnState::FullyTransparent);
        // hor_pad passes through; s''[y][x] is reset to all-transparent
        // so the caller can see the column is unfilled.
        assert_eq!(out, hor_pad);
        assert_eq!(s_dp, [tr(); 8]);
    }

    #[test]
    fn above_only_fill_replicates_above_neighbour() {
        // Opaque on y = 0..=3, transparent on y = 4..=7. §7.6.1.2:
        // y'' does not exist for y >= 4 (no opaque sample below);
        // only y' = 3 exists, so hv_pad[4..=7] = hor_pad[3] = 40.
        let hor_pad = [10, 20, 30, 40, 99, 99, 99, 99];
        let s_prime = [op(), op(), op(), op(), tr(), tr(), tr(), tr()];
        let mut out = [0; 8];
        let mut s_dp = [tr(); 8];
        let state = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(state, ColumnState::FullyFilled);
        assert_eq!(out, [10, 20, 30, 40, 40, 40, 40, 40]);
        assert_eq!(s_dp, [op(); 8]);
    }

    #[test]
    fn below_only_fill_replicates_below_neighbour() {
        // Transparent on y = 0..=3, opaque on y = 4..=7. §7.6.1.2:
        // y' does not exist for y <= 3 (no opaque sample at or above);
        // only y'' = 4 exists, so hv_pad[0..=3] = hor_pad[4] = 50.
        let hor_pad = [99, 99, 99, 99, 50, 60, 70, 80];
        let s_prime = [tr(), tr(), tr(), tr(), op(), op(), op(), op()];
        let mut out = [0; 8];
        let mut s_dp = [tr(); 8];
        let state = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(state, ColumnState::FullyFilled);
        assert_eq!(out, [50, 50, 50, 50, 50, 60, 70, 80]);
        assert_eq!(s_dp, [op(); 8]);
    }

    #[test]
    fn two_sided_fill_averages_with_truncation() {
        // Opaque at y = 0 and y = 7; transparent everywhere between.
        // y' = 0 for every interior position (hor_pad[0] = 10) and
        // y'' = 7 (hor_pad[7] = 41). hv_pad[1..=6] = (10 + 41) // 2 =
        // 51 // 2 = 25 (§3.4 truncation toward zero on positive sum).
        let hor_pad = [10, 0, 0, 0, 0, 0, 0, 41];
        let s_prime = [op(), tr(), tr(), tr(), tr(), tr(), tr(), op()];
        let mut out = [0; 8];
        let mut s_dp = [tr(); 8];
        let state = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(state, ColumnState::FullyFilled);
        assert_eq!(out, [10, 25, 25, 25, 25, 25, 25, 41]);
        assert_eq!(s_dp, [op(); 8]);
    }

    #[test]
    fn interior_transparent_uses_immediate_neighbours() {
        // Opaque at y = 0..=2 and y = 5..=7; transparent at y = 3, 4.
        // For y = 3: y' = 2 (hor_pad[2] = 30), y'' = 5 (hor_pad[5] = 60),
        //   hv_pad[3] = (30 + 60) // 2 = 45.
        // For y = 4: y' = 2 (nearest opaque above is still 2),
        //   y'' = 5 still. hv_pad[4] = (30 + 60) // 2 = 45.
        let hor_pad = [10, 20, 30, 0, 0, 60, 70, 80];
        let s_prime = [op(), op(), op(), tr(), tr(), op(), op(), op()];
        let mut out = [0; 8];
        let mut s_dp = [tr(); 8];
        let state = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(state, ColumnState::FullyFilled);
        assert_eq!(out, [10, 20, 30, 45, 45, 60, 70, 80]);
        assert_eq!(s_dp, [op(); 8]);
    }

    #[test]
    fn average_rounds_toward_zero_on_odd_sum() {
        // 11 + 20 = 31; 31 // 2 = 15 (§3.4 truncation toward zero).
        let hor_pad = [11, 0, 0, 20];
        let s_prime = [op(), tr(), tr(), op()];
        let mut out = [0; 4];
        let mut s_dp = [tr(); 4];
        let state = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(state, ColumnState::FullyFilled);
        assert_eq!(out, [11, 15, 15, 20]);
    }

    #[test]
    fn isolated_opaque_sample_replicated_across_column() {
        // One opaque sample at y = 3 in an otherwise transparent column.
        // For y < 3: only y'' = 3 exists, so out[y] = hor_pad[3].
        // For y > 3: only y' = 3 exists, so out[y] = hor_pad[3].
        let hor_pad = [99, 99, 99, 42, 99, 99, 99, 99];
        let s_prime = [tr(), tr(), tr(), op(), tr(), tr(), tr(), tr()];
        let mut out = [0; 8];
        let mut s_dp = [tr(); 8];
        let state = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(state, ColumnState::FullyFilled);
        assert_eq!(out, [42; 8]);
        assert_eq!(s_dp, [op(); 8]);
    }

    #[test]
    fn average_branch_at_extreme_display_range_no_overflow() {
        // For bits_per_pixel = 12 the upper bound is 4095. Two 4095
        // samples averaged: (4095 + 4095) // 2 = 4095. i32 comfortably
        // holds the intermediate.
        let hor_pad = [4095, 0, 0, 0, 4095];
        let s_prime = [op(), tr(), tr(), tr(), op()];
        let mut out = [0; 5];
        let mut s_dp = [tr(); 5];
        let _ = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(out, [4095; 5]);
    }

    #[test]
    fn s_double_prime_is_reset_between_calls() {
        // Verify that a caller reusing the s'' buffer always sees a
        // from-scratch §7.6.1.2 sentinel.
        let hor_pad = [5, 10, 15, 20];
        let s_prime = [op(), tr(), op(), tr()];
        let mut out = [0; 4];
        let mut s_dp = [op(); 4];
        let _ = vertical_repetitive_padding_column(&hor_pad, &s_prime, &mut out, &mut s_dp);
        assert_eq!(s_dp, [op(); 4]);
        // y = 1: average of hor_pad[0] (5) and hor_pad[2] (15) = 20 // 2 = 10.
        // y = 3: only y' = 2 exists, so hor_pad[2] = 15.
        assert_eq!(out, [5, 10, 15, 15]);
    }

    #[test]
    fn full_macroblock_horizontal_then_vertical_pass() {
        // A diagonal boundary: row y has opaque samples for x <= y, the
        // rest transparent. After §7.6.1.1, every row is FullyFilled.
        // The §7.6.1.2 pass should therefore be an identity on
        // (hor_pad, s_prime) because every s'[y][x] is already Opaque.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        for y in 0..LUMA_SIDE {
            for x in 0..LUMA_SIDE {
                decoded[y][x] = (y * LUMA_SIDE + x) as i32;
                if x <= y {
                    shape[y][x] = op();
                }
            }
        }
        let (hor_pad, s_prime, row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        let (hv_pad, s_dp, col_states) =
            vertical_repetitive_padding_luma(&hor_pad, &s_prime, &row_states);
        // Every column had at least one opaque sample → all FullyFilled.
        assert_eq!(col_states, [ColumnState::FullyFilled; LUMA_SIDE]);
        // Every s''[y][x] is Opaque.
        for y in 0..LUMA_SIDE {
            assert_eq!(s_dp[y], [op(); LUMA_SIDE]);
        }
        // Because the horizontal pass already filled every row,
        // hv_pad must equal hor_pad pixel-for-pixel.
        assert_eq!(hv_pad, hor_pad);
    }

    #[test]
    fn full_macroblock_top_opaque_bottom_transparent() {
        // Top half (y < 8) fully opaque, bottom half fully transparent.
        // §7.6.1.1 fills the top rows (identity) and skips the bottom
        // rows as FullyTransparent. §7.6.1.2 then fills the bottom
        // rows column-by-column: for each column x, y' = 7 (last
        // opaque row above), y'' does not exist (no opaque sample
        // below) → out[y][x] = hor_pad[7][x] = decoded[7][x] for
        // y >= 8.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        for y in 0..LUMA_SIDE {
            for x in 0..LUMA_SIDE {
                decoded[y][x] = (y * LUMA_SIDE + x) as i32;
                if y < 8 {
                    shape[y][x] = op();
                }
            }
        }
        let (hor_pad, s_prime, row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        let (hv_pad, s_dp, col_states) =
            vertical_repetitive_padding_luma(&hor_pad, &s_prime, &row_states);
        // Every column has 8 opaque rows in the top half → FullyFilled.
        assert_eq!(col_states, [ColumnState::FullyFilled; LUMA_SIDE]);
        // Top rows unchanged.
        for y in 0..8 {
            assert_eq!(hv_pad[y], decoded[y]);
            assert_eq!(s_dp[y], [op(); LUMA_SIDE]);
        }
        // Bottom rows replicate row 7 of hor_pad (== row 7 of decoded
        // since row 7 is fully opaque).
        for y in 8..LUMA_SIDE {
            assert_eq!(hv_pad[y], decoded[7]);
            assert_eq!(s_dp[y], [op(); LUMA_SIDE]);
        }
    }

    #[test]
    fn full_macroblock_left_opaque_right_transparent_yields_horizontal_only() {
        // Left half (x < 8) opaque, right half transparent. Every row
        // has opaque samples, so §7.6.1.1 fills every row. After
        // §7.6.1.1, s'[y][x] = Opaque everywhere → §7.6.1.2 is an
        // identity.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        for y in 0..LUMA_SIDE {
            for x in 0..LUMA_SIDE {
                decoded[y][x] = (y * LUMA_SIDE + x) as i32;
                if x < 8 {
                    shape[y][x] = op();
                }
            }
        }
        let (hor_pad, s_prime, row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        let (hv_pad, s_dp, col_states) =
            vertical_repetitive_padding_luma(&hor_pad, &s_prime, &row_states);
        assert_eq!(col_states, [ColumnState::FullyFilled; LUMA_SIDE]);
        for y in 0..LUMA_SIDE {
            assert_eq!(s_dp[y], [op(); LUMA_SIDE]);
        }
        assert_eq!(hv_pad, hor_pad);
    }

    #[test]
    fn full_macroblock_middle_opaque_strip() {
        // Only rows y = 4..=7 contain opaque samples (the strip is
        // also restricted to x = 4..=7 in each of those rows). After
        // §7.6.1.1 every row in 4..=7 becomes fully filled with
        // hor_pad[y][x] = decoded[y][4..=7] replicated outward; rows
        // 0..=3 and 8..=15 stay FullyTransparent. The §7.6.1.2 pass
        // then fills every column from the strip — columns 0..=15 are
        // filled because each contains s'==1 within rows 4..=7.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        for y in 4..=7 {
            for x in 4..=7 {
                decoded[y][x] = (y * LUMA_SIDE + x) as i32;
                shape[y][x] = op();
            }
        }
        let (hor_pad, s_prime, row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        let (_hv_pad, s_dp, col_states) =
            vertical_repetitive_padding_luma(&hor_pad, &s_prime, &row_states);
        // Every column intersects rows 4..=7 → FullyFilled.
        assert_eq!(col_states, [ColumnState::FullyFilled; LUMA_SIDE]);
        for y in 0..LUMA_SIDE {
            assert_eq!(s_dp[y], [op(); LUMA_SIDE]);
        }
    }

    #[test]
    fn full_macroblock_all_transparent_reports_fully_transparent_columns() {
        // No opaque samples anywhere — §7.6.1.1 yields all-Transparent
        // s' and every row is FullyTransparent; §7.6.1.2 reports every
        // column as FullyTransparent. The caller must route the
        // macroblock to §7.6.1.3 extended padding.
        let decoded = [[7i32; LUMA_SIDE]; LUMA_SIDE];
        let shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        let (hor_pad, s_prime, row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        let (hv_pad, s_dp, col_states) =
            vertical_repetitive_padding_luma(&hor_pad, &s_prime, &row_states);
        assert_eq!(col_states, [ColumnState::FullyTransparent; LUMA_SIDE]);
        for y in 0..LUMA_SIDE {
            assert_eq!(s_dp[y], [tr(); LUMA_SIDE]);
        }
        // hv_pad == decoded (untouched).
        assert_eq!(hv_pad, decoded);
    }

    #[test]
    fn chroma_block_uses_8x8_side() {
        // Sanity that the 8×8 entry point exercises the same column code.
        // Top-right corner (rows 0..=3, cols 4..=7) is opaque; bottom-
        // left (rows 4..=7, cols 0..=3) is transparent. The §7.6.1.1
        // pass: rows 0..=3 are filled (hor_pad[y][0..=3] = decoded[y][4]
        // by left-only fill); rows 4..=7 are FullyTransparent. §7.6.1.2:
        // for every column x, y' = 3, y'' does not exist → fill bottom
        // rows with hor_pad[3][x].
        let mut decoded = [[0i32; CHROMA_SIDE]; CHROMA_SIDE];
        let mut shape = [[tr(); CHROMA_SIDE]; CHROMA_SIDE];
        for y in 0..4 {
            for x in 4..CHROMA_SIDE {
                decoded[y][x] = (y * CHROMA_SIDE + x) as i32;
                shape[y][x] = op();
            }
        }
        let (hor_pad, s_prime, row_states) = horizontal_repetitive_padding_chroma(&decoded, &shape);
        let (hv_pad, s_dp, col_states) =
            vertical_repetitive_padding_chroma(&hor_pad, &s_prime, &row_states);
        // Every column intersects rows 0..=3 (where at least column 4
        // was opaque, so §7.6.1.1 filled all of x=0..=3 from x'' = 4
        // and x=4..=7 were already opaque) → FullyFilled.
        assert_eq!(col_states, [ColumnState::FullyFilled; CHROMA_SIDE]);
        for y in 0..CHROMA_SIDE {
            assert_eq!(s_dp[y], [op(); CHROMA_SIDE]);
            // Bottom rows replicate row 3 of hor_pad.
            if y >= 4 {
                assert_eq!(hv_pad[y], hor_pad[3]);
            }
        }
    }

    #[test]
    fn checker_columns_resolve_each_independently() {
        // Mixed column shapes within one block. Column 0: opaque at
        // y = 0..=3 only → fills rows 4..=7 with hor_pad[3][0].
        // Column 1: opaque at y = 4..=7 only → fills rows 0..=3 with
        // hor_pad[4][1]. Both columns have at least one opaque sample
        // so both report FullyFilled and §7.6.1.2 yields s'' = Opaque
        // everywhere.
        let mut hor_pad = [[0i32; 2]; 8];
        let mut s_prime = [[tr(); 2]; 8];
        for y in 0..4 {
            hor_pad[y][0] = (10 + y) as i32;
            s_prime[y][0] = op();
        }
        for y in 4..8 {
            hor_pad[y][1] = (20 + y) as i32;
            s_prime[y][1] = op();
        }
        // For each column, call the column primitive directly.
        let mut col0_in = [0i32; 8];
        let mut col0_s = [tr(); 8];
        let mut col1_in = [0i32; 8];
        let mut col1_s = [tr(); 8];
        for y in 0..8 {
            col0_in[y] = hor_pad[y][0];
            col0_s[y] = s_prime[y][0];
            col1_in[y] = hor_pad[y][1];
            col1_s[y] = s_prime[y][1];
        }
        let mut col0_out = [0i32; 8];
        let mut col0_sdp = [tr(); 8];
        let mut col1_out = [0i32; 8];
        let mut col1_sdp = [tr(); 8];
        let state0 =
            vertical_repetitive_padding_column(&col0_in, &col0_s, &mut col0_out, &mut col0_sdp);
        let state1 =
            vertical_repetitive_padding_column(&col1_in, &col1_s, &mut col1_out, &mut col1_sdp);
        assert_eq!(state0, ColumnState::FullyFilled);
        assert_eq!(state1, ColumnState::FullyFilled);
        // Column 0: rows 4..=7 take hor_pad[3][0] = 13.
        assert_eq!(col0_out, [10, 11, 12, 13, 13, 13, 13, 13]);
        // Column 1: rows 0..=3 take hor_pad[4][1] = 24.
        assert_eq!(col1_out, [24, 24, 24, 24, 24, 25, 26, 27]);
        assert_eq!(col0_sdp, [op(); 8]);
        assert_eq!(col1_sdp, [op(); 8]);
    }
}

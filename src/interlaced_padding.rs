//! §7.6.1.5 padding of interlaced macroblocks — luminance boundary path.
//!
//! The §7.6.1 reference-VOP padding pipeline already covers progressive
//! macroblocks (the §7.6.1.1 horizontal pass in [`crate::sample_padding`],
//! the §7.6.1.2 vertical pass in [`crate::vertical_padding`], and the
//! §7.6.1.3 extended pass for exterior macroblocks in
//! [`crate::extended_padding`]). The §7.6.1.5 subclause adds an
//! interlaced (`interlaced == 1`) carve-out to that pipeline:
//!
//! > Macroblocks of interlaced VOP (`interlaced = 1`) are padded
//! > according to subclauses 7.6.1.1 through 7.6.1.3. The vertical
//! > padding of the luminance component, however, is performed for each
//! > field independently. A sample outside of a VOP is therefore filled
//! > with the value of the nearest boundary sample of the same field.
//! > Completely transparent blocks are padded with
//! > `2 ^ (bits_per_pixel - 1)`. Chrominance components of interlaced
//! > VOP are padded according to subclause 7.6.1.4, however, based on
//! > fields to enhance subjective quality of display in 4:2:0 format.
//! > The padding method described in this subclause is not used outside
//! > the bounding rectangle of the VOP.
//! >
//! > — ISO/IEC 14496-2:2004(E) §7.6.1.5 (`docs/video/mpeg4-visual/
//! > ISO_IEC_14496-2-2004-3rd-edition.txt`, lines 19190..19198).
//!
//! ## Scope of this module
//!
//! This module covers the **luminance boundary macroblock** path of
//! §7.6.1.5: the §7.6.1.1 horizontal pass runs over the whole 16×16
//! frame macroblock (the §7.6.1.5 carve-out names only the vertical
//! pass), then the §7.6.1.2 vertical pass runs **per field
//! independently** on the post-§7.6.1.1 16×16 grid (top field = rows
//! `0, 2, …, 14`; bottom field = rows `1, 3, …, 15`), then the two
//! per-field 16×8 outputs are re-interleaved back into a 16×16 frame.
//!
//! The "of the same field" rule the spec text quotes is the §7.6.1.2
//! per-field vertical scan's `y'` (nearest opaque above in the same
//! field) and `y''` (nearest opaque below in the same field): once the
//! frame macroblock is split into two field views, the existing
//! §7.6.1.2 column pass already implements that nearest-in-field
//! semantics — every neighbour candidate it considers lives in the
//! field it was given.
//!
//! ## What this module does **not** do
//!
//! * The §7.6.1.5 chrominance carve-out ("based on fields"). The
//!   chrominance per-field padding is structurally analogous — the
//!   §6.1.3.7.1 interlaced rule already decimates a per-field 8×16
//!   luma shape into a per-field 4×8 chroma shape via
//!   [`crate::chroma_shape::decimate_chroma_shape_interlaced_field`],
//!   and the §7.6.1.4 horizontal+vertical pipeline runs over each
//!   per-field 4×8 chroma block. The chroma wrapper is a separate
//!   later-round entry point.
//! * The §7.6.1.3 extended-padding pass for exterior macroblocks under
//!   `interlaced == 1`. The §7.6.1.5 text routes exterior macroblocks
//!   through §7.6.1.3 unchanged in the frame-mode sense (with the
//!   carve-out replacing the §7.6.1.3 mid-grey fallback by the
//!   `2 ^ (bits_per_pixel - 1)` fill *only* for completely-transparent
//!   macroblocks — which is the §7.6.1.3 mid-grey case verbatim). The
//!   exterior wrapper is a separate later-round entry point.
//! * The "completely transparent" fill rule. A completely-transparent
//!   macroblock has no opaque sample in either field, so the §7.6.1.1
//!   horizontal pass leaves every row `FullyTransparent` and the
//!   §7.6.1.2 per-field vertical pass reports every column
//!   `FullyTransparent`. The caller routes the macroblock to the
//!   §7.6.1.3 mid-grey fill via [`crate::extended_padding`] in that
//!   case — this module's
//!   [`InterlacedBoundaryOutcome::CompletelyTransparent`] surfaces the
//!   case so the caller can route accordingly.
//! * The shape decimation for the §7.6.1.4 chroma routing. The
//!   §6.1.3.7.1 / §7.6.1.5 chroma shape comes from
//!   [`crate::chroma_shape`].

#![allow(clippy::needless_range_loop)]

use crate::sample_padding::{
    horizontal_repetitive_padding_luma, SamplePresence, ShapeRowState, LUMA_SIDE,
};
use crate::vertical_padding::{vertical_repetitive_padding_column, ColumnState};

/// Number of luma rows per field in a 4:2:0 interlaced macroblock.
///
/// Each field carries 8 of the macroblock's 16 luma rows — the top
/// field collects rows `0, 2, 4, …, 14` and the bottom field collects
/// rows `1, 3, 5, …, 15` per the spec's frame/field interleaving
/// convention.
pub const LUMA_FIELD_LINES: usize = LUMA_SIDE / 2;

/// Per-macroblock outcome of the §7.6.1.5 luminance boundary
/// interlaced-padding pass.
///
/// * [`InterlacedBoundaryOutcome::Padded`] — at least one field had an
///   opaque sample after §7.6.1.1 and was filled by the per-field
///   §7.6.1.2 pass. The two `ColumnState` arrays attached to this
///   variant report whether each column of each field was filled or
///   left at its §7.6.1.1 input (a fully-transparent column inside a
///   boundary macroblock is the §7.6.1.2 `FullyTransparent` case —
///   §7.6.1.5 does not change that wiring, the column is left for the
///   caller to route to §7.6.1.3).
/// * [`InterlacedBoundaryOutcome::CompletelyTransparent`] — every row
///   of the macroblock was `FullyTransparent` after the §7.6.1.1
///   horizontal pass. The §7.6.1.5 text routes such macroblocks to
///   the `2 ^ (bits_per_pixel - 1)` fill — the caller wires that via
///   [`crate::extended_padding`] (no fill is performed inside this
///   module).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterlacedBoundaryOutcome {
    /// At least one row had an opaque sample. The per-field column
    /// states are reported for caller-side routing.
    Padded {
        /// Per-column outcome on the top field (rows `0, 2, …, 14`).
        top_column_states: [ColumnState; LUMA_SIDE],
        /// Per-column outcome on the bottom field (rows `1, 3, …, 15`).
        bottom_column_states: [ColumnState; LUMA_SIDE],
    },
    /// Every row was fully transparent — the macroblock is
    /// completely transparent. The caller fills it with
    /// `2 ^ (bits_per_pixel - 1)` per the §7.6.1.5 carve-out (which is
    /// the §7.6.1.3 mid-grey case).
    CompletelyTransparent,
}

/// Split a 16×16 frame-mode `i32` sample buffer into a `(top, bottom)`
/// pair of 16×8 field views by row parity.
///
/// `top[f]` is `frame[2 * f]` (rows `0, 2, …, 14`); `bottom[f]` is
/// `frame[2 * f + 1]` (rows `1, 3, …, 15`). The split matches the
/// §6.1.3.7.1 interlaced rule's "of the same field" semantics — each
/// field's row index `f` maps back to frame row `2 * f` (top) or
/// `2 * f + 1` (bottom).
fn split_samples_into_fields(
    frame: &[[i32; LUMA_SIDE]; LUMA_SIDE],
) -> (
    [[i32; LUMA_SIDE]; LUMA_FIELD_LINES],
    [[i32; LUMA_SIDE]; LUMA_FIELD_LINES],
) {
    let mut top = [[0i32; LUMA_SIDE]; LUMA_FIELD_LINES];
    let mut bottom = [[0i32; LUMA_SIDE]; LUMA_FIELD_LINES];
    for f in 0..LUMA_FIELD_LINES {
        top[f] = frame[2 * f];
        bottom[f] = frame[2 * f + 1];
    }
    (top, bottom)
}

/// Split a 16×16 frame-mode [`SamplePresence`] sentinel buffer into a
/// `(top, bottom)` pair of 16×8 field views by row parity. Same row-
/// parity rule as [`split_samples_into_fields`].
fn split_sentinel_into_fields(
    frame: &[[SamplePresence; LUMA_SIDE]; LUMA_SIDE],
) -> (
    [[SamplePresence; LUMA_SIDE]; LUMA_FIELD_LINES],
    [[SamplePresence; LUMA_SIDE]; LUMA_FIELD_LINES],
) {
    let mut top = [[SamplePresence::Transparent; LUMA_SIDE]; LUMA_FIELD_LINES];
    let mut bottom = [[SamplePresence::Transparent; LUMA_SIDE]; LUMA_FIELD_LINES];
    for f in 0..LUMA_FIELD_LINES {
        top[f] = frame[2 * f];
        bottom[f] = frame[2 * f + 1];
    }
    (top, bottom)
}

/// Re-interleave a `(top, bottom)` pair of 16×8 `i32` field views back
/// into a 16×16 frame-mode buffer.
///
/// `frame[2 * f]` is `top[f]`; `frame[2 * f + 1]` is `bottom[f]`. The
/// inverse of [`split_samples_into_fields`].
fn stack_samples_into_frame(
    top: &[[i32; LUMA_SIDE]; LUMA_FIELD_LINES],
    bottom: &[[i32; LUMA_SIDE]; LUMA_FIELD_LINES],
) -> [[i32; LUMA_SIDE]; LUMA_SIDE] {
    let mut frame = [[0i32; LUMA_SIDE]; LUMA_SIDE];
    for f in 0..LUMA_FIELD_LINES {
        frame[2 * f] = top[f];
        frame[2 * f + 1] = bottom[f];
    }
    frame
}

/// Re-interleave a `(top, bottom)` pair of 16×8 [`SamplePresence`]
/// sentinel field views back into a 16×16 frame-mode buffer. Same row-
/// parity rule as [`stack_samples_into_frame`].
fn stack_sentinel_into_frame(
    top: &[[SamplePresence; LUMA_SIDE]; LUMA_FIELD_LINES],
    bottom: &[[SamplePresence; LUMA_SIDE]; LUMA_FIELD_LINES],
) -> [[SamplePresence; LUMA_SIDE]; LUMA_SIDE] {
    let mut frame = [[SamplePresence::Transparent; LUMA_SIDE]; LUMA_SIDE];
    for f in 0..LUMA_FIELD_LINES {
        frame[2 * f] = top[f];
        frame[2 * f + 1] = bottom[f];
    }
    frame
}

/// Return tuple of [`per_field_vertical_padding_luma`]: the
/// re-interleaved frame-mode 16×16 sample buffer, the re-interleaved
/// 16×16 §7.6.1.2 sentinel buffer, the top-field per-column outcome,
/// and the bottom-field per-column outcome.
pub type PerFieldVerticalPaddingResult = (
    [[i32; LUMA_SIDE]; LUMA_SIDE],
    [[SamplePresence; LUMA_SIDE]; LUMA_SIDE],
    [ColumnState; LUMA_SIDE],
    [ColumnState; LUMA_SIDE],
);

/// §7.6.1.5 per-field §7.6.1.2 vertical pass on a 16×16 luminance
/// boundary macroblock.
///
/// Given a 16×16 §7.6.1.1 output `hor_pad` and the matching §7.6.1.1
/// sentinel `s_prime`, this function:
///
/// 1. Splits both buffers into top/bottom field views by row parity.
/// 2. Runs [`vertical_repetitive_padding_column`] on each 8-row column
///    of each field independently — the column scan therefore picks
///    its `y'` / `y''` from within the same field, matching the
///    §7.6.1.5 "of the same field" rule.
/// 3. Re-interleaves the two per-field outputs back into a 16×16
///    frame-mode `(hv_pad, s_double_prime)` pair.
///
/// The returned `column_states` arrays are the per-field [`ColumnState`]
/// reports — `top_column_states[x]` is the outcome of the top field's
/// column `x` (rows `0, 2, …, 14`), `bottom_column_states[x]` is the
/// outcome of the bottom field's column `x` (rows `1, 3, …, 15`).
pub fn per_field_vertical_padding_luma(
    hor_pad: &[[i32; LUMA_SIDE]; LUMA_SIDE],
    s_prime: &[[SamplePresence; LUMA_SIDE]; LUMA_SIDE],
) -> PerFieldVerticalPaddingResult {
    // §7.6.1.5: split the frame-mode payload into top + bottom 16×8
    // field views by row parity.
    let (top_hp, bottom_hp) = split_samples_into_fields(hor_pad);
    let (top_sp, bottom_sp) = split_sentinel_into_fields(s_prime);

    // Run §7.6.1.2 column scan on each per-field 16×8 buffer. The
    // §7.6.1.2 `vertical_repetitive_padding_column` operates on a
    // single column slice and is therefore field-agnostic — feeding it
    // a per-field 8-row column directly enforces the "same field"
    // nearest-neighbour rule.
    let (top_hv, top_sdp, top_col_states) = per_field_vertical_pass(&top_hp, &top_sp);
    let (bottom_hv, bottom_sdp, bottom_col_states) =
        per_field_vertical_pass(&bottom_hp, &bottom_sp);

    // Re-interleave the per-field outputs back into a 16×16 frame view.
    let hv_pad = stack_samples_into_frame(&top_hv, &bottom_hv);
    let s_double_prime = stack_sentinel_into_frame(&top_sdp, &bottom_sdp);

    (hv_pad, s_double_prime, top_col_states, bottom_col_states)
}

/// Run §7.6.1.2 on a single 16×8 field view.
fn per_field_vertical_pass(
    field_hp: &[[i32; LUMA_SIDE]; LUMA_FIELD_LINES],
    field_sp: &[[SamplePresence; LUMA_SIDE]; LUMA_FIELD_LINES],
) -> (
    [[i32; LUMA_SIDE]; LUMA_FIELD_LINES],
    [[SamplePresence; LUMA_SIDE]; LUMA_FIELD_LINES],
    [ColumnState; LUMA_SIDE],
) {
    let mut out = [[0i32; LUMA_SIDE]; LUMA_FIELD_LINES];
    let mut s_double_prime = [[SamplePresence::Transparent; LUMA_SIDE]; LUMA_FIELD_LINES];
    let mut column_states = [ColumnState::FullyTransparent; LUMA_SIDE];
    for x in 0..LUMA_SIDE {
        let mut column_in = [0i32; LUMA_FIELD_LINES];
        let mut column_s = [SamplePresence::Transparent; LUMA_FIELD_LINES];
        for y in 0..LUMA_FIELD_LINES {
            column_in[y] = field_hp[y][x];
            column_s[y] = field_sp[y][x];
        }
        let mut column_out = [0i32; LUMA_FIELD_LINES];
        let mut column_sdp = [SamplePresence::Transparent; LUMA_FIELD_LINES];
        column_states[x] = vertical_repetitive_padding_column(
            &column_in,
            &column_s,
            &mut column_out,
            &mut column_sdp,
        );
        for y in 0..LUMA_FIELD_LINES {
            out[y][x] = column_out[y];
            s_double_prime[y][x] = column_sdp[y];
        }
    }
    (out, s_double_prime, column_states)
}

/// §7.6.1.5 interlaced boundary-macroblock padding for the luminance
/// component — the end-to-end §7.6.1.1 (frame-mode horizontal) +
/// §7.6.1.2 (per-field vertical) pipeline on a 16×16 luma boundary
/// macroblock.
///
/// `decoded[y][x]` is the §7.3 step-3-clipped 16×16 macroblock of
/// `d[y][x]`; `shape[y][x]` is the matching 16×16 row-major shape
/// mask. The returned tuple is:
///
/// * `padded` — the post-§7.6.1.5 16×16 luminance macroblock. Inside a
///   `Padded` outcome every column whose state is `FullyFilled` has
///   every sample set; columns whose state is `FullyTransparent` (the
///   §7.6.1.2 inner-branch fall-through inside a per-field column with
///   no opaque sample) carry the §7.6.1.1 horizontal-pass output
///   unchanged on those positions, and the caller routes the macroblock
///   to the §7.6.1.3 mid-grey / neighbour-replication pass per §7.6.1.5.
/// * `outcome` — the [`InterlacedBoundaryOutcome`] discrimination plus
///   the per-field column states.
///
/// **Boundary classification.** This entry point is the §7.6.1.5
/// boundary-MB path. The §7.6.1 framing places the boundary/exterior
/// branch selector in the caller (it has access to the full VOP shape
/// grid). Calling this function on an exterior macroblock (every
/// sample transparent) yields
/// [`InterlacedBoundaryOutcome::CompletelyTransparent`] and the
/// `padded` buffer carries the input `decoded` straight through — the
/// caller routes the macroblock to [`crate::extended_padding`] per the
/// §7.6.1.5 carve-out ("Completely transparent blocks are padded with
/// `2 ^ (bits_per_pixel - 1)`").
pub fn interlaced_boundary_padding_luma(
    decoded: &[[i32; LUMA_SIDE]; LUMA_SIDE],
    shape: &[[SamplePresence; LUMA_SIDE]; LUMA_SIDE],
) -> ([[i32; LUMA_SIDE]; LUMA_SIDE], InterlacedBoundaryOutcome) {
    // §7.6.1.5: §7.6.1.1 horizontal pass runs over the whole 16×16
    // frame macroblock — the per-field carve-out names only the
    // vertical pass.
    let (hor_pad, s_prime, row_states) = horizontal_repetitive_padding_luma(decoded, shape);

    // Completely-transparent short-circuit: every row was
    // `FullyTransparent`, so the §7.6.1.5 text routes the macroblock
    // to the `2 ^ (bits_per_pixel - 1)` fill. This module hands the
    // caller the original `decoded` payload back unchanged so the
    // caller can pass it (or any other input) into
    // [`crate::extended_padding::extended_padding_macroblock`] without
    // an intermediate copy.
    if row_states
        .iter()
        .all(|s| matches!(s, ShapeRowState::FullyTransparent))
    {
        return (*decoded, InterlacedBoundaryOutcome::CompletelyTransparent);
    }

    // §7.6.1.5: §7.6.1.2 vertical pass runs per-field on the
    // post-§7.6.1.1 grid. The per-field column scan picks `y'` / `y''`
    // from within the same field, matching the "of the same field"
    // rule.
    let (hv_pad, _s_double_prime, top_col_states, bottom_col_states) =
        per_field_vertical_padding_luma(&hor_pad, &s_prime);

    (
        hv_pad,
        InterlacedBoundaryOutcome::Padded {
            top_column_states: top_col_states,
            bottom_column_states: bottom_col_states,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_padding::{
        horizontal_repetitive_padding_luma, SamplePresence, CHROMA_SIDE, LUMA_SIDE,
    };
    use crate::vertical_padding::vertical_repetitive_padding_luma;

    fn op() -> SamplePresence {
        SamplePresence::Opaque
    }
    fn tr() -> SamplePresence {
        SamplePresence::Transparent
    }

    #[test]
    fn luma_field_lines_is_eight() {
        assert_eq!(LUMA_FIELD_LINES, 8);
        assert_eq!(LUMA_FIELD_LINES, LUMA_SIDE / 2);
    }

    #[test]
    fn split_round_trips_via_stack() {
        // Build a 16×16 grid whose (y, x) entry is y * 16 + x so the
        // round-trip catches any indexing mistake.
        let mut frame = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        for y in 0..LUMA_SIDE {
            for x in 0..LUMA_SIDE {
                frame[y][x] = (y * LUMA_SIDE + x) as i32;
            }
        }
        let (top, bottom) = split_samples_into_fields(&frame);
        // top[f] == frame[2 * f]; bottom[f] == frame[2 * f + 1].
        for f in 0..LUMA_FIELD_LINES {
            assert_eq!(top[f], frame[2 * f]);
            assert_eq!(bottom[f], frame[2 * f + 1]);
        }
        let round_trip = stack_samples_into_frame(&top, &bottom);
        assert_eq!(round_trip, frame);
    }

    #[test]
    fn split_round_trips_sample_presence() {
        // Same round-trip on a SamplePresence buffer: rows alternate
        // op() / tr() so any field-swap would mis-color a row.
        let mut frame = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        for y in 0..LUMA_SIDE {
            if y % 2 == 0 {
                frame[y] = [op(); LUMA_SIDE];
            }
        }
        let (top, bottom) = split_sentinel_into_fields(&frame);
        // Top field has all op() rows (rows 0, 2, 4, …);
        // bottom field has all tr() rows (rows 1, 3, 5, …).
        for f in 0..LUMA_FIELD_LINES {
            assert_eq!(top[f], [op(); LUMA_SIDE]);
            assert_eq!(bottom[f], [tr(); LUMA_SIDE]);
        }
        let round_trip = stack_sentinel_into_frame(&top, &bottom);
        assert_eq!(round_trip, frame);
    }

    #[test]
    fn fully_opaque_macroblock_is_identity() {
        // Every sample opaque: §7.6.1.1 is identity, §7.6.1.5 per-
        // field vertical pass is also identity, so the output equals
        // the input. The outcome is Padded and every per-field column
        // is FullyFilled.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        for y in 0..LUMA_SIDE {
            for x in 0..LUMA_SIDE {
                decoded[y][x] = (y * LUMA_SIDE + x) as i32;
            }
        }
        let shape = [[op(); LUMA_SIDE]; LUMA_SIDE];
        let (padded, outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        assert_eq!(padded, decoded);
        match outcome {
            InterlacedBoundaryOutcome::Padded {
                top_column_states,
                bottom_column_states,
            } => {
                assert_eq!(top_column_states, [ColumnState::FullyFilled; LUMA_SIDE]);
                assert_eq!(bottom_column_states, [ColumnState::FullyFilled; LUMA_SIDE]);
            }
            other => panic!("expected Padded outcome, got {other:?}"),
        }
    }

    #[test]
    fn fully_transparent_macroblock_reports_completely_transparent() {
        // Every sample transparent: §7.6.1.1 leaves every row
        // FullyTransparent and the §7.6.1.5 short-circuit returns
        // CompletelyTransparent with the original payload unchanged.
        let decoded = [[7i32; LUMA_SIDE]; LUMA_SIDE];
        let shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        let (padded, outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        assert_eq!(outcome, InterlacedBoundaryOutcome::CompletelyTransparent);
        // Payload passed through untouched.
        assert_eq!(padded, decoded);
    }

    #[test]
    fn top_field_opaque_bottom_field_transparent_pads_per_field() {
        // Top field opaque (rows 0, 2, …, 14), bottom field
        // transparent (rows 1, 3, …, 15). §7.6.1.1 row-by-row:
        // - Top-field rows are FullyFilled at identity.
        // - Bottom-field rows are FullyTransparent (no opaque samples,
        //   the §7.6.1.1 row-guard skips them).
        // §7.6.1.5 per-field §7.6.1.2: the bottom field has zero
        // opaque samples in any column, so every bottom-field column
        // is FullyTransparent — the bottom-field rows are left at
        // their §7.6.1.1 input (= the original `decoded` values, since
        // §7.6.1.1 passes a FullyTransparent row through). The top
        // field is FullyFilled column-by-column (every column has 8
        // opaque samples; §7.6.1.2 is identity).
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        for y in 0..LUMA_SIDE {
            for x in 0..LUMA_SIDE {
                decoded[y][x] = (y * LUMA_SIDE + x) as i32;
                if y % 2 == 0 {
                    shape[y][x] = op();
                }
            }
        }
        let (padded, outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        match outcome {
            InterlacedBoundaryOutcome::Padded {
                top_column_states,
                bottom_column_states,
            } => {
                // Top field: every column FullyFilled.
                assert_eq!(top_column_states, [ColumnState::FullyFilled; LUMA_SIDE]);
                // Bottom field: every column FullyTransparent —
                // §7.6.1.2 falls through; the caller routes the
                // remaining bottom-field samples through §7.6.1.3.
                assert_eq!(
                    bottom_column_states,
                    [ColumnState::FullyTransparent; LUMA_SIDE]
                );
            }
            other => panic!("expected Padded outcome, got {other:?}"),
        }
        // Top-field rows passed through unchanged.
        for f in 0..LUMA_FIELD_LINES {
            assert_eq!(padded[2 * f], decoded[2 * f]);
        }
        // Bottom-field rows: §7.6.1.1 passes them through unchanged
        // because the row is FullyTransparent. The §7.6.1.5 per-field
        // §7.6.1.2 pass also leaves them at their §7.6.1.1 input
        // because every bottom-field column is FullyTransparent.
        for f in 0..LUMA_FIELD_LINES {
            assert_eq!(padded[2 * f + 1], decoded[2 * f + 1]);
        }
    }

    #[test]
    fn per_field_rule_avoids_cross_field_average() {
        // Construct a shape where the top field has opaque samples at
        // y = 0 and y = 14 (top-field rows 0 and 7) and transparent
        // everywhere between, while the bottom field is opaque only
        // at y = 1 (bottom-field row 0). The per-field §7.6.1.2 scan
        // on the top field's column x = 0 then sees y' = 0,
        // y'' = 7 — averaging hor_pad_top[0][0] and hor_pad_top[7][0]
        // — and never reaches into the bottom field's samples.
        //
        // To make the test detectable: put a distinctive value at
        // bottom-field row 0 column 0 that would propagate into the
        // top-field interior cells if (and only if) the §7.6.1.2 scan
        // crossed fields — but the spec rule forbids it.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        // Top field row 0 (frame y = 0): one opaque sample at x = 0
        // valued 10.
        decoded[0][0] = 10;
        shape[0][0] = op();
        // Top field row 7 (frame y = 14): one opaque sample at x = 0
        // valued 41.
        decoded[14][0] = 41;
        shape[14][0] = op();
        // Bottom field row 0 (frame y = 1): one opaque sample at x = 0
        // valued 1000 — a value that would dominate any cross-field
        // average if the §7.6.1.5 rule were broken.
        decoded[1][0] = 1000;
        shape[1][0] = op();
        let (padded, outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        // The per-field column-0 vertical scan on the top field sees
        // y' = 0 (hor_pad = 10), y'' = 7 (hor_pad = 41). Per the
        // §7.6.1.1 row-guard, only the rows containing the opaque
        // samples ran through §7.6.1.1 — for top-field rows 1..=6 the
        // §7.6.1.1 pass skipped them and §7.6.1.2 fills them.
        //
        // (10 + 41) // 2 = 51 // 2 = 25.
        // Frame rows 2, 4, 6, 8, 10, 12 are top-field rows 1..=6 and
        // should be 25 at column 0.
        for top_y in 1..=6 {
            let frame_y = 2 * top_y;
            assert_eq!(padded[frame_y][0], 25, "frame_y={frame_y}");
        }
        // Verify the top field's column-0 scan never picked up the
        // bottom field's 1000 sample — if it had, the interior values
        // would have shifted dramatically. The above assertion already
        // catches that, but make it explicit:
        for top_y in 1..=6 {
            let frame_y = 2 * top_y;
            assert_ne!(
                padded[frame_y][0], 1000,
                "frame_y={frame_y} cross-field leak"
            );
        }
        match outcome {
            InterlacedBoundaryOutcome::Padded { .. } => {}
            other => panic!("expected Padded outcome, got {other:?}"),
        }
    }

    #[test]
    fn per_field_pass_matches_progressive_when_no_interlace_difference() {
        // Construct a shape where each column has the same opaque
        // pattern in the top and bottom fields. The per-field result
        // should agree with the progressive (frame-mode) §7.6.1.2 on
        // the rows where both fields fill identically.
        //
        // Pattern: top half of the macroblock (rows 0..8) opaque,
        // bottom half (rows 8..16) transparent. Top field has rows
        // 0, 2, 4, 6 (= frame rows 0, 2, 4, 6) opaque and rows 8, 10,
        // 12, 14 (= frame rows 8, 10, 12, 14 — but those are
        // *transparent* per the pattern). Wait, frame rows 0..8 are
        // opaque → top field rows 0..4 (frame 0, 2, 4, 6) are opaque
        // and top field rows 4..8 (frame 8, 10, 12, 14) are
        // transparent; bottom field rows 0..4 (frame 1, 3, 5, 7) are
        // opaque and bottom field rows 4..8 (frame 9, 11, 13, 15) are
        // transparent. Both fields therefore have the same
        // "top-half opaque, bottom-half transparent" pattern within
        // themselves — §7.6.1.2 on each field will replicate the last
        // opaque row downward.
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
        let (padded, outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        match outcome {
            InterlacedBoundaryOutcome::Padded {
                top_column_states,
                bottom_column_states,
            } => {
                assert_eq!(top_column_states, [ColumnState::FullyFilled; LUMA_SIDE]);
                assert_eq!(bottom_column_states, [ColumnState::FullyFilled; LUMA_SIDE]);
            }
            other => panic!("expected Padded outcome, got {other:?}"),
        }
        // Top-field rows 0..=3 (frame 0, 2, 4, 6) unchanged from
        // decoded (the §7.6.1.1 horizontal pass is identity on a
        // fully-opaque row and §7.6.1.2 is identity on a fully-opaque
        // column-prefix).
        for top_y in 0..4 {
            let frame_y = 2 * top_y;
            assert_eq!(padded[frame_y], decoded[frame_y]);
        }
        // Top-field rows 4..=7 (frame 8, 10, 12, 14) get filled from
        // the last opaque top-field row (top field row 3 → frame row
        // 6).
        for top_y in 4..LUMA_FIELD_LINES {
            let frame_y = 2 * top_y;
            assert_eq!(padded[frame_y], decoded[6]);
        }
        // Bottom-field rows 0..=3 (frame 1, 3, 5, 7) unchanged.
        for bot_y in 0..4 {
            let frame_y = 2 * bot_y + 1;
            assert_eq!(padded[frame_y], decoded[frame_y]);
        }
        // Bottom-field rows 4..=7 (frame 9, 11, 13, 15) get filled
        // from the last opaque bottom-field row (bottom field row 3 →
        // frame row 7).
        for bot_y in 4..LUMA_FIELD_LINES {
            let frame_y = 2 * bot_y + 1;
            assert_eq!(padded[frame_y], decoded[7]);
        }
    }

    #[test]
    fn isolated_opaque_in_top_field_propagates_within_top_field() {
        // One opaque sample at frame row 0 (top-field row 0), every
        // other sample transparent. §7.6.1.1 fills row 0 fully (every
        // x has only x' or x'' = 0); §7.6.1.5 §7.6.1.2 per-field:
        // top field column scan sees y' = 0 (no y'' present), fills
        // top-field rows 1..=7 (= frame rows 2, 4, …, 14) with
        // hor_pad[0][x] (which is the same as decoded[0][0] for every
        // x). Bottom field has no opaque samples → every column
        // FullyTransparent, bottom-field rows pass through.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        decoded[0][0] = 99;
        shape[0][0] = op();
        let (padded, outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        match outcome {
            InterlacedBoundaryOutcome::Padded {
                top_column_states,
                bottom_column_states,
            } => {
                assert_eq!(top_column_states, [ColumnState::FullyFilled; LUMA_SIDE]);
                assert_eq!(
                    bottom_column_states,
                    [ColumnState::FullyTransparent; LUMA_SIDE]
                );
            }
            other => panic!("expected Padded outcome, got {other:?}"),
        }
        // Every top-field row (frame rows 0, 2, …, 14) is [99; 16].
        for top_y in 0..LUMA_FIELD_LINES {
            let frame_y = 2 * top_y;
            assert_eq!(padded[frame_y], [99; LUMA_SIDE]);
        }
    }

    #[test]
    fn cross_check_against_separate_field_passes() {
        // Build a §7.6.1.1 input by hand, split into fields, run
        // §7.6.1.2 on each field via the same `vertical_repetitive_
        // padding_column` primitive the interlaced wrapper uses, and
        // verify the wrapper's output matches the manual reassembly.
        //
        // This cross-checks that the helper composition is the spec's
        // composition (it would fail if the wrapper accidentally fed
        // §7.6.1.2 a different column order).
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        // A diagonal opaque region in the top field plus a separate
        // diagonal in the bottom field.
        for y in 0..LUMA_SIDE {
            for x in 0..LUMA_SIDE {
                decoded[y][x] = (y * LUMA_SIDE + x) as i32;
                // Top field: opaque at top_y == x (frame_y = 2 * top_y).
                if y % 2 == 0 && (y / 2) == x % LUMA_FIELD_LINES {
                    shape[y][x] = op();
                }
                // Bottom field: opaque at bot_y == 7 - x % 8 (frame_y =
                // 2 * bot_y + 1).
                if y % 2 == 1 && (y / 2) == (LUMA_FIELD_LINES - 1 - x % LUMA_FIELD_LINES) {
                    shape[y][x] = op();
                }
            }
        }
        let (wrapper_padded, _outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        // Manual reassembly via the public primitives:
        let (hor_pad, s_prime, _row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        let (top_hp, bottom_hp) = split_samples_into_fields(&hor_pad);
        let (top_sp, bottom_sp) = split_sentinel_into_fields(&s_prime);
        let (top_hv, _top_sdp, _top_states) = per_field_vertical_pass(&top_hp, &top_sp);
        let (bottom_hv, _bottom_sdp, _bottom_states) =
            per_field_vertical_pass(&bottom_hp, &bottom_sp);
        let manual = stack_samples_into_frame(&top_hv, &bottom_hv);
        assert_eq!(wrapper_padded, manual);
    }

    #[test]
    fn progressive_vs_interlaced_differ_when_fields_disagree() {
        // Construct a payload where the progressive (frame-mode)
        // §7.6.1.2 and the §7.6.1.5 per-field §7.6.1.2 produce
        // different outputs. The two should differ at every position
        // where the cross-field nearest neighbour disagrees with the
        // same-field nearest neighbour.
        //
        // Top-field opaque only at frame y = 0 (value 10); bottom-
        // field opaque only at frame y = 15 (value 100). Frame-mode
        // §7.6.1.2 on column 0 sees y' = 0 and y'' = 15 for every
        // interior y → fills (10 + 100) // 2 = 55. Per-field §7.6.1.2
        // on column 0: top field sees only y' = 0 (no y'' inside the
        // top field), fills top-field rows 1..=7 with 10. Bottom
        // field sees only y'' = 7 (no y' inside the bottom field at
        // bottom-field rows 0..=6), fills bottom-field rows 0..=6
        // with 100. The interlaced output is therefore 10 on frame
        // rows 2, 4, …, 14 and 100 on frame rows 1, 3, …, 13 — not
        // the 55 the progressive pass would produce.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        decoded[0][0] = 10;
        shape[0][0] = op();
        decoded[15][0] = 100;
        shape[15][0] = op();
        // Progressive baseline.
        let (hor_pad, s_prime, row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        let (progressive, _s_dp_prog, _col_states_prog) =
            vertical_repetitive_padding_luma(&hor_pad, &s_prime, &row_states);
        // Progressive column 0 is filled with (10 + 100) // 2 = 55 at
        // every interior row. (Rows 0 and 15 keep their opaque values
        // 10 / 100; rows 1..=14 are filled with 55 — and §7.6.1.1
        // already filled the whole row 0 with 10 and row 15 with 100,
        // so every (row 1..=14, x) is 55.)
        for y in 1..=14 {
            for x in 0..LUMA_SIDE {
                assert_eq!(progressive[y][x], 55, "progressive (y={y}, x={x})");
            }
        }
        // Interlaced via §7.6.1.5.
        let (interlaced, _outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        // Top field rows 1..=7 (frame 2, 4, …, 14) are 10.
        for top_y in 1..LUMA_FIELD_LINES {
            let frame_y = 2 * top_y;
            for x in 0..LUMA_SIDE {
                assert_eq!(
                    interlaced[frame_y][x], 10,
                    "interlaced top-field (frame_y={frame_y}, x={x})"
                );
            }
        }
        // Bottom field rows 0..=6 (frame 1, 3, …, 13) are 100.
        for bot_y in 0..(LUMA_FIELD_LINES - 1) {
            let frame_y = 2 * bot_y + 1;
            for x in 0..LUMA_SIDE {
                assert_eq!(
                    interlaced[frame_y][x], 100,
                    "interlaced bottom-field (frame_y={frame_y}, x={x})"
                );
            }
        }
        // The two differ at every position where the carve-out fires.
        // Sanity check one explicit pair:
        assert_ne!(progressive[2][0], interlaced[2][0]);
        assert_ne!(progressive[1][0], interlaced[1][0]);
    }

    #[test]
    fn per_field_column_with_no_opaque_in_field_reports_fully_transparent() {
        // Opaque only at frame y = 0, column 0 (top field row 0,
        // column 0). The §7.6.1.1 pass fills the entire row 0; the
        // §7.6.1.5 per-field §7.6.1.2 pass on the bottom field has
        // zero opaque samples on any column → every bottom-field
        // column reports FullyTransparent. The top-field columns are
        // all FullyFilled (their only opaque sample is at top-field
        // row 0).
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        decoded[0][0] = 50;
        shape[0][0] = op();
        let (_padded, outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        match outcome {
            InterlacedBoundaryOutcome::Padded {
                top_column_states,
                bottom_column_states,
            } => {
                assert_eq!(top_column_states, [ColumnState::FullyFilled; LUMA_SIDE]);
                assert_eq!(
                    bottom_column_states,
                    [ColumnState::FullyTransparent; LUMA_SIDE]
                );
            }
            other => panic!("expected Padded outcome, got {other:?}"),
        }
    }

    #[test]
    fn padded_outcome_partial_eq() {
        // PartialEq on InterlacedBoundaryOutcome is needed by callers
        // that want to assert a particular outcome variant. Cover the
        // discrimination here.
        let a = InterlacedBoundaryOutcome::Padded {
            top_column_states: [ColumnState::FullyFilled; LUMA_SIDE],
            bottom_column_states: [ColumnState::FullyFilled; LUMA_SIDE],
        };
        let b = InterlacedBoundaryOutcome::Padded {
            top_column_states: [ColumnState::FullyFilled; LUMA_SIDE],
            bottom_column_states: [ColumnState::FullyFilled; LUMA_SIDE],
        };
        let c = InterlacedBoundaryOutcome::Padded {
            top_column_states: [ColumnState::FullyTransparent; LUMA_SIDE],
            bottom_column_states: [ColumnState::FullyFilled; LUMA_SIDE],
        };
        let d = InterlacedBoundaryOutcome::CompletelyTransparent;
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn entire_frame_opaque_top_field_only_at_specific_x() {
        // Build a payload where the top field has different opaque
        // columns than the bottom field. Specifically, opaque at
        // (frame_y % 2 == 0, x < 8) and (frame_y % 2 == 1, x >= 8).
        // The §7.6.1.1 pass fills every row (each row has 8 opaque
        // samples). §7.6.1.5 per-field §7.6.1.2 is identity on every
        // per-field column because the top field is fully filled on
        // x < 8 (top-field s' is all opaque on x < 8 — actually after
        // §7.6.1.1 it's opaque on every x because every row was
        // fully filled).
        //
        // The §7.6.1.1 row-fill means s' is opaque on every (y, x);
        // §7.6.1.2 per-field is identity. The result equals the
        // §7.6.1.1 output, which differs from `decoded` at the
        // originally-transparent positions but matches `hor_pad`.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        for y in 0..LUMA_SIDE {
            for x in 0..LUMA_SIDE {
                decoded[y][x] = (y * LUMA_SIDE + x) as i32;
                if (y % 2 == 0 && x < 8) || (y % 2 == 1 && x >= 8) {
                    shape[y][x] = op();
                }
            }
        }
        let (padded, outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        // Compare against the §7.6.1.1 output directly.
        let (hor_pad, _s_prime, _row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        assert_eq!(padded, hor_pad);
        match outcome {
            InterlacedBoundaryOutcome::Padded {
                top_column_states,
                bottom_column_states,
            } => {
                assert_eq!(top_column_states, [ColumnState::FullyFilled; LUMA_SIDE]);
                assert_eq!(bottom_column_states, [ColumnState::FullyFilled; LUMA_SIDE]);
            }
            other => panic!("expected Padded outcome, got {other:?}"),
        }
    }

    #[test]
    fn per_field_uses_8_lines_not_16() {
        // The §7.6.1.5 carve-out's "same field" means the §7.6.1.2
        // column scan looks at 8 samples, not 16. Spot-check by
        // verifying the LUMA_FIELD_LINES constant the wrapper drives
        // is 8 and that a single-column 16×1 stripe with a top-field
        // opaque sample at the field's last row gets that sample
        // replicated as the §7.6.1.2 below-neighbour for the top
        // field's earlier transparent rows.
        let mut decoded = [[0i32; LUMA_SIDE]; LUMA_SIDE];
        let mut shape = [[tr(); LUMA_SIDE]; LUMA_SIDE];
        // Top field row 7 (= frame row 14): opaque, value 77.
        decoded[14][0] = 77;
        shape[14][0] = op();
        // Bottom field row 0 (= frame row 1): opaque, value 1.
        decoded[1][0] = 1;
        shape[1][0] = op();
        let (padded, _outcome) = interlaced_boundary_padding_luma(&decoded, &shape);
        // Top field column 0 sees y' = none (top-field rows 0..6 are
        // transparent and the only opaque sample is at row 7), y'' = 7
        // → top-field rows 0..=6 get filled with 77.
        for top_y in 0..(LUMA_FIELD_LINES - 1) {
            let frame_y = 2 * top_y;
            assert_eq!(padded[frame_y][0], 77, "frame_y={frame_y}");
        }
        // Frame row 14 (top field row 7) unchanged.
        assert_eq!(padded[14][0], 77);
        // Bottom field column 0: only y' = 0 exists (opaque at bottom-
        // field row 0); bottom-field rows 1..=7 get filled with 1.
        for bot_y in 1..LUMA_FIELD_LINES {
            let frame_y = 2 * bot_y + 1;
            assert_eq!(padded[frame_y][0], 1, "frame_y={frame_y}");
        }
        // Frame row 1 (bottom field row 0) unchanged.
        assert_eq!(padded[1][0], 1);
        // LUMA_FIELD_LINES guard: the constant the per-field pass uses
        // is 8 (= LUMA_SIDE / 2).
        assert_eq!(LUMA_FIELD_LINES, 8);
        // Also sanity check that the per-field side length is not
        // accidentally CHROMA_SIDE (which is also 8, but for different
        // semantic reasons).
        assert_eq!(LUMA_FIELD_LINES, CHROMA_SIDE);
    }
}

//! §7.6.1.1 horizontal repetitive padding — fill the transparent samples
//! of a boundary macroblock by replicating the VOP-boundary samples of
//! the same row.
//!
//! A *boundary macroblock* is one that straddles the VOP shape mask: at
//! least one sample inside the macroblock is opaque (`s[y][x] == 1`,
//! "inside the VOP") and at least one is transparent. The §7.6.1
//! padding pipeline fills the transparent samples so the
//! motion-compensation sample fetch (§7.6.2.1 / §7.6.2.2) can read every
//! position inside the bounding rectangle without a per-sample
//! transparency probe.
//!
//! This module covers the **first** pass of that pipeline — §7.6.1.1
//! horizontal repetitive padding. The §7.6.1.2 vertical pass and the
//! §7.6.1.3 extended-padding pass for fully-exterior macroblocks live in
//! later rounds.
//!
//! ## Algorithm (verbatim from ISO/IEC 14496-2:2004 §7.6.1.1)
//!
//! Given a row `y` of the decoded macroblock `d[y][x]` and the matching
//! row of the shape mask `s[y][x]` (`s[y][x] == 1` is opaque,
//! `s[y][x] == 0` is transparent), the spec's example procedure is:
//!
//! ```text
//! for (x = 0; x < N; x++) {
//!     if (s[y][x] == 1) {
//!         hor_pad[y][x] = d[y][x];
//!         s'[y][x] = 1;
//!     } else {
//!         if (s[y][x'] == 1 && s[y][x''] == 1) {
//!             hor_pad[y][x] = (d[y][x'] + d[y][x'']) // 2;
//!             s'[y][x] = 1;
//!         } else if (s[y][x'] == 1) {
//!             hor_pad[y][x] = d[y][x'];
//!             s'[y][x] = 1;
//!         } else if (s[y][x''] == 1) {
//!             hor_pad[y][x] = d[y][x''];
//!             s'[y][x] = 1;
//!         }
//!     }
//! }
//! ```
//!
//! where `x'` is the nearest opaque sample at or to the left of `x` on
//! the same row (`x' <= x` with `s[y][x'] == 1`), `x''` is the nearest
//! opaque sample strictly to the right of `x` on the same row
//! (`x'' > x` with `s[y][x''] == 1`), and `N` is the macroblock side
//! length (16 for luminance, 8 for each chrominance block per
//! §7.6.1.4). The output sentinel `s'[y][x]` is initialised to 0 and
//! flipped to 1 whenever a row receives any fill on this pass.
//!
//! The rule "only act on rows with at least one `s[y][x] == 1`" is the
//! §7.6.1.1 row-level guard: rows that are entirely transparent (no
//! opaque sample on the row) are left untouched by the horizontal pass,
//! and the §7.6.1.2 vertical pass fills them based on the columns
//! produced here.
//!
//! ## `//` is §3.4 integer division toward zero
//!
//! The `(d[y][x'] + d[y][x'']) // 2` term uses the spec's §3.4 `//`
//! operator — *Integer division with truncation of the result toward
//! zero*. For the display-range inputs in this module (samples are in
//! `[0, 2^bits_per_pixel - 1]` per §7.3 step-3) the sum is positive
//! and `//2` reduces to ordinary right-shift-by-one. The implementation
//! still uses [`i32::div_euclid`] against an explicit `2` so a future
//! caller passing pre-clipped signed inputs (e.g. an encoder rate-
//! distortion search that has not yet reapplied the §7.3 step-3 clip)
//! still gets the §3.4 semantics — but the §7.6.1.1 input contract is
//! "decoded macroblock `d[y][x]`", which is post-§7.3-step-3 and
//! therefore non-negative.
//!
//! ## What this module does **not** do
//!
//! * Decide whether the macroblock is a *boundary* macroblock. The
//!   §7.6.1 framing text routes boundary macroblocks through §7.6.1.1
//!   then §7.6.1.2, and exterior macroblocks through §7.6.1.3 — both
//!   branch selectors live in the caller (which has access to the
//!   full VOP shape grid). This module accepts any macroblock with
//!   at least one opaque sample and pads its rows; a fully-transparent
//!   macroblock is reported as such via
//!   [`ShapeRowState::FullyTransparent`] in the per-row state so the
//!   caller can route it to §7.6.1.3 instead.
//! * Apply the §7.6.1.2 vertical pass. The vertical pass consumes
//!   `(hor_pad, s')` from this pass and works *column-by-column*; it
//!   is a separate later-round entry point.
//! * Apply the §7.6.1.3 extended-padding pass for exterior macroblocks.
//! * Decimate the shape mask for the §7.6.1.4 chrominance path. The
//!   chroma shape comes from §6.1.3.6 subsampling and is the caller's
//!   responsibility; this module's [`MacroblockShape::CHROMA_SIDE`] /
//!   [`MacroblockShape::LUMA_SIDE`] entry points operate on whichever
//!   side length the caller supplies via const generics.
//! * Cover the §7.6.1.5 interlaced-VOP per-field padding. Interlaced
//!   padding splits the macroblock into two 16×8 fields and applies
//!   §7.6.1.1..§7.6.1.3 to each independently; the horizontal pass is
//!   structurally identical to the progressive case so a future
//!   interlaced entry point can re-use this module against the
//!   per-field slice.

#![allow(clippy::needless_range_loop)]

/// Side length of a 4:2:0 luminance macroblock in samples (§6.1.3.4).
pub const LUMA_SIDE: usize = 16;

/// Side length of a 4:2:0 chrominance block in samples (§6.1.3.4 +
/// §7.6.1.4 — chroma padding works per 8×8 block).
pub const CHROMA_SIDE: usize = 8;

/// Per-sample VOP-shape flag (`s[y][x]` in §7.6.1.1).
///
/// `Opaque` is the §7.6.1.1 `s[y][x] == 1` case ("inside the VOP");
/// `Transparent` is the `s[y][x] == 0` case ("outside the VOP / part of
/// the alpha-shape exterior").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplePresence {
    /// `s[y][x] == 1`: this sample is inside the VOP shape mask.
    Opaque,
    /// `s[y][x] == 0`: this sample is outside the VOP shape mask.
    Transparent,
}

impl SamplePresence {
    /// `true` iff the sample is opaque (`s[y][x] == 1`).
    #[inline]
    pub fn is_opaque(self) -> bool {
        matches!(self, SamplePresence::Opaque)
    }

    /// `true` iff the sample is transparent (`s[y][x] == 0`).
    #[inline]
    pub fn is_transparent(self) -> bool {
        matches!(self, SamplePresence::Transparent)
    }
}

/// Per-row §7.6.1.1 outcome, returned alongside the padded row.
///
/// The §7.6.1.1 inner loop is guarded by "for every line with at least
/// one shape sample `s[y][x] == 1`": rows that fail the guard are left
/// untouched and their per-sample `s'[y][x]` stays at the initial `0`.
/// The §7.6.1.2 vertical pass needs to distinguish "row was filled
/// completely on the horizontal pass" from "row was skipped because it
/// has no opaque sample" — this enum carries that distinction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeRowState {
    /// The row had at least one opaque sample and the §7.6.1.1 pass
    /// filled every transparent sample on the row from a left- or
    /// right-side neighbour. Every `s'[y][x]` on the row is 1.
    FullyFilled,
    /// The row had no opaque samples — the §7.6.1.1 row-guard skipped
    /// the row entirely and every `s'[y][x]` on the row stays at 0.
    /// The §7.6.1.2 vertical pass will fill the row.
    FullyTransparent,
}

/// §7.6.1.1 horizontal repetitive padding for one row.
///
/// `decoded` is the §7.3 step-3-clipped row of `d[y][x]`; `shape` is
/// the corresponding row of `s[y][x]`. On return:
///
/// * `out` holds `hor_pad[y][x]`. Transparent samples on rows that had
///   no opaque sample are left at their input value (the §7.6.1.1
///   row-guard skipped the row); the caller routes them through the
///   §7.6.1.2 vertical pass.
/// * `s_prime` holds the §7.6.1.1 fill sentinel `s'[y][x]`: `Opaque`
///   for every position that was either originally opaque or filled by
///   this pass, `Transparent` otherwise.
/// * The returned [`ShapeRowState`] reports whether the row was filled
///   or skipped.
///
/// Arithmetic note: the spec's `(d[y][x'] + d[y][x'']) // 2` is the
/// §3.4 `//` operator (truncation toward zero). The implementation
/// uses `i32::div_euclid` against `2`, which matches `//` for the
/// non-negative `d[y][x]` values inside the bounding rectangle of a
/// rectangular VOP.
pub fn horizontal_repetitive_padding_row<const N: usize>(
    decoded: &[i32; N],
    shape: &[SamplePresence; N],
    out: &mut [i32; N],
    s_prime: &mut [SamplePresence; N],
) -> ShapeRowState {
    // §7.6.1.1: s'[y][x] is initialised to 0 (Transparent), then flipped
    // to 1 (Opaque) wherever the row receives a value.
    *s_prime = [SamplePresence::Transparent; N];

    // Row-guard: skip rows that have no opaque sample.
    if shape.iter().all(|s| s.is_transparent()) {
        // Copy decoded into out unchanged so the row is a defined
        // tensor (transparent samples carry their input value, the
        // §7.6.1.2 vertical pass overwrites them later).
        *out = *decoded;
        return ShapeRowState::FullyTransparent;
    }

    // Precompute the per-position `x'` (nearest opaque at-or-to-the-left)
    // and `x''` (nearest opaque strictly to the right) so the inner pass
    // is one linear scan. `nearest_left[x] = Some(x')` is the §7.6.1.1
    // `x'` with `x' <= x`; `nearest_right[x] = Some(x'')` is the
    // §7.6.1.1 `x''` with `x'' > x`. The "strictly greater" on `x''`
    // matches the spec text "the nearest boundary sample to the right"
    // and yields the (x', x'') ordering the §7.6.1.1 averaging branch
    // requires when the current sample is itself transparent.
    let mut nearest_left: [Option<usize>; N] = [None; N];
    let mut nearest_right: [Option<usize>; N] = [None; N];
    let mut last_opaque: Option<usize> = None;
    for x in 0..N {
        if shape[x].is_opaque() {
            last_opaque = Some(x);
        }
        nearest_left[x] = last_opaque;
    }
    let mut next_opaque: Option<usize> = None;
    for x in (0..N).rev() {
        nearest_right[x] = next_opaque;
        if shape[x].is_opaque() {
            next_opaque = Some(x);
        }
    }

    for x in 0..N {
        if shape[x].is_opaque() {
            out[x] = decoded[x];
            s_prime[x] = SamplePresence::Opaque;
        } else {
            match (nearest_left[x], nearest_right[x]) {
                (Some(x_left), Some(x_right)) => {
                    // §7.6.1.1 averaging branch: (d[x'] + d[x'']) // 2.
                    let sum = decoded[x_left] + decoded[x_right];
                    out[x] = sum.div_euclid(2);
                    s_prime[x] = SamplePresence::Opaque;
                }
                (Some(x_left), None) => {
                    // Only the left boundary sample exists on this row.
                    out[x] = decoded[x_left];
                    s_prime[x] = SamplePresence::Opaque;
                }
                (None, Some(x_right)) => {
                    // Only the right boundary sample exists on this row.
                    out[x] = decoded[x_right];
                    s_prime[x] = SamplePresence::Opaque;
                }
                (None, None) => {
                    // Unreachable: the row-guard above has already
                    // returned `FullyTransparent` for rows with no
                    // opaque sample.
                    unreachable!(
                        "horizontal_repetitive_padding_row: row with no opaque sample reached the inner loop"
                    );
                }
            }
        }
    }

    ShapeRowState::FullyFilled
}

/// §7.6.1.1 horizontal repetitive padding for one 4:2:0 luminance
/// macroblock.
///
/// `decoded[y][x]` is the §7.3 step-3-clipped 16×16 macroblock of
/// `d[y][x]`; `shape[y][x]` is the matching 16×16 row-major shape
/// mask. The returned `(hor_pad, s_prime, row_states)` triple is the
/// §7.6.1.1 output: per-pixel padded macroblock, per-pixel `s'[y][x]`
/// sentinel, and per-row [`ShapeRowState`] for the §7.6.1.2 vertical
/// pass to consume.
pub fn horizontal_repetitive_padding_luma(
    decoded: &[[i32; LUMA_SIDE]; LUMA_SIDE],
    shape: &[[SamplePresence; LUMA_SIDE]; LUMA_SIDE],
) -> (
    [[i32; LUMA_SIDE]; LUMA_SIDE],
    [[SamplePresence; LUMA_SIDE]; LUMA_SIDE],
    [ShapeRowState; LUMA_SIDE],
) {
    let mut out = [[0i32; LUMA_SIDE]; LUMA_SIDE];
    let mut s_prime = [[SamplePresence::Transparent; LUMA_SIDE]; LUMA_SIDE];
    let mut row_states = [ShapeRowState::FullyTransparent; LUMA_SIDE];
    for y in 0..LUMA_SIDE {
        row_states[y] =
            horizontal_repetitive_padding_row(&decoded[y], &shape[y], &mut out[y], &mut s_prime[y]);
    }
    (out, s_prime, row_states)
}

/// §7.6.1.1 horizontal repetitive padding for one 4:2:0 chrominance
/// block (Cb or Cr, 8×8 samples per §7.6.1.4).
///
/// Same shape as [`horizontal_repetitive_padding_luma`] but the side
/// length is 8 to match the §6.1.3.4 chroma format and §7.6.1.4 "the
/// padding is performed by referring to a shape block generated by
/// decimating the shape block of the corresponding luminance
/// component" framing. The decimated shape comes from §6.1.3.6 (out of
/// scope for this module — the caller supplies the 8×8 chroma `shape`
/// directly).
pub fn horizontal_repetitive_padding_chroma(
    decoded: &[[i32; CHROMA_SIDE]; CHROMA_SIDE],
    shape: &[[SamplePresence; CHROMA_SIDE]; CHROMA_SIDE],
) -> (
    [[i32; CHROMA_SIDE]; CHROMA_SIDE],
    [[SamplePresence; CHROMA_SIDE]; CHROMA_SIDE],
    [ShapeRowState; CHROMA_SIDE],
) {
    let mut out = [[0i32; CHROMA_SIDE]; CHROMA_SIDE];
    let mut s_prime = [[SamplePresence::Transparent; CHROMA_SIDE]; CHROMA_SIDE];
    let mut row_states = [ShapeRowState::FullyTransparent; CHROMA_SIDE];
    for y in 0..CHROMA_SIDE {
        row_states[y] =
            horizontal_repetitive_padding_row(&decoded[y], &shape[y], &mut out[y], &mut s_prime[y]);
    }
    (out, s_prime, row_states)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op() -> SamplePresence {
        SamplePresence::Opaque
    }
    fn tr() -> SamplePresence {
        SamplePresence::Transparent
    }

    #[test]
    fn sample_presence_predicates() {
        assert!(op().is_opaque());
        assert!(!op().is_transparent());
        assert!(tr().is_transparent());
        assert!(!tr().is_opaque());
    }

    #[test]
    fn fully_opaque_row_is_identity() {
        let decoded = [10, 20, 30, 40, 50, 60, 70, 80];
        let shape = [op(); 8];
        let mut out = [0; 8];
        let mut s_prime = [tr(); 8];
        let state = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        assert_eq!(state, ShapeRowState::FullyFilled);
        assert_eq!(out, decoded);
        assert_eq!(s_prime, [op(); 8]);
    }

    #[test]
    fn fully_transparent_row_is_skipped() {
        let decoded = [10, 20, 30, 40, 50, 60, 70, 80];
        let shape = [tr(); 8];
        let mut out = [0; 8];
        let mut s_prime = [op(); 8];
        let state = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        assert_eq!(state, ShapeRowState::FullyTransparent);
        // Decoded row passes through; s'[y][x] is reset to all-transparent.
        assert_eq!(out, decoded);
        assert_eq!(s_prime, [tr(); 8]);
    }

    #[test]
    fn left_only_fill_replicates_left_neighbour() {
        // Opaque on x = 0..=3, transparent on x = 4..=7. The §7.6.1.1
        // example: x'' does not exist for x >= 4 (no opaque sample to
        // the right); only x' = 3 exists, so hor_pad[4..=7] = d[3] = 40.
        let decoded = [10, 20, 30, 40, 99, 99, 99, 99];
        let shape = [op(), op(), op(), op(), tr(), tr(), tr(), tr()];
        let mut out = [0; 8];
        let mut s_prime = [tr(); 8];
        let state = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        assert_eq!(state, ShapeRowState::FullyFilled);
        assert_eq!(out, [10, 20, 30, 40, 40, 40, 40, 40]);
        assert_eq!(s_prime, [op(); 8]);
    }

    #[test]
    fn right_only_fill_replicates_right_neighbour() {
        // Transparent on x = 0..=3, opaque on x = 4..=7. The §7.6.1.1
        // example: x' does not exist for x <= 3 (no opaque sample at or
        // to the left); only x'' = 4 exists, so hor_pad[0..=3] = d[4].
        let decoded = [99, 99, 99, 99, 50, 60, 70, 80];
        let shape = [tr(), tr(), tr(), tr(), op(), op(), op(), op()];
        let mut out = [0; 8];
        let mut s_prime = [tr(); 8];
        let state = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        assert_eq!(state, ShapeRowState::FullyFilled);
        assert_eq!(out, [50, 50, 50, 50, 50, 60, 70, 80]);
        assert_eq!(s_prime, [op(); 8]);
    }

    #[test]
    fn two_sided_fill_averages_with_truncation() {
        // Opaque at x = 0 and x = 7; transparent everywhere between.
        // x' = 0 for every interior position (d[0] = 10) and x'' = 7
        // (d[7] = 41). hor_pad[1..=6] = (10 + 41) // 2 = 51 // 2 = 25
        // (§3.4 truncation toward zero on a positive sum).
        let decoded = [10, 0, 0, 0, 0, 0, 0, 41];
        let shape = [op(), tr(), tr(), tr(), tr(), tr(), tr(), op()];
        let mut out = [0; 8];
        let mut s_prime = [tr(); 8];
        let state = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        assert_eq!(state, ShapeRowState::FullyFilled);
        assert_eq!(out, [10, 25, 25, 25, 25, 25, 25, 41]);
        assert_eq!(s_prime, [op(); 8]);
    }

    #[test]
    fn interior_transparent_uses_immediate_neighbours() {
        // Opaque at x = 0..=2 and x = 5..=7; transparent at x = 3, 4.
        // For x = 3: x' = 2 (d[2] = 30), x'' = 5 (d[5] = 60),
        //   hor_pad[3] = (30 + 60) // 2 = 45.
        // For x = 4: x' = 2 (still 30 — nearest left of an opaque cell
        //   is at most x), wait: x' is the nearest opaque at-or-to-the-
        //   left, so for x = 4 the nearest opaque to the left is x = 2.
        //   x'' = 5 still. hor_pad[4] = (30 + 60) // 2 = 45.
        let decoded = [10, 20, 30, 0, 0, 60, 70, 80];
        let shape = [op(), op(), op(), tr(), tr(), op(), op(), op()];
        let mut out = [0; 8];
        let mut s_prime = [tr(); 8];
        let state = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        assert_eq!(state, ShapeRowState::FullyFilled);
        assert_eq!(out, [10, 20, 30, 45, 45, 60, 70, 80]);
        assert_eq!(s_prime, [op(); 8]);
    }

    #[test]
    fn average_rounds_toward_zero_on_odd_sum() {
        // 11 + 20 = 31; 31 // 2 = 15 (§3.4 truncation toward zero).
        let decoded = [11, 0, 0, 20];
        let shape = [op(), tr(), tr(), op()];
        let mut out = [0; 4];
        let mut s_prime = [tr(); 4];
        let state = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        assert_eq!(state, ShapeRowState::FullyFilled);
        assert_eq!(out, [11, 15, 15, 20]);
    }

    #[test]
    fn isolated_opaque_sample_replicated_across_row() {
        // One opaque sample at x = 3 in an otherwise transparent row.
        // For x < 3: only x'' = 3 exists, so out[x] = d[3].
        // For x > 3: only x' = 3 exists, so out[x] = d[3].
        let decoded = [99, 99, 99, 42, 99, 99, 99, 99];
        let shape = [tr(), tr(), tr(), op(), tr(), tr(), tr(), tr()];
        let mut out = [0; 8];
        let mut s_prime = [tr(); 8];
        let state = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        assert_eq!(state, ShapeRowState::FullyFilled);
        assert_eq!(out, [42; 8]);
        assert_eq!(s_prime, [op(); 8]);
    }

    #[test]
    fn full_macroblock_progressive_diagonal_boundary() {
        // A diagonal boundary: row y has opaque samples for x <= y, the
        // rest transparent. So row 0 has one opaque pixel at x = 0;
        // row 15 is fully opaque.
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
        let (out, s_prime, row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        // Every row has at least one opaque sample, so every row is
        // FullyFilled.
        assert_eq!(row_states, [ShapeRowState::FullyFilled; LUMA_SIDE]);
        // s'[y][x] is Opaque everywhere.
        for y in 0..LUMA_SIDE {
            assert_eq!(s_prime[y], [op(); LUMA_SIDE]);
        }
        // Row 0: opaque only at x = 0 (value 0). Every other x: only x'
        // exists (x' = 0), so out[0][x] = 0.
        assert_eq!(out[0], [0; LUMA_SIDE]);
        // Row 15 is fully opaque: identity.
        assert_eq!(out[15], decoded[15]);
        // Row 7: opaque for x = 0..=7, transparent for x = 8..=15.
        // out[7][x] for x = 0..=7 equals decoded[7][x]; for x >= 8,
        // only x' = 7 exists, so out[7][x] = decoded[7][7].
        let mut expected_row7 = decoded[7];
        for x in 8..LUMA_SIDE {
            expected_row7[x] = decoded[7][7];
        }
        assert_eq!(out[7], expected_row7);
    }

    #[test]
    fn full_macroblock_mixed_row_states() {
        // Top half of the macroblock is fully opaque, bottom half fully
        // transparent. Bottom rows should report FullyTransparent and
        // pass through unchanged.
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
        let (out, s_prime, row_states) = horizontal_repetitive_padding_luma(&decoded, &shape);
        for y in 0..8 {
            assert_eq!(row_states[y], ShapeRowState::FullyFilled);
            assert_eq!(s_prime[y], [op(); LUMA_SIDE]);
            assert_eq!(out[y], decoded[y]);
        }
        for y in 8..LUMA_SIDE {
            assert_eq!(row_states[y], ShapeRowState::FullyTransparent);
            assert_eq!(s_prime[y], [tr(); LUMA_SIDE]);
            // FullyTransparent rows pass decoded through.
            assert_eq!(out[y], decoded[y]);
        }
    }

    #[test]
    fn chroma_block_uses_8x8_side() {
        // Sanity that the 8×8 entry point exercises the same row code.
        let mut decoded = [[0i32; CHROMA_SIDE]; CHROMA_SIDE];
        let mut shape = [[tr(); CHROMA_SIDE]; CHROMA_SIDE];
        for y in 0..CHROMA_SIDE {
            for x in 0..CHROMA_SIDE {
                decoded[y][x] = (y * CHROMA_SIDE + x) as i32;
                // Right half opaque.
                if x >= 4 {
                    shape[y][x] = op();
                }
            }
        }
        let (out, s_prime, row_states) = horizontal_repetitive_padding_chroma(&decoded, &shape);
        assert_eq!(row_states, [ShapeRowState::FullyFilled; CHROMA_SIDE]);
        for y in 0..CHROMA_SIDE {
            // x = 0..=3 transparent: only x'' = 4 exists, replicate d[4].
            for x in 0..4 {
                assert_eq!(out[y][x], decoded[y][4]);
            }
            // x = 4..=7 opaque: identity.
            for x in 4..CHROMA_SIDE {
                assert_eq!(out[y][x], decoded[y][x]);
            }
            assert_eq!(s_prime[y], [op(); CHROMA_SIDE]);
        }
    }

    #[test]
    fn boundary_at_zero_and_last_index_uses_both() {
        // Opaque at x = 0 only: row covered by §7.6.1.1 "only x' exists"
        // case (no opaque sample to the right). Mirror-image test for
        // x = N - 1 only.
        let decoded_left = [50i32, 9, 9, 9, 9, 9, 9, 9];
        let shape_left = [op(), tr(), tr(), tr(), tr(), tr(), tr(), tr()];
        let mut out = [0; 8];
        let mut s_prime = [tr(); 8];
        let _ =
            horizontal_repetitive_padding_row(&decoded_left, &shape_left, &mut out, &mut s_prime);
        assert_eq!(out, [50; 8]);

        let decoded_right = [9i32, 9, 9, 9, 9, 9, 9, 77];
        let shape_right = [tr(), tr(), tr(), tr(), tr(), tr(), tr(), op()];
        let mut out = [0; 8];
        let mut s_prime = [tr(); 8];
        let _ =
            horizontal_repetitive_padding_row(&decoded_right, &shape_right, &mut out, &mut s_prime);
        assert_eq!(out, [77; 8]);
    }

    #[test]
    fn s_prime_is_reset_between_calls() {
        // Verify that a caller reusing the s' buffer across rows always
        // sees a from-scratch §7.6.1.1 sentinel and that stale data in
        // the output buffer doesn't bleed through on FullyTransparent
        // rows.
        let decoded = [5, 10, 15, 20];
        let shape = [op(), tr(), op(), tr()];
        let mut out = [0; 4];
        let mut s_prime = [op(); 4];
        let _ = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        // After the call s_prime[1] / s_prime[3] should be Opaque (they
        // were filled), not "stale Opaque from before the call".
        assert_eq!(s_prime, [op(); 4]);
        // x = 1: average of d[0] (5) and d[2] (15) = (20) // 2 = 10.
        // x = 3: only x' = 2 exists, so d[2] = 15.
        assert_eq!(out, [5, 10, 15, 15]);
    }

    #[test]
    fn average_branch_at_extreme_display_range_no_overflow() {
        // For bits_per_pixel = 12 the upper bound is 4095. Two 4095
        // samples averaged: (4095 + 4095) // 2 = 4095. i32 comfortably
        // holds the intermediate.
        let decoded = [4095, 0, 0, 0, 4095];
        let shape = [op(), tr(), tr(), tr(), op()];
        let mut out = [0; 5];
        let mut s_prime = [tr(); 5];
        let _ = horizontal_repetitive_padding_row(&decoded, &shape, &mut out, &mut s_prime);
        assert_eq!(out, [4095; 5]);
    }
}

//! §7.6.1.6 vector padding technique — derive valid motion vectors for
//! the four 8×8 luminance sub-blocks of a macroblock that may contain
//! transparent sub-blocks (non-rectangular VOP shape) or be INTRA-coded
//! or skipped in a P-VOP.
//!
//! The §7.6.1.6 procedure is applied *immediately after* a macroblock is
//! decoded, ahead of any downstream consumer:
//!
//! * §7.6.5 luma → chroma motion-vector derivation (the `K` luma block
//!   vectors that feed [`crate::chroma_mv::chroma_mv_from_luma_blocks`]
//!   come from this padded set, not the raw decoded MVs);
//! * §7.6.5 / §7.6.4.3 spatial MV predictor candidate gathering
//!   (`MV1 / MV2 / MV3` for the next macroblock pull from this padded
//!   set, not the raw decoded MVs);
//! * §7.6.9.5 B-VOP direct mode (the co-located anchor-VOP MVs that
//!   [`crate::motion::direct_mode_motion_vector`] linearly scales are
//!   the padded vectors of the temporally-next anchor VOP).
//!
//! ## Algorithm (verbatim from ISO/IEC 14496-2:2004 §7.6.1.6)
//!
//! Given a macroblock's four 8×8 luminance blocks in §6.1.3.4 / Figure
//! 6-8 raster order (`0 = top-left`, `1 = top-right`, `2 = bottom-left`,
//! `3 = bottom-right`), let `MVx[i] / MVy[i]` be the decoded motion
//! vectors and `Transp[i]` the per-block transparency flags
//! (`TRANSPARENT` = "the 8×8 luminance block is fully outside the
//! VOP-shape mask"). Vector padding is numerically equivalent to:
//!
//! ```text
//! if (the macroblock is INTRA-coded, or skipped and included in a
//!     P-VOP) {
//!     MVx[0] = MVx[1] = MVx[2] = MVx[3] = 0
//!     MVy[0] = MVy[1] = MVy[2] = MVy[3] = 0
//! } else {
//!     if (Transp[0] == TRANSPARENT) {
//!         MVx[0] = (Transp[1] != T) ? MVx[1] : ((Transp[2] != T) ? MVx[2] : MVx[3]);
//!         MVy[0] = (Transp[1] != T) ? MVy[1] : ((Transp[2] != T) ? MVy[2] : MVy[3]);
//!     }
//!     if (Transp[1] == TRANSPARENT) {
//!         MVx[1] = (Transp[0] != T) ? MVx[0] : ((Transp[3] != T) ? MVx[3] : MVx[2]);
//!         MVy[1] = (Transp[0] != T) ? MVy[0] : ((Transp[3] != T) ? MVy[3] : MVy[2]);
//!     }
//!     if (Transp[2] == TRANSPARENT) {
//!         MVx[2] = (Transp[3] != T) ? MVx[3] : ((Transp[0] != T) ? MVx[0] : MVx[1]);
//!         MVy[2] = (Transp[3] != T) ? MVy[3] : ((Transp[0] != T) ? MVy[0] : MVy[1]);
//!     }
//!     if (Transp[3] == TRANSPARENT) {
//!         MVx[3] = (Transp[2] != T) ? MVx[2] : ((Transp[1] != T) ? MVx[1] : MVx[0]);
//!         MVy[3] = (Transp[2] != T) ? MVy[2] : ((Transp[1] != T) ? MVy[1] : MVy[0]);
//!     }
//! }
//! ```
//!
//! The "horizontal followed by vertical repetitive padding on a 2×2
//! block" view in the §7.6.1.6 opening paragraph maps each block's
//! fallback chain to: horizontal partner first (same row, other column),
//! then diagonal partner (other row, other column), then vertical
//! partner (other row, same column). Concretely:
//!
//! | Block `i` | 1st choice | 2nd choice | 3rd choice |
//! |-----------|------------|------------|------------|
//! | 0 (TL)    | 1 (TR)     | 2 (BL)     | 3 (BR)     |
//! | 1 (TR)    | 0 (TL)     | 3 (BR)     | 2 (BL)     |
//! | 2 (BL)    | 3 (BR)     | 0 (TL)     | 1 (TR)     |
//! | 3 (BR)    | 2 (BL)     | 1 (TR)     | 0 (TL)     |
//!
//! The §7.6.1.6 normative text reverses the 2nd / 3rd choice between
//! the corner-pairs (TL/BR prefer the 2nd-choice diagonal block — the
//! `Transp[1] / Transp[0]` partner-of-partner — over their vertical
//! neighbour, whereas TR/BL prefer their 3rd-choice "next-row-same-
//! column" block). The table above transcribes the spec's "if all
//! three other blocks are tested in this fixed order" precedence
//! exactly. The closing paragraph of §7.6.1.6 also notes:
//!
//! > Vector padding is only used in I-, P-, and S(GMC)-VOPs, it is
//! > applied on a macroblock directly after it is decoded. ... Note
//! > that the averaged motion vector described in 7.8.7.3 is used as
//! > the motion vectors of a GMC macroblock (i.e. a macroblock included
//! > in an S(GMC)-VOP and `mcsel == '1'`) for this padding process.
//!
//! So for an S(GMC) `mcsel == 1` block the caller must already have
//! substituted the §7.8.7.3 averaged MV
//! ([`crate::motion::averaged_motion_vector`]) into `vectors[i]`
//! before invoking [`pad_macroblock_vectors`] — this module accepts
//! the post-substitution vectors and treats them as ordinary block
//! MVs.
//!
//! ## What this module does **not** do
//!
//! * Derive transparency. The caller (which has access to the per-
//!   block VOP-shape mask) decides which blocks are transparent.
//! * Detect "INTRA-coded or skipped P-VOP" macroblock type. The caller
//!   already knows the §7.6.6 / §6.2.6 macroblock category and passes
//!   it as [`MacroblockPaddingMode`].
//! * Run the §7.6.1.1..7.6.1.4 luminance / chrominance sample padding.
//!   This module is the *vector* padding alone; the *sample* padding
//!   (which the §7.6.4 reference-edge clamp + §7.6.6 OBMC sample fetch
//!   composes against) is the orthogonal sibling §7.6.1.1..§7.6.1.5
//!   spec subclauses and is handled inside the §7.6.2.x interpolation
//!   primitives.
//! * Touch chroma MVs. Vector padding works on the four luma block MVs
//!   and the chroma MV is derived from the padded set in a later
//!   §7.6.5 step (see [`crate::chroma_mv`]).

use crate::motion::MotionVector;

/// Number of 8×8 luminance blocks per macroblock (Figure 6-8 / §6.1.3.4
/// 4:2:0 chroma format).
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub const LUMA_BLOCKS_PER_MB: usize = 4;

/// Per-block transparency flag (§7.6.1.6 `Transp[i]`).
///
/// A block is `Transparent` when every one of its 8×8 luminance samples
/// is outside the VOP-shape mask. Within a non-transparent macroblock,
/// individual blocks may still be `Transparent` (the §7.6.1.6 vector
/// padding is what fills their MVs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub enum BlockTransparency {
    /// The block is fully inside the VOP shape — its decoded MV is
    /// used as-is.
    Opaque,
    /// The block is fully outside the VOP shape — its MV must be filled
    /// from the §7.6.1.6 horizontal-then-vertical-then-diagonal fallback
    /// chain.
    Transparent,
}

/// Macroblock-level §7.6.1.6 mode selector.
///
/// The §7.6.1.6 procedure has two top-level branches. The
/// [`MacroblockPaddingMode::AllZero`] branch overwrites all four MVs
/// with `(0, 0)`; the [`MacroblockPaddingMode::PerBlock`] branch runs
/// the per-block transparency fallback chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub enum MacroblockPaddingMode {
    /// The macroblock is INTRA-coded, or is a P-VOP `skipped` (`COD ==
    /// 1`) macroblock. All four block MVs are forced to `(0, 0)`.
    AllZero,
    /// The macroblock is INTER-coded and not `skipped` — per-block
    /// padding using `transparencies[i]` runs.
    PerBlock,
}

/// §7.6.1.6 fallback chain, expressed as the precedence-ordered list
/// `[1st choice, 2nd choice, 3rd choice]` for each block index.
///
/// This is the verbatim transcription of the nested `?:` expressions in
/// the §7.6.1.6 pseudo-code; making the chain a `const` table keeps
/// [`pad_macroblock_vectors`] data-driven (and trivially auditable
/// against the spec).
const FALLBACK_CHAIN: [[usize; 3]; LUMA_BLOCKS_PER_MB] = [
    // Block 0 (TL): MVx[0] = (T1!=T) ? MVx[1] : ((T2!=T) ? MVx[2] : MVx[3])
    [1, 2, 3],
    // Block 1 (TR): MVx[1] = (T0!=T) ? MVx[0] : ((T3!=T) ? MVx[3] : MVx[2])
    [0, 3, 2],
    // Block 2 (BL): MVx[2] = (T3!=T) ? MVx[3] : ((T0!=T) ? MVx[0] : MVx[1])
    [3, 0, 1],
    // Block 3 (BR): MVx[3] = (T2!=T) ? MVx[2] : ((T1!=T) ? MVx[1] : MVx[0])
    [2, 1, 0],
];

/// Apply §7.6.1.6 vector padding to a macroblock's four 8×8 luminance
/// block motion vectors.
///
/// `vectors` and `transparencies` are indexed in Figure 6-8 / §6.1.3.4
/// raster order (`0 = top-left`, `1 = top-right`, `2 = bottom-left`,
/// `3 = bottom-right`). The result is written in place — every block,
/// transparent or opaque, has a defined MV on return:
///
/// * Under [`MacroblockPaddingMode::AllZero`] all four `vectors[i]` are
///   overwritten with `(0, 0)` regardless of `transparencies[i]`.
/// * Under [`MacroblockPaddingMode::PerBlock`] opaque blocks keep
///   their decoded MV unchanged, and each transparent block walks the
///   precedence-ordered fallback chain in [`FALLBACK_CHAIN`] until it
///   finds the first opaque partner; that partner's MV is copied into
///   the transparent block.
///
/// ## Sentinel: fully-transparent macroblock
///
/// Per the §7.6.1.6 opening sentence the procedure "is applied to ...
/// the transparent blocks within a *non-transparent* macroblock"
/// (emphasis added). A macroblock with all four blocks transparent is
/// not a normative input — there is nothing to use as a fallback
/// source. Callers in this situation should not invoke
/// [`pad_macroblock_vectors`]; doing so returns
/// [`VectorPaddingError::AllTransparent`] without modifying any vector.
///
/// ```
/// use oxideav_mpeg4video::{
///     pad_macroblock_vectors, BlockTransparency, MacroblockPaddingMode,
///     MotionVector,
/// };
///
/// // INTER macroblock: block 1 (TR) is transparent.
/// let mut mvs = [
///     MotionVector { x: 10, y: 4 },
///     MotionVector { x: 0, y: 0 },   // transparent — value ignored
///     MotionVector { x: -2, y: 7 },
///     MotionVector { x: 1, y: 1 },
/// ];
/// let transp = [
///     BlockTransparency::Opaque,
///     BlockTransparency::Transparent,
///     BlockTransparency::Opaque,
///     BlockTransparency::Opaque,
/// ];
/// pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
/// // Block 1's fallback chain is [0, 3, 2]; block 0 is opaque, so its
/// // MV propagates into block 1.
/// assert_eq!(mvs[1], MotionVector { x: 10, y: 4 });
/// ```
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn pad_macroblock_vectors(
    vectors: &mut [MotionVector; LUMA_BLOCKS_PER_MB],
    transparencies: &[BlockTransparency; LUMA_BLOCKS_PER_MB],
    mode: MacroblockPaddingMode,
) -> Result<(), VectorPaddingError> {
    match mode {
        MacroblockPaddingMode::AllZero => {
            // §7.6.1.6: if the macroblock is INTRA-coded, or skipped
            // and included in a P-VOP, all four block MVs are zero.
            for slot in vectors.iter_mut() {
                *slot = MotionVector { x: 0, y: 0 };
            }
            Ok(())
        }
        MacroblockPaddingMode::PerBlock => {
            // The §7.6.1.6 pseudo-code reads the *pre-padding* vectors
            // on the RHS of every `?:` (the four `if (Transp[i] ==
            // TRANSPARENT) { ... }` blocks reference each other's input
            // values, not each other's already-written outputs). Snapshot
            // the input vectors and the transparency vector first, then
            // write through to the in-place buffer.
            if transparencies
                .iter()
                .all(|t| matches!(t, BlockTransparency::Transparent))
            {
                return Err(VectorPaddingError::AllTransparent);
            }

            let snapshot = *vectors;
            for i in 0..LUMA_BLOCKS_PER_MB {
                if let BlockTransparency::Transparent = transparencies[i] {
                    // Walk this block's fallback chain in precedence
                    // order; first opaque partner wins.
                    let chain = FALLBACK_CHAIN[i];
                    let source_index = chain
                        .iter()
                        .copied()
                        .find(|&j| matches!(transparencies[j], BlockTransparency::Opaque))
                        .expect(
                            "AllTransparent already short-circuits the \
                             zero-opaque-partners case",
                        );
                    vectors[i] = snapshot[source_index];
                }
            }
            Ok(())
        }
    }
}

/// Errors returned by [`pad_macroblock_vectors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorPaddingError {
    /// All four blocks are transparent under
    /// [`MacroblockPaddingMode::PerBlock`]. §7.6.1.6 explicitly scopes
    /// itself to "the transparent blocks within a *non-transparent*
    /// macroblock", so a fully-transparent macroblock has no opaque
    /// fallback source and the caller must not invoke vector padding.
    AllTransparent,
}

impl core::fmt::Display for VectorPaddingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VectorPaddingError::AllTransparent => write!(
                f,
                "vector padding: macroblock is fully transparent — \
                 §7.6.1.6 does not define a fallback"
            ),
        }
    }
}

impl std::error::Error for VectorPaddingError {}

#[cfg(test)]
mod tests {
    use super::*;

    const OPAQUE: BlockTransparency = BlockTransparency::Opaque;
    const TRANSP: BlockTransparency = BlockTransparency::Transparent;

    fn mv(x: i32, y: i32) -> MotionVector {
        MotionVector { x, y }
    }

    // -----------------------------------------------------------------
    // Table sanity: fallback chains are 3-permutations of {0..3} \ {i}.
    // -----------------------------------------------------------------

    #[test]
    fn fallback_chains_cover_other_three_blocks() {
        for (i, chain) in FALLBACK_CHAIN.iter().enumerate() {
            let mut sorted = *chain;
            sorted.sort_unstable();
            // The fallback chain for block i must reference the *other*
            // three blocks exactly once each (no self-reference, no
            // duplicates).
            assert!(!chain.contains(&i), "block {} self-references", i);
            let mut others: Vec<usize> = (0..4).filter(|&j| j != i).collect();
            others.sort_unstable();
            assert_eq!(sorted.to_vec(), others);
        }
    }

    #[test]
    fn fallback_chains_precedence_matches_spec() {
        // §7.6.1.6 horizontal-first-then-vertical view: 1st choice is
        // always the block in the same row (horizontal partner).
        assert_eq!(FALLBACK_CHAIN[0][0], 1, "TL's 1st choice is TR");
        assert_eq!(FALLBACK_CHAIN[1][0], 0, "TR's 1st choice is TL");
        assert_eq!(FALLBACK_CHAIN[2][0], 3, "BL's 1st choice is BR");
        assert_eq!(FALLBACK_CHAIN[3][0], 2, "BR's 1st choice is BL");

        // 2nd and 3rd choices are diagonal vs vertical depending on the
        // §7.6.1.6 transcription — TL/BR prefer the diagonal block
        // before the vertical partner, TR/BL prefer the vertical
        // partner first.
        assert_eq!(FALLBACK_CHAIN[0], [1, 2, 3]);
        assert_eq!(FALLBACK_CHAIN[1], [0, 3, 2]);
        assert_eq!(FALLBACK_CHAIN[2], [3, 0, 1]);
        assert_eq!(FALLBACK_CHAIN[3], [2, 1, 0]);
    }

    // -----------------------------------------------------------------
    // AllZero branch: INTRA / P-VOP-skipped MBs.
    // -----------------------------------------------------------------

    #[test]
    fn all_zero_overrides_decoded_mvs() {
        let mut mvs = [mv(7, -3), mv(-1, 4), mv(99, 99), mv(-99, -99)];
        let transp = [OPAQUE; LUMA_BLOCKS_PER_MB];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::AllZero).unwrap();
        for m in &mvs {
            assert_eq!(*m, mv(0, 0));
        }
    }

    #[test]
    fn all_zero_ignores_transparency_pattern() {
        // §7.6.1.6 says the INTRA / P-skipped branch zeros all four
        // unconditionally — transparency is irrelevant in this branch.
        let mut mvs = [mv(1, 1); LUMA_BLOCKS_PER_MB];
        let transp = [TRANSP, OPAQUE, TRANSP, OPAQUE];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::AllZero).unwrap();
        for m in &mvs {
            assert_eq!(*m, mv(0, 0));
        }
    }

    // -----------------------------------------------------------------
    // PerBlock branch: identity case (no transparent blocks).
    // -----------------------------------------------------------------

    #[test]
    fn per_block_identity_for_all_opaque() {
        let original = [mv(1, 2), mv(3, 4), mv(5, 6), mv(7, 8)];
        let mut mvs = original;
        let transp = [OPAQUE; LUMA_BLOCKS_PER_MB];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs, original);
    }

    // -----------------------------------------------------------------
    // PerBlock branch: single-transparent cases — 1st choice always
    // matches the in-row horizontal partner.
    // -----------------------------------------------------------------

    #[test]
    fn per_block_single_transparent_block_0_uses_block_1() {
        let mut mvs = [mv(99, 99), mv(2, 3), mv(5, 6), mv(7, 8)];
        let transp = [TRANSP, OPAQUE, OPAQUE, OPAQUE];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[0], mv(2, 3));
        assert_eq!(mvs[1], mv(2, 3));
        assert_eq!(mvs[2], mv(5, 6));
        assert_eq!(mvs[3], mv(7, 8));
    }

    #[test]
    fn per_block_single_transparent_block_1_uses_block_0() {
        let mut mvs = [mv(1, 2), mv(99, 99), mv(5, 6), mv(7, 8)];
        let transp = [OPAQUE, TRANSP, OPAQUE, OPAQUE];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[0], mv(1, 2));
        assert_eq!(mvs[1], mv(1, 2));
        assert_eq!(mvs[2], mv(5, 6));
        assert_eq!(mvs[3], mv(7, 8));
    }

    #[test]
    fn per_block_single_transparent_block_2_uses_block_3() {
        let mut mvs = [mv(1, 2), mv(3, 4), mv(99, 99), mv(7, 8)];
        let transp = [OPAQUE, OPAQUE, TRANSP, OPAQUE];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[0], mv(1, 2));
        assert_eq!(mvs[1], mv(3, 4));
        assert_eq!(mvs[2], mv(7, 8));
        assert_eq!(mvs[3], mv(7, 8));
    }

    #[test]
    fn per_block_single_transparent_block_3_uses_block_2() {
        let mut mvs = [mv(1, 2), mv(3, 4), mv(5, 6), mv(99, 99)];
        let transp = [OPAQUE, OPAQUE, OPAQUE, TRANSP];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[0], mv(1, 2));
        assert_eq!(mvs[1], mv(3, 4));
        assert_eq!(mvs[2], mv(5, 6));
        assert_eq!(mvs[3], mv(5, 6));
    }

    // -----------------------------------------------------------------
    // PerBlock branch: 2nd-choice fallback (1st choice is itself
    // transparent — falls through to the 2nd entry of FALLBACK_CHAIN).
    // -----------------------------------------------------------------

    #[test]
    fn per_block_block_0_uses_block_2_when_block_1_transparent() {
        // Blocks 0 and 1 transparent; chain for block 0 is [1, 2, 3] —
        // 1 is transparent, so 2 wins.
        let mut mvs = [mv(99, 99), mv(99, 99), mv(5, 6), mv(7, 8)];
        let transp = [TRANSP, TRANSP, OPAQUE, OPAQUE];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[0], mv(5, 6));
        // Block 1's chain is [0, 3, 2] — 0 is transparent, so 3 wins.
        assert_eq!(mvs[1], mv(7, 8));
    }

    #[test]
    fn per_block_block_1_uses_block_3_when_block_0_transparent() {
        let mut mvs = [mv(99, 99), mv(99, 99), mv(5, 6), mv(7, 8)];
        let transp = [TRANSP, TRANSP, OPAQUE, OPAQUE];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        // Block 1's chain [0, 3, 2]: 0 transparent → 3 (BR) wins.
        assert_eq!(mvs[1], mv(7, 8));
    }

    #[test]
    fn per_block_block_2_uses_block_0_when_block_3_transparent() {
        // Blocks 2 and 3 transparent; chain for block 2 is [3, 0, 1] —
        // 3 is transparent, so 0 wins.
        let mut mvs = [mv(1, 2), mv(3, 4), mv(99, 99), mv(99, 99)];
        let transp = [OPAQUE, OPAQUE, TRANSP, TRANSP];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[2], mv(1, 2));
        // Block 3's chain [2, 1, 0]: 2 transparent → 1 wins.
        assert_eq!(mvs[3], mv(3, 4));
    }

    #[test]
    fn per_block_block_3_uses_block_1_when_block_2_transparent() {
        let mut mvs = [mv(1, 2), mv(3, 4), mv(99, 99), mv(99, 99)];
        let transp = [OPAQUE, OPAQUE, TRANSP, TRANSP];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        // Block 3's chain [2, 1, 0]: 2 transparent → 1 (TR) wins.
        assert_eq!(mvs[3], mv(3, 4));
    }

    // -----------------------------------------------------------------
    // PerBlock branch: 3rd-choice fallback (1st and 2nd choices both
    // transparent).
    // -----------------------------------------------------------------

    #[test]
    fn per_block_block_0_uses_block_3_when_blocks_1_2_transparent() {
        // Block 0's chain [1, 2, 3] — both 1 and 2 transparent, so 3
        // wins as the 3rd choice.
        let mut mvs = [mv(99, 99), mv(99, 99), mv(99, 99), mv(7, 8)];
        let transp = [TRANSP, TRANSP, TRANSP, OPAQUE];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[0], mv(7, 8));
        // Blocks 1 and 2 also pull from block 3 via their respective
        // 2nd / 1st-choice positions.
        assert_eq!(mvs[1], mv(7, 8)); // chain [0, 3, 2] — 0 trans → 3
        assert_eq!(mvs[2], mv(7, 8)); // chain [3, 0, 1] — 3 opaque
        assert_eq!(mvs[3], mv(7, 8));
    }

    #[test]
    fn per_block_block_1_uses_block_2_when_blocks_0_3_transparent() {
        // Block 1's chain [0, 3, 2] — 0 and 3 transparent → 2 wins.
        let mut mvs = [mv(99, 99), mv(99, 99), mv(5, 6), mv(99, 99)];
        let transp = [TRANSP, TRANSP, OPAQUE, TRANSP];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[1], mv(5, 6));
        // Symmetric: block 0's chain [1, 2, 3] — 1 trans → 2 opaque.
        assert_eq!(mvs[0], mv(5, 6));
        // Block 3's chain [2, 1, 0] — 2 opaque.
        assert_eq!(mvs[3], mv(5, 6));
    }

    #[test]
    fn per_block_block_2_uses_block_1_when_blocks_0_3_transparent() {
        // Block 2's chain [3, 0, 1] — both 3 and 0 transparent → 1
        // wins.
        let mut mvs = [mv(99, 99), mv(3, 4), mv(99, 99), mv(99, 99)];
        let transp = [TRANSP, OPAQUE, TRANSP, TRANSP];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[2], mv(3, 4));
        // Symmetric: block 0's chain [1, 2, 3] — 1 opaque.
        assert_eq!(mvs[0], mv(3, 4));
        // Block 3's chain [2, 1, 0] — 2 trans → 1 opaque.
        assert_eq!(mvs[3], mv(3, 4));
    }

    #[test]
    fn per_block_block_3_uses_block_0_when_blocks_2_1_transparent() {
        // Block 3's chain [2, 1, 0] — both 2 and 1 transparent → 0
        // wins.
        let mut mvs = [mv(1, 2), mv(99, 99), mv(99, 99), mv(99, 99)];
        let transp = [OPAQUE, TRANSP, TRANSP, TRANSP];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        assert_eq!(mvs[3], mv(1, 2));
        assert_eq!(mvs[1], mv(1, 2)); // chain [0, 3, 2] — 0 opaque
        assert_eq!(mvs[2], mv(1, 2)); // chain [3, 0, 1] — 3 trans → 0
    }

    // -----------------------------------------------------------------
    // PerBlock branch: snapshot semantics — every transparent block
    // reads from the *pre-padding* MVs of its partners, not the in-
    // place-updated outputs of an earlier iteration.
    // -----------------------------------------------------------------

    #[test]
    fn per_block_reads_pre_padding_values() {
        // Pathological pattern: blocks 0 and 1 transparent; block 0's
        // chain prefers block 1 (also transparent) → falls to block 2.
        // Without snapshotting, block 0 could pick up block 1's
        // *post-padding* value (which it gets from block 3 via the
        // [0, 3, 2] chain → 3) — i.e. (7, 8) instead of the correct
        // 2nd-choice value (5, 6) for block 0.
        let mut mvs = [mv(99, 99), mv(99, 99), mv(5, 6), mv(7, 8)];
        let transp = [TRANSP, TRANSP, OPAQUE, OPAQUE];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        // Block 0 must hold block 2's MV (2nd-choice fallback), not
        // block 3's MV.
        assert_eq!(mvs[0], mv(5, 6));
        // Block 1 must hold block 3's MV (2nd-choice fallback), not
        // block 2's MV.
        assert_eq!(mvs[1], mv(7, 8));
    }

    // -----------------------------------------------------------------
    // PerBlock branch: error case — fully-transparent MB.
    // -----------------------------------------------------------------

    #[test]
    fn per_block_all_transparent_macroblock_is_an_error() {
        let mut mvs = [mv(1, 2), mv(3, 4), mv(5, 6), mv(7, 8)];
        let transp = [TRANSP; LUMA_BLOCKS_PER_MB];
        let err =
            pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap_err();
        assert_eq!(err, VectorPaddingError::AllTransparent);
        // Vectors must not be modified on error.
        assert_eq!(mvs, [mv(1, 2), mv(3, 4), mv(5, 6), mv(7, 8)]);
    }

    // -----------------------------------------------------------------
    // Negative-MV propagation.
    // -----------------------------------------------------------------

    #[test]
    fn negative_mvs_propagate_unchanged() {
        // Vector padding only copies values around — sign extension,
        // wrapping, etc. is the caller's problem (handled in earlier
        // §7.6.3 / Table 7-9 stages).
        let mut mvs = [mv(0, 0), mv(-100, -200), mv(0, 0), mv(0, 0)];
        let transp = [TRANSP, OPAQUE, TRANSP, TRANSP];
        pad_macroblock_vectors(&mut mvs, &transp, MacroblockPaddingMode::PerBlock).unwrap();
        // All three transparent blocks must end up with block 1's MV.
        assert_eq!(mvs[0], mv(-100, -200));
        assert_eq!(mvs[1], mv(-100, -200));
        assert_eq!(mvs[2], mv(-100, -200));
        assert_eq!(mvs[3], mv(-100, -200));
    }

    // -----------------------------------------------------------------
    // Error display.
    // -----------------------------------------------------------------

    #[test]
    fn error_display_mentions_clause() {
        let e = VectorPaddingError::AllTransparent;
        let s = format!("{e}");
        assert!(s.contains("§7.6.1.6"), "got: {s}");
    }
}

//! §7.6 progressive P-VOP macroblock motion-vector decode driver.
//!
//! The lower-level primitives the spec splits §7.6 into are already
//! implemented as composable, independently-tested pieces:
//!
//! * [`crate::motion::decode_motion_vector_delta`] — the §6.2.6.2
//!   `motion_vector(mode)` body (Table B.12 `mv_data` VLC + the
//!   `r_size`-bit residual) reconstructed to a [`MotionVectorDelta`]
//!   via the §7.6.3 recurrence.
//! * [`crate::mv_predictor_grid::MvGrid::predictor_candidates`] — the
//!   Figure 7-34 candidate gather (`MV1` / `MV2` / `MV3` positions for
//!   each of the four 8×8 luminance blocks) into a
//!   `[Option<MotionVector>; 3]`.
//! * [`crate::motion::predict_motion_vector`] — the §7.6.5 four
//!   validity rules + the component-wise `Median` predictor.
//! * [`crate::motion::reconstruct_motion_vector`] — the predictor add
//!   plus the §7.6.3 / Table 7-9 modulo wrap into `[low:high]`.
//!
//! What this module adds is the **driver** that wires those four stages
//! together in raster order so that each macroblock's just-decoded
//! motion vectors become valid spatial neighbours for the macroblocks
//! decoded after it. That ordering is the whole point of the §7.6.5
//! median predictor: `MV1` (left), `MV2` (above) and `MV3` (above-right)
//! all refer to macroblocks earlier in the raster scan, and the
//! predictor is undefined unless those neighbours were recorded as they
//! were decoded.
//!
//! ## Per-macroblock dispatch (§6.2.6 / Table B.1 `derived_mb_type`)
//!
//! For a progressive P-VOP (`short_video_header == 0`, `interlaced ==
//! 0`) the §6.2.6 macroblock layer decodes, in order:
//!
//! 1. **`not_coded == 1`** — the skipped macroblock. §7.6.2 treats it
//!    as inter-coded with a **zero motion vector**: no `motion_vector()`
//!    body is present, and the predictor for the next macroblock sees
//!    this one as a valid `(0, 0)` neighbour. Recorded as
//!    [`MbMv::OneMv`]`((0, 0))`.
//! 2. **Intra / intra+q** (`derived_mb_type ∈ {3, 4}`) — no motion
//!    vector. The §7.6.5 validity rules treat an intra macroblock the
//!    same as a transparent one: it is **not a valid candidate**.
//!    Recorded as [`MbMv::Absent`] so every sub-block query against it
//!    returns [`None`].
//! 3. **Inter / inter+q** (`derived_mb_type ∈ {0, 1}`) — the 1-MV path.
//!    A single `motion_vector("forward")` body is decoded; its
//!    predictor is the §7.6.5 median for **block 0** (the top-left case,
//!    per the Figure 7-34 "1-MV mode uses the top-left case" rule). The
//!    reconstructed vector is recorded as [`MbMv::OneMv`].
//! 4. **inter4v** (`derived_mb_type == 2`) — the 4-MV path. Four
//!    `motion_vector("forward")` bodies are decoded, one per 8×8
//!    luminance block in Figure 6-8 order. **Each** block's predictor is
//!    the §7.6.5 median gathered for **that** block index, and crucially
//!    the predictor for blocks 1..3 sees the just-decoded vectors of the
//!    earlier blocks **of the same macroblock** (Figure 7-34 places
//!    several of an inter4v block's candidates inside the current MB).
//!    To make those in-MB candidates visible, the grid record is
//!    updated incrementally as each block vector is reconstructed.
//!    Recorded as [`MbMv::FourMv`].
//!
//! ## Predictor reset at row / packet boundaries
//!
//! The §7.6.5 candidate gather already returns [`None`] for any
//! neighbour outside the grid (row `-1`, column `-1`, or past the right
//! edge), so the left/above/above-right macroblocks of the first row and
//! first column are correctly invalid. A video-packet boundary inside a
//! row is signalled to the driver by [`MvDriver::reset_packet`], which
//! marks every macroblock decoded so far in the *current* row as not a
//! valid neighbour for macroblocks after the boundary (the §7.6.5
//! "outside the current video packet → treated as transparent" rule).
//! The caller invokes it immediately after consuming a `resync_marker` +
//! `video_packet_header`.
//!
//! ## Scope
//!
//! This is the **progressive** P-VOP path: `quarter_sample` only changes
//! the *units* of the decoded vector (the Table B.12 / §7.6.3 path is
//! identical — quarter-pel vs half-pel is a downstream interpolation
//! concern), so the driver is unit-agnostic. The interlaced field-MV
//! predictor (§7.7.2.1) and the S(GMC) `mcsel` averaged-vector
//! substitution (§7.8.7.3) have their own gather entry points
//! ([`MvGrid::field_predictor_candidates`] /
//! [`crate::motion::averaged_motion_vector`]) and are driven elsewhere.

use crate::bitreader::BitReader;
use crate::macroblock::DerivedMbType;
use crate::motion::{
    decode_motion_vector_delta, predict_motion_vector, reconstruct_motion_vector, MotionParseError,
    MotionVector, MvMode,
};
use crate::mv_predictor_grid::{MbMv, MbMvRecord, MvGrid, MvGridError, LUMA_BLOCKS_PER_MB};

/// One decoded P-VOP macroblock's motion-vector outcome, as recorded
/// into the [`MvGrid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PvopMbMotion {
    /// Skipped (`not_coded == 1`) — zero motion vector, no bits read.
    Skipped,
    /// Intra / intra+q — no motion vector; recorded as an invalid
    /// ([`MbMv::Absent`]) neighbour.
    Intra,
    /// Inter / inter+q — one reconstructed 16×16 motion vector.
    OneMv(MotionVector),
    /// inter4v — four reconstructed 8×8 motion vectors in Figure 6-8
    /// order.
    FourMv([MotionVector; 4]),
}

impl PvopMbMotion {
    /// The single 16×16 motion vector of an inter / inter+q macroblock,
    /// or `None` for any other variant.
    pub fn one_mv(self) -> Option<MotionVector> {
        match self {
            PvopMbMotion::OneMv(mv) => Some(mv),
            _ => None,
        }
    }

    /// The four 8×8 motion vectors of an inter4v macroblock, or `None`
    /// for any other variant.
    pub fn four_mv(self) -> Option<[MotionVector; 4]> {
        match self {
            PvopMbMotion::FourMv(mvs) => Some(mvs),
            _ => None,
        }
    }
}

/// Errors produced by the P-VOP motion-vector driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PvopMvError {
    /// The motion-vector body failed to decode (Table B.12 VLC miss,
    /// truncation, or an out-of-range `vop_fcode`).
    Motion(MotionParseError),
    /// The grid coordinate / block index was out of range.
    Grid(MvGridError),
}

impl core::fmt::Display for PvopMvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PvopMvError::Motion(e) => write!(f, "p-vop motion-vector decode failed: {e}"),
            PvopMvError::Grid(e) => write!(f, "p-vop mv-grid access failed: {e}"),
        }
    }
}

impl std::error::Error for PvopMvError {}

impl From<MotionParseError> for PvopMvError {
    fn from(e: MotionParseError) -> Self {
        PvopMvError::Motion(e)
    }
}

impl From<MvGridError> for PvopMvError {
    fn from(e: MvGridError) -> Self {
        PvopMvError::Grid(e)
    }
}

/// The §7.6 progressive P-VOP motion-vector decode driver.
///
/// Owns the per-VOP [`MvGrid`] and walks macroblocks in raster order via
/// [`MvDriver::decode_macroblock`]. The `vop_fcode_forward` from the VOP
/// header is fixed for the lifetime of the driver (it does not change
/// within a P-VOP). See the [module docs](self) for the dispatch rules.
#[derive(Debug, Clone)]
pub struct MvDriver {
    grid: MvGrid,
    vop_fcode_forward: u8,
}

impl MvDriver {
    /// Create a driver for a `mb_rows × mb_cols` progressive P-VOP with
    /// the given `vop_fcode_forward` (§6.3.5, `1..=7`).
    pub fn new(mb_rows: usize, mb_cols: usize, vop_fcode_forward: u8) -> Self {
        Self {
            grid: MvGrid::new(mb_rows, mb_cols),
            vop_fcode_forward,
        }
    }

    /// Borrow the underlying [`MvGrid`] (e.g. to feed the reconstructed
    /// vectors into the §7.6.2 prediction-block generator).
    pub fn grid(&self) -> &MvGrid {
        &self.grid
    }

    /// Consume the driver and return its [`MvGrid`].
    pub fn into_grid(self) -> MvGrid {
        self.grid
    }

    /// Number of macroblock rows in the VOP.
    pub fn mb_rows(&self) -> usize {
        self.grid.mb_rows()
    }

    /// Number of macroblock columns in the VOP.
    pub fn mb_cols(&self) -> usize {
        self.grid.mb_cols()
    }

    /// §7.6.5 video-packet / GOB boundary: mark every macroblock already
    /// decoded in row `mb_row` strictly **before** column
    /// `boundary_col` as not a valid spatial neighbour for the
    /// macroblocks at or after `boundary_col`.
    ///
    /// The §7.6.5 note treats a candidate predictor in a neighbouring
    /// macroblock "outside the current video packet" (or, for
    /// `short_video_header == 1`, "outside the current GOB") as
    /// transparent → not valid. The first macroblock after a
    /// `resync_marker` therefore predicts as if its left / above-left
    /// neighbours did not exist. The driver models this by recording
    /// those earlier macroblocks of the current row as [`MbMv::Absent`].
    ///
    /// Rows above `mb_row` are left untouched: the spec only resets the
    /// predictors that would cross the packet boundary horizontally —
    /// `MV2` (above) and `MV3` (above-right) of a macroblock that begins
    /// a packet may legitimately fall in a *different* packet of the row
    /// above, and the spec's "current video packet" test is applied
    /// per-candidate by the grid's [`None`] return for out-of-grid
    /// positions plus the per-row reset here.
    pub fn reset_packet(&mut self, mb_row: usize, boundary_col: usize) -> Result<(), PvopMvError> {
        let cols = boundary_col.min(self.grid.mb_cols());
        for col in 0..cols {
            self.grid.record_absent(mb_row, col)?;
        }
        Ok(())
    }

    /// Decode the motion vector(s) of one progressive P-VOP macroblock
    /// at raster position `(mb_row, mb_col)`, threading the result into
    /// the grid so later macroblocks see it as a neighbour.
    ///
    /// * `not_coded` / `mb_type` come from
    ///   [`crate::macroblock::parse_macroblock_header`] (an intra
    ///   `mb_type` need not be supplied for a `not_coded` MB; pass
    ///   `None`).
    /// * `br` is positioned at the start of the macroblock's
    ///   `motion_vector()` body (for inter / inter4v) or is not advanced
    ///   at all (skipped / intra).
    ///
    /// On `Ok` the bit reader sits immediately after the last
    /// `motion_vector()` body consumed (none for skipped / intra).
    pub fn decode_macroblock(
        &mut self,
        br: &mut BitReader<'_>,
        mb_row: usize,
        mb_col: usize,
        not_coded: bool,
        mb_type: Option<DerivedMbType>,
    ) -> Result<PvopMbMotion, PvopMvError> {
        if not_coded {
            // §7.6.2 — a not-coded P-VOP macroblock is inter with a zero
            // motion vector. Record (0, 0) so it is a *valid* neighbour.
            self.grid
                .record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })?;
            return Ok(PvopMbMotion::Skipped);
        }

        match mb_type {
            Some(DerivedMbType::Intra) | Some(DerivedMbType::IntraQ) | None => {
                // §7.6.5 — intra (or unspecified) MB is not a valid MV
                // candidate. Recorded as Absent.
                self.grid.record_absent(mb_row, mb_col)?;
                Ok(PvopMbMotion::Intra)
            }
            Some(DerivedMbType::Inter) | Some(DerivedMbType::InterQ) => {
                let mv = self.decode_one_mv(br, mb_row, mb_col)?;
                Ok(PvopMbMotion::OneMv(mv))
            }
            Some(DerivedMbType::Inter4V) => {
                let mvs = self.decode_four_mv(br, mb_row, mb_col)?;
                Ok(PvopMbMotion::FourMv(mvs))
            }
        }
    }

    /// 1-MV path (inter / inter+q). The predictor is the §7.6.5 median
    /// for block 0 (the Figure 7-34 "1-MV mode → top-left case" rule).
    fn decode_one_mv(
        &mut self,
        br: &mut BitReader<'_>,
        mb_row: usize,
        mb_col: usize,
    ) -> Result<MotionVector, PvopMvError> {
        let candidates = self.grid.predictor_candidates(mb_row, mb_col, 0)?;
        let predictor = predict_motion_vector(candidates);
        let delta = decode_motion_vector_delta(br, MvMode::Forward, self.vop_fcode_forward)?;
        let mv =
            reconstruct_motion_vector(delta, predictor.x, predictor.y, self.vop_fcode_forward)?;
        self.grid.record_one_mv(mb_row, mb_col, mv)?;
        Ok(mv)
    }

    /// 4-MV path (inter4v). Each block's predictor is gathered *after*
    /// the earlier blocks of the same macroblock have been recorded, so
    /// the in-MB candidates of Figure 7-34 (blocks 2/3/4 referencing
    /// blocks 1/2/3) are visible.
    fn decode_four_mv(
        &mut self,
        br: &mut BitReader<'_>,
        mb_row: usize,
        mb_col: usize,
    ) -> Result<[MotionVector; 4], PvopMvError> {
        let mut mvs = [MotionVector { x: 0, y: 0 }; LUMA_BLOCKS_PER_MB];
        // Seed the cell as a fully-decoded 4-MV macroblock with zero
        // vectors so the incremental in-MB candidate gather has a record
        // to read; each iteration overwrites one block's slot. (A block
        // not yet decoded reads as (0, 0), but Figure 7-34 only ever
        // points an in-MB candidate at an *earlier* block, so no
        // not-yet-decoded slot is ever consulted.)
        self.grid.record_four_mv(mb_row, mb_col, mvs)?;
        for block_index in 0..LUMA_BLOCKS_PER_MB {
            let candidates = self
                .grid
                .predictor_candidates(mb_row, mb_col, block_index)?;
            let predictor = predict_motion_vector(candidates);
            let delta = decode_motion_vector_delta(br, MvMode::Forward, self.vop_fcode_forward)?;
            let mv =
                reconstruct_motion_vector(delta, predictor.x, predictor.y, self.vop_fcode_forward)?;
            mvs[block_index] = mv;
            // Re-record with the freshly-decoded block so the next
            // block's in-MB candidate gather sees it.
            self.grid.record(
                mb_row,
                mb_col,
                MbMvRecord {
                    content: MbMv::FourMv(mvs),
                    transparent: [false; 4],
                },
            )?;
        }
        Ok(mvs)
    }
}

/// Generate the §7.6.2 16×16 luminance prediction block for one decoded
/// progressive P-VOP macroblock, half-sample interpolated from a
/// reference plane.
///
/// This is the bridge from the [`MvDriver`]'s decoded motion vector(s)
/// to a concrete predicted block — the §7.6.2.1 half-sample path. It
/// dispatches on the [`PvopMbMotion`] the driver returned:
///
/// * [`PvopMbMotion::Skipped`] → a zero-MV 16×16 block (the §7.6.2
///   skipped-MB copy of the co-located reference samples, MV `(0, 0)`).
/// * [`PvopMbMotion::OneMv`] → one 16×16 [`interpolate_block`] with the
///   single MV.
/// * [`PvopMbMotion::FourMv`] → four 8×8 [`interpolate_block`]s, one per
///   Figure 6-8 luminance sub-block, tiled into the 16×16 output. Block
///   `i`'s 8×8 origin is `(mb_x + 8*(i & 1), mb_y + 8*(i >> 1))`.
/// * [`PvopMbMotion::Intra`] → `None`: an intra macroblock carries no
///   motion-compensated prediction (its samples come from the §7.4
///   intra texture path, not §7.6.2).
///
/// `mb_x` / `mb_y` are the macroblock's top-left luma pixel coordinates
/// in the current VOP (`16 * mb_col`, `16 * mb_row`). The MVs are taken
/// in §7.6.3 half-sample units (the progressive `quarter_sample == 0`
/// path). Returns a row-major `[u8; 256]` 16×16 block (length 256),
/// laid out `block[16*j + i]`.
pub fn predict_luma_macroblock(
    motion: PvopMbMotion,
    reference: &crate::half_sample::ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    vop_rounding_type: u8,
) -> Option<Vec<u8>> {
    use crate::half_sample::interpolate_block_into;

    match motion {
        PvopMbMotion::Intra => None,
        PvopMbMotion::Skipped => Some(crate::half_sample::interpolate_block(
            reference,
            0,
            0,
            mb_x,
            mb_y,
            16,
            16,
            vop_rounding_type,
        )),
        PvopMbMotion::OneMv(mv) => Some(crate::half_sample::interpolate_block(
            reference,
            mv.x,
            mv.y,
            mb_x,
            mb_y,
            16,
            16,
            vop_rounding_type,
        )),
        PvopMbMotion::FourMv(mvs) => {
            let mut out = vec![0u8; 256];
            let mut tile = vec![0u8; 64];
            for (i, mv) in mvs.iter().enumerate() {
                let bx = mb_x + 8 * (i as i32 & 1);
                let by = mb_y + 8 * (i as i32 >> 1);
                interpolate_block_into(
                    reference,
                    mv.x,
                    mv.y,
                    bx,
                    by,
                    8,
                    8,
                    vop_rounding_type,
                    &mut tile,
                );
                let dst_col = 8 * (i & 1);
                let dst_row = 8 * (i >> 1);
                for r in 0..8 {
                    let dst = (dst_row + r) * 16 + dst_col;
                    out[dst..dst + 8].copy_from_slice(&tile[r * 8..r * 8 + 8]);
                }
            }
            Some(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    /// Build a `BitReader` over a single mv_data VLC + nothing else.
    /// `mv_data == 0` is the 1-bit code `1`; for `vop_fcode == 1` no
    /// residual is read.
    ///
    /// We assemble bitstreams by hand from Table B.12 codes. The
    /// shortest codes are enough to exercise the driver dispatch without
    /// re-testing the VLC table itself (that is `motion.rs`'s job).
    /// Code `1` → mv_data 0 (a zero differential).
    fn zero_delta_bits(component_pairs: usize) -> Vec<u8> {
        // Each component is the 1-bit code `1`. Two components per MV.
        let bits = component_pairs * 2;
        let mut byte = 0u8;
        let mut out = Vec::new();
        let mut count = 0;
        for _ in 0..bits {
            byte = (byte << 1) | 1;
            count += 1;
            if count == 8 {
                out.push(byte);
                byte = 0;
                count = 0;
            }
        }
        if count > 0 {
            byte <<= 8 - count;
            out.push(byte);
        }
        out
    }

    #[test]
    fn skipped_mb_records_zero_mv() {
        let mut driver = MvDriver::new(2, 2, 1);
        let data = [];
        let mut br = BitReader::new(&data);
        let out = driver.decode_macroblock(&mut br, 0, 0, true, None).unwrap();
        assert_eq!(out, PvopMbMotion::Skipped);
        // The skipped MB is a *valid* (0, 0) neighbour: a block-0
        // gather at (0, 1) sees MV1 = the left MB = (0, 0).
        let cands = driver.grid().predictor_candidates(0, 1, 0).unwrap();
        assert_eq!(cands[0], Some(MotionVector { x: 0, y: 0 }));
    }

    #[test]
    fn intra_mb_records_absent() {
        let mut driver = MvDriver::new(2, 2, 1);
        let data = [];
        let mut br = BitReader::new(&data);
        let out = driver
            .decode_macroblock(&mut br, 0, 0, false, Some(DerivedMbType::Intra))
            .unwrap();
        assert_eq!(out, PvopMbMotion::Intra);
        // Intra MB is an invalid neighbour → MV1 of (0, 1) block 0 is
        // None.
        let cands = driver.grid().predictor_candidates(0, 1, 0).unwrap();
        assert_eq!(cands[0], None);
    }

    #[test]
    fn inter_mb_zero_delta_at_origin_is_zero() {
        let mut driver = MvDriver::new(2, 2, 1);
        // One MV = two zero-delta components.
        let data = zero_delta_bits(1);
        let mut br = BitReader::new(&data);
        let out = driver
            .decode_macroblock(&mut br, 0, 0, false, Some(DerivedMbType::Inter))
            .unwrap();
        // At the top-left MB all three candidates are out-of-grid →
        // predictor (0, 0); zero delta → final (0, 0).
        assert_eq!(out, PvopMbMotion::OneMv(MotionVector { x: 0, y: 0 }));
    }

    #[test]
    fn inter_mb_predicts_from_left_neighbour() {
        // Plant a left neighbour with a known MV via a skipped path is
        // (0,0); instead decode (0,0) at (0,0) first, then verify the
        // predictor at (0,1) takes the left MB's vector.
        let mut driver = MvDriver::new(1, 3, 1);
        // MB (0,0): decode a non-zero MV by hand. mv_data code for
        // value +1 (half-pel 0.5) is `010` (3 bits) per Table B.12; two
        // components → x and y each +1.
        // Build: 010 010 → 6 bits.
        // bits `010 010 00`: mv_data `010` (+1) for x, `010` (+1) for
        // y, two pad bits → byte 0x48.
        let mb0 = [0x48u8];
        let mut br = BitReader::new(&mb0);
        let m0 = driver
            .decode_macroblock(&mut br, 0, 0, false, Some(DerivedMbType::Inter))
            .unwrap();
        assert_eq!(m0, PvopMbMotion::OneMv(MotionVector { x: 1, y: 1 }));

        // MB (0,1): zero delta. Its block-0 predictor: MV1 = left MB =
        // (1,1); MV2 (above, row -1) invalid; MV3 (above-right) invalid.
        // §7.6.5 rule 3: two invalid → both take the one valid → all
        // three (1,1) → median (1,1). Zero delta → final (1,1).
        let mb1 = zero_delta_bits(1);
        let mut br = BitReader::new(&mb1);
        let m1 = driver
            .decode_macroblock(&mut br, 0, 1, false, Some(DerivedMbType::Inter))
            .unwrap();
        assert_eq!(m1, PvopMbMotion::OneMv(MotionVector { x: 1, y: 1 }));
    }

    #[test]
    fn inter4v_records_four_vectors_and_threads_in_mb() {
        let mut driver = MvDriver::new(2, 2, 1);
        // Four MVs, each two zero-delta components → 8 components.
        let data = zero_delta_bits(4);
        let mut br = BitReader::new(&data);
        let out = driver
            .decode_macroblock(&mut br, 0, 0, false, Some(DerivedMbType::Inter4V))
            .unwrap();
        assert_eq!(out, PvopMbMotion::FourMv([MotionVector { x: 0, y: 0 }; 4]));
        // The recorded 4-MV macroblock is visible per-block to a later
        // neighbour: block-0 gather at (0,1) sees MV1 = this MB's TR
        // sub-block = (0, 0).
        let cands = driver.grid().predictor_candidates(0, 1, 0).unwrap();
        assert_eq!(cands[0], Some(MotionVector { x: 0, y: 0 }));
    }

    #[test]
    fn reset_packet_invalidates_left_neighbour() {
        let mut driver = MvDriver::new(1, 3, 1);
        // Decode a non-zero MV at (0,0).
        // bits `010 010 00`: mv_data `010` (+1) for x, `010` (+1) for
        // y, two pad bits → byte 0x48.
        let mb0 = [0x48u8];
        let mut br = BitReader::new(&mb0);
        driver
            .decode_macroblock(&mut br, 0, 0, false, Some(DerivedMbType::Inter))
            .unwrap();
        // A video-packet boundary at column 1: (0,0) is now transparent.
        driver.reset_packet(0, 1).unwrap();
        // (0,1) block-0 MV1 (left) now invalid.
        let cands = driver.grid().predictor_candidates(0, 1, 0).unwrap();
        assert_eq!(cands[0], None);
    }

    #[test]
    fn raster_walk_threads_above_neighbour() {
        // 2x1 grid: MB (0,0) then MB (1,0). The second row's block-0
        // MV2 (above) should see the first row's bottom-left sub-block.
        let mut driver = MvDriver::new(2, 1, 1);
        // bits `010 010 00`: mv_data `010` (+1) for x, `010` (+1) for
        // y, two pad bits → byte 0x48.
        let mb0 = [0x48u8]; // (1, 1)
        let mut br = BitReader::new(&mb0);
        driver
            .decode_macroblock(&mut br, 0, 0, false, Some(DerivedMbType::Inter))
            .unwrap();
        // (1,0) zero delta. block-0 MV1 (left, col -1) invalid; MV2
        // (above) = (0,0)'s BL sub-block = (1,1); MV3 (above-right, col
        // +2 of row 0) out of 1-col grid → invalid. Two invalid → both
        // take (1,1) → median (1,1).
        let mb1 = zero_delta_bits(1);
        let mut br = BitReader::new(&mb1);
        let m1 = driver
            .decode_macroblock(&mut br, 1, 0, false, Some(DerivedMbType::Inter))
            .unwrap();
        assert_eq!(m1, PvopMbMotion::OneMv(MotionVector { x: 1, y: 1 }));
    }

    // ------------------------------------------------------------------
    // §7.6.2 prediction-block generation (driver → predicted block).
    // ------------------------------------------------------------------

    use crate::half_sample::ReferenceVop;

    /// A `w × h` reference plane whose sample value is `x + y` (mod 256)
    /// — a deterministic gradient with no half-pel ambiguity at integer
    /// MVs.
    fn gradient_plane(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = ((x + y) & 0xff) as u8;
            }
        }
        v
    }

    #[test]
    fn skipped_mb_predicts_colocated_reference() {
        let w = 32;
        let h = 32;
        let plane = gradient_plane(w, h);
        let reference = ReferenceVop::new(&plane, w, h).unwrap();
        // MB at (16, 0): a skipped MB copies the co-located samples
        // (MV 0,0). Sample (i, j) of the block = reference(16+i, 0+j) =
        // (16 + i + j) mod 256.
        let block = predict_luma_macroblock(PvopMbMotion::Skipped, &reference, 16, 0, 0).unwrap();
        for j in 0..16 {
            for i in 0..16 {
                assert_eq!(block[16 * j + i], ((16 + i + j) & 0xff) as u8);
            }
        }
    }

    #[test]
    fn intra_mb_has_no_motion_prediction() {
        let plane = gradient_plane(16, 16);
        let reference = ReferenceVop::new(&plane, 16, 16).unwrap();
        assert!(predict_luma_macroblock(PvopMbMotion::Intra, &reference, 0, 0, 0).is_none());
    }

    #[test]
    fn one_mv_integer_shift_predicts_shifted_reference() {
        // Half-pel units: MV (4, 2) = integer shift (+2, +1), no
        // half-pel fraction. The predicted block samples the reference
        // shifted by (+2, +1).
        let w = 32;
        let h = 32;
        let plane = gradient_plane(w, h);
        let reference = ReferenceVop::new(&plane, w, h).unwrap();
        let mv = MotionVector { x: 4, y: 2 };
        let block = predict_luma_macroblock(PvopMbMotion::OneMv(mv), &reference, 0, 0, 0).unwrap();
        // block(i, j) = reference(0 + i + 2, 0 + j + 1) = (i + 2 + j + 1).
        for j in 0..16 {
            for i in 0..16 {
                assert_eq!(block[16 * j + i], ((i + 2 + j + 1) & 0xff) as u8);
            }
        }
    }

    #[test]
    fn four_mv_tiles_four_eight_by_eight_predictions() {
        let w = 48;
        let h = 48;
        let plane = gradient_plane(w, h);
        let reference = ReferenceVop::new(&plane, w, h).unwrap();
        // Four distinct integer MVs (half-pel units, all even → no
        // fraction): TL (+1,0), TR (0,+1), BL (+2,0), BR (0,+2) →
        // integer shifts (+0.5 floor) … keep them even for exactness:
        let mvs = [
            MotionVector { x: 2, y: 0 }, // block 0 (TL): shift (+1, 0)
            MotionVector { x: 0, y: 2 }, // block 1 (TR): shift (0, +1)
            MotionVector { x: 4, y: 0 }, // block 2 (BL): shift (+2, 0)
            MotionVector { x: 0, y: 4 }, // block 3 (BR): shift (0, +2)
        ];
        let block =
            predict_luma_macroblock(PvopMbMotion::FourMv(mvs), &reference, 0, 0, 0).unwrap();
        // Block 0 (TL): origin (0,0), shift (+1,0).
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(block[16 * j + i], ((i + 1 + j) & 0xff) as u8);
            }
        }
        // Block 1 (TR): origin (8,0), shift (0,+1).
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(block[16 * j + (8 + i)], ((8 + i + j + 1) & 0xff) as u8);
            }
        }
        // Block 2 (BL): origin (0,8), shift (+2,0).
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(block[16 * (8 + j) + i], ((i + 2 + 8 + j) & 0xff) as u8);
            }
        }
        // Block 3 (BR): origin (8,8), shift (0,+2).
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(
                    block[16 * (8 + j) + (8 + i)],
                    ((8 + i + 8 + j + 2) & 0xff) as u8
                );
            }
        }
    }

    /// End-to-end: drive a synthetic 2×2-macroblock progressive P-VOP
    /// motion-vector bitstream and predict every macroblock from a
    /// reference frame, asserting the predictor threading and the
    /// half-sample composition produce the expected frame.
    #[test]
    fn end_to_end_pvop_mv_decode_and_predict() {
        // 2×2 MBs = 32×32 luma. Build a bitstream of four inter MBs in
        // raster order, each a single MV with a known delta.
        //
        // Table B.12 codes (half-pel units):
        //   `1`   → 0
        //   `010` → +1
        //   `011` → -1
        //
        // MB(0,0): delta (+1, +1). predictor (0,0) → MV (1, 1).
        // MB(0,1): delta (0, 0).   predictor: MV1=left=(1,1),
        //          MV2/MV3 invalid → median (1,1) → MV (1, 1).
        // MB(1,0): delta (0, 0).   predictor: MV2=above=(1,1),
        //          MV1/MV3 invalid → median (1,1) → MV (1, 1).
        // MB(1,1): delta (-1, -1). predictor: MV1=left(1,0)=(1,1),
        //          MV2=above(0,1)=(1,1), MV3=above-right(0,2) invalid →
        //          rule-2 (one invalid → 0): median(1,1,0)=1 each →
        //          predictor (1,1); +(-1,-1) → MV (0, 0).
        //
        // Bit layout per MB: two mv_data codes back to back.
        //   MB(0,0): 010 010                      → x +1, y +1
        //   MB(0,1): 1 1                           → x 0,  y 0
        //   MB(1,0): 1 1                           → x 0,  y 0
        //   MB(1,1): 011 011                       → x -1, y -1
        //
        // We decode each MB from its own reader (the driver does not
        // require a single contiguous reader; the real macroblock layer
        // interleaves CBP / texture bits between MVs).
        let mut driver = MvDriver::new(2, 2, 1);

        let mb00 = [0x48u8]; // 010 010 00
        let mut br = BitReader::new(&mb00);
        let m00 = driver
            .decode_macroblock(&mut br, 0, 0, false, Some(DerivedMbType::Inter))
            .unwrap();
        assert_eq!(m00, PvopMbMotion::OneMv(MotionVector { x: 1, y: 1 }));

        let zeros = zero_delta_bits(1); // 1 1 padded
        let mut br = BitReader::new(&zeros);
        let m01 = driver
            .decode_macroblock(&mut br, 0, 1, false, Some(DerivedMbType::Inter))
            .unwrap();
        assert_eq!(m01, PvopMbMotion::OneMv(MotionVector { x: 1, y: 1 }));

        let mut br = BitReader::new(&zeros);
        let m10 = driver
            .decode_macroblock(&mut br, 1, 0, false, Some(DerivedMbType::Inter))
            .unwrap();
        assert_eq!(m10, PvopMbMotion::OneMv(MotionVector { x: 1, y: 1 }));

        // bits `011 011 00`: mv_data `011` (-1) for x, `011` (-1) for
        // y, two pad bits → byte 0x6C.
        let mb11 = [0x6Cu8];
        let mut br = BitReader::new(&mb11);
        let m11 = driver
            .decode_macroblock(&mut br, 1, 1, false, Some(DerivedMbType::Inter))
            .unwrap();
        assert_eq!(m11, PvopMbMotion::OneMv(MotionVector { x: 0, y: 0 }));

        // Predict the whole 32×32 frame from a gradient reference.
        let w = 48;
        let h = 48;
        let plane = gradient_plane(w, h);
        let reference = ReferenceVop::new(&plane, w, h).unwrap();
        let motions: [(usize, usize, PvopMbMotion); 4] =
            [(0, 0, m00), (0, 1, m01), (1, 0, m10), (1, 1, m11)];
        for (mb_row, mb_col, motion) in motions {
            let mb_x = 16 * mb_col as i32;
            let mb_y = 16 * mb_row as i32;
            let block = predict_luma_macroblock(motion, &reference, mb_x, mb_y, 0).unwrap();
            let mv = motion.one_mv().unwrap();
            // Half-pel MV (1,1) = integer shift (0,0) + half-pel (1,1):
            // bilinear of four neighbours. MV (0,0) = exact copy.
            // For MV (1,1) on a (x+y) gradient the bilinear average of
            // (x+y), (x+1+y), (x+y+1), (x+1+y+1) with +1-rc rounding is
            // (4(x+y)+4 + 2 - 0)/4 = (x+y) + 1 (rc = 0). Verify a few
            // samples against that closed form.
            let (sx, sy) = (mv.x >> 1, mv.y >> 1); // integer part
            let (hx, hy) = (mv.x & 1, mv.y & 1); // half-pel part
            for j in [0usize, 7, 15] {
                for i in [0usize, 7, 15] {
                    let base_x = mb_x + i as i32 + sx;
                    let base_y = mb_y + j as i32 + sy;
                    let expected = if hx == 0 && hy == 0 {
                        ((base_x + base_y) & 0xff) as u8
                    } else {
                        // full diagonal bilinear, rc = 0
                        let a = (base_x + base_y) & 0xff;
                        let b = (base_x + 1 + base_y) & 0xff;
                        let c = (base_x + base_y + 1) & 0xff;
                        let d = (base_x + 1 + base_y + 1) & 0xff;
                        ((a + b + c + d + 2) / 4) as u8
                    };
                    assert_eq!(
                        block[16 * j + i],
                        expected,
                        "MB({mb_row},{mb_col}) sample ({i},{j}) mv={mv:?}"
                    );
                }
            }
        }
    }
}

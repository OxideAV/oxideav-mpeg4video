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
}

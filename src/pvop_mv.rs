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

    /// §7.8.7.3: record an `mcsel == 1` S(GMC)-VOP macroblock's
    /// **averaged motion vector** into the predictor grid without
    /// consuming any bits.
    ///
    /// A global-motion-compensated macroblock transmits no local motion
    /// vectors, but §7.8.7.3 substitutes the averaged pel-wise warping
    /// vector ([`crate::motion::averaged_motion_vector`]) as the
    /// candidate predictor later macroblocks' §7.6.5 median consults.
    /// The driver records it as the macroblock's single 16×16 vector so
    /// the Figure 7-34 candidate gathering sees a *valid* neighbour.
    pub fn record_gmc_macroblock(
        &mut self,
        mb_row: usize,
        mb_col: usize,
        averaged_mv: MotionVector,
    ) -> Result<(), PvopMvError> {
        self.grid.record_one_mv(mb_row, mb_col, averaged_mv)?;
        Ok(())
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

/// Interpolate one `w × h` luminance prediction block on the sub-pel
/// grid `mode` selects: §7.6.2.1 half-sample bilinear or §7.6.2.2
/// quarter-sample FIR + bilinear (with per-block boundary mirroring).
// One argument per §7.6.2 interpolation input, mirroring the two
// wrapped primitives' signatures.
#[allow(clippy::too_many_arguments)]
fn interpolate_luma_block_into(
    reference: &crate::half_sample::ReferenceVop<'_>,
    mv: MotionVector,
    origin_x: i32,
    origin_y: i32,
    w: usize,
    h: usize,
    vop_rounding_type: u8,
    mode: crate::bvop_prediction::BVopSampleMode,
    out: &mut [u8],
) {
    match mode {
        crate::bvop_prediction::BVopSampleMode::HalfPel => {
            crate::half_sample::interpolate_block_into(
                reference,
                mv.x,
                mv.y,
                origin_x,
                origin_y,
                w,
                h,
                vop_rounding_type,
                out,
            );
        }
        crate::bvop_prediction::BVopSampleMode::QuarterPel { bits_per_pixel } => {
            crate::quarter_sample::interpolate_block_qpel_into(
                reference,
                mv.x,
                mv.y,
                origin_x,
                origin_y,
                w,
                h,
                vop_rounding_type,
                bits_per_pixel,
                out,
            );
        }
    }
}

/// Generate the §7.6.2 16×16 luminance prediction block for one decoded
/// progressive P-VOP macroblock, sub-pel interpolated from a reference
/// plane.
///
/// This is the bridge from the [`MvDriver`]'s decoded motion vector(s)
/// to a concrete predicted block — the §7.6.2.1 half-sample path when
/// `mode` is [`BVopSampleMode::HalfPel`](crate::bvop_prediction::BVopSampleMode::HalfPel),
/// or the §7.6.2.2 quarter-sample path (8-tap FIR + bilinear, per-block
/// boundary mirroring) when the VOL coded `quarter_sample == 1`. It
/// dispatches on the [`PvopMbMotion`] the driver returned:
///
/// * [`PvopMbMotion::Skipped`] → a zero-MV 16×16 block (the §7.6.2
///   skipped-MB copy of the co-located reference samples, MV `(0, 0)`).
/// * [`PvopMbMotion::OneMv`] → one 16×16 interpolation with the
///   single MV.
/// * [`PvopMbMotion::FourMv`] → four 8×8 interpolations, one per
///   Figure 6-8 luminance sub-block, tiled into the 16×16 output. Block
///   `i`'s 8×8 origin is `(mb_x + 8*(i & 1), mb_y + 8*(i >> 1))`.
/// * [`PvopMbMotion::Intra`] → `None`: an intra macroblock carries no
///   motion-compensated prediction (its samples come from the §7.4
///   intra texture path, not §7.6.2).
///
/// `mb_x` / `mb_y` are the macroblock's top-left luma pixel coordinates
/// in the current VOP (`16 * mb_col`, `16 * mb_row`). The MVs are taken
/// in the §7.6.3 unitless-integer representation matching `mode`
/// (half-sample units when `quarter_sample == 0`, quarter-sample units
/// when `quarter_sample == 1`). Returns a row-major `[u8; 256]` 16×16
/// block (length 256), laid out `block[16*j + i]`.
pub fn predict_luma_macroblock(
    motion: PvopMbMotion,
    reference: &crate::half_sample::ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    vop_rounding_type: u8,
    mode: crate::bvop_prediction::BVopSampleMode,
) -> Option<Vec<u8>> {
    match motion {
        PvopMbMotion::Intra => None,
        PvopMbMotion::Skipped => {
            let mut out = vec![0u8; 256];
            interpolate_luma_block_into(
                reference,
                MotionVector { x: 0, y: 0 },
                mb_x,
                mb_y,
                16,
                16,
                vop_rounding_type,
                mode,
                &mut out,
            );
            Some(out)
        }
        PvopMbMotion::OneMv(mv) => {
            let mut out = vec![0u8; 256];
            interpolate_luma_block_into(
                reference,
                mv,
                mb_x,
                mb_y,
                16,
                16,
                vop_rounding_type,
                mode,
                &mut out,
            );
            Some(out)
        }
        PvopMbMotion::FourMv(mvs) => {
            let mut out = vec![0u8; 256];
            let mut tile = vec![0u8; 64];
            for (i, mv) in mvs.iter().enumerate() {
                let bx = mb_x + 8 * (i as i32 & 1);
                let by = mb_y + 8 * (i as i32 >> 1);
                interpolate_luma_block_into(
                    reference,
                    *mv,
                    bx,
                    by,
                    8,
                    8,
                    vop_rounding_type,
                    mode,
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

/// Derive the §6.1.3.4 / §7.6.5 chrominance motion vector for one
/// decoded progressive P-VOP macroblock from its luminance motion
/// vector(s).
///
/// For 4:2:0 the chroma MV is the §7.6.5 `sum / 2K` reduction of the `K`
/// contributing luminance sub-block vectors
/// ([`chroma_mv_from_luma_blocks`], half-sample units):
///
/// * [`PvopMbMotion::Skipped`] → `(0, 0)` (a skipped MB's chroma is the
///   co-located reference, MV zero).
/// * [`PvopMbMotion::OneMv`] → `K = 1`: the chroma MV is the single luma
///   MV reduced by `sum / 2`.
/// * [`PvopMbMotion::FourMv`] → `K = 4`: the four luma MVs summed and
///   reduced by `sum / 8`.
/// * [`PvopMbMotion::Intra`] → `None`.
///
/// The MVs are taken in the unitless-integer representation matching
/// `mode`. In quarter-sample mode §7.6.5 divides each vector by 2
/// before the summation — the sum then sits on a finer grid, handled
/// by [`crate::chroma_mv::chroma_mv_from_luma_blocks_qpel`] — so the
/// returned chroma MV is **always** in half-sample units, ready for
/// the §7.6.2.1 chroma interpolation (§7.6.2.2 applies to luminance
/// only). Returns `None` for an intra macroblock.
pub fn chroma_mv_for_macroblock(
    motion: PvopMbMotion,
    mode: crate::bvop_prediction::BVopSampleMode,
) -> Option<MotionVector> {
    let derive = |mvs: &[MotionVector]| match mode {
        crate::bvop_prediction::BVopSampleMode::HalfPel => {
            crate::chroma_mv::chroma_mv_from_luma_blocks(mvs).ok()
        }
        crate::bvop_prediction::BVopSampleMode::QuarterPel { .. } => {
            crate::chroma_mv::chroma_mv_from_luma_blocks_qpel(mvs).ok()
        }
    };
    match motion {
        PvopMbMotion::Intra => None,
        PvopMbMotion::Skipped => Some(MotionVector { x: 0, y: 0 }),
        PvopMbMotion::OneMv(mv) => derive(&[mv]),
        PvopMbMotion::FourMv(mvs) => derive(&mvs),
    }
}

/// Generate the §7.6.2 8×8 chrominance prediction block (one of Cb /
/// Cr) for one decoded progressive 4:2:0 P-VOP macroblock, half-sample
/// interpolated from a chroma reference plane.
///
/// The chroma MV is derived from the decoded [`PvopMbMotion`] via
/// [`chroma_mv_for_macroblock`] (§7.6.5 `sum / 2K`), then one 8×8
/// [`interpolate_block`] samples the chroma reference. `cmb_x` / `cmb_y`
/// are the macroblock's top-left **chroma** pixel coordinates (`8 *
/// mb_col`, `8 * mb_row` for 4:2:0). Returns a row-major `[u8; 64]`
/// block, or `None` for an intra macroblock (no §7.6.2 prediction).
pub fn predict_chroma_macroblock(
    motion: PvopMbMotion,
    reference: &crate::half_sample::ReferenceVop<'_>,
    cmb_x: i32,
    cmb_y: i32,
    vop_rounding_type: u8,
    mode: crate::bvop_prediction::BVopSampleMode,
) -> Option<Vec<u8>> {
    let mv = chroma_mv_for_macroblock(motion, mode)?;
    Some(crate::half_sample::interpolate_block(
        reference,
        mv.x,
        mv.y,
        cmb_x,
        cmb_y,
        8,
        8,
        vop_rounding_type,
    ))
}

/// Build the §7.6.2 motion-compensated prediction macroblock
/// `p[y][x]` (16×16 luma + 8×8 Cb + 8×8 Cr, 4:2:0) for one decoded
/// progressive P-VOP macroblock, ready as the §7.3 step-2 input to
/// [`crate::reconstruct::reconstruct_inter_macroblock`].
///
/// This is the bridge between the [`MvDriver`]'s decoded
/// [`PvopMbMotion`] and the §7.3 reconstruction layer: where
/// [`predict_luma_macroblock`] / [`predict_chroma_macroblock`] emit
/// flat `Vec<u8>` blocks (useful for direct blit), this function packs
/// the same §7.6.2.1 half-sample interpolation results into the
/// row-major [`InterPredictionMacroblock`] `[[i32; …]; …]` shape that
/// [`reconstruct_inter_macroblock`](crate::reconstruct::reconstruct_inter_macroblock)
/// adds the §7.4 residual to.
///
/// * `luma_ref` is the previous reconstructed VOP's luma plane; `cb_ref`
///   / `cr_ref` are its two 4:2:0 chroma planes.
/// * `mb_x` / `mb_y` are the macroblock's top-left **luma** pixel
///   coordinates in the current VOP (`16 * mb_col`, `16 * mb_row`); the
///   chroma origin is `(mb_x / 2, mb_y / 2)` per §6.1.3.4 4:2:0
///   sub-sampling.
/// * MVs are in half-sample units (the progressive `quarter_sample == 0`
///   path); the chroma MV is the §7.6.5 `sum / 2K` reduction
///   ([`chroma_mv_for_macroblock`]).
///
/// Returns [`None`] for an intra macroblock (it carries no §7.6.2
/// prediction — its samples come from the §7.4 intra texture path).
// The §7.6.2 / §7.3 inputs map one-for-one; see reconstruct_pvop_macroblock.
#[allow(clippy::too_many_arguments)]
pub fn predict_inter_macroblock(
    motion: PvopMbMotion,
    luma_ref: &crate::half_sample::ReferenceVop<'_>,
    cb_ref: &crate::half_sample::ReferenceVop<'_>,
    cr_ref: &crate::half_sample::ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
    vop_rounding_type: u8,
    mode: crate::bvop_prediction::BVopSampleMode,
) -> Option<crate::reconstruct::InterPredictionMacroblock> {
    use crate::reconstruct::{
        InterPredictionMacroblock, MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE,
    };

    if matches!(motion, PvopMbMotion::Intra) {
        return None;
    }

    let luma_flat = predict_luma_macroblock(motion, luma_ref, mb_x, mb_y, vop_rounding_type, mode)?;

    let cmb_x = mb_x / 2;
    let cmb_y = mb_y / 2;
    let cb_flat = predict_chroma_macroblock(motion, cb_ref, cmb_x, cmb_y, vop_rounding_type, mode)?;
    let cr_flat = predict_chroma_macroblock(motion, cr_ref, cmb_x, cmb_y, vop_rounding_type, mode)?;

    let mut out = InterPredictionMacroblock::zero();
    for y in 0..MACROBLOCK_LUMA_SIDE {
        for x in 0..MACROBLOCK_LUMA_SIDE {
            out.luma[y][x] = luma_flat[y * MACROBLOCK_LUMA_SIDE + x] as i32;
        }
    }
    for y in 0..MACROBLOCK_CHROMA_SIDE {
        for x in 0..MACROBLOCK_CHROMA_SIDE {
            out.cb[y][x] = cb_flat[y * MACROBLOCK_CHROMA_SIDE + x] as i32;
            out.cr[y][x] = cr_flat[y * MACROBLOCK_CHROMA_SIDE + x] as i32;
        }
    }
    Some(out)
}

/// Fully reconstruct one progressive P-VOP **inter** macroblock
/// end-to-end: §7.6.2 motion-compensated prediction
/// ([`predict_inter_macroblock`]) plus the §7.3 step-2 add of the §7.4
/// residual and the step-3 `[0, 2^bpp - 1]` display clip
/// ([`reconstruct_inter_macroblock`](crate::reconstruct::reconstruct_inter_macroblock)).
///
/// This is the single call that closes the progressive-P-VOP
/// motion-compensation path: given a [`MvDriver`]-decoded
/// [`PvopMbMotion`], the previous reconstructed VOP's three planes, and
/// the macroblock's already-decoded §7.4 residual, it produces the
/// `d[y][x]` [`ReconstructedMacroblock`] ready to blit into the current
/// VOP's frame buffer.
///
/// * `motion` — the driver's per-MB outcome (`OneMv` / `FourMv` /
///   `Skipped`). A `Skipped` MB has a zero residual *and* a zero MV, so
///   the caller passes a zero `residual`; the result is the co-located
///   reference samples.
/// * `residual` — the §7.4 inverse-quantised + inverse-transformed
///   signed residual `f[y][x]` for this macroblock.
/// * `mb_x` / `mb_y` — top-left **luma** pixel coordinates (`16 *
///   mb_col`, `16 * mb_row`).
/// * `bits_per_pixel` — §6.3.3 `bits_per_pixel` (8 unless `not_8_bit`).
///
/// Returns [`None`] for an intra macroblock — an intra MB is
/// reconstructed by the §7.4 intra texture path + the §7.3 step-1
/// `d = f` (see [`reconstruct_intra_macroblock`](crate::reconstruct::reconstruct_intra_macroblock)),
/// not this motion-compensated path.
//
// Each argument is a distinct §7.6.2 / §7.3 input (the decoded motion,
// the three reference planes, the residual, the MB origin pair, plus the
// VOP-level rounding-control bit and bit-depth). Bundling them into a
// struct would add a shim layer without simplifying the call site.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_pvop_macroblock(
    motion: PvopMbMotion,
    luma_ref: &crate::half_sample::ReferenceVop<'_>,
    cb_ref: &crate::half_sample::ReferenceVop<'_>,
    cr_ref: &crate::half_sample::ReferenceVop<'_>,
    residual: &crate::block::InterMacroblock,
    mb_x: i32,
    mb_y: i32,
    vop_rounding_type: u8,
    mode: crate::bvop_prediction::BVopSampleMode,
    bits_per_pixel: u32,
) -> Option<crate::reconstruct::ReconstructedMacroblock> {
    let prediction = predict_inter_macroblock(
        motion,
        luma_ref,
        cb_ref,
        cr_ref,
        mb_x,
        mb_y,
        vop_rounding_type,
        mode,
    )?;
    Some(crate::reconstruct::reconstruct_inter_macroblock(
        &prediction,
        residual,
        bits_per_pixel,
    ))
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

    /// Half-pel sample mode shorthand for the legacy-path tests.
    const HP: crate::bvop_prediction::BVopSampleMode =
        crate::bvop_prediction::BVopSampleMode::HalfPel;

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
        let block =
            predict_luma_macroblock(PvopMbMotion::Skipped, &reference, 16, 0, 0, HP).unwrap();
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
        assert!(predict_luma_macroblock(PvopMbMotion::Intra, &reference, 0, 0, 0, HP).is_none());
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
        let block =
            predict_luma_macroblock(PvopMbMotion::OneMv(mv), &reference, 0, 0, 0, HP).unwrap();
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
            predict_luma_macroblock(PvopMbMotion::FourMv(mvs), &reference, 0, 0, 0, HP).unwrap();
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
            let block = predict_luma_macroblock(motion, &reference, mb_x, mb_y, 0, HP).unwrap();
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

    // ------------------------------------------------------------------
    // §6.1.3.4 / §7.6.5 chroma-MV derivation + chroma prediction block.
    // ------------------------------------------------------------------

    #[test]
    fn chroma_mv_one_mv_is_half_of_luma() {
        // §7.6.5 K=1: chroma MV = sum / 2 (half-sample units).
        let m = PvopMbMotion::OneMv(MotionVector { x: 4, y: -2 });
        assert_eq!(
            chroma_mv_for_macroblock(m, HP),
            Some(MotionVector { x: 2, y: -1 })
        );
    }

    #[test]
    fn chroma_mv_four_mv_sums_then_reduces() {
        // §7.6.5 K=4: sum (16, 16) / 8 = (2, 2).
        let m = PvopMbMotion::FourMv([MotionVector { x: 4, y: 4 }; 4]);
        assert_eq!(
            chroma_mv_for_macroblock(m, HP),
            Some(MotionVector { x: 2, y: 2 })
        );
    }

    #[test]
    fn chroma_mv_skipped_is_zero_intra_is_none() {
        assert_eq!(
            chroma_mv_for_macroblock(PvopMbMotion::Skipped, HP),
            Some(MotionVector { x: 0, y: 0 })
        );
        assert_eq!(chroma_mv_for_macroblock(PvopMbMotion::Intra, HP), None);
    }

    #[test]
    fn chroma_prediction_block_integer_shift() {
        // K=1, luma MV (4, 0) → chroma MV (2, 0) half-pel = integer
        // shift (+1, 0). 8x8 chroma block from a gradient chroma plane.
        let w = 24;
        let h = 24;
        let plane = gradient_plane(w, h);
        let reference = ReferenceVop::new(&plane, w, h).unwrap();
        let m = PvopMbMotion::OneMv(MotionVector { x: 4, y: 0 });
        // MB col 1 → chroma origin (8, 0).
        let block = predict_chroma_macroblock(m, &reference, 8, 0, 0, HP).unwrap();
        for j in 0..8 {
            for i in 0..8 {
                // chroma MV (2,0) = shift (+1,0): block(i,j) =
                // reference(8 + i + 1, 0 + j).
                assert_eq!(block[8 * j + i], ((8 + i + 1 + j) & 0xff) as u8);
            }
        }
    }

    #[test]
    fn chroma_prediction_intra_is_none() {
        let plane = gradient_plane(8, 8);
        let reference = ReferenceVop::new(&plane, 8, 8).unwrap();
        assert!(predict_chroma_macroblock(PvopMbMotion::Intra, &reference, 0, 0, 0, HP).is_none());
    }

    // ------------------------------------------------------------------
    // §7.6.2 → §7.3 bridge: InterPredictionMacroblock packing.
    // ------------------------------------------------------------------

    #[test]
    fn inter_prediction_macroblock_intra_is_none() {
        let luma = gradient_plane(16, 16);
        let chroma = gradient_plane(8, 8);
        let luma_ref = ReferenceVop::new(&luma, 16, 16).unwrap();
        let cb_ref = ReferenceVop::new(&chroma, 8, 8).unwrap();
        let cr_ref = ReferenceVop::new(&chroma, 8, 8).unwrap();
        assert!(predict_inter_macroblock(
            PvopMbMotion::Intra,
            &luma_ref,
            &cb_ref,
            &cr_ref,
            0,
            0,
            0,
            HP
        )
        .is_none());
    }

    #[test]
    fn inter_prediction_macroblock_packs_one_mv() {
        // 1-MV integer shift: luma MV (4, 2) = shift (+2, +1); chroma MV
        // = sum/2 = (2, 1) = shift (+1, 0)+half(0,1)? (2,1)→ x even
        // (shift +1), y odd (half 1). Use even chroma to stay exact:
        // luma MV (4, 4) → chroma (2, 2) = integer shift (+1, +1).
        let lw = 48;
        let lh = 48;
        let cw = 24;
        let ch = 24;
        let luma = gradient_plane(lw, lh);
        let cb = gradient_plane(cw, ch);
        let cr = gradient_plane(cw, ch);
        let luma_ref = ReferenceVop::new(&luma, lw, lh).unwrap();
        let cb_ref = ReferenceVop::new(&cb, cw, ch).unwrap();
        let cr_ref = ReferenceVop::new(&cr, cw, ch).unwrap();

        let motion = PvopMbMotion::OneMv(MotionVector { x: 4, y: 4 });
        // Macroblock at luma (16, 16) → chroma (8, 8).
        let pred =
            predict_inter_macroblock(motion, &luma_ref, &cb_ref, &cr_ref, 16, 16, 0, HP).unwrap();

        // Luma: shift (+2, +2). p[y][x] = ref(16 + x + 2, 16 + y + 2).
        for y in 0..16 {
            for x in 0..16 {
                let expect = ((16 + x + 2 + 16 + y + 2) & 0xff) as i32;
                assert_eq!(pred.luma[y][x], expect, "luma ({x},{y})");
            }
        }
        // Chroma: MV (2,2) = shift (+1,+1). p[y][x] = ref(8 + x + 1,
        // 8 + y + 1).
        for y in 0..8 {
            for x in 0..8 {
                let expect = ((8 + x + 1 + 8 + y + 1) & 0xff) as i32;
                assert_eq!(pred.cb[y][x], expect, "cb ({x},{y})");
                assert_eq!(pred.cr[y][x], expect, "cr ({x},{y})");
            }
        }
    }

    #[test]
    fn inter_prediction_macroblock_feeds_reconstruction() {
        // The packed prediction adds cleanly to a §7.4 residual via the
        // §7.3 reconstruction path: a zero MV + zero residual returns the
        // co-located reference, and a constant residual offsets it.
        use crate::block::InterMacroblock;
        use crate::reconstruct::{
            reconstruct_inter_macroblock, MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE,
        };

        let lw = 32;
        let lh = 32;
        let cw = 16;
        let ch = 16;
        let luma = gradient_plane(lw, lh);
        let cb = gradient_plane(cw, ch);
        let cr = gradient_plane(cw, ch);
        let luma_ref = ReferenceVop::new(&luma, lw, lh).unwrap();
        let cb_ref = ReferenceVop::new(&cb, cw, ch).unwrap();
        let cr_ref = ReferenceVop::new(&cr, cw, ch).unwrap();

        // Skipped MB at (0,0): zero MV → prediction = co-located ref.
        let pred = predict_inter_macroblock(
            PvopMbMotion::Skipped,
            &luma_ref,
            &cb_ref,
            &cr_ref,
            0,
            0,
            0,
            HP,
        )
        .unwrap();

        // Residual = +5 everywhere.
        let residual = InterMacroblock {
            luma: [[5i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[5i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[5i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        };
        let recon = reconstruct_inter_macroblock(&pred, &residual, 8);

        for y in 0..16 {
            for x in 0..16 {
                let p = ((x + y) & 0xff) as i32;
                assert_eq!(recon.luma[y][x], (p + 5).min(255), "luma ({x},{y})");
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                let p = ((x + y) & 0xff) as i32;
                assert_eq!(recon.cb[y][x], (p + 5).min(255));
                assert_eq!(recon.cr[y][x], (p + 5).min(255));
            }
        }
    }

    // ------------------------------------------------------------------
    // §7.6.2 + §7.3 end-to-end P-VOP macroblock reconstruction.
    // ------------------------------------------------------------------

    #[test]
    fn reconstruct_pvop_macroblock_intra_is_none() {
        use crate::block::InterMacroblock;
        use crate::reconstruct::{MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};
        let luma = gradient_plane(16, 16);
        let chroma = gradient_plane(8, 8);
        let luma_ref = ReferenceVop::new(&luma, 16, 16).unwrap();
        let cb_ref = ReferenceVop::new(&chroma, 8, 8).unwrap();
        let cr_ref = ReferenceVop::new(&chroma, 8, 8).unwrap();
        let residual = InterMacroblock {
            luma: [[0i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        };
        assert!(reconstruct_pvop_macroblock(
            PvopMbMotion::Intra,
            &luma_ref,
            &cb_ref,
            &cr_ref,
            &residual,
            0,
            0,
            0,
            HP,
            8,
        )
        .is_none());
    }

    #[test]
    fn reconstruct_pvop_macroblock_one_mv_plus_residual() {
        // §7.3 d = p + f, clipped. Luma MV (4, 4) = shift (+2, +2);
        // chroma MV (2, 2) = shift (+1, +1). A per-sample varying residual
        // verifies the add is positional, not a constant.
        use crate::block::InterMacroblock;
        use crate::reconstruct::{MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};

        let lw = 48;
        let lh = 48;
        let cw = 24;
        let ch = 24;
        let luma = gradient_plane(lw, lh);
        let cb = gradient_plane(cw, ch);
        let cr = gradient_plane(cw, ch);
        let luma_ref = ReferenceVop::new(&luma, lw, lh).unwrap();
        let cb_ref = ReferenceVop::new(&cb, cw, ch).unwrap();
        let cr_ref = ReferenceVop::new(&cr, cw, ch).unwrap();

        let mut residual = InterMacroblock {
            luma: [[0i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        };
        // A checkerboard ±3 residual (kept small so no clip triggers).
        for y in 0..MACROBLOCK_LUMA_SIDE {
            for x in 0..MACROBLOCK_LUMA_SIDE {
                residual.luma[y][x] = if (x + y) % 2 == 0 { 3 } else { -3 };
            }
        }
        for y in 0..MACROBLOCK_CHROMA_SIDE {
            for x in 0..MACROBLOCK_CHROMA_SIDE {
                residual.cb[y][x] = if (x + y) % 2 == 0 { 2 } else { -2 };
                residual.cr[y][x] = if (x + y) % 2 == 0 { 1 } else { -1 };
            }
        }

        let motion = PvopMbMotion::OneMv(MotionVector { x: 4, y: 4 });
        let recon = reconstruct_pvop_macroblock(
            motion, &luma_ref, &cb_ref, &cr_ref, &residual, 16, 16, 0, HP, 8,
        )
        .unwrap();

        for y in 0..16 {
            for x in 0..16 {
                // p = ref(16 + x + 2, 16 + y + 2); f = ±3.
                let p = ((16 + x + 2 + 16 + y + 2) & 0xff) as i32;
                let f = if (x + y) % 2 == 0 { 3 } else { -3 };
                assert_eq!(recon.luma[y][x], (p + f).clamp(0, 255), "luma ({x},{y})");
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                let p = ((8 + x + 1 + 8 + y + 1) & 0xff) as i32;
                let fcb = if (x + y) % 2 == 0 { 2 } else { -2 };
                let fcr = if (x + y) % 2 == 0 { 1 } else { -1 };
                assert_eq!(recon.cb[y][x], (p + fcb).clamp(0, 255), "cb ({x},{y})");
                assert_eq!(recon.cr[y][x], (p + fcr).clamp(0, 255), "cr ({x},{y})");
            }
        }
    }

    #[test]
    fn frame_level_pvop_decode_predict_reconstruct() {
        // Full §7.6 + §7.3 P-VOP path over a 2×2-macroblock frame:
        // drive the motion bitstream through MvDriver, then reconstruct
        // every macroblock against a reference frame and blit into a
        // 32×32 luma + 16×16 chroma frame buffer. Asserts the assembled
        // luma plane matches the per-MB prediction (zero residual → the
        // reconstruction is exactly the §7.6.2 prediction).
        use crate::block::InterMacroblock;
        use crate::reconstruct::{MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};

        // Reference planes (with padding margin so half-pel fetches at
        // the right/bottom edge clamp cleanly): 48×48 luma, 24×24 chroma.
        let lw = 48;
        let lh = 48;
        let cw = 24;
        let ch = 24;
        let luma = gradient_plane(lw, lh);
        let cb = gradient_plane(cw, ch);
        let cr = gradient_plane(cw, ch);
        let luma_ref = ReferenceVop::new(&luma, lw, lh).unwrap();
        let cb_ref = ReferenceVop::new(&cb, cw, ch).unwrap();
        let cr_ref = ReferenceVop::new(&cr, cw, ch).unwrap();

        // Decode four inter MBs (same bitstream as
        // end_to_end_pvop_mv_decode_and_predict): all reconstruct to
        // MV (1,1) except MB(1,1) = (0,0).
        let mut driver = MvDriver::new(2, 2, 1);
        let mb00 = [0x48u8]; // 010 010 → (+1,+1)
        let mut br = BitReader::new(&mb00);
        let m00 = driver
            .decode_macroblock(&mut br, 0, 0, false, Some(DerivedMbType::Inter))
            .unwrap();
        let zeros = zero_delta_bits(1);
        let mut br = BitReader::new(&zeros);
        let m01 = driver
            .decode_macroblock(&mut br, 0, 1, false, Some(DerivedMbType::Inter))
            .unwrap();
        let mut br = BitReader::new(&zeros);
        let m10 = driver
            .decode_macroblock(&mut br, 1, 0, false, Some(DerivedMbType::Inter))
            .unwrap();
        let mb11 = [0x6Cu8]; // 011 011 → (-1,-1)
        let mut br = BitReader::new(&mb11);
        let m11 = driver
            .decode_macroblock(&mut br, 1, 1, false, Some(DerivedMbType::Inter))
            .unwrap();

        // Frame buffers: 32×32 luma, 16×16 Cb/Cr.
        let mut frame_y = vec![0i32; 32 * 32];
        let mut frame_cb = vec![0i32; 16 * 16];
        let mut frame_cr = vec![0i32; 16 * 16];

        let zero_residual = InterMacroblock {
            luma: [[0i32; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[0i32; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        };
        let motions: [(usize, usize, PvopMbMotion); 4] =
            [(0, 0, m00), (0, 1, m01), (1, 0, m10), (1, 1, m11)];
        for (mb_row, mb_col, motion) in motions {
            let mb_x = 16 * mb_col as i32;
            let mb_y = 16 * mb_row as i32;
            let recon = reconstruct_pvop_macroblock(
                motion,
                &luma_ref,
                &cb_ref,
                &cr_ref,
                &zero_residual,
                mb_x,
                mb_y,
                0,
                HP,
                8,
            )
            .unwrap();
            // Blit luma.
            for y in 0..16 {
                for x in 0..16 {
                    frame_y[(mb_y as usize + y) * 32 + (mb_x as usize + x)] = recon.luma[y][x];
                }
            }
            // Blit chroma at half resolution.
            let cmb_x = (mb_x / 2) as usize;
            let cmb_y = (mb_y / 2) as usize;
            for y in 0..8 {
                for x in 0..8 {
                    frame_cb[(cmb_y + y) * 16 + (cmb_x + x)] = recon.cb[y][x];
                    frame_cr[(cmb_y + y) * 16 + (cmb_x + x)] = recon.cr[y][x];
                }
            }
        }

        // Cross-check: the assembled luma equals the per-MB §7.6.2
        // prediction (zero residual). Recompute the prediction for one
        // sample of each MB and compare to the frame buffer.
        for (mb_row, mb_col, motion) in motions {
            let mb_x = 16 * mb_col as i32;
            let mb_y = 16 * mb_row as i32;
            let pred = predict_luma_macroblock(motion, &luma_ref, mb_x, mb_y, 0, HP).unwrap();
            for y in [0usize, 7, 15] {
                for x in [0usize, 7, 15] {
                    let fb = frame_y[(mb_y as usize + y) * 32 + (mb_x as usize + x)];
                    assert_eq!(
                        fb,
                        pred[16 * y + x] as i32,
                        "luma frame ({},{}) MB({mb_row},{mb_col})",
                        mb_x as usize + x,
                        mb_y as usize + y
                    );
                }
            }
        }

        // The skip-free top-left MB used MV (1,1) → half-pel diagonal;
        // confirm the frame value is the bilinear of the four reference
        // neighbours (rc = 0) for one interior sample.
        let p00 = frame_y[8 * 32 + 8]; // MB(0,0), sample (8,8)
                                       // sample(8,8) origin in ref: base (8,8), half (1,1) → bilinear of
                                       // (16),(17),(17),(18) … using (x+y) gradient: base_x=8,base_y=8.
        let a = (8 + 8) & 0xff;
        let b = (8 + 1 + 8) & 0xff;
        let c = (8 + 8 + 1) & 0xff;
        let d = (8 + 1 + 8 + 1) & 0xff;
        assert_eq!(p00, (a + b + c + d + 2) / 4);
    }
}

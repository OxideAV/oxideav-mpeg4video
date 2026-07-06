//! Bitstream-driven per-VOP macroblock walks (§6.2.5
//! `motion_shape_texture()` → §6.2.6 `macroblock()` loops).
//!
//! The per-stage modules decode one macroblock's header
//! ([`crate::macroblock::parse_macroblock_header`]), motion
//! ([`crate::pvop_mv::MvDriver`]) and texture
//! ([`crate::block::decode_intra_block_full`] /
//! [`crate::block::decode_inter_macroblock`]); the frame-assembly
//! drivers ([`crate::frame_decode`]) turn *already-decoded* per-MB
//! content into a blitted frame. This module closes the gap between the
//! two: it walks a rectangular VOP's macroblocks in raster order
//! straight off the bitstream, threading the cross-macroblock state the
//! §6.2.6 layer requires —
//!
//! * the running quantiser scale (`vop_quant` adjusted by each
//!   macroblock's `dquant`, §6.3.6 Table 6-32, clipped to
//!   `[1, 2^quant_precision - 1]`),
//! * the §6.3.5 / Table 6-25 `use_intra_dc_vlc` decision per macroblock
//!   (from `intra_dc_vlc_thr` and the running quantiser),
//! * the §7.4.3 intra DC/AC predictor grid
//!   ([`crate::neighbour::IntraBlockGrid`], fed with each decoded intra
//!   block's traced `QF` / `F[0][0]`),
//! * the §7.6.5 / Figure 7-34 motion-vector predictor grid (owned by
//!   the [`MvDriver`]),
//!
//! and returns the per-MB content the frame assemblers consume
//! ([`ReconstructedMacroblock`]s for an I-VOP, [`PVopMbContent`]s for a
//! P-VOP).
//!
//! ## Scope
//!
//! Rectangular shape, progressive, non-data-partitioned,
//! non-short-video-header VOPs. Half-sample
//! (§7.6.2.1) and quarter-sample (§7.6.2.2) VOLs both walk here — the
//! §7.6.3 motion syntax is unit-agnostic; the sub-pel grid only
//! matters at frame assembly.
//! Out-of-scope VOL configurations are rejected with a typed
//! [`VopDecodeError::Unsupported`] before any bit is consumed, so the
//! reader never drifts. The data-partitioned layouts live in
//! [`crate::data_partition`]; the interlaced texture path (dct_type
//! re-ordering + alternate vertical scan) is a later milestone.

use crate::bitreader::BitReader;
use crate::block::{
    decode_b_vop_inter_macroblock, decode_inter_macroblock, decode_intra_block_full,
    intra_quant_matrix, nonintra_quant_matrix, pattern_code, BlockAssemblyError, IntraMacroblock,
    MacroblockTextureContext,
};
use crate::bvop_mv::{
    BVopMbTexturedDecode, BVopMvDriver, BVopMvDriverError, BVopTextureParams, CoLocatedAnchor,
};
use crate::data_partition::use_intra_dc_vlc;
use crate::frame_decode::{PVopMbContent, SGmcMbContent};
use crate::macroblock::{parse_macroblock_header, MacroblockParseError};
use crate::motion::{averaged_motion_vector, DirectCoLocatedMv, MotionVector, AMV_PIXEL_COUNT};
use crate::neighbour::{BlockNeighbour, IntraBlockGrid};
use crate::pvop_mv::{MvDriver, PvopMbMotion, PvopMvError};
use crate::reconstruct::{reconstruct_intra_macroblock, ReconstructedMacroblock};
use crate::sprite::SpriteTrajectory;
use crate::video_packet::{
    consume_next_resync_marker, parse_video_packet_header, probe_resync_marker, VideoPacketContext,
    VideoPacketHeader, VideoPacketParseError,
};
use crate::vol::{SpriteEnable, SpriteWarpingAccuracy, VolHeader};
use crate::vop::{VopCodingType, VopHeader};
use crate::warp::WarpGeometry;

/// Errors raised by the bitstream-driven VOP macroblock walks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VopDecodeError {
    /// The VOL / VOP configuration selects a branch this walk does not
    /// decode (data partitioning, interlace, OBMC, quarter-sample, a
    /// mismatched VOP coding type, or an uncoded VOP). The name of the
    /// offending gate is carried verbatim.
    Unsupported(&'static str),
    /// A §6.2.6 macroblock-header parse failed.
    Macroblock(MacroblockParseError),
    /// A §6.2.6.2 / §7.6.3 motion-vector decode failed.
    Motion(PvopMvError),
    /// A §6.2.7 / §7.4 texture decode failed.
    Texture(BlockAssemblyError),
    /// A B-VOP macroblock decode (header / motion / residual) failed.
    BVop(BVopMvDriverError),
    /// A §6.2.5 `video_packet_header` failed to parse.
    VideoPacket(VideoPacketParseError),
    /// A §6.2.5.3 data-partitioned partition structure failed to parse
    /// (missing `dc_marker` / `motion_marker`, truncated partition, or
    /// a header field inside a partition failed).
    DataPartition(crate::data_partition::DataPartitionError),
    /// A video packet's `macroblock_number` did not equal the next
    /// macroblock in raster order — the packet skips ahead (an
    /// error-resilience gap this clean-path walk does not conceal).
    VideoPacketSkip {
        /// The `macroblock_number` the packet header carried.
        packet_mb: u32,
        /// The raster index the walk expected.
        expected_mb: u32,
    },
}

impl core::fmt::Display for VopDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VopDecodeError::Unsupported(name) => {
                write!(f, "vop decode: unsupported configuration '{name}'")
            }
            VopDecodeError::Macroblock(e) => write!(f, "vop decode: macroblock header: {e}"),
            VopDecodeError::Motion(e) => write!(f, "vop decode: motion: {e}"),
            VopDecodeError::Texture(e) => write!(f, "vop decode: texture: {e}"),
            VopDecodeError::BVop(e) => write!(f, "vop decode: b-vop macroblock: {e}"),
            VopDecodeError::VideoPacket(e) => write!(f, "vop decode: video_packet_header: {e}"),
            VopDecodeError::DataPartition(e) => write!(f, "vop decode: data partition: {e}"),
            VopDecodeError::VideoPacketSkip {
                packet_mb,
                expected_mb,
            } => write!(
                f,
                "vop decode: video packet resumes at macroblock {packet_mb} \
                 but {expected_mb} was expected"
            ),
        }
    }
}

impl std::error::Error for VopDecodeError {}

impl From<VideoPacketParseError> for VopDecodeError {
    fn from(e: VideoPacketParseError) -> Self {
        VopDecodeError::VideoPacket(e)
    }
}

impl From<MacroblockParseError> for VopDecodeError {
    fn from(e: MacroblockParseError) -> Self {
        VopDecodeError::Macroblock(e)
    }
}

impl From<PvopMvError> for VopDecodeError {
    fn from(e: PvopMvError) -> Self {
        VopDecodeError::Motion(e)
    }
}

impl From<BlockAssemblyError> for VopDecodeError {
    fn from(e: BlockAssemblyError) -> Self {
        VopDecodeError::Texture(e)
    }
}

/// The macroblock grid of a rectangular VOL: `((width + 15) / 16,
/// (height + 15) / 16)` — §6.3.3 `video_object_layer_width` /
/// `_height` cover the visible samples; a partial right / bottom
/// macroblock is still a whole coded macroblock.
pub fn vop_mb_dimensions(vol: &VolHeader) -> (usize, usize) {
    (
        (vol.width as usize).div_ceil(16),
        (vol.height as usize).div_ceil(16),
    )
}

/// Validate the shared VOL gates of the rectangular walks.
///
/// `interlaced == 1` is walk-specific: the I-VOP walk decodes it (the
/// §6.2.6.3 `dct_type` field DCT), the inter walks still gate it off
/// until the §7.7.2 field-MC paths are wired.
fn check_vol_supported(vol: &VolHeader) -> Result<(), VopDecodeError> {
    if vol.video_object_layer_shape != 0 {
        return Err(VopDecodeError::Unsupported("non-rectangular shape"));
    }
    Ok(())
}

/// The combined-syntax I-/P-/S-walk gate: a `data_partitioned` VOL
/// rearranges the macroblock layer into the §6.2.5.3 partitions, which
/// the [`decode_i_vop_macroblocks_dp`] / [`decode_p_vop_macroblocks_dp`]
/// walks decode instead. (B-VOPs always use the combined syntax —
/// §6.2.5.3 NOTE: data partitioning is not supported in B-VOPs — so the
/// B walks do not call this.)
fn check_combined_syntax(vol: &VolHeader) -> Result<(), VopDecodeError> {
    if vol.data_partitioned {
        return Err(VopDecodeError::Unsupported("data_partitioned"));
    }
    Ok(())
}

/// The inter walks' interlaced gate (see [`check_vol_supported`]).
fn check_progressive(vol: &VolHeader) -> Result<(), VopDecodeError> {
    if vol.interlaced {
        return Err(VopDecodeError::Unsupported("interlaced"));
    }
    Ok(())
}

use crate::reconstruct::inverse_field_dct_luma;

/// The §6.2.6.3 forward reference-field selection bits of a
/// field-predicted P-/S(GMC)-VOP macroblock (`field_prediction == 1`),
/// as raw §6.3.7.2 bits (`false` = top reference field). `None` for a
/// frame-predicted (or progressive) macroblock.
fn header_forward_field_refs(header: &crate::macroblock::MacroblockHeader) -> Option<(bool, bool)> {
    header
        .interlaced_info
        .as_ref()
        .and_then(|info| info.field_prediction)
        .and_then(|fp| fp.forward)
        .map(|(top, bottom)| (top.as_bit(), bottom.as_bit()))
}

/// Whether one parsed macroblock header selected the §6.2.6.3 field
/// DCT (`dct_type == 1`).
fn header_field_dct(header: &crate::macroblock::MacroblockHeader) -> bool {
    header
        .interlaced_info
        .as_ref()
        .and_then(|info| info.dct_type)
        .map(|d| matches!(d, crate::interlaced_information::DctType::Field))
        .unwrap_or(false)
}

/// The running quantiser ceiling `2^quant_precision - 1` (§6.3.3).
fn max_quantiser_scale(vol: &VolHeader) -> u32 {
    (1u32 << vol.quant_precision) - 1
}

/// Build the [`VideoPacketContext`] for the walks' resync handling.
fn video_packet_context(vol: &VolHeader, vop: &VopHeader) -> VideoPacketContext {
    VideoPacketContext {
        coding_type: vop.coding_type,
        fcode_fwd: vop.fcode_fwd,
        fcode_bwd: vop.fcode_bwd,
        quant_precision: vol.quant_precision,
        time_increment_resolution: vol.time_increment_resolution,
        video_object_layer_shape: vol.video_object_layer_shape,
        resync_marker_disable: vol.resync_marker_disable,
        newpred_enable: vol.newpred_enable,
        reduced_resolution_vop_enable: vol.reduced_resolution_vop_enable,
        sprite_gmc: matches!(vol.sprite_enable, SpriteEnable::Gmc),
        total_macroblocks: crate::video_packet::total_macroblocks(
            u32::from(vol.width),
            u32::from(vol.height),
        ),
    }
}

/// Non-destructively test whether a §6.2.5.2 video packet follows: the
/// §5.2.5 stuffing run (`0` then `1`s to the byte boundary) and the
/// byte-aligned `resync_marker`.
///
/// The marker's `16 + fcode` leading zeros cannot alias any legal
/// macroblock-layer VLC, so a positive probe on a conforming stream is
/// definitive.
fn video_packet_follows(br: &BitReader<'_>, ctx: &VideoPacketContext) -> bool {
    let mut probe = br.clone();
    if consume_next_resync_marker(&mut probe).is_err() {
        return false;
    }
    probe_resync_marker(&probe, ctx.coding_type, ctx.fcode_fwd, ctx.fcode_bwd)
}

/// Consume one video packet header at a macroblock boundary and check
/// its `macroblock_number` resumes exactly at `expected_mb` (this walk
/// decodes clean streams; error-resilience gaps are out of scope).
fn enter_video_packet(
    br: &mut BitReader<'_>,
    ctx: &VideoPacketContext,
    expected_mb: u32,
) -> Result<VideoPacketHeader, VopDecodeError> {
    let header = parse_video_packet_header(br, ctx)?;
    if header.macroblock_number != expected_mb {
        return Err(VopDecodeError::VideoPacketSkip {
            packet_mb: header.macroblock_number,
            expected_mb,
        });
    }
    Ok(header)
}

/// Apply one macroblock's `dquant` delta to the running quantiser
/// scale, clipped to `[1, 2^quant_precision - 1]` (§6.3.6).
fn apply_dquant(running_qp: u32, delta: Option<i8>, max_qp: u32) -> u32 {
    match delta {
        Some(d) => (running_qp as i64 + d as i64).clamp(1, max_qp as i64) as u32,
        None => running_qp,
    }
}

/// Decode the six §6.2.7 `block(i)` bodies of one **intra** macroblock
/// with per-block §7.4.3 predictors from `grid`, recording each decoded
/// block back into the grid, and assemble the [`IntraMacroblock`].
///
/// This is the grid-threaded sibling of
/// [`crate::block::decode_intra_macroblock`] (which applies one
/// [`BlockPredictors`](crate::block::BlockPredictors) to all six
/// blocks): here every block resolves its own Figure 7-5 `A` / `B` /
/// `C` neighbours against the running per-VOP grid, which is what a
/// real frame walk requires.
#[allow(clippy::too_many_arguments)]
fn decode_intra_mb_with_grid(
    br: &mut BitReader<'_>,
    grid: &mut IntraBlockGrid,
    mb_row: usize,
    mb_col: usize,
    coded: [bool; 6],
    use_dc_vlc: bool,
    field_dct: bool,
    ctx: MacroblockTextureContext,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<IntraMacroblock, VopDecodeError> {
    let mut blocks: [[[i32; 8]; 8]; 6] = [[[0i32; 8]; 8]; 6];
    for (i, block) in blocks.iter_mut().enumerate() {
        let predictors =
            grid.predictors_for(mb_row, mb_col, i, ctx.bits_per_pixel, ctx.quantiser_scale);
        let full =
            decode_intra_block_full(br, i, coded[i], use_dc_vlc, ctx, predictors, quant_matrix)?;
        grid.record(
            mb_row,
            mb_col,
            i,
            Some(BlockNeighbour::from_qf(
                &full.qf,
                full.dc,
                ctx.quantiser_scale,
            )),
        );
        *block = full.spatial;
    }

    // Assemble the 16×16 luminance plane from the four 8×8 luma blocks
    // (Figure 6-8: 0 top-left, 1 top-right, 2 bottom-left, 3
    // bottom-right).
    let mut luma = [[0i32; 16]; 16];
    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];
    for (b, &(row_off, col_off)) in LUMA_OFFSETS.iter().enumerate() {
        for y in 0..8 {
            for x in 0..8 {
                luma[row_off + y][col_off + x] = blocks[b][y][x];
            }
        }
    }
    // §7.7.1: with dct_type == 1 the assembled upper/lower halves are
    // the top/bottom fields — inverse-permute back to frame lines.
    if field_dct {
        luma = inverse_field_dct_luma(&luma);
    }
    Ok(IntraMacroblock {
        luma,
        cb: blocks[4],
        cr: blocks[5],
    })
}

/// Decode a complete rectangular progressive **I-VOP**'s macroblock
/// layer straight off the bitstream, returning the reconstructed
/// macroblocks in raster order.
///
/// `br` must be positioned at the first bit of
/// `combined_motion_shape_texture()` — i.e. immediately after the VOP
/// header ([`crate::vop::parse_vop_header_body`] leaves it there). On
/// success it sits immediately after the last macroblock's last texture
/// block (the caller consumes any `next_start_code()` stuffing).
///
/// The walk threads all the §6.2.6 cross-macroblock state: the running
/// quantiser (`vop_quant` + per-MB `dquant`), the per-MB Table 6-25
/// `use_intra_dc_vlc` decision, and the §7.4.3 intra predictor grid
/// with per-block Figure 7-5 neighbour resolution.
pub fn decode_i_vop_macroblocks(
    br: &mut BitReader<'_>,
    vol: &VolHeader,
    vop: &VopHeader,
) -> Result<Vec<ReconstructedMacroblock>, VopDecodeError> {
    check_vol_supported(vol)?;
    check_combined_syntax(vol)?;
    if !matches!(vop.coding_type, VopCodingType::I) {
        return Err(VopDecodeError::Unsupported("not an I-VOP"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }

    let (mb_width, mb_height) = vop_mb_dimensions(vol);
    let bpp = u32::from(vol.bits_per_pixel);
    let max_qp = max_quantiser_scale(vol);
    let quant_matrix = intra_quant_matrix(vol);
    let vp_ctx = video_packet_context(vol, vop);
    let mut grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut running_qp = u32::from(vop.quant).clamp(1, max_qp);
    let mut dc_thr = vop.intra_dc_vlc_thr;
    let mut out = Vec::with_capacity(mb_width * mb_height);

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let mb_index = (mb_row * mb_width + mb_col) as u32;
            if mb_index > 0 && !vol.resync_marker_disable && video_packet_follows(br, &vp_ctx) {
                let packet = enter_video_packet(br, &vp_ctx, mb_index)?;
                // §E.1.2: a video-packet boundary is treated like a VOP
                // boundary — no §7.4.3 prediction crosses it, and the
                // running quantiser restarts from the packet's
                // quant_scale (§6.3.5 running-Qp rule).
                grid = IntraBlockGrid::new(mb_height, mb_width);
                running_qp = u32::from(packet.quant_scale).clamp(1, max_qp);
                if let Some(thr) = packet.intra_dc_vlc_thr {
                    dc_thr = thr;
                }
            }
            let header = parse_macroblock_header(br, VopCodingType::I, vol)?;
            running_qp = apply_dquant(running_qp, header.dquant_delta, max_qp);
            let use_dc_vlc = use_intra_dc_vlc(dc_thr, running_qp);
            let ctx = MacroblockTextureContext {
                quantiser_scale: running_qp,
                bits_per_pixel: bpp,
                quant_type: vol.quant_type,
                ac_pred_flag: header.ac_pred_flag,
                alternate_vertical_scan: vop.alternate_vertical_scan,
            };
            let coded = pattern_code(header.cbpy, header.cbpc);
            let mb = decode_intra_mb_with_grid(
                br,
                &mut grid,
                mb_row,
                mb_col,
                coded,
                use_dc_vlc,
                header_field_dct(&header),
                ctx,
                &quant_matrix,
            )?;
            out.push(reconstruct_intra_macroblock(&mb, bpp));
        }
    }
    Ok(out)
}

/// Decode the six §6.2.7 `block(i)` bodies of one **intra** macroblock
/// from a §6.2.5.3 data-partitioned *texture partition*, with per-block
/// §7.4.3 predictors from `grid`.
///
/// The differential intra DC (when the packet used the Table 6-25 DC
/// VLC) was already decoded from the header partition and arrives in
/// `supplied_dc` (Figure 6-8 block order); `None` means the DC is coded
/// as an AC coefficient inside the texture partition. `reversible_vlc`
/// selects the Table B.23 RVLC Tcoef table for the AC EVENT loops.
#[allow(clippy::too_many_arguments)]
fn decode_intra_mb_partitioned(
    br: &mut BitReader<'_>,
    grid: &mut IntraBlockGrid,
    mb_row: usize,
    mb_col: usize,
    coded: [bool; 6],
    supplied_dc: Option<[i32; 6]>,
    reversible_vlc: bool,
    ctx: MacroblockTextureContext,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<IntraMacroblock, VopDecodeError> {
    let mut blocks: [[[i32; 8]; 8]; 6] = [[[0i32; 8]; 8]; 6];
    for (i, block) in blocks.iter_mut().enumerate() {
        let predictors =
            grid.predictors_for(mb_row, mb_col, i, ctx.bits_per_pixel, ctx.quantiser_scale);
        let full = crate::block::decode_intra_block_partitioned(
            br,
            i,
            coded[i],
            supplied_dc.map(|dc| dc[i]),
            reversible_vlc,
            ctx,
            predictors,
            quant_matrix,
        )?;
        grid.record(
            mb_row,
            mb_col,
            i,
            Some(BlockNeighbour::from_qf(
                &full.qf,
                full.dc,
                ctx.quantiser_scale,
            )),
        );
        *block = full.spatial;
    }

    let mut luma = [[0i32; 16]; 16];
    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];
    for (b, &(row_off, col_off)) in LUMA_OFFSETS.iter().enumerate() {
        for y in 0..8 {
            for x in 0..8 {
                luma[row_off + y][col_off + x] = blocks[b][y][x];
            }
        }
    }
    Ok(IntraMacroblock {
        luma,
        cb: blocks[4],
        cr: blocks[5],
    })
}

/// Decode one **inter** macroblock's six residual blocks from a
/// data-partitioned texture partition and assemble the
/// [`InterMacroblock`](crate::block::InterMacroblock) residual.
fn decode_inter_mb_partitioned(
    br: &mut BitReader<'_>,
    coded: [bool; 6],
    reversible_vlc: bool,
    ctx: MacroblockTextureContext,
    quant_matrix: &[[u8; 8]; 8],
) -> Result<crate::block::InterMacroblock, VopDecodeError> {
    let mut blocks: [[[i32; 8]; 8]; 6] = [[[0i32; 8]; 8]; 6];
    for (i, block) in blocks.iter_mut().enumerate() {
        *block = crate::block::decode_inter_block_partitioned(
            br,
            i,
            coded[i],
            reversible_vlc,
            ctx,
            quant_matrix,
        )?;
    }
    let mut luma = [[0i32; 16]; 16];
    const LUMA_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 8), (8, 0), (8, 8)];
    for (b, &(row_off, col_off)) in LUMA_OFFSETS.iter().enumerate() {
        for y in 0..8 {
            for x in 0..8 {
                luma[row_off + y][col_off + x] = blocks[b][y][x];
            }
        }
    }
    Ok(crate::block::InterMacroblock {
        luma,
        cb: blocks[4],
        cr: blocks[5],
    })
}

/// The data-partitioned walks' shared gates (§6.2.5.3 rectangular
/// progressive layouts only).
fn check_dp_supported(vol: &VolHeader) -> Result<(), VopDecodeError> {
    check_vol_supported(vol)?;
    if !vol.data_partitioned {
        return Err(VopDecodeError::Unsupported("not a data_partitioned VOL"));
    }
    if vol.interlaced {
        // §6.2.5.3 data partitioning with the interlaced macroblock
        // tools is outside this walk's scope.
        return Err(VopDecodeError::Unsupported("data_partitioned + interlaced"));
    }
    Ok(())
}

/// Decode a complete rectangular progressive **I-VOP** coded with
/// §6.2.5.3 **data partitioning** (`data_partitioned == 1`), returning
/// the reconstructed macroblocks in raster order.
///
/// Each video packet is decoded as `data_partitioned_i_vop()`:
/// partition 1 (per-MB `mcbpc` + `dquant` + DC-VLC intra DC, up to the
/// 19-bit `dc_marker`), partition 2 (`ac_pred_flag` + `cbpy`), then the
/// texture partition holding every macroblock's AC `block()` data —
/// coded with the Table B.23 reversible VLC when the VOL set
/// `reversible_vlc == 1`. Prediction state (§7.4.3 grid, running
/// quantiser, `intra_dc_vlc_thr`) resets at each packet boundary
/// exactly as in the combined walk (§E.1.2).
pub fn decode_i_vop_macroblocks_dp(
    br: &mut BitReader<'_>,
    vol: &VolHeader,
    vop: &VopHeader,
) -> Result<Vec<ReconstructedMacroblock>, VopDecodeError> {
    check_dp_supported(vol)?;
    if !matches!(vop.coding_type, VopCodingType::I) {
        return Err(VopDecodeError::Unsupported("not an I-VOP"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }

    let (mb_width, mb_height) = vop_mb_dimensions(vol);
    let total = mb_width * mb_height;
    let bpp = u32::from(vol.bits_per_pixel);
    let max_qp = max_quantiser_scale(vol);
    let quant_matrix = intra_quant_matrix(vol);
    let vp_ctx = video_packet_context(vol, vop);
    let mut running_qp = u32::from(vop.quant).clamp(1, max_qp);
    let mut dc_thr = vop.intra_dc_vlc_thr;
    let mut out = Vec::with_capacity(total);
    let mut mb_index = 0usize;

    while mb_index < total {
        if mb_index > 0 {
            // Between packets: §5.2.5 stuffing + resync_marker + header.
            let packet = enter_video_packet(br, &vp_ctx, mb_index as u32)?;
            running_qp = u32::from(packet.quant_scale).clamp(1, max_qp);
            if let Some(thr) = packet.intra_dc_vlc_thr {
                dc_thr = thr;
            }
        }
        // §E.1.2: no prediction crosses a packet boundary.
        let mut grid = IntraBlockGrid::new(mb_height, mb_width);
        let parsed = crate::data_partition::parse_data_partitioned_i_vop(
            br,
            total - mb_index,
            dc_thr,
            running_qp,
        )
        .map_err(VopDecodeError::DataPartition)?;

        for (k, (mb, tex)) in parsed.mbs.iter().zip(parsed.tex_headers.iter()).enumerate() {
            let idx = mb_index + k;
            let (mb_row, mb_col) = (idx / mb_width, idx % mb_width);
            running_qp = apply_dquant(running_qp, mb.dquant_delta, max_qp);
            let ctx = MacroblockTextureContext {
                quantiser_scale: running_qp,
                bits_per_pixel: bpp,
                quant_type: vol.quant_type,
                ac_pred_flag: tex.ac_pred_flag,
                alternate_vertical_scan: vop.alternate_vertical_scan,
            };
            let coded = pattern_code(tex.cbpy, mb.cbpc);
            let imb = decode_intra_mb_partitioned(
                br,
                &mut grid,
                mb_row,
                mb_col,
                coded,
                mb.intra_dc,
                vol.reversible_vlc,
                ctx,
                &quant_matrix,
            )?;
            out.push(reconstruct_intra_macroblock(&imb, bpp));
        }
        mb_index += parsed.mbs.len();
    }
    Ok(out)
}

/// Decode a complete rectangular progressive **P-VOP** coded with
/// §6.2.5.3 **data partitioning**, returning one [`PVopMbContent`] per
/// macroblock in raster order (drop-in replacement for
/// [`decode_p_vop_macroblocks`] on a `data_partitioned` VOL).
///
/// Each video packet is decoded as `data_partitioned_p_vop()`:
/// partition 1 (per-MB `not_coded` + `mcbpc` + `motion_coding()`, up to
/// the 17-bit `motion_marker`) drives the §7.6.5 [`MvDriver`] in raster
/// order — a skipped MB records the valid zero vector and an intra MB
/// the valid zero candidate, exactly as in the combined walk — then
/// partition 2 (`ac_pred_flag` + `cbpy` + `dquant` + DC-VLC intra DC)
/// and the texture partition (RVLC when `reversible_vlc == 1`).
pub fn decode_p_vop_macroblocks_dp(
    br: &mut BitReader<'_>,
    vol: &VolHeader,
    vop: &VopHeader,
) -> Result<Vec<PVopMbContent>, VopDecodeError> {
    use crate::data_partition::DpMbEvent;
    use crate::macroblock::DerivedMbType;

    check_dp_supported(vol)?;
    if !matches!(vop.coding_type, VopCodingType::P) {
        return Err(VopDecodeError::Unsupported("not a P-VOP"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }
    if !vol.obmc_disable {
        return Err(VopDecodeError::Unsupported("data_partitioned + obmc"));
    }

    let (mb_width, mb_height) = vop_mb_dimensions(vol);
    let total = mb_width * mb_height;
    let bpp = u32::from(vol.bits_per_pixel);
    let max_qp = max_quantiser_scale(vol);
    let intra_matrix = intra_quant_matrix(vol);
    let inter_matrix = nonintra_quant_matrix(vol);
    let vp_ctx = video_packet_context(vol, vop);
    let mut running_qp = u32::from(vop.quant).clamp(1, max_qp);
    let mut dc_thr = vop.intra_dc_vlc_thr;
    let mut out: Vec<PVopMbContent> = Vec::with_capacity(total);
    let mut mb_index = 0usize;

    while mb_index < total {
        if mb_index > 0 {
            let packet = enter_video_packet(br, &vp_ctx, mb_index as u32)?;
            running_qp = u32::from(packet.quant_scale).clamp(1, max_qp);
            if let Some(thr) = packet.intra_dc_vlc_thr {
                dc_thr = thr;
            }
        }
        // §E.1.2 packet boundary: fresh §7.6.5 / §7.4.3 state.
        let mut driver = MvDriver::new(mb_height, mb_width, vop.fcode_fwd);
        let mut grid = IntraBlockGrid::new(mb_height, mb_width);
        let base = mb_index;
        let mut motions: Vec<PvopMbMotion> = Vec::new();

        let parsed = crate::data_partition::parse_data_partitioned_p_vop(
            br,
            total - base,
            false,
            dc_thr,
            running_qp,
            |b, ev| {
                let idx = base + motions.len();
                let (row, col) = (idx / mb_width, idx % mb_width);
                let m = match ev {
                    DpMbEvent::NotCoded => driver.decode_macroblock(b, row, col, true, None),
                    DpMbEvent::Intra => {
                        driver.decode_macroblock(b, row, col, false, Some(DerivedMbType::Intra))
                    }
                    DpMbEvent::Motion(ty) => driver.decode_macroblock(b, row, col, false, Some(ty)),
                    // GMC needs the S(GMC) walk; `is_s_gmc == false`
                    // keeps this arm unreachable on conforming input.
                    DpMbEvent::Gmc => {
                        return Err(crate::data_partition::DataPartitionError::Truncated)
                    }
                }
                .map_err(|_| crate::data_partition::DataPartitionError::Truncated)?;
                motions.push(m);
                Ok(())
            },
        )
        .map_err(VopDecodeError::DataPartition)?;

        for (k, (mb, tex)) in parsed.mbs.iter().zip(parsed.tex_headers.iter()).enumerate() {
            let idx = base + k;
            let (mb_row, mb_col) = (idx / mb_width, idx % mb_width);
            if mb.not_coded {
                out.push(PVopMbContent::Inter {
                    motion: motions[k],
                    residual: crate::block::InterMacroblock::zero(),
                });
                continue;
            }
            running_qp = apply_dquant(running_qp, tex.dquant_delta, max_qp);
            let coded = pattern_code(tex.cbpy, mb.cbpc);
            let is_intra = mb.mb_type.map(|t| t.is_intra()).unwrap_or(false);
            if is_intra {
                let ctx = MacroblockTextureContext {
                    quantiser_scale: running_qp,
                    bits_per_pixel: bpp,
                    quant_type: vol.quant_type,
                    ac_pred_flag: tex.ac_pred_flag,
                    alternate_vertical_scan: vop.alternate_vertical_scan,
                };
                let imb = decode_intra_mb_partitioned(
                    br,
                    &mut grid,
                    mb_row,
                    mb_col,
                    coded,
                    tex.intra_dc,
                    vol.reversible_vlc,
                    ctx,
                    &intra_matrix,
                )?;
                out.push(PVopMbContent::Intra(reconstruct_intra_macroblock(
                    &imb, bpp,
                )));
            } else {
                let ctx = MacroblockTextureContext {
                    quantiser_scale: running_qp,
                    bits_per_pixel: bpp,
                    quant_type: vol.quant_type,
                    ac_pred_flag: false,
                    alternate_vertical_scan: vop.alternate_vertical_scan,
                };
                let residual =
                    decode_inter_mb_partitioned(br, coded, vol.reversible_vlc, ctx, &inter_matrix)?;
                out.push(PVopMbContent::Inter {
                    motion: motions[k],
                    residual,
                });
            }
        }
        mb_index += parsed.mbs.len();
    }
    Ok(out)
}

/// Decode a complete rectangular progressive **P-VOP**'s macroblock
/// layer straight off the bitstream, returning one [`PVopMbContent`]
/// per macroblock in raster order — ready for
/// [`crate::frame_decode::assemble_p_vop_frame`] /
/// [`crate::sequence::SequenceDecoder::push_p_vop`].
///
/// `br` must be positioned at the first bit of
/// `combined_motion_shape_texture()`. Per macroblock the §6.2.6 order
/// is: header (`not_coded` / `mcbpc` / `ac_pred_flag` / `cbpy` /
/// `dquant`) → `motion_vector()` bodies → `block()` texture. The walk
/// threads the running quantiser, the Table 6-25 `use_intra_dc_vlc`
/// decision, the §7.4.3 intra predictor grid (inter macroblocks leave
/// their grid cells unset — §7.4.3.1 treats them as outside), and the
/// §7.6.5 / Figure 7-34 MV predictor grid inside the [`MvDriver`]
/// (intra macroblocks record as invalid candidates, skipped ones as
/// zero vectors).
///
/// Both the half-sample (§7.6.2.1) and quarter-sample (§7.6.2.2) VOLs
/// decode here — the §7.6.3 motion-vector syntax is unit-agnostic, so
/// the walk is identical; the caller selects the matching sub-pel
/// interpolation at frame assembly (and the §7.6.6 OBMC assembly when
/// the VOL coded `obmc_disable == 0` — the macroblock syntax itself is
/// unaffected by OBMC).
pub fn decode_p_vop_macroblocks(
    br: &mut BitReader<'_>,
    vol: &VolHeader,
    vop: &VopHeader,
) -> Result<Vec<PVopMbContent>, VopDecodeError> {
    check_vol_supported(vol)?;
    check_combined_syntax(vol)?;
    if vol.interlaced && !vol.obmc_disable {
        // §7.7.2.1 defines OBMC-with-field-neighbour semantics, but
        // the combination is outside the supported profiles; keep it
        // typed-rejected rather than half-wired.
        return Err(VopDecodeError::Unsupported("obmc + interlaced"));
    }
    if !matches!(vop.coding_type, VopCodingType::P) {
        return Err(VopDecodeError::Unsupported("not a P-VOP"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }
    if !vol.obmc_disable && vol.quarter_sample {
        // §7.6.6 is a half-sample-mode tool; a VOL asserting both is
        // outside any conforming profile.
        return Err(VopDecodeError::Unsupported("obmc + quarter_sample"));
    }

    let (mb_width, mb_height) = vop_mb_dimensions(vol);
    let bpp = u32::from(vol.bits_per_pixel);
    let max_qp = max_quantiser_scale(vol);
    let intra_matrix = intra_quant_matrix(vol);
    let inter_matrix = nonintra_quant_matrix(vol);
    let vp_ctx = video_packet_context(vol, vop);
    let mut intra_grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut driver = MvDriver::new(mb_height, mb_width, vop.fcode_fwd);
    let mut running_qp = u32::from(vop.quant).clamp(1, max_qp);
    let mut dc_thr = vop.intra_dc_vlc_thr;
    let mut out = Vec::with_capacity(mb_width * mb_height);

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let mb_index = (mb_row * mb_width + mb_col) as u32;
            if mb_index > 0 && !vol.resync_marker_disable && video_packet_follows(br, &vp_ctx) {
                let packet = enter_video_packet(br, &vp_ctx, mb_index)?;
                // §E.1.2: the packet boundary is a prediction boundary —
                // fresh §7.4.3 and §7.6.5 predictor state, running
                // quantiser restarted from the packet's quant_scale.
                intra_grid = IntraBlockGrid::new(mb_height, mb_width);
                driver = MvDriver::new(mb_height, mb_width, vop.fcode_fwd);
                running_qp = u32::from(packet.quant_scale).clamp(1, max_qp);
                if let Some(thr) = packet.intra_dc_vlc_thr {
                    dc_thr = thr;
                }
            }
            let header = parse_macroblock_header(br, VopCodingType::P, vol)?;

            if header.not_coded {
                // §6.3.6: a skipped P-VOP macroblock is inter with the
                // zero vector and no residual. Record it in the MV grid
                // (a *valid* zero-MV neighbour) and emit the copy MB.
                let motion = driver.decode_macroblock(br, mb_row, mb_col, true, None)?;
                out.push(PVopMbContent::Inter {
                    motion,
                    residual: crate::block::InterMacroblock::zero(),
                });
                continue;
            }

            running_qp = apply_dquant(running_qp, header.dquant_delta, max_qp);
            let ctx = MacroblockTextureContext {
                quantiser_scale: running_qp,
                bits_per_pixel: bpp,
                quant_type: vol.quant_type,
                ac_pred_flag: header.ac_pred_flag,
                alternate_vertical_scan: vop.alternate_vertical_scan,
            };

            let is_intra = header.mb_type.map(|t| t.is_intra()).unwrap_or(false);
            if is_intra {
                // Record the intra MB as an invalid MV-predictor
                // candidate (§7.6.5); no motion bits are coded.
                driver.decode_macroblock(br, mb_row, mb_col, false, header.mb_type)?;
                let use_dc_vlc = use_intra_dc_vlc(dc_thr, running_qp);
                let coded = pattern_code(header.cbpy, header.cbpc);
                let mb = decode_intra_mb_with_grid(
                    br,
                    &mut intra_grid,
                    mb_row,
                    mb_col,
                    coded,
                    use_dc_vlc,
                    header_field_dct(&header),
                    ctx,
                    &intra_matrix,
                )?;
                out.push(PVopMbContent::Intra(reconstruct_intra_macroblock(&mb, bpp)));
            } else if let Some((top_ref, bottom_ref)) = header_forward_field_refs(&header) {
                // §7.7.2.1 field-predicted macroblock: two field MV
                // bodies with the shared CASE 1/2/3 predictor.
                let mvs = driver.decode_field_macroblock(br, mb_row, mb_col)?;
                let mut residual = decode_inter_macroblock(br, &header, ctx, &inter_matrix)?;
                if header_field_dct(&header) {
                    residual.luma = inverse_field_dct_luma(&residual.luma);
                }
                out.push(PVopMbContent::FieldInter {
                    mvs,
                    top_field_ref: top_ref,
                    bottom_field_ref: bottom_ref,
                    residual,
                });
            } else {
                // §6.2.6: the motion_vector() bodies precede the texture.
                let motion = driver.decode_macroblock(br, mb_row, mb_col, false, header.mb_type)?;
                let mut residual = decode_inter_macroblock(br, &header, ctx, &inter_matrix)?;
                // §7.7.1: an interlaced frame-predicted inter MB may
                // still code its residual with the field DCT.
                if header_field_dct(&header) {
                    residual.luma = inverse_field_dct_luma(&residual.luma);
                }
                out.push(PVopMbContent::Inter { motion, residual });
            }
        }
    }
    Ok(out)
}

/// Resolve the §7.6.9.5.1 / §7.6.9.6 co-located anchor state for one
/// B-VOP macroblock from the *future* (most recently decoded) anchor's
/// per-macroblock motion.
///
/// * a skipped anchor MB → the §7.6.9.6 override flag,
/// * a 1-MV inter MB → its vector replicated into all four block
///   slots (§7.6.1.6 vector padding of a 1-MV macroblock),
/// * a 4-MV inter MB → its four sub-block vectors `{MV[i]}` in
///   Figure 6-8 order (§7.6.9.5.2 gives each 8×8 block its own
///   scaled `(MVF[i], MVB[i])` pair),
/// * an intra MB (or a GMC MB, which carries no local vector) → the
///   §7.6.9.5.1 final-sentence zero-vector fallback.
fn co_located_from_motion(motion: PvopMbMotion) -> CoLocatedAnchor {
    match motion {
        PvopMbMotion::Skipped => {
            CoLocatedAnchor::uniform(true, DirectCoLocatedMv::TransparentOrAbsent)
        }
        PvopMbMotion::OneMv(mv) => CoLocatedAnchor::uniform(false, DirectCoLocatedMv::Mv(mv)),
        PvopMbMotion::FourMv(mvs) => CoLocatedAnchor {
            skipped: false,
            mvs: mvs.map(DirectCoLocatedMv::Mv),
        },
        PvopMbMotion::Intra => CoLocatedAnchor::default(),
    }
}

/// Decode a complete rectangular progressive **B-VOP**'s macroblock
/// layer straight off the bitstream, returning one
/// [`BVopMbTexturedDecode`] per macroblock in raster order — ready for
/// [`crate::frame_decode::assemble_b_vop_frame`] /
/// [`crate::sequence::SequenceDecoder::push_b_vop`].
///
/// This is the header-level wrapper over
/// [`BVopMvDriver::decode_vop`](crate::bvop_mv::BVopMvDriver::decode_vop):
/// it derives the driver's dimensions / f_codes / texture parameters
/// from the VOL + VOP headers and adapts the future anchor's decoded
/// per-macroblock motion (`anchor_motion`, raster order — e.g. the
/// [`PvopMbMotion`]s the P-VOP walk produced, or `None` after an
/// intra-only anchor) into the per-MB §7.6.9.5.1 / §7.6.9.6 co-located
/// state.
///
/// `trb` / `trd` are the §7.6.7 temporal references in
/// `vop_time_increment_resolution` ticks (`trb`: this B-VOP minus the
/// past anchor; `trd`: future anchor minus past anchor).
pub fn decode_b_vop_macroblocks(
    br: &mut BitReader<'_>,
    vol: &VolHeader,
    vop: &VopHeader,
    trb: i32,
    trd: i32,
    anchor_motion: Option<&[PvopMbMotion]>,
) -> Result<Vec<BVopMbTexturedDecode>, VopDecodeError> {
    check_vol_supported(vol)?;
    check_progressive(vol)?;
    if !matches!(vop.coding_type, VopCodingType::B) {
        return Err(VopDecodeError::Unsupported("not a B-VOP"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }

    let (mb_width, mb_height) = vop_mb_dimensions(vol);
    let max_qp = max_quantiser_scale(vol);
    let mut driver = BVopMvDriver::new(mb_height, mb_width, vop.fcode_fwd, vop.fcode_bwd, trb, trd)
        .with_quarter_sample(vol.quarter_sample);
    let texture = BVopTextureParams {
        base_quantiser_scale: u32::from(vop.quant).clamp(1, max_qp),
        alternate_vertical_scan: vop.alternate_vertical_scan,
        max_quantiser_scale: max_qp,
        bits_per_pixel: u32::from(vol.bits_per_pixel),
        quant_type: vol.quant_type,
    };
    // Resync-aware raster loop — the video-packet analogue of
    // `BVopMvDriver::decode_vop`: per §7.6.8 a resync marker resets the
    // running MV predictors exactly like a row start, and the running
    // quantiser restarts from the packet's `quant_scale`.
    let vp_ctx = video_packet_context(vol, vop);
    let quant_matrix = nonintra_quant_matrix(vol);
    let mut quantiser_scale = texture.base_quantiser_scale;
    let mut out = Vec::with_capacity(mb_width * mb_height);
    for mb_row in 0..mb_height {
        driver.start_row();
        for mb_col in 0..mb_width {
            let mb_index = (mb_row * mb_width + mb_col) as u32;
            if mb_index > 0 && !vol.resync_marker_disable && video_packet_follows(br, &vp_ctx) {
                // A B-VOP packet may resume at a macroblock *ahead* of
                // the raster index: the intervening `co_located_not_coded`
                // macroblocks transmit no bits, so a packet cut placed
                // before one of them appears "early" in the bit walk.
                // Parse on a clone, accept the resume point if every
                // in-between macroblock is a non-transmitting one, then
                // advance the real reader past the header.
                let mut probe = br.clone();
                let packet = parse_video_packet_header(&mut probe, &vp_ctx)?;
                let resume = packet.macroblock_number;
                let gap_all_skipped = (mb_index..resume).all(|i| {
                    anchor_motion
                        .map(|m| matches!(m[i as usize], PvopMbMotion::Skipped))
                        .unwrap_or(false)
                });
                if resume < mb_index || !gap_all_skipped {
                    return Err(VopDecodeError::VideoPacketSkip {
                        packet_mb: resume,
                        expected_mb: mb_index,
                    });
                }
                let consumed = probe.bit_position() - br.bit_position();
                br.skip_bits(consumed)
                    .map_err(|_| VopDecodeError::VideoPacket(VideoPacketParseError::Truncated))?;
                driver.start_row();
                quantiser_scale =
                    u32::from(packet.quant_scale).clamp(1, texture.max_quantiser_scale);
            }
            let co_motion = anchor_motion.map(|m| m[mb_row * mb_width + mb_col]);

            // §6.2.6 / §6.3.6 `co_located_not_coded`: when the future
            // reference is a P-VOP and its co-located macroblock was
            // skipped (`not_coded == 1`), the B macroblock transmits
            // **no bits at all** — the whole `modb` subtree is gated
            // out. §7.6.9.6: it reconstructs as forward mode with the
            // zero motion vector (a copy of the past anchor).
            if matches!(co_motion, Some(PvopMbMotion::Skipped)) {
                out.push(forward_zero_b_mb(quantiser_scale));
                continue;
            }

            let anchor = co_motion.map(co_located_from_motion).unwrap_or_default();
            let motion = driver
                .decode_macroblock(
                    br,
                    vol,
                    vop.coding_type,
                    mb_row,
                    mb_col,
                    anchor.skipped,
                    anchor.mvs,
                )
                .map_err(VopDecodeError::BVop)?;

            // §6.3.6: apply this macroblock's dbquant to the running
            // quantiser scale, clipped to [1, max_quantiser_scale].
            if let Some(delta) = motion.dbquant_delta {
                let updated = i64::from(quantiser_scale) + i64::from(delta);
                quantiser_scale = updated.clamp(1, i64::from(texture.max_quantiser_scale)) as u32;
            }

            let ctx = MacroblockTextureContext {
                quantiser_scale,
                bits_per_pixel: texture.bits_per_pixel,
                quant_type: texture.quant_type,
                ac_pred_flag: false,
                alternate_vertical_scan: vop.alternate_vertical_scan,
            };
            let residual = decode_b_vop_inter_macroblock(br, motion.cbpb, ctx, &quant_matrix)
                .map_err(|e| VopDecodeError::BVop(BVopMvDriverError::Texture(e)))?;

            out.push(BVopMbTexturedDecode {
                motion,
                residual,
                quantiser_scale,
            });
        }
    }
    Ok(out)
}

/// One anchor macroblock's motion as recorded for the following
/// B-VOPs' co-located consultation (§7.6.9.5.1 / §7.7.2.2).
///
/// The interlaced direct mode needs the co-located future macroblock's
/// two forward **field** motion vectors and reference-field selections
/// — information the progressive [`PvopMbMotion`] cannot carry — so
/// the anchor record keeps the field shape intact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorMbMotion {
    /// A frame-predicted (or intra / skipped) anchor macroblock.
    Frame(PvopMbMotion),
    /// A §7.7.2.1 field-predicted anchor macroblock: its reconstructed
    /// field MV pair plus the §6.3.7.2 reference-field bits.
    Field {
        /// Reconstructed top/bottom field motion vectors.
        mvs: crate::field_motion::FieldMotionVectors,
        /// `forward_top_field_reference` raw bit.
        top_ref: bool,
        /// `forward_bottom_field_reference` raw bit.
        bottom_ref: bool,
    },
}

impl AnchorMbMotion {
    /// Collapse to the progressive co-located representation: a field
    /// pair averages per §7.7.2.1 `Div2Round(MVf1 + MVf2)`.
    pub fn progressive(self) -> PvopMbMotion {
        match self {
            AnchorMbMotion::Frame(m) => m,
            AnchorMbMotion::Field { mvs, .. } => PvopMbMotion::OneMv(crate::motion::MotionVector {
                x: crate::field_motion::div2_round(mvs.top.x + mvs.bottom.x),
                y: crate::field_motion::div2_round(mvs.top.y + mvs.bottom.y),
            }),
        }
    }
}

/// Resolve the per-macroblock [`BVopInterlacedAnchor`] for the
/// interlaced B walk.
fn interlaced_anchor_of(
    co: Option<AnchorMbMotion>,
    top_field_first: bool,
) -> crate::bvop_mv::BVopInterlacedAnchor {
    use crate::bvop_mv::{BVopInterlacedAnchor, ColocatedFutureFieldMvs};
    use crate::interlaced_information::FieldReference;
    match co {
        None => BVopInterlacedAnchor {
            progressive: CoLocatedAnchor::default(),
            future_field_mvs: None,
            top_field_first,
        },
        Some(AnchorMbMotion::Frame(m)) => BVopInterlacedAnchor {
            progressive: co_located_from_motion(m),
            future_field_mvs: None,
            top_field_first,
        },
        Some(
            field @ AnchorMbMotion::Field {
                mvs,
                top_ref,
                bottom_ref,
            },
        ) => BVopInterlacedAnchor {
            progressive: co_located_from_motion(field.progressive()),
            future_field_mvs: Some(ColocatedFutureFieldMvs::from_field_motion(
                mvs,
                FieldReference::from_bit(top_ref),
                FieldReference::from_bit(bottom_ref),
            )),
            top_field_first,
        },
    }
}

/// Decode a complete rectangular **interlaced B-VOP**'s macroblock
/// layer straight off the bitstream, returning one
/// [`BVopInterlacedTexturedDecode`](crate::bvop_mv::BVopInterlacedTexturedDecode)
/// per macroblock in raster order — ready for
/// [`crate::frame_decode::assemble_b_vop_interlaced_frame`].
///
/// The per-macroblock dispatch is
/// [`BVopMvDriver::decode_interlaced_macroblock`]: progressive
/// (frame-predicted / progressive-direct), §7.7.2.2 field-predicted
/// forward / backward / bidirectional (through the Table 7-14/7-15
/// four-PMV bank), or §7.7.2.2 interlaced direct (when the co-located
/// future macroblock was field-predicted — `anchor_motion` supplies
/// the field pairs via [`AnchorMbMotion::Field`]). Residual luminance
/// coded with `dct_type == 1` is inverse-field-DCT permuted (§7.7.1).
///
/// The §6.2.6 `co_located_not_coded` zero-bit rule applies exactly as
/// in the progressive walk.
pub fn decode_b_vop_interlaced_macroblocks(
    br: &mut BitReader<'_>,
    vol: &VolHeader,
    vop: &VopHeader,
    trb: i32,
    trd: i32,
    anchor_motion: Option<&[AnchorMbMotion]>,
) -> Result<Vec<crate::bvop_mv::BVopInterlacedTexturedDecode>, VopDecodeError> {
    use crate::bvop_mv::BVopInterlacedTexturedDecode;

    check_vol_supported(vol)?;
    if !vol.interlaced {
        return Err(VopDecodeError::Unsupported("not an interlaced VOL"));
    }
    if !matches!(vop.coding_type, VopCodingType::B) {
        return Err(VopDecodeError::Unsupported("not a B-VOP"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }

    let (mb_width, mb_height) = vop_mb_dimensions(vol);
    let max_qp = max_quantiser_scale(vol);
    let mut driver = BVopMvDriver::new(mb_height, mb_width, vop.fcode_fwd, vop.fcode_bwd, trb, trd)
        .with_quarter_sample(vol.quarter_sample);
    let quant_matrix = nonintra_quant_matrix(vol);
    let vp_ctx = video_packet_context(vol, vop);
    let mut quantiser_scale = u32::from(vop.quant).clamp(1, max_qp);
    let mut out = Vec::with_capacity(mb_width * mb_height);
    for mb_row in 0..mb_height {
        driver.start_row();
        for mb_col in 0..mb_width {
            let mb_index = mb_row * mb_width + mb_col;
            // §6.2.5.2 video-packet resync — same rules as the
            // progressive B walk: a packet may resume ahead of the
            // raster index across a run of zero-bit
            // co_located_not_coded macroblocks.
            if mb_index > 0 && !vol.resync_marker_disable && video_packet_follows(br, &vp_ctx) {
                let mut probe = br.clone();
                let packet = parse_video_packet_header(&mut probe, &vp_ctx)?;
                let resume = packet.macroblock_number as usize;
                let gap_all_skipped = (mb_index..resume).all(|i| {
                    anchor_motion
                        .map(|m| matches!(m[i], AnchorMbMotion::Frame(PvopMbMotion::Skipped)))
                        .unwrap_or(false)
                });
                if resume < mb_index || !gap_all_skipped {
                    return Err(VopDecodeError::VideoPacketSkip {
                        packet_mb: resume as u32,
                        expected_mb: mb_index as u32,
                    });
                }
                let consumed = probe.bit_position() - br.bit_position();
                br.skip_bits(consumed)
                    .map_err(|_| VopDecodeError::VideoPacket(VideoPacketParseError::Truncated))?;
                driver.start_row();
                quantiser_scale = u32::from(packet.quant_scale).clamp(1, max_qp);
            }
            let co = anchor_motion.map(|m| m[mb_index]);

            // §6.2.6 / §7.6.9.6 co_located_not_coded: no bits at all.
            if matches!(co, Some(AnchorMbMotion::Frame(PvopMbMotion::Skipped))) {
                let fz = forward_zero_b_mb(quantiser_scale);
                out.push(BVopInterlacedTexturedDecode {
                    motion: crate::bvop_mv::BVopInterlacedMb::Progressive(fz.motion),
                    residual: fz.residual,
                    quantiser_scale,
                });
                continue;
            }

            let anchor = interlaced_anchor_of(co, vop.top_field_first);
            let (motion, field_dct) = driver
                .decode_interlaced_macroblock(br, vol, vop.coding_type, mb_row, mb_col, anchor)
                .map_err(VopDecodeError::BVop)?;

            if let Some(delta) = motion.dbquant_delta() {
                let updated = i64::from(quantiser_scale) + i64::from(delta);
                quantiser_scale = updated.clamp(1, i64::from(max_qp)) as u32;
            }

            let ctx = MacroblockTextureContext {
                quantiser_scale,
                bits_per_pixel: u32::from(vol.bits_per_pixel),
                quant_type: vol.quant_type,
                ac_pred_flag: false,
                alternate_vertical_scan: vop.alternate_vertical_scan,
            };
            let mut residual = decode_b_vop_inter_macroblock(br, motion.cbpb(), ctx, &quant_matrix)
                .map_err(|e| VopDecodeError::BVop(BVopMvDriverError::Texture(e)))?;
            if field_dct {
                residual.luma = inverse_field_dct_luma(&residual.luma);
            }

            out.push(BVopInterlacedTexturedDecode {
                motion,
                residual,
                quantiser_scale,
            });
        }
    }
    Ok(out)
}

/// The §7.6.9.6 reconstruction of a syntax-skipped B macroblock
/// (`co_located_not_coded == 1`): forward mode, zero motion vector,
/// zero residual — a pure copy of the past (forward) anchor.
fn forward_zero_b_mb(quantiser_scale: u32) -> BVopMbTexturedDecode {
    use crate::bvop::BVopMbType;
    use crate::bvop_mv::BVopMbDecode;
    use crate::bvop_prediction::{BVopMvPair, BVopPredictionMode};
    let zero = MotionVector { x: 0, y: 0 };
    BVopMbTexturedDecode {
        motion: BVopMbDecode {
            mb_type: BVopMbType::Forward,
            prediction_mode: BVopPredictionMode::ForwardOnly,
            cbpb: None,
            dbquant_delta: None,
            mvs: [BVopMvPair {
                forward: zero,
                backward: zero,
            }; 4],
            forward_chroma_mv: zero,
            backward_chroma_mv: zero,
        },
        residual: crate::block::InterMacroblock::zero(),
        quantiser_scale,
    }
}

/// Compute the §7.8.7.3 averaged motion vector of one `mcsel == 1`
/// macroblock from the warp: the 256 luminance pel-wise warping vectors
/// `(F(i,j) − s·i, G(i,j) − s·j)` (each in `1/s`-pel units), summed and
/// quantised to the half-pel (or quarter-pel) grid with the Table 7-9
/// clip for `vop_fcode`.
fn gmc_averaged_mv(
    geometry: &WarpGeometry,
    mb_x: i64,
    mb_y: i64,
    quarter_sample: bool,
    vop_fcode: u8,
) -> Result<MotionVector, VopDecodeError> {
    let mut mvs_x = [0i64; AMV_PIXEL_COUNT];
    let mut mvs_y = [0i64; AMV_PIXEL_COUNT];
    for j in 0..16i64 {
        for i in 0..16i64 {
            let px = mb_x + i;
            let py = mb_y + j;
            let [f, g] = geometry.luma_fg(px, py);
            let idx = (j * 16 + i) as usize;
            mvs_x[idx] = f - geometry.s * px;
            mvs_y[idx] = g - geometry.s * py;
        }
    }
    averaged_motion_vector(&mvs_x, &mvs_y, geometry.s as u32, quarter_sample, vop_fcode)
        .map_err(|e| VopDecodeError::Motion(PvopMvError::Motion(e)))
}

/// Decode a complete rectangular progressive **S(GMC)-VOP**'s
/// macroblock layer straight off the bitstream, returning one
/// [`SGmcMbContent`] per macroblock in raster order plus the decoded
/// §7.8.4/§7.8.5 [`WarpGeometry`] — ready for
/// [`crate::frame_decode::assemble_s_gmc_vop_frame`] /
/// [`crate::sequence::SequenceDecoder::push_s_gmc_vop`].
///
/// `br` must be positioned at the first bit of
/// `combined_motion_shape_texture()` (the VOP header — including the
/// §6.2.5 `sprite_trajectory()` already captured on
/// [`VopHeader::sprite_trajectory`] — has been consumed).
///
/// Per §6.3.6 the S(GMC) macroblock layer is the P-VOP layer plus the
/// `mcsel` flag on inter / inter+q macroblocks:
///
/// * `not_coded == 1` → implied `mcsel == 1`: a GMC-predicted copy
///   (zero residual). The §7.8.7.3 averaged warping vector is recorded
///   as the MV-predictor candidate.
/// * coded inter with `mcsel == 1` → GMC prediction, no local MV
///   bodies, §7.4 residual follows; AMV recorded.
/// * coded inter with `mcsel == 0` (and every inter4v MB — `mcsel` is
///   only coded for 1-MV inter types) → the plain P-VOP local-MC path.
/// * intra / intra+q → the grid-threaded §7.4 intra path.
pub fn decode_s_gmc_vop_macroblocks(
    br: &mut BitReader<'_>,
    vol: &VolHeader,
    vop: &VopHeader,
) -> Result<(Vec<SGmcMbContent>, WarpGeometry), VopDecodeError> {
    check_vol_supported(vol)?;
    check_combined_syntax(vol)?;
    check_progressive(vol)?;
    if !matches!(vop.coding_type, VopCodingType::S) {
        return Err(VopDecodeError::Unsupported("not an S-VOP"));
    }
    if !matches!(vol.sprite_enable, SpriteEnable::Gmc) {
        return Err(VopDecodeError::Unsupported("sprite_enable != GMC"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }
    if !vol.obmc_disable {
        return Err(VopDecodeError::Unsupported("obmc"));
    }

    let stationary = SpriteTrajectory::stationary();
    let trajectory = vop.sprite_trajectory.as_ref().unwrap_or(&stationary);
    let accuracy = vol
        .sprite_warping_accuracy
        .unwrap_or(SpriteWarpingAccuracy::HalfPel);
    let geometry = WarpGeometry::decode(
        trajectory,
        u32::from(vol.width),
        u32::from(vol.height),
        accuracy,
    );

    let (mb_width, mb_height) = vop_mb_dimensions(vol);
    let bpp = u32::from(vol.bits_per_pixel);
    let max_qp = max_quantiser_scale(vol);
    let intra_matrix = intra_quant_matrix(vol);
    let inter_matrix = nonintra_quant_matrix(vol);
    let vp_ctx = video_packet_context(vol, vop);
    let mut intra_grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut driver = MvDriver::new(mb_height, mb_width, vop.fcode_fwd);
    let mut running_qp = u32::from(vop.quant).clamp(1, max_qp);
    let mut dc_thr = vop.intra_dc_vlc_thr;
    let mut out = Vec::with_capacity(mb_width * mb_height);

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let mb_index = (mb_row * mb_width + mb_col) as u32;
            if mb_index > 0 && !vol.resync_marker_disable && video_packet_follows(br, &vp_ctx) {
                let packet = enter_video_packet(br, &vp_ctx, mb_index)?;
                // §E.1.2: fresh predictor state at the packet boundary.
                intra_grid = IntraBlockGrid::new(mb_height, mb_width);
                driver = MvDriver::new(mb_height, mb_width, vop.fcode_fwd);
                running_qp = u32::from(packet.quant_scale).clamp(1, max_qp);
                if let Some(thr) = packet.intra_dc_vlc_thr {
                    dc_thr = thr;
                }
            }
            let header = parse_macroblock_header(br, VopCodingType::S, vol)?;
            let mb_x = (mb_col * 16) as i64;
            let mb_y = (mb_row * 16) as i64;

            if header.not_coded {
                // §6.3.6: a not-coded S(GMC) macroblock is GMC-predicted
                // (implied mcsel == 1) with a zero residual.
                let amv =
                    gmc_averaged_mv(&geometry, mb_x, mb_y, vol.quarter_sample, vop.fcode_fwd)?;
                driver.record_gmc_macroblock(mb_row, mb_col, amv)?;
                out.push(SGmcMbContent::Gmc {
                    residual: crate::block::InterMacroblock::zero(),
                });
                continue;
            }

            running_qp = apply_dquant(running_qp, header.dquant_delta, max_qp);
            let ctx = MacroblockTextureContext {
                quantiser_scale: running_qp,
                bits_per_pixel: bpp,
                quant_type: vol.quant_type,
                ac_pred_flag: header.ac_pred_flag,
                alternate_vertical_scan: vop.alternate_vertical_scan,
            };

            let is_intra = header.mb_type.map(|t| t.is_intra()).unwrap_or(false);
            if is_intra {
                driver.decode_macroblock(br, mb_row, mb_col, false, header.mb_type)?;
                let use_dc_vlc = use_intra_dc_vlc(dc_thr, running_qp);
                let coded = pattern_code(header.cbpy, header.cbpc);
                let mb = decode_intra_mb_with_grid(
                    br,
                    &mut intra_grid,
                    mb_row,
                    mb_col,
                    coded,
                    use_dc_vlc,
                    header_field_dct(&header),
                    ctx,
                    &intra_matrix,
                )?;
                out.push(SGmcMbContent::Intra(reconstruct_intra_macroblock(&mb, bpp)));
            } else if header.mcsel == Some(true) {
                // §6.3.6: mcsel == 1 codes no local motion vectors; the
                // texture follows the header directly.
                let amv =
                    gmc_averaged_mv(&geometry, mb_x, mb_y, vol.quarter_sample, vop.fcode_fwd)?;
                driver.record_gmc_macroblock(mb_row, mb_col, amv)?;
                let residual = decode_inter_macroblock(br, &header, ctx, &inter_matrix)?;
                out.push(SGmcMbContent::Gmc { residual });
            } else {
                // mcsel == 0 (or inter4v): the plain P-VOP local path.
                let motion = driver.decode_macroblock(br, mb_row, mb_col, false, header.mb_type)?;
                let residual = decode_inter_macroblock(br, &header, ctx, &inter_matrix)?;
                out.push(SGmcMbContent::Local { motion, residual });
            }
        }
    }
    Ok((out, geometry))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_decode::{assemble_p_vop_frame, decode_i_vop};
    use crate::framestore::FrameStore;
    use crate::vop::VopContext;

    /// MSB-first bit writer for synthesising macroblock-layer streams.
    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        bit: u8,
        acc: u8,
    }

    impl BitWriter {
        fn write_bits(&mut self, value: u32, n: usize) {
            for i in (0..n).rev() {
                let b = ((value >> i) & 1) as u8;
                self.acc = (self.acc << 1) | b;
                self.bit += 1;
                if self.bit == 8 {
                    self.bytes.push(self.acc);
                    self.acc = 0;
                    self.bit = 0;
                }
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.bit > 0 {
                self.acc <<= 8 - self.bit;
                self.bytes.push(self.acc);
            }
            self.bytes
        }
    }

    /// A minimal rectangular Simple-Profile-shaped VOL header for a
    /// `width × height` progressive VOL (method-2 quant, no data
    /// partitioning, OBMC off).
    fn test_vol(width: u16, height: u16) -> VolHeader {
        // Build via the real parser to stay honest to the struct: a
        // hand-rolled literal would silently miss future fields. The
        // fixture mirrors vol.rs's own test bitstreams.
        let mut w = BitWriter::default();
        w.write_bits(0x0000_0120, 32); // video_object_layer_start_code
        w.write_bits(0, 1); // random_accessible_vol
        w.write_bits(1, 8); // video_object_type_indication (simple)
        w.write_bits(0, 1); // is_object_layer_identifier
        w.write_bits(1, 4); // aspect_ratio_info: 1:1
        w.write_bits(0, 1); // vol_control_parameters
        w.write_bits(0, 2); // video_object_layer_shape: rectangular
        w.write_bits(1, 1); // marker
        w.write_bits(30, 16); // vop_time_increment_resolution
        w.write_bits(1, 1); // marker
        w.write_bits(0, 1); // fixed_vop_rate
        w.write_bits(1, 1); // marker
        w.write_bits(u32::from(width), 13);
        w.write_bits(1, 1); // marker
        w.write_bits(u32::from(height), 13);
        w.write_bits(1, 1); // marker
        w.write_bits(0, 1); // interlaced
        w.write_bits(1, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable (verid 1: 1 bit... see below)
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type (method 2)
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(1, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // scalability
        let data = w.finish();
        crate::vol::parse_video_object_layer(&data, 0).expect("test VOL must parse")
    }

    /// A coded I-VOP header (quant = `quant`, intra_dc_vlc_thr 0).
    fn test_i_vop(vol: &VolHeader, quant: u32) -> VopHeader {
        let mut w = BitWriter::default();
        w.write_bits(0x0000_01B6, 32); // vop_start_code
        w.write_bits(0b00, 2); // vop_coding_type: I
        w.write_bits(0, 1); // modulo_time_base terminator
        w.write_bits(1, 1); // marker
        w.write_bits(0, 5); // vop_time_increment (5 bits for res 30)
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(quant, 5); // vop_quant
        let data = w.finish();
        crate::vop::parse_video_object_plane_header(
            &data,
            vol.time_increment_resolution,
            VopContext::from_vol(vol),
        )
        .expect("test VOP must parse")
    }

    /// A coded P-VOP header (quant, fcode_fwd 1, rounding 0).
    fn test_p_vop(vol: &VolHeader, quant: u32) -> VopHeader {
        let mut w = BitWriter::default();
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(0b01, 2); // P
        w.write_bits(0, 1); // modulo_time_base
        w.write_bits(1, 1); // marker
        w.write_bits(0, 5); // time increment
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 1); // vop_rounding_type
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(quant, 5); // vop_quant
        w.write_bits(1, 3); // vop_fcode_forward
        let data = w.finish();
        crate::vop::parse_video_object_plane_header(
            &data,
            vol.time_increment_resolution,
            VopContext::from_vol(vol),
        )
        .expect("test VOP must parse")
    }

    /// Write one DC-only intra macroblock (I-VOP): mcbpc "1" (intra,
    /// cbpc 00), ac_pred 0, cbpy "0011" (intra pattern 0), six size-0
    /// DCs (Table B.13 luma "011" ×4, Table B.14 chroma "11" ×2).
    fn write_dc_only_intra_mb(w: &mut BitWriter) {
        w.write_bits(0b1, 1); // mcbpc: intra, cbpc 00
        w.write_bits(0, 1); // ac_pred_flag
        w.write_bits(0b0011, 4); // cbpy: intra 0 (Table B.8)
        for _ in 0..4 {
            w.write_bits(0b011, 3); // luma DC size 0
        }
        for _ in 0..2 {
            w.write_bits(0b11, 2); // chroma DC size 0
        }
    }

    /// Write one intra macroblock whose *first* luma block carries a +1
    /// DC differential (all other blocks size-0).
    fn write_intra_mb_first_block_dc_plus1(w: &mut BitWriter) {
        w.write_bits(0b1, 1); // mcbpc: intra, cbpc 00
        w.write_bits(0, 1); // ac_pred_flag
        w.write_bits(0b0011, 4); // cbpy 0
        w.write_bits(0b11, 2); // luma DC size 1 (Table B.13 "11")
        w.write_bits(1, 1); // differential +1 (Table B.15)
        for _ in 0..3 {
            w.write_bits(0b011, 3);
        }
        for _ in 0..2 {
            w.write_bits(0b11, 2);
        }
    }

    #[test]
    fn mb_dimensions_round_up() {
        let vol = test_vol(48, 32);
        assert_eq!(vop_mb_dimensions(&vol), (3, 2));
        let vol = test_vol(40, 24);
        assert_eq!(vop_mb_dimensions(&vol), (3, 2));
    }

    #[test]
    fn i_vop_dc_only_frame_is_flat_grey() {
        // 32×16: two macroblocks, both DC-only with zero differentials.
        // The §7.4.3 predicted DC (1024) reconstructs every sample to
        // 128 — mid grey (bpp 8, qs 8 → dc_scaler 16).
        let vol = test_vol(32, 16);
        let vop = test_i_vop(&vol, 8);
        let mut w = BitWriter::default();
        write_dc_only_intra_mb(&mut w);
        write_dc_only_intra_mb(&mut w);
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let mbs = decode_i_vop_macroblocks(&mut br, &vol, &vop).unwrap();
        assert_eq!(mbs.len(), 2);
        for mb in &mbs {
            for row in mb.luma.iter() {
                for &px in row.iter() {
                    assert_eq!(px, 128);
                }
            }
            assert_eq!(mb.cb[0][0], 128);
            assert_eq!(mb.cr[0][0], 128);
        }
    }

    #[test]
    fn i_vop_dc_prediction_crosses_macroblocks() {
        // MB0's first luma block carries a +1 differential (→ DC 1040,
        // flat 130); MB1 is all-zero-differential. MB1's block 0
        // predicts from its left neighbour — MB0's block 1 (DC 1024,
        // since only MB0's block 0 got the +1) — NOT the out-of-VOP
        // default. Verify via the exact reconstruction values.
        let vol = test_vol(32, 16);
        let vop = test_i_vop(&vol, 8);
        let mut w = BitWriter::default();
        write_intra_mb_first_block_dc_plus1(&mut w);
        write_dc_only_intra_mb(&mut w);
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let mbs = decode_i_vop_macroblocks(&mut br, &vol, &vop).unwrap();
        // MB0 block 0 (top-left 8×8): flat 130. Blocks 1..3: their DC
        // predictor resolves inside the MB; block 1 predicts from block
        // 0's QF (65) — direction: FA=1040, FB=1024, FC=1024 →
        // |FA-FB|=16 >= |FB-FC|=0 → predict from A (left, block 0) →
        // QF = 0 + 1040/16 = 65 → flat 130 as well.
        assert_eq!(mbs[0].luma[0][0], 130);
        assert_eq!(mbs[0].luma[0][8], 130);
        // MB0 block 2 predicts from above (block 0, C): FA=default
        // (left of block 2 is outside), FB=default, FC=1040 →
        // |FA-FB|=0 < |FB-FC|=16 → from C → QF = 1040/16 = 65 → 130.
        assert_eq!(mbs[0].luma[8][0], 130);
        // MB1's blocks see real neighbours; with all differentials 0
        // each block reconstructs to its predictor / dc_scaler *
        // dc_scaler / 8. Block 0 of MB1: A = MB0 block 1 (DC 1040 after
        // its own prediction), B = outside?, C = outside → direction:
        // FA=1040, FB=1024 (default), FC=1024 (default): |FA-FB|=16 >=
        // |FB-FC|=0 → from A → QF = 1040/16 = 65 → 130.
        assert_eq!(mbs[1].luma[0][0], 130);
    }

    #[test]
    fn i_vop_video_packet_resets_dc_prediction() {
        // Two macroblocks split by a §6.2.5.2 video packet. MB0 carries
        // a +1 DC differential (flat 130). MB1 is all-zero-differential:
        // without the packet boundary its block 0 would predict from
        // MB0's block 1 (→ 130, see the cross-MB test above); with the
        // boundary the predictor grid resets and it reconstructs from
        // the §7.4.3.1 default (→ flat 128).
        let mut vol = test_vol(32, 16);
        vol.resync_marker_disable = false;
        let vop = test_i_vop(&vol, 8);
        let mut w = BitWriter::default();
        write_intra_mb_first_block_dc_plus1(&mut w); // 22 bits
                                                     // §5.2.5 stuffing to the byte boundary: '0' + '1's.
        w.write_bits(0, 1);
        w.write_bits(1, 1); // now at bit 24
                            // resync_marker: 16 zeros + 1 (I-VOP → 17 bits).
        w.write_bits(1, 17);
        // macroblock_number: 2 MBs total → 1 bit (Table 6-27).
        w.write_bits(1, 1);
        // quant_scale (quant_precision = 5).
        w.write_bits(8, 5);
        // header_extension_code = 0.
        w.write_bits(0, 1);
        write_dc_only_intra_mb(&mut w);
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let mbs = decode_i_vop_macroblocks(&mut br, &vol, &vop).unwrap();
        assert_eq!(mbs.len(), 2);
        assert_eq!(mbs[0].luma[0][0], 130);
        // The packet boundary blocked the §7.4.3 prediction: MB1 falls
        // back to the outside-VOP default DC.
        assert_eq!(mbs[1].luma[0][0], 128);
        assert_eq!(mbs[1].luma[15][15], 128);
    }

    #[test]
    fn i_vop_video_packet_skip_is_rejected() {
        // A packet whose macroblock_number jumps ahead (error
        // resilience gap) is rejected with VideoPacketSkip.
        let mut vol = test_vol(48, 16); // 3 MBs → 2-bit mb_number
        vol.resync_marker_disable = false;
        let vop = test_i_vop(&vol, 8);
        let mut w = BitWriter::default();
        write_intra_mb_first_block_dc_plus1(&mut w); // 22 bits
        w.write_bits(0, 1);
        w.write_bits(1, 1);
        w.write_bits(1, 17); // resync marker
        w.write_bits(2, 2); // macroblock_number = 2, expected 1
        w.write_bits(8, 5);
        w.write_bits(0, 1);
        write_dc_only_intra_mb(&mut w);
        let data = w.finish();
        let mut br = BitReader::new(&data);
        assert_eq!(
            decode_i_vop_macroblocks(&mut br, &vol, &vop),
            Err(VopDecodeError::VideoPacketSkip {
                packet_mb: 2,
                expected_mb: 1
            })
        );
    }

    #[test]
    fn p_vop_video_packet_resets_mv_prediction() {
        // Two P macroblocks split by a video packet, each coded inter
        // with a zero MV delta. Without the boundary MB1's §7.6.5
        // median predictor would see MB0's vector; with it the fresh
        // grid yields a zero predictor. Both decode to zero MVs here
        // (deltas are zero), so the assertion is on clean parse +
        // correct count — the boundary handling itself is proven by the
        // bit walk not desyncing across the marker.
        let mut vol = test_vol(32, 16);
        vol.resync_marker_disable = false;
        let vop = test_p_vop(&vol, 8);
        let mut w = BitWriter::default();
        // MB0: coded inter, cbp 0, zero MVD. 1+1+2+1+1 = 6 bits.
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b1, 1); // mcbpc inter cbpc 00
        w.write_bits(0b11, 2); // cbpy inter 0
        w.write_bits(1, 1); // MVDx 0
        w.write_bits(1, 1); // MVDy 0
                            // Stuffing from bit 6: '0' + '1' → bit 8 (byte aligned).
        w.write_bits(0, 1);
        w.write_bits(1, 1);
        // P-VOP resync marker: 15 + fcode(1) zeros + 1 → 17 bits.
        w.write_bits(1, 17);
        w.write_bits(1, 1); // macroblock_number = 1
        w.write_bits(8, 5); // quant_scale
        w.write_bits(0, 1); // header_extension_code
                            // MB1: same coded inter MB.
        w.write_bits(0, 1);
        w.write_bits(0b1, 1);
        w.write_bits(0b11, 2);
        w.write_bits(1, 1);
        w.write_bits(1, 1);
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let entries = decode_p_vop_macroblocks(&mut br, &vol, &vop).unwrap();
        assert_eq!(entries.len(), 2);
        for e in &entries {
            match e {
                PVopMbContent::Inter { motion, .. } => {
                    assert_eq!(motion.one_mv(), Some(MotionVector { x: 0, y: 0 }));
                }
                other => panic!("expected inter, got {other:?}"),
            }
        }
    }

    #[test]
    fn i_vop_rejects_p_header() {
        let vol = test_vol(16, 16);
        let vop = test_p_vop(&vol, 8);
        let mut br = BitReader::new(&[0u8; 4]);
        assert_eq!(
            decode_i_vop_macroblocks(&mut br, &vol, &vop),
            Err(VopDecodeError::Unsupported("not an I-VOP"))
        );
    }

    #[test]
    fn p_vop_all_skipped_copies_reference() {
        // Anchor: a flat I-VOP decoded through the bitstream walk, then
        // a P-VOP whose two macroblocks are both not_coded ("1" each).
        let vol = test_vol(32, 16);
        let ivop = test_i_vop(&vol, 8);
        let mut w = BitWriter::default();
        write_dc_only_intra_mb(&mut w);
        write_dc_only_intra_mb(&mut w);
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let imbs = decode_i_vop_macroblocks(&mut br, &vol, &ivop).unwrap();

        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 2, 1, &imbs).unwrap();

        let pvop = test_p_vop(&vol, 8);
        let mut w = BitWriter::default();
        w.write_bits(1, 1); // MB0 not_coded
        w.write_bits(1, 1); // MB1 not_coded
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let entries = decode_p_vop_macroblocks(&mut br, &vol, &pvop).unwrap();
        assert_eq!(entries.len(), 2);
        let frame = assemble_p_vop_frame(
            &store,
            2,
            1,
            &entries,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(128));
        assert_eq!(frame.luma_at(31, 15), Some(128));
    }

    #[test]
    fn p_vop_inter_mb_zero_mv_no_residual_is_copy() {
        // One coded inter macroblock: not_coded 0, mcbpc "1" (inter,
        // cbpc 00), cbpy "0011" (inter pattern 0 — Table B.8 code
        // "0011" is intra 0 / inter 15?): use the inter column: the
        // parser stores the coded-pattern form. Code "0011" → intra 0,
        // inter 15 would code ALL blocks; we want zero coded blocks →
        // intra 15 / inter 0 → code "11". Then one zero MV delta
        // ("1" + "1" for x and y with fcode 1) and no texture.
        let vol = test_vol(16, 16);
        let ivop = test_i_vop(&vol, 8);
        let mut w = BitWriter::default();
        write_intra_mb_first_block_dc_plus1(&mut w);
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let imbs = decode_i_vop_macroblocks(&mut br, &vol, &ivop).unwrap();
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 1, 1, &imbs).unwrap();
        let anchor_00 = store.backward().unwrap().luma_at(0, 0).unwrap();
        let anchor_1515 = store.backward().unwrap().luma_at(15, 15).unwrap();

        let pvop = test_p_vop(&vol, 8);
        let mut w = BitWriter::default();
        w.write_bits(0, 1); // not_coded = 0
        w.write_bits(0b1, 1); // mcbpc: inter, cbpc 00 (P Table B.7 "1")
        w.write_bits(0b11, 2); // cbpy: inter pattern 0000 (Table B.8 "11")
        w.write_bits(1, 1); // MVDx: Table B.12 code "1" → 0
        w.write_bits(1, 1); // MVDy: 0
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let entries = decode_p_vop_macroblocks(&mut br, &vol, &pvop).unwrap();
        assert_eq!(entries.len(), 1);
        let frame = assemble_p_vop_frame(
            &store,
            1,
            1,
            &entries,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        // Zero MV + zero residual = pure copy of the anchor.
        assert_eq!(frame.luma_at(0, 0), Some(anchor_00));
        assert_eq!(frame.luma_at(15, 15), Some(anchor_1515));
        // The +1 DC differential propagated through the §7.4.3 DC
        // prediction, so the anchor (and the copy) is flat 130.
        assert_eq!(anchor_00, 130);
    }

    #[test]
    fn p_vop_intra_mb_decodes_via_intra_path() {
        // A P-VOP whose single macroblock is intra: not_coded 0, mcbpc
        // P-table intra code ("00011"), ac_pred 0, cbpy intra-0
        // ("0011"), six size-0 DCs → flat 128 regardless of the anchor.
        let vol = test_vol(16, 16);
        let ivop = test_i_vop(&vol, 8);
        let mut w = BitWriter::default();
        write_intra_mb_first_block_dc_plus1(&mut w); // anchor luma 130
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let imbs = decode_i_vop_macroblocks(&mut br, &vol, &ivop).unwrap();
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 1, 1, &imbs).unwrap();

        let pvop = test_p_vop(&vol, 8);
        let mut w = BitWriter::default();
        w.write_bits(0, 1); // not_coded = 0
        w.write_bits(0b00011, 5); // mcbpc: intra, cbpc 00 (Table B.7)
        w.write_bits(0, 1); // ac_pred_flag
        w.write_bits(0b0011, 4); // cbpy intra 0
        for _ in 0..4 {
            w.write_bits(0b011, 3);
        }
        for _ in 0..2 {
            w.write_bits(0b11, 2);
        }
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let entries = decode_p_vop_macroblocks(&mut br, &vol, &pvop).unwrap();
        let frame = assemble_p_vop_frame(
            &store,
            1,
            1,
            &entries,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(128));
    }

    /// A rectangular GMC VOL (verid 2, sprite_enable "10", 0 warping
    /// points, half-pel accuracy).
    fn test_gmc_vol(width: u16, height: u16, points: u32) -> VolHeader {
        let mut w = BitWriter::default();
        w.write_bits(0x0000_0120, 32);
        w.write_bits(0, 1); // random_accessible_vol
        w.write_bits(1, 8); // video_object_type_indication
        w.write_bits(1, 1); // is_object_layer_identifier
        w.write_bits(2, 4); // video_object_layer_verid = 2
        w.write_bits(0, 3); // video_object_layer_priority
        w.write_bits(1, 4); // aspect_ratio_info 1:1
        w.write_bits(0, 1); // vol_control_parameters
        w.write_bits(0, 2); // shape rectangular
        w.write_bits(1, 1); // marker
        w.write_bits(30, 16); // vop_time_increment_resolution
        w.write_bits(1, 1); // marker
        w.write_bits(0, 1); // fixed_vop_rate
        w.write_bits(1, 1); // marker
        w.write_bits(u32::from(width), 13);
        w.write_bits(1, 1); // marker
        w.write_bits(u32::from(height), 13);
        w.write_bits(1, 1); // marker
        w.write_bits(0, 1); // interlaced
        w.write_bits(1, 1); // obmc_disable
        w.write_bits(0b10, 2); // sprite_enable: GMC (verid != 1)
        w.write_bits(points, 6); // no_of_sprite_warping_points
        w.write_bits(0, 2); // sprite_warping_accuracy: half-pel
        w.write_bits(0, 1); // sprite_brightness_change
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type
        w.write_bits(0, 1); // quarter_sample (verid != 1)
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(1, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // newpred_enable (verid != 1)
        w.write_bits(0, 1); // reduced_resolution_vop_enable
        w.write_bits(0, 1); // scalability
        let data = w.finish();
        crate::vol::parse_video_object_layer(&data, 0).expect("test GMC VOL must parse")
    }

    /// A coded S(GMC)-VOP header. `du` supplies the single warping
    /// point's `(du, dv)` when the VOL declared one point; the Table
    /// B.34 warping_mv_code for 0 is the 2-bit VLC "00" + no residual
    /// bits... encode via the sprite module's expectations: dpcm code
    /// length table row 0 has code "00" and zero magnitude bits.
    fn test_s_vop(vol: &VolHeader, quant: u32, du: Option<(i32, i32)>) -> VopHeader {
        let mut w = BitWriter::default();
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(0b11, 2); // vop_coding_type: S
        w.write_bits(0, 1); // modulo_time_base
        w.write_bits(1, 1); // marker
        w.write_bits(0, 5); // time increment
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 1); // vop_rounding_type (S-GMC)
        w.write_bits(0, 3); // intra_dc_vlc_thr
        if let Some((du, dv)) = du {
            // §6.2.5 sprite_trajectory(): one warping point. Table B.34:
            // value 0 → dmv_length SSS = 0 (a lone "0" unary terminator,
            // no magnitude bits) + the trailing marker_bit. Only (0, 0)
            // is emitted by this helper.
            assert_eq!((du, dv), (0, 0), "helper only encodes a zero du/dv");
            w.write_bits(0, 1); // warping_mv_code(du): SSS = 0
            w.write_bits(1, 1); // marker
            w.write_bits(0, 1); // warping_mv_code(dv): SSS = 0
            w.write_bits(1, 1); // marker
        }
        w.write_bits(quant, 5); // vop_quant
        w.write_bits(1, 3); // vop_fcode_forward
        let data = w.finish();
        crate::vop::parse_video_object_plane_header(
            &data,
            vol.time_increment_resolution,
            VopContext::from_vol(vol),
        )
        .expect("test S-VOP must parse")
    }

    /// Decode a flat I-VOP anchor and install it in a fresh store.
    fn flat_anchor_store(vol: &VolHeader) -> FrameStore {
        let ivop = test_i_vop(vol, 8);
        let (mbw, mbh) = vop_mb_dimensions(vol);
        let mut w = BitWriter::default();
        for _ in 0..mbw * mbh {
            write_dc_only_intra_mb(&mut w);
        }
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let imbs = decode_i_vop_macroblocks(&mut br, vol, &ivop).unwrap();
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, mbw, mbh, &imbs).unwrap();
        store
    }

    #[test]
    fn s_gmc_not_coded_mb_is_stationary_gmc_copy() {
        // 0 warping points → stationary warp; a not_coded S(GMC) MB is
        // an implied mcsel == 1 GMC copy of the reference.
        let vol = test_gmc_vol(16, 16, 0);
        let store = flat_anchor_store(&vol);
        let svop = test_s_vop(&vol, 8, None);

        let mut w = BitWriter::default();
        w.write_bits(1, 1); // not_coded
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let (entries, geometry) = decode_s_gmc_vop_macroblocks(&mut br, &vol, &svop).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0], SGmcMbContent::Gmc { .. }));
        let frame = crate::frame_decode::assemble_s_gmc_vop_frame(
            &store,
            1,
            1,
            &entries,
            &geometry,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(128));
        assert_eq!(frame.luma_at(15, 15), Some(128));
    }

    #[test]
    fn s_gmc_mcsel1_coded_mb_reads_no_motion_bits() {
        // A coded inter MB with mcsel == 1 and cbp == 0: header bits
        // only (not_coded 0, mcbpc "1", mcsel 1, cbpy "11"), no MV
        // bodies, no texture — a pure GMC copy.
        let vol = test_gmc_vol(16, 16, 0);
        let store = flat_anchor_store(&vol);
        let svop = test_s_vop(&vol, 8, None);

        let mut w = BitWriter::default();
        w.write_bits(0, 1); // not_coded = 0
        w.write_bits(0b1, 1); // mcbpc: inter, cbpc 00
        w.write_bits(1, 1); // mcsel = 1
        w.write_bits(0b11, 2); // cbpy: inter pattern 0
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let (entries, geometry) = decode_s_gmc_vop_macroblocks(&mut br, &vol, &svop).unwrap();
        assert_eq!(br.bit_position(), 5, "no motion/texture bits consumed");
        let frame = crate::frame_decode::assemble_s_gmc_vop_frame(
            &store,
            1,
            1,
            &entries,
            &geometry,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(128));
    }

    #[test]
    fn s_gmc_mcsel0_local_mb_takes_p_vop_path() {
        // mcsel == 0: the plain P-VOP local path (one zero MVD pair).
        let vol = test_gmc_vol(16, 16, 0);
        let store = flat_anchor_store(&vol);
        let svop = test_s_vop(&vol, 8, None);

        let mut w = BitWriter::default();
        w.write_bits(0, 1); // not_coded = 0
        w.write_bits(0b1, 1); // mcbpc: inter, cbpc 00
        w.write_bits(0, 1); // mcsel = 0
        w.write_bits(0b11, 2); // cbpy: inter pattern 0
        w.write_bits(1, 1); // MVDx = 0
        w.write_bits(1, 1); // MVDy = 0
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let (entries, geometry) = decode_s_gmc_vop_macroblocks(&mut br, &vol, &svop).unwrap();
        assert!(matches!(entries[0], SGmcMbContent::Local { .. }));
        let frame = crate::frame_decode::assemble_s_gmc_vop_frame(
            &store,
            1,
            1,
            &entries,
            &geometry,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(128));
    }

    #[test]
    fn s_gmc_one_point_zero_trajectory_parses_and_copies() {
        // One warping point with du = dv = 0 exercises the §6.2.5
        // sprite_trajectory() capture on the VOP header + the 1-point
        // translation warp (identity for a zero delta).
        let vol = test_gmc_vol(16, 16, 1);
        let store = flat_anchor_store(&vol);
        let svop = test_s_vop(&vol, 8, Some((0, 0)));
        assert!(svop.sprite_trajectory.is_some());

        let mut w = BitWriter::default();
        w.write_bits(1, 1); // not_coded
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let (entries, geometry) = decode_s_gmc_vop_macroblocks(&mut br, &vol, &svop).unwrap();
        let frame = crate::frame_decode::assemble_s_gmc_vop_frame(
            &store,
            1,
            1,
            &entries,
            &geometry,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(128));
    }

    /// A coded B-VOP header (quant 8, fcode_fwd = fcode_bwd = 1).
    fn test_b_vop(vol: &VolHeader, quant: u32) -> VopHeader {
        let mut w = BitWriter::default();
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(0b10, 2); // vop_coding_type: B
        w.write_bits(0, 1); // modulo_time_base
        w.write_bits(1, 1); // marker
        w.write_bits(0, 5); // time increment
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(quant, 5); // vop_quant
        w.write_bits(1, 3); // vop_fcode_forward
        w.write_bits(1, 3); // vop_fcode_backward
        let data = w.finish();
        crate::vop::parse_video_object_plane_header(
            &data,
            vol.time_increment_resolution,
            VopContext::from_vol(vol),
        )
        .expect("test B-VOP must parse")
    }

    /// A store with distinguishable anchors: forward I (flat 40),
    /// backward P (flat 80, intra content).
    fn two_flat_anchor_store() -> FrameStore {
        use crate::reconstruct::{
            ReconstructedMacroblock, MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE,
        };
        let flat = |v: i32| ReconstructedMacroblock {
            luma: [[v; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[v; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[v; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        };
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 1, 1, &[flat(40)]).unwrap();
        let entries = vec![PVopMbContent::Intra(flat(80))];
        crate::frame_decode::decode_p_vop(
            &mut store,
            1,
            1,
            &entries,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        store
    }

    #[test]
    fn b_vop_forward_mb_copies_forward_anchor() {
        // modb "01" (mb_type present, no cbpb) + mb_type "0001"
        // (forward) + zero MVDf ("1", "1") → a forward copy of the past
        // anchor (luma 40).
        let vol = test_vol(16, 16);
        let store = two_flat_anchor_store();
        let bvop = test_b_vop(&vol, 8);

        let mut w = BitWriter::default();
        w.write_bits(0b01, 2); // modb
        w.write_bits(0b0001, 4); // mb_type: forward
        w.write_bits(1, 1); // MVDfx = 0
        w.write_bits(1, 1); // MVDfy = 0
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let anchor = [PvopMbMotion::Intra];
        let entries = decode_b_vop_macroblocks(&mut br, &vol, &bvop, 1, 2, Some(&anchor)).unwrap();
        assert_eq!(entries.len(), 1);
        let frame = crate::frame_decode::assemble_b_vop_frame(
            &store,
            1,
            1,
            &entries,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(40));
    }

    #[test]
    fn b_vop_modb1_direct_zero_averages_anchors() {
        // modb "1": no mb_type / cbpb / motion bits — direct mode with
        // a zero delta over a zero co-located MV → both derived MVs are
        // zero → the §7.6.9.4 average of the anchors: (40+80+1)>>1 = 60.
        let vol = test_vol(16, 16);
        let store = two_flat_anchor_store();
        let bvop = test_b_vop(&vol, 8);

        let mut w = BitWriter::default();
        w.write_bits(0b1, 1); // modb = "1"
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let anchor = [PvopMbMotion::OneMv(MotionVector { x: 0, y: 0 })];
        let entries = decode_b_vop_macroblocks(&mut br, &vol, &bvop, 1, 2, Some(&anchor)).unwrap();
        assert_eq!(br.bit_position(), 1, "modb '1' consumes exactly one bit");
        let frame = crate::frame_decode::assemble_b_vop_frame(
            &store,
            1,
            1,
            &entries,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(60));
    }

    #[test]
    fn b_vop_skipped_co_located_transmits_nothing() {
        // §6.2.6 co_located_not_coded: the co-located future-P-anchor
        // MB was skipped, so the B macroblock transmits NO bits (the
        // whole modb subtree is gated out) and reconstructs per
        // §7.6.9.6 as forward mode with the zero MV — a copy of the
        // past anchor (luma 40).
        let vol = test_vol(16, 16);
        let store = two_flat_anchor_store();
        let bvop = test_b_vop(&vol, 8);

        let data = [0xFFu8; 2]; // arbitrary — must remain unread
        let mut br = BitReader::new(&data);
        let anchor = [PvopMbMotion::Skipped];
        let entries = decode_b_vop_macroblocks(&mut br, &vol, &bvop, 1, 2, Some(&anchor)).unwrap();
        assert_eq!(br.bit_position(), 0, "no bits may be consumed");
        let frame = crate::frame_decode::assemble_b_vop_frame(
            &store,
            1,
            1,
            &entries,
            0,
            crate::bvop_prediction::BVopSampleMode::HalfPel,
            8,
        )
        .unwrap();
        // Forward-zero copy of the past anchor.
        assert_eq!(frame.luma_at(0, 0), Some(40));
    }

    #[test]
    fn s_gmc_rejects_non_gmc_vol() {
        let vol = test_vol(16, 16); // sprite_enable: NotUsed
        let svop = test_i_vop(&vol, 8); // wrong type on purpose
        let mut br = BitReader::new(&[0u8; 2]);
        assert!(matches!(
            decode_s_gmc_vop_macroblocks(&mut br, &vol, &svop),
            Err(VopDecodeError::Unsupported("not an S-VOP"))
        ));
    }

    #[test]
    fn p_vop_quarter_sample_vol_decodes() {
        // §7.6.3: the motion-vector syntax is unit-agnostic, so a
        // quarter-sample VOL walks the same bit layout — the decoded
        // integers simply are quarter-pel units. A one-MB P-VOP with a
        // zero-delta inter MB must decode (not be gated off).
        let mut vol = test_vol(16, 16);
        vol.quarter_sample = true;
        let pvop = test_p_vop(&vol, 8);
        let mut w = BitWriter::default();
        w.write_bits(0, 1); // not_coded = 0
        w.write_bits(0b1, 1); // mcbpc: inter, cbpc 00 (Table B.7 "1")
        w.write_bits(0b11, 2); // cbpy: inter pattern 0 (Table B.8 "11")
        w.write_bits(1, 1); // MVDx = 0 (Table B.12 code "1")
        w.write_bits(1, 1); // MVDy = 0
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let entries = decode_p_vop_macroblocks(&mut br, &vol, &pvop).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(matches!(
            entries[0],
            PVopMbContent::Inter {
                motion: PvopMbMotion::OneMv(MotionVector { x: 0, y: 0 }),
                ..
            }
        ));
    }
}

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
//! non-short-video-header VOPs with `obmc_disable == 1` and
//! `quarter_sample == 0` (the half-sample §7.6.2.1 prediction path).
//! Out-of-scope VOL configurations are rejected with a typed
//! [`VopDecodeError::Unsupported`] before any bit is consumed, so the
//! reader never drifts. The data-partitioned layouts live in
//! [`crate::data_partition`]; the interlaced texture path (dct_type
//! re-ordering + alternate vertical scan) is a later milestone.

use crate::bitreader::BitReader;
use crate::block::{
    decode_inter_macroblock, decode_intra_block_full, intra_quant_matrix, nonintra_quant_matrix,
    pattern_code, BlockAssemblyError, IntraMacroblock, MacroblockTextureContext,
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
        }
    }
}

impl std::error::Error for VopDecodeError {}

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

/// Validate the shared VOL gates of the progressive rectangular walks.
fn check_vol_supported(vol: &VolHeader) -> Result<(), VopDecodeError> {
    if vol.video_object_layer_shape != 0 {
        return Err(VopDecodeError::Unsupported("non-rectangular shape"));
    }
    if vol.data_partitioned {
        return Err(VopDecodeError::Unsupported("data_partitioned"));
    }
    if vol.interlaced {
        return Err(VopDecodeError::Unsupported("interlaced"));
    }
    Ok(())
}

/// The running quantiser ceiling `2^quant_precision - 1` (§6.3.3).
fn max_quantiser_scale(vol: &VolHeader) -> u32 {
    (1u32 << vol.quant_precision) - 1
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
    let mut grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut running_qp = u32::from(vop.quant).clamp(1, max_qp);
    let mut out = Vec::with_capacity(mb_width * mb_height);

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let header = parse_macroblock_header(br, VopCodingType::I, vol)?;
            running_qp = apply_dquant(running_qp, header.dquant_delta, max_qp);
            let use_dc_vlc = use_intra_dc_vlc(vop.intra_dc_vlc_thr, running_qp);
            let ctx = MacroblockTextureContext {
                quantiser_scale: running_qp,
                bits_per_pixel: bpp,
                quant_type: vol.quant_type,
                ac_pred_flag: header.ac_pred_flag,
            };
            let coded = pattern_code(header.cbpy, header.cbpc);
            let mb = decode_intra_mb_with_grid(
                br,
                &mut grid,
                mb_row,
                mb_col,
                coded,
                use_dc_vlc,
                ctx,
                &quant_matrix,
            )?;
            out.push(reconstruct_intra_macroblock(&mb, bpp));
        }
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
/// The half-sample §7.6.2.1 path only: `quarter_sample == 1` and
/// `obmc_disable == 0` are rejected up front.
pub fn decode_p_vop_macroblocks(
    br: &mut BitReader<'_>,
    vol: &VolHeader,
    vop: &VopHeader,
) -> Result<Vec<PVopMbContent>, VopDecodeError> {
    check_vol_supported(vol)?;
    if !matches!(vop.coding_type, VopCodingType::P) {
        return Err(VopDecodeError::Unsupported("not a P-VOP"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }
    if vol.quarter_sample {
        return Err(VopDecodeError::Unsupported("quarter_sample"));
    }
    if !vol.obmc_disable {
        return Err(VopDecodeError::Unsupported("obmc"));
    }

    let (mb_width, mb_height) = vop_mb_dimensions(vol);
    let bpp = u32::from(vol.bits_per_pixel);
    let max_qp = max_quantiser_scale(vol);
    let intra_matrix = intra_quant_matrix(vol);
    let inter_matrix = nonintra_quant_matrix(vol);
    let mut intra_grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut driver = MvDriver::new(mb_height, mb_width, vop.fcode_fwd);
    let mut running_qp = u32::from(vop.quant).clamp(1, max_qp);
    let mut out = Vec::with_capacity(mb_width * mb_height);

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
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
            };

            let is_intra = header.mb_type.map(|t| t.is_intra()).unwrap_or(false);
            if is_intra {
                // Record the intra MB as an invalid MV-predictor
                // candidate (§7.6.5); no motion bits are coded.
                driver.decode_macroblock(br, mb_row, mb_col, false, header.mb_type)?;
                let use_dc_vlc = use_intra_dc_vlc(vop.intra_dc_vlc_thr, running_qp);
                let coded = pattern_code(header.cbpy, header.cbpc);
                let mb = decode_intra_mb_with_grid(
                    br,
                    &mut intra_grid,
                    mb_row,
                    mb_col,
                    coded,
                    use_dc_vlc,
                    ctx,
                    &intra_matrix,
                )?;
                out.push(PVopMbContent::Intra(reconstruct_intra_macroblock(&mb, bpp)));
            } else {
                // §6.2.6: the motion_vector() bodies precede the texture.
                let motion = driver.decode_macroblock(br, mb_row, mb_col, false, header.mb_type)?;
                let residual = decode_inter_macroblock(br, &header, ctx, &inter_matrix)?;
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
/// * a 1-MV inter MB → its vector,
/// * a 4-MV inter MB → its **first** sub-block vector (the driver's
///   direct derivation applies one co-located vector per macroblock;
///   per-sub-block co-located vectors are a later refinement),
/// * an intra MB (or a GMC MB, which carries no local vector) → the
///   §7.6.9.5.1 final-sentence zero-vector fallback.
fn co_located_from_motion(motion: PvopMbMotion) -> CoLocatedAnchor {
    match motion {
        PvopMbMotion::Skipped => CoLocatedAnchor {
            skipped: true,
            mv: DirectCoLocatedMv::TransparentOrAbsent,
        },
        PvopMbMotion::OneMv(mv) => CoLocatedAnchor {
            skipped: false,
            mv: DirectCoLocatedMv::Mv(mv),
        },
        PvopMbMotion::FourMv(mvs) => CoLocatedAnchor {
            skipped: false,
            mv: DirectCoLocatedMv::Mv(mvs[0]),
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
    if !matches!(vop.coding_type, VopCodingType::B) {
        return Err(VopDecodeError::Unsupported("not a B-VOP"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }
    if vol.quarter_sample {
        return Err(VopDecodeError::Unsupported("quarter_sample"));
    }

    let (mb_width, mb_height) = vop_mb_dimensions(vol);
    let max_qp = max_quantiser_scale(vol);
    let mut driver = BVopMvDriver::new(mb_height, mb_width, vop.fcode_fwd, vop.fcode_bwd, trb, trd);
    let texture = BVopTextureParams {
        base_quantiser_scale: u32::from(vop.quant).clamp(1, max_qp),
        max_quantiser_scale: max_qp,
        bits_per_pixel: u32::from(vol.bits_per_pixel),
        quant_type: vol.quant_type,
    };
    driver
        .decode_vop(
            br,
            vol,
            vop.coding_type,
            texture,
            |mb_row, mb_col| match anchor_motion {
                Some(motion) => co_located_from_motion(motion[mb_row * mb_width + mb_col]),
                None => CoLocatedAnchor::default(),
            },
        )
        .map_err(VopDecodeError::BVop)
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
    if !matches!(vop.coding_type, VopCodingType::S) {
        return Err(VopDecodeError::Unsupported("not an S-VOP"));
    }
    if !matches!(vol.sprite_enable, SpriteEnable::Gmc) {
        return Err(VopDecodeError::Unsupported("sprite_enable != GMC"));
    }
    if !vop.coded {
        return Err(VopDecodeError::Unsupported("vop_coded == 0"));
    }
    if vol.quarter_sample {
        return Err(VopDecodeError::Unsupported("quarter_sample"));
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
    let mut intra_grid = IntraBlockGrid::new(mb_height, mb_width);
    let mut driver = MvDriver::new(mb_height, mb_width, vop.fcode_fwd);
    let mut running_qp = u32::from(vop.quant).clamp(1, max_qp);
    let mut out = Vec::with_capacity(mb_width * mb_height);

    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            let header = parse_macroblock_header(br, VopCodingType::S, vol)?;
            let mb_x = (mb_col * 16) as i64;
            let mb_y = (mb_row * 16) as i64;

            if header.not_coded {
                // §6.3.6: a not-coded S(GMC) macroblock is GMC-predicted
                // (implied mcsel == 1) with a zero residual.
                let amv = gmc_averaged_mv(&geometry, mb_x, mb_y, false, vop.fcode_fwd)?;
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
            };

            let is_intra = header.mb_type.map(|t| t.is_intra()).unwrap_or(false);
            if is_intra {
                driver.decode_macroblock(br, mb_row, mb_col, false, header.mb_type)?;
                let use_dc_vlc = use_intra_dc_vlc(vop.intra_dc_vlc_thr, running_qp);
                let coded = pattern_code(header.cbpy, header.cbpc);
                let mb = decode_intra_mb_with_grid(
                    br,
                    &mut intra_grid,
                    mb_row,
                    mb_col,
                    coded,
                    use_dc_vlc,
                    ctx,
                    &intra_matrix,
                )?;
                out.push(SGmcMbContent::Intra(reconstruct_intra_macroblock(&mb, bpp)));
            } else if header.mcsel == Some(true) {
                // §6.3.6: mcsel == 1 codes no local motion vectors; the
                // texture follows the header directly.
                let amv = gmc_averaged_mv(&geometry, mb_x, mb_y, false, vop.fcode_fwd)?;
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
        let frame = assemble_p_vop_frame(&store, 2, 1, &entries, 0, 8).unwrap();
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
        let frame = assemble_p_vop_frame(&store, 1, 1, &entries, 0, 8).unwrap();
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
        let frame = assemble_p_vop_frame(&store, 1, 1, &entries, 0, 8).unwrap();
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
        let frame =
            crate::frame_decode::assemble_s_gmc_vop_frame(&store, 1, 1, &entries, &geometry, 0, 8)
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
        let frame =
            crate::frame_decode::assemble_s_gmc_vop_frame(&store, 1, 1, &entries, &geometry, 0, 8)
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
        let frame =
            crate::frame_decode::assemble_s_gmc_vop_frame(&store, 1, 1, &entries, &geometry, 0, 8)
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
        let frame =
            crate::frame_decode::assemble_s_gmc_vop_frame(&store, 1, 1, &entries, &geometry, 0, 8)
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
        crate::frame_decode::decode_p_vop(&mut store, 1, 1, &entries, 0, 8).unwrap();
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
    fn b_vop_skipped_co_located_forces_override() {
        // The co-located future-anchor MB was skipped: §7.6.9.6 forces
        // the modb "1" macroblock to direct-zero regardless — and the
        // walk must route the skipped flag through.
        let vol = test_vol(16, 16);
        let store = two_flat_anchor_store();
        let bvop = test_b_vop(&vol, 8);

        let mut w = BitWriter::default();
        w.write_bits(0b1, 1); // modb = "1"
        let data = w.finish();
        let mut br = BitReader::new(&data);
        let anchor = [PvopMbMotion::Skipped];
        let entries = decode_b_vop_macroblocks(&mut br, &vol, &bvop, 1, 2, Some(&anchor)).unwrap();
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
        // Direct zero over zero co-located → the bidirectional average.
        assert_eq!(frame.luma_at(0, 0), Some(60));
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
    fn p_vop_rejects_quarter_sample() {
        let mut vol = test_vol(16, 16);
        let pvop = test_p_vop(&vol, 8);
        vol.quarter_sample = true;
        let mut br = BitReader::new(&[0u8; 4]);
        assert!(matches!(
            decode_p_vop_macroblocks(&mut br, &vol, &pvop),
            Err(VopDecodeError::Unsupported("quarter_sample"))
        ));
    }
}

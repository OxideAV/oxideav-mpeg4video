//! Frame-assembly drivers that thread the §7.6.1 reference-frame chain
//! ([`FrameStore`]) into the per-macroblock reconstruction routines and
//! blit the results into a fresh [`DecodedFrame`].
//!
//! The motion-vector + residual decode for each macroblock already lands
//! at the per-MB level ([`MvDriver`](crate::pvop_mv::MvDriver),
//! [`BVopMvDriver`](crate::bvop_mv::BVopMvDriver), the §7.4 texture
//! routines). What this module adds is the *frame-level* loop that:
//!
//! 1. pulls the correct reference plane(s) out of the [`FrameStore`] per
//!    the §7.5.2.1.2 reference-VOP rules (forward anchor for a P-/S-VOP,
//!    the bracketing forward+backward pair for a B-VOP),
//! 2. reconstructs each macroblock to `d[y][x]` pixels, and
//! 3. blits the macroblock into the output frame at its grid position.
//!
//! The result is a complete decoded VOP ready to (a) be displayed and
//! (b) — for an I-/P-/S-VOP — be pushed back into the store as the next
//! anchor.
//!
//! This module is deliberately *content-agnostic* about how each
//! macroblock's motion and residual were decoded: the caller supplies a
//! per-MB [`PVopMbContent`] (already-decoded motion + residual, or an
//! already-reconstructed intra macroblock). That keeps the frame loop
//! independent of the §6.2.6 bitstream walk while still closing the
//! "reference-plane selection per the VOP reference-frame chain" gap.

use crate::block::InterMacroblock;
use crate::bvop_mv::{BVopAnchorPlanes, BVopMbTexturedDecode};
use crate::bvop_prediction::BVopSampleMode;
use crate::framestore::{DecodedFrame, FrameStore, FrameStoreError};
use crate::pvop_mv::{predict_inter_macroblock, reconstruct_pvop_macroblock, PvopMbMotion};
use crate::reconstruct::{reconstruct_inter_macroblock, ReconstructedMacroblock};
use crate::s_gmc_recon::{gmc_prediction_macroblock, GmcReferencePlanes};
use crate::vop::VopCodingType;
use crate::warp::WarpGeometry;

/// Errors raised while assembling a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDecodeError {
    /// The reference-frame chain lacked the anchor(s) this VOP type needs
    /// (a P-VOP with no preceding I/P/S-VOP, or a B-VOP missing one of
    /// its two anchors).
    MissingReference,
    /// A per-macroblock entry's `(mb_col, mb_row)` fell outside the
    /// frame's macroblock grid, or the output frame failed to build.
    FrameStore(FrameStoreError),
    /// The number of supplied macroblock entries did not equal
    /// `mb_width * mb_height`.
    MacroblockCountMismatch {
        /// Entries the caller supplied.
        supplied: usize,
        /// Entries expected (`mb_width * mb_height`).
        expected: usize,
    },
    /// A reference plane could not be reconstructed for an inter
    /// macroblock (e.g. the §7.6.2 prediction returned `None` for a
    /// macroblock the caller tagged inter). This indicates an
    /// intra/inter tagging inconsistency in the caller's content.
    InterPredictionFailed {
        /// Macroblock column where the prediction failed.
        mb_col: usize,
        /// Macroblock row where the prediction failed.
        mb_row: usize,
    },
}

impl From<FrameStoreError> for FrameDecodeError {
    fn from(e: FrameStoreError) -> Self {
        FrameDecodeError::FrameStore(e)
    }
}

/// Per-macroblock content for a P-VOP frame assembly.
///
/// Either an inter macroblock (its decoded §7.6 motion + §7.4 residual)
/// or an intra macroblock (already reconstructed to `d[y][x]` samples by
/// the §7.4 intra texture path). The §7.6.1 reference plane is supplied
/// by the frame loop, not the caller.
#[derive(Debug, Clone, PartialEq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub enum PVopMbContent {
    /// An inter macroblock: the [`PvopMbMotion`] the
    /// [`MvDriver`](crate::pvop_mv::MvDriver) decoded plus the §7.4 inter
    /// residual. Reconstructed against the forward reference plane.
    Inter {
        /// Decoded motion (1-MV / 4-MV / skipped). `Intra` here is a
        /// caller error — use [`PVopMbContent::Intra`] instead.
        motion: PvopMbMotion,
        /// §7.4 inter residual (`InterMacroblock::zero()` for an
        /// all-zero `cbp`).
        residual: InterMacroblock,
    },
    /// A §7.7.2.1 **field-predicted** inter macroblock of an interlaced
    /// VOL: two field motion vectors (frame coordinates, even vertical
    /// components) plus the §6.2.6.3 reference-field selection bits and
    /// the §7.4 residual (its luminance already inverse-field-DCT
    /// permuted when the macroblock coded `dct_type == 1`).
    FieldInter {
        /// The reconstructed top/bottom field motion vectors.
        mvs: crate::field_motion::FieldMotionVectors,
        /// `forward_top_field_reference` (§6.3.7.2 raw bit — `false` =
        /// top reference field, `true` = bottom).
        top_field_ref: bool,
        /// `forward_bottom_field_reference` (same convention).
        bottom_field_ref: bool,
        /// §7.4 inter residual.
        residual: InterMacroblock,
    },
    /// An intra macroblock already reconstructed to pixels.
    Intra(ReconstructedMacroblock),
}

/// Assemble a complete P-VOP frame from per-macroblock content against
/// the forward reference plane in `store`, returning the decoded frame
/// (not yet pushed into the chain).
///
/// `mb_width` / `mb_height` give the frame's macroblock grid; `entries`
/// must contain exactly `mb_width * mb_height` items in raster order
/// (row-major: `(0,0), (1,0), …, (mb_width-1, 0), (0,1), …`).
///
/// The forward reference is the §7.5.2.1.2 most-recently-decoded
/// I-/P-/S-VOP — i.e. [`FrameStore::p_vop_reference`]. Each inter
/// macroblock is reconstructed via [`reconstruct_pvop_macroblock`]
/// against that plane; each intra macroblock is blitted as-is.
///
/// `vop_rounding_type` is the §7.6.7 half-pel rounding control;
/// `sample_mode` the VOL-level `quarter_sample` selection (§7.6.2.1 vs
/// §7.6.2.2 luminance interpolation); `bits_per_pixel` the §6.3.3
/// sample depth for the §7.3 clip.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn assemble_p_vop_frame(
    store: &FrameStore,
    mb_width: usize,
    mb_height: usize,
    entries: &[PVopMbContent],
    vop_rounding_type: u8,
    sample_mode: BVopSampleMode,
    bits_per_pixel: u32,
) -> Result<DecodedFrame, FrameDecodeError> {
    let expected = mb_width * mb_height;
    if entries.len() != expected {
        return Err(FrameDecodeError::MacroblockCountMismatch {
            supplied: entries.len(),
            expected,
        });
    }
    let reference = store
        .p_vop_reference()
        .ok_or(FrameDecodeError::MissingReference)?;
    let luma_ref = reference.luma_reference();
    let cb_ref = reference.cb_reference();
    let cr_ref = reference.cr_reference();

    let mut frame = DecodedFrame::new(mb_width * 16, mb_height * 16, VopCodingType::P)?;

    for (idx, entry) in entries.iter().enumerate() {
        let mb_col = idx % mb_width;
        let mb_row = idx / mb_width;
        let reconstructed = match entry {
            PVopMbContent::Inter { motion, residual } => reconstruct_pvop_macroblock(
                *motion,
                &luma_ref,
                &cb_ref,
                &cr_ref,
                residual,
                (mb_col * 16) as i32,
                (mb_row * 16) as i32,
                vop_rounding_type,
                sample_mode,
                bits_per_pixel,
            )
            .ok_or(FrameDecodeError::InterPredictionFailed { mb_col, mb_row })?,
            PVopMbContent::FieldInter {
                mvs,
                top_field_ref,
                bottom_field_ref,
                residual,
            } => {
                // §7.7.2.1 field_motion_compensate_one_reference — the
                // quarter-sample VOL swaps in the §7.6.2.2 field
                // interpolation for the two luma field blocks.
                let prediction = match sample_mode {
                    BVopSampleMode::HalfPel => {
                        crate::field_motion::field_motion_compensate_one_reference(
                            &luma_ref,
                            &cb_ref,
                            &cr_ref,
                            *mvs,
                            *top_field_ref,
                            *bottom_field_ref,
                            (mb_col * 16) as i32,
                            (mb_row * 16) as i32,
                            vop_rounding_type,
                        )
                    }
                    BVopSampleMode::QuarterPel { bits_per_pixel } => {
                        crate::field_motion::field_motion_compensate_one_reference_qpel(
                            &luma_ref,
                            &cb_ref,
                            &cr_ref,
                            *mvs,
                            *top_field_ref,
                            *bottom_field_ref,
                            (mb_col * 16) as i32,
                            (mb_row * 16) as i32,
                            vop_rounding_type,
                            bits_per_pixel,
                        )
                    }
                };
                reconstruct_inter_macroblock(&prediction, residual, bits_per_pixel)
            }
            PVopMbContent::Intra(mb) => mb.clone(),
        };
        frame.blit_macroblock(mb_col, mb_row, &reconstructed)?;
    }

    Ok(frame)
}

/// Per-8×8-luminance-block motion classification feeding the §7.6.6
/// remote-MV substitution rules.
#[derive(Debug, Clone, Copy)]
enum ObmcBlockState {
    /// The owning macroblock is intra-coded (§7.6.6 rule 2).
    Intra,
    /// The owning macroblock was not coded / skipped (§7.6.6 rule 1;
    /// its own OBMC prediction runs with the zero vector).
    NotCoded,
    /// An inter block with its §7.6.1.6-padded motion vector
    /// (half-sample units — OBMC is a half-sample-mode tool).
    Mv(crate::motion::MotionVector),
}

/// Expand the per-macroblock P-VOP content into the frame's 8×8-block
/// motion field (`2·mb_width × 2·mb_height`, Figure 6-8 order within
/// each macroblock).
fn obmc_block_field(entries: &[PVopMbContent], mb_width: usize) -> Vec<ObmcBlockState> {
    let mut field = vec![ObmcBlockState::NotCoded; entries.len() * 4];
    let bw = 2 * mb_width;
    for (idx, entry) in entries.iter().enumerate() {
        let mb_col = idx % mb_width;
        let mb_row = idx / mb_width;
        for b in 0..4usize {
            let bx = 2 * mb_col + (b & 1);
            let by = 2 * mb_row + (b >> 1);
            field[by * bw + bx] = match entry {
                PVopMbContent::Intra(_) => ObmcBlockState::Intra,
                // §7.7.2.1: a field-predicted neighbour's OBMC vector is
                // the Div2Round-averaged pair, "computed in the same
                // manner as the candidate predictor vectors". (The walk
                // gates the interlaced + OBMC combination off, so this
                // arm is exercised only by direct API callers.)
                PVopMbContent::FieldInter { mvs, .. } => {
                    ObmcBlockState::Mv(crate::motion::MotionVector {
                        x: crate::field_motion::div2_round(mvs.top.x + mvs.bottom.x),
                        y: crate::field_motion::div2_round(mvs.top.y + mvs.bottom.y),
                    })
                }
                PVopMbContent::Inter { motion, .. } => match motion {
                    PvopMbMotion::Skipped => ObmcBlockState::NotCoded,
                    PvopMbMotion::Intra => ObmcBlockState::Intra,
                    PvopMbMotion::OneMv(mv) => ObmcBlockState::Mv(*mv),
                    PvopMbMotion::FourMv(mvs) => ObmcBlockState::Mv(mvs[b]),
                },
            };
        }
    }
    field
}

/// Resolve one §7.6.6 remote neighbour: outside the VOP → rule 3
/// (`Absent`, current MV); an intra macroblock → rule 2; a not-coded
/// macroblock → rule 1 (zero vector); otherwise the block's vector.
fn obmc_remote_kind(
    field: &[ObmcBlockState],
    bw: usize,
    bh: usize,
    bx: isize,
    by: isize,
) -> crate::obmc::RemoteBlockKind {
    use crate::obmc::{ObmcMv, RemoteBlockKind};
    if bx < 0 || by < 0 || bx as usize >= bw || by as usize >= bh {
        return RemoteBlockKind::Absent;
    }
    match field[by as usize * bw + bx as usize] {
        ObmcBlockState::Intra => RemoteBlockKind::Intra,
        ObmcBlockState::NotCoded => RemoteBlockKind::NotCoded,
        ObmcBlockState::Mv(mv) => RemoteBlockKind::Inter(ObmcMv::new(mv.x, mv.y)),
    }
}

/// Assemble a complete P-VOP frame with §7.6.6 **overlapped motion
/// compensation** (`obmc_disable == 0`) for the luminance plane.
///
/// Every inter macroblock's four 8×8 luminance prediction blocks are
/// the three-vector weighted blends of [`crate::obmc::obmc_predict_block`]:
/// the block's own vector plus the per-pixel above-or-below and
/// left-or-right remote vectors, gathered from the frame-wide 8×8
/// motion field with the four §7.6.6 substitution rules (not-coded →
/// zero, intra → current, outside the VOP → current, below-MB → the
/// current vector). A skipped macroblock is overlapped too — its own
/// vector is zero but its neighbours' vectors still blend in.
///
/// Chrominance is never overlapped (§7.6.6 covers luminance only):
/// the §7.6.5 `MVDCHR` path of the plain assembly applies unchanged.
///
/// OBMC is a half-sample-mode tool; motion vectors are taken in
/// half-sample units.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn assemble_p_vop_frame_obmc(
    store: &FrameStore,
    mb_width: usize,
    mb_height: usize,
    entries: &[PVopMbContent],
    vop_rounding_type: u8,
    bits_per_pixel: u32,
) -> Result<DecodedFrame, FrameDecodeError> {
    use crate::half_sample::{fetch_clamped_sample, split_half_pel};
    use crate::obmc::{obmc_predict_block, ObmcConfig, ObmcMv, ObmcNeighbourhood};
    use crate::reconstruct::InterPredictionMacroblock;

    let expected = mb_width * mb_height;
    if entries.len() != expected {
        return Err(FrameDecodeError::MacroblockCountMismatch {
            supplied: entries.len(),
            expected,
        });
    }
    let reference = store
        .p_vop_reference()
        .ok_or(FrameDecodeError::MissingReference)?;
    let luma_ref = reference.luma_reference();
    let cb_ref = reference.cb_reference();
    let cr_ref = reference.cr_reference();

    let field = obmc_block_field(entries, mb_width);
    let bw = 2 * mb_width;
    let bh = 2 * mb_height;

    let mut frame = DecodedFrame::new(mb_width * 16, mb_height * 16, VopCodingType::P)?;
    for (idx, entry) in entries.iter().enumerate() {
        let mb_col = idx % mb_width;
        let mb_row = idx / mb_width;
        let reconstructed = match entry {
            PVopMbContent::Intra(mb) => mb.clone(),
            // §7.7.2.1: "the OBMC is not applied if the current MB is
            // field-predicted" — plain field motion compensation.
            PVopMbContent::FieldInter {
                mvs,
                top_field_ref,
                bottom_field_ref,
                residual,
            } => {
                let prediction = crate::field_motion::field_motion_compensate_one_reference(
                    &luma_ref,
                    &cb_ref,
                    &cr_ref,
                    *mvs,
                    *top_field_ref,
                    *bottom_field_ref,
                    (mb_col * 16) as i32,
                    (mb_row * 16) as i32,
                    vop_rounding_type,
                );
                reconstruct_inter_macroblock(&prediction, residual, bits_per_pixel)
            }
            PVopMbContent::Inter { motion, residual } => {
                if matches!(motion, PvopMbMotion::Intra) {
                    return Err(FrameDecodeError::InterPredictionFailed { mb_col, mb_row });
                }
                let mut pred = InterPredictionMacroblock::zero();
                for b in 0..4usize {
                    let bx = 2 * mb_col + (b & 1);
                    let by = 2 * mb_row + (b >> 1);
                    let mv0 = match field[by * bw + bx] {
                        ObmcBlockState::Mv(mv) => ObmcMv::new(mv.x, mv.y),
                        _ => ObmcMv::ZERO,
                    };
                    let neighbours = ObmcNeighbourhood {
                        above: obmc_remote_kind(&field, bw, bh, bx as isize, by as isize - 1),
                        below: obmc_remote_kind(&field, bw, bh, bx as isize, by as isize + 1),
                        left: obmc_remote_kind(&field, bw, bh, bx as isize - 1, by as isize),
                        right: obmc_remote_kind(&field, bw, bh, bx as isize + 1, by as isize),
                        // §7.6.6 rule 4: only the bottom row of the
                        // macroblock has its below-MB remote replaced.
                        current_block_at_mb_bottom: (b >> 1) == 1,
                    };
                    let origin_x = (bx * 8) as i32;
                    let origin_y = (by * 8) as i32;
                    let block =
                        obmc_predict_block(mv0, neighbours, ObmcConfig::enabled(), |mv, i, j| {
                            let (ix, hx) = split_half_pel(mv.x);
                            let (iy, hy) = split_half_pel(mv.y);
                            fetch_clamped_sample(
                                &luma_ref,
                                origin_x + i as i32 + ix,
                                origin_y + j as i32 + iy,
                                hx,
                                hy,
                                vop_rounding_type,
                            )
                        });
                    let col0 = 8 * (b & 1);
                    let row0 = 8 * (b >> 1);
                    for (j, row) in block.iter().enumerate() {
                        for (i, &px) in row.iter().enumerate() {
                            pred.luma[row0 + j][col0 + i] = i32::from(px);
                        }
                    }
                }
                // §7.6.6: chrominance is not overlapped — plain §7.6.5
                // MVDCHR + §7.6.2.1 bilinear interpolation.
                let cmb_x = (mb_col * 8) as i32;
                let cmb_y = (mb_row * 8) as i32;
                let mode = BVopSampleMode::HalfPel;
                let cb = crate::pvop_mv::predict_chroma_macroblock(
                    *motion,
                    &cb_ref,
                    cmb_x,
                    cmb_y,
                    vop_rounding_type,
                    mode,
                )
                .ok_or(FrameDecodeError::InterPredictionFailed { mb_col, mb_row })?;
                let cr = crate::pvop_mv::predict_chroma_macroblock(
                    *motion,
                    &cr_ref,
                    cmb_x,
                    cmb_y,
                    vop_rounding_type,
                    mode,
                )
                .ok_or(FrameDecodeError::InterPredictionFailed { mb_col, mb_row })?;
                for y in 0..8 {
                    for x in 0..8 {
                        pred.cb[y][x] = i32::from(cb[y * 8 + x]);
                        pred.cr[y][x] = i32::from(cr[y * 8 + x]);
                    }
                }
                reconstruct_inter_macroblock(&pred, residual, bits_per_pixel)
            }
        };
        frame.blit_macroblock(mb_col, mb_row, &reconstructed)?;
    }
    Ok(frame)
}

/// Decode a P-VOP and advance the reference-frame chain in one call.
///
/// This is the typical frame-decode-loop entry: assemble the P-VOP
/// ([`assemble_p_vop_frame`]) then [`FrameStore::push_anchor`] it so the
/// next VOP sees it as the new forward reference. Returns a borrow of the
/// freshly-installed anchor.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn decode_p_vop<'a>(
    store: &'a mut FrameStore,
    mb_width: usize,
    mb_height: usize,
    entries: &[PVopMbContent],
    vop_rounding_type: u8,
    sample_mode: BVopSampleMode,
    bits_per_pixel: u32,
) -> Result<&'a DecodedFrame, FrameDecodeError> {
    let frame = assemble_p_vop_frame(
        store,
        mb_width,
        mb_height,
        entries,
        vop_rounding_type,
        sample_mode,
        bits_per_pixel,
    )?;
    store.push_anchor(frame);
    // Just pushed: it is the backward (most-recent) anchor.
    Ok(store
        .backward()
        .expect("anchor just pushed must be present"))
}

/// Install a decoded I-VOP as the leading anchor of the chain.
///
/// An I-VOP is intra-only: every macroblock is supplied already
/// reconstructed to pixels. The assembled frame is pushed into the store
/// as the new anchor. Returns a borrow of the installed frame.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn decode_i_vop<'a>(
    store: &'a mut FrameStore,
    mb_width: usize,
    mb_height: usize,
    macroblocks: &[ReconstructedMacroblock],
) -> Result<&'a DecodedFrame, FrameDecodeError> {
    let expected = mb_width * mb_height;
    if macroblocks.len() != expected {
        return Err(FrameDecodeError::MacroblockCountMismatch {
            supplied: macroblocks.len(),
            expected,
        });
    }
    let mut frame = DecodedFrame::new(mb_width * 16, mb_height * 16, VopCodingType::I)?;
    for (idx, mb) in macroblocks.iter().enumerate() {
        let mb_col = idx % mb_width;
        let mb_row = idx / mb_width;
        frame.blit_macroblock(mb_col, mb_row, mb)?;
    }
    store.push_anchor(frame);
    Ok(store
        .backward()
        .expect("anchor just pushed must be present"))
}

/// Assemble a complete **progressive B-VOP** frame from per-macroblock
/// textured decodes against the bracketing forward+backward anchors in
/// `store`.
///
/// §7.6.1: a B-VOP interpolates between the forward (past) and backward
/// (future) anchors — the two I/P/S-VOPs that bracket it in display order
/// — and is **never** pushed into the reference-frame chain. The store is
/// therefore taken by shared reference; the assembled frame is returned
/// for display and discarded from the chain's point of view.
///
/// `entries` must contain exactly `mb_width * mb_height`
/// [`BVopMbTexturedDecode`]s in raster order (each carrying its §7.6.8
/// motion + §7.4 residual). Each is reconstructed via
/// [`BVopMbDecode::reconstruct`](crate::bvop_mv::BVopMbDecode::reconstruct)
/// against the six §7.6.9.5.1 anchor planes built from the chain.
///
/// Returns [`FrameDecodeError::MissingReference`] if either anchor is
/// absent (a B-VOP before two anchors have been decoded).
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn assemble_b_vop_frame(
    store: &FrameStore,
    mb_width: usize,
    mb_height: usize,
    entries: &[BVopMbTexturedDecode],
    vop_rounding_type: u8,
    sample_mode: BVopSampleMode,
    bits_per_pixel: u32,
) -> Result<DecodedFrame, FrameDecodeError> {
    let expected = mb_width * mb_height;
    if entries.len() != expected {
        return Err(FrameDecodeError::MacroblockCountMismatch {
            supplied: entries.len(),
            expected,
        });
    }
    let (forward, backward) = store
        .b_vop_references()
        .ok_or(FrameDecodeError::MissingReference)?;
    let fwd_luma = forward.luma_reference();
    let bwd_luma = backward.luma_reference();
    let fwd_cb = forward.cb_reference();
    let bwd_cb = backward.cb_reference();
    let fwd_cr = forward.cr_reference();
    let bwd_cr = backward.cr_reference();
    let anchors = BVopAnchorPlanes {
        forward_luma: &fwd_luma,
        backward_luma: &bwd_luma,
        forward_cb: &fwd_cb,
        backward_cb: &bwd_cb,
        forward_cr: &fwd_cr,
        backward_cr: &bwd_cr,
    };

    let mut frame = DecodedFrame::new(mb_width * 16, mb_height * 16, VopCodingType::B)?;
    for (idx, entry) in entries.iter().enumerate() {
        let mb_col = idx % mb_width;
        let mb_row = idx / mb_width;
        // §7.6.9.5.2: direct-mode vectors are half-pel even in a
        // quarter-sample VOL; every other mode follows the VOL grid.
        let reconstructed = entry.motion.reconstruct(
            &anchors,
            &entry.residual,
            (mb_col * 16) as i32,
            (mb_row * 16) as i32,
            vop_rounding_type,
            entry.motion.luma_sample_mode(sample_mode),
            bits_per_pixel,
        );
        frame.blit_macroblock(mb_col, mb_row, &reconstructed)?;
    }
    Ok(frame)
}

/// Assemble a complete **interlaced B-VOP** frame from per-macroblock
/// path-tagged textured decodes against the bracketing anchors in
/// `store`.
///
/// The per-entry dispatch is
/// [`BVopInterlacedTexturedDecode::reconstruct`](crate::bvop_mv::BVopInterlacedTexturedDecode::reconstruct):
/// progressive macroblocks read the six [`BVopAnchorPlanes`] with
/// `progressive_mode`, field-predicted and interlaced-direct
/// macroblocks read the same planes through the
/// [`BVopFieldReferences`](crate::bvop_field_motion::BVopFieldReferences)
/// per-field view with `field_mode`. Like its progressive sibling the
/// frame never enters the reference chain.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn assemble_b_vop_interlaced_frame(
    store: &FrameStore,
    mb_width: usize,
    mb_height: usize,
    entries: &[crate::bvop_mv::BVopInterlacedTexturedDecode],
    vop_rounding_type: u8,
    progressive_mode: BVopSampleMode,
    field_mode: crate::bvop_field_motion::FieldSampleMode,
    bits_per_pixel: u32,
) -> Result<DecodedFrame, FrameDecodeError> {
    let expected = mb_width * mb_height;
    if entries.len() != expected {
        return Err(FrameDecodeError::MacroblockCountMismatch {
            supplied: entries.len(),
            expected,
        });
    }
    let (forward, backward) = store
        .b_vop_references()
        .ok_or(FrameDecodeError::MissingReference)?;
    let fwd_luma = forward.luma_reference();
    let bwd_luma = backward.luma_reference();
    let fwd_cb = forward.cb_reference();
    let bwd_cb = backward.cb_reference();
    let fwd_cr = forward.cr_reference();
    let bwd_cr = backward.cr_reference();
    let anchors = BVopAnchorPlanes {
        forward_luma: &fwd_luma,
        backward_luma: &bwd_luma,
        forward_cb: &fwd_cb,
        backward_cb: &bwd_cb,
        forward_cr: &fwd_cr,
        backward_cr: &bwd_cr,
    };
    let field_refs = crate::bvop_field_motion::BVopFieldReferences {
        forward_luma: &fwd_luma,
        forward_cb: &fwd_cb,
        forward_cr: &fwd_cr,
        backward_luma: &bwd_luma,
        backward_cb: &bwd_cb,
        backward_cr: &bwd_cr,
    };

    let mut frame = DecodedFrame::new(mb_width * 16, mb_height * 16, VopCodingType::B)?;
    for (idx, entry) in entries.iter().enumerate() {
        let mb_col = idx % mb_width;
        let mb_row = idx / mb_width;
        let reconstructed = entry.reconstruct(
            &anchors,
            progressive_mode,
            &field_refs,
            field_mode,
            (mb_col * 16) as i32,
            (mb_row * 16) as i32,
            vop_rounding_type,
            bits_per_pixel,
        );
        frame.blit_macroblock(mb_col, mb_row, &reconstructed)?;
    }
    Ok(frame)
}

/// Per-macroblock content for an **S(GMC)-VOP** frame assembly.
///
/// §6.3.6 / §7.8.7.1: each inter macroblock of an S(GMC)-VOP carries an
/// `mcsel` flag selecting global-motion (warp) vs. local-MC prediction.
/// Intra macroblocks are supplied already reconstructed.
#[derive(Debug, Clone)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub enum SGmcMbContent {
    /// `mcsel == 1`: global-motion-compensated. The §7.8 warp is applied
    /// to the forward reference plane; the §7.4 residual is carried
    /// together with the macroblock's §7.8.7.3 **averaged motion
    /// vector** and its skip status, which the §7.6.9.6 co-located
    /// substitution for a following B-VOP consumes (a *skipped*
    /// co-located S(GMC) macroblock is treated as non-skipped with the
    /// AMV; a coded GMC macroblock keeps the §7.6.9.5.1 zero-vector
    /// fallback).
    Gmc {
        /// The §7.8.7.3 averaged pel-wise warping vector for this
        /// macroblock (VOL-grid units: half-pel, or quarter-pel when
        /// `quarter_sample == 1`).
        amv: crate::motion::MotionVector,
        /// `not_coded == 1` (skipped, implied `mcsel == 1`).
        not_coded: bool,
        /// §7.4 inter residual (zero for a skipped macroblock).
        residual: InterMacroblock,
    },
    /// `mcsel == 0`: local-motion-compensated exactly as in a P-VOP.
    Local {
        /// Decoded local motion.
        motion: PvopMbMotion,
        /// §7.4 inter residual.
        residual: InterMacroblock,
    },
    /// `mcsel == 0` with §7.7.2.1 field prediction (interlaced VOL):
    /// the P-VOP [`PVopMbContent::FieldInter`] shape.
    FieldLocal {
        /// The reconstructed top/bottom field motion vectors.
        mvs: crate::field_motion::FieldMotionVectors,
        /// `forward_top_field_reference` raw bit.
        top_field_ref: bool,
        /// `forward_bottom_field_reference` raw bit.
        bottom_field_ref: bool,
        /// §7.4 inter residual.
        residual: InterMacroblock,
    },
    /// An intra macroblock already reconstructed to pixels.
    Intra(ReconstructedMacroblock),
}

/// Assemble a complete **S(GMC)-VOP** frame from per-macroblock content
/// against the §7.8.7.1 warped / local prediction of the forward
/// reference plane in `store`.
///
/// The single reference is the §7.5.2.1.2 forward (most-recent) anchor —
/// [`FrameStore::p_vop_reference`] (a P- or S(GMC)-VOP shares this rule).
/// Each `mcsel == 1` macroblock is built by warping that plane via
/// `geometry` ([`s_gmc_prediction_macroblock`](crate::s_gmc_recon::s_gmc_prediction_macroblock)
/// GMC branch); each
/// `mcsel == 0` macroblock is built by the §7.6.2 local-MC path; each
/// intra macroblock is blitted as-is. The result is **not** pushed into
/// the chain — use [`FrameStore::push_anchor`] (an S(GMC)-VOP is an
/// anchor).
///
/// `geometry` is the §7.8.4/§7.8.5 warp the caller decoded from the
/// VOP's `sprite_trajectory()`. `sample_mode` selects the §7.6.2.1 vs
/// §7.6.2.2 luminance interpolation for the `mcsel == 0` local-MC
/// macroblocks (the GMC warp path has its own
/// `sprite_warping_accuracy` sub-pel handling).
#[allow(clippy::too_many_arguments)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn assemble_s_gmc_vop_frame(
    store: &FrameStore,
    mb_width: usize,
    mb_height: usize,
    entries: &[SGmcMbContent],
    geometry: &WarpGeometry,
    vop_rounding_type: u8,
    sample_mode: BVopSampleMode,
    bits_per_pixel: u32,
) -> Result<DecodedFrame, FrameDecodeError> {
    let expected = mb_width * mb_height;
    if entries.len() != expected {
        return Err(FrameDecodeError::MacroblockCountMismatch {
            supplied: entries.len(),
            expected,
        });
    }
    let reference = store
        .p_vop_reference()
        .ok_or(FrameDecodeError::MissingReference)?;
    let luma_ref = reference.luma_reference();
    let cb_ref = reference.cb_reference();
    let cr_ref = reference.cr_reference();
    let gmc_planes = GmcReferencePlanes {
        luma: luma_ref,
        cb: cb_ref,
        cr: cr_ref,
    };

    let mut frame = DecodedFrame::new(mb_width * 16, mb_height * 16, VopCodingType::S)?;
    for (idx, entry) in entries.iter().enumerate() {
        let mb_col = idx % mb_width;
        let mb_row = idx / mb_width;
        let mb_x = (mb_col * 16) as i32;
        let mb_y = (mb_row * 16) as i32;
        let reconstructed = match entry {
            SGmcMbContent::Intra(mb) => mb.clone(),
            SGmcMbContent::Gmc { residual, .. } => {
                let prediction = gmc_prediction_macroblock(
                    geometry,
                    &gmc_planes,
                    mb_x as i64,
                    mb_y as i64,
                    vop_rounding_type,
                    bits_per_pixel,
                );
                reconstruct_inter_macroblock(&prediction, residual, bits_per_pixel)
            }
            SGmcMbContent::Local { motion, residual } => {
                let prediction = predict_inter_macroblock(
                    *motion,
                    &gmc_planes.luma,
                    &gmc_planes.cb,
                    &gmc_planes.cr,
                    mb_x,
                    mb_y,
                    vop_rounding_type,
                    sample_mode,
                )
                .ok_or(FrameDecodeError::InterPredictionFailed { mb_col, mb_row })?;
                reconstruct_inter_macroblock(&prediction, residual, bits_per_pixel)
            }
            SGmcMbContent::FieldLocal {
                mvs,
                top_field_ref,
                bottom_field_ref,
                residual,
            } => {
                let prediction = match sample_mode {
                    BVopSampleMode::HalfPel => {
                        crate::field_motion::field_motion_compensate_one_reference(
                            &gmc_planes.luma,
                            &gmc_planes.cb,
                            &gmc_planes.cr,
                            *mvs,
                            *top_field_ref,
                            *bottom_field_ref,
                            mb_x,
                            mb_y,
                            vop_rounding_type,
                        )
                    }
                    BVopSampleMode::QuarterPel { bits_per_pixel } => {
                        crate::field_motion::field_motion_compensate_one_reference_qpel(
                            &gmc_planes.luma,
                            &gmc_planes.cb,
                            &gmc_planes.cr,
                            *mvs,
                            *top_field_ref,
                            *bottom_field_ref,
                            mb_x,
                            mb_y,
                            vop_rounding_type,
                            bits_per_pixel,
                        )
                    }
                };
                reconstruct_inter_macroblock(&prediction, residual, bits_per_pixel)
            }
        };
        frame.blit_macroblock(mb_col, mb_row, &reconstructed)?;
    }
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::MotionVector;
    use crate::reconstruct::{MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};

    fn flat_recon(luma: i32, cb: i32, cr: i32) -> ReconstructedMacroblock {
        ReconstructedMacroblock {
            luma: [[luma; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[cb; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[cr; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        }
    }

    #[test]
    fn p_vop_without_reference_errors() {
        let store = FrameStore::new();
        let entries = vec![PVopMbContent::Inter {
            motion: PvopMbMotion::Skipped,
            residual: InterMacroblock::zero(),
        }];
        let r = assemble_p_vop_frame(&store, 1, 1, &entries, 0, BVopSampleMode::HalfPel, 8);
        assert_eq!(r, Err(FrameDecodeError::MissingReference));
    }

    #[test]
    fn entry_count_must_match_grid() {
        let mut store = FrameStore::new();
        decode_i_vop(
            &mut store,
            2,
            1,
            &[flat_recon(50, 50, 50), flat_recon(60, 60, 60)],
        )
        .unwrap();
        let entries = vec![PVopMbContent::Inter {
            motion: PvopMbMotion::Skipped,
            residual: InterMacroblock::zero(),
        }];
        let r = assemble_p_vop_frame(&store, 2, 1, &entries, 0, BVopSampleMode::HalfPel, 8);
        assert_eq!(
            r,
            Err(FrameDecodeError::MacroblockCountMismatch {
                supplied: 1,
                expected: 2
            })
        );
    }

    #[test]
    fn skipped_p_vop_copies_reference_plane() {
        // I-VOP: two MBs at luma 50 / 60.
        let mut store = FrameStore::new();
        decode_i_vop(
            &mut store,
            2,
            1,
            &[flat_recon(50, 11, 22), flat_recon(60, 33, 44)],
        )
        .unwrap();

        // P-VOP: both MBs skipped (zero MV) with zero residual ->
        // pure copy of the reference plane.
        let entries = vec![
            PVopMbContent::Inter {
                motion: PvopMbMotion::Skipped,
                residual: InterMacroblock::zero(),
            },
            PVopMbContent::Inter {
                motion: PvopMbMotion::Skipped,
                residual: InterMacroblock::zero(),
            },
        ];
        let frame =
            assemble_p_vop_frame(&store, 2, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        // Left MB copies luma 50; right MB copies luma 60.
        assert_eq!(frame.luma_at(0, 0), Some(50));
        assert_eq!(frame.luma_at(15, 15), Some(50));
        assert_eq!(frame.luma_at(16, 0), Some(60));
        assert_eq!(frame.luma_at(31, 15), Some(60));
        // Chroma copied too.
        assert_eq!(frame.cb_samples()[0], 11);
        assert_eq!(frame.cr_samples()[0], 22);
    }

    #[test]
    fn intra_mb_in_p_vop_blits_verbatim() {
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 1, 1, &[flat_recon(50, 50, 50)]).unwrap();
        let entries = vec![PVopMbContent::Intra(flat_recon(200, 100, 150))];
        let frame =
            assemble_p_vop_frame(&store, 1, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(200));
        assert_eq!(frame.cb_samples()[0], 100);
        assert_eq!(frame.cr_samples()[0], 150);
    }

    #[test]
    fn p_vop_with_motion_shifts_reference() {
        // Reference: left MB luma 50, right MB luma 60, so the boundary
        // between them is at x=16. A skipped (0,0) MV copies in place; a
        // +16-pel (32 half-pel) horizontal MV on the left MB samples the
        // right MB's column instead.
        let mut store = FrameStore::new();
        decode_i_vop(
            &mut store,
            2,
            1,
            &[flat_recon(50, 11, 22), flat_recon(60, 33, 44)],
        )
        .unwrap();
        // MV in half-pel units: x = 32 (=16 full pels), y = 0.
        let mv = MotionVector { x: 32, y: 0 };
        let entries = vec![
            PVopMbContent::Inter {
                motion: PvopMbMotion::OneMv(mv),
                residual: InterMacroblock::zero(),
            },
            PVopMbContent::Inter {
                motion: PvopMbMotion::Skipped,
                residual: InterMacroblock::zero(),
            },
        ];
        let frame =
            assemble_p_vop_frame(&store, 2, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        // The left output MB now reads the reference at x in [16,32) ->
        // luma 60.
        assert_eq!(frame.luma_at(0, 0), Some(60));
        assert_eq!(frame.luma_at(15, 0), Some(60));
    }

    #[test]
    fn decode_p_vop_advances_chain() {
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 1, 1, &[flat_recon(50, 50, 50)]).unwrap();
        let entries = vec![PVopMbContent::Inter {
            motion: PvopMbMotion::Skipped,
            residual: InterMacroblock::zero(),
        }];
        decode_p_vop(&mut store, 1, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        // Chain advanced: I-VOP slid to forward, P-VOP is backward.
        assert_eq!(store.forward().unwrap().luma_at(0, 0), Some(50));
        assert_eq!(store.backward().unwrap().luma_at(0, 0), Some(50));
        assert_eq!(store.backward().unwrap().coding_type(), VopCodingType::P);
    }

    // --- B-VOP frame assembly ---

    use crate::bvop::BVopMbType;
    use crate::bvop_mv::BVopMbDecode;
    use crate::bvop_prediction::{BVopMvPair, BVopPredictionMode};

    /// A single-direction B-VOP textured decode with zero MVs and zero
    /// residual: a pure motion-compensated copy of one anchor.
    fn zero_b_mb(mode: BVopPredictionMode, mb_type: BVopMbType) -> BVopMbTexturedDecode {
        let zero = MotionVector { x: 0, y: 0 };
        BVopMbTexturedDecode {
            motion: BVopMbDecode {
                mb_type,
                prediction_mode: mode,
                cbpb: None,
                dbquant_delta: None,
                mvs: [BVopMvPair {
                    forward: zero,
                    backward: zero,
                }; 4],
                forward_chroma_mv: zero,
                backward_chroma_mv: zero,
            },
            residual: InterMacroblock::zero(),
            quantiser_scale: 8,
        }
    }

    fn two_anchor_store() -> FrameStore {
        let mut store = FrameStore::new();
        // I-VOP forward anchor: luma 40. P-VOP backward anchor: luma 80.
        decode_i_vop(&mut store, 1, 1, &[flat_recon(40, 10, 20)]).unwrap();
        let p = vec![PVopMbContent::Intra(flat_recon(80, 30, 50))];
        decode_p_vop(&mut store, 1, 1, &p, 0, BVopSampleMode::HalfPel, 8).unwrap();
        store
    }

    #[test]
    fn b_vop_without_two_anchors_errors() {
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 1, 1, &[flat_recon(40, 40, 40)]).unwrap();
        let entries = vec![zero_b_mb(
            BVopPredictionMode::ForwardOnly,
            BVopMbType::Forward,
        )];
        let r = assemble_b_vop_frame(&store, 1, 1, &entries, 0, BVopSampleMode::HalfPel, 8);
        assert_eq!(r, Err(FrameDecodeError::MissingReference));
    }

    #[test]
    fn b_vop_forward_mode_copies_forward_anchor() {
        let store = two_anchor_store();
        let entries = vec![zero_b_mb(
            BVopPredictionMode::ForwardOnly,
            BVopMbType::Forward,
        )];
        let frame =
            assemble_b_vop_frame(&store, 1, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        // Forward anchor is the I-VOP (luma 40).
        assert_eq!(frame.luma_at(0, 0), Some(40));
        assert_eq!(frame.cb_samples()[0], 10);
        // B-VOP must NOT have entered the chain.
        assert_eq!(frame.coding_type(), VopCodingType::B);
    }

    #[test]
    fn b_vop_backward_mode_copies_backward_anchor() {
        let store = two_anchor_store();
        let entries = vec![zero_b_mb(
            BVopPredictionMode::BackwardOnly,
            BVopMbType::Backward,
        )];
        let frame =
            assemble_b_vop_frame(&store, 1, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        // Backward anchor is the P-VOP (luma 80).
        assert_eq!(frame.luma_at(0, 0), Some(80));
        assert_eq!(frame.cb_samples()[0], 30);
    }

    #[test]
    fn b_vop_bidirectional_averages_anchors() {
        let store = two_anchor_store();
        let entries = vec![zero_b_mb(
            BVopPredictionMode::Bidirectional,
            BVopMbType::Interpolated,
        )];
        let frame =
            assemble_b_vop_frame(&store, 1, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        // §7.6.9.4 average of forward (40) and backward (80): (40+80+1)>>1
        // = 60.
        assert_eq!(frame.luma_at(0, 0), Some(60));
        // Chroma average of Cb (10, 30) = 20.
        assert_eq!(frame.cb_samples()[0], 20);
    }

    #[test]
    fn b_vop_does_not_advance_chain() {
        let store = two_anchor_store();
        let entries = vec![zero_b_mb(
            BVopPredictionMode::ForwardOnly,
            BVopMbType::Forward,
        )];
        let _ =
            assemble_b_vop_frame(&store, 1, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        // Chain unchanged: forward=I(40), backward=P(80).
        assert_eq!(store.forward().unwrap().luma_at(0, 0), Some(40));
        assert_eq!(store.backward().unwrap().luma_at(0, 0), Some(80));
    }

    // --- §7.6.6 OBMC P-VOP frame assembly ---

    /// A reference frame whose luma is a horizontal ramp (`luma(x, y)
    /// = 2x`), chroma flat.
    fn ramp_store(mb_width: usize) -> FrameStore {
        let mut store = FrameStore::new();
        let mbs: Vec<ReconstructedMacroblock> = (0..mb_width)
            .map(|mb| {
                let mut m = flat_recon(0, 90, 91);
                for (y, row) in m.luma.iter_mut().enumerate() {
                    for (x, px) in row.iter_mut().enumerate() {
                        let _ = y;
                        *px = 2 * (mb * 16 + x) as i32;
                    }
                }
                m
            })
            .collect();
        decode_i_vop(&mut store, mb_width, 1, &mbs).unwrap();
        store
    }

    #[test]
    fn obmc_uniform_motion_field_matches_plain_assembly() {
        // When every block shares one motion vector, all three §7.6.6
        // fetches see the same sample and (q+r+s weighted, weights sum
        // 8) reproduce the plain motion-compensated prediction.
        let store = ramp_store(2);
        let mv = MotionVector { x: 2, y: 0 }; // +1 full pel
        let entries = vec![
            PVopMbContent::Inter {
                motion: PvopMbMotion::OneMv(mv),
                residual: InterMacroblock::zero(),
            },
            PVopMbContent::Inter {
                motion: PvopMbMotion::OneMv(mv),
                residual: InterMacroblock::zero(),
            },
        ];
        let plain =
            assemble_p_vop_frame(&store, 2, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        let obmc = assemble_p_vop_frame_obmc(&store, 2, 1, &entries, 0, 8).unwrap();
        assert_eq!(plain.luma_samples(), obmc.luma_samples());
        assert_eq!(plain.cb_samples(), obmc.cb_samples());
    }

    #[test]
    fn obmc_blends_neighbouring_block_vector() {
        // Left MB at rest, right MB displaced +4 full pels on a
        // luma(x) = 2x ramp. In the left MB's block 1 (its right 8x8
        // block), right-half pixels blend the right neighbour's
        // vector: p = (q*H0 + q*H1 + s*H2 + 4)/8 with s = q + 8
        // (above remote is outside the VOP → rule 3 → current).
        let store = ramp_store(2);
        let entries = vec![
            PVopMbContent::Inter {
                motion: PvopMbMotion::OneMv(MotionVector { x: 0, y: 0 }),
                residual: InterMacroblock::zero(),
            },
            PVopMbContent::Inter {
                motion: PvopMbMotion::OneMv(MotionVector { x: 8, y: 0 }),
                residual: InterMacroblock::zero(),
            },
        ];
        let frame = assemble_p_vop_frame_obmc(&store, 2, 1, &entries, 0, 8).unwrap();
        use crate::obmc::{OBMC_H0, OBMC_H1, OBMC_H2};
        // Pixel (x = 15, y = 3): block 1 of MB 0, in-block (i = 7,
        // j = 3). q = r = 2*15 = 30 (current MV zero; above remote is
        // outside → current). s = ramp at x+4 = 2*19 = 38 (right
        // remote (8,0) = +4 pels).
        let (i, j) = (7usize, 3usize);
        let q = 30u32;
        let sref = 38u32;
        let expect = (q * u32::from(OBMC_H0[j][i])
            + q * u32::from(OBMC_H1[j][i])
            + sref * u32::from(OBMC_H2[j][i])
            + 4)
            / 8;
        assert_eq!(frame.luma_at(15, 3), Some(expect as u8));
        // Left-half pixels of the same block take the in-MB left
        // neighbour (same zero MV) → plain value 30 survives only if
        // every fetch agrees; verify no bleed at (x = 8, y = 3).
        assert_eq!(frame.luma_at(8, 3), Some(16));
    }

    #[test]
    fn obmc_skipped_macroblock_still_overlaps() {
        // A skipped MB owns the zero vector but §7.6.6 still blends
        // its inter neighbour's vector into the boundary pixels.
        let store = ramp_store(2);
        let entries = vec![
            PVopMbContent::Inter {
                motion: PvopMbMotion::Skipped,
                residual: InterMacroblock::zero(),
            },
            PVopMbContent::Inter {
                motion: PvopMbMotion::OneMv(MotionVector { x: 8, y: 0 }),
                residual: InterMacroblock::zero(),
            },
        ];
        let plain =
            assemble_p_vop_frame(&store, 2, 1, &entries, 0, BVopSampleMode::HalfPel, 8).unwrap();
        let obmc = assemble_p_vop_frame_obmc(&store, 2, 1, &entries, 0, 8).unwrap();
        // §7.6.6 rule 1 in the *other* direction: the skipped MB is a
        // not-coded neighbour of the moving MB, whose left-half pixels
        // take the zero vector — and the skipped MB's own right edge
        // blends the moving MB's vector. Either way the frames differ
        // near the boundary but agree far from it.
        assert_ne!(plain.luma_samples(), obmc.luma_samples());
        assert_eq!(plain.luma_at(0, 8), obmc.luma_at(0, 8));
    }

    // --- S(GMC)-VOP frame assembly ---

    use crate::sprite::SpriteTrajectory;
    use crate::vol::SpriteWarpingAccuracy;

    fn stationary_geometry(w: u32, h: u32) -> WarpGeometry {
        let traj = SpriteTrajectory::stationary();
        WarpGeometry::decode(&traj, w, h, SpriteWarpingAccuracy::HalfPel)
    }

    #[test]
    fn s_gmc_without_reference_errors() {
        let store = FrameStore::new();
        let geo = stationary_geometry(16, 16);
        let entries = vec![SGmcMbContent::Gmc {
            amv: crate::motion::MotionVector { x: 0, y: 0 },
            not_coded: false,
            residual: InterMacroblock::zero(),
        }];
        let r =
            assemble_s_gmc_vop_frame(&store, 1, 1, &entries, &geo, 0, BVopSampleMode::HalfPel, 8);
        assert_eq!(r, Err(FrameDecodeError::MissingReference));
    }

    #[test]
    fn s_gmc_stationary_warp_copies_reference() {
        // Reference: flat luma 70. A stationary GMC warp samples each
        // output pel from the identical reference position -> copy.
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 1, 1, &[flat_recon(70, 21, 31)]).unwrap();
        let geo = stationary_geometry(16, 16);
        let entries = vec![SGmcMbContent::Gmc {
            amv: crate::motion::MotionVector { x: 0, y: 0 },
            not_coded: false,
            residual: InterMacroblock::zero(),
        }];
        let frame =
            assemble_s_gmc_vop_frame(&store, 1, 1, &entries, &geo, 0, BVopSampleMode::HalfPel, 8)
                .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(70));
        assert_eq!(frame.luma_at(15, 15), Some(70));
        assert_eq!(frame.cb_samples()[0], 21);
        assert_eq!(frame.cr_samples()[0], 31);
        assert_eq!(frame.coding_type(), VopCodingType::S);
    }

    #[test]
    fn s_gmc_local_mcsel0_uses_p_vop_path() {
        // mcsel == 0 macroblock predicts exactly as a P-VOP. Two-MB
        // reference (50 | 60); a +16-pel local MV on the left MB samples
        // the right reference MB, matching the P-VOP behaviour.
        let mut store = FrameStore::new();
        decode_i_vop(
            &mut store,
            2,
            1,
            &[flat_recon(50, 11, 22), flat_recon(60, 33, 44)],
        )
        .unwrap();
        let geo = stationary_geometry(32, 16);
        let mv = MotionVector { x: 32, y: 0 };
        let entries = vec![
            SGmcMbContent::Local {
                motion: PvopMbMotion::OneMv(mv),
                residual: InterMacroblock::zero(),
            },
            SGmcMbContent::Local {
                motion: PvopMbMotion::Skipped,
                residual: InterMacroblock::zero(),
            },
        ];
        let frame =
            assemble_s_gmc_vop_frame(&store, 2, 1, &entries, &geo, 0, BVopSampleMode::HalfPel, 8)
                .unwrap();
        // Left MB read the right reference MB (luma 60).
        assert_eq!(frame.luma_at(0, 0), Some(60));
        // Right MB skipped -> copy of reference MB (luma 60).
        assert_eq!(frame.luma_at(16, 0), Some(60));
    }

    #[test]
    fn s_gmc_intra_blits_verbatim() {
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 1, 1, &[flat_recon(70, 70, 70)]).unwrap();
        let geo = stationary_geometry(16, 16);
        let entries = vec![SGmcMbContent::Intra(flat_recon(123, 45, 67))];
        let frame =
            assemble_s_gmc_vop_frame(&store, 1, 1, &entries, &geo, 0, BVopSampleMode::HalfPel, 8)
                .unwrap();
        assert_eq!(frame.luma_at(0, 0), Some(123));
        assert_eq!(frame.cb_samples()[0], 45);
        assert_eq!(frame.cr_samples()[0], 67);
    }

    #[test]
    fn s_gmc_vop_pushes_as_anchor() {
        // An S(GMC)-VOP is an anchor: pushing it advances the chain.
        let mut store = FrameStore::new();
        decode_i_vop(&mut store, 1, 1, &[flat_recon(70, 70, 70)]).unwrap();
        let geo = stationary_geometry(16, 16);
        let entries = vec![SGmcMbContent::Gmc {
            amv: crate::motion::MotionVector { x: 0, y: 0 },
            not_coded: false,
            residual: InterMacroblock::zero(),
        }];
        let frame =
            assemble_s_gmc_vop_frame(&store, 1, 1, &entries, &geo, 0, BVopSampleMode::HalfPel, 8)
                .unwrap();
        store.push_anchor(frame);
        assert_eq!(store.backward().unwrap().coding_type(), VopCodingType::S);
        assert_eq!(store.forward().unwrap().coding_type(), VopCodingType::I);
    }
}

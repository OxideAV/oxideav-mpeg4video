//! §7.6.8 frame-level B-VOP motion-vector decode driver.
//!
//! This is the B-VOP analogue of the P-VOP [`MvDriver`](crate::pvop_mv::MvDriver):
//! it walks the macroblocks of one **progressive, non-scalable** B-VOP
//! in raster order, decoding each one's §6.2.6 header + motion-vector
//! bodies, resolving the §7.6.9 prediction mode (direct / forward /
//! backward / bidirectional), reconstructing the forward / backward
//! motion vectors against the §7.6.8 running predictor bank, and emitting
//! a [`BVopMbDecode`] ready to feed the §7.6.9 →§7.3 reconstruction
//! bridge ([`predict_b_vop_macroblock`](crate::bvop_prediction::predict_b_vop_macroblock)).
//!
//! ## §7.6.8 predictor threading
//!
//! Unlike a P-VOP — whose predictor is the §7.6.5 spatial median of the
//! three neighbouring block vectors — a progressive B-VOP carries a
//! single running predictor *per direction*:
//!
//! > "the forward and backward vectors have their own vector predictors,
//! > which are reset to zero only at the beginning of each macroblock
//! > row. The vector predictors are updated in the following three
//! > cases:
//! >  * after decoding a macroblock of forward mode only the forward
//! >    predictor is set to the decoded forward vector
//! >  * after decoding a macroblock of backward mode only the backward
//! >    predictor is set to the decoded backward vector
//! >  * after decoding a macroblock of bi-directional mode both the
//! >    forward and backward predictors are updated separately with the
//! >    decoded vectors of the same type (forward/backward)."
//!
//! Direct-mode macroblocks neither read nor update these predictors:
//! their delta vector `MVD` is decoded with a predictor of zero and an
//! `f_code` of one (§7.6.8 first paragraph), and §7.6.9.5.2 then scales
//! the co-located anchor MV to produce `MVF` / `MVB`.
//!
//! ## §6.2.6 `modb` discriminator
//!
//! The §6.2.6 `modb` codeword distinguishes `"1"` (no `mb_type`, no
//! `cbpb`; the macroblock takes the default type and codes no motion
//! bodies) from `"01"` (an explicit `mb_type` follows). Both store the
//! numeric value `1` in some encodings — this driver distinguishes them
//! via the `mb_type_present` flag returned by
//! [`parse_b_vop_mb_header`], surfaced as
//! [`BVopMbHeader::mb_type_present`].
//!
//! ## §7.6.9.6 skipped-macroblock handling
//!
//! When the co-located macroblock in the most recently decoded anchor
//! (I- / P-VOP) is *skipped* (`not_coded == 1`), §7.6.9.6 forces the
//! current B-macroblock:
//!
//! * if `modb == "1"` → direct mode with a **zero** delta vector;
//! * otherwise → forward mode with the **zero** motion vector.
//!
//! The caller supplies the co-located-skipped flag to
//! [`BVopMvDriver::decode_macroblock`]; the driver applies these
//! overrides before consuming any `mb_type` / motion bits.
//!
//! ## Clean-room provenance
//!
//! Truth taken from `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`
//! §6.2.6 (`macroblock()` B-VOP branch), §7.6.8 (vector decoding
//! process of non-scalable progressive B-VOPs), §7.6.9 (motion
//! compensation), §7.6.9.5.1/.2 (direct mode), §7.6.9.6 (skipped MBs).
//! No external MPEG-4 implementation consulted.

use crate::bitreader::BitReader;
use crate::block::{decode_b_vop_inter_macroblock, InterMacroblock, MacroblockTextureContext};
use crate::bvop::{
    decode_b_vop_mb_motion_vectors, parse_b_vop_mb_header, BMbTypeTable, BVopMbHeader,
    BVopMbParseError, BVopMbType, BVopMotionVectors, BVopMvBody,
};
use crate::bvop_field_direct::{
    interlaced_direct_mvs, interlaced_direct_prediction, ColocatedFutureField, InterlacedDirectMvs,
};
use crate::bvop_field_motion::{
    field_backward_prediction, field_bidirectional_prediction, field_forward_prediction,
    BVopFieldReferences, FieldMvDeltas, FieldReferenceFlags, FieldSampleMode,
};
use crate::bvop_field_predictor::FieldPmvBank;
use crate::bvop_prediction::{BVopMvPair, BVopPredictionMode, MB_SUB_BLOCKS};
use crate::chroma_mv::{chroma_mv_from_luma_blocks, ChromaMvError};
use crate::interlaced_information::FieldReference;
use crate::motion::{
    direct_mode_motion_vector, reconstruct_motion_vector, DirectCoLocatedMv, DirectModeMv,
    DirectMvError, DirectMvUnits, MotionVector, MotionVectorDelta,
};
use crate::vol::VolHeader;
use crate::vop::VopCodingType;

/// Errors surfaced while driving one B-VOP macroblock through the
/// §7.6.8 decode pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BVopMvDriverError {
    /// A §6.2.6 header / motion-vector body failed to decode.
    Parse(BVopMbParseError),
    /// The §7.6.9.5.2 direct-mode derivation rejected its temporal
    /// references (TRB / TRD out of range).
    Direct(DirectMvError),
    /// The §7.6.5 chroma-MV reduction failed.
    Chroma(ChromaMvError),
    /// `decode_macroblock` was called with a raster position outside the
    /// driver's `mb_rows × mb_cols` grid.
    OutOfBounds {
        /// The requested macroblock row.
        mb_row: usize,
        /// The requested macroblock column.
        mb_col: usize,
    },
    /// The interlaced field-prediction path is not yet handled by this
    /// frame driver. Field-predicted B-VOP macroblocks decode their
    /// motion bodies into [`BVopMvBody::Field`] pairs; that path is
    /// threaded through [`crate::bvop_field_predictor`] in a later
    /// round, not this progressive frame walk.
    FieldPredictionUnsupported,
    /// The §6.2.6 / §7.4 texture (residual) decode for this macroblock
    /// failed (e.g. a malformed Tcoef EVENT).
    Texture(crate::block::BlockAssemblyError),
}

impl core::fmt::Display for BVopMvDriverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BVopMvDriverError::Parse(e) => write!(f, "b-vop macroblock decode failed: {e}"),
            BVopMvDriverError::Direct(e) => {
                write!(f, "b-vop direct-mode mv derivation failed: {e}")
            }
            BVopMvDriverError::Chroma(e) => {
                write!(f, "b-vop chroma-mv reduction failed: {e}")
            }
            BVopMvDriverError::OutOfBounds { mb_row, mb_col } => write!(
                f,
                "b-vop macroblock ({mb_row}, {mb_col}) is outside the VOP grid"
            ),
            BVopMvDriverError::FieldPredictionUnsupported => write!(
                f,
                "b-vop field-prediction is not handled by the progressive frame driver"
            ),
            BVopMvDriverError::Texture(e) => {
                write!(f, "b-vop residual texture decode failed: {e}")
            }
        }
    }
}

impl From<crate::block::BlockAssemblyError> for BVopMvDriverError {
    fn from(e: crate::block::BlockAssemblyError) -> Self {
        BVopMvDriverError::Texture(e)
    }
}

impl std::error::Error for BVopMvDriverError {}

impl From<BVopMbParseError> for BVopMvDriverError {
    fn from(e: BVopMbParseError) -> Self {
        BVopMvDriverError::Parse(e)
    }
}

impl From<DirectMvError> for BVopMvDriverError {
    fn from(e: DirectMvError) -> Self {
        BVopMvDriverError::Direct(e)
    }
}

impl From<ChromaMvError> for BVopMvDriverError {
    fn from(e: ChromaMvError) -> Self {
        BVopMvDriverError::Chroma(e)
    }
}

/// The fully resolved motion state of one decoded B-VOP macroblock,
/// ready as the §7.6.9 input to
/// [`predict_b_vop_macroblock`](crate::bvop_prediction::predict_b_vop_macroblock).
///
/// `prediction_mode` selects which §7.6.9 sub-clause the prediction
/// generator runs; `mvs` carries the four per-sub-block `(MVF, MVB)`
/// pairs (only `mvs[0]` is read for the non-direct modes — the spec
/// replicates the single MB vector across all four sub-blocks);
/// `forward_chroma_mv` / `backward_chroma_mv` are the §7.6.5-reduced
/// chroma vectors the generator consumes directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BVopMbDecode {
    /// The decoded §6.2.6 macroblock type.
    pub mb_type: BVopMbType,
    /// The resolved §7.6.9 prediction mode.
    pub prediction_mode: BVopPredictionMode,
    /// `cbpb` (coded-block pattern for B-VOP) — `None` when `modb`
    /// carried no `cbpb` field (no residual is coded; the MB is a pure
    /// motion-compensated copy / average).
    pub cbpb: Option<u8>,
    /// `dbquant` delta applied to the running quantiser (§6.3.6), if
    /// present.
    pub dbquant_delta: Option<i8>,
    /// The four per-sub-block `(MVF, MVB)` pairs.
    pub mvs: [BVopMvPair; MB_SUB_BLOCKS],
    /// The §7.6.5-reduced forward chroma motion vector (half-sample
    /// units).
    pub forward_chroma_mv: MotionVector,
    /// The §7.6.5-reduced backward chroma motion vector (half-sample
    /// units).
    pub backward_chroma_mv: MotionVector,
}

/// The six §7.6.9.5.1-padded anchor reference planes (forward + backward
/// luma / Cb / Cr) a B-VOP macroblock reconstruction reads from.
///
/// Grouped into one struct so the [`BVopMbDecode::reconstruct`] bridge
/// avoids a 16-argument signature. `forward_*` are the previous anchor
/// VOP's planes, `backward_*` the temporally next anchor's.
#[derive(Debug, Clone, Copy)]
pub struct BVopAnchorPlanes<'a> {
    /// Forward (previous anchor) luma plane.
    pub forward_luma: &'a crate::half_sample::ReferenceVop<'a>,
    /// Backward (next anchor) luma plane.
    pub backward_luma: &'a crate::half_sample::ReferenceVop<'a>,
    /// Forward Cb plane.
    pub forward_cb: &'a crate::half_sample::ReferenceVop<'a>,
    /// Backward Cb plane.
    pub backward_cb: &'a crate::half_sample::ReferenceVop<'a>,
    /// Forward Cr plane.
    pub forward_cr: &'a crate::half_sample::ReferenceVop<'a>,
    /// Backward Cr plane.
    pub backward_cr: &'a crate::half_sample::ReferenceVop<'a>,
}

impl BVopMbDecode {
    /// Fully reconstruct this macroblock end-to-end: §7.6.9
    /// motion-compensated prediction (forward / backward / bidirectional
    /// / direct, dispatched on [`Self::prediction_mode`]) plus the §7.3
    /// step-2 add of the §7.4 residual and the step-3 display clip.
    ///
    /// This is the §7.6.9 → §7.3 bridge: it routes the decoded motion
    /// state ([`Self::mvs`], [`Self::forward_chroma_mv`],
    /// [`Self::backward_chroma_mv`], [`Self::prediction_mode`]) into
    /// [`reconstruct_b_vop_macroblock`](crate::bvop_prediction::reconstruct_b_vop_macroblock).
    ///
    /// * `anchors` — the six §7.6.9.5.1-padded reference planes.
    /// * `residual` — the already-decoded §7.4 inter residual macroblock
    ///   for this MB (`cbpb == None` / all-zero blocks → a zero residual,
    ///   yielding a pure motion-compensated copy / average).
    /// * `mb_origin_x` / `mb_origin_y` — top-left **luma** pixel position
    ///   in the current B-VOP (`16 * mb_col`, `16 * mb_row`).
    /// * `vop_rounding_type` — §7.6.7 half-pel rounding control.
    /// * `mode` — half-pel vs quarter-pel luma interpolation.
    /// * `bits_per_pixel` — §6.3.3 sample depth for the §7.3 step-3 clip.
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct(
        &self,
        anchors: &BVopAnchorPlanes<'_>,
        residual: &crate::block::InterMacroblock,
        mb_origin_x: i32,
        mb_origin_y: i32,
        vop_rounding_type: u8,
        mode: crate::bvop_prediction::BVopSampleMode,
        bits_per_pixel: u32,
    ) -> crate::reconstruct::ReconstructedMacroblock {
        crate::bvop_prediction::reconstruct_b_vop_macroblock(
            anchors.forward_luma,
            anchors.backward_luma,
            anchors.forward_cb,
            anchors.backward_cb,
            anchors.forward_cr,
            anchors.backward_cr,
            &self.mvs,
            self.forward_chroma_mv,
            self.backward_chroma_mv,
            residual,
            mb_origin_x,
            mb_origin_y,
            vop_rounding_type,
            mode,
            self.prediction_mode,
            bits_per_pixel,
        )
    }
}

/// The §7.6.9.5.1 / §7.6.9.6 co-located anchor state for one B-VOP
/// macroblock, supplied by the caller to
/// [`BVopMvDriver::decode_vop_motion`].
///
/// `skipped` is the §7.6.9.6 flag (the co-located macroblock in the
/// most recently decoded anchor was `not_coded`); `mv` is the
/// §7.6.9.5.1 co-located block vector (after §7.6.1.6 vector padding)
/// consulted by direct mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoLocatedAnchor {
    /// §7.6.9.6: whether the co-located anchor macroblock was skipped.
    pub skipped: bool,
    /// §7.6.9.5.1: the co-located anchor block vector (or the
    /// transparent / unavailable fallback).
    pub mv: DirectCoLocatedMv,
}

impl Default for CoLocatedAnchor {
    /// A non-skipped, transparent/absent co-located macroblock — the
    /// safe default when the caller has no anchor MV grid (direct mode
    /// then derives `MVF` / `MVB` from a zero `MV` per §7.6.9.5.1's
    /// final sentence).
    fn default() -> Self {
        Self {
            skipped: false,
            mv: DirectCoLocatedMv::TransparentOrAbsent,
        }
    }
}

/// §7.6.8 running motion-vector predictor pair for one direction.
///
/// Reset to `(0, 0)` at the start of every macroblock row; updated only
/// after a macroblock that decoded a vector in this direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectionPredictor {
    forward: MotionVector,
    backward: MotionVector,
}

impl Default for DirectionPredictor {
    fn default() -> Self {
        Self {
            forward: MotionVector { x: 0, y: 0 },
            backward: MotionVector { x: 0, y: 0 },
        }
    }
}

/// Per-VOP texture parameters the residual-threaded B-VOP frame loop
/// ([`BVopMvDriver::decode_vop`]) needs in addition to the motion state.
///
/// `base_quantiser_scale` is the B-VOP's `vop_quant` (§6.3.5); the
/// driver applies each macroblock's `dbquant` delta to a running copy of
/// it (§6.3.6, Table 6-33) and clips to `[1, max_quantiser_scale]`
/// (`max_quantiser_scale == 2^quant_precision - 1`, default 31).
/// `bits_per_pixel` and `quant_type` come from the §6.3.3 / §6.3.2 VOL
/// header and map straight onto [`MacroblockTextureContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BVopTextureParams {
    /// §6.3.5 `vop_quant` — the B-VOP base quantiser scale.
    pub base_quantiser_scale: u32,
    /// `2^quant_precision - 1` — the §6.3.3 upper clip for the running
    /// quantiser after each `dbquant` (31 for the default 5-bit
    /// precision).
    pub max_quantiser_scale: u32,
    /// §6.3.3 sample depth used by the §7.4.5 IDCT saturation.
    pub bits_per_pixel: u32,
    /// §6.3.2 `quant_type` — `true` selects §7.4.4.1 method 1 (with a
    /// quantisation matrix), `false` selects §7.4.4.2 method 2.
    pub quant_type: bool,
}

/// One fully-decoded progressive B-VOP macroblock: the §7.6.8 motion
/// state plus its §7.4 inter residual, with the running quantiser scale
/// that decoded the residual.
///
/// Produced by [`BVopMvDriver::decode_vop`]; ready to feed
/// [`BVopMbDecode::reconstruct`] (the residual is in `residual`).
#[derive(Debug, Clone)]
pub struct BVopMbTexturedDecode {
    /// The §7.6.8 motion state (type, prediction mode, MVs, chroma MVs).
    pub motion: BVopMbDecode,
    /// The §7.4 inter residual for this macroblock (wholly zero when the
    /// macroblock coded no `cbpb`).
    pub residual: InterMacroblock,
    /// The §6.3.6 running quantiser scale used to dequantise this
    /// macroblock's residual (after applying its `dbquant`).
    pub quantiser_scale: u32,
}

/// The §7.7.2.2 prediction mode of an interlaced field-predicted B-VOP
/// macroblock, carrying its already-bank-updated motion state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BVopFieldMode {
    /// Field forward (`mb_type == "0001"`): forward top/bottom field
    /// deltas, compensated from the forward (past) anchor.
    Forward(FieldMvDeltas),
    /// Field backward (`mb_type == "001"`): backward top/bottom field
    /// deltas, compensated from the backward (future) anchor.
    Backward(FieldMvDeltas),
    /// Field bidirectional (`mb_type == "01"`): the four differentials in
    /// bitstream order `MVD[0..4]` (top-fwd, bot-fwd, top-bwd, bot-bwd),
    /// forward + backward compensated and averaged.
    Bidirectional([MotionVector; 4]),
}

/// One fully-decoded interlaced **field-predicted** B-VOP macroblock
/// (§7.7.2.2 forward / backward / bidirectional). Produced by
/// [`BVopMvDriver::decode_field_macroblock`]; reconstructed to pixels via
/// [`BVopFieldMbDecode::reconstruct`].
///
/// The motion bodies have already been added to the driver's running
/// four-PMV bank, so `mode` carries the decoded **differentials**; the
/// reconstruction re-runs the bank-equivalent field MC against the
/// supplied reference planes. (The bank state at decode time is captured
/// so reconstruction is independent of subsequent macroblocks.)
#[derive(Debug, Clone)]
pub struct BVopFieldMbDecode {
    /// The §7.7.2.2 field prediction mode + decoded differentials.
    pub mode: BVopFieldMode,
    /// The four §6.3.6.3 `*_field_reference` flags.
    pub references: FieldReferenceFlags,
    /// `cbpb` (coded-block pattern) — `None` when no residual is coded.
    pub cbpb: Option<u8>,
    /// `dbquant` delta applied to the running quantiser (§6.3.6).
    pub dbquant_delta: Option<i8>,
    /// The four-PMV bank snapshot **before** this macroblock's update,
    /// so [`BVopFieldMbDecode::reconstruct`] reproduces the exact field
    /// MVs the driver decoded.
    pub bank_before: FieldPmvBank,
}

impl BVopFieldMbDecode {
    /// Reconstruct this field-predicted macroblock end-to-end: §7.7.2.2
    /// field motion-compensated prediction + §7.3 step-2 residual add +
    /// step-3 display clip.
    ///
    /// * `references` — the six forward/backward reference planes.
    /// * `residual` — the already-decoded §7.4 inter residual.
    /// * `mb_origin_x` / `mb_origin_y` — top-left luma pixel position.
    /// * `vop_rounding_type` — §7.6.7 half-pel rounding control.
    /// * `sample_mode` — half- vs quarter-sample luma interpolation.
    /// * `bits_per_pixel` — §6.3.3 sample depth for the §7.3 clip.
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct(
        &self,
        references: &BVopFieldReferences<'_>,
        residual: &InterMacroblock,
        mb_origin_x: i32,
        mb_origin_y: i32,
        vop_rounding_type: u8,
        sample_mode: FieldSampleMode,
        bits_per_pixel: u32,
    ) -> crate::reconstruct::ReconstructedMacroblock {
        let mut bank = self.bank_before;
        let prediction = match self.mode {
            BVopFieldMode::Forward(deltas) => field_forward_prediction(
                &mut bank,
                deltas,
                references,
                self.references,
                mb_origin_x,
                mb_origin_y,
                vop_rounding_type,
                sample_mode,
            ),
            BVopFieldMode::Backward(deltas) => field_backward_prediction(
                &mut bank,
                deltas,
                references,
                self.references,
                mb_origin_x,
                mb_origin_y,
                vop_rounding_type,
                sample_mode,
            ),
            BVopFieldMode::Bidirectional(mvd) => field_bidirectional_prediction(
                &mut bank,
                mvd,
                references,
                self.references,
                mb_origin_x,
                mb_origin_y,
                vop_rounding_type,
                sample_mode,
            ),
        };
        crate::reconstruct::reconstruct_inter_macroblock(&prediction, residual, bits_per_pixel)
    }
}

/// The §7.7.2.2 co-located **future** P-VOP macroblock's two forward
/// field motion vectors (`MV[0]` top, `MV[1]` bottom) and their per-field
/// reference selections, supplied to
/// [`BVopMvDriver::decode_interlaced_direct_macroblock`].
///
/// These come from the reference-frame chain: the macroblock at the same
/// coordinates in the temporally next (future) anchor must itself be
/// field-predicted (`field_prediction == 1`) for interlaced direct mode
/// to apply (§7.7.2.2; otherwise progressive direct mode is used). The
/// driver derives the four direct field MVs from these plus the single
/// transmitted `MVD[0]` and the frame-period `TRB` / `TRD` it already
/// owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColocatedFutureFieldMvs {
    /// `MV[0]` — the future macroblock's top-field forward MV + reference
    /// field.
    pub top: ColocatedFutureField,
    /// `MV[1]` — the future macroblock's bottom-field forward MV +
    /// reference field.
    pub bottom: ColocatedFutureField,
}

impl ColocatedFutureFieldMvs {
    /// Build the §7.7.2.2 co-located future field MVs from a decoded
    /// interlaced P-VOP macroblock's reconstructed forward field motion
    /// vectors ([`crate::field_motion::FieldMotionVectors`]) and that
    /// macroblock's `top_field_reference` / `bottom_field_reference`
    /// flags (§6.3.6.3).
    ///
    /// This is the reference-frame-chain bridge: when a B-VOP's co-located
    /// future P-VOP macroblock was field-predicted, its two forward field
    /// MVs (`MVx f1 / MVy f1`, `MVx f2 / MVy f2`) and the reference fields
    /// it selected drive interlaced direct mode. Feed the result straight
    /// into [`BVopMvDriver::decode_interlaced_direct_macroblock`].
    pub fn from_field_motion(
        field_mvs: crate::field_motion::FieldMotionVectors,
        top_field_reference: FieldReference,
        bottom_field_reference: FieldReference,
    ) -> Self {
        Self {
            top: ColocatedFutureField {
                mv: field_mvs.top,
                reference_field: top_field_reference,
            },
            bottom: ColocatedFutureField {
                mv: field_mvs.bottom,
                reference_field: bottom_field_reference,
            },
        }
    }
}

/// One fully-decoded interlaced **direct** B-VOP macroblock (§7.7.2.2
/// last pseudo-code block). Produced by
/// [`BVopMvDriver::decode_interlaced_direct_macroblock`]; reconstructed to
/// pixels via [`BVopInterlacedDirectMbDecode::reconstruct`].
///
/// Unlike the three non-direct field modes, interlaced direct mode does
/// not touch the §7.7.2.2 four-PMV bank — its four field MVs are derived
/// purely from the co-located future field MVs, the single `MVD[0]`, and
/// the field-period temporal references (§7.7.2.2). The derived MVs and
/// the future macroblock's forward reference fields are captured here so
/// reconstruction is self-contained.
#[derive(Debug, Clone, Copy)]
pub struct BVopInterlacedDirectMbDecode {
    /// The four derived field MVs (`mvf[0..2]` forward, `mvb[0..2]`
    /// backward).
    pub mvs: InterlacedDirectMvs,
    /// The future macroblock's `colocated_future_mb_top_field_reference`
    /// (the forward MC's top reference field).
    pub forward_top_ref: FieldReference,
    /// The future macroblock's
    /// `colocated_future_mb_bottom_field_reference` (the forward MC's
    /// bottom reference field).
    pub forward_bottom_ref: FieldReference,
    /// `cbpb` (coded-block pattern) — `None` when no residual is coded.
    pub cbpb: Option<u8>,
    /// `dbquant` delta applied to the running quantiser (§6.3.6).
    pub dbquant_delta: Option<i8>,
}

impl BVopInterlacedDirectMbDecode {
    /// Reconstruct this interlaced-direct macroblock end-to-end: §7.7.2.2
    /// forward + backward field MC, the `(fwd + bak + 1) >> 1` average,
    /// the §7.3 step-2 residual add, and the step-3 display clip.
    ///
    /// * `references` — the six forward/backward reference planes.
    /// * `residual` — the already-decoded §7.4 inter residual.
    /// * `mb_origin_x` / `mb_origin_y` — top-left luma pixel position.
    /// * `bits_per_pixel` — §6.3.3 sample depth for the §7.3 clip.
    ///
    /// Direct-mode B-VOP MC always uses half-sample luma interpolation
    /// with rounding control `0` (§7.7.2.2 pseudo code's final `mc`
    /// argument).
    pub fn reconstruct(
        &self,
        references: &BVopFieldReferences<'_>,
        residual: &InterMacroblock,
        mb_origin_x: i32,
        mb_origin_y: i32,
        bits_per_pixel: u32,
    ) -> crate::reconstruct::ReconstructedMacroblock {
        let prediction = interlaced_direct_prediction(
            self.mvs,
            self.forward_top_ref,
            self.forward_bottom_ref,
            references,
            mb_origin_x,
            mb_origin_y,
            0,
        );
        crate::reconstruct::reconstruct_inter_macroblock(&prediction, residual, bits_per_pixel)
    }
}

/// The co-located anchor state a unified interlaced B-VOP macroblock
/// decode ([`BVopMvDriver::decode_interlaced_macroblock`]) needs, covering
/// all three §7.6.9 / §7.7.2.2 paths.
///
/// The macroblock's prediction mode is only known after the §6.2.6 header
/// is parsed, so the dispatcher must hold the inputs for every path it
/// might take: the §7.6.9.5.1 / §7.6.9.6 progressive co-located anchor
/// (used by a *progressive* direct MB), and — when the co-located *future*
/// macroblock was field-predicted — its forward field MVs + the B-VOP's
/// `top_field_first` (used by an *interlaced* direct MB). When
/// `future_field_mvs` is `None`, a Direct macroblock resolves to
/// progressive direct mode (§7.7.2.2: the future MB is skipped / GMC /
/// intra / frame-predicted).
#[derive(Debug, Clone, Copy)]
pub struct BVopInterlacedAnchor {
    /// The §7.6.9.5.1 / §7.6.9.6 progressive co-located anchor (skipped
    /// flag + co-located block MV) for the progressive paths.
    pub progressive: CoLocatedAnchor,
    /// The co-located *future* macroblock's forward field MVs + reference
    /// fields, present only when that macroblock was field-predicted
    /// (enabling interlaced direct mode). `None` forces progressive direct.
    pub future_field_mvs: Option<ColocatedFutureFieldMvs>,
    /// The B-VOP's `top_field_first` flag (§6.3.5), used by the interlaced
    /// direct δ selection.
    pub top_field_first: bool,
}

/// One decoded interlaced B-VOP macroblock, tagged by which §7.6.9 /
/// §7.7.2.2 path [`BVopMvDriver::decode_interlaced_macroblock`] dispatched.
///
/// Each variant carries the same per-MB decode the dedicated entry point
/// produces, so the caller reconstructs via the matching `reconstruct`.
#[derive(Debug, Clone)]
pub enum BVopInterlacedMb {
    /// A progressive (frame-predicted, or progressive-direct) macroblock —
    /// reconstruct with [`BVopMbDecode::reconstruct`] + anchor planes.
    Progressive(BVopMbDecode),
    /// An interlaced field-predicted macroblock (forward / backward /
    /// bidirectional) — reconstruct with [`BVopFieldMbDecode::reconstruct`].
    Field(BVopFieldMbDecode),
    /// An interlaced-direct macroblock — reconstruct with
    /// [`BVopInterlacedDirectMbDecode::reconstruct`].
    InterlacedDirect(BVopInterlacedDirectMbDecode),
}

/// The §7.6.8 progressive, non-scalable B-VOP motion-vector decode
/// driver.
///
/// Owns the running §7.6.8 predictor bank and the per-VOP temporal /
/// fcode parameters. Walk one macroblock at a time via
/// [`BVopMvDriver::decode_macroblock`], calling
/// [`BVopMvDriver::start_row`] at the start of each macroblock row to
/// honour the §7.6.8 "reset to zero only at the beginning of each
/// macroblock row" rule.
#[derive(Debug, Clone)]
pub struct BVopMvDriver {
    mb_rows: usize,
    mb_cols: usize,
    vop_fcode_forward: u8,
    vop_fcode_backward: u8,
    /// §7.6.7 temporal distance B-VOP → previous (forward) anchor.
    trb: i32,
    /// §7.6.7 temporal distance next anchor → previous anchor.
    trd: i32,
    predictor: DirectionPredictor,
    /// §7.7.2.2 four-PMV bank for the interlaced field-prediction path,
    /// threaded alongside the progressive `predictor` and reset together
    /// at each row start.
    field_predictor: FieldPmvBank,
}

impl BVopMvDriver {
    /// Create a driver for a `mb_rows × mb_cols` progressive,
    /// non-scalable B-VOP.
    ///
    /// `vop_fcode_forward` / `vop_fcode_backward` come from the VOP
    /// header (§6.3.5, `1..=7`). `trb` / `trd` are the §7.6.7 temporal
    /// references used by direct mode (`trb`: this B-VOP to the previous
    /// anchor; `trd`: next anchor to previous anchor). Pass any
    /// positive `trd >= trb` when the VOP contains no direct-mode
    /// macroblocks.
    pub fn new(
        mb_rows: usize,
        mb_cols: usize,
        vop_fcode_forward: u8,
        vop_fcode_backward: u8,
        trb: i32,
        trd: i32,
    ) -> Self {
        Self {
            mb_rows,
            mb_cols,
            vop_fcode_forward,
            vop_fcode_backward,
            trb,
            trd,
            predictor: DirectionPredictor::default(),
            field_predictor: FieldPmvBank::new(),
        }
    }

    /// Number of macroblock rows in the VOP.
    pub fn mb_rows(&self) -> usize {
        self.mb_rows
    }

    /// Number of macroblock columns in the VOP.
    pub fn mb_cols(&self) -> usize {
        self.mb_cols
    }

    /// §7.6.8 row-start reset: zero the forward and backward running
    /// predictors. Call once before the first macroblock of each row
    /// (and after a video-packet `resync_marker`, which §7.6.8 treats
    /// the same as a new row for predictor purposes).
    pub fn start_row(&mut self) {
        self.predictor = DirectionPredictor::default();
        self.field_predictor.reset_row();
    }

    /// Decode one progressive B-VOP macroblock at raster position
    /// `(mb_row, mb_col)`.
    ///
    /// * `br` is positioned at the start of the macroblock's §6.2.6
    ///   `modb` field. On `Ok` it sits immediately after the last
    ///   motion-vector body consumed.
    /// * `vol` / `vop_coding_type` gate the §6.2.6 header decode (the
    ///   driver is rectangular 4:2:0 progressive only — non-rectangular
    ///   / non-4:2:0 / interlaced field-prediction return typed
    ///   errors).
    /// * `co_located_skipped` is the §7.6.9.6 flag: `true` when the
    ///   co-located macroblock in the most recently decoded anchor was
    ///   *skipped* (`not_coded`). When set, the §7.6.9.6 overrides
    ///   apply before any `mb_type` decode.
    /// * `co_located_mv` is the §7.6.9.5.1 co-located anchor block
    ///   vector (after §7.6.1.6 vector padding) used by direct mode;
    ///   pass [`DirectCoLocatedMv::TransparentOrAbsent`] for a
    ///   transparent / unavailable slot.
    // Args map directly to the §6.2.6 / §7.6.9 macroblock-decode inputs;
    // grouping them into a struct would obscure the call site.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_macroblock(
        &mut self,
        br: &mut BitReader<'_>,
        vol: &VolHeader,
        vop_coding_type: VopCodingType,
        mb_row: usize,
        mb_col: usize,
        co_located_skipped: bool,
        co_located_mv: DirectCoLocatedMv,
    ) -> Result<BVopMbDecode, BVopMvDriverError> {
        if mb_row >= self.mb_rows || mb_col >= self.mb_cols {
            return Err(BVopMvDriverError::OutOfBounds { mb_row, mb_col });
        }

        let header = parse_b_vop_mb_header(br, vol, vop_coding_type, BMbTypeTable::B4)?;

        // §7.6.9.6 skipped co-located macroblock override.
        if co_located_skipped {
            return self.decode_co_located_skipped(br, &header, co_located_mv);
        }

        // Field prediction is out of scope for this progressive frame
        // driver; surface a typed error rather than silently treating
        // a field-pair as a frame vector.
        // `field_prediction` surfaces as `Some(_)` for both the
        // `field_prediction == 1` (populated forward / backward field
        // references) and the `field_prediction == 0`
        // (frame-prediction-in-interlaced, empty body) cases; only the
        // former codes field-pair motion bodies. Detect actual field
        // prediction by a populated forward / backward reference.
        let field_predicted = header.interlaced_info.as_ref().is_some_and(|info| {
            info.field_prediction
                .is_some_and(|fp| fp.forward.is_some() || fp.backward.is_some())
        });
        if field_predicted {
            return Err(BVopMvDriverError::FieldPredictionUnsupported);
        }

        match header.mb_type {
            BVopMbType::Direct => self.decode_direct(br, &header, co_located_mv),
            BVopMbType::Forward => self.decode_forward(br, &header),
            BVopMbType::Backward => self.decode_backward(br, &header),
            BVopMbType::Interpolated => self.decode_bidirectional(br, &header),
        }
    }

    /// Decode one interlaced **field-predicted** B-VOP macroblock
    /// (§7.7.2.2 forward / backward / bidirectional) at raster position
    /// `(mb_row, mb_col)`.
    ///
    /// This is the field-prediction analogue of
    /// [`BVopMvDriver::decode_macroblock`]: where that method rejects a
    /// field-predicted macroblock with
    /// [`BVopMvDriverError::FieldPredictionUnsupported`], this method
    /// decodes the §6.2.6 field motion bodies, threads them through the
    /// §7.7.2.2 four-PMV bank (reset per row by
    /// [`BVopMvDriver::start_row`]), and returns a [`BVopFieldMbDecode`]
    /// ready for [`BVopFieldMbDecode::reconstruct`].
    ///
    /// Interlaced **direct** mode is still rejected with
    /// `FieldPredictionUnsupported` — it needs the §7.7.2.2 field-period
    /// `TRB[i]` / `TRD[i]` temporal references and the Table 7-16 `δ`
    /// parity selection, a distinct sub-subsystem. A macroblock that is
    /// *not* field-predicted (frame prediction, even in an interlaced
    /// VOL) returns `FieldPredictionUnsupported` too — drive it through
    /// [`BVopMvDriver::decode_macroblock`] instead.
    ///
    /// `br` is positioned at the start of the macroblock's §6.2.6 `modb`
    /// field. On `Ok` it sits immediately after the last motion-vector
    /// body (before the texture).
    pub fn decode_field_macroblock(
        &mut self,
        br: &mut BitReader<'_>,
        vol: &VolHeader,
        vop_coding_type: VopCodingType,
        mb_row: usize,
        mb_col: usize,
    ) -> Result<BVopFieldMbDecode, BVopMvDriverError> {
        if mb_row >= self.mb_rows || mb_col >= self.mb_cols {
            return Err(BVopMvDriverError::OutOfBounds { mb_row, mb_col });
        }

        let header = parse_b_vop_mb_header(br, vol, vop_coding_type, BMbTypeTable::B4)?;

        // The macroblock must actually be field-predicted (a populated
        // forward / backward field-reference pair). A frame-predicted MB
        // belongs on `decode_macroblock`.
        let field = header
            .interlaced_info
            .as_ref()
            .and_then(|info| info.field_prediction)
            .filter(|fp| fp.forward.is_some() || fp.backward.is_some());
        let Some(field) = field else {
            return Err(BVopMvDriverError::FieldPredictionUnsupported);
        };

        // Interlaced direct mode is out of scope (§7.7.2.2 last block).
        if header.mb_type == BVopMbType::Direct {
            return Err(BVopMvDriverError::FieldPredictionUnsupported);
        }

        self.field_from_header(br, &header, &field)
    }

    /// Decode the §6.2.6 field motion bodies of an already-parsed,
    /// confirmed-field-predicted, non-direct macroblock header, threading
    /// them through the §7.7.2.2 four-PMV bank.
    fn field_from_header(
        &mut self,
        br: &mut BitReader<'_>,
        header: &BVopMbHeader,
        field: &crate::interlaced_information::FieldPrediction,
    ) -> Result<BVopFieldMbDecode, BVopMvDriverError> {
        // Interlaced direct mode is handled separately (§7.7.2.2 last
        // block); this helper is only reached for the non-direct field
        // modes.
        if header.mb_type == BVopMbType::Direct {
            return Err(BVopMvDriverError::FieldPredictionUnsupported);
        }

        let references = field_reference_flags(field);
        let bank_before = self.field_predictor;

        let bodies = decode_b_vop_mb_motion_vectors(
            br,
            header.mb_type,
            true,
            self.vop_fcode_forward,
            self.vop_fcode_backward,
        )?;

        let mode = self.apply_field_bodies(header.mb_type, &bodies)?;

        Ok(BVopFieldMbDecode {
            mode,
            references,
            cbpb: header.cbpb,
            dbquant_delta: header.dbquant_delta,
            bank_before,
        })
    }

    /// Thread the decoded §6.2.6 field motion bodies through the
    /// §7.7.2.2 four-PMV bank, updating it in place and returning the
    /// decoded differentials packaged by mode.
    fn apply_field_bodies(
        &mut self,
        mb_type: BVopMbType,
        bodies: &BVopMotionVectors,
    ) -> Result<BVopFieldMode, BVopMvDriverError> {
        match mb_type {
            BVopMbType::Forward => {
                let deltas = field_deltas(bodies.forward)?;
                self.field_predictor
                    .field_forward(deltas.top, deltas.bottom);
                Ok(BVopFieldMode::Forward(deltas))
            }
            BVopMbType::Backward => {
                let deltas = field_deltas(bodies.backward)?;
                self.field_predictor
                    .field_backward(deltas.top, deltas.bottom);
                Ok(BVopFieldMode::Backward(deltas))
            }
            BVopMbType::Interpolated => {
                let fwd = field_deltas(bodies.forward)?;
                let bwd = field_deltas(bodies.backward)?;
                let mvd = [fwd.top, fwd.bottom, bwd.top, bwd.bottom];
                self.field_predictor.field_bidirectional(mvd);
                Ok(BVopFieldMode::Bidirectional(mvd))
            }
            BVopMbType::Direct => Err(BVopMvDriverError::FieldPredictionUnsupported),
        }
    }

    /// Decode one interlaced **direct** B-VOP macroblock (§7.7.2.2 last
    /// pseudo-code block) at raster position `(mb_row, mb_col)`.
    ///
    /// Interlaced direct mode applies when the co-located macroblock of
    /// the *future* reference VOP is itself field-predicted
    /// (`field_prediction == 1`, §7.7.2.2). The caller establishes that
    /// from the reference-frame chain and supplies the future macroblock's
    /// two forward field MVs (`future_field_mvs`) plus the B-VOP's
    /// `top_field_first` flag. The four derived field MVs are then
    /// computed from those, the single transmitted `MVD[0]`, and the
    /// driver's frame-period `TRB` / `TRD` (the same `trb` / `trd` the
    /// progressive §7.6.9.5.2 direct path uses).
    ///
    /// `br` is positioned at the start of the macroblock's §6.2.6 `modb`
    /// field. The macroblock must decode to the Direct type (a `modb`
    /// `"1"` default or an explicit Direct `mb_type`); a non-direct
    /// macroblock returns [`BVopMvDriverError::FieldPredictionUnsupported`]
    /// (drive it through [`BVopMvDriver::decode_field_macroblock`] or
    /// [`BVopMvDriver::decode_macroblock`] instead). On `Ok`, `br` sits
    /// immediately after the single `MVD[0]` body (before the texture);
    /// a `modb == "1"` direct macroblock codes no `MVD[0]` (the delta is
    /// implicitly zero) and the reader stops after `modb`.
    ///
    /// `MVD[0]` is decoded assuming `f_code == 1` regardless of the VOP
    /// header f_codes (§7.7.2.2). The four-PMV bank is **not** touched —
    /// direct mode neither reads nor updates it.
    // Args map directly to the §6.2.6 / §7.7.2.2 interlaced-direct decode
    // inputs; grouping them into a struct would obscure the call site.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_interlaced_direct_macroblock(
        &mut self,
        br: &mut BitReader<'_>,
        vol: &VolHeader,
        vop_coding_type: VopCodingType,
        mb_row: usize,
        mb_col: usize,
        future_field_mvs: ColocatedFutureFieldMvs,
        top_field_first: bool,
    ) -> Result<BVopInterlacedDirectMbDecode, BVopMvDriverError> {
        if mb_row >= self.mb_rows || mb_col >= self.mb_cols {
            return Err(BVopMvDriverError::OutOfBounds { mb_row, mb_col });
        }

        let header = parse_b_vop_mb_header(br, vol, vop_coding_type, BMbTypeTable::B4)?;
        self.interlaced_direct_from_header(br, &header, future_field_mvs, top_field_first)
    }

    /// Decode the single `MVD[0]` body of an already-parsed Direct
    /// macroblock header and derive the four §7.7.2.2 interlaced-direct
    /// field MVs.
    fn interlaced_direct_from_header(
        &mut self,
        br: &mut BitReader<'_>,
        header: &BVopMbHeader,
        future_field_mvs: ColocatedFutureFieldMvs,
        top_field_first: bool,
    ) -> Result<BVopInterlacedDirectMbDecode, BVopMvDriverError> {
        if header.mb_type != BVopMbType::Direct {
            return Err(BVopMvDriverError::FieldPredictionUnsupported);
        }

        // §6.2.6: a `modb == "1"` direct macroblock codes no `MVD[0]`
        // body (the whole subtree after `modb` is absent) → the delta is
        // implicitly zero. An explicit Direct `mb_type` (`modb == "01"`)
        // carries the single direct body, decoded with f_code 1.
        let delta = if header.mb_type_present {
            let bodies = decode_b_vop_mb_motion_vectors(
                br,
                BVopMbType::Direct,
                false,
                self.vop_fcode_forward,
                self.vop_fcode_backward,
            )?;
            bodies.direct.unwrap_or(MotionVectorDelta { dx: 0, dy: 0 })
        } else {
            MotionVectorDelta { dx: 0, dy: 0 }
        };
        let delta_mv = MotionVector {
            x: delta.dx,
            y: delta.dy,
        };

        let mvs = interlaced_direct_mvs(
            future_field_mvs.top,
            future_field_mvs.bottom,
            delta_mv,
            self.trb,
            self.trd,
            top_field_first,
        );

        Ok(BVopInterlacedDirectMbDecode {
            mvs,
            forward_top_ref: future_field_mvs.top.reference_field,
            forward_bottom_ref: future_field_mvs.bottom.reference_field,
            cbpb: header.cbpb,
            dbquant_delta: header.dbquant_delta,
        })
    }

    /// Decode one interlaced B-VOP macroblock with **automatic path
    /// dispatch**: parse the §6.2.6 header once, then route to the
    /// progressive (§7.6.9), interlaced field-prediction (§7.7.2.2
    /// forward / backward / bidirectional), or interlaced-direct
    /// (§7.7.2.2) decode based on the decoded `mb_type` +
    /// `field_prediction`.
    ///
    /// This is the unified entry the dedicated
    /// [`BVopMvDriver::decode_macroblock`] /
    /// [`BVopMvDriver::decode_field_macroblock`] /
    /// [`BVopMvDriver::decode_interlaced_direct_macroblock`] methods feed:
    /// a frame loop over an interlaced B-VOP can call this once per
    /// macroblock without peeking the header to pre-select the path.
    ///
    /// Dispatch rules:
    /// * `field_prediction == 1` (populated forward / backward field
    ///   references) + non-direct → field-predicted
    ///   ([`BVopInterlacedMb::Field`]).
    /// * Direct `mb_type` with `anchor.future_field_mvs == Some(_)` (the
    ///   co-located future macroblock was field-predicted) → interlaced
    ///   direct ([`BVopInterlacedMb::InterlacedDirect`]).
    /// * Otherwise → progressive ([`BVopInterlacedMb::Progressive`]),
    ///   including a progressive-direct MB (Direct type with no future
    ///   field MVs) and the §7.6.9.6 co-located-skipped overrides.
    ///
    /// `br` is positioned at the start of the macroblock's `modb` field;
    /// on `Ok` it sits after the last motion body (before the texture).
    pub fn decode_interlaced_macroblock(
        &mut self,
        br: &mut BitReader<'_>,
        vol: &VolHeader,
        vop_coding_type: VopCodingType,
        mb_row: usize,
        mb_col: usize,
        anchor: BVopInterlacedAnchor,
    ) -> Result<BVopInterlacedMb, BVopMvDriverError> {
        if mb_row >= self.mb_rows || mb_col >= self.mb_cols {
            return Err(BVopMvDriverError::OutOfBounds { mb_row, mb_col });
        }

        let header = parse_b_vop_mb_header(br, vol, vop_coding_type, BMbTypeTable::B4)?;

        // §7.6.9.6 co-located-skipped override takes precedence over every
        // path — a skipped co-located anchor forces progressive direct /
        // forward regardless of the decoded type (§7.6.9.6).
        if anchor.progressive.skipped {
            return self
                .decode_co_located_skipped(br, &header, anchor.progressive.mv)
                .map(BVopInterlacedMb::Progressive);
        }

        // Field-predicted, non-direct → field path.
        let field = header
            .interlaced_info
            .as_ref()
            .and_then(|info| info.field_prediction)
            .filter(|fp| fp.forward.is_some() || fp.backward.is_some());
        if let Some(field) = field {
            if header.mb_type != BVopMbType::Direct {
                return self
                    .field_from_header(br, &header, &field)
                    .map(BVopInterlacedMb::Field);
            }
        }

        // Direct macroblock: interlaced direct when the future MB was
        // field-predicted, progressive direct otherwise (§7.7.2.2).
        if header.mb_type == BVopMbType::Direct {
            if let Some(future) = anchor.future_field_mvs {
                return self
                    .interlaced_direct_from_header(br, &header, future, anchor.top_field_first)
                    .map(BVopInterlacedMb::InterlacedDirect);
            }
            return self
                .decode_direct(br, &header, anchor.progressive.mv)
                .map(BVopInterlacedMb::Progressive);
        }

        // Progressive frame-predicted forward / backward / interpolated.
        match header.mb_type {
            BVopMbType::Forward => self.decode_forward(br, &header),
            BVopMbType::Backward => self.decode_backward(br, &header),
            BVopMbType::Interpolated => self.decode_bidirectional(br, &header),
            BVopMbType::Direct => unreachable!("Direct handled above"),
        }
        .map(BVopInterlacedMb::Progressive)
    }

    /// Decode the motion state of an entire progressive B-VOP in raster
    /// order, returning one [`BVopMbDecode`] per macroblock (row-major,
    /// `mb_rows * mb_cols` entries).
    ///
    /// This is the frame-level §7.6.8 walk: it calls
    /// [`BVopMvDriver::start_row`] at the start of each macroblock row
    /// (honouring the "predictors reset to zero only at the beginning of
    /// each macroblock row" rule) and threads the running predictor bank
    /// across the macroblocks of each row via repeated
    /// [`BVopMvDriver::decode_macroblock`] calls.
    ///
    /// `co_located` supplies, for each `(mb_row, mb_col)`, the §7.6.9.5.1
    /// / §7.6.9.6 co-located anchor state — whether the co-located
    /// macroblock in the most recently decoded anchor was *skipped* and
    /// its (vector-padded) block MV for direct mode. The closure is
    /// called exactly once per macroblock in raster order.
    ///
    /// The bit reader is left positioned immediately after the final
    /// macroblock's last motion-vector body. This walker does **not**
    /// consume the §6.2.6 texture (residual) bodies — those are decoded
    /// separately by the caller's texture path; this driver owns only
    /// the §7.6.8 motion-vector decode and predictor threading. A VOP
    /// that interleaves texture between macroblocks must instead drive
    /// macroblocks one at a time via [`BVopMvDriver::decode_macroblock`].
    pub fn decode_vop_motion<F>(
        &mut self,
        br: &mut BitReader<'_>,
        vol: &VolHeader,
        vop_coding_type: VopCodingType,
        mut co_located: F,
    ) -> Result<Vec<BVopMbDecode>, BVopMvDriverError>
    where
        F: FnMut(usize, usize) -> CoLocatedAnchor,
    {
        let mut out = Vec::with_capacity(self.mb_rows * self.mb_cols);
        for mb_row in 0..self.mb_rows {
            self.start_row();
            for mb_col in 0..self.mb_cols {
                let anchor = co_located(mb_row, mb_col);
                let decode = self.decode_macroblock(
                    br,
                    vol,
                    vop_coding_type,
                    mb_row,
                    mb_col,
                    anchor.skipped,
                    anchor.mv,
                )?;
                out.push(decode);
            }
        }
        Ok(out)
    }

    /// Decode an entire progressive B-VOP **end-to-end** in raster order:
    /// the §7.6.8 motion walk threaded with the §6.2.6 / §7.4 residual
    /// (texture) decode that follows each macroblock's motion bodies.
    ///
    /// This is the residual-threaded frame loop on top of
    /// [`BVopMvDriver::decode_vop_motion`]: for each `(mb_row, mb_col)`
    /// it decodes the motion state (leaving `br` at the start of the
    /// macroblock's texture), applies the macroblock's `dbquant` delta to
    /// the running quantiser scale (§6.3.6, clipped to
    /// `[1, max_quantiser_scale]`), then consumes the §7.4 inter residual
    /// gated by the macroblock's `cbpb`. `br` is left positioned
    /// immediately after the final macroblock's last texture block.
    ///
    /// The returned [`BVopMbTexturedDecode`]s (row-major,
    /// `mb_rows * mb_cols` entries) each carry the motion state, the
    /// decoded residual, and the running quantiser scale; feed each into
    /// [`BVopMbDecode::reconstruct`] with the anchor planes to produce the
    /// displayed macroblock.
    ///
    /// `co_located` supplies the §7.6.9.5.1 / §7.6.9.6 anchor state per
    /// macroblock exactly as for [`BVopMvDriver::decode_vop_motion`].
    /// This frame loop assumes the texture immediately follows the motion
    /// bodies of the *same* macroblock (the §6.2.6 macroblock-layer
    /// order); a VOP carrying video packets with `resync_marker`-split
    /// texture must drive macroblocks one at a time.
    pub fn decode_vop<F>(
        &mut self,
        br: &mut BitReader<'_>,
        vol: &VolHeader,
        vop_coding_type: VopCodingType,
        texture: BVopTextureParams,
        mut co_located: F,
    ) -> Result<Vec<BVopMbTexturedDecode>, BVopMvDriverError>
    where
        F: FnMut(usize, usize) -> CoLocatedAnchor,
    {
        let quant_matrix = crate::block::nonintra_quant_matrix(vol);
        let mut out = Vec::with_capacity(self.mb_rows * self.mb_cols);
        // §6.3.6: the running quantiser scale carries across macroblocks
        // within the VOP, modified only by each macroblock's dbquant.
        let mut quantiser_scale = texture.base_quantiser_scale;
        for mb_row in 0..self.mb_rows {
            self.start_row();
            for mb_col in 0..self.mb_cols {
                let anchor = co_located(mb_row, mb_col);
                let motion = self.decode_macroblock(
                    br,
                    vol,
                    vop_coding_type,
                    mb_row,
                    mb_col,
                    anchor.skipped,
                    anchor.mv,
                )?;

                // §6.3.6: apply this macroblock's dbquant (Table 6-33
                // value already resolved into `dbquant_delta`) to the
                // running quantiser scale, clipped to
                // `[1, max_quantiser_scale]`. dbquant is present only when
                // cbpb != 0 and the type is non-direct, so the residual
                // it gates always uses the updated scale.
                if let Some(delta) = motion.dbquant_delta {
                    let updated = quantiser_scale as i64 + delta as i64;
                    quantiser_scale = updated.clamp(1, texture.max_quantiser_scale as i64) as u32;
                }

                let ctx = MacroblockTextureContext {
                    quantiser_scale,
                    bits_per_pixel: texture.bits_per_pixel,
                    quant_type: texture.quant_type,
                    ac_pred_flag: false,
                };
                let residual = decode_b_vop_inter_macroblock(br, motion.cbpb, ctx, &quant_matrix)?;

                out.push(BVopMbTexturedDecode {
                    motion,
                    residual,
                    quantiser_scale,
                });
            }
        }
        Ok(out)
    }

    /// §7.6.9.6: a skipped co-located anchor macroblock forces the
    /// current B-MB to direct (zero delta) when `modb == "1"`, else
    /// forward with the zero MV. No motion bits are present in the
    /// `modb == "1"` case (its whole subtree is absent); the explicit
    /// branch still parses the header but the spec's override replaces
    /// the decoded type's reconstruction.
    fn decode_co_located_skipped(
        &mut self,
        br: &mut BitReader<'_>,
        header: &BVopMbHeader,
        co_located_mv: DirectCoLocatedMv,
    ) -> Result<BVopMbDecode, BVopMvDriverError> {
        if !header.mb_type_present {
            // modb == "1" → direct mode with a zero delta vector. No
            // motion bodies are coded.
            let zero_delta = MotionVectorDelta { dx: 0, dy: 0 };
            let mvs = self.direct_mvs(co_located_mv, zero_delta)?;
            let (fwd_chroma, bwd_chroma) = direct_chroma_mvs(&mvs)?;
            return Ok(BVopMbDecode {
                mb_type: BVopMbType::Direct,
                prediction_mode: BVopPredictionMode::Direct,
                cbpb: header.cbpb,
                dbquant_delta: header.dbquant_delta,
                mvs,
                forward_chroma_mv: fwd_chroma,
                backward_chroma_mv: bwd_chroma,
            });
        }

        // modb != "1" but co-located skipped → forward mode, zero MV.
        // The §6.2.6 body for the decoded mb_type would normally code
        // motion vectors; the spec's §7.6.9.6 override supersedes the
        // reconstruction, but the bitstream still carries whatever
        // bodies the decoded mb_type implies, so consume them to keep
        // the reader aligned, then discard.
        let _ = decode_b_vop_mb_motion_vectors(
            br,
            header.mb_type,
            false,
            self.vop_fcode_forward,
            self.vop_fcode_backward,
        )?;
        let zero = MotionVector { x: 0, y: 0 };
        let mvs = [BVopMvPair {
            forward: zero,
            backward: zero,
        }; MB_SUB_BLOCKS];
        // §7.6.8: forward mode updates only the forward predictor.
        self.predictor.forward = zero;
        Ok(BVopMbDecode {
            mb_type: BVopMbType::Forward,
            prediction_mode: BVopPredictionMode::ForwardOnly,
            cbpb: header.cbpb,
            dbquant_delta: header.dbquant_delta,
            mvs,
            forward_chroma_mv: zero,
            backward_chroma_mv: zero,
        })
    }

    /// Direct mode (§7.6.9.5). The single delta vector is decoded with
    /// predictor zero and f_code one (§7.6.8); the predictor bank is not
    /// touched.
    fn decode_direct(
        &mut self,
        br: &mut BitReader<'_>,
        header: &BVopMbHeader,
        co_located_mv: DirectCoLocatedMv,
    ) -> Result<BVopMbDecode, BVopMvDriverError> {
        // §6.2.6: when `modb == "1"` (no explicit `mb_type`), the whole
        // subtree after `modb` is absent — there is no `motion_vector`
        // body. The direct delta vector is then implicitly zero. Only a
        // `modb == "01"` macroblock that *decoded* the Direct type
        // (Table B.4 row `1`) carries a direct body on the wire.
        let delta = if header.mb_type_present {
            let bodies = decode_b_vop_mb_motion_vectors(
                br,
                BVopMbType::Direct,
                false,
                self.vop_fcode_forward,
                self.vop_fcode_backward,
            )?;
            bodies.direct.unwrap_or(MotionVectorDelta { dx: 0, dy: 0 })
        } else {
            MotionVectorDelta { dx: 0, dy: 0 }
        };
        let mvs = self.direct_mvs(co_located_mv, delta)?;
        let (fwd_chroma, bwd_chroma) = direct_chroma_mvs(&mvs)?;
        Ok(BVopMbDecode {
            mb_type: BVopMbType::Direct,
            prediction_mode: BVopPredictionMode::Direct,
            cbpb: header.cbpb,
            dbquant_delta: header.dbquant_delta,
            mvs,
            forward_chroma_mv: fwd_chroma,
            backward_chroma_mv: bwd_chroma,
        })
    }

    /// Forward mode (§7.6.9.2). One forward MV, predicted from the
    /// running forward predictor; the predictor is then set to the
    /// decoded vector (§7.6.8 case 1).
    fn decode_forward(
        &mut self,
        br: &mut BitReader<'_>,
        header: &BVopMbHeader,
    ) -> Result<BVopMbDecode, BVopMvDriverError> {
        let bodies = decode_b_vop_mb_motion_vectors(
            br,
            BVopMbType::Forward,
            false,
            self.vop_fcode_forward,
            self.vop_fcode_backward,
        )?;
        let delta = frame_delta(bodies.forward)?;
        let mv = reconstruct_motion_vector(
            delta,
            self.predictor.forward.x,
            self.predictor.forward.y,
            self.vop_fcode_forward,
        )
        .map_err(BVopMbParseError::Motion)?;
        self.predictor.forward = mv;
        let mvs = replicated_pair(mv, MotionVector { x: 0, y: 0 });
        let fwd_chroma = chroma_mv_from_luma_blocks(&[mv])?;
        Ok(BVopMbDecode {
            mb_type: BVopMbType::Forward,
            prediction_mode: BVopPredictionMode::ForwardOnly,
            cbpb: header.cbpb,
            dbquant_delta: header.dbquant_delta,
            mvs,
            forward_chroma_mv: fwd_chroma,
            backward_chroma_mv: MotionVector { x: 0, y: 0 },
        })
    }

    /// Backward mode (§7.6.9.3). One backward MV, predicted from the
    /// running backward predictor; the predictor is then set to the
    /// decoded vector (§7.6.8 case 2).
    fn decode_backward(
        &mut self,
        br: &mut BitReader<'_>,
        header: &BVopMbHeader,
    ) -> Result<BVopMbDecode, BVopMvDriverError> {
        let bodies = decode_b_vop_mb_motion_vectors(
            br,
            BVopMbType::Backward,
            false,
            self.vop_fcode_forward,
            self.vop_fcode_backward,
        )?;
        let delta = frame_delta(bodies.backward)?;
        let mv = reconstruct_motion_vector(
            delta,
            self.predictor.backward.x,
            self.predictor.backward.y,
            self.vop_fcode_backward,
        )
        .map_err(BVopMbParseError::Motion)?;
        self.predictor.backward = mv;
        let mvs = replicated_pair(MotionVector { x: 0, y: 0 }, mv);
        let bwd_chroma = chroma_mv_from_luma_blocks(&[mv])?;
        Ok(BVopMbDecode {
            mb_type: BVopMbType::Backward,
            prediction_mode: BVopPredictionMode::BackwardOnly,
            cbpb: header.cbpb,
            dbquant_delta: header.dbquant_delta,
            mvs,
            forward_chroma_mv: MotionVector { x: 0, y: 0 },
            backward_chroma_mv: bwd_chroma,
        })
    }

    /// Bidirectional / interpolated mode (§7.6.9.4). Forward and
    /// backward MVs, each predicted from + then updating its own running
    /// predictor (§7.6.8 case 3).
    fn decode_bidirectional(
        &mut self,
        br: &mut BitReader<'_>,
        header: &BVopMbHeader,
    ) -> Result<BVopMbDecode, BVopMvDriverError> {
        let bodies = decode_b_vop_mb_motion_vectors(
            br,
            BVopMbType::Interpolated,
            false,
            self.vop_fcode_forward,
            self.vop_fcode_backward,
        )?;
        let fwd_delta = frame_delta(bodies.forward)?;
        let bwd_delta = frame_delta(bodies.backward)?;
        let fwd_mv = reconstruct_motion_vector(
            fwd_delta,
            self.predictor.forward.x,
            self.predictor.forward.y,
            self.vop_fcode_forward,
        )
        .map_err(BVopMbParseError::Motion)?;
        let bwd_mv = reconstruct_motion_vector(
            bwd_delta,
            self.predictor.backward.x,
            self.predictor.backward.y,
            self.vop_fcode_backward,
        )
        .map_err(BVopMbParseError::Motion)?;
        self.predictor.forward = fwd_mv;
        self.predictor.backward = bwd_mv;
        let mvs = replicated_pair(fwd_mv, bwd_mv);
        let fwd_chroma = chroma_mv_from_luma_blocks(&[fwd_mv])?;
        let bwd_chroma = chroma_mv_from_luma_blocks(&[bwd_mv])?;
        Ok(BVopMbDecode {
            mb_type: BVopMbType::Interpolated,
            prediction_mode: BVopPredictionMode::Bidirectional,
            cbpb: header.cbpb,
            dbquant_delta: header.dbquant_delta,
            mvs,
            forward_chroma_mv: fwd_chroma,
            backward_chroma_mv: bwd_chroma,
        })
    }

    /// §7.6.9.5.2 direct-mode (MVF, MVB) pairs for the four luminance
    /// sub-blocks, scaling the single co-located anchor MV by the
    /// §7.6.7 TRB / TRD distances. The progressive (frame) path uses
    /// the same co-located MV for every sub-block.
    fn direct_mvs(
        &self,
        co_located_mv: DirectCoLocatedMv,
        delta: MotionVectorDelta,
    ) -> Result<[BVopMvPair; MB_SUB_BLOCKS], BVopMvDriverError> {
        let DirectModeMv { forward, backward } = direct_mode_motion_vector(
            co_located_mv,
            delta,
            self.trb,
            self.trd,
            DirectMvUnits::Match,
        )?;
        Ok([BVopMvPair { forward, backward }; MB_SUB_BLOCKS])
    }
}

/// §7.6.9.5.3 chroma MVs for direct mode: the four luminance forward /
/// backward sub-block vectors are reduced via §7.6.5 `sum / 2K`.
fn direct_chroma_mvs(
    mvs: &[BVopMvPair; MB_SUB_BLOCKS],
) -> Result<(MotionVector, MotionVector), BVopMvDriverError> {
    let fwd: [MotionVector; MB_SUB_BLOCKS] = [
        mvs[0].forward,
        mvs[1].forward,
        mvs[2].forward,
        mvs[3].forward,
    ];
    let bwd: [MotionVector; MB_SUB_BLOCKS] = [
        mvs[0].backward,
        mvs[1].backward,
        mvs[2].backward,
        mvs[3].backward,
    ];
    Ok((
        chroma_mv_from_luma_blocks(&fwd)?,
        chroma_mv_from_luma_blocks(&bwd)?,
    ))
}

/// Map a decoded §6.2.6.3 `FieldPrediction` to the four
/// `*_field_reference` flags. An absent forward / backward pair (the
/// direction this macroblock does not predict) defaults to
/// [`FieldReference::Top`] — the corresponding direction is never
/// compensated, so the value is unused.
fn field_reference_flags(
    fp: &crate::interlaced_information::FieldPrediction,
) -> FieldReferenceFlags {
    let (forward_top, forward_bottom) = fp
        .forward
        .unwrap_or((FieldReference::Top, FieldReference::Top));
    let (backward_top, backward_bottom) = fp
        .backward
        .unwrap_or((FieldReference::Top, FieldReference::Top));
    FieldReferenceFlags {
        forward_top,
        forward_bottom,
        backward_top,
        backward_bottom,
    }
}

/// Extract the top/bottom field differentials from a §6.2.6 motion-vector
/// body, requiring a field pair (the field-prediction driver path).
fn field_deltas(body: Option<BVopMvBody>) -> Result<FieldMvDeltas, BVopMvDriverError> {
    match body {
        Some(BVopMvBody::Field(pair)) => Ok(FieldMvDeltas {
            top: MotionVector {
                x: pair.top.dx,
                y: pair.top.dy,
            },
            bottom: MotionVector {
                x: pair.bottom.dx,
                y: pair.bottom.dy,
            },
        }),
        // A frame body where a field pair was expected, or an absent
        // direction the mb_type promised — neither is reachable for a
        // correctly-decoded field-predicted macroblock, but stay robust.
        Some(BVopMvBody::Frame(_)) | None => Err(BVopMvDriverError::FieldPredictionUnsupported),
    }
}

/// Extract the frame differential from a §6.2.6 motion-vector body,
/// rejecting a field pair (which this progressive driver does not
/// handle).
fn frame_delta(body: Option<BVopMvBody>) -> Result<MotionVectorDelta, BVopMvDriverError> {
    match body {
        Some(BVopMvBody::Frame(delta)) => Ok(delta),
        Some(BVopMvBody::Field(_)) => Err(BVopMvDriverError::FieldPredictionUnsupported),
        // A direction the mb_type promised should always be present;
        // treat absence as a zero differential to stay robust.
        None => Ok(MotionVectorDelta { dx: 0, dy: 0 }),
    }
}

/// Build the four-sub-block `(MVF, MVB)` array by replicating a single
/// forward + backward MV pair across all four sub-blocks (the §7.6.9.2
/// /.3 /.4 one-MV-per-MB rule).
fn replicated_pair(forward: MotionVector, backward: MotionVector) -> [BVopMvPair; MB_SUB_BLOCKS] {
    [BVopMvPair { forward, backward }; MB_SUB_BLOCKS]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vol::{AspectRatio, SpriteEnable, VolHeader};

    fn make_vol() -> VolHeader {
        VolHeader {
            profile_level: 0x01,
            width: 176,
            height: 144,
            time_increment_resolution: 30_000,
            fixed_vop_rate: false,
            fixed_vop_time_increment: None,
            aspect_ratio: AspectRatio::Square,
            vol_control: None,
            random_accessible_vol: false,
            video_object_type_indication: 1,
            is_object_layer_identifier: false,
            video_object_layer_verid: 1,
            video_object_layer_priority: 0,
            video_object_layer_shape: 0,
            interlaced: false,
            obmc_disable: false,
            sprite_enable: SpriteEnable::NotUsed,
            no_of_sprite_warping_points: None,
            sprite_warping_accuracy: None,
            sprite_brightness_change: None,
            sprite_geometry: None,
            low_latency_sprite_enable: None,
            not_8_bit: false,
            quant_precision: 5,
            bits_per_pixel: 8,
            quant_type: false,
            intra_quant_mat: None,
            nonintra_quant_mat: None,
            quarter_sample: false,
            complexity_estimation_disable: true,
            resync_marker_disable: true,
            data_partitioned: false,
            reversible_vlc: false,
            newpred_enable: false,
            reduced_resolution_vop_enable: false,
            scalability: false,
        }
    }

    /// MSB-first bit writer matching the spec's bslbf / uimsbf
    /// convention (mirrors `bvop::tests::BitWriter`).
    struct BitWriter {
        buf: Vec<u8>,
        bit_pos: usize,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                buf: Vec::new(),
                bit_pos: 0,
            }
        }
        fn write_bits(&mut self, value: u32, n: usize) {
            for i in (0..n).rev() {
                let bit = ((value >> i) & 1) as u8;
                if self.bit_pos % 8 == 0 {
                    self.buf.push(0);
                }
                let byte = self.buf.last_mut().unwrap();
                *byte |= bit << (7 - (self.bit_pos % 8));
                self.bit_pos += 1;
            }
        }
        /// Write the Table B.12 `mv_data == 0` codeword (`1`) — a zero
        /// differential component. The forward / backward / direct
        /// bodies are two of these back to back when the delta is
        /// (0, 0).
        fn write_zero_mv_data(&mut self) {
            self.write_bits(0b1, 1);
        }
        /// Write a full `(0, 0)` motion-vector body (two zero `mv_data`
        /// codewords — the horizontal then vertical component).
        fn write_zero_mv_data_pair(&mut self) {
            self.write_zero_mv_data();
            self.write_zero_mv_data();
        }
        fn align(&mut self) {
            while self.bit_pos % 8 != 0 {
                self.write_bits(0, 1);
            }
        }
    }

    #[test]
    fn out_of_bounds_macroblock_rejected() {
        let vol = make_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb == "1"
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(2, 2, 1, 1, 1, 2);
        let err = driver
            .decode_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                5,
                5,
                false,
                DirectCoLocatedMv::TransparentOrAbsent,
            )
            .unwrap_err();
        assert_eq!(
            err,
            BVopMvDriverError::OutOfBounds {
                mb_row: 5,
                mb_col: 5
            }
        );
    }

    #[test]
    fn modb_one_default_direct_zero_delta() {
        // modb == "1" → default type (direct, non-scalable), no motion
        // bodies. The driver derives the direct MVs from a zero
        // co-located anchor MV with a zero delta → all-zero MVF/MVB.
        let vol = make_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        driver.start_row();
        let mb = driver
            .decode_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                0,
                0,
                false,
                DirectCoLocatedMv::TransparentOrAbsent,
            )
            .unwrap();
        assert_eq!(mb.mb_type, BVopMbType::Direct);
        assert_eq!(mb.prediction_mode, BVopPredictionMode::Direct);
        for pair in mb.mvs {
            assert_eq!(pair.forward, MotionVector { x: 0, y: 0 });
            assert_eq!(pair.backward, MotionVector { x: 0, y: 0 });
        }
    }

    #[test]
    fn forward_predictor_threads_across_macroblocks() {
        // Two forward-mode MBs in a row. Each codes mvdf = (0, 0), so
        // each decoded forward MV equals the running predictor. The
        // first reconstructs to (0, 0) (predictor starts at zero); the
        // predictor then stays (0, 0). We verify the §7.6.8 "forward
        // mode updates only the forward predictor" path runs.
        let vol = make_vol();
        let mut w = BitWriter::new();
        // MB0: modb "01" (mb_type present), mb_type "0001" (forward),
        // dbquant skipped (no cbpb so cbpb == None → dbquant gate
        // requires cbpb != 0, absent → skipped), then mvdf body: two
        // zero mv_data codewords.
        w.write_bits(0b01, 2); // modb "01"
        w.write_bits(0b0001, 4); // mb_type forward (Table B.4)
        w.write_zero_mv_data(); // mvdf.x == 0
        w.write_zero_mv_data(); // mvdf.y == 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 2, 1, 1, 1, 2);
        driver.start_row();
        let mb = driver
            .decode_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                0,
                0,
                false,
                DirectCoLocatedMv::TransparentOrAbsent,
            )
            .unwrap();
        assert_eq!(mb.mb_type, BVopMbType::Forward);
        assert_eq!(mb.prediction_mode, BVopPredictionMode::ForwardOnly);
        assert_eq!(mb.mvs[0].forward, MotionVector { x: 0, y: 0 });
        assert_eq!(mb.mvs[0].backward, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn co_located_skipped_modb_not_one_forces_forward_zero() {
        // co-located skipped + modb != "1" → forward zero per §7.6.9.6.
        // Use modb "01", mb_type forward, mvdf two zeros (consumed +
        // discarded). The reconstruction is overridden to forward-zero.
        let vol = make_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2);
        w.write_bits(0b0001, 4);
        w.write_zero_mv_data();
        w.write_zero_mv_data();
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        driver.start_row();
        let mb = driver
            .decode_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                0,
                0,
                true,
                DirectCoLocatedMv::TransparentOrAbsent,
            )
            .unwrap();
        assert_eq!(mb.mb_type, BVopMbType::Forward);
        assert_eq!(mb.prediction_mode, BVopPredictionMode::ForwardOnly);
        assert_eq!(mb.mvs[0].forward, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn co_located_skipped_modb_one_forces_direct_zero_delta() {
        // co-located skipped + modb == "1" → direct mode, zero delta.
        let vol = make_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        driver.start_row();
        let mb = driver
            .decode_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                0,
                0,
                true,
                DirectCoLocatedMv::Mv(MotionVector { x: 6, y: -6 }),
            )
            .unwrap();
        assert_eq!(mb.mb_type, BVopMbType::Direct);
        assert_eq!(mb.prediction_mode, BVopPredictionMode::Direct);
        // With a zero delta the §7.6.9.5.2 formulas reduce to
        // MVF = TRB*MV/TRD, MVB = (TRB-TRD)*MV/TRD. TRB=1, TRD=2, MV=(6,-6):
        // MVF = (1*6)/2 = 3, (1*-6)/2 = -3.
        // MVB = ((1-2)*6)/2 = -3, ((1-2)*-6)/2 = 3.
        assert_eq!(mb.mvs[0].forward, MotionVector { x: 3, y: -3 });
        assert_eq!(mb.mvs[0].backward, MotionVector { x: -3, y: 3 });
    }

    #[test]
    fn start_row_resets_predictors() {
        let mut driver = BVopMvDriver::new(2, 2, 2, 2, 1, 2);
        driver.predictor.forward = MotionVector { x: 9, y: 9 };
        driver.predictor.backward = MotionVector { x: -9, y: -9 };
        driver.start_row();
        assert_eq!(driver.predictor.forward, MotionVector { x: 0, y: 0 });
        assert_eq!(driver.predictor.backward, MotionVector { x: 0, y: 0 });
    }

    /// Table B.12 codeword for `mv_data == 2` (`0010`, 4 bits). With
    /// `vop_fcode == 1` the reconstructed differential equals `mv_data`,
    /// so a forward body of two of these is a `(2, 2)` delta.
    fn write_mv_data_two(w: &mut BitWriter) {
        w.write_bits(0b0010, 4);
    }

    #[test]
    fn decode_vop_motion_threads_forward_predictor_within_row() {
        // 1×2 VOP, two forward MBs, each coding mvdf == (2, 2) with
        // fcode 1. §7.6.8: MB0 reconstructs (0,0)+(2,2) = (2,2) and sets
        // the forward predictor to (2,2); MB1 reconstructs (2,2)+(2,2) =
        // (4,4). This proves the running predictor threads across the
        // row rather than resetting per macroblock.
        let vol = make_vol();
        let mut w = BitWriter::new();
        for _ in 0..2 {
            w.write_bits(0b01, 2); // modb "01"
            w.write_bits(0b0001, 4); // mb_type forward
            write_mv_data_two(&mut w); // mvdf.x == 2
            write_mv_data_two(&mut w); // mvdf.y == 2
        }
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 2, 1, 1, 1, 2);
        let mbs = driver
            .decode_vop_motion(&mut br, &vol, VopCodingType::B, |_, _| {
                CoLocatedAnchor::default()
            })
            .unwrap();
        assert_eq!(mbs.len(), 2);
        assert_eq!(mbs[0].mvs[0].forward, MotionVector { x: 2, y: 2 });
        assert_eq!(mbs[1].mvs[0].forward, MotionVector { x: 4, y: 4 });
    }

    #[test]
    fn decode_vop_motion_resets_predictor_between_rows() {
        // 2×1 VOP, two forward MBs in separate rows, each mvdf == (2,2).
        // §7.6.8 resets the predictor at each row start, so BOTH MBs
        // reconstruct (0,0)+(2,2) = (2,2) — the row-1 MB does not see
        // the row-0 predictor.
        let vol = make_vol();
        let mut w = BitWriter::new();
        for _ in 0..2 {
            w.write_bits(0b01, 2);
            w.write_bits(0b0001, 4);
            write_mv_data_two(&mut w);
            write_mv_data_two(&mut w);
        }
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(2, 1, 1, 1, 1, 2);
        let mbs = driver
            .decode_vop_motion(&mut br, &vol, VopCodingType::B, |_, _| {
                CoLocatedAnchor::default()
            })
            .unwrap();
        assert_eq!(mbs.len(), 2);
        assert_eq!(mbs[0].mvs[0].forward, MotionVector { x: 2, y: 2 });
        assert_eq!(mbs[1].mvs[0].forward, MotionVector { x: 2, y: 2 });
    }

    #[test]
    fn decode_vop_motion_per_mb_co_located_anchor_consulted() {
        // 1×2 VOP: MB0 is a direct MB (modb "1", no body) with a skipped
        // co-located anchor → direct/zero-delta from MV (8,0): TRB=1,
        // TRD=2 → MVF.x = (1*8)/2 = 4. MB1 is modb "1" with a
        // NON-skipped transparent anchor → also direct, MV zero → all
        // zero. Proves the closure's per-MB anchor reaches the driver.
        let vol = make_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // MB0 modb "1"
        w.write_bits(0b1, 1); // MB1 modb "1"
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 2, 1, 1, 1, 2);
        let mbs = driver
            .decode_vop_motion(&mut br, &vol, VopCodingType::B, |_, col| {
                if col == 0 {
                    CoLocatedAnchor {
                        skipped: true,
                        mv: DirectCoLocatedMv::Mv(MotionVector { x: 8, y: 0 }),
                    }
                } else {
                    CoLocatedAnchor::default()
                }
            })
            .unwrap();
        assert_eq!(mbs.len(), 2);
        assert_eq!(mbs[0].mb_type, BVopMbType::Direct);
        assert_eq!(mbs[0].mvs[0].forward, MotionVector { x: 4, y: 0 });
        assert_eq!(mbs[1].mvs[0].forward, MotionVector { x: 0, y: 0 });
    }

    #[test]
    fn reconstruct_forward_zero_mv_zero_residual_copies_forward_anchor() {
        // A forward-only B-MB with a zero MV + zero residual is a pure
        // §7.6.9.2 copy of the co-located forward-anchor samples. Build a
        // 16×16 luma / 8×8 chroma uniform reference and verify the
        // §7.6.9→§7.3 bridge reproduces it exactly.
        use crate::block::InterMacroblock;
        use crate::bvop_prediction::BVopSampleMode;
        use crate::half_sample::ReferenceVop;

        let luma_plane = vec![137u8; 16 * 16];
        let chroma_plane = vec![88u8; 8 * 8];
        let fwd_luma = ReferenceVop::new(&luma_plane, 16, 16).unwrap();
        let fwd_cb = ReferenceVop::new(&chroma_plane, 8, 8).unwrap();
        let fwd_cr = ReferenceVop::new(&chroma_plane, 8, 8).unwrap();
        // Backward planes are unused for forward-only mode; reuse the
        // same buffers so the references are valid.
        let anchors = BVopAnchorPlanes {
            forward_luma: &fwd_luma,
            backward_luma: &fwd_luma,
            forward_cb: &fwd_cb,
            backward_cb: &fwd_cb,
            forward_cr: &fwd_cr,
            backward_cr: &fwd_cr,
        };

        let zero = MotionVector { x: 0, y: 0 };
        let decode = BVopMbDecode {
            mb_type: BVopMbType::Forward,
            prediction_mode: BVopPredictionMode::ForwardOnly,
            cbpb: None,
            dbquant_delta: None,
            mvs: [BVopMvPair {
                forward: zero,
                backward: zero,
            }; MB_SUB_BLOCKS],
            forward_chroma_mv: zero,
            backward_chroma_mv: zero,
        };

        let residual = InterMacroblock {
            luma: [[0i32; 16]; 16],
            cb: [[0i32; 8]; 8],
            cr: [[0i32; 8]; 8],
        };

        let recon = decode.reconstruct(&anchors, &residual, 0, 0, 0, BVopSampleMode::HalfPel, 8);
        for row in recon.luma {
            for px in row {
                assert_eq!(px, 137);
            }
        }
        for row in recon.cb {
            for px in row {
                assert_eq!(px, 88);
            }
        }
        for row in recon.cr {
            for px in row {
                assert_eq!(px, 88);
            }
        }
    }

    fn texture_params() -> BVopTextureParams {
        BVopTextureParams {
            base_quantiser_scale: 8,
            max_quantiser_scale: 31,
            bits_per_pixel: 8,
            quant_type: false,
        }
    }

    /// `decode_vop` threads the §7.4 residual after the motion bodies:
    /// a single forward MB whose `modb == "00"` codes a non-zero `cbpb`
    /// (block 0 coded) carries a residual on the wire that the driver
    /// decodes into the macroblock's luma top-left 8×8, leaving the
    /// other blocks zero. The reader ends exactly at the texture's end.
    #[test]
    fn decode_vop_threads_residual_after_motion() {
        let vol = make_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb "00" — mb_type + cbpb present
        w.write_bits(0b0001, 4); // mb_type forward
        w.write_bits(0b10_0000, 6); // cbpb — only block 0 coded
                                    // mb_type carries dbquant only when non-direct; forward does, but
                                    // dbquant is gated on cbpb != 0, so it IS present here. Write the
                                    // Table 6-33 "no change" code (`0`).
        w.write_bits(0b0, 1); // dbquant code 0 → delta 0
        w.write_zero_mv_data_pair(); // mvdf == (0, 0)
                                     // Texture: block 0 one positive inter EVENT (LAST=1,RUN=0,LVL=1).
        w.write_bits(0b0111, 4);
        w.write_bits(0b0, 1); // sign +
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let mbs = driver
            .decode_vop(&mut br, &vol, VopCodingType::B, texture_params(), |_, _| {
                CoLocatedAnchor::default()
            })
            .unwrap();
        assert_eq!(mbs.len(), 1);
        let mb = &mbs[0];
        assert_eq!(mb.motion.mb_type, BVopMbType::Forward);
        assert_eq!(mb.quantiser_scale, 8); // base + delta 0
                                           // Block 0 carries the residual near 2; everything else is zero.
        for y in 0..8 {
            for x in 0..8 {
                assert!(
                    (mb.residual.luma[y][x] - 2).abs() <= 1,
                    "block 0 residual ({y},{x}) = {}",
                    mb.residual.luma[y][x]
                );
            }
        }
        for y in 0..8 {
            for x in 8..16 {
                assert_eq!(mb.residual.luma[y][x], 0);
            }
        }
        for y in 8..16 {
            for x in 0..16 {
                assert_eq!(mb.residual.luma[y][x], 0);
            }
        }
        assert!(mb.residual.cb.iter().all(|r| r.iter().all(|&p| p == 0)));
        assert!(mb.residual.cr.iter().all(|r| r.iter().all(|&p| p == 0)));
    }

    /// `decode_vop` applies each macroblock's `dbquant` (Table 6-33,
    /// §6.3.6) to the running quantiser scale, clipped to
    /// `[1, max_quantiser_scale]`, and threads it across macroblocks
    /// within the VOP.
    #[test]
    fn decode_vop_threads_running_quantiser() {
        let vol = make_vol();
        let mut w = BitWriter::new();
        // MB0 — forward, cbpb block 0 coded, dbquant code "11" → +2.
        w.write_bits(0b00, 2);
        w.write_bits(0b0001, 4);
        w.write_bits(0b10_0000, 6);
        w.write_bits(0b11, 2); // dbquant +2
        w.write_zero_mv_data_pair();
        w.write_bits(0b0111, 4); // block-0 event
        w.write_bits(0b0, 1);
        // MB1 — forward, cbpb block 0 coded, dbquant code "10" → -2.
        w.write_bits(0b00, 2);
        w.write_bits(0b0001, 4);
        w.write_bits(0b10_0000, 6);
        w.write_bits(0b10, 2); // dbquant -2
        w.write_zero_mv_data_pair();
        w.write_bits(0b0111, 4);
        w.write_bits(0b0, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 2, 1, 1, 1, 2);
        let mbs = driver
            .decode_vop(&mut br, &vol, VopCodingType::B, texture_params(), |_, _| {
                CoLocatedAnchor::default()
            })
            .unwrap();
        // base 8, +2 → 10, then -2 → 8.
        assert_eq!(mbs[0].quantiser_scale, 10);
        assert_eq!(mbs[1].quantiser_scale, 8);
    }

    /// A `modb == "1"` macroblock codes no cbpb (no residual); `decode_vop`
    /// yields a wholly-zero residual and consumes no texture bits.
    #[test]
    fn decode_vop_no_cbpb_zero_residual() {
        let vol = make_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb "1" → default direct, no cbpb
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let mbs = driver
            .decode_vop(&mut br, &vol, VopCodingType::B, texture_params(), |_, _| {
                CoLocatedAnchor::default()
            })
            .unwrap();
        assert_eq!(mbs[0].motion.cbpb, None);
        assert!(mbs[0]
            .residual
            .luma
            .iter()
            .all(|r| r.iter().all(|&p| p == 0)));
    }

    /// End-to-end frame decode: `decode_vop` decodes a single forward
    /// B-VOP macroblock (zero MV, one coded luma block) and the
    /// per-macroblock [`BVopMbDecode::reconstruct`] bridge adds that
    /// residual onto a uniform forward anchor. The reconstructed luma
    /// top-left 8×8 = anchor (100) + residual (~2); every other sample =
    /// the bare anchor copy. This exercises the full §7.6.8 motion →
    /// §7.4 residual → §7.6.9/§7.3 reconstruction chain wired through the
    /// frame loop.
    #[test]
    fn decode_vop_reconstruct_forward_anchor_plus_residual() {
        use crate::bvop_prediction::BVopSampleMode;
        use crate::half_sample::ReferenceVop;

        let vol = make_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb "00" — mb_type + cbpb
        w.write_bits(0b0001, 4); // forward
        w.write_bits(0b10_0000, 6); // cbpb — block 0 coded
        w.write_bits(0b0, 1); // dbquant code 0 → delta 0
        w.write_zero_mv_data_pair(); // mvdf == (0, 0)
        w.write_bits(0b0111, 4); // block-0 inter EVENT
        w.write_bits(0b0, 1); // sign +
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let mbs = driver
            .decode_vop(&mut br, &vol, VopCodingType::B, texture_params(), |_, _| {
                CoLocatedAnchor::default()
            })
            .unwrap();

        // Uniform forward anchor; backward planes reuse the same buffers
        // (forward-only mode never samples them).
        let luma_plane = vec![100u8; 16 * 16];
        let chroma_plane = vec![100u8; 8 * 8];
        let fwd_luma = ReferenceVop::new(&luma_plane, 16, 16).unwrap();
        let fwd_cb = ReferenceVop::new(&chroma_plane, 8, 8).unwrap();
        let fwd_cr = ReferenceVop::new(&chroma_plane, 8, 8).unwrap();
        let anchors = BVopAnchorPlanes {
            forward_luma: &fwd_luma,
            backward_luma: &fwd_luma,
            forward_cb: &fwd_cb,
            backward_cb: &fwd_cb,
            forward_cr: &fwd_cr,
            backward_cr: &fwd_cr,
        };

        let recon = mbs[0].motion.reconstruct(
            &anchors,
            &mbs[0].residual,
            0,
            0,
            0,
            BVopSampleMode::HalfPel,
            8,
        );

        // Block 0 (luma[0..8][0..8]) = anchor 100 + residual ~2.
        for y in 0..8 {
            for x in 0..8 {
                let px = recon.luma[y][x] as i32;
                assert!(
                    (px - 102).abs() <= 1,
                    "block 0 reconstructed ({y},{x}) = {px}, expected near 102"
                );
            }
        }
        // The other three luma blocks have zero residual → bare copy 100.
        for y in 0..8 {
            for x in 8..16 {
                assert_eq!(recon.luma[y][x], 100);
            }
        }
        for y in 8..16 {
            for x in 0..16 {
                assert_eq!(recon.luma[y][x], 100);
            }
        }
        // Chroma residual is zero → bare anchor copy.
        for row in recon.cb {
            for px in row {
                assert_eq!(px, 100);
            }
        }
        for row in recon.cr {
            for px in row {
                assert_eq!(px, 100);
            }
        }
    }

    // -----------------------------------------------------------------
    // §7.7.2.2 interlaced field-prediction B-VOP path
    // -----------------------------------------------------------------

    fn make_interlaced_vol() -> VolHeader {
        VolHeader {
            interlaced: true,
            ..make_vol()
        }
    }

    use crate::bvop_field_motion::BVopFieldReferences;
    use crate::half_sample::ReferenceVop;

    /// A flat `width × height` reference plane of constant value.
    fn flat_plane(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    #[test]
    fn decode_field_forward_macroblock_reconstructs_pixels() {
        // Interlaced VOL, field-forward MB. modb "01" → mb_type "0001"
        // → interlaced_information (no dct_type since cbp == 0;
        // field_prediction = 1 + forward pair refs) → two field bodies
        // (forward-top, forward-bottom), each (0, 0).
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb "01"
        w.write_bits(0b0001, 4); // mb_type forward
        w.write_bits(0b1, 1); // field_prediction = 1
        w.write_bits(0b0, 1); // forward_top_field_reference → Top
        w.write_bits(0b0, 1); // forward_bottom_field_reference → Top
        w.write_zero_mv_data_pair(); // MVD[0] (top): (0, 0)
        w.write_zero_mv_data_pair(); // MVD[1] (bottom): (0, 0)
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);

        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        driver.start_row();
        let decode = driver
            .decode_field_macroblock(&mut br, &vol, VopCodingType::B, 0, 0)
            .unwrap();
        assert!(matches!(decode.mode, BVopFieldMode::Forward(_)));
        assert_eq!(decode.cbpb, None);

        // Reconstruct against a flat forward reference (200); zero MV +
        // zero residual ⇒ exact copy 200 across the macroblock.
        let fwd = flat_plane(48, 48, 200);
        let bak = flat_plane(48, 48, 50);
        let fwd_c = flat_plane(24, 24, 200);
        let bak_c = flat_plane(24, 24, 50);
        let fl = ReferenceVop::new(&fwd, 48, 48).unwrap();
        let bl = ReferenceVop::new(&bak, 48, 48).unwrap();
        let fc = ReferenceVop::new(&fwd_c, 24, 24).unwrap();
        let bc = ReferenceVop::new(&bak_c, 24, 24).unwrap();
        let refs = BVopFieldReferences {
            forward_luma: &fl,
            forward_cb: &fc,
            forward_cr: &fc,
            backward_luma: &bl,
            backward_cb: &bc,
            backward_cr: &bc,
        };
        let residual = InterMacroblock {
            luma: [[0i32; 16]; 16],
            cb: [[0i32; 8]; 8],
            cr: [[0i32; 8]; 8],
        };
        let recon = decode.reconstruct(&refs, &residual, 16, 16, 0, FieldSampleMode::HalfSample, 8);
        for row in recon.luma {
            for px in row {
                assert_eq!(px, 200, "forward-only must copy the forward ref");
            }
        }
    }

    #[test]
    fn decode_field_bidirectional_averages_references() {
        // Field-bidirectional MB: modb "01" → mb_type "01" → field
        // prediction with forward + backward pairs → four field bodies.
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb "01"
        w.write_bits(0b01, 2); // mb_type interpolated
        w.write_bits(0b1, 1); // field_prediction = 1
        w.write_bits(0b0, 1); // forward_top → Top
        w.write_bits(0b0, 1); // forward_bottom → Top
        w.write_bits(0b0, 1); // backward_top → Top
        w.write_bits(0b0, 1); // backward_bottom → Top
        w.write_zero_mv_data_pair(); // MVD[0] forward-top
        w.write_zero_mv_data_pair(); // MVD[1] forward-bottom
        w.write_zero_mv_data_pair(); // MVD[2] backward-top
        w.write_zero_mv_data_pair(); // MVD[3] backward-bottom
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);

        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        driver.start_row();
        let decode = driver
            .decode_field_macroblock(&mut br, &vol, VopCodingType::B, 0, 0)
            .unwrap();
        assert!(matches!(decode.mode, BVopFieldMode::Bidirectional(_)));

        // Forward ref 100, backward 200 ⇒ average 150.
        let fwd = flat_plane(48, 48, 100);
        let bak = flat_plane(48, 48, 200);
        let fwd_c = flat_plane(24, 24, 100);
        let bak_c = flat_plane(24, 24, 200);
        let fl = ReferenceVop::new(&fwd, 48, 48).unwrap();
        let bl = ReferenceVop::new(&bak, 48, 48).unwrap();
        let fc = ReferenceVop::new(&fwd_c, 24, 24).unwrap();
        let bc = ReferenceVop::new(&bak_c, 24, 24).unwrap();
        let refs = BVopFieldReferences {
            forward_luma: &fl,
            forward_cb: &fc,
            forward_cr: &fc,
            backward_luma: &bl,
            backward_cb: &bc,
            backward_cr: &bc,
        };
        let residual = InterMacroblock {
            luma: [[0i32; 16]; 16],
            cb: [[0i32; 8]; 8],
            cr: [[0i32; 8]; 8],
        };
        let recon = decode.reconstruct(&refs, &residual, 16, 16, 0, FieldSampleMode::HalfSample, 8);
        for row in recon.luma {
            for px in row {
                assert_eq!(px, 150);
            }
        }
    }

    #[test]
    fn field_predictor_resets_at_row_start() {
        // The field PMV bank is reset alongside the progressive predictor.
        let mut driver = BVopMvDriver::new(2, 2, 1, 1, 1, 2);
        driver
            .field_predictor
            .field_forward(MotionVector { x: 7, y: 4 }, MotionVector { x: 3, y: 2 });
        assert_ne!(
            driver
                .field_predictor
                .get(crate::bvop_field_predictor::PMV_TOP_FWD),
            MotionVector { x: 0, y: 0 }
        );
        driver.start_row();
        for slot in 0..4 {
            assert_eq!(
                driver.field_predictor.get(slot),
                MotionVector { x: 0, y: 0 }
            );
        }
    }

    // -----------------------------------------------------------------
    // §7.7.2.2 interlaced-direct B-VOP path (frame-driver wiring)
    // -----------------------------------------------------------------

    fn future_field(x: i32, y: i32, r: FieldReference) -> ColocatedFutureField {
        ColocatedFutureField {
            mv: MotionVector { x, y },
            reference_field: r,
        }
    }

    #[test]
    fn interlaced_direct_modb_one_zero_delta_derives_mvs() {
        // modb == "1" → default direct, no MVD[0] body (implicit zero).
        // Future field MVs MV[0]=(6,-6) Top, MV[1]=(6,-6) Bottom →
        // δ = (Top,Bottom) row = (0,0); TRB=2*1=2, TRD=2*2=4.
        // mvf = 2*6/4 = 3, 2*-6/4 = -3 → (3,-3); mvb (zero delta) =
        // (2-4)*6/4 = -3, (2-4)*-6/4 = 3 → (-3,3).
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb "1"
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let future = ColocatedFutureFieldMvs {
            top: future_field(6, -6, FieldReference::Top),
            bottom: future_field(6, -6, FieldReference::Bottom),
        };
        let decode = driver
            .decode_interlaced_direct_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                0,
                0,
                future,
                false,
            )
            .unwrap();
        assert_eq!(decode.mvs.forward_top, MotionVector { x: 3, y: -3 });
        assert_eq!(decode.mvs.backward_top, MotionVector { x: -3, y: 3 });
        assert_eq!(decode.mvs.forward_bottom, MotionVector { x: 3, y: -3 });
        assert_eq!(decode.mvs.backward_bottom, MotionVector { x: -3, y: 3 });
        assert_eq!(decode.forward_top_ref, FieldReference::Top);
        assert_eq!(decode.forward_bottom_ref, FieldReference::Bottom);
        assert_eq!(decode.cbpb, None);
    }

    #[test]
    fn interlaced_direct_explicit_mvd_applied() {
        // modb == "01" (mb_type present), mb_type "1" (direct, Table B.4),
        // then a single direct MVD[0] body = (1, 0) (mv_data 2 then 0
        // under f_code 1 ⇒ but write (1,0) via mv_data 1).
        // mv_data codeword for 1 is "010" (Table B.12); with f_code 1 the
        // reconstructed component is the mv_data value (1).
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb "01"
        w.write_bits(0b1, 1); // mb_type "1" → direct
                              // MVD[0] = (1, 0): mv_data 1 then mv_data 0.
        w.write_bits(0b010, 3); // mv_data == 1
        w.write_zero_mv_data(); // mv_data == 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        // Future MV[0]=(8,4) Top, MV[1]=(8,4) Bottom → δ=(0,0); TRB=2, TRD=4.
        // mvf.x = 2*8/4 + 1 = 5; mvf.y = 2*4/4 + 0 = 2.
        // mvb.x (delta.x != 0) = mvf.x - MV.x = 5 - 8 = -3.
        // mvb.y (delta.y == 0) = (2-4)*4/4 = -2.
        let future = ColocatedFutureFieldMvs {
            top: future_field(8, 4, FieldReference::Top),
            bottom: future_field(8, 4, FieldReference::Bottom),
        };
        let decode = driver
            .decode_interlaced_direct_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                0,
                0,
                future,
                false,
            )
            .unwrap();
        assert_eq!(decode.mvs.forward_top, MotionVector { x: 5, y: 2 });
        assert_eq!(decode.mvs.backward_top, MotionVector { x: -3, y: -2 });
    }

    #[test]
    fn interlaced_direct_rejects_non_direct() {
        // A forward macroblock routed to the interlaced-direct method is
        // rejected (it belongs on decode_field_macroblock / decode_macroblock).
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb "01"
        w.write_bits(0b0001, 4); // mb_type forward
        w.write_bits(0b0, 1); // field_prediction = 0 (frame)
        w.write_zero_mv_data_pair();
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let future = ColocatedFutureFieldMvs {
            top: future_field(0, 0, FieldReference::Top),
            bottom: future_field(0, 0, FieldReference::Bottom),
        };
        let err = driver
            .decode_interlaced_direct_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                0,
                0,
                future,
                false,
            )
            .unwrap_err();
        assert_eq!(err, BVopMvDriverError::FieldPredictionUnsupported);
    }

    #[test]
    fn interlaced_direct_out_of_bounds_rejected() {
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let future = ColocatedFutureFieldMvs {
            top: future_field(0, 0, FieldReference::Top),
            bottom: future_field(0, 0, FieldReference::Bottom),
        };
        let err = driver
            .decode_interlaced_direct_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                3,
                3,
                future,
                false,
            )
            .unwrap_err();
        assert_eq!(
            err,
            BVopMvDriverError::OutOfBounds {
                mb_row: 3,
                mb_col: 3
            }
        );
    }

    #[test]
    fn interlaced_direct_reconstructs_pixels_zero_mv() {
        // All-zero derived MVs (zero future MV + zero delta) ⇒ each output
        // field copies its reference; forward 90 / backward 150 average to
        // (90 + 150 + 1) >> 1 = 120, plus a zero residual.
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb "1" → direct, zero delta
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let future = ColocatedFutureFieldMvs {
            top: future_field(0, 0, FieldReference::Top),
            bottom: future_field(0, 0, FieldReference::Bottom),
        };
        let decode = driver
            .decode_interlaced_direct_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                0,
                0,
                future,
                false,
            )
            .unwrap();
        assert_eq!(decode.mvs.forward_top, MotionVector { x: 0, y: 0 });

        let fwd = flat_plane(48, 48, 90);
        let bak = flat_plane(48, 48, 150);
        let fwd_c = flat_plane(24, 24, 90);
        let bak_c = flat_plane(24, 24, 150);
        let fl = ReferenceVop::new(&fwd, 48, 48).unwrap();
        let bl = ReferenceVop::new(&bak, 48, 48).unwrap();
        let fc = ReferenceVop::new(&fwd_c, 24, 24).unwrap();
        let bc = ReferenceVop::new(&bak_c, 24, 24).unwrap();
        let refs = BVopFieldReferences {
            forward_luma: &fl,
            forward_cb: &fc,
            forward_cr: &fc,
            backward_luma: &bl,
            backward_cb: &bc,
            backward_cr: &bc,
        };
        let residual = InterMacroblock {
            luma: [[0i32; 16]; 16],
            cb: [[0i32; 8]; 8],
            cr: [[0i32; 8]; 8],
        };
        let recon = decode.reconstruct(&refs, &residual, 16, 16, 8);
        for row in recon.luma {
            for px in row {
                assert_eq!(px, 120);
            }
        }
        for row in recon.cb {
            for px in row {
                assert_eq!(px, 120);
            }
        }
    }

    #[test]
    fn colocated_future_from_field_motion_bridges_reference_chain() {
        // The reference-frame-chain bridge: a decoded P-VOP field-MV pair
        // + the future MB's field references build the ColocatedFutureFieldMvs.
        use crate::field_motion::FieldMotionVectors;
        let field_mvs = FieldMotionVectors {
            top: MotionVector { x: 6, y: -6 },
            bottom: MotionVector { x: 4, y: 2 },
        };
        let future = ColocatedFutureFieldMvs::from_field_motion(
            field_mvs,
            FieldReference::Bottom,
            FieldReference::Top,
        );
        assert_eq!(future.top.mv, MotionVector { x: 6, y: -6 });
        assert_eq!(future.top.reference_field, FieldReference::Bottom);
        assert_eq!(future.bottom.mv, MotionVector { x: 4, y: 2 });
        assert_eq!(future.bottom.reference_field, FieldReference::Top);

        // Feed straight into the interlaced-direct decode. modb "1" → zero
        // delta. Top ref Bottom, bottom ref Top, tff=0 → δ=(1,-1).
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 2, 4);
        let decode = driver
            .decode_interlaced_direct_macroblock(
                &mut br,
                &vol,
                VopCodingType::B,
                0,
                0,
                future,
                false,
            )
            .unwrap();
        // δ top = 1 → TRB=2*2+1=5, TRD=2*4+1=9; mvf top.x = 5*6/9 = 3.
        assert_eq!(decode.mvs.forward_top.x, 3);
        assert_eq!(decode.forward_top_ref, FieldReference::Bottom);
    }

    // -----------------------------------------------------------------
    // Unified interlaced B-VOP dispatch (decode_interlaced_macroblock)
    // -----------------------------------------------------------------

    fn plain_anchor() -> BVopInterlacedAnchor {
        BVopInterlacedAnchor {
            progressive: CoLocatedAnchor::default(),
            future_field_mvs: None,
            top_field_first: false,
        }
    }

    #[test]
    fn dispatch_routes_progressive_forward() {
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb "01"
        w.write_bits(0b0001, 4); // forward
        w.write_bits(0b0, 1); // field_prediction = 0 → frame
        write_mv_data_two(&mut w);
        write_mv_data_two(&mut w);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        driver.start_row();
        let mb = driver
            .decode_interlaced_macroblock(&mut br, &vol, VopCodingType::B, 0, 0, plain_anchor())
            .unwrap();
        match mb {
            BVopInterlacedMb::Progressive(d) => {
                assert_eq!(d.mb_type, BVopMbType::Forward);
                assert_eq!(d.mvs[0].forward, MotionVector { x: 2, y: 2 });
            }
            other => panic!("expected Progressive, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_routes_field_predicted() {
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb "01"
        w.write_bits(0b0001, 4); // forward
        w.write_bits(0b1, 1); // field_prediction = 1
        w.write_bits(0b0, 1); // forward_top → Top
        w.write_bits(0b0, 1); // forward_bottom → Top
        w.write_zero_mv_data_pair();
        w.write_zero_mv_data_pair();
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        driver.start_row();
        let mb = driver
            .decode_interlaced_macroblock(&mut br, &vol, VopCodingType::B, 0, 0, plain_anchor())
            .unwrap();
        assert!(matches!(
            mb,
            BVopInterlacedMb::Field(d) if matches!(d.mode, BVopFieldMode::Forward(_))
        ));
    }

    #[test]
    fn dispatch_routes_interlaced_direct_when_future_field_predicted() {
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb "1" → default direct
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let anchor = BVopInterlacedAnchor {
            progressive: CoLocatedAnchor::default(),
            future_field_mvs: Some(ColocatedFutureFieldMvs {
                top: future_field(6, -6, FieldReference::Top),
                bottom: future_field(6, -6, FieldReference::Bottom),
            }),
            top_field_first: false,
        };
        let mb = driver
            .decode_interlaced_macroblock(&mut br, &vol, VopCodingType::B, 0, 0, anchor)
            .unwrap();
        match mb {
            BVopInterlacedMb::InterlacedDirect(d) => {
                // δ=(0,0), TRB=2, TRD=4 → mvf top = (3, -3).
                assert_eq!(d.mvs.forward_top, MotionVector { x: 3, y: -3 });
            }
            other => panic!("expected InterlacedDirect, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_routes_progressive_direct_when_no_future_field_mvs() {
        // A Direct MB with no future field MVs (future MB not
        // field-predicted) resolves to progressive direct mode.
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb "1" → default direct
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let anchor = BVopInterlacedAnchor {
            progressive: CoLocatedAnchor {
                skipped: false,
                mv: DirectCoLocatedMv::Mv(MotionVector { x: 6, y: -6 }),
            },
            future_field_mvs: None,
            top_field_first: false,
        };
        let mb = driver
            .decode_interlaced_macroblock(&mut br, &vol, VopCodingType::B, 0, 0, anchor)
            .unwrap();
        match mb {
            BVopInterlacedMb::Progressive(d) => {
                assert_eq!(d.mb_type, BVopMbType::Direct);
                // Progressive direct: TRB=1, TRD=2, MV=(6,-6) → MVF=(3,-3).
                assert_eq!(d.mvs[0].forward, MotionVector { x: 3, y: -3 });
            }
            other => panic!("expected Progressive direct, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_co_located_skipped_overrides_to_progressive() {
        // A skipped co-located anchor forces progressive direct/forward
        // even in an interlaced VOL, regardless of future field MVs.
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb "1"
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let anchor = BVopInterlacedAnchor {
            progressive: CoLocatedAnchor {
                skipped: true,
                mv: DirectCoLocatedMv::Mv(MotionVector { x: 8, y: 0 }),
            },
            future_field_mvs: Some(ColocatedFutureFieldMvs {
                top: future_field(6, -6, FieldReference::Top),
                bottom: future_field(6, -6, FieldReference::Bottom),
            }),
            top_field_first: false,
        };
        let mb = driver
            .decode_interlaced_macroblock(&mut br, &vol, VopCodingType::B, 0, 0, anchor)
            .unwrap();
        // §7.6.9.6: modb "1" + skipped → direct, zero delta, from MV (8,0).
        // TRB=1, TRD=2 → MVF.x = (1*8)/2 = 4.
        match mb {
            BVopInterlacedMb::Progressive(d) => {
                assert_eq!(d.mb_type, BVopMbType::Direct);
                assert_eq!(d.mvs[0].forward, MotionVector { x: 4, y: 0 });
            }
            other => panic!("expected Progressive (skip override), got {other:?}"),
        }
    }

    #[test]
    fn dispatch_out_of_bounds_rejected() {
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        let err = driver
            .decode_interlaced_macroblock(&mut br, &vol, VopCodingType::B, 9, 9, plain_anchor())
            .unwrap_err();
        assert_eq!(
            err,
            BVopMvDriverError::OutOfBounds {
                mb_row: 9,
                mb_col: 9
            }
        );
    }

    #[test]
    fn decode_field_rejects_frame_predicted_macroblock() {
        // A frame-predicted MB (field_prediction == 0) in an interlaced
        // VOL must route to decode_macroblock, not decode_field_macroblock.
        let vol = make_interlaced_vol();
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb "01"
        w.write_bits(0b0001, 4); // mb_type forward
        w.write_bits(0b0, 1); // field_prediction = 0
        w.write_zero_mv_data_pair(); // frame mvdf (0, 0)
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let mut driver = BVopMvDriver::new(1, 1, 1, 1, 1, 2);
        driver.start_row();
        let err = driver
            .decode_field_macroblock(&mut br, &vol, VopCodingType::B, 0, 0)
            .unwrap_err();
        assert_eq!(err, BVopMvDriverError::FieldPredictionUnsupported);
    }
}

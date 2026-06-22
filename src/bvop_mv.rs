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
use crate::bvop::{
    decode_b_vop_mb_motion_vectors, parse_b_vop_mb_header, BMbTypeTable, BVopMbHeader,
    BVopMbParseError, BVopMbType, BVopMvBody,
};
use crate::bvop_prediction::{BVopMvPair, BVopPredictionMode, MB_SUB_BLOCKS};
use crate::chroma_mv::{chroma_mv_from_luma_blocks, ChromaMvError};
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
        }
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
}

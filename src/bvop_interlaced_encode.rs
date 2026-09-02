//! Rectangular **interlaced B-VOP encoder** — the §7.7.2.2 mode set
//! over the progressive B encoder's machinery.
//!
//! An interlaced B-VOP macroblock chooses among
//!
//! * the progressive §7.6.9 modes — direct (when the co-located future
//!   macroblock is frame-predicted), forward, backward, interpolated —
//!   with their motion bodies coded against the Table 7-14 four-PMV
//!   bank exactly as the decoder's driver reads them (a frame vector
//!   uses `PMV[0]` / `PMV[2]` and updates both slots of its direction,
//!   Table 7-15);
//! * the §7.7.2.2 **field** modes — field forward / backward /
//!   bidirectional: one vector per output field against the chosen
//!   reference field parity (the §6.3.6.3 `*_field_reference` bits),
//!   each field body coded against its own bank slot
//!   (`MVx = PMVx + MVDx`, `MVy = 2 * (PMVy / 2 + MVDy)`);
//! * **interlaced direct** (§7.7.2.2 last block) when the co-located
//!   future macroblock was itself field-predicted: the four field
//!   vectors derive from its field pair, the single `MVD[0]` (searched
//!   over a small window, coded with `f_code == 1`) and the Table 7-16
//!   δ-corrected field-period `TRB[i]` / `TRD[i]`.
//!
//! Every candidate is scored by the SAD of the prediction the crate's
//! own decoder machinery forms for it ([`BVopMbDecode::reconstruct`],
//! the `bvop_field_motion` predictors over a copy of the bank,
//! [`interlaced_direct_prediction`]), so the decision measures exactly
//! what a conformant decoder reconstructs. The residual takes the
//! §7.7.1 `dct_type` election (field DCT when same-field lines
//! correlate better) and the §6.2.6 emission writes the
//! `interlaced_information()` body between `dbquant` and the motion
//! bodies. The finished VOP decodes back through
//! [`decode_b_vop_interlaced_macroblocks`] +
//! [`assemble_b_vop_interlaced_frame`], closing the loop.
//!
//! **Ecosystem-compat emission** (`ecosystem_compat == true`): the
//! §7.7.2.2 interlaced-direct derivation is the one clause the
//! deployed decoder ecosystem is observed to read differently from
//! the printed text (`crate::compat`, divergence 1). A macroblock
//! whose co-located future macroblock is field-predicted is then
//! never coded in direct mode — the stream stays inside the subset
//! both readings decode identically, at a small rate cost. The
//! default is the spec-literal tool set.
//!
//! Provenance: §6.2.6 / §6.2.6.3 syntax, §6.3.6.3 semantics,
//! §7.6.8–§7.6.9 and §7.7.2.2 (Tables 7-14 / 7-15 / 7-16) of ISO/IEC
//! 14496-2:2004 (3rd edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`,
//! via the crate's decoder transcriptions (with the in-repo errata
//! E1 / E2 the decoder applies). No third-party source was consulted.

use crate::bitreader::BitReader;
use crate::bitwriter::BitWriter;
use crate::block::{nonintra_quant_matrix, InterMacroblock};
use crate::bvop::BVopMbType;
use crate::bvop_encode::{
    candidate_sad, direct_candidate, one_mv_candidate, put_b_mb_type, write_b_vop_header,
    ModeCandidate, DIRECT_SEARCH, INTERP_MODE_BIAS, ONE_MV_MODE_BIAS,
};
use crate::bvop_field_direct::{interlaced_direct_mvs, interlaced_direct_prediction};
use crate::bvop_field_motion::{
    field_backward_prediction, field_bidirectional_prediction, field_forward_prediction,
    BVopFieldReferences, FieldMvDeltas, FieldReferenceFlags, FieldSampleMode,
};
use crate::bvop_field_predictor::{
    FieldPmvBank, PMV_BOT_BWD, PMV_BOT_FWD, PMV_TOP_BWD, PMV_TOP_FWD,
};
use crate::bvop_mv::{BVopAnchorPlanes, ColocatedFutureFieldMvs};
use crate::bvop_prediction::BVopSampleMode;
use crate::field_encode::{estimate_field_motion, field_mv_differential, FieldEstimate};
use crate::framestore::{DecodedFrame, FrameStore};
use crate::interlaced_information::FieldReference;
use crate::ivop_encode::{elect_field_dct, field_dct_luma, EncoderConfig, FrameView};
use crate::motion::{DirectCoLocatedMv, MotionVector, MotionVectorDelta};
use crate::pvop_encode::{
    estimate_motion, inter_scan, macroblock_residual, quantise_inter_residual, sample_mode_of,
    source_luma_mb,
};
use crate::pvop_mv::PvopMbMotion;
use crate::reconstruct::InterPredictionMacroblock;
use crate::texture::TcoefTable;
use crate::vlc_encode::{put_ac_events, put_motion_vector};
use crate::vop::{parse_vop_header_body, vop_time_increment_bits, VopCodingType, VopContext};
use crate::vop_decode::{decode_b_vop_interlaced_macroblocks, AnchorMbMotion};

const VOP_START_CODE: u32 = 0x0000_01B6;

/// SAD bias of a single-direction field mode (two motion bodies + the
/// `field_prediction` and reference bits).
const FIELD_ONE_DIR_BIAS: u32 = 128;
/// SAD bias of the field bidirectional mode (four motion bodies).
const FIELD_BIDIR_BIAS: u32 = 192;

/// Per-VOP interlaced-B encode statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BVopInterlacedEncodeStats {
    /// §6.2.6 zero-bit macroblocks (co-located future MB skipped).
    pub zero_bit: usize,
    /// `modb == "1"` macroblocks (direct, zero delta, no residual).
    pub modb_one: usize,
    /// Progressive direct macroblocks (frame-predicted co-located MB).
    pub direct: usize,
    /// §7.7.2.2 interlaced-direct macroblocks.
    pub interlaced_direct: usize,
    /// Frame forward / backward / interpolated macroblocks.
    pub frame_forward: usize,
    /// Frame backward macroblocks.
    pub frame_backward: usize,
    /// Frame interpolated macroblocks.
    pub frame_interpolated: usize,
    /// Field forward macroblocks.
    pub field_forward: usize,
    /// Field backward macroblocks.
    pub field_backward: usize,
    /// Field bidirectional macroblocks.
    pub field_bidirectional: usize,
    /// Macroblocks coded with the field DCT.
    pub field_dct: usize,
    /// Macroblocks that carried a non-zero `dbquant`.
    pub dbquant: usize,
    /// Video packets cut inside the VOP.
    pub packets: usize,
    /// Direct candidates suppressed by the ecosystem-compat emission.
    pub compat_direct_suppressed: usize,
}

/// The chosen prediction mode of one macroblock, with everything the
/// emission and the bank update need.
enum Chosen {
    /// Progressive (frame) mode — direct / forward / backward /
    /// interpolated, as the progressive encoder shapes it.
    Frame(ModeCandidate),
    /// §7.7.2.2 interlaced direct with its searched `MVD[0]`.
    InterlacedDirect(MotionVectorDelta),
    /// Field forward (`0001`) with the top / bottom estimates.
    FieldForward(FieldEstimate, FieldEstimate),
    /// Field backward (`001`).
    FieldBackward(FieldEstimate, FieldEstimate),
    /// Field bidirectional (`01`): forward pair then backward pair.
    FieldBidirectional([FieldEstimate; 4]),
}

impl Chosen {
    fn mb_type(&self) -> BVopMbType {
        match self {
            Chosen::Frame(c) => c.decode.mb_type,
            Chosen::InterlacedDirect(_) => BVopMbType::Direct,
            Chosen::FieldForward(..) => BVopMbType::Forward,
            Chosen::FieldBackward(..) => BVopMbType::Backward,
            Chosen::FieldBidirectional(_) => BVopMbType::Interpolated,
        }
    }

    fn is_field(&self) -> bool {
        matches!(
            self,
            Chosen::FieldForward(..) | Chosen::FieldBackward(..) | Chosen::FieldBidirectional(_)
        )
    }
}

/// The §7.6.9.5.1 co-located block vectors of a frame-shaped anchor
/// macroblock (mirrors the decoder's `co_located_from_motion`).
fn co_located_mvs(motion: PvopMbMotion) -> [DirectCoLocatedMv; 4] {
    match motion {
        PvopMbMotion::OneMv(mv) => [DirectCoLocatedMv::Mv(mv); 4],
        PvopMbMotion::FourMv(mvs) => mvs.map(DirectCoLocatedMv::Mv),
        PvopMbMotion::Intra | PvopMbMotion::Skipped => [DirectCoLocatedMv::TransparentOrAbsent; 4],
    }
}

/// Luma SAD of a prediction macroblock against the source.
fn sad_of(src: &[[i32; 16]; 16], pred: &InterPredictionMacroblock) -> u32 {
    let mut sad = 0u32;
    for (j, row) in src.iter().enumerate() {
        for (i, &s) in row.iter().enumerate() {
            sad += (s - pred.luma[j][i]).unsigned_abs();
        }
    }
    sad
}

/// The §7.6.8 field differentials of a `(top, bottom)` estimate pair
/// against the bank slots `(top_slot, bottom_slot)`.
fn field_deltas(
    bank: &FieldPmvBank,
    top: &FieldEstimate,
    bottom: &FieldEstimate,
    slots: (usize, usize),
) -> FieldMvDeltas {
    let (tx, ty) = field_mv_differential(top.mv, bank.get(slots.0));
    let (bx, by) = field_mv_differential(bottom.mv, bank.get(slots.1));
    FieldMvDeltas {
        top: MotionVector { x: tx, y: ty },
        bottom: MotionVector { x: bx, y: by },
    }
}

/// Encode one rectangular **interlaced** B-VOP between the two anchors
/// in `store` (the closed-loop reconstructions of the emitted anchor
/// units). `anchor_motion` is the future anchor's per-macroblock
/// motion in the interlaced shape [`AnchorMbMotion`] (as
/// [`crate::pvop_encode::reconstruct_own_p_vop_with_anchor_motion`]
/// returns it; `None` after an intra anchor) — the same input the
/// decoder's [`decode_b_vop_interlaced_macroblocks`] consumes.
/// `trb` / `trd` are the §7.6.7 frame-period temporal references.
///
/// Returns the emitted unit, the closed-loop reconstruction and the
/// mode statistics.
#[allow(clippy::too_many_arguments)]
pub fn encode_b_vop_interlaced(
    vol: &crate::vol::VolHeader,
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    store: &FrameStore,
    anchor_motion: Option<&[AnchorMbMotion]>,
    trb: i32,
    trd: i32,
    modulo_time_base: u32,
    time_increment: u16,
    qp: u32,
    ecosystem_compat: bool,
) -> (Vec<u8>, DecodedFrame, BVopInterlacedEncodeStats) {
    assert!((1..=31).contains(&qp), "vop_quant {qp} out of range");
    assert!(
        cfg.interlaced,
        "encode_b_vop_interlaced needs an interlaced VOL"
    );
    assert!(
        trd > 0 && trb > 0 && trb < trd,
        "§7.6.7 requires 0 < TRB ({trb}) < TRD ({trd})"
    );
    let (mb_width, mb_height) = cfg.mb_dimensions();
    if let Some(m) = anchor_motion {
        assert_eq!(m.len(), mb_width * mb_height, "anchor motion grid shape");
    }
    let w_inter = nonintra_quant_matrix(vol);
    let mode = sample_mode_of(vol);
    let field_mode = match mode {
        BVopSampleMode::HalfPel => FieldSampleMode::HalfSample,
        BVopSampleMode::QuarterPel { bits_per_pixel } => {
            FieldSampleMode::QuarterSample { bits_per_pixel }
        }
    };
    let scan = inter_scan(cfg);
    let (fwd_anchor, bwd_anchor) = store
        .b_vop_references()
        .expect("both anchors present before a B-VOP");
    let fwd_luma = fwd_anchor.luma_reference();
    let bwd_luma = bwd_anchor.luma_reference();
    let fwd_cb = fwd_anchor.cb_reference();
    let bwd_cb = bwd_anchor.cb_reference();
    let fwd_cr = fwd_anchor.cr_reference();
    let bwd_cr = bwd_anchor.cr_reference();
    let anchors = BVopAnchorPlanes {
        forward_luma: &fwd_luma,
        backward_luma: &bwd_luma,
        forward_cb: &fwd_cb,
        backward_cb: &bwd_cb,
        forward_cr: &fwd_cr,
        backward_cr: &bwd_cr,
    };
    let field_refs = BVopFieldReferences {
        forward_luma: &fwd_luma,
        forward_cb: &fwd_cb,
        forward_cr: &fwd_cr,
        backward_luma: &bwd_luma,
        backward_cb: &bwd_cb,
        backward_cr: &bwd_cr,
    };
    let interlace = cfg.vop_interlace().expect("interlaced VOL");
    let top_field_first = interlace.top_field_first;

    let fcode = cfg.fcode;
    let mut header = BitWriter::new();
    write_b_vop_header(
        &mut header,
        cfg.time_increment_resolution,
        modulo_time_base,
        time_increment,
        qp,
        fcode,
        Some(interlace),
        cfg.intra_dc_vlc_thr,
    );
    let mut pw = crate::packet_encode::PacketWriter::new(
        header,
        cfg.resilience,
        crate::packet_encode::PacketVopInfo {
            coding_type: VopCodingType::B,
            fcode_fwd: fcode,
            fcode_bwd: fcode,
            modulo_time_base,
            time_increment,
            time_increment_bits: vop_time_increment_bits(cfg.time_increment_resolution),
            intra_dc_vlc_thr: cfg.intra_dc_vlc_thr,
            total_macroblocks: (mb_width * mb_height) as u32,
            interlaced: true,
            sprite_trajectory: None,
        },
        crate::packet_encode::Layout::Combined,
    );

    let mut stats = BVopInterlacedEncodeStats::default();
    let vop_qp = qp;
    let mut running_qp = vop_qp;
    for mb_row in 0..mb_height {
        // §7.6.8 / Table 7-14: the four-PMV bank resets at each row
        // start (mirrors BVopMvDriver::start_row).
        let mut bank = FieldPmvBank::new();
        for mb_col in 0..mb_width {
            let idx = mb_row * mb_width + mb_col;
            if pw.maybe_cut(idx, running_qp) {
                bank.reset_row();
            }
            let co = anchor_motion.map(|m| m[idx]);

            // §6.2.6 co_located_not_coded: no bits at all.
            if matches!(co, Some(AnchorMbMotion::Frame(PvopMbMotion::Skipped))) {
                stats.zero_bit += 1;
                continue;
            }
            let (co_mvs, future_field): ([DirectCoLocatedMv; 4], Option<ColocatedFutureFieldMvs>) =
                match co {
                    None => ([DirectCoLocatedMv::TransparentOrAbsent; 4], None),
                    Some(AnchorMbMotion::Frame(m)) => (co_located_mvs(m), None),
                    Some(
                        field @ AnchorMbMotion::Field {
                            mvs,
                            top_ref,
                            bottom_ref,
                        },
                    ) => (
                        co_located_mvs(field.progressive()),
                        Some(ColocatedFutureFieldMvs::from_field_motion(
                            mvs,
                            FieldReference::from_bit(top_ref),
                            FieldReference::from_bit(bottom_ref),
                        )),
                    ),
                };

            let (mb_x, mb_y) = ((mb_col * 16) as i32, (mb_row * 16) as i32);
            let src = source_luma_mb(frame, mb_row, mb_col);

            // ---- Candidate modes -------------------------------------
            let mut best: Option<(Chosen, u32)> = None;
            let mut consider = |chosen: Chosen, cost: u32| {
                if best.as_ref().map_or(true, |(_, c)| cost < *c) {
                    best = Some((chosen, cost));
                }
            };

            // Direct: progressive over a frame-predicted co-located
            // MB, interlaced over a field-predicted one (unless the
            // ecosystem-compat emission suppresses it).
            match future_field {
                None => {
                    for dy in -DIRECT_SEARCH..=DIRECT_SEARCH {
                        for dx in -DIRECT_SEARCH..=DIRECT_SEARCH {
                            let cand = direct_candidate(
                                co_mvs,
                                MotionVectorDelta { dx, dy },
                                trb,
                                trd,
                                mode,
                            );
                            let cost = candidate_sad(&cand, &anchors, &src, mb_x, mb_y, mode);
                            consider(Chosen::Frame(cand), cost);
                        }
                    }
                }
                Some(future) if !ecosystem_compat => {
                    for dy in -DIRECT_SEARCH..=DIRECT_SEARCH {
                        for dx in -DIRECT_SEARCH..=DIRECT_SEARCH {
                            let delta = MotionVectorDelta { dx, dy };
                            let mvs = interlaced_direct_mvs(
                                future.top,
                                future.bottom,
                                MotionVector { x: dx, y: dy },
                                trb,
                                trd,
                                top_field_first,
                            );
                            let pred = interlaced_direct_prediction(
                                mvs,
                                future.top.reference_field,
                                future.bottom.reference_field,
                                &field_refs,
                                mb_x,
                                mb_y,
                                0,
                                field_mode,
                            );
                            consider(Chosen::InterlacedDirect(delta), sad_of(&src, &pred));
                        }
                    }
                }
                Some(_) => stats.compat_direct_suppressed += 1,
            }

            // Frame forward / backward / interpolated against the bank's
            // top slots (Table 7-15).
            let (mv_f, _) = estimate_motion(&src, &fwd_luma, mb_x, mb_y, mode, fcode);
            let (mv_b, _) = estimate_motion(&src, &bwd_luma, mb_x, mb_y, mode, fcode);
            let zero = MotionVector { x: 0, y: 0 };
            for cand in [
                one_mv_candidate(BVopMbType::Forward, mv_f, zero, mode),
                one_mv_candidate(BVopMbType::Backward, zero, mv_b, mode),
                one_mv_candidate(BVopMbType::Interpolated, mv_f, mv_b, mode),
            ] {
                let bias = if cand.decode.mb_type == BVopMbType::Interpolated {
                    INTERP_MODE_BIAS
                } else {
                    ONE_MV_MODE_BIAS
                };
                let cost =
                    candidate_sad(&cand, &anchors, &src, mb_x, mb_y, mode).saturating_add(bias);
                consider(Chosen::Frame(cand), cost);
            }

            // §7.7.2.2 field modes: per-field estimates against each
            // anchor, codable against the bank slot of that field and
            // direction, scored through the decoder's field predictors
            // over a copy of the bank.
            let f_top = estimate_field_motion(
                &src,
                &fwd_luma,
                mb_x,
                mb_y,
                false,
                bank.get(PMV_TOP_FWD),
                mode,
                fcode,
            );
            let f_bot = estimate_field_motion(
                &src,
                &fwd_luma,
                mb_x,
                mb_y,
                true,
                bank.get(PMV_BOT_FWD),
                mode,
                fcode,
            );
            let b_top = estimate_field_motion(
                &src,
                &bwd_luma,
                mb_x,
                mb_y,
                false,
                bank.get(PMV_TOP_BWD),
                mode,
                fcode,
            );
            let b_bot = estimate_field_motion(
                &src,
                &bwd_luma,
                mb_x,
                mb_y,
                true,
                bank.get(PMV_BOT_BWD),
                mode,
                fcode,
            );
            let flags = FieldReferenceFlags {
                forward_top: FieldReference::from_bit(f_top.ref_field),
                forward_bottom: FieldReference::from_bit(f_bot.ref_field),
                backward_top: FieldReference::from_bit(b_top.ref_field),
                backward_bottom: FieldReference::from_bit(b_bot.ref_field),
            };
            {
                let deltas = field_deltas(&bank, &f_top, &f_bot, (PMV_TOP_FWD, PMV_BOT_FWD));
                let mut scratch = bank;
                let pred = field_forward_prediction(
                    &mut scratch,
                    deltas,
                    &field_refs,
                    flags,
                    mb_x,
                    mb_y,
                    0,
                    field_mode,
                );
                consider(
                    Chosen::FieldForward(f_top, f_bot),
                    sad_of(&src, &pred).saturating_add(FIELD_ONE_DIR_BIAS),
                );
            }
            {
                let deltas = field_deltas(&bank, &b_top, &b_bot, (PMV_TOP_BWD, PMV_BOT_BWD));
                let mut scratch = bank;
                let pred = field_backward_prediction(
                    &mut scratch,
                    deltas,
                    &field_refs,
                    flags,
                    mb_x,
                    mb_y,
                    0,
                    field_mode,
                );
                consider(
                    Chosen::FieldBackward(b_top, b_bot),
                    sad_of(&src, &pred).saturating_add(FIELD_ONE_DIR_BIAS),
                );
            }
            {
                let fd = field_deltas(&bank, &f_top, &f_bot, (PMV_TOP_FWD, PMV_BOT_FWD));
                let bd = field_deltas(&bank, &b_top, &b_bot, (PMV_TOP_BWD, PMV_BOT_BWD));
                let mvd = [fd.top, fd.bottom, bd.top, bd.bottom];
                let mut scratch = bank;
                let pred = field_bidirectional_prediction(
                    &mut scratch,
                    mvd,
                    &field_refs,
                    flags,
                    mb_x,
                    mb_y,
                    0,
                    field_mode,
                );
                consider(
                    Chosen::FieldBidirectional([f_top, f_bot, b_top, b_bot]),
                    sad_of(&src, &pred).saturating_add(FIELD_BIDIR_BIAS),
                );
            }

            let (chosen, _) = best.expect("at least the frame modes were scored");
            let mb_type = chosen.mb_type();

            // ---- Prediction (the decoder's, with the live bank) -------
            let pred = match &chosen {
                Chosen::Frame(c) => {
                    c.decode
                        .reconstruct(&anchors, &InterMacroblock::zero(), mb_x, mb_y, 0, mode, 8)
                }
                Chosen::InterlacedDirect(delta) => {
                    let future = future_field.expect("interlaced direct needs a field anchor");
                    let mvs = interlaced_direct_mvs(
                        future.top,
                        future.bottom,
                        MotionVector {
                            x: delta.dx,
                            y: delta.dy,
                        },
                        trb,
                        trd,
                        top_field_first,
                    );
                    let p = interlaced_direct_prediction(
                        mvs,
                        future.top.reference_field,
                        future.bottom.reference_field,
                        &field_refs,
                        mb_x,
                        mb_y,
                        0,
                        field_mode,
                    );
                    crate::reconstruct::reconstruct_inter_macroblock(
                        &p,
                        &InterMacroblock::zero(),
                        8,
                    )
                }
                Chosen::FieldForward(t, b) => {
                    let deltas = field_deltas(&bank, t, b, (PMV_TOP_FWD, PMV_BOT_FWD));
                    let mut scratch = bank;
                    let p = field_forward_prediction(
                        &mut scratch,
                        deltas,
                        &field_refs,
                        flags,
                        mb_x,
                        mb_y,
                        0,
                        field_mode,
                    );
                    crate::reconstruct::reconstruct_inter_macroblock(
                        &p,
                        &InterMacroblock::zero(),
                        8,
                    )
                }
                Chosen::FieldBackward(t, b) => {
                    let deltas = field_deltas(&bank, t, b, (PMV_TOP_BWD, PMV_BOT_BWD));
                    let mut scratch = bank;
                    let p = field_backward_prediction(
                        &mut scratch,
                        deltas,
                        &field_refs,
                        flags,
                        mb_x,
                        mb_y,
                        0,
                        field_mode,
                    );
                    crate::reconstruct::reconstruct_inter_macroblock(
                        &p,
                        &InterMacroblock::zero(),
                        8,
                    )
                }
                Chosen::FieldBidirectional([ft, fb, bt, bb]) => {
                    let fd = field_deltas(&bank, ft, fb, (PMV_TOP_FWD, PMV_BOT_FWD));
                    let bd = field_deltas(&bank, bt, bb, (PMV_TOP_BWD, PMV_BOT_BWD));
                    let mut scratch = bank;
                    let p = field_bidirectional_prediction(
                        &mut scratch,
                        [fd.top, fd.bottom, bd.top, bd.bottom],
                        &field_refs,
                        flags,
                        mb_x,
                        mb_y,
                        0,
                        field_mode,
                    );
                    crate::reconstruct::reconstruct_inter_macroblock(
                        &p,
                        &InterMacroblock::zero(),
                        8,
                    )
                }
            };
            let pred_view = InterPredictionMacroblock {
                luma: pred.luma,
                cb: pred.cb,
                cr: pred.cr,
            };

            // ---- Quantiser -------------------------------------------
            let (qp, dbquant) = if cfg.adaptive_quant && mb_type != BVopMbType::Direct {
                let class =
                    crate::mb_quant::activity_class(crate::pvop_encode::intra_activity(&src));
                crate::mb_quant::plan_dbquant(running_qp, crate::mb_quant::target_qp(vop_qp, class))
            } else {
                (running_qp, None)
            };

            // ---- Residual + dct_type ---------------------------------
            let (res_luma, res_cb, res_cr) = macroblock_residual(frame, mb_row, mb_col, &pred_view);
            let field_dct = elect_field_dct(&res_luma);
            let res_luma = if field_dct {
                field_dct_luma(&res_luma)
            } else {
                res_luma
            };
            let events = quantise_inter_residual(
                &res_luma,
                &res_cb,
                &res_cr,
                qp,
                cfg.quant_type,
                &w_inter,
                scan,
            );
            let all_zero = events.iter().all(|e| e.is_empty());
            let field_dct = field_dct && !all_zero;
            let cbpb: u8 = events
                .iter()
                .enumerate()
                .map(|(i, e)| u8::from(!e.is_empty()) << (5 - i))
                .sum();

            // ---- Emission --------------------------------------------
            let direct_zero = match &chosen {
                Chosen::Frame(c) => {
                    c.decode.mb_type == BVopMbType::Direct
                        && c.direct_delta == (MotionVectorDelta { dx: 0, dy: 0 })
                }
                Chosen::InterlacedDirect(d) => *d == (MotionVectorDelta { dx: 0, dy: 0 }),
                _ => false,
            };
            if direct_zero && all_zero {
                // modb "1": direct, zero delta, nothing coded — no
                // interlaced_information(), no bank update.
                pw.writer().write_bit(true);
                stats.modb_one += 1;
                continue;
            }
            if all_zero {
                pw.writer().write_bit(false);
                pw.writer().write_bit(true); // modb "01"
            } else {
                pw.writer().write_bit(false);
                pw.writer().write_bit(false); // modb "00"
            }
            put_b_mb_type(pw.writer(), mb_type);
            if !all_zero {
                pw.writer().write_bits(u32::from(cbpb), 6);
            }
            if mb_type != BVopMbType::Direct && !all_zero {
                crate::vlc_encode::put_dbquant(pw.writer(), dbquant.unwrap_or(0));
                running_qp = qp;
                if dbquant.is_some() {
                    stats.dbquant += 1;
                }
            }
            // §6.2.6.3 interlaced_information(): dct_type when a block
            // is coded; field_prediction on every non-direct type, with
            // the reference bits of the directions the type predicts.
            if !all_zero {
                pw.writer().write_bit(field_dct);
            }
            if field_dct {
                stats.field_dct += 1;
            }
            if mb_type != BVopMbType::Direct {
                let is_field = chosen.is_field();
                pw.writer().write_bit(is_field);
                if is_field {
                    if mb_type != BVopMbType::Backward {
                        pw.writer().write_bit(flags.forward_top.as_bit());
                        pw.writer().write_bit(flags.forward_bottom.as_bit());
                    }
                    if mb_type != BVopMbType::Forward {
                        pw.writer().write_bit(flags.backward_top.as_bit());
                        pw.writer().write_bit(flags.backward_bottom.as_bit());
                    }
                }
            }

            // Motion bodies in §6.2.6 order (forward, backward, direct)
            // plus the Table 7-15 bank update the decoder performs.
            match &chosen {
                Chosen::Frame(c) => match mb_type {
                    BVopMbType::Forward => {
                        let mv = c.decode.mvs[0].forward;
                        let p = bank.get(PMV_TOP_FWD);
                        put_motion_vector(pw.writer(), mv.x - p.x, mv.y - p.y, fcode);
                        bank.set_frame_forward(mv);
                        stats.frame_forward += 1;
                    }
                    BVopMbType::Backward => {
                        let mv = c.decode.mvs[0].backward;
                        let p = bank.get(PMV_TOP_BWD);
                        put_motion_vector(pw.writer(), mv.x - p.x, mv.y - p.y, fcode);
                        bank.set_frame_backward(mv);
                        stats.frame_backward += 1;
                    }
                    BVopMbType::Interpolated => {
                        let f = c.decode.mvs[0].forward;
                        let b = c.decode.mvs[0].backward;
                        let pf = bank.get(PMV_TOP_FWD);
                        let pb = bank.get(PMV_TOP_BWD);
                        put_motion_vector(pw.writer(), f.x - pf.x, f.y - pf.y, fcode);
                        put_motion_vector(pw.writer(), b.x - pb.x, b.y - pb.y, fcode);
                        bank.set_frame_forward(f);
                        bank.set_frame_backward(b);
                        stats.frame_interpolated += 1;
                    }
                    BVopMbType::Direct => {
                        put_motion_vector(pw.writer(), c.direct_delta.dx, c.direct_delta.dy, 1);
                        stats.direct += 1;
                    }
                },
                Chosen::InterlacedDirect(d) => {
                    // §7.7.2.2: MVD[0] with f_code 1; the bank is untouched.
                    put_motion_vector(pw.writer(), d.dx, d.dy, 1);
                    stats.interlaced_direct += 1;
                }
                Chosen::FieldForward(t, b) => {
                    let d = field_deltas(&bank, t, b, (PMV_TOP_FWD, PMV_BOT_FWD));
                    put_motion_vector(pw.writer(), d.top.x, d.top.y, fcode);
                    put_motion_vector(pw.writer(), d.bottom.x, d.bottom.y, fcode);
                    bank.field_forward(d.top, d.bottom);
                    stats.field_forward += 1;
                }
                Chosen::FieldBackward(t, b) => {
                    let d = field_deltas(&bank, t, b, (PMV_TOP_BWD, PMV_BOT_BWD));
                    put_motion_vector(pw.writer(), d.top.x, d.top.y, fcode);
                    put_motion_vector(pw.writer(), d.bottom.x, d.bottom.y, fcode);
                    bank.field_backward(d.top, d.bottom);
                    stats.field_backward += 1;
                }
                Chosen::FieldBidirectional([ft, fb, bt, bb]) => {
                    let fd = field_deltas(&bank, ft, fb, (PMV_TOP_FWD, PMV_BOT_FWD));
                    let bd = field_deltas(&bank, bt, bb, (PMV_TOP_BWD, PMV_BOT_BWD));
                    put_motion_vector(pw.writer(), fd.top.x, fd.top.y, fcode);
                    put_motion_vector(pw.writer(), fd.bottom.x, fd.bottom.y, fcode);
                    put_motion_vector(pw.writer(), bd.top.x, bd.top.y, fcode);
                    put_motion_vector(pw.writer(), bd.bottom.x, bd.bottom.y, fcode);
                    bank.field_bidirectional([fd.top, fd.bottom, bd.top, bd.bottom]);
                    stats.field_bidirectional += 1;
                }
            }

            // Texture: Table B.17 inter EVENTs per coded block.
            for ev in &events {
                if !ev.is_empty() {
                    put_ac_events(pw.writer(), TcoefTable::Inter, ev);
                }
            }
        }
    }
    stats.packets = pw.packets_cut();
    let bytes = pw.finish();
    let recon = reconstruct_own_b_vop_interlaced(vol, &bytes, store, anchor_motion, trb, trd);
    (bytes, recon, stats)
}

/// Decode an emitted interlaced B-VOP unit through the crate's own
/// interlaced walk against the anchors in `store` — the closed-loop
/// reconstruction.
pub fn reconstruct_own_b_vop_interlaced(
    vol: &crate::vol::VolHeader,
    unit: &[u8],
    store: &FrameStore,
    anchor_motion: Option<&[AnchorMbMotion]>,
    trb: i32,
    trd: i32,
) -> DecodedFrame {
    let (mb_width, mb_height) = (
        usize::from(vol.width).div_ceil(16),
        usize::from(vol.height).div_ceil(16),
    );
    let mut br = BitReader::new(unit);
    let sc = br.read_bits(32).expect("unit starts with a start code");
    assert_eq!(sc, VOP_START_CODE, "encoder emitted a malformed unit");
    let vop = parse_vop_header_body(
        &mut br,
        vol.time_increment_resolution,
        VopContext::from_vol(vol),
    )
    .expect("own VOP header must parse");
    assert!(matches!(vop.coding_type, VopCodingType::B));
    let entries = decode_b_vop_interlaced_macroblocks(
        &mut br,
        vol,
        &vop,
        trb,
        trd,
        anchor_motion,
        crate::compat::DecodeOptions::spec(),
    )
    .expect("own interlaced B-VOP payload must decode");
    let field_mode = if vol.quarter_sample {
        FieldSampleMode::QuarterSample {
            bits_per_pixel: u32::from(vol.bits_per_pixel),
        }
    } else {
        FieldSampleMode::HalfSample
    };
    crate::frame_decode::assemble_b_vop_interlaced_frame(
        store,
        mb_width,
        mb_height,
        &entries,
        vop.rounding_type,
        sample_mode_of(vol),
        field_mode,
        8,
    )
    .expect("own interlaced B-VOP must assemble")
}

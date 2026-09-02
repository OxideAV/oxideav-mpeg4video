//! §6.2.5.2 **short-header encoder** — H.263-compatible I and P
//! pictures (`short_video_header == 1`).
//!
//! The picture is one of the five Table 6-29 source formats; the
//! syntax is the Table 6-28 fixed tool set: method-2 quantisation,
//! `vop_fcode_forward == 1`, `vop_rounding_type == 0`, half-sample
//! prediction, one vector per macroblock, no OBMC / resync / data
//! partitioning / B pictures / interlace. Per macroblock the walk
//! runs the progressive P encoder's motion search — restricted to
//! vectors whose reference block (interpolation neighbours included,
//! luma and chroma) stays inside the picture, since §7.6.4 admits
//! unrestricted vectors only when the short header is *not* in use —
//! the §7.6.5 block-0 median predictor with the GOB rule (a GOB header
//! resets the candidate grid across GOB boundaries), the intra / inter
//! decision and skip of the P walk, and the §6.2.6 emission with its
//! short-header gates: `intra_dc_coefficient` as the 8-bit FLC
//! (`QF = F // 8`, clamped to the codable `1..=254`, 128 sent as 255),
//! every AC EVENT through the Table B.17 VLC or the §7.4.1.3 Type-4
//! escape (levels clamped to `±127`), no `ac_pred_flag`. GOB headers
//! (`gob_resync_marker` + `gob_number` + `gob_frame_id` +
//! `quant_scale`, byte-aligned) are emitted on every GOB after the
//! first when `EncoderConfig::gob_headers` is set.
//!
//! The finished picture is decoded back through
//! [`crate::short_header::decode_short_header_macroblocks`] — the
//! same closed loop as the long-header encoders.
//!
//! Provenance: §6.2.5.2, §6.2.6, §6.2.7, §6.3.5.2, §6.3.7, §7.4.1.3
//! Type 4, §7.4.4.3, §7.6.4, §7.6.5 of ISO/IEC 14496-2:2004 (3rd
//! edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`. No
//! third-party source was consulted.

use crate::bitreader::BitReader;
use crate::bitwriter::BitWriter;
use crate::block::InterMacroblock;
use crate::bvop_prediction::BVopSampleMode;
use crate::fdct::forward_dct_8x8;
use crate::framestore::{DecodedFrame, FrameStore};
use crate::half_sample::ReferenceVop;
use crate::ivop_encode::{forward_scan, qfs_to_events, EncoderConfig, FrameView};
use crate::motion::{predict_motion_vector, MotionVector};
use crate::mv_predictor_grid::MvGrid;
use crate::pvop_encode::{
    inter_cbp, intra_activity, macroblock_residual, quantise_inter_block, source_luma_mb,
    PVopEncodeStats,
};
use crate::pvop_mv::{predict_inter_macroblock, PvopMbMotion};
use crate::quantise::quantise_method2_intra;
use crate::scan::ScanType;
use crate::short_header::{
    decode_short_header_macroblocks, intra_dc_code, parse_short_header_picture, SourceFormat,
    GOB_MARKER_BITS, GOB_RESYNC_MARKER, SHORT_MARKER_BITS, SHORT_VIDEO_END_MARKER,
    SHORT_VIDEO_START_MARKER,
};
use crate::texture::AcEvent;
use crate::vlc_encode::{
    put_ac_events_short_video_header, put_cbpy, put_dquant, put_mcbpc_i, put_mcbpc_p,
    put_motion_vector,
};
use crate::vop::VopCodingType;

/// Largest |level| the §7.4.1.3 Type-4 escape can carry (8-bit
/// two's complement, `-128` reserved).
pub const SHORT_HEADER_MAX_LEVEL: i32 = 127;

/// Emit the `video_plane_with_short_header()` picture header through
/// the (empty) `pei` loop. The writer must be byte-aligned.
pub fn write_short_header_picture(
    bw: &mut BitWriter,
    temporal_reference: u8,
    source_format: SourceFormat,
    coding_type: VopCodingType,
    quant: u32,
) {
    assert!(
        bw.is_byte_aligned(),
        "short_video_start_marker is byte-aligned"
    );
    assert!((1..=31).contains(&quant), "vop_quant {quant} out of range");
    bw.write_bits(SHORT_VIDEO_START_MARKER, SHORT_MARKER_BITS);
    bw.write_bits(u32::from(temporal_reference), 8);
    bw.write_marker(); // marker_bit
    bw.write_bit(false); // zero_bit
    bw.write_bit(false); // split_screen_indicator
    bw.write_bit(false); // document_camera_indicator
    bw.write_bit(false); // full_picture_freeze_release
    bw.write_bits(source_format.code(), 3);
    bw.write_bit(matches!(coding_type, VopCodingType::P)); // picture_coding_type
    bw.write_bits(0, 4); // four_reserved_zero_bits
    bw.write_bits(quant, 5);
    bw.write_bit(false); // zero_bit
    bw.write_bit(false); // pei = 0
}

/// Emit the `short_video_end_marker` (byte-aligned with zero bits).
pub fn write_short_video_end_marker(bw: &mut BitWriter) {
    bw.align_zero();
    bw.write_bits(SHORT_VIDEO_END_MARKER, SHORT_MARKER_BITS);
    bw.align_zero();
}

/// Emit a GOB header for `gob_number >= 1`: zero-bit byte alignment,
/// the 17-bit `gob_resync_marker`, `gob_number`, `gob_frame_id` and
/// `quant_scale`.
fn write_gob_header(bw: &mut BitWriter, gob_number: usize, gob_frame_id: u32, quant: u32) {
    assert!(gob_number >= 1, "the first GOB carries no header");
    bw.align_zero();
    bw.write_bits(GOB_RESYNC_MARKER, GOB_MARKER_BITS);
    bw.write_bits(gob_number as u32, 5);
    bw.write_bits(gob_frame_id & 0b11, 2);
    bw.write_bits(quant, 5);
}

/// Quantise one intra block for the short header: DC as `F // 8`
/// clamped into the FLC domain, AC with the method-2 intra quantiser
/// clamped to the Type-4 escape range.
fn quantise_intra_block_short(f: &[[i32; 8]; 8], qp: u32) -> (i32, Vec<AcEvent>) {
    let dc = {
        let f00 = f[0][0];
        let q = if f00 >= 0 {
            (f00 + 4) / 8
        } else {
            -((-f00 + 4) / 8)
        };
        q.clamp(1, 254)
    };
    let mut qf = [[0i32; 8]; 8];
    for v in 0..8 {
        for u in 0..8 {
            if (v, u) != (0, 0) {
                qf[v][u] = quantise_method2_intra(f[v][u], qp)
                    .clamp(-SHORT_HEADER_MAX_LEVEL, SHORT_HEADER_MAX_LEVEL);
            }
        }
    }
    let qfs = forward_scan(&qf, ScanType::Zigzag);
    (dc, qfs_to_events(&qfs, 1))
}

/// Clamp an inter block's EVENT levels to the Type-4 escape range.
fn clamp_levels(events: &mut [AcEvent]) {
    for ev in events {
        ev.level = ev
            .level
            .clamp(-SHORT_HEADER_MAX_LEVEL, SHORT_HEADER_MAX_LEVEL);
    }
}

/// The six quantised intra blocks (DC + AC EVENTs) of one macroblock.
fn intra_blocks_short(
    frame: &FrameView<'_>,
    mb_row: usize,
    mb_col: usize,
    qp: u32,
) -> Vec<(i32, Vec<AcEvent>)> {
    (0..6)
        .map(|i| {
            let samples = frame.block(mb_row, mb_col, i);
            let f = forward_dct_8x8(&samples, 8);
            quantise_intra_block_short(&f, qp)
        })
        .collect()
}

/// Emit one intra macroblock body: `mcbpc`, `cbpy`, `dquant`, six
/// blocks (8-bit DC then the AC EVENTs of the coded ones).
fn write_intra_mb(
    bw: &mut BitWriter,
    in_p_picture: bool,
    blocks: &[(i32, Vec<AcEvent>)],
    dquant: Option<i8>,
) {
    let coded: Vec<bool> = blocks.iter().map(|(_, ev)| !ev.is_empty()).collect();
    let cbpy = (u8::from(coded[0]) << 3)
        | (u8::from(coded[1]) << 2)
        | (u8::from(coded[2]) << 1)
        | u8::from(coded[3]);
    let cbpc = (u8::from(coded[4]) << 1) | u8::from(coded[5]);
    let mb_type = if dquant.is_some() { 4 } else { 3 };
    if in_p_picture {
        bw.write_bit(false); // not_coded = 0
        put_mcbpc_p(bw, mb_type, cbpc);
    } else {
        put_mcbpc_i(bw, mb_type, cbpc);
    }
    put_cbpy(bw, cbpy, true);
    if let Some(d) = dquant {
        put_dquant(bw, d);
    }
    for (dc, events) in blocks {
        bw.write_bits(intra_dc_code(*dc), 8);
        put_ac_events_short_video_header(bw, events);
    }
}

/// Whether a half-sample luma vector keeps the 16×16 reference block
/// — and the derived chroma block — inside the picture (§7.6.4: no
/// unrestricted vectors under the short header). Interpolation
/// neighbours count: a half-sample position reads one sample beyond
/// the block on that axis.
fn mv_inside_picture(mv: MotionVector, mb_x: i32, mb_y: i32, width: i32, height: i32) -> bool {
    let span = |origin: i32, comp: i32, size: i32, limit: i32| -> bool {
        let int = comp >> 1; // floor
        let extra = i32::from(comp & 1 != 0);
        origin + int >= 0 && origin + int + size - 1 + extra < limit
    };
    if !span(mb_x, mv.x, 16, width) || !span(mb_y, mv.y, 16, height) {
        return false;
    }
    let cmv = crate::chroma_mv::chroma_mv_from_luma_blocks(&[mv]).expect("one vector reduces");
    span(mb_x / 2, cmv.x, 8, width / 2) && span(mb_y / 2, cmv.y, 8, height / 2)
}

/// Restricted §7.6 motion search: full-pel window inside the picture,
/// half-sample ring refinement through the decoder's interpolator,
/// every candidate kept inside the picture and inside the `f_code
/// == 1` range `[-32, 31]`. Returns `(mv, sad)`.
fn estimate_motion_restricted(
    src: &[[i32; 16]; 16],
    reference: &ReferenceVop<'_>,
    mb_x: i32,
    mb_y: i32,
) -> (MotionVector, u32) {
    let (w, h) = (reference.width() as i32, reference.height() as i32);
    let sad_of = |mv: MotionVector| -> u32 {
        let pred = crate::pvop_mv::predict_luma_macroblock(
            PvopMbMotion::OneMv(mv),
            reference,
            mb_x,
            mb_y,
            0,
            BVopSampleMode::HalfPel,
        )
        .expect("OneMv always yields a prediction");
        let mut sad = 0u32;
        for (j, row) in src.iter().enumerate() {
            for (i, &s) in row.iter().enumerate() {
                sad += (s - i32::from(pred[j * 16 + i])).unsigned_abs();
            }
        }
        sad
    };
    let zero = MotionVector { x: 0, y: 0 };
    let mut best = zero;
    let mut best_sad = sad_of(zero).saturating_sub(128);
    let range = crate::pvop_encode::SEARCH_RANGE;
    for dy in -range..=range {
        for dx in -range..=range {
            if (dx, dy) == (0, 0) {
                continue;
            }
            let mv = MotionVector {
                x: 2 * dx,
                y: 2 * dy,
            };
            if !mv_inside_picture(mv, mb_x, mb_y, w, h) {
                continue;
            }
            let sad = sad_of(mv);
            if sad < best_sad {
                best_sad = sad;
                best = mv;
            }
        }
    }
    let centre = best;
    for hy in -1..=1 {
        for hx in -1..=1 {
            if (hx, hy) == (0, 0) {
                continue;
            }
            let mv = MotionVector {
                x: (centre.x + hx).clamp(-32, 31),
                y: (centre.y + hy).clamp(-32, 31),
            };
            if !mv_inside_picture(mv, mb_x, mb_y, w, h) {
                continue;
            }
            let sad = sad_of(mv);
            if sad < best_sad {
                best_sad = sad;
                best = mv;
            }
        }
    }
    (best, best_sad)
}

/// Encode one short-header picture (I when `reference` is `None`,
/// else P against it) at `temporal_reference`. Returns the emitted
/// picture (byte-aligned, no end marker), its closed-loop
/// reconstruction, and the P statistics (all-intra for an I picture).
pub fn encode_short_header_picture(
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    reference: Option<&DecodedFrame>,
    temporal_reference: u8,
    qp: u32,
) -> (Vec<u8>, DecodedFrame, PVopEncodeStats) {
    assert!(
        cfg.short_header,
        "encode_short_header_picture needs a short-header config"
    );
    let sf = SourceFormat::from_dimensions(u32::from(cfg.width), u32::from(cfg.height))
        .expect("short-header dimensions are one of the Table 6-29 formats");
    let (mb_width, mb_height) = sf.mb_dimensions();
    let per_gob = sf.macroblocks_per_gob();
    let gobs = sf.gobs_per_picture();
    let coding_type = if reference.is_some() {
        VopCodingType::P
    } else {
        VopCodingType::I
    };
    let is_p = reference.is_some();

    let mut bw = BitWriter::new();
    write_short_header_picture(&mut bw, temporal_reference, sf, coding_type, qp);

    let luma_ref = reference.map(|r| r.luma_reference());
    let cb_ref = reference.map(|r| r.cb_reference());
    let cr_ref = reference.map(|r| r.cr_reference());
    let mut mv_grid = MvGrid::new(mb_height, mb_width);
    let mut stats = PVopEncodeStats::default();
    let vop_qp = qp;
    let mut running_qp = vop_qp;
    let no_matrix = [[0u8; 8]; 8];

    for gob in 0..gobs {
        if gob != 0 && cfg.gob_headers {
            // A GOB header restarts the running quantiser from
            // quant_scale (restated as the current running value) and
            // — §7.6.5 — cuts the predictor candidates off from the
            // earlier GOBs.
            write_gob_header(&mut bw, gob, 0, running_qp);
            let first_row = (gob * per_gob) / mb_width;
            for r in 0..first_row {
                for c in 0..mb_width {
                    mv_grid
                        .record_absent(r, c)
                        .expect("grid coordinates in range");
                }
            }
        }
        for k in 0..per_gob {
            let idx = gob * per_gob + k;
            let (mb_row, mb_col) = (idx / mb_width, idx % mb_width);
            let (mb_x, mb_y) = ((mb_col * 16) as i32, (mb_row * 16) as i32);
            let src = source_luma_mb(frame, mb_row, mb_col);
            let activity = intra_activity(&src);
            let plan_quant = |running: u32| -> (u32, Option<i8>) {
                if cfg.adaptive_quant {
                    let class = crate::mb_quant::activity_class(activity);
                    crate::mb_quant::plan_dquant(running, crate::mb_quant::target_qp(vop_qp, class))
                } else {
                    (running, None)
                }
            };

            if !is_p {
                let (qp, dquant) = plan_quant(running_qp);
                running_qp = qp;
                if dquant.is_some() {
                    stats.dquant += 1;
                }
                stats.intra += 1;
                let blocks = intra_blocks_short(frame, mb_row, mb_col, qp);
                write_intra_mb(&mut bw, false, &blocks, dquant);
                continue;
            }

            let luma_ref = luma_ref.as_ref().expect("P picture has a reference");
            let (mv, inter_sad) = estimate_motion_restricted(&src, luma_ref, mb_x, mb_y);
            let choose_intra = activity + 512 < inter_sad;
            if choose_intra {
                let (qp, dquant) = plan_quant(running_qp);
                running_qp = qp;
                if dquant.is_some() {
                    stats.dquant += 1;
                }
                stats.intra += 1;
                mv_grid
                    .record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })
                    .expect("grid coordinates in range");
                let blocks = intra_blocks_short(frame, mb_row, mb_col, qp);
                write_intra_mb(&mut bw, true, &blocks, dquant);
                continue;
            }

            let (qp, dquant) = plan_quant(running_qp);
            let prediction = predict_inter_macroblock(
                PvopMbMotion::OneMv(mv),
                luma_ref,
                cb_ref.as_ref().expect("reference"),
                cr_ref.as_ref().expect("reference"),
                mb_x,
                mb_y,
                0,
                BVopSampleMode::HalfPel,
            )
            .expect("inter motion always yields a prediction");
            let (res_luma, res_cb, res_cr) =
                macroblock_residual(frame, mb_row, mb_col, &prediction);
            let mut events: Vec<Vec<AcEvent>> = Vec::with_capacity(6);
            for i in 0..4 {
                let (row0, col0) = (8 * (i / 2), 8 * (i % 2));
                let mut block = [[0i32; 8]; 8];
                for (y, row) in block.iter_mut().enumerate() {
                    row.copy_from_slice(&res_luma[row0 + y][col0..col0 + 8]);
                }
                let (mut ev, _) =
                    quantise_inter_block(&block, qp, false, &no_matrix, ScanType::Zigzag);
                clamp_levels(&mut ev);
                events.push(ev);
            }
            for chroma in [&res_cb, &res_cr] {
                let (mut ev, _) =
                    quantise_inter_block(chroma, qp, false, &no_matrix, ScanType::Zigzag);
                clamp_levels(&mut ev);
                events.push(ev);
            }
            let all_zero = events.iter().all(|e| e.is_empty());
            if all_zero && mv == (MotionVector { x: 0, y: 0 }) {
                stats.skipped += 1;
                bw.write_bit(true); // not_coded
                mv_grid
                    .record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })
                    .expect("grid coordinates in range");
                continue;
            }
            running_qp = qp;
            if dquant.is_some() {
                stats.dquant += 1;
            }
            stats.inter += 1;
            let (cbpy, cbpc) = inter_cbp(&events);
            let candidates = mv_grid
                .predictor_candidates(mb_row, mb_col, 0)
                .expect("grid coordinates in range");
            let predictor = predict_motion_vector(candidates);
            mv_grid
                .record_one_mv(mb_row, mb_col, mv)
                .expect("grid coordinates in range");
            bw.write_bit(false); // not_coded = 0
            put_mcbpc_p(&mut bw, if dquant.is_some() { 1 } else { 0 }, cbpc);
            put_cbpy(&mut bw, cbpy, false);
            if let Some(d) = dquant {
                put_dquant(&mut bw, d);
            }
            put_motion_vector(&mut bw, mv.x - predictor.x, mv.y - predictor.y, 1);
            for ev in &events {
                put_ac_events_short_video_header(&mut bw, ev);
            }
        }
    }
    // §6.2.5.2: pad to the byte boundary with zero bits.
    bw.align_zero();
    let bytes = bw.into_bytes();
    let recon = reconstruct_own_short_header_picture(&bytes, reference);
    (bytes, recon, stats)
}

/// Decode an emitted short-header picture through the crate's own
/// short-header walk against `reference` (the previous anchor for a
/// P picture) — the closed-loop reconstruction.
pub fn reconstruct_own_short_header_picture(
    unit: &[u8],
    reference: Option<&DecodedFrame>,
) -> DecodedFrame {
    let mut br = BitReader::new(unit);
    let pic = parse_short_header_picture(&mut br).expect("own short header must parse");
    let entries = decode_short_header_macroblocks(&mut br, &pic).expect("own picture must decode");
    let (mb_width, mb_height) = pic.source_format.mb_dimensions();
    match reference {
        None => {
            let mbs = crate::short_header::intra_macroblocks(&entries)
                .expect("an I picture is all intra");
            let mut frame = DecodedFrame::new(mb_width * 16, mb_height * 16, VopCodingType::I)
                .expect("frame dimensions are valid");
            for (idx, mb) in mbs.iter().enumerate() {
                frame
                    .blit_macroblock(idx % mb_width, idx / mb_width, mb)
                    .expect("grid-shaped blit");
            }
            frame
        }
        Some(r) => {
            let mut store = FrameStore::new();
            store.push_anchor(r.clone());
            let _ = InterMacroblock::zero();
            crate::frame_decode::assemble_p_vop_frame(
                &store,
                mb_width,
                mb_height,
                &entries,
                0,
                BVopSampleMode::HalfPel,
                8,
            )
            .expect("own P picture must assemble")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picture_header_round_trips() {
        let mut bw = BitWriter::new();
        write_short_header_picture(&mut bw, 37, SourceFormat::Qcif, VopCodingType::P, 9);
        bw.align_zero();
        let bytes = bw.into_bytes();
        assert!(crate::short_header::is_short_header_picture_start(
            &bytes, 0
        ));
        let mut br = BitReader::new(&bytes);
        let pic = parse_short_header_picture(&mut br).unwrap();
        assert_eq!(pic.temporal_reference, 37);
        assert_eq!(pic.source_format, SourceFormat::Qcif);
        assert_eq!(pic.coding_type, VopCodingType::P);
        assert_eq!(pic.quant, 9);
    }

    #[test]
    fn intra_dc_quantisation_stays_in_the_flc_domain() {
        let mut f = [[0i32; 8]; 8];
        for (dc, expect) in [
            (0, 1),
            (4, 1),
            (8, 1),
            (1024, 128),
            (2040, 254),
            (2047, 254),
        ] {
            f[0][0] = dc;
            assert_eq!(quantise_intra_block_short(&f, 4).0, expect, "dc {dc}");
        }
    }

    #[test]
    fn vectors_outside_the_picture_are_rejected() {
        // 176×144, macroblock at the top-left corner.
        assert!(mv_inside_picture(
            MotionVector { x: 0, y: 0 },
            0,
            0,
            176,
            144
        ));
        assert!(!mv_inside_picture(
            MotionVector { x: -1, y: 0 },
            0,
            0,
            176,
            144
        ));
        assert!(!mv_inside_picture(
            MotionVector { x: 0, y: -2 },
            0,
            0,
            176,
            144
        ));
        // Bottom-right macroblock: a positive half-sample vector reads
        // past the edge.
        assert!(mv_inside_picture(
            MotionVector { x: 0, y: 0 },
            160,
            128,
            176,
            144
        ));
        assert!(!mv_inside_picture(
            MotionVector { x: 1, y: 0 },
            160,
            128,
            176,
            144
        ));
        assert!(!mv_inside_picture(
            MotionVector { x: 0, y: 2 },
            160,
            128,
            176,
            144
        ));
        assert!(mv_inside_picture(
            MotionVector { x: -2, y: -2 },
            160,
            128,
            176,
            144
        ));
    }
}

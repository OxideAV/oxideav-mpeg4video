//! §E.1.4.4 two-way RVLC recovery, driven end-to-end through the
//! data-partitioned P-VOP macroblock walk: a corrupted DCT-coefficient
//! region no longer aborts the decode — the recovery runs, kept
//! macroblocks reconstruct from their recovered EVENT runs, discarded
//! ones fall back to motion-compensated concealment, and the walk
//! resumes cleanly at the next video packet.
//!
//! The streams are hand-written with the same in-repo bit-writing
//! approach as the unit tests: a 64×64 `data_partitioned == 1` /
//! `reversible_vlc == 1` VOL and a P-VOP with two 8-macroblock video
//! packets. Every macroblock is inter (mcbpc "1", zero MVD, cbpy
//! selecting luma block 0), and each coded block is the single
//! Table B.23 RVLC EVENT (LAST=1, RUN=0, LEVEL=+1) — code "1011" +
//! sign "0". The corrupted variant truncates packet 1's texture
//! partition to five of its eight EVENT runs, so the forward decode
//! exhausts the region early (§E.1.4.4.1) and the backward decode
//! re-reads the same five runs from the region end.

use oxideav_mpeg4video::bitreader::BitReader;
use oxideav_mpeg4video::compat::DecodeOptions;
use oxideav_mpeg4video::frame_decode::PVopMbContent;
use oxideav_mpeg4video::video_packet::macroblock_number_bit_width;
use oxideav_mpeg4video::vol::parse_video_object_layer;
use oxideav_mpeg4video::vop::{parse_vop_header_body, VopContext};
use oxideav_mpeg4video::vop_decode::decode_p_vop_macroblocks_dp;

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
    /// Video-packet stuffing: '0' then '1's to the byte boundary
    /// (always at least one bit — a full "01111111" byte when already
    /// aligned).
    fn stuff_to_byte(&mut self) {
        self.write_bits(0, 1);
        while self.bit != 0 {
            self.write_bits(1, 1);
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.bit != 0 {
            self.write_bits(0, 1);
            while self.bit != 0 {
                self.write_bits(1, 1);
            }
        }
        self.bytes
    }
}

/// Write the test VOL: rectangular 64×64, verid 1, method-2 quant,
/// resolution 30, resync markers enabled, `data_partitioned == 1`,
/// `reversible_vlc == 1`.
fn write_vol(w: &mut BitWriter) {
    w.write_bits(0x0000_0120, 32);
    w.write_bits(0, 1); // random_accessible_vol
    w.write_bits(1, 8); // video_object_type_indication
    w.write_bits(0, 1); // is_object_layer_identifier
    w.write_bits(1, 4); // aspect_ratio_info 1:1
    w.write_bits(0, 1); // vol_control_parameters
    w.write_bits(0, 2); // shape rectangular
    w.write_bits(1, 1); // marker
    w.write_bits(30, 16); // vop_time_increment_resolution
    w.write_bits(1, 1); // marker
    w.write_bits(0, 1); // fixed_vop_rate
    w.write_bits(1, 1); // marker
    w.write_bits(64, 13); // width
    w.write_bits(1, 1); // marker
    w.write_bits(64, 13); // height
    w.write_bits(1, 1); // marker
    w.write_bits(0, 1); // interlaced
    w.write_bits(1, 1); // obmc_disable
    w.write_bits(0, 1); // sprite_enable
    w.write_bits(0, 1); // not_8_bit
    w.write_bits(0, 1); // quant_type (method 2)
    w.write_bits(1, 1); // complexity_estimation_disable
    w.write_bits(0, 1); // resync_marker_disable == 0 (packets)
    w.write_bits(1, 1); // data_partitioned
    w.write_bits(1, 1); // reversible_vlc
    w.write_bits(0, 1); // scalability
}

const MOTION_MARKER: u32 = 0b1_1111_0000_0000_0001;

/// Write one packet's partitions for `n` inter macroblocks: partition 1
/// (`not_coded 0`, mcbpc "1" inter/cbpc 00, zero MVD "1" "1"),
/// `motion_marker`, partition 2 (`cbpy` code "1011" → inter 1000: luma
/// block 0 coded), then `events` single-EVENT texture runs
/// ("1011" + sign 0 → LAST=1, RUN=0, LEVEL=+1 on the inter column).
fn write_packet_body(w: &mut BitWriter, n: usize, events: usize) {
    for _ in 0..n {
        w.write_bits(0, 1); // not_coded
        w.write_bits(0b1, 1); // mcbpc: inter, cbpc 00
        w.write_bits(0b1, 1); // MVDx = 0 (Table B.12 "1")
        w.write_bits(0b1, 1); // MVDy = 0
    }
    w.write_bits(MOTION_MARKER, 17);
    for _ in 0..n {
        w.write_bits(0b1011, 4); // cbpy: inter 1000 (Table B.8)
    }
    for _ in 0..events {
        w.write_bits(0b1011, 4); // RVLC EVENT (1, 0, 1)
        w.write_bits(0, 1); // sign +
    }
}

/// Build the full elementary unit (VOL + one P-VOP) with `p1_events`
/// EVENT runs in packet 1's texture partition (8 = clean, fewer =
/// truncated/corrupt).
fn build_stream(p1_events: usize) -> Vec<u8> {
    let mut w = BitWriter::default();
    write_vol(&mut w);
    if w.bit != 0 {
        w.stuff_to_byte();
    }
    // P-VOP header.
    w.write_bits(0x0000_01B6, 32);
    w.write_bits(0b01, 2); // P
    w.write_bits(0, 1); // modulo_time_base terminator
    w.write_bits(1, 1); // marker
    w.write_bits(1, 5); // vop_time_increment (res 30 → 5 bits)
    w.write_bits(1, 1); // marker
    w.write_bits(1, 1); // vop_coded
    w.write_bits(0, 1); // vop_rounding_type
    w.write_bits(0, 3); // intra_dc_vlc_thr
    w.write_bits(8, 5); // vop_quant
    w.write_bits(1, 3); // vop_fcode_forward

    // Packet 1: macroblocks 0..8.
    write_packet_body(&mut w, 8, p1_events);

    // Packet 2: stuffing, resync marker (P, fcode 1 → 17 bits),
    // macroblock_number, quant_scale, header_extension_code 0.
    w.stuff_to_byte();
    w.write_bits(1, 17); // resync_marker: 16 zeros + 1
    let mb_bits = usize::from(macroblock_number_bit_width(16));
    w.write_bits(8, mb_bits); // macroblock_number = 8
    w.write_bits(8, 5); // quant_scale
    w.write_bits(0, 1); // header_extension_code
    write_packet_body(&mut w, 8, 8);

    w.finish()
}

fn decode(stream: &[u8]) -> Vec<PVopMbContent> {
    // VOL starts at byte 0; the VOP start code is byte-aligned after it.
    let vol = parse_video_object_layer(stream, 1).expect("vol");
    assert!(vol.data_partitioned && vol.reversible_vlc && !vol.resync_marker_disable);
    let vop_off = stream
        .windows(4)
        .position(|win| win == [0, 0, 1, 0xB6])
        .expect("vop start code");
    let mut br = BitReader::new(&stream[vop_off..]);
    br.skip_bits(32).unwrap();
    let ctx = VopContext::from_vol(&vol);
    let vop = parse_vop_header_body(&mut br, vol.time_increment_resolution, ctx).expect("vop");
    decode_p_vop_macroblocks_dp(&mut br, &vol, &vop, DecodeOptions::spec()).expect("dp walk")
}

fn residual_is_zero(mb: &PVopMbContent) -> bool {
    match mb {
        PVopMbContent::Inter { residual, .. } => {
            residual.luma.iter().all(|row| row.iter().all(|&v| v == 0))
                && residual.cb.iter().all(|row| row.iter().all(|&v| v == 0))
                && residual.cr.iter().all(|row| row.iter().all(|&v| v == 0))
        }
        _ => false,
    }
}

#[test]
fn clean_rvlc_dp_stream_decodes_every_macroblock() {
    let clean = decode(&build_stream(8));
    assert_eq!(clean.len(), 16);
    for (i, mb) in clean.iter().enumerate() {
        assert!(
            matches!(mb, PVopMbContent::Inter { .. }),
            "MB {i} must be inter"
        );
        assert!(
            !residual_is_zero(mb),
            "MB {i} must carry the EVENT residual"
        );
    }
}

#[test]
fn truncated_rvlc_texture_recovers_and_resumes_at_the_next_packet() {
    let clean = decode(&build_stream(8));
    // Packet 1's texture carries EVENT runs for only 5 of its 8
    // macroblocks: the forward decode exhausts the region after MB 4
    // (§E.1.4.4.1), the backward decode re-reads the same five runs
    // from the region end (landing on macroblocks 3..8), and the
    // §E.1.4.4.2.1 arbitration keeps a front and a back span.
    let recovered = decode(&build_stream(5));
    assert_eq!(recovered.len(), 16);

    // Packet 2 is untouched by the recovery: bit-identical decode.
    for i in 8..16 {
        assert_eq!(recovered[i], clean[i], "packet-2 MB {i} must match clean");
    }

    // Packet 1: every macroblock keeps its trusted motion (inter, zero
    // MV) — the recovery only affects residuals.
    for i in 0..8 {
        match (&recovered[i], &clean[i]) {
            (PVopMbContent::Inter { motion: rm, .. }, PVopMbContent::Inter { motion: cm, .. }) => {
                assert_eq!(rm, cm, "MB {i} motion must stay trusted")
            }
            _ => panic!("MB {i}: unexpected content kind"),
        }
    }

    // Every kept residual equals the clean single-EVENT residual (all
    // 8 macroblocks share it — same EVENT, same quantiser); discarded
    // middle MBs conceal to the zero residual. At least one MB from
    // each end must be kept, and at least one middle MB discarded
    // (the region held 5 runs for 8 macroblocks).
    let mut kept = 0usize;
    let mut concealed = 0usize;
    for i in 0..8 {
        if residual_is_zero(&recovered[i]) {
            concealed += 1;
        } else {
            assert_eq!(recovered[i], clean[i], "kept MB {i} must match clean");
            kept += 1;
        }
    }
    assert!(kept >= 2, "front + back spans must keep MBs (kept {kept})");
    assert!(
        concealed >= 3,
        "the errored middle must conceal (concealed {concealed})"
    );
    assert!(
        !residual_is_zero(&recovered[0]),
        "MB 0 heads the forward-kept span"
    );
    assert!(
        !residual_is_zero(&recovered[7]),
        "MB 7 tails the backward-kept span"
    );
}

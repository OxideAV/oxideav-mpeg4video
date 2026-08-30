//! Error-resilience **emission**: §6.2.5 video packets (`resync_marker`
//! plus `video_packet_header()` incl. the `header_extension_code`
//! body), §6.2.5.3 data partitioning (`dc_marker` / `motion_marker`)
//! and the reversible-VLC texture partition (Table B.23, emitted by
//! [`crate::vlc_encode::put_ac_events_rvlc`]).
//!
//! The encoders describe every macroblock as an [`MbFields`] record
//! (the syntax elements, already VLC-decided) and hand it to a
//! [`PacketWriter`], which owns the layout:
//!
//! * **combined syntax** (`data_partitioned == 0`) — the §6.2.6
//!   `macroblock()` order: `not_coded`, `mcbpc`, `ac_pred_flag`,
//!   `cbpy`, `dquant`, `motion_vector()`s, then the six `block()`
//!   bodies (intra DC inside each block);
//! * **`data_partitioned_i_vop()`** — partition 1 = per MB `mcbpc`
//!   [+ `dquant`] + the six intra-DC differentials, `dc_marker`,
//!   partition 2 = per MB `ac_pred_flag` + `cbpy`, partition 3 = the AC
//!   texture;
//! * **`data_partitioned_p_vop()`** — partition 1 = per MB `not_coded`
//!   [+ `mcbpc` + `motion_vector()`s], `motion_marker`, partition 2 =
//!   per coded MB [`ac_pred_flag`] + `cbpy` [+ `dquant`] [+ intra DC],
//!   partition 3 = the AC texture.
//!
//! A packet is cut at the first macroblock boundary after the packet
//! has accumulated `packet_bits` bits (encoder freedom — §6.3.3 only
//! requires that a packet start with a byte-aligned `resync_marker`
//! and a `video_packet_header`). The cut resets every prediction state
//! the decoder resets (§E.1.2: intra predictors, motion predictors,
//! and the running quantiser re-seeded by `quant_scale`) — the caller
//! performs those resets when [`PacketWriter::maybe_cut`] reports a
//! cut. `header_extension_code` alternates packet by packet so both
//! decoder branches are exercised (HEC on the first packet after the
//! VOP header carries the duplicated timing / type / fcode fields).
//!
//! Provenance: ISO/IEC 14496-2:2004 (3rd edition) §5.2.5, §6.2.5
//! (`video_packet_header`, `resync_marker` lengths per §6.3.3),
//! §6.2.5.3 (partition layouts, `dc_marker`, `motion_marker`),
//! Table 6-27 (`macroblock_number` width) read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`,
//! mirrored through the crate's own decode-side parsers.

use crate::bitwriter::BitWriter;
use crate::data_partition::{DC_MARKER, DC_MARKER_BITS, MOTION_MARKER, MOTION_MARKER_BITS};
use crate::texture::{AcEvent, DcComponent, TcoefTable};
use crate::video_packet::{macroblock_number_bit_width, resync_marker_length};
use crate::vlc_encode::{
    put_ac_events, put_ac_events_rvlc, put_cbpy, put_dquant, put_intra_dc, put_mcbpc_i,
    put_mcbpc_p, put_motion_vector,
};
use crate::vop::VopCodingType;

/// The VOL-level error-resilience tool selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResilienceConfig {
    /// Target video-packet size in bits (`0` = no video packets:
    /// `resync_marker_disable == 1`, one packet per VOP).
    pub packet_bits: u32,
    /// §6.2.5.3 data partitioning of I-/P-VOPs (`data_partitioned`).
    pub data_partitioned: bool,
    /// Reversible VLCs for the data-partitioned texture partition
    /// (`reversible_vlc`; requires `data_partitioned`).
    pub reversible_vlc: bool,
}

/// The macroblock-layer layout a [`PacketWriter`] serialises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    /// §6.2.6 combined syntax (any VOP type; B-VOPs always).
    Combined,
    /// `data_partitioned_i_vop()`.
    PartitionedI,
    /// `data_partitioned_p_vop()`.
    PartitionedP,
}

/// One macroblock's syntax elements, VLC-decided but not yet laid
/// out.
#[derive(Debug, Clone)]
pub(crate) struct MbFields {
    /// P-VOP `not_coded == 1` (no other field is emitted).
    pub not_coded: bool,
    /// Table B.6 / B.7 `derived_mb_type` (0 inter, 1 inter+q, 2
    /// inter4v, 3 intra, 4 intra+q).
    pub mb_type: u8,
    /// 2-bit `cbpc`.
    pub cbpc: u8,
    /// 4-bit `cbpy` (§6.3.7 "1 = coded").
    pub cbpy: u8,
    /// `ac_pred_flag` (intra macroblocks only).
    pub ac_pred_flag: bool,
    /// Table 6-32 `dquant` delta (types 1 / 4 only).
    pub dquant: Option<i8>,
    /// Raw `(dx, dy)` motion-vector differentials in emission order
    /// (one for inter / inter+q, four for inter4v, none for intra),
    /// wrapped by [`put_motion_vector`] under `fcode`.
    pub mvds: Vec<(i32, i32)>,
    /// `vop_fcode_forward` the differentials are coded under.
    pub fcode: u8,
    /// Intra-DC differentials (Figure 6-8 order) when the macroblock is
    /// intra and `use_intra_dc_vlc` holds.
    pub intra_dc: Option<[i32; 6]>,
    /// Per-block AC EVENT lists (DC excluded when `intra_dc` carries
    /// it).
    pub blocks: [Vec<AcEvent>; 6],
}

impl MbFields {
    fn is_intra(&self) -> bool {
        self.mb_type >= 3
    }

    fn table(&self) -> TcoefTable {
        if self.is_intra() {
            TcoefTable::Intra
        } else {
            TcoefTable::Inter
        }
    }

    fn write_mcbpc(&self, bw: &mut BitWriter, intra_vop: bool) {
        if intra_vop {
            put_mcbpc_i(bw, self.mb_type, self.cbpc);
        } else {
            put_mcbpc_p(bw, self.mb_type, self.cbpc);
        }
    }

    fn write_intra_dc(&self, bw: &mut BitWriter) {
        if let Some(dc) = &self.intra_dc {
            for (i, &d) in dc.iter().enumerate() {
                put_intra_dc(bw, DcComponent::from_block_index(i), d);
            }
        }
    }

    fn write_mvs(&self, bw: &mut BitWriter) {
        for &(dx, dy) in &self.mvds {
            put_motion_vector(bw, dx, dy, self.fcode);
        }
    }

    /// Partition-3 (or combined-syntax) texture: the AC EVENTs of
    /// every coded block, RVLC-coded when `rvlc`.
    fn write_ac(&self, bw: &mut BitWriter, rvlc: bool) {
        let table = self.table();
        for ev in &self.blocks {
            if !ev.is_empty() {
                if rvlc {
                    put_ac_events_rvlc(bw, table, ev);
                } else {
                    put_ac_events(bw, table, ev);
                }
            }
        }
    }

    /// §6.2.6 `macroblock()` in a combined-syntax VOP.
    fn write_combined(&self, bw: &mut BitWriter, intra_vop: bool) {
        if !intra_vop {
            bw.write_bit(self.not_coded);
            if self.not_coded {
                return;
            }
        }
        self.write_mcbpc(bw, intra_vop);
        if self.is_intra() {
            bw.write_bit(self.ac_pred_flag);
        }
        put_cbpy(bw, self.cbpy, self.is_intra());
        if let Some(d) = self.dquant {
            put_dquant(bw, d);
        }
        self.write_mvs(bw);
        // §6.2.7 block(): the DC prologue rides inside each block.
        let table = self.table();
        for (i, ev) in self.blocks.iter().enumerate() {
            if let Some(dc) = &self.intra_dc {
                put_intra_dc(bw, DcComponent::from_block_index(i), dc[i]);
            }
            if !ev.is_empty() {
                put_ac_events(bw, table, ev);
            }
        }
    }

    /// The bit cost of this macroblock under `layout` (the sum of its
    /// partition contributions).
    pub(crate) fn bit_cost(&self, layout: Layout, rvlc: bool) -> usize {
        let mut p = [BitWriter::new(), BitWriter::new(), BitWriter::new()];
        self.write_into(&mut p, layout, rvlc);
        p.iter().map(BitWriter::bit_position).sum()
    }

    fn write_into(&self, p: &mut [BitWriter; 3], layout: Layout, rvlc: bool) {
        match layout {
            Layout::Combined => {
                // The caller distinguishes I-VOP (no not_coded, Table
                // B.6) from P-VOP by the writer's layout; combined
                // I-VOPs are tagged through `mvds.is_empty() &&
                // mb_type >= 3` never carrying `not_coded`, so the
                // PacketWriter passes the VOP type explicitly.
                unreachable!("combined layout is written by PacketWriter::push")
            }
            Layout::PartitionedI => {
                self.write_mcbpc(&mut p[0], true);
                if let Some(d) = self.dquant {
                    put_dquant(&mut p[0], d);
                }
                self.write_intra_dc(&mut p[0]);
                p[1].write_bit(self.ac_pred_flag);
                put_cbpy(&mut p[1], self.cbpy, true);
                self.write_ac(&mut p[2], rvlc);
            }
            Layout::PartitionedP => {
                p[0].write_bit(self.not_coded);
                if self.not_coded {
                    return;
                }
                self.write_mcbpc(&mut p[0], false);
                self.write_mvs(&mut p[0]);
                if self.is_intra() {
                    p[1].write_bit(self.ac_pred_flag);
                }
                put_cbpy(&mut p[1], self.cbpy, self.is_intra());
                if let Some(d) = self.dquant {
                    put_dquant(&mut p[1], d);
                }
                self.write_intra_dc(&mut p[1]);
                self.write_ac(&mut p[2], rvlc);
            }
        }
    }
}

/// The per-VOP values a `video_packet_header` restates under
/// `header_extension_code == 1`, plus the marker / field widths.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PacketVopInfo {
    pub coding_type: VopCodingType,
    pub fcode_fwd: u8,
    pub fcode_bwd: u8,
    pub modulo_time_base: u32,
    pub time_increment: u16,
    pub time_increment_bits: u8,
    pub intra_dc_vlc_thr: u8,
    pub total_macroblocks: u32,
}

/// Serialises a VOP's macroblock layer into video packets under a
/// [`Layout`].
#[derive(Debug)]
pub(crate) struct PacketWriter {
    main: BitWriter,
    cfg: ResilienceConfig,
    info: PacketVopInfo,
    layout: Layout,
    /// Partition writers of the open packet (combined syntax uses
    /// `parts[0]` only).
    parts: [BitWriter; 3],
    packet_start_mb: usize,
    packets_cut: usize,
}

impl PacketWriter {
    /// Wrap `main` (positioned just after the VOP header) for a VOP
    /// described by `info`. `layout` must be [`Layout::Combined`]
    /// unless `cfg.data_partitioned`.
    pub(crate) fn new(
        main: BitWriter,
        cfg: ResilienceConfig,
        info: PacketVopInfo,
        layout: Layout,
    ) -> Self {
        assert!(
            cfg.data_partitioned || layout == Layout::Combined,
            "partitioned layout on a combined-syntax VOL"
        );
        assert!(
            !cfg.reversible_vlc || cfg.data_partitioned,
            "reversible_vlc requires data_partitioned"
        );
        Self {
            main,
            cfg,
            info,
            layout,
            parts: [BitWriter::new(), BitWriter::new(), BitWriter::new()],
            packet_start_mb: 0,
            packets_cut: 0,
        }
    }

    /// Whether the texture partition uses the reversible VLC.
    fn rvlc(&self) -> bool {
        self.cfg.reversible_vlc && self.layout != Layout::Combined
    }

    /// The direct writer for combined-syntax macroblocks (B-VOPs).
    pub(crate) fn writer(&mut self) -> &mut BitWriter {
        assert_eq!(self.layout, Layout::Combined);
        &mut self.parts[0]
    }

    /// Bits accumulated in the open packet's macroblock layer.
    fn open_bits(&self) -> usize {
        self.parts.iter().map(BitWriter::bit_position).sum()
    }

    /// Number of video packets cut so far (resync markers emitted).
    pub(crate) fn packets_cut(&self) -> usize {
        self.packets_cut
    }

    /// Before macroblock `mb_index`: cut a new video packet when the
    /// open one has reached the size target. `quant_scale` is the
    /// running quantiser the new packet re-seeds (§6.3.5). Returns
    /// `true` when a cut happened — the caller must then reset its
    /// prediction state exactly as the decoder does.
    pub(crate) fn maybe_cut(&mut self, mb_index: usize, quant_scale: u32) -> bool {
        if self.cfg.packet_bits == 0
            || mb_index == self.packet_start_mb
            || self.open_bits() < self.cfg.packet_bits as usize
        {
            return false;
        }
        self.flush_packet();
        self.packets_cut += 1;
        self.packet_start_mb = mb_index;
        self.write_packet_header(mb_index as u32, quant_scale);
        true
    }

    /// §6.2.5 `video_packet_header()` (rectangular shape).
    fn write_packet_header(&mut self, macroblock_number: u32, quant_scale: u32) {
        let bw = &mut self.main;
        // §5.2.5 next_resync_marker(): the §5.2.4 stuffing.
        bw.next_start_code();
        let len = resync_marker_length(
            self.info.coding_type,
            self.info.fcode_fwd,
            self.info.fcode_bwd,
        );
        // `len - 1` zeros then a one.
        bw.write_bits(1, usize::from(len));
        let mb_bits = macroblock_number_bit_width(self.info.total_macroblocks);
        bw.write_bits(macroblock_number, usize::from(mb_bits));
        assert!((1..=31).contains(&quant_scale));
        bw.write_bits(quant_scale, 5); // quant_precision 5
        let hec = self.packets_cut % 2 == 1;
        bw.write_bit(hec); // header_extension_code
        if hec {
            for _ in 0..self.info.modulo_time_base {
                bw.write_bit(true);
            }
            bw.write_bit(false);
            bw.write_marker();
            bw.write_bits(
                u32::from(self.info.time_increment),
                usize::from(self.info.time_increment_bits),
            );
            bw.write_marker();
            bw.write_bits(self.info.coding_type.to_bits(), 2);
            bw.write_bits(u32::from(self.info.intra_dc_vlc_thr), 3);
            if !matches!(self.info.coding_type, VopCodingType::I) {
                bw.write_bits(u32::from(self.info.fcode_fwd), 3);
            }
            if matches!(self.info.coding_type, VopCodingType::B) {
                bw.write_bits(u32::from(self.info.fcode_bwd), 3);
            }
        }
    }

    /// Append one macroblock to the open packet.
    pub(crate) fn push(&mut self, fields: &MbFields) {
        match self.layout {
            Layout::Combined => {
                let intra_vop = matches!(self.info.coding_type, VopCodingType::I);
                fields.write_combined(&mut self.parts[0], intra_vop);
            }
            other => {
                let rvlc = self.rvlc();
                fields.write_into(&mut self.parts, other, rvlc);
            }
        }
    }

    /// The bit cost of `fields` under this writer's layout.
    pub(crate) fn cost_of(&self, fields: &MbFields) -> usize {
        match self.layout {
            Layout::Combined => {
                let mut bw = BitWriter::new();
                let intra_vop = matches!(self.info.coding_type, VopCodingType::I);
                fields.write_combined(&mut bw, intra_vop);
                bw.bit_position()
            }
            other => fields.bit_cost(other, self.rvlc()),
        }
    }

    /// Close the open packet into `main`: partitions in order with the
    /// §6.2.5.3 marker between the first two.
    fn flush_packet(&mut self) {
        let parts = std::mem::replace(
            &mut self.parts,
            [BitWriter::new(), BitWriter::new(), BitWriter::new()],
        );
        self.main.append(&parts[0]);
        match self.layout {
            Layout::Combined => {}
            Layout::PartitionedI => self.main.write_bits(DC_MARKER, DC_MARKER_BITS),
            Layout::PartitionedP => self.main.write_bits(MOTION_MARKER, MOTION_MARKER_BITS),
        }
        self.main.append(&parts[1]);
        self.main.append(&parts[2]);
    }

    /// Close the last packet, stuff to the byte boundary and return the
    /// finished unit.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.flush_packet();
        self.main.next_start_code();
        self.main.into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::video_packet::{parse_video_packet_header, VideoPacketContext};

    fn info(coding_type: VopCodingType) -> PacketVopInfo {
        PacketVopInfo {
            coding_type,
            fcode_fwd: 2,
            fcode_bwd: 1,
            modulo_time_base: 1,
            time_increment: 7,
            time_increment_bits: 5,
            intra_dc_vlc_thr: 0,
            total_macroblocks: 12,
        }
    }

    fn ctx(coding_type: VopCodingType) -> VideoPacketContext {
        VideoPacketContext {
            coding_type,
            fcode_fwd: 2,
            fcode_bwd: 1,
            quant_precision: 5,
            time_increment_resolution: 25,
            video_object_layer_shape: 0,
            resync_marker_disable: false,
            newpred_enable: false,
            reduced_resolution_vop_enable: false,
            sprite_gmc: false,
            total_macroblocks: 12,
        }
    }

    /// Two headers (HEC on then off) parse back through the decoder's
    /// video_packet_header parser with the restated fields intact.
    #[test]
    fn packet_headers_round_trip_both_hec_branches() {
        for coding in [VopCodingType::I, VopCodingType::P, VopCodingType::B] {
            let cfg = ResilienceConfig {
                packet_bits: 1,
                ..Default::default()
            };
            let mut pw = PacketWriter::new(BitWriter::new(), cfg, info(coding), Layout::Combined);
            pw.writer().write_bits(0b101, 3);
            assert!(pw.maybe_cut(3, 9));
            pw.writer().write_bits(0b1, 1);
            assert!(pw.maybe_cut(7, 12));
            let bytes = pw.finish();
            let mut br = BitReader::new(&bytes);
            br.skip_bits(3).unwrap();
            let h = parse_video_packet_header(&mut br, &ctx(coding)).unwrap();
            assert_eq!((h.macroblock_number, h.quant_scale), (3, 9));
            assert!(h.header_extension_code, "first cut carries the HEC body");
            let (first, second) = (h, {
                br.skip_bits(1).unwrap();
                parse_video_packet_header(&mut br, &ctx(coding)).unwrap()
            });
            assert_eq!((second.macroblock_number, second.quant_scale), (7, 12));
            assert!(!second.header_extension_code);
            let h = first;
            assert_eq!(h.modulo_time_base, Some(1));
            assert_eq!(h.vop_time_increment, Some(7));
            assert_eq!(h.vop_coding_type, Some(coding));
            assert_eq!(h.intra_dc_vlc_thr, Some(0));
            assert_eq!(
                h.vop_fcode_forward,
                (!matches!(coding, VopCodingType::I)).then_some(2)
            );
            assert_eq!(
                h.vop_fcode_backward,
                matches!(coding, VopCodingType::B).then_some(1)
            );
        }
    }

    #[test]
    fn no_cut_without_a_size_target_or_at_the_packet_start() {
        let mut pw = PacketWriter::new(
            BitWriter::new(),
            ResilienceConfig::default(),
            info(VopCodingType::P),
            Layout::Combined,
        );
        pw.writer().write_bits(0xFFFF, 16);
        assert!(!pw.maybe_cut(5, 4));
        let mut pw = PacketWriter::new(
            BitWriter::new(),
            ResilienceConfig {
                packet_bits: 1,
                ..Default::default()
            },
            info(VopCodingType::P),
            Layout::Combined,
        );
        pw.writer().write_bits(0xFFFF, 16);
        assert!(!pw.maybe_cut(0, 4), "never cut at the packet's own start");
    }
}

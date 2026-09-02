//! §6.2 configuration-header emission + rectangular progressive
//! **I-VOP encoder**.
//!
//! This is the first end-to-end stage of the encoder arc. The header
//! emitters write the §6.2.1/§6.2.2/§6.2.3 configuration run
//! (VisualObjectSequence → VisualObject → VideoObject → VideoObjectLayer)
//! and the §6.2.5 VOP header; [`encode_i_vop`] runs the §6.2.6/§6.2.7
//! combined-syntax intra macroblock walk:
//!
//! * per block: Annex A.1 forward DCT ([`crate::fdct`]) → forward
//!   quantisation ([`crate::quantise`], method 1 or 2 per the VOL's
//!   `quant_type`) → §7.4.3 spatial prediction *emission* (the exact
//!   inverse of the decoder's predictor add: the same Figure 7-5
//!   neighbour resolution via [`IntraBlockGrid`], the same §7.4.3.1
//!   direction rule, differentials instead of adds) → §7.4.2 forward
//!   scan → Table B.16 VLC emission ([`crate::vlc_encode`]);
//! * per macroblock: `mcbpc` / `ac_pred_flag` / `cbpy` header
//!   emission, with the `ac_pred_flag` decided by **measured cost**
//!   (both variants are emitted to probe writers and the cheaper one
//!   is kept — the spec leaves the decision to the encoder);
//! * per VOP: the finished unit is **decoded back through the
//!   crate's own decoder walk** ([`decode_i_vop_macroblocks`]) and
//!   the reconstruction returned to the caller — the closed decode
//!   loop that keeps encoder reference state drift-free by
//!   construction.
//!
//! The emitted VOL is deliberately minimal-and-literal: rectangular
//! shape, verid 1, `obmc_disable == 1`, `resync_marker_disable == 1`,
//! combined syntax, method-1 or method-2 quantisation with the
//! default matrices, `intra_dc_vlc_thr == 0` (Table 6-25: DC VLC for
//! the whole VOP).
//!
//! Provenance: every field layout mirrors the crate's §6.2 parsers,
//! themselves transcribed from ISO/IEC 14496-2:2004 (3rd edition),
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`
//! (§6.2.2 VisualObject / Table G.1 profiles / Table 6-11 object
//! types / §6.2.3 VOL / §6.2.5 VOP / §6.2.6–§6.2.7 macroblock and
//! block syntax). No third-party source was consulted.

use crate::bitreader::BitReader;
use crate::bitwriter::BitWriter;
use crate::block::intra_quant_matrix;
use crate::data_partition::use_intra_dc_vlc;
use crate::fdct::forward_dct_8x8;
use crate::framestore::DecodedFrame;
use crate::inverse_quant::{inverse_quant_intra_dc, saturate_fprime};
use crate::neighbour::{BlockNeighbour, IntraBlockGrid};
use crate::packet_encode::{InterlacedMbInfo, Layout, MbFields, PacketVopInfo, PacketWriter};
use crate::predictor::{dc_scaler, predict_intra_dc, select_dc_direction};
use crate::quantise::{quantise_intra_dc, quantise_method1_intra, quantise_method2_intra};
use crate::scan::{inverse_scan, select_scan_type, DcPredictionDirection, ScanType};
use crate::texture::{AcEvent, DcComponent};
use crate::vop::{vop_time_increment_bits, VopCodingType};
use crate::vop_decode::decode_i_vop_macroblocks;

/// Start codes the emitters write (§6.2.1 Table 6-3).
const VISUAL_OBJECT_SEQUENCE_START_CODE: u32 = 0x0000_01B0;
const VISUAL_OBJECT_START_CODE: u32 = 0x0000_01B5;
const VIDEO_OBJECT_START_CODE: u32 = 0x0000_0100; // video_object id 0
const VIDEO_OBJECT_LAYER_START_CODE: u32 = 0x0000_0120; // VOL id 0
const VOP_START_CODE: u32 = 0x0000_01B6;

/// Static encoder configuration — the fields that shape the VOL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Visible luma width in samples (13-bit field, 1..=8191).
    pub width: u16,
    /// Visible luma height in samples (13-bit field, 1..=8191).
    pub height: u16,
    /// §6.3.3 `vop_time_increment_resolution` — ticks per second.
    pub time_increment_resolution: u16,
    /// `quant_type`: `false` = §7.4.4.2 method 2, `true` = §7.4.4.1
    /// method 1 with the default matrices.
    pub quant_type: bool,
    /// Enable the §7.4.3.3 AC-prediction emission (per-macroblock
    /// cost-decided). DC prediction is always on — it is not optional
    /// in the syntax.
    pub ac_prediction: bool,
    /// Enable §6.3.7 four-motion-vector (`inter4v`) P-VOP macroblocks
    /// (per-macroblock cost-decided). Purely an encode-side mode
    /// choice — the VOL carries no flag for it.
    pub four_mv: bool,
    /// `quarter_sample` (§6.3.3): emit motion vectors on the
    /// quarter-sample grid and compensate through the §7.6.2.2
    /// interpolation. Requires (and selects) the verid-2 VOL layout
    /// and the ASP profile.
    pub quarter_sample: bool,
    /// The stream will contain B-VOPs (`crate::bvop_encode`). Affects
    /// only the profile signalling (B-VOPs are an ASP tool; the VOL
    /// carries no flag for them).
    pub b_vops: bool,
    /// Optional §6.2.3 `vbv_parameters` signalling (emitted inside a
    /// `vol_control_parameters` block; see [`VbvSignalling`] /
    /// `crate::rate_control`).
    pub vbv: Option<VbvSignalling>,
    /// `vop_fcode_forward` / `vop_fcode_backward` (1..=7) for every
    /// P- and B-VOP: selects the Table 7-9 motion-vector range
    /// `[-32·2^(fcode-1), 32·2^(fcode-1) - 1]` (half- or
    /// quarter-sample units per `quarter_sample`) and the matching
    /// §7.6 search window. Default 1.
    pub fcode: u8,
    /// Per-macroblock quantiser modulation (`crate::mb_quant`):
    /// activity-classed `dquant` (I-/P-VOPs, `inter+q` / `intra+q`
    /// macroblock types) and `dbquant` (B-VOPs) steps around the VOP
    /// quantiser. Default off.
    pub adaptive_quant: bool,
    /// Error-resilience tools (`crate::packet_encode`): video packets,
    /// data partitioning, reversible VLCs. Default: none.
    pub resilience: crate::packet_encode::ResilienceConfig,
    /// `sprite_enable == "GMC"` with one warping point at half-pel
    /// accuracy (`crate::svop_encode`): anchors after the first I-VOP
    /// are S(GMC)-VOPs. Requires (and selects) the verid-2 VOL and
    /// the ASP profile; incompatible with `data_partitioned`.
    pub gmc: bool,
    /// `no_of_sprite_warping_points` (1..=3) of a GMC VOL: one point
    /// codes a pure translation, two a §7.8.5 similarity (rotation +
    /// isotropic scale + translation), three a full affine warp — the
    /// encoder fits the model to its per-macroblock motion field.
    /// Ignored unless `gmc`. Default 1.
    pub gmc_points: u8,
    /// `interlaced` (§6.3.3): the VOL codes interlaced VOPs — every
    /// VOP header carries `top_field_first` /
    /// `alternate_vertical_scan_flag`, every macroblock the §6.2.6.3
    /// `interlaced_information()` body (field DCT decided per
    /// macroblock, §7.7.2.1 field prediction cost-decided per inter
    /// macroblock). Selects the ASP profile; incompatible with
    /// `data_partitioned` and `gmc` (the decoder's S(GMC) walk is
    /// progressive-only).
    pub interlaced: bool,
    /// §6.3.5 `top_field_first` written on every VOP of an interlaced
    /// VOL (ignored otherwise).
    pub top_field_first: bool,
    /// §6.3.5 `alternate_vertical_scan_flag`: code every block of an
    /// interlaced VOP with the Figure 7-4 (b) alternate-vertical scan
    /// (ignored on a progressive VOL).
    pub alternate_scan: bool,
    /// `short_video_header == 1` (§6.2.5.2): emit H.263-compatible
    /// short-header pictures instead of the VOS/VOL/VOP syntax
    /// (`crate::short_header_encode`). The dimensions must be one of
    /// the Table 6-29 source formats; the Table 6-28 fixed tool set
    /// applies (every other tool flag must be at its default).
    pub short_header: bool,
    /// Short header only: emit a GOB header (`gob_resync_marker`,
    /// `gob_number`, `gob_frame_id`, `quant_scale`) on every GOB after
    /// the first. Default on.
    pub gob_headers: bool,
    /// §6.3.5 `intra_dc_vlc_thr` (Table 6-25, 0..=7) written on every
    /// VOP header: the running-quantiser threshold above which an intra
    /// macroblock's DC differentials ride the AC (Table B.16) VLC
    /// instead of the DC VLC. Default 0 (DC VLC for the whole VOP).
    pub intra_dc_vlc_thr: u8,
}

/// The two §6.3.5 interlaced VOP-header flags, written right after
/// `intra_dc_vlc_thr` when the VOL codes `interlaced == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VopInterlaceFlags {
    /// `top_field_first`.
    pub top_field_first: bool,
    /// `alternate_vertical_scan_flag`.
    pub alternate_vertical_scan: bool,
}

impl VopInterlaceFlags {
    /// Emit the two flags (call only on an interlaced VOL).
    pub(crate) fn write(self, bw: &mut BitWriter) {
        bw.write_bit(self.top_field_first);
        bw.write_bit(self.alternate_vertical_scan);
    }
}

/// The §6.2.6.3 forward field-DCT permutation (Figure 6-12): the four
/// luminance blocks of a `dct_type == 1` macroblock carry the top
/// field's lines (blocks 0/1, rows `0, 2, …, 14`) and the bottom
/// field's lines (blocks 2/3, rows `1, 3, …, 15`). Exact inverse of
/// [`crate::reconstruct::inverse_field_dct_luma`].
pub fn field_dct_luma(luma: &[[i32; 16]; 16]) -> [[i32; 16]; 16] {
    let mut out = [[0i32; 16]; 16];
    for k in 0..8 {
        out[k] = luma[2 * k];
        out[8 + k] = luma[2 * k + 1];
    }
    out
}

/// The encoder's `dct_type` election for one 16×16 luminance block
/// (source samples for an intra macroblock, the prediction residual
/// for an inter one): field DCT when adjacent same-field lines
/// correlate better than adjacent frame lines — the mean absolute
/// vertical difference over the 14 same-field line pairs against the
/// 15 frame line pairs. The spec leaves the decision to the encoder;
/// this is a pure content statistic, independent of any predictor
/// state, so it can be taken before the blocks are quantised.
pub fn elect_field_dct(luma: &[[i32; 16]; 16]) -> bool {
    let row_diff = |a: &[i32; 16], b: &[i32; 16]| -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(&p, &q)| u64::from((p - q).unsigned_abs()))
            .sum()
    };
    let frame: u64 = luma.windows(2).map(|w| row_diff(&w[0], &w[1])).sum();
    let field: u64 = luma.windows(3).map(|w| row_diff(&w[0], &w[2])).sum();
    // Compare per-pair means: field / 14 < frame / 15.
    field * 15 < frame * 14
}

/// The §6.2.3 `vbv_parameters` triple (Annex D rate-buffer model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VbvSignalling {
    /// Peak rate in 400-bit-per-second units (30 bits, non-zero).
    pub bit_rate_400: u32,
    /// Buffer size in 16384-bit units (18 bits, non-zero).
    pub buffer_units: u32,
    /// Initial occupancy in 64-bit units (26 bits).
    pub occupancy_64: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            time_increment_resolution: 25,
            quant_type: false,
            ac_prediction: true,
            four_mv: false,
            quarter_sample: false,
            b_vops: false,
            vbv: None,
            fcode: 1,
            adaptive_quant: false,
            resilience: crate::packet_encode::ResilienceConfig::default(),
            gmc: false,
            gmc_points: 1,
            interlaced: false,
            top_field_first: true,
            alternate_scan: false,
            short_header: false,
            gob_headers: true,
            intra_dc_vlc_thr: 0,
        }
    }
}

impl EncoderConfig {
    /// Whether the configuration uses any Advanced-Simple-profile
    /// tool (method-1 quantisation, quarter-sample MC, B-VOPs).
    fn uses_asp_tools(&self) -> bool {
        self.quant_type || self.quarter_sample || self.b_vops || self.gmc || self.interlaced
    }

    /// The §6.3.5 VOP-header interlace flags for this configuration
    /// (`None` on a progressive VOL).
    pub fn vop_interlace(&self) -> Option<VopInterlaceFlags> {
        self.interlaced.then_some(VopInterlaceFlags {
            top_field_first: self.top_field_first,
            alternate_vertical_scan: self.alternate_scan,
        })
    }

    /// The scan every block of a VOP uses when the §6.3.5
    /// `alternate_vertical_scan_flag` overrides the §7.4.2 per-block
    /// selection (`None` = the per-block rule).
    pub(crate) fn forced_scan(&self) -> Option<ScanType> {
        (self.interlaced && self.alternate_scan).then_some(ScanType::AlternateVertical)
    }

    /// Whether the VOL needs `video_object_layer_verid == 2` (the
    /// `quarter_sample` flag and the 2-bit `sprite_enable` with its
    /// GMC value only exist there).
    fn verid2(&self) -> bool {
        self.quarter_sample || self.gmc
    }

    /// Table G.1 `profile_and_level_indication` for this
    /// configuration: Simple Profile/Level 3 (`0x03`) for the plain
    /// method-2 tool set, Advanced Simple Profile/Level 3 (`0xF3`)
    /// once an ASP tool (method-1 quantisation, quarter-sample) is
    /// selected.
    pub fn profile_and_level(&self) -> u8 {
        if self.uses_asp_tools() {
            0xF3
        } else {
            0x03
        }
    }

    /// Table 6-11 `video_object_type_indication`: Simple (`1`) or
    /// Advanced Simple (`0x11`), matching
    /// [`Self::profile_and_level`].
    pub fn video_object_type(&self) -> u8 {
        if self.uses_asp_tools() {
            0x11
        } else {
            0x01
        }
    }

    /// Macroblock grid dimensions (`§6.3.3`: a partial edge macroblock
    /// is still a whole coded macroblock).
    pub fn mb_dimensions(&self) -> (usize, usize) {
        (
            usize::from(self.width).div_ceil(16),
            usize::from(self.height).div_ceil(16),
        )
    }
}

/// Emit the §6.2 configuration run: VisualObjectSequence (start code +
/// Table G.1 profile), VisualObject (`video ID`, no signal type),
/// VideoObject id 0, and the VideoObjectLayer described by `cfg`.
/// Every unit ends with the §5.2.4 stuffing so the next start code is
/// byte-aligned.
pub fn write_configuration_headers(cfg: &EncoderConfig) -> Vec<u8> {
    assert!(
        !cfg.short_header,
        "a short-header stream carries no configuration headers"
    );
    let mut bw = BitWriter::new();
    // VisualObjectSequence() — §6.2.2.
    bw.write_start_code(VISUAL_OBJECT_SEQUENCE_START_CODE);
    bw.write_bits(u32::from(cfg.profile_and_level()), 8);
    bw.next_start_code();
    // VisualObject() — §6.2.2.
    bw.write_start_code(VISUAL_OBJECT_START_CODE);
    bw.write_bit(false); // is_visual_object_identifier = 0 (verid defaults to 1)
    bw.write_bits(1, 4); // visual_object_type = video ID (Table 6-6)
    bw.write_bit(false); // video_signal_type = 0 (§6.3.2.4 defaults)
    bw.next_start_code();
    // VideoObject() — the bare video_object_start_code (id 0).
    bw.write_start_code(VIDEO_OBJECT_START_CODE);
    // VideoObjectLayer() — §6.2.3, rectangular / verid-1 branch.
    bw.write_start_code(VIDEO_OBJECT_LAYER_START_CODE);
    bw.write_bit(false); // random_accessible_vol
    bw.write_bits(u32::from(cfg.video_object_type()), 8);
    if cfg.verid2() {
        // §6.2.3: the quarter_sample flag and the GMC sprite_enable
        // value exist only when video_object_layer_verid != 1, so
        // declare verid 2.
        bw.write_bit(true); // is_object_layer_identifier = 1
        bw.write_bits(2, 4); // video_object_layer_verid = 2
        bw.write_bits(1, 3); // video_object_layer_priority = 1
    } else {
        bw.write_bit(false); // is_object_layer_identifier = 0 → verid 1
    }
    bw.write_bits(1, 4); // aspect_ratio_info = 1:1
    if cfg.b_vops || cfg.vbv.is_some() {
        // §6.2.3 vol_control_parameters: declare `low_delay` (0 when
        // the VOL contains B-VOPs, so a decoder never has to guess the
        // reorder delay) and the Annex D vbv_parameters when rate
        // control is active.
        bw.write_bit(true); // vol_control_parameters = 1
        bw.write_bits(0b01, 2); // chroma_format = 4:2:0 (Table 6-15)
        bw.write_bit(!cfg.b_vops); // low_delay
        match &cfg.vbv {
            Some(v) => {
                assert!(v.bit_rate_400 > 0 && v.bit_rate_400 < (1 << 30));
                assert!(v.buffer_units > 0 && v.buffer_units < (1 << 18));
                assert!(v.occupancy_64 < (1 << 26));
                bw.write_bit(true); // vbv_parameters = 1
                bw.write_bits(v.bit_rate_400 >> 15, 15); // first_half_bit_rate
                bw.write_marker();
                bw.write_bits(v.bit_rate_400 & 0x7FFF, 15); // latter_half_bit_rate
                bw.write_marker();
                bw.write_bits(v.buffer_units >> 3, 15); // first_half_vbv_buffer_size
                bw.write_marker();
                bw.write_bits(v.buffer_units & 0x7, 3); // latter_half_vbv_buffer_size
                bw.write_bits(v.occupancy_64 >> 15, 11); // first_half_vbv_occupancy
                bw.write_marker();
                bw.write_bits(v.occupancy_64 & 0x7FFF, 15); // latter_half_vbv_occupancy
                bw.write_marker();
            }
            None => bw.write_bit(false), // vbv_parameters = 0
        }
    } else {
        bw.write_bit(false); // vol_control_parameters = 0
    }
    bw.write_bits(0, 2); // video_object_layer_shape = rectangular
    bw.write_marker();
    bw.write_bits(u32::from(cfg.time_increment_resolution), 16);
    bw.write_marker();
    bw.write_bit(false); // fixed_vop_rate = 0
    bw.write_marker();
    bw.write_bits(u32::from(cfg.width), 13);
    bw.write_marker();
    bw.write_bits(u32::from(cfg.height), 13);
    bw.write_marker();
    if cfg.interlaced {
        assert!(
            !cfg.resilience.data_partitioned && !cfg.gmc,
            "interlaced VOLs use the combined syntax without GMC"
        );
    }
    bw.write_bit(cfg.interlaced); // interlaced
    bw.write_bit(true); // obmc_disable = 1
    if cfg.verid2() {
        // sprite_enable (verid 2 → 2 bits): 10 = GMC, 00 = not used.
        bw.write_bits(if cfg.gmc { 0b10 } else { 0b00 }, 2);
        if cfg.gmc {
            assert!(
                !cfg.resilience.data_partitioned,
                "GMC S-VOPs use the combined syntax only"
            );
            assert!(
                (1..=3).contains(&cfg.gmc_points),
                "GMC needs 1..=3 warping points"
            );
            bw.write_bits(u32::from(cfg.gmc_points), 6); // no_of_sprite_warping_points
            bw.write_bits(0b00, 2); // sprite_warping_accuracy = 1/2 pel
            bw.write_bit(false); // sprite_brightness_change = 0 (§6.3.3)
        }
    } else {
        bw.write_bit(false); // sprite_enable = 0 (verid 1 → 1 bit)
    }
    bw.write_bit(false); // not_8_bit = 0
    bw.write_bit(cfg.quant_type); // quant_type
    if cfg.quant_type {
        bw.write_bit(false); // load_intra_quant_mat = 0 (default matrix)
        bw.write_bit(false); // load_nonintra_quant_mat = 0
    }
    if cfg.verid2() {
        // Present only under verid != 1 (declared above).
        bw.write_bit(cfg.quarter_sample); // quarter_sample
    }
    bw.write_bit(true); // complexity_estimation_disable = 1
    bw.write_bit(cfg.resilience.packet_bits == 0); // resync_marker_disable
    bw.write_bit(cfg.resilience.data_partitioned); // data_partitioned
    if cfg.resilience.data_partitioned {
        bw.write_bit(cfg.resilience.reversible_vlc); // reversible_vlc
    } else {
        assert!(
            !cfg.resilience.reversible_vlc,
            "reversible_vlc requires data_partitioned (§6.2.3)"
        );
    }
    if cfg.verid2() {
        // Present only under verid != 1.
        bw.write_bit(false); // newpred_enable = 0
        bw.write_bit(false); // reduced_resolution_vop_enable = 0
    }
    bw.write_bit(false); // scalability = 0
    bw.next_start_code();
    bw.into_bytes()
}

/// Emit a §6.2.5 I-VOP header (through `vop_quant`; an I-VOP carries
/// no `vop_fcode_forward`). The writer is left mid-unit — the
/// macroblock walk follows immediately.
pub fn write_i_vop_header(
    bw: &mut BitWriter,
    resolution: u16,
    modulo_time_base: u32,
    time_increment: u16,
    quant: u32,
    interlace: Option<VopInterlaceFlags>,
    intra_dc_vlc_thr: u8,
) {
    bw.write_start_code(VOP_START_CODE);
    bw.write_bits(0b00, 2); // vop_coding_type = I
    for _ in 0..modulo_time_base {
        bw.write_bit(true);
    }
    bw.write_bit(false); // modulo_time_base terminator
    bw.write_marker();
    bw.write_bits(
        u32::from(time_increment),
        usize::from(vop_time_increment_bits(resolution)),
    );
    bw.write_marker();
    bw.write_bit(true); // vop_coded = 1
    assert!(intra_dc_vlc_thr <= 7, "intra_dc_vlc_thr is a 3-bit field");
    bw.write_bits(u32::from(intra_dc_vlc_thr), 3); // intra_dc_vlc_thr (Table 6-25)
    if let Some(flags) = interlace {
        flags.write(bw); // top_field_first + alternate_vertical_scan_flag
    }
    assert!((1..=31).contains(&quant), "vop_quant {quant} out of range");
    bw.write_bits(quant, 5);
}

/// A borrowed planar 4:2:0 input picture (stride == width; chroma
/// planes are `(width/2 rounded up) × (height/2 rounded up)`).
#[derive(Debug, Clone, Copy)]
pub struct FrameView<'a> {
    /// Luma plane, `width * height` bytes.
    pub y: &'a [u8],
    /// Cb plane.
    pub cb: &'a [u8],
    /// Cr plane.
    pub cr: &'a [u8],
    /// Visible luma width in samples.
    pub width: usize,
    /// Visible luma height in samples.
    pub height: usize,
}

impl<'a> FrameView<'a> {
    /// Chroma plane dimensions (4:2:0 — rounded up).
    fn chroma_dims(&self) -> (usize, usize) {
        (self.width.div_ceil(2), self.height.div_ceil(2))
    }

    /// Fetch one 8×8 block from a plane with edge replication for
    /// samples beyond the visible area (the encoder's padding choice
    /// for the partial edge macroblocks §6.3.3 requires it to code).
    fn block_from(plane: &[u8], pw: usize, ph: usize, y0: usize, x0: usize) -> [[i32; 8]; 8] {
        let mut out = [[0i32; 8]; 8];
        for (dy, row) in out.iter_mut().enumerate() {
            let sy = (y0 + dy).min(ph.saturating_sub(1));
            for (dx, cell) in row.iter_mut().enumerate() {
                let sx = (x0 + dx).min(pw.saturating_sub(1));
                *cell = i32::from(plane[sy * pw + sx]);
            }
        }
        out
    }

    /// [`FrameView::block`] with the §7.7.1 field-DCT permutation
    /// applied to the luminance blocks when `field_dct` (blocks 0/1 =
    /// the top field's 16 lines, 2/3 = the bottom field's); chroma is
    /// untouched by `dct_type`.
    pub(crate) fn block_with_field_dct(
        &self,
        mb_row: usize,
        mb_col: usize,
        i: usize,
        field_dct: bool,
    ) -> [[i32; 8]; 8] {
        if !field_dct || i >= 4 {
            return self.block(mb_row, mb_col, i);
        }
        let permuted = field_dct_luma(&crate::pvop_encode::source_luma_mb(self, mb_row, mb_col));
        let (row0, col0) = (8 * (i / 2), 8 * (i % 2));
        let mut out = [[0i32; 8]; 8];
        for (y, row) in out.iter_mut().enumerate() {
            row.copy_from_slice(&permuted[row0 + y][col0..col0 + 8]);
        }
        out
    }

    /// The Figure 6-8 block `i` (0..=5) of macroblock
    /// `(mb_row, mb_col)`.
    pub(crate) fn block(&self, mb_row: usize, mb_col: usize, i: usize) -> [[i32; 8]; 8] {
        match i {
            0..=3 => {
                let y0 = mb_row * 16 + if i >= 2 { 8 } else { 0 };
                let x0 = mb_col * 16 + if i % 2 == 1 { 8 } else { 0 };
                Self::block_from(self.y, self.width, self.height, y0, x0)
            }
            4 | 5 => {
                let (cw, ch) = self.chroma_dims();
                let plane = if i == 4 { self.cb } else { self.cr };
                Self::block_from(plane, cw, ch, mb_row * 8, mb_col * 8)
            }
            _ => panic!("block index {i} out of the 4:2:0 range"),
        }
    }
}

/// Forward-scan position list for a scan pattern: entry `n` is the
/// `(v, u)` cell whose coefficient is `QFS[n]`. Derived from the
/// decoder's [`inverse_scan`] so the two directions can never drift.
fn scan_positions(scan_type: ScanType) -> [(usize, usize); 64] {
    let mut identity = [0i32; 64];
    for (n, cell) in identity.iter_mut().enumerate() {
        *cell = n as i32;
    }
    let grid = inverse_scan(&identity, scan_type);
    let mut positions = [(0usize, 0usize); 64];
    for v in 0..8 {
        for u in 0..8 {
            positions[grid[v][u] as usize] = (v, u);
        }
    }
    positions
}

/// Serialise an 8×8 `PQF` block into the 1-D `QFS[64]` stream under
/// `scan_type` (the inverse of §7.4.2's inverse scan).
pub(crate) fn forward_scan(pqf: &[[i32; 8]; 8], scan_type: ScanType) -> [i32; 64] {
    let positions = scan_positions(scan_type);
    let mut qfs = [0i32; 64];
    for (n, &(v, u)) in positions.iter().enumerate() {
        qfs[n] = pqf[v][u];
    }
    qfs
}

/// Convert `QFS[from..64]` into the §7.4.1.2 EVENT list (`from` is 1
/// when the DC is carried by the DC VLC, 0 when it rides the AC
/// stream). Empty when every coefficient is zero.
pub(crate) fn qfs_to_events(qfs: &[i32; 64], from: usize) -> Vec<AcEvent> {
    let mut events = Vec::new();
    let mut run = 0u32;
    for &q in &qfs[from..] {
        if q == 0 {
            run += 1;
        } else {
            events.push(AcEvent {
                last: false,
                run,
                level: q,
            });
            run = 0;
        }
    }
    if let Some(last) = events.last_mut() {
        last.last = true;
    }
    events
}

/// One quantised intra block prepared for emission.
pub(crate) struct PreparedBlock {
    /// Quantised coefficients `QF` (DC at `[0][0]`) — the value the
    /// decoder reconstructs after its §7.4.3 predictor adds.
    pub(crate) qf: [[i32; 8]; 8],
    /// The §7.4.4.1.1 inverse-quantised DC (`F[0][0]`), for the
    /// neighbour grid.
    pub(crate) dc_f: i32,
}

/// Quantise one intra block per the VOL's method.
pub(crate) fn quantise_intra_block(
    f: &[[i32; 8]; 8],
    component: DcComponent,
    qp: u32,
    quant_type: bool,
    w_intra: &[[u8; 8]; 8],
) -> PreparedBlock {
    let mut qf = [[0i32; 8]; 8];
    for v in 0..8 {
        for u in 0..8 {
            qf[v][u] = if (v, u) == (0, 0) {
                quantise_intra_dc(f[0][0], component, qp)
            } else if quant_type {
                quantise_method1_intra(f[v][u], w_intra[v][u], qp)
            } else {
                quantise_method2_intra(f[v][u], qp)
            };
        }
    }
    let dc_f = saturate_fprime(inverse_quant_intra_dc(qf[0][0], component, qp, false), 8);
    PreparedBlock { qf, dc_f }
}

/// The per-block emission plan of one variant (`ac_pred_flag` on or
/// off): the DC differential, the AC EVENT list, and the coded flag.
pub(crate) struct BlockPlan {
    pub(crate) dc_differential: i32,
    pub(crate) events: Vec<AcEvent>,
}

/// Build the emission plan for one block under a given `ac_pred_flag`,
/// or `None` when AC prediction pushes a differential outside the
/// escape-codable domain (the caller then falls back to the
/// no-prediction variant for the whole macroblock).
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_block(
    prepared: &PreparedBlock,
    predictors: &crate::block::BlockPredictors,
    direction: DcPredictionDirection,
    component: DcComponent,
    qp: u32,
    ac_pred: bool,
    forced_scan: Option<ScanType>,
    use_dc_vlc: bool,
) -> Option<BlockPlan> {
    let qf = &prepared.qf;
    let scaler = dc_scaler(component, qp);
    // §7.4.3.2 inverse: the emitted differential is QF[0][0] minus the
    // decoder's predictor term (predict_intra_dc with a zero
    // differential *is* that term).
    let dc_pred = predict_intra_dc(0, direction, predictors.fa_dc, predictors.fc_dc, scaler);
    let dc_differential = qf[0][0] - dc_pred;

    // §7.4.3.3 inverse on the first row / column when predicting.
    let mut pqf = *qf;
    // Table 6-25: with the DC VLC the differential rides the block
    // prologue and scan position 0 stays empty; otherwise it is the
    // first coefficient of the AC EVENT stream (§6.2.7 / §7.4.1).
    pqf[0][0] = if use_dc_vlc { 0 } else { dc_differential };
    if ac_pred {
        match direction {
            DcPredictionDirection::FromLeft => {
                if let Some(col) = predictors.a_first_column {
                    for v in 1..8 {
                        pqf[v][0] -= scale_ac_pred(col[v - 1], predictors.qp_a, qp);
                    }
                }
            }
            DcPredictionDirection::FromAbove => {
                if let Some(row) = predictors.c_first_row {
                    for u in 1..8 {
                        pqf[0][u] -= scale_ac_pred(row[u - 1], predictors.qp_c, qp);
                    }
                }
            }
        }
        // A differential outside the Table B.16 escape domain cannot
        // be emitted — reject the variant.
        for row in &pqf {
            for &cell in row {
                if !(-2047..=2047).contains(&cell) {
                    return None;
                }
            }
        }
    }

    // §6.3.5: an interlaced VOP's alternate_vertical_scan_flag
    // overrides the §7.4.2 per-block selection.
    let scan_type = forced_scan.unwrap_or_else(|| select_scan_type(true, ac_pred, direction));
    let qfs = forward_scan(&pqf, scan_type);
    if !use_dc_vlc && !(-2047..=2047).contains(&dc_differential) {
        return None;
    }
    let events = qfs_to_events(&qfs, usize::from(use_dc_vlc));
    Some(BlockPlan {
        dc_differential,
        events,
    })
}

/// The §7.4.3.3 predictor term `(QF_neighbour * Qp_neighbour) // Qp_x`
/// (§4.1 `//`, half away from zero) — the quantity the decoder adds
/// back.
fn scale_ac_pred(qf_neighbour: i32, qp_neighbour: u32, qp_x: u32) -> i32 {
    let n = qf_neighbour * qp_neighbour as i32;
    let d = qp_x as i32;
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

/// Build the [`MbFields`] record of one intra macroblock (`mcbpc` type
/// 3, or 4 with a `dquant`) for a fixed `ac_pred_flag` choice. With
/// `use_dc_vlc` the six DC differentials ride the DC VLC (partition 1
/// / 2 under data partitioning, the per-block prologue otherwise).
pub(crate) fn intra_mb_fields(
    plans: &[BlockPlan; 6],
    ac_pred_flag: bool,
    use_dc_vlc: bool,
    dquant: Option<i8>,
    interlaced: Option<InterlacedMbInfo>,
) -> MbFields {
    let (cbpy, cbpc) = intra_mb_cbp(plans);
    let mut dc = [0i32; 6];
    for (slot, plan) in dc.iter_mut().zip(plans.iter()) {
        *slot = plan.dc_differential;
    }
    MbFields {
        not_coded: false,
        mb_type: if dquant.is_some() { 4 } else { 3 },
        cbpc,
        cbpy,
        ac_pred_flag,
        dquant,
        mcsel: None,
        mvds: Vec::new(),
        fcode: 1,
        intra_dc: use_dc_vlc.then_some(dc),
        blocks: std::array::from_fn(|i| plans[i].events.clone()),
        interlaced,
    }
}

/// The `(cbpy, cbpc)` coded-block pattern of an intra macroblock's six
/// emission plans (§6.3.7 "1 = coded" convention, Figure 6-8 order).
pub(crate) fn intra_mb_cbp(plans: &[BlockPlan; 6]) -> (u8, u8) {
    let coded: Vec<bool> = plans.iter().map(|p| !p.events.is_empty()).collect();
    let cbpy = (u8::from(coded[0]) << 3)
        | (u8::from(coded[1]) << 2)
        | (u8::from(coded[2]) << 1)
        | u8::from(coded[3]);
    let cbpc = (u8::from(coded[4]) << 1) | u8::from(coded[5]);
    (cbpy, cbpc)
}

/// Encode one rectangular progressive I-VOP: VOP header + §6.2.6
/// intra macroblock walk + closing stuffing. Returns the emitted unit
/// (start-code delimited, byte-aligned) and the decoder-loop
/// reconstruction obtained by running the crate's own
/// [`decode_i_vop_macroblocks`] over the freshly emitted bits.
///
/// `vol` must be the parsed form of the VOL emitted by
/// [`write_configuration_headers`] for the same config (see
/// [`crate::encoder`] for the packaged pairing); `qp` is the VOP
/// quantiser (1..=31).
pub fn encode_i_vop(
    vol: &crate::vol::VolHeader,
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    modulo_time_base: u32,
    time_increment: u16,
    qp: u32,
) -> (Vec<u8>, DecodedFrame) {
    encode_i_vop_with_thr(
        vol,
        cfg,
        frame,
        modulo_time_base,
        time_increment,
        qp,
        cfg.intra_dc_vlc_thr,
    )
}

/// [`encode_i_vop`] with the §6.3.5 `intra_dc_vlc_thr` **elected by
/// measured cost**: the VOP is coded under the two extreme Table 6-25
/// settings (0 — DC VLC throughout, 7 — AC VLC throughout) and the
/// smaller unit is kept. Returns the unit, its reconstruction and the
/// winning threshold (the caller carries it to the following P/S
/// VOPs' headers). Ties keep the DC VLC.
pub fn encode_i_vop_elect_thr(
    vol: &crate::vol::VolHeader,
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    modulo_time_base: u32,
    time_increment: u16,
    qp: u32,
) -> (Vec<u8>, DecodedFrame, u8) {
    let dc = encode_i_vop_with_thr(vol, cfg, frame, modulo_time_base, time_increment, qp, 0);
    let ac = encode_i_vop_with_thr(vol, cfg, frame, modulo_time_base, time_increment, qp, 7);
    if ac.0.len() < dc.0.len() {
        (ac.0, ac.1, 7)
    } else {
        (dc.0, dc.1, 0)
    }
}

/// [`encode_i_vop`] with an explicit `intra_dc_vlc_thr` (Table 6-25):
/// each intra macroblock's DC differentials ride the DC VLC while its
/// running quantiser stays below the threshold, and the AC EVENT
/// stream (scan position 0) otherwise — exactly the decoder's
/// `use_intra_dc_vlc` rule.
#[allow(clippy::too_many_arguments)]
pub fn encode_i_vop_with_thr(
    vol: &crate::vol::VolHeader,
    cfg: &EncoderConfig,
    frame: &FrameView<'_>,
    modulo_time_base: u32,
    time_increment: u16,
    qp: u32,
    thr: u8,
) -> (Vec<u8>, DecodedFrame) {
    assert!((1..=31).contains(&qp), "vop_quant {qp} out of range");
    assert!(thr <= 7, "intra_dc_vlc_thr {thr} out of range");
    let (mb_width, mb_height) = cfg.mb_dimensions();
    let w_intra = intra_quant_matrix(vol);

    let mut bw = BitWriter::new();
    write_i_vop_header(
        &mut bw,
        cfg.time_increment_resolution,
        modulo_time_base,
        time_increment,
        qp,
        cfg.vop_interlace(),
        thr,
    );
    let forced_scan = cfg.forced_scan();
    let layout = if cfg.resilience.data_partitioned {
        Layout::PartitionedI
    } else {
        Layout::Combined
    };
    let mut pw = PacketWriter::new(
        bw,
        cfg.resilience,
        PacketVopInfo {
            coding_type: VopCodingType::I,
            fcode_fwd: 0,
            fcode_bwd: 0,
            modulo_time_base,
            time_increment,
            time_increment_bits: vop_time_increment_bits(cfg.time_increment_resolution),
            intra_dc_vlc_thr: thr,
            total_macroblocks: (mb_width * mb_height) as u32,
            interlaced: cfg.interlaced,
            sprite_trajectory: None,
        },
        layout,
    );

    let mut grid = IntraBlockGrid::new(mb_height, mb_width);
    // §6.3.7 running quantiser (seeded by vop_quant, moved by dquant,
    // re-seeded by each video packet's quant_scale).
    let mut running_qp = qp;
    for mb_row in 0..mb_height {
        for mb_col in 0..mb_width {
            if pw.maybe_cut(mb_row * mb_width + mb_col, running_qp) {
                // §E.1.2: no prediction crosses a packet boundary.
                grid = IntraBlockGrid::new(mb_height, mb_width);
            }
            // Per-macroblock quantiser: the activity-classed dquant
            // step from the running value (or the VOP quantiser).
            let (qp, dquant) = if cfg.adaptive_quant {
                let src = crate::pvop_encode::source_luma_mb(frame, mb_row, mb_col);
                let class =
                    crate::mb_quant::activity_class(crate::pvop_encode::intra_activity(&src));
                crate::mb_quant::plan_dquant(running_qp, crate::mb_quant::target_qp(qp, class))
            } else {
                (qp, None)
            };
            running_qp = qp;
            // Table 6-25 against the running quantiser of this MB.
            let use_dc_vlc = use_intra_dc_vlc(thr, qp);
            // §7.7.1 dct_type election (interlaced VOL only): field DCT
            // permutes the luminance lines before the blocks are cut.
            let field_dct = cfg.interlaced
                && elect_field_dct(&crate::pvop_encode::source_luma_mb(frame, mb_row, mb_col));
            // Quantise the six blocks first (grid state is variant-
            // independent — the decoder records post-prediction QF).
            let mut prepared: Vec<PreparedBlock> = Vec::with_capacity(6);
            for i in 0..6 {
                let samples = frame.block_with_field_dct(mb_row, mb_col, i, field_dct);
                let f = forward_dct_8x8(&samples, 8);
                prepared.push(quantise_intra_block(
                    &f,
                    DcComponent::from_block_index(i),
                    qp,
                    cfg.quant_type,
                    &w_intra,
                ));
            }

            // Resolve predictors + direction per block, then build the
            // emission plans for both ac_pred variants. Each block is
            // recorded into the grid *immediately* — the decoder does
            // the same (`decode_intra_mb_with_grid`), so blocks 1..5
            // of this macroblock see blocks 0..4 as Figure 7-5
            // neighbours.
            let mut plans_off: Vec<BlockPlan> = Vec::with_capacity(6);
            let mut plans_on: Option<Vec<BlockPlan>> = if cfg.ac_prediction {
                Some(Vec::with_capacity(6))
            } else {
                None
            };
            for (i, prep) in prepared.iter().enumerate() {
                let component = DcComponent::from_block_index(i);
                let predictors = grid.predictors_for(mb_row, mb_col, i, 8, qp);
                let direction =
                    select_dc_direction(predictors.fa_dc, predictors.fb_dc, predictors.fc_dc);
                let off = plan_block(
                    prep,
                    &predictors,
                    direction,
                    component,
                    qp,
                    false,
                    forced_scan,
                    use_dc_vlc,
                )
                .expect("no-prediction differentials are always codable");
                plans_off.push(off);
                if let Some(on) = plans_on.as_mut() {
                    match plan_block(
                        prep,
                        &predictors,
                        direction,
                        component,
                        qp,
                        true,
                        forced_scan,
                        use_dc_vlc,
                    ) {
                        Some(p) => on.push(p),
                        None => plans_on = None, // fall back for the whole MB
                    }
                }
                grid.record(
                    mb_row,
                    mb_col,
                    i,
                    Some(BlockNeighbour::from_qf(&prep.qf, prep.dc_f, qp)),
                );
            }
            let plans_off: [BlockPlan; 6] = plans_off
                .try_into()
                .unwrap_or_else(|_| unreachable!("six blocks per macroblock"));

            // Measured-cost ac_pred decision.
            let interlaced = cfg.interlaced.then_some(InterlacedMbInfo {
                field_dct,
                field_refs: None,
            });
            let fields_off = intra_mb_fields(&plans_off, false, use_dc_vlc, dquant, interlaced);
            let chosen = plans_on
                .and_then(|on| {
                    let on: [BlockPlan; 6] = on
                        .try_into()
                        .unwrap_or_else(|_| unreachable!("six blocks per macroblock"));
                    let fields_on = intra_mb_fields(&on, true, use_dc_vlc, dquant, interlaced);
                    (pw.cost_of(&fields_on) < pw.cost_of(&fields_off)).then_some(fields_on)
                })
                .unwrap_or(fields_off);
            pw.push(&chosen);
        }
    }
    let bytes = pw.finish();

    // Closed decode loop: run the crate's own decoder walk over the
    // freshly emitted unit and blit the reconstruction.
    let recon = decode_own_i_vop(vol, &bytes, mb_width, mb_height);
    (bytes, recon)
}

/// Decode an emitted I-VOP unit through the crate's decoder walk into
/// a [`DecodedFrame`].
fn decode_own_i_vop(
    vol: &crate::vol::VolHeader,
    unit: &[u8],
    mb_width: usize,
    mb_height: usize,
) -> DecodedFrame {
    use crate::compat::DecodeOptions;
    use crate::vop::{parse_vop_header_body, VopContext};
    let mut br = BitReader::new(unit);
    let sc = br
        .read_bits(32)
        .expect("unit starts with the VOP start code");
    assert_eq!(sc, VOP_START_CODE, "encoder emitted a malformed unit");
    let vop = parse_vop_header_body(
        &mut br,
        vol.time_increment_resolution,
        VopContext::from_vol(vol),
    )
    .expect("own VOP header must parse");
    let mbs = if vol.data_partitioned {
        crate::vop_decode::decode_i_vop_macroblocks_dp(&mut br, vol, &vop, DecodeOptions::spec())
    } else {
        decode_i_vop_macroblocks(&mut br, vol, &vop, DecodeOptions::spec())
    }
    .expect("own I-VOP payload must decode");
    let mut frame = DecodedFrame::new(mb_width * 16, mb_height * 16, VopCodingType::I)
        .expect("frame dimensions are valid");
    for (idx, mb) in mbs.iter().enumerate() {
        frame
            .blit_macroblock(idx % mb_width, idx / mb_width, mb)
            .expect("grid-shaped blit");
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vol::parse_video_object_layer;

    #[test]
    fn configuration_headers_parse_back() {
        let cfg = EncoderConfig {
            width: 64,
            height: 48,
            ..EncoderConfig::default()
        };
        let bytes = write_configuration_headers(&cfg);
        // Find the VOL start code and parse the layer back.
        let pos = bytes
            .windows(4)
            .position(|w| w == [0, 0, 1, 0x20])
            .expect("VOL start code present");
        let vol = parse_video_object_layer(&bytes[pos..], cfg.profile_and_level()).unwrap();
        assert_eq!(vol.width, 64);
        assert_eq!(vol.height, 48);
        assert_eq!(vol.time_increment_resolution, 25);
        assert!(!vol.quant_type);
        assert!(vol.resync_marker_disable);
        assert!(!vol.data_partitioned);
        assert!(vol.obmc_disable);
    }

    #[test]
    fn method1_headers_flag_quant_type() {
        let cfg = EncoderConfig {
            width: 32,
            height: 32,
            quant_type: true,
            ..EncoderConfig::default()
        };
        let bytes = write_configuration_headers(&cfg);
        let pos = bytes
            .windows(4)
            .position(|w| w == [0, 0, 1, 0x20])
            .expect("VOL start code present");
        let vol = parse_video_object_layer(&bytes[pos..], cfg.profile_and_level()).unwrap();
        assert!(vol.quant_type);
        assert!(vol.intra_quant_mat.is_none(), "default matrices");
        assert_eq!(cfg.profile_and_level(), 0xF3);
    }

    #[test]
    fn scan_positions_invert_inverse_scan() {
        for scan in [
            ScanType::Zigzag,
            ScanType::AlternateHorizontal,
            ScanType::AlternateVertical,
        ] {
            let mut qfs = [0i32; 64];
            for (n, c) in qfs.iter_mut().enumerate() {
                *c = (n as i32) * 3 - 50;
            }
            let pqf = inverse_scan(&qfs, scan);
            assert_eq!(forward_scan(&pqf, scan), qfs);
        }
    }

    #[test]
    fn qfs_events_terminate_with_last() {
        let mut qfs = [0i32; 64];
        qfs[1] = 5;
        qfs[4] = -2;
        let events = qfs_to_events(&qfs, 1);
        assert_eq!(events.len(), 2);
        assert_eq!(
            (events[0].run, events[0].level, events[0].last),
            (0, 5, false)
        );
        assert_eq!(
            (events[1].run, events[1].level, events[1].last),
            (2, -2, true)
        );
        assert!(qfs_to_events(&[0i32; 64], 1).is_empty());
    }
}

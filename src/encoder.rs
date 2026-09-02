//! Registry-facing encoder: [`Mpeg4VideoEncoder`] implements
//! [`oxideav_core::Encoder`] over the crate's VOP encoders, and
//! [`make_encoder`] is the direct factory endpoint (the dual-API
//! sibling of [`crate::decoder::make_decoder`]).
//!
//! The current tool set is the round-438 encoder arc: rectangular
//! progressive **I- and P-VOPs** (method-1 or method-2 quantisation,
//! cost-decided AC prediction, §7.6 motion estimation with half-pel
//! refinement, `not_coded` skips, `gop-size`-driven keyframe cadence).
//! Every emitted VOP is reconstructed through the crate's own decoder
//! walk before the packet is surfaced, so the encoder's reference
//! state can never drift from a conformant decoder's.
//!
//! Timing: the output elementary stream uses the VOL's §6.3.5 time
//! model with `vop_time_increment_resolution` taken from the
//! caller's `frame_rate` numerator (default 25/1); each frame
//! advances the clock by the frame-rate denominator in ticks.
//! Packets carry `pts`/`dts` in that tick time base.

use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, TimeBase,
};

use crate::bvop_encode::encode_b_vop;
use crate::framestore::FrameStore;
use crate::ivop_encode::{encode_i_vop, write_configuration_headers, EncoderConfig, FrameView};
use crate::pvop_encode::{encode_p_vop, reconstruct_own_p_vop_with_anchor_motion};
use crate::vol::{parse_video_object_layer, VolHeader};

/// Typed options struct for the registry / options-bag construction
/// path ([`oxideav_core::CodecParameters::options`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mpeg4EncoderOptions {
    /// `qp` — the VOP quantiser scale (1..=31). Default 4.
    pub qp: u32,
    /// `mpeg-quant` — select §7.4.4.1 method-1 (matrix) quantisation
    /// instead of the default §7.4.4.2 method 2.
    pub mpeg_quant: bool,
    /// `ac-pred` — enable the cost-decided §7.4.3.3 AC-prediction
    /// emission (default on).
    pub ac_pred: bool,
    /// `four-mv` — enable cost-decided §6.3.7 four-motion-vector
    /// (`inter4v`) P-VOP macroblocks (default off).
    pub four_mv: bool,
    /// `qpel` — emit quarter-sample motion (§7.6.2.2; sets the VOL's
    /// `quarter_sample` flag and the ASP profile). Default off.
    pub qpel: bool,
    /// `bf` — number of B-VOPs between anchors (0..=8; selects the
    /// ASP profile when non-zero). Default 0.
    pub bf: u32,
    /// `bitrate` — target/peak rate in bits per second. Non-zero
    /// enables the Annex D VBV-regulated quantiser adaptation
    /// (`crate::rate_control`; `qp` then only seeds the controller)
    /// and signals `vbv_parameters` in the VOL. Default 0 (constant
    /// quantiser).
    pub bitrate: u32,
    /// `vbv-buffer` — VBV buffer size in 16384-bit units (Annex D
    /// item 2). 0 (the default) sizes the buffer automatically to two
    /// seconds at `bitrate`.
    pub vbv_buffer: u32,
    /// `gop-size` — I-VOP cadence: one keyframe every `gop_size`
    /// frames (`1` = intra-only). Default 12.
    pub gop_size: u32,
    /// `fcode` — `vop_fcode_forward` / `vop_fcode_backward` (1..=7):
    /// the Table 7-9 motion-vector range and the motion search
    /// window (±`16 << (fcode - 1)` pels in half-sample mode,
    /// ±`8 << (fcode - 1)` in quarter-sample mode). Default 1.
    pub fcode: u32,
    /// `mb-aq` — per-macroblock adaptive quantisation: activity-classed
    /// `dquant` / `dbquant` steps in a ±2 band around the VOP
    /// quantiser (`crate::mb_quant`). Default off.
    pub mb_aq: bool,
    /// `packet-bits` — target video-packet size in bits (0 = no
    /// resync markers). Non-zero clears the VOL's
    /// `resync_marker_disable` and cuts §6.2.5 video packets.
    pub packet_bits: u32,
    /// `data-partitioned` — §6.2.5.3 data partitioning of I-/P-VOPs.
    pub data_partitioned: bool,
    /// `rvlc` — reversible VLCs for the partitioned texture (requires
    /// `data-partitioned`).
    pub rvlc: bool,
    /// `gmc` — global motion compensation: non-keyframe anchors become
    /// S(GMC)-VOPs (one §7.8.4 warping point, half-pel accuracy; per-MB
    /// `mcsel` GMC-vs-local decision). Selects the ASP profile;
    /// incompatible with `data-partitioned`.
    pub gmc: bool,
    /// `gmc-points` — `no_of_sprite_warping_points` (1..=3): the
    /// global-motion model the S(GMC)-VOP trajectory carries (1 =
    /// translation, 2 = similarity, 3 = affine). Default 1.
    pub gmc_points: u32,
    /// `interlaced` — code an interlaced VOL (§6.3.3): per-macroblock
    /// field DCT (`dct_type`), §7.7.2.1 field-predicted P macroblocks
    /// and §7.7.2.2 field / interlaced-direct B macroblocks, all
    /// cost-decided. Selects the ASP profile; incompatible with
    /// `data-partitioned` and `gmc`.
    pub interlaced: bool,
    /// `top-field-first` — the §6.3.5 `top_field_first` flag written
    /// on every VOP of an interlaced VOL. Default true.
    pub top_field_first: bool,
    /// `alt-scan` — set the §6.3.5 `alternate_vertical_scan_flag` on
    /// every VOP of an interlaced VOL (all blocks use the Figure 7-4
    /// (b) scan). Default off.
    pub alt_scan: bool,
    /// `ecosystem-compat` — keep the emitted syntax inside the subset
    /// the deployed decoder ecosystem reads exactly as this crate
    /// does (see `crate::compat`): an interlaced B macroblock whose
    /// co-located future macroblock is field-predicted is never coded
    /// in direct mode (the §7.7.2.2 interlaced-direct derivation is
    /// the one clause where the ecosystem's reading diverges from the
    /// printed text). Default off (the spec-literal tool set).
    pub ecosystem_compat: bool,
    /// `short-header` — emit the §6.2.5.2 H.263-compatible short
    /// header syntax (`short_video_header == 1`) instead of the
    /// VOS/VOL/VOP stream: no configuration headers, one of the Table
    /// 6-29 picture sizes (128×96, 176×144, 352×288, 704×576,
    /// 1408×1152), the Table 6-28 fixed tool set (every other tool
    /// option must stay at its default; `qp`, `mb-aq`, `gop-size`,
    /// `bitrate` / `vbv-buffer` remain available). Default off.
    pub short_header: bool,
    /// `gob-headers` — short header only: emit a GOB header
    /// (`gob_resync_marker`, `gob_number`, `gob_frame_id`,
    /// `quant_scale`) on every GOB after the first. Default true.
    pub gob_headers: bool,
}

impl Default for Mpeg4EncoderOptions {
    fn default() -> Self {
        Self {
            qp: 4,
            mpeg_quant: false,
            ac_pred: true,
            four_mv: false,
            qpel: false,
            bf: 0,
            bitrate: 0,
            vbv_buffer: 0,
            gop_size: 12,
            fcode: 1,
            mb_aq: false,
            packet_bits: 0,
            data_partitioned: false,
            rvlc: false,
            gmc: false,
            gmc_points: 1,
            interlaced: false,
            top_field_first: true,
            alt_scan: false,
            ecosystem_compat: false,
            short_header: false,
            gob_headers: true,
        }
    }
}

impl oxideav_core::CodecOptionsStruct for Mpeg4EncoderOptions {
    const SCHEMA: &'static [oxideav_core::OptionField] = &[
        oxideav_core::OptionField {
            name: "qp",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(4),
            help: "VOP quantiser scale (1..=31); lower = higher quality/rate",
        },
        oxideav_core::OptionField {
            name: "mpeg-quant",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "use ISO/IEC 14496-2 §7.4.4.1 method-1 (matrix) quantisation \
                   instead of method 2",
        },
        oxideav_core::OptionField {
            name: "ac-pred",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(true),
            help: "enable cost-decided §7.4.3.3 AC-prediction emission on intra \
                   macroblocks",
        },
        oxideav_core::OptionField {
            name: "four-mv",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "enable cost-decided four-motion-vector (inter4v) P-VOP \
                   macroblocks",
        },
        oxideav_core::OptionField {
            name: "qpel",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "emit quarter-sample motion vectors (ISO/IEC 14496-2 \
                   §7.6.2.2; selects the ASP profile)",
        },
        oxideav_core::OptionField {
            name: "bf",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(0),
            help: "number of B-VOPs between anchors (0..=8; non-zero \
                   selects the ASP profile and adds one anchor of \
                   encoder latency)",
        },
        oxideav_core::OptionField {
            name: "bitrate",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(0),
            help: "target/peak rate in bits per second; non-zero enables the \
                   ISO/IEC 14496-2 Annex D VBV-regulated quantiser adaptation \
                   (0 = constant qp)",
        },
        oxideav_core::OptionField {
            name: "vbv-buffer",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(0),
            help: "VBV buffer size in 16384-bit units (0 = automatic: two \
                   seconds at the target bitrate)",
        },
        oxideav_core::OptionField {
            name: "gop-size",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(12),
            help: "keyframe cadence: one I-VOP every gop-size frames (1 = intra-only)",
        },
        oxideav_core::OptionField {
            name: "fcode",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(1),
            help: "vop_fcode_forward/backward (1..=7): ISO/IEC 14496-2 Table 7-9 \
                   motion-vector range and search window (±16<<(fcode-1) pels)",
        },
        oxideav_core::OptionField {
            name: "mb-aq",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "per-macroblock adaptive quantisation: activity-classed dquant / \
                   dbquant steps (±2) around the VOP quantiser",
        },
        oxideav_core::OptionField {
            name: "packet-bits",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(0),
            help: "target video-packet size in bits (ISO/IEC 14496-2 §6.2.5 \
                   resync markers + video_packet_header); 0 = one packet per VOP",
        },
        oxideav_core::OptionField {
            name: "data-partitioned",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "§6.2.5.3 data partitioning of I-/P-VOPs (dc_marker / motion_marker)",
        },
        oxideav_core::OptionField {
            name: "rvlc",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "reversible VLCs (Table B.23) for the data-partitioned texture \
                   partition; requires data-partitioned",
        },
        oxideav_core::OptionField {
            name: "gmc",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "global motion compensation: S(GMC)-VOP anchors with one \
                   warping point (ISO/IEC 14496-2 §7.8; ASP profile); \
                   incompatible with data-partitioned",
        },
        oxideav_core::OptionField {
            name: "gmc-points",
            kind: oxideav_core::OptionKind::U32,
            default: oxideav_core::OptionValue::U32(1),
            help: "no_of_sprite_warping_points of a gmc stream (1..=3): translation, \
                   similarity or affine global motion (ISO/IEC 14496-2 §7.8.4)",
        },
        oxideav_core::OptionField {
            name: "interlaced",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "code an interlaced VOL: field DCT, ISO/IEC 14496-2 §7.7.2 field \
                   motion prediction (P and B); ASP profile; incompatible with \
                   data-partitioned and gmc",
        },
        oxideav_core::OptionField {
            name: "top-field-first",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(true),
            help: "§6.3.5 top_field_first flag of every interlaced VOP",
        },
        oxideav_core::OptionField {
            name: "alt-scan",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "§6.3.5 alternate_vertical_scan_flag on every interlaced VOP \
                   (alternate-vertical coefficient scan for every block)",
        },
        oxideav_core::OptionField {
            name: "ecosystem-compat",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "avoid the syntax whose deployed-decoder reading diverges from \
                   ISO/IEC 14496-2 (interlaced direct mode over a field-predicted \
                   co-located macroblock)",
        },
        oxideav_core::OptionField {
            name: "short-header",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "emit the ISO/IEC 14496-2 §6.2.5.2 short header (H.263-compatible) \
                   syntax: no VOS/VOL, sub-QCIF/QCIF/CIF/4CIF/16CIF sizes only, \
                   I/P pictures with the Table 6-28 fixed tool set",
        },
        oxideav_core::OptionField {
            name: "gob-headers",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(true),
            help: "short header only: emit a GOB header (gob_resync_marker) on every \
                   GOB after the first",
        },
    ];

    fn apply(&mut self, key: &str, value: &oxideav_core::OptionValue) -> Result<()> {
        match key {
            "qp" => {
                let qp = value.as_u32()?;
                if !(1..=31).contains(&qp) {
                    return Err(Error::invalid("qp must be in 1..=31"));
                }
                self.qp = qp;
            }
            "mpeg-quant" => self.mpeg_quant = value.as_bool()?,
            "ac-pred" => self.ac_pred = value.as_bool()?,
            "four-mv" => self.four_mv = value.as_bool()?,
            "qpel" => self.qpel = value.as_bool()?,
            "bf" => {
                let bf = value.as_u32()?;
                if bf > 8 {
                    return Err(Error::invalid("bf must be in 0..=8"));
                }
                self.bf = bf;
            }
            "bitrate" => self.bitrate = value.as_u32()?,
            "vbv-buffer" => {
                let v = value.as_u32()?;
                if v >= (1 << 18) {
                    return Err(Error::invalid("vbv-buffer must fit the 18-bit field"));
                }
                self.vbv_buffer = v;
            }
            "gop-size" => {
                let g = value.as_u32()?;
                if g == 0 {
                    return Err(Error::invalid("gop-size must be >= 1"));
                }
                self.gop_size = g;
            }
            "fcode" => {
                let f = value.as_u32()?;
                if !(1..=7).contains(&f) {
                    return Err(Error::invalid("fcode must be in 1..=7"));
                }
                self.fcode = f;
            }
            "mb-aq" => self.mb_aq = value.as_bool()?,
            "packet-bits" => self.packet_bits = value.as_u32()?,
            "data-partitioned" => self.data_partitioned = value.as_bool()?,
            "rvlc" => self.rvlc = value.as_bool()?,
            "gmc" => self.gmc = value.as_bool()?,
            "gmc-points" => {
                let n = value.as_u32()?;
                if !(1..=3).contains(&n) {
                    return Err(Error::invalid("gmc-points must be in 1..=3"));
                }
                self.gmc_points = n;
            }
            "interlaced" => self.interlaced = value.as_bool()?,
            "top-field-first" => self.top_field_first = value.as_bool()?,
            "alt-scan" => self.alt_scan = value.as_bool()?,
            "ecosystem-compat" => self.ecosystem_compat = value.as_bool()?,
            "short-header" => self.short_header = value.as_bool()?,
            "gob-headers" => self.gob_headers = value.as_bool()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// One buffered input picture awaiting its bracketing future anchor
/// (the B-VOP reorder queue).
#[derive(Debug)]
struct PendingFrame {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    /// §6.3.5 composed VOP time in ticks.
    ticks: u64,
    /// Caller-supplied presentation stamp.
    pts: Option<i64>,
}

/// The registry-facing MPEG-4 Part 2 Visual encoder.
#[derive(Debug)]
pub struct Mpeg4VideoEncoder {
    codec_id: CodecId,
    output_params: CodecParameters,
    cfg: EncoderConfig,
    /// The parsed form of the emitted VOL (`None` for a short-header
    /// stream, which carries no VOL).
    vol: Option<VolHeader>,
    options: Mpeg4EncoderOptions,
    /// Short header: the `temporal_reference` of the next picture and
    /// its per-frame increment.
    short_tr: u8,
    short_tr_step: u8,
    /// Configuration-header run, prepended to the first packet.
    config_headers: Vec<u8>,
    /// Ticks the clock advances per input frame (the frame-rate
    /// denominator against a `time_increment_resolution` numerator).
    ticks_per_frame: u64,
    /// Running tick clock of the next frame to encode.
    next_ticks: u64,
    /// §6.3.5 seconds accumulated at the most recently coded anchor's
    /// sync point (mirrors the decoder's `sync_sec`).
    sync_sec: u64,
    /// The sync value one anchor earlier — the base a B-VOP's
    /// `modulo_time_base` counts from.
    b_base_sec: u64,
    /// Absolute tick time of the past (forward) anchor.
    prev_anchor_ticks: Option<u64>,
    /// Absolute tick time of the most recent (backward) anchor.
    last_anchor_ticks: Option<u64>,
    /// Display-order input frame counter (keyframe cadence).
    frames_seen: u64,
    /// Coded (bitstream-order) packet counter.
    frames_coded: u64,
    /// The B-VOP reorder queue (display order).
    pending_bs: Vec<PendingFrame>,
    /// The future anchor's per-macroblock decoded motion — the
    /// §7.6.9.5.1 / §6.2.6 co-located source for the B-VOPs it
    /// brackets (`None` after an intra anchor); field-predicted
    /// anchors keep their §7.7.2.2 field shape.
    anchor_motion: Option<Vec<crate::vop_decode::AnchorMbMotion>>,
    /// The Annex D VBV-regulated quantiser controller (`Some` when a
    /// bitrate target is set).
    rc: Option<crate::rate_control::RateController>,
    /// §7.6.1 anchor chain — the closed-loop references produced by
    /// decoding our own emitted units.
    store: FrameStore,
    ready: VecDeque<Packet>,
    flushed: bool,
}

impl Mpeg4VideoEncoder {
    /// Construct from codec parameters (see [`make_encoder`]).
    pub fn from_params(params: &CodecParameters) -> Result<Self> {
        let options: Mpeg4EncoderOptions = oxideav_core::parse_options(&params.options)?;
        if options.rvlc && !options.data_partitioned {
            return Err(Error::invalid("rvlc requires data-partitioned"));
        }
        if options.gmc && options.data_partitioned {
            return Err(Error::invalid(
                "gmc S-VOPs use the combined syntax (no data-partitioned)",
            ));
        }
        if options.interlaced && (options.data_partitioned || options.gmc) {
            return Err(Error::invalid(
                "interlaced VOLs use the combined syntax without GMC",
            ));
        }
        let width = params
            .width
            .ok_or_else(|| Error::invalid("encoder needs width"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("encoder needs height"))?;
        if width == 0 || height == 0 || width > 8191 || height > 8191 {
            return Err(Error::invalid("dimensions out of the 13-bit VOL range"));
        }
        if options.short_header {
            if crate::short_header::SourceFormat::from_dimensions(width, height).is_none() {
                return Err(Error::invalid(
                    "short-header pictures must be 128x96, 176x144, 352x288, 704x576 \
                     or 1408x1152 (ISO/IEC 14496-2 Table 6-29)",
                ));
            }
            let fixed_tools_only = !options.mpeg_quant
                && !options.four_mv
                && !options.qpel
                && options.bf == 0
                && options.fcode == 1
                && options.packet_bits == 0
                && !options.data_partitioned
                && !options.rvlc
                && !options.gmc
                && !options.interlaced;
            if !fixed_tools_only {
                return Err(Error::invalid(
                    "short-header uses the Table 6-28 fixed tool set (no mpeg-quant, \
                     four-mv, qpel, bf, fcode > 1, packets, data-partitioned, rvlc, gmc \
                     or interlaced)",
                ));
            }
        }
        if let Some(pf) = params.pixel_format {
            if pf != PixelFormat::Yuv420P {
                return Err(Error::unsupported(
                    "mpeg4video encoder accepts Yuv420P input only",
                ));
            }
        }
        let frame_rate = params.frame_rate.unwrap_or(Rational::new(25, 1));
        if frame_rate.num <= 0 || frame_rate.den <= 0 || frame_rate.num > 65_535 {
            return Err(Error::invalid(
                "frame rate must be positive with a numerator <= 65535 \
                 (it becomes vop_time_increment_resolution)",
            ));
        }
        // Annex D VBV signalling + controller (rate control active
        // when a bitrate target is given).
        let vbv = if options.bitrate > 0 {
            let buffer_units = if options.vbv_buffer > 0 {
                options.vbv_buffer
            } else {
                // Automatic: two seconds at the target rate.
                ((u64::from(options.bitrate) * 2).div_ceil(16384) as u32).clamp(1, (1 << 18) - 1)
            };
            Some(crate::ivop_encode::VbvSignalling {
                bit_rate_400: (u64::from(options.bitrate).div_ceil(400) as u32)
                    .clamp(1, (1 << 30) - 1),
                buffer_units,
                // The Annex D default operating point: vbv_occupancy =
                // 170 × vbv_buffer_size (≈ two-thirds of the buffer).
                occupancy_64: (170u32.saturating_mul(buffer_units)).min((1 << 26) - 1),
            })
        } else {
            None
        };
        let cfg = EncoderConfig {
            width: width as u16,
            height: height as u16,
            time_increment_resolution: frame_rate.num as u16,
            quant_type: options.mpeg_quant,
            ac_prediction: options.ac_pred,
            four_mv: options.four_mv,
            quarter_sample: options.qpel,
            b_vops: options.bf > 0,
            vbv,
            fcode: options.fcode as u8,
            adaptive_quant: options.mb_aq,
            resilience: crate::packet_encode::ResilienceConfig {
                packet_bits: options.packet_bits,
                data_partitioned: options.data_partitioned,
                reversible_vlc: options.rvlc,
            },
            gmc: options.gmc,
            gmc_points: options.gmc_points as u8,
            interlaced: options.interlaced,
            top_field_first: options.top_field_first,
            alternate_scan: options.alt_scan,
            short_header: options.short_header,
            gob_headers: options.gob_headers,
        };
        let (config_headers, vol) = if cfg.short_header {
            // §6.2.5.2: no configuration headers at all.
            (Vec::new(), None)
        } else {
            let headers = write_configuration_headers(&cfg);
            let vol_pos = headers
                .windows(4)
                .position(|w| w == [0, 0, 1, 0x20])
                .expect("emitted headers contain the VOL start code");
            let vol = parse_video_object_layer(&headers[vol_pos..], cfg.profile_and_level())
                .map_err(|e| Error::invalid(format!("emitted VOL failed to re-parse: {e}")))?;
            (headers, Some(vol))
        };

        let mut output_params = CodecParameters::video(params.codec_id.clone());
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(PixelFormat::Yuv420P);
        output_params.frame_rate = Some(frame_rate);
        output_params.extradata = config_headers.clone();
        output_params.tag = Some(oxideav_core::CodecTag::fourcc(if cfg.short_header {
            b"H263"
        } else {
            b"FMP4"
        }));
        // §6.3.5.2: temporal_reference counts 30000/1001 Hz ticks; one
        // input frame advances it by the nearest whole tick count.
        let tr_step = if cfg.short_header {
            let per_frame = (30000.0 / 1001.0) * frame_rate.den as f64 / frame_rate.num as f64;
            per_frame.round().clamp(1.0, 255.0) as u8
        } else {
            0
        };

        let rc = vbv.map(|v| {
            crate::rate_control::RateController::new(
                crate::rate_control::RateControlConfig {
                    bit_rate: u64::from(options.bitrate),
                    vbv_buffer_units: v.buffer_units,
                    occupancy_64: v.occupancy_64,
                    seconds_per_vop: frame_rate.den as f64 / frame_rate.num as f64,
                    initial_qp: options.qp,
                },
                config_headers.len() as u64 * 8,
            )
        });

        Ok(Self {
            codec_id: params.codec_id.clone(),
            output_params,
            cfg,
            vol,
            options,
            short_tr: 0,
            short_tr_step: tr_step,
            config_headers,
            ticks_per_frame: frame_rate.den as u64,
            next_ticks: 0,
            sync_sec: 0,
            b_base_sec: 0,
            prev_anchor_ticks: None,
            last_anchor_ticks: None,
            frames_seen: 0,
            frames_coded: 0,
            pending_bs: Vec::new(),
            anchor_motion: None,
            rc,
            store: FrameStore::new(),
            ready: VecDeque::new(),
            flushed: false,
        })
    }

    /// The packet time base: one tick of the VOL clock.
    fn time_base(&self) -> TimeBase {
        TimeBase::new(1, i64::from(self.cfg.time_increment_resolution))
    }

    /// Copy one plane out of a [`oxideav_core::VideoFrame`] into a
    /// tightly-packed buffer of `w × h` (dropping stride padding).
    fn tight_plane(
        plane: &oxideav_core::VideoPlane,
        w: usize,
        h: usize,
        label: &str,
    ) -> Result<Vec<u8>> {
        let stride = plane.stride;
        if stride < w || plane.data.len() < stride * (h - 1) + w {
            return Err(Error::invalid(format!("{label} plane too small")));
        }
        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            out.extend_from_slice(&plane.data[row * stride..row * stride + w]);
        }
        Ok(out)
    }
}

impl oxideav_core::Encoder for Mpeg4VideoEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.flushed {
            return Err(Error::invalid("send_frame after flush"));
        }
        let video = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("mpeg4video encoder accepts video frames")),
        };
        let (w, h) = (usize::from(self.cfg.width), usize::from(self.cfg.height));
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let planes = video.image_planes();
        if planes.len() != 3 {
            return Err(Error::invalid("expected 3 Yuv420P planes"));
        }
        let pending = PendingFrame {
            y: Self::tight_plane(&planes[0], w, h, "luma")?,
            cb: Self::tight_plane(&planes[1], cw, ch, "cb")?,
            cr: Self::tight_plane(&planes[2], cw, ch, "cr")?,
            ticks: self.next_ticks,
            pts: video.pts,
        };
        self.next_ticks += self.ticks_per_frame;
        let display_index = self.frames_seen;
        self.frames_seen += 1;

        // Anchor selection: the first frame, the keyframe cadence, and
        // every (bf + 1)-th frame terminate a B run.
        let keyframe_due = display_index % u64::from(self.options.gop_size) == 0;
        let no_anchor = self.store.backward().is_none();
        let is_anchor =
            no_anchor || keyframe_due || self.pending_bs.len() as u32 >= self.options.bf;
        if !is_anchor {
            self.pending_bs.push(pending);
            return Ok(());
        }
        self.encode_anchor(&pending, no_anchor || keyframe_due);
        self.drain_pending_bs();
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        match self.ready.pop_front() {
            Some(p) => Ok(p),
            None if self.flushed => Err(Error::Eof),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        // A tail of buffered pictures with no future anchor yet: code
        // the last one as the terminating anchor, the rest as the
        // B-VOPs it brackets.
        if let Some(last) = self.pending_bs.pop() {
            let keyframe_due = self.store.backward().is_none() || {
                // The anchor's display index is frames_seen - 1.
                (self.frames_seen - 1) % u64::from(self.options.gop_size) == 0
            };
            self.encode_anchor(&last, keyframe_due);
            self.drain_pending_bs();
        }
        self.flushed = true;
        Ok(())
    }
}

impl Mpeg4VideoEncoder {
    /// Bitstream-order packet emission: wrap `unit` (prepending the
    /// configuration run on the very first packet) with the Annex D
    /// item-7 timing — an anchor's decode time is the previous
    /// anchor's composition time once B-VOPs are in play (`bf > 0`);
    /// with `low_delay`-style `bf == 0` streams `dts == pts`.
    fn push_packet(
        &mut self,
        unit: Vec<u8>,
        ticks: u64,
        pts: Option<i64>,
        keyframe: bool,
        dts: i64,
    ) {
        let mut data = Vec::new();
        if self.frames_coded == 0 {
            data.extend_from_slice(&self.config_headers);
        }
        data.extend_from_slice(&unit);
        let pts = pts.unwrap_or(ticks as i64);
        let packet = Packet::new(0, self.time_base(), data)
            .with_pts(pts)
            .with_dts(dts)
            .with_duration(self.ticks_per_frame as i64)
            .with_keyframe(keyframe);
        self.ready.push_back(packet);
        self.frames_coded += 1;
    }

    /// The quantiser for the next VOP: the VBV controller's when rate
    /// control is active, else the constant `qp` option.
    fn current_qp(&self) -> u32 {
        self.rc
            .as_ref()
            .map(|rc| rc.qp())
            .unwrap_or(self.options.qp)
    }

    /// Annex D item-9 admission for a freshly encoded VOP of
    /// `unit_bytes` (plus the configuration run on the first packet):
    /// `true` accepts the unit (occupancy committed); `false` asks the
    /// caller to re-encode at the controller's coarsened quantiser.
    fn rc_admit(&mut self, unit_bytes: usize) -> bool {
        let extra = if self.frames_coded == 0 {
            self.config_headers.len()
        } else {
            0
        };
        let d_bits = (unit_bytes + extra) as u64 * 8;
        match &mut self.rc {
            None => true,
            Some(rc) => {
                if rc.accepts(d_bits) || !rc.escalate() {
                    rc.commit(d_bits);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Encode one anchor VOP (I when `force_i`, else P), advance the
    /// §7.6.1 chain + §6.3.5 anchor time bases, and emit its packet.
    fn encode_anchor(&mut self, frame: &PendingFrame, force_i: bool) {
        let (w, h) = (usize::from(self.cfg.width), usize::from(self.cfg.height));
        let view = FrameView {
            y: &frame.y,
            cb: &frame.cb,
            cr: &frame.cr,
            width: w,
            height: h,
        };
        let res = u64::from(self.cfg.time_increment_resolution);
        let seconds = frame.ticks / res;
        let modulo = (seconds - self.sync_sec) as u32;
        let increment = (frame.ticks % res) as u16;

        if self.cfg.short_header {
            // §6.2.5.2: I or P picture at the running temporal_reference.
            let reference = if force_i {
                None
            } else {
                self.store.backward().cloned()
            };
            let tr = self.short_tr;
            let (unit, recon) = loop {
                let qp = self.current_qp();
                let (unit, recon, _stats) = crate::short_header_encode::encode_short_header_picture(
                    &self.cfg,
                    &view,
                    reference.as_ref(),
                    tr,
                    qp,
                );
                if self.rc_admit(unit.len()) {
                    break (unit, recon);
                }
            };
            self.short_tr = self.short_tr.wrapping_add(self.short_tr_step);
            self.store.push_anchor(recon);
            self.anchor_motion = None;
            self.prev_anchor_ticks = self.last_anchor_ticks;
            self.last_anchor_ticks = Some(frame.ticks);
            let dts = frame.pts.unwrap_or(frame.ticks as i64);
            self.push_packet(unit, frame.ticks, frame.pts, force_i, dts);
            return;
        }
        let vol = self.vol.expect("long-header streams carry a VOL");

        let unit = if force_i {
            let (unit, recon) = loop {
                let qp = self.current_qp();
                let produced = encode_i_vop(&vol, &self.cfg, &view, modulo, increment, qp);
                if self.rc_admit(produced.0.len()) {
                    break produced;
                }
            };
            self.store.push_anchor(recon);
            self.anchor_motion = None;
            unit
        } else if self.cfg.gmc {
            // GMC: non-keyframe anchors are S(GMC)-VOPs.
            let reference = self
                .store
                .backward()
                .expect("anchor present on the S path")
                .clone();
            let unit = loop {
                let qp = self.current_qp();
                let (unit, _stats) = crate::svop_encode::encode_s_vop(
                    &vol, &self.cfg, &view, &reference, modulo, increment, qp,
                );
                if self.rc_admit(unit.len()) {
                    break unit;
                }
            };
            let (_recon, motion) =
                crate::svop_encode::reconstruct_own_s_vop_with_motion(&vol, &unit, &mut self.store);
            self.anchor_motion = Some(
                motion
                    .into_iter()
                    .map(crate::vop_decode::AnchorMbMotion::Frame)
                    .collect(),
            );
            unit
        } else {
            let reference = self
                .store
                .backward()
                .expect("anchor present on the P path")
                .clone();
            let unit = loop {
                let qp = self.current_qp();
                let (unit, _stats) =
                    encode_p_vop(&vol, &self.cfg, &view, &reference, modulo, increment, qp);
                if self.rc_admit(unit.len()) {
                    break unit;
                }
            };
            let (_recon, motion) =
                reconstruct_own_p_vop_with_anchor_motion(&vol, &unit, &mut self.store);
            self.anchor_motion = Some(motion);
            unit
        };

        // §6.3.5 anchor time bookkeeping (mirrors the decoder).
        self.b_base_sec = self.sync_sec;
        self.sync_sec = seconds;
        self.prev_anchor_ticks = self.last_anchor_ticks;
        self.last_anchor_ticks = Some(frame.ticks);

        // Annex D item 7: with B-VOPs in play the anchor's decode time
        // is the *previous* anchor's composition time (t0 = τ0 − Δ for
        // the first anchor); without B-VOPs, dts == pts.
        let dts = if self.options.bf == 0 {
            frame.pts.unwrap_or(frame.ticks as i64)
        } else {
            match self.prev_anchor_ticks {
                Some(prev) => prev as i64,
                None => frame.ticks as i64 - self.ticks_per_frame as i64,
            }
        };
        self.push_packet(unit, frame.ticks, frame.pts, force_i, dts);
    }

    /// Encode every buffered B-VOP (display order) between the two
    /// anchors now in the chain and emit their packets.
    fn drain_pending_bs(&mut self) {
        if self.pending_bs.is_empty() {
            return;
        }
        let (prev, last) = (
            self.prev_anchor_ticks.expect("B-VOPs need a past anchor"),
            self.last_anchor_ticks.expect("B-VOPs need a future anchor"),
        );
        let trd = (last - prev) as i32;
        let res = u64::from(self.cfg.time_increment_resolution);
        let (w, h) = (usize::from(self.cfg.width), usize::from(self.cfg.height));
        let pending = std::mem::take(&mut self.pending_bs);
        let vol = self.vol.expect("B-VOPs need a VOL");
        // Progressive B-VOPs consume the §7.6.9.5.1 co-located shape.
        let progressive: Option<Vec<crate::pvop_mv::PvopMbMotion>> = self
            .anchor_motion
            .as_deref()
            .map(|m| m.iter().map(|a| a.progressive()).collect());
        for b in &pending {
            let view = FrameView {
                y: &b.y,
                cb: &b.cb,
                cr: &b.cr,
                width: w,
                height: h,
            };
            let trb = (b.ticks - prev) as i32;
            let seconds = b.ticks / res;
            let modulo = (seconds - self.b_base_sec) as u32;
            let increment = (b.ticks % res) as u16;
            let unit = loop {
                let qp = self.current_qp();
                let unit = if self.cfg.interlaced {
                    let (unit, _recon, _stats) =
                        crate::bvop_interlaced_encode::encode_b_vop_interlaced(
                            &vol,
                            &self.cfg,
                            &view,
                            &self.store,
                            self.anchor_motion.as_deref(),
                            trb,
                            trd,
                            modulo,
                            increment,
                            qp,
                            self.options.ecosystem_compat,
                        );
                    unit
                } else {
                    let (unit, _recon, _stats) = encode_b_vop(
                        &vol,
                        &self.cfg,
                        &view,
                        &self.store,
                        progressive.as_deref(),
                        trb,
                        trd,
                        modulo,
                        increment,
                        qp,
                    );
                    unit
                };
                if self.rc_admit(unit.len()) {
                    break unit;
                }
            };
            // Annex D item 7: a B-VOP's decode time is its own
            // composition time.
            self.push_packet(unit, b.ticks, b.pts, false, b.ticks as i64);
        }
    }
}

/// Direct factory endpoint: build a boxed registry-compatible encoder
/// from [`oxideav_core::CodecParameters`] (`width` / `height`
/// required; `frame_rate` optional, default 25/1; options per
/// [`Mpeg4EncoderOptions`]).
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn oxideav_core::Encoder>> {
    Ok(Box::new(Mpeg4VideoEncoder::from_params(params)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{Encoder as _, VideoFrame, VideoPlane};

    fn gray_frame(w: usize, h: usize, luma: u8) -> Frame {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        Frame::Video(VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: w,
                    data: vec![luma; w * h],
                },
                VideoPlane {
                    stride: cw,
                    data: vec![128; cw * ch],
                },
                VideoPlane {
                    stride: cw,
                    data: vec![128; cw * ch],
                },
            ],
        })
    }

    fn base_params() -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new("mpeg4video"));
        p.width = Some(48);
        p.height = Some(32);
        p.pixel_format = Some(PixelFormat::Yuv420P);
        p
    }

    #[test]
    fn encodes_decodable_ip_packets_with_extradata() {
        let mut enc = Mpeg4VideoEncoder::from_params(&base_params()).unwrap();
        assert!(!enc.output_params().extradata.is_empty());
        for k in 0..3 {
            enc.send_frame(&gray_frame(48, 32, 100 + k * 30)).unwrap();
        }
        enc.flush().unwrap();
        let mut stream = Vec::new();
        let mut keyframes = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(p) => {
                    keyframes.push(p.flags.keyframe);
                    stream.extend_from_slice(&p.data);
                }
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected {e}"),
            }
        }
        // gop-size 12 → frame 0 is the only keyframe of the three.
        assert_eq!(keyframes, vec![true, false, false]);
        let mut dec = crate::decoder::Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        assert_eq!(frames.len(), 3);
        // Flat gray reconstructs exactly at low qp, including through
        // the two P-VOPs' luminance step changes.
        assert!(frames[0].luma_samples()[..48].iter().all(|&s| s == 100));
        assert!(frames[2].luma_samples()[..48].iter().all(|&s| s == 160));
    }

    #[test]
    fn gop_size_one_is_intra_only() {
        let mut p = base_params();
        p.options = oxideav_core::CodecOptions::default().set("gop-size", "1");
        let mut enc = Mpeg4VideoEncoder::from_params(&p).unwrap();
        for _ in 0..3 {
            enc.send_frame(&gray_frame(48, 32, 90)).unwrap();
        }
        enc.flush().unwrap();
        let mut count = 0;
        loop {
            match enc.receive_packet() {
                Ok(pk) => {
                    assert!(pk.flags.keyframe, "every packet must be an I-VOP");
                    count += 1;
                }
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected {e}"),
            }
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn rejects_bad_options_and_dimensions() {
        let mut p = base_params();
        p.width = None;
        assert!(Mpeg4VideoEncoder::from_params(&p).is_err());
        let mut p = base_params();
        p.options = oxideav_core::CodecOptions::default().set("qp", "0");
        assert!(Mpeg4VideoEncoder::from_params(&p).is_err());
        let mut p = base_params();
        p.options = oxideav_core::CodecOptions::default().set("qp", "31");
        assert!(Mpeg4VideoEncoder::from_params(&p).is_ok());
    }

    #[test]
    fn mpeg_quant_option_flips_the_vol() {
        let mut p = base_params();
        p.options = oxideav_core::CodecOptions::default().set("mpeg-quant", "true");
        let enc = Mpeg4VideoEncoder::from_params(&p).unwrap();
        assert!(enc.vol.as_ref().unwrap().quant_type);
        assert_eq!(enc.cfg.profile_and_level(), 0xF3);
    }
}

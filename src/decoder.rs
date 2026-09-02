//! Top-level MPEG-4 Visual elementary-stream decoder.
//!
//! [`Mpeg4VideoDecoder`] is the single entry point that turns a raw
//! ISO/IEC 14496-2 video elementary stream (the byte stream of §6.2.1
//! `VisualObjectSequence()` — start-code-delimited configuration
//! headers followed by VOPs) into display-order decoded frames. It
//! stitches together every layer this crate already implements:
//!
//! 1. **Start-code scan** (§6.2.1 / Table 6-3): the stream is split at
//!    each byte-aligned `00 00 01 xx` boundary and each unit is
//!    dispatched on its start-code value.
//! 2. **Configuration headers** (§6.2): `visual_object_sequence`
//!    (profile/level), `visual_object`, `video_object`,
//!    `video_object_layer` — parsed by [`crate::vol`].
//! 3. **Frame headers** (§6.2.4 / §6.2.5): `group_of_vop` (time-code
//!    sync reset) and `video_object_plane` — parsed by [`crate::vop`],
//!    leaving the bit reader at the macroblock layer.
//! 4. **Macroblock walks** (§6.2.6): the [`crate::vop_decode`]
//!    bitstream drivers for I- / P- / S(GMC)- / B-VOPs.
//! 5. **Frame assembly + reorder** (§7.6.1 / §6.1.3.8): the
//!    [`crate::sequence::SequenceDecoder`] owns the reference-frame
//!    chain and emits frames in display order.
//!
//! ## The §6.3.5 time model
//!
//! `modulo_time_base` seconds accumulate against the synchronisation
//! point of the previously decoded I-/P-/S(GMC)-VOP (or the last GOV
//! `time_code`); a B-VOP's `modulo_time_base` is instead relative to
//! the previous anchor **in display order** — the *past* anchor, whose
//! sync value the decoder keeps one slot behind the running one. From
//! the composed VOP times the §7.6.7 `TRB` / `TRD` temporal references
//! of each B-VOP fall out as plain tick differences.
//!
//! ## Scope
//!
//! Rectangular, progressive, non-scalable streams on the half-sample
//! prediction path — the [`crate::vop_decode`] gates apply. An uncoded
//! VOP (`vop_coded == 0`) reconstructs as a copy of the forward
//! reference per §6.3.5.

use crate::bitreader::BitReader;
use crate::block::InterMacroblock;
use crate::bvop_prediction::BVopSampleMode;
use crate::compat::DecodeOptions;
use crate::frame_decode::{FrameDecodeError, PVopMbContent, SGmcMbContent};
use crate::framestore::DecodedFrame;
use crate::pvop_mv::PvopMbMotion;
use crate::sequence::SequenceDecoder;
use crate::vol::{
    parse_video_object_layer, parse_visual_object_header, parse_visual_object_sequence_header,
    VolHeader, VolParseError, VIDEO_OBJECT_LAYER_START_CODE_MAX, VIDEO_OBJECT_LAYER_START_CODE_MIN,
    VIDEO_OBJECT_START_CODE_MAX, VIDEO_OBJECT_START_CODE_MIN, VISUAL_OBJECT_SEQUENCE_END_CODE,
    VISUAL_OBJECT_SEQUENCE_START_CODE, VISUAL_OBJECT_START_CODE,
};
use crate::vop::{
    parse_group_of_vop_header, parse_vop_header_body, VopCodingType, VopContext, VopHeader,
    VopParseError, GROUP_OF_VOP_START_CODE, VOP_START_CODE,
};
use crate::vop_decode::{
    decode_b_vop_interlaced_macroblocks, decode_b_vop_macroblocks, decode_i_vop_macroblocks,
    decode_p_vop_macroblocks, decode_s_gmc_vop_macroblocks, vop_mb_dimensions, AnchorMbMotion,
    VopDecodeError,
};

/// §6.2.1 `user_data_start_code`.
const USER_DATA_START_CODE: u32 = 0x0000_01B2;

/// Errors raised by the elementary-stream decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDecodeError {
    /// A configuration header failed to parse.
    Vol(VolParseError),
    /// A GOV / VOP header failed to parse.
    Vop(VopParseError),
    /// A VOP's macroblock layer failed to decode.
    VopDecode(VopDecodeError),
    /// Frame assembly / the reference chain failed.
    Frame(FrameDecodeError),
    /// A VOP arrived before any `video_object_layer` header.
    MissingVol,
    /// A §6.2.5.2 short-header picture failed to parse or decode.
    ShortHeader(crate::short_header::ShortHeaderError),
    /// A P-/S(GMC)-/B-VOP (or an uncoded VOP) arrived without the
    /// anchor frame(s) it references.
    MissingAnchor,
    /// A B-VOP's §7.6.7 temporal references were inconsistent
    /// (`TRD <= 0` or `TRB` outside `(0, TRD)`), meaning the bracketing
    /// anchor times do not straddle the B-VOP time.
    BadTemporalReference {
        /// Computed `TRB` (B-VOP minus past anchor, ticks).
        trb: i64,
        /// Computed `TRD` (future anchor minus past anchor, ticks).
        trd: i64,
    },
}

impl core::fmt::Display for StreamDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StreamDecodeError::Vol(e) => write!(f, "stream decode: configuration header: {e}"),
            StreamDecodeError::Vop(e) => write!(f, "stream decode: VOP header: {e}"),
            StreamDecodeError::VopDecode(e) => write!(f, "stream decode: macroblock layer: {e}"),
            StreamDecodeError::Frame(e) => write!(f, "stream decode: frame assembly: {e:?}"),
            StreamDecodeError::ShortHeader(e) => write!(f, "stream decode: {e}"),
            StreamDecodeError::MissingVol => {
                write!(f, "stream decode: VOP before any video_object_layer header")
            }
            StreamDecodeError::MissingAnchor => {
                write!(f, "stream decode: VOP references a missing anchor frame")
            }
            StreamDecodeError::BadTemporalReference { trb, trd } => write!(
                f,
                "stream decode: inconsistent B-VOP temporal references (TRB {trb}, TRD {trd})"
            ),
        }
    }
}

impl std::error::Error for StreamDecodeError {}

impl From<crate::short_header::ShortHeaderError> for StreamDecodeError {
    fn from(e: crate::short_header::ShortHeaderError) -> Self {
        StreamDecodeError::ShortHeader(e)
    }
}

impl From<VolParseError> for StreamDecodeError {
    fn from(e: VolParseError) -> Self {
        StreamDecodeError::Vol(e)
    }
}

impl From<VopParseError> for StreamDecodeError {
    fn from(e: VopParseError) -> Self {
        StreamDecodeError::Vop(e)
    }
}

impl From<VopDecodeError> for StreamDecodeError {
    fn from(e: VopDecodeError) -> Self {
        StreamDecodeError::VopDecode(e)
    }
}

impl From<FrameDecodeError> for StreamDecodeError {
    fn from(e: FrameDecodeError) -> Self {
        StreamDecodeError::Frame(e)
    }
}

/// The stateful top-level decoder for one MPEG-4 Visual elementary
/// stream (one video object layer).
///
/// Feed whole start-code-delimited byte buffers to
/// [`Mpeg4VideoDecoder::decode`]; each call returns the frames that
/// became displayable, in display order. Call
/// [`Mpeg4VideoDecoder::flush`] after the last buffer to release the
/// final held anchor (§6.1.3.8 one-slot reorder delay).
#[derive(Debug, Default)]
pub struct Mpeg4VideoDecoder {
    /// The [`crate::compat`] behaviour selection (default: literal
    /// spec).
    options: DecodeOptions,
    /// `profile_and_level_indication` from the VOS header (0 = unseen).
    profile_level: u8,
    /// The active VOL configuration.
    vol: Option<VolHeader>,
    /// §7.6.1 reference chain + §6.1.3.8 reorder buffer.
    sequence: SequenceDecoder,
    /// The most recent anchor's per-macroblock motion (raster order) —
    /// the §7.6.9.5.1 / §7.7.2.2 co-located source for B-VOP direct
    /// modes (field-predicted anchors keep their field pairs). `None`
    /// after an intra-only anchor.
    anchor_motion: Option<Vec<AnchorMbMotion>>,
    /// §6.3.5: seconds accumulated at the sync point of the most
    /// recently decoded anchor.
    sync_sec: u64,
    /// The sync value one anchor earlier — the base a B-VOP's
    /// `modulo_time_base` counts from (previous anchor in display
    /// order).
    b_base_sec: u64,
    /// Absolute time (ticks) of the past (forward) anchor.
    prev_anchor_ticks: Option<i64>,
    /// Absolute time (ticks) of the most recent (backward) anchor.
    last_anchor_ticks: Option<i64>,
    /// §6.2.5.2 short-header mode, entered when the first unit of a
    /// VOL-less stream is a `video_plane_with_short_header()`: the
    /// running 30000/1001 Hz tick clock and the previous
    /// `temporal_reference` (modulo-256 arithmetic, §6.3.5.2).
    short_header: Option<ShortHeaderClock>,
}

/// The short-header picture clock (see
/// [`Mpeg4VideoDecoder::decode_with_pts`]).
#[derive(Debug, Clone, Copy, Default)]
struct ShortHeaderClock {
    /// Ticks of the most recent picture (30000/1001 Hz units, starting
    /// at the first picture's `temporal_reference`).
    ticks: i64,
    /// `temporal_reference` of the most recent picture.
    last_tr: Option<u8>,
}

impl Mpeg4VideoDecoder {
    /// A fresh decoder with no configuration and an empty chain,
    /// decoding with the literal-spec behaviour
    /// ([`DecodeOptions::spec`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh decoder with an explicit [`crate::compat`] behaviour
    /// selection — pass [`DecodeOptions::ecosystem`] to match the
    /// black-box-observed ecosystem behaviour on the two documented
    /// spec divergences bit-for-bit.
    pub fn with_options(options: DecodeOptions) -> Self {
        Self {
            options,
            ..Self::default()
        }
    }

    /// The [`crate::compat`] behaviour selection this decoder was
    /// built with.
    pub fn options(&self) -> DecodeOptions {
        self.options
    }

    /// The active VOL header, once one has been decoded.
    pub fn vol(&self) -> Option<&VolHeader> {
        self.vol.as_ref()
    }

    /// The `profile_and_level_indication` from the VOS header (0 when
    /// the stream carried none so far).
    pub fn profile_level(&self) -> u8 {
        self.profile_level
    }

    /// Decode every start-code-delimited unit in `data`, returning the
    /// frames that became displayable, in display order.
    ///
    /// `data` must start at (or before) a `00 00 01` start-code prefix;
    /// bytes ahead of the first start code are skipped. Units are
    /// processed in stream order; configuration headers update the
    /// decoder state, VOPs decode to frames.
    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<DecodedFrame>, StreamDecodeError> {
        self.decode_with_pts(data, None)
    }

    /// [`Mpeg4VideoDecoder::decode`] with a container presentation
    /// timestamp: every frame produced by a VOP in `data` gets
    /// `container_pts` as its [`DecodedFrame::pts`] (a container
    /// packet's pts is a display-order stamp for the VOP it carries,
    /// so it survives the §6.1.3.8 reorder attached to the right
    /// frame). Frames *released* by this call but produced by an
    /// earlier packet keep that packet's stamp. Every frame also
    /// carries the §6.3.5 composed tick time
    /// ([`DecodedFrame::pts_ticks`], units of
    /// `1 / vop_time_increment_resolution`).
    pub fn decode_with_pts(
        &mut self,
        data: &[u8],
        container_pts: Option<i64>,
    ) -> Result<Vec<DecodedFrame>, StreamDecodeError> {
        let mut out = Vec::new();
        // §6.2.1 / §6.2.5.2: a stream whose first unit is a
        // `short_video_start_marker` (no VOL ever seen) is a
        // short-header stream; every subsequent picture follows the
        // abbreviated syntax.
        if self.vol.is_none() && self.short_header.is_none() {
            let first_sc = scan_start_codes(data).first().copied();
            let first_sh = crate::short_header::scan_short_header_pictures(data)
                .first()
                .copied();
            if let Some(sh) = first_sh {
                if first_sc.map_or(true, |sc| sh < sc) {
                    self.short_header = Some(ShortHeaderClock::default());
                }
            }
        }
        if self.short_header.is_some() {
            let starts = crate::short_header::scan_short_header_pictures(data);
            for (idx, &start) in starts.iter().enumerate() {
                let end = starts.get(idx + 1).copied().unwrap_or(data.len());
                self.decode_short_header_unit(&data[start..end], container_pts, &mut out)?;
            }
            return Ok(out);
        }
        let starts = scan_start_codes(data);
        for (idx, &start) in starts.iter().enumerate() {
            let end = starts.get(idx + 1).copied().unwrap_or(data.len());
            let unit = &data[start..end];
            self.decode_unit(unit, container_pts, &mut out)?;
        }
        Ok(out)
    }

    /// Release the final held anchor at end-of-stream (§6.1.3.8).
    pub fn flush(&mut self) -> Vec<DecodedFrame> {
        self.sequence.flush()
    }

    /// Decode one `video_plane_with_short_header()` picture
    /// (§6.2.5.2): I pictures enter the chain as intra anchors, P
    /// pictures predict from the previous picture with the Table 6-28
    /// fixed tools (half-sample, `vop_rounding_type == 0`). The frame's
    /// tick time counts `temporal_reference` at 30000/1001 Hz from the
    /// first picture.
    fn decode_short_header_unit(
        &mut self,
        unit: &[u8],
        container_pts: Option<i64>,
        out: &mut Vec<DecodedFrame>,
    ) -> Result<(), StreamDecodeError> {
        let mut br = BitReader::new(unit);
        let pic = crate::short_header::parse_short_header_picture(&mut br)?;
        let entries = crate::short_header::decode_short_header_macroblocks(&mut br, &pic)?;
        let (mb_width, mb_height) = pic.source_format.mb_dimensions();
        match pic.coding_type {
            VopCodingType::I => {
                let mbs = crate::short_header::intra_macroblocks(&entries)
                    .expect("an I picture decodes to intra macroblocks only");
                out.extend(self.sequence.push_i_vop(mb_width, mb_height, &mbs)?);
            }
            _ => {
                if self.sequence.store().p_vop_reference().is_none() {
                    return Err(StreamDecodeError::MissingAnchor);
                }
                out.extend(self.sequence.push_p_vop(
                    mb_width,
                    mb_height,
                    &entries,
                    0,
                    BVopSampleMode::HalfPel,
                    8,
                )?);
            }
        }
        self.anchor_motion = None;
        // §6.3.5.2 temporal_reference: modulo-256 increments at
        // 30000/1001 Hz.
        let clock = self
            .short_header
            .get_or_insert_with(ShortHeaderClock::default);
        let ticks = match clock.last_tr {
            None => i64::from(pic.temporal_reference),
            Some(prev) => clock.ticks + i64::from(pic.temporal_reference.wrapping_sub(prev)),
        };
        clock.ticks = ticks;
        clock.last_tr = Some(pic.temporal_reference);
        if let Some(pending) = self.sequence.pending_anchor_mut() {
            pending.set_pts(container_pts);
            pending.set_pts_ticks(Some(ticks));
        }
        Ok(())
    }

    /// Dispatch one start-code-delimited unit.
    fn decode_unit(
        &mut self,
        unit: &[u8],
        container_pts: Option<i64>,
        out: &mut Vec<DecodedFrame>,
    ) -> Result<(), StreamDecodeError> {
        let code = u32::from_be_bytes([unit[0], unit[1], unit[2], unit[3]]);
        match code {
            VISUAL_OBJECT_SEQUENCE_START_CODE => {
                self.profile_level = parse_visual_object_sequence_header(unit)?;
            }
            VISUAL_OBJECT_SEQUENCE_END_CODE => {}
            USER_DATA_START_CODE => {}
            VISUAL_OBJECT_START_CODE => {
                let _ = parse_visual_object_header(unit)?;
            }
            VIDEO_OBJECT_START_CODE_MIN..=VIDEO_OBJECT_START_CODE_MAX => {
                // A bare video_object_start_code carries only its id.
            }
            VIDEO_OBJECT_LAYER_START_CODE_MIN..=VIDEO_OBJECT_LAYER_START_CODE_MAX => {
                self.vol = Some(parse_video_object_layer(unit, self.profile_level)?);
            }
            GROUP_OF_VOP_START_CODE => {
                let gov = parse_group_of_vop_header(unit)?;
                // §6.3.4: the time_code marks a new synchronisation
                // point for the following VOPs.
                let secs = u64::from(gov.time_code.hours) * 3600
                    + u64::from(gov.time_code.minutes) * 60
                    + u64::from(gov.time_code.seconds);
                self.sync_sec = secs;
                self.b_base_sec = secs;
            }
            VOP_START_CODE => {
                self.decode_vop_unit(unit, container_pts, out)?;
            }
            _ => {
                // Unknown / reserved start code: skip. (Stuffing and
                // extension units do not affect the layers we decode.)
            }
        }
        Ok(())
    }

    /// Decode one `video_object_plane` unit end-to-end.
    fn decode_vop_unit(
        &mut self,
        unit: &[u8],
        container_pts: Option<i64>,
        out: &mut Vec<DecodedFrame>,
    ) -> Result<(), StreamDecodeError> {
        let vol = self.vol.ok_or(StreamDecodeError::MissingVol)?;
        let (mb_width, mb_height) = vop_mb_dimensions(&vol);
        let mb_count = mb_width * mb_height;

        let mut br = BitReader::new(unit);
        br.skip_bits(32)
            .map_err(|_| StreamDecodeError::Vop(VopParseError::Truncated))?;
        let vop = parse_vop_header_body(
            &mut br,
            vol.time_increment_resolution,
            VopContext::from_vol(&vol),
        )?;

        let res = i64::from(vol.time_increment_resolution.max(1));
        let is_anchor = !matches!(vop.coding_type, VopCodingType::B);
        // §6.3.3 `quarter_sample`: selects the §7.6.2.1 vs §7.6.2.2
        // luminance sub-pel grid for every inter path of this VOL.
        let sample_mode = sample_mode_of(&vol);

        // §6.3.5 composed display time of *this* VOP, in ticks of
        // 1 / vop_time_increment_resolution: an anchor counts its
        // modulo_time_base from the previous anchor's sync point, a
        // B-VOP from the past anchor in display order.
        let base_sec = if is_anchor {
            self.sync_sec
        } else {
            self.b_base_sec
        };
        let vop_ticks = (base_sec + u64::from(vop.modulo_time_base)) as i64 * res
            + i64::from(vop.time_increment);

        if !vop.coded {
            return self.decode_uncoded_vop(
                &vop,
                mb_count,
                mb_width,
                mb_height,
                res,
                (container_pts, vop_ticks),
                out,
            );
        }

        match vop.coding_type {
            VopCodingType::I => {
                // §6.2.5.3: a data_partitioned VOL rearranges the I-/P-
                // VOP macroblock layer into marker-separated partitions.
                let mbs = if vol.data_partitioned {
                    crate::vop_decode::decode_i_vop_macroblocks_dp(
                        &mut br,
                        &vol,
                        &vop,
                        self.options,
                    )?
                } else {
                    decode_i_vop_macroblocks(&mut br, &vol, &vop, self.options)?
                };
                out.extend(self.sequence.push_i_vop(mb_width, mb_height, &mbs)?);
                self.anchor_motion = None;
            }
            VopCodingType::P => {
                if self.sequence.store().p_vop_reference().is_none() {
                    return Err(StreamDecodeError::MissingAnchor);
                }
                let entries = if vol.data_partitioned {
                    crate::vop_decode::decode_p_vop_macroblocks_dp(
                        &mut br,
                        &vol,
                        &vop,
                        self.options,
                    )?
                } else {
                    decode_p_vop_macroblocks(&mut br, &vol, &vop, self.options)?
                };
                self.anchor_motion = Some(motion_of_p_entries(&entries));
                out.extend(if vol.obmc_disable {
                    self.sequence.push_p_vop(
                        mb_width,
                        mb_height,
                        &entries,
                        vop.rounding_type,
                        sample_mode,
                        u32::from(vol.bits_per_pixel),
                    )?
                } else {
                    // §7.6.6: overlapped motion compensation for the
                    // luminance plane (half-sample mode only — the
                    // walk gates the qpel+OBMC combination).
                    self.sequence.push_p_vop_obmc(
                        mb_width,
                        mb_height,
                        &entries,
                        vop.rounding_type,
                        u32::from(vol.bits_per_pixel),
                    )?
                });
            }
            VopCodingType::S => {
                if self.sequence.store().p_vop_reference().is_none() {
                    return Err(StreamDecodeError::MissingAnchor);
                }
                let (entries, geometry) =
                    decode_s_gmc_vop_macroblocks(&mut br, &vol, &vop, self.options)?;
                self.anchor_motion = Some(motion_of_s_entries(&entries));
                out.extend(self.sequence.push_s_gmc_vop(
                    mb_width,
                    mb_height,
                    &entries,
                    &geometry,
                    vop.rounding_type,
                    sample_mode,
                    u32::from(vol.bits_per_pixel),
                )?);
            }
            VopCodingType::B => {
                let (trb, trd) = self.b_vop_temporal_refs(&vop, res)?;
                if vol.interlaced {
                    let entries = decode_b_vop_interlaced_macroblocks(
                        &mut br,
                        &vol,
                        &vop,
                        trb,
                        trd,
                        self.anchor_motion.as_deref(),
                        self.options,
                    )?;
                    let field_mode = if vol.quarter_sample {
                        crate::bvop_field_motion::FieldSampleMode::QuarterSample {
                            bits_per_pixel: u32::from(vol.bits_per_pixel),
                        }
                    } else {
                        crate::bvop_field_motion::FieldSampleMode::HalfSample
                    };
                    out.extend(self.sequence.push_b_vop_interlaced(
                        mb_width,
                        mb_height,
                        &entries,
                        vop.rounding_type,
                        sample_mode,
                        field_mode,
                        u32::from(vol.bits_per_pixel),
                    )?);
                } else {
                    let progressive: Vec<PvopMbMotion> = self
                        .anchor_motion
                        .as_deref()
                        .map(|m| m.iter().map(|a| a.progressive()).collect())
                        .unwrap_or_default();
                    let entries = decode_b_vop_macroblocks(
                        &mut br,
                        &vol,
                        &vop,
                        trb,
                        trd,
                        self.anchor_motion.as_deref().map(|_| &progressive[..]),
                        self.options,
                    )?;
                    out.extend(self.sequence.push_b_vop(
                        mb_width,
                        mb_height,
                        &entries,
                        vop.rounding_type,
                        sample_mode,
                        u32::from(vol.bits_per_pixel),
                    )?);
                }
            }
        }

        // Stamp the frame this VOP produced: a B-VOP was appended to
        // `out` (displayed immediately); an anchor is the held
        // §6.1.3.8 pending frame.
        if is_anchor {
            if let Some(pending) = self.sequence.pending_anchor_mut() {
                pending.set_pts(container_pts);
                pending.set_pts_ticks(Some(vop_ticks));
            }
            self.advance_anchor_time(&vop, res);
        } else if let Some(frame) = out.last_mut() {
            frame.set_pts(container_pts);
            frame.set_pts_ticks(Some(vop_ticks));
        }
        Ok(())
    }

    /// §6.3.5 `vop_coded == 0`: the reconstructed VOP is a copy of the
    /// forward reference (per §7.6.7). An uncoded anchor still enters
    /// the reference chain and advances the time base; an uncoded
    /// B-VOP displays the copy immediately.
    #[allow(clippy::too_many_arguments)]
    fn decode_uncoded_vop(
        &mut self,
        vop: &VopHeader,
        mb_count: usize,
        mb_width: usize,
        mb_height: usize,
        res: i64,
        (container_pts, vop_ticks): (Option<i64>, i64),
        out: &mut Vec<DecodedFrame>,
    ) -> Result<(), StreamDecodeError> {
        if self.sequence.store().p_vop_reference().is_none() {
            return Err(StreamDecodeError::MissingAnchor);
        }
        let vol = self.vol.as_ref().expect("vol checked by caller");
        let bpp = u32::from(vol.bits_per_pixel);
        match vop.coding_type {
            VopCodingType::B => {
                // Copy of the forward reference, displayed immediately:
                // a forward-only zero-MV, zero-residual B-VOP.
                let entries: Vec<_> = (0..mb_count).map(|_| forward_copy_b_mb()).collect();
                out.extend(self.sequence.push_b_vop(
                    mb_width,
                    mb_height,
                    &entries,
                    0,
                    BVopSampleMode::HalfPel,
                    bpp,
                )?);
                if let Some(frame) = out.last_mut() {
                    frame.set_pts(container_pts);
                    frame.set_pts_ticks(Some(vop_ticks));
                }
            }
            _ => {
                // Anchor: an all-skipped P assembly copies the forward
                // reference and installs the copy as the new anchor.
                let entries: Vec<_> = (0..mb_count)
                    .map(|_| PVopMbContent::Inter {
                        motion: PvopMbMotion::Skipped,
                        residual: InterMacroblock::zero(),
                    })
                    .collect();
                self.anchor_motion =
                    Some(vec![AnchorMbMotion::Frame(PvopMbMotion::Skipped); mb_count]);
                // All-skipped MBs are integer zero-MV copies, so the
                // sub-pel mode is immaterial; pass the VOL's anyway.
                let sample_mode = sample_mode_of(vol);
                out.extend(self.sequence.push_p_vop(
                    mb_width,
                    mb_height,
                    &entries,
                    0,
                    sample_mode,
                    bpp,
                )?);
                if let Some(pending) = self.sequence.pending_anchor_mut() {
                    pending.set_pts(container_pts);
                    pending.set_pts_ticks(Some(vop_ticks));
                }
                self.advance_anchor_time(vop, res);
            }
        }
        Ok(())
    }

    /// Advance the §6.3.5 anchor time base with a just-decoded anchor
    /// VOP's `modulo_time_base` + `vop_time_increment`.
    fn advance_anchor_time(&mut self, vop: &VopHeader, res: i64) {
        // The B-VOP base becomes the sync point *this* anchor counted
        // from (the previous anchor's sync).
        self.b_base_sec = self.sync_sec;
        self.sync_sec += u64::from(vop.modulo_time_base);
        let ticks = self.sync_sec as i64 * res + i64::from(vop.time_increment);
        self.prev_anchor_ticks = self.last_anchor_ticks;
        self.last_anchor_ticks = Some(ticks);
    }

    /// Compute the §7.6.7 `TRB` / `TRD` for a B-VOP from the composed
    /// VOP times.
    fn b_vop_temporal_refs(
        &self,
        vop: &VopHeader,
        res: i64,
    ) -> Result<(i32, i32), StreamDecodeError> {
        let (Some(prev), Some(last)) = (self.prev_anchor_ticks, self.last_anchor_ticks) else {
            return Err(StreamDecodeError::MissingAnchor);
        };
        // §6.3.5: a B-VOP's modulo_time_base counts from the previous
        // anchor in *display* order — the past anchor's sync point.
        let b_ticks = (self.b_base_sec + u64::from(vop.modulo_time_base)) as i64 * res
            + i64::from(vop.time_increment);
        let trd = last - prev;
        let trb = b_ticks - prev;
        if trd <= 0 || trb <= 0 || trb >= trd {
            return Err(StreamDecodeError::BadTemporalReference { trb, trd });
        }
        Ok((trb as i32, trd as i32))
    }
}

/// The §6.3.3 `quarter_sample` → sub-pel-grid selection for one VOL.
fn sample_mode_of(vol: &VolHeader) -> BVopSampleMode {
    if vol.quarter_sample {
        BVopSampleMode::QuarterPel {
            bits_per_pixel: u32::from(vol.bits_per_pixel),
        }
    } else {
        BVopSampleMode::HalfPel
    }
}

/// Extract the per-macroblock motion of a decoded P-VOP (co-located
/// source for the next B-VOPs).
fn motion_of_p_entries(entries: &[PVopMbContent]) -> Vec<AnchorMbMotion> {
    entries
        .iter()
        .map(|e| match e {
            PVopMbContent::Inter { motion, .. } => AnchorMbMotion::Frame(*motion),
            // A field-predicted anchor macroblock keeps its field MV
            // pair + reference selections — the §7.7.2.2 interlaced
            // direct mode consumes them; progressive consumers collapse
            // via AnchorMbMotion::progressive().
            PVopMbContent::FieldInter {
                mvs,
                top_field_ref,
                bottom_field_ref,
                ..
            } => AnchorMbMotion::Field {
                mvs: *mvs,
                top_ref: *top_field_ref,
                bottom_ref: *bottom_field_ref,
            },
            PVopMbContent::Intra(_) => AnchorMbMotion::Frame(PvopMbMotion::Intra),
        })
        .collect()
}

/// Extract the per-macroblock motion of a decoded S(GMC)-VOP.
///
/// * A **skipped** GMC macroblock applies the §7.6.9.6 substitution:
///   "If the co-located macroblock in the most recently decoded
///   S(GMC)-VOP is skipped, this co-located macroblock is treated as a
///   non-skipped macroblock with the averaged motion vector defined in
///   subclause 7.8.7.3 for the current B-macroblock" — surfaced as a
///   1-MV macroblock carrying the AMV, so a direct-mode B macroblock
///   scales the AMV instead of taking the skipped-co-located forward
///   zero-MV path.
/// * A **coded** GMC macroblock and an intra macroblock carry no local
///   vector — they resolve to the §7.6.9.5.1 zero-vector fallback
///   (represented as `Intra` for the co-located adapter).
fn motion_of_s_entries(entries: &[SGmcMbContent]) -> Vec<AnchorMbMotion> {
    entries
        .iter()
        .map(|e| match e {
            SGmcMbContent::Local { motion, .. } => AnchorMbMotion::Frame(*motion),
            SGmcMbContent::Gmc {
                amv,
                not_coded: true,
                ..
            } => AnchorMbMotion::Frame(PvopMbMotion::OneMv(*amv)),
            SGmcMbContent::Gmc {
                not_coded: false, ..
            }
            | SGmcMbContent::Intra(_) => AnchorMbMotion::Frame(PvopMbMotion::Intra),
        })
        .collect()
}

/// A forward-only zero-MV zero-residual B-VOP macroblock — the §6.3.5
/// `vop_coded == 0` copy of the forward reference.
fn forward_copy_b_mb() -> crate::bvop_mv::BVopMbTexturedDecode {
    use crate::bvop::BVopMbType;
    use crate::bvop_mv::{BVopMbDecode, BVopMbTexturedDecode};
    use crate::bvop_prediction::{BVopMvPair, BVopPredictionMode};
    use crate::motion::MotionVector;
    let zero = MotionVector { x: 0, y: 0 };
    BVopMbTexturedDecode {
        motion: BVopMbDecode {
            mb_type: BVopMbType::Forward,
            prediction_mode: BVopPredictionMode::ForwardOnly,
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
        quantiser_scale: 1,
    }
}

// ─────────────────── runtime-registry integration ───────────────────

/// The registry-facing packet decoder: wraps [`Mpeg4VideoDecoder`]
/// behind the `oxideav_core` [`Decoder`](oxideav_core::Decoder)
/// contract (send-packet / receive-frame with §6.1.3.8 display-order
/// output).
///
/// Construction goes through [`make_decoder`] (the crate's direct
/// factory endpoint) or the runtime registry after
/// [`crate::register`]. `CodecParameters::extradata` — the container's
/// decoder-configuration blob, which for MPEG-4 Visual is the
/// VOS/VO/VOL header run — is decoded once up front so the first data
/// packet can already reference a configured VOL.
pub struct Mpeg4PacketDecoder {
    codec_id: oxideav_core::CodecId,
    inner: Mpeg4VideoDecoder,
    /// Display-order frames not yet handed out.
    ready: std::collections::VecDeque<DecodedFrame>,
    /// Retained so [`Decoder::reset`](oxideav_core::Decoder::reset)
    /// can re-prime a fresh inner decoder after a seek.
    extradata: Vec<u8>,
    /// `max_pixels_per_frame` DoS cap from `CodecParameters::limits`.
    max_pixels: u64,
    flushed: bool,
}

impl std::fmt::Debug for Mpeg4PacketDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mpeg4PacketDecoder")
            .field("codec_id", &self.codec_id.as_str())
            .field("ready", &self.ready.len())
            .field("flushed", &self.flushed)
            .finish()
    }
}

/// Map a [`StreamDecodeError`] onto the framework error surface.
fn to_core_error(e: StreamDecodeError) -> oxideav_core::Error {
    oxideav_core::Error::invalid(e.to_string())
}

impl Mpeg4PacketDecoder {
    /// Enforce the `max_pixels_per_frame` cap once a VOL is known.
    fn check_limits(&self) -> oxideav_core::Result<()> {
        if let Some(vol) = self.inner.vol() {
            let pixels = u64::from(vol.width) * u64::from(vol.height);
            if pixels > self.max_pixels {
                return Err(oxideav_core::Error::resource_exhausted(format!(
                    "mpeg4video: {}x{} exceeds max_pixels_per_frame {}",
                    vol.width, vol.height, self.max_pixels
                )));
            }
        }
        Ok(())
    }

    /// Convert one decoded VOP into the framework's planar 4:2:0
    /// [`VideoFrame`](oxideav_core::VideoFrame) (Y then Cb then Cr,
    /// each plane row-major with its natural stride).
    ///
    /// The frame `pts` is the container packet pts when the demuxer
    /// supplied one, else the §6.3.5 bitstream tick time (units of
    /// `1 / vop_time_increment_resolution` — the natural stream time
    /// base for a raw MPEG-4 Visual elementary stream).
    fn frame_to_video(frame: &DecodedFrame) -> oxideav_core::VideoFrame {
        let w = frame.width();
        oxideav_core::VideoFrame {
            pts: frame.pts().or(frame.pts_ticks()),
            planes: vec![
                oxideav_core::VideoPlane {
                    stride: w,
                    data: frame.luma_samples().to_vec(),
                },
                oxideav_core::VideoPlane {
                    stride: w / 2,
                    data: frame.cb_samples().to_vec(),
                },
                oxideav_core::VideoPlane {
                    stride: w / 2,
                    data: frame.cr_samples().to_vec(),
                },
            ],
        }
    }
}

impl oxideav_core::Decoder for Mpeg4PacketDecoder {
    fn codec_id(&self) -> &oxideav_core::CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &oxideav_core::Packet) -> oxideav_core::Result<()> {
        let frames = self
            .inner
            .decode_with_pts(&packet.data, packet.pts)
            .map_err(to_core_error)?;
        self.check_limits()?;
        self.ready.extend(frames);
        Ok(())
    }

    fn receive_frame(&mut self) -> oxideav_core::Result<oxideav_core::Frame> {
        match self.ready.pop_front() {
            Some(frame) => Ok(oxideav_core::Frame::Video(Self::frame_to_video(&frame))),
            None if self.flushed => Err(oxideav_core::Error::Eof),
            None => Err(oxideav_core::Error::NeedMore),
        }
    }

    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.ready.extend(self.inner.flush());
        self.flushed = true;
        Ok(())
    }

    fn reset(&mut self) -> oxideav_core::Result<()> {
        // Preserve the compat behaviour selection across seeks.
        self.inner = Mpeg4VideoDecoder::with_options(self.inner.options());
        self.ready.clear();
        self.flushed = false;
        if !self.extradata.is_empty() {
            // Re-prime the configuration headers so the next packet
            // decodes against the same VOL.
            let extradata = std::mem::take(&mut self.extradata);
            let _ = self.inner.decode(&extradata).map_err(to_core_error)?;
            self.extradata = extradata;
        }
        Ok(())
    }
}

/// Typed options struct for the registry / options-bag construction
/// path ([`oxideav_core::CodecParameters::options`]).
///
/// One key is recognised:
///
/// * `ecosystem-compat` (bool, default `false`) — decode with the
///   ecosystem-compat behaviour on the two documented spec
///   divergences (see [`crate::compat`]): the §7.7.2.2
///   interlaced-direct derivation reads the co-located field MVs as
///   zero, and the §7.4.4.5 method-1 mismatch toggle is skipped on
///   intra blocks. The default is the literal spec text.
///
/// Typed consumers can skip the bag and pass
/// [`DecodeOptions`](crate::compat::DecodeOptions) to
/// [`Mpeg4VideoDecoder::with_options`] directly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mpeg4DecoderOptions {
    /// The [`crate::compat`] ecosystem-compat switch.
    pub ecosystem_compat: bool,
}

impl From<Mpeg4DecoderOptions> for DecodeOptions {
    fn from(o: Mpeg4DecoderOptions) -> Self {
        DecodeOptions {
            ecosystem_compat: o.ecosystem_compat,
        }
    }
}

impl oxideav_core::CodecOptionsStruct for Mpeg4DecoderOptions {
    const SCHEMA: &'static [oxideav_core::OptionField] = &[oxideav_core::OptionField {
        name: "ecosystem-compat",
        kind: oxideav_core::OptionKind::Bool,
        default: oxideav_core::OptionValue::Bool(false),
        help: "match the black-box-observed ecosystem behaviour on the two documented \
               ISO/IEC 14496-2 divergences (interlaced-direct co-located MVs read as zero; \
               §7.4.4.5 mismatch control skipped on intra blocks) instead of the literal \
               spec text",
    }];

    fn apply(&mut self, key: &str, value: &oxideav_core::OptionValue) -> oxideav_core::Result<()> {
        match key {
            "ecosystem-compat" => self.ecosystem_compat = value.as_bool()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// Direct factory endpoint: build a boxed registry-compatible decoder
/// from [`oxideav_core::CodecParameters`].
///
/// `params.extradata` (the VOS/VO/VOL configuration run, when the
/// container carries one) is decoded immediately; data packets follow
/// via [`oxideav_core::Decoder::send_packet`]. `params.options` is
/// parsed against the [`Mpeg4DecoderOptions`] schema (unknown keys and
/// malformed values are rejected with
/// [`Error::InvalidData`](oxideav_core::Error::InvalidData)).
pub fn make_decoder(
    params: &oxideav_core::CodecParameters,
) -> oxideav_core::Result<Box<dyn oxideav_core::Decoder>> {
    let options: Mpeg4DecoderOptions = oxideav_core::parse_options(&params.options)?;
    let mut dec = Mpeg4PacketDecoder {
        codec_id: params.codec_id.clone(),
        inner: Mpeg4VideoDecoder::with_options(options.into()),
        ready: std::collections::VecDeque::new(),
        extradata: params.extradata.clone(),
        max_pixels: params.limits.max_pixels_per_frame,
        flushed: false,
    };
    if !dec.extradata.is_empty() {
        let frames = dec.inner.decode(&params.extradata).map_err(to_core_error)?;
        dec.check_limits()?;
        dec.ready.extend(frames);
    }
    Ok(Box::new(dec))
}

/// Return the byte offsets of every `00 00 01 xx` start code in `data`
/// (the offset of the leading `00`).
fn scan_start_codes(data: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            out.push(i);
            // The next start code cannot overlap the 4-byte header.
            i += 4;
        } else if data[i + 2] > 1 {
            // Skip past a byte that can't begin a prefix.
            i += 3;
        } else if data[i + 1] != 0 {
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSB-first bit writer for synthesising elementary streams.
    #[derive(Default)]
    pub(crate) struct BitWriter {
        bytes: Vec<u8>,
        bit: u8,
        acc: u8,
    }

    impl BitWriter {
        pub(crate) fn write_bits(&mut self, value: u32, n: usize) {
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
        /// §5.2.3-style stuffing: pad to the byte boundary so the next
        /// start code is byte-aligned ('0' then '1's).
        pub(crate) fn align(&mut self) {
            if self.bit != 0 {
                self.write_bits(0, 1);
                while self.bit != 0 {
                    self.write_bits(1, 1);
                }
            }
        }
        pub(crate) fn finish(mut self) -> Vec<u8> {
            self.align();
            self.bytes
        }
    }

    /// Write the rectangular test VOL used across the walk tests
    /// (verid 1, method-2 quant, res 30, OBMC off).
    pub(crate) fn write_vol(w: &mut BitWriter, width: u32, height: u32) {
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
        w.write_bits(width, 13);
        w.write_bits(1, 1); // marker
        w.write_bits(height, 13);
        w.write_bits(1, 1); // marker
        w.write_bits(0, 1); // interlaced
        w.write_bits(1, 1); // obmc_disable
        w.write_bits(0, 1); // sprite_enable
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(1, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // scalability
        w.align();
    }

    /// Write one DC-only intra macroblock (see vop_decode tests).
    fn write_dc_only_intra_mb(w: &mut BitWriter) {
        w.write_bits(0b1, 1); // mcbpc: intra, cbpc 00
        w.write_bits(0, 1); // ac_pred_flag
        w.write_bits(0b0011, 4); // cbpy intra 0
        for _ in 0..4 {
            w.write_bits(0b011, 3); // luma DC size 0
        }
        for _ in 0..2 {
            w.write_bits(0b11, 2); // chroma DC size 0
        }
    }

    /// Write an I-VOP unit (quant 8, tinc ticks) whose macroblocks are
    /// all DC-only intra.
    pub(crate) fn write_i_vop(w: &mut BitWriter, mb_count: usize, tinc: u32) {
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(0b00, 2); // I
        w.write_bits(0, 1); // modulo_time_base
        w.write_bits(1, 1); // marker
        w.write_bits(tinc, 5); // time increment (res 30 → 5 bits)
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(8, 5); // vop_quant
        for _ in 0..mb_count {
            write_dc_only_intra_mb(w);
        }
        w.align();
    }

    /// Write a P-VOP unit whose macroblocks are all not_coded.
    pub(crate) fn write_skipped_p_vop(w: &mut BitWriter, mb_count: usize, tinc: u32) {
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(0b01, 2); // P
        w.write_bits(0, 1); // modulo_time_base
        w.write_bits(1, 1); // marker
        w.write_bits(tinc, 5);
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 1); // vop_rounding_type
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(8, 5); // vop_quant
        w.write_bits(1, 3); // vop_fcode_forward
        for _ in 0..mb_count {
            w.write_bits(1, 1); // not_coded
        }
        w.align();
    }

    /// Write a B-VOP unit whose macroblocks are all modb "1" (direct,
    /// zero delta, no residual).
    fn write_direct_b_vop(w: &mut BitWriter, mb_count: usize, tinc: u32) {
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(0b10, 2); // B
        w.write_bits(0, 1); // modulo_time_base
        w.write_bits(1, 1); // marker
        w.write_bits(tinc, 5);
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(8, 5); // vop_quant
        w.write_bits(1, 3); // vop_fcode_forward
        w.write_bits(1, 3); // vop_fcode_backward
        for _ in 0..mb_count {
            w.write_bits(1, 1); // modb "1"
        }
        w.align();
    }

    /// Write an uncoded P-VOP (vop_coded == 0).
    fn write_uncoded_p_vop(w: &mut BitWriter, tinc: u32) {
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(0b01, 2); // P
        w.write_bits(0, 1); // modulo_time_base
        w.write_bits(1, 1); // marker
        w.write_bits(tinc, 5);
        w.write_bits(1, 1); // marker
        w.write_bits(0, 1); // vop_coded = 0
        w.align();
    }

    pub(crate) fn write_vos(w: &mut BitWriter) {
        w.write_bits(VISUAL_OBJECT_SEQUENCE_START_CODE, 32);
        w.write_bits(0x01, 8); // profile_and_level_indication
    }

    #[test]
    fn ip_stream_decodes_in_display_order() {
        // VOS + VOL(32×16) + I(t=0) + P(t=1, all skipped) → after flush,
        // two display-order frames: the flat-128 I then its copy.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol(&mut w, 32, 16);
        write_i_vop(&mut w, 2, 0);
        write_skipped_p_vop(&mut w, 2, 1);
        let stream = w.finish();

        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].coding_type(), VopCodingType::I);
        assert_eq!(frames[1].coding_type(), VopCodingType::P);
        assert_eq!(frames[0].luma_at(0, 0), Some(128));
        assert_eq!(frames[1].luma_at(0, 0), Some(128));
        assert_eq!(frames[1].luma_at(31, 15), Some(128));
        assert_eq!(dec.profile_level(), 0x01);
    }

    #[test]
    fn ipb_stream_reorders_to_display_order() {
        // Coding order I(t=0) P(t=2) B(t=1) → display order I B P
        // (§6.1.3.8). The direct B averages the two anchors; with both
        // flat 128 the average is 128 as well, but the coding types
        // prove the reorder.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol(&mut w, 16, 16);
        write_i_vop(&mut w, 1, 0);
        write_skipped_p_vop(&mut w, 1, 2);
        write_direct_b_vop(&mut w, 1, 1);
        let stream = w.finish();

        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        let types: Vec<_> = frames.iter().map(|f| f.coding_type()).collect();
        assert_eq!(
            types,
            vec![VopCodingType::I, VopCodingType::B, VopCodingType::P]
        );
        assert_eq!(frames[1].luma_at(0, 0), Some(128));
    }

    #[test]
    fn pts_ticks_follow_display_order() {
        // Coding order I(t=0) P(t=2) B(t=1) → display order I B P with
        // §6.3.5 tick times 0, 1, 2 (res = 30 ticks/s, all inside the
        // first second).
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol(&mut w, 16, 16);
        write_i_vop(&mut w, 1, 0);
        write_skipped_p_vop(&mut w, 1, 2);
        write_direct_b_vop(&mut w, 1, 1);
        let stream = w.finish();
        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        let ticks: Vec<_> = frames.iter().map(|f| f.pts_ticks()).collect();
        assert_eq!(ticks, vec![Some(0), Some(1), Some(2)]);
        // No container timestamps were supplied.
        assert!(frames.iter().all(|f| f.pts().is_none()));
    }

    #[test]
    fn modulo_time_base_advances_pts_ticks() {
        // An anchor whose modulo_time_base crosses a second boundary
        // composes ticks = (seconds * res) + increment.
        let mut w = BitWriter::default();
        write_vol(&mut w, 16, 16);
        write_i_vop(&mut w, 1, 0);
        // P with modulo_time_base = 1 (one second) and tinc = 5.
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(0b01, 2); // P
        w.write_bits(0b10, 2); // modulo_time_base = 1
        w.write_bits(1, 1); // marker
        w.write_bits(5, 5); // tinc
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 1); // rounding
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(8, 5); // quant
        w.write_bits(1, 3); // fcode fwd
        w.write_bits(1, 1); // not_coded
        w.align();
        let stream = w.finish();
        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        assert_eq!(frames[0].pts_ticks(), Some(0)); // I at t = 0
        assert_eq!(frames[1].pts_ticks(), Some(35)); // P at 1 s + 5 ticks
    }

    #[test]
    fn container_pts_survives_reorder() {
        // Three packets in coding order carrying display-order pts
        // I=100, P=300, B=200: the display-order frames must come out
        // stamped 100, 200, 300.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol(&mut w, 16, 16);
        write_i_vop(&mut w, 1, 0);
        let unit_i = w.finish();
        let mut w = BitWriter::default();
        write_skipped_p_vop(&mut w, 1, 2);
        let unit_p = w.finish();
        let mut w = BitWriter::default();
        write_direct_b_vop(&mut w, 1, 1);
        let unit_b = w.finish();

        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = Vec::new();
        frames.extend(dec.decode_with_pts(&unit_i, Some(100)).unwrap());
        frames.extend(dec.decode_with_pts(&unit_p, Some(300)).unwrap());
        frames.extend(dec.decode_with_pts(&unit_b, Some(200)).unwrap());
        frames.extend(dec.flush());
        let pts: Vec<_> = frames.iter().map(|f| f.pts()).collect();
        assert_eq!(pts, vec![Some(100), Some(200), Some(300)]);
        let types: Vec<_> = frames.iter().map(|f| f.coding_type()).collect();
        assert_eq!(
            types,
            vec![VopCodingType::I, VopCodingType::B, VopCodingType::P]
        );
    }

    #[test]
    fn uncoded_p_vop_copies_forward_reference() {
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol(&mut w, 16, 16);
        write_i_vop(&mut w, 1, 0);
        write_uncoded_p_vop(&mut w, 1);
        let stream = w.finish();

        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        assert_eq!(frames.len(), 2);
        // The uncoded P is a copy of the I.
        assert_eq!(frames[1].luma_at(0, 0), Some(128));
    }

    /// The test VOL with `obmc_disable = 0` (all other fields as
    /// [`write_vol`]).
    fn write_obmc_vol(w: &mut BitWriter, width: u32, height: u32) {
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
        w.write_bits(width, 13);
        w.write_bits(1, 1); // marker
        w.write_bits(height, 13);
        w.write_bits(1, 1); // marker
        w.write_bits(0, 1); // interlaced
        w.write_bits(0, 1); // obmc_disable = 0 → §7.6.6 applies
        w.write_bits(0, 1); // sprite_enable
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type
        w.write_bits(1, 1); // complexity_estimation_disable
        w.write_bits(1, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        w.write_bits(0, 1); // scalability
        w.align();
    }

    #[test]
    fn obmc_vol_p_vop_decodes_via_overlapped_assembly() {
        // A VOL with obmc_disable == 0 must decode (the macroblock
        // syntax is unchanged; only the luminance prediction blends).
        // Flat-grey I + all-skipped P: every §7.6.6 fetch sees 128, so
        // the overlapped copy is exact.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_obmc_vol(&mut w, 32, 16);
        write_i_vop(&mut w, 2, 0);
        write_skipped_p_vop(&mut w, 2, 1);
        let stream = w.finish();
        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        assert_eq!(frames.len(), 2);
        assert!(!dec.vol().unwrap().obmc_disable);
        assert_eq!(frames[1].coding_type(), VopCodingType::P);
        assert_eq!(frames[1].luma_at(0, 0), Some(128));
        assert_eq!(frames[1].luma_at(31, 15), Some(128));
    }

    #[test]
    fn vop_before_vol_errors() {
        let mut w = BitWriter::default();
        write_i_vop(&mut w, 1, 0);
        let stream = w.finish();
        let mut dec = Mpeg4VideoDecoder::new();
        assert_eq!(dec.decode(&stream), Err(StreamDecodeError::MissingVol));
    }

    #[test]
    fn p_vop_without_anchor_errors() {
        let mut w = BitWriter::default();
        write_vol(&mut w, 16, 16);
        write_skipped_p_vop(&mut w, 1, 0);
        let stream = w.finish();
        let mut dec = Mpeg4VideoDecoder::new();
        assert_eq!(dec.decode(&stream), Err(StreamDecodeError::MissingAnchor));
    }

    #[test]
    fn b_vop_with_bad_times_errors() {
        // B at t=5 outside the anchor bracket [0, 2] → TRB >= TRD.
        let mut w = BitWriter::default();
        write_vol(&mut w, 16, 16);
        write_i_vop(&mut w, 1, 0);
        write_skipped_p_vop(&mut w, 1, 2);
        write_direct_b_vop(&mut w, 1, 5);
        let stream = w.finish();
        let mut dec = Mpeg4VideoDecoder::new();
        assert_eq!(
            dec.decode(&stream),
            Err(StreamDecodeError::BadTemporalReference { trb: 5, trd: 2 })
        );
    }

    #[test]
    fn modulo_time_base_advances_anchor_seconds() {
        // I at t=0; P with modulo_time_base=1 and tinc=1 → t = 31
        // ticks. A B at tinc=15 (base = I's sync, 0 s) → TRB 15,
        // TRD 31. All decode cleanly.
        let mut w = BitWriter::default();
        write_vol(&mut w, 16, 16);
        write_i_vop(&mut w, 1, 0);
        // P with modulo_time_base "10" (one second elapsed).
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(0b01, 2); // P
        w.write_bits(0b10, 2); // modulo_time_base = 1
        w.write_bits(1, 1); // marker
        w.write_bits(1, 5); // tinc = 1
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        w.write_bits(0, 1); // rounding
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(8, 5); // quant
        w.write_bits(1, 3); // fcode fwd
        w.write_bits(1, 1); // MB not_coded
        w.align();
        write_direct_b_vop(&mut w, 1, 15);
        let stream = w.finish();

        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn scan_finds_all_start_codes() {
        let data = [
            0x00, 0x00, 0x01, 0xB0, 0x01, // VOS
            0x00, 0x00, 0x01, 0xB6, 0xFF, // VOP
            0x00, 0x00, 0x01, 0xB1, // VOS end
        ];
        assert_eq!(scan_start_codes(&data), vec![0, 5, 10]);
    }
}

#[cfg(test)]
mod registry_tests {
    use super::tests::{write_i_vop, write_skipped_p_vop, write_vol, write_vos, BitWriter};
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};

    fn packet(data: Vec<u8>) -> Packet {
        Packet::new(0, TimeBase::new(1, 30), data)
    }

    #[test]
    fn registry_round_trip_decodes_display_order_frames() {
        // Register into a fresh RuntimeContext, resolve by id, then
        // feed a VOS+VOL config packet and two VOP packets.
        let mut ctx = oxideav_core::RuntimeContext::new();
        crate::register(&mut ctx);
        let params = CodecParameters::video(CodecId::new("mpeg4video"));
        let mut dec = ctx.codecs.first_decoder(&params).unwrap();

        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol(&mut w, 32, 16);
        let config = w.finish();
        dec.send_packet(&packet(config)).unwrap();
        assert!(matches!(
            dec.receive_frame(),
            Err(oxideav_core::Error::NeedMore)
        ));

        let mut w = BitWriter::default();
        write_i_vop(&mut w, 2, 0);
        dec.send_packet(&packet(w.finish())).unwrap();
        // The I-VOP is held back by the §6.1.3.8 one-slot reorder.
        assert!(matches!(
            dec.receive_frame(),
            Err(oxideav_core::Error::NeedMore)
        ));

        let mut w = BitWriter::default();
        write_skipped_p_vop(&mut w, 2, 1);
        dec.send_packet(&packet(w.finish())).unwrap();
        // The P-VOP released the I-VOP.
        let frame = dec.receive_frame().unwrap();
        let oxideav_core::Frame::Video(v) = frame else {
            panic!("expected a video frame");
        };
        assert_eq!(v.planes.len(), 3);
        // No container pts on the packet → §6.3.5 tick fallback
        // (I-VOP at t = 0).
        assert_eq!(v.pts, Some(0));
        assert_eq!(v.planes[0].stride, 32);
        assert_eq!(v.planes[0].data.len(), 32 * 16);
        assert_eq!(v.planes[0].data[0], 128); // flat mid-grey I-VOP
        assert_eq!(v.planes[1].stride, 16);
        assert_eq!(v.planes[1].data.len(), 16 * 8);

        // Flush releases the held P-VOP, then Eof.
        dec.flush().unwrap();
        let frame = dec.receive_frame().unwrap();
        let oxideav_core::Frame::Video(v) = frame else {
            panic!("expected a video frame");
        };
        assert_eq!(v.pts, Some(1)); // P-VOP tick time (tinc = 1)
        assert_eq!(v.planes[0].data[0], 128);
        assert!(matches!(dec.receive_frame(), Err(oxideav_core::Error::Eof)));
    }

    #[test]
    fn make_decoder_primes_extradata_and_resets() {
        // Configuration in extradata; data packets carry only VOPs.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol(&mut w, 16, 16);
        let extradata = w.finish();

        let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
        params.extradata = extradata;
        let mut dec = make_decoder(&params).unwrap();

        let mut w = BitWriter::default();
        write_i_vop(&mut w, 1, 0);
        let ivop = w.finish();
        dec.send_packet(&packet(ivop.clone())).unwrap();
        dec.flush().unwrap();
        assert!(dec.receive_frame().is_ok());
        assert!(matches!(dec.receive_frame(), Err(oxideav_core::Error::Eof)));

        // reset(): the VOL must survive (re-primed from extradata), so
        // a fresh I-VOP decodes without a config packet.
        dec.reset().unwrap();
        dec.send_packet(&packet(ivop)).unwrap();
        dec.flush().unwrap();
        assert!(dec.receive_frame().is_ok());
    }

    #[test]
    fn limits_cap_rejects_oversized_vol() {
        let mut w = BitWriter::default();
        write_vol(&mut w, 1280, 720);
        let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
        params.extradata = w.finish();
        params.limits.max_pixels_per_frame = 640 * 480;
        assert!(matches!(
            make_decoder(&params),
            Err(oxideav_core::Error::ResourceExhausted(_))
        ));
    }

    #[test]
    fn decoder_options_bag_parses_ecosystem_compat() {
        // Empty bag → spec defaults.
        let d: Mpeg4DecoderOptions =
            oxideav_core::parse_options(&oxideav_core::CodecOptions::new()).unwrap();
        assert!(!d.ecosystem_compat);
        assert_eq!(DecodeOptions::from(d), DecodeOptions::spec());
        // Bool synonyms coerce.
        let d: Mpeg4DecoderOptions = oxideav_core::parse_options(
            &oxideav_core::CodecOptions::new().set("ecosystem-compat", "on"),
        )
        .unwrap();
        assert!(d.ecosystem_compat);
        assert_eq!(DecodeOptions::from(d), DecodeOptions::ecosystem());
    }

    #[test]
    fn make_decoder_accepts_the_ecosystem_compat_option() {
        let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
        params.options = oxideav_core::CodecOptions::new().set("ecosystem-compat", "true");
        assert!(make_decoder(&params).is_ok());
    }

    #[test]
    fn make_decoder_rejects_unknown_or_malformed_options() {
        let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
        params.options = oxideav_core::CodecOptions::new().set("no-such-option", "1");
        assert!(matches!(
            make_decoder(&params),
            Err(oxideav_core::Error::InvalidData(_))
        ));
        let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
        params.options = oxideav_core::CodecOptions::new().set("ecosystem-compat", "maybe");
        assert!(matches!(
            make_decoder(&params),
            Err(oxideav_core::Error::InvalidData(_))
        ));
    }

    #[test]
    fn with_options_reports_and_survives_reset() {
        // The typed entry point records the selection…
        let dec = Mpeg4VideoDecoder::with_options(DecodeOptions::ecosystem());
        assert_eq!(dec.options(), DecodeOptions::ecosystem());
        assert_eq!(Mpeg4VideoDecoder::new().options(), DecodeOptions::spec());

        // …and a registry decoder built with the option keeps it across
        // reset() (the inner decoder is rebuilt with the same options).
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol(&mut w, 16, 16);
        let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
        params.extradata = w.finish();
        params.options = oxideav_core::CodecOptions::new().set("ecosystem-compat", "true");
        let mut dec = make_decoder(&params).unwrap();
        let mut w = BitWriter::default();
        write_i_vop(&mut w, 1, 0);
        let ivop = w.finish();
        dec.send_packet(&packet(ivop.clone())).unwrap();
        dec.reset().unwrap();
        dec.send_packet(&packet(ivop)).unwrap();
        dec.flush().unwrap();
        assert!(dec.receive_frame().is_ok());
    }
    /// End-to-end proof that the registry options bag reaches the
    /// decode behaviour: the method-1 conformance stream decodes
    /// differently with `ecosystem-compat` on vs off (the §7.4.4.5
    /// intra exemption), and the differences are the expected ±1
    /// F[7][7]-toggle ripple.
    #[test]
    fn registry_option_changes_method1_decode_output() {
        let stream = std::fs::read(format!(
            "{}/tests/fixtures/mq_ipb_64x64.m4v",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let run = |compat: bool| {
            let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
            if compat {
                params.options = oxideav_core::CodecOptions::new().set("ecosystem-compat", "true");
            }
            let mut dec = make_decoder(&params).unwrap();
            dec.send_packet(&packet(stream.clone())).unwrap();
            dec.flush().unwrap();
            let mut planes = Vec::new();
            while let Ok(oxideav_core::Frame::Video(v)) = dec.receive_frame() {
                for p in v.planes {
                    planes.extend(p.data);
                }
            }
            planes
        };
        let spec = run(false);
        let compat = run(true);
        assert_eq!(spec.len(), compat.len());
        assert!(!spec.is_empty());
        let differing = spec
            .iter()
            .zip(compat.iter())
            .filter(|(a, b)| a != b)
            .count();
        let max = spec
            .iter()
            .zip(compat.iter())
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).unsigned_abs())
            .max()
            .unwrap();
        assert!(
            differing > 1000,
            "expected the intra-mismatch exemption to alter the decode, got {differing}"
        );
        assert!(max <= 2, "toggle ripple must stay small, got max {max}");
    }
    #[test]
    fn skipped_s_gmc_colocated_maps_to_the_averaged_motion_vector() {
        // §7.6.9.6: a skipped co-located S(GMC) macroblock is treated as
        // a NON-skipped macroblock carrying the §7.8.7.3 averaged motion
        // vector; a coded GMC macroblock (and an intra one) keeps the
        // §7.6.9.5.1 zero-vector fallback.
        use crate::block::InterMacroblock;
        use crate::motion::MotionVector;
        use crate::reconstruct::ReconstructedMacroblock;

        let amv = MotionVector { x: 6, y: -2 };
        let entries = vec![
            SGmcMbContent::Gmc {
                amv,
                not_coded: true,
                residual: InterMacroblock::zero(),
            },
            SGmcMbContent::Gmc {
                amv,
                not_coded: false,
                residual: InterMacroblock::zero(),
            },
            SGmcMbContent::Local {
                motion: PvopMbMotion::OneMv(MotionVector { x: 1, y: 2 }),
                residual: InterMacroblock::zero(),
            },
            SGmcMbContent::Intra(ReconstructedMacroblock {
                luma: [[0; 16]; 16],
                cb: [[0; 8]; 8],
                cr: [[0; 8]; 8],
            }),
        ];
        let motion = motion_of_s_entries(&entries);
        assert_eq!(motion[0], AnchorMbMotion::Frame(PvopMbMotion::OneMv(amv)));
        assert_eq!(motion[1], AnchorMbMotion::Frame(PvopMbMotion::Intra));
        assert_eq!(
            motion[2],
            AnchorMbMotion::Frame(PvopMbMotion::OneMv(MotionVector { x: 1, y: 2 }))
        );
        assert_eq!(motion[3], AnchorMbMotion::Frame(PvopMbMotion::Intra));
    }
}

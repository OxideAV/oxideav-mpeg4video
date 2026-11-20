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
use crate::compat::{DecodeOptions, InverseTransformSelection};
use crate::frame_decode::{FrameDecodeError, PVopMbContent, SGmcMbContent};
use crate::framestore::DecodedFrame;
use crate::pvop_mv::PvopMbMotion;
use crate::sequence::SequenceDecoder;
use crate::vol::{
    SpriteEnable, VIDEO_OBJECT_LAYER_START_CODE_MAX, VIDEO_OBJECT_LAYER_START_CODE_MIN,
    VIDEO_OBJECT_START_CODE_MAX, VIDEO_OBJECT_START_CODE_MIN, VISUAL_OBJECT_SEQUENCE_END_CODE,
    VISUAL_OBJECT_SEQUENCE_START_CODE, VISUAL_OBJECT_START_CODE, VolHeader, VolParseError,
    parse_video_object_layer, parse_visual_object_header, parse_visual_object_sequence_header,
};
use crate::vop::{
    GROUP_OF_VOP_START_CODE, VOP_START_CODE, VopCodingType, VopContext, VopHeader, VopParseError,
    parse_group_of_vop_header, parse_vop_header_body,
};
use crate::vop_decode::{
    AnchorMbMotion, VopDecodeError, decode_b_vop_interlaced_macroblocks, decode_b_vop_macroblocks,
    decode_i_vop_macroblocks, decode_p_vop_macroblocks, decode_s_gmc_vop_macroblocks,
    vop_mb_dimensions,
};

/// §6.2.1 `user_data_start_code`.
const USER_DATA_START_CODE: u32 = 0x0000_01B2;

/// The four bytes a user-data payload must begin with for the
/// inverse-transform stream evidence (see [`encoder_build_of`]).
const ENCODER_SIGNATURE: &[u8; 4] = b"XviD";

/// The inverse-transform stream evidence of one user-data payload (the
/// bytes after the four start-code bytes): `Some(build)` when the
/// payload begins with the signature bytes immediately followed by at
/// least one ASCII decimal digit — `build` being the maximal digit
/// run read as an unsigned decimal number — and `None` otherwise.
///
/// The match is anchored at payload offset 0 and case-sensitive; a
/// digit run that does not fit 32 bits is no match; bytes after the
/// digit run are ignored.
///
/// The payload is one §6.2.2.1 `user_data()`: its bytes run to the
/// next start-code prefix, so a signature split over two units (or
/// over two calls of the decoder) is never assembled.
fn encoder_build_of(payload: &[u8]) -> Option<u32> {
    let rest = payload.strip_prefix(ENCODER_SIGNATURE)?;
    let digits = rest.iter().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    rest[..digits].iter().try_fold(0u32, |build, &digit| {
        build.checked_mul(10)?.checked_add(u32::from(digit - b'0'))
    })
}

/// Errors raised by the elementary-stream decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    /// §7.14 NEWPRED: a VOP names, in `vop_id_for_prediction`, a
    /// reference VOP that is not in the reference memory (never
    /// decoded, decoded before the last I-VOP, or pushed out of the
    /// memory by newer VOPs).
    MissingNewpredReference {
        /// The `vop_id` the VOP asked to be predicted from.
        vop_id: u16,
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
            StreamDecodeError::MissingNewpredReference { vop_id } => write!(
                f,
                "stream decode: NEWPRED reference vop_id {vop_id} is not in the reference memory"
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
    /// §7.8.2 sprite memory of a basic static-sprite VOL: the
    /// reconstructed initial I-VOP (never displayed).
    sprite: Option<crate::static_sprite::SpritePlanes>,
    /// §7.14 NEWPRED reference memory: the anchors decoded since the
    /// last I-VOP, by `vop_id`, newest last (at most
    /// [`NEWPRED_MEMORY_VOPS`]).
    newpred_memory: std::collections::VecDeque<(u16, DecodedFrame)>,
    /// `video_object_id` of the most recent `video_object_start_code`
    /// (0 when the stream carried none).
    video_object_id: u8,
    /// The layer the fields above describe: `(video_object_id,
    /// video_object_layer_id)` of the most recent
    /// `video_object_layer_start_code`. `None` before the first VOL.
    layer: Option<(u8, u8)>,
    /// The decoding state of the other layers met so far (at most
    /// [`MAX_PARKED_LAYERS`], oldest first).
    parked: Vec<((u8, u8), LayerState)>,
    /// VOPs of scalability enhancement layers stepped over.
    enhancement_vops_skipped: u64,
    /// What the opt-in error concealment did so far.
    concealment: ConcealmentStats,
    /// The caller's worker budget (0 and 1 both mean serial).
    threads: usize,
    /// The build number of the most recent user-data unit carrying the
    /// encoder signature ([`encoder_build_of`]). Once set it is never
    /// cleared.
    encoder_build: Option<u32>,
}

/// What the opt-in error concealment
/// ([`DecodeOptions::error_concealment`]) did to a stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConcealmentStats {
    /// Damaged VOPs replaced by a copy of their forward reference.
    pub concealed_vops: u64,
    /// Damaged VOPs that produced no picture: their header could not
    /// be read, or there was no reference to copy.
    pub dropped_vops: u64,
}

/// The decoding state that belongs to one video object layer: its
/// configuration, reference chain, time bases and memories.
#[derive(Debug, Default)]
struct LayerState {
    vol: Option<VolHeader>,
    sequence: SequenceDecoder,
    anchor_motion: Option<Vec<AnchorMbMotion>>,
    sync_sec: u64,
    b_base_sec: u64,
    prev_anchor_ticks: Option<i64>,
    last_anchor_ticks: Option<i64>,
    sprite: Option<crate::static_sprite::SpritePlanes>,
    newpred_memory: std::collections::VecDeque<(u16, DecodedFrame)>,
}

/// Number of layers, besides the one being decoded, whose decoding
/// state (reference pictures included) is kept while the stream moves
/// between layers. A stream that visits more layers than this loses
/// the state of the least recently visited one: its next VOPs need a
/// fresh I-VOP.
pub const MAX_PARKED_LAYERS: usize = 4;

/// Number of decoded anchors the §7.14 NEWPRED reference memory keeps
/// (the specification leaves the size of the "additional reference
/// VOP memory" to the decoder; a selection older than this, or older
/// than the last I-VOP, is reported as
/// [`StreamDecodeError::MissingNewpredReference`]).
pub const NEWPRED_MEMORY_VOPS: usize = 16;

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

    /// The encoder build number found in the stream's user data, if
    /// any: a user-data unit (`00 00 01 B2`) whose payload begins with
    /// the bytes `XviD` immediately followed by decimal digits has been
    /// decoded, and this is its digit run as a number (the most recent
    /// one when several units matched). Units are examined wherever
    /// they appear in the data handed to [`Mpeg4VideoDecoder::decode`],
    /// configuration data included, and in every mode — this accessor
    /// reports the evidence even when the options do not act on it.
    ///
    /// Under [`InverseTransformSelection::FromStream`] (what the
    /// ecosystem-compat mode selects) a detected build switches the
    /// decoder to the integer row–column inverse transform for every
    /// texture block decoded afterwards; the value itself does not
    /// influence the transform. It is never cleared.
    pub fn detected_encoder_build(&self) -> Option<u32> {
        self.encoder_build
    }

    /// The inverse transform the next VOP's texture blocks will be
    /// reconstructed with: [`InverseTransformSelection::Exact`] or
    /// [`InverseTransformSelection::VariantRc`] — the options'
    /// selection with `FromStream` resolved against
    /// [`Mpeg4VideoDecoder::detected_encoder_build`]. (Short-header
    /// pictures always use the exact transform.)
    pub fn active_inverse_transform(&self) -> InverseTransformSelection {
        match self.options.inverse_transform_selection() {
            InverseTransformSelection::VariantRc => InverseTransformSelection::VariantRc,
            InverseTransformSelection::FromStream if self.encoder_build.is_some() => {
                InverseTransformSelection::VariantRc
            }
            _ => InverseTransformSelection::Exact,
        }
    }

    /// The options one VOP's macroblock walk runs with: the decoder's
    /// own, with the inverse-transform selection resolved to the
    /// transform currently in effect.
    fn vop_options(&self) -> DecodeOptions {
        self.options
            .with_inverse_transform(self.active_inverse_transform())
    }

    /// Carry stream evidence over from a previous decoder instance of
    /// the same stream (a seek rebuilds the decoder; the selection
    /// lasts for the stream).
    fn inherit_stream_evidence(&mut self, previous: &Self) {
        self.encoder_build = previous.encoder_build;
    }

    /// Return each VOP's frame from the call that decodes it, in coding
    /// order, rather than in §6.1.3.8 display order (for a container
    /// whose presentation times order the frames). Each frame carries
    /// the container pts of the call that decoded it and its §6.3.5
    /// tick time; [`Mpeg4VideoDecoder::flush`] then returns nothing.
    /// Default `false` (display order). Select it before the first
    /// VOP: an anchor already held for display when it is switched on
    /// is released only by the next flush.
    pub fn set_coded_order_output(&mut self, coded_order: bool) {
        self.sequence.set_coded_order(coded_order);
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
        let threads = self.threads;
        crate::frame_decode::with_assembly_threads(threads, || {
            self.decode_units(data, container_pts)
        })
    }

    /// Let the decoder use up to `threads` workers of its own (default
    /// 1: everything runs on the calling thread). The budget goes to
    /// the motion-compensated reconstruction of P- and B-VOPs, split
    /// over bands of macroblock rows on scoped threads for the
    /// duration of a call; the decoded pictures are the same at every
    /// budget.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// The worker budget in effect ([`Mpeg4VideoDecoder::set_threads`]).
    pub fn threads(&self) -> usize {
        self.threads.max(1)
    }

    fn decode_units(
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
            if let Some(sh) = first_sh
                && first_sc.is_none_or(|sc| sh < sc)
            {
                self.short_header = Some(ShortHeaderClock::default());
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

    /// Release the final held anchor at end-of-stream (§6.1.3.8) —
    /// of the current layer first, then of every other layer the
    /// stream visited.
    pub fn flush(&mut self) -> Vec<DecodedFrame> {
        let current = self.layer.unwrap_or((0, 0));
        let mut out = self.sequence.flush();
        for frame in out.iter_mut() {
            frame.set_layer(current);
        }
        for (id, state) in self.parked.iter_mut() {
            for mut frame in state.sequence.flush() {
                frame.set_layer(*id);
                out.push(frame);
            }
        }
        out
    }

    /// The layer the next VOP belongs to: `(video_object_id,
    /// video_object_layer_id)` of the most recent
    /// `video_object_layer_start_code`; `None` before the first one.
    pub fn current_layer(&self) -> Option<(u8, u8)> {
        self.layer
    }

    /// Number of VOPs stepped over because their layer is a
    /// scalability enhancement layer (`scalability == 1`): the base
    /// layer of such a stream decodes, the enhancement does not.
    pub fn enhancement_vops_skipped(&self) -> u64 {
        self.enhancement_vops_skipped
    }

    /// Make `id` the current layer: park the state of the layer being
    /// left and bring back (or start) the state of the one entered.
    fn enter_layer(&mut self, id: (u8, u8)) {
        if self.layer == Some(id) {
            return;
        }
        let Some(leaving) = self.layer.replace(id) else {
            // The first layer of the stream: nothing to park.
            return;
        };
        let coded_order = self.sequence.coded_order();
        let entering = match self.parked.iter().position(|(held, _)| *held == id) {
            Some(index) => self.parked.remove(index).1,
            None => {
                let mut fresh = LayerState::default();
                fresh.sequence.set_coded_order(coded_order);
                fresh
            }
        };
        let left = LayerState {
            vol: std::mem::replace(&mut self.vol, entering.vol),
            sequence: std::mem::replace(&mut self.sequence, entering.sequence),
            anchor_motion: std::mem::replace(&mut self.anchor_motion, entering.anchor_motion),
            sync_sec: std::mem::replace(&mut self.sync_sec, entering.sync_sec),
            b_base_sec: std::mem::replace(&mut self.b_base_sec, entering.b_base_sec),
            prev_anchor_ticks: std::mem::replace(
                &mut self.prev_anchor_ticks,
                entering.prev_anchor_ticks,
            ),
            last_anchor_ticks: std::mem::replace(
                &mut self.last_anchor_ticks,
                entering.last_anchor_ticks,
            ),
            sprite: std::mem::replace(&mut self.sprite, entering.sprite),
            newpred_memory: std::mem::replace(&mut self.newpred_memory, entering.newpred_memory),
        };
        if self.parked.len() == MAX_PARKED_LAYERS {
            self.parked.remove(0);
        }
        self.parked.push((leaving, left));
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
        if let Some(pending) = produced_anchor(&mut self.sequence, out) {
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
            USER_DATA_START_CODE => {
                // Inverse-transform stream evidence: sticky once seen.
                if let Some(build) = encoder_build_of(&unit[4..]) {
                    self.encoder_build = Some(build);
                }
            }
            VISUAL_OBJECT_START_CODE => {
                let _ = parse_visual_object_header(unit)?;
            }
            VIDEO_OBJECT_START_CODE_MIN..=VIDEO_OBJECT_START_CODE_MAX => {
                // A bare video_object_start_code carries only its id:
                // the object the following layers belong to.
                self.video_object_id = (code - VIDEO_OBJECT_START_CODE_MIN) as u8;
            }
            VIDEO_OBJECT_LAYER_START_CODE_MIN..=VIDEO_OBJECT_LAYER_START_CODE_MAX => {
                // §6.2.1: the VOPs that follow belong to this layer.
                // Each layer has its own reference chain and clock.
                let layer_id = (code - VIDEO_OBJECT_LAYER_START_CODE_MIN) as u8;
                self.enter_layer((self.video_object_id, layer_id));
                let vol = parse_video_object_layer(unit, self.profile_level)?;
                if self.vol != Some(vol) {
                    // A different layer configuration: its sprite (if
                    // any) arrives with its own initial I-VOP.
                    self.sprite = None;
                }
                self.vol = Some(vol);
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
                if self.vol.is_some_and(|vol| vol.scalability) {
                    // A scalability enhancement layer: its VOPs refine
                    // another layer's pictures (ref_select_code,
                    // §7.9). The layers it builds on decode without
                    // it; its own VOPs are stepped over.
                    self.enhancement_vops_skipped += 1;
                    return Ok(());
                }
                let first = out.len();
                match self.decode_vop_unit(unit, container_pts, out, false) {
                    Ok(()) => {}
                    Err(error) if self.options.error_concealment && is_stream_damage(&error) => {
                        self.conceal_vop(unit, container_pts, out);
                    }
                    Err(error) => return Err(error),
                }
                let layer = self.layer.unwrap_or((0, 0));
                for frame in &mut out[first..] {
                    frame.set_layer(layer);
                }
            }
            _ => {
                // Unknown / reserved start code: skip. (Stuffing and
                // extension units do not affect the layers we decode.)
            }
        }
        Ok(())
    }

    /// [`DecodeOptions::error_concealment`]: stand in for a VOP that
    /// failed to decode. With a readable header and a reference, the
    /// VOP becomes the §6.3.5 `vop_coded == 0` copy of its forward
    /// reference (an anchor still enters the chain and moves the
    /// clock); otherwise it is dropped.
    fn conceal_vop(
        &mut self,
        unit: &[u8],
        container_pts: Option<i64>,
        out: &mut Vec<DecodedFrame>,
    ) {
        let first = out.len();
        self.sequence.set_reference_override(None);
        if self
            .decode_vop_unit(unit, container_pts, out, true)
            .is_err()
        {
            out.truncate(first);
            self.concealment.dropped_vops += 1;
            return;
        }
        self.concealment.concealed_vops += 1;
    }

    /// What the opt-in error concealment did so far (all zero while
    /// [`DecodeOptions::error_concealment`] is off).
    pub fn concealment(&self) -> ConcealmentStats {
        self.concealment
    }

    /// Decode one `video_object_plane` unit end-to-end.
    /// `as_uncoded` reconstructs it as the §6.3.5 `vop_coded == 0`
    /// copy of its forward reference whatever its body says (the
    /// concealment of a damaged VOP).
    fn decode_vop_unit(
        &mut self,
        unit: &[u8],
        container_pts: Option<i64>,
        out: &mut Vec<DecodedFrame>,
        as_uncoded: bool,
    ) -> Result<(), StreamDecodeError> {
        let vol = self.vol.ok_or(StreamDecodeError::MissingVol)?;
        if !(4..=12).contains(&vol.bits_per_pixel) {
            // §6.3.3: `bits_per_pixel` takes values between 4 and 12.
            return Err(StreamDecodeError::Vol(VolParseError::UnsupportedBranch(
                "bits_per_pixel outside 4..=12",
            )));
        }
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

        if matches!(vol.sprite_enable, SpriteEnable::Static) {
            if as_uncoded {
                // A sprite VOP is not a copy of a previous picture.
                return Err(StreamDecodeError::MissingAnchor);
            }
            return self.decode_static_sprite_vop(
                &vol,
                &vop,
                &mut br,
                res,
                (container_pts, vop_ticks),
                out,
            );
        }

        if !vop.coded || as_uncoded {
            return self.decode_uncoded_vop(
                &vop,
                (mb_count, as_uncoded),
                mb_width,
                mb_height,
                res,
                (container_pts, vop_ticks),
                out,
            );
        }

        // §7.14 NEWPRED: select the reference this VOP names.
        self.sequence.set_reference_override(None);
        if let Some(newpred) = vol.newpred {
            if !newpred.segment_is_vop {
                // Per-packet reference selection restricts motion
                // compensation to each packet's own area of its own
                // reference picture.
                return Err(StreamDecodeError::Vop(VopParseError::UnsupportedBranch(
                    "newpred_segment_type == Video Packet (per-packet reference selection, \
                     subclause 7.14.1)",
                )));
            }
            if let Some(id) = vop.vop_id_for_prediction {
                match vop.coding_type {
                    // An I-VOP predicts from nothing.
                    VopCodingType::I => {}
                    VopCodingType::P | VopCodingType::S => {
                        let reference = self
                            .newpred_memory
                            .iter()
                            .rev()
                            .find(|(held, _)| *held == id)
                            .map(|(_, frame)| frame.clone())
                            .ok_or(StreamDecodeError::MissingNewpredReference { vop_id: id })?;
                        self.sequence.set_reference_override(Some(reference));
                    }
                    VopCodingType::B => {
                        // §7.14.1 replaces "the reference VOP"; a
                        // B-VOP has two.
                        return Err(StreamDecodeError::Vop(VopParseError::UnsupportedBranch(
                            "vop_id_for_prediction on a B-VOP",
                        )));
                    }
                }
            }
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
                        self.vop_options(),
                    )?
                } else {
                    decode_i_vop_macroblocks(&mut br, &vol, &vop, self.vop_options())?
                };
                out.extend(self.sequence.push_i_vop_depth(
                    mb_width,
                    mb_height,
                    &mbs,
                    u32::from(vol.bits_per_pixel),
                )?);
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
                        self.vop_options(),
                    )?
                } else {
                    decode_p_vop_macroblocks(&mut br, &vol, &vop, self.vop_options())?
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
                let (entries, geometry) = if vol.data_partitioned {
                    crate::vop_decode::decode_s_gmc_vop_macroblocks_dp(
                        &mut br,
                        &vol,
                        &vop,
                        self.vop_options(),
                    )?
                } else {
                    decode_s_gmc_vop_macroblocks(&mut br, &vol, &vop, self.vop_options())?
                };
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
                        self.vop_options(),
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
                        self.vop_options(),
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
            if let Some(pending) = produced_anchor(&mut self.sequence, out) {
                pending.set_pts(container_pts);
                pending.set_pts_ticks(Some(vop_ticks));
            }
            self.advance_anchor_time(&vop, res);
            // §7.14 / §E.1.6.4: a correctly decoded anchor joins the
            // NEWPRED reference memory; nothing before the last I-VOP
            // may be referred to.
            if let Some(id) = vop.vop_id {
                if matches!(vop.coding_type, VopCodingType::I) {
                    self.newpred_memory.clear();
                }
                if let Some(frame) = self.sequence.store().backward() {
                    if self.newpred_memory.len() == NEWPRED_MEMORY_VOPS {
                        self.newpred_memory.pop_front();
                    }
                    self.newpred_memory.push_back((id, frame.clone()));
                }
            }
        } else if let Some(frame) = out.last_mut() {
            frame.set_pts(container_pts);
            frame.set_pts_ticks(Some(vop_ticks));
        }
        Ok(())
    }

    /// One VOP of a basic static-sprite VOL (§7.8, `sprite_enable ==
    /// "static"`, `low_latency_sprite_enable == 0`).
    ///
    /// §7.8.1: the initial I-VOP is the sprite — reconstructed at
    /// `sprite_width × sprite_height` (§7.8.2 gives the sprite arrays
    /// those dimensions, and §6.2.5 sends no VOP size for it), stored
    /// in the sprite memory and not displayed. Every later VOP is an
    /// S-VOP carrying only `sprite_trajectory()` (+ the brightness
    /// factor): it is reconstructed by warping the sprite memory
    /// (§7.8.4 reference points, §7.8.5 warp — affine for up to three
    /// points, perspective for four — and the §7.8.6 static sample
    /// reconstruction).
    fn decode_static_sprite_vop(
        &mut self,
        vol: &VolHeader,
        vop: &VopHeader,
        br: &mut BitReader<'_>,
        res: i64,
        (container_pts, vop_ticks): (Option<i64>, i64),
        out: &mut Vec<DecodedFrame>,
    ) -> Result<(), StreamDecodeError> {
        let unsupported = |what| StreamDecodeError::Vop(VopParseError::UnsupportedBranch(what));
        if vol.low_latency_sprite_enable.unwrap_or(false) {
            return Err(unsupported("low_latency_sprite_enable"));
        }
        if vol.bits_per_pixel != 8 {
            // The §7.8.2 sprite memory is an 8-bit store.
            return Err(unsupported("static sprite with bits_per_pixel != 8"));
        }
        let geometry = vol
            .sprite_geometry
            .ok_or_else(|| unsupported("static sprite without geometry"))?;
        match vop.coding_type {
            VopCodingType::I => {
                if !vop.coded {
                    return Err(unsupported("static sprite I-VOP with vop_coded == 0"));
                }
                // The sprite I-VOP is coded on the sprite's own
                // macroblock grid.
                let sprite_vol = VolHeader {
                    width: geometry.width,
                    height: geometry.height,
                    ..*vol
                };
                let (mb_width, mb_height) = vop_mb_dimensions(&sprite_vol);
                let mbs = if vol.data_partitioned {
                    crate::vop_decode::decode_i_vop_macroblocks_dp(
                        br,
                        &sprite_vol,
                        vop,
                        self.vop_options(),
                    )?
                } else {
                    decode_i_vop_macroblocks(br, &sprite_vol, vop, self.vop_options())?
                };
                let mut frame = DecodedFrame::new(mb_width * 16, mb_height * 16, VopCodingType::I)
                    .map_err(FrameDecodeError::from)?;
                for (idx, mb) in mbs.iter().enumerate() {
                    frame
                        .blit_macroblock(idx % mb_width, idx / mb_width, mb)
                        .map_err(FrameDecodeError::from)?;
                }
                // Crop the macroblock-aligned reconstruction to the
                // sprite arrays (§7.8.2: sprite_width × sprite_height
                // luminance, `// 2` chrominance).
                self.sprite = Some(
                    crate::static_sprite::SpritePlanes::from_aligned(
                        frame.luma_samples(),
                        frame.cb_samples(),
                        frame.cr_samples(),
                        mb_width * 16,
                        usize::from(geometry.width),
                        usize::from(geometry.height),
                        i64::from(geometry.left),
                        i64::from(geometry.top),
                    )
                    .ok_or_else(|| unsupported("static sprite geometry"))?,
                );
                self.anchor_motion = None;
                self.advance_anchor_time(vop, res);
                Ok(())
            }
            VopCodingType::S => {
                let sprite = self
                    .sprite
                    .as_ref()
                    .ok_or(StreamDecodeError::MissingAnchor)?;
                if vol.width == 0 || vol.height == 0 {
                    return Err(unsupported("static sprite VOP of zero size"));
                }
                let (mb_width, mb_height) = vop_mb_dimensions(vol);
                let (fw, fh) = (mb_width * 16, mb_height * 16);
                let accuracy = vol
                    .sprite_warping_accuracy
                    .ok_or_else(|| unsupported("static sprite without warping accuracy"))?;
                let bpp = u32::from(vol.bits_per_pixel);
                // `vop_coded == 0` carries no trajectory: the VOP shows
                // the sprite through the identity warp.
                let block = vop.static_sprite.unwrap_or(crate::vop::StaticSpriteVop {
                    warping_points: 0,
                    trajectory: [[0; 2]; 4],
                    brightness_change: None,
                });
                let (luma, cb, cr) = crate::static_sprite::warp_static_sprite_vop(
                    sprite,
                    u32::from(vol.width),
                    u32::from(vol.height),
                    fw,
                    fh,
                    accuracy,
                    bpp,
                    block.warping_points,
                    &block.trajectory,
                    block.brightness_change,
                )
                .ok_or_else(|| unsupported("static sprite warp"))?;
                let frame = DecodedFrame::from_planes(fw, fh, VopCodingType::S, luma, cb, cr)
                    .map_err(FrameDecodeError::from)?;
                out.extend(self.sequence.push_reconstructed_anchor(frame));
                self.anchor_motion = None;
                if let Some(pending) = produced_anchor(&mut self.sequence, out) {
                    pending.set_pts(container_pts);
                    pending.set_pts_ticks(Some(vop_ticks));
                }
                self.advance_anchor_time(vop, res);
                Ok(())
            }
            // §7.8.1: a static-sprite VOL holds one I-VOP and S-VOPs.
            _ => Err(unsupported("P/B-VOP in a static sprite VOL")),
        }
    }

    /// §6.3.5 `vop_coded == 0`: the reconstructed VOP is a copy of the
    /// forward reference (per §7.6.7). An uncoded anchor still enters
    /// the reference chain and advances the time base; an uncoded
    /// B-VOP displays the copy immediately.
    #[allow(clippy::too_many_arguments)]
    fn decode_uncoded_vop(
        &mut self,
        vop: &VopHeader,
        (mb_count, concealed): (usize, bool),
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
                    if concealed {
                        frame.set_concealed_macroblocks(mb_count as u32);
                    }
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
                if let Some(pending) = produced_anchor(&mut self.sequence, out) {
                    pending.set_pts(container_pts);
                    pending.set_pts_ticks(Some(vop_ticks));
                    if concealed {
                        pending.set_concealed_macroblocks(mb_count as u32);
                    }
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
/// * A **coded** GMC macroblock (`mcsel == 1`) carries the same AMV:
///   §7.6.9.5.1 "The averaged motion vector defined in subclause 7.8.7.3
///   is used as the MV of the co-located macroblock when this
///   co-located macroblock is included in an S(GMC)-VOP and mcsel ==
///   '1'", restated by §7.8.7.3 for the frame-based direct mode.
/// * An intra macroblock carries no vector — the §7.6.9.5.1 zero-vector
///   fallback (represented as `Intra` for the co-located adapter).
fn motion_of_s_entries(entries: &[SGmcMbContent]) -> Vec<AnchorMbMotion> {
    entries
        .iter()
        .map(|e| match e {
            SGmcMbContent::Local { motion, .. } => AnchorMbMotion::Frame(*motion),
            SGmcMbContent::FieldLocal {
                mvs,
                top_field_ref,
                bottom_field_ref,
                ..
            } => AnchorMbMotion::Field {
                mvs: *mvs,
                top_ref: *top_field_ref,
                bottom_ref: *bottom_field_ref,
            },
            SGmcMbContent::Gmc { amv, .. } => AnchorMbMotion::Frame(PvopMbMotion::OneMv(*amv)),
            SGmcMbContent::Intra(_) => AnchorMbMotion::Frame(PvopMbMotion::Intra),
        })
        .collect()
}

/// The frame the anchor VOP just pushed produced: the §6.1.3.8 held
/// anchor in display order, or the frame appended to `out` in coded
/// order ([`Mpeg4VideoDecoder::set_coded_order_output`]).
/// Whether a VOP decode error points at damaged stream data (which
/// [`DecodeOptions::error_concealment`] conceals) rather than at a
/// tool this decoder declines or a stream that never had the anchors
/// the VOP needs (which stay errors).
fn is_stream_damage(error: &StreamDecodeError) -> bool {
    match error {
        StreamDecodeError::Vop(VopParseError::UnsupportedBranch(_))
        | StreamDecodeError::VopDecode(VopDecodeError::Unsupported(_))
        | StreamDecodeError::Vol(_)
        | StreamDecodeError::MissingVol
        | StreamDecodeError::MissingAnchor
        | StreamDecodeError::MissingNewpredReference { .. }
        | StreamDecodeError::ShortHeader(_) => false,
        StreamDecodeError::Vop(_)
        | StreamDecodeError::VopDecode(_)
        | StreamDecodeError::Frame(_)
        | StreamDecodeError::BadTemporalReference { .. } => true,
    }
}

fn produced_anchor<'a>(
    sequence: &'a mut SequenceDecoder,
    out: &'a mut [DecodedFrame],
) -> Option<&'a mut DecodedFrame> {
    if sequence.coded_order() {
        out.last_mut()
    } else {
        sequence.pending_anchor_mut()
    }
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
    /// The encoder build number found in the stream's user data so far
    /// (see [`Mpeg4VideoDecoder::detected_encoder_build`]); kept across
    /// [`Decoder::reset`](oxideav_core::Decoder::reset).
    pub fn detected_encoder_build(&self) -> Option<u32> {
        self.inner.detected_encoder_build()
    }

    /// The inverse transform currently in effect (see
    /// [`Mpeg4VideoDecoder::active_inverse_transform`]).
    pub fn active_inverse_transform(&self) -> InverseTransformSelection {
        self.inner.active_inverse_transform()
    }

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
            // The §7.8.2 sprite memory is a picture buffer too.
            if let Some(sprite) = vol.sprite_geometry {
                let pixels = u64::from(sprite.width) * u64::from(sprite.height);
                if pixels > self.max_pixels {
                    return Err(oxideav_core::Error::resource_exhausted(format!(
                        "mpeg4video: sprite {}x{} exceeds max_pixels_per_frame {}",
                        sprite.width, sprite.height, self.max_pixels
                    )));
                }
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
        let bpp = frame.bits_per_pixel();
        let (_, target) = registry_format(bpp);
        let shift = target - bpp;
        let narrow = |plane: &[u8]| -> Vec<u8> {
            if shift == 0 {
                plane.to_vec()
            } else {
                plane.iter().map(|&v| v << shift).collect()
            }
        };
        let wide = |plane: &[u16]| -> Vec<u8> {
            plane
                .iter()
                .flat_map(|&v| (v << shift).to_le_bytes())
                .collect()
        };
        let planes = match (
            frame.luma_samples16(),
            frame.cb_samples16(),
            frame.cr_samples16(),
        ) {
            (Some(y), Some(cb), Some(cr)) => vec![
                oxideav_core::VideoPlane {
                    stride: 2 * w,
                    data: wide(y),
                },
                oxideav_core::VideoPlane {
                    stride: w,
                    data: wide(cb),
                },
                oxideav_core::VideoPlane {
                    stride: w,
                    data: wide(cr),
                },
            ],
            _ => vec![
                oxideav_core::VideoPlane {
                    stride: w,
                    data: narrow(frame.luma_samples()),
                },
                oxideav_core::VideoPlane {
                    stride: w / 2,
                    data: narrow(frame.cb_samples()),
                },
                oxideav_core::VideoPlane {
                    stride: w / 2,
                    data: narrow(frame.cr_samples()),
                },
            ],
        };
        oxideav_core::VideoFrame {
            pts: frame.pts().or(frame.pts_ticks()),
            planes,
        }
    }
}

/// The framework pixel format carrying a §6.3.3 `bits_per_pixel`
/// picture, with that format's sample depth: 8-bit planar 4:2:0 up to
/// 8 bits, the 10-bit format for 9 and 10, the 12-bit format for 11
/// and 12. A picture shallower than its format is scaled up by a left
/// shift of the difference (exactly undone by the same right shift),
/// so black and full scale land where the format expects them.
fn registry_format(bits_per_pixel: u32) -> (oxideav_core::PixelFormat, u32) {
    match bits_per_pixel {
        0..=8 => (oxideav_core::PixelFormat::Yuv420P, 8),
        9 | 10 => (oxideav_core::PixelFormat::Yuv420P10Le, 10),
        _ => (oxideav_core::PixelFormat::Yuv420P12Le, 12),
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

    /// Planar 4:2:0, known before the first packet whenever the
    /// configuration headers arrived as extradata, so a pipeline needs
    /// no pre-roll to label the frames.
    ///
    /// The layout is fixed by the syntax rather than by a per-picture
    /// choice: the VOL's `chroma_format` has a single defined value,
    /// 4:2:0 (§6.3.3 Table 6-15 — the others are reserved), and
    /// §6.2.5.2 short-header pictures are 4:2:0 as well. The sample
    /// depth is the VOL's: one byte per sample unless `not_8_bit`
    /// selects more than 8 bits, in which case the planes are 16-bit
    /// little-endian words of the 10-bit (9 or 10 bits) or 12-bit (11
    /// or 12 bits) format. Before any VOL is known the answer is the
    /// 8-bit layout every non-N-bit object type uses.
    fn output_pixel_format(&self) -> Option<oxideav_core::PixelFormat> {
        let bpp = self
            .inner
            .vol()
            .map_or(8, |vol| u32::from(vol.bits_per_pixel));
        Some(registry_format(bpp).0)
    }

    /// The caller's thread budget: see
    /// [`Mpeg4VideoDecoder::set_threads`]. Without this call the
    /// decoder is serial.
    fn set_execution_context(&mut self, ctx: &oxideav_core::ExecutionContext) {
        self.inner.set_threads(ctx.threads);
    }

    fn reset(&mut self) -> oxideav_core::Result<()> {
        // Preserve the compat behaviour selection — and the stream
        // evidence that selected the inverse transform — across seeks.
        let mut inner = Mpeg4VideoDecoder::with_options(self.inner.options());
        inner.inherit_stream_evidence(&self.inner);
        inner.set_threads(self.inner.threads());
        self.inner = inner;
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
/// Two keys are recognised:
///
/// * `ecosystem-compat` (bool, default `false`) — decode with the
///   ecosystem-compat behaviour on the three documented spec
///   divergences (see [`crate::compat`]): the §7.7.2.2
///   interlaced-direct derivation reads the co-located field MVs as
///   zero, the §7.4.4.5 method-1 mismatch toggle is skipped on intra
///   blocks, and non-positive §7.8.7.3 GMC averaged-MV components of a
///   half-sample translational warp derive one MV-grid unit lower.
///   The default is the literal spec text.
/// * `inverse-transform` (`exact` / `variant-rc` / `from-stream`,
///   default unset) — force the
///   [`InverseTransformSelection`]. Unset follows the mode: `exact`
///   for the literal-spec decode, `from-stream` under
///   `ecosystem-compat`.
///
/// Typed consumers can skip the bag and pass [`DecodeOptions`] to
/// [`Mpeg4VideoDecoder::with_options`] directly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Mpeg4DecoderOptions {
    /// The [`crate::compat`] ecosystem-compat switch.
    pub ecosystem_compat: bool,
    /// Explicit inverse-transform selection (`None` follows the mode;
    /// see [`DecodeOptions::inverse_transform`]).
    pub inverse_transform: Option<InverseTransformSelection>,
    /// The opt-in error concealment
    /// ([`DecodeOptions::error_concealment`]).
    pub error_concealment: bool,
}

impl Mpeg4DecoderOptions {
    /// Set `ecosystem_compat`.
    pub fn with_ecosystem_compat(mut self, v: bool) -> Self {
        self.ecosystem_compat = v;
        self
    }

    /// Set `inverse_transform` (an [`InverseTransformSelection`] forces
    /// it, `None` follows the mode).
    pub fn with_inverse_transform(
        mut self,
        selection: impl Into<Option<InverseTransformSelection>>,
    ) -> Self {
        self.inverse_transform = selection.into();
        self
    }
}

impl From<Mpeg4DecoderOptions> for DecodeOptions {
    fn from(o: Mpeg4DecoderOptions) -> Self {
        DecodeOptions::spec()
            .with_ecosystem_compat(o.ecosystem_compat)
            .with_inverse_transform(o.inverse_transform)
            .with_error_concealment(o.error_concealment)
    }
}

impl oxideav_core::CodecOptionsStruct for Mpeg4DecoderOptions {
    const SCHEMA: &'static [oxideav_core::OptionField] = &[
        oxideav_core::OptionField {
            name: "ecosystem-compat",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "match the black-box-observed ecosystem behaviour on the three documented \
                   ISO/IEC 14496-2 divergences (interlaced-direct co-located MVs read as zero; \
                   §7.4.4.5 mismatch control skipped on intra blocks; half-sample translational \
                   GMC averaged MVs lowered on non-positive components) instead of the literal \
                   spec text; also selects the inverse transform from stream evidence unless \
                   inverse-transform is set",
        },
        oxideav_core::OptionField {
            name: "error-concealment",
            kind: oxideav_core::OptionKind::Bool,
            default: oxideav_core::OptionValue::Bool(false),
            help: "replace a VOP that fails to decode by a copy of its forward reference \
                   instead of reporting an error (ISO/IEC 14496-2 E.1.4.2)",
        },
        oxideav_core::OptionField {
            name: "inverse-transform",
            kind: oxideav_core::OptionKind::Enum(&["exact", "variant-rc", "from-stream"]),
            default: oxideav_core::OptionValue::String(String::new()),
            help: "inverse transform for texture blocks: exact (Annex A, every stream), \
                   variant-rc (the integer row-column transform, every stream) or from-stream \
                   (variant-rc once a user-data unit carries the encoder signature with a build \
                   number, for the rest of the stream); unset = exact, or from-stream under \
                   ecosystem-compat",
        },
    ];

    fn apply(&mut self, key: &str, value: &oxideav_core::OptionValue) -> oxideav_core::Result<()> {
        match key {
            "ecosystem-compat" => self.ecosystem_compat = value.as_bool()?,
            "error-concealment" => self.error_concealment = value.as_bool()?,
            "inverse-transform" => {
                self.inverse_transform = Some(match value.as_str()? {
                    "exact" => InverseTransformSelection::Exact,
                    "variant-rc" => InverseTransformSelection::VariantRc,
                    "from-stream" => InverseTransformSelection::FromStream,
                    other => {
                        return Err(oxideav_core::Error::invalid(format!(
                            "mpeg4video: inverse-transform: unknown value {other:?}"
                        )));
                    }
                })
            }
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
    #[derive(Default, Clone)]
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
    fn encoder_build_needs_the_signature_and_a_digit_run_at_offset_zero() {
        assert_eq!(encoder_build_of(b"XviD0050"), Some(50));
        assert_eq!(encoder_build_of(b"XviD0073 anything"), Some(73));
        assert_eq!(encoder_build_of(b"XviD7"), Some(7));
        assert_eq!(encoder_build_of(b"XviD"), None);
        assert_eq!(encoder_build_of(b"XviD-beta"), None);
        assert_eq!(encoder_build_of(b"xvid0050"), None);
        assert_eq!(encoder_build_of(b" XviD0050"), None);
        assert_eq!(encoder_build_of(b""), None);
        assert_eq!(encoder_build_of(b"XviD4294967295"), Some(u32::MAX));
        assert_eq!(encoder_build_of(b"XviD4294967296"), None);
        assert_eq!(encoder_build_of(b"XviD00000000000000000001"), Some(1));
    }

    #[test]
    fn user_data_evidence_is_recorded_but_only_acted_on_by_selection() {
        // VOS + user data + VOL + I + P: the build is reported in
        // every mode; only a from-stream selection turns it into the
        // integer transform, and nothing turns it off again.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        w.write_bits(USER_DATA_START_CODE, 32);
        for &b in b"XviD0064" {
            w.write_bits(u32::from(b), 8);
        }
        write_vol(&mut w, 32, 16);
        write_i_vop(&mut w, 2, 0);
        write_skipped_p_vop(&mut w, 2, 1);
        let stream = w.finish();

        let mut spec = Mpeg4VideoDecoder::new();
        assert_eq!(spec.detected_encoder_build(), None);
        let frames = spec.decode(&stream).unwrap();
        assert_eq!(frames.len() + spec.flush().len(), 2);
        assert_eq!(spec.detected_encoder_build(), Some(64));
        assert_eq!(
            spec.active_inverse_transform(),
            InverseTransformSelection::Exact
        );
        assert_eq!(
            spec.vop_options(),
            DecodeOptions::spec().with_inverse_transform(InverseTransformSelection::Exact)
        );

        let mut eco = Mpeg4VideoDecoder::with_options(DecodeOptions::ecosystem());
        assert_eq!(
            eco.active_inverse_transform(),
            InverseTransformSelection::Exact
        );
        let mut frames = eco.decode(&stream).unwrap();
        frames.extend(eco.flush());
        assert_eq!(eco.detected_encoder_build(), Some(64));
        assert_eq!(
            eco.active_inverse_transform(),
            InverseTransformSelection::VariantRc
        );
        assert_eq!(
            eco.vop_options().block_transform(),
            crate::idct::BlockTransform::VariantRc
        );
        // The caller's options are untouched by the resolution.
        assert_eq!(eco.options(), DecodeOptions::ecosystem());
        // Flat mid-grey intra blocks (F[0][0] = 1024) are 128 under
        // either transform.
        assert_eq!(frames[0].luma_at(0, 0), Some(128));
        assert_eq!(frames[1].luma_at(31, 15), Some(128));

        let forced = Mpeg4VideoDecoder::with_options(
            DecodeOptions::ecosystem().with_inverse_transform(InverseTransformSelection::Exact),
        );
        assert_eq!(
            forced.active_inverse_transform(),
            InverseTransformSelection::Exact
        );
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
    fn coded_order_output_returns_each_vop_from_its_own_call() {
        // Coding order I(t=0) P(t=2) B(t=1), fed one VOP per call with
        // coded-order output: each call returns its own VOP's frame
        // (I, P, B), stamped with that call's container pts, and the
        // flush returns nothing more.
        let mut header = BitWriter::default();
        write_vos(&mut header);
        write_vol(&mut header, 16, 16);
        let mut i = BitWriter::default();
        write_i_vop(&mut i, 1, 0);
        let mut p = BitWriter::default();
        write_skipped_p_vop(&mut p, 1, 2);
        let mut b = BitWriter::default();
        write_direct_b_vop(&mut b, 1, 1);

        let mut dec = Mpeg4VideoDecoder::new();
        dec.set_coded_order_output(true);
        assert!(dec.decode(&header.finish()).unwrap().is_empty());
        let mut types = Vec::new();
        for (pts, unit) in [(0, i), (2, p), (1, b)] {
            let frames = dec.decode_with_pts(&unit.finish(), Some(pts)).unwrap();
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].pts(), Some(pts));
            types.push(frames[0].coding_type());
        }
        assert_eq!(
            types,
            vec![VopCodingType::I, VopCodingType::P, VopCodingType::B]
        );
        assert!(dec.flush().is_empty());
    }

    #[test]
    fn coded_order_frames_reorder_to_the_display_order_output() {
        // The same I P B units decoded both ways: sorting the
        // coded-order frames by their tick time reproduces the
        // display-order output frame for frame (samples, pts, ticks).
        let units = || {
            let mut header = BitWriter::default();
            write_vos(&mut header);
            write_vol(&mut header, 16, 16);
            let mut i = BitWriter::default();
            write_i_vop(&mut i, 1, 0);
            let mut p = BitWriter::default();
            write_skipped_p_vop(&mut p, 1, 2);
            let mut b = BitWriter::default();
            write_direct_b_vop(&mut b, 1, 1);
            (
                header.finish(),
                [(0, i.finish()), (2, p.finish()), (1, b.finish())],
            )
        };
        let run = |coded: bool| {
            let (header, vops) = units();
            let mut dec = Mpeg4VideoDecoder::new();
            dec.set_coded_order_output(coded);
            let mut frames = dec.decode(&header).unwrap();
            for (pts, unit) in &vops {
                frames.extend(dec.decode_with_pts(unit, Some(*pts)).unwrap());
            }
            frames.extend(dec.flush());
            frames
        };
        let display = run(false);
        let mut coded = run(true);
        coded.sort_by_key(|f| f.pts_ticks());
        assert_eq!(display.len(), 3);
        assert_eq!(coded.len(), display.len());
        for (c, d) in coded.iter().zip(&display) {
            assert_eq!(c.pts(), d.pts());
            assert_eq!(c.pts_ticks(), d.pts_ticks());
            assert_eq!(c.coding_type(), d.coding_type());
            assert_eq!(c.luma_samples(), d.luma_samples());
            assert_eq!(c.cb_samples(), d.cb_samples());
            assert_eq!(c.cr_samples(), d.cr_samples());
        }
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

    // ---- §6.2.3 / §6.2.5 optional header blocks ---------------------

    /// What the extended test VOL declares beyond the plain one.
    #[derive(Clone, Copy, Default)]
    struct VolExtras {
        /// `define_vop_complexity_estimation_header()` with the
        /// intra_blocks, vlc_bits, inter_blocks, interpolate_mc_q and
        /// quarterpel statistics.
        complexity: bool,
        /// `newpred_enable` with this `newpred_segment_type`.
        newpred_segment_is_vop: Option<bool>,
        /// `reduced_resolution_vop_enable`.
        reduced_resolution: bool,
    }

    /// A verid-2 rectangular VOL (resolution 30, method-2 quant).
    fn write_extended_vol(w: &mut BitWriter, width: u32, height: u32, x: VolExtras) {
        w.write_bits(0x0000_0120, 32);
        w.write_bits(0, 1); // random_accessible_vol
        w.write_bits(1, 8); // video_object_type_indication
        w.write_bits(1, 1); // is_object_layer_identifier
        w.write_bits(2, 4); // video_object_layer_verid
        w.write_bits(1, 3); // video_object_layer_priority
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
        w.write_bits(0, 2); // sprite_enable (two bits under verid 2)
        w.write_bits(0, 1); // not_8_bit
        w.write_bits(0, 1); // quant_type
        w.write_bits(0, 1); // quarter_sample
        if x.complexity {
            w.write_bits(0, 1); // complexity_estimation_disable = 0
            w.write_bits(0b01, 2); // estimation_method (version 2)
            w.write_bits(1, 1); // shape set disabled
            w.write_bits(0, 1); // texture set 1 enabled
            w.write_bits(0b1100, 4); // intra_blocks, inter_blocks
            w.write_bits(1, 1); // marker
            w.write_bits(0, 1); // texture set 2 enabled
            w.write_bits(0b0001, 4); // vlc_bits
            w.write_bits(0, 1); // motion set enabled
            w.write_bits(0b001000, 6); // interpolate_mc_q
            w.write_bits(1, 1); // marker
            w.write_bits(0, 1); // version-2 set enabled
            w.write_bits(0b01, 2); // quarterpel
        } else {
            w.write_bits(1, 1); // complexity_estimation_disable
        }
        w.write_bits(1, 1); // resync_marker_disable
        w.write_bits(0, 1); // data_partitioned
        match x.newpred_segment_is_vop {
            Some(segment_is_vop) => {
                w.write_bits(1, 1); // newpred_enable
                w.write_bits(0b01, 2); // requested_upstream_message_type
                w.write_bits(u32::from(segment_is_vop), 1); // newpred_segment_type
            }
            None => w.write_bits(0, 1),
        }
        w.write_bits(u32::from(x.reduced_resolution), 1);
        w.write_bits(0, 1); // scalability
        w.align();
    }

    /// The per-VOP side of [`VolExtras`].
    #[derive(Clone, Copy, Default)]
    struct VopExtras {
        vol: VolExtras,
        /// `(vop_id, vop_id_for_prediction)` of a NEWPRED VOL.
        ids: Option<(u32, Option<u32>)>,
        /// `vop_reduced_resolution` (I- and P-VOPs of an RRV VOL).
        reduced: bool,
    }

    /// The macroblock layer of a test VOP.
    #[derive(Clone, Copy)]
    enum Body {
        /// I-VOP: DC-only intra macroblocks at the default DC (grey 128).
        IntraGrey,
        /// P-VOP: every macroblock `not_coded`.
        Skipped,
        /// P-VOP: intra macroblocks whose first luminance block is one
        /// DC level above its prediction (at quant 8 the luminance DC
        /// scaler is 16: 130 in the first macroblock, 132 in the
        /// second, ...).
        IntraBrighter,
        /// B-VOP: every macroblock `modb == "1"` (direct, no residual).
        Direct,
    }

    /// One VOP of the extended VOL (quant 8, `tinc` ticks).
    fn write_extended_vop(w: &mut BitWriter, body: Body, mbs: usize, tinc: u32, x: VopExtras) {
        let coding = match body {
            Body::IntraGrey => 0b00,
            Body::Skipped | Body::IntraBrighter => 0b01,
            Body::Direct => 0b10,
        };
        w.write_bits(0x0000_01B6, 32);
        w.write_bits(coding, 2);
        w.write_bits(0, 1); // modulo_time_base
        w.write_bits(1, 1); // marker
        w.write_bits(tinc, 5);
        w.write_bits(1, 1); // marker
        w.write_bits(1, 1); // vop_coded
        if let Some((id, for_prediction)) = x.ids {
            w.write_bits(id, 8); // vop_id (5-bit increment + 3)
            w.write_bits(u32::from(for_prediction.is_some()), 1);
            if let Some(reference) = for_prediction {
                w.write_bits(reference, 8);
            }
            w.write_bits(1, 1); // marker
        }
        if coding == 0b01 {
            w.write_bits(0, 1); // vop_rounding_type
        }
        if x.vol.reduced_resolution && coding != 0b10 {
            w.write_bits(u32::from(x.reduced), 1); // vop_reduced_resolution
        }
        if x.vol.complexity {
            // intra_blocks (8) + vlc_bits (4) on every type; P and B
            // add inter_blocks and quarterpel, B interpolate_mc_q.
            let bytes = match coding {
                0b00 => 1,
                0b01 => 3,
                _ => 4,
            };
            w.write_bits(0x55, 8); // dcecs_intra_blocks
            w.write_bits(0b1010, 4); // dcecs_vlc_bits
            for k in 1..bytes {
                w.write_bits(0x30 + k, 8);
            }
        }
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(8, 5); // vop_quant
        if coding != 0b00 {
            w.write_bits(1, 3); // vop_fcode_forward
        }
        if coding == 0b10 {
            w.write_bits(1, 3); // vop_fcode_backward
        }
        for _ in 0..mbs {
            match body {
                Body::IntraGrey => write_dc_only_intra_mb(w),
                Body::Skipped | Body::Direct => w.write_bits(1, 1),
                Body::IntraBrighter => {
                    w.write_bits(0, 1); // not_coded
                    w.write_bits(0b0_0011, 5); // mcbpc: intra, cbpc 00
                    w.write_bits(0, 1); // ac_pred_flag
                    w.write_bits(0b0011, 4); // cbpy intra 0
                    w.write_bits(0b11, 2); // luma DC size 1
                    w.write_bits(1, 1); // differential +1
                    for _ in 0..3 {
                        w.write_bits(0b011, 3); // luma DC size 0
                    }
                    for _ in 0..2 {
                        w.write_bits(0b11, 2); // chroma DC size 0
                    }
                }
            }
        }
        w.align();
    }

    fn decode_all(stream: &[u8]) -> Result<Vec<DecodedFrame>, StreamDecodeError> {
        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(stream)?;
        frames.extend(dec.flush());
        Ok(frames)
    }

    #[test]
    fn complexity_estimation_statistics_do_not_change_the_pictures() {
        // The same I / P(intra) / B / P VOPs with and without the
        // complexity-estimation blocks in the VOL and every VOP header.
        let build = |complexity: bool| {
            let vol = VolExtras {
                complexity,
                ..VolExtras::default()
            };
            let x = VopExtras {
                vol,
                ..VopExtras::default()
            };
            let mut w = BitWriter::default();
            write_vos(&mut w);
            write_extended_vol(&mut w, 32, 16, vol);
            write_extended_vop(&mut w, Body::IntraGrey, 2, 0, x);
            write_extended_vop(&mut w, Body::IntraBrighter, 2, 2, x);
            write_extended_vop(&mut w, Body::Direct, 2, 1, x);
            write_extended_vop(&mut w, Body::Skipped, 2, 3, x);
            w.finish()
        };
        let plain = decode_all(&build(false)).unwrap();
        let with_statistics = decode_all(&build(true)).unwrap();
        assert_eq!(plain.len(), 4);
        assert_eq!(plain, with_statistics);
        // The pictures are what the macroblock layers say.
        assert_eq!(plain[0].luma_at(5, 5), Some(128));
        assert_eq!(plain[2].luma_at(5, 5), Some(130));
        assert_eq!(plain[3].luma_at(31, 15), Some(132));

        let mut dec = Mpeg4VideoDecoder::new();
        dec.decode(&build(true)).unwrap();
        let ce = dec.vol().unwrap().complexity_estimation.unwrap();
        assert!(ce.intra_blocks && ce.inter_blocks && ce.vlc_bits);
        assert!(ce.interpolate_mc_q && ce.quarterpel && !ce.sadct);
    }

    #[test]
    fn newpred_selects_the_named_reference_vop() {
        let vol = VolExtras {
            newpred_segment_is_vop: Some(true),
            ..VolExtras::default()
        };
        let ids = |id: u32, reference: Option<u32>| VopExtras {
            vol,
            ids: Some((id, reference)),
            ..VopExtras::default()
        };
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 16, 16, vol);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, ids(1, None)); // 128
        write_extended_vop(&mut w, Body::IntraBrighter, 1, 1, ids(2, None)); // 130
        // Copies of their reference: the I-VOP, then the brighter P.
        write_extended_vop(&mut w, Body::Skipped, 1, 2, ids(3, Some(1)));
        write_extended_vop(&mut w, Body::Skipped, 1, 3, ids(4, Some(2)));
        // No selection: the previous VOP.
        write_extended_vop(&mut w, Body::Skipped, 1, 4, ids(5, None));
        // The previous VOP named explicitly, then one two back.
        write_extended_vop(&mut w, Body::Skipped, 1, 5, ids(6, Some(5)));
        write_extended_vop(&mut w, Body::Skipped, 1, 6, ids(7, Some(3)));
        let frames = decode_all(&w.finish()).unwrap();
        let luma: Vec<u8> = frames.iter().map(|f| f.luma_at(7, 9).unwrap()).collect();
        assert_eq!(luma, [128, 130, 128, 130, 130, 130, 128]);

        let mut dec = Mpeg4VideoDecoder::new();
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 16, 16, vol);
        dec.decode(&w.finish()).unwrap();
        let config = dec.vol().unwrap().newpred.unwrap();
        assert!(config.segment_is_vop);
        assert_eq!(config.requested_upstream_message_type, 1);
    }

    #[test]
    fn newpred_b_vops_keep_their_display_order_anchors() {
        // I(1) P(2, brighter) then P(3) predicted from the I-VOP, with
        // a B-VOP between P(2) and P(3). Its co-located macroblock in
        // P(3) is not_coded, so it is the copy of its forward anchor
        // (§6.3.6 co_located_not_coded) — P(2), the previous anchor in
        // display order, not the reference P(3) selected.
        let vol = VolExtras {
            newpred_segment_is_vop: Some(true),
            ..VolExtras::default()
        };
        let ids = |id: u32, reference: Option<u32>| VopExtras {
            vol,
            ids: Some((id, reference)),
            ..VopExtras::default()
        };
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 16, 16, vol);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, ids(1, None));
        write_extended_vop(&mut w, Body::IntraBrighter, 1, 1, ids(2, None));
        write_extended_vop(&mut w, Body::Skipped, 1, 3, ids(3, Some(1)));
        write_extended_vop(&mut w, Body::Direct, 1, 2, ids(4, None));
        let frames = decode_all(&w.finish()).unwrap();
        let luma: Vec<u8> = frames.iter().map(|f| f.luma_at(3, 3).unwrap()).collect();
        // Display order: I, P(2), B, P(3).
        assert_eq!(luma, [128, 130, 130, 128]);
        assert_eq!(frames[2].coding_type(), VopCodingType::B);
    }

    #[test]
    fn newpred_references_outside_the_memory_are_reported() {
        let vol = VolExtras {
            newpred_segment_is_vop: Some(true),
            ..VolExtras::default()
        };
        let ids = |id: u32, reference: Option<u32>| VopExtras {
            vol,
            ids: Some((id, reference)),
            ..VopExtras::default()
        };
        // A vop_id that was never decoded.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 16, 16, vol);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, ids(1, None));
        write_extended_vop(&mut w, Body::Skipped, 1, 1, ids(2, Some(77)));
        assert_eq!(
            decode_all(&w.finish()).unwrap_err(),
            StreamDecodeError::MissingNewpredReference { vop_id: 77 }
        );
        // §E.1.6.4: nothing decoded before the last I-VOP is referred to.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 16, 16, vol);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, ids(1, None));
        write_extended_vop(&mut w, Body::IntraBrighter, 1, 1, ids(2, None));
        write_extended_vop(&mut w, Body::IntraGrey, 1, 2, ids(3, None));
        write_extended_vop(&mut w, Body::Skipped, 1, 3, ids(4, Some(2)));
        assert_eq!(
            decode_all(&w.finish()).unwrap_err(),
            StreamDecodeError::MissingNewpredReference { vop_id: 2 }
        );
        // A selection older than the memory holds.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 16, 16, vol);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, ids(1, None));
        for k in 0..NEWPRED_MEMORY_VOPS as u32 {
            write_extended_vop(&mut w, Body::Skipped, 1, (k + 1) % 30, ids(k + 2, None));
        }
        let within = {
            let mut w2 = BitWriter::default();
            write_extended_vop(&mut w2, Body::Skipped, 1, 20, ids(200, Some(2)));
            w2.finish()
        };
        let beyond = {
            let mut w2 = BitWriter::default();
            write_extended_vop(&mut w2, Body::Skipped, 1, 20, ids(200, Some(1)));
            w2.finish()
        };
        let head = w.finish();
        assert!(decode_all(&[head.clone(), within].concat()).is_ok());
        assert_eq!(
            decode_all(&[head, beyond].concat()).unwrap_err(),
            StreamDecodeError::MissingNewpredReference { vop_id: 1 }
        );
    }

    #[test]
    fn newpred_forms_outside_the_defined_decode_are_named() {
        // Per-packet reference selection.
        let vol = VolExtras {
            newpred_segment_is_vop: Some(false),
            ..VolExtras::default()
        };
        let x = VopExtras {
            vol,
            ids: Some((1, None)),
            ..VopExtras::default()
        };
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 16, 16, vol);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, x);
        let err = decode_all(&w.finish()).unwrap_err();
        assert!(matches!(
            err,
            StreamDecodeError::Vop(VopParseError::UnsupportedBranch(what)) if what.contains("Video Packet")
        ));
        // A reference selection on a B-VOP.
        let vol = VolExtras {
            newpred_segment_is_vop: Some(true),
            ..VolExtras::default()
        };
        let ids = |id: u32, reference: Option<u32>| VopExtras {
            vol,
            ids: Some((id, reference)),
            ..VopExtras::default()
        };
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 16, 16, vol);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, ids(1, None));
        write_extended_vop(&mut w, Body::Skipped, 1, 2, ids(2, None));
        write_extended_vop(&mut w, Body::Direct, 1, 1, ids(3, Some(1)));
        let err = decode_all(&w.finish()).unwrap_err();
        assert!(matches!(
            err,
            StreamDecodeError::Vop(VopParseError::UnsupportedBranch(what)) if what.contains("B-VOP")
        ));
    }

    #[test]
    fn reduced_resolution_vol_decodes_its_full_resolution_vops() {
        let vol = VolExtras {
            reduced_resolution: true,
            ..VolExtras::default()
        };
        let x = VopExtras {
            vol,
            ..VopExtras::default()
        };
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 32, 16, vol);
        write_extended_vop(&mut w, Body::IntraGrey, 2, 0, x);
        write_extended_vop(&mut w, Body::IntraBrighter, 2, 2, x);
        write_extended_vop(&mut w, Body::Direct, 2, 1, x);
        let frames = decode_all(&w.finish()).unwrap();
        let luma: Vec<u8> = frames.iter().map(|f| f.luma_at(20, 4).unwrap()).collect();
        // Display order I, B, P; the second macroblock of the P-VOP is
        // 132 and the direct B-VOP averages its anchors.
        assert_eq!(luma, [128, 130, 132]);

        // A VOP that sets vop_reduced_resolution is named, not
        // mis-decoded at full resolution.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_extended_vol(&mut w, 32, 16, vol);
        write_extended_vop(
            &mut w,
            Body::IntraGrey,
            2,
            0,
            VopExtras { reduced: true, ..x },
        );
        let err = decode_all(&w.finish()).unwrap_err();
        assert!(matches!(
            err,
            StreamDecodeError::Vop(VopParseError::UnsupportedBranch(what)) if what.contains("7.6.10")
        ));
    }

    // ---- several layers in one stream ------------------------------

    /// [`write_vol`] with a `video_object_layer_id` and the
    /// `scalability` flag.
    fn write_vol_of_layer(w: &mut BitWriter, id: u32, width: u32, height: u32, scalable: bool) {
        w.write_bits(0x0000_0120 + id, 32);
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
        w.write_bits(u32::from(scalable), 1); // scalability
        w.align();
    }

    #[test]
    fn each_layer_keeps_its_own_reference_chain() {
        let plain = VopExtras::default();
        let mut w = BitWriter::default();
        write_vos(&mut w);
        // Layer 0: 16×16, a grey I-VOP.
        write_vol_of_layer(&mut w, 0, 16, 16, false);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, plain);
        // Layer 1: 32×16, an I-VOP and a brighter P-VOP.
        write_vol_of_layer(&mut w, 1, 32, 16, false);
        write_extended_vop(&mut w, Body::IntraGrey, 2, 0, plain);
        write_extended_vop(&mut w, Body::IntraBrighter, 2, 1, plain);
        // Back to layer 0: a P-VOP that copies *its* I-VOP.
        write_vol_of_layer(&mut w, 0, 16, 16, false);
        write_extended_vop(&mut w, Body::Skipped, 1, 1, plain);
        // And to layer 1: a copy of its brighter P-VOP.
        write_vol_of_layer(&mut w, 1, 32, 16, false);
        write_extended_vop(&mut w, Body::Skipped, 2, 2, plain);
        let stream = w.finish();

        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        assert_eq!(dec.current_layer(), Some((0, 1)));
        assert_eq!(dec.vol().unwrap().width, 32);
        frames.extend(dec.flush());
        let of = |layer: (u8, u8)| -> Vec<&DecodedFrame> {
            frames.iter().filter(|f| f.layer() == layer).collect()
        };
        let layer0 = of((0, 0));
        let layer1 = of((0, 1));
        assert_eq!(layer0.len(), 2);
        assert_eq!(layer1.len(), 3);
        assert!(layer0.iter().all(|f| f.width() == 16));
        assert!(layer1.iter().all(|f| f.width() == 32));
        let luma = |fs: &[&DecodedFrame]| -> Vec<u8> {
            fs.iter().map(|f| f.luma_at(2, 2).unwrap()).collect()
        };
        assert_eq!(luma(&layer0), [128, 128]);
        assert_eq!(luma(&layer1), [128, 130, 130]);
        // Each layer runs its own clock.
        let ticks = |fs: &[&DecodedFrame]| -> Vec<i64> {
            fs.iter().map(|f| f.pts_ticks().unwrap()).collect()
        };
        assert_eq!(ticks(&layer0), [0, 1]);
        assert_eq!(ticks(&layer1), [0, 1, 2]);
    }

    #[test]
    fn video_objects_are_separate_layers_too() {
        let plain = VopExtras::default();
        let mut w = BitWriter::default();
        write_vos(&mut w);
        w.write_bits(VIDEO_OBJECT_START_CODE_MIN, 32);
        write_vol_of_layer(&mut w, 0, 16, 16, false);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, plain);
        w.write_bits(VIDEO_OBJECT_START_CODE_MIN + 3, 32);
        write_vol_of_layer(&mut w, 0, 16, 16, false);
        // No anchor yet in object 3's layer 0.
        write_extended_vop(&mut w, Body::Skipped, 1, 1, plain);
        let stream = w.finish();
        let mut dec = Mpeg4VideoDecoder::new();
        assert_eq!(
            dec.decode(&stream).unwrap_err(),
            StreamDecodeError::MissingAnchor
        );
        assert_eq!(dec.current_layer(), Some((3, 0)));
        // The first object's picture is still released.
        let frames = dec.flush();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].layer(), (0, 0));
    }

    #[test]
    fn enhancement_layer_vops_are_stepped_over() {
        let plain = VopExtras::default();
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol_of_layer(&mut w, 0, 16, 16, false);
        write_extended_vop(&mut w, Body::IntraGrey, 1, 0, plain);
        // An enhancement layer with two VOPs of its own syntax.
        write_vol_of_layer(&mut w, 1, 16, 16, true);
        for _ in 0..2 {
            w.write_bits(0x0000_01B6, 32);
            w.write_bits(0xDEAD_BEEF, 32);
            w.align();
        }
        write_vol_of_layer(&mut w, 0, 16, 16, false);
        write_extended_vop(&mut w, Body::IntraBrighter, 1, 1, plain);
        let stream = w.finish();
        let mut dec = Mpeg4VideoDecoder::new();
        let mut frames = dec.decode(&stream).unwrap();
        frames.extend(dec.flush());
        assert_eq!(dec.enhancement_vops_skipped(), 2);
        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(|f| f.layer() == (0, 0)));
        assert_eq!(frames[1].luma_at(0, 0), Some(130));
    }

    #[test]
    fn a_stream_visiting_many_layers_keeps_a_bounded_number_of_them() {
        let plain = VopExtras::default();
        let layers = MAX_PARKED_LAYERS as u32 + 3;
        let mut w = BitWriter::default();
        write_vos(&mut w);
        for id in 0..layers {
            write_vol_of_layer(&mut w, id, 16, 16, false);
            write_extended_vop(&mut w, Body::IntraGrey, 1, 0, plain);
        }
        // Layer 0 was visited first: its state is gone, so a P-VOP
        // there has no anchor; the most recent layers still have theirs.
        let mut recent = w.clone();
        write_vol_of_layer(&mut recent, layers - 2, 16, 16, false);
        write_extended_vop(&mut recent, Body::Skipped, 1, 1, plain);
        assert!(decode_all(&recent.finish()).is_ok());
        write_vol_of_layer(&mut w, 0, 16, 16, false);
        write_extended_vop(&mut w, Body::Skipped, 1, 1, plain);
        assert_eq!(
            decode_all(&w.finish()).unwrap_err(),
            StreamDecodeError::MissingAnchor
        );
    }
}

#[cfg(test)]
mod registry_tests {
    use super::tests::{BitWriter, write_i_vop, write_skipped_p_vop, write_vol, write_vos};
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
    fn output_pixel_format_is_stated_before_the_first_frame() {
        // Without extradata (configuration in-band, as in AVI) and with
        // it (MP4 decoder-specific info), the layout is answered before
        // any packet, and the frames that follow have that shape.
        let mut w = BitWriter::default();
        write_vos(&mut w);
        write_vol(&mut w, 48, 32);
        let config = w.finish();
        for extradata in [Vec::new(), config.clone()] {
            let mut params = CodecParameters::video(CodecId::new("mpeg4video"));
            params.extradata = extradata.clone();
            let mut dec = make_decoder(&params).unwrap();
            assert_eq!(
                dec.output_pixel_format(),
                Some(oxideav_core::PixelFormat::Yuv420P)
            );
            if extradata.is_empty() {
                dec.send_packet(&packet(config.clone())).unwrap();
            }
            let mut w = BitWriter::default();
            write_i_vop(&mut w, 6, 0); // 3×2 macroblocks
            dec.send_packet(&packet(w.finish())).unwrap();
            dec.flush().unwrap();
            let oxideav_core::Frame::Video(v) = dec.receive_frame().unwrap() else {
                panic!("expected a video frame");
            };
            let layout: Vec<_> = v.planes.iter().map(|p| (p.stride, p.data.len())).collect();
            assert_eq!(layout, vec![(48, 48 * 32), (24, 24 * 16), (24, 24 * 16)]);
            assert_eq!(
                dec.output_pixel_format(),
                Some(oxideav_core::PixelFormat::Yuv420P)
            );
            dec.reset().unwrap();
            assert_eq!(
                dec.output_pixel_format(),
                Some(oxideav_core::PixelFormat::Yuv420P)
            );
        }
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
    fn s_gmc_colocated_maps_to_the_averaged_motion_vector() {
        // §7.6.9.6: a skipped co-located S(GMC) macroblock is treated as
        // a NON-skipped macroblock carrying the §7.8.7.3 averaged motion
        // vector; §7.6.9.5.1: a coded GMC (mcsel == 1) macroblock
        // supplies the same averaged motion vector; an intra one keeps
        // the zero-vector fallback.
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
        assert_eq!(motion[1], AnchorMbMotion::Frame(PvopMbMotion::OneMv(amv)));
        assert_eq!(
            motion[2],
            AnchorMbMotion::Frame(PvopMbMotion::OneMv(MotionVector { x: 1, y: 2 }))
        );
        assert_eq!(motion[3], AnchorMbMotion::Frame(PvopMbMotion::Intra));
    }
}

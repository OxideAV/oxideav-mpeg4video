//! §6.2.5 `video_packet_header` decode + the §5.2.5 / §6.3.3 helpers.
//!
//! The MPEG-4 Visual bitstream interleaves zero or more *video packets*
//! inside a VOP. Each packet starts with a `resync_marker` (§6.3.3) —
//! a variable-length zero string terminated by a `1`, byte-aligned —
//! followed by `video_packet_header()` (§6.2.5, the syntax table that
//! lives just below `video_object_plane()`). The header re-states the
//! macroblock number where decoding resumes, optionally re-states the
//! VOP-level fields the decoder needs after a packet loss
//! (`vop_coding_type`, the fcodes, `intra_dc_vlc_thr`, …) when
//! `header_extension_code == 1`, and lets the rest of the VOP decode
//! independently.
//!
//! Round 18 of the clean-room rebuild covers the rectangular-shape
//! path:
//!
//! * [`macroblock_number_bit_width`] — Table 6-27 mapping from a VOP's
//!   total-macroblock count to the bit length of the `macroblock_number`
//!   variable-length code (1..=14 bits).
//! * [`resync_marker_length`] — §6.3.3 `resync_marker` length formula:
//!   17 bits for I-VOPs and binary-only shape, `15 + fcode` for P /
//!   S(GMC), `max(15 + max(fcode_fwd, fcode_bwd), 17)` zeros for B.
//! * [`consume_next_resync_marker`] — §5.2.5 `next_resync_marker()`:
//!   one `0` bit then a run of `1`s up to the next byte boundary,
//!   which the encoder writes immediately before the resync marker so
//!   the marker itself is byte-aligned.
//! * [`probe_resync_marker`] — non-destructive peek that asks "are the
//!   next bits (after byte alignment) a `resync_marker` of the
//!   expected length?".
//! * [`parse_video_packet_header`] — the §6.2.5 syntax table for the
//!   rectangular, non-binary-only shape path. Reads
//!   `macroblock_number`, `quant_scale`, `header_extension_code`, and
//!   — when the extension bit is set — re-states `modulo_time_base`,
//!   `vop_time_increment`, `vop_coding_type`, `intra_dc_vlc_thr`, and
//!   the per-coding-type `vop_fcode_forward` / `vop_fcode_backward`.
//!
//! Non-rectangular and binary-only shape branches are deliberately out
//! of scope: the shape-extension body (`vop_width` / `vop_height` /
//! `vop_horizontal_mc_spatial_ref` / `vop_vertical_mc_spatial_ref`)
//! introduces another four 13-bit fields plus marker bits, plus the
//! `change_conv_ratio_disable` / `vop_shape_coding_type` rectangular
//! shape gates inside the extension body. Sprite trajectory, newpred,
//! and reduced-resolution VOP bodies are also out of scope; this round
//! returns [`VideoPacketParseError::UnsupportedBranch`] for them, the
//! same way the §6.2.5 VOP header rejects them.
//!
//! Provenance: the syntax tables and the semantics are sourced from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`,
//! specifically §5.2.5 `next_resync_marker()`, §6.2.5 the
//! `video_packet_header()` syntax table, and §6.3.3 (the long
//! semantics list whose entries cover `resync_marker`,
//! `macroblock_number`, `quant_scale`, and `header_extension_code`).

use crate::bitreader::{BitReader, BitReaderError};
use crate::vop::VopCodingType;

/// Errors produced by the `video_packet_header` parsers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoPacketParseError {
    /// The supplied byte slice ran out before the parser finished
    /// reading a field. Includes the §5.2.5 stuffing run, the resync
    /// marker itself, the `macroblock_number` VLC, the quant scale, or
    /// any of the extension-body fields.
    Truncated,
    /// `resync_marker_disable == true` (the §6.2.3 VOL flag). The
    /// caller must not invoke `parse_video_packet_header` in that
    /// VOL configuration; the bitstream cannot legally carry a video
    /// packet header.
    ResyncDisabled,
    /// The bits at the (byte-aligned) reader position were not the
    /// expected resync marker. `expected_bits` is the number of bits
    /// (between 17 and 23) the §6.3.3 formula predicted for the
    /// current VOP coding type and fcode.
    MissingResyncMarker {
        /// Number of bits the §6.3.3 formula predicts for the current
        /// VOP.
        expected_bits: u8,
        /// Actual bits observed at the byte-aligned position.
        observed: u32,
    },
    /// `macroblock_number` exceeded the total-macroblock count given
    /// to the parser. The spec mandates `macroblock_number` is "the
    /// number within a VOP", strictly less than the total.
    MacroblockNumberOutOfRange {
        /// Decoded value.
        value: u32,
        /// Total macroblocks the parser was told the VOP has.
        total: u32,
    },
    /// `quant_scale` was transmitted as `0`. §6.3.3 calls it "an
    /// unsigned integer which specifies the absolute value of the
    /// quantiser scale"; `0` is invalid as a quantiser scale and is
    /// rejected to prevent divide-by-zero downstream.
    ForbiddenQuantScale,
    /// `vop_fcode_forward` / `vop_fcode_backward` was transmitted as
    /// `0`. The §6.3.5 `vop_fcode_*` semantics forbid that value.
    ForbiddenFcode,
    /// `quant_precision` outside the §6.3.3 `[3, 9]` range.
    BadQuantPrecision(u8),
    /// A branch the round-18 parser deliberately rejects (binary-only
    /// shape, non-rectangular shape, sprite trajectory, newpred,
    /// reduced-resolution VOP).
    UnsupportedBranch(&'static str),
}

impl core::fmt::Display for VideoPacketParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VideoPacketParseError::Truncated => write!(f, "video_packet_header truncated"),
            VideoPacketParseError::ResyncDisabled => {
                write!(f, "resync_marker_disable is set; no video packets allowed")
            }
            VideoPacketParseError::MissingResyncMarker {
                expected_bits,
                observed,
            } => write!(
                f,
                "expected resync_marker ({expected_bits} bits), found 0x{observed:08X}"
            ),
            VideoPacketParseError::MacroblockNumberOutOfRange { value, total } => {
                write!(f, "macroblock_number {value} >= total {total}")
            }
            VideoPacketParseError::ForbiddenQuantScale => {
                write!(f, "quant_scale of 0 is forbidden")
            }
            VideoPacketParseError::ForbiddenFcode => {
                write!(f, "vop_fcode_forward/backward of 0 is forbidden")
            }
            VideoPacketParseError::BadQuantPrecision(p) => {
                write!(f, "quant_precision {p} is outside the allowed 3..=9 range")
            }
            VideoPacketParseError::UnsupportedBranch(name) => {
                write!(f, "video_packet_header branch '{name}' not supported")
            }
        }
    }
}

impl std::error::Error for VideoPacketParseError {}

impl From<BitReaderError> for VideoPacketParseError {
    fn from(_: BitReaderError) -> Self {
        VideoPacketParseError::Truncated
    }
}

/// Per-VOP context the `video_packet_header` parser needs from the
/// surrounding VOP. The fields are exactly those §6.2.5 reads from
/// the enclosing VOP header (or from the previous video packet) before
/// invoking `video_packet_header()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoPacketContext {
    /// Current `vop_coding_type` (per the most recent VOP header or
    /// preceding `video_packet_header()` extension body).
    pub coding_type: VopCodingType,
    /// Current `vop_fcode_forward` (1..=7 for P / B / S(GMC); 0 for
    /// I-VOPs since the field is not transmitted there).
    pub fcode_fwd: u8,
    /// Current `vop_fcode_backward` (1..=7 for B; 0 otherwise).
    pub fcode_bwd: u8,
    /// `quant_precision` from the VOL header (default 5).
    pub quant_precision: u8,
    /// `vop_time_increment_resolution` from the VOL header. Used to
    /// determine the bit-width of `vop_time_increment` when the
    /// extension body re-states it.
    pub time_increment_resolution: u16,
    /// `video_object_layer_shape` from the VOL header. Only value `0`
    /// (rectangular) is supported by the round-18 parser.
    pub video_object_layer_shape: u8,
    /// `resync_marker_disable` from the VOL header. When `true`, no
    /// video packets may appear; the parser refuses to run.
    pub resync_marker_disable: bool,
    /// `newpred_enable` from the VOL header. When `true`, the parser
    /// refuses to handle the extension body (the spec's
    /// `if (newpred_enable)` `vop_id` / `vop_id_for_prediction` block
    /// is out of round-18 scope).
    pub newpred_enable: bool,
    /// `reduced_resolution_vop_enable` from the VOL header. When
    /// `true`, the parser refuses the extension body (the
    /// `vop_reduced_resolution` bit is out of scope).
    pub reduced_resolution_vop_enable: bool,
    /// Whether the enclosing VOL declares `sprite_enable ==
    /// GMC`. When `true` and `vop_coding_type == S`, the extension
    /// body would also carry `sprite_trajectory()` — out of round-18
    /// scope.
    pub sprite_gmc: bool,
    /// `((video_object_layer_width + 15) / 16) *
    /// ((video_object_layer_height + 15) / 16)`. Total macroblocks
    /// the parser uses to (a) pick the Table 6-27 bit width for
    /// `macroblock_number` and (b) validate the decoded number is
    /// in-range.
    pub total_macroblocks: u32,
}

/// Decoded §6.2.5 `video_packet_header`.
///
/// All optional fields default to `None` when `header_extension_code
/// == 0`. The caller composes the new `(modulo_time_base,
/// vop_time_increment, vop_coding_type, intra_dc_vlc_thr, fcode_*)`
/// state from the extension fields when present; otherwise it
/// inherits the values from the enclosing VOP header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoPacketHeader {
    /// `macroblock_number` — the macroblock index where decoding
    /// resumes. Always present.
    pub macroblock_number: u32,
    /// `quant_scale` — the absolute quantiser scale that applies to
    /// the first macroblock of the packet (and to every subsequent
    /// macroblock until `dquant` updates it).
    pub quant_scale: u16,
    /// `header_extension_code` — when `true`, the extension body
    /// below is populated; otherwise every following field is `None`.
    pub header_extension_code: bool,
    /// Re-stated `modulo_time_base` (number of leading `1`s in the
    /// `do { … } while (modulo_time_base != '0')` loop).
    pub modulo_time_base: Option<u32>,
    /// Re-stated `vop_time_increment` (bit width per
    /// [`crate::vop::vop_time_increment_bits`]).
    pub vop_time_increment: Option<u16>,
    /// Re-stated `vop_coding_type`.
    pub vop_coding_type: Option<VopCodingType>,
    /// Re-stated `intra_dc_vlc_thr` (Table 6-25 index, 0..=7).
    pub intra_dc_vlc_thr: Option<u8>,
    /// Re-stated `vop_fcode_forward` (1..=7; absent for I-VOPs).
    pub vop_fcode_forward: Option<u8>,
    /// Re-stated `vop_fcode_backward` (1..=7; absent unless
    /// `vop_coding_type == B`).
    pub vop_fcode_backward: Option<u8>,
}

/// Number of bits used to encode `macroblock_number` per Table 6-27.
///
/// The table maps `total_macroblocks` (the product `((width + 15) /
/// 16) * ((height + 15) / 16)`) to the bit-length of the fixed-length
/// `macroblock_number` code:
///
/// | total          | bits |
/// | -------------- | ---- |
/// | 1..=2          | 1    |
/// | 3..=4          | 2    |
/// | 5..=8          | 3    |
/// | 9..=16         | 4    |
/// | 17..=32        | 5    |
/// | 33..=64        | 6    |
/// | 65..=128       | 7    |
/// | 129..=256      | 8    |
/// | 257..=512      | 9    |
/// | 513..=1024     | 10   |
/// | 1025..=2048    | 11   |
/// | 2049..=4096    | 12   |
/// | 4097..=8192    | 13   |
/// | 8193..=16384   | 14   |
///
/// Mathematically this is `ceil(log2(total_macroblocks))`, with the
/// minimum of 1 bit for `total <= 2` (the §6.2.5 `macroblock_number`
/// is "1-14 bits", never 0). A `total` of `0` is invalid; the
/// function returns `1` for that degenerate input so callers can
/// proceed without panicking and the malformed bitstream will fail
/// later through `MacroblockNumberOutOfRange`.
pub fn macroblock_number_bit_width(total_macroblocks: u32) -> u8 {
    if total_macroblocks <= 2 {
        return 1;
    }
    // bits = ceil(log2(N)). For N=3 -> 2, N=4 -> 2, N=5 -> 3.
    32 - (total_macroblocks - 1).leading_zeros() as u8
}

/// `((width + 15) / 16) * ((height + 15) / 16)` — Table 6-27's input
/// expression in plain integer arithmetic.
pub fn total_macroblocks(video_object_layer_width: u32, video_object_layer_height: u32) -> u32 {
    let mb_cols = video_object_layer_width.div_ceil(16);
    let mb_rows = video_object_layer_height.div_ceil(16);
    mb_cols.saturating_mul(mb_rows)
}

/// Length of `resync_marker` in bits per §6.3.3:
///
/// * I-VOP: 17 bits (16 zeros + a one).
/// * P-VOP / S(GMC)-VOP: `15 + vop_fcode_forward` zeros followed by a
///   one, i.e. `16 + vop_fcode_forward` bits total.
/// * B-VOP: `max(15 + max(fcode_fwd, fcode_bwd), 17)` **zeros**
///   followed by a one — the §6.3.3 `max` applies to the zero count,
///   so the shortest B marker is 17 zeros + 1 = 18 bits (fcode 1 and
///   fcode 2 both floor at 17 zeros).
///
/// Returns the total marker length in bits (the leading-zeros count
/// plus the trailing `1`). The minimum is 17 and the maximum is `15 +
/// 7 + 1 = 23` bits (`fcode = 7`).
pub fn resync_marker_length(coding_type: VopCodingType, fcode_fwd: u8, fcode_bwd: u8) -> u8 {
    match coding_type {
        VopCodingType::I => 17,
        VopCodingType::P | VopCodingType::S => {
            // `15 + fcode` zeros + 1 → 16 + fcode bits total.
            // Clamp the unrealistic case `fcode == 0` to the I-VOP
            // length so we never compute a marker shorter than 17.
            let fcode = fcode_fwd.max(1) as u32;
            (15 + fcode + 1) as u8
        }
        VopCodingType::B => {
            let fcode = fcode_fwd.max(fcode_bwd).max(1) as u32;
            let zeros = (15 + fcode).max(17); // §6.3.3: max over the zeros
            (zeros + 1) as u8
        }
    }
}

/// §5.2.5 `next_resync_marker()`: consume one `0` bit, then a run of
/// `1` bits up to (but not into) the next byte boundary. The §5.2.5
/// pseudo-code is identical to `next_start_code()` (§5.2.4) modulo
/// the diagnostic name.
///
/// On success the reader is left at a byte boundary, positioned at
/// the first bit of the resync marker itself. Returns
/// [`VideoPacketParseError::Truncated`] on underflow and
/// [`VideoPacketParseError::MissingResyncMarker`] (with
/// `expected_bits = 0`, since we cannot peek a meaningful marker yet)
/// if the first stuffing bit is not `0`.
pub fn consume_next_resync_marker(br: &mut BitReader<'_>) -> Result<(), VideoPacketParseError> {
    // The first stuffing bit must be `0`. `next_start_code()` and
    // `next_resync_marker()` share the convention.
    let zero = br.read_bool()?;
    if zero {
        return Err(VideoPacketParseError::MissingResyncMarker {
            expected_bits: 0,
            observed: 1,
        });
    }
    while !br.is_byte_aligned() {
        // Stuffing 1 bits until byte alignment.
        let one = br.read_bool()?;
        if !one {
            return Err(VideoPacketParseError::MissingResyncMarker {
                expected_bits: 0,
                observed: 0,
            });
        }
    }
    Ok(())
}

/// Non-destructively peek at the next bits and return whether they
/// form the `resync_marker` predicted by [`resync_marker_length`].
///
/// The reader must already be byte-aligned — §6.3.3 mandates "A
/// resync marker shall only be located immediately before a
/// macroblock and aligned with a byte." Callers chain
/// [`consume_next_resync_marker`] before this peek if the encoder
/// inserted stuffing.
pub fn probe_resync_marker(
    br: &BitReader<'_>,
    coding_type: VopCodingType,
    fcode_fwd: u8,
    fcode_bwd: u8,
) -> bool {
    if !br.is_byte_aligned() {
        return false;
    }
    let n = resync_marker_length(coding_type, fcode_fwd, fcode_bwd) as usize;
    if br.remaining_bits() < n {
        return false;
    }
    let bits = match br.next_bits(n) {
        Ok(v) => v,
        Err(_) => return false,
    };
    // The marker is `(n-1)` zeros followed by `1`. Equivalent: the
    // value as a binary integer is exactly `1`.
    bits == 1
}

/// Read a `resync_marker` from the (byte-aligned) reader position.
///
/// Returns [`VideoPacketParseError::MissingResyncMarker`] if the bits
/// do not match the §6.3.3 pattern. On success the reader is
/// positioned immediately after the trailing `1`.
fn read_resync_marker(
    br: &mut BitReader<'_>,
    coding_type: VopCodingType,
    fcode_fwd: u8,
    fcode_bwd: u8,
) -> Result<(), VideoPacketParseError> {
    if !br.is_byte_aligned() {
        return Err(VideoPacketParseError::MissingResyncMarker {
            expected_bits: resync_marker_length(coding_type, fcode_fwd, fcode_bwd),
            observed: 0,
        });
    }
    let n = resync_marker_length(coding_type, fcode_fwd, fcode_bwd);
    let bits = br.read_bits(n as usize)?;
    if bits != 1 {
        return Err(VideoPacketParseError::MissingResyncMarker {
            expected_bits: n,
            observed: bits,
        });
    }
    Ok(())
}

/// Read `modulo_time_base`: leading `1` bits up to a terminating
/// `0`. Identical to the §6.2.5 VOP-header loop, surfaced here so the
/// `video_packet_header` extension body re-uses the same code path.
fn read_modulo_time_base(br: &mut BitReader<'_>) -> Result<u32, VideoPacketParseError> {
    let mut count: u32 = 0;
    loop {
        let bit = br.read_bool()?;
        if !bit {
            break;
        }
        count = count
            .checked_add(1)
            .ok_or(VideoPacketParseError::Truncated)?;
    }
    Ok(count)
}

fn read_marker(br: &mut BitReader<'_>) -> Result<(), VideoPacketParseError> {
    if br.read_bool()? {
        Ok(())
    } else {
        // Use Truncated as the closest existing variant for a bad
        // marker bit; we don't add a separate variant to keep the
        // public error surface narrow.
        Err(VideoPacketParseError::Truncated)
    }
}

/// Parse one §6.2.5 `video_packet_header` from the (byte-aligned)
/// `BitReader` position.
///
/// The reader must already be positioned just before the
/// `resync_marker`. If the caller is unsure whether the stream is
/// stuffed or aligned, run [`consume_next_resync_marker`] first to
/// consume the §5.2.5 stuffing — the spec wrapper around every
/// `video_packet_header()` invocation calls `next_resync_marker()`
/// inside the syntax table itself, so this function does too.
///
/// Round 18 supports only the rectangular non-binary-only path, the
/// same gating as the VOP-header parser. Sprite-GMC trajectory
/// bodies, newpred, and reduced-resolution VOP are rejected with
/// [`VideoPacketParseError::UnsupportedBranch`]. Non-rectangular
/// shape returns `UnsupportedBranch("non-rectangular shape")`.
pub fn parse_video_packet_header(
    br: &mut BitReader<'_>,
    ctx: &VideoPacketContext,
) -> Result<VideoPacketHeader, VideoPacketParseError> {
    if ctx.resync_marker_disable {
        return Err(VideoPacketParseError::ResyncDisabled);
    }
    if ctx.video_object_layer_shape != 0 {
        // Binary-only or grayscale: out of round-18 scope.
        return Err(VideoPacketParseError::UnsupportedBranch(
            "non-rectangular shape",
        ));
    }
    if !(3..=9).contains(&ctx.quant_precision) {
        return Err(VideoPacketParseError::BadQuantPrecision(
            ctx.quant_precision,
        ));
    }

    // 1. §5.2.5 next_resync_marker() — consume the `0 1*` stuffing run
    //    so the marker itself sits on a byte boundary.
    consume_next_resync_marker(br)?;

    // 2. resync_marker (17..=23 bits, byte-aligned).
    read_resync_marker(br, ctx.coding_type, ctx.fcode_fwd, ctx.fcode_bwd)?;

    // 3. If video_object_layer_shape != rectangular, the spec walks the
    //    shape-extension body here. Round-18 scope rejects that above,
    //    so we skip straight to macroblock_number.

    // 4. macroblock_number: fixed-length per Table 6-27.
    let mb_bits = macroblock_number_bit_width(ctx.total_macroblocks);
    let macroblock_number = br.read_bits(mb_bits as usize)?;
    if ctx.total_macroblocks > 0 && macroblock_number >= ctx.total_macroblocks {
        return Err(VideoPacketParseError::MacroblockNumberOutOfRange {
            value: macroblock_number,
            total: ctx.total_macroblocks,
        });
    }

    // 5. quant_scale: only present when video_object_layer_shape !=
    //    "binary only". We rejected non-rectangular above (binary-only
    //    is a non-rectangular sub-kind), so quant_scale is always here.
    let quant_scale = br.read_bits(ctx.quant_precision as usize)? as u16;
    if quant_scale == 0 {
        return Err(VideoPacketParseError::ForbiddenQuantScale);
    }

    // 6. header_extension_code: only present when
    //    video_object_layer_shape == "rectangular". (For non-rectangular
    //    shapes the extension code was already read at the top of the
    //    syntax table.)
    let header_extension_code = br.read_bool()?;

    if !header_extension_code {
        return Ok(VideoPacketHeader {
            macroblock_number,
            quant_scale,
            header_extension_code: false,
            modulo_time_base: None,
            vop_time_increment: None,
            vop_coding_type: None,
            intra_dc_vlc_thr: None,
            vop_fcode_forward: None,
            vop_fcode_backward: None,
        });
    }

    // 7. Extension body (rectangular path): modulo_time_base,
    //    vop_time_increment, vop_coding_type, intra_dc_vlc_thr,
    //    optional fcodes. Out-of-scope items (sprite_trajectory,
    //    vop_reduced_resolution) are rejected before they would be
    //    consumed so we never silently misalign the reader.
    let mtb = read_modulo_time_base(br)?;
    read_marker(br)?;
    let bits = crate::vop::vop_time_increment_bits(ctx.time_increment_resolution) as usize;
    let vop_time_increment = br.read_bits(bits)? as u16;
    read_marker(br)?;
    let vop_coding_type = VopCodingType::from_bits(br.read_bits(2)?);

    // shape == rectangular: change_conv_ratio_disable /
    // vop_shape_coding_type are NOT read. (The §6.2.5 syntax gates
    // them on `video_object_layer_shape != "rectangular"`.)

    let intra_dc_vlc_thr = br.read_bits(3)? as u8;

    // sprite_trajectory() body — out of round-18 scope when
    // (sprite_enable == "GMC" && coding_type == S &&
    // no_of_sprite_warping_points > 0). Reject the entire S-coded
    // path under GMC up front so we never under-read the body.
    if matches!(vop_coding_type, VopCodingType::S) && ctx.sprite_gmc {
        return Err(VideoPacketParseError::UnsupportedBranch(
            "sprite_trajectory in video_packet_header extension",
        ));
    }

    // vop_reduced_resolution — also out of scope.
    if ctx.reduced_resolution_vop_enable {
        return Err(VideoPacketParseError::UnsupportedBranch(
            "reduced_resolution_vop_enable in video_packet_header extension",
        ));
    }

    let vop_fcode_forward = if !matches!(vop_coding_type, VopCodingType::I) {
        let v = br.read_bits(3)? as u8;
        if v == 0 {
            return Err(VideoPacketParseError::ForbiddenFcode);
        }
        Some(v)
    } else {
        None
    };
    let vop_fcode_backward = if matches!(vop_coding_type, VopCodingType::B) {
        let v = br.read_bits(3)? as u8;
        if v == 0 {
            return Err(VideoPacketParseError::ForbiddenFcode);
        }
        Some(v)
    } else {
        None
    };

    // newpred_enable extension (vop_id / vop_id_for_prediction) —
    // out of round-18 scope.
    if ctx.newpred_enable {
        return Err(VideoPacketParseError::UnsupportedBranch(
            "newpred_enable in video_packet_header extension",
        ));
    }

    Ok(VideoPacketHeader {
        macroblock_number,
        quant_scale,
        header_extension_code: true,
        modulo_time_base: Some(mtb),
        vop_time_increment: Some(vop_time_increment),
        vop_coding_type: Some(vop_coding_type),
        intra_dc_vlc_thr: Some(intra_dc_vlc_thr),
        vop_fcode_forward,
        vop_fcode_backward,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------- Table 6-27 -------

    #[test]
    fn table_6_27_boundaries() {
        // Each row of Table 6-27, walked at the lower and upper edges.
        for (lo, hi, expect) in [
            (1, 2, 1u8),
            (3, 4, 2),
            (5, 8, 3),
            (9, 16, 4),
            (17, 32, 5),
            (33, 64, 6),
            (65, 128, 7),
            (129, 256, 8),
            (257, 512, 9),
            (513, 1024, 10),
            (1025, 2048, 11),
            (2049, 4096, 12),
            (4097, 8192, 13),
            (8193, 16384, 14),
        ] {
            assert_eq!(
                macroblock_number_bit_width(lo),
                expect,
                "lo={lo} expected {expect}"
            );
            assert_eq!(
                macroblock_number_bit_width(hi),
                expect,
                "hi={hi} expected {expect}"
            );
        }
    }

    #[test]
    fn table_6_27_zero_input_returns_one_bit() {
        assert_eq!(macroblock_number_bit_width(0), 1);
    }

    #[test]
    fn total_macroblocks_qcif() {
        // QCIF 176×144 → 11 columns × 9 rows = 99 macroblocks.
        assert_eq!(total_macroblocks(176, 144), 99);
        assert_eq!(macroblock_number_bit_width(99), 7);
    }

    #[test]
    fn total_macroblocks_rounds_up_to_16_grid() {
        // 320×240 → 20 × 15 = 300 macroblocks.
        assert_eq!(total_macroblocks(320, 240), 300);
        // 321×241 still 21 × 16 = 336 (ceil-div).
        assert_eq!(total_macroblocks(321, 241), 21 * 16);
    }

    // ------- §6.3.3 resync_marker length -------

    #[test]
    fn resync_marker_length_i_vop_is_17() {
        assert_eq!(resync_marker_length(VopCodingType::I, 0, 0), 17);
        // I-VOP ignores any fcode value.
        assert_eq!(resync_marker_length(VopCodingType::I, 7, 7), 17);
    }

    #[test]
    fn resync_marker_length_p_vop_scales_with_fcode() {
        // fcode = 1 → 15+1 zeros + 1 one = 17 bits.
        assert_eq!(resync_marker_length(VopCodingType::P, 1, 0), 17);
        // fcode = 4 → 15+4 zeros + 1 = 20.
        assert_eq!(resync_marker_length(VopCodingType::P, 4, 0), 20);
        // fcode = 7 → 15+7 zeros + 1 = 23 (longest legal marker).
        assert_eq!(resync_marker_length(VopCodingType::P, 7, 0), 23);
    }

    #[test]
    fn resync_marker_length_s_vop_uses_forward_fcode() {
        assert_eq!(resync_marker_length(VopCodingType::S, 3, 0), 19);
    }

    #[test]
    fn resync_marker_length_b_vop_takes_max_and_floors_at_17() {
        // §6.3.3: max(15 + fcode, 17) ZEROS + a one.
        // max(fwd, bwd) = 2 → zeros = max(17, 17) = 17 → 18 bits.
        assert_eq!(resync_marker_length(VopCodingType::B, 2, 1), 18);
        // max = 1 → zeros = max(16, 17) = 17 → 18 bits (the floor).
        assert_eq!(resync_marker_length(VopCodingType::B, 1, 1), 18);
        // max = 7 → zeros = 22 → 23 bits.
        assert_eq!(resync_marker_length(VopCodingType::B, 4, 7), 23);
    }

    // ------- §5.2.5 stuffing -------

    /// Build a byte starting with `0` followed by `n` `1` bits, then
    /// zeros for any remaining bits. The caller picks `n` so the
    /// pattern matches the position the §5.2.5 stuffing expects to
    /// consume.
    fn stuffing_byte(bits_already_taken: usize) -> u8 {
        // After `bits_already_taken` bits we want a `0` then `1`s up
        // to byte alignment. For a fresh byte (`bits_already_taken =
        // 0`) the result is `0 1111111` = 0x7F.
        let remaining = 8 - bits_already_taken;
        if remaining == 0 {
            return 0;
        }
        let ones = remaining - 1;
        if ones == 0 {
            0
        } else {
            (1u8 << ones) - 1
        }
    }

    #[test]
    fn consume_next_resync_marker_aligned_reader_consumes_full_byte() {
        // Reader at bit 0. Stuffing is `0 1111111` = 0x7F. After the
        // stuffing the position should be the next byte boundary.
        assert_eq!(stuffing_byte(0), 0x7F);
        let data = [0x7F, 0x00];
        let mut br = BitReader::new(&data);
        consume_next_resync_marker(&mut br).unwrap();
        assert_eq!(br.bit_position(), 8);
    }

    #[test]
    fn consume_next_resync_marker_partial_byte_works() {
        // 3 bits already consumed → stuffing is `0` then `1111` (4
        // ones) to reach byte alignment.
        let mut data = vec![0u8; 2];
        // Top 3 bits arbitrary 1s, next bit `0`, then 4 `1`s.
        data[0] = 0b1110_1111;
        let mut br = BitReader::new(&data);
        br.read_bits(3).unwrap();
        consume_next_resync_marker(&mut br).unwrap();
        assert_eq!(br.bit_position(), 8);
    }

    #[test]
    fn consume_next_resync_marker_rejects_non_zero_first_bit() {
        let data = [0xFF];
        let mut br = BitReader::new(&data);
        let err = consume_next_resync_marker(&mut br).unwrap_err();
        assert!(matches!(
            err,
            VideoPacketParseError::MissingResyncMarker { .. }
        ));
    }

    #[test]
    fn consume_next_resync_marker_rejects_stuffing_with_zero_in_tail() {
        // Should be `0 1111111` but we give `0 1110111`. The first 0
        // is consumed, then `111` are accepted, then a `0` appears
        // before byte alignment → MissingResyncMarker.
        let data = [0b0111_0111];
        let mut br = BitReader::new(&data);
        let err = consume_next_resync_marker(&mut br).unwrap_err();
        assert!(matches!(
            err,
            VideoPacketParseError::MissingResyncMarker { .. }
        ));
    }

    // ------- probe_resync_marker -------

    #[test]
    fn probe_resync_marker_finds_17_bit_i_vop_marker() {
        // 16 zeros + 1 = 0x0001 in the high two bytes, then padding.
        // 0000 0000 0000 0001 _ ...  → first 17 bits = `1`.
        let data = [0x00, 0x00, 0x80, 0x00]; // bit 16 is the trailing `1`
        let br = BitReader::new(&data);
        assert!(probe_resync_marker(&br, VopCodingType::I, 0, 0));
    }

    #[test]
    fn probe_resync_marker_rejects_non_aligned_reader() {
        let data = [0x00, 0x00, 0x80, 0x00];
        let mut br = BitReader::new(&data);
        br.read_bits(1).unwrap();
        // Even though the marker would still parse if we read 17
        // bits from this position, the §6.3.3 alignment rule forbids
        // it. probe must return false.
        assert!(!probe_resync_marker(&br, VopCodingType::I, 0, 0));
    }

    #[test]
    fn probe_resync_marker_distinguishes_p_vop_length() {
        // P-VOP with fcode=2 needs 18 bits: 17 zeros + 1.
        // Pattern: 0000_0000 0000_0000 0100_0000 → bit 17 is `1`.
        let data = [0x00, 0x00, 0x40, 0x00];
        let br = BitReader::new(&data);
        assert!(probe_resync_marker(&br, VopCodingType::P, 2, 0));
        // An I-VOP probe at the same position would not see the
        // marker (it expects bit 16 = 1, but the data has bit 17 = 1).
        assert!(!probe_resync_marker(&br, VopCodingType::I, 0, 0));
    }

    // ------- parse_video_packet_header (rectangular path) -------

    /// Minimal helper: build a byte vector by emitting fields onto a
    /// bit-level writer.
    struct BitWriter {
        bits: Vec<bool>,
    }
    impl BitWriter {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn write_bits(&mut self, value: u32, n: usize) {
            for i in (0..n).rev() {
                self.bits.push(((value >> i) & 1) == 1);
            }
        }
        fn write_bool(&mut self, v: bool) {
            self.bits.push(v);
        }
        fn finish(mut self) -> Vec<u8> {
            // Pad to byte alignment with 0s.
            while self.bits.len() % 8 != 0 {
                self.bits.push(false);
            }
            let mut out = vec![0u8; self.bits.len() / 8];
            for (i, b) in self.bits.iter().enumerate() {
                if *b {
                    out[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            out
        }
    }

    fn default_ctx() -> VideoPacketContext {
        VideoPacketContext {
            coding_type: VopCodingType::I,
            fcode_fwd: 0,
            fcode_bwd: 0,
            quant_precision: 5,
            time_increment_resolution: 30,
            video_object_layer_shape: 0,
            resync_marker_disable: false,
            newpred_enable: false,
            reduced_resolution_vop_enable: false,
            sprite_gmc: false,
            total_macroblocks: 99, // QCIF
        }
    }

    #[test]
    fn parse_minimal_i_vop_packet_header_without_extension() {
        let mut w = BitWriter::new();
        // §5.2.5 stuffing: we'll start fresh so the reader is
        // already byte-aligned; the parser still consumes one
        // `0 1*` stuffing block.
        w.write_bool(false); // first bit '0'
        for _ in 0..7 {
            w.write_bool(true);
        } // pad to byte boundary
          // resync_marker for I-VOP: 17 bits = 16 zeros + 1.
        w.write_bits(1, 17);
        // macroblock_number: QCIF total=99 → 7 bits. Pick 42.
        w.write_bits(42, 7);
        // quant_scale: 5 bits, pick 12.
        w.write_bits(12, 5);
        // header_extension_code: 0.
        w.write_bool(false);
        let data = w.finish();

        let ctx = default_ctx();
        let mut br = BitReader::new(&data);
        let hdr = parse_video_packet_header(&mut br, &ctx).unwrap();
        assert_eq!(hdr.macroblock_number, 42);
        assert_eq!(hdr.quant_scale, 12);
        assert!(!hdr.header_extension_code);
        assert_eq!(hdr.vop_coding_type, None);
    }

    #[test]
    fn parse_i_vop_packet_header_with_extension_body() {
        let mut w = BitWriter::new();
        // stuffing to align
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        // resync_marker (17 bits for I).
        w.write_bits(1, 17);
        // macroblock_number (7 bits).
        w.write_bits(0, 7);
        // quant_scale (5 bits).
        w.write_bits(10, 5);
        // header_extension_code = 1.
        w.write_bool(true);
        // modulo_time_base = 0 → just the terminator '0'.
        w.write_bool(false);
        // marker bit.
        w.write_bool(true);
        // vop_time_increment: with resolution 30 →
        // ceil(log2(30)) = 5 bits. Pick 7.
        w.write_bits(7, 5);
        // marker bit.
        w.write_bool(true);
        // vop_coding_type = I = 00.
        w.write_bits(0b00, 2);
        // intra_dc_vlc_thr (3 bits) = 3.
        w.write_bits(3, 3);
        // No fcodes (I-VOP).
        let data = w.finish();

        let ctx = default_ctx();
        let mut br = BitReader::new(&data);
        let hdr = parse_video_packet_header(&mut br, &ctx).unwrap();
        assert!(hdr.header_extension_code);
        assert_eq!(hdr.modulo_time_base, Some(0));
        assert_eq!(hdr.vop_time_increment, Some(7));
        assert_eq!(hdr.vop_coding_type, Some(VopCodingType::I));
        assert_eq!(hdr.intra_dc_vlc_thr, Some(3));
        assert_eq!(hdr.vop_fcode_forward, None);
        assert_eq!(hdr.vop_fcode_backward, None);
    }

    #[test]
    fn parse_p_vop_packet_header_with_extension_includes_forward_fcode() {
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        // P-VOP with fcode_fwd = 1 → marker length = 17.
        w.write_bits(1, 17);
        w.write_bits(5, 7); // macroblock_number
        w.write_bits(7, 5); // quant_scale
        w.write_bool(true); // header_extension_code
                            // modulo_time_base = 0
        w.write_bool(false);
        w.write_bool(true); // marker
        w.write_bits(0, 5); // vop_time_increment
        w.write_bool(true); // marker
                            // vop_coding_type = P = 01
        w.write_bits(0b01, 2);
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(1, 3); // vop_fcode_forward
        let data = w.finish();

        let mut ctx = default_ctx();
        ctx.coding_type = VopCodingType::P;
        ctx.fcode_fwd = 1;
        let mut br = BitReader::new(&data);
        let hdr = parse_video_packet_header(&mut br, &ctx).unwrap();
        assert_eq!(hdr.vop_coding_type, Some(VopCodingType::P));
        assert_eq!(hdr.vop_fcode_forward, Some(1));
        assert_eq!(hdr.vop_fcode_backward, None);
    }

    #[test]
    fn parse_b_vop_packet_header_with_extension_includes_both_fcodes() {
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        // B-VOP with fwd=2 bwd=1 → max=2, zeros = max(15+2, 17) = 17 → 18 bits.
        w.write_bits(1, 18);
        w.write_bits(1, 7);
        w.write_bits(3, 5);
        w.write_bool(true);
        w.write_bool(false);
        w.write_bool(true);
        w.write_bits(1, 5);
        w.write_bool(true);
        w.write_bits(0b10, 2); // B
        w.write_bits(0, 3);
        w.write_bits(2, 3); // fwd
        w.write_bits(1, 3); // bwd
        let data = w.finish();

        let mut ctx = default_ctx();
        ctx.coding_type = VopCodingType::B;
        ctx.fcode_fwd = 2;
        ctx.fcode_bwd = 1;
        let mut br = BitReader::new(&data);
        let hdr = parse_video_packet_header(&mut br, &ctx).unwrap();
        assert_eq!(hdr.vop_coding_type, Some(VopCodingType::B));
        assert_eq!(hdr.vop_fcode_forward, Some(2));
        assert_eq!(hdr.vop_fcode_backward, Some(1));
    }

    #[test]
    fn parser_rejects_resync_marker_disable() {
        let mut ctx = default_ctx();
        ctx.resync_marker_disable = true;
        let mut br = BitReader::new(&[0u8; 4]);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert_eq!(err, VideoPacketParseError::ResyncDisabled);
    }

    #[test]
    fn parser_rejects_non_rectangular_shape() {
        let mut ctx = default_ctx();
        ctx.video_object_layer_shape = 2; // binary-only
        let mut br = BitReader::new(&[0u8; 4]);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert!(matches!(err, VideoPacketParseError::UnsupportedBranch(_)));
    }

    #[test]
    fn parser_rejects_bad_quant_precision() {
        let mut ctx = default_ctx();
        ctx.quant_precision = 2;
        let mut br = BitReader::new(&[0u8; 4]);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert_eq!(err, VideoPacketParseError::BadQuantPrecision(2));
    }

    #[test]
    fn parser_rejects_missing_resync_marker() {
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        // Write 17 bits that are NOT a resync marker (all ones).
        w.write_bits(0x1FFFF, 17);
        let data = w.finish();
        let ctx = default_ctx();
        let mut br = BitReader::new(&data);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert!(matches!(
            err,
            VideoPacketParseError::MissingResyncMarker { .. }
        ));
    }

    #[test]
    fn parser_rejects_macroblock_number_out_of_range() {
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        w.write_bits(1, 17);
        // macroblock_number = 127 (max for 7 bits) but total = 99.
        w.write_bits(127, 7);
        w.write_bits(1, 5);
        w.write_bool(false);
        let data = w.finish();
        let ctx = default_ctx();
        let mut br = BitReader::new(&data);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert_eq!(
            err,
            VideoPacketParseError::MacroblockNumberOutOfRange {
                value: 127,
                total: 99
            }
        );
    }

    #[test]
    fn parser_rejects_zero_quant_scale() {
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        w.write_bits(1, 17);
        w.write_bits(0, 7);
        w.write_bits(0, 5); // quant_scale = 0
        w.write_bool(false);
        let data = w.finish();
        let ctx = default_ctx();
        let mut br = BitReader::new(&data);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert_eq!(err, VideoPacketParseError::ForbiddenQuantScale);
    }

    #[test]
    fn parser_rejects_zero_fcode_in_extension() {
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        w.write_bits(1, 17); // P with fcode=1, 17 bits
        w.write_bits(0, 7);
        w.write_bits(5, 5);
        w.write_bool(true); // header_extension_code
        w.write_bool(false);
        w.write_bool(true);
        w.write_bits(0, 5);
        w.write_bool(true);
        w.write_bits(0b01, 2); // P
        w.write_bits(0, 3); // intra_dc_vlc_thr
        w.write_bits(0, 3); // forbidden fcode
        let data = w.finish();
        let mut ctx = default_ctx();
        ctx.coding_type = VopCodingType::P;
        ctx.fcode_fwd = 1;
        let mut br = BitReader::new(&data);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert_eq!(err, VideoPacketParseError::ForbiddenFcode);
    }

    #[test]
    fn parser_rejects_sprite_gmc_extension() {
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        // S-VOP with fcode 1 → 17-bit marker.
        w.write_bits(1, 17);
        w.write_bits(0, 7);
        w.write_bits(5, 5);
        w.write_bool(true);
        w.write_bool(false);
        w.write_bool(true);
        w.write_bits(0, 5);
        w.write_bool(true);
        w.write_bits(0b11, 2); // S
        w.write_bits(0, 3);
        let data = w.finish();
        let mut ctx = default_ctx();
        ctx.coding_type = VopCodingType::S;
        ctx.fcode_fwd = 1;
        ctx.sprite_gmc = true;
        let mut br = BitReader::new(&data);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert!(matches!(err, VideoPacketParseError::UnsupportedBranch(_)));
    }

    #[test]
    fn parser_rejects_newpred_extension() {
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        w.write_bits(1, 17);
        w.write_bits(0, 7);
        w.write_bits(5, 5);
        w.write_bool(true);
        w.write_bool(false);
        w.write_bool(true);
        w.write_bits(0, 5);
        w.write_bool(true);
        w.write_bits(0b00, 2); // I
        w.write_bits(0, 3);
        let data = w.finish();
        let mut ctx = default_ctx();
        ctx.newpred_enable = true;
        let mut br = BitReader::new(&data);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert!(matches!(err, VideoPacketParseError::UnsupportedBranch(_)));
    }

    #[test]
    fn parser_rejects_reduced_resolution_extension() {
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        w.write_bits(1, 17);
        w.write_bits(0, 7);
        w.write_bits(5, 5);
        w.write_bool(true);
        w.write_bool(false);
        w.write_bool(true);
        w.write_bits(0, 5);
        w.write_bool(true);
        w.write_bits(0b00, 2); // I
        w.write_bits(0, 3);
        let data = w.finish();
        let mut ctx = default_ctx();
        ctx.reduced_resolution_vop_enable = true;
        let mut br = BitReader::new(&data);
        let err = parse_video_packet_header(&mut br, &ctx).unwrap_err();
        assert!(matches!(err, VideoPacketParseError::UnsupportedBranch(_)));
    }

    // ------- Error trait/display coverage -------

    #[test]
    fn error_display_covers_each_variant() {
        for e in [
            VideoPacketParseError::Truncated,
            VideoPacketParseError::ResyncDisabled,
            VideoPacketParseError::MissingResyncMarker {
                expected_bits: 17,
                observed: 0x1FFFF,
            },
            VideoPacketParseError::MacroblockNumberOutOfRange {
                value: 99,
                total: 99,
            },
            VideoPacketParseError::ForbiddenQuantScale,
            VideoPacketParseError::ForbiddenFcode,
            VideoPacketParseError::BadQuantPrecision(2),
            VideoPacketParseError::UnsupportedBranch("foo"),
        ] {
            let s = format!("{e}");
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn bitreader_truncation_maps_to_truncated() {
        let e: VideoPacketParseError = BitReaderError::EndOfStream.into();
        assert_eq!(e, VideoPacketParseError::Truncated);
        let e: VideoPacketParseError = BitReaderError::TooManyBits.into();
        assert_eq!(e, VideoPacketParseError::Truncated);
    }

    #[test]
    fn header_extension_code_false_returns_none_for_optionals() {
        // Re-run the minimal test and inspect every optional field.
        let mut w = BitWriter::new();
        w.write_bool(false);
        for _ in 0..7 {
            w.write_bool(true);
        }
        w.write_bits(1, 17);
        w.write_bits(0, 7);
        w.write_bits(1, 5);
        w.write_bool(false);
        let data = w.finish();
        let ctx = default_ctx();
        let mut br = BitReader::new(&data);
        let hdr = parse_video_packet_header(&mut br, &ctx).unwrap();
        assert!(hdr.modulo_time_base.is_none());
        assert!(hdr.vop_time_increment.is_none());
        assert!(hdr.vop_coding_type.is_none());
        assert!(hdr.intra_dc_vlc_thr.is_none());
        assert!(hdr.vop_fcode_forward.is_none());
        assert!(hdr.vop_fcode_backward.is_none());
    }
}

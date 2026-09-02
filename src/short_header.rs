//! §6.2.5.2 **video plane with short header** — the H.263-compatible
//! abbreviated syntax (`short_video_header == 1`): parse + decode.
//!
//! A short-header stream carries no VisualObjectSequence / VOL
//! configuration at all: every picture starts with the 22-bit
//! `short_video_start_marker` and the Table 6-28 fixed settings apply
//! (rectangular, `obmc_disable == 1`, method-2 quantisation,
//! `vop_fcode_forward == 1`, `vop_rounding_type == 0`, progressive,
//! no resync markers, no data partitioning, 8-bit samples). The
//! picture dimensions come from the Table 6-29 `source_format`
//! (sub-QCIF … 16CIF), which also fixes the GOB geometry
//! (`num_macroblocks_in_gob` / `num_gobs_in_vop`).
//!
//! Inside the picture the §6.2.6 macroblock syntax runs with its
//! `short_video_header` gates: no `ac_pred_flag`, `derived_mb_type
//! == 2` (inter4v) forbidden, one `motion_vector("forward")` per inter
//! macroblock (Table B.12 with `f_code == 1`), and the §6.2.7 block
//! whose intra DC is the 8-bit `intra_dc_coefficient` FLC
//! (§6.3.7: 0 and 128 reserved, 255 signals 128) reconstructed with
//! `dc_scaler = 8` (§7.4.4.3), with no §7.4.3 DC/AC prediction at all
//! and the Table B.17 (inter) Tcoef VLC for every block plus the
//! §7.4.1.3 Type-4 escape ([`crate::texture::decode_ac_event_short_video_header`]).
//! Motion vectors are predicted per §7.6.5 (median over the block-0
//! Figure 7-34 candidates) with the GOB rule: a candidate "outside the
//! current GOB (when `short_video_header` is 1) for which
//! `gob_header_empty` is 0 is treated as transparent" — every
//! macroblock of the earlier GOBs is invalidated when a GOB header
//! (`gob_resync_marker` + `gob_number` + `gob_frame_id` +
//! `quant_scale`) is present, and the running quantiser restarts from
//! that `quant_scale`.
//!
//! The decoded macroblocks are handed to the reference-frame chain in
//! the same [`PVopMbContent`] / [`ReconstructedMacroblock`] shape the
//! long-header walks produce, so the §7.6 half-sample prediction and
//! the §7.6.1 anchor chain are shared unchanged.
//!
//! Provenance: §6.2.5.2 / §6.3.5.2 (Tables 6-28, 6-29), §6.2.6,
//! §6.2.7, §6.3.7 (`intra_dc_coefficient`), §7.4.1 / §7.4.1.3 Type 4,
//! §7.4.3 (prediction gated off), §7.4.4.3 (`dc_scaler = 8`), §7.6.4
//! (no unrestricted vectors), §7.6.5 (GOB rule) of ISO/IEC
//! 14496-2:2004 (3rd edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`. No
//! third-party source was consulted.

use crate::bitreader::{BitReader, BitReaderError};
use crate::block::{
    inter_block_from_events, pattern_code, InterMacroblock, IntraMacroblock,
    MacroblockTextureContext,
};
use crate::frame_decode::PVopMbContent;
use crate::idct::idct_8x8;
use crate::inverse_quant::{inverse_quant_method2, InverseQuantContext};
use crate::macroblock::{
    decode_cbpy4, decode_mcbpc, dquant_value, DerivedMbType, MCBPC_I, MCBPC_P,
};
use crate::motion::{
    decode_motion_vector_delta, predict_motion_vector, reconstruct_motion_vector, MotionVector,
    MvMode,
};
use crate::mv_predictor_grid::MvGrid;
use crate::pvop_mv::PvopMbMotion;
use crate::reconstruct::{reconstruct_intra_macroblock, ReconstructedMacroblock};
use crate::scan::{events_to_qfs, inverse_scan, ScanType};
use crate::texture::{
    decode_ac_events_short_video_header, AcEvent, DcComponent, TcoefTable, TextureParseError,
};
use crate::vop::VopCodingType;

/// `short_video_start_marker` — 22 bits `0000 0000 0000 0000 1000 00`.
pub const SHORT_VIDEO_START_MARKER: u32 = 0x20;
/// `short_video_end_marker` — 22 bits `0000 0000 0000 0000 1111 11`.
pub const SHORT_VIDEO_END_MARKER: u32 = 0x3F;
/// `gob_resync_marker` — 17 bits `0000 0000 0000 0000 1`.
pub const GOB_RESYNC_MARKER: u32 = 1;
/// Bit width of the two picture-level markers.
pub const SHORT_MARKER_BITS: usize = 22;
/// Bit width of the GOB marker.
pub const GOB_MARKER_BITS: usize = 17;

/// Whether `data[pos..]` starts a byte-aligned
/// `video_plane_with_short_header()`: the 22-bit start marker is
/// `00 00 8x` with the top two bits of `temporal_reference` in `x`'s
/// low bits (`0x80..=0x83`). A byte-aligned `gob_resync_marker`
/// (`gob_number >= 1`) reads `00 00 84..=FF` and the end marker
/// `00 00 FC..=FF`, so the picture start is unambiguous.
pub fn is_short_header_picture_start(data: &[u8], pos: usize) -> bool {
    data.len() >= pos + 3 && data[pos] == 0 && data[pos + 1] == 0 && (data[pos + 2] & 0xFC) == 0x80
}

/// Byte offsets of every short-header picture start in `data`.
pub fn scan_short_header_pictures(data: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 2 < data.len() {
        if is_short_header_picture_start(data, i) {
            out.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    out
}

/// Table 6-29 `source_format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    /// `001` — 128×96.
    SubQcif,
    /// `010` — 176×144.
    Qcif,
    /// `011` — 352×288.
    Cif,
    /// `100` — 704×576.
    Cif4,
    /// `101` — 1408×1152.
    Cif16,
}

impl SourceFormat {
    /// Decode the 3-bit field (`None` for the reserved values).
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            0b001 => Some(Self::SubQcif),
            0b010 => Some(Self::Qcif),
            0b011 => Some(Self::Cif),
            0b100 => Some(Self::Cif4),
            0b101 => Some(Self::Cif16),
            _ => None,
        }
    }

    /// The 3-bit field value.
    pub fn code(self) -> u32 {
        match self {
            Self::SubQcif => 0b001,
            Self::Qcif => 0b010,
            Self::Cif => 0b011,
            Self::Cif4 => 0b100,
            Self::Cif16 => 0b101,
        }
    }

    /// The format whose picture size is exactly `width × height`.
    pub fn from_dimensions(width: u32, height: u32) -> Option<Self> {
        [
            Self::SubQcif,
            Self::Qcif,
            Self::Cif,
            Self::Cif4,
            Self::Cif16,
        ]
        .into_iter()
        .find(|f| {
            let (w, h) = f.dimensions();
            (u32::from(w), u32::from(h)) == (width, height)
        })
    }

    /// `(vop_width, vop_height)`.
    pub fn dimensions(self) -> (u16, u16) {
        match self {
            Self::SubQcif => (128, 96),
            Self::Qcif => (176, 144),
            Self::Cif => (352, 288),
            Self::Cif4 => (704, 576),
            Self::Cif16 => (1408, 1152),
        }
    }

    /// `num_macroblocks_in_gob`.
    pub fn macroblocks_per_gob(self) -> usize {
        match self {
            Self::SubQcif => 8,
            Self::Qcif => 11,
            Self::Cif => 22,
            Self::Cif4 => 88,
            Self::Cif16 => 352,
        }
    }

    /// `num_gobs_in_vop`.
    pub fn gobs_per_picture(self) -> usize {
        match self {
            Self::SubQcif => 6,
            Self::Qcif => 9,
            Self::Cif | Self::Cif4 | Self::Cif16 => 18,
        }
    }

    /// Macroblock grid `(mb_width, mb_height)`.
    pub fn mb_dimensions(self) -> (usize, usize) {
        let (w, h) = self.dimensions();
        (usize::from(w) / 16, usize::from(h) / 16)
    }
}

/// The decoded `video_plane_with_short_header()` picture header
/// (through the `pei` loop).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortHeaderPicture {
    /// 8-bit `temporal_reference` (30000/1001 Hz ticks, modulo 256).
    pub temporal_reference: u8,
    /// `split_screen_indicator` (display hint only).
    pub split_screen: bool,
    /// `document_camera_indicator` (display hint only).
    pub document_camera: bool,
    /// `full_picture_freeze_release` (display hint only).
    pub freeze_release: bool,
    /// Table 6-29 `source_format`.
    pub source_format: SourceFormat,
    /// `picture_coding_type`: I or P.
    pub coding_type: VopCodingType,
    /// 5-bit `vop_quant`.
    pub quant: u8,
}

/// Short-header parse / decode errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShortHeaderError {
    /// The unit does not start with `short_video_start_marker`.
    MissingStartMarker,
    /// A fixed-value bit (marker, zero bit, reserved field) held the
    /// wrong value.
    BadFixedBits(&'static str),
    /// A reserved `source_format` value.
    ReservedSourceFormat(u32),
    /// `vop_quant == 0`.
    ZeroQuant,
    /// A `gob_number` read from a GOB header disagreed with the
    /// inferred one (§6.3.5.2).
    GobNumberMismatch {
        /// The `gob_number` field as read.
        read: u32,
        /// The value inferred from the macroblock count.
        expected: u32,
    },
    /// `quant_scale == 0` in a GOB header.
    ZeroGobQuant,
    /// `derived_mb_type == 2` (inter4v) in a short-header P picture.
    Inter4vForbidden,
    /// An intra `mcbpc` type inside a P picture that Table B.7 does
    /// not admit, or an inter type in an I picture.
    BadMbType(u8),
    /// `intra_dc_coefficient` 0 or 128 (reserved).
    ReservedIntraDc(u32),
    /// The unit ended mid-field.
    Truncated,
    /// A macroblock-header VLC did not match.
    Macroblock(crate::macroblock::MacroblockParseError),
    /// A texture VLC / escape failed.
    Texture(TextureParseError),
    /// A `motion_vector()` body failed.
    Motion(crate::motion::MotionParseError),
    /// A block assembly failure.
    Block(crate::block::BlockAssemblyError),
}

impl core::fmt::Display for ShortHeaderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingStartMarker => write!(f, "missing short_video_start_marker"),
            Self::BadFixedBits(what) => write!(f, "short header: bad fixed bits ({what})"),
            Self::ReservedSourceFormat(v) => write!(f, "short header: reserved source_format {v}"),
            Self::ZeroQuant => write!(f, "short header: vop_quant == 0"),
            Self::GobNumberMismatch { read, expected } => {
                write!(f, "short header: gob_number {read} (expected {expected})")
            }
            Self::ZeroGobQuant => write!(f, "short header: GOB quant_scale == 0"),
            Self::Inter4vForbidden => write!(f, "short header: inter4v macroblock"),
            Self::BadMbType(t) => write!(f, "short header: macroblock type {t} not allowed"),
            Self::ReservedIntraDc(v) => {
                write!(f, "short header: reserved intra_dc_coefficient {v}")
            }
            Self::Truncated => write!(f, "short header: truncated"),
            Self::Macroblock(e) => write!(f, "short header: {e}"),
            Self::Texture(e) => write!(f, "short header: {e}"),
            Self::Motion(e) => write!(f, "short header: {e}"),
            Self::Block(e) => write!(f, "short header: {e}"),
        }
    }
}

impl std::error::Error for ShortHeaderError {}

impl From<BitReaderError> for ShortHeaderError {
    fn from(_: BitReaderError) -> Self {
        Self::Truncated
    }
}
impl From<crate::macroblock::MacroblockParseError> for ShortHeaderError {
    fn from(e: crate::macroblock::MacroblockParseError) -> Self {
        Self::Macroblock(e)
    }
}
impl From<TextureParseError> for ShortHeaderError {
    fn from(e: TextureParseError) -> Self {
        Self::Texture(e)
    }
}
impl From<crate::motion::MotionParseError> for ShortHeaderError {
    fn from(e: crate::motion::MotionParseError) -> Self {
        Self::Motion(e)
    }
}
impl From<crate::block::BlockAssemblyError> for ShortHeaderError {
    fn from(e: crate::block::BlockAssemblyError) -> Self {
        Self::Block(e)
    }
}
impl From<crate::scan::InverseScanError> for ShortHeaderError {
    fn from(_: crate::scan::InverseScanError) -> Self {
        Self::Block(crate::block::BlockAssemblyError::NotCoded)
    }
}

/// Parse the picture header of a `video_plane_with_short_header()`
/// from a reader positioned at the (byte-aligned) start marker; on
/// success the reader sits at the first GOB's first macroblock.
pub fn parse_short_header_picture(
    br: &mut BitReader<'_>,
) -> Result<ShortHeaderPicture, ShortHeaderError> {
    if br.read_bits(SHORT_MARKER_BITS)? != SHORT_VIDEO_START_MARKER {
        return Err(ShortHeaderError::MissingStartMarker);
    }
    let temporal_reference = br.read_bits(8)? as u8;
    if !br.read_bool()? {
        return Err(ShortHeaderError::BadFixedBits("marker_bit"));
    }
    if br.read_bool()? {
        return Err(ShortHeaderError::BadFixedBits("zero_bit"));
    }
    let split_screen = br.read_bool()?;
    let document_camera = br.read_bool()?;
    let freeze_release = br.read_bool()?;
    let sf = br.read_bits(3)?;
    let source_format =
        SourceFormat::from_code(sf).ok_or(ShortHeaderError::ReservedSourceFormat(sf))?;
    let coding_type = if br.read_bool()? {
        VopCodingType::P
    } else {
        VopCodingType::I
    };
    if br.read_bits(4)? != 0 {
        return Err(ShortHeaderError::BadFixedBits("four_reserved_zero_bits"));
    }
    let quant = br.read_bits(5)? as u8;
    if quant == 0 {
        return Err(ShortHeaderError::ZeroQuant);
    }
    if br.read_bool()? {
        return Err(ShortHeaderError::BadFixedBits("zero_bit after vop_quant"));
    }
    // pei / psupp: accept and discard (§6.3.5.2).
    while br.read_bool()? {
        br.skip_bits(8)?;
    }
    Ok(ShortHeaderPicture {
        temporal_reference,
        split_screen,
        document_camera,
        freeze_release,
        source_format,
        coding_type,
        quant,
    })
}

/// One decoded short-header macroblock header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShortMbHeader {
    not_coded: bool,
    mb_type: DerivedMbType,
    cbpc: u8,
    cbpy: u8,
    dquant: Option<i8>,
}

/// The §6.2.6 macroblock header under `short_video_header == 1`:
/// `not_coded` (P), `mcbpc` (stuffing skipped), `cbpy`, `dquant` — no
/// `ac_pred_flag`, no `interlaced_information()`.
fn parse_short_mb_header(
    br: &mut BitReader<'_>,
    coding_type: VopCodingType,
) -> Result<ShortMbHeader, ShortHeaderError> {
    let is_p = matches!(coding_type, VopCodingType::P);
    loop {
        if is_p && br.read_bool()? {
            return Ok(ShortMbHeader {
                not_coded: true,
                mb_type: DerivedMbType::Inter,
                cbpc: 0,
                cbpy: 0,
                dquant: None,
            });
        }
        let (_len, raw, cbpc) = decode_mcbpc(br, if is_p { MCBPC_P } else { MCBPC_I })?;
        if raw == 5 {
            continue; // stuffing
        }
        if raw == 2 {
            return Err(ShortHeaderError::Inter4vForbidden);
        }
        let mb_type = DerivedMbType::from_raw(raw).ok_or(ShortHeaderError::BadMbType(raw))?;
        let (_clen, cbpy_intra, cbpy_inter) = decode_cbpy4(br)?;
        let cbpy = if mb_type.is_intra() {
            cbpy_intra
        } else {
            cbpy_inter
        };
        let dquant = if mb_type.has_dquant() {
            Some(dquant_value(br.read_bits(2)? as u8))
        } else {
            None
        };
        return Ok(ShortMbHeader {
            not_coded: false,
            mb_type,
            cbpc,
            cbpy,
            dquant,
        });
    }
}

/// Decode the 8-bit `intra_dc_coefficient` (§6.3.7): 0 and 128
/// reserved, 255 → 128.
fn read_intra_dc(br: &mut BitReader<'_>) -> Result<i32, ShortHeaderError> {
    let raw = br.read_bits(8)?;
    match raw {
        0 | 128 => Err(ShortHeaderError::ReservedIntraDc(raw)),
        255 => Ok(128),
        v => Ok(v as i32),
    }
}

/// The value written on the wire for a quantised intra DC in
/// `1..=254` (128 travels as 255).
pub fn intra_dc_code(dc: i32) -> u32 {
    debug_assert!(
        (1..=254).contains(&dc),
        "intra DC {dc} outside the FLC domain"
    );
    if dc == 128 {
        255
    } else {
        dc as u32
    }
}

/// The §7.4 reconstruction of one short-header **intra** block: DC ×
/// `dc_scaler = 8`, method-2 intra AC, zigzag scan, no prediction.
fn intra_block(
    dc: i32,
    events: &[AcEvent],
    qp: u32,
    component: DcComponent,
) -> Result<[[i32; 8]; 8], ShortHeaderError> {
    let qfs = events_to_qfs(events, Some(dc))?;
    let qf = inverse_scan(&qfs, ScanType::Zigzag);
    let f = inverse_quant_method2(
        &qf,
        InverseQuantContext {
            macroblock_intra: true,
            component,
            quantiser_scale: qp,
            bits_per_pixel: 8,
            short_video_header: true,
        },
    );
    Ok(idct_8x8(&f, 8))
}

/// Decode the macroblock layer of one short-header picture from a
/// reader positioned right after [`parse_short_header_picture`]
/// (GOB layers included). Returns the raster-order macroblocks in the
/// [`PVopMbContent`] shape (every macroblock of an I picture is
/// `Intra`).
pub fn decode_short_header_macroblocks(
    br: &mut BitReader<'_>,
    pic: &ShortHeaderPicture,
) -> Result<Vec<PVopMbContent>, ShortHeaderError> {
    let sf = pic.source_format;
    let (mb_width, mb_height) = sf.mb_dimensions();
    let per_gob = sf.macroblocks_per_gob();
    let gobs = sf.gobs_per_picture();
    debug_assert_eq!(per_gob * gobs, mb_width * mb_height);
    let is_p = matches!(pic.coding_type, VopCodingType::P);
    let mut grid = MvGrid::new(mb_height, mb_width);
    let mut running_qp = u32::from(pic.quant);
    let mut out = Vec::with_capacity(mb_width * mb_height);
    let no_matrix = [[0u8; 8]; 8];

    for gob in 0..gobs {
        // §6.2.5.2 gob_layer(): an optional byte-aligned GOB header on
        // every GOB but the first.
        if gob != 0 {
            let mut probe = br.clone();
            probe.align_to_byte();
            let aligned_marker = probe.remaining_bits() >= GOB_MARKER_BITS
                && probe.next_bits(GOB_MARKER_BITS)? == GOB_RESYNC_MARKER;
            let unaligned_marker = br.remaining_bits() >= GOB_MARKER_BITS
                && br.next_bits(GOB_MARKER_BITS)? == GOB_RESYNC_MARKER;
            if aligned_marker || unaligned_marker {
                if !unaligned_marker {
                    br.align_to_byte();
                }
                br.skip_bits(GOB_MARKER_BITS)?;
                let gob_number = br.read_bits(5)?;
                if gob_number as usize != gob {
                    return Err(ShortHeaderError::GobNumberMismatch {
                        read: gob_number,
                        expected: gob as u32,
                    });
                }
                let _gob_frame_id = br.read_bits(2)?;
                let quant_scale = br.read_bits(5)?;
                if quant_scale == 0 {
                    return Err(ShortHeaderError::ZeroGobQuant);
                }
                running_qp = quant_scale;
                // §7.6.5: with a GOB header present, every macroblock of
                // the earlier GOBs is outside the current GOB → not a
                // valid predictor candidate.
                let first_row = (gob * per_gob) / mb_width;
                for r in 0..first_row {
                    for c in 0..mb_width {
                        grid.record_absent(r, c).expect("grid coordinates in range");
                    }
                }
            }
        }
        for k in 0..per_gob {
            let idx = gob * per_gob + k;
            let (mb_row, mb_col) = (idx / mb_width, idx % mb_width);
            let header = parse_short_mb_header(br, pic.coding_type)?;
            if header.not_coded {
                grid.record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })
                    .expect("grid coordinates in range");
                out.push(PVopMbContent::Inter {
                    motion: PvopMbMotion::Skipped,
                    residual: InterMacroblock::zero(),
                });
                continue;
            }
            if !is_p && !header.mb_type.is_intra() {
                return Err(ShortHeaderError::BadMbType(header.mb_type.as_u8()));
            }
            if let Some(d) = header.dquant {
                running_qp = (running_qp as i64 + i64::from(d)).clamp(1, 31) as u32;
            }
            let coded = pattern_code(header.cbpy, header.cbpc);
            if header.mb_type.is_intra() {
                // §7.6.5: an intra macroblock is a zero-vector candidate.
                grid.record_one_mv(mb_row, mb_col, MotionVector { x: 0, y: 0 })
                    .expect("grid coordinates in range");
                let mut blocks = [[[0i32; 8]; 8]; 6];
                for (i, block) in blocks.iter_mut().enumerate() {
                    let dc = read_intra_dc(br)?;
                    let events = if coded[i] {
                        decode_ac_events_short_video_header(br, TcoefTable::Inter)?
                    } else {
                        Vec::new()
                    };
                    *block =
                        intra_block(dc, &events, running_qp, DcComponent::from_block_index(i))?;
                }
                let mut luma = [[0i32; 16]; 16];
                for (b, &(row_off, col_off)) in [(0usize, 0usize), (0, 8), (8, 0), (8, 8)]
                    .iter()
                    .enumerate()
                {
                    for y in 0..8 {
                        luma[row_off + y][col_off..col_off + 8].copy_from_slice(&blocks[b][y]);
                    }
                }
                let mb = IntraMacroblock {
                    luma,
                    cb: blocks[4],
                    cr: blocks[5],
                };
                out.push(PVopMbContent::Intra(reconstruct_intra_macroblock(&mb, 8)));
            } else {
                // One motion_vector("forward") under f_code 1 against the
                // §7.6.5 block-0 median.
                let candidates = grid
                    .predictor_candidates(mb_row, mb_col, 0)
                    .expect("grid coordinates in range");
                let predictor = predict_motion_vector(candidates);
                let delta = decode_motion_vector_delta(br, MvMode::Forward, 1)?;
                let mv = reconstruct_motion_vector(delta, predictor.x, predictor.y, 1)?;
                grid.record_one_mv(mb_row, mb_col, mv)
                    .expect("grid coordinates in range");
                let ctx = MacroblockTextureContext {
                    quantiser_scale: running_qp,
                    bits_per_pixel: 8,
                    quant_type: false,
                    ac_pred_flag: false,
                    alternate_vertical_scan: false,
                    intra_mismatch_exempt: false,
                };
                let mut residual = InterMacroblock::zero();
                for (i, &is_coded) in coded.iter().enumerate() {
                    if !is_coded {
                        continue;
                    }
                    let events = decode_ac_events_short_video_header(br, TcoefTable::Inter)?;
                    let block = inter_block_from_events(
                        &events,
                        ctx,
                        &no_matrix,
                        DcComponent::from_block_index(i),
                    )?;
                    match i {
                        0..=3 => {
                            let (row_off, col_off) = (8 * (i / 2), 8 * (i % 2));
                            for (y, row) in block.iter().enumerate() {
                                residual.luma[row_off + y][col_off..col_off + 8]
                                    .copy_from_slice(row);
                            }
                        }
                        4 => residual.cb = block,
                        _ => residual.cr = block,
                    }
                }
                out.push(PVopMbContent::Inter {
                    motion: PvopMbMotion::OneMv(mv),
                    residual,
                });
            }
        }
    }
    Ok(out)
}

/// The intra macroblocks of a decoded I picture, for
/// [`crate::sequence::SequenceDecoder::push_i_vop`].
pub fn intra_macroblocks(entries: &[PVopMbContent]) -> Option<Vec<ReconstructedMacroblock>> {
    entries
        .iter()
        .map(|e| match e {
            PVopMbContent::Intra(mb) => Some(mb.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_formats_round_trip_and_tile_the_picture() {
        for f in [
            SourceFormat::SubQcif,
            SourceFormat::Qcif,
            SourceFormat::Cif,
            SourceFormat::Cif4,
            SourceFormat::Cif16,
        ] {
            assert_eq!(SourceFormat::from_code(f.code()), Some(f));
            let (w, h) = f.dimensions();
            assert_eq!(SourceFormat::from_dimensions(w.into(), h.into()), Some(f));
            let (mbw, mbh) = f.mb_dimensions();
            assert_eq!(f.macroblocks_per_gob() * f.gobs_per_picture(), mbw * mbh);
            // A GOB always starts at column 0.
            assert_eq!(f.macroblocks_per_gob() % mbw, 0);
        }
        assert_eq!(SourceFormat::from_code(0), None);
        assert_eq!(SourceFormat::from_dimensions(64, 64), None);
    }

    #[test]
    fn picture_start_detection_is_unambiguous() {
        assert!(is_short_header_picture_start(&[0, 0, 0x80, 0], 0));
        assert!(is_short_header_picture_start(&[0, 0, 0x83, 0], 0));
        // GOB header (gob_number 1) and end marker are not starts.
        assert!(!is_short_header_picture_start(&[0, 0, 0x84, 0], 0));
        assert!(!is_short_header_picture_start(&[0, 0, 0xFC, 0], 0));
        // MPEG-4 start codes are not starts either.
        assert!(!is_short_header_picture_start(&[0, 0, 1, 0xB6], 0));
        assert_eq!(
            scan_short_header_pictures(&[0, 0, 0x80, 5, 0, 0, 0x81, 6]),
            vec![0, 4]
        );
    }

    #[test]
    fn intra_dc_code_maps_128_to_255() {
        assert_eq!(intra_dc_code(1), 1);
        assert_eq!(intra_dc_code(127), 127);
        assert_eq!(intra_dc_code(128), 255);
        assert_eq!(intra_dc_code(254), 254);
    }
}

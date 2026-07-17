//! §6.2.5.3 data-partitioned macroblock-layer parsing
//! (`data_partitioned_motion_shape_texture()`).
//!
//! When the VOL header's `data_partitioned` flag is set (§6.3.3), each
//! video packet rearranges its macroblock data into *partitions*
//! separated by a unique marker, so a bit error in one partition does
//! not corrupt the others. The §6.2.5.3 syntax defines two rectangular
//! layouts:
//!
//! * **`data_partitioned_i_vop()`** — partition 1 carries every
//!   macroblock's `mcbpc` (+ `dquant` + intra-DC when
//!   `use_intra_dc_vlc`); it is terminated by the 19-bit
//!   `dc_marker` `110 1011 0000 0000 0001`. Partition 2 then carries
//!   `ac_pred_flag` + `cbpy` for every macroblock, and partition 3 the
//!   `block()` texture data for every macroblock.
//! * **`data_partitioned_p_vop()`** — partition 1 carries every
//!   macroblock's `not_coded` + `mcbpc` (+ `mcsel` + `motion_coding`),
//!   terminated by the 17-bit `motion_marker`
//!   `1 1111 0000 0000 0001`. Partition 2 then carries
//!   `ac_pred_flag` (intra MBs only) + `cbpy` + `dquant` + intra-DC,
//!   and partition 3 the `block()` texture data.
//!
//! This module parses the *partition structure* — walking the
//! macroblock headers across the three partitions, detecting the
//! `dc_marker` / `motion_marker`, and re-assembling each macroblock's
//! header fields scattered across the partitions into one
//! [`DataPartitionedMb`] record per macroblock. The DC values gathered
//! in partition 1 (I-VOP) and the texture `block()` data in partition 3
//! are handed back to the caller as bit-region offsets / decoded values
//! so the existing [`crate::block`] / [`crate::texture`] machinery can
//! finish the reconstruction.
//!
//! Provenance: every field and bit layout traces to ISO/IEC
//! 14496-2:2004 (3rd edition) §6.2.5.3 / §6.3.5, read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`. The
//! `dc_marker` / `motion_marker` bit strings are the §6.3.5 verbatim
//! definitions (lines 4802 / 4878 of that text).

use crate::bitreader::BitReader;
use crate::macroblock::{
    decode_cbpy4, decode_mcbpc, DerivedMbType, MacroblockParseError, MCBPC_I, MCBPC_P,
};
use crate::texture::{decode_intra_dc, DcComponent, TextureParseError};

/// §6.3.5 `dc_marker` — `110 1011 0000 0000 0001`, 19 bits. Inserted
/// after the §6.2.5.3 first partition of a data-partitioned I-VOP.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub const DC_MARKER: u32 = 0b110_1011_0000_0000_0001;
/// Length of [`DC_MARKER`] in bits.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub const DC_MARKER_BITS: usize = 19;

/// §6.3.5 `motion_marker` — `1 1111 0000 0000 0001`, 17 bits. Inserted
/// after the §6.2.5.3 first partition of a data-partitioned P-VOP.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub const MOTION_MARKER: u32 = 0b1_1111_0000_0000_0001;
/// Length of [`MOTION_MARKER`] in bits.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub const MOTION_MARKER_BITS: usize = 17;

/// Errors emitted while parsing a data-partitioned macroblock layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataPartitionError {
    /// The bit reader ran out mid-partition.
    Truncated,
    /// A macroblock-header field failed to decode (propagated from the
    /// shared §6.2.6 macroblock primitives).
    Macroblock(MacroblockParseError),
    /// An intra-DC field failed to decode (propagated from
    /// [`crate::texture`]).
    Texture(TextureParseError),
    /// The expected partition marker (`dc_marker` / `motion_marker`) was
    /// not found within the macroblock count for the video packet. The
    /// number of macroblocks decoded before the search gave up is
    /// reported for diagnostics.
    MarkerNotFound {
        /// How many macroblocks were decoded before the marker search
        /// exhausted `mb_in_video_packet`.
        decoded: usize,
    },
    /// Data partitioning is not defined for B-VOPs (§6.2.5.3 NOTE: "Data
    /// partitioning is not supported in B-VOPs"). The combined
    /// `combined_motion_shape_texture()` path applies instead.
    UnsupportedBVop,
    /// Non-rectangular shape is out of scope for this parser.
    UnsupportedShape,
}

impl core::fmt::Display for DataPartitionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DataPartitionError::Truncated => write!(f, "data-partitioned layer truncated"),
            DataPartitionError::Macroblock(e) => write!(f, "macroblock field: {e}"),
            DataPartitionError::Texture(e) => write!(f, "intra-DC field: {e}"),
            DataPartitionError::MarkerNotFound { decoded } => {
                write!(f, "partition marker not found after {decoded} macroblocks")
            }
            DataPartitionError::UnsupportedBVop => {
                write!(f, "data partitioning is not supported in B-VOPs")
            }
            DataPartitionError::UnsupportedShape => {
                write!(f, "data-partitioned parsing requires rectangular shape")
            }
        }
    }
}

impl std::error::Error for DataPartitionError {}

impl From<crate::bitreader::BitReaderError> for DataPartitionError {
    fn from(_: crate::bitreader::BitReaderError) -> Self {
        DataPartitionError::Truncated
    }
}

impl From<MacroblockParseError> for DataPartitionError {
    fn from(e: MacroblockParseError) -> Self {
        DataPartitionError::Macroblock(e)
    }
}

impl From<TextureParseError> for DataPartitionError {
    fn from(e: TextureParseError) -> Self {
        DataPartitionError::Texture(e)
    }
}

/// §6.2.5.3 / Table 6-25 derivation of the internal `use_intra_dc_vlc`
/// flag from `intra_dc_vlc_thr` and the running quantiser scale.
///
/// Table 6-25 maps `intra_dc_vlc_thr` to a quantiser-scale threshold at
/// or above which intra DC coefficients switch from the dedicated DC VLC
/// (Tables B.13–B.15) to the intra AC VLC (Table 6-20):
///
/// | `intra_dc_vlc_thr` | switch to AC VLC at running Qp ≥ |
/// |--------------------|----------------------------------|
/// | 0                  | never (DC VLC for the entire VOP)|
/// | 1                  | 13                               |
/// | 2                  | 15                               |
/// | 3                  | 17                               |
/// | 4                  | 19                               |
/// | 5                  | 21                               |
/// | 6                  | 23                               |
/// | 7                  | 1 (AC VLC for the entire VOP)    |
///
/// `use_intra_dc_vlc` is `true` when the running quantiser scale is
/// *below* that threshold (so the DC VLC is still in force).
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn use_intra_dc_vlc(intra_dc_vlc_thr: u8, running_qp: u32) -> bool {
    let threshold: u32 = match intra_dc_vlc_thr & 0b111 {
        0 => u32::MAX, // DC VLC for the entire VOP
        1 => 13,
        2 => 15,
        3 => 17,
        4 => 19,
        5 => 21,
        6 => 23,
        _ => 1, // 7: AC VLC for the entire VOP
    };
    running_qp < threshold
}

/// One macroblock's worth of partition-1 header fields gathered from the
/// motion / shape partition of a data-partitioned VOP.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub struct DataPartitionedMb {
    /// `not_coded` (P-VOP only; always `false` for I-VOP).
    pub not_coded: bool,
    /// Derived macroblock type. `None` only for a `not_coded` P-VOP MB.
    pub mb_type: Option<DerivedMbType>,
    /// 2-bit chroma coded-block pattern from `mcbpc`.
    pub cbpc: u8,
    /// `dquant` differential (Table 6-32 ±1 / ±2). I-VOP partition 1
    /// carries `dquant` (when `mb_type == 4`); the P-VOP carries it in
    /// partition 2 instead, so this is `None` for a P-VOP record here.
    pub dquant_delta: Option<i8>,
    /// `mcsel` (P-VOP S(GMC) only). `None` otherwise.
    pub mcsel: Option<bool>,
    /// The six intra-DC differentials gathered in I-VOP partition 1 when
    /// `use_intra_dc_vlc`. Index 0..4 luminance, 4..6 chrominance.
    /// `None` when DC values are coded as AC (deferred to partition 3)
    /// or the MB is not intra.
    pub intra_dc: Option<[i32; 6]>,
}

/// The second-partition fields for one macroblock: `ac_pred_flag` and
/// `cbpy` (plus `dquant` + intra-DC for the P-VOP, which the I-VOP
/// carries in partition 1).
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub struct DataPartitionedTexHeader {
    /// `ac_pred_flag` (intra MBs only; `false` for inter MBs).
    pub ac_pred_flag: bool,
    /// `cbpy` for the four luminance blocks.
    pub cbpy: u8,
    /// `dquant` (P-VOP partition 2 only; `None` for I-VOP).
    pub dquant_delta: Option<i8>,
    /// Intra-DC differentials (P-VOP partition 2, intra MBs with
    /// `use_intra_dc_vlc`; `None` otherwise).
    pub intra_dc: Option<[i32; 6]>,
}

/// Per-macroblock partition-1 event handed to the caller's closure by
/// [`parse_data_partitioned_p_vop`], in raster order. The closure sees
/// **every** macroblock of the packet — not just the ones carrying
/// motion — so it can thread the §7.6.5 predictor grid exactly as the
/// combined-syntax walk does (a skipped MB records a valid zero
/// vector, an intra MB a valid zero candidate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub enum DpMbEvent {
    /// `not_coded == 1` — a skipped macroblock (zero-MV inter copy).
    /// No motion bits follow.
    NotCoded,
    /// An intra / intra+q macroblock — no motion bits.
    Intra,
    /// A GMC-selected macroblock (S(GMC) with `mcsel == 1`) — no local
    /// motion bits.
    Gmc,
    /// A coded inter macroblock carrying `motion_coding()`: the
    /// closure must consume the §6.2.6.2 motion-vector bodies (one for
    /// types 0/1, four for the inter-4V type 2).
    Motion(DerivedMbType),
}

/// The decode of a data-partitioned I-VOP video packet's header
/// partitions: per-MB partition-1 records, plus per-MB partition-2
/// (`ac_pred_flag` + `cbpy`) records. The block (texture) partition that
/// follows is left for the caller to decode from `texture_start_bit`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub struct DataPartitionedIVop {
    /// One [`DataPartitionedMb`] per macroblock, in raster order.
    pub mbs: Vec<DataPartitionedMb>,
    /// One [`DataPartitionedTexHeader`] per macroblock, in raster order.
    pub tex_headers: Vec<DataPartitionedTexHeader>,
    /// Absolute bit position at the start of the third (block / texture)
    /// partition, ready for the caller's `block()` decode.
    pub texture_start_bit: usize,
}

/// The decode of a data-partitioned P-VOP video packet's header
/// partitions.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub struct DataPartitionedPVop {
    /// One [`DataPartitionedMb`] per macroblock (partition 1: not_coded,
    /// mcbpc, mcsel — motion vectors are skipped over but their bit
    /// span is not surfaced here).
    pub mbs: Vec<DataPartitionedMb>,
    /// One [`DataPartitionedTexHeader`] per macroblock (partition 2:
    /// ac_pred_flag, cbpy, dquant, intra-DC).
    pub tex_headers: Vec<DataPartitionedTexHeader>,
    /// Absolute bit position at the start of the third (block / texture)
    /// partition.
    pub texture_start_bit: usize,
}

/// Test whether the next [`DC_MARKER_BITS`] bits equal the
/// §6.3.5 `dc_marker`.
fn at_dc_marker(br: &BitReader<'_>) -> bool {
    br.remaining_bits() >= DC_MARKER_BITS
        && br.next_bits(DC_MARKER_BITS).map(|w| w == DC_MARKER) == Ok(true)
}

/// Test whether the next [`MOTION_MARKER_BITS`] bits equal the
/// §6.3.5 `motion_marker`.
fn at_motion_marker(br: &BitReader<'_>) -> bool {
    br.remaining_bits() >= MOTION_MARKER_BITS
        && br.next_bits(MOTION_MARKER_BITS).map(|w| w == MOTION_MARKER) == Ok(true)
}

/// Decode the six intra-DC differentials of an intra macroblock (4
/// luminance + 2 chrominance), each via [`decode_intra_dc`] with its
/// `> 8 → marker_bit` consumption. Returns the differentials in block
/// order (0..4 luma, 4..6 chroma).
fn decode_intra_dc_six(br: &mut BitReader<'_>) -> Result<[i32; 6], DataPartitionError> {
    let mut dc = [0i32; 6];
    for slot in dc.iter_mut().take(4) {
        let d = decode_intra_dc(br, DcComponent::Luminance)?;
        *slot = d.differential;
    }
    for slot in dc.iter_mut().skip(4) {
        let d = decode_intra_dc(br, DcComponent::Chrominance)?;
        *slot = d.differential;
    }
    Ok(dc)
}

/// Consume the §6.2.5.2 `motion_coding("forward", type_of_mb)` MV-delta
/// field group of a data-partitioned P-VOP macroblock, returning the raw
/// forward MV deltas.
///
/// `motion_coding(mode, type_of_mb)` reads one `motion_vector(mode)`,
/// plus three more when `type_of_mb == 2` (the inter-4V macroblock with
/// four 8×8 block vectors). Each `motion_vector` body is decoded by
/// [`crate::motion::decode_motion_vector_delta`] with the forward mode
/// and `vop_fcode_forward`. The returned deltas are *not* yet combined
/// with a predictor — the §7.6.5 median predictor + §7.6.3 reconstruction
/// is the caller's responsibility (it needs the running predictor grid),
/// matching how the combined path separates delta decode from
/// reconstruction. This is a ready-made `decode_motion` closure body for
/// [`parse_data_partitioned_p_vop`].
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn decode_motion_coding(
    br: &mut BitReader<'_>,
    mb_type: DerivedMbType,
    vop_fcode_forward: u8,
) -> Result<Vec<crate::motion::MotionVectorDelta>, crate::motion::MotionParseError> {
    use crate::motion::{decode_motion_vector_delta, MvMode};
    let count = if matches!(mb_type, DerivedMbType::Inter4V) {
        4
    } else {
        1
    };
    let mut deltas = Vec::with_capacity(count);
    for _ in 0..count {
        deltas.push(decode_motion_vector_delta(
            br,
            MvMode::Forward,
            vop_fcode_forward,
        )?);
    }
    Ok(deltas)
}

/// Derive the §E.1.4.4 texture-partition [`MbBlockLayout`] for one
/// data-partitioned macroblock from its parsed partition-1 record and
/// partition-2 texture header.
///
/// The texture partition (partition 3) of a data-partitioned VOP carries
/// the `block()` AC-coefficient runs for every coded block, in §6.3.5
/// block order (luma 0..3 then chroma 4..5). A block is *coded* — i.e.
/// contributes a Tcoef EVENT run to the texture partition — when its
/// §6.3.7 pattern bit is set: the `cbpy` luminance bits (block 0..3) and
/// the `cbpc` chrominance bits from `mcbpc` (block 4..5). Each coded
/// block uses Table B.16 (intra) or Table B.17 (inter) depending on
/// whether the macroblock is intra (`derived_mb_type >= 3`).
///
/// A `not_coded` P-VOP macroblock contributes no texture blocks (empty
/// layout). Note that in data-partitioned mode the intra DC is carried in
/// partition 1 / 2 (not the texture partition), so the texture-partition
/// EVENT run of an intra block is its AC coefficients only — but the
/// Tcoef table selection (intra vs. inter) is unchanged.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn mb_block_layout(
    mb: &DataPartitionedMb,
    tex: &DataPartitionedTexHeader,
) -> crate::rvlc_recovery::MbBlockLayout {
    use crate::rvlc_recovery::MbBlockLayout;
    use crate::texture::TcoefTable;

    if mb.not_coded {
        return MbBlockLayout::empty();
    }
    let is_intra = mb.mb_type.map(|t| t.is_intra()).unwrap_or(false);
    let table = if is_intra {
        TcoefTable::Intra
    } else {
        TcoefTable::Inter
    };
    // §6.3.7 pattern: cbpy → luma blocks 0..3, cbpc → chroma blocks 4..5.
    let coded = crate::block::pattern_code(tex.cbpy, mb.cbpc);
    let blocks: Vec<TcoefTable> = coded.iter().filter(|&&c| c).map(|_| table).collect();
    MbBlockLayout { blocks }
}

/// Parse the §6.2.5.3 `data_partitioned_i_vop()` body of one video
/// packet (rectangular shape).
///
/// `mb_in_video_packet` bounds the partition-1 macroblock walk: the
/// `do { … } while (next_bits() != dc_marker)` loop consumes at most
/// this many macroblocks before the marker must appear. (Stuffing
/// macroblocks — `derived_mb_type == 5` in `mcbpc` — are consumed
/// transparently and do *not* count toward the limit, per NOTE 1.)
///
/// `intra_dc_vlc_thr` + `running_qp` derive `use_intra_dc_vlc`
/// (Table 6-25) which gates whether partition 1 also carries the six
/// intra-DC differentials per macroblock.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn parse_data_partitioned_i_vop(
    br: &mut BitReader<'_>,
    mb_in_video_packet: usize,
    intra_dc_vlc_thr: u8,
    running_qp: u32,
) -> Result<DataPartitionedIVop, DataPartitionError> {
    let dc_vlc = use_intra_dc_vlc(intra_dc_vlc_thr, running_qp);

    // ---- Partition 1: mcbpc (+ dquant + intra-DC), until dc_marker ----
    let mut mbs: Vec<DataPartitionedMb> = Vec::new();
    loop {
        if at_dc_marker(br) {
            break;
        }
        if mbs.len() >= mb_in_video_packet {
            return Err(DataPartitionError::MarkerNotFound { decoded: mbs.len() });
        }
        // mcbpc (transparently consuming stuffing entries).
        let (_len, raw, cbpc) = loop {
            let (len, raw, cbpc) = decode_mcbpc(br, MCBPC_I)?;
            if raw == 5 {
                // Stuffing — read the next mcbpc.
                continue;
            }
            break (len, raw, cbpc);
        };
        let mb_type = DerivedMbType::from_raw(raw).ok_or(DataPartitionError::Macroblock(
            MacroblockParseError::Truncated,
        ))?;

        // mb_type == 4 → dquant (§6.2.5.3: `if (mb_type == 4) dquant`).
        let dquant_delta = if raw == 4 {
            let code = br.read_bits(2)? as u8;
            Some(crate::macroblock::dquant_value(code))
        } else {
            None
        };

        // use_intra_dc_vlc → the six intra-DC differentials.
        let intra_dc = if dc_vlc {
            Some(decode_intra_dc_six(br)?)
        } else {
            None
        };

        mbs.push(DataPartitionedMb {
            not_coded: false,
            mb_type: Some(mb_type),
            cbpc,
            dquant_delta,
            mcsel: None,
            intra_dc,
        });
    }
    // Consume the dc_marker.
    br.skip_bits(DC_MARKER_BITS)?;

    // ---- Partition 2: ac_pred_flag + cbpy, for every macroblock ----
    let mut tex_headers: Vec<DataPartitionedTexHeader> = Vec::with_capacity(mbs.len());
    for _ in 0..mbs.len() {
        let ac_pred_flag = br.read_bool()?;
        let (_clen, cbpy_intra, _cbpy_inter) = decode_cbpy4(br)?;
        // I-VOP MBs are all intra, so the intra cbpy column applies.
        tex_headers.push(DataPartitionedTexHeader {
            ac_pred_flag,
            cbpy: cbpy_intra,
            dquant_delta: None,
            intra_dc: None,
        });
    }

    let texture_start_bit = br.bit_position();
    Ok(DataPartitionedIVop {
        mbs,
        tex_headers,
        texture_start_bit,
    })
}

/// Parse the §6.2.5.3 `data_partitioned_p_vop()` partition-1 (motion /
/// shape) and partition-2 (ac_pred + cbpy + dquant + intra-DC) bodies of
/// one video packet (rectangular shape).
///
/// `decode_motion` is a caller-supplied closure that consumes the
/// `motion_coding()` field group for a coded inter macroblock from the
/// reader (the §6.2.5.2 `motion_coding(mode, type_of_mb)` syntax — one
/// motion vector, four for `type_of_mb == 2`). It is invoked when the
/// macroblock is coded, not GMC-selected, and its type carries motion
/// (`derived_mb_type < 2 || derived_mb_type == 2`). The closure returns
/// `Ok(())` on success; any error aborts the parse. Passing a closure
/// keeps the motion-vector decode (which depends on the predictor grid
/// state the caller maintains) outside this structural parser.
///
/// `is_s_gmc` selects the §6.3.6 `mcsel` syntax (`sprite_enable ==
/// "GMC"` S-VOP). `intra_dc_vlc_thr` + `running_qp` derive
/// `use_intra_dc_vlc` for the partition-2 intra-DC of intra MBs.
#[doc(hidden)] // internal decode plumbing, not the crate's stable public API
pub fn parse_data_partitioned_p_vop<F>(
    br: &mut BitReader<'_>,
    mb_in_video_packet: usize,
    is_s_gmc: bool,
    intra_dc_vlc_thr: u8,
    running_qp: u32,
    mut decode_motion: F,
) -> Result<DataPartitionedPVop, DataPartitionError>
where
    F: FnMut(&mut BitReader<'_>, DpMbEvent) -> Result<(), DataPartitionError>,
{
    let dc_vlc = use_intra_dc_vlc(intra_dc_vlc_thr, running_qp);

    // ---- Partition 1: not_coded + mcbpc + mcsel + motion, until marker ----
    let mut mbs: Vec<DataPartitionedMb> = Vec::new();
    loop {
        if at_motion_marker(br) {
            break;
        }
        if mbs.len() >= mb_in_video_packet {
            return Err(DataPartitionError::MarkerNotFound { decoded: mbs.len() });
        }

        let not_coded = br.read_bool()?;
        if not_coded {
            decode_motion(br, DpMbEvent::NotCoded)?;
            mbs.push(DataPartitionedMb {
                not_coded: true,
                mb_type: None,
                cbpc: 0,
                dquant_delta: None,
                // §6.3.6: a not-coded S(GMC) MB has implied mcsel == 1.
                mcsel: if is_s_gmc { Some(true) } else { None },
                intra_dc: None,
            });
            continue;
        }

        // mcbpc (consuming stuffing entries transparently).
        let (_len, raw, cbpc) = loop {
            let (len, raw, cbpc) = decode_mcbpc(br, MCBPC_P)?;
            if raw == 5 {
                continue;
            }
            break (len, raw, cbpc);
        };
        let mb_type = DerivedMbType::from_raw(raw).ok_or(DataPartitionError::Macroblock(
            MacroblockParseError::Truncated,
        ))?;

        // §6.3.6 mcsel: present iff S(GMC) and derived_mb_type < 2.
        let mcsel = if is_s_gmc && raw < 2 {
            Some(br.read_bool()?)
        } else {
            None
        };

        // motion_coding(): present when the MB carries motion, i.e.
        //   !(S(GMC) && mcsel) && derived_mb_type < 2  ||  derived_mb_type == 2.
        let gmc_selected = mcsel == Some(true);
        let carries_motion = (!gmc_selected && raw < 2) || raw == 2;
        if carries_motion {
            decode_motion(br, DpMbEvent::Motion(mb_type))?;
        } else if gmc_selected {
            decode_motion(br, DpMbEvent::Gmc)?;
        } else {
            // Intra / intra+q — no motion bits, but the closure still
            // sees the macroblock for predictor bookkeeping.
            decode_motion(br, DpMbEvent::Intra)?;
        }

        mbs.push(DataPartitionedMb {
            not_coded: false,
            mb_type: Some(mb_type),
            cbpc,
            dquant_delta: None,
            mcsel,
            intra_dc: None,
        });
    }
    br.skip_bits(MOTION_MARKER_BITS)?;

    // ---- Partition 2: ac_pred + cbpy + dquant + intra-DC ----
    let mut tex_headers: Vec<DataPartitionedTexHeader> = Vec::with_capacity(mbs.len());
    for mb in &mbs {
        if mb.not_coded {
            // §6.2.5.3 partition-2 loop body is `if (!not_coded) { … }`.
            tex_headers.push(DataPartitionedTexHeader {
                ac_pred_flag: false,
                cbpy: 0,
                dquant_delta: None,
                intra_dc: None,
            });
            continue;
        }
        let mb_type = mb.mb_type.expect("coded MB has a type");
        let raw = mb_type.as_u8();

        // ac_pred_flag — only for intra (derived_mb_type >= 3).
        let ac_pred_flag = if raw >= 3 { br.read_bool()? } else { false };

        // cbpy (intra column for intra MBs, inter column otherwise).
        let (_clen, cbpy_intra, cbpy_inter) = decode_cbpy4(br)?;
        let cbpy = if raw >= 3 { cbpy_intra } else { cbpy_inter };

        // dquant — derived_mb_type == 1 || == 4.
        let dquant_delta = if raw == 1 || raw == 4 {
            let code = br.read_bits(2)? as u8;
            Some(crate::macroblock::dquant_value(code))
        } else {
            None
        };

        // intra-DC — derived_mb_type >= 3 && use_intra_dc_vlc.
        let intra_dc = if raw >= 3 && dc_vlc {
            Some(decode_intra_dc_six(br)?)
        } else {
            None
        };

        tex_headers.push(DataPartitionedTexHeader {
            ac_pred_flag,
            cbpy,
            dquant_delta,
            intra_dc,
        });
    }

    let texture_start_bit = br.bit_position();
    Ok(DataPartitionedPVop {
        mbs,
        tex_headers,
        texture_start_bit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build bytes from an MSB-first `'0'`/`'1'` string (whitespace
    /// ignored), zero-padded to a byte.
    fn bits(s: &str) -> Vec<u8> {
        let cleaned: String = s.chars().filter(|c| *c == '0' || *c == '1').collect();
        let mut out = Vec::new();
        let mut cur = 0u8;
        let mut n = 0u8;
        for c in cleaned.chars() {
            cur = (cur << 1) | if c == '1' { 1 } else { 0 };
            n += 1;
            if n == 8 {
                out.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            cur <<= 8 - n;
            out.push(cur);
        }
        out
    }

    #[test]
    fn markers_have_documented_values() {
        // §6.3.5: dc_marker = 110 1011 0000 0000 0001 (19 bits).
        assert_eq!(DC_MARKER, 0b110_1011_0000_0000_0001);
        assert_eq!(DC_MARKER_BITS, 19);
        // motion_marker = 1 1111 0000 0000 0001 (17 bits).
        assert_eq!(MOTION_MARKER, 0b1_1111_0000_0000_0001);
        assert_eq!(MOTION_MARKER_BITS, 17);
    }

    #[test]
    fn use_intra_dc_vlc_table_6_25() {
        // thr 0 → DC VLC for the entire VOP (always true).
        assert!(use_intra_dc_vlc(0, 31));
        // thr 7 → AC VLC for the entire VOP (threshold 1 → false for Qp>=1).
        assert!(!use_intra_dc_vlc(7, 1));
        assert!(!use_intra_dc_vlc(7, 31));
        // thr 1 → switch at Qp>=13.
        assert!(use_intra_dc_vlc(1, 12));
        assert!(!use_intra_dc_vlc(1, 13));
        // thr 4 → switch at Qp>=19.
        assert!(use_intra_dc_vlc(4, 18));
        assert!(!use_intra_dc_vlc(4, 19));
        // thr 6 → switch at Qp>=23.
        assert!(use_intra_dc_vlc(6, 22));
        assert!(!use_intra_dc_vlc(6, 23));
    }

    #[test]
    fn at_dc_marker_detects_exactly() {
        let data = bits("110 1011 0000 0000 0001 0");
        let br = BitReader::new(&data);
        assert!(at_dc_marker(&br));
        // One bit off → not a marker.
        let off = bits("010 1011 0000 0000 0001 0");
        let br2 = BitReader::new(&off);
        assert!(!at_dc_marker(&br2));
    }

    #[test]
    fn i_vop_two_intra_mbs_no_dc_vlc() {
        // Two intra MBs (mb_type 3, cbpc 00 → mcbpc I code "1"), no
        // dquant (mb_type != 4), use_intra_dc_vlc == false (thr 7), so
        // partition 1 is just two "1" mcbpc codes, then the dc_marker,
        // then partition 2's two (ac_pred_flag, cbpy) pairs.
        //
        // mcbpc I Table B.6: code "1" → mb_type 3, cbpc 00.
        // cbpy intra "1111" (all-AC) → Table B.8 code for cbpy=15.
        //   Table B.8: cbpy 15 (1111) intra → code "11" (len 2).
        // ac_pred_flag = 0.
        let stream = bits(concat!(
            // partition 1: two mcbpc "1" codes
            "1",
            "1",
            // dc_marker
            "110 1011 0000 0000 0001",
            // partition 2: (ac_pred=0, cbpy code for 1111 = "11") x2
            "0",
            "11",
            "0",
            "11",
            // a little texture-partition padding
            "0000"
        ));
        let mut br = BitReader::new(&stream);
        let out = parse_data_partitioned_i_vop(&mut br, 4, 7, 31).unwrap();
        assert_eq!(out.mbs.len(), 2);
        assert_eq!(out.tex_headers.len(), 2);
        for mb in &out.mbs {
            assert_eq!(mb.mb_type, Some(DerivedMbType::Intra));
            assert_eq!(mb.cbpc, 0);
            assert!(mb.intra_dc.is_none());
        }
        for th in &out.tex_headers {
            assert!(!th.ac_pred_flag);
            assert_eq!(th.cbpy, 0b1111);
        }
        // texture_start_bit is past partition 1 (2 bits) + dc_marker (19)
        // + partition 2 ((1+2)*2 = 6 bits) = 27.
        assert_eq!(out.texture_start_bit, 2 + 19 + 6);
    }

    #[test]
    fn i_vop_marker_not_found_errors() {
        // Partition 1 never reaches a dc_marker within mb_in_video_packet.
        let stream = bits("1 1 1 1 0000 0000");
        let mut br = BitReader::new(&stream);
        let err = parse_data_partitioned_i_vop(&mut br, 2, 7, 31).unwrap_err();
        assert!(matches!(
            err,
            DataPartitionError::MarkerNotFound { decoded: 2 }
        ));
    }

    #[test]
    fn p_vop_with_concrete_motion_coding_closure() {
        // One coded inter MB whose motion is decoded by the stock
        // decode_motion_coding helper (fcode 1 → no residuals; mv_data
        // code "1" → 0). motion_coding(forward, inter) = mvd_x "1" + mvd_y
        // "1" = "11" (delta 0,0).
        let stream = bits(concat!(
            // partition 1
            "0",  // not_coded = 0
            "1",  // mcbpc P "1" → inter (type 0)
            "11", // motion_coding: mvd_x "1", mvd_y "1" (both 0)
            // motion_marker
            "1 1111 0000 0000 0001",
            // partition 2: inter MB → cbpy "11" (inter column 0000)
            "11",
            "0000",
        ));
        let mut br = BitReader::new(&stream);
        let mut captured = Vec::new();
        let out = parse_data_partitioned_p_vop(&mut br, 4, false, 7, 31, |b, ev| {
            let DpMbEvent::Motion(ty) = ev else {
                return Ok(());
            };
            let deltas = decode_motion_coding(b, ty, 1)
                .map_err(|_| DataPartitionError::Macroblock(MacroblockParseError::Truncated))?;
            captured = deltas;
            Ok(())
        })
        .unwrap();
        assert_eq!(out.mbs.len(), 1);
        assert_eq!(out.mbs[0].mb_type, Some(DerivedMbType::Inter));
        // One forward MV delta (0, 0).
        assert_eq!(captured.len(), 1);
        assert_eq!((captured[0].dx, captured[0].dy), (0, 0));
        assert_eq!(out.tex_headers[0].cbpy, 0); // inter column of "11"
    }

    #[test]
    fn decode_motion_coding_inter4v_reads_four() {
        // Inter4V (type 2): motion_coding reads four MVs. fcode 1, all
        // mv_data "1" → four (0,0) deltas, 8 bits total.
        let stream = bits("11 11 11 11 0000");
        let mut br = BitReader::new(&stream);
        let deltas = decode_motion_coding(&mut br, DerivedMbType::Inter4V, 1).unwrap();
        assert_eq!(deltas.len(), 4);
        for d in &deltas {
            assert_eq!((d.dx, d.dy), (0, 0));
        }
        assert_eq!(br.bit_position(), 8);
    }

    #[test]
    fn i_vop_data_partitioned_to_rvlc_recovery_end_to_end() {
        use crate::rvlc_recovery::{recover_video_packet_dct, RvlcRecovery};
        // Build a data-partitioned, reversible-VLC I-VOP packet with two
        // intra MBs, each with exactly one coded luma block (block 0):
        //   cbpy intra 1000 → block 0 coded only → Table B.8 code "0001 0"
        //   (5 bits, intra 1000 / inter 0111).
        // Partition 1: two mcbpc "1" (intra, cbpc 00), no dquant, no
        //   intra-DC (thr 7 → AC VLC for the VOP → use_intra_dc_vlc false).
        // dc_marker.
        // Partition 2: two (ac_pred=0, cbpy "00010") pairs.
        // Texture partition (RVLC): each block one EVENT
        //   (LAST=1,RUN=0,LEVEL=1) intra reversible = "1011" + sign "0".
        let stream = bits(concat!(
            // partition 1
            "1",
            "1",
            // dc_marker
            "110 1011 0000 0000 0001",
            // partition 2: (ac_pred 0, cbpy 00010) x2
            "0",
            "00010",
            "0",
            "00010",
            // texture partition: two RVLC EVENTs "1011 0" each
            "1011 0",
            "1011 0",
        ));
        let mut br = BitReader::new(&stream);
        let parsed = parse_data_partitioned_i_vop(&mut br, 4, 7, 31).unwrap();
        assert_eq!(parsed.mbs.len(), 2);
        // Each MB has cbpy 1000 (block 0 coded).
        for th in &parsed.tex_headers {
            assert_eq!(th.cbpy, 0b1000);
        }
        // Build layouts and run the RVLC recovery over the texture region.
        let layouts: Vec<_> = parsed
            .mbs
            .iter()
            .zip(&parsed.tex_headers)
            .map(|(mb, tex)| mb_block_layout(mb, tex))
            .collect();
        for l in &layouts {
            assert_eq!(l.blocks.len(), 1); // one coded block per MB
        }
        let end_bit = stream.len() * 8;
        let rec =
            recover_video_packet_dct(&stream, parsed.texture_start_bit, end_bit, &layouts).unwrap();
        match rec {
            RvlcRecovery::Clean { mbs } => {
                assert_eq!(mbs.len(), 2);
                for mb in &mbs {
                    assert_eq!(mb.blocks.len(), 1);
                    assert_eq!(mb.blocks[0].len(), 1);
                    let ev = mb.blocks[0][0];
                    assert_eq!((ev.last, ev.run, ev.level), (true, 0, 1));
                }
            }
            RvlcRecovery::Recovered { .. } => panic!("clean stream must not recover"),
        }
    }

    #[test]
    fn mb_block_layout_intra_and_inter() {
        use crate::texture::TcoefTable;
        // Intra MB, cbpy = 1010 (blocks 0 + 2 coded), cbpc = 01 (block 5).
        let mb = DataPartitionedMb {
            not_coded: false,
            mb_type: Some(DerivedMbType::Intra),
            cbpc: 0b01,
            dquant_delta: None,
            mcsel: None,
            intra_dc: None,
        };
        let tex = DataPartitionedTexHeader {
            ac_pred_flag: false,
            cbpy: 0b1010,
            dquant_delta: None,
            intra_dc: None,
        };
        let layout = mb_block_layout(&mb, &tex);
        // blocks 0, 2 (luma) + 5 (Cr) → 3 coded blocks, all intra table.
        assert_eq!(layout.blocks, vec![TcoefTable::Intra; 3]);

        // Inter MB, cbpy = 1111 (all four luma), cbpc = 11 (both chroma).
        let mb_i = DataPartitionedMb {
            not_coded: false,
            mb_type: Some(DerivedMbType::Inter),
            cbpc: 0b11,
            dquant_delta: None,
            mcsel: None,
            intra_dc: None,
        };
        let tex_i = DataPartitionedTexHeader {
            ac_pred_flag: false,
            cbpy: 0b1111,
            dquant_delta: None,
            intra_dc: None,
        };
        let layout_i = mb_block_layout(&mb_i, &tex_i);
        assert_eq!(layout_i.blocks, vec![TcoefTable::Inter; 6]);

        // not_coded → empty layout.
        let skipped = DataPartitionedMb {
            not_coded: true,
            mb_type: None,
            cbpc: 0,
            dquant_delta: None,
            mcsel: None,
            intra_dc: None,
        };
        assert!(mb_block_layout(&skipped, &tex_i).blocks.is_empty());
    }

    #[test]
    fn p_vop_skipped_and_one_inter() {
        // P-VOP partition 1: one not_coded MB (bit "1"), then one coded
        // inter MB (not_coded "0", mcbpc P "1" → mb_type 0 / inter,
        // cbpc 00; carries one motion vector consumed by the closure
        // which here eats a fixed 4 bits), then the motion_marker. Then
        // partition 2: skipped MB → no fields; coded inter MB → no
        // ac_pred (inter), cbpy inter, no dquant (mb_type 0), no DC.
        //
        // mcbpc P Table B.7: code "1" → mb_type 0, cbpc 00.
        // cbpy: for inter, decode the same codeword; "11" decodes to
        //   cbpy_intra 15 / cbpy_inter 0 (inter column inverts).
        let stream = bits(concat!(
            // partition 1
            "1",    // MB0 not_coded
            "0",    // MB1 not_coded = 0
            "1",    // MB1 mcbpc "1" → inter
            "1010", // MB1 motion (4 bits eaten by closure)
            // motion_marker
            "1 1111 0000 0000 0001",
            // partition 2: MB0 skipped (no bits); MB1 inter → cbpy "11"
            "11",
            "0000"
        ));
        let mut br = BitReader::new(&stream);
        let mut motion_calls = 0;
        let out = parse_data_partitioned_p_vop(&mut br, 4, false, 7, 31, |b, ev| {
            let DpMbEvent::Motion(ty) = ev else {
                return Ok(());
            };
            motion_calls += 1;
            assert_eq!(ty, DerivedMbType::Inter);
            b.skip_bits(4)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(motion_calls, 1);
        assert_eq!(out.mbs.len(), 2);
        assert!(out.mbs[0].not_coded);
        assert_eq!(out.mbs[1].mb_type, Some(DerivedMbType::Inter));
        // partition 2: MB0 placeholder, MB1 inter cbpy.
        assert_eq!(out.tex_headers.len(), 2);
        assert_eq!(out.tex_headers[1].cbpy, 0); // inter column of "11"
        assert!(!out.tex_headers[1].ac_pred_flag);
    }
}

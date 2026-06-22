//! B-VOP macroblock-header bit-walk — Tables B.3 / B.4 / B.5 from
//! ISO/IEC 14496-2:2004 §B.1.1.
//!
//! The §6.2.6 syntax for a coded macroblock in a B-VOP starts with the
//! 1-or-2-bit `modb` codeword (Table B.3) and then optionally pulls in
//! `mb_type` (Table B.4 in the non-scalable case, Table B.5 in the
//! ref_select_code == "00" && scalability == 1 case) and a fixed-length
//! `cbpb` coded-block pattern across the macroblock's non-transparent
//! blocks. After the cbpb (when present and non-zero) the spec
//! optionally pulls in the 1-or-2-bit `dbquant` differential
//! (Table 6-33).
//!
//! This module decodes exactly that prefix plus — since round 41 —
//! the §6.2.6 `if (interlaced) interlaced_information()` body that
//! follows `dbquant`; `parse_b_vop_mb_header` stops there. Since
//! round 42 the motion-vector bodies that follow are covered by the
//! separate [`decode_b_vop_mb_motion_vectors`] walker (see its docs
//! for the §6.2.6 Table B.4-branch dispatch and the interlaced
//! field-prediction second invocations). `transparent_mb()` /
//! `co_located_not_coded` branches are gated out ahead of either call.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition), read by
//! the agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`:
//!
//! * §6.2.6 `macroblock()` — B-VOP branch (the `else { if (modb …) }`
//!   sub-tree that starts at `modb 1-2 vlclbf`), including its
//!   `motion_vector("forward" / "backward" / "direct")` lines and
//!   their `if (interlaced && field_prediction)` second invocations.
//! * §6.3.6 — `modb`, `mb_type`, `cbpb`, `dbquant` semantics for B-VOPs.
//! * §7.7.2.1 / §7.7.2.2 + Tables 7-14 / 7-15 — field-differential
//!   ordering (top field first, bottom field second; forward pair
//!   before backward pair).
//! * Table B.3 — VLC for `modb` (`1` / `01` / `00`).
//! * Table B.4 — `mb_type` for B-VOPs (ref_select_code != '00' or
//!   scalability == 0): codes `1 / 01 / 001 / 0001` → direct /
//!   interpolate / backward / forward, all with `dbquant` except direct.
//! * Table B.5 — `mb_type` for B-VOPs (ref_select_code == '00' &&
//!   scalability != 0): codes `01 / 001 / 1` → interpolate / backward /
//!   forward, all with `dbquant`. (No "direct" row.)
//! * Table 6-33 — `dbquant` codes (`10 → -2`, `0 → 0`, `11 → +2`).
//!
//! ## Out of scope (this round)
//!
//! * The §7.7.2.2 motion-vector *reconstruction* (PMV add, the
//!   even-vertical field recovery `MVy fi = 2 * (MVDy fi + (Py / 2))`)
//!   and motion compensation — the walker surfaces differentials only.
//! * The Table B.5 scalable enhancement-layer MV line (§6.2.6's
//!   separate `ref_select_code == '00' && scalability` branch carries
//!   its own dbquant + single-forward-MV gating).
//! * `transparent_mb()` / `co_located_not_coded` — both come from a
//!   wider VOP-level state machine and are the caller's job to gate.
//! * Sprite / grayscale / data-partitioned / FGS overlays — out of the
//!   rectangular-shape clean-room scope.

use crate::bitreader::{BitReader, BitReaderError};
use crate::interlaced_information::{
    parse_interlaced_information, InterlacedInfoContext, InterlacedInformation,
};
use crate::motion::{
    decode_field_motion_vector_pair, decode_motion_vector_delta, FieldMvPair, MotionParseError,
    MotionVectorDelta, MvMode,
};
use crate::vol::VolHeader;
use crate::vop::VopCodingType;

/// Decoded B-VOP macroblock type, per Tables B.4 (non-scalable) and
/// B.5 (scalable enhancement layer with `ref_select_code == 00`).
///
/// "Direct" is only emitted from Table B.4 — Table B.5 has no direct
/// row (spec §7.9.2.8.3: "the direct mode shall not be used" in
/// enhancement layers of spatial scalability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BVopMbType {
    /// Table B.4 row `1`. No `dbquant`, no `mvdf`, no `mvdb`. The MV
    /// itself is reconstructed from the co-located P-VOP forward MV
    /// (the "direct mode" body the spec describes in §7.6.5).
    Direct,
    /// Table B.4 row `0001` / Table B.5 row `1`. Carries `dbquant` and
    /// a forward MV delta (`mvdf`). No backward delta.
    Forward,
    /// Table B.4 row `001` / Table B.5 row `001`. Carries `dbquant`
    /// and a backward MV delta (`mvdb`). No forward delta.
    Backward,
    /// Table B.4 row `01` / Table B.5 row `01`. Carries `dbquant`,
    /// `mvdf`, and `mvdb` — bidirectional / "interpolated" prediction.
    Interpolated,
}

impl BVopMbType {
    /// Whether the type-row carries a forward motion-vector delta
    /// (`mvdf` present). Direct has neither delta; Forward and
    /// Interpolated have a forward delta.
    pub fn has_forward_mv(self) -> bool {
        matches!(self, BVopMbType::Forward | BVopMbType::Interpolated)
    }

    /// Whether the type-row carries a backward motion-vector delta
    /// (`mvdb` present). Direct has neither; Backward and Interpolated
    /// have a backward delta.
    pub fn has_backward_mv(self) -> bool {
        matches!(self, BVopMbType::Backward | BVopMbType::Interpolated)
    }

    /// Whether the type-row optionally carries a `dbquant` delta (the
    /// spec gates `dbquant` further on `cbpb != 0`, so this is a
    /// necessary-not-sufficient predicate). Direct never carries
    /// `dbquant`; the other three rows can.
    pub fn may_have_dbquant(self) -> bool {
        !matches!(self, BVopMbType::Direct)
    }
}

/// Default macroblock type when `modb == '1'` and `mb_type` is therefore
/// absent from the bitstream. Per §6.3.6:
///
/// > "The default macroblock type for the enhancement layer of
/// > spatially scalable bitstreams (i.e. `ref_select_code == '00' &&
/// > scalability == '1'`) is "forward mc + Q". Otherwise, the default
/// > macroblock type is "direct"."
///
/// See also §7.9.2.8.3.
pub fn default_b_mb_type(scalable: bool) -> BVopMbType {
    if scalable {
        BVopMbType::Forward
    } else {
        BVopMbType::Direct
    }
}

/// `dbquant` differential per Table 6-33.
///
/// The 1-or-2-bit `dbquant` field decodes as:
///   * `0`   → `0`
///   * `10`  → `-2`
///   * `11`  → `+2`
///
/// Returns the consumed bit count alongside the signed delta so the
/// caller can advance the bit reader the right amount.
pub fn parse_dbquant(br: &mut BitReader<'_>) -> Result<(u8, i8), BVopMbParseError> {
    if br.read_bool()? {
        // 1x — read one more bit to discriminate `10` (-2) vs `11` (+2).
        let sign = br.read_bool()?;
        Ok((2, if sign { 2 } else { -2 }))
    } else {
        // 0 — value is 0, one bit consumed.
        Ok((1, 0))
    }
}

/// Typed view of a B-VOP macroblock header prefix.
///
/// The structure captures the bits up to and including `dbquant` and
/// — when the §6.2.6 dispatch is reached — the
/// `interlaced_information()` body. The motion-vector bodies (`mvdf`,
/// `mvdb`, "direct" MV, interlaced field-prediction extra MVs) are
/// out of scope this round and the bit reader is left positioned at
/// their start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BVopMbHeader {
    /// Raw `modb` code from Table B.3:
    ///   * `0b1` (`1`)  — mb_type absent (default type) + no cbpb.
    ///   * `0b01` (`01`) — mb_type present, no cbpb.
    ///   * `0b00` (`00`) — mb_type present, cbpb present.
    pub modb: u8,
    /// Decoded `mb_type`. When `modb == 0b1` this is the spec's
    /// default type ("forward mc + q" for the scalable enhancement
    /// branch, "direct" otherwise — see [`default_b_mb_type`]).
    pub mb_type: BVopMbType,
    /// Whether an explicit `mb_type` codeword was present in the
    /// bitstream. The §6.2.6 `modb` discriminator distinguishes
    /// `"1"` (no `mb_type` — `mb_type` here is the *default* type) from
    /// `"01"` / `"00"` (an explicit `mb_type` followed). Both `"1"` and
    /// `"01"` store the numeric raw value `1` in some encodings, so the
    /// `modb` field alone cannot separate them — this flag is the
    /// reliable discriminator (`false` ⟺ `modb == "1"`). The
    /// §7.6.9.6 skipped-macroblock override keys on it: a skipped
    /// co-located MB with `mb_type_present == false` (`modb == "1"`)
    /// reconstructs as direct mode with a zero delta, otherwise as
    /// forward mode with the zero MV.
    pub mb_type_present: bool,
    /// `cbpb` — coded block pattern across the macroblock's
    /// non-transparent blocks. `Some(value)` only when `modb == 0b00`
    /// (Table B.3 with `cbpb` column set). For 4:2:0 rectangular VOL
    /// (`block_count == 6`) this is a 6-bit value where bit 5
    /// corresponds to the top-left luminance block and bit 0 to the
    /// chroma-Cr block (MSB-first per spec §6.3.6).
    pub cbpb: Option<u8>,
    /// Whether a forward MV delta (`mvdf`) follows in the bitstream
    /// (i.e. body work for the next round). Derived purely from
    /// `mb_type` via [`BVopMbType::has_forward_mv`] — surfaced here
    /// for the caller's convenience.
    pub mvdf_present: bool,
    /// Whether a backward MV delta (`mvdb`) follows. Derived from
    /// `mb_type` via [`BVopMbType::has_backward_mv`].
    pub mvdb_present: bool,
    /// `dbquant` differential per Table 6-33, when present (i.e. the
    /// `mb_type != direct && cbpb != 0` gate in §6.2.6 was satisfied).
    /// `None` otherwise.
    pub dbquant_delta: Option<i8>,
    /// The §6.2.6.3 `interlaced_information()` body, decoded when the
    /// §6.2.6 `if (interlaced) interlaced_information()` line is
    /// reached. That line sits inside two enclosing gates: the
    /// `if (modb != '1')` subtree (a `modb == '1'` macroblock skips
    /// everything after `modb`, including this body) and the
    /// `if (ref_select_code != '00' || !scalability)` branch (the
    /// Table B.4 path — the scalable enhancement-layer Table B.5
    /// branch carries **no** `interlaced_information()` line in
    /// §6.2.6). `None` when the VOL is progressive (`interlaced ==
    /// 0`), when `modb == '1'`, or on the Table B.5 path; `Some(_)`
    /// otherwise — possibly a zero-bit body (a Direct macroblock with
    /// `cbpb == 0` fires neither §6.2.6.3 gate).
    pub interlaced_info: Option<InterlacedInformation>,
}

/// Errors produced by the B-VOP macroblock header parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BVopMbParseError {
    /// The supplied bit reader ran out mid-field.
    Truncated,
    /// The `modb` prefix did not match `1` / `01` / `00`. Cannot happen
    /// given a non-empty bit stream — every 2-bit window matches one
    /// of the three rows — but kept for defensive completeness.
    InvalidModb {
        /// The next-2-bits window we tried to match.
        window: u8,
    },
    /// The `mb_type` prefix did not match any code in the active
    /// table. The 4-bit window we tried to match is reported.
    InvalidMbType {
        /// The next-4-bits window we tried to match against Table B.4
        /// or Table B.5.
        window: u8,
    },
    /// Caller invoked the B-VOP parser on a non-B-VOP coding type.
    NotBVop(VopCodingType),
    /// The VOL shape is non-rectangular; `mb_binary_shape_coding()`
    /// would be needed first. The VOL header parser already rejects
    /// non-rectangular shapes, so this error is a defensive double
    /// check.
    UnsupportedShape(u8),
    /// The VOL `chroma_format` is not 4:2:0. The §6.2.6 NOTE pins
    /// `block_count = 6` for 4:2:0; other chroma formats need a
    /// different cbpb width and a different `block_count` and are out
    /// of scope this round.
    UnsupportedChromaFormat(u8),
    /// A §6.2.6.2 `motion_vector(mode)` body failed to decode (or
    /// `field_prediction` was supplied for a direct macroblock, which
    /// the §6.2.6.3 `mb_type != "1"` clause forbids).
    Motion(MotionParseError),
}

impl core::fmt::Display for BVopMbParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BVopMbParseError::Truncated => write!(f, "B-VOP macroblock header truncated"),
            BVopMbParseError::InvalidModb { window } => {
                write!(f, "invalid modb prefix (next-2-bits = 0b{window:02b})")
            }
            BVopMbParseError::InvalidMbType { window } => {
                write!(
                    f,
                    "invalid B-VOP mb_type prefix (next-4-bits = 0b{window:04b})"
                )
            }
            BVopMbParseError::NotBVop(k) => {
                write!(f, "B-VOP parser invoked on non-B VOP type ({k:?})")
            }
            BVopMbParseError::UnsupportedShape(s) => {
                write!(
                    f,
                    "B-VOP macroblock parsing requires rectangular shape (shape={s})"
                )
            }
            BVopMbParseError::UnsupportedChromaFormat(c) => {
                write!(
                    f,
                    "B-VOP macroblock parsing requires 4:2:0 chroma (chroma_format={c})"
                )
            }
            BVopMbParseError::Motion(e) => {
                write!(f, "B-VOP motion-vector body: {e}")
            }
        }
    }
}

impl std::error::Error for BVopMbParseError {}

impl From<BitReaderError> for BVopMbParseError {
    fn from(_: BitReaderError) -> Self {
        BVopMbParseError::Truncated
    }
}

impl From<MotionParseError> for BVopMbParseError {
    fn from(e: MotionParseError) -> Self {
        BVopMbParseError::Motion(e)
    }
}

/// Active mb_type table — determined by the
/// `ref_select_code == '00' && scalability` predicate from §6.2.6.
///
/// Non-scalable streams (the round-94 default path) read Table B.4;
/// the spatially-scalable enhancement-layer path reads Table B.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BMbTypeTable {
    /// Table B.4 — ref_select_code != '00' || scalability == 0.
    B4,
    /// Table B.5 — ref_select_code == '00' && scalability != 0.
    B5,
}

// ---------------------------------------------------------------------------
// VLC tables — verbatim from ISO/IEC 14496-2:2004 §B.1.1.
//
// Each entry is (code_bits, code_len, BVopMbType). The tables are
// short (3..=4 rows) so a linear prefix scan is fast enough.
// ---------------------------------------------------------------------------

/// Table B.4 — non-scalable B-VOP `mb_type`. Codes are read MSB-first.
const MB_TYPE_B4_TABLE: &[(u8, u8, BVopMbType)] = &[
    // 1     → direct (no dbquant, no mvdf, no mvdb)
    (0b1, 1, BVopMbType::Direct),
    // 01    → interpolate mc+q (dbquant + mvdf + mvdb)
    (0b01, 2, BVopMbType::Interpolated),
    // 001   → backward mc+q (dbquant + mvdb)
    (0b001, 3, BVopMbType::Backward),
    // 0001  → forward mc+q (dbquant + mvdf)
    (0b0001, 4, BVopMbType::Forward),
];

/// Table B.5 — scalable enhancement-layer B-VOP `mb_type`. No direct
/// mode; codes are read MSB-first.
const MB_TYPE_B5_TABLE: &[(u8, u8, BVopMbType)] = &[
    // 1   → forward mc+q (dbquant + mvdf)
    (0b1, 1, BVopMbType::Forward),
    // 01  → interpolate mc+q (dbquant + mvdf + mvdb)
    (0b01, 2, BVopMbType::Interpolated),
    // 001 → backward mc+q (dbquant + mvdb)
    (0b001, 3, BVopMbType::Backward),
];

/// Decode a `modb` code (Table B.3). Returns `(consumed_bits, raw_modb,
/// mb_type_present, cbpb_present)`.
///
/// Bit windowing:
///   * If the next bit is `1` → modb = `1`, 1 bit consumed.
///   * If the next bits are `01` → modb = `01`, 2 bits consumed.
///   * If the next bits are `00` → modb = `00`, 2 bits consumed.
fn decode_modb(br: &mut BitReader<'_>) -> Result<(u8, u8, bool, bool), BVopMbParseError> {
    if br.read_bool()? {
        // `1` — neither mb_type nor cbpb follow.
        Ok((1, 0b1, false, false))
    } else if br.read_bool()? {
        // `01` — mb_type present, cbpb absent.
        Ok((2, 0b01, true, false))
    } else {
        // `00` — both mb_type and cbpb present.
        Ok((2, 0b00, true, true))
    }
}

/// Decode a B-VOP `mb_type` codeword against the chosen table. Returns
/// the matched `BVopMbType`.
fn decode_mb_type(
    br: &mut BitReader<'_>,
    table: BMbTypeTable,
) -> Result<BVopMbType, BVopMbParseError> {
    let rows: &[(u8, u8, BVopMbType)] = match table {
        BMbTypeTable::B4 => MB_TYPE_B4_TABLE,
        BMbTypeTable::B5 => MB_TYPE_B5_TABLE,
    };
    let remaining = br.remaining_bits().min(4);
    if remaining == 0 {
        return Err(BVopMbParseError::Truncated);
    }
    let window = br.next_bits(remaining)? as u8;
    let window_len = remaining as u8;
    for &(code, len, ty) in rows {
        if len <= window_len {
            let shift = window_len - len;
            if (window >> shift) == code {
                br.skip_bits(len as usize)?;
                return Ok(ty);
            }
        }
    }
    Err(BVopMbParseError::InvalidMbType { window })
}

/// Parse a B-VOP macroblock header prefix from `br`.
///
/// The caller must have already gated this on the `co_located_not_coded`
/// / `transparent_mb()` / `binary_only` checks the §6.2.6 syntax
/// applies before the `modb` field. Round-94 scope is rectangular
/// shape + 4:2:0 chroma; both are validated up-front and yield a
/// typed error if violated.
///
/// On return, the bit reader is positioned immediately after the
/// `interlaced_information()` body (when reached), after `dbquant`
/// (when present), or immediately after the final preceding field.
/// The caller continues with `motion_vector("…")` in a later round.
pub fn parse_b_vop_mb_header(
    br: &mut BitReader<'_>,
    vol: &VolHeader,
    vop_coding_type: VopCodingType,
    table: BMbTypeTable,
) -> Result<BVopMbHeader, BVopMbParseError> {
    if vop_coding_type != VopCodingType::B {
        return Err(BVopMbParseError::NotBVop(vop_coding_type));
    }
    if vol.video_object_layer_shape != 0 {
        // Defensive: VOL parser already rejects non-rectangular shapes.
        return Err(BVopMbParseError::UnsupportedShape(
            vol.video_object_layer_shape,
        ));
    }
    // chroma_format check — when a vol_control_parameters block was
    // present the spec carries it explicitly (Table 6-15). Without
    // the block the format defaults to 4:2:0 (the only non-reserved
    // value per §6.2.3). The round-94 cbpb decode is pinned to
    // block_count == 6 (4:2:0 rectangular).
    if let Some(ctrl) = vol.vol_control {
        if ctrl.chroma_format != 0b01 {
            return Err(BVopMbParseError::UnsupportedChromaFormat(
                ctrl.chroma_format,
            ));
        }
    }

    let (_modb_bits, modb, mb_type_present, cbpb_present) = decode_modb(br)?;

    let scalable = matches!(table, BMbTypeTable::B5);
    let mb_type = if mb_type_present {
        decode_mb_type(br, table)?
    } else {
        default_b_mb_type(scalable)
    };

    // cbpb — 6 bits for 4:2:0 rectangular (block_count == 6 per §6.2.6
    // NOTE). The bits run leftmost = top-left block, rightmost = last
    // non-transparent block.
    let cbpb = if cbpb_present {
        Some(br.read_bits(6)? as u8)
    } else {
        None
    };

    // dbquant — present iff mb_type carries dbquant AND cbpb != 0
    // (§6.2.6 syntax line `if (mb_type != "1" && cbpb!=0) dbquant`).
    // Mapping `mb_type != "1"` ↔ "not direct" matches both Table B.4
    // and Table B.5 (in B.5 the row '1' is "forward mc+q" which still
    // carries dbquant per the table; but the syntax gate uses the raw
    // mb_type bit pattern). We follow the table-derived predicate via
    // BVopMbType::may_have_dbquant.
    let dbquant_delta = if mb_type.may_have_dbquant() && matches!(cbpb, Some(v) if v != 0) {
        let (_bits, delta) = parse_dbquant(br)?;
        Some(delta)
    } else {
        None
    };

    // §6.2.6: `if (interlaced) interlaced_information()` — fires
    // immediately after the (optional) `dbquant` and before the
    // motion-vector bodies, but only inside the `if (modb != '1')`
    // subtree (gated here via `mb_type_present` — the raw `modb`
    // values 0b1 and 0b01 are numerically identical) and only on the
    // `ref_select_code != '00' || !scalability` branch (Table B.4
    // path). The scalable enhancement-layer branch (Table B.5) has no
    // `interlaced_information()` line in §6.2.6. The §6.2.6.3 first
    // gate's `cbp != 0` predicate is `cbpb != 0` — a `modb == '01'`
    // macroblock carries no `cbpb`, meaning no block is coded, so the
    // absent case collapses to `cbp == 0`. Note the dispatch fires
    // for Direct macroblocks too (the §6.2.6 line is unconditional
    // within the branch); §6.2.6.3 then suppresses `field_prediction`
    // via its `mb_type != "1"` clause.
    let interlaced_info = if vol.interlaced && mb_type_present && matches!(table, BMbTypeTable::B4)
    {
        let any_block_coded = matches!(cbpb, Some(v) if v != 0);
        let ctx = InterlacedInfoContext::b_vop(mb_type, any_block_coded);
        Some(parse_interlaced_information(br, &ctx)?)
    } else {
        None
    };

    Ok(BVopMbHeader {
        modb,
        mb_type,
        mb_type_present,
        cbpb,
        mvdf_present: mb_type.has_forward_mv(),
        mvdb_present: mb_type.has_backward_mv(),
        dbquant_delta,
        interlaced_info,
    })
}

// ---------------------------------------------------------------------------
// §6.2.6 B-VOP macroblock-level motion-vector bodies
// ---------------------------------------------------------------------------
//
// The §6.2.6 Table B.4 branch follows the `interlaced_information()` body
// with three guarded motion-vector blocks, in this syntax order:
//
//   if (mb_type == '01' || mb_type == '0001') {
//         motion_vector("forward")
//         if (interlaced && field_prediction)
//               motion_vector("forward")
//   }
//   if (mb_type == '01' || mb_type == '001') {
//         motion_vector("backward")
//         if (interlaced && field_prediction)
//               motion_vector("backward")
//   }
//   if (mb_type == "1")
//         motion_vector("direct")
//
// Mapping the raw codes through Table B.4: '01' = interpolate, '001' =
// backward, '0001' = forward, '1' = direct — i.e. the forward block fires
// exactly when `BVopMbType::has_forward_mv()`, the backward block exactly
// when `BVopMbType::has_backward_mv()`, and an interpolated macroblock
// reads its forward body (or field pair) before its backward one.
//
// §7.7.2.2 fixes the field-pair ordering (Table 7-14: PMV 0 = top field
// forward, 1 = bottom field forward, 2 = top field backward, 3 = bottom
// field backward; the field-bidirectional pseudo code walks `MVD[0..3]`
// against `PMV[0..3]` in that order), so a field-predicted interpolated
// macroblock's four bodies decode as forward-top, forward-bottom,
// backward-top, backward-bottom.
//
// The direct body has no residual components (§6.2.6.2's `mode ==
// "direct"` branch reads only the two `mv_data` VLCs), which is the
// fcode-1 reconstruction; §7.6.9.5.1 calls it the single delta vector
// `MVD` shared by all four direct-mode block vectors.

/// One decoded prediction direction of a B-VOP macroblock — a single
/// frame differential, or the top/bottom field pair when
/// `field_prediction == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BVopMvBody {
    /// Frame prediction — one §6.2.6.2 body.
    Frame(MotionVectorDelta),
    /// Field prediction — two §6.2.6.2 bodies, top then bottom
    /// (§7.7.2.1 / §7.7.2.2).
    Field(FieldMvPair),
}

/// All motion-vector bodies of one §6.2.6 B-VOP macroblock (Table B.4
/// branch), as decoded by [`decode_b_vop_mb_motion_vectors`].
///
/// Exactly one of the three patterns is populated per Table B.4 row:
/// forward only (`mb_type == '0001'`), backward only (`'001'`), forward
/// + backward (`'01'`, interpolated), or direct only (`'1'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BVopMotionVectors {
    /// `mvdf` — present for Forward and Interpolated macroblocks.
    pub forward: Option<BVopMvBody>,
    /// `mvdb` — present for Backward and Interpolated macroblocks.
    pub backward: Option<BVopMvBody>,
    /// `mvdb`-direct — the single §7.6.9.5.1 delta vector, present for
    /// Direct macroblocks only. Never a field pair (§6.2.6 has no
    /// second invocation on the direct line).
    pub direct: Option<MotionVectorDelta>,
}

/// Decode the §6.2.6 B-VOP macroblock-level motion-vector bodies that
/// follow the [`parse_b_vop_mb_header`] prefix on the Table B.4
/// (non-scalable / `ref_select_code != '00'`) branch.
///
/// `field_prediction` is the §6.2.6.3 bit from the header's
/// `interlaced_information()` body
/// (`BVopMbHeader::interlaced_info` →
/// `InterlacedInformation::field_prediction.is_some()`); pass `false`
/// for progressive VOLs or frame-predicted macroblocks. When `true`,
/// each populated direction decodes the §6.2.6 second invocation into
/// a [`FieldMvPair`]. A `true` flag with a Direct macroblock is
/// rejected without consuming bits — the §6.2.6.3 `mb_type != "1"`
/// clause never codes `field_prediction` for direct.
///
/// The forward bodies use `vop_fcode_forward`, the backward bodies
/// `vop_fcode_backward` (§6.2.6.2 residual gating). The caller must
/// skip this walker entirely for `modb == '1'` macroblocks — the
/// whole `if (modb != '1')` subtree, motion vectors included, is
/// absent from the bitstream.
pub fn decode_b_vop_mb_motion_vectors(
    br: &mut BitReader<'_>,
    mb_type: BVopMbType,
    field_prediction: bool,
    vop_fcode_forward: u8,
    vop_fcode_backward: u8,
) -> Result<BVopMotionVectors, BVopMbParseError> {
    if field_prediction && mb_type == BVopMbType::Direct {
        return Err(BVopMbParseError::Motion(
            MotionParseError::InvalidFieldPredictionContext,
        ));
    }

    let decode_direction =
        |br: &mut BitReader<'_>, mode: MvMode, fcode: u8| -> Result<BVopMvBody, BVopMbParseError> {
            if field_prediction {
                Ok(BVopMvBody::Field(decode_field_motion_vector_pair(
                    br, mode, fcode,
                )?))
            } else {
                Ok(BVopMvBody::Frame(decode_motion_vector_delta(
                    br, mode, fcode,
                )?))
            }
        };

    // §6.2.6 syntax order: forward block, then backward block, then
    // direct.
    let forward = if mb_type.has_forward_mv() {
        Some(decode_direction(br, MvMode::Forward, vop_fcode_forward)?)
    } else {
        None
    };
    let backward = if mb_type.has_backward_mv() {
        Some(decode_direction(br, MvMode::Backward, vop_fcode_backward)?)
    } else {
        None
    };
    let direct = if mb_type == BVopMbType::Direct {
        // The direct body never reads residuals (§6.2.6.2), so its
        // reconstruction is the fcode-1 case regardless of the VOP
        // header fcodes.
        Some(decode_motion_vector_delta(br, MvMode::Direct, 1)?)
    } else {
        None
    };

    Ok(BVopMotionVectors {
        forward,
        backward,
        direct,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vol::{AspectRatio, SpriteEnable, VolHeader};

    fn make_vol() -> VolHeader {
        VolHeader {
            profile_level: 0x01,
            width: 176,
            height: 144,
            time_increment_resolution: 30_000,
            fixed_vop_rate: false,
            fixed_vop_time_increment: None,
            aspect_ratio: AspectRatio::Square,
            vol_control: None,
            random_accessible_vol: false,
            video_object_type_indication: 1,
            is_object_layer_identifier: false,
            video_object_layer_verid: 1,
            video_object_layer_priority: 0,
            video_object_layer_shape: 0,
            interlaced: false,
            obmc_disable: false,
            sprite_enable: SpriteEnable::NotUsed,
            no_of_sprite_warping_points: None,
            sprite_warping_accuracy: None,
            sprite_brightness_change: None,
            sprite_geometry: None,
            low_latency_sprite_enable: None,
            not_8_bit: false,
            quant_precision: 5,
            bits_per_pixel: 8,
            quant_type: false,
            intra_quant_mat: None,
            nonintra_quant_mat: None,
            quarter_sample: false,
            complexity_estimation_disable: true,
            resync_marker_disable: true,
            data_partitioned: false,
            reversible_vlc: false,
            newpred_enable: false,
            reduced_resolution_vop_enable: false,
            scalability: false,
        }
    }

    /// MSB-first bit writer matching the spec's bslbf / uimsbf
    /// convention. Mirrors the helper in `macroblock::tests`.
    struct BitWriter {
        buf: Vec<u8>,
        bit_pos: usize,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                buf: Vec::new(),
                bit_pos: 0,
            }
        }
        fn write_bits(&mut self, value: u32, n: usize) {
            for i in (0..n).rev() {
                let bit = ((value >> i) & 1) as u8;
                if self.bit_pos % 8 == 0 {
                    self.buf.push(0);
                }
                let byte = self.buf.last_mut().unwrap();
                *byte |= bit << (7 - (self.bit_pos % 8));
                self.bit_pos += 1;
            }
        }
        fn align(&mut self) {
            while self.bit_pos % 8 != 0 {
                self.write_bits(0, 1);
            }
        }
    }

    #[test]
    fn modb_table_b3_round_trip() {
        for (bits, len, modb_expected, mb_type_present, cbpb_present) in [
            (0b1u32, 1usize, 0b1u8, false, false),
            (0b01u32, 2, 0b01, true, false),
            (0b00u32, 2, 0b00, true, true),
        ] {
            let mut w = BitWriter::new();
            w.write_bits(bits, len);
            w.align();
            let data = w.buf;
            let mut br = BitReader::new(&data);
            let (consumed, raw, has_mb_type, has_cbpb) = decode_modb(&mut br).unwrap();
            assert_eq!(consumed, len as u8);
            assert_eq!(raw, modb_expected, "bits={bits:b}");
            assert_eq!(has_mb_type, mb_type_present);
            assert_eq!(has_cbpb, cbpb_present);
        }
    }

    #[test]
    fn dbquant_table_6_33_round_trip() {
        for (bits, len, expected) in [(0b0u32, 1usize, 0i8), (0b10, 2, -2), (0b11, 2, 2)] {
            let mut w = BitWriter::new();
            w.write_bits(bits, len);
            w.align();
            let data = w.buf;
            let mut br = BitReader::new(&data);
            let (consumed, delta) = parse_dbquant(&mut br).unwrap();
            assert_eq!(consumed, len as u8);
            assert_eq!(delta, expected, "bits={bits:b}");
        }
    }

    #[test]
    fn b_mb_type_predicates() {
        assert!(BVopMbType::Forward.has_forward_mv());
        assert!(BVopMbType::Interpolated.has_forward_mv());
        assert!(!BVopMbType::Backward.has_forward_mv());
        assert!(!BVopMbType::Direct.has_forward_mv());
        assert!(BVopMbType::Backward.has_backward_mv());
        assert!(BVopMbType::Interpolated.has_backward_mv());
        assert!(!BVopMbType::Forward.has_backward_mv());
        assert!(!BVopMbType::Direct.has_backward_mv());
        assert!(BVopMbType::Forward.may_have_dbquant());
        assert!(BVopMbType::Backward.may_have_dbquant());
        assert!(BVopMbType::Interpolated.may_have_dbquant());
        assert!(!BVopMbType::Direct.may_have_dbquant());
    }

    #[test]
    fn default_b_mb_type_per_spec() {
        // §6.3.6: non-scalable default is "direct"; scalable
        // enhancement layer with ref_select_code == '00' defaults to
        // "forward mc + Q".
        assert_eq!(default_b_mb_type(false), BVopMbType::Direct);
        assert_eq!(default_b_mb_type(true), BVopMbType::Forward);
    }

    #[test]
    fn parses_modb_1_default_direct() {
        // modb = 1 → no mb_type, no cbpb. Non-scalable default ⇒ Direct.
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb = 1
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        assert_eq!(hdr.modb, 0b1);
        assert_eq!(hdr.mb_type, BVopMbType::Direct);
        assert_eq!(hdr.cbpb, None);
        assert!(!hdr.mvdf_present);
        assert!(!hdr.mvdb_present);
        assert_eq!(hdr.dbquant_delta, None);
    }

    #[test]
    fn parses_modb_1_default_scalable_forward() {
        // Scalable enhancement layer (Table B.5 path) defaults to
        // Forward when mb_type is absent.
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B5)
            .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Forward);
        assert!(hdr.mvdf_present);
        assert!(!hdr.mvdb_present);
        assert_eq!(hdr.cbpb, None);
        assert_eq!(hdr.dbquant_delta, None);
    }

    #[test]
    fn parses_modb_01_forward_b4_no_dbquant() {
        // modb=01 → mb_type present, cbpb absent. mb_type=0001 → Forward.
        // dbquant gate is "cbpb != 0", and cbpb is absent here, so no
        // dbquant.
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb
        w.write_bits(0b0001, 4); // mb_type → Forward
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        assert_eq!(hdr.modb, 0b01);
        assert_eq!(hdr.mb_type, BVopMbType::Forward);
        assert_eq!(hdr.cbpb, None);
        assert!(hdr.mvdf_present);
        assert!(!hdr.mvdb_present);
        assert_eq!(hdr.dbquant_delta, None);
    }

    #[test]
    fn parses_modb_00_interpolated_b4_with_dbquant() {
        // modb=00 → mb_type + cbpb. mb_type=01 → Interpolated. cbpb=6
        // bits with at least one 1 → dbquant present. dbquant 11 → +2.
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb
        w.write_bits(0b01, 2); // mb_type → Interpolated
        w.write_bits(0b100000, 6); // cbpb (non-zero)
        w.write_bits(0b11, 2); // dbquant = +2
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        assert_eq!(hdr.modb, 0b00);
        assert_eq!(hdr.mb_type, BVopMbType::Interpolated);
        assert_eq!(hdr.cbpb, Some(0b100000));
        assert!(hdr.mvdf_present);
        assert!(hdr.mvdb_present);
        assert_eq!(hdr.dbquant_delta, Some(2));
    }

    #[test]
    fn parses_modb_00_backward_b4_cbpb_zero_no_dbquant() {
        // modb=00 → mb_type + cbpb. mb_type=001 → Backward. cbpb=000000
        // → dbquant absent (cbpb==0 gate).
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb
        w.write_bits(0b001, 3); // mb_type → Backward
        w.write_bits(0b000000, 6); // cbpb = 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Backward);
        assert_eq!(hdr.cbpb, Some(0));
        assert!(!hdr.mvdf_present);
        assert!(hdr.mvdb_present);
        assert_eq!(hdr.dbquant_delta, None);
    }

    #[test]
    fn parses_modb_00_direct_b4_no_dbquant_even_if_cbpb_nonzero() {
        // modb=00, mb_type=1 → Direct. Even with cbpb non-zero, the
        // dbquant gate in §6.2.6 reads "mb_type != '1'" — Direct
        // (the row coded as bit-pattern '1' in Table B.4) is excluded.
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb
        w.write_bits(0b1, 1); // mb_type → Direct
        w.write_bits(0b101010, 6); // cbpb (non-zero)
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Direct);
        assert_eq!(hdr.cbpb, Some(0b101010));
        assert!(!hdr.mvdf_present);
        assert!(!hdr.mvdb_present);
        assert_eq!(hdr.dbquant_delta, None);
    }

    #[test]
    fn parses_table_b5_mb_type_forward_row_1() {
        // In Table B.5 the `1` code is "forward mc+q" (no Direct row).
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb = 01 → mb_type present, no cbpb
        w.write_bits(0b1, 1); // mb_type → Forward (Table B.5 row 1)
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B5)
            .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Forward);
        assert!(hdr.mvdf_present);
        assert!(!hdr.mvdb_present);
    }

    #[test]
    fn full_round_trip_table_b4() {
        // Round-trip every Table B.4 row with modb=01 (mb_type present,
        // no cbpb). Asserts mb_type and the derived mvd flags.
        for &(code, len, ty) in MB_TYPE_B4_TABLE {
            let mut w = BitWriter::new();
            w.write_bits(0b01, 2); // modb
            w.write_bits(code as u32, len as usize); // mb_type
            w.align();
            let data = w.buf;
            let mut br = BitReader::new(&data);
            let hdr =
                parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
                    .unwrap();
            assert_eq!(hdr.mb_type, ty, "code={code:b}");
            assert_eq!(hdr.mvdf_present, ty.has_forward_mv(), "code={code:b}");
            assert_eq!(hdr.mvdb_present, ty.has_backward_mv(), "code={code:b}");
            assert_eq!(hdr.cbpb, None);
            assert_eq!(hdr.dbquant_delta, None);
        }
    }

    #[test]
    fn full_round_trip_table_b5() {
        for &(code, len, ty) in MB_TYPE_B5_TABLE {
            let mut w = BitWriter::new();
            w.write_bits(0b01, 2); // modb
            w.write_bits(code as u32, len as usize); // mb_type
            w.align();
            let data = w.buf;
            let mut br = BitReader::new(&data);
            let hdr =
                parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B5)
                    .unwrap();
            assert_eq!(hdr.mb_type, ty, "code={code:b}");
            assert_eq!(hdr.mvdf_present, ty.has_forward_mv(), "code={code:b}");
            assert_eq!(hdr.mvdb_present, ty.has_backward_mv(), "code={code:b}");
        }
    }

    #[test]
    fn dbquant_appears_only_when_cbpb_nonzero_and_non_direct() {
        // mb_type=01 (Interpolated) but cbpb=0 → dbquant absent.
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb
        w.write_bits(0b01, 2); // mb_type → Interpolated
        w.write_bits(0b000000, 6); // cbpb = 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Interpolated);
        assert_eq!(hdr.cbpb, Some(0));
        assert_eq!(hdr.dbquant_delta, None);
    }

    #[test]
    fn dbquant_zero_value_is_encoded_as_single_zero_bit() {
        // mb_type=Forward, cbpb non-zero, dbquant = 0 (single bit 0).
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb
        w.write_bits(0b0001, 4); // mb_type → Forward
        w.write_bits(0b000001, 6); // cbpb (non-zero)
        w.write_bits(0b0, 1); // dbquant = 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Forward);
        assert_eq!(hdr.cbpb, Some(0b000001));
        assert_eq!(hdr.dbquant_delta, Some(0));
    }

    #[test]
    fn dbquant_minus_two_encoded_as_10() {
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb
        w.write_bits(0b001, 3); // mb_type → Backward
        w.write_bits(0b010000, 6); // cbpb non-zero
        w.write_bits(0b10, 2); // dbquant = -2
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        assert_eq!(hdr.dbquant_delta, Some(-2));
    }

    #[test]
    fn rejects_non_b_vop() {
        let data = vec![0xFF];
        for ct in [VopCodingType::I, VopCodingType::P, VopCodingType::S] {
            let mut br = BitReader::new(&data);
            let err =
                parse_b_vop_mb_header(&mut br, &make_vol(), ct, BMbTypeTable::B4).unwrap_err();
            assert!(matches!(err, BVopMbParseError::NotBVop(_)), "ct={ct:?}");
        }
    }

    #[test]
    fn rejects_non_rectangular_shape() {
        let mut vol = make_vol();
        vol.video_object_layer_shape = 1;
        let data = vec![0xFF];
        let mut br = BitReader::new(&data);
        let err =
            parse_b_vop_mb_header(&mut br, &vol, VopCodingType::B, BMbTypeTable::B4).unwrap_err();
        assert!(matches!(err, BVopMbParseError::UnsupportedShape(1)));
    }

    #[test]
    fn rejects_non_420_chroma() {
        use crate::vol::VolControlParameters;
        let mut vol = make_vol();
        vol.vol_control = Some(VolControlParameters {
            chroma_format: 0b10, // 4:2:2
            low_delay: true,
            vbv: None,
        });
        let data = vec![0xFF];
        let mut br = BitReader::new(&data);
        let err =
            parse_b_vop_mb_header(&mut br, &vol, VopCodingType::B, BMbTypeTable::B4).unwrap_err();
        assert!(matches!(
            err,
            BVopMbParseError::UnsupportedChromaFormat(0b10)
        ));
    }

    #[test]
    fn accepts_420_chroma_when_explicit() {
        use crate::vol::VolControlParameters;
        let mut vol = make_vol();
        vol.vol_control = Some(VolControlParameters {
            chroma_format: 0b01, // 4:2:0
            low_delay: true,
            vbv: None,
        });
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb = 1 → default direct
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &vol, VopCodingType::B, BMbTypeTable::B4).unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Direct);
    }

    #[test]
    fn truncated_on_empty_buffer() {
        let data: Vec<u8> = Vec::new();
        let mut br = BitReader::new(&data);
        let err = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap_err();
        assert_eq!(err, BVopMbParseError::Truncated);
    }

    #[test]
    fn truncated_mid_mb_type() {
        // modb=01 then EOF before any mb_type bit.
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb
        w.align(); // pads to one byte; mb_type window is empty
                   // But the byte is now 0b0100_0000 — when we read modb=01, the
                   // next 6 bits are zeros (no valid mb_type in Table B.4).
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let err = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap_err();
        assert!(matches!(err, BVopMbParseError::InvalidMbType { window: 0 }));
    }

    #[test]
    fn truncated_mid_dbquant() {
        // modb=00, mb_type=Forward (0001), cbpb has at least one bit set,
        // then EOF before dbquant.
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2);
        w.write_bits(0b0001, 4); // mb_type = Forward
        w.write_bits(0b100000, 6); // cbpb non-zero
                                   // Total bits so far: 12 → 1.5 bytes; ends at bit 12, byte
                                   // boundary at 16. We deliberately allocate only those 12
                                   // bits (truncated by clamping the buffer to the bit count
                                   // already consumed).
        let bits_consumed: usize = 12;
        let bytes_needed = bits_consumed.div_ceil(8);
        let mut data = w.buf;
        data.truncate(bytes_needed);
        // The trailing byte already has zeros in its low 4 bits, so
        // reading the next 2 bits would yield 0b00 → dbquant = 0.
        // To force truncation, drop the trailing byte too.
        data.truncate(bytes_needed - 1);
        let mut br = BitReader::new(&data);
        let err = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap_err();
        assert_eq!(err, BVopMbParseError::Truncated);
    }

    #[test]
    fn invalid_mb_type_b5_window_is_zeros() {
        // Table B.5 has no code that matches 4 zero bits.
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb
        w.write_bits(0b0000, 4); // garbage mb_type window
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let err = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B5)
            .unwrap_err();
        assert!(matches!(err, BVopMbParseError::InvalidMbType { window: 0 }));
    }

    #[test]
    fn error_display_covers_all_variants() {
        let cases = [
            BVopMbParseError::Truncated,
            BVopMbParseError::InvalidModb { window: 0 },
            BVopMbParseError::InvalidMbType { window: 0 },
            BVopMbParseError::NotBVop(VopCodingType::I),
            BVopMbParseError::UnsupportedShape(2),
            BVopMbParseError::UnsupportedChromaFormat(0b10),
            BVopMbParseError::Motion(MotionParseError::Truncated),
        ];
        for e in cases {
            let s = format!("{e}");
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn bit_reader_advances_past_dbquant() {
        // mb_type=0001 (Forward, 4 bits) + modb=00 (2 bits) + cbpb (6
        // bits) + dbquant 0 (1 bit) = 13 bits.
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2);
        w.write_bits(0b0001, 4);
        w.write_bits(0b000001, 6);
        w.write_bits(0b0, 1);
        // Add a sentinel byte the caller would treat as the start of
        // the next field (e.g. motion_vector body).
        w.write_bits(0b1010_1010, 8);
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let _ = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        // The parser should have stopped at bit 13. Read 8 bits and
        // verify we see our sentinel 0xAA.
        let next = br.read_bits(8).unwrap();
        assert_eq!(next, 0xAA, "parser stopped at the wrong bit position");
    }

    // -----------------------------------------------------------------
    // §6.2.6 `if (interlaced) interlaced_information()` dispatch.
    // -----------------------------------------------------------------

    use crate::interlaced_information::{DctType, FieldReference};

    fn make_interlaced_vol() -> VolHeader {
        let mut vol = make_vol();
        vol.interlaced = true;
        vol
    }

    /// Progressive VOLs (`interlaced == 0`) never read the §6.2.6.3
    /// body — `interlaced_info` stays `None`.
    #[test]
    fn progressive_vol_has_no_interlaced_info() {
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb
        w.write_bits(0b01, 2); // mb_type → Interpolated
        w.write_bits(0b100000, 6); // cbpb (non-zero)
        w.write_bits(0b0, 1); // dbquant = 0
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(&mut br, &make_vol(), VopCodingType::B, BMbTypeTable::B4)
            .unwrap();
        assert_eq!(hdr.interlaced_info, None);
    }

    /// `modb == '1'` skips the entire `if (modb != '1')` subtree —
    /// including the `interlaced_information()` line — even in an
    /// interlaced VOL. The bit reader must stop right after `modb`.
    #[test]
    fn interlaced_modb_1_skips_interlaced_information() {
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1); // modb = 1
        w.write_bits(0b1010_1010, 8); // sentinel
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(
            &mut br,
            &make_interlaced_vol(),
            VopCodingType::B,
            BMbTypeTable::B4,
        )
        .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Direct);
        assert_eq!(hdr.interlaced_info, None);
        assert_eq!(br.read_bits(8).unwrap(), 0xAA);
    }

    /// Direct macroblock with `cbpb != 0`: the §6.2.6 dispatch still
    /// fires (the line is unconditional within the Table B.4 branch),
    /// but §6.2.6.3 emits only `dct_type` — the `field_prediction`
    /// guard's `mb_type != "1"` clause excludes Direct. No `dbquant`
    /// either (its own `mb_type != "1"` gate).
    #[test]
    fn interlaced_direct_cbpb_nonzero_reads_dct_type_only() {
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb → mb_type + cbpb
        w.write_bits(0b1, 1); // mb_type → Direct
        w.write_bits(0b000001, 6); // cbpb (non-zero)
        w.write_bits(0b1, 1); // dct_type = 1 → Field
        w.write_bits(0b1010_1010, 8); // sentinel
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(
            &mut br,
            &make_interlaced_vol(),
            VopCodingType::B,
            BMbTypeTable::B4,
        )
        .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Direct);
        assert_eq!(hdr.dbquant_delta, None);
        let info = hdr.interlaced_info.unwrap();
        assert_eq!(info.dct_type, Some(DctType::Field));
        assert_eq!(info.field_prediction, None);
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 1);
        assert_eq!(br.read_bits(8).unwrap(), 0xAA);
    }

    /// Direct macroblock with `cbpb == 0`: neither §6.2.6.3 gate fires
    /// — the body is zero bits but still surfaced as `Some(_)` (the
    /// dispatch was reached).
    #[test]
    fn interlaced_direct_cbpb_zero_zero_bit_body() {
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb → mb_type + cbpb
        w.write_bits(0b1, 1); // mb_type → Direct
        w.write_bits(0b000000, 6); // cbpb = 0
        w.write_bits(0b1010_1010, 8); // sentinel
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(
            &mut br,
            &make_interlaced_vol(),
            VopCodingType::B,
            BMbTypeTable::B4,
        )
        .unwrap();
        let info = hdr.interlaced_info.unwrap();
        assert_eq!(info.dct_type, None);
        assert_eq!(info.field_prediction, None);
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 0);
        assert_eq!(br.read_bits(8).unwrap(), 0xAA);
    }

    /// Interpolated macroblock, `cbpb != 0`, `field_prediction == 1`:
    /// the full 6-bit §6.2.6.3 body — `dct_type` + `field_prediction`
    /// + forward pair + backward pair — follows `dbquant`.
    #[test]
    fn interlaced_interpolated_full_body_after_dbquant() {
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb → mb_type + cbpb
        w.write_bits(0b01, 2); // mb_type → Interpolated
        w.write_bits(0b100000, 6); // cbpb (non-zero)
        w.write_bits(0b11, 2); // dbquant = +2
        w.write_bits(0b1, 1); // dct_type = 1 → Field
        w.write_bits(0b1, 1); // field_prediction = 1
        w.write_bits(0b0, 1); // forward_top_field_reference → Top
        w.write_bits(0b1, 1); // forward_bottom_field_reference → Bottom
        w.write_bits(0b1, 1); // backward_top_field_reference → Bottom
        w.write_bits(0b0, 1); // backward_bottom_field_reference → Top
        w.write_bits(0b1010_1010, 8); // sentinel
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(
            &mut br,
            &make_interlaced_vol(),
            VopCodingType::B,
            BMbTypeTable::B4,
        )
        .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Interpolated);
        assert_eq!(hdr.dbquant_delta, Some(2));
        let info = hdr.interlaced_info.unwrap();
        assert_eq!(info.dct_type, Some(DctType::Field));
        assert!(info.field_prediction_guard_fired);
        let fp = info.field_prediction.unwrap();
        assert_eq!(
            fp.forward,
            Some((FieldReference::Top, FieldReference::Bottom))
        );
        assert_eq!(
            fp.backward,
            Some((FieldReference::Bottom, FieldReference::Top))
        );
        assert_eq!(info.bit_count(), 6);
        assert_eq!(br.read_bits(8).unwrap(), 0xAA);
    }

    /// Forward macroblock with `modb == '01'` (no `cbpb` → `cbp == 0`):
    /// no `dct_type`, but the `field_prediction` guard fires (`mb_type
    /// != "1"`). A `field_prediction == 0` bit yields a 1-bit body
    /// with the guard flag set.
    #[test]
    fn interlaced_forward_no_cbpb_field_prediction_zero() {
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb → mb_type, no cbpb
        w.write_bits(0b0001, 4); // mb_type → Forward
        w.write_bits(0b0, 1); // field_prediction = 0
        w.write_bits(0b1010_1010, 8); // sentinel
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(
            &mut br,
            &make_interlaced_vol(),
            VopCodingType::B,
            BMbTypeTable::B4,
        )
        .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Forward);
        assert_eq!(hdr.cbpb, None);
        let info = hdr.interlaced_info.unwrap();
        assert_eq!(info.dct_type, None);
        assert_eq!(info.field_prediction, None);
        assert!(info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 1);
        assert_eq!(br.read_bits(8).unwrap(), 0xAA);
    }

    /// Backward macroblock with `cbpb != 0` and `field_prediction ==
    /// 1`: the §6.2.6.3 forward sub-gate's `mb_type != "001"` clause
    /// suppresses the forward pair; only the backward pair follows
    /// (4-bit body: dct + fp + 2 backward reference bits).
    #[test]
    fn interlaced_backward_fp_one_backward_pair_only() {
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb → mb_type + cbpb
        w.write_bits(0b001, 3); // mb_type → Backward
        w.write_bits(0b010000, 6); // cbpb (non-zero)
        w.write_bits(0b10, 2); // dbquant = -2
        w.write_bits(0b0, 1); // dct_type = 0 → Frame
        w.write_bits(0b1, 1); // field_prediction = 1
        w.write_bits(0b1, 1); // backward_top_field_reference → Bottom
        w.write_bits(0b1, 1); // backward_bottom_field_reference → Bottom
        w.write_bits(0b1010_1010, 8); // sentinel
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(
            &mut br,
            &make_interlaced_vol(),
            VopCodingType::B,
            BMbTypeTable::B4,
        )
        .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Backward);
        assert_eq!(hdr.dbquant_delta, Some(-2));
        let info = hdr.interlaced_info.unwrap();
        assert_eq!(info.dct_type, Some(DctType::Frame));
        let fp = info.field_prediction.unwrap();
        assert_eq!(fp.forward, None);
        assert_eq!(
            fp.backward,
            Some((FieldReference::Bottom, FieldReference::Bottom))
        );
        assert_eq!(info.bit_count(), 4);
        assert_eq!(br.read_bits(8).unwrap(), 0xAA);
    }

    /// The scalable enhancement-layer branch (`ref_select_code == '00'
    /// && scalability` — the Table B.5 path) carries no
    /// `interlaced_information()` line in §6.2.6: even in an
    /// interlaced VOL the parser must not consume body bits there.
    #[test]
    fn scalable_b5_branch_has_no_interlaced_information() {
        let mut w = BitWriter::new();
        w.write_bits(0b01, 2); // modb → mb_type, no cbpb
        w.write_bits(0b1, 1); // mb_type → Forward (Table B.5 row 1)
        w.write_bits(0b1010_1010, 8); // sentinel
        w.align();
        let data = w.buf;
        let mut br = BitReader::new(&data);
        let hdr = parse_b_vop_mb_header(
            &mut br,
            &make_interlaced_vol(),
            VopCodingType::B,
            BMbTypeTable::B5,
        )
        .unwrap();
        assert_eq!(hdr.mb_type, BVopMbType::Forward);
        assert_eq!(hdr.interlaced_info, None);
        assert_eq!(br.read_bits(8).unwrap(), 0xAA);
    }

    /// Truncation inside the §6.2.6.3 body surfaces as `Truncated`.
    #[test]
    fn interlaced_truncated_mid_body() {
        let mut w = BitWriter::new();
        w.write_bits(0b00, 2); // modb → mb_type + cbpb
        w.write_bits(0b01, 2); // mb_type → Interpolated
        w.write_bits(0b100000, 6); // cbpb (non-zero)
        w.write_bits(0b0, 1); // dbquant = 0 — prefix = 11 bits
        w.write_bits(0b1, 1); // dct_type (bit 11)
        w.write_bits(0b1, 1); // field_prediction = 1 (bit 12)
        w.write_bits(0b0, 1); // forward_top_field_reference (bit 13)
        w.write_bits(0b1, 1); // forward_bottom_field_reference (bit 14)
        w.write_bits(0b1, 1); // backward_top_field_reference (bit 15)
        w.write_bits(0b0, 1); // backward_bottom_field_reference (bit 16)
        w.align();
        // Truncate to 2 bytes (16 bits): the backward_bottom read —
        // the body's final reference bit — runs out mid-body.
        let mut data = w.buf;
        data.truncate(2);
        let mut br = BitReader::new(&data);
        let err = parse_b_vop_mb_header(
            &mut br,
            &make_interlaced_vol(),
            VopCodingType::B,
            BMbTypeTable::B4,
        )
        .unwrap_err();
        assert_eq!(err, BVopMbParseError::Truncated);
    }

    // -----------------------------------------------------------------
    // §6.2.6 B-VOP motion-vector body walker tests.
    // -----------------------------------------------------------------

    /// Write one Table B.12 `mv_data` VLC. Only the small doubled-
    /// integer values these tests need are mapped; the full table
    /// lives in `crate::motion`.
    fn write_mvd(w: &mut BitWriter, mv_data: i32) {
        let (code, len) = match mv_data {
            0 => (0b1, 1),
            1 => (0b010, 3),
            -1 => (0b011, 3),
            2 => (0b0010, 4),
            -2 => (0b0011, 4),
            _ => panic!("mv_data {mv_data} not mapped in test helper"),
        };
        w.write_bits(code, len);
    }

    #[test]
    fn b_mv_direct_single_body_no_residuals() {
        // mb_type == "1" → one motion_vector("direct") body. The
        // §6.2.6.2 direct branch reads no residuals even with
        // fcode > 1, and the delta reconstructs as the bare mv_data
        // (fcode-1 case).
        let mut w = BitWriter::new();
        write_mvd(&mut w, 2); // horizontal_mv_data
        write_mvd(&mut w, -2); // vertical_mv_data
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = decode_b_vop_mb_motion_vectors(&mut br, BVopMbType::Direct, false, 3, 2).unwrap();
        assert_eq!(got.forward, None);
        assert_eq!(got.backward, None);
        assert_eq!(got.direct, Some(MotionVectorDelta { dx: 2, dy: -2 }));
        assert_eq!(br.bit_position(), 8);
    }

    #[test]
    fn b_mv_direct_rejects_field_prediction_without_consuming_bits() {
        // §6.2.6.3's `mb_type != "1"` clause never codes
        // field_prediction for a direct macroblock.
        let data: Vec<u8> = vec![0xFF, 0xFF];
        let mut br = BitReader::new(&data);
        let err =
            decode_b_vop_mb_motion_vectors(&mut br, BVopMbType::Direct, true, 1, 1).unwrap_err();
        assert_eq!(
            err,
            BVopMbParseError::Motion(MotionParseError::InvalidFieldPredictionContext)
        );
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn b_mv_forward_frame_single_forward_body() {
        // mb_type == '0001' (Forward) → forward body only.
        let mut w = BitWriter::new();
        write_mvd(&mut w, -1);
        write_mvd(&mut w, 1);
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got =
            decode_b_vop_mb_motion_vectors(&mut br, BVopMbType::Forward, false, 1, 1).unwrap();
        assert_eq!(
            got.forward,
            Some(BVopMvBody::Frame(MotionVectorDelta { dx: -1, dy: 1 }))
        );
        assert_eq!(got.backward, None);
        assert_eq!(got.direct, None);
        assert_eq!(br.bit_position(), 6);
    }

    #[test]
    fn b_mv_forward_field_prediction_decodes_pair() {
        // Forward MB with field_prediction == 1 → the §6.2.6 second
        // invocation fires: top then bottom (§7.7.2.1).
        let mut w = BitWriter::new();
        write_mvd(&mut w, 2);
        write_mvd(&mut w, 0);
        write_mvd(&mut w, -2);
        write_mvd(&mut w, 1);
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got = decode_b_vop_mb_motion_vectors(&mut br, BVopMbType::Forward, true, 1, 1).unwrap();
        assert_eq!(
            got.forward,
            Some(BVopMvBody::Field(FieldMvPair {
                top: MotionVectorDelta { dx: 2, dy: 0 },
                bottom: MotionVectorDelta { dx: -2, dy: 1 },
            }))
        );
        assert_eq!(got.backward, None);
        assert_eq!(br.bit_position(), 12);
    }

    #[test]
    fn b_mv_backward_field_pair_uses_backward_fcode() {
        // Backward MB at fcode_forward == 1 / fcode_backward == 2: the
        // backward bodies must gate residuals on vop_fcode_backward.
        // Top: h_data=2 + res 1 → (2-1)*2+1+1 = +4; v_data=0 → 0.
        // Bottom: h_data=0 → 0; v_data=-2 + res 0 → -3.
        let mut w = BitWriter::new();
        write_mvd(&mut w, 2);
        w.write_bits(1, 1); // h residual (r_size = 1)
        write_mvd(&mut w, 0);
        write_mvd(&mut w, 0);
        write_mvd(&mut w, -2);
        w.write_bits(0, 1); // v residual
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got =
            decode_b_vop_mb_motion_vectors(&mut br, BVopMbType::Backward, true, 1, 2).unwrap();
        assert_eq!(got.forward, None);
        assert_eq!(
            got.backward,
            Some(BVopMvBody::Field(FieldMvPair {
                top: MotionVectorDelta { dx: 4, dy: 0 },
                bottom: MotionVectorDelta { dx: 0, dy: -3 },
            }))
        );
        assert_eq!(br.bit_position(), 12);
    }

    #[test]
    fn b_mv_interpolated_frame_forward_then_backward() {
        // mb_type == '01' (Interpolated) → forward body first, then
        // backward (§6.2.6 syntax order). Distinct deltas prove the
        // assignment.
        let mut w = BitWriter::new();
        write_mvd(&mut w, 1); // forward h
        write_mvd(&mut w, -1); // forward v
        write_mvd(&mut w, -2); // backward h
        write_mvd(&mut w, 2); // backward v
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got =
            decode_b_vop_mb_motion_vectors(&mut br, BVopMbType::Interpolated, false, 1, 1).unwrap();
        assert_eq!(
            got.forward,
            Some(BVopMvBody::Frame(MotionVectorDelta { dx: 1, dy: -1 }))
        );
        assert_eq!(
            got.backward,
            Some(BVopMvBody::Frame(MotionVectorDelta { dx: -2, dy: 2 }))
        );
        assert_eq!(got.direct, None);
        assert_eq!(br.bit_position(), 14);
    }

    #[test]
    fn b_mv_interpolated_field_four_bodies_in_pmv_order() {
        // Field bidirectional: four bodies in Table 7-14 / §7.7.2.2
        // MVD[0..3] order — forward-top, forward-bottom, backward-top,
        // backward-bottom.
        let mut w = BitWriter::new();
        let deltas = [(1, 0), (-1, 0), (2, 0), (-2, 0)];
        for &(h, v) in &deltas {
            write_mvd(&mut w, h);
            write_mvd(&mut w, v);
        }
        w.align();
        let bytes = w.buf;
        let mut br = BitReader::new(&bytes);
        let got =
            decode_b_vop_mb_motion_vectors(&mut br, BVopMbType::Interpolated, true, 1, 1).unwrap();
        assert_eq!(
            got.forward,
            Some(BVopMvBody::Field(FieldMvPair {
                top: MotionVectorDelta { dx: 1, dy: 0 },
                bottom: MotionVectorDelta { dx: -1, dy: 0 },
            }))
        );
        assert_eq!(
            got.backward,
            Some(BVopMvBody::Field(FieldMvPair {
                top: MotionVectorDelta { dx: 2, dy: 0 },
                bottom: MotionVectorDelta { dx: -2, dy: 0 },
            }))
        );
        assert_eq!(got.direct, None);
        // (3+1) + (3+1) + (4+1) + (4+1) = 18 bits.
        assert_eq!(br.bit_position(), 18);
    }

    #[test]
    fn b_mv_truncated_mid_backward_pair() {
        // Interpolated field MB cut off exactly at a byte boundary in
        // the middle of the backward-top body → Truncated, surfaced
        // through the BVopMbParseError::Motion wrapper. The forward
        // pair (four zero VLCs, 4 bits) plus the backward-top h_data
        // (-2, 4 bits) consume the whole first byte, so the
        // backward-top v_data read starts with zero bits remaining.
        let mut w = BitWriter::new();
        for _ in 0..4 {
            write_mvd(&mut w, 0); // fwd pair (0,0)(0,0) → 4 bits
        }
        write_mvd(&mut w, -2); // bwd top h_data → bit 8 boundary
        write_mvd(&mut w, 0); // filler (second byte, truncated away)
        write_mvd(&mut w, 0);
        w.align();
        let mut data = w.buf;
        data.truncate(1);
        let mut br = BitReader::new(&data);
        let err = decode_b_vop_mb_motion_vectors(&mut br, BVopMbType::Interpolated, true, 1, 1)
            .unwrap_err();
        assert_eq!(err, BVopMbParseError::Motion(MotionParseError::Truncated));
    }
}

//! §6.2.6.3 `interlaced_information()` — the per-macroblock interlaced
//! syntax body that carries the optional `dct_type` bit and the
//! optional `field_prediction` block (with up to four field-reference
//! bits).
//!
//! ## Spec framing
//!
//! §6.2.6.3 wraps two independent gates inside one syntax body. The
//! `interlaced_information()` body is only emitted by the §6.2.6
//! macroblock layer when the VOL-header `interlaced` flag is set; the
//! caller drives that outer check before invoking this module.
//!
//! The first gate is the `dct_type` bit:
//!
//! > `if ((derived_mb_type == 3) || (derived_mb_type == 4) || (cbp != 0)) dct_type`
//!
//! Per §6.3.6.3:
//!
//! > `dct_type`: This is a 1-bit flag indicating whether the macroblock
//! > is frame DCT coded or field DCT coded. If this flag is set to "1",
//! > the macroblock is field DCT coded; otherwise, the macroblock is
//! > frame DCT coded. This flag is only present in the bitstream if
//! > the `interlaced` flag is set to "1" and the macroblock is coded
//! > (coded block pattern is non-zero) or intra-coded.
//!
//! The second gate is the `field_prediction` block. Its outer guard is
//! the disjunction:
//!
//! > `((vop_coding_type == "P") && ((derived_mb_type == 0) || (derived_mb_type == 1)))`
//! > `|| ((sprite_enable == "GMC") && (vop_coding_type == "S") && (derived_mb_type < 2) && (!mcsel))`
//! > `|| ((vop_coding_type == "B") && (mb_type != "1"))`
//!
//! Per §6.3.6.3:
//!
//! > `field_prediction`: This is a 1-bit flag indicating whether the
//! > macroblock is field predicted or frame predicted. This flag is set
//! > to '1' when the macroblock is predicted using field motion
//! > vectors. If it is set to '0' then frame prediction (16x16 or 8x8)
//! > will be used. This flag is only present when `interlaced == '1'`
//! > for the following types of macroblocks: a macroblock in a P-VOP
//! > with `derived_mb_type < 2`; a non-direct mode macroblock in a
//! > B-VOP; or a macroblock in an S (GMC)-VOP with `mcsel == '0'`.
//!
//! When `field_prediction == 1`, two further sub-gates run. The
//! forward sub-gate fires for P-VOPs, S-VOPs, and B-VOPs whose
//! `mb_type` is not the backward-only `001` row:
//!
//! > `if (vop_coding_type == "P" || vop_coding_type == "S" ||`
//! > ` (vop_coding_type == "B" && mb_type != "001")) {`
//! > `    forward_top_field_reference     1 bslbf`
//! > `    forward_bottom_field_reference  1 bslbf`
//! > `}`
//!
//! The backward sub-gate fires only inside a B-VOP whose `mb_type` is
//! not the forward-only `0001` row:
//!
//! > `if ((vop_coding_type == "B") && (mb_type != "0001")) {`
//! > `    backward_top_field_reference    1 bslbf`
//! > `    backward_bottom_field_reference 1 bslbf`
//! > `}`
//!
//! Per §6.3.6.3 each of the four reference bits selects which of the
//! two reference fields is used for the corresponding prediction:
//!
//! > When this flag is set to '0', the top field is used as the
//! > reference field. If it is set to '1' then the bottom field will
//! > be used as the reference field.
//!
//! ## What this module exposes
//!
//! * [`DctType`] — typed view of the `dct_type` bit (frame vs field
//!   DCT coding).
//! * [`FieldReference`] — typed view of one of the four reference
//!   bits (top vs bottom reference field).
//! * [`FieldPrediction`] — typed view of the populated reference bits
//!   when `field_prediction == 1`.
//! * [`InterlacedInformation`] — the §6.2.6.3 body return type,
//!   bundling the optional `dct_type` and optional [`FieldPrediction`].
//! * [`McSel`] — the S(GMC)-VOP `mcsel` flag, surfaced as a typed
//!   marker so callers can't accidentally route an `mcsel == 1`
//!   macroblock through the §6.2.6.3 gate (the §6.2.6.3 disjunction
//!   excludes that case).
//! * [`InterlacedInfoContext`] — the caller-decided gating context
//!   (VOP coding type + derived/B-VOP macroblock type + cbp +
//!   sprite-GMC + mcsel) that drives the two §6.2.6.3 guards. The
//!   construction is checked: a P-VOP can only carry `derived_mb_type`
//!   in `0..=4`; a B-VOP can only carry [`BVopMbType`]; an I-VOP only
//!   exposes the `dct_type` gate.
//! * [`parse_interlaced_information`] — the §6.2.6.3 body parser.
//!   Consumes `0..=5` bits depending on which gates fire, and never
//!   emits a flag whose syntax-level guard is not satisfied.
//! * [`field_prediction_present`] / [`dct_type_present`] — pure
//!   predicates that surface the §6.2.6.3 guards without parsing,
//!   useful for callers that need to advance their bit position to
//!   the next §6.2.6 syntax field.
//!
//! ## Spec gates encoded structurally
//!
//! The §6.2.6.3 outer guard for `field_prediction` is encoded
//! structurally in [`InterlacedInfoContext`]: there is no `S(GMC)`
//! variant with `mcsel == 1` — the matching constructor refuses it
//! up-front via [`McSel`]. Likewise the I-VOP variant cannot reach
//! [`field_prediction_present`] because the §6.2.6.3 disjunction has
//! no I-VOP clause. The result is that [`parse_interlaced_information`]
//! cannot produce a `FieldPrediction` for an I-VOP or an
//! `mcsel == 1` S-VOP macroblock at the type level.
//!
//! ## Scope
//!
//! This round covers strictly the §6.2.6.3 body — the parser does not
//! make any choices about how the resulting `field_prediction` /
//! reference bits feed into §7.6.5 motion-vector prediction or §7.6.2
//! field-mode interpolation. The §6.2.6 "`if (interlaced &&
//! field_prediction) motion_vector(\"forward\")`" second-invocation
//! line in the §6.2.6 motion-coding sequence — which fires off the
//! `field_prediction` bit decoded here — is left to a later round.
//! The §6.2.6 outer `if (interlaced) interlaced_information()` gate
//! is the caller's responsibility (see [`crate::macroblock`] for the
//! P-VOP macroblock-header parser; the dispatch into this module is
//! not yet wired).

use crate::bitreader::{BitReader, BitReaderError};
use crate::bvop::BVopMbType;
use crate::macroblock::DerivedMbType;

/// `dct_type` (§6.2.6.3 / §6.3.6.3). The 1-bit flag chooses between
/// frame-DCT and field-DCT coding of the macroblock's luminance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctType {
    /// `dct_type == 0` — frame DCT coding. The four 8x8 luminance
    /// blocks span the full 16x16 macroblock in raster order.
    Frame,
    /// `dct_type == 1` — field DCT coding. The eight even-numbered
    /// lines form two 8x8 blocks (top field) and the eight odd-numbered
    /// lines form two 8x8 blocks (bottom field).
    Field,
}

impl DctType {
    /// Decode a single `dct_type` bit per §6.2.6.3.
    pub fn from_bit(bit: bool) -> Self {
        if bit {
            DctType::Field
        } else {
            DctType::Frame
        }
    }

    /// Re-encode this flag back to its 1-bit form.
    pub fn as_bit(self) -> bool {
        matches!(self, DctType::Field)
    }
}

/// One of the four §6.2.6.3 field-reference bits
/// (`forward_top_field_reference`, `forward_bottom_field_reference`,
/// `backward_top_field_reference`, `backward_bottom_field_reference`).
/// Per §6.3.6.3 the bit selects which of the two reference fields
/// (top or bottom) is used for the corresponding prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldReference {
    /// Reference bit `0` — the top field of the reference VOP is used.
    Top,
    /// Reference bit `1` — the bottom field of the reference VOP is
    /// used.
    Bottom,
}

impl FieldReference {
    /// Decode a single reference bit per §6.3.6.3.
    pub fn from_bit(bit: bool) -> Self {
        if bit {
            FieldReference::Bottom
        } else {
            FieldReference::Top
        }
    }

    /// Re-encode this flag back to its 1-bit form.
    pub fn as_bit(self) -> bool {
        matches!(self, FieldReference::Bottom)
    }
}

/// The four (or fewer) §6.2.6.3 field-reference bits decoded when
/// `field_prediction == 1`.
///
/// The forward pair is present when the §6.2.6.3 forward sub-gate
/// fires (P-VOP, S-VOP, or B-VOP with `mb_type != "001"`); the
/// backward pair is present only inside a B-VOP whose `mb_type !=
/// "0001"`. The two pairs are independent — both, the forward pair
/// alone, or the backward pair alone may be populated, but never
/// neither (the spec's outer guard would not have fired).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldPrediction {
    /// `(forward_top_field_reference, forward_bottom_field_reference)`,
    /// present when the §6.2.6.3 forward sub-gate fires.
    pub forward: Option<(FieldReference, FieldReference)>,
    /// `(backward_top_field_reference, backward_bottom_field_reference)`,
    /// present when the §6.2.6.3 backward sub-gate fires.
    pub backward: Option<(FieldReference, FieldReference)>,
}

impl FieldPrediction {
    /// Number of bits consumed by this `field_prediction == 1` body
    /// — `2` per present pair, `0` for an absent pair.
    pub fn bit_count(&self) -> usize {
        let fwd = if self.forward.is_some() { 2 } else { 0 };
        let bwd = if self.backward.is_some() { 2 } else { 0 };
        fwd + bwd
    }
}

/// The §6.2.6.3 body's typed return.
///
/// Both fields are independently optional — `dct_type` is present iff
/// the §6.2.6.3 first guard fires, `field_prediction` is present iff
/// the §6.2.6.3 second guard fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterlacedInformation {
    /// Decoded `dct_type` bit, when the §6.2.6.3 first guard
    /// (`derived_mb_type ∈ {3, 4}` || `cbp != 0`) fires.
    pub dct_type: Option<DctType>,
    /// Decoded `field_prediction` block, when the §6.2.6.3 second
    /// guard fires and the bit value is `1`. A `field_prediction`
    /// value of `0` (frame prediction in an interlaced macroblock)
    /// surfaces as `Some(FieldPrediction { forward: None, backward:
    /// None })` only when the second guard fires; the
    /// [`field_prediction_present`] flag on the body distinguishes
    /// the two `None`-like states.
    pub field_prediction: Option<FieldPrediction>,
    /// Whether the §6.2.6.3 second guard fired, regardless of the
    /// decoded `field_prediction` value. `field_prediction` is the
    /// `Some(_)` discriminant only when the bit value was `1`; this
    /// flag stays `true` for the `field_prediction == 0` case so the
    /// caller can distinguish "field prediction explicitly disabled
    /// for this MB" from "the §6.2.6.3 second guard didn't fire".
    pub field_prediction_guard_fired: bool,
}

impl InterlacedInformation {
    /// Total number of bits the §6.2.6.3 body consumed for this
    /// macroblock.
    pub fn bit_count(&self) -> usize {
        let dct = usize::from(self.dct_type.is_some());
        let fp_guard = usize::from(self.field_prediction_guard_fired);
        let fp_body = self.field_prediction.map(|fp| fp.bit_count()).unwrap_or(0);
        dct + fp_guard + fp_body
    }
}

/// The S(GMC)-VOP `mcsel` flag, gating the §6.2.6.3 second-disjunct
/// clause. The §6.2.6.3 outer guard for the S-VOP case requires
/// `!mcsel`, so [`InterlacedInfoContext::s_gmc_vop`] only accepts
/// [`McSel::Off`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McSel {
    /// `mcsel == 0` — the macroblock carries its own block motion
    /// vector. This is the only case in which the §6.2.6.3 S-VOP
    /// disjunct fires.
    Off,
    /// `mcsel == 1` — the macroblock's pel-wise motion vectors are
    /// produced by §7.8.5 sprite warping; the §6.2.6.3 S-VOP disjunct
    /// is suppressed and [`field_prediction_present`] returns
    /// `false`.
    On,
}

/// Per-VOP / per-macroblock context used by the §6.2.6.3 gates.
///
/// The outer `interlaced == 1` check is the caller's responsibility;
/// this type assumes it has already fired. The two §6.2.6.3 guards
/// further depend on:
///
/// * the VOP coding type (I / P / B / S),
/// * the macroblock type (`derived_mb_type` for P-VOPs / S-VOPs,
///   [`BVopMbType`] for B-VOPs),
/// * the coded-block pattern `cbp` (for the `dct_type` guard),
/// * the VOL header `sprite_enable == "GMC"` flag (for the S-VOP
///   disjunct),
/// * the `mcsel` flag of the current macroblock (for the same
///   S-VOP disjunct).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterlacedInfoContext {
    /// I-VOP macroblock. Only the §6.2.6.3 `dct_type` gate is
    /// reachable; an I-VOP macroblock is always intra-coded so the
    /// gate's `derived_mb_type ∈ {3, 4}` clause always fires (and
    /// the `cbp` value is therefore not consulted).
    IVop {
        /// `derived_mb_type` of the macroblock. The I-VOP type table
        /// only contains the two intra rows (`Intra` and `IntraQ`);
        /// the constructor refuses the inter rows up-front.
        mb_type: DerivedMbType,
    },
    /// P-VOP macroblock.
    PVop {
        /// `derived_mb_type` of the macroblock (Table B.1, any of
        /// the five rows).
        mb_type: DerivedMbType,
        /// `cbp` per §6.3.5 — the sum of `cbpy << 2 | cbpc`'s
        /// per-block bits, or any equivalent representation of "is
        /// any of the six 8x8 blocks coded". The §6.2.6.3 first
        /// gate only checks `cbp != 0`, so this field is collapsed
        /// to a bool.
        any_block_coded: bool,
    },
    /// B-VOP macroblock.
    BVop {
        /// B-VOP macroblock type per Table B.4 / Table B.5.
        mb_type: BVopMbType,
        /// `cbp` per §6.3.5; collapsed as in `PVop`.
        any_block_coded: bool,
    },
    /// S(GMC)-VOP macroblock with `mcsel == 1` (GMC-predicted): the
    /// §6.2.6 `if (interlaced) interlaced_information()` line is
    /// unconditional, so the body is still invoked — only the
    /// `field_prediction` gate (§6.2.6.3 `!mcsel`) is off; `dct_type`
    /// keeps its `cbp != 0` gate (§6.3.6.3: "present ... if the
    /// macroblock is coded ... or intra-coded").
    SGmcVopGmc {
        /// `cbp != 0` — collapsed as in `PVop`.
        any_block_coded: bool,
    },
    /// S(GMC)-VOP macroblock with `mcsel == 0` (local MC).
    SGmcVop {
        /// `derived_mb_type`. Only `< 2` triggers the §6.2.6.3
        /// second gate's S-VOP disjunct — the constructor refuses
        /// the higher rows up-front.
        mb_type: DerivedMbType,
        /// `cbp` per §6.3.5; collapsed as in `PVop`.
        any_block_coded: bool,
    },
}

/// Errors emitted when constructing an [`InterlacedInfoContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterlacedInfoContextError {
    /// An I-VOP macroblock was constructed with an inter `mb_type`.
    /// I-VOPs only carry intra macroblocks (§6.2.5 — "an I-VOP shall
    /// only contain macroblocks of type intra or intra+q"), so the
    /// other rows are typed-rejected at construction time.
    IVopWithInterMbType(DerivedMbType),
    /// An S(GMC)-VOP macroblock was constructed with
    /// `derived_mb_type >= 2`. The §6.2.6.3 S-disjunct requires
    /// `derived_mb_type < 2`; the higher rows leave the
    /// `field_prediction` gate unfired (the dct_type gate still
    /// applies — callers should route those via [`InterlacedInfoContext::PVop`]
    /// equivalents instead).
    SGmcVopWithHighMbType(DerivedMbType),
}

impl core::fmt::Display for InterlacedInfoContextError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InterlacedInfoContextError::IVopWithInterMbType(mb) => write!(
                f,
                "§6.2.6.3: I-VOP macroblock must be intra-coded, got {mb:?}"
            ),
            InterlacedInfoContextError::SGmcVopWithHighMbType(mb) => write!(
                f,
                "§6.2.6.3: S(GMC)-VOP macroblock for field_prediction gate \
                 requires derived_mb_type < 2, got {mb:?}"
            ),
        }
    }
}

impl std::error::Error for InterlacedInfoContextError {}

impl InterlacedInfoContext {
    /// Construct the I-VOP context. Refuses inter `mb_type` rows
    /// per §6.2.5's I-VOP intra-only invariant.
    pub fn i_vop(mb_type: DerivedMbType) -> Result<Self, InterlacedInfoContextError> {
        if mb_type.is_intra() {
            Ok(InterlacedInfoContext::IVop { mb_type })
        } else {
            Err(InterlacedInfoContextError::IVopWithInterMbType(mb_type))
        }
    }

    /// Construct the P-VOP context.
    pub fn p_vop(mb_type: DerivedMbType, any_block_coded: bool) -> Self {
        InterlacedInfoContext::PVop {
            mb_type,
            any_block_coded,
        }
    }

    /// Construct the B-VOP context.
    pub fn b_vop(mb_type: BVopMbType, any_block_coded: bool) -> Self {
        InterlacedInfoContext::BVop {
            mb_type,
            any_block_coded,
        }
    }

    /// Construct the S(GMC)-VOP context for an inter / inter+q
    /// macroblock. Refuses `derived_mb_type >= 2` (the §6.2.6.3
    /// S-disjunct's `derived_mb_type < 2` clause). With `mcsel == 1`
    /// the body still runs (§6.2.6's `if (interlaced)` line is
    /// unconditional) but only its `dct_type` gate can fire
    /// ([`InterlacedInfoContext::SGmcVopGmc`]); the `mcsel`
    /// discrimination is at the type level via [`McSel`].
    pub fn s_gmc_vop(
        mb_type: DerivedMbType,
        mcsel: McSel,
        any_block_coded: bool,
    ) -> Result<Self, InterlacedInfoContextError> {
        match mb_type {
            DerivedMbType::Inter | DerivedMbType::InterQ => Ok(match mcsel {
                McSel::On => InterlacedInfoContext::SGmcVopGmc { any_block_coded },
                McSel::Off => InterlacedInfoContext::SGmcVop {
                    mb_type,
                    any_block_coded,
                },
            }),
            _ => Err(InterlacedInfoContextError::SGmcVopWithHighMbType(mb_type)),
        }
    }
}

/// §6.2.6.3 first gate — `dct_type` is present in the bitstream iff
/// the macroblock is intra-coded (`derived_mb_type ∈ {3, 4}`) or any
/// 8x8 block is coded (`cbp != 0`).
pub fn dct_type_present(ctx: &InterlacedInfoContext) -> bool {
    match *ctx {
        // I-VOP macroblocks are always intra-coded — the first clause
        // of the §6.2.6.3 disjunction (derived_mb_type ∈ {3, 4})
        // always fires.
        InterlacedInfoContext::IVop { .. } => true,
        InterlacedInfoContext::PVop {
            mb_type,
            any_block_coded,
        }
        | InterlacedInfoContext::SGmcVop {
            mb_type,
            any_block_coded,
        } => mb_type.is_intra() || any_block_coded,
        // A GMC-predicted macroblock is never intra: only `cbp != 0`.
        InterlacedInfoContext::SGmcVopGmc { any_block_coded } => any_block_coded,
        // B-VOP macroblocks are never intra-coded (Table B.4 / B.5
        // only carry the inter rows). The first clause therefore
        // only fires when any 8x8 block is coded (cbp != 0).
        InterlacedInfoContext::BVop {
            any_block_coded, ..
        } => any_block_coded,
    }
}

/// §6.2.6.3 second gate — the `field_prediction` block is present in
/// the bitstream iff one of the three §6.2.6.3 disjuncts fires:
///
/// * P-VOP with `derived_mb_type ∈ {0, 1}`,
/// * S(GMC)-VOP with `derived_mb_type < 2` and `mcsel == 0`,
/// * B-VOP with `mb_type != "1"` ([`BVopMbType::Direct`] — `"1"` is
///   the Table B.4 / B.5 direct-mode row).
pub fn field_prediction_present(ctx: &InterlacedInfoContext) -> bool {
    match *ctx {
        InterlacedInfoContext::IVop { .. } => false,
        // §6.2.6.3 `!mcsel`: a GMC macroblock is frame-predicted.
        InterlacedInfoContext::SGmcVopGmc { .. } => false,
        InterlacedInfoContext::PVop { mb_type, .. } => {
            matches!(mb_type, DerivedMbType::Inter | DerivedMbType::InterQ)
        }
        // SGmcVop's constructor already refused mb_type >= 2 and
        // mcsel == 1, so reaching this arm means the disjunct fires.
        InterlacedInfoContext::SGmcVop { .. } => true,
        InterlacedInfoContext::BVop { mb_type, .. } => !matches!(mb_type, BVopMbType::Direct),
    }
}

/// §6.2.6.3 forward sub-gate — the forward reference pair is present
/// when `field_prediction == 1` and one of:
///
/// * the VOP is P-coded,
/// * the VOP is S-coded,
/// * the VOP is B-coded with `mb_type != "001"` (Table B.4 / B.5
///   backward-only row).
fn forward_reference_pair_present(ctx: &InterlacedInfoContext) -> bool {
    match *ctx {
        InterlacedInfoContext::IVop { .. } | InterlacedInfoContext::SGmcVopGmc { .. } => false,
        InterlacedInfoContext::PVop { .. } | InterlacedInfoContext::SGmcVop { .. } => true,
        InterlacedInfoContext::BVop { mb_type, .. } => !matches!(mb_type, BVopMbType::Backward),
    }
}

/// §6.2.6.3 backward sub-gate — the backward reference pair is present
/// when `field_prediction == 1`, the VOP is B-coded, and `mb_type !=
/// "0001"` (Table B.4 / B.5 forward-only row).
fn backward_reference_pair_present(ctx: &InterlacedInfoContext) -> bool {
    match *ctx {
        InterlacedInfoContext::BVop { mb_type, .. } => !matches!(mb_type, BVopMbType::Forward),
        _ => false,
    }
}

/// Parse the §6.2.6.3 `interlaced_information()` body.
///
/// The number of bits consumed is determined entirely by `ctx`:
///
/// * `0` if neither gate fires (e.g. a P-VOP macroblock with
///   `derived_mb_type ∈ {2, 3, 4}` and `cbp == 0`).
/// * `1` if only the `dct_type` gate fires, or only the
///   `field_prediction == 0` case fires.
/// * `2` if `dct_type` and `field_prediction == 0` both fire.
/// * `3..=5` for combinations of `dct_type` + `field_prediction == 1`
///   + one or both reference pairs.
///
/// The caller is responsible for the outer `if (interlaced)` gate;
/// this function assumes it has fired.
pub fn parse_interlaced_information(
    br: &mut BitReader<'_>,
    ctx: &InterlacedInfoContext,
) -> Result<InterlacedInformation, BitReaderError> {
    let dct_type = if dct_type_present(ctx) {
        Some(DctType::from_bit(br.read_bool()?))
    } else {
        None
    };

    let (field_prediction, guard_fired) = if field_prediction_present(ctx) {
        let bit = br.read_bool()?;
        if !bit {
            (None, true)
        } else {
            let forward = if forward_reference_pair_present(ctx) {
                let top = FieldReference::from_bit(br.read_bool()?);
                let bottom = FieldReference::from_bit(br.read_bool()?);
                Some((top, bottom))
            } else {
                None
            };
            let backward = if backward_reference_pair_present(ctx) {
                let top = FieldReference::from_bit(br.read_bool()?);
                let bottom = FieldReference::from_bit(br.read_bool()?);
                Some((top, bottom))
            } else {
                None
            };
            (Some(FieldPrediction { forward, backward }), true)
        }
    } else {
        (None, false)
    };

    Ok(InterlacedInformation {
        dct_type,
        field_prediction,
        field_prediction_guard_fired: guard_fired,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn br_from_bits(bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, b) in bits.iter().enumerate() {
            if *b != 0 {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        out
    }

    // -----------------------------------------------------------------
    // §6.2.6.3 first guard — dct_type presence.
    // -----------------------------------------------------------------

    #[test]
    fn i_vop_always_has_dct_type() {
        let ctx = InterlacedInfoContext::i_vop(DerivedMbType::Intra).unwrap();
        assert!(dct_type_present(&ctx));
        let ctx = InterlacedInfoContext::i_vop(DerivedMbType::IntraQ).unwrap();
        assert!(dct_type_present(&ctx));
    }

    #[test]
    fn i_vop_refuses_inter_mb_types() {
        for mb in [
            DerivedMbType::Inter,
            DerivedMbType::InterQ,
            DerivedMbType::Inter4V,
        ] {
            let err = InterlacedInfoContext::i_vop(mb).unwrap_err();
            assert!(matches!(
                err,
                InterlacedInfoContextError::IVopWithInterMbType(_)
            ));
        }
    }

    #[test]
    fn p_vop_intra_has_dct_type_regardless_of_cbp() {
        for mb in [DerivedMbType::Intra, DerivedMbType::IntraQ] {
            assert!(dct_type_present(&InterlacedInfoContext::p_vop(mb, false)));
            assert!(dct_type_present(&InterlacedInfoContext::p_vop(mb, true)));
        }
    }

    #[test]
    fn p_vop_inter_has_dct_type_only_when_cbp_nonzero() {
        for mb in [
            DerivedMbType::Inter,
            DerivedMbType::InterQ,
            DerivedMbType::Inter4V,
        ] {
            assert!(!dct_type_present(&InterlacedInfoContext::p_vop(mb, false)));
            assert!(dct_type_present(&InterlacedInfoContext::p_vop(mb, true)));
        }
    }

    #[test]
    fn b_vop_dct_type_follows_cbp_only() {
        for mb in [
            BVopMbType::Direct,
            BVopMbType::Forward,
            BVopMbType::Backward,
            BVopMbType::Interpolated,
        ] {
            assert!(!dct_type_present(&InterlacedInfoContext::b_vop(mb, false)));
            assert!(dct_type_present(&InterlacedInfoContext::b_vop(mb, true)));
        }
    }

    // -----------------------------------------------------------------
    // §6.2.6.3 second guard — field_prediction presence.
    // -----------------------------------------------------------------

    #[test]
    fn i_vop_never_has_field_prediction() {
        let ctx = InterlacedInfoContext::i_vop(DerivedMbType::Intra).unwrap();
        assert!(!field_prediction_present(&ctx));
        let ctx = InterlacedInfoContext::i_vop(DerivedMbType::IntraQ).unwrap();
        assert!(!field_prediction_present(&ctx));
    }

    #[test]
    fn p_vop_field_prediction_only_for_inter_or_interq() {
        assert!(field_prediction_present(&InterlacedInfoContext::p_vop(
            DerivedMbType::Inter,
            false
        )));
        assert!(field_prediction_present(&InterlacedInfoContext::p_vop(
            DerivedMbType::InterQ,
            false
        )));
        for mb in [
            DerivedMbType::Inter4V,
            DerivedMbType::Intra,
            DerivedMbType::IntraQ,
        ] {
            assert!(!field_prediction_present(&InterlacedInfoContext::p_vop(
                mb, false
            )));
        }
    }

    #[test]
    fn b_vop_field_prediction_excludes_direct_only() {
        for mb in [
            BVopMbType::Forward,
            BVopMbType::Backward,
            BVopMbType::Interpolated,
        ] {
            assert!(field_prediction_present(&InterlacedInfoContext::b_vop(
                mb, false
            )));
        }
        assert!(!field_prediction_present(&InterlacedInfoContext::b_vop(
            BVopMbType::Direct,
            false
        )));
    }

    #[test]
    fn s_gmc_vop_mcsel_on_keeps_only_the_dct_type_gate() {
        // mcsel == 1 → the §6.2.6.3 S-disjunct is gated off (no
        // field_prediction), but dct_type still follows cbp != 0.
        let uncoded =
            InterlacedInfoContext::s_gmc_vop(DerivedMbType::Inter, McSel::On, false).unwrap();
        assert!(!dct_type_present(&uncoded));
        assert!(!field_prediction_present(&uncoded));
        let coded =
            InterlacedInfoContext::s_gmc_vop(DerivedMbType::InterQ, McSel::On, true).unwrap();
        assert!(dct_type_present(&coded));
        assert!(!field_prediction_present(&coded));
    }

    #[test]
    fn s_gmc_vop_high_mb_type_rejected() {
        for mb in [
            DerivedMbType::Inter4V,
            DerivedMbType::Intra,
            DerivedMbType::IntraQ,
        ] {
            let err = InterlacedInfoContext::s_gmc_vop(mb, McSel::Off, false).unwrap_err();
            assert!(matches!(
                err,
                InterlacedInfoContextError::SGmcVopWithHighMbType(_)
            ));
        }
    }

    #[test]
    fn s_gmc_vop_low_mb_type_field_prediction_fires() {
        for mb in [DerivedMbType::Inter, DerivedMbType::InterQ] {
            let ctx = InterlacedInfoContext::s_gmc_vop(mb, McSel::Off, false).unwrap();
            assert!(field_prediction_present(&ctx));
        }
    }

    // -----------------------------------------------------------------
    // Parser — bit-counted end-to-end coverage.
    // -----------------------------------------------------------------

    #[test]
    fn parse_p_vop_intra_dct_field_only() {
        // I-coded MB in a P-VOP: dct_type fires, field_prediction
        // does not. One bit consumed (the dct_type value `1` →
        // Field).
        let bytes = br_from_bits(&[1]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::p_vop(DerivedMbType::Intra, true),
        )
        .unwrap();
        assert_eq!(info.dct_type, Some(DctType::Field));
        assert_eq!(info.field_prediction, None);
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 1);
        assert_eq!(br.bit_position(), 1);
    }

    #[test]
    fn parse_p_vop_inter_cbp_zero_field_prediction_off() {
        // Inter MB, cbp == 0 → dct_type does NOT fire, but
        // field_prediction does. The field_prediction bit is 0 →
        // frame prediction → no further bits.
        let bytes = br_from_bits(&[0]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::p_vop(DerivedMbType::Inter, false),
        )
        .unwrap();
        assert_eq!(info.dct_type, None);
        assert_eq!(info.field_prediction, None);
        assert!(info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 1);
        assert_eq!(br.bit_position(), 1);
    }

    #[test]
    fn parse_p_vop_inter_cbp_nonzero_field_prediction_on() {
        // Inter MB, cbp != 0, field_prediction == 1.
        // Bits: dct_type=0, field_prediction=1, forward_top=1,
        // forward_bottom=0. P-VOP has no backward pair.
        let bytes = br_from_bits(&[0, 1, 1, 0]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::p_vop(DerivedMbType::InterQ, true),
        )
        .unwrap();
        assert_eq!(info.dct_type, Some(DctType::Frame));
        let fp = info.field_prediction.expect("field_prediction == 1");
        assert_eq!(
            fp.forward,
            Some((FieldReference::Bottom, FieldReference::Top))
        );
        assert_eq!(fp.backward, None);
        assert!(info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 4);
        assert_eq!(br.bit_position(), 4);
    }

    #[test]
    fn parse_p_vop_inter4v_no_bits() {
        // Inter4V MB (derived_mb_type == 2): neither gate fires
        // (Inter4V is not intra-coded and cbp == 0; the field_pred
        // disjunct requires derived_mb_type ∈ {0, 1}).
        let bytes = br_from_bits(&[]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::p_vop(DerivedMbType::Inter4V, false),
        )
        .unwrap();
        assert_eq!(info.dct_type, None);
        assert_eq!(info.field_prediction, None);
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 0);
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn parse_b_vop_interpolated_both_pairs_present() {
        // B-VOP Interpolated MB: dct_type fires (cbp != 0),
        // field_prediction fires (mb_type != Direct), forward pair
        // fires (mb_type != Backward), backward pair fires
        // (mb_type != Forward). 5 bits.
        // Layout: dct=1, fp=1, ftop=0, fbot=1, btop=1, bbot=0 → 6 bits.
        let bytes = br_from_bits(&[1, 1, 0, 1, 1, 0]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::b_vop(BVopMbType::Interpolated, true),
        )
        .unwrap();
        assert_eq!(info.dct_type, Some(DctType::Field));
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
        assert_eq!(br.bit_position(), 6);
    }

    #[test]
    fn parse_b_vop_forward_only_pair() {
        // B-VOP Forward MB: dct_type follows cbp; field_prediction
        // fires (Forward != Direct), forward pair fires
        // (Forward != Backward), backward pair does NOT fire
        // (mb_type == Forward → mb_type == "0001" → suppressed).
        // Layout (cbp = 0): no dct_type. fp=1, ftop=0, fbot=0. 3 bits.
        let bytes = br_from_bits(&[1, 0, 0]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::b_vop(BVopMbType::Forward, false),
        )
        .unwrap();
        assert_eq!(info.dct_type, None);
        let fp = info.field_prediction.unwrap();
        assert_eq!(fp.forward, Some((FieldReference::Top, FieldReference::Top)));
        assert_eq!(fp.backward, None);
        assert_eq!(info.bit_count(), 3);
        assert_eq!(br.bit_position(), 3);
    }

    #[test]
    fn parse_b_vop_backward_only_pair() {
        // B-VOP Backward MB: forward pair does NOT fire
        // (mb_type == Backward → "001"). Only backward pair fires.
        // Layout (cbp = 0): fp=1, btop=1, bbot=1. 3 bits.
        let bytes = br_from_bits(&[1, 1, 1]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::b_vop(BVopMbType::Backward, false),
        )
        .unwrap();
        assert_eq!(info.dct_type, None);
        let fp = info.field_prediction.unwrap();
        assert_eq!(fp.forward, None);
        assert_eq!(
            fp.backward,
            Some((FieldReference::Bottom, FieldReference::Bottom))
        );
        assert_eq!(info.bit_count(), 3);
        assert_eq!(br.bit_position(), 3);
    }

    #[test]
    fn parse_b_vop_direct_zero_bits_when_no_cbp() {
        // B-VOP Direct MB with cbp == 0: neither gate fires.
        let bytes = br_from_bits(&[]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::b_vop(BVopMbType::Direct, false),
        )
        .unwrap();
        assert_eq!(info.dct_type, None);
        assert_eq!(info.field_prediction, None);
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 0);
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn parse_b_vop_direct_dct_only_when_cbp_set() {
        // B-VOP Direct with cbp != 0: dct_type fires, field_pred
        // does not (Direct is the "1" row).
        let bytes = br_from_bits(&[1]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::b_vop(BVopMbType::Direct, true),
        )
        .unwrap();
        assert_eq!(info.dct_type, Some(DctType::Field));
        assert_eq!(info.field_prediction, None);
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 1);
    }

    #[test]
    fn parse_i_vop_dct_only_one_bit() {
        // I-VOP MB: dct_type fires, field_prediction does not.
        // cbp value is irrelevant.
        let bytes = br_from_bits(&[0]);
        let mut br = BitReader::new(&bytes);
        let ctx = InterlacedInfoContext::i_vop(DerivedMbType::Intra).unwrap();
        let info = parse_interlaced_information(&mut br, &ctx).unwrap();
        assert_eq!(info.dct_type, Some(DctType::Frame));
        assert_eq!(info.field_prediction, None);
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(info.bit_count(), 1);
    }

    #[test]
    fn parse_s_gmc_vop_mcsel_off_full_body() {
        // S-VOP mcsel == 0 Inter MB: dct_type follows cbp,
        // field_prediction fires (no backward pair — S-VOP has only
        // forward reference per §7.8). cbp == 1 here.
        // Bits: dct=0, fp=1, ftop=1, fbot=1. 4 bits.
        let bytes = br_from_bits(&[0, 1, 1, 1]);
        let mut br = BitReader::new(&bytes);
        let ctx = InterlacedInfoContext::s_gmc_vop(DerivedMbType::Inter, McSel::Off, true).unwrap();
        let info = parse_interlaced_information(&mut br, &ctx).unwrap();
        assert_eq!(info.dct_type, Some(DctType::Frame));
        let fp = info.field_prediction.unwrap();
        assert_eq!(
            fp.forward,
            Some((FieldReference::Bottom, FieldReference::Bottom))
        );
        assert_eq!(fp.backward, None);
        assert_eq!(info.bit_count(), 4);
    }

    #[test]
    fn parse_truncated_input_errors() {
        // Confirm the parser surfaces `EndOfStream` rather than
        // half-decoding a body when the bit slice runs out mid-field.
        // A P-VOP Inter MB with `cbp != 0` consumes 1..=4 bits (dct
        // + fp + possibly the two forward reference bits); an empty
        // slice cannot even read the `dct_type`.
        let empty: [u8; 0] = [];
        let mut br = BitReader::new(&empty);
        let err = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::p_vop(DerivedMbType::Inter, true),
        )
        .unwrap_err();
        assert!(matches!(err, BitReaderError::EndOfStream));
    }

    // -----------------------------------------------------------------
    // Round-trip / property tests.
    // -----------------------------------------------------------------

    #[test]
    fn dct_type_bit_round_trip() {
        for b in [false, true] {
            assert_eq!(DctType::from_bit(b).as_bit(), b);
        }
    }

    #[test]
    fn field_reference_bit_round_trip() {
        for b in [false, true] {
            assert_eq!(FieldReference::from_bit(b).as_bit(), b);
        }
    }

    #[test]
    fn bit_count_matches_parsed_position_across_all_contexts() {
        // Sweep: every (ctx, bit_payload) pair where the parser
        // succeeds — `bit_count()` must equal the bit reader's
        // `bit_position()` after parsing. Confirms the §6.2.6.3
        // body never silently drops or overshoots its bit budget.
        let contexts: &[InterlacedInfoContext] = &[
            InterlacedInfoContext::i_vop(DerivedMbType::Intra).unwrap(),
            InterlacedInfoContext::i_vop(DerivedMbType::IntraQ).unwrap(),
            InterlacedInfoContext::p_vop(DerivedMbType::Inter, false),
            InterlacedInfoContext::p_vop(DerivedMbType::Inter, true),
            InterlacedInfoContext::p_vop(DerivedMbType::InterQ, false),
            InterlacedInfoContext::p_vop(DerivedMbType::InterQ, true),
            InterlacedInfoContext::p_vop(DerivedMbType::Inter4V, false),
            InterlacedInfoContext::p_vop(DerivedMbType::Inter4V, true),
            InterlacedInfoContext::p_vop(DerivedMbType::Intra, false),
            InterlacedInfoContext::p_vop(DerivedMbType::Intra, true),
            InterlacedInfoContext::p_vop(DerivedMbType::IntraQ, false),
            InterlacedInfoContext::p_vop(DerivedMbType::IntraQ, true),
            InterlacedInfoContext::b_vop(BVopMbType::Direct, false),
            InterlacedInfoContext::b_vop(BVopMbType::Direct, true),
            InterlacedInfoContext::b_vop(BVopMbType::Forward, false),
            InterlacedInfoContext::b_vop(BVopMbType::Forward, true),
            InterlacedInfoContext::b_vop(BVopMbType::Backward, false),
            InterlacedInfoContext::b_vop(BVopMbType::Backward, true),
            InterlacedInfoContext::b_vop(BVopMbType::Interpolated, false),
            InterlacedInfoContext::b_vop(BVopMbType::Interpolated, true),
            InterlacedInfoContext::s_gmc_vop(DerivedMbType::Inter, McSel::Off, false).unwrap(),
            InterlacedInfoContext::s_gmc_vop(DerivedMbType::Inter, McSel::Off, true).unwrap(),
            InterlacedInfoContext::s_gmc_vop(DerivedMbType::InterQ, McSel::Off, false).unwrap(),
            InterlacedInfoContext::s_gmc_vop(DerivedMbType::InterQ, McSel::Off, true).unwrap(),
        ];
        // A worst-case payload of 6 set bits covers every combination
        // (dct + fp + fwd_top + fwd_bot + bwd_top + bwd_bot).
        let bytes = br_from_bits(&[1, 1, 1, 1, 1, 1]);
        for ctx in contexts {
            let mut br = BitReader::new(&bytes);
            let info = parse_interlaced_information(&mut br, ctx).unwrap();
            assert_eq!(
                info.bit_count(),
                br.bit_position(),
                "bit_count mismatch for ctx {ctx:?}"
            );
        }
    }

    #[test]
    fn s_gmc_vop_mcsel_on_parses_exactly_the_dct_type_bit() {
        // mcsel == 1, cbp != 0: one dct_type bit, nothing else.
        let ctx = InterlacedInfoContext::s_gmc_vop(DerivedMbType::Inter, McSel::On, true).unwrap();
        let bytes = br_from_bits(&[1, 1, 1]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(&mut br, &ctx).unwrap();
        assert_eq!(info.dct_type, Some(DctType::Field));
        assert!(info.field_prediction.is_none());
        assert!(!info.field_prediction_guard_fired);
        assert_eq!(br.bit_position(), 1);
        // mcsel == 1, cbp == 0: no bits at all.
        let ctx = InterlacedInfoContext::s_gmc_vop(DerivedMbType::Inter, McSel::On, false).unwrap();
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(&mut br, &ctx).unwrap();
        assert!(info.dct_type.is_none());
        assert_eq!(br.bit_position(), 0);
    }

    #[test]
    fn field_prediction_guard_fired_distinguishes_zero_from_absent() {
        // A field_prediction value of 0 surfaces with the guard-fired
        // flag set; an absent field_prediction (gate not fired)
        // leaves it clear. This is the discriminant callers need to
        // decide whether to skip the §6.2.6 "interlaced &&
        // field_prediction" motion-vector second-invocation.
        // Case A: P-VOP Inter, fp=0.
        let bytes = br_from_bits(&[0]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::p_vop(DerivedMbType::Inter, false),
        )
        .unwrap();
        assert!(info.field_prediction_guard_fired);
        assert!(info.field_prediction.is_none());
        // Case B: P-VOP Inter4V — gate absent.
        let bytes = br_from_bits(&[]);
        let mut br = BitReader::new(&bytes);
        let info = parse_interlaced_information(
            &mut br,
            &InterlacedInfoContext::p_vop(DerivedMbType::Inter4V, false),
        )
        .unwrap();
        assert!(!info.field_prediction_guard_fired);
        assert!(info.field_prediction.is_none());
    }
}

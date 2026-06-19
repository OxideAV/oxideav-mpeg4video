//! §7.3.5 / Table 7-2 — per-8×8-block inverse-transform selection.
//!
//! Once the inverse-quantised coefficient block `F[v][u]` (laid out in the
//! `PQF[v][u]` packing produced by the §7.4.2 modified inverse scan) is
//! ready, the decoder must decide *which* inverse transform reconverts it
//! to the texture `f[y][x]`. ISO/IEC 14496-2:2004 §7.3.5 specifies three
//! transforms and the decision rule that picks between them (Table 7-2):
//!
//! * the textbook 8×8 inverse DCT (Annex A §A.1 / [`crate::idct::idct_8x8`]),
//! * the inverse shape-adaptive DCT (Annex A §A.3.2 /
//!   [`crate::inverse_sadct::inverse_sadct`]),
//! * the inverse ∆DC-SA-DCT (Annex A §A.4.2 /
//!   [`crate::inverse_sadct::inverse_delta_dc_sadct`]).
//!
//! Until this module, all three transform bodies existed but the §7.3.5
//! decision that routes a given block to one of them was unwritten. This
//! module is that routing: it transcribes Table 7-2 into [`select_transform`]
//! and applies the chosen transform in [`inverse_transform_block`].
//!
//! ## Table 7-2 (verbatim decision rule)
//!
//! | decision rule | DCT tool |
//! | --- | --- |
//! | `(video_object_layer_shape == "rectangular") \|\| (sadct_disable == 1) \|\| (opaque_pels == 64)` | 8×8-DCT |
//! | `(video_object_layer_shape != "rectangular") && (sadct_disable == 0) && (opaque_pels < 64) && (vop_coding_type != "B") && ((derived_mb_type == 3) \|\| (derived_mb_type == 4))` | ∆DC-SA-DCT |
//! | `(video_object_layer_shape != "rectangular") && (sadct_disable == 0) && (opaque_pels < 64) && (((vop_coding_type == "P") && (derived_mb_type != 3) && (derived_mb_type != 4)) \|\| (vop_coding_type == "B"))` | SA-DCT |
//!
//! The first row's `opaque_pels == 64` clause subsumes the common
//! rectangular / fully-opaque case onto the plain 8×8-DCT regardless of
//! `sadct_disable`. The remaining two rows partition the `opaque_pels < 64`
//! non-rectangular `sadct_disable == 0` space by VOP coding type and
//! `derived_mb_type`:
//!
//! * ∆DC-SA-DCT applies to **intra** blocks (`derived_mb_type ∈ {3, 4}`,
//!   i.e. [`DerivedMbType::Intra`] / [`DerivedMbType::IntraQ`]) of a
//!   non-B VOP — the extended SA-DCT that carries the block mean as a
//!   re-scaled DC term (Annex A §A.4.2).
//! * SA-DCT applies to **inter** blocks of a P-VOP (`derived_mb_type ∉
//!   {3, 4}`) and to **every** block of a B-VOP — the plain shape-adaptive
//!   transform (Annex A §A.3.2).
//!
//! Note that a B-VOP intra block (the `vop_coding_type == "B"` arm of the
//! SA-DCT row) takes plain SA-DCT, **not** ∆DC-SA-DCT: the ∆DC variant is
//! gated on `vop_coding_type != "B"`. The two rows are mutually exclusive
//! and jointly exhaustive over the `opaque_pels < 64` non-rectangular
//! `sadct_disable == 0` space, so the selection is total.
//!
//! ## References
//!
//! * §7.3.5 / Table 7-2 — the decision rule (ISO/IEC 14496-2:2004).
//! * Annex A §A.1 — the 8×8 inverse DCT.
//! * Annex A §A.3.2 — the inverse SA-DCT.
//! * Annex A §A.4.2 — the inverse ∆DC-SA-DCT.

use crate::idct::idct_8x8;
use crate::inverse_sadct::{inverse_delta_dc_sadct, inverse_sadct};
use crate::macroblock::DerivedMbType;
use crate::sample_padding::SamplePresence;
use crate::vop::VopCodingType;

/// `video_object_layer_shape == "rectangular"` (Table 6-16 value `0`).
///
/// The §7.3.5 decision only distinguishes rectangular from non-rectangular,
/// so the full Table 6-16 value is reduced to this boolean at the call
/// site. Round 2's VOL parser stores the raw `u8`; rectangular is `0`.
pub const SHAPE_RECTANGULAR: u8 = 0;

/// The three §7.3.5 inverse transforms a block may be routed to.
///
/// The variants correspond one-to-one to the "DCT tool" column of
/// Table 7-2 and to the three transform bodies already implemented in
/// the crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseTransform {
    /// The textbook 8×8 inverse DCT (Annex A §A.1).
    ///
    /// Selected for rectangular VOLs, `sadct_disable == 1`, or any block
    /// whose 64 pels are all opaque — Table 7-2 row 1.
    Dct8x8,
    /// The inverse shape-adaptive DCT (Annex A §A.3.2).
    ///
    /// Selected for `opaque_pels < 64` blocks of a non-rectangular
    /// `sadct_disable == 0` VOL that are either inter blocks of a P-VOP or
    /// any block of a B-VOP — Table 7-2 row 3.
    Sadct,
    /// The inverse ∆DC-SA-DCT (Annex A §A.4.2).
    ///
    /// Selected for `opaque_pels < 64` intra blocks (`derived_mb_type ∈
    /// {3, 4}`) of a non-B, non-rectangular `sadct_disable == 0` VOL —
    /// Table 7-2 row 2.
    DeltaDcSadct,
}

/// The inputs to the §7.3.5 / Table 7-2 decision rule.
///
/// Every field is a header- or shape-derived quantity already produced
/// upstream in the decode pipeline:
///
/// * `shape` and `sadct_disable` come from the VOL header
///   ([`crate::vol::VolHeader::video_object_layer_shape`] reduced via
///   [`SHAPE_RECTANGULAR`], and the §6.2.3 `sadct_disable` flag);
/// * `coding_type` comes from the VOP header
///   ([`crate::vop::VopCodingType`]);
/// * `derived_mb_type` comes from the macroblock layer
///   ([`crate::macroblock::DerivedMbType`]);
/// * `opaque_pels` is the §7.4.2 / Annex A §A.3.2 count of opaque samples
///   in the 8×8 block, derived from the decoded binary shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformSelection {
    /// Whether the VOL is rectangular (Table 6-16 value `0`).
    ///
    /// `true` forces the 8×8-DCT row of Table 7-2 unconditionally; the
    /// SA-DCT family only exists for non-rectangular (arbitrary-shape)
    /// VOLs.
    pub rectangular: bool,
    /// The §6.2.3 `sadct_disable` flag. `true` (the spec's `== 1`) forces
    /// the 8×8-DCT for all blocks; `false` enables the adaptive family.
    pub sadct_disable: bool,
    /// The §6.3.5 `vop_coding_type` of the current VOP.
    pub coding_type: VopCodingType,
    /// The §6.3.6 `derived_mb_type` of the macroblock owning the block.
    /// `None` is treated as a non-intra inter type (the B-VOP / skipped
    /// case carries no `mcbpc`-derived type); see [`select_transform`].
    pub derived_mb_type: Option<DerivedMbType>,
    /// `opaque_pels` — the count of opaque samples in the 8×8 block,
    /// `0..=64`. `64` means the block is fully opaque.
    pub opaque_pels: u8,
}

/// Apply Table 7-2 and return the chosen inverse transform.
///
/// The three rows are evaluated in spec order. The first row is the
/// "fall-through to plain DCT" predicate (`rectangular || sadct_disable ||
/// opaque_pels == 64`); when it holds, the SA-DCT family does not apply.
/// Only when it fails — a non-rectangular `sadct_disable == 0` block with
/// `opaque_pels < 64` — do rows 2 and 3 partition the choice by VOP coding
/// type and `derived_mb_type`.
///
/// ## `derived_mb_type` treatment
///
/// Table 7-2 references `derived_mb_type == 3` / `== 4` (intra / intra+Q).
/// [`DerivedMbType::value`] returns exactly those integers for the intra
/// variants, so the predicate is expressed against
/// [`DerivedMbType::is_intra`]. A `None` `derived_mb_type` (a block whose
/// macroblock carries no decoded type, e.g. a not-coded / skipped MB) is
/// treated as non-intra: such a block carries no texture to transform,
/// but should it be routed here it falls to the inter SA-DCT arm rather
/// than the intra ∆DC arm, matching the `derived_mb_type != 3 && != 4`
/// branch.
///
/// ## Totality
///
/// For a non-rectangular `sadct_disable == 0` `opaque_pels < 64` block the
/// two SA-DCT rows are exhaustive: a non-B intra block takes ∆DC-SA-DCT
/// (row 2), and everything else — P-VOP inter, B-VOP intra, B-VOP inter —
/// takes SA-DCT (row 3). The I-VOP case cannot reach here, because an
/// arbitrary-shape I-VOP block with `opaque_pels < 64` is intra and
/// `vop_coding_type != "B"`, hence ∆DC-SA-DCT.
pub fn select_transform(sel: &TransformSelection) -> InverseTransform {
    // Table 7-2, row 1 — the plain 8×8-DCT predicate.
    if sel.rectangular || sel.sadct_disable || sel.opaque_pels == 64 {
        return InverseTransform::Dct8x8;
    }

    // From here: non-rectangular, sadct_disable == 0, opaque_pels < 64.
    let is_intra = sel
        .derived_mb_type
        .map(DerivedMbType::is_intra)
        .unwrap_or(false);

    // Table 7-2, row 2 — ∆DC-SA-DCT: intra block of a non-B VOP.
    if sel.coding_type != VopCodingType::B && is_intra {
        return InverseTransform::DeltaDcSadct;
    }

    // Table 7-2, row 3 — SA-DCT: P-VOP inter block, or any B-VOP block.
    InverseTransform::Sadct
}

/// Apply the §7.3.5-selected inverse transform to one 8×8 coefficient
/// block, returning the texture `f[y][x]`.
///
/// `pqf` is the inverse-quantised coefficient block `F[v][u]` in the
/// `PQF[v][u]` packing (rectangular path: the natural row/column layout
/// after the §7.4.1 inverse scan; SA-DCT path: the §7.4.2 modified
/// inverse-scan packing). `f_shape` is the decoded binary shape of the
/// block — required by both SA-DCT bodies and ignored by the plain DCT.
///
/// The return value is the raw inverse-transform output, *before* the
/// §7.3 prediction add and the §7.3 step-3 display clip — exactly what
/// [`crate::idct::idct_8x8`], [`crate::inverse_sadct::inverse_sadct`] and
/// [`crate::inverse_sadct::inverse_delta_dc_sadct`] each return. The
/// plain-DCT arm applies the §7.4.5 `[-2^bpp, 2^bpp-1]` saturation (it is
/// intrinsic to [`idct_8x8`]); the two SA-DCT arms return the rounded
/// transform output unsaturated, matching their respective Annex A
/// definitions.
pub fn inverse_transform_block(
    transform: InverseTransform,
    pqf: &[[i32; 8]; 8],
    f_shape: &[[SamplePresence; 8]; 8],
    bits_per_pixel: u32,
) -> [[i32; 8]; 8] {
    match transform {
        InverseTransform::Dct8x8 => idct_8x8(pqf, bits_per_pixel),
        InverseTransform::Sadct => inverse_sadct(pqf, f_shape),
        InverseTransform::DeltaDcSadct => inverse_delta_dc_sadct(pqf, f_shape),
    }
}

/// Convenience: select via Table 7-2 and apply the chosen transform in one
/// call.
///
/// Equivalent to `inverse_transform_block(select_transform(sel), pqf,
/// f_shape, bits_per_pixel)`; useful at the per-block reconstruction site
/// where the selection inputs and the coefficient block are both in hand.
pub fn select_and_inverse_transform(
    sel: &TransformSelection,
    pqf: &[[i32; 8]; 8],
    f_shape: &[[SamplePresence; 8]; 8],
    bits_per_pixel: u32,
) -> [[i32; 8]; 8] {
    inverse_transform_block(select_transform(sel), pqf, f_shape, bits_per_pixel)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_shape() -> [[SamplePresence; 8]; 8] {
        [[SamplePresence::Opaque; 8]; 8]
    }

    /// Build an `f_shape` from a bool opacity grid.
    fn shape(rows: [[bool; 8]; 8]) -> [[SamplePresence; 8]; 8] {
        let mut s = [[SamplePresence::Transparent; 8]; 8];
        for (y, row) in rows.iter().enumerate() {
            for (x, &op) in row.iter().enumerate() {
                if op {
                    s[y][x] = SamplePresence::Opaque;
                }
            }
        }
        s
    }

    fn count_opaque(f_shape: &[[SamplePresence; 8]; 8]) -> u8 {
        f_shape.iter().flatten().filter(|p| p.is_opaque()).count() as u8
    }

    // ----- Table 7-2 row 1: 8×8-DCT predicates -------------------------

    #[test]
    fn rectangular_forces_dct() {
        // Row 1, first clause: rectangular VOL → 8×8-DCT regardless of
        // everything else.
        let sel = TransformSelection {
            rectangular: true,
            sadct_disable: false,
            coding_type: VopCodingType::P,
            derived_mb_type: Some(DerivedMbType::Inter),
            opaque_pels: 30,
        };
        assert_eq!(select_transform(&sel), InverseTransform::Dct8x8);
    }

    #[test]
    fn sadct_disable_forces_dct() {
        // Row 1, second clause: sadct_disable == 1 → 8×8-DCT even for a
        // non-rectangular partially-opaque block.
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: true,
            coding_type: VopCodingType::I,
            derived_mb_type: Some(DerivedMbType::Intra),
            opaque_pels: 10,
        };
        assert_eq!(select_transform(&sel), InverseTransform::Dct8x8);
    }

    #[test]
    fn fully_opaque_forces_dct() {
        // Row 1, third clause: opaque_pels == 64 → 8×8-DCT even for a
        // non-rectangular sadct_disable == 0 VOL.
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: false,
            coding_type: VopCodingType::P,
            derived_mb_type: Some(DerivedMbType::Intra),
            opaque_pels: 64,
        };
        assert_eq!(select_transform(&sel), InverseTransform::Dct8x8);
    }

    // ----- Table 7-2 row 2: ∆DC-SA-DCT --------------------------------

    #[test]
    fn intra_i_vop_partial_is_delta_dc() {
        // I-VOP intra block, non-rectangular, sadct_disable == 0,
        // opaque_pels < 64 → ∆DC-SA-DCT (row 2: vop != B && intra).
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: false,
            coding_type: VopCodingType::I,
            derived_mb_type: Some(DerivedMbType::Intra),
            opaque_pels: 20,
        };
        assert_eq!(select_transform(&sel), InverseTransform::DeltaDcSadct);
    }

    #[test]
    fn intra_q_p_vop_partial_is_delta_dc() {
        // P-VOP intra+Q block (derived_mb_type == 4) → ∆DC-SA-DCT: row 2
        // covers both intra variants and any non-B VOP.
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: false,
            coding_type: VopCodingType::P,
            derived_mb_type: Some(DerivedMbType::IntraQ),
            opaque_pels: 40,
        };
        assert_eq!(select_transform(&sel), InverseTransform::DeltaDcSadct);
    }

    // ----- Table 7-2 row 3: SA-DCT ------------------------------------

    #[test]
    fn inter_p_vop_partial_is_sadct() {
        // P-VOP inter block (derived_mb_type != 3, 4) → SA-DCT (row 3,
        // first arm).
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: false,
            coding_type: VopCodingType::P,
            derived_mb_type: Some(DerivedMbType::Inter),
            opaque_pels: 33,
        };
        assert_eq!(select_transform(&sel), InverseTransform::Sadct);
    }

    #[test]
    fn inter4v_p_vop_partial_is_sadct() {
        // P-VOP Inter4V block is still inter (derived_mb_type == 2) →
        // SA-DCT.
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: false,
            coding_type: VopCodingType::P,
            derived_mb_type: Some(DerivedMbType::Inter4V),
            opaque_pels: 12,
        };
        assert_eq!(select_transform(&sel), InverseTransform::Sadct);
    }

    #[test]
    fn b_vop_inter_partial_is_sadct() {
        // B-VOP inter block → SA-DCT (row 3, second arm: vop == B).
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: false,
            coding_type: VopCodingType::B,
            derived_mb_type: Some(DerivedMbType::Inter),
            opaque_pels: 50,
        };
        assert_eq!(select_transform(&sel), InverseTransform::Sadct);
    }

    #[test]
    fn b_vop_intra_partial_is_sadct_not_delta_dc() {
        // The subtle case: a B-VOP *intra* block takes plain SA-DCT, NOT
        // ∆DC-SA-DCT — the ∆DC row is gated on vop_coding_type != "B".
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: false,
            coding_type: VopCodingType::B,
            derived_mb_type: Some(DerivedMbType::Intra),
            opaque_pels: 18,
        };
        assert_eq!(select_transform(&sel), InverseTransform::Sadct);
    }

    #[test]
    fn none_mb_type_partial_is_sadct() {
        // A block with no decoded derived_mb_type is treated as non-intra:
        // it falls to the SA-DCT inter arm rather than the ∆DC intra arm.
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: false,
            coding_type: VopCodingType::P,
            derived_mb_type: None,
            opaque_pels: 25,
        };
        assert_eq!(select_transform(&sel), InverseTransform::Sadct);
    }

    // ----- exhaustiveness over the SA-DCT space ------------------------

    #[test]
    fn sadct_space_is_total_and_partitioned() {
        // For every (coding_type, intra?) combination in the
        // non-rectangular sadct_disable==0 opaque_pels<64 space, the
        // selection is exactly one of the two SA-DCT variants, and the
        // ∆DC variant occurs iff (vop != B && intra).
        for &ct in &[
            VopCodingType::I,
            VopCodingType::P,
            VopCodingType::B,
            VopCodingType::S,
        ] {
            for &mbt in &[
                DerivedMbType::Inter,
                DerivedMbType::InterQ,
                DerivedMbType::Inter4V,
                DerivedMbType::Intra,
                DerivedMbType::IntraQ,
            ] {
                let sel = TransformSelection {
                    rectangular: false,
                    sadct_disable: false,
                    coding_type: ct,
                    derived_mb_type: Some(mbt),
                    opaque_pels: 30,
                };
                let got = select_transform(&sel);
                let expect_delta = ct != VopCodingType::B && mbt.is_intra();
                if expect_delta {
                    assert_eq!(got, InverseTransform::DeltaDcSadct, "ct={ct:?} mbt={mbt:?}");
                } else {
                    assert_eq!(got, InverseTransform::Sadct, "ct={ct:?} mbt={mbt:?}");
                }
            }
        }
    }

    // ----- application: dispatch matches the underlying bodies ---------

    #[test]
    fn apply_dct_matches_idct_8x8() {
        // The Dct8x8 arm must produce exactly idct_8x8's output.
        let mut pqf = [[0i32; 8]; 8];
        pqf[0][0] = 320;
        pqf[1][2] = -40;
        pqf[3][3] = 15;
        let got = inverse_transform_block(InverseTransform::Dct8x8, &pqf, &full_shape(), 8);
        assert_eq!(got, idct_8x8(&pqf, 8));
    }

    #[test]
    fn apply_sadct_matches_inverse_sadct() {
        // The Sadct arm must produce exactly inverse_sadct's output.
        let s = shape([
            [true, true, true, true, false, false, false, false],
            [true, true, true, true, false, false, false, false],
            [true, true, true, true, false, false, false, false],
            [true, true, true, true, false, false, false, false],
            [false; 8],
            [false; 8],
            [false; 8],
            [false; 8],
        ]);
        let mut pqf = [[0i32; 8]; 8];
        pqf[0][0] = 100;
        pqf[0][1] = -20;
        pqf[1][0] = 30;
        let got = inverse_transform_block(InverseTransform::Sadct, &pqf, &s, 8);
        assert_eq!(got, inverse_sadct(&pqf, &s));
    }

    #[test]
    fn apply_delta_dc_matches_inverse_delta_dc_sadct() {
        // The DeltaDcSadct arm must produce exactly
        // inverse_delta_dc_sadct's output.
        let s = shape([
            [true, true, true, true, true, true, false, false],
            [true, true, true, true, true, true, false, false],
            [true, true, true, true, true, true, false, false],
            [true, true, true, true, true, true, false, false],
            [true, true, true, true, true, true, false, false],
            [true, true, true, true, true, true, false, false],
            [false; 8],
            [false; 8],
        ]);
        let mut pqf = [[0i32; 8]; 8];
        pqf[0][0] = 256;
        pqf[1][1] = -12;
        let got = inverse_transform_block(InverseTransform::DeltaDcSadct, &pqf, &s, 8);
        assert_eq!(got, inverse_delta_dc_sadct(&pqf, &s));
    }

    #[test]
    fn select_and_apply_convenience_agrees() {
        // The one-call helper must equal the two-step form.
        let s = shape([
            [true, true, true, false, false, false, false, false],
            [true, true, true, false, false, false, false, false],
            [true, true, true, false, false, false, false, false],
            [false; 8],
            [false; 8],
            [false; 8],
            [false; 8],
            [false; 8],
        ]);
        let sel = TransformSelection {
            rectangular: false,
            sadct_disable: false,
            coding_type: VopCodingType::P,
            derived_mb_type: Some(DerivedMbType::Inter),
            opaque_pels: count_opaque(&s),
        };
        let mut pqf = [[0i32; 8]; 8];
        pqf[0][0] = 64;
        pqf[1][0] = 8;
        let one = select_and_inverse_transform(&sel, &pqf, &s, 8);
        let two = inverse_transform_block(select_transform(&sel), &pqf, &s, 8);
        assert_eq!(one, two);
    }

    #[test]
    fn rectangular_path_ignores_shape() {
        // For the Dct8x8 arm the f_shape argument is irrelevant: a
        // rectangular block always has a full opaque shape, but even a
        // partial shape passed in must not change the DCT output.
        let mut pqf = [[0i32; 8]; 8];
        pqf[0][0] = 200;
        let partial = shape([
            [true, true, false, false, false, false, false, false],
            [true, true, false, false, false, false, false, false],
            [false; 8],
            [false; 8],
            [false; 8],
            [false; 8],
            [false; 8],
            [false; 8],
        ]);
        let with_full = inverse_transform_block(InverseTransform::Dct8x8, &pqf, &full_shape(), 8);
        let with_partial = inverse_transform_block(InverseTransform::Dct8x8, &pqf, &partial, 8);
        assert_eq!(with_full, with_partial);
    }
}

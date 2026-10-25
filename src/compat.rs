//! Decode-behaviour selection where the ISO/IEC 14496-2 text and the
//! deployed decoder ecosystem disagree.
//!
//! This crate's default behaviour is always the **literal
//! specification text**. For three clauses, black-box comparison
//! against reference decodes of conformant streams (pixel-level output
//! comparison only — no implementation source was consulted) shows the
//! deployed ecosystem behaves differently, so real-world streams
//! produced/consumed by that ecosystem reconstruct slightly
//! differently from the printed clauses. [`DecodeOptions`] carries the
//! opt-in **ecosystem-compat** switch that reproduces the observed
//! behaviour bit-for-bit; it covers exactly these three divergences:
//!
//! 1. **§7.7.2.2 interlaced direct mode.** The spec derives the four
//!    field motion vectors "from the forward field motion vectors of
//!    the co-located macroblock of the future reference VOP" — the
//!    `MV[i]` term of the pseudo code. Observed reference decodes
//!    evaluate the same (erratum-corrected) derivation with the
//!    co-located field motion vectors read as **zero**, so the derived
//!    vectors reduce to `mvf[i] = mvb[i] = MVD[0]` on the field grid,
//!    while the forward reference fields still follow the co-located
//!    macroblock's field selections (root-caused by exhaustive
//!    per-macroblock-field pixel search — see `tests/conformance.rs`).
//!
//!    Constructed-probe arbitration with provably **non-zero**
//!    co-located field MVs over textured anchors
//!    (`tests/direct_mode_probes.rs`) confirmed this model
//!    unconditionally for a transmitted non-zero `MVD[0]` and for the
//!    zero-bit `modb == "1"` form. The same probes established a
//!    boundary the compat mode deliberately does **not** reproduce
//!    yet: for a direct macroblock whose `MVD[0]` is *transmitted but
//!    exactly (0, 0)*, the observed ecosystem runs **progressive**
//!    direct mode over the co-located frame vector
//!    `Div2Round(MVf1 + MVf2)` instead. No conformance-corpus stream
//!    contains such a macroblock; whether the compat mode should
//!    adopt that branch awaits a project ruling (the probe pins
//!    record both modes' measured envelopes so any change is
//!    deliberate).
//!
//! 2. **§7.4.4.5 mismatch control.** The spec applies the method-1
//!    (`quant_type == 1`) sum-parity toggle of `F[7][7]` to every
//!    block. Observed reference decodes apply it to **non-intra blocks
//!    only** (verified per block class: suppressing the intra toggle
//!    collapses a method-1 stream's differences to isolated IDCT
//!    near-ties).
//!
//! 3. **§7.8.7.3 GMC averaged motion vector.** The spec quantises the
//!    averaged pel-wise warping vector of a GMC macroblock to the
//!    half- (or quarter-) sample grid with the `//` rounding and uses
//!    it as the §7.6.5 predictor candidate and the §7.6.9 frame-direct
//!    co-located vector. Observed reference decodes derive each
//!    **non-positive** component one MV-grid unit lower (strictly
//!    positive components exact; per-component independent). Measured
//!    on crafted GMC-neighbour + zero-MVD-local probes at half-sample
//!    (du −2→−3, −3→−4, −4→−5, −10→−11, 0→−1) and quarter-sample
//!    (−6→−7, −8→−9, −18→−19, −20→−21); pinned bit-exact by the
//!    `dec_sgmc_*` fixture pairs (`tests/compat_gmc_amv.rs`). Those
//!    probes are one-point (translational) warps at half-sample sprite
//!    accuracy, and the adjustment applies only there: XviD GMC streams
//!    (three warping points at 1/16-sample accuracy, both zooms and
//!    pure pans) decode like the reference with the §7.8.7.3 value,
//!    non-positive components included. This matches a rounding right
//!    shift by the sprite accuracy that biases non-positive values by
//!    one only when the shift is zero.
//!
//! Within tolerance, not a divergence: Annex A bounds the IDCT by
//! accuracy only, so an exact half-integer result may round to either
//! neighbour. The default rounds it away from zero (§4's convention for
//! half-integers); ecosystem-compat rounds it to the even neighbour, as
//! black-box reference decodes do on every DC-only block whose
//! `F[0][0] / 8` is a half-integer (202.5 → 202, 203.5 → 204). These
//! halves arise at most quantisers above 4 (`dc_scaler` not 8 or 16),
//! so on plain ecosystem-encoded streams this choice alone separates
//! 4–25 % of an I-VOP's samples by one and then drifts through the
//! prediction chain.
//!
//! Not a divergence: §7.6.9.5.1 gives a co-located S(GMC) macroblock
//! with `mcsel == 1` (coded or skipped) its §7.8.7.3 averaged motion
//! vector as the direct-mode MV, and both modes do exactly that; the
//! reference decodes of GMC streams with B-VOPs agree.
//!
//! Everything else decodes identically in both modes. The switch is
//! wired through every public decode surface: the
//! [`crate::vop_decode`] macroblock walks take a [`DecodeOptions`]
//! argument, [`crate::decoder::Mpeg4VideoDecoder::with_options`]
//! configures the elementary-stream decoder, and the registry /
//! [`crate::decoder::make_decoder`] path reads the `ecosystem-compat`
//! key from [`CodecParameters::options`](oxideav_core::CodecParameters)
//! (see [`crate::decoder::Mpeg4DecoderOptions`]).

/// Options selecting the decoder's behaviour on the spec-vs-ecosystem
/// divergences documented at [`crate::compat`] (module docs).
///
/// Which 8×8 inverse DCT reconstructs texture blocks. Both are conformant (Annex A gives a
/// tolerance, not a bit-exact transform); they differ by a level or two per block, which a long
/// GOP compounds, so matching the encoder's transform is what keeps a decode bit-exact with it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IdctKind {
    /// The exact transform in `f64` ([`crate::idct::idct_8x8`]).
    #[default]
    Exact,
    /// XviD's integer transform ([`crate::xvid_idct::xvid_idct_8x8`]), which the ecosystem's
    /// reference decoder switches to for streams whose user data names XviD; the decoder makes
    /// the same switch when it sees that user data.
    Xvid,
}

/// `Default` is the literal-spec behaviour on every axis.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeOptions {
    /// The inverse DCT for texture blocks; `Exact` unless set, or until the stream's user data
    /// names XviD (see [`IdctKind`]).
    pub idct: IdctKind,
    /// Opt-in ecosystem-compat mode. When `true` the decoder
    /// reproduces the black-box-observed ecosystem behaviour on
    /// exactly three clauses:
    ///
    /// * §7.7.2.2 interlaced-direct derivation runs with the
    ///   co-located field motion vectors read as zero
    ///   (`mvf[i] = mvb[i] = MVD[0]` on the field grid; forward
    ///   reference fields keep the co-located selections);
    /// * the §7.4.4.5 method-1 mismatch toggle is skipped on intra
    ///   blocks (non-intra blocks keep it);
    /// * for a translational warp at half-sample sprite accuracy, each
    ///   non-positive §7.8.7.3 GMC averaged-MV component is derived one
    ///   MV-grid unit lower than the spec quantisation (zero included:
    ///   0 → −1).
    ///
    /// It also resolves exact half-integer IDCT results to the even
    /// neighbour ([`crate::idct::HalfIntegerRounding::ToEven`]), a
    /// choice inside the Annex A accuracy tolerance rather than a
    /// divergence from the text.
    ///
    /// When `false` (the default) all three clauses follow the printed
    /// specification text.
    pub ecosystem_compat: bool,
}

impl DecodeOptions {
    /// Literal-spec behaviour on every axis (same as `Default`).
    pub const fn spec() -> Self {
        Self {
            ecosystem_compat: false,
            idct: IdctKind::Exact,
        }
    }

    /// Ecosystem-compat behaviour on all documented divergences.
    pub const fn ecosystem() -> Self {
        Self {
            ecosystem_compat: true,
            idct: IdctKind::Exact,
        }
    }

    /// How the IDCT resolves exact half-integers under these options.
    pub const fn idct_rounding(self) -> crate::idct::HalfIntegerRounding {
        if self.ecosystem_compat {
            crate::idct::HalfIntegerRounding::ToEven
        } else {
            crate::idct::HalfIntegerRounding::AwayFromZero
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_spec_mode() {
        assert_eq!(DecodeOptions::default(), DecodeOptions::spec());
        assert!(!DecodeOptions::default().ecosystem_compat);
    }

    #[test]
    fn ecosystem_constructor_sets_the_switch() {
        assert!(DecodeOptions::ecosystem().ecosystem_compat);
        assert_ne!(DecodeOptions::ecosystem(), DecodeOptions::spec());
    }
}

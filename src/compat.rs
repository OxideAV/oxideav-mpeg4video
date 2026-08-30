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
//!    `dec_sgmc_*` fixture pairs (`tests/compat_gmc_amv.rs`).
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
/// `Default` is the literal-spec behaviour on every axis.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeOptions {
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
    /// * each non-positive §7.8.7.3 GMC averaged-MV component is
    ///   derived one MV-grid unit lower than the spec quantisation
    ///   (zero included: 0 → −1).
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
        }
    }

    /// Ecosystem-compat behaviour on all documented divergences.
    pub const fn ecosystem() -> Self {
        Self {
            ecosystem_compat: true,
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

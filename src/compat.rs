//! Decode-behaviour selection where the ISO/IEC 14496-2 text and the
//! deployed decoder ecosystem disagree.
//!
//! This crate's default behaviour is always the **literal
//! specification text**. For two clauses, black-box comparison against
//! reference decodes of conformant streams (pixel-level output
//! comparison only — no implementation source was consulted) shows the
//! deployed ecosystem behaves differently, so real-world streams
//! produced/consumed by that ecosystem reconstruct slightly
//! differently from the printed clauses. [`DecodeOptions`] carries the
//! opt-in **ecosystem-compat** switch that reproduces the observed
//! behaviour bit-for-bit; it covers exactly these two divergences:
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
//! 2. **§7.4.4.5 mismatch control.** The spec applies the method-1
//!    (`quant_type == 1`) sum-parity toggle of `F[7][7]` to every
//!    block. Observed reference decodes apply it to **non-intra blocks
//!    only** (verified per block class: suppressing the intra toggle
//!    collapses a method-1 stream's differences to isolated IDCT
//!    near-ties).
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
    /// exactly two clauses:
    ///
    /// * §7.7.2.2 interlaced-direct derivation runs with the
    ///   co-located field motion vectors read as zero
    ///   (`mvf[i] = mvb[i] = MVD[0]` on the field grid; forward
    ///   reference fields keep the co-located selections);
    /// * the §7.4.4.5 method-1 mismatch toggle is skipped on intra
    ///   blocks (non-intra blocks keep it).
    ///
    /// When `false` (the default) both clauses follow the printed
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

    /// Ecosystem-compat behaviour on both documented divergences.
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

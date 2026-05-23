//! # oxideav-mpeg4video
//!
//! Pure-Rust decoder for ISO/IEC 14496-2 (MPEG-4 Visual / Part 2 / ASP)
//! video. The crate was reset to an orphan-rebuild scaffold on
//! 2026-05-18 under the workspace clean-room policy; round 1 of the
//! rebuild lands the §6.2 configuration-header parsers
//! (`VisualObjectSequence` / `VisualObject` / `VideoObjectLayer`).
//!
//! ## Round-1..4 scope
//!
//! * Identification of the three start codes — `0x000001B0`,
//!   `0x000001B5`, `0x000001Bx` — that delimit configuration data.
//! * Structural decoding of the VOL header up to (but not including)
//!   the first VOP: shape (rectangular only), aspect ratio, pixel
//!   dimensions, time-increment resolution, optional
//!   `vol_control_parameters` + VBV block, and the marker bits that
//!   stop start-code emulation inside the header.
//! * Round 3: promotion of the §6.2.3 trailing fields onto
//!   `VolHeader` (`interlaced`, `obmc_disable`, `sprite_enable`,
//!   `not_8_bit` / `quant_precision` / `bits_per_pixel`,
//!   `quant_type`, `quarter_sample`,
//!   `complexity_estimation_disable`, `resync_marker_disable`,
//!   `data_partitioned` / `reversible_vlc`, `newpred_enable`,
//!   `reduced_resolution_vop_enable`, `scalability`), together with
//!   `VopContext::from_vol(&vol)` and `VopHeader::from_vol(&vol,
//!   payload)` convenience entry points.
//! * Round 4: §6.2.3.3 `quant_type == 1` matrix-load body decode —
//!   `load_intra_quant_mat` / `load_nonintra_quant_mat` plus the
//!   `8*[2-64]` zigzag-ordered 8-bit list (with the 0-sentinel
//!   run-length expansion from §6.3.3) surface as
//!   `VolHeader::intra_quant_mat: Option<[u8; 64]>` /
//!   `VolHeader::nonintra_quant_mat: Option<[u8; 64]>`.
//! * Round 5: §6.2.6 macroblock-layer header bit-walk — `not_coded`
//!   (P-VOP), `mcbpc` (Tables B.6 / B.7), `ac_pred_flag` (intra MB),
//!   `cbpy` (Table B.8, 4 non-transparent blocks), `dquant`
//!   (Table 6-32). Surfaces as `MacroblockHeader { not_coded,
//!   mb_type, cbpc, ac_pred_flag, cbpy, dquant_delta }` via
//!   `parse_macroblock_header`. Stuffing macroblocks are
//!   transparently skipped per §6.2.6.
//! * Round 6: §6.2.6 B-VOP macroblock-header prefix — `modb`
//!   (Table B.3), `mb_type` (Table B.4 non-scalable / Table B.5
//!   scalable enhancement layer), `cbpb` (6-bit, 4:2:0 rectangular),
//!   `dbquant` (Table 6-33). Surfaces as `BVopMbHeader { modb,
//!   mb_type, cbpb, mvdf_present, mvdb_present, dbquant_delta }` via
//!   `parse_b_vop_mb_header`. Motion-vector bodies remain out of
//!   scope; the bit reader is left positioned at their start.
//! * Round 7: §6.2.6.2 `motion_vector(mode)` decode + §7.6.3 general
//!   motion-vector decoding. `decode_motion_vector_delta(br, mode,
//!   vop_fcode)` reads a forward/backward/direct MV body (Table B.12
//!   `mv_data` VLCs + `r_size`-bit residuals gated on
//!   `vop_fcode != 1 && mv_data != 0`) and reconstructs `(MVDx, MVDy)`;
//!   `reconstruct_motion_vector(delta, px, py, vop_fcode)` adds a
//!   caller-supplied predictor and applies the Table 7-9 modulo wrap.
//!   The MV predictor itself (median, §7.6.5) is later-round work.
//! * Strict failure on Studio Profiles, FGS layers, and non-rectangular
//!   shapes — those branches are recognised and rejected with a typed
//!   error, never silently mis-parsed. Sprite bodies, complexity-
//!   estimation headers, and `newpred_enable` bodies are likewise
//!   typed-rejected (`VolParseError::UnsupportedBranch`) so the bit
//!   position never drifts past a branch we don't yet decode.
//!
//! Macroblock-level VOP decoding is **not** included; it lands in
//! later rounds.
//!
//! ## Provenance
//!
//! Every numeric value and bit layout in this crate is sourced from
//! ISO/IEC 14496-2:2004 (3rd edition), specifically the syntax
//! tables in §6.2 and the semantic tables (6-3 start codes; 6-14
//! aspect ratios; 6-15 chroma formats; 6-16 shape types). The agent
//! read the spec text from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.
//! No third-party MPEG-4 source was consulted.

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

use oxideav_core::RuntimeContext;

pub mod bitreader;
pub mod bvop;
pub mod macroblock;
pub mod motion;
pub mod vol;
pub mod vop;

pub use bitreader::{BitReader, BitReaderError};
pub use bvop::{
    default_b_mb_type, parse_b_vop_mb_header, parse_dbquant, BMbTypeTable, BVopMbHeader,
    BVopMbParseError, BVopMbType,
};
pub use macroblock::{
    dquant_value, parse_macroblock_header, DerivedMbType, MacroblockHeader, MacroblockParseError,
};
pub use motion::{
    decode_motion_vector_delta, reconstruct_motion_vector, MotionParseError, MotionVector,
    MotionVectorDelta, MvMode,
};
pub use vol::{
    parse_video_object_layer, parse_visual_object_header, parse_visual_object_sequence_header,
    AspectRatio, SpriteEnable, VbvParameters, VolControlParameters, VolHeader, VolParseError,
    VIDEO_OBJECT_LAYER_START_CODE_MAX, VIDEO_OBJECT_LAYER_START_CODE_MIN,
    VIDEO_OBJECT_START_CODE_MAX, VIDEO_OBJECT_START_CODE_MIN, VISUAL_OBJECT_SEQUENCE_END_CODE,
    VISUAL_OBJECT_SEQUENCE_START_CODE, VISUAL_OBJECT_START_CODE,
};
pub use vop::{
    parse_group_of_vop_header, parse_video_object_plane_header, GovHeader, TimeCode, VopCodingType,
    VopContext, VopHeader, VopParseError, GROUP_OF_VOP_START_CODE, VOP_START_CODE,
};

/// Crate-level error surface. Decoding entry points map their internal
/// parse errors into this enum so callers don't need to depend on the
/// private modules' error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Returned by code paths that haven't been wired up yet (VOP
    /// decode, encoder, etc.). The round-1 scope intentionally stops
    /// at the VOL header.
    NotImplemented,
    /// A configuration-header parse failed. See [`VolParseError`] for
    /// the discrimination.
    Vol(VolParseError),
    /// A VOP / Group-of-VOP header parse failed. See [`VopParseError`]
    /// for the discrimination.
    Vop(VopParseError),
    /// A macroblock-layer header parse failed. See
    /// [`MacroblockParseError`] for the discrimination.
    Macroblock(MacroblockParseError),
    /// A B-VOP macroblock-header parse failed. See [`BVopMbParseError`]
    /// for the discrimination.
    BVopMacroblock(BVopMbParseError),
    /// A `motion_vector(mode)` body parse / reconstruction failed. See
    /// [`MotionParseError`] for the discrimination.
    Motion(MotionParseError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => write!(
                f,
                "oxideav-mpeg4video: feature not yet implemented in this round"
            ),
            Error::Vol(err) => write!(f, "oxideav-mpeg4video: VOL parse error: {err}"),
            Error::Vop(err) => write!(f, "oxideav-mpeg4video: VOP parse error: {err}"),
            Error::Macroblock(err) => {
                write!(f, "oxideav-mpeg4video: macroblock parse error: {err}")
            }
            Error::BVopMacroblock(err) => {
                write!(f, "oxideav-mpeg4video: B-VOP macroblock parse error: {err}")
            }
            Error::Motion(err) => {
                write!(f, "oxideav-mpeg4video: motion vector parse error: {err}")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<VolParseError> for Error {
    fn from(err: VolParseError) -> Self {
        Error::Vol(err)
    }
}

impl From<VopParseError> for Error {
    fn from(err: VopParseError) -> Self {
        Error::Vop(err)
    }
}

impl From<MacroblockParseError> for Error {
    fn from(err: MacroblockParseError) -> Self {
        Error::Macroblock(err)
    }
}

impl From<BVopMbParseError> for Error {
    fn from(err: BVopMbParseError) -> Self {
        Error::BVopMacroblock(err)
    }
}

impl From<MotionParseError> for Error {
    fn from(err: MotionParseError) -> Self {
        Error::Motion(err)
    }
}

/// No-op codec registration. Until a `Decoder` impl lands the crate
/// registers no entries against the runtime context; the function
/// exists so the `oxideav_core::register!` glue compiles.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("mpeg4video", register);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_displays() {
        let e = Error::NotImplemented;
        assert!(format!("{e}").contains("not yet implemented"));
    }

    #[test]
    fn vol_error_round_trip() {
        let e: Error = VolParseError::Truncated.into();
        assert!(matches!(e, Error::Vol(VolParseError::Truncated)));
    }

    #[test]
    fn vop_error_round_trip() {
        let e: Error = VopParseError::Truncated.into();
        assert!(matches!(e, Error::Vop(VopParseError::Truncated)));
        assert!(format!("{e}").contains("VOP parse error"));
    }

    #[test]
    fn macroblock_error_round_trip() {
        let e: Error = MacroblockParseError::Truncated.into();
        assert!(matches!(
            e,
            Error::Macroblock(MacroblockParseError::Truncated)
        ));
        assert!(format!("{e}").contains("macroblock parse error"));
    }

    #[test]
    fn b_vop_macroblock_error_round_trip() {
        let e: Error = BVopMbParseError::Truncated.into();
        assert!(matches!(
            e,
            Error::BVopMacroblock(BVopMbParseError::Truncated)
        ));
        assert!(format!("{e}").contains("B-VOP macroblock parse error"));
    }

    #[test]
    fn motion_error_round_trip() {
        let e: Error = MotionParseError::Truncated.into();
        assert!(matches!(e, Error::Motion(MotionParseError::Truncated)));
        assert!(format!("{e}").contains("motion vector parse error"));
    }
}

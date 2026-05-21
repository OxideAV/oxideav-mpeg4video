//! # oxideav-mpeg4video
//!
//! Pure-Rust decoder for ISO/IEC 14496-2 (MPEG-4 Visual / Part 2 / ASP)
//! video. The crate was reset to an orphan-rebuild scaffold on
//! 2026-05-18 under the workspace clean-room policy; round 1 of the
//! rebuild lands the §6.2 configuration-header parsers
//! (`VisualObjectSequence` / `VisualObject` / `VideoObjectLayer`).
//!
//! ## Round-1 scope
//!
//! * Identification of the three start codes — `0x000001B0`,
//!   `0x000001B5`, `0x000001Bx` — that delimit configuration data.
//! * Structural decoding of the VOL header up to (but not including)
//!   the first VOP: shape (rectangular only), aspect ratio, pixel
//!   dimensions, time-increment resolution, optional
//!   `vol_control_parameters` + VBV block, and the marker bits that
//!   stop start-code emulation inside the header.
//! * Strict failure on Studio Profiles, FGS layers, and non-rectangular
//!   shapes — those branches are recognised and rejected with a typed
//!   error, never silently mis-parsed.
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
pub mod vol;

pub use bitreader::{BitReader, BitReaderError};
pub use vol::{
    parse_video_object_layer, parse_visual_object_header, parse_visual_object_sequence_header,
    AspectRatio, VbvParameters, VolControlParameters, VolHeader, VolParseError,
    VIDEO_OBJECT_LAYER_START_CODE_MAX, VIDEO_OBJECT_LAYER_START_CODE_MIN,
    VIDEO_OBJECT_START_CODE_MAX, VIDEO_OBJECT_START_CODE_MIN, VISUAL_OBJECT_SEQUENCE_END_CODE,
    VISUAL_OBJECT_SEQUENCE_START_CODE, VISUAL_OBJECT_START_CODE,
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
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => write!(
                f,
                "oxideav-mpeg4video: feature not yet implemented in this round"
            ),
            Error::Vol(err) => write!(f, "oxideav-mpeg4video: VOL parse error: {err}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<VolParseError> for Error {
    fn from(err: VolParseError) -> Self {
        Error::Vol(err)
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
}

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
//! * Round 8: §7.6.5 median-filter MV predictor.
//!   `predict_motion_vector([Option<MotionVector>; 3])` resolves the
//!   three candidate predictors (`MV1`/`MV2`/`MV3`, `None` = invalid /
//!   transparent neighbour) by the four §7.6.5 validity rules and
//!   computes `Px = Median(MV1x, MV2x, MV3x)` / `Py = Median(MV1y,
//!   MV2y, MV3y)`; the result feeds straight into
//!   `reconstruct_motion_vector`. Gathering the candidates from the
//!   spatial neighbourhood (Figure 7-34 positions) is later-round work.
//! * Round 9: §7.4.1.1 intra-DC texture decode — the first stage of
//!   the §6.2.7 `block(i)` syntax. `decode_intra_dc(br, component)`
//!   reads `dct_dc_size_luminance` (Table B.13) or
//!   `dct_dc_size_chrominance` (Table B.14), the `size`-bit
//!   `dct_dc_differential`, and the trailing `marker_bit` (Table B.15
//!   NOTE 2, when `size > 8`), returning the Table B.15 sign-decoded
//!   *differential* DC value via `IntraDcDifferential { size,
//!   differential }`. The §7.4.3 spatial DC/AC predictor that turns
//!   the differential into the final coefficient is later-round work.
//! * Round 10: §7.4.1.2 AC-coefficient (EVENT) decode — the
//!   `while (!last) DCT coefficient` loop of §6.2.7 `block(i)`, for the
//!   `short_video_header == 0` / `reversible_vlc == 0` path.
//!   `decode_ac_event(br, table_kind)` decodes one `(LAST, RUN, LEVEL)`
//!   EVENT: a Table B.16 (intra) / Table B.17 (inter) Tcoef VLC plus a
//!   sign bit, or one of the three §7.4.1.3 escape modes (Type 1 LMAX,
//!   Type 2 RMAX, Type 3 fixed-length with markers).
//!   `decode_ac_events(br, table_kind)` runs the full loop. Surfaces as
//!   `AcEvent { last, run, level }` / `TcoefTable { Intra, Inter }`. The
//!   §7.4.2 inverse scan and the §7.4.3 spatial predictor are
//!   later-round work.
//! * Round 11: §7.4.2 inverse scan — the conversion of the
//!   one-dimensional `QFS[64]` coefficient stream into the
//!   two-dimensional `PQF[v][u]` 8×8 block under one of the three
//!   Figure 7-4 scan patterns. `events_to_qfs(events, intra_dc)`
//!   expands a §7.4.1.2 AC EVENT sequence (with an optional
//!   §7.4.1.1 intra-DC value at scan position 0) into the dense
//!   `[i32; 64]` array; `inverse_scan(qfs, scan_type)` applies the
//!   `PQF[inv_scan_v[scan_type][n]][inv_scan_u[scan_type][n]] =
//!   QFS[n]` loop. `select_scan_type(is_intra, ac_pred_flag,
//!   dc_direction)` encodes the §7.4.2 selection rule (non-intra or
//!   `ac_pred_flag == 0` → zigzag; intra + AC-pred + DC predictor
//!   from above → alternate-vertical; intra + AC-pred + DC predictor
//!   from left → alternate-horizontal). The §7.4.3 spatial DC/AC
//!   predictor add (that supplies `dc_direction`) and the
//!   §7.4.4 inverse quantisation / inverse DCT remain later-round
//!   work.
//! * Round 12: §7.4.3 spatial DC/AC predictor for intra macroblocks
//!   (`short_video_header == 0`). `default_neighbour_dc(bpp)` returns
//!   the §7.4.3.1 fallback value `2^(bits_per_pixel + 2)` for
//!   neighbours outside the VOP / video packet or in non-intra MBs.
//!   `dc_scaler(component, qs)` evaluates Table 7-1 (the piece-wise
//!   linear scaler with separate Type 1 luminance and Type 2
//!   chrominance formulas across the `1..=4` / `5..=8` / `9..=24` /
//!   `>= 25` quantiser bands; the chrominance row merges the `5..=8`
//!   and `9..=24` columns under `(qs + 13) / 2`).
//!   `select_dc_direction(fa, fb, fc)` applies the §7.4.3.1 rule
//!   `|FA-FB| < |FB-FC|` → from C, else from A.
//!   `predict_intra_dc(pqfx_dc, dir, fa, fc, dc_scaler_x)` evaluates
//!   the §7.4.3.2 reconstruction
//!   `QFX[0][0] = PQFX[0][0] + chosen / dc_scaler`.
//!   `predict_intra_ac_row` / `predict_intra_ac_column` apply the
//!   §7.4.3.3 first row / column add scaled by `QpC/QpX` or
//!   `QpA/QpX`, returning `PQFX` unchanged when the predictor block is
//!   outside the VOP / video packet (all prediction coefficients
//!   taken as zero per §7.4.3.3). `saturate_qf` / `saturate_block`
//!   apply the §7.4.3.4 `[-2048, 2047]` clamp.
//! * Round 13: §7.4.4 inverse quantisation — Figure 7-7's full
//!   `QF[v][u] -> F''[v][u] -> F'[v][u] -> F[v][u]` pipeline for
//!   one 8×8 DCT block. `inverse_quant_intra_dc(qf00, comp, qs,
//!   short_video_header)` evaluates the §7.4.4.1.1 intra DC formula
//!   `F''[0][0] = dc_scaler * QF[0][0]` (Table 7-1 for
//!   `short_video_header == 0`, fixed `dc_scaler = 8` otherwise per
//!   §7.4.4.3). `inverse_quant_method1_coef(qf, w, qs, intra)` /
//!   `inverse_quant_method1(qf, w, ctx)` implement the §7.4.4.1.2
//!   first method — intra `(qf*W*qs*2)/16`, non-intra
//!   `((2*qf + Sign(qf))*W*qs)/16`, fused with §7.4.4.4 saturation
//!   to `[-2^(bpp+3), 2^(bpp+3) - 1]` and §7.4.4.5 mismatch control
//!   (per the §7.4.4.6 summary pseudo-code). `inverse_quant_method2_coef`
//!   / `inverse_quant_method2` implement the §7.4.4.2.1 second method
//!   (`(2*|qf|+1)*qs` for odd `qs`, the same minus one for even `qs`,
//!   sign re-applied via §7.4.4.2.1's trailing sentence) including
//!   the §7.4.4.2 instruction to keep using §7.4.4.1.1 for the intra
//!   DC coefficient. `InverseQuantContext` bundles the per-block
//!   scalars (`macroblock_intra`, `component`, `quantiser_scale`,
//!   `bits_per_pixel`, `short_video_header`).
//! * Round 14: §7.4.5 + Annex A inverse DCT — Annex A.1's orthonormal
//!   8×8 IDCT, evaluated as two passes of the 1-D 8-point IDCT
//!   `f(x) = √(2/N) Σ_u C(u) F(u) cos((2x+1)uπ/(2N))` with `N = 8`,
//!   `C(0) = 1/√2`, `C(k) = 1` otherwise. `idct_8x8(coefficients,
//!   bits_per_pixel)` returns the rounded + §7.4.5-saturated
//!   `[[i32; 8]; 8]` block; `idct_saturation_bounds(bpp)` /
//!   `saturate_idct_sample(value, bpp)` expose the §7.4.5
//!   `[-2^bpp, 2^bpp - 1]` clamp. Round-trips a flat block + a
//!   deterministic random block within ±1 LSB (the IEEE 1180-1990 §3.3
//!   peak-error tolerance referenced by Annex A.1's normative
//!   modifications), cross-validates against the §7.4.4 intra-DC
//!   inverse-quant path, and exercises both saturation polarities and
//!   the high-frequency checkerboard case.
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
pub mod idct;
pub mod inverse_quant;
pub mod macroblock;
pub mod motion;
pub mod predictor;
pub mod scan;
pub mod texture;
pub mod vol;
pub mod vop;

pub use bitreader::{BitReader, BitReaderError};
pub use bvop::{
    default_b_mb_type, parse_b_vop_mb_header, parse_dbquant, BMbTypeTable, BVopMbHeader,
    BVopMbParseError, BVopMbType,
};
pub use idct::{idct_8x8, idct_saturation_bounds, saturate_idct_sample};
pub use inverse_quant::{
    inverse_quant_intra_dc, inverse_quant_method1, inverse_quant_method1_coef,
    inverse_quant_method2, inverse_quant_method2_coef, saturate_fprime, saturation_bounds,
    InverseQuantContext,
};
pub use macroblock::{
    dquant_value, parse_macroblock_header, DerivedMbType, MacroblockHeader, MacroblockParseError,
};
pub use motion::{
    decode_motion_vector_delta, predict_motion_vector, reconstruct_motion_vector, MotionParseError,
    MotionVector, MotionVectorDelta, MvMode,
};
pub use predictor::{
    dc_scaler, default_neighbour_dc, predict_intra_ac_column, predict_intra_ac_row,
    predict_intra_dc, saturate_block, saturate_qf, select_dc_direction, NeighbourBlock,
    NeighbourPosition,
};
pub use scan::{
    events_to_pqf, events_to_qfs, inverse_scan, select_scan_type, DcPredictionDirection,
    InverseScanError, ScanType,
};
pub use texture::{
    decode_ac_event, decode_ac_events, decode_intra_dc, AcEvent, DcComponent, IntraDcDifferential,
    TcoefTable, TextureParseError,
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
    /// An intra-DC texture-coefficient decode failed. See
    /// [`TextureParseError`] for the discrimination.
    Texture(TextureParseError),
    /// A §7.4.2 inverse-scan expansion failed (an AC EVENT stream
    /// walked past coefficient 63). See [`InverseScanError`] for the
    /// discrimination.
    InverseScan(InverseScanError),
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
            Error::Texture(err) => {
                write!(f, "oxideav-mpeg4video: texture parse error: {err}")
            }
            Error::InverseScan(err) => {
                write!(f, "oxideav-mpeg4video: inverse-scan error: {err}")
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

impl From<TextureParseError> for Error {
    fn from(err: TextureParseError) -> Self {
        Error::Texture(err)
    }
}

impl From<InverseScanError> for Error {
    fn from(err: InverseScanError) -> Self {
        Error::InverseScan(err)
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

    #[test]
    fn texture_error_round_trip() {
        let e: Error = TextureParseError::Truncated.into();
        assert!(matches!(e, Error::Texture(TextureParseError::Truncated)));
        assert!(format!("{e}").contains("texture parse error"));
    }

    #[test]
    fn inverse_scan_error_round_trip() {
        let e: Error = InverseScanError::Overflow { position: 64 }.into();
        assert!(matches!(
            e,
            Error::InverseScan(InverseScanError::Overflow { position: 64 })
        ));
        assert!(format!("{e}").contains("inverse-scan error"));
    }
}

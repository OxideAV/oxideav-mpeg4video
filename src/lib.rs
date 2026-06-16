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
//! * Round 25: §7.6.9.5.2 direct-mode forward + backward motion-vector
//!   derivation for B-VOPs.
//!   `direct_mode_motion_vector(co_located, mvd, trb, trd, units)`
//!   linearly scales the co-located anchor-VOP MV by the §7.6.7
//!   temporal-reference ratio and applies the delta vector via the
//!   §7.6.9.5.2 formulas
//!   `MVF = (TRB * MV) / TRD + MVD` /
//!   `MVB = (MVD == 0) ? ((TRB - TRD) * MV) / TRD : MVF - MV`,
//!   with the division `/` taken as §3.4 truncation-toward-zero.
//!   `DirectCoLocatedMv::TransparentOrAbsent` substitutes
//!   `MV = (0, 0)` per the §7.6.9.5.1 final-sentence fallback so
//!   direct mode stays enabled when the reference slot is unavailable.
//!   `DirectMvUnits::QpelMvToHalfPel` covers the §7.6.9.5.2
//!   fourth-paragraph quarter→half-pel reduction (MV halved
//!   component-wise + Table 7-13 rounding) for the
//!   `quarter_sample == 1` / half-pel-`MVD` mismatch case;
//!   `direct_mode_reduce_qpel_to_half_pel` exposes the reduction
//!   directly so callers can pre-apply it once per macroblock.
//!   `DirectMvError::{InvalidTrd, TrbOutOfRange}` enforces the §7.6.7
//!   preconditions `TRD > 0` and `0 <= TRB <= TRD`. The result is
//!   intentionally not Table-7-9-clipped — §7.6.9.5.2's linear scaling
//!   keeps the magnitude bounded relative to the co-located `MV` and
//!   the §7.6.9.5.3 prediction-block generator consumes the algebraic
//!   value directly. 17 new unit tests.
//! * Round 24: §7.6.2.2 quarter-sample mode interpolation (Figures
//!   7-31 / 7-32) + Table 7-13 chroma MV reduction. New
//!   `src/quarter_sample.rs`. `split_quarter_pel(mv)` decomposes a
//!   §7.6.3 quarter-pel MV into `(integer_part, qfrac ∈ 0..=3)` via
//!   arithmetic shift by 2 (negative MVs floor toward `-∞`).
//!   `fir_8tap_clip(samples, rc, bpp)` evaluates the §7.6.2.2.1 8-tap
//!   symmetric FIR `(C[4], C[3], C[2], C[1], C[1], C[2], C[3], C[4])
//!   / 256` with `C = [160, -48, 24, -8]`, the `+ 128 - rc` rounding
//!   offset, and the `[0, 2^bpp - 1]` clip. `half_pel_b/c/d` are the
//!   horizontal-, vertical-, and diagonal-FIR half-pel helpers (the
//!   diagonal cascades a per-row horizontal FIR through a vertical
//!   FIR, each independently clipped). `interpolate_quarter_pixel`
//!   resolves any of the 16 sub-pel positions in Figure 7-32 (`a`,
//!   `e_{-1}`, `b_{-1}`, `f_{-1}`, `g`, `h`, `i`, `j`, `c`, `k`,
//!   `d`, `l`, `m`, `n`, `o`, `p`) via the §7.6.2.2.2 bilinear
//!   blends with `+ 1 - rc` rounding. `interpolate_block_qpel(vop,
//!   mv_x, mv_y, origin_x, origin_y, w, h, vop_rounding_type, bpp)`
//!   and `interpolate_block_qpel_into(...)` fill a luma prediction
//!   block. `reduce_qpel_to_half_pel_chroma(c)` applies Table 7-13
//!   (quarter-pel-units luma → half-pel-units chroma) for the §6.1.3.4
//!   4:2:0 chroma path, biasing any non-zero quarter fraction toward
//!   the +0.5 chroma-pel position per the table. For the §7.6.2
//!   interlaced case, `FieldRefView` adapts the progressive reference
//!   plane to one field's line grid (vertical neighbours are
//!   same-field lines two frame-lines apart),
//!   `field_mvy_to_field_grid(mvy)` halves the always-even
//!   frame-coordinate field MVy into a field-grid quarter-pel
//!   coordinate, and `interpolate_block_qpel_field[_into](...)`
//!   interpolates one 16×8 luma field block through the identical
//!   quarter-pel cascade.
//! * Round 23: §7.6.2.1 half-sample bilinear interpolation (Figure
//!   7-29). `interpolate_pixel(A, B, C, D, half_x, half_y,
//!   rounding_control)` evaluates one of the four per-pixel formulas
//!   `a = A`, `b = (A+B+1-rc)/2`, `c = (A+C+1-rc)/2`,
//!   `d = (A+B+C+D+2-rc)/4` (integer division, `rc ∈ {0, 1}` from
//!   `vop_rounding_type`). `split_half_pel(mv)` decomposes a half-pel
//!   MV into `(integer_part, half_pel_bit)` via arithmetic shift, so
//!   negative MVs round toward `-∞` as the spec's §3.4 division
//!   requires. `ReferenceVop` wraps a `&[u8]` reconstructed reference
//!   plane (with optional row-stride) and exposes
//!   `fetch_clamped(x, y)` for the §7.6.4 last-full-pel edge clamp
//!   (per-component clipping, matching Figure 7-33).
//!   `fetch_clamped_sample(vop, int_x, int_y, half_x, half_y, rc)`
//!   composes the up-to-four neighbour fetches needed for one sub-pel
//!   sample, lazily skipping `B`/`C`/`D` when the corresponding
//!   half-pel bit is zero. `interpolate_block(vop, mv_x, mv_y,
//!   origin_x, origin_y, w, h, vop_rounding_type)` /
//!   `interpolate_block_into(...)` fill an entire prediction block.
//!   §7.6.2.2 quarter-sample mode (the 8-tap FIR + bilinear quarter-
//!   pel) and §7.6.1 reference-VOP padding remain later-round work.
//! * Round 22: §7.6.6 Overlapped Motion Compensation (OBMC) for the
//!   8×8 luminance prediction block. `obmc_predict_block(current_mv,
//!   neighbours, cfg, sample)` composes
//!   `p(i, j) = (q(i,j)*H0(i,j) + r(i,j)*H1(i,j) + s(i,j)*H2(i,j) +
//!   4) // 8` over the 64 pixels of one luminance block, where
//!   `H0` / `H1` / `H2` are the Figure 7-35 / 7-36 / 7-37 weighting
//!   matrices (each cell summing to 8). MV1 (the above-or-below
//!   remote) is taken from `neighbours.above` for `j < 4` and from
//!   `neighbours.below` for `j >= 4`; MV2 (left-or-right) similarly
//!   pivots on `i < 4` / `i >= 4`. `RemoteBlockKind::{NotCoded,
//!   Intra, Absent}` encode the four §7.6.6 remote-MV substitution
//!   rules (not-coded → zero; intra → current MV; VOP/packet border
//!   absent → current MV; current block on MB-bottom row → below
//!   remote replaced by current MV via the `current_block_at_mb_bottom`
//!   flag). `ObmcConfig::disabled()` short-circuits to the bare MV0
//!   prediction `p = q` for the §7.6.6 opening-paragraph
//!   `mcsel`-boundary case of S(GMC)-VOPs and the VOL-header
//!   `obmc_disable == 1` case. The reference-frame sample fetch —
//!   §7.6.2 bilinear half-sample interpolation — remains the caller's
//!   responsibility; `obmc_predict_block` accepts an
//!   `FnMut(ObmcMv, usize, usize) -> u8` sample-provider closure.
//! * Round 19: §7.8.7.3 S(GMC)-VOP averaged-vector substitution —
//!   `averaged_motion_vector(pel_mvs_x, pel_mvs_y, pel_denominator,
//!   quarter_sample, vop_fcode)` returns the candidate motion-vector
//!   predictor that downstream §7.6.5 median calls substitute for an
//!   `mcsel == 1` GMC neighbour. The function sums the 256 luminance
//!   pel-wise motion vectors (the §7.8.7.3 note fixes `Nb = 256`),
//!   divides by 256 with the spec's `//` operator (§3.4 — round to
//!   nearest, ties away from zero), quantises to half-pel (when
//!   `quarter_sample == 0`) or quarter-pel (when `quarter_sample == 1`)
//!   per the §7.8.7.3 bin table, and clips to the Table 7-9
//!   `[low:high]` range for the supplied `vop_fcode`.
//! * Round 17: §7.4.1.3 Type 4 escape — the `short_video_header == 1`
//!   AC-EVENT escape coding. `decode_ac_event_short_video_header(br,
//!   table_kind)` decodes one EVENT under the short-video-header
//!   discipline: the common Tcoef VLC + sign bit (Tables B.16 / B.17) is
//!   unchanged from `decode_ac_event`, but a Type 4 escape — ESC + 1-bit
//!   LAST + 6-bit RUN + 8-bit signed-two's-complement LEVEL, with no
//!   marker bits, per Table B.18 a / c — replaces the §7.4.1.3 Type
//!   1..=3 escapes. The reserved LEVEL values `0` and `-128` are
//!   rejected as [`TextureParseError::ReservedEscapeLevel`].
//!   `decode_ac_events_short_video_header` runs the §6.2.7 `while
//!   (!last) DCT coefficient` loop under the Type 4 path.
//! * Round 16: §7.4.3 / Figure 7-5 predictor candidate gathering —
//!   the cross-block neighbour walk that resolves each block's
//!   `A` (left) / `B` (above-left) / `C` (above) predictor blocks
//!   from a per-VOP grid of already-decoded blocks. `IntraBlockGrid`
//!   stores the luma `2 * mb_rows × 2 * mb_cols` sub-grid plus the
//!   Cb / Cr `mb_rows × mb_cols` sub-grids of `Option<BlockNeighbour>`
//!   (each carrying the inverse-quantised DC, the quantiser scale, the
//!   first row / first column of quantised AC coefficients, and an
//!   `is_intra` flag). `IntraBlockGrid::predictors_for(mb_row, mb_col,
//!   i, bpp, qs)` walks Figure 7-5 against this grid to produce the
//!   `BlockPredictors` argument that `decode_intra_block` already
//!   consumes; neighbours outside the sub-grid or recorded as
//!   `None` / non-intra fall back to the §7.4.3.1 default
//!   `F[0][0] = 2^(bpp + 2)` (with §7.4.3.3 AC prediction coefficients
//!   zeroed via `None`). `block_grid_position(mb_row, mb_col, i)` is
//!   the static Figure 6-8 → sub-grid coordinate mapping; the four
//!   in-MB luma blocks land at the 2 × 2 cells `(2*mb_row + top_bit,
//!   2*mb_col + left_bit)`, Cb and Cr each at `(mb_row, mb_col)`.
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
pub mod block;
pub mod bvop;
pub mod bvop_prediction;
pub mod chroma_mv;
pub mod chroma_shape;
pub mod extended_padding;
pub mod field_motion;
pub mod half_sample;
pub mod idct;
pub mod interlaced_information;
pub mod interlaced_padding;
pub mod inverse_quant;
pub mod macroblock;
pub mod motion;
pub mod mv_predictor_grid;
pub mod neighbour;
pub mod obmc;
pub mod predictor;
pub mod quarter_sample;
pub mod reconstruct;
pub mod rvlc_arbitration;
pub mod sample_padding;
pub mod scan;
pub mod texture;
pub mod vector_padding;
pub mod vertical_padding;
pub mod video_packet;
pub mod vol;
pub mod vop;

pub use bitreader::{BitReader, BitReaderError};
pub use block::{
    de_zigzag, decode_inter_block, decode_inter_macroblock, decode_intra_block,
    decode_intra_macroblock, intra_quant_matrix, nonintra_quant_matrix, pattern_code,
    BlockAssemblyError, BlockPredictors, InterMacroblock, IntraMacroblock,
    MacroblockTextureContext, DEFAULT_INTRA_QUANT_MATRIX, DEFAULT_NONINTRA_QUANT_MATRIX,
};
pub use bvop::{
    decode_b_vop_mb_motion_vectors, default_b_mb_type, parse_b_vop_mb_header, parse_dbquant,
    BMbTypeTable, BVopMbHeader, BVopMbParseError, BVopMbType, BVopMotionVectors, BVopMvBody,
};
pub use bvop_prediction::{
    average_bidirectional, average_bidirectional_into, generate_b_vop_luma_prediction,
    generate_b_vop_luma_prediction_into, BVopMvPair, BVopPredictionMode, BVopSampleMode,
    MB_LUMA_PIXELS, MB_LUMA_SIDE, MB_SUB_BLOCKS, SUB_BLOCK_OFFSETS, SUB_BLOCK_SIDE,
};
pub use chroma_mv::{
    chroma_mv_from_luma_blocks, ChromaMvError, TABLE_7_10, TABLE_7_11, TABLE_7_12, TABLE_7_13,
};
pub use chroma_shape::{
    decimate_chroma_shape, decimate_chroma_shape_interlaced_field,
    decimate_chroma_shape_progressive, decimate_chroma_shape_sample, split_luma_shape_into_fields,
    stack_interlaced_chroma_shape, CHROMA_FIELD_LINES,
};
pub use extended_padding::{
    extended_padding_chroma, extended_padding_luma, extended_padding_macroblock, mid_grey_value,
    BoundaryNeighbours, ExteriorNeighbourPosition, ExteriorPaddingOutcome,
};
pub use field_motion::{
    div2_round, field_motion_compensate_one_reference, field_motion_compensate_one_reference_qpel,
    half_pel_chroma_mv_from_qpel, mc, reconstruct_field_motion_vectors, FieldMotionVectors,
};
pub use half_sample::{
    fetch_clamped_sample, interpolate_block, interpolate_block_into, interpolate_pixel,
    split_half_pel, ReferenceVop,
};
pub use idct::{idct_8x8, idct_saturation_bounds, saturate_idct_sample};
pub use interlaced_information::{
    dct_type_present, field_prediction_present, parse_interlaced_information, DctType,
    FieldPrediction, FieldReference, InterlacedInfoContext, InterlacedInfoContextError,
    InterlacedInformation, McSel,
};
pub use interlaced_padding::{
    interlaced_boundary_padding_luma, per_field_vertical_padding_luma, InterlacedBoundaryOutcome,
    PerFieldVerticalPaddingResult, LUMA_FIELD_LINES,
};
pub use inverse_quant::{
    inverse_quant_intra_dc, inverse_quant_method1, inverse_quant_method1_coef,
    inverse_quant_method2, inverse_quant_method2_coef, saturate_fprime, saturation_bounds,
    InverseQuantContext,
};
pub use macroblock::{
    dquant_value, parse_macroblock_header, DerivedMbType, MacroblockHeader, MacroblockParseError,
};
pub use motion::{
    averaged_motion_vector, decode_field_motion_vector_pair, decode_motion_vector_delta,
    decode_p_macroblock_motion_vectors, decode_p_macroblock_motion_vectors_interlaced,
    decode_p_macroblock_motion_vectors_with_shape, direct_mode_motion_vector,
    direct_mode_reduce_qpel_to_half_pel, motion_coding, predict_field_motion_vector,
    predict_motion_vector, reconstruct_motion_vector, BinaryShapeBlockOpacity, BinaryShapeFourMv,
    DirectCoLocatedMv, DirectModeMv, DirectMvError, DirectMvUnits, FieldMvPair, FieldPredCandidate,
    MotionCodingDeltas, MotionParseError, MotionVector, MotionVectorDelta, MvMode,
    PMbMotionVectors, TypeOfMb, AMV_PIXEL_COUNT,
};
pub use mv_predictor_grid::{
    gather_field_mv_predictor_candidates, gather_mv_predictor_candidates, MbMv, MbMvRecord, MvGrid,
    MvGridError,
};
pub use neighbour::{
    block_grid_position, BlockGridPosition, BlockNeighbour, ChromaPlane, IntraBlockGrid,
};
pub use obmc::{
    obmc_predict_block, ObmcConfig, ObmcMv, ObmcNeighbourhood, RemoteBlockKind, OBMC_H0, OBMC_H1,
    OBMC_H2,
};
pub use predictor::{
    dc_scaler, default_neighbour_dc, predict_intra_ac_column, predict_intra_ac_row,
    predict_intra_dc, saturate_block, saturate_qf, select_dc_direction, NeighbourBlock,
    NeighbourPosition,
};
pub use quarter_sample::{
    field_mvy_to_field_grid, fir_8tap_clip, half_pel_b, half_pel_c, half_pel_d,
    interpolate_block_qpel, interpolate_block_qpel_field, interpolate_block_qpel_field_into,
    interpolate_block_qpel_into, interpolate_quarter_pixel, reduce_qpel_to_half_pel_chroma,
    split_quarter_pel, FieldRefView, QPEL_FIR_C,
};
pub use reconstruct::{
    clip_display_sample, reconstruct_inter_block_8x8, reconstruct_inter_block_8x8_into,
    reconstruct_inter_macroblock, reconstruct_inter_macroblock_into, reconstruct_intra_block_8x8,
    reconstruct_intra_macroblock, InterPredictionMacroblock, ReconstructedMacroblock, BLOCK_SIDE,
    MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE,
};
pub use rvlc_arbitration::{RvlcArbitration, RvlcArbitrationInput, RvlcStrategy, RVLC_THRESHOLD};
pub use sample_padding::{
    horizontal_repetitive_padding_chroma, horizontal_repetitive_padding_luma,
    horizontal_repetitive_padding_row, SamplePresence, ShapeRowState,
    CHROMA_SIDE as PADDING_CHROMA_SIDE, LUMA_SIDE as PADDING_LUMA_SIDE,
};
pub use scan::{
    events_to_pqf, events_to_qfs, inverse_scan, select_scan_type, DcPredictionDirection,
    InverseScanError, ScanType,
};
pub use texture::{
    decode_ac_event, decode_ac_event_rvlc, decode_ac_event_short_video_header, decode_ac_events,
    decode_ac_events_rvlc, decode_ac_events_short_video_header, decode_intra_dc, AcEvent,
    DcComponent, IntraDcDifferential, TcoefTable, TextureParseError,
};
pub use vector_padding::{
    pad_macroblock_vectors, BlockTransparency, MacroblockPaddingMode, VectorPaddingError,
    LUMA_BLOCKS_PER_MB,
};
pub use vertical_padding::{
    vertical_repetitive_padding_chroma, vertical_repetitive_padding_column,
    vertical_repetitive_padding_luma, ColumnState,
};
pub use video_packet::{
    consume_next_resync_marker, macroblock_number_bit_width, parse_video_packet_header,
    probe_resync_marker, resync_marker_length, total_macroblocks, VideoPacketContext,
    VideoPacketHeader, VideoPacketParseError,
};
pub use vol::{
    parse_video_object_layer, parse_visual_object_header, parse_visual_object_sequence_header,
    AspectRatio, ColourDescription, SpriteEnable, VbvParameters, VideoSignalType,
    VisualObjectHeader, VolControlParameters, VolHeader, VolParseError,
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
    /// A §6.2.7 `block(i)` / macroblock texture assembly failed. See
    /// [`BlockAssemblyError`] for the discrimination.
    BlockAssembly(BlockAssemblyError),
    /// A §6.2.5 `video_packet_header` parse failed. See
    /// [`VideoPacketParseError`] for the discrimination.
    VideoPacket(VideoPacketParseError),
    /// A §7.6.9.5.2 direct-mode motion-vector derivation precondition
    /// (`TRD > 0`, `0 <= TRB <= TRD`) failed. See [`DirectMvError`]
    /// for the discrimination.
    DirectMv(DirectMvError),
    /// A §7.6.1.6 vector padding precondition (the macroblock must not
    /// be fully transparent under [`MacroblockPaddingMode::PerBlock`])
    /// failed. See [`VectorPaddingError`] for the discrimination.
    VectorPadding(VectorPaddingError),
    /// A §7.6.5 / Figure 7-34 MV-predictor candidate-gathering call
    /// referred to a macroblock or sub-block index outside the grid.
    /// See [`MvGridError`] for the discrimination.
    MvGrid(MvGridError),
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
            Error::BlockAssembly(err) => {
                write!(f, "oxideav-mpeg4video: block assembly error: {err}")
            }
            Error::VideoPacket(err) => {
                write!(f, "oxideav-mpeg4video: video_packet_header error: {err}")
            }
            Error::DirectMv(err) => {
                write!(f, "oxideav-mpeg4video: direct-mode MV error: {err}")
            }
            Error::VectorPadding(err) => {
                write!(f, "oxideav-mpeg4video: vector padding error: {err}")
            }
            Error::MvGrid(err) => {
                write!(f, "oxideav-mpeg4video: MV-predictor grid error: {err}")
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

impl From<BlockAssemblyError> for Error {
    fn from(err: BlockAssemblyError) -> Self {
        Error::BlockAssembly(err)
    }
}

impl From<VideoPacketParseError> for Error {
    fn from(err: VideoPacketParseError) -> Self {
        Error::VideoPacket(err)
    }
}

impl From<DirectMvError> for Error {
    fn from(err: DirectMvError) -> Self {
        Error::DirectMv(err)
    }
}

impl From<VectorPaddingError> for Error {
    fn from(err: VectorPaddingError) -> Self {
        Error::VectorPadding(err)
    }
}

impl From<MvGridError> for Error {
    fn from(err: MvGridError) -> Self {
        Error::MvGrid(err)
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

    #[test]
    fn video_packet_error_round_trip() {
        let e: Error = VideoPacketParseError::Truncated.into();
        assert!(matches!(
            e,
            Error::VideoPacket(VideoPacketParseError::Truncated)
        ));
        assert!(format!("{e}").contains("video_packet_header error"));
    }

    #[test]
    fn direct_mv_error_round_trip() {
        let e: Error = DirectMvError::InvalidTrd(0).into();
        assert!(matches!(e, Error::DirectMv(DirectMvError::InvalidTrd(0))));
        assert!(format!("{e}").contains("direct-mode MV error"));
    }

    #[test]
    fn vector_padding_error_round_trip() {
        let e: Error = VectorPaddingError::AllTransparent.into();
        assert!(matches!(
            e,
            Error::VectorPadding(VectorPaddingError::AllTransparent)
        ));
        assert!(format!("{e}").contains("vector padding error"));
    }

    #[test]
    fn mv_grid_error_round_trip() {
        let e: Error = MvGridError::InvalidBlockIndex(7).into();
        assert!(matches!(
            e,
            Error::MvGrid(MvGridError::InvalidBlockIndex(7))
        ));
        assert!(format!("{e}").contains("MV-predictor grid error"));
    }
}

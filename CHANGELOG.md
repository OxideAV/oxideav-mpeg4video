# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5](https://github.com/OxideAV/oxideav-mpeg4video/releases/tag/v0.1.5) - 2026-05-30

### Other

- round 24: §7.6.2.2 quarter-sample mode interpolation (Figures 7-31, 7-32)
- round 23: §7.6.2.1 half-sample bilinear interpolation (Figure 7-29)
- round 22: §7.6.6 overlapped motion compensation (OBMC)
- round 21: §6.2.7 block(i) driver for inter macroblocks
- round 20: §6.2.2 VisualObject() header — verid/priority + video_signal_type()
- relax averaged_motion_vector pel_denominator precondition
- round 19: §7.8.7.3 S(GMC)-VOP averaged-vector substitution
- round 18: §6.2.5 video_packet_header decode (rectangular shape)
- round 17: §7.4.1.3 Type 4 escape (short_video_header == 1)
- round 16: §7.4.3 / Figure 7-5 predictor candidate gathering
- round 15: §6.2.7 block(i) intra macroblock-level assembly
- Round 14: §7.4.5 + Annex A inverse DCT
- round 13: §7.4.4 inverse quantisation pipeline (Figure 7-7)
- round 12: §7.4.3 spatial DC/AC predictor for intra macroblocks
- round 11: §7.4.2 inverse scan (QFS[n] → PQF[v][u])
- round 10 — §7.4.1.2 AC-coefficient (EVENT) decode
- round 9 — §7.4.1.1 intra-DC texture decode
- §7.6.5 median-filter MV predictor (round 8)
- mpeg4video r7: §6.2.6.2 motion_vector decode + §7.6.3 differential-MV reconstruction
- round 6: §6.2.6 B-VOP macroblock header prefix (modb/mb_type/cbpb/dbquant)
- round 5: §6.2.6 macroblock-layer header bit-walk (mcbpc/cbpy/dquant)
- round 4: decode quant_type==1 quantiser-matrix load body (§6.2.3.3)
- round 3: promote §6.2.3 VOL fields onto VolHeader; VopHeader::from_vol
- round 2: §6.2.4 GOV + §6.2.5 VOP header parse
- round 1: §6.2 VisualObjectSequence / VisualObject / VOL header parse
- orphan rebuild: clean-room scaffold post 2026-05-18 audit

### Added

- Round 25 of the clean-room rebuild: §7.6.9.5.2 direct-mode forward
  and backward motion-vector derivation for B-VOPs.
  `direct_mode_motion_vector(co_located, mvd, trb, trd, units)`
  linearly scales the co-located anchor-VOP MV by the §7.6.7
  temporal-reference ratio and applies the delta vector via the
  §7.6.9.5.2 formulas
  `MVF = (TRB * MV) / TRD + MVD` and
  `MVB = (MVD == 0) ? ((TRB - TRD) * MV) / TRD : MVF - MV`,
  with the division `/` taken as §3.4 truncation-toward-zero
  (matching Rust's `i32::Div` on signed operands).
  `DirectCoLocatedMv::TransparentOrAbsent` substitutes
  `MV = (0, 0)` per the §7.6.9.5.1 final-sentence fallback so direct
  mode stays enabled when the reference slot is transparent or
  out-of-bounds; the delta MVD passes through unchanged into both
  `MVF` and `MVB`. `DirectMvUnits::QpelMvToHalfPel` covers the
  §7.6.9.5.2 fourth-paragraph quarter-to-half-pel reduction (MV
  halved component-wise plus Table 7-13 rounding) for the
  `quarter_sample == 1` / half-pel-`MVD` mismatch case;
  `direct_mode_reduce_qpel_to_half_pel(mv)` exposes the reduction
  directly so callers can pre-apply it once per macroblock when the
  reference is four block-vectors. `DirectMvError::{InvalidTrd,
  TrbOutOfRange}` enforces the §7.6.7 preconditions `TRD > 0` and
  `0 <= TRB <= TRD`. The returned `(MVF, MVB)` pair is intentionally
  not clipped to the Table 7-9 `[low:high]` range — the §7.6.9.5.2
  linear scaling factor `TRB / TRD ∈ [0, 1]` already keeps the
  magnitude bounded relative to the co-located `MV`, and the
  §7.6.9.5.3 prediction-block generator that follows consumes the
  algebraic value directly. The reference-grid lookup (which
  co-located MB sits at the current MB position in the temporally
  next anchor VOP, plus the §7.6.1.6 vector padding step) remains
  the caller's responsibility. 17 new unit tests cover: canonical
  splits (`TRB == TRD`, `TRB == 0`, both zero-deltas), the §3.4
  truncation-toward-zero division for both positive and negative
  dividends, the transparent-with-zero-delta skipped-MB zero pair
  (§7.6.9.6), the per-component branch independence (`dx != 0`
  takes the subtract branch while `dy == 0` takes the scaled
  formula), the QpelMvToHalfPel reduction matching
  `quarter_sample::reduce_qpel_to_half_pel_chroma` componentwise,
  and the end-to-end Direct-mode MVD round-trip with `f_code == 1`
  per §7.6.3's closing paragraph. Crate test count now 491 + 5 doc.
- Round 24 of the clean-room rebuild: §7.6.2.2 quarter-sample mode
  interpolation (Figures 7-31 and 7-32) for the luminance component
  of P-, S(GMC)-, and B-VOPs, plus Table 7-13 chroma motion-vector
  reduction. New `src/quarter_sample.rs`. `split_quarter_pel(mv)`
  decomposes a §7.6.3 quarter-pel motion-vector component into
  `(integer_part, qfrac ∈ 0..=3)` via arithmetic shift by 2;
  negative MVs floor toward `-∞` so the fractional pair is always
  non-negative (e.g. `split_quarter_pel(-1) == (-1, 3)`).
  `fir_8tap_clip(samples, rounding_control, bits_per_pixel)`
  evaluates the §7.6.2.2.1 8-tap symmetric FIR
  `(C[4], C[3], C[2], C[1], C[1], C[2], C[3], C[4]) / 256` with
  `C = [160, -48, 24, -8]`, the `+ 128 - rc` rounding offset, and
  the `[0, 2^bpp - 1]` clip. `half_pel_b` / `half_pel_c` /
  `half_pel_d` are the horizontal / vertical / diagonal half-pel
  building blocks; `half_pel_d` cascades the horizontal FIR through
  a vertical FIR with each intermediate value independently
  clipped (matching the spec's two-step description).
  `interpolate_quarter_pixel(vop, int_x, int_y, qfrac_x, qfrac_y,
  rounding_control, bits_per_pixel)` resolves any of the 16
  Figure 7-32 sub-pel positions (`a`, `e_{-1}`, `b_{-1}`,
  `f_{-1}`, `g`, `h`, `i`, `j`, `c`, `k`, `d`, `l`, `m`, `n`, `o`,
  `p`) via the §7.6.2.2.2 bilinear `(X + Y + 1 - rc) / 2` blends,
  with `k` and `l` themselves applying the 8-tap FIR vertically to
  the `e` / `f` quarter-pel columns.
  `interpolate_block_qpel(vop, mv_x, mv_y, origin_x, origin_y,
  block_w, block_h, vop_rounding_type, bits_per_pixel)` and
  `interpolate_block_qpel_into(...)` fill a luminance prediction
  block from a single quarter-pel motion vector, applying §7.6.4
  edge clamping per fetch via `ReferenceVop::fetch_clamped`.
  `reduce_qpel_to_half_pel_chroma(c)` applies Table 7-13's
  `{0, 1, 1, 1}` fractional mapping to convert a quarter-pel luma
  component into a half-pel chrominance component (the integer
  part doubles; non-zero quarter fractions round toward the +0.5
  chroma-pel position; negative inputs floor via the same
  arithmetic-shift convention as `split_quarter_pel`). §7.6.1
  padding for arbitrarily shaped VOPs and interlaced field-based
  motion compensation remain later-round work; the caller still
  hands `ReferenceVop` a fully reconstructed and padded reference
  plane. 29 new unit tests + 3 doctests cover the Table 7-13
  fractional + integer rows, the §7.6.2.2.1 FIR sum and clipping
  (both polarities of saturation), the half-pel helpers'
  consistency with the quarter-pel resolver, the integer / half /
  full pel sub-pel selection, §7.6.4 edge-clamp behaviour at
  negative origins, sub-pel rounding-control ties, the block-fill
  flat-reference property, and a horizontal-gradient monotonicity
  property. Total crate test count now 472 + 5 doc.

- Round 23 of the clean-room rebuild: §7.6.2.1 half-sample bilinear
  interpolation (Figure 7-29). New `src/half_sample.rs` evaluates the
  four §7.6.2.1 per-pixel formulas
  `a = A`, `b = (A + B + 1 - rc) / 2`, `c = (A + C + 1 - rc) / 2`,
  `d = (A + B + C + D + 2 - rc) / 4` (integer division;
  `rc ∈ {0, 1}` from `vop_rounding_type`).
  `interpolate_pixel(A, B, C, D, half_x, half_y, rounding_control)`
  is the per-pixel kernel; `split_half_pel(mv)` decomposes a §7.6.3
  half-pel motion-vector component into `(integer_part,
  half_pel_bit)` via arithmetic shift (negative MVs round toward
  `-∞`, matching §3.4 division). `ReferenceVop` wraps a raster-order
  `&[u8]` reference plane (with optional row-stride) and exposes
  `fetch_clamped(x, y)` for the §7.6.4 last-full-pel edge clamp
  (per-component clipping, matching Figure 7-33).
  `fetch_clamped_sample(vop, int_x, int_y, half_x, half_y, rc)`
  composes the up-to-four neighbour fetches for one sub-pel sample,
  lazily skipping `B` / `C` / `D` when the corresponding half-pel
  bit is `0`. `interpolate_block(vop, mv_x, mv_y, origin_x,
  origin_y, block_w, block_h, vop_rounding_type)` and
  `interpolate_block_into(...)` fill an entire `block_w × block_h`
  prediction block from one motion vector. The §7.6.6 OBMC
  `FnMut(ObmcMv, i, j) -> u8` sample-provider closure now has a
  concrete reference-plane implementation callers can wire up.
  §7.6.2.2 quarter-sample mode (the 8-tap FIR + bilinear quarter-pel
  step) and §7.6.1 reference-VOP padding remain later-round work.
  25 new unit tests + 2 new doctests; total crate test count now
  443 + 2 doc.

- Round 22 of the clean-room rebuild: §7.6.6 Overlapped Motion
  Compensation (OBMC). `obmc_predict_block(current_mv, neighbours,
  cfg, sample)` composes the §7.6.6 luminance equation
  `p(i, j) = (q(i,j)*H0(i,j) + r(i,j)*H1(i,j) + s(i,j)*H2(i,j) + 4)
  // 8` over one 8×8 prediction block. `OBMC_H0` / `OBMC_H1` /
  `OBMC_H2` transcribe Figures 7-35 / 7-36 / 7-37 of ISO/IEC
  14496-2:2004 as `[[u8; 8]; 8]` constants; a unit test verifies
  `H0 + H1 + H2 = 8` in every cell, so flat reference samples
  (`q == r == s == C`) reproduce `C` exactly. The §7.6.6 per-pixel
  remote-MV selection — MV1 from `above` for `j < 4` and from
  `below` for `j >= 4`; MV2 from `left` for `i < 4` and from `right`
  for `i >= 4` — is implemented on `ObmcNeighbourhood`. The four
  §7.6.6 substitution rules surface as
  `RemoteBlockKind::{Inter, NotCoded, Intra, Absent}` (not-coded →
  zero MV; intra-coded → current MV; absent-at-border → current MV)
  with the rule-4 "below-MB" override carried by
  `ObmcNeighbourhood::current_block_at_mb_bottom`. `ObmcConfig::
  disabled()` short-circuits to the bare MV0-only prediction
  (one sample fetch per pixel, `p = q`) for the
  `obmc_disable == 1` and S(GMC)-VOP mcsel-boundary cases of the
  §7.6.6 opening paragraph. The reference-frame sample fetch
  (§7.6.2 bilinear half-sample interpolation) stays in the caller
  via the `FnMut(ObmcMv, usize, usize) -> u8` sample-provider
  closure. 13 new unit tests; total crate test count now 418.

- Round 21 of the clean-room rebuild: §6.2.7 `block(i)` driver for
  inter macroblocks. `decode_inter_block(br, i, coded, ctx,
  quant_matrix)` runs one inter block's §6.2.7 syntax — `if
  (pattern_code[i]) while (!last) DCT coefficient`, no intra-DC
  prologue, no §7.4.3 spatial predictor — through the §7.4.x chain:
  `decode_ac_events(TcoefTable::Inter)` → `events_to_qfs(events,
  None)` (no DC) → §7.4.2 zigzag `inverse_scan` (§7.4.2 "non-intra
  blocks → zigzag") → §7.4.3.4 `saturate_block` → §7.4.4 inverse
  quant with `macroblock_intra == false` (method 1 with `W[1]` when
  `quant_type == 1`, else method 2) → §7.4.5 + Annex A `idct_8x8`
  saturating to `[-2^bpp, 2^bpp - 1]`. The output is the §7.3 step-2
  residual `f[y][x]`, not clipped to `[0, 2^bpp - 1]`. When
  `coded == false` no bits are consumed and the residual is the
  all-zero block. `decode_inter_macroblock` walks Figure 6-8 over
  six blocks for any `DerivedMbType ∈ {Inter, InterQ, Inter4V}` and
  assembles a 16×16 luma + 8×8 Cb / 8×8 Cr `InterMacroblock` of
  signed residuals. `not_coded` short-circuits to
  `BlockAssemblyError::NotCoded`; non-inter `mb_type` returns
  `BlockAssemblyError::NotInter`. `nonintra_quant_matrix(vol)`
  resolves `W[1]` from the VOL header (loaded `nonintra_quant_mat`
  de-zigzagged when present, else the §6.3.3 default non-intra
  matrix).

- Round 20 of the clean-room rebuild: §6.2.2 / §6.3.2.3 / §6.3.2.4
  `VisualObject()` header tightening. `parse_visual_object_header`
  now returns a typed `VisualObjectHeader { visual_object_verid,
  visual_object_priority, is_visual_object_identifier,
  visual_object_type, video_signal_type }` instead of the bare
  `visual_object_type` byte. `visual_object_verid` defaults to `1`
  (per §6.3.2.3 — "When this field does not exist, the value … is
  `0001`") and `visual_object_priority` defaults to `1` (highest
  legal value; `0` is reserved per §6.3.2.3) when
  `is_visual_object_identifier == 0`. The §6.2.2 `video_signal_type()`
  body now decodes when the leading flag bit is set: 3-bit
  `video_format` (Table 6-7), 1-bit `video_range`, and the optional
  8 + 8 + 8 `colour_primaries` / `transfer_characteristics` /
  `matrix_coefficients` triple (Tables 6-8 / 6-9 / 6-10), surfaced via
  `VideoSignalType` + `ColourDescription`.
  `ColourDescription::default_when_absent()` returns the §6.3.2.4
  BT.709 fallback (`1, 1, 1`) for callers that need the absent-block
  default. The 1-bit `video_signal_type` flag is consumed
  unconditionally so the bit reader stays aligned with downstream
  start-code search.

- Round 19 of the clean-room rebuild: §7.8.7.3 S(GMC)-VOP
  averaged-vector substitution. `averaged_motion_vector(pel_mvs_x,
  pel_mvs_y, pel_denominator, quarter_sample, vop_fcode)` returns the
  candidate motion-vector predictor for `mcsel == 1` macroblocks (which
  have no own block motion vector — pel-wise motion vectors arrive
  from sprite warping per §7.8.5). Sums `Nb = 256` luminance pel-wise
  MVs, divides by 256 with the spec's `//` operator (§3.4 — round to
  nearest integer, ties away from zero), quantises to half-pel or
  quarter-pel units according to `quarter_sample` per the §7.8.7.3 bin
  table, and clips to the Table 7-9 `[low:high]` range for the
  supplied `vop_fcode`. `pel_denominator` is the caller's pel-wise
  fixed-point grid (`1` integer-pel, `2` half-pel, `4` quarter-pel,
  `16` for the sixteenth-pel grid used by §7.8.5 sub-pel warping, …);
  any positive value yields well-defined integer arithmetic via the
  spec's `//` rounding. `AMV_PIXEL_COUNT` exposes the fixed `Nb = 256`.

- Round 18 of the clean-room rebuild: §6.2.5 `video_packet_header`
  decode (rectangular shape). New `src/video_packet.rs`.
  `parse_video_packet_header(br, &VideoPacketContext)` consumes the
  §5.2.5 `next_resync_marker()` stuffing run (`0 1*` to byte
  alignment), the 17..=23-bit `resync_marker` per §6.3.3 length
  formula (17 for I, `16 + fcode` for P / S(GMC),
  `max(16 + max(fcode_fwd, fcode_bwd), 17)` for B), the Table 6-27
  `macroblock_number` (`ceil(log2(total_mbs))` bits, 1..=14), the
  `quant_precision`-bit `quant_scale` (1..=2^precision - 1; 0
  rejected as `ForbiddenQuantScale`), and the `header_extension_code`.
  When the extension bit is set the rectangular extension body —
  `modulo_time_base`, `vop_time_increment`, `vop_coding_type`,
  `intra_dc_vlc_thr`, plus optional `vop_fcode_forward` (not I) and
  `vop_fcode_backward` (B only) — is decoded into the corresponding
  optional fields of `VideoPacketHeader`.
  `macroblock_number_bit_width(total)` evaluates Table 6-27.
  `total_macroblocks(width, height)` exposes the input expression.
  `resync_marker_length(coding_type, fcode_fwd, fcode_bwd)` evaluates
  the §6.3.3 length formula. `consume_next_resync_marker(br)` runs the
  §5.2.5 stuffing; `probe_resync_marker(br, …)` non-destructively
  peeks for the marker at a byte-aligned position. Non-rectangular
  and binary-only shape, sprite-GMC trajectory inside the extension,
  newpred-enable `vop_id` extension, and reduced-resolution VOP
  extension are typed-rejected via
  `VideoPacketParseError::UnsupportedBranch`. 33 new tests cover
  Table 6-27 boundaries (every row at lo + hi), the §6.3.3 length
  formula across I / P / S / B with all relevant fcode ranges, the
  §5.2.5 stuffing positive path + reader-aligned and partial-byte
  cases + first-bit and tail-bit rejections, `probe_resync_marker`
  positive / non-aligned-reader / P-vs-I disambiguation, and the
  full parser path on I / P / B with and without the extension body
  plus every rejection arm (resync-disabled, non-rectangular,
  bad-quant-precision, missing-marker, mb-number out of range,
  zero quant_scale, zero fcode in extension, sprite-GMC + newpred +
  reduced-resolution extension rejections).

- Round 17 of the clean-room rebuild: §7.4.1.3 Type 4 escape — the
  `short_video_header == 1` AC-EVENT escape coding.
  `decode_ac_event_short_video_header(br, table_kind)` decodes one
  §7.4.1.2 `DCT coefficient` EVENT under the short-video-header
  discipline. The common Tcoef VLC + sign bit (Table B.16 for intra,
  Table B.17 for inter) is unchanged from `decode_ac_event`, but the
  §7.4.1.3 Type 1..=3 escapes are replaced by Type 4: `ESC`
  (`0000 011`) + 1-bit LAST + 6-bit RUN + 8-bit signed
  two's-complement LEVEL (Table B.18 a / c), no marker bits per
  §7.4.1.3 paragraph 4. The reserved LEVEL values `0000 0000` (= 0)
  and `1000 0000` (= -128) are rejected as
  `TextureParseError::ReservedEscapeLevel`.
  `decode_ac_events_short_video_header(br, table_kind)` runs the
  §6.2.7 `while (!last) DCT coefficient` loop against the Type-4 path.
  14 new tests cover the common-path passthrough (intra positive /
  inter negative), the Type-4 escape positive / negative LEVELs, the
  max-legal-positive (`+127`) and min-legal-negative (`-127`)
  boundaries, the full RUN range (`63`), both reserved LEVEL
  rejections (`0` and `-128`), the inline LAST flag, the loop
  terminator behaviour (LAST=1), the loop's truncation handling
  (empty reader → `Truncated`), the absence of marker bits (LEVEL
  bits that would be a marker in Type 3 are now LEVEL bits), and
  truncation mid-LEVEL byte.
- Round 16 of the clean-room rebuild: §7.4.3 / Figure 7-5 predictor
  candidate gathering. New `src/neighbour.rs` module owns the cross-block
  walk that resolves, for each block-to-decode `X`, the three Figure 7-5
  neighbours `A` (left), `B` (above-left), and `C` (above) from a per-VOP
  grid of already-decoded blocks. `IntraBlockGrid::new(mb_rows, mb_cols)`
  allocates one `(2*mb_rows) × (2*mb_cols)` luma sub-grid plus two
  `mb_rows × mb_cols` Cb / Cr sub-grids of `Option<BlockNeighbour>`;
  every cell starts `None` and is filled by `record(mb_row, mb_col, i,
  Some(BlockNeighbour))` as blocks are decoded.
  `predictors_for(mb_row, mb_col, i, bits_per_pixel, quantiser_scale)`
  walks Figure 7-5 against the recorded grid and returns the
  `BlockPredictors` argument `decode_intra_block` already consumes.
  Neighbours outside the sub-grid, recorded `None`, or recorded with
  `is_intra == false` fall back to the §7.4.3.1 default `F[0][0] =
  2^(bits_per_pixel + 2)` (and the §7.4.3.3 AC prediction coefficients
  are zeroed via `None` `first_row` / `first_column`). `BlockNeighbour`
  carries the inverse-quantised DC, the quantiser scale `Qp`, the first
  row (`QF[0][1..=7]`) and first column (`QF[1..=7][0]`) of quantised AC
  coefficients, and an `is_intra` flag; `BlockNeighbour::from_qf(&qf,
  dc, qp)` builds it from a reconstructed intra block.
  `block_grid_position(mb_row, mb_col, i)` exposes the static Figure 6-8
  mapping from `(mb_row, mb_col, i)` to component sub-grid coordinates
  (luma at `(2*mb_row + top_bit, 2*mb_col + left_bit)`, Cb / Cr each at
  `(mb_row, mb_col)`). 19 tests cover the in-MB neighbour cases (block 1
  picks block 0 as `A`; block 2 picks block 0 as `C`; block 3 sees all
  three of blocks 2 / 0 / 1 as `A` / `B` / `C`), the cross-MB cases
  (block 0 of the next column / row / inner MB picks the appropriate
  block of the left / above / diagonal MB), the chroma sub-grid isolation
  (Cb and Cr do not leak into each other or into the luma sub-grid), the
  non-intra-neighbour fallback (`is_intra == false` triggers the
  §7.4.3.1 default), the explicit `None` record (e.g. across a
  video-packet boundary), and a deterministic 2×2 MB raster-walk
  round-trip. The §7.4.3 spatial DC/AC predictor math (rounds 12 / 15)
  consumes the resulting `BlockPredictors` unchanged.
- Round 15 of the clean-room rebuild: §6.2.7 `block(i)` macroblock-level
  texture assembly for intra I-VOP macroblocks. New `src/block.rs`
  module drives the §7.4.x pipeline built in rounds 9..14 end-to-end on
  a per-macroblock basis. `decode_intra_block(br, i, coded, ctx,
  predictors, quant_matrix)` runs one block's §6.2.7 `block(i)` syntax
  (the always-present differential intra-DC, then the
  `if (pattern_code[i]) while (!last) DCT coefficient` AC loop) through
  the full chain: `decode_intra_dc` + `decode_ac_events` →
  `events_to_qfs` → §7.4.2 `inverse_scan` (scan pattern from
  `select_scan_type`) → §7.4.3 spatial DC/AC predictor (`predict_intra_dc`
  / `predict_intra_ac_row` / `predict_intra_ac_column`, gated by
  `ac_pred_flag`) → §7.4.3.4 `saturate_block` → §7.4.4 inverse
  quantisation (method 1 with the `W[0]` intra matrix when
  `quant_type == 1`, else method 2) → §7.4.5 + Annex A `idct_8x8` →
  §6.3.2 final clip to the display range `[0, 2^bpp - 1]`.
  `decode_intra_macroblock(br, header, ctx, predictors, quant_matrix)`
  walks the §6.1.3.9 / Figure 6-8 4:2:0 block order (0,1 / 2,3
  luminance; 4 Cb; 5 Cr) and assembles the reconstructed 16×16
  luminance + 8×8 Cb / 8×8 Cr `IntraMacroblock`. `pattern_code(cbpy,
  cbpc)` derives the six §6.2.7 per-block coded flags from the
  macroblock header. `BlockPredictors::outside(bpp, qs)` supplies the
  §7.4.3.1 / §7.4.3.3 "neighbour outside the VOP" predictor state
  (default DC `2^(bpp+2)`, zero AC prediction). The §6.3.3 default
  intra / non-intra quantisation matrices (`DEFAULT_INTRA_QUANT_MATRIX`
  / `DEFAULT_NONINTRA_QUANT_MATRIX`) and `de_zigzag` / `intra_quant_matrix`
  resolve the method-1 `W[0]` matrix. Tests cover the `pattern_code`
  bit-to-block mapping, `de_zigzag` placement, a DC-only block
  reconstructing flat (method 2 exact 128, method 1 within ±1 LSB of
  the §7.4.4.5 mismatch-control perturbation), a known +1 DC
  differential reconstructing to a flat 130, a coded Type-3-escape AC
  EVENT breaking block flatness, full six-block macroblock assembly
  (flat luma 128 + flat chroma 128), and `NotCoded` / `NotIntra`
  rejections.
- Round 14 of the clean-room rebuild: §7.4.5 + Annex A inverse
  discrete cosine transform. New `src/idct.rs` module evaluating
  Annex A.1's orthonormal 8×8 IDCT
  `f(x, y) = (2/N) Σ_u Σ_v C(u) C(v) F(u, v) cos((2x+1)uπ/(2N))
  cos((2y+1)vπ/(2N))` with `N = 8`, `C(0) = 1/√2`, `C(k) = 1` for
  `k ≠ 0`. Implemented as a separable two-pass 1-D IDCT
  (`f(x) = √(2/N) Σ_u C(u) F(u) cos((2x+1)uπ/(2N))`) against a
  lazily-initialised `f64` cosine-table `COS[u][x] =
  cos((2x+1)uπ/16)`. The final value is rounded to nearest (§4.1)
  and saturated to `[-2^bits_per_pixel, 2^bits_per_pixel - 1]` per
  the §7.4.5 closing sentence. `idct_8x8(coefficients,
  bits_per_pixel)` is the entry point; `idct_saturation_bounds(bpp)`
  / `saturate_idct_sample(value, bpp)` surface the §7.4.5 clamp.
  Tests cover: DC-only block (`F[0][0] = 256` → `f[y][x] = 32`),
  flat-block round-trip via a forward-DCT helper (recon ≤ ±1 LSB),
  deterministic pseudo-random block round-trip (recon ≤ ±1 LSB per
  IEEE 1180-1990 §3.3 with Annex A.1's normative deviations),
  cross-validation against the §7.4.4 intra-DC inverse-quant path
  (`QF = 4` at `qs = 5` → `F''[0][0] = 40` → `f[y][x] = 5`), zero
  block at every supported `bpp` (4 / 8 / 10 / 12), both saturation
  polarities, the §7.4.5 cosine-table sanity check at `cos(0) = 1`
  / `cos(π/4) = √2/2`, the `F[0][*]` single-row `y`-uniform
  property, and the highest-frequency `F[7][7]` checkerboard sign
  pattern. 17 new unit tests, 297 total.
- Round 13 of the clean-room rebuild: §7.4.4 inverse quantisation —
  Figure 7-7's full `QF[v][u] -> F''[v][u] -> F'[v][u] -> F[v][u]`
  pipeline for one 8×8 DCT block. New `src/inverse_quant.rs` module.
  `inverse_quant_intra_dc(qf00, component, quantiser_scale,
  short_video_header)` evaluates the §7.4.4.1.1 intra DC formula
  `F''[0][0] = dc_scaler * QF[0][0]` — Table 7-1 (via
  `predictor::dc_scaler`) when `short_video_header == 0`, the fixed
  `dc_scaler = 8` of §7.4.4.3 / §7.4.1.1 otherwise.
  `inverse_quant_method1_coef(qf, w, qs, intra)` /
  `inverse_quant_method1(qf, w, ctx)` implement the §7.4.4.1.2 first
  inverse-quantisation method — intra `(QF * W[0] * qs * 2) / 16`,
  non-intra `((2*QF + Sign(QF)) * W[1] * qs) / 16` — and the whole-block
  driver fuses §7.4.4.4 saturation to
  `[-2^(bpp+3), 2^(bpp+3) - 1]` plus §7.4.4.5 mismatch control
  (sum-parity gated LSB toggle on `F[7][7]`, implemented via XOR per
  the §7.4.4.5 NOTE 1) following the §7.4.4.6 summary pseudo-code.
  `inverse_quant_method2_coef(qf, qs)` /
  `inverse_quant_method2(qf, ctx)` implement the §7.4.4.2.1 second
  method (`(2*|QF|+1)*qs` for odd `qs`, the same minus one for even
  `qs`, sign re-applied via the §7.4.4.2.1 trailing sentence) and
  honour §7.4.4.2's instruction to keep using §7.4.4.1.1 for the
  intra-DC coefficient. `saturation_bounds(bpp)` /
  `saturate_fprime(value, bpp)` expose the §7.4.4.4 clamp.
  `InverseQuantContext { macroblock_intra, component, quantiser_scale,
  bits_per_pixel, short_video_header }` bundles the per-block scalars.
  Method-1 / method-2 path arithmetic, both signs, both
  `quantiser_scale` parities, both saturation polarities, the
  short-video-header DC path, and the parity-driven mismatch toggle
  are each covered by dedicated tests (+30 unit tests; 280 round-1..13
  tests pass total).
- Round 12 of the clean-room rebuild: §7.4.3 spatial DC / AC predictor
  for intra macroblocks (the `short_video_header == 0` path). New
  `src/predictor.rs` module. `default_neighbour_dc(bits_per_pixel)`
  returns the §7.4.3.1 fallback `F[0][0] = 2^(bits_per_pixel + 2)` for
  neighbours outside the VOP / video packet or in a non-intra
  macroblock. `dc_scaler(component, quantiser_scale)` evaluates
  Table 7-1's piece-wise linear non-linear DC scaler — with separate
  Type 1 (luminance) and Type 2 (chrominance) formulas across the
  `1..=4` / `5..=8` / `9..=24` / `>= 25` quantiser bands (chrominance
  merges the `5..=8` and `9..=24` columns under `(qs + 13) / 2`).
  `select_dc_direction(fa, fb, fc)` applies the §7.4.3.1 rule
  `|FA-FB| < |FB-FC|` → predict from `C` (above), else from `A`
  (left). `predict_intra_dc(pqfx_dc, dir, fa, fc, dc_scaler_x)`
  evaluates the §7.4.3.2 reconstruction
  `QFX[0][0] = PQFX[0][0] + chosen / dc_scaler`.
  `predict_intra_ac_row` / `predict_intra_ac_column` apply the
  §7.4.3.3 first-row / first-column scaled-by-`QpC/QpX` (or
  `QpA/QpX`) add; if the predictor neighbour is `None` (outside the
  VOP / video packet) the call returns `PQFX` unchanged per §7.4.3.3
  ("all the prediction coefficients of that block are assumed to be
  zero"). `saturate_qf` / `saturate_block` apply the §7.4.3.4
  `[-2048, 2047]` clamp.
- `NeighbourPosition { Left, AboveLeft, Above }` and `NeighbourBlock
  { dc, qp, first_row, first_column }` value types surfacing the
  Figure 7-5 three-neighbour layout to callers that will gather
  neighbours from a block grid in a future round.
- 31 round-12 unit tests covering: `2^(bpp+2)` at `bpp` 4 / 8 / 12 (the
  §6.3.3 valid range) and the panic guard at `bpp = 31`; every
  Table 7-1 boundary value for luminance (`qs` 1..=4 / 5..=8 / 9..=24
  / 25..=31) and chrominance (`qs` 1..=4 / 5..=24 / 25..=31, with the
  truncated `(qs+13)/2`); luminance monotonicity across the full
  5-bit quantiser range; the three direction-selection paths
  (horizontal gradient → from C, vertical gradient → from A, equal
  diffs → from A by the strict-`<` rule, default-neighbour-all-equal
  → from A); `predict_intra_dc` from-above and from-left, with
  truncation-toward-zero on both signs; `predict_intra_ac_column`
  and `predict_intra_ac_row` with equal Qp ratios, doubled Qp,
  truncating Qp ratios, and missing-neighbour pass-through; saturation
  at the `[-2048, 2047]` boundary and beyond (including `i32::MIN/MAX`);
  full DC-predictor integration on a luminance block (qs=5) and a
  chrominance block (qs=7).
- Re-export: `predict_intra_dc`, `predict_intra_ac_row`,
  `predict_intra_ac_column`, `saturate_qf`, `saturate_block`,
  `dc_scaler`, `default_neighbour_dc`, `select_dc_direction`,
  `NeighbourBlock`, `NeighbourPosition` at the crate root. The
  pre-existing `DcPredictionDirection` enum is shared with `scan` via
  a `pub use`.

- Round 11 of the clean-room rebuild: §7.4.2 inverse scan — the
  conversion of the one-dimensional decoded coefficient stream
  `QFS[64]` into the two-dimensional `PQF[v][u]` 8×8 block under one
  of three scan patterns. The three scan tables from Figure 7-4 —
  (a) Alternate-Horizontal, (b) Alternate-Vertical, (c) Zigzag — are
  transcribed verbatim into `src/scan.rs` as `[[u8; 8]; 8]` grids
  (`ALT_HORIZONTAL` / `ALT_VERTICAL` / `ZIGZAG`).
  `inverse_scan(qfs, scan_type)` applies the §7.4.2
  `PQF[inv_scan_v[scan_type][n]][inv_scan_u[scan_type][n]] = QFS[n]`
  loop. `events_to_qfs(events, intra_dc)` expands a §7.4.1.2 AC EVENT
  sequence (with an optional §7.4.1.1 intra-DC value at scan position
  0) into the dense `[i32; 64]` array; defensively returns
  `InverseScanError::Overflow { position }` if a malformed stream
  would write past coefficient 63. `events_to_pqf(events, intra_dc,
  scan_type)` is the one-call combination.
  `select_scan_type(is_intra, ac_pred_flag, dc_direction)` encodes the
  §7.4.2 per-block scan-selection rule: non-intra or `ac_pred_flag ==
  0` → zigzag; intra + AC-pred + DC predictor from above (C) →
  alternate-vertical; intra + AC-pred + DC predictor from left (A) →
  alternate-horizontal.
- `ScanType { AlternateHorizontal, AlternateVertical, Zigzag }`,
  `DcPredictionDirection { FromLeft, FromAbove }`, and
  `InverseScanError { Overflow { position } }` value types; the new
  crate-level `Error::InverseScan` variant + `From` conversion.
- 23 round-11 unit tests: each of the three scan grids is a
  permutation of `0..=63`; zigzag starts in the canonical JPEG order
  and ends with `63` at `(7, 7)`; alt-vertical's first column walks
  `0,1,2,3,10,11,12,13` and alt-horizontal's first row matches it
  (the two scans are each other's transpose, also asserted
  cell-by-cell); `inverse_scan` round-trips a DC-only block, the
  first six zigzag positions, the alt-vertical first column, the
  alt-horizontal first row, and `QFS[63]` → `PQF[7][7]` for all three
  scans; `events_to_qfs` with intra-DC (`run`-skips + LEVEL at the
  right position) and without (AC EVENT walks from position 0);
  overflow rejection both from position 0 and post-intra-DC; the
  one-call `events_to_pqf` end-to-end; the four cases of
  `select_scan_type` (intra without AC-pred = zigzag, non-intra always
  zigzag, intra + AC-pred + above/left); and `Display` for
  `InverseScanError`. Plus 1 round-11 `Error::InverseScan` lib-test
  round-trip.

- Round 10 of the clean-room rebuild: §7.4.1.2 AC-coefficient (EVENT)
  decode — the `while (!last) DCT coefficient` loop of the §6.2.7
  `block(i)` texture syntax, for the `short_video_header == 0` /
  `reversible_vlc == 0` path. New `decode_ac_event(br, table_kind)`
  decodes one `(LAST, RUN, LEVEL)` EVENT: the common case is a
  Table B.16 (intra) / Table B.17 (inter) Tcoef VLC selected by the new
  `TcoefTable` argument plus a trailing sign bit (`0` positive, `1`
  negative). The §7.4.1.3 escape prefix `0000 011` selects one of the
  first three escape modes — Type 1 (`ESC 0` + Tcoef VLC; `LEVEL =
  sign * (abs(LEVEL) + LMAX(LAST, RUN))` via Tables B.19 (intra) /
  B.20 (inter)), Type 2 (`ESC 10` + Tcoef VLC; `RUN = RUN +
  RMAX(LAST, abs(LEVEL)) + 1` via Tables B.21 (intra) / B.22 (inter)),
  and Type 3 (`ESC 11` + fixed-length `LAST(1) RUN(6) marker_bit
  LEVEL(12) marker_bit`; the 12-bit LEVEL is signed two's-complement
  with `0` and `-2048` reserved per Table B.18 b). New
  `decode_ac_events(br, table_kind)` runs the full §6.2.7 loop,
  returning every EVENT up to and including the `LAST == 1` terminator.
- `TcoefTable { Intra, Inter }` selector and `AcEvent { last, run,
  level }` value type, re-exported at the crate root.
- The 102-entry Table B.16 (intra) and Table B.17 (inter) Tcoef EVENT
  VLC tables in `src/tcoef_tables.rs`, `include!`d into `texture.rs`;
  generated verbatim from the spec tables (each `(code_bits, code_len,
  last, run, level)` without the trailing sign bit).
- `TextureParseError::{InvalidTcoef { window }, EscapeMarkerBitMissing,
  ReservedEscapeLevel}` variants with `Display`, threaded through the
  existing `Error::Texture` conversion.
- 23 round-10 unit tests: both Tcoef tables 102 entries, prefix-free,
  no duplicates, and disjoint from the escape prefix; intra/inter
  common-case EVENTs (with the intra-vs-inter `110s` divergence) and
  sign handling; the `LAST == 1` terminator and the `decode_ac_events`
  loop; escape Type 1 (LMAX add, both signs), Type 2 (RMAX run
  expansion, intra & inter), Type 3 (positive/negative/min-legal
  LEVEL, reserved `0` / `-2048` rejection, missing-marker rejection);
  invalid-Tcoef and truncated-stream rejection; full round-trips over
  every Table B.16 and Table B.17 entry; and LMAX/RMAX spot-checks.

- Round 9 of the clean-room rebuild: §7.4.1.1 intra-DC texture-
  coefficient decode — the first stage of the §6.2.7 `block(i)`
  texture syntax. New `decode_intra_dc(br, component)` reads
  `dct_dc_size_luminance` (Table B.13, for `block(i)` index `i < 4`)
  or `dct_dc_size_chrominance` (Table B.14, for `i >= 4`) selected by
  the `DcComponent` argument, the `size`-bit `dct_dc_differential`
  additional code (read only when `size != 0`), and the trailing
  `marker_bit` (consumed and validated only when `size > 8`, per
  Table B.15 NOTE 2's start-code-emulation guard). It applies the
  Table B.15 sign-decode (`half_range = 2^(size-1)`; an additional
  code `c >= half_range` → `+c`, else `(c + 1) - 2*half_range`) and
  returns the signed *differential* DC value as
  `IntraDcDifferential { size, differential }`. The result is the
  block's differential DC; the §7.4.3.1 spatial predictor add
  (`QF[0] = dct_dc_pred + differential`) and the §7.4.1.2 AC
  coefficient decode are later-round work.
- `DcComponent { Luminance, Chrominance }` with
  `DcComponent::from_block_index(i)` (the §6.2.7 `i < 4` luminance /
  `i >= 4` chrominance split for 4:2:0); `IntraDcDifferential { size,
  differential }` value type.
- `TextureParseError { Truncated, InvalidDcSize { window },
  MarkerBitMissing }` + `Display` + `Error` + `From<BitReaderError>`,
  surfaced through the new crate-level `Error::Texture` variant +
  `From` conversion.
- 23 round-9 unit tests: both `dct_dc_size` tables prefix-free and
  covering sizes 0..=12; Table B.15 sign-decode for sizes 1/2/3 and
  the size-8 boundaries (`-255`/`-128`/`128`/`255`); `size == 0` → 0;
  `from_block_index` luma/chroma split; luminance & chrominance
  size-0 (no additional code), size-1 positive/negative, size-2,
  size-3; size-9 luminance positive with marker-bit consumption;
  size-9 chrominance negative (`-511`) with marker; marker-bit-zero
  rejection; invalid `dct_dc_size` prefix rejection; truncated empty
  reader; truncated mid-additional-code; full round-trip of every
  Table B.13 row (all-ones additional → `+(2^size - 1)`) and every
  Table B.14 row (all-zeros additional → `1 - 2^size`); and `Display`
  coverage for the three error variants. Plus one crate-level
  `Error::Texture` round-trip test. Total unit test count rises from
  149 to 173.
- Round 8 of the clean-room rebuild: §7.6.5 median-filter
  motion-vector predictor for progressive P-/S(GMC)-VOPs. New
  `predict_motion_vector([Option<MotionVector>; 3])` takes the three
  spatial candidate predictors (`MV1`/`MV2`/`MV3`, where `None` marks
  an invalid neighbour — a transparent macroblock/block, or a
  neighbour outside the current VOP / video packet / GOB, all "treated
  as transparent" per the §7.6.5 note), applies the four §7.6.5
  validity decision rules (a valid candidate keeps its block vector;
  exactly one invalid → set to zero; exactly two invalid → both set to
  the one remaining valid candidate; all three invalid → zero), and
  computes the predictor `Px = Median(MV1x, MV2x, MV3x)` /
  `Py = Median(MV1y, MV2y, MV3y)`. The §7.6.5 worked example
  (`MV1=(-2,3)`, `MV2=(1,5)`, `MV3=(-1,7)` → `Px=-1`, `Py=5`) fixes
  `Median(a, b, c)` as the middle of three integers — the §4.1
  arithmetic-operator clause does not list `Median`. The resolved
  `(Px, Py)` feeds straight into the round-7
  `reconstruct_motion_vector`. Internal `median3` and
  `resolve_candidates` helpers back the public entry point.
- Gathering the candidate predictors from the spatial neighbourhood
  (Figure 7-34 block positions, the four-MV vs single-MV cases, and
  the S(GMC)-VOP `mcsel == '1'` averaged-vector substitution of
  §7.8.7.3) is deliberately left to a later round: Figure 7-34 is a
  diagram with no textual position list in the spec text, so this
  round resolves and medians candidates the caller has already
  gathered.
- 7 round-8 unit tests: `median3` middle-of-three (spec worked-example
  components, permutation-invariance, duplicates, negative-spanning);
  the full §7.6.5 worked example via `predict_motion_vector`; rule 1
  (all three valid → component-wise median); rule 2 (one invalid → set
  to zero); rule 3 (two invalid → both take the third, asserted for
  each of the three valid slots); rule 4 (all invalid → zero); and an
  end-to-end `predict_motion_vector` → `reconstruct_motion_vector`
  pipeline. Total unit test count rises from 142 to 149.
- Round 7 of the clean-room rebuild: §6.2.6.2 `motion_vector(mode)`
  bitstream decode plus the §7.6.3 general motion-vector decoding
  process. New `decode_motion_vector_delta(br, mode, vop_fcode)` walks
  one `motion_vector("forward" / "backward" / "direct")` body — a pair
  of Table B.12 `mv_data` VLCs (65 codes, "vector differences" −16…+16
  in half-pel steps; the on-wire value is the doubled integer −32…+32
  per the §7.6.3 note), each followed by an `r_size`-bit
  (`r_size = vop_fcode - 1`) `*_mv_residual` when
  `vop_fcode != 1 && mv_data != 0`. It reconstructs the differential
  vector `(MVDx, MVDy)` via the §7.6.3 recurrence
  (`f = 1 << r_size`, `MVD = (Abs(mv_data)-1)*f + residual + 1`, sign
  from `mv_data`) and returns a `MotionVectorDelta { dx, dy }`. The
  `"direct"` branch reads only the two `mv_data` VLCs (no residuals).
- `reconstruct_motion_vector(delta, px, py, vop_fcode)` — adds a
  caller-supplied predictor `(Px, Py)` and applies the §7.6.3 /
  Table 7-9 modulo wrap into `[low:high]` (`low = -32*f`,
  `high = 32*f - 1`, `range = 64*f`), yielding the final
  `MotionVector { x, y }`.
- `MvMode { Direct, Forward, Backward }` selector mirroring the
  §6.2.6.2 `mode` argument; `MotionVectorDelta { dx, dy }` and
  `MotionVector { x, y }` value types.
- `MotionParseError { Truncated, InvalidMvData { window },
  InvalidFcode(u8) }` + `Display` + `Error` + `From<BitReaderError>`,
  surfaced through the crate-level `Error::Motion` variant.
- The motion-vector path is exercised by a full Table B.12 round-trip,
  a prefix-free-table assertion (all 65 codes mutually
  non-prefixing), a Table 7-9 bounds check, residual-gating cases for
  `vop_fcode` 1 vs 2, and modulo-wrap boundary tests. Total unit test
  count rises from 119 to 142.
- Round 6 of the clean-room rebuild: §6.2.6 B-VOP macroblock-header
  prefix bit-walk for rectangular VOL shape with 4:2:0 chroma. New
  `parse_b_vop_mb_header(br, vol, vop_coding_type, table)` consumes
  `modb` (Table B.3 — `1` / `01` / `00`), `mb_type` (Table B.4 for
  non-scalable B-VOPs or Table B.5 for the spatially-scalable
  enhancement layer with `ref_select_code == 00`), the 6-bit `cbpb`
  (block_count == 6 per §6.2.6 NOTE), and the 1-or-2-bit `dbquant`
  (Table 6-33: `0` → 0, `10` → -2, `11` → +2) into a typed
  `BVopMbHeader { modb, mb_type, cbpb, mvdf_present, mvdb_present,
  dbquant_delta }`. Motion-vector bodies are deliberately left
  unread; the bit reader is positioned at the start of the
  `interlaced_information()` / `motion_vector("…")` block per the
  spec syntax.
- `BVopMbType` enum (`Direct`, `Forward`, `Backward`, `Interpolated`)
  with `has_forward_mv`, `has_backward_mv`, `may_have_dbquant`
  predicates. `Direct` is reachable only via Table B.4; Table B.5
  has no direct row per §7.9.2.8.3.
- `BMbTypeTable { B4, B5 }` selector — callers pick the active
  table from the VOL/VOP-level `ref_select_code && scalability`
  predicate from §6.2.6.
- `default_b_mb_type(scalable: bool) -> BVopMbType` — implements the
  §6.3.6 default-type rule: scalable enhancement layer → "forward
  mc + Q", otherwise → "direct". Used when `modb == '1'` and
  therefore `mb_type` is absent from the bitstream.
- `parse_dbquant(br)` standalone helper returning `(consumed_bits,
  delta)` for the 1-or-2-bit Table 6-33 VLC.
- `BVopMbParseError { Truncated, InvalidModb { window },
  InvalidMbType { window }, NotBVop(VopCodingType),
  UnsupportedShape(u8), UnsupportedChromaFormat(u8) }` + `Display`
  + `Error` + `From<BitReaderError>`.
- `Error::BVopMacroblock(BVopMbParseError)` variant + `From`
  conversion on the crate-level error surface.
- 27 round-6 unit tests: `modb` Table B.3 full round-trip;
  Table 6-33 `dbquant` full round-trip; `BVopMbType` predicates;
  default-type per scalability; `modb == 1` default direct vs
  scalable default forward; `modb == 01` mb_type-present cases;
  `modb == 00` cbpb-zero (no dbquant); `modb == 00` cbpb-non-zero
  (dbquant present); Direct row never carries dbquant even when
  cbpb is non-zero (§6.2.6 syntax `mb_type != '1'` gate); full
  round-trip of every Table B.4 row; full round-trip of every
  Table B.5 row; Table B.5 `1` decodes to Forward (not Direct);
  dbquant = 0 (single 0 bit) and dbquant = -2 (`10`) round-trip;
  rejection of non-B VOP coding types; rejection of non-rectangular
  shape; rejection of non-4:2:0 chroma format (when explicit via
  `vol_control`); 4:2:0 acceptance via explicit control block;
  truncated buffer; truncated mid-mb_type; truncated mid-dbquant;
  invalid mb_type window in Table B.5; `Display` coverage for all
  six error variants; bit-reader position correctness past dbquant;
  `BVopMbParseError -> Error` conversion + `Display` contains
  "B-VOP macroblock parse error".

- Round 5 of the clean-room rebuild: §6.2.6 macroblock-layer
  header bit-walk for I-VOPs and P-VOPs with rectangular VOL shape
  and 4 non-transparent blocks. New `parse_macroblock_header(br,
  vop_coding_type, vol)` consumes `not_coded` (P-VOP only — Table B.1
  shows it absent on I-VOP), `mcbpc` (Tables B.6 / B.7), the intra-MB
  `ac_pred_flag`, `cbpy` (Table B.8, 4 non-transparent blocks), and
  `dquant` (Table 6-32) into a typed `MacroblockHeader { not_coded,
  mb_type, cbpc, ac_pred_flag, cbpy, dquant_delta }`. Stuffing
  macroblocks (mcbpc → "Stuffing") are consumed transparently per
  §6.2.6; the function returns the first non-stuffing header.
- `MacroblockHeader::SKIPPED` const for the not_coded=1 P-VOP case
  (§6.3.6 — decoder treats as inter with zero MV and no DCT data).
- `DerivedMbType` enum (`Inter`, `InterQ`, `Inter4V`, `Intra`,
  `IntraQ`) matching the Table B.1 derived_mb_type integers, with
  `as_u8`, `is_intra`, `has_dquant` predicates.
- `dquant_value(code: u8) -> i8` expanding Table 6-32 (`00 → -1`,
  `01 → -2`, `10 → +1`, `11 → +2`).
- `MacroblockParseError { Truncated, InvalidMcbpc { window },
  InvalidCbpy { window }, UnsupportedVopKind(VopCodingType),
  UnsupportedShape(u8) }` + `Display` + `Error` + `From<BitReaderError>`.
- `Error::Macroblock(MacroblockParseError)` variant + `From`
  conversion on the crate-level error surface.
- 23 round-5 unit tests: Table 6-32 dquant round-trip;
  `DerivedMbType` predicates; minimal I-VOP intra MB; I-VOP intra+q
  with dquant; P-VOP not_coded skip; P-VOP inter MB (no dquant);
  P-VOP inter+q with dquant -2; P-VOP intra with ac_pred; P-VOP
  inter4v; I-VOP stuffing skip; P-VOP stuffing skip; B-VOP /
  S-VOP rejection; non-rectangular-shape rejection; invalid mcbpc
  prefix; invalid cbpy prefix; truncated; Display coverage for all
  five error variants; `MacroblockHeader::SKIPPED` defaults;
  full round-trip of every Table B.6 / B.7 non-stuffing entry; full
  round-trip of every Table B.8 entry (intra + inter columns);
  `MacroblockParseError -> Error` conversion + Display contains
  "macroblock parse error".

- Round 4 of the clean-room rebuild: §6.2.3.3 `quant_type == 1`
  quantiser-matrix load body decode. After `quant_type = 1`, the
  parser reads `load_intra_quant_mat` (1 bit) and, if set, the
  `8*[2-64]` zigzag-ordered 8-bit `intra_quant_mat` list terminated
  by an optional 0 sentinel (with the remaining entries set to the
  last non-zero value per §6.3.3); same for `load_nonintra_quant_mat`
  / `nonintra_quant_mat`. The grayscale follow-on
  (`load_*_quant_mat_grayscale`) is gated on
  `video_object_layer_shape == "grayscale"`, which the parser
  rejects upfront.
- `VolHeader::intra_quant_mat: Option<[u8; 64]>` and
  `VolHeader::nonintra_quant_mat: Option<[u8; 64]>`. `Some(_)` only
  when the corresponding `load_*_quant_mat` flag was set; entries
  are stored in zigzag scan order with the 0-sentinel run-length
  expansion already applied.
- `VolParseError::EmptyQuantMatrix(&'static str)` for the malformed
  case where the first transmitted matrix byte is 0 (the `8*[2-64]`
  syntax requires at least two values; a leading 0 implies zero
  transmitted).
- 9 round-4 unit tests: full 64-entry intra matrix with no sentinel;
  two-entry intra with zero-sentinel run-length fill; both intra +
  nonintra loaded simultaneously; intra-only / nonintra-only mixed
  with the un-loaded sibling staying `None`; leading-0 intra and
  leading-0 nonintra rejection; minimal two-entry list; full-64 no
  sentinel verifies the tail is *not* re-filled; direct truncation
  of the `parse_quant_matrix` helper. `vol_error_branch_displays`
  was extended to assert the new error's `Display` text.
- Round 3 of the clean-room rebuild: promotion of the ISO/IEC 14496-2
  §6.2.3 trailing VOL fields onto `VolHeader` — `interlaced`,
  `obmc_disable`, `sprite_enable` (Table 6-19), `not_8_bit`,
  `quant_precision`, `bits_per_pixel`, `quant_type`,
  `quarter_sample` (when verid != 1), `complexity_estimation_disable`,
  `resync_marker_disable`, `data_partitioned`, `reversible_vlc`,
  `newpred_enable` (verid != 1), `reduced_resolution_vop_enable`
  (verid != 1), and `scalability`.
- `SpriteEnable` enum (`NotUsed` / `Static` / `Gmc` / `Reserved`)
  matching the Table 6-19 one-bit (verid == 1) and two-bit (verid !=
  1) on-wire encodings.
- `VolParseError::BadQuantPrecision(u8)` for `not_8_bit` paths that
  advertise a `quant_precision` outside §6.3.3's 3..=9 range.
- `VolParseError::UnsupportedBranch(&'static str)` for the
  recognised-but-out-of-scope §6.2.3 branches (`sprite_enable static`
  / `GMC` / reserved body, `quant_type` load matrix bodies, custom
  complexity-estimation header, `newpred_enable` body).
- `VopContext::from_vol(&vol)` convenience constructor that pulls the
  promoted fields out of `VolHeader` so callers no longer hand-stitch
  context.
- `VopHeader::from_vol(&vol, payload)` one-call entry point that
  composes `parse_video_object_plane_header(payload,
  vol.time_increment_resolution, VopContext::from_vol(&vol))`.
- 16 round-3 unit tests: minimal VOL trailing-block parse;
  `interlaced` / `obmc_disable` / `sprite_enable` carry-back;
  `not_8_bit` + `quant_precision` + `bits_per_pixel` decode;
  out-of-range `quant_precision` rejection; sprite-static rejection;
  verid==2 two-bit `sprite_enable` (NotUsed + GMC + Reserved cases);
  complexity-estimation-header branch rejection;
  `data_partitioned` + `reversible_vlc` carry-back; `scalability`
  carry-back; `quant_type` load-matrix rejection; `quant_type` no-load
  success; `VolParseError::UnsupportedBranch` + `BadQuantPrecision`
  display; `VopContext::from_vol` projection; `VopHeader::from_vol`
  on a minimal I-VOP; `VopHeader::from_vol` propagates the VOL's
  `vop_time_increment_resolution`.
- Round 2 of the clean-room rebuild: structural parsers for the
  ISO/IEC 14496-2 §6.2.4 Group-of-VOP and §6.2.5 Video Object Plane
  headers, stopping cleanly at the macroblock layer.
- Public `parse_group_of_vop_header` + `parse_video_object_plane_header`
  entry points, with typed `GovHeader { time_code, closed_gov,
  broken_link }`, `VopHeader { coding_type, modulo_time_base,
  time_increment, composed_ticks, coded, rounding_type,
  intra_dc_vlc_thr, quant, fcode_fwd, fcode_bwd }`, `VopCodingType`
  (Table 6-24), and `TimeCode` (Table 6-23).
- `VopContext` wrapper carrying the VOL-side bits the §6.2.5 syntax
  table depends on (`quant_precision`, `interlaced`, `sprite_gmc`,
  `sprite_static`, `scalability`, `newpred_enable`,
  `reduced_resolution_vop_enable`, `complexity_estimation_disable`).
- Forbidden `vop_fcode == 0` rejection (§6.3.5).
- Out-of-scope branch rejection (`scalability` / `newpred_enable` /
  `reduced_resolution_vop_enable` / sprite / complexity-estimation
  header) with typed `VopParseError::UnsupportedBranch`.
- 24 round-2 unit tests covering: VOP start-code constant
  (`0x000001B6`); GOV start-code constant (`0x000001B3`);
  `VopCodingType::from_bits` Table-6-24 mapping;
  `vop_time_increment_bits` resolution-1 special case; minimal I-VOP
  parse; `modulo_time_base` accumulation; P-VOP `fcode_forward`; P-VOP
  `fcode_forward == 0` rejection; B-VOP both fcodes; B-VOP
  `fcode_backward == 0` rejection; `vop_coded == 0` early return with
  default fields; missing VOP start code; marker-bit violation;
  scalability / newpred / bad-quant-precision rejection;
  `quant_precision == 9` 9-bit quant; interlaced consumes
  `top_field_first` + `alt_vert_scan`; GOV time-code parse; GOV missing
  start code / missing marker; `VopParseError` display +
  `VopParseError -> VolParseError` conversion + `Error::Vop` round
  trip.
- Round 1 of the clean-room rebuild: structural parsers for the
  ISO/IEC 14496-2 §6.2 configuration headers
  (`VisualObjectSequence`, `VisualObject`, `VideoObjectLayer`).
- Public `parse_visual_object_sequence_header`,
  `parse_visual_object_header`, and `parse_video_object_layer`
  entry points.
- Typed `VolHeader { profile_level, width, height,
  time_increment_resolution, aspect_ratio, vol_control, … }`,
  `AspectRatio`, `VolControlParameters`, `VbvParameters`.
- MSB-first `BitReader` with `read_bits`, `read_bool`, `next_bits`,
  `align_to_byte`, `skip_bits`.
- Start-code constants for `VS` (`0x000001B0`), `VS_END`
  (`0x000001B1`), `VO` (`0x000001B5`), `VO_LAYER` range
  (`0x00000120`..=`0x0000012F`), and `VO` range
  (`0x00000100`..=`0x0000011F`).
- 20 round-1 unit tests covering bit reader, marker-bit failures,
  forbidden aspect ratio, extended-PAR, `vol_control_parameters` +
  VBV decode, fixed-VOP rate, Studio Profile rejection, FGS branch
  awareness, non-rectangular shape rejection, and
  `vop_time_increment_bits` width formula.

### Changed

- Round 20: `parse_visual_object_header` return type changed from
  `Result<u8, VolParseError>` to `Result<VisualObjectHeader,
  VolParseError>`. Callers reading the old `visual_object_type` byte
  should use the new struct's `.visual_object_type` field.

### Notes

- Macroblock decoding (motion vectors, DCT coefficients, MB headers)
  is not in scope this round; the round-2 parser stops cleanly at the
  start of `motion_shape_texture()`.
- Studio Profiles (`profile_and_level_indication` 0xE1..=0xE8) and
  Fine-Granularity-Scalable VOL headers are recognised and rejected
  with typed errors rather than silently mis-parsed.
- Sprite, scalability, newpred, reduced-resolution, and complexity-
  estimation VOP branches are explicitly rejected via
  `VopParseError::UnsupportedBranch` rather than mis-aligning the
  bitstream.
- All numeric values sourced from ISO/IEC 14496-2:2004 (3rd edition)
  text at
  `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.

### Erased

- Prior master history was force-erased on **2026-05-18** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub. Every
  public API returns `Error::NotImplemented`. The crates.io version
  (`0.1.5`) is preserved on the new master to avoid breaking
  downstream version pins; the published versions on crates.io will
  be yanked by the maintainer.

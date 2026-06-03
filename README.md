# oxideav-mpeg4video

A pure-Rust MPEG-4 Part 2 Video codec (ISO/IEC 14496-2) for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Round 30 of the clean-room rebuild (2026-06-03).** The prior
implementation was retired on 2026-05-18 under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md);
the VLC tables admitted to sourcing their numeric entries from an
external library. Master history was fully erased per the Hat-3 cold-
enforcement procedure.

* Round 30 — §7.6.5 / Figure 7-34 spatial motion-vector predictor
  candidate gathering. New `src/mv_predictor_grid.rs`. `MvGrid::new(
  mb_rows, mb_cols)` allocates a per-VOP grid sized `mb_rows × mb_cols`
  macroblocks; each cell carries an `MbMvRecord { content, transparent
  }` where `content ∈ {Absent, OneMv(MV), FourMv([MV; 4])}` covers the
  three §6.2.6.2 / §7.6.5 MB-level modes (1-MV inter, 4-MV inter4v,
  and "outside current VOP / video packet / GOB / wholly-transparent
  MB" per the §7.6.5 boundary-substitution rule). The four-element
  `transparent` mask handles the per-block §7.6.1.6 transparency case
  within a non-transparent macroblock — a set bit yields a `None`
  candidate for the corresponding sub-block. `predictor_candidates(
  mb_row, mb_col, block_index) -> [Option<MotionVector>; 3]` resolves
  the three Figure 7-34 spatial positions for the current 8×8 luminance
  block (Figure 6-8 numbering: `0 = TL`, `1 = TR`, `2 = BL`, `3 = BR`)
  into a triple ready to feed straight into [`predict_motion_vector`].
  The four block-position cases follow the in-repo ASCII transcription
  of Figure 7-34 (`docs/video/mpeg4-visual/figure-7-34-mv-predictor-
  layout.md`): block 0 pulls MV1 from the TR sub-block of the left MB,
  MV2 from the BL sub-block of the above MB, and MV3 from the BL
  sub-block of the above-right MB; block 1 pulls MV1 from block 0 of
  the current MB, MV2 from the BR sub-block of the above MB, and MV3
  from the BL sub-block of the above-right MB; block 2 pulls MV1 from
  the BR sub-block of the left MB and MV2 / MV3 from blocks 0 / 1 of
  the current MB; block 3 pulls MV1 / MV2 / MV3 from blocks 2 / 0 / 1
  of the current MB respectively. The "1-MV current MB uses block 0's
  layout" rule is the caller's responsibility (pass `block_index = 0`).
  On the neighbour side a 1-MV MB returns the same MV for every
  sub-block query inside it, so no special-casing on the gather side is
  needed. Sub-grid coordinates negative or beyond `2 * mb_rows` /
  `2 * mb_cols` collapse to `None`, exercising the VOP-boundary case.
  Crossing a video-packet / GOB boundary is handled by the caller via
  `record_absent` on the boundary-side MBs. The §7.6.5 four
  substitution rules (one invalid → zero, two invalid → the third's
  value, three invalid → all zero) continue to live in
  [`predict_motion_vector`], which now has its candidate input
  produced end-to-end against a Figure 7-34 layout. 22 new unit tests
  + 1 doctest; total crate test count now 587 + 9 doc.
* Round 29 — §7.6.9.5.3 second-paragraph + §7.6.9.4 chrominance
  motion-compensation plane for B-VOPs. Extends `src/bvop_prediction.rs`
  with `generate_b_vop_chroma_prediction(forward_chroma_ref,
  backward_chroma_ref, forward_chroma_mv, backward_chroma_mv,
  chroma_mb_origin_x, chroma_mb_origin_y, vop_rounding_type,
  prediction_mode)` (plus the `_into` buffer-out variant). The function
  fills one 8×8 chroma prediction block (Cb or Cr — the caller passes
  the matching anchor-VOP plane and runs it once per component) by
  applying §7.6.2.1 half-sample bilinear interpolation to the supplied
  chroma MV against the forward and / or backward chroma reference
  plane, then averages pixel-by-pixel via `Pi[i][j] = (Pf[i][j] +
  Pb[i][j] + 1) >> 1` for `Bidirectional` and `Direct` modes (the
  §7.6.9.5.3 last paragraph "*The rest process is the same as the
  chrominance motion compensation of the bi-directional mode described
  in subclause 7.6.9.4*" rule). The forward-only and backward-only
  modes write the chosen single-side interpolation directly. Chroma
  uses half-pel bilinear regardless of the VOL `quarter_sample` flag —
  the round-27 `chroma_mv_from_luma_blocks` already reduced the four
  luma MVs to a single half-pel chroma MV per direction, and §7.6.5
  paragraph above Table 7-13 fixes the §7.6.2.1 bilinear (not §7.6.2.2
  FIR) for chroma. The chrominance macroblock origin is in chroma
  samples per §6.1.3.4: macroblock column `c`, row `r` maps to `(8*c,
  8*r)` in 4:2:0. The output is the §7.3 step-1 `p[y][x]` chroma
  prediction; the §7.3 step-2 residual add and step-3 `[0, 2^bpp - 1]`
  clip happen one layer up. §7.6.1.6 vector padding remains the
  caller's responsibility (the round-28 `pad_macroblock_vectors`
  produces the K vectors that round-27 reduces). `CHROMA_BLOCK_SIDE`
  = 8 and `CHROMA_BLOCK_PIXELS` = 64 surface as public constants. 12
  new unit tests; total crate test count now 565 + 8 doc.
* Round 28 — §7.6.1.6 vector padding technique. New
  `src/vector_padding.rs`.
  `pad_macroblock_vectors(vectors, transparencies, mode)` applies the
  per-macroblock §7.6.1.6 procedure to a `[MotionVector; 4]` of luma
  block MVs (Figure 6-8 / §6.1.3.4 raster order: `0 = TL`, `1 = TR`,
  `2 = BL`, `3 = BR`). `MacroblockPaddingMode::AllZero` covers the
  §7.6.1.6 top-level branch for INTRA-coded macroblocks and P-VOP
  `skipped` macroblocks — all four `vectors[i]` are overwritten with
  `(0, 0)` regardless of `transparencies[i]`.
  `MacroblockPaddingMode::PerBlock` covers the per-block fallback
  branch: each `BlockTransparency::Transparent` block walks the
  precedence-ordered table
  `FALLBACK_CHAIN = [[1,2,3], [0,3,2], [3,0,1], [2,1,0]]` (the verbatim
  transcription of the §7.6.1.6 nested `?:` expressions) until the first
  `Opaque` partner is found, and copies that partner's MV in. The
  partner MVs are read from a pre-padding snapshot so the fallback
  chain reads the §7.6.1.6 *input* vectors, not the in-place-updated
  outputs of a prior iteration. `VectorPaddingError::AllTransparent`
  rejects a fully-transparent macroblock under `PerBlock` (§7.6.1.6
  scopes itself to "the transparent blocks within a *non-transparent*
  macroblock" — there is no fallback source for a fully-transparent
  macroblock and the caller must not invoke this routine). The output
  feeds three downstream consumers: §7.6.5 luma → chroma derivation
  (the `K` luma MVs that flow into `chroma_mv_from_luma_blocks`), the
  §7.6.5 spatial MV predictor candidate gathering (`MV1 / MV2 / MV3`
  pulled from already-decoded neighbours), and §7.6.9.5 B-VOP direct
  mode (the temporally-next anchor VOP's co-located MVs that
  `direct_mode_motion_vector` linearly scales). 21 new unit tests +
  1 doctest; total crate test count now 553 + 8 doc.
* Round 27 — §7.6.5 chrominance motion-vector derivation `MVDCHR`
  from `K ∈ {1, 2, 3, 4}` luminance sub-block motion vectors. New
  `src/chroma_mv.rs`.
  `chroma_mv_from_luma_blocks(&[MotionVector])` sums the K luma MVs
  component-wise, divides by `2 * K` via floor (`i32::div_euclid` so
  the residue is well-defined for negative MVs), and applies the
  §7.6.5 fractional rounding by indexing one of four newly-
  transcribed tables based on `K`: `TABLE_7_13` (K = 1, "fourth
  sample resolution"), `TABLE_7_12` (K = 2, "eighth"), `TABLE_7_11`
  (K = 3, "twelfth"), `TABLE_7_10` (K = 4, "sixteenth"). Each table
  outputs a per-component half-sample offset in `{0, 1, 2}` to add
  to the integer half-sample part; an output of `2` represents a
  carry into the next integer chroma-pel (one full chroma-pel = 2
  half-sample offsets). The residue is pre-scaled by 2 so the
  half-sample sum indexes the table's `1/(4 * K)`-sample-grid index
  domain. `ChromaMvError::InvalidBlockCount` rejects `K = 0` and
  `K > 4`. Half-sample-mode entry-point only this round; quarter-
  sample-mode pre-divide ("in quarter sample mode the vectors are
  divided by 2 before summation") is left to the caller — the spec
  text doesn't pin the rounding for that pre-divide step, so
  callers should componentwise apply
  `quarter_sample::reduce_qpel_to_half_pel_chroma` until the docs
  collaborator confirms the rule. 24 new unit tests + 1 doctest;
  total crate test count now 532 + 7 doc.
* Round 26 — §7.6.9.5.3 B-VOP luminance prediction-block generation.
  New `src/bvop_prediction.rs`.
  `generate_b_vop_luma_prediction(forward_ref, backward_ref, mvs,
  mb_origin_x, mb_origin_y, vop_rounding_type, mode, prediction_mode)`
  builds the 16×16 luminance B-VOP prediction macroblock by running
  per-sub-block §7.6.2.1 (`BVopSampleMode::HalfPel`) or §7.6.2.2
  (`BVopSampleMode::QuarterPel { bits_per_pixel }`) interpolation on
  four 8×8 Figure-6-8-ordered sub-blocks (TL, TR, BL, BR) and
  averaging the forward / backward predictions pixel-by-pixel via
  the §7.6.9.4 / §7.6.9.5.3 rule
  `Pi[i][j] = (Pf[i][j] + Pb[i][j] + 1) >> 1`.
  `BVopPredictionMode::{ForwardOnly, BackwardOnly, Bidirectional,
  Direct}` selects the §7.6.9.2 / §7.6.9.3 / §7.6.9.4 / §7.6.9.5
  per-MB mode: the three non-Direct modes replicate the single
  bitstream MV across all four sub-blocks (§7.6.9.2 / §7.6.9.3 /
  §7.6.9.4 each carry one MV per macroblock), Direct mode consumes
  the four per-sub-block `MVF[i]` / `MVB[i]` pairs produced by
  [`direct_mode_motion_vector`]. `average_bidirectional_into(a, b,
  out)` exposes the §7.6.9.4 averaging primitive directly; arithmetic
  is in `u16` so `(255 + 255 + 1) >> 1` reproduces 255 exactly.
  Per the spec note in §7.6.9.5.3, four 8×8 sub-blocks with the same
  vector in quarter-pel mode do NOT collapse to one 16×16 fetch
  (§7.6.2.2 block-boundary mirroring), and this module preserves
  that property by interpolating one sub-block at a time. The output
  is the §7.3 step-1 `p[y][x]` prediction; the §7.3 step-2 residual
  add and step-3 `[0, 2^bpp - 1]` clip happen one layer up. The
  chroma per-block MV reduction (Tables 7-10..7-12 plus the existing
  Table 7-13) landed in round 27 below; §7.6.1.6 vector padding
  remains a follow-up. 17 new unit tests + 1 doctest; total crate
  test count now 508 + 6 doc.
* Round 25 — §7.6.9.5.2 direct-mode forward + backward motion-vector
  derivation for B-VOPs. `direct_mode_motion_vector(co_located, mvd,
  trb, trd, units)` linearly scales the co-located anchor-VOP MV by
  the §7.6.7 temporal-reference ratio and applies the delta vector via
  `MVF = (TRB * MV) / TRD + MVD` and the §7.6.9.5.2 zero-vs-non-zero
  backward branch `MVB = (MVD == 0) ? ((TRB - TRD) * MV) / TRD :
  MVF - MV`, with the division `/` taken as §3.4
  truncation-toward-zero. `DirectCoLocatedMv::TransparentOrAbsent`
  substitutes `MV = (0, 0)` per the §7.6.9.5.1 final-sentence fallback
  so direct mode stays enabled when the co-located reference slot is
  unavailable. `DirectMvUnits::QpelMvToHalfPel` covers the
  §7.6.9.5.2 fourth-paragraph quarter→half-pel reduction (`MV` halved
  component-wise plus Table 7-13 rounding) for the `quarter_sample ==
  1` / half-pel-`MVD` mismatch case; `direct_mode_reduce_qpel_to_half_pel`
  exposes the reduction directly so callers can pre-apply it once per
  macroblock. `DirectMvError::{InvalidTrd, TrbOutOfRange}` enforces the
  §7.6.7 preconditions `TRD > 0` and `0 <= TRB <= TRD`. The result is
  intentionally not Table-7-9-clipped — §7.6.9.5.2's linear scaling
  keeps the magnitude bounded relative to the co-located `MV` and the
  §7.6.9.5.3 prediction-block generator consumes the algebraic value
  directly. Tests cover canonical splits (TRB == TRD, TRB == 0, both
  zero deltas, mixed-axis delta branches), the §3.4 truncation-toward-
  zero division for both positive and negative dividends, the
  transparent-with-zero-delta skipped-MB zero pair (§7.6.9.6), and the
  end-to-end Direct-mode MVD round-trip with `f_code == 1`. 17 new
  unit tests + the lib-level Error round-trip; total crate test count
  now 491 + 5 doc.
* Round 24 — §7.6.2.2 quarter-sample mode interpolation (Figures 7-31
  and 7-32) + Table 7-13 chroma motion-vector reduction. New
  `src/quarter_sample.rs`. `split_quarter_pel(mv)` decomposes a §7.6.3
  quarter-pel motion-vector component into `(integer_part, qfrac ∈
  0..=3)` via arithmetic shift by 2 — negative MVs floor toward `-∞`
  so the fractional pair is always non-negative.
  `fir_8tap_clip(samples, rc, bpp)` evaluates the §7.6.2.2.1 8-tap
  symmetric FIR `(C[4], C[3], C[2], C[1], C[1], C[2], C[3], C[4]) /
  256` with `C = [160, -48, 24, -8]`, the `+ 128 - rc` rounding offset,
  and the `[0, 2^bpp - 1]` clip. `half_pel_b` / `half_pel_c` /
  `half_pel_d` cover the horizontal / vertical / diagonal half-pel
  helpers; `half_pel_d` cascades the horizontal FIR through a vertical
  FIR with each intermediate value independently clipped (matching the
  spec's two-step description).
  `interpolate_quarter_pixel(vop, int_x, int_y, qfrac_x, qfrac_y, rc,
  bpp)` resolves any of the 16 Figure 7-32 sub-pel positions (`a`,
  `e_{-1}`, `b_{-1}`, `f_{-1}`, `g`, `h`, `i`, `j`, `c`, `k`, `d`, `l`,
  `m`, `n`, `o`, `p`) via the §7.6.2.2.2 bilinear `+ 1 - rc` blends.
  `interpolate_block_qpel(vop, mv_x, mv_y, origin_x, origin_y, w, h,
  vop_rounding_type, bpp)` and `interpolate_block_qpel_into(...)` fill
  a luminance prediction block. `reduce_qpel_to_half_pel_chroma(c)`
  applies Table 7-13's `{0, 1, 1, 1}` fractional mapping to convert a
  quarter-pel luma component into a half-pel chrominance component
  (the integer part doubles; non-zero quarter fractions round toward
  the +0.5 chroma-pel position; negative inputs floor via
  `split_quarter_pel`). §7.6.1 reference-VOP padding and field-based
  interlaced motion compensation remain later-round work. 29 new unit
  tests + 3 doctests; total crate test count now 472 + 5 doc.
* Round 23 — §7.6.2.1 half-sample bilinear interpolation (Figure 7-29).
  New `src/half_sample.rs` with the four per-pixel formulas
  `a = A`, `b = (A+B+1-rc)/2`, `c = (A+C+1-rc)/2`,
  `d = (A+B+C+D+2-rc)/4` (integer division; `rc ∈ {0, 1}` from the
  VOP-header `vop_rounding_type`).
  `interpolate_pixel(A, B, C, D, half_x, half_y, rounding_control)`
  selects the right formula for one pixel.
  `split_half_pel(mv)` decomposes a §7.6.3 half-pel motion-vector
  component into `(integer_part, half_pel_bit)` via arithmetic shift
  — negative MVs round toward `-∞`, matching the spec's §3.4
  division. `ReferenceVop::{new, with_stride, fetch_clamped}` wraps a
  raster-order `&[u8]` plane and applies the §7.6.4 last-full-pel
  edge clamp per component (Figure 7-33).
  `fetch_clamped_sample(...)` skips unused neighbour fetches for
  cheaper sub-pel cases (just `A` for integer-pel, two samples for
  the horizontal / vertical half-pel cases, all four for the
  diagonal). `interpolate_block(vop, mv_x, mv_y, origin_x, origin_y,
  block_w, block_h, vop_rounding_type)` and
  `interpolate_block_into(...)` fill an entire prediction block from
  a single motion vector. The `Round 22` OBMC `FnMut(ObmcMv, i, j) -> u8`
  sample-provider closure now has a concrete implementation that the
  caller can wire to a reference frame. §7.6.2.2 quarter-sample mode
  (the 8-tap FIR + bilinear quarter-pel) and §7.6.1 reference-VOP
  padding remain later-round work. 25 new unit tests + 2 doctests;
  total crate test count now 443 + 2 doc.
* Round 22 — §7.6.6 Overlapped Motion Compensation (OBMC) for the
  8×8 luminance prediction block. `obmc_predict_block(current_mv,
  neighbours, cfg, sample)` composes the §7.6.6 luminance equation
  `p(i, j) = (q(i,j)*H0(i,j) + r(i,j)*H1(i,j) + s(i,j)*H2(i,j) + 4)
  // 8` over one 8×8 block. `OBMC_H0` / `OBMC_H1` / `OBMC_H2`
  transcribe Figures 7-35 / 7-36 / 7-37 as `[[u8; 8]; 8]`
  constants; a unit test verifies `H0 + H1 + H2 = 8` in every cell,
  so flat reference samples reproduce the input. The per-pixel
  remote-MV selection is implemented on `ObmcNeighbourhood`: MV1
  takes `above` for `j < 4` and `below` for `j >= 4`; MV2 takes
  `left` for `i < 4` and `right` for `i >= 4`. The four §7.6.6
  substitution rules surface as `RemoteBlockKind::{Inter,
  NotCoded, Intra, Absent}` (not-coded → zero; intra → current MV;
  absent-at-border → current MV) with rule-4 below-MB carried by
  `current_block_at_mb_bottom`. `ObmcConfig::disabled()` returns
  the bare MV0 prediction for `obmc_disable == 1` and the §7.6.6
  opening-paragraph mcsel-boundary case of S(GMC)-VOPs. The
  §7.6.2 reference-frame sample fetch stays in the caller via an
  `FnMut(ObmcMv, usize, usize) -> u8` sample-provider closure.
* Round 21 — §6.2.7 `block(i)` driver for inter macroblocks (the
  §7.4.x DCT-coefficient pipeline through the non-intra branch, no
  MV-predictor dependency). `decode_inter_block(br, i, coded, ctx,
  quant_matrix)` runs one inter block's §6.2.7 syntax — `if
  (pattern_code[i]) while (!last) DCT coefficient`, no intra-DC
  prologue, no §7.4.3 spatial predictor — through the full §7.4.x
  chain: read coefficients (`decode_ac_events(TcoefTable::Inter)`) →
  `events_to_qfs(events, None)` → §7.4.2 zigzag `inverse_scan` →
  §7.4.3.4 `saturate_block` → §7.4.4 inverse quant with
  `macroblock_intra == false` (method 1 with `W[1]` when
  `quant_type == 1`, else method 2 with the `(2*|QF|+1)*qs` formula
  and the §7.4.4.2 sign-incorporation) → §7.4.5 + Annex A `idct_8x8`
  saturating to `[-2^bpp, 2^bpp - 1]`. The output is the §7.3 step-2
  residual `f[y][x]` — NOT clipped to `[0, 2^bpp - 1]` because the
  §7.3 step-3 display clip happens after `d[y][x] = p[y][x] +
  f[y][x]` in the caller's motion-compensation stage. When
  `coded == false` no bits are consumed and the residual is the
  all-zero block (the §6.2.7 `if (pattern_code[i])` guard).
  `decode_inter_macroblock(br, header, ctx, quant_matrix)` accepts
  `DerivedMbType ∈ {Inter, InterQ, Inter4V}`, walks Figure 6-8
  (luma 0..=3 in 2×2; Cb 4; Cr 5), and assembles a 16×16 luma + 8×8
  Cb / 8×8 Cr `InterMacroblock` of signed residuals. `not_coded`
  short-circuits via `BlockAssemblyError::NotCoded` (the §7.5
  skipped-MB zero-MV / zero-residual reconstruction is the caller's
  motion-compensation responsibility); a non-inter `mb_type` returns
  `BlockAssemblyError::NotInter`. `nonintra_quant_matrix(vol)`
  resolves `W[1]` from the VOL header — the loaded
  `nonintra_quant_mat` (de-zigzagged) when present, else the §6.3.3
  default non-intra matrix.
* Round 20 — §6.2.2 / §6.3.2.3 / §6.3.2.4 `VisualObject()` header
  tightening. `parse_visual_object_header` now returns a typed
  `VisualObjectHeader { visual_object_verid, visual_object_priority,
  is_visual_object_identifier, visual_object_type, video_signal_type }`
  in place of the bare `visual_object_type` byte. The
  `is_visual_object_identifier == 0` branch now surfaces the §6.3.2.3
  defaults (`verid = 1` per "When this field does not exist, the value
  … is `0001`"; `priority = 1` per "value of zero is reserved"). The
  §6.2.2 `video_signal_type()` body now decodes when the leading flag
  bit is set: 3-bit `video_format` (Table 6-7), 1-bit `video_range`,
  and the optional 8 + 8 + 8 `colour_primaries` /
  `transfer_characteristics` / `matrix_coefficients` triple (Tables
  6-8 / 6-9 / 6-10), surfaced via `VideoSignalType` +
  `ColourDescription`. `ColourDescription::default_when_absent()`
  exposes the §6.3.2.4 BT.709 fallback (`1, 1, 1`) for callers that
  need the absent-block default. The 1-bit `video_signal_type` flag
  is consumed unconditionally so the bit reader stays aligned with the
  downstream `next_start_code()` search.
* Round 19 — §7.8.7.3 S(GMC)-VOP averaged-vector substitution.
  `averaged_motion_vector(pel_mvs_x, pel_mvs_y, pel_denominator,
  quarter_sample, vop_fcode)` computes the candidate motion-vector
  predictor for `mcsel == 1` macroblocks (those that have no own block
  motion vector — pel-wise motion vectors come from sprite warping per
  §7.8.5). The function sums `Nb = 256` luminance pel-wise MVs (the
  §7.8.7.3 note fixes `Nb` at 256), divides by 256 using the spec's
  `//` operator (§3.4 — rounding to the nearest integer, half away
  from zero), quantises to half-pel (when `quarter_sample == 0`) or
  quarter-pel (when `quarter_sample == 1`) per the §7.8.7.3 bin table,
  and clips to the Table 7-9 `[low:high]` range for the supplied
  `vop_fcode`. `pel_denominator` is the caller's pel-wise fixed-point
  grid (`1` for integer-pel input, `2` for half-pel, `4` for
  quarter-pel, `16` for the sixteenth-pel grid used by §7.8.5 sub-pel
  warping, …); any positive value yields well-defined integer
  arithmetic via the spec's `//` rounding. Surfaces as
  `AMV_PIXEL_COUNT` + `averaged_motion_vector` alongside the existing
  §7.6.5 `predict_motion_vector`.
* Round 18 — §6.2.5 `video_packet_header` decode (rectangular shape).
  New `src/video_packet.rs`. `parse_video_packet_header(br, &ctx)`
  consumes the §5.2.5 `next_resync_marker()` stuffing run, reads the
  17..=23-bit `resync_marker`, decodes the Table 6-27
  `macroblock_number` (`ceil(log2(total_mbs))` bits, 1..=14),
  the `quant_precision`-bit `quant_scale` (1..=2^precision - 1; 0
  rejected), and the `header_extension_code`. When the extension bit
  is set, the rectangular extension body — `modulo_time_base`,
  `vop_time_increment`, `vop_coding_type`, `intra_dc_vlc_thr`, and
  the per-coding-type `vop_fcode_forward` / `vop_fcode_backward` —
  is decoded into the optional `VideoPacketHeader` fields.
  `macroblock_number_bit_width(total_mbs)` evaluates Table 6-27.
  `resync_marker_length(coding_type, fcode_fwd, fcode_bwd)` evaluates
  the §6.3.3 length rule (17 for I, `16 + fcode` for P / S(GMC),
  `max(16 + max(fcode_fwd, fcode_bwd), 17)` for B).
  `consume_next_resync_marker(br)` runs the §5.2.5 stuffing loop;
  `probe_resync_marker(br, …)` non-destructively peeks for the
  marker at a byte-aligned position. Non-rectangular and binary-only
  shapes, sprite-GMC trajectory, newpred, and reduced-resolution VOP
  branches are typed-rejected via
  `VideoPacketParseError::UnsupportedBranch`.
* Round 17 — §7.4.1.3 Type 4 escape (`short_video_header == 1`).
  `decode_ac_event_short_video_header(br, table_kind)` decodes one
  §7.4.1.2 AC EVENT under the short-video-header path: the common
  Table B.16 / B.17 Tcoef VLC + sign bit is unchanged from
  `decode_ac_event`, but the §7.4.1.3 Type 1..=3 escapes are replaced
  by Type 4 — `ESC` (`0000 011`) + 1-bit LAST + 6-bit RUN + 8-bit
  signed two's-complement LEVEL (Table B.18 a / c), no marker bits.
  The reserved LEVEL values `0000 0000` (0) and `1000 0000` (-128)
  are rejected as `TextureParseError::ReservedEscapeLevel`.
  `decode_ac_events_short_video_header(br, table_kind)` runs the
  §6.2.7 `while (!last) DCT coefficient` loop against the Type-4 path.
* Round 16 — §7.4.3 / Figure 7-5 predictor candidate gathering. New
  `src/neighbour.rs`. `IntraBlockGrid::new(mb_rows, mb_cols)` allocates
  the Figure 6-8 block sub-grids (luma `2*mb_rows × 2*mb_cols` +
  Cb / Cr `mb_rows × mb_cols`) of `Option<BlockNeighbour>` for one
  VOP. `predictors_for(mb_row, mb_col, i, bits_per_pixel,
  quantiser_scale)` walks Figure 7-5 against the recorded grid and
  returns the `BlockPredictors` argument round-15's
  `decode_intra_block` already consumes. Neighbours outside the
  sub-grid, recorded `None`, or marked `is_intra == false` fall back to
  the §7.4.3.1 default (DC `2^(bits_per_pixel + 2)`, AC prediction
  coefficients zeroed via `None`). `BlockNeighbour::from_qf(&qf, dc,
  qp)` builds the stored state from a reconstructed intra block;
  `block_grid_position` exposes the static Figure 6-8 mapping.
* Round 1 — structural parsing of the §6.2 configuration headers
  (`VisualObjectSequence` / `VisualObject` / `VideoObjectLayer`).
* Round 2 — structural parsing of §6.2.4 Group-of-VOP and §6.2.5 Video
  Object Plane headers up to (but not including) the macroblock layer.
* Round 3 — promotion of the §6.2.3 trailing fields onto
  `VolHeader` (`interlaced`, `obmc_disable`, `sprite_enable`,
  `quant_precision`, `quant_type`, `complexity_estimation_disable`,
  `resync_marker_disable`, `data_partitioned`, `newpred_enable`,
  `reduced_resolution_vop_enable`, `scalability`) plus a
  `VopHeader::from_vol(vol, payload)` / `VopContext::from_vol(&vol)`
  convenience pair.
* Round 4 — §6.2.3.3 `quant_type == 1` matrix-load body decode:
  `load_intra_quant_mat` / `load_nonintra_quant_mat` followed by the
  `8*[2-64]` zigzag-ordered 8-bit list with the 0-sentinel run-length
  expansion ("remaining values set equal to the last non-zero value"
  per §6.3.3). New `VolHeader::intra_quant_mat: Option<[u8; 64]>` and
  `nonintra_quant_mat: Option<[u8; 64]>` fields surface the resulting
  matrices in zigzag scan order.
* Round 5 — §6.2.6 macroblock-layer header bit-walk for I-VOPs and
  P-VOPs (rectangular shape, 4 non-transparent blocks). New
  `parse_macroblock_header` decodes `not_coded` (P-only), `mcbpc`
  (Tables B.6 / B.7), `ac_pred_flag` (intra MB), `cbpy` (Table B.8),
  and `dquant` (Table 6-32) into a typed
  `MacroblockHeader { not_coded, mb_type, cbpc, ac_pred_flag, cbpy,
  dquant_delta }`. Stuffing macroblocks are transparently skipped
  per §6.2.6. B-VOP and sprite branches return
  `MacroblockParseError::UnsupportedVopKind`.
* Round 6 — §6.2.6 B-VOP macroblock-header prefix. New
  `parse_b_vop_mb_header` decodes `modb` (Table B.3), `mb_type`
  (Table B.4 non-scalable / Table B.5 scalable enhancement layer),
  the 6-bit `cbpb` (4:2:0 rectangular), and `dbquant` (Table 6-33)
  into a typed `BVopMbHeader { modb, mb_type, cbpb, mvdf_present,
  mvdb_present, dbquant_delta }`. `BVopMbType` enumerates
  `Direct` / `Forward` / `Backward` / `Interpolated` per the table
  rows; non-B VOP types return
  `BVopMbParseError::NotBVop`. Motion-vector bodies remain out of
  scope; the bit reader is left positioned at their start.
* Round 7 — §6.2.6.2 `motion_vector(mode)` bitstream decode plus the
  §7.6.3 general motion-vector decoding process. New
  `decode_motion_vector_delta(br, mode, vop_fcode)` walks one
  `motion_vector("forward" / "backward" / "direct")` body — two
  Table B.12 `mv_data` VLCs, each followed by an `r_size`-bit residual
  when `vop_fcode != 1 && mv_data != 0` — and reconstructs the
  differential vector `(MVDx, MVDy)` via the §7.6.3 recurrence
  (`f = 1 << (vop_fcode-1)`,
  `MVD = (Abs(mv_data)-1)*f + residual + 1`, sign from `mv_data`). The
  on-wire `mv_data` is the doubled "vector differences" integer per the
  §7.6.3 note. `reconstruct_motion_vector(delta, px, py, vop_fcode)`
  adds a caller-supplied predictor and applies the Table 7-9 modulo
  wrap into `[low:high]`, yielding the final `(MVx, MVy)`.
* Round 8 — §7.6.5 median-filter motion-vector predictor.
  `predict_motion_vector([Option<MotionVector>; 3])` takes the three
  candidate predictors (`MV1`/`MV2`/`MV3`; `None` marks an invalid
  neighbour — transparent, or outside the current VOP / video packet /
  GOB, all "treated as transparent" per §7.6.5), applies the four
  §7.6.5 validity rules (one invalid → zero; two invalid → both take
  the third; all invalid → zero), and computes `Px = Median(MV1x,
  MV2x, MV3x)` / `Py = Median(MV1y, MV2y, MV3y)`. The spec's worked
  example (`MV1=(-2,3)`, `MV2=(1,5)`, `MV3=(-1,7)` → `Px=-1`, `Py=5`)
  pins `Median(a, b, c)` as the middle of three. The resolved
  `(Px, Py)` feeds straight into `reconstruct_motion_vector`.
  Gathering the candidates from the spatial neighbourhood
  (Figure 7-34 block positions, the four-MV vs single-MV cases, and
  the S(GMC)-VOP averaged-vector substitution of §7.8.7.3) remains
  later-round work — Figure 7-34 is a diagram with no textual position
  list. Direct-mode predictor scaling also remains later-round work.
* Round 9 — §7.4.1.1 intra-DC texture decode, the first stage of the
  §6.2.7 `block(i)` syntax. `decode_intra_dc(br, component)` reads
  `dct_dc_size_luminance` (Table B.13, block index `i < 4`) or
  `dct_dc_size_chrominance` (Table B.14, `i >= 4`), the `size`-bit
  `dct_dc_differential`, and the trailing `marker_bit` (Table B.15
  NOTE 2, only when `size > 8`), then applies the Table B.15
  sign-decode (`half_range = 2^(size-1)`; `c >= half_range` → `+c`,
  else `(c + 1) - 2*half_range`) and returns the signed *differential*
  DC value as `IntraDcDifferential { size, differential }`. The
  §7.4.3.1 spatial DC predictor add (`QF[0] = dct_dc_pred +
  differential`) and the §7.4.1.2 AC coefficient decode remain
  later-round work.

* Round 10 — §7.4.1.2 AC-coefficient (EVENT) decode, the
  `while (!last) DCT coefficient` loop of §6.2.7 `block(i)`, for the
  `short_video_header == 0` / `reversible_vlc == 0` path.
  `decode_ac_event(br, table_kind)` decodes one `(LAST, RUN, LEVEL)`
  EVENT: the common case is a Table B.16 (intra) / Table B.17 (inter)
  Tcoef VLC plus a trailing sign bit (`0` positive, `1` negative). The
  §7.4.1.3 escape prefix `0000 011` selects one of the first three
  escape modes — Type 1 (`ESC 0` + Tcoef VLC, `LEVEL = sign *
  (abs + LMAX)` via Tables B.19/B.20), Type 2 (`ESC 10` + Tcoef VLC,
  `RUN += RMAX + 1` via Tables B.21/B.22), and Type 3 (`ESC 11` +
  `LAST(1) RUN(6) marker LEVEL(12) marker` fixed-length, signed
  two's-complement LEVEL with `0` / `-2048` reserved per Table B.18 b).
  `decode_ac_events(br, table_kind)` runs the full loop, returning every
  EVENT up to and including the `LAST == 1` terminator. The 102-entry
  Tcoef tables live in `src/tcoef_tables.rs` (generated verbatim from
  Tables B.16 / B.17). The reversible-VLC tables (B.23..B.25 + Type 5
  escape), the Type 4 (`short_video_header == 1`) escape, and the
  §7.4.2 inverse scan that places `(RUN, LEVEL)` into the
  zigzag-ordered coefficient array remain later-round work.
* Round 11 — §7.4.2 inverse scan, the conversion of the
  one-dimensional decoded coefficient stream `QFS[64]` into the
  two-dimensional `PQF[v][u]` 8×8 block. Three scan tables transcribed
  verbatim from Figure 7-4 — (a) Alternate-Horizontal, (b) Alternate-
  Vertical, (c) Zigzag — live in `src/scan.rs` as `[[u8; 8]; 8]`
  grids. `events_to_qfs(events, intra_dc)` expands a §7.4.1.2 AC
  EVENT sequence (with an optional §7.4.1.1 intra-DC value at scan
  position 0) into the dense `[i32; 64]` array, defensively returning
  `InverseScanError::Overflow { position }` on a malformed stream that
  would walk past coefficient 63. `inverse_scan(qfs, scan_type)`
  applies the §7.4.2
  `PQF[inv_scan_v[scan_type][n]][inv_scan_u[scan_type][n]] = QFS[n]`
  loop; `events_to_pqf(events, intra_dc, scan_type)` is the one-call
  combination. `select_scan_type(is_intra, ac_pred_flag,
  dc_direction)` encodes the §7.4.2 selection rule (non-intra or
  `ac_pred_flag == 0` → zigzag; intra + AC-pred + DC predictor from
  above C → alternate-vertical; intra + AC-pred + DC predictor from
  left A → alternate-horizontal); the predictor pass that supplies
  `dc_direction` lands later. The `sadct_disable == 0` modified
  inverse scan for non-rectangular VOPs remains out of scope.

* Round 12 — §7.4.3 spatial DC / AC predictor for intra macroblocks
  (the `short_video_header == 0` path). New `src/predictor.rs` module.
  `default_neighbour_dc(bpp)` returns the §7.4.3.1 fallback
  `F[0][0] = 2^(bpp + 2)` used for neighbours outside the VOP / video
  packet or in non-intra MBs. `dc_scaler(component, qs)` evaluates
  Table 7-1's piece-wise linear non-linear DC scaler — Type 1
  (luminance) and Type 2 (chrominance) formulas across
  `1..=4` / `5..=8` / `9..=24` / `>= 25` quantiser bands (chrominance
  merges the middle two columns under `(qs + 13) / 2`).
  `select_dc_direction(fa, fb, fc)` applies the §7.4.3.1 rule
  `|FA-FB| < |FB-FC|` → predict from `C` (above), else from `A`
  (left). `predict_intra_dc(pqfx_dc, dir, fa, fc, dc_scaler_x)`
  evaluates §7.4.3.2 `QFX[0][0] = PQFX[0][0] + chosen / dc_scaler`.
  `predict_intra_ac_row` / `predict_intra_ac_column` apply the
  §7.4.3.3 first-row / first-column scaled-by-`QpC/QpX` (or
  `QpA/QpX`) add, returning `PQFX` unchanged when the predictor
  neighbour is `None` (out of VOP / video packet — "all the
  prediction coefficients of that block are assumed to be zero").
  `saturate_qf` / `saturate_block` apply the §7.4.3.4 `[-2048, 2047]`
  clamp. `NeighbourBlock { dc, qp, first_row, first_column }` and
  `NeighbourPosition { Left, AboveLeft, Above }` surface Figure 7-5
  for the predictor-gathering pass that will land in a later round.

Round 13 covers §7.4.4 inverse quantisation end-to-end for one 8×8
block (Figure 7-7's `QF[v][u] -> F''[v][u] -> F'[v][u] -> F[v][u]`
pipeline). `inverse_quant_method1(qf, w, ctx)` runs the §7.4.4.6
summary pseudo-code: §7.4.4.1.1 intra DC `dc_scaler * QF[0][0]` (via
the existing `predictor::dc_scaler`), §7.4.4.1.2 first method (intra
`(QF*W[0]*qs*2)/16`, non-intra `((2*QF + Sign(QF))*W[1]*qs)/16`),
§7.4.4.4 saturation to `[-2^(bpp+3), 2^(bpp+3) - 1]`, and §7.4.4.5
mismatch control (LSB toggle on `F[7][7]` when the block sum is even,
implemented via XOR per NOTE 1). `inverse_quant_method2` covers the
§7.4.4.2 second method (`(2*|QF|+1)*qs` / `... - 1` for even `qs`,
sign re-incorporated), with §7.4.4.2's "intra DC uses the same method
as method 1" rule honoured. `inverse_quant_intra_dc` /
`inverse_quant_method1_coef` / `inverse_quant_method2_coef` /
`saturation_bounds` / `saturate_fprime` / `InverseQuantContext` are
the public surface.

Round 14 covers §7.4.5 + Annex A inverse DCT. `idct_8x8(coefficients,
bits_per_pixel)` evaluates Annex A.1's orthonormal 8×8 IDCT
`f(x, y) = (2/N) Σ_u Σ_v C(u) C(v) F(u, v) cos((2x+1)uπ/(2N))
cos((2y+1)vπ/(2N))` as a separable two-pass 1-D IDCT against a
lazily-initialised `f64` cosine-table `COS[u][x] =
cos((2x+1)uπ/16)`, then rounds to nearest (§4.1) and saturates the
output to `[-2^bpp, 2^bpp - 1]` per the §7.4.5 closing sentence.
Round-trips a flat block and a deterministic random block within
±1 LSB (the IEEE 1180-1990 §3.3 peak-error tolerance referenced by
Annex A.1's normative modifications) and chains correctly off the
§7.4.4 intra-DC inverse-quant path (`QF[0][0] = 4` at `qs = 5` →
`F''[0][0] = 40` → uniform `f[y][x] = 5`).
`idct_saturation_bounds(bpp)` / `saturate_idct_sample(value, bpp)`
expose the §7.4.5 clamp.

Round 15 covers §6.2.7 `block(i)` macroblock-level texture assembly
for intra I-VOP macroblocks — the per-macroblock driver that wires
rounds 9..14 together. New `src/block.rs`. `decode_intra_block(br, i,
coded, ctx, predictors, quant_matrix)` runs one block's §6.2.7
`block(i)` syntax (always-present differential intra-DC; the
`if (pattern_code[i]) while (!last) DCT coefficient` AC loop) through
the full §7.4.x chain — read coefficients (`decode_intra_dc` +
`decode_ac_events`) → `events_to_qfs` → §7.4.2 `inverse_scan` →
§7.4.3 spatial DC/AC predictor (gated by `ac_pred_flag`) →
§7.4.3.4 `saturate_block` → §7.4.4 inverse quant (method 1 with the
`W[0]` matrix when `quant_type == 1`, else method 2) → §7.4.5 +
Annex A `idct_8x8` → §6.3.2 final clip to `[0, 2^bpp - 1]`.
`decode_intra_macroblock(...)` walks the §6.1.3.9 / Figure 6-8 4:2:0
block order (0,1 / 2,3 luma; 4 Cb; 5 Cr) and assembles the
reconstructed 16×16 luma + 8×8 Cb / 8×8 Cr `IntraMacroblock`.
`pattern_code(cbpy, cbpc)` derives the six per-block coded flags;
`BlockPredictors::outside(bpp, qs)` supplies the §7.4.3 "neighbour
outside the VOP" state; `DEFAULT_INTRA_QUANT_MATRIX` /
`DEFAULT_NONINTRA_QUANT_MATRIX` (§6.3.3) and `de_zigzag` /
`intra_quant_matrix` resolve the method-1 matrix. A synthetic intra MB
with a known DC differential reconstructs to the expected flat spatial
block (method 2 exact; method 1 within the ±1-LSB §7.4.4.5
mismatch-control tolerance), and a coded AC EVENT breaks block
flatness.

Round 16 covers §7.4.3 / Figure 7-5 predictor candidate gathering — the
cross-block walk that resolves each block-to-decode's three Figure 7-5
neighbours `A` (left), `B` (above-left), `C` (above) from a per-VOP
grid of already-decoded blocks. New `src/neighbour.rs`.
`IntraBlockGrid::new(mb_rows, mb_cols)` allocates one
`(2*mb_rows) × (2*mb_cols)` luma sub-grid plus two `mb_rows × mb_cols`
Cb / Cr sub-grids of `Option<BlockNeighbour>`; `record(mb_row, mb_col,
i, Some(BlockNeighbour))` fills cells as blocks are decoded;
`predictors_for(mb_row, mb_col, i, bpp, qs)` walks Figure 7-5 and
returns the `BlockPredictors` argument round 15's
`decode_intra_block` already consumes. Neighbours outside the sub-grid,
recorded `None` (e.g. video-packet boundary), or recorded with
`is_intra == false` fall back to the §7.4.3.1 default
`F[0][0] = 2^(bpp + 2)` (and the §7.4.3.3 first-row / first-column AC
prediction coefficients are zeroed via `None`).
`BlockNeighbour::from_qf(&qf, dc, qp)` extracts the §7.4.3 state from
a reconstructed intra block. `block_grid_position(mb_row, mb_col, i)`
exposes the Figure 6-8 mapping (luma at
`(2*mb_row + top_bit, 2*mb_col + left_bit)`; Cb / Cr each at
`(mb_row, mb_col)`).

The MV-predictor candidate gathering of Figure 7-34 lands in later
rounds, along with inter / B-VOP reconstruction (motion compensation),
the §7.4.4 mismatch-control "feed F[v][u] back into the grid for the
next MB" hook-up, and the `short_video_header == 1` / `data_partitioned`
paths.

Round 18 covers §6.2.5 `video_packet_header` for the rectangular shape
— the resync-marker walk that lets the decoder recover from packet
loss in error-prone transports. New `src/video_packet.rs`.
`parse_video_packet_header(br, &VideoPacketContext)` consumes the
§5.2.5 `next_resync_marker()` stuffing (`0 1*` to byte alignment),
the 17..=23-bit `resync_marker` (`(15+fcode)` zeros + `1` per §6.3.3,
17 for I-VOPs and binary-only, `max(15 + max(fcode_fwd, fcode_bwd),
17)` for B), the Table 6-27 `macroblock_number` (`ceil(log2(total_mbs))`
bits, 1..=14), the `quant_precision`-bit `quant_scale` (0 rejected),
and the `header_extension_code`. When the extension bit is set the
rectangular extension body — `modulo_time_base`, `vop_time_increment`,
`vop_coding_type`, `intra_dc_vlc_thr`, plus `vop_fcode_forward` /
`vop_fcode_backward` per coding type — is decoded into the optional
fields of `VideoPacketHeader`. `macroblock_number_bit_width(total)` /
`total_macroblocks(width, height)` expose the Table 6-27 inputs;
`resync_marker_length(coding_type, fcode_fwd, fcode_bwd)` exposes the
§6.3.3 length formula; `consume_next_resync_marker(br)` runs the
§5.2.5 stuffing; `probe_resync_marker(br, …)` non-destructively asks
"is the next byte-aligned position a marker of the expected length?".
Non-rectangular and binary-only shape, sprite-GMC trajectory, newpred,
and reduced-resolution VOP extension bodies are typed-rejected via
`VideoPacketParseError::UnsupportedBranch`.

## What works today

| Surface                                       | Status |
| --------------------------------------------- | ------ |
| `0x000001B0` Visual Object Sequence start     | parsed |
| `profile_and_level_indication`                | surfaced |
| `0x000001B5` Visual Object start              | typed `VisualObjectHeader` (round 20) |
| `is_visual_object_identifier` + verid / priority | decoded, §6.3.2.3 defaults applied (round 20) |
| `video_signal_type()` flag + body              | typed `Option<VideoSignalType>` (round 20) |
| `colour_description` triple                    | typed `Option<ColourDescription>` (round 20) |
| `ColourDescription::default_when_absent()`     | §6.3.2.4 BT.709 fallback (round 20) |
| `0x000001Bx` Video Object Layer start         | parsed |
| `aspect_ratio_info` 0001..0101 + 1111         | typed `AspectRatio` |
| `vop_time_increment_resolution`               | u16, marker-bit checked |
| `fixed_vop_rate` + `fixed_vop_time_increment` | width per spec formula |
| `video_object_layer_width` / `height`         | 13-bit, marker-bit checked |
| `vol_control_parameters` + `vbv_parameters`   | composed 30/18/26-bit values |
| Studio Profile (`profile == 0xE1..=0xE8`)     | rejected (typed error) |
| FGS branch                                    | branch path not entered |
| Non-rectangular shape                         | rejected (typed error) |
| `0x000001B3` Group-of-VOP header              | parsed (time-code + closed/broken) |
| `0x000001B6` Video Object Plane header        | parsed (I / P / B / S coding types) |
| `vop_coding_type` (Table 6-24)                | typed `VopCodingType` |
| `modulo_time_base` + `vop_time_increment`     | composed into 64-bit tick count |
| `vop_coded == 0` early return                 | typed default-fields VopHeader |
| `vop_rounding_type` / `intra_dc_vlc_thr`      | surfaced (rounding gated on P/S-GMC) |
| `vop_quant` (`quant_precision` 3..=9)         | surfaced as `u16`, default 5-bit width |
| `vop_fcode_forward` / `vop_fcode_backward`    | surfaced, 0 rejected as forbidden |
| Interlaced `top_field_first` / `alt_vert_scan`| consumed structurally (kept aligned) |
| Sprite / scalability / newpred VOP branches   | rejected (typed errors) |
| `interlaced` / `obmc_disable` / `sprite_enable` VOL fields | surfaced (Table 6-19) |
| `quant_precision` / `bits_per_pixel` VOL fields | surfaced (`not_8_bit` path) |
| `quant_type` flag                              | surfaced |
| `load_intra_quant_mat` / `load_nonintra_quant_mat` bodies | decoded into `[u8; 64]` zigzag (round 4) |
| `complexity_estimation_disable` / `resync_marker_disable` | surfaced |
| `data_partitioned` / `reversible_vlc` VOL flags | surfaced |
| `newpred_enable` / `reduced_resolution_vop_enable` (verid != 1) | surfaced; body rejected |
| `scalability` VOL flag                         | surfaced |
| `VopContext::from_vol(&vol)`                   | populates context from VolHeader |
| `VopHeader::from_vol(&vol, payload)`           | one-call VOL+VOP plumbing |
| Sprite-body / complexity-est-header VOL branches | typed `UnsupportedBranch` |
| I/P-VOP macroblock-header bit-walk (rect, 4 blocks) | `parse_macroblock_header` (round 5) |
| `mcbpc` Tables B.6 / B.7                      | linear-prefix VLC decode (round 5) |
| `cbpy` Table B.8 (4 non-transparent blocks)   | linear-prefix VLC decode (round 5) |
| `dquant` Table 6-32                           | 2-bit → ±1/±2 expansion (round 5) |
| `ac_pred_flag` (intra MB)                     | surfaced (round 5) |
| `not_coded` skip (P-VOP)                      | `MacroblockHeader::SKIPPED` (round 5) |
| Stuffing macroblock skip                      | transparent (round 5) |
| B-VOP / S-VOP macroblock (round-5 path)       | typed `UnsupportedVopKind` |
| B-VOP macroblock header prefix                | `parse_b_vop_mb_header` (round 6) |
| `modb` Table B.3 (1 / 01 / 00)                | linear-prefix decode (round 6) |
| `mb_type` Table B.4 (non-scalable B-VOP)      | linear-prefix decode (round 6) |
| `mb_type` Table B.5 (scalable enhancement)    | linear-prefix decode (round 6) |
| `cbpb` (6-bit, 4:2:0 rectangular)             | fixed-width decode (round 6) |
| `dbquant` Table 6-33                          | 1-or-2-bit → 0 / ±2 (round 6) |
| Default `mb_type` when `modb == 1`            | `direct` / `forward` per scalability (round 6) |
| `motion_vector(mode)` body (fwd/bwd/direct)   | `decode_motion_vector_delta` (round 7) |
| `mv_data` Table B.12 VLC (65 codes)           | linear prefix-free decode (round 7) |
| `*_mv_residual` (`r_size = vop_fcode-1`)      | gated read per §6.2.6.2 (round 7) |
| §7.6.3 differential-MV reconstruction         | `(Abs-1)*f + res + 1`, sign (round 7) |
| §7.6.3 predictor add + Table 7-9 modulo wrap  | `reconstruct_motion_vector` (round 7) |
| §7.6.5 median MV predictor + validity rules    | `predict_motion_vector` (round 8) |
| MV-predictor candidate gathering (Figure 7-34) | `MvGrid::predictor_candidates` / `gather_mv_predictor_candidates` (round 30) |
| 1-MV vs 4-MV per-MB MV storage                 | `MbMv::{Absent, OneMv, FourMv}` (round 30) |
| Per-luma-block transparency mask within an MB  | `MbMvRecord::transparent` (round 30) |
| Video-packet / GOB boundary substitution       | `MvGrid::record_absent` (round 30) |
| `dct_dc_size_luminance` Table B.13            | prefix-free VLC decode (round 9) |
| `dct_dc_size_chrominance` Table B.14          | prefix-free VLC decode (round 9) |
| `dct_dc_differential` Table B.15 sign-decode  | `decode_intra_dc` (round 9) |
| intra-DC `marker_bit` (`size > 8`)            | consumed + validated (round 9) |
| AC EVENT Tcoef VLC Tables B.16 / B.17         | prefix-free decode + sign bit (round 10) |
| §7.4.1.3 escape Type 1 (`ESC 0`, LMAX)        | `decode_ac_event` (round 10) |
| §7.4.1.3 escape Type 2 (`ESC 10`, RMAX)       | `decode_ac_event` (round 10) |
| §7.4.1.3 escape Type 3 (`ESC 11`, FLC + markers) | `decode_ac_event` (round 10) |
| §7.4.1.3 escape Type 4 (`short_video_header == 1`, ESC + LAST + RUN + 8-bit LEVEL) | `decode_ac_event_short_video_header` (round 17) |
| §6.2.7 `while (!last)` EVENT loop             | `decode_ac_events` (round 10) |
| §6.2.7 `while (!last)` EVENT loop (SVH=1)     | `decode_ac_events_short_video_header` (round 17) |
| Figure 7-4 (a) Alternate-Horizontal scan table | transcribed (round 11) |
| Figure 7-4 (b) Alternate-Vertical scan table   | transcribed (round 11) |
| Figure 7-4 (c) Zigzag scan table               | transcribed (round 11) |
| §7.4.2 `QFS[n] → PQF[v][u]` inverse scan      | `inverse_scan` (round 11) |
| AC EVENTs (+ optional intra-DC) → `QFS[64]`   | `events_to_qfs` (round 11) |
| §7.4.2 per-block scan-type selection          | `select_scan_type` (round 11) |
| §7.4.3.1 default neighbour `F[0][0] = 2^(bpp+2)` | `default_neighbour_dc` (round 12) |
| Table 7-1 `dc_scaler` (Type 1 / Type 2)          | `dc_scaler` (round 12) |
| §7.4.3.1 `\|FA-FB\| < \|FB-FC\|` direction rule  | `select_dc_direction` (round 12) |
| §7.4.3.2 `QFX[0][0] = PQFX + chosen / dc_scaler` | `predict_intra_dc` (round 12) |
| §7.4.3.3 AC first-row `(QFC*QpC)/QpX` add        | `predict_intra_ac_row` (round 12) |
| §7.4.3.3 AC first-col `(QFA*QpA)/QpX` add        | `predict_intra_ac_column` (round 12) |
| §7.4.3.3 missing-neighbour zero-coefficient rule | `qfa/qfc_col == None` → pass-through (round 12) |
| §7.4.3.4 `QF[v][u]` saturation `[-2048, 2047]`   | `saturate_qf` / `saturate_block` (round 12) |
| §7.4.3 predictor-neighbour gathering            | `IntraBlockGrid::predictors_for` (round 16) |
| Figure 6-8 block-grid layout (4:2:0)            | `block_grid_position` (round 16) |
| §7.4.3.1 non-intra-neighbour fallback           | `is_intra` gate in grid lookup (round 16) |
| §7.4.3.3 out-of-VOP zero AC prediction          | `None` first row / column from grid (round 16) |
| §7.4.4.1.1 intra DC `F''[0][0] = dc_scaler * QF[0][0]` | `inverse_quant_intra_dc` (round 13) |
| §7.4.4.1.2 method 1 intra `(QF*W*qs*2)/16`        | `inverse_quant_method1_coef` (round 13) |
| §7.4.4.1.2 method 1 non-intra `((2*QF+Sign)*W*qs)/16` | `inverse_quant_method1_coef` (round 13) |
| §7.4.4.2.1 method 2 (odd / even `qs`)             | `inverse_quant_method2_coef` (round 13) |
| §7.4.4.3 `short_video_header == 1` DC fixed-8     | gated in `inverse_quant_intra_dc` (round 13) |
| §7.4.4.4 saturation `[-2^(bpp+3), 2^(bpp+3) - 1]` | `saturate_fprime` / `saturation_bounds` (round 13) |
| §7.4.4.5 mismatch control on `F[7][7]`            | fused into `inverse_quant_method1` (round 13) |
| §7.4.4.6 method-1 summary pseudo-code             | `inverse_quant_method1` (round 13) |
| §7.4.5 + Annex A.1 orthonormal 8×8 IDCT           | `idct_8x8` (round 14) |
| §7.4.5 output saturation `[-2^bpp, 2^bpp - 1]`    | `idct_saturation_bounds` / `saturate_idct_sample` (round 14) |
| §6.2.7 `block(i)` intra-DC + AC loop driver       | `decode_intra_block` (round 15) |
| §6.2.7 `pattern_code[i]` from `cbpy` / `cbpc`     | `pattern_code` (round 15) |
| §6.1.3.9 / Figure 6-8 4:2:0 block assembly        | `decode_intra_macroblock` (round 15) |
| Intra MB reconstruct → 16×16 luma + 8×8 Cb/Cr     | `IntraMacroblock` (round 15) |
| §6.3.3 default intra / non-intra quant matrices   | `DEFAULT_*_QUANT_MATRIX` (round 15) |
| zigzag → raster quant-matrix de-scan              | `de_zigzag` / `intra_quant_matrix` (round 15) |
| §6.3.2 intra final clip `[0, 2^bpp - 1]`          | fused into `decode_intra_block` (round 15) |
| §6.2.7 `block(i)` inter-block driver (no DC)       | `decode_inter_block` (round 21) |
| §7.4.2 zigzag scan for inter (non-intra) blocks    | gated in `decode_inter_block` (round 21) |
| §7.4.4 inverse quant non-intra (method 1 + 2)      | `macroblock_intra == false` path (round 21) |
| §7.3 step-2 inter residual `f[y][x]` (no clip)     | `decode_inter_block` output (round 21) |
| Inter MB residual assembly (16×16 luma + 8×8 Cb/Cr) | `InterMacroblock` (round 21) |
| `decode_inter_macroblock` (Inter / InterQ / Inter4V) | round 21 |
| `nonintra_quant_matrix(vol)` `W[1]` resolver       | round 21 |
| Motion compensation (`p[y][x]` + `f[y][x]` + §7.3 step-3 clip) | not yet |
| §7.6.9.5.3 B-VOP 8×8 luminance prediction-block generation | `generate_b_vop_luma_prediction` (round 26) |
| §7.6.9.4 / §7.6.9.5.3 bidirectional averaging `(Pf + Pb + 1) >> 1` | `average_bidirectional_into` (round 26) |
| §7.6.9.2 / §7.6.9.3 / §7.6.9.4 forward / backward / interpolated MB modes | `BVopPredictionMode::{ForwardOnly, BackwardOnly, Bidirectional}` (round 26) |
| §7.6.9.5 direct mode 16×16 luma prediction              | `BVopPredictionMode::Direct` (round 26) |
| B-VOP reconstruction (motion comp)                | luma: round 26; chroma derivation: round 27; chroma MC + residual add still pending |
| §7.6.5 chrominance MV derivation from K = 1 luma MV (Table 7-13) | `chroma_mv_from_luma_blocks` (round 27) |
| §7.6.5 chrominance MV derivation from K = 2 luma MVs (Table 7-12) | `chroma_mv_from_luma_blocks` (round 27) |
| §7.6.5 chrominance MV derivation from K = 3 luma MVs (Table 7-11) | `chroma_mv_from_luma_blocks` (round 27) |
| §7.6.5 chrominance MV derivation from K = 4 luma MVs (Table 7-10) | `chroma_mv_from_luma_blocks` (round 27) |
| §7.6.5 quarter-sample-mode chroma-MV pre-divide rounding | caller-applied via `reduce_qpel_to_half_pel_chroma`; spec text gap |
| RVLC Tcoef tables (B.23..B.25) + Type 5 escape | not yet |
| `sadct_disable == 0` modified inverse scan    | not yet |
| §5.2.5 `next_resync_marker()` stuffing        | `consume_next_resync_marker` (round 18) |
| §6.3.3 `resync_marker` length (17..=23)        | `resync_marker_length` (round 18) |
| Table 6-27 `macroblock_number` bit width       | `macroblock_number_bit_width` (round 18) |
| §6.2.5 `video_packet_header` (rectangular)     | `parse_video_packet_header` (round 18) |
| `header_extension_code == 1` rectangular body  | typed `VideoPacketHeader` (round 18) |
| Encoder                                       | not yet |

587 round-1..30 unit tests + 9 doctests pass.

## Provenance

Every numeric value and bit layout in this crate is sourced from
ISO/IEC 14496-2:2004 (3rd edition) — Tables 6-3 (start codes), 6-14
(aspect ratios), 6-15 (chroma formats), 6-16 (shape types), 6-19
(`sprite_enable`), 6-23 (time-code layout), 6-24 (`vop_coding_type`),
6-25 (`intra_dc_vlc_thr`), 6-32 (`dquant` codes), Annex B Tables B.1
(mb_type ↔ derived_mb_type ↔ included elements), B.3 (modb VLC),
B.4 (B-VOP mb_type, non-scalable), B.5 (B-VOP mb_type, scalable
enhancement layer), B.6 (mcbpc I-VOP), B.7 (mcbpc P-VOP), B.8
(cbpy 4-block), B.12 (MVD VLC, 65 codes), B.13 (dct_dc_size_luminance),
B.14 (dct_dc_size_chrominance), B.15 (dct_dc_differential additional
codes), B.16 (intra Tcoef EVENT VLC, 102 codes), B.17 (inter Tcoef
EVENT VLC, 102 codes), B.18 (FLC for Type-3 escape RUN/LEVEL),
B.19/B.20 (LMAX, intra/inter), B.21/B.22 (RMAX, intra/inter), B.18 c
(FLC for Type-4 escape LEVEL in the short-video-header path: 8-bit
signed two's-complement with `0000 0000` and `1000 0000` reserved) — and
Table 6-33 (dbquant),
Table 7-9 (motion-vector range per vop_fcode), Table 7-10
(sixteenth-sample chroma-MV rounding, 16 entries), Table 7-11
(twelfth-sample chroma-MV rounding, 12 entries), Table 7-12
(eighth-sample chroma-MV rounding, 8 entries), Table 7-13
(fourth-sample chroma-MV rounding, 4 entries), and the syntax tables of
§6.2.2 / §6.2.3 / §6.2.4 / §6.2.5 / §6.2.6 plus the semantics in
§6.3.3 (including the `8*[2-64]` zigzag-ordered quant-matrix list,
the zero-sentinel "remaining values set equal to the last non-zero
value" rule, and the intra "shall always be 8" / non-intra "shall
not be 0" first-value constraints) / §6.3.4 / §6.3.5 / §6.3.6
(macroblock-related semantics, including the B-VOP `modb` / `mb_type`
/ `cbpb` / `dbquant` rules and the default-type "direct" vs
"forward mc + Q" choice from §7.9.2.8.3), §6.2.6.2 (`motion_vector`
syntax), §6.3.6.2 (`*_mv_data` / `*_mv_residual` + the
`r_size = vop_fcode - 1` rule), and §7.6.3 (the general motion-vector
decoding process: the `f`/`low`/`high`/`range` recurrence, the
`MVD = (Abs(mv_data)-1)*f + residual + 1` reconstruction, the
predictor add and the `[low:high]` modulo wrap, plus the note that
`mv_data` is two times the Table B.12 "vector differences" value), and
§7.6.5 (the progressive P-/S(GMC)-VOP median-filter predictor: the
four candidate-predictor validity rules, the `Px = Median(MV1x, MV2x,
MV3x)` / `Py = Median(MV1y, MV2y, MV3y)` combination, and the worked
example `MV1=(-2,3)`, `MV2=(1,5)`, `MV3=(-1,7)` → `Px=-1`, `Py=5`
that fixes `Median(a, b, c)` as the middle of three since the §4.1
operator clause does not define it; plus Figure 7-34's per-current-block
spatial layout of MV1 / MV2 / MV3 transcribed in
`docs/video/mpeg4-visual/figure-7-34-mv-predictor-layout.md` as the
authoritative source for the four block-position cases of the §7.6.5
gather, the "1-MV macroblock uses the top-left case" rule from the
paragraph preceding Figure 7-34, and the boundary-substitution rule
that treats neighbouring MBs outside the current VOP / video packet /
GOB as transparent), and §7.4.1.1 / §6.2.7 (`block(i)`)
/ §6.3.7 (the intra-DC texture decode: the `dct_dc_size_luminance`
(`i < 4`) / `dct_dc_size_chrominance` (`i >= 4`) split, the
`if (dct_dc_size != 0) dct_dc_differential` and
`if (dct_dc_size > 8) marker_bit` gates, and the Table B.15
sign-decode — `half_range = 2^(size-1)`; an additional code
`c >= half_range` → `+c`, else `(c + 1) - 2*half_range` — confirmed
against every Table B.15 boundary row), and §7.4.1.2 / §7.4.1.3
(the AC-EVENT decode: the Table B.16 (intra) / B.17 (inter) Tcoef VLC
+ sign-bit common path, and the three `short_video_header == 0`
escape modes — Type 1 `LEVEL = sign * (abs + LMAX(LAST, RUN))` from
Tables B.19/B.20, Type 2 `RUN = RUN + RMAX(LAST, LEVEL) + 1` from
Tables B.21/B.22, and the Type 3 fixed-length `LAST(1) RUN(6) marker
LEVEL(12) marker` with the signed-12-bit two's-complement LEVEL and
the reserved `0` / `-2048` values from Table B.18 b), and §7.4.2
(the inverse-scan algorithm: the
`PQF[inv_scan_v[scan_type][n]][inv_scan_u[scan_type][n]] = QFS[n]`
loop over the three Figure 7-4 scan tables (a) Alternate-Horizontal,
(b) Alternate-Vertical, (c) Zigzag, and the per-block scan-type
selection — non-intra → zigzag; intra + `ac_pred_flag == 0` → zigzag
for the whole macroblock; intra + `ac_pred_flag == 1` + DC predictor
from C → alternate-vertical; intra + `ac_pred_flag == 1` + DC
predictor from A → alternate-horizontal), and §7.4.3 (the spatial
DC/AC predictor for intra macroblocks: §7.4.3.1's
`|FA-FB| < |FB-FC|` direction selection and the
`F[0][0] = 2^(bits_per_pixel + 2)` default-neighbour rule; §7.4.3.2's
`QFX[0][0] = PQFX[0][0] + chosen / dc_scaler` reconstruction with
the chosen value being `FA[0][0]` or `FC[0][0]` per the §7.4.3.1
direction; Table 7-1's piece-wise linear `dc_scaler` — luminance
(Type 1) `1..=4` → 8, `5..=8` → 2*qs, `9..=24` → qs+8,
`>= 25` → 2*qs-16; chrominance (Type 2) `1..=4` → 8,
`5..=24` → (qs+13)/2, `>= 25` → qs-6; §7.4.3.3's
`QFX[v][0] = PQFX[v][0] + (QFA[v][0] * QpA) // QpX` and
`QFX[0][u] = PQFX[0][u] + (QFC[0][u] * QpC) // QpX` AC scaling, with
the §7.4.3.3 "all the prediction coefficients of that block are
assumed to be zero" rule for an out-of-VOP predictor block; and
§7.4.3.4's `[-2048, 2047]` saturation), and §7.4.4 (the inverse
quantisation pipeline of Figure 7-7 for one 8×8 block: §7.4.4.1.1's
intra DC formula `F''[0][0] = dc_scaler * QF[0][0]` with the Table
7-1 (a.k.a. §7.4.4.3 nonlinear DC) scaler for
`short_video_header == 0` and the fixed `dc_scaler = 8` of §7.4.1.1
otherwise; §7.4.4.1.2's first method —
`F''[v][u] = (QF[v][u] * W[0][v][u] * quantiser_scale * 2) / 16` for
intra blocks and
`F''[v][u] = ((2*QF[v][u] + Sign(QF[v][u])) * W[1][v][u] * quantiser_scale) / 16`
for non-intra blocks; §7.4.4.2.1's second method —
`F''[v][u] = (2*|QF[v][u]| + 1) * quantiser_scale` for odd
`quantiser_scale`, the same minus one for even `quantiser_scale`,
sign re-applied to obtain the final `F''[v][u]`, with §7.4.4.2's
note that the intra DC coefficient is still quantised by the method
of §7.4.4.1.1; §7.4.4.4's saturation of `F''[v][u]` to
`[-2^(bits_per_pixel + 3), 2^(bits_per_pixel + 3) - 1]`; §7.4.4.5's
mismatch control on the parity of the block sum, with the LSB toggle
on `F[7][7]` per NOTE 1; and §7.4.4.6's "summary of quantiser process
for method 1" pseudo-code that fuses all of the above, together with
§4.1's clarification that `/` is integer division with truncation
toward zero — Rust's `/` on signed integers matches — and the §4.1
definition of `Sign(x) = 1` for `x >= 0`, `-1` for `x < 0`), and
§7.4.5 + Annex A.1 (the inverse DCT: §7.4.5's "the inverse DCT
transform defined in Annex A shall be applied to obtain the inverse
transformed values, `f[y][x]`. These values shall be saturated so
that `-2^bits_per_pixel ≤ f[y][x] ≤ 2^bits_per_pixel - 1`"; and
Annex A.1's normative IDCT formula
`f(x, y) = (2/N) Σ_u Σ_v C(u) C(v) F(u, v) cos((2x+1)uπ/(2N))
cos((2y+1)vπ/(2N))` with `N = 8`, `C(0) = 1/√2`, `C(k) = 1` for
`k ≠ 0`, together with Annex A.1's reference to IEEE Std 1180-1990
with the two normative deviations on §3.2 (test set size +
parameters) and §3.3 (peak per-pixel error ≤ 1 for any IDCT
claiming compliance against the reference saturated mathematical
integer-number IDCT)), and §6.2.7 / §6.1.3.9 / §6.3.7 / §6.3.3
(the `block(i)` macroblock-level assembly: the §6.2.7 `block(i)`
intra-DC branch followed by the `if (pattern_code[i]) while (!last)
DCT coefficient` AC loop; the Figure 6-8 4:2:0 block ordering — four
luminance blocks (0,1 / 2,3) then Cb (4) and Cr (5); the §6.2.6
`for (i = 0; i < block_count; i++) block(i)` loop with `block_count
= 6`; the §6.3.7 `cbpy` / `cbpc` semantics that set `pattern_code[i]`
when a block carries one or more coded AC coefficients, mapped to the
six blocks via `cbp = (cbpy << 2) | cbpc` per the §6.2.7
`if (cbp & (1 << (5 - i)))` derivation; and the §6.3.3 default intra
matrix `[[8,17,18,…],…]` and default non-intra matrix
`[[16,17,18,…],…]` used when no quantiser matrix was loaded), and
§7.4.3 / Figure 7-5 / §6.1.3 / Figure 6-8 (the predictor candidate
gathering of round 16: the Figure 7-5 `A` (left) / `B` (above-left) /
`C` (above) layout for the block-to-decode `X`; the Figure 6-8 4:2:0
block ordering that determines, given a macroblock at `(mb_row, mb_col)`
and block index `i ∈ 0..6`, the per-component sub-grid position
(luma 0..=3 at the 2×2 cells `(2*mb_row + top_bit, 2*mb_col + left_bit)`,
Cb 4 / Cr 5 each at `(mb_row, mb_col)`); and §7.4.3.1's two defaults
applied when a candidate is unavailable — "If any of the blocks A, B or
C are outside of the VOP boundary, or the video packet boundary, or
they do not belong to an intra coded macroblock, their `F[0][0]`
values are assumed to take a value of `2^(bits_per_pixel + 2)`" plus
§7.4.3.3's "If the prediction block (block 'A' or block 'C') is
outside of the boundary of the VOP or video packet, then all the
prediction coefficients of that block are assumed to be zero"), and
§5.2.5 / §6.2.5 / §6.3.3 / Table 6-27 (the `video_packet_header` of
round 18: §5.2.5's `next_resync_marker()` zero-bit-plus-`1`-stuffing
to byte alignment; the §6.2.5 `video_packet_header()` syntax table
read in the order — `next_resync_marker()` → `resync_marker` →
`macroblock_number` → `quant_scale` (rectangular shape skips the
shape-extension body since `video_object_layer_shape != "rectangular"`
gates it out) → `header_extension_code` → (when set)
`modulo_time_base` → `vop_time_increment` → `vop_coding_type` →
`intra_dc_vlc_thr` → optional `vop_fcode_forward` (not I) → optional
`vop_fcode_backward` (B only); the §6.3.3 `resync_marker` length
formula — 17 bits for I-VOPs and binary-only shape (16 zeros + 1),
`(15 + fcode)` zeros + 1 for P / S(GMC), `max(15 + max(fcode_fwd,
fcode_bwd), 17)` zeros + 1 for B; §6.3.3's `macroblock_number`
fixed-length code in 1..=14 bits whose width is selected from Table
6-27 against `((video_object_layer_width + 15) / 16) *
((video_object_layer_height + 15) / 16)`; §6.3.3's `quant_scale` "an
unsigned integer which specifies the absolute value of quantiser
scale", `quant_precision` bits wide (default 5); §6.3.3's
`header_extension_code` semantics that enumerate exactly which
optional fields land in the extension body and the rectangular shape
case where the bit is read *after* `quant_scale`), and §6.2.2 /
§6.3.2.3 / §6.3.2.4 (the `VisualObject()` header of round 20: the
1-bit `is_visual_object_identifier` flag and the 4-bit
`visual_object_verid` + 3-bit `visual_object_priority` it gates;
the §6.3.2.3 defaults `verid = 1` ("When this field does not
exist, the value of `visual_object_verid` is `0001`") and
`priority = 1` (the highest legal priority, since "value of zero is
reserved"); the §6.2.2 `video_signal_type()` body — 1-bit
`video_signal_type` flag, 3-bit `video_format` (Table 6-7), 1-bit
`video_range`, 1-bit `colour_description`, and — when set — 8-bit
`colour_primaries` (Table 6-8) + 8-bit `transfer_characteristics`
(Table 6-9) + 8-bit `matrix_coefficients` (Table 6-10); and the
§6.3.2.4 absent-block defaults — "In the case that
`video_signal_type()` is not present in the bitstream or
`colour_description` is zero the chromaticity / transfer
characteristics / matrix coefficients are assumed to be those
corresponding to ... having the value 1" + "In the case that
`video_signal_type()` is not present in the bitstream, `video_range`
is assumed to have the value 0"), and §6.2.7 / §7.3 / §7.4.x (the
round-21 inter-block driver: the §6.2.7 `block(i)` syntax table gating
the entire intra-DC prologue on `(!data_partitioned &&
(derived_mb_type == 3 || derived_mb_type == 4))` — i.e. an inter
`block(i)` is `if (pattern_code[i]) while (!last) DCT coefficient`,
with no DC bits, no marker, and no §7.4.3 spatial predictor; §7.4.2's
"non-intra blocks → zigzag" scan-selection rule; §7.4.4's non-intra
method-1 formula `F''[v][u] = ((2*QF[v][u] + Sign(QF[v][u])) *
W[1][v][u] * quantiser_scale) / 16` and §7.4.4.2 method-2 formula
`F''[v][u] = (2*|QF[v][u]| + 1) * quantiser_scale` (or the same minus
one for even `quantiser_scale`, with the §7.4.4.2 trailing sign-
incorporation `F''[v][u] = Sign(QF[v][u]) * |F''[v][u]|`); §7.4.5 +
Annex A.1's IDCT saturation of `f[y][x]` to `[-2^bpp, 2^bpp - 1]`; and
the §7.3 VOP-reconstruction split — step 2 "In case of inter
macroblocks, … the decoded texture data `f[y][x]` is added to the
prediction values, resulting in the final luminance and chrominance
values of the VOP: `d[y][x] = p[y][x] + f[y][x]`" plus step 3 "the
calculated luminance and chrominance values of the reconstructed VOP
are saturated so that `0 ≤ d[y][x] ≤ 2^bpp - 1`" — fixing that the
inter block-driver's output is the §7.3 step-2 residual `f[y][x]`,
NOT the §7.3 step-3 display-clipped `d[y][x]`), and §7.6.9.2 /
§7.6.9.3 / §7.6.9.4 / §7.6.9.5.3 (the round-26 B-VOP luminance
prediction-block generation: §7.6.9.2's forward-mode "Only the
forward vector (MVFx, MVFy) is applied in this mode. The prediction
blocks Pf_Y, Pf_U, and Pf_V are…"; §7.6.9.3's backward-mode mirror
formulation; §7.6.9.4's bi-directional rule "Both the forward vector
(MVFx, MVFy) and the backward vector (MVBx, MVBy) are applied in
this mode. The prediction blocks Pi_Y, Pi_U, and Pi_V are generated
from the forward and backward reference VOPs by doing the forward
prediction, the backward prediction and then averaging both
predictions pixel by pixel" with the explicit `Pi[i][j] = (Pf[i][j]
+ Pb[i][j] + 1) >> 1` formula for luma 16×16 and chroma 8×8; and
§7.6.9.5.3's "Motion compensation for luminance is performed
individually on 8x8 blocks to generate a macroblock. The process of
generating a prediction block consists of using computed forward
and backward motion vectors {(MVFx[i], MVFy[i]), (MVBx[i],
MVBy[i]), i = 0,1,2,3} to obtain appropriate blocks from reference
VOPs and averaging these blocks, same as the case of bi-directional
mode except that motion compensation is performed on 8x8 blocks",
together with §7.6.9.5.3's note on §7.6.2.2 block-boundary
mirroring in quarter-sample mode — four 8×8 sub-blocks with the
same MV do not collapse to one 16×16 fetch — and Figure 6-8's
TL/TR/BL/BR 8×8 sub-block ordering inside a 16×16 macroblock),
and §7.6.5 (the round-27 chrominance MV derivation: the
paragraph immediately above Tables 7-10..7-13 — "Motion vector
MVDCHR for both chrominance blocks is derived by calculating the
sum of the K luminance vectors, that corresponds to K 8x8 blocks
that do not lie outside the VOP shape and dividing this sum by
2*K ... The component values of the resulting
sixteenth/twelfth/eighth/fourth sample resolution vectors are
modified towards the nearest half sample position as indicated
below." — plus the four tables themselves verbatim: Table 7-10
(16 entries, sixteenth resolution → half), Table 7-11 (12
entries, twelfth), Table 7-12 (8 entries, eighth), Table 7-13
(4 entries, fourth), and the §7.6.9.5.3 closing reference "For
the motion compensation of both chrominance blocks, the forward
motion vector (MVFx_chro, MVFy_chro) is calculated by the sum
of K forward luminance motion vectors dividing by 2K and then
rounding toward the nearest half sample position as defined in
Table 7-10 to Table 7-13" that ties the round-27 entry point
into the B-VOP direct-mode chroma path).
The text was read from
`docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`. No
third-party MPEG-4 source was consulted.

## License

MIT — see [LICENSE](./LICENSE).

# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Round 53 of the clean-room rebuild — **GMC subsystem, part 1
  (configuration syntax)**: the §6.2.3 `sprite_enable == "GMC"` VOL
  body. `parse_video_object_layer` now decodes
  `no_of_sprite_warping_points` (Table 6-20: `0` stationary, `1`
  translation, `2`/`3` affine — `4` perspective is typed-rejected as
  disallowed under GMC, `5..=63` rejected as `ReservedWarpingPoints`),
  `sprite_warping_accuracy` (Table 6-21 → new `SpriteWarpingAccuracy`
  enum with `s()` returning the `2 / 4 / 8 / 16` sub-pel denominator),
  and `sprite_brightness_change` (mandated `0` under GMC, a non-zero
  bit is rejected). The GMC branch correctly skips the static-only
  `sprite_width` / `sprite_height` / coordinate fields and the
  `low_latency_sprite_enable` bit (all gated `if (sprite_enable !=
  "GMC")` in §6.2.3). The `static` branch stays typed-rejected — it
  needs the §7.8.2 sprite-object buffer + §7.8.3 piece-update
  machinery, out of scope for the GMC milestone. New `VolHeader`
  fields `no_of_sprite_warping_points` / `sprite_warping_accuracy` /
  `sprite_brightness_change` (all `Option`, `None` when sprite coding
  is off). Tests cover the affine 2-point body, the stationary
  0-point body, the perspective-rejection and reserved-count paths,
  and the accuracy → `s` mapping.
  ∆DC-SA-DCT is the transform used for intra-coded 8×8 blocks of a
  non-rectangular VOP with `opaque_pels < 64`: the encoder separates the
  block mean before the forward SA-DCT (∆S1) and re-injects it as a scaled
  `F[0][0] = 8·mean_value` (∆S3). The decoder undoes this in steps
  I-∆S1..I-∆S4:
  - **I-∆S1** extracts `mean_value = F[0][0] / 8` and zeroes `F[0][0]`.
  - **I-∆S2** runs the inverse SA-DCT body (I-S1..I-S5) on the modified
    coefficients, refactored to a shared floating-point core
    (`inverse_sadct_float`) so the correction operates on the unrounded
    `f[y][x]` per Annex A NOTE 2.
  - **I-∆S3** derives the correction term `corr_term = check_sum / sqrt_sum`,
    with `check_sum = Σ f[y][x]` over opaque samples and
    `sqrt_sum = Σ √pels_height[x]` over non-empty columns.
  - **I-∆S4** adds `mean_value − corr_term/√pels_height[x]` to every opaque
    sample and rounds to integers.
  Tests cover the flat-mean reconstruction, the closed-form I-∆S4 additive
  identity, the zero-correction degenerate case, and a full forward/inverse
  round-trip through a test-local forward SA-DCT.
- Round 51 of the clean-room rebuild: the Annex A §A.3.2 **inverse
  shape-adaptive DCT** (SA-DCT) transform body — steps I-S1..I-S5 — in a
  new `inverse_sadct` module. This is the §7.3.5 / Table 7-2 transform
  that replaces the textbook 8×8 inverse DCT in non-rectangular VOPs
  (`sadct_disable == 0`) for blocks with `opaque_pels < 64`. It consumes
  the `PQF[v][u]` layout produced by the §7.4.2 modified inverse scan and
  reconverts it to decoded texture `f[y][x]`:
  - **I-S1** re-derives the full shape-parameter set (`coeff_width[v]`,
    `pels_height[x]`, `shift_shape[y][x]`) from the decoded binary shape
    `f_shape[y][x]` — the `coeff_width` / `opaque_pels` halves already
    drove the scan; the transform also needs `pels_height` and
    `shift_shape` for the re-shifts.
  - **I-S2 / I-S4** apply variable-length, shape-adaptive 1-D inverse
    DCTs (`coeff_width[v]`-point per row, then `pels_height[x]`-point per
    column) using the orthonormal `√(2/N)` / `C(0)=√0.5` kernel with `N`
    taken from the shape parameter rather than fixed at 8.
  - **I-S3 / I-S5** re-shift the intermediate coefficients and final
    pels from their left-/top-packed positions back to the original
    column/row positions defined by `shift_shape` / `f_shape`.
  Floating-point per Annex A NOTE 3, output rounded half-away-from-zero
  (§4.1). Two transcription typos in the ISO listing (`coff_count` in
  I-S3, the missing `[pels_count]` row index in I-S5) are documented and
  corrected. 8 new tests: the full-opaque block reduces exactly to the
  Annex A §A.1 8×8 IDCT, DC-only flat reconstruction, single-opaque-row
  and single-opaque-column DC behaviour, staircase shape-param
  invariants, transparent-position and all-zero-block guarantees.
- Round 50 of the clean-room rebuild: the §7.4.2 **modified inverse
  scan** for the `sadct_disable == 0` shape-adaptive-DCT path. In a
  non-rectangular VOP, an 8×8 block with fewer than 64 opaque pels
  carries only `opaque_pels` SA-DCT coefficients, packed `coeff_width[v]`
  per row against the left edge. New `scan::modified_inverse_scan`
  implements the spec's `coeff_width`-aware loop — it walks the chosen
  Figure 7-4 scan path, writes a decoded coefficient only where
  `coeff_width[v] > u`, and zero-fills every SA-DCT-undefined position
  (§7.4.2 NOTE 1, so subsequent AC prediction is not confused). The
  auxiliary `coeff_width[v]` array and the `opaque_pels` total are
  derived from the decoded 8×8 binary shape via
  `scan::ShapeParams::from_shape`, transcribing Annex A §A.3.2 step
  I-S1 (per-column vertical shift of opaque pels, then a per-row count).
  `scan::events_to_pqf_sadct` is the one-call EVENTs → `PQF[v][u]`
  convenience for the SA-DCT path. 13 new tests cover the shape-param
  derivation (full / empty / quadrant / non-contiguous vertical-shift /
  staircase / L-shape), the full-width equivalence to the plain inverse
  scan, the zero-fill constraint, scan-order coefficient packing, and
  the `coeff_count == opaque_pels` invariant.
- Round 49 of the clean-room rebuild: the §E.1.4.4.2.1 **two-way RVLC
  error-recovery strategy selection** — the arbitration that decides,
  after both the forward and the backward `reversible_vlc == 1` Tcoef
  decodes have run, how many macroblocks to keep from the beginning
  (forward result) and how many from the end (backward result), with
  the straddling middle discarded. New `src/rvlc_arbitration.rs`:
  `RvlcArbitration::select(&RvlcArbitrationInput)` classifies the packet
  into one of the four §E.1.4.4.2.1 strategies from the two predicates
  `L1 + L2 >= L` (decodable-bit overlap) and `N1 + N2 >= N`
  (completely-decoded-MB overlap), then computes `keep_front` /
  `keep_back` from the strategy's formula — Strategy 1 `f_mb(L1-T)` /
  `b_mb(L2-T)`, Strategy 2 `N-N2-1` / `N-N1-1`, Strategy 3 `N-b_mb(L2)`
  / `N-f_mb(L1)`, Strategy 4 the `min` of the two. The `f_mb(S)` /
  `b_mb(S)` step-inverse counters are evaluated over per-MB cumulative
  bit-cost tables (`partition_point`), with the §E.1.4.4.2 counter rule
  (a non-positive `S` yields 0). `RVLC_THRESHOLD` exposes the spec's
  `T = 90`. The two kept regions are clamped disjoint so the discard
  region never goes negative. `RvlcArbitration::displayed_mbs` applies
  the §E.1.4.4.2.2 INTRA-MB concealment pass — every INTRA MB in an
  errored packet is concealed (not displayed), even one a strategy
  would otherwise have kept. Nine unit tests cover each strategy's
  formula, the step-inverse counters, the disjoint-region clamp, the
  `0 <= N1,N2 <= N-1` bound, and the concealment pass. Re-exported at
  the crate root.
- Round 48 of the clean-room rebuild: the §E.1.4.4 **backward
  (reverse-direction) reversible-VLC Tcoef decode** — the second half of
  the Annex E two-way RVLC error-recovery method (round 47 landed the
  forward half). A new `BackwardBitReader` (`src/bitreader.rs`) reads a
  half-open absolute bit range `[start_bit, end_bit)` from `end_bit`
  toward `start_bit` (`read_bit_reverse` emits stream bits tail-first;
  `read_bits_value(n)` reassembles a fixed-length field in forward bit
  order). `decode_ac_event_rvlc_reverse` decodes one EVENT in reverse:
  the sign bit `s` is consumed first, then the four bits below it select
  the common path (a Table B.23 reversible VLC, matched against its
  bit-reversed code — RVLC codes are reverse-decodable by construction,
  §E.1.3) or the Type-5 escape, whose closing `0000` delimiter is the
  escape marker in the reverse direction (`00001` is *not* used backward,
  §E.1.4.4.1). The escape payload `m LEVEL(11) m RUN(6) LAST(1)` is read
  in reverse order, terminating with the `00001` opener (tail-first
  `10000`). `decode_ac_events_rvlc_reverse(data, start_bit, end_bit,
  table_kind)` runs the reverse loop over a video-packet DCT-coefficient
  region across block boundaries and returns the recovered EVENTs in
  **forward** scan order; per Annex E, a mid-region illegal RVLC stops
  the decode while preserving the already-recovered trailing EVENTs. A
  new `TextureParseError::RvlcEscapeOpenerMissing { window }` flags a
  malformed backward opener. `reverse_bits(code, len)` bit-reverses a
  forward codeword for the reverse match. 13 new tests: backward-reader
  tail-first emission / forward-field reassembly / start-bound, the
  `reverse_bits` roundtrip, forward↔backward equality on common-path and
  escape sequences (single + multi-EVENT, intra + inter column), the
  longest 15-bit code, the Annex-E error-recovery tail-preservation, and
  first-EVENT error propagation. (843 lib tests, +12.)
- Round 47 of the clean-room rebuild: the §7.4.1.2 reversible-VLC Tcoef
  decode for the `reversible_vlc == 1` path. A new `rvlc_tables.rs`
  transcribes Table B.23 — the 169 reversible EVENTs, each carrying both
  its intra `(LAST, RUN, LEVEL)` and inter `(LAST, RUN, LEVEL)` triple
  under one prefix-free code (selected by the existing `TcoefTable`
  argument). `decode_ac_event_rvlc` decodes one EVENT: the common path
  is a Table B.23 code plus its trailing sign bit `s`; combinations not
  in the table use the §7.4.1.3 Type-5 escape (the only escape mode
  permitted when `reversible_vlc == 1`) — opener `00001`, fixed-length
  `LAST(1) RUN(6) marker LEVEL(11) marker`, closing delimiter `0000`,
  sign bit `s`, with the 11-bit unsigned LEVEL magnitude per Table B.25
  (`0` forbidden) and the 6-bit RUN per Table B.24.
  `decode_ac_events_rvlc` runs the §6.2.7 `while (!last)` loop. A new
  `TextureParseError::RvlcEscapeDelimiterMissing { window }` flags a
  malformed closing delimiter; the existing `EscapeMarkerBitMissing` and
  `ReservedEscapeLevel` cover the marker bits and the forbidden LEVEL.
  The escape opener `00001` is prefix-disjoint from every Table B.23
  code (none begins with `0000`), so the common and escape paths are
  distinguished by the leading five bits. 13 new tests (prefix-freeness
  + 169-entry count, intra/inter column divergence, longest 16-bit code,
  Type-5 positive/negative escapes, forbidden-LEVEL / marker / delimiter
  / invalid-code rejections, event loop). (831 lib tests, +13.)
- Round 46 of the clean-room rebuild: §7.7.2.1 / Figure 7-46 / Figure
  7-47 field-aware spatial neighbour selection, closing the round-45
  follow-up. A new `MbMv::Field { top, bottom }` records a
  field-predicted neighbour's two field motion vectors;
  `MvGrid::record_field` / `MbMvRecord::field` set it.
  `MvGrid::field_predictor_candidates(mb_row, mb_col)` (and the free
  `gather_field_mv_predictor_candidates`) resolve the three Figure 7-46 /
  7-47 macroblock-level positions (`MV1` left, `MV2` above, `MV3`
  above-right — coinciding with the §7.6.5 frame-mode top-left-block
  layout) into a `[FieldPredCandidate; 3]` ready for round 45's
  `predict_field_motion_vector`. A frame-predicted neighbour (`OneMv`, or
  — per the §7.7.2.1 "8x8 block motion vector closest to the upper left
  block of the current MB" rule — the selected sub-block of a `FourMv`)
  → `FieldPredCandidate::Frame`; a field-predicted neighbour →
  `FieldPredCandidate::Field`; an absent / transparent / out-of-VOP
  neighbour → `FieldPredCandidate::Invalid`. The existing frame-mode
  `MvGrid::predictor_candidates` query now collapses a `Field` neighbour
  to its `Div2Round(MVf1 + MVf2)` frame candidate (§7.7.2.1 CASE 2 /
  Figure 7-47, also the §7.6.6 OBMC adjacent-field-MB rule). With this
  the interlaced P-VOP field-prediction path is end-to-end (parse rounds
  39–42, predictor selection round 45 + gathering this round,
  reconstruction round 43, half-/quarter-sample field MC rounds 43/44).
  9 new tests. (818 lib tests, +9; 12 doc, +1.)
- Round 45 of the clean-room rebuild: §7.7.2.1 field-MV predictor
  selection (CASE 1 / CASE 2 / CASE 3), closing the round-43/44
  follow-up. New `FieldPredCandidate` enum (`Frame` /
  `Field { top, bottom }` / `Invalid`) classifies each §7.6.5 spatial
  neighbour, and `predict_field_motion_vector([FieldPredCandidate; 3])`
  maps a field-predicted neighbour to its per-component
  `Div2Round(MVf1 + MVf2)` average (Figure 7-47), a frame-predicted
  neighbour to its frame MV, applies the four §7.6.5 validity rules,
  and medians component-wise to the shared `(Px, Py)` predictor. CASE 1
  reduces to the progressive median; CASE 2 feeds `MVx = MVDx + Px` /
  `MVy = MVDy + Py`; CASE 3 feeds the shared predictor into
  `reconstruct_field_motion_vectors`. 6 new tests. (809 lib tests.)
- Round 44 of the clean-room rebuild: §7.6.2.2 quarter-sample field
  motion compensation, closing the round-43 follow-up. A new
  `FieldRefView` (`src/quarter_sample.rs`) presents one field of the
  progressive reference plane as a contiguous line grid (field-line `n`
  → frame line `field_y0 + 2n`, §7.6.4 clamp applied in field-line
  space), realising the §7.6.2 interlaced rule "vertically interpolated
  between two successive lines of the same field … the vertical
  coordinates of the integer samples differ by 2". The §7.6.2.2.1/.2
  8-tap-FIR + bilinear quarter-pel cascade is now generic over a
  `QpelSource` trait — the same math runs on the frame plane or a
  `FieldRefView` with no duplication, and the public `half_pel_b/c/d`,
  `interpolate_quarter_pixel`, and `interpolate_block_qpel*` signatures
  are unchanged. `field_mvy_to_field_grid(mvy)` halves the always-even
  (§7.7.2.1) frame-coordinate field MVy into a field-grid quarter-pel
  coordinate. `interpolate_block_qpel_field[_into](…, ref_field_y0, …)`
  interpolates a 16×8 luma field block.
  `field_motion_compensate_one_reference_qpel(…, bits_per_pixel)`
  (`src/field_motion.rs`) replaces the two luma `mc` calls of the
  half-sample driver with quarter-pel field-block interpolation (top
  output field → even rows, bottom → odd) and keeps the four §7.7.2.1
  chroma `mc` calls in half-sample field mode; per §7.7.2.2 the
  quarter-sample chroma MV is
  `div2_round(half_pel_chroma_mv_from_qpel(c))` (`Div2Round` of the
  luma qpel component divided by 2, `/` truncating toward 0). 17 tests
  pin the path. (803 lib tests, +17.)

- Round 43 of the clean-room rebuild: §7.7.2.1 field-MV reconstruction
  + the field motion-compensation driver (`src/field_motion.rs`), so an
  interlaced P-VOP field-predicted macroblock reconstructs pixels.
  `reconstruct_field_motion_vectors(pair, px, py)` turns a decoded
  `FieldMvPair` differential + the shared `(Px, Py)` predictor into the
  two final field MVs per `MVx fi = MVDx fi + Px`,
  `MVy fi = 2 * (MVDy fi + (Py / 2))` (vertical component always even in
  half-pel frame coordinates). `mc(...)` reproduces the §7.7.2.1 /
  §7.6.2 half-sample reference routine verbatim with the `pred_y0` /
  `ref_y0` / `y_incr` field parameters — `y_incr = 2` writes only every
  other destination line and averages the two adjacent same-field
  reference lines (`y_ref`, `y_ref + 2`). `div2_round(x) =
  (x >> 1) | (x & 1)` derives the chrominance field MVs.
  `field_motion_compensate_one_reference(...)` issues the six §7.7.2.1
  `mc` calls (top/bottom luma, then top/bottom Cb / Cr) and assembles an
  `InterPredictionMacroblock` ready for the §7.3 residual add. The
  output field is selected by `pred_y0`; the reference field by the
  `forward_top_field_reference` / `forward_bottom_field_reference`
  flags through `ref_y0`. 12 tests pin the path (Div2Round bit
  definition, even-vertical reconstruction, truncation-toward-0 of
  `Py / 2`, frame/field `mc` equivalence, same-field vertical half-pel
  averaging, independent top/bottom reference-field selection, and the
  end-to-end field-MC + §7.3 residual-add reconstruction with display
  clip). Half-sample mode only; quarter-sample field MC is a follow-up.

- Round 42 of the clean-room rebuild: §6.2.6 B-VOP motion-vector
  bodies + interlaced field-prediction second `motion_vector()`
  invocations. `decode_field_motion_vector_pair` decodes the
  unconditional body plus the `if (interlaced && field_prediction)`
  second one into `FieldMvPair { top, bottom }` (§7.7.2.1 ordering:
  top-field differential first; both bodies share the direction's
  fcode). `decode_p_macroblock_motion_vectors_interlaced` extends the
  round-37 P-VOP walker with the field route
  (`PMbMotionVectors::{Frame, Field}`); the new
  `MotionParseError::InvalidFieldPredictionContext` rejects
  `field_prediction` combinations §6.2.6.3 cannot code (inter4v /
  intra rows, direct mode) without consuming bits.
  `decode_b_vop_mb_motion_vectors` walks the §6.2.6 Table B.4 branch
  after `parse_b_vop_mb_header`: forward body for `mb_type == '01' ||
  '0001'`, backward for `'01' || '001'` (syntax order, backward gated
  on `vop_fcode_backward`), single residual-less
  `motion_vector("direct")` for `'1'`; each direction surfaces
  `BVopMvBody::Frame` or `::Field`, with field-interpolated
  macroblocks decoding four bodies in the §7.7.2.2 / Table 7-14
  `MVD[0..3]` order (forward-top, forward-bottom, backward-top,
  backward-bottom). New `BVopMbParseError::Motion` wraps body-level
  failures. Reconstruction (`MVy fi = 2 * (MVDy fi + (Py / 2))`) and
  §7.7.2 field motion compensation stay later-round work. 17 new unit
  tests; total crate test count now 774 + 9 doc.
- Round 41 of the clean-room rebuild: §6.2.6 B-VOP `if (interlaced)
  interlaced_information()` dispatch wiring. `parse_b_vop_mb_header`
  now consumes the §6.2.6.3 body (round 39's
  `parse_interlaced_information`) immediately after the (optional)
  `dbquant` and before the (later-round) motion-vector bodies,
  completing the dispatch work round 40 left open. The §6.2.6 B-VOP
  layer places the call inside two enclosing gates the I/P-VOP layer
  does not have: the `if (modb != '1')` subtree (a `modb == '1'`
  macroblock skips everything after `modb`, including the body) and
  the `if (ref_select_code != '00' || !scalability)` branch — the
  scalable enhancement-layer path (Table B.5, `BMbTypeTable::B5`)
  carries no `interlaced_information()` line in §6.2.6, so an
  interlaced VOL on that path still yields `None`. The gate is driven
  by `mb_type_present` rather than the raw `modb` value (the Table
  B.3 raw codes `1` and `01` are numerically identical as integers).
  The §6.2.6.3 first gate's `cbp != 0` predicate is `cbpb != 0`, with
  an absent `cbpb` (`modb == '01'`) collapsing to `cbp == 0`. The
  dispatch fires for Direct macroblocks too (the §6.2.6 line is
  unconditional within the Table B.4 branch); §6.2.6.3 then
  suppresses `field_prediction` via its `mb_type != "1"` clause, so a
  Direct MB carries at most the `dct_type` bit. New field
  `BVopMbHeader::interlaced_info: Option<InterlacedInformation>`
  surfaces the decoded body via the round-39
  `InterlacedInfoContext::b_vop` context. Out of scope (later
  rounds): the B-VOP motion-vector bodies and their `if (interlaced
  && field_prediction) motion_vector(…)` second invocations. With
  this, every macroblock-header parser in the crate (I-VOP / P-VOP /
  B-VOP) routes the §6.2.6 interlaced dispatch. 9 new unit tests:
  progressive-VOL no-body, `modb == '1'` skip, Direct `cbpb != 0`
  dct-only, Direct `cbpb == 0` zero-bit body, Interpolated full
  6-bit body after `dbquant`, Forward `modb == '01'` 1-bit
  `field_prediction == 0` body, Backward backward-pair-only body,
  Table B.5 no-dispatch, and a mid-body truncation — each with a
  sentinel bit-position check. Total crate test count now 757 + 9
  doc.
- Round 40 of the clean-room rebuild: §6.2.6 `if (interlaced)
  interlaced_information()` dispatch wiring. `parse_macroblock_header`
  now consumes the §6.2.6.3 body (round 39's
  `parse_interlaced_information`) for I- and P-VOP macroblocks when the
  VOL header carries `interlaced == 1`. The call fires immediately
  after `dquant` and before the (later-round) motion / texture data,
  per the §6.2.6 syntax line `if (interlaced)
  interlaced_information()`. The §6.2.6.3 first gate's `cbp != 0`
  predicate is derived from the macroblock header's coded-block
  pattern (`cbpy != 0 || cbpc != 0`, both already in the §6.3.7
  "1 == coded" convention — the inter `cbpy` column of Table B.8 is
  stored in coded-block-pattern form). A new
  `MacroblockHeader::interlaced_info: Option<InterlacedInformation>`
  field surfaces the decoded body: `None` for progressive VOLs and for
  `not_coded` macroblocks (the §6.2.6 not-coded short-circuit returns
  before the `interlaced_information()` call), `Some(_)` for every
  coded macroblock in an interlaced VOL (possibly a zero-bit body when
  neither §6.2.6.3 gate fires). The I-VOP context is built via the
  round-39 checked `InterlacedInfoContext::i_vop` (always carries the
  `dct_type` gate, never `field_prediction`); the P-VOP context via
  `InterlacedInfoContext::p_vop`. A defensive
  `MacroblockParseError::InvalidInterlacedContext` guards an
  inconsistent I-VOP context construction. The S(GMC)-VOP `mcsel`
  route and the §6.2.6 `if (interlaced && field_prediction)
  motion_vector("forward")` second invocation remain out of scope (the
  header parser rejects S-VOPs and stops before motion vectors). 8 new
  unit tests: progressive-VOL no-body, I-VOP dct_type-only, P-VOP
  inter `cbp == 0` field_prediction bit, P-VOP inter `cbp != 0` full
  forward-pair body, P-VOP inter4v zero-bit body with bit-position
  check, P-VOP intra dct_type-only, not_coded skip, and an interlaced
  body truncation that surfaces `Truncated`. Total crate test count now
  748 + 9 doc.
- Round 39 of the clean-room rebuild: §6.2.6.3
  `interlaced_information()` body parser. New
  `src/interlaced_information.rs` decodes the two §6.2.6.3 gates —
  the optional `dct_type` bit (fires when `derived_mb_type ∈ {3, 4}`
  or `cbp != 0`) and the optional `field_prediction` block (fires
  under one of the three §6.2.6.3 disjuncts: P-VOP with
  `derived_mb_type < 2`, S(GMC)-VOP with `derived_mb_type < 2` and
  `!mcsel`, or B-VOP with `mb_type != "1"`). When `field_prediction
  == 1`, up to four reference bits follow — the forward pair
  (`forward_top_field_reference` + `forward_bottom_field_reference`,
  present for P / S / B-non-backward) and the backward pair
  (`backward_top_field_reference` + `backward_bottom_field_reference`,
  B-non-forward only). `parse_interlaced_information(br, ctx)`
  consumes 0..=6 bits depending on which gates fire and never emits
  a flag whose syntax-level guard is not satisfied.
  `InterlacedInfoContext::{i_vop, p_vop, b_vop, s_gmc_vop}` are
  checked constructors that encode the spec's coding-type-specific
  preconditions structurally: I-VOP refuses inter `mb_type`,
  S(GMC)-VOP refuses `derived_mb_type >= 2`, and `s_gmc_vop` with
  `McSel::On` surfaces `None` (the §6.2.6.3 S-disjunct's `!mcsel`
  requirement). `DctType::{Frame, Field}` and `FieldReference::{Top,
  Bottom}` carry the §6.3.6.3 semantics directly.
  `InterlacedInformation::field_prediction_guard_fired` discriminates
  "second gate fired but bit value was 0" from "gate didn't fire" so
  callers know whether to skip the §6.2.6 `if (interlaced &&
  field_prediction) motion_vector("forward")` second invocation.
  `dct_type_present` and `field_prediction_present` are pure
  predicates that surface the gates without parsing. Out of scope
  (later rounds): the §6.2.6 outer `if (interlaced)
  interlaced_information()` dispatch from the macroblock-header
  parsers, and the §7.6.5 / §7.6.2.5 consumption of `dct_type` and
  `field_prediction` by the motion-compensation / block-grouping
  paths. 28 new unit tests covering all 24 reachable
  `InterlacedInfoContext` permutations, the four B-VOP `BVopMbType`
  rows × reference-pair presence rules, an end-to-end Interpolated
  B-VOP roundtrip (6 bits — dct + fp + forward pair + backward pair),
  partial-presence Forward / Backward B-VOP roundtrips (3 bits each),
  a P-VOP Inter MB roundtrip (4 bits), a zero-bit Inter4V path, the
  S(GMC) `mcsel == 1` `None`-yielding constructor, a truncated-input
  `EndOfStream` error, and a sweep that verifies
  `InterlacedInformation::bit_count()` matches
  `BitReader::bit_position()` across every reachable context under
  a worst-case payload; total crate test count now 740 + 9 doc.
- Round 38 of the clean-room rebuild: §6.2.6 binary-shape
  `transparent_block(j)` elision for the four-MV inter4v branch. The
  §6.2.6 P-VOP macroblock-layer text spells the `derived_mb_type == 2`
  branch as `for (j=0; j<4; j++) if (!transparent_block(j))
  motion_vector("forward")` — when the §6.1.3.4 binary shape leaves an
  8x8 sub-block fully transparent, that sub-block's MV body is omitted
  from the bitstream entirely. Round 37 handled the rectangular case
  (§6.1.3.4 NOTE 2 — every sub-block opaque, four
  `motion_vector("forward")` invocations); round 38 adds
  `decode_p_macroblock_motion_vectors_with_shape(br, derived_mb_type,
  vop_fcode_forward, BinaryShapeBlockOpacity)` so binary-shape VOPs
  elide transparent sub-block MV bodies without consuming bits for
  them. `BinaryShapeBlockOpacity { opaque: [bool; 4] }` encodes the
  §5.2.7 `transparent_block(j)` **negation** per sub-block in Figure
  6-8 raster order (`0 = TL`, `1 = TR`, `2 = BL`, `3 = BR`), with an
  `ALL_OPAQUE` const for the rectangular-shape default and a
  `motion_vector_invocation_count` helper reporting the expected
  number of `motion_vector("forward")` invocations under the mask.
  `BinaryShapeFourMv { deltas: [Option<MotionVectorDelta>; 4] }`
  surfaces the decoded result with `None` for elided slots;
  `iter_present()` walks only the populated `(j, delta)` pairs and
  `to_motion_coding_deltas()` lifts the all-opaque case into the
  existing round-37 `MotionCodingDeltas::FourMv` view so the
  rectangular-shape predictor chain (§7.6.5 + §7.6.3) is unchanged.
  Single-MV / intra `derived_mb_type` values surface `Ok(None)`
  without consuming bits — the §6.2.6 `transparent_block(j)` guard
  only fires inside `if (derived_mb_type == 2)`. Out of scope (later
  rounds): the §6.1.3.4 binary-shape decoder that produces the
  per-sub-block opacity flags, the interlaced `field_prediction`
  second-invocation gate, and the S(GMC)-VOP `mcsel == 1`
  sprite-warping route. 5 new unit tests: all-opaque equivalence with
  round 37's driver (same bit consumption + same per-slot deltas);
  partial-elision case (TL + BL opaque, TR + BR transparent — exactly
  16 bits consumed, two MV pairs at fcode 1, lift-to-FourMv refuses);
  all-transparent zero-bit case (no slot populated); non-Inter4V
  routing guard (Inter / InterQ / Intra / IntraQ all surface `None`
  without consuming bits); `motion_vector_invocation_count` matches
  the populated-flag count across four-mask permutations.
- Round 37 of the clean-room rebuild: §6.2.5
  `motion_coding(mode, type_of_mb)` driver + §6.2.6 P-VOP
  macroblock-level MV-body walker. The §6.2.5 syntax wraps one or four
  invocations of the §6.2.6.2 `motion_vector(mode)` body: the
  unconditional opening call plus a `if (type_of_mb == 2) for (i = 0;
  i < 3; i++) motion_vector(mode)` loop for the `inter4v`-cardinality
  case. `motion_coding(br, mode, type_of_mb, vop_fcode)` decodes the
  body list against the round-7 `decode_motion_vector_delta`
  per-invocation primitive, returning
  `MotionCodingDeltas::{OneMv(MotionVectorDelta), FourMv([MotionVectorDelta; 4])}`.
  `TypeOfMb::{One, Four}` encodes the §6.2.5 `type_of_mb` integer. The
  four `FourMv` slots map to Figure 6-8 raster order (`0 = TL`,
  `1 = TR`, `2 = BL`, `3 = BR`) — the same numbering round 30's
  `MvGrid::FourMv` consumes, so the round-37 output of `motion_coding`
  feeds directly into round-30's `MvGrid::record` without
  re-permutation. `decode_p_macroblock_motion_vectors(br,
  derived_mb_type, vop_fcode_forward)` is the §6.2.6 P-VOP
  macroblock-level MV-body driver: it dispatches on `derived_mb_type`
  per the §6.2.6 syntax — `Inter` / `InterQ` (`derived_mb_type == 0
  || 1`) → `motion_coding("forward", TypeOfMb::One)`; `Inter4V`
  (`derived_mb_type == 2`) → `motion_coding("forward",
  TypeOfMb::Four)`; `Intra` / `IntraQ` (`derived_mb_type == 3 || 4`)
  → no MV body, returns `Ok(None)` without consuming bits. The
  §6.2.6 gates `(derived_mb_type == 0 || derived_mb_type == 1)` and
  `(derived_mb_type == 2)` both exclude the intra branches; the
  caller skips straight to the `for (i = 0; i < block_count; i++)
  block(i)` loop on the `Ok(None)` return. The §7.6.5 predictor add
  stays at the caller layer: each decoded `MotionVectorDelta` pairs
  with its block-position-specific Figure 7-34 predictor via the
  existing round-7 `reconstruct_motion_vector` + round-8
  `predict_motion_vector` + round-30
  `MvGrid::predictor_candidates` chain. The interlaced
  `field_prediction` second-invocation, the S(GMC)-VOP `mcsel == 1`
  sprite-warping route, and the binary-shape `transparent_block(j)`
  elision are intentionally out of scope — rectangular shape
  (§6.1.3.4 NOTE 2) guarantees every 8x8 sub-block is opaque so the
  §6.2.6 transparency guard always fires. 12 new unit tests including
  an Inter / InterQ / Inter4V / Intra / IntraQ exhaustive cross-check
  and an end-to-end `motion_coding → reconstruct_motion_vector`
  composition test that validates the §6.2.6.2 + §7.6.3 + §6.2.5 call
  chain together; total crate test count now 707 + 9 doc.
- Round 36 of the clean-room rebuild: §7.6.1.5 padding of interlaced
  macroblocks — luminance boundary path. New module
  `interlaced_padding`. §7.6.1.5 says verbatim "Macroblocks of
  interlaced VOP (interlaced = 1) are padded according to subclauses
  7.6.1.1 through 7.6.1.3. The vertical padding of the luminance
  component, however, is performed for each field independently. A
  sample outside of a VOP is therefore filled with the value of the
  nearest boundary sample of the same field." The carve-out names only
  the vertical pass, so the §7.6.1.1 horizontal pass keeps its
  frame-mode behaviour and the §7.6.1.2 vertical pass runs per field.
  `interlaced_boundary_padding_luma(decoded, shape)` is the
  end-to-end §7.6.1.5 boundary entry point for a 16×16 luma
  macroblock: §7.6.1.1 frame-mode horizontal pass, then per-field
  §7.6.1.2 vertical pass on the top + bottom field views (rows
  `0, 2, …, 14` / `1, 3, …, 15`), then re-interleave back into a
  16×16 frame. `per_field_vertical_padding_luma(hor_pad, s_prime)`
  exposes the per-field §7.6.1.2 step as a standalone primitive for
  callers that have already run §7.6.1.1 themselves. The returned
  `InterlacedBoundaryOutcome { Padded { top_column_states,
  bottom_column_states }, CompletelyTransparent }` discriminates the
  three cases the §7.6.1.5 routing cares about: at least one field
  filled (each `ColumnState` array reports which per-field columns
  completed); every row fully transparent (the caller routes the
  macroblock to the `2 ^ (bits_per_pixel - 1)` fill via the §7.6.1.3
  `extended_padding` entry points). `LUMA_FIELD_LINES = LUMA_SIDE / 2`
  is the per-field row count (8 of the macroblock's 16 luma rows per
  field). The §7.6.1.5 chrominance "based on fields" carve-out and
  the exterior-MB per-field §7.6.1.3 path are intentionally out of
  scope for this round (the §6.1.3.7.1 +
  `decimate_chroma_shape_interlaced_field` infrastructure shipped in
  round 35 already gives the chroma wrapper its shape-decimation
  prerequisite, and §7.6.1.3 itself is unchanged in the frame-mode
  sense — the carve-out only replaces the §7.6.1.3 mid-grey fallback
  by the `2 ^ (bits_per_pixel - 1)` fill for completely-transparent
  macroblocks, which is the §7.6.1.3 mid-grey case verbatim). 15 new
  unit tests including a cross-check against a manually-composed
  §7.6.1.1 + per-field §7.6.1.2 reassembly and an explicit
  progressive-vs-interlaced divergence on a macroblock whose two
  fields differ in their nearest-neighbour candidates; total crate
  test count now 695 + 9 doc.
- Round 35 of the clean-room rebuild: §6.1.3.7.1 binary-shape decimation
  driving the §7.6.1.4 chrominance-padding shape mask. New module
  `chroma_shape`. §7.6.1.4 says verbatim "Chrominance components are
  padded according to subclauses 7.6.1.1 through 7.6.1.3 for each 8×8
  block. The padding is performed by referring to a shape block
  generated by decimating the shape block of the corresponding
  luminance component. This decimating of the shape block is performed
  by the subsampling process described in subclause 6.1.3.6." The
  3rd-edition section the §7.6.1.4 pointer refers to is §6.1.3.7.1
  ("4:2:0 Format"), which carries the actual rule: "For each 2×2 block
  of the binary alpha plane associated with the luminance plane of the
  bounding rectangle (of the same frame for the progressive and of the
  same field for the interlaced case), the associated pixel value of
  the binary alpha plane associated with the chrominance planes is set
  to 255 if any pixel of said 2×2 block of the binary alpha plane
  associated with the luminance plane equals 255" — a logical OR over
  each 2×2 luma block.
  `decimate_chroma_shape_sample(block: [SamplePresence; 4])` is the
  single-sample primitive (returns `Opaque` iff any of the four input
  samples is opaque). `decimate_chroma_shape::<M, TWO_M>(luma)` is the
  const-generic kernel that walks the `(2M) × (2M)` luma shape and
  emits an `M × M` chroma shape via per-2×2 OR.
  `decimate_chroma_shape_progressive(luma)` is the §7.6.1.4 frame-mode
  entry point pinned to luma 16×16 → chroma 8×8; the output is the
  shape block the §7.6.1.4 padding pipeline routes into
  `horizontal_repetitive_padding_chroma` →
  `vertical_repetitive_padding_chroma` → `extended_padding_chroma`.
  `decimate_chroma_shape_interlaced_field(field)` handles the
  §6.1.3.7.1 interlaced "of the same field" rule — one field's 8×16
  luma shape decimates to a 4×8 chroma field. `CHROMA_FIELD_LINES`
  (= 4) is the per-field chroma row count.
  `stack_interlaced_chroma_shape(top, bottom)` re-interleaves the two
  per-field chroma shape blocks back into a single 8×8 frame block
  (top[0] → row 0, bottom[0] → row 1, …). `split_luma_shape_into_fields(
  luma_frame)` is the matching helper that pulls the two per-field 8×16
  luma blocks out of a 16×16 frame luma shape. The §7.6.1.4 chroma
  padding pipeline now has the shape input it needs; only the §7.6.1.5
  interlaced per-field §7.6.1.1/§7.6.1.2 wrapper remains for the
  §7.6.1 boundary-padding pipeline. 19 new unit tests including the
  spec-rule cross-check over every 2⁴ = 16 mask of a single 2×2 luma
  block; total crate test count now 680 + 9 doc.

- Round 34 of the clean-room rebuild: §7.6.1.3 extended padding — the
  third (and final) pass of the §7.6.1 reference-VOP padding pipeline.
  New module `extended_padding`. After §7.6.1.1 + §7.6.1.2 have left
  every boundary macroblock fully opaque, the §7.6.1.3 pass fills the
  remaining *exterior* macroblocks (every `s[y][x] == 0`) by
  replicating the border row / column of a neighbouring boundary
  macroblock, falling through to the `2^(bits_per_pixel - 1)` mid-grey
  fill when no side-adjacent boundary neighbour exists.
  `extended_padding_macroblock::<N>(neighbours, bits_per_pixel)`
  consumes a `BoundaryNeighbours<N>` (the four optional side-adjacent
  post-§7.6.1.2 boundary macroblocks: `left` / `above` / `right` /
  `below`) and the channel's `bits_per_pixel`, picks the highest-
  priority present neighbour per Figure 7-28 (`3 > 2 > 1 > 0` — left,
  above, right, below in that order), and replicates the matching
  border into the exterior MB: left → rightmost column rightwards;
  above → bottom row downwards; right → leftmost column leftwards;
  below → top row upwards. With no neighbour present, every sample is
  set to `mid_grey_value(bits_per_pixel) = 2^(bits_per_pixel - 1)`
  (128 for the canonical 8-bit case, 512 for 10-bit).
  `extended_padding_luma(neighbours, bits_per_pixel)` /
  `extended_padding_chroma(neighbours, bits_per_pixel)` are the 16×16
  / 8×8 macroblock-level entry points. New public types:
  `ExteriorNeighbourPosition ∈ {Left, Above, Right, Below}` (named to
  not collide with the existing intra-DC `predictor::NeighbourPosition`),
  `BoundaryNeighbours<'a, N>` (the four optional side-adjacent
  post-§7.6.1.2 neighbour grids), `ExteriorPaddingOutcome ∈
  {FromNeighbour(ExteriorNeighbourPosition), MidGrey}` (per-MB outcome
  reporting which §7.6.1.3 branch fired). `BoundaryNeighbours::none()`
  builds an all-absent neighbour set in const context;
  `BoundaryNeighbours::highest_priority_position()` returns the
  Figure 7-28 winner; `ExteriorNeighbourPosition::priority()` exposes
  the numeric priority (`3..=0`) so callers building a custom selector
  don't have to duplicate the table. The §7.6.1.4 chroma shape
  decimation (§6.1.3.6) and the §7.6.1.5 interlaced per-field
  application remain caller-level routing concerns; §7.6.1.3 itself
  is field-independent. 21 new unit tests; total crate test count now
  661 + 9 doc.

- Round 33 of the clean-room rebuild: §7.6.1.2 vertical repetitive
  padding — the second pass of the §7.6.1 reference-VOP padding
  pipeline. New module `vertical_padding` consuming the
  `(hor_pad, s_prime, row_states)` triple produced by §7.6.1.1 and
  filling the remaining transparent samples column-by-column.
  `vertical_repetitive_padding_column::<M>(hor_pad, s_prime, out,
  s_double_prime)` applies the spec's verbatim per-column procedure:
  positions where `s'[y][x] == 1` map straight through; transparent
  positions look up for `y'` (nearest `s'==1` at-or-above `y`) and
  down for `y''` (nearest `s'==1` strictly below `y`), filling with
  `(hor_pad[y'] + hor_pad[y'']) // 2` (§3.4 truncation toward zero
  via `i32::div_euclid`) when both exist, or with the single
  available neighbour when only one exists. Columns with no
  `s'==1` sample fall through to the column-guard and report
  `ColumnState::FullyTransparent` so the caller can route the
  macroblock to §7.6.1.3 extended padding.
  `vertical_repetitive_padding_luma(hor_pad, s_prime, row_states)` /
  `vertical_repetitive_padding_chroma(hor_pad, s_prime, row_states)`
  are the 16×16 / 8×8 macroblock-level entry points that loop the
  per-column routine over the macroblock side and return the
  `(hv_pad, s_double_prime, column_states)` triple. New public type:
  `ColumnState ∈ {FullyFilled, FullyTransparent}` (the per-column
  §7.6.1.2 outcome). The §7.6.1.2 fill sentinel `s''[y][x]` surfaces
  directly via the `s_double_prime` output. The §7.6.1.3 extended
  padding for exterior macroblocks, §7.6.1.4 chroma shape decimation,
  and §7.6.1.5 interlaced per-field padding remain later-round work.
  17 new unit tests; total crate test count now 640 + 9 doc.

- Round 32 of the clean-room rebuild: §7.6.1.1 horizontal repetitive
  padding — the first pass of the §7.6.1 reference-VOP padding
  pipeline. New module `sample_padding` filling the transparent
  samples of a boundary macroblock by replicating the VOP-boundary
  samples of the same row.
  `horizontal_repetitive_padding_row::<N>(decoded, shape, out,
  s_prime)` applies the spec's verbatim per-row procedure to any row
  size: for each transparent sample, look left for `x'` (nearest
  opaque at-or-before `x`) and right for `x''` (nearest opaque
  strictly after `x`); both exist → fill with `(d[x'] + d[x'']) // 2`
  (§3.4 truncation toward zero via `i32::div_euclid`); only one side
  exists → replicate that boundary sample; neither exists → the
  row-guard reports `ShapeRowState::FullyTransparent` for the
  §7.6.1.2 vertical pass to handle later.
  `horizontal_repetitive_padding_luma(decoded, shape)` /
  `horizontal_repetitive_padding_chroma(decoded, shape)` are the 16×16
  / 8×8 macroblock-level entry points (matching the §6.1.3.4 luma side
  and the §7.6.1.4 chroma side) that return the
  `(hor_pad, s_prime, row_states)` triple. New public types:
  `SamplePresence ∈ {Opaque, Transparent}` (the per-sample `s[y][x]`
  flag) and `ShapeRowState ∈ {FullyFilled, FullyTransparent}` (the
  per-row §7.6.1.1 row-guard outcome). Public constants `LUMA_SIDE =
  16` and `CHROMA_SIDE = 8`. The §7.6.1.1 fill sentinel `s'[y][x]`
  (initialised to 0 then flipped to 1 on any fill) surfaces directly
  via the `s_prime` output so the §7.6.1.2 vertical pass can pick up
  per-sample row state. The §7.6.1.2 vertical pass, §7.6.1.3 extended
  padding for exterior macroblocks, §7.6.1.4 chroma shape decimation,
  and §7.6.1.5 interlaced per-field padding remain later-round work.
  15 new unit tests; total crate test count now 623 + 9 doc.

- Round 31 of the clean-room rebuild: §7.3 VOP reconstruction — the
  per-pixel `d[y][x] = p[y][x] + f[y][x]` step-2 sum plus the step-3
  `[0, 2^bits_per_pixel - 1]` display-range saturation that closes the
  decoder pipeline. New module `reconstruct`. `clip_display_sample(
  value, bits_per_pixel)` is the per-sample §7.3 step-3 primitive (the
  three-branch clip "`2^bpp - 1` when `d > 2^bpp - 1`, `d` when
  `0 <= d <= 2^bpp - 1`, `0` when `d < 0`").
  `reconstruct_inter_block_8x8(prediction, residual, bits_per_pixel)`
  (plus the `_into` buffer-out variant) applies §7.3 step-2 + step-3 to
  one 8×8 inter block — `out[y][x] = clip(prediction[y][x] +
  residual[y][x], 0, 2^bpp - 1)`. `reconstruct_intra_block_8x8(sample,
  bits_per_pixel)` covers the §7.3 step-1 intra branch (`d = f`, then
  the §7.3 step-3 clip). `reconstruct_inter_macroblock(prediction,
  residual, bits_per_pixel)` and `reconstruct_intra_macroblock(sample,
  bits_per_pixel)` (plus the `_into` variant for the inter path)
  operate at the 4:2:0 macroblock granularity, consuming the existing
  16×16-luma + 8×8-chroma `InterMacroblock` / `IntraMacroblock`
  residual / sample shapes produced by `decode_inter_macroblock` /
  `decode_intra_macroblock` and the new `InterPredictionMacroblock`
  shape that holds the `p[y][x]` plane outputs from the §7.6
  half-sample / quarter-sample / B-VOP prediction modules.
  `InterPredictionMacroblock::zero()` constructs the all-zero
  prediction macroblock for the §7.6.9.6 / boundary-substitution
  fallback case. `ReconstructedMacroblock` carries the `d[y][x]`
  output planes in the §7.3 step-3 display range `[0, 2^bpp - 1]`,
  ready to be blitted into the VOP frame buffer. The luma / Cb / Cr
  planes are processed independently — a §7.3 step-3 clip on one
  plane does not affect the others. The §6.3.3 `not_8_bit` /
  `bits_per_pixel != 8` path is honoured by every entry point. Public
  constants `BLOCK_SIDE = 8`, `MACROBLOCK_LUMA_SIDE = 16`,
  `MACROBLOCK_CHROMA_SIDE = 8`. 21 new unit tests; total crate test
  count now 608 + 9 doc.
- Round 30 of the clean-room rebuild: §7.6.5 / Figure 7-34 spatial
  motion-vector predictor candidate gathering. New module
  `mv_predictor_grid` providing `MvGrid::new(mb_rows, mb_cols)` and
  per-macroblock `record_one_mv` / `record_four_mv` / `record_absent`
  setters (plus generic `record(mb_row, mb_col, MbMvRecord)`). The
  `predictor_candidates(mb_row, mb_col, block_index)` query (also
  available as the free function `gather_mv_predictor_candidates`)
  resolves the three Figure 7-34 spatial positions for the current
  8×8 luminance block (Figure 6-8 numbering: `0 = TL`, `1 = TR`,
  `2 = BL`, `3 = BR`) into a `[Option<MotionVector>; 3]` triple ready
  to feed directly into `predict_motion_vector`. New public types:
  `MbMv ∈ {Absent, OneMv(MotionVector), FourMv([MotionVector; 4])}`,
  `MbMvRecord { content, transparent: [bool; 4] }`, `MvGrid`,
  `MvGridError`. The §7.6.5 boundary-substitution rule for
  neighbours outside the current VOP / video packet / GOB is handled
  by `MvGrid::record_absent` on the boundary MBs. Per-luma-block
  transparency within an otherwise-opaque macroblock is handled by
  the four-element `MbMvRecord::transparent` mask. The four
  block-position cases follow the in-repo ASCII transcription of
  Figure 7-34 in
  `docs/video/mpeg4-visual/figure-7-34-mv-predictor-layout.md`. 22
  new unit tests + 1 doctest; total crate test count now 587 + 9 doc.
- Round 29 of the clean-room rebuild: §7.6.9.5.3 second-paragraph +
  §7.6.9.4 chrominance motion-compensation plane for B-VOPs.
  `generate_b_vop_chroma_prediction(forward_chroma_ref,
  backward_chroma_ref, forward_chroma_mv, backward_chroma_mv,
  chroma_mb_origin_x, chroma_mb_origin_y, vop_rounding_type,
  prediction_mode)` (plus the `_into` buffer-out variant) fills one
  8×8 chroma prediction block (Cb or Cr — the caller passes the
  matching anchor-VOP plane and runs it once per component) by
  applying §7.6.2.1 half-sample bilinear interpolation to the supplied
  chroma MV against the forward and / or backward chroma reference
  plane, then averages pixel-by-pixel via
  `Pi[i][j] = (Pf[i][j] + Pb[i][j] + 1) >> 1` for `Bidirectional` and
  `Direct` modes (the §7.6.9.5.3 last paragraph rule). Chroma uses
  half-pel bilinear regardless of the VOL `quarter_sample` flag — the
  round-27 `chroma_mv_from_luma_blocks` already reduced the K luma MVs
  to a single half-pel chroma MV per direction, and §7.6.5 paragraph
  above Table 7-13 fixes §7.6.2.1 bilinear (not §7.6.2.2 FIR) for
  chroma. New public constants `CHROMA_BLOCK_SIDE = 8` and
  `CHROMA_BLOCK_PIXELS = 64`. 12 new unit tests; total crate test
  count now 565 + 8 doc.
- Round 28 of the clean-room rebuild: §7.6.1.6 vector padding technique.
  New `src/vector_padding.rs`.
  `pad_macroblock_vectors(vectors, transparencies, mode)` applies the
  per-macroblock §7.6.1.6 procedure to a `[MotionVector; 4]` of luma
  block MVs in Figure 6-8 / §6.1.3.4 raster order
  (`0 = TL, 1 = TR, 2 = BL, 3 = BR`).
  `MacroblockPaddingMode::AllZero` covers the §7.6.1.6 top-level branch
  for INTRA-coded macroblocks and P-VOP `skipped` macroblocks — all
  four `vectors[i]` are overwritten with `(0, 0)` regardless of
  `transparencies[i]`. `MacroblockPaddingMode::PerBlock` runs the
  per-block fallback branch: each `BlockTransparency::Transparent` block
  walks the precedence-ordered table
  `FALLBACK_CHAIN = [[1,2,3], [0,3,2], [3,0,1], [2,1,0]]` (the verbatim
  transcription of the §7.6.1.6 nested `?:` expressions: horizontal
  partner first, then diagonal / vertical in the alternating spec
  order) until the first `Opaque` partner is found, and copies that
  partner's MV in. Partner MVs come from a pre-padding snapshot so
  the fallback chain always reads the §7.6.1.6 *input* vectors, not
  the in-place-updated outputs of a prior iteration (the spec
  pseudo-code reads `MVx[j]` on every `?:` RHS — these must be the
  pre-padding values). `VectorPaddingError::AllTransparent` rejects a
  fully-transparent macroblock under `PerBlock` per the §7.6.1.6
  opening sentence ("applied to ... the transparent blocks within a
  *non-transparent* macroblock"). The output feeds three downstream
  consumers: the `K` luma MVs that flow into
  `chroma_mv_from_luma_blocks` (§7.6.5 luma → chroma derivation),
  the spatial MV predictor `MV1 / MV2 / MV3` candidate gathering
  (§7.6.5), and the temporally-next anchor VOP's co-located MVs that
  `direct_mode_motion_vector` linearly scales (§7.6.9.5 B-VOP direct
  mode). Per the §7.6.1.6 closing paragraph, S(GMC) `mcsel == 1`
  blocks must have the §7.8.7.3 averaged MV substituted into
  `vectors[i]` before invocation — this module accepts the
  post-substitution vectors and treats them as ordinary block MVs.
  Test coverage: 21 new unit tests (fallback-table shape sanity,
  AllZero branch overrides decoded MVs + ignores transparency
  pattern, PerBlock identity for all-opaque, each block as the
  single transparent slot, 2nd-choice fallback cases, 3rd-choice
  fallback cases, snapshot semantics verification, AllTransparent
  rejection without modifying vectors, negative-MV propagation,
  error display contains the spec clause) + 1 doctest + 1
  lib-level Error round-trip. Total crate test count now 553 + 8 doc.
- Round 27 of the clean-room rebuild: §7.6.5 chrominance motion-vector
  derivation `MVDCHR` from `K ∈ {1, 2, 3, 4}` luminance sub-block
  motion vectors (4:2:0 rectangular VOP). New `src/chroma_mv.rs`.
  `chroma_mv_from_luma_blocks(&[MotionVector])` sums the K luma MVs
  component-wise, divides by `2 * K` via floor (`i32::div_euclid`),
  and applies the §7.6.5 fractional rounding by indexing one of four
  newly-transcribed tables based on `K`:
  * `TABLE_7_13` (K = 1, "fourth sample resolution", 4 entries
    `[0, 1, 1, 1]`);
  * `TABLE_7_12` (K = 2, "eighth sample resolution", 8 entries
    `[0, 0, 1, 1, 1, 1, 1, 2]`);
  * `TABLE_7_11` (K = 3, "twelfth sample resolution", 12 entries
    `[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2]`);
  * `TABLE_7_10` (K = 4, "sixteenth sample resolution", 16 entries
    `[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]`).
  The table output `2` represents a carry into the next integer
  chroma-pel (one full chroma-pel = 2 half-sample offsets). The
  residue is pre-scaled by 2 so that the half-sample-grid sum
  indexes the table's `1/(4 * K)`-sample grid. Floor division
  (`div_euclid` / `rem_euclid`) keeps the residue non-negative for
  negative MV inputs so the lookup is well-defined across the sign
  range, matching the convention established by
  `quarter_sample::reduce_qpel_to_half_pel_chroma`. The §7.6.5
  "in quarter sample mode the vectors are divided by 2 before
  summation" pre-divide rule is left to the caller (the spec text
  doesn't pin the rounding for that pre-divide step); the module
  documents the suggested componentwise
  `reduce_qpel_to_half_pel_chroma` wrapping until the docs
  collaborator confirms the rule. `ChromaMvError::InvalidBlockCount`
  rejects `K = 0` and `K > 4`. Test coverage: 24 new unit tests
  (table-shape checks, K = 1 / 2 / 3 / 4 worked examples
  including the §7.6.5 worked spec example `(4, 4) * 4 → (2, 2)`,
  negative-input floor-division symmetry, error display) + 1
  doctest. Total crate test count now 532 + 7 doc.
- Round 26 of the clean-room rebuild: §7.6.9.5.3 B-VOP luminance
  prediction-block generation. New `src/bvop_prediction.rs`.
  `generate_b_vop_luma_prediction(forward_ref, backward_ref, mvs,
  mb_origin_x, mb_origin_y, vop_rounding_type, mode, prediction_mode)`
  builds the 16×16 luminance prediction macroblock by interpolating
  four Figure-6-8-ordered 8×8 sub-blocks via the §7.6.2.1 (half-pel)
  or §7.6.2.2 (quarter-pel) primitives and applying the §7.6.9.4 /
  §7.6.9.5.3 rule `Pi[i][j] = (Pf[i][j] + Pb[i][j] + 1) >> 1`.
  `BVopPredictionMode::{ForwardOnly, BackwardOnly, Bidirectional,
  Direct}` selects the §7.6.9 mode: the three non-Direct modes
  replicate the single bitstream MV across all four sub-blocks
  (§7.6.9.2 / §7.6.9.3 / §7.6.9.4 carry one MV per macroblock);
  Direct mode (§7.6.9.5) consumes the four per-sub-block
  `MVF[i]` / `MVB[i]` pairs produced by
  `direct_mode_motion_vector`. `BVopSampleMode::{HalfPel,
  QuarterPel { bits_per_pixel }}` plumbs the VOL `quarter_sample`
  flag through to the right interpolation primitive.
  `average_bidirectional_into(a, b, out)` exposes the §7.6.9.4
  averaging primitive directly (arithmetic in `u16` so the `+ 1`
  cannot overflow even at `u8::MAX` in both operands). Per the
  §7.6.9.5.3 explicit note, four 8×8 sub-blocks with the same MV
  in quarter-sample mode do NOT collapse to one 16×16 fetch —
  this module preserves the property by interpolating one sub-block
  at a time. The output is the §7.3 step-1 `p[y][x]` prediction;
  the §7.3 step-2 residual add and step-3 `[0, 2^bpp - 1]` clip
  remain the caller's responsibility (deliberately separated to
  keep each layer testable). 17 new unit tests + 1 doctest; total
  crate test count now 508 + 6 doc.

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

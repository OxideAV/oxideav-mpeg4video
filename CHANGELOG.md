# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

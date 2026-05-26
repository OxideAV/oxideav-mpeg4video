# oxideav-mpeg4video

A pure-Rust MPEG-4 Part 2 Video codec (ISO/IEC 14496-2) for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Round 17 of the clean-room rebuild (2026-05-26).** The prior
implementation was retired on 2026-05-18 under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md);
the VLC tables admitted to sourcing their numeric entries from an
external library. Master history was fully erased per the Hat-3 cold-
enforcement procedure.

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

## What works today

| Surface                                       | Status |
| --------------------------------------------- | ------ |
| `0x000001B0` Visual Object Sequence start     | parsed |
| `profile_and_level_indication`                | surfaced |
| `0x000001B5` Visual Object start              | parsed (video ID only) |
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
| MV-predictor candidate gathering (Figure 7-34) | not yet |
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
| Inter / B-VOP reconstruction (motion comp)        | not yet |
| RVLC Tcoef tables (B.23..B.25) + Type 5 escape | not yet |
| `sadct_disable == 0` modified inverse scan    | not yet |
| Encoder                                       | not yet |

339 round-1..17 unit tests pass.

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
Table 7-9 (motion-vector range per vop_fcode), and the syntax tables of
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
operator clause does not define it), and §7.4.1.1 / §6.2.7 (`block(i)`)
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
prediction coefficients of that block are assumed to be zero"). The
text was read from
`docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`. No
third-party MPEG-4 source was consulted.

## License

MIT — see [LICENSE](./LICENSE).

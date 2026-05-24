# oxideav-mpeg4video

A pure-Rust MPEG-4 Part 2 Video codec (ISO/IEC 14496-2) for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Round 9 of the clean-room rebuild (2026-05-24).** The prior
implementation was retired on 2026-05-18 under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md);
the VLC tables admitted to sourcing their numeric entries from an
external library. Master history was fully erased per the Hat-3 cold-
enforcement procedure.

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

The §7.4.3 spatial DC/AC predictor, AC-coefficient decode, and the
MV-predictor candidate gathering land in later rounds.

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
| §7.4.3 spatial DC/AC predictor add            | not yet |
| AC coefficient decode (§7.4.1.2)              | not yet |
| Encoder                                       | not yet |

173 round-1..9 unit tests pass.

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
codes) — and Table 6-33 (dbquant),
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
against every Table B.15 boundary row). The
text was read from
`docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`. No
third-party MPEG-4 source was consulted.

## License

MIT — see [LICENSE](./LICENSE).

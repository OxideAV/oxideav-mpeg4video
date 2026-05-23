# oxideav-mpeg4video

A pure-Rust MPEG-4 Part 2 Video codec (ISO/IEC 14496-2) for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Round 7 of the clean-room rebuild (2026-05-24).** The prior
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
  §7.6.3 differential-MV reconstruction. New `decode_mv_data` decodes
  the Table B.12 MVD VLC (`*_mv_data`, 1-13 bits) into the signed
  half-sample-unit value (range `[-32, 32]`). `decode_mv_component`
  conditionally reads the `*_mv_residual` field
  (`r_size = vop_fcode - 1` bits, gated on
  `vop_fcode != 1 && mv_data != 0`) and reconstructs the component as
  `MVD = (|mv_data|-1)*f + residual + 1` (with sign carry).
  `decode_motion_vector` walks the `forward` / `backward` / `direct`
  branches into a typed `MotionVectorDelta { mvdx, mvdy }` (direct mode
  reads two raw VLCs with no residual). `apply_predictor` performs the
  §7.6.3 `MV = P + MVD` step with the Table 7-9 `[low:high]` modulo
  wrap; the predictor `(Px, Py)` derivation (median-of-three over the
  MB grid, §7.6.2) and direct-mode temporal scaling (§7.6.5) remain
  for later rounds.

Predictor derivation + DCT-coefficient decode land in later rounds.

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
| `motion_vector(mode)` body (fwd/bwd/direct)   | `decode_motion_vector` (round 7) |
| `*_mv_data` Table B.12 VLC                     | linear-prefix decode, range [-32,32] (round 7) |
| `*_mv_residual` (`r_size = vop_fcode-1`)       | gated read + §7.6.3 reconstruction (round 7) |
| `MV = P + MVD` modulo wrap (Table 7-9)        | `apply_predictor` (round 7) |
| Predictor `(Px, Py)` derivation (§7.6.2)      | not yet (caller supplies) |
| Direct-mode temporal MV scaling (§7.6.5)      | not yet |
| DCT-coefficient decode                         | not yet |
| Encoder                                       | not yet |

145 round-1..7 unit tests pass.

## Provenance

Every numeric value and bit layout in this crate is sourced from
ISO/IEC 14496-2:2004 (3rd edition) — Tables 6-3 (start codes), 6-14
(aspect ratios), 6-15 (chroma formats), 6-16 (shape types), 6-19
(`sprite_enable`), 6-23 (time-code layout), 6-24 (`vop_coding_type`),
6-25 (`intra_dc_vlc_thr`), 6-32 (`dquant` codes), Annex B Tables B.1
(mb_type ↔ derived_mb_type ↔ included elements), B.3 (modb VLC),
B.4 (B-VOP mb_type, non-scalable), B.5 (B-VOP mb_type, scalable
enhancement layer), B.6 (mcbpc I-VOP), B.7 (mcbpc P-VOP), B.8
(cbpy 4-block), B.12 (MVD VLC) — and Table 6-33 (dbquant), Table 7-9
(motion-vector `[low:high]` range per `vop_fcode`), and the syntax
tables of §6.2.2 / §6.2.3 / §6.2.4 / §6.2.5 / §6.2.6 (including
§6.2.6.2 `motion_vector(mode)`) plus the semantics in
§6.3.3 (including the `8*[2-64]` zigzag-ordered quant-matrix list,
the zero-sentinel "remaining values set equal to the last non-zero
value" rule, and the intra "shall always be 8" / non-intra "shall
not be 0" first-value constraints) / §6.3.4 / §6.3.5 / §6.3.6
(macroblock-related semantics, including the B-VOP `modb` / `mb_type`
/ `cbpb` / `dbquant` rules, the §6.3.6.2 `*_mv_data` / `*_mv_residual`
/ `r_size = vop_fcode - 1` rules, and the default-type "direct" vs
"forward mc + Q" choice from §7.9.2.8.3) / §7.6.3 (general motion-
vector decoding process: the `f` / `high` / `low` / `range`
reconstruction and the `mv_data == 2 * (vector difference)` note).
The text was read from
`docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`. No
third-party MPEG-4 source was consulted.

## License

MIT — see [LICENSE](./LICENSE).

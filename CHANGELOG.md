# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

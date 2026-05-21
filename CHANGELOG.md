# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- VOP and macroblock decoding are not in scope this round.
- Studio Profiles (`profile_and_level_indication` 0xE1..=0xE8) and
  Fine-Granularity-Scalable VOL headers are recognised and rejected
  with typed errors rather than silently mis-parsed.
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

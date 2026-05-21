# oxideav-mpeg4video

A pure-Rust MPEG-4 Part 2 Video codec (ISO/IEC 14496-2) for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Round 1 of the clean-room rebuild (2026-05-21).** The prior
implementation was retired on 2026-05-18 under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md);
the VLC tables admitted to sourcing their numeric entries from an
external library. Master history was fully erased per the Hat-3 cold-
enforcement procedure.

Round 1 brings back **structural parsing of the §6.2 configuration
headers** — enough to identify a VOL, surface its profile / level
byte, pixel dimensions, time-increment resolution, aspect ratio,
optional `vol_control_parameters` + VBV block, and detect Studio /
FGS branches with typed errors instead of silent mis-parse.

VOP- and macroblock-level decoding land in later rounds.

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
| VOP header                                    | not yet |
| Macroblock decode                             | not yet |
| Encoder                                       | not yet |

20 round-1 unit tests pass.

## Provenance

Every numeric value and bit layout in this crate is sourced from
ISO/IEC 14496-2:2004 (3rd edition) — Tables 6-3 (start codes), 6-14
(aspect ratios), 6-15 (chroma formats), 6-16 (shape types) — and the
syntax tables of §6.2. The text was read from
`docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`. No
third-party MPEG-4 source was consulted.

## License

MIT — see [LICENSE](./LICENSE).

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- encoder: single-warp-point Global Motion Compensation (GMC) — VOL
  advertises `sprite_enable = 2` + `no_of_sprite_warping_points = 1`
  + `sprite_warping_accuracy = 0` (1/2-pel `s = 2`); each P-VOP is
  emitted as an `S(GMC)`-VOP with one `(du, dv)` `sprite_trajectory()`;
  every Inter / InterQ MB carries an `mcsel` bit picking between
  translational MC and warp-predicted MC. Per-VOP global translation
  is estimated by a coarse `±16`-pel SAD scan against the reference;
  per-MB GMC vs translational SAD comparison drives the `mcsel`
  decision. Exposed through the `gmc` codec option (`"1"` / `"true"`
  to enable; defaults off, preserving round-19 behaviour).
  See `gmc::encode_warping_mv`, `gmc::encode_sprite_trajectory`,
  `encoder::build_gmc_trajectory`, and the `gmc` mode-decision +
  `mcsel` emission paths in `pvop::estimate_and_encode_mb` /
  `pvop::emit_p_mb`. ffmpeg cross-decode validated on a synthetic
  global-pan testsrc (`tests/p_vop.rs::gmc_ffmpeg_decode`).
- encoder: per-VOP-type quantiser knobs (`qp`, `qp_i`, `qp_p`, `qp_b`)
  exposed via `CodecParameters::options`. Each value is range-checked
  to `[1, 31]` (the 5-bit `vop_quant` field) and rejected with
  `Error::invalid` when out of range.
- encoder: `g` (GOP-size) knob — picks the I-VOP cadence in frames.
  Range-checked to `[1, 300]`.

### Changed

- decoder: `parse_vop` now follows the §6.2.5 spec field order — when
  `vop_coding_type == "S"` AND the VOL has `sprite_enable in
  {static, GMC}`, `sprite_trajectory()` (and optionally
  `brightness_change_factor()`) is read BEFORE `vop_quant`, not after
  `vop_fcode_forward`. `vop_rounding_type` is now also emitted/read
  for `S(GMC)` VOPs per the spec.
- decoder: `inter::decode_p_mb` reads the `mcsel` bit RIGHT AFTER
  `mcbpc` (matching the §6.3.7 macroblock() syntax order) instead of
  after `cbpy` / dquant / `interlaced_information`. The pre-r20 order
  was internally consistent (encode + decode round-tripped) but
  rejected by ffmpeg and any spec-conformant decoder.
- decoder: `Mpeg4VideoDecoder::process_vop` now routes
  `VopCodingType::S` through the P-VOP body decoder when
  `vol.sprite_enable == 2` (S(GMC) is a P-substitute per §6.2.5).
  Static-sprite S-VOPs (`sprite_enable == 1`) still return
  `Error::Unsupported`.
- encoder: `cargo fmt` reflows on `bvop_enc.rs`, `encoder.rs`, `pvop.rs`
  + silenced an unused `vol` binding in `tests/reference_clips.rs` so
  `cargo clippy --all-targets -- -D warnings` is green again.

## [0.1.2](https://github.com/OxideAV/oxideav-mpeg4video/compare/v0.1.1...v0.1.2) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.1.1](https://github.com/OxideAV/oxideav-mpeg4video/compare/v0.1.0...v0.1.1) - 2026-04-24

### Other

- bump criterion 0.5 → 0.8
- bit-exact B-VOP decode - dbquant VLC + I-VOP backward ref fix
- B-VOP encoder — emit bidirectional VOPs under -bf N
- disable semver_check so struct-field additions bump patch
- field-sample MC for B-VOPs (§7.6.2.2)
- field-MV decode + field-DCT residual reorder in B-VOPs
- parse GOV time_code into absolute_base_seconds
- parse interlaced_information() inside B-VOP MB-layer
- treat None co_mv_grid as all-MBs-skipped for I-VOP backward ref
- 4MV direct mode in B-VOPs (§7.5.9.5.2)
- decode-order to display-order reorder buffer
- finish B-VOP MB decode + ffmpeg-based integration test
- propagate P-MB not_coded flag into MV grid
- fix MODB / MBTYPE tables per ISO Table 11-3 / 11-4
- update docs to reflect interlaced MB-layer decode
- wire interlaced_information() into intra + inter MB paths
- interlaced MB-layer parser module foundation
- integration tests for interlaced bitstream parsing
- parse static-sprite VOL rectangle (sprite_enable=1)
- parse VOP-level interlaced flags + brightness_change_factor VLC
- update docs + add GMC reference-clip VOL parse test
- wire GMC into P-VOP + MB decode paths
- GMC skeleton — VOL parse + warp/trajectory module
- update lib + decoder docstrings for B-VOP support
- B-VOP decoder skeleton (bvop.rs + decoder integration)
- add MODB + MBTYPE VLC tables (Tables B-16 / B-18)
- wire quarter-pel MC into VOL parser + inter MB path
- add quarter-pel 8-tap filter (§7.6.2.2)
- simd module + benchmarks + MC interior fast-path
- release v0.0.7

## [0.1.0](https://github.com/OxideAV/oxideav-mpeg4video/compare/v0.0.6...v0.1.0) - 2026-04-19

### Other

- promote to 0.1 as confirmed workign (decoding)
- 4MV-mode P-VOP: flush partial MB into mv_grid between blocks
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- reject mislabelled MS-MPEG4 bitstreams on first packet

## [0.0.6](https://github.com/OxideAV/oxideav-mpeg4video/compare/v0.0.5...v0.0.6) - 2026-04-19

### Other

- bump oxideav-codec to 0.0.5
- bump oxideav-core to 0.0.6
- claim AVI FourCCs via the new CodecTag registry

## [0.0.4](https://github.com/OxideAV/oxideav-mpeg4video/compare/v0.0.3...v0.0.4) - 2026-04-18

### Other

- update mpeg1video references to mpeg12video
- rewrite to enumerate decoder + encoder coverage
- reject interlaced/data-partitioned VOLs at VOP time
- re-emit previous reference on not-coded VOP
- implement plain-DC path for P-VOP intra MBs

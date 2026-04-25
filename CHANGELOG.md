# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

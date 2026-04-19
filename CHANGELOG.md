# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/OxideAV/oxideav-mpeg4video/compare/v0.1.0...v0.1.1) - 2026-04-19

### Other

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

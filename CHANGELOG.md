# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- encoder + decoder: **reversible VLC (RVLC)** for DCT coefficients
  (ISO/IEC 14496-2 Tables B.23–B.25, §7.4.1.2). Enabled with the
  `rvlc` codec option, which requires `dp=1` (per §6.2.5: RVLC is only
  legal inside `data_partitioned_motion_shape_texture()`). When on,
  the VOL flips `reversible_vlc = 1` and every per-MB AC walk inside
  the DP body — intra (I-VOP and Intra-in-P) AND inter (P-VOP) — is
  routed through Table B.23 (169 short codewords + 30-bit escape
  `00001 LAST(1) RUN(6) m LEVEL(11) m 0000 sign`) instead of the
  standard Tables B.16 / B.17 used by the combined-mode encoder.
  Both intra and inter blocks share the same prefix codewords; the
  triplet a prefix decodes to depends on the block type. The shared
  `0000` pattern between the opening and closing escape markers is
  reserved (no short codeword starts with `0000`), giving a forward
  parser a clean way to spot escape boundaries. Self-roundtrip PSNR
  on the synthetic moving-gradient fixture is 42.5 dB; ffmpeg
  cross-decode reports 43.5 dB on the I-VOP. Bit overhead vs DP-only
  at the same Q is about +2.2 % on the same fixture. New module:
  `src/rvlc.rs` (RVLC encode + forward decode + 12 lib tests + 5
  integration tests in `tests/rvlc.rs`). Reverse-direction
  error-recovery decode (Annex E.1.4.4 strategies 1–4) is a future
  follow-up — round-22 reads RVLC streams forward only, sufficient
  for clean transport.
- encoder + decoder: **data partitioning** (ISO/IEC 14496-2 §6.2.6 /
  §6.3.7). Enabled with the `dp` codec option. When on, the VOL
  advertises ARTS@L1 (`profile_and_level_indication = 0x91`,
  `video_object_type_indication = 10`) + `data_partitioned = 1` +
  `resync_marker_disable = 0` + `reversible_vlc = 0`; every I-VOP
  body is emitted as `data_partitioned_i_vop()` (per-MB
  `mcbpc + DC VLCs`, 19-bit DC marker `110 1011 0000 0000 0001`,
  per-MB `ac_pred_flag + cbpy`, per-MB AC walks); every P-VOP body
  is emitted as `data_partitioned_p_vop()` (per-MB
  `not_coded + mcbpc + MV`, 17-bit motion marker
  `1 1111 0000 0000 0001`, per-MB `cbpy`, per-MB AC walks). Round-21
  scope: 1MV-Inter only — DP is rejected by the encoder factory when
  combined with `qpel`/`gmc`/`bf>0` and the body emitter forces
  intra-in-P decisions back to inter (intra-in-P + 4MV under DP are
  round-22 follow-ups). The trailing
  `next_start_code()` stuffing now follows §5.2.4 ('0' then '1'-bits,
  or a full `0x7F` when already byte-aligned) for both DP and
  combined-mode emission so spec-conformant decoders don't keep
  parsing into the trailing zeros. ffmpeg cross-decode validated
  end-to-end (`tests/dp.rs::dp_ffmpeg_decode`).
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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4](https://github.com/OxideAV/oxideav-mpeg4video/compare/v0.1.3...v0.1.4) - 2026-05-05

### Other

- install ffmpeg via reusable workflow's extra_packages_* inputs
- Inter4MV under data partitioning + spec-correct 4MV chroma + predict_mv fix

### Fixed

- encoder/decoder: **§7.6.2 fig 7-6 MV-predictor neighbour-substitution
  rules.** The `predict_mv` 4MV path collapsed cases 2 and 3 of the
  spec's "set invalid candidate to zero / fill in with the valid one"
  rules: when *only* MV2 was invalid (rule 2: MV2 = 0), we instead
  set MV1 = MV2 = MV3 = MV1 (rule 3 — applies only when TWO are
  invalid). Likewise the `(None, _, _)` arm collapsed all four
  "MV1-invalid" cases into rule-4 zero-fill. The bug was self-
  consistent (encoder and decoder shared the same `predict_mv`) so
  self-roundtrips always passed; ffmpeg cross-decode of any 4MV
  stream that triggered rule 2 (e.g. first-MB-of-first-row 4MV with
  no top neighbour) drifted on the post-I P-VOP and accumulated 16
  dB+ losses by frame 3. Rewritten to count valid candidates and
  apply the correct rule (1/2/3/4) directly. The combined-mode 4MV
  test `tests/p_vop.rs::p_vop_4mv_subblock_motion_roundtrip` already
  covered the encode→our-decode round-trip; the new
  `tests/dp.rs::dp_p_vop_inter4mv_roundtrip` is the first hard-
  asserted ffmpeg cross-decode of a multi-frame 4MV stream and would
  have caught this bug had it existed at the time of round 13.
- encoder/decoder: **§7.6.5 + Table 7-10 4MV chroma MV derivation.**
  The 4MV chroma branch previously short-circuited as
  `luma_mv_to_chroma(sum / 4)` (the K=1 single-MV reduction applied
  to the 4MV average). For K=4 the spec routes `MVDCHR` through the
  Table 7-10 sixteenth-pel modifier table, which disagrees with the
  shortcut at the ±1/16-sample boundary (e.g. `sum=14` → spec
  chroma_half_pel=2, shortcut=1). New helper
  `mc::luma_4mv_sum_to_chroma` implements the spec table directly;
  both the P-VOP encoder (`pvop::estimate_and_encode_mb_inner`) and
  the P-VOP decoder (`inter::decode_p_mb`) now use it for the
  half-pel-luma 4MV path. QPel 4MV stays on the average-then-reduce
  path because the spec halves the QPel components before summation.

### Added

- encoder: **Inter4MV under data partitioning (§6.3.7).** The DP P-VOP
  body emitter (`dp::encode_p_vop_body_dp_with_grid`) now picks
  between 1MV-Inter / Inter4MV / Intra-in-P with the same
  SAD+lambda heuristic the combined-mode encoder uses, instead of
  forcing 1MV. Per-MB part 1 emits Table B-13 rows 16..=19 (Inter4MV
  MCBPC) followed by four MVDs (median predictor commits each MV to
  the in-MB grid before predicting the next sub-block, per §7.6.2
  fig 7-6). Per-MB part 2 (cbpy) and part 3 (AC walks) reuse the
  combined-mode emitters. The DP decoder's part-1 motion-decode
  branch grew an `Inter4MV` arm that mirrors the encoder. The
  `PMbState::Inter` carrier now stores `mv4: [(i32,i32); 4]` plus
  `four_mv: bool` so the reconstruct path can dispatch between
  `predict_luma_mb` (1MV) and `predict_luma_mb_4mv` (4MV) and pick
  the correct chroma-MV derivation. New hard-asserted ffmpeg
  cross-decode test `tests/dp.rs::dp_p_vop_inter4mv_roundtrip`
  verifies an Inter4MV MCBPC prefix appears in the post-I P-VOP and
  that ffmpeg decodes every frame within 25 dB of source (~38 dB on
  the synthetic per-block-motion fixture). Mutually exclusive with
  GMC / QPel / B-frames is unchanged.
- encoder: **Profile bump to ASP@L1 for `dp=1`.** The DP path
  previously advertised ARTS@L1 (PLI `0x91`, vot `10`), but per
  ISO/IEC 14496-2 Annex G the ARTS profile lacks Inter4MV; ASP@L1
  (`0xF1`, vot `4`, verid `2`) is the smallest profile that admits
  DP + Inter4MV simultaneously. The `verid=2` bump pulls in
  `quarter_sample`, `newpred_enable` and `reduced_resolution_vop_enable`
  bits in the VOL — all emitted as 0 for the DP path.

- decoder: **RVLC strategy 1-4 production picker** (ISO/IEC 14496-2
  §E.1.4.4.2.1). Wires the round-24 forward + reverse walker
  primitives into the production DP I-VOP and P-VOP decoders through a
  new public function `rvlc::decode_rvlc_ac_partition`. The picker
  runs both walkers across the AC partition, captures `(N1, L1)` from
  the forward direction and `(N2, L2)` from the reverse direction
  (reading a bit-reversed copy of the partition forward), then merges
  by the four strategies the spec defines:
  * Strategy 1 (`L1+L2 < L && N1+N2 < N`) — gap; head from forward,
    tail from reverse, middle blocks zeroed (= concealed; the part-1
    DC still drives the picture so concealment shows as a flat-DC
    patch instead of garbage).
  * Strategy 2 (`L1+L2 < L && N1+N2 >= N`) — overlap; same keep-rule
    as Strategy 4 (forward through the midpoint, reverse past it).
  * Strategy 3 (`L1+L2 >= L && N1+N2 < N`) — same gap-conceal as 1.
  * Strategy 4 (`L1+L2 >= L && N1+N2 >= N`) — overlap; forward owns
    `[0..N1]`, reverse owns `[N1..N]`. The L1+L2 vs L test is
    informative-only — the actual block-keeping rule is governed by
    `N1+N2` vs `N`. New types `RvlcBlockDesc`, `RvlcBlockOutcome`,
    `RvlcPickerStats` carry the descriptor / per-block outcome /
    summary stats. `dp::decode_ivop_dp` and
    `dp::decode_pvop_dp_with_grid` route their AC partition through
    the picker when `vol.reversible_vlc = 1`. Four new lib tests
    (clean partition Strategy-4, mid-byte XOR Strategy-1, head bit
    flip + reverse-only recovery, bit-misaligned `start_bit` offset
    handling) plus one new integration test
    (`rvlc_picker_recovers_at_least_baseline`) — total 127 lib + 32
    integration tests after this change. The 16-block mid-byte XOR
    fixture from #175 still recovers ≥14/16 bit-exactly through the
    production picker; clean streams stay bit-exact (Strategy 4 keeps
    the forward output unchanged).
- decoder: **RVLC reverse-direction decoder + best-effort recovery
  walkers** (ISO/IEC 14496-2 Annex E.1.4.4). Five new public functions
  in `src/rvlc.rs`: `bit_reverse_buffer`, `decode_intra_ac_reverse`,
  `decode_inter_ac_reverse`, `try_decode_intra_ac`, `try_decode_inter_ac`.
  The reverse decoder rests on a property the spec hints at but doesn't
  spell out: bit-reversing every short B.23 codeword (prefix +
  sign-as-LSB) yields a SECOND valid prefix code over the same
  169-symbol set — verified by the new
  `tests::reverse_table_is_prefix_code` lib test (no two reverse
  codewords coincide; no reverse codeword is a prefix of another). The
  reverse parser walks this second table over a bit-reversed copy of
  the AC partition; the 30-bit RVLC escape `00001 LAST(1) RUN(6) m
  LEVEL(11) m 0000 sign` reverses to `sign 0000 m LEVEL_rev(11) m
  RUN_rev(6) LAST 10000` and is recognized by its `s0000` opening
  signature. Acceptance test
  `tests/rvlc.rs::rvlc_corruption_recovery_beats_baseline` builds the
  same 16-block AC coefficient stream under both the RVLC writer and
  the standard B.16 Tcoef writer, corrupts a 3-byte window in the
  middle of each, and counts blocks recovered bit-exactly: standard
  Tcoef stops at the damage and recovers 7/16 forward + 0/16 reverse;
  RVLC recovers 7/16 forward + 7/16 reverse = 14/16. Eleven new tests
  total (10 lib + 1 integration). Strategy 1-4 picker
  (§E.1.4.4.2.1 per-MB N1/N2 + L1/L2 merging) landed in the next
  round — see the round-25 entry above.
- encoder + decoder: **Intra-in-P macroblocks under data partitioning**
  (ISO/IEC 14496-2 §6.2.5.3 `data_partitioned_p_vop()`). The DP P-VOP
  body emitter now mirrors the combined-mode intra-in-P decision (§6.3.7
  / Table B-22 mb_type=3): when the per-MB inter SAD exceeds the intra
  MAD proxy by `INTRA_IN_P_BIAS + INTRA_MARGIN`, the MB is re-encoded
  as intra and routed through the DP partitions per spec — Intra MCBPC
  (Table B-13 rows 4..=7, NO motion vector) into part 1, then
  `ac_pred_flag + raw cbpy + 6 intra DC differentials` into part 2
  (after `motion_marker`), then intra AC walks (Table B-16 / B.23 RVLC)
  into part 3. Decoder mirror: part 1 dispatches on `derived_mb_type`
  to skip the MV read for intra MBs and accumulate cbpy + DC diffs in
  part 2; reconstruction uses the I-VOP DP intra recipe (DC predictor
  from neighbour grid, dequant, IDCT) per intra MB, with inter MBs
  resetting their predictor slot. Required a `force_one_mv` knob on
  `pvop::estimate_and_encode_mb` (new
  `estimate_and_encode_mb_one_mv`) — without it the inner ME could
  pick 4MV for a high-residual MB and the DP path's single-MV emission
  would desync encoder/decoder reconstruction. ffmpeg cross-decode
  validated on a synthetic scene-change clip with mixed intra+inter MBs
  (`tests/dp.rs::dp_p_vop_intra_in_p_scene_change_roundtrip`, cut-frame
  PSNR ~38.7 dB through both our decoder and ffmpeg).
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

## [0.1.3](https://github.com/OxideAV/oxideav-mpeg4video/compare/v0.1.2...v0.1.3) - 2026-05-03

### Other

- drop duplicate semver_check key
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- RVLC strategy 1-4 production picker (Annex E.1.4.4.2.1)
- RVLC reverse-direction decoder + per-block error recovery
- encoder + decoder: Intra-in-P macroblocks under data partitioning
- encoder + decoder reversible VLC (Tables B.23-B.25)
- encoder + decoder data partitioning (§6.2.6 / §6.3.7)
- encoder single-warp-point GMC
- round-19 encoder qp + g codec options
- cargo fmt cleanup + silence unused vol binding
- adopt slim VideoFrame shape
- adopt slim VideoFrame shape
- quarter-pel motion estimation + MC (§7.6.2.2)
- quarter-pel motion estimation (§7.6.2.2 / §7.5.4)
- intra-MB-in-P fallback for scene changes
- 4MV mode decision + per-block ME + Inter4MV bitstream
- 4MV-direct + DIRECT_BONUS sweep + vti_bits fix
- wire B-MB cbpb residual emit + dbquant=0 sidechannel
- pin release-plz to patch-only bumps

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

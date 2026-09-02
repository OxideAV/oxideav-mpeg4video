# Conformance-fixture provenance

Every fixture pairs an MPEG-4 Visual **elementary stream** (`.m4v`,
start-code-delimited §6.2.1 units — no container) with the raw
`yuv420p` **reference decode** of that stream in display order
(`.yuv`). Both sides were produced by **black-box invocations of the
`ffmpeg` binary** (version 8.1) used opaquely as an encode/decode
oracle; no external implementation source was consulted.

## Reference decodes (`.yuv`)

Every expected output was generated with the **floating-point IDCT**
selected, so the reference decoder's inverse transform is the
mathematical Annex A.1 transform rather than an integer approximation
— this is what makes whole-stream bit-exact comparison meaningful
(our decoder evaluates the same ideal transform in `f64`):

```
ffmpeg -idct faani -i <name>.m4v -f rawvideo -pix_fmt yuv420p <name>.yuv
```

Two caveats measured against this oracle (see `tests/conformance.rs`
for the per-stream consequences):

* **Near-tie samples.** The oracle computes the ideal IDCT in single
  precision; where the ideal spatial value lies within ~1e-5 of a
  rounding boundary (e.g. 12.5000007, 238.4999993 — measured by
  instrumenting our `f64` transform), its float error can cross the
  boundary. Our double-precision rounding is the mathematically
  correct one, so such isolated samples legitimately differ by ±1.
* **§7.4.4.5 mismatch control.** The oracle applies the method-1
  mismatch toggle to non-intra blocks only (verified by toggling our
  implementation per block class: intra-skip collapses the
  `mpeg_quant` stream's diffs from 3062 to 4). §7.4.4.5 has no intra
  exemption, so our decoder keeps the spec behaviour by default and
  the `mq_ipb_64x64` assertion carries a ±1 envelope; the opt-in
  ecosystem-compat mode (`DecodeOptions::ecosystem`) reproduces the
  oracle up to the 4 near-ties (`compat_*` pins in
  `tests/conformance.rs`).

## Streams (`.m4v`)

### Generated this round (commands recorded verbatim)

Source is the deterministic `lavfi` `testsrc2` generator; all encodes
are Simple/Advanced-Simple-profile MPEG-4 Part 2 at fixed `-qscale:v 4`.

```
# method-1 (MPEG) inverse quantisation + §7.4.4.5 mismatch control, I/P/B
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -mpeg_quant 1 -f m4v mq_ipb_64x64.m4v

# progressive alternate scan (§6.3.5 alternate_vertical_scan_flag), I/P/B
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -alternate_scan 1 -f m4v altscan_ipb_64x64.m4v

# §7.4.3.3 AC prediction (ac_pred_flag == 1 intra macroblocks), I/P/B
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -flags +aic -f m4v aic_ipb_64x64.m4v

# quarter-sample + four-MV anchors (direct over 4-MV co-located), I/P/B
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -flags +qpel+mv4 -f m4v qpel_mv4_ipb_64x64.m4v

# QCIF I/P/B with ~120-byte video packets (resync markers)
ffmpeg -f lavfi -i "testsrc2=size=176x144:rate=25:duration=0.6" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -ps 120 -f m4v ipb_176x144.m4v

# interlaced field DCT + field motion estimation + quarter-sample, I+P
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 0 -flags +ildct+ilme+qpel -top 1 \
  -f m4v ilaced_qpel_ip_64x64.m4v

# method-1 quantisation crossed with the other axes (added once the
# ecosystem-compat mode made the mpeg_quant streams exact-verifiable):
# mpeg_quant + quarter-sample + 4MV, I/P/B
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -mpeg_quant 1 -flags +qpel+mv4 \
  -f m4v mq_qpel_mv4_ipb_64x64.m4v

# mpeg_quant + interlaced field DCT/ME + B-VOPs (both compat
# divergences in one stream)
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -mpeg_quant 1 -flags +ildct+ilme \
  -top 1 -f m4v mq_ilaced_ipb_64x64.m4v

# mpeg_quant + §6.2.5.3 data partitioning, I/P/B (the reference decode
# is byte-identical to mq_ipb_64x64.yuv — same encoder decisions)
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -mpeg_quant 1 -data_partitioning 1 \
  -f m4v mq_dp_ipb_64x64.m4v

# interlaced field DCT/ME + quarter-sample + B-VOPs (field-qpel through
# the §7.7.2.2 B modes and the interlaced-direct path)
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -flags +ildct+ilme+qpel -top 1 \
  -f m4v ilaced_qpel_ipb_64x64.m4v

# mpeg_quant + interlaced field DCT/ME + quarter-sample + B-VOPs
# (every divergence axis in one stream)
ffmpeg -f lavfi -i "testsrc2=size=64x64:rate=25:duration=0.48" \
  -c:v mpeg4 -qscale:v 4 -g 6 -bf 2 -mpeg_quant 1 \
  -flags +ildct+ilme+qpel -top 1 -f m4v mq_ilaced_qpel_ipb_64x64.m4v
```

### Field-qpel probe pins (`fq_probe_*.yuv`)

The seven `fq_probe_*.yuv` files are reference decodes of
**constructed** probe streams that arbitrated the §7.7.2.1
quarter-sample field-interpolation geometry (see
`tests/field_qpel_probes.rs`, which rebuilds each probe bitstream
deterministically — the streams themselves are therefore not stored).
Each probe is the `ilaced_qpel_ip_64x64.m4v` configuration headers +
I-VOP followed by one hand-written P-VOP: every macroblock skipped
except macroblock (1, 1), which is field-predicted
(`field_prediction == 1`) with chosen field reference selections and
field MVDs and no residual. Each expected output was produced by the
same black-box command as every other reference decode:

```
ffmpeg -idct faani -i fq_probe_<name>.m4v -f rawvideo -pix_fmt yuv420p fq_probe_<name>.yuv
```

These probes pinned two geometry facts the printed spec text leaves
open (both now the decoder's default reading, asserted bit-exact in
both behaviour modes): the 16-wide luma field block interpolates as
two 8×8 §7.6.2.2 blocks with per-sub-block Figure 7-30 mirroring, and
the chroma field MV's vertical quarter → half halving floors on the
field grid (`Div2Round(mv_y >> 2)`).

### Interlaced-direct probe pins (`dm_probe_*.yuv`)

The eight `dm_probe_*.yuv` files are reference decodes of constructed
probe streams that arbitrated the §7.7.2.2 interlaced-direct
derivation with **non-zero co-located field MVs over textured
anchors** (see `tests/direct_mode_probes.rs`, which rebuilds each
probe bitstream deterministically). Each probe is the
`ilaced_ipb_64x64.m4v` configuration headers + I-VOP, a hand-written
P-VOP whose macroblock (1, 1) is field-predicted with chosen field
reference selections and field MVs (all else skipped), and a
hand-written B-VOP whose macroblock (1, 1) is a direct macroblock
(zero-bit `modb "1"`, or explicit `modb "01"` + `mb_type "1"` with a
chosen `MVD[0]`). Expected outputs via the standard oracle command:

```
ffmpeg -idct faani -i dm_probe_<name>.m4v -f rawvideo -pix_fmt yuv420p dm_probe_<name>.yuv
```

Findings: the compat zero-co-located model is confirmed
unconditionally for non-zero and absent `MVD[0]`; a transmitted
`MVD[0] == (0, 0)` instead observes progressive direct over
`Div2Round(MVf1 + MVf2)` (documented divergence, ruling pending —
`tests/direct_mode_probes.rs` module docs).

### From earlier rounds

The remaining thirteen `.m4v` streams predate this notes file; they
were produced by the same black-box encoder from 64×64 `testsrc` /
`testsrc2` sources (Simple-Profile settings, GOP/`-bf`/tool flags as
described per-test in `tests/conformance.rs`), and
`ilaced_direct_176x144.m4v` is the interlaced direct-mode stream
staged in `docs/video/mpeg4-visual/fixtures/interlaced-direct-bframes/`
(refs #176). Their exact command lines were not recorded at the time;
their SHA-256 below pins the byte-exact inputs, and every `.yuv` was
regenerated this round with the command above.

## Encoder-produced streams (round 438)

The `enc_intra_*` pairs flow the OTHER way from every fixture above:
the `.m4v` streams were produced by **this crate's own encoder**
(`ivop_encode`, see `tests/encoder_blackbox.rs` — 3 I-VOPs, 64x64,
qp 4, cost-decided AC prediction, method-2 / method-1 quantisation)
from the deterministic synthetic source embedded in that test, and
the `.yuv` files are the black-box reference decodes of those
streams with the floating-point IDCT:

```
ffmpeg -idct faani -i enc_intra_m2_64x64.m4v -f rawvideo -pix_fmt yuv420p enc_intra_m2_64x64.yuv
ffmpeg -idct faani -i enc_intra_m1_64x64.m4v -f rawvideo -pix_fmt yuv420p enc_intra_m1_64x64.yuv
ffmpeg -idct faani -i enc_ip_m2_64x64.m4v -f rawvideo -pix_fmt yuv420p enc_ip_m2_64x64.yuv
```

`enc_ip_m2_64x64.m4v` is the I+P sibling (1 I-VOP + 5 P-VOPs over a
translating scene: §7.6 motion estimation, half-pel refinement,
`not_coded` skips; method-2, qp 4); its reference decode is
**bit-exact** against this crate's decode of the same stream.

The tests assert byte-determinism of re-encoding and bit-exact
agreement between this crate's decode of the streams and the
reference decode (the method-1 pair carries the documented §7.4.4.5
intra-mismatch divergence: ecosystem-compat mode is bit-exact, the
literal-spec decode differs on 834 samples by at most ±1). If the
encoder's output changes, regenerate BOTH files and re-measure.

## Encoder-produced streams (round 443)

The round-443 encoder tail added three more encoder-produced pairs
(each 1 I-VOP + 4 P-VOPs, 64x64, method-2, qp 4; sources embedded in
`tests/encoder_blackbox.rs`; regenerate the streams with
`OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES=1 cargo test --test encoder_blackbox`):

* `enc_ip_4mv_64x64` — §6.3.7 inter4v (four-MV) macroblocks over a
  checkerboard of divergent 8x8-block motion fields;
* `enc_ip_qpel_64x64` — `quarter_sample == 1` (verid-2 VOL, ASP)
  over a smooth quarter-grid texture translating by (3, 1) quarter
  pels per frame (true fractional §7.6.2.2 motion);
* `enc_ip_qpel4mv_64x64` — both tools on the divergent-motion scene.
* `enc_ipb_64x64` — I/P/B via the registry encoder (`bf` 2: coded
  order I0 P3 B1 B2 + the flush tail; direct / forward / backward /
  interpolated §7.6.9 modes, `co_located_not_coded` zero-bit MBs,
  `vol_control_parameters` with `low_delay == 0`), 6 frames of the
  translating scene.
* `enc_ipb_qpel4mv_64x64` — the combined-tools sibling: `bf` 2 +
  `quarter_sample` + inter4v in one stream (direct mode over 4-MV
  co-located anchors on the quarter grid), 6 frames of the
  divergent-motion scene.

```
ffmpeg -idct faani -i enc_ip_4mv_64x64.m4v -f rawvideo -pix_fmt yuv420p enc_ip_4mv_64x64.yuv
ffmpeg -idct faani -i enc_ip_qpel_64x64.m4v -f rawvideo -pix_fmt yuv420p enc_ip_qpel_64x64.yuv
ffmpeg -idct faani -i enc_ip_qpel4mv_64x64.m4v -f rawvideo -pix_fmt yuv420p enc_ip_qpel4mv_64x64.yuv
ffmpeg -idct faani -i enc_ipb_64x64.m4v -f rawvideo -pix_fmt yuv420p enc_ipb_64x64.yuv
ffmpeg -idct faani -i enc_ipb_qpel4mv_64x64.m4v -f rawvideo -pix_fmt yuv420p enc_ipb_qpel4mv_64x64.yuv
```

All five reference decodes are **bit-exact** against this crate's
decode of the same streams (asserted in `tests/encoder_blackbox.rs`).

Round 452 added the `fcode > 1` pair (96×64, a textured background
translating by (20, 5) pels per frame — outside the `fcode == 1`
Table 7-9 range, so every P/B vector rides the `r_size`-bit residual
form of §6.2.6.2 `motion_vector()`):

* `enc_ip_fcode2_96x64` — `fcode` 2, half-sample I+P via the registry
  encoder (`gop-size` 12, qp 4), 6 frames;
* `enc_ipb_fcode3_qpel4mv_96x64` — `fcode` 3 + `quarter_sample` +
  inter4v + `bf` 2 (forward / backward / interpolated B vectors under
  the wide range; direct-mode deltas stay `fcode == 1`), 6 frames.

```
ffmpeg -idct faani -i enc_ip_fcode2_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_ip_fcode2_96x64.yuv
ffmpeg -idct faani -i enc_ipb_fcode3_qpel4mv_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_ipb_fcode3_qpel4mv_96x64.yuv
```

Both are bit-exact. (The `bf` 1 sibling of the second stream at qp 4
hit one near-tie sample — a single ±1 in one B-VOP, with the qp 3 /
5 / 6 and `bf` 2 variants all bit-exact — which is the documented
single-precision oracle caveat above, so the `bf` 2 variant was
pinned.)

The per-macroblock quantiser pair (96×48, four bands spanning every
activity class — flat / gradient / texture / full-range noise —
translating by (2, 1) pels per frame):

* `enc_ipb_aq4mv_96x48` — `mb-aq` (activity-classed `dquant` on
  I/P-VOPs as `intra+q` / `inter+q`, `dbquant` on B-VOPs) + inter4v
  (which carries no `dquant` and keeps the running quantiser) + `bf`
  2, qp 10, 6 frames.

```
ffmpeg -idct faani -i enc_ipb_aq4mv_96x48.m4v -f rawvideo -pix_fmt yuv420p enc_ipb_aq4mv_96x48.yuv
```

Bit-exact.

The error-resilience trio (registry encoder, 6 frames each):

* `enc_ipb_vp_fcode2_96x64` — §6.2.5 video packets (~500-bit
  target; `header_extension_code` alternating 1/0 so both header
  branches are decoded) in a combined-syntax I/P/B stream (`bf` 2,
  `fcode` 2 — the P marker is 17 zeros + 1, the B marker 17 zeros +
  1) over the 20-pel-per-frame scene;
* `enc_ip_dp_aq_96x48` — §6.2.5.3 data partitioning (`dc_marker` /
  `motion_marker`) + ~400-bit packets + per-macroblock `dquant`
  (`intra+q` in partition 1 of the I-VOP, `inter+q` / `intra+q` in
  partition 2 of the P-VOPs), I + 5 P over the mixed-activity scene;
* `enc_ipb_dprvlc_aq4mv_96x48` — data partitioning + reversible VLCs
  (Table B.23 texture partition incl. the Type-5 escape) + packets +
  `dquant` / `dbquant` + inter4v + `fcode` 2 + `bf` 2 (the B-VOPs stay
  combined-syntax inside the partitioned VOL, §6.2.5.3 NOTE).

```
ffmpeg -idct faani -i enc_ipb_vp_fcode2_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_ipb_vp_fcode2_96x64.yuv
ffmpeg -idct faani -i enc_ip_dp_aq_96x48.m4v -f rawvideo -pix_fmt yuv420p enc_ip_dp_aq_96x48.yuv
ffmpeg -idct faani -i enc_ipb_dprvlc_aq4mv_96x48.m4v -f rawvideo -pix_fmt yuv420p enc_ipb_dprvlc_aq4mv_96x48.yuv
```

All three bit-exact.

The GMC pair (96×64, a background panning by (6, 2) pels per frame):

* `enc_isb_gmc_qpel_96x64` — S(GMC)-VOP anchors (one §7.8.4 warping
  point, half-pel accuracy, per-MB `mcsel`) + quarter-sample + `bf` 2
  + `fcode` 3 + ~600-bit video packets (S packets carry no HEC), 6
  frames.

```
ffmpeg -idct faani -i enc_isb_gmc_qpel_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_isb_gmc_qpel_96x64.yuv
```

Bit-exact. Two findings from this pair's validation:

* **Table B.34 `dmv_length`** — the crate's `warping_mv_code()`
  reader (and the first encoder cut) treated `dmv_length` as a plain
  unary run; Table B.34 actually assigns `00`→0, `010`..`110`→1..=5
  and `SSS−3` one-bits + `0` for 6..=14. Fixed on both sides; crafted
  pure-GMC-copy probes (integer-pel, half-pel, negative, ±16-pel,
  vertical trajectories) then decoded **bit-exact** through the
  reference binary.
* **§7.8.7.3 non-positive averaged-MV divergence** — the reference
  decoder derives each *non-positive* AMV component one MV-grid unit
  lower than the §7.8.7.3 quantisation (probed with crafted
  GMC-neighbour + zero-MVD-local streams: half-sample du −2→−3,
  −3→−4, −4→−5, −10→−11, 0→−1; quarter-sample −6→−7, −8→−9, −18→−19,
  −20→−21; strictly positive components exact; per-component
  independent). The AMV feeds the §7.6.5 predictor candidates of
  local macroblocks and the §7.6.9 frame-direct co-located
  substitution. The decoder keeps the spec-literal quantisation by
  default; the opt-in **ecosystem-compat** mode now covers the rule
  (compat divergence 3), pinned by the `dec_sgmc_*` pairs below. The
  encoder fixture pans so every trajectory component stays strictly
  positive.

The compat-divergence-3 pins (`tests/compat_gmc_amv.rs`; the `.m4v`
sides are deterministic builds of that file, the `.yuv` sides the
reference decodes):

* `dec_sgmc_negamv_hp_64x64` — crafted half-pel probe (trajectory
  (−3, −7); GMC-coded MB, zero-MVD local MB, `not_coded` GMC copies);
* `dec_sgmc_negamv_qp_64x64` — the quarter-sample sibling
  (trajectory (−4, −10));
* `dec_sgmc_negtraj_96x64` — a full encoder-produced 4-frame S(GMC)
  stream whose dominant-motion trajectories go negative and zero
  (fcode 1 against 20-pel-per-frame motion). Ecosystem-compat decode
  is **bit-exact** against all three reference decodes; the
  spec-literal decode differs exactly where the rule bites.

```
ffmpeg -idct faani -i dec_sgmc_negamv_hp_64x64.m4v -f rawvideo -pix_fmt yuv420p dec_sgmc_negamv_hp_64x64.yuv
ffmpeg -idct faani -i dec_sgmc_negamv_qp_64x64.m4v -f rawvideo -pix_fmt yuv420p dec_sgmc_negamv_qp_64x64.yuv
ffmpeg -idct faani -i dec_sgmc_negtraj_96x64.m4v -f rawvideo -pix_fmt yuv420p dec_sgmc_negtraj_96x64.yuv
```

## Encoder-produced streams (round 455 — interlaced tools)

Deterministic builds of `tests/encoder_interlaced.rs` (the `.m4v`
sides; `OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES=1` regenerates them) over a
synthetic scene whose two fields translate independently, with the
reference decodes produced as above:

* `enc_ilaced_ip_64x64` — interlaced VOL, I + 3 P: per-macroblock
  §7.7.1 field DCT (`dct_type`), §7.7.2.1 field-predicted macroblocks
  (both reference-field parities, CASE 1/2/3 predictors), half-sample,
  fcode 1, qp 4. **Bit-exact.**
* `enc_ilaced_ipbb_compat_96x64` — interlaced I P B B P B B (coding
  order I0 P3 B1 B2 P6 B4 B5) with the ecosystem-compat emission (no
  direct mode over a field-predicted co-located macroblock): field
  forward / backward / bidirectional B macroblocks through the Table
  7-14 four-PMV bank, frame B modes, progressive direct, field DCT on
  B residuals. **Bit-exact.**
* `enc_ilaced_ipbb_spec_qpel_96x64` — the spec-literal emission with
  quarter-sample motion: §7.7.2.2 interlaced-direct macroblocks over
  field-predicted anchors. Anchors bit-exact; every differing sample
  lies inside an interlaced-direct macroblock, and the crate's
  ecosystem-compat decode (`DecodeOptions::ecosystem`) of this stream
  reproduces the reference decode **bit-exactly** — compat divergence
  1 confirmed on encoder-produced content.

```
ffmpeg -idct faani -i enc_ilaced_ip_64x64.m4v -f rawvideo -pix_fmt yuv420p enc_ilaced_ip_64x64.yuv
ffmpeg -idct faani -i enc_ilaced_ipbb_compat_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_ilaced_ipbb_compat_96x64.yuv
ffmpeg -idct faani -i enc_ilaced_ipbb_spec_qpel_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_ilaced_ipbb_spec_qpel_96x64.yuv
```

Two decoder findings from this validation (both black-box-arbitrated
on these pairs, conformance corpus unchanged):

* **§7.6.3 wrap on field vectors** — the reference decoder applies the
  §7.6.3 `[low:high]` modulo wrap to a §7.7.2.1 field vector on its
  own grid (horizontal in frame units, vertical in *field* units before
  the doubling); the crate's P-VOP field reconstruction now does the
  same (`reconstruct_field_motion_vectors_wrapped`), and the encoder
  keeps every field component inside the range so the wrap never
  fires on its own streams.
* **§7.6.4 clamp of a quarter-sample field read** — a field block whose
  vector reaches past the bottom edge clamps to the VOP's edge line on
  the *frame* grid (whatever its parity), as the half-sample field
  `mc` routine already did; the quarter-sample field view previously
  clamped within the field.

## Short-header streams (round 455 — §6.2.5.2 `short_video_header == 1`)

Raw H.263-compatible elementary streams (`.h263`: byte-aligned
`short_video_start_marker` pictures, no VOS/VOL), decoded with the
reference binary's raw short-header demuxer selected:

```
ffmpeg -idct faani -f h263 -i <name>.h263 -f rawvideo -pix_fmt yuv420p <name>.yuv
```

* `enc_sh_ippip_176x144` — deterministic build of
  `tests/encoder_short_header.rs` (`OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES=1`
  regenerates it): QCIF I P P I P, GOB headers on every GOB after the
  first, adaptive `dquant`, 8-bit `intra_dc_coefficient` FLC, Table
  B.17 + Type-4 escapes, picture-restricted half-sample vectors, end
  marker. The reference decode is **bit-exact** against our closed
  loop.
* `sh_ipp_176x144` — reference-encoder-produced QCIF I P P I P
  (`-c:v h263 -qscale:v 4 -g 3`, the H.263 baseline the short header
  encapsulates):

```
ffmpeg -f lavfi -i "testsrc2=size=176x144:rate=25:duration=0.2" \
  -c:v h263 -qscale:v 4 -g 3 -f h263 sh_ipp_176x144.h263
```

  Our short-header decode is **bit-exact** against the reference
  decode (no near-tie samples on this stream).

## SHA-256

```
d1d2c853300d6c1c13928b47dc1292e8cef810b61d2c94f64b089d0bb516cbea  enc_sh_ippip_176x144.h263
0966dc1b5fb73a87d6f9ff4c69f434ef6caade843fb658cedd801176da4b0f98  enc_sh_ippip_176x144.yuv
3860e79a00ab83d886fa7ac9c5873854d5b9a47019abba80dd67c6e4d396e50d  sh_ipp_176x144.h263
6942702085ec90a51652c86418bc3af86f432062d629d48f491964dfcb051729  sh_ipp_176x144.yuv
b164f0899ed06c42d17c9f401ff73ad551de36e51c2834377ddf08073feb784a  enc_ilaced_ip_64x64.m4v
558dddc1e6308df876704546b4fab28d63170e547bb19768793f905541c62c31  enc_ilaced_ip_64x64.yuv
94191bce28ecc0146014ce725c653f7de13f239330885ba9e9dfdf3cf59ce1aa  enc_ilaced_ipbb_compat_96x64.m4v
08fb2a3fd201286775c6909a9c7fd27f75100ca9918834be206617d97dc7aa4d  enc_ilaced_ipbb_compat_96x64.yuv
da3cc9c84b4ae41e279d0cb70e45df4dc02488dfe580d07564b008bffe9ee5ab  enc_ilaced_ipbb_spec_qpel_96x64.m4v
9e95fa2f33cf8d17977e512b09a585a9ae4a2d15733845e77de0eda306aa605d  enc_ilaced_ipbb_spec_qpel_96x64.yuv
026fcd2048c0d2a09be9fbcfd686f0596a8f55b9b67d5658be896f961d20301c  intra_64x64.m4v
c1e893305b462feedbeeced8979ae1b05084c514558a22a6aac6ef7350d261f0  intra_64x64.yuv
4fa229741f6c20b0e59b45d2955c55c33ca7c34c2e98bd5f2218ed7463229f5c  ip_64x64.m4v
16b79ef30645f2c5f323e4420f1d2066555e2cef0dea175ebfb718436ed85160  ip_64x64.yuv
e12650c7068b41634a194b2017133186bc6fb036d5267f3c62c9e5165d05b55d  ipb_64x64.m4v
402560f13a5220882ba48fa093512f67244028d5b0754920e80e7ebb7c76f45e  ipb_64x64.yuv
16e23ea88b749007b19298df532920b842faba9d566d544ae0a6f41931acc8e2  qpel_ip_64x64.m4v
8b7e69bfb584a5fb4c35a62b567de7cc7ce4d6bcfcecb39cb326d2293cf7ec2c  qpel_ip_64x64.yuv
35849caf154f76627b7eea77b538b9029096e3ec114d9eb790798b51addc7820  qpel_ipb_64x64.m4v
0369989106fd47e6f8607aadf471fa90ad2e97bcb597f69275d2847a9ecefe8a  qpel_ipb_64x64.yuv
d6e708c4a2688cfb5e1efd23f2a213579f3f61fad44d8fd776d0499fb14213df  mv4_ipb_64x64.m4v
e7eeccceaeb835382495b5e9446c8d2639cea30a6a8500dc25c6758097206463  mv4_ipb_64x64.yuv
a7f9b00ff5441ce5dd0352ff498d4770aa6607c6fffb161309d683dbdd4678c3  dp_ipb_64x64.m4v
e5c3f053f125f0ca5fca76d659912a24adeed31e8ff3a0173837af21df107bd4  dp_ipb_64x64.yuv
f3d6d0b2d521270e7542e2c7dd2b26357c55317a76c748fa79924ea729587c41  ilaced_intra_64x64.m4v
a964741780717d1c17f79c128ab8c67871f9d41771a374a2ea84f7492fafc6c8  ilaced_intra_64x64.yuv
5df88d65f627079da854f8241273a9bcf87cfc8985f249df199cfc8b2d0809fa  ilaced_altscan_intra_64x64.m4v
a964741780717d1c17f79c128ab8c67871f9d41771a374a2ea84f7492fafc6c8  ilaced_altscan_intra_64x64.yuv
d95f708d507e709539744f27d19fe455335cd34518aff8ad724547d0f77c51b1  ilaced_ip_64x64.m4v
933cae705cdf7e96e6506e152e98def76d9730bb344b4d82c50bf7b85d0ce07f  ilaced_ip_64x64.yuv
960f5c8f4f570760e160119a46b34194784d82133d365d15e9a0a6e21c7f48c6  ilaced_ip2_64x64.m4v
f729a160db8e7e0b5b493cee3bdbd602e707f55b3ea662df1dd0c00f707c3bc8  ilaced_ip2_64x64.yuv
89020ece21136c5ef89bdb077b43ede7d8e5fe07e406b91ac88d88bd46ae5e39  ilaced_ipb_64x64.m4v
451feced85890f34224b64b6673d52231f8996207bfe78a3cfc1e960f47f1e7c  ilaced_ipb_64x64.yuv
ad45ef215ac1da12d8dce0c13488c4757f9117a2c831c30a24d8cda9b7eb77be  ilaced_direct_176x144.m4v
58d740bd04793f89cf5d56d5bf5414d1a134bd7d08a136f4730794340e808f1b  ilaced_direct_176x144.yuv
f455f3ee01d9c2b4efe5956b9855840593c13acbd90145a16319b255278d686f  mq_ipb_64x64.m4v
ca2cb8b3cbb12801ede055f71d101004a65e13707f514bf7a8084a2feb74839d  mq_ipb_64x64.yuv
dd72d58639b988e72094f8c85fec74b5949a80782c6208a938f2e6e71eab07a9  altscan_ipb_64x64.m4v
db9a5417df965d855349a659f26020da8a8074fb3b0884249e75713e06f34149  altscan_ipb_64x64.yuv
6356ea6a2b808b09d5cefb707b69936c595efc19675baf02e8b52cb5b80509eb  aic_ipb_64x64.m4v
db9a5417df965d855349a659f26020da8a8074fb3b0884249e75713e06f34149  aic_ipb_64x64.yuv
3004a220df00a49b414480377e4dc1be3976ea0562e90215311ccd714c5dbd0d  qpel_mv4_ipb_64x64.m4v
017108c644b1d20e9800da550e09da519832aa589dd0d6a209e4ee6d57c3fde5  qpel_mv4_ipb_64x64.yuv
db1725c277485611ffb7e8604b9109cdd6df61a7d0609f2ee682e64ea0a85b52  ipb_176x144.m4v
97a9d06708982cbacf8cb974798af8cc5bd4f2e23c96d703287cd14df6736b4a  ipb_176x144.yuv
9600630769ff16bc829bd5cbf25a0ea61113e7ce852d4ddf79868ef99663340a  ilaced_qpel_ip_64x64.m4v
67c5323154a8a80b02c1d12d2834bac0c1f3ab9c905d00b0ebceff4c5e577d83  ilaced_qpel_ip_64x64.yuv
c6c38a7b94714432027886065c6f4f6bd0044cb81bba99ecaaec2d265e66defd  mq_qpel_mv4_ipb_64x64.m4v
98bb74ca9189a0c47dd06c51c11654919fc0643b1a74bb5d889d4f7600995fb0  mq_qpel_mv4_ipb_64x64.yuv
4bfce1cee6f977c82c5746960394a2435e9ba3e9bffb92aef05a6d695100e674  mq_ilaced_ipb_64x64.m4v
6507284607652f4bb09805cf997683729a6e1d04c327e2d86b8a4e93d6882bf6  mq_ilaced_ipb_64x64.yuv
7a34ef255a41ededc5a3a8090a80a69a3899a7fbecbf84ff03e9ad38589b11d5  mq_dp_ipb_64x64.m4v
ca2cb8b3cbb12801ede055f71d101004a65e13707f514bf7a8084a2feb74839d  mq_dp_ipb_64x64.yuv
7f3af441a9b0f41f5d8bbf689339388607581fc842be1b3c467c291d33bd9a8b  ilaced_qpel_ipb_64x64.m4v
96dccdd4f23f2eacecc76357a8dfcfbebde066ff6964d0a50e61ef0a4a14f82c  mq_ilaced_qpel_ipb_64x64.m4v
c50528e63995fc6fbbff8a12661632c2bd1b893cc186370fc665a72cc5fae431  mq_ilaced_qpel_ipb_64x64.yuv
ce464e80c84845d59f2a674d98ce37bedae4cd828c8429e26876d5725cc21df9  ilaced_qpel_ipb_64x64.yuv
5355f5c045c119ffe20d58a1c9c7938d0c4879c14773de02332115c96a2b8e2d  dm_probe_mixed.yuv
b58a72baed33fc9c2bc5b4f449c8e7c8c070b7e4c5d5d304ccab8489984a16a6  dm_probe_modb1.yuv
625426911d12683b122cf46975c87182ac9bc0a9916dd98188fea7c1307eba13  dm_probe_mvd0_a.yuv
0bbb86abc6a33704e3d5d11c0e6927eacdd18a6098799df759dee11d67236b8a  dm_probe_mvd0_b.yuv
03c5ca94625cf38431dd1427f0e7e6477b7c66ac0ef2fbf4902c0e6692041f08  dm_probe_mvd0_c.yuv
45d3f03857c26c9b7a479897e5433081ea6571a26ec2eca29e6767d7ac4bf916  dm_probe_mvdnz.yuv
0ccac0ec839f99f37348004c392e34afc029ca07be08781c84d7c2a62c3c4ef3  dm_probe_xref.yuv
bd5ebcb899e9cbf564155ffae0e0383486b5d03f7e35d3353547838a5bf60206  dm_probe_zero_ctl.yuv
750f917ee35f593d5766bf33309aa6659b39884ed56af3cf9d785a2a609f2c1d  fq_probe_diag.yuv
b29588dfd261fe8f4e04a4f8cba31df8150a7b77bb10b2714af5f1ab0c9d46ab  fq_probe_half_seam.yuv
133f043fbdb383ef5721aaf501a7a183efe840fba8a4a255f7e98bdc26006235  fq_probe_mixed.yuv
7bd25b9b4dec547be35ade5a29bc4145fb28001aa6467dd0dfd48225925cfe44  fq_probe_neg.yuv
1b754a60781267ac0f8cea8ff94608283f7c3240ef04976e11e9564965de19dc  fq_probe_odd_a.yuv
60918961e4ead055d11d56659173eef8b78a87f82a5c43b2160b69bb673ca9fd  fq_probe_odd_c.yuv
53e56fede4435ec365f4498832930c976414d7a52efe2d489e7ee275fb1ad6d7  fq_probe_odd_d.yuv
430a470fa3379dd9631e6f95dcbc618c181fc63ec88bedf5755542b825733351  enc_intra_m1_64x64.m4v
2321da80b0202a1a8ac870f69f06fba2cc982eaeea821bad73c5783cbb463f6f  enc_intra_m1_64x64.yuv
59ce00adf97ed7c44c757f8856f46705635f555ef69a09fc93c256e7c4fdf7d7  enc_intra_m2_64x64.m4v
4f13eb5e87a747ce02d5a859e49af6a06b4a90b4200d7f5f475d7756f734ae99  enc_intra_m2_64x64.yuv
9137ec57b46495fe659de018b8f27db6ee15b3add9529128aadf7fa7f9009d68  enc_ip_m2_64x64.m4v
9b9a9e622de3afb760bf461afe24c90478001acad21a19137fe5d5db3e737542  enc_ip_m2_64x64.yuv
4a14ef0d5089a070f71e0b79df13f310f9c0a1ff5de1652845550870544892b6  enc_ip_4mv_64x64.m4v
945a4da88674c8dd81c0bec23c0afbad480964d78f9f41d827861485505067c1  enc_ip_4mv_64x64.yuv
1dbefcff4a7749287a8588564f4e23f39b84a3058f0607fa5cf7e3af3c415996  enc_ip_qpel_64x64.m4v
e1e4c9d7fce8198331e805bc6a3743b5839c96b213747e9bd5a84ec438ec894f  enc_ip_qpel_64x64.yuv
67af5c1c95fdf1fd2a607cb6b646f234a39dd6ce79e25171784e550429d7ff2d  enc_ip_qpel4mv_64x64.m4v
37eb3c51c6a921daa19f9636de8b3c702a5f08f82dcbd9a88df81300fa1f2e3b  enc_ip_qpel4mv_64x64.yuv
4ba90df847a537c63f54275ca36306f44e35814ed4db130f034358d50615d1eb  enc_ipb_64x64.m4v
96f18d82ee50631c03daa101d460e00eafeed2643ab36bde90af32de1997b3b2  enc_ipb_64x64.yuv
5961b4f908f8ed069ac17ac29c8051ae2f51893324417d23fe36ccad27e3972e  enc_ipb_qpel4mv_64x64.m4v
2bb083df0e8d5a92f041822b026b119be92c0b5384e1170a0ce2ff9b62807c64  enc_ipb_qpel4mv_64x64.yuv
4fb8ad3bd96556986773058d121112fc479a2237993178a651053451c33c32e1  enc_ip_fcode2_96x64.m4v
769fc2f8f88c2038b18592649e8aa1bf324a452187533a7b399ee89ea193b1a9  enc_ip_fcode2_96x64.yuv
7b9ce94a56292b8541b2fabe1927d2ca30f867cd1fb3544f2daeeca4b5791e46  enc_ipb_fcode3_qpel4mv_96x64.m4v
584711548f66a8f862ce5bee4cc8c5fd48fd78f264c732097956756dbcf73779  enc_ipb_fcode3_qpel4mv_96x64.yuv
23aacccba29bac9f850ab4729e811dcd531302c3cc61e86c3111add3c3bfce6f  enc_isb_gmc_qpel_96x64.m4v
f1a65b08166aaab11863c5478cadefc201912db150551c5ed2f88c658900aa28  enc_isb_gmc_qpel_96x64.yuv
43ba930a07b45c825b5df81b66331bc1a8cfef066540ef9f22af8cd02a78d844  dec_sgmc_negamv_hp_64x64.m4v
e560f779d1d598419c80bf30629463fd1e0046834a2d0ac1ecf592ea130c8483  dec_sgmc_negamv_hp_64x64.yuv
60fd1ec7342131f55276c248c308be9f3c0d4a691d74338c77f9d7f874d31880  dec_sgmc_negamv_qp_64x64.m4v
69dd677fbcf59914484d3b9f27a67e3f7bbc2c36e45df2c688933de88db31e17  dec_sgmc_negamv_qp_64x64.yuv
7a8d5a086be97592b96ad1b4f276642df250feec564728637305b78afd44edf2  dec_sgmc_negtraj_96x64.m4v
fc61e5e5bb934ea9c3894ad9ab6316067307e2aef53f9642ad2a32d3ee6f97c5  dec_sgmc_negtraj_96x64.yuv
7f200f5e4089ebcc29b8899bc746db11fa5dbd6b0802d5a70998a5bb05ea6a48  enc_ipb_aq4mv_96x48.m4v
66e3662a5fc4fd1c58068e40ce5b7e722300c6efa02c2f744f308544a454a63a  enc_ipb_aq4mv_96x48.yuv
81e92c81b778fec93a78607b06fa3000343e58dd137e402227573324cb8b58d3  enc_ipb_vp_fcode2_96x64.m4v
4f8f03392dccd42405d26e33ccf97940be497c60834b743453e80efd17217685  enc_ipb_vp_fcode2_96x64.yuv
ea4de652b9352b4ebadec471d53b71435a1bb36101d636e8c504c57dc5a7e409  enc_ip_dp_aq_96x48.m4v
f958498a6b00db8595f0be1c28f77e042f8e7c4ef8fc863bb0b05537d76c30a0  enc_ip_dp_aq_96x48.yuv
498b14b46e2aa39f720f1c085bd62280ca52e00561529e30373034c268340d79  enc_ipb_dprvlc_aq4mv_96x48.m4v
57305ad87ed09378b0b41fe56cab77636e43d4bba0a62615caa2faa86604e2a8  enc_ipb_dprvlc_aq4mv_96x48.yuv
```

(Note: `aic_ipb_64x64.yuv` and `altscan_ipb_64x64.yuv` are
byte-identical — the two encodes quantise the same source to the same
reconstruction — as are the two interlaced-intra decodes.)

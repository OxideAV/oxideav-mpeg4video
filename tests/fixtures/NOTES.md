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

Bit-exact (stream regenerated twice in round 455: after the
GMC-vs-local preference became quantiser-scaled, and again once the
S(GMC)-VOP video packets started carrying `header_extension_code`
bodies with the `sprite_trajectory()` restatement — the reference
decode is byte-identical across both, and the reference decoder's
`untested` remark on those HEC bodies is its own path-coverage note).
Two findings from this pair's validation:

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
  spec-literal decode differs exactly where the rule bites. (Pair
  regenerated in round 455 after the GMC-vs-local preference became
  quantiser-scaled — same construction, new decisions, same result.)

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

## Field-vector range set (erratum E4)

`field_mv_range_fcode{1,2}.m4v` / `.yuv` and `field_mv_range_trace.csv`
are byte copies of the staged
`docs/video/mpeg4-visual/fixtures/interlaced-field-mv-range/` set
(`fcode{1,2}.m4v`, `fcode{1,2}.expected.yuv`, `field-mv-trace.csv`;
generation and oracle described in that directory's `notes.md`): two
352×288 interlaced I-P-P streams (`vop_fcode_forward` 1 and 2) whose
forced field vectors sit on, at and beyond the frame-unit Table 7-9
boundary, the decode obtained when the §7.6.3 reduction is applied to
`Py / 2 + MVDy` in **field** units before the §7.7.2.1 doubling
(equal to the producing encoder's own reconstruction), and the
per-field-vector trace of predictor, differential and the vector each
candidate reading yields. `tests/field_mv_range.rs` decodes both
streams bit-exact (both decode modes) and replays all 134 trace rows
through the field reconstruction. A black-box decode of the two
streams (`ffmpeg -idct faani …`) is byte-identical to the staged
expected output.

* `enc_ilaced_fieldrange_ipbb_64x160` — encoder-side pin of the same
  rule (deterministic from `tests/encoder_interlaced.rs`; `interlaced`,
  `bf 2`, `fcode 2`, qp 4, ecosystem-compat emission): fields moving 9
  field lines per frame in opposite directions, the right half
  mirrored. P field vectors reach 27 field lines (108 frame
  half-samples; the frame-unit bound is 63), B field vectors pass
  through the four-PMV bank with 26 reconstructions whose field-unit
  sum leaves `[low:high]` and is reduced. The reference decode is
  **bit-exact**, log clean — the §7.6.3 reduction of a B-VOP field
  vector is a field-unit reduction as well.

```
ffmpeg -idct faani -i enc_ilaced_fieldrange_ipbb_64x160.m4v -f rawvideo -pix_fmt yuv420p enc_ilaced_fieldrange_ipbb_64x160.yuv
```

## Interlaced S(GMC) stream (round 455)

* `enc_isb_ilaced_gmc_compat_96x64` — registry build (`interlaced`,
  `gmc`, `fcode 2`, `bf 2`, `ecosystem-compat`, qp 5; deterministic
  from `tests/encoder_interlaced.rs`) over a split scene: the left half
  pans coherently (GMC macroblocks, frame-predicted per §7.8.7.2, field
  DCT on their residuals), the right half has independent field
  velocities (§7.7.2.1 field-predicted `mcsel == 0` macroblocks against
  the GMC neighbours' averaged-MV candidates); interlaced B-VOPs over
  the S anchors' field motion (compat emission — no interlaced direct).
  The reference decode is **bit-exact**, log clean — the reference
  decoder also reads `dct_type` on a coded `mcsel == 1` macroblock, the
  §6.3.6.3 reading this crate's parser now follows. Regenerated in
  round 480 once the encoder's reconstruction gave coded `mcsel == 1`
  co-located macroblocks their §7.6.9.5.1 averaged MV: the pair now
  pins that rule (the zero-vector reading misses B-VOP samples by up
  to 146). Regenerated again once the field estimator searched the
  full field-unit Table 7-9 range (erratum E4): 4611 bytes (was 4690),
  reference decode still bit-exact.

```
ffmpeg -idct faani -i enc_isb_ilaced_gmc_compat_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_isb_ilaced_gmc_compat_96x64.yuv
```

## Multi-point GMC streams (round 455 — §7.8.4 two / three warping points)

Deterministic builds of `tests/encoder_gmc_affine.rs` over a
bilinearly-resampled synthetic scene (fixed-point, byte-deterministic),
reference decodes as above (`ffmpeg -idct faani -i <name>.m4v …`):

* `enc_is_gmc3_zoom_96x64` — I + 3 S(GMC) with
  `no_of_sprite_warping_points == 3`: a 1/64-per-frame zoom about the
  origin plus a (2, 1)-pel pan, fitted (mode-seeded robust least
  squares + coordinate-descent refinement on the decoder's own warp)
  into the affine trajectory `[[4, 2], [3, 0], [0, 2]]`; every
  trajectory component and every §7.8.7.3 averaged MV is
  non-negative. **Bit-exact** — the first black-box confirmation of the
  §7.8.4 virtual-sprite-point / §7.8.5 three-point affine arithmetic
  on both sides.
* `enc_is_gmc2_rot_96x64` — I + 3 S(GMC) with two warping points: a
  1/32-rad rotation plus a (1, 0)-pel pan → `[[2, 0], [0, 6]]`. The
  S pictures are bit-exact (the §7.8.5 similarity warp included, with
  averaged MVs of both signs feeding the §7.6.5 predictors of the local
  macroblocks); the I picture carries **one** near-tie IDCT sample,
  which the first S picture copies through its GMC prediction — 2 of
  36864 samples, ±1.

## AC-VLC intra DC + S(GMC) packet HEC (round 455)

* `enc_isb_dcvlc7_hec_96x64` — registry build (`gmc`, `fcode 2`, `bf 1`,
  `dc-vlc-thr 7`, `mb-aq`, `packet-bits 500`, qp 6; deterministic from
  `tests/encoder_dc_vlc_hec.rs`): every intra DC differential rides the
  AC VLC (`intra_dc_vlc_thr == 7`, Table 6-25), the S(GMC)-VOP video
  packets alternate `header_extension_code` bodies that restate
  `sprite_trajectory()` (§6.2.5), B-VOPs between the S anchors. The
  reference decode is **bit-exact**. Black-box observation: the
  reference decoder logs an `untested` notice while consuming the
  S-VOP packet HEC bodies (its own path-coverage remark — the pair
  without S-VOP HEC bodies decodes silently), yet reconstructs every
  sample identically. Regenerated in round 480 for the §7.6.9.5.1
  coded-GMC co-located averaged MV (the zero-vector reading misses
  the B-VOPs by up to 184 per sample).

```
ffmpeg -idct faani -i enc_isb_dcvlc7_hec_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_isb_dcvlc7_hec_96x64.yuv
```

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

## Budget-driven rate control (round 458)

Registry builds from `tests/encoder_rate_blackbox.rs` (96×64, 25 fps,
`bitrate 300000`, `gop-size 12`, `bf 2`, `mb-aq`, `four-mv`, seed qp
6; a panning texture whose second half gains a second texture layer):
every I/P macroblock's `dquant` and every coded non-direct B
macroblock's `dbquant` is driven by the GOP budget through the
per-macroblock regulator, so the running quantiser moves in both
directions inside every VOP.

* `enc_ipb_rcbudget_96x64` — one-pass budget mode: 301 692 b/s
  (×1.006 of the target), luma PSNR 33.49 dB. **Bit-exact** reference
  decode.
* `enc_ipb_rc2pass_96x64` — two-pass (the first run's per-VOP
  statistics fed back through `with_first_pass_stats`): 299 875 b/s
  (×1.000), luma PSNR 33.62 dB. **Bit-exact** reference decode.

* `enc_ilaced_rcbudget_compat_96x64` — the same budget mode on an
  interlaced VOL (`interlaced`, `ecosystem-compat`, `qpel`): budget-
  driven `dquant` on field-DCT / field-predicted anchors, `dbquant` on
  interlaced B macroblocks; 303 492 b/s (×1.012), luma PSNR 33.17 dB.
  **Bit-exact** reference decode.

```
ffmpeg -idct faani -i enc_ipb_rcbudget_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_ipb_rcbudget_96x64.yuv
ffmpeg -idct faani -i enc_ipb_rc2pass_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_ipb_rc2pass_96x64.yuv
ffmpeg -idct faani -i enc_ilaced_rcbudget_compat_96x64.m4v -f rawvideo -pix_fmt yuv420p enc_ilaced_rcbudget_compat_96x64.yuv
```

## S(GMC) + data partitioning (round 458 — black-box observation, no pair)

`tests/encoder_gmc_dp.rs` builds S(GMC)-VOPs on the §6.2.5.3
`data_partitioned_p_vop()` layout (partition 1: `not_coded`, `mcbpc`,
`mcsel` on `derived_mb_type < 2`, `motion_coding()` only when
`mcsel == 0`). The reference decoder does **not** follow those
clauses: on a data-partitioned S-VOP it desynchronises at the very
first macroblock (measured per macroblock: every macroblock of the
first S picture differs, intra ones included, while the leading I-VOP
is bit-exact and the same encoder's combined-syntax S(GMC) streams —
`enc_isb_*` — stay bit-exact; packets, RVLC, warping-point count and
B-VOPs change nothing). No reference decode is therefore committed
for the combination; the crate keeps the printed syntax on both sides
and validates it through its own decoder walk.

## IDCT half-integer rounding (round 480)

* `ties_q6_96x64` — plain reference-encoder stream at `-qscale:v 6`
  (I P P P P I P P P P). At this quantiser the luminance `dc_scaler`
  is 12 (Table 7-1), so every DC-only block with an odd `QF[0][0]`
  reconstructs `F[0][0] / 8` to an exact half-integer. The reference
  decode resolves those halves to the even neighbour, the default
  decode away from zero (both inside the Annex A tolerance): the
  ecosystem-compat decode is **bit-exact**, the default one differs on
  8262 samples, every one by exactly 1 (`tests/conformance.rs`
  `compat_idct_rounds_exact_halves_to_even`).

```
ffmpeg -f lavfi -i testsrc2=s=96x64:r=25 -frames:v 10 -c:v mpeg4 -g 5 -bf 0 -qscale:v 6 -f m4v ties_q6_96x64.m4v
ffmpeg -idct faani -i ties_q6_96x64.m4v -f rawvideo -pix_fmt yuv420p ties_q6_96x64.yuv
```

## SHA-256

```
8d8f6b92b676319e531d431bc5c2309925b02a5635fdbd75684200e98c56409b  enc_isb_ilaced_gmc_compat_96x64.m4v
3751f69dd032455da282aa89ba4920f443a587424457ca7d0d935784ada2a523  enc_isb_ilaced_gmc_compat_96x64.yuv
40c860b1c1fbf46c258c3287ae9f5c8c84061f3350e4e2020cbb128e1315466f  enc_ilaced_fieldrange_ipbb_64x160.m4v
512d6b0d64d318bfb163748f96397a96473a1976c5257a605fcfae6e69e61839  enc_ilaced_fieldrange_ipbb_64x160.yuv
41adcc96bbe4a15a873d65f7fb59a8d6cccb6285db7998744a5c29d1b3f3a5f9  field_mv_range_fcode1.m4v
e069aca5cee1846ef2f50b7d4e121f11bc3dd97eec4a2968dcac531dc2daf412  field_mv_range_fcode1.yuv
50a58e7c05fc7fe780038943dbab8fd2dff762ffdbe87e72b76926d56834a0e9  field_mv_range_fcode2.m4v
f8d051fbf7abd41e1d16e76ee900f9b15cc5c290b610a700b830026b7eac9283  field_mv_range_fcode2.yuv
f0fd66621b8553d5bf78a9d8e307b5508e722f81689179916def8ad7a815b821  field_mv_range_trace.csv
8ce2e247f3d0ba6d82400dfe5e48a482ad8ddda2e017649786e2aa52b03c879e  enc_isb_dcvlc7_hec_96x64.m4v
9f720ffe695eb7b2f4f1594dc3092420aa636ffde91de470bc0816ca99c972e4  enc_isb_dcvlc7_hec_96x64.yuv
1b3f275670116b291b656d64a6649ebd7bff87a9bb169ac5ea045ce9c5ed1777  enc_is_gmc3_zoom_96x64.m4v
2f0677e7da0fd2a0b743dc098e6931ebb53dc93c1e49e7e7688adbcd89ab3e1b  enc_is_gmc3_zoom_96x64.yuv
ef61f2af0ce809fab9332d8fba8419d7751c0ba99f84b7b6bbdc47b28eac6483  enc_is_gmc2_rot_96x64.m4v
8ee02b80cceda915bd5aa62a577959561f167ff044f4f7f0737e24c940ca8f8c  enc_is_gmc2_rot_96x64.yuv
7a9810ee7120e8825d1882e5cb01be56af98cf4e2270b6a76b5206d0910c24a5  enc_sh_ippip_176x144.h263
85f4947b888350fc63af9fc7279ada1574a82ac2696c3e2b3b87fd55982101d0  enc_sh_ippip_176x144.yuv
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
01c5624f33678e3c1bdae08826f978f994a44d478e84bb554f4d57786cba9dc5  enc_ip_m2_64x64.m4v
c1f94ed91043cba8289b609a05af186aded43393fe4e820cd6c08e535e9ee464  enc_ip_m2_64x64.yuv
d9a3d9327f828e2f9fe9a6c59adce0ccd4d96232d88c239cc64a0289c93d6aa6  enc_ip_4mv_64x64.m4v
889ca3df99d8d861643aaf7422ee40aed5572d0bd9dffac9856c68649c1f6de9  enc_ip_4mv_64x64.yuv
6ae22214447cba24b298b13f47c889fbeb497229a7f333c51d11000f7a8d7bce  enc_ip_qpel_64x64.m4v
0cadbafa74fe72bb32b018cd65ec5cbf552c74371262121f876d2ba0ff6e6fff  enc_ip_qpel_64x64.yuv
f63816ecc0a2ffbf2613a95ca054181593b316c5b15a82ad7c5c06b0a99f3629  enc_ip_qpel4mv_64x64.m4v
1aa737a90dfcdb3c9703a5779ff93da9f03222790ddcb3c078f924ab55ee9167  enc_ip_qpel4mv_64x64.yuv
2ff701132e4e99c7ea838d8b5ee653d5c109c362066fbe366b6a28da9e0f698b  enc_ipb_64x64.m4v
146365e08903274e2d6e8b520894c5f8255ee46ce49d2a060a883e822a291d62  enc_ipb_64x64.yuv
748c3ad1fa272cd5f5423d50e95998d5e17bf44ae4dce73def3c393cd05d99f6  enc_ipb_qpel4mv_64x64.m4v
c6081d372a4e0a6fc0e7b4e3037a692e5bee2fe07ed44abdc359599c8b24388c  enc_ipb_qpel4mv_64x64.yuv
e41283803b7043ccfe2c43bfe572251f8ef1022f02f795ac70dc0dd5d384b394  enc_ip_fcode2_96x64.m4v
7088e0d866ef010338d48af58c4825388b323f0cf4f5d0f2ff7e9400bfbc24f4  enc_ip_fcode2_96x64.yuv
a522305b45c16c73d13c63eb5adbddfc60dbf1337922c5e14016decad2d21f94  enc_ipb_fcode3_qpel4mv_96x64.m4v
ad7e4a4a3855efac4737392dbbad792b677861ba2f24217db562e0bed66b42bf  enc_ipb_fcode3_qpel4mv_96x64.yuv
11bf8b623580577ecc24ff81f45b5b4a4c7648fc2cd3b20c606cbbf6388af6d3  enc_isb_gmc_qpel_96x64.m4v
43f6ebaabfacda901dd91b3753fd8a65f2a810d917fde076f8664972f57022d5  enc_isb_gmc_qpel_96x64.yuv
43ba930a07b45c825b5df81b66331bc1a8cfef066540ef9f22af8cd02a78d844  dec_sgmc_negamv_hp_64x64.m4v
e560f779d1d598419c80bf30629463fd1e0046834a2d0ac1ecf592ea130c8483  dec_sgmc_negamv_hp_64x64.yuv
60fd1ec7342131f55276c248c308be9f3c0d4a691d74338c77f9d7f874d31880  dec_sgmc_negamv_qp_64x64.m4v
69dd677fbcf59914484d3b9f27a67e3f7bbc2c36e45df2c688933de88db31e17  dec_sgmc_negamv_qp_64x64.yuv
7a65759f1cb5867326230b3977a1c48258a4cc641590262598b7abaa7e3e9a03  dec_sgmc_negtraj_96x64.m4v
b01b3d5e9211d36a7e8333a6c85bdbe22aa2f9e56811504f2f97f4bd6bfdfcca  dec_sgmc_negtraj_96x64.yuv
af7bc0ab7446f1be5d2ef409fc6208e3114b84040c484f54e67aa08d7bba7048  enc_ipb_aq4mv_96x48.m4v
999fd1815daf52e5ac1a61abd435905920153ca75bda9adaf6e66d04f8a31b84  enc_ipb_aq4mv_96x48.yuv
e3cb16490c82b8e725dab493f967f5037f53bbaa2413d68451c90a2d4ef105de  enc_ipb_vp_fcode2_96x64.m4v
10d165da46e34eaa7729246b06585d4047c6b9015bdf17bdc7eb667b658edd39  enc_ipb_vp_fcode2_96x64.yuv
9646c3b7d78effafb2e4ff42599a95e16b7ca465dd8b8e16a85b3bb74800bc95  enc_ip_dp_aq_96x48.m4v
bcb0b2159d42a02fd74200629d9d4c7bc66b93ec3eabf0acb7f68dfd040bea9b  enc_ip_dp_aq_96x48.yuv
abc9249d527753b3bb581d47c0139a6fa57d577700962b33884475ac85059854  enc_ipb_dprvlc_aq4mv_96x48.m4v
695d329ff9a3ae56ff09bb897e608180e60d4094afcea96eb2e6fda042bcba24  enc_ipb_dprvlc_aq4mv_96x48.yuv
6113c5cb48ffe830bc72bb40959039b20ab2e8e2af85b5ff8c63b604260c7a71  enc_ipb_rcbudget_96x64.m4v
1acb7761cc251a9181af3cc316c81f59f65e7ec8b7e40f8476e0aa15f8e0dfdd  enc_ipb_rcbudget_96x64.yuv
5b09f978e8e3120f527101912284d76370039421c3d47e07c5f62a0f55984464  enc_ipb_rc2pass_96x64.m4v
3abf35c252b873c4ad6464f51eb4741a183d14904bb01f69c7be46d20df584f5  enc_ipb_rc2pass_96x64.yuv
01cc892e7262279fc817e679c5a535504af61fb1e30e385b9a440da90089fafa  enc_ilaced_rcbudget_compat_96x64.m4v
2837aeeafaea1cdda5d65a64593d6d4bf21084440792270e73838b5c4ce232d6  enc_ilaced_rcbudget_compat_96x64.yuv
b7ea047a2b675e23c06baf2dfc6f00cc60fa105f37b89bd609c5578e3888c854  ties_q6_96x64.m4v
aadaa7fd85f1455ad6a5a75a416c41b1906db567393439151a55803d9c699154  ties_q6_96x64.yuv
```

(Note: `aic_ipb_64x64.yuv` and `altscan_ipb_64x64.yuv` are
byte-identical — the two encodes quantise the same source to the same
reconstruction — as are the two interlaced-intra decodes.)

## Encoder-produced streams regenerated (round 491)

The P- and B-VOP motion search became predictor-seeded and rate-aware,
constant-quantiser B-VOPs take a coarser quantiser and the `mb-aq`
classes scale with the quantiser, so every encoder-produced stream
that contains P- or B-VOPs changed. All `enc_*` pairs listed as
modified in that round's commit were rebuilt with
`OXIDEAV_MPEG4VIDEO_WRITE_FIXTURES=1 cargo test` and their reference
decodes regenerated with the commands of their own sections (`-idct
faani`; `-f h263` for the short-header pair). `enc_sh_ippip_176x144`
is now built at quantiser 6: at 5 the new stream has three samples on
an inverse-transform near-tie, which the reference decode's
single-precision transform rounds the other way. The agreement
statements of the individual sections hold unchanged for the
regenerated pairs (`cargo test` asserts each of them).

## `integer-idct-vectors/vectors.txt`

Byte-for-byte copy of the staged
`docs/video/mpeg4-visual/fixtures/integer-idct-vectors/vectors.txt`
(format and block classes in that directory's `notes.md`): 64
input / expected-output block pairs of the integer row–column inverse
transform, black-box input/output data, raw 16-bit results without
output saturation. Consumed by `tests/variant_rc_vectors.rs`.

```
b55e863cd3367a77496f95cc9aa57a171eaa89edd6bc6c4cf696dc2efc5da45e  integer-idct-vectors/vectors.txt
```

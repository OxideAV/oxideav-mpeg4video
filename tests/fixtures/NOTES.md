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
```

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

## SHA-256

```
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
```

(Note: `aic_ipb_64x64.yuv` and `altscan_ipb_64x64.yuv` are
byte-identical — the two encodes quantise the same source to the same
reconstruction — as are the two interlaced-intra decodes.)

# oxideav-mpeg4video

[![CI](https://github.com/OxideAV/oxideav-mpeg4video/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-mpeg4video/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-mpeg4video.svg)](https://crates.io/crates/oxideav-mpeg4video) [![docs.rs](https://docs.rs/oxideav-mpeg4video/badge.svg)](https://docs.rs/oxideav-mpeg4video) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust clean-room decoder for MPEG-4 Part 2 Video (ISO/IEC 14496-2 /
MPEG-4 Visual / ASP) for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework. This
is the standard MPEG-4 Part 2 bitstream (XVID / DIVX / DX50 / FMP4 /
MP4V) — *not* the pre-standard Microsoft MPEG-4 family, which lives in
[`oxideav-msmpeg4`](https://github.com/OxideAV/oxideav-msmpeg4).

## Status

Clean-room rebuild: **the decoder is now a working end-to-end codec**.
`decoder::Mpeg4VideoDecoder` consumes a raw MPEG-4 Visual elementary
stream (start-code-delimited §6.2.1 units) and emits display-order
frames: start-code scan → §6.2 configuration headers → §6.2.4 GOV /
§6.2.5 VOP headers → the `vop_decode` bitstream macroblock walks (I /
P / B / S(GMC), with §6.2.5.2 video-packet resync handling) → the
§7.6.1 reference-frame chain → §6.1.3.8 display reorder. The §6.3.5
time model derives each B-VOP's §7.6.7 `TRB`/`TRD` from the composed
VOP times. **Real-stream conformance is now bit-exact**: against a
black-box reference decode using the ideal (floating-point) Annex A.1
IDCT, eleven reference-encoder-produced streams — intra-only, I+P,
I/P/B, qpel-IP, qpel-IPB, qpel+4MV-IPB, AC-prediction IPB,
progressive alternate-scan IPB, data-partitioned IPB, 176×144 IPB
with video packets, and an interlaced I+P — decode **bit-for-bit
identically**; four more match up to a documented budget of ±1
"near-tie" samples (the oracle's single-precision IDCT crossing a
rounding boundary the ideal value sits within ~1e-5 of), and the
remaining three carry bounded, root-caused oracle deviations
(`tests/conformance.rs`, fixture provenance + SHA-256 in
`tests/fixtures/NOTES.md`). Two of those deviations are
**spec-vs-ecosystem divergences** now covered by the opt-in
[ecosystem-compat mode](#compatibility-modes): with it enabled the
method-1 (`mpeg_quant`) stream and the interlaced-I/P/B stream
collapse to near-tie-only differences, and 29 of the corpus' 30
§7.7.2.2 interlaced-direct macroblocks decode bit-exact. The crate
**registers
into the runtime codec registry** (`mpeg4video`, FourCCs XVID / DIVX /
DX50 / FMP4 / MP4V / M4S2 + MP4 OTI `0x20`) via a packet decoder that
supports extradata priming, seeks (`reset`), and the pixel-count DoS
cap; `decoder::make_decoder` is the direct factory endpoint. The
bitstream walks cover the rectangular path in **half- and
quarter-sample** modes (§7.6.2.1 / §7.6.2.2 with the Figure 7-30
per-block boundary mirroring), with **§7.6.6 overlapped motion
compensation** (`obmc_disable == 0`), **per-sub-block §7.6.9.5.2
direct-mode co-located MVs**, **presentation timestamps** (container
pts through the §6.1.3.8 reorder + §6.3.5 tick times), and
**interlaced tools**: §7.7.1 field DCT + alternate vertical scan,
§7.7.2.1 field-predicted P macroblocks (CASE 1/2/3 predictors, field
MC, Div2Round field chroma), and §7.7.2.2 interlaced B-VOPs (field
forward / backward / bidirectional through the Table 7-15 **unified
four-PMV bank** shared by frame- and field-predicted macroblocks +
interlaced direct mode). The §7.7.2.2 printed pseudo code carries
several defects; this decoder implements the corrected readings
(erratum E1 backward-call field ordering, E2 `MVD[0]` gate, same-slot
`PMV[3].x` field-backward reconstruction, §7.6.5 intra neighbours as
valid zero MV candidates, and the f_code==1 direct-delta rule), each
**black-box-arbitrated** against a conformant interlaced direct-mode
B-frame stream (docs fixture, refs #176), plus the field-grid
vertical arithmetic the §7.7.2 evenness invariant requires. The
long-standing interlaced-direct deviation is now **root-caused**: an
exhaustive per-macroblock-field pixel search (12 uniquely-determined
macroblock-fields, all consistent) proves the reference decoder
evaluates the corrected §7.7.2.2 pseudo code with the co-located
field motion vectors substituted by zero (its derived vectors reduce
to `mvf[i] = mvb[i] = MVD[0]` on the field grid, forward reference
fields still following the co-located selections); §7.7.2.2 derives
"from the forward field motion vectors of the co-located future
reference VOP", so this decoder keeps the spec's `MV[i]` term **by
default** and the affected macroblocks carry a bounded, documented
envelope; the [ecosystem-compat mode](#compatibility-modes) opts into
the observed zero-co-located derivation instead.
**§6.2.5.3 data-partitioned I-/P-VOPs
decode end-to-end** (`decode_i_vop_macroblocks_dp` /
`decode_p_vop_macroblocks_dp`: per-packet dc_marker / motion_marker
partition structure, §E.1.2 prediction resets, header-partition intra
DC feeding the texture partition, Table B.23 RVLC texture when
`reversible_vlc == 1`), with B-VOPs on the combined syntax per the
§6.2.5.3 NOTE. Twenty-two black-box conformance fixtures (intra / IP /
IPB / qpel {IP, IPB, +4MV} / AC-prediction / alternate-scan /
data-partitioned IPB / QCIF-resync IPB / interlaced × {intra,
alt-scan, IP, IP-motion, IPB, qpel-IP, direct-B 176×144} / mpeg-quant
× {plain, qpel+4MV, interlaced-B, data-partitioned}) are asserted
bit-exact, near-exact, or envelope-bounded as described above — each
also pinned under the compat mode. There is no encoder.

## Compatibility modes

The decoder's default behaviour is always the **literal ISO/IEC
14496-2 text**. For two clauses, black-box pixel comparison against
reference decodes of conformant streams shows the deployed decoder
ecosystem behaves differently (no implementation source was consulted
— outputs only). The opt-in **ecosystem-compat** mode reproduces the
observed behaviour bit-for-bit so real-world files can be matched
exactly; it covers exactly these two divergences (`crate::compat`
module docs carry the full write-up):

1. **§7.7.2.2 interlaced direct mode** — spec: the four field MVs are
   derived from the co-located future macroblock's forward field
   motion vectors (`MV[i]`); observed: the same erratum-corrected
   derivation with those vectors read as **zero**, i.e.
   `mvf[i] = mvb[i] = MVD[0]` on the field grid (forward reference
   fields still follow the co-located selections).
2. **§7.4.4.5 mismatch control** — spec: the method-1 sum-parity
   toggle of `F[7][7]` applies to every block; observed: **non-intra
   blocks only**.

Selection is wired through every decode surface:

* typed: `Mpeg4VideoDecoder::with_options(DecodeOptions::ecosystem())`
  (default `new()` / `DecodeOptions::spec()` is the literal spec);
  every `vop_decode` macroblock walk takes the same `DecodeOptions`;
* registry / options bag: key **`ecosystem-compat`** (bool, default
  `false`) on `CodecParameters::options`, declared by the
  `Mpeg4DecoderOptions` schema and parsed in `make_decoder`; the
  selection survives `Decoder::reset`.

Measured effect (`tests/conformance.rs` `compat_*` pins): the
`mpeg_quant` I/P/B stream collapses from a 3062-sample envelope to 4
near-tie samples, the interlaced I/P/B stream from 650 samples to 7
near-ties, and the 176×144 interlaced-direct stream from 6202 samples
(max 114) to 2777 (0.29 %, max 64) with 29/30 interlaced-direct
macroblocks bit-exact — the single residual macroblock's oracle
reconstruction is uniquely pinned by exhaustive per-field search and
is consistent with the *printed* §7.7.2.2 formulas evaluated from a
co-located field-MV state that differs from the bitstream-reconstructed
one (the co-located anchor region is flat, so that internal state is
not determinable from pixels; a docs-fixture trace ask is filed).
Streams exercising neither clause decode sample-identically in both
modes (asserted).

## What works today

- **End-to-end elementary-stream decode** (`decoder`):
  [`Mpeg4VideoDecoder`] (start-code scan, §6.3.5 VOP time model,
  §7.6.7 `TRB`/`TRD` derivation, `vop_coded == 0` forward-reference
  copies) plus the registry-facing [`Mpeg4PacketDecoder`] /
  [`make_decoder`] factory and a live `register()` (id `mpeg4video`,
  the common FourCC tags + MP4 OTI `0x20`).
- **Bitstream-driven macroblock walks** (`vop_decode`):
  [`decode_i_vop_macroblocks`] (running quantiser + Table 6-25
  `use_intra_dc_vlc` + per-block Figure 7-5 predictor resolution),
  [`decode_p_vop_macroblocks`], [`decode_s_gmc_vop_macroblocks`]
  (`mcsel` routing + §7.8.7.3 averaged-MV predictor recording), and
  [`decode_b_vop_macroblocks`] (§6.2.6 `co_located_not_coded`
  zero-bit macroblocks) — each with §6.2.5.2 video-packet resync
  handling (probe, header decode, §E.1.2 predictor/quantiser reset).
- **Bit-exact real-stream conformance** against a black-box reference
  decode with the ideal floating-point IDCT (`tests/conformance.rs`,
  provenance in `tests/fixtures/NOTES.md`): eleven streams bit-exact,
  four near-exact (±1 near-tie IDCT samples), three bounded with
  root-caused oracle deviations — plus the full `compat_*` pin set
  under the [ecosystem-compat mode](#compatibility-modes) (two of the
  bounded streams collapse to near-tie-only).
- **Configuration headers** (§6.2): Visual Object Sequence
  (`0x000001B0` + profile/level), Visual Object (`0x000001B5`, verid,
  video-signal-type, colour description), and Video Object Layer
  (`0x000001Bx`) — shape (rectangular), aspect ratio, dimensions,
  time-increment resolution, `vol_control_parameters` / VBV, the §6.2.3
  trailing flags (`interlaced`, `obmc_disable`, `sprite_enable`,
  `quant_type`, `quarter_sample`, `data_partitioned` / `reversible_vlc`,
  `scalability`, …), and the `quant_type == 1` matrix-load bodies.
- **Frame headers**: Group-of-VOP (`0x000001B3`, time code) and Video
  Object Plane (`0x000001B6`) — coding type (I / P / B / S), modulo
  time base, `vop_quant`, `vop_fcode_forward` / `_backward`, rounding
  type, and the interlaced flags.
- **Macroblock layer**: I/P-VOP macroblock-header bit-walk (mcbpc /
  cbpy / dquant / ac_pred / not-coded skip), the §6.3.6 **S(GMC)-VOP**
  macroblock layer (shares the P-VOP MCBPC table + not-coded syntax, plus
  the `mcsel` flag — GMC vs. local-MC reference selection — for inter /
  inter+q macroblocks, with the §6.3.6 implied `mcsel == 1` for a
  not-coded GMC macroblock and the §6.2.6.3 / line-11715 rule that an
  `mcsel == 1` macroblock invokes no `interlaced_information()` body),
  B-VOP header prefix (modb / mb_type / cbpb / dbquant), and the
  `interlaced_information()` body.
- **Motion vectors**: the `motion_vector()` body and MVD VLC (Table
  B.12), §7.6.3 differential reconstruction with the modulo wrap, the
  §7.6.5 median predictor with the four candidate-validity rules,
  the Figure 7-34 candidate gathering via `MvGrid`, 1-MV and Inter4V
  cardinality, chrominance-MV derivation from K luminance MVs (Tables
  7-10..7-13), and the §7.7.2.1 interlaced field-MV predictor (CASE 1 /
  2 / 3) with field-aware neighbour selection.
- **Residual + reconstruction**: intra DC prediction, AC prediction,
  the intra/inter Tcoef EVENT VLCs (Tables B.16 / B.17) with the escape
  forms, the reversible-VLC Tcoef table (Table B.23, intra + inter
  columns) with its Type-5 escape (`00001` + LAST/RUN/marker/LEVEL/marker
  + closing `0000` + sign, Tables B.24 / B.25) for the
  `reversible_vlc == 1` path in **both** the forward and the §E.1.4.4
  backward (reverse-direction) decode, zigzag / alternate scan, the
  §7.4.2 `sadct_disable == 0` modified inverse scan (`coeff_width[]`-aware
  packing with the NOTE 1 zero-fill, plus the Annex A §A.3.2 I-S1
  `coeff_width[v]` / `opaque_pels` derivation from the decoded binary
  shape), §7.4 inverse
  quantisation (methods 1 and 2), the 8×8 IDCT, the Annex A §A.3.2
  inverse **shape-adaptive DCT** (SA-DCT) transform body (steps
  I-S1..I-S5: the full shape-parameter derivation `coeff_width[v]` /
  `pels_height[x]` / `shift_shape[y][x]`, the variable-length
  `coeff_width[v]`- / `pels_height[x]`-point 1-D inverse DCT kernels,
  and the I-S3 / I-S5 column / row re-shifts) reconverting the
  `PQF[v][u]` layout back to texture `f[y][x]`, the Annex A §A.4.2 inverse
  **∆DC-SA-DCT** post-processing (steps I-∆S1..I-∆S4: extract the re-scaled
  mean `F[0][0]/8` and zero `F[0][0]`, run the inverse SA-DCT body, derive
  the ∆DC correction term `corr_term = check_sum / sqrt_sum` over the
  opaque samples, and add `mean_value − corr_term/√pels_height[x]` back per
  opaque pel — the path used for intra 8×8-blocks with `opaque_pels < 64`),
  and the §7.3 `d[y][x]` reconstruction with the display clip for I-, P-,
  and inter macroblocks.
- **B-VOP prediction + reconstruction**: forward / backward /
  interpolated / direct modes, bidirectional averaging, 16×16 luma +
  8×8 Cb / Cr prediction-block generation, and the §7.6.9 → §7.3 bridge
  ([`predict_b_vop_macroblock`] packs the prediction into an
  `InterPredictionMacroblock`; [`reconstruct_b_vop_macroblock`] runs the
  full predict + §7.3 `d = p + f` add + display clip end-to-end across
  both anchor VOPs). The §7.7.2.2 **interlaced field-prediction** B-VOP
  modes (field forward / backward / bidirectional) reconstruct to pixels
  via the `bvop_field_motion` module: the §7.7.2.2 four-PMV bank produces
  the top/bottom field MVs, [`field_forward_prediction`] /
  [`field_backward_prediction`] / [`field_bidirectional_prediction`] drive
  the §7.7.2.1 `field_motion_compensate_one_reference` per active
  direction (half- or quarter-sample luma), and the bidirectional case
  averages forward + backward with the §7.7.2.2 `(fwd + bak + 1) >> 1`
  rounding.
- **Frame-level B-VOP motion-vector decode driver** ([`BVopMvDriver`],
  `bvop_mv` module): the B-VOP analogue of the P-VOP [`MvDriver`].
  `decode_macroblock` decodes one macroblock's §6.2.6 header + motion
  bodies, resolves the §7.6.9 prediction mode, and reconstructs the
  forward / backward MVs against the §7.6.8 running per-direction
  predictor bank (reset per row via `start_row`, updated only by the
  matching direction; direct mode uses predictor zero + f_code 1 and
  §7.6.9.5.2 TRB/TRD scaling). `decode_vop_motion` walks a full
  progressive B-VOP in raster order with the row-reset threading built
  in, returning one [`BVopMbDecode`] per macroblock; the per-MB
  §7.6.9.5.1 / §7.6.9.6 co-located anchor state is supplied via a
  [`CoLocatedAnchor`] closure. [`BVopMbDecode::reconstruct`] then bridges
  the decoded motion straight into [`reconstruct_b_vop_macroblock`]. The
  §6.2.6 `modb "1"` vs `"01"` discriminator is resolved via
  [`BVopMbHeader::mb_type_present`]. [`BVopMvDriver::decode_vop`] now
  threads the §6.2.6 / §7.4 residual (texture) decode that follows each
  macroblock's motion bodies into the same raster loop: it applies each
  macroblock's `dbquant` (Table 6-33, §6.3.6) to a running quantiser
  scale and consumes the inter residual gated by the macroblock's
  `cbpb` (via [`decode_b_vop_inter_macroblock`] / [`cbpb_pattern_code`]),
  returning one [`BVopMbTexturedDecode`] (motion + residual + quantiser
  scale) per macroblock — ready to feed [`BVopMbDecode::reconstruct`].
- **GMC (global motion compensation)** end-to-end for rectangular
  S(GMC)-VOPs: the §6.2.3 `sprite_enable == "GMC"` VOL body
  (`no_of_sprite_warping_points`, `sprite_warping_accuracy`,
  `sprite_brightness_change`), the §6.2.5 `sprite_trajectory()` syntax
  (`warping_mv_code` VLC, Table B.34, → `du[i]`/`dv[i]`), the §7.8.4
  sprite reference-point + virtual-point geometry, the §7.8.5 warping
  transform `(F,G)`/`(Fc,Gc)` for 0/1/2/3 warping points (stationary /
  translation / affine — perspective is disallowed under GMC but
  supported for static sprites via `perspective_warp`), and the
  §7.8.6 sample reconstruction that bilinearly warps a reference VOP
  into a 16×16 luma / 8×8 chroma GMC prediction block with
  `vop_rounding_type` control and §7.6.4 edge clamping.
- **Frame-level decode pipeline** (§7.6.1 / §6.1.3.8): the
  [`framestore`] decoded-picture buffer — [`DecodedFrame`] owns one VOP's
  three 4:2:0 planes, blits each [`ReconstructedMacroblock`] into the
  macroblock grid, and hands out [`ReferenceVop`] plane views; [`FrameStore`]
  threads the §7.5.2.1.2 forward (past) + backward (future) anchor chain
  (`push_anchor` advances on I/P/S-VOPs, B-VOPs never enter the chain),
  selecting the single P/S reference and the bracketing B-VOP anchor pair.
  The [`frame_decode`] assemblers reconstruct a complete VOP against those
  references and blit it: `assemble_p_vop_frame` / `decode_p_vop` /
  `decode_i_vop` (forward reference), `assemble_b_vop_frame` (bracketing
  anchors), `assemble_s_gmc_vop_frame` (§7.8.7.1 per-MB warped/local
  `mcsel` selection). The [`sequence::SequenceDecoder`] applies §6.1.3.8
  VOP reordering — a one-slot anchor delay that turns a coding-order VOP
  stream into display order (`1I 4P 2B 3B` → `1I 2B 3B 4P`).
- **Half-sample / quarter-sample** motion compensation, OBMC, and the
  padding stages (sample / vertical / extended / interlaced).
- **RVLC error recovery — now driven end-to-end**: the §E.1.4.4.2.1
  two-way strategy selection — the Strategy 1–4 arbitration
  (`RvlcArbitration::select`) that picks how many macroblocks to keep
  from the forward decode at the head and from the backward decode at
  the tail, from the `L1+L2 >= L` / `N1+N2 >= N` predicates, the `f_mb` /
  `b_mb` step-inverse counters, and the threshold `T = 90` — plus the
  §E.1.4.4.2.2 intra-MB concealment pass (`displayed_mbs`). These were
  composable pieces; [`recover_video_packet_dct`] now assembles them
  into the actual recovery walk: it forward-decodes a video packet's
  DCT-coefficient region macroblock-by-macroblock (per a
  [`MbBlockLayout`] giving each MB's coded blocks + Tcoef tables),
  tracking per-MB cumulative bit costs `L1` / `N1`; on a §E.1.4.4.1
  forward error it backward-decodes from the packet end (segmenting
  EVENTs into blocks on the `LAST` flag via a non-consuming peek over a
  `Clone`d `BackwardBitReader`), gathers `L2` / `N2`, runs the
  arbitration, and returns a `RvlcRecovery::Recovered`.
  [`RvlcRecovery::stitch`] then collapses the recovery into the final
  per-macroblock decode set — applying the keep decision (errored middle
  discarded) and the §E.1.4.4.2.2 INTRA concealment.
- **§6.2.5.3 data partitioning**: [`parse_data_partitioned_i_vop`] /
  [`parse_data_partitioned_p_vop`] walk the rectangular data-partitioned
  I-/P-VOP layouts — partition 1 (`mcbpc` + `dquant` + intra-DC, or
  `not_coded` + `mcbpc` + `mcsel` + `motion_coding`) to the §6.3.5
  19-bit `dc_marker` / 17-bit `motion_marker`, then partition 2
  (`ac_pred_flag` + `cbpy` [+ `dquant` + intra-DC for P]) — and return
  the bit offset of the partition-3 `block()` texture region.
  [`use_intra_dc_vlc`] transcribes the Table 6-25 derivation; the
  [`mb_block_layout`] bridge turns a parsed MB into the [`MbBlockLayout`]
  the RVLC driver consumes, closing the data-partitioned bitstream →
  texture-decode loop.

## Not yet supported

- Pixel-exact conformance for the **interlaced quarter-sample field
  interpolation** (`ildct + ilme + qpel` streams): isolated
  field-predicted macroblocks diverge from the reference decode, and
  an exhaustive (MV, reference-field, rounding) search shows the
  oracle's quarter-sample *field* FIR cascade itself differs from our
  §7.6.2.2-per-field reading — the fixture carries a bounded envelope
  until a dedicated arbitration pass (or a clean-room §7.7.2.1
  quarter-sample field-interpolation trace) settles the geometry.
- The §7.6.9.6 averaged-MV substitution for a skipped co-located
  S(GMC) macroblock (the walk uses the zero-vector fallback; the AMV
  is computed in the S walk but not yet retained per-macroblock).
- Encoder.
- Blitting the §E.1.4.4 RVLC recovery into a reconstructed *frame*. The
  driver is now assembled — [`recover_video_packet_dct`] detects the
  forward-decode error, runs both directions, gathers the
  `L/N/L1/L2/N1/N2` inputs, and [`RvlcRecovery::stitch`] applies the
  kept-MB decision + §E.1.4.4.2.2 INTRA concealment to produce the final
  per-macroblock EVENT set. What remains is the caller feeding the kept
  EVENT runs through the §7.4 reconstruction and writing the resulting
  pixels into the output VOP (the same per-MB residual blit the
  non-partitioned path already uses).
- The final routing of the §7.3.5 / Table 7-2 per-block transform
  selection from a *live decoded shape* inside the macroblock
  reconstruction loop. The decision rule itself is now implemented
  ([`transform_select`]: `select_transform` transcribes the three Table
  7-2 rows — 8×8-DCT for rectangular / `sadct_disable == 1` /
  `opaque_pels == 64`; ∆DC-SA-DCT for non-B intra blocks; SA-DCT for
  P-VOP inter and all B-VOP blocks — and `inverse_transform_block` /
  `select_and_inverse_transform` apply the chosen one of the three
  transform bodies). What remains is calling it from the residual loop
  with the per-block `opaque_pels` count and `f_shape` derived from the
  decoded binary shape of the current macroblock.
- The §7.8.3 low-latency static-sprite piece-update machinery. The
  **basic** static sprite (`low_latency_sprite_enable == 0`) warps the
  §7.8.2 sprite memory onto the visible VOP via the §7.8.6 static blend
  (incl. the `brightness_change_factor` post-adjustment). The
  **low-latency** syntax shell is now parsed too: `sprite_piece` decodes
  the §6.2.5.4 `decode_sprite_piece()` header (`piece_quant` /
  `piece_width` / `piece_height` / `piece_xoffset` / `piece_yoffset`),
  the Table 6-26 `sprite_transmit_mode` (stop / piece / update / pause)
  with its `do {…} while` piece loop (`drive_sprite_piece_loop`), the
  Table B.35 `brightness_change_factor()` VLC, and the composed §6.2.5
  static S-VOP block (`parse_static_sprite_vop_block` — trajectory +
  brightness + piece loop). The §7.8.5 **four-point perspective** warp
  (`perspective_warp::PerspectiveWarp`, the
  `no_of_sprite_warping_points == 4` case) is implemented and wired into
  static-sprite reconstruction (`static_sprite_luma_perspective`). The
  §7.8.3.1 / §7.8.3.2 **hole handling** is modelled by
  `SpriteObjectBuffer`: it tracks the per-macroblock `send_mb()`
  occupancy of the sprite-object grid (`object_piece_new_macroblocks`
  returns the new MBs an object-piece carries a body for, skipping holes
  already sent by an earlier piece) and validates update-pieces
  (`update_piece_refined_macroblocks` — every refined MB's object MB must
  already exist, per the §7.8.3.2 ordering rule). What remains end-to-end
  is decoding each piece's `sprite_shape_texture()` macroblock body into
  sprite memory (the object-piece I-VOP / update-piece P-VOP macroblock
  texture subset).
- Scalability enhancement layers, Studio Profile, and non-rectangular
  shapes (rejected with typed errors). GMC global-motion warping *is*
  supported; the §6.3.6 `mcsel` flag is now routed into the §7.3 recon
  loop (`s_gmc_recon::s_gmc_prediction_macroblock` selects warped vs.
  translational per-MB), and the §7.8.7.3 averaged MV predictor and the
  §7.6.8 four-PMV interlaced-B-VOP field predictor are implemented.
- Brightness change in GMC/sprite warping (`brightness_change_factor()`
  / `sprite_brightness_change == 1`) — typed-rejected, since the spec
  mandates `sprite_brightness_change == 0` under GMC.

## Provenance

Every numeric value and bit layout traces to ISO/IEC 14496-2:2004 (3rd
edition), read from the specification text staged under
`docs/video/mpeg4-visual/`. No third-party MPEG-4 source was consulted.

## License

MIT — see [LICENSE](./LICENSE).

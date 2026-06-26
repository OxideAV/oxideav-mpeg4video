# oxideav-mpeg4video

Pure-Rust clean-room decoder for MPEG-4 Part 2 Video (ISO/IEC 14496-2 /
MPEG-4 Visual / ASP) for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework. This
is the standard MPEG-4 Part 2 bitstream (XVID / DIVX / DX50 / FMP4 /
MP4V) — *not* the pre-standard Microsoft MPEG-4 family, which lives in
[`oxideav-msmpeg4`](https://github.com/OxideAV/oxideav-msmpeg4).

## Status

Clean-room rebuild in progress. The decode pipeline is implemented as a
set of composable, per-stage public modules covering configuration- and
frame-header parsing, motion-vector reconstruction, residual decode and
pixel reconstruction for I-, P-, and B-VOPs. It is not yet wired into
the runtime codec registry — `register` is a no-op placeholder, so the
codec is consumed today through its direct module APIs. There is no
encoder.

## What works today

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

- Runtime registration (`register` is a no-op) and a single top-level
  frame-decode entry point. The §7.6 **progressive P-VOP
  motion-vector + luma-prediction subsystem is now wired end-to-end**:
  [`MvDriver`] walks the macroblocks of a P-VOP in raster order —
  dispatching skipped / intra / inter / inter4v per Table B.1, gathering
  the Figure 7-34 candidates from the running [`MvGrid`], applying the
  §7.6.5 median, decoding the §6.2.6.2 MVD body, and reconstructing the
  §7.6.3 vector — and [`predict_luma_macroblock`] turns the result into
  a §7.6.2 half-sample-interpolated 16×16 luma prediction block
  (1-MV / inter4v / skipped), and [`predict_chroma_macroblock`] derives
  the §6.1.3.4 / §7.6.5 4:2:0 chroma MV (`sum / 2K`) and produces the
  8×8 Cb / Cr prediction. The §7.6.2 → §7.3 bridge is now closed at the
  macroblock level: [`predict_inter_macroblock`] packs the prediction
  into an `InterPredictionMacroblock`, and
  [`reconstruct_pvop_macroblock`] runs the full
  §7.6.2-predict + §7.3 `d = p + f` add + display clip end-to-end,
  returning a `ReconstructedMacroblock` ready to blit (verified by a
  frame-level test that drives a four-macroblock motion bitstream
  through [`MvDriver`] and reassembles a 32×32 luma / 16×16 chroma
  frame). What remains for a *full* frame decoder is threading the
  per-macroblock §7.4 residual-texture decode into the same loop (the
  residual is currently supplied by the caller) and selecting the
  reference plane per the VOP reference-frame chain. The **B-VOP motion
  subsystem is now wired one step further**: [`BVopMvDriver::decode_vop`]
  walks a progressive B-VOP in raster order with the §7.6.8 row-reset
  predictor threading **and** threads the §6.2.6 / §7.4 residual decode
  + `dbquant` running-quantiser accumulation into the same loop,
  returning one [`BVopMbTexturedDecode`] (motion + residual) per
  macroblock; [`BVopMbDecode::reconstruct`] closes the §7.6.9 → §7.3
  bridge per macroblock. The remaining work for a full B-VOP frame
  decode is reference-plane selection per the VOP reference-frame chain.
  The **interlaced field-prediction B-VOP path is now wired into the
  frame driver** for the three non-direct field modes:
  [`BVopMvDriver::decode_field_macroblock`] decodes a field-predicted
  macroblock (forward `mb_type == "0001"` / backward `"001"` /
  bidirectional `"01"`), threads the §6.2.6 field motion bodies through
  the §7.7.2.2 four-PMV bank ([`FieldPmvBank`], reset per row alongside
  the progressive predictor), and emits a [`BVopFieldMbDecode`];
  [`BVopFieldMbDecode::reconstruct`] runs the §7.7.2.2 field MC (forward /
  backward / bidirectional-average) against the six reference planes plus
  the §7.3 residual add + display clip, so an interlaced B-VOP
  field-predicted macroblock decodes to real pixels. The §7.7.2.2
  **interlaced direct** mode is now implemented as a standalone
  derivation ([`interlaced_direct_mvs`] in the `bvop_field_direct`
  module): the four derived field MVs `mvf[0..2]` / `mvb[0..2]` from the
  co-located future macroblock's two forward field MVs, the single
  transmitted `MVD[0]`, and the field-period `TRB[i]` / `TRD[i]` (the
  `2*frame_distance + δ` conversion with the Table 7-16 `δ` parity
  selection), plus [`interlaced_direct_prediction`] which runs the
  §7.7.2.2 forward + backward (mvb[1]-for-both-fields) field MC and
  averages them. **Interlaced direct mode is now wired into the frame
  driver**: [`BVopMvDriver::decode_interlaced_direct_macroblock`] parses
  the §6.2.6 header (a `modb == "1"` default direct, or an explicit Direct
  `mb_type`), reads the single `MVD[0]` body (`f_code == 1`, §7.7.2.2;
  implicitly zero for `modb == "1"`), and threads the caller-supplied
  co-located *future* P-VOP macroblock's two forward field MVs + reference
  fields ([`ColocatedFutureFieldMvs`]) and `top_field_first` through
  [`interlaced_direct_mvs`] with the driver's frame-period `TRB` / `TRD`,
  returning a [`BVopInterlacedDirectMbDecode`] whose `reconstruct` runs
  the §7.7.2.2 → §7.3 forward/backward field-MC + average + residual add
  + display clip. The caller establishes interlaced-direct applicability
  (future macroblock field-predicted) and supplies the future field MVs
  from the reference-frame chain via
  [`ColocatedFutureFieldMvs::from_field_motion`], which builds them from a
  decoded interlaced P-VOP macroblock's reconstructed forward field MVs
  ([`FieldMotionVectors`]) and its §6.3.6.3 top/bottom field references.
  A single **unified dispatch** entry
  [`BVopMvDriver::decode_interlaced_macroblock`] parses the §6.2.6 header
  once and routes each macroblock to the progressive / field-prediction /
  interlaced-direct path automatically (driven by a per-MB
  [`BVopInterlacedAnchor`] and returning a tagged [`BVopInterlacedMb`]),
  so an interlaced B-VOP frame loop never has to peek the header to
  pre-select a decode path. [`BVopMvDriver::decode_interlaced_vop`] then
  closes the raster loop — the interlaced analogue of `decode_vop`: it
  walks the interlaced B-VOP in raster order driving each macroblock
  through the unified dispatch, resets both the progressive and §7.7.2.2
  four-PMV predictors per row, applies each `dbquant`, and threads the
  §7.4 residual decode into the same loop, returning one
  [`BVopInterlacedTexturedDecode`] per macroblock, whose
  [`BVopInterlacedTexturedDecode::reconstruct`] dispatches on the path
  variant (progressive `BVopAnchorPlanes` / field & interlaced-direct
  `BVopFieldReferences`) to close the §7.6.9 / §7.7.2.2 → §7.3 loop to
  pixels. What remains for a full interlaced B-VOP *frame* decode is the
  caller supplying each macroblock's co-located future/anchor state from
  the reference-frame chain and blitting the per-MB reconstruction.
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
  static-sprite reconstruction (`static_sprite_luma_perspective`). What
  remains end-to-end is decoding each piece's `sprite_shape_texture()`
  body into sprite memory (the object-piece I-VOP / update-piece P-VOP
  macroblock subset) and the §7.8.3.2 hole-handling.
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

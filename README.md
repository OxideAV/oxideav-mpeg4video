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
  cbpy / dquant / ac_pred / not-coded skip), B-VOP header prefix (modb /
  mb_type / cbpb / dbquant), and the `interlaced_information()` body.
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
- **B-VOP prediction**: forward / backward / interpolated / direct
  modes, bidirectional averaging, and 8×8 luminance prediction-block
  generation.
- **GMC (global motion compensation)** end-to-end for rectangular
  S(GMC)-VOPs: the §6.2.3 `sprite_enable == "GMC"` VOL body
  (`no_of_sprite_warping_points`, `sprite_warping_accuracy`,
  `sprite_brightness_change`), the §6.2.5 `sprite_trajectory()` syntax
  (`warping_mv_code` VLC, Table B.34, → `du[i]`/`dv[i]`), the §7.8.4
  sprite reference-point + virtual-point geometry, the §7.8.5 warping
  transform `(F,G)`/`(Fc,Gc)` for 0/1/2/3 warping points (stationary /
  translation / affine — perspective is disallowed under GMC), and the
  §7.8.6 sample reconstruction that bilinearly warps a reference VOP
  into a 16×16 luma / 8×8 chroma GMC prediction block with
  `vop_rounding_type` control and §7.6.4 edge clamping.
- **Half-sample / quarter-sample** motion compensation, OBMC, and the
  padding stages (sample / vertical / extended / interlaced).
- **RVLC error recovery**: the §E.1.4.4.2.1 two-way strategy selection —
  the Strategy 1–4 arbitration (`RvlcArbitration::select`) that picks
  how many macroblocks to keep from the forward decode at the head and
  from the backward decode at the tail, from the `L1+L2 >= L` /
  `N1+N2 >= N` predicates, the `f_mb` / `b_mb` step-inverse counters, and
  the threshold `T = 90` — plus the §E.1.4.4.2.2 intra-MB concealment
  pass (`displayed_mbs`).

## Not yet supported

- Runtime registration (`register` is a no-op) and a single top-level
  frame-decode entry point.
- Encoder.
- The end-to-end wiring of the §E.1.4.4 two-way RVLC error recovery: the
  forward / backward Tcoef decodes (§E.1.4.4.1) and the §E.1.4.4.2.1
  Strategy 1–4 arbitration + §E.1.4.4.2.2 intra-MB concealment are all
  implemented as composable pieces, but the video-packet driver that
  detects the forward-decode error, runs both directions, gathers the
  `L/N/L1/L2/N1/N2` inputs, and applies the kept-MB decision to the
  reconstructed frame is not yet assembled.
- The §7.3.5 / Table 7-2 per-block transform-selection wiring (8×8-DCT vs
  SA-DCT vs ∆DC-SA-DCT) into the reconstruction loop. All three transform
  bodies are now implemented — the 8×8 IDCT, the inverse SA-DCT (Annex A
  §A.3.2 steps I-S1..I-S5), and the inverse ∆DC-SA-DCT (Annex A §A.4.2
  steps I-∆S1..I-∆S4) — but the per-block decision that picks which one to
  apply (from the shape's `opaque_pels` count and `sadct_disable`) is not
  yet routed into the residual-reconstruction path.
- Static-sprite bodies (the §7.8.2 sprite-object buffer + §7.8.3
  piece-update / `sprite_transmit_mode` machinery), scalability
  enhancement layers, Studio Profile, and non-rectangular shapes
  (rejected with typed errors). GMC global-motion warping *is* now
  supported (see "What works today"); the remaining GMC wiring is the
  per-macroblock `mcsel`-gated routing of the warped prediction into
  the §7.3 reconstruction loop (the warp generator + §7.8.7.3 averaged
  MV predictor are implemented as composable pieces).
- Brightness change in GMC/sprite warping (`brightness_change_factor()`
  / `sprite_brightness_change == 1`) — typed-rejected, since the spec
  mandates `sprite_brightness_change == 0` under GMC.

## Provenance

Every numeric value and bit layout traces to ISO/IEC 14496-2:2004 (3rd
edition), read from the specification text staged under
`docs/video/mpeg4-visual/`. No third-party MPEG-4 source was consulted.

## License

MIT — see [LICENSE](./LICENSE).

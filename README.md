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
  `PQF[v][u]` layout back to texture `f[y][x]`, and the §7.3 `d[y][x]`
  reconstruction with the display clip for I-, P-, and inter
  macroblocks.
- **B-VOP prediction**: forward / backward / interpolated / direct
  modes, bidirectional averaging, and 8×8 luminance prediction-block
  generation.
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
- The ∆DC-SA-DCT extension (Annex A §A.4: the DC-separation /
  ∆DC-correction pre- and post-processing steps used in intra-coded
  8×8-blocks with `opaque_pels < 64`). The base inverse SA-DCT transform
  body (§7.3.5 / Annex A §A.3.2 steps I-S1..I-S5) it builds on **is**
  implemented; only the ∆DC wrapper and the §7.3.5 / Table 7-2
  per-block transform-selection wiring (8×8-DCT vs SA-DCT vs
  ∆DC-SA-DCT) into the reconstruction loop remain.
- Sprite / GMC bodies, scalability enhancement layers, Studio Profile,
  and non-rectangular shapes (rejected with typed errors).

## Provenance

Every numeric value and bit layout traces to ISO/IEC 14496-2:2004 (3rd
edition), read from the specification text staged under
`docs/video/mpeg4-visual/`. No third-party MPEG-4 source was consulted.

## License

MIT — see [LICENSE](./LICENSE).

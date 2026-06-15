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
  `reversible_vlc == 1` path, zigzag / alternate scan, §7.4 inverse
  quantisation (methods 1 and 2), the 8×8 IDCT, and the §7.3 `d[y][x]`
  reconstruction with the display clip for I-, P-, and inter
  macroblocks.
- **B-VOP prediction**: forward / backward / interpolated / direct
  modes, bidirectional averaging, and 8×8 luminance prediction-block
  generation.
- **Half-sample / quarter-sample** motion compensation, OBMC, and the
  padding stages (sample / vertical / extended / interlaced).

## Not yet supported

- Runtime registration (`register` is a no-op) and a single top-level
  frame-decode entry point.
- Encoder.
- The Annex E.1.4.4 two-way / backward RVLC error-recovery decode (the
  forward `reversible_vlc == 1` Tcoef decode is implemented; the
  bit-discard error-concealment strategies are not), and the SA-DCT
  modified inverse scan.
- Sprite / GMC bodies, scalability enhancement layers, Studio Profile,
  and non-rectangular shapes (rejected with typed errors).

## Provenance

Every numeric value and bit layout traces to ISO/IEC 14496-2:2004 (3rd
edition), read from the specification text staged under
`docs/video/mpeg4-visual/`. No third-party MPEG-4 source was consulted.

## License

MIT — see [LICENSE](./LICENSE).

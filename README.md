# oxideav-mpeg4video

Pure-Rust MPEG-4 Part 2 Video (ISO/IEC 14496-2) decoder **and encoder** for oxideav.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a
100% pure Rust media transcoding and streaming stack. No C libraries, no FFI
wrappers, no `*-sys` crates.

## Status

**Decoder** — Advanced Simple Profile (ASP) levels 1-5, covering:

* VOS / Visual Object / Video Object Layer / VOP header parsing.
* **I-VOP** decode — AC/DC prediction + H.263 / MPEG-4 dequantisation + IDCT.
* **P-VOP** decode — half-pel motion compensation, 1-MV + 4-MV paths, inter
  texture reconstruction, MV-median prediction, skipped-MB pass-through.
* Video-packet resync markers (§6.3.5.2).

**Encoder** — Simple Profile @ Level 1, producing elementary streams that
ffmpeg's `mpeg4` decoder consumes without complaint:

* **I-VOP** encode — MCBPC/CBPY, intra DC VLC, intra AC tcoef walk (Table
  B-16) with third-escape fallback for any `(last, run, level)` not in the
  short table.
* **P-VOP** encode — integer-pel diamond motion search (±7 pel) followed by
  half-pel refinement, 1-MV mode, median-predicted MVD (Table B-12), inter
  texture coding with H.263 quant + Table B-17 tcoef walk. `not_coded`
  skip MBs when the residual is all-zero and MV == (0, 0).
* GOP cadence: I every 16 frames by default; P-VOPs in between.
* H.263 quantisation (`mpeg_quant = 0`), constant `vop_quant = 5`.

Round-trip PSNR on the synthetic 64×64 test (`tests/p_vop.rs`): **~43 dB on
the I-VOP, ~41.6 dB on the 15th P-VOP** (drift under a single GOP), with
P-VOP totals around **21 %** of the all-I equivalent bitrate.

### Out of scope

* B-VOPs, S-VOPs (sprites), GMC.
* Quarter-pel motion, interlaced field coding, scalability, data
  partitioning, reversible VLCs.
* 4-MV-per-MB encoding (decoder accepts it; encoder is 1-MV).
* MPEG-4 Studio / AVC Simple profiles.

## Usage

```toml
[dependencies]
oxideav-mpeg4video = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).

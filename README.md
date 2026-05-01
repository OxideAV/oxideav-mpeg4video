# oxideav-mpeg4video

Pure-Rust **MPEG-4 Part 2 Visual** (ISO/IEC 14496-2) decoder and encoder —
the codec behind XVID / DivX / FMP4 / MP4V elementary streams. Zero C
dependencies, no `*-sys` crates, no FFI.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-mpeg4video = "0.0"
```

## Supported tooling

### Decoder

Rectangular-shape Simple / Advanced-Simple-style bitstreams. Exercised
against ffmpeg-generated reference clips (I-only and GOP-of-10).

- **Headers.** VOS, Visual Object, Video Object, Video Object Layer,
  GOV and VOP start codes. Simple Profile (`video_object_type_indication
  = 1`) and Advanced Simple (`= 17`) both parse; profile/level
  identification is read from the VOS.
- **Frame types.** I-VOPs and P-VOPs. B-VOPs and S-VOPs (sprite)
  return `Error::Unsupported`.
- **Intra coding.** Both intra DC VLC path (`intra_dc_vlc_thr` VLC
  mode) and plain 8-bit DC path (high-quant mode). Gradient-direction
  DC prediction (§7.4.3.1), AC prediction with zigzag /
  alternate-horizontal / alternate-vertical scans (§7.4.3.3).
- **Inter coding (P-VOP).** Half-pel motion compensation with 1-MV and
  4-MV modes, MV median prediction with first-slice-line special
  cases, inter texture decode (H.263 inter quant + Table B-17 tcoef
  walk), and skipped-MB pass-through.
- **Quantisation.** H.263 quant (default XVID/DivX mode) and MPEG-4
  matrix quant (`mpeg_quant = 1`), including user quant matrices and
  mismatch control for inter blocks.
- **Escape codes.** All three TCOEF escape modes (§6.3.8) including
  third-escape signed 12-bit level.
- **Resync.** Video-packet resync markers (§6.3.5.2) with forward-MB
  number validation. Per-packet predictor state reset.
- **Data partitioning (§6.2.6 / §6.3.7).** I-VOPs and P-VOPs (and
  S(GMC)-VOPs) emitted in `data_partitioned_motion_shape_texture()`
  layout: per-MB header/MV bits in part 1, 19-bit DC marker
  (`110 1011 0000 0000 0001`) or 17-bit motion marker
  (`1 1111 0000 0000 0001`), per-MB texture bits in part 2, per-MB AC
  walks in part 3. Round-21 DP decoder treats one VOP as one video
  packet; mid-VOP `video_packet_header()` splits in DP mode are a
  follow-up.
- **Reversible VLC (Table B.23, round 22).** Decoder picks up
  `reversible_vlc = 1` from the VOL when DP is on and routes every
  DCT-coefficient AC walk through `crate::rvlc::decode_intra_ac` /
  `decode_inter_ac` instead of the standard Table B.16/B.17 walker.
  Forward-direction decode is implemented; reverse-direction error
  recovery (Annex E.1.4.4 four strategies) is a follow-up.
- **Picture store.** One reference frame, refreshed by each I-VOP and
  each P-VOP. Not-coded VOPs re-emit the previous reference at the
  new pts.

Out of scope — returns `Error::Unsupported`:

- B-VOPs and S-VOPs (sprites) / GMC.
- Quarter-pel motion (`quarter_sample`).
- Interlaced field coding, scalability.
- Non-rectangular shape (binary / grayscale shape coding).
- `newpred_enable`, complexity-estimation header, reduced-resolution
  VOP.
- MPEG-4 Studio / AVC Simple profiles.

### Encoder

Simple Profile @ Level 1 elementary streams that ffmpeg's `mpeg4`
decoder accepts as-is. Input is `Yuv420P` only.

- **I-VOP.** Per-MB MCBPC (Table B-10) + ac_pred (always 0) + CBPY
  (Table B-9), six 8×8 blocks with intra DC VLC + signed residual,
  intra AC tcoef walk (Table B-16), third-escape fallback for any
  `(last, run, level)` not in the short table.
- **P-VOP.** Integer-pel diamond motion search (±7 pel) then half-pel
  refinement, 1-MV mode, median-predicted MVD (Table B-12), inter
  texture coding with H.263 quant and Table B-17 tcoef walk.
  `not_coded` skip MBs emitted when the residual is all-zero and
  MV == (0, 0).
- **GOP cadence.** I-VOP every `DEFAULT_GOP_SIZE` frames (= 16); all
  other frames are P-VOPs. Override per-encoder via the `g` codec
  option (1..=300).
- **GMC (Global Motion Compensation, §7.6.7 / §7.7).** Optional —
  enabled with the `gmc` codec option. When on, the VOL advertises
  `sprite_enable = 2` + 1 warping point at half-pel accuracy; each
  P-VOP becomes an `S(GMC)`-VOP carrying one `(du, dv)`
  `sprite_trajectory()` derived from a coarse `±16`-pel global-
  translation SAD search; per-MB `mcsel` picks between translational
  MC and warp prediction. ffmpeg cross-decode validated on synthetic
  global-pan content. Single-warp-point only — 2/3/4-point
  affine/perspective warps are a follow-up.
- **Quantisation.** H.263 quant (`mpeg_quant = 0`), constant
  `vop_quant = 5` by default, no dquant. Override per-encoder via
  the `qp` codec option (1..=31), or split per VOP-type with
  `qp_i` / `qp_p` / `qp_b`.
- **Resync markers.** Not emitted (`resync_marker_disable = 1`).

Round-trip PSNR on the synthetic 64×64 moving-gradient test
(`tests/p_vop.rs`): around 43 dB on the I-VOP, around 41.6 dB on the
15th P-VOP of a GOP of 16. P-VOP total byte count is around 21 % of
the all-I equivalent.

Out of scope for the encoder:

- 4-MV mode for P-VOPs (decoder accepts 4MV; encoder is 1-MV).
- Multi-point GMC warp (2/3/4-point affine / perspective);
  `sprite_brightness_change`. Encoder ships single-warp-point
  translational GMC only — sufficient for camera-pan / dolly-style
  global motion.
- Static sprites (`sprite_enable = 1`, S-VOP coding).
- B-VOPs round-trip is supported via `bf=N`; combined with `gmc`
  the encoder advertises both features in the VOL but does not
  warp B-VOP references through the trajectory.
- **Data partitioning (`dp=1`).** Round-21: per-VOP DP layout
  (§6.2.5.3 / §6.2.6) for I and P VOPs at half-pel only — VOL flips
  to ARTS@L1 (PLI `0x91`, vot `10`) + `data_partitioned = 1` +
  `resync_marker_disable = 0`. Each VOP body becomes one video packet
  with MV/header bits in part 1, DC marker / motion marker, texture
  bits in part 2, AC walks in part 3, then spec-conformant
  `next_start_code()` stuffing (`0` then `1`'s, or full `0x7F` if
  byte-aligned). Mutually exclusive with `qpel`/`gmc`/`bf>0` for
  now. ffmpeg cross-decode validated on the synthetic moving-gradient
  fixture (`tests/dp.rs::dp_ffmpeg_decode`).
- **Reversible VLC (`rvlc=1`, round 22).** Routes every DCT-coefficient
  AC walk through Table B.23 (intra and inter columns share the same
  prefix codes; the same prefix decodes to a different `(LAST, RUN,
  LEVEL)` triplet depending on the block type). Required by the spec
  to be combined with `dp=1` (§6.2.5: `reversible_vlc` only legal
  inside `data_partitioned_motion_shape_texture()`). 30-bit RVLC
  escape `00001 LAST(1) RUN(6) m LEVEL(11) m 0000 sign` covers any
  triplet not in B.23. Bit overhead vs non-RVLC at the same Q on the
  synthetic moving-gradient fixture: about +2.2 % bytes. ffmpeg
  cross-decode validated (`tests/rvlc.rs::rvlc_ffmpeg_decode`,
  43.5 dB I-VOP PSNR). Reverse-direction error-recovery decode
  (Annex E.1.4.4 strategies) is a follow-up — round-22 reads the
  RVLC stream forward only, which is sufficient for clean transport.
- Interlace, scalability.
- MPEG-4 matrix quant (`mpeg_quant = 1`).

## Performance

Hot paths go through the `simd` module (mirrors the `oxideav-vorbis`
layout) with three implementations:

- `scalar` — reference + test oracle, always compiled.
- `chunked` — stable-Rust default. Fixed-size `[f32; 8]` / `[i32; 8]`
  / `[u8; 8]` tiles that LLVM auto-vectorises to AVX2 / NEON / SSE on
  release builds.
- `portable` — `std::simd` (`f32x8` / `i32x8`) behind the `nightly`
  feature flag.

Kernels wrapped: `idct8x8` / `fdct8x8`, `dequant_h263`,
`clip_block_to_u8`, `add_residual_clip_block`, `copy_block_u8`,
`copy_mb_luma`, `copy_mb_chroma`. Bit-exact-vs-scalar tests live in
`src/simd/mod.rs`.

Motion compensation has an interior-only fast path that skips
per-pixel edge-replication clamping when the block footprint is in
bounds — the common case for typical encoders.

### Benchmarks

Four criterion suites under `benches/`:

- `idct_bench` — IDCT and FDCT, 1000 blocks per iteration.
- `dequant_bench` — H.263 dequantisation at sparse / medium / dense
  coefficient density, plus residual-add+clip.
- `mc_bench` — 8×8 and 16×16 half-pel motion compensation across the
  four sub-pel positions, plus the skipped-MB copy.
- `frame_bench` — end-to-end encode one I-VOP, decode one I-VOP,
  decode an IPP GOP (256×256).

Measured on `-C target-cpu=x86-64-v3` (AVX2 + FMA):

| kernel                             | before | after | Δ    |
|------------------------------------|-------:|------:|-----:|
| `predict_block 16×16 int`          | 222 µs |  28 µs | -87% |
| `predict_block 8×8 half_hv`        | 117 µs |  47 µs | -60% |
| `predict_block 8×8 half_h`         | 104 µs |  40 µs | -61% |
| `dequant_h263` dense               |  52 µs |  25 µs | -52% |
| `add_residual_clip_block`          |  24 µs |  21 µs | -10% |

IDCT and FDCT are unchanged — the scalar form is already well
vectorised by LLVM at `x86-64-v3` and above; the kernels exist for
portability to profiles where auto-vectorisation is weaker. Per-block
times are dominated by motion compensation, so the interior fast-path
win is what matters end-to-end.

Run: `cargo bench -p oxideav-mpeg4video`.

## Quick use

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase};

let mut codecs = CodecRegistry::new();
oxideav_mpeg4video::register(&mut codecs);

// Decode: feed bytes as Packets; receive VideoFrame on receive_frame.
let dec_params = CodecParameters::video(CodecId::new(oxideav_mpeg4video::CODEC_ID_STR));
let mut dec = codecs.make_decoder(&dec_params)?;
let pkt = Packet::new(0, TimeBase::new(1, 90_000), bitstream_bytes);
dec.send_packet(&pkt)?;
while let Ok(Frame::Video(_vf)) = dec.receive_frame() {
    // vf.format == PixelFormat::Yuv420P
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Encoder setup mirrors other oxideav video encoders: build
`CodecParameters` with the codec id, width/height, `PixelFormat::Yuv420P`
and a frame rate, then `make_encoder(&params)`. Feed frames via
`send_frame` and pull packets via `receive_packet`.

## Codec id

- Codec: `"mpeg4video"`. Container-level FourCCs like `XVID`, `DIVX`,
  `DX50`, `MP4V`, `FMP4` all resolve to this codec.
- Accepted pixel format: `Yuv420P`.

## Role in oxideav

The bitstream lower-layer for every MPEG-4 ASP variant. The
`oxideav-h263` crate depends on this one for shared VLC tables and
block-reconstruction helpers, so changes here are load-bearing for
H.263 baseline too.

## License

MIT — see [LICENSE](LICENSE).

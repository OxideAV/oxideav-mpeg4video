# oxideav-mpeg4video

Pure-Rust **MPEG-4 Part 2 Visual** (ISO/IEC 14496-2) decoder and encoder —
the codec behind XVID / DivX / FMP4 / MP4V elementary streams. Zero C
dependencies, no `*-sys` crates, no FFI.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.0"
oxideav-codec = "0.0"
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
- **Picture store.** One reference frame, refreshed by each I-VOP and
  each P-VOP. Not-coded VOPs re-emit the previous reference at the
  new pts.

Out of scope — returns `Error::Unsupported`:

- B-VOPs and S-VOPs (sprites) / GMC.
- Quarter-pel motion (`quarter_sample`).
- Interlaced field coding, scalability, data partitioning, reversible
  VLCs.
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
  other frames are P-VOPs. Configurable in source.
- **Quantisation.** H.263 quant (`mpeg_quant = 0`), constant
  `vop_quant = 5`, no dquant.
- **Resync markers.** Not emitted (`resync_marker_disable = 1`).

Round-trip PSNR on the synthetic 64×64 moving-gradient test
(`tests/p_vop.rs`): around 43 dB on the I-VOP, around 41.6 dB on the
15th P-VOP of a GOP of 16. P-VOP total byte count is around 21 % of
the all-I equivalent.

Out of scope for the encoder:

- 4-MV mode for P-VOPs (decoder accepts 4MV; encoder is 1-MV).
- B-VOPs, S-VOPs, sprites / GMC.
- Interlace, scalability, data partitioning, reversible VLCs.
- MPEG-4 matrix quant (`mpeg_quant = 1`).

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

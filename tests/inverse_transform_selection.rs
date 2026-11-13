//! Inverse-transform selection (`compat::InverseTransformSelection`)
//! through the whole decoder, per
//! `docs/video/mpeg4-visual/integer-idct-ecosystem-variant.md` §6–§7.
//!
//! No stream of the committed corpus carries the selecting signature
//! (they carry another encoder's user data, or none), so every stream
//! doubles as a probe: a *signed* copy gets one extra user-data unit
//! right after its `video_object_layer` header. The properties pinned
//! over the full corpus, in both decode modes:
//!
//! * the literal-spec decode never switches — signed and unsigned
//!   copies decode identically, although the build number is reported;
//! * the ecosystem-compat decode switches on the signed copy, to
//!   exactly the decode a forced `VariantRc` gives, and stays on the
//!   exact transform for the unsigned copy;
//! * a forced selection wins in either mode;
//! * the integer transform reaches every reconstruction path (each
//!   stream's forced decode differs from its exact decode) and stays an
//!   inverse DCT there (the two decodes remain close).

use oxideav_mpeg4video::compat::{DecodeOptions, InverseTransformSelection};
use oxideav_mpeg4video::decoder::{Mpeg4VideoDecoder, StreamDecodeError};

fn fixture_dir() -> String {
    format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/{name}", fixture_dir());
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Every `.m4v` elementary stream of the corpus, sorted.
fn corpus() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(fixture_dir())
        .expect("fixture directory")
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.ends_with(".m4v"))
        .collect();
    names.sort();
    assert!(names.len() >= 40, "corpus present: {}", names.len());
    if cfg!(miri) {
        // An interpreter run keeps two small streams (combined I/P/B
        // and data-partitioned).
        names.retain(|n| n == "ipb_64x64.m4v" || n == "dp_ipb_64x64.m4v");
    }
    names
}

/// Offsets of every `00 00 01 xx` start code.
fn start_codes(stream: &[u8]) -> Vec<usize> {
    (0..stream.len().saturating_sub(3))
        .filter(|&i| stream[i..i + 3] == [0, 0, 1])
        .collect()
}

/// `stream` with a user-data unit carrying `payload` inserted right
/// after the first `video_object_layer` unit.
fn with_user_data_after_vol(stream: &[u8], payload: &[u8]) -> Vec<u8> {
    let codes = start_codes(stream);
    let vol = codes
        .iter()
        .position(|&i| (0x20..=0x2F).contains(&stream[i + 3]))
        .expect("VOL start code");
    insert_user_data(stream, codes[vol + 1], payload)
}

fn insert_user_data(stream: &[u8], at: usize, payload: &[u8]) -> Vec<u8> {
    let mut out = stream[..at].to_vec();
    out.extend_from_slice(&[0, 0, 1, 0xB2]);
    out.extend_from_slice(payload);
    out.extend_from_slice(&stream[at..]);
    out
}

#[derive(Debug, PartialEq, Eq)]
struct Decode {
    /// Y, Cb, Cr of every frame, display order.
    frames: Vec<Vec<u8>>,
    build: Option<u32>,
    active: InverseTransformSelection,
}

fn decode(stream: &[u8], options: DecodeOptions) -> Result<Decode, StreamDecodeError> {
    let mut dec = Mpeg4VideoDecoder::with_options(options);
    let mut frames = dec.decode(stream)?;
    frames.extend(dec.flush());
    Ok(Decode {
        frames: frames
            .iter()
            .map(|f| {
                let mut all = f.luma_samples().to_vec();
                all.extend_from_slice(f.cb_samples());
                all.extend_from_slice(f.cr_samples());
                all
            })
            .collect(),
        build: dec.detected_encoder_build(),
        active: dec.active_inverse_transform(),
    })
}

const EXACT: InverseTransformSelection = InverseTransformSelection::Exact;
const VARIANT: InverseTransformSelection = InverseTransformSelection::VariantRc;
const FROM_STREAM: InverseTransformSelection = InverseTransformSelection::FromStream;

#[test]
fn spec_mode_never_switches_and_compat_mode_follows_the_signature() {
    for name in corpus() {
        let plain = fixture(&name);
        let signed = with_user_data_after_vol(&plain, b"XviD0050");
        for mode in [DecodeOptions::spec(), DecodeOptions::ecosystem()] {
            let exact = decode(&plain, mode.with_inverse_transform(EXACT))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let variant = decode(&plain, mode.with_inverse_transform(VARIANT))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(exact.build, None, "{name}: corpus stream is unsigned");
            assert_eq!((exact.active, variant.active), (EXACT, VARIANT));

            // Unsigned: the mode's own selection is the exact transform.
            let own = decode(&plain, mode).unwrap();
            assert_eq!(own, exact, "{name}: unsigned stream, mode default");
            let auto = decode(&plain, mode.with_inverse_transform(FROM_STREAM)).unwrap();
            assert_eq!(auto, exact, "{name}: unsigned stream, from-stream");

            // Signed.
            let own = decode(&signed, mode).unwrap();
            assert_eq!(own.build, Some(50), "{name}");
            if mode.ecosystem_compat {
                assert_eq!(own.active, VARIANT, "{name}");
                assert_eq!(own.frames, variant.frames, "{name}: compat switches");
            } else {
                assert_eq!(own.active, EXACT, "{name}");
                assert_eq!(own.frames, exact.frames, "{name}: spec never switches");
            }
            let auto = decode(&signed, mode.with_inverse_transform(FROM_STREAM)).unwrap();
            assert_eq!(auto.frames, variant.frames, "{name}: explicit from-stream");
            let forced = decode(&signed, mode.with_inverse_transform(EXACT)).unwrap();
            assert_eq!(forced.frames, exact.frames, "{name}: forced exact");
            assert_eq!((forced.build, forced.active), (Some(50), EXACT));
            let forced = decode(&signed, mode.with_inverse_transform(VARIANT)).unwrap();
            assert_eq!(forced.frames, variant.frames, "{name}: forced variant");
        }
    }
}

#[test]
fn the_integer_transform_reaches_every_reconstruction_path() {
    // Each corpus stream — intra-only, I+P, I+P+B, four-vector,
    // quarter-sample, interlaced (field DCT, field motion, direct),
    // data-partitioned (with and without RVLC), matrix quantisation,
    // S(GMC), video packets — decodes differently under the integer
    // transform, and close to the exact decode: an inverse DCT within
    // a level per block that only accumulates along prediction chains.
    let mut worst = 0u8;
    for name in corpus() {
        let plain = fixture(&name);
        let exact = decode(&plain, DecodeOptions::spec()).unwrap();
        let variant = decode(
            &plain,
            DecodeOptions::spec().with_inverse_transform(VARIANT),
        )
        .unwrap();
        assert_eq!(exact.frames.len(), variant.frames.len(), "{name}");
        let mut peak = 0u8;
        let mut first_frame_peak = 0u8;
        for (k, (a, b)) in exact.frames.iter().zip(&variant.frames).enumerate() {
            let frame_peak = a
                .iter()
                .zip(b)
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap_or(0);
            if k == 0 {
                first_frame_peak = frame_peak;
            }
            peak = peak.max(frame_peak);
        }
        assert!(peak > 0, "{name}: the transform selection had no effect");
        // The first displayed frame is the intra picture: no
        // accumulation, so the two transforms agree within one level
        // (measured: exactly 1 on every stream; up to 3 after the
        // short prediction chains of the corpus).
        assert!(
            first_frame_peak <= 2,
            "{name}: intra peak {first_frame_peak}"
        );
        assert!(peak <= 12, "{name}: peak {peak}");
        worst = worst.max(peak);
        if std::env::var_os("MPEG4VIDEO_TEST_VERBOSE").is_some() {
            println!("{name}: intra peak {first_frame_peak}, peak {peak}");
        }
    }
    assert!(worst >= 1);
}

#[test]
fn static_sprite_texture_follows_the_selection() {
    // The sprite I-VOP of a static-sprite VOL is texture like any
    // other: its blocks follow the selection, and the S-VOPs warp
    // whatever was reconstructed.
    use oxideav_mpeg4video::ivop_encode::{EncoderConfig, FrameView, StaticSpriteVol};
    use oxideav_mpeg4video::sprite_encode::StaticSpriteEncoder;
    let (sw, sh) = (40usize, 24usize);
    let y: Vec<u8> = (0..sw * sh)
        .map(|i| ((i % sw) * 5 + (i / sw) * 9 + (i * i) % 23) as u8)
        .collect();
    let c: Vec<u8> = (0..(sw / 2) * (sh / 2))
        .map(|i| (90 + i % 50) as u8)
        .collect();
    let view = FrameView {
        y: &y,
        cb: &c,
        cr: &c,
        width: sw,
        height: sh,
    };
    let cfg = EncoderConfig::default()
        .with_width(32)
        .with_height(16)
        .with_static_sprite(Some(
            StaticSpriteVol::new(sw as u16, sh as u16).with_warping_points(1),
        ));
    let (mut enc, mut plain) = StaticSpriteEncoder::new(&cfg, &view, 6);
    for k in 1..=2u16 {
        let trajectory = [[2 * i32::from(k), 0], [0; 2], [0; 2], [0; 2]];
        plain.extend_from_slice(&enc.encode_vop_with_trajectory(&trajectory, 0, k).0);
    }
    let signed = with_user_data_after_vol(&plain, b"XviD0050");
    let spec = DecodeOptions::spec();
    let eco = DecodeOptions::ecosystem();
    let exact = decode(&plain, spec).unwrap();
    let variant = decode(&plain, spec.with_inverse_transform(VARIANT)).unwrap();
    assert_eq!(exact.frames.len(), 2);
    assert_ne!(exact.frames, variant.frames);
    assert_eq!(decode(&signed, spec).unwrap().frames, exact.frames);
    assert_eq!(
        decode(&signed, eco).unwrap().frames,
        decode(&plain, eco.with_inverse_transform(VARIANT))
            .unwrap()
            .frames
    );
}

#[test]
fn short_header_pictures_keep_the_exact_transform() {
    // The H.263-baseline path is out of the selection's scope: forcing
    // the integer transform changes nothing there.
    for name in ["sh_ipp_176x144.h263", "enc_sh_ippip_176x144.h263"] {
        let stream = fixture(name);
        let exact = decode(&stream, DecodeOptions::spec()).unwrap();
        assert!(!exact.frames.is_empty());
        for mode in [DecodeOptions::spec(), DecodeOptions::ecosystem()] {
            let base = decode(&stream, mode).unwrap();
            let forced = decode(&stream, mode.with_inverse_transform(VARIANT)).unwrap();
            assert_eq!(forced.frames, base.frames, "{name}");
        }
    }
}

#[test]
fn selection_is_sticky_from_the_matching_unit_onwards() {
    // I + P stream, coded order = display order. The signature arrives
    // before the third VOP: the first two frames are the exact
    // decode's, the rest are reconstructed with the integer transform
    // — on top of the exact-transform references, so the tail matches
    // neither pure decode.
    let name = "ip_64x64.m4v";
    let plain = fixture(name);
    let vops: Vec<usize> = start_codes(&plain)
        .into_iter()
        .filter(|&i| plain[i + 3] == 0xB6)
        .collect();
    assert!(vops.len() >= 5, "{name}: {} VOPs", vops.len());
    let late = insert_user_data(&plain, vops[2], b"XviD0073 anything");
    let mode = DecodeOptions::ecosystem();
    let exact = decode(&plain, mode).unwrap();
    let variant = decode(&plain, mode.with_inverse_transform(VARIANT)).unwrap();
    let got = decode(&late, mode).unwrap();
    assert_eq!(got.build, Some(73));
    assert_eq!(got.active, VARIANT);
    assert_eq!(got.frames.len(), exact.frames.len());
    assert_eq!(got.frames[..2], exact.frames[..2], "before the unit");
    assert_ne!(got.frames[2], exact.frames[2], "first VOP after the unit");
    assert_ne!(got.frames[2..], variant.frames[2..]);

    // Unit-by-unit: the switch happens at the unit, not before, and
    // nothing afterwards reverts it — not the other encoder's
    // signature, not a non-matching unit, not a repeated VOL.
    let mut dec = Mpeg4VideoDecoder::with_options(mode);
    dec.decode(&late[..vops[2]]).unwrap();
    assert_eq!(dec.active_inverse_transform(), EXACT);
    assert_eq!(dec.detected_encoder_build(), None);
    dec.decode(&late[vops[2]..vops[2] + 4 + 17]).unwrap();
    assert_eq!(dec.active_inverse_transform(), VARIANT);
    assert_eq!(dec.detected_encoder_build(), Some(73));
    for unit in [
        &b"\x00\x00\x01\xB2Lavc62.28.100"[..],
        b"\x00\x00\x01\xB2XviD-beta",
        b"\x00\x00\x01\xB2",
        b"\x00\x00\x01\xB2xvid0001",
    ] {
        dec.decode(unit).unwrap();
        assert_eq!(dec.active_inverse_transform(), VARIANT);
        assert_eq!(dec.detected_encoder_build(), Some(73));
    }
    let codes = start_codes(&plain);
    let vol = codes
        .iter()
        .position(|&i| (0x20..=0x2F).contains(&plain[i + 3]))
        .unwrap();
    dec.decode(&plain[..codes[vol + 1]]).unwrap();
    assert_eq!(dec.active_inverse_transform(), VARIANT);
    // A later matching unit updates the reported build.
    dec.decode(b"\x00\x00\x01\xB2XviD0050").unwrap();
    assert_eq!(dec.detected_encoder_build(), Some(50));
    assert_eq!(dec.active_inverse_transform(), VARIANT);

    // The same tail, with non-matching units and a repeated
    // configuration between the VOPs, decodes like the plain late
    // stream.
    let mut noisy = late[..vops[3] + 17 + 4].to_vec();
    noisy.extend_from_slice(b"\x00\x00\x01\xB2DivX503b1393p");
    noisy.extend_from_slice(&plain[..codes[vol + 1]]);
    noisy.extend_from_slice(&late[vops[3] + 17 + 4..]);
    assert_eq!(decode(&noisy, mode).unwrap().frames, got.frames);
}

#[test]
fn signature_matching_follows_the_evidence_table() {
    // (payload, parsed build) — the document's §6.1 table plus the
    // edge cases its open questions leave to the conservative reading.
    let cases: &[(&[u8], Option<u32>)] = &[
        (b"XviD0050", Some(50)),
        (b"XviD0073 anything", Some(73)),
        (b"XviD7", Some(7)),
        (b"XviD", None),
        (b"XviD-beta", None),
        (b"xvid0050", None),
        (b" XviD0050", None),
        (b"Lavc62.28.100", None),
        (b"DivX503b1393p", None),
        (b"", None),
        // Non-decimal suffix after a digit run: the run counts.
        (b"XviD12ab", Some(12)),
        (b"XviD0x10", Some(0)),
        // A sign or a space is not a digit.
        (b"XviD+50", None),
        (b"XviD 50", None),
        // The run must fit an unsigned 32-bit integer.
        (b"XviD4294967295", Some(u32::MAX)),
        (b"XviD4294967296", None),
        (b"XviD99999999999999999999", None),
        (b"XviD000000000000000000000050", Some(50)),
        // The signature elsewhere in the payload does not count.
        (b"encoder=XviD0050", None),
        (b"XviXviD0050", None),
        (b"XviD\x000050", None),
        (b"Xvi", None),
    ];
    for &(payload, want) in cases {
        let mut unit = vec![0, 0, 1, 0xB2];
        unit.extend_from_slice(payload);
        let mut dec = Mpeg4VideoDecoder::with_options(DecodeOptions::ecosystem());
        dec.decode(&unit).unwrap();
        let shown = String::from_utf8_lossy(payload).into_owned();
        assert_eq!(dec.detected_encoder_build(), want, "{shown:?}");
        assert_eq!(
            dec.active_inverse_transform(),
            if want.is_some() { VARIANT } else { EXACT },
            "{shown:?}"
        );
        // Evidence is reported in the literal-spec mode too; it just
        // does not act on it.
        let mut dec = Mpeg4VideoDecoder::new();
        dec.decode(&unit).unwrap();
        assert_eq!(dec.detected_encoder_build(), want, "{shown:?}");
        assert_eq!(dec.active_inverse_transform(), EXACT, "{shown:?}");
    }

    // A signature split across two decode calls is two fragments, not
    // a unit: no match.
    let mut dec = Mpeg4VideoDecoder::with_options(DecodeOptions::ecosystem());
    dec.decode(b"\x00\x00\x01\xB2Xv").unwrap();
    dec.decode(b"iD0050").unwrap();
    assert_eq!(dec.detected_encoder_build(), None);
    // A unit ends at the next start code: digits beyond it are not
    // part of the run.
    let mut dec = Mpeg4VideoDecoder::with_options(DecodeOptions::ecosystem());
    dec.decode(b"\x00\x00\x01\xB2XviD00\x00\x00\x01\xB250")
        .unwrap();
    assert_eq!(dec.detected_encoder_build(), Some(0));
}

// ───────────────────── registry / options bag ─────────────────────

fn registry_decode(stream: &[u8], options: &[(&str, &str)], reset_after_first: bool) -> Vec<u8> {
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    let mut bag = oxideav_core::CodecOptions::default();
    for (k, v) in options {
        bag = bag.set(*k, *v);
    }
    params.options = bag;
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&params).expect("decoder");
    let mut out = Vec::new();
    let drain = |dec: &mut Box<dyn oxideav_core::Decoder>, out: &mut Vec<u8>| loop {
        match dec.receive_frame() {
            Ok(oxideav_core::Frame::Video(f)) => {
                for p in &f.planes {
                    out.extend_from_slice(&p.data);
                }
            }
            Ok(_) => unreachable!("video decoder"),
            Err(_) => break,
        }
    };
    let packet = |data: &[u8]| {
        oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 25), data.to_vec())
    };
    if reset_after_first {
        // Feed the signed stream once, seek back, feed an unsigned
        // copy: only the second pass is kept.
        dec.send_packet(&packet(&with_user_data_after_vol(stream, b"XviD0050")))
            .unwrap();
        dec.reset().unwrap();
    }
    dec.send_packet(&packet(stream)).unwrap();
    dec.flush().unwrap();
    drain(&mut dec, &mut out);
    out
}

#[test]
fn registry_option_selects_the_transform() {
    let plain = fixture("ipb_64x64.m4v");
    let signed = with_user_data_after_vol(&plain, b"XviD0050");
    let flat = |d: Decode| d.frames.concat();
    for (compat, mode) in [
        ("false", DecodeOptions::spec()),
        ("true", DecodeOptions::ecosystem()),
    ] {
        let exact = flat(decode(&plain, mode.with_inverse_transform(EXACT)).unwrap());
        let variant = flat(decode(&plain, mode.with_inverse_transform(VARIANT)).unwrap());
        assert_ne!(exact, variant);
        let c = ("ecosystem-compat", compat);
        // Unset: the mode decides.
        assert_eq!(registry_decode(&plain, &[c], false), exact);
        let want = if mode.ecosystem_compat {
            &variant
        } else {
            &exact
        };
        assert_eq!(&registry_decode(&signed, &[c], false), want);
        // Set: forced.
        let it = |v| ("inverse-transform", v);
        assert_eq!(registry_decode(&signed, &[c, it("exact")], false), exact);
        assert_eq!(
            registry_decode(&plain, &[c, it("variant-rc")], false),
            variant
        );
        assert_eq!(
            registry_decode(&plain, &[c, it("from-stream")], false),
            exact
        );
        assert_eq!(
            registry_decode(&signed, &[c, it("from-stream")], false),
            variant
        );
        // A seek keeps the evidence gathered before it.
        assert_eq!(&registry_decode(&plain, &[c], true), want);
    }

    // Unknown values are rejected by the schema.
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.options = oxideav_core::CodecOptions::default().set("inverse-transform", "fast");
    assert!(oxideav_mpeg4video::decoder::make_decoder(&params).is_err());

    // Signed configuration data handed over out of band counts.
    let codes = start_codes(&signed);
    let first_vop = codes
        .iter()
        .find(|&&i| signed[i + 3] == 0xB6)
        .copied()
        .unwrap();
    let first_gov = codes.iter().find(|&&i| signed[i + 3] == 0xB3).copied();
    let split = first_gov.map_or(first_vop, |g| g.min(first_vop));
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.options = oxideav_core::CodecOptions::default().set("ecosystem-compat", "true");
    params.extradata = signed[..split].to_vec();
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&params).unwrap();
    dec.send_packet(&oxideav_core::Packet::new(
        0,
        oxideav_core::TimeBase::new(1, 25),
        signed[split..].to_vec(),
    ))
    .unwrap();
    dec.flush().unwrap();
    let mut out = Vec::new();
    while let Ok(oxideav_core::Frame::Video(f)) = dec.receive_frame() {
        for p in &f.planes {
            out.extend_from_slice(&p.data);
        }
    }
    let variant = flat(
        decode(
            &plain,
            DecodeOptions::ecosystem().with_inverse_transform(VARIANT),
        )
        .unwrap(),
    );
    assert_eq!(out, variant);
}

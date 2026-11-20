//! Opt-in error concealment (`DecodeOptions::error_concealment`,
//! ISO/IEC 14496-2 E.1.4.2): a VOP that fails to decode becomes a copy
//! of its forward reference instead of an error; the reference chain
//! and the clock keep moving.

use oxideav_mpeg4video::compat::DecodeOptions;
use oxideav_mpeg4video::decoder::{Mpeg4VideoDecoder, StreamDecodeError};
use oxideav_mpeg4video::framestore::DecodedFrame;

const W: usize = 48;
const H: usize = 32;

fn picture(k: usize) -> [Vec<u8>; 3] {
    let bg =
        |x: usize, y: usize| -> u8 { (40 + (x * 7 + y * 5 + (x / 9 + y / 7) * 6) % 170) as u8 };
    let y = (0..W * H).map(|i| bg(i % W + 2 * k, i / W + k)).collect();
    let c = (0..(W / 2) * (H / 2))
        .map(|i| 100 + ((i % (W / 2) + k) % 32) as u8)
        .collect::<Vec<u8>>();
    [y, c.clone(), c]
}

/// Encode `frames` pictures; one packet per VOP (the first carries the
/// configuration run).
fn encode(frames: usize, extra: &[(&str, &str)]) -> Vec<Vec<u8>> {
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.width = Some(W as u32);
    params.height = Some(H as u32);
    params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    params.frame_rate = Some(oxideav_core::Rational::new(25, 1));
    let mut opts = oxideav_core::CodecOptions::default().set("gop-size", "50");
    for (k, v) in extra {
        opts = opts.set(*k, v.to_string());
    }
    params.options = opts;
    let mut enc = oxideav_mpeg4video::encoder::make_encoder(&params).unwrap();
    for k in 0..frames {
        let [y, cb, cr] = picture(k);
        enc.send_frame(&oxideav_core::Frame::Video(oxideav_core::VideoFrame {
            pts: None,
            planes: vec![
                oxideav_core::VideoPlane { stride: W, data: y },
                oxideav_core::VideoPlane {
                    stride: W / 2,
                    data: cb,
                },
                oxideav_core::VideoPlane {
                    stride: W / 2,
                    data: cr,
                },
            ],
        }))
        .unwrap();
    }
    enc.flush().unwrap();
    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p.data),
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("unexpected {e}"),
        }
    }
    packets
}

/// The byte offset of the VOP start code in a packet.
fn vop_offset(packet: &[u8]) -> usize {
    packet
        .windows(4)
        .position(|w| w == [0, 0, 1, 0xB6])
        .expect("a VOP in every packet")
}

fn decode(
    packets: &[Vec<u8>],
    options: DecodeOptions,
) -> (
    Result<Vec<DecodedFrame>, StreamDecodeError>,
    Mpeg4VideoDecoder,
) {
    let mut dec = Mpeg4VideoDecoder::with_options(options);
    let mut frames = Vec::new();
    for p in packets {
        match dec.decode(p) {
            Ok(f) => frames.extend(f),
            Err(e) => return (Err(e), dec),
        }
    }
    frames.extend(dec.flush());
    (Ok(frames), dec)
}

#[test]
fn a_truncated_p_vop_becomes_a_copy_of_its_reference() {
    let clean = encode(6, &[]);
    let mut damaged = clean.clone();
    // Cut the fourth VOP (a P-VOP) a few bytes into its macroblock
    // layer.
    let cut = vop_offset(&damaged[3]) + 9;
    damaged[3].truncate(cut);

    // Without concealment the damage is an error.
    let (plain, dec) = decode(&damaged, DecodeOptions::spec());
    assert!(plain.is_err());
    assert_eq!(dec.concealment().concealed_vops, 0);

    let (frames, dec) = decode(&damaged, DecodeOptions::spec().with_error_concealment(true));
    let frames = frames.unwrap();
    assert_eq!(frames.len(), 6, "every VOP still yields a picture");
    assert_eq!(dec.concealment().concealed_vops, 1);
    assert_eq!(dec.concealment().dropped_vops, 0);
    // The concealed picture is its reference, sample for sample.
    assert_eq!(frames[3].luma_samples(), frames[2].luma_samples());
    assert_eq!(frames[3].cb_samples(), frames[2].cb_samples());
    let mbs = ((W / 16) * (H / 16)) as u32;
    let concealed: Vec<u32> = frames.iter().map(|f| f.concealed_macroblocks()).collect();
    assert_eq!(concealed, [0, 0, 0, mbs, 0, 0]);
    // The clock kept moving: one tick per picture.
    let ticks: Vec<i64> = frames.iter().map(|f| f.pts_ticks().unwrap()).collect();
    assert_eq!(ticks, [0, 1, 2, 3, 4, 5]);
    // The pictures before the damage are the clean decode's.
    let (reference, _) = decode(&clean, DecodeOptions::spec());
    let reference = reference.unwrap();
    for k in 0..3 {
        assert_eq!(frames[k], reference[k], "frame {k}");
    }
    // The VOPs after it still decode (against the concealed copy).
    assert_ne!(frames[4].luma_samples(), frames[3].luma_samples());
}

#[test]
fn a_truncated_b_vop_is_concealed_without_touching_the_anchors() {
    let clean = encode(7, &[("bf", "2")]);
    let (reference, _) = decode(&clean, DecodeOptions::spec());
    let reference = reference.unwrap();
    // Coding order I P B B P B B: damage the first B-VOP.
    let mut damaged = clean.clone();
    let cut = vop_offset(&damaged[2]) + 8;
    damaged[2].truncate(cut);
    let (frames, dec) = decode(&damaged, DecodeOptions::spec().with_error_concealment(true));
    let frames = frames.unwrap();
    assert_eq!(frames.len(), 7);
    assert_eq!(dec.concealment().concealed_vops, 1);
    // Display order: I B B P B B P — only the damaged B differs, and
    // it is the copy of its forward anchor.
    for (k, (got, want)) in frames.iter().zip(&reference).enumerate() {
        if k == 1 {
            assert_eq!(got.luma_samples(), frames[0].luma_samples());
            assert!(got.concealed_macroblocks() > 0);
        } else {
            assert_eq!(got, want, "frame {k}");
        }
    }
}

#[test]
fn a_vop_without_a_readable_header_or_a_reference_is_dropped() {
    let clean = encode(4, &[]);
    // Only the start code and one byte of the second VOP survive.
    let mut damaged = clean.clone();
    let cut = vop_offset(&damaged[1]) + 5;
    damaged[1].truncate(cut);
    let (frames, dec) = decode(&damaged, DecodeOptions::spec().with_error_concealment(true));
    let frames = frames.unwrap();
    assert_eq!(dec.concealment().dropped_vops, 1);
    assert_eq!(frames.len(), 3);

    // A damaged first VOP has nothing to be a copy of.
    let mut damaged = clean.clone();
    let cut = vop_offset(&damaged[0]) + 12;
    damaged[0].truncate(cut);
    let (frames, dec) = decode(
        &damaged[..1],
        DecodeOptions::spec().with_error_concealment(true),
    );
    assert!(frames.unwrap().is_empty());
    assert_eq!(dec.concealment().dropped_vops, 1);
    assert_eq!(dec.concealment().concealed_vops, 0);
}

#[test]
fn concealment_leaves_clean_streams_and_declined_tools_alone() {
    let clean = encode(5, &[("bf", "1")]);
    let (plain, _) = decode(&clean, DecodeOptions::spec());
    let (concealing, dec) = decode(&clean, DecodeOptions::spec().with_error_concealment(true));
    assert_eq!(plain.unwrap(), concealing.unwrap());
    assert_eq!(dec.concealment(), Default::default());

    // The registry key reaches the same switch.
    let mut params = oxideav_core::CodecParameters::video(oxideav_core::CodecId::new("mpeg4video"));
    params.options = oxideav_core::CodecOptions::default().set("error-concealment", "true");
    let mut dec = oxideav_mpeg4video::decoder::make_decoder(&params).unwrap();
    let mut damaged = clean.clone();
    let cut = vop_offset(&damaged[1]) + 9;
    damaged[1].truncate(cut);

    let mut frames = 0;
    for data in damaged {
        let packet = oxideav_core::Packet::new(0, oxideav_core::TimeBase::new(1, 25), data);
        dec.send_packet(&packet)
            .expect("a concealed VOP is not an error");
        while dec.receive_frame().is_ok() {
            frames += 1;
        }
    }
    dec.flush().unwrap();
    while dec.receive_frame().is_ok() {
        frames += 1;
    }
    assert_eq!(frames, 5);
}

#[test]
fn the_thread_budget_does_not_change_the_pictures() {
    // P- and B-VOPs with real motion, decoded serially and with
    // budgets above and below the number of macroblock rows.
    for extra in [
        &[][..],
        &[("bf", "2")][..],
        &[("bf", "1"), ("qpel", "true"), ("four-mv", "true")][..],
    ] {
        let packets = encode(8, extra);
        let stream: Vec<u8> = packets.concat();
        let mut serial = Mpeg4VideoDecoder::new();
        assert_eq!(serial.threads(), 1);
        let mut want = serial.decode(&stream).unwrap();
        want.extend(serial.flush());
        assert_eq!(want.len(), 8);
        for threads in [2usize, 3, 16] {
            let mut dec = Mpeg4VideoDecoder::new();
            dec.set_threads(threads);
            let mut got = dec.decode(&stream).unwrap();
            got.extend(dec.flush());
            assert_eq!(got, want, "threads = {threads}, options {extra:?}");
        }
    }
}

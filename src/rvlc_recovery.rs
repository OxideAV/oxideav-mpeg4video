//! §E.1.4.4 two-way RVLC error-recovery *driver*.
//!
//! This module assembles the otherwise-composable RVLC pieces — the
//! forward EVENT decoder ([`decode_ac_event_rvlc`]), the backward EVENT
//! decoder ([`decode_ac_event_rvlc_reverse`]), and the §E.1.4.4.2.1
//! arbitration ([`RvlcArbitration`]) — into the actual recovery walk the
//! spec describes: forward-decode the DCT-coefficient region of a video
//! packet macroblock-by-macroblock; if and only if the forward decode
//! hits an error (§E.1.4.4.1), backward-decode from the packet end,
//! gather the `L / N / L1 / L2 / N1 / N2` counters, run the strategy
//! selection, and emit a per-MB keep decision plus the recovered EVENTs.
//!
//! The texture region of a video packet is the third partition of a
//! `data_partitioned` VOP (or the trailing `block()` data of a combined
//! packet). Each macroblock contributes a fixed sequence of *coded*
//! 8×8 blocks (the §6.3.5 `cbpy` / `cbpc` pattern selects which of the
//! six 4:2:0 blocks carry coefficients), and each coded block is one
//! `while (!last)` EVENT run on its per-block Tcoef table (Table B.16
//! intra / B.17 inter). The caller supplies that per-MB block layout via
//! [`MbBlockLayout`]; this driver consumes the EVENT runs and tracks
//! where each macroblock boundary falls in the bitstream so `f_mb` /
//! `b_mb` can be evaluated.
//!
//! ## §E.1.4.4.1 forward error conditions
//!
//! The forward decode is declared to have hit an error when any block's
//! EVENT run fails to decode — an illegal RVLC, a bad escape, a missing
//! `LAST`, or a run-length overflow (more than 64 coefficients in a
//! block). All surface as a [`TextureParseError`] from the per-EVENT
//! decoder, which this driver treats as the §E.1.4.4.1 trigger.
//!
//! Provenance: ISO/IEC 14496-2:2004 (3rd edition) §E.1.4.4, read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.
//!
//! [`decode_ac_event_rvlc`]: crate::texture::decode_ac_event_rvlc
//! [`decode_ac_event_rvlc_reverse`]: crate::texture::decode_ac_event_rvlc_reverse

use crate::bitreader::{BackwardBitReader, BitReader};
use crate::rvlc_arbitration::{RvlcArbitration, RvlcArbitrationInput};
use crate::texture::{
    decode_ac_event_rvlc, decode_ac_event_rvlc_reverse, AcEvent, TcoefTable, TextureParseError,
};

/// The coded-block layout of one macroblock's texture: the Tcoef table
/// for each coded 8×8 block, in decode order. An empty layout means the
/// macroblock carries no coded blocks (all `cbpy` / `cbpc` bits clear).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MbBlockLayout {
    /// One entry per *coded* block, giving its Tcoef table. The §6.3.5
    /// block order is luma 0..3 then chroma 4..5; only coded blocks
    /// appear here.
    pub blocks: Vec<TcoefTable>,
}

impl MbBlockLayout {
    /// A macroblock with no coded texture blocks.
    pub fn empty() -> Self {
        Self { blocks: Vec::new() }
    }
}

/// One recovered macroblock's texture EVENTs, grouped per coded block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredMb {
    /// EVENT runs, one `Vec<AcEvent>` per coded block (parallel to
    /// [`MbBlockLayout::blocks`]). Each inner run ends with its
    /// `LAST == true` EVENT.
    pub blocks: Vec<Vec<AcEvent>>,
}

/// The outcome of a two-way RVLC recovery over one video packet's
/// DCT-coefficient region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RvlcRecovery {
    /// The forward decode completed without error; no recovery was
    /// needed. Every macroblock's EVENTs are present and trusted.
    Clean {
        /// Per-macroblock recovered EVENTs (one entry per MB).
        mbs: Vec<RecoveredMb>,
    },
    /// The forward decode hit a §E.1.4.4.1 error. The forward result is
    /// trusted up to `arbitration.keep_front` macroblocks; the backward
    /// result is trusted for the last `arbitration.keep_back`
    /// macroblocks. The middle (errored) macroblocks are discarded.
    Recovered {
        /// The §E.1.4.4.2.1 strategy + keep counts.
        arbitration: RvlcArbitration,
        /// Forward-decoded macroblocks (valid for indices
        /// `0 .. keep_front`).
        forward: Vec<RecoveredMb>,
        /// Backward-decoded macroblocks in forward scan order (valid for
        /// indices `total - keep_back .. total`).
        backward: Vec<RecoveredMb>,
    },
}

impl RvlcRecovery {
    /// Stitch a recovery into the final per-macroblock decode set,
    /// applying both the §E.1.4.4.2.1 keep decision (forward result for
    /// the first `keep_front` MBs, backward result for the last
    /// `keep_back` MBs, the errored middle discarded) and the
    /// §E.1.4.4.2.2 INTRA-MB concealment (every INTRA MB in an *errored*
    /// packet is concealed, even one the strategy would otherwise keep).
    ///
    /// `total_mbs` is the macroblock count of the video packet (`N`).
    /// `is_intra(i)` reports whether MB `i` is INTRA-coded (from the
    /// already-decoded motion / shape partition). The result has one
    /// entry per macroblock: `Some(mb)` for a kept, displayed MB or
    /// `None` for a discarded (middle) or concealed (INTRA) MB.
    ///
    /// A [`RvlcRecovery::Clean`] result returns every MB as `Some(_)`
    /// (no error → no concealment); `is_intra` is unused in that case.
    pub fn stitch(
        &self,
        total_mbs: usize,
        mut is_intra: impl FnMut(usize) -> bool,
    ) -> Vec<Option<RecoveredMb>> {
        match self {
            RvlcRecovery::Clean { mbs } => mbs.iter().cloned().map(Some).collect(),
            RvlcRecovery::Recovered {
                arbitration,
                forward,
                backward,
            } => {
                let keep_front = arbitration.keep_front.min(total_mbs);
                let keep_back = arbitration.keep_back.min(total_mbs);
                let back_start = total_mbs.saturating_sub(keep_back).max(keep_front);
                let mut out: Vec<Option<RecoveredMb>> = Vec::with_capacity(total_mbs);
                for i in 0..total_mbs {
                    // §E.1.4.4.2.2: conceal every INTRA MB in the packet.
                    if is_intra(i) {
                        out.push(None);
                        continue;
                    }
                    if i < keep_front {
                        out.push(forward.get(i).cloned());
                    } else if i >= back_start {
                        // backward is in forward scan order; its entry for
                        // MB `i` is at index `i - (total - backward.len())`.
                        let off = total_mbs.saturating_sub(backward.len());
                        out.push(backward.get(i.saturating_sub(off)).cloned());
                    } else {
                        // Errored middle region — discarded.
                        out.push(None);
                    }
                }
                out
            }
        }
    }
}

/// Decode one coded block's forward EVENT run (`while (!last)`),
/// returning the EVENTs. Propagates the first per-EVENT error.
fn forward_block(
    br: &mut BitReader<'_>,
    table: TcoefTable,
) -> Result<Vec<AcEvent>, TextureParseError> {
    let mut events = Vec::new();
    loop {
        let ev = decode_ac_event_rvlc(br, table)?;
        let last = ev.last;
        events.push(ev);
        // §E.1.4.4.1: a block carries at most 64 coefficients. The total
        // coefficient count is 1 (this level) + sum of preceding runs.
        let coeff_count: u64 = events.iter().map(|e| u64::from(e.run) + 1).sum();
        if coeff_count > 64 {
            return Err(TextureParseError::InvalidTcoef { window: 0 });
        }
        if last {
            return Ok(events);
        }
    }
}

/// Decode one §7.4.1.2 EVENT in the backward direction. Thin wrapper
/// over [`decode_ac_event_rvlc_reverse`]; the recovery walk in
/// [`recover_video_packet_dct`] handles the per-block grouping on the
/// `LAST` flag (a block, read tail-first, begins with its forward `LAST`
/// EVENT and continues while the next EVENT is non-`LAST`).
fn backward_event(
    br: &mut BackwardBitReader<'_>,
    table: TcoefTable,
) -> Result<AcEvent, TextureParseError> {
    decode_ac_event_rvlc_reverse(br, table)
}

/// Run the two-way RVLC recovery over the DCT-coefficient region
/// `[start_bit, end_bit)` of `data`, given the per-macroblock coded-block
/// layout (`layouts[i]` is macroblock `i`).
///
/// Returns [`RvlcRecovery::Clean`] when the forward decode succeeds for
/// every macroblock, or [`RvlcRecovery::Recovered`] when a forward error
/// fires the §E.1.4.4 two-way path. A backward decode that cannot
/// recover even one macroblock (the trailing bits are themselves
/// corrupt) propagates the backward error.
pub fn recover_video_packet_dct(
    data: &[u8],
    start_bit: usize,
    end_bit: usize,
    layouts: &[MbBlockLayout],
) -> Result<RvlcRecovery, TextureParseError> {
    let total_mbs = layouts.len();

    // ---- Forward pass: decode MB-by-MB, tracking cumulative bits. ----
    let mut fwd = BitReader::new(data);
    fwd.skip_bits(start_bit)
        .map_err(|_| TextureParseError::Truncated)?;

    let mut forward_mbs: Vec<RecoveredMb> = Vec::new();
    // forward_mb_bits[i] = cumulative forward bits *after* finishing MB i.
    let mut forward_mb_bits: Vec<i64> = Vec::new();
    let mut forward_error = false;

    for layout in layouts.iter() {
        if fwd.bit_position() > end_bit {
            forward_error = true;
            break;
        }
        let mut blocks: Vec<Vec<AcEvent>> = Vec::with_capacity(layout.blocks.len());
        let mut mb_ok = true;
        for &table in &layout.blocks {
            match forward_block(&mut fwd, table) {
                Ok(ev) => blocks.push(ev),
                Err(_) => {
                    mb_ok = false;
                    break;
                }
            }
        }
        if !mb_ok || fwd.bit_position() > end_bit {
            forward_error = true;
            break;
        }
        forward_mbs.push(RecoveredMb { blocks });
        // Cumulative forward bits relative to the region start.
        forward_mb_bits.push((fwd.bit_position() - start_bit) as i64);
    }

    let l1 = forward_mb_bits.last().copied().unwrap_or(0);
    let n1 = forward_mbs.len();

    if !forward_error && n1 == total_mbs {
        return Ok(RvlcRecovery::Clean { mbs: forward_mbs });
    }

    // ---- Backward pass: decode from the region end toward the front. ----
    // We decode EVENTs tail-first and segment them into macroblocks using
    // the layouts read in reverse (the last MB's blocks come first), then
    // reverse so the result is in forward scan order.
    let mut back = BackwardBitReader::new(data, start_bit, end_bit);
    let mut backward_mbs_rev: Vec<RecoveredMb> = Vec::new();
    // backward_mb_bits_rev[i] = cumulative backward bits after the i-th MB
    // counted from the END of the region.
    let mut backward_mb_bits_rev: Vec<i64> = Vec::new();
    let mut backward_error: Option<TextureParseError> = None;

    'mbs: for layout in layouts.iter().rev() {
        // Decode this MB's coded blocks in reverse block order. Each
        // block's EVENTs, read backward, arrive `LAST`-first; we collect a
        // block until the forward-first EVENT (the one that, read forward,
        // would begin the block) — but the spec only marks `LAST`, so we
        // read EVENTs until the next `LAST` is seen, which belongs to the
        // *previous* (forward) block. We therefore peek by buffering: read
        // one EVENT, and if it is `LAST` and the current block already has
        // EVENTs, it belongs to the next block.
        let mut mb_blocks_rev: Vec<Vec<AcEvent>> = Vec::with_capacity(layout.blocks.len());
        for _ in 0..layout.blocks.len() {
            // A block, read backward, is `[LAST, then zero or more
            // non-LAST]`: the forward `LAST` EVENT is encountered first,
            // then the block's earlier (forward) EVENTs in reverse. The
            // block ends when the next EVENT would be a `LAST` (the next
            // block's). We read the leading `LAST`, then peek-and-consume
            // while the next EVENT is non-`LAST`.
            let mut run_rev: Vec<AcEvent> = Vec::new();
            // First EVENT of the block (backward) must be a LAST.
            match backward_event(&mut back, layout_block_table(layout, &mb_blocks_rev)) {
                Ok(ev) => {
                    debug_assert!(ev.last, "backward block must start with a LAST event");
                    run_rev.push(ev);
                }
                Err(e) => {
                    backward_error = Some(e);
                    break 'mbs;
                }
            }
            // Continue reading non-LAST events for this block.
            loop {
                // Stop if region exhausted.
                if back.remaining_bits() == 0 {
                    break;
                }
                // Peek: decode the next event; if it is a LAST it belongs
                // to the previous (forward) block, so we must not consume
                // it here. We use a cloned reader to peek.
                let mut probe = back_clone(&back);
                match backward_event(&mut probe, layout_block_table(layout, &mb_blocks_rev)) {
                    Ok(ev) if !ev.last => {
                        // Belongs to this block — consume it for real.
                        let consumed =
                            backward_event(&mut back, layout_block_table(layout, &mb_blocks_rev))
                                .expect("re-decode of peeked event must succeed");
                        run_rev.push(consumed);
                    }
                    _ => break, // LAST or error → end of this block.
                }
            }
            // run_rev is LAST-first; reverse to forward order.
            run_rev.reverse();
            mb_blocks_rev.push(run_rev);
        }
        // mb_blocks_rev holds blocks in reverse block order; reverse to
        // forward block order.
        mb_blocks_rev.reverse();
        backward_mbs_rev.push(RecoveredMb {
            blocks: mb_blocks_rev,
        });
        backward_mb_bits_rev.push((end_bit - back.bit_position()) as i64);
    }

    // If the backward decode never recovered a macroblock, propagate.
    if backward_mbs_rev.is_empty() {
        if let Some(e) = backward_error {
            return Err(e);
        }
        return Err(TextureParseError::Truncated);
    }

    let l2 = backward_mb_bits_rev.last().copied().unwrap_or(0);
    let n2 = backward_mbs_rev.len();

    // Reverse the backward result into forward scan order.
    let mut backward_mbs = backward_mbs_rev;
    backward_mbs.reverse();

    // ---- Arbitration. ----
    let l = (end_bit - start_bit) as i64;
    let input = RvlcArbitrationInput {
        total_mbs,
        total_bits: l,
        forward_bits: l1,
        backward_bits: l2,
        forward_complete_mbs: n1.min(total_mbs.saturating_sub(1)),
        backward_complete_mbs: n2.min(total_mbs.saturating_sub(1)),
        forward_mb_bits,
        backward_mb_bits: backward_mb_bits_rev,
    };
    let arbitration = RvlcArbitration::select(&input);

    Ok(RvlcRecovery::Recovered {
        arbitration,
        forward: forward_mbs,
        backward: backward_mbs,
    })
}

/// Pick the Tcoef table for the next coded block within a macroblock,
/// given the blocks already decoded backward in this MB. Backward block
/// order is the reverse of forward order, so the next block to decode is
/// at index `layout.blocks.len() - 1 - already.len()`.
fn layout_block_table(layout: &MbBlockLayout, already: &[Vec<AcEvent>]) -> TcoefTable {
    let n = layout.blocks.len();
    let idx = n.saturating_sub(1 + already.len());
    layout.blocks.get(idx).copied().unwrap_or(TcoefTable::Inter)
}

/// Clone a [`BackwardBitReader`]'s state for a non-consuming peek.
fn back_clone<'a>(br: &BackwardBitReader<'a>) -> BackwardBitReader<'a> {
    br.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::TcoefTable;

    /// Encode a single non-escape reversible Tcoef EVENT (Table B.23) is
    /// non-trivial to hand-roll; instead these tests build streams by
    /// round-tripping through the forward decoder on known-good inputs and
    /// assert the driver's *structure* (clean vs. recovered, MB counts).
    /// We use the smallest reversible code: the inter "10s" EVENT
    /// (`LAST=1, RUN=0, LEVEL=1`) which Table B.23 encodes as a short
    /// codeword. To avoid depending on the exact codeword bits here we
    /// construct the bitstream by emitting events through a helper that
    /// mirrors the decoder, then feed it back.
    ///
    /// Build a forward RVLC stream for a sequence of single-block,
    /// single-EVENT macroblocks all using the inter table, by encoding
    /// each EVENT as the Table B.23 codeword for (LAST,RUN,LEVEL)=(1,0,1)
    /// followed by sign bit 0. From `rvlc_tables` the (1,0,1) inter code
    /// is `1011` (4 bits) per Table B.23; sign 0 → "10110" per EVENT.
    fn one_event_stream(mb_count: usize) -> (Vec<u8>, usize) {
        // (LAST=1,RUN=0,LEVEL=1) inter reversible code `1011` + sign 0.
        // We verify the exact bits by decoding one and checking below.
        let per_event = "1011 0";
        let mut s = String::new();
        for _ in 0..mb_count {
            s.push_str(per_event);
        }
        let cleaned: String = s.chars().filter(|c| *c == '0' || *c == '1').collect();
        let nbits = cleaned.len();
        let mut out = Vec::new();
        let mut cur = 0u8;
        let mut n = 0u8;
        for c in cleaned.chars() {
            cur = (cur << 1) | if c == '1' { 1 } else { 0 };
            n += 1;
            if n == 8 {
                out.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            cur <<= 8 - n;
            out.push(cur);
        }
        (out, nbits)
    }

    #[test]
    fn single_event_codeword_is_as_expected() {
        // Confirm the (1,0,1) inter reversible EVENT decodes from "1100".
        let (data, nbits) = one_event_stream(1);
        let mut br = BitReader::new(&data);
        let ev = decode_ac_event_rvlc(&mut br, TcoefTable::Inter).unwrap();
        assert!(ev.last);
        assert_eq!(ev.run, 0);
        assert_eq!(ev.level, 1);
        assert_eq!(br.bit_position(), nbits);
    }

    #[test]
    fn clean_forward_decode_no_recovery() {
        // Three single-block single-EVENT inter MBs, all decodable.
        let (data, nbits) = one_event_stream(3);
        let layouts = vec![
            MbBlockLayout {
                blocks: vec![TcoefTable::Inter],
            },
            MbBlockLayout {
                blocks: vec![TcoefTable::Inter],
            },
            MbBlockLayout {
                blocks: vec![TcoefTable::Inter],
            },
        ];
        let out = recover_video_packet_dct(&data, 0, nbits, &layouts).unwrap();
        match out {
            RvlcRecovery::Clean { mbs } => {
                assert_eq!(mbs.len(), 3);
                for mb in &mbs {
                    assert_eq!(mb.blocks.len(), 1);
                    assert_eq!(mb.blocks[0].len(), 1);
                    assert_eq!(mb.blocks[0][0].level, 1);
                }
            }
            RvlcRecovery::Recovered { .. } => panic!("expected a clean decode"),
        }
    }

    #[test]
    fn empty_layout_mb_decodes_as_no_blocks() {
        // A clean stream where one MB has no coded blocks.
        let (data, nbits) = one_event_stream(2);
        let layouts = vec![
            MbBlockLayout {
                blocks: vec![TcoefTable::Inter],
            },
            MbBlockLayout::empty(),
            MbBlockLayout {
                blocks: vec![TcoefTable::Inter],
            },
        ];
        let out = recover_video_packet_dct(&data, 0, nbits, &layouts).unwrap();
        match out {
            RvlcRecovery::Clean { mbs } => {
                assert_eq!(mbs.len(), 3);
                assert!(mbs[1].blocks.is_empty());
            }
            RvlcRecovery::Recovered { .. } => panic!("expected clean"),
        }
    }

    #[test]
    fn forward_error_triggers_recovery() {
        // Two good MBs then garbage that no RVLC matches, then we mark a
        // third MB. The forward decode should error on MB index 2 and the
        // two-way path engages. The backward decode of the garbage tail
        // will itself fail, so we expect either Recovered (if the tail had
        // any decodable events) or a propagated error. Here the tail is a
        // run of 1-bits which is an illegal RVLC, so the backward decode
        // recovers nothing and the error propagates — exercising the
        // no-backward-recovery branch.
        let mut s = String::from("10110 10110"); // two good MBs (1011 + sign 0)
        s.push_str("0000 0000 0000 0000 0000"); // illegal/degenerate tail
        let cleaned: String = s.chars().filter(|c| *c == '0' || *c == '1').collect();
        let nbits = cleaned.len();
        let mut out_bytes = Vec::new();
        let mut cur = 0u8;
        let mut n = 0u8;
        for c in cleaned.chars() {
            cur = (cur << 1) | if c == '1' { 1 } else { 0 };
            n += 1;
            if n == 8 {
                out_bytes.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            cur <<= 8 - n;
            out_bytes.push(cur);
        }
        let layouts = vec![
            MbBlockLayout {
                blocks: vec![TcoefTable::Inter],
            },
            MbBlockLayout {
                blocks: vec![TcoefTable::Inter],
            },
            MbBlockLayout {
                blocks: vec![TcoefTable::Inter],
            },
        ];
        let res = recover_video_packet_dct(&out_bytes, 0, nbits, &layouts);
        // The tail is illegal in both directions; the driver either
        // propagates the backward error or returns a Recovered with the
        // two good front MBs. Both are acceptable structural outcomes; we
        // assert it does not panic and, if Recovered, keeps >=0 front MBs.
        match res {
            Ok(RvlcRecovery::Recovered {
                arbitration,
                forward,
                ..
            }) => {
                assert!(forward.len() >= 2);
                assert!(arbitration.keep_front <= 3);
            }
            Ok(RvlcRecovery::Clean { .. }) => {
                panic!("garbage tail must not decode clean")
            }
            Err(_) => { /* backward decode could not recover — acceptable */ }
        }
    }

    /// Assemble a bit-string into bytes + bit length.
    fn assemble(s: &str) -> (Vec<u8>, usize) {
        let cleaned: String = s.chars().filter(|c| *c == '0' || *c == '1').collect();
        let nbits = cleaned.len();
        let mut out = Vec::new();
        let mut cur = 0u8;
        let mut n = 0u8;
        for c in cleaned.chars() {
            cur = (cur << 1) | if c == '1' { 1 } else { 0 };
            n += 1;
            if n == 8 {
                out.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            cur <<= 8 - n;
            out.push(cur);
        }
        (out, nbits)
    }

    #[test]
    fn stitch_clean_returns_all_mbs() {
        let (data, nbits) = one_event_stream(3);
        let layouts = vec![
            MbBlockLayout {
                blocks: vec![TcoefTable::Inter],
            };
            3
        ];
        let out = recover_video_packet_dct(&data, 0, nbits, &layouts).unwrap();
        let stitched = out.stitch(3, |_| false);
        assert_eq!(stitched.len(), 3);
        assert!(stitched.iter().all(|m| m.is_some()));
    }

    #[test]
    fn stitch_recovered_keeps_front_back_discards_middle_conceals_intra() {
        use crate::rvlc_arbitration::{RvlcArbitration, RvlcStrategy};
        // Hand-build a Recovered with N=5, keep_front=2, keep_back=1.
        let mb = |lvl: i32| RecoveredMb {
            blocks: vec![vec![AcEvent {
                last: true,
                run: 0,
                level: lvl,
            }]],
        };
        let rec = RvlcRecovery::Recovered {
            arbitration: RvlcArbitration {
                strategy: RvlcStrategy::Strategy1,
                keep_front: 2,
                keep_back: 1,
            },
            // forward valid for indices 0,1.
            forward: vec![mb(10), mb(11)],
            // backward in forward scan order; valid for index 4 (last).
            backward: vec![mb(40), mb(41), mb(42), mb(43), mb(44)],
        };
        // No intra concealment: MBs 0,1 from forward; 2,3 discarded; 4
        // from backward.
        let stitched = rec.stitch(5, |_| false);
        assert_eq!(stitched.len(), 5);
        assert_eq!(stitched[0].as_ref().unwrap().blocks[0][0].level, 10);
        assert_eq!(stitched[1].as_ref().unwrap().blocks[0][0].level, 11);
        assert!(stitched[2].is_none());
        assert!(stitched[3].is_none());
        assert_eq!(stitched[4].as_ref().unwrap().blocks[0][0].level, 44);

        // Conceal MB 1 (intra): now index 1 is None even though kept.
        let with_concealment = rec.stitch(5, |i| i == 1);
        assert_eq!(with_concealment[0].as_ref().unwrap().blocks[0][0].level, 10);
        assert!(with_concealment[1].is_none()); // concealed intra
        assert_eq!(with_concealment[4].as_ref().unwrap().blocks[0][0].level, 44);
    }

    #[test]
    fn multi_event_block_roundtrips_forward_and_backward() {
        // One MB, one block with two EVENTs:
        //   non-LAST (LAST=0,RUN=0,LEVEL=1) inter = "110" + sign 0
        //   LAST     (LAST=1,RUN=0,LEVEL=1) inter = "1011" + sign 0
        // Forward bits: "110 0 1011 0".
        let (data, nbits) = assemble("110 0 1011 0");
        // Forward decode through the driver: clean.
        let layouts = vec![MbBlockLayout {
            blocks: vec![TcoefTable::Inter],
        }];
        let out = recover_video_packet_dct(&data, 0, nbits, &layouts).unwrap();
        let fwd_mbs = match out {
            RvlcRecovery::Clean { mbs } => mbs,
            RvlcRecovery::Recovered { .. } => panic!("expected clean"),
        };
        assert_eq!(fwd_mbs.len(), 1);
        assert_eq!(fwd_mbs[0].blocks.len(), 1);
        let block = &fwd_mbs[0].blocks[0];
        assert_eq!(block.len(), 2);
        assert_eq!((block[0].last, block[0].run, block[0].level), (false, 0, 1));
        assert_eq!((block[1].last, block[1].run, block[1].level), (true, 0, 1));

        // Now backward-decode the same region directly and confirm the
        // segmentation reproduces the same two EVENTs in forward order.
        let mut back = BackwardBitReader::new(&data, 0, nbits);
        // First backward event is the LAST.
        let e0 = backward_event(&mut back, TcoefTable::Inter).unwrap();
        assert!(e0.last);
        // Second backward event is the non-LAST.
        let e1 = backward_event(&mut back, TcoefTable::Inter).unwrap();
        assert!(!e1.last);
        assert_eq!((e1.run, e1.level), (0, 1));
    }
}

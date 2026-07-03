//! §6.1.3.8 VOP reordering — coding-order → display-order sequence
//! driver.
//!
//! A video object layer transmits VOPs in **decoding (coding) order**,
//! which differs from **display order** whenever B-VOPs are present
//! (§3.227 / §6.1.3.8). This module owns that reorder buffer on top of
//! the per-VOP frame-assembly drivers in [`crate::frame_decode`] and the
//! reference-frame chain in [`crate::framestore`].
//!
//! ## The reorder rule (§6.1.3.8)
//!
//! > * If the current VOP in decoding order is a B-VOP the output VOP is
//! >   the VOP reconstructed from that B-VOP.
//! > * If the current VOP in decoding order is an I-, P-, or S(GMC)-VOP
//! >   the output VOP is the VOP reconstructed from the *previous* I-,
//! >   P-, or S(GMC)-VOP if one exists. If none exists, at the start of
//! >   the video object layer, no VOP is output.
//!
//! Concretely: an anchor (I/P/S) is held back by one slot. When the next
//! anchor arrives the held one is released for display; any B-VOPs that
//! arrived in between are released *before* the next anchor (they display
//! between the two anchors they bracket). This is the classic one-anchor
//! reorder delay; the worked §6.1.3.8 example `1I 4P 2B 3B 7P …` →
//! `1I 2B 3B 4P …` falls straight out of it.
//!
//! ## Usage
//!
//! Feed coded VOPs in bitstream order via [`SequenceDecoder::push_i_vop`]
//! / [`SequenceDecoder::push_p_vop`] / [`SequenceDecoder::push_s_gmc_vop`]
//! / [`SequenceDecoder::push_b_vop`].
//! Each call returns the **display-order** frames that became ready as a
//! result (zero, one, or — at a flush — more). After the last VOP, call
//! [`SequenceDecoder::flush`] to release the final held anchor.
//!
//! S(GMC)-VOPs are reordered exactly as P-VOPs (§6.1.3.8 final sentence).

use crate::frame_decode::{
    assemble_b_vop_frame, assemble_p_vop_frame, assemble_s_gmc_vop_frame, FrameDecodeError,
    PVopMbContent, SGmcMbContent,
};
use crate::framestore::{DecodedFrame, FrameStore};
use crate::reconstruct::ReconstructedMacroblock;
use crate::warp::WarpGeometry;

/// A coding-order → display-order reorder driver over one video object
/// layer.
///
/// Holds the §7.6.1 reference-frame chain ([`FrameStore`]) plus the
/// §6.1.3.8 one-slot anchor reorder buffer. The output of each `push_*`
/// is the list of frames that became displayable, in display order.
#[derive(Debug, Default)]
pub struct SequenceDecoder {
    store: FrameStore,
    /// The most recently decoded anchor that has not yet been released
    /// for display (held back by the §6.1.3.8 one-slot delay). `None`
    /// before the first anchor and immediately after a flush.
    pending_anchor: Option<DecodedFrame>,
}

impl SequenceDecoder {
    /// A fresh decoder with an empty chain and no pending anchor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the underlying reference-frame chain (e.g. to inspect the
    /// current anchors).
    #[inline]
    pub fn store(&self) -> &FrameStore {
        &self.store
    }

    /// Decode an I-VOP (intra-only, supplied already reconstructed) in
    /// coding order. Releases the previously-held anchor for display, if
    /// any, then holds this I-VOP back.
    pub fn push_i_vop(
        &mut self,
        mb_width: usize,
        mb_height: usize,
        macroblocks: &[ReconstructedMacroblock],
    ) -> Result<Vec<DecodedFrame>, FrameDecodeError> {
        let expected = mb_width * mb_height;
        if macroblocks.len() != expected {
            return Err(FrameDecodeError::MacroblockCountMismatch {
                supplied: macroblocks.len(),
                expected,
            });
        }
        let mut frame =
            DecodedFrame::new(mb_width * 16, mb_height * 16, crate::vop::VopCodingType::I)?;
        for (idx, mb) in macroblocks.iter().enumerate() {
            frame.blit_macroblock(idx % mb_width, idx / mb_width, mb)?;
        }
        Ok(self.accept_anchor(frame))
    }

    /// Decode a P-VOP in coding order against the current forward anchor.
    /// Releases the previously-held anchor for display, then holds this
    /// P-VOP back.
    pub fn push_p_vop(
        &mut self,
        mb_width: usize,
        mb_height: usize,
        entries: &[PVopMbContent],
        vop_rounding_type: u8,
        sample_mode: crate::bvop_prediction::BVopSampleMode,
        bits_per_pixel: u32,
    ) -> Result<Vec<DecodedFrame>, FrameDecodeError> {
        let frame = assemble_p_vop_frame(
            &self.store,
            mb_width,
            mb_height,
            entries,
            vop_rounding_type,
            sample_mode,
            bits_per_pixel,
        )?;
        Ok(self.accept_anchor(frame))
    }

    /// Decode a P-VOP whose VOL coded `obmc_disable == 0`: the §7.6.6
    /// overlapped-motion-compensation assembly
    /// ([`assemble_p_vop_frame_obmc`](crate::frame_decode::assemble_p_vop_frame_obmc)).
    /// Reordered exactly as [`SequenceDecoder::push_p_vop`].
    pub fn push_p_vop_obmc(
        &mut self,
        mb_width: usize,
        mb_height: usize,
        entries: &[PVopMbContent],
        vop_rounding_type: u8,
        bits_per_pixel: u32,
    ) -> Result<Vec<DecodedFrame>, FrameDecodeError> {
        let frame = crate::frame_decode::assemble_p_vop_frame_obmc(
            &self.store,
            mb_width,
            mb_height,
            entries,
            vop_rounding_type,
            bits_per_pixel,
        )?;
        Ok(self.accept_anchor(frame))
    }

    /// Decode an S(GMC)-VOP in coding order. Reordered exactly as a
    /// P-VOP (§6.1.3.8).
    #[allow(clippy::too_many_arguments)]
    pub fn push_s_gmc_vop(
        &mut self,
        mb_width: usize,
        mb_height: usize,
        entries: &[SGmcMbContent],
        geometry: &WarpGeometry,
        vop_rounding_type: u8,
        sample_mode: crate::bvop_prediction::BVopSampleMode,
        bits_per_pixel: u32,
    ) -> Result<Vec<DecodedFrame>, FrameDecodeError> {
        let frame = assemble_s_gmc_vop_frame(
            &self.store,
            mb_width,
            mb_height,
            entries,
            geometry,
            vop_rounding_type,
            sample_mode,
            bits_per_pixel,
        )?;
        Ok(self.accept_anchor(frame))
    }

    /// Decode a progressive B-VOP in coding order against the bracketing
    /// anchors. A B-VOP is displayed immediately (§6.1.3.8) and never
    /// enters the chain.
    pub fn push_b_vop(
        &mut self,
        mb_width: usize,
        mb_height: usize,
        entries: &[crate::bvop_mv::BVopMbTexturedDecode],
        vop_rounding_type: u8,
        sample_mode: crate::bvop_prediction::BVopSampleMode,
        bits_per_pixel: u32,
    ) -> Result<Vec<DecodedFrame>, FrameDecodeError> {
        let frame = assemble_b_vop_frame(
            &self.store,
            mb_width,
            mb_height,
            entries,
            vop_rounding_type,
            sample_mode,
            bits_per_pixel,
        )?;
        // A B-VOP is output right away, in coding-order position (which
        // is its display-order position between the two anchors).
        Ok(vec![frame])
    }

    /// Release the final held anchor at end-of-sequence. Returns the
    /// held anchor (if any) for display, leaving the decoder empty of a
    /// pending anchor.
    pub fn flush(&mut self) -> Vec<DecodedFrame> {
        match self.pending_anchor.take() {
            Some(f) => vec![f],
            None => Vec::new(),
        }
    }

    /// Common anchor-handling: the new anchor displaces the held one
    /// (which becomes displayable) and is itself pushed into the chain as
    /// the new reference *and* held back for the §6.1.3.8 one-slot delay.
    fn accept_anchor(&mut self, frame: DecodedFrame) -> Vec<DecodedFrame> {
        // Push into the reference chain so subsequent VOPs can reference
        // it (this is decode-order state, independent of display delay).
        self.store.push_anchor(frame.clone());
        // The previously-held anchor is now displayable.
        let mut out = Vec::new();
        if let Some(prev) = self.pending_anchor.take() {
            out.push(prev);
        }
        self.pending_anchor = Some(frame);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::InterMacroblock;
    use crate::bvop::BVopMbType;
    use crate::bvop_mv::{BVopMbDecode, BVopMbTexturedDecode};
    use crate::bvop_prediction::{BVopMvPair, BVopPredictionMode, BVopSampleMode};
    use crate::motion::MotionVector;
    use crate::pvop_mv::PvopMbMotion;
    use crate::reconstruct::{MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};
    use crate::vop::VopCodingType;

    fn flat_recon(luma: i32) -> ReconstructedMacroblock {
        ReconstructedMacroblock {
            luma: [[luma; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[luma; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[luma; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        }
    }

    fn skip_p() -> Vec<PVopMbContent> {
        vec![PVopMbContent::Inter {
            motion: PvopMbMotion::Skipped,
            residual: InterMacroblock::zero(),
        }]
    }

    /// A P-VOP made of a single intra macroblock with a distinguishable
    /// luma value (so its pixels are not just a copy of the reference).
    fn intra_p(luma: i32) -> Vec<PVopMbContent> {
        vec![PVopMbContent::Intra(flat_recon(luma))]
    }

    fn fwd_b() -> Vec<BVopMbTexturedDecode> {
        let zero = MotionVector { x: 0, y: 0 };
        vec![BVopMbTexturedDecode {
            motion: BVopMbDecode {
                mb_type: BVopMbType::Forward,
                prediction_mode: BVopPredictionMode::ForwardOnly,
                cbpb: None,
                dbquant_delta: None,
                mvs: [BVopMvPair {
                    forward: zero,
                    backward: zero,
                }; 4],
                forward_chroma_mv: zero,
                backward_chroma_mv: zero,
            },
            residual: InterMacroblock::zero(),
            quantiser_scale: 8,
        }]
    }

    #[test]
    fn leading_i_vop_outputs_nothing_until_next_anchor() {
        let mut dec = SequenceDecoder::new();
        // First I-VOP: held back, nothing displayed yet (§6.1.3.8 "no VOP
        // is output").
        let out = dec.push_i_vop(1, 1, &[flat_recon(10)]).unwrap();
        assert!(out.is_empty());
        // Next anchor (P) releases the I-VOP for display.
        let out = dec
            .push_p_vop(1, 1, &skip_p(), 0, BVopSampleMode::HalfPel, 8)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].luma_at(0, 0), Some(10));
        assert_eq!(out[0].coding_type(), VopCodingType::I);
    }

    #[test]
    fn worked_example_1i_4p_2b_3b_displays_in_order() {
        // §6.1.3.8 worked example: coded 1I 4P 2B 3B -> display
        // 1I 2B 3B 4P. Use distinguishable luma values: I=10, P=40,
        // B's copy the forward anchor (I=10).
        let mut dec = SequenceDecoder::new();
        let mut display = Vec::new();

        display.extend(dec.push_i_vop(1, 1, &[flat_recon(10)]).unwrap()); // [] held
        display.extend(
            dec.push_p_vop(1, 1, &intra_p(40), 0, BVopSampleMode::HalfPel, 8)
                .unwrap(),
        ); // -> 1I
        display.extend(
            dec.push_b_vop(1, 1, &fwd_b(), 0, BVopSampleMode::HalfPel, 8)
                .unwrap(),
        ); // -> 2B
        display.extend(
            dec.push_b_vop(1, 1, &fwd_b(), 0, BVopSampleMode::HalfPel, 8)
                .unwrap(),
        ); // -> 3B
        display.extend(dec.flush()); // -> 4P

        let coding: Vec<_> = display.iter().map(|f| f.coding_type()).collect();
        assert_eq!(
            coding,
            vec![
                VopCodingType::I, // 1I
                VopCodingType::B, // 2B
                VopCodingType::B, // 3B
                VopCodingType::P, // 4P
            ]
        );
        // 1I luma 10; 2B/3B forward-copy the forward anchor (I=10); the
        // 4P is the intra P (luma 40), displayed last.
        assert_eq!(display[0].luma_at(0, 0), Some(10)); // I
        assert_eq!(display[1].luma_at(0, 0), Some(10)); // 2B (forward = I)
        assert_eq!(display[2].luma_at(0, 0), Some(10)); // 3B (forward = I)
        assert_eq!(display[3].luma_at(0, 0), Some(40)); // 4P (intra 40)
    }

    #[test]
    fn no_b_vops_is_identity_order() {
        // Without B-VOPs decode order == display order, but the one-slot
        // anchor delay still applies (each anchor displays one push late).
        let mut dec = SequenceDecoder::new();
        let mut display = Vec::new();
        display.extend(dec.push_i_vop(1, 1, &[flat_recon(10)]).unwrap()); // []
        display.extend(
            dec.push_p_vop(1, 1, &skip_p(), 0, BVopSampleMode::HalfPel, 8)
                .unwrap(),
        ); // I
        display.extend(
            dec.push_p_vop(1, 1, &skip_p(), 0, BVopSampleMode::HalfPel, 8)
                .unwrap(),
        ); // P1
        display.extend(dec.flush()); // P2
        assert_eq!(display.len(), 3);
        let coding: Vec<_> = display.iter().map(|f| f.coding_type()).collect();
        assert_eq!(
            coding,
            vec![VopCodingType::I, VopCodingType::P, VopCodingType::P]
        );
    }

    #[test]
    fn b_vop_references_both_bracketing_anchors() {
        // I=10 (forward) and P=40 (backward) bracket a backward-mode
        // B-VOP, which should copy the P (backward) anchor = 40.
        let mut dec = SequenceDecoder::new();
        dec.push_i_vop(1, 1, &[flat_recon(10)]).unwrap();
        dec.push_p_vop(1, 1, &intra_p(40), 0, BVopSampleMode::HalfPel, 8)
            .unwrap();
        let zero = MotionVector { x: 0, y: 0 };
        let bwd = vec![BVopMbTexturedDecode {
            motion: BVopMbDecode {
                mb_type: BVopMbType::Backward,
                prediction_mode: BVopPredictionMode::BackwardOnly,
                cbpb: None,
                dbquant_delta: None,
                mvs: [BVopMvPair {
                    forward: zero,
                    backward: zero,
                }; 4],
                forward_chroma_mv: zero,
                backward_chroma_mv: zero,
            },
            residual: InterMacroblock::zero(),
            quantiser_scale: 8,
        }];
        let out = dec
            .push_b_vop(1, 1, &bwd, 0, BVopSampleMode::HalfPel, 8)
            .unwrap();
        assert_eq!(out.len(), 1);
        // Backward anchor is the intra P-VOP (luma 40); backward-mode
        // B-VOP copies it, proving the backward slot is selected (not the
        // forward I=10).
        assert_eq!(out[0].luma_at(0, 0), Some(40));
        assert_eq!(out[0].coding_type(), VopCodingType::B);
    }

    #[test]
    fn flush_on_empty_decoder_is_empty() {
        let mut dec = SequenceDecoder::new();
        assert!(dec.flush().is_empty());
    }
}

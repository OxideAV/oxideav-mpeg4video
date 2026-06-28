//! Decoded-picture buffer (frame store) for the §7.6.1 reference-frame
//! chain.
//!
//! The per-macroblock motion-compensation drivers ([`MvDriver`],
//! [`BVopMvDriver`]) consume *reference planes* — borrowed
//! [`ReferenceVop`] views over previously-decoded VOP samples — and emit
//! one [`ReconstructedMacroblock`] per macroblock. Until now the caller
//! had to materialise those planes by hand and decide, per VOP, which
//! decoded frame is the forward (past) anchor and which is the backward
//! (future) anchor. This module owns that bookkeeping.
//!
//! Two layers:
//!
//! * [`DecodedFrame`] — owns the three 4:2:0 sample planes (luma +
//!   Cb + Cr) of one fully-decoded VOP as flat `Vec<u8>` buffers. A
//!   macroblock decode loop [`DecodedFrame::blit_macroblock`]s each
//!   [`ReconstructedMacroblock`] into place; the finished frame then
//!   hands out [`ReferenceVop`] plane views via
//!   [`DecodedFrame::luma_reference`] / [`Self::cb_reference`] /
//!   [`Self::cr_reference`] for the next VOP's motion compensation.
//!
//! * [`FrameStore`] — holds the forward (past) and backward (future)
//!   anchor frames and threads the §7.6.1 reference-frame chain: after
//!   an I-, P-, or S(GMC)-VOP is decoded it becomes the new past anchor
//!   (the old past anchor is retired); a B-VOP never updates the chain
//!   (§7.6.1: B-VOPs are not used as references). The §7.5.2.1.2 /
//!   line-17467 forward-vs-backward reference-VOP selection for a B-VOP
//!   is exposed via [`FrameStore::b_vop_reference_views`].
//!
//! All sample values live in the §7.3 step-3 display range
//! `[0, 2^bpp - 1]`; [`ReconstructedMacroblock`] guarantees that, and
//! the blit clamps defensively into `u8` (MPEG-4 Part 2's natural
//! `bits_per_pixel == 8` baseline — wider-depth frame storage is a
//! separate follow-up, tracked in the crate README).

use crate::half_sample::ReferenceVop;
use crate::reconstruct::{ReconstructedMacroblock, MACROBLOCK_CHROMA_SIDE, MACROBLOCK_LUMA_SIDE};
use crate::vop::VopCodingType;

/// Errors from frame-store construction and macroblock blitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameStoreError {
    /// A zero width or height was requested, or a dimension was not a
    /// whole number of macroblocks. MPEG-4 Part 2 luma planes are an
    /// integral number of 16×16 macroblocks (§6.3.2 padding to the next
    /// macroblock boundary); the frame store stores the padded plane.
    InvalidDimensions {
        /// Requested luma width in samples.
        width: usize,
        /// Requested luma height in samples.
        height: usize,
    },
    /// A macroblock blit addressed a column/row outside the frame's
    /// macroblock grid.
    MacroblockOutOfBounds {
        /// Macroblock column requested.
        mb_col: usize,
        /// Macroblock row requested.
        mb_row: usize,
        /// Frame width in macroblocks.
        mb_width: usize,
        /// Frame height in macroblocks.
        mb_height: usize,
    },
}

/// One fully-decoded 4:2:0 VOP held in the decoded-picture buffer.
///
/// The luma plane is `width × height` samples; the two chroma planes
/// are `(width / 2) × (height / 2)` per §6.1.3.4 4:2:0 sampling. `width`
/// and `height` are the macroblock-padded luma dimensions (a whole
/// number of 16×16 macroblocks), so a partial macroblock at the right /
/// bottom edge still has full backing storage to blit into and to read
/// back from during the next VOP's §7.6.4 last-pel-clamped motion
/// compensation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    luma: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    width: usize,
    height: usize,
    coding_type: VopCodingType,
}

impl DecodedFrame {
    /// Allocate a mid-grey frame of `width × height` luma samples
    /// (chroma `width/2 × height/2`), tagged with its `coding_type`.
    ///
    /// `width` and `height` must be non-zero multiples of
    /// [`MACROBLOCK_LUMA_SIDE`] (16). The buffers are filled with the
    /// §7.6.4 mid-grey value `128` so a frame that is only partially
    /// blitted still reads back deterministic samples (useful for the
    /// not-yet-decoded region of a frame mid-loop, and matching the
    /// §6.3.2 unrestricted-MV padding seed before real padding runs).
    pub fn new(
        width: usize,
        height: usize,
        coding_type: VopCodingType,
    ) -> Result<Self, FrameStoreError> {
        if width == 0
            || height == 0
            || width % MACROBLOCK_LUMA_SIDE != 0
            || height % MACROBLOCK_LUMA_SIDE != 0
        {
            return Err(FrameStoreError::InvalidDimensions { width, height });
        }
        let chroma_w = width / 2;
        let chroma_h = height / 2;
        Ok(Self {
            luma: vec![128u8; width * height],
            cb: vec![128u8; chroma_w * chroma_h],
            cr: vec![128u8; chroma_w * chroma_h],
            width,
            height,
            coding_type,
        })
    }

    /// Luma plane width in samples (macroblock-padded).
    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Luma plane height in samples (macroblock-padded).
    #[inline]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Frame width in 16×16 macroblocks.
    #[inline]
    pub const fn mb_width(&self) -> usize {
        self.width / MACROBLOCK_LUMA_SIDE
    }

    /// Frame height in 16×16 macroblocks.
    #[inline]
    pub const fn mb_height(&self) -> usize {
        self.height / MACROBLOCK_LUMA_SIDE
    }

    /// The VOP coding type this frame was decoded as.
    #[inline]
    pub const fn coding_type(&self) -> VopCodingType {
        self.coding_type
    }

    /// Borrow the luma plane samples (row-major, stride == width).
    #[inline]
    pub fn luma_samples(&self) -> &[u8] {
        &self.luma
    }

    /// Borrow the Cb plane samples (row-major, stride == width/2).
    #[inline]
    pub fn cb_samples(&self) -> &[u8] {
        &self.cb
    }

    /// Borrow the Cr plane samples (row-major, stride == width/2).
    #[inline]
    pub fn cr_samples(&self) -> &[u8] {
        &self.cr
    }

    /// A [`ReferenceVop`] view over the luma plane for motion
    /// compensation of the next VOP.
    #[inline]
    pub fn luma_reference(&self) -> ReferenceVop<'_> {
        // dimensions validated in `new`, so the view is always Some.
        ReferenceVop::new(&self.luma, self.width, self.height)
            .expect("luma dimensions validated at construction")
    }

    /// A [`ReferenceVop`] view over the Cb plane.
    #[inline]
    pub fn cb_reference(&self) -> ReferenceVop<'_> {
        ReferenceVop::new(&self.cb, self.width / 2, self.height / 2)
            .expect("chroma dimensions validated at construction")
    }

    /// A [`ReferenceVop`] view over the Cr plane.
    #[inline]
    pub fn cr_reference(&self) -> ReferenceVop<'_> {
        ReferenceVop::new(&self.cr, self.width / 2, self.height / 2)
            .expect("chroma dimensions validated at construction")
    }

    /// Blit one reconstructed macroblock into this frame at macroblock
    /// position `(mb_col, mb_row)`.
    ///
    /// The 16×16 luma block lands at luma pixel `(16 * mb_col,
    /// 16 * mb_row)`; the two 8×8 chroma blocks land at chroma pixel
    /// `(8 * mb_col, 8 * mb_row)`. Each `i32` sample is clamped into
    /// `[0, 255]` defensively before storing — [`ReconstructedMacroblock`]
    /// already guarantees the §7.3 display range for `bits_per_pixel ==
    /// 8`, so the clamp is a no-op on valid input.
    pub fn blit_macroblock(
        &mut self,
        mb_col: usize,
        mb_row: usize,
        mb: &ReconstructedMacroblock,
    ) -> Result<(), FrameStoreError> {
        let mb_w = self.mb_width();
        let mb_h = self.mb_height();
        if mb_col >= mb_w || mb_row >= mb_h {
            return Err(FrameStoreError::MacroblockOutOfBounds {
                mb_col,
                mb_row,
                mb_width: mb_w,
                mb_height: mb_h,
            });
        }

        // Luma: 16×16 at (16*col, 16*row), stride == width.
        let luma_x0 = mb_col * MACROBLOCK_LUMA_SIDE;
        let luma_y0 = mb_row * MACROBLOCK_LUMA_SIDE;
        for (dy, row) in mb.luma.iter().enumerate() {
            let base = (luma_y0 + dy) * self.width + luma_x0;
            for (dx, &sample) in row.iter().enumerate() {
                self.luma[base + dx] = sample.clamp(0, 255) as u8;
            }
        }

        // Chroma: 8×8 at (8*col, 8*row), stride == width/2.
        let chroma_w = self.width / 2;
        let chroma_x0 = mb_col * MACROBLOCK_CHROMA_SIDE;
        let chroma_y0 = mb_row * MACROBLOCK_CHROMA_SIDE;
        for (dy, (cb_row, cr_row)) in mb.cb.iter().zip(mb.cr.iter()).enumerate() {
            let base = (chroma_y0 + dy) * chroma_w + chroma_x0;
            for (dx, (&cb_s, &cr_s)) in cb_row.iter().zip(cr_row.iter()).enumerate() {
                self.cb[base + dx] = cb_s.clamp(0, 255) as u8;
                self.cr[base + dx] = cr_s.clamp(0, 255) as u8;
            }
        }
        Ok(())
    }

    /// Read back a single luma sample (debug / test helper). Returns
    /// `None` if out of bounds.
    #[inline]
    pub fn luma_at(&self, x: usize, y: usize) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.luma[y * self.width + x])
    }
}

/// The §7.6.1 reference-frame chain: the decoded-picture buffer that
/// remembers which decoded VOP is the forward (past) anchor and which
/// is the backward (future) anchor.
///
/// Update discipline (§7.6.1, §7.5.2.1.2):
///
/// * Decoding an **I-, P-, or S(GMC)-VOP** advances the chain: the VOP
///   just decoded becomes the new *backward* (most recently decoded in
///   the future) anchor and the previous backward anchor slides into
///   the *forward* (past) slot. In display order this means the two
///   anchors a B-VOP interpolates between are exactly the previous and
///   the next I/P/S-VOP.
/// * Decoding a **B-VOP** never updates the chain — a B-VOP is never a
///   reference (§7.6.1).
///
/// The store is seeded empty; the first decoded I-VOP populates the
/// backward slot with the forward slot still empty (a leading I-VOP has
/// no past reference, which is correct — it is intra-only).
#[derive(Debug, Clone, Default)]
pub struct FrameStore {
    /// Most recently decoded I/P/S-VOP in the *past* (forward reference).
    forward: Option<DecodedFrame>,
    /// Most recently decoded I/P/S-VOP in the *future* (backward
    /// reference) relative to any B-VOP that sits between the two.
    backward: Option<DecodedFrame>,
}

impl FrameStore {
    /// An empty store — no references decoded yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// The forward (past) reference frame, if one has been decoded.
    #[inline]
    pub fn forward(&self) -> Option<&DecodedFrame> {
        self.forward.as_ref()
    }

    /// The backward (future) reference frame, if one has been decoded.
    #[inline]
    pub fn backward(&self) -> Option<&DecodedFrame> {
        self.backward.as_ref()
    }

    /// Advance the reference-frame chain with a freshly-decoded
    /// **anchor** (I-, P-, or S(GMC)-VOP). The new frame becomes the
    /// backward anchor; the old backward anchor becomes the forward
    /// anchor (the old forward anchor is retired).
    ///
    /// The `frame.coding_type()` must be one of I / P / S. Passing a
    /// B-VOP is a caller error: it panics in debug builds to surface the
    /// misuse early and is a no-op in release builds (the chain is left
    /// untouched rather than corrupted).
    pub fn push_anchor(&mut self, frame: DecodedFrame) {
        debug_assert!(
            !matches!(frame.coding_type(), VopCodingType::B),
            "B-VOP must never enter the reference-frame chain (§7.6.1)"
        );
        if matches!(frame.coding_type(), VopCodingType::B) {
            // Release safety net: do not corrupt the chain on misuse.
            return;
        }
        // Slide backward -> forward, install new frame as backward.
        self.forward = self.backward.take();
        self.backward = Some(frame);
    }

    /// Select the single reference frame for a **P- or S(GMC)-VOP**.
    ///
    /// §7.5.2.1.2 / §7.6.1: a P- or S(GMC)-VOP predicts from the forward
    /// (past) reference VOP, which is the most recently decoded I/P/S-VOP
    /// — i.e. the *backward* slot of this store (the one just pushed as
    /// the new anchor sits in `backward`; the P-VOP being decoded *now*
    /// has not been pushed yet, so its past reference is the current
    /// `backward`).
    ///
    /// Returns `None` if no anchor has been decoded yet (a P-VOP with no
    /// preceding I-VOP — a malformed stream).
    #[inline]
    pub fn p_vop_reference(&self) -> Option<&DecodedFrame> {
        self.backward.as_ref()
    }

    /// Select the (forward, backward) anchor pair for a **B-VOP**.
    ///
    /// §7.6.1: a B-VOP interpolates between the forward (past) anchor and
    /// the backward (future) anchor — the two I/P/S-VOPs that bracket it.
    /// Both slots must be populated; returns `None` if either is missing
    /// (a B-VOP before two anchors have been decoded — malformed).
    #[inline]
    pub fn b_vop_references(&self) -> Option<(&DecodedFrame, &DecodedFrame)> {
        match (self.forward.as_ref(), self.backward.as_ref()) {
            (Some(f), Some(b)) => Some((f, b)),
            _ => None,
        }
    }

    /// Build the six §7.6.9.5.1 [`BVopAnchorPlanes`] reference views for a
    /// B-VOP macroblock reconstruction from the current chain.
    ///
    /// The caller must hold the returned [`ReferenceVop`]s alive for the
    /// duration of the reconstruction — they borrow this store. The
    /// six-tuple is `(forward_luma, backward_luma, forward_cb,
    /// backward_cb, forward_cr, backward_cr)`. Returns `None` if the
    /// chain lacks either anchor.
    #[allow(clippy::type_complexity)]
    pub fn b_vop_reference_views(
        &self,
    ) -> Option<(
        ReferenceVop<'_>,
        ReferenceVop<'_>,
        ReferenceVop<'_>,
        ReferenceVop<'_>,
        ReferenceVop<'_>,
        ReferenceVop<'_>,
    )> {
        let (f, b) = self.b_vop_references()?;
        Some((
            f.luma_reference(),
            b.luma_reference(),
            f.cb_reference(),
            b.cb_reference(),
            f.cr_reference(),
            b.cr_reference(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_mb(luma: i32, cb: i32, cr: i32) -> ReconstructedMacroblock {
        ReconstructedMacroblock {
            luma: [[luma; MACROBLOCK_LUMA_SIDE]; MACROBLOCK_LUMA_SIDE],
            cb: [[cb; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
            cr: [[cr; MACROBLOCK_CHROMA_SIDE]; MACROBLOCK_CHROMA_SIDE],
        }
    }

    #[test]
    fn new_rejects_non_macroblock_dimensions() {
        assert!(matches!(
            DecodedFrame::new(17, 16, VopCodingType::I),
            Err(FrameStoreError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            DecodedFrame::new(16, 0, VopCodingType::I),
            Err(FrameStoreError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn new_seeds_mid_grey() {
        let f = DecodedFrame::new(32, 16, VopCodingType::P).unwrap();
        assert_eq!(f.width(), 32);
        assert_eq!(f.height(), 16);
        assert_eq!(f.mb_width(), 2);
        assert_eq!(f.mb_height(), 1);
        assert_eq!(f.luma_at(0, 0), Some(128));
        assert_eq!(f.luma_at(31, 15), Some(128));
        assert_eq!(f.coding_type(), VopCodingType::P);
    }

    #[test]
    fn blit_places_macroblock_at_grid_position() {
        let mut f = DecodedFrame::new(32, 16, VopCodingType::I).unwrap();
        f.blit_macroblock(0, 0, &flat_mb(10, 20, 30)).unwrap();
        f.blit_macroblock(1, 0, &flat_mb(40, 50, 60)).unwrap();

        // Left MB luma occupies x in [0, 16), right MB x in [16, 32).
        assert_eq!(f.luma_at(0, 0), Some(10));
        assert_eq!(f.luma_at(15, 15), Some(10));
        assert_eq!(f.luma_at(16, 0), Some(40));
        assert_eq!(f.luma_at(31, 15), Some(40));

        // Chroma: left MB cb in [0,8), right MB in [8,16).
        let cb = f.cb_samples();
        let cw = 16; // width/2
        assert_eq!(cb[0], 20);
        assert_eq!(cb[7], 20);
        assert_eq!(cb[8], 50);
        assert_eq!(cb[15], 50);
        // bottom-right chroma sample of the right MB.
        assert_eq!(cb[7 * cw + 15], 50);
    }

    #[test]
    fn blit_clamps_out_of_range_samples() {
        let mut f = DecodedFrame::new(16, 16, VopCodingType::I).unwrap();
        f.blit_macroblock(0, 0, &flat_mb(300, -5, 128)).unwrap();
        assert_eq!(f.luma_at(0, 0), Some(255));
        assert_eq!(f.cb_samples()[0], 0);
        assert_eq!(f.cr_samples()[0], 128);
    }

    #[test]
    fn blit_rejects_out_of_grid() {
        let mut f = DecodedFrame::new(16, 16, VopCodingType::I).unwrap();
        assert!(matches!(
            f.blit_macroblock(1, 0, &flat_mb(0, 0, 0)),
            Err(FrameStoreError::MacroblockOutOfBounds { .. })
        ));
    }

    #[test]
    fn reference_views_read_back_blitted_samples() {
        let mut f = DecodedFrame::new(16, 16, VopCodingType::I).unwrap();
        f.blit_macroblock(0, 0, &flat_mb(77, 88, 99)).unwrap();
        let luma = f.luma_reference();
        assert_eq!(luma.width(), 16);
        assert_eq!(luma.height(), 16);
        assert_eq!(luma.fetch_clamped(0, 0), 77);
        // §7.6.4 clamp: reading past the right edge returns the edge pel.
        assert_eq!(luma.fetch_clamped(100, 100), 77);
        let cb = f.cb_reference();
        assert_eq!(cb.width(), 8);
        assert_eq!(cb.fetch_clamped(0, 0), 88);
        let cr = f.cr_reference();
        assert_eq!(cr.fetch_clamped(0, 0), 99);
    }

    fn anchor(coding: VopCodingType, seed: u8) -> DecodedFrame {
        let mut f = DecodedFrame::new(16, 16, coding).unwrap();
        f.blit_macroblock(0, 0, &flat_mb(seed as i32, seed as i32, seed as i32))
            .unwrap();
        f
    }

    #[test]
    fn empty_store_has_no_references() {
        let fs = FrameStore::new();
        assert!(fs.forward().is_none());
        assert!(fs.backward().is_none());
        assert!(fs.p_vop_reference().is_none());
        assert!(fs.b_vop_references().is_none());
        assert!(fs.b_vop_reference_views().is_none());
    }

    #[test]
    fn first_anchor_populates_backward_only() {
        let mut fs = FrameStore::new();
        fs.push_anchor(anchor(VopCodingType::I, 10));
        // Leading I-VOP: no past reference, it is the most-recent anchor.
        assert!(fs.forward().is_none());
        assert_eq!(fs.backward().unwrap().luma_at(0, 0), Some(10));
        // A P-VOP decoded next predicts from this I-VOP.
        assert_eq!(fs.p_vop_reference().unwrap().luma_at(0, 0), Some(10));
        // A B-VOP cannot bracket with only one anchor.
        assert!(fs.b_vop_references().is_none());
    }

    #[test]
    fn second_anchor_slides_chain() {
        let mut fs = FrameStore::new();
        fs.push_anchor(anchor(VopCodingType::I, 10));
        fs.push_anchor(anchor(VopCodingType::P, 20));
        // I slides to forward (past); P is backward (most-recent).
        assert_eq!(fs.forward().unwrap().luma_at(0, 0), Some(10));
        assert_eq!(fs.backward().unwrap().luma_at(0, 0), Some(20));
        // A B-VOP between them brackets I (forward) and P (backward).
        let (f, b) = fs.b_vop_references().unwrap();
        assert_eq!(f.luma_at(0, 0), Some(10));
        assert_eq!(b.luma_at(0, 0), Some(20));
        // A P-VOP decoded now predicts from the most-recent anchor (P=20).
        assert_eq!(fs.p_vop_reference().unwrap().luma_at(0, 0), Some(20));
    }

    #[test]
    fn third_anchor_retires_oldest() {
        let mut fs = FrameStore::new();
        fs.push_anchor(anchor(VopCodingType::I, 10));
        fs.push_anchor(anchor(VopCodingType::P, 20));
        fs.push_anchor(anchor(VopCodingType::P, 30));
        // 10 is retired; chain is now (20, 30).
        assert_eq!(fs.forward().unwrap().luma_at(0, 0), Some(20));
        assert_eq!(fs.backward().unwrap().luma_at(0, 0), Some(30));
    }

    #[test]
    fn b_vop_reference_views_bind_both_anchors() {
        let mut fs = FrameStore::new();
        fs.push_anchor(anchor(VopCodingType::I, 10));
        fs.push_anchor(anchor(VopCodingType::P, 20));
        let (fl, bl, fcb, bcb, fcr, bcr) = fs.b_vop_reference_views().unwrap();
        assert_eq!(fl.fetch_clamped(0, 0), 10);
        assert_eq!(bl.fetch_clamped(0, 0), 20);
        assert_eq!(fcb.fetch_clamped(0, 0), 10);
        assert_eq!(bcb.fetch_clamped(0, 0), 20);
        assert_eq!(fcr.fetch_clamped(0, 0), 10);
        assert_eq!(bcr.fetch_clamped(0, 0), 20);
    }

    #[test]
    fn b_vop_does_not_enter_chain_in_release() {
        // The release-mode safety net: a B-VOP push is a no-op. (Debug
        // builds assert; this test documents the release contract via
        // the public API without tripping the debug_assert, so we only
        // run the no-op check when debug assertions are disabled.)
        if cfg!(debug_assertions) {
            return;
        }
        let mut fs = FrameStore::new();
        fs.push_anchor(anchor(VopCodingType::I, 10));
        fs.push_anchor(anchor(VopCodingType::B, 99));
        // Chain unchanged: backward still the I-VOP.
        assert_eq!(fs.backward().unwrap().luma_at(0, 0), Some(10));
    }
}

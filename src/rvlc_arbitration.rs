//! §E.1.4.4.2.1 two-way RVLC error-recovery strategy selection.
//!
//! When `reversible_vlc == 1` and the forward decode of a video packet's
//! DCT-coefficient region hits an error (an illegal RVLC, an out-of-range
//! escape, a bad stuffing pattern, or more than 64 coefficients in a
//! block — the §E.1.4.4.1 error conditions), the bitstream is decoded
//! *both* forward and backward and the two partial results are stitched
//! together. The forward and backward EVENT decoders themselves live in
//! [`crate::texture`] ([`decode_ac_events_rvlc`] /
//! [`decode_ac_events_rvlc_reverse`]); this module implements the
//! §E.1.4.4.2.1 *arbitration* that, given how far each direction got,
//! decides how many macroblocks to keep from the beginning (the forward
//! result) and how many from the end (the backward result). The
//! remaining middle macroblocks straddle the error and are discarded.
//!
//! ## Definitions (verbatim from §E.1.4.4.2)
//!
//! * `L`  — total number of bits for the DCT-coefficients part in a VP.
//! * `N`  — total number of macroblocks (MBs) in a VP.
//! * `L1` — number of bits which can be decoded in a forward decoding.
//! * `L2` — number of bits which can be decoded in a backward decoding.
//! * `N1` — number of MBs which can be completely decoded forward
//!   (`0 <= N1 <= N-1`).
//! * `N2` — number of MBs which can be completely decoded backward
//!   (`0 <= N2 <= N-1`).
//! * `f_mb(S)` — number of decoded MBs when `S` bits can be decoded in
//!   the forward direction. The counter increments once one *or more*
//!   bits of a MB have been decoded.
//! * `b_mb(S)` — number of decoded MBs when `S` bits can be decoded in
//!   the backward direction.
//! * `T` — threshold (the spec uses 90), see [`RVLC_THRESHOLD`].
//!
//! ## The four strategies (§E.1.4.4.2.1)
//!
//! The strategy is chosen from two predicates — whether the two
//! directions' decodable bit-counts overlap (`L1 + L2 >= L`) and whether
//! their completely-decoded MB-counts overlap (`N1 + N2 >= N`):
//!
//! | strategy | predicate                       | keep from beginning      | keep from end             |
//! |----------|---------------------------------|--------------------------|---------------------------|
//! | 1        | `L1+L2 < L` and `N1+N2 < N`     | `f_mb(L1 - T)`           | `b_mb(L2 - T)`            |
//! | 2        | `L1+L2 < L` and `N1+N2 >= N`    | `N - N2 - 1`             | `N - N1 - 1`              |
//! | 3        | `L1+L2 >= L` and `N1+N2 < N`    | `N - b_mb(L2)`           | `N - f_mb(L1)`            |
//! | 4        | `L1+L2 >= L` and `N1+N2 >= N`   | `min(N-b_mb(L2), N-N2-1)`| `min(N-f_mb(L1), N-N1-1)` |
//!
//! "Keep from the beginning" is a count of macroblocks taken from the
//! forward result (MB indices `0 .. keep_front`); "keep from the end" is
//! a count taken from the backward result (the last `keep_back` MB
//! indices, `N - keep_back .. N`). The dark region of Figures E.4–E.7
//! — the MBs in between — is discarded.
//!
//! ## §E.1.4.4.2.2 INTRA-MB concealment
//!
//! Once a video packet is determined to contain errors, *every* INTRA MB
//! in the packet is concealed rather than displayed, even ones the
//! strategy would otherwise have kept: "Although these intra MBs are
//! thought to be correct, the result of displaying an Intra MB that does
//! contain an error can substantially degrade the quality of the video."
//! [`RvlcArbitration::displayed_mbs`] applies this pass to a kept-MB
//! decision given a per-MB INTRA predicate.
//!
//! ## Spec references
//!
//! All numbers come from ISO/IEC 14496-2:2004 (3rd edition) §E.1.4.4,
//! read by the agent from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.
//!
//! [`decode_ac_events_rvlc`]: crate::texture::decode_ac_events_rvlc
//! [`decode_ac_events_rvlc_reverse`]: crate::texture::decode_ac_events_rvlc_reverse

/// §E.1.4.4.2 threshold `T`. The spec states "Threshold (90 is used
/// now)"; Strategy 1 keeps `f_mb(L1 - T)` / `b_mb(L2 - T)` MBs, backing
/// off `T` bits from each error position before counting recovered MBs.
pub const RVLC_THRESHOLD: i64 = 90;

/// Which of the four §E.1.4.4.2.1 strategies was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvlcStrategy {
    /// `L1 + L2 < L` and `N1 + N2 < N` — Figure E.4.
    Strategy1,
    /// `L1 + L2 < L` and `N1 + N2 >= N` — Figure E.5.
    Strategy2,
    /// `L1 + L2 >= L` and `N1 + N2 < N` — Figure E.6.
    Strategy3,
    /// `L1 + L2 >= L` and `N1 + N2 >= N` — Figure E.7.
    Strategy4,
}

/// The §E.1.4.4.2 inputs describing how far each decode direction got.
///
/// All bit and MB counts are taken over the DCT-coefficients part of a
/// single video packet, as the spec defines them. `forward_mb_bits` and
/// `backward_mb_bits` carry the per-MB cumulative bit costs used to
/// evaluate `f_mb()` / `b_mb()`:
///
/// * `forward_mb_bits[i]` = total forward-decoded bits *after* finishing
///   MB `i` (cumulative; strictly increasing as MBs are consumed).
/// * `backward_mb_bits[i]` = total backward-decoded bits after finishing
///   the `i`-th MB *counted from the end* of the packet.
///
/// These are exactly the running bit totals a decoder already produces
/// while walking the EVENT stream; `f_mb`/`b_mb` are then monotone
/// step-inverse lookups over them.
#[derive(Debug, Clone)]
pub struct RvlcArbitrationInput {
    /// `N` — total macroblocks in the video packet.
    pub total_mbs: usize,
    /// `L` — total DCT-coefficient bits in the video packet.
    pub total_bits: i64,
    /// `L1` — bits decodable in the forward direction.
    pub forward_bits: i64,
    /// `L2` — bits decodable in the backward direction.
    pub backward_bits: i64,
    /// `N1` — MBs completely decoded forward (`0 <= N1 <= N-1`).
    pub forward_complete_mbs: usize,
    /// `N2` — MBs completely decoded backward (`0 <= N2 <= N-1`).
    pub backward_complete_mbs: usize,
    /// Per-MB cumulative forward bit costs (see struct docs). Index `i`
    /// is the running total after MB `i`; used to evaluate `f_mb(S)`.
    pub forward_mb_bits: Vec<i64>,
    /// Per-MB cumulative backward bit costs counted from the end of the
    /// packet; used to evaluate `b_mb(S)`.
    pub backward_mb_bits: Vec<i64>,
}

/// The §E.1.4.4.2.1 decision: which strategy fired and how many MBs to
/// keep from each end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RvlcArbitration {
    /// Which strategy was selected.
    pub strategy: RvlcStrategy,
    /// Number of MBs to keep from the beginning (forward result). MB
    /// indices `0 .. keep_front`.
    pub keep_front: usize,
    /// Number of MBs to keep from the end (backward result). MB indices
    /// `total_mbs - keep_back .. total_mbs`.
    pub keep_back: usize,
}

/// `f_mb(S)` / `b_mb(S)`: number of MBs whose cumulative bit cost is
/// `<= S`. `cumulative[i]` holds the running bit total after MB `i`, so
/// the count of fully-affordable MBs is the number of entries `<= S`.
///
/// Per §E.1.4.4.2 the counter increments once "equal to or more than one
/// bit can be decoded in a MB"; a non-positive `S` therefore yields 0.
/// The result is clamped to `[0, cumulative.len()]`.
fn mb_count_for_bits(cumulative: &[i64], s: i64) -> usize {
    if s <= 0 {
        return 0;
    }
    // cumulative is non-decreasing, so partition_point gives the count
    // of entries <= s directly.
    cumulative.partition_point(|&c| c <= s)
}

impl RvlcArbitrationInput {
    /// Evaluate `f_mb(S)`.
    pub fn f_mb(&self, s: i64) -> usize {
        mb_count_for_bits(&self.forward_mb_bits, s)
    }

    /// Evaluate `b_mb(S)`.
    pub fn b_mb(&self, s: i64) -> usize {
        mb_count_for_bits(&self.backward_mb_bits, s)
    }
}

impl RvlcArbitration {
    /// Run the §E.1.4.4.2.1 strategy selection over `input`, returning
    /// the selected strategy and the `keep_front` / `keep_back` MB
    /// counts.
    ///
    /// The two `keep_*` counts are each clamped to `[0, N]` and their
    /// sum is clamped so the two kept regions never overlap (a kept
    /// region from each end cannot together exceed `N` MBs); when the
    /// raw counts would overlap, the discarded middle collapses to zero
    /// and the front region is trimmed so `keep_front + keep_back == N`.
    pub fn select(input: &RvlcArbitrationInput) -> Self {
        let n = input.total_mbs;
        let l = input.total_bits;
        let l1 = input.forward_bits;
        let l2 = input.backward_bits;
        let n1 = input.forward_complete_mbs;
        let n2 = input.backward_complete_mbs;

        let bits_overlap = l1 + l2 >= l;
        let mbs_overlap = n1 + n2 >= n;

        let (strategy, raw_front, raw_back) = match (bits_overlap, mbs_overlap) {
            // Strategy 1: L1+L2 < L and N1+N2 < N
            (false, false) => (
                RvlcStrategy::Strategy1,
                input.f_mb(l1 - RVLC_THRESHOLD) as i64,
                input.b_mb(l2 - RVLC_THRESHOLD) as i64,
            ),
            // Strategy 2: L1+L2 < L and N1+N2 >= N
            (false, true) => (
                RvlcStrategy::Strategy2,
                n as i64 - n2 as i64 - 1,
                n as i64 - n1 as i64 - 1,
            ),
            // Strategy 3: L1+L2 >= L and N1+N2 < N
            (true, false) => (
                RvlcStrategy::Strategy3,
                n as i64 - input.b_mb(l2) as i64,
                n as i64 - input.f_mb(l1) as i64,
            ),
            // Strategy 4: L1+L2 >= L and N1+N2 >= N
            (true, true) => (
                RvlcStrategy::Strategy4,
                (n as i64 - input.b_mb(l2) as i64).min(n as i64 - n2 as i64 - 1),
                (n as i64 - input.f_mb(l1) as i64).min(n as i64 - n1 as i64 - 1),
            ),
        };

        let n_i = n as i64;
        let mut keep_front = raw_front.clamp(0, n_i);
        let keep_back = raw_back.clamp(0, n_i);
        // The two kept regions are disjoint (front from index 0, back
        // from index N): their combined length cannot exceed N. If the
        // raw counts overlap, the middle (discard) region is empty and
        // the back region's claim is honoured first (it sits adjacent to
        // the error tail), trimming the front to fill the rest.
        if keep_front + keep_back > n_i {
            keep_front = n_i - keep_back;
        }

        RvlcArbitration {
            strategy,
            keep_front: keep_front as usize,
            keep_back: keep_back as usize,
        }
    }

    /// §E.1.4.4.2.2 INTRA-MB concealment: returns the set of MB indices
    /// (`0 .. N`) that should actually be displayed after the strategy's
    /// kept regions are taken and then every INTRA MB in the *whole*
    /// packet is removed.
    ///
    /// `is_intra(i)` reports whether MB `i` is INTRA-coded. A kept MB is
    /// displayed only when it is non-INTRA; an INTRA MB anywhere in an
    /// errored packet is concealed (not displayed) per §E.1.4.4.2.2.
    /// MB indices in the discarded middle region are likewise absent.
    pub fn displayed_mbs(
        &self,
        total_mbs: usize,
        mut is_intra: impl FnMut(usize) -> bool,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        for i in 0..self.keep_front.min(total_mbs) {
            if !is_intra(i) {
                out.push(i);
            }
        }
        let back_start = total_mbs.saturating_sub(self.keep_back);
        // Avoid double-listing if the front region already reached it.
        let back_start = back_start.max(self.keep_front);
        for i in back_start..total_mbs {
            if !is_intra(i) {
                out.push(i);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn input(
        n: usize,
        l: i64,
        l1: i64,
        l2: i64,
        n1: usize,
        n2: usize,
        fwd: Vec<i64>,
        bwd: Vec<i64>,
    ) -> RvlcArbitrationInput {
        RvlcArbitrationInput {
            total_mbs: n,
            total_bits: l,
            forward_bits: l1,
            backward_bits: l2,
            forward_complete_mbs: n1,
            backward_complete_mbs: n2,
            forward_mb_bits: fwd,
            backward_mb_bits: bwd,
        }
    }

    #[test]
    fn f_mb_b_mb_are_monotone_step_inverses() {
        let inp = input(
            4,
            400,
            0,
            0,
            0,
            0,
            vec![100, 200, 300, 400],
            vec![100, 200, 300, 400],
        );
        // S below the first MB cost: 0 MBs.
        assert_eq!(inp.f_mb(0), 0);
        assert_eq!(inp.f_mb(99), 0);
        // S equal to a cumulative boundary counts that MB.
        assert_eq!(inp.f_mb(100), 1);
        assert_eq!(inp.f_mb(150), 1);
        assert_eq!(inp.f_mb(200), 2);
        assert_eq!(inp.f_mb(400), 4);
        // S beyond the last cost saturates.
        assert_eq!(inp.f_mb(99999), 4);
        // Non-positive S yields 0 per the counter rule.
        assert_eq!(inp.b_mb(0), 0);
        assert_eq!(inp.b_mb(-50), 0);
        assert_eq!(inp.b_mb(300), 3);
    }

    #[test]
    fn strategy1_picked_when_no_bit_or_mb_overlap() {
        // L1+L2 < L and N1+N2 < N.
        let fwd = vec![100, 200, 300, 400, 500];
        let bwd = vec![100, 200, 300, 400, 500];
        let inp = input(6, 2000, 300, 300, 2, 2, fwd, bwd);
        let dec = RvlcArbitration::select(&inp);
        assert_eq!(dec.strategy, RvlcStrategy::Strategy1);
        // f_mb(L1 - T) = f_mb(300 - 90) = f_mb(210) = 2 (100,200 <= 210).
        assert_eq!(dec.keep_front, 2);
        // b_mb(L2 - T) = b_mb(210) = 2.
        assert_eq!(dec.keep_back, 2);
    }

    #[test]
    fn strategy2_picked_when_mbs_overlap_but_bits_dont() {
        // L1+L2 < L and N1+N2 >= N.
        let inp = input(5, 2000, 400, 400, 3, 3, vec![100], vec![100]);
        let dec = RvlcArbitration::select(&inp);
        assert_eq!(dec.strategy, RvlcStrategy::Strategy2);
        // N - N2 - 1 = 5 - 3 - 1 = 1.
        assert_eq!(dec.keep_front, 1);
        // N - N1 - 1 = 1.
        assert_eq!(dec.keep_back, 1);
    }

    #[test]
    fn strategy3_picked_when_bits_overlap_but_mbs_dont() {
        // L1+L2 >= L and N1+N2 < N.
        let fwd = vec![100, 200, 300, 400, 500, 600];
        let bwd = vec![100, 200, 300, 400, 500, 600];
        let inp = input(6, 800, 500, 500, 1, 1, fwd, bwd);
        let dec = RvlcArbitration::select(&inp);
        assert_eq!(dec.strategy, RvlcStrategy::Strategy3);
        // b_mb(L2) = b_mb(500) = 5; N - 5 = 1.
        assert_eq!(dec.keep_front, 1);
        // f_mb(L1) = f_mb(500) = 5; N - 5 = 1.
        assert_eq!(dec.keep_back, 1);
    }

    #[test]
    fn strategy4_picked_when_both_overlap_takes_min() {
        // L1+L2 >= L and N1+N2 >= N.
        let fwd = vec![100, 200, 300, 400, 500, 600];
        let bwd = vec![100, 200, 300, 400, 500, 600];
        let inp = input(6, 800, 500, 500, 4, 4, fwd, bwd);
        let dec = RvlcArbitration::select(&inp);
        assert_eq!(dec.strategy, RvlcStrategy::Strategy4);
        // front = min(N - b_mb(L2), N - N2 - 1) = min(6-5, 6-4-1) = min(1,1) = 1.
        assert_eq!(dec.keep_front, 1);
        // back  = min(N - f_mb(L1), N - N1 - 1) = min(1, 1) = 1.
        assert_eq!(dec.keep_back, 1);
    }

    #[test]
    fn kept_regions_never_overlap() {
        // Construct a case whose raw counts would exceed N.
        let fwd = vec![10, 20, 30, 40];
        let bwd = vec![10, 20, 30, 40];
        // Strategy 3: keep_front = N - b_mb(L2), keep_back = N - f_mb(L1).
        // Pick L1, L2 small so b_mb/f_mb are small => keep_* large.
        let inp = input(4, 50, 30, 30, 0, 0, fwd, bwd);
        let dec = RvlcArbitration::select(&inp);
        // Raw: b_mb(30)=3 -> front = 1; f_mb(30)=3 -> back = 1. No overlap.
        assert!(dec.keep_front + dec.keep_back <= 4);

        // Force an overlap: keep bits overlapping (L small => Strategy 3)
        // but make per-MB costs large so f_mb/b_mb are 0 => keep_* = N each.
        let inp2 = input(
            4,
            10,
            10,
            10,
            0,
            0,
            vec![100, 200, 300, 400],
            vec![100, 200, 300, 400],
        );
        let dec2 = RvlcArbitration::select(&inp2);
        // L1+L2=20 >= L=10 and N1+N2=0 < 4 => Strategy 3.
        // b_mb(10)=0 -> front=4; f_mb(10)=0 -> back=4; sum=8 > 4 => trimmed.
        assert_eq!(dec2.strategy, RvlcStrategy::Strategy3);
        assert_eq!(dec2.keep_front + dec2.keep_back, 4);
        assert_eq!(dec2.keep_back, 4);
        assert_eq!(dec2.keep_front, 0);
    }

    #[test]
    fn intra_mbs_are_concealed_across_whole_packet() {
        // Keep front 2, back 2 of a 6-MB packet; MBs 1 and 4 are INTRA.
        let dec = RvlcArbitration {
            strategy: RvlcStrategy::Strategy1,
            keep_front: 2,
            keep_back: 2,
        };
        let intra = [false, true, false, false, true, false];
        let displayed = dec.displayed_mbs(6, |i| intra[i]);
        // Kept indices: 0,1 (front) and 4,5 (back). Drop INTRA 1 and 4.
        assert_eq!(displayed, vec![0, 5]);
    }

    #[test]
    fn displayed_mbs_no_double_listing_when_regions_meet() {
        // keep_front + keep_back == N: every MB is kept exactly once.
        let dec = RvlcArbitration {
            strategy: RvlcStrategy::Strategy2,
            keep_front: 2,
            keep_back: 2,
        };
        let displayed = dec.displayed_mbs(4, |_| false);
        assert_eq!(displayed, vec![0, 1, 2, 3]);
    }

    #[test]
    fn n1_n2_respect_zero_to_n_minus_one_bound() {
        // N2 = N-1 maximal: Strategy 2 front = N - (N-1) - 1 = 0.
        let inp = input(5, 2000, 100, 100, 4, 4, vec![10], vec![10]);
        let dec = RvlcArbitration::select(&inp);
        assert_eq!(dec.strategy, RvlcStrategy::Strategy2);
        assert_eq!(dec.keep_front, 0);
        assert_eq!(dec.keep_back, 0);
    }
}

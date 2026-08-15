//! Annex D video-rate-buffer (VBV) model + bit-budget-regulated
//! quantiser adaptation.
//!
//! The normative side is the Annex D.2 rate-buffer model: a decoder
//! buffer of `B = 16384 × vbv_buffer_size` bits fills from the channel
//! at up to the peak rate `Rvol(t) <= 400 × bit_rate` (item 3) and has
//! each coded VOP's `d_i` bits removed **instantaneously** at its
//! decoding time `t_i` (items 5–7; with the uniform input frame period
//! this encoder uses, consecutive decoding times are one frame period
//! apart in both the `low_delay` and the reordered Annex D item-7
//! schedules — Table D.1). The occupancy recurrence is item 8
//! (`b_{i+1} = b_i + ∫Rvol − d_{i+1}`, real-valued arithmetic), seeded
//! per item 4/8 with `b_0 = 64 × vbv_occupancy + (configuration
//! bits) − d_0`, and item 9 requires `0 <= b_i`, `b_i + d_i <= B`, and
//! `d_i < B` for every VOP.
//!
//! [`RateController`] simulates exactly that model on the encoder side
//! (a constant-delay channel, item D.2 closing note): the buffer fills
//! by `min(B − b, R × Δt)` per decode interval — the peak-rate channel
//! simply idles against a full buffer, so overflow cannot occur and
//! the encoder's only normative obligation is to keep every `d_i`
//! within the current occupancy ([`RateController::accepts`]; the
//! caller re-encodes at a coarser quantiser via
//! [`RateController::escalate`] until the VOP fits).
//!
//! The quantiser *adaptation* is an encoder choice the standard leaves
//! free: after each VOP the controller compares the spent bits against
//! the per-VOP budget (the channel refill plus a proportional
//! correction steering the occupancy towards two-thirds of `B` — the
//! same operating point as the Annex D default `vbv_occupancy`) and
//! scales the quantiser multiplicatively, bounded to ±2 per VOP.
//!
//! Provenance: Annex D (§D.1/§D.2) of ISO/IEC 14496-2:2004 (3rd
//! edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt`.
//! No third-party source was consulted.

/// Static parameters of the Annex D simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateControlConfig {
    /// Peak channel rate `Rvol` in bits per second (item 3; the VOL
    /// signals `ceil(bit_rate / 400)` in 400-bit units).
    pub bit_rate: u64,
    /// `vbv_buffer_size` in 16384-bit units (item 2; `B = 16384 ×
    /// vbv_buffer_size`).
    pub vbv_buffer_units: u32,
    /// Initial `vbv_occupancy` in 64-bit units (item 4).
    pub occupancy_64: u32,
    /// Seconds between consecutive VOP decoding times (the uniform
    /// input frame period).
    pub seconds_per_vop: f64,
    /// Starting quantiser scale (1..=31).
    pub initial_qp: u32,
}

/// The Annex D VBV simulation + reactive quantiser controller.
#[derive(Debug, Clone)]
pub struct RateController {
    cfg: RateControlConfig,
    /// Buffer occupancy in bits immediately **before** the next VOP's
    /// removal (real-valued per Annex D item 9).
    buf: f64,
    /// Running quantiser scale for the next VOP.
    qp: u32,
    /// Count of item-9 violations (a VOP that could not be shrunk
    /// under the occupancy even at quantiser 31). Zero on any feasible
    /// configuration; exposed for observability.
    pub underflows: u64,
}

impl RateController {
    /// Build the controller. `config_bits` is the size of the §6.2.1
    /// configuration run that precedes the first VOP (Annex D items
    /// 4/8: it sits in the buffer alongside the first VOP's bits and
    /// is part of `d_0`).
    pub fn new(cfg: RateControlConfig, config_bits: u64) -> Self {
        assert!(cfg.bit_rate > 0, "rate control needs a positive bit rate");
        assert!(cfg.vbv_buffer_units > 0, "vbv_buffer_size 0 is forbidden");
        assert!((1..=31).contains(&cfg.initial_qp));
        Self {
            buf: 64.0 * f64::from(cfg.occupancy_64) + config_bits as f64,
            qp: cfg.initial_qp,
            cfg,
            underflows: 0,
        }
    }

    /// The VBV buffer size `B` in bits (item 2).
    pub fn buffer_bits(&self) -> f64 {
        16384.0 * f64::from(self.cfg.vbv_buffer_units)
    }

    /// The quantiser scale to encode the next VOP with.
    pub fn qp(&self) -> u32 {
        self.qp
    }

    /// Item 9: would removing a `d_bits`-bit VOP keep the buffer
    /// non-negative (and is the VOP smaller than the buffer)?
    pub fn accepts(&self, d_bits: u64) -> bool {
        let d = d_bits as f64;
        d <= self.buf && d < self.buffer_bits()
    }

    /// Coarsen the quantiser for a re-encode after a rejected VOP.
    /// Returns `false` when the quantiser is already saturated at 31
    /// (the caller then commits the oversized VOP and the violation is
    /// counted).
    pub fn escalate(&mut self) -> bool {
        if self.qp >= 31 {
            return false;
        }
        self.qp = (self.qp + 4).min(31);
        true
    }

    /// Remove a committed VOP of `d_bits` bits (item 6), refill from
    /// the peak-rate channel over one decode interval (items 3/8), and
    /// adapt the quantiser for the next VOP.
    pub fn commit(&mut self, d_bits: u64) {
        let b_cap = self.buffer_bits();
        let d = d_bits as f64;
        if d > self.buf {
            self.underflows += 1;
        }
        let b = self.buf - d; // occupancy after removal (item 8)

        // Quantiser adaptation (encoder freedom): steer the occupancy
        // towards the two-thirds operating point with a per-VOP budget
        // of channel-refill + a 1/16 proportional correction, scaling
        // qp by the overspend ratio, bounded ±2 per VOP.
        let refill_per_vop = self.cfg.bit_rate as f64 * self.cfg.seconds_per_vop;
        let target_occupancy = b_cap * 2.0 / 3.0;
        let budget = (refill_per_vop + (b - target_occupancy) / 16.0).max(refill_per_vop * 0.25);
        let ratio = (d / budget).clamp(0.5, 2.0);
        let scaled = (f64::from(self.qp) * ratio).round();
        let bounded = scaled.clamp(f64::from(self.qp) - 2.0, f64::from(self.qp) + 2.0);
        self.qp = (bounded as i64).clamp(1, 31) as u32;

        // Channel refill until the next decoding time: the peak-rate
        // channel idles against a full buffer, so the fill saturates
        // at B (no overflow is possible under this model).
        self.buf = (b + refill_per_vop).min(b_cap).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RateControlConfig {
        RateControlConfig {
            bit_rate: 100_000,
            vbv_buffer_units: 13, // ≈ 213 kbit
            occupancy_64: 170 * 13,
            seconds_per_vop: 0.04,
            initial_qp: 8,
        }
    }

    #[test]
    fn accepts_tracks_occupancy() {
        let rc = RateController::new(cfg(), 800);
        // Initial occupancy 64*170*13 + 800 = 142,240 bits.
        assert!(rc.accepts(100_000));
        assert!(!rc.accepts(150_000));
        // A VOP as large as the whole buffer is barred (d_i < B).
        assert!(!rc.accepts(16384 * 13));
    }

    #[test]
    fn qp_rises_on_overspend_and_falls_on_underspend() {
        let mut rc = RateController::new(cfg(), 0);
        // Per-VOP refill is 4000 bits; spending 40k with a near-target
        // occupancy must raise qp (bounded +2).
        rc.commit(40_000);
        assert_eq!(rc.qp(), 10);
        // Spending almost nothing must lower it (bounded −2).
        let before = rc.qp();
        rc.commit(100);
        assert_eq!(rc.qp(), before - 2);
    }

    #[test]
    fn escalate_saturates_at_31() {
        let mut rc = RateController::new(cfg(), 0);
        let mut guard = 0;
        while rc.escalate() {
            guard += 1;
            assert!(guard < 10, "escalation must terminate");
        }
        assert_eq!(rc.qp(), 31);
    }

    #[test]
    fn refill_saturates_at_buffer_size() {
        let mut rc = RateController::new(cfg(), 0);
        // Tiny VOPs: the buffer climbs to B and stays there.
        for _ in 0..1000 {
            rc.commit(8);
        }
        assert!(rc.buf <= rc.buffer_bits());
        assert!(rc.buf > rc.buffer_bits() - 8_192.0);
        assert_eq!(rc.underflows, 0);
    }
}

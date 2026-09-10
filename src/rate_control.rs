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
//! free. Two modes exist:
//!
//! * the **per-VOP reactive** mode ([`RateController::commit`]): after
//!   each VOP the controller compares the spent bits against the
//!   per-VOP budget (the channel refill plus a proportional correction
//!   steering the occupancy towards two-thirds of `B` — the same
//!   operating point as the Annex D default `vbv_occupancy`) and
//!   scales the quantiser multiplicatively, bounded to ±2 per VOP;
//! * the **budget-driven** mode ([`BudgetPlanner`]): a bit budget is
//!   allotted per GOP (`gop_size × bits-per-frame`, carrying the
//!   previous GOP's surplus or deficit), split over the GOP's
//!   remaining I-/P-/B-VOPs in proportion to a per-class complexity
//!   model (`X = bits × mean quantiser`, the `bits ∝ X / qp` law, with
//!   B-VOPs held at a coarser quantiser by a fixed ratio), clamped
//!   against the VBV occupancy so item 9 holds with margin, and each
//!   VOP's target is then handed to the macroblock loop as a
//!   [`crate::mb_quant::MbBudget`] — the per-macroblock `dquant` /
//!   `dbquant` steps regulate the spend inside the VOP. An optional
//!   two-pass plan ([`FirstPassStats`]) replaces the running
//!   complexity model with the measured complexity of every frame of
//!   the sequence, so the split is exact rather than predictive.
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

    /// Buffer occupancy in bits immediately before the next VOP's
    /// removal (Annex D item 8's `b_i`).
    pub fn occupancy(&self) -> f64 {
        self.buf
    }

    /// Override the quantiser the next VOP is coded with (the
    /// budget-driven planner's choice).
    pub fn set_qp(&mut self, qp: u32) {
        assert!((1..=31).contains(&qp));
        self.qp = qp;
    }

    /// Remove a committed VOP of `d_bits` bits (item 6) and refill from
    /// the peak-rate channel over one decode interval (items 3/8)
    /// **without** touching the quantiser — the budget-driven mode,
    /// where the quantiser comes from the [`BudgetPlanner`].
    pub fn commit_buffer(&mut self, d_bits: u64) {
        let b_cap = self.buffer_bits();
        let d = d_bits as f64;
        if d > self.buf {
            self.underflows += 1;
        }
        let b = self.buf - d; // occupancy after removal (item 8)
        let refill_per_vop = self.cfg.bit_rate as f64 * self.cfg.seconds_per_vop;
        self.buf = (b + refill_per_vop).min(b_cap).max(0.0);
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

/// The coding class a VOP budget is planned for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VopClass {
    /// I-VOP.
    I,
    /// P-VOP or S(GMC)-VOP (same complexity class).
    P,
    /// B-VOP.
    B,
}

impl VopClass {
    fn index(self) -> usize {
        match self {
            VopClass::I => 0,
            VopClass::P => 1,
            VopClass::B => 2,
        }
    }

    /// One-letter tag (the first-pass statistics format).
    pub fn tag(self) -> char {
        match self {
            VopClass::I => 'I',
            VopClass::P => 'P',
            VopClass::B => 'B',
        }
    }

    /// Parse a one-letter tag.
    pub fn from_tag(c: char) -> Option<Self> {
        match c {
            'I' => Some(VopClass::I),
            'P' => Some(VopClass::P),
            'B' => Some(VopClass::B),
            _ => None,
        }
    }
}

/// The quantiser ratio B-VOPs are held at relative to the anchors
/// (the bits model works in "anchor-quantiser" units, so a B-VOP's
/// share of the budget is `X_B / K_B`).
const B_QP_RATIO: f64 = 1.4;

/// Per-frame record of a first (analysis) pass: the numbers the
/// two-pass planner needs to allot the sequence budget exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameStat {
    /// Coding class in bitstream order.
    pub class: VopClass,
    /// Size of the coded unit in bits (configuration run excluded).
    pub bits: u64,
    /// Mean quantiser of the coded macroblocks.
    pub mean_qp: f64,
}

impl FrameStat {
    /// The complexity `X = bits × qp` of the frame.
    pub fn complexity(&self) -> f64 {
        self.bits as f64 * self.mean_qp
    }
}

/// First-pass statistics of a whole sequence — one [`FrameStat`] per
/// coded VOP in bitstream order — with a line-oriented text form for
/// the registry `stats-file` round trip.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FirstPassStats {
    /// One record per coded VOP, bitstream order.
    pub frames: Vec<FrameStat>,
}

impl FirstPassStats {
    const HEADER: &'static str = "# oxideav-mpeg4video first-pass v1";

    /// Serialise: a header line, then `frame <class> <bits> <mean_qp>`
    /// per VOP.
    pub fn to_text(&self) -> String {
        let mut out = String::from(Self::HEADER);
        out.push('\n');
        for f in &self.frames {
            out.push_str(&format!(
                "frame {} {} {:.4}\n",
                f.class.tag(),
                f.bits,
                f.mean_qp
            ));
        }
        out
    }

    /// Parse the [`FirstPassStats::to_text`] form (blank lines and `#`
    /// comments ignored).
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        match lines.next() {
            Some(h) if h == Self::HEADER => {}
            other => return Err(format!("missing first-pass header line (got {other:?})")),
        }
        let mut frames = Vec::new();
        for (n, line) in lines.enumerate() {
            if line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let kw = it.next();
            if kw != Some("frame") {
                return Err(format!("record {n}: expected `frame`, got {kw:?}"));
            }
            let class = it
                .next()
                .and_then(|t| t.chars().next())
                .and_then(VopClass::from_tag)
                .ok_or_else(|| format!("record {n}: bad class"))?;
            let bits: u64 = it
                .next()
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| format!("record {n}: bad bit count"))?;
            let mean_qp: f64 = it
                .next()
                .and_then(|t| t.parse().ok())
                .filter(|q: &f64| (1.0..=31.0).contains(q))
                .ok_or_else(|| format!("record {n}: bad mean quantiser"))?;
            frames.push(FrameStat {
                class,
                bits,
                mean_qp,
            });
        }
        Ok(Self { frames })
    }
}

/// The plan of one VOP: the quantiser to seed it with and the bit
/// budget its macroblock layer must land on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VopPlan {
    /// `vop_quant` (1..=31).
    pub qp: u32,
    /// Target unit size in bits (headers included).
    pub target_bits: u32,
}

/// The budget-driven allocator: GOP budgets, per-class complexity
/// model, VBV clamp, optional two-pass plan (see the module
/// documentation).
#[derive(Debug, Clone)]
pub struct BudgetPlanner {
    bit_rate: f64,
    bits_per_frame: f64,
    gop_size: u32,
    bf: u32,
    /// Complexity per class (`X = bits × mean qp`), `None` until the
    /// first VOP of that class was measured.
    complexity: [Option<f64>; 3],
    /// Quantiser the last VOP of each class was coded with.
    last_qp: [u32; 3],
    /// Bits left in the current GOP budget (may go negative: the
    /// deficit carries into the next GOP).
    gop_remaining: f64,
    /// Surplus (positive) or debt (negative) carried between GOPs,
    /// repaid at one second's worth of GOPs per second.
    carry: f64,
    /// Classes whose first VOP has been calibrated (`calibrate`).
    calibrated: [bool; 3],
    /// The previous GOP's unspent budget and the number of its B-VOPs
    /// still to code (they are drained right after the new GOP's
    /// I-VOP in bitstream order, so they are charged to the GOP they
    /// belong to); folded into the carry once they are coded.
    prev_gop: Option<(f64, u32)>,
    /// Whether the VOP currently planned is charged to `prev_gop`.
    current_prev: bool,
    /// VOPs of each class still to code in the current GOP.
    left: [u32; 3],
    /// Two-pass plan: the first-pass records plus the running
    /// bookkeeping (next record index, planned bits still ahead).
    plan: Option<(FirstPassStats, usize, f64)>,
    /// Total sequence budget under the two-pass plan.
    sequence_budget: f64,
    /// Bits spent so far under the two-pass plan.
    spent: f64,
    /// The VOP currently planned (class + plan), for `escalate`.
    current: Option<(VopClass, VopPlan)>,
}

impl BudgetPlanner {
    /// Build the planner: `bit_rate` in bits per second, the uniform
    /// frame period, the display-order keyframe cadence and the B-VOP
    /// run length, and the quantiser the first I-VOP is coded with.
    pub fn new(
        bit_rate: u64,
        seconds_per_vop: f64,
        gop_size: u32,
        bf: u32,
        initial_qp: u32,
    ) -> Self {
        assert!(bit_rate > 0 && gop_size > 0 && (1..=31).contains(&initial_qp));
        Self {
            bit_rate: bit_rate as f64,
            bits_per_frame: bit_rate as f64 * seconds_per_vop,
            gop_size,
            bf,
            complexity: [None; 3],
            last_qp: [initial_qp; 3],
            gop_remaining: 0.0,
            carry: 0.0,
            calibrated: [false; 3],
            prev_gop: None,
            current_prev: false,
            left: [0; 3],
            plan: None,
            sequence_budget: 0.0,
            spent: 0.0,
            current: None,
        }
    }

    /// Attach a two-pass plan: the sequence budget is
    /// `frames × bits-per-frame` and each VOP's share follows its
    /// first-pass complexity. The plan is consumed in bitstream order;
    /// once its records run out (or a class mismatches), the planner
    /// falls back to the one-pass GOP model for the rest.
    pub fn with_first_pass(mut self, stats: FirstPassStats) -> Self {
        let planned: f64 = stats
            .frames
            .iter()
            .map(|f| f.complexity() / Self::class_ratio(f.class))
            .sum();
        self.sequence_budget = stats.frames.len() as f64 * self.bits_per_frame;
        self.plan = Some((stats, 0, planned));
        self
    }

    fn class_ratio(class: VopClass) -> f64 {
        if class == VopClass::B {
            B_QP_RATIO
        } else {
            1.0
        }
    }

    /// A closed GOP's leftover (or overspend) joins the carry, bounded
    /// to one second of channel bits; the open GOP then repays /
    /// spends the fraction of the carry a GOP's duration represents
    /// in a second (a whole-second GOP settles it at once, an
    /// intra-only stream one frame's share per frame).
    fn fold_leftover(&mut self, leftover: f64) {
        let cap = self.carry_cap();
        self.carry = (self.carry + leftover).clamp(-cap, cap);
        let repay = self.carry * (f64::from(self.gop_size) / self.frames_per_second()).min(1.0);
        self.carry -= repay;
        self.gop_remaining += repay;
    }

    /// The largest surplus / deficit carried from one GOP into the
    /// next: one second of channel bits (so a short GOP cannot forgive
    /// its overspend, and a long one is not asked to repay more than a
    /// second's worth at once).
    fn carry_cap(&self) -> f64 {
        self.bits_per_frame * self.frames_per_second()
    }

    fn frames_per_second(&self) -> f64 {
        (self.bit_rate / self.bits_per_frame).max(1.0)
    }

    /// The I/P/B counts of one GOP under the encoder's anchor rule
    /// (anchors every `bf + 1` display frames, the first one intra).
    fn gop_counts(&self) -> [u32; 3] {
        let anchors = self.gop_size.div_ceil(self.bf + 1);
        [1, anchors - 1, self.gop_size - anchors]
    }

    /// Plan the next VOP of `class` against the VBV state
    /// (`occupancy` bits available before the removal, `buffer_bits`
    /// the buffer size `B`). Returns the quantiser and the bit target.
    pub fn plan(&mut self, class: VopClass, occupancy: f64, buffer_bits: f64) -> VopPlan {
        self.current_prev = false;
        if class == VopClass::I {
            // A new GOP. The previous GOP closes — except for the
            // B-VOPs it still owes (drained right after this I-VOP):
            // its unspent budget stays theirs until they are coded,
            // then what is left over (or overspent) joins the carry.
            let gop_budget = f64::from(self.gop_size) * self.bits_per_frame;
            let pending_b = self.left[VopClass::B.index()];
            if self.prev_gop.is_some() || pending_b == 0 {
                // Nothing (or nothing more) owed: fold now.
                let leftover = self.gop_remaining + self.prev_gop.take().map_or(0.0, |(r, _)| r);
                self.gop_remaining = gop_budget;
                self.fold_leftover(leftover);
            } else {
                self.prev_gop = Some((self.gop_remaining, pending_b));
                self.gop_remaining = gop_budget;
            }
            self.left = self.gop_counts();
        }
        let idx = class.index();
        if class == VopClass::B {
            if let Some((remaining, pending)) = self.prev_gop {
                if pending > 0 {
                    // A B-VOP owed by the previous GOP: an equal split
                    // of what that GOP has left.
                    self.current_prev = true;
                    let target = (remaining / f64::from(pending))
                        .min(occupancy * 0.9)
                        .min(buffer_bits * 0.9)
                        .max(self.bits_per_frame * 0.05)
                        .max(64.0);
                    let qp = match self.two_pass_qp(class, target) {
                        Some(q) => q,
                        None => match self.complexity[idx] {
                            Some(x) => (x / target).round(),
                            None => f64::from(self.seed_qp(class)),
                        },
                    };
                    let last = f64::from(self.last_qp[idx]);
                    let qp = qp.clamp(last - 4.0, last + 4.0).clamp(1.0, 31.0) as u32;
                    let plan = VopPlan {
                        qp,
                        target_bits: target.round() as u32,
                    };
                    self.current = Some((class, plan));
                    return plan;
                }
            }
        }
        if self.left[idx] == 0 {
            // A truncated GOP at flush: extend the budget by one frame.
            self.left[idx] = 1;
            self.gop_remaining += self.bits_per_frame;
        }

        let mut target = match self.two_pass_target(class) {
            Some(t) => t,
            None => self.one_pass_target(class),
        };
        // VBV: item 9 needs d_i <= b_i; keep a margin, and d_i < B.
        target = target
            .min(occupancy * 0.9)
            .min(buffer_bits * 0.9)
            .max(self.bits_per_frame * 0.05)
            .max(64.0);

        let qp = match self.two_pass_qp(class, target) {
            Some(q) => q,
            None => match self.complexity[idx] {
                Some(x) => (x / target).round(),
                None => f64::from(self.seed_qp(class)),
            },
        };
        // Frame-to-frame smoothness: at most ±4 from the last VOP of
        // the same class — or from the class seed while the class is
        // unmeasured (the macroblock regulator absorbs the rest).
        let last = if self.complexity[idx].is_some() {
            f64::from(self.last_qp[idx])
        } else {
            f64::from(self.seed_qp(class))
        };
        let qp = qp.clamp(last - 4.0, last + 4.0).clamp(1.0, 31.0) as u32;
        let plan = VopPlan {
            qp,
            target_bits: target.round() as u32,
        };
        self.current = Some((class, plan));
        plan
    }

    /// One-pass: the GOP budget split by the complexity model over
    /// the VOPs still to code.
    fn one_pass_target(&self, class: VopClass) -> f64 {
        let x = |c: VopClass| -> f64 {
            self.complexity[c.index()].unwrap_or_else(|| self.assumed_complexity(c))
                / Self::class_ratio(c)
        };
        let weighted: f64 = [VopClass::I, VopClass::P, VopClass::B]
            .iter()
            .map(|&c| f64::from(self.left[c.index()]) * x(c))
            .sum();
        if weighted <= 0.0 {
            return self.bits_per_frame;
        }
        self.gop_remaining * x(class) / weighted
    }

    /// Two-pass: the next record's share of the sequence budget,
    /// rescaled so the remaining plan spends exactly the remaining
    /// budget.
    fn two_pass_target(&self, class: VopClass) -> Option<f64> {
        let (stats, next, planned_ahead) = self.plan.as_ref()?;
        let rec = stats.frames.get(*next)?;
        if rec.class != class || *planned_ahead <= 0.0 {
            return None;
        }
        let remaining_budget = self.sequence_budget - self.spent;
        let share = rec.complexity() / Self::class_ratio(class) / planned_ahead;
        Some((remaining_budget * share).max(0.0))
    }

    /// Two-pass: the quantiser that makes this frame's first-pass
    /// complexity land on `target`.
    fn two_pass_qp(&self, class: VopClass, target: f64) -> Option<f64> {
        let (stats, next, _) = self.plan.as_ref()?;
        let rec = stats.frames.get(*next)?;
        if rec.class != class || rec.bits == 0 {
            return None;
        }
        Some((rec.complexity() / target).round())
    }

    /// Complexity guess for a class not yet measured, from the classes
    /// that were: P ≈ 0.4 × I, B ≈ 0.25 × I (I ≈ 2.5 × P).
    fn assumed_complexity(&self, class: VopClass) -> f64 {
        let [xi, xp, xb] = self.complexity;
        match class {
            VopClass::I => xi
                .or(xp.map(|x| x * 2.5))
                .or(xb.map(|x| x * 4.0))
                .unwrap_or(self.bits_per_frame * 5.0 * f64::from(self.last_qp[0])),
            VopClass::P => xp
                .or(xi.map(|x| x * 0.4))
                .or(xb.map(|x| x * 1.6))
                .unwrap_or(self.bits_per_frame * 2.0 * f64::from(self.last_qp[1])),
            VopClass::B => xb
                .or(xp.map(|x| x * 0.625))
                .or(xi.map(|x| x * 0.25))
                .unwrap_or(self.bits_per_frame * 1.25 * f64::from(self.last_qp[2])),
        }
    }

    /// The quantiser seed for a class not yet measured: the last
    /// anchor quantiser (B-VOPs one step coarser).
    fn seed_qp(&self, class: VopClass) -> u32 {
        match class {
            VopClass::I => self.last_qp[0],
            VopClass::P => self.last_qp[1].max(self.last_qp[0]),
            VopClass::B => (self.last_qp[1].max(self.last_qp[0]) + 1).min(31),
        }
    }

    /// First-VOP calibration: the first VOP of a class is planned on
    /// an assumed complexity, so after it is coded once the planner
    /// checks the outcome against the target — when `bits` misses it
    /// by more than a quarter, the VOP is re-planned on its measured
    /// complexity (`bits × mean_qp`) and the caller re-encodes at the
    /// returned quantiser (once per class over the sequence; `None`
    /// = keep the VOP). No-op under a two-pass plan, whose
    /// complexities are measured already.
    pub fn calibrate(&mut self, class: VopClass, bits: u64, mean_qp: f64) -> Option<u32> {
        let idx = class.index();
        if self.calibrated[idx] || self.two_pass_target(class).is_some() {
            return None;
        }
        self.calibrated[idx] = true;
        let (_, plan) = self.current?;
        let target = f64::from(plan.target_bits);
        let ratio = bits as f64 / target;
        if (0.75..=1.25).contains(&ratio) {
            return None;
        }
        let measured = bits as f64 * mean_qp.clamp(1.0, 31.0);
        self.complexity[idx] = Some(measured);
        let qp = (measured / target).round().clamp(1.0, 31.0) as u32;
        if qp == plan.qp {
            return None;
        }
        self.current = Some((class, VopPlan { qp, ..plan }));
        Some(qp)
    }

    /// The plan of the VOP currently being coded (between `plan` and
    /// `commit`).
    pub fn current_plan(&self) -> Option<VopPlan> {
        self.current.map(|(_, p)| p)
    }

    /// The planned VOP did not fit the VBV even after regulation:
    /// coarsen its quantiser for a re-encode. Returns the new plan, or
    /// `None` at quantiser 31.
    pub fn escalate(&mut self) -> Option<VopPlan> {
        let (class, mut plan) = self.current?;
        if plan.qp >= 31 {
            return None;
        }
        plan.qp = (plan.qp + 4).min(31);
        self.current = Some((class, plan));
        Some(plan)
    }

    /// Book the committed VOP: `bits` spent at a mean macroblock
    /// quantiser of `mean_qp`. Updates the complexity model (an
    /// equal-weight blend with the previous estimate), the GOP budget
    /// and the two-pass bookkeeping.
    pub fn commit(&mut self, class: VopClass, bits: u64, mean_qp: f64) {
        let idx = class.index();
        let measured = bits as f64 * mean_qp.clamp(1.0, 31.0);
        self.complexity[idx] = Some(match self.complexity[idx] {
            Some(prev) => 0.5 * prev + 0.5 * measured,
            None => measured,
        });
        if let Some((_, plan)) = self.current.take() {
            self.last_qp[idx] = plan.qp;
        }
        if self.current_prev {
            // Charged to the previous GOP; fold it once settled.
            self.current_prev = false;
            if let Some((remaining, pending)) = self.prev_gop.as_mut() {
                *remaining -= bits as f64;
                *pending = pending.saturating_sub(1);
                if *pending == 0 {
                    let (leftover, _) = self.prev_gop.take().expect("just checked");
                    self.fold_leftover(leftover);
                }
            }
        } else {
            self.gop_remaining -= bits as f64;
            self.left[idx] = self.left[idx].saturating_sub(1);
        }
        self.spent += bits as f64;
        if let Some((stats, next, planned_ahead)) = self.plan.as_mut() {
            if let Some(rec) = stats.frames.get(*next) {
                if rec.class == class {
                    *planned_ahead -= rec.complexity() / Self::class_ratio(class);
                    *next += 1;
                } else {
                    // Structure mismatch: abandon the plan.
                    *next = stats.frames.len();
                }
            }
        }
    }

    /// The complexity model's current estimate for `class`, if
    /// measured (observability).
    pub fn complexity(&self, class: VopClass) -> Option<f64> {
        self.complexity[class.index()]
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

    #[test]
    fn planner_splits_a_gop_by_class() {
        // 25 fps, 250 kb/s → 10 000 bits per frame; GOP 12, bf 2.
        let mut p = BudgetPlanner::new(250_000, 0.04, 12, 2, 8);
        assert_eq!(p.gop_counts(), [1, 3, 8]);
        let b = 16384.0 * 31.0;
        let i = p.plan(VopClass::I, b * 0.66, b);
        assert_eq!(i.qp, 8);
        // An I-VOP owns the largest share of the GOP budget.
        assert!(i.target_bits > 10_000, "{i:?}");
        p.commit(VopClass::I, 40_000, 8.0);
        let pp = p.plan(VopClass::P, b * 0.66, b);
        assert!(pp.target_bits < i.target_bits, "{pp:?} vs {i:?}");
        p.commit(VopClass::P, 12_000, 8.0);
        // Measured P complexity 96 000; a target of 8 000 asks qp 12.
        let pb = p.plan(VopClass::B, b * 0.66, b);
        assert!((1..=31).contains(&pb.qp));
        assert!(pb.target_bits < pp.target_bits);
    }

    #[test]
    fn planner_clamps_to_the_vbv_occupancy() {
        let mut p = BudgetPlanner::new(250_000, 0.04, 1, 0, 8);
        let plan = p.plan(VopClass::I, 5_000.0, 16384.0 * 31.0);
        assert!(plan.target_bits <= 4_500);
        assert!(p.escalate().is_some_and(|e| e.qp == 12));
    }

    #[test]
    fn two_pass_follows_the_measured_complexities() {
        let stats = FirstPassStats {
            frames: vec![
                FrameStat {
                    class: VopClass::I,
                    bits: 30_000,
                    mean_qp: 8.0,
                },
                FrameStat {
                    class: VopClass::P,
                    bits: 10_000,
                    mean_qp: 8.0,
                },
                FrameStat {
                    class: VopClass::P,
                    bits: 10_000,
                    mean_qp: 8.0,
                },
            ],
        };
        let text = stats.to_text();
        assert_eq!(FirstPassStats::parse(&text).unwrap(), stats);
        assert!(FirstPassStats::parse("frame I 1 2").is_err());
        // Budget 3 × 10 000 = 30 000 for a first pass of 50 000 bits
        // at qp 8 → every frame lands at 3/5 of its size, i.e. qp ~13.
        let mut p = BudgetPlanner::new(250_000, 0.04, 12, 0, 8).with_first_pass(stats);
        let b = 16384.0 * 31.0;
        let i = p.plan(VopClass::I, b, b);
        assert_eq!(i.target_bits, 18_000);
        assert_eq!(i.qp, 12); // 13 clamped to initial + 4
        p.commit(VopClass::I, 18_000, 12.0);
        let pp = p.plan(VopClass::P, b, b);
        assert_eq!(pp.target_bits, 6_000);
        assert_eq!(pp.qp, 13);
        // Overspend rescales the remainder.
        p.commit(VopClass::P, 9_000, 13.0);
        let last = p.plan(VopClass::P, b, b);
        assert_eq!(last.target_bits, 3_000);
    }
}

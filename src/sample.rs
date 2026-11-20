//! Picture sample storage types.
//!
//! §6.3.3 `bits_per_pixel` ranges over 4..=12 for the N-bit object
//! type (8 everywhere else). Pictures of up to 8 bits per sample are
//! stored one byte per sample; deeper pictures are stored in 16-bit
//! words. The motion-compensation primitives are written once over
//! [`Sample`] and instantiated for both stores, so the 8-bit paths
//! compile to exactly the code they were before the wider store
//! existed.

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
}

/// One stored picture sample: `u8` for `bits_per_pixel <= 8`, `u16`
/// for 9..=12 bits. Sealed — the two stores are the whole family.
pub trait Sample:
    Copy + Default + Eq + Ord + core::fmt::Debug + Send + Sync + sealed::Sealed + 'static
{
    /// The zero sample.
    const ZERO: Self;
    /// Largest `bits_per_pixel` this store holds.
    const MAX_BITS: u32;

    /// The sample as a signed working value.
    fn working(self) -> i32;

    /// Store a working value already inside the sample range (values
    /// outside it are truncated to the store's width).
    fn narrow(value: i32) -> Self;

    /// §7.6.2.1 two-sample average `(a + b + 1 - rc) / 2`, `rc` being
    /// `rounding_control` (0 or 1).
    fn average2(a: Self, b: Self, rc: u32) -> Self;

    /// §7.6.2.1 four-sample average `(a + b + c + d + 2 - rc) / 4`.
    fn average4(a: Self, b: Self, c: Self, d: Self, rc: u32) -> Self;

    /// Sum of absolute differences of two equally long runs.
    fn run_sad(a: &[Self], b: &[Self]) -> u32;

    /// Store a working value when it is representable (`None` when it
    /// falls outside the store's range).
    fn try_narrow(value: i32) -> Option<Self>;

    /// Sum of absolute differences of two sixteen-sample rows.
    fn row_sad16(a: &[Self; 16], b: &[Self; 16]) -> u32;

    /// Select this store's plane out of a narrow / wide pair.
    #[doc(hidden)]
    fn pick<'a>(narrow: Option<&'a [u8]>, wide: Option<&'a [u16]>) -> Option<&'a [Self]>;
}

impl Sample for u8 {
    const ZERO: Self = 0;
    const MAX_BITS: u32 = 8;

    #[inline(always)]
    fn try_narrow(value: i32) -> Option<Self> {
        u8::try_from(value).ok()
    }

    #[inline(always)]
    fn row_sad16(a: &[Self; 16], b: &[Self; 16]) -> u32 {
        let mut sad = 0u32;
        for i in 0..16 {
            sad += u32::from(a[i].abs_diff(b[i]));
        }
        sad
    }

    #[inline]
    fn pick<'a>(narrow: Option<&'a [u8]>, _wide: Option<&'a [u16]>) -> Option<&'a [u8]> {
        narrow
    }

    #[inline(always)]
    fn working(self) -> i32 {
        i32::from(self)
    }

    #[inline(always)]
    fn narrow(value: i32) -> Self {
        value as u8
    }

    #[inline(always)]
    fn average2(a: Self, b: Self, rc: u32) -> Self {
        ((u16::from(a) + u16::from(b) + 1 - rc as u16) >> 1) as u8
    }

    #[inline(always)]
    fn average4(a: Self, b: Self, c: Self, d: Self, rc: u32) -> Self {
        ((u16::from(a) + u16::from(b) + u16::from(c) + u16::from(d) + 2 - rc as u16) >> 2) as u8
    }

    #[inline(always)]
    fn run_sad(a: &[Self], b: &[Self]) -> u32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| u32::from(x.abs_diff(y)))
            .sum()
    }
}

impl Sample for u16 {
    const ZERO: Self = 0;
    const MAX_BITS: u32 = 16;

    #[inline(always)]
    fn try_narrow(value: i32) -> Option<Self> {
        u16::try_from(value).ok()
    }

    #[inline(always)]
    fn row_sad16(a: &[Self; 16], b: &[Self; 16]) -> u32 {
        let mut sad = 0u32;
        for i in 0..16 {
            sad += u32::from(a[i].abs_diff(b[i]));
        }
        sad
    }

    #[inline]
    fn pick<'a>(_narrow: Option<&'a [u8]>, wide: Option<&'a [u16]>) -> Option<&'a [u16]> {
        wide
    }

    #[inline(always)]
    fn working(self) -> i32 {
        i32::from(self)
    }

    #[inline(always)]
    fn narrow(value: i32) -> Self {
        value as u16
    }

    #[inline(always)]
    fn average2(a: Self, b: Self, rc: u32) -> Self {
        ((u32::from(a) + u32::from(b) + 1 - rc) >> 1) as u16
    }

    #[inline(always)]
    fn average4(a: Self, b: Self, c: Self, d: Self, rc: u32) -> Self {
        ((u32::from(a) + u32::from(b) + u32::from(c) + u32::from(d) + 2 - rc) >> 2) as u16
    }

    #[inline(always)]
    fn run_sad(a: &[Self], b: &[Self]) -> u32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| u32::from(x.abs_diff(y)))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_agree_between_the_two_stores() {
        for rc in 0..2u32 {
            for a in (0..=255u16).step_by(5) {
                for b in (0..=255u16).step_by(7) {
                    let n = u8::average2(a as u8, b as u8, rc);
                    let w = u16::average2(a, b, rc);
                    assert_eq!(u16::from(n), w);
                    let n4 = u8::average4(a as u8, b as u8, 255, 0, rc);
                    let w4 = u16::average4(a, b, 255, 0, rc);
                    assert_eq!(u16::from(n4), w4);
                }
            }
        }
    }

    #[test]
    fn wide_averages_hold_twelve_bit_samples() {
        assert_eq!(u16::average2(4095, 4095, 0), 4095);
        assert_eq!(u16::average2(4095, 4094, 1), 4094);
        assert_eq!(u16::average4(4095, 4095, 4095, 4095, 0), 4095);
        assert_eq!(u16::run_sad(&[0, 4095], &[4095, 0]), 8190);
    }
}

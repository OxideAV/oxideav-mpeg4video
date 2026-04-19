//! MPEG-4 Part 2 specific bit-reader extensions.
//!
//! The generic bit reader lives in [`oxideav_core::bits`]. MPEG-4 adds
//! one codec-specific method — `read_marker()` — that reads a single
//! bit and refuses to continue if it isn't `1`, matching the
//! "marker_bit ≡ 1" contract scattered throughout §6.2 of the MPEG-4
//! bitstream. Callers `use oxideav_mpeg4video::bits_ext::BitReaderExt;`
//! to get `br.read_marker()` back in method syntax.

use oxideav_core::{bits::BitReader, Error, Result};

pub trait BitReaderExt {
    /// Consume one bit and error if it isn't `1`. The spec requires
    /// marker bits everywhere a zero could theoretically start code
    /// emulation, so a `0` at a marker position means we've lost
    /// synchronisation.
    fn read_marker(&mut self) -> Result<()>;
}

impl BitReaderExt for BitReader<'_> {
    fn read_marker(&mut self) -> Result<()> {
        let m = self.read_u1()?;
        if m != 1 {
            return Err(Error::invalid("mpeg4video: marker bit != 1"));
        }
        Ok(())
    }
}

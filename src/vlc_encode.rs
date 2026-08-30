//! Encoder-side VLC emission — the exact inverses of the crate's
//! decode tables.
//!
//! Every codeword written here comes from the *same* transcribed
//! tables the decoder matches against (Tables B.6/B.7 `mcbpc`, B.8
//! `cbpy`, B.12 `mv_data`, B.13/B.14 `dct_dc_size`, B.15
//! `dct_dc_differential`, B.16/B.17 Tcoef with the §7.4.1.3 escape
//! modes, Table 6-32 `dquant`), so an encode → decode round trip is
//! bijective by construction and the unit tests below assert exactly
//! that against the decoder's own entry points.
//!
//! ## Escape-mode selection (§7.4.1.3)
//!
//! [`put_ac_event`] emits the cheapest legal representation in the
//! fixed precedence order the escape design implies: a direct Table
//! B.16/B.17 codeword when the EVENT is tabulated; else Type 1
//! (`ESC 0` + VLC after `LEVEL −= LMAX`); else Type 2 (`ESC 10` + VLC
//! after `RUN −= RMAX + 1`); else the fixed-length Type 3
//! (`ESC 11` + LAST/RUN/marker/LEVEL/marker). Type 3 can represent
//! every EVENT the §7.4.3.4-saturated coefficient domain produces
//! (LEVEL ∈ `[-2047, 2047] \ {0}`, RUN ≤ 63), so emission is total.
//!
//! Provenance: the tables and escape formats are those of ISO/IEC
//! 14496-2:2004 (3rd edition), read from
//! `docs/video/mpeg4-visual/ISO_IEC_14496-2-2004-3rd-edition.txt` and
//! transcribed in [`crate::texture`] / [`crate::macroblock`] /
//! [`crate::motion`]. No third-party source was consulted.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::bitwriter::BitWriter;
use crate::macroblock::{CBPY_4, MCBPC_I, MCBPC_P};
use crate::motion::mvd_table;
use crate::texture::{
    dc_size_table, lmax, rmax, tcoef_table, AcEvent, DcComponent, TcoefTable, ESCAPE,
};

/// Emit a §7.4.1.1 differential intra DC: `dct_dc_size` VLC (Table
/// B.13 / B.14), the `size`-bit Table B.15 additional code, and the
/// NOTE-2 `marker_bit` when `size > 8`.
///
/// # Panics
///
/// Panics when `differential` is outside `[-4095, 4095]` (the Table
/// B.15 size-12 domain; the §7.4.3.4-saturated quantised domain never
/// exceeds it).
pub fn put_intra_dc(bw: &mut BitWriter, component: DcComponent, differential: i32) {
    let mag = differential.unsigned_abs();
    assert!(
        mag < (1 << 12),
        "intra DC differential {differential} out of Table B.15 range"
    );
    // `size` is the bit-length of |differential| (0 for zero).
    let size = (32 - mag.leading_zeros()) as u8;
    let table = dc_size_table(component);
    let &(code, len, _) = table
        .iter()
        .find(|&&(_, _, s)| s == size)
        .expect("sizes 0..=12 are all tabulated");
    bw.write_bits(u32::from(code), usize::from(len));
    if size == 0 {
        return;
    }
    // Table B.15: positive differentials are coded as themselves (the
    // top bit of the size-wide field is set by construction); negative
    // ones as `differential - 1 + 2^size`.
    let additional = if differential > 0 {
        differential as u32
    } else {
        (differential - 1 + (1 << size)) as u32
    };
    bw.write_bits(additional, usize::from(size));
    if size > 8 {
        bw.write_marker();
    }
}

/// Reverse-lookup key for one Tcoef EVENT: `(last, run, |level|)`.
type TcoefKey = (bool, u32, i32);
/// Reverse-lookup value: `(code_bits, code_len)`.
type TcoefCode = (u32, u8);

/// Reverse lookup for one Tcoef table: `(last, run, |level|)` →
/// `(code_bits, code_len)`.
fn tcoef_reverse(kind: TcoefTable) -> &'static HashMap<TcoefKey, TcoefCode> {
    static INTRA: OnceLock<HashMap<TcoefKey, TcoefCode>> = OnceLock::new();
    static INTER: OnceLock<HashMap<TcoefKey, TcoefCode>> = OnceLock::new();
    let cell = match kind {
        TcoefTable::Intra => &INTRA,
        TcoefTable::Inter => &INTER,
    };
    cell.get_or_init(|| {
        tcoef_table(kind)
            .iter()
            .map(|&(code, len, last, run, level)| {
                ((last != 0, u32::from(run), i32::from(level)), (code, len))
            })
            .collect()
    })
}

/// Emit the Table B.16/B.17 VLC for a tabulated `(last, run, |level|)`
/// plus the sign bit. Returns `false` (writing nothing) when the
/// EVENT is not tabulated.
fn try_put_tcoef_vlc(
    bw: &mut BitWriter,
    kind: TcoefTable,
    last: bool,
    run: u32,
    level: i32,
) -> bool {
    debug_assert!(level != 0);
    match tcoef_reverse(kind).get(&(last, run, level.abs())) {
        Some(&(code, len)) => {
            bw.write_bits(code, usize::from(len));
            bw.write_bit(level < 0);
            true
        }
        None => false,
    }
}

/// Emit one §7.4.1.2 DCT-coefficient EVENT (`short_video_header == 0`,
/// `reversible_vlc == 0` path): direct Table B.16/B.17 codeword, or
/// the cheapest applicable §7.4.1.3 escape (Type 1 → Type 2 →
/// Type 3).
///
/// # Panics
///
/// Panics when the EVENT is unrepresentable — `level == 0`,
/// `|level| > 2047` (or the reserved `-2048`), or `run > 63` — which
/// the encoder-side quantisers and 64-coefficient scan never produce.
pub fn put_ac_event(bw: &mut BitWriter, kind: TcoefTable, event: AcEvent) {
    let AcEvent { last, run, level } = event;
    assert!(level != 0, "AC EVENT level must be non-zero");
    assert!(
        (-2047..=2047).contains(&level),
        "AC EVENT level {level} outside escape domain"
    );
    assert!(
        run <= 63,
        "AC EVENT run {run} exceeds the 6-bit escape field"
    );

    // Direct table hit.
    if try_put_tcoef_vlc(bw, kind, last, run, level) {
        return;
    }

    let (esc_code, esc_len) = ESCAPE;

    // Type 1: LEVEL reduced by LMAX(last, run) must itself be a
    // tabulated EVENT for the same (last, run).
    if let Some(lm) = lmax(kind, last, run) {
        let reduced = level.abs() - lm;
        if reduced >= 1 {
            let signed = if level < 0 { -reduced } else { reduced };
            let mut probe = BitWriter::new();
            if try_put_tcoef_vlc(&mut probe, kind, last, run, signed) {
                bw.write_bits(esc_code, usize::from(esc_len));
                bw.write_bit(false); // "0" — Type 1
                let ok = try_put_tcoef_vlc(bw, kind, last, run, signed);
                debug_assert!(ok);
                return;
            }
        }
    }

    // Type 2: RUN reduced by RMAX(last, |level|) + 1 must be a
    // tabulated EVENT for the same (last, level).
    if let Some(rm) = rmax(kind, last, level.abs()) {
        if let Some(reduced) = run.checked_sub(rm + 1) {
            let mut probe = BitWriter::new();
            if try_put_tcoef_vlc(&mut probe, kind, last, reduced, level) {
                bw.write_bits(esc_code, usize::from(esc_len));
                bw.write_bit(true); // "1"
                bw.write_bit(false); // "0" — Type 2
                let ok = try_put_tcoef_vlc(bw, kind, last, reduced, level);
                debug_assert!(ok);
                return;
            }
        }
    }

    // Type 3: ESC + "11" + LAST(1) RUN(6) marker LEVEL(12) marker.
    bw.write_bits(esc_code, usize::from(esc_len));
    bw.write_bit(true);
    bw.write_bit(true);
    bw.write_bit(last);
    bw.write_bits(run, 6);
    bw.write_marker();
    bw.write_bits((level & 0xFFF) as u32, 12);
    bw.write_marker();
}

/// Emit every EVENT of one block's §6.2.7 `while (!last)` loop.
pub fn put_ac_events(bw: &mut BitWriter, kind: TcoefTable, events: &[AcEvent]) {
    for ev in events {
        put_ac_event(bw, kind, *ev);
    }
}

/// Emit one EVENT with the reversible VLC (Table B.23 column for
/// `kind` + sign bit), falling back to the §7.4.1.3 Type-5 escape —
/// opener `00001`, `LAST(1) RUN(6) marker LEVEL(11) marker`, closer
/// `0000`, sign — when the triple is not tabulated. Exact inverse of
/// [`crate::texture::decode_ac_event_rvlc`].
///
/// # Panics
///
/// Panics when `run > 63` or `|level|` is outside `1..=2047`.
pub fn put_ac_event_rvlc(bw: &mut BitWriter, kind: TcoefTable, event: AcEvent) {
    let magnitude = event.level.unsigned_abs();
    assert!(
        (1..=2047).contains(&magnitude) && event.run <= 63,
        "RVLC EVENT {event:?} outside the Type-5 escape domain"
    );
    let last = u8::from(event.last);
    let hit = crate::texture::rvlc_tcoef_table().iter().find(
        |&&(_, _, i_last, i_run, i_level, n_last, n_run, n_level)| match kind {
            TcoefTable::Intra => {
                i_last == last && u32::from(i_run) == event.run && u32::from(i_level) == magnitude
            }
            TcoefTable::Inter => {
                n_last == last && u32::from(n_run) == event.run && u32::from(n_level) == magnitude
            }
        },
    );
    match hit {
        Some(&(code, len, ..)) => {
            bw.write_bits(code, usize::from(len));
            bw.write_bit(event.level < 0);
        }
        None => {
            let (open, open_len) = crate::texture::RVLC_ESCAPE;
            bw.write_bits(open, usize::from(open_len));
            bw.write_bit(event.last);
            bw.write_bits(event.run, 6);
            bw.write_marker();
            bw.write_bits(magnitude, 11);
            bw.write_marker();
            bw.write_bits(0, 4);
            bw.write_bit(event.level < 0);
        }
    }
}

/// Emit every EVENT of one block with the reversible VLC.
pub fn put_ac_events_rvlc(bw: &mut BitWriter, kind: TcoefTable, events: &[AcEvent]) {
    for ev in events {
        put_ac_event_rvlc(bw, kind, *ev);
    }
}

/// Emit an I-VOP `mcbpc` (Table B.6) for `derived_mb_type` 3 (intra)
/// or 4 (intra+q) with the 2-bit `cbpc`.
///
/// # Panics
///
/// Panics on a `(mb_type, cbpc)` pair outside Table B.6.
pub fn put_mcbpc_i(bw: &mut BitWriter, derived_mb_type: u8, cbpc: u8) {
    let &(code, len, ..) = MCBPC_I
        .iter()
        .find(|&&(_, _, t, c)| t == derived_mb_type && c == cbpc)
        .unwrap_or_else(|| panic!("mcbpc I ({derived_mb_type}, {cbpc:#04b}) not in Table B.6"));
    bw.write_bits(u32::from(code), usize::from(len));
}

/// Emit a P-/S(GMC)-VOP `mcbpc` (Table B.7) for `derived_mb_type`
/// 0..=4 with the 2-bit `cbpc`.
///
/// # Panics
///
/// Panics on a `(mb_type, cbpc)` pair outside Table B.7.
pub fn put_mcbpc_p(bw: &mut BitWriter, derived_mb_type: u8, cbpc: u8) {
    let &(code, len, ..) = MCBPC_P
        .iter()
        .find(|&&(_, _, t, c)| t == derived_mb_type && c == cbpc)
        .unwrap_or_else(|| panic!("mcbpc P ({derived_mb_type}, {cbpc:#04b}) not in Table B.7"));
    bw.write_bits(u32::from(code), usize::from(len));
}

/// Emit a `cbpy` (Table B.8, 4 non-transparent blocks). `cbpy` is the
/// 4-bit luminance coded-block pattern in the §6.3.7 "1 = coded"
/// convention; `intra` selects which of the table's two columns that
/// pattern is read from.
///
/// # Panics
///
/// Panics on `cbpy > 0b1111`.
pub fn put_cbpy(bw: &mut BitWriter, cbpy: u8, intra: bool) {
    assert!(cbpy <= 0b1111, "cbpy {cbpy:#x} exceeds 4 bits");
    let &(code, len, ..) = CBPY_4
        .iter()
        .find(|&&(_, _, ci, cp)| if intra { ci == cbpy } else { cp == cbpy })
        .expect("all 16 patterns are tabulated");
    bw.write_bits(u32::from(code), usize::from(len));
}

/// Emit a 2-bit `dquant` (Table 6-32) for a delta in `{-2, -1, +1, +2}`.
///
/// # Panics
///
/// Panics on any other delta.
pub fn put_dquant(bw: &mut BitWriter, delta: i8) {
    let code = match delta {
        -1 => 0b00,
        -2 => 0b01,
        1 => 0b10,
        2 => 0b11,
        other => panic!("dquant delta {other} not in Table 6-32"),
    };
    bw.write_bits(code, 2);
}

/// Emit a B-VOP `dbquant` (Table 6-33): `0` → "0", `-2` → "10",
/// `+2` → "11".
///
/// # Panics
///
/// Panics on any other delta.
pub fn put_dbquant(bw: &mut BitWriter, delta: i8) {
    match delta {
        0 => bw.write_bit(false),
        -2 => bw.write_bits(0b10, 2),
        2 => bw.write_bits(0b11, 2),
        other => panic!("dbquant delta {other} not in Table 6-33"),
    }
}

/// Wrap a raw differential (MV − predictor) into the §7.6.3
/// `[low, high]` range for `fcode` by adding / subtracting `range`
/// once — the exact inverse the decoder's modulo wrap undoes.
///
/// # Panics
///
/// Panics when `fcode` is outside `1..=7` or the wrapped value still
/// falls outside the range (i.e. the caller's MV/predictor pair is
/// not representable under this `fcode`).
pub fn wrap_mvd_component(diff: i32, fcode: u8) -> i32 {
    assert!((1..=7).contains(&fcode), "fcode {fcode} out of range");
    let f = 1i32 << (fcode - 1);
    let (low, high, range) = (-32 * f, 32 * f - 1, 64 * f);
    let mut d = diff;
    if d < low {
        d += range;
    } else if d > high {
        d -= range;
    }
    assert!(
        (low..=high).contains(&d),
        "MVD {diff} unrepresentable under fcode {fcode}"
    );
    d
}

/// Emit one differential MV component (already wrapped into the
/// §7.6.3 `[low, high]` range, in half- or quarter-sample units): the
/// Table B.12 `mv_data` VLC plus the `r_size`-bit residual gated on
/// `fcode != 1 && mv_data != 0` (§6.2.6.2).
///
/// # Panics
///
/// Panics when the component is outside the `fcode` range.
pub fn put_mv_component(bw: &mut BitWriter, mvd: i32, fcode: u8) {
    assert!((1..=7).contains(&fcode), "fcode {fcode} out of range");
    let f = 1i32 << (fcode - 1);
    assert!(
        (-32 * f..=32 * f - 1).contains(&mvd),
        "MVD component {mvd} outside fcode-{fcode} range"
    );
    let (mv_data, residual) = if mvd == 0 {
        (0, 0)
    } else if f == 1 {
        (mvd, 0)
    } else {
        // §7.6.3 inverse: magnitude = (|mv_data| - 1) * f + residual + 1.
        let magnitude = mvd.abs();
        let data_abs = (magnitude - 1) / f + 1;
        let residual = (magnitude - 1) % f;
        (if mvd < 0 { -data_abs } else { data_abs }, residual)
    };
    let &(code, len, _) = mvd_table()
        .iter()
        .find(|&&(_, _, d)| d == mv_data)
        .expect("mv_data -32..=32 is fully tabulated");
    bw.write_bits(u32::from(code), usize::from(len));
    if f != 1 && mv_data != 0 {
        bw.write_bits(residual as u32, usize::from(fcode - 1));
    }
}

/// Emit a full §6.2.6.2 `motion_vector()` body: horizontal then
/// vertical differential, each wrapped into range first.
pub fn put_motion_vector(bw: &mut BitWriter, dx: i32, dy: i32, fcode: u8) {
    put_mv_component(bw, wrap_mvd_component(dx, fcode), fcode);
    put_mv_component(bw, wrap_mvd_component(dy, fcode), fcode);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::macroblock::{decode_cbpy4, decode_mcbpc};
    use crate::motion::{decode_motion_vector_delta, MvMode};
    use crate::texture::{decode_ac_event, decode_intra_dc};

    #[test]
    fn intra_dc_round_trips_full_domain() {
        for &component in &[DcComponent::Luminance, DcComponent::Chrominance] {
            for diff in -2048..=2048 {
                let mut bw = BitWriter::new();
                put_intra_dc(&mut bw, component, diff);
                bw.next_start_code();
                let bytes = bw.into_bytes();
                let mut br = BitReader::new(&bytes);
                let got = decode_intra_dc(&mut br, component).unwrap();
                assert_eq!(got.differential, diff, "component {component:?}");
            }
        }
    }

    #[test]
    fn rvlc_events_round_trip_table_and_escape() {
        for kind in [TcoefTable::Intra, TcoefTable::Inter] {
            for last in [false, true] {
                for run in [0u32, 1, 2, 5, 12, 20, 40, 63] {
                    for level in [1i32, -1, 2, -3, 7, -20, 100, -2047, 2047] {
                        let ev = AcEvent { last, run, level };
                        let mut bw = BitWriter::new();
                        put_ac_event_rvlc(&mut bw, kind, ev);
                        bw.next_start_code();
                        let bytes = bw.into_bytes();
                        let mut br = BitReader::new(&bytes);
                        let got = crate::texture::decode_ac_event_rvlc(&mut br, kind).unwrap();
                        assert_eq!(got, ev, "{kind:?}");
                    }
                }
            }
        }
        // Every tabulated row survives its own column.
        for &(_, _, i_last, i_run, i_level, n_last, n_run, n_level) in
            crate::texture::rvlc_tcoef_table()
        {
            for (kind, last, run, level) in [
                (TcoefTable::Intra, i_last, i_run, i_level),
                (TcoefTable::Inter, n_last, n_run, n_level),
            ] {
                let ev = AcEvent {
                    last: last != 0,
                    run: u32::from(run),
                    level: -i32::from(level),
                };
                let mut bw = BitWriter::new();
                put_ac_event_rvlc(&mut bw, kind, ev);
                bw.next_start_code();
                let bytes = bw.into_bytes();
                let mut br = BitReader::new(&bytes);
                assert_eq!(
                    crate::texture::decode_ac_event_rvlc(&mut br, kind).unwrap(),
                    ev
                );
            }
        }
    }

    fn roundtrip_event(kind: TcoefTable, ev: AcEvent) {
        let mut bw = BitWriter::new();
        put_ac_event(&mut bw, kind, ev);
        bw.next_start_code();
        let bytes = bw.into_bytes();
        let mut br = BitReader::new(&bytes);
        let got = decode_ac_event(&mut br, kind).unwrap();
        assert_eq!(
            (got.last, got.run, got.level),
            (ev.last, ev.run, ev.level),
            "{kind:?} {ev:?}"
        );
    }

    #[test]
    fn every_tabulated_event_round_trips_both_signs() {
        for &kind in &[TcoefTable::Intra, TcoefTable::Inter] {
            for &(_, _, last, run, level) in tcoef_table(kind) {
                for sign in [1i32, -1] {
                    roundtrip_event(
                        kind,
                        AcEvent {
                            last: last != 0,
                            run: u32::from(run),
                            level: sign * i32::from(level),
                        },
                    );
                }
            }
        }
    }

    #[test]
    fn escape_events_round_trip() {
        // A sweep that forces Type 1 (level just above LMAX), Type 2
        // (run just above RMAX), and Type 3 (far outside both).
        for &kind in &[TcoefTable::Intra, TcoefTable::Inter] {
            for last in [false, true] {
                for run in 0..=63u32 {
                    for &level_abs in &[1i32, 2, 3, 5, 9, 13, 28, 41, 200, 2047] {
                        for sign in [1i32, -1] {
                            roundtrip_event(
                                kind,
                                AcEvent {
                                    last,
                                    run,
                                    level: sign * level_abs,
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mcbpc_round_trips() {
        for &(_, _, t, c) in MCBPC_I.iter().filter(|&&(_, _, t, _)| t == 3 || t == 4) {
            let mut bw = BitWriter::new();
            put_mcbpc_i(&mut bw, t, c);
            bw.next_start_code();
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            let (_, mb_type, cbpc) = decode_mcbpc(&mut br, MCBPC_I).unwrap();
            assert_eq!((mb_type, cbpc), (t, c));
        }
        for &(_, _, t, c) in MCBPC_P.iter().filter(|&&(_, _, t, _)| t <= 4) {
            let mut bw = BitWriter::new();
            put_mcbpc_p(&mut bw, t, c);
            bw.next_start_code();
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            let (_, mb_type, cbpc) = decode_mcbpc(&mut br, MCBPC_P).unwrap();
            assert_eq!((mb_type, cbpc), (t, c));
        }
    }

    #[test]
    fn cbpy_round_trips_all_patterns() {
        for cbpy in 0..=0b1111u8 {
            for intra in [true, false] {
                let mut bw = BitWriter::new();
                put_cbpy(&mut bw, cbpy, intra);
                bw.next_start_code();
                let bytes = bw.into_bytes();
                let mut br = BitReader::new(&bytes);
                let (_, ci, cp) = decode_cbpy4(&mut br).unwrap();
                assert_eq!(if intra { ci } else { cp }, cbpy);
            }
        }
    }

    #[test]
    fn dquant_round_trips() {
        use crate::macroblock::dquant_value;
        for delta in [-2i8, -1, 1, 2] {
            let mut bw = BitWriter::new();
            put_dquant(&mut bw, delta);
            bw.next_start_code();
            let bytes = bw.into_bytes();
            let mut br = BitReader::new(&bytes);
            let code = br.read_bits(2).unwrap() as u8;
            assert_eq!(dquant_value(code), delta);
        }
    }

    #[test]
    fn motion_vector_round_trips_across_fcodes() {
        for fcode in 1..=7u8 {
            let f = 1i32 << (fcode - 1);
            let (low, high) = (-32 * f, 32 * f - 1);
            // Sample the corners, zero, and a spread of interior values.
            let mut probes = vec![low, low + 1, -1, 0, 1, high - 1, high];
            let step = (high - low) / 23;
            probes.extend((0..23).map(|k| low + k * step));
            for &dx in &probes {
                for &dy in &[low, 0, high, dx] {
                    let mut bw = BitWriter::new();
                    put_mv_component(&mut bw, dx, fcode);
                    put_mv_component(&mut bw, dy, fcode);
                    bw.next_start_code();
                    let bytes = bw.into_bytes();
                    let mut br = BitReader::new(&bytes);
                    let delta =
                        decode_motion_vector_delta(&mut br, MvMode::Forward, fcode).unwrap();
                    assert_eq!((delta.dx, delta.dy), (dx, dy), "fcode {fcode}");
                }
            }
        }
    }

    #[test]
    fn wrap_brings_out_of_range_diffs_back() {
        // fcode 1: range [-32, 31], range width 64.
        assert_eq!(wrap_mvd_component(40, 1), 40 - 64);
        assert_eq!(wrap_mvd_component(-40, 1), -40 + 64);
        assert_eq!(wrap_mvd_component(31, 1), 31);
        // And the decoder-side wrap restores the original MV: the
        // §7.6.3 wrap is on MV = P + MVD, exercised in the encoder
        // walks' round trips.
    }
}

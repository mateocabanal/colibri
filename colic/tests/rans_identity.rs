//! Differential rANS identity: colic's encoder <-> colibri-format's decoder
//! must be byte-exact (plan RW-014 gate). colibri-format's RansTable carries
//! the same freq/start arrays; decode_record() must invert encode_bytes()
//! exactly.

use colibri_format::codecs::{RANS_M, RansTable};

fn round_trip(hist: [u64; 16], raw: &[u8]) {
    let table = colic::codec::rans256::Table::from_histogram(hist).unwrap();
    let encoded = colic::codec::rans256::encode_bytes(raw, &table).unwrap();
    let rt = RansTable {
        freq: table.freq,
        start: table.start,
    };
    let decoded = rt.decode_record(&encoded, raw.len()).unwrap();
    assert_eq!(decoded, raw, "rANS round trip must be byte-identical");
}

#[test]
fn rans_round_trip_flat_histogram() {
    // Every nibble equally likely (flat), like a dense matrix would be.
    let raw: Vec<u8> = (0..512).map(|i| (i * 37 % 256) as u8).collect();
    let hist = [raw.len() as u64 / 16; 16];
    round_trip(hist, &raw);
}

#[test]
fn rans_round_trip_sparse_histogram() {
    // Sparse: mostly zeros (INT4-ish values cluster near one nibble).
    let raw: Vec<u8> = (0..256)
        .map(|i| if i % 7 == 0 { 0xab } else { 0x00 })
        .collect();
    let mut hist = [0_u64; 16];
    for b in &raw {
        hist[(b & 0x0f) as usize] += 1;
        hist[(b >> 4) as usize] += 1;
    }
    round_trip(hist, &raw);
}

#[test]
fn rans_round_trip_largest_remainder_stability() {
    // A histogram with awkward proportions exercises the stable
    // largest-remainder table rule and the amplification bound.
    let raw: Vec<u8> = (0..1024).map(|i| ((i * 13) % 256) as u8).collect();
    let mut hist = [0_u64; 16];
    for b in &raw {
        hist[(b & 0x0f) as usize] += 1;
        hist[(b >> 4) as usize] += 1;
    }
    // Force a skewed distribution on top.
    hist[0] = hist[0].max(RANS_M as u64 / 2);
    round_trip(hist, &raw);
}

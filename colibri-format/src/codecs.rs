//! COLI record codecs shared by `colic` (encode) and the runtime (decode).
//! Decode-only by design: the writer stays in `colic` so compiler and runtime
//! cannot drift (plan RW-014).
//!
//! Formats: rANS256-g0-nibble (CODEC 0x0001), Apple8 MXFP4 tile8x32 (layout
//! 0x0103, which is rANS-compressed tiles), and grouped INT4 (MATH 0x0022:
//! biased signed nibbles + LE f32 scale per 32 columns).

use crate::verify::{invalid, usage, Result};

// ---------------------------------------------------------------------------
// rANS256 (bit-exact port of colic/src/codec/rans256.rs decode surface)
// ---------------------------------------------------------------------------

pub const RANS_CODEC_ID: u16 = 0x0001;
pub const RANS_TABLE_ID: u32 = 1;
pub const RANS_N_STREAMS: usize = 256;
pub const RANS_SCALE_BITS: u32 = 14;
pub const RANS_M: u32 = 1 << RANS_SCALE_BITS;
pub const RANS_L: u32 = 1 << 23;
pub const RANS_TABLE_BLOB_BYTES: usize = 160;
pub const RANS_AUTO_MIN_SAVINGS_BPS: u32 = 500;
pub const RANS_AUTO_MIN_SAVINGS_BYTES: usize = 256;
const RANS_TABLE_MAGIC: &[u8; 8] = b"COLIRN01";

/// 16-symbol rANS table (symbols are 4-bit nibble values).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RansTable {
    pub freq: [u32; 16],
    pub start: [u32; 16],
}

impl RansTable {
    fn validate(&self) -> Result<()> {
        let total: u64 = self.freq.iter().map(|f| u64::from(*f)).sum();
        if total != RANS_M as u64 {
            return invalid("rANS frequencies do not sum to M");
        }
        let mut cursor = 0_u64;
        for i in 0..16 {
            if u64::from(self.start[i]) != cursor {
                return invalid("rANS start offsets are not contiguous");
            }
            cursor += u64::from(self.freq[i]);
            if cursor > RANS_M as u64 {
                return invalid("rANS cumulative frequencies overflow M");
            }
        }
        if cursor != RANS_M as u64 {
            return invalid("rANS cumulative frequencies do not cover M");
        }
        Ok(())
    }

    fn slot_to_symbol(&self) -> Result<Vec<u16>> {
        self.validate()?;
        let mut slots = vec![0_u16; RANS_M as usize];
        for symbol in 0..16 {
            let begin = self.start[symbol] as usize;
            let end = begin + self.freq[symbol] as usize;
            for slot in &mut slots[begin..end] {
                *slot = symbol as u16;
            }
        }
        Ok(slots)
    }

    /// Parses a 160-byte table blob (the exact checks colic's decode_blob
    /// enforces: magic, version, scale bits, stream count, auto-savings
    /// params, zeroed reserved bytes).
    fn decode_blob(blob: &[u8]) -> Result<Self> {
        if blob.len() != RANS_TABLE_BLOB_BYTES
            || blob.get(..8) != Some(RANS_TABLE_MAGIC.as_slice())
            || u16_at(blob, 8)? != 1
            || u16_at(blob, 10)? != RANS_SCALE_BITS as u16
            || u32_at(blob, 12)? != RANS_N_STREAMS as u32
            || u32_at(blob, 16)? != RANS_AUTO_MIN_SAVINGS_BPS
            || u32_at(blob, 20)? != RANS_AUTO_MIN_SAVINGS_BYTES as u32
            || blob[24..32].iter().any(|b| *b != 0)
        {
            return invalid("invalid rANS256-g0-nibble codec table blob");
        }
        let mut freq = [0_u32; 16];
        let mut start = [0_u32; 16];
        for symbol in 0..16 {
            freq[symbol] = u32_at(blob, 32 + symbol * 4)?;
            start[symbol] = u32_at(blob, 96 + symbol * 4)?;
        }
        let table = RansTable { freq, start };
        table.validate()?;
        Ok(table)
    }

    /// Resolves a codec table reference from the manifest's codec-table
    /// region (bit-exact with colic's `table_from_manifest`).
    pub fn from_manifest(manifest: &[u8], table_id: u32, codec: u16) -> Result<Self> {
        if table_id == 0 || codec != RANS_CODEC_ID {
            return invalid("invalid rANS codec/table reference");
        }
        let count = u32_at(manifest, 160)? as usize;
        let region_offset =
            usize::try_from(u64_at(manifest, 168)?).map_err(|_| usage("codec table offset exceeds usize"))?;
        let region_bytes =
            usize::try_from(u64_at(manifest, 176)?).map_err(|_| usage("codec table size exceeds usize"))?;
        let region_end = region_offset
            .checked_add(region_bytes)
            .ok_or_else(|| usage("codec table region overflows usize"))?;
        let region = manifest
            .get(region_offset..region_end)
            .ok_or_else(|| usage("codec table region is outside manifest"))?;
        for index in 0..count {
            let desc = index
                .checked_mul(64)
                .ok_or_else(|| usage("codec table descriptor overflows"))?;
            if desc + 64 > region.len() {
                return invalid("truncated codec table descriptor");
            }
            if u32_at(region, desc)? != table_id {
                continue;
            }
            if u16_at(region, desc + 4)? != codec
                || u16_at(region, desc + 6)? != 0
                || i32_at(region, desc + 8)? != -1
                || u32_at(region, desc + 12)? != 0
                || u32_at(region, desc + 36)? != 0
                || region[desc + 40..desc + 64].iter().any(|b| *b != 0)
            {
                return invalid("invalid rANS codec table descriptor");
            }
            let data_offset =
                usize::try_from(u64_at(region, desc + 16)?).map_err(|_| usage("codec table data offset exceeds usize"))?;
            let data_bytes =
                usize::try_from(u64_at(region, desc + 24)?).map_err(|_| usage("codec table data size exceeds usize"))?;
            let blob_end = data_offset
                .checked_add(data_bytes)
                .ok_or_else(|| usage("codec table data span overflows usize"))?;
            let blob = region
                .get(data_offset..blob_end)
                .ok_or_else(|| usage("codec table data is outside region"))?;
            if crate::crc32c(blob) != u32_at(region, desc + 32)? {
                return invalid("codec table CRC32C mismatch");
            }
            return Self::decode_blob(blob);
        }
        invalid(format!("missing rANS codec table {table_id}"))
    }

    /// Decodes a full rANS record (256 interleaved streams, header + offset
    /// table, big-endian per-stream states) into `expected_bytes` of packed
    /// bytes. Bit-exact with colic's `decode_bytes`.
    pub fn decode_record(&self, record: &[u8], expected_bytes: usize) -> Result<Vec<u8>> {
        let parsed = ParsedRansRecord::parse(record)?;
        if parsed.packed_bytes != expected_bytes
            || parsed.n_symbols != expected_bytes * 2
        {
            return invalid("rANS decoded length does not match matrix descriptor");
        }
        let slots = self.slot_to_symbol()?;
        let mut output = vec![0_u8; expected_bytes];
        for stream in 0..RANS_N_STREAMS {
            let begin = parsed.offsets[stream] as usize;
            let end = parsed.offsets[stream + 1] as usize;
            let symbols = stream_symbol_count(parsed.n_symbols, stream);
            let decoded = decode_stream_checked(&parsed.payload[begin..end], symbols, self, &slots)?;
            for (index, symbol) in decoded.into_iter().enumerate() {
                let logical = stream + index * RANS_N_STREAMS;
                let byte = &mut output[logical / 2];
                if logical % 2 == 0 {
                    *byte = (*byte & 0xf0) | symbol;
                } else {
                    *byte = (*byte & 0x0f) | (symbol << 4);
                }
            }
        }
        Ok(output)
    }
}

struct ParsedRansRecord<'a> {
    n_symbols: usize,
    packed_bytes: usize,
    offsets: Vec<u32>,
    payload: &'a [u8],
}

fn rans_header_bytes() -> usize {
    round16(16 + (RANS_N_STREAMS + 1) * 4)
}

impl<'a> ParsedRansRecord<'a> {
    fn parse(record: &'a [u8]) -> Result<Self> {
        let header = rans_header_bytes();
        if record.len() < header {
            return invalid("truncated rANS record");
        }
        let n_symbols = usize::try_from(u64_at(record, 0)?)
            .map_err(|_| usage("rANS symbol count exceeds usize"))?;
        let packed_bytes = usize::try_from(u64_at(record, 8)?)
            .map_err(|_| usage("rANS packed-byte count exceeds usize"))?;
        if n_symbols == 0 || packed_bytes != n_symbols / 2 + n_symbols % 2 {
            return invalid("invalid rANS symbol/byte counts");
        }
        let mut offsets = Vec::with_capacity(RANS_N_STREAMS + 1);
        for stream in 0..=RANS_N_STREAMS {
            offsets.push(u32_at(record, 16 + stream * 4)?);
        }
        if offsets[0] != 0 {
            return invalid("rANS first stream offset is nonzero");
        }
        for pair in offsets.windows(2) {
            if pair[1] < pair[0] || pair[1] - pair[0] < 4 {
                return invalid("rANS stream offsets are malformed");
            }
        }
        let payload_bytes = offsets[RANS_N_STREAMS] as usize;
        let expected = header
            .checked_add(round16(payload_bytes))
            .ok_or_else(|| usage("rANS record length overflows"))?;
        if expected != record.len() || header + payload_bytes > record.len() {
            return invalid("rANS record length mismatch");
        }
        if record[16 + (RANS_N_STREAMS + 1) * 4..header].iter().any(|b| *b != 0)
            || record[header + payload_bytes..].iter().any(|b| *b != 0)
        {
            return invalid("rANS record padding is nonzero");
        }
        let max_symbols = (payload_bytes as u128) * 8 * (1_u128 << 15);
        if n_symbols as u128 > max_symbols {
            return invalid("rANS record exceeds amplification bound");
        }
        Ok(Self {
            n_symbols,
            packed_bytes,
            offsets,
            payload: &record[header..header + payload_bytes],
        })
    }
}

fn decode_stream_checked(
    stream: &[u8],
    symbols: usize,
    table: &RansTable,
    slots: &[u16],
) -> Result<Vec<u8>> {
    if stream.len() < 4 {
        return invalid("rANS stream is shorter than its state");
    }
    let mut cursor = 4_usize;
    let mut state = u32::from_be_bytes(stream[..4].try_into().unwrap());
    if state < RANS_L {
        return invalid("rANS initial state is out of range");
    }
    let mask = RANS_M - 1;
    let mut output = Vec::with_capacity(symbols);
    for _ in 0..symbols {
        let slot = state & mask;
        let symbol = slots[slot as usize] as usize;
        output.push(symbol as u8);
        state = table.freq[symbol] * (state >> RANS_SCALE_BITS) + slot - table.start[symbol];
        while state < RANS_L && cursor < stream.len() {
            state = (state << 8) | u32::from(stream[cursor]);
            cursor += 1;
        }
    }
    if cursor != stream.len() {
        return invalid("rANS stream has trailing bytes");
    }
    if state != RANS_L {
        return invalid("rANS stream final state is invalid");
    }
    Ok(output)
}

fn stream_symbol_count(total: usize, stream: usize) -> usize {
    if stream >= total {
        0
    } else {
        (total - 1 - stream) / RANS_N_STREAMS + 1
    }
}

fn round16(value: usize) -> usize {
    (value + 15) & !15
}

// ---------------------------------------------------------------------------
// Apple8 MXFP4 tile8x32 (layout 0x0103). Tiles are stored rANS-compressed
// (colic apple8.rs encodes tile bytes); decode = rANS decode of the record.
// ---------------------------------------------------------------------------

pub const APPLE8_TILE_ROWS: u64 = 8;
pub const APPLE8_TILE_COLUMNS: u64 = 32;
pub const APPLE8_TILE_BYTES: u64 = 136; // 128 weight + 8 scale
pub const APPLE8_WEIGHT_BYTES: u64 = 128;

/// Tile geometry for a matrix: decoded (execution) byte count for
/// `rows x columns` of Apple8 (checked multiplication throughout).
pub fn apple8_tile_bytes(rows: u64, columns: u64) -> Result<u64> {
    if rows == 0 || columns == 0 {
        return invalid("Apple8 matrix has a zero dimension");
    }
    rows.div_ceil(APPLE8_TILE_ROWS)
        .checked_mul(columns.div_ceil(APPLE8_TILE_COLUMNS))
        .and_then(|tiles| tiles.checked_mul(APPLE8_TILE_BYTES))
        .ok_or_else(|| usage("Apple8 matrix byte count overflows"))
}

/// Decodes an Apple8 record: rANS-compressed tile bytes -> execution bytes.
pub fn apple8_decode(record: &[u8], table: &RansTable, rows: u64, columns: u64) -> Result<Vec<u8>> {
    let expected = apple8_tile_bytes(rows, columns)?;
    table.decode_record(record, expected as usize)
}

/// E2M1 positive magnitudes (bit 3 of the nibble is the sign bit).
pub const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decodes an Apple8 tile8x32 record (rANS-decoded execution bytes) into an
/// f32 row-major matrix. Each tile is 136 bytes: 128 weight bytes (8 rows x
/// 16 bytes = 32 E2M1 nibbles each) + 8 E8M0 scale bytes (one per row).
/// value = sign * E2M1_MAGNITUDES[nibble & 7] * e8m0(scale), e8m0(code) =
/// f32::from_bits(code << 23) (the runtime kernels' fast path; code 0 is
/// never emitted by colic).
pub fn apple8_mxfp4_decode(tiles: &[u8], rows: u64, columns: u64) -> Result<Vec<f32>> {
    let expected = apple8_tile_bytes(rows, columns)?;
    if tiles.len() != expected as usize {
        return invalid(format!(
            "Apple8 tile payload {} != expected {expected}",
            tiles.len()
        ));
    }
    let row_tiles = rows.div_ceil(APPLE8_TILE_ROWS);
    let col_tiles = columns.div_ceil(APPLE8_TILE_COLUMNS);
    let mut out = vec![0.0_f32; (rows * columns) as usize];
    for rt in 0..row_tiles {
        for ct in 0..col_tiles {
            let tile = rt * col_tiles + ct;
            let base = (tile * APPLE8_TILE_BYTES) as usize;
            let weights = &tiles[base..base + APPLE8_WEIGHT_BYTES as usize];
            let scales = &tiles[base + APPLE8_WEIGHT_BYTES as usize..base + APPLE8_TILE_BYTES as usize];
            for r in 0..APPLE8_TILE_ROWS as usize {
                let scale = f32::from_bits(u32::from(scales[r]) << 23);
                for c in 0..APPLE8_TILE_COLUMNS as usize {
                    let byte = weights[r * 16 + c / 2];
                    let nibble = if c % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    let mag = E2M1_MAGNITUDES[(nibble & 0x07) as usize];
                    let sign = if nibble & 0x08 != 0 { -1.0 } else { 1.0 };
                    let out_row = (rt * APPLE8_TILE_ROWS) as usize + r;
                    let out_col = (ct * APPLE8_TILE_COLUMNS) as usize + c;
                    out[out_row * columns as usize + out_col] = sign * mag * scale;
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Grouped INT4 (MATH 0x0022) — matches colic's emission (int4.rs +
// int4_record.rs): row-major, even col = low nibble, biased signed
// (nibble - 8), one LE f32 scale per 32 columns.
// ---------------------------------------------------------------------------

pub const INT4_GROUP_SIZE: usize = 32;
pub const INT4_VALUES_PER_BYTE: usize = 2;
pub const INT4_MATH_FORMAT: u16 = 0x0022;
pub const INT4_SCALE_FORMAT: u16 = 0x0001;

/// Decodes an INT4-G32 matrix (packed weights + LE f32 scales, both
/// row-major) into f32, applying each row's per-group scale.
pub fn int4_grouped_decode(
    weights: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let packed_row_bytes = cols.div_ceil(INT4_VALUES_PER_BYTE);
    let scales_per_row = cols.div_ceil(INT4_GROUP_SIZE);
    let want_weights = rows
        .checked_mul(packed_row_bytes)
        .ok_or_else(|| usage("INT4 weight length overflows"))?;
    let want_scales = rows
        .checked_mul(scales_per_row)
        .ok_or_else(|| usage("INT4 scale length overflows"))?;
    if weights.len() != want_weights {
        return invalid(format!(
            "INT4 weight length {} != expected {want_weights}",
            weights.len()
        ));
    }
    if scales.len() != want_scales * 4 {
        return invalid(format!(
            "INT4 scale length {} != expected {}",
            scales.len(),
            want_scales * 4
        ));
    }
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let row_w = &weights[r * packed_row_bytes..(r + 1) * packed_row_bytes];
        let row_s = &scales[r * scales_per_row * 4..(r + 1) * scales_per_row * 4];
        for c in 0..cols {
            let byte = row_w[c / INT4_VALUES_PER_BYTE];
            let nibble = if c % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            let q = (nibble as i32) - 8;
            let group = c / INT4_GROUP_SIZE;
            let scale = f32::from_le_bytes(row_s[group * 4..group * 4 + 4].try_into().unwrap());
            out[r * cols + c] = q as f32 * scale;
        }
    }
    Ok(out)
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    bytes
        .get(offset..offset + 2)
        .and_then(|v| v.try_into().ok())
        .map(u16::from_le_bytes)
        .ok_or_else(|| usage("truncated COLI structure"))
}

fn i32_at(bytes: &[u8], offset: usize) -> Result<i32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|v| v.try_into().ok())
        .map(i32::from_le_bytes)
        .ok_or_else(|| usage("truncated COLI structure"))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|v| v.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| usage("truncated COLI structure"))
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    bytes
        .get(offset..offset + 8)
        .and_then(|v| v.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| usage("truncated COLI structure"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int4_grouped_roundtrip_matches_layout() {
        // 1 row, 64 cols = 2 groups. weights: 32 bytes (2 nibbles each);
        // scales: 2 f32. q=-8..7, scale 0.5 -> expect -4.0..3.5.
        let mut weights = vec![0_u8; 32];
        for c in 0..64 {
            let q = ((c % 16) as i32) - 8;
            let nib = (q + 8) as u8;
            if c % 2 == 0 {
                weights[c / 2] = nib;
            } else {
                weights[c / 2] |= nib << 4;
            }
        }
        let mut scales = Vec::new();
        for _ in 0..2 {
            scales.extend_from_slice(&0.5_f32.to_le_bytes());
        }
        let out = int4_grouped_decode(&weights, &scales, 1, 64).unwrap();
        for c in 0..64 {
            let q = ((c % 16) as i32) - 8;
            assert_eq!(out[c], q as f32 * 0.5);
        }
    }

    #[test]
    fn int4_grouped_rejects_bad_lengths() {
        assert!(int4_grouped_decode(&[0; 1], &[0; 4], 1, 64).is_err());
        assert!(int4_grouped_decode(&[0; 32], &[0; 4], 1, 64).is_err());
    }

    #[test]
    #[test]
    fn apple8_mxfp4_decode_roundtrip() {
        // 1 tile (8x32): nibble codes 0..7 with sign on odd, scale 1.0
        // (code 127) -> mag values, sign flips on odd columns.
        let mut tiles = vec![0_u8; 136];
        for r in 0..8 {
            for c in 0..32 {
                let nib: u8 = (c % 8) as u8 | if c % 2 == 1 { 0x8 } else { 0 };
                let byte = r * 16 + c / 2;
                if c % 2 == 0 {
                    tiles[byte] = nib;
                } else {
                    tiles[byte] |= nib << 4;
                }
            }
            tiles[128 + r] = 127; // scale 1.0
        }
        let out = apple8_mxfp4_decode(&tiles, 8, 32).unwrap();
        for c in 0..32 {
            let mag = E2M1_MAGNITUDES[c % 8];
            let sign = if c % 2 == 1 { -1.0 } else { 1.0 };
            assert_eq!(out[c], sign * mag);
        }
        assert!(apple8_mxfp4_decode(&tiles, 9, 32).is_err()); // wrong geometry
    }

    #[test]
    fn apple8_tile_bytes_matches_registry() {
        assert_eq!(apple8_tile_bytes(8, 32).unwrap(), 136);
        assert_eq!(apple8_tile_bytes(1, 1).unwrap(), 136);
        assert_eq!(apple8_tile_bytes(9, 33).unwrap(), 544);
        assert!(apple8_tile_bytes(0, 32).is_err());
    }

    #[test]
    fn rans_record_rejects_truncation() {
        let table = RansTable { freq: [0; 16], start: [0; 16] };
        assert!(table.decode_record(&[0; 10], 1).is_err());
        // invalid table (freqs don't sum to M) fails at slot construction
        let mut freq = [0_u32; 16];
        freq[0] = RANS_M;
        let table = RansTable { freq, start: [0; 16] };
        assert!(table.decode_record(&[0; 1100], 1).is_err());
    }
}

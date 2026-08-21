use crate::{
    error::{ColicError, Result},
    storage,
};

pub const CODEC_ID: u16 = 0x0001;
pub const TABLE_ID: u32 = 1;
pub const N_STREAMS: usize = 256;
pub const SCALE_BITS: u32 = 14;
pub const M: u32 = 1 << SCALE_BITS;
pub const RANS_L: u32 = 1 << 23;
pub const READABLE_SLACK: usize = 64;
pub const TABLE_BLOB_BYTES: usize = 160;
pub const AUTO_MIN_SAVINGS_BPS: u32 = 500;
pub const AUTO_MIN_SAVINGS_BYTES: usize = 256;
const TABLE_MAGIC: &[u8; 8] = b"COLIRN01";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub freq: [u32; 16],
    pub start: [u32; 16],
}

impl Table {
    /// Build the generation-0 table with the exact stable largest-remainder
    /// rule used by c/tools/rans_format.py. Present symbols get at least one
    /// slot, positive remainder ties resolve by symbol index, and a negative
    /// deficit is corrected by repeatedly walking one fixed frequency-sorted
    /// order (the reference deliberately does not re-sort after decrements).
    pub fn from_histogram(hist: [u64; 16]) -> Result<Self> {
        let total = hist
            .iter()
            .try_fold(0_u64, |sum, value| sum.checked_add(*value))
            .ok_or_else(|| ColicError::Usage("rANS histogram total overflows u64".into()))?;
        if total == 0 {
            return Err(ColicError::Usage(
                "rANS table requires at least one nibble".into(),
            ));
        }

        let present = hist.iter().filter(|value| **value != 0).count();
        if present > M as usize {
            return Err(ColicError::Usage(
                "rANS table scale cannot represent every present symbol".into(),
            ));
        }

        let mut freq = [0_u32; 16];
        let mut remainder = [-1.0_f64; 16];
        let mut sum = 0_i64;
        let total_f64 = total as f64;
        for symbol in 0..16 {
            if hist[symbol] == 0 {
                continue;
            }
            let raw = hist[symbol] as f64 / total_f64 * f64::from(M);
            let floor = raw.floor();
            let assigned = (floor as u32).max(1);
            freq[symbol] = assigned;
            remainder[symbol] = raw - floor;
            sum += i64::from(assigned);
        }

        let target = i64::from(M);
        if sum < target {
            let mut order = (0..16).collect::<Vec<_>>();
            order.sort_by(|&left, &right| {
                remainder[right]
                    .total_cmp(&remainder[left])
                    .then_with(|| left.cmp(&right))
            });
            let mut deficit = target - sum;
            for symbol in order {
                if deficit == 0 {
                    break;
                }
                if hist[symbol] != 0 {
                    freq[symbol] = freq[symbol]
                        .checked_add(1)
                        .ok_or_else(|| ColicError::Usage("rANS frequency overflows u32".into()))?;
                    deficit -= 1;
                }
            }
            if deficit != 0 {
                return Err(ColicError::Usage(
                    "rANS largest-remainder normalization left a positive deficit".into(),
                ));
            }
        } else if sum > target {
            let mut order = (0..16).collect::<Vec<_>>();
            order.sort_by(|&left, &right| {
                freq[right].cmp(&freq[left]).then_with(|| left.cmp(&right))
            });
            let mut deficit = target - sum;
            let mut cursor = 0usize;
            while deficit < 0 {
                let symbol = order[cursor % order.len()];
                if freq[symbol] > 1 {
                    freq[symbol] -= 1;
                    deficit += 1;
                }
                cursor = cursor.checked_add(1).ok_or_else(|| {
                    ColicError::Usage("rANS normalization cursor overflows".into())
                })?;
            }
        }

        let mut start = [0_u32; 16];
        let mut cursor = 0_u32;
        for symbol in 0..16 {
            start[symbol] = cursor;
            cursor = cursor
                .checked_add(freq[symbol])
                .ok_or_else(|| ColicError::Usage("rANS start table overflows u32".into()))?;
        }
        if cursor != M {
            return Err(ColicError::Usage(format!(
                "rANS frequency sum is {cursor}, expected {M}"
            )));
        }
        Ok(Self { freq, start })
    }

    pub fn validate(&self) -> Result<()> {
        let mut cursor = 0_u32;
        for symbol in 0..16 {
            if self.start[symbol] != cursor {
                return Err(ColicError::Usage(
                    "rANS table starts are not the prefix sum of frequencies".into(),
                ));
            }
            cursor = cursor
                .checked_add(self.freq[symbol])
                .ok_or_else(|| ColicError::Usage("rANS table sum overflows u32".into()))?;
        }
        if cursor != M {
            return Err(ColicError::Usage(
                "rANS table frequencies do not sum to the scale".into(),
            ));
        }
        Ok(())
    }

    pub fn slot_to_symbol(&self) -> Result<Vec<u16>> {
        self.validate()?;
        let mut slots = vec![0_u16; M as usize];
        for symbol in 0..16 {
            let begin = self.start[symbol] as usize;
            let end = begin + self.freq[symbol] as usize;
            for slot in &mut slots[begin..end] {
                *slot = symbol as u16;
            }
        }
        Ok(slots)
    }

    pub fn encode_blob(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut blob = vec![0_u8; TABLE_BLOB_BYTES];
        blob[..8].copy_from_slice(TABLE_MAGIC);
        put_u16(&mut blob, 8, 1);
        put_u16(&mut blob, 10, SCALE_BITS as u16);
        put_u32(&mut blob, 12, N_STREAMS as u32);
        put_u32(&mut blob, 16, AUTO_MIN_SAVINGS_BPS);
        put_u32(&mut blob, 20, AUTO_MIN_SAVINGS_BYTES as u32);
        for symbol in 0..16 {
            put_u32(&mut blob, 32 + symbol * 4, self.freq[symbol]);
            put_u32(&mut blob, 96 + symbol * 4, self.start[symbol]);
        }
        Ok(blob)
    }

    pub fn decode_blob(blob: &[u8]) -> Result<Self> {
        if blob.len() != TABLE_BLOB_BYTES
            || blob.get(..8) != Some(TABLE_MAGIC.as_slice())
            || u16_at(blob, 8)? != 1
            || u16_at(blob, 10)? != SCALE_BITS as u16
            || u32_at(blob, 12)? != N_STREAMS as u32
            || u32_at(blob, 16)? != AUTO_MIN_SAVINGS_BPS
            || u32_at(blob, 20)? != AUTO_MIN_SAVINGS_BYTES as u32
            || blob[24..32].iter().any(|byte| *byte != 0)
        {
            return Err(ColicError::Usage(
                "invalid rANS256-g0-nibble codec table blob".into(),
            ));
        }
        let mut freq = [0_u32; 16];
        let mut start = [0_u32; 16];
        for symbol in 0..16 {
            freq[symbol] = u32_at(blob, 32 + symbol * 4)?;
            start[symbol] = u32_at(blob, 96 + symbol * 4)?;
        }
        let table = Self { freq, start };
        table.validate()?;
        Ok(table)
    }
}

pub fn histogram_bytes<'a>(chunks: impl IntoIterator<Item = &'a [u8]>) -> Result<[u64; 16]> {
    let mut hist = [0_u64; 16];
    for bytes in chunks {
        for byte in bytes {
            let lo = (*byte & 0x0f) as usize;
            let hi = (*byte >> 4) as usize;
            hist[lo] = hist[lo]
                .checked_add(1)
                .ok_or_else(|| ColicError::Usage("rANS histogram overflows u64".into()))?;
            hist[hi] = hist[hi]
                .checked_add(1)
                .ok_or_else(|| ColicError::Usage("rANS histogram overflows u64".into()))?;
        }
    }
    Ok(hist)
}

pub fn encode_bytes(input: &[u8], table: &Table) -> Result<Vec<u8>> {
    table.validate()?;
    if input.is_empty() {
        return Err(ColicError::Usage(
            "rANS cannot encode an empty matrix".into(),
        ));
    }
    let n_symbols = input
        .len()
        .checked_mul(2)
        .ok_or_else(|| ColicError::Usage("rANS symbol count overflows usize".into()))?;
    let mut streams = vec![Vec::<u8>::new(); N_STREAMS];
    for (stream, slot) in streams.iter_mut().enumerate() {
        let mut symbols = Vec::with_capacity(n_symbols.div_ceil(N_STREAMS));
        let mut logical = stream;
        while logical < n_symbols {
            let byte = input[logical / 2];
            symbols.push(if logical.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            });
            logical += N_STREAMS;
        }
        *slot = encode_stream(&symbols, table)?;
    }

    let header_bytes = round16(16 + (N_STREAMS + 1) * 4);
    let payload_bytes = streams.iter().try_fold(0usize, |sum, stream| {
        sum.checked_add(stream.len())
            .ok_or_else(|| ColicError::Usage("rANS payload size overflows usize".into()))
    })?;
    if payload_bytes > u32::MAX as usize {
        return Err(ColicError::Usage(
            "rANS payload exceeds 32-bit stream-offset contract".into(),
        ));
    }
    let total = header_bytes
        .checked_add(round16(payload_bytes))
        .ok_or_else(|| ColicError::Usage("rANS record size overflows usize".into()))?;
    let mut output = vec![0_u8; total];
    put_u64(&mut output, 0, n_symbols as u64);
    put_u64(&mut output, 8, input.len() as u64);
    let mut cursor = 0usize;
    for (index, stream) in streams.iter().enumerate() {
        put_u32(&mut output, 16 + index * 4, cursor as u32);
        let begin = header_bytes + cursor;
        output[begin..begin + stream.len()].copy_from_slice(stream);
        cursor += stream.len();
    }
    put_u32(&mut output, 16 + N_STREAMS * 4, cursor as u32);
    Ok(output)
}

pub fn decode_bytes(record: &[u8], table: &Table, expected_bytes: usize) -> Result<Vec<u8>> {
    table.validate()?;
    let parsed = ParsedRecord::parse(record)?;
    if parsed.packed_bytes != expected_bytes
        || parsed.n_symbols
            != expected_bytes.checked_mul(2).ok_or_else(|| {
                ColicError::Usage("expected rANS symbol count overflows usize".into())
            })?
    {
        return Err(ColicError::Usage(
            "rANS decoded length does not match matrix descriptor".into(),
        ));
    }
    let slots = table.slot_to_symbol()?;
    let mut output = vec![0_u8; expected_bytes];
    for stream in 0..N_STREAMS {
        let begin = parsed.offsets[stream] as usize;
        let end = parsed.offsets[stream + 1] as usize;
        let symbols = stream_symbol_count(parsed.n_symbols, stream);
        let decoded = decode_stream_checked(&parsed.payload[begin..end], symbols, table, &slots)?;
        for (index, symbol) in decoded.into_iter().enumerate() {
            let logical = stream + index * N_STREAMS;
            let byte = &mut output[logical / 2];
            if logical.is_multiple_of(2) {
                *byte = (*byte & 0xf0) | symbol;
            } else {
                *byte = (*byte & 0x0f) | (symbol << 4);
            }
        }
    }
    Ok(output)
}

pub fn auto_should_use(raw_bytes: usize, encoded_bytes: usize) -> bool {
    let encoded_with_slack = match encoded_bytes.checked_add(READABLE_SLACK) {
        Some(value) => value,
        None => return false,
    };
    if encoded_with_slack >= raw_bytes {
        return false;
    }
    let saved = raw_bytes - encoded_with_slack;
    if saved < AUTO_MIN_SAVINGS_BYTES {
        return false;
    }
    (saved as u128) * 10_000 >= (raw_bytes as u128) * u128::from(AUTO_MIN_SAVINGS_BPS)
}

pub fn manifest_table_region(mut manifest: Vec<u8>, table: Option<&Table>) -> Result<Vec<u8>> {
    if table.is_none() {
        return Ok(manifest);
    }
    if manifest.len() < 256 {
        return Err(ColicError::Usage(
            "manifest is too short for codec tables".into(),
        ));
    }
    if u32_at(&manifest, 160)? != 0 || u64_at(&manifest, 168)? != 0 || u64_at(&manifest, 176)? != 0
    {
        return Err(ColicError::Usage(
            "manifest already contains a codec table region".into(),
        ));
    }
    let table = table.unwrap();
    let blob = table.encode_blob()?;
    const DESC_BYTES: usize = 64;
    let region_offset = round16(manifest.len());
    manifest.resize(region_offset, 0);
    let data_offset = round16(DESC_BYTES);
    let region_bytes = data_offset
        .checked_add(blob.len())
        .and_then(|value| value.checked_add(15))
        .map(|value| value & !15)
        .ok_or_else(|| ColicError::Usage("codec table region size overflows".into()))?;
    let start = manifest.len();
    manifest.resize(start + region_bytes, 0);
    put_u32(&mut manifest, start, TABLE_ID);
    put_u16(&mut manifest, start + 4, CODEC_ID);
    put_i32(&mut manifest, start + 8, -1);
    put_u64(&mut manifest, start + 16, data_offset as u64);
    put_u64(&mut manifest, start + 24, blob.len() as u64);
    put_u32(&mut manifest, start + 32, storage::crc32c(&blob));
    manifest[start + data_offset..start + data_offset + blob.len()].copy_from_slice(&blob);

    put_u32(&mut manifest, 160, 1);
    put_u64(&mut manifest, 168, region_offset as u64);
    put_u64(&mut manifest, 176, region_bytes as u64);
    manifest[144..148].fill(0);
    let crc = storage::crc32c(&manifest);
    put_u32(&mut manifest, 144, crc);
    Ok(manifest)
}

pub fn table_from_manifest(manifest: &[u8], table_id: u32, codec: u16) -> Result<Table> {
    if table_id == 0 || codec != CODEC_ID {
        return Err(ColicError::Usage(
            "invalid rANS codec/table reference".into(),
        ));
    }
    let count = u32_at(manifest, 160)? as usize;
    let region_offset = usize::try_from(u64_at(manifest, 168)?)
        .map_err(|_| ColicError::Usage("codec table offset exceeds usize".into()))?;
    let region_bytes = usize::try_from(u64_at(manifest, 176)?)
        .map_err(|_| ColicError::Usage("codec table size exceeds usize".into()))?;
    let region = manifest
        .get(
            region_offset
                ..region_offset.checked_add(region_bytes).ok_or_else(|| {
                    ColicError::Usage("codec table region overflows usize".into())
                })?,
        )
        .ok_or_else(|| ColicError::Usage("codec table region is outside manifest".into()))?;
    for index in 0..count {
        let desc = index
            .checked_mul(64)
            .ok_or_else(|| ColicError::Usage("codec table descriptor overflows".into()))?;
        if desc + 64 > region.len() {
            return Err(ColicError::Usage("truncated codec table descriptor".into()));
        }
        if u32_at(region, desc)? != table_id {
            continue;
        }
        if u16_at(region, desc + 4)? != codec
            || u16_at(region, desc + 6)? != 0
            || i32_at(region, desc + 8)? != -1
            || u32_at(region, desc + 12)? != 0
            || u32_at(region, desc + 36)? != 0
            || region[desc + 40..desc + 64].iter().any(|byte| *byte != 0)
        {
            return Err(ColicError::Usage(
                "invalid rANS codec table descriptor".into(),
            ));
        }
        let data_offset = usize::try_from(u64_at(region, desc + 16)?)
            .map_err(|_| ColicError::Usage("codec table data offset exceeds usize".into()))?;
        let data_bytes = usize::try_from(u64_at(region, desc + 24)?)
            .map_err(|_| ColicError::Usage("codec table data size exceeds usize".into()))?;
        let blob = region
            .get(
                data_offset
                    ..data_offset.checked_add(data_bytes).ok_or_else(|| {
                        ColicError::Usage("codec table data span overflows".into())
                    })?,
            )
            .ok_or_else(|| ColicError::Usage("codec table data is outside region".into()))?;
        if storage::crc32c(blob) != u32_at(region, desc + 32)? {
            return Err(ColicError::Usage("codec table CRC32C mismatch".into()));
        }
        return Table::decode_blob(blob);
    }
    Err(ColicError::Usage(format!(
        "missing rANS codec table {table_id}"
    )))
}

fn encode_stream(symbols: &[u8], table: &Table) -> Result<Vec<u8>> {
    let mut state = RANS_L;
    let mut output = Vec::new();
    for symbol in symbols.iter().rev().copied() {
        if symbol > 15 || table.freq[symbol as usize] == 0 {
            return Err(ColicError::Usage(
                "rANS input contains a symbol absent from its table".into(),
            ));
        }
        let frequency = table.freq[symbol as usize];
        let start = table.start[symbol as usize];
        let x_max = ((RANS_L >> SCALE_BITS) << 8)
            .checked_mul(frequency)
            .ok_or_else(|| ColicError::Usage("rANS normalization threshold overflows".into()))?;
        while state >= x_max {
            output.push((state & 0xff) as u8);
            state >>= 8;
        }
        state = ((state / frequency) << SCALE_BITS) + (state % frequency) + start;
    }
    for _ in 0..4 {
        output.push((state & 0xff) as u8);
        state >>= 8;
    }
    output.reverse();
    Ok(output)
}

fn decode_stream_checked(
    stream: &[u8],
    symbols: usize,
    table: &Table,
    slots: &[u16],
) -> Result<Vec<u8>> {
    if stream.len() < 4 {
        return Err(ColicError::Usage(
            "rANS stream is shorter than its state".into(),
        ));
    }
    let mut cursor = 4usize;
    let mut state = u32::from_be_bytes(stream[..4].try_into().unwrap());
    if state < RANS_L {
        return Err(ColicError::Usage(
            "rANS initial state is out of range".into(),
        ));
    }
    let mask = M - 1;
    let mut output = Vec::with_capacity(symbols);
    for _ in 0..symbols {
        let slot = state & mask;
        let symbol = slots[slot as usize] as usize;
        output.push(symbol as u8);
        state = table.freq[symbol] * (state >> SCALE_BITS) + slot - table.start[symbol];
        while state < RANS_L && cursor < stream.len() {
            state = (state << 8) | u32::from(stream[cursor]);
            cursor += 1;
        }
    }
    if cursor != stream.len() {
        return Err(ColicError::Usage("rANS stream has trailing bytes".into()));
    }
    if state != RANS_L {
        return Err(ColicError::Usage(
            "rANS stream final state is invalid".into(),
        ));
    }
    Ok(output)
}

struct ParsedRecord<'a> {
    n_symbols: usize,
    packed_bytes: usize,
    offsets: Vec<u32>,
    payload: &'a [u8],
}

impl<'a> ParsedRecord<'a> {
    fn parse(record: &'a [u8]) -> Result<Self> {
        let header = round16(16 + (N_STREAMS + 1) * 4);
        if record.len() < header {
            return Err(ColicError::Usage("truncated rANS record".into()));
        }
        let n_symbols = usize::try_from(u64_at(record, 0)?)
            .map_err(|_| ColicError::Usage("rANS symbol count exceeds usize".into()))?;
        let packed_bytes = usize::try_from(u64_at(record, 8)?)
            .map_err(|_| ColicError::Usage("rANS packed-byte count exceeds usize".into()))?;
        if n_symbols == 0 || packed_bytes != n_symbols / 2 + n_symbols % 2 {
            return Err(ColicError::Usage("invalid rANS symbol/byte counts".into()));
        }
        let mut offsets = Vec::with_capacity(N_STREAMS + 1);
        for stream in 0..=N_STREAMS {
            offsets.push(u32_at(record, 16 + stream * 4)?);
        }
        if offsets[0] != 0 {
            return Err(ColicError::Usage(
                "rANS first stream offset is nonzero".into(),
            ));
        }
        for pair in offsets.windows(2) {
            if pair[1] < pair[0] || pair[1] - pair[0] < 4 {
                return Err(ColicError::Usage(
                    "rANS stream offsets are malformed".into(),
                ));
            }
        }
        let payload_bytes = offsets[N_STREAMS] as usize;
        let expected = header
            .checked_add(round16(payload_bytes))
            .ok_or_else(|| ColicError::Usage("rANS record length overflows".into()))?;
        if expected != record.len() || header + payload_bytes > record.len() {
            return Err(ColicError::Usage("rANS record length mismatch".into()));
        }
        if record[16 + (N_STREAMS + 1) * 4..header]
            .iter()
            .any(|byte| *byte != 0)
            || record[header + payload_bytes..]
                .iter()
                .any(|byte| *byte != 0)
        {
            return Err(ColicError::Usage("rANS record padding is nonzero".into()));
        }
        let max_symbols = (payload_bytes as u128) * 8 * (1_u128 << 15);
        if n_symbols as u128 > max_symbols {
            return Err(ColicError::Usage(
                "rANS record exceeds amplification bound".into(),
            ));
        }
        Ok(Self {
            n_symbols,
            packed_bytes,
            offsets,
            payload: &record[header..header + payload_bytes],
        })
    }
}

fn stream_symbol_count(total: usize, stream: usize) -> usize {
    if stream >= total {
        0
    } else {
        (total - 1 - stream) / N_STREAMS + 1
    }
}

fn round16(value: usize) -> usize {
    (value + 15) & !15
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or_else(|| ColicError::Usage("truncated rANS u16".into()))?
            .try_into()
            .unwrap(),
    ))
}
fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| ColicError::Usage("truncated rANS u32".into()))?
            .try_into()
            .unwrap(),
    ))
}
fn i32_at(bytes: &[u8], offset: usize) -> Result<i32> {
    Ok(i32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| ColicError::Usage("truncated rANS i32".into()))?
            .try_into()
            .unwrap(),
    ))
}
fn u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| ColicError::Usage("truncated rANS u64".into()))?
            .try_into()
            .unwrap(),
    ))
}
fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        (0..8192)
            .map(|index| match index % 17 {
                0..=11 => 0x11,
                12..=14 => 0x00,
                15 => 0x21,
                _ => 0x10,
            })
            .collect()
    }

    #[test]
    fn deterministic_table_and_round_trip() {
        let bytes = fixture();
        let table = Table::from_histogram(histogram_bytes([bytes.as_slice()]).unwrap()).unwrap();
        let encoded_a = encode_bytes(&bytes, &table).unwrap();
        let encoded_b = encode_bytes(&bytes, &table).unwrap();
        assert_eq!(encoded_a, encoded_b);
        assert_eq!(
            decode_bytes(&encoded_a, &table, bytes.len()).unwrap(),
            bytes
        );
        assert_eq!(
            Table::decode_blob(&table.encode_blob().unwrap()).unwrap(),
            table
        );
    }

    #[test]
    fn table_quantization_matches_reference_negative_deficit_order() {
        let histogram = [1, 2, 100, 1, 0, 1, 1, 3, 2, 10, 0, 1, 1, 1, 0, 100];
        let table = Table::from_histogram(histogram).unwrap();
        assert_eq!(
            table.freq,
            [
                73, 146, 7315, 73, 0, 73, 73, 220, 146, 732, 0, 73, 73, 73, 0, 7314
            ]
        );
    }

    #[test]
    fn covers_every_nibble_value() {
        let bytes = (0_u8..=255).cycle().take(4096).collect::<Vec<_>>();
        let table = Table::from_histogram(histogram_bytes([bytes.as_slice()]).unwrap()).unwrap();
        let encoded = encode_bytes(&bytes, &table).unwrap();
        assert_eq!(decode_bytes(&encoded, &table, bytes.len()).unwrap(), bytes);
    }

    #[test]
    fn malformed_records_fail_closed() {
        let bytes = fixture();
        let table = Table::from_histogram(histogram_bytes([bytes.as_slice()]).unwrap()).unwrap();
        let encoded = encode_bytes(&bytes, &table).unwrap();
        for mutation in [0usize, 16, encoded.len() - 1] {
            let mut damaged = encoded.clone();
            damaged[mutation] ^= 0x5a;
            assert!(decode_bytes(&damaged, &table, bytes.len()).is_err());
        }
        assert!(decode_bytes(&encoded[..encoded.len() - 16], &table, bytes.len()).is_err());
    }

    #[test]
    fn auto_policy_requires_meaningful_savings() {
        assert!(!auto_should_use(4096, 3900));
        assert!(auto_should_use(4096, 3500));
        assert!(auto_should_use(1024, 700));
        assert!(!auto_should_use(1024, 710));
    }
}

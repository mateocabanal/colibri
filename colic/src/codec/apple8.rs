use std::collections::BTreeMap;

use crate::{
    codec::rans256,
    error::{ColicError, Result},
};

const PREFIX: usize = 64 + 3 * 128;
const DESC_BYTES: usize = 128;
const MATRIX_COUNT: usize = 3;
const ALIGNMENT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Force,
    Auto,
}

#[derive(Debug, Clone)]
pub struct Prepared {
    pub table: rans256::Table,
    pub experts: BTreeMap<(u32, u32), Vec<u8>>,
    pub compressed_matrices: usize,
    pub raw_matrix_bytes: u64,
    pub stored_matrix_bytes: u64,
}

pub fn prepare(
    raw_experts: &BTreeMap<(u32, u32), Vec<u8>>,
    mode: Mode,
) -> Result<Prepared> {
    if raw_experts.is_empty() {
        return Err(ColicError::Usage(
            "Apple8 rANS preparation requires routed experts".into(),
        ));
    }
    let mut matrix_payloads = Vec::new();
    for raw in raw_experts.values() {
        let parsed = RawExpert::parse(raw)?;
        for matrix in parsed.matrices {
            matrix_payloads.push(raw[matrix.offset..matrix.offset + matrix.bytes].to_vec());
        }
    }
    let hist = rans256::histogram_bytes(matrix_payloads.iter().map(Vec::as_slice))?;
    let table = rans256::Table::from_histogram(hist)?;

    let mut experts = BTreeMap::new();
    let mut compressed_matrices = 0usize;
    let mut raw_matrix_bytes = 0_u64;
    let mut stored_matrix_bytes = 0_u64;
    for (key, raw) in raw_experts {
        let (encoded, stats) = encode_expert(raw, &table, mode)?;
        compressed_matrices = compressed_matrices
            .checked_add(stats.compressed)
            .ok_or_else(|| ColicError::Usage("compressed matrix count overflows usize".into()))?;
        raw_matrix_bytes = raw_matrix_bytes
            .checked_add(stats.raw_bytes)
            .ok_or_else(|| ColicError::Usage("raw matrix byte total overflows u64".into()))?;
        stored_matrix_bytes = stored_matrix_bytes
            .checked_add(stats.stored_bytes)
            .ok_or_else(|| ColicError::Usage("stored matrix byte total overflows u64".into()))?;
        experts.insert(*key, encoded);
    }
    Ok(Prepared {
        table,
        experts,
        compressed_matrices,
        raw_matrix_bytes,
        stored_matrix_bytes,
    })
}

#[derive(Debug, Clone, Copy)]
struct MatrixSpan {
    offset: usize,
    bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct RawExpert {
    matrices: [MatrixSpan; MATRIX_COUNT],
}

impl RawExpert {
    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < PREFIX
            || bytes.get(..8) != Some(b"COLIEXPT".as_slice())
            || u16_at(bytes, 8)? != 1
            || u32_at(bytes, 12)? != 64
            || u16_at(bytes, 24)? != 3
            || u32_at(bytes, 28)? != 128
            || u64_at(bytes, 40)? != PREFIX as u64
        {
            return Err(ColicError::Usage(
                "rANS input is not a canonical Apple8 expert record".into(),
            ));
        }
        let mut matrices = [MatrixSpan { offset: 0, bytes: 0 }; MATRIX_COUNT];
        for (index, slot) in matrices.iter_mut().enumerate() {
            let desc = 64 + index * DESC_BYTES;
            let codec = u16_at(bytes, desc + 8)?;
            let table = u32_at(bytes, desc + 40)?;
            let offset = usize::try_from(u64_at(bytes, desc + 48)?)
                .map_err(|_| ColicError::Usage("Apple8 matrix offset exceeds usize".into()))?;
            let stored = usize::try_from(u64_at(bytes, desc + 56)?)
                .map_err(|_| ColicError::Usage("Apple8 matrix size exceeds usize".into()))?;
            let decoded = usize::try_from(u64_at(bytes, desc + 64)?)
                .map_err(|_| ColicError::Usage("Apple8 decoded size exceeds usize".into()))?;
            let end = offset
                .checked_add(stored)
                .ok_or_else(|| ColicError::Usage("Apple8 matrix span overflows usize".into()))?;
            if codec != 0 || table != 0 || stored != decoded || offset < PREFIX || end > bytes.len() {
                return Err(ColicError::Usage(
                    "rANS input Apple8 record is not raw Design A".into(),
                ));
            }
            *slot = MatrixSpan {
                offset,
                bytes: decoded,
            };
        }
        Ok(Self { matrices })
    }
}

#[derive(Default)]
struct Stats {
    compressed: usize,
    raw_bytes: u64,
    stored_bytes: u64,
}

fn encode_expert(raw: &[u8], table: &rans256::Table, mode: Mode) -> Result<(Vec<u8>, Stats)> {
    let parsed = RawExpert::parse(raw)?;
    let mut output = raw[..PREFIX].to_vec();
    let mut stats = Stats::default();
    for (index, matrix) in parsed.matrices.iter().enumerate() {
        let raw_payload = &raw[matrix.offset..matrix.offset + matrix.bytes];
        let encoded = rans256::encode_bytes(raw_payload, table)?;
        let use_codec = match mode {
            Mode::Force => true,
            Mode::Auto => rans256::auto_should_use(raw_payload.len(), encoded.len()),
        };
        align_vec(&mut output, ALIGNMENT)?;
        let offset = output.len();
        let desc = 64 + index * DESC_BYTES;
        if use_codec {
            put_u16(&mut output, desc + 8, rans256::CODEC_ID);
            put_u32(&mut output, desc + 40, rans256::TABLE_ID);
            put_u64(&mut output, desc + 48, offset as u64);
            put_u64(&mut output, desc + 56, encoded.len() as u64);
            put_u64(&mut output, desc + 64, raw_payload.len() as u64);
            output.extend_from_slice(&encoded);
            output.resize(
                output
                    .len()
                    .checked_add(rans256::READABLE_SLACK)
                    .ok_or_else(|| ColicError::Usage("rANS expert slack overflows usize".into()))?,
                0,
            );
            stats.compressed += 1;
            stats.stored_bytes = stats
                .stored_bytes
                .checked_add((encoded.len() + rans256::READABLE_SLACK) as u64)
                .ok_or_else(|| ColicError::Usage("rANS stored total overflows u64".into()))?;
        } else {
            put_u16(&mut output, desc + 8, 0);
            put_u32(&mut output, desc + 40, 0);
            put_u64(&mut output, desc + 48, offset as u64);
            put_u64(&mut output, desc + 56, raw_payload.len() as u64);
            put_u64(&mut output, desc + 64, raw_payload.len() as u64);
            output.extend_from_slice(raw_payload);
            stats.stored_bytes = stats
                .stored_bytes
                .checked_add(raw_payload.len() as u64)
                .ok_or_else(|| ColicError::Usage("raw stored total overflows u64".into()))?;
        }
        stats.raw_bytes = stats
            .raw_bytes
            .checked_add(raw_payload.len() as u64)
            .ok_or_else(|| ColicError::Usage("raw matrix total overflows u64".into()))?;
    }
    Ok((output, stats))
}

pub fn decode_to_raw(encoded: &[u8], table: &rans256::Table) -> Result<Vec<u8>> {
    if encoded.len() < PREFIX || encoded.get(..8) != Some(b"COLIEXPT".as_slice()) {
        return Err(ColicError::Usage("invalid encoded Apple8 expert".into()));
    }
    let mut output = encoded[..PREFIX].to_vec();
    for index in 0..MATRIX_COUNT {
        let desc = 64 + index * DESC_BYTES;
        let codec = u16_at(encoded, desc + 8)?;
        let table_id = u32_at(encoded, desc + 40)?;
        let source_offset = usize::try_from(u64_at(encoded, desc + 48)?)
            .map_err(|_| ColicError::Usage("encoded matrix offset exceeds usize".into()))?;
        let stored = usize::try_from(u64_at(encoded, desc + 56)?)
            .map_err(|_| ColicError::Usage("encoded matrix size exceeds usize".into()))?;
        let decoded = usize::try_from(u64_at(encoded, desc + 64)?)
            .map_err(|_| ColicError::Usage("decoded matrix size exceeds usize".into()))?;
        let source_end = source_offset
            .checked_add(stored)
            .ok_or_else(|| ColicError::Usage("encoded matrix span overflows usize".into()))?;
        let source = encoded
            .get(source_offset..source_end)
            .ok_or_else(|| ColicError::Usage("encoded matrix lies outside expert record".into()))?;
        let payload = if codec == 0 {
            if table_id != 0 || stored != decoded {
                return Err(ColicError::Usage("invalid raw Apple8 matrix codec fields".into()));
            }
            source.to_vec()
        } else if codec == rans256::CODEC_ID && table_id == rans256::TABLE_ID {
            let slack_end = source_end
                .checked_add(rans256::READABLE_SLACK)
                .ok_or_else(|| ColicError::Usage("rANS slack span overflows usize".into()))?;
            if encoded
                .get(source_end..slack_end)
                .is_none_or(|slack| slack.iter().any(|byte| *byte != 0))
            {
                return Err(ColicError::Usage("rANS readable slack is missing or nonzero".into()));
            }
            rans256::decode_bytes(source, table, decoded)?
        } else {
            return Err(ColicError::Usage("unsupported Apple8 matrix codec".into()));
        };
        if crate::storage::crc32c(&payload) != u32_at(encoded, desc + 96)? {
            return Err(ColicError::Usage("decoded Apple8 matrix CRC32C mismatch".into()));
        }
        align_vec(&mut output, ALIGNMENT)?;
        let destination = output.len();
        put_u16(&mut output, desc + 8, 0);
        put_u32(&mut output, desc + 40, 0);
        put_u64(&mut output, desc + 48, destination as u64);
        put_u64(&mut output, desc + 56, decoded as u64);
        put_u64(&mut output, desc + 64, decoded as u64);
        output.extend_from_slice(&payload);
    }
    Ok(output)
}

fn align_vec(bytes: &mut Vec<u8>, alignment: usize) -> Result<()> {
    let target = bytes
        .len()
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| ColicError::Usage("Apple8 codec alignment overflows usize".into()))?;
    bytes.resize(target, 0);
    Ok(())
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or_else(|| ColicError::Usage("truncated Apple8 u16".into()))?
            .try_into()
            .unwrap(),
    ))
}
fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| ColicError::Usage("truncated Apple8 u32".into()))?
            .try_into()
            .unwrap(),
    ))
}
fn u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| ColicError::Usage("truncated Apple8 u64".into()))?
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
fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_expert(matrix: &[u8]) -> Vec<u8> {
        let mut out = vec![0_u8; PREFIX];
        out[..8].copy_from_slice(b"COLIEXPT");
        put_u16(&mut out, 8, 1);
        out[12..16].copy_from_slice(&64_u32.to_le_bytes());
        put_u16(&mut out, 24, 3);
        out[28..32].copy_from_slice(&128_u32.to_le_bytes());
        put_u64(&mut out, 32, 64);
        put_u64(&mut out, 40, PREFIX as u64);
        put_u64(&mut out, 48, (matrix.len() * 3) as u64);
        for index in 0..3 {
            let desc = 64 + index * 128;
            put_u16(&mut out, desc, (index + 1) as u16);
            put_u64(&mut out, desc + 16, 8);
            put_u64(&mut out, desc + 24, 32);
            let offset = out.len();
            put_u64(&mut out, desc + 48, offset as u64);
            put_u64(&mut out, desc + 56, matrix.len() as u64);
            put_u64(&mut out, desc + 64, matrix.len() as u64);
            put_u32(&mut out, desc + 96, crate::storage::crc32c(matrix));
            out.extend_from_slice(matrix);
        }
        out
    }

    #[test]
    fn compressed_record_decodes_to_byte_identical_raw_record() {
        let matrix = vec![0x11; 4096];
        let raw = raw_expert(&matrix);
        let mut source = BTreeMap::new();
        source.insert((0, 0), raw.clone());
        let prepared = prepare(&source, Mode::Force).unwrap();
        assert_eq!(prepared.compressed_matrices, 3);
        let encoded = prepared.experts.get(&(0, 0)).unwrap();
        assert_eq!(decode_to_raw(encoded, &prepared.table).unwrap(), raw);
    }
}

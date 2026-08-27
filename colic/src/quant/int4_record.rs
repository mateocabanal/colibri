//! COLIEXPT lowering for compiler-quantized grouped INT4 routed experts.
//!
//! The payload matches the existing CPU runtime contract: canonical row-major
//! biased INT4 nibbles, f32 scales per 32 input columns, and group_size=32.

use crate::{
    error::{ColicError, Result},
    ir::{Matrix, RoutedExpert},
    quant::int4::{self, PackedMatrix},
    storage::{align_up, crc32c},
};

const HEADER_BYTES: usize = 64;
const DESC_BYTES: usize = 128;
const MATRIX_COUNT: usize = 3;
const DATA_OFFSET: usize = HEADER_BYTES + DESC_BYTES * MATRIX_COUNT;
const DATA_ALIGNMENT: u64 = 16;

pub const MATH_FORMAT_INT4_GROUPED: u16 = 0x22;
pub const SCALE_FORMAT_F32: u16 = 1;
pub const BLOCK_ROWS: u32 = 1;
pub const BLOCK_COLUMNS: u32 = int4::GROUP_SIZE as u32;
pub const GROUP_SIZE: u32 = int4::GROUP_SIZE as u32;

pub fn lower_expert(expert: &RoutedExpert) -> Result<Vec<u8>> {
    let gate = int4::quantize_matrix(&expert.gate)?;
    let up = int4::quantize_matrix(&expert.up)?;
    let down = int4::quantize_matrix(&expert.down)?;
    lower_packed_expert(expert.layer, expert.expert, [&gate, &up, &down])
}

pub fn stored_bytes(expert: &RoutedExpert) -> Result<u64> {
    [&expert.gate, &expert.up, &expert.down]
        .into_iter()
        .try_fold(DATA_OFFSET as u64, |cursor, matrix| {
            let after_weight = align_up(cursor, DATA_ALIGNMENT)?
                .checked_add(packed_weight_bytes(matrix)?)
                .ok_or_else(|| ColicError::Usage("INT4 expert size overflows u64".into()))?;
            align_up(after_weight, DATA_ALIGNMENT)?
                .checked_add(packed_scale_bytes(matrix)?)
                .ok_or_else(|| ColicError::Usage("INT4 expert size overflows u64".into()))
        })
}

pub fn resident_bytes(expert: &RoutedExpert) -> Result<u64> {
    [&expert.gate, &expert.up, &expert.down]
        .into_iter()
        .try_fold(0_u64, |total, matrix| {
            total
                .checked_add(packed_weight_bytes(matrix)?)
                .and_then(|bytes| bytes.checked_add(packed_scale_bytes(matrix).ok()?))
                .ok_or_else(|| ColicError::Usage("INT4 resident size overflows u64".into()))
        })
}

fn packed_weight_bytes(matrix: &Matrix) -> Result<u64> {
    let row_bytes = u64::from(matrix.columns).div_ceil(2);
    u64::from(matrix.rows)
        .checked_mul(row_bytes)
        .ok_or_else(|| ColicError::Usage("INT4 weight size overflows u64".into()))
}

fn packed_scale_bytes(matrix: &Matrix) -> Result<u64> {
    let groups = u64::from(matrix.columns).div_ceil(int4::GROUP_SIZE as u64);
    u64::from(matrix.rows)
        .checked_mul(groups)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| ColicError::Usage("INT4 scale size overflows u64".into()))
}

fn lower_packed_expert(
    layer: u32,
    expert: u32,
    matrices: [&PackedMatrix; MATRIX_COUNT],
) -> Result<Vec<u8>> {
    let mut payload = vec![0_u8; DATA_OFFSET];
    payload[..8].copy_from_slice(b"COLIEXPT");
    put_u16(&mut payload, 8, 1);
    put_u16(&mut payload, 10, 0);
    put_u32(&mut payload, 12, HEADER_BYTES as u32);
    put_i32(
        &mut payload,
        16,
        i32::try_from(layer)
            .map_err(|_| ColicError::Usage("INT4 expert layer exceeds COLI i32 range".into()))?,
    );
    put_i32(
        &mut payload,
        20,
        i32::try_from(expert)
            .map_err(|_| ColicError::Usage("INT4 expert id exceeds COLI i32 range".into()))?,
    );
    put_u16(&mut payload, 24, MATRIX_COUNT as u16);
    put_u32(&mut payload, 28, DESC_BYTES as u32);
    put_u64(&mut payload, 32, HEADER_BYTES as u64);
    put_u64(&mut payload, 40, DATA_OFFSET as u64);

    let roles = [1_u16, 2_u16, 3_u16];
    let mut resident = 0_u64;
    for (index, matrix) in matrices.into_iter().enumerate() {
        let scale_bytes = matrix.scale_bytes_le();
        let weight_offset = append_aligned(&mut payload, &matrix.weights)?;
        let scale_offset = append_aligned(&mut payload, &scale_bytes)?;
        let desc = HEADER_BYTES + index * DESC_BYTES;
        put_u16(&mut payload, desc, roles[index]);
        put_u16(&mut payload, desc + 4, MATH_FORMAT_INT4_GROUPED);
        put_u16(&mut payload, desc + 6, SCALE_FORMAT_F32);
        put_u64(&mut payload, desc + 16, u64::from(matrix.rows));
        put_u64(&mut payload, desc + 24, u64::from(matrix.columns));
        put_u32(&mut payload, desc + 32, BLOCK_ROWS);
        put_u32(&mut payload, desc + 36, BLOCK_COLUMNS);
        put_u64(&mut payload, desc + 48, weight_offset);
        put_u64(&mut payload, desc + 56, matrix.weights.len() as u64);
        put_u64(&mut payload, desc + 64, matrix.weights.len() as u64);
        put_u64(&mut payload, desc + 72, scale_offset);
        put_u64(&mut payload, desc + 80, scale_bytes.len() as u64);
        put_u64(&mut payload, desc + 88, scale_bytes.len() as u64);
        put_u32(&mut payload, desc + 104, GROUP_SIZE);

        let mut logical = Vec::with_capacity(matrix.weights.len() + scale_bytes.len());
        logical.extend_from_slice(&matrix.weights);
        logical.extend_from_slice(&scale_bytes);
        put_u32(&mut payload, desc + 96, crc32c(&logical));
        resident = resident
            .checked_add(logical.len() as u64)
            .ok_or_else(|| ColicError::Usage("INT4 resident size overflows u64".into()))?;
    }
    put_u64(&mut payload, 48, resident);
    Ok(payload)
}

fn append_aligned(output: &mut Vec<u8>, bytes: &[u8]) -> Result<u64> {
    let offset = align_up(output.len() as u64, DATA_ALIGNMENT)?;
    let offset_usize = usize::try_from(offset)
        .map_err(|_| ColicError::Usage("INT4 record offset exceeds usize".into()))?;
    output.resize(offset_usize, 0);
    output.extend_from_slice(bytes);
    Ok(offset)
}

fn put_u16(buffer: &mut [u8], offset: usize, value: u16) {
    buffer[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(buffer: &mut [u8], offset: usize, value: u64) {
    buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn put_i32(buffer: &mut [u8], offset: usize, value: i32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::{ir::Matrix, source::TensorRef};

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        bytes
    }

    fn matrix(path: &std::path::Path, offset: u64, rows: u32, columns: u32) -> Matrix {
        Matrix {
            source: TensorRef {
                source: path.to_owned(),
                offset,
                len: u64::from(rows) * u64::from(columns) * 2,
                dtype: "BF16".into(),
                shape: vec![u64::from(rows), u64::from(columns)],
            },
            rows,
            columns,
            scale: None,
        }
    }

    #[test]
    fn lowers_into_grouped_int4_runtime_contract() {
        let path = std::env::temp_dir().join(format!("colic-int4-record-{}", std::process::id()));
        let gate = vec![1.0_f32; 64];
        let up = vec![2.0_f32; 64];
        let down = vec![3.0_f32; 64];
        let mut source = bf16_bytes(&gate);
        let up_offset = source.len() as u64;
        source.extend_from_slice(&bf16_bytes(&up));
        let down_offset = source.len() as u64;
        source.extend_from_slice(&bf16_bytes(&down));
        fs::write(&path, &source).unwrap();
        let expert = RoutedExpert {
            layer: 2,
            expert: 5,
            gate: matrix(&path, 0, 2, 32),
            up: matrix(&path, up_offset, 2, 32),
            down: matrix(&path, down_offset, 2, 32),
        };

        let bytes = lower_expert(&expert).unwrap();
        assert_eq!(&bytes[..8], b"COLIEXPT");
        for index in 0..3 {
            let desc = HEADER_BYTES + index * DESC_BYTES;
            assert_eq!(u16::from_le_bytes(bytes[desc + 4..desc + 6].try_into().unwrap()), MATH_FORMAT_INT4_GROUPED);
            assert_eq!(u16::from_le_bytes(bytes[desc + 6..desc + 8].try_into().unwrap()), SCALE_FORMAT_F32);
            assert_eq!(u32::from_le_bytes(bytes[desc + 32..desc + 36].try_into().unwrap()), 1);
            assert_eq!(u32::from_le_bytes(bytes[desc + 36..desc + 40].try_into().unwrap()), 32);
            assert_eq!(u32::from_le_bytes(bytes[desc + 104..desc + 108].try_into().unwrap()), 32);
            assert_eq!(u64::from_le_bytes(bytes[desc + 56..desc + 64].try_into().unwrap()), 32);
            assert_eq!(u64::from_le_bytes(bytes[desc + 80..desc + 88].try_into().unwrap()), 8);
        }
        assert_eq!(stored_bytes(&expert).unwrap(), bytes.len() as u64);
        assert_eq!(resident_bytes(&expert).unwrap(), 120);
        fs::remove_file(path).unwrap();
    }
}

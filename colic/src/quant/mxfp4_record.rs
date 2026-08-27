//! COLIEXPT lowering for compiler-quantized MXFP4 routed experts.
//!
//! This deliberately reuses the existing compound expert envelope. The inner
//! matrix descriptors carry the physical contract: E2M1 packed nibbles,
//! E8M0 scales, block axis=columns, block size=32. No container-version change
//! is needed for this quantization profile.

use crate::{
    error::{ColicError, Result},
    ir::{Matrix, RoutedExpert},
    quant::mxfp4::{self, PackedMatrix},
    storage::{align_up, crc32c},
};

const HEADER_BYTES: usize = 64;
const DESC_BYTES: usize = 128;
const MATRIX_COUNT: usize = 3;
const DATA_OFFSET: usize = HEADER_BYTES + DESC_BYTES * MATRIX_COUNT;
const DATA_ALIGNMENT: u64 = 16;

/// Existing COLI expert-descriptor IDs used by the runtime for MXFP4.
pub const MATH_FORMAT_MXFP4: u16 = 0x20;
pub const SCALE_FORMAT_E8M0: u16 = 4;
pub const BLOCK_AXIS_COLUMNS: u32 = 1;
pub const BLOCK_SIZE: u32 = 32;

pub fn lower_expert(expert: &RoutedExpert) -> Result<Vec<u8>> {
    let gate = mxfp4::quantize_matrix(&expert.gate)?;
    let up = mxfp4::quantize_matrix(&expert.up)?;
    let down = mxfp4::quantize_matrix(&expert.down)?;
    lower_packed_expert(expert.layer, expert.expert, [&gate, &up, &down])
}

pub fn stored_bytes(expert: &RoutedExpert) -> Result<u64> {
    [&expert.gate, &expert.up, &expert.down]
        .into_iter()
        .try_fold(DATA_OFFSET as u64, |cursor, matrix| {
            let weight_bytes = packed_weight_bytes(matrix)?;
            let scale_bytes = packed_scale_bytes(matrix)?;
            let after_weight = align_up(cursor, DATA_ALIGNMENT)?
                .checked_add(weight_bytes)
                .ok_or_else(|| ColicError::Usage("MXFP4 expert size overflows u64".into()))?;
            align_up(after_weight, DATA_ALIGNMENT)?
                .checked_add(scale_bytes)
                .ok_or_else(|| ColicError::Usage("MXFP4 expert size overflows u64".into()))
        })
}

/// Bytes that need to be resident for execution after the COLIEXPT framing is
/// stripped: packed E2M1 weights plus raw E8M0 scales.
pub fn resident_bytes(expert: &RoutedExpert) -> Result<u64> {
    [&expert.gate, &expert.up, &expert.down]
        .into_iter()
        .try_fold(0_u64, |total, matrix| {
            total
                .checked_add(packed_weight_bytes(matrix)?)
                .and_then(|bytes| bytes.checked_add(packed_scale_bytes(matrix).ok()?))
                .ok_or_else(|| ColicError::Usage("MXFP4 resident size overflows u64".into()))
        })
}

fn packed_weight_bytes(matrix: &Matrix) -> Result<u64> {
    let row_bytes = u64::from(matrix.columns).div_ceil(2);
    u64::from(matrix.rows)
        .checked_mul(row_bytes)
        .ok_or_else(|| ColicError::Usage("MXFP4 weight size overflows u64".into()))
}

fn packed_scale_bytes(matrix: &Matrix) -> Result<u64> {
    let groups = u64::from(matrix.columns).div_ceil(mxfp4::GROUP_SIZE as u64);
    u64::from(matrix.rows)
        .checked_mul(groups)
        .ok_or_else(|| ColicError::Usage("MXFP4 scale size overflows u64".into()))
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
            .map_err(|_| ColicError::Usage("MXFP4 expert layer exceeds COLI i32 range".into()))?,
    );
    put_i32(
        &mut payload,
        20,
        i32::try_from(expert)
            .map_err(|_| ColicError::Usage("MXFP4 expert id exceeds COLI i32 range".into()))?,
    );
    put_u16(&mut payload, 24, MATRIX_COUNT as u16);
    put_u32(&mut payload, 28, DESC_BYTES as u32);
    put_u64(&mut payload, 32, HEADER_BYTES as u64);
    put_u64(&mut payload, 40, DATA_OFFSET as u64);

    let roles = [1_u16, 2_u16, 3_u16];
    let mut resident = 0_u64;
    for (index, matrix) in matrices.into_iter().enumerate() {
        let weight_offset = append_aligned(&mut payload, &matrix.weights)?;
        let scale_offset = append_aligned(&mut payload, &matrix.scales)?;
        let desc = HEADER_BYTES + index * DESC_BYTES;
        put_u16(&mut payload, desc, roles[index]);
        put_u16(&mut payload, desc + 4, MATH_FORMAT_MXFP4);
        put_u16(&mut payload, desc + 6, SCALE_FORMAT_E8M0);
        put_u64(&mut payload, desc + 16, u64::from(matrix.rows));
        put_u64(&mut payload, desc + 24, u64::from(matrix.columns));
        put_u32(&mut payload, desc + 32, BLOCK_AXIS_COLUMNS);
        put_u32(&mut payload, desc + 36, BLOCK_SIZE);
        put_u64(&mut payload, desc + 48, weight_offset);
        put_u64(&mut payload, desc + 56, matrix.weights.len() as u64);
        put_u64(&mut payload, desc + 64, matrix.weights.len() as u64);
        put_u64(&mut payload, desc + 72, scale_offset);
        put_u64(&mut payload, desc + 80, matrix.scales.len() as u64);
        put_u64(&mut payload, desc + 88, matrix.scales.len() as u64);

        let mut logical = Vec::with_capacity(matrix.weights.len() + matrix.scales.len());
        logical.extend_from_slice(&matrix.weights);
        logical.extend_from_slice(&matrix.scales);
        put_u32(&mut payload, desc + 96, crc32c(&logical));
        resident = resident
            .checked_add(logical.len() as u64)
            .ok_or_else(|| ColicError::Usage("MXFP4 resident size overflows u64".into()))?;
    }
    put_u64(&mut payload, 48, resident);
    Ok(payload)
}

fn append_aligned(output: &mut Vec<u8>, bytes: &[u8]) -> Result<u64> {
    let offset = align_up(output.len() as u64, DATA_ALIGNMENT)?;
    let offset_usize = usize::try_from(offset)
        .map_err(|_| ColicError::Usage("MXFP4 record offset exceeds usize".into()))?;
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
    fn lowers_qwen_expert_into_existing_mxfp4_descriptor_contract() {
        let path =
            std::env::temp_dir().join(format!("colic-qwen-mxfp4-record-{}", std::process::id()));
        let gate_values = vec![1.0_f32; 64];
        let up_values = vec![2.0_f32; 64];
        let down_values = vec![3.0_f32; 64];
        let mut source = bf16_bytes(&gate_values);
        let up_offset = source.len() as u64;
        source.extend_from_slice(&bf16_bytes(&up_values));
        let down_offset = source.len() as u64;
        source.extend_from_slice(&bf16_bytes(&down_values));
        fs::write(&path, &source).unwrap();

        let expert = RoutedExpert {
            layer: 7,
            expert: 11,
            gate: matrix(&path, 0, 2, 32),
            up: matrix(&path, up_offset, 2, 32),
            down: matrix(&path, down_offset, 2, 32),
        };
        let bytes = lower_expert(&expert).unwrap();
        assert_eq!(&bytes[..8], b"COLIEXPT");
        assert_eq!(i32::from_le_bytes(bytes[16..20].try_into().unwrap()), 7);
        assert_eq!(i32::from_le_bytes(bytes[20..24].try_into().unwrap()), 11);
        assert_eq!(u64::from_le_bytes(bytes[48..56].try_into().unwrap()), 102);

        for index in 0..3 {
            let desc = HEADER_BYTES + index * DESC_BYTES;
            assert_eq!(
                u16::from_le_bytes(bytes[desc..desc + 2].try_into().unwrap()),
                (index + 1) as u16
            );
            assert_eq!(
                u16::from_le_bytes(bytes[desc + 4..desc + 6].try_into().unwrap()),
                MATH_FORMAT_MXFP4
            );
            assert_eq!(
                u16::from_le_bytes(bytes[desc + 6..desc + 8].try_into().unwrap()),
                SCALE_FORMAT_E8M0
            );
            assert_eq!(
                u64::from_le_bytes(bytes[desc + 16..desc + 24].try_into().unwrap()),
                2
            );
            assert_eq!(
                u64::from_le_bytes(bytes[desc + 24..desc + 32].try_into().unwrap()),
                32
            );
            assert_eq!(
                u32::from_le_bytes(bytes[desc + 32..desc + 36].try_into().unwrap()),
                BLOCK_AXIS_COLUMNS
            );
            assert_eq!(
                u32::from_le_bytes(bytes[desc + 36..desc + 40].try_into().unwrap()),
                BLOCK_SIZE
            );
            assert_eq!(
                u64::from_le_bytes(bytes[desc + 56..desc + 64].try_into().unwrap()),
                32
            );
            assert_eq!(
                u64::from_le_bytes(bytes[desc + 80..desc + 88].try_into().unwrap()),
                2
            );
        }
        assert_eq!(stored_bytes(&expert).unwrap(), bytes.len() as u64);
        assert_eq!(resident_bytes(&expert).unwrap(), 102);
        fs::remove_file(path).unwrap();
    }
}

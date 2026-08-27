//! COLIEXPT lowering for canonical 1D NVFP4 routed experts.
//!
//! Scale payload for each matrix is `f32 global_scale` followed by one raw
//! E4M3FN block-scale byte per 16 input columns per row. This is deliberately
//! distinct from MXFP4's E8M0/32-column contract.

use crate::{
    error::{ColicError, Result},
    ir::{Matrix, RoutedExpert},
    quant::nvfp4::{self, PackedMatrix},
    storage::{align_up, crc32c},
};

const HEADER_BYTES: usize = 64;
const DESC_BYTES: usize = 128;
const MATRIX_COUNT: usize = 3;
const DATA_OFFSET: usize = HEADER_BYTES + DESC_BYTES * MATRIX_COUNT;
const DATA_ALIGNMENT: u64 = 16;

/// CSF v1 extension IDs reserved by this compiler pass.
pub const MATH_FORMAT_NVFP4_E2M1: u16 = 0x23;
pub const SCALE_FORMAT_FP8_E4M3FN: u16 = 0x05;
pub const BLOCK_ROWS: u32 = 1;
pub const BLOCK_COLUMNS: u32 = nvfp4::GROUP_SIZE as u32;

pub fn lower_expert(expert: &RoutedExpert) -> Result<Vec<u8>> {
    let gate = nvfp4::quantize_matrix(&expert.gate)?;
    let up = nvfp4::quantize_matrix(&expert.up)?;
    let down = nvfp4::quantize_matrix(&expert.down)?;
    lower_packed_expert(expert.layer, expert.expert, [&gate, &up, &down])
}

pub fn stored_bytes(expert: &RoutedExpert) -> Result<u64> {
    [&expert.gate, &expert.up, &expert.down].into_iter().try_fold(DATA_OFFSET as u64, |cursor, matrix| {
        let after_weight = align_up(cursor, DATA_ALIGNMENT)?
            .checked_add(packed_weight_bytes(matrix)?)
            .ok_or_else(|| ColicError::Usage("NVFP4 expert size overflows u64".into()))?;
        align_up(after_weight, DATA_ALIGNMENT)?
            .checked_add(packed_scale_bytes(matrix)?)
            .ok_or_else(|| ColicError::Usage("NVFP4 expert size overflows u64".into()))
    })
}

pub fn resident_bytes(expert: &RoutedExpert) -> Result<u64> {
    [&expert.gate, &expert.up, &expert.down].into_iter().try_fold(0_u64, |total, matrix| {
        total.checked_add(packed_weight_bytes(matrix)?)
            .and_then(|v| v.checked_add(packed_scale_bytes(matrix).ok()?))
            .ok_or_else(|| ColicError::Usage("NVFP4 resident size overflows u64".into()))
    })
}

fn packed_weight_bytes(matrix: &Matrix) -> Result<u64> {
    u64::from(matrix.rows)
        .checked_mul(u64::from(matrix.columns).div_ceil(2))
        .ok_or_else(|| ColicError::Usage("NVFP4 weight size overflows u64".into()))
}

fn packed_scale_bytes(matrix: &Matrix) -> Result<u64> {
    let blocks = u64::from(matrix.rows)
        .checked_mul(u64::from(matrix.columns).div_ceil(nvfp4::GROUP_SIZE as u64))
        .ok_or_else(|| ColicError::Usage("NVFP4 scale size overflows u64".into()))?;
    blocks.checked_add(4).ok_or_else(|| ColicError::Usage("NVFP4 scale size overflows u64".into()))
}

fn lower_packed_expert(layer: u32, expert: u32, matrices: [&PackedMatrix; MATRIX_COUNT]) -> Result<Vec<u8>> {
    let mut payload = vec![0_u8; DATA_OFFSET];
    payload[..8].copy_from_slice(b"COLIEXPT");
    put_u16(&mut payload, 8, 1);
    put_u16(&mut payload, 10, 0);
    put_u32(&mut payload, 12, HEADER_BYTES as u32);
    put_i32(&mut payload, 16, i32::try_from(layer).map_err(|_| ColicError::Usage("NVFP4 expert layer exceeds COLI i32 range".into()))?);
    put_i32(&mut payload, 20, i32::try_from(expert).map_err(|_| ColicError::Usage("NVFP4 expert id exceeds COLI i32 range".into()))?);
    put_u16(&mut payload, 24, MATRIX_COUNT as u16);
    put_u32(&mut payload, 28, DESC_BYTES as u32);
    put_u64(&mut payload, 32, HEADER_BYTES as u64);
    put_u64(&mut payload, 40, DATA_OFFSET as u64);

    let roles = [1_u16, 2_u16, 3_u16];
    let mut resident = 0_u64;
    for (index, matrix) in matrices.into_iter().enumerate() {
        let scale_bytes = matrix.scale_bytes();
        let weight_offset = append_aligned(&mut payload, &matrix.weights)?;
        let scale_offset = append_aligned(&mut payload, &scale_bytes)?;
        let desc = HEADER_BYTES + index * DESC_BYTES;
        put_u16(&mut payload, desc, roles[index]);
        put_u16(&mut payload, desc + 4, MATH_FORMAT_NVFP4_E2M1);
        put_u16(&mut payload, desc + 6, SCALE_FORMAT_FP8_E4M3FN);
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

        let mut logical = Vec::with_capacity(matrix.weights.len() + scale_bytes.len());
        logical.extend_from_slice(&matrix.weights);
        logical.extend_from_slice(&scale_bytes);
        put_u32(&mut payload, desc + 96, crc32c(&logical));
        resident = resident.checked_add(logical.len() as u64)
            .ok_or_else(|| ColicError::Usage("NVFP4 resident size overflows u64".into()))?;
    }
    put_u64(&mut payload, 48, resident);
    Ok(payload)
}

fn append_aligned(output: &mut Vec<u8>, bytes: &[u8]) -> Result<u64> {
    let offset = align_up(output.len() as u64, DATA_ALIGNMENT)?;
    let offset_usize = usize::try_from(offset)
        .map_err(|_| ColicError::Usage("NVFP4 record offset exceeds usize".into()))?;
    output.resize(offset_usize, 0);
    output.extend_from_slice(bytes);
    Ok(offset)
}

fn put_u16(buffer: &mut [u8], offset: usize, value: u16) { buffer[offset..offset + 2].copy_from_slice(&value.to_le_bytes()); }
fn put_u32(buffer: &mut [u8], offset: usize, value: u32) { buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes()); }
fn put_u64(buffer: &mut [u8], offset: usize, value: u64) { buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes()); }
fn put_i32(buffer: &mut [u8], offset: usize, value: i32) { buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes()); }

#[cfg(test)]
mod tests {
    use std::fs;
    use super::*;
    use crate::{ir::Matrix, source::TensorRef};

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
    }
    fn matrix(path: &std::path::Path, offset: u64, rows: u32, columns: u32) -> Matrix {
        Matrix { source: TensorRef { source: path.to_owned(), offset, len: u64::from(rows) * u64::from(columns) * 2, dtype: "BF16".into(), shape: vec![u64::from(rows), u64::from(columns)] }, rows, columns, scale: None }
    }

    #[test]
    fn emits_distinct_nvfp4_geometry_and_scale_payload() {
        let path = std::env::temp_dir().join(format!("colic-nvfp4-record-{}", std::process::id()));
        let values = vec![1.0_f32; 32];
        let mut source = bf16_bytes(&values);
        let up = source.len() as u64; source.extend_from_slice(&bf16_bytes(&values));
        let down = source.len() as u64; source.extend_from_slice(&bf16_bytes(&values));
        fs::write(&path, &source).unwrap();
        let expert = RoutedExpert { layer: 1, expert: 3, gate: matrix(&path, 0, 1, 32), up: matrix(&path, up, 1, 32), down: matrix(&path, down, 1, 32) };
        let bytes = lower_expert(&expert).unwrap();
        for index in 0..3 {
            let desc = HEADER_BYTES + index * DESC_BYTES;
            assert_eq!(u16::from_le_bytes(bytes[desc + 4..desc + 6].try_into().unwrap()), MATH_FORMAT_NVFP4_E2M1);
            assert_eq!(u16::from_le_bytes(bytes[desc + 6..desc + 8].try_into().unwrap()), SCALE_FORMAT_FP8_E4M3FN);
            assert_eq!(u32::from_le_bytes(bytes[desc + 36..desc + 40].try_into().unwrap()), 16);
            assert_eq!(u64::from_le_bytes(bytes[desc + 56..desc + 64].try_into().unwrap()), 16);
            assert_eq!(u64::from_le_bytes(bytes[desc + 80..desc + 88].try_into().unwrap()), 6); // f32 global + 2 block scales
        }
        assert_eq!(stored_bytes(&expert).unwrap(), bytes.len() as u64);
        assert_eq!(resident_bytes(&expert).unwrap(), 66);
        fs::remove_file(path).unwrap();
    }
}

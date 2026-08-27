//! Deterministic BF16 -> grouped INT4 packing for offline target lowering.
//!
//! This is the canonical x86-friendly representation consumed by Colibri's
//! existing `matmul_i4_grouped` path: two biased signed nibbles per byte and
//! one little-endian f32 scale per 32 input columns.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use crate::{
    error::{ColicError, Result},
    ir::Matrix,
};

pub const GROUP_SIZE: usize = 32;
pub const VALUES_PER_BYTE: usize = 2;
pub const QMIN: i32 = -8;
pub const QMAX: i32 = 7;

#[derive(Debug, Clone, PartialEq)]
pub struct PackedMatrix {
    pub rows: u32,
    pub columns: u32,
    /// Row-major biased signed INT4. Even columns occupy the low nibble.
    /// Runtime decode is `(nibble as i32) - 8`.
    pub weights: Vec<u8>,
    /// Row-major f32 scales, one per 32-column group.
    pub scales: Vec<f32>,
}

impl PackedMatrix {
    pub fn row_bytes(&self) -> usize {
        (self.columns as usize).div_ceil(VALUES_PER_BYTE)
    }

    pub fn scales_per_row(&self) -> usize {
        (self.columns as usize).div_ceil(GROUP_SIZE)
    }

    pub fn scale_bytes_le(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.scales.len() * 4);
        for scale in &self.scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        bytes
    }
}

pub fn quantize_matrix(matrix: &Matrix) -> Result<PackedMatrix> {
    validate_source(matrix)?;
    let row_source_bytes = u64::from(matrix.columns)
        .checked_mul(2)
        .ok_or_else(|| ColicError::Usage("INT4 source row size overflows u64".into()))?;
    let row_bytes = usize::try_from(row_source_bytes)
        .map_err(|_| ColicError::Usage("INT4 row is too large for this host".into()))?;
    let packed_row_bytes = (matrix.columns as usize).div_ceil(VALUES_PER_BYTE);
    let scales_per_row = (matrix.columns as usize).div_ceil(GROUP_SIZE);
    let weight_capacity = (matrix.rows as usize)
        .checked_mul(packed_row_bytes)
        .ok_or_else(|| ColicError::Usage("INT4 packed matrix size overflows usize".into()))?;
    let scale_capacity = (matrix.rows as usize)
        .checked_mul(scales_per_row)
        .ok_or_else(|| ColicError::Usage("INT4 scale matrix size overflows usize".into()))?;

    let mut file = File::open(&matrix.source.source).map_err(|source| ColicError::Io {
        path: matrix.source.source.clone(),
        source,
    })?;
    file.seek(SeekFrom::Start(matrix.source.offset))
        .map_err(|source| ColicError::Io {
            path: matrix.source.source.clone(),
            source,
        })?;

    let mut weights = Vec::with_capacity(weight_capacity);
    let mut scales = Vec::with_capacity(scale_capacity);
    let mut source_row = vec![0_u8; row_bytes];
    if matrix.source.dtype == "F8_E4M3" {
        quantize_fp8_matrix(matrix, &mut file, &mut weights, &mut scales)?;
    } else {
        for _ in 0..matrix.rows {
            file.read_exact(&mut source_row)
                .map_err(|source| ColicError::Io {
                    path: matrix.source.source.clone(),
                    source,
                })?;
            quantize_bf16_row(&source_row, &mut weights, &mut scales)?;
        }
    }

    debug_assert_eq!(weights.len(), weight_capacity);
    debug_assert_eq!(scales.len(), scale_capacity);
    Ok(PackedMatrix {
        rows: matrix.rows,
        columns: matrix.columns,
        weights,
        scales,
    })
}

/// FP8 E4M3 block-scaled path (Qwen3.8-Next experts): read each row, dequant
/// through the per-128×128-block `weight_scale_inv` BF16 scales into BF16
/// bytes, then feed the same BF16→INT4 row quantizer.
fn quantize_fp8_matrix(
    matrix: &Matrix,
    file: &mut File,
    packed_weights: &mut Vec<u8>,
    scales: &mut Vec<f32>,
) -> Result<()> {
    let Some(scale_ref) = &matrix.scale else {
        return Err(ColicError::Usage(
            "F8_E4M3 source matrix is missing its weight_scale_inv block-scale tensor".into(),
        ));
    };
    if scale_ref.dtype != "BF16" {
        return Err(ColicError::Usage(format!(
            "weight_scale_inv dtype is {} (expected BF16)",
            scale_ref.dtype
        )));
    }
    let block_rows = matrix.rows.div_ceil(super::fp8::FP8_BLOCK as u32);
    let block_columns = matrix.columns.div_ceil(super::fp8::FP8_BLOCK as u32);
    let expected_scale_len = u64::from(block_rows) * u64::from(block_columns) * 2;
    if scale_ref.len != expected_scale_len {
        return Err(ColicError::Usage(format!(
            "weight_scale_inv is {} bytes, expected {expected_scale_len} for {block_rows}x{block_columns} blocks",
            scale_ref.len
        )));
    }
    let mut scale_file = File::open(&scale_ref.source).map_err(|source| ColicError::Io {
        path: scale_ref.source.clone(),
        source,
    })?;
    scale_file
        .seek(SeekFrom::Start(scale_ref.offset))
        .map_err(|source| ColicError::Io {
            path: scale_ref.source.clone(),
            source,
        })?;

    let fp8_row_bytes = matrix.columns as usize; // 1 byte per element
    let bf16_row_bytes = matrix.columns as usize * 2;
    let scale_row_bytes = block_columns as usize * 2;
    let mut fp8_row = vec![0_u8; fp8_row_bytes];
    let mut bf16_row = vec![0_u8; bf16_row_bytes];
    let mut scale_row = vec![0_u8; scale_row_bytes];
    for row in 0..matrix.rows {
        file.read_exact(&mut fp8_row)
            .map_err(|source| ColicError::Io {
                path: matrix.source.source.clone(),
                source,
            })?;
        scale_file
            .read_exact(&mut scale_row)
            .map_err(|source| ColicError::Io {
                path: scale_ref.source.clone(),
                source,
            })?;
        let block_row = row / super::fp8::FP8_BLOCK as u32;
        for column in 0..matrix.columns as usize {
            let block_col = column / super::fp8::FP8_BLOCK;
            let scale_bits = u16::from_le_bytes([
                scale_row[block_col * 2],
                scale_row[block_col * 2 + 1],
            ]);
            let scale = f32::from_bits(u32::from(scale_bits) << 16);
            let value = super::fp8::decode_e4m3(fp8_row[column]) * scale;
            if !value.is_finite() {
                return Err(ColicError::Usage(format!(
                    "FP8 dequant produced non-finite value at row {row} column {column}"
                )));
            }
            let bits = super::fp8::f32_to_bf16_bits(value);
            bf16_row[column * 2] = (bits & 0xff) as u8;
            bf16_row[column * 2 + 1] = (bits >> 8) as u8;
        }
        quantize_bf16_row(&bf16_row, packed_weights, scales)?;
    }
    Ok(())
}

fn validate_source(matrix: &Matrix) -> Result<()> {
    let element_bytes = match matrix.source.dtype.as_str() {
        "BF16" => 2_u64,
        "F8_E4M3" => 1,
        other => {
            return Err(ColicError::unsupported(
                "INT4 quantization",
                format!(
                    "matrix at {} has dtype `{other}`; grouped INT4 lowering requires BF16 or F8_E4M3 source weights",
                    matrix.source.source.display(),
                ),
            ));
        }
    };
    if matrix.scale.is_some() && matrix.source.dtype != "F8_E4M3" {
        return Err(ColicError::unsupported(
            "INT4 quantization",
            "pre-scaled matrices are not accepted by the BF16 -> grouped INT4 pass",
        ));
    }
    let row_bytes = u64::from(matrix.columns)
        .checked_mul(element_bytes)
        .ok_or_else(|| ColicError::Usage("INT4 source row size overflows u64".into()))?;
    let expected = u64::from(matrix.rows)
        .checked_mul(row_bytes)
        .ok_or_else(|| ColicError::Usage("INT4 source matrix size overflows u64".into()))?;
    if matrix.source.len != expected {
        return Err(ColicError::InvalidSource {
            path: matrix.source.source.clone(),
            detail: format!(
                "{} matrix payload is {} bytes, expected {expected} for {}x{}",
                matrix.source.dtype, matrix.source.len, matrix.rows, matrix.columns
            ),
        });
    }
    Ok(())
}

pub fn quantize_bf16_row(
    row_bytes: &[u8],
    packed_weights: &mut Vec<u8>,
    scales: &mut Vec<f32>,
) -> Result<()> {
    if !row_bytes.len().is_multiple_of(2) {
        return Err(ColicError::Usage(
            "BF16 row has an odd byte count during INT4 quantization".into(),
        ));
    }
    let columns = row_bytes.len() / 2;
    let mut nibbles = Vec::with_capacity(columns);
    let mut values = Vec::with_capacity(GROUP_SIZE);

    for group_start in (0..columns).step_by(GROUP_SIZE) {
        let group_end = (group_start + GROUP_SIZE).min(columns);
        values.clear();
        for column in group_start..group_end {
            let offset = column * 2;
            let bits = u16::from_le_bytes([row_bytes[offset], row_bytes[offset + 1]]);
            let value = f32::from_bits(u32::from(bits) << 16);
            if !value.is_finite() {
                return Err(ColicError::Usage(format!(
                    "INT4 quantization refuses non-finite BF16 value at column {column}"
                )));
            }
            values.push(value);
        }

        let scale = choose_scale(&values);
        scales.push(scale);
        for &value in &values {
            nibbles.push(quantize_value(value, scale));
        }
    }

    for pair in nibbles.chunks(2) {
        let low = pair[0] & 0x0f;
        // 8 decodes to zero; keep padding neutral for odd-width rows.
        let high = pair.get(1).copied().unwrap_or(8) & 0x0f;
        packed_weights.push(low | (high << 4));
    }
    Ok(())
}

fn choose_scale(values: &[f32]) -> f32 {
    let max_abs = values
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    if max_abs == 0.0 {
        1.0
    } else {
        // Symmetric PTQ deliberately uses [-7, 7] even though the storage
        // code has an extra -8 value. That avoids a positive-side clipping
        // bias and matches the existing `(nibble - 8) * scale` kernel.
        max_abs / QMAX as f32
    }
}

fn quantize_value(value: f32, scale: f32) -> u8 {
    let q = (value / scale).round().clamp(-(QMAX as f32), QMAX as f32) as i32;
    debug_assert!((QMIN..=QMAX).contains(&q));
    (q + 8) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        bytes
    }

    fn decode(code: u8, scale: f32) -> f32 {
        ((code as i32) - 8) as f32 * scale
    }

    #[test]
    fn packs_low_nibble_first_with_runtime_bias() {
        let values = [-7.0_f32, -1.0, 0.0, 1.0, 7.0];
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![1.0]);
        assert_eq!(weights, vec![0x71, 0x98, 0x8f]);
    }

    #[test]
    fn group_scale_restarts_every_32_values() {
        let mut values = vec![7.0_f32; GROUP_SIZE];
        values.extend_from_slice(&[70.0; GROUP_SIZE]);
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![1.0, 10.0]);
        assert_eq!(weights.len(), GROUP_SIZE);
    }

    #[test]
    fn roundtrip_is_bounded_by_half_a_step_away_from_clipping() {
        let values = [-3.25_f32, -1.0, 0.0, 2.1, 7.0];
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        let scale = scales[0];
        for (index, expected) in values.iter().copied().enumerate() {
            let byte = weights[index / 2];
            let code = if index & 1 == 0 { byte & 0x0f } else { byte >> 4 };
            let got = decode(code, scale);
            assert!((got - expected).abs() <= scale * 0.51);
        }
    }

    #[test]
    fn rejects_nonfinite_input() {
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        assert!(quantize_bf16_row(&bf16_bytes(&[f32::NAN]), &mut weights, &mut scales)
            .unwrap_err()
            .to_string()
            .contains("non-finite"));
    }

    #[test]
    fn fp8_block_scaled_matrix_roundtrips_through_int4() {
        use crate::source::TensorRef;
        // 1 row x 64 cols of constant value 1.0 in FP8 (code 0x38), block
        // scale 2.0. Dequant = 2.0 everywhere -> INT4 scale = 2/7, max code.
        let dir = std::env::temp_dir();
        let weight_path = dir.join(format!("colic-i4-fp8-w-{}", std::process::id()));
        let scale_path = dir.join(format!("colic-i4-fp8-s-{}", std::process::id()));
        std::fs::write(&weight_path, vec![0x38_u8; 64]).unwrap();
        let scale_bits = super::super::fp8::f32_to_bf16_bits(2.0) as u16;
        std::fs::write(&scale_path, scale_bits.to_le_bytes()).unwrap();
        let matrix = Matrix {
            source: TensorRef {
                source: weight_path.clone(),
                offset: 0,
                len: 64,
                dtype: "F8_E4M3".into(),
                shape: vec![1, 64],
            },
            rows: 1,
            columns: 64,
            scale: Some(TensorRef {
                source: scale_path.clone(),
                offset: 0,
                len: 2,
                dtype: "BF16".into(),
                shape: vec![1, 1],
            }),
        };
        let packed = quantize_matrix(&matrix).unwrap();
        assert_eq!(packed.scales, vec![2.0 / 7.0, 2.0 / 7.0]);
        for index in 0..4 {
            let byte = packed.weights[index / 2];
            let code = if index & 1 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            assert_eq!(
                (code as i32) - 8,
                7,
                "constant 2.0 must quantize to the max code"
            );
        }
        let _ = std::fs::remove_file(&weight_path);
        let _ = std::fs::remove_file(&scale_path);
    }
}

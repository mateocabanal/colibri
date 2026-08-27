//! Deterministic BF16 -> OCP MXFP4 packing for offline target lowering.
//!
//! The runtime already consumes MXFP4 as row-major E2M1 nibbles (low nibble
//! first) plus one raw E8M0 scale byte per 32 input values. This module owns
//! the compiler side of that contract so target lowering does not need to
//! rediscover or reimplement the quantization rules.

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
pub const MAX_E2M1: f32 = 6.0;

/// Positive E2M1 magnitudes. Bit 3 of the packed nibble is the sign bit.
pub const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedMatrix {
    pub rows: u32,
    pub columns: u32,
    /// Row-major E2M1 nibbles, two logical values per byte. The even column is
    /// stored in the low nibble and the odd column in the high nibble.
    pub weights: Vec<u8>,
    /// Row-major raw E8M0 scales, one byte per 32-column group.
    pub scales: Vec<u8>,
}

impl PackedMatrix {
    pub fn row_bytes(&self) -> usize {
        (self.columns as usize).div_ceil(VALUES_PER_BYTE)
    }

    pub fn scale_bytes_per_row(&self) -> usize {
        (self.columns as usize).div_ceil(GROUP_SIZE)
    }
}

/// Quantize one semantic BF16 matrix without materializing its safetensors
/// shard. The source tensor span is opened/seeks once and then consumed one
/// matrix row at a time, so hot-path I/O remains sequential.
pub fn quantize_matrix(matrix: &Matrix) -> Result<PackedMatrix> {
    if matrix.source.dtype != "BF16" {
        return Err(ColicError::unsupported(
            "MXFP4 quantization",
            format!(
                "matrix at {} has dtype `{}`; Qwen MXFP4 lowering currently requires BF16 source weights",
                matrix.source.source.display(),
                matrix.source.dtype
            ),
        ));
    }
    if matrix.scale.is_some() {
        return Err(ColicError::unsupported(
            "MXFP4 quantization",
            "pre-scaled matrices are not accepted by the BF16 -> MXFP4 pass",
        ));
    }

    let row_source_bytes = u64::from(matrix.columns)
        .checked_mul(2)
        .ok_or_else(|| ColicError::Usage("MXFP4 source row size overflows u64".into()))?;
    let expected = u64::from(matrix.rows)
        .checked_mul(row_source_bytes)
        .ok_or_else(|| ColicError::Usage("MXFP4 source matrix size overflows u64".into()))?;
    if matrix.source.len != expected {
        return Err(ColicError::InvalidSource {
            path: matrix.source.source.clone(),
            detail: format!(
                "BF16 matrix payload is {} bytes, expected {expected} for {}x{}",
                matrix.source.len, matrix.rows, matrix.columns
            ),
        });
    }

    let row_bytes = usize::try_from(row_source_bytes)
        .map_err(|_| ColicError::Usage("MXFP4 row is too large for this host".into()))?;
    let packed_row_bytes = (matrix.columns as usize).div_ceil(VALUES_PER_BYTE);
    let scale_row_bytes = (matrix.columns as usize).div_ceil(GROUP_SIZE);
    let weight_capacity = (matrix.rows as usize)
        .checked_mul(packed_row_bytes)
        .ok_or_else(|| ColicError::Usage("MXFP4 packed matrix size overflows usize".into()))?;
    let scale_capacity = (matrix.rows as usize)
        .checked_mul(scale_row_bytes)
        .ok_or_else(|| ColicError::Usage("MXFP4 scale matrix size overflows usize".into()))?;

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
    for _ in 0..matrix.rows {
        file.read_exact(&mut source_row)
            .map_err(|source| ColicError::Io {
                path: matrix.source.source.clone(),
                source,
            })?;
        quantize_bf16_row(&source_row, &mut weights, &mut scales)?;
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

/// Pack one little-endian BF16 row. This is public so target lowering can
/// stream directly into an expert record later without retaining all packed
/// weight bytes in memory.
pub fn quantize_bf16_row(
    row_bytes: &[u8],
    packed_weights: &mut Vec<u8>,
    scales: &mut Vec<u8>,
) -> Result<()> {
    if !row_bytes.len().is_multiple_of(2) {
        return Err(ColicError::Usage(
            "BF16 row has an odd byte count during MXFP4 quantization".into(),
        ));
    }
    let columns = row_bytes.len() / 2;
    let mut values = Vec::with_capacity(columns.min(GROUP_SIZE));
    let mut nibbles = Vec::with_capacity(columns);

    for group_start in (0..columns).step_by(GROUP_SIZE) {
        let group_end = (group_start + GROUP_SIZE).min(columns);
        values.clear();
        for column in group_start..group_end {
            let offset = column * 2;
            let bits = u16::from_le_bytes([row_bytes[offset], row_bytes[offset + 1]]);
            let value = f32::from_bits(u32::from(bits) << 16);
            if !value.is_finite() {
                return Err(ColicError::Usage(format!(
                    "MXFP4 quantization refuses non-finite BF16 value at column {column}"
                )));
            }
            values.push(value);
        }

        let (scale_code, scale) = choose_scale(&values);
        scales.push(scale_code);
        for &value in &values {
            nibbles.push(quantize_value(value, scale));
        }
    }

    for pair in nibbles.chunks(2) {
        let low = pair[0] & 0x0f;
        let high = pair.get(1).copied().unwrap_or(0) & 0x0f;
        packed_weights.push(low | (high << 4));
    }
    Ok(())
}

/// Choose the smallest runtime-supported power-of-two scale that can hold the
/// largest magnitude without E2M1 saturation. This is a deterministic PTQ
/// policy; OCP permits conversion algorithms other than its baseline recipe.
///
/// The existing kernels decode E8M0 through the Float32 exponent-bit fast path,
/// which is exact for codes 1..=254. Code 0 is a valid OCP E8M0 encoding for
/// 2^-127, but those kernels intentionally do not implement that denormal edge.
/// We therefore never emit code 0; an all-zero block uses scale 1.0 instead.
fn choose_scale(values: &[f32]) -> (u8, f32) {
    let max_abs = values
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    if max_abs == 0.0 {
        return (127, 1.0);
    }

    let bits = max_abs.to_bits();
    let biased = ((bits >> 23) & 0xff) as i32;
    let max_exp = if biased == 0 {
        // f32 subnormal: value = mantissa * 2^-149.
        let mantissa = bits & 0x007f_ffff;
        (31 - mantissa.leading_zeros() as i32) - 149
    } else {
        biased - 127
    };
    let mut scale_exp = max_exp - 2; // E2M1's largest power-of-two magnitude is 4.
    scale_exp = scale_exp.clamp(-126, 127);
    let mut scale_code = (scale_exp + 127) as u8;
    let mut scale = runtime_e8m0_to_f32(scale_code);

    // Values in [6*scale, 8*scale) need the next power-of-two scale to avoid
    // clipping at the E2M1 maximum magnitude 6.
    if max_abs > MAX_E2M1 * scale && scale_exp < 127 {
        scale_exp += 1;
        scale_code = (scale_exp + 127) as u8;
        scale = runtime_e8m0_to_f32(scale_code);
    }
    (scale_code, scale)
}

fn quantize_value(value: f32, scale: f32) -> u8 {
    let magnitude = (value.abs() / scale).min(MAX_E2M1);
    let mut best_code = 0_u8;
    let mut best_error = f32::INFINITY;
    for (code, candidate) in E2M1_MAGNITUDES.iter().copied().enumerate() {
        let error = (magnitude - candidate).abs();
        // E2M1's positive code parity tracks the low mantissa bit for the
        // adjacent representable values. Prefer even codes on an exact tie,
        // matching round-to-nearest-even at midpoint values.
        if error < best_error || (error == best_error && (code & 1) == 0 && (best_code & 1) != 0) {
            best_error = error;
            best_code = code as u8;
        }
    }
    if value.is_sign_negative() {
        best_code | 0x8
    } else {
        best_code
    }
}

#[inline]
pub fn runtime_e8m0_to_f32(code: u8) -> f32 {
    debug_assert!((1..=254).contains(&code));
    f32::from_bits(u32::from(code) << 23)
}

#[cfg(test)]
fn decode_nibble(code: u8, scale: u8) -> f32 {
    let magnitude = E2M1_MAGNITUDES[(code & 0x7) as usize];
    let signed = if code & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    };
    signed * runtime_e8m0_to_f32(scale)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::{ir::Matrix, source::TensorRef};

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            let bf16 = (value.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bf16.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn packs_all_e2m1_codes_low_nibble_first() {
        let values = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![127]);
        assert_eq!(
            weights,
            vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]
        );
    }

    #[test]
    fn zero_block_uses_finite_nonzero_e8m0_scale() {
        let values = [0.0_f32; GROUP_SIZE];
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![127]);
        assert!(weights.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn scale_is_power_of_two_and_avoids_saturation() {
        let values = [7.0_f32; GROUP_SIZE];
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        quantize_bf16_row(&bf16_bytes(&values), &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![128]); // 2^(128-127) = 2
        let code = weights[0] & 0x0f;
        assert_eq!(decode_nibble(code, scales[0]), 8.0);
    }

    #[test]
    fn scale_groups_restart_for_every_row() {
        let path = std::env::temp_dir().join(format!("colic-mxfp4-{}", std::process::id()));
        let mut source = Vec::new();
        source.extend_from_slice(&bf16_bytes(&[1.0; 33]));
        source.extend_from_slice(&bf16_bytes(&[16.0; 33]));
        fs::write(&path, &source).unwrap();
        let matrix = Matrix {
            source: TensorRef {
                source: path.clone(),
                offset: 0,
                len: source.len() as u64,
                dtype: "BF16".into(),
                shape: vec![2, 33],
            },
            rows: 2,
            columns: 33,
            scale: None,
        };
        let packed = quantize_matrix(&matrix).unwrap();
        assert_eq!(packed.row_bytes(), 17);
        assert_eq!(packed.scale_bytes_per_row(), 2);
        assert_eq!(packed.weights.len(), 34);
        assert_eq!(packed.scales, vec![125, 125, 129, 129]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn sliced_matrix_starts_at_tensor_offset() {
        let path = std::env::temp_dir().join(format!("colic-mxfp4-offset-{}", std::process::id()));
        let prefix = bf16_bytes(&[64.0; 32]);
        let wanted = bf16_bytes(&[1.0; 32]);
        let mut source = prefix.clone();
        source.extend_from_slice(&wanted);
        fs::write(&path, &source).unwrap();
        let matrix = Matrix {
            source: TensorRef {
                source: path.clone(),
                offset: prefix.len() as u64,
                len: wanted.len() as u64,
                dtype: "BF16".into(),
                shape: vec![1, 32],
            },
            rows: 1,
            columns: 32,
            scale: None,
        };
        let packed = quantize_matrix(&matrix).unwrap();
        assert_eq!(packed.scales, vec![125]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_nonfinite_bf16_source() {
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        let error = quantize_bf16_row(&bf16_bytes(&[f32::INFINITY]), &mut weights, &mut scales)
            .unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }
}

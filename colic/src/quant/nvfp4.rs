//! Deterministic BF16 -> canonical 1D NVFP4 packing.
//!
//! NVFP4 is E2M1 data with hierarchical scaling:
//! `x ~= e2m1 * e4m3_block_scale * f32_global_scale`.
//! This canonical compiler form uses one E4M3 scale per 16 consecutive input
//! columns. Target-specific Blackwell/Metal/x86 lowering may repack it later.

use std::{fs::File, io::{Read, Seek, SeekFrom}};

use crate::{error::{ColicError, Result}, ir::Matrix};

pub const GROUP_SIZE: usize = 16;
pub const VALUES_PER_BYTE: usize = 2;
pub const MAX_E2M1: f32 = 6.0;
pub const MAX_E4M3: f32 = 448.0;
pub const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[derive(Debug, Clone, PartialEq)]
pub struct PackedMatrix {
    pub rows: u32,
    pub columns: u32,
    pub weights: Vec<u8>,
    pub block_scales: Vec<u8>,
    pub global_scale: f32,
}

impl PackedMatrix {
    pub fn row_bytes(&self) -> usize { (self.columns as usize).div_ceil(2) }
    pub fn scales_per_row(&self) -> usize { (self.columns as usize).div_ceil(GROUP_SIZE) }
    /// Scale payload ABI: little-endian f32 global scale followed by E4M3 block scales.
    pub fn scale_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.block_scales.len());
        out.extend_from_slice(&self.global_scale.to_le_bytes());
        out.extend_from_slice(&self.block_scales);
        out
    }
}

pub fn quantize_matrix(matrix: &Matrix) -> Result<PackedMatrix> {
    validate_source(matrix)?;
    let mut file = File::open(&matrix.source.source).map_err(|source| ColicError::Io {
        path: matrix.source.source.clone(), source,
    })?;
    file.seek(SeekFrom::Start(matrix.source.offset)).map_err(|source| ColicError::Io {
        path: matrix.source.source.clone(), source,
    })?;
    let bytes_len = usize::try_from(matrix.source.len)
        .map_err(|_| ColicError::Usage("NVFP4 source matrix is too large for this host".into()))?;
    let mut source = vec![0_u8; bytes_len];
    file.read_exact(&mut source).map_err(|source_error| ColicError::Io {
        path: matrix.source.source.clone(), source: source_error,
    })?;

    let global_amax = source.chunks_exact(2).enumerate().try_fold(0.0_f32, |amax, (i, pair)| {
        let value = bf16_pair(pair);
        if !value.is_finite() {
            return Err(ColicError::Usage(format!("NVFP4 quantization refuses non-finite BF16 value at element {i}")));
        }
        Ok(amax.max(value.abs()))
    })?;
    let global_scale = if global_amax == 0.0 { 1.0 } else { global_amax / (MAX_E4M3 * MAX_E2M1) };

    let row_source_bytes = matrix.columns as usize * 2;
    let packed_row_bytes = (matrix.columns as usize).div_ceil(2);
    let scale_row_bytes = (matrix.columns as usize).div_ceil(GROUP_SIZE);
    let mut weights = Vec::with_capacity(matrix.rows as usize * packed_row_bytes);
    let mut block_scales = Vec::with_capacity(matrix.rows as usize * scale_row_bytes);
    for row in source.chunks_exact(row_source_bytes) {
        quantize_bf16_row(row, global_scale, &mut weights, &mut block_scales)?;
    }
    Ok(PackedMatrix { rows: matrix.rows, columns: matrix.columns, weights, block_scales, global_scale })
}

fn validate_source(matrix: &Matrix) -> Result<()> {
    if matrix.source.dtype != "BF16" {
        return Err(ColicError::unsupported("NVFP4 quantization", format!(
            "matrix at {} has dtype `{}`; NVFP4 lowering currently requires BF16 source weights",
            matrix.source.source.display(), matrix.source.dtype)));
    }
    if matrix.scale.is_some() {
        return Err(ColicError::unsupported("NVFP4 quantization",
            "pre-scaled matrices are not accepted by the BF16 -> NVFP4 pass"));
    }
    let expected = u64::from(matrix.rows)
        .checked_mul(u64::from(matrix.columns)).and_then(|v| v.checked_mul(2))
        .ok_or_else(|| ColicError::Usage("NVFP4 source matrix size overflows u64".into()))?;
    if matrix.source.len != expected {
        return Err(ColicError::InvalidSource { path: matrix.source.source.clone(), detail: format!(
            "BF16 matrix payload is {} bytes, expected {expected} for {}x{}",
            matrix.source.len, matrix.rows, matrix.columns) });
    }
    Ok(())
}

pub fn quantize_bf16_row(row: &[u8], global_scale: f32, weights: &mut Vec<u8>, scales: &mut Vec<u8>) -> Result<()> {
    if !row.len().is_multiple_of(2) || !(global_scale > 0.0 && global_scale.is_finite()) {
        return Err(ColicError::Usage("invalid BF16 row/global scale during NVFP4 quantization".into()));
    }
    let columns = row.len() / 2;
    let mut nibbles = Vec::with_capacity(columns);
    for start in (0..columns).step_by(GROUP_SIZE) {
        let end = (start + GROUP_SIZE).min(columns);
        let mut values = Vec::with_capacity(end - start);
        let mut amax = 0.0_f32;
        for column in start..end {
            let value = bf16_pair(&row[column * 2..column * 2 + 2]);
            if !value.is_finite() {
                return Err(ColicError::Usage(format!("NVFP4 quantization refuses non-finite BF16 value at column {column}")));
            }
            amax = amax.max(value.abs());
            values.push(value);
        }
        if amax == 0.0 {
            scales.push(0);
            nibbles.extend(std::iter::repeat_n(0, values.len()));
            continue;
        }
        let wanted = ((amax / MAX_E2M1) / global_scale).min(MAX_E4M3);
        let scale_code = encode_positive_e4m3(wanted).max(1);
        let block_scale = decode_positive_e4m3(scale_code);
        scales.push(scale_code);
        let combined = global_scale * block_scale;
        for value in values { nibbles.push(quantize_e2m1(value, combined)); }
    }
    for pair in nibbles.chunks(2) {
        weights.push((pair[0] & 0x0f) | ((pair.get(1).copied().unwrap_or(0) & 0x0f) << 4));
    }
    Ok(())
}

fn bf16_pair(pair: &[u8]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16)
}

fn quantize_e2m1(value: f32, scale: f32) -> u8 {
    let magnitude = (value.abs() / scale).min(MAX_E2M1);
    let mut best = 0_u8;
    let mut error = f32::INFINITY;
    for (code, candidate) in E2M1_MAGNITUDES.iter().copied().enumerate() {
        let next = (magnitude - candidate).abs();
        if next < error || (next == error && code & 1 == 0 && best & 1 != 0) {
            best = code as u8; error = next;
        }
    }
    if value.is_sign_negative() { best | 8 } else { best }
}

pub fn decode_positive_e4m3(code: u8) -> f32 {
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 0x07;
    if exponent == 0 {
        (mantissa as f32) * 2.0_f32.powi(-9)
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
    }
}

fn encode_positive_e4m3(value: f32) -> u8 {
    if value <= 0.0 { return 0; }
    let target = value.min(MAX_E4M3);
    let mut best = 1_u8;
    let mut best_error = f32::INFINITY;
    // 0x7f is NaN in E4M3FN; 0x7e is max finite 448.
    for code in 1_u8..=0x7e {
        let candidate = decode_positive_e4m3(code);
        let error = (candidate - target).abs();
        if error < best_error || (error == best_error && code & 1 == 0 && best & 1 != 0) {
            best = code; best_error = error;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bf16(values: &[f32]) -> Vec<u8> { values.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect() }

    #[test]
    fn e4m3_known_values_match_fn_encoding() {
        assert_eq!(encode_positive_e4m3(1.0), 0x38);
        assert_eq!(encode_positive_e4m3(448.0), 0x7e);
        assert_eq!(decode_positive_e4m3(0x38), 1.0);
        assert_eq!(decode_positive_e4m3(0x7e), 448.0);
    }

    #[test]
    fn uses_sixteen_value_blocks_and_global_scale() {
        let mut values = vec![6.0_f32; 16]; values.extend_from_slice(&[2688.0; 16]);
        let global = 1.0;
        let mut weights = Vec::new(); let mut scales = Vec::new();
        quantize_bf16_row(&bf16(&values), global, &mut weights, &mut scales).unwrap();
        assert_eq!(scales.len(), 2);
        assert_eq!(scales[0], 0x38); // 1
        assert_eq!(scales[1], 0x7e); // 448
        assert_eq!(weights.len(), 16);
    }

    #[test]
    fn all_zero_block_has_zero_scale_and_zero_codes() {
        let mut weights = Vec::new(); let mut scales = Vec::new();
        quantize_bf16_row(&bf16(&[0.0; 16]), 1.0, &mut weights, &mut scales).unwrap();
        assert_eq!(scales, vec![0]);
        assert!(weights.iter().all(|v| *v == 0));
    }
}

//! Deterministic FP8 E4M3 -> f32 decode for compiler-side requantization.
//!
//! The Qwen3.8-Flash-Next checkpoint stores routed experts and the PLE n-gram
//! table as F8_E4M3 payloads with per-128×128-block BF16 `weight_scale_inv`
//! scales (block-quantized, `dequant = e4m3 * scale_inv[block]`). colic reads
//! them, dequantizes block-wise, and feeds the existing BF16->INT4/MXFP4
//! quantizers — one code path, no duplicated packers.

use crate::error::{ColicError, Result};

/// E4M3 (FN, no inf) LUT: value = mantissa_lut[exp][code & 0x7] with sign.
/// Built once at first use via the same bit-exact construction the C runtime
/// uses (mant × 2^(exp-7), subnormals = mant/8 × 2^-6).
pub const E4M3_VALUES: [f32; 256] = build_e4m3();

const fn build_e4m3() -> [f32; 256] {
    let mut table = [0.0_f32; 256];
    let mut code = 0usize;
    while code < 256 {
        let sign = if code & 0x80 != 0 { -1.0_f32 } else { 1.0 };
        let exp = ((code >> 3) & 0x0f) as i32;
        let mant = (code & 0x07) as i32;
        let value = if exp == 0 {
            if mant == 0 {
                0.0
            } else {
                // Subnormal: mant/8 * 2^-6.
                (mant as f32) * 0.125 * 0.015625
            }
        } else if exp == 15 && mant == 7 {
            // E4M3FN saturates at 448 instead of inf/nan.
            f32::MAX // never matched exactly; handled by caller clamp
        } else {
            // Normal: (1 + mant/8) * 2^(exp-7).
            (1.0 + (mant as f32) * 0.125) * pow2(exp - 7)
        };
        table[code] = sign * value;
        code += 1;
    }
    table
}

const fn pow2(exp: i32) -> f32 {
    // Const power of two via bit construction: 2^e for e in [-126, 127].
    f32::from_bits((((exp + 127) as u32) << 23) as u32)
}

/// Decode one E4M3 byte (FN variant: NaN only at 0x7F/0xFF, saturates 448).
pub fn decode_e4m3(code: u8) -> f32 {
    let exp = (code >> 3) & 0x0f;
    let mant = code & 0x07;
    if exp == 15 && mant == 7 {
        // 0x7F / 0xFF are NaN in E4M3FN; reject rather than guess.
        f32::NAN
    } else {
        E4M3_VALUES[code as usize]
    }
}

/// Block scale layout: `scale_inv[block_row][block_col]`, BF16, one entry per
/// 128×128 block of the source matrix.
pub const FP8_BLOCK: usize = 128;

#[derive(Debug, Clone)]
pub struct Fp8Matrix {
    pub rows: u32,
    pub columns: u32,
    /// Row-major E4M3 payload.
    pub payload: Vec<u8>,
    /// BF16 `weight_scale_inv` entries, row-major over the block grid.
    pub scale_inv_bf16: Vec<u16>,
    pub block_rows: u32,
    pub block_columns: u32,
}

impl Fp8Matrix {
    /// Dequantize into a BF16 byte buffer (the quantizers' native input).
    pub fn dequantize_to_bf16(&self) -> Result<Vec<u8>> {
        let rows = self.rows as usize;
        let columns = self.columns as usize;
        let block_columns = self.block_columns as usize;
        let mut out = Vec::with_capacity(rows * columns * 2);
        for row in 0..rows {
            let block_row = row / FP8_BLOCK;
            for column in 0..columns {
                let block_col = column / FP8_BLOCK;
                let scale_bits = self.scale_inv_bf16[block_row * block_columns + block_col];
                let scale = f32::from_bits(u32::from(scale_bits) << 16);
                let code = self.payload[row * columns + column];
                let value = decode_e4m3(code);
                if !value.is_finite() {
                    return Err(ColicError::Usage(format!(
                        "FP8 matrix contains NaN E4M3 code at ({row},{column})"
                    )));
                }
                let scaled = value * scale;
                let bits = (f32_to_bf16_bits(scaled) as u16).to_le_bytes();
                out.push(bits[0]);
                out.push(bits[1]);
            }
        }
        Ok(out)
    }
}

/// Round-to-nearest-even f32 -> bf16 bits (matches the C runtime's view).
pub fn f32_to_bf16_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    // Round half to even: add 0x7FFF + lsb before truncation.
    (bits + 0x7FFF + lsb) >> 16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_known_values_decode() {
        // 0x38 = 1.0, 0xB8 = -1.0, 0x30 = 0.5, 0x36 = 0.875, 0x07 = subnormal 7/8*2^-6.
        assert_eq!(decode_e4m3(0x38), 1.0);
        assert_eq!(decode_e4m3(0xB8), -1.0);
        assert_eq!(decode_e4m3(0x30), 0.5);
        assert_eq!(decode_e4m3(0x36), 0.875);
        assert_eq!(decode_e4m3(0x07), 0.013671875);
        assert_eq!(decode_e4m3(0x00), 0.0);
        assert_eq!(decode_e4m3(0x80), -0.0);
        assert_eq!(decode_e4m3(0x7E), 448.0);
    }

    #[test]
    fn nan_code_is_detected() {
        assert!(decode_e4m3(0x7F).is_nan());
        assert!(decode_e4m3(0xFF).is_nan());
    }

    #[test]
    fn block_dequant_multiplies_correct_scale() {
        // 1 row x 256 cols, two 128-col blocks with scales 1.0 and 2.0.
        let matrix = Fp8Matrix {
            rows: 1,
            columns: 256,
            payload: vec![0x38; 256],
            scale_inv_bf16: vec![
                f32_to_bf16_bits(1.0) as u16,
                f32_to_bf16_bits(2.0) as u16,
            ],
            block_rows: 1,
            block_columns: 2,
        };
        let bf16 = matrix.dequantize_to_bf16().unwrap();
        let first = f32::from_bits(u32::from(u16::from_le_bytes([bf16[0], bf16[1]])) << 16);
        let second = f32::from_bits(
            u32::from(u16::from_le_bytes([bf16[256], bf16[257]])) << 16,
        );
        assert_eq!(first, 1.0);
        assert_eq!(second, 2.0);
    }
}
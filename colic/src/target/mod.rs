//! Versioned target compatibility profiles and lowering boundary.

use std::io::Write;

use crate::{
    error::{ColicError, Result},
    ir::RoutedExpert,
    pipeline::TargetRequest,
    source,
    storage::{align_up, crc32c},
};

const TENSOR_HEADER_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetProfile {
    pub id: u32,
    pub name: &'static str,
    pub operating_system: &'static str,
    pub architecture: &'static str,
    pub backend: Backend,
    pub execution_layout_abi: u32,
    pub kernel_abi: u32,
    pub record_alignment: u64,
    pub preferred_io_granularity: u64,
}

pub const MACOS_ARM64_METAL_APPLE8_V1: TargetProfile = TargetProfile {
    id: 1,
    name: "macos-arm64-metal-apple8-v1",
    operating_system: "macos",
    architecture: "aarch64",
    backend: Backend::Metal,
    execution_layout_abi: 1,
    kernel_abi: 1,
    record_alignment: 16 * 1024,
    preferred_io_granularity: 16 * 1024,
};

pub const LINUX_X86_64_AVX2_V1: TargetProfile = TargetProfile {
    id: 2,
    name: "linux-x86_64-avx2-v1",
    operating_system: "linux",
    architecture: "x86_64",
    backend: Backend::Cpu,
    execution_layout_abi: 1,
    kernel_abi: 1,
    record_alignment: 4096,
    preferred_io_granularity: 4096,
};

pub const PROFILES: &[TargetProfile] = &[MACOS_ARM64_METAL_APPLE8_V1, LINUX_X86_64_AVX2_V1];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCapabilities {
    pub operating_system: &'static str,
    pub architecture: &'static str,
    pub avx2: bool,
}

impl HostCapabilities {
    pub fn current() -> Self {
        Self {
            operating_system: std::env::consts::OS,
            architecture: std::env::consts::ARCH,
            avx2: current_has_avx2(),
        }
    }
}

pub fn resolve(request: &TargetRequest, host: HostCapabilities) -> Result<TargetProfile> {
    match request {
        TargetRequest::Native => native(host),
        TargetRequest::Profile(name) => PROFILES
            .iter()
            .find(|profile| profile.name == name)
            .copied()
            .ok_or_else(|| {
                ColicError::Usage(format!("unknown or unsupported target profile `{name}`"))
            }),
    }
}

fn native(host: HostCapabilities) -> Result<TargetProfile> {
    if host.operating_system == "macos" && host.architecture == "aarch64" {
        return Ok(MACOS_ARM64_METAL_APPLE8_V1);
    }
    if host.operating_system == "linux" && host.architecture == "x86_64" && host.avx2 {
        return Ok(LINUX_X86_64_AVX2_V1);
    }
    Err(ColicError::unsupported(
        "target detection",
        format!(
            "no target profile supports {} / {} with these detected capabilities",
            host.operating_system, host.architecture
        ),
    ))
}

#[cfg(target_arch = "x86_64")]
fn current_has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}
#[cfg(not(target_arch = "x86_64"))]
fn current_has_avx2() -> bool {
    false
}

/// The physical lowering interface. Backends receive semantic roles, never HF names.
pub trait TargetBackend {
    fn profile(&self) -> TargetProfile;
}

/// Initial exact expert lowering. It preserves the source matrix and scale
/// bytes but packages them into the frozen COLIEXPT envelope for later target
/// layout substitutions under a new ABI/profile version.
pub fn lower_exact_expert(expert: &RoutedExpert) -> Result<Vec<u8>> {
    const HEADER_BYTES: usize = 64;
    const DESC_BYTES: usize = 128;
    const TABLE_BYTES: usize = DESC_BYTES * 3;
    const DATA_OFFSET: usize = HEADER_BYTES + TABLE_BYTES;
    let matrices = [
        (&expert.gate, 1_u16),
        (&expert.up, 2_u16),
        (&expert.down, 3_u16),
    ];
    let mut payload = vec![0_u8; DATA_OFFSET];
    payload[0..8].copy_from_slice(b"COLIEXPT");
    put_u16(&mut payload, 8, 1);
    put_u16(&mut payload, 10, 0);
    put_u32(&mut payload, 12, HEADER_BYTES as u32);
    put_i32(&mut payload, 16, expert.layer as i32);
    put_i32(&mut payload, 20, expert.expert as i32);
    put_u16(&mut payload, 24, 3);
    put_u32(&mut payload, 28, DESC_BYTES as u32);
    put_u64(&mut payload, 32, HEADER_BYTES as u64);
    put_u64(&mut payload, 40, DATA_OFFSET as u64);
    let mut logical = Vec::new();
    for (index, (matrix, role)) in matrices.into_iter().enumerate() {
        let weight = read_tensor(&matrix.source)?;
        let (scale, scale_format) = match &matrix.scale {
            Some(scale) => (read_tensor(scale)?, scale_format(&scale.dtype)?),
            None => (Vec::new(), 0),
        };
        let weight_offset = append_aligned(&mut payload, &weight)?;
        let scale_offset = if scale.is_empty() {
            0
        } else {
            append_aligned(&mut payload, &scale)?
        };
        let desc = HEADER_BYTES + index * DESC_BYTES;
        put_u16(&mut payload, desc, role);
        put_u16(
            &mut payload,
            desc + 4,
            expert_math_format(&matrix.source.dtype)?,
        );
        put_u16(&mut payload, desc + 6, scale_format);
        put_u64(&mut payload, desc + 16, matrix.rows as u64);
        put_u64(&mut payload, desc + 24, matrix.columns as u64);
        if matrix.source.dtype == "I8" {
            put_u32(&mut payload, desc + 32, 1);
            put_u32(&mut payload, desc + 36, 32);
        }
        put_u64(&mut payload, desc + 48, weight_offset);
        put_u64(&mut payload, desc + 56, weight.len() as u64);
        put_u64(&mut payload, desc + 64, weight.len() as u64);
        put_u64(&mut payload, desc + 72, scale_offset);
        put_u64(&mut payload, desc + 80, scale.len() as u64);
        put_u64(&mut payload, desc + 88, scale.len() as u64);
        let mut matrix_logical = weight.clone();
        matrix_logical.extend_from_slice(&scale);
        put_u32(&mut payload, desc + 96, crc32c(&matrix_logical));
        logical.extend_from_slice(&matrix_logical);
    }
    put_u64(&mut payload, 48, logical.len() as u64);
    Ok(payload)
}

pub fn exact_expert_stored_bytes(expert: &RoutedExpert) -> Result<u64> {
    let mut bytes = 64_u64 + 128 * 3;
    for matrix in [&expert.gate, &expert.up, &expert.down] {
        bytes = align_up(bytes, 16)?
            .checked_add(matrix.source.len)
            .ok_or_else(|| {
                ColicError::Usage("projected expert payload size overflows u64".into())
            })?;
        if let Some(scale) = &matrix.scale {
            bytes = align_up(bytes, 16)?.checked_add(scale.len).ok_or_else(|| {
                ColicError::Usage("projected expert payload size overflows u64".into())
            })?;
        }
    }
    Ok(bytes)
}

pub fn exact_expert_decoded_bytes(expert: &RoutedExpert) -> Result<u64> {
    [&expert.gate, &expert.up, &expert.down]
        .into_iter()
        .try_fold(0_u64, |total, matrix| {
            let scale_bytes = matrix.scale.as_ref().map_or(0, |scale| scale.len);
            total
                .checked_add(matrix.source.len)
                .and_then(|bytes| bytes.checked_add(scale_bytes))
                .ok_or_else(|| ColicError::Usage("expert logical byte count overflows u64".into()))
        })
}

/// Losslessly wraps one source tensor in the frozen COLITENS envelope.
/// Static model roles stay independently addressable and retain canonical bytes.
pub fn lower_exact_tensor(tensor: &source::TensorRef) -> Result<Vec<u8>> {
    if tensor.shape.len() > 8 {
        return Err(ColicError::unsupported(
            "exact tensor lowering",
            format!("rank {} exceeds the COLI v1 limit", tensor.shape.len()),
        ));
    }
    // The envelope carries raw canonical bytes, but the outer record will need
    // this registered format when it is added to the manifest.
    let _ = math_format_for_dtype(&tensor.dtype)?;
    let data = read_tensor(tensor)?;
    let mut payload = vec![0_u8; TENSOR_HEADER_BYTES];
    payload[0..8].copy_from_slice(b"COLITENS");
    put_u16(&mut payload, 8, 1);
    put_u32(&mut payload, 12, TENSOR_HEADER_BYTES as u32);
    put_u16(&mut payload, 16, tensor.shape.len() as u16);
    for (index, dimension) in tensor.shape.iter().enumerate() {
        put_u64(&mut payload, 32 + index * 8, *dimension);
    }
    put_u64(&mut payload, 96, TENSOR_HEADER_BYTES as u64);
    put_u64(&mut payload, 104, data.len() as u64);
    put_u64(&mut payload, 112, data.len() as u64);
    put_u32(&mut payload, 120, crc32c(&data));
    payload.extend_from_slice(&data);
    Ok(payload)
}

pub fn exact_tensor_stored_bytes(tensor: &source::TensorRef) -> Result<u64> {
    if tensor.shape.len() > 8 {
        return Err(ColicError::unsupported(
            "exact tensor lowering",
            format!("rank {} exceeds the COLI v1 limit", tensor.shape.len()),
        ));
    }
    let _ = math_format_for_dtype(&tensor.dtype)?;
    (TENSOR_HEADER_BYTES as u64)
        .checked_add(tensor.len)
        .ok_or_else(|| ColicError::Usage("projected tensor payload size overflows u64".into()))
}

/// Writes an exact tensor envelope without materialising its payload. The
/// source is read once for its CRC and once for emission; memory stays bounded
/// by the 8 MiB transfer buffer.
pub fn stream_exact_tensor(
    tensor: &source::TensorRef,
    output: &mut dyn Write,
) -> Result<(u32, u32)> {
    exact_tensor_stored_bytes(tensor)?;
    let logical_crc32c = tensor_crc32c(tensor)?;
    let mut header = [0_u8; TENSOR_HEADER_BYTES];
    header[0..8].copy_from_slice(b"COLITENS");
    put_u16(&mut header, 8, 1);
    put_u32(&mut header, 12, TENSOR_HEADER_BYTES as u32);
    put_u16(&mut header, 16, tensor.shape.len() as u16);
    for (index, dimension) in tensor.shape.iter().enumerate() {
        put_u64(&mut header, 32 + index * 8, *dimension);
    }
    put_u64(&mut header, 96, TENSOR_HEADER_BYTES as u64);
    put_u64(&mut header, 104, tensor.len);
    put_u64(&mut header, 112, tensor.len);
    put_u32(&mut header, 120, logical_crc32c);
    output.write_all(&header).map_err(|source| ColicError::Io {
        path: tensor.source.clone(),
        source,
    })?;
    let mut stored_crc = crc32c_state(!0_u32, &header);
    let mut offset = 0_u64;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    while offset < tensor.len {
        let count = (tensor.len - offset).min(buffer.len() as u64) as usize;
        source::read_range(tensor, offset..offset + count as u64, &mut buffer[..count])?;
        output
            .write_all(&buffer[..count])
            .map_err(|source| ColicError::Io {
                path: tensor.source.clone(),
                source,
            })?;
        stored_crc = crc32c_state(stored_crc, &buffer[..count]);
        offset += count as u64;
    }
    Ok((logical_crc32c, !stored_crc))
}

fn tensor_crc32c(tensor: &source::TensorRef) -> Result<u32> {
    let mut crc = !0_u32;
    let mut offset = 0_u64;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    while offset < tensor.len {
        let count = (tensor.len - offset).min(buffer.len() as u64) as usize;
        source::read_range(tensor, offset..offset + count as u64, &mut buffer[..count])?;
        crc = crc32c_state(crc, &buffer[..count]);
        offset += count as u64;
    }
    Ok(!crc)
}

fn crc32c_state(mut crc: u32, bytes: &[u8]) -> u32 {
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0x82f6_3b78 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    crc
}

fn append_aligned(output: &mut Vec<u8>, bytes: &[u8]) -> Result<u64> {
    let offset = align_up(output.len() as u64, 16)?;
    output.resize(offset as usize, 0);
    output.extend_from_slice(bytes);
    Ok(offset)
}
fn read_tensor(tensor: &source::TensorRef) -> Result<Vec<u8>> {
    let mut bytes = vec![
        0;
        tensor.len.try_into().map_err(|_| ColicError::Usage(
            "tensor is too large for the current record-lowering address space".into()
        ))?
    ];
    source::read_range(tensor, 0..tensor.len, &mut bytes)?;
    Ok(bytes)
}
pub fn math_format_for_dtype(dtype: &str) -> Result<u16> {
    match dtype {
        "F32" => Ok(1),
        "F16" => Ok(2),
        "BF16" => Ok(3),
        "U8" => Ok(5),
        // UE8M0 is registered as a scale format, not a standalone math format;
        // standalone static scale tensors retain its byte representation as U8.
        "F8_E8M0" | "F8_E8M0FNU" => Ok(5),
        "I64" => Ok(0x0a),
        "I8" => Ok(0x20),
        "F8_E4M3" | "F8_E4M3FN" => Ok(0x10),
        "F8_E5M2" => Ok(0x11),
        _ => Err(ColicError::unsupported(
            "exact expert lowering",
            format!("unsupported matrix dtype `{dtype}`"),
        )),
    }
}
fn expert_math_format(dtype: &str) -> Result<u16> {
    match dtype {
        "I8" => Ok(0x20),
        other => math_format_for_dtype(other),
    }
}
fn scale_format(dtype: &str) -> Result<u16> {
    match dtype {
        "F32" => Ok(1),
        "F16" => Ok(2),
        "BF16" => Ok(3),
        "U8" | "F8_E8M0" | "F8_E8M0FNU" => Ok(4),
        _ => Err(ColicError::unsupported(
            "exact expert lowering",
            format!("unsupported scale dtype `{dtype}`"),
        )),
    }
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
    use super::*;
    use crate::{ir::Matrix, source::TensorRef};

    #[test]
    fn native_apple_is_versioned_metal_profile() {
        assert_eq!(
            resolve(
                &TargetRequest::Native,
                HostCapabilities {
                    operating_system: "macos",
                    architecture: "aarch64",
                    avx2: false
                }
            )
            .unwrap(),
            MACOS_ARM64_METAL_APPLE8_V1
        );
    }

    #[test]
    fn native_linux_requires_avx2() {
        assert!(
            resolve(
                &TargetRequest::Native,
                HostCapabilities {
                    operating_system: "linux",
                    architecture: "x86_64",
                    avx2: false
                }
            )
            .is_err()
        );
        assert_eq!(
            resolve(
                &TargetRequest::Native,
                HostCapabilities {
                    operating_system: "linux",
                    architecture: "x86_64",
                    avx2: true
                }
            )
            .unwrap(),
            LINUX_X86_64_AVX2_V1
        );
    }

    #[test]
    fn profile_names_are_explicit_and_no_portable_profile_exists() {
        assert_eq!(
            resolve(
                &TargetRequest::Profile("macos-arm64-metal-apple8-v1".into()),
                HostCapabilities::current()
            )
            .unwrap()
            .id,
            1
        );
        assert!(
            resolve(
                &TargetRequest::Profile("portable-v1".into()),
                HostCapabilities::current()
            )
            .is_err()
        );
    }

    #[test]
    fn exact_expert_lowering_emits_envelope_and_preserves_matrix_bytes() {
        let path = std::env::temp_dir().join(format!("colic-expert-{}", std::process::id()));
        std::fs::write(&path, [10_u8, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61]).unwrap();
        let matrix = |offset: u64, dtype: &str| Matrix {
            source: TensorRef {
                source: path.clone(),
                offset,
                len: 2,
                dtype: dtype.into(),
                shape: vec![1, 2],
            },
            rows: 1,
            columns: 2,
            scale: Some(TensorRef {
                source: path.clone(),
                offset: offset + 2,
                len: 2,
                dtype: "F8_E8M0".into(),
                shape: vec![1, 1],
            }),
        };
        let expert = RoutedExpert {
            layer: 4,
            expert: 2,
            gate: matrix(0, "F8_E4M3FN"),
            up: matrix(4, "F8_E4M3FN"),
            down: matrix(8, "F8_E4M3FN"),
        };
        let bytes = lower_exact_expert(&expert).unwrap();
        assert_eq!(
            exact_expert_stored_bytes(&expert).unwrap() as usize,
            bytes.len()
        );
        assert_eq!(&bytes[0..8], b"COLIEXPT");
        assert_eq!(i32::from_le_bytes(bytes[16..20].try_into().unwrap()), 4);
        assert_eq!(u16::from_le_bytes(bytes[64..66].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(bytes[70..72].try_into().unwrap()), 4);
        let gate_weight_offset = u64::from_le_bytes(bytes[112..120].try_into().unwrap()) as usize;
        let gate_scale_offset = u64::from_le_bytes(bytes[136..144].try_into().unwrap()) as usize;
        assert_eq!(
            &bytes[gate_weight_offset..gate_weight_offset + 2],
            &[10, 11]
        );
        assert_eq!(&bytes[gate_scale_offset..gate_scale_offset + 2], &[20, 21]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn packed_mxfp4_expert_uses_logical_shape_and_block_metadata() {
        let path = std::env::temp_dir().join(format!("colic-mxfp4-expert-{}", std::process::id()));
        std::fs::write(&path, [10_u8, 20, 30, 40, 50, 60]).unwrap();
        let matrix = |offset| Matrix {
            source: TensorRef {
                source: path.clone(),
                offset,
                len: 1,
                dtype: "I8".into(),
                shape: vec![1, 1],
            },
            rows: 1,
            columns: 2,
            scale: Some(TensorRef {
                source: path.clone(),
                offset: offset + 1,
                len: 1,
                dtype: "F8_E8M0".into(),
                shape: vec![1, 1],
            }),
        };
        let expert = RoutedExpert {
            layer: 0,
            expert: 0,
            gate: matrix(0),
            up: matrix(2),
            down: matrix(4),
        };
        let bytes = lower_exact_expert(&expert).unwrap();
        assert_eq!(u16::from_le_bytes(bytes[68..70].try_into().unwrap()), 0x20);
        assert_eq!(u64::from_le_bytes(bytes[88..96].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[96..100].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(bytes[100..104].try_into().unwrap()), 32);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn exact_tensor_lowering_emits_envelope_and_preserves_bytes() {
        let path = std::env::temp_dir().join(format!("colic-tensor-{}", std::process::id()));
        std::fs::write(&path, [0x2a_u8, 0x3b]).unwrap();
        let tensor = TensorRef {
            source: path.clone(),
            offset: 0,
            len: 2,
            dtype: "U8".into(),
            shape: vec![1, 2],
        };
        let bytes = lower_exact_tensor(&tensor).unwrap();
        assert_eq!(
            exact_tensor_stored_bytes(&tensor).unwrap() as usize,
            bytes.len()
        );
        assert_eq!(&bytes[..8], b"COLITENS");
        assert_eq!(u16::from_le_bytes(bytes[16..18].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(bytes[96..104].try_into().unwrap()), 128);
        assert_eq!(&bytes[128..], &[0x2a, 0x3b]);
        std::fs::remove_file(path).unwrap();
    }
}

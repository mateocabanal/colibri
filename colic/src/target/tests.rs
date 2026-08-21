use super::*;
use crate::{ir::Matrix, source::TensorRef};

#[test]
fn crc32c_combine_matches_a_contiguous_stream() {
    let left = b"a record header";
    let right = b" and its payload";
    let mut joined = left.to_vec();
    joined.extend_from_slice(right);
    assert_eq!(
        crc32c_combine(crc32c(left), crc32c(right), right.len() as u64),
        crc32c(&joined)
    );
}

#[test]
fn native_apple_refuses_false_row_artifact() {
    let error = resolve(
        &TargetRequest::Native,
        HostCapabilities {
            operating_system: "macos",
            architecture: "aarch64",
            avx2: false,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("production lowerer is not implemented"));
}

#[test]
fn native_linux_requires_avx2() {
    assert!(resolve(
        &TargetRequest::Native,
        HostCapabilities {
            operating_system: "linux",
            architecture: "x86_64",
            avx2: false,
        },
    )
    .is_err());
    assert_eq!(
        resolve(
            &TargetRequest::Native,
            HostCapabilities {
                operating_system: "linux",
                architecture: "x86_64",
                avx2: true,
            },
        )
        .unwrap(),
        LINUX_X86_64_AVX2_V1
    );
}

#[test]
fn profile_identity_is_registry_owned() {
    assert_eq!(MACOS_ARM64_METAL_APPLE8_V1.id, target_registry::APPLE8_PROFILE_ID);
    assert_eq!(MACOS_ARM64_METAL_APPLE8_V1.name, target_registry::APPLE8_PROFILE_NAME);
    assert_eq!(MACOS_ARM64_METAL_APPLE8_V1.target_profile_abi, target_registry::APPLE8_TARGET_PROFILE_ABI);
    assert_eq!(MACOS_ARM64_METAL_APPLE8_V1.execution_layout_abi, target_registry::APPLE8_EXECUTION_LAYOUT_ABI);
    assert_eq!(MACOS_ARM64_METAL_APPLE8_V1.kernel_abi, target_registry::APPLE8_KERNEL_ABI);
    assert_eq!(MACOS_ARM64_METAL_APPLE8_V1.target_class, target_registry::APPLE8_TARGET_CLASS);
    assert!(resolve(
        &TargetRequest::Profile(target_registry::APPLE8_PROFILE_NAME.into()),
        HostCapabilities::current(),
    )
    .is_err());
    assert!(resolve(
        &TargetRequest::Profile("portable-v1".into()),
        HostCapabilities::current(),
    )
    .is_err());
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
    let mut streamed = std::io::Cursor::new(Vec::new());
    let streamed_crc = stream_exact_expert(&expert, &mut streamed).unwrap();
    assert_eq!(streamed.into_inner(), bytes);
    assert_eq!(streamed_crc, crc32c(&bytes));
    assert_eq!(exact_expert_stored_bytes(&expert).unwrap() as usize, bytes.len());
    assert_eq!(&bytes[0..8], b"COLIEXPT");
    assert_eq!(i32::from_le_bytes(bytes[16..20].try_into().unwrap()), 4);
    assert_eq!(u16::from_le_bytes(bytes[64..66].try_into().unwrap()), 1);
    assert_eq!(u16::from_le_bytes(bytes[70..72].try_into().unwrap()), 4);
    let gate_weight_offset = u64::from_le_bytes(bytes[112..120].try_into().unwrap()) as usize;
    let gate_scale_offset = u64::from_le_bytes(bytes[136..144].try_into().unwrap()) as usize;
    assert_eq!(&bytes[gate_weight_offset..gate_weight_offset + 2], &[10, 11]);
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
    let mut streamed = std::io::Cursor::new(Vec::new());
    let (logical_crc, stored_crc) = stream_exact_tensor(&tensor, &mut streamed).unwrap();
    assert_eq!(streamed.into_inner(), bytes);
    assert_eq!(logical_crc, crc32c(&bytes[128..]));
    assert_eq!(stored_crc, crc32c(&bytes));
    assert_eq!(exact_tensor_stored_bytes(&tensor).unwrap() as usize, bytes.len());
    assert_eq!(&bytes[..8], b"COLITENS");
    assert_eq!(u16::from_le_bytes(bytes[16..18].try_into().unwrap()), 2);
    assert_eq!(u64::from_le_bytes(bytes[96..104].try_into().unwrap()), 128);
    assert_eq!(&bytes[128..], &[0x2a, 0x3b]);
    std::fs::remove_file(path).unwrap();
}

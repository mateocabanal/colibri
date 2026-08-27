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
fn native_apple_resolves_production_lowerer() {
    assert_eq!(
        resolve(
            &TargetRequest::Native,
            HostCapabilities {
                operating_system: "macos",
                architecture: "aarch64",
                avx2: false,
            },
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
                avx2: false,
            },
        )
        .is_err()
    );
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
    assert_eq!(
        MACOS_ARM64_METAL_APPLE8_V1.id,
        target_registry::APPLE8_PROFILE_ID
    );
    assert_eq!(
        MACOS_ARM64_METAL_APPLE8_V1.name,
        target_registry::APPLE8_PROFILE_NAME
    );
    assert_eq!(
        MACOS_ARM64_METAL_APPLE8_V1.target_profile_abi,
        target_registry::APPLE8_TARGET_PROFILE_ABI
    );
    assert_eq!(
        MACOS_ARM64_METAL_APPLE8_V1.execution_layout_abi,
        target_registry::APPLE8_EXECUTION_LAYOUT_ABI
    );
    assert_eq!(
        MACOS_ARM64_METAL_APPLE8_V1.kernel_abi,
        target_registry::APPLE8_KERNEL_ABI
    );
    assert_eq!(
        MACOS_ARM64_METAL_APPLE8_V1.target_class,
        target_registry::APPLE8_TARGET_CLASS
    );
    assert_eq!(
        resolve(
            &TargetRequest::Profile(target_registry::APPLE8_PROFILE_NAME.into()),
            HostCapabilities::current(),
        )
        .unwrap(),
        MACOS_ARM64_METAL_APPLE8_V1
    );
    assert!(
        resolve(
            &TargetRequest::Profile("portable-v1".into()),
            HostCapabilities::current(),
        )
        .is_err()
    );
}

#[test]
fn apple8_tile_sizes_match_frozen_contract() {
    for (rows, columns, expected) in [
        (1, 1, 136),
        (1, 31, 136),
        (1, 32, 136),
        (1, 33, 272),
        (7, 32, 136),
        (8, 32, 136),
        (9, 32, 272),
        (8, 31, 136),
        (8, 33, 272),
        (9, 33, 544),
    ] {
        assert_eq!(apple8_tile_bytes(rows, columns).unwrap(), expected);
    }
    assert!(apple8_tile_bytes(0, 32).is_err());
}

#[test]
fn apple8_exact_mxfp4_repack_matches_frozen_tile_order() {
    let path = std::env::temp_dir().join(format!("colic-apple8-exact-{}", std::process::id()));
    let rows = 2_u32;
    let columns = 33_u32;
    let row_bytes = columns.div_ceil(2) as usize;
    let groups = columns.div_ceil(32) as usize;
    let mut bytes = Vec::new();
    let mut offsets = Vec::new();
    for role in 0..3_u8 {
        let weight_offset = bytes.len() as u64;
        let weights = (0..rows as usize * row_bytes)
            .map(|index| role.wrapping_mul(40).wrapping_add(index as u8 + 1))
            .collect::<Vec<_>>();
        bytes.extend_from_slice(&weights);
        let scale_offset = bytes.len() as u64;
        let scales = (0..rows as usize * groups)
            .map(|index| 120_u8.wrapping_add(role * 8).wrapping_add(index as u8))
            .collect::<Vec<_>>();
        bytes.extend_from_slice(&scales);
        offsets.push((weight_offset, scale_offset));
    }
    std::fs::write(&path, &bytes).unwrap();
    let matrix = |index: usize| Matrix {
        source: TensorRef {
            source: path.clone(),
            offset: offsets[index].0,
            len: (rows as usize * row_bytes) as u64,
            dtype: "I8".into(),
            shape: vec![rows as u64, row_bytes as u64],
        },
        rows,
        columns,
        scale: Some(TensorRef {
            source: path.clone(),
            offset: offsets[index].1,
            len: (rows as usize * groups) as u64,
            dtype: "F8_E8M0".into(),
            shape: vec![rows as u64, groups as u64],
        }),
    };
    let expert = RoutedExpert {
        layer: 7,
        expert: 11,
        gate: matrix(0),
        up: matrix(1),
        down: matrix(2),
    };
    let lowered = lower_apple8_exact_mxfp4_expert(&expert).unwrap();
    assert_eq!(
        lowered.len() as u64,
        apple8_expert_stored_bytes(&expert).unwrap()
    );
    assert_eq!(
        u64::from_le_bytes(lowered[48..56].try_into().unwrap()),
        3 * 272
    );
    for (index, &(weight_offset, scale_offset)) in offsets.iter().enumerate() {
        let desc = 64 + index * 128;
        assert_eq!(
            u16::from_le_bytes(lowered[desc + 12..desc + 14].try_into().unwrap()),
            target_registry::APPLE8_MXFP4_TILE_LAYOUT
        );
        assert_eq!(
            u32::from_le_bytes(lowered[desc + 32..desc + 36].try_into().unwrap()),
            1
        );
        assert_eq!(
            u32::from_le_bytes(lowered[desc + 36..desc + 40].try_into().unwrap()),
            32
        );
        assert_eq!(
            u64::from_le_bytes(lowered[desc + 72..desc + 80].try_into().unwrap()),
            0
        );
        assert_eq!(
            u64::from_le_bytes(lowered[desc + 80..desc + 88].try_into().unwrap()),
            0
        );
        assert_eq!(
            u64::from_le_bytes(lowered[desc + 88..desc + 96].try_into().unwrap()),
            0
        );
        let matrix_offset =
            u64::from_le_bytes(lowered[desc + 48..desc + 56].try_into().unwrap()) as usize;
        let source_weight = weight_offset as usize;
        let source_scale = scale_offset as usize;
        assert_eq!(
            &lowered[matrix_offset..matrix_offset + 16],
            &bytes[source_weight..source_weight + 16]
        );
        assert_eq!(lowered[matrix_offset + 128], bytes[source_scale]);
        assert_eq!(lowered[matrix_offset + 136], bytes[source_weight + 16]);
        assert_eq!(lowered[matrix_offset + 136 + 128], bytes[source_scale + 1]);
        assert!(
            lowered[matrix_offset + 137..matrix_offset + 136 + 16]
                .iter()
                .all(|byte| *byte == 0)
        );
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn apple8_exact_rejects_fp8_source() {
    let path = std::env::temp_dir().join(format!("colic-apple8-fp8-{}", std::process::id()));
    std::fs::write(&path, [0_u8; 6]).unwrap();
    let matrix = Matrix {
        source: TensorRef {
            source: path.clone(),
            offset: 0,
            len: 2,
            dtype: "F8_E4M3FN".into(),
            shape: vec![1, 2],
        },
        rows: 1,
        columns: 2,
        scale: Some(TensorRef {
            source: path.clone(),
            offset: 2,
            len: 1,
            dtype: "F8_E8M0".into(),
            shape: vec![1, 1],
        }),
    };
    let expert = RoutedExpert {
        layer: 0,
        expert: 0,
        gate: matrix.clone(),
        up: matrix.clone(),
        down: matrix,
    };
    assert!(validate_apple8_exact_mxfp4_expert(&expert).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn apple8_compiler_quantized_mxfp4_emits_production_layout() {
    let path = std::env::temp_dir().join(format!("colic-apple8-quant-{}", std::process::id()));
    let mut bytes = Vec::new();
    for value in 0..96_u16 {
        let bf16 = ((value as f32 / 16.0).to_bits() >> 16) as u16;
        bytes.extend_from_slice(&bf16.to_le_bytes());
    }
    std::fs::write(&path, &bytes).unwrap();
    let matrix = |offset: u64| Matrix {
        source: TensorRef {
            source: path.clone(),
            offset,
            len: 64,
            dtype: "BF16".into(),
            shape: vec![1, 32],
        },
        rows: 1,
        columns: 32,
        scale: None,
    };
    let expert = RoutedExpert {
        layer: 2,
        expert: 3,
        gate: matrix(0),
        up: matrix(64),
        down: matrix(128),
    };
    let lowered = lower_apple8_quantized_mxfp4_expert(&expert).unwrap();
    assert_eq!(lowered.len() as u64, 872);
    assert_eq!(u64::from_le_bytes(lowered[48..56].try_into().unwrap()), 408);
    for index in 0..3 {
        let desc = 64 + index * 128;
        assert_eq!(
            u16::from_le_bytes(lowered[desc + 4..desc + 6].try_into().unwrap()),
            0x20
        );
        assert_eq!(
            u16::from_le_bytes(lowered[desc + 6..desc + 8].try_into().unwrap()),
            4
        );
        assert_eq!(
            u16::from_le_bytes(lowered[desc + 12..desc + 14].try_into().unwrap()),
            0x0103
        );
        assert_eq!(
            u64::from_le_bytes(lowered[desc + 56..desc + 64].try_into().unwrap()),
            136
        );
    }
    std::fs::remove_file(path).unwrap();
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
    let mut streamed = std::io::Cursor::new(Vec::new());
    let (logical_crc, stored_crc) = stream_exact_tensor(&tensor, &mut streamed).unwrap();
    assert_eq!(streamed.into_inner(), bytes);
    assert_eq!(logical_crc, crc32c(&bytes[128..]));
    assert_eq!(stored_crc, crc32c(&bytes));
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

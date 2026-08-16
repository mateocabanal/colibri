use crate::{error::{ColicError, Result}, ir::{MathFormat, RoutedExpert}, target::TargetProfile};

pub const SEMANTIC_ABI: &str = "deepseek-v4-exec-v1";
pub const TARGET_PROFILE_ABI: u32 = 1;
pub const EXECUTION_LAYOUT_ABI: u32 = 1;
pub const KERNEL_ABI: u32 = 1;

pub const APPLE_LINEAR_ROW_MAJOR_V1: u16 = 0x0101;
pub const APPLE_MXFP4_ROW32_V1: u16 = 0x0102;
pub const X86_LINEAR_ROW_MAJOR_V1: u16 = 0x0201;
pub const X86_MXFP4_ROW32_V1: u16 = 0x0202;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetIdentity {
    pub target_os: u16,
    pub target_arch: u16,
    pub backend: u16,
    pub gpu_kind: u16,
    pub flags: u32,
    pub cpu_feature_mask: u64,
    pub gpu_family_min: u32,
    pub gpu_family_max: u32,
    pub gpu_capability_min: u32,
    pub gpu_capability_max: u32,
    pub target_profile_abi: u32,
    pub execution_layout_abi: u32,
    pub kernel_abi_min: u32,
    pub kernel_abi_max: u32,
    pub record_alignment: u32,
    pub io_granularity: u32,
    pub resident_alignment: u32,
    pub required_runtime_features: u64,
    pub kernel_profile: &'static str,
    pub target_triple: &'static str,
    pub semantic_abi: &'static str,
    pub linear_layout: u16,
    pub mxfp4_layout: u16,
}

pub fn for_profile(profile: TargetProfile) -> Result<TargetIdentity> {
    match profile.name {
        "macos-arm64-metal-apple8-v1" => Ok(TargetIdentity {
            target_os: 1,
            target_arch: 1,
            backend: 2,
            gpu_kind: 1,
            flags: (1 << 1) | (1 << 2) | (1 << 3),
            cpu_feature_mask: 1 << 0,
            gpu_family_min: 8,
            gpu_family_max: 0,
            gpu_capability_min: 0,
            gpu_capability_max: 0,
            target_profile_abi: TARGET_PROFILE_ABI,
            execution_layout_abi: EXECUTION_LAYOUT_ABI,
            kernel_abi_min: KERNEL_ABI,
            kernel_abi_max: KERNEL_ABI,
            record_alignment: 16 * 1024,
            io_granularity: 16 * 1024,
            resident_alignment: 16 * 1024,
            required_runtime_features: (1 << 0) | (1 << 1),
            kernel_profile: "deepseek-v4-metal-mxfp4-v1",
            target_triple: "arm64-apple-macos",
            semantic_abi: SEMANTIC_ABI,
            linear_layout: APPLE_LINEAR_ROW_MAJOR_V1,
            mxfp4_layout: APPLE_MXFP4_ROW32_V1,
        }),
        "linux-x86_64-cpu-avx2-v1" | "linux-x86_64-avx2-v1" => Ok(TargetIdentity {
            target_os: 2,
            target_arch: 2,
            backend: 1,
            gpu_kind: 0,
            flags: 1 << 2,
            cpu_feature_mask: (1 << 1) | (1 << 2),
            gpu_family_min: 0,
            gpu_family_max: 0,
            gpu_capability_min: 0,
            gpu_capability_max: 0,
            target_profile_abi: TARGET_PROFILE_ABI,
            execution_layout_abi: EXECUTION_LAYOUT_ABI,
            kernel_abi_min: KERNEL_ABI,
            kernel_abi_max: KERNEL_ABI,
            record_alignment: 4096,
            io_granularity: 4096,
            resident_alignment: 64,
            required_runtime_features: 0,
            kernel_profile: "deepseek-v4-cpu-mxfp4-v1",
            target_triple: "x86_64-linux-gnu",
            semantic_abi: SEMANTIC_ABI,
            linear_layout: X86_LINEAR_ROW_MAJOR_V1,
            mxfp4_layout: X86_MXFP4_ROW32_V1,
        }),
        other => Err(ColicError::unsupported(
            "target identity",
            format!("target profile `{other}` has no CSF v1.1 identity"),
        )),
    }
}

pub fn expert_layout(profile: TargetProfile, expert: &RoutedExpert) -> Result<u16> {
    let identity = for_profile(profile)?;
    for matrix in [&expert.gate, &expert.up, &expert.down] {
        if matrix.quantization.math_format != MathFormat::MxFp4E2M1
            || matrix.quantization.scale_block_rows != 1
            || matrix.quantization.scale_block_columns != 32
        {
            return Err(ColicError::unsupported(
                "target lowering",
                "target-profile ABI v1 requires routed experts with exact MXFP4_E2M1/UE8M0 row32 semantics",
            ));
        }
    }
    Ok(identity.mxfp4_layout)
}

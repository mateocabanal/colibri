//! Versioned target compatibility profiles and lowering boundary.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

use crate::{
    error::{ColicError, Result},
    ir::RoutedExpert,
    pipeline::TargetRequest,
    source,
    storage::{align_up, crc32c},
    target_registry,
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
    pub target_profile_abi: u32,
    pub execution_layout_abi: u32,
    pub kernel_abi: u32,
    pub target_class: u32,
    pub compiler_emission_supported: bool,
    pub record_alignment: u64,
    pub preferred_io_granularity: u64,
}

pub const MACOS_ARM64_METAL_APPLE8_V1: TargetProfile = TargetProfile {
    id: target_registry::MACOS_ARM64_METAL_APPLE8_V1.profile_id,
    name: target_registry::MACOS_ARM64_METAL_APPLE8_V1.name,
    operating_system: target_registry::MACOS_ARM64_METAL_APPLE8_V1.operating_system,
    architecture: target_registry::MACOS_ARM64_METAL_APPLE8_V1.architecture,
    backend: Backend::Metal,
    target_profile_abi: target_registry::MACOS_ARM64_METAL_APPLE8_V1.target_profile_abi,
    execution_layout_abi: target_registry::MACOS_ARM64_METAL_APPLE8_V1.execution_layout_abi,
    kernel_abi: target_registry::MACOS_ARM64_METAL_APPLE8_V1.kernel_abi,
    target_class: target_registry::MACOS_ARM64_METAL_APPLE8_V1.target_class,
    compiler_emission_supported: target_registry::MACOS_ARM64_METAL_APPLE8_V1
        .compiler_emission_supported,
    record_alignment: target_registry::MACOS_ARM64_METAL_APPLE8_V1.record_alignment,
    preferred_io_granularity: target_registry::MACOS_ARM64_METAL_APPLE8_V1.io_granularity,
};

pub const LINUX_X86_64_AVX2_V1: TargetProfile = TargetProfile {
    id: target_registry::LINUX_X86_64_AVX2_V1.profile_id,
    name: target_registry::LINUX_X86_64_AVX2_V1.name,
    operating_system: target_registry::LINUX_X86_64_AVX2_V1.operating_system,
    architecture: target_registry::LINUX_X86_64_AVX2_V1.architecture,
    backend: Backend::Cpu,
    target_profile_abi: target_registry::LINUX_X86_64_AVX2_V1.target_profile_abi,
    execution_layout_abi: target_registry::LINUX_X86_64_AVX2_V1.execution_layout_abi,
    kernel_abi: target_registry::LINUX_X86_64_AVX2_V1.kernel_abi,
    target_class: target_registry::LINUX_X86_64_AVX2_V1.target_class,
    compiler_emission_supported: target_registry::LINUX_X86_64_AVX2_V1.compiler_emission_supported,
    record_alignment: target_registry::LINUX_X86_64_AVX2_V1.record_alignment,
    preferred_io_granularity: target_registry::LINUX_X86_64_AVX2_V1.io_granularity,
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
    let profile = match request {
        TargetRequest::Native => native(host)?,
        TargetRequest::Profile(name) => PROFILES
            .iter()
            .find(|profile| profile.name == name)
            .copied()
            .ok_or_else(|| {
                ColicError::Usage(format!("unknown or unsupported target profile `{name}`"))
            })?,
    };
    if !profile.compiler_emission_supported {
        return Err(ColicError::unsupported(
            "target planning",
            format!(
                "target profile `{}` is frozen to execution layout 0x{:04x}, but its production lowerer is not implemented; refusing to emit a row/canonical artifact under that profile",
                profile.name,
                target_registry::APPLE8_MXFP4_TILE_LAYOUT
            ),
        ));
    }
    Ok(profile)
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

pub trait TargetBackend {
    fn profile(&self) -> TargetProfile;
}

include!("lowering_expert.rs");
include!("lowering_tensor.rs");
include!("lowering_apple8.rs");

/// Test/reference seam used to compare the Rust production packer against the
/// independently compiled C oracle. This does not participate in compilation.
#[doc(hidden)]
pub fn apple8_repack_reference_input(
    matrix: &crate::quant::mxfp4::PackedMatrix,
) -> Result<Vec<u8>> {
    repack_apple8_matrix(matrix)
}

#[cfg(test)]
mod tests;

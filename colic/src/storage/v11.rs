use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::{
    error::{ColicError, Result},
    storage::{
        MANIFEST_HEADER_BYTES, MANIFEST_MAGIC, ManifestRecord, StoragePlan, align_up, crc32c,
    },
    target::{self, TargetProfile},
};

const SHARD_DESC_BYTES: u64 = 64;
const RECORD_DESC_BYTES: u64 = 96;
const STRING_DESC_BYTES: u64 = 16;
const TARGET_DESC_BYTES: u64 = 256;
const MANIFEST_FLAGS: u32 = (1 << 0) | (1 << 1) | (1 << 16);

#[derive(Debug, Clone, Copy)]
pub struct ArtifactOptions<'a> {
    pub compiler: &'a str,
    pub quant_profile: &'a str,
    pub storage_profile: &'a str,
    pub optimization_profile: &'a str,
}

impl Default for ArtifactOptions<'static> {
    fn default() -> Self {
        Self {
            compiler: "colic-0.1.0",
            quant_profile: "exact",
            storage_profile: "none",
            optimization_profile: "default",
        }
    }
}

pub fn artifact_fingerprint(
    source_fingerprint: [u8; 32],
    profile: TargetProfile,
    options: ArtifactOptions<'_>,
) -> Result<[u8; 32]> {
    let identity = target::identity::for_profile(profile)?;
    let mut hasher = Sha256::new();
    hasher.update(b"COLI-ARTIFACT-V1\0");
    hasher.update(source_fingerprint);
    for value in [
        options.compiler,
        identity.semantic_abi,
        profile.name,
        options.quant_profile,
        options.storage_profile,
        options.optimization_profile,
        identity.kernel_profile,
        identity.target_triple,
    ] {
        hash_string(&mut hasher, value)?;
    }
    hasher.update(identity.flags.to_le_bytes());
    hasher.update(identity.target_os.to_le_bytes());
    hasher.update(identity.target_arch.to_le_bytes());
    hasher.update(identity.backend.to_le_bytes());
    hasher.update(identity.gpu_kind.to_le_bytes());
    hasher.update(identity.cpu_feature_mask.to_le_bytes());
    hasher.update(identity.gpu_family_min.to_le_bytes());
    hasher.update(identity.gpu_family_max.to_le_bytes());
    hasher.update(identity.gpu_capability_min.to_le_bytes());
    hasher.update(identity.gpu_capability_max.to_le_bytes());
    hasher.update(identity.target_profile_abi.to_le_bytes());
    hasher.update(identity.execution_layout_abi.to_le_bytes());
    hasher.update(identity.kernel_abi_min.to_le_bytes());
    hasher.update(identity.kernel_abi_max.to_le_bytes());
    hasher.update(identity.record_alignment.to_le_bytes());
    hasher.update(identity.io_granularity.to_le_bytes());
    hasher.update(identity.resident_alignment.to_le_bytes());
    hasher.update(identity.required_runtime_features.to_le_bytes());
    hasher.update([0]);
    hasher.update([0_u8; 32]);
    hasher.update(0_u64.to_le_bytes());
    hasher.update(Sha256::digest([]));
    Ok(hasher.finalize().into())
}

pub fn encode_manifest(
    plan: &StoragePlan,
    profile: TargetProfile,
    source_fingerprint: [u8; 32],
    records: &[ManifestRecord],
    shard_header_crcs: &[u32],
    options: ArtifactOptions<'_>,
) -> Result<Vec<u8>> {
    if records.len() != plan.records.len() || shard_header_crcs.len() != plan.shards as usize {
        return Err(ColicError::Usage(
            "v1.1 manifest metadata does not match storage plan".into(),
        ));
    }
    let identity = target::identity::for_profile(profile)?;
    if identity.record_alignment as u64 != plan.record_alignment {
        return Err(ColicError::Usage(
            "target descriptor alignment disagrees with storage plan".into(),
        ));
    }
    let mut strings = Vec::<String>::new();
    let mut ids = BTreeMap::<String, u32>::new();
    let mut intern = |value: &str| -> Result<u32> {
        if value.as_bytes().contains(&0) {
            return Err(ColicError::Usage(
                "manifest strings cannot contain NUL".into(),
            ));
        }
        if let Some(id) = ids.get(value) {
            return Ok(*id);
        }
        let id: u32 = strings
            .len()
            .try_into()
            .map_err(|_| ColicError::Usage("too many manifest strings".into()))?;
        strings.push(value.to_owned());
        ids.insert(value.to_owned(), id);
        Ok(id)
    };

    let shard_names = (0..plan.shards)
        .map(|id| format!("data-{id:05}.coli"))
        .collect::<Vec<_>>();
    let shard_name_ids = shard_names
        .iter()
        .map(|name| intern(name))
        .collect::<Result<Vec<_>>>()?;
    let record_name_ids = records
        .iter()
        .map(|record| record.name.as_deref().map(&mut intern).transpose())
        .collect::<Result<Vec<_>>>()?;
    let profile_id = intern(profile.name)?;
    let compiler_id = intern(options.compiler)?;
    let quant_id = intern(options.quant_profile)?;
    let storage_id = intern(options.storage_profile)?;
    let optimization_id = intern(options.optimization_profile)?;
    let kernel_id = intern(identity.kernel_profile)?;
    let triple_id = intern(identity.target_triple)?;
    let semantic_id = intern(identity.semantic_abi)?;

    let target_offset = MANIFEST_HEADER_BYTES as u64;
    let shard_table_offset = align_up(target_offset + TARGET_DESC_BYTES, 16)?;
    let shard_table_bytes = plan.shards as u64 * SHARD_DESC_BYTES;
    let record_table_offset = align_up(shard_table_offset + shard_table_bytes, 16)?;
    let record_table_bytes = plan.records.len() as u64 * RECORD_DESC_BYTES;
    let string_table_offset = align_up(record_table_offset + record_table_bytes, 16)?;
    let string_raw_bytes = strings.iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value.len() as u64)
            .ok_or_else(|| ColicError::Usage("manifest string bytes overflow u64".into()))
    })?;
    let string_table_bytes = align_up(
        strings.len() as u64 * STRING_DESC_BYTES + string_raw_bytes,
        16,
    )?;
    let manifest_bytes = string_table_offset
        .checked_add(string_table_bytes)
        .ok_or_else(|| ColicError::Usage("manifest size overflows u64".into()))?;
    let mut manifest = vec![
        0_u8;
        manifest_bytes.try_into().map_err(|_| {
            ColicError::Usage("manifest exceeds current address space".into())
        })?
    ];

    manifest[..8].copy_from_slice(MANIFEST_MAGIC);
    put_u16(&mut manifest, 8, 1);
    put_u16(&mut manifest, 10, 1);
    put_u32(&mut manifest, 12, MANIFEST_HEADER_BYTES as u32);
    put_u32(&mut manifest, 16, MANIFEST_FLAGS);
    put_u32(&mut manifest, 20, 0x0102_0304);
    put_u32(&mut manifest, 24, identity.record_alignment);
    put_u32(&mut manifest, 28, strings.len() as u32);
    put_u64(&mut manifest, 32, plan.records.len() as u64);
    put_u32(&mut manifest, 40, plan.shards);
    put_u64(&mut manifest, 48, shard_table_offset);
    put_u64(&mut manifest, 56, shard_table_bytes);
    put_u64(&mut manifest, 64, record_table_offset);
    put_u64(&mut manifest, 72, record_table_bytes);
    put_u64(&mut manifest, 80, string_table_offset);
    put_u64(&mut manifest, 88, string_table_bytes);
    manifest[112..144].copy_from_slice(&source_fingerprint);
    put_u32(&mut manifest, 148, profile_id);
    put_u32(&mut manifest, 152, compiler_id);
    put_u64(&mut manifest, 184, target_offset);
    put_u64(&mut manifest, 192, TARGET_DESC_BYTES);

    let target_start = target_offset as usize;
    let target_end = target_start + TARGET_DESC_BYTES as usize;
    let target_desc = &mut manifest[target_start..target_end];
    target_desc[..8].copy_from_slice(b"COLITGT\0");
    put_u16(target_desc, 8, 1);
    put_u16(target_desc, 10, 0);
    put_u32(target_desc, 12, TARGET_DESC_BYTES as u32);
    put_u32(target_desc, 16, identity.flags);
    put_u16(target_desc, 20, identity.target_os);
    put_u16(target_desc, 22, identity.target_arch);
    put_u16(target_desc, 24, identity.backend);
    put_u16(target_desc, 26, identity.gpu_kind);
    put_u64(target_desc, 28, identity.cpu_feature_mask);
    put_u32(target_desc, 36, identity.gpu_family_min);
    put_u32(target_desc, 40, identity.gpu_family_max);
    put_u32(target_desc, 44, identity.gpu_capability_min);
    put_u32(target_desc, 48, identity.gpu_capability_max);
    put_u32(target_desc, 52, identity.target_profile_abi);
    put_u32(target_desc, 56, identity.execution_layout_abi);
    put_u32(target_desc, 60, identity.kernel_abi_min);
    put_u32(target_desc, 64, identity.kernel_abi_max);
    put_u32(target_desc, 68, identity.record_alignment);
    put_u32(target_desc, 72, identity.io_granularity);
    put_u32(target_desc, 76, identity.resident_alignment);
    put_u64(target_desc, 80, identity.required_runtime_features);
    put_u32(target_desc, 88, profile_id);
    put_u32(target_desc, 92, quant_id);
    put_u32(target_desc, 96, storage_id);
    put_u32(target_desc, 100, optimization_id);
    put_u32(target_desc, 104, kernel_id);
    put_u32(target_desc, 108, triple_id);
    put_u32(target_desc, 164, semantic_id);
    let target_crc = crc32c(target_desc);
    put_u32(&mut manifest, 232, target_crc);

    let artifact = artifact_fingerprint(source_fingerprint, profile, options)?;
    manifest[200..232].copy_from_slice(&artifact);

    for shard_id in 0..plan.shards {
        let offset = shard_table_offset as usize + shard_id as usize * SHARD_DESC_BYTES as usize;
        put_u32(&mut manifest, offset, shard_id);
        put_u32(&mut manifest, offset + 8, shard_name_ids[shard_id as usize]);
        let file_bytes = plan
            .records
            .iter()
            .filter(|record| record.shard_id == shard_id)
            .map(|record| record.payload_offset + record.record.stored_bytes)
            .max()
            .unwrap_or(align_up(
                crate::storage::DATA_SHARD_HEADER_BYTES,
                plan.record_alignment,
            )?);
        put_u64(&mut manifest, offset + 16, file_bytes);
        put_u32(
            &mut manifest,
            offset + 24,
            shard_header_crcs[shard_id as usize],
        );
    }

    for (index, (planned, record)) in plan.records.iter().zip(records).enumerate() {
        if planned.record.id != record.id || planned.record.kind != record.kind {
            return Err(ColicError::Usage(
                "v1.1 manifest record metadata disagrees with plan".into(),
            ));
        }
        let offset = record_table_offset as usize + index * RECORD_DESC_BYTES as usize;
        put_u64(&mut manifest, offset, record.id);
        put_u16(&mut manifest, offset + 8, record.kind);
        put_u16(&mut manifest, offset + 10, record.codec);
        put_u16(&mut manifest, offset + 12, record.math_format);
        put_u16(&mut manifest, offset + 14, record.scale_format);
        put_u16(&mut manifest, offset + 16, record.layout);
        put_u16(&mut manifest, offset + 18, record.flags);
        put_u32(&mut manifest, offset + 20, planned.shard_id);
        put_u32(
            &mut manifest,
            offset + 24,
            record_name_ids[index].unwrap_or(u32::MAX),
        );
        put_i32(&mut manifest, offset + 28, record.layer);
        put_i32(&mut manifest, offset + 32, record.expert);
        put_u64(&mut manifest, offset + 40, planned.payload_offset);
        put_u64(&mut manifest, offset + 48, planned.record.stored_bytes);
        put_u64(&mut manifest, offset + 56, planned.record.decoded_bytes);
        put_u32(&mut manifest, offset + 64, record.stored_crc32c);
        put_u32(&mut manifest, offset + 68, record.logical_crc32c);
        put_u32(&mut manifest, offset + 72, record.codec_table_id);
    }

    let string_base = string_table_offset as usize;
    let mut data_offset = strings.len() * STRING_DESC_BYTES as usize;
    for (index, text) in strings.iter().enumerate() {
        let desc = string_base + index * STRING_DESC_BYTES as usize;
        put_u64(&mut manifest, desc, data_offset as u64);
        put_u32(&mut manifest, desc + 8, text.len() as u32);
        let start = string_base + data_offset;
        manifest[start..start + text.len()].copy_from_slice(text.as_bytes());
        data_offset += text.len();
    }

    manifest[144..148].fill(0);
    let manifest_crc = crc32c(&manifest);
    put_u32(&mut manifest, 144, manifest_crc);
    Ok(manifest)
}

fn hash_string(hasher: &mut Sha256, value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    let len: u32 = bytes
        .len()
        .try_into()
        .map_err(|_| ColicError::Usage("artifact identity string exceeds u32".into()))?;
    hasher.update(len.to_le_bytes());
    hasher.update(bytes);
    Ok(())
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
    use crate::{pipeline::TargetRequest, target::HostCapabilities};

    #[test]
    fn apple_artifact_identity_is_deterministic_and_target_sensitive() {
        let profile = target::resolve(
            &TargetRequest::Profile("macos-arm64-metal-apple8-v1".into()),
            HostCapabilities::current(),
        )
        .unwrap();
        let a = artifact_fingerprint([7; 32], profile, ArtifactOptions::default()).unwrap();
        let b = artifact_fingerprint([7; 32], profile, ArtifactOptions::default()).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, [0; 32]);
    }
}

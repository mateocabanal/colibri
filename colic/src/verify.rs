//! Independent on-disk validation for emitted COLI packages.

use std::{
    collections::BTreeSet,
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{
    error::{ColicError, Result},
    storage::{self, DATA_MAGIC, DATA_SHARD_HEADER_BYTES, MANIFEST_HEADER_BYTES, MANIFEST_MAGIC},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationSummary {
    pub shards: u32,
    pub records: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationProgress {
    pub completed_records: u64,
    pub total_records: u64,
    pub verified_bytes: u64,
    pub current_shard: u32,
    pub total_shards: u32,
}

/// Validates final package bytes without using compiler planning state.
pub fn verify_package(package: &Path) -> Result<VerificationSummary> {
    verify_package_with_progress(package, &mut |_| {})
}

/// Validates final package bytes and reports bounded-rate progress. The
/// verifier deliberately reports after integrity checks, not merely after
/// metadata parsing, so reported bytes correspond to completed validation.
pub fn verify_package_with_progress(
    package: &Path,
    progress: &mut dyn FnMut(VerificationProgress),
) -> Result<VerificationSummary> {
    let manifest_path = package.join("manifest.coli");
    let manifest = fs::read(&manifest_path).map_err(|source| ColicError::Io {
        path: manifest_path,
        source,
    })?;
    if manifest.len() < MANIFEST_HEADER_BYTES || &manifest[..8] != MANIFEST_MAGIC {
        return invalid("manifest magic/header is invalid");
    }
    let minor = u16_at(&manifest, 10)?;
    if u16_at(&manifest, 8)? != 1
        || !matches!(minor, 0 | 1)
        || u32_at(&manifest, 12)? != MANIFEST_HEADER_BYTES as u32
    {
        return invalid("manifest header version/size is invalid");
    }
    let manifest_flags = u32_at(&manifest, 16)?;
    if minor == 0 && manifest_flags & 0xffff_0000 != 0 {
        return invalid("manifest contains unknown required feature flags");
    }
    if minor == 1 {
        let required = (1 << 0) | (1 << 1) | (1 << 16);
        if manifest_flags & required != required || manifest_flags & !required != 0 {
            return invalid("v1.1 manifest target/fingerprint flags are invalid");
        }
        let target = variable_region(&manifest, 184, 192, "target descriptor")?;
        if target.len() != 256 || &manifest[target.start..target.start + 8] != b"COLITGT\0" {
            return invalid("v1.1 target descriptor is invalid");
        }
        if storage::crc32c(&manifest[target.clone()]) != u32_at(&manifest, 232)? {
            return invalid("v1.1 target descriptor CRC32C does not match");
        }
        if !manifest[200..232].iter().any(|byte| *byte != 0) {
            return invalid("v1.1 artifact fingerprint is zero");
        }
    }
    let expected_crc = u32_at(&manifest, 144)?;
    let mut crc_bytes = manifest.clone();
    crc_bytes[144..148].fill(0);
    if storage::crc32c(&crc_bytes) != expected_crc {
        return invalid("manifest CRC32C does not match");
    }
    let alignment = u32_at(&manifest, 24)? as u64;
    if !alignment.is_power_of_two() || !(4096..=1024 * 1024).contains(&alignment) {
        return invalid("manifest record alignment is outside COLI v1 limits");
    }
    let records = u64_at(&manifest, 32)?;
    let shards = u32_at(&manifest, 40)?;
    let strings = u32_at(&manifest, 28)?;
    let shard_table = region(&manifest, 48, 56, shards as u64 * 64, "shard table")?;
    let record_bytes = records
        .checked_mul(96)
        .ok_or_else(|| usage("record table overflows"))?;
    let record_table = region(&manifest, 64, 72, record_bytes, "record table")?;
    let string_table = variable_region(&manifest, 80, 88, "string table")?;
    let string_desc_bytes = (strings as usize)
        .checked_mul(16)
        .ok_or_else(|| usage("string descriptor table overflows"))?;
    if string_desc_bytes > string_table.len() {
        return invalid("string table is shorter than its descriptor array");
    }
    let profile_string_id = u32_at(&manifest, 148)?;
    let compiler_string_id = u32_at(&manifest, 152)?;
    validate_string_id(&manifest, &string_table, strings, profile_string_id)?;
    validate_string_id(&manifest, &string_table, strings, compiler_string_id)?;
    let source_fingerprint: [u8; 32] = manifest[112..144].try_into().unwrap();
    if (manifest_flags & 1 != 0) != source_fingerprint.iter().any(|byte| *byte != 0) {
        return invalid("manifest source fingerprint validity flag disagrees with bytes");
    }
    let mut shard_sizes = Vec::with_capacity(shards as usize);
    let mut shard_paths = Vec::with_capacity(shards as usize);
    for shard_id in 0..shards {
        let desc = shard_table.start + shard_id as usize * 64;
        if u32_at(&manifest, desc)? != shard_id {
            return invalid("shard IDs are not contiguous");
        }
        validate_string_id(
            &manifest,
            &string_table,
            strings,
            u32_at(&manifest, desc + 8)?,
        )?;
        let file_bytes = u64_at(&manifest, desc + 16)?;
        let header_crc = u32_at(&manifest, desc + 24)?;
        let path = package.join(format!("data-{shard_id:05}.coli"));
        if fs::metadata(&path)
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?
            .len()
            != file_bytes
        {
            return invalid("shard file size does not match manifest");
        }
        let mut header = [0_u8; DATA_SHARD_HEADER_BYTES as usize];
        fs::File::open(&path)
            .map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })?
            .read_exact(&mut header)
            .map_err(|source| ColicError::Io { path, source })?;
        if &header[..8] != DATA_MAGIC
            || u16_at(&header, 8)? != 1
            || u16_at(&header, 10)? != minor
            || u32_at(&header, 12)? != DATA_SHARD_HEADER_BYTES as u32
            || u32_at(&header, 20)? != shard_id
            || u32_at(&header, 24)? as u64 != alignment
            || u64_at(&header, 32)? != file_bytes
            || header[40..72] != source_fingerprint
        {
            return invalid("data shard header disagrees with manifest");
        }
        let actual_header_crc = u32_at(&header, 72)?;
        let mut crc_header = header;
        crc_header[72..76].fill(0);
        if actual_header_crc != header_crc || storage::crc32c(&crc_header) != actual_header_crc {
            return invalid("data shard header CRC32C does not match");
        }
        shard_sizes.push(file_bytes);
        shard_paths.push(package.join(format!("data-{shard_id:05}.coli")));
    }
    let mut ids = BTreeSet::new();
    let mut verified_bytes = 0_u64;
    for index in 0..records {
        let desc = record_table.start + index as usize * 96;
        let id = u64_at(&manifest, desc)?;
        let kind = u16_at(&manifest, desc + 8)?;
        let codec = u16_at(&manifest, desc + 10)?;
        let flags = u16_at(&manifest, desc + 18)?;
        let shard_id = u32_at(&manifest, desc + 20)? as usize;
        let layer = i32_at(&manifest, desc + 28)?;
        let expert = i32_at(&manifest, desc + 32)?;
        let offset = u64_at(&manifest, desc + 40)?;
        let stored = u64_at(&manifest, desc + 48)?;
        let resident = u64_at(&manifest, desc + 56)?;
        let stored_crc = u32_at(&manifest, desc + 64)?;
        let logical_crc = u32_at(&manifest, desc + 68)?;
        let name_string_id = u32_at(&manifest, desc + 24)?;
        if id == 0 || !ids.insert(id) || shard_id >= shard_sizes.len() {
            return invalid("record ID or shard reference is invalid");
        }
        if name_string_id != u32::MAX {
            validate_string_id(&manifest, &string_table, strings, name_string_id)?;
        }
        if offset % alignment != 0
            || offset
                .checked_add(stored)
                .is_none_or(|end| end > shard_sizes[shard_id])
        {
            return invalid("record range is invalid");
        }
        if crc32c_file_range(&shard_paths[shard_id], offset, stored)? != stored_crc {
            return invalid("record stored CRC32C does not match");
        }
        match kind {
            1 => verify_tensor_record(
                &shard_paths[shard_id],
                offset,
                stored,
                resident,
                codec,
                flags,
                logical_crc,
            )?,
            2 => verify_expert_record(
                &shard_paths[shard_id],
                offset,
                stored,
                resident,
                layer,
                expert,
                codec,
            )?,
            _ => {}
        }
        verified_bytes = verified_bytes
            .checked_add(stored)
            .ok_or_else(|| usage("verified byte total overflows"))?;
        let completed_records = index + 1;
        if completed_records % 64 == 0 || completed_records == records {
            progress(VerificationProgress {
                completed_records,
                total_records: records,
                verified_bytes,
                current_shard: shard_id as u32,
                total_shards: shards,
            });
        }
    }
    Ok(VerificationSummary { shards, records })
}

fn variable_region(
    bytes: &[u8],
    offset_field: usize,
    bytes_field: usize,
    label: &str,
) -> Result<std::ops::Range<usize>> {
    let offset = u64_at(bytes, offset_field)?;
    let length = u64_at(bytes, bytes_field)?;
    if (offset == 0) != (length == 0) || (length != 0 && offset % 16 != 0) {
        return invalid(format!("{label} has an invalid offset or alignment"));
    }
    let end = offset
        .checked_add(length)
        .ok_or_else(|| usage("manifest region overflows"))?;
    if end > bytes.len() as u64 {
        return invalid(format!("{label} is outside the manifest"));
    }
    Ok(offset as usize..end as usize)
}

fn validate_string_id(
    manifest: &[u8],
    string_table: &std::ops::Range<usize>,
    count: u32,
    id: u32,
) -> Result<()> {
    if id >= count {
        return invalid("manifest refers to an invalid string ID");
    }
    let desc = string_table.start + id as usize * 16;
    let offset = u64_at(manifest, desc)?;
    let bytes = u32_at(manifest, desc + 8)? as u64;
    let data_start = string_table
        .start
        .checked_add(offset as usize)
        .ok_or_else(|| usage("string offset overflows"))?;
    let data_end = data_start
        .checked_add(bytes as usize)
        .ok_or_else(|| usage("string length overflows"))?;
    if data_start < string_table.start + count as usize * 16
        || data_end > string_table.end
        || std::str::from_utf8(&manifest[data_start..data_end]).is_err()
        || manifest[data_start..data_end].contains(&0)
    {
        return invalid("manifest string descriptor is invalid");
    }
    Ok(())
}

fn verify_tensor_record(
    path: &Path,
    offset: u64,
    stored: u64,
    decoded: u64,
    codec: u16,
    flags: u16,
    logical_crc: u32,
) -> Result<()> {
    if codec != 0 || stored < 128 {
        return invalid("tensor record uses an unsupported codec or is truncated");
    }
    let header = read_file_range(path, offset, 128)?;
    if &header[..8] != b"COLITENS" || u32_at(&header, 12)? != 128 || u16_at(&header, 16)? > 8 {
        return invalid("tensor envelope header is invalid");
    }
    let data_offset = u64_at(&header, 96)?;
    let data_stored = u64_at(&header, 104)?;
    let data_decoded = u64_at(&header, 112)?;
    if data_offset < 128
        || data_offset % 16 != 0
        || data_stored != data_decoded
        || data_decoded != decoded
        || data_offset
            .checked_add(data_stored)
            .is_none_or(|end| end > stored)
    {
        return invalid("tensor envelope lengths are invalid");
    }
    if flags & 2 != 0 {
        let envelope_crc = u32_at(&header, 120)?;
        if envelope_crc != logical_crc
            || crc32c_file_range(path, offset + data_offset, data_decoded)? != logical_crc
        {
            return invalid("tensor logical CRC32C does not match");
        }
    }
    Ok(())
}

fn verify_expert_record(
    path: &Path,
    offset: u64,
    stored: u64,
    decoded: u64,
    layer: i32,
    expert: i32,
    codec: u16,
) -> Result<()> {
    if codec != 0 || stored < 448 || layer < 0 || expert < 0 {
        return invalid("expert record descriptor is invalid");
    }
    let header = read_file_range(path, offset, 64)?;
    if &header[..8] != b"COLIEXPT"
        || u32_at(&header, 12)? != 64
        || i32_at(&header, 16)? != layer
        || i32_at(&header, 20)? != expert
        || u16_at(&header, 24)? != 3
        || u32_at(&header, 28)? != 128
        || u64_at(&header, 32)? != 64
        || u64_at(&header, 40)? != 448
        || u64_at(&header, 48)? != decoded
    {
        return invalid("expert envelope header disagrees with its descriptor");
    }
    Ok(())
}

fn read_file_range(path: &Path, offset: u64, length: u64) -> Result<Vec<u8>> {
    let mut file = fs::File::open(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
    let mut bytes = vec![0; length as usize];
    file.read_exact(&mut bytes)
        .map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
    Ok(bytes)
}

fn crc32c_file_range(path: &Path, offset: u64, length: u64) -> Result<u32> {
    let mut file = fs::File::open(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
    let mut remaining = length;
    let mut state = !0_u32;
    let mut buffer = [0_u8; 64 * 1024];
    while remaining != 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..count])
            .map_err(|source| ColicError::Io {
                path: path.to_owned(),
                source,
            })?;
        for byte in &buffer[..count] {
            state ^= *byte as u32;
            for _ in 0..8 {
                state = (state >> 1) ^ (0x82f6_3b78 & (0_u32.wrapping_sub(state & 1)));
            }
        }
        remaining -= count as u64;
    }
    Ok(!state)
}

fn region(
    bytes: &[u8],
    offset_field: usize,
    bytes_field: usize,
    expected: u64,
    label: &str,
) -> Result<std::ops::Range<usize>> {
    let offset = u64_at(bytes, offset_field)?;
    let length = u64_at(bytes, bytes_field)?;
    if length != expected || (length != 0 && offset % 16 != 0) {
        return invalid(format!("{label} has an invalid size or alignment"));
    }
    let end = offset
        .checked_add(length)
        .ok_or_else(|| usage("manifest region overflows"))?;
    if end > bytes.len() as u64 {
        return invalid(format!("{label} is outside the manifest"));
    }
    Ok(offset as usize..end as usize)
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| usage("truncated COLI structure"))
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    bytes
        .get(offset..offset + 2)
        .and_then(|value| value.try_into().ok())
        .map(u16::from_le_bytes)
        .ok_or_else(|| usage("truncated COLI structure"))
}

fn i32_at(bytes: &[u8], offset: usize) -> Result<i32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(i32::from_le_bytes)
        .ok_or_else(|| usage("truncated COLI structure"))
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    bytes
        .get(offset..offset + 8)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| usage("truncated COLI structure"))
}

fn usage(detail: impl Into<String>) -> ColicError {
    ColicError::Usage(detail.into())
}

fn invalid<T>(detail: impl Into<String>) -> Result<T> {
    Err(usage(format!("invalid COLI package: {}", detail.into())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        storage::{
            LoweredRecord, ManifestRecord, encode_manifest_with_records, plan_records,
            write_data_shard,
        },
        target::MACOS_ARM64_METAL_APPLE8_V1,
    };

    #[test]
    fn verifies_a_fully_written_single_record_package() {
        let package = std::env::temp_dir().join(format!("colic-verify-{}", std::process::id()));
        std::fs::create_dir(&package).unwrap();
        let record = LoweredRecord {
            id: 1,
            kind: 4,
            stored_bytes: 1,
            decoded_bytes: 1,
        };
        let plan = plan_records(&[record], MACOS_ARM64_METAL_APPLE8_V1, 32 * 1024).unwrap();
        let fingerprint = [6_u8; 32];
        let shard = package.join("data-00000.coli");
        write_data_shard(
            &shard,
            0,
            &plan,
            &[(plan.records[0].clone(), b"x")],
            fingerprint,
        )
        .unwrap();
        let header = std::fs::read(&shard).unwrap();
        let metadata = [ManifestRecord {
            id: 1,
            name: Some("tiny.weight".into()),
            layer: -1,
            expert: -1,
            kind: 4,
            codec: 0,
            math_format: 5,
            scale_format: 0,
            layout: 0,
            flags: 2,
            stored_crc32c: storage::crc32c(b"x"),
            logical_crc32c: storage::crc32c(b"x"),
            codec_table_id: 0,
        }];
        let manifest = encode_manifest_with_records(
            &plan,
            MACOS_ARM64_METAL_APPLE8_V1.name,
            fingerprint,
            &metadata,
            &[u32::from_le_bytes(header[72..76].try_into().unwrap())],
        )
        .unwrap();
        std::fs::write(package.join("manifest.coli"), manifest).unwrap();
        let mut progress = Vec::new();
        assert_eq!(
            verify_package_with_progress(&package, &mut |update| progress.push(update)).unwrap(),
            VerificationSummary {
                shards: 1,
                records: 1
            }
        );
        assert_eq!(
            progress,
            vec![VerificationProgress {
                completed_records: 1,
                total_records: 1,
                verified_bytes: 1,
                current_shard: 0,
                total_shards: 1,
            }]
        );
        let mut corrupt = std::fs::read(&shard).unwrap();
        corrupt[plan.records[0].payload_offset as usize] ^= 1;
        std::fs::write(&shard, corrupt).unwrap();
        assert!(verify_package(&package).is_err());
        std::fs::remove_dir_all(package).unwrap();
    }
}

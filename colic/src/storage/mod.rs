//! Deterministic physical record planning and artifact publication.

pub mod v11;

use std::{
    fs::{self, File},
    io::{Seek, SeekFrom, Write},
    path::Path,
};

use crate::{
    error::{ColicError, Result},
    target::TargetProfile,
};

pub const DATA_SHARD_HEADER_BYTES: u64 = 128;
pub const MANIFEST_HEADER_BYTES: usize = 256;
pub const DATA_MAGIC: &[u8; 8] = b"COLIDAT\0";
pub const MANIFEST_MAGIC: &[u8; 8] = b"COLI\r\n\x1a\n";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredRecord {
    pub id: u64,
    pub kind: u16,
    pub stored_bytes: u64,
    pub decoded_bytes: u64,
}

/// Semantic metadata for a planned record. It is deliberately separate from
/// placement so the writer can assign shard offsets without losing loader ABI
/// information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestRecord {
    pub id: u64,
    pub name: Option<String>,
    pub layer: i32,
    pub expert: i32,
    pub kind: u16,
    pub codec: u16,
    pub math_format: u16,
    pub scale_format: u16,
    pub layout: u16,
    pub flags: u16,
    pub stored_crc32c: u32,
    pub logical_crc32c: u32,
    pub codec_table_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRecord {
    pub record: LoweredRecord,
    pub shard_id: u32,
    pub payload_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoragePlan {
    pub record_alignment: u64,
    pub shard_size_limit: u64,
    pub shards: u32,
    pub records: Vec<PlannedRecord>,
    pub projected_stored_bytes: u64,
    pub projected_padding_bytes: u64,
}

pub fn plan_records(
    records: &[LoweredRecord],
    target: TargetProfile,
    shard_size_limit: u64,
) -> Result<StoragePlan> {
    let alignment = target.record_alignment;
    if !alignment.is_power_of_two() || !(4096..=1024 * 1024).contains(&alignment) {
        return Err(ColicError::Usage(
            "target record alignment is outside COLI v1 limits".into(),
        ));
    }
    let first_offset = align_up(DATA_SHARD_HEADER_BYTES, alignment)?;
    if shard_size_limit < first_offset {
        return Err(ColicError::Usage(
            "shard size limit cannot hold a data-shard header".into(),
        ));
    }
    let mut plan = StoragePlan {
        record_alignment: alignment,
        shard_size_limit,
        shards: 1,
        records: Vec::with_capacity(records.len()),
        projected_stored_bytes: 0,
        projected_padding_bytes: first_offset - DATA_SHARD_HEADER_BYTES,
    };
    let mut previous_id = 0_u64;
    let mut shard_id = 0_u32;
    let mut offset = first_offset;
    for record in records {
        if record.id == 0 || record.id <= previous_id {
            return Err(ColicError::Usage(
                "lowered records must have strictly increasing non-zero IDs".into(),
            ));
        }
        previous_id = record.id;
        if record.stored_bytes > shard_size_limit - first_offset {
            return Err(ColicError::Usage(format!(
                "record {} exceeds the shard-size limit",
                record.id
            )));
        }
        let end = offset.checked_add(record.stored_bytes).ok_or_else(|| {
            ColicError::Usage(format!("record {} offset overflows u64", record.id))
        })?;
        if end > shard_size_limit {
            shard_id = shard_id
                .checked_add(1)
                .ok_or_else(|| ColicError::Usage("shard count overflows u32".into()))?;
            plan.shards = shard_id
                .checked_add(1)
                .ok_or_else(|| ColicError::Usage("shard count overflows u32".into()))?;
            plan.projected_padding_bytes = plan
                .projected_padding_bytes
                .checked_add(first_offset - DATA_SHARD_HEADER_BYTES)
                .ok_or_else(|| ColicError::Usage("padding total overflows u64".into()))?;
            offset = first_offset;
        }
        let payload_offset = offset;
        offset = offset.checked_add(record.stored_bytes).ok_or_else(|| {
            ColicError::Usage(format!("record {} offset overflows u64", record.id))
        })?;
        let next = align_up(offset, alignment)?;
        plan.projected_padding_bytes = plan
            .projected_padding_bytes
            .checked_add(next - offset)
            .ok_or_else(|| ColicError::Usage("padding total overflows u64".into()))?;
        offset = next;
        plan.projected_stored_bytes = plan
            .projected_stored_bytes
            .checked_add(record.stored_bytes)
            .ok_or_else(|| ColicError::Usage("stored-byte total overflows u64".into()))?;
        plan.records.push(PlannedRecord {
            record: record.clone(),
            shard_id,
            payload_offset,
        });
    }
    Ok(plan)
}

pub fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if !alignment.is_power_of_two() {
        return Err(ColicError::Usage("alignment must be a power of two".into()));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| ColicError::Usage("alignment calculation overflows u64".into()))
}

/// Castagnoli CRC32C used by all COLI v1 integrity fields.
pub fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0x82f6_3b78 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

pub fn encode_data_shard_header(
    shard_id: u32,
    file_bytes: u64,
    record_alignment: u64,
    source_fingerprint: [u8; 32],
) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {
    encode_data_shard_header_version(
        shard_id,
        file_bytes,
        record_alignment,
        source_fingerprint,
        0,
    )
}
pub fn encode_data_shard_header_v11(
    shard_id: u32,
    file_bytes: u64,
    record_alignment: u64,
    source_fingerprint: [u8; 32],
) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {
    encode_data_shard_header_version(
        shard_id,
        file_bytes,
        record_alignment,
        source_fingerprint,
        1,
    )
}
fn encode_data_shard_header_version(
    shard_id: u32,
    file_bytes: u64,
    record_alignment: u64,
    source_fingerprint: [u8; 32],
    minor: u16,
) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {
    let alignment: u32 = record_alignment
        .try_into()
        .map_err(|_| ColicError::Usage("record alignment cannot fit COLI v1 data header".into()))?;
    let mut header = [0_u8; DATA_SHARD_HEADER_BYTES as usize];
    header[0..8].copy_from_slice(DATA_MAGIC);
    put_u16(&mut header, 8, 1);
    put_u16(&mut header, 10, minor);
    put_u32(&mut header, 12, DATA_SHARD_HEADER_BYTES as u32);
    put_u32(&mut header, 20, shard_id);
    put_u32(&mut header, 24, alignment);
    put_u64(&mut header, 32, file_bytes);
    header[40..72].copy_from_slice(&source_fingerprint);
    let crc = crc32c(&header);
    put_u32(&mut header, 72, crc);
    Ok(header)
}

pub fn encode_manifest_header(
    record_alignment: u64,
    record_count: u64,
    shard_count: u32,
    source_fingerprint: [u8; 32],
) -> Result<[u8; MANIFEST_HEADER_BYTES]> {
    let alignment: u32 = record_alignment.try_into().map_err(|_| {
        ColicError::Usage("record alignment cannot fit COLI v1 manifest header".into())
    })?;
    let mut header = [0_u8; MANIFEST_HEADER_BYTES];
    header[0..8].copy_from_slice(MANIFEST_MAGIC);
    put_u16(&mut header, 8, 1);
    put_u16(&mut header, 10, 0);
    put_u32(&mut header, 12, MANIFEST_HEADER_BYTES as u32);
    put_u32(&mut header, 16, 1);
    put_u32(&mut header, 20, 0x0102_0304);
    put_u32(&mut header, 24, alignment);
    put_u64(&mut header, 32, record_count);
    put_u32(&mut header, 40, shard_count);
    header[112..144].copy_from_slice(&source_fingerprint);
    let crc = crc32c(&header);
    put_u32(&mut header, 144, crc);
    Ok(header)
}

pub fn encode_manifest(
    plan: &StoragePlan,
    profile_name: &str,
    source_fingerprint: [u8; 32],
) -> Result<Vec<u8>> {
    let records = plan
        .records
        .iter()
        .map(|planned| ManifestRecord {
            id: planned.record.id,
            name: None,
            layer: -1,
            expert: -1,
            kind: planned.record.kind,
            codec: 0,
            math_format: 0,
            scale_format: 0,
            layout: 0,
            flags: 0,
            stored_crc32c: 0,
            logical_crc32c: 0,
            codec_table_id: 0,
        })
        .collect::<Vec<_>>();
    encode_manifest_with_records(plan, profile_name, source_fingerprint, &records, &[])
}

/// Encodes a manifest using final payload metadata and already-written shard
/// header CRCs. This must be called only after every shard has been finalized.
pub fn encode_manifest_with_records(
    plan: &StoragePlan,
    profile_name: &str,
    source_fingerprint: [u8; 32],
    records: &[ManifestRecord],
    shard_header_crcs: &[u32],
) -> Result<Vec<u8>> {
    const SHARD_DESC_BYTES: usize = 64;
    const RECORD_DESC_BYTES: usize = 96;
    const STRING_DESC_BYTES: usize = 16;
    let shard_table_offset = MANIFEST_HEADER_BYTES as u64;
    let shard_table_bytes = plan.shards as u64 * SHARD_DESC_BYTES as u64;
    let record_table_offset = align_up(shard_table_offset + shard_table_bytes, 16)?;
    let record_table_bytes = plan.records.len() as u64 * RECORD_DESC_BYTES as u64;
    let string_table_offset = align_up(record_table_offset + record_table_bytes, 16)?;
    if records.len() != plan.records.len()
        || records
            .iter()
            .zip(&plan.records)
            .any(|(metadata, planned)| {
                metadata.id != planned.record.id || metadata.kind != planned.record.kind
            })
    {
        return Err(ColicError::Usage(
            "manifest metadata does not match the storage plan".into(),
        ));
    }
    if !shard_header_crcs.is_empty() && shard_header_crcs.len() != plan.shards as usize {
        return Err(ColicError::Usage(
            "shard header CRC count does not match the storage plan".into(),
        ));
    }
    let shard_names: Vec<String> = (0..plan.shards)
        .map(|id| format!("data-{id:05}.coli"))
        .collect();
    let mut strings = Vec::new();
    let mut string_ids = std::collections::BTreeMap::new();
    let mut intern = |value: &str| -> Result<u32> {
        if value.as_bytes().contains(&0) {
            return Err(ColicError::Usage(
                "manifest strings cannot contain NUL".into(),
            ));
        }
        if let Some(id) = string_ids.get(value) {
            return Ok(*id);
        }
        let id: u32 = strings
            .len()
            .try_into()
            .map_err(|_| ColicError::Usage("too many manifest strings".into()))?;
        strings.push(value.to_owned());
        string_ids.insert(value.to_owned(), id);
        Ok(id)
    };
    let shard_name_ids = shard_names
        .iter()
        .map(|name| intern(name))
        .collect::<Result<Vec<_>>>()?;
    let record_name_ids = records
        .iter()
        .map(|record| record.name.as_deref().map(&mut intern).transpose())
        .collect::<Result<Vec<_>>>()?;
    let profile_name_string_id = intern(profile_name)?;
    let compiler_string_id = intern("colic-0.1.0")?;
    let string_data_bytes: u64 = strings.iter().map(|text| text.len() as u64).sum();
    let string_table_bytes = align_up(
        strings.len() as u64 * STRING_DESC_BYTES as u64 + string_data_bytes,
        16,
    )?;
    let manifest_bytes = string_table_offset
        .checked_add(string_table_bytes)
        .ok_or_else(|| ColicError::Usage("manifest size overflows u64".into()))?;
    let mut manifest = vec![
        0_u8;
        manifest_bytes.try_into().map_err(|_| ColicError::Usage(
            "manifest exceeds current address space".into()
        ))?
    ];
    let mut header = encode_manifest_header(
        plan.record_alignment,
        plan.records.len() as u64,
        plan.shards,
        source_fingerprint,
    )?;
    put_u64(&mut header, 48, shard_table_offset);
    put_u64(&mut header, 56, shard_table_bytes);
    put_u64(&mut header, 64, record_table_offset);
    put_u64(&mut header, 72, record_table_bytes);
    put_u32(&mut header, 28, strings.len() as u32);
    put_u64(&mut header, 80, string_table_offset);
    put_u64(&mut header, 88, string_table_bytes);
    put_u32(&mut header, 148, profile_name_string_id);
    put_u32(&mut header, 152, compiler_string_id);
    manifest[..MANIFEST_HEADER_BYTES].copy_from_slice(&header);
    for shard_id in 0..plan.shards {
        let offset = shard_table_offset as usize + shard_id as usize * SHARD_DESC_BYTES;
        put_u32(&mut manifest, offset, shard_id);
        put_u32(&mut manifest, offset + 8, shard_name_ids[shard_id as usize]);
        let file_bytes = plan
            .records
            .iter()
            .filter(|record| record.shard_id == shard_id)
            .map(|record| record.payload_offset + record.record.stored_bytes)
            .max()
            .unwrap_or(align_up(DATA_SHARD_HEADER_BYTES, plan.record_alignment)?);
        put_u64(&mut manifest, offset + 16, file_bytes);
        if let Some(crc) = shard_header_crcs.get(shard_id as usize) {
            put_u32(&mut manifest, offset + 24, *crc);
        }
    }
    for (index, (record, metadata)) in plan.records.iter().zip(records).enumerate() {
        let offset = record_table_offset as usize + index * RECORD_DESC_BYTES;
        put_u64(&mut manifest, offset, record.record.id);
        put_u16(&mut manifest, offset + 8, metadata.kind);
        put_u16(&mut manifest, offset + 10, metadata.codec);
        put_u16(&mut manifest, offset + 12, metadata.math_format);
        put_u16(&mut manifest, offset + 14, metadata.scale_format);
        put_u16(&mut manifest, offset + 16, metadata.layout);
        put_u16(&mut manifest, offset + 18, metadata.flags);
        put_u32(&mut manifest, offset + 20, record.shard_id);
        put_u32(
            &mut manifest,
            offset + 24,
            record_name_ids[index].unwrap_or(u32::MAX),
        );
        put_i32(&mut manifest, offset + 28, metadata.layer);
        put_i32(&mut manifest, offset + 32, metadata.expert);
        put_u64(&mut manifest, offset + 40, record.payload_offset);
        put_u64(&mut manifest, offset + 48, record.record.stored_bytes);
        put_u64(&mut manifest, offset + 56, record.record.decoded_bytes);
        put_u32(&mut manifest, offset + 64, metadata.stored_crc32c);
        put_u32(&mut manifest, offset + 68, metadata.logical_crc32c);
        put_u32(&mut manifest, offset + 72, metadata.codec_table_id);
    }
    let string_base = string_table_offset as usize;
    let mut data_offset = strings.len() * STRING_DESC_BYTES;
    for (index, text) in strings.iter().enumerate() {
        let desc = string_base + index * STRING_DESC_BYTES;
        put_u64(&mut manifest, desc, data_offset as u64);
        put_u32(&mut manifest, desc + 8, text.len() as u32);
        let data = string_base + data_offset;
        manifest[data..data + text.len()].copy_from_slice(text.as_bytes());
        data_offset += text.len();
    }
    manifest[144..148].fill(0);
    let crc = crc32c(&manifest);
    put_u32(&mut manifest, 144, crc);
    Ok(manifest)
}

/// Writes one planned data shard without buffering the complete shard in memory.
/// Callers provide records in deterministic plan order and retain ownership of their
/// current lowered payload only until its write completes.
pub fn write_data_shard(
    path: &Path,
    shard_id: u32,
    plan: &StoragePlan,
    payloads: &[(PlannedRecord, &[u8])],
    source_fingerprint: [u8; 32],
) -> Result<()> {
    let mut file = File::create(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut file_bytes = align_up(DATA_SHARD_HEADER_BYTES, plan.record_alignment)?;
    for (record, payload) in payloads {
        if record.shard_id != shard_id || payload.len() as u64 != record.record.stored_bytes {
            return Err(ColicError::Usage(
                "payload does not match its planned data-shard record".into(),
            ));
        }
        let end = record
            .payload_offset
            .checked_add(record.record.stored_bytes)
            .ok_or_else(|| ColicError::Usage("planned record end overflows u64".into()))?;
        file_bytes = file_bytes.max(end);
    }
    file.set_len(file_bytes).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    for (record, payload) in payloads {
        file.seek(SeekFrom::Start(record.payload_offset))
            .map_err(|source| ColicError::Io {
                path: path.to_owned(),
                source,
            })?;
        file.write_all(payload).map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
    }
    let header = encode_data_shard_header(
        shard_id,
        file_bytes,
        plan.record_alignment,
        source_fingerprint,
    )?;
    file.seek(SeekFrom::Start(0))
        .map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
    file.write_all(&header).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    file.sync_all().map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })
}

pub struct DataShardWriter {
    path: std::path::PathBuf,
    shard_id: u32,
    alignment: u64,
    source_fingerprint: [u8; 32],
    minor: u16,
    file: File,
    file_bytes: u64,
}

impl DataShardWriter {
    pub fn create(
        path: &Path,
        shard_id: u32,
        alignment: u64,
        source_fingerprint: [u8; 32],
    ) -> Result<Self> {
        Self::create_version(path, shard_id, alignment, source_fingerprint, 0)
    }
    pub fn create_v11(
        path: &Path,
        shard_id: u32,
        alignment: u64,
        source_fingerprint: [u8; 32],
    ) -> Result<Self> {
        Self::create_version(path, shard_id, alignment, source_fingerprint, 1)
    }
    fn create_version(
        path: &Path,
        shard_id: u32,
        alignment: u64,
        source_fingerprint: [u8; 32],
        minor: u16,
    ) -> Result<Self> {
        let file = File::create(path).map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
        Ok(Self {
            path: path.to_owned(),
            shard_id,
            alignment,
            source_fingerprint,
            minor,
            file,
            file_bytes: align_up(DATA_SHARD_HEADER_BYTES, alignment)?,
        })
    }
    pub fn write_record(&mut self, record: &PlannedRecord, payload: &[u8]) -> Result<()> {
        if record.shard_id != self.shard_id || payload.len() as u64 != record.record.stored_bytes {
            return Err(ColicError::Usage(
                "payload does not match its planned streaming shard record".into(),
            ));
        }
        self.file
            .seek(SeekFrom::Start(record.payload_offset))
            .map_err(|source| ColicError::Io {
                path: self.path.clone(),
                source,
            })?;
        self.file
            .write_all(payload)
            .map_err(|source| ColicError::Io {
                path: self.path.clone(),
                source,
            })?;
        self.file_bytes = self.file_bytes.max(
            record
                .payload_offset
                .checked_add(record.record.stored_bytes)
                .ok_or_else(|| ColicError::Usage("planned record end overflows u64".into()))?,
        );
        Ok(())
    }

    /// Streams exactly one planned payload into its reserved slot. The callback
    /// receives the positioned file and must write precisely `stored_bytes`.
    pub fn write_record_stream(
        &mut self,
        record: &PlannedRecord,
        write: impl FnOnce(&mut File) -> Result<u64>,
    ) -> Result<()> {
        if record.shard_id != self.shard_id {
            return Err(ColicError::Usage(
                "record belongs to a different data shard".into(),
            ));
        }
        self.file
            .seek(SeekFrom::Start(record.payload_offset))
            .map_err(|source| ColicError::Io {
                path: self.path.clone(),
                source,
            })?;
        let written = write(&mut self.file)?;
        if written != record.record.stored_bytes {
            return Err(ColicError::Usage(
                "streamed payload does not match planned record size".into(),
            ));
        }
        self.file_bytes = self.file_bytes.max(
            record
                .payload_offset
                .checked_add(written)
                .ok_or_else(|| ColicError::Usage("planned record end overflows u64".into()))?,
        );
        Ok(())
    }
    pub fn finish(mut self) -> Result<u64> {
        self.file
            .set_len(self.file_bytes)
            .map_err(|source| ColicError::Io {
                path: self.path.clone(),
                source,
            })?;
        let header = encode_data_shard_header_version(
            self.shard_id,
            self.file_bytes,
            self.alignment,
            self.source_fingerprint,
            self.minor,
        )?;
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|source| ColicError::Io {
                path: self.path.clone(),
                source,
            })?;
        self.file
            .write_all(&header)
            .map_err(|source| ColicError::Io {
                path: self.path.clone(),
                source,
            })?;
        self.file.sync_all().map_err(|source| ColicError::Io {
            path: self.path,
            source,
        })?;
        Ok(self.file_bytes)
    }
}

/// Creates a temporary sibling directory so a failed compile cannot publish a
/// partial package. The caller must populate and verify it before publication.
pub fn temporary_package_path(output: &Path) -> Result<std::path::PathBuf> {
    let parent = output
        .parent()
        .ok_or_else(|| ColicError::Usage("output package path has no parent directory".into()))?;
    let name = output
        .file_name()
        .ok_or_else(|| ColicError::Usage("output package path has no file name".into()))?
        .to_string_lossy();
    let temporary = parent.join(format!("{name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(ColicError::Usage(format!(
            "temporary output already exists: {}",
            temporary.display()
        )));
    }
    fs::create_dir(&temporary).map_err(|source| ColicError::Io {
        path: temporary.clone(),
        source,
    })?;
    Ok(temporary)
}

/// Atomically makes a fully-written sibling directory visible. `--force` uses
/// a separate safe-replacement flow and is intentionally not implicit here.
pub fn publish_package(temporary: &Path, output: &Path) -> Result<()> {
    if output.exists() {
        return Err(ColicError::Usage(format!(
            "output package already exists: {} (use --force after safe replacement support lands)",
            output.display()
        )));
    }
    fs::rename(temporary, output).map_err(|source| ColicError::Io {
        path: output.to_owned(),
        source,
    })
}

pub fn replace_package(temporary: &Path, output: &Path) -> Result<()> {
    if !output.exists() {
        return publish_package(temporary, output);
    }
    let backup = output.with_extension(format!("coli.previous-{}", std::process::id()));
    if backup.exists() {
        return Err(ColicError::Usage(format!(
            "safe-replacement backup already exists: {}",
            backup.display()
        )));
    }
    fs::rename(output, &backup).map_err(|source| ColicError::Io {
        path: output.to_owned(),
        source,
    })?;
    if let Err(source) = fs::rename(temporary, output) {
        let _ = fs::rename(&backup, output);
        return Err(ColicError::Io {
            path: output.to_owned(),
            source,
        });
    }
    fs::remove_dir_all(&backup).map_err(|source| ColicError::Io {
        path: backup,
        source,
    })
}

fn put_u16(buffer: &mut [u8], offset: usize, value: u16) {
    buffer[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_i32(buffer: &mut [u8], offset: usize, value: i32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(buffer: &mut [u8], offset: usize, value: u64) {
    buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::MACOS_ARM64_METAL_APPLE8_V1;

    #[test]
    fn crc32c_matches_castagnoli_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
    }

    #[test]
    fn planner_aligns_and_rolls_shards_deterministically() {
        let records = [
            LoweredRecord {
                id: 1,
                kind: 1,
                stored_bytes: 100,
                decoded_bytes: 100,
            },
            LoweredRecord {
                id: 2,
                kind: 2,
                stored_bytes: 100,
                decoded_bytes: 100,
            },
        ];
        let plan = plan_records(&records, MACOS_ARM64_METAL_APPLE8_V1, 32 * 1024).unwrap();
        assert_eq!(plan.records[0].payload_offset, 16 * 1024);
        assert_eq!(plan.records[1].shard_id, 1);
        assert_eq!(plan.shards, 2);
    }

    #[test]
    fn planner_rejects_duplicate_ids_and_oversized_records() {
        let duplicate = [
            LoweredRecord {
                id: 1,
                kind: 1,
                stored_bytes: 1,
                decoded_bytes: 1,
            },
            LoweredRecord {
                id: 1,
                kind: 1,
                stored_bytes: 1,
                decoded_bytes: 1,
            },
        ];
        assert!(plan_records(&duplicate, MACOS_ARM64_METAL_APPLE8_V1, 32 * 1024).is_err());
        let oversized = [LoweredRecord {
            id: 1,
            kind: 1,
            stored_bytes: 16 * 1024 + 1,
            decoded_bytes: 1,
        }];
        assert!(plan_records(&oversized, MACOS_ARM64_METAL_APPLE8_V1, 32 * 1024).is_err());
    }

    #[test]
    fn v1_headers_have_spec_magic_fields_and_self_crcs() {
        let fingerprint = [9_u8; 32];
        let shard = encode_data_shard_header(3, 16_384, 4096, fingerprint).unwrap();
        assert_eq!(&shard[0..8], DATA_MAGIC);
        assert_eq!(
            u32::from_le_bytes(shard[72..76].try_into().unwrap()),
            crc32c(&{
                let mut copy = shard;
                copy[72..76].fill(0);
                copy
            })
        );
        let manifest = encode_manifest_header(4096, 2, 1, fingerprint).unwrap();
        assert_eq!(&manifest[0..8], MANIFEST_MAGIC);
        assert_eq!(
            u32::from_le_bytes(manifest[144..148].try_into().unwrap()),
            crc32c(&{
                let mut copy = manifest;
                copy[144..148].fill(0);
                copy
            })
        );
    }

    #[test]
    fn shard_writer_obeys_planned_offsets_without_whole_shard_buffering() {
        let record = LoweredRecord {
            id: 1,
            kind: 1,
            stored_bytes: 3,
            decoded_bytes: 3,
        };
        let plan = plan_records(&[record], MACOS_ARM64_METAL_APPLE8_V1, 32 * 1024).unwrap();
        let path = std::env::temp_dir().join(format!("colic-shard-{}", std::process::id()));
        write_data_shard(
            &path,
            0,
            &plan,
            &[(plan.records[0].clone(), b"abc")],
            [7; 32],
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], DATA_MAGIC);
        assert_eq!(
            &bytes[plan.records[0].payload_offset as usize
                ..plan.records[0].payload_offset as usize + 3],
            b"abc"
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn streaming_shard_writer_backpatches_a_valid_header_on_finish() {
        let plan = plan_records(
            &[LoweredRecord {
                id: 1,
                kind: 1,
                stored_bytes: 2,
                decoded_bytes: 2,
            }],
            MACOS_ARM64_METAL_APPLE8_V1,
            32 * 1024,
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("colic-stream-{}", std::process::id()));
        let mut writer = DataShardWriter::create(&path, 0, plan.record_alignment, [8; 32]).unwrap();
        writer.write_record(&plan.records[0], b"ok").unwrap();
        let size = writer.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, size);
        assert_eq!(&bytes[..8], DATA_MAGIC);
        assert_eq!(u64::from_le_bytes(bytes[32..40].try_into().unwrap()), size);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn package_publication_is_atomic_and_never_overwrites_an_existing_output() {
        let parent = std::env::temp_dir();
        let output = parent.join(format!("colic-package-{}", std::process::id()));
        let temporary = temporary_package_path(&output).unwrap();
        std::fs::write(temporary.join("manifest.coli"), b"complete").unwrap();
        publish_package(&temporary, &output).unwrap();
        assert_eq!(
            std::fs::read(output.join("manifest.coli")).unwrap(),
            b"complete"
        );
        assert!(publish_package(&temporary, &output).is_err());
        std::fs::remove_dir_all(output).unwrap();
    }

    #[test]
    fn force_replacement_promotes_new_package_only_after_safe_backup() {
        let parent = std::env::temp_dir();
        let output = parent.join(format!("colic-replace-{}", std::process::id()));
        std::fs::create_dir(&output).unwrap();
        std::fs::write(output.join("manifest.coli"), b"old").unwrap();
        let temporary = temporary_package_path(&output).unwrap();
        std::fs::write(temporary.join("manifest.coli"), b"new").unwrap();
        replace_package(&temporary, &output).unwrap();
        assert_eq!(std::fs::read(output.join("manifest.coli")).unwrap(), b"new");
        std::fs::remove_dir_all(output).unwrap();
    }

    #[test]
    fn manifest_encoder_writes_tables_and_a_full_file_crc() {
        let plan = plan_records(
            &[LoweredRecord {
                id: 7,
                kind: 2,
                stored_bytes: 12,
                decoded_bytes: 12,
            }],
            MACOS_ARM64_METAL_APPLE8_V1,
            32 * 1024,
        )
        .unwrap();
        let manifest = encode_manifest(&plan, "macos-arm64-metal-apple8-v1", [4; 32]).unwrap();
        assert_eq!(&manifest[..8], MANIFEST_MAGIC);
        let record_table = u64::from_le_bytes(manifest[64..72].try_into().unwrap()) as usize;
        assert_eq!(
            u64::from_le_bytes(manifest[record_table..record_table + 8].try_into().unwrap()),
            7
        );
        let expected_crc = u32::from_le_bytes(manifest[144..148].try_into().unwrap());
        let mut crc_input = manifest.clone();
        crc_input[144..148].fill(0);
        assert_eq!(expected_crc, crc32c(&crc_input));
    }

    #[test]
    fn metadata_manifest_carries_record_scope_formats_and_checksums() {
        let record = LoweredRecord {
            id: 7,
            kind: 1,
            stored_bytes: 129,
            decoded_bytes: 1,
        };
        let plan = plan_records(&[record], MACOS_ARM64_METAL_APPLE8_V1, 32 * 1024).unwrap();
        let metadata = [ManifestRecord {
            id: 7,
            name: Some("layers.3.attn.weight".into()),
            layer: 3,
            expert: -1,
            kind: 1,
            codec: 0,
            math_format: 5,
            scale_format: 0,
            layout: 0,
            flags: 2,
            stored_crc32c: 0x1122_3344,
            logical_crc32c: 0x5566_7788,
            codec_table_id: 0,
        }];
        let manifest = encode_manifest_with_records(
            &plan,
            "macos-arm64-metal-apple8-v1",
            [3; 32],
            &metadata,
            &[0xaabb_ccdd],
        )
        .unwrap();
        let record_table = u64::from_le_bytes(manifest[64..72].try_into().unwrap()) as usize;
        assert_eq!(
            u16::from_le_bytes(
                manifest[record_table + 12..record_table + 14]
                    .try_into()
                    .unwrap()
            ),
            5
        );
        assert_eq!(
            i32::from_le_bytes(
                manifest[record_table + 28..record_table + 32]
                    .try_into()
                    .unwrap()
            ),
            3
        );
        assert_eq!(
            u32::from_le_bytes(
                manifest[record_table + 64..record_table + 68]
                    .try_into()
                    .unwrap()
            ),
            0x1122_3344
        );
        assert_ne!(
            u32::from_le_bytes(
                manifest[record_table + 24..record_table + 28]
                    .try_into()
                    .unwrap()
            ),
            u32::MAX
        );
        let shard_table = u64::from_le_bytes(manifest[48..56].try_into().unwrap()) as usize;
        assert_eq!(
            u32::from_le_bytes(
                manifest[shard_table + 24..shard_table + 28]
                    .try_into()
                    .unwrap()
            ),
            0xaabb_ccdd
        );
    }
}

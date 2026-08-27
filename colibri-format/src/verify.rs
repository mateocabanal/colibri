//! Independent on-disk validation for emitted COLI packages. Shared by the
//! `colic` compiler and the future Rust inference runtime; never depends on
//! compiler internals.

use std::{
    collections::BTreeSet,
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{DATA_MAGIC, DATA_SHARD_HEADER_BYTES, MANIFEST_HEADER_BYTES, MANIFEST_MAGIC};

/// Errors produced while reading or validating COLI CSF artifacts.
#[derive(Debug)]
pub enum FormatError {
    /// The artifact is structurally or semantically invalid.
    Invalid(String),
    /// The artifact could not be read.
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "{message}"),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for FormatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, FormatError>;

/// Record-scope execution-layout sentinel: the record carries no fixed
/// execution layout (expert records carry per-matrix layouts in their
/// envelope instead).
pub(crate) const LAYOUT_NONE: u16 = 0xfffe;

/// Rounds a value up to a power-of-two alignment.
pub(crate) fn align_up_impl(value: u64, alignment: u64) -> Result<u64> {
    if !alignment.is_power_of_two() {
        return Err(usage("alignment must be a power of two"));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| usage("alignment calculation overflows u64"))
}

/// Castagnoli CRC32C used by all COLI v1 integrity fields.
pub(crate) fn crc32c_impl(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0x82f6_3b78 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

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
    let manifest = fs::read(&manifest_path).map_err(|source| FormatError::Io {
        path: manifest_path,
        source,
    })?;
    if manifest.len() < MANIFEST_HEADER_BYTES || &manifest[..8] != MANIFEST_MAGIC {
        return invalid("manifest magic/header is invalid");
    }
    if u16_at(&manifest, 8)? != 1
        || u16_at(&manifest, 10)? != 0
        || u32_at(&manifest, 12)? != MANIFEST_HEADER_BYTES as u32
    {
        return invalid("manifest header size is invalid");
    }
    let manifest_flags = u32_at(&manifest, 16)?;
    if manifest_flags & 0xffff_0000 != 0 {
        return invalid("manifest contains unknown required feature flags");
    }
    let expected_crc = u32_at(&manifest, 144)?;
    let mut crc_bytes = manifest.clone();
    crc_bytes[144..148].fill(0);
    if crate::crc32c(&crc_bytes) != expected_crc {
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
            .map_err(|source| FormatError::Io {
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
            .map_err(|source| FormatError::Io {
                path: path.clone(),
                source,
            })?
            .read_exact(&mut header)
            .map_err(|source| FormatError::Io { path, source })?;
        if &header[..8] != DATA_MAGIC
            || u16_at(&header, 8)? != 1
            || u16_at(&header, 10)? != 0
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
        if actual_header_crc != header_crc || crate::crc32c(&crc_header) != actual_header_crc {
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
        let layout = u16_at(&manifest, desc + 16)?;
        // `LAYOUT_NONE` is the record-scope sentinel for records that carry
        // their execution layouts inside their envelope (expert records, whose
        // per-matrix layouts are validated against the registry elsewhere).
        // Every other value must be a registered execution layout.
        if layout != LAYOUT_NONE
            && !colibri_abi::generated::target_registry::layout_registered(layout)
        {
            return invalid("record uses an unregistered execution layout");
        }
        let flags = u16_at(&manifest, desc + 18)?;
        let shard_id = u32_at(&manifest, desc + 20)? as usize;
        let layer = i32_at(&manifest, desc + 28)?;
        let expert = i32_at(&manifest, desc + 32)?;
        let offset = u64_at(&manifest, desc + 40)?;
        let stored = u64_at(&manifest, desc + 48)?;
        let decoded = u64_at(&manifest, desc + 56)?;
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
                decoded,
                codec,
                flags,
                logical_crc,
            )?,
            2 => verify_expert_record(
                &shard_paths[shard_id],
                offset,
                stored,
                decoded,
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
    let mut file = fs::File::open(path).map_err(|source| FormatError::Io {
        path: path.to_owned(),
        source,
    })?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| FormatError::Io {
            path: path.to_owned(),
            source,
        })?;
    let mut bytes = vec![0; length as usize];
    file.read_exact(&mut bytes)
        .map_err(|source| FormatError::Io {
            path: path.to_owned(),
            source,
        })?;
    Ok(bytes)
}

fn crc32c_file_range(path: &Path, offset: u64, length: u64) -> Result<u32> {
    let mut file = fs::File::open(path).map_err(|source| FormatError::Io {
        path: path.to_owned(),
        source,
    })?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| FormatError::Io {
            path: path.to_owned(),
            source,
        })?;
    let mut remaining = length;
    let mut state = !0_u32;
    let mut buffer = [0_u8; 64 * 1024];
    while remaining != 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..count])
            .map_err(|source| FormatError::Io {
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

fn usage(detail: impl Into<String>) -> FormatError {
    FormatError::Invalid(detail.into())
}

fn invalid<T>(detail: impl Into<String>) -> Result<T> {
    Err(usage(format!("invalid COLI package: {}", detail.into())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{align_up, crc32c};

    #[test]
    fn crc32c_matches_castagnoli_check_value() {
        // Standard CRC-32C("123456789") check value; the bitwise Castagnoli
        // implementation here matches it exactly.
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        // Deterministic: same input twice gives the same output, and a
        // different input gives a different output.
        assert_eq!(crc32c(b"123456789"), crc32c(b"123456789"));
        assert_ne!(crc32c(b"123456789"), crc32c(b"123456790"));
    }

    #[test]
    fn align_up_rounds_to_power_of_two_and_rejects_bad_alignment() {
        assert_eq!(align_up(1, 4096).unwrap(), 4096);
        assert_eq!(align_up(4096, 4096).unwrap(), 4096);
        assert_eq!(align_up(4097, 4096).unwrap(), 8192);
        assert!(align_up(1, 3).is_err());
        assert!(align_up(u64::MAX, 4096).is_err());
    }

    /// Builds a minimal fully-valid COLI v1 package in a fresh tempdir:
    /// one shard (expert record kind=2) and a manifest whose CRC32C is
    /// computed over the manifest with its CRC field zeroed.
    fn write_valid_package() -> std::path::PathBuf {
        let package = std::env::temp_dir().join(format!(
            "colibri-format-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&package).unwrap();

        let fingerprint = [6_u8; 32];
        const ALIGNMENT: u32 = 4096;
        const FILE_BYTES: u64 = 4544; // header + one 448-byte expert envelope
        const PROFILE: &str = "macos-arm64-metal-apple8-v1";
        const SHARD_NAME: &str = "data-00000.coli";

        // --- data shard: header + expert envelope at offset 4096 ---
        let mut shard = vec![0_u8; FILE_BYTES as usize];
        shard[0..8].copy_from_slice(DATA_MAGIC);
        shard[8..10].copy_from_slice(&1_u16.to_le_bytes());
        shard[10..12].copy_from_slice(&0_u16.to_le_bytes());
        shard[12..16].copy_from_slice(&(DATA_SHARD_HEADER_BYTES as u32).to_le_bytes());
        shard[20..24].copy_from_slice(&0_u32.to_le_bytes());
        shard[24..28].copy_from_slice(&ALIGNMENT.to_le_bytes());
        shard[32..40].copy_from_slice(&FILE_BYTES.to_le_bytes());
        shard[40..72].copy_from_slice(&fingerprint);
        let header_crc = crc32c(&shard[..128]);
        shard[72..76].copy_from_slice(&header_crc.to_le_bytes());

        let envelope = 4096_usize;
        shard[envelope..envelope + 8].copy_from_slice(b"COLIEXPT");
        shard[envelope + 12..envelope + 16].copy_from_slice(&64_u32.to_le_bytes());
        shard[envelope + 16..envelope + 20].copy_from_slice(&0_i32.to_le_bytes());
        shard[envelope + 20..envelope + 24].copy_from_slice(&0_i32.to_le_bytes());
        shard[envelope + 24..envelope + 26].copy_from_slice(&3_u16.to_le_bytes());
        shard[envelope + 28..envelope + 32].copy_from_slice(&128_u32.to_le_bytes());
        shard[envelope + 32..envelope + 40].copy_from_slice(&64_u64.to_le_bytes());
        shard[envelope + 40..envelope + 48].copy_from_slice(&448_u64.to_le_bytes());
        shard[envelope + 48..envelope + 56].copy_from_slice(&1_u64.to_le_bytes());
        let stored_crc = crc32c(&shard[4096..4096 + 448]);
        std::fs::write(package.join(SHARD_NAME), &shard).unwrap();

        // --- manifest ---
        const STRING_TABLE_OFFSET: usize = 416;
        const STRING_TABLE_BYTES: u64 = 80;
        let mut manifest = vec![0_u8; STRING_TABLE_OFFSET + STRING_TABLE_BYTES as usize];
        manifest[0..8].copy_from_slice(MANIFEST_MAGIC);
        manifest[8..10].copy_from_slice(&1_u16.to_le_bytes());
        manifest[10..12].copy_from_slice(&0_u16.to_le_bytes());
        manifest[12..16].copy_from_slice(&(MANIFEST_HEADER_BYTES as u32).to_le_bytes());
        manifest[16..20].copy_from_slice(&1_u32.to_le_bytes()); // fingerprint-validity flag
        manifest[20..24].copy_from_slice(&0x0102_0304_u32.to_le_bytes());
        manifest[24..28].copy_from_slice(&ALIGNMENT.to_le_bytes());
        manifest[28..32].copy_from_slice(&2_u32.to_le_bytes()); // string count
        manifest[32..40].copy_from_slice(&1_u64.to_le_bytes()); // records
        manifest[40..44].copy_from_slice(&1_u32.to_le_bytes()); // shards
        manifest[48..56].copy_from_slice(&256_u64.to_le_bytes()); // shard table
        manifest[56..64].copy_from_slice(&64_u64.to_le_bytes());
        manifest[64..72].copy_from_slice(&320_u64.to_le_bytes()); // record table
        manifest[72..80].copy_from_slice(&96_u64.to_le_bytes());
        manifest[80..88].copy_from_slice(&(STRING_TABLE_OFFSET as u64).to_le_bytes());
        manifest[88..96].copy_from_slice(&STRING_TABLE_BYTES.to_le_bytes());
        manifest[112..144].copy_from_slice(&fingerprint);
        manifest[148..152].copy_from_slice(&0_u32.to_le_bytes()); // profile string
        manifest[152..156].copy_from_slice(&1_u32.to_le_bytes()); // compiler string

        // shard table @256: id 0, name string 0, file bytes, header crc
        manifest[256..260].copy_from_slice(&0_u32.to_le_bytes());
        manifest[264..268].copy_from_slice(&0_u32.to_le_bytes());
        manifest[272..280].copy_from_slice(&FILE_BYTES.to_le_bytes());
        manifest[280..284].copy_from_slice(&header_crc.to_le_bytes());

        // record table @320: expert record, layout 0, offset 4096, stored 448,
        // decoded 1, stored crc of the 448-byte envelope region
        manifest[320..328].copy_from_slice(&1_u64.to_le_bytes()); // id
        manifest[328..330].copy_from_slice(&2_u16.to_le_bytes()); // kind = expert
        manifest[330..332].copy_from_slice(&0_u16.to_le_bytes()); // codec
        manifest[332..334].copy_from_slice(&0_u16.to_le_bytes()); // math format
        manifest[334..336].copy_from_slice(&0_u16.to_le_bytes()); // scale format
        manifest[336..338].copy_from_slice(&0_u16.to_le_bytes()); // layout (registered)
        manifest[338..340].copy_from_slice(&0_u16.to_le_bytes()); // flags
        manifest[340..344].copy_from_slice(&0_u32.to_le_bytes()); // shard id
        manifest[344..348].copy_from_slice(&u32::MAX.to_le_bytes()); // no name
        manifest[348..352].copy_from_slice(&0_i32.to_le_bytes()); // layer
        manifest[352..356].copy_from_slice(&0_i32.to_le_bytes()); // expert
        manifest[360..368].copy_from_slice(&4096_u64.to_le_bytes()); // offset
        manifest[368..376].copy_from_slice(&448_u64.to_le_bytes()); // stored
        manifest[376..384].copy_from_slice(&1_u64.to_le_bytes()); // decoded
        manifest[384..388].copy_from_slice(&stored_crc.to_le_bytes());

        // string table @416: two descriptors, then data "data-00000.coli" and
        // the profile name (descriptor data starts right after the table)
        let desc0 = STRING_TABLE_OFFSET;
        let desc1 = STRING_TABLE_OFFSET + 16;
        manifest[desc0..desc0 + 8].copy_from_slice(&32_u64.to_le_bytes());
        manifest[desc0 + 8..desc0 + 12].copy_from_slice(&(SHARD_NAME.len() as u32).to_le_bytes());
        manifest[desc1..desc1 + 8].copy_from_slice(&(32 + SHARD_NAME.len() as u64).to_le_bytes());
        manifest[desc1 + 8..desc1 + 12].copy_from_slice(&(PROFILE.len() as u32).to_le_bytes());
        let data0 = STRING_TABLE_OFFSET + 32;
        manifest[data0..data0 + SHARD_NAME.len()].copy_from_slice(SHARD_NAME.as_bytes());
        let data1 = data0 + SHARD_NAME.len();
        manifest[data1..data1 + PROFILE.len()].copy_from_slice(PROFILE.as_bytes());

        // manifest CRC32C over the whole file with the CRC field zeroed
        manifest[144..148].fill(0);
        let manifest_crc = crc32c(&manifest);
        manifest[144..148].copy_from_slice(&manifest_crc.to_le_bytes());
        std::fs::write(package.join("manifest.coli"), manifest).unwrap();

        package
    }

    #[test]
    fn verifies_a_minimal_valid_package() {
        let package = write_valid_package();
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
                verified_bytes: 448,
                current_shard: 0,
                total_shards: 1,
            }]
        );
        std::fs::remove_dir_all(package).unwrap();
    }

    #[test]
    fn rejects_a_manifest_with_a_flipped_byte() {
        let package = write_valid_package();
        let manifest_path = package.join("manifest.coli");
        let mut manifest = std::fs::read(&manifest_path).unwrap();
        manifest[200] ^= 0xff;
        std::fs::write(&manifest_path, manifest).unwrap();
        assert!(verify_package(&package).is_err());
        std::fs::remove_dir_all(package).unwrap();
    }

    #[test]
    fn rejects_a_record_with_an_unregistered_layout() {
        let package = write_valid_package();
        let manifest_path = package.join("manifest.coli");
        let mut manifest = std::fs::read(&manifest_path).unwrap();
        // Record descriptor layout field lives at record_table_offset + 16.
        manifest[320 + 16..320 + 18].copy_from_slice(&0x9999_u16.to_le_bytes());
        let mut crc_input = manifest.clone();
        crc_input[144..148].fill(0);
        let manifest_crc = crc32c(&crc_input);
        manifest[144..148].copy_from_slice(&manifest_crc.to_le_bytes());
        std::fs::write(&manifest_path, manifest).unwrap();
        let error = verify_package(&package).unwrap_err();
        assert!(
            matches!(&error, FormatError::Invalid(message) if message.contains("unregistered execution layout")),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(package).unwrap();
    }
}

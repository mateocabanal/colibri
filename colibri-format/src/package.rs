//! Typed COLI package reader: structural open, record index, and lazy
//! CRC-verified payload reads. The full-byte verifier lives in [`crate::verify`];
//! opening a package here validates structure only, so large packages open
//! without reading every record byte (plan RW-010: "package open, record
//! index, and capability preflight without model execution").

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use crate::{
    MANIFEST_HEADER_BYTES, MANIFEST_MAGIC,
    verify::{
        FormatError, LAYOUT_NONE, Result, i32_at, invalid, read_file_range, region, string_at,
        u16_at, u32_at, u64_at, usage, validate_string_id, variable_region,
    },
};

/// One entry of the manifest record table (96 bytes on disk).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordInfo {
    pub id: u64,
    pub kind: u16,
    pub codec: u16,
    pub math_format: u16,
    pub scale_format: u16,
    pub layout: u16,
    pub flags: u16,
    pub shard_id: u32,
    pub name: Option<String>,
    pub layer: i32,
    pub expert: i32,
    pub offset: u64,
    pub stored: u64,
    pub decoded: u64,
    pub stored_crc: u32,
    pub logical_crc: u32,
}

/// An opened COLI package: validated manifest, shard table, and record index.
/// No record bytes are read until a payload is requested.
#[derive(Debug)]
pub struct Package {
    root: PathBuf,
    alignment: u64,
    profile: String,
    compiler: String,
    fingerprint: [u8; 32],
    records: Vec<RecordInfo>,
    by_id: HashMap<u64, usize>,
    by_name: HashMap<String, usize>,
    by_expert: HashMap<(i32, i32), Vec<usize>>,
}

impl Package {
    /// Opens and structurally validates a `.coli` package directory.
    pub fn open(root: &Path) -> Result<Package> {
        let manifest_path = root.join("manifest.coli");
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
        let flags = u32_at(&manifest, 16)?;
        if flags & 0xffff_0000 != 0 {
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
        let profile =
            string_at(&manifest, &string_table, strings, u32_at(&manifest, 148)?)?.to_owned();
        let compiler =
            string_at(&manifest, &string_table, strings, u32_at(&manifest, 152)?)?.to_owned();
        let fingerprint: [u8; 32] = manifest[112..144].try_into().unwrap();
        if (flags & 1 != 0) != fingerprint.iter().any(|byte| *byte != 0) {
            return invalid("manifest source fingerprint validity flag disagrees with bytes");
        }

        // Shard table: contiguous IDs; files exist with the manifest sizes.
        let mut shard_sizes = Vec::with_capacity(shards as usize);
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
            let path = root.join(format!("data-{shard_id:05}.coli"));
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
            shard_sizes.push(file_bytes);
        }

        // Record table: unique IDs, unique names, in-bounds shard/range refs.
        let record_count = records;
        let mut records = Vec::with_capacity(record_count as usize);
        let mut by_id = HashMap::with_capacity(record_count as usize);
        let mut by_name = HashMap::new();
        let mut by_expert: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for index in 0..record_count {
            let desc = record_table.start + index as usize * 96;
            let id = u64_at(&manifest, desc)?;
            let kind = u16_at(&manifest, desc + 8)?;
            let codec = u16_at(&manifest, desc + 10)?;
            let layout = u16_at(&manifest, desc + 16)?;
            // LAYOUT_NONE is the record-scope sentinel for expert records,
            // whose per-matrix layouts live in their envelope.
            if layout != LAYOUT_NONE
                && !colibri_abi::generated::target_registry::layout_registered(layout)
            {
                return invalid("record uses an unregistered execution layout");
            }
            let name = match u32_at(&manifest, desc + 24)? {
                u32::MAX => None,
                name_id => {
                    validate_string_id(&manifest, &string_table, strings, name_id)?;
                    Some(string_at(&manifest, &string_table, strings, name_id)?.to_owned())
                }
            };
            let info = RecordInfo {
                id,
                kind,
                codec,
                math_format: u16_at(&manifest, desc + 12)?,
                scale_format: u16_at(&manifest, desc + 14)?,
                layout,
                flags: u16_at(&manifest, desc + 18)?,
                shard_id: u32_at(&manifest, desc + 20)?,
                name,
                layer: i32_at(&manifest, desc + 28)?,
                expert: i32_at(&manifest, desc + 32)?,
                offset: u64_at(&manifest, desc + 40)?,
                stored: u64_at(&manifest, desc + 48)?,
                decoded: u64_at(&manifest, desc + 56)?,
                stored_crc: u32_at(&manifest, desc + 64)?,
                logical_crc: u32_at(&manifest, desc + 68)?,
            };
            if id == 0
                || by_id.insert(id, index as usize).is_some()
                || info.shard_id as usize >= shard_sizes.len()
            {
                return invalid("record ID or shard reference is invalid");
            }
            if info.offset % alignment != 0
                || info
                    .offset
                    .checked_add(info.stored)
                    .is_none_or(|end| end > shard_sizes[info.shard_id as usize])
            {
                return invalid("record range is invalid");
            }
            if let Some(name) = &info.name {
                if by_name.insert(name.clone(), index as usize).is_some() {
                    return invalid("record name is not unique");
                }
            }
            if kind == 2 {
                // Multiple records per (layer, expert) are legal: they are
                // alternative representations of the same semantic expert.
                by_expert
                    .entry((info.layer, info.expert))
                    .or_default()
                    .push(index as usize);
            }
            records.push(info);
        }
        Ok(Package {
            root: root.to_owned(),
            alignment,
            profile,
            compiler,
            fingerprint,
            records,
            by_id,
            by_name,
            by_expert,
        })
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }
    pub fn compiler(&self) -> &str {
        &self.compiler
    }
    pub fn alignment(&self) -> u64 {
        self.alignment
    }
    pub fn fingerprint(&self) -> &[u8; 32] {
        &self.fingerprint
    }
    pub fn records(&self) -> &[RecordInfo] {
        &self.records
    }
    pub fn record_by_id(&self, id: u64) -> Option<&RecordInfo> {
        self.by_id.get(&id).map(|index| &self.records[*index])
    }
    pub fn record_by_name(&self, name: &str) -> Option<&RecordInfo> {
        self.by_name.get(name).map(|index| &self.records[*index])
    }
    /// All records for a semantic (layer, expert): one per representation.
    pub fn expert_records(&self, layer: i32, expert: i32) -> Vec<&RecordInfo> {
        self.by_expert
            .get(&(layer, expert))
            .map(|indices| indices.iter().map(|index| &self.records[*index]).collect())
            .unwrap_or_default()
    }

    /// Reads a record's raw stored bytes and verifies the stored CRC32C.
    pub fn read_record(&self, record: &RecordInfo) -> Result<Vec<u8>> {
        let path = self.root.join(format!("data-{:05}.coli", record.shard_id));
        let bytes = read_file_range(&path, record.offset, record.stored)?;
        if crate::crc32c(&bytes) != record.stored_crc {
            return invalid("record stored CRC32C does not match");
        }
        Ok(bytes)
    }

    /// Reads a tensor record (kind 1) and returns its payload bytes with the
    /// envelope validated and the logical CRC checked when the record carries
    /// one. Fails if the record is not a codec-none tensor (rANS decode is
    /// plan RW-014).
    pub fn read_tensor_payload(&self, record: &RecordInfo) -> Result<Vec<u8>> {
        if record.kind != 1 {
            return invalid("record is not a tensor record");
        }
        if record.codec != 0 {
            return invalid(
                "tensor record uses an unsupported codec (rANS decode lands with RW-014)",
            );
        }
        let bytes = self.read_record(record)?;
        if &bytes[..8] != b"COLITENS" || u32_at(&bytes, 12)? != 128 || u16_at(&bytes, 16)? > 8 {
            return invalid("tensor envelope header is invalid");
        }
        let data_offset = u64_at(&bytes, 96)?;
        let data_stored = u64_at(&bytes, 104)?;
        let data_decoded = u64_at(&bytes, 112)?;
        if data_offset < 128
            || data_offset % 16 != 0
            || data_stored != data_decoded
            || data_stored != record.decoded
            || data_offset
                .checked_add(data_stored)
                .is_none_or(|end| end > bytes.len() as u64)
        {
            return invalid("tensor envelope lengths are invalid");
        }
        let data_end = (data_offset + data_stored) as usize;
        if record.flags & 2 != 0 {
            let envelope_crc = u32_at(&bytes, 120)?;
            if envelope_crc != record.logical_crc
                || crate::crc32c(&bytes[data_offset as usize..data_end]) != record.logical_crc
            {
                return invalid("tensor logical CRC32C does not match");
            }
        }
        Ok(bytes[data_offset as usize..data_end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::tests::write_valid_package;

    #[test]
    fn opens_and_indexes_the_valid_fixture() {
        let package = write_valid_package();
        let opened = Package::open(&package).unwrap();
        assert_eq!(opened.profile(), "macos-arm64-metal-apple8-v1");
        assert_eq!(opened.records().len(), 1);
        let record = opened.record_by_id(1).unwrap().clone();
        assert_eq!(record.kind, 2);
        assert_eq!(opened.expert_records(0, 0), vec![&record]);
        let bytes = opened.read_record(&record).unwrap();
        assert_eq!(&bytes[..8], b"COLIEXPT");
        std::fs::remove_dir_all(package).unwrap();
    }

    #[test]
    fn read_record_rejects_corrupted_payload() {
        let package = write_valid_package();
        let shard = package.join("data-00000.coli");
        let mut bytes = std::fs::read(&shard).unwrap();
        bytes[4096 + 24] ^= 0xff;
        std::fs::write(&shard, bytes).unwrap();
        // Open is structural, so it still succeeds; the lazy read must fail.
        let opened = Package::open(&package).unwrap();
        let record = opened.record_by_id(1).unwrap();
        assert!(opened.read_record(record).is_err());
        std::fs::remove_dir_all(package).unwrap();
    }

    #[test]
    fn record_by_name_and_tensor_payload() {
        let package = write_valid_package();
        // Turn the fixture's expert record into a named tensor record (kind 1,
        // name = string 0) and rewrite the shard envelope as COLITENS with a
        // one-byte payload; recompute both CRCs.
        let manifest_path = package.join("manifest.coli");
        let mut manifest = std::fs::read(&manifest_path).unwrap();
        manifest[320 + 8..320 + 10].copy_from_slice(&1_u16.to_le_bytes()); // kind
        manifest[320 + 24..320 + 28].copy_from_slice(&0_u32.to_le_bytes()); // name id
        let mut crc_input = manifest.clone();
        crc_input[144..148].fill(0);
        manifest[144..148].copy_from_slice(&crate::crc32c(&crc_input).to_le_bytes());
        std::fs::write(&manifest_path, manifest).unwrap();

        let shard_path = package.join("data-00000.coli");
        let mut shard = std::fs::read(&shard_path).unwrap();
        let e = 4096;
        shard[e..e + 8].copy_from_slice(b"COLITENS");
        shard[e + 12..e + 16].copy_from_slice(&128_u32.to_le_bytes()); // header size
        shard[e + 16] = 1; // ndim
        shard[e + 96..e + 104].copy_from_slice(&128_u64.to_le_bytes()); // data offset
        shard[e + 104..e + 112].copy_from_slice(&1_u64.to_le_bytes()); // data stored
        shard[e + 112..e + 120].copy_from_slice(&1_u64.to_le_bytes()); // data decoded
        shard[e + 128] = 0x2a; // payload
        let stored_crc = crate::crc32c(&shard[e..e + 448]);
        std::fs::write(&shard_path, shard).unwrap();

        let mut manifest = std::fs::read(&manifest_path).unwrap();
        manifest[320 + 64..320 + 68].copy_from_slice(&stored_crc.to_le_bytes());
        let mut crc_input = manifest.clone();
        crc_input[144..148].fill(0);
        manifest[144..148].copy_from_slice(&crate::crc32c(&crc_input).to_le_bytes());
        std::fs::write(&manifest_path, manifest).unwrap();

        let opened = Package::open(&package).unwrap();
        let record = opened.record_by_name("data-00000.coli").unwrap();
        assert_eq!(opened.read_tensor_payload(record).unwrap(), vec![0x2a]);
        std::fs::remove_dir_all(package).unwrap();
    }
}

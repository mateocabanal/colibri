#!/usr/bin/env python3
from pathlib import Path


def rep(path: str, old: str, new: str, count: int = 1) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"missing anchor in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, count))


rep("colic/src/target/mod.rs", "//! Versioned target compatibility profiles and lowering boundary.\n", "//! Versioned target compatibility profiles and lowering boundary.\n\npub mod identity;\n")
rep("colic/src/target/mod.rs", "    ir::RoutedExpert,", "    ir::{MathFormat, RoutedExpert, ScaleFormat},")
rep("colic/src/target/mod.rs", 'name: "linux-x86_64-avx2-v1",', 'name: "linux-x86_64-cpu-avx2-v1",')

p = Path("colic/src/target/mod.rs")
t = p.read_text()
t = t.replace("expert_math_format(&matrix.source.dtype)?", "math_format_for_semantics(matrix.quantization.math_format)")
t = t.replace("scale_format(&scale.dtype)?", "scale_format_for_semantics(matrix.quantization.scale_format)")
t = t.replace('''        if matrix.source.dtype == "I8" {
            put_u32(&mut payload, desc + 32, 1);
            put_u32(&mut payload, desc + 36, 32);
        }''','''        put_u32(&mut payload, desc + 32, matrix.quantization.scale_block_rows);
        put_u32(&mut payload, desc + 36, matrix.quantization.scale_block_columns);''')
t = t.replace('''        if matrix.source.dtype == "I8" {
            put_u32(&mut header, desc + 32, 1);
            put_u32(&mut header, desc + 36, 32);
        }''','''        put_u32(&mut header, desc + 32, matrix.quantization.scale_block_rows);
        put_u32(&mut header, desc + 36, matrix.quantization.scale_block_columns);''')
old = "pub fn stream_exact_expert<W: Write + Seek>(expert: &RoutedExpert, output: &mut W) -> Result<u32> {"
if old not in t: raise SystemExit("stream_exact_expert signature missing")
t = t.replace(old, '''pub fn stream_exact_expert<W: Write + Seek>(expert: &RoutedExpert, output: &mut W) -> Result<u32> {
    stream_expert_with_layout(expert, None, output)
}

pub fn stream_target_expert<W: Write + Seek>(profile: TargetProfile, expert: &RoutedExpert, output: &mut W) -> Result<u32> {
    let layout = identity::expert_layout(profile, expert)?;
    stream_expert_with_layout(expert, Some(layout), output)
}

fn stream_expert_with_layout<W: Write + Seek>(expert: &RoutedExpert, layout: Option<u16>, output: &mut W) -> Result<u32> {''', 1)
rep_text = '''    header[..8].copy_from_slice(b"COLIEXPT");
    put_u16(&mut header, 8, 1);
    put_u32(&mut header, 12, HEADER_BYTES as u32);'''
if rep_text not in t: raise SystemExit("streamed expert header anchor missing")
t = t.replace(rep_text, '''    header[..8].copy_from_slice(b"COLIEXPT");
    put_u16(&mut header, 8, 1);
    put_u16(&mut header, 10, u16::from(layout.is_some()));
    put_u32(&mut header, 12, HEADER_BYTES as u32);''', 1)
anchor = "        put_u16(&mut header, desc + 6, scale_id);\n        put_u64(&mut header, desc + 16, matrix.rows as u64);"
if anchor not in t: raise SystemExit("streamed expert descriptor anchor missing")
t = t.replace(anchor, "        put_u16(&mut header, desc + 6, scale_id);\n        put_u16(&mut header, desc + 12, layout.unwrap_or(0));\n        put_u64(&mut header, desc + 16, matrix.rows as u64);", 1)

tensor_sig = '''pub fn stream_exact_tensor<W: Write + Seek>(
    tensor: &source::TensorRef,
    output: &mut W,
) -> Result<(u32, u32)> {'''
if tensor_sig not in t: raise SystemExit("stream_exact_tensor signature missing")
t = t.replace(tensor_sig, '''pub fn stream_exact_tensor<W: Write + Seek>(tensor: &source::TensorRef, output: &mut W) -> Result<(u32, u32)> {
    stream_tensor_version(tensor, 0, output)
}

pub fn stream_target_tensor<W: Write + Seek>(tensor: &source::TensorRef, output: &mut W) -> Result<(u32, u32)> {
    stream_tensor_version(tensor, 1, output)
}

fn stream_tensor_version<W: Write + Seek>(tensor: &source::TensorRef, minor: u16, output: &mut W) -> Result<(u32, u32)> {''', 1)
tensor_header = '''    header[0..8].copy_from_slice(b"COLITENS");
    put_u16(&mut header, 8, 1);
    put_u32(&mut header, 12, TENSOR_HEADER_BYTES as u32);'''
if tensor_header not in t: raise SystemExit("streamed tensor header anchor missing")
t = t.replace(tensor_header, '''    header[0..8].copy_from_slice(b"COLITENS");
    put_u16(&mut header, 8, 1);
    put_u16(&mut header, 10, minor);
    put_u32(&mut header, 12, TENSOR_HEADER_BYTES as u32);''', 1)

start = t.find("fn expert_math_format(dtype: &str) -> Result<u16> {")
end = t.find("fn put_u16(buffer:", start)
if start < 0 or end < 0: raise SystemExit("old target semantic helpers missing")
t = t[:start] + '''fn math_format_for_semantics(format: MathFormat) -> u16 {
    match format { MathFormat::Fp8E4M3 => 0x10, MathFormat::MxFp4E2M1 => 0x20 }
}
fn scale_format_for_semantics(format: ScaleFormat) -> u16 {
    match format { ScaleFormat::Ue8m0 => 4 }
}
''' + t[end:]
p.write_text(t)

rep("colic/src/storage/mod.rs", "//! Deterministic physical record planning and artifact publication.\n", "//! Deterministic physical record planning and artifact publication.\n\npub mod v11;\n")
rep("colic/src/storage/mod.rs", '''pub fn encode_data_shard_header(
    shard_id: u32,
    file_bytes: u64,
    record_alignment: u64,
    source_fingerprint: [u8; 32],
) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {''', '''pub fn encode_data_shard_header(
    shard_id: u32, file_bytes: u64, record_alignment: u64, source_fingerprint: [u8; 32],
) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {
    encode_data_shard_header_version(shard_id, file_bytes, record_alignment, source_fingerprint, 0)
}
pub fn encode_data_shard_header_v11(
    shard_id: u32, file_bytes: u64, record_alignment: u64, source_fingerprint: [u8; 32],
) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {
    encode_data_shard_header_version(shard_id, file_bytes, record_alignment, source_fingerprint, 1)
}
fn encode_data_shard_header_version(
    shard_id: u32, file_bytes: u64, record_alignment: u64, source_fingerprint: [u8; 32], minor: u16,
) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {''')
rep("colic/src/storage/mod.rs", "    put_u16(&mut header, 10, 0);\n    put_u32(&mut header, 12, DATA_SHARD_HEADER_BYTES as u32);", "    put_u16(&mut header, 10, minor);\n    put_u32(&mut header, 12, DATA_SHARD_HEADER_BYTES as u32);", 1)
rep("colic/src/storage/mod.rs", "    source_fingerprint: [u8; 32],\n    file: File,", "    source_fingerprint: [u8; 32],\n    minor: u16,\n    file: File,")
rep("colic/src/storage/mod.rs", '''    pub fn create(
        path: &Path,
        shard_id: u32,
        alignment: u64,
        source_fingerprint: [u8; 32],
    ) -> Result<Self> {
        let file = File::create(path).map_err(|source| ColicError::Io {''', '''    pub fn create(path: &Path, shard_id: u32, alignment: u64, source_fingerprint: [u8; 32]) -> Result<Self> {
        Self::create_version(path, shard_id, alignment, source_fingerprint, 0)
    }
    pub fn create_v11(path: &Path, shard_id: u32, alignment: u64, source_fingerprint: [u8; 32]) -> Result<Self> {
        Self::create_version(path, shard_id, alignment, source_fingerprint, 1)
    }
    fn create_version(path: &Path, shard_id: u32, alignment: u64, source_fingerprint: [u8; 32], minor: u16) -> Result<Self> {
        let file = File::create(path).map_err(|source| ColicError::Io {''')
rep("colic/src/storage/mod.rs", "            source_fingerprint,\n            file,", "            source_fingerprint,\n            minor,\n            file,")
rep("colic/src/storage/mod.rs", '''        let header = encode_data_shard_header(
            self.shard_id,
            self.file_bytes,
            self.alignment,
            self.source_fingerprint,
        )?;''', '''        let header = encode_data_shard_header_version(
            self.shard_id, self.file_bytes, self.alignment, self.source_fingerprint, self.minor,
        )?;''')

rep("colic/src/pipeline.rs", '''fn stream_payload(
    writer: &mut storage::DataShardWriter,
    planned: &storage::PlannedRecord,
    source: &ExactSource,
) -> Result<ManifestRecord> {''', '''fn stream_payload(
    writer: &mut storage::DataShardWriter,
    planned: &storage::PlannedRecord,
    source: &ExactSource,
    profile: target::TargetProfile,
) -> Result<ManifestRecord> {''')
rep("colic/src/pipeline.rs", '''            writer.write_record_stream(planned, |file| {
                checksums = target::stream_exact_tensor(tensor, file)?;
                Ok(planned.record.stored_bytes)
            })?;''', '''            writer.write_record_stream(planned, |file| {
                checksums = target::stream_target_tensor(tensor, file)?;
                Ok(planned.record.stored_bytes)
            })?;''')
rep("colic/src/pipeline.rs", '''                math_format: target::math_format_for_dtype(&tensor.dtype)?,
                scale_format: 0,
                layout: 0,
                flags: 0b10,
                stored_crc32c: checksums.1,''', '''                math_format: target::math_format_for_dtype(&tensor.dtype)?,
                scale_format: 0,
                layout: target::identity::for_profile(profile)?.linear_layout,
                flags: 0b10,
                stored_crc32c: checksums.1,''')
rep("colic/src/pipeline.rs", "                crc = target::stream_exact_expert(expert, file)?;", "                crc = target::stream_target_expert(profile, expert, file)?;")
rep("colic/src/pipeline.rs", "            let mut writer = storage::DataShardWriter::create(\n", "            let mut writer = storage::DataShardWriter::create_v11(\n")
rep("colic/src/pipeline.rs", "                let manifest = stream_payload(&mut writer, planned, source)?;", "                let manifest = stream_payload(&mut writer, planned, source, target)?;")
rep("colic/src/pipeline.rs", '''        let manifest = storage::encode_manifest_with_records(
            &plan,
            target.name,
            fingerprint,
            &metadata,
            &header_crcs,
        )?;''', '''        let manifest = storage::v11::encode_manifest(
            &plan, target, fingerprint, &metadata, &header_crcs,
            storage::v11::ArtifactOptions::default(),
        )?;''')

rep("colic/src/verify.rs", '''    if u16_at(&manifest, 8)? != 1
        || u16_at(&manifest, 10)? != 0
        || u32_at(&manifest, 12)? != MANIFEST_HEADER_BYTES as u32
    {
        return invalid("manifest header size is invalid");
    }
    let manifest_flags = u32_at(&manifest, 16)?;
    if manifest_flags & 0xffff_0000 != 0 {
        return invalid("manifest contains unknown required feature flags");
    }''', '''    let minor = u16_at(&manifest, 10)?;
    if u16_at(&manifest, 8)? != 1 || !matches!(minor, 0 | 1) || u32_at(&manifest, 12)? != MANIFEST_HEADER_BYTES as u32 {
        return invalid("manifest header version/size is invalid");
    }
    let manifest_flags = u32_at(&manifest, 16)?;
    if minor == 0 && manifest_flags & 0xffff_0000 != 0 { return invalid("manifest contains unknown required feature flags"); }
    if minor == 1 {
        let required = (1 << 0) | (1 << 1) | (1 << 16);
        if manifest_flags & required != required || manifest_flags & !required != 0 { return invalid("v1.1 manifest target/fingerprint flags are invalid"); }
        let target = variable_region(&manifest, 184, 192, "target descriptor")?;
        if target.len() != 256 || &manifest[target.start..target.start + 8] != b"COLITGT\\0" { return invalid("v1.1 target descriptor is invalid"); }
        if storage::crc32c(&manifest[target.clone()]) != u32_at(&manifest, 232)? { return invalid("v1.1 target descriptor CRC32C does not match"); }
        if !manifest[200..232].iter().any(|byte| *byte != 0) { return invalid("v1.1 artifact fingerprint is zero"); }
    }''')
rep("colic/src/verify.rs", "            || u16_at(&header, 10)? != 0", "            || u16_at(&header, 10)? != minor")
rep("colic/src/verify.rs", "        let decoded = u64_at(&manifest, desc + 56)?;", "        let resident = u64_at(&manifest, desc + 56)?;")
rep("colic/src/verify.rs", "                decoded,\n                codec,", "                resident,\n                codec,", 1)
rep("colic/src/verify.rs", "                decoded,\n                layer,", "                resident,\n                layer,", 1)

rep("colic/src/main.rs", '''                println!("semantic_resident_tensors={}", model.resident_tensors.len());''', '''                println!("semantic_resident_tensors={}", model.resident_tensors.len());
                println!("semantic_unclassified_tensors={}", model.resident_tensors.len());
                println!("semantic_assets={}", model.assets.tokenizer.len() + model.assets.config.is_some() as usize);''')

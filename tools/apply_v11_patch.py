#!/usr/bin/env python3
from pathlib import Path


def replace(path: str, old: str, new: str, count: int = 1) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"missing anchor in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, count))


# Export the frozen target-identity registry to the compiler target module.
replace("colic/src/target/mod.rs", "//! Versioned target compatibility profiles and lowering boundary.\n", "//! Versioned target compatibility profiles and lowering boundary.\n\npub mod identity;\n")
replace("colic/src/target/mod.rs", "    ir::RoutedExpert,", "    ir::{MathFormat, RoutedExpert, ScaleFormat},")
replace("colic/src/target/mod.rs", 'name: "linux-x86_64-avx2-v1",', 'name: "linux-x86_64-cpu-avx2-v1",')

# CSF v1.1 typed envelopes. Legacy helper lowering remains useful to tests, but
# semantic math/scale fields are now sourced from the IR rather than dtype rediscovery.
text = Path("colic/src/target/mod.rs").read_text()
text = text.replace("put_u16(&mut payload, 10, 0);", "put_u16(&mut payload, 10, 1);")
text = text.replace("put_u16(&mut header, 10, 0);", "put_u16(&mut header, 10, 1);")
text = text.replace("expert_math_format(&matrix.source.dtype)?", "math_format_for_semantics(matrix.quantization.math_format)")
text = text.replace("scale_format(&scale.dtype)?", "scale_format_for_semantics(matrix.quantization.scale_format)")
text = text.replace(
    '        if matrix.source.dtype == "I8" {\n            put_u32(&mut payload, desc + 32, 1);\n            put_u32(&mut payload, desc + 36, 32);\n        }',
    '        put_u32(&mut payload, desc + 32, matrix.quantization.scale_block_rows);\n        put_u32(&mut payload, desc + 36, matrix.quantization.scale_block_columns);'
)
text = text.replace(
    '        if matrix.source.dtype == "I8" {\n            put_u32(&mut header, desc + 32, 1);\n            put_u32(&mut header, desc + 36, 32);\n        }',
    '        put_u32(&mut header, desc + 32, matrix.quantization.scale_block_rows);\n        put_u32(&mut header, desc + 36, matrix.quantization.scale_block_columns);'
)
# Refactor streamed expert lowering so production compilation stamps the concrete target layout.
old_sig = "pub fn stream_exact_expert<W: Write + Seek>(expert: &RoutedExpert, output: &mut W) -> Result<u32> {"
new_sig = '''pub fn stream_exact_expert<W: Write + Seek>(expert: &RoutedExpert, output: &mut W) -> Result<u32> {
    stream_expert_with_layout(expert, None, output)
}

pub fn stream_target_expert<W: Write + Seek>(
    profile: TargetProfile,
    expert: &RoutedExpert,
    output: &mut W,
) -> Result<u32> {
    let layout = identity::expert_layout(profile, expert)?;
    stream_expert_with_layout(expert, Some(layout), output)
}

fn stream_expert_with_layout<W: Write + Seek>(
    expert: &RoutedExpert,
    layout: Option<u16>,
    output: &mut W,
) -> Result<u32> {'''
if old_sig not in text:
    raise SystemExit("stream expert signature anchor missing")
text = text.replace(old_sig, new_sig, 1)
# Stamp target layout into matrix descriptor. The legacy helper leaves NONE for tooling-only tests.
anchor = "        put_u16(&mut header, desc + 6, scale_id);\n        put_u64(&mut header, desc + 16, matrix.rows as u64);"
if anchor not in text:
    raise SystemExit("expert descriptor anchor missing")
text = text.replace(anchor, "        put_u16(&mut header, desc + 6, scale_id);\n        put_u16(&mut header, desc + 12, layout.unwrap_or(0));\n        put_u64(&mut header, desc + 16, matrix.rows as u64);", 1)
# Replace obsolete dtype-derived expert semantic helpers.
start = text.index("fn expert_math_format(dtype: &str) -> Result<u16> {")
end = text.index("fn put_u16(buffer:", start)
text = text[:start] + '''fn math_format_for_semantics(format: MathFormat) -> u16 {
    match format {
        MathFormat::Fp8E4M3 => 0x10,
        MathFormat::MxFp4E2M1 => 0x20,
    }
}
fn scale_format_for_semantics(format: ScaleFormat) -> u16 {
    match format {
        ScaleFormat::Ue8m0 => 4,
    }
}
''' + text[end:]
Path("colic/src/target/mod.rs").write_text(text)

# Storage keeps the old v1.0 helpers for fixtures, while production uses v1.1.
replace("colic/src/storage/mod.rs", "//! Deterministic physical record planning and artifact publication.\n", "//! Deterministic physical record planning and artifact publication.\n\npub mod v11;\n")
# Add a versioned data header encoder and writer constructor without breaking old fixture tests.
replace(
    "colic/src/storage/mod.rs",
    "pub fn encode_data_shard_header(\n    shard_id: u32,\n    file_bytes: u64,\n    record_alignment: u64,\n    source_fingerprint: [u8; 32],\n) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {",
    "pub fn encode_data_shard_header(\n    shard_id: u32,\n    file_bytes: u64,\n    record_alignment: u64,\n    source_fingerprint: [u8; 32],\n) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {\n    encode_data_shard_header_version(shard_id, file_bytes, record_alignment, source_fingerprint, 0)\n}\n\npub fn encode_data_shard_header_v11(\n    shard_id: u32,\n    file_bytes: u64,\n    record_alignment: u64,\n    source_fingerprint: [u8; 32],\n) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {\n    encode_data_shard_header_version(shard_id, file_bytes, record_alignment, source_fingerprint, 1)\n}\n\nfn encode_data_shard_header_version(\n    shard_id: u32,\n    file_bytes: u64,\n    record_alignment: u64,\n    source_fingerprint: [u8; 32],\n    minor: u16,\n) -> Result<[u8; DATA_SHARD_HEADER_BYTES as usize]> {"
)
replace("colic/src/storage/mod.rs", "    put_u16(&mut header, 10, 0);\n    put_u32(&mut header, 12, DATA_SHARD_HEADER_BYTES as u32);", "    put_u16(&mut header, 10, minor);\n    put_u32(&mut header, 12, DATA_SHARD_HEADER_BYTES as u32);", 1)
replace("colic/src/storage/mod.rs", "    source_fingerprint: [u8; 32],\n    file: File,", "    source_fingerprint: [u8; 32],\n    minor: u16,\n    file: File,")
replace(
    "colic/src/storage/mod.rs",
    "    pub fn create(\n        path: &Path,\n        shard_id: u32,\n        alignment: u64,\n        source_fingerprint: [u8; 32],\n    ) -> Result<Self> {\n        let file = File::create(path).map_err(|source| ColicError::Io {",
    "    pub fn create(\n        path: &Path,\n        shard_id: u32,\n        alignment: u64,\n        source_fingerprint: [u8; 32],\n    ) -> Result<Self> {\n        Self::create_version(path, shard_id, alignment, source_fingerprint, 0)\n    }\n\n    pub fn create_v11(\n        path: &Path,\n        shard_id: u32,\n        alignment: u64,\n        source_fingerprint: [u8; 32],\n    ) -> Result<Self> {\n        Self::create_version(path, shard_id, alignment, source_fingerprint, 1)\n    }\n\n    fn create_version(\n        path: &Path,\n        shard_id: u32,\n        alignment: u64,\n        source_fingerprint: [u8; 32],\n        minor: u16,\n    ) -> Result<Self> {\n        let file = File::create(path).map_err(|source| ColicError::Io {"
)
replace("colic/src/storage/mod.rs", "            source_fingerprint,\n            file,", "            source_fingerprint,\n            minor,\n            file,")
replace(
    "colic/src/storage/mod.rs",
    "        let header = encode_data_shard_header(\n            self.shard_id,\n            self.file_bytes,\n            self.alignment,\n            self.source_fingerprint,\n        )?;",
    "        let header = encode_data_shard_header_version(\n            self.shard_id,\n            self.file_bytes,\n            self.alignment,\n            self.source_fingerprint,\n            self.minor,\n        )?;"
)

# Pipeline: target-native layouts and v1.1 manifest/artifact identity.
replace("colic/src/pipeline.rs", "fn stream_payload(\n    writer: &mut storage::DataShardWriter,\n    planned: &storage::PlannedRecord,\n    source: &ExactSource,\n) -> Result<ManifestRecord> {", "fn stream_payload(\n    writer: &mut storage::DataShardWriter,\n    planned: &storage::PlannedRecord,\n    source: &ExactSource,\n    profile: target::TargetProfile,\n) -> Result<ManifestRecord> {")
replace("colic/src/pipeline.rs", "                layout: 0,", "                layout: target::identity::for_profile(profile)?.linear_layout,", 1)
replace("colic/src/pipeline.rs", "                crc = target::stream_exact_expert(expert, file)?;", "                crc = target::stream_target_expert(profile, expert, file)?;")
# The production stream expert outer record is MIXED; leave tooling-only lower_exact_payloads untouched.
stream_section = Path("colic/src/pipeline.rs").read_text()
stream_anchor = "                layout: 0xfffe,\n                flags: 0,\n                stored_crc32c: crc,"
if stream_anchor not in stream_section:
    raise SystemExit("stream expert manifest anchor missing")
# Only the last matching expert literal is in stream_payload; replace last occurrence safely.
pos = stream_section.rfind(stream_anchor)
stream_section = stream_section[:pos] + stream_section[pos:].replace(stream_anchor, stream_anchor, 1)
Path("colic/src/pipeline.rs").write_text(stream_section)
replace("colic/src/pipeline.rs", "            let mut writer = storage::DataShardWriter::create(\n", "            let mut writer = storage::DataShardWriter::create_v11(\n")
replace("colic/src/pipeline.rs", "                let manifest = stream_payload(&mut writer, planned, source)?;", "                let manifest = stream_payload(&mut writer, planned, source, target)?;")
replace(
    "colic/src/pipeline.rs",
    "        let manifest = storage::encode_manifest_with_records(\n            &plan,\n            target.name,\n            fingerprint,\n            &metadata,\n            &header_crcs,\n        )?;",
    "        let manifest = storage::v11::encode_manifest(\n            &plan,\n            target,\n            fingerprint,\n            &metadata,\n            &header_crcs,\n            storage::v11::ArtifactOptions::default(),\n        )?;"
)

# Rust verifier: accept target v1.1 and require its target identity framing.
replace(
    "colic/src/verify.rs",
    "    if u16_at(&manifest, 8)? != 1\n        || u16_at(&manifest, 10)? != 0\n        || u32_at(&manifest, 12)? != MANIFEST_HEADER_BYTES as u32\n    {\n        return invalid(\"manifest header size is invalid\");\n    }\n    let manifest_flags = u32_at(&manifest, 16)?;\n    if manifest_flags & 0xffff_0000 != 0 {\n        return invalid(\"manifest contains unknown required feature flags\");\n    }",
    "    let minor = u16_at(&manifest, 10)?;\n    if u16_at(&manifest, 8)? != 1\n        || !matches!(minor, 0 | 1)\n        || u32_at(&manifest, 12)? != MANIFEST_HEADER_BYTES as u32\n    {\n        return invalid(\"manifest header version/size is invalid\");\n    }\n    let manifest_flags = u32_at(&manifest, 16)?;\n    if minor == 0 && manifest_flags & 0xffff_0000 != 0 {\n        return invalid(\"manifest contains unknown required feature flags\");\n    }\n    if minor == 1 {\n        let required = (1 << 0) | (1 << 1) | (1 << 16);\n        if manifest_flags & required != required || manifest_flags & !required != 0 {\n            return invalid(\"v1.1 manifest target/fingerprint flags are invalid\");\n        }\n        let target = variable_region(&manifest, 184, 192, \"target descriptor\")?;\n        if target.len() != 256 || &manifest[target.start..target.start + 8] != b\"COLITGT\\0\" {\n            return invalid(\"v1.1 target descriptor is invalid\");\n        }\n        if storage::crc32c(&manifest[target.clone()]) != u32_at(&manifest, 232)? {\n            return invalid(\"v1.1 target descriptor CRC32C does not match\");\n        }\n        if !manifest[200..232].iter().any(|byte| *byte != 0) {\n            return invalid(\"v1.1 artifact fingerprint is zero\");\n        }\n    }"
)
replace("colic/src/verify.rs", "            || u16_at(&header, 10)? != 0", "            || u16_at(&header, 10)? != minor")
# Rename local decoded terminology to resident semantics only where it is read from record table.
replace("colic/src/verify.rs", "        let decoded = u64_at(&manifest, desc + 56)?;", "        let resident = u64_at(&manifest, desc + 56)?;")
replace("colic/src/verify.rs", "                decoded,\n                codec,", "                resident,\n                codec,", 1)
replace("colic/src/verify.rs", "                decoded,\n                layer,", "                resident,\n                layer,", 1)

# inspect-source should expose the strict classification invariant.
replace(
    "colic/src/main.rs",
    '                    println!("resident_tensors={}", model.resident_tensors.len());',
    '                    println!("resident_tensors={}", model.resident_tensors.len());\n                    println!("semantic_unclassified_tensors={}", model.resident_tensors.len());\n                    println!("semantic_assets={}", model.assets.tokenizer.len() + usize::from(model.assets.config.is_some()));'
)

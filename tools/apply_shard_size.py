#!/usr/bin/env python3
from pathlib import Path


def rep(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"missing anchor in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))


# Compile request and planner ----------------------------------------------------
rep("colic/src/pipeline.rs", "    pub force: bool,\n}", "    pub force: bool,\n    pub shard_size_bytes: u64,\n}")
rep("colic/src/pipeline.rs", "            force: false,\n        }", "            force: false,\n            shard_size_bytes: 4 * 1024 * 1024 * 1024,\n        }")
rep("colic/src/pipeline.rs", "    let plan = storage::plan_records(&records, target, 4 * 1024 * 1024 * 1024)?;", "    let plan = storage::plan_records(&records, target, request.shard_size_bytes)?;",)
rep("colic/src/pipeline.rs", "    let plan = storage::plan_records(&records, target, 4 * 1024 * 1024 * 1024)?;", "    let plan = storage::plan_records(&records, target, request.shard_size_bytes)?;",)
rep(
    "colic/src/pipeline.rs",
    "    if !matches!(request.quant, QuantRequest::Exact) {",
    "    if request.shard_size_bytes == 0 {\n        return Err(ColicError::Usage(\"shard size must be greater than zero\".into()));\n    }\n    if !matches!(request.quant, QuantRequest::Exact) {",
)

# CLI ---------------------------------------------------------------------------
rep(
    "colic/src/cli.rs",
    'pub const USAGE: &str = "Usage:\\n  colic inspect-source MODEL_DIR\\n  colic verify PACKAGE_DIR\\n  colic compile MODEL_DIR --target native|PROFILE --quant exact|PROFILE --codec none|auto|PROFILE --opt default|size|latency -o OUTPUT [--dry-run] [--verify] [--force]";',
    'pub const USAGE: &str = "Usage:\\n  colic inspect-source MODEL_DIR\\n  colic verify PACKAGE_DIR\\n  colic compile MODEL_DIR --target native|PROFILE --quant exact|PROFILE --codec none|auto|PROFILE --opt default|size|latency -o OUTPUT [--shard-size-gb N|--shard-size-mib N] [--dry-run] [--verify] [--force]";',
)
rep(
    "colic/src/cli.rs",
    '''            "--dry-run" => request.dry_run = true,
            "--verify" => request.verify = true,
            "--force" => request.force = true,''',
    '''            "--shard-size-gb" => {
                request.shard_size_bytes = parse_shard_size(&value(&mut args, "--shard-size-gb")?, 1024 * 1024 * 1024)?
            }
            "--shard-size-mib" => {
                request.shard_size_bytes = parse_shard_size(&value(&mut args, "--shard-size-mib")?, 1024 * 1024)?
            }
            "--dry-run" => request.dry_run = true,
            "--verify" => request.verify = true,
            "--force" => request.force = true,''',
)
rep(
    "colic/src/cli.rs",
    "    Ok(Command::Compile(request))\n}\n\n#[cfg(test)]",
    '''    Ok(Command::Compile(request))
}

fn parse_shard_size(value: &str, unit_bytes: u64) -> Result<u64> {
    let units: u64 = value
        .parse()
        .map_err(|_| ColicError::Usage(format!("invalid shard size `{value}`")))?;
    if units == 0 {
        return Err(ColicError::Usage("shard size must be greater than zero".into()));
    }
    units
        .checked_mul(unit_bytes)
        .ok_or_else(|| ColicError::Usage("shard size overflows u64".into()))
}

#[cfg(test)]''',
)
rep(
    "colic/src/cli.rs",
    '''    #[test]
    fn portable_target_is_rejected() {''',
    '''    #[test]
    fn parses_explicit_shard_size() {
        let Command::Compile(request) = parse(
            ["compile", "fixture", "--shard-size-mib", "2", "--dry-run"].map(str::to_owned),
        )
        .unwrap()
        else {
            panic!("expected compile")
        };
        assert_eq!(request.shard_size_bytes, 2 * 1024 * 1024);
        assert!(parse(
            ["compile", "fixture", "--shard-size-gb", "0", "--dry-run"].map(str::to_owned)
        )
        .is_err());
    }

    #[test]
    fn portable_target_is_rejected() {''',
)

# Artifact identity/profile_data ------------------------------------------------
rep(
    "colic/src/storage/v11.rs",
    '''pub struct ArtifactOptions<'a> {
    pub compiler: &'a str,
    pub quant_profile: &'a str,
    pub storage_profile: &'a str,
    pub optimization_profile: &'a str,
}''',
    '''pub struct ArtifactOptions<'a> {
    pub compiler: &'a str,
    pub quant_profile: &'a str,
    pub storage_profile: &'a str,
    pub optimization_profile: &'a str,
    pub profile_data: &'a [u8],
}''',
)
rep(
    "colic/src/storage/v11.rs",
    '''            storage_profile: "none",
            optimization_profile: "default",
        }''',
    '''            storage_profile: "none",
            optimization_profile: "default",
            profile_data: &[],
        }''',
)
rep(
    "colic/src/storage/v11.rs",
    '''    hasher.update([0]);
    hasher.update([0_u8; 32]);
    hasher.update(0_u64.to_le_bytes());
    hasher.update(Sha256::digest([]));''',
    '''    hasher.update([0]);
    hasher.update([0_u8; 32]);
    let profile_data_bytes: u64 = options
        .profile_data
        .len()
        .try_into()
        .map_err(|_| ColicError::Usage("profile data exceeds u64".into()))?;
    hasher.update(profile_data_bytes.to_le_bytes());
    hasher.update(Sha256::digest(options.profile_data));''',
)
rep(
    "colic/src/storage/v11.rs",
    '''    let manifest_bytes = string_table_offset
        .checked_add(string_table_bytes)
        .ok_or_else(|| ColicError::Usage("manifest size overflows u64".into()))?;''',
    '''    let string_table_end = string_table_offset
        .checked_add(string_table_bytes)
        .ok_or_else(|| ColicError::Usage("manifest size overflows u64".into()))?;
    let profile_data_bytes: u64 = options
        .profile_data
        .len()
        .try_into()
        .map_err(|_| ColicError::Usage("profile data exceeds u64".into()))?;
    let profile_data_offset = if profile_data_bytes == 0 {
        0
    } else {
        align_up(string_table_end, 16)?
    };
    let manifest_bytes = if profile_data_bytes == 0 {
        string_table_end
    } else {
        profile_data_offset
            .checked_add(profile_data_bytes)
            .ok_or_else(|| ColicError::Usage("profile data span overflows u64".into()))?
    };''',
)
rep(
    "colic/src/storage/v11.rs",
    '''    put_u32(target_desc, 108, triple_id);
    put_u32(target_desc, 164, semantic_id);''',
    '''    put_u32(target_desc, 108, triple_id);
    put_u64(target_desc, 144, profile_data_offset);
    put_u64(target_desc, 152, profile_data_bytes);
    if profile_data_bytes != 0 {
        put_u32(target_desc, 160, crc32c(options.profile_data));
    }
    put_u32(target_desc, 164, semantic_id);''',
)
rep(
    "colic/src/storage/v11.rs",
    '''    let artifact = artifact_fingerprint(source_fingerprint, profile, options)?;
    manifest[200..232].copy_from_slice(&artifact);''',
    '''    if profile_data_bytes != 0 {
        let start: usize = profile_data_offset
            .try_into()
            .map_err(|_| ColicError::Usage("profile data offset exceeds usize".into()))?;
        manifest[start..start + options.profile_data.len()].copy_from_slice(options.profile_data);
    }

    let artifact = artifact_fingerprint(source_fingerprint, profile, options)?;
    manifest[200..232].copy_from_slice(&artifact);''',
)
rep(
    "colic/src/storage/v11.rs",
    '''fn hash_string(hasher: &mut Sha256, value: &str) -> Result<()> {''',
    '''pub fn storage_profile_data(shard_size_bytes: u64) -> [u8; 24] {
    let mut data = [0_u8; 24];
    data[..16].copy_from_slice(b"COLI-STORAGE-V1\\0");
    data[16..24].copy_from_slice(&shard_size_bytes.to_le_bytes());
    data
}

fn hash_string(hasher: &mut Sha256, value: &str) -> Result<()> {''',
)
rep(
    "colic/src/storage/v11.rs",
    '''        assert_eq!(a, b);
        assert_ne!(a, [0; 32]);''',
    '''        assert_eq!(a, b);
        assert_ne!(a, [0; 32]);
        let profile_data = storage_profile_data(1024 * 1024);
        let changed = artifact_fingerprint(
            [7; 32],
            profile,
            ArtifactOptions {
                profile_data: &profile_data,
                ..ArtifactOptions::default()
            },
        )
        .unwrap();
        assert_ne!(a, changed);''',
)

# Production pipeline includes the structural option in profile_data ------------
rep(
    "colic/src/pipeline.rs",
    '''        let manifest = storage::v11::encode_manifest(
            &plan,
            target,
            fingerprint,
            &metadata,
            &header_crcs,
            storage::v11::ArtifactOptions::default(),
        )?;''',
    '''        let profile_data = storage::v11::storage_profile_data(request.shard_size_bytes);
        let manifest = storage::v11::encode_manifest(
            &plan,
            target,
            fingerprint,
            &metadata,
            &header_crcs,
            storage::v11::ArtifactOptions {
                profile_data: &profile_data,
                ..storage::v11::ArtifactOptions::default()
            },
        )?;''',
)

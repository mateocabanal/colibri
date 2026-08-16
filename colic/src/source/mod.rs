use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{ColicError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceInventory {
    pub root: PathBuf,
    pub files: Vec<PathBuf>,
    pub tensors: BTreeMap<String, TensorRef>,
    pub source_stored_bytes: u64,
    pub dtype_counts: BTreeMap<String, u64>,
    pub source_fingerprint: String,
    pub config_fingerprint: Option<String>,
    pub architecture_hint: Option<String>,
}

/// A seekable source tensor. `offset` and `len` address only the tensor payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRef {
    pub source: PathBuf,
    pub offset: u64,
    pub len: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryProgress {
    pub completed_files: usize,
    pub total_files: usize,
    pub path: PathBuf,
    pub bytes_hashed: u64,
}

pub fn fingerprint_bytes(hex: &str) -> Result<[u8; 32]> {
    if hex.len() != 64 {
        return Err(ColicError::Usage(
            "source fingerprint must contain exactly 64 hexadecimal characters".into(),
        ));
    }
    let mut bytes = [0_u8; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let encoded = &hex[index * 2..index * 2 + 2];
        *byte = u8::from_str_radix(encoded, 16).map_err(|_| {
            ColicError::Usage("source fingerprint contains non-hexadecimal characters".into())
        })?;
    }
    Ok(bytes)
}

pub fn discover(root: &Path) -> Result<SourceInventory> {
    discover_impl(root, None)
}

pub fn discover_with_progress(
    root: &Path,
    progress: &mut dyn FnMut(DiscoveryProgress),
) -> Result<SourceInventory> {
    discover_impl(root, Some(progress))
}

fn discover_impl(
    root: &Path,
    progress: Option<&mut dyn FnMut(DiscoveryProgress)>,
) -> Result<SourceInventory> {
    if !root.exists() {
        return Err(ColicError::SourceNotFound(root.to_owned()));
    }
    if !root.is_dir() {
        return Err(ColicError::Usage(format!(
            "source model must be a directory: {}",
            root.display()
        )));
    }
    let entries = fs::read_dir(root).map_err(|source| ColicError::Io {
        path: root.to_owned(),
        source,
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| ColicError::Io {
            path: root.to_owned(),
            source,
        })?;
        if entry
            .file_type()
            .map_err(|source| ColicError::Io {
                path: entry.path(),
                source,
            })?
            .is_file()
        {
            files.push(entry.path());
        }
    }
    files.sort();
    let shards = discover_shards(root, &files)?;
    let mut tensors = BTreeMap::new();
    for shard in &shards {
        for (name, tensor) in parse_safetensors(shard)? {
            if tensors.insert(name.clone(), tensor).is_some() {
                return invalid(shard, format!("duplicate tensor `{name}` across shards"));
            }
        }
    }
    let source_stored_bytes = tensors.values().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(tensor.len)
            .ok_or_else(|| ColicError::InvalidSource {
                path: root.to_owned(),
                detail: "tensor byte total overflows u64".into(),
            })
    })?;
    let mut dtype_counts = BTreeMap::new();
    for tensor in tensors.values() {
        *dtype_counts.entry(tensor.dtype.clone()).or_default() += 1;
    }
    Ok(SourceInventory {
        root: root.to_owned(),
        files,
        source_stored_bytes,
        dtype_counts,
        source_fingerprint: fingerprint(root, &shards, progress)?,
        config_fingerprint: config_fingerprint(root)?,
        architecture_hint: architecture_hint(root)?,
        tensors,
    })
}

/// Reads a bounded range of a tensor without materializing its entire shard.
pub fn read_range(tensor: &TensorRef, range: Range<u64>, dst: &mut [u8]) -> Result<()> {
    let range_len =
        range
            .end
            .checked_sub(range.start)
            .ok_or_else(|| ColicError::InvalidSource {
                path: tensor.source.clone(),
                detail: "range end precedes start".into(),
            })?;
    if range.end > tensor.len || range_len != dst.len() as u64 {
        return invalid(
            &tensor.source,
            "range is outside tensor bounds or destination has the wrong length",
        );
    }
    let offset =
        tensor
            .offset
            .checked_add(range.start)
            .ok_or_else(|| ColicError::InvalidSource {
                path: tensor.source.clone(),
                detail: "tensor range offset overflows u64".into(),
            })?;
    let mut file = File::open(&tensor.source).map_err(|source| ColicError::Io {
        path: tensor.source.clone(),
        source,
    })?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| ColicError::Io {
            path: tensor.source.clone(),
            source,
        })?;
    file.read_exact(dst).map_err(|source| ColicError::Io {
        path: tensor.source.clone(),
        source,
    })
}

fn discover_shards(root: &Path, files: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let index = root.join("model.safetensors.index.json");
    if index.is_file() {
        let value = parse_json_file(&index)?;
        let weight_map = value
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| ColicError::InvalidSource {
                path: index.clone(),
                detail: "missing object `weight_map`".into(),
            })?;
        let mut names = BTreeSet::new();
        for shard in weight_map.values() {
            let Some(shard) = shard.as_str() else {
                return invalid(&index, "weight_map value is not a string");
            };
            let shard_path = root.join(shard);
            if !shard_path.is_file() {
                return Err(ColicError::SourceNotFound(shard_path));
            }
            names.insert(shard_path);
        }
        return Ok(names.into_iter().collect());
    }
    let shards: Vec<_> = files
        .iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .cloned()
        .collect();
    if shards.is_empty() {
        return invalid(
            root,
            "no .safetensors shard or model.safetensors.index.json found",
        );
    }
    Ok(shards)
}

fn parse_safetensors(path: &Path) -> Result<BTreeMap<String, TensorRef>> {
    let mut file = File::open(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?
        .len();
    if file_len < 8 {
        return invalid(path, "file is shorter than its 8-byte header length");
    }
    let mut header_len = [0_u8; 8];
    file.read_exact(&mut header_len)
        .map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
    let header_len = u64::from_le_bytes(header_len);
    const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
    if header_len > MAX_HEADER_BYTES {
        return invalid(path, "header exceeds 64 MiB compiler safety limit");
    }
    let payload_start = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| ColicError::InvalidSource {
            path: path.to_owned(),
            detail: "header offset overflows u64".into(),
        })?;
    if payload_start > file_len {
        return invalid(path, "header extends beyond end of file");
    }
    let header_len: usize = header_len
        .try_into()
        .map_err(|_| ColicError::InvalidSource {
            path: path.to_owned(),
            detail: "header length cannot fit memory address space".into(),
        })?;
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)
        .map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
    let value: Value =
        serde_json::from_slice(&header).map_err(|source| ColicError::InvalidSource {
            path: path.to_owned(),
            detail: format!("invalid safetensors JSON header: {source}"),
        })?;
    let object = value.as_object().ok_or_else(|| ColicError::InvalidSource {
        path: path.to_owned(),
        detail: "safetensors header is not a JSON object".into(),
    })?;
    let mut tensor_spans = Vec::new();
    let mut tensors = BTreeMap::new();
    for (name, descriptor) in object {
        if name == "__metadata__" {
            continue;
        }
        let descriptor = descriptor
            .as_object()
            .ok_or_else(|| ColicError::InvalidSource {
                path: path.to_owned(),
                detail: format!("tensor `{name}` descriptor is not an object"),
            })?;
        let dtype = descriptor
            .get("dtype")
            .and_then(Value::as_str)
            .filter(|dtype| known_dtype(dtype))
            .ok_or_else(|| ColicError::InvalidSource {
                path: path.to_owned(),
                detail: format!("tensor `{name}` has unsupported or missing dtype"),
            })?
            .to_owned();
        let shape = descriptor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| ColicError::InvalidSource {
                path: path.to_owned(),
                detail: format!("tensor `{name}` has missing shape"),
            })?
            .iter()
            .map(|value| {
                value.as_u64().ok_or_else(|| ColicError::InvalidSource {
                    path: path.to_owned(),
                    detail: format!("tensor `{name}` has invalid shape dimension"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| ColicError::InvalidSource {
                path: path.to_owned(),
                detail: format!("tensor `{name}` has invalid data_offsets"),
            })?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| ColicError::InvalidSource {
                path: path.to_owned(),
                detail: format!("tensor `{name}` has invalid start offset"),
            })?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| ColicError::InvalidSource {
                path: path.to_owned(),
                detail: format!("tensor `{name}` has invalid end offset"),
            })?;
        if end < start {
            return invalid(path, format!("tensor `{name}` end offset precedes start"));
        }
        let absolute_end =
            payload_start
                .checked_add(end)
                .ok_or_else(|| ColicError::InvalidSource {
                    path: path.to_owned(),
                    detail: format!("tensor `{name}` offset overflows u64"),
                })?;
        if absolute_end > file_len {
            return invalid(path, format!("tensor `{name}` extends beyond file"));
        }
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or_else(|| ColicError::InvalidSource {
                    path: path.to_owned(),
                    detail: format!("tensor `{name}` element count overflows u64"),
                })
        })?;
        let expected_bytes =
            elements
                .checked_mul(dtype_size(&dtype))
                .ok_or_else(|| ColicError::InvalidSource {
                    path: path.to_owned(),
                    detail: format!("tensor `{name}` byte count overflows u64"),
                })?;
        if end - start != expected_bytes {
            return invalid(
                path,
                format!("tensor `{name}` span does not match its dtype and shape"),
            );
        }
        tensor_spans.push((start, end, name));
        tensors.insert(
            name.clone(),
            TensorRef {
                source: path.to_owned(),
                offset: payload_start + start,
                len: end - start,
                dtype,
                shape,
            },
        );
    }
    tensor_spans.sort_unstable_by_key(|(start, _, _)| *start);
    for pair in tensor_spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return invalid(
                path,
                format!("tensor ranges `{}` and `{}` overlap", pair[0].2, pair[1].2),
            );
        }
    }
    Ok(tensors)
}

fn fingerprint(
    root: &Path,
    shards: &[PathBuf],
    mut progress: Option<&mut dyn FnMut(DiscoveryProgress)>,
) -> Result<String> {
    let mut entries: Vec<(u8, PathBuf)> = shards.iter().cloned().map(|path| (1, path)).collect();
    let index = root.join("model.safetensors.index.json");
    if index.is_file() {
        entries.push((2, index));
    }
    entries.extend(source_assets(root).into_iter().map(|path| (3, path)));
    entries.sort_unstable_by(|left, right| {
        relative_path(root, &left.1).cmp(&relative_path(root, &right.1))
    });
    let count: u32 = entries
        .len()
        .try_into()
        .map_err(|_| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: "too many source files to fingerprint".into(),
        })?;
    let mut hasher = Sha256::new();
    hasher.update(b"COLI-SOURCE-V1\0");
    hasher.update(count.to_le_bytes());
    let total_files = entries.len();
    for (index, (kind, path)) in entries.into_iter().enumerate() {
        let relative = relative_path(root, &path);
        let path_len: u32 = relative
            .len()
            .try_into()
            .map_err(|_| ColicError::InvalidSource {
                path: path.clone(),
                detail: "source path is too long for fingerprint".into(),
            })?;
        let (file_bytes, content_hash) = file_sha256(&path)?;
        hasher.update([kind]);
        hasher.update(path_len.to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update(file_bytes.to_le_bytes());
        hasher.update(content_hash);
        if let Some(progress) = progress.as_deref_mut() {
            progress(DiscoveryProgress {
                completed_files: index + 1,
                total_files,
                path,
                bytes_hashed: file_bytes,
            });
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn file_sha256(path: &Path) -> Result<(u64, [u8; 32])> {
    let mut file = File::open(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut byte_count = 0_u64;
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut chunk).map_err(|source| ColicError::Io {
            path: path.to_owned(),
            source,
        })?;
        if read == 0 {
            break;
        }
        byte_count =
            byte_count
                .checked_add(read as u64)
                .ok_or_else(|| ColicError::InvalidSource {
                    path: path.to_owned(),
                    detail: "source file size overflows u64".into(),
                })?;
        hasher.update(&chunk[..read]);
    }
    Ok((byte_count, hasher.finalize().into()))
}

fn source_assets(root: &Path) -> Vec<PathBuf> {
    const ASSETS: &[&str] = &[
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ];
    ASSETS
        .iter()
        .map(|name| root.join(name))
        .filter(|path| path.is_file())
        .collect()
}

fn config_bytes(root: &Path) -> Result<Option<Vec<u8>>> {
    let path = root.join("config.json");
    if !path.is_file() {
        return Ok(None);
    }
    fs::read(&path)
        .map(Some)
        .map_err(|source| ColicError::Io { path, source })
}

fn config_fingerprint(root: &Path) -> Result<Option<String>> {
    let Some(config) = config_bytes(root)? else {
        return Ok(None);
    };
    Ok(Some(format!("{:x}", Sha256::digest(config))))
}

fn architecture_hint(root: &Path) -> Result<Option<String>> {
    let path = root.join("config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let config = parse_json_file(&path)?;
    let architecture = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(Value::as_str)
        .or_else(|| config.get("model_type").and_then(Value::as_str));
    Ok(architecture.map(str::to_owned))
}

fn parse_json_file(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|source| ColicError::Io {
        path: path.to_owned(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| ColicError::InvalidSource {
        path: path.to_owned(),
        detail: format!("invalid JSON: {source}"),
    })
}

/// Loads the optional Hugging Face config without touching tensor payloads.
pub fn config(root: &Path) -> Result<Option<Value>> {
    let path = root.join("config.json");
    if !path.is_file() {
        return Ok(None);
    }
    parse_json_file(&path).map(Some)
}

fn known_dtype(dtype: &str) -> bool {
    matches!(
        dtype,
        "BOOL"
            | "U8"
            | "I8"
            | "I16"
            | "U16"
            | "I32"
            | "U32"
            | "I64"
            | "U64"
            | "F16"
            | "BF16"
            | "F32"
            | "F64"
            | "F8_E4M3"
            | "F8_E4M3FN"
            | "F8_E4M3FNUZ"
            | "F8_E5M2"
            | "F8_E5M2FNUZ"
            | "F8_E8M0"
            | "F8_E8M0FNU"
    )
}

fn dtype_size(dtype: &str) -> u64 {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E4M3FN" | "F8_E4M3FNUZ" | "F8_E5M2"
        | "F8_E5M2FNUZ" | "F8_E8M0" | "F8_E8M0FNU" => 1,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => unreachable!("known_dtype must be checked before dtype_size"),
    }
}

fn invalid<T>(path: &Path, detail: impl Into<String>) -> Result<T> {
    Err(ColicError::InvalidSource {
        path: path.to_owned(),
        detail: detail.into(),
    })
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "colic-source-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_shard(path: &Path, entries: &[(&str, &str, &[u64], u64, u64)], payload: &[u8]) {
        let descriptors = entries.iter().map(|(name, dtype, shape, start, end)| {
            let shape = shape.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
            format!(r#""{name}":{{"dtype":"{dtype}","shape":[{shape}],"data_offsets":[{start},{end}]}}"#)
        }).collect::<Vec<_>>().join(",");
        let header = format!("{{{descriptors}}}");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn inventories_seekable_tensors_and_hashes_source_bytes() {
        let root = temp_dir("valid");
        fs::write(root.join("config.json"), r#"{"architectures":["TinyMoE"]}"#).unwrap();
        let shard = root.join("model.safetensors");
        write_shard(
            &shard,
            &[("alpha", "U8", &[4], 0, 4), ("beta", "F16", &[2], 4, 8)],
            &[1, 2, 3, 4, 5, 6, 7, 8],
        );

        let inventory = discover(&root).unwrap();
        assert_eq!(inventory.architecture_hint.as_deref(), Some("TinyMoE"));
        assert_eq!(inventory.tensors.len(), 2);
        assert_eq!(inventory.source_stored_bytes, 8);
        assert_eq!(inventory.dtype_counts.get("U8"), Some(&1));
        let mut bytes = [0; 2];
        read_range(inventory.tensors.get("beta").unwrap(), 1..3, &mut bytes).unwrap();
        assert_eq!(bytes, [6, 7]);
        let second = discover(&root).unwrap();
        assert_eq!(inventory.source_fingerprint, second.source_fingerprint);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_overlapping_tensor_ranges() {
        let root = temp_dir("overlap");
        write_shard(
            &root.join("model.safetensors"),
            &[("left", "U8", &[4], 0, 4), ("right", "U8", &[4], 3, 7)],
            &[0; 7],
        );
        assert!(matches!(
            discover(&root),
            Err(ColicError::InvalidSource { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn indexed_shards_are_sorted_independent_of_weight_map_order() {
        let root = temp_dir("indexed");
        write_shard(
            &root.join("z.safetensors"),
            &[("z", "U8", &[1], 0, 1)],
            &[9],
        );
        write_shard(
            &root.join("a.safetensors"),
            &[("a", "U8", &[1], 0, 1)],
            &[1],
        );
        fs::write(
            root.join("model.safetensors.index.json"),
            r#"{"weight_map":{"z":"z.safetensors","a":"a.safetensors"}}"#,
        )
        .unwrap();
        let inventory = discover(&root).unwrap();
        assert_eq!(
            inventory.tensors.keys().cloned().collect::<Vec<_>>(),
            vec!["a", "z"]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fails_closed_for_missing_shard_and_malformed_ranges() {
        let root = temp_dir("invalid");
        assert!(matches!(
            discover(&root),
            Err(ColicError::InvalidSource { .. })
        ));

        write_shard(
            &root.join("model.safetensors"),
            &[("shape_mismatch", "F16", &[2], 0, 2)],
            &[0; 2],
        );
        assert!(matches!(
            discover(&root),
            Err(ColicError::InvalidSource { .. })
        ));

        write_shard(
            &root.join("model.safetensors"),
            &[("outside", "U8", &[8], 0, 8)],
            &[0; 4],
        );
        assert!(matches!(
            discover(&root),
            Err(ColicError::InvalidSource { .. })
        ));

        write_shard(
            &root.join("model.safetensors"),
            &[("bad_dtype", "NOT_A_DTYPE", &[1], 0, 1)],
            &[0],
        );
        assert!(matches!(
            discover(&root),
            Err(ColicError::InvalidSource { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fingerprint_covers_tokenizer_assets_and_tensor_payload() {
        let root = temp_dir("fingerprint");
        let shard = root.join("model.safetensors");
        write_shard(&shard, &[("weight", "U8", &[2], 0, 2)], &[1, 2]);
        fs::write(root.join("tokenizer_config.json"), "first").unwrap();
        let initial = discover(&root).unwrap().source_fingerprint;
        fs::write(root.join("tokenizer_config.json"), "second").unwrap();
        assert_ne!(initial, discover(&root).unwrap().source_fingerprint);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_fingerprint_hex_decodes_strictly() {
        assert_eq!(fingerprint_bytes(&"ab".repeat(32)).unwrap(), [0xab; 32]);
        assert!(fingerprint_bytes("short").is_err());
        assert!(fingerprint_bytes(&format!("{}zz", "00".repeat(31))).is_err());
    }
}

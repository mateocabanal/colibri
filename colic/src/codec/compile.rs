use std::{collections::BTreeMap, fs, io::Read};

use crate::{
    codec::{apple8, rans256},
    error::{ColicError, Result},
    ir::{Architecture, SemanticModel},
    pipeline::{
        CodecRequest, CompileRequest, DryRunSummary, OptimizationProfile, ProgressSink, QuantRequest,
        Stage,
    },
    source::{self, TensorRef},
    storage::{self, LoweredRecord, ManifestRecord},
    target, verify,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertQuantization {
    Exact,
    Mxfp4,
}

#[derive(Clone)]
enum Payload {
    Tensor {
        name: String,
        layer: i32,
        tensor: TensorRef,
    },
    Expert {
        layer: u32,
        expert: u32,
        bytes: Vec<u8>,
        decoded_bytes: u64,
    },
}

struct PreparedCompile {
    target: target::TargetProfile,
    sources: Vec<Payload>,
    records: Vec<LoweredRecord>,
    table: Option<rans256::Table>,
}

pub fn handles(request: &CompileRequest) -> bool {
    matches!(
        &request.codec,
        CodecRequest::Auto | CodecRequest::Profile(profile) if profile == "rans256-g0-nibble"
    )
}

pub fn dry_run(request: &CompileRequest) -> Result<DryRunSummary> {
    validate_options(request)?;
    let inventory = source::discover(&request.source)?;
    let model = crate::pipeline::build_semantic_ir(&inventory)?.ok_or_else(|| {
        ColicError::unsupported(
            Stage::SemanticIr.as_str(),
            "no supported architecture frontend matched this source model",
        )
    })?;
    let prepared = prepare(request, &model)?;
    let plan = storage::plan_records(&prepared.records, prepared.target, 4 * 1024 * 1024 * 1024)?;
    Ok(DryRunSummary {
        target_name: prepared.target.name,
        source_tensors: inventory.tensors.len(),
        source_stored_bytes: inventory.source_stored_bytes,
        plan,
    })
}

pub fn compile(request: &CompileRequest, progress: &mut dyn ProgressSink) -> Result<()> {
    if request.dry_run {
        let _ = dry_run(request)?;
        return Ok(());
    }
    validate_options(request)?;
    progress.stage(Stage::SourceDiscovery);
    let inventory = source::discover_with_progress(&request.source, &mut |update| {
        progress.source_file(&update);
    })?;
    progress.stage(Stage::SemanticIr);
    let model = crate::pipeline::build_semantic_ir(&inventory)?.ok_or_else(|| {
        ColicError::unsupported(
            Stage::SemanticIr.as_str(),
            "no supported architecture frontend matched this source model",
        )
    })?;
    progress.stage(Stage::TargetPlanning);
    let prepared = prepare(request, &model)?;
    let output = request
        .output
        .as_ref()
        .ok_or_else(|| ColicError::Usage("compile requires an output package path".into()))?;
    progress.stage(Stage::StoragePlanning);
    let plan = storage::plan_records(&prepared.records, prepared.target, 4 * 1024 * 1024 * 1024)?;
    let fingerprint = source::fingerprint_bytes(&inventory.source_fingerprint)?;
    let temporary = storage::temporary_package_path(output)?;
    progress.stage(Stage::Emission);

    let write_result = (|| -> Result<()> {
        let mut header_crcs = Vec::with_capacity(plan.shards as usize);
        let mut metadata = Vec::with_capacity(plan.records.len());
        let mut completed_bytes = 0_u64;
        for shard_id in 0..plan.shards {
            let path = temporary.join(format!("data-{shard_id:05}.coli"));
            let mut writer = storage::DataShardWriter::create(
                &path,
                shard_id,
                plan.record_alignment,
                fingerprint,
            )?;
            for (index, planned) in plan
                .records
                .iter()
                .enumerate()
                .filter(|(_, record)| record.shard_id == shard_id)
            {
                let source = prepared
                    .sources
                    .get(index)
                    .ok_or_else(|| ColicError::Usage("codec source/plan order mismatch".into()))?;
                let manifest = write_payload(&mut writer, planned, source)?;
                completed_bytes = completed_bytes
                    .checked_add(planned.record.stored_bytes)
                    .ok_or_else(|| ColicError::Usage("emitted byte total overflows u64".into()))?;
                metadata.push(manifest);
                progress.emission(
                    (index + 1) as u64,
                    plan.records.len() as u64,
                    completed_bytes,
                    plan.projected_stored_bytes,
                );
            }
            writer.finish()?;
            let mut header = [0_u8; storage::DATA_SHARD_HEADER_BYTES as usize];
            fs::File::open(&path)
                .map_err(|source| ColicError::Io {
                    path: path.clone(),
                    source,
                })?
                .read_exact(&mut header)
                .map_err(|source| ColicError::Io {
                    path: path.clone(),
                    source,
                })?;
            header_crcs.push(u32::from_le_bytes(header[72..76].try_into().unwrap()));
        }
        let manifest = storage::encode_manifest_with_records(
            &plan,
            prepared.target.name,
            fingerprint,
            &metadata,
            &header_crcs,
        )?;
        let manifest = rans256::manifest_table_region(manifest, prepared.table.as_ref())?;
        let manifest_path = temporary.join("manifest.coli");
        fs::write(&manifest_path, manifest).map_err(|source| ColicError::Io {
            path: manifest_path,
            source,
        })?;
        copy_package_json_metadata(&inventory.root, &temporary)?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_dir_all(&temporary);
        return Err(error);
    }
    if request.force {
        storage::replace_package(&temporary, output)
    } else {
        storage::publish_package(&temporary, output)
    }?;
    if request.verify {
        progress.stage(Stage::Verification);
        let _ = verify::verify_package(output)?;
    }
    Ok(())
}

fn prepare(request: &CompileRequest, model: &SemanticModel) -> Result<PreparedCompile> {
    let quantization = resolve_quantization(request, model)?;
    let target_profile = target::resolve(&request.target, target::HostCapabilities::current())?;
    if target_profile != target::MACOS_ARM64_METAL_APPLE8_V1 {
        return Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            "rANS256-g0-nibble storage is currently defined for the Apple8 target only",
        ));
    }

    let mut raw_experts = BTreeMap::new();
    for (key, expert) in &model.routed_experts {
        let raw = match quantization {
            ExpertQuantization::Exact => target::lower_apple8_exact_mxfp4_expert(expert)?,
            ExpertQuantization::Mxfp4 => target::lower_apple8_quantized_mxfp4_expert(expert)?,
        };
        raw_experts.insert(*key, raw);
    }
    let mode = match request.codec {
        CodecRequest::Auto => apple8::Mode::Auto,
        CodecRequest::Profile(ref profile) if profile == "rans256-g0-nibble" => apple8::Mode::Force,
        _ => {
            return Err(ColicError::Usage(
                "codec compiler called for a non-rANS request".into(),
            ));
        }
    };
    let encoded = apple8::prepare(&raw_experts, mode)?;
    let use_table = encoded.compressed_matrices != 0;

    let mut sources = Vec::new();
    let mut records = Vec::new();
    let mut id = 1_u64;
    for (name, tensor) in &model.global_tensors {
        push_tensor(&mut sources, &mut records, &mut id, name.clone(), -1, tensor)?;
    }
    for (layer, tensors) in &model.layer_static_tensors {
        let layer_i32: i32 = (*layer)
            .try_into()
            .map_err(|_| ColicError::Usage("layer number exceeds i32".into()))?;
        for (role, tensor) in tensors {
            push_tensor(
                &mut sources,
                &mut records,
                &mut id,
                format!("layers.{layer}.{role}"),
                layer_i32,
                tensor,
            )?;
        }
    }
    for (key, expert) in &model.routed_experts {
        let bytes = encoded
            .experts
            .get(key)
            .ok_or_else(|| ColicError::Usage("missing prepared rANS expert".into()))?
            .clone();
        let decoded_bytes = [&expert.gate, &expert.up, &expert.down]
            .into_iter()
            .try_fold(0_u64, |sum, matrix| {
                sum.checked_add(target::apple8_tile_bytes(matrix.rows, matrix.columns)?)
                    .ok_or_else(|| ColicError::Usage("expert decoded byte total overflows".into()))
            })?;
        records.push(LoweredRecord {
            id,
            kind: 2,
            stored_bytes: bytes.len() as u64,
            decoded_bytes,
        });
        sources.push(Payload::Expert {
            layer: key.0,
            expert: key.1,
            bytes,
            decoded_bytes,
        });
        id = next_id(id)?;
    }
    for (name, tensor) in &model.resident_tensors {
        push_tensor(&mut sources, &mut records, &mut id, name.clone(), -2, tensor)?;
    }

    Ok(PreparedCompile {
        target: target_profile,
        sources,
        records,
        table: use_table.then_some(encoded.table),
    })
}

fn push_tensor(
    sources: &mut Vec<Payload>,
    records: &mut Vec<LoweredRecord>,
    id: &mut u64,
    name: String,
    layer: i32,
    tensor: &TensorRef,
) -> Result<()> {
    records.push(LoweredRecord {
        id: *id,
        kind: 1,
        stored_bytes: target::exact_tensor_stored_bytes(tensor)?,
        decoded_bytes: tensor.len,
    });
    sources.push(Payload::Tensor {
        name,
        layer,
        tensor: tensor.clone(),
    });
    *id = next_id(*id)?;
    Ok(())
}

fn write_payload(
    writer: &mut storage::DataShardWriter,
    planned: &storage::PlannedRecord,
    source: &Payload,
) -> Result<ManifestRecord> {
    match source {
        Payload::Tensor {
            name,
            layer,
            tensor,
        } => {
            let mut checksums = (0, 0);
            writer.write_record_stream(planned, |file| {
                checksums = target::stream_exact_tensor(tensor, file)?;
                Ok(planned.record.stored_bytes)
            })?;
            Ok(ManifestRecord {
                id: planned.record.id,
                name: Some(name.clone()),
                layer: *layer,
                expert: -1,
                kind: 1,
                codec: 0,
                math_format: target::math_format_for_dtype(&tensor.dtype)?,
                scale_format: 0,
                layout: 0,
                flags: 0b10,
                stored_crc32c: checksums.1,
                logical_crc32c: checksums.0,
                codec_table_id: 0,
            })
        }
        Payload::Expert {
            layer,
            expert,
            bytes,
            decoded_bytes,
        } => {
            if bytes.len() as u64 != planned.record.stored_bytes
                || *decoded_bytes != planned.record.decoded_bytes
            {
                return Err(ColicError::Usage(
                    "prepared rANS expert does not match storage plan".into(),
                ));
            }
            let crc = storage::crc32c(bytes);
            writer.write_record(planned, bytes)?;
            Ok(ManifestRecord {
                id: planned.record.id,
                name: Some(format!("layers.{layer}.ffn.experts.{expert}")),
                layer: (*layer)
                    .try_into()
                    .map_err(|_| ColicError::Usage("layer number exceeds i32".into()))?,
                expert: (*expert)
                    .try_into()
                    .map_err(|_| ColicError::Usage("expert number exceeds i32".into()))?,
                kind: 2,
                codec: 0,
                math_format: 0xfffe,
                scale_format: 0xfffe,
                layout: 0xfffe,
                flags: 0,
                stored_crc32c: crc,
                logical_crc32c: 0,
                codec_table_id: 0,
            })
        }
    }
}

fn resolve_quantization(request: &CompileRequest, model: &SemanticModel) -> Result<ExpertQuantization> {
    match &request.quant {
        QuantRequest::Exact => Ok(ExpertQuantization::Exact),
        QuantRequest::Profile(profile) if profile == "mxfp4" => {
            if model.architecture != Architecture::Qwen3_5MoeMoE {
                return Err(ColicError::unsupported(
                    Stage::TargetPlanning.as_str(),
                    "`--quant mxfp4` currently supports Qwen3.5/3.6/3.7 MoE routed experts only",
                ));
            }
            Ok(ExpertQuantization::Mxfp4)
        }
        QuantRequest::Profile(profile) => Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            format!("quantization profile `{profile}` is not implemented"),
        )),
    }
}

fn validate_options(request: &CompileRequest) -> Result<()> {
    if !handles(request) {
        return Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            "codec profile is not implemented by the PR3 codec compiler",
        ));
    }
    if request.optimization != OptimizationProfile::Default {
        return Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            "non-default optimization profiles are not implemented; use `--opt default`",
        ));
    }
    Ok(())
}

fn next_id(id: u64) -> Result<u64> {
    id.checked_add(1)
        .ok_or_else(|| ColicError::Usage("record ID overflows u64".into()))
}

fn is_source_weight_index_json(name: &str) -> bool {
    name.ends_with(".safetensors.index.json")
        || name.ends_with(".bin.index.json")
        || name.ends_with(".msgpack.index.json")
        || name.ends_with(".h5.index.json")
}

fn copy_package_json_metadata(source_root: &std::path::Path, package_root: &std::path::Path) -> Result<()> {
    let entries = fs::read_dir(source_root).map_err(|source| ColicError::Io {
        path: source_root.to_path_buf(),
        source,
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| ColicError::Io {
            path: source_root.to_path_buf(),
            source,
        })?;
        let source_path = entry.path();
        let Some(name) = source_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.ends_with(".json") || is_source_weight_index_json(name) {
            continue;
        }
        if fs::metadata(&source_path)
            .map_err(|source| ColicError::Io {
                path: source_path.clone(),
                source,
            })?
            .is_file()
        {
            files.push((name.to_owned(), source_path));
        }
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));
    for (name, source_path) in files {
        let destination = package_root.join(name);
        fs::copy(&source_path, &destination).map_err(|source| ColicError::Io {
            path: source_path,
            source,
        })?;
    }
    Ok(())
}

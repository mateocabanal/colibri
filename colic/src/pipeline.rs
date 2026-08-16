use std::{fs, io::Read, path::PathBuf};

use crate::{
    error::{ColicError, Result},
    ir::SemanticModel,
    model::deepseek_v4::DeepSeekV4Frontend,
    source,
    storage::{self, LoweredRecord, ManifestRecord, StoragePlan},
    target, verify,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileRequest {
    pub source: PathBuf,
    pub output: Option<PathBuf>,
    pub target: TargetRequest,
    pub quant: QuantRequest,
    pub codec: CodecRequest,
    pub optimization: OptimizationProfile,
    pub dry_run: bool,
    pub verify: bool,
    pub force: bool,
}

impl CompileRequest {
    pub fn new(source: PathBuf) -> Self {
        Self {
            source,
            output: None,
            target: TargetRequest::Native,
            quant: QuantRequest::Exact,
            codec: CodecRequest::None,
            optimization: OptimizationProfile::Default,
            dry_run: false,
            verify: false,
            force: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetRequest {
    Native,
    Profile(String),
}
impl TargetRequest {
    pub fn parse(value: &str) -> Result<Self> {
        if value == "native" {
            return Ok(Self::Native);
        }
        if value.starts_with("portable") {
            return Err(ColicError::Usage(
                "portable compiler targets are not supported".into(),
            ));
        }
        if value.is_empty() {
            return Err(ColicError::Usage("target profile cannot be empty".into()));
        }
        Ok(Self::Profile(value.into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantRequest {
    Exact,
    Profile(String),
}
impl QuantRequest {
    pub fn parse(value: &str) -> Result<Self> {
        if value == "exact" {
            Ok(Self::Exact)
        } else if value.is_empty() {
            Err(ColicError::Usage("quant profile cannot be empty".into()))
        } else {
            Ok(Self::Profile(value.into()))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecRequest {
    None,
    Auto,
    Profile(String),
}
impl CodecRequest {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "none" => Ok(Self::None),
            "auto" => Ok(Self::Auto),
            "" => Err(ColicError::Usage("codec profile cannot be empty".into())),
            other => Ok(Self::Profile(other.into())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationProfile {
    Default,
    Size,
    Latency,
}
impl OptimizationProfile {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "default" => Ok(Self::Default),
            "size" => Ok(Self::Size),
            "latency" => Ok(Self::Latency),
            other => Err(ColicError::Usage(format!(
                "unknown optimization profile `{other}`"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    SourceDiscovery,
    SemanticIr,
    Validation,
    TargetPlanning,
    StoragePlanning,
    Emission,
    Verification,
}
impl Stage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SourceDiscovery => "source discovery",
            Self::SemanticIr => "semantic IR",
            Self::Validation => "validation",
            Self::TargetPlanning => "target planning",
            Self::StoragePlanning => "storage planning",
            Self::Emission => "artifact emission",
            Self::Verification => "verification",
        }
    }
}

pub trait ProgressSink {
    fn stage(&mut self, stage: Stage);

    fn source_file(&mut self, _: &source::DiscoveryProgress) {}

    fn emission(&mut self, _: u64, _: u64, _: u64, _: u64) {}
}

#[derive(Debug, Clone)]
pub struct DryRunSummary {
    pub target_name: &'static str,
    pub source_tensors: usize,
    pub source_stored_bytes: u64,
    pub plan: StoragePlan,
}
pub struct NoProgress;
impl ProgressSink for NoProgress {
    fn stage(&mut self, _: Stage) {}
}

pub fn inspect_source(source_path: &std::path::Path) -> Result<source::SourceInventory> {
    source::discover(source_path)
}

pub fn build_semantic_ir(inventory: &source::SourceInventory) -> Result<Option<SemanticModel>> {
    if DeepSeekV4Frontend::probe(inventory)? {
        Ok(Some(DeepSeekV4Frontend::build(inventory)?))
    } else {
        Ok(None)
    }
}

pub fn dry_run(request: &CompileRequest) -> Result<DryRunSummary> {
    validate_supported_options(request)?;
    let inventory = source::discover(&request.source)?;
    let model = build_semantic_ir(&inventory)?.ok_or_else(|| {
        ColicError::unsupported(
            Stage::SemanticIr.as_str(),
            "no supported architecture frontend matched this source model",
        )
    })?;
    let target = target::resolve(&request.target, target::HostCapabilities::current())?;
    let records = exact_record_inventory(&model)?;
    let plan = storage::plan_records(&records, target, 4 * 1024 * 1024 * 1024)?;
    Ok(DryRunSummary {
        target_name: target.name,
        source_tensors: inventory.tensors.len(),
        source_stored_bytes: inventory.source_stored_bytes,
        plan,
    })
}

/// Stable v1 record order: globals, layer-static tensors, then pageable experts.
pub fn exact_record_inventory(model: &SemanticModel) -> Result<Vec<LoweredRecord>> {
    let mut records = Vec::new();
    let mut id = 1_u64;
    for tensor in model.global_tensors.values() {
        records.push(exact_tensor_record(id, tensor)?);
        id = next_record_id(id)?;
    }
    for tensors in model.layer_static_tensors.values() {
        for tensor in tensors.values() {
            records.push(exact_tensor_record(id, tensor)?);
            id = next_record_id(id)?;
        }
    }
    for expert in model.routed_experts.values() {
        let payload_bytes = target::exact_expert_stored_bytes(expert)?;
        records.push(LoweredRecord {
            id,
            kind: 2,
            stored_bytes: payload_bytes,
            decoded_bytes: target::exact_expert_decoded_bytes(expert)?,
        });
        id = next_record_id(id)?;
    }
    for tensor in model.resident_tensors.values() {
        records.push(exact_tensor_record(id, tensor)?);
        id = next_record_id(id)?;
    }
    Ok(records)
}

#[allow(dead_code)]
#[derive(Debug)]
struct ExactPayload {
    record: LoweredRecord,
    manifest: ManifestRecord,
    bytes: Vec<u8>,
}

#[allow(dead_code)]
fn lower_exact_payloads(model: &SemanticModel) -> Result<Vec<ExactPayload>> {
    let mut payloads = Vec::new();
    let mut id = 1_u64;
    for (name, tensor) in &model.global_tensors {
        payloads.push(lower_exact_tensor_payload(
            id,
            name.clone(),
            -1,
            -1,
            tensor,
        )?);
        id = next_record_id(id)?;
    }
    for (layer, tensors) in &model.layer_static_tensors {
        let layer: i32 = (*layer)
            .try_into()
            .map_err(|_| ColicError::Usage("layer number exceeds COLI i32 range".into()))?;
        for (role, tensor) in tensors {
            payloads.push(lower_exact_tensor_payload(
                id,
                format!("layers.{layer}.{role}"),
                layer,
                -1,
                tensor,
            )?);
            id = next_record_id(id)?;
        }
    }
    for expert in model.routed_experts.values() {
        let bytes = target::lower_exact_expert(expert)?;
        let stored_bytes: u64 = bytes
            .len()
            .try_into()
            .map_err(|_| ColicError::Usage("expert payload exceeds u64".into()))?;
        let layer: i32 = expert
            .layer
            .try_into()
            .map_err(|_| ColicError::Usage("layer number exceeds COLI i32 range".into()))?;
        let expert_id: i32 = expert
            .expert
            .try_into()
            .map_err(|_| ColicError::Usage("expert number exceeds COLI i32 range".into()))?;
        payloads.push(ExactPayload {
            record: LoweredRecord {
                id,
                kind: 2,
                stored_bytes,
                decoded_bytes: target::exact_expert_decoded_bytes(expert)?,
            },
            manifest: ManifestRecord {
                id,
                name: Some(format!(
                    "layers.{}.ffn.experts.{}",
                    expert.layer, expert.expert
                )),
                layer,
                expert: expert_id,
                kind: 2,
                codec: 0,
                math_format: 0xfffe,
                scale_format: 0xfffe,
                layout: 0xfffe,
                flags: 0,
                stored_crc32c: storage::crc32c(&bytes),
                logical_crc32c: 0,
                codec_table_id: 0,
            },
            bytes,
        });
        id = next_record_id(id)?;
    }
    for (name, tensor) in &model.resident_tensors {
        payloads.push(lower_exact_tensor_payload(
            id,
            name.clone(),
            -2,
            -1,
            tensor,
        )?);
        id = next_record_id(id)?;
    }
    Ok(payloads)
}

#[allow(dead_code)]
fn lower_exact_tensor_payload(
    id: u64,
    name: String,
    layer: i32,
    expert: i32,
    tensor: &source::TensorRef,
) -> Result<ExactPayload> {
    let bytes = target::lower_exact_tensor(tensor)?;
    let logical_crc32c = storage::crc32c(&bytes[128..]);
    let stored_bytes: u64 = bytes
        .len()
        .try_into()
        .map_err(|_| ColicError::Usage("tensor payload exceeds u64".into()))?;
    Ok(ExactPayload {
        record: LoweredRecord {
            id,
            kind: 1,
            stored_bytes,
            decoded_bytes: tensor.len,
        },
        manifest: ManifestRecord {
            id,
            name: Some(name),
            layer,
            expert,
            kind: 1,
            codec: 0,
            math_format: target::math_format_for_dtype(&tensor.dtype)?,
            scale_format: 0,
            layout: 0,
            flags: 0b10,
            stored_crc32c: storage::crc32c(&bytes),
            logical_crc32c,
            codec_table_id: 0,
        },
        bytes,
    })
}

fn next_record_id(id: u64) -> Result<u64> {
    id.checked_add(1)
        .ok_or_else(|| ColicError::Usage("record ID overflows u64".into()))
}

fn exact_tensor_record(id: u64, tensor: &source::TensorRef) -> Result<LoweredRecord> {
    let payload_bytes = target::exact_tensor_stored_bytes(tensor)?;
    Ok(LoweredRecord {
        id,
        kind: 1,
        stored_bytes: payload_bytes,
        decoded_bytes: tensor.len,
    })
}

#[derive(Clone)]
enum ExactSource {
    Tensor {
        name: String,
        layer: i32,
        tensor: source::TensorRef,
    },
    Expert(Box<crate::ir::RoutedExpert>),
}

fn exact_sources(model: &SemanticModel) -> Vec<ExactSource> {
    let mut sources = Vec::new();
    sources.extend(
        model
            .global_tensors
            .iter()
            .map(|(name, tensor)| ExactSource::Tensor {
                name: name.clone(),
                layer: -1,
                tensor: tensor.clone(),
            }),
    );
    for (layer, tensors) in &model.layer_static_tensors {
        sources.extend(tensors.iter().map(|(role, tensor)| ExactSource::Tensor {
            name: format!("layers.{layer}.{role}"),
            layer: *layer as i32,
            tensor: tensor.clone(),
        }));
    }
    sources.extend(
        model
            .routed_experts
            .values()
            .cloned()
            .map(|expert| ExactSource::Expert(Box::new(expert))),
    );
    sources.extend(
        model
            .resident_tensors
            .iter()
            .map(|(name, tensor)| ExactSource::Tensor {
                name: name.clone(),
                layer: -2,
                tensor: tensor.clone(),
            }),
    );
    sources
}

fn stream_payload(
    writer: &mut storage::DataShardWriter,
    planned: &storage::PlannedRecord,
    source: &ExactSource,
) -> Result<ManifestRecord> {
    match source {
        ExactSource::Tensor {
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
        ExactSource::Expert(expert) => {
            let bytes = target::lower_exact_expert(expert)?;
            let crc = storage::crc32c(&bytes);
            writer.write_record(planned, &bytes)?;
            Ok(ManifestRecord {
                id: planned.record.id,
                name: Some(format!(
                    "layers.{}.ffn.experts.{}",
                    expert.layer, expert.expert
                )),
                layer: expert.layer as i32,
                expert: expert.expert as i32,
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

pub fn compile(request: &CompileRequest, progress: &mut dyn ProgressSink) -> Result<()> {
    if request.dry_run {
        let _summary = dry_run(request)?;
        return Ok(());
    }
    validate_supported_options(request)?;
    progress.stage(Stage::SourceDiscovery);
    let inventory = source::discover_with_progress(&request.source, &mut |update| {
        progress.source_file(&update);
    })?;
    progress.stage(Stage::SemanticIr);
    let model = build_semantic_ir(&inventory)?.ok_or_else(|| {
        ColicError::unsupported(
            Stage::SemanticIr.as_str(),
            "no supported architecture frontend matched this source model",
        )
    })?;
    progress.stage(Stage::TargetPlanning);
    let target = target::resolve(&request.target, target::HostCapabilities::current())?;
    let output = request
        .output
        .as_ref()
        .ok_or_else(|| ColicError::Usage("compile requires an output package path".into()))?;
    progress.stage(Stage::StoragePlanning);
    let sources = exact_sources(&model);
    let records = exact_record_inventory(&model)?;
    let plan = storage::plan_records(&records, target, 4 * 1024 * 1024 * 1024)?;
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
                let source = &sources[index];
                let manifest = stream_payload(&mut writer, planned, source)?;
                completed_bytes += planned.record.stored_bytes;
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
            target.name,
            fingerprint,
            &metadata,
            &header_crcs,
        )?;
        let manifest_path = temporary.join("manifest.coli");
        fs::write(&manifest_path, manifest).map_err(|source| ColicError::Io {
            path: manifest_path,
            source,
        })?;
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
        let _summary = verify::verify_package(output)?;
    }
    Ok(())
}

fn validate_supported_options(request: &CompileRequest) -> Result<()> {
    if !matches!(request.quant, QuantRequest::Exact) {
        return Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            "quantization profiles are not implemented; use `--quant exact`",
        ));
    }
    if !matches!(request.codec, CodecRequest::None) {
        return Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            "storage codecs are not implemented; use `--codec none`",
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

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::{
        ir::{Architecture, Matrix, ModelGeometry, RoutedExpert},
        source::TensorRef,
    };

    fn tensor(len: u64) -> TensorRef {
        TensorRef {
            source: "fixture.safetensors".into(),
            offset: 0,
            len,
            dtype: "U8".into(),
            shape: vec![len],
        }
    }

    fn synthetic_v4_source(root: &std::path::Path) {
        fs::create_dir_all(root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{"model_type":"deepseek_v4","hidden_size":2,"num_hidden_layers":1,"n_routed_experts":2,"moe_intermediate_size":3,"vocab_size":4,"hc_mult":2,"num_hash_layers":1,"num_experts_per_tok":1,"num_attention_heads":1,"head_dim":2,"q_lora_rank":1,"o_groups":1,"o_lora_rank":1,"index_n_heads":1,"index_head_dim":1,"compress_ratios":[0]}"#,
        )
        .unwrap();
        let mut specs = BTreeMap::<String, (&str, Vec<u64>)>::new();
        let mut add = |name: String, dtype: &'static str, shape: Vec<u64>| {
            specs.insert(name, (dtype, shape));
        };
        for expert in 0..2 {
            for (role, shape) in [("w1", vec![3, 2]), ("w2", vec![2, 3]), ("w3", vec![3, 2])] {
                add(
                    format!("layers.0.ffn.experts.{expert}.{role}.weight"),
                    "F8_E4M3FN",
                    shape,
                );
                add(
                    format!("layers.0.ffn.experts.{expert}.{role}.scale"),
                    "F8_E8M0",
                    vec![1, 1],
                );
            }
        }
        for (name, dtype, shape) in [
            ("embed.weight", "BF16", vec![4, 2]),
            ("head.weight", "BF16", vec![4, 2]),
            ("norm.weight", "BF16", vec![2]),
            ("hc_head_base", "F32", vec![2]),
            ("hc_head_fn", "F32", vec![2, 4]),
            ("hc_head_scale", "F32", vec![1]),
            ("layers.0.ffn.gate.weight", "BF16", vec![2, 2]),
            ("layers.0.ffn.gate.tid2eid", "I64", vec![4, 1]),
            (
                "layers.0.ffn.shared_experts.w1.weight",
                "F8_E4M3FN",
                vec![3, 2],
            ),
            (
                "layers.0.ffn.shared_experts.w2.weight",
                "F8_E4M3FN",
                vec![2, 3],
            ),
            (
                "layers.0.ffn.shared_experts.w3.weight",
                "F8_E4M3FN",
                vec![3, 2],
            ),
            (
                "layers.0.ffn.shared_experts.w1.scale",
                "F8_E8M0",
                vec![1, 1],
            ),
            (
                "layers.0.ffn.shared_experts.w2.scale",
                "F8_E8M0",
                vec![1, 1],
            ),
            (
                "layers.0.ffn.shared_experts.w3.scale",
                "F8_E8M0",
                vec![1, 1],
            ),
            ("layers.0.ffn_norm.weight", "BF16", vec![2]),
            ("layers.0.attn.attn_sink", "F32", vec![1]),
            ("layers.0.attn.kv_norm.weight", "BF16", vec![2]),
            ("layers.0.attn.q_norm.weight", "BF16", vec![1]),
            ("layers.0.attn.wkv.weight", "F8_E4M3FN", vec![2, 2]),
            ("layers.0.attn.wkv.scale", "F8_E8M0", vec![1, 1]),
            ("layers.0.attn.wo_a.weight", "F8_E4M3FN", vec![1, 2]),
            ("layers.0.attn.wo_a.scale", "F8_E8M0", vec![1, 1]),
            ("layers.0.attn.wo_b.weight", "F8_E4M3FN", vec![2, 1]),
            ("layers.0.attn.wo_b.scale", "F8_E8M0", vec![1, 1]),
            ("layers.0.attn.wq_a.weight", "F8_E4M3FN", vec![1, 2]),
            ("layers.0.attn.wq_a.scale", "F8_E8M0", vec![1, 1]),
            ("layers.0.attn.wq_b.weight", "F8_E4M3FN", vec![2, 1]),
            ("layers.0.attn.wq_b.scale", "F8_E8M0", vec![1, 1]),
            ("layers.0.attn_norm.weight", "BF16", vec![2]),
            ("layers.0.hc_attn_base", "F32", vec![8]),
            ("layers.0.hc_attn_fn", "F32", vec![8, 4]),
            ("layers.0.hc_attn_scale", "F32", vec![3]),
            ("layers.0.hc_ffn_base", "F32", vec![8]),
            ("layers.0.hc_ffn_fn", "F32", vec![8, 4]),
            ("layers.0.hc_ffn_scale", "F32", vec![3]),
        ] {
            add(name.into(), dtype, shape);
        }
        let mut offset = 0_u64;
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, (dtype, shape)) in specs {
            let size = match dtype {
                "U8" | "F8_E4M3FN" | "F8_E8M0" => 1,
                "BF16" => 2,
                "F32" => 4,
                "I64" => 8,
                _ => unreachable!(),
            };
            let bytes = shape.iter().product::<u64>() * size;
            header.insert(name, serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}));
            payload.resize(payload.len() + bytes as usize, 0);
            offset += bytes;
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        fs::write(root.join("model.safetensors"), file).unwrap();
    }

    #[test]
    fn exact_inventory_orders_static_tensors_before_pageable_experts() {
        let matrix = Matrix {
            source: tensor(1),
            rows: 1,
            columns: 1,
            scale: None,
        };
        let mut globals = BTreeMap::new();
        globals.insert("embed.weight".into(), tensor(2));
        let mut layer = BTreeMap::new();
        layer.insert("ffn.gate.weight".into(), tensor(3));
        layer.insert("ffn_norm.weight".into(), tensor(4));
        let mut layers = BTreeMap::new();
        layers.insert(0, layer);
        let mut experts = BTreeMap::new();
        experts.insert(
            (0, 0),
            RoutedExpert {
                layer: 0,
                expert: 0,
                gate: matrix.clone(),
                up: matrix.clone(),
                down: matrix,
            },
        );
        let model = SemanticModel {
            architecture: Architecture::DeepSeekV4Flash,
            geometry: ModelGeometry {
                hidden_size: 1,
                layers: 1,
                routed_experts_per_layer: 1,
                moe_intermediate_size: 1,
                vocab_size: 1,
                hc_mult: 1,
                num_hash_layers: 0,
                experts_per_token: 1,
                attention_heads: 1,
                head_dim: 1,
                q_lora_rank: 1,
                o_groups: 1,
                o_lora_rank: 1,
                index_heads: 1,
                index_head_dim: 1,
                compression_ratios: vec![0],
            },
            routed_experts: experts,
            global_tensors: globals,
            layer_static_tensors: layers,
            resident_tensors: BTreeMap::new(),
        };
        let records = exact_record_inventory(&model).unwrap();
        assert_eq!(
            records.iter().map(|record| record.id).collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
        assert_eq!(
            records.iter().map(|record| record.kind).collect::<Vec<_>>(),
            [1, 1, 1, 2]
        );
        assert_eq!(records[0].decoded_bytes, 2);
    }

    #[test]
    fn compile_and_verify_synthetic_v4_package() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("colic-e2e-{}-{nonce}", std::process::id()));
        let source = root.join("source");
        synthetic_v4_source(&source);
        let output = root.join("compiled.coli");
        let mut request = CompileRequest::new(source);
        request.output = Some(output.clone());
        request.target = TargetRequest::Profile("macos-arm64-metal-apple8-v1".into());
        request.verify = true;
        compile(&request, &mut NoProgress).unwrap();
        assert_eq!(verify::verify_package(&output).unwrap().records, 37);
        let second_output = root.join("compiled-again.coli");
        let mut second_request = request.clone();
        second_request.output = Some(second_output.clone());
        compile(&second_request, &mut NoProgress).unwrap();
        for name in ["manifest.coli", "data-00000.coli"] {
            assert_eq!(
                fs::read(output.join(name)).unwrap(),
                fs::read(second_output.join(name)).unwrap(),
                "recompiled {name} differs"
            );
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unsupported_transform_options_fail_before_source_reads() {
        let mut request = CompileRequest::new("definitely-not-a-model".into());
        request.codec = CodecRequest::Auto;
        assert!(matches!(
            dry_run(&request),
            Err(ColicError::Unsupported { .. })
        ));
        request.codec = CodecRequest::None;
        request.quant = QuantRequest::Profile("mxfp4".into());
        assert!(matches!(
            dry_run(&request),
            Err(ColicError::Unsupported { .. })
        ));
    }
}

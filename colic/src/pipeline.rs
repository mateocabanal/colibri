use std::{fs, io::Read, path::PathBuf};

use crate::{
    error::{ColicError, Result},
    ir::{Architecture, SemanticModel},
    model::deepseek_v4::DeepSeekV4Frontend,
    model::qwen_moe::QwenMoeFrontend,
    quant::{int4_record, mxfp4_record, nvfp4_record},
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
    /// `--plan FILE` (#201): replay a saved plan manifest.
    pub plan: Option<PathBuf>,
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
            plan: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetRequest {
    Native,
    /// `--target auto` (#201): the inline planner resolves quant + profile
    /// from the probed machine before emission.
    Auto,
    Profile(String),
}
impl TargetRequest {
    pub fn parse(value: &str) -> Result<Self> {
        if value == "native" {
            return Ok(Self::Native);
        }
        if value == "auto" {
            return Ok(Self::Auto);
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
            Err(ColicError::Usage("target profile cannot be empty".into()))
        } else {
            Ok(Self::Profile(value.into()))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertQuantization {
    Exact,
    Mxfp4,
    Int4,
    Nvfp4,
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
    } else if crate::model::qwen4_exp::Qwen4ExpFrontend::probe(inventory)? {
        Ok(Some(crate::model::qwen4_exp::Qwen4ExpFrontend::build(inventory)?))
    } else if QwenMoeFrontend::probe(inventory)? {
        Ok(Some(QwenMoeFrontend::build(inventory)?))
    } else {
        Ok(None)
    }
}

fn resolve_expert_quantization(
    request: &CompileRequest,
    model: &SemanticModel,
) -> Result<ExpertQuantization> {
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
        QuantRequest::Profile(profile) if profile == "int4" || profile == "int4-g32" => {
            if model.architecture != Architecture::Qwen3_5MoeMoE {
                return Err(ColicError::unsupported(
                    Stage::TargetPlanning.as_str(),
                    "`--quant int4` currently supports Qwen3.5/3.6/3.7 MoE routed experts only",
                ));
            }
            Ok(ExpertQuantization::Int4)
        }
        QuantRequest::Profile(profile) if profile == "nvfp4" || profile == "nvfp4-1d" => {
            if model.architecture != Architecture::Qwen3_5MoeMoE {
                return Err(ColicError::unsupported(
                    Stage::TargetPlanning.as_str(),
                    "`--quant nvfp4` currently supports Qwen3.5/3.6/3.7 MoE routed experts only",
                ));
            }
            Ok(ExpertQuantization::Nvfp4)
        }
        QuantRequest::Profile(profile) => Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            format!("quantization profile `{profile}` is not implemented"),
        )),
    }
}

fn reject_incompatible_target(
    quantization: ExpertQuantization,
    target_profile: target::TargetProfile,
) -> Result<()> {
    if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1
        && matches!(quantization, ExpertQuantization::Int4 | ExpertQuantization::Nvfp4)
    {
        return Err(ColicError::unsupported(
            Stage::TargetPlanning.as_str(),
            "canonical INT4/NVFP4 records are not Apple8 execution tiles; use `--quant mxfp4` for the Apple8 target until target-specific FP4 lowering is added",
        ));
    }
    Ok(())
}

/// Resolve `--target auto` (#201): run the inline planner on the probed
/// machine and translate its routed-expert decision into the concrete target
/// profile + quantization for emission.
fn resolve_auto_target(
    request: &CompileRequest,
    model: &SemanticModel,
    inventory: &source::SourceInventory,
) -> Result<(target::TargetProfile, ExpertQuantization)> {
    use crate::plan::{
        cost::Objective,
        ir::{BackendKind, MathFormat, TensorRole},
        machine::MachineProfile,
        planner::{build_plan, PlanRequest},
    };
    let machine = MachineProfile::probe();
    let config = source::config(&inventory.root)?;
    let plan = build_plan(
        model,
        &machine,
        config.as_ref(),
        &PlanRequest {
            objective: Objective::Balanced,
            context_tokens: 8192,
            batch: 1,
            target_profile: None,
        },
        Some(inventory.source_fingerprint.clone()),
    )?;
    let expert_math = plan
        .tensors
        .iter()
        .find(|tensor| tensor.role == TensorRole::RoutedExpert)
        .map(|tensor| (tensor.math, tensor.backend));
    match expert_math {
        Some((MathFormat::Int4G32, BackendKind::Cpu)) => {
            Ok((target::LINUX_X86_64_AVX2_V1, ExpertQuantization::Int4))
        }
        Some((MathFormat::Mxfp4, BackendKind::Metal)) => {
            Ok((target::MACOS_ARM64_METAL_APPLE8_V1, ExpertQuantization::Mxfp4))
        }
        Some((MathFormat::Mxfp4, BackendKind::Cuda)) => {
            Ok((target::LINUX_X86_64_AVX2_V1, ExpertQuantization::Int4))
        }
        _ => Ok((target::LINUX_X86_64_AVX2_V1, ExpertQuantization::Exact)),
    }
}

/// Resolve the concrete (profile, quantization) pair for a request.
fn resolve_plan_options(
    request: &CompileRequest,
    model: &SemanticModel,
    inventory: &source::SourceInventory,
) -> Result<(target::TargetProfile, ExpertQuantization)> {
    if request.target == TargetRequest::Auto {
        return resolve_auto_target(request, model, inventory);
    }
    let quantization = resolve_expert_quantization(request, model)?;
    let target_profile = target::resolve(&request.target, target::HostCapabilities::current())?;
    reject_incompatible_target(quantization, target_profile)?;
    Ok((target_profile, quantization))
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
    let (target_profile, expert_quantization) =
        resolve_plan_options(request, &model, &inventory)?;
    let records = record_inventory(&model, expert_quantization, target_profile)?;
    let plan = storage::plan_records(&records, target_profile, 4 * 1024 * 1024 * 1024)?;
    Ok(DryRunSummary {
        target_name: target_profile.name,
        source_tensors: inventory.tensors.len(),
        source_stored_bytes: inventory.source_stored_bytes,
        plan,
    })
}

/// Stable v1 record order: globals, layer-static tensors, then pageable experts.
pub fn exact_record_inventory(model: &SemanticModel) -> Result<Vec<LoweredRecord>> {
    record_inventory(
        model,
        ExpertQuantization::Exact,
        target::LINUX_X86_64_AVX2_V1,
    )
}

fn record_inventory(
    model: &SemanticModel,
    expert_quantization: ExpertQuantization,
    target_profile: target::TargetProfile,
) -> Result<Vec<LoweredRecord>> {
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
        let (stored_bytes, decoded_bytes) = if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1 {
            match expert_quantization {
                ExpertQuantization::Exact => target::validate_apple8_exact_mxfp4_expert(expert)?,
                ExpertQuantization::Mxfp4 => target::validate_apple8_quantized_mxfp4_expert(expert)?,
                ExpertQuantization::Int4 | ExpertQuantization::Nvfp4 => {
                    return Err(ColicError::unsupported(
                        Stage::TargetPlanning.as_str(),
                        "INT4/NVFP4 do not yet have an Apple8 target lowering",
                    ));
                }
            }
            (
                target::apple8_expert_stored_bytes(expert)?,
                target::apple8_expert_decoded_bytes(expert)?,
            )
        } else {
            match expert_quantization {
                ExpertQuantization::Exact => (
                    target::exact_expert_stored_bytes(expert)?,
                    target::exact_expert_decoded_bytes(expert)?,
                ),
                ExpertQuantization::Mxfp4 => (
                    mxfp4_record::stored_bytes(expert)?,
                    mxfp4_record::resident_bytes(expert)?,
                ),
                ExpertQuantization::Int4 => (
                    int4_record::stored_bytes(expert)?,
                    int4_record::resident_bytes(expert)?,
                ),
                ExpertQuantization::Nvfp4 => (
                    nvfp4_record::stored_bytes(expert)?,
                    nvfp4_record::resident_bytes(expert)?,
                ),
            }
        };
        records.push(LoweredRecord {
            id,
            kind: 2,
            stored_bytes,
            decoded_bytes,
        });
        id = next_record_id(id)?;
    }
    for tensor in model.resident_tensors.values() {
        records.push(exact_tensor_record(id, tensor)?);
        id = next_record_id(id)?;
    }
    Ok(records)
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
    Expert {
        expert: Box<crate::ir::RoutedExpert>,
        quantization: ExpertQuantization,
    },
}

fn exact_sources(model: &SemanticModel, expert_quantization: ExpertQuantization) -> Vec<ExactSource> {
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
    sources.extend(model.routed_experts.values().cloned().map(|expert| {
        ExactSource::Expert {
            expert: Box::new(expert),
            quantization: expert_quantization,
        }
    }));
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
    target_profile: target::TargetProfile,
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
        ExactSource::Expert {
            expert,
            quantization,
        } => {
            let crc = if target_profile == target::MACOS_ARM64_METAL_APPLE8_V1 {
                let bytes = match quantization {
                    ExpertQuantization::Exact => target::lower_apple8_exact_mxfp4_expert(expert)?,
                    ExpertQuantization::Mxfp4 => target::lower_apple8_quantized_mxfp4_expert(expert)?,
                    ExpertQuantization::Int4 | ExpertQuantization::Nvfp4 => {
                        return Err(ColicError::unsupported(
                            Stage::Emission.as_str(),
                            "INT4/NVFP4 do not yet have an Apple8 target lowering",
                        ));
                    }
                };
                if bytes.len() as u64 != planned.record.stored_bytes {
                    return Err(ColicError::Usage(
                        "Apple8 expert emission does not match its raw storage plan".into(),
                    ));
                }
                let crc = storage::crc32c(&bytes);
                writer.write_record(planned, &bytes)?;
                crc
            } else {
                match quantization {
                    ExpertQuantization::Exact => {
                        let mut crc = 0;
                        writer.write_record_stream(planned, |file| {
                            crc = target::stream_exact_expert(expert, file)?;
                            Ok(planned.record.stored_bytes)
                        })?;
                        crc
                    }
                    ExpertQuantization::Mxfp4 => {
                        let bytes = mxfp4_record::lower_expert(expert)?;
                        let crc = storage::crc32c(&bytes);
                        writer.write_record(planned, &bytes)?;
                        crc
                    }
                    ExpertQuantization::Int4 => {
                        let bytes = int4_record::lower_expert(expert)?;
                        let crc = storage::crc32c(&bytes);
                        writer.write_record(planned, &bytes)?;
                        crc
                    }
                    ExpertQuantization::Nvfp4 => {
                        let bytes = nvfp4_record::lower_expert(expert)?;
                        let crc = storage::crc32c(&bytes);
                        writer.write_record(planned, &bytes)?;
                        crc
                    }
                }
            };
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

fn is_source_weight_index_json(name: &str) -> bool {
    name.ends_with(".safetensors.index.json")
        || name.ends_with(".bin.index.json")
        || name.ends_with(".msgpack.index.json")
        || name.ends_with(".h5.index.json")
}

/// Copy runtime/model metadata JSON verbatim into the compiled package.
/// Weight index JSON is intentionally omitted because its shard paths refer
/// to source checkpoint files that are not part of a COLI package.
fn copy_package_json_metadata(
    source_root: &std::path::Path,
    package_root: &std::path::Path,
) -> Result<()> {
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
        let metadata = fs::metadata(&source_path).map_err(|source| ColicError::Io {
            path: source_path.clone(),
            source,
        })?;
        if metadata.is_file() {
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

/// Validate a saved plan manifest against the source inventory + request
/// before emission (#201). Replay validates; it never re-runs policy.
fn validate_plan_replay(
    plan_path: &std::path::Path,
    inventory: &source::SourceInventory,
    request: &CompileRequest,
) -> Result<()> {
    let bytes = std::fs::read(plan_path).map_err(|error| ColicError::Io {
        path: plan_path.to_path_buf(),
        source: error,
    })?;
    let plan: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| ColicError::Usage(format!(
            "plan manifest {} is not valid JSON: {error}",
            plan_path.display()
        )))?;
    let schema = plan
        .get("plan_schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema != crate::plan::ir::PLAN_SCHEMA_VERSION {
        return Err(ColicError::Usage(format!(
            "plan schema version {schema} unsupported (expected {})",
            crate::plan::ir::PLAN_SCHEMA_VERSION
        )));
    }
    let pinned = plan
        .get("source_fingerprint")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            ColicError::Usage("plan manifest is missing `source_fingerprint`".into())
        })?;
    if pinned != inventory.source_fingerprint {
        return Err(ColicError::Usage(format!(
            "plan manifest was built for source fingerprint {pinned}, but this source hashes to {}; re-run `colic plan` on this checkpoint",
            inventory.source_fingerprint
        )));
    }
    // v1 replay contract: the plan's routed-expert math must match the
    // requested quantization (the only per-record format the current
    // emission path varies). Non-expert families are emitted exact.
    let expert_math = plan
        .get("tensors")
        .and_then(serde_json::Value::as_array)
        .and_then(|tensors| {
            tensors
                .iter()
                .find(|tensor| tensor.get("role").and_then(serde_json::Value::as_str) == Some("routed_expert"))
        })
        .and_then(|tensor| tensor.get("math"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("bf16");
    let requested = match &request.quant {
        QuantRequest::Exact => "bf16",
        QuantRequest::Profile(profile) => match profile.as_str() {
            "int4" | "int4-g32" => "int4_g32",
            "mxfp4" => "mxfp4",
            "nvfp4" | "nvfp4-1d" => "nvfp4",
            other => other,
        },
    };
    if expert_math != requested {
        return Err(ColicError::Usage(format!(
            "plan manifest expects expert math `{expert_math}` but --quant `{requested}` was requested; replay the plan with matching options"
        )));
    }
    Ok(())
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
    // Plan replay (#201): validate the saved manifest against this source and
    // the requested options BEFORE emission. Replay validates; it never
    // re-runs policy.
    if let Some(plan_path) = &request.plan {
        validate_plan_replay(plan_path, &inventory, request)?;
    }
    progress.stage(Stage::SemanticIr);
    let model = build_semantic_ir(&inventory)?.ok_or_else(|| {
        ColicError::unsupported(
            Stage::SemanticIr.as_str(),
            "no supported architecture frontend matched this source model",
        )
    })?;
    let (target_profile, expert_quantization) =
        resolve_plan_options(request, &model, &inventory)?;
    progress.stage(Stage::TargetPlanning);
    let output = request
        .output
        .as_ref()
        .ok_or_else(|| ColicError::Usage("compile requires an output package path".into()))?;
    progress.stage(Stage::StoragePlanning);
    let sources = exact_sources(&model, expert_quantization);
    let records = record_inventory(&model, expert_quantization, target_profile)?;
    let plan = storage::plan_records(&records, target_profile, 4 * 1024 * 1024 * 1024)?;
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
                let manifest = stream_payload(&mut writer, planned, source, target_profile)?;
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
            target_profile.name,
            fingerprint,
            &metadata,
            &header_crcs,
        )?;
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
        let _summary = verify::verify_package(output)?;
    }
    Ok(())
}

fn validate_supported_options(request: &CompileRequest) -> Result<()> {
    match &request.quant {
        QuantRequest::Exact => {}
        QuantRequest::Profile(profile)
            if matches!(
                profile.as_str(),
                "mxfp4" | "int4" | "int4-g32" | "nvfp4" | "nvfp4-1d"
            ) => {}
        QuantRequest::Profile(profile) => {
            return Err(ColicError::unsupported(
                Stage::TargetPlanning.as_str(),
                format!("quantization profile `{profile}` is not implemented"),
            ));
        }
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
    use std::{collections::BTreeMap, fs};

    use super::*;
    use crate::{
        ir::{Matrix, ModelGeometry, RoutedExpert},
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

    fn qwen_quant_inventory_model() -> SemanticModel {
        let matrix = Matrix {
            source: TensorRef {
                source: "fixture.safetensors".into(),
                offset: 0,
                len: 2 * 2 * 32,
                dtype: "BF16".into(),
                shape: vec![2, 32],
            },
            rows: 2,
            columns: 32,
            scale: None,
        };
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
        SemanticModel {
            architecture: Architecture::Qwen3_5MoeMoE,
            geometry: ModelGeometry {
                hidden_size: 32,
                layers: 1,
                routed_experts_per_layer: 1,
                moe_intermediate_size: 2,
                vocab_size: 1,
                hc_mult: 0,
                num_hash_layers: 0,
                experts_per_token: 1,
                attention_heads: 1,
                head_dim: 32,
                num_key_value_heads: 1,
                linear_key_head_dim: 0,
                q_lora_rank: 0,
                o_groups: 0,
                o_lora_rank: 0,
                index_heads: 0,
                index_head_dim: 0,
                compression_ratios: Vec::new(),
            },
            routed_experts: experts,
            global_tensors: BTreeMap::new(),
            layer_static_tensors: BTreeMap::new(),
            resident_tensors: BTreeMap::new(),
        }
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
                num_key_value_heads: 0,
                linear_key_head_dim: 0,
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
        assert_eq!(records.iter().map(|r| r.id).collect::<Vec<_>>(), [1, 2, 3, 4]);
        assert_eq!(records.iter().map(|r| r.kind).collect::<Vec<_>>(), [1, 1, 1, 2]);
    }

    #[test]
    fn quantized_inventory_reduces_qwen_expert_resident_bytes() {
        let model = qwen_quant_inventory_model();
        let exact = record_inventory(&model, ExpertQuantization::Exact, target::LINUX_X86_64_AVX2_V1).unwrap();
        let mx = record_inventory(&model, ExpertQuantization::Mxfp4, target::LINUX_X86_64_AVX2_V1).unwrap();
        let i4 = record_inventory(&model, ExpertQuantization::Int4, target::LINUX_X86_64_AVX2_V1).unwrap();
        let nv = record_inventory(&model, ExpertQuantization::Nvfp4, target::LINUX_X86_64_AVX2_V1).unwrap();
        assert!(mx[0].decoded_bytes < exact[0].decoded_bytes);
        assert!(i4[0].decoded_bytes < exact[0].decoded_bytes);
        assert!(nv[0].decoded_bytes < exact[0].decoded_bytes);
    }

    #[test]
    fn apple8_rejects_non_apple8_quant_layouts() {
        let mut request = CompileRequest::new("unused".into());
        request.quant = QuantRequest::Profile("int4".into());
        assert!(reject_incompatible_target(ExpertQuantization::Int4, target::MACOS_ARM64_METAL_APPLE8_V1).is_err());
        request.quant = QuantRequest::Profile("nvfp4".into());
        assert!(reject_incompatible_target(ExpertQuantization::Nvfp4, target::MACOS_ARM64_METAL_APPLE8_V1).is_err());
    }

    #[test]
    fn supported_quant_names_pass_option_validation() {
        for profile in ["mxfp4", "int4", "int4-g32", "nvfp4", "nvfp4-1d"] {
            let mut request = CompileRequest::new("unused".into());
            request.quant = QuantRequest::Profile(profile.into());
            validate_supported_options(&request).unwrap();
        }
    }

    #[test]
    fn unsupported_transform_options_fail_before_source_reads() {
        let mut request = CompileRequest::new("definitely-not-a-model".into());
        request.codec = CodecRequest::Auto;
        assert!(matches!(dry_run(&request), Err(ColicError::Unsupported { .. })));
        request.codec = CodecRequest::None;
        request.quant = QuantRequest::Profile("not-a-format".into());
        assert!(matches!(dry_run(&request), Err(ColicError::Unsupported { .. })));
    }

    #[test]
    fn source_index_detection_is_unchanged() {
        for name in [
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "flax_model.msgpack.index.json",
            "tf_model.h5.index.json",
        ] {
            assert!(is_source_weight_index_json(name));
        }
        assert!(!is_source_weight_index_json("config.json"));
        let _ = fs::metadata(".");
    }
}

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    ir::{
        Activation, Architecture, MathFormat, Matrix, ModelAssets, ModelGeometry, Quantization,
        RoutedExpert, ScaleFormat, SemanticModel, SourceRepresentation,
    },
    source::{self, SourceInventory, TensorRef},
};

pub struct DeepSeekV4Frontend;

impl DeepSeekV4Frontend {
    pub fn probe(source: &SourceInventory) -> Result<bool> {
        let Some(config) = source::config(&source.root)? else {
            return Ok(false);
        };
        Ok(config.get("model_type").and_then(Value::as_str) == Some("deepseek_v4"))
    }

    pub fn build(source: &SourceInventory) -> Result<SemanticModel> {
        let config = source::config(&source.root)?.ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "DeepSeek V4 source is missing config.json".into(),
        })?;
        if config.get("model_type").and_then(Value::as_str) != Some("deepseek_v4") {
            return invalid(&source.root, "config model_type is not `deepseek_v4`");
        }

        let geometry = geometry(&source.root, &config)?;
        validate_geometry(&source.root, &geometry)?;

        let mut claimed = BTreeSet::new();
        let mut members: BTreeMap<(u32, u32), BTreeMap<ExpertMember, (&str, &TensorRef)>> =
            BTreeMap::new();
        for (name, tensor) in &source.tensors {
            if let Some((layer, expert, member)) = parse_expert_name(name) {
                if layer >= geometry.layers || expert >= geometry.routed_experts_per_layer {
                    return invalid(
                        &source.root,
                        format!("expert tensor `{name}` is outside config layer/expert bounds"),
                    );
                }
                if members
                    .entry((layer, expert))
                    .or_default()
                    .insert(member, (name.as_str(), tensor))
                    .is_some()
                {
                    return invalid(
                        &source.root,
                        format!("duplicate semantic expert member for `{name}`"),
                    );
                }
            }
        }

        let expected = geometry
            .layers
            .checked_mul(geometry.routed_experts_per_layer)
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "expert count overflows u32".into(),
            })?;
        if members.len() != expected as usize {
            return invalid(
                &source.root,
                format!("expected {expected} routed experts, found {}", members.len()),
            );
        }

        let mut routed_experts = BTreeMap::new();
        for layer in 0..geometry.layers {
            for expert in 0..geometry.routed_experts_per_layer {
                let group = members.get(&(layer, expert)).ok_or_else(|| ColicError::InvalidSource {
                    path: source.root.clone(),
                    detail: format!("missing routed expert ({layer}, {expert})"),
                })?;
                let location = ExpertLocation { layer, expert };
                let gate = matrix(&source.root, group, location, MatrixSpec::w1(&geometry), &mut claimed)?;
                let down = matrix(&source.root, group, location, MatrixSpec::w2(&geometry), &mut claimed)?;
                let up = matrix(&source.root, group, location, MatrixSpec::w3(&geometry), &mut claimed)?;
                routed_experts.insert(
                    (layer, expert),
                    RoutedExpert {
                        layer,
                        expert,
                        gate,
                        up,
                        down,
                        activation: Activation::SwiGlu,
                    },
                );
            }
        }

        let mut global_tensors = BTreeMap::new();
        let hc_params = geometry.hc_mult;
        for (role, name, dtype, shape) in [
            (
                "embedding",
                "embed.weight",
                "BF16",
                vec![geometry.vocab_size as u64, geometry.hidden_size as u64],
            ),
            (
                "lm_head",
                "head.weight",
                "BF16",
                vec![geometry.vocab_size as u64, geometry.hidden_size as u64],
            ),
            ("final_norm", "norm.weight", "BF16", vec![geometry.hidden_size as u64]),
            ("hc_head_base", "hc_head_base", "F32", vec![hc_params as u64]),
            (
                "hc_head_fn",
                "hc_head_fn",
                "F32",
                vec![hc_params as u64, (geometry.hc_mult * geometry.hidden_size) as u64],
            ),
            ("hc_head_scale", "hc_head_scale", "F32", vec![1]),
        ] {
            global_tensors.insert(
                role.to_owned(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    name,
                    dtype,
                    &shape,
                    &mut claimed,
                )?,
            );
        }

        let mut layer_static_tensors = BTreeMap::new();
        for layer in 0..geometry.layers {
            let mut tensors = BTreeMap::new();
            validate_attention_and_hc(
                &source.root,
                &source.tensors,
                layer,
                &geometry,
                &mut tensors,
                &mut claimed,
            )?;

            let prefix = format!("layers.{layer}.ffn");
            tensors.insert(
                "router.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{prefix}.gate.weight"),
                    "BF16",
                    &[
                        geometry.routed_experts_per_layer as u64,
                        geometry.hidden_size as u64,
                    ],
                    &mut claimed,
                )?,
            );
            if layer < geometry.num_hash_layers {
                tensors.insert(
                    "router.token_to_expert".into(),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{prefix}.gate.tid2eid"),
                        "I64",
                        &[
                            geometry.vocab_size as u64,
                            geometry.experts_per_token as u64,
                        ],
                        &mut claimed,
                    )?,
                );
            } else {
                tensors.insert(
                    "router.bias".into(),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{prefix}.gate.bias"),
                        "F32",
                        &[geometry.routed_experts_per_layer as u64],
                        &mut claimed,
                    )?,
                );
            }

            for (role, rows, columns) in [
                ("gate", geometry.moe_intermediate_size, geometry.hidden_size),
                ("down", geometry.hidden_size, geometry.moe_intermediate_size),
                ("up", geometry.moe_intermediate_size, geometry.hidden_size),
            ] {
                let source_role = match role {
                    "gate" => "w1",
                    "down" => "w2",
                    _ => "w3",
                };
                tensors.insert(
                    format!("shared_expert.{role}.weight"),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{prefix}.shared_experts.{source_role}.weight"),
                        "F8_E4M3FN",
                        &[rows as u64, columns as u64],
                        &mut claimed,
                    )?,
                );
                tensors.insert(
                    format!("shared_expert.{role}.scale"),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{prefix}.shared_experts.{source_role}.scale"),
                        "F8_E8M0",
                        &fp8_scale_shape(rows, columns),
                        &mut claimed,
                    )?,
                );
            }
            tensors.insert(
                "ffn_norm".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("layers.{layer}.ffn_norm.weight"),
                    "BF16",
                    &[geometry.hidden_size as u64],
                    &mut claimed,
                )?,
            );
            layer_static_tensors.insert(layer, tensors);
        }

        // Optional draft/DSpark tensors are semantically global to the draft
        // subsystem. Their exact source representation is retained for the
        // first exact target ABI, but they are not left unclassified.
        for (name, tensor) in &source.tensors {
            if name.starts_with("mtp.") && !claimed.contains(name) {
                global_tensors.insert(format!("draft.{name}"), tensor.clone());
                claimed.insert(name.clone());
            }
        }

        let unclassified = source
            .tensors
            .keys()
            .filter(|name| !claimed.contains(*name))
            .cloned()
            .collect::<Vec<_>>();
        if !unclassified.is_empty() {
            let sample = unclassified.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
            return invalid(
                &source.root,
                format!(
                    "{} unclassified DeepSeek V4 tensor(s); first: {sample}",
                    unclassified.len()
                ),
            );
        }

        Ok(SemanticModel {
            architecture: Architecture::DeepSeekV4Flash,
            geometry,
            routed_experts,
            global_tensors,
            layer_static_tensors,
            resident_tensors: BTreeMap::new(),
            assets: classify_assets(source),
        })
    }
}

fn classify_assets(source: &SourceInventory) -> ModelAssets {
    let mut assets = ModelAssets::default();
    for path in &source.files {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        match name {
            "config.json" => assets.config = Some(path.clone()),
            "tokenizer.json"
            | "tokenizer_config.json"
            | "special_tokens_map.json"
            | "chat_template.jinja"
            | "generation_config.json" => assets.tokenizer.push(path.clone()),
            _ => {}
        }
    }
    assets.tokenizer.sort();
    assets
}

fn geometry(root: &Path, config: &Value) -> Result<ModelGeometry> {
    Ok(ModelGeometry {
        hidden_size: required_u32(root, config, "hidden_size")?,
        layers: required_u32(root, config, "num_hidden_layers")?,
        routed_experts_per_layer: required_u32(root, config, "n_routed_experts")?,
        moe_intermediate_size: required_u32(root, config, "moe_intermediate_size")?,
        vocab_size: required_u32(root, config, "vocab_size")?,
        hc_mult: required_u32(root, config, "hc_mult")?,
        num_hash_layers: required_u32_allow_zero(root, config, "num_hash_layers")?,
        experts_per_token: required_u32(root, config, "num_experts_per_tok")?,
        attention_heads: required_u32(root, config, "num_attention_heads")?,
        head_dim: required_u32(root, config, "head_dim")?,
        q_lora_rank: required_u32(root, config, "q_lora_rank")?,
        o_groups: required_u32(root, config, "o_groups")?,
        o_lora_rank: required_u32(root, config, "o_lora_rank")?,
        index_heads: required_u32(root, config, "index_n_heads")?,
        index_head_dim: required_u32(root, config, "index_head_dim")?,
        compression_ratios: required_u32_array(root, config, "compress_ratios")?,
    })
}

fn validate_geometry(root: &Path, geometry: &ModelGeometry) -> Result<()> {
    if geometry.num_hash_layers > geometry.layers {
        return invalid(root, "config field `num_hash_layers` cannot exceed `num_hidden_layers`");
    }
    if !geometry.attention_heads.is_multiple_of(geometry.o_groups) {
        return invalid(root, "config `num_attention_heads` must be divisible by `o_groups`");
    }
    if geometry.compression_ratios.len() < geometry.layers as usize {
        return invalid(root, "config `compress_ratios` must cover every hidden layer");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ExpertMember {
    W1Weight,
    W1Scale,
    W2Weight,
    W2Scale,
    W3Weight,
    W3Scale,
}

#[derive(Debug, Clone, Copy)]
struct ExpertLocation {
    layer: u32,
    expert: u32,
}

#[derive(Debug, Clone, Copy)]
struct MatrixSpec {
    weight: ExpertMember,
    scale: ExpertMember,
    rows: u32,
    columns: u32,
    role: &'static str,
}

impl MatrixSpec {
    fn w1(geometry: &ModelGeometry) -> Self {
        Self {
            weight: ExpertMember::W1Weight,
            scale: ExpertMember::W1Scale,
            rows: geometry.moe_intermediate_size,
            columns: geometry.hidden_size,
            role: "gate",
        }
    }
    fn w2(geometry: &ModelGeometry) -> Self {
        Self {
            weight: ExpertMember::W2Weight,
            scale: ExpertMember::W2Scale,
            rows: geometry.hidden_size,
            columns: geometry.moe_intermediate_size,
            role: "down",
        }
    }
    fn w3(geometry: &ModelGeometry) -> Self {
        Self {
            weight: ExpertMember::W3Weight,
            scale: ExpertMember::W3Scale,
            rows: geometry.moe_intermediate_size,
            columns: geometry.hidden_size,
            role: "up",
        }
    }
}

fn parse_expert_name(name: &str) -> Option<(u32, u32, ExpertMember)> {
    let mut parts = name.split('.');
    (parts.next()? == "layers").then_some(())?;
    let layer = parts.next()?.parse().ok()?;
    (parts.next()? == "ffn").then_some(())?;
    (parts.next()? == "experts").then_some(())?;
    let expert = parts.next()?.parse().ok()?;
    let member = match (parts.next()?, parts.next()?, parts.next()) {
        ("w1", "weight", None) => ExpertMember::W1Weight,
        ("w1", "scale", None) => ExpertMember::W1Scale,
        ("w2", "weight", None) => ExpertMember::W2Weight,
        ("w2", "scale", None) => ExpertMember::W2Scale,
        ("w3", "weight", None) => ExpertMember::W3Weight,
        ("w3", "scale", None) => ExpertMember::W3Scale,
        _ => return None,
    };
    Some((layer, expert, member))
}

fn matrix(
    root: &Path,
    members: &BTreeMap<ExpertMember, (&str, &TensorRef)>,
    location: ExpertLocation,
    spec: MatrixSpec,
    claimed: &mut BTreeSet<String>,
) -> Result<Matrix> {
    let (weight_name, weight) = members.get(&spec.weight).copied().ok_or_else(|| {
        ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!(
                "expert ({}, {}) is missing {} weight",
                location.layer, location.expert, spec.role
            ),
        }
    })?;
    let (scale_name, scale) = members.get(&spec.scale).copied().ok_or_else(|| {
        ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!(
                "expert ({}, {}) is missing {} scale",
                location.layer, location.expert, spec.role
            ),
        }
    })?;

    let quantization = if is_e4m3(&weight.dtype)
        && weight.shape == [spec.rows as u64, spec.columns as u64]
        && is_ue8m0(&scale.dtype)
        && scale.shape == fp8_scale_shape(spec.rows, spec.columns)
    {
        Quantization {
            math_format: MathFormat::Fp8E4M3,
            source_representation: SourceRepresentation::NativeFp8,
            scale_format: ScaleFormat::Ue8m0,
            scale_block_rows: 128,
            scale_block_columns: 128,
        }
    } else if weight.dtype == "I8"
        && weight.shape == [spec.rows as u64, spec.columns.div_ceil(2) as u64]
        && is_ue8m0(&scale.dtype)
        && scale.shape == [spec.rows as u64, spec.columns.div_ceil(32) as u64]
    {
        Quantization {
            math_format: MathFormat::MxFp4E2M1,
            source_representation: SourceRepresentation::PackedMxFp4Nibbles,
            scale_format: ScaleFormat::Ue8m0,
            scale_block_rows: 1,
            scale_block_columns: 32,
        }
    } else {
        return invalid(
            root,
            format!(
                "expert ({}, {}) {} has unsupported weight {:?}/{:?} and scale {:?}/{:?}",
                location.layer,
                location.expert,
                spec.role,
                weight.dtype,
                weight.shape,
                scale.dtype,
                scale.shape
            ),
        );
    };

    claimed.insert(weight_name.to_owned());
    claimed.insert(scale_name.to_owned());
    Ok(Matrix {
        source: weight.clone(),
        rows: spec.rows,
        columns: spec.columns,
        scale: Some(scale.clone()),
        quantization,
    })
}

fn validate_attention_and_hc(
    root: &Path,
    source: &BTreeMap<String, TensorRef>,
    layer: u32,
    geometry: &ModelGeometry,
    output: &mut BTreeMap<String, TensorRef>,
    claimed: &mut BTreeSet<String>,
) -> Result<()> {
    let prefix = format!("layers.{layer}");
    let output_group_width = (geometry.attention_heads / geometry.o_groups) * geometry.head_dim;
    let output_width = geometry.o_groups * geometry.o_lora_rank;
    for (role, rows, columns) in [
        ("attention.wkv", geometry.head_dim, geometry.hidden_size),
        ("attention.wo_a", output_width, output_group_width),
        ("attention.wo_b", geometry.hidden_size, output_width),
        ("attention.wq_a", geometry.q_lora_rank, geometry.hidden_size),
        (
            "attention.wq_b",
            geometry.attention_heads * geometry.head_dim,
            geometry.q_lora_rank,
        ),
    ] {
        let source_role = role.strip_prefix("attention.").unwrap();
        output.insert(
            format!("{role}.weight"),
            validate_tensor(
                root,
                source,
                &format!("{prefix}.attn.{source_role}.weight"),
                "F8_E4M3FN",
                &[rows as u64, columns as u64],
                claimed,
            )?,
        );
        output.insert(
            format!("{role}.scale"),
            validate_tensor(
                root,
                source,
                &format!("{prefix}.attn.{source_role}.scale"),
                "F8_E8M0",
                &fp8_scale_shape(rows, columns),
                claimed,
            )?,
        );
    }

    for (role, source_role, dtype, shape) in [
        ("attention.sink", "attn.attn_sink", "F32", vec![geometry.attention_heads as u64]),
        ("attention.kv_norm", "attn.kv_norm.weight", "BF16", vec![geometry.head_dim as u64]),
        ("attention.q_norm", "attn.q_norm.weight", "BF16", vec![geometry.q_lora_rank as u64]),
        ("attention_norm", "attn_norm.weight", "BF16", vec![geometry.hidden_size as u64]),
    ] {
        output.insert(
            role.into(),
            validate_tensor(
                root,
                source,
                &format!("{prefix}.{source_role}"),
                dtype,
                &shape,
                claimed,
            )?,
        );
    }

    let ratio = geometry.compression_ratios[layer as usize];
    if ratio != 0 {
        let coff = if ratio == 4 { 2 } else { 1 };
        for (role, dtype, shape) in [
            ("compressor.ape", "F32", vec![ratio as u64, (coff * geometry.head_dim) as u64]),
            ("compressor.norm", "BF16", vec![geometry.head_dim as u64]),
            (
                "compressor.wgate",
                "BF16",
                vec![(coff * geometry.head_dim) as u64, geometry.hidden_size as u64],
            ),
            (
                "compressor.wkv",
                "BF16",
                vec![(coff * geometry.head_dim) as u64, geometry.hidden_size as u64],
            ),
        ] {
            let suffix = match role {
                "compressor.norm" => "compressor.norm.weight".to_owned(),
                "compressor.wgate" => "compressor.wgate.weight".to_owned(),
                "compressor.wkv" => "compressor.wkv.weight".to_owned(),
                _ => role.to_owned(),
            };
            output.insert(
                format!("attention.{role}"),
                validate_tensor(
                    root,
                    source,
                    &format!("{prefix}.attn.{suffix}"),
                    dtype,
                    &shape,
                    claimed,
                )?,
            );
        }
    }

    if ratio == 4 {
        let heads = geometry.index_heads;
        let ih = geometry.index_head_dim;
        for (role, source_role, dtype, shape) in [
            ("indexer.compressor.ape", "indexer.compressor.ape", "F32", vec![4, (2 * ih) as u64]),
            ("indexer.compressor.norm", "indexer.compressor.norm.weight", "BF16", vec![ih as u64]),
            (
                "indexer.compressor.wgate",
                "indexer.compressor.wgate.weight",
                "BF16",
                vec![(2 * ih) as u64, geometry.hidden_size as u64],
            ),
            (
                "indexer.compressor.wkv",
                "indexer.compressor.wkv.weight",
                "BF16",
                vec![(2 * ih) as u64, geometry.hidden_size as u64],
            ),
            (
                "indexer.weights_proj",
                "indexer.weights_proj.weight",
                "BF16",
                vec![heads as u64, geometry.hidden_size as u64],
            ),
            (
                "indexer.wq_b.weight",
                "indexer.wq_b.weight",
                "F8_E4M3FN",
                vec![(heads * ih) as u64, geometry.q_lora_rank as u64],
            ),
            (
                "indexer.wq_b.scale",
                "indexer.wq_b.scale",
                "F8_E8M0",
                fp8_scale_shape(heads * ih, geometry.q_lora_rank),
            ),
        ] {
            output.insert(
                format!("attention.{role}"),
                validate_tensor(
                    root,
                    source,
                    &format!("{prefix}.attn.{source_role}"),
                    dtype,
                    &shape,
                    claimed,
                )?,
            );
        }
    }

    let hc_params = (2 + geometry.hc_mult) * geometry.hc_mult;
    for (role, shape) in [
        ("hc.attention.base", vec![hc_params as u64]),
        (
            "hc.attention.fn",
            vec![hc_params as u64, (geometry.hc_mult * geometry.hidden_size) as u64],
        ),
        ("hc.attention.scale", vec![3]),
        ("hc.ffn.base", vec![hc_params as u64]),
        (
            "hc.ffn.fn",
            vec![hc_params as u64, (geometry.hc_mult * geometry.hidden_size) as u64],
        ),
        ("hc.ffn.scale", vec![3]),
    ] {
        let source_role = match role {
            "hc.attention.base" => "hc_attn_base",
            "hc.attention.fn" => "hc_attn_fn",
            "hc.attention.scale" => "hc_attn_scale",
            "hc.ffn.base" => "hc_ffn_base",
            "hc.ffn.fn" => "hc_ffn_fn",
            _ => "hc_ffn_scale",
        };
        output.insert(
            role.into(),
            validate_tensor(
                root,
                source,
                &format!("{prefix}.{source_role}"),
                "F32",
                &shape,
                claimed,
            )?,
        );
    }
    Ok(())
}

fn validate_tensor(
    root: &Path,
    tensors: &BTreeMap<String, TensorRef>,
    name: &str,
    dtype: &str,
    shape: &[u64],
    claimed: &mut BTreeSet<String>,
) -> Result<TensorRef> {
    let tensor = tensors.get(name).ok_or_else(|| ColicError::InvalidSource {
        path: root.to_owned(),
        detail: format!("missing required tensor `{name}`"),
    })?;
    let dtype_matches = if dtype == "F8_E4M3FN" {
        is_e4m3(&tensor.dtype)
    } else if dtype == "F8_E8M0" {
        is_ue8m0(&tensor.dtype)
    } else {
        tensor.dtype == dtype
    };
    if !dtype_matches || tensor.shape != shape {
        return invalid(
            root,
            format!(
                "tensor `{name}` has dtype/shape {:?}/{:?}, expected {dtype}/{shape:?}",
                tensor.dtype, tensor.shape
            ),
        );
    }
    if !claimed.insert(name.to_owned()) {
        return invalid(root, format!("tensor `{name}` was classified more than once"));
    }
    Ok(tensor.clone())
}

fn fp8_scale_shape(rows: u32, columns: u32) -> Vec<u64> {
    vec![rows.div_ceil(128) as u64, columns.div_ceil(128) as u64]
}

fn is_ue8m0(dtype: &str) -> bool {
    matches!(dtype, "F8_E8M0" | "F8_E8M0FNU")
}

fn is_e4m3(dtype: &str) -> bool {
    matches!(dtype, "F8_E4M3" | "F8_E4M3FN")
}

fn required_u32(root: &Path, config: &Value, field: &str) -> Result<u32> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| value.try_into().ok())
        .filter(|value: &u32| *value > 0)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("config field `{field}` must be a non-zero u32"),
        })
}

fn required_u32_allow_zero(root: &Path, config: &Value, field: &str) -> Result<u32> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("config field `{field}` must be a u32"),
        })
}

fn required_u32_array(root: &Path, config: &Value, field: &str) -> Result<Vec<u32>> {
    let values = config
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("config field `{field}` must be an array of u32"),
        })?;
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| value.try_into().ok())
                .ok_or_else(|| ColicError::InvalidSource {
                    path: root.to_owned(),
                    detail: format!("config field `{field}` contains a non-u32 value"),
                })
        })
        .collect()
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
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn root() -> PathBuf {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "colic-v4-semantic-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{"model_type":"deepseek_v4","hidden_size":2,"num_hidden_layers":1,"n_routed_experts":1,"moe_intermediate_size":3,"vocab_size":4,"hc_mult":2,"num_hash_layers":0,"num_experts_per_tok":1,"num_attention_heads":1,"head_dim":2,"q_lora_rank":1,"o_groups":1,"o_lora_rank":1,"index_n_heads":1,"index_head_dim":1,"compress_ratios":[0]}"#,
        )
        .unwrap();
        root
    }

    fn tensor(dtype: &str, shape: &[u64]) -> TensorRef {
        TensorRef {
            source: "fixture.safetensors".into(),
            offset: 0,
            len: 0,
            dtype: dtype.into(),
            shape: shape.to_vec(),
        }
    }

    fn fixture() -> SourceInventory {
        let root = root();
        let mut tensors = BTreeMap::new();
        for (role, shape) in [("w1", &[3, 2][..]), ("w2", &[2, 3][..]), ("w3", &[3, 2][..])] {
            tensors.insert(format!("layers.0.ffn.experts.0.{role}.weight"), tensor("I8", &[shape[0], shape[1].div_ceil(2)]));
            tensors.insert(format!("layers.0.ffn.experts.0.{role}.scale"), tensor("F8_E8M0", &[shape[0], shape[1].div_ceil(32)]));
        }
        for (name, dtype, shape) in [
            ("embed.weight", "BF16", vec![4, 2]),
            ("head.weight", "BF16", vec![4, 2]),
            ("norm.weight", "BF16", vec![2]),
            ("hc_head_base", "F32", vec![2]),
            ("hc_head_fn", "F32", vec![2, 4]),
            ("hc_head_scale", "F32", vec![1]),
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
            ("layers.0.ffn.gate.weight", "BF16", vec![1, 2]),
            ("layers.0.ffn.gate.bias", "F32", vec![1]),
            ("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3FN", vec![3, 2]),
            ("layers.0.ffn.shared_experts.w1.scale", "F8_E8M0", vec![1, 1]),
            ("layers.0.ffn.shared_experts.w2.weight", "F8_E4M3FN", vec![2, 3]),
            ("layers.0.ffn.shared_experts.w2.scale", "F8_E8M0", vec![1, 1]),
            ("layers.0.ffn.shared_experts.w3.weight", "F8_E4M3FN", vec![3, 2]),
            ("layers.0.ffn.shared_experts.w3.scale", "F8_E8M0", vec![1, 1]),
            ("layers.0.ffn_norm.weight", "BF16", vec![2]),
        ] {
            tensors.insert(name.into(), tensor(dtype, &shape));
        }
        SourceInventory {
            root: root.clone(),
            files: vec![root.join("config.json")],
            tensors,
            source_stored_bytes: 0,
            dtype_counts: BTreeMap::new(),
            source_fingerprint: "fixture".into(),
            config_fingerprint: None,
            architecture_hint: Some("DeepseekV4ForCausalLM".into()),
        }
    }

    #[test]
    fn explicit_mxfp4_semantics_and_activation_are_classified() {
        let source = fixture();
        let model = DeepSeekV4Frontend::build(&source).unwrap();
        let expert = &model.routed_experts[&(0, 0)];
        assert_eq!(expert.activation, Activation::SwiGlu);
        assert_eq!(expert.gate.quantization.math_format, MathFormat::MxFp4E2M1);
        assert_eq!(expert.gate.quantization.scale_block_columns, 32);
        assert!(model.resident_tensors.is_empty());
        fs::remove_dir_all(source.root).unwrap();
    }

    #[test]
    fn rejects_bad_scale_geometry() {
        let mut source = fixture();
        source.tensors.get_mut("layers.0.ffn.experts.0.w1.scale").unwrap().shape = vec![2, 2];
        assert!(DeepSeekV4Frontend::build(&source).is_err());
        fs::remove_dir_all(source.root).unwrap();
    }

    #[test]
    fn rejects_unclassified_tensor() {
        let mut source = fixture();
        source.tensors.insert("mystery.weight".into(), tensor("BF16", &[1]));
        assert!(DeepSeekV4Frontend::build(&source).is_err());
        fs::remove_dir_all(source.root).unwrap();
    }

    #[test]
    fn accepts_optional_mtp_tensor_as_classified_draft_record() {
        let mut source = fixture();
        source.tensors.insert("mtp.0.main_proj.weight".into(), tensor("BF16", &[1]));
        let model = DeepSeekV4Frontend::build(&source).unwrap();
        assert!(model.global_tensors.contains_key("draft.mtp.0.main_proj.weight"));
        fs::remove_dir_all(source.root).unwrap();
    }
}

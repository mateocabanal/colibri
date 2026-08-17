use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    ir::{Architecture, Matrix, ModelGeometry, RoutedExpert, SemanticModel},
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
        let geometry = ModelGeometry {
            hidden_size: required_u32(&source.root, &config, "hidden_size")?,
            layers: required_u32(&source.root, &config, "num_hidden_layers")?,
            routed_experts_per_layer: required_u32(&source.root, &config, "n_routed_experts")?,
            moe_intermediate_size: required_u32(&source.root, &config, "moe_intermediate_size")?,
            vocab_size: required_u32(&source.root, &config, "vocab_size")?,
            hc_mult: required_u32(&source.root, &config, "hc_mult")?,
            num_hash_layers: required_u32_allow_zero(&source.root, &config, "num_hash_layers")?,
            experts_per_token: required_u32(&source.root, &config, "num_experts_per_tok")?,
            attention_heads: required_u32(&source.root, &config, "num_attention_heads")?,
            head_dim: required_u32(&source.root, &config, "head_dim")?,
            num_key_value_heads: 0,
            linear_key_head_dim: 0,
            q_lora_rank: required_u32(&source.root, &config, "q_lora_rank")?,
            o_groups: required_u32(&source.root, &config, "o_groups")?,
            o_lora_rank: required_u32(&source.root, &config, "o_lora_rank")?,
            index_heads: required_u32(&source.root, &config, "index_n_heads")?,
            index_head_dim: required_u32(&source.root, &config, "index_head_dim")?,
            compression_ratios: required_u32_array(&source.root, &config, "compress_ratios")?,
        };
        if geometry.num_hash_layers > geometry.layers {
            return invalid(
                &source.root,
                "config field `num_hash_layers` cannot exceed `num_hidden_layers`",
            );
        }
        if !geometry.attention_heads.is_multiple_of(geometry.o_groups) {
            return invalid(
                &source.root,
                "config `num_attention_heads` must be divisible by `o_groups`",
            );
        }
        if geometry.compression_ratios.len() < geometry.layers as usize {
            return invalid(
                &source.root,
                "config `compress_ratios` must cover every `num_hidden_layers` entry",
            );
        }
        let mut members: BTreeMap<(u32, u32), BTreeMap<ExpertMember, &TensorRef>> = BTreeMap::new();
        let mut expert_names = BTreeSet::new();
        for (name, tensor) in &source.tensors {
            if let Some((layer, expert, member)) = parse_expert_name(name) {
                if layer >= geometry.layers || expert >= geometry.routed_experts_per_layer {
                    return invalid(
                        &source.root,
                        format!("expert tensor `{name}` is outside config layer/expert bounds"),
                    );
                }
                let key = (layer, expert);
                if members
                    .entry(key)
                    .or_default()
                    .insert(member, tensor)
                    .is_some()
                {
                    return invalid(
                        &source.root,
                        format!("duplicate semantic member for tensor `{name}`"),
                    );
                }
                expert_names.insert(name.clone());
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
                format!(
                    "expected {expected} routed experts, found {}",
                    members.len()
                ),
            );
        }
        let mut routed_experts = BTreeMap::new();
        for layer in 0..geometry.layers {
            for expert in 0..geometry.routed_experts_per_layer {
                let group =
                    members
                        .get(&(layer, expert))
                        .ok_or_else(|| ColicError::InvalidSource {
                            path: source.root.clone(),
                            detail: format!("missing routed expert ({layer}, {expert})"),
                        })?;
                let location = ExpertLocation { layer, expert };
                let gate = matrix(&source.root, group, location, MatrixSpec::w1(&geometry))?;
                let down = matrix(&source.root, group, location, MatrixSpec::w2(&geometry))?;
                let up = matrix(&source.root, group, location, MatrixSpec::w3(&geometry))?;
                routed_experts.insert(
                    (layer, expert),
                    RoutedExpert {
                        layer,
                        expert,
                        gate,
                        up,
                        down,
                    },
                );
            }
        }
        let hc_params = geometry.hc_mult;
        let global_specs = [
            (
                "embed.weight",
                "BF16",
                vec![geometry.vocab_size as u64, geometry.hidden_size as u64],
            ),
            (
                "head.weight",
                "BF16",
                vec![geometry.vocab_size as u64, geometry.hidden_size as u64],
            ),
            ("norm.weight", "BF16", vec![geometry.hidden_size as u64]),
            ("hc_head_base", "F32", vec![hc_params as u64]),
            (
                "hc_head_fn",
                "F32",
                vec![
                    hc_params as u64,
                    (geometry.hc_mult * geometry.hidden_size) as u64,
                ],
            ),
            ("hc_head_scale", "F32", vec![1]),
        ];
        let mut global_tensors = BTreeMap::new();
        for (name, dtype, shape) in global_specs {
            let tensor = source
                .tensors
                .get(name)
                .ok_or_else(|| ColicError::InvalidSource {
                    path: source.root.clone(),
                    detail: format!("missing required global tensor `{name}`"),
                })?;
            if tensor.dtype != dtype || tensor.shape != shape {
                return invalid(
                    &source.root,
                    format!(
                        "global tensor `{name}` has dtype/shape {:?}/{:?}, expected {dtype}/{shape:?}",
                        tensor.dtype, tensor.shape
                    ),
                );
            }
            global_tensors.insert(name.to_owned(), tensor.clone());
        }
        let mut layer_static_tensors = BTreeMap::new();
        for layer in 0..geometry.layers {
            let prefix = format!("layers.{layer}.ffn");
            let mut static_tensors = BTreeMap::new();
            validate_attention_and_hc(
                &source.root,
                &source.tensors,
                layer,
                &geometry,
                &mut static_tensors,
            )?;
            static_tensors.insert(
                "ffn.gate.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{prefix}.gate.weight"),
                    "BF16",
                    &[
                        geometry.routed_experts_per_layer as u64,
                        geometry.hidden_size as u64,
                    ],
                )?,
            );
            if layer < geometry.num_hash_layers {
                static_tensors.insert(
                    "ffn.gate.tid2eid".into(),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{prefix}.gate.tid2eid"),
                        "I64",
                        &[
                            geometry.vocab_size as u64,
                            geometry.experts_per_token as u64,
                        ],
                    )?,
                );
            } else {
                static_tensors.insert(
                    "ffn.gate.bias".into(),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{prefix}.gate.bias"),
                        "F32",
                        &[geometry.routed_experts_per_layer as u64],
                    )?,
                );
            }
            for (role, rows, columns) in [
                ("w1", geometry.moe_intermediate_size, geometry.hidden_size),
                ("w2", geometry.hidden_size, geometry.moe_intermediate_size),
                ("w3", geometry.moe_intermediate_size, geometry.hidden_size),
            ] {
                static_tensors.insert(
                    format!("ffn.shared_experts.{role}.weight"),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{prefix}.shared_experts.{role}.weight"),
                        "F8_E4M3FN",
                        &[rows as u64, columns as u64],
                    )?,
                );
                static_tensors.insert(
                    format!("ffn.shared_experts.{role}.scale"),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{prefix}.shared_experts.{role}.scale"),
                        "F8_E8M0",
                        &fp8_scale_shape(rows, columns),
                    )?,
                );
            }
            static_tensors.insert(
                "ffn_norm.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("layers.{layer}.ffn_norm.weight"),
                    "BF16",
                    &[geometry.hidden_size as u64],
                )?,
            );
            layer_static_tensors.insert(layer, static_tensors);
        }
        let layer_static_names = layer_static_tensors
            .iter()
            .flat_map(|(layer, tensors)| {
                tensors
                    .keys()
                    .map(move |role| format!("layers.{layer}.{role}"))
            })
            .collect::<BTreeSet<_>>();
        let resident_tensors = source
            .tensors
            .iter()
            .filter(|(name, _)| {
                !expert_names.contains(*name)
                    && !global_tensors.contains_key(*name)
                    && !layer_static_names.contains(*name)
            })
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        Ok(SemanticModel {
            architecture: Architecture::DeepSeekV4Flash,
            geometry,
            routed_experts,
            global_tensors,
            layer_static_tensors,
            resident_tensors,
        })
    }
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
            role: "w1",
        }
    }
    fn w2(geometry: &ModelGeometry) -> Self {
        Self {
            weight: ExpertMember::W2Weight,
            scale: ExpertMember::W2Scale,
            rows: geometry.hidden_size,
            columns: geometry.moe_intermediate_size,
            role: "w2",
        }
    }
    fn w3(geometry: &ModelGeometry) -> Self {
        Self {
            weight: ExpertMember::W3Weight,
            scale: ExpertMember::W3Scale,
            rows: geometry.moe_intermediate_size,
            columns: geometry.hidden_size,
            role: "w3",
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
    let weight = match (parts.next()?, parts.next()?, parts.next()) {
        ("w1", "weight", None) => ExpertMember::W1Weight,
        ("w1", "scale", None) => ExpertMember::W1Scale,
        ("w2", "weight", None) => ExpertMember::W2Weight,
        ("w2", "scale", None) => ExpertMember::W2Scale,
        ("w3", "weight", None) => ExpertMember::W3Weight,
        ("w3", "scale", None) => ExpertMember::W3Scale,
        _ => return None,
    };
    Some((layer, expert, weight))
}

fn matrix(
    root: &Path,
    members: &BTreeMap<ExpertMember, &TensorRef>,
    location: ExpertLocation,
    spec: MatrixSpec,
) -> Result<Matrix> {
    let source = members
        .get(&spec.weight)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!(
                "expert ({}, {}) is missing {} weight",
                location.layer, location.expert, spec.role
            ),
        })?;
    let scale = members
        .get(&spec.scale)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!(
                "expert ({}, {}) is missing {} scale",
                location.layer, location.expert, spec.role
            ),
        })?;
    let fp8 = is_e4m3(&source.dtype)
        && source.shape == [spec.rows as u64, spec.columns as u64]
        && is_ue8m0(&scale.dtype)
        && scale.shape == fp8_scale_shape(spec.rows, spec.columns);
    let fp4 = source.dtype == "I8"
        && source.shape == [spec.rows as u64, spec.columns.div_ceil(2) as u64]
        && is_ue8m0(&scale.dtype)
        && scale.shape == [spec.rows as u64, spec.columns.div_ceil(32) as u64];
    if !fp8 && !fp4 {
        return invalid(
            root,
            format!(
                "expert ({}, {}) {} has weight dtype/shape {:?}/{:?} and scale dtype/shape {:?}/{:?}; expected FP8 F8_E4M3/[{}, {}] with F8_E8M0/{:?}, or packed MXFP4 I8/[{}, {}] with F8_E8M0/[{}, {}]",
                location.layer,
                location.expert,
                spec.role,
                source.dtype,
                source.shape,
                scale.dtype,
                scale.shape,
                spec.rows,
                spec.columns,
                fp8_scale_shape(spec.rows, spec.columns),
                spec.rows,
                spec.columns.div_ceil(2),
                spec.rows,
                spec.columns.div_ceil(32),
            ),
        );
    }
    Ok(Matrix {
        source: (*source).clone(),
        rows: spec.rows,
        columns: spec.columns,
        scale: Some((*scale).clone()),
    })
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
    let values =
        config
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

fn validate_attention_and_hc(
    root: &Path,
    tensors: &BTreeMap<String, TensorRef>,
    layer: u32,
    geometry: &ModelGeometry,
    output: &mut BTreeMap<String, TensorRef>,
) -> Result<()> {
    let prefix = format!("layers.{layer}");
    let output_group_width = (geometry.attention_heads / geometry.o_groups) * geometry.head_dim;
    let output_width = geometry.o_groups * geometry.o_lora_rank;
    let fp8 = [
        ("attn.wkv", geometry.head_dim, geometry.hidden_size),
        ("attn.wo_a", output_width, output_group_width),
        ("attn.wo_b", geometry.hidden_size, output_width),
        ("attn.wq_a", geometry.q_lora_rank, geometry.hidden_size),
        (
            "attn.wq_b",
            geometry.attention_heads * geometry.head_dim,
            geometry.q_lora_rank,
        ),
    ];
    for (role, rows, columns) in fp8 {
        output.insert(
            format!("{role}.weight"),
            validate_tensor(
                root,
                tensors,
                &format!("{prefix}.{role}.weight"),
                "F8_E4M3FN",
                &[rows as u64, columns as u64],
            )?,
        );
        output.insert(
            format!("{role}.scale"),
            validate_tensor(
                root,
                tensors,
                &format!("{prefix}.{role}.scale"),
                "F8_E8M0",
                &fp8_scale_shape(rows, columns),
            )?,
        );
    }
    for (role, dtype, shape) in [
        (
            "attn.attn_sink",
            "F32",
            vec![geometry.attention_heads as u64],
        ),
        (
            "attn.kv_norm.weight",
            "BF16",
            vec![geometry.head_dim as u64],
        ),
        (
            "attn.q_norm.weight",
            "BF16",
            vec![geometry.q_lora_rank as u64],
        ),
        (
            "attn_norm.weight",
            "BF16",
            vec![geometry.hidden_size as u64],
        ),
    ] {
        output.insert(
            role.into(),
            validate_tensor(root, tensors, &format!("{prefix}.{role}"), dtype, &shape)?,
        );
    }
    let ratio = geometry.compression_ratios[layer as usize];
    if ratio != 0 {
        let coff = if ratio == 4 { 2 } else { 1 };
        for (role, dtype, shape) in [
            (
                "attn.compressor.ape",
                "F32",
                vec![ratio as u64, (coff * geometry.head_dim) as u64],
            ),
            (
                "attn.compressor.norm.weight",
                "BF16",
                vec![geometry.head_dim as u64],
            ),
            (
                "attn.compressor.wgate.weight",
                "BF16",
                vec![
                    (coff * geometry.head_dim) as u64,
                    geometry.hidden_size as u64,
                ],
            ),
            (
                "attn.compressor.wkv.weight",
                "BF16",
                vec![
                    (coff * geometry.head_dim) as u64,
                    geometry.hidden_size as u64,
                ],
            ),
        ] {
            output.insert(
                role.into(),
                validate_tensor(root, tensors, &format!("{prefix}.{role}"), dtype, &shape)?,
            );
        }
    }
    if ratio == 4 {
        let ih = geometry.index_head_dim;
        let heads = geometry.index_heads;
        for (role, dtype, shape) in [
            (
                "attn.indexer.compressor.ape",
                "F32",
                vec![4, (2 * ih) as u64],
            ),
            (
                "attn.indexer.compressor.norm.weight",
                "BF16",
                vec![ih as u64],
            ),
            (
                "attn.indexer.compressor.wgate.weight",
                "BF16",
                vec![(2 * ih) as u64, geometry.hidden_size as u64],
            ),
            (
                "attn.indexer.compressor.wkv.weight",
                "BF16",
                vec![(2 * ih) as u64, geometry.hidden_size as u64],
            ),
            (
                "attn.indexer.weights_proj.weight",
                "BF16",
                vec![heads as u64, geometry.hidden_size as u64],
            ),
            (
                "attn.indexer.wq_b.weight",
                "F8_E4M3FN",
                vec![(heads * ih) as u64, geometry.q_lora_rank as u64],
            ),
            (
                "attn.indexer.wq_b.scale",
                "F8_E8M0",
                fp8_scale_shape(heads * ih, geometry.q_lora_rank),
            ),
        ] {
            output.insert(
                role.into(),
                validate_tensor(root, tensors, &format!("{prefix}.{role}"), dtype, &shape)?,
            );
        }
    }
    let hc_params = (2 + geometry.hc_mult) * geometry.hc_mult;
    for (role, shape) in [
        ("hc_attn_base", vec![hc_params as u64]),
        (
            "hc_attn_fn",
            vec![
                hc_params as u64,
                (geometry.hc_mult * geometry.hidden_size) as u64,
            ],
        ),
        ("hc_attn_scale", vec![3]),
        ("hc_ffn_base", vec![hc_params as u64]),
        (
            "hc_ffn_fn",
            vec![
                hc_params as u64,
                (geometry.hc_mult * geometry.hidden_size) as u64,
            ],
        ),
        ("hc_ffn_scale", vec![3]),
    ] {
        output.insert(
            role.into(),
            validate_tensor(root, tensors, &format!("{prefix}.{role}"), "F32", &shape)?,
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
    if !dtype_matches || (!shape.is_empty() && tensor.shape != shape) {
        return invalid(
            root,
            format!(
                "tensor `{name}` has dtype/shape {:?}/{:?}, expected {dtype}/{shape:?}",
                tensor.dtype, tensor.shape
            ),
        );
    }
    Ok(tensor.clone())
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

    static FIXTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn fixture_root(num_hash_layers: u32) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let sequence = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "colic-v4-ir-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            format!(
                r#"{{"model_type":"deepseek_v4","hidden_size":2,"num_hidden_layers":1,"n_routed_experts":2,"moe_intermediate_size":3,"vocab_size":4,"hc_mult":2,"num_hash_layers":{num_hash_layers},"num_experts_per_tok":1,"num_attention_heads":1,"head_dim":2,"q_lora_rank":1,"o_groups":1,"o_lora_rank":1,"index_n_heads":1,"index_head_dim":1,"compress_ratios":[0]}}"#
            ),
        )
        .unwrap();
        root
    }

    fn tensor_with_dtype(dtype: &str, shape: &[u64]) -> TensorRef {
        TensorRef {
            source: "fixture.safetensors".into(),
            offset: 0,
            len: 0,
            dtype: dtype.into(),
            shape: shape.to_vec(),
        }
    }

    fn source(complete: bool) -> SourceInventory {
        source_with_hash_layers(complete, 1)
    }

    fn source_with_hash_layers(complete: bool, num_hash_layers: u32) -> SourceInventory {
        let root = fixture_root(num_hash_layers);
        let mut tensors = BTreeMap::new();
        for expert in 0..2 {
            for (name, shape) in [
                ("w1", &[3, 2][..]),
                ("w2", &[2, 3][..]),
                ("w3", &[3, 2][..]),
            ] {
                tensors.insert(
                    format!("layers.0.ffn.experts.{expert}.{name}.weight"),
                    tensor_with_dtype("F8_E4M3FN", shape),
                );
                if complete || !(expert == 1 && name == "w3") {
                    tensors.insert(
                        format!("layers.0.ffn.experts.{expert}.{name}.scale"),
                        tensor_with_dtype("F8_E8M0", &[1, 1]),
                    );
                }
            }
        }
        for (name, dtype, shape) in [
            ("embed.weight", "BF16", &[4, 2][..]),
            ("head.weight", "BF16", &[4, 2][..]),
            ("norm.weight", "BF16", &[2][..]),
            ("hc_head_base", "F32", &[2][..]),
            ("hc_head_fn", "F32", &[2, 4][..]),
            ("hc_head_scale", "F32", &[1][..]),
        ] {
            tensors.insert(
                name.into(),
                TensorRef {
                    source: "fixture.safetensors".into(),
                    offset: 0,
                    len: 0,
                    dtype: dtype.into(),
                    shape: shape.to_vec(),
                },
            );
        }
        for (name, dtype, shape) in [
            ("layers.0.attn.attn_sink", "F32", &[1][..]),
            ("layers.0.attn.kv_norm.weight", "BF16", &[2][..]),
            ("layers.0.attn.q_norm.weight", "BF16", &[1][..]),
            ("layers.0.attn.wkv.weight", "F8_E4M3FN", &[2, 2][..]),
            ("layers.0.attn.wkv.scale", "F8_E8M0", &[1, 1][..]),
            ("layers.0.attn.wo_a.weight", "F8_E4M3FN", &[1, 2][..]),
            ("layers.0.attn.wo_a.scale", "F8_E8M0", &[1, 1][..]),
            ("layers.0.attn.wo_b.weight", "F8_E4M3FN", &[2, 1][..]),
            ("layers.0.attn.wo_b.scale", "F8_E8M0", &[1, 1][..]),
            ("layers.0.attn.wq_a.weight", "F8_E4M3FN", &[1, 2][..]),
            ("layers.0.attn.wq_a.scale", "F8_E8M0", &[1, 1][..]),
            ("layers.0.attn.wq_b.weight", "F8_E4M3FN", &[2, 1][..]),
            ("layers.0.attn.wq_b.scale", "F8_E8M0", &[1, 1][..]),
            ("layers.0.attn_norm.weight", "BF16", &[2][..]),
            ("layers.0.hc_attn_base", "F32", &[8][..]),
            ("layers.0.hc_attn_fn", "F32", &[8, 4][..]),
            ("layers.0.hc_attn_scale", "F32", &[3][..]),
            ("layers.0.hc_ffn_base", "F32", &[8][..]),
            ("layers.0.hc_ffn_fn", "F32", &[8, 4][..]),
            ("layers.0.hc_ffn_scale", "F32", &[3][..]),
        ] {
            tensors.insert(
                name.into(),
                TensorRef {
                    source: "fixture.safetensors".into(),
                    offset: 0,
                    len: 0,
                    dtype: dtype.into(),
                    shape: shape.to_vec(),
                },
            );
        }
        for (name, dtype, shape) in [
            ("layers.0.ffn.gate.weight", "BF16", &[2, 2][..]),
            (
                "layers.0.ffn.shared_experts.w1.weight",
                "F8_E4M3FN",
                &[3, 2][..],
            ),
            (
                "layers.0.ffn.shared_experts.w2.weight",
                "F8_E4M3FN",
                &[2, 3][..],
            ),
            (
                "layers.0.ffn.shared_experts.w3.weight",
                "F8_E4M3FN",
                &[3, 2][..],
            ),
            (
                "layers.0.ffn.shared_experts.w1.scale",
                "F8_E8M0",
                &[1, 1][..],
            ),
            (
                "layers.0.ffn.shared_experts.w2.scale",
                "F8_E8M0",
                &[1, 1][..],
            ),
            (
                "layers.0.ffn.shared_experts.w3.scale",
                "F8_E8M0",
                &[1, 1][..],
            ),
            ("layers.0.ffn_norm.weight", "BF16", &[2][..]),
        ] {
            tensors.insert(
                name.into(),
                TensorRef {
                    source: "fixture.safetensors".into(),
                    offset: 0,
                    len: 0,
                    dtype: dtype.into(),
                    shape: shape.to_vec(),
                },
            );
        }
        let (router_name, router_dtype, router_shape) = if num_hash_layers == 0 {
            ("layers.0.ffn.gate.bias", "F32", &[2][..])
        } else {
            ("layers.0.ffn.gate.tid2eid", "I64", &[4, 1][..])
        };
        tensors.insert(
            router_name.into(),
            TensorRef {
                source: "fixture.safetensors".into(),
                offset: 0,
                len: 0,
                dtype: router_dtype.into(),
                shape: router_shape.to_vec(),
            },
        );
        SourceInventory {
            root,
            files: vec![],
            tensors,
            source_stored_bytes: 0,
            dtype_counts: BTreeMap::new(),
            source_fingerprint: "fixture".into(),
            config_fingerprint: None,
            architecture_hint: Some("DeepseekV4ForCausalLM".into()),
        }
    }

    fn indexed_compression_source() -> SourceInventory {
        let mut source = source(true);
        fs::write(
            source.root.join("config.json"),
            r#"{"model_type":"deepseek_v4","hidden_size":2,"num_hidden_layers":1,"n_routed_experts":2,"moe_intermediate_size":3,"vocab_size":4,"hc_mult":2,"num_hash_layers":1,"num_experts_per_tok":1,"num_attention_heads":1,"head_dim":2,"q_lora_rank":1,"o_groups":1,"o_lora_rank":1,"index_n_heads":1,"index_head_dim":1,"compress_ratios":[4]}"#,
        )
        .unwrap();
        for (name, dtype, shape) in [
            ("layers.0.attn.compressor.ape", "F32", &[4, 4][..]),
            ("layers.0.attn.compressor.norm.weight", "BF16", &[2][..]),
            ("layers.0.attn.compressor.wgate.weight", "BF16", &[4, 2][..]),
            ("layers.0.attn.compressor.wkv.weight", "BF16", &[4, 2][..]),
            ("layers.0.attn.indexer.compressor.ape", "F32", &[4, 2][..]),
            (
                "layers.0.attn.indexer.compressor.norm.weight",
                "BF16",
                &[1][..],
            ),
            (
                "layers.0.attn.indexer.compressor.wgate.weight",
                "BF16",
                &[2, 2][..],
            ),
            (
                "layers.0.attn.indexer.compressor.wkv.weight",
                "BF16",
                &[2, 2][..],
            ),
            (
                "layers.0.attn.indexer.weights_proj.weight",
                "BF16",
                &[1, 2][..],
            ),
            (
                "layers.0.attn.indexer.wq_b.weight",
                "F8_E4M3FN",
                &[1, 1][..],
            ),
            ("layers.0.attn.indexer.wq_b.scale", "F8_E8M0", &[1, 1][..]),
        ] {
            source.tensors.insert(
                name.into(),
                TensorRef {
                    source: "fixture.safetensors".into(),
                    offset: 0,
                    len: 0,
                    dtype: dtype.into(),
                    shape: shape.to_vec(),
                },
            );
        }
        source
    }

    #[test]
    fn builds_deterministically_from_complete_expert_inventory() {
        let source = source(true);
        assert!(DeepSeekV4Frontend::probe(&source).unwrap());
        let model = DeepSeekV4Frontend::build(&source).unwrap();
        assert_eq!(model.routed_experts.len(), 2);
        assert!(model.global_tensors.contains_key("embed.weight"));
        assert!(model.layer_static_tensors[&0].contains_key("ffn.gate.tid2eid"));
        assert!(model.layer_static_tensors[&0].contains_key("attn.wq_b.weight"));
        assert!(model.resident_tensors.is_empty());
        assert_eq!(model.routed_experts[&(0, 1)].gate.rows, 3);
        assert_eq!(model.routed_experts[&(0, 1)].down.columns, 3);
        fs::remove_dir_all(source.root).unwrap();
    }

    #[test]
    fn rejects_incomplete_expert_members() {
        let source = source(false);
        assert!(matches!(
            DeepSeekV4Frontend::build(&source),
            Err(ColicError::InvalidSource { .. })
        ));
        fs::remove_dir_all(source.root).unwrap();
    }

    #[test]
    fn accepts_non_hash_router_layers() {
        let source = source_with_hash_layers(true, 0);
        assert!(DeepSeekV4Frontend::build(&source).is_ok());
        fs::remove_dir_all(source.root).unwrap();
    }

    #[test]
    fn accepts_fnu_ue8m0_scale_spelling() {
        let mut source = source(true);
        source
            .tensors
            .get_mut("layers.0.ffn.experts.0.w1.scale")
            .unwrap()
            .dtype = "F8_E8M0FNU".into();
        assert!(DeepSeekV4Frontend::build(&source).is_ok());
        fs::remove_dir_all(source.root).unwrap();
    }

    #[test]
    fn rejects_missing_static_router_tensor() {
        let mut source = source(true);
        source.tensors.remove("layers.0.ffn.gate.weight");
        assert!(matches!(
            DeepSeekV4Frontend::build(&source),
            Err(ColicError::InvalidSource { .. })
        ));
        fs::remove_dir_all(source.root).unwrap();
    }

    #[test]
    fn rejects_missing_attention_tensor() {
        let mut source = source(true);
        source.tensors.remove("layers.0.attn.wq_b.weight");
        assert!(matches!(
            DeepSeekV4Frontend::build(&source),
            Err(ColicError::InvalidSource { .. })
        ));
        fs::remove_dir_all(source.root).unwrap();
    }

    #[test]
    fn classifies_indexed_compression_roles() {
        let source = indexed_compression_source();
        let model = DeepSeekV4Frontend::build(&source).unwrap();
        assert!(model.layer_static_tensors[&0].contains_key("attn.compressor.wkv.weight"));
        assert!(model.layer_static_tensors[&0].contains_key("attn.indexer.wq_b.weight"));
        assert!(model.resident_tensors.is_empty());
        fs::remove_dir_all(source.root).unwrap();
    }
}

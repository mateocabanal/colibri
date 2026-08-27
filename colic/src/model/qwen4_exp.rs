//! Qwen3.8-Flash-Next (Qwen4-Exp) frontend: FP8 block-scaled experts.
//!
//! The checkpoint stores routed experts as SEPARATE gate/up/down F8_E4M3
//! matrices with per-128×128-block BF16 `weight_scale_inv` scales. Everything
//! else (dense, GDN, QSA indexer, hc, PLE n-gram shards, embed, lm_head, MTP)
//! is carried as exact records by exclusion, so the compiler can package the
//! full model while the planner decides per-family requantization.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    ir::{Architecture, Matrix, ModelGeometry, RoutedExpert, SemanticModel},
    source::{self, SourceInventory, TensorRef},
};

pub struct Qwen4ExpFrontend;

impl Qwen4ExpFrontend {
    pub fn probe(source: &SourceInventory) -> Result<bool> {
        let Some(config) = source::config(&source.root)? else {
            return Ok(false);
        };
        // Top-level or text_config `model_type` (vision wrapper or bare text).
        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .or_else(|| {
                config
                    .get("text_config")
                    .and_then(|tc| tc.get("model_type"))
                    .and_then(Value::as_str)
            });
        Ok(model_type == Some("qwen4_exp"))
    }

    pub fn build(source: &SourceInventory) -> Result<SemanticModel> {
        let config = source::config(&source.root)?.ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "Qwen4-Exp source is missing config.json".into(),
        })?;
        let tc = config
            .get("text_config")
            .and_then(Value::as_object)
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "Qwen4-Exp config is missing `text_config`".into(),
            })?;
        let tc = Value::Object(tc.clone());
        if tc.get("model_type").and_then(Value::as_str) != Some("qwen4_exp_text") {
            return invalid(
                &source.root,
                format!(
                    "text_config model_type is not `qwen4_exp_text` (got {:?})",
                    tc.get("model_type").and_then(Value::as_str)
                ),
            );
        }

        let layers = required_u32(&source.root, &tc, "num_hidden_layers")?;
        let layer_types = tc
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "text_config is missing `layer_types`".into(),
            })?;
        if layer_types.len() != layers as usize {
            return invalid(
                &source.root,
                format!(
                    "`layer_types` has {} entries but `num_hidden_layers` is {layers}",
                    layer_types.len()
                ),
            );
        }
        for (layer, layer_type) in layer_types.iter().enumerate() {
            match layer_type.as_str() {
                Some("full_attention" | "linear_attention") => {}
                other => {
                    return invalid(
                        &source.root,
                        format!("layer {layer} has unsupported layer type {other:?}"),
                    );
                }
            }
        }

        let geometry = ModelGeometry {
            hidden_size: required_u32(&source.root, &tc, "hidden_size")?,
            layers,
            routed_experts_per_layer: required_u32(&source.root, &tc, "num_experts")?,
            moe_intermediate_size: required_u32(&source.root, &tc, "moe_intermediate_size")?,
            vocab_size: required_u32(&source.root, &tc, "vocab_size")?,
            hc_mult: tc.get("hc_count").and_then(Value::as_u64).unwrap_or(0) as u32,
            num_hash_layers: 0,
            experts_per_token: required_u32(&source.root, &tc, "num_experts_per_tok")?,
            attention_heads: required_u32(&source.root, &tc, "num_attention_heads")?,
            head_dim: required_u32(&source.root, &tc, "head_dim")?,
            num_key_value_heads: required_u32(&source.root, &tc, "num_key_value_heads")?,
            linear_key_head_dim: required_u32(&source.root, &tc, "linear_key_head_dim")?,
            q_lora_rank: 0,
            o_groups: 1,
            o_lora_rank: 0,
            index_heads: tc
                .get("indexer_n_heads")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32,
            index_head_dim: tc
                .get("indexer_head_dim")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32,
            compression_ratios: vec![0; layers as usize],
        };

        let inter = geometry.moe_intermediate_size;
        let hidden = geometry.hidden_size;
        let prefix = "model.language_model.layers";

        // ---- routed experts: separate FP8 gate/up/down + block scales ----
        let mut routed_experts = BTreeMap::new();
        for layer in 0..geometry.layers {
            let lp = format!("{prefix}.{layer}.mlp.experts");
            for expert in 0..geometry.routed_experts_per_layer {
                let gate = fp8_expert_matrix(source, &lp, layer, expert, "gate", inter, hidden)?;
                let up = fp8_expert_matrix(source, &lp, layer, expert, "up", inter, hidden)?;
                let down = fp8_expert_matrix(source, &lp, layer, expert, "down", hidden, inter)?;
                routed_experts.insert((layer, expert), RoutedExpert { layer, expert, gate, up, down });
            }
        }

        // ---- MTP experts (stage 0 = layer n_layers) when present ----
        let mtp_experts = tc
            .get("mtp")
            .and_then(|mtp| mtp.get("num_hidden_layers"))
            .and_then(Value::as_u64)
            .or_else(|| tc.get("mtp_num_hidden_layers").and_then(Value::as_u64))
            .unwrap_or(0) as u32;
        for stage in 0..mtp_experts {
            let layer = geometry.layers + stage;
            let lp = format!("mtp.layers.{stage}.mlp.experts");
            for expert in 0..geometry.routed_experts_per_layer {
                let gate = fp8_expert_matrix(source, &lp, layer, expert, "gate", inter, hidden)?;
                let up = fp8_expert_matrix(source, &lp, layer, expert, "up", inter, hidden)?;
                let down = fp8_expert_matrix(source, &lp, layer, expert, "down", hidden, inter)?;
                routed_experts.insert((layer, expert), RoutedExpert { layer, expert, gate, up, down });
            }
        }

        // ---- global tensors ----
        let mut global_tensors: BTreeMap<String, TensorRef> = BTreeMap::new();
        global_tensors.insert(
            "embed.weight".into(),
            validate_tensor(
                &source.root,
                &source.tensors,
                "model.language_model.embed_tokens.weight",
                "BF16",
                &[u64::from(geometry.vocab_size), u64::from(geometry.hidden_size)],
            )?,
        );
        // Qwen3.8-Next has no final rmsnorm (the hyper-connection mixer
        // replaces it); carry it only when present.
        if let Some(norm) = source.tensors.get("model.language_model.norm.weight") {
            global_tensors.insert("norm.weight".into(), norm.clone());
        }
        global_tensors.insert(
            "head.weight".into(),
            validate_tensor(
                &source.root,
                &source.tensors,
                "lm_head.weight",
                "BF16",
                &[u64::from(geometry.vocab_size), u64::from(geometry.hidden_size)],
            )?,
        );

        // ---- layer-static: everything except experts, by real source name ----
        let mut layer_static_tensors = BTreeMap::new();
        let mut expert_owned: BTreeSet<String> = BTreeSet::new();
        let mut static_owned: BTreeSet<String> = BTreeSet::new();
        for layer in 0..geometry.layers {
            let lp = format!("{prefix}.{layer}");
            let layer_prefix = format!("{lp}.");
            let mut static_tensors = BTreeMap::new();
            for (name, _) in source.tensors.range(format!("{lp}.\u{0}")..) {
                if !name.starts_with(&layer_prefix) {
                    break;
                }
                if is_expert_tensor_name(name) {
                    expert_owned.insert(name.clone());
                    continue;
                }
                let role = name.trim_start_matches(&layer_prefix).to_string();
                static_tensors.insert(role, tensor_by_name(source, name)?.clone());
                static_owned.insert(name.clone());
            }
            layer_static_tensors.insert(layer, static_tensors);
        }

        // ---- MTP layer-static tensors ----
        for stage in 0..mtp_experts {
            let lp = format!("mtp.layers.{stage}");
            let layer_prefix = format!("{lp}.");
            let mut static_tensors = BTreeMap::new();
            for (name, _) in source.tensors.range(format!("{lp}.\u{0}")..) {
                if !name.starts_with(&layer_prefix) {
                    break;
                }
                if is_expert_tensor_name(name) {
                    expert_owned.insert(name.clone());
                    continue;
                }
                static_tensors.insert(
                    format!("mtp.{stage}.{}", name.trim_start_matches(&layer_prefix)),
                    tensor_by_name(source, name)?.clone(),
                );
                static_owned.insert(name.clone());
            }
            layer_static_tensors.insert(geometry.layers + stage, static_tensors);
        }

        // ---- global + MTP-head statics by exclusion ----
        let global_owned: BTreeSet<String> = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let mut resident_tensors = BTreeMap::new();
        for name in source.tensors.keys() {
            if expert_owned.contains(name) || static_owned.contains(name) || global_owned.contains(name) {
                continue;
            }
            resident_tensors.insert(name.clone(), tensor_by_name(source, name)?.clone());
        }

        Ok(SemanticModel {
            architecture: Architecture::Qwen3_5MoeMoE,
            geometry,
            routed_experts,
            global_tensors,
            layer_static_tensors,
            resident_tensors,
        })
    }
}

/// Tensor names belonging to expert payloads (both data + block scales),
/// including the MTP stage copies.
fn is_expert_tensor_name(name: &str) -> bool {
    if name.contains("weight_scale_inv") {
        return name.contains(".mlp.experts.");
    }
    name.contains(".mlp.experts.")
}

/// Map a canonical global key to its source tensor name for exclusion.
fn t_key(tensor: &TensorRef) -> String {
    // TensorRef carries no name; exclusion is done via global_sources below.
    String::new()
}

/// Build an expert Matrix (F8 payload) + verify its 128×128 block scale ref.
fn fp8_expert_matrix(
    source: &SourceInventory,
    experts_path: &str,
    _layer: u32,
    expert: u32,
    role: &str,
    rows: u32,
    columns: u32,
) -> Result<Matrix> {
    let weight_name = format!("{experts_path}.{expert}.{role}_proj.weight");
    let scale_name = format!("{experts_path}.{expert}.{role}_proj.weight_scale_inv");
    let weight = tensor_by_name(source, &weight_name)?;
    let scale = tensor_by_name(source, &scale_name)?;
    if weight.dtype != "F8_E4M3" {
        return invalid(
            &source.root,
            format!("`{weight_name}` dtype is {} (expected F8_E4M3)", weight.dtype),
        );
    }
    let expected_weight = u64::from(rows) * u64::from(columns);
    if weight.shape != [u64::from(rows), u64::from(columns)] || weight.len != expected_weight {
        return invalid(
            &source.root,
            format!(
                "`{weight_name}` shape/len {:?}/{} != [{rows},{columns}]/{expected_weight}",
                weight.shape, weight.len
            ),
        );
    }
    let block_rows = rows.div_ceil(128);
    let block_columns = columns.div_ceil(128);
    let expected_scale = u64::from(block_rows) * u64::from(block_columns) * 2;
    if scale.shape != [u64::from(block_rows), u64::from(block_columns)]
        || scale.dtype != "BF16"
        || scale.len != expected_scale
    {
        return invalid(
            &source.root,
            format!(
                "`{scale_name}` shape/dtype/len {:?}/{}/{} != [{block_rows},{block_columns}]/BF16/{expected_scale}",
                scale.shape, scale.dtype, scale.len, scale_name = scale_name
            ),
        );
    }
    Ok(Matrix {
        source: TensorRef {
            source: weight.source.clone(),
            offset: weight.offset,
            len: weight.len,
            dtype: "F8_E4M3".into(),
            shape: weight.shape.clone(),
        },
        rows,
        columns,
        scale: Some(TensorRef {
            source: scale.source.clone(),
            offset: scale.offset,
            len: scale.len,
            dtype: "BF16".into(),
            shape: scale.shape.clone(),
        }),
    })
}

fn tensor_by_name<'a>(source: &'a SourceInventory, name: &str) -> Result<&'a TensorRef> {
    source
        .tensors
        .get(name)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("missing required tensor `{name}`"),
        })
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
    if tensor.dtype != dtype || (!shape.is_empty() && tensor.shape != shape) {
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

fn required_u32(root: &Path, config: &Value, field: &str) -> Result<u32> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| value.try_into().ok())
        .filter(|value: &u32| *value > 0)
        .ok_or_else(|| ColicError::InvalidSource {
            path: root.to_owned(),
            detail: format!("config `{field}` must be a positive u32"),
        })
}

fn invalid<T>(path: &Path, detail: impl Into<String>) -> Result<T> {
    Err(ColicError::InvalidSource {
        path: path.to_owned(),
        detail: detail.into(),
    })
}
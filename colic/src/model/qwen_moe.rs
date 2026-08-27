//! Qwen3.5 / Qwen3.6 / Qwen3.7 fine-grained MoE source frontend for colic.
//!
//! Mirrors `deepseek_v4.rs` in structure, but targets the HF `qwen3_5_moe`
//! family (model_type `qwen3_5_moe`, arch `Qwen3_5MoeForConditionalGeneration`
//! — the vision wrapper whose text backbone config lives in `text_config`).
//!
//! Key differences from DeepSeek V4:
//! - Text backbone tensors live under `model.language_model.*` (vision under
//!   `model.visual.*` is ignored; `lm_head.weight` is top-level, no prefix).
//! - Routed experts are FUSED per layer: `mlp.experts.gate_up_proj` [E,2·I,H]
//!   and `mlp.experts.down_proj` [E,H,I], all BF16, NO `.weight` suffix and NO
//!   per-expert scale. Each routed expert therefore becomes a sub-`TensorRef`
//!   slicing the fused payload on a per-layer `offset`/`len` (see `read_tensor`
//!   contract in target/mod.rs: it reads exactly `offset..offset+len`).
//! - Hybrid attention: `full_attention` layers (self_attn, every 4th) vs
//!   `linear_attention` layers (Gated DeltaNet / Mamba-style `linear_attn`).
//! - Shared expert + router gate + shared_expert_gate are layer-static.
//! - No HC, no hash router, no compression ratios, no FP8 scales.

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

pub struct QwenMoeFrontend;

impl QwenMoeFrontend {
    pub fn probe(source: &SourceInventory) -> Result<bool> {
        let Some(config) = source::config(&source.root)? else {
            return Ok(false);
        };
        Ok(config.get("model_type").and_then(Value::as_str) == Some("qwen3_5_moe"))
    }

    pub fn build(source: &SourceInventory) -> Result<SemanticModel> {
        let config = source::config(&source.root)?.ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "Qwen MoE source is missing config.json".into(),
        })?;
        if config.get("model_type").and_then(Value::as_str) != Some("qwen3_5_moe") {
            return invalid(&source.root, "config model_type is not `qwen3_5_moe`");
        }
        // The real geometry lives in text_config (vision wrapper).
        let tc = config
            .get("text_config")
            .and_then(Value::as_object)
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "Qwen MoE config is missing `text_config`".into(),
            })?;
        let tc = Value::Object(tc.clone());

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

        let shared_intermediate_size =
            required_u32(&source.root, &tc, "shared_expert_intermediate_size")?;
        let linear_num_key_heads = required_u32(&source.root, &tc, "linear_num_key_heads")?;
        let linear_num_value_heads = required_u32(&source.root, &tc, "linear_num_value_heads")?;
        let linear_value_head_dim = required_u32(&source.root, &tc, "linear_value_head_dim")?;
        let linear_conv_kernel_dim = required_u32(&source.root, &tc, "linear_conv_kernel_dim")?;

        let geometry = ModelGeometry {
            hidden_size: required_u32(&source.root, &tc, "hidden_size")?,
            layers,
            routed_experts_per_layer: required_u32(&source.root, &tc, "num_experts")?,
            moe_intermediate_size: required_u32(&source.root, &tc, "moe_intermediate_size")?,
            vocab_size: required_u32(&source.root, &tc, "vocab_size")?,
            // Fields below are unused by downstream for Qwen (colic only reads
            // geometry.layers), but remain in the shared struct.
            hc_mult: 1,
            num_hash_layers: 0,
            experts_per_token: required_u32(&source.root, &tc, "num_experts_per_tok")?,
            attention_heads: required_u32(&source.root, &tc, "num_attention_heads")?,
            head_dim: required_u32(&source.root, &tc, "head_dim")?,
            num_key_value_heads: required_u32(&source.root, &tc, "num_key_value_heads")?,
            linear_key_head_dim: required_u32(&source.root, &tc, "linear_key_head_dim")?,
            q_lora_rank: 0,
            o_groups: 1,
            o_lora_rank: 0,
            index_heads: 0,
            index_head_dim: 0,
            compression_ratios: vec![0; layers as usize],
        };

        let linear_key_width = u64::from(linear_num_key_heads)
            .checked_mul(u64::from(geometry.linear_key_head_dim))
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "linear-attention key width overflows u64".into(),
            })?;
        let linear_value_width = u64::from(linear_num_value_heads)
            .checked_mul(u64::from(linear_value_head_dim))
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "linear-attention value width overflows u64".into(),
            })?;
        let linear_qkv_width = linear_key_width
            .checked_mul(2)
            .and_then(|value| value.checked_add(linear_value_width))
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "linear-attention QKV width overflows u64".into(),
            })?;

        // ---- routed experts: slice the fused per-layer tensors ----
        let inter = geometry.moe_intermediate_size;
        let hidden = geometry.hidden_size;
        let prefix = "model.language_model.layers";

        let mut routed_experts = BTreeMap::new();
        for layer in 0..geometry.layers {
            let lp = format!("{prefix}.{layer}.mlp.experts");
            let gu_name = format!("{lp}.gate_up_proj");
            let dn_name = format!("{lp}.down_proj");
            let guz = tensor_by_name(source, &gu_name)?;
            let dz = tensor_by_name(source, &dn_name)?;
            // gate_up_proj is [E, 2·I, H] and down_proj is [E, H, I], both BF16.
            let expected_gu_bytes =
                geometry.routed_experts_per_layer as u64 * 2 * inter as u64 * hidden as u64 * 2;
            let expected_d_bytes =
                geometry.routed_experts_per_layer as u64 * hidden as u64 * inter as u64 * 2;
            if guz.len != expected_gu_bytes {
                return invalid(
                    &source.root,
                    format!(
                        "layer {layer} gate_up_proj len {} != expected {expected_gu_bytes}",
                        guz.len
                    ),
                );
            }
            if dz.len != expected_d_bytes {
                return invalid(
                    &source.root,
                    format!(
                        "layer {layer} down_proj len {} != expected {expected_d_bytes}",
                        dz.len
                    ),
                );
            }
            for expert in 0..geometry.routed_experts_per_layer {
                let gate = slice_fused(guz, expert, 2 * inter, hidden, 0, inter)?;
                let up = slice_fused(guz, expert, 2 * inter, hidden, inter, inter)?;
                let down = slice_fused(dz, expert, hidden, inter, 0, hidden)?;
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

        // ---- MTP routed experts: layer = n_layers + stage (speculative head) ----
        // The MTP head is one full-attention MoE layer per stage with the same
        // fused expert layout as the main layers; its experts ride the same
        // (layer, expert) index space at layer n_layers + stage.
        if let Some(mtp) = crate::model::qwen_mtp::inspect(source)? {
            for (stage, st) in mtp.stages.iter().enumerate() {
                let layer = geometry.layers + stage as u32;
                for expert in 0..mtp.experts {
                    let gate =
                        slice_fused(&st.expert_gate_up, expert, 2 * inter, hidden, 0, inter)?;
                    let up =
                        slice_fused(&st.expert_gate_up, expert, 2 * inter, hidden, inter, inter)?;
                    let down = slice_fused(&st.expert_down, expert, hidden, inter, 0, hidden)?;
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
        }

        // ---- global tensors (embed, final norm, lm_head) ----
        let mut global_tensors: BTreeMap<String, TensorRef> = BTreeMap::new();
        global_tensors.insert(
            "embed.weight".into(),
            validate_tensor(
                &source.root,
                &source.tensors,
                "model.language_model.embed_tokens.weight",
                "BF16",
                &[geometry.vocab_size as u64, geometry.hidden_size as u64],
            )?,
        );
        global_tensors.insert(
            "norm.weight".into(),
            validate_tensor(
                &source.root,
                &source.tensors,
                "model.language_model.norm.weight",
                "BF16",
                &[geometry.hidden_size as u64],
            )?,
        );
        global_tensors.insert(
            "head.weight".into(),
            validate_tensor(
                &source.root,
                &source.tensors,
                "lm_head.weight",
                "BF16",
                &[geometry.vocab_size as u64, geometry.hidden_size as u64],
            )?,
        );

        // ---- layer-static tensors ----
        let mut layer_static_tensors = BTreeMap::new();
        for layer in 0..geometry.layers {
            let lp = format!("{prefix}.{layer}");
            let mut static_tensors = BTreeMap::new();
            let is_full = layer_types[layer as usize].as_str() == Some("full_attention");
            if is_full {
                // attn_output_gate=true doubles q but o stays single-width.
                let q_width = 2 * geometry.attention_heads * geometry.head_dim;
                let kv_width = geometry.num_key_value_heads * geometry.head_dim;
                let o_width = geometry.attention_heads * geometry.head_dim;
                for (role, rows, columns) in [
                    ("q_proj", q_width, geometry.hidden_size),
                    ("k_proj", kv_width, geometry.hidden_size),
                    ("v_proj", kv_width, geometry.hidden_size),
                    ("o_proj", geometry.hidden_size, o_width),
                ] {
                    static_tensors.insert(
                        format!("attn.{role}.weight"),
                        validate_tensor(
                            &source.root,
                            &source.tensors,
                            &format!("{lp}.self_attn.{role}.weight"),
                            "BF16",
                            &[rows as u64, columns as u64],
                        )?,
                    );
                }
                for role in ["q_norm", "k_norm"] {
                    static_tensors.insert(
                        format!("attn.{role}.weight"),
                        validate_tensor(
                            &source.root,
                            &source.tensors,
                            &format!("{lp}.self_attn.{role}.weight"),
                            "BF16",
                            &[geometry.head_dim as u64],
                        )?,
                    );
                }
            } else {
                for (role, shape) in [
                    ("A_log", vec![u64::from(linear_num_value_heads)]),
                    ("dt_bias", vec![u64::from(linear_num_value_heads)]),
                    (
                        "conv1d.weight",
                        vec![linear_qkv_width, 1, u64::from(linear_conv_kernel_dim)],
                    ),
                    (
                        "in_proj_a.weight",
                        vec![
                            u64::from(linear_num_value_heads),
                            u64::from(geometry.hidden_size),
                        ],
                    ),
                    (
                        "in_proj_b.weight",
                        vec![
                            u64::from(linear_num_value_heads),
                            u64::from(geometry.hidden_size),
                        ],
                    ),
                    (
                        "in_proj_qkv.weight",
                        vec![linear_qkv_width, u64::from(geometry.hidden_size)],
                    ),
                    (
                        "in_proj_z.weight",
                        vec![linear_value_width, u64::from(geometry.hidden_size)],
                    ),
                    (
                        "out_proj.weight",
                        vec![u64::from(geometry.hidden_size), linear_value_width],
                    ),
                ] {
                    static_tensors.insert(
                        format!("linear_attn.{role}"),
                        validate_tensor(
                            &source.root,
                            &source.tensors,
                            &format!("{lp}.linear_attn.{role}"),
                            "BF16",
                            &shape,
                        )?,
                    );
                }
                static_tensors.insert(
                    "attn_norm.weight".into(),
                    validate_tensor(
                        &source.root,
                        &source.tensors,
                        &format!("{lp}.linear_attn.norm.weight"),
                        "BF16",
                        &[u64::from(linear_value_head_dim)],
                    )?,
                );
            }
            static_tensors.insert(
                "input_layernorm.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.input_layernorm.weight"),
                    "BF16",
                    &[geometry.hidden_size as u64],
                )?,
            );
            static_tensors.insert(
                "post_attention_layernorm.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.post_attention_layernorm.weight"),
                    "BF16",
                    &[geometry.hidden_size as u64],
                )?,
            );
            // router gate
            static_tensors.insert(
                "ffn.gate.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.gate.weight"),
                    "BF16",
                    &[
                        geometry.routed_experts_per_layer as u64,
                        geometry.hidden_size as u64,
                    ],
                )?,
            );
            // Shared expert projection gate must have a distinct semantic key
            // from the scalar shared_expert_gate. Keeping these separate avoids
            // silently replacing one record in the BTreeMap.
            static_tensors.insert(
                "ffn.shared_experts.gate_proj.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
                    "BF16",
                    &[
                        u64::from(shared_intermediate_size),
                        u64::from(geometry.hidden_size),
                    ],
                )?,
            );
            static_tensors.insert(
                "ffn.shared_experts.up.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.shared_expert.up_proj.weight"),
                    "BF16",
                    &[
                        u64::from(shared_intermediate_size),
                        u64::from(geometry.hidden_size),
                    ],
                )?,
            );
            static_tensors.insert(
                "ffn.shared_experts.down.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.shared_expert.down_proj.weight"),
                    "BF16",
                    &[
                        u64::from(geometry.hidden_size),
                        u64::from(shared_intermediate_size),
                    ],
                )?,
            );
            static_tensors.insert(
                "ffn.shared_experts.gate.weight".into(),
                validate_tensor(
                    &source.root,
                    &source.tensors,
                    &format!("{lp}.mlp.shared_expert_gate.weight"),
                    "BF16",
                    &[geometry.hidden_size as u64],
                )?,
            );
            layer_static_tensors.insert(layer, static_tensors);
        }

        // Map canonical layer-static roles back to real HF source names, for
        // resident-tensor exclusion.
        let mut static_sources = BTreeSet::new();
        for layer in 0..geometry.layers {
            let lp = format!("{prefix}.{layer}");
            for canon in layer_static_tensors[&layer].keys() {
                let src = if canon == "attn_norm.weight" {
                    format!("{lp}.linear_attn.norm.weight")
                } else if let Some(role) = canon.strip_prefix("attn.") {
                    format!("{lp}.self_attn.{role}")
                } else if canon.starts_with("linear_attn.") {
                    format!("{lp}.{canon}")
                } else if canon == "ffn.gate.weight" {
                    format!("{lp}.mlp.gate.weight")
                } else if canon == "ffn.shared_experts.gate.weight" {
                    format!("{lp}.mlp.shared_expert_gate.weight")
                } else if canon == "ffn.shared_experts.gate_proj.weight" {
                    format!("{lp}.mlp.shared_expert.gate_proj.weight")
                } else if canon.starts_with("ffn.shared_experts.") {
                    let role = canon
                        .trim_start_matches("ffn.shared_experts.")
                        .trim_end_matches(".weight");
                    format!("{lp}.mlp.shared_expert.{role}_proj.weight")
                } else if canon == "input_layernorm.weight" {
                    format!("{lp}.input_layernorm.weight")
                } else if canon == "post_attention_layernorm.weight" {
                    format!("{lp}.post_attention_layernorm.weight")
                } else {
                    format!("{lp}.{canon}")
                };
                static_sources.insert(src);
            }
        }
        let mut global_sources = BTreeSet::new();
        for canon in global_tensors.keys() {
            global_sources.insert(match canon.as_str() {
                "head.weight" => "lm_head.weight".to_owned(),
                "embed.weight" => "model.language_model.embed_tokens.weight".to_owned(),
                _ => format!("model.language_model.{canon}"),
            });
        }
        let resident_tensors = source
            .tensors
            .iter()
            .filter(|(name, _)| {
                // Exclude any tensor that is a global, a layer-static, or an
                // expert fused payload. Sub-slices share the same shard, so
                // exclusion must remain exact-name based rather than file based.
                !global_sources.contains(*name)
                    && !static_sources.contains(*name)
                    && !is_expert_fused_name(name)
            })
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();

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

/// A tensor name that is one of the two fused per-layer expert payloads.
/// (Used only to avoid double-classifying the same shard tensor as resident.)
fn is_expert_fused_name(name: &str) -> bool {
    match name.rsplit_once(".mlp.experts.") {
        Some((_, tail)) => tail == "gate_up_proj" || tail == "down_proj",
        None => false,
    }
}

/// Split a fused expert tensor's payload for one expert.
///
/// `fused` has shape [E, M, K] (contiguous BF16). For expert `e`, this returns
/// a sub-`TensorRef` covering rows `[row_start, row_start+row_len)` of the
/// M-axis, i.e. byte offset `e*M*K + row_start*K` (×2 for BF16), length
/// `row_len*K*2`. `read_tensor` in target/mod.rs reads exactly
/// `offset..offset+len`.
fn slice_fused(
    fused: &TensorRef,
    expert: u32,
    m: u32,
    k: u32,
    row_start: u32,
    row_len: u32,
) -> Result<Matrix> {
    if fused.dtype != "BF16" {
        return invalid(
            &fused.source,
            format!("fused expert payload is not BF16 (got {})", fused.dtype),
        );
    }
    let bytes_per_row = (k as u64) * 2;
    let base = fused
        .offset
        .checked_add((expert as u64) * (m as u64) * (k as u64) * 2)
        .ok_or_else(|| ColicError::InvalidSource {
            path: fused.source.clone(),
            detail: format!("expert {expert} offset overflows u64"),
        })?;
    let len = (row_len as u64) * bytes_per_row;
    Ok(Matrix {
        source: TensorRef {
            source: fused.source.clone(),
            offset: base + (row_start as u64) * bytes_per_row,
            len,
            dtype: "BF16".into(),
            shape: vec![row_len as u64, k as u64],
        },
        rows: row_len,
        columns: k,
        scale: None,
    })
}

fn tensor_by_name<'a>(source: &'a SourceInventory, name: &'a str) -> Result<&'a TensorRef> {
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
    let shape_matches = shape.is_empty()
        || tensor.shape == shape
        || (shape.len() == 1
            && tensor.shape.len() == 2
            && tensor.shape[0] == 1
            && tensor.shape[1] == shape[0]);
    if tensor.dtype != dtype || !shape_matches {
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

//! Qwen native MTP source inventory.
//!
//! This module deliberately stops at semantic/source classification. Runtime
//! proposal, transactional state, and target-specific lowering live elsewhere.

use std::{collections::BTreeMap, fs};

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    source::{SourceInventory, TensorRef},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenMtpStageInventory {
    pub stage: u32,
    /// Fused routed-expert gate+up payload: [experts, 2*intermediate, hidden].
    pub expert_gate_up: TensorRef,
    /// Fused routed-expert down payload: [experts, hidden, intermediate].
    pub expert_down: TensorRef,
    /// Stage-local tensors other than the independently pageable expert bank.
    /// Keys are relative to `mtp.layers.<stage>.`.
    pub static_tensors: BTreeMap<String, TensorRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenMtpInventory {
    pub hidden_layers: u32,
    pub use_dedicated_embeddings: bool,
    pub hidden_size: u32,
    pub experts: u32,
    pub moe_intermediate_size: u32,
    /// MTP-global tensors such as fc/norm/pre-fc norms. Keys preserve source names.
    pub global_tensors: BTreeMap<String, TensorRef>,
    pub stages: Vec<QwenMtpStageInventory>,
}

pub fn inspect(source: &SourceInventory) -> Result<Option<QwenMtpInventory>> {
    let config_path = source.root.join("config.json");
    let bytes = fs::read(&config_path).map_err(|source_error| ColicError::Io {
        path: config_path.clone(),
        source: source_error,
    })?;
    let config: Value =
        serde_json::from_slice(&bytes).map_err(|error| ColicError::InvalidSource {
            path: config_path.clone(),
            detail: format!("invalid config.json: {error}"),
        })?;
    let text_config = config
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| ColicError::InvalidSource {
            path: config_path.clone(),
            detail: "Qwen MoE config is missing `text_config`".into(),
        })?;

    let hidden_layers = optional_u32(
        &source.root,
        text_config.get("mtp_num_hidden_layers"),
        "mtp_num_hidden_layers",
    )?
    .unwrap_or(0);
    let has_mtp_tensors = source.tensors.keys().any(|name| name.starts_with("mtp."));
    if hidden_layers == 0 {
        if has_mtp_tensors {
            return invalid(
                source,
                "checkpoint contains `mtp.*` tensors but config declares no MTP hidden layers",
            );
        }
        return Ok(None);
    }
    if !has_mtp_tensors {
        return invalid(
            source,
            format!(
                "config declares {hidden_layers} MTP hidden layer(s) but checkpoint contains no `mtp.*` tensors"
            ),
        );
    }

    let use_dedicated_embeddings = text_config
        .get("mtp_use_dedicated_embeddings")
        .map(|value| {
            value.as_bool().ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "`mtp_use_dedicated_embeddings` is not a boolean".into(),
            })
        })
        .transpose()?
        .unwrap_or(false);
    let hidden_size = required_u32(source, text_config.get("hidden_size"), "hidden_size")?;
    let experts = required_u32(source, text_config.get("num_experts"), "num_experts")?;
    let moe_intermediate_size = required_u32(
        source,
        text_config.get("moe_intermediate_size"),
        "moe_intermediate_size",
    )?;
    let two_hidden = hidden_size
        .checked_mul(2)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: "MTP fc input width overflows u32".into(),
        })?;
    let two_intermediate =
        moe_intermediate_size
            .checked_mul(2)
            .ok_or_else(|| ColicError::InvalidSource {
                path: source.root.clone(),
                detail: "MTP fused gate/up width overflows u32".into(),
            })?;

    for (name, expected) in [
        ("mtp.fc.weight", vec![hidden_size as u64, two_hidden as u64]),
        ("mtp.norm.weight", vec![hidden_size as u64]),
        ("mtp.pre_fc_norm_embedding.weight", vec![hidden_size as u64]),
        ("mtp.pre_fc_norm_hidden.weight", vec![hidden_size as u64]),
    ] {
        require_shape(source, name, &expected)?;
    }

    let mut global_tensors = BTreeMap::new();
    for (name, tensor) in &source.tensors {
        if name.starts_with("mtp.") && !name.starts_with("mtp.layers.") {
            global_tensors.insert(name.clone(), tensor.clone());
        }
    }

    let mut stages = Vec::with_capacity(hidden_layers as usize);
    for stage in 0..hidden_layers {
        let prefix = format!("mtp.layers.{stage}.");
        let gate_up_name = format!("{prefix}mlp.experts.gate_up_proj");
        let down_name = format!("{prefix}mlp.experts.down_proj");
        let expert_gate_up = require_shape(
            source,
            &gate_up_name,
            &[experts as u64, two_intermediate as u64, hidden_size as u64],
        )?
        .clone();
        let expert_down = require_shape(
            source,
            &down_name,
            &[
                experts as u64,
                hidden_size as u64,
                moe_intermediate_size as u64,
            ],
        )?
        .clone();

        for role in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate.weight",
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert_gate.weight",
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ] {
            require_tensor(source, &format!("{prefix}{role}"))?;
        }

        let mut static_tensors = BTreeMap::new();
        for (name, tensor) in &source.tensors {
            let Some(role) = name.strip_prefix(&prefix) else {
                continue;
            };
            if name == &gate_up_name || name == &down_name {
                continue;
            }
            static_tensors.insert(role.to_owned(), tensor.clone());
        }
        stages.push(QwenMtpStageInventory {
            stage,
            expert_gate_up,
            expert_down,
            static_tensors,
        });
    }

    for name in source
        .tensors
        .keys()
        .filter(|name| name.starts_with("mtp.layers."))
    {
        let rest = &name["mtp.layers.".len()..];
        let Some((stage, _)) = rest.split_once('.') else {
            return invalid(source, format!("invalid MTP layer tensor name `{name}`"));
        };
        let stage: u32 = stage.parse().map_err(|_| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("invalid MTP layer tensor name `{name}`"),
        })?;
        if stage >= hidden_layers {
            return invalid(
                source,
                format!(
                    "checkpoint contains MTP layer {stage} but config declares only {hidden_layers} layer(s)"
                ),
            );
        }
    }

    Ok(Some(QwenMtpInventory {
        hidden_layers,
        use_dedicated_embeddings,
        hidden_size,
        experts,
        moe_intermediate_size,
        global_tensors,
        stages,
    }))
}

fn optional_u32(root: &std::path::Path, value: Option<&Value>, key: &str) -> Result<Option<u32>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let raw = value.as_u64().ok_or_else(|| ColicError::InvalidSource {
        path: root.to_owned(),
        detail: format!("`{key}` is not a non-negative integer"),
    })?;
    let value = u32::try_from(raw).map_err(|_| ColicError::InvalidSource {
        path: root.to_owned(),
        detail: format!("`{key}` exceeds u32"),
    })?;
    Ok(Some(value))
}

fn required_u32(source: &SourceInventory, value: Option<&Value>, key: &str) -> Result<u32> {
    optional_u32(&source.root, value, key)?.ok_or_else(|| ColicError::InvalidSource {
        path: source.root.clone(),
        detail: format!("Qwen text_config is missing `{key}`"),
    })
}

fn require_tensor<'a>(source: &'a SourceInventory, name: &str) -> Result<&'a TensorRef> {
    source
        .tensors
        .get(name)
        .ok_or_else(|| ColicError::InvalidSource {
            path: source.root.clone(),
            detail: format!("MTP checkpoint is missing required tensor `{name}`"),
        })
}

fn require_shape<'a>(
    source: &'a SourceInventory,
    name: &str,
    expected: &[u64],
) -> Result<&'a TensorRef> {
    let tensor = require_tensor(source, name)?;
    if tensor.shape != expected {
        return invalid(
            source,
            format!(
                "MTP tensor `{name}` has shape {:?}, expected {:?}",
                tensor.shape, expected
            ),
        );
    }
    Ok(tensor)
}

fn invalid<T>(source: &SourceInventory, detail: impl Into<String>) -> Result<T> {
    Err(ColicError::InvalidSource {
        path: source.root.clone(),
        detail: detail.into(),
    })
}

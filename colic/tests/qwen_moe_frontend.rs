use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use colic::{
    ir::Architecture,
    model::qwen_moe::QwenMoeFrontend,
    source::{SourceInventory, TensorRef},
};

struct Fixture {
    root: PathBuf,
    inventory: SourceInventory,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn tensor_len(shape: &[u64]) -> u64 {
    shape.iter().copied().product::<u64>() * 2
}

fn add_tensor(
    tensors: &mut BTreeMap<String, TensorRef>,
    source: &PathBuf,
    next_offset: &mut u64,
    name: impl Into<String>,
    shape: &[u64],
) {
    let len = tensor_len(shape);
    tensors.insert(
        name.into(),
        TensorRef {
            source: source.clone(),
            offset: *next_offset,
            len,
            dtype: "BF16".into(),
            shape: shape.to_vec(),
        },
    );
    *next_offset += len + 32;
}

fn fixture() -> Fixture {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "colic-qwen-frontend-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("config.json"),
        r#"{
          "model_type": "qwen3_5_moe",
          "text_config": {
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "hidden_size": 4,
            "num_experts": 2,
            "moe_intermediate_size": 3,
            "shared_expert_intermediate_size": 5,
            "vocab_size": 7,
            "num_experts_per_tok": 1,
            "num_attention_heads": 2,
            "head_dim": 2,
            "num_key_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 2,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 3,
            "linear_conv_kernel_dim": 4
          }
        }"#,
    )
    .unwrap();

    let weights = root.join("weights.bin");
    fs::write(&weights, []).unwrap();
    let mut tensors = BTreeMap::new();
    let mut offset = 1024;

    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        "model.language_model.embed_tokens.weight",
        &[7, 4],
    );
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        "model.language_model.norm.weight",
        &[4],
    );
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        "lm_head.weight",
        &[7, 4],
    );

    for layer in 0..2 {
        let lp = format!("model.language_model.layers.{layer}");
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.mlp.experts.gate_up_proj"),
            &[2, 6, 4],
        );
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.mlp.experts.down_proj"),
            &[2, 4, 3],
        );
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.input_layernorm.weight"),
            &[4],
        );
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.post_attention_layernorm.weight"),
            &[4],
        );
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.mlp.gate.weight"),
            &[2, 4],
        );
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.mlp.shared_expert.gate_proj.weight"),
            &[5, 4],
        );
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.mlp.shared_expert.up_proj.weight"),
            &[5, 4],
        );
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.mlp.shared_expert.down_proj.weight"),
            &[4, 5],
        );
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.mlp.shared_expert_gate.weight"),
            &[4],
        );
    }

    let lp = "model.language_model.layers.0";
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        format!("{lp}.linear_attn.A_log"),
        &[2],
    );
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        format!("{lp}.linear_attn.dt_bias"),
        &[2],
    );
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        format!("{lp}.linear_attn.conv1d.weight"),
        &[10, 1, 4],
    );
    for role in ["in_proj_a", "in_proj_b"] {
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.linear_attn.{role}.weight"),
            &[2, 4],
        );
    }
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        format!("{lp}.linear_attn.in_proj_qkv.weight"),
        &[10, 4],
    );
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        format!("{lp}.linear_attn.in_proj_z.weight"),
        &[6, 4],
    );
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        format!("{lp}.linear_attn.norm.weight"),
        &[3],
    );
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        format!("{lp}.linear_attn.out_proj.weight"),
        &[4, 6],
    );

    let lp = "model.language_model.layers.1";
    for (role, shape) in [
        ("q_proj", vec![8, 4]),
        ("k_proj", vec![2, 4]),
        ("v_proj", vec![2, 4]),
        ("o_proj", vec![4, 4]),
    ] {
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.self_attn.{role}.weight"),
            &shape,
        );
    }
    for role in ["q_norm", "k_norm"] {
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{lp}.self_attn.{role}.weight"),
            &[2],
        );
    }

    let inventory = SourceInventory {
        root: root.clone(),
        files: vec![root.join("config.json"), weights],
        source_stored_bytes: tensors.values().map(|tensor| tensor.len).sum(),
        dtype_counts: BTreeMap::from([("BF16".to_owned(), tensors.len() as u64)]),
        source_fingerprint: "00".repeat(32),
        config_fingerprint: None,
        architecture_hint: Some("Qwen3_5MoeForConditionalGeneration".into()),
        tensors,
    };

    Fixture { root, inventory }
}

#[test]
fn lowers_hybrid_qwen_roles_without_collisions() {
    let fixture = fixture();
    let model = QwenMoeFrontend::build(&fixture.inventory).unwrap();
    assert_eq!(model.architecture, Architecture::Qwen3_5MoeMoE);

    let gu = fixture
        .inventory
        .tensors
        .get("model.language_model.layers.0.mlp.experts.gate_up_proj")
        .unwrap();
    let down = fixture
        .inventory
        .tensors
        .get("model.language_model.layers.0.mlp.experts.down_proj")
        .unwrap();
    let expert = model.routed_experts.get(&(0, 1)).unwrap();
    assert_eq!(expert.gate.source.offset, gu.offset + 48);
    assert_eq!(expert.gate.source.len, 24);
    assert_eq!(expert.up.source.offset, gu.offset + 72);
    assert_eq!(expert.up.source.len, 24);
    assert_eq!(expert.down.source.offset, down.offset + 24);
    assert_eq!(expert.down.source.len, 24);

    let linear = model.layer_static_tensors.get(&0).unwrap();
    for role in [
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.conv1d.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight",
        "attn_norm.weight",
    ] {
        assert!(
            linear.contains_key(role),
            "missing canonical GDN role {role}"
        );
    }
    assert_eq!(linear["attn_norm.weight"].shape, vec![3]);

    for layer in 0..2 {
        let static_tensors = model.layer_static_tensors.get(&layer).unwrap();
        assert_eq!(
            static_tensors["ffn.shared_experts.gate_proj.weight"].shape,
            vec![5, 4]
        );
        assert_eq!(
            static_tensors["ffn.shared_experts.gate.weight"].shape,
            vec![4]
        );
    }

    for source_name in [
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.1.mlp.experts.down_proj",
    ] {
        assert!(
            !model.resident_tensors.contains_key(source_name),
            "{source_name} was incorrectly duplicated as a resident tensor"
        );
    }
}

#[test]
fn rejects_truncated_fused_expert_payload() {
    let mut fixture = fixture();
    let tensor = fixture
        .inventory
        .tensors
        .get_mut("model.language_model.layers.0.mlp.experts.gate_up_proj")
        .unwrap();
    tensor.len -= 2;

    let error = QwenMoeFrontend::build(&fixture.inventory).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("gate_up_proj len"),
        "unexpected error: {message}"
    );
}

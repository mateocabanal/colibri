use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use colic::{
    model::qwen_mtp,
    source::{SourceInventory, TensorRef},
};

static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(1);

struct Fixture {
    root: PathBuf,
    inventory: SourceInventory,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn add_tensor(
    tensors: &mut BTreeMap<String, TensorRef>,
    source: &PathBuf,
    offset: &mut u64,
    name: impl Into<String>,
    shape: &[u64],
) {
    let elements = shape.iter().copied().product::<u64>();
    let len = elements * 2;
    tensors.insert(
        name.into(),
        TensorRef {
            source: source.clone(),
            offset: *offset,
            len,
            dtype: "BF16".into(),
            shape: shape.to_vec(),
        },
    );
    *offset += len + 16;
}

fn fixture() -> Fixture {
    let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "colic-qwen-mtp-{}-{fixture_id}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("config.json"),
        r#"{
          "model_type": "qwen3_5_moe",
          "text_config": {
            "hidden_size": 4,
            "num_experts": 2,
            "moe_intermediate_size": 3,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false
          }
        }"#,
    )
    .unwrap();

    let weights = root.join("weights.bin");
    fs::write(&weights, []).unwrap();
    let mut tensors = BTreeMap::new();
    let mut offset = 128;

    for (name, shape) in [
        ("mtp.fc.weight", vec![4, 8]),
        ("mtp.norm.weight", vec![4]),
        ("mtp.pre_fc_norm_embedding.weight", vec![4]),
        ("mtp.pre_fc_norm_hidden.weight", vec![4]),
    ] {
        add_tensor(&mut tensors, &weights, &mut offset, name, &shape);
    }

    let prefix = "mtp.layers.0";
    for (role, shape) in [
        ("input_layernorm.weight", vec![4]),
        ("post_attention_layernorm.weight", vec![4]),
        ("mlp.gate.weight", vec![2, 4]),
        ("mlp.shared_expert.down_proj.weight", vec![4, 5]),
        ("mlp.shared_expert.gate_proj.weight", vec![5, 4]),
        ("mlp.shared_expert.up_proj.weight", vec![5, 4]),
        ("mlp.shared_expert_gate.weight", vec![4]),
        ("self_attn.k_norm.weight", vec![2]),
        ("self_attn.k_proj.weight", vec![2, 4]),
        ("self_attn.o_proj.weight", vec![4, 4]),
        ("self_attn.q_norm.weight", vec![2]),
        ("self_attn.q_proj.weight", vec![8, 4]),
        ("self_attn.v_proj.weight", vec![2, 4]),
    ] {
        add_tensor(
            &mut tensors,
            &weights,
            &mut offset,
            format!("{prefix}.{role}"),
            &shape,
        );
    }
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        format!("{prefix}.mlp.experts.gate_up_proj"),
        &[2, 6, 4],
    );
    add_tensor(
        &mut tensors,
        &weights,
        &mut offset,
        format!("{prefix}.mlp.experts.down_proj"),
        &[2, 4, 3],
    );

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
fn classifies_native_mtp_and_keeps_experts_pageable() {
    let fixture = fixture();
    let mtp = qwen_mtp::inspect(&fixture.inventory)
        .unwrap()
        .expect("fixture declares MTP");

    assert_eq!(mtp.hidden_layers, 1);
    assert!(!mtp.use_dedicated_embeddings);
    assert_eq!(mtp.hidden_size, 4);
    assert_eq!(mtp.experts, 2);
    assert_eq!(mtp.moe_intermediate_size, 3);
    assert_eq!(mtp.global_tensors["mtp.fc.weight"].shape, vec![4, 8]);

    let stage = &mtp.stages[0];
    assert_eq!(stage.stage, 0);
    assert_eq!(stage.expert_gate_up.shape, vec![2, 6, 4]);
    assert_eq!(stage.expert_down.shape, vec![2, 4, 3]);
    assert!(stage.static_tensors.contains_key("self_attn.q_proj.weight"));
    assert!(stage.static_tensors.contains_key("mlp.shared_expert.gate_proj.weight"));
    assert!(
        !stage.static_tensors.contains_key("mlp.experts.gate_up_proj"),
        "routed MTP expert payload must not become static residency"
    );
    assert!(
        !stage.static_tensors.contains_key("mlp.experts.down_proj"),
        "routed MTP expert payload must not become static residency"
    );
}

#[test]
fn rejects_bad_mtp_expert_geometry() {
    let mut fixture = fixture();
    fixture
        .inventory
        .tensors
        .get_mut("mtp.layers.0.mlp.experts.gate_up_proj")
        .unwrap()
        .shape = vec![2, 5, 4];

    let error = qwen_mtp::inspect(&fixture.inventory).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("mtp.layers.0.mlp.experts.gate_up_proj"));
    assert!(message.contains("expected"));
}

#[test]
fn absent_mtp_config_and_tensors_is_not_an_error() {
    let mut fixture = fixture();
    fs::write(
        fixture.root.join("config.json"),
        r#"{
          "model_type": "qwen3_5_moe",
          "text_config": {
            "hidden_size": 4,
            "num_experts": 2,
            "moe_intermediate_size": 3
          }
        }"#,
    )
    .unwrap();
    fixture
        .inventory
        .tensors
        .retain(|name, _| !name.starts_with("mtp."));

    assert!(qwen_mtp::inspect(&fixture.inventory).unwrap().is_none());
}

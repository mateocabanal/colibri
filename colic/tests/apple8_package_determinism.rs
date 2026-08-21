use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use colic::{
    pipeline::{compile, CompileRequest, NoProgress, TargetRequest},
    verify,
};

fn synthetic_v4_source(root: &Path) {
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("config.json"),
        r#"{"model_type":"deepseek_v4","hidden_size":2,"num_hidden_layers":1,"n_routed_experts":2,"moe_intermediate_size":3,"vocab_size":4,"hc_mult":2,"num_hash_layers":1,"num_experts_per_tok":1,"num_attention_heads":1,"head_dim":2,"q_lora_rank":1,"o_groups":1,"o_lora_rank":1,"index_n_heads":1,"index_head_dim":1,"compress_ratios":[0]}"#,
    )
    .unwrap();
    fs::write(root.join("tokenizer.json"), br#"{"model":{"type":"BPE"}}"#).unwrap();
    fs::write(
        root.join("tokenizer_config.json"),
        br#"{"chat_template":"fixture"}"#,
    )
    .unwrap();
    fs::write(
        root.join("generation_config.json"),
        br#"{"eos_token_id":3}"#,
    )
    .unwrap();
    fs::write(
        root.join("special_tokens_map.json"),
        br#"{"eos_token":"</s>"}"#,
    )
    .unwrap();
    fs::write(root.join("model_metadata.json"), br#"{"fixture":true}"#).unwrap();

    let mut specs = BTreeMap::<String, (&str, Vec<u64>)>::new();
    let mut add = |name: String, dtype: &'static str, shape: Vec<u64>| {
        specs.insert(name, (dtype, shape));
    };
    for expert in 0..2 {
        for (role, rows, columns) in [("w1", 3_u64, 2_u64), ("w2", 2, 3), ("w3", 3, 2)] {
            add(
                format!("layers.0.ffn.experts.{expert}.{role}.weight"),
                "I8",
                vec![rows, columns.div_ceil(2)],
            );
            add(
                format!("layers.0.ffn.experts.{expert}.{role}.scale"),
                "F8_E8M0",
                vec![rows, columns.div_ceil(32)],
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
        ("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3FN", vec![3, 2]),
        ("layers.0.ffn.shared_experts.w2.weight", "F8_E4M3FN", vec![2, 3]),
        ("layers.0.ffn.shared_experts.w3.weight", "F8_E4M3FN", vec![3, 2]),
        ("layers.0.ffn.shared_experts.w1.scale", "F8_E8M0", vec![1, 1]),
        ("layers.0.ffn.shared_experts.w2.scale", "F8_E8M0", vec![1, 1]),
        ("layers.0.ffn.shared_experts.w3.scale", "F8_E8M0", vec![1, 1]),
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
            "U8" | "I8" | "F8_E4M3FN" | "F8_E8M0" => 1,
            "BF16" => 2,
            "F32" => 4,
            "I64" => 8,
            _ => unreachable!(),
        };
        let bytes = shape.iter().product::<u64>() * size;
        header.insert(
            name,
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
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
fn apple8_package_output_is_deterministic() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "colic-apple8-determinism-{}-{nonce}",
        std::process::id()
    ));
    let source = root.join("source");
    synthetic_v4_source(&source);

    let first = root.join("first.coli");
    let second = root.join("second.coli");
    let mut request = CompileRequest::new(source);
    request.target = TargetRequest::Profile("macos-arm64-metal-apple8-v1".into());
    request.verify = true;
    request.output = Some(first.clone());
    compile(&request, &mut NoProgress).unwrap();
    request.output = Some(second.clone());
    compile(&request, &mut NoProgress).unwrap();

    assert_eq!(verify::verify_package(&first).unwrap().records, 37);
    assert_eq!(verify::verify_package(&second).unwrap().records, 37);
    for name in ["manifest.coli", "data-00000.coli"] {
        assert_eq!(
            fs::read(first.join(name)).unwrap(),
            fs::read(second.join(name)).unwrap(),
            "Apple8 package output differs for {name}"
        );
    }

    fs::remove_dir_all(root).unwrap();
}

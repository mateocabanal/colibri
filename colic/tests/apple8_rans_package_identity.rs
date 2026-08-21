use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use colic::{
    codec,
    pipeline::{self, CodecRequest, CompileRequest, NoProgress, TargetRequest},
    verify, verify_target,
};

fn temp_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "colic-apple8-rans-package-{}-{nonce}",
        std::process::id()
    ))
}

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
            "U8" | "I8" | "F8_E4M3FN" | "F8_E8M0" => 1,
            "BF16" => 2,
            "F32" => 4,
            "I64" => 8,
            _ => unreachable!(),
        };
        let bytes = shape.iter().product::<u64>() * size;
        header.insert(
            name,
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes]
            }),
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

fn compile_c_decoder(root: &Path) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let c = manifest.parent().unwrap().join("c");
    let output = root.join("apple8_decode_package");
    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&compiler)
        .current_dir(&c)
        .args([
            "-O2",
            "-std=gnu11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-function",
            "tests/apple8_decode_package.c",
            "coli_executor.c",
            "coli_format.c",
            "coli_target.c",
            "-pthread",
            "-o",
        ])
        .arg(&output)
        .status()
        .unwrap_or_else(|error| panic!("failed to execute C compiler `{compiler}`: {error}"));
    assert!(status.success(), "C Apple8 decoder failed to compile");
    output
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn i32_at(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn first_expert_record(package: &Path) -> (u64, Vec<u8>) {
    let manifest = fs::read(package.join("manifest.coli")).unwrap();
    let records = u64_at(&manifest, 32) as usize;
    let table = u64_at(&manifest, 64) as usize;
    for index in 0..records {
        let descriptor = table + index * 96;
        if u16_at(&manifest, descriptor + 8) == 2
            && i32_at(&manifest, descriptor + 28) == 0
            && i32_at(&manifest, descriptor + 32) == 0
        {
            let offset = u64_at(&manifest, descriptor + 40);
            let bytes = u64_at(&manifest, descriptor + 48) as usize;
            let shard = u32_at(&manifest, descriptor + 20);
            assert_eq!(shard, 0);
            let data = fs::read(package.join("data-00000.coli")).unwrap();
            return (
                offset,
                data[offset as usize..offset as usize + bytes].to_vec(),
            );
        }
    }
    panic!("synthetic package has no expert 0/0");
}

fn run_decoder(decoder: &Path, package: &Path, output: &Path) {
    let status = Command::new(decoder)
        .arg(package)
        .args(["0", "0"])
        .arg(output)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "C package decoder failed for {}",
        package.display()
    );
}

#[test]
fn rans_package_is_deterministic_and_decodes_to_raw_resident_bytes() {
    let root = temp_root();
    fs::create_dir_all(&root).unwrap();
    let source = root.join("source");
    synthetic_v4_source(&source);

    let raw = root.join("raw.coli");
    let rans_a = root.join("rans-a.coli");
    let rans_b = root.join("rans-b.coli");
    let auto = root.join("auto.coli");

    let mut raw_request = CompileRequest::new(source.clone());
    raw_request.target = TargetRequest::Profile("macos-arm64-metal-apple8-v1".into());
    raw_request.verify = true;
    raw_request.output = Some(raw.clone());
    pipeline::compile(&raw_request, &mut NoProgress).unwrap();

    let mut rans_request = CompileRequest::new(source.clone());
    rans_request.target = TargetRequest::Profile("macos-arm64-metal-apple8-v1".into());
    rans_request.codec = CodecRequest::Profile("rans256-g0-nibble".into());
    rans_request.verify = true;
    rans_request.output = Some(rans_a.clone());
    codec::compile::compile(&rans_request, &mut NoProgress).unwrap();
    rans_request.output = Some(rans_b.clone());
    codec::compile::compile(&rans_request, &mut NoProgress).unwrap();

    assert_eq!(verify::verify_package(&rans_a).unwrap().records, 37);
    verify_target::verify_target_layouts(&rans_a).unwrap();
    for name in ["manifest.coli", "data-00000.coli"] {
        assert_eq!(
            fs::read(rans_a.join(name)).unwrap(),
            fs::read(rans_b.join(name)).unwrap(),
            "forced rANS package differs for {name}"
        );
    }

    let manifest = fs::read(rans_a.join("manifest.coli")).unwrap();
    assert_eq!(
        u32_at(&manifest, 160),
        1,
        "forced rANS needs one artifact table"
    );
    let table_region = u64_at(&manifest, 168) as usize;
    assert_eq!(u32_at(&manifest, table_region), 1);
    assert_eq!(u16_at(&manifest, table_region + 4), 1);
    assert_eq!(i32_at(&manifest, table_region + 8), -1);

    let (_, encoded_expert) = first_expert_record(&rans_a);
    for matrix in 0..3 {
        let descriptor = 64 + matrix * 128;
        assert_eq!(u16_at(&encoded_expert, descriptor + 8), 1);
        assert_eq!(u32_at(&encoded_expert, descriptor + 40), 1);
        assert_ne!(u64_at(&encoded_expert, descriptor + 56), 0);
        assert_eq!(u64_at(&encoded_expert, descriptor + 72), 0);
    }

    let decoder = compile_c_decoder(&root);
    let raw_resident = root.join("raw-expert.bin");
    let rans_resident = root.join("rans-expert.bin");
    run_decoder(&decoder, &raw, &raw_resident);
    run_decoder(&decoder, &rans_a, &rans_resident);
    assert_eq!(
        fs::read(&raw_resident).unwrap(),
        fs::read(&rans_resident).unwrap(),
        "synchronous rANS decode did not reconstruct the exact raw Apple8 resident record"
    );

    let mut auto_request = CompileRequest::new(source);
    auto_request.target = TargetRequest::Profile("macos-arm64-metal-apple8-v1".into());
    auto_request.codec = CodecRequest::Auto;
    auto_request.verify = true;
    auto_request.output = Some(auto.clone());
    codec::compile::compile(&auto_request, &mut NoProgress).unwrap();
    let auto_manifest = fs::read(auto.join("manifest.coli")).unwrap();
    assert_eq!(
        u32_at(&auto_manifest, 160),
        0,
        "tiny matrices should stay raw under the deterministic auto threshold"
    );
    let (_, auto_expert) = first_expert_record(&auto);
    for matrix in 0..3 {
        let descriptor = 64 + matrix * 128;
        assert_eq!(u16_at(&auto_expert, descriptor + 8), 0);
        assert_eq!(u32_at(&auto_expert, descriptor + 40), 0);
    }

    fs::remove_dir_all(root).unwrap();
}

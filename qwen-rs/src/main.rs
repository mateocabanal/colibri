//! qwen-rs walking skeleton: greedy decode on the tiny Qwen MoE fixture,
//! gated against `ref.json` `greedy_new_ids` (plan RW-041 exit gate).
//!
//! Usage: qwen-rs <fixture-dir>   (dir with config.json + model.safetensors)

use std::path::Path;

use qwen_rs::{load_cfg, Model, StFile};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: qwen-rs <fixture-dir>");
        std::process::exit(2);
    }
    let dir = Path::new(&args[1]);
    let cfg = load_cfg(&dir.join("config.json")).unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });
    let st = StFile::open(&dir.join("model.safetensors")).unwrap_or_else(|e| {
        eprintln!("safetensors error: {e}");
        std::process::exit(1);
    });
    let mut model = Model::load(&st, &cfg).unwrap_or_else(|e| {
        eprintln!("model error: {e}");
        std::process::exit(1);
    });

    // ref.json gate: short case greedy_new_ids
    let ref_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("ref.json")).expect("ref.json"),
    )
    .expect("ref.json parse");
    let cases = ref_json["cases"].as_object().expect("cases");
    let short = cases["short"].as_object().expect("short case");
    let prompt: Vec<u32> = short["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let expected: Vec<u32> = short["greedy_new_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let max_new = short["max_new_tokens"].as_u64().unwrap() as usize;

    // prefill: forward each prompt token, keep state
    for (i, &t) in prompt.iter().enumerate() {
        model.forward_token(t as usize, i);
    }
    // decode: greedy argmax
    let mut generated: Vec<u32> = Vec::new();
    let mut last = *prompt.last().unwrap();
    for pos in prompt.len()..prompt.len() + max_new {
        let logits = model.forward_token(last as usize, pos);
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        generated.push(next);
        last = next;
    }

    println!("prompt      : {:?}", prompt);
    println!("generated   : {:?}", generated);
    println!("expected    : {:?}", expected);
    if generated == expected {
        println!("GATE: PASS token-identity (short)");
    } else {
        println!("GATE: FAIL token-identity (short)");
        std::process::exit(1);
    }

    // mixed case: longer prompt -> exercises full_attention layer with KV cache
    let mixed = cases["mixed"].as_object().expect("mixed case");
    let prompt_m: Vec<u32> = mixed["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let expected_m: Vec<u32> = mixed["greedy_new_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let max_new_m = mixed["max_new_tokens"].as_u64().unwrap() as usize;

    let mut model = Model::load(&st, &cfg).unwrap();
    for (i, &t) in prompt_m.iter().enumerate() {
        model.forward_token(t as usize, i);
    }
    let mut generated_m: Vec<u32> = Vec::new();
    let mut last = *prompt_m.last().unwrap();
    for pos in prompt_m.len()..prompt_m.len() + max_new_m {
        let logits = model.forward_token(last as usize, pos);
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        generated_m.push(next);
        last = next;
    }
    println!("generated(m): {:?}", generated_m);
    println!("expected (m): {:?}", expected_m);
    if generated_m == expected_m {
        println!("GATE: PASS token-identity (mixed)");
    } else {
        println!("GATE: FAIL token-identity (mixed)");
        std::process::exit(1);
    }
}

//! qwen4-rs: greedy decode on the tiny Qwen4 fixture, gated against
//! `ref.json` greedy_new_ids (short + mixed cases).

use std::path::Path;

use qwen4_rs::{load_cfg, Model, StFile};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: qwen4-rs <fixture-dir>");
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

    let ref_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("ref.json")).expect("ref.json"))
            .expect("ref.json parse");
    let cases = ref_json["cases"].as_object().expect("cases");

    let mut all_pass = true;
    for case_name in ["short", "mixed"] {
        let case = cases[case_name].as_object().unwrap();
        let prompt: Vec<u32> = case["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expected: Vec<u32> = case["greedy_new_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let max_new = case["max_new_tokens"].as_u64().unwrap() as usize;

        let mut model = Model::load(&st, &cfg).unwrap();
        for (i, &t) in prompt.iter().enumerate() {
            model.forward_token(t as usize, i);
        }
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
        let pass = generated == expected;
        all_pass &= pass;
        println!(
            "case {case_name}: generated {:?} expected {:?} {}",
            generated,
            expected,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    if all_pass {
        println!("GATE: PASS token-identity");
    } else {
        println!("GATE: FAIL token-identity");
        std::process::exit(1);
    }
}

//! qwen4-rs: greedy decode on the tiny Qwen4 fixture, gated against
//! `ref.json` greedy_new_ids (short + mixed cases). Also loads `.coli`
//! packages (Apple8/MXFP4) via colibri-format.

use std::path::Path;

use qwen4_rs::{load_cfg, Model, StFile};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: qwen4-rs <fixture-dir | .coli-package>");
        std::process::exit(2);
    }
    let dir = Path::new(&args[1]);
    let cfg = load_cfg(&dir.join("config.json")).unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(1);
    });
    // .coli package mode: no model.safetensors -> load via colibri-format
    let is_coli = !dir.join("model.safetensors").exists();
    let model = if is_coli {
        let src = qwen4_rs::colisource::ColiSource::open(dir).unwrap_or_else(|e| {
            eprintln!("coli error: {e}");
            std::process::exit(1);
        });
        Model::load_coli(&src, &cfg).unwrap_or_else(|e| {
            eprintln!("model error: {e}");
            std::process::exit(1);
        })
    } else {
        let st = StFile::open(&dir.join("model.safetensors")).unwrap_or_else(|e| {
            eprintln!("safetensors error: {e}");
            std::process::exit(1);
        });
        Model::load(&st, &cfg).unwrap_or_else(|e| {
            eprintln!("model error: {e}");
            std::process::exit(1);
        })
    };
    let mut model = model;

    let ref_path = dir.join("ref.json");
    if !ref_path.exists() {
        // Real package: no oracle — plain greedy decode.
        let prompt: Vec<u32> = std::env::var("QWEN_PROMPT")
            .unwrap_or_else(|_| "1 2 3 4 5".into())
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let max_new: usize = std::env::var("QWEN_MAX_NEW")
            .unwrap_or_else(|_| "8".into())
            .parse()
            .unwrap();
        for (i, &t) in prompt.iter().enumerate() {
            model.forward_token(t as usize, i);
        }
        let mut out: Vec<u32> = Vec::new();
        let mut last = *prompt.last().unwrap();
        for pos in prompt.len()..prompt.len() + max_new {
            let logits = model.forward_token(last as usize, pos);
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            out.push(next);
            last = next;
        }
        println!("generated: {out:?}");
        std::process::exit(0);
    }
    let ref_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ref_path).expect("ref.json"))
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

        // Fresh model per case (state reset), same source as the main load.
        let model = if is_coli {
            let src = qwen4_rs::colisource::ColiSource::open(dir).unwrap();
            Model::load_coli(&src, &cfg).unwrap()
        } else {
            let st = StFile::open(&dir.join("model.safetensors")).unwrap();
            Model::load(&st, &cfg).unwrap()
        };
        let mut model = model;
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

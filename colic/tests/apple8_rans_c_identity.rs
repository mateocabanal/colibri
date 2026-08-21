use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use colic::codec::rans256::{self, Table};

fn temp_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "colic-rans-rust-c-{}-{nonce}",
        std::process::id()
    ))
}

fn compile_c_oracle(root: &Path) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = manifest
        .parent()
        .unwrap()
        .join("c/tests/rans256_record_oracle.c");
    let output = root.join("rans256_record_oracle");
    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&compiler)
        .args([
            "-O2",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-function",
        ])
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap_or_else(|error| panic!("failed to execute C compiler `{compiler}`: {error}"));
    assert!(status.success(), "C rANS oracle failed to compile");
    output
}

fn fixture(case: usize) -> Vec<u8> {
    match case {
        0 => (0..32 * 1024)
            .map(|index| match index % 23 {
                0..=15 => 0x11,
                16..=18 => 0x00,
                19 => 0x21,
                20 => 0x10,
                21 => 0x12,
                _ => 0x01,
            })
            .collect(),
        1 => (0_u8..=255).cycle().take(32 * 1024).collect(),
        _ => (0..32 * 1024)
            .map(|index| ((index * 73 + index / 11 + 19) & 0xff) as u8)
            .collect(),
    }
}

#[test]
fn rust_and_c_rans256_records_are_byte_identical() {
    let root = temp_root();
    fs::create_dir_all(&root).unwrap();
    let oracle = compile_c_oracle(&root);

    for case in 0..3 {
        let input = fixture(case);
        let histogram = rans256::histogram_bytes([input.as_slice()]).unwrap();
        let table = Table::from_histogram(histogram).unwrap();
        let rust = rans256::encode_bytes(&input, &table).unwrap();

        let table_path = root.join(format!("table-{case}.bin"));
        let input_path = root.join(format!("input-{case}.bin"));
        let output_path = root.join(format!("c-{case}.bin"));
        fs::write(&table_path, table.encode_blob().unwrap()).unwrap();
        fs::write(&input_path, &input).unwrap();
        let status = Command::new(&oracle)
            .arg(&table_path)
            .arg(&input_path)
            .arg(&output_path)
            .status()
            .unwrap();
        assert!(status.success(), "C rANS oracle failed for fixture {case}");
        assert_eq!(
            rust,
            fs::read(&output_path).unwrap(),
            "Rust/C rANS record mismatch for fixture {case}"
        );
        assert_eq!(
            rans256::decode_bytes(&rust, &table, input.len()).unwrap(),
            input,
            "Rust rANS round trip failed for fixture {case}"
        );
    }

    fs::remove_dir_all(root).unwrap();
}

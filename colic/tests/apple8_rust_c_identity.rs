use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use colic::{quant::mxfp4::PackedMatrix, target::apple8_repack_reference_input};

fn temp_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "colic-apple8-rust-c-{}-{nonce}",
        std::process::id()
    ))
}

fn compile_c_oracle(root: &Path) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = manifest
        .parent()
        .unwrap()
        .join("c/tests/apple8_reference_packer.c");
    let output = root.join("apple8_reference_packer");
    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&compiler)
        .args(["-O2", "-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap_or_else(|error| panic!("failed to execute C compiler `{compiler}`: {error}"));
    assert!(status.success(), "C Apple8 oracle failed to compile");
    output
}

fn fixture(rows: u32, columns: u32) -> PackedMatrix {
    let row_bytes = (columns as usize).div_ceil(2);
    let groups = (columns as usize).div_ceil(32);
    let mut weights = vec![0_u8; rows as usize * row_bytes];
    let mut scales = vec![0_u8; rows as usize * groups];

    for row in 0..rows as usize {
        for byte in 0..row_bytes {
            weights[row * row_bytes + byte] = ((row * 29 + byte * 17 + 3) & 0xff) as u8;
        }
        if !columns.is_multiple_of(2) {
            weights[row * row_bytes + row_bytes - 1] &= 0x0f;
        }
        for group in 0..groups {
            scales[row * groups + group] = 1 + ((row * 11 + group * 23 + 97) % 254) as u8;
        }
    }

    PackedMatrix {
        rows,
        columns,
        weights,
        scales,
    }
}

#[test]
fn rust_and_c_apple8_packers_are_byte_identical() {
    let root = temp_root();
    fs::create_dir_all(&root).unwrap();
    let oracle = compile_c_oracle(&root);

    let shapes = [
        (1_u32, 1_u32),
        (1, 31),
        (1, 32),
        (1, 33),
        (7, 32),
        (8, 32),
        (9, 32),
        (8, 31),
        (8, 33),
        (9, 33),
        // Real V4 gate/up and down-like matrix geometries.
        (2048, 4096),
        (4096, 2048),
    ];

    for (case, (rows, columns)) in shapes.into_iter().enumerate() {
        let matrix = fixture(rows, columns);
        let rust = apple8_repack_reference_input(&matrix).unwrap();
        let weights = root.join(format!("weights-{case}.bin"));
        let scales = root.join(format!("scales-{case}.bin"));
        let output = root.join(format!("c-{case}.bin"));
        fs::write(&weights, &matrix.weights).unwrap();
        fs::write(&scales, &matrix.scales).unwrap();

        let status = Command::new(&oracle)
            .arg(rows.to_string())
            .arg(columns.to_string())
            .arg(&weights)
            .arg(&scales)
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success(), "C oracle failed for {rows}x{columns}");
        let c = fs::read(&output).unwrap();
        assert_eq!(rust, c, "Rust/C Apple8 mismatch for {rows}x{columns}");
    }

    fs::remove_dir_all(root).unwrap();
}

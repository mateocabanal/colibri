// Build the proven C Metal backend (backend_metal.mm) into the crate.
// fmt 7 (MXFP4) is byte-compatible with Apple8 tiles: nibbles + raw E8M0
// scales. GPU does the dequant; host never decodes.
//
// ALSO builds apple8_metalio_direct.mm — the fused direct Apple8 execution
// seam (slot-resident expert GEMV/SwiGLU, one-command-buffer moe_topk, and
// the coalesced Metal GDN kernels). That file #includes
// "generated/coli_target_registry.h" (via apple8_contract.h) and metalio.h,
// all vendored under qwen4-rs/metal/ so the include root is the metal dir.
// MetalIO + direct paths compile to the same sources as c/ (byte-identical
// copies, verified by md5 in the port session).
fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    let metal_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("metal");
    let src = metal_dir.join("backend_metal.mm");
    if !src.exists() {
        // Not built on this checkout (Mac-only path) — emit nothing.
        return;
    }
    let obj = std::path::Path::new(&out).join("backend_metal.o");
    let status = std::process::Command::new("clang++")
        .args([
            "-x",
            "objective-c++",
            "-std=gnu++17",
            "-fobjc-arc",
            "-O3",
            "-fobjc-exceptions",
            "-I",
            metal_dir.to_str().unwrap(),
            "-c",
            src.to_str().unwrap(),
            "-o",
            obj.to_str().unwrap(),
        ])
        .status()
        .expect("clang++ must be available on macOS");
    assert!(status.success(), "backend_metal.mm failed to compile");
    // archive the object so rustc can link -lstatic=backend_metal
    let lib = std::path::Path::new(&out).join("libbackend_metal.a");
    let ar = std::process::Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .expect("ar must be available");
    assert!(ar.success(), "ar failed");

    // metalio.mm — async NVMe -> MTLBuffer expert streaming (also a colibri C file)
    let mio_src = metal_dir.join("metalio.mm");
    if mio_src.exists() {
        let mio_obj = std::path::Path::new(&out).join("metalio.o");
        let status = std::process::Command::new("clang++")
            .args([
                "-x",
                "objective-c++",
                "-std=gnu++17",
                "-fobjc-arc",
                "-O3",
                "-fobjc-exceptions",
                "-I",
                metal_dir.to_str().unwrap(),
                "-c",
                mio_src.to_str().unwrap(),
                "-o",
                mio_obj.to_str().unwrap(),
            ])
            .status()
            .expect("clang++ must be available on macOS");
        assert!(status.success(), "metalio.mm failed to compile");
        let ar = std::process::Command::new("ar")
            .args(["rcs", lib.to_str().unwrap(), mio_obj.to_str().unwrap()])
            .status()
            .expect("ar failed for metalio.o");
        assert!(ar.success(), "ar failed for metalio.o");
        println!("cargo:rerun-if-changed=metal/metalio.mm");
    }

    // apple8_metalio_direct.mm — fused direct Apple8 execution seam:
    // slot-resident expert matmul/swiglu, one-command-buffer moe_topk
    // (begin/finish split phase), and the coalesced Metal GDN kernels
    // (qwen_gdn_input_bf16 / qwen_gdn_conv_recur_norm / qwen_gdn_output_bf16).
    // Needs metalio.h + apple8_contract.h (-> generated/coli_target_registry.h),
    // all vendored under metal/. Must link AFTER metalio.o (it calls
    // metalio_active/metalio_slot_* — static-archive order matters).
    let direct_src = metal_dir.join("apple8_metalio_direct.mm");
    if direct_src.exists() {
        let direct_obj = std::path::Path::new(&out).join("apple8_metalio_direct.o");
        let status = std::process::Command::new("clang++")
            .args([
                "-x",
                "objective-c++",
                "-std=gnu++17",
                "-fobjc-arc",
                "-O3",
                "-fobjc-exceptions",
                "-I",
                metal_dir.to_str().unwrap(),
                "-c",
                direct_src.to_str().unwrap(),
                "-o",
                direct_obj.to_str().unwrap(),
            ])
            .status()
            .expect("clang++ must be available on macOS");
        assert!(
            status.success(),
            "apple8_metalio_direct.mm failed to compile"
        );
        let ar = std::process::Command::new("ar")
            .args(["rcs", lib.to_str().unwrap(), direct_obj.to_str().unwrap()])
            .status()
            .expect("ar failed for apple8_metalio_direct.o");
        assert!(ar.success(), "ar failed for apple8_metalio_direct.o");
        println!("cargo:rerun-if-changed=metal/apple8_metalio_direct.mm");
        println!("cargo:rerun-if-changed=metal/apple8_contract.h");
        println!("cargo:rerun-if-changed=metal/generated/coli_target_registry.h");
    }

    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=backend_metal");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rerun-if-changed=metal/backend_metal.mm");
    println!("cargo:rerun-if-changed=metal/backend_metal.h");
}

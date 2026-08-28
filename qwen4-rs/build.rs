// Build the proven C Metal backend (backend_metal.mm) into the crate.
// fmt 7 (MXFP4) is byte-compatible with Apple8 tiles: nibbles + raw E8M0
// scales. GPU does the dequant; host never decodes.
fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("metal")
        .join("backend_metal.mm");
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
            src.parent().unwrap().to_str().unwrap(),
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
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=backend_metal");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rerun-if-changed=metal/backend_metal.mm");
}

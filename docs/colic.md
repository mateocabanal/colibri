# `colic` compiler smoke paths

`colic` is the offline Rust compiler for target-compiled COLI packages. Source
safetensors are compiler input only; production inference consumes the emitted
package.

## DeepSeek V4 Flash — Apple Silicon exact baseline

On an Apple-Silicon development machine:

```bash
cargo run --release --locked --manifest-path colic/Cargo.toml -- inspect-source \
  /path/to/DeepSeek-V4-Flash

cargo run --release --locked --manifest-path colic/Cargo.toml -- compile \
  /path/to/DeepSeek-V4-Flash \
  --target native \
  --quant exact \
  --codec none \
  --verify \
  -o /path/to/DeepSeek-V4-Flash.apple.coli
```

For reproducible target-ABI testing independent of host detection, replace
`--target native` with:

```text
--target macos-arm64-metal-apple8-v1
```

The exact v1 Apple target accepts routed experts that are already native
MXFP4/E2M1 with UE8M0 row32 scales. `--quant exact` does not silently convert or
relabel an FP8 routed expert as MXFP4; unsupported source semantics fail closed.

A successful compile publishes only after temporary-package verification. The
runtime package is self-contained: source safetensors are not needed by the
strict C target loader or expert execution path.

## Tiny cross-language gate

`.github/workflows/colic-e2e.yml` constructs a deterministic tiny V4 source,
compiles it twice, proves byte-identical output, deletes the source directory,
opens the package through the strict C target reader, executes a compiler-emitted
MXFP4 routed expert through the engine's CPU MXFP4 kernel, and checks the result
against an independent SwiGLU oracle. It also proves incompatible target
capabilities and stored-byte corruption fail closed.

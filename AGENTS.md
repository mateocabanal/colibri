# Agent instructions

All coding agents and automated contributors working in this repository must follow [`PROJECT_POLICY.md`](PROJECT_POLICY.md).

In particular:

- design reusable performance/runtime mechanisms for multiple MoE engines, not only the model that motivated the work;
- keep model semantics in engines and generic residency, I/O, caching, resource planning, backend policy, quant/kernel dispatch, and telemetry in the shared runtime;
- make safe high-performance behavior automatic by default; performance flags are overrides/diagnostics, not required setup;
- keep Apple Silicon/macOS and x86_64 Linux + CUDA first-class;
- avoid model-name dispatch and duplicate engine-local global mechanisms;
- prefer focused deterministic validation and targeted benchmarks with provenance.

If an exception is genuinely model-specific, document why the mechanism cannot live behind a shared runtime contract.

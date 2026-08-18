# Colibri project policy

These rules apply project-wide to model engines, runtime infrastructure, backends, compiler work, tests, benchmarks, and future contributions.

## Product scope

- Colibri is an MoE-first inference engine. New runtime mechanisms must be designed for multiple MoE model families unless the behavior is genuinely model-semantic.
- Apple Silicon on macOS and x86_64 Linux + CUDA are first-class targets. Platform-specific acceleration belongs behind shared runtime contracts rather than inside model math.

## Engine/runtime boundary

Model engines own model semantics: configuration and geometry, layer math, routing semantics, attention/recurrent state semantics, and model-specific validation.

Shared runtime owns reusable mechanism and policy: compiled-record I/O, expert activation telemetry, expert residency, global RAM/UMA/VRAM planning, prefix caching, sequence-state lifecycle, backend policy/lifecycle, quant/layout/kernel descriptors, scheduling, prefetch, and profiling.

A model engine may be the first consumer of a new optimization, but reusable infrastructure must not be implemented as an engine-local subsystem merely because that engine motivated the work. Prefer a narrow shared contract plus an engine adapter.

Do not introduce model-name dispatch into generic runtime policy. Dispatch by explicit descriptors, capabilities, geometry, quant/layout/kernel ABI, and engine-provided semantic hooks.

## Performance is the default

The ordinary engine invocation must select the best safe supported path automatically. Users should not need a list of environment variables or command-line flags to obtain normal high-performance execution.

Performance features should therefore be **default-on or auto-selected** when their correctness preconditions and platform capabilities are satisfied. Examples include GPU backends, fused quantized kernels, expert union/batching, async I/O, prefetch, prompt caching, adaptive residency, direct/uncached I/O, and sensible loader concurrency.

Environment variables and command-line performance controls should primarily be diagnostics, benchmarking controls, explicit caps, or ways to force/disable behavior. Avoid adding a new enable flag for an optimization that should simply be normal execution.

Do not replace automatic policy with a `performance=max` switch that hides required tuning flags. The runtime should make the decision itself and report what it chose.

Prefer global/shared knobs (`COLI_*`) over engine-specific knobs (`V4_*`, `QWEN_*`) for generic behavior. Engine-specific knobs are appropriate only for genuinely model-specific semantics or temporary migration/debugging controls, and should have a removal path.

## Resource and residency policy

- RAM, Apple UMA, pinned host memory, and device VRAM are globally planned resources, not independent per-engine budgets.
- On Apple Silicon, shared UMA allocations count once. On discrete CUDA systems, host/pinned/device tiers are distinct resources.
- Optional residency should compete on measured expected benefit per resident byte, preferably exposed time avoided per byte; do not permanently partition memory into arbitrary dense-vs-expert pools when the runtime can compare them.
- Logical expert activation frequency must be captured before batching/union destroys multiplicity. Physical I/O counts are not a substitute for routing hotness.
- Persistent expert policy should be frequency/benefit-aware rather than blindly LRU when reuse distance exceeds cache capacity.
- Keep transient execution concurrency separate from persistent locality/residency.

## I/O and backend policy

Compiled-record requests use one shared request/lifecycle contract. Platform backends such as `pread`, `io_uring`, MetalIO, CUDA staging, or future mechanisms sit below that contract.

Backends should fail closed for incompatible descriptors and fall back safely when an optional acceleration path is unavailable. Model code should not start duplicate physical reads, manage raw residency generations, or own backend lifecycle independently.

## Cache and state policy

There is one canonical persistent prefix-cache architecture and one shared sequence-state lifecycle. Engines provide serialization/restoration adapters for their state geometry; they do not create competing persistent indexes or cache formats.

## Measurement and exceptions

Performance policy changes need targeted measurements with reproducible provenance: commit, exact command/config, hardware, storage, warm/cold policy, and the metric being optimized. Prefer small deterministic oracles and focused benchmarks over repeated production-length runs during development.

An engine-local optimization is acceptable when it depends on genuinely model-specific semantics or measured model-specific fusion. Document why it cannot live behind a shared contract.

When migrating old engine-local mechanisms, use a strangler approach: introduce the shared contract, migrate consumers, then delete duplicate policy/mechanism rather than maintaining two long-lived implementations.

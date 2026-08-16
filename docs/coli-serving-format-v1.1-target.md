# Colibri Serving Format (`.coli`) v1.1 — target-compiled deployment contract

Status: **proposed active deployment contract for #22**.

This document supersedes the **portability and deployment-policy portions** of `docs/coli-serving-format-v1.md` while deliberately reusing its v1.0 binary framing where possible.

The v1.0 `portable-v1` artifact and tiny fixture remain useful as legacy parser/golden-test material. They are **not** the model format that `colic` should emit for production inference.

The central rule is:

> **Safetensors/Hugging Face checkpoints are compiler inputs. A `.coli` artifact is a target-compiled deployment image consumed directly by the Colibri runtime.**

A production `.coli` artifact is therefore closer to an object/executable image than to an interchange checkpoint.

---

## 1. Migration from v1.0

v1.1 keeps the following v1.0 framing so the already-implemented parser work remains useful:

- 256-byte `manifest.coli` header;
- shard descriptor framing;
- data-shard header framing;
- top-level record descriptor framing;
- string-table framing;
- codec-table framing;
- typed record envelopes where still applicable;
- CRC32C and checked-offset rules;
- little-endian fixed-width serialization;
- source fingerprint field at manifest offset 112.

v1.1 changes the execution contract:

1. `portable-v1` is legacy/parser-only and MUST NOT be emitted by the production compiler.
2. Every executable v1.1 artifact declares a target compatibility profile.
3. The manifest metadata span, opaque in v1.0, becomes a required fixed target descriptor in v1.1.
4. A required manifest feature bit marks the artifact as target-compiled.
5. Physical record payloads MAY be transposed, reordered, quantized, compressed, fused, padded, or otherwise lowered for the target.
6. Runtime inference MUST NOT require a canonical/source-tensor reconstruction step.
7. Incompatible artifacts are rejected; they are not repacked into a generic fallback representation at runtime.

Existing v1.0 tooling MAY continue to parse legacy packages. Production model execution should require the v1.1 target contract once #23/#25 land.

---

## 2. Compiler/runtime boundary

The intended toolchain is:

```text
HF / safetensors / future source format
                 |
                 v
              `colic`
      frontend -> semantic IR
                 |
          optimization passes
                 |
            target lowering
          /                \
 Apple Silicon/Metal     x86_64/Linux/CUDA
          \                /
                 v
        target-compiled `.coli`
                 |
                 v
       target-validating loader
                 |
       final resident/kernel layout
                 |
                 v
          CPU / Metal / CUDA
```

Anything that can be decided safely offline should normally be a compiler responsibility, including:

- source tensor relationship discovery;
- model-specific name interpretation;
- transpose/reordering;
- scale packing;
- expert grouping;
- shard/record ordering;
- target alignment;
- quantization selection;
- lossless storage coding;
- immutable backend metadata;
- kernel-ready physical layout.

The runtime should route experts, move compiled bytes, schedule work, and execute kernels.

---

## 3. Semantic ABI vs physical ABI

### 3.1 Semantic/model ABI

The semantic layer describes what the model means:

```text
model architecture
layer/expert geometry
operator and tensor roles
MoE routing semantics
logical shapes
activation kinds
quantization semantics
tokenizer/config identity
```

### 3.2 Physical/target ABI

The physical layer describes how this artifact executes:

```text
OS / host architecture
backend
CPU feature requirements
GPU family or compute capability
execution-layout ABI
kernel ABI
quantization profile
storage-codec profile
optimization profile
record/read alignment
resident-slot alignment
optional tuning identity
```

Container-major versioning and target/profile ABI versioning are independent. A new Apple or CUDA physical layout should not require a new CSF major version when framing remains compatible.

---

## 4. v1.1 manifest requirements

The v1.0 256-byte manifest header remains unchanged in size and field offsets.

For v1.1 production artifacts:

```text
version_major = 1
version_minor = 1
```

The previously reserved required-feature range is used as follows:

```text
manifest flags bit 16 = CSF_MANIFEST_F_REQUIRED_TARGET_COMPILED
```

Readers that do not understand this required feature MUST reject the package. This is intentional: a v1.0 portable reader must not accidentally execute target-specific bytes as canonical tensors.

The v1.0 fields:

```text
metadata_offset
metadata_bytes
```

are required in v1.1 and point to exactly one `ColiTargetDescV1` structure.

The target descriptor is covered by the existing whole-manifest CRC32C because it resides inside `manifest.coli`.

`profile_name_string_id` remains the human-readable compiled profile name, for example:

```text
macos-arm64-metal-apple8-v1
linux-x86_64-cuda-sm89-v1
linux-x86_64-avx2-v1
```

`compiler_string_id` remains the human-readable compiler identity/version.

---

## 5. `ColiTargetDescV1`

`ColiTargetDescV1` is exactly **256 bytes**, 16-byte aligned in `manifest.coli`.

Magic, exactly 8 bytes:

```text
43 4f 4c 49 54 47 54 00    # "COLITGT\0"
```

### Byte layout

| Offset | Bytes | Type | Field | Rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | `COLITGT\0` |
| 8 | 2 | u16 | `version_major` | `1` |
| 10 | 2 | u16 | `version_minor` | `0` for this descriptor ABI |
| 12 | 4 | u32 | `descriptor_bytes` | `256` |
| 16 | 4 | u32 | `flags` | target flags below |
| 20 | 2 | u16 | `target_os` | registry below |
| 22 | 2 | u16 | `target_arch` | registry below |
| 24 | 2 | u16 | `backend` | registry below |
| 26 | 2 | u16 | `accelerator_family` | registry/profile-defined |
| 28 | 8 | u64 | `cpu_feature_bits` | required host CPU features |
| 36 | 2 | u16 | `accelerator_cap_major` | e.g. CUDA SM major |
| 38 | 2 | u16 | `accelerator_cap_minor` | e.g. CUDA SM minor |
| 40 | 4 | u32 | `execution_layout_abi` | required physical-layout ABI |
| 44 | 4 | u32 | `kernel_abi` | required kernel interpretation ABI |
| 48 | 4 | u32 | `quant_profile_id` | compiler/runtime shared registry |
| 52 | 4 | u32 | `storage_profile_id` | compiler/runtime shared registry |
| 56 | 4 | u32 | `optimization_profile_id` | compiler/runtime shared registry |
| 60 | 4 | u32 | `resident_alignment` | power of two |
| 64 | 4 | u32 | `preferred_read_alignment` | power of two |
| 68 | 4 | u32 | `compat_class_string_id` | string-table ID |
| 72 | 4 | u32 | `execution_profile_string_id` | string-table ID |
| 76 | 4 | u32 | `quant_profile_string_id` | string-table ID |
| 80 | 4 | u32 | `storage_profile_string_id` | string-table ID |
| 84 | 4 | u32 | `optimization_profile_string_id` | string-table ID |
| 88 | 4 | u32 | `kernel_name_string_id` | string-table ID or `UINT32_MAX` |
| 92 | 4 | u32 | `reserved0` | zero |
| 96 | 32 | bytes | `artifact_fingerprint` | SHA-256 identity below |
| 128 | 32 | bytes | `tuning_fingerprint` | zero unless tuned flag is set |
| 160 | 96 | bytes | `reserved` | zero |

All reserved bytes MUST be zero. Unknown required target flags are a hard error.

### Target flags

```text
bit 0      CSF_TARGET_F_EXACT_LOGICAL_MODEL
bit 1      CSF_TARGET_F_LOSSY_QUANTIZATION
bit 2      CSF_TARGET_F_AUTOTUNED
bit 3      CSF_TARGET_F_DIRECT_RESIDENT_LAYOUT
bits 4-15  optional target flags
bits 16-31 required target flags
```

`EXACT_LOGICAL_MODEL` and `LOSSY_QUANTIZATION` are mutually exclusive for v1.1 canonical output.

`AUTOTUNED` requires a nonzero `tuning_fingerprint`.

`DIRECT_RESIDENT_LAYOUT` means the compiler/codec output is already in the declared resident execution layout; normal inference must not insert a record-sized canonical repack.

---

## 6. Initial target registries

These registry values are container/profile IDs, not operating-system SDK constants.

### `target_os` (`u16`)

```text
0x0000  INVALID
0x0001  MACOS
0x0002  LINUX
0xffff  INVALID_SENTINEL
```

Additional OS targets require explicit registry allocation.

### `target_arch` (`u16`)

```text
0x0000  INVALID
0x0001  ARM64
0x0002  X86_64
0xffff  INVALID_SENTINEL
```

### `backend` (`u16`)

```text
0x0000  INVALID
0x0001  CPU
0x0002  METAL
0x0003  CUDA
0x0004  HYBRID
0xffff  INVALID_SENTINEL
```

`HYBRID` means the target profile itself defines a stable multi-backend execution contract. It must not mean “choose anything available at runtime.”

### `accelerator_family` (`u16`)

The meaning is backend-scoped.

Initial ranges:

```text
0x0000         NONE
0x0100-0x01ff  Apple GPU-family compatibility classes
0x0200-0x02ff  NVIDIA/CUDA compatibility classes when a family ID is useful
0x8000-0xfffe  future/private experimental allocation
0xffff         INVALID
```

For CUDA, the authoritative minimum capability is `accelerator_cap_major/minor` unless the profile says otherwise.

For Metal, #26 owns the mapping from Apple GPU-family compatibility class to the actual runtime capability checks.

### CPU feature bits

The exact shared registry is owned by #26. Initial candidates include:

```text
ARM64: NEON/AdvSIMD, FP16, dot-product as actually required
x86_64: AVX2, FMA, AVX-512 subsets only when a profile truly requires them
```

A compiler MUST NOT set requirements merely because the compiling host has a feature; only features required to execute the emitted bytes/kernel ABI belong here.

---

## 7. Artifact identity

The existing v1.0 `source_fingerprint` remains the provenance identity of the source model.

v1.1 adds `artifact_fingerprint`, conceptually:

```text
artifact_fingerprint = SHA256(
    "COLI-ARTIFACT-V1\0" ||
    source_fingerprint ||
    compiler_identity ||
    target_profile_identity ||
    execution_layout_abi ||
    kernel_abi ||
    quant_profile_identity ||
    storage_profile_identity ||
    optimization_profile_identity ||
    tuning_fingerprint
)
```

The exact canonical serialization used by `colic` MUST be documented before #24 is closed. It must not depend on locale, timestamps, filesystem traversal order, or pointer-sized host values.

For deterministic non-autotuned profiles, identical source + compiler + target/options MUST produce byte-identical output.

A tuned artifact includes the tuning fingerprint so a cache never confuses benchmark-selected bytes with the generic target profile.

---

## 8. Compatibility rule

A v1.1 artifact is executable only when the runtime satisfies every required target/profile constraint.

At minimum compare:

```text
target_os
target_arch
backend
required cpu_feature_bits
accelerator family/capability
execution_layout_abi
kernel_abi
supported quant_profile_id
supported storage_profile_id
required target flags
```

The runtime MUST fail before model execution when incompatible.

The diagnostic should identify both sides, for example:

```text
incompatible COLI target
artifact: macos-arm64 / Metal / Apple8-class / layout ABI 3 / kernel ABI 7
runtime:  linux-x86_64 / CUDA / sm_89
recompile with: colic compile ... --target native
```

The runtime MUST NOT:

- treat an unknown target layout as `CANONICAL`;
- search for nearby safetensors and silently run a different path;
- perform a generic source reconstruction/repack to make the artifact executable.

Recompilation is the compatibility fallback.

---

## 9. Target profiles are compatibility classes

`colic --target native` should produce an artifact for the narrowest **stable reusable compatibility class**, not normally for one physical serial-number-level machine.

Examples:

```text
macos-arm64-metal-apple8-v1
linux-x86_64-cuda-sm89-v1
linux-x86_64-avx2-v1
```

A profile may cover several physical machines if they satisfy the same execution ABI.

Explicit autotuning may narrow the artifact identity when measured choices depend on the specific machine/storage configuration.

---

## 10. Physical record semantics

The v1.0 top-level record descriptor framing remains valid, but v1.1 changes the interpretation of `layout` and payload bytes:

> The physical payload is whatever the declared target execution-layout/kernel ABI consumes.

A compiled record MAY contain:

- transposed matrices;
- blocked/tiled matrices;
- target-specific row/interleave order;
- fused gate/up/down storage;
- target-packed scales;
- exact or lossy quantized bytes;
- independently compressed bytes;
- deterministic alignment padding internal to the typed record contract;
- immutable kernel metadata.

The payload does **not** need to map one-to-one onto source safetensors tensors.

Tooling for exact profiles MAY inverse-transform target bytes to verify source equivalence. Runtime inference must not depend on that inverse transform.

---

## 11. Routed-expert contract

A routed expert remains the key execution primitive.

The target hot path should converge toward:

```text
(layer, expert)
  -> O(1) compiled descriptor
  -> one aligned record read where practical
  -> optional independent codec decode
  -> final resident target layout
  -> publish generation
  -> target kernel
```

Not:

```text
(layer, expert)
  -> six source tensor-name lookups
  -> assemble logical expert
  -> canonical buffer
  -> transpose/repack
  -> target buffer
```

The expert record still carries enough semantic information for validation/debugging:

```text
layer / expert
matrix roles: gate / up / down
logical rows/columns
quant/math semantics
scale semantics
```

Its physical subregions are governed by the target execution-layout/kernel ABI.

---

## 12. Quantization and storage coding

Keep these dimensions separately identifiable even when one target profile chooses them together:

```text
logical model/math semantics
quantization profile
scale format
storage codec/profile
execution layout ABI
kernel ABI
```

Examples:

```text
exact MXFP4 + rANS storage + Apple Metal layout
exact MXFP4 + rANS storage + CUDA layout
calibrated mixed 3/4-bit + Apple Metal layout
```

Lossless storage coding is not quantization.

A lossy compiled profile is valid when explicitly identified and verified against its quality policy. It is not required to be reversible to the source checkpoint.

---

## 13. Alignment and I/O

`record_alignment` remains in the v1 manifest/data headers, but v1.1 interprets it as part of the compiled target profile rather than a portable recommendation.

Examples to validate in #26:

```text
Apple Silicon / Metal:
  16 KiB-friendly record/resident layout where measurements support it

Linux / CUDA:
  alignment/read granularity chosen for host I/O + pinned/staging/device path
```

`preferred_read_alignment` and `resident_alignment` in `ColiTargetDescV1` express target requirements that are not necessarily identical to top-level record alignment.

---

## 14. First-class target families

### 14.1 Apple Silicon / macOS

M-series chips are first-class targets. The initial compiler/runtime profile family should cover:

```text
arm64
Metal
explicit Apple GPU-family compatibility
required CPU features
unified-memory/resident-slot layout
streamed-expert disk alignment
```

The first stable profile should be validated on the M2 development target.

The compiler should emit the physical MXFP4/scale layout consumed by #34/#32 directly once that ABI is frozen.

### 14.2 Linux / x86_64 + CUDA

CUDA is a separate first-class target family, not a portable fallback.

A CUDA profile should declare:

```text
x86_64 host requirements
CUDA compute capability
host/staging alignment
pinned/mapped/staging policy when ABI-relevant
GPU-ready weight/scale packing
kernel ABI
```

The same source model may therefore compile into physically different Apple and CUDA `.coli` artifacts with the same source fingerprint.

---

## 15. Core AI / ANE

Core AI is not a portability reason to preserve canonical source tensors.

The stable Apple baseline remains the target-compiled Metal path. If Core AI later requires immutable compiled resident data that is suitable for shipping, it must be represented by an explicit Apple target/profile capability or a distinct profile ABI.

Framework-required runtime specialization caches may remain runtime-generated, but they must derive from compiled semantic/target metadata rather than source safetensors.

See #36.

---

## 16. Source-model policy

Safetensors remains a supported **compiler and verification input**.

It is not required in an installed runtime package.

After successful compilation, a higher-level installer may allow the user to delete the source checkpoint to reclaim storage. The compiler must never mutate/delete source files itself as an implicit side effect.

A future source frontend may support GGUF or another interchange format without changing the target `.coli` runtime contract.

---

## 17. Distribution and cache policy

Common target artifacts may be distributed precompiled, for example:

```text
DeepSeek-V4-Flash/
  macos-arm64-metal-apple8-exact.coli
  macos-arm64-metal-apple8-fast.coli
  linux-x86_64-cuda-sm89-exact.coli
  linux-x86_64-cuda-sm90-exact.coli
```

Selection is based on explicit target/profile compatibility and artifact identity, never filename heuristics alone.

If no compatible artifact exists and source is available, the normal fallback is:

```text
colic compile SOURCE --target native
```

See #28.

---

## 18. Legacy v1.0 policy

The existing `portable-v1` fixture and reader tests are not discarded.

They remain useful for:

- parser regression/fuzz tests;
- documenting the historical v1.0 framing;
- source/oracle experiments where explicitly requested.

However:

- `colic` production output MUST NOT emit `portable-v1`;
- the V4 production runtime MUST NOT select v1.0 portable artifacts as deployable models;
- v1.1 target-aware open is the execution boundary;
- compatibility fallback is recompilation, not runtime repacking.

This allows the work merged in #39/#42 to remain valuable without making its original portability decision permanent.

---

## 19. Required implementation changes

### #22 / spec
- freeze this target descriptor and registry contract;
- freeze exact artifact-fingerprint canonicalization;
- define required target feature handling.

### #23 / loader
- parse v1.1 target metadata;
- add target-aware open/capability validation;
- preserve v1.0 parser support only as legacy/tooling where useful;
- expose physical record/resident metadata without canonical repack.

### #24 / `colic`
- build source frontend + semantic IR + target-lowering pipeline;
- emit v1.1 only for production;
- compute target/artifact identity;
- emit kernel-ready expert records.

### #25 / V4 runtime
- production runtime accepts compatible target `.coli`;
- no permanent safetensors model-source abstraction;
- resolve semantic roles once and execute physical records.

### #26 / target profiles
- freeze the first Apple/Metal and Linux/CUDA physical ABIs;
- share target/profile registry semantics between compiler/runtime.

---

## 20. Acceptance criteria for the v1.1 contract

- [ ] v1.0 portable framing is explicitly legacy, not the production target.
- [ ] v1.1 reuses the v1 header/record framing without ambiguous execution semantics.
- [ ] `CSF_MANIFEST_F_REQUIRED_TARGET_COMPILED` is required for executable v1.1 artifacts.
- [ ] `ColiTargetDescV1` byte layout is frozen.
- [ ] source fingerprint and artifact fingerprint have distinct documented roles.
- [ ] target compatibility is fail-closed and precisely diagnosable.
- [ ] target profiles are reusable hardware compatibility classes by default.
- [ ] physical record payloads may be kernel-ready and need not reconstruct source tensors.
- [ ] Apple Silicon/Metal and Linux/x86_64/CUDA are both represented by the design.
- [ ] no runtime safetensors fallback or portable canonical repack is required.
- [ ] exact profiles remain verifiable against source through tooling.
- [ ] lossy profiles are explicitly identified and quality-gated.

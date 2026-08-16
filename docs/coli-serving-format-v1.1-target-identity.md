# CSF v1.1 target identity amendment

Status: **normative amendment to `docs/coli-serving-format-v1.md` v1.1**.

This file closes two identity gaps found during the pre-merge ABI review. Until
the long-form v1 document is consolidated, the rules below take precedence over
conflicting text in sections 7, 9, and 10 of that document.

## 1. Semantic/model ABI is explicit

The target descriptor must identify both the *semantic model ABI* and the
physical target ABI. A runtime must be able to reject, for example, a compiled
artifact for an engine/model semantic ABI it does not implement even when the
host OS/backend/layout happen to be compatible.

`ColiTargetDesc` remains exactly 256 bytes. The former reserved field at offset
164 is consumed:

| Offset | Bytes | Type | Field | Rule |
|---:|---:|---|---|---|
| 164 | 4 | u32 | `semantic_abi_string_id` | required UTF-8 string ID |
| 168 | 88 | bytes | `reserved` | all zero |

All target-descriptor string IDs are therefore required and in range:

```text
profile_name_string_id
quant_profile_string_id
storage_profile_string_id
optimization_profile_string_id
kernel_profile_string_id
target_triple_string_id
semantic_abi_string_id
```

The semantic ABI string names a versioned compiler/runtime contract, not a
source-checkpoint class name. Examples may include future values such as
`deepseek-v4-exec-v1`; the generic CSF container does not reserve model-specific
strings.

Target compatibility checks MUST include support for the declared semantic ABI
before execution records are published.

## 2. Artifact fingerprint covers every independently variable target identity

The v1.1 artifact-fingerprint canonical stream is amended to include:

- `semantic_abi` string;
- `target_triple` string; and
- the complete `target_flags` u32.

The exact canonical stream is therefore:

```text
ASCII "COLI-ARTIFACT-V1\0"
32 bytes source_fingerprint
u32 compiler_bytes;             compiler UTF-8 bytes
u32 semantic_abi_bytes;         semantic ABI UTF-8 bytes
u32 target_profile_bytes;       target profile UTF-8 bytes
u32 quant_profile_bytes;        quant profile UTF-8 bytes
u32 storage_profile_bytes;      storage profile UTF-8 bytes
u32 optimization_profile_bytes; optimization profile UTF-8 bytes
u32 kernel_profile_bytes;       kernel profile UTF-8 bytes
u32 target_triple_bytes;        target triple UTF-8 bytes
u32 target_flags
u16 target_os
u16 target_arch
u16 backend
u16 gpu_kind
u64 cpu_feature_mask
u32 gpu_family_min
u32 gpu_family_max
u32 gpu_capability_min
u32 gpu_capability_max
u32 target_profile_abi
u32 execution_layout_abi
u32 kernel_abi_min
u32 kernel_abi_max
u32 record_alignment
u32 io_granularity
u32 resident_alignment
u64 required_runtime_features
u8  tuning_valid
32 bytes tuning_fingerprint      # all zero when tuning_valid=0
u64 profile_data_bytes
32 bytes SHA256(profile_data)    # SHA256(empty) when absent
```

All integers are little-endian. Each `u32 *_bytes` is immediately followed by
exactly that many UTF-8 bytes.

`target_flags` participates as the complete serialized value, not a selected
subset. Consequently accelerator requirements, resident accessibility/staging
requirements, and any future optional target flag all affect artifact identity.

The target triple is diagnostic/compatibility identity even when its components
appear elsewhere; hashing it prevents two byte-distinct descriptors with the
same numeric capabilities from sharing an artifact cache key.

## 3. Determinism rule

For deterministic profiles, a conforming writer MUST NOT expose any other
option that can change emitted target bytes without either:

1. changing one of the canonical fingerprint inputs above; or
2. changing `profile_data` (whose byte length and SHA-256 are fingerprinted).

This is the cache-safety rule for target-compiled `.coli`: if bytes can change,
artifact identity must change.

## 4. Fixture semantic ABI

The independent target-compatibility fixtures use:

```text
semantic_abi = fixture-blob-v1
```

because their only execution record is deliberately layout-neutral BLOB data.
They exist to prove generic target-open behavior without claiming a DeepSeek-V4
execution ABI or freezing #26 matrix layout IDs.

# Colibri Serving Format (`.coli`) v1

Status: **target-compiled CSF v1.1 byte contract for issue #22**.

CSF is Colibri's compiled deployment/serving format. Hugging Face/safetensors is
an input to `colic`; the normal inference runtime consumes target-ready `.coli`
records and does not reconstruct the source checkpoint.

The v1.0 `portable-v1` prototype is superseded. Its fixed framing remains useful
for parser history, but production artifacts MUST use v1.1 or later and MUST
carry the required target descriptor defined here. There is no portable
execution profile in the v1.1 runtime contract.

## 1. Design rules

1. All multi-byte integers are little-endian.
2. All top-level offsets and byte lengths are `u64`.
3. No C/C++ struct ABI is serialized directly.
4. Container ABI, target-profile ABI, execution-layout ABI and kernel ABI are
   independently versioned.
5. A routed expert is a first-class independently readable execution record.
6. Mathematical/quant semantics, scale semantics, storage codec and physical
   execution layout remain separate fields.
7. Physical payload bytes are target output. They need not resemble the source
   safetensors bytes and need not be a portable intermediate representation.
8. No compression stream crosses a top-level record boundary.
9. A record never straddles data shards.
10. Runtime state such as KV cache, hotness counters and resident generations is
    not serialized in the model package.
11. Checksums detect corruption; they are not signatures.
12. An incompatible target artifact is rejected before model execution. A
    runtime MUST NOT silently reinterpret or repack an unsupported layout.

## 2. Package layout

A v1 package is a directory:

```text
MODEL.coli/
  manifest.coli
  data-00000.coli
  data-00001.coli
  ...
  config.json / tokenizer assets as emitted by colic
```

The package does not contain or require source safetensors framing. Sidecars may
remain ordinary deterministic files in v1; they are part of source/artifact
provenance when the compiler includes them.

Shard filenames referenced by the manifest MUST be a single portable filename
matching `[A-Za-z0-9._-]+`. `/`, `\\`, `:`, NUL, `.`, and `..` are forbidden.

## 3. Versioning and fail-closed transition

```text
CSF_VERSION_MAJOR = 1
CSF_VERSION_MINOR_TARGET_COMPILED = 1
```

v1.1 keeps the v1.0 256-byte manifest header, 64-byte shard descriptor and
96-byte record descriptor sizes. It consumes previously reserved manifest bytes
for target/artifact identity and sets a required feature bit.

A v1.0-only reader therefore fails closed in two independent ways:

- `version_minor == 1` is newer than it understands; and
- manifest required feature bit 16 is set.

A production v1.1 writer MUST NOT emit `portable-v1`.

## 4. Common constants

```text
CSF_MANIFEST_HEADER_BYTES     = 256
CSF_TARGET_DESC_BYTES         = 256
CSF_SHARD_DESC_BYTES          = 64
CSF_RECORD_DESC_BYTES         = 96
CSF_STRING_DESC_BYTES         = 16
CSF_CODEC_TABLE_DESC_BYTES    = 64
CSF_DATA_HEADER_BYTES         = 128
CSF_TENSOR_HEADER_BYTES       = 128
CSF_EXPERT_HEADER_BYTES       = 64
CSF_EXPERT_MATRIX_DESC_BYTES  = 128
CSF_INTERNAL_ALIGNMENT        = 16
CSF_RANS_READABLE_SLACK       = 64
```

Manifest magic:

```text
43 4f 4c 49 0d 0a 1a 0a     # COLI + CR LF SUB LF
```

Data-shard magic:

```text
43 4f 4c 49 44 41 54 00     # COLIDAT\0
```

Target descriptor magic:

```text
43 4f 4c 49 54 47 54 00     # COLITGT\0
```

Typed record magics remain `COLITENS` and `COLIEXPT`.

The byte-order tag is integer `0x01020304`, serialized little-endian.

## 5. CRC32C

Every CRC is CRC-32C/Castagnoli, reflected polynomial `0x82f63b78`, initial
state `0xffffffff`, final XOR `0xffffffff`.

```text
crc32c("123456789") == 0xe3069283
```

When a structure contains its own CRC field, that field is treated as four zero
bytes while calculating that structure's CRC.

## 6. `manifest.coli` fixed header

The header remains exactly 256 bytes.

| Offset | Bytes | Type | Field | v1.1 rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | manifest magic |
| 8 | 2 | u16 | `version_major` | `1` |
| 10 | 2 | u16 | `version_minor` | `1` |
| 12 | 4 | u32 | `header_bytes` | `256` |
| 16 | 4 | u32 | `flags` | target required; see below |
| 20 | 4 | u32 | `byte_order_tag` | `0x01020304` |
| 24 | 4 | u32 | `record_alignment` | target-required power of two |
| 28 | 4 | u32 | `string_count` | string descriptors |
| 32 | 8 | u64 | `record_count` | execution-record descriptors |
| 40 | 4 | u32 | `shard_count` | data shards |
| 44 | 4 | u32 | `reserved0` | zero |
| 48 | 8 | u64 | `shard_table_offset` | manifest-relative |
| 56 | 8 | u64 | `shard_table_bytes` | `shard_count * 64` |
| 64 | 8 | u64 | `record_table_offset` | manifest-relative |
| 72 | 8 | u64 | `record_table_bytes` | `record_count * 96` |
| 80 | 8 | u64 | `string_table_offset` | manifest-relative |
| 88 | 8 | u64 | `string_table_bytes` | descriptors + bytes + zero pad |
| 96 | 8 | u64 | `metadata_offset` | optional semantic metadata region |
| 104 | 8 | u64 | `metadata_bytes` | zero when absent |
| 112 | 32 | bytes | `source_fingerprint` | compiler-source identity |
| 144 | 4 | u32 | `manifest_crc32c` | complete manifest, field zeroed |
| 148 | 4 | u32 | `profile_name_string_id` | same target profile as target desc |
| 152 | 4 | u32 | `compiler_string_id` | compiler/version identity |
| 156 | 4 | u32 | `reserved1` | zero |
| 160 | 4 | u32 | `codec_table_count` | codec table descriptors |
| 164 | 4 | u32 | `reserved2` | zero |
| 168 | 8 | u64 | `codec_table_offset` | zero when absent |
| 176 | 8 | u64 | `codec_table_bytes` | zero when absent |
| 184 | 8 | u64 | `target_desc_offset` | required, 16-byte aligned |
| 192 | 8 | u64 | `target_desc_bytes` | exactly `256` in v1.1 |
| 200 | 32 | bytes | `artifact_fingerprint` | compiled artifact identity |
| 232 | 4 | u32 | `target_desc_crc32c` | CRC of exact 256-byte descriptor |
| 236 | 4 | u32 | `reserved3` | zero |
| 240 | 16 | bytes | `reserved` | all zero |

### Manifest flags

```text
bit 0       CSF_MANIFEST_F_SOURCE_FINGERPRINT_VALID
bit 1       CSF_MANIFEST_F_ARTIFACT_FINGERPRINT_VALID
bits 2-15   optional/ignorable container feature bits
bit 16      CSF_MANIFEST_F_TARGET_DESCRIPTOR_REQUIRED
bits 17-31  other required container feature bits
```

A target-compiled v1.1 artifact MUST set bits 0, 1 and 16. `source_fingerprint`
and `artifact_fingerprint` MUST be nonzero canonical digests. Unknown required
feature bits are a hard open failure.

`profile_name_string_id` is a quick manifest diagnostic and MUST equal the
`profile_name_string_id` inside the target descriptor.

### Manifest region rules

Every non-empty manifest region is 16-byte aligned, contained in
`manifest.coli`, and non-overlapping with the fixed header and every other
manifest region. Empty optional regions have both offset and bytes equal to
zero. The required target descriptor is never empty.

All count multiplication and `offset + bytes` operations use checked arithmetic
before pointer construction or allocation.

The optional `metadata` region is semantic/compiler metadata, not target layout
metadata. Its schema is architecture/compiler-owned; a generic reader only
bounds-checks it. Required target compatibility must be decidable from the fixed
target descriptor without parsing this region.

## 7. Required target descriptor

`target_desc_offset` points at exactly one 256-byte `ColiTargetDesc`.

### `ColiTargetDesc` (256 bytes)

| Offset | Bytes | Type | Field | Rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | `COLITGT\0` |
| 8 | 2 | u16 | `desc_major` | `1` |
| 10 | 2 | u16 | `desc_minor` | `0` |
| 12 | 4 | u32 | `header_bytes` | `256` |
| 16 | 4 | u32 | `flags` | target flags below |
| 20 | 2 | u16 | `target_os` | registry below |
| 22 | 2 | u16 | `target_arch` | registry below |
| 24 | 2 | u16 | `backend` | registry below |
| 26 | 2 | u16 | `gpu_kind` | registry below |
| 28 | 8 | u64 | `cpu_feature_mask` | required CPU features |
| 36 | 4 | u32 | `gpu_family_min` | family floor; `0` if N/A |
| 40 | 4 | u32 | `gpu_family_max` | `0` = no upper bound |
| 44 | 4 | u32 | `gpu_capability_min` | backend numeric capability floor |
| 48 | 4 | u32 | `gpu_capability_max` | `0` = no upper bound |
| 52 | 4 | u32 | `target_profile_abi` | profile ABI version |
| 56 | 4 | u32 | `execution_layout_abi` | physical layout ABI version |
| 60 | 4 | u32 | `kernel_abi_min` | minimum runtime kernel ABI |
| 64 | 4 | u32 | `kernel_abi_max` | maximum accepted; `0` = no upper bound |
| 68 | 4 | u32 | `record_alignment` | must equal manifest field |
| 72 | 4 | u32 | `io_granularity` | required/preferred physical read unit |
| 76 | 4 | u32 | `resident_alignment` | final resident-slot alignment |
| 80 | 8 | u64 | `required_runtime_features` | backend/profile feature mask |
| 88 | 4 | u32 | `profile_name_string_id` | concrete target profile |
| 92 | 4 | u32 | `quant_profile_string_id` | exact/source/lossy profile |
| 96 | 4 | u32 | `storage_profile_string_id` | none/rANS/etc. |
| 100 | 4 | u32 | `optimization_profile_string_id` | default/size/latency/tuned |
| 104 | 4 | u32 | `kernel_profile_string_id` | kernel interpretation name |
| 108 | 4 | u32 | `target_triple_string_id` | diagnostic target triple |
| 112 | 32 | bytes | `tuning_fingerprint` | zero unless tuned flag set |
| 144 | 8 | u64 | `profile_data_offset` | manifest-relative optional target data |
| 152 | 8 | u64 | `profile_data_bytes` | zero when absent |
| 160 | 4 | u32 | `profile_data_crc32c` | zero when absent |
| 164 | 4 | u32 | `reserved0` | zero |
| 168 | 88 | bytes | `reserved` | all zero |

`profile_data` is an optional target-profile-specific immutable parameter block.
Its byte schema is owned by the named target-profile ABI. A generic reader
bounds-checks it and validates its CRC; target-specific code may interpret it.
It may contain lowering parameters such as block sizes or compiled kernel
constants, but never host pointers/handles.

### Target descriptor flags

```text
bit 0       CSF_TARGET_F_TUNING_FINGERPRINT_VALID
bit 1       CSF_TARGET_F_ACCELERATOR_REQUIRED
bit 2       CSF_TARGET_F_RESIDENT_CPU_READABLE
bit 3       CSF_TARGET_F_RESIDENT_GPU_READABLE
bit 4       CSF_TARGET_F_STAGING_REQUIRED
bits 5-15   optional target flags
bits 16-31  required target flags
```

If tuned flag is clear, `tuning_fingerprint` is 32 zero bytes.

## 8. Target identity registries

### OS (`u16`)

```text
0  INVALID
1  MACOS
2  LINUX
3  WINDOWS
```

### Architecture (`u16`)

```text
0  INVALID
1  ARM64
2  X86_64
```

### Backend (`u16`)

```text
0  INVALID
1  CPU
2  METAL
3  CUDA
4  HYBRID
```

### GPU-kind namespace (`u16`)

```text
0  NONE
1  APPLE_GPU_FAMILY
2  CUDA_SM
```

For `APPLE_GPU_FAMILY`, `gpu_family_min`/`max` use the Apple Metal family
number (for example Apple family 8 is numeric family `8`).

For `CUDA_SM`, `gpu_capability_*` uses decimal compute capability `major*10 +
minor`, for example sm_89 = `89`.

### CPU feature mask (`u64`)

```text
bit 0  ARM64_ASIMD
bit 1  X86_AVX2
bit 2  X86_FMA
bit 3  X86_AVX512F
bit 4  X86_AVX512_BF16
```

Additional bits require a compatible target-profile ABI revision/registry
update. A runtime satisfies the artifact only when:

```text
(required_cpu_features & runtime_cpu_features) == required_cpu_features
```

### Required runtime feature mask (`u64`)

Initial generic bits:

```text
bit 0  APPLE_UNIFIED_MEMORY
bit 1  METAL_SHARED_STORAGE
bit 2  CUDA_RUNTIME
bit 3  CUDA_ASYNC_COPY
bit 4  HOST_PINNED_STAGING
```

Profile-specific required behavior belongs in the target profile and optional
profile data; generic bits should be added only when multiple profiles benefit.

## 9. Compatibility algorithm

`#23` package open for execution compares the required target descriptor with a
runtime capability descriptor before publishing execution records.

At minimum:

1. OS and architecture match exactly.
2. Backend matches or the runtime explicitly declares the named hybrid profile.
3. Required CPU bits are a subset of runtime CPU bits.
4. Required runtime-feature bits are a subset of runtime feature bits.
5. GPU kind matches when accelerator-required is set.
6. Apple family requirement succeeds when the runtime reports support for the
   required minimum family and does not violate a nonzero maximum.
7. CUDA capability lies within the declared min/max range.
8. Target-profile name and `target_profile_abi` are supported.
9. `execution_layout_abi` is supported by that profile.
10. Runtime kernel ABI lies in `[kernel_abi_min, kernel_abi_max]`, treating max
    zero as unbounded.
11. Quantization and storage profile IDs are supported by that runtime profile.
12. Runtime can satisfy record/I/O/resident alignment requirements.

Failure is reported as required-vs-provided capabilities. There is no automatic
canonical fallback or record-sized runtime repack.

## 10. Artifact fingerprint

Source identity and compiled artifact identity are distinct.

`source_fingerprint` uses the v1 source-inventory procedure: hash the canonical
source files used by `colic` without timestamps, inode data, or absolute paths.

`artifact_fingerprint` is SHA-256 over this exact canonical stream:

```text
ASCII "COLI-ARTIFACT-V1\0"
32 bytes source_fingerprint
u32 compiler_bytes;             compiler UTF-8 bytes
u32 target_profile_bytes;       target profile UTF-8 bytes
u32 quant_profile_bytes;        quant profile UTF-8 bytes
u32 storage_profile_bytes;      storage profile UTF-8 bytes
u32 optimization_profile_bytes; optimization profile UTF-8 bytes
u32 kernel_profile_bytes;       kernel profile UTF-8 bytes
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

All integer fields are little-endian. String lengths precede exactly that
string's bytes. The compiler MUST place every deterministic lowering option
that can change emitted bytes into one of these named profiles or into
`profile_data`; otherwise artifact cache identity is incomplete.

The artifact fingerprint intentionally does not hash the final manifest/shards,
which would be circular because the fingerprint is stored in the manifest.
For deterministic profiles, identical canonical inputs above MUST produce
byte-identical artifacts.

## 11. String table

Unchanged from v1 framing. The string table starts with `string_count` 16-byte
descriptors:

```text
offset +0   u64 data_offset relative to string-table start
offset +8   u32 data_bytes
offset +12  u32 flags = 0
```

Strings are UTF-8, have no terminating NUL, and cannot contain embedded NUL.
Canonical writers store bytes tightly in ID order and zero-pad the table to a
16-byte boundary. `UINT32_MAX` is the explicit no-string sentinel where a field
allows it.

All string IDs in the target descriptor are required and in range. A target
profile is identified by the resolved string plus its numeric ABI, never by an
unversioned heuristic.

## 12. Shard descriptor and data header

`ColiShardDesc` remains 64 bytes:

```text
+0   u32 shard_id
+4   u32 flags = 0
+8   u32 file_name_string_id
+12  u32 reserved = 0
+16  u64 file_bytes
+24  u32 header_crc32c
+28  u32 reserved = 0
+32  32 bytes reserved = 0
```

Data shard header remains 128 bytes:

```text
+0   8 bytes COLIDAT\0
+8   u16 version_major = 1
+10  u16 version_minor = 1
+12  u32 header_bytes = 128
+16  u32 flags = 0
+20  u32 shard_id
+24  u32 record_alignment        # matches manifest/target descriptor
+28  u32 reserved = 0
+32  u64 file_bytes
+40  32 bytes source_fingerprint
+72  u32 header_crc32c
+76  52 bytes reserved = 0
```

The source fingerprint remains in each shard header for fast package-mixup
detection. Artifact identity is manifest-level because every shard belongs to
that manifest and target descriptor.

Records begin no earlier than `align_up(128, record_alignment)` and every
top-level record offset obeys target record alignment. Canonical compiler output
zero-fills deterministic alignment gaps.

## 13. Execution record descriptor

The fixed descriptor remains 96 bytes; semantics become explicitly physical:

| Offset | Bytes | Type | Field |
|---:|---:|---|---|
| 0 | 8 | u64 | `record_id` |
| 8 | 2 | u16 | `kind` |
| 10 | 2 | u16 | `codec` |
| 12 | 2 | u16 | `math_format` |
| 14 | 2 | u16 | `scale_format` |
| 16 | 2 | u16 | `layout` |
| 18 | 2 | u16 | `flags` |
| 20 | 4 | u32 | `shard_id` |
| 24 | 4 | u32 | `name_string_id` or `UINT32_MAX` |
| 28 | 4 | i32 | `layer` |
| 32 | 4 | i32 | `expert` |
| 36 | 4 | u32 | `reserved0` = 0 |
| 40 | 8 | u64 | `payload_offset` |
| 48 | 8 | u64 | `stored_bytes` |
| 56 | 8 | u64 | `resident_bytes` |
| 64 | 4 | u32 | `stored_crc32c` |
| 68 | 4 | u32 | `logical_crc32c` |
| 72 | 4 | u32 | `codec_table_id` |
| 76 | 4 | u32 | `reserved1` = 0 |
| 80 | 16 | bytes | `reserved` = 0 |

The old field name `decoded_bytes` is superseded by `resident_bytes`. For codec
NONE they may still differ when the stored record contains framing/padding.
For a storage codec, `resident_bytes` is the exact final target byte count after
decode, not a source/canonical tensor byte count.

`logical_crc32c`, when present, is a tooling/oracle checksum over the record
kind's declared logical ordering after any inverse target transform required by
that exact/lossless profile. The production runtime does not inverse-transform
records to verify it on every load.

### Record flags

```text
bit 0       OPTIONAL
bit 1       HAS_LOGICAL_CRC32C
bits 2-7    optional flags
bits 8-15   required record flags
```

## 14. Record kinds and semantic/physical boundary

```text
0x0000 INVALID
0x0001 TENSOR
0x0002 EXPERT
0x0003 LAYER_PACK
0x0004 BLOB
```

`TENSOR`, `EXPERT` and `LAYER_PACK` are execution records. Their `layout` is a
physical target layout registered by #26 unless the record's semantics are
explicitly layout-neutral.

`BLOB` is immutable auxiliary data such as target-neutral configuration. It may
use `layout=NONE`.

The compiler may preserve source tensor names for diagnostics, but target
runtime architecture must not depend on source name reconstruction for hot
records.

## 15. Mathematical, scale, codec and layout registries

Mathematical and scale IDs retain the existing v1 values:

```text
math: NONE=0, F32=1, F16=2, BF16=3, I8=4, U8=5,
      I16=6, U16=7, I32=8, U32=9, I64=10, U64=11,
      BOOL=12, FP8_E4M3FN=0x10, FP8_E5M2=0x11,
      MXFP4_E2M1=0x20, INT4_PACKED=0x21, INT4_GROUPED=0x22,
      MIXED=0xfffe
scale: NONE=0, F32=1, F16=2, BF16=3, UE8M0=4, MIXED=0xfffe
codec: NONE=0, RANS256_G0_NIBBLE=1, RANS256_G0_U8=2
```

Execution-layout IDs are target ABI identifiers, not source tensor-format IDs:

```text
0x0000          NONE / layout-neutral auxiliary bytes
0x0001-0x00ff   reserved generic/tooling layouts; not production fallback
0x0100-0x01ff   Apple target layouts (#26)
0x0200-0x02ff   x86 CPU target layouts (#26)
0x0300-0x03ff   CUDA target layouts (#26)
0xfffe          MIXED compound outer record only
0xffff          INVALID
```

A numeric range reservation does not make an ID executable. #26 registers the
concrete ID, byte layout, resident layout and kernel ABI together. Required
records using an unknown layout ID are rejected.

## 16. Codec table region

Codec table framing remains 64-byte descriptors followed by 16-byte-aligned
codec-specific blobs:

```text
+0   u32 table_id (nonzero unique)
+4   u16 codec
+6   u16 flags
+8   i32 shard_id (-1 package-global)
+12  u32 reserved
+16  u64 data_offset relative to codec-table region
+24  u64 data_bytes
+32  u32 data_crc32c
+36  u32 reserved
+40  24 bytes reserved
```

A shard-scoped table can only be referenced from that shard. Codec semantics
and direct-to-target decode contracts are registered jointly with #6/#26.

The container provides 64 readable zero bytes after rANS subblobs when the
registered decoder requires branch-free readable overrun. Slack is included in
the containing top-level `stored_bytes`/stored CRC but excluded from the
subblob's codec length.

## 17. `TENSOR` payload envelope

The 128-byte envelope framing is retained so tooling can inspect semantic shape
while the data blob is target physical bytes:

```text
+0    8 bytes COLITENS
+8    u16 envelope_major = 1
+10   u16 envelope_minor = 1
+12   u32 header_bytes = 128
+16   u16 rank (0..8)
+18   u16 flags
+20   u32 scale_block_rows
+24   u32 scale_block_columns
+28   u32 group_size
+32   u64 dims[8]             # logical dimensions
+96   u64 data_offset
+104  u64 data_stored_bytes
+112  u64 data_resident_bytes
+120  u32 logical_crc32c
+124  u32 reserved
```

`dims` describe logical model geometry. `data_resident_bytes` describes target
resident bytes. Neither field implies source safetensors shape or byte layout.
Target `layout` + target-profile ABI define the physical interpretation.

## 18. `EXPERT` payload envelope

A routed expert still has one 64-byte header followed by three fixed 128-byte
matrix descriptors in semantic order `GATE`, `UP`, `DOWN`.

Outer record:

```text
kind         = EXPERT
math_format  = MIXED
scale_format = MIXED
layout       = MIXED
layer >= 0
expert >= 0
```

### Expert header (64 bytes)

```text
+0   8 bytes COLIEXPT
+8   u16 envelope_major = 1
+10  u16 envelope_minor = 1
+12  u32 header_bytes = 64
+16  i32 layer
+20  i32 expert
+24  u16 matrix_count = 3
+26  u16 flags
+28  u32 matrix_desc_bytes = 128
+32  u64 matrix_table_offset = 64
+40  u64 data_offset >= 448
+48  u64 logical_bytes       # tooling semantic byte count when meaningful
+56  u64 reserved = 0
```

### Expert matrix descriptor (128 bytes)

```text
+0   u16 role                # GATE=1, UP=2, DOWN=3
+2   u16 flags
+4   u16 math_format
+6   u16 scale_format
+8   u16 weight_codec
+10  u16 scale_codec
+12  u16 layout              # concrete target layout ID
+14  u16 reserved
+16  u64 rows                # logical rows
+24  u64 columns             # logical columns
+32  u32 scale_block_rows
+36  u32 scale_block_columns
+40  u32 weight_codec_table_id
+44  u32 scale_codec_table_id
+48  u64 weight_offset
+56  u64 weight_stored_bytes
+64  u64 weight_resident_bytes
+72  u64 scale_offset
+80  u64 scale_stored_bytes
+88  u64 scale_resident_bytes
+96  u32 logical_crc32c
+100 u32 reserved
+104 u32 group_size
+108 u32 reserved
+112 16 bytes reserved
```

The matrix `layout` and the package target/execution-layout ABI jointly define
how weight/scale bytes land in final resident slots. A runtime must not assume
low-nibble-first source MXFP4 or any other canonical source packing unless that
specific registered target layout says so.

Weight/scale subranges are 16-byte aligned at minimum, contained in the record,
and non-overlapping. Target profiles may require stronger alignment.

## 19. Source fingerprint

The source fingerprint remains deterministic provenance identity. The compiler
constructs a sorted canonical inventory including all source checkpoint files
and sidecars it consumes.

For each inventory file compute SHA-256 while streaming its bytes. Hash:

```text
ASCII "COLI-SOURCE-V1\0"
u32 entry_count
for each entry sorted by raw UTF-8 relative path bytes:
    u8 kind              # 1=safetensors, 2=index, 3=sidecar/other source
    u32 path_bytes
    path bytes
    u64 file_bytes
    32-byte SHA256(file_bytes)
```

Absolute paths, timestamps, ownership and inode metadata are forbidden from the
canonical stream.

## 20. Deterministic target writer

For a deterministic target/optimization profile, identical canonical artifact
fingerprint inputs produce byte-identical output:

- no timestamps in CSF binary files;
- deterministic record IDs/order and shard names;
- deterministic target lowering and padding;
- all reserved/padding bytes zero unless a registered layout says otherwise;
- target/profile data deterministic;
- checksums computed after final bytes are known;
- source files remain unchanged;
- final artifact is published atomically after verification.

Machine autotuning is allowed only when `TUNING_FINGERPRINT_VALID` is set and
the benchmark-selected tuning identity is included in artifact fingerprinting.

## 21. Initial target fixture descriptors

These fixtures freeze compatibility metadata only. Concrete expert execution
layout byte IDs remain #26 work and are not invented by the container spec.
The fixture record is a layout-neutral `BLOB`, so it can validate target-open
logic before target kernels land.

### Apple M2 / Metal compatibility fixture

```text
profile_name                 = macos-arm64-metal-apple8-v1
target_os                    = MACOS
target_arch                  = ARM64
backend                      = METAL
gpu_kind                     = APPLE_GPU_FAMILY
cpu_feature_mask             = ARM64_ASIMD
gpu_family_min               = 8
gpu_family_max               = 0          # newer compatible family allowed
execution_layout_abi         = 1           # profile ABI framing; no matrix in fixture
kernel_abi_min               = 1
kernel_abi_max               = 1
record_alignment             = 16384
io_granularity               = 16384
resident_alignment           = 16384
required_runtime_features    = APPLE_UNIFIED_MEMORY | METAL_SHARED_STORAGE
```

The M2 compatibility floor is Apple GPU family 8. Concrete runtime detection
uses Metal's family-support query rather than a product-name string.

### Linux x86_64 / CUDA sm_89 compatibility fixture

```text
profile_name                 = linux-x86_64-cuda-sm89-v1
target_os                    = LINUX
target_arch                  = X86_64
backend                      = CUDA
gpu_kind                     = CUDA_SM
cpu_feature_mask             = X86_AVX2 | X86_FMA
gpu_capability_min           = 89
gpu_capability_max           = 89
execution_layout_abi         = 1
kernel_abi_min               = 1
kernel_abi_max               = 1
record_alignment             = 4096
io_granularity               = 4096
resident_alignment           = 256
required_runtime_features    = CUDA_RUNTIME | CUDA_ASYNC_COPY | HOST_PINNED_STAGING
```

#26 may introduce a wider reusable CUDA compatibility class after the kernel
binary/PTX compatibility policy is fixed. The fixture deliberately uses an
exact sm_89 capability so the generic loader test has unambiguous pass/fail
semantics.

## 22. Required parser/refusal behavior

Before execution record publication, the generic reader validates:

1. manifest magic/version/header/byte-order;
2. known required container flags, including target-required bit;
3. checked manifest region bounds/non-overlap;
4. exact descriptor table byte counts;
5. manifest CRC32C;
6. string table framing/UTF-8/path safety;
7. required target descriptor size/magic/reserved fields/CRC;
8. artifact/source fingerprint flag consistency;
9. profile-data bounds/CRC;
10. runtime target compatibility;
11. shard headers, fingerprint, alignment and actual file sizes;
12. record IDs, target layout/codec combinations, aligned spans and duplicate
    `(layer,expert)` keys;
13. type-specific envelopes lazily on record use/deep verify;
14. stored CRC on record load when policy requests it;
15. logical/inverse-transform verification only in tooling/exact-profile paths.

At minimum, reject with a precise category for:

```text
bad magic / unsupported container version
unknown required container or target feature
missing/invalid target descriptor
bad source/artifact fingerprint framing
integer overflow / region outside file / overlap
bad manifest or target/profile-data CRC
invalid UTF-8 / shard traversal
wrong target OS/arch/backend
missing required CPU/backend features
unsupported Apple family / CUDA capability
unsupported target-profile/layout/kernel ABI
unsupported quant/storage profile
bad shard header / size mismatch
zero/duplicate record ID
record outside shard / record overlap
duplicate routed expert
invalid codec table reference
truncated typed envelope / invalid target subrange
bad stored CRC / logical verification failure
codec structural error
```

Normal package open does not scan multi-gigabyte payloads.

## 23. Versioning policy

- Container major changes only when fixed framing cannot remain compatible.
- Container minor may consume reserved fields and add required features when
  older readers fail closed via minor/required-feature checks.
- Target-profile ABI changes independently from container major/minor.
- Execution-layout and kernel ABI change independently from target-profile
  naming but are constrained by that profile.
- Unknown required target/runtime semantics are rejected, never guessed.
- An artifact compiled for a different target is not converted in place by the
  runtime; the user/compiler cache must produce a compatible artifact.

## 24. Scope owned by related issues

This document freezes the generic container and target-identity framing.

- #26 registers concrete Apple/x86/CUDA execution-layout bytes, resident slot
  layouts and kernel ABIs.
- #24 implements `colic` frontend/IR/lowering/writer and artifact identity.
- #23 implements strict target-aware open and immutable indexes.
- #25 consumes target execution records in V4 without safetensors.
- #6 defines optional lossless rANS codec semantics/direct target decode.
- #14 may add calibrated lossy target quantization profiles.

The normal v1.1 lifecycle is:

```text
HF/safetensors source
  -> colic semantic IR
  -> concrete #26 target lowering
  -> target-compiled .coli v1.1
  -> #23 compatibility validation
  -> #25 final-layout execution
```

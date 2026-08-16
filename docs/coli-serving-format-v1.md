# Colibri Serving Format (`.coli`) v1

Status: **v1.0 byte contract for issue #22**.

This document defines the portable Colibri Serving Format (CSF) container used by the offline compiler and strict reader. Safetensors remains the source/interchange format. A `.coli` package is a reproducible compiled serving artifact optimized for random-access MoE inference.

The v1.0 design intentionally freezes only the pieces needed to unblock the compiler (#24), reader (#23), V4 source integration (#25), and lossless codec integration (#6). Native execution profiles remain a later concern (#26).

## 1. Design rules

1. All integers are unsigned unless explicitly declared signed.
2. All multi-byte integers are serialized **little-endian**, regardless of host ABI.
3. All offsets and byte lengths in top-level descriptors are `u64`.
4. No C/C++ struct ABI is serialized directly. Writers emit each field at the byte offsets below; readers decode each field explicitly.
5. A routed expert is a first-class compound record containing gate/up/down matrices and their scale streams.
6. Mathematical format, scale format, storage codec, and execution layout are independent fields.
7. Every hot record is independently addressable. No compression stream may cross a top-level record boundary.
8. A record never straddles data shards.
9. Checksums detect accidental corruption; they are not signatures.
10. Runtime state (KV cache, prompt cache, hot-expert counters, RAM plans) is not stored in the model package.

## 2. Directory package

A v1 package is a directory:

```text
MODEL.coli/
  manifest.coli
  data-00000.coli
  data-00001.coli
  ...
  config.json
  tokenizer.json
  tokenizer_config.json       # when present in source
  chat_template.jinja         # when present in source
  ... copied source sidecars
```

`manifest.coli` and the `data-*.coli` files are CSF binary files. Model/tokenizer sidecars remain ordinary files in v1 so adopting CSF does not require another config/tokenizer schema.

Shard filenames referenced by the manifest MUST be a single portable filename matching `[A-Za-z0-9._-]+`; `/`, `\\`, `:`, NUL, `.` and `..` are forbidden. This makes path traversal impossible without platform-specific path heuristics.

## 3. Common constants

```text
CSF_VERSION_MAJOR             = 1
CSF_VERSION_MINOR             = 0
CSF_MANIFEST_HEADER_BYTES     = 256
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

Manifest magic, exactly 8 bytes:

```text
43 4f 4c 49 0d 0a 1a 0a     # "COLI" + CR LF SUB LF
```

Data-shard magic, exactly 8 bytes:

```text
43 4f 4c 49 44 41 54 00     # "COLIDAT\0"
```

Typed payload magics:

```text
COLITENS                     # tensor envelope, 8 ASCII bytes
COLIEXPT                     # expert envelope, 8 ASCII bytes
```

The byte-order tag is the integer `0x01020304`, serialized little-endian.

## 4. CRC32C

Every CRC in v1 is CRC-32C (Castagnoli), reflected polynomial `0x82f63b78`, initial state `0xffffffff`, final XOR `0xffffffff`. The standard check value is:

```text
crc32c("123456789") == 0xe3069283
```

When a structure contains its own CRC field, that field is treated as four zero bytes while the CRC is computed.

## 5. `manifest.coli` header

The fixed v1 header is 256 bytes.

| Offset | Bytes | Type | Field | v1.0 rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | manifest magic above |
| 8 | 2 | u16 | `version_major` | `1` |
| 10 | 2 | u16 | `version_minor` | `0` for v1.0 writer |
| 12 | 4 | u32 | `header_bytes` | `256` |
| 16 | 4 | u32 | `flags` | see below |
| 20 | 4 | u32 | `byte_order_tag` | `0x01020304` |
| 24 | 4 | u32 | `record_alignment` | power of two |
| 28 | 4 | u32 | `string_count` | number of string descriptors |
| 32 | 8 | u64 | `record_count` | number of record descriptors |
| 40 | 4 | u32 | `shard_count` | number of shard descriptors |
| 44 | 4 | u32 | `reserved0` | zero |
| 48 | 8 | u64 | `shard_table_offset` | manifest-relative |
| 56 | 8 | u64 | `shard_table_bytes` | exactly `shard_count * 64` |
| 64 | 8 | u64 | `record_table_offset` | manifest-relative |
| 72 | 8 | u64 | `record_table_bytes` | exactly `record_count * 96` |
| 80 | 8 | u64 | `string_table_offset` | manifest-relative |
| 88 | 8 | u64 | `string_table_bytes` | includes descriptors + bytes + zero pad |
| 96 | 8 | u64 | `metadata_offset` | `0` when absent |
| 104 | 8 | u64 | `metadata_bytes` | `0` when absent |
| 112 | 32 | bytes | `source_fingerprint` | SHA-256 or zero when flag absent |
| 144 | 4 | u32 | `manifest_crc32c` | CRC over complete `manifest.coli`, this field zeroed |
| 148 | 4 | u32 | `profile_name_string_id` | e.g. `portable-v1` |
| 152 | 4 | u32 | `compiler_string_id` | compiler identity/version |
| 156 | 4 | u32 | `reserved1` | zero |
| 160 | 4 | u32 | `codec_table_count` | number of codec table descriptors |
| 164 | 4 | u32 | `reserved2` | zero |
| 168 | 8 | u64 | `codec_table_offset` | `0` when count is zero |
| 176 | 8 | u64 | `codec_table_bytes` | `0` when count is zero |
| 184 | 72 | bytes | `reserved` | all zero |

### Manifest flags

```text
bit 0      CSF_MANIFEST_F_SOURCE_FINGERPRINT_VALID
bits 1-15  optional/ignorable feature bits; zero in v1.0
bits 16-31 required feature bits; zero in v1.0
```

A reader MAY ignore an unknown optional bit. It MUST reject an unknown required bit.

If `SOURCE_FINGERPRINT_VALID` is clear, `source_fingerprint` MUST be 32 zero bytes. The production compiler from #24 sets the flag and writes the canonical source fingerprint described below.

### Manifest region rules

All non-empty manifest regions begin at a 16-byte-aligned offset, are fully contained in `manifest.coli`, and do not overlap the fixed header or each other. Empty regions have both offset and byte length equal to zero.

Before multiplying counts by descriptor sizes or adding offsets and lengths, readers MUST use checked arithmetic.

## 6. String table

The string table begins with `string_count` fixed 16-byte descriptors. String IDs are zero-based descriptor indexes.

### `ColiStringDesc` (16 bytes)

| Offset | Bytes | Type | Field |
|---:|---:|---|---|
| 0 | 8 | u64 | `data_offset` relative to start of string table |
| 8 | 4 | u32 | `data_bytes` |
| 12 | 4 | u32 | `flags` (`0` in v1.0) |

Strings are UTF-8 byte strings without a terminating NUL. Embedded NUL is forbidden. In canonical writer output, string data follows the descriptor array tightly in ID order; any final padding to a 16-byte table size is zero.

`UINT32_MAX` is the sentinel for “no string” in descriptor fields that explicitly permit unnamed records.

## 7. Shard descriptor

`shard_table_bytes` is exactly `shard_count * 64`.

### `ColiShardDesc` (64 bytes)

| Offset | Bytes | Type | Field | v1.0 rule |
|---:|---:|---|---|---|
| 0 | 4 | u32 | `shard_id` | canonical writer uses table index |
| 4 | 4 | u32 | `flags` | zero |
| 8 | 4 | u32 | `file_name_string_id` | valid portable filename |
| 12 | 4 | u32 | `reserved0` | zero |
| 16 | 8 | u64 | `file_bytes` | exact `fstat` size |
| 24 | 4 | u32 | `header_crc32c` | copy of validated data-header CRC |
| 28 | 4 | u32 | `reserved1` | zero |
| 32 | 32 | bytes | `reserved` | zero |

Shard IDs are unique. v1.0 canonical output uses contiguous IDs `0..shard_count-1`.

## 8. Data-shard header

Every `data-*.coli` shard begins with a 128-byte header.

| Offset | Bytes | Type | Field | v1.0 rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | `COLIDAT\0` |
| 8 | 2 | u16 | `version_major` | `1` |
| 10 | 2 | u16 | `version_minor` | `0` |
| 12 | 4 | u32 | `header_bytes` | `128` |
| 16 | 4 | u32 | `flags` | zero |
| 20 | 4 | u32 | `shard_id` | must match manifest descriptor |
| 24 | 4 | u32 | `record_alignment` | must match manifest |
| 28 | 4 | u32 | `reserved0` | zero |
| 32 | 8 | u64 | `file_bytes` | must match manifest and actual file size |
| 40 | 32 | bytes | `source_fingerprint` | byte-identical to manifest field |
| 72 | 4 | u32 | `header_crc32c` | CRC over 128-byte header with this field zeroed |
| 76 | 52 | bytes | `reserved` | zero |

The first possible record starts at `align_up(128, record_alignment)`. Top-level record offsets are always aligned to `record_alignment`.

Alignment gaps are zero-filled by canonical writers. Normal package open does not need to scan those gaps; deep verification tooling may validate them.

## 9. Record descriptor

`record_table_bytes` is exactly `record_count * 96`.

### `ColiRecordDesc` (96 bytes)

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
| 24 | 4 | u32 | `name_string_id` |
| 28 | 4 | i32 | `layer` (`-1` = not layer-scoped) |
| 32 | 4 | i32 | `expert` (`-1` = not expert-scoped) |
| 36 | 4 | u32 | `reserved0` (zero) |
| 40 | 8 | u64 | `payload_offset` (shard-relative) |
| 48 | 8 | u64 | `stored_bytes` |
| 56 | 8 | u64 | `decoded_bytes` |
| 64 | 4 | u32 | `stored_crc32c` |
| 68 | 4 | u32 | `logical_crc32c` |
| 72 | 4 | u32 | `codec_table_id` (`0` = none) |
| 76 | 4 | u32 | `reserved1` (zero) |
| 80 | 16 | bytes | `reserved` (zero) |

Record IDs are nonzero and unique. `payload_offset` is aligned to `record_alignment`; `payload_offset + stored_bytes` must fit in the referenced shard with checked addition.

Top-level record ranges within a shard may not overlap.

### Record flags

```text
bit 0      CSF_RECORD_F_OPTIONAL
bit 1      CSF_RECORD_F_HAS_LOGICAL_CRC32C
bits 2-7   optional/ignorable record flags; zero in v1.0
bits 8-15  required record flags; zero in v1.0
```

`stored_crc32c` is always present and covers exactly the `stored_bytes` range, including typed envelope bytes and codec-required internal slack, but excluding top-level alignment padding.

If `HAS_LOGICAL_CRC32C` is set, `logical_crc32c` is interpreted according to record kind. A CRC result of zero is valid; the flag, not the numeric value, says whether it exists.

An unknown record kind may be skipped only when `OPTIONAL` is set. Unknown codec/math/layout required to interpret a non-optional record is a hard unsupported-feature error.

## 10. Enum registry

These values are CSF container IDs. They are **not** the historical `QT.fmt` public ordinals in `docs/FORMATS.md`.

### Record kind (`u16`)

```text
0x0000  INVALID
0x0001  TENSOR
0x0002  EXPERT
0x0003  LAYER_PACK_RESERVED
0x0004  BLOB
```

`LAYER_PACK_RESERVED` reserves the v1 kind number but **portable-v1 minor 0 MUST NOT emit it**. Its payload ABI is intentionally deferred. A v1.0 reader skips it only if marked optional; otherwise it returns unsupported. This avoids freezing a dense-layer packing ABI before it is useful.

### Storage codec (`u16`)

```text
0x0000  NONE
0x0001  RANS256_G0_NIBBLE
0x0002  RANS256_G0_U8
```

The rANS algorithms/table blob semantics are owned by #6. CSF only defines how a record refers to a codec table and how much readable slack the container supplies.

### Mathematical format (`u16`)

```text
0x0000  NONE
0x0001  F32
0x0002  F16
0x0003  BF16
0x0004  I8
0x0005  U8
0x0006  I16
0x0007  U16
0x0008  I32
0x0009  U32
0x000a  I64
0x000b  U64
0x000c  BOOL
0x0010  FP8_E4M3FN
0x0011  FP8_E5M2
0x0020  MXFP4_E2M1
0x0021  INT4_PACKED
0x0022  INT4_GROUPED
0xfffe  MIXED              # compound record only
0xffff  INVALID
```

### Scale format (`u16`)

```text
0x0000  NONE
0x0001  F32
0x0002  F16
0x0003  BF16
0x0004  UE8M0
0xfffe  MIXED              # compound record only
0xffff  INVALID
```

### Execution layout (`u16`)

```text
0x0000  CANONICAL
0x0001  ROWS16             # ID registered; not portable-v1
0x0100-0x01ff  reserved for versioned Apple-native layouts
0x0200-0x02ff  reserved for versioned x86-native layouts
0x0300-0x03ff  reserved for versioned CUDA-native layouts
0xfffe  MIXED              # compound record only
0xffff  INVALID
```

`portable-v1` permits only `CANONICAL`. A native layout is not valid merely because its numeric range is reserved; #26 must freeze its byte contract before a required record can use it.

### Expert matrix role (`u16`)

```text
0x0001  GATE
0x0002  UP
0x0003  DOWN
```

The model compiler maps source tensor naming to these semantic roles. CSF itself does not encode model-specific `w1/w2/w3` names.

## 11. Codec table region

Codec tables let #6 use package-global or shard-local static probability tables without embedding a copy in every expert.

When `codec_table_count == 0`, both codec-table offset and byte length are zero. Otherwise the region starts with `codec_table_count` fixed 64-byte descriptors followed by codec-specific table blobs.

### `ColiCodecTableDesc` (64 bytes)

| Offset | Bytes | Type | Field |
|---:|---:|---|---|
| 0 | 4 | u32 | `table_id` (nonzero, unique) |
| 4 | 2 | u16 | `codec` |
| 6 | 2 | u16 | `flags` (zero in v1.0) |
| 8 | 4 | i32 | `shard_id` (`-1` package-global, otherwise shard-scoped) |
| 12 | 4 | u32 | `reserved0` (zero) |
| 16 | 8 | u64 | `data_offset` relative to codec-table region |
| 24 | 8 | u64 | `data_bytes` |
| 32 | 4 | u32 | `data_crc32c` |
| 36 | 4 | u32 | `reserved1` (zero) |
| 40 | 24 | bytes | `reserved` (zero) |

Codec table blobs are 16-byte aligned, contained in the codec-table region, non-overlapping, and covered by `data_crc32c`.

A `NONE` codec uses `codec_table_id == 0`. A codec that requires a table uses a nonzero ID whose descriptor has the same codec. A shard-scoped table may only be referenced by a record in that shard.

## 12. Tensor payload

A `TENSOR` record stores one typed envelope plus one data blob. The record descriptor's `codec` applies to the data blob, not to the 128-byte envelope.

### `ColiTensorHeader` (128 bytes)

| Offset | Bytes | Type | Field |
|---:|---:|---|---|
| 0 | 8 | bytes | `COLITENS` |
| 8 | 2 | u16 | `version_major` = 1 |
| 10 | 2 | u16 | `version_minor` = 0 |
| 12 | 4 | u32 | `header_bytes` = 128 |
| 16 | 2 | u16 | `rank` (`0..8`) |
| 18 | 2 | u16 | `flags` (zero in v1.0) |
| 20 | 4 | u32 | `scale_block_rows` (`0` when not applicable) |
| 24 | 4 | u32 | `scale_block_columns` (`0` when not applicable) |
| 28 | 4 | u32 | `group_size` (`0` when not applicable) |
| 32 | 64 | u64[8] | `dims`; entries >= rank are zero |
| 96 | 8 | u64 | `data_offset` relative to tensor envelope |
| 104 | 8 | u64 | `data_stored_bytes` |
| 112 | 8 | u64 | `data_decoded_bytes` |
| 120 | 4 | u32 | `logical_crc32c` |
| 124 | 4 | u32 | `reserved` (zero) |

`data_offset` is at least 128 and 16-byte aligned. `data_decoded_bytes` equals the outer descriptor's `decoded_bytes`. If `HAS_LOGICAL_CRC32C` is set, the header CRC value equals the outer `logical_crc32c` and covers the decoded tensor data bytes only.

For `codec == NONE`, `data_stored_bytes == data_decoded_bytes`, `codec_table_id == 0`, and the data blob is already canonical.

For a codec requiring readable overrun, the codec-required slack follows the data blob inside the top-level record and is included in outer `stored_bytes`/stored CRC but not in `data_stored_bytes`.

## 13. Expert payload

An `EXPERT` record is a compound envelope. The outer record descriptor MUST use:

```text
kind         = EXPERT
codec        = NONE
math_format  = MIXED
scale_format = MIXED
layout       = MIXED
codec_table_id = 0
layer >= 0
expert >= 0
```

Compression is selected independently for each matrix weight and scale subblob.

### `ColiExpertHeader` (64 bytes)

| Offset | Bytes | Type | Field |
|---:|---:|---|---|
| 0 | 8 | bytes | `COLIEXPT` |
| 8 | 2 | u16 | `version_major` = 1 |
| 10 | 2 | u16 | `version_minor` = 0 |
| 12 | 4 | u32 | `header_bytes` = 64 |
| 16 | 4 | i32 | `layer` |
| 20 | 4 | i32 | `expert` |
| 24 | 2 | u16 | `matrix_count` = 3 |
| 26 | 2 | u16 | `flags` (zero in v1.0) |
| 28 | 4 | u32 | `matrix_desc_bytes` = 128 |
| 32 | 8 | u64 | `matrix_table_offset` = 64 |
| 40 | 8 | u64 | `data_offset` (>= 448, 16-byte aligned) |
| 48 | 8 | u64 | `logical_bytes` |
| 56 | 8 | u64 | `reserved` (zero) |

The three matrix descriptors appear in semantic order `GATE`, `UP`, `DOWN`.

### `ColiExpertMatrixDesc` (128 bytes)

| Offset | Bytes | Type | Field |
|---:|---:|---|---|
| 0 | 2 | u16 | `role` |
| 2 | 2 | u16 | `flags` (zero in v1.0) |
| 4 | 2 | u16 | `math_format` |
| 6 | 2 | u16 | `scale_format` |
| 8 | 2 | u16 | `weight_codec` |
| 10 | 2 | u16 | `scale_codec` |
| 12 | 2 | u16 | `layout` |
| 14 | 2 | u16 | `reserved0` (zero) |
| 16 | 8 | u64 | `rows` |
| 24 | 8 | u64 | `columns` |
| 32 | 4 | u32 | `scale_block_rows` |
| 36 | 4 | u32 | `scale_block_columns` |
| 40 | 4 | u32 | `weight_codec_table_id` |
| 44 | 4 | u32 | `scale_codec_table_id` |
| 48 | 8 | u64 | `weight_offset` relative to expert envelope |
| 56 | 8 | u64 | `weight_stored_bytes` |
| 64 | 8 | u64 | `weight_decoded_bytes` |
| 72 | 8 | u64 | `scale_offset` relative to expert envelope; zero when absent |
| 80 | 8 | u64 | `scale_stored_bytes` |
| 88 | 8 | u64 | `scale_decoded_bytes` |
| 96 | 4 | u32 | `logical_crc32c` over decoded `weight || scale` |
| 100 | 4 | u32 | `reserved1` (zero) |
| 104 | 4 | u32 | `group_size` (`0` when not applicable) |
| 108 | 4 | u32 | `reserved2` (zero) |
| 112 | 16 | bytes | `reserved` (zero) |

Weight/scale subblob offsets are 16-byte aligned and may not overlap the header, descriptor table, or one another.

For absent scales: `scale_format=NONE`, `scale_codec=NONE`, `scale_codec_table_id=0`, and all scale offset/length/block fields are zero.

The outer expert `decoded_bytes` and `logical_bytes` both equal the sum, in GATE/UP/DOWN order, of each matrix's decoded weight bytes plus decoded scale bytes.

If the outer record has `HAS_LOGICAL_CRC32C`, its logical CRC covers this exact concatenation:

```text
gate.weight || gate.scale || up.weight || up.scale || down.weight || down.scale
```

and each matrix descriptor's logical CRC covers that matrix's `weight || scale` pair.

### Canonical MXFP4 expert representation

For `math_format=MXFP4_E2M1`, `scale_format=UE8M0`, `layout=CANONICAL`:

```text
scale_block_rows    = 1
scale_block_columns = 32
group_size          = 0
weight bytes/row    = ceil(columns / 2)
scale bytes/row     = ceil(columns / 32)
```

Weight bytes are row-major E2M1 packed two nibbles per byte, low nibble first. Scale bytes are row-major UE8M0, one byte per 32-column block. Lossless codecs MUST reconstruct these bytes exactly; numerical equivalence is insufficient.

## 14. rANS container contract

`RANS256_G0_NIBBLE` and `RANS256_G0_U8` are storage codecs, not mathematical formats and not public `QT.fmt` ordinals.

The codec-specific stream framing/table bytes are defined and tested independently by #6. CSF adds these requirements:

- stored lengths are explicit; no fixed compression ratio is inferred;
- every encoded blob has a referenced compatible codec table;
- every encoded blob begins at a 16-byte-aligned address;
- the container provides **64 readable bytes after the codec blob**;
- those slack bytes are zero in canonical output;
- slack bytes are excluded from the subblob's `*_stored_bytes`, but included in the containing top-level record's `stored_bytes` and stored CRC;
- compression streams never cross a top-level record boundary.

This permits the existing branch-free/SIMD rANS decoder over-read contract without page-boundary tricks or per-miss record-sized bounce allocations.

## 15. `portable-v1` profile

The profile string is exactly `portable-v1`.

Requirements:

- `record_alignment` is a power of two in `[4096, 1048576]`; the v1 compiler default is 4096;
- all required tensors/expert matrices use `layout=CANONICAL`;
- no host pointer, Objective-C object, Metal handle, CUDA object, page address, or compiler ABI layout appears on disk;
- byte-order is always little-endian;
- uncompressed logical data preserves source bytes exactly;
- lossless codecs decode to the same canonical logical bytes;
- a record may be read/decompressed independently of every other record except its explicitly referenced static codec table;
- native `ROWS16`/Apple/x86/CUDA layouts are not emitted as required portable-v1 records.

Native profiles use the same CSF v1 container but receive separately versioned layout contracts under #26.

## 16. Source fingerprint

The production compiler sets `SOURCE_FINGERPRINT_VALID` and computes SHA-256 over a deterministic source inventory. This is provenance/cache identity, not an authentication signature.

### Included files

The inventory contains:

1. every safetensors shard consumed by the compiler;
2. a safetensors index file when one is used;
3. every config/tokenizer/chat-template sidecar copied into the `.coli` package.

No timestamps, inode numbers, absolute paths, ownership, or filesystem metadata are included.

### Canonical path

Each inventory path is UTF-8, relative to the source model root, with `/` separators. `.` and `..` path components are forbidden. Sort entries by raw UTF-8 path bytes; ties are impossible and are rejected.

### Canonical hash stream

For each file, first compute `SHA256(file_bytes)` while the compiler is already streaming the source. Then hash this root stream:

```text
ASCII "COLI-SOURCE-V1\0"
u32 entry_count
for each sorted entry:
    u8  kind              # 1=safetensors shard, 2=index, 3=copied sidecar
    u32 path_bytes
    u8[path_bytes] path
    u64 file_bytes
    u8[32] content_sha256
```

All integers in the root stream are little-endian. The final SHA-256 digest is the 32-byte `source_fingerprint` copied verbatim into every data-shard header.

Hashing may be folded into the compiler's normal streaming pass; no second full-model read is required.

## 17. Required parser/refusal behavior

A strict reader validates in this order before exposing records:

1. manifest magic/version/header size/byte-order tag;
2. known required manifest flags;
3. checked region arithmetic and non-overlap;
4. exact descriptor-table byte counts;
5. manifest CRC32C;
6. string descriptors and UTF-8/path constraints;
7. shard descriptor IDs/names;
8. each data-shard header CRC, ID, alignment, fingerprint, declared size and actual size;
9. record IDs, enum combinations, sentinels, shard IDs, aligned spans, checked end offsets and duplicate indexes;
10. codec-table bounds/CRC/reference compatibility;
11. type-specific envelope framing when a record is read or deep-validated;
12. stored CRC before publishing decoded data;
13. logical CRC after lossless decode when present.

Malformed input is rejected; there is no partial publication of an expert.

At minimum, reject by name/category:

```text
bad magic / unsupported major
unsupported required feature
integer overflow
manifest region outside file / illegal overlap
bad manifest CRC
invalid UTF-8 or shard path traversal
bad shard header / size mismatch
invalid enum combination
zero/duplicate record ID
duplicate required name
duplicate (layer, expert)
record outside shard / overlapping record
invalid codec table reference
truncated typed envelope
subrange outside record / overlapping expert subrange
bad stored CRC
bad logical CRC
codec structural/truncation error
```

Readers may impose lower resource limits than the format's integer widths (record count, string bytes, decoded bytes, etc.) to prevent absurd allocation. Such limits must fail explicitly rather than wrap or truncate.

Normal package open MUST NOT scan multi-gigabyte record payloads just to validate the manifest. Stored/logical CRCs are checked on record load or by explicit verification tooling.

## 18. Record-kind invariants

### TENSOR

```text
name_string_id != UINT32_MAX
expert == -1
math_format != MIXED
scale_format != MIXED
layout != MIXED
```

`layer` may be `-1` or a nonnegative informational layer index.

### EXPERT

```text
name_string_id may be UINT32_MAX
layer >= 0
expert >= 0
codec == NONE
codec_table_id == 0
math_format == MIXED
scale_format == MIXED
layout == MIXED
```

Only one non-optional expert record for a `(layer, expert)` pair is allowed.

### BLOB

```text
math_format == NONE
scale_format == NONE
layout == CANONICAL
layer == -1
expert == -1
```

v1.0 BLOB uses `codec=NONE` and stores raw immutable bytes.

### LAYER_PACK_RESERVED

Not emitted by portable-v1 minor 0. Its numeric kind is reserved so later work does not collide, but its payload ABI is deliberately not frozen by #22.

## 19. Deterministic writer rules

For identical source, compiler ABI/profile/options and codec tables, canonical output is byte-identical:

- no timestamps in binary files;
- all reserved bytes and alignment/slack padding are zero;
- shard names are deterministic (`data-%05u.coli` initially);
- shard IDs are contiguous from zero;
- record IDs are deterministic, nonzero, and unique;
- records are emitted in stable order: global/static named tensors, layer-scoped tensors, experts by `(layer, expert)`, then blobs;
- string IDs are assigned deterministically by first use in that stable traversal;
- codec table IDs are deterministic;
- checksums are computed only after final bytes are known;
- package creation is atomic at the directory level as specified in #24.

## 20. Versioning

- Major version changes when any existing required field changes offset/size/meaning incompatibly.
- Major version 1 keeps the descriptor sizes and field offsets in this document stable.
- A minor version may consume reserved fields, add optional record kinds, or define extension semantics without changing existing offsets.
- A v1 reader may open a newer v1 minor only when every required feature bit/record interpretation it encounters is understood.
- Unknown required functionality is rejected explicitly; optional unknown records may be skipped only when marked optional.

## 21. Golden hand fixture

`c/tests/fixtures/csf-v1-tiny/` contains a manually constructed v1 package as pure hex plus reconstruction instructions. It is intentionally not generated by the future compiler, so #23 can use it as an independent parser oracle.

The fixture contains one `TENSOR` record named `tiny.weight`, shape `[1,1]`, `U8`, canonical layout, one logical byte `0x2a` at shard offset 4096.

Its expected checksums and SHA-256 digests are documented beside the fixture. Those bytes are part of the v1 compatibility contract and must not be silently regenerated to accommodate a breaking parser/writer change.

## 22. Scope intentionally deferred

The following are not frozen by v1.0 and must not block the first `.coli` development path:

- `LAYER_PACK` payload ABI;
- Apple/x86/CUDA native execution-layout byte contracts (#26);
- direct-to-native decode details (#26);
- lossy 3-bit mathematical format details (#14);
- runtime KV/prompt-cache serialization;
- signing/authentication;
- network streaming protocol.

The first useful path is deliberately simpler:

```text
safetensors
  -> #24 compiler
  -> portable-v1 .coli
  -> #23 strict reader
  -> #25 V4 source
  -> #6 lossless expert codec when ready
```

# Colibri target profiles / execution layouts v1

Status: **initial target-profile ABI for issue #26**.

This document assigns the first concrete execution-layout IDs inside the ranges
reserved by CSF v1.1. These IDs describe the **final resident bytes consumed by
target kernels**. They are not source safetensors formats and they are not
storage-codec IDs.

The central rule is:

```text
source semantic tensor
  -> colic target lowering
  -> [optional storage codec]
  -> exact resident layout defined here
  -> target kernel ABI
```

A storage codec such as `RANS256_G0_NIBBLE` changes stored bytes only. After
decode, the resident bytes MUST be exactly the same layout that an uncompressed
record with the same execution-layout ID would expose.

## 1. ABI versions

```text
TARGET_PROFILE_ABI_V1    = 1
EXECUTION_LAYOUT_ABI_V1  = 1
KERNEL_ABI_V1            = 1
```

A runtime must match the named target profile and these ABI fields before any
required execution record using the layouts below is published.

## 2. Concrete layout IDs

### Apple / Metal range

```text
0x0101  APPLE_LINEAR_ROW_MAJOR_V1
0x0102  APPLE_MXFP4_ROW32_V1
```

### Linux x86_64 CPU range

```text
0x0201  X86_LINEAR_ROW_MAJOR_V1
0x0202  X86_MXFP4_ROW32_V1
```

### CUDA range

```text
0x0301  CUDA_LINEAR_ROW_MAJOR_V1
0x0302  CUDA_MXFP4_ROW32_V1
```

The byte geometry of the CUDA layouts is frozen here so `colic`, artifact
identity, and the loader have stable target IDs. **This document does not claim
that a production CUDA kernel consuming those IDs is already present in the
repository.** A CUDA runtime may advertise `KERNEL_ABI_V1` for these layouts
only after the corresponding kernel path is implemented and tested. Until then,
CSF target compatibility rejects that artifact/runtime combination rather than
falling back to a portable interpretation.

The Apple IDs map directly onto the current Metal MXFP4 contract. The x86 IDs
map onto the CPU/reference resident representation and provide a distinct target
identity even when some physical bytes happen to match Apple/CUDA v1.

## 3. `*_LINEAR_ROW_MAJOR_V1`

This layout is for unscaled dense/static tensors whose semantic data is a
contiguous row-major tensor.

Allowed mathematical formats in ABI v1:

```text
F32
F16
BF16
I8
U8
I16
U16
I32
U32
I64
U64
BOOL
FP8_E4M3FN
FP8_E5M2
```

Required scale format:

```text
NONE
```

The resident byte stream is exactly the row-major logical element stream using
the CSF mathematical format encoding. There is no per-row header, no hidden
stride and no source-checkpoint framing.

For shape `dims[0..rank-1]`:

```text
resident_bytes = product(dims) * element_bytes(math_format)
```

Rank-0 scalar tensors contain exactly one element. Zero logical dimensions are
not valid in execution records.

Target-specific IDs remain distinct even though the v1 byte order is identical.
That separation is deliberate: future Apple/x86/CUDA ABIs can evolve
independently without turning byte coincidence into a cross-target fallback
contract.

### Alignment

The start of the top-level record obeys the profile record alignment. The data
subblob is at least 16-byte aligned within the record. A compiler may use
stronger target alignment, but changing deterministic padding policy requires a
new target/optimization identity so artifact fingerprints change.

## 4. `*_MXFP4_ROW32_V1`

This is the first routed-expert matrix resident layout.

Required semantics:

```text
math_format        = MXFP4_E2M1
scale_format       = UE8M0
scale_block_rows   = 1
scale_block_columns= 32
group_size         = 0
```

Logical matrix shape is `[rows, columns]`.

### Weight plane

Each logical weight is one E2M1 nibble. Nibbles use the existing Colibri/MXFP4
encoding:

```text
0x0  +0
0x1  +0.5
0x2  +1
0x3  +1.5
0x4  +2
0x5  +3
0x6  +4
0x7  +6
0x8  -0
0x9  -0.5
0xa  -1
0xb  -1.5
0xc  -2
0xd  -3
0xe  -4
0xf  -6
```

Two logical values are packed per byte, **even column in the low nibble, odd
column in the high nibble**.

Rows are contiguous and have no hidden padding:

```text
weight_row_bytes      = ceil(columns / 2)
weight_resident_bytes = rows * weight_row_bytes
```

For odd `columns`, the unused high nibble of the final byte in every row MUST be
zero in deterministic compiler output and MUST be ignored by the kernel.

### UE8M0 scale plane

There is one unsigned E8M0 exponent byte for each 32-column block in each row.
Rows are contiguous:

```text
scale_row_bytes      = ceil(columns / 32)
scale_resident_bytes = rows * scale_row_bytes
```

Scale byte `e` represents the exact power-of-two scale whose IEEE-754 binary32
bit pattern is:

```text
uint32_bits = e << 23
```

This deliberately preserves the existing engine/Metal interpretation, including
subnormal/zero behavior implied by that bit construction. `colic` must not
replace it with a numerically similar floating representation.

For a logical column `c`:

```text
scale_index = row * ceil(columns / 32) + floor(c / 32)
```

### Expert-record placement

Weight and scale are distinct matrix subblobs in `ColiExpertMatrixDesc`.

Requirements:

- each subblob begins at a 16-byte-aligned record-relative offset;
- the two resident subblobs do not overlap;
- `weight_resident_bytes` and `scale_resident_bytes` equal the formulas above;
- `record.resident_bytes` is the sum of the six GATE/UP/DOWN resident subblob
  lengths, excluding envelope bytes and alignment/storage slack;
- storage codec slack/padding is never part of resident bytes;
- for codec `NONE`, stored bytes for a subblob equal resident bytes;
- after a lossless codec decode, produced bytes must exactly match this resident
  representation, including nibble order and E8M0 bytes.

### Why this is a target layout even when bytes match source MXFP4

The physical bytes happen to align with the current native MXFP4 checkpoint
geometry. The ABI does **not** promise source passthrough. `colic` owns the
lowering and must validate/rewrite input when source framing, padding, nibble
order, or scale geometry differs. Runtime code consumes only this target ID.

## 5. Apple M2-first profile

Canonical profile name:

```text
macos-arm64-metal-apple8-v1
```

Target descriptor requirements:

```text
target_os                  = MACOS
target_arch                = ARM64
backend                    = METAL
gpu_kind                   = APPLE_GPU_FAMILY
cpu_feature_mask           = ARM64_ASIMD
gpu_family_min             = 8
gpu_family_max             = 0
target_profile_abi         = 1
execution_layout_abi       = 1
kernel_abi_min             = 1
kernel_abi_max             = 1
record_alignment           = 16384
io_granularity             = 16384
resident_alignment         = 16384
required_runtime_features  = APPLE_UNIFIED_MEMORY | METAL_SHARED_STORAGE
```

Allowed required execution layouts in target-profile ABI v1:

```text
APPLE_LINEAR_ROW_MAJOR_V1
APPLE_MXFP4_ROW32_V1
```

The MXFP4 layout is intentionally the representation already consumed by the
current `fmt=7` Metal path: packed E2M1 row-major weights plus a separate
row-major UE8M0 scale byte stream. The profile therefore permits final expert
slots to be registered/wrapped directly by Metal without a record-sized runtime
repack.

The profile requires final resident-slot allocation to honor at least 16 KiB
alignment. An implementation may expose both CPU and GPU access to unified
memory, but the target descriptor—not an implicit macOS assumption—declares
which accessibility flags are required.

## 6. Linux x86_64 CPU profile

Canonical profile name:

```text
linux-x86_64-cpu-avx2-v1
```

Requirements:

```text
target_os                  = LINUX
target_arch                = X86_64
backend                    = CPU
gpu_kind                   = NONE
cpu_feature_mask           = X86_AVX2 | X86_FMA
target_profile_abi         = 1
execution_layout_abi       = 1
kernel_abi_min             = 1
kernel_abi_max             = 1
record_alignment           = 4096
io_granularity             = 4096
resident_alignment         = 64
required_runtime_features  = 0
```

Allowed required layouts:

```text
X86_LINEAR_ROW_MAJOR_V1
X86_MXFP4_ROW32_V1
```

`X86_MXFP4_ROW32_V1` uses the same nibble/scale geometry specified in section 4
but remains a separate target ID. Scalar/reference execution can consume it;
AVX2 kernels may vectorize that same ABI without changing compiled artifacts.

## 7. Linux x86_64 CUDA profile class

Canonical profile name:

```text
linux-x86_64-cuda-v1
```

The target descriptor carries the concrete CUDA compute-capability range used
for the artifact. `colic --target ...` may intentionally pin both min and max to
a single capability (for example 89) when the emitted kernel/profile is exact
architecture-specific.

Base requirements:

```text
target_os                  = LINUX
target_arch                = X86_64
backend                    = CUDA
gpu_kind                   = CUDA_SM
cpu_feature_mask           = X86_AVX2 | X86_FMA
target_profile_abi         = 1
execution_layout_abi       = 1
kernel_abi_min             = 1
kernel_abi_max             = 1
record_alignment           = 4096
io_granularity             = 4096
resident_alignment         = 256
required_runtime_features  = CUDA_RUNTIME | CUDA_ASYNC_COPY |
                             HOST_PINNED_STAGING
```

Allowed execution-layout IDs once the runtime advertises kernel ABI v1:

```text
CUDA_LINEAR_ROW_MAJOR_V1
CUDA_MXFP4_ROW32_V1
```

The CUDA MXFP4 byte geometry is section 4. The runtime is **not** allowed to
claim kernel ABI v1 merely because it can copy those bytes to a GPU. It must
have target kernels that interpret these layout IDs directly. Until that path
is implemented, CUDA target artifacts fail compatibility rather than falling
back through x86/Apple/canonical layouts.

## 8. Profile/layout validation matrix

A strict loader applies all of the following before publishing a required
execution record:

```text
profile Apple8:
  TENSOR/LAYER_PACK layout in 0x0100..0x01ff
  EXPERT inner matrix layout in 0x0100..0x01ff

profile x86 CPU:
  TENSOR/LAYER_PACK layout in 0x0200..0x02ff
  EXPERT inner matrix layout in 0x0200..0x02ff

profile CUDA:
  TENSOR/LAYER_PACK layout in 0x0300..0x03ff
  EXPERT inner matrix layout in 0x0300..0x03ff
```

Within the range, the ID must also be one of the concrete IDs registered by
this ABI version. A reserved unknown ID is unsupported, not future-compatible
by default.

BLOB records use layout `NONE` and are exempt from target layout range checks.
Compound EXPERT outer descriptors use `MIXED`; each matrix carries a concrete
target layout.

## 9. `colic` lowering requirements

For target-profile ABI v1, `colic` must lower semantic IR as follows:

```text
unscaled contiguous tensor
  -> TARGET_LINEAR_ROW_MAJOR_V1

routed expert matrix with exact MXFP4 semantics
  -> TARGET_MXFP4_ROW32_V1
```

The target prefix is Apple/x86/CUDA according to the selected target profile.

Compiler validation includes:

- logical shape/element count;
- exact mathematical and scale semantics;
- MXFP4 nibble order;
- one UE8M0 scale byte per 32 logical columns and row;
- deterministic odd-column high nibble zeroing;
- target subblob alignment;
- exact resident-byte formulas;
- optional storage codec decodes to the same resident bytes;
- target/profile/layout/kernel ABI identity participates in artifact identity.

No runtime source-name lookup, tensor repack or portable-layout conversion is a
valid substitute for target lowering.

## 10. Kernel ABI boundary

`KERNEL_ABI_V1` means the runtime kernel entrypoint accepts final resident bytes
with the exact layout semantics above and does not require a model-record-sized
conversion first.

Backend-specific launch metadata (threadgroup shape, CUDA block shape, command
buffer state, pointers/handles) is runtime state and is not serialized in CSF.
If a future kernel needs a different persistent resident arrangement, that is a
new execution-layout ID and usually a new execution-layout ABI revision.

A pure kernel scheduling/tuning change that preserves the resident byte
contract may keep the layout ID, but deterministic target artifacts whose
compiler output changes must still change optimization/profile/tuning identity
per the CSF artifact-fingerprint rules.

## 11. What remains implementation work

This ABI is sufficient for:

- #23 to reject target records outside the selected profile's concrete layout
  registry;
- #24 to lower semantic IR into Apple/x86/CUDA resident bytes;
- #6 to losslessly decode directly into `*_MXFP4_ROW32_V1` slots;
- #25 to execute `.coli` without reconstructing safetensors tensors.

The Apple and x86 resident representations already map to current executable
paths. CUDA layout IDs are now stable, but advertising CUDA kernel ABI v1 is
blocked on an actual direct-consuming CUDA kernel implementation and tests.

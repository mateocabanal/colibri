# Colibri Engineering Handoff: rANS Storage, Model-Agnostic MetalIO, and Apple8 Lowering in `colic`

**Date:** 2026-08-20  
**Repository:** `mateocabanal/colibri`  
**Primary baseline:** `main` at `9b20e11522b6b285db442cc95be1a778b1d2b749` (`bench(apple): compare Apple8 4-bit execution formats`)  
**Active Apple8 engine experiment:** `feat/wire-tile-engine`, latest known head `5f0dd0b04a9213280f5dc9302a45fdcb1ec81ae7`  
**Audience:** next primary programmer/architect working on the `.coli` compiler/runtime and Apple-Silicon disk-streamed MoE path.

---

## 0. Executive summary

There are three major remaining issues that should be treated as **one end-to-end data-path project**, not three independent optimizations:

1. **Lossless storage encoding (rANS)** — reduce routed-expert bytes on disk and therefore physical NVMe traffic without changing a single decoded quantization bit.
2. **Model-agnostic MetalIO backend** — stop making individual model engines own Apple-specific asynchronous I/O logic; give the `.coli` executor/residency stack a reusable asynchronous physical-record loader that can use MetalIO on Apple Silicon and `pread()` as the portable fallback.
3. **Apple8 tile format emitted directly by `colic`** — when compiling for `macos-arm64-metal-apple8-v1`, emit the exact MXFP4 Apple8 execution representation selected by the Apple format benchmark, rather than emitting canonical row MXFP4 and paying a runtime repack.

The intended final path is:

```text
source checkpoint
      |
      v
    colic
      |
      | semantic/model frontend
      | MXFP4 quantization or exact MXFP4 ingest
      | Apple8 physical lowering
      | optional lossless rANS storage codec
      v
 target-specific .coli package
      |
      | record locator (shard/path + offset + stored/decoded sizes + codec)
      v
 model-neutral async I/O service
      |
      +--------------------------+
      | codec = NONE             | codec = rANS
      v                          v
 MetalIO -> final shared     MetalIO -> compressed staging
 MTLBuffer slot                  |
      |                          v
      |                    CPU rANS decode
      |                          |
      |                          v
      +------------------> final shared Apple8 MTLBuffer
                                 |
                                 v
                         Apple8 Metal expert kernel
```

The important architectural separation is:

- **quantization/math representation** answers “what values do these bytes mean?”;
- **execution layout** answers “how are the exact values physically arranged for the kernel?”;
- **storage codec** answers “how are those execution bytes compressed on disk?”;
- **I/O backend** answers “how do stored bytes move from a shard into memory?”;
- **residency/dispatch** answers “which ready representation is resident and safe to execute now?”

Do not merge these identities. In particular, `ColiRepresentationId` deliberately excludes storage codecs. That is correct and must remain true.

### Desired steady-state fast paths

**Uncompressed Apple8 package:**

```text
NVMe -> MTLIOCommandQueue -> persistent MTLBuffer (Shared) -> Apple8 kernel
```

No `pread()`, no row-format slab, no `newBufferWithBytesNoCopy` registration churn, and no row→tile repack.

**rANS-compressed Apple8 package:**

```text
NVMe -> MTLIO compressed staging -> rANS decode -> persistent Apple8 MTLBuffer -> kernel
```

The I/O, CPU decode, and GPU execution must be pipelined so token time tends toward:

```text
max(storage I/O, decompression, GPU expert compute)
```

rather than:

```text
storage I/O + decompression + GPU expert compute
```

The existing Qwen MetalIO experiment already demonstrated that **demand-only MetalIO is nearly neutral** while exact asynchronous issue and arena-wave pipelining are materially faster. Do not build a generic backend that immediately waits after each submit and then declare MetalIO ineffective.

---

# 1. Current repository state and relevant contracts

## 1.1 `.coli`/CSF already has most of the vocabulary required

`c/coli_format.h` already defines:

- target profiles including `macos-arm64-metal-apple8-v1`;
- record-level `codec`, `math_format`, `scale_format`, `layout`;
- `stored_bytes` and `decoded_bytes`;
- record and logical CRCs;
- `codec_table_id`;
- codec IDs:
  - `COLI_CSF_CODEC_NONE`
  - `COLI_CSF_CODEC_RANS256_G0_NIBBLE`
  - `COLI_CSF_CODEC_RANS256_G0_U8`
- `ColiExpertMatrixInfo`, which also has per-matrix `weight_codec`, `scale_codec`, codec table IDs, stored/decoded byte counts, and layout.

That means this project should **finish the existing format model**, not invent a second storage format beside CSF.

The generic package reader currently exposes model-neutral metadata and `pread()`-based range/record reads. It is intentionally thread-safe because reads use `pread`/`compat_pread`, but the physical read implementation is synchronous and CPU-mediated.

## 1.2 Representation identity already draws the correct codec boundary

`c/expert_representation.h` is especially important. `ColiRepresentationId` contains:

- math format;
- scale format;
- execution layout;
- layout ABI;
- kernel ABI;
- target class;
- group and scale geometry;
- flags.

It explicitly **does not contain the storage codec**. Its comment says two differently framed/compressed records may decode to the same resident representation.

Keep this invariant:

```text
stored record identity != resident execution representation identity
```

Example:

```text
RANS256 compressed Apple8 MXFP4
             |
             | decode
             v
Apple8 MXFP4 resident representation
```

and:

```text
uncompressed Apple8 MXFP4
             |
             | direct load
             v
Apple8 MXFP4 resident representation
```

must converge on the **same** `ColiRepresentationId`.

## 1.3 `colic` target naming is ahead of its actual physical lowering

`colic/src/target/mod.rs` already has:

```text
macos-arm64-metal-apple8-v1
```

with Metal backend, 16 KiB record alignment, and 16 KiB preferred I/O granularity.

However, the current exact expert lowering explicitly preserves source matrix/scale bytes in source order, and `colic/src/quant/mxfp4_record.rs` emits canonical row MXFP4:

- packed E2M1 weight rows;
- separate E8M0 scale bytes;
- column group size 32.

So the profile currently **claims Apple8 without actually lowering routed MXFP4 experts to Apple8**. This is the main compiler correctness/identity hole that workstream 3 must close.

## 1.4 `colic` already has a codec CLI/request concept but rejects it

`CompileRequest` contains a `CodecRequest` with:

- `None`
- `Auto`
- named profile.

Today `validate_supported_options()` rejects anything except `--codec none` with “storage codecs are not implemented”. This is a useful seam: do not add an unrelated codec CLI.

## 1.5 Apple8 execution format is already selected experimentally

The format chosen by the Apple 4-bit benchmark is **MXFP4-Apple8-tile**:

```text
8 output rows x 32 K values per tile
136 bytes per tile
  128 bytes packed E2M1 values
    8 bytes colocated E8M0 scales (1 scale per output row for that K group)
```

It is a pure physical repack. There is **no dequantization and no requantization**.

For logical output row `o` and K-group `g`:

```text
otile = o >> 3
orow  = o & 7
ng    = ceil(columns / 32)
tile  = base + ((otile * ng + g) * 136)

packed E2M1 bytes for row o, group g:
    tile[orow * 16 .. orow * 16 + 16]

E8M0 scale for row o, group g:
    tile[128 + orow]
```

Partial output-row or K-column edge tiles are deterministically zero-padded.

The branch `feat/wire-tile-engine` contains a reference repacker/validator and an experimental transform registration. Its IDs are **fixture IDs only**; they explicitly are not production target/layout/kernel ABI assignments.

## 1.6 Existing MetalIO work is real and should be reused

`c/metalio.h/.mm` already implements:

- runtime-gated Apple Metal 3 `MTLIOCommandQueue`;
- persistent shared-storage `MTLBuffer` slots;
- file handles by shard path/URL;
- single and vectored asynchronous loads;
- event values and waits;
- demand/async/speculative accounting;
- reusable slots;
- fallback-compatible failure semantics;
- latency, queue-depth, wait, byte and prefetch metrics.

The Qwen engine contains actual integration, including exact asynchronous issue of selected experts and reusable arena-wave buffers.

Measured on real Qwen3.6-35B-A3B q8 on M2/16 GB:

| Path | Wall time |
|---|---:|
| `pread()` baseline | 15.0–16.7 s |
| demand-only MetalIO | ~15.5–16.0 s |
| + exact asynchronous decode issue | ~13.2 s |
| + asynchronous pipelined arena waves | **11.5–11.8 s** |

The lesson is not merely “MetalIO is faster”. The lesson is:

> **MetalIO pays when the scheduler permits overlap. A synchronous submit+wait translation of `pread()` is not enough.**

---

# 2. Global invariants — do not violate these

These invariants apply to all three workstreams.

## 2.1 Exactness

Lossless storage and Apple8 physical lowering must preserve exact underlying MXFP4 values and E8M0 scales.

For a source row MXFP4 representation `R` and Apple8 representation `A`:

```text
inverse_logical(A) == R
```

byte-for-byte for every logical E2M1 nibble and E8M0 scale.

For a codec `C`:

```text
decode(C(execution_bytes)) == execution_bytes
```

byte-for-byte.

No “numerically close” acceptance for either stage.

## 2.2 Compiler target bytes must be deterministic

Same source fingerprint + same compiler/profile inputs must emit the same artifact bytes.

If any option can change emitted target bytes, it must participate in the target/artifact identity as required by the CSF v1.1 target-identity amendment.

## 2.3 Codec and representation are orthogonal

Do not create representation IDs such as “Apple8-rANS”.

Correct:

```text
representation = MXFP4/E8M0/Apple8/g32/ABI...
codec          = RANS256_G0_NIBBLE
```

Incorrect:

```text
representation = APPLE8_RANS
```

## 2.4 Model engines do not own physical I/O policy

No new `deepseek_metalio_*`, `qwen_metalio_*`, or model-specific shard loader.

The model engine may say **which expert it needs**. The generic package/residency layer must know **where it is stored**, and the generic asynchronous I/O backend must know **how to move its bytes**.

## 2.5 MetalIO is optional

Every MetalIO initialization, file-handle, slot-allocation, submit, completion or decode failure must have a safe fallback where feasible.

The portable path remains valid.

## 2.6 Do not overwrite storage/resident slots that are still in use

There are two independent hazards:

1. an I/O operation completing into a slot that has already been recycled;
2. a new load overwriting a final MTLBuffer while a GPU command buffer still reads it.

Both require explicit generation/lease semantics.

## 2.7 Do not pay runtime target compilation when `colic` already knows the target

If a package is compiled specifically for Apple8, the normal path should not load canonical rows and then invoke #135 to create Apple8.

Runtime transforms are useful for compatibility/multi-variant residency, not as the default implementation of a target-specific compiler profile.

---

# 3. Workstream A — production lossless rANS storage encoding

## 3.1 Goal

Reduce the number of bytes stored and physically read for routed experts while preserving the exact decoded execution representation.

For disk-streamed MoE, this attacks two costs at once:

- package size;
- bytes that must traverse NVMe on a miss.

The existing `int4-rans256-g0` experiment achieved roughly **0.76 stored/original ratio**, i.e. about **24% fewer bytes**, on a full int4 GLM-5.2 corpus with byte-exact reconstruction.

That work already includes a substantial, hardened codec implementation and validation model. The task is to turn the codec into a first-class CSF storage stage rather than keep it as an offline safetensors-only experiment.

## 3.2 Existing rANS implementation

Relevant files:

```text
c/rans.h
c/tools/rans_format.py
c/tools/repack_rans.py
c/tools/rans_verify.py
c/tools/rans_ctypes.c
c/tests/test_rans.c
c/tests/fuzz_rans.c
c/tests/test_rans_repack.py
docs/int4-rans256-g0.md
docs/FORMATS.md
```

Important properties already established:

- static-table byte-renormalized rANS;
- 16-symbol nibble alphabet;
- 256 independent round-robin streams;
- deterministic encode;
- forward-order decode;
- stream framing and exact-consumption invariants;
- corruption/refusal checks;
- amplification/decompression-bomb bound;
- ARM/NEON support in the codec implementation;
- per-shard table concept in the old safetensors container experiment.

The interleave is particularly valuable: logical nibble `j` belongs to stream `j % 256`, which permits groups of decoder lanes to reconstruct contiguous output bytes.

## 3.3 Do not blindly transplant `int4-rans256-g0` as “the Apple8 format”

The old prototype is described in terms of **int4 weight tensors** and its `g0` table was trained for that corpus. Apple8 contains:

- packed E2M1 weight bytes; and
- E8M0 scale bytes embedded into each 136-byte physical tile.

Scale bytes can contain arbitrary high/low nibbles. An old table that assigns zero frequency to a symbol absent from the old int4 corpus is not necessarily capable of encoding arbitrary Apple8 bytes.

Therefore:

> Reuse the rANS **algorithm, framing, interleave, validation and implementation**, but make CSF codec tables artifact data rather than assuming the old int4 `g0` frequencies are universally valid.

The CSF format already has `codec_table_id`; use it.

## 3.4 Recommended v1 production codec strategy

Start with **generic nibble-rANS over the exact final execution byte sequence**.

Conceptually:

```text
Apple8 execution bytes
  -> split each byte low nibble, high nibble
  -> 256-way static-table rANS
  -> stored payload
```

On decode:

```text
stored rANS payload
  -> decode nibbles
  -> repack low/high nibble pairs
  -> exact original Apple8 byte stream
```

Advantages:

- works for arbitrary byte content, including scale bytes, if the artifact table covers all observed symbols;
- does not understand model semantics;
- does not understand Apple8 internals;
- can compress canonical row MXFP4, Apple8, int4, or other nibble-friendly physical streams;
- keeps codec below the layout layer;
- simplest correctness story: compare decoded bytes to the exact pre-codec payload.

Potential disadvantage:

- scale bytes are not necessarily nibble-compressible and may dilute the ratio slightly.

That is acceptable for v1. **Measure before complicating the format.**

### Possible v2 if data justifies it

Only if benchmarks show material benefit, consider a compound strategy:

- nibble-rANS weight region;
- raw or U8-rANS scale region;
- reconstruct the final Apple8 interleaving during decode.

Do not start there. It couples codec logic to the layout and creates extra metadata/decoder complexity.

## 3.5 Codec table ownership

The compiler should create deterministic codec tables from a well-defined corpus.

Recommended policy options, in preference order:

### Option A — one table per artifact/profile

Build histogram across every record selected for the same codec, normalize frequencies deterministically, then embed one codec table in `manifest.coli` and reference its ID from records.

Pros:

- table overhead paid once;
- deterministic;
- good aggregate statistics;
- simple runtime cache.

### Option B — one table per shard

Matches the old experiment more closely but complicates the CSF manifest and can make deterministic shard planning depend on compressed sizes/table choice.

### Option C — one table per record

Reject for now. Better compression may not justify metadata, cache, planning and validation overhead.

**Recommendation:** artifact-wide table first unless a current CSF normative rule already mandates a different scope. Keep the API capable of multiple table IDs.

## 3.6 Deterministic two-pass compilation issue

Compression changes `stored_bytes`, and storage planning needs `stored_bytes` before assigning shard offsets.

This creates an unavoidable planning question:

```text
need compressed size -> to plan record placement
need encoded record -> to know compressed size
```

Do not guess compressed sizes for final planning.

Recommended compiler flow for compressed pageable experts:

```text
Pass 1: target lowering + codec census/size pass
    - produce or stream target execution bytes
    - accumulate table histogram
    - if table not known yet, either:
        a) perform histogram-only pass first, then encode-size pass, or
        b) spool lowered records temporarily

Freeze codec table

Pass 2: encode each lowered record deterministically
    - compute exact stored size
    - create StoragePlan

Pass 3 / emission:
    - emit encoded records to planned offsets
    - calculate stored CRC
    - retain logical/decoded CRC
```

For very large models, do not retain all lowered experts in RAM. Prefer bounded buffers or deterministic temporary spool files.

A practical implementation can reduce passes by:

1. histogram source/final bytes using bounded streaming;
2. freeze table;
3. encode each expert to temporary per-record/spool storage while collecting exact sizes;
4. plan shards;
5. copy encoded spool records into final `.coli` shards.

Disk space is cheaper than accidentally requiring hundreds of GiB of RAM.

## 3.7 What exactly gets checksummed

Keep two integrity domains distinct:

### Stored CRC

CRC over the exact compressed bytes present in the data shard.

Catches on-disk corruption before/while decoding.

### Logical/decoded CRC

CRC over the exact execution bytes after codec decode.

For an Apple8 expert this should be over the decoded Apple8 physical bytes, not over a reconstructed canonical row representation.

This makes validation straightforward:

```text
read stored bytes
 -> verify stored CRC if policy requests
 -> decode
 -> verify decoded/logical CRC
 -> publish resident representation
```

## 3.8 Runtime decoder API

Do not make the runtime know about the old safetensors stamping scheme.

Create a small CSF codec layer, for example:

```c
/* names illustrative */
typedef struct ColiCodecTableView ColiCodecTableView;

typedef struct {
    uint16_t codec;
    uint32_t table_id;
    uint64_t stored_bytes;
    uint64_t decoded_bytes;
} ColiCodecDesc;

int coli_codec_decode(
    const ColiCodecDesc *desc,
    const ColiCodecTableView *table,
    const void *src, size_t src_bytes,
    void *dst, size_t dst_bytes,
    char *error, size_t error_size);
```

Properties:

- no model inputs;
- no layer/expert IDs needed;
- no Metal types;
- exact destination size required;
- fail closed on malformed framing/table/symbol count;
- can select scalar/NEON implementation internally.

Longer term an asynchronous executor can wrap this synchronous primitive in a decode worker pool.

## 3.9 Integration with `ColiExpertMatrixInfo`

Be careful about codec scope.

CSF currently has both:

- top-level record `codec`;
- per-matrix weight/scale codecs.

Choose one level for the first production implementation and document it normatively.

**Recommended first implementation for Apple8:** top-level record codec over the target-specific expert envelope/payload **if and only if** this keeps the envelope parseable or the record reader knows it must decode before typed-envelope parsing.

If CSF requires typed `COLIEXPT` headers to remain directly readable, use an uncompressed envelope header with individually compressed matrix payloads.

The important requirement is that generic metadata needed to schedule I/O must be available **without first decoding the whole record**.

The scheduler must know at minimum:

```text
shard/path
payload offset
stored bytes
decoded bytes
codec
table id
resident representation/layout
```

before issuing the read.

## 3.10 Decompression placement with MetalIO

Compressed bytes must generally land in a **staging buffer**, not the final execution slot, because decoded output is larger.

Recommended pipeline:

```text
persistent compressed staging slot
        |
        | MetalIO fills asynchronously
        v
CPU decoder job
        |
        | writes exact decoded bytes
        v
persistent final Shared MTLBuffer
        |
        v
GPU execution lease
```

Avoid decode-in-place unless a codec-specific proof and benchmark justify it.

## 3.11 NEON decoder optimization target

The first performance goal is not “maximum GB/s in a synthetic decoder”. It is:

> decompression should be hidden behind I/O/GPU work often enough that it does not become the new critical path.

Measure:

- decoded GiB/s;
- CPU time per expert;
- wall decode latency per expert;
- percentage of decode work overlapped with MetalIO/GPU;
- final token throughput.

If scalar decode already hides under I/O, further NEON work is lower priority. If decode becomes critical, optimize the existing 256-way interleave around Apple M2/M-series NEON.

## 3.12 Security/corruption requirements

Carry forward the strong refusal model from `rans.h` / the current spec:

- checked arithmetic for all sizes/offsets;
- table validation before allocation;
- symbol-count amplification bound;
- exact stream consumption;
- valid initial/final rANS states;
- every stream offset in range and monotonic;
- no silent trailing garbage where the format forbids it;
- decoded length must exactly equal manifest `decoded_bytes`;
- logical CRC must match before publish.

Do not allow malformed package metadata to request unbounded allocations.

## 3.13 Compiler CLI semantics

Suggested behavior:

```text
--codec none
    always emit raw target execution bytes

--codec rans256-g0-nibble
    force codec; fail if unsupported for selected records/profile

--codec auto
    evaluate eligible records; use codec only when it clears a deterministic
    minimum savings threshold
```

For `auto`, the threshold must be part of a named storage/optimization profile or fingerprinted profile data. It cannot be an invisible heuristic that changes artifact bytes without identity changes.

## 3.14 rANS acceptance tests

Minimum test matrix:

### Codec unit tests

- empty/invalid records rejected per contract;
- random byte payload round trip;
- every nibble value 0–15 represented;
- pathological single-symbol payload;
- short records smaller than 256 streams;
- non-multiple-of-2 nibble count if format permits it;
- corruption in header/table/offset/state/payload;
- deterministic encoding.

### Apple8 composition tests

For random canonical MXFP4 matrices including partial tiles:

```text
row source
 -> Apple8 pack
 -> rANS encode
 -> rANS decode
 -> compare exact Apple8 bytes
 -> Apple8 inverse validator against row source
```

Both comparisons must pass.

### Package tests

- compile same fixture twice, package SHA identical;
- open package and validate codec table references;
- read/decode every expert;
- stored CRC corruption refused;
- decoded/logical CRC corruption refused;
- `--codec none` package remains accepted by unchanged portable path.

### Full-model gate

On a real V4/Qwen artifact:

- package size ratio;
- total routed-expert bytes ratio;
- exact token IDs against uncompressed same-layout package;
- best-of-3 warm decode tok/s;
- CPU decompression share;
- MetalIO read bytes reduction.

## 3.15 rANS success criteria

Do not merge the optimization as default solely because the package gets smaller.

For disk-streamed MoE, production success should require:

1. **byte-exact decoded payloads**;
2. **token-identical generation**;
3. meaningful routed-expert stored-byte reduction;
4. no pathological decode CPU bottleneck;
5. neutral or better end-to-end warm decode throughput;
6. ideally measurable improvement in cold/miss-heavy workloads.

---

# 4. Workstream B — model-agnostic asynchronous I/O with MetalIO backend

## 4.1 Goal

Move the proven Apple MetalIO mechanism out of Qwen-specific scheduling/loading code and make it usable by any `.coli` model/runtime path.

The abstraction should be **model-agnostic**, not necessarily platform-agnostic internally:

```text
model / expert store
       |
       v
 generic .coli record/residency loader
       |
       v
 async I/O interface
       |
       +----------------+
       |                |
       v                v
 MetalIO backend     portable pread backend
```

Later, the same upper abstraction could support `io_uring`, Windows async I/O, etc., but do not let hypothetical portability delay the Apple implementation.

## 4.2 Why current `pread()` is not the desired Apple end state

Current generic package reads are approximately:

```text
NVMe
 -> pread()
 -> host allocation/slab
 -> possibly register/wrap as MTLBuffer
 -> GPU kernel
```

On unified memory there is no discrete PCIe upload, but there are still:

- synchronous CPU read calls on the critical path;
- CPU orchestration;
- slab/slot bookkeeping;
- registration/wrapper lifetime work;
- lost opportunity to queue multiple reads while useful GPU/CPU work continues.

MetalIO’s value is primarily **asynchrony + persistent GPU-visible destinations**, not magical higher raw SSD bandwidth.

## 4.3 Existing Qwen implementation is the behavioral reference, not the final architecture

Do not discard it. Extract the useful mechanisms:

- persistent `MTLBuffer` slots;
- `metalio_loadv()`;
- event-valued asynchronous submission;
- demand/async distinction;
- issue several exact selected expert loads before waiting;
- arena-wave reuse;
- metrics;
- failure fallback.

But remove these model-specific assumptions from the I/O layer:

- Qwen tensor names;
- Qwen router history;
- layer-lookahead predictor;
- Qwen cache structures;
- Qwen-specific env variables.

The generic service should never inspect a model architecture or tensor name.

## 4.4 Required package-reader addition: physical record locator

The generic I/O backend needs a stable way to identify a physical source span.

`ColiRecordInfo` already contains:

- `shard_id`;
- `payload_offset`;
- `stored_bytes`.

But MetalIO’s current SDK integration opens files by **path/URL**, not by the POSIX fd used by `pread()`.

Add a package-level accessor or internal executor API that can resolve:

```c
typedef struct {
    uint32_t shard_id;
    const char *path;      /* package-owned, stable while package is open */
    uint64_t offset;
    uint64_t stored_bytes;
} ColiPhysicalSpan;
```

Do not expose model semantics.

Do not force the MetalIO backend to reconstruct paths from package-directory naming conventions.

## 4.5 Proposed asynchronous I/O API

Names are illustrative. The important part is the contract.

```c
typedef uint64_t ColiIoTicket;
typedef uint32_t ColiIoSlotId;

typedef enum {
    COLI_IO_DEMAND,
    COLI_IO_ASYNC,
    COLI_IO_SPECULATIVE
} ColiIoKind;

typedef struct {
    uint32_t shard_id;
    const char *path;
    uint64_t src_offset;
    uint64_t bytes;
    uint64_t dst_offset;
} ColiIoRegion;

typedef struct {
    void *ptr;
    uint64_t bytes;
    uint64_t generation;
    /* backend-private handle */
} ColiIoSlotView;

int coli_io_backend_open(...);
int coli_io_slot_alloc(...);
int coli_io_slot_free(...);
int coli_io_submitv(..., const ColiIoRegion *, size_t count,
                    ColiIoSlotId dst, uint64_t generation,
                    ColiIoKind kind, ColiIoTicket *ticket);
int coli_io_poll(..., ColiIoTicket ticket, int *done);
int coli_io_wait(..., ColiIoTicket ticket);
void *coli_io_slot_ptr(..., ColiIoSlotId slot);
```

Backend selection:

- Apple + Metal + supported runtime -> MetalIO;
- otherwise -> portable backend.

The portable backend may initially execute synchronously while conforming to the same completion contract. Do not fake asynchronous behavior if it is not present.

## 4.6 Destination slots versus resident slots

Keep two concepts distinct because compression requires it.

### I/O/staging slot

A byte buffer the storage backend may write.

For raw uncompressed Apple8 this can also be the final resident execution slot.

### Resident execution slot

A buffer containing fully decoded, validated bytes in a known `ColiRepresentationId` and safe for backend execution.

For compressed records:

```text
I/O slot != resident slot
```

for at least the first implementation.

## 4.7 Ticket + generation is non-negotiable

A slot ID alone is unsafe.

Scenario:

```text
submit expert A -> slot 7
slot 7 evicted/reassigned to expert B
old A I/O completes late -> writes/publishes stale A into B's slot
```

Every submission must capture a monotonically changing slot generation.

Completion/publication rule:

```text
if completion.slot_generation != current_slot_generation:
    discard completion / never publish it
```

The same principle already exists elsewhere in the residency stack and should be unified rather than reinvented.

## 4.8 GPU-use lifetime is a separate lease

Even after I/O is complete and the slot generation matches, the system must not overwrite a shared MTLBuffer that an in-flight Metal command buffer still reads.

Need one of:

- residency lease held until GPU completion;
- backend completion callback decrements use count;
- shared-event/fence sequencing proving new I/O cannot begin until prior GPU read completes.

The clean long-term design is for residency to own a physical slot and expose an execution lease. Slot reuse requires:

```text
I/O complete
AND decode complete
AND no active execution lease
```

## 4.9 Raw fast path

For `codec == NONE` and a package record already in the final executable Apple8 representation:

```text
1. residency allocates/reuses final MetalIO-backed Shared slot
2. issue MTLIO load from shard payload directly into the final slot
3. do useful work / issue other exact loads
4. wait only at first true dependency
5. validate if required
6. publish representation resident
7. execute Apple8 kernel using that same buffer
```

There should be no intermediate host copy.

## 4.10 Compressed fast path

For rANS:

```text
1. allocate/reuse compressed staging slot sized to stored_bytes
2. allocate/reuse final Shared execution slot sized to decoded_bytes
3. submit MetalIO read into staging
4. continue useful work
5. when I/O event completes, schedule CPU decode
6. decoder writes directly into final Shared slot
7. verify logical CRC / exact decoded size
8. publish final representation
9. GPU kernel reads final slot
```

A bounded decode-worker pool is preferable to launching arbitrary threads.

On Apple Silicon, `MTLStorageModeShared` makes the final buffer CPU-writable and GPU-visible, which is exactly what this path needs.

## 4.11 Scheduling: exact async first, prediction later or never

The previous Qwen work already showed a route predictor with roughly 31–39% useful precision did not clear the chosen usefulness gate. Do not make prediction a prerequisite for generic MetalIO.

There is a simpler, exact source of overlap:

```text
router computes K selected experts
 -> immediately submit all missing selected expert reads
 -> process resident experts / other work
 -> wait for each selected miss only when its bytes are required
```

This is not speculation. Every submitted expert is known to be needed.

Then add layer/wave pipelining where dependency structure permits.

## 4.12 Interaction with #133–#137 residency stack

The generic I/O service should sit **under** residency, not beside it.

Conceptually:

```text
request representation R for expert E
              |
              v
      representation residency
              |
       hit? --+--> lease
              |
             miss
              v
      physical package source
              |
              v
       async I/O backend
              |
       codec decode if any
              |
              v
       resident slot publish
              |
              v
             lease
```

If a package stores canonical rows but the desired execution representation is Apple8, #135 transform may then create the derived representation. But for an Apple8-targeted artifact, that transform should not be needed.

## 4.13 Do not put MetalIO in `coli_format.c`

`coli_format.c` should remain the parser/validator for CSF.

It can expose physical source information, but it should not import Metal frameworks or own an `MTLIOCommandQueue`.

Suggested split:

```text
c/coli_format.*             parse/validate package metadata
c/coli_io.*                 generic async-I/O contract / backend selection
c/coli_io_pread.*           portable synchronous fallback
c/coli_io_metal.mm          MetalIO implementation or wrapper around metalio.mm
c/metalio.*                 low-level MetalIO primitive (can initially remain)
c/coli_executor.*           composes package, codec, residency, I/O
```

The exact filenames are flexible; the layering is not.

## 4.14 Migration path from current `metalio.*`

Do not rewrite a working MetalIO subsystem from scratch.

### Phase 1

Wrap existing `metalio.h/.mm` behind the new generic I/O interface.

### Phase 2

Move Qwen to the generic interface without changing behavior.

Acceptance gate:

- Qwen output identity unchanged;
- queue/wait statistics comparable;
- no meaningful regression against the 11.5–11.8 s pipelined behavior in a comparable run.

### Phase 3

Wire `.coli` executor / V4 expert store to generic I/O.

### Phase 4

Add codec staging/decode composition.

### Phase 5

Remove redundant Qwen-specific MetalIO plumbing once all paths use the shared service.

## 4.15 Metrics required

Expose metrics at layers that answer different questions.

### Physical I/O

- submitted loads;
- submitted bytes;
- completed bytes;
- failures/fallbacks;
- outstanding and peak queue depth;
- raw I/O latency distribution;
- wait calls;
- total actual wait time.

### Codec

- compressed bytes;
- decoded bytes;
- ratio;
- decode jobs;
- decode CPU time;
- decode wall latency;
- decode failures;
- staging bytes/peak staging residency.

### Residency

- hits/misses;
- evictions;
- stale completion drops;
- lease waits;
- useful/wasted async loads.

### End-to-end

- tok/s;
- expert-path wall share;
- GPU kernel time;
- percentage time actually blocked on storage;
- percentage decompression hidden by other work.

If possible, include an overlap metric such as:

```text
sum(component busy time) - wall critical-path time
```

or direct timeline tracing.

## 4.16 MetalIO test plan

### Unit/fixture

- file registration by path;
- aligned/unaligned legal spans;
- bounds rejection;
- vectored load exact bytes;
- slot reuse only after completion;
- generation mismatch does not publish;
- failure fallback;
- shutdown drains safely.

### Concurrency

- many outstanding loads;
- reuse pressure;
- I/O completing out of request order;
- concurrent package readers;
- slot eviction while another slot executes;
- GPU use + attempted reuse ordering test.

### Codec composition

- compressed staging load then decode into final slot;
- corrupted compressed record never publishes resident bytes;
- stale compressed completion never decodes into a new generation.

### End-to-end A/B

At minimum:

```text
A: pread + raw
B: MetalIO demand-only + raw
C: MetalIO exact-async + raw
D: MetalIO pipelined + raw
E: MetalIO pipelined + rANS
```

Run cold/miss-heavy and warm-cache cases separately.

## 4.17 Success criteria

A generic MetalIO backend is successful only if:

1. no model-specific semantics leak into it;
2. all current portable paths still work;
3. exact bytes and token output match;
4. persistent slot reuse is safe under I/O and GPU concurrency;
5. V4 and Qwen can both use it;
6. asynchronous mode gives a measurable E2E win where I/O matters;
7. failures degrade to fallback rather than corruption.

---

# 5. Workstream C — emit Apple8 directly from `colic`

## 5.1 Goal

Make `macos-arm64-metal-apple8-v1` mean what its name says.

For eligible routed-expert matrices, `colic` should emit the exact Apple8 target execution layout directly into the `.coli` artifact.

The normal Apple execution path should therefore not be:

```text
source -> colic row MXFP4 -> disk -> row load -> runtime repack -> Apple8 -> GPU
```

It should be:

```text
source -> colic Apple8 -> disk -> load/decode -> Apple8 -> GPU
```

## 5.2 Why this belongs in target lowering, not the model frontend

Apple8 is not a DeepSeek or Qwen semantic property.

It is valid whenever the matrix math contract is:

```text
MXFP4 E2M1
E8M0 scale
K-group size 32
compatible scale geometry
```

Therefore:

- DeepSeek source that is already exact MXFP4 can be repacked losslessly;
- Qwen BF16 source can first be compiler-quantized to MXFP4, then Apple8-packed;
- future model frontends using the same expert math can use the same lowerer.

The model frontend should emit semantic `RoutedExpert`/matrix roles, not Apple tile bytes.

## 5.3 Compiler layering

Recommended stages:

```text
source frontend
    |
    v
semantic Matrix/RoutedExpert
    |
    v
math lowering / quant profile
    |  exact source MXFP4 OR compiler MXFP4
    v
canonical logical MXFP4 matrix object
    |
    v
target execution-layout lowering
    |  canonical row OR Apple8 tile
    v
uncompressed target execution bytes
    |
    v
optional storage codec
    v
storage planner/emitter
```

This gives `--quant` and `--target` independent responsibilities:

- `--quant mxfp4`: select mathematical encoding;
- `--target macos-arm64-metal-apple8-v1`: select Apple8 physical layout;
- `--codec ...`: select storage compression after target lowering.

## 5.4 Exact Apple8 matrix formula

For matrix `(rows = O, columns = I)`:

```text
output-row tiles = ceil(O / 8)
K groups         = ceil(I / 32)
bytes            = ceil(O / 8) * ceil(I / 32) * 136
```

Each tile is:

```text
byte 0..127    8 rows x 16 bytes packed E2M1
byte 128..135  8 E8M0 scales
```

For each logical row `o` and K group `g`:

```text
otile = o / 8
orow  = o % 8
tile_index = otile * ng + g
base = tile_index * 136

copy up to 16 packed weight bytes for K[32g..32g+31]
    into base + orow*16

copy scale[o,g]
    into base + 128 + orow
```

Unused bytes in edge tiles = zero.

This must be implemented identically in:

- Rust compiler packer;
- C runtime transform/reference validator;
- Metal kernel interpretation.

One should be used as the oracle for tests against the others.

## 5.5 Important unresolved format issue: Apple8 co-locates weights and scales

Current canonical `COLIEXPT` matrix descriptors model:

- contiguous weight span;
- contiguous scale span.

Apple8 deliberately **interleaves/co-locates scales inside every 136-byte tile**.

Do not fake this by claiming there is a normal separate scale span when there is not.

This needs an explicit production contract under the Apple8 layout ID.

Two plausible designs:

### Design A — layout-specific combined matrix payload (recommended)

For Apple8 layout:

- one matrix payload span contains the combined 136-byte tile records;
- descriptor still declares `math_format=MXFP4_E2M1` and `scale_format=UE8M0`;
- layout ID tells the runtime that scales are embedded in the combined physical span;
- separate scale offset/length is zero or another normatively defined sentinel;
- decoded/resident bytes are the combined tile bytes.

This most closely reflects the actual execution representation and the milestone-1 transform header.

### Design B — invent a new envelope subtype

Use a target-specific expert envelope analogous to the experimental `MT8A` header.

This is possible but increases format/parser complexity and risks fragmenting `COLIEXPT`.

**Recommendation:** prefer Design A if CSF v1.x layout-specific descriptor semantics can represent it cleanly. Amend validator rules explicitly. Do not rely on accidental tolerance in current parser code.

## 5.6 Production IDs must come from the target/layout registry

The branch currently uses fixture values such as:

```text
layout         = 0x7131
layout ABI     = 0x0131
kernel ABI     = 0x0131
target class   = 0x01310008
```

These are intentionally non-production.

Do not copy them into `colic` or `coli_format.h` as permanent IDs.

The production work must land/finalize the #26-owned assignments for:

- Apple8 execution layout ID;
- execution-layout ABI;
- kernel ABI range/version;
- target compatibility class/family as appropriate.

The compiler and runtime must consume the same registry definitions or generated constants so they cannot drift.

## 5.7 Target identity must change if physical output changes

The CSF v1.1 target-identity amendment requires target profile, semantic ABI, target triple, layout/kernel ABI and related profile data to participate in artifact identity.

Do not silently change what `macos-arm64-metal-apple8-v1` emits while leaving an artifact identity that could collide with old row-layout packages.

If the current profile has shipped only as experimental/dev state, decide whether:

- redefine/finalize its ABI before production; or
- bump to a new profile/ABI name.

The runtime should fail closed on incompatible profile/layout/kernel ABI.

## 5.8 Refactor `mxfp4_record.rs`

Current `colic/src/quant/mxfp4_record.rs` does too much physical-layout work for a file under `quant/`:

- it quantizes;
- it assumes row physical layout;
- it emits the expert envelope.

Refactor toward:

```text
quant/mxfp4.rs
    quantization math / PackedMatrix logical canonical data

target/canonical_mxfp4.rs
    canonical row execution layout

target/apple8_mxfp4.rs
    Apple8 execution layout

target/expert_record.rs (or equivalent)
    envelope/descriptor emission based on selected target layout
```

Exact names are flexible.

Key rule: the target profile, not the architecture, chooses `apple8_mxfp4`.

## 5.9 Support both exact-source MXFP4 and compiler-quantized MXFP4

### Existing MXFP4 source

If the source matrix already has the exact E2M1/E8M0/g32 contract:

```text
read packed row weights/scales
 -> Apple8 exact repack
```

No float conversion.

### BF16/F16/F32 source compiled with `--quant mxfp4`

```text
source floats
 -> MXFP4 quantize
 -> Apple8 physical pack
```

Avoid materializing a full canonical row copy solely to repack if a streaming/direct tile quantizer can produce identical results.

Correctness comes first: it is fine for the first implementation to quantize to the proven canonical `PackedMatrix` then pack Apple8, followed by a later streaming optimization.

## 5.10 Streaming compiler implementation

Full-model compilation must remain bounded-memory.

For exact source MXFP4, Apple8 packing is naturally tile-streamable:

```text
for output tile of 8 rows:
    for K group of 32:
        gather 8 x 16 packed bytes
        gather 8 scales
        write 136-byte tile
```

Only a small source window is needed if source tensors support efficient range reads.

For compiler quantization from floats, a reasonable first bounded unit is 8 output rows x a manageable K chunk, with identical per-32 quantization behavior.

Do not allocate a complete multi-GiB model tensor just for target lowering.

## 5.11 Compiler storage planning

For uncompressed Apple8, matrix decoded/stored sizes are deterministic from geometry:

```text
matrix_bytes = ceil(rows/8) * ceil(columns/32) * 136
```

Expert record size follows from the envelope plus three aligned matrix payloads.

For compressed Apple8, final `stored_bytes` is data-dependent and must be provided by the codec planning stage described in workstream A.

## 5.12 Direct equivalence test against the C reference packer

Create shared fixtures that prove Rust and C agree byte-for-byte.

Shapes should include:

```text
1 x 1
1 x 31
1 x 32
1 x 33
7 x 32
8 x 32
9 x 32
8 x 31
8 x 33
9 x 33
real gate/up/down-like shapes
```

Generate deterministic weight nibbles and scale bytes.

Assert:

```text
rust_apple8_pack == c_reference_apple8_pack
```

Then validate both against canonical row source with the existing C inverse validator.

## 5.13 Kernel contract test

A compiler-layout test alone is insufficient. Add an engine fixture where:

- canonical row and compiler-emitted Apple8 encode the same matrix;
- row kernel and Apple8 kernel produce identical or accepted exact target token behavior;
- for the target milestone, token IDs across a real deterministic decode are identical.

The current Apple8 kernel intentionally mirrors the row MXFP4 lane accumulation order. Preserve that unless a deliberate numerical ABI change is made and revalidated.

## 5.14 `colic --verify` must understand target layout

Verification should not merely check stored CRCs.

For Apple8 expert records it should verify:

- descriptor/layout ID is compatible with target profile;
- matrix byte size matches geometry exactly;
- edge padding is zero where required;
- logical scale/weight geometry is coherent;
- codec decode size is exact if compressed;
- decoded CRC matches;
- optional inverse/reference validation for compiler fixtures or full verification mode.

## 5.15 Apple8 compiler acceptance gates

Do not call this finished until:

1. Apple8 profile emits actual Apple8 records;
2. compiler and C reference packers are byte-identical;
3. no runtime row→tile repack occurs for Apple8-target packages;
4. runtime recognizes the production layout/ABI and rejects mismatches;
5. real model deterministic token traces match the row reference;
6. package can run both uncompressed and, later, rANS-compressed through the same execution representation;
7. default/non-Apple target behavior is unchanged.

---

# 6. How the three workstreams compose

## 6.1 Correct order of transformations

The conceptual order is:

```text
SEMANTIC MODEL
    |
    v
MATH / QUANTIZATION
    |
    v
EXECUTION LAYOUT
    |
    v
STORAGE CODEC
    |
    v
PHYSICAL SHARD PLACEMENT
```

At runtime the inverse storage-only part is:

```text
PHYSICAL SHARD BYTES
    |
    | I/O
    v
STORED RECORD
    |
    | codec decode
    v
EXECUTION LAYOUT BYTES
    |
    v
RESIDENCY + DISPATCH
```

Never decode Apple8 back to canonical rows merely because the storage codec is removed.

## 6.2 Best final raw path

```text
colic:
  MXFP4 -> Apple8

.coli:
  raw Apple8 record

runtime:
  MetalIO direct to persistent Shared final slot
  -> publish Apple8 representation
  -> Apple8 kernel
```

## 6.3 Best final compressed path

```text
colic:
  MXFP4 -> Apple8 -> rANS

.coli:
  rANS(Apple8 bytes)

runtime:
  MetalIO -> compressed staging
  -> CPU/NEON rANS decode directly into persistent Shared final slot
  -> publish Apple8 representation
  -> Apple8 kernel
```

## 6.4 Runtime transform fallback

If a package contains canonical row MXFP4 but runtime/backend wants Apple8:

```text
row package
 -> load/decode row resident representation
 -> #135 exact transform
 -> Apple8 derived representation
 -> execute
```

This is useful for compatibility and the #133–#137 multi-variant architecture.

It should **not** be the normal path for a package explicitly compiled for Apple8.

---

# 7. Recommended implementation sequence / PR ladder

The safest sequence is not exactly “finish one whole issue, then the next”. Some interfaces should land early so the three paths compose cleanly.

## PR 1 — freeze production Apple8 representation contract

Scope:

- assign production Apple8 layout/kernel/target IDs via #26 owner;
- document 8x32/136-byte physical layout;
- define Apple8 descriptor semantics for embedded scales;
- update `coli_format` validator rules and representation resolution;
- no performance behavior change required.

Tests:

- descriptor fixtures accepted/rejected correctly;
- representation ID exact matching;
- malformed combined spans rejected.

Why first: rANS and compiler output need to know what the final decoded byte representation actually is.

## PR 2 — Apple8 target lowerer in `colic`

Scope:

- Rust Apple8 packer;
- target-based dispatch;
- exact source MXFP4 repack;
- compiler-quantized MXFP4 -> Apple8;
- storage planning for raw Apple8;
- `--verify` support;
- no rANS yet.

Tests:

- Rust/C byte identity;
- deterministic package output;
- real package token identity;
- prove runtime repack count = 0 for native Apple8 package.

## PR 3 — generic CSF codec/table plumbing

Scope:

- codec-table manifest encode/decode API;
- `CodecRequest` implementation;
- generic nibble-rANS wrapper around `rans.h` semantics;
- compiler deterministic table generation;
- compressed-size planning/spooling;
- runtime synchronous decode path first;
- no MetalIO dependency required.

Tests:

- package round trips;
- malformed codec tables/records rejected;
- raw vs rANS decoded Apple8 bytes identical;
- token identity.

This makes codec correctness independently reviewable.

## PR 4 — generic async I/O interface + MetalIO adapter

Scope:

- physical span accessor;
- generic slot/ticket/generation interface;
- wrap existing `metalio.*`;
- portable `pread` implementation;
- migrate Qwen to generic interface or build compatibility adapter.

Tests:

- byte identity;
- generation/slot reuse;
- Qwen performance non-regression.

## PR 5 — `.coli` residency/executor uses generic async I/O

Scope:

- V4/package expert store submits exact async loads;
- raw Apple8 direct-to-final-slot fast path;
- proper execution lease before reuse;
- instrumentation.

A/B:

- pread vs MetalIO demand vs MetalIO async/pipeline.

## PR 6 — compose rANS with MetalIO pipeline

Scope:

- compressed staging pool;
- bounded decode workers;
- decode directly into final Shared resident slots;
- generation-safe completion chain;
- pipeline I/O + decode + GPU.

A/B:

- raw MetalIO Apple8;
- rANS MetalIO Apple8;
- cold/miss-heavy and warm-cache.

## PR 7 — remove redundant/ad-hoc paths

Only after all gates pass:

- delete/retire model-specific MetalIO code;
- retire runtime Apple8 repack as the normal native-target path;
- retain exact transform as compatibility/derived-representation path;
- simplify temporary milestone branch wrappers/caches.

---

# 8. Performance model and what to measure

## 8.1 Do not extrapolate the 2.4x kernel win to tok/s

Apple8’s synthetic kernel result is useful but only bounds one component.

If expert kernel work is fraction `f` of token time and kernel speedup is `S`, the best Amdahl-style token speedup ignoring new I/O effects is:

```text
1 / ((1-f) + f/S)
```

But the real project also changes I/O bytes and overlap, so direct E2E measurement is more important than this model.

## 8.2 Compression changes the storage term

If rANS ratio is `r`:

```text
physical bytes ~= r * raw execution bytes
```

With `r = 0.76`, roughly 24% fewer expert bytes cross storage in miss-heavy conditions.

But effective gain depends on:

- page cache;
- SSD queueing;
- whether storage was critical;
- decode cost;
- overlap.

## 8.3 The target is overlap, not component sum

Idealized pipelined stage times:

```text
Tio
Tdecode
Tgpu
```

Naive:

```text
T ~= Tio + Tdecode + Tgpu
```

Well-pipelined steady state:

```text
T ~= max(Tio, Tdecode, Tgpu) + scheduling bubbles
```

Instrument bubbles/waits, not just raw component totals.

## 8.4 Standard benchmark matrix

For every meaningful real-model change, report:

| Variant | codec | I/O | layout | tok/s best-of-3 | token IDs | read GiB | decode ms | storage wait ms | expert GPU ms |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| baseline | none | pread | row | | | | | | |
| tile/raw | none | pread | Apple8 | | | | | | |
| tile/MIO demand | none | MetalIO | Apple8 | | | | | | |
| tile/MIO async | none | MetalIO | Apple8 | | | | | | |
| compressed | rANS | MetalIO async | Apple8 | | | | | | |

Also separately record cold/miss-heavy runs and warm/page-cache runs.

## 8.5 Token identity

Use exact token ID traces and `cmp`, not generated text.

If any stage claims to be lossless/exact and tokens diverge, treat it as a correctness failure before analyzing speed.

---

# 9. Major failure modes to watch for

## 9.1 Compressing the wrong representation

Bad:

```text
row MXFP4 -> rANS -> disk -> decode row -> runtime Apple8 repack
```

for an Apple8-target package.

Better:

```text
row/logical MXFP4 -> Apple8 -> rANS -> disk
```

Compression should apply to the final target execution bytes.

## 9.2 Codec table cannot encode all symbols

The old int4 corpus may omit symbols that appear in arbitrary Apple8 scale bytes.

Validate table coverage during compile. Do not discover this at runtime on a 155 GB artifact.

## 9.3 Demand-only MetalIO regression/neutrality

A generic backend that does:

```text
submit
wait
submit
wait
```

may be neutral or worse than `pread`.

The scheduler must issue multiple known-needed requests and defer waits.

## 9.4 Stale slot completion

Any asynchronous load without generation validation can corrupt expert identity under LRU reuse.

## 9.5 GPU reads overwritten slot

I/O generation correctness does not protect against overwriting a buffer still used by GPU compute. Execution leases/fences are required too.

## 9.6 Hidden copy reintroduced

Audit every stage for accidental:

```text
MetalIO buffer -> malloc -> MTLBuffer
```

Raw Apple8 should remain direct. Compressed Apple8 needs one expansion step, but decode should write directly to the final Shared execution buffer.

## 9.7 Compiler profile lies about layout

Do not leave `macos-arm64-metal-apple8-v1` producing canonical row MXFP4 after this workstream.

## 9.8 Inventing production IDs from fixture values

The experimental `0x7131`/`0x0131` values are forbidden as permanent registry IDs unless #26 explicitly adopts them through the real registry process.

## 9.9 Descriptor pretending scales are contiguous

Apple8 embeds scales. The CSF descriptor/validator must state that honestly.

## 9.10 Compression accidentally becomes representation identity

Do not put codec into `ColiRepresentationId`, transform matching, backend execution capability, or representation cache keys.

## 9.11 Full-record allocations during compilation

A compiler that works on toy fixtures but needs massive RAM for real routed experts is not complete. Use bounded streaming/spooling.

## 9.12 Page-cache-warm benchmarks hiding I/O

MetalIO/compression benefits can disappear when all tested bytes are already page-cache resident. Maintain explicitly cold/miss-heavy tests or enough working set to force real reads.

---

# 10. File-level map for the next programmer

## rANS / codecs

```text
c/rans.h
c/tools/rans_format.py
c/tools/repack_rans.py
c/tools/rans_verify.py
c/tools/rans_ctypes.c
c/tests/test_rans.c
c/tests/fuzz_rans.c
c/tests/test_rans_repack.py
docs/int4-rans256-g0.md
docs/FORMATS.md
c/coli_format.h
c/coli_format.c
colic/src/pipeline.rs
colic/src/storage/mod.rs
colic/src/verify.rs
```

Likely new/refactored areas:

```text
c/coli_codec.[ch]                 # generic decode/table dispatch
colic/src/codec/...               # compiler rANS/table generation
```

## MetalIO / async physical I/O

```text
c/metalio.h
c/metalio.mm
c/tests/test_metalio.mm
docs/metal-io.md
c/qwen_moe.c                      # current proven integration reference
c/coli_format.[ch]                # source-span metadata/accessor
c/coli_executor.[ch]
c/coli_v4_expert_store.c
c/expert_residency.[ch]
c/expert_residency_policy.[ch]
c/expert_dispatch.h
```

Likely new/refactored areas:

```text
c/coli_io.[ch]
c/coli_io_metal.mm
c/coli_io_pread.c
```

## Apple8 target lowering

```text
colic/src/target/mod.rs
colic/src/pipeline.rs
colic/src/quant/mxfp4.rs
colic/src/quant/mxfp4_record.rs
colic/src/storage/mod.rs
colic/src/verify.rs
c/coli_format.[ch]
c/expert_representation.h
c/expert_transform.[ch]
```

Reference/experimental Apple8 files:

```text
c/tests/apple4_bench_shader.h
c/tests/apple4_bench_pack.h
c/tests/test_apple_4bit_bench.mm

# feat/wire-tile-engine
c/mxfp4_apple8_tile.h
c/mxfp4_apple8_tile.c
c/mxfp4_apple8_tile_transform.c
c/backend_metal_tile.h
c/backend_metal_tile.mm
c/coli_executor_tile.c
c/tests/test_mxfp4_apple8_transform.c
c/tools/deepseek_v4_tile_ab.c
c/Makefile.wire-tile
```

Target identity:

```text
docs/coli-serving-format-v1.md
docs/coli-serving-format-v1.1-target-identity.md
```

Representation/residency stack:

```text
c/expert_representation.h
c/expert_residency.h
c/expert_residency_policy.h
c/expert_transform.h
c/expert_transform.c
c/expert_derived_cache.h
c/expert_derived_cache.c
c/expert_dispatch.h
```

---

# 11. Decisions that should be made explicitly before coding too far

## Decision 1 — exact Apple8 CSF matrix descriptor semantics

Need a normative answer for co-located scales.

Recommended: combined layout-specific matrix data span, no fake separate scale span.

## Decision 2 — production Apple8 layout/kernel IDs

Must come from #26 registry ownership.

## Decision 3 — codec table scope

Recommended first: artifact-wide deterministic table with record references by `codec_table_id`.

## Decision 4 — rANS coverage

Recommended first: generic nibble-rANS over exact target execution bytes, including scale bytes, with a table trained over those bytes.

Do not assume old int4 `g0` frequencies are sufficient.

## Decision 5 — codec granularity

Need to decide whether the outer expert envelope remains raw while matrix payloads are compressed, or whether the whole typed record is coded and decoded before parse.

Recommendation: keep enough uncompressed scheduling metadata outside the encoded payload that runtime can issue I/O without decoding first.

## Decision 6 — async I/O abstraction ownership

Recommended: generic executor/residency service owns requests; MetalIO is one backend; package parser only exposes spans.

## Decision 7 — staging pool policy

Compressed reads need bounded reusable staging buffers. Size them from observed maximum stored record/chunk and cap concurrency according to memory budget/queue depth.

## Decision 8 — decode worker concurrency

Start conservative. Too many CPU decoders can steal cycles/memory bandwidth from model CPU work and reduce token throughput despite higher codec GB/s.

Benchmark 1/2/4 workers or derive from actual expert wave concurrency.

---

# 12. Suggested concrete APIs/data structures

These are design sketches, not frozen ABI.

## 12.1 Compiler target-lowered matrix

```rust
struct LoweredMatrix {
    role: MatrixRole,
    math_format: MathFormat,
    scale_format: ScaleFormat,
    layout: ExecutionLayout,
    rows: u64,
    columns: u64,
    group_size: u32,
    scale_block_rows: u32,
    scale_block_columns: u32,
    decoded_bytes: u64,
    payload: LoweredPayload,
}

enum LoweredPayload {
    Canonical { weights: ..., scales: ... },
    Combined { bytes: ... }, // Apple8
}
```

The storage codec consumes `LoweredPayload` bytes and should not care which model produced them.

## 12.2 Runtime stored-record source

```c
typedef struct {
    uint64_t record_id;
    uint32_t shard_id;
    const char *shard_path;
    uint64_t offset;
    uint64_t stored_bytes;
    uint64_t decoded_bytes;
    uint16_t codec;
    uint32_t codec_table_id;
    uint32_t stored_crc32c;
    uint32_t logical_crc32c;
    ColiRepresentationId decoded_representation;
} ColiStoredRepresentationSource;
```

This is roughly the information needed to bridge package metadata to generic I/O/residency.

## 12.3 Async load state machine

```text
EMPTY
  |
  | allocate generation G
  v
IO_SUBMITTED(G)
  |
  | event completes + generation still G
  v
STORED_READY(G)
  |
  +-- codec none -------------------+
  |                                 |
  +-- codec rANS -> DECODING(G)     |
                      |             |
                      v             |
                DECODED_READY(G) <--+
                      |
                      | CRC/validation
                      v
                RESIDENT_READY(G)
                      |
                      | lease
                      v
                 EXECUTING(G)
                      |
                      v
                RESIDENT_READY(G)
```

Any failure before `RESIDENT_READY` must leave the representation unpublished.

## 12.4 Completion token

```c
typedef struct {
    uint64_t ticket_id;
    uint32_t slot_id;
    uint64_t generation;
} ColiIoCompletionKey;
```

Never publish by ticket ID alone.

---

# 13. Definition of done for the whole project

The three major issues are collectively done when all of the following are true.

## Compiler

- `macos-arm64-metal-apple8-v1` emits production Apple8 bytes for eligible MXFP4 routed experts.
- Apple8 output is byte-identical to the approved reference packer.
- target identity correctly fingerprints physical layout/kernel/storage choices.
- `--codec none`, forced rANS, and `--codec auto` have deterministic documented behavior.
- large models compile in bounded memory.

## Format/runtime correctness

- CSF represents Apple8 embedded scales honestly.
- codec tables are validated and referenced correctly.
- decoded representation identity excludes codec.
- malformed compressed data cannot publish resident bytes.
- runtime rejects incompatible target/layout/kernel ABI.

## I/O

- a model-neutral async I/O interface exists.
- Apple MetalIO implements it using persistent shared MTLBuffers.
- `pread` remains a safe fallback.
- exact known-needed expert reads can be issued asynchronously before waits.
- slot generations prevent stale completion corruption.
- execution leases prevent overwrite while GPU reads.

## Composition

- raw Apple8 can load directly from MetalIO into final execution slots.
- compressed Apple8 loads to staging and decodes directly into final execution slots.
- no unnecessary host copy or runtime row→tile repack exists on the native Apple8 path.

## Verification

- exact decoded bytes match uncompressed target bytes.
- real deterministic token traces are identical.
- best-of-3 E2E benchmarks include component telemetry.
- rANS produces meaningful physical byte savings.
- MetalIO async/pipelined path produces a measurable gain on an I/O-sensitive workload.
- compressed MetalIO path is neutral or faster E2E; if slower, codec remains optional rather than default.

---

# 14. Recommended immediate next action

The **first concrete task should be to freeze the production Apple8 descriptor/layout contract** before adding more runtime machinery.

Reason:

- `colic` needs the contract to emit final bytes;
- rANS needs to know exactly what bytes it is compressing;
- the runtime needs the layout ID to know whether a raw MetalIO destination is immediately executable;
- artifact identity must distinguish the final physical ABI;
- the current experimental branch intentionally uses fixture IDs and a private `MT8A` transform envelope, so copying it directly into compiler output would prematurely freeze the wrong ABI.

After that, implement Apple8 in `colic` **without compression first** and prove:

```text
compiler Apple8 bytes == C reference Apple8 bytes
runtime repack count == 0
token IDs == row baseline
```

Then make rANS a generic CSF codec over those final target bytes. Once synchronous codec correctness is locked, move the existing proven MetalIO machinery behind the generic async-I/O interface and compose the pipeline.

That sequencing gives every optimization a clean oracle and prevents three simultaneous sources of corruption from being debugged at once.

---

# 15. Short architectural north star

If implementation choices become confusing, return to this rule:

> **`colic` should do target compilation once; the package should store exactly that target representation, optionally compressed; the runtime should move/decompress it asynchronously into a representation-aware resident slot and execute it without translating it again.**

For Apple Silicon V4, the ideal steady-state miss is therefore:

```text
compressed Apple8 bytes on NVMe
    -> MetalIO staging
    -> exact rANS decode into Shared Apple8 slot
    -> Apple8 Metal kernel
```

and the ideal raw miss is even simpler:

```text
raw Apple8 bytes on NVMe
    -> MetalIO directly into Shared Apple8 slot
    -> Apple8 Metal kernel
```

Everything else should be judged against whether it moves the system closer to those paths while preserving exactness, deterministic target identity, safe residency, and portable fallback.

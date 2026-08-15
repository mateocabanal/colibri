# MetalIO expert streaming — architecture note

Status: subsystem landed (metalio.h/.mm + unit test). Engine wiring in progress.

## Existing path (CPU-mediated)

    NVMe -> pread() -> 16 KiB-aligned host slab -> newBufferWithBytesNoCopy
         -> MTLBuffer (same unified-memory pages) -> Metal expert kernels

Apple Silicon has unified memory, so there is no discrete-GPU upload to
eliminate. The cost that remains is CPU-mediated I/O on the critical path:
pread syscalls, slab bookkeeping, buffer registration, and the CPU polling or
waiting between "read done" and "GPU dispatch".

## New path (experimental, COLI_METAL_IO=1)

    NVMe -> MTLIOCommandQueue -> persistent shared MTLBuffer -> Metal

The goal is overlap: token time -> max(IO, GPU) instead of IO + GPU.

- metalio.h/.mm: a C API over MTLIOCommandQueue (macOS 13+ / Metal 3,
  runtime-gated with @available; compile-gated to darwin via the Makefile).
- Persistent slot pool: MTLBuffers are allocated once (shared storage,
  16 KiB-aligned) and reused across many expert replacements — never a new
  buffer per miss.
- Loads: `metalio_load(slot, file, offset, bytes)` enqueues an
  MTLIOCommandBuffer load + a shared-event signal and returns the event
  value; the caller waits (`metalio_wait`) only when a layer genuinely
  needs a slot whose IO has not completed.
- File handles are URL-based (the macOS 27 SDK removed the fd descriptor);
  the engine passes shard paths.
- Fallback: every failure path returns an error and the engine keeps using
  pread. MetalIO is never mandatory.
- Metrics: loads/bytes/waits/fails, prefetch used/wasted, outstanding +
  peak, average latency + log2 histogram (ColiMetalioStats).

## SDK notes (macOS 27 / Xcode beta)

- `newIOCommandQueueWithDescriptor:error:` (the one-arg form is gone).
- `newIOFileHandleWithURL:error:` (no MTLIOFileHandleDescriptor class).
- `waitUntilSignaledValue:timeoutMS:` (the bare one-arg form is gone).
- `signalEvent:value:` (renamed from `signalEvent:atValue:`).
- Pitfall: a local variable named `id` in Objective-C++ shadows the `id`
  type and breaks `id<Protocol>` parsing — name locals fid/sid.

## Benchmark plan (H1-H5 of the handoff)

Baseline (already measured for the CPU path): prefill memory-bandwidth-bound
~10 GFLOPS, 78% expert LRU miss at cap 8 (qwen engine, same class).
MetalIO measurements required: CPU%, GPU%, storage-stall, tok/s for
CPU-only vs METAL=1 vs METAL=1+METALIO, cold-cache and warm-cache, queue
depth 1/2/4/8. Pending the engine wiring + a GLM snapshot to drive.

## Engine wiring design (next increment, colibri.c)

Integration point (verified in code, 2026-08-15): moe()'s Metal path builds
resident-expert blocks via MB_BUILD (pointer arrays MG/MU/MD + scales at
~colibri.c:4360-4389) and dispatches `coli_metal_moe_block_begin` async;
demand misses load AFTER the submit (line ~4395: PIPE async or blocking
`expert_load(..., demand=1)`), so preads already overlap GPU compute.
MetalIO replaces the pread inside expert_load for Metal-bound experts:

1. `COLI_METAL_IO=1` (requires `g_metal_enabled`): `metalio_init()` at engine
   start; `metalio_file_add()` per shard path (st.h has the paths).
2. Per cache slot, allocate one persistent metalio slot sized to the row's
   widest expert (slab sizing rule already exists). In expert_load's demand
   path: `metalio_load(slot, file, off, wtot)` + `metalio_wait(ev)` instead
   of the pread; set the QT's weight pointers to `metalio_slot_ptr(slot)`
   (shared storage — natively GPU-visible, no coli_metal_register needed;
   scales stay in the registered fslab, copied from the slot contents).
   The slab becomes unused for metalio-resident experts.
3. Slot reuse on eviction: new load into the same MTLBuffer — no
   register/unregister churn. Ordering (never overwrite a slot a GPU CB is
   still reading): v1 relies on the existing t_ewait blocking at the next
   use; the full GPU-side wait (compute CB `waitForEvent:value:` on the IO
   event, prompt's "enqueue IO / enqueue compute / GPU waits") is the
   follow-up commit.
4. PILOT/WILLNEED (line ~4410): prefetch = enqueue `metalio_load` WITHOUT
   waiting; the demand wait then returns immediately (event already
   signaled). Track prefetch used/wasted via `metalio_slot_consumed` /
   `metalio_prefetch_done`.
5. Fallback: any metalio failure (enqueue -1, init 0) → existing pread path;
   `COLI_METAL_IO` off by default. Shutdown drains via metalio_shutdown().

Verification: build with Metal, run GLM selftest + a snapshot (need a GLM
tiny fixture), CPU vs METAL vs METAL+METALIO probe, then the H1-H5 numbers.


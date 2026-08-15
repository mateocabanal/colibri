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

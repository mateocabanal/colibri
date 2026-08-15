/* metalio.h — MetalIO expert streaming for colibri (Apple Silicon, Metal 3).
 *
 * Async expert-weight loads from NVMe straight into PERSISTENT MTLBuffers via
 * MTLIOCommandQueue, so disk I/O overlaps GPU compute without the CPU-mediated
 * pread() -> slab -> register chain. Buffers are MTLResourceStorageModeShared:
 * the CPU can still read expert bytes (fallback/matmul paths keep working).
 *
 * Lifecycle: metalio_init() -> metalio_file_add(fd) -> metalio_slot_alloc()
 * -> metalio_load() (non-blocking, returns an event value) -> metalio_wait()
 * at the layer that genuinely needs the expert. metalio_shutdown() drains.
 * Every failure returns an error code and the caller falls back to the
 * existing pread path — MetalIO is never mandatory.
 *
 * Compile-gated to darwin; the header is plain C so engines can include it
 * unconditionally behind #ifdef COLI_METAL. */
#ifndef COLI_METALIO_H
#define COLI_METALIO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- lifecycle ---------------------------------------------------------- */
int  metalio_init(void);              /* 1 = active, 0 = unavailable/failed */
void metalio_shutdown(void);          /* drains outstanding IO, releases everything */
int  metalio_active(void);

/* --- file handles ------------------------------------------------------- */
/* Wrap a model shard path in a persistent MTLIOFileHandle (the SDK's IO
 * handles are URL-based; the engine owns the file). Returns a file id >= 0
 * for metalio_load(), or -1 (caller keeps using pread). */
int  metalio_file_add(const char *path);

/* --- persistent slot pool ------------------------------------------------ */
/* One shared-storage MTLBuffer, 16 KiB-aligned length, alive across many
 * expert replacements. Returns slot id or -1. */
int  metalio_slot_alloc(size_t max_bytes);
void metalio_slot_free(int slot);
void *metalio_slot_ptr(int slot);             /* CPU-visible (shared storage) */
size_t metalio_slot_bytes(int slot);

/* --- async loads --------------------------------------------------------- */
/* Enqueue a load of [offset, offset+bytes) from `file` into `slot`. Returns
 * the shared-event value to wait on, or -1 on failure (fall back to pread). */
int64_t metalio_load(int slot, int file, uint64_t offset, size_t bytes);

/* CPU wait until the load signalling `event_value` (and everything before it
 * on the IO queue) has completed. Returns 0 when done. */
int metalio_wait(int64_t event_value);

/* --- prefetch accounting ------------------------------------------------- */
/* Mark a slot's latest load as consumed by compute (prefetch_used), or as
 * wasted when it is evicted/replaced before use (prefetch_wasted). */
void metalio_slot_consumed(int slot);
void metalio_prefetch_done(int slot);

/* --- metrics ------------------------------------------------------------- */
typedef struct {
    uint64_t loads;                 /* MTLIO loads enqueued */
    uint64_t bytes;                 /* bytes loaded through MTLIO */
    uint64_t waits;                 /* CPU waits issued */
    uint64_t fails;                 /* enqueue failures (fallbacks) */
    uint64_t prefetch_loads;        /* loads marked as prefetch */
    uint64_t prefetch_used;         /* prefetched slots consumed by compute */
    uint64_t prefetch_wasted;       /* prefetched slots evicted unused */
    uint64_t outstanding;           /* loads enqueued, not yet waited */
    uint64_t peak_outstanding;
    uint64_t latency_samples;
    double  total_latency_s;        /* avg latency = total/samples */
    uint64_t lat_hist[32];          /* log2-bucketed latency histogram (us) */
} ColiMetalioStats;
void metalio_stats(ColiMetalioStats *out);

/* --- debug --------------------------------------------------------------- */
void metalio_verbose(int on);       /* [metalio] trace lines */

#ifdef __cplusplus
}
#endif
#endif /* COLI_METALIO_H */

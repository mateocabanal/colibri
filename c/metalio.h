/* metalio.h — Apple MetalIO provider for the generic record I/O backend.
 *
 * New runtime/model code should consume record_io_backend.h and obtain this
 * provider with coli_record_io_metal_default(). The historical metalio_* API
 * remains available as ordinary functions for compatibility and focused Apple
 * backend tests. Do not macro-wrap those functions: compound-literal region
 * vectors contain commas and must remain valid C/C++ call arguments.
 *
 * MetalIO is optional: initialization may fail on unsupported systems and the
 * caller must retain its ordinary pread/synchronous fallback.
 */
#ifndef COLI_METALIO_H
#define COLI_METALIO_H

#include <stddef.h>
#include <stdint.h>

#include "record_io_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Shared process-local Apple provider. It contains no model/MoE policy; files,
 * buffers, regions and completion events are expressed through the generic
 * ColiRecordIoBackend contract. */
ColiRecordIoBackend *coli_record_io_metal_default(void);

/* Legacy/source-compatibility surface. New runtime code should prefer
 * record_io_backend.h directly. These are real functions, never macros. */
int  metalio_init(void);
void metalio_shutdown(void);
int  metalio_active(void);
int  metalio_file_add(const char *path);
int  metalio_slot_alloc(size_t max_bytes);
void metalio_slot_free(int slot);
void *metalio_slot_ptr(int slot);
size_t metalio_slot_bytes(int slot);

typedef enum {
    MIO_LOAD_DEMAND = COLI_RECORD_IO_INTENT_DEMAND,
    MIO_LOAD_ASYNC = COLI_RECORD_IO_INTENT_ASYNC,
    MIO_LOAD_SPEC = COLI_RECORD_IO_INTENT_SPECULATIVE,
} ColiMetalioKind;

typedef ColiRecordIoBackendRegion ColiMetalioRegion;

int64_t metalio_loadv(int slot, const ColiMetalioRegion *regions, int count,
                      ColiMetalioKind kind);
int64_t metalio_load(int slot, int file, uint64_t offset, size_t bytes);
int metalio_wait(int64_t event_value);
void metalio_slot_consumed(int slot);
void metalio_prefetch_done(int slot);

typedef struct {
    uint64_t loads;
    uint64_t bytes;
    uint64_t waits;
    uint64_t fails;
    uint64_t prefetch_loads;
    uint64_t prefetch_used;
    uint64_t prefetch_wasted;
    uint64_t outstanding;
    uint64_t peak_outstanding;
    uint64_t latency_samples;
    double total_latency_s;
    uint64_t lat_hist[32];
} ColiMetalioStats;
void metalio_stats(ColiMetalioStats *out);
void metalio_verbose(int on);

#ifdef __cplusplus
}
#endif
#endif /* COLI_METALIO_H */

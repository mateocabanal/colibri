/* metalio.h — Apple MetalIO provider for the generic record I/O backend.
 *
 * New runtime/model code should consume record_io_backend.h and obtain this
 * provider with coli_record_io_metal_default(). The historical metalio_* API
 * remains source-compatible below, but normal callers are transparently routed
 * through the generic backend contract. metalio.mm defines
 * COLI_METALIO_IMPLEMENTATION so it can implement the raw Apple symbols.
 *
 * MetalIO is optional: initialization may fail on unsupported systems and the
 * caller must retain its ordinary pread/synchronous fallback.
 */
#ifndef COLI_METALIO_H
#define COLI_METALIO_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "record_io_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Shared process-local Apple provider. It contains no model/MoE policy; files,
 * buffers, regions and completion events are all expressed through the generic
 * ColiRecordIoBackend contract. */
ColiRecordIoBackend *coli_record_io_metal_default(void);

/* -------------------------------------------------------------------------
 * Legacy MetalIO surface.
 *
 * Keep the raw ABI for old binaries/tests and for metalio.mm itself. In normal
 * C/C++ translation units the function-like macros below route these calls via
 * the generic provider. This lets Qwen migrate without a model-code rewrite and
 * leaves one generic API for V4 and future engines.
 * ------------------------------------------------------------------------- */
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

#ifndef COLI_METALIO_IMPLEMENTATION
static inline ColiRecordIoBackend *coli_metalio_compat_backend(void) {
    return coli_record_io_metal_default();
}

static inline int coli_metalio_compat_init(void) {
    return coli_record_io_backend_init(coli_metalio_compat_backend());
}
static inline void coli_metalio_compat_shutdown(void) {
    coli_record_io_backend_shutdown(coli_metalio_compat_backend());
}
static inline int coli_metalio_compat_active(void) {
    return coli_record_io_backend_active(coli_metalio_compat_backend());
}
static inline int coli_metalio_compat_file_add(const char *path) {
    return (int)coli_record_io_backend_file_add(
        coli_metalio_compat_backend(), path);
}
static inline int coli_metalio_compat_slot_alloc(size_t max_bytes) {
    return (int)coli_record_io_backend_buffer_alloc(
        coli_metalio_compat_backend(), max_bytes);
}
static inline void coli_metalio_compat_slot_free(int slot) {
    coli_record_io_backend_buffer_free(
        coli_metalio_compat_backend(), (ColiRecordIoBackendBuffer)slot);
}
static inline void *coli_metalio_compat_slot_ptr(int slot) {
    return coli_record_io_backend_buffer_ptr(
        coli_metalio_compat_backend(), (ColiRecordIoBackendBuffer)slot);
}
static inline size_t coli_metalio_compat_slot_bytes(int slot) {
    return coli_record_io_backend_buffer_bytes(
        coli_metalio_compat_backend(), (ColiRecordIoBackendBuffer)slot);
}
static inline int64_t coli_metalio_compat_loadv(
        int slot, const ColiMetalioRegion *regions, int count,
        ColiMetalioKind kind) {
    return (int64_t)coli_record_io_backend_submitv(
        coli_metalio_compat_backend(), (ColiRecordIoBackendBuffer)slot,
        (const ColiRecordIoBackendRegion *)regions, count,
        (ColiRecordIoBackendIntent)kind);
}
static inline int64_t coli_metalio_compat_load(
        int slot, int file, uint64_t offset, size_t bytes) {
    return (int64_t)coli_record_io_backend_submit(
        coli_metalio_compat_backend(), (ColiRecordIoBackendBuffer)slot,
        (ColiRecordIoBackendFile)file, offset, bytes, 0,
        COLI_RECORD_IO_INTENT_DEMAND);
}
static inline int coli_metalio_compat_wait(int64_t event_value) {
    return coli_record_io_backend_wait(
        coli_metalio_compat_backend(), (ColiRecordIoBackendEvent)event_value);
}
static inline void coli_metalio_compat_slot_consumed(int slot) {
    coli_record_io_backend_buffer_consumed(
        coli_metalio_compat_backend(), (ColiRecordIoBackendBuffer)slot);
}
static inline void coli_metalio_compat_prefetch_done(int slot) {
    coli_record_io_backend_buffer_discarded(
        coli_metalio_compat_backend(), (ColiRecordIoBackendBuffer)slot);
}
static inline void coli_metalio_compat_stats(ColiMetalioStats *out) {
    if (!out) return;
    ColiRecordIoBackendStats stats;
    coli_record_io_backend_stats(coli_metalio_compat_backend(), &stats);
    memset(out, 0, sizeof(*out));
    out->loads = stats.submissions;
    out->bytes = stats.bytes;
    out->waits = stats.waits;
    out->fails = stats.failures;
    out->prefetch_loads = stats.speculative_submissions;
    out->prefetch_used = stats.speculative_consumed;
    out->prefetch_wasted = stats.speculative_discarded;
    out->outstanding = stats.outstanding;
    out->peak_outstanding = stats.peak_outstanding;
    out->latency_samples = stats.latency_samples;
    out->total_latency_s = stats.total_latency_s;
    for (int i = 0; i < 32; i++) out->lat_hist[i] = stats.lat_hist[i];
}
static inline void coli_metalio_compat_verbose(int on) {
    coli_record_io_backend_verbose(coli_metalio_compat_backend(), on);
}

#define metalio_init() coli_metalio_compat_init()
#define metalio_shutdown() coli_metalio_compat_shutdown()
#define metalio_active() coli_metalio_compat_active()
#define metalio_file_add(_path) coli_metalio_compat_file_add((_path))
#define metalio_slot_alloc(_bytes) coli_metalio_compat_slot_alloc((_bytes))
#define metalio_slot_free(_slot) coli_metalio_compat_slot_free((_slot))
#define metalio_slot_ptr(_slot) coli_metalio_compat_slot_ptr((_slot))
#define metalio_slot_bytes(_slot) coli_metalio_compat_slot_bytes((_slot))
#define metalio_loadv(_slot, _regions, _count, _kind) \
    coli_metalio_compat_loadv((_slot), (_regions), (_count), (_kind))
#define metalio_load(_slot, _file, _offset, _bytes) \
    coli_metalio_compat_load((_slot), (_file), (_offset), (_bytes))
#define metalio_wait(_event) coli_metalio_compat_wait((_event))
#define metalio_slot_consumed(_slot) coli_metalio_compat_slot_consumed((_slot))
#define metalio_prefetch_done(_slot) coli_metalio_compat_prefetch_done((_slot))
#define metalio_stats(_out) coli_metalio_compat_stats((_out))
#define metalio_verbose(_on) coli_metalio_compat_verbose((_on))
#endif /* !COLI_METALIO_IMPLEMENTATION */

#ifdef __cplusplus
}
#endif
#endif /* COLI_METALIO_H */

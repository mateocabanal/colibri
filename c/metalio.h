/* metalio.h — Apple MetalIO provider for the generic record I/O backend.
 *
 * New runtime/model code should consume record_io_backend.h and obtain this
 * provider with coli_record_io_metal_default(). The historical metalio_* API
 * remains source-compatible and is routed through the generic backend contract.
 *
 * IMPORTANT: compatibility aliases are object-like macros, not function-like
 * macros. Function-like wrappers break valid compound-literal calls such as
 *   metalio_loadv(slot, (ColiMetalioRegion[]){ { ... }, { ... } }, 2, kind)
 * because the preprocessor treats commas inside the braced initializer as macro
 * argument separators. Object-like aliases only replace the function token and
 * leave C/C++ argument parsing to the compiler.
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
 * buffers, regions and completion events are expressed through the generic
 * ColiRecordIoBackend contract. */
ColiRecordIoBackend *coli_record_io_metal_default(void);

/* Raw ABI implemented by metalio.mm. The implementation translation unit sets
 * COLI_METALIO_IMPLEMENTATION so it sees these names directly. */
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
    return (int)coli_record_io_backend_file_add(coli_metalio_compat_backend(), path);
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

/* Object-like aliases preserve compound literals and route existing Qwen/tests
 * through ColiRecordIoBackend without touching model code. */
#define metalio_init coli_metalio_compat_init
#define metalio_shutdown coli_metalio_compat_shutdown
#define metalio_active coli_metalio_compat_active
#define metalio_file_add coli_metalio_compat_file_add
#define metalio_slot_alloc coli_metalio_compat_slot_alloc
#define metalio_slot_free coli_metalio_compat_slot_free
#define metalio_slot_ptr coli_metalio_compat_slot_ptr
#define metalio_slot_bytes coli_metalio_compat_slot_bytes
#define metalio_loadv coli_metalio_compat_loadv
#define metalio_load coli_metalio_compat_load
#define metalio_wait coli_metalio_compat_wait
#define metalio_slot_consumed coli_metalio_compat_slot_consumed
#define metalio_prefetch_done coli_metalio_compat_prefetch_done
#define metalio_stats coli_metalio_compat_stats
#define metalio_verbose coli_metalio_compat_verbose
#endif /* !COLI_METALIO_IMPLEMENTATION */

#ifdef __cplusplus
}
#endif
#endif /* COLI_METALIO_H */

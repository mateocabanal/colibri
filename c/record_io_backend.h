#ifndef COLIBRI_RECORD_IO_BACKEND_H
#define COLIBRI_RECORD_IO_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Physical asynchronous I/O backend for record_io.h.
 *
 * record_io.h owns logical request state (owner/join/hit/cancel/generation).
 * This interface deliberately owns none of that policy. It only describes how
 * bytes move from artifact files into reusable backend buffers and how callers
 * observe completion. MetalIO is the first provider; io_uring, CUDA GDS, or a
 * synchronous/pread provider can implement the same contract later.
 *
 * No model, MoE, quantization, Metal, CUDA, or filesystem-cache policy belongs
 * here. Engines translate their physical tensor/record layout into regions.
 */

typedef int32_t ColiRecordIoBackendFile;
typedef int32_t ColiRecordIoBackendBuffer;
typedef int64_t ColiRecordIoBackendEvent;

#define COLI_RECORD_IO_BACKEND_INVALID_FILE   ((ColiRecordIoBackendFile)-1)
#define COLI_RECORD_IO_BACKEND_INVALID_BUFFER ((ColiRecordIoBackendBuffer)-1)
#define COLI_RECORD_IO_BACKEND_INVALID_EVENT  ((ColiRecordIoBackendEvent)-1)

typedef enum {
    COLI_RECORD_IO_INTENT_DEMAND = 0,
    COLI_RECORD_IO_INTENT_ASYNC = 1,
    COLI_RECORD_IO_INTENT_SPECULATIVE = 2,
} ColiRecordIoBackendIntent;

typedef enum {
    COLI_RECORD_IO_CAP_ASYNC = 1u << 0,
    COLI_RECORD_IO_CAP_VECTORED = 1u << 1,
    COLI_RECORD_IO_CAP_CPU_VISIBLE_BUFFER = 1u << 2,
    COLI_RECORD_IO_CAP_DEVICE_VISIBLE_BUFFER = 1u << 3,
    /* Backend can submit reads directly into caller-owned CPU-addressable
     * storage. The caller keeps that storage alive until completion. */
    COLI_RECORD_IO_CAP_DIRECT_POINTER = 1u << 4,
} ColiRecordIoBackendCapability;

typedef struct {
    ColiRecordIoBackendFile file;
    uint64_t src_off;
    size_t bytes;
    uint64_t dst_off;
} ColiRecordIoBackendRegion;

typedef struct {
    uint64_t submissions;
    uint64_t bytes;
    uint64_t waits;
    uint64_t failures;
    uint64_t speculative_submissions;
    uint64_t speculative_consumed;
    uint64_t speculative_discarded;
    uint64_t outstanding;
    uint64_t peak_outstanding;
    uint64_t latency_samples;
    double total_latency_s;
    uint64_t lat_hist[32];
} ColiRecordIoBackendStats;

typedef struct ColiRecordIoBackendOps {
    const char *name;
    uint32_t capabilities;

    int (*init)(void *context);
    void (*shutdown)(void *context);
    int (*active)(void *context);

    /* Providers should make repeated file_add(path) calls idempotent. */
    ColiRecordIoBackendFile (*file_add)(void *context, const char *path);

    ColiRecordIoBackendBuffer (*buffer_alloc)(void *context, size_t max_bytes);
    void (*buffer_free)(void *context, ColiRecordIoBackendBuffer buffer);
    void *(*buffer_ptr)(void *context, ColiRecordIoBackendBuffer buffer);
    size_t (*buffer_bytes)(void *context, ColiRecordIoBackendBuffer buffer);

    ColiRecordIoBackendEvent (*submitv)(
        void *context, ColiRecordIoBackendBuffer buffer,
        const ColiRecordIoBackendRegion *regions, int count,
        ColiRecordIoBackendIntent intent);

    /* Optional direct-to-caller-storage form. dst_off is relative to
     * destination and destination_bytes bounds every region. */
    ColiRecordIoBackendEvent (*submitv_ptr)(
        void *context, void *destination, size_t destination_bytes,
        const ColiRecordIoBackendRegion *regions, int count,
        ColiRecordIoBackendIntent intent);

    int (*wait)(void *context, ColiRecordIoBackendEvent event);

    void (*buffer_consumed)(void *context, ColiRecordIoBackendBuffer buffer);
    void (*buffer_discarded)(void *context, ColiRecordIoBackendBuffer buffer);

    void (*stats)(void *context, ColiRecordIoBackendStats *out);
    void (*verbose)(void *context, int on);
} ColiRecordIoBackendOps;

typedef struct {
    const ColiRecordIoBackendOps *ops;
    void *context;
} ColiRecordIoBackend;

static inline int coli_record_io_backend_valid(const ColiRecordIoBackend *b) {
    return b && b->ops && b->ops->init && b->ops->shutdown && b->ops->active &&
           b->ops->file_add && b->ops->buffer_alloc && b->ops->buffer_free &&
           b->ops->buffer_bytes && b->ops->submitv && b->ops->wait;
}

static inline const char *coli_record_io_backend_name(
        const ColiRecordIoBackend *b) {
    return b && b->ops && b->ops->name ? b->ops->name : "none";
}

static inline uint32_t coli_record_io_backend_capabilities(
        const ColiRecordIoBackend *b) {
    return b && b->ops ? b->ops->capabilities : 0;
}

static inline int coli_record_io_backend_init(ColiRecordIoBackend *b) {
    return coli_record_io_backend_valid(b) ? b->ops->init(b->context) : 0;
}

static inline void coli_record_io_backend_shutdown(ColiRecordIoBackend *b) {
    if (coli_record_io_backend_valid(b)) b->ops->shutdown(b->context);
}

static inline int coli_record_io_backend_active(const ColiRecordIoBackend *b) {
    return coli_record_io_backend_valid(b) ? b->ops->active(b->context) : 0;
}

static inline ColiRecordIoBackendFile coli_record_io_backend_file_add(
        ColiRecordIoBackend *b, const char *path) {
    return coli_record_io_backend_valid(b) && path && path[0]
        ? b->ops->file_add(b->context, path)
        : COLI_RECORD_IO_BACKEND_INVALID_FILE;
}

static inline ColiRecordIoBackendBuffer coli_record_io_backend_buffer_alloc(
        ColiRecordIoBackend *b, size_t max_bytes) {
    return coli_record_io_backend_valid(b) && max_bytes
        ? b->ops->buffer_alloc(b->context, max_bytes)
        : COLI_RECORD_IO_BACKEND_INVALID_BUFFER;
}

static inline void coli_record_io_backend_buffer_free(
        ColiRecordIoBackend *b, ColiRecordIoBackendBuffer buffer) {
    if (coli_record_io_backend_valid(b) &&
        buffer != COLI_RECORD_IO_BACKEND_INVALID_BUFFER)
        b->ops->buffer_free(b->context, buffer);
}

static inline void *coli_record_io_backend_buffer_ptr(
        ColiRecordIoBackend *b, ColiRecordIoBackendBuffer buffer) {
    return coli_record_io_backend_valid(b) && b->ops->buffer_ptr &&
           buffer != COLI_RECORD_IO_BACKEND_INVALID_BUFFER
        ? b->ops->buffer_ptr(b->context, buffer) : NULL;
}

static inline size_t coli_record_io_backend_buffer_bytes(
        ColiRecordIoBackend *b, ColiRecordIoBackendBuffer buffer) {
    return coli_record_io_backend_valid(b) &&
           buffer != COLI_RECORD_IO_BACKEND_INVALID_BUFFER
        ? b->ops->buffer_bytes(b->context, buffer) : 0;
}

static inline int coli_record_io_backend_intent_valid(
        ColiRecordIoBackendIntent intent) {
    return intent >= COLI_RECORD_IO_INTENT_DEMAND &&
           intent <= COLI_RECORD_IO_INTENT_SPECULATIVE;
}

static inline ColiRecordIoBackendEvent coli_record_io_backend_submitv(
        ColiRecordIoBackend *b, ColiRecordIoBackendBuffer buffer,
        const ColiRecordIoBackendRegion *regions, int count,
        ColiRecordIoBackendIntent intent) {
    if (!coli_record_io_backend_valid(b) ||
        buffer == COLI_RECORD_IO_BACKEND_INVALID_BUFFER || !regions || count < 1 ||
        !coli_record_io_backend_intent_valid(intent))
        return COLI_RECORD_IO_BACKEND_INVALID_EVENT;
    return b->ops->submitv(b->context, buffer, regions, count, intent);
}

static inline ColiRecordIoBackendEvent coli_record_io_backend_submit(
        ColiRecordIoBackend *b, ColiRecordIoBackendBuffer buffer,
        ColiRecordIoBackendFile file, uint64_t src_off, size_t bytes,
        uint64_t dst_off, ColiRecordIoBackendIntent intent) {
    ColiRecordIoBackendRegion region = {file, src_off, bytes, dst_off};
    return coli_record_io_backend_submitv(b, buffer, &region, 1, intent);
}

static inline ColiRecordIoBackendEvent coli_record_io_backend_submitv_ptr(
        ColiRecordIoBackend *b, void *destination, size_t destination_bytes,
        const ColiRecordIoBackendRegion *regions, int count,
        ColiRecordIoBackendIntent intent) {
    if (!coli_record_io_backend_valid(b) || !b->ops->submitv_ptr ||
        !destination || !destination_bytes || !regions || count < 1 ||
        !coli_record_io_backend_intent_valid(intent))
        return COLI_RECORD_IO_BACKEND_INVALID_EVENT;
    return b->ops->submitv_ptr(b->context, destination, destination_bytes,
                               regions, count, intent);
}

static inline ColiRecordIoBackendEvent coli_record_io_backend_submit_ptr(
        ColiRecordIoBackend *b, void *destination, size_t destination_bytes,
        ColiRecordIoBackendFile file, uint64_t src_off, size_t bytes,
        size_t dst_off, ColiRecordIoBackendIntent intent) {
    ColiRecordIoBackendRegion region = {file, src_off, bytes, dst_off};
    return coli_record_io_backend_submitv_ptr(
        b, destination, destination_bytes, &region, 1, intent);
}

static inline int coli_record_io_backend_wait(
        ColiRecordIoBackend *b, ColiRecordIoBackendEvent event) {
    return coli_record_io_backend_valid(b) && event > 0
        ? b->ops->wait(b->context, event) : -1;
}

static inline void coli_record_io_backend_buffer_consumed(
        ColiRecordIoBackend *b, ColiRecordIoBackendBuffer buffer) {
    if (coli_record_io_backend_valid(b) && b->ops->buffer_consumed &&
        buffer != COLI_RECORD_IO_BACKEND_INVALID_BUFFER)
        b->ops->buffer_consumed(b->context, buffer);
}

static inline void coli_record_io_backend_buffer_discarded(
        ColiRecordIoBackend *b, ColiRecordIoBackendBuffer buffer) {
    if (coli_record_io_backend_valid(b) && b->ops->buffer_discarded &&
        buffer != COLI_RECORD_IO_BACKEND_INVALID_BUFFER)
        b->ops->buffer_discarded(b->context, buffer);
}

static inline void coli_record_io_backend_stats(
        ColiRecordIoBackend *b, ColiRecordIoBackendStats *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (coli_record_io_backend_valid(b) && b->ops->stats)
        b->ops->stats(b->context, out);
}

static inline void coli_record_io_backend_verbose(
        ColiRecordIoBackend *b, int on) {
    if (coli_record_io_backend_valid(b) && b->ops->verbose)
        b->ops->verbose(b->context, on);
}

/* Platform provider seam. Generic compiled-record code calls this and remains
 * oblivious to MetalIO/io_uring/GDS names. A portable weak implementation may
 * return NULL; a platform provider can supply the strong definition. */
#ifdef __cplusplus
extern "C" {
#endif
ColiRecordIoBackend *coli_record_io_platform_default(void);
#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_RECORD_IO_BACKEND_H */

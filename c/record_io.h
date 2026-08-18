#ifndef COLIBRI_RECORD_IO_H
#define COLIBRI_RECORD_IO_H

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

typedef enum {
    COLI_RECORD_IO_PREFETCH = 0,
    COLI_RECORD_IO_BLOCKING = 1,
} ColiRecordIoPriority;

typedef enum {
    COLI_RECORD_IO_IDLE = 0,
    COLI_RECORD_IO_QUEUED = 1,
    COLI_RECORD_IO_READING = 2,
    COLI_RECORD_IO_READY = 3,
    COLI_RECORD_IO_FAILED = 4,
    COLI_RECORD_IO_CANCELLED = 5,
} ColiRecordIoState;

typedef enum {
    COLI_RECORD_IO_INVALID = -1,
    COLI_RECORD_IO_OWNER = 1,
    COLI_RECORD_IO_JOIN = 2,
    COLI_RECORD_IO_HIT = 3,
} ColiRecordIoRequestResult;

typedef struct {
    uint64_t artifact_hi;
    uint64_t artifact_lo;
    uint64_t record_id;
} ColiRecordIoKey;

typedef struct {
    ColiRecordIoKey key;
    atomic_int state;
    atomic_int priority;
    atomic_uint waiters;
    atomic_uint blocking_waiters;
    atomic_uint_fast64_t generation;
    atomic_uint_fast64_t stored_bytes;
    atomic_int error_code;
} ColiRecordIoEntry;

typedef struct {
    ColiRecordIoEntry *entry;
    ColiRecordIoKey key;
    uint64_t generation;
    ColiRecordIoPriority priority;
} ColiRecordIoHandle;

static inline int coli_record_io_key_equal(ColiRecordIoKey a,
                                           ColiRecordIoKey b) {
    return a.artifact_hi == b.artifact_hi &&
           a.artifact_lo == b.artifact_lo &&
           a.record_id == b.record_id;
}

static inline int coli_record_io_entry_init(ColiRecordIoEntry *entry,
                                             ColiRecordIoKey key) {
    if (!entry || (!key.artifact_hi && !key.artifact_lo) || !key.record_id)
        return -1;
    memset(entry, 0, sizeof(*entry));
    entry->key = key;
    atomic_init(&entry->state, COLI_RECORD_IO_IDLE);
    atomic_init(&entry->priority, COLI_RECORD_IO_PREFETCH);
    atomic_init(&entry->waiters, 0);
    atomic_init(&entry->blocking_waiters, 0);
    atomic_init(&entry->generation, 0);
    atomic_init(&entry->stored_bytes, 0);
    atomic_init(&entry->error_code, 0);
    return 0;
}

static inline void coli_record_io_escalate(ColiRecordIoEntry *entry,
                                            ColiRecordIoPriority priority) {
    if (!entry || priority != COLI_RECORD_IO_BLOCKING) return;
    int current = atomic_load_explicit(&entry->priority, memory_order_acquire);
    while (current < COLI_RECORD_IO_BLOCKING &&
           !atomic_compare_exchange_weak_explicit(
               &entry->priority, &current, COLI_RECORD_IO_BLOCKING,
               memory_order_acq_rel, memory_order_acquire)) {}
}

/* The generation identifies one physical read/prepare attempt. One contender
 * owns IDLE/FAILED/CANCELLED -> QUEUED; all others join that generation. */
static inline ColiRecordIoRequestResult coli_record_io_request(
        ColiRecordIoEntry *entry, ColiRecordIoPriority priority,
        ColiRecordIoHandle *handle) {
    if (!entry || !handle ||
        (priority != COLI_RECORD_IO_PREFETCH &&
         priority != COLI_RECORD_IO_BLOCKING))
        return COLI_RECORD_IO_INVALID;

    for (;;) {
        int state = atomic_load_explicit(&entry->state, memory_order_acquire);
        if (state == COLI_RECORD_IO_READY) {
            uint64_t generation = atomic_load_explicit(
                &entry->generation, memory_order_acquire);
            if (!generation) return COLI_RECORD_IO_INVALID;
            handle->entry = entry;
            handle->key = entry->key;
            handle->generation = generation;
            handle->priority = priority;
            return COLI_RECORD_IO_HIT;
        }
        if (state == COLI_RECORD_IO_QUEUED || state == COLI_RECORD_IO_READING) {
            coli_record_io_escalate(entry, priority);
            atomic_fetch_add_explicit(&entry->waiters, 1, memory_order_acq_rel);
            if (priority == COLI_RECORD_IO_BLOCKING)
                atomic_fetch_add_explicit(&entry->blocking_waiters, 1,
                                          memory_order_acq_rel);
            uint64_t generation = atomic_load_explicit(
                &entry->generation, memory_order_acquire);
            if (!generation ||
                (atomic_load_explicit(&entry->state, memory_order_acquire) !=
                    COLI_RECORD_IO_QUEUED &&
                 atomic_load_explicit(&entry->state, memory_order_acquire) !=
                    COLI_RECORD_IO_READING)) {
                atomic_fetch_sub_explicit(&entry->waiters, 1,
                                          memory_order_acq_rel);
                if (priority == COLI_RECORD_IO_BLOCKING)
                    atomic_fetch_sub_explicit(&entry->blocking_waiters, 1,
                                              memory_order_acq_rel);
                continue;
            }
            handle->entry = entry;
            handle->key = entry->key;
            handle->generation = generation;
            handle->priority = priority;
            return COLI_RECORD_IO_JOIN;
        }
        if (state != COLI_RECORD_IO_IDLE && state != COLI_RECORD_IO_FAILED &&
            state != COLI_RECORD_IO_CANCELLED)
            return COLI_RECORD_IO_INVALID;

        int expected = state;
        if (!atomic_compare_exchange_weak_explicit(
                &entry->state, &expected, COLI_RECORD_IO_QUEUED,
                memory_order_acq_rel, memory_order_acquire))
            continue;

        uint64_t generation = atomic_fetch_add_explicit(
            &entry->generation, 1, memory_order_acq_rel) + 1;
        atomic_store_explicit(&entry->priority, priority, memory_order_release);
        atomic_store_explicit(&entry->waiters, 1, memory_order_release);
        atomic_store_explicit(&entry->blocking_waiters,
                              priority == COLI_RECORD_IO_BLOCKING ? 1u : 0u,
                              memory_order_release);
        atomic_store_explicit(&entry->stored_bytes, 0, memory_order_release);
        atomic_store_explicit(&entry->error_code, 0, memory_order_release);
        handle->entry = entry;
        handle->key = entry->key;
        handle->generation = generation;
        handle->priority = priority;
        return COLI_RECORD_IO_OWNER;
    }
}

static inline int coli_record_io_begin_read(ColiRecordIoHandle *handle) {
    if (!handle || !handle->entry || !handle->generation ||
        atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
            handle->generation)
        return -1;
    int expected = COLI_RECORD_IO_QUEUED;
    return atomic_compare_exchange_strong_explicit(
        &handle->entry->state, &expected, COLI_RECORD_IO_READING,
        memory_order_acq_rel, memory_order_acquire) ? 0 : -1;
}

/* Complete/fail only the exact attempt. A stale async completion from generation
 * N can never publish bytes after the entry has advanced to generation N+1. */
static inline int coli_record_io_complete(ColiRecordIoHandle *handle,
                                          uint64_t stored_bytes) {
    if (!handle || !handle->entry || !handle->generation || !stored_bytes ||
        atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
            handle->generation ||
        atomic_load_explicit(&handle->entry->state, memory_order_acquire) !=
            COLI_RECORD_IO_READING)
        return -1;
    atomic_store_explicit(&handle->entry->stored_bytes, stored_bytes,
                          memory_order_release);
    atomic_store_explicit(&handle->entry->state, COLI_RECORD_IO_READY,
                          memory_order_release);
    return 0;
}

static inline int coli_record_io_fail(ColiRecordIoHandle *handle,
                                      int error_code) {
    if (!handle || !handle->entry || !handle->generation ||
        atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
            handle->generation)
        return -1;
    int state = atomic_load_explicit(&handle->entry->state, memory_order_acquire);
    if (state != COLI_RECORD_IO_QUEUED && state != COLI_RECORD_IO_READING)
        return -1;
    atomic_store_explicit(&handle->entry->error_code, error_code,
                          memory_order_release);
    atomic_store_explicit(&handle->entry->state, COLI_RECORD_IO_FAILED,
                          memory_order_release);
    return 0;
}

/* Speculative work may be cancelled only while still queued and only if no
 * blocking waiter has joined/escalated it. */
static inline int coli_record_io_cancel_prefetch(ColiRecordIoHandle *handle) {
    if (!handle || !handle->entry || handle->priority != COLI_RECORD_IO_PREFETCH ||
        atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
            handle->generation ||
        atomic_load_explicit(&handle->entry->blocking_waiters,
                             memory_order_acquire) != 0)
        return 0;
    int expected = COLI_RECORD_IO_QUEUED;
    return atomic_compare_exchange_strong_explicit(
        &handle->entry->state, &expected, COLI_RECORD_IO_CANCELLED,
        memory_order_acq_rel, memory_order_acquire) ? 1 : 0;
}

static inline int coli_record_io_release(ColiRecordIoHandle *handle) {
    if (!handle || !handle->entry || !handle->generation ||
        !coli_record_io_key_equal(handle->entry->key, handle->key) ||
        atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
            handle->generation)
        return -1;
    unsigned waiters = atomic_load_explicit(&handle->entry->waiters,
                                            memory_order_acquire);
    for (;;) {
        if (!waiters) return -1;
        if (atomic_compare_exchange_weak_explicit(
                &handle->entry->waiters, &waiters, waiters - 1,
                memory_order_acq_rel, memory_order_acquire))
            break;
    }
    if (handle->priority == COLI_RECORD_IO_BLOCKING) {
        unsigned blockers = atomic_load_explicit(&handle->entry->blocking_waiters,
                                                  memory_order_acquire);
        for (;;) {
            if (!blockers) return -1;
            if (atomic_compare_exchange_weak_explicit(
                    &handle->entry->blocking_waiters, &blockers, blockers - 1,
                    memory_order_acq_rel, memory_order_acquire))
                break;
        }
    }
    memset(handle, 0, sizeof(*handle));
    return 0;
}

#endif

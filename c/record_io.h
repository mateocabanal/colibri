#ifndef COLIBRI_RECORD_IO_H
#define COLIBRI_RECORD_IO_H

#include <limits.h>
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
    /* Internal publication states. STARTING keeps joiners away until the new
     * generation/counters are initialized. FINISHING serializes competing
     * terminal callbacks before READY/FAILED becomes visible. */
    COLI_RECORD_IO_STARTING = 6,
    COLI_RECORD_IO_FINISHING = 7,
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
    /* OWNER/JOIN participate in the in-flight waiter count and must release.
     * READY hits do not, so releasing a hit is a harmless handle clear. */
    int retained;
} ColiRecordIoHandle;

/* UINT_MAX is never a real blocker count. Cancellation temporarily owns this
 * value so a blocking join cannot cross the zero-blocker decision boundary. */
#define COLI_RECORD_IO_BLOCKER_CANCEL_LOCK UINT_MAX

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

static inline void coli_record_io_fill_handle(ColiRecordIoHandle *handle,
                                               ColiRecordIoEntry *entry,
                                               uint64_t generation,
                                               ColiRecordIoPriority priority,
                                               int retained) {
    memset(handle, 0, sizeof(*handle));
    handle->entry = entry;
    handle->key = entry->key;
    handle->generation = generation;
    handle->priority = priority;
    handle->retained = retained;
}

static inline int coli_record_io_blocker_retain(ColiRecordIoEntry *entry) {
    if (!entry) return -1;
    unsigned current = atomic_load_explicit(&entry->blocking_waiters,
                                             memory_order_acquire);
    for (;;) {
        if (current == COLI_RECORD_IO_BLOCKER_CANCEL_LOCK) return 0;
        if (current == COLI_RECORD_IO_BLOCKER_CANCEL_LOCK - 1u) return -1;
        if (atomic_compare_exchange_weak_explicit(
                &entry->blocking_waiters, &current, current + 1u,
                memory_order_acq_rel, memory_order_acquire))
            return 1;
    }
}

static inline int coli_record_io_blocker_release(ColiRecordIoEntry *entry) {
    if (!entry) return -1;
    unsigned current = atomic_load_explicit(&entry->blocking_waiters,
                                             memory_order_acquire);
    for (;;) {
        if (!current || current == COLI_RECORD_IO_BLOCKER_CANCEL_LOCK) return -1;
        if (atomic_compare_exchange_weak_explicit(
                &entry->blocking_waiters, &current, current - 1u,
                memory_order_acq_rel, memory_order_acquire))
            return 0;
    }
}

/* The generation identifies one physical read/prepare attempt. One contender
 * owns IDLE/FAILED/CANCELLED -> STARTING, initializes the complete attempt, then
 * release-publishes QUEUED. Joiners never observe half-initialized generation or
 * waiter fields. Failed/cancelled attempts cannot advance until all retained
 * handles from the previous generation have been reaped. */
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
            coli_record_io_fill_handle(handle, entry, generation, priority, 0);
            return COLI_RECORD_IO_HIT;
        }
        if (state == COLI_RECORD_IO_STARTING)
            continue;
        if (state == COLI_RECORD_IO_QUEUED ||
            state == COLI_RECORD_IO_READING ||
            state == COLI_RECORD_IO_FINISHING) {
            int blocker_retained = 0;
            if (priority == COLI_RECORD_IO_BLOCKING) {
                blocker_retained = coli_record_io_blocker_retain(entry);
                if (blocker_retained == 0) continue;
                if (blocker_retained < 0) return COLI_RECORD_IO_INVALID;
            }
            coli_record_io_escalate(entry, priority);
            atomic_fetch_add_explicit(&entry->waiters, 1, memory_order_acq_rel);
            uint64_t generation = atomic_load_explicit(
                &entry->generation, memory_order_acquire);
            int after = atomic_load_explicit(&entry->state, memory_order_acquire);
            if (!generation ||
                (after != COLI_RECORD_IO_QUEUED &&
                 after != COLI_RECORD_IO_READING &&
                 after != COLI_RECORD_IO_FINISHING)) {
                atomic_fetch_sub_explicit(&entry->waiters, 1,
                                          memory_order_acq_rel);
                if (blocker_retained > 0)
                    (void)coli_record_io_blocker_release(entry);
                continue;
            }
            coli_record_io_fill_handle(handle, entry, generation, priority, 1);
            return COLI_RECORD_IO_JOIN;
        }
        if (state != COLI_RECORD_IO_IDLE && state != COLI_RECORD_IO_FAILED &&
            state != COLI_RECORD_IO_CANCELLED)
            return COLI_RECORD_IO_INVALID;

        if (atomic_load_explicit(&entry->waiters, memory_order_acquire) != 0 ||
            atomic_load_explicit(&entry->blocking_waiters, memory_order_acquire) != 0)
            return COLI_RECORD_IO_INVALID;

        int expected = state;
        if (!atomic_compare_exchange_weak_explicit(
                &entry->state, &expected, COLI_RECORD_IO_STARTING,
                memory_order_acq_rel, memory_order_acquire))
            continue;

        /* No new joiner can retain STARTING. Recheck catches any old handle that
         * had not actually been fully reaped before the claim. */
        if (atomic_load_explicit(&entry->waiters, memory_order_acquire) != 0 ||
            atomic_load_explicit(&entry->blocking_waiters, memory_order_acquire) != 0) {
            atomic_store_explicit(&entry->state, state, memory_order_release);
            return COLI_RECORD_IO_INVALID;
        }

        uint64_t previous = atomic_fetch_add_explicit(
            &entry->generation, 1, memory_order_acq_rel);
        if (previous == UINT64_MAX) {
            atomic_store_explicit(&entry->generation, UINT64_MAX,
                                  memory_order_release);
            atomic_store_explicit(&entry->state, state, memory_order_release);
            return COLI_RECORD_IO_INVALID;
        }
        uint64_t generation = previous + 1;
        atomic_store_explicit(&entry->priority, priority, memory_order_relaxed);
        atomic_store_explicit(&entry->waiters, 1, memory_order_relaxed);
        atomic_store_explicit(&entry->blocking_waiters,
                              priority == COLI_RECORD_IO_BLOCKING ? 1u : 0u,
                              memory_order_relaxed);
        atomic_store_explicit(&entry->stored_bytes, 0, memory_order_relaxed);
        atomic_store_explicit(&entry->error_code, 0, memory_order_relaxed);
        atomic_store_explicit(&entry->state, COLI_RECORD_IO_QUEUED,
                              memory_order_release);
        coli_record_io_fill_handle(handle, entry, generation, priority, 1);
        return COLI_RECORD_IO_OWNER;
    }
}

static inline int coli_record_io_begin_read(ColiRecordIoHandle *handle) {
    if (!handle || !handle->entry || !handle->generation || !handle->retained ||
        atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
            handle->generation)
        return -1;
    int expected = COLI_RECORD_IO_QUEUED;
    return atomic_compare_exchange_strong_explicit(
        &handle->entry->state, &expected, COLI_RECORD_IO_READING,
        memory_order_acq_rel, memory_order_acquire) ? 0 : -1;
}

/* Claim FINISHING before publishing terminal metadata. Exactly one completion
 * callback for the generation can win, so READY and FAILED cannot overwrite one
 * another after check-then-store races. */
static inline int coli_record_io_complete(ColiRecordIoHandle *handle,
                                          uint64_t stored_bytes) {
    if (!handle || !handle->entry || !handle->generation || !handle->retained ||
        !stored_bytes ||
        atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
            handle->generation)
        return -1;
    int expected = COLI_RECORD_IO_READING;
    if (!atomic_compare_exchange_strong_explicit(
            &handle->entry->state, &expected, COLI_RECORD_IO_FINISHING,
            memory_order_acq_rel, memory_order_acquire))
        return -1;
    atomic_store_explicit(&handle->entry->stored_bytes, stored_bytes,
                          memory_order_relaxed);
    atomic_store_explicit(&handle->entry->error_code, 0, memory_order_relaxed);
    atomic_store_explicit(&handle->entry->state, COLI_RECORD_IO_READY,
                          memory_order_release);
    return 0;
}

static inline int coli_record_io_fail(ColiRecordIoHandle *handle,
                                      int error_code) {
    if (!handle || !handle->entry || !handle->generation || !handle->retained ||
        atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
            handle->generation)
        return -1;
    int expected = COLI_RECORD_IO_QUEUED;
    if (!atomic_compare_exchange_strong_explicit(
            &handle->entry->state, &expected, COLI_RECORD_IO_FINISHING,
            memory_order_acq_rel, memory_order_acquire)) {
        expected = COLI_RECORD_IO_READING;
        if (!atomic_compare_exchange_strong_explicit(
                &handle->entry->state, &expected, COLI_RECORD_IO_FINISHING,
                memory_order_acq_rel, memory_order_acquire))
            return -1;
    }
    atomic_store_explicit(&handle->entry->stored_bytes, 0, memory_order_relaxed);
    atomic_store_explicit(&handle->entry->error_code, error_code,
                          memory_order_relaxed);
    atomic_store_explicit(&handle->entry->state, COLI_RECORD_IO_FAILED,
                          memory_order_release);
    return 0;
}

/* Speculative work may be cancelled only while still queued and only if no
 * blocking waiter has joined/escalated it. The temporary UINT_MAX blocker lock
 * makes the zero-blocker decision atomic with respect to new blocking joins. */
static inline int coli_record_io_cancel_prefetch(ColiRecordIoHandle *handle) {
    if (!handle || !handle->entry || !handle->retained ||
        handle->priority != COLI_RECORD_IO_PREFETCH ||
        atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
            handle->generation)
        return 0;

    unsigned zero = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &handle->entry->blocking_waiters, &zero,
            COLI_RECORD_IO_BLOCKER_CANCEL_LOCK,
            memory_order_acq_rel, memory_order_acquire))
        return 0;

    int expected = COLI_RECORD_IO_QUEUED;
    int cancelled = atomic_compare_exchange_strong_explicit(
        &handle->entry->state, &expected, COLI_RECORD_IO_CANCELLED,
        memory_order_acq_rel, memory_order_acquire) ? 1 : 0;
    atomic_store_explicit(&handle->entry->blocking_waiters, 0,
                          memory_order_release);
    return cancelled;
}

static inline int coli_record_io_release(ColiRecordIoHandle *handle) {
    if (!handle || !handle->entry || !handle->generation ||
        !coli_record_io_key_equal(handle->entry->key, handle->key))
        return -1;
    if (!handle->retained) {
        memset(handle, 0, sizeof(*handle));
        return 0;
    }
    if (atomic_load_explicit(&handle->entry->generation, memory_order_acquire) !=
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
    if (handle->priority == COLI_RECORD_IO_BLOCKING &&
        coli_record_io_blocker_release(handle->entry) != 0)
        return -1;
    memset(handle, 0, sizeof(*handle));
    return 0;
}

#endif

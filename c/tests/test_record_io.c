#include "../record_io.h"

#include <stdio.h>

static int test_existing_lifecycle(void) {
    ColiRecordIoKey key = {0x1234, 0x5678, 99};
    ColiRecordIoEntry entry;
    ColiRecordIoHandle owner = {0}, join = {0}, hit = {0}, stale = {0};

    if (coli_record_io_entry_init(&entry, key) != 0) return 1;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH, &owner) !=
            COLI_RECORD_IO_OWNER || owner.generation != 1 || !owner.retained ||
        !owner.owner || atomic_load(&entry.priority) != COLI_RECORD_IO_PREFETCH)
        return 2;

    stale = owner;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &join) !=
            COLI_RECORD_IO_JOIN || join.generation != owner.generation ||
        !join.retained || join.owner ||
        atomic_load(&entry.priority) != COLI_RECORD_IO_BLOCKING ||
        atomic_load(&entry.waiters) != 2 ||
        atomic_load(&entry.blocking_waiters) != 1)
        return 3;

    /* A JOIN is a waiter only. It cannot take over or publish the physical I/O
     * attempt even if it races the owner before the backend read starts. */
    if (coli_record_io_begin_read(&join) == 0 ||
        coli_record_io_complete(&join, 1) == 0 ||
        coli_record_io_fail(&join, 77) == 0 ||
        coli_record_io_cancel_prefetch(&join) != 0)
        return 4;

    if (coli_record_io_cancel_prefetch(&owner) != 0) return 5;
    if (coli_record_io_begin_read(&owner) != 0 ||
        coli_record_io_complete(&owner, 4096) != 0 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_READY ||
        atomic_load(&entry.stored_bytes) != 4096)
        return 6;
    if (coli_record_io_fail(&owner, 88) == 0 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_READY ||
        atomic_load(&entry.stored_bytes) != 4096)
        return 7;

    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &hit) !=
            COLI_RECORD_IO_HIT || hit.generation != 1 || hit.retained || hit.owner ||
        hit.request_started_ns != 0 || coli_record_io_exposed_wait_ns(&hit) != 0)
        return 8;
    if (coli_record_io_release(&hit) != 0 || hit.entry != NULL ||
        atomic_load(&entry.waiters) != 2)
        return 9;
    if (coli_record_io_release(&join) != 0 ||
        coli_record_io_release(&owner) != 0 ||
        atomic_load(&entry.waiters) != 0 ||
        atomic_load(&entry.blocking_waiters) != 0)
        return 10;

    atomic_store(&entry.state, COLI_RECORD_IO_FAILED);
    ColiRecordIoHandle retry = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &retry) !=
            COLI_RECORD_IO_OWNER || retry.generation != 2 || !retry.owner)
        return 11;
    if (coli_record_io_complete(&stale, 123) == 0) return 12;
    if (coli_record_io_begin_read(&retry) != 0 ||
        coli_record_io_fail(&retry, 5) != 0 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_FAILED ||
        atomic_load(&entry.error_code) != 5 ||
        atomic_load(&entry.stored_bytes) != 0)
        return 13;
    if (coli_record_io_complete(&retry, 123) == 0 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_FAILED)
        return 14;
    ColiRecordIoHandle premature = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH, &premature) !=
            COLI_RECORD_IO_INVALID ||
        atomic_load(&entry.generation) != 2)
        return 15;
    if (coli_record_io_release(&retry) != 0 ||
        atomic_load(&entry.waiters) != 0)
        return 16;

    ColiRecordIoHandle prefetch = {0}, prefetch_join = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH, &prefetch) !=
            COLI_RECORD_IO_OWNER || prefetch.generation != 3 || !prefetch.owner)
        return 17;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH, &prefetch_join) !=
            COLI_RECORD_IO_JOIN || prefetch_join.owner ||
        coli_record_io_cancel_prefetch(&prefetch_join) != 0)
        return 18;
    if (coli_record_io_release(&prefetch_join) != 0 ||
        coli_record_io_cancel_prefetch(&prefetch) != 1 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_CANCELLED)
        return 19;
    if (coli_record_io_release(&prefetch) != 0 ||
        atomic_load(&entry.waiters) != 0)
        return 20;

    ColiRecordIoHandle cancelled_owner = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH,
                               &cancelled_owner) != COLI_RECORD_IO_OWNER ||
        cancelled_owner.generation != 4 || !cancelled_owner.owner ||
        coli_record_io_cancel_prefetch(&cancelled_owner) != 1)
        return 21;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &premature) !=
            COLI_RECORD_IO_INVALID || atomic_load(&entry.generation) != 4)
        return 22;
    if (coli_record_io_release(&cancelled_owner) != 0) return 23;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &retry) !=
            COLI_RECORD_IO_OWNER || retry.generation != 5 || !retry.owner)
        return 24;
    if (coli_record_io_release(&retry) != 0) return 25;
    return 0;
}

static int test_prefetch_hidden_timing(void) {
    ColiRecordIoKey key = {0x1111, 0x2222, 1};
    ColiRecordIoEntry entry;
    ColiRecordIoHandle prefetch = {0}, join = {0}, hit = {0};
    if (coli_record_io_entry_init(&entry, key) != 0) return 30;

    /* Prefetch owns the physical work at t=100. A user-visible blocking request
     * arrives later at t=150, so only 100ns of the 150ns end-to-end prefetch is
     * exposed to that caller. */
    if (coli_record_io_request_at(
            &entry, COLI_RECORD_IO_PREFETCH, 100, &prefetch) !=
            COLI_RECORD_IO_OWNER || prefetch.request_started_ns != 0)
        return 31;
    if (coli_record_io_begin_read_at(&prefetch, 120) != 0)
        return 32;
    if (coli_record_io_request_at(
            &entry, COLI_RECORD_IO_BLOCKING, 150, &join) !=
            COLI_RECORD_IO_JOIN || join.request_started_ns != 150)
        return 33;
    if (coli_record_io_complete_at(&prefetch, 4096, 250) != 0)
        return 34;
    if (coli_record_io_physical_load_ns(&prefetch) != 130 ||
        coli_record_io_exposed_wait_ns(&prefetch) != 0 ||
        coli_record_io_exposed_wait_ns(&join) != 100)
        return 35;

    if (coli_record_io_request_at(
            &entry, COLI_RECORD_IO_BLOCKING, 300, &hit) !=
            COLI_RECORD_IO_HIT || hit.request_started_ns != 0 ||
        coli_record_io_exposed_wait_ns(&hit) != 0)
        return 36;
    if (coli_record_io_release(&hit) != 0 ||
        coli_record_io_release(&join) != 0 ||
        coli_record_io_release(&prefetch) != 0)
        return 37;
    return 0;
}

static int test_blocking_owner_timing(void) {
    ColiRecordIoKey key = {0x3333, 0x4444, 2};
    ColiRecordIoEntry entry;
    ColiRecordIoHandle owner = {0};
    if (coli_record_io_entry_init(&entry, key) != 0) return 40;
    if (coli_record_io_request_at(
            &entry, COLI_RECORD_IO_BLOCKING, 1000, &owner) !=
            COLI_RECORD_IO_OWNER || owner.request_started_ns != 1000)
        return 41;
    if (coli_record_io_begin_read_at(&owner, 1100) != 0 ||
        coli_record_io_complete_at(&owner, 8192, 1400) != 0)
        return 42;
    if (coli_record_io_physical_load_ns(&owner) != 300 ||
        coli_record_io_exposed_wait_ns(&owner) != 400)
        return 43;
    if (coli_record_io_release(&owner) != 0) return 44;
    return 0;
}

int main(void) {
    int rc = test_existing_lifecycle();
    if (rc) return rc;
    rc = test_prefetch_hidden_timing();
    if (rc) return rc;
    rc = test_blocking_owner_timing();
    if (rc) return rc;
    puts("shared compiled-record I/O state + timing: ok");
    return 0;
}

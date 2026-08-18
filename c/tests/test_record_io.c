#include "../record_io.h"

#include <stdio.h>

int main(void) {
    ColiRecordIoKey key = {0x1234, 0x5678, 99};
    ColiRecordIoEntry entry;
    ColiRecordIoHandle owner = {0}, join = {0}, hit = {0}, stale = {0};

    if (coli_record_io_entry_init(&entry, key) != 0) return 1;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH, &owner) !=
            COLI_RECORD_IO_OWNER || owner.generation != 1 || !owner.retained ||
        atomic_load(&entry.priority) != COLI_RECORD_IO_PREFETCH)
        return 2;

    stale = owner;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &join) !=
            COLI_RECORD_IO_JOIN || join.generation != owner.generation ||
        !join.retained ||
        atomic_load(&entry.priority) != COLI_RECORD_IO_BLOCKING ||
        atomic_load(&entry.waiters) != 2 ||
        atomic_load(&entry.blocking_waiters) != 1)
        return 3;

    /* Once a real blocking miss joins, speculative cancellation may not kill it. */
    if (coli_record_io_cancel_prefetch(&owner) != 0) return 4;
    if (coli_record_io_begin_read(&owner) != 0 ||
        coli_record_io_complete(&owner, 4096) != 0 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_READY ||
        atomic_load(&entry.stored_bytes) != 4096)
        return 5;

    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &hit) !=
            COLI_RECORD_IO_HIT || hit.generation != 1 || hit.retained)
        return 6;
    /* READY hits never joined the waiter count; release is a harmless clear. */
    if (coli_record_io_release(&hit) != 0 || hit.entry != NULL ||
        atomic_load(&entry.waiters) != 2)
        return 7;
    if (coli_record_io_release(&join) != 0 ||
        coli_record_io_release(&owner) != 0 ||
        atomic_load(&entry.waiters) != 0 ||
        atomic_load(&entry.blocking_waiters) != 0)
        return 8;

    /* Reset to a failed attempt. Retry must not advance generation until the
     * failed owner's retained handle is reaped. */
    atomic_store(&entry.state, COLI_RECORD_IO_FAILED);
    ColiRecordIoHandle retry = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &retry) !=
            COLI_RECORD_IO_OWNER || retry.generation != 2)
        return 9;
    if (coli_record_io_complete(&stale, 123) == 0) return 10;
    if (coli_record_io_begin_read(&retry) != 0 ||
        coli_record_io_fail(&retry, 5) != 0 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_FAILED ||
        atomic_load(&entry.error_code) != 5)
        return 11;
    ColiRecordIoHandle premature = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH, &premature) !=
            COLI_RECORD_IO_INVALID ||
        atomic_load(&entry.generation) != 2)
        return 12;
    if (coli_record_io_release(&retry) != 0 ||
        atomic_load(&entry.waiters) != 0)
        return 13;

    /* Pure prefetch can still be retried and cancelled once the old generation
     * has been fully reaped. */
    ColiRecordIoHandle prefetch = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH, &prefetch) !=
            COLI_RECORD_IO_OWNER || prefetch.generation != 3 ||
        coli_record_io_cancel_prefetch(&prefetch) != 1 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_CANCELLED)
        return 14;
    if (coli_record_io_release(&prefetch) != 0 ||
        atomic_load(&entry.waiters) != 0)
        return 15;

    /* Cancelled generations obey the same reap-before-retry rule. */
    ColiRecordIoHandle cancelled_owner = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH,
                               &cancelled_owner) != COLI_RECORD_IO_OWNER ||
        cancelled_owner.generation != 4 ||
        coli_record_io_cancel_prefetch(&cancelled_owner) != 1)
        return 16;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &premature) !=
            COLI_RECORD_IO_INVALID || atomic_load(&entry.generation) != 4)
        return 17;
    if (coli_record_io_release(&cancelled_owner) != 0) return 18;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &retry) !=
            COLI_RECORD_IO_OWNER || retry.generation != 5)
        return 19;
    if (coli_record_io_release(&retry) != 0) return 20;

    puts("shared compiled-record I/O state: ok");
    return 0;
}

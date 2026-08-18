#include "../record_io.h"

#include <stdio.h>

int main(void) {
    ColiRecordIoKey key = {0x1234, 0x5678, 99};
    ColiRecordIoEntry entry;
    ColiRecordIoHandle owner = {0}, join = {0}, hit = {0}, stale = {0};

    if (coli_record_io_entry_init(&entry, key) != 0) return 1;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH, &owner) !=
            COLI_RECORD_IO_OWNER || owner.generation != 1 ||
        atomic_load(&entry.priority) != COLI_RECORD_IO_PREFETCH)
        return 2;

    stale = owner;
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &join) !=
            COLI_RECORD_IO_JOIN || join.generation != owner.generation ||
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
            COLI_RECORD_IO_HIT || hit.generation != 1)
        return 6;
    if (coli_record_io_release(&join) != 0 ||
        coli_record_io_release(&owner) != 0)
        return 7;

    /* Reset to a failed attempt, then prove a new owner advances generation and
     * stale async completions from generation 1 cannot publish into generation 2. */
    atomic_store(&entry.state, COLI_RECORD_IO_FAILED);
    ColiRecordIoHandle retry = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_BLOCKING, &retry) !=
            COLI_RECORD_IO_OWNER || retry.generation != 2)
        return 8;
    if (coli_record_io_complete(&stale, 123) == 0) return 9;
    if (coli_record_io_begin_read(&retry) != 0 ||
        coli_record_io_fail(&retry, 5) != 0 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_FAILED ||
        atomic_load(&entry.error_code) != 5)
        return 10;
    if (coli_record_io_release(&retry) != 0) return 11;

    /* Pure prefetch can still be cancelled before I/O begins. */
    ColiRecordIoHandle prefetch = {0};
    if (coli_record_io_request(&entry, COLI_RECORD_IO_PREFETCH, &prefetch) !=
            COLI_RECORD_IO_OWNER || prefetch.generation != 3 ||
        coli_record_io_cancel_prefetch(&prefetch) != 1 ||
        atomic_load(&entry.state) != COLI_RECORD_IO_CANCELLED)
        return 12;
    if (coli_record_io_release(&prefetch) != 0) return 13;

    puts("shared compiled-record I/O state: ok");
    return 0;
}

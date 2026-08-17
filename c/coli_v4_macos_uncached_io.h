#ifndef COLI_V4_MACOS_UNCACHED_IO_H
#define COLI_V4_MACOS_UNCACHED_IO_H

/*
 * The legacy V4 path used F_NOCACHE only for large transient expert payloads;
 * dense/static/head reads stayed buffered.  Applying F_NOCACHE to every CSF
 * descriptor destroys readahead for the dozens of synchronous dense records
 * loaded before each layer and can make the package path latency-bound.
 *
 * coli_v4_expert_store.c marks only its loader threads with the TLS flag below.
 * This translation-unit-local pread interposition then keeps F_NOCACHE enabled
 * on a descriptor while one or more expert reads are in flight.  The per-fd
 * reference count matters because the V4 loader issues several preads in
 * parallel against the same shard.  Once the last expert read finishes the fd
 * immediately returns to ordinary buffered semantics for the next dense layer.
 *
 * COLI_V4_DIRECT=0 is implemented by the expert store not setting the TLS flag,
 * preserving the existing A/B diagnostic contract.
 */
#ifdef __APPLE__
#include <fcntl.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

extern __thread int coli_v4_expert_io_active;

typedef struct ColiV4NoCacheFd {
    int fd;
    unsigned refs;
} ColiV4NoCacheFd;

/* A COLI package has a small bounded shard set.  128 entries avoids allocation
 * in the read hot path while leaving ample room for large compiled packages. */
static ColiV4NoCacheFd coli_v4_nocache_fds[128];
static pthread_mutex_t coli_v4_nocache_mutex = PTHREAD_MUTEX_INITIALIZER;

static ColiV4NoCacheFd *coli_v4_nocache_slot(int fd) {
    ColiV4NoCacheFd *empty = NULL;
    for (size_t i = 0; i < sizeof(coli_v4_nocache_fds) / sizeof(coli_v4_nocache_fds[0]); i++) {
        ColiV4NoCacheFd *slot = &coli_v4_nocache_fds[i];
        if (slot->refs && slot->fd == fd) return slot;
        if (!slot->refs && !empty) empty = slot;
    }
    return empty;
}

static void coli_v4_nocache_begin(int fd) {
    pthread_mutex_lock(&coli_v4_nocache_mutex);
    ColiV4NoCacheFd *slot = coli_v4_nocache_slot(fd);
    if (slot) {
        if (!slot->refs) {
            slot->fd = fd;
            (void)fcntl(fd, F_NOCACHE, 1);
        }
        slot->refs++;
    }
    pthread_mutex_unlock(&coli_v4_nocache_mutex);
}

static void coli_v4_nocache_end(int fd) {
    pthread_mutex_lock(&coli_v4_nocache_mutex);
    for (size_t i = 0; i < sizeof(coli_v4_nocache_fds) / sizeof(coli_v4_nocache_fds[0]); i++) {
        ColiV4NoCacheFd *slot = &coli_v4_nocache_fds[i];
        if (slot->refs && slot->fd == fd) {
            if (--slot->refs == 0) {
                (void)fcntl(fd, F_NOCACHE, 0);
                slot->fd = -1;
            }
            break;
        }
    }
    pthread_mutex_unlock(&coli_v4_nocache_mutex);
}

static inline ssize_t coli_v4_scoped_pread(int fd, void *buf, size_t n, off_t off) {
    if (!coli_v4_expert_io_active) return pread(fd, buf, n, off);
    coli_v4_nocache_begin(fd);
    ssize_t result = pread(fd, buf, n, off);
    coli_v4_nocache_end(fd);
    return result;
}

/* Defined after coli_v4_scoped_pread so its internal pread calls are not recursive. */
#define pread(fd, buf, n, off) coli_v4_scoped_pread((fd), (buf), (n), (off))
#endif

#endif /* COLI_V4_MACOS_UNCACHED_IO_H */

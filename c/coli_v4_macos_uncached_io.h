#ifndef COLI_V4_MACOS_UNCACHED_IO_H
#define COLI_V4_MACOS_UNCACHED_IO_H

/*
 * DeepSeek-V4 streams routed experts as large, transient reads.  The legacy
 * safetensors path opened a separate F_NOCACHE fd for those reads, which is
 * important on memory-constrained Apple Silicon: otherwise tens of GiB of
 * one-shot expert traffic churn the unified buffer cache and sustained SSD
 * throughput collapses.
 *
 * CSF already opens a direct twin, but the v1 record read API currently uses
 * the ordinary fd.  Until the CSF API grows an explicit streaming read, make
 * the ordinary fds opened by the V4 coli_format.o uncached on macOS.  This is
 * intentionally a V4 build-local interposition; generic CSF tools keep their
 * normal buffered semantics, and Linux/Windows are untouched.
 *
 * Keep the existing V4 direct-I/O debugging contract: COLI_V4_DIRECT=0 opts
 * out, which also makes A/B diagnosis of this regression straightforward.
 */
#ifdef __APPLE__
#include <fcntl.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

static inline int coli_v4_uncached_open(const char *path, int flags) {
    int fd = open(path, flags);
    const char *setting = getenv("COLI_V4_DIRECT");
    if (fd >= 0 && (!setting || atoi(setting) != 0))
        (void)fcntl(fd, F_NOCACHE, 1);
    return fd;
}

/* coli_format.c only opens model/package files read-only with two arguments. */
#define open(path, flags) coli_v4_uncached_open((path), (flags))
#endif

#endif /* COLI_V4_MACOS_UNCACHED_IO_H */

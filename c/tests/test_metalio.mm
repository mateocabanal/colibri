/* test_metalio.mm — lifecycle test for the MTLIO expert-streaming subsystem.
 *
 * Exercises the real async path: init -> file wrap -> persistent slot ->
 * non-blocking load -> event wait -> byte-exact contents, plus multiple
 * outstanding loads, slot reuse, prefetch accounting, and clean shutdown
 * with outstanding IO. Skips (exit 0) when MetalIO is unavailable. */
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#include "../metalio.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; \
} } while (0)

static char g_path[512];

/* position-seeded payload: byte(i) = splitmix64(i) & 0xFF — O(1) per byte,
 * so the loader check can verify any range without replaying the stream. */
static uint64_t splitmix64(uint64_t x){
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static unsigned char payload_byte(size_t i){
    return (unsigned char)(splitmix64((uint64_t)i) >> 24);
}

static int make_file(size_t bytes){
    snprintf(g_path, sizeof(g_path), "/tmp/metalio_test_%d.bin", (int)getpid());
    int fd = open(g_path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (fd < 0) return -1;
    unsigned char buf[4096];
    for (size_t off = 0; off < bytes; off += sizeof(buf)) {
        size_t n = bytes - off < sizeof(buf) ? bytes - off : sizeof(buf);
        for (size_t i = 0; i < n; i++) buf[i] = payload_byte(off + i);
        if (write(fd, buf, n) != (ssize_t)n) { close(fd); return -1; }
    }
    return fd;
}

int main(void){
    @autoreleasepool {
        if (!metalio_init()) {
            printf("metalio: unavailable on this system — skipping\n");
            return 0;
        }
        metalio_verbose(1);

        /* --- file + one slot, load + wait, byte-exact -------------------- */
        const size_t F = 1u << 20;         /* 1 MiB file */
        int fd = make_file(F);
        CHECK(fd >= 0, "make_file");
        int file = metalio_file_add(g_path);
        CHECK(file >= 0, "file_add");
        int slot = metalio_slot_alloc(256u << 10);
        CHECK(slot >= 0, "slot_alloc");
        CHECK(metalio_slot_bytes(slot) >= (256u << 10), "slot size");
        int64_t ev = metalio_load(slot, file, 4096, 128u << 10);
        CHECK(ev > 0, "load event");
        CHECK(metalio_wait(ev) == 0, "wait");
        const unsigned char *p = (const unsigned char *)metalio_slot_ptr(slot);
        CHECK(p != NULL, "slot ptr");
        int bad = 0;
        for (size_t i = 0; i < 128u << 10; i++) {
            if (p[i] != payload_byte(4096 + i)) { bad = 1; break; }
        }
        CHECK(!bad, "loaded bytes match file contents at offset 4096");

        /* --- multiple outstanding loads ---------------------------------- */
        int64_t ev2 = metalio_load(slot, file, 0, 256u << 10);
        int64_t ev3 = metalio_load(slot, file, F - 1024, 1024);
        CHECK(ev2 > ev && ev3 > ev2, "event values increase");
        CHECK(metalio_wait(ev3) == 0, "wait covers earlier loads");

        /* --- slot reuse: overwrite with a different range ---------------- */
        int64_t ev4 = metalio_load(slot, file, 65536, 4096);
        CHECK(metalio_wait(ev4) == 0, "reuse wait");
        CHECK(metalio_slot_ptr(slot) != NULL, "reuse ptr");

        /* --- prefetch accounting ------------------------------------------ */
        metalio_slot_consumed(slot);
        metalio_prefetch_done(slot);
        ColiMetalioStats st;
        metalio_stats(&st);
        CHECK(st.loads >= 4, "load counter: %llu", (unsigned long long)st.loads);
        CHECK(st.waits >= 3, "wait counter: %llu", (unsigned long long)st.waits);
        CHECK(st.bytes >= (128u << 10) + (256u << 10) + 1024 + 4096, "byte counter");
        CHECK(st.latency_samples >= 3, "latency samples: %llu", (unsigned long long)st.latency_samples);
        CHECK(st.peak_outstanding >= 2, "peak outstanding: %llu", (unsigned long long)st.peak_outstanding);

        /* --- shutdown with outstanding IO -------------------------------- */
        int64_t ev5 = metalio_load(slot, file, 0, 1024);
        CHECK(ev5 > 0, "load before shutdown");
        metalio_shutdown();
        CHECK(!metalio_active(), "inactive after shutdown");
        CHECK(metalio_init() == 1, "re-init after shutdown");   /* model reload path */
        metalio_shutdown();

        close(fd);
        unlink(g_path);
    }
    if (failures) { fprintf(stderr, "metalio: %d failure(s)\n", failures); return 1; }
    printf("metalio: init, load, wait, reuse, prefetch accounting, shutdown ok\n");
    return 0;
}

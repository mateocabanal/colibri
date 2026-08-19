#include "../record_io_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char storage[4096];
    size_t bytes;
    int active;
    int file_added;
    int buffer_in_use;
    int64_t next_event;
    ColiRecordIoBackendStats stats;
} FakeIo;

static int fake_init(void *ctx) {
    FakeIo *f = (FakeIo *)ctx;
    f->active = 1;
    f->next_event = 1;
    return 1;
}
static void fake_shutdown(void *ctx) { ((FakeIo *)ctx)->active = 0; }
static int fake_active(void *ctx) { return ((FakeIo *)ctx)->active; }
static ColiRecordIoBackendFile fake_file_add(void *ctx, const char *path) {
    FakeIo *f = (FakeIo *)ctx;
    if (!f->active || !path || !path[0]) return -1;
    f->file_added = 1;
    return 7;
}
static ColiRecordIoBackendBuffer fake_buffer_alloc(void *ctx, size_t bytes) {
    FakeIo *f = (FakeIo *)ctx;
    if (!f->active || !bytes || bytes > sizeof(f->storage)) return -1;
    f->buffer_in_use = 1;
    f->bytes = bytes;
    return 3;
}
static void fake_buffer_free(void *ctx, ColiRecordIoBackendBuffer buffer) {
    FakeIo *f = (FakeIo *)ctx;
    if (buffer == 3) f->buffer_in_use = 0;
}
static void *fake_buffer_ptr(void *ctx, ColiRecordIoBackendBuffer buffer) {
    FakeIo *f = (FakeIo *)ctx;
    return f->buffer_in_use && buffer == 3 ? f->storage : NULL;
}
static size_t fake_buffer_bytes(void *ctx, ColiRecordIoBackendBuffer buffer) {
    FakeIo *f = (FakeIo *)ctx;
    return f->buffer_in_use && buffer == 3 ? f->bytes : 0;
}
static ColiRecordIoBackendEvent fake_submitv(
        void *ctx, ColiRecordIoBackendBuffer buffer,
        const ColiRecordIoBackendRegion *regions, int count,
        ColiRecordIoBackendIntent intent) {
    FakeIo *f = (FakeIo *)ctx;
    if (!f->active || !f->buffer_in_use || buffer != 3 || count < 1)
        return -1;
    for (int i = 0; i < count; i++) {
        const ColiRecordIoBackendRegion *r = &regions[i];
        if (r->file != 7 || !r->bytes || r->dst_off > f->bytes ||
            r->bytes > f->bytes - r->dst_off)
            return -1;
        memset(f->storage + r->dst_off,
               (int)((r->src_off + (uint64_t)i) & 0xffu), r->bytes);
        f->stats.bytes += r->bytes;
    }
    f->stats.submissions++;
    if (intent == COLI_RECORD_IO_INTENT_SPECULATIVE)
        f->stats.speculative_submissions++;
    f->stats.outstanding++;
    if (f->stats.outstanding > f->stats.peak_outstanding)
        f->stats.peak_outstanding = f->stats.outstanding;
    return f->next_event++;
}
static int fake_wait(void *ctx, ColiRecordIoBackendEvent event) {
    FakeIo *f = (FakeIo *)ctx;
    if (!f->active || event <= 0 || event >= f->next_event) return -1;
    f->stats.waits++;
    if (f->stats.outstanding) f->stats.outstanding--;
    f->stats.latency_samples++;
    f->stats.total_latency_s += 0.001;
    return 0;
}
static void fake_consumed(void *ctx, ColiRecordIoBackendBuffer buffer) {
    FakeIo *f = (FakeIo *)ctx;
    if (buffer == 3) f->stats.speculative_consumed++;
}
static void fake_discarded(void *ctx, ColiRecordIoBackendBuffer buffer) {
    FakeIo *f = (FakeIo *)ctx;
    if (buffer == 3) f->stats.speculative_discarded++;
}
static void fake_stats(void *ctx, ColiRecordIoBackendStats *out) {
    *out = ((FakeIo *)ctx)->stats;
}
static void fake_verbose(void *ctx, int on) { (void)ctx; (void)on; }

static const ColiRecordIoBackendOps fake_ops = {
    .name = "fake-async",
    .capabilities = COLI_RECORD_IO_CAP_ASYNC |
                    COLI_RECORD_IO_CAP_VECTORED |
                    COLI_RECORD_IO_CAP_CPU_VISIBLE_BUFFER,
    .init = fake_init,
    .shutdown = fake_shutdown,
    .active = fake_active,
    .file_add = fake_file_add,
    .buffer_alloc = fake_buffer_alloc,
    .buffer_free = fake_buffer_free,
    .buffer_ptr = fake_buffer_ptr,
    .buffer_bytes = fake_buffer_bytes,
    .submitv = fake_submitv,
    .wait = fake_wait,
    .buffer_consumed = fake_consumed,
    .buffer_discarded = fake_discarded,
    .stats = fake_stats,
    .verbose = fake_verbose,
};

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "record I/O backend check failed at line %d: %s\n", \
                __LINE__, #x); \
        return 1; \
    } \
} while (0)

int main(void) {
    FakeIo fake;
    memset(&fake, 0, sizeof(fake));
    ColiRecordIoBackend backend = {&fake_ops, &fake};

    CHECK(coli_record_io_backend_valid(&backend));
    CHECK(strcmp(coli_record_io_backend_name(&backend), "fake-async") == 0);
    CHECK((coli_record_io_backend_capabilities(&backend) &
           COLI_RECORD_IO_CAP_VECTORED) != 0);
    CHECK(!coli_record_io_backend_active(&backend));
    CHECK(coli_record_io_backend_init(&backend) == 1);
    CHECK(coli_record_io_backend_active(&backend));

    ColiRecordIoBackendFile file =
        coli_record_io_backend_file_add(&backend, "/fake/model.coli");
    CHECK(file == 7);
    ColiRecordIoBackendBuffer buffer =
        coli_record_io_backend_buffer_alloc(&backend, 1024);
    CHECK(buffer == 3);
    CHECK(coli_record_io_backend_buffer_bytes(&backend, buffer) == 1024);
    CHECK(coli_record_io_backend_buffer_ptr(&backend, buffer) != NULL);

    ColiRecordIoBackendRegion regions[2] = {
        {file, 0x11, 32, 0},
        {file, 0x22, 16, 64},
    };
    ColiRecordIoBackendEvent e1 = coli_record_io_backend_submitv(
        &backend, buffer, regions, 2, COLI_RECORD_IO_INTENT_SPECULATIVE);
    CHECK(e1 == 1);
    CHECK(((unsigned char *)coli_record_io_backend_buffer_ptr(
               &backend, buffer))[0] == 0x11);
    CHECK(((unsigned char *)coli_record_io_backend_buffer_ptr(
               &backend, buffer))[64] == 0x23);

    ColiRecordIoBackendEvent e2 = coli_record_io_backend_submit(
        &backend, buffer, file, 0x33, 8, 128, COLI_RECORD_IO_INTENT_ASYNC);
    CHECK(e2 == 2);
    CHECK(coli_record_io_backend_wait(&backend, e2) == 0);
    coli_record_io_backend_buffer_consumed(&backend, buffer);
    coli_record_io_backend_buffer_discarded(&backend, buffer);

    ColiRecordIoBackendStats stats;
    coli_record_io_backend_stats(&backend, &stats);
    CHECK(stats.submissions == 2);
    CHECK(stats.bytes == 56);
    CHECK(stats.waits == 1);
    CHECK(stats.speculative_submissions == 1);
    CHECK(stats.speculative_consumed == 1);
    CHECK(stats.speculative_discarded == 1);
    CHECK(stats.peak_outstanding == 2);
    CHECK(stats.outstanding == 1);

    CHECK(coli_record_io_backend_submitv(
              &backend, buffer, NULL, 0,
              COLI_RECORD_IO_INTENT_DEMAND) ==
          COLI_RECORD_IO_BACKEND_INVALID_EVENT);
    CHECK(coli_record_io_backend_buffer_alloc(&backend, 0) ==
          COLI_RECORD_IO_BACKEND_INVALID_BUFFER);

    coli_record_io_backend_buffer_free(&backend, buffer);
    CHECK(coli_record_io_backend_buffer_ptr(&backend, buffer) == NULL);
    coli_record_io_backend_shutdown(&backend);
    CHECK(!coli_record_io_backend_active(&backend));

    printf("generic record I/O backend: ok\n");
    return 0;
}

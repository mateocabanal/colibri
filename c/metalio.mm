/* metalio.mm — Apple MetalIO provider for ColiRecordIoBackend.
 *
 * Physical responsibility only:
 *   file path -> persistent MTLIOFileHandle
 *   reusable buffer -> shared-storage MTLBuffer
 *   region vector -> one async MTLIOCommandBuffer + completion event
 *
 * record_io.h owns logical request/join/cancel state. Residency policy, MoE
 * routing, expert identity and quantization never enter this file.
 *
 * Apple Silicon uses unified memory, so shared MTLBuffers are simultaneously
 * CPU-visible fallback storage and GPU-visible compute storage. MetalIO removes
 * the CPU pread/copy leg and, more importantly, allows storage reads to overlap
 * compute. Every error remains recoverable by the caller's ordinary I/O path.
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <stdarg.h>
#import <stdatomic.h>

#define COLI_METALIO_IMPLEMENTATION 1
#include "metalio.h"

#define METALIO_ALIGN 16384u
#define METALIO_MAX_FILES 64
#define METALIO_MAX_SLOTS 256

static _Atomic int g_active = 0;
static _Atomic int g_verbose = 0;

static _Atomic uint64_t m_loads, m_bytes, m_waits, m_fails;
static _Atomic uint64_t m_prefetch_loads, m_prefetch_used, m_prefetch_wasted;
static _Atomic uint64_t m_outstanding, m_peak_outstanding;
static _Atomic uint64_t m_lat_samples, m_lat_total_us;
static _Atomic uint64_t m_lat_hist[32];

static id<MTLDevice> g_dev;
static id<MTLIOCommandQueue> g_iq;
static id<MTLSharedEvent> g_ev;
static uint64_t g_ev_val;
static int64_t g_consumed_high;
static NSRecursiveLock *g_lock;

static id<MTLIOFileHandle> g_files[METALIO_MAX_FILES];
static int g_nfiles;

static struct {
    id<MTLBuffer> buf;
    size_t bytes;
    int in_use;
    _Atomic int64_t last_event;
    _Atomic int64_t consumed;
} g_slots[METALIO_MAX_SLOTS];
static int g_nslots;

static void verbose(const char *fmt, ...) {
    if (!atomic_load_explicit(&g_verbose, memory_order_relaxed)) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[metalio] ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static void hist_add(uint64_t us) {
    unsigned bucket = 0;
    while (us > 1 && bucket < 31) {
        us >>= 1;
        bucket++;
    }
    atomic_fetch_add_explicit(&m_lat_hist[bucket], 1, memory_order_relaxed);
}

static void stats_reset(void) {
    atomic_store_explicit(&m_loads, 0, memory_order_relaxed);
    atomic_store_explicit(&m_bytes, 0, memory_order_relaxed);
    atomic_store_explicit(&m_waits, 0, memory_order_relaxed);
    atomic_store_explicit(&m_fails, 0, memory_order_relaxed);
    atomic_store_explicit(&m_prefetch_loads, 0, memory_order_relaxed);
    atomic_store_explicit(&m_prefetch_used, 0, memory_order_relaxed);
    atomic_store_explicit(&m_prefetch_wasted, 0, memory_order_relaxed);
    atomic_store_explicit(&m_outstanding, 0, memory_order_relaxed);
    atomic_store_explicit(&m_peak_outstanding, 0, memory_order_relaxed);
    atomic_store_explicit(&m_lat_samples, 0, memory_order_relaxed);
    atomic_store_explicit(&m_lat_total_us, 0, memory_order_relaxed);
    for (int i = 0; i < 32; i++)
        atomic_store_explicit(&m_lat_hist[i], 0, memory_order_relaxed);
}

int metalio_active(void) {
    return atomic_load_explicit(&g_active, memory_order_relaxed);
}

void metalio_verbose(int on) {
    atomic_store_explicit(&g_verbose, on, memory_order_relaxed);
}

int metalio_init(void) {
    if (metalio_active()) return 1;
    if (@available(macOS 13.0, *)) {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;

        MTLIOCommandQueueDescriptor *desc = [MTLIOCommandQueueDescriptor new];
        const char *depth_env = getenv("MTLIO_DEPTH");
        int depth = depth_env && atoi(depth_env) > 0 ? atoi(depth_env) : 64;
        if (depth > 1024) depth = 1024;
        desc.maxCommandBufferCount = depth;
        desc.priority = MTLIOPriorityHigh;

        NSError *error = nil;
        g_iq = [g_dev newIOCommandQueueWithDescriptor:desc error:&error];
        if (!g_iq) {
            fprintf(stderr, "[metalio] IO queue creation failed: %s\n",
                    error ? error.description.UTF8String : "unknown");
            g_dev = nil;
            return 0;
        }

        g_ev = [g_dev newSharedEvent];
        if (!g_ev) {
            g_iq = nil;
            g_dev = nil;
            return 0;
        }

        g_lock = [NSRecursiveLock new];
        g_ev_val = 1;
        g_consumed_high = 0;
        g_nfiles = 0;
        g_nslots = 0;
        stats_reset();
        atomic_store_explicit(&g_active, 1, memory_order_release);
        verbose("init: device=%s queue_depth=%d",
                g_dev.name.UTF8String ? g_dev.name.UTF8String : "?", depth);
        return 1;
    }
    return 0;
}

void metalio_shutdown(void) {
    if (!metalio_active()) return;
    [g_lock lock];
    if (g_ev && g_ev_val > 1) {
        [g_ev waitUntilSignaledValue:g_ev_val - 1 timeoutMS:UINT64_MAX];
        verbose("shutdown: drained event=%llu",
                (unsigned long long)(g_ev_val - 1));
    }

    for (int i = 0; i < g_nfiles; i++) g_files[i] = nil;
    for (int i = 0; i < g_nslots; i++) {
        g_slots[i].buf = nil;
        g_slots[i].bytes = 0;
        g_slots[i].in_use = 0;
        atomic_store_explicit(&g_slots[i].last_event, 0, memory_order_relaxed);
        atomic_store_explicit(&g_slots[i].consumed, 0, memory_order_relaxed);
    }
    g_nfiles = 0;
    g_nslots = 0;
    g_ev_val = 1;
    g_consumed_high = 0;
    g_iq = nil;
    g_ev = nil;
    g_dev = nil;
    [g_lock unlock];
    g_lock = nil;
    atomic_store_explicit(&g_active, 0, memory_order_release);
}

int metalio_file_add(const char *path) {
    if (!metalio_active() || !path || !path[0]) return -1;
    [g_lock lock];
    int file = -1;
    if (@available(macOS 13.0, *)) {
        if (g_nfiles < METALIO_MAX_FILES) {
            NSString *string = [NSString stringWithUTF8String:path];
            if (string) {
                NSURL *url = [NSURL fileURLWithPath:string];
                NSError *error = nil;
                id<MTLIOFileHandle> handle =
                    [g_dev newIOFileHandleWithURL:url error:&error];
                if (handle) {
                    file = g_nfiles;
                    g_files[g_nfiles++] = handle;
                    verbose("file_add: path=%s id=%d", path, file);
                } else if (error) {
                    fprintf(stderr, "[metalio] file handle failed for %s: %s\n",
                            path, error.description.UTF8String);
                }
            }
        }
    }
    [g_lock unlock];
    return file;
}

int metalio_slot_alloc(size_t max_bytes) {
    if (!metalio_active() || !max_bytes ||
        max_bytes > SIZE_MAX - (METALIO_ALIGN - 1))
        return -1;
    size_t length =
        (max_bytes + METALIO_ALIGN - 1) & ~(size_t)(METALIO_ALIGN - 1);

    [g_lock lock];
    int slot = -1;
    if (@available(macOS 13.0, *)) {
        for (int i = 0; i < g_nslots && slot < 0; i++)
            if (!g_slots[i].in_use) slot = i;
        if (slot < 0 && g_nslots < METALIO_MAX_SLOTS) slot = g_nslots++;

        if (slot >= 0) {
            id<MTLBuffer> buffer =
                [g_dev newBufferWithLength:length
                                    options:MTLResourceStorageModeShared];
            if (buffer) {
                g_slots[slot].buf = buffer;
                g_slots[slot].bytes = length;
                g_slots[slot].in_use = 1;
                atomic_store_explicit(&g_slots[slot].last_event, 0,
                                      memory_order_relaxed);
                atomic_store_explicit(&g_slots[slot].consumed, 0,
                                      memory_order_relaxed);
                verbose("buffer_alloc: id=%d bytes=%zu", slot, length);
            } else {
                if (slot == g_nslots - 1 && !g_slots[slot].in_use) g_nslots--;
                slot = -1;
            }
        }
    }
    [g_lock unlock];
    return slot;
}

void metalio_slot_free(int slot) {
    if (!metalio_active() || slot < 0 || slot >= g_nslots) return;
    [g_lock lock];
    if (!g_slots[slot].in_use) {
        [g_lock unlock];
        return;
    }
    int64_t event = atomic_load_explicit(&g_slots[slot].last_event,
                                         memory_order_acquire);
    if (event > 0 && g_ev)
        [g_ev waitUntilSignaledValue:(uint64_t)event timeoutMS:UINT64_MAX];
    g_slots[slot].buf = nil;
    g_slots[slot].bytes = 0;
    g_slots[slot].in_use = 0;
    atomic_store_explicit(&g_slots[slot].last_event, 0, memory_order_relaxed);
    atomic_store_explicit(&g_slots[slot].consumed, 0, memory_order_relaxed);
    [g_lock unlock];
}

void *metalio_slot_ptr(int slot) {
    if (!metalio_active() || slot < 0 || slot >= g_nslots ||
        !g_slots[slot].in_use)
        return NULL;
    return g_slots[slot].buf ? g_slots[slot].buf.contents : NULL;
}

size_t metalio_slot_bytes(int slot) {
    if (!metalio_active() || slot < 0 || slot >= g_nslots ||
        !g_slots[slot].in_use)
        return 0;
    return g_slots[slot].bytes;
}

static int regions_ok(int slot, const ColiMetalioRegion *regions, int count) {
    if (slot < 0 || slot >= g_nslots || !g_slots[slot].in_use ||
        !regions || count < 1)
        return 0;
    size_t capacity = g_slots[slot].bytes;
    for (int i = 0; i < count; i++) {
        const ColiMetalioRegion *region = &regions[i];
        if (region->file < 0 || region->file >= g_nfiles ||
            !g_files[region->file] || !region->bytes)
            return 0;
        if (region->dst_off > capacity ||
            region->bytes > capacity - (size_t)region->dst_off)
            return 0;
        if (region->src_off > UINT64_MAX - region->bytes) return 0;
    }
    return 1;
}

int64_t metalio_loadv(int slot, const ColiMetalioRegion *regions, int count,
                      ColiMetalioKind kind) {
    if (!metalio_active()) return -1;
    if (kind != MIO_LOAD_DEMAND && kind != MIO_LOAD_ASYNC &&
        kind != MIO_LOAD_SPEC) {
        atomic_fetch_add_explicit(&m_fails, 1, memory_order_relaxed);
        return -1;
    }

    [g_lock lock];
    int64_t event = -1;
    if (@available(macOS 13.0, *)) {
        if (regions_ok(slot, regions, count)) {
            id<MTLIOCommandBuffer> command = [g_iq commandBuffer];
            if (command) {
                for (int i = 0; i < count; i++) {
                    const ColiMetalioRegion *region = &regions[i];
                    [command loadBuffer:g_slots[slot].buf
                                 offset:region->dst_off
                                   size:region->bytes
                           sourceHandle:g_files[region->file]
                      sourceHandleOffset:region->src_off];
                }
                uint64_t value = g_ev_val++;
                [command signalEvent:g_ev value:value];
                [command commit];
                atomic_store_explicit(&g_slots[slot].last_event,
                                      (int64_t)value, memory_order_release);
                event = (int64_t)value;

                atomic_fetch_add_explicit(&m_loads, 1, memory_order_relaxed);
                if (kind == MIO_LOAD_SPEC)
                    atomic_fetch_add_explicit(&m_prefetch_loads, 1,
                                              memory_order_relaxed);
                for (int i = 0; i < count; i++)
                    atomic_fetch_add_explicit(&m_bytes, regions[i].bytes,
                                              memory_order_relaxed);
                uint64_t outstanding =
                    atomic_fetch_add_explicit(&m_outstanding, 1,
                                              memory_order_relaxed) + 1;
                uint64_t peak = atomic_load_explicit(&m_peak_outstanding,
                                                     memory_order_relaxed);
                while (outstanding > peak &&
                       !atomic_compare_exchange_weak_explicit(
                           &m_peak_outstanding, &peak, outstanding,
                           memory_order_relaxed, memory_order_relaxed)) {}
                verbose("submit: buffer=%d regions=%d event=%llu intent=%d",
                        slot, count, (unsigned long long)value, (int)kind);
            }
        }
    }
    [g_lock unlock];

    if (event <= 0)
        atomic_fetch_add_explicit(&m_fails, 1, memory_order_relaxed);
    return event;
}

int64_t metalio_load(int slot, int file, uint64_t offset, size_t bytes) {
    ColiMetalioRegion region = {file, offset, bytes, 0};
    return metalio_loadv(slot, &region, 1, MIO_LOAD_DEMAND);
}

int metalio_wait(int64_t event_value) {
    if (!metalio_active() || event_value <= 0 || !g_ev) return -1;

    [g_lock lock];
    if (event_value <= g_consumed_high) {
        [g_lock unlock];
        return 0;
    }
    if (event_value > (int64_t)g_ev_val - 1) {
        [g_lock unlock];
        return -1;
    }
    [g_lock unlock];

    uint64_t began = mach_absolute_time();
    [g_ev waitUntilSignaledValue:(uint64_t)event_value timeoutMS:UINT64_MAX];

    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    uint64_t elapsed_us =
        (mach_absolute_time() - began) * timebase.numer / timebase.denom / 1000;

    [g_lock lock];
    if (event_value > g_consumed_high) {
        g_consumed_high = event_value;
        uint64_t issued = g_ev_val - 1;
        uint64_t remaining = issued > (uint64_t)g_consumed_high
            ? issued - (uint64_t)g_consumed_high : 0;
        atomic_store_explicit(&m_outstanding, remaining, memory_order_relaxed);
    }
    [g_lock unlock];

    atomic_fetch_add_explicit(&m_waits, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&m_lat_samples, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&m_lat_total_us, elapsed_us, memory_order_relaxed);
    hist_add(elapsed_us);
    verbose("wait: event=%lld latency=%lluus", (long long)event_value,
            (unsigned long long)elapsed_us);
    return 0;
}

void metalio_slot_consumed(int slot) {
    if (!metalio_active() || slot < 0 || slot >= g_nslots ||
        !g_slots[slot].in_use)
        return;
    int64_t event = atomic_load_explicit(&g_slots[slot].last_event,
                                         memory_order_acquire);
    int64_t consumed = atomic_load_explicit(&g_slots[slot].consumed,
                                            memory_order_acquire);
    if (event > consumed) {
        atomic_store_explicit(&g_slots[slot].consumed, event,
                              memory_order_release);
        atomic_fetch_add_explicit(&m_prefetch_used, 1, memory_order_relaxed);
    }
}

void metalio_prefetch_done(int slot) {
    if (!metalio_active() || slot < 0 || slot >= g_nslots ||
        !g_slots[slot].in_use)
        return;
    int64_t event = atomic_load_explicit(&g_slots[slot].last_event,
                                         memory_order_acquire);
    int64_t consumed = atomic_load_explicit(&g_slots[slot].consumed,
                                            memory_order_acquire);
    if (event > consumed)
        atomic_fetch_add_explicit(&m_prefetch_wasted, 1, memory_order_relaxed);
}

void metalio_stats(ColiMetalioStats *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->loads = atomic_load_explicit(&m_loads, memory_order_relaxed);
    out->bytes = atomic_load_explicit(&m_bytes, memory_order_relaxed);
    out->waits = atomic_load_explicit(&m_waits, memory_order_relaxed);
    out->fails = atomic_load_explicit(&m_fails, memory_order_relaxed);
    out->prefetch_loads =
        atomic_load_explicit(&m_prefetch_loads, memory_order_relaxed);
    out->prefetch_used =
        atomic_load_explicit(&m_prefetch_used, memory_order_relaxed);
    out->prefetch_wasted =
        atomic_load_explicit(&m_prefetch_wasted, memory_order_relaxed);
    out->outstanding =
        atomic_load_explicit(&m_outstanding, memory_order_relaxed);
    out->peak_outstanding =
        atomic_load_explicit(&m_peak_outstanding, memory_order_relaxed);
    out->latency_samples =
        atomic_load_explicit(&m_lat_samples, memory_order_relaxed);
    uint64_t total_us =
        atomic_load_explicit(&m_lat_total_us, memory_order_relaxed);
    out->total_latency_s = (double)total_us / 1.0e6;
    for (int i = 0; i < 32; i++)
        out->lat_hist[i] =
            atomic_load_explicit(&m_lat_hist[i], memory_order_relaxed);
}

/* -------------------------------------------------------------------------
 * Generic record-I/O provider.
 * ------------------------------------------------------------------------- */
static int metal_backend_init(void *context) {
    (void)context;
    return metalio_init();
}
static void metal_backend_shutdown(void *context) {
    (void)context;
    metalio_shutdown();
}
static int metal_backend_active(void *context) {
    (void)context;
    return metalio_active();
}
static ColiRecordIoBackendFile metal_backend_file_add(
        void *context, const char *path) {
    (void)context;
    return (ColiRecordIoBackendFile)metalio_file_add(path);
}
static ColiRecordIoBackendBuffer metal_backend_buffer_alloc(
        void *context, size_t max_bytes) {
    (void)context;
    return (ColiRecordIoBackendBuffer)metalio_slot_alloc(max_bytes);
}
static void metal_backend_buffer_free(
        void *context, ColiRecordIoBackendBuffer buffer) {
    (void)context;
    metalio_slot_free((int)buffer);
}
static void *metal_backend_buffer_ptr(
        void *context, ColiRecordIoBackendBuffer buffer) {
    (void)context;
    return metalio_slot_ptr((int)buffer);
}
static size_t metal_backend_buffer_bytes(
        void *context, ColiRecordIoBackendBuffer buffer) {
    (void)context;
    return metalio_slot_bytes((int)buffer);
}
static ColiRecordIoBackendEvent metal_backend_submitv(
        void *context, ColiRecordIoBackendBuffer buffer,
        const ColiRecordIoBackendRegion *regions, int count,
        ColiRecordIoBackendIntent intent) {
    (void)context;
    ColiMetalioKind kind;
    switch (intent) {
        case COLI_RECORD_IO_INTENT_DEMAND: kind = MIO_LOAD_DEMAND; break;
        case COLI_RECORD_IO_INTENT_ASYNC: kind = MIO_LOAD_ASYNC; break;
        case COLI_RECORD_IO_INTENT_SPECULATIVE: kind = MIO_LOAD_SPEC; break;
        default: return COLI_RECORD_IO_BACKEND_INVALID_EVENT;
    }
    return (ColiRecordIoBackendEvent)metalio_loadv(
        (int)buffer, (const ColiMetalioRegion *)regions, count, kind);
}
static int metal_backend_wait(void *context, ColiRecordIoBackendEvent event) {
    (void)context;
    return metalio_wait((int64_t)event);
}
static void metal_backend_consumed(
        void *context, ColiRecordIoBackendBuffer buffer) {
    (void)context;
    metalio_slot_consumed((int)buffer);
}
static void metal_backend_discarded(
        void *context, ColiRecordIoBackendBuffer buffer) {
    (void)context;
    metalio_prefetch_done((int)buffer);
}
static void metal_backend_stats(
        void *context, ColiRecordIoBackendStats *out) {
    (void)context;
    if (!out) return;
    ColiMetalioStats metal;
    metalio_stats(&metal);
    memset(out, 0, sizeof(*out));
    out->submissions = metal.loads;
    out->bytes = metal.bytes;
    out->waits = metal.waits;
    out->failures = metal.fails;
    out->speculative_submissions = metal.prefetch_loads;
    out->speculative_consumed = metal.prefetch_used;
    out->speculative_discarded = metal.prefetch_wasted;
    out->outstanding = metal.outstanding;
    out->peak_outstanding = metal.peak_outstanding;
    out->latency_samples = metal.latency_samples;
    out->total_latency_s = metal.total_latency_s;
    for (int i = 0; i < 32; i++) out->lat_hist[i] = metal.lat_hist[i];
}
static void metal_backend_verbose(void *context, int on) {
    (void)context;
    metalio_verbose(on);
}

static const ColiRecordIoBackendOps g_metal_backend_ops = {
    .name = "metalio",
    .capabilities = COLI_RECORD_IO_CAP_ASYNC |
                    COLI_RECORD_IO_CAP_VECTORED |
                    COLI_RECORD_IO_CAP_CPU_VISIBLE_BUFFER |
                    COLI_RECORD_IO_CAP_DEVICE_VISIBLE_BUFFER,
    .init = metal_backend_init,
    .shutdown = metal_backend_shutdown,
    .active = metal_backend_active,
    .file_add = metal_backend_file_add,
    .buffer_alloc = metal_backend_buffer_alloc,
    .buffer_free = metal_backend_buffer_free,
    .buffer_ptr = metal_backend_buffer_ptr,
    .buffer_bytes = metal_backend_buffer_bytes,
    .submitv = metal_backend_submitv,
    .wait = metal_backend_wait,
    .buffer_consumed = metal_backend_consumed,
    .buffer_discarded = metal_backend_discarded,
    .stats = metal_backend_stats,
    .verbose = metal_backend_verbose,
};

static ColiRecordIoBackend g_metal_backend = {
    .ops = &g_metal_backend_ops,
    .context = NULL,
};

ColiRecordIoBackend *coli_record_io_metal_default(void) {
    return &g_metal_backend;
}

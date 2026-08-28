/* metalio.mm — MTLIOCommandQueue expert streaming (Apple Silicon, Metal 3).
 *
 * Replaces, when active, the chain
 *     NVMe -> pread -> 16KiB host slab -> newBufferWithBytesNoCopy -> Metal
 * with
 *     NVMe -> MTLIOCommandQueue -> persistent shared MTLBuffer -> Metal
 * Apple Silicon has unified memory, so the win is NOT avoiding a discrete-GPU
 * upload: it is removing CPU-mediated I/O from the critical path so disk I/O
 * and GPU compute overlap (token time -> max(IO, GPU) instead of IO + GPU).
 *
 * Design (per the MetalIO handoff):
 *  - persistent slot pool: MTLBuffers are allocated once and reused across
 *    many expert replacements; never per-miss allocation
 *  - one MTLIOCommandQueue + one MTLSharedEvent; monotonically increasing
 *    event values make every load's completion observable without polling
 *  - loads are enqueued non-blocking; the CPU waits only when a layer
 *    genuinely needs a slot whose IO has not completed
 *  - shared storage: CPU fallback paths keep reading expert bytes
 *  - every failure path returns an error so the engine falls back to pread
 *
 * Compile-gated to darwin via the Makefile (METAL_OBJ); runtime-gated to
 * macOS 13+ / Metal 3 via @available. */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <stdarg.h>
#import <stdatomic.h>

#include "metalio.h"

#define METALIO_ALIGN 16384u   /* 16 KiB: page size on Apple Silicon */

static _Atomic int g_active = 0;
static _Atomic int g_verbose = 0;

/* --- metrics (atomics, lock-free reads via relaxed) ---------------------- */
static _Atomic uint64_t m_loads, m_bytes, m_waits, m_fails;
static _Atomic uint64_t m_prefetch_loads, m_prefetch_used, m_prefetch_wasted;
static _Atomic uint64_t m_outstanding, m_peak_outstanding;
static _Atomic uint64_t m_lat_samples;
static _Atomic uint64_t m_lat_total_us;
static _Atomic uint64_t m_lat_hist[32];

static void hist_add(uint64_t us){
    unsigned b = 0;
    while (us > 1 && b < 31) { us >>= 1; b++; }
    atomic_fetch_add_explicit(&m_lat_hist[b], 1, memory_order_relaxed);
}

/* --- state --------------------------------------------------------------- */
static id<MTLDevice> g_dev;
static id<MTLIOCommandQueue> g_iq;
static id<MTLSharedEvent> g_ev;
static uint64_t g_ev_val;          /* next value to signal; only touched under g_lock */

/* file handles: [file id] -> MTLIOFileHandle */
#define METALIO_MAX_FILES 64
static id<MTLIOFileHandle> g_files[METALIO_MAX_FILES];
static int g_nfiles;

/* persistent slots: [slot id] -> buffer + size + last-load event value.
 * Ids are REUSABLE: freed ids return to the pool, active set stays bounded. */
#define METALIO_MAX_SLOTS 4096
static struct {
    id<MTLBuffer> buf;
    size_t bytes;
    int in_use;
    _Atomic int64_t last_event;    /* event value of the most recent load */
    _Atomic int64_t consumed;      /* event value already consumed by compute (prefetch_used) */
} g_slots[METALIO_MAX_SLOTS];
static int g_nslots;
static int64_t g_consumed_high;    /* highest event value already waited (g_lock) */

static NSRecursiveLock *g_lock;

static void verbose(const char *fmt, ...){
    if (!atomic_load_explicit(&g_verbose, memory_order_relaxed)) return;
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[metalio] ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

int metalio_active(void){
    return atomic_load_explicit(&g_active, memory_order_relaxed);
}

void metalio_verbose(int on){
    atomic_store_explicit(&g_verbose, on, memory_order_relaxed);
}

int metalio_init(void){
    if (atomic_load_explicit(&g_active, memory_order_relaxed)) return 1;
    if (@available(macOS 13.0, *)) {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;
        MTLIOCommandQueueDescriptor *desc = [MTLIOCommandQueueDescriptor new];
        const char *depth_env = getenv("MTLIO_DEPTH");
        int depth = depth_env && atoi(depth_env) > 0 ? atoi(depth_env) : 64;
        if (depth > 1024) depth = 1024;
        desc.maxCommandBufferCount = depth;
        desc.priority = MTLIOPriorityHigh;
        NSError *err = nil;
        g_iq = [g_dev newIOCommandQueueWithDescriptor:desc error:&err];
        if (!g_iq) {
            fprintf(stderr, "[metalio] IO queue creation failed: %s\n",
                    err ? err.description.UTF8String : "unknown");
            return 0;
        }
        g_ev = [g_dev newSharedEvent];
        if (!g_ev) return 0;
        g_lock = [NSRecursiveLock new];
        g_ev_val = 1;
        atomic_store_explicit(&g_active, 1, memory_order_relaxed);
        verbose("init: device=%s queue=ok", g_dev.name.UTF8String ? g_dev.name.UTF8String : "?");
        return 1;
    }
    return 0;
}

void metalio_shutdown(void){
    if (!atomic_load_explicit(&g_active, memory_order_relaxed)) return;
    [g_lock lock];
    if (g_ev && g_ev_val > 1) {
        [g_ev waitUntilSignaledValue:g_ev_val - 1 timeoutMS:UINT64_MAX];
        verbose("shutdown: drained event=%llu", (unsigned long long)(g_ev_val - 1));
    }
    for (int i = 0; i < g_nfiles; i++) g_files[i] = nil;
    g_nfiles = 0;
    for (int i = 0; i < g_nslots; i++) { g_slots[i].buf = nil; g_slots[i].bytes = 0; }
    g_nslots = 0;
    g_iq = nil; g_ev = nil; g_dev = nil;
    [g_lock unlock];
    atomic_store_explicit(&g_active, 0, memory_order_relaxed);
}

int metalio_file_add(const char *path){
    if (!atomic_load_explicit(&g_active, memory_order_relaxed)) return -1;
    if (!path || !path[0]) return -1;
    [g_lock lock];
    int fid = -1;
    if (@available(macOS 13.0, *)) {
        if (g_nfiles < METALIO_MAX_FILES) {
            NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
            NSError *err = nil;
            id<MTLIOFileHandle> h = [g_dev newIOFileHandleWithURL:url error:&err];
            if (h) {
                fid = g_nfiles;
                g_files[g_nfiles++] = h;
                verbose("file_add: path=%s id=%d", path, fid);
            } else if (err) {
                fprintf(stderr, "[metalio] file handle failed for %s: %s\n",
                        path, err.description.UTF8String);
            }
        }
    }
    [g_lock unlock];
    return fid;
}

int metalio_slot_alloc(size_t max_bytes){
    if (!atomic_load_explicit(&g_active, memory_order_relaxed)) return -1;
    size_t len = (max_bytes + METALIO_ALIGN - 1) & ~(size_t)(METALIO_ALIGN - 1);
    [g_lock lock];
    int sid = -1;
    if (@available(macOS 13.0, *)) {
        for (int i = 0; i < g_nslots && sid < 0; i++)
            if (!g_slots[i].in_use) sid = i;
        if (sid < 0 && g_nslots < METALIO_MAX_SLOTS) sid = g_nslots++;
        if (sid >= 0) {
            id<MTLBuffer> b = [g_dev newBufferWithLength:len
                                                 options:MTLResourceStorageModeShared];
            if (b) {
                g_slots[sid].buf = b;
                g_slots[sid].bytes = len;
                g_slots[sid].in_use = 1;
                atomic_store_explicit(&g_slots[sid].last_event, 0, memory_order_relaxed);
                atomic_store_explicit(&g_slots[sid].consumed, 0, memory_order_relaxed);
                verbose("slot_alloc: id=%d bytes=%zu", sid, len);
            } else sid = -1;
        }
    }
    [g_lock unlock];
    return sid;
}

void metalio_slot_free(int slot){
    if (!atomic_load_explicit(&g_active, memory_order_relaxed)) return;
    if (slot < 0 || slot >= g_nslots || !g_slots[slot].in_use) return;
    [g_lock lock];
    int64_t ev = atomic_load_explicit(&g_slots[slot].last_event, memory_order_relaxed);
    if (ev > 0 && g_ev) [g_ev waitUntilSignaledValue:(uint64_t)ev timeoutMS:UINT64_MAX];
    g_slots[slot].buf = nil;
    g_slots[slot].bytes = 0;
    g_slots[slot].in_use = 0;
    [g_lock unlock];
}

void *metalio_slot_ptr(int slot){
    if (slot < 0 || slot >= g_nslots) return NULL;
    return g_slots[slot].buf ? g_slots[slot].buf.contents : NULL;
}

size_t metalio_slot_bytes(int slot){
    if (slot < 0 || slot >= g_nslots) return 0;
    return g_slots[slot].bytes;
}

void *metalio_slot_native_buffer(int slot){
    if (!atomic_load_explicit(&g_active, memory_order_relaxed)) return NULL;
    [g_lock lock];
    void *out = NULL;
    if (slot >= 0 && slot < g_nslots && g_slots[slot].in_use && g_slots[slot].buf)
        out = (__bridge void *)g_slots[slot].buf;
    [g_lock unlock];
    return out;
}

static int regions_ok(int slot, const ColiMetalioRegion *regions, int count){
    if (count <= 0 || !regions) return 0;
    size_t cap = g_slots[slot].bytes, highest = 0;
    uint64_t total = 0;
    for (int i = 0; i < count; i++) {
        const ColiMetalioRegion *r = &regions[i];
        if (r->file < 0 || r->file >= g_nfiles || !g_files[r->file]) return 0;
        if (r->bytes == 0) return 0;
        if (r->dst_off > cap || r->bytes > cap - r->dst_off) return 0;
        if (r->src_off > UINT64_MAX - r->bytes) return 0;
        if (r->bytes > SIZE_MAX - total) return 0;
        total += r->bytes;
        if (r->dst_off + r->bytes > highest) highest = r->dst_off + r->bytes;
    }
    (void)highest;
    return 1;
}

int64_t metalio_loadv(int slot, const ColiMetalioRegion *regions, int count,
                      ColiMetalioKind kind){
    if (!atomic_load_explicit(&g_active, memory_order_relaxed)) return -1;
    if (slot < 0 || slot >= g_nslots || !g_slots[slot].in_use) return -1;
    [g_lock lock];
    int64_t ev = -1;
    if (@available(macOS 13.0, *)) {
        if (regions_ok(slot, regions, count)) {
            id<MTLIOCommandBuffer> ioCB = [g_iq commandBuffer];
            for (int i = 0; i < count; i++) {
                const ColiMetalioRegion *r = &regions[i];
                [ioCB loadBuffer:g_slots[slot].buf
                          offset:r->dst_off
                            size:r->bytes
                    sourceHandle:g_files[r->file]
               sourceHandleOffset:r->src_off];
            }
            uint64_t v = g_ev_val++;
            [ioCB signalEvent:g_ev value:v];
            [ioCB commit];
            atomic_store_explicit(&g_slots[slot].last_event, (int64_t)v, memory_order_relaxed);
            ev = (int64_t)v;
            atomic_fetch_add_explicit(&m_loads, 1, memory_order_relaxed);
            if (kind == MIO_LOAD_SPEC)
                atomic_fetch_add_explicit(&m_prefetch_loads, 1, memory_order_relaxed);
            for (int i = 0; i < count; i++)
                atomic_fetch_add_explicit(&m_bytes, regions[i].bytes, memory_order_relaxed);
            uint64_t out = atomic_fetch_add_explicit(&m_outstanding, 1, memory_order_relaxed) + 1;
            uint64_t peak = atomic_load_explicit(&m_peak_outstanding, memory_order_relaxed);
            while (out > peak &&
                   !atomic_compare_exchange_weak_explicit(&m_peak_outstanding, &peak, out,
                                                          memory_order_relaxed, memory_order_relaxed)) {}
            verbose("load: slot=%d regions=%d event=%llu", slot, count, (unsigned long long)v);
        }
    }
    [g_lock unlock];
    return ev;
}

int64_t metalio_load(int slot, int file, uint64_t offset, size_t bytes){
    ColiMetalioRegion r = { file, offset, bytes, 0 };
    return metalio_loadv(slot, &r, 1, MIO_LOAD_DEMAND);
}

int metalio_wait(int64_t event_value){
    if (!atomic_load_explicit(&g_active, memory_order_relaxed)) return -1;
    if (event_value <= 0 || !g_ev) return -1;
    [g_lock lock];
    if (event_value <= g_consumed_high) {
        [g_lock unlock];
        return 0;
    }
    if (event_value > (int64_t)g_ev_val - 1) { [g_lock unlock]; return -1; }
    [g_lock unlock];
    uint64_t t0 = mach_absolute_time();
    [g_ev waitUntilSignaledValue:(uint64_t)event_value timeoutMS:UINT64_MAX];
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t us = (mach_absolute_time() - t0) * tb.numer / tb.denom / 1000;
    [g_lock lock];
    if (event_value > g_consumed_high) {
        g_consumed_high = event_value;
        atomic_store_explicit(&m_outstanding, g_ev_val - 1 - (uint64_t)g_consumed_high,
                              memory_order_relaxed);
    }
    [g_lock unlock];
    atomic_fetch_add_explicit(&m_waits, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&m_lat_samples, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&m_lat_total_us, us, memory_order_relaxed);
    hist_add(us);
    verbose("wait: event=%lld latency=%lluus", (long long)event_value, (unsigned long long)us);
    return 0;
}

void metalio_slot_consumed(int slot){
    if (slot < 0 || slot >= g_nslots) return;
    int64_t ev = atomic_load_explicit(&g_slots[slot].last_event, memory_order_relaxed);
    int64_t cons = atomic_load_explicit(&g_slots[slot].consumed, memory_order_relaxed);
    if (ev > cons) {
        atomic_store_explicit(&g_slots[slot].consumed, ev, memory_order_relaxed);
        atomic_fetch_add_explicit(&m_prefetch_used, 1, memory_order_relaxed);
    }
}

void metalio_prefetch_done(int slot){
    if (slot < 0 || slot >= g_nslots) return;
    int64_t ev = atomic_load_explicit(&g_slots[slot].last_event, memory_order_relaxed);
    int64_t cons = atomic_load_explicit(&g_slots[slot].consumed, memory_order_relaxed);
    if (ev > cons)
        atomic_fetch_add_explicit(&m_prefetch_wasted, 1, memory_order_relaxed);
}

void metalio_stats(ColiMetalioStats *out){
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->loads = atomic_load_explicit(&m_loads, memory_order_relaxed);
    out->bytes = atomic_load_explicit(&m_bytes, memory_order_relaxed);
    out->waits = atomic_load_explicit(&m_waits, memory_order_relaxed);
    out->fails = atomic_load_explicit(&m_fails, memory_order_relaxed);
    out->prefetch_loads = atomic_load_explicit(&m_prefetch_loads, memory_order_relaxed);
    out->prefetch_used = atomic_load_explicit(&m_prefetch_used, memory_order_relaxed);
    out->prefetch_wasted = atomic_load_explicit(&m_prefetch_wasted, memory_order_relaxed);
    out->outstanding = atomic_load_explicit(&m_outstanding, memory_order_relaxed);
    out->peak_outstanding = atomic_load_explicit(&m_peak_outstanding, memory_order_relaxed);
    out->latency_samples = atomic_load_explicit(&m_lat_samples, memory_order_relaxed);
    uint64_t us = atomic_load_explicit(&m_lat_total_us, memory_order_relaxed);
    out->total_latency_s = out->latency_samples ? (double)us / 1e6 : 0.0;
    for (int i = 0; i < 32; i++)
        out->lat_hist[i] = atomic_load_explicit(&m_lat_hist[i], memory_order_relaxed);
}

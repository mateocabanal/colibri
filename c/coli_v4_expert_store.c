#include "coli_v4_expert_store.h"
#include "coli_executor.h"
#include "coli_v4_static.h"
#include "compat.h"
#ifdef COLI_METAL
#include "backend_metal.h"
#endif

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef COLI_V4_GIT_SHA
#define COLI_V4_GIT_SHA "unknown"
#endif

#ifdef __APPLE__
/* Consumed by the V4-local pread interposition in coli_v4_macos_uncached_io.h.
 * Loader threads set this only while reading a routed-expert record. */
__thread int coli_v4_expert_io_active;
#endif

typedef struct {
    const ColiRecordInfo *record;
    ColiExpertInfo info;
} Record;

typedef struct {
    int expert;
    unsigned refs, loading;
    int load_failed;
    unsigned char *data;
    uint64_t generation;
    uint64_t last_used;
} Slot;

typedef struct {
    ColiExecutor *executor;
    int layers, experts, slots_per_layer;
    uint64_t record_bytes, slot_bytes, clock;
    uint64_t total_cache_budget, dense_cache_budget;
    Record *records;
    Slot *slots;
    uint64_t *activation_counts;
    unsigned active_leases;
    ColiExpertStoreStats stats;
    pthread_mutex_t mutex;
    pthread_cond_t changed;

    time_t io_started_at, io_last_report_at;
    uint64_t io_reads, io_bytes;
    unsigned io_inflight, io_peak_inflight;
    int progress_enabled, progress_interval_s;
} State;

static int fail(char *e, size_t n, const char *f, ...) {
    va_list a;
    if (e && n) {
        va_start(a, f);
        vsnprintf(e, n, f, a);
        va_end(a);
    }
    return -1;
}

static Record *record_for(State *s, ColiExpertKey k) {
    if (k.layer < 0 || k.layer >= s->layers ||
        k.expert < 0 || k.expert >= s->experts)
        return NULL;
    return &s->records[(size_t)k.layer * s->experts + k.expert];
}

static Slot *slots_for(State *s, int layer) {
    return s->slots + (size_t)layer * s->slots_per_layer;
}

static int env_int(const char *name, int fallback, int lo, int hi) {
    const char *text = getenv(name);
    if (!text || !*text) return fallback;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (!end || *end || value < lo || value > hi) return fallback;
    return (int)value;
}

static long double env_ratio(const char *name, long double fallback) {
    const char *text = getenv(name);
    if (!text || !*text) return fallback;
    char *end = NULL;
    long double value = strtold(text, &end);
    if (!end || *end || value < 0.0L || value > 1.0L) return fallback;
    return value;
}

static uint64_t env_u64(const char *name, uint64_t fallback, int *present) {
    const char *text = getenv(name);
    if (present) *present = 0;
    if (!text || !*text) return fallback;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (!end || *end) return fallback;
    if (present) *present = 1;
    return (uint64_t)value;
}

static void print_execution_mode(void) {
#ifdef COLI_METAL
    const char *setting = getenv("V4_METAL_EXPERTS");
    const int metal_requested = (!setting || !*setting) ? 1 : (atoi(setting) != 0);
    const int metal_ready = metal_requested && coli_metal_init() && coli_metal_available();
    if (metal_ready) {
        fprintf(stderr,
                "v4_execution build=%s mode=hybrid "
                "cpu=control+unsupported-kernels+fallback "
                "gpu=metal-mxfp4 metal=available fallback=cpu\n",
                COLI_V4_GIT_SHA);
    } else if (!metal_requested) {
        fprintf(stderr,
                "v4_execution build=%s mode=cpu gpu=disabled "
                "reason=V4_METAL_EXPERTS=0\n",
                COLI_V4_GIT_SHA);
    } else {
        fprintf(stderr,
                "v4_execution build=%s mode=cpu gpu=unavailable "
                "requested=metal fallback=cpu\n",
                COLI_V4_GIT_SHA);
    }
#else
    fprintf(stderr,
            "v4_execution build=%s mode=cpu gpu=not-built "
            "reason=COLI_METAL-disabled\n",
            COLI_V4_GIT_SHA);
#endif
}

static void io_begin_locked(State *s, ColiExpertKey key) {
    const time_t now = time(NULL);
    if (!s->io_started_at) {
        s->io_started_at = now;
        s->io_last_report_at = now;
        if (s->progress_enabled) {
            fprintf(stderr,
                    "v4_progress phase=expert-stream status=started "
                    "record=%.2fMiB cache_slots_per_layer=%d layer=%d expert=%d\n",
                    (double)s->record_bytes / (1024.0 * 1024.0),
                    s->slots_per_layer, key.layer, key.expert);
            fflush(stderr);
        }
    }
    s->io_inflight++;
    s->stats.loads_started++;
    uint64_t inflight_bytes = (uint64_t)s->io_inflight * s->slot_bytes;
    if (inflight_bytes > s->stats.peak_inflight_bytes)
        s->stats.peak_inflight_bytes = inflight_bytes;
    if (s->io_inflight > s->io_peak_inflight)
        s->io_peak_inflight = s->io_inflight;
}

static void io_finish_locked(State *s, ColiExpertKey key, int success) {
    const time_t now = time(NULL);
    if (s->io_inflight) s->io_inflight--;
    if (success) {
        s->io_reads++;
        s->io_bytes += s->record_bytes;
    } else {
        s->stats.loads_failed++;
    }
    if (!s->progress_enabled || !s->io_started_at ||
        now - s->io_last_report_at < s->progress_interval_s)
        return;

    double elapsed = difftime(now, s->io_started_at);
    if (elapsed < 1.0) elapsed = 1.0;
    const double gib = (double)s->io_bytes / (1024.0 * 1024.0 * 1024.0);
    const double expert_mib_s =
        ((double)s->io_bytes / (1024.0 * 1024.0)) / elapsed;
    const double hit_pct = s->stats.requests
        ? 100.0 * (double)s->stats.hits / (double)s->stats.requests : 0.0;

    fprintf(stderr,
            "\nv4_progress phase=expert-stream elapsed=%.0fs reads=%llu "
            "bytes=%.2fGiB expert_effective=%.1fMiB/s "
            "inflight=%u peak_inflight=%u cache_hit=%.1f%% "
            "joins=%llu evictions=%llu layer=%d/%d expert=%d\n",
            elapsed, (unsigned long long)s->io_reads, gib, expert_mib_s,
            s->io_inflight, s->io_peak_inflight, hit_pct,
            (unsigned long long)s->stats.inflight_joins,
            (unsigned long long)s->stats.evictions,
            key.layer + 1, s->layers, key.expert);
    fflush(stderr);
    s->io_last_report_at = now;
}

static int tensor_format(const ColiExpertMatrixInfo *m, ColiTensorView *v,
                         const unsigned char *data) {
    if (m->math_format != COLI_CSF_MATH_MXFP4_E2M1 ||
        m->scale_format != COLI_CSF_SCALE_UE8M0 ||
        m->layout != COLI_CSF_LAYOUT_CANONICAL ||
        m->weight_codec != COLI_CSF_CODEC_NONE ||
        m->scale_codec != COLI_CSF_CODEC_NONE ||
        m->rows > (uint64_t)INT64_MAX ||
        m->columns > (uint64_t)INT64_MAX ||
        m->weight_stored_bytes > SIZE_MAX ||
        m->scale_stored_bytes > SIZE_MAX)
        return -1;

    memset(v, 0, sizeof(*v));
    v->format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    v->scale_format = COLI_SCALE_UE8M0;
    v->data = data + m->weight_offset;
    v->scales = data + m->scale_offset;
    v->data_bytes = (size_t)m->weight_stored_bytes;
    v->scale_bytes = (size_t)m->scale_stored_bytes;
    v->rows = (int64_t)m->rows;
    v->columns = (int64_t)m->columns;
    v->block_rows = m->scale_block_rows;
    v->block_columns = m->scale_block_columns;
    return 0;
}

static int fill(ColiExpertView *v, ColiExpertKey k, const Record *r,
                const Slot *s) {
    const ColiExpertMatrixInfo *gate = NULL, *up = NULL, *down = NULL;
    for (int i = 0; i < 3; i++) {
        const ColiExpertMatrixInfo *m = &r->info.matrices[i];
        if (m->role == 1) gate = m;
        else if (m->role == 2) up = m;
        else if (m->role == 3) down = m;
    }
    if (!gate || !up || !down ||
        tensor_format(gate, &v->gate, s->data) ||
        tensor_format(up, &v->up, s->data) ||
        tensor_format(down, &v->down, s->data))
        return -1;
    v->key = k;
    v->lease = (void *)s;
    v->generation = s->generation;
    return 0;
}

static Slot *find_matching_slot_locked(State *s, ColiExpertKey key) {
    Slot *slots = slots_for(s, key.layer);
    for (int i = 0; i < s->slots_per_layer; i++)
        if (slots[i].expert == key.expert && slots[i].data) return &slots[i];
    return NULL;
}

static Slot *choose_victim_locked(State *s, int layer) {
    Slot *slots = slots_for(s, layer);
    Slot *victim = NULL;
    for (int i = 0; i < s->slots_per_layer; i++) {
        Slot *slot = &slots[i];
        if (slot->refs || slot->loading) continue;
        if (!slot->data) return slot;
        if (!victim || slot->last_used < victim->last_used) victim = slot;
    }
    return victim;
}

static int lookup(ColiExpertStore *store, ColiExpertKey key,
                  ColiExpertView *view) {
    State *s = store ? store->state : NULL;
    Record *r = s ? record_for(s, key) : NULL;
    if (!s || !r || !view) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }

    pthread_mutex_lock(&s->mutex);
    s->stats.requests++;
    s->activation_counts[(size_t)key.layer * s->experts + key.expert]++;

retry:
    Slot *slot = find_matching_slot_locked(s, key);
    if (slot) {
        if (slot->loading) {
            s->stats.inflight_joins++;
            while (slot->loading && slot->expert == key.expert)
                pthread_cond_wait(&s->changed, &s->mutex);
            if (slot->expert != key.expert || slot->load_failed) goto retry;
        }
        if (!slot->load_failed && slot->expert == key.expert) {
            s->stats.hits++;
            slot->last_used = ++s->clock;
            slot->refs++;
            s->active_leases++;
            int bad_view = fill(view, key, r, slot);
            if (bad_view) {
                slot->refs--;
                s->active_leases--;
            }
            pthread_mutex_unlock(&s->mutex);
            return bad_view ? -1 : 0;
        }
    }

    slot = choose_victim_locked(s, key.layer);
    if (!slot) {
        /* With a deliberately small transient pool all slots may briefly be
         * leased/loading. Wait for a release instead of turning cache pressure
         * into a model execution failure. */
        pthread_cond_wait(&s->changed, &s->mutex);
        goto retry;
    }

    if (!slot->data) {
        if (s->slot_bytes > SIZE_MAX ||
            posix_memalign((void **)&slot->data, 16384,
                           (size_t)s->slot_bytes)) {
            pthread_mutex_unlock(&s->mutex);
            memset(view, 0, sizeof(*view));
            return -1;
        }
#ifdef COLI_METAL
        if (coli_metal_init())
            coli_metal_register(slot->data, (size_t)s->slot_bytes);
#endif
        s->stats.resident_bytes += s->slot_bytes;
    } else if (slot->expert >= 0 && slot->expert != key.expert) {
        s->stats.evictions++;
    }

    slot->expert = key.expert;
    slot->loading = 1;
    slot->load_failed = 0;
    slot->generation++;
    slot->last_used = ++s->clock;
    io_begin_locked(s, key);
    pthread_mutex_unlock(&s->mutex);

    char error[256];
#ifdef __APPLE__
    const char *direct = getenv("COLI_V4_DIRECT");
    coli_v4_expert_io_active = !direct || atoi(direct) != 0;
#endif
    int bad_load = coli_executor_load_expert(
        s->executor, key.layer, key.expert, slot->data,
        (size_t)s->record_bytes, error, sizeof(error));
#ifdef __APPLE__
    coli_v4_expert_io_active = 0;
#endif

    pthread_mutex_lock(&s->mutex);
    slot->loading = 0;
    slot->load_failed = bad_load != 0;
    io_finish_locked(s, key, !bad_load);
    if (bad_load) {
        slot->expert = -1;
        pthread_cond_broadcast(&s->changed);
        fprintf(stderr,
                "v4_coli expert-load failed layer=%d expert=%d: %s\n",
                key.layer, key.expert, error);
        pthread_mutex_unlock(&s->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }

    s->stats.misses++;
    s->stats.bytes_read += s->record_bytes;
    slot->last_used = ++s->clock;
    slot->refs++;
    s->active_leases++;
    pthread_cond_broadcast(&s->changed);

    int bad_view = fill(view, key, r, slot);
    if (bad_view) {
        slot->refs--;
        s->active_leases--;
        pthread_cond_broadcast(&s->changed);
    }
    pthread_mutex_unlock(&s->mutex);
    return bad_view ? -1 : 0;
}

static void release(ColiExpertStore *store, ColiExpertView *v) {
    State *s = store ? store->state : NULL;
    Slot *slot = v ? v->lease : NULL;
    if (!s || !slot) return;
    pthread_mutex_lock(&s->mutex);
    if (slot->generation == v->generation && slot->refs) {
        slot->refs--;
        if (s->active_leases) s->active_leases--;
        slot->last_used = ++s->clock;
        pthread_cond_broadcast(&s->changed);
    }
    pthread_mutex_unlock(&s->mutex);
}

static int prefetch(ColiExpertStore *store, const ColiExpertKey *k, size_t n) {
    (void)store;
    (void)k;
    (void)n;
    return 0;
}

static void stats(const ColiExpertStore *store, ColiExpertStoreStats *out) {
    State *s = store ? store->state : NULL;
    if (!s || !out) return;
    pthread_mutex_lock(&s->mutex);
    *out = s->stats;
    pthread_mutex_unlock(&s->mutex);
}

static void dump_activation_trace(State *s) {
    const char *path = getenv("V4_EXPERT_TRACE_PATH");
    if (!path || !*path) return;
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "v4_trace warning=cannot-open path=%s\n", path);
        return;
    }
    fprintf(fp,
            "{\"type\":\"summary\",\"requests\":%llu,\"hits\":%llu,"
            "\"misses\":%llu,\"inflight_joins\":%llu,\"evictions\":%llu,"
            "\"bytes_read\":%llu}\n",
            (unsigned long long)s->stats.requests,
            (unsigned long long)s->stats.hits,
            (unsigned long long)s->stats.misses,
            (unsigned long long)s->stats.inflight_joins,
            (unsigned long long)s->stats.evictions,
            (unsigned long long)s->stats.bytes_read);
    for (int layer = 0; layer < s->layers; layer++) {
        for (int expert = 0; expert < s->experts; expert++) {
            uint64_t count = s->activation_counts[(size_t)layer * s->experts + expert];
            if (!count) continue;
            fprintf(fp,
                    "{\"type\":\"expert_activation\",\"layer\":%d,"
                    "\"expert\":%d,\"requests\":%llu}\n",
                    layer, expert, (unsigned long long)count);
        }
    }
    fclose(fp);
    fprintf(stderr, "v4_trace kind=expert-aggregate path=%s\n", path);
}

static void destroy(ColiExpertStore *store) {
    State *s = store ? store->state : NULL;
    if (s) {
        if (s->progress_enabled && s->io_started_at && s->io_reads) {
            double elapsed = difftime(time(NULL), s->io_started_at);
            if (elapsed < 1.0) elapsed = 1.0;
            fprintf(stderr,
                    "v4_progress phase=expert-stream status=done elapsed=%.0fs "
                    "reads=%llu bytes=%.2fGiB expert_effective=%.1fMiB/s "
                    "peak_inflight=%u joins=%llu evictions=%llu\n",
                    elapsed, (unsigned long long)s->io_reads,
                    (double)s->io_bytes / (1024.0 * 1024.0 * 1024.0),
                    ((double)s->io_bytes / (1024.0 * 1024.0)) / elapsed,
                    s->io_peak_inflight,
                    (unsigned long long)s->stats.inflight_joins,
                    (unsigned long long)s->stats.evictions);
        }
        ColiV4DenseCacheStats dense;
        memset(&dense, 0, sizeof(dense));
        coli_v4_dense_cache_stats(&dense);
        if (dense.budget_bytes) {
            fprintf(stderr,
                    "v4_dense_cache budget=%.2fGiB resident=%.2fGiB hits=%llu "
                    "misses=%llu inserts=%llu evictions=%llu avoided=%.2fGiB\n",
                    dense.budget_bytes / 1073741824.0,
                    dense.resident_bytes / 1073741824.0,
                    (unsigned long long)dense.hits,
                    (unsigned long long)dense.misses,
                    (unsigned long long)dense.inserts,
                    (unsigned long long)dense.evictions,
                    dense.stored_bytes_avoided / 1073741824.0);
        }
        dump_activation_trace(s);
        for (int i = 0; i < s->layers * s->slots_per_layer; i++) {
#ifdef COLI_METAL
            if (s->slots[i].data)
                coli_metal_unregister(s->slots[i].data);
#endif
            compat_aligned_free(s->slots[i].data);
        }
        pthread_cond_destroy(&s->changed);
        pthread_mutex_destroy(&s->mutex);
        coli_executor_close(s->executor);
        free(s->activation_counts);
        free(s->slots);
        free(s->records);
        coli_v4_dense_cache_shutdown();
        free(s);
    }
    free(store);
}

int coli_v4_coli_expert_store_open(const ColiV4ColiExpertStoreOptions *o,
                                   ColiExpertStore **out,
                                   char *e, size_t n) {
    static const ColiExpertStoreOps ops = {
        lookup, release, prefetch, stats, destroy
    };
    ColiExpertStore *store = NULL;
    State *s = NULL;
    ColiExecutorOpenOptions xo = {0};

    if (out) *out = NULL;
    if (!o || !out || !o->package_dir || !o->required_profile ||
        o->layers < 1 || o->experts_per_layer < 1 || !o->cache_bytes)
        return fail(e, n, "invalid COLI V4 expert-store options");

    store = calloc(1, sizeof(*store));
    s = calloc(1, sizeof(*s));
    if (!store || !s) {
        free(store);
        free(s);
        return fail(e, n, "out of memory creating COLI expert store");
    }
    pthread_mutex_init(&s->mutex, NULL);
    pthread_cond_init(&s->changed, NULL);

    s->progress_enabled = !getenv("V4_PROGRESS") ||
                          atoi(getenv("V4_PROGRESS")) != 0;
    s->progress_interval_s = env_int("V4_PROGRESS_INTERVAL", 5, 1, 60);

    xo.required_profile = o->required_profile;
    xo.checksum_policy = getenv("COLI_VERIFY_RECORDS") &&
                         atoi(getenv("COLI_VERIFY_RECORDS"))
        ? COLI_CSF_CHECKSUM_RECORD_ON_READ
        : COLI_CSF_CHECKSUM_MANIFEST_ONLY;
    if (coli_executor_open(&s->executor, o->package_dir, &xo, e, n))
        goto bad;

    s->layers = o->layers;
    s->experts = o->experts_per_layer;
    s->total_cache_budget = o->cache_bytes;
    s->records = calloc((size_t)s->layers * s->experts, sizeof(*s->records));
    s->activation_counts = calloc((size_t)s->layers * s->experts,
                                  sizeof(*s->activation_counts));
    if (!s->records || !s->activation_counts) {
        fail(e, n, "out of memory indexing COLI experts");
        goto bad;
    }

    for (int l = 0; l < s->layers; l++) {
        for (int x = 0; x < s->experts; x++) {
            Record *r = &s->records[(size_t)l * s->experts + x];
            r->record = coli_executor_expert(s->executor, l, x);
            if (!r->record ||
                coli_executor_expert_info(s->executor, l, x, &r->info, e, n)) {
                fail(e, n, "COLI package is missing/invalid expert (%d,%d)", l, x);
                goto bad;
            }
            if (!s->record_bytes) s->record_bytes = r->record->stored_bytes;
            if (r->record->stored_bytes != s->record_bytes) {
                fail(e, n, "COLI experts have non-uniform stored sizes");
                goto bad;
            }
        }
    }

    if (s->record_bytes > UINT64_MAX - 16383u) {
        fail(e, n, "COLI expert slot size overflow");
        goto bad;
    }
    s->slot_bytes = (s->record_bytes + 16383u) & ~UINT64_C(16383);
    if (s->slot_bytes > SIZE_MAX) {
        fail(e, n, "COLI expert slot exceeds address space");
        goto bad;
    }

    const uint64_t per_layer_slot_bytes = (uint64_t)s->layers * s->slot_bytes;
    int max_slots = (int)(o->cache_bytes / per_layer_slot_bytes);
    if (max_slots < 1) {
        fail(e, n, "COLI cache budget cannot hold one expert per layer");
        goto bad;
    }
    if (max_slots > s->experts) max_slots = s->experts;

    int loader_lanes = env_int("V4_LOADER_LANES", 3, 1, 16);
    int default_floor = loader_lanes < 16 ? loader_lanes + 1 : 16;
    int min_slots = env_int("V4_TRANSIENT_EXPERT_SLOTS", default_floor, 1, 16);
    if (min_slots > max_slots) min_slots = max_slots;

    uint64_t min_expert_bytes = (uint64_t)min_slots * per_layer_slot_bytes;
    uint64_t optional_bytes = o->cache_bytes > min_expert_bytes
        ? o->cache_bytes - min_expert_bytes : 0;
    long double dense_share = env_ratio("V4_DENSE_CACHE_SHARE", 1.0L);
    uint64_t desired_dense = (uint64_t)((long double)optional_bytes * dense_share);
    int explicit_dense = 0;
    desired_dense = env_u64("V4_DENSE_CACHE_BYTES", desired_dense, &explicit_dense);
    if (desired_dense > optional_bytes) desired_dense = optional_bytes;

    uint64_t expert_budget = o->cache_bytes - desired_dense;
    int selected_slots = (int)(expert_budget / per_layer_slot_bytes);
    if (selected_slots < min_slots) selected_slots = min_slots;
    if (selected_slots > max_slots) selected_slots = max_slots;
    s->slots_per_layer = selected_slots;
    uint64_t actual_expert_capacity =
        (uint64_t)s->slots_per_layer * per_layer_slot_bytes;
    s->dense_cache_budget = o->cache_bytes > actual_expert_capacity
        ? o->cache_bytes - actual_expert_capacity : 0;
    coli_v4_dense_cache_configure(s->dense_cache_budget);

    s->slots = calloc((size_t)s->layers * s->slots_per_layer,
                      sizeof(*s->slots));
    if (!s->slots) {
        fail(e, n, "out of memory creating COLI expert slots");
        goto bad;
    }
    for (int i = 0; i < s->layers * s->slots_per_layer; i++)
        s->slots[i].expert = -1;
    s->stats.capacity_bytes = actual_expert_capacity;

    print_execution_mode();
    fprintf(stderr,
            "v4_residency policy=benefit-per-byte total=%.2fGiB "
            "expert=%.2fGiB dense=%.2fGiB expert_slots_per_layer=%d "
            "transient_floor=%d dense_share=%.2Lf%s\n",
            o->cache_bytes / 1073741824.0,
            actual_expert_capacity / 1073741824.0,
            s->dense_cache_budget / 1073741824.0,
            s->slots_per_layer, min_slots, dense_share,
            explicit_dense ? " dense_override=bytes" : "");
    if (s->progress_enabled) {
        fprintf(stderr,
                "v4_progress enabled=1 interval=%ds "
                "metric=expert-bytes-per-wall-second "
                "hint=V4_PROGRESS=0-to-disable\n",
                s->progress_interval_s);
    }

    store->ops = &ops;
    store->state = s;
    *out = store;
    return 0;

bad:
    destroy(store);
    return -1;
}

#include "coli_v4_expert_store.h"
#include "coli_executor.h"
#include "coli_v4_residency.h"
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

#ifndef COLI_V4_EXPERT_LOADER_COUNT
#define COLI_V4_EXPERT_LOADER_COUNT 3
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

typedef enum {
    SLOT_EMPTY = 0,
    SLOT_LOADING = 1,
    SLOT_RESIDENT = 2,
} SlotState;

typedef enum {
    SLOT_TIER_TRANSIENT = 0,
    SLOT_TIER_PERSISTENT = 1,
} SlotTier;

typedef struct {
    int layer;
    int expert;
    int home_layer; /* persistent tier only; -1 for global transient slots */
    SlotState state;
    SlotTier tier;
    unsigned refs;
    unsigned char *data;
    uint64_t generation;
    uint64_t last_use;
} Slot;

typedef enum {
    TRACE_REQUEST = 1,
    TRACE_HIT,
    TRACE_INFLIGHT_JOIN,
    TRACE_LOAD_BEGIN,
    TRACE_LOAD_COMPLETE,
    TRACE_LOAD_FAILED,
    TRACE_EVICT,
    TRACE_RELEASE,
    TRACE_SLOT_WAIT,
} TraceKind;

typedef struct {
    uint64_t seq;
    uint64_t generation;
    uint64_t bytes;
    int layer;
    int expert;
    unsigned char kind;
    unsigned char tier;
    unsigned char has_tier;
} TraceEvent;

typedef struct {
    ColiExecutor *executor;
    int layers, experts;
    uint64_t record_bytes, slot_bytes, clock;
    Record *records;

    /* #57: separate persistent locality from execution concurrency. Persistent
     * slots are partitioned per layer; transient slots form one global pool
     * shared by all layers. The latter therefore scales with loader width, not
     * layers * loader width. */
    Slot *slots;
    int persistent_slots_per_layer;
    int transient_slots;
    int total_slots;
    int legacy_layout;
    int loader_lanes;
    uint64_t offered_cache_bytes;
    uint64_t dense_cache_budget_bytes;
    uint64_t hot_hysteresis;
    int dense_cache_configured;

    /* #56 aggregate activation/locality trace. */
    uint64_t *usage;
    TraceEvent *trace;
    size_t trace_count, trace_capacity;
    uint64_t trace_dropped, trace_seq;
    char *trace_path;

    unsigned active_leases;
    ColiExpertStoreStats stats;
    pthread_mutex_t mutex;
    pthread_cond_t changed;

    /* End-user progress + I/O diagnostics. These counters are guarded by mutex.
     * They measure physical expert loads, not logical requests. */
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

static char *copy_string(const char *value) {
    if (!value) return NULL;
    size_t n = strlen(value) + 1;
    char *copy = malloc(n);
    if (copy) memcpy(copy, value, n);
    return copy;
}

static int loader_lane_count(void) {
    const char *value = getenv("V4_LOADER_LANES");
    int lanes = value && *value ? atoi(value) : COLI_V4_EXPERT_LOADER_COUNT;
    if (lanes < 1) lanes = COLI_V4_EXPERT_LOADER_COUNT;
    if (lanes > 16) lanes = 16;
    return lanes;
}

static const char *tier_name(SlotTier tier) {
    return tier == SLOT_TIER_PERSISTENT ? "persistent" : "transient";
}

static const char *trace_name(unsigned kind) {
    switch ((TraceKind)kind) {
    case TRACE_REQUEST: return "request";
    case TRACE_HIT: return "hit";
    case TRACE_INFLIGHT_JOIN: return "inflight_join";
    case TRACE_LOAD_BEGIN: return "load_begin";
    case TRACE_LOAD_COMPLETE: return "load_complete";
    case TRACE_LOAD_FAILED: return "load_failed";
    case TRACE_EVICT: return "evict";
    case TRACE_RELEASE: return "release";
    case TRACE_SLOT_WAIT: return "slot_wait";
    default: return "unknown";
    }
}

static Record *record_for(State *s, ColiExpertKey k) {
    if (k.layer < 0 || k.layer >= s->layers ||
        k.expert < 0 || k.expert >= s->experts)
        return NULL;
    return &s->records[(size_t)k.layer * s->experts + k.expert];
}

static uint64_t *usage_for(State *s, ColiExpertKey key) {
    if (!s->usage || key.layer < 0 || key.layer >= s->layers ||
        key.expert < 0 || key.expert >= s->experts)
        return NULL;
    return &s->usage[(size_t)key.layer * s->experts + key.expert];
}

static uint64_t slot_usage(State *s, const Slot *slot) {
    if (!slot || slot->layer < 0 || slot->expert < 0) return 0;
    uint64_t *value = usage_for(
        s, (ColiExpertKey){slot->layer, slot->expert});
    return value ? *value : 0;
}

static Slot *persistent_for(State *s, int layer) {
    if (!s->persistent_slots_per_layer || layer < 0 || layer >= s->layers)
        return NULL;
    return s->slots + (size_t)layer * s->persistent_slots_per_layer;
}

static Slot *transient_base(State *s) {
    return s->slots + (size_t)s->layers * s->persistent_slots_per_layer;
}

static int same_key(const Slot *slot, ColiExpertKey key) {
    return slot && slot->state != SLOT_EMPTY &&
           slot->layer == key.layer && slot->expert == key.expert;
}

static void trace_add_locked(State *s, TraceKind kind, ColiExpertKey key,
                             const Slot *slot, uint64_t bytes) {
    if (!s->trace) return;
    if (s->trace_count >= s->trace_capacity) {
        s->trace_dropped++;
        return;
    }
    TraceEvent *event = &s->trace[s->trace_count++];
    event->seq = ++s->trace_seq;
    event->generation = slot ? slot->generation : 0;
    event->bytes = bytes;
    event->layer = key.layer;
    event->expert = key.expert;
    event->kind = (unsigned char)kind;
    event->tier = (unsigned char)(slot ? slot->tier : SLOT_TIER_TRANSIENT);
    event->has_tier = slot != NULL;
}

static void trace_flush(State *s) {
    if (!s || !s->trace_path) return;
    FILE *file = fopen(s->trace_path, "w");
    if (!file) {
        fprintf(stderr, "v4_expert_trace status=error path=%s reason=open\n",
                s->trace_path);
        return;
    }

    fprintf(file,
            "{\"schema\":\"colibri.v4.expert_trace.v1\","
            "\"build\":\"%s\",\"record_bytes\":%llu,"
            "\"events\":%llu,\"dropped\":%llu}\n",
            COLI_V4_GIT_SHA,
            (unsigned long long)s->record_bytes,
            (unsigned long long)s->trace_count,
            (unsigned long long)s->trace_dropped);
    for (size_t i = 0; i < s->trace_count; i++) {
        const TraceEvent *event = &s->trace[i];
        if (event->has_tier) {
            fprintf(file,
                    "{\"seq\":%llu,\"event\":\"%s\","
                    "\"layer\":%d,\"expert\":%d,\"tier\":\"%s\","
                    "\"generation\":%llu,\"bytes\":%llu}\n",
                    (unsigned long long)event->seq,
                    trace_name(event->kind), event->layer, event->expert,
                    tier_name((SlotTier)event->tier),
                    (unsigned long long)event->generation,
                    (unsigned long long)event->bytes);
        } else {
            fprintf(file,
                    "{\"seq\":%llu,\"event\":\"%s\","
                    "\"layer\":%d,\"expert\":%d,\"tier\":null,"
                    "\"generation\":0,\"bytes\":%llu}\n",
                    (unsigned long long)event->seq,
                    trace_name(event->kind), event->layer, event->expert,
                    (unsigned long long)event->bytes);
        }
    }

    /* Layer summaries make activation skew useful immediately without requiring
     * a second inference run or parsing every event. */
    if (s->usage) {
        for (int layer = 0; layer < s->layers; layer++) {
            int hot = -1;
            uint64_t hot_count = 0, total = 0;
            for (int expert = 0; expert < s->experts; expert++) {
                uint64_t count = s->usage[(size_t)layer * s->experts + expert];
                total += count;
                if (count > hot_count) {
                    hot_count = count;
                    hot = expert;
                }
            }
            fprintf(file,
                    "{\"event\":\"layer_summary\",\"layer\":%d,"
                    "\"requests\":%llu,\"hot_expert\":%d,"
                    "\"hot_requests\":%llu}\n",
                    layer, (unsigned long long)total, hot,
                    (unsigned long long)hot_count);
        }
    }
    fclose(file);
    fprintf(stderr,
            "v4_expert_trace status=written path=%s events=%llu dropped=%llu\n",
            s->trace_path,
            (unsigned long long)s->trace_count,
            (unsigned long long)s->trace_dropped);
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

/* Called with s->mutex held. */
static void io_begin_locked(State *s, ColiExpertKey key) {
    const time_t now = time(NULL);
    if (!s->io_started_at) {
        s->io_started_at = now;
        s->io_last_report_at = now;
        if (s->progress_enabled) {
            fprintf(stderr,
                    "v4_progress phase=expert-stream status=started "
                    "record=%.2fMiB transient_slots=%d "
                    "persistent_slots_per_layer=%d layer=%d expert=%d\n",
                    (double)s->record_bytes / (1024.0 * 1024.0),
                    s->transient_slots, s->persistent_slots_per_layer,
                    key.layer, key.expert);
            fflush(stderr);
        }
    }
    s->io_inflight++;
    s->stats.loads_started++;
    if (s->io_inflight > s->io_peak_inflight)
        s->io_peak_inflight = s->io_inflight;
    if (s->io_peak_inflight > s->stats.peak_inflight)
        s->stats.peak_inflight = s->io_peak_inflight;
}

/* Called with s->mutex held after a storage operation. */
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
            "persistent_hits=%llu transient_hits=%llu joins=%llu "
            "layer=%d/%d expert=%d\n",
            elapsed, (unsigned long long)s->io_reads, gib, expert_mib_s,
            s->io_inflight, s->io_peak_inflight, hit_pct,
            (unsigned long long)s->stats.persistent_hits,
            (unsigned long long)s->stats.transient_hits,
            (unsigned long long)s->stats.inflight_joins,
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
                const Slot *slot) {
    const ColiExpertMatrixInfo *gate = NULL, *up = NULL, *down = NULL;
    for (int i = 0; i < 3; i++) {
        const ColiExpertMatrixInfo *m = &r->info.matrices[i];
        if (m->role == 1) gate = m;
        else if (m->role == 2) up = m;
        else if (m->role == 3) down = m;
    }
    if (!gate || !up || !down ||
        tensor_format(gate, &v->gate, slot->data) ||
        tensor_format(up, &v->up, slot->data) ||
        tensor_format(down, &v->down, slot->data))
        return -1;
    v->key = k;
    v->lease = (void *)slot;
    v->lease_generation = slot->generation;
    return 0;
}

static Slot *find_exact_locked(State *s, ColiExpertKey key) {
    Slot *resident = NULL;
    Slot *loading = NULL;
    Slot *persistent = persistent_for(s, key.layer);
    for (int i = 0; persistent && i < s->persistent_slots_per_layer; i++) {
        Slot *slot = &persistent[i];
        if (!same_key(slot, key)) continue;
        if (slot->state == SLOT_RESIDENT) resident = slot;
        else if (slot->state == SLOT_LOADING) loading = slot;
    }
    Slot *transient = transient_base(s);
    for (int i = 0; i < s->transient_slots; i++) {
        Slot *slot = &transient[i];
        if (!same_key(slot, key)) continue;
        if (slot->state == SLOT_RESIDENT) resident = slot;
        else if (slot->state == SLOT_LOADING) loading = slot;
    }
    return resident ? resident : loading;
}

static Slot *choose_persistent_locked(State *s, ColiExpertKey key) {
    Slot *slots = persistent_for(s, key.layer);
    if (!slots) return NULL;

    Slot *empty = NULL;
    Slot *victim = NULL;
    uint64_t victim_usage = UINT64_MAX;
    for (int i = 0; i < s->persistent_slots_per_layer; i++) {
        Slot *slot = &slots[i];
        if (slot->refs || slot->state == SLOT_LOADING) continue;
        if (slot->state == SLOT_EMPTY) {
            empty = slot;
            break;
        }
        uint64_t usage = slot_usage(s, slot);
        if (!victim || usage < victim_usage ||
            (usage == victim_usage && slot->last_use < victim->last_use)) {
            victim = slot;
            victim_usage = usage;
        }
    }
    if (empty) return empty;
    if (!victim) return NULL;
    if (s->legacy_layout) return victim;

    uint64_t *candidate = usage_for(s, key);
    uint64_t candidate_usage = candidate ? *candidate : 0;
    return candidate_usage > victim_usage + s->hot_hysteresis ? victim : NULL;
}

static Slot *choose_transient_locked(State *s) {
    Slot *slots = transient_base(s);
    Slot *victim = NULL;
    for (int i = 0; i < s->transient_slots; i++) {
        Slot *slot = &slots[i];
        if (slot->refs || slot->state == SLOT_LOADING) continue;
        if (slot->state == SLOT_EMPTY) return slot;
        if (!victim || slot->last_use < victim->last_use) victim = slot;
    }
    return victim;
}

static int prepare_slot_locked(State *s, Slot *slot, ColiExpertKey key) {
    if (!slot) return -1;
    if (!slot->data) {
        if (s->slot_bytes > SIZE_MAX ||
            posix_memalign((void **)&slot->data, 16384,
                           (size_t)s->slot_bytes))
            return -1;
#ifdef COLI_METAL
        if (coli_metal_init())
            coli_metal_register(slot->data, (size_t)s->slot_bytes);
#endif
        s->stats.resident_bytes += s->slot_bytes;
    }

    if (slot->state == SLOT_RESIDENT) {
        ColiExpertKey old = {slot->layer, slot->expert};
        s->stats.evictions++;
        trace_add_locked(s, TRACE_EVICT, old, slot, s->record_bytes);
    }
    slot->generation++;
    if (!slot->generation) slot->generation = 1; /* wrap away from sentinel 0 */
    slot->layer = key.layer;
    slot->expert = key.expert;
    slot->state = SLOT_LOADING;
    slot->last_use = ++s->clock;
    io_begin_locked(s, key);
    trace_add_locked(s, TRACE_LOAD_BEGIN, key, slot, s->record_bytes);
    return 0;
}

static int lease_resident_locked(State *s, Slot *slot, Record *record,
                                 ColiExpertKey key, ColiExpertView *view,
                                 int count_hit) {
    if (!slot || slot->state != SLOT_RESIDENT || !same_key(slot, key))
        return -1;
    if (count_hit) {
        s->stats.hits++;
        if (slot->tier == SLOT_TIER_PERSISTENT)
            s->stats.persistent_hits++;
        else
            s->stats.transient_hits++;
        trace_add_locked(s, TRACE_HIT, key, slot, 0);
    }
    slot->refs++;
    slot->last_use = ++s->clock;
    s->active_leases++;
    if (fill(view, key, record, slot)) {
        slot->refs--;
        s->active_leases--;
        return -1;
    }
    return 0;
}

static int lookup(ColiExpertStore *store, ColiExpertKey key,
                  ColiExpertView *view) {
    State *s = store ? store->state : NULL;
    Record *record = s ? record_for(s, key) : NULL;
    if (!s || !record || !view) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }

    pthread_mutex_lock(&s->mutex);
    s->stats.requests++;
    uint64_t *usage = usage_for(s, key);
    if (usage && *usage != UINT64_MAX) (*usage)++;
    trace_add_locked(s, TRACE_REQUEST, key, NULL, 0);

retry:
    {
        Slot *exact = find_exact_locked(s, key);
        if (exact && exact->state == SLOT_RESIDENT) {
            int result = lease_resident_locked(s, exact, record, key, view, 1);
            pthread_mutex_unlock(&s->mutex);
            if (result) memset(view, 0, sizeof(*view));
            return result;
        }
        if (exact && exact->state == SLOT_LOADING) {
            uint64_t generation = exact->generation;
            s->stats.inflight_joins++;
            trace_add_locked(s, TRACE_INFLIGHT_JOIN, key, exact, 0);
            while (exact->state == SLOT_LOADING &&
                   exact->generation == generation && same_key(exact, key))
                pthread_cond_wait(&s->changed, &s->mutex);
            if (exact->state == SLOT_RESIDENT &&
                exact->generation == generation && same_key(exact, key)) {
                int result = lease_resident_locked(
                    s, exact, record, key, view, 1);
                pthread_mutex_unlock(&s->mutex);
                if (result) memset(view, 0, sizeof(*view));
                return result;
            }
            goto retry;
        }
    }

    Slot *slot = choose_persistent_locked(s, key);
    if (!slot) slot = choose_transient_locked(s);
    if (!slot) {
        /* The global transient floor is loader_lanes + one consumer lease, but
         * future multi-session execution can temporarily exceed that. Wait for
         * a generation to become reclaimable instead of turning pressure into
         * an inference error. */
        s->stats.slot_waits++;
        trace_add_locked(s, TRACE_SLOT_WAIT, key, NULL, 0);
        pthread_cond_wait(&s->changed, &s->mutex);
        goto retry;
    }

    if (prepare_slot_locked(s, slot, key)) {
        pthread_mutex_unlock(&s->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }
    uint64_t generation = slot->generation;
    pthread_mutex_unlock(&s->mutex);

    char error[256];
#ifdef __APPLE__
    const char *direct = getenv("COLI_V4_DIRECT");
    coli_v4_expert_io_active = !direct || atoi(direct) != 0;
#endif
    int load_bad = coli_executor_load_expert(
        s->executor, key.layer, key.expert, slot->data,
        (size_t)s->record_bytes, error, sizeof(error));
#ifdef __APPLE__
    coli_v4_expert_io_active = 0;
#endif

    pthread_mutex_lock(&s->mutex);
    /* This slot cannot be repurposed while SLOT_LOADING, so a generation/key
     * mismatch here indicates internal corruption rather than normal churn. */
    if (slot->generation != generation || !same_key(slot, key) ||
        slot->state != SLOT_LOADING) {
        io_finish_locked(s, key, 0);
        pthread_cond_broadcast(&s->changed);
        pthread_mutex_unlock(&s->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }

    if (load_bad) {
        io_finish_locked(s, key, 0);
        trace_add_locked(s, TRACE_LOAD_FAILED, key, slot, 0);
        slot->state = SLOT_EMPTY;
        slot->layer = -1;
        slot->expert = -1;
        fprintf(stderr,
                "v4_coli expert-load failed layer=%d expert=%d: %s\n",
                key.layer, key.expert, error);
        pthread_cond_broadcast(&s->changed);
        pthread_mutex_unlock(&s->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }

    slot->state = SLOT_RESIDENT;
    slot->last_use = ++s->clock;
    s->stats.misses++;
    s->stats.bytes_read += s->record_bytes;
    io_finish_locked(s, key, 1);
    trace_add_locked(s, TRACE_LOAD_COMPLETE, key, slot, s->record_bytes);

    /* The caller that performed the physical load owns the first lease; this
     * is a miss, not a cache hit. Joiners will count as hits when they wake. */
    int result = lease_resident_locked(s, slot, record, key, view, 0);
    pthread_cond_broadcast(&s->changed);
    pthread_mutex_unlock(&s->mutex);
    if (result) {
        fprintf(stderr,
                "v4_coli expert-view invalid layer=%d expert=%d\n",
                key.layer, key.expert);
        memset(view, 0, sizeof(*view));
        return -1;
    }
    return 0;
}

static void release(ColiExpertStore *store, ColiExpertView *view) {
    State *s = store ? store->state : NULL;
    Slot *slot = view ? view->lease : NULL;
    if (!s || !slot) return;

    pthread_mutex_lock(&s->mutex);
    if (view->lease_generation != slot->generation ||
        view->key.layer != slot->layer || view->key.expert != slot->expert) {
        fprintf(stderr,
                "v4_residency stale-lease layer=%d expert=%d "
                "lease_generation=%llu slot_generation=%llu\n",
                view->key.layer, view->key.expert,
                (unsigned long long)view->lease_generation,
                (unsigned long long)slot->generation);
        pthread_mutex_unlock(&s->mutex);
        return;
    }
    if (slot->refs) slot->refs--;
    if (s->active_leases) s->active_leases--;
    slot->last_use = ++s->clock;
    trace_add_locked(s, TRACE_RELEASE, view->key, slot, 0);
    pthread_cond_broadcast(&s->changed);
    pthread_mutex_unlock(&s->mutex);
}

/* #57 keeps the policy boundary explicit. COLI prefetch is still advisory and
 * currently does not initiate storage work; blocking misses use the deduplicated
 * state machine above. A later #18 integration can enqueue here without changing
 * lease/generation semantics. */
static int prefetch(ColiExpertStore *store, const ColiExpertKey *keys, size_t n) {
    (void)store;
    (void)keys;
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

static void destroy(ColiExpertStore *store) {
    State *s = store ? store->state : NULL;
    if (s) {
        if (s->progress_enabled && s->io_started_at && s->io_reads) {
            double elapsed = difftime(time(NULL), s->io_started_at);
            if (elapsed < 1.0) elapsed = 1.0;
            fprintf(stderr,
                    "v4_progress phase=expert-stream status=done elapsed=%.0fs "
                    "reads=%llu bytes=%.2fGiB expert_effective=%.1fMiB/s "
                    "peak_inflight=%u\n",
                    elapsed, (unsigned long long)s->io_reads,
                    (double)s->io_bytes / (1024.0 * 1024.0 * 1024.0),
                    ((double)s->io_bytes / (1024.0 * 1024.0)) / elapsed,
                    s->io_peak_inflight);
        }

        if (s->executor) {
            fprintf(stderr,
                    "v4_residency_stats requests=%llu hits=%llu misses=%llu "
                    "persistent_hits=%llu transient_hits=%llu inflight_joins=%llu "
                    "loads=%llu load_failures=%llu evictions=%llu slot_waits=%llu "
                    "expert_capacity=%.2fGiB dense_cache_budget=%.2fGiB\n",
                    (unsigned long long)s->stats.requests,
                    (unsigned long long)s->stats.hits,
                    (unsigned long long)s->stats.misses,
                    (unsigned long long)s->stats.persistent_hits,
                    (unsigned long long)s->stats.transient_hits,
                    (unsigned long long)s->stats.inflight_joins,
                    (unsigned long long)s->stats.loads_started,
                    (unsigned long long)s->stats.loads_failed,
                    (unsigned long long)s->stats.evictions,
                    (unsigned long long)s->stats.slot_waits,
                    s->stats.capacity_bytes / (1024.0 * 1024.0 * 1024.0),
                    s->dense_cache_budget_bytes / (1024.0 * 1024.0 * 1024.0));
        }

        trace_flush(s);
        if (s->dense_cache_configured)
            coli_v4_dense_cache_reset();

        if (s->slots) {
            for (int i = 0; i < s->total_slots; i++) {
#ifdef COLI_METAL
                if (s->slots[i].data)
                    coli_metal_unregister(s->slots[i].data);
#endif
                compat_aligned_free(s->slots[i].data);
            }
        }
        pthread_cond_destroy(&s->changed);
        pthread_mutex_destroy(&s->mutex);
        coli_executor_close(s->executor);
        free(s->trace_path);
        free(s->trace);
        free(s->usage);
        free(s->slots);
        free(s->records);
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
    ColiRuntimeTarget apple8_runtime;

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
    /* Publish ownership immediately so every later `goto bad` tears down the
     * partially initialized state rather than leaking the executor/index. */
    store->state = s;

    s->progress_enabled = !getenv("V4_PROGRESS") ||
                          atoi(getenv("V4_PROGRESS")) != 0;
    s->progress_interval_s = 5;
    {
        const char *p = getenv("V4_PROGRESS_INTERVAL");
        if (p && *p) {
            int v = atoi(p);
            if (v >= 1 && v <= 60) s->progress_interval_s = v;
        }
    }
    s->loader_lanes = loader_lane_count();
    s->hot_hysteresis = 2;
    {
        const char *value = getenv("V4_HOT_EXPERT_HYSTERESIS");
        if (value && *value) {
            long long parsed = atoll(value);
            if (parsed >= 0) s->hot_hysteresis = (uint64_t)parsed;
        }
    }

    xo.required_profile = o->required_profile;
    xo.checksum_policy = getenv("COLI_VERIFY_RECORDS") &&
                         atoi(getenv("COLI_VERIFY_RECORDS"))
        ? COLI_CSF_CHECKSUM_RECORD_ON_READ
        : COLI_CSF_CHECKSUM_MANIFEST_ONLY;
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    /* Apple8 executor open requires a populated runtime target (fail-closed
     * contract, #140/#141). Fill it when the store opens an Apple8 package. */
    if (!xo.runtime_target && o->required_profile &&
        !strcmp(o->required_profile,
                COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1)) {
        memset(&apple8_runtime, 0, sizeof(apple8_runtime));
        apple8_runtime.profile_name = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
        apple8_runtime.target_os = COLI_TARGET_OS_MACOS;
        apple8_runtime.target_arch = COLI_TARGET_ARCH_ARM64;
        apple8_runtime.backend = COLI_TARGET_BACKEND_METAL;
        apple8_runtime.gpu_kind = COLI_TARGET_GPU_APPLE_FAMILY;
        apple8_runtime.cpu_feature_mask = COLI_TARGET_CPU_ARM64_ASIMD;
        apple8_runtime.gpu_family = COLI_APPLE8_GPU_FAMILY_MIN;
        apple8_runtime.runtime_features = COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY |
                                          COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE;
        apple8_runtime.target_profile_abi = COLI_TARGET_PROFILE_ABI_APPLE8_V1;
        apple8_runtime.execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1;
        apple8_runtime.kernel_abi = COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1;
        apple8_runtime.target_class = COLI_TARGET_CLASS_APPLE8_METAL_V1;
        apple8_runtime.max_record_alignment = COLI_APPLE8_RECORD_ALIGNMENT;
        apple8_runtime.max_io_granularity = COLI_APPLE8_IO_GRANULARITY;
        apple8_runtime.max_resident_alignment = COLI_APPLE8_RESIDENT_ALIGNMENT;
        xo.runtime_target = &apple8_runtime;
    }
#endif
    if (coli_executor_open(&s->executor, o->package_dir, &xo, e, n))
        goto bad;

    s->layers = o->layers;
    s->experts = o->experts_per_layer;
    s->offered_cache_bytes = o->cache_bytes;
    s->records = calloc((size_t)s->layers * s->experts, sizeof(*s->records));
    s->usage = calloc((size_t)s->layers * s->experts, sizeof(*s->usage));
    if (!s->records || !s->usage) {
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

    uint64_t offered_slots = o->cache_bytes / s->slot_bytes;
    const char *policy = getenv("V4_RESIDENCY_POLICY");
    s->legacy_layout = policy && !strcmp(policy, "legacy");
    if (s->legacy_layout) {
        /* Reproduce the old store's exact slot-count formula for a trustworthy
         * A/B. Physical allocations remain 16 KiB rounded for Metal, just as
         * they were on #63; the logical cache count is based on record bytes. */
        if (s->record_bytes > UINT64_MAX / (uint64_t)s->layers) {
            fail(e, n, "COLI legacy expert cache size overflow");
            goto bad;
        }
        uint64_t per_layer_bytes = (uint64_t)s->layers * s->record_bytes;
        uint64_t per_layer = o->cache_bytes / per_layer_bytes;
        if (!per_layer) {
            fail(e, n, "COLI legacy cache budget cannot hold one expert per layer");
            goto bad;
        }
        if (per_layer > (uint64_t)s->experts) per_layer = (uint64_t)s->experts;
        s->persistent_slots_per_layer = (int)per_layer;
        s->transient_slots = 0;
        s->total_slots = s->layers * s->persistent_slots_per_layer;
        s->dense_cache_budget_bytes = 0;
    } else {
        /* One global pool must cover every loader plus the expert currently
         * consumed by compute. This is the real concurrency floor previously
         * multiplied by every layer. */
        s->transient_slots = s->loader_lanes + 1;
        if (offered_slots < (uint64_t)s->transient_slots) {
            fail(e, n,
                 "COLI cache budget cannot hold %d global transient expert slots",
                 s->transient_slots);
            goto bad;
        }

        /* Dense deterministic tensors have measured benefit/byte near 1.0:
         * every resident byte avoids roughly one recurring disk byte per token.
         * The measured 5-slot/layer expert cache saved only ~0.15 recurring
         * bytes per resident byte, so balanced mode starts dense-first and lets
         * #56 traces justify persistent expert capacity explicitly. */
        int requested_persistent = 0;
        {
            const char *value = getenv("V4_PERSISTENT_EXPERT_SLOTS_PER_LAYER");
            if (value && *value) requested_persistent = atoi(value);
            if (requested_persistent < 0) requested_persistent = 0;
            if (requested_persistent > 16) requested_persistent = 16;
        }
        uint64_t remaining_slots = offered_slots - (uint64_t)s->transient_slots;
        uint64_t max_persistent = remaining_slots / (uint64_t)s->layers;
        if (max_persistent > (uint64_t)s->experts)
            max_persistent = (uint64_t)s->experts;
        if ((uint64_t)requested_persistent > max_persistent)
            requested_persistent = (int)max_persistent;
        s->persistent_slots_per_layer = requested_persistent;
        s->total_slots = s->transient_slots +
            s->layers * s->persistent_slots_per_layer;
        uint64_t expert_capacity = (uint64_t)s->total_slots * s->slot_bytes;
        s->dense_cache_budget_bytes = o->cache_bytes > expert_capacity
            ? o->cache_bytes - expert_capacity : 0;
    }

    if (s->total_slots < 1 ||
        (uint64_t)s->total_slots > SIZE_MAX / sizeof(*s->slots)) {
        fail(e, n, "invalid COLI residency slot count");
        goto bad;
    }
    s->slots = calloc((size_t)s->total_slots, sizeof(*s->slots));
    if (!s->slots) {
        fail(e, n, "out of memory creating COLI residency slots");
        goto bad;
    }

    for (int layer = 0; layer < s->layers; layer++) {
        Slot *slots = persistent_for(s, layer);
        for (int i = 0; i < s->persistent_slots_per_layer; i++) {
            slots[i].layer = -1;
            slots[i].expert = -1;
            slots[i].home_layer = layer;
            slots[i].tier = SLOT_TIER_PERSISTENT;
        }
    }
    Slot *transient = transient_base(s);
    for (int i = 0; i < s->transient_slots; i++) {
        transient[i].layer = -1;
        transient[i].expert = -1;
        transient[i].home_layer = -1;
        transient[i].tier = SLOT_TIER_TRANSIENT;
    }
    s->stats.capacity_bytes = (uint64_t)s->total_slots * s->slot_bytes;

    /* Optional bounded detailed trace. Events are kept in RAM and flushed only
     * at teardown so tracing never synchronously writes on the inference path. */
    {
        const char *path = getenv("V4_EXPERT_TRACE");
        if (path && *path) {
            size_t capacity = 65536;
            const char *cap = getenv("V4_EXPERT_TRACE_CAP");
            if (cap && *cap) {
                unsigned long long parsed = strtoull(cap, NULL, 10);
                if (parsed >= 1024 && parsed <= 10000000)
                    capacity = (size_t)parsed;
            }
            s->trace = calloc(capacity, sizeof(*s->trace));
            s->trace_path = copy_string(path);
            if (s->trace && s->trace_path) {
                s->trace_capacity = capacity;
                fprintf(stderr,
                        "v4_expert_trace status=buffering path=%s capacity=%llu\n",
                        s->trace_path, (unsigned long long)capacity);
            } else {
                free(s->trace);
                free(s->trace_path);
                s->trace = NULL;
                s->trace_path = NULL;
            }
        }
    }

    coli_v4_dense_cache_configure(s->dense_cache_budget_bytes);
    s->dense_cache_configured = 1;
    print_execution_mode();
    fprintf(stderr,
            "v4_residency policy=%s offered=%.2fGiB expert_capacity=%.2fGiB "
            "dense_cache=%.2fGiB transient_slots=%d "
            "persistent_slots_per_layer=%d total_expert_slots=%d "
            "loader_lanes=%d hot_hysteresis=%llu\n",
            s->legacy_layout ? "legacy" : "balanced",
            s->offered_cache_bytes / (1024.0 * 1024.0 * 1024.0),
            s->stats.capacity_bytes / (1024.0 * 1024.0 * 1024.0),
            s->dense_cache_budget_bytes / (1024.0 * 1024.0 * 1024.0),
            s->transient_slots, s->persistent_slots_per_layer,
            s->total_slots, s->loader_lanes,
            (unsigned long long)s->hot_hysteresis);
    if (s->progress_enabled) {
        fprintf(stderr,
                "v4_progress enabled=1 interval=%ds "
                "metric=expert-bytes-per-wall-second "
                "hint=V4_PROGRESS=0-to-disable\n",
                s->progress_interval_s);
    }

    store->ops = &ops;
    *out = store;
    return 0;

bad:
    destroy(store);
    return -1;
}

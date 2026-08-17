#include "coli_v4_static.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DENSE_CACHE_BUCKETS 2048u

typedef struct DenseCacheEntry {
    char *name;
    unsigned char *data;
    uint64_t resident_bytes;
    uint64_t stored_bytes;
    uint64_t last_used;
    struct DenseCacheEntry *next;
    struct DenseCacheEntry *hash_next;
} DenseCacheEntry;

static struct {
    pthread_mutex_t mutex;
    DenseCacheEntry *entries;
    DenseCacheEntry *buckets[DENSE_CACHE_BUCKETS];
    ColiV4DenseCacheStats stats;
    uint64_t clock;
    uint64_t hit_copy_bytes;
    uint64_t hit_copy_ns;
    long double min_score;
} g_dense_cache = { PTHREAD_MUTEX_INITIALIZER, NULL, {0}, {0}, 0, 0, 0, 0.0L };

static int bad(char *e, size_t n, const char *f, const char *s) {
    if (e && n) snprintf(e, n, f, s);
    return -1;
}

static uint16_t fmt(ColiSafetensorsDType d) {
    switch (d) {
    case COLI_ST_BF16: return COLI_CSF_MATH_BF16;
    case COLI_ST_F32: return COLI_CSF_MATH_F32;
    case COLI_ST_F8_E4M3: return COLI_CSF_MATH_FP8_E4M3FN;
    /* E8M0 is an exponent byte, not a floating-point tensor payload. */
    case COLI_ST_F8_E8M0: return COLI_CSF_MATH_U8;
    case COLI_ST_I64: return COLI_CSF_MATH_I64;
    default: return COLI_CSF_MATH_INVALID;
    }
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static unsigned cache_hash(const char *name) {
    uint32_t h = UINT32_C(2166136261);
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
        h ^= *p;
        h *= UINT32_C(16777619);
    }
    return h & (DENSE_CACHE_BUCKETS - 1u);
}

static long double cache_score(uint64_t stored, uint64_t resident) {
    return resident ? (long double)stored / (long double)resident : 0.0L;
}

static void cache_entry_free(DenseCacheEntry *entry) {
    if (!entry) return;
    free(entry->name);
    free(entry->data);
    free(entry);
}

static DenseCacheEntry *cache_find_locked(const char *name) {
    for (DenseCacheEntry *entry = g_dense_cache.buckets[cache_hash(name)];
         entry; entry = entry->hash_next)
        if (strcmp(entry->name, name) == 0) return entry;
    return NULL;
}

static void cache_recompute_min_score_locked(void) {
    if (!g_dense_cache.entries) {
        g_dense_cache.min_score = 0.0L;
        return;
    }
    long double value = cache_score(g_dense_cache.entries->stored_bytes,
                                    g_dense_cache.entries->resident_bytes);
    for (DenseCacheEntry *entry = g_dense_cache.entries->next;
         entry; entry = entry->next) {
        long double score = cache_score(entry->stored_bytes, entry->resident_bytes);
        if (score < value) value = score;
    }
    g_dense_cache.min_score = value;
}

/*
 * Dense/static access is a deterministic scan over the same layer records on
 * every token. Ordinary LRU is pathological when the working set is larger
 * than the cache: the next scan evicts exactly the objects that would have
 * been reused later in that scan. Residency is therefore an admission
 * problem, not a recency problem.
 *
 * Once the budget is full, a candidate may displace only an object with a
 * STRICTLY lower recurring-I/O-saved/resident-byte score. Equal-value objects
 * remain pinned. This makes the cache scan-resistant after its warmup pass.
 */
static DenseCacheEntry *cache_lower_value_victim_locked(long double candidate_score) {
    if (g_dense_cache.entries && candidate_score <= g_dense_cache.min_score)
        return NULL;
    DenseCacheEntry *victim = NULL;
    long double victim_score = 0.0L;
    for (DenseCacheEntry *entry = g_dense_cache.entries; entry; entry = entry->next) {
        long double score = cache_score(entry->stored_bytes, entry->resident_bytes);
        if (!(score < candidate_score)) continue;
        if (!victim || score < victim_score ||
            (score == victim_score && entry->last_used < victim->last_used)) {
            victim = entry;
            victim_score = score;
        }
    }
    return victim;
}

static void cache_remove_locked(DenseCacheEntry *victim) {
    DenseCacheEntry **link = &g_dense_cache.entries;
    while (*link && *link != victim) link = &(*link)->next;
    if (!*link) return;
    *link = victim->next;

    unsigned bucket = cache_hash(victim->name);
    DenseCacheEntry **hash_link = &g_dense_cache.buckets[bucket];
    while (*hash_link && *hash_link != victim) hash_link = &(*hash_link)->hash_next;
    if (*hash_link) *hash_link = victim->hash_next;

    if (g_dense_cache.stats.resident_bytes >= victim->resident_bytes)
        g_dense_cache.stats.resident_bytes -= victim->resident_bytes;
    else
        g_dense_cache.stats.resident_bytes = 0;
    g_dense_cache.stats.evictions++;
    cache_entry_free(victim);
    cache_recompute_min_score_locked();
}

void coli_v4_dense_cache_configure(uint64_t budget_bytes) {
    pthread_mutex_lock(&g_dense_cache.mutex);
    g_dense_cache.stats.budget_bytes = budget_bytes;
    while (g_dense_cache.stats.resident_bytes > budget_bytes && g_dense_cache.entries) {
        DenseCacheEntry *victim = NULL;
        for (DenseCacheEntry *entry = g_dense_cache.entries; entry; entry = entry->next)
            if (!victim || entry->last_used < victim->last_used) victim = entry;
        cache_remove_locked(victim);
    }
    pthread_mutex_unlock(&g_dense_cache.mutex);
}

void coli_v4_dense_cache_stats(ColiV4DenseCacheStats *stats) {
    if (!stats) return;
    pthread_mutex_lock(&g_dense_cache.mutex);
    *stats = g_dense_cache.stats;
    pthread_mutex_unlock(&g_dense_cache.mutex);
}

void coli_v4_dense_cache_shutdown(void) {
    pthread_mutex_lock(&g_dense_cache.mutex);
    if (g_dense_cache.hit_copy_bytes) {
        double ms = g_dense_cache.hit_copy_ns / 1000000.0;
        double gib = g_dense_cache.hit_copy_bytes / 1073741824.0;
        double gib_s = g_dense_cache.hit_copy_ns
            ? gib / ((double)g_dense_cache.hit_copy_ns / 1000000000.0) : 0.0;
        fprintf(stderr,
                "v4_dense_cache_copy bytes=%.2fGiB time_ms=%.3f bandwidth=%.2fGiB/s\n",
                gib, ms, gib_s);
    }
    DenseCacheEntry *entry = g_dense_cache.entries;
    while (entry) {
        DenseCacheEntry *next = entry->next;
        cache_entry_free(entry);
        entry = next;
    }
    g_dense_cache.entries = NULL;
    memset(g_dense_cache.buckets, 0, sizeof(g_dense_cache.buckets));
    g_dense_cache.stats.resident_bytes = 0;
    g_dense_cache.min_score = 0.0L;
    g_dense_cache.hit_copy_bytes = 0;
    g_dense_cache.hit_copy_ns = 0;
    pthread_mutex_unlock(&g_dense_cache.mutex);
}

/* Returns 1 on hit, 0 on miss. output is always caller-owned for now.
 * Copy cost is measured under V4_PROFILE so the next A/B quantifies exactly
 * how much zero-copy borrowed views can recover. */
static int cache_get(const char *name, uint64_t expected_bytes,
                     uint64_t stored_bytes, unsigned char **output) {
    *output = NULL;
    pthread_mutex_lock(&g_dense_cache.mutex);
    DenseCacheEntry *entry = cache_find_locked(name);
    if (!entry || entry->resident_bytes != expected_bytes) {
        g_dense_cache.stats.misses++;
        pthread_mutex_unlock(&g_dense_cache.mutex);
        return 0;
    }
    const uint64_t began = g_coli_v4_profile_on ? now_ns() : 0;
    unsigned char *copy = malloc((size_t)entry->resident_bytes);
    if (!copy) {
        g_dense_cache.stats.misses++;
        pthread_mutex_unlock(&g_dense_cache.mutex);
        return 0;
    }
    memcpy(copy, entry->data, (size_t)entry->resident_bytes);
    if (began) {
        g_dense_cache.hit_copy_ns += now_ns() - began;
        g_dense_cache.hit_copy_bytes += entry->resident_bytes;
    }
    entry->last_used = ++g_dense_cache.clock;
    g_dense_cache.stats.hits++;
    g_dense_cache.stats.stored_bytes_avoided += stored_bytes;
    pthread_mutex_unlock(&g_dense_cache.mutex);
    *output = copy;
    return 1;
}

static void cache_put(const char *name, const unsigned char *data,
                      uint64_t resident_bytes, uint64_t stored_bytes) {
    if (!name || !data || !resident_bytes || resident_bytes > SIZE_MAX) return;
    pthread_mutex_lock(&g_dense_cache.mutex);
    const uint64_t budget = g_dense_cache.stats.budget_bytes;
    if (!budget || resident_bytes > budget || cache_find_locked(name)) {
        pthread_mutex_unlock(&g_dense_cache.mutex);
        return;
    }

    const long double candidate_score = cache_score(stored_bytes, resident_bytes);
    while (g_dense_cache.stats.resident_bytes > budget - resident_bytes) {
        DenseCacheEntry *victim = cache_lower_value_victim_locked(candidate_score);
        if (!victim) {
            pthread_mutex_unlock(&g_dense_cache.mutex);
            return;
        }
        cache_remove_locked(victim);
    }

    DenseCacheEntry *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        pthread_mutex_unlock(&g_dense_cache.mutex);
        return;
    }
    entry->name = strdup(name);
    entry->data = malloc((size_t)resident_bytes);
    if (!entry->name || !entry->data) {
        cache_entry_free(entry);
        pthread_mutex_unlock(&g_dense_cache.mutex);
        return;
    }
    memcpy(entry->data, data, (size_t)resident_bytes);
    entry->resident_bytes = resident_bytes;
    entry->stored_bytes = stored_bytes;
    entry->last_used = ++g_dense_cache.clock;
    entry->next = g_dense_cache.entries;
    g_dense_cache.entries = entry;
    unsigned bucket = cache_hash(name);
    entry->hash_next = g_dense_cache.buckets[bucket];
    g_dense_cache.buckets[bucket] = entry;
    g_dense_cache.stats.resident_bytes += resident_bytes;
    g_dense_cache.stats.inserts++;
    if (g_dense_cache.stats.inserts == 1 || candidate_score < g_dense_cache.min_score)
        g_dense_cache.min_score = candidate_score;
    pthread_mutex_unlock(&g_dense_cache.mutex);
}

int coli_v4_coli_layer_load(ColiExecutor *x, ColiDeepSeekV4LayerWeights *w,
                            const ColiDeepSeekV4Config *c, int layer,
                            char *e, size_t n) {
    const int profiling = g_coli_v4_profile_on;

    if (!x || !w || coli_v4_layer_plan(&w->plan, c, layer, e, n)) return -1;
    memset(w->data, 0, sizeof(w->data));
    memset(&w->stats, 0, sizeof(w->stats));

    for (size_t i = 0; i < w->plan.tensor_count; i++) {
        ColiDeepSeekV4TensorSpec *s = &w->plan.tensors[i];
        const ColiRecordInfo *r = coli_executor_record_by_name(x, s->name);
        ColiTensorInfo t;
        unsigned char *b = NULL;
        uint64_t resident_bytes;

        if (!r || coli_package_tensor_info(coli_executor_package(x), r, &t, e, n) ||
            t.rank != (uint16_t)s->rank || r->math_format != fmt(s->dtype))
            goto fail;
        for (int d = 0; d < s->rank; d++)
            if (t.dims[d] != (uint64_t)s->shape[d]) {
                bad(e, n, "COLI static shape mismatch: %s", s->name);
                goto fail;
            }
        if (r->stored_bytes > SIZE_MAX || t.data_offset > r->stored_bytes ||
            t.data_stored_bytes > r->stored_bytes - t.data_offset) {
            bad(e, n, "COLI static span invalid: %s", s->name);
            goto fail;
        }

        resident_bytes = t.data_stored_bytes;
        if (s->dtype == COLI_ST_F8_E8M0) {
            if (resident_bytes > SIZE_MAX / sizeof(float)) {
                bad(e, n, "COLI static scale span invalid: %s", s->name);
                goto fail;
            }
            resident_bytes *= sizeof(float);
        }

        if (cache_get(s->name, resident_bytes, r->stored_bytes, &b)) {
            w->data[i] = b;
            if (s->dtype == COLI_ST_F8_E8M0)
                w->stats.fp8_scale_bytes += resident_bytes;
            w->stats.total_bytes += resident_bytes;
            w->stats.tensor_count++;
            continue;
        }

        unsigned char *record = malloc((size_t)r->stored_bytes);
        if (!record) {
            bad(e, n, "COLI static allocation failed: %s", s->name);
            goto fail;
        }
        uint64_t began = profiling ? coli_v4_profile_now() : 0;
        if (coli_executor_load_record(x, r, record, (size_t)r->stored_bytes, e, n)) {
            free(record);
            goto fail;
        }
        if (profiling) {
            coli_v4_profile_add(COLI_V4_PROF_DENSE_READ,
                                coli_v4_profile_now() - began);
            coli_v4_profile_add_bytes(COLI_V4_PROF_DENSE_READ, r->stored_bytes);
        }

        if (s->dtype == COLI_ST_F8_E8M0) {
            size_t k = (size_t)t.data_stored_bytes;
            float *out = malloc(k * sizeof(*out));
            if (!out) {
                free(record);
                goto fail;
            }
            for (size_t q = 0; q < k; q++)
                out[q] = record[t.data_offset + q] == 255
                    ? NAN : ldexpf(1.f, (int)record[t.data_offset + q] - 127);
            free(record);
            b = (unsigned char *)out;
            w->stats.fp8_scale_bytes += resident_bytes;
        } else {
            memmove(record, record + t.data_offset, (size_t)t.data_stored_bytes);
            b = record;
        }
        w->data[i] = b;
        w->stats.total_bytes += resident_bytes;
        w->stats.tensor_count++;
        cache_put(s->name, b, resident_bytes, r->stored_bytes);
    }
    return 0;

fail:
    coli_v4_layer_free(NULL, w);
    return -1;
}

int coli_v4_coli_layer_bytes(ColiExecutor *x, const ColiDeepSeekV4Config *c,
                             int layer, uint64_t *bytes, char *e, size_t n) {
    ColiDeepSeekV4LayerPlan p;
    uint64_t total = 0;
    if (!x || !bytes || coli_v4_layer_plan(&p, c, layer, e, n)) return -1;
    for (size_t i = 0; i < p.tensor_count; i++) {
        const ColiRecordInfo *r = coli_executor_record_by_name(x, p.tensors[i].name);
        ColiTensorInfo t;
        uint64_t resident;
        if (!r || coli_package_tensor_info(coli_executor_package(x), r, &t, e, n))
            return bad(e, n, "COLI planner missing tensor: %s", p.tensors[i].name);
        resident = t.data_stored_bytes;
        if (p.tensors[i].dtype == COLI_ST_F8_E8M0) {
            if (resident > UINT64_MAX / sizeof(float))
                return bad(e, n, "COLI planner scale overflow: %s", p.tensors[i].name);
            resident *= sizeof(float);
        }
        if (UINT64_MAX - total < resident)
            return bad(e, n, "COLI planner byte overflow: %s", p.tensors[i].name);
        total += resident;
    }
    *bytes = total;
    return 0;
}

int coli_v4_coli_tensor_load_f32(ColiExecutor *x, ColiFloatTensor *out,
                                 const char *name, char *e, size_t n) {
    const ColiRecordInfo *r;
    ColiTensorInfo t;
    unsigned char *raw = NULL;
    if (!x || !out || !name)
        return bad(e, n, "invalid COLI float tensor: %s", name ? name : "(null)");
    memset(out, 0, sizeof(*out));
    r = coli_executor_record_by_name(x, name);
    if (!r || (r->math_format != COLI_CSF_MATH_F32 &&
               r->math_format != COLI_CSF_MATH_BF16) ||
        coli_package_tensor_info(coli_executor_package(x), r, &t, e, n) ||
        t.rank > COLI_ST_MAX_RANK || !t.data_stored_bytes ||
        t.data_stored_bytes > SIZE_MAX || t.data_decoded_bytes > SIZE_MAX)
        return bad(e, n, "invalid COLI float tensor: %s", name);

    uint64_t count = 1;
    for (uint16_t i = 0; i < t.rank; i++) {
        if (!t.dims[i] || count > UINT64_MAX / t.dims[i])
            return bad(e, n, "COLI tensor shape overflow: %s", name);
        count *= t.dims[i];
        out->shape[i] = (int64_t)t.dims[i];
    }
    if ((r->math_format == COLI_CSF_MATH_F32 &&
         t.data_stored_bytes != count * sizeof(float)) ||
        (r->math_format == COLI_CSF_MATH_BF16 &&
         t.data_stored_bytes != count * sizeof(uint16_t)) ||
        count > SIZE_MAX / sizeof(float))
        return bad(e, n, "COLI tensor size mismatch: %s", name);

    raw = malloc((size_t)t.data_stored_bytes);
    out->data = malloc((size_t)count * sizeof(float));
    if (!raw || !out->data ||
        coli_package_read_range(coli_executor_package(x), r, t.data_offset,
                                raw, (size_t)t.data_stored_bytes, e, n)) {
        free(raw);
        coli_float_tensor_free(out);
        return -1;
    }
    if (r->math_format == COLI_CSF_MATH_F32) {
        memcpy(out->data, raw, (size_t)t.data_stored_bytes);
    } else {
        for (uint64_t i = 0; i < count; i++) {
            uint16_t v;
            uint32_t bits;
            memcpy(&v, raw + i * 2, 2);
            bits = (uint32_t)v << 16;
            memcpy(out->data + i, &bits, 4);
        }
    }
    free(raw);
    out->count = count;
    out->rank = t.rank;
    return 0;
}

#include "coli_v4_static.h"
#include "coli_v4_residency.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Tensor-granular deterministic residency for package-only V4.
 *
 * Package-only layer loads borrow resident payload pointers directly instead of
 * materializing caller-owned memcpy copies on every token. Borrow lifetimes are
 * tracked per cache entry; an entry may be reclaimed only after its final live
 * layer/session borrow releases it. The separately compiled layer-resident unit
 * routes payload free() through coli_v4_layer_payload_free(), which is therefore
 * the single ownership/release seam for both cached and ordinary allocations.
 *
 * This is deliberately an ownership bridge for the current layer API. A future
 * typed mixed-residency execution view can make borrowed ownership explicit in
 * the data structure instead of at the payload-release boundary.
 */
typedef struct {
    char name[COLI_V4_MAX_TENSOR_NAME];
    unsigned char *data;
    uint64_t resident_bytes;
    uint64_t stored_bytes_avoided;
    uint64_t hits;
    uint64_t refs;
} DenseCacheEntry;

typedef struct {
    pthread_mutex_t mutex;
    DenseCacheEntry *entries;
    size_t count;
    size_t capacity;
    uint64_t budget_bytes;
    uint64_t resident_bytes;
    uint64_t pinned_bytes;
    uint64_t hits;
    uint64_t misses;
    uint64_t admissions;
    uint64_t evictions;
    uint64_t evicted_bytes;
    uint64_t rejected_bytes;
    uint64_t bytes_avoided;
    /* Retained in the stable diagnostics contract. Zero-copy borrowing keeps
     * these at zero; a non-zero value would indicate materialization regressed. */
    uint64_t copy_bytes;
    uint64_t copy_ns;
    double minimum_benefit;
} DenseCache;

static DenseCache g_dense_cache = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .minimum_benefit = 0.75,
};

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

static DenseCacheEntry *dense_cache_find_locked(const char *name) {
    for (size_t i = 0; i < g_dense_cache.count; i++)
        if (!strcmp(g_dense_cache.entries[i].name, name))
            return &g_dense_cache.entries[i];
    return NULL;
}

static DenseCacheEntry *dense_cache_find_pointer_locked(
        const void *pointer, size_t *index_out) {
    for (size_t i = 0; i < g_dense_cache.count; i++) {
        if (g_dense_cache.entries[i].data != pointer) continue;
        if (index_out) *index_out = i;
        return &g_dense_cache.entries[i];
    }
    return NULL;
}

/* Reclaim only entries with no live borrowers. Hits are the stable reuse signal
 * already collected by this cache, so evict the least-used idle tensor first;
 * ties prefer the larger allocation to converge on the requested budget with
 * fewer frees. Entry metadata may move, but payload addresses never move while
 * borrowed and no caller keeps a pointer to DenseCacheEntry itself. */
static void dense_cache_evict_unpinned_locked(void) {
    while (g_dense_cache.resident_bytes > g_dense_cache.budget_bytes) {
        size_t victim = SIZE_MAX;
        for (size_t i = 0; i < g_dense_cache.count; i++) {
            DenseCacheEntry *entry = &g_dense_cache.entries[i];
            if (entry->refs) continue;
            if (victim == SIZE_MAX ||
                entry->hits < g_dense_cache.entries[victim].hits ||
                (entry->hits == g_dense_cache.entries[victim].hits &&
                 entry->resident_bytes >
                     g_dense_cache.entries[victim].resident_bytes))
                victim = i;
        }
        if (victim == SIZE_MAX) break;
        DenseCacheEntry dead = g_dense_cache.entries[victim];
        free(dead.data);
        if (g_dense_cache.resident_bytes >= dead.resident_bytes)
            g_dense_cache.resident_bytes -= dead.resident_bytes;
        else
            g_dense_cache.resident_bytes = 0;
        g_dense_cache.evictions++;
        g_dense_cache.evicted_bytes += dead.resident_bytes;
        g_dense_cache.count--;
        if (victim != g_dense_cache.count)
            g_dense_cache.entries[victim] =
                g_dense_cache.entries[g_dense_cache.count];
        memset(&g_dense_cache.entries[g_dense_cache.count], 0,
               sizeof(g_dense_cache.entries[g_dense_cache.count]));
    }
}

/* Used only by the COLI_V4_UNIT_LAYER_RESIDENT object through a target-local
 * `free` macro. This keeps the legacy layer cleanup contract intact while a
 * package-only layer is borrowing immutable resident tensor storage. */
void coli_v4_layer_payload_free(void *pointer) {
    if (!pointer) return;
    int borrowed = 0;
    pthread_mutex_lock(&g_dense_cache.mutex);
    DenseCacheEntry *entry = dense_cache_find_pointer_locked(pointer, NULL);
    if (entry) {
        borrowed = 1;
        if (entry->refs) {
            entry->refs--;
            if (!entry->refs) {
                if (g_dense_cache.pinned_bytes >= entry->resident_bytes)
                    g_dense_cache.pinned_bytes -= entry->resident_bytes;
                else
                    g_dense_cache.pinned_bytes = 0;
            }
        }
        dense_cache_evict_unpinned_locked();
    }
    pthread_mutex_unlock(&g_dense_cache.mutex);
    if (!borrowed) free(pointer);
}

static void dense_cache_clear_locked(void) {
    for (size_t i = 0; i < g_dense_cache.count; i++)
        free(g_dense_cache.entries[i].data);
    free(g_dense_cache.entries);
    g_dense_cache.entries = NULL;
    g_dense_cache.count = 0;
    g_dense_cache.capacity = 0;
    g_dense_cache.resident_bytes = 0;
    g_dense_cache.pinned_bytes = 0;
}

void coli_v4_dense_cache_configure(uint64_t budget_bytes) {
    pthread_mutex_lock(&g_dense_cache.mutex);
    dense_cache_clear_locked();
    g_dense_cache.budget_bytes = budget_bytes;
    g_dense_cache.hits = 0;
    g_dense_cache.misses = 0;
    g_dense_cache.admissions = 0;
    g_dense_cache.evictions = 0;
    g_dense_cache.evicted_bytes = 0;
    g_dense_cache.rejected_bytes = 0;
    g_dense_cache.bytes_avoided = 0;
    g_dense_cache.copy_bytes = 0;
    g_dense_cache.copy_ns = 0;
    g_dense_cache.minimum_benefit = 0.75;
    {
        const char *value = getenv("V4_DENSE_CACHE_MIN_BENEFIT");
        if (value && *value) {
            char *end = NULL;
            double parsed = strtod(value, &end);
            if (end != value && parsed >= 0.0 && parsed <= 8.0)
                g_dense_cache.minimum_benefit = parsed;
        }
    }
    pthread_mutex_unlock(&g_dense_cache.mutex);

    if (budget_bytes) {
        fprintf(stderr,
                "v4_dense_cache status=configured budget=%.2fGiB "
                "policy=tensor-greedy-zero-copy min_benefit=%.2f\n",
                budget_bytes / (1024.0 * 1024.0 * 1024.0),
                g_dense_cache.minimum_benefit);
    }
}

uint64_t coli_v4_dense_cache_set_budget(uint64_t budget_bytes) {
    pthread_mutex_lock(&g_dense_cache.mutex);
    if (budget_bytes < g_dense_cache.pinned_bytes)
        budget_bytes = g_dense_cache.pinned_bytes;
    g_dense_cache.budget_bytes = budget_bytes;
    dense_cache_evict_unpinned_locked();
    pthread_mutex_unlock(&g_dense_cache.mutex);
    return budget_bytes;
}

void coli_v4_dense_cache_reset(void) {
    ColiV4DenseCacheStats snapshot = {0};
    pthread_mutex_lock(&g_dense_cache.mutex);
    snapshot.budget_bytes = g_dense_cache.budget_bytes;
    snapshot.resident_bytes = g_dense_cache.resident_bytes;
    snapshot.pinned_bytes = g_dense_cache.pinned_bytes;
    snapshot.entries = g_dense_cache.count;
    snapshot.hits = g_dense_cache.hits;
    snapshot.misses = g_dense_cache.misses;
    snapshot.admissions = g_dense_cache.admissions;
    snapshot.evictions = g_dense_cache.evictions;
    snapshot.evicted_bytes = g_dense_cache.evicted_bytes;
    snapshot.rejected_bytes = g_dense_cache.rejected_bytes;
    snapshot.bytes_avoided = g_dense_cache.bytes_avoided;
    snapshot.copy_bytes = g_dense_cache.copy_bytes;
    snapshot.copy_ns = g_dense_cache.copy_ns;
    dense_cache_clear_locked();
    g_dense_cache.budget_bytes = 0;
    g_dense_cache.copy_bytes = 0;
    g_dense_cache.copy_ns = 0;
    pthread_mutex_unlock(&g_dense_cache.mutex);

    if (snapshot.budget_bytes) {
        double copy_gib = snapshot.copy_bytes / (1024.0 * 1024.0 * 1024.0);
        double copy_ms = snapshot.copy_ns / 1000000.0;
        double copy_gib_s = snapshot.copy_ns
            ? copy_gib / ((double)snapshot.copy_ns / 1000000000.0) : 0.0;
        fprintf(stderr,
                "v4_dense_cache status=done resident=%.2fGiB pinned=%.2fGiB "
                "budget=%.2fGiB entries=%llu hits=%llu misses=%llu "
                "admissions=%llu evictions=%llu evicted=%.2fGiB "
                "bytes_avoided=%.2fGiB rejected=%.2fGiB "
                "copy=%.2fGiB copy_ms=%.3f copy_bw=%.2fGiB/s\n",
                snapshot.resident_bytes / (1024.0 * 1024.0 * 1024.0),
                snapshot.pinned_bytes / (1024.0 * 1024.0 * 1024.0),
                snapshot.budget_bytes / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)snapshot.entries,
                (unsigned long long)snapshot.hits,
                (unsigned long long)snapshot.misses,
                (unsigned long long)snapshot.admissions,
                (unsigned long long)snapshot.evictions,
                snapshot.evicted_bytes / (1024.0 * 1024.0 * 1024.0),
                snapshot.bytes_avoided / (1024.0 * 1024.0 * 1024.0),
                snapshot.rejected_bytes / (1024.0 * 1024.0 * 1024.0),
                copy_gib, copy_ms, copy_gib_s);
    }
}

void coli_v4_dense_cache_stats(ColiV4DenseCacheStats *out) {
    if (!out) return;
    pthread_mutex_lock(&g_dense_cache.mutex);
    *out = (ColiV4DenseCacheStats){
        .budget_bytes = g_dense_cache.budget_bytes,
        .resident_bytes = g_dense_cache.resident_bytes,
        .pinned_bytes = g_dense_cache.pinned_bytes,
        .entries = g_dense_cache.count,
        .hits = g_dense_cache.hits,
        .misses = g_dense_cache.misses,
        .admissions = g_dense_cache.admissions,
        .evictions = g_dense_cache.evictions,
        .evicted_bytes = g_dense_cache.evicted_bytes,
        .rejected_bytes = g_dense_cache.rejected_bytes,
        .bytes_avoided = g_dense_cache.bytes_avoided,
        .copy_bytes = g_dense_cache.copy_bytes,
        .copy_ns = g_dense_cache.copy_ns,
    };
    pthread_mutex_unlock(&g_dense_cache.mutex);
}

/* Return 1 and an immutable borrowed pointer on hit, 0 on miss. A hit pins the
 * entry until coli_v4_layer_payload_free releases that borrow. */
static int dense_cache_borrow(const char *name, uint64_t resident_bytes,
                              void **output) {
    if (!name || !output || resident_bytes > SIZE_MAX) return 0;
    *output = NULL;

    pthread_mutex_lock(&g_dense_cache.mutex);
    DenseCacheEntry *entry = dense_cache_find_locked(name);
    if (!entry || entry->resident_bytes != resident_bytes) {
        g_dense_cache.misses++;
        pthread_mutex_unlock(&g_dense_cache.mutex);
        return 0;
    }
    if (!entry->refs)
        g_dense_cache.pinned_bytes += entry->resident_bytes;
    entry->refs++;
    entry->hits++;
    g_dense_cache.hits++;
    g_dense_cache.bytes_avoided += entry->stored_bytes_avoided;
    *output = entry->data;
    pthread_mutex_unlock(&g_dense_cache.mutex);
    return 1;
}

/* Transfer ownership of an already-materialized immutable execution payload to
 * the cache. The current layer immediately owns the first borrow, so a newly
 * admitted entry starts with refs=1 and stays pinned until that layer releases
 * it. Returns 1 when the cache owns `data`; on 0 the caller retains ownership. */
static int dense_cache_admit_take(const char *name, unsigned char *data,
                                  uint64_t resident_bytes,
                                  uint64_t stored_bytes_avoided) {
    if (!name || !data || !resident_bytes || resident_bytes > SIZE_MAX) return 0;

    double benefit = (double)stored_bytes_avoided / (double)resident_bytes;
    pthread_mutex_lock(&g_dense_cache.mutex);
    DenseCacheEntry *existing = dense_cache_find_locked(name);
    int no_room = g_dense_cache.resident_bytes > g_dense_cache.budget_bytes ||
                  resident_bytes > g_dense_cache.budget_bytes -
                                   g_dense_cache.resident_bytes;
    if (!g_dense_cache.budget_bytes ||
        benefit < g_dense_cache.minimum_benefit || no_room || existing) {
        if (g_dense_cache.budget_bytes &&
            benefit >= g_dense_cache.minimum_benefit && !existing && no_room)
            g_dense_cache.rejected_bytes += resident_bytes;
        pthread_mutex_unlock(&g_dense_cache.mutex);
        return 0;
    }

    if (g_dense_cache.count == g_dense_cache.capacity) {
        size_t next = g_dense_cache.capacity ? g_dense_cache.capacity * 2 : 64;
        if (next < g_dense_cache.capacity ||
            next > SIZE_MAX / sizeof(*g_dense_cache.entries)) {
            g_dense_cache.rejected_bytes += resident_bytes;
            pthread_mutex_unlock(&g_dense_cache.mutex);
            return 0;
        }
        DenseCacheEntry *grown = realloc(
            g_dense_cache.entries, next * sizeof(*g_dense_cache.entries));
        if (!grown) {
            g_dense_cache.rejected_bytes += resident_bytes;
            pthread_mutex_unlock(&g_dense_cache.mutex);
            return 0;
        }
        g_dense_cache.entries = grown;
        g_dense_cache.capacity = next;
    }

    DenseCacheEntry *entry = &g_dense_cache.entries[g_dense_cache.count++];
    memset(entry, 0, sizeof(*entry));
    snprintf(entry->name, sizeof(entry->name), "%s", name);
    entry->data = data;
    entry->resident_bytes = resident_bytes;
    entry->stored_bytes_avoided = stored_bytes_avoided;
    entry->refs = 1;
    g_dense_cache.resident_bytes += resident_bytes;
    g_dense_cache.pinned_bytes += resident_bytes;
    g_dense_cache.admissions++;
    pthread_mutex_unlock(&g_dense_cache.mutex);
    return 1;
}

int coli_v4_coli_layer_load(ColiExecutor *x, ColiDeepSeekV4LayerWeights *w,
                            const ColiDeepSeekV4Config *c, int layer,
                            char *e, size_t n) {
    const int profiling = g_coli_v4_profile_on;
    const uint64_t began = profiling ? coli_v4_profile_now() : 0;
    uint64_t stored_bytes_read = 0;

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
            if (resident_bytes > UINT64_MAX / sizeof(float)) {
                bad(e, n, "COLI static scale size overflow: %s", s->name);
                goto fail;
            }
            resident_bytes *= sizeof(float);
        }
        if (resident_bytes > SIZE_MAX) {
            bad(e, n, "COLI static resident span invalid: %s", s->name);
            goto fail;
        }

        if (dense_cache_borrow(s->name, resident_bytes, (void **)&b)) {
            w->data[i] = b;
            if (s->dtype == COLI_ST_F8_E8M0)
                w->stats.fp8_scale_bytes += resident_bytes;
            w->stats.total_bytes += resident_bytes;
            w->stats.tensor_count++;
            continue;
        }

        b = malloc((size_t)r->stored_bytes);
        if (!b) {
            bad(e, n, "COLI static allocation failed: %s", s->name);
            goto fail;
        }
        if (coli_executor_load_record(x, r, b, (size_t)r->stored_bytes, e, n)) {
            free(b);
            goto fail;
        }
        stored_bytes_read += r->stored_bytes;

        if (s->dtype == COLI_ST_F8_E8M0) {
            size_t k = (size_t)t.data_stored_bytes;
            float *out = malloc(k * sizeof(*out));
            if (!out) {
                free(b);
                goto fail;
            }
            for (size_t q = 0; q < k; q++)
                out[q] = b[t.data_offset + q] == 255
                    ? NAN : ldexpf(1.f, (int)b[t.data_offset + q] - 127);
            free(b);
            b = (unsigned char *)out;
            (void)dense_cache_admit_take(
                s->name, b, resident_bytes, r->stored_bytes);
            w->data[i] = b;
            w->stats.fp8_scale_bytes += resident_bytes;
            w->stats.total_bytes += resident_bytes;
        } else {
            memmove(b, b + t.data_offset, (size_t)t.data_stored_bytes);
            /* Package records include metadata around the execution payload.
             * Shrink before transferring ownership so resident accounting is
             * also an allocated-byte bound. If shrinking fails, keep the
             * caller-owned record and simply skip admission for this tensor. */
            unsigned char *resident = b;
            int exact_allocation = r->stored_bytes == resident_bytes;
            if (!exact_allocation) {
                unsigned char *shrunk = realloc(b, (size_t)resident_bytes);
                if (shrunk) {
                    resident = shrunk;
                    exact_allocation = 1;
                }
            }
            b = resident;
            if (exact_allocation)
                (void)dense_cache_admit_take(
                    s->name, b, resident_bytes, r->stored_bytes);
            w->data[i] = b;
            w->stats.total_bytes += resident_bytes;
        }
        w->stats.tensor_count++;
    }

    if (profiling) {
        coli_v4_profile_add(COLI_V4_PROF_DENSE_READ,
                            coli_v4_profile_now() - began);
        coli_v4_profile_add_bytes(COLI_V4_PROF_DENSE_READ,
                                  stored_bytes_read);
    }
    return 0;

fail:
    if (profiling) {
        coli_v4_profile_add(COLI_V4_PROF_DENSE_READ,
                            coli_v4_profile_now() - began);
        coli_v4_profile_add_bytes(COLI_V4_PROF_DENSE_READ,
                                  stored_bytes_read);
    }
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

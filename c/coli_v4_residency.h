#ifndef COLI_V4_RESIDENCY_H
#define COLI_V4_RESIDENCY_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Cross-module contract for COLI V4 residency.
 *
 * #64 established a safe minimum transient expert pool and tensor-granular
 * deterministic residency. #3 now needs the engine planner to decide which
 * optional resident objects deserve scarce RAM. Keep the ranking primitive
 * here so the top-level planner and store-specific code share one deterministic
 * policy rather than growing separate heuristics.
 */
typedef struct {
    uint64_t budget_bytes;
    uint64_t resident_bytes;
    uint64_t entries;
    uint64_t hits;
    uint64_t misses;
    uint64_t admissions;
    uint64_t rejected_bytes;
    uint64_t bytes_avoided;
    /* Diagnostic only: bytes/time spent materializing caller-owned copies from
     * persistent dense residency. Zero-copy cache hits keep these at zero. */
    uint64_t copy_bytes;
    uint64_t copy_ns;
} ColiV4DenseCacheStats;

typedef enum {
    /* Stable fallback until exposed-wait telemetry is comparable across every
     * candidate class. */
    COLI_V4_RESIDENCY_VALUE_BYTES = 0,
    /* Preferred #3 metric once telemetry is reliable: expected exposed I/O
     * nanoseconds avoided per resident byte. */
    COLI_V4_RESIDENCY_VALUE_EXPOSED_NS = 1,
} ColiV4ResidencyValueMode;

typedef enum {
    COLI_V4_RESIDENCY_DENSE_TENSOR = 0,
    COLI_V4_RESIDENCY_HEAD = 1,
    COLI_V4_RESIDENCY_PERSISTENT_EXPERT = 2,
    COLI_V4_RESIDENCY_OTHER = 3,
} ColiV4ResidencyKind;

typedef struct {
    ColiV4ResidencyKind kind;
    uint32_t id;
    uint64_t resident_bytes;
    uint64_t expected_bytes_avoided;
    uint64_t expected_exposed_ns_avoided;
} ColiV4ResidencyCandidate;

typedef struct {
    uint64_t budget_bytes;
    uint64_t selected_resident_bytes;
    uint64_t expected_bytes_avoided;
    uint64_t expected_exposed_ns_avoided;
    size_t selected_count;
} ColiV4ResidencySelection;

/* Exact comparison of a_num/a_den and b_num/b_den without cross-multiplication
 * overflow. Continued-fraction comparison alternates direction after taking a
 * reciprocal. Returns -1/0/+1. Denominators must be non-zero. */
static inline int coli_v4_residency_ratio_compare(
        uint64_t a_num, uint64_t a_den,
        uint64_t b_num, uint64_t b_den) {
    int direction = 1;
    for (;;) {
        uint64_t aq = a_num / a_den;
        uint64_t bq = b_num / b_den;
        if (aq != bq)
            return aq > bq ? direction : -direction;

        uint64_t ar = a_num % a_den;
        uint64_t br = b_num % b_den;
        if (!ar || !br) {
            if (!ar && !br) return 0;
            return ar ? direction : -direction;
        }

        a_num = a_den;
        a_den = ar;
        b_num = b_den;
        b_den = br;
        direction = -direction;
    }
}

static inline uint64_t coli_v4_residency_candidate_value(
        const ColiV4ResidencyCandidate *candidate,
        ColiV4ResidencyValueMode mode) {
    return mode == COLI_V4_RESIDENCY_VALUE_EXPOSED_NS
        ? candidate->expected_exposed_ns_avoided
        : candidate->expected_bytes_avoided;
}

/* Deterministic greedy optional-residency allocator.
 *
 * Mandatory runtime/KV/scratch and the minimum transient expert pool are
 * reserved by the caller before passing `budget_bytes`; this function allocates
 * only the optional remainder. At each step it chooses the highest value per
 * resident byte among candidates that still fit. Zero-value objects are never
 * admitted. Equal ratios break by kind, then id, then input position so the
 * same inputs always produce the same plan on every backend.
 *
 * `selected` is a caller-owned `count`-byte bitmap. The primitive deliberately
 * does not allocate memory and has no backend assumptions, making it suitable
 * for Apple UMA and explicit CUDA host/VRAM candidate lists alike.
 */
static inline int coli_v4_residency_select(
        const ColiV4ResidencyCandidate *candidates, size_t count,
        uint64_t budget_bytes, ColiV4ResidencyValueMode mode,
        unsigned char *selected, ColiV4ResidencySelection *out) {
    if ((!candidates && count) || (!selected && count) || !out ||
        (mode != COLI_V4_RESIDENCY_VALUE_BYTES &&
         mode != COLI_V4_RESIDENCY_VALUE_EXPOSED_NS))
        return -1;

    if (count) memset(selected, 0, count);
    memset(out, 0, sizeof(*out));
    out->budget_bytes = budget_bytes;

    uint64_t remaining = budget_bytes;
    for (;;) {
        size_t best = count;
        uint64_t best_value = 0;

        for (size_t index = 0; index < count; index++) {
            const ColiV4ResidencyCandidate *candidate = &candidates[index];
            if (selected[index] || !candidate->resident_bytes ||
                candidate->resident_bytes > remaining)
                continue;

            uint64_t value = coli_v4_residency_candidate_value(candidate, mode);
            if (!value) continue;

            int better = 0;
            if (best == count) {
                better = 1;
            } else {
                const ColiV4ResidencyCandidate *incumbent = &candidates[best];
                int ratio = coli_v4_residency_ratio_compare(
                    value, candidate->resident_bytes,
                    best_value, incumbent->resident_bytes);
                if (ratio > 0) {
                    better = 1;
                } else if (ratio == 0) {
                    if (candidate->kind < incumbent->kind ||
                        (candidate->kind == incumbent->kind &&
                         (candidate->id < incumbent->id ||
                          (candidate->id == incumbent->id && index < best))))
                        better = 1;
                }
            }
            if (better) {
                best = index;
                best_value = value;
            }
        }

        if (best == count) break;
        const ColiV4ResidencyCandidate *chosen = &candidates[best];
        selected[best] = 1;
        remaining -= chosen->resident_bytes;
        out->selected_resident_bytes += chosen->resident_bytes;
        out->expected_bytes_avoided += chosen->expected_bytes_avoided;
        out->expected_exposed_ns_avoided += chosen->expected_exposed_ns_avoided;
        out->selected_count++;
    }
    return 0;
}

/* Reconfigure the process-local COLI dense/static cache. This is called during
 * V4 engine setup before inference starts. Reconfiguration drops any previous
 * cache generation; normal single-engine use therefore has deterministic
 * ownership and teardown. */
void coli_v4_dense_cache_configure(uint64_t budget_bytes);

/* Release all persistent dense/static cache allocations. */
void coli_v4_dense_cache_reset(void);

/* Snapshot counters for diagnostics/tests. */
void coli_v4_dense_cache_stats(ColiV4DenseCacheStats *out);

#ifdef __cplusplus
}
#endif

#endif /* COLI_V4_RESIDENCY_H */

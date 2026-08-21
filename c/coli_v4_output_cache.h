#ifndef COLIBRI_V4_OUTPUT_CACHE_H
#define COLIBRI_V4_OUTPUT_CACHE_H

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4.h"

typedef struct {
    int *prompt_ids;
    int prompt_count;
    int *tokens;
    float *logits;
    int *positions;
    int *ordinals;
    int generated_count;
    uint64_t read_bytes;
    double restore_sec;
} ColiV4OutputCacheHit;

typedef struct {
    uint64_t lookups;
    uint64_t hits;
    uint64_t stores;
    uint64_t bypasses;
    uint64_t corruptions;
    uint64_t identity_rejects;
    uint64_t read_bytes;
    uint64_t write_bytes;
} ColiV4OutputCacheStats;

/* Stage-3 v1 is deliberately narrow: only target-only greedy generation
 * (options->no_dspark != 0) is eligible. Cache failures are optimization misses,
 * never generation failures. */
int coli_v4_output_cache_lookup(ColiV4Session *session,
                                const char *prompt, size_t prompt_length,
                                const ColiV4SessionGenerateOptions *options,
                                ColiV4OutputCacheHit *hit);
int coli_v4_output_cache_store(ColiV4Session *session,
                               const ColiV4SessionGenerateOptions *options,
                               const int *tokens, const float *logits,
                               const int *positions, const int *ordinals,
                               int generated_count);
void coli_v4_output_cache_hit_free(ColiV4OutputCacheHit *hit);
void coli_v4_output_cache_stats(ColiV4OutputCacheStats *stats);

#ifdef COLI_V4_OUTPUT_CACHE_TESTING
/* Tiny V4 fixtures have no package. Acceptance builds may install an explicit
 * synthetic identity; production builds do not expose this seam. */
void coli_v4_output_cache_test_identity(
    ColiV4Engine *engine, const uint8_t model_fingerprint[32],
    const uint8_t tokenizer_template_fingerprint[32]);
#endif

#endif /* COLIBRI_V4_OUTPUT_CACHE_H */

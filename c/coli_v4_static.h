#ifndef COLI_V4_STATIC_H
#define COLI_V4_STATIC_H
#include "deepseek_v4_internal.h"
#include "coli_executor.h"

typedef struct {
    uint64_t budget_bytes;
    uint64_t resident_bytes;
    uint64_t hits;
    uint64_t misses;
    uint64_t inserts;
    uint64_t evictions;
    uint64_t stored_bytes_avoided;
} ColiV4DenseCacheStats;

/* Configure the process-local immutable tensor cache. The cache stores decoded
 * execution payloads and, for this tranche, returns caller-owned copies so the
 * existing layer free contract remains unchanged. Admission is scan-resistant:
 * equal benefit/byte tensors do not evict one another during deterministic
 * layer walks. A zero budget disables and empties the cache. */
void coli_v4_dense_cache_configure(uint64_t budget_bytes);
void coli_v4_dense_cache_stats(ColiV4DenseCacheStats *stats);
void coli_v4_dense_cache_shutdown(void);

int coli_v4_coli_layer_load(ColiExecutor *executor, ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config, int layer,
                            char *error, size_t error_size);
int coli_v4_coli_layer_bytes(ColiExecutor *executor, const ColiDeepSeekV4Config *config,
                             int layer, uint64_t *bytes, char *error, size_t error_size);
/* Load a cold non-layer BF16/F32 tensor directly from its typed COLI record.
 * This is deliberately a loader helper rather than a generic runtime tensor
 * abstraction: the caller owns the converted float buffer. */
int coli_v4_coli_tensor_load_f32(ColiExecutor *executor, ColiFloatTensor *output,
                                 const char *name, char *error, size_t error_size);
#endif

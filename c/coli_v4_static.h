#ifndef COLI_V4_STATIC_H
#define COLI_V4_STATIC_H
#include "deepseek_v4_internal.h"
#include "coli_executor.h"
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

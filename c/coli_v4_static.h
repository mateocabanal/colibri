#ifndef COLI_V4_STATIC_H
#define COLI_V4_STATIC_H
#include "deepseek_v4_internal.h"
#include "coli_executor.h"
int coli_v4_coli_layer_load(ColiExecutor *executor, ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config, int layer,
                            char *error, size_t error_size);
#endif

#ifndef COLI_V4_COLI_EXPERT_STORE_H
#define COLI_V4_COLI_EXPERT_STORE_H

#include "expert_store.h"

typedef struct {
    const char *package_dir;
    const char *required_profile;
    int layers;
    int experts_per_layer;
    uint64_t cache_bytes;
} ColiV4ColiExpertStoreOptions;

/* Creates a bounded V4 routed-expert cache backed by target-compiled COLI
 * records. Every miss copies one complete compiled expert into a caller-ready
 * resident slot; no safetensors lookup or per-matrix repacking occurs. */
int coli_v4_coli_expert_store_open(const ColiV4ColiExpertStoreOptions *options,
                                   ColiExpertStore **output,
                                   char *error, size_t error_size);

#endif /* COLI_V4_COLI_EXPERT_STORE_H */

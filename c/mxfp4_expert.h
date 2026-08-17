#ifndef COLIBRI_MXFP4_EXPERT_H
#define COLIBRI_MXFP4_EXPERT_H

#include <stddef.h>
#include <stdint.h>

#include "coli_format.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    COLI_MXFP4_EXPERT_ROLE_GATE = 1,
    COLI_MXFP4_EXPERT_ROLE_UP = 2,
    COLI_MXFP4_EXPERT_ROLE_DOWN = 3,
};

typedef struct ColiMxfp4ExpertLayout {
    size_t gate_weight_bytes;
    size_t gate_scale_bytes;
    size_t up_weight_bytes;
    size_t up_scale_bytes;
    size_t down_weight_bytes;
    size_t down_scale_bytes;
    size_t resident_bytes;
} ColiMxfp4ExpertLayout;

typedef struct ColiMxfp4ExpertBuffers {
    uint8_t *gate_weights;
    size_t gate_weight_capacity;
    uint8_t *gate_scales;
    size_t gate_scale_capacity;

    uint8_t *up_weights;
    size_t up_weight_capacity;
    uint8_t *up_scales;
    size_t up_scale_capacity;

    uint8_t *down_weights;
    size_t down_weight_capacity;
    uint8_t *down_scales;
    size_t down_scale_capacity;
} ColiMxfp4ExpertBuffers;

/*
 * Validate the compiler/runtime ABI for one routed SwiGLU expert and return
 * the exact resident byte requirements. This is deliberately independent of
 * Qwen's cache implementation so the same validator can be reused by other
 * MoE frontends that emit the same gate/up/down contract.
 *
 * gate/up: [intermediate, hidden]
 * down:    [hidden, intermediate]
 * weights: row-major packed E2M1, two columns per byte
 * scales:  raw UE8M0, one scale per 1x32 weight block
 */
int coli_mxfp4_expert_validate_info(const ColiExpertInfo *info,
                                    int hidden, int intermediate,
                                    ColiMxfp4ExpertLayout *layout,
                                    char *error, size_t error_size);

/*
 * Parse + validate RECORD then read only its six executable MXFP4 spans into
 * caller-owned cache storage. The COLIEXPT framing and padding never become
 * resident, and this function performs no heap allocation.
 */
int coli_mxfp4_expert_load(const ColiPackage *package,
                           const ColiRecordInfo *record,
                           int hidden, int intermediate,
                           const ColiMxfp4ExpertBuffers *buffers,
                           ColiMxfp4ExpertLayout *layout,
                           char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_MXFP4_EXPERT_H */

#ifndef COLIBRI_MXFP4_EXPERT_H
#define COLIBRI_MXFP4_EXPERT_H

#include <stddef.h>
#include <stdint.h>

#include "coli_format.h"
/* qwen_moe includes this expert contract immediately after coli_executor.h.
 * The wrapper supplies the frozen Apple8 host descriptor to its legacy open
 * call without weakening coli_executor_open() for any other runtime. */
#include "qwen_coli_compat.h"

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

    /* Source package execution layout. The caller-owned buffers are ALWAYS
     * canonical row-major MXFP4 weights + separate UE8M0 scales. A canonical
     * source can be read directly; an Apple8 source is synchronously decoded
     * (including rANS when present) and detiled once on cache load. */
    uint16_t source_layout;

    /* Canonical source only: smallest record-relative span covering all six
     * executable regions. The *_span_offset fields are relative to
     * record_span_offset. Apple8 sources set record_span_bytes=0 because their
     * combined tile payloads must go through the decode+detile path. */
    uint64_t record_span_offset;
    size_t record_span_bytes;
    size_t gate_weight_span_offset;
    size_t gate_scale_span_offset;
    size_t up_weight_span_offset;
    size_t up_scale_span_offset;
    size_t down_weight_span_offset;
    size_t down_scale_span_offset;
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
 * the exact CANONICAL resident byte requirements. Accepted package layouts:
 *
 *   COLI_CSF_LAYOUT_CANONICAL
 *     separate row-major packed E2M1 weights + raw UE8M0 scales, codec none.
 *
 *   COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1
 *     production Apple8 Design-A combined 8x32 tiles. The combined matrix may
 *     be raw or RANS256_G0_NIBBLE; storage decoding does not change the target
 *     execution bytes. load_ex/load then detile those bytes into the same
 *     canonical caller buffers used by the existing Qwen fmt=7 kernels.
 *
 * gate/up: [intermediate, hidden]
 * down:    [hidden, intermediate]
 */
int coli_mxfp4_expert_validate_info(const ColiExpertInfo *info,
                                    int hidden, int intermediate,
                                    ColiMxfp4ExpertLayout *layout,
                                    char *error, size_t error_size);

/* Pure layout conversion used by the Apple8 load path and its contract tests.
 * TILES must be the already-decoded production Apple8 combined payload for one
 * matrix. Output is canonical row-major packed E2M1 + separate UE8M0 scales. */
int coli_mxfp4_apple8_detile_matrix(const uint8_t *tiles, size_t tile_bytes,
                                    uint64_t rows, uint64_t columns,
                                    uint8_t *weights, size_t weight_capacity,
                                    uint8_t *scales, size_t scale_capacity,
                                    char *error, size_t error_size);

/*
 * Parse + validate RECORD and populate canonical caller-owned cache storage.
 * Canonical packages use direct range reads. Apple8 packages use the CSF
 * synchronous expert decoder first (stored CRC -> strict rANS/table checks ->
 * decoded CRC/padding), then detile once. No MetalIO composition is attempted
 * for Apple8 in this path.
 */
int coli_mxfp4_expert_load_ex(const ColiPackage *package,
                              const ColiRecordInfo *record,
                              int hidden, int intermediate,
                              const ColiMxfp4ExpertBuffers *buffers,
                              ColiMxfp4ExpertLayout *layout,
                              uint32_t read_flags,
                              char *error, size_t error_size);
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

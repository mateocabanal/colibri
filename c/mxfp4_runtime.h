#ifndef COLIBRI_MXFP4_RUNTIME_H
#define COLIBRI_MXFP4_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "generated/coli_target_registry.h"
#include "coli_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Canonical MXFP4 uses a real UE8M0 scale span. Apple8 Design A embeds its
 * eight UE8M0 bytes inside each 136-byte tile, so streamed runtimes use a tiny
 * sidecar in the legacy `scales` cache slot to carry execution-layout metadata
 * without repacking execution bytes. The sidecar is runtime metadata only. */
#define COLI_MXFP4_RUNTIME_TAG_MAGIC UINT64_C(0x315447543450584d)
typedef struct ColiMxfp4RuntimeTag {
    uint64_t magic;
    uint16_t layout;
    uint16_t version;
    uint32_t reserved;
} ColiMxfp4RuntimeTag;

enum { COLI_MXFP4_RUNTIME_TAG_VERSION = 1 };

void coli_mxfp4_runtime_tag_init(void *storage, uint16_t layout);
uint16_t coli_mxfp4_runtime_layout(const uint8_t *tag_storage);
int coli_mxfp4_runtime_is_apple8(const uint8_t *tag_storage);

/* Explicit execution-layout API. */
int coli_mxfp4_matmul_layout(float *y, const float *x,
                             const uint8_t *weights, const uint8_t *scales_or_tag,
                             uint16_t layout,
                             int rows, int input_columns, int output_rows);

/* Historical ABI remains canonical-only. No metadata probing is performed. */
void coli_mxfp4_matmul(float *y, const float *x,
                       const uint8_t *weights, const uint8_t *scales,
                       int rows, int input_columns, int output_rows);

/* Qwen composition entry: tag_storage must point at either the Apple8 runtime
 * tag or a canonical scale array known by the caller to contain at least one
 * ColiMxfp4RuntimeTag-sized readable prefix. */
void coli_mxfp4_matmul_auto(float *y, const float *x,
                            const uint8_t *weights, const uint8_t *scales_or_tag,
                            int rows, int input_columns, int output_rows);

int coli_mxfp4_swiglu_expert_layout(float *output, const float *x,
                                    const uint8_t *gate, const uint8_t *gate_scales_or_tag,
                                    const uint8_t *up, const uint8_t *up_scales_or_tag,
                                    const uint8_t *down, const uint8_t *down_scales_or_tag,
                                    uint16_t layout,
                                    int rows, int hidden, int intermediate,
                                    float route_weight,
                                    float *scratch_gate, float *scratch_up,
                                    float *scratch_h, float *scratch_y);

/* Historical ABI remains canonical-only. */
void coli_mxfp4_swiglu_expert(float *output, const float *x,
                              const uint8_t *gate, const uint8_t *gate_scales,
                              const uint8_t *up, const uint8_t *up_scales,
                              const uint8_t *down, const uint8_t *down_scales,
                              int rows, int hidden, int intermediate,
                              float route_weight,
                              float *scratch_gate, float *scratch_up,
                              float *scratch_h, float *scratch_y);

/* Qwen layout-auto composition entry. */
void coli_mxfp4_swiglu_expert_auto(float *output, const float *x,
                                   const uint8_t *gate, const uint8_t *gate_scales_or_tag,
                                   const uint8_t *up, const uint8_t *up_scales_or_tag,
                                   const uint8_t *down, const uint8_t *down_scales_or_tag,
                                   int rows, int hidden, int intermediate,
                                   float route_weight,
                                   float *scratch_gate, float *scratch_up,
                                   float *scratch_h, float *scratch_y);

#ifdef __cplusplus
}
#endif

#endif

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
 * eight UE8M0 bytes inside each 136-byte tile, so streamed runtimes need a
 * tiny sidecar in the legacy `scales` slot to carry execution-layout metadata
 * without repacking the execution bytes. The sidecar is runtime metadata only:
 * gate/up/down weight buffers remain the exact decoded Apple8 tile bytes. */
#define COLI_MXFP4_RUNTIME_TAG_MAGIC UINT64_C(0x315447543450584d) /* "MXP4TGT1" LE */
typedef struct ColiMxfp4RuntimeTag {
    uint64_t magic;
    uint16_t layout;
    uint16_t version;
    uint32_t reserved;
} ColiMxfp4RuntimeTag;

enum { COLI_MXFP4_RUNTIME_TAG_VERSION = 1 };

void coli_mxfp4_runtime_tag_init(void *storage, uint16_t layout);
uint16_t coli_mxfp4_runtime_layout(const uint8_t *scales_or_tag);
int coli_mxfp4_runtime_is_apple8(const uint8_t *scales_or_tag);

/*
 * Runtime contract shared with colic's MXFP4 lowering.
 *
 * canonical:
 *   weights: row-major E2M1, two nibbles/byte, low nibble first
 *   scales:  row-major raw E8M0, one byte per 32 input columns
 *
 * Apple8 0x0103:
 *   weights: exact 8x32/136-byte target tiles (128 packed E2M1 + 8 UE8M0)
 *   scales:  ColiMxfp4RuntimeTag with layout 0x0103
 */
int coli_mxfp4_matmul_layout(float *y, const float *x,
                             const uint8_t *weights, const uint8_t *scales_or_tag,
                             uint16_t layout,
                             int rows, int input_columns, int output_rows);

/* Backward-compatible dispatch. Canonical scale arrays remain unchanged;
 * Apple8 callers pass the explicit runtime tag created by the shared loader. */
void coli_mxfp4_matmul(float *y, const float *x,
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

/* One SwiGLU routed expert for decode/small batches. This compatibility entry
 * infers Apple8 from the runtime sidecar and otherwise uses canonical MXFP4. */
void coli_mxfp4_swiglu_expert(float *output, const float *x,
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

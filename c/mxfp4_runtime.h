#ifndef COLIBRI_MXFP4_RUNTIME_H
#define COLIBRI_MXFP4_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Runtime contract shared with colic's MXFP4 lowering:
 *   weights: row-major E2M1, two nibbles/byte, low nibble first
 *   scales:  row-major raw E8M0, one byte per 32 input columns
 *
 * These wrappers intentionally expose no model-specific storage names. Qwen's
 * streamed expert loader can point directly at compiler-emitted matrix spans.
 */
void coli_mxfp4_matmul(float *y, const float *x,
                       const uint8_t *weights, const uint8_t *scales,
                       int rows, int input_columns, int output_rows);

/*
 * One SwiGLU routed expert for decode/small batches. Scratch is caller-owned so
 * the engine can reuse it across streamed experts instead of allocating on the
 * hot path.
 *
 * gate/up: [I,D], down: [D,I]
 * x:       [rows,D]
 * output:  [rows,D], accumulated with route_weight
 * scratch_gate/up/h: rows*I floats; scratch_y: rows*D floats.
 */
void coli_mxfp4_swiglu_expert(float *output, const float *x,
                              const uint8_t *gate, const uint8_t *gate_scales,
                              const uint8_t *up, const uint8_t *up_scales,
                              const uint8_t *down, const uint8_t *down_scales,
                              int rows, int hidden, int intermediate,
                              float route_weight,
                              float *scratch_gate, float *scratch_up,
                              float *scratch_h, float *scratch_y);

#ifdef __cplusplus
}
#endif

#endif

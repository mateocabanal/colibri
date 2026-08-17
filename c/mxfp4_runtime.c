#include "mxfp4_runtime.h"

#include <math.h>

#include "quant.h"

static inline float coli_mxfp4_silu(float x) {
    return x / (1.0f + expf(-x));
}

void coli_mxfp4_matmul(float *y, const float *x,
                       const uint8_t *weights, const uint8_t *scales,
                       int rows, int input_columns, int output_rows) {
    matmul_mxfp4(y, x, weights, scales, rows, input_columns, output_rows);
}

void coli_mxfp4_swiglu_expert(float *output, const float *x,
                              const uint8_t *gate, const uint8_t *gate_scales,
                              const uint8_t *up, const uint8_t *up_scales,
                              const uint8_t *down, const uint8_t *down_scales,
                              int rows, int hidden, int intermediate,
                              float route_weight,
                              float *scratch_gate, float *scratch_up,
                              float *scratch_h, float *scratch_y) {
    matmul_mxfp4(scratch_gate, x, gate, gate_scales,
                 rows, hidden, intermediate);
    matmul_mxfp4(scratch_up, x, up, up_scales,
                 rows, hidden, intermediate);
    for (int row = 0; row < rows; ++row) {
        const int64_t base = (int64_t)row * intermediate;
        for (int i = 0; i < intermediate; ++i) {
            const int64_t at = base + i;
            scratch_h[at] = coli_mxfp4_silu(scratch_gate[at]) * scratch_up[at];
        }
    }
    matmul_mxfp4(scratch_y, scratch_h, down, down_scales,
                 rows, intermediate, hidden);
    for (int row = 0; row < rows; ++row) {
        const int64_t base = (int64_t)row * hidden;
        for (int d = 0; d < hidden; ++d) {
            const int64_t at = base + d;
            output[at] += route_weight * scratch_y[at];
        }
    }
}

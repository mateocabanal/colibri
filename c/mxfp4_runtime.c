#include "mxfp4_runtime.h"

#include <math.h>
#include <string.h>

#include "quant.h"

static inline float coli_mxfp4_silu(float x) {
    return x / (1.0f + expf(-x));
}

void coli_mxfp4_runtime_tag_init(void *storage, uint16_t layout) {
    ColiMxfp4RuntimeTag tag;
    if (!storage) return;
    memset(&tag, 0, sizeof(tag));
    tag.magic = COLI_MXFP4_RUNTIME_TAG_MAGIC;
    tag.layout = layout;
    tag.version = COLI_MXFP4_RUNTIME_TAG_VERSION;
    memcpy(storage, &tag, sizeof(tag));
}

uint16_t coli_mxfp4_runtime_layout(const uint8_t *scales_or_tag) {
    ColiMxfp4RuntimeTag tag;
    if (!scales_or_tag) return COLI_CSF_LAYOUT_CANONICAL;
    memcpy(&tag, scales_or_tag, sizeof(tag));
    if (tag.magic == COLI_MXFP4_RUNTIME_TAG_MAGIC &&
        tag.version == COLI_MXFP4_RUNTIME_TAG_VERSION &&
        tag.reserved == 0 &&
        tag.layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1)
        return tag.layout;
    return COLI_CSF_LAYOUT_CANONICAL;
}

int coli_mxfp4_runtime_is_apple8(const uint8_t *scales_or_tag) {
    return coli_mxfp4_runtime_layout(scales_or_tag) ==
           COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1;
}

/* Exact Apple8 execution-layout reader. Preserve the scalar canonical kernel's
 * accumulation structure: one ga over each logical 32-column group, multiply
 * that group by UE8M0 once, then add it to the row accumulator. */
static void coli_mxfp4_matmul_apple8(float *y, const float *x,
                                     const uint8_t *tiles,
                                     int rows, int input_columns,
                                     int output_rows) {
    const int groups = (input_columns + 31) / 32;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < output_rows; ++o) {
        const int output_tile = o / 8;
        const int tile_row = o & 7;
        for (int s = 0; s < rows; ++s) {
            const float *xs = x + (int64_t)s * input_columns;
            float acc = 0.0f;
            for (int g = 0; g < groups; ++g) {
                const uint8_t *tile = tiles +
                    ((int64_t)output_tile * groups + g) *
                    COLI_APPLE8_MXFP4_TILE_BYTES;
                const uint8_t *packed = tile + tile_row *
                    COLI_APPLE8_MXFP4_WEIGHT_ROW_BYTES;
                const float scale = mx4_scale(tile[
                    COLI_APPLE8_MXFP4_WEIGHT_BYTES + tile_row]);
                const int base = g * 32;
                int glen = input_columns - base;
                float ga = 0.0f;
                if (glen > 32) glen = 32;
                for (int i = 0; i < glen; i += 2) {
                    const uint8_t byte = packed[i >> 1];
                    ga += xs[base + i] * mx4_lut[byte & 0x0f];
                    if (i + 1 < glen)
                        ga += xs[base + i + 1] * mx4_lut[byte >> 4];
                }
                acc += ga * scale;
            }
            y[(int64_t)s * output_rows + o] = acc;
        }
    }
}

int coli_mxfp4_matmul_layout(float *y, const float *x,
                             const uint8_t *weights,
                             const uint8_t *scales_or_tag,
                             uint16_t layout,
                             int rows, int input_columns,
                             int output_rows) {
    if (!y || !x || !weights || rows <= 0 || input_columns <= 0 || output_rows <= 0)
        return -1;
    if (layout == COLI_CSF_LAYOUT_CANONICAL) {
        if (!scales_or_tag) return -1;
        matmul_mxfp4(y, x, weights, scales_or_tag,
                     rows, input_columns, output_rows);
        return 0;
    }
    if (layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1) {
        coli_mxfp4_matmul_apple8(y, x, weights,
                                 rows, input_columns, output_rows);
        return 0;
    }
    return -1;
}

void coli_mxfp4_matmul(float *y, const float *x,
                       const uint8_t *weights, const uint8_t *scales_or_tag,
                       int rows, int input_columns, int output_rows) {
    const uint16_t layout = coli_mxfp4_runtime_layout(scales_or_tag);
    if (coli_mxfp4_matmul_layout(y, x, weights, scales_or_tag, layout,
                                 rows, input_columns, output_rows) != 0) {
        /* Historical API is void. Fail closed to deterministic zeroes instead
         * of accidentally interpreting a target layout as canonical bytes. */
        if (y && rows > 0 && output_rows > 0)
            memset(y, 0, (size_t)rows * (size_t)output_rows * sizeof(float));
    }
}

int coli_mxfp4_swiglu_expert_layout(float *output, const float *x,
                                    const uint8_t *gate,
                                    const uint8_t *gate_scales_or_tag,
                                    const uint8_t *up,
                                    const uint8_t *up_scales_or_tag,
                                    const uint8_t *down,
                                    const uint8_t *down_scales_or_tag,
                                    uint16_t layout,
                                    int rows, int hidden, int intermediate,
                                    float route_weight,
                                    float *scratch_gate, float *scratch_up,
                                    float *scratch_h, float *scratch_y) {
    if (!output || !x || !gate || !up || !down || !scratch_gate || !scratch_up ||
        !scratch_h || !scratch_y || rows <= 0 || hidden <= 0 || intermediate <= 0)
        return -1;
    if (coli_mxfp4_matmul_layout(scratch_gate, x, gate, gate_scales_or_tag,
                                 layout, rows, hidden, intermediate) != 0 ||
        coli_mxfp4_matmul_layout(scratch_up, x, up, up_scales_or_tag,
                                 layout, rows, hidden, intermediate) != 0)
        return -1;
    for (int row = 0; row < rows; ++row) {
        const int64_t base = (int64_t)row * intermediate;
        for (int i = 0; i < intermediate; ++i) {
            const int64_t at = base + i;
            scratch_h[at] = coli_mxfp4_silu(scratch_gate[at]) * scratch_up[at];
        }
    }
    if (coli_mxfp4_matmul_layout(scratch_y, scratch_h, down, down_scales_or_tag,
                                 layout, rows, intermediate, hidden) != 0)
        return -1;
    for (int row = 0; row < rows; ++row) {
        const int64_t base = (int64_t)row * hidden;
        for (int d = 0; d < hidden; ++d) {
            const int64_t at = base + d;
            output[at] += route_weight * scratch_y[at];
        }
    }
    return 0;
}

void coli_mxfp4_swiglu_expert(float *output, const float *x,
                              const uint8_t *gate,
                              const uint8_t *gate_scales_or_tag,
                              const uint8_t *up,
                              const uint8_t *up_scales_or_tag,
                              const uint8_t *down,
                              const uint8_t *down_scales_or_tag,
                              int rows, int hidden, int intermediate,
                              float route_weight,
                              float *scratch_gate, float *scratch_up,
                              float *scratch_h, float *scratch_y) {
    const uint16_t gate_layout = coli_mxfp4_runtime_layout(gate_scales_or_tag);
    const uint16_t up_layout = coli_mxfp4_runtime_layout(up_scales_or_tag);
    const uint16_t down_layout = coli_mxfp4_runtime_layout(down_scales_or_tag);
    if (gate_layout != up_layout || gate_layout != down_layout ||
        coli_mxfp4_swiglu_expert_layout(output, x,
                                        gate, gate_scales_or_tag,
                                        up, up_scales_or_tag,
                                        down, down_scales_or_tag,
                                        gate_layout,
                                        rows, hidden, intermediate,
                                        route_weight,
                                        scratch_gate, scratch_up,
                                        scratch_h, scratch_y) != 0) {
        /* No silent reinterpretation on malformed/mixed layouts. */
        return;
    }
}

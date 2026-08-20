/* Experimental milestone-1 executor overlay.
 *
 * The authoritative .coli load is unchanged. After a successful routed-expert
 * record read, V4_METAL_TILE=1 prepares lossless Apple8 tile copies for the
 * three MXFP4 matrices. Failure to prepare is advisory: canonical row bytes
 * remain intact and the Metal wrapper falls back to the stock fmt7 path.
 */
#define coli_executor_load_expert coli_executor_load_expert_row
#include "coli_executor.c"
#undef coli_executor_load_expert

#include "backend_metal_tile.h"

#include <stdatomic.h>
#include <stdint.h>

static atomic_uint_fast64_t g_tile_load_generation = 1;

static int tile_matrix_eligible(const ColiExpertMatrixInfo *m,
                                size_t resident_bytes) {
    if (!m || m->math_format != COLI_CSF_MATH_MXFP4_E2M1 ||
        m->scale_format != COLI_CSF_SCALE_UE8M0 ||
        m->layout != COLI_CSF_LAYOUT_CANONICAL ||
        m->weight_codec != COLI_CSF_CODEC_NONE ||
        m->scale_codec != COLI_CSF_CODEC_NONE ||
        !m->rows || !m->columns ||
        m->rows > INT32_MAX || m->columns > INT32_MAX)
        return 0;

    uint64_t row_bytes = (m->columns + 1u) / 2u;
    uint64_t groups = (m->columns + 31u) / 32u;
    if (m->rows > UINT64_MAX / row_bytes ||
        m->rows > UINT64_MAX / groups)
        return 0;
    uint64_t weight_need = m->rows * row_bytes;
    uint64_t scale_need = m->rows * groups;
    if (weight_need > m->weight_stored_bytes ||
        scale_need > m->scale_stored_bytes ||
        m->weight_offset > resident_bytes ||
        weight_need > resident_bytes - m->weight_offset ||
        m->scale_offset > resident_bytes ||
        scale_need > resident_bytes - m->scale_offset)
        return 0;
    return 1;
}

int coli_executor_load_expert(const ColiExecutor *executor,
                              int32_t layer, int32_t expert,
                              void *resident_slot, size_t resident_bytes,
                              char *error, size_t error_size) {
    int result = coli_executor_load_expert_row(
        executor, layer, expert, resident_slot, resident_bytes,
        error, error_size);
    if (result || !coli_metal_tile_enabled()) return result;

    ColiExpertInfo info;
    char ignored[128] = {0};
    if (coli_executor_expert_info(
            executor, layer, expert, &info, ignored, sizeof(ignored)) != 0)
        return result;

    uint64_t generation = atomic_fetch_add_explicit(
        &g_tile_load_generation, UINT64_C(1), memory_order_relaxed);
    if (!generation)
        generation = atomic_fetch_add_explicit(
            &g_tile_load_generation, UINT64_C(1), memory_order_relaxed);

    const uint8_t *base = (const uint8_t *)resident_slot;
    for (int i = 0; i < 3; ++i) {
        const ColiExpertMatrixInfo *m = &info.matrices[i];
        if (!tile_matrix_eligible(m, resident_bytes)) continue;
        (void)coli_metal_tile_prepare_matrix(
            base + m->weight_offset,
            base + m->scale_offset,
            (int)m->rows, (int)m->columns,
            generation);
    }
    return result;
}

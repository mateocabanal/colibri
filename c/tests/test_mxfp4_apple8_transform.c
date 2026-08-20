#include "../mxfp4_apple8_tile.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_bytes(uint8_t *p, size_t n, uint32_t seed) {
    uint32_t x = seed;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = (uint8_t)(x >> 24);
    }
}

static int make_matrix(ColiMxfp4Apple8RowMatrix *m, uint32_t role,
                       uint64_t rows, uint64_t columns,
                       uint8_t **weights_out, uint8_t **scales_out) {
    size_t wb = (size_t)rows * (size_t)((columns + 1u) / 2u);
    size_t sb = (size_t)rows * (size_t)((columns + 31u) / 32u);
    uint8_t *w = (uint8_t *)malloc(wb);
    uint8_t *s = (uint8_t *)malloc(sb);
    if (!w || !s) { free(w); free(s); return -1; }
    fill_bytes(w, wb, 0x13100000u ^ role ^ (uint32_t)rows);
    fill_bytes(s, sb, 0x8a000000u ^ role ^ (uint32_t)columns);
    for (size_t i = 0; i < sb; ++i) s[i] = (uint8_t)(110u + (s[i] % 60u));
    *m = (ColiMxfp4Apple8RowMatrix){
        .role = role,
        .rows = rows,
        .columns = columns,
        .weights = w,
        .weight_bytes = wb,
        .scales = s,
        .scale_bytes = sb,
    };
    *weights_out = w;
    *scales_out = s;
    return 0;
}

int main(void) {
    ColiMxfp4Apple8SourceExpert source = {.matrix_count = 3};
    uint8_t *weights[3] = {0}, *scales[3] = {0};
    if (make_matrix(&source.matrices[0], 1, 8, 32, &weights[0], &scales[0]) ||
        make_matrix(&source.matrices[1], 2, 9, 37, &weights[1], &scales[1]) ||
        make_matrix(&source.matrices[2], 3, 16, 64, &weights[2], &scales[2]))
        return 1;

    ColiRepresentationId src_rep, dst_rep;
    coli_mxfp4_apple8_source_fixture_representation(&src_rep);
    coli_mxfp4_apple8_target_fixture_representation(&dst_rep);
    if (!coli_representation_exact_math_compatible(&src_rep, &dst_rep) ||
        coli_representation_equal(&src_rep, &dst_rep))
        return 2;

    ColiRepresentationTransformRegistry registry;
    coli_jit_transform_registry_init(&registry);
    if (coli_mxfp4_apple8_register_transform(&registry) != 0) return 3;
    ColiRepresentationTransformOps ops;
    if (coli_jit_transform_registry_find(
            &registry, &src_rep, &dst_rep,
            COLI_MXFP4_APPLE8_FIXTURE_TRANSFORM_ABI, &ops) != 0)
        return 4;

    uint64_t canonical_bytes = 0;
    for (int i = 0; i < 3; ++i)
        canonical_bytes += source.matrices[i].weight_bytes + source.matrices[i].scale_bytes;
    ColiExpertResidentView view = {
        .key = {7, 11},
        .representation = src_rep,
        .generation = 42,
        .tier_mask = COLI_EXPERT_TIER_UMA,
        .resident_bytes = canonical_bytes,
        .allocation_bytes = canonical_bytes,
        .physical = &source,
    };
    ColiJitTransformEstimate estimate;
    if (ops.estimate(ops.context, &view, &dst_rep, &estimate) != 0 ||
        !estimate.resident_bytes || estimate.scratch_bytes || estimate.staging_bytes)
        return 5;
    void *output = malloc((size_t)estimate.allocation_bytes);
    if (!output) return 6;
    if (ops.prepare(ops.context, &view, &dst_rep,
                    output, estimate.allocation_bytes,
                    NULL, 0, NULL, 0) != 0 ||
        ops.validate(ops.context, &view, &dst_rep,
                     output, estimate.resident_bytes) != 0)
        return 7;

    ColiMxfp4Apple8TileExpertHeader *header =
        (ColiMxfp4Apple8TileExpertHeader *)output;
    if (header->magic != COLI_MXFP4_APPLE8_TILE_EXPERT_MAGIC ||
        header->matrix_count != 3)
        return 8;
    uint8_t *payload = (uint8_t *)output + header->matrices[1].offset;
    payload[0] ^= 1u;
    if (ops.validate(ops.context, &view, &dst_rep,
                     output, estimate.resident_bytes) == 0)
        return 9;
    payload[0] ^= 1u;
    if (ops.validate(ops.context, &view, &dst_rep,
                     output, estimate.resident_bytes) != 0)
        return 10;

    free(output);
    for (int i = 0; i < 3; ++i) { free(weights[i]); free(scales[i]); }
    puts("PASS MXFP4 Apple8 exact repack + #135 transform registration");
    return 0;
}

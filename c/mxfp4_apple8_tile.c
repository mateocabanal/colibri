#include "mxfp4_apple8_tile.h"

#include <limits.h>
#include <string.h>

static int mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a && b > UINT64_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

size_t coli_mxfp4_apple8_tile_bytes(uint64_t rows, uint64_t columns) {
    if (!rows || !columns) return 0;
    uint64_t out_tiles = (rows + 7u) / 8u;
    uint64_t k_tiles = (columns + 31u) / 32u;
    uint64_t tiles = 0, bytes = 0;
    if (mul_u64(out_tiles, k_tiles, &tiles) ||
        mul_u64(tiles, COLI_MXFP4_APPLE8_TILE_BYTES, &bytes) ||
        bytes > SIZE_MAX)
        return 0;
    return (size_t)bytes;
}

static int canonical_sizes(uint64_t rows, uint64_t columns,
                           uint64_t *weight_need, uint64_t *scale_need) {
    if (!rows || !columns || !weight_need || !scale_need) return -1;
    uint64_t row_bytes = (columns + 1u) / 2u;
    uint64_t groups = (columns + 31u) / 32u;
    return mul_u64(rows, row_bytes, weight_need) ||
           mul_u64(rows, groups, scale_need) ? -1 : 0;
}

int coli_mxfp4_apple8_tile_repack(
        void *destination, size_t destination_bytes,
        const void *weights, size_t weight_bytes,
        const void *scales, size_t scale_bytes,
        uint64_t rows, uint64_t columns) {
    if (!destination || !weights || !scales) return -1;
    size_t needed = coli_mxfp4_apple8_tile_bytes(rows, columns);
    uint64_t weight_need = 0, scale_need = 0;
    if (!needed || destination_bytes < needed ||
        canonical_sizes(rows, columns, &weight_need, &scale_need) ||
        weight_need > weight_bytes || scale_need > scale_bytes)
        return -1;

    const uint8_t *w = (const uint8_t *)weights;
    const uint8_t *s = (const uint8_t *)scales;
    uint8_t *out = (uint8_t *)destination;
    memset(out, 0, needed);

    const uint64_t row_bytes = (columns + 1u) / 2u;
    const uint64_t groups = (columns + 31u) / 32u;
    for (uint64_t o = 0; o < rows; ++o) {
        const uint8_t *wr = w + o * row_bytes;
        const uint8_t *sr = s + o * groups;
        uint64_t otile = o >> 3;
        uint64_t orow = o & 7u;
        for (uint64_t g = 0; g < groups; ++g) {
            uint8_t *tile = out + (otile * groups + g) *
                                    COLI_MXFP4_APPLE8_TILE_BYTES;
            uint64_t k0 = g * 32u;
            uint64_t remaining = columns - k0;
            uint64_t n = remaining < 32u ? remaining : 32u;
            size_t bytes = (size_t)((n + 1u) / 2u);
            memcpy(tile + orow * 16u, wr + (k0 >> 1), bytes);
            tile[128u + orow] = sr[g];
        }
    }
    return 0;
}

int coli_mxfp4_apple8_tile_validate(
        const void *tile_memory, size_t tile_bytes,
        const void *weights, size_t weight_bytes,
        const void *scales, size_t scale_bytes,
        uint64_t rows, uint64_t columns) {
    if (!tile_memory || !weights || !scales) return -1;
    size_t needed = coli_mxfp4_apple8_tile_bytes(rows, columns);
    uint64_t weight_need = 0, scale_need = 0;
    if (!needed || tile_bytes < needed ||
        canonical_sizes(rows, columns, &weight_need, &scale_need) ||
        weight_need > weight_bytes || scale_need > scale_bytes)
        return -1;

    const uint8_t *tiles = (const uint8_t *)tile_memory;
    const uint8_t *w = (const uint8_t *)weights;
    const uint8_t *s = (const uint8_t *)scales;
    const uint64_t row_bytes = (columns + 1u) / 2u;
    const uint64_t groups = (columns + 31u) / 32u;

    for (uint64_t o = 0; o < rows; ++o) {
        const uint8_t *wr = w + o * row_bytes;
        const uint8_t *sr = s + o * groups;
        uint64_t otile = o >> 3;
        uint64_t orow = o & 7u;
        for (uint64_t g = 0; g < groups; ++g) {
            const uint8_t *tile = tiles + (otile * groups + g) *
                                          COLI_MXFP4_APPLE8_TILE_BYTES;
            uint64_t k0 = g * 32u;
            uint64_t remaining = columns - k0;
            uint64_t n = remaining < 32u ? remaining : 32u;
            size_t bytes = (size_t)((n + 1u) / 2u);
            if (memcmp(tile + orow * 16u, wr + (k0 >> 1), bytes) != 0 ||
                tile[128u + orow] != sr[g])
                return -1;
        }
    }
    return 0;
}

void coli_mxfp4_apple8_source_fixture_representation(ColiRepresentationId *out) {
    if (!out) return;
    *out = (ColiRepresentationId){
        .math_format = COLI_CSF_MATH_MXFP4_E2M1,
        .scale_format = COLI_CSF_SCALE_UE8M0,
        .execution_layout = COLI_CSF_LAYOUT_CANONICAL,
        .execution_layout_abi = 0,
        .kernel_abi = 0,
        .reserved = 0,
        .target_class = 0,
        .group_size = 32,
        .scale_block_rows = 1,
        .scale_block_columns = 32,
        .flags = 0,
    };
}

void coli_mxfp4_apple8_target_fixture_representation(ColiRepresentationId *out) {
    if (!out) return;
    *out = (ColiRepresentationId){
        .math_format = COLI_CSF_MATH_MXFP4_E2M1,
        .scale_format = COLI_CSF_SCALE_UE8M0,
        .execution_layout = COLI_MXFP4_APPLE8_FIXTURE_LAYOUT,
        .execution_layout_abi = COLI_MXFP4_APPLE8_FIXTURE_LAYOUT_ABI,
        .kernel_abi = COLI_MXFP4_APPLE8_FIXTURE_KERNEL_ABI,
        .reserved = 0,
        .target_class = COLI_MXFP4_APPLE8_FIXTURE_TARGET_CLASS,
        .group_size = 32,
        .scale_block_rows = 1,
        .scale_block_columns = 32,
        .flags = 0,
    };
}

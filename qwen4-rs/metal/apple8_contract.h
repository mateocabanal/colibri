#ifndef COLIBRI_APPLE8_CONTRACT_H
#define COLIBRI_APPLE8_CONTRACT_H

#include "coli_format.h"
#include "generated/coli_target_registry.h"

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int coli_apple8_profile_is_v1(const char *profile) {
    return profile && !strcmp(profile, COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1);
}

static inline int coli_apple8_target_contract_compatible(
        const char *profile, uint32_t target_profile_abi,
        uint32_t execution_layout_abi, uint32_t kernel_abi,
        uint32_t target_class) {
    return coli_apple8_profile_is_v1(profile) &&
        target_profile_abi == COLI_TARGET_PROFILE_ABI_APPLE8_V1 &&
        execution_layout_abi == COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1 &&
        kernel_abi == COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1 &&
        target_class == COLI_TARGET_CLASS_APPLE8_METAL_V1;
}

static inline int coli_apple8_tile_matrix_bytes(
        uint64_t rows, uint64_t columns, uint64_t *out) {
    uint64_t row_tiles, k_groups, tiles;
    if (!rows || !columns || !out) return -1;
    row_tiles = rows / COLI_APPLE8_MXFP4_TILE_ROWS +
        (rows % COLI_APPLE8_MXFP4_TILE_ROWS != 0);
    k_groups = columns / COLI_APPLE8_MXFP4_TILE_COLUMNS +
        (columns % COLI_APPLE8_MXFP4_TILE_COLUMNS != 0);
    if (row_tiles && k_groups > UINT64_MAX / row_tiles) return -1;
    tiles = row_tiles * k_groups;
    if (tiles > UINT64_MAX / COLI_APPLE8_MXFP4_TILE_BYTES) return -1;
    *out = tiles * COLI_APPLE8_MXFP4_TILE_BYTES;
    return 0;
}

/* Production Design A: APPLE_MXFP4_TILE8X32_V1 stores one combined matrix
 * payload. weight_* names that primary physical payload for historical CSF
 * descriptor compatibility; the UE8M0 scale bytes are embedded at bytes
 * 128..135 of each 136-byte tile. There is no separate physical scale span.
 *
 * PR3 permits the existing RANS256_G0_NIBBLE codec on that combined physical
 * payload. Decoding reconstructs exactly the same `bytes` execution stream as
 * the raw form; no row-layout transform is involved. */
static inline int coli_apple8_matrix_descriptor_valid(
        const ColiExpertMatrixInfo *m, uint64_t *expected_bytes) {
    uint64_t bytes;
    int weight_codec_valid;
    if (!m || coli_apple8_tile_matrix_bytes(m->rows, m->columns, &bytes)) return 0;
    weight_codec_valid =
        (m->weight_codec == COLI_CSF_CODEC_NONE &&
         m->weight_codec_table_id == 0 &&
         m->weight_stored_bytes == bytes) ||
        (m->weight_codec == COLI_CSF_CODEC_RANS256_G0_NIBBLE &&
         m->weight_codec_table_id != 0 &&
         m->weight_stored_bytes != 0);
    if (m->layout != COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1 ||
        m->math_format != COLI_APPLE8_MXFP4_MATH_FORMAT ||
        m->scale_format != COLI_APPLE8_MXFP4_SCALE_FORMAT ||
        m->scale_block_rows != COLI_APPLE8_MXFP4_SCALE_BLOCK_ROWS ||
        m->scale_block_columns != COLI_APPLE8_MXFP4_SCALE_BLOCK_COLUMNS ||
        m->group_size != COLI_APPLE8_MXFP4_GROUP_SIZE ||
        !weight_codec_valid ||
        m->weight_offset == 0 || (m->weight_offset & 15u) != 0 ||
        m->weight_decoded_bytes != bytes ||
        m->scale_codec != COLI_CSF_CODEC_NONE ||
        m->scale_codec_table_id != 0 || m->scale_offset != 0 ||
        m->scale_stored_bytes != 0 || m->scale_decoded_bytes != 0)
        return 0;
    if (expected_bytes) *expected_bytes = bytes;
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif

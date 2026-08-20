#ifndef COLIBRI_MXFP4_APPLE8_TILE_H
#define COLIBRI_MXFP4_APPLE8_TILE_H

#include "expert_transform.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Deterministic milestone-1 identity only. These values are intentionally NOT
 * production #26 registry assignments. #26 owns the eventual Apple8 physical
 * layout/kernel ABI and target compatibility class.
 */
enum {
    COLI_MXFP4_APPLE8_FIXTURE_LAYOUT = 0x7131,
    COLI_MXFP4_APPLE8_FIXTURE_LAYOUT_ABI = 0x0131,
    COLI_MXFP4_APPLE8_FIXTURE_KERNEL_ABI = 0x0131,
    COLI_MXFP4_APPLE8_FIXTURE_TRANSFORM_ABI = 0x00013101u,
};
#define COLI_MXFP4_APPLE8_FIXTURE_TARGET_CLASS UINT32_C(0x01310008)

/* Apple8 physical matrix tile selected by #131:
 *   8 output rows x 32 K values
 *   128 bytes packed E2M1 payload + 8 colocated E8M0 scales
 * The transformation is a byte-exact physical repack: no dequant/requant. */
enum {
    COLI_MXFP4_APPLE8_TILE_ROWS = 8,
    COLI_MXFP4_APPLE8_TILE_COLUMNS = 32,
    COLI_MXFP4_APPLE8_TILE_BYTES = 136,
};

typedef struct {
    uint32_t role;
    uint64_t rows;
    uint64_t columns;
    const uint8_t *weights;
    uint64_t weight_bytes;
    const uint8_t *scales;
    uint64_t scale_bytes;
} ColiMxfp4Apple8RowMatrix;

typedef struct {
    uint32_t matrix_count;
    ColiMxfp4Apple8RowMatrix matrices[3];
} ColiMxfp4Apple8SourceExpert;

typedef struct {
    uint32_t role;
    uint32_t reserved;
    uint64_t rows;
    uint64_t columns;
    uint64_t offset;
    uint64_t bytes;
} ColiMxfp4Apple8TileMatrix;

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t matrix_count;
    uint32_t header_bytes;
    uint32_t reserved;
    ColiMxfp4Apple8TileMatrix matrices[3];
} ColiMxfp4Apple8TileExpertHeader;

#define COLI_MXFP4_APPLE8_TILE_EXPERT_MAGIC UINT32_C(0x4138544d) /* "MT8A" LE */
#define COLI_MXFP4_APPLE8_TILE_EXPERT_VERSION 1u

size_t coli_mxfp4_apple8_tile_bytes(uint64_t rows, uint64_t columns);

/* Repack one canonical row-major MXFP4 matrix to the #131 Apple8 tile layout.
 * Returns zero on success. The destination may not alias either source span. */
int coli_mxfp4_apple8_tile_repack(
    void *destination, size_t destination_bytes,
    const void *weights, size_t weight_bytes,
    const void *scales, size_t scale_bytes,
    uint64_t rows, uint64_t columns);

/* Byte-level inverse check without materializing a row copy. Padding bytes in
 * partial edge tiles are ignored; every logical E2M1 nibble and E8M0 scale must
 * match the canonical source. */
int coli_mxfp4_apple8_tile_validate(
    const void *tile, size_t tile_bytes,
    const void *weights, size_t weight_bytes,
    const void *scales, size_t scale_bytes,
    uint64_t rows, uint64_t columns);

void coli_mxfp4_apple8_source_fixture_representation(ColiRepresentationId *out);
void coli_mxfp4_apple8_target_fixture_representation(ColiRepresentationId *out);

/* Build/register the exact transform descriptor consumed by #135. The source
 * resident physical pointer is a ColiMxfp4Apple8SourceExpert descriptor; the
 * published target is a relocatable ColiMxfp4Apple8TileExpertHeader blob.
 * No engine decode path depends on this registration in milestone 1. */
int coli_mxfp4_apple8_transform_ops(ColiRepresentationTransformOps *ops);
int coli_mxfp4_apple8_register_transform(ColiRepresentationTransformRegistry *registry);

#ifdef __cplusplus
}
#endif

#endif

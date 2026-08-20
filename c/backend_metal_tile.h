#ifndef COLIBRI_BACKEND_METAL_TILE_H
#define COLIBRI_BACKEND_METAL_TILE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t repack_count;
    uint64_t repack_bytes;
    uint64_t repack_ns;
    uint64_t single_calls;
    uint64_t moe_calls;
    uint64_t fallback_calls;
    uint64_t experts;
    uint64_t wall_ns;
    uint64_t kernel_ns;
    uint64_t scatter_ns;
} ColiMetalTileStats;

/* Milestone-1 opt-in gate. Unset/0 leaves the stock row-fmt7 path untouched. */
int coli_metal_tile_enabled(void);

/* Prepare/replace one source matrix generation in the bounded Apple8 tile cache.
 * The canonical source bytes remain untouched for exact CPU/row fallback. */
int coli_metal_tile_prepare_matrix(const void *weights, const void *scales,
                                   int rows, int columns,
                                   uint64_t source_generation);

void coli_metal_tile_stats(ColiMetalTileStats *stats);

#ifdef __cplusplus
}
#endif

#endif

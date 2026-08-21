#ifndef COLIBRI_APPLE8_METALIO_DIRECT_H
#define COLIBRI_APPLE8_METALIO_DIRECT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Experimental direct Apple8 execution seam.
 *
 * The matrix bytes must already be present in a MetalIO shared-storage slot in
 * COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1 order (8 output rows x 32 input values,
 * 128 packed E2M1 weight bytes followed by 8 UE8M0 scale bytes per 136-byte
 * tile). No canonical MXFP4 buffer and no detile/repack is created.
 *
 * y[S,O] = x[S,I] @ W[O,I]^T
 *
 * `slot_offset` is the byte offset of the combined Apple8 matrix payload inside
 * the MetalIO slot. `matrix_bytes` is the exact logical Apple8 matrix length.
 * Returns 1 on GPU success, 0 when the direct path is unavailable/refused.
 */
int coli_apple8_metalio_direct_init(void);
void coli_apple8_metalio_direct_shutdown(void);
int coli_apple8_metalio_matmul_slot(int slot,
                                    size_t slot_offset,
                                    size_t matrix_bytes,
                                    const float *x,
                                    float *y,
                                    int S,
                                    int I,
                                    int O);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_APPLE8_METALIO_DIRECT_H */

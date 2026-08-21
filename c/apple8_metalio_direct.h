#ifndef COLIBRI_APPLE8_METALIO_DIRECT_H
#define COLIBRI_APPLE8_METALIO_DIRECT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Experimental direct Apple8 execution seam.
 *
 * Matrix bytes must already be present in a MetalIO shared-storage slot in
 * COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1 order (8 output rows x 32 input values,
 * 128 packed E2M1 weight bytes followed by 8 UE8M0 scale bytes per 136-byte
 * tile). No canonical MXFP4 buffer and no detile/repack is created.
 */
int coli_apple8_metalio_direct_init(void);
void coli_apple8_metalio_direct_shutdown(void);

/* y[S,O] = x[S,I] @ W[O,I]^T. */
int coli_apple8_metalio_matmul_slot(int slot,
                                    size_t slot_offset,
                                    size_t matrix_bytes,
                                    const float *x,
                                    float *y,
                                    int S,
                                    int I,
                                    int O);

/*
 * Direct routed-expert primitive:
 *
 *   mid = silu(gate(x)) * up(x)
 *   y   = down(mid)
 *
 * gate/up are [intermediate, hidden], down is [hidden, intermediate]. All
 * three Apple8 payloads live in the same MetalIO slot and are consumed by the
 * GPU in native tile order. The two compute stages are encoded in one command
 * buffer and the intermediate never returns to the CPU.
 */
int coli_apple8_metalio_swiglu_slot(int slot,
                                    size_t gate_offset,
                                    size_t gate_bytes,
                                    size_t up_offset,
                                    size_t up_bytes,
                                    size_t down_offset,
                                    size_t down_bytes,
                                    const float *x,
                                    float *y,
                                    int S,
                                    int hidden,
                                    int intermediate);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_APPLE8_METALIO_DIRECT_H */

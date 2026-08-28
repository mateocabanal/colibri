#ifndef COLIBRI_APPLE8_METALIO_DIRECT_H
#define COLIBRI_APPLE8_METALIO_DIRECT_H

#include <stddef.h>
#include <stdint.h>

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

typedef struct ColiApple8MetalioExpert {
    int slot;
    size_t gate_offset, gate_bytes;
    size_t up_offset, up_bytes;
    size_t down_offset, down_bytes;
} ColiApple8MetalioExpert;

/*
 * Decode-only fused routed layer. For K experts this submits exactly one Metal
 * command buffer containing three ordered stages:
 *
 *   1. all K gate+up projections + SwiGLU
 *   2. all K down projections
 *   3. deterministic K-order weighted reduction
 *
 * The host waits once after the reduction. `route_weights` are consumed in
 * caller order, matching Qwen's top-k accumulation order. K is limited to 64.
 */
int coli_apple8_metalio_moe_topk(const ColiApple8MetalioExpert *experts,
                                 const float *route_weights,
                                 int expert_count,
                                 const float *x,
                                 float *y,
                                 int hidden,
                                 int intermediate);

/* Direct-path profiling uses nanoseconds, matching backend_metal's profiler.
 * The counters are process-local and reset when direct_init creates pipelines. */
void coli_apple8_metalio_profile_get(uint64_t *encode_ns,
                                     uint64_t *submit_ns,
                                     uint64_t *wait_ns,
                                     uint64_t *kernel_ns,
                                     uint64_t *fused_calls,
                                     uint64_t *fused_experts);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_APPLE8_METALIO_DIRECT_H */

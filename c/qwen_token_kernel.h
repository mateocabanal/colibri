#ifndef COLIBRI_QWEN_TOKEN_KERNEL_H
#define COLIBRI_QWEN_TOKEN_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Whole-token Qwen Metal ABI.
 *
 * Stage 1 defines only immutable geometry and the packed weight-blob layout.
 * Runtime/device state is deliberately NOT part of this ABI yet.
 *
 * All offsets are byte offsets from buffer(0), the single packed weight blob.
 * uint64_t is required: the resident Apple8 bank can exceed 4 GiB.
 */
#define QWEN_TOKEN_KERNEL_ABI        1u
#define QWEN_TOKEN_MAX_LAYERS       64u
#define QWEN_TOKEN_BLOB_ALIGN       256u
#define QWEN_TOKEN_OFF_NONE         UINT64_MAX

#define QWEN_TOKEN_LAYER_ATTN       0u
#define QWEN_TOKEN_LAYER_GDN        1u

/*
 * Dense matrix contract for the first implementation:
 *
 *   WT projections/router/shared-expert matrices: BF16 row-major
 *   norms/A_log/dt_bias/conv1d:                    F32
 *   routed experts:                               Apple8 MXFP4 tile8x32
 *
 * The engine seam must reject/fallback rather than reinterpret another format.
 */
typedef struct QwenTokenLayerBlob {
    uint32_t kind;                  /* QWEN_TOKEN_LAYER_* */
    uint32_t expert_slots;          /* resident Apple8 bank slots */

    /* common */
    uint64_t in_ln;
    uint64_t post_ln;

    /* full attention; QWEN_TOKEN_OFF_NONE on GDN layers */
    uint64_t attn_q;
    uint64_t attn_k;
    uint64_t attn_v;
    uint64_t attn_o;
    uint64_t attn_qn;
    uint64_t attn_kn;

    /* Gated DeltaNet; QWEN_TOKEN_OFF_NONE on attention layers */
    uint64_t gdn_A_log;
    uint64_t gdn_dt_bias;
    uint64_t gdn_conv1d;
    uint64_t gdn_in_a;
    uint64_t gdn_in_b;
    uint64_t gdn_in_qkv;
    uint64_t gdn_in_z;
    uint64_t gdn_out;
    uint64_t gdn_norm;

    /* MoE, present on every layer */
    uint64_t router;
    uint64_t se_gate;
    uint64_t se_up;
    uint64_t se_down;
    uint64_t se_g;

    /*
     * Fixed-stride resident routed-expert bank for this layer.
     * Slot contents use the Apple8 relative offsets in QwenTokenKernelParams.
     */
    uint64_t expert_bank;
    uint64_t expert_slot_stride;
} QwenTokenLayerBlob;

typedef struct QwenTokenKernelParams {
    uint32_t abi_version;
    uint32_t n_layers;

    uint32_t hidden;
    uint32_t max_t;

    /* full attention geometry */
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;

    /* GDN geometry */
    uint32_t lin_k_heads;
    uint32_t lin_k_dim;
    uint32_t lin_v_heads;
    uint32_t lin_v_dim;
    uint32_t conv_kernel;

    /* MoE geometry */
    uint32_t n_experts;
    uint32_t topk;
    uint32_t moe_inter;
    uint32_t shared_inter;

    float eps;
    float theta;

    /*
     * Canonical contents of one resident Apple8 expert slot:
     *
     *   slot_base + expert_gate_rel
     *   slot_base + expert_up_rel
     *   slot_base + expert_down_rel
     *
     * The corresponding byte counts let the builder validate that every
     * loaded expert exactly matches the geometry before exposing the slot.
     */
    uint64_t expert_gate_rel;
    uint64_t expert_gate_bytes;
    uint64_t expert_up_rel;
    uint64_t expert_up_bytes;
    uint64_t expert_down_rel;
    uint64_t expert_down_bytes;

    uint64_t blob_bytes;

    QwenTokenLayerBlob layer[QWEN_TOKEN_MAX_LAYERS];
} QwenTokenKernelParams;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* COLIBRI_QWEN_TOKEN_KERNEL_H */
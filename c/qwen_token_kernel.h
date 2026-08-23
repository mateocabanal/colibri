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

/* ---------- Stage 2: mutable whole-token device state ---------- */

#define QWEN_TOKEN_MAX_TOPK          16u
#define QWEN_TOKEN_STATE_ALIGN       256u
#define QWEN_TOKEN_WORK_VECS         4u
#define QWEN_TOKEN_EXPERT_EMPTY      UINT32_MAX

#define QWEN_TOKEN_STATUS_RUNNABLE       0u
#define QWEN_TOKEN_STATUS_DONE           1u
#define QWEN_TOKEN_STATUS_NEED_EXPERTS   2u
#define QWEN_TOKEN_STATUS_ERROR          3u

#define QWEN_TOKEN_PHASE_LAYER_START     0u
#define QWEN_TOKEN_PHASE_MOE             1u

/*
 * Host-visible continuation record.
 *
 * Critical invariant:
 *   NEED_EXPERTS => resume_phase == QWEN_TOKEN_PHASE_MOE.
 *
 * At that point the layer's attention/GDN path has already executed and any
 * recurrent state mutation is committed.  routed_expert/routed_weight and
 * the postnorm vector at layout.moe_input_off are therefore the authoritative
 * continuation state.  Resume MUST NOT rerun the router or pre-MoE half.
 */
typedef struct QwenTokenMissRecord {
    uint32_t status;
    uint32_t resume_phase;
    uint32_t layer;
    uint32_t token_pos;

    uint32_t routed_count;
    uint32_t missing_count;
    uint32_t error_code;
    uint32_t reserved0;

    uint64_t dispatch_id;

    uint32_t routed_expert[QWEN_TOKEN_MAX_TOPK];
    float    routed_weight[QWEN_TOKEN_MAX_TOPK];
    uint32_t missing_expert[QWEN_TOKEN_MAX_TOPK];
} QwenTokenMissRecord;

/*
 * Byte offsets into one mutable MTLBuffer.
 *
 * The regions are deliberately disjoint:
 *
 *   hidden_scratch | KV | GDN conv | GDN S | expert map | miss record
 *
 * In particular, gdn_s_* can never alias kv_*.
 */
typedef struct QwenTokenDeviceLayout {
    uint64_t total_bytes;

    uint64_t hidden_scratch_off;
    uint64_t hidden_scratch_bytes;

    uint64_t kv_off;
    uint64_t kv_bytes;

    uint64_t gdn_conv_off;
    uint64_t gdn_conv_bytes;

    uint64_t gdn_s_off;
    uint64_t gdn_s_bytes;

    uint64_t expert_map_off;
    uint64_t expert_map_bytes;

    uint64_t miss_off;
    uint64_t miss_bytes;

    /*
     * Persistent/scratch vectors inside hidden_scratch.
     *
     * residual:
     *   current residual stream.
     *
     * moe_input:
     *   postnorm input to MoE.  This MUST survive NEED_EXPERTS because it is
     *   the input used when resuming directly at QWEN_TOKEN_PHASE_MOE.
     *
     * work[]:
     *   generic device-memory workspace.  No large GDN temporaries belong in
     *   threadgroup memory.
     */
    uint64_t residual_off;
    uint64_t normed_off;
    uint64_t branch_off;
    uint64_t moe_input_off;
    uint64_t moe_acc_off;
    uint64_t work_off[QWEN_TOKEN_WORK_VECS];
    uint64_t work_width;

    /*
     * Per-layer offsets.  Irrelevant entries are QWEN_TOKEN_OFF_NONE.
     *
     * Attention KV is F32.
     * GDN convolution history is F32.
     * GDN recurrent S is F32.
     * Expert-map entries are uint32_t expert IDs, one per bank slot.
     */
    uint64_t kv_k_off[QWEN_TOKEN_MAX_LAYERS];
    uint64_t kv_v_off[QWEN_TOKEN_MAX_LAYERS];

    uint64_t gdn_conv_layer_off[QWEN_TOKEN_MAX_LAYERS];
    uint64_t gdn_s_layer_off[QWEN_TOKEN_MAX_LAYERS];

    uint64_t expert_map_layer_off[QWEN_TOKEN_MAX_LAYERS];
    uint32_t expert_map_slots[QWEN_TOKEN_MAX_LAYERS];
} QwenTokenDeviceLayout;

typedef struct QwenTokenDeviceState QwenTokenDeviceState;

/*
 * Immutable packed-weight buffer used by the whole-token kernel.
 *
 * Static dense/vector spans are copied at creation time.  Routed-expert bank
 * regions are part of blob_bytes but intentionally have no source span:
 * Stage 5 populates those regions on demand and publishes the corresponding
 * expert ID through the mutable state's resident-expert map.
 */
typedef struct QwenTokenWeightBlob QwenTokenWeightBlob;

typedef struct QwenTokenBlobSpan {
    const void *src;
    uint64_t dst_offset;
    uint64_t bytes;
} QwenTokenBlobSpan;

QwenTokenWeightBlob *qwen_token_weight_blob_create(
    const QwenTokenKernelParams *p,
    const QwenTokenBlobSpan *spans,
    uint32_t span_count,
    char *err,
    uint64_t err_cap);

void qwen_token_weight_blob_destroy(QwenTokenWeightBlob *blob);

void *qwen_token_weight_blob_mtl_buffer(QwenTokenWeightBlob *blob);
void *qwen_token_weight_blob_contents(QwenTokenWeightBlob *blob);

/*
 * Pure layout builder: no Metal interaction.
 * Returns 1 on success, 0 on invalid geometry/overflow.
 */
int qwen_token_device_layout_init(
    const QwenTokenKernelParams *p,
    QwenTokenDeviceLayout *layout,
    char *err,
    uint64_t err_cap);

/*
 * Stage-2 standalone Metal allocation.
 *
 * The buffer uses Shared storage on Apple Silicon: it is directly visible to
 * the GPU while allowing the host to inspect the small miss record after the
 * token dispatch.  Kernel execution still keeps KV/GDN state resident in this
 * buffer; there is no host round-trip for those regions.
 */
QwenTokenDeviceState *qwen_token_device_state_create(
    const QwenTokenKernelParams *p,
    char *err,
    uint64_t err_cap);

void qwen_token_device_state_destroy(QwenTokenDeviceState *state);

const QwenTokenDeviceLayout *qwen_token_device_state_layout(
    const QwenTokenDeviceState *state);

/* Stage-4/kernel plumbing will consume the opaque MTLBuffer handle. */
void *qwen_token_device_state_mtl_buffer(QwenTokenDeviceState *state);

/* Host-visible contents, primarily for the miss record and Stage-2 tests. */
void *qwen_token_device_state_contents(QwenTokenDeviceState *state);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* COLIBRI_QWEN_TOKEN_KERNEL_H */
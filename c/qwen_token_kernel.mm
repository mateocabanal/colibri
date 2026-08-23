#include "qwen_token_kernel.h"

#import <Metal/Metal.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct QwenTokenDeviceState {
    id<MTLDevice> device;
    id<MTLBuffer> buffer;
    QwenTokenDeviceLayout layout;
};

struct QwenTokenWeightBlob {
    id<MTLDevice> device;
    id<MTLBuffer> buffer;
    uint64_t bytes;
};

static int
qtk_fail(char *err, uint64_t cap, const char *msg)
{
    if (err && cap)
        snprintf(err, (size_t)cap, "%s", msg);
    return 0;
}

static int
qtk_mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a && b > UINT64_MAX / a)
        return 0;
    *out = a * b;
    return 1;
}

static int
qtk_align(uint64_t x, uint64_t *out)
{
    const uint64_t a = QWEN_TOKEN_STATE_ALIGN;
    if (x > UINT64_MAX - (a - 1))
        return 0;
    *out = (x + a - 1) & ~(a - 1);
    return 1;
}

static int
qtk_reserve(uint64_t *cursor, uint64_t bytes, uint64_t *off)
{
    uint64_t p;
    if (!qtk_align(*cursor, &p))
        return 0;
    if (bytes > UINT64_MAX - p)
        return 0;
    *off = p;
    *cursor = p + bytes;
    return 1;
}

static uint64_t
qtk_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

int
qwen_token_device_layout_init(
    const QwenTokenKernelParams *p,
    QwenTokenDeviceLayout *l,
    char *err,
    uint64_t err_cap)
{
    if (!p || !l)
        return qtk_fail(err, err_cap, "qwen token state: null params/layout");
    if (p->abi_version != QWEN_TOKEN_KERNEL_ABI)
        return qtk_fail(err, err_cap, "qwen token state: ABI mismatch");
    if (!p->n_layers || p->n_layers > QWEN_TOKEN_MAX_LAYERS)
        return qtk_fail(err, err_cap, "qwen token state: invalid layer count");
    if (!p->hidden)
        return qtk_fail(err, err_cap, "qwen token state: hidden=0");
    if (!p->topk || p->topk > QWEN_TOKEN_MAX_TOPK)
        return qtk_fail(err, err_cap, "qwen token state: unsupported top-k");

    memset(l, 0, sizeof(*l));

    for (uint32_t i = 0; i < QWEN_TOKEN_MAX_LAYERS; i++) {
        l->kv_k_off[i] = QWEN_TOKEN_OFF_NONE;
        l->kv_v_off[i] = QWEN_TOKEN_OFF_NONE;
        l->gdn_conv_layer_off[i] = QWEN_TOKEN_OFF_NONE;
        l->gdn_s_layer_off[i] = QWEN_TOKEN_OFF_NONE;
        l->expert_map_layer_off[i] = QWEN_TOKEN_OFF_NONE;
    }

    uint64_t k_width, v_width, gdn_conv_width;
    uint64_t attn_q_width, attn_kv_width;

    if (!qtk_mul(p->lin_k_heads, p->lin_k_dim, &k_width) ||
        !qtk_mul(p->lin_v_heads, p->lin_v_dim, &v_width) ||
        !qtk_mul(p->n_heads, p->head_dim, &attn_q_width) ||
        !qtk_mul(p->n_kv_heads, p->head_dim, &attn_kv_width))
        return qtk_fail(err, err_cap, "qwen token state: geometry overflow");

    if (k_width > (UINT64_MAX - v_width) / 2)
        return qtk_fail(err, err_cap, "qwen token state: GDN width overflow");

    /* q + k + v causal-convolution channels. */
    gdn_conv_width = 2 * k_width + v_width;

    /*
     * Four generic work vectors.  Width covers every currently-known
     * projection/intermediate class; no large activation needs threadgroup
     * storage merely because it is wider than hidden.
     */
    uint64_t work_width = p->hidden;
    work_width = qtk_max(work_width, attn_q_width);
    work_width = qtk_max(work_width, attn_kv_width);
    work_width = qtk_max(work_width, gdn_conv_width);
    work_width = qtk_max(work_width, p->moe_inter);
    work_width = qtk_max(work_width, p->shared_inter);
    work_width = qtk_max(work_width, p->n_experts);
    l->work_width = work_width;

    uint64_t cursor = 0;
    uint64_t bytes;

    /*
     * Hidden/scratch region:
     *   5 x hidden vectors
     *   4 x work_width vectors
     */
    if (!qtk_align(cursor, &l->hidden_scratch_off))
        return qtk_fail(err, err_cap, "qwen token state: layout overflow");

#define RESERVE_F32(count_, off_)                                          \
    do {                                                                   \
        if (!qtk_mul((uint64_t)(count_), sizeof(float), &bytes) ||         \
            !qtk_reserve(&cursor, bytes, &(off_)))                         \
            return qtk_fail(err, err_cap,                                  \
                            "qwen token state: layout overflow");           \
    } while (0)

    RESERVE_F32(p->hidden, l->residual_off);
    RESERVE_F32(p->hidden, l->normed_off);
    RESERVE_F32(p->hidden, l->branch_off);
    RESERVE_F32(p->hidden, l->moe_input_off);
    RESERVE_F32(p->hidden, l->moe_acc_off);

    for (uint32_t i = 0; i < QWEN_TOKEN_WORK_VECS; i++)
        RESERVE_F32(work_width, l->work_off[i]);

    l->hidden_scratch_bytes = cursor - l->hidden_scratch_off;

    /* ---------- full-attention KV region ---------- */

    if (!qtk_align(cursor, &l->kv_off))
        return qtk_fail(err, err_cap, "qwen token state: KV alignment overflow");

    for (uint32_t i = 0; i < p->n_layers; i++) {
        if (p->layer[i].kind != QWEN_TOKEN_LAYER_ATTN)
            continue;

        if (!p->max_t)
            return qtk_fail(err, err_cap,
                            "qwen token state: attention layer with max_t=0");

        uint64_t elems;
        if (!qtk_mul(p->max_t, attn_kv_width, &elems))
            return qtk_fail(err, err_cap, "qwen token state: KV size overflow");

        RESERVE_F32(elems, l->kv_k_off[i]);
        RESERVE_F32(elems, l->kv_v_off[i]);
    }

    l->kv_bytes = cursor - l->kv_off;

    /* ---------- GDN causal-convolution history ---------- */

    if (!qtk_align(cursor, &l->gdn_conv_off))
        return qtk_fail(err, err_cap,
                        "qwen token state: GDN conv alignment overflow");

    for (uint32_t i = 0; i < p->n_layers; i++) {
        if (p->layer[i].kind != QWEN_TOKEN_LAYER_GDN)
            continue;

        /*
         * Persist only the previous K-1 q/k/v samples.
         * The current projected sample remains in work memory.
         */
        uint64_t history = p->conv_kernel ? p->conv_kernel - 1 : 0;
        uint64_t elems;

        if (!qtk_mul(history, gdn_conv_width, &elems))
            return qtk_fail(err, err_cap,
                            "qwen token state: GDN conv size overflow");

        if (elems)
            RESERVE_F32(elems, l->gdn_conv_layer_off[i]);
    }

    l->gdn_conv_bytes = cursor - l->gdn_conv_off;

    /* ---------- GDN recurrent S: never aliases KV ---------- */

    if (!qtk_align(cursor, &l->gdn_s_off))
        return qtk_fail(err, err_cap,
                        "qwen token state: GDN S alignment overflow");

    uint64_t gdn_s_elems;
    if (!qtk_mul(p->lin_v_heads, p->lin_k_dim, &gdn_s_elems) ||
        !qtk_mul(gdn_s_elems, p->lin_v_dim, &gdn_s_elems))
        return qtk_fail(err, err_cap, "qwen token state: GDN S size overflow");

    for (uint32_t i = 0; i < p->n_layers; i++) {
        if (p->layer[i].kind == QWEN_TOKEN_LAYER_GDN)
            RESERVE_F32(gdn_s_elems, l->gdn_s_layer_off[i]);
    }

    l->gdn_s_bytes = cursor - l->gdn_s_off;

    /* ---------- resident expert-id map ---------- */

    if (!qtk_align(cursor, &l->expert_map_off))
        return qtk_fail(err, err_cap,
                        "qwen token state: expert-map alignment overflow");

    for (uint32_t i = 0; i < p->n_layers; i++) {
        uint64_t map_bytes;

        l->expert_map_slots[i] = p->layer[i].expert_slots;
        if (!p->layer[i].expert_slots)
            continue;

        if (!qtk_mul(p->layer[i].expert_slots,
                     sizeof(uint32_t), &map_bytes) ||
            !qtk_reserve(&cursor, map_bytes,
                         &l->expert_map_layer_off[i]))
            return qtk_fail(err, err_cap,
                            "qwen token state: expert-map size overflow");
    }

    l->expert_map_bytes = cursor - l->expert_map_off;

    /* ---------- continuation / miss record ---------- */

    if (!qtk_align(cursor, &l->miss_off))
        return qtk_fail(err, err_cap,
                        "qwen token state: miss-record alignment overflow");

    if (!qtk_reserve(&cursor, sizeof(QwenTokenMissRecord), &l->miss_off))
        return qtk_fail(err, err_cap,
                        "qwen token state: miss-record size overflow");

    l->miss_bytes = sizeof(QwenTokenMissRecord);

    if (!qtk_align(cursor, &l->total_bytes))
        return qtk_fail(err, err_cap,
                        "qwen token state: final size overflow");

#undef RESERVE_F32

    return 1;
}

QwenTokenDeviceState *
qwen_token_device_state_create(
    const QwenTokenKernelParams *p,
    char *err,
    uint64_t err_cap)
{
    QwenTokenDeviceLayout layout;
    if (!qwen_token_device_layout_init(p, &layout, err, err_cap))
        return NULL;

    if (layout.total_bytes > (uint64_t)SIZE_MAX) {
        qtk_fail(err, err_cap, "qwen token state: buffer exceeds size_t");
        return NULL;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        qtk_fail(err, err_cap, "qwen token state: no Metal device");
        return NULL;
    }

    QwenTokenDeviceState *s =
        (QwenTokenDeviceState *)calloc(1, sizeof(*s));
    if (!s) {
        qtk_fail(err, err_cap, "qwen token state: OOM");
        return NULL;
    }

    s->device = [device retain];
    s->buffer =
        [device newBufferWithLength:(NSUInteger)layout.total_bytes
                            options:MTLResourceStorageModeShared];

    if (!s->buffer) {
        [s->device release];
        free(s);
        qtk_fail(err, err_cap, "qwen token state: MTLBuffer allocation failed");
        return NULL;
    }

    s->layout = layout;

    uint8_t *base = (uint8_t *)[s->buffer contents];
    memset(base, 0, (size_t)layout.total_bytes);

    /* Empty resident-bank map.  UINT32_MAX can never be a legal expert ID. */
    for (uint32_t l = 0; l < p->n_layers; l++) {
        if (layout.expert_map_layer_off[l] == QWEN_TOKEN_OFF_NONE)
            continue;

        uint32_t *map = (uint32_t *)
            (base + layout.expert_map_layer_off[l]);

        for (uint32_t j = 0; j < layout.expert_map_slots[l]; j++)
            map[j] = QWEN_TOKEN_EXPERT_EMPTY;
    }

    QwenTokenMissRecord *m =
        (QwenTokenMissRecord *)(base + layout.miss_off);

    m->status = QWEN_TOKEN_STATUS_RUNNABLE;
    m->resume_phase = QWEN_TOKEN_PHASE_LAYER_START;
    m->layer = 0;
    m->token_pos = 0;

    return s;
}

void
qwen_token_device_state_destroy(QwenTokenDeviceState *s)
{
    if (!s)
        return;

    [s->buffer release];
    [s->device release];
    free(s);
}

const QwenTokenDeviceLayout *
qwen_token_device_state_layout(const QwenTokenDeviceState *s)
{
    return s ? &s->layout : NULL;
}

void *
qwen_token_device_state_mtl_buffer(QwenTokenDeviceState *s)
{
    return s ? (void *)s->buffer : NULL;
}

void *
qwen_token_device_state_contents(QwenTokenDeviceState *s)
{
    return s ? [s->buffer contents] : NULL;
}

QwenTokenWeightBlob *
qwen_token_weight_blob_create(
    const QwenTokenKernelParams *p,
    const QwenTokenBlobSpan *spans,
    uint32_t span_count,
    char *err,
    uint64_t err_cap)
{
    if (!p || !p->blob_bytes) {
        qtk_fail(err, err_cap, "qwen token blob: empty parameters/blob");
        return NULL;
    }
    if (span_count && !spans) {
        qtk_fail(err, err_cap, "qwen token blob: null span table");
        return NULL;
    }
    if (p->blob_bytes > (uint64_t)SIZE_MAX) {
        qtk_fail(err, err_cap, "qwen token blob: size exceeds host address space");
        return NULL;
    }

    for (uint32_t i = 0; i < span_count; i++) {
        const QwenTokenBlobSpan *s = &spans[i];
        if (!s->src || !s->bytes ||
            s->dst_offset > p->blob_bytes ||
            s->bytes > p->blob_bytes - s->dst_offset) {
            qtk_fail(err, err_cap, "qwen token blob: invalid source span");
            return NULL;
        }
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        qtk_fail(err, err_cap, "qwen token blob: no Metal device");
        return NULL;
    }

    QwenTokenWeightBlob *b =
        (QwenTokenWeightBlob *)calloc(1, sizeof(*b));
    if (!b) {
        qtk_fail(err, err_cap, "qwen token blob: OOM");
        return NULL;
    }

    b->device = [device retain];
    b->buffer =
        [device newBufferWithLength:(NSUInteger)p->blob_bytes
                            options:MTLResourceStorageModeShared];

    if (!b->buffer) {
        [b->device release];
        free(b);
        qtk_fail(err, err_cap, "qwen token blob: MTLBuffer allocation failed");
        return NULL;
    }

    b->bytes = p->blob_bytes;

    /*
     * Do NOT memset the full buffer.  Expert-bank ranges may be multi-GB and
     * are invalid until their expert-map entries are published anyway.
     * Touch only immutable static weights here.
     */
    uint8_t *base = (uint8_t *)[b->buffer contents];
    for (uint32_t i = 0; i < span_count; i++)
        memcpy(base + spans[i].dst_offset, spans[i].src,
               (size_t)spans[i].bytes);

    return b;
}

void
qwen_token_weight_blob_destroy(QwenTokenWeightBlob *b)
{
    if (!b)
        return;
    [b->buffer release];
    [b->device release];
    free(b);
}

void *
qwen_token_weight_blob_mtl_buffer(QwenTokenWeightBlob *b)
{
    return b ? (void *)b->buffer : NULL;
}

void *
qwen_token_weight_blob_contents(QwenTokenWeightBlob *b)
{
    return b ? [b->buffer contents] : NULL;
}

// ---- Stage-4 runtime compile selfcheck (Hermes) ----
extern "C" int qwen_token_kernel_selfcheck(char *err, uint64_t err_cap) {
    static const char *QTK_MSL = R"METAL(
#include <metal_stdlib>
using namespace metal;


// ---- MSL ABI mirror of qwen_token_kernel.h (host C header is not visible to MSL) ----
#define QWEN_TOKEN_MAX_LAYERS 64u
#define QWEN_TOKEN_MAX_TOPK    16u
#define QWEN_TOKEN_WORK_VECS   4u
#define QWEN_TOKEN_OFF_NONE    (~(ulong)0)

#define QWEN_TOKEN_LAYER_ATTN  0u
#define QWEN_TOKEN_LAYER_GDN   1u

struct QwenTokenLayerBlob {
    uint  kind;
    uint  expert_slots;
    ulong in_ln;
    ulong post_ln;
    ulong attn_q, attn_k, attn_v, attn_o, attn_qn, attn_kn;
    ulong gdn_A_log, gdn_dt_bias, gdn_conv1d;
    ulong gdn_in_a, gdn_in_b, gdn_in_qkv, gdn_in_z, gdn_out, gdn_norm;
    ulong router, se_gate, se_up, se_down, se_g;
    ulong expert_bank;
    ulong expert_slot_stride;
};

struct QwenTokenKernelParams {
    uint  abi_version;
    uint  n_layers;
    uint  hidden;
    uint  max_t;
    uint  n_heads, n_kv_heads, head_dim, rotary_dim;
    uint  lin_k_heads, lin_k_dim, lin_v_heads, lin_v_dim, conv_kernel;
    uint  n_experts, topk, moe_inter, shared_inter;
    float eps, theta;
    ulong expert_gate_rel, expert_gate_bytes;
    ulong expert_up_rel, expert_up_bytes;
    ulong expert_down_rel, expert_down_bytes;
    ulong blob_bytes;
    QwenTokenLayerBlob layer[QWEN_TOKEN_MAX_LAYERS];
};

struct QwenTokenMissRecord {
    uint  status, resume_phase, layer, token_pos;
    uint  routed_count, missing_count, error_code, reserved0;
    ulong dispatch_id;
    uint  routed_expert[QWEN_TOKEN_MAX_TOPK];
    float routed_weight[QWEN_TOKEN_MAX_TOPK];
    uint  missing_expert[QWEN_TOKEN_MAX_TOPK];
};

struct QwenTokenDeviceLayout {
    ulong total_bytes;
    ulong hidden_scratch_off, hidden_scratch_bytes;
    ulong kv_off, kv_bytes;
    ulong gdn_conv_off, gdn_conv_bytes;
    ulong gdn_s_off, gdn_s_bytes;
    ulong expert_map_off, expert_map_bytes;
    ulong miss_off, miss_bytes;
    ulong residual_off, normed_off, branch_off, moe_input_off, moe_acc_off;
    ulong work_off[QWEN_TOKEN_WORK_VECS];
    ulong work_width;
    ulong kv_k_off[QWEN_TOKEN_MAX_LAYERS];
    ulong kv_v_off[QWEN_TOKEN_MAX_LAYERS];
    ulong gdn_conv_layer_off[QWEN_TOKEN_MAX_LAYERS];
    ulong gdn_s_layer_off[QWEN_TOKEN_MAX_LAYERS];
    ulong expert_map_layer_off[QWEN_TOKEN_MAX_LAYERS];
    uint  expert_map_slots[QWEN_TOKEN_MAX_LAYERS];
};

#define QTK_THREADS 1024u

static inline float
qtk_bf16(device const uchar *blob, ulong byte_off, uint i)
{
    device const ushort *p =
        reinterpret_cast<device const ushort *>(blob + byte_off);
    return as_type<float>((uint)p[i] << 16);
}

static inline float
qtk_f32(device const uchar *blob, ulong byte_off, uint i)
{
    device const float *p =
        reinterpret_cast<device const float *>(blob + byte_off);
    return p[i];
}

static inline device float *
qtk_state_f32(device uchar *state, ulong byte_off)
{
    return reinterpret_cast<device float *>(state + byte_off);
}

static inline device const float *
qtk_state_cf32(device const uchar *state, ulong byte_off)
{
    return reinterpret_cast<device const float *>(state + byte_off);
}

static inline float
qtk_silu(float x)
{
    return x / (1.0f + exp(-x));
}

static inline float
qtk_sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

/*
 * Dense BF16 row-major matvec.
 *
 * dst[rows] = W[rows, cols] . x[cols]
 *
 * One thread owns each row and performs its reduction in ascending-column
 * order.  Rows > 1024 are handled by the same owner at +1024 strides.
 */
static inline void
qtk_bf16_matvec(device const uchar *blob,
                ulong w_off,
                device const float *x,
                device float *dst,
                uint rows,
                uint cols,
                uint tid)
{
    for (uint r = tid; r < rows; r += QTK_THREADS) {
        float s = 0.0f;
        ulong base = (ulong)r * (ulong)cols;

        for (uint c = 0; c < cols; ++c)
            s += qtk_bf16(blob, w_off, (uint)(base + c)) * x[c];

        dst[r] = s;
    }
}

/*
 * Exact-order RMSNorm.
 *
 * Thread 0 performs the scalar reduction.  The 1024 threads then apply the
 * resulting scale elementwise.
 */
static inline void
qtk_rmsnorm(device const float *x,
            device const uchar *blob,
            ulong w_off,
            device float *y,
            uint n,
            float eps,
            uint tid,
            threadgroup float *tg_scalar)
{
    if (tid == 0u) {
        float ss = 0.0f;

        for (uint i = 0; i < n; ++i) {
            float v = x[i];
            ss += v * v;
        }

        tg_scalar[0] =
            rsqrt(ss / (float)n + eps);
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    const float inv = tg_scalar[0];

    for (uint i = tid; i < n; i += QTK_THREADS)
        y[i] = x[i] * qtk_f32(blob, w_off, i) * inv;

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

static inline void
qtk_residual_add(device float *residual,
                 device const float *delta,
                 uint n,
                 uint tid)
{
    for (uint i = tid; i < n; i += QTK_THREADS)
        residual[i] += delta[i];

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * Split-half partial RoPE.
 *
 * rotary_dim is the number of scalar dimensions rotated.  For rotary_dim=64,
 * dimensions [0,31] rotate against [32,63]; dimensions >=64 are untouched.
 */
static inline void
qtk_rope_head(device float *x,
              uint head_base,
              uint rotary_dim,
              uint token_pos,
              float theta)
{
    const uint rot_half = rotary_dim >> 1;

    for (uint i = 0; i < rot_half; ++i) {
        const float freq =
            pow(theta, -2.0f * (float)i / (float)rotary_dim);
        const float a = (float)token_pos * freq;
        const float cs = cos(a);
        const float sn = sin(a);

        const uint i0 = head_base + i;
        const uint i1 = head_base + rot_half + i;

        const float x0 = x[i0];
        const float x1 = x[i1];

        x[i0] = x0 * cs - x1 * sn;
        x[i1] = x1 * cs + x0 * sn;
    }
}

/*
 * Per-head RMS q/k norm followed by partial split-half RoPE.
 *
 * qn/kn are head-dimension weights and therefore reused by every attention
 * head.
 */
static inline void
qtk_attn_norm_rope(device float *q,
                   device float *k,
                   device const uchar *blob,
                   ulong qn_off,
                   ulong kn_off,
                   uint n_heads,
                   uint n_kv_heads,
                   uint head_dim,
                   uint rotary_dim,
                   uint token_pos,
                   float theta,
                   float eps,
                   uint tid)
{
    if (tid < n_heads) {
        const uint base = tid * head_dim;

        float ss = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            const float v = q[base + d];
            ss += v * v;
        }

        const float inv =
            rsqrt(ss / (float)head_dim + eps);

        for (uint d = 0; d < head_dim; ++d)
            q[base + d] =
                q[base + d] * qtk_f32(blob, qn_off, d) * inv;

        qtk_rope_head(q, base, rotary_dim, token_pos, theta);
    }

    if (tid < n_kv_heads) {
        const uint base = tid * head_dim;

        float ss = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            const float v = k[base + d];
            ss += v * v;
        }

        const float inv =
            rsqrt(ss / (float)head_dim + eps);

        for (uint d = 0; d < head_dim; ++d)
            k[base + d] =
                k[base + d] * qtk_f32(blob, kn_off, d) * inv;

        qtk_rope_head(k, base, rotary_dim, token_pos, theta);
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * Depthwise causal conv for the GDN q/k/v projection.
 *
 * Persistent state holds exactly kk-1 samples/channel:
 *
 *     [channel][oldest ... newest]
 *
 * For kk=4, conv taps consume:
 *
 *     tap0*prev0 + tap1*prev1 + tap2*prev2 + tap3*current
 *
 * The state is shifted only after the convolution value has been formed.
 */
static inline void
qtk_gdn_conv(device float *qkv,
             device float *conv_state,
             device const uchar *blob,
             ulong conv_w_off,
             uint channels,
             uint kk,
             uint tid)
{
    const uint hist = kk - 1u;

    for (uint c = tid; c < channels; c += QTK_THREADS) {
        device float *cs = conv_state + (ulong)c * hist;

        float y = 0.0f;

        for (uint j = 0; j < hist; ++j)
            y += qtk_f32(blob,
                         conv_w_off,
                         c * kk + j) * cs[j];

        const float cur = qkv[c];

        y += qtk_f32(blob,
                     conv_w_off,
                     c * kk + hist) * cur;

        for (uint j = 0; j + 1u < hist; ++j)
            cs[j] = cs[j + 1u];

        cs[hist - 1u] = cur;

        qkv[c] = qtk_silu(y);
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * One thread owns one complete GDN head.
 *
 * qkv layout:
 *   q = [kdim]
 *   k = [kdim]
 *   v = [vdim]
 *
 * S layout:
 *   [head][kd][vd]
 *
 * The q/k vectors are normalized independently per head.  State decay,
 * delta update, and output accumulation all use ascending kd order.
 */
static inline void
qtk_gdn_recurrence(device float *qkv,
                   device const float *a,
                   device const float *b,
                   device float *S,
                   device float *gdn_y,
                   device const uchar *blob,
                   ulong A_log_off,
                   ulong dt_bias_off,
                   uint k_heads,
                   uint k_head_dim,
                   uint v_heads,
                   uint v_head_dim,
                   uint tid)
{
    if (tid >= v_heads)
        return;

    const uint h = tid;

    /*
     * Current model geometry is 1:1:
     *     k_heads == v_heads == 8.
     *
     * Keep the indexing explicit so no head writes another head's state.
     */
    const uint kh = h;

    const uint kdim = k_heads * k_head_dim;
    const uint vdim = v_heads * v_head_dim;

    device float *q =
        qkv + kh * k_head_dim;

    device float *k =
        qkv + kdim + kh * k_head_dim;

    device float *v =
        qkv + 2u * kdim + h * v_head_dim;

    device float *yh =
        gdn_y + h * v_head_dim;

    device float *Sh =
        S + (ulong)h *
            (ulong)k_head_dim *
            (ulong)v_head_dim;

    float qss = 0.0f;
    float kss = 0.0f;

    for (uint kd = 0; kd < k_head_dim; ++kd) {
        const float qv = q[kd];
        const float kv = k[kd];

        qss += qv * qv;
        kss += kv * kv;
    }

    const float qinv = rsqrt(qss + 1.0e-6f);
    const float kinv = rsqrt(kss + 1.0e-6f);

    for (uint kd = 0; kd < k_head_dim; ++kd) {
        q[kd] *= qinv;
        k[kd] *= kinv;
    }

    const float A =
        qtk_f32(blob, A_log_off, h);

    const float dt =
        qtk_f32(blob, dt_bias_off, h);

    const float softplus =
        log(1.0f + exp(a[h] + dt));

    const float gamma =
        exp(-exp(A) * softplus);

    const float beta =
        qtk_sigmoid(b[h]);

    for (uint vd = 0; vd < v_head_dim; ++vd) {
        float kv_mem = 0.0f;

        /*
         * Apply decay before reading the old memory contribution.
         */
        for (uint kd = 0; kd < k_head_dim; ++kd) {
            const ulong si =
                (ulong)kd * (ulong)v_head_dim + vd;

            const float decayed =
                Sh[si] * gamma;

            Sh[si] = decayed;
            kv_mem += decayed * k[kd];
        }

        const float delta =
            (v[vd] - kv_mem) * beta;

        float out = 0.0f;

        /*
         * Update S, then consume the updated state for q.
         */
        for (uint kd = 0; kd < k_head_dim; ++kd) {
            const ulong si =
                (ulong)kd * (ulong)v_head_dim + vd;

            const float updated =
                Sh[si] + k[kd] * delta;

            Sh[si] = updated;
            out += updated * q[kd];
        }

        yh[vd] = out;
    }
}

/*
 * Per-value-head GDN output norm + z gate.
 *
 * gdn_norm is shared across heads over v_head_dim, matching the CPU
 * RMSNormGated shape.
 */
static inline void
qtk_gdn_output_norm(device float *y,
                    device const float *z,
                    device const uchar *blob,
                    ulong norm_off,
                    uint v_heads,
                    uint v_head_dim,
                    float eps,
                    uint tid)
{
    if (tid < v_heads) {
        const uint base = tid * v_head_dim;

        float ss = 0.0f;

        for (uint d = 0; d < v_head_dim; ++d) {
            const float x = y[base + d];
            ss += x * x;
        }

        const float inv =
            rsqrt(ss / (float)v_head_dim + eps);

        for (uint d = 0; d < v_head_dim; ++d) {
            const uint i = base + d;

            y[i] =
                y[i] *
                inv *
                qtk_f32(blob, norm_off, d) *
                qtk_silu(z[i]);
        }
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

static inline void
qtk_gdn_layer(device const uchar *blob,
              constant QwenTokenKernelParams &p,
              QwenTokenLayerBlob L,
              QwenTokenDeviceLayout dl,
              device uchar *state,
              device const float *normed,
              device float *attn,
              uint layer,
              uint tid)
{
    const uint hidden =
        p.hidden;

    const uint kdim =
        p.lin_k_heads * p.lin_k_dim;

    const uint vdim =
        p.lin_v_heads * p.lin_v_dim;

    const uint qkv_dim =
        2u * kdim + vdim;

    device float *w0 =
        qtk_state_f32(state, dl.work_off[0]);

    device float *w1 =
        qtk_state_f32(state, dl.work_off[1]);

    device float *w2 =
        qtk_state_f32(state, dl.work_off[2]);

    device float *w3 =
        qtk_state_f32(state, dl.work_off[3]);

    /*
     * qkv, z, a, b must all be projected before normed is reused.
     */
    qtk_bf16_matvec(blob,
                    L.gdn_in_qkv,
                    normed,
                    w0,
                    qkv_dim,
                    hidden,
                    tid);

    qtk_bf16_matvec(blob,
                    L.gdn_in_z,
                    normed,
                    w1,
                    vdim,
                    hidden,
                    tid);

    /*
     * a and b are one scalar per value head.
     * Pack them into w2 as:
     *     [a[0..H-1], b[0..H-1]]
     */
    qtk_bf16_matvec(blob,
                    L.gdn_in_a,
                    normed,
                    w2,
                    p.lin_v_heads,
                    hidden,
                    tid);

    qtk_bf16_matvec(blob,
                    L.gdn_in_b,
                    normed,
                    w2 + p.lin_v_heads,
                    p.lin_v_heads,
                    hidden,
                    tid);

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    device float *conv_state =
        qtk_state_f32(state,
                      dl.gdn_conv_layer_off[layer]);

    qtk_gdn_conv(w0,
                 conv_state,
                 blob,
                 L.gdn_conv1d,
                 qkv_dim,
                 p.conv_kernel,
                 tid);

    device float *S =
        qtk_state_f32(state,
                      dl.gdn_s_layer_off[layer]);

    if (tid < p.lin_v_heads) {
        qtk_gdn_recurrence(w0,
                           w2,
                           w2 + p.lin_v_heads,
                           S,
                           w3,
                           blob,
                           L.gdn_A_log,
                           L.gdn_dt_bias,
                           p.lin_k_heads,
                           p.lin_k_dim,
                           p.lin_v_heads,
                           p.lin_v_dim,
                           tid);
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    qtk_gdn_output_norm(w3,
                        w1,
                        blob,
                        L.gdn_norm,
                        p.lin_v_heads,
                        p.lin_v_dim,
                        p.eps,
                        tid);

    qtk_bf16_matvec(blob,
                    L.gdn_out,
                    w3,
                    attn,
                    hidden,
                    vdim,
                    tid);

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * Persist the current token's K/V before the attention scan.
 *
 * KV memory layout:
 *     [token][kv_head][head_dim]
 */
static inline void
qtk_attn_store_kv(device float *kv_k,
                  device float *kv_v,
                  device const float *k,
                  device const float *v,
                  uint token_pos,
                  uint n_kv_heads,
                  uint head_dim,
                  uint tid)
{
    const uint width =
        n_kv_heads * head_dim;

    const ulong base =
        (ulong)token_pos * (ulong)width;

    for (uint i = tid; i < width; i += QTK_THREADS) {
        kv_k[base + i] = k[i];
        kv_v[base + i] = v[i];
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * Full causal attention.
 *
 * One thread owns one complete query head.  Scores are recomputed on the
 * second scan rather than storing a logits array, which keeps the kernel
 * bounded to the existing device-state layout.
 *
 * Pass 1: maximum score.
 * Pass 2: denominator + weighted V.
 *
 * GQA mapping is integer head grouping.  For the target geometry it reduces
 * to q_head == kv_head.
 */
static inline void
qtk_attention_scan(device const float *q,
                   device const float *kv_k,
                   device const float *kv_v,
                   device float *ctx,
                   uint token_pos,
                   uint n_heads,
                   uint n_kv_heads,
                   uint head_dim,
                   uint tid)
{
    if (tid >= n_heads)
        return;

    const uint h = tid;

    const uint kvh =
        (h * n_kv_heads) / n_heads;

    const uint qbase =
        h * head_dim;

    const uint kv_width =
        n_kv_heads * head_dim;

    const float scale =
        rsqrt((float)head_dim);

    float max_score = -INFINITY;

    /*
     * Exact causal scan, oldest token to newest.
     */
    for (uint t = 0; t <= token_pos; ++t) {
        const ulong kb =
            (ulong)t * (ulong)kv_width +
            (ulong)kvh * (ulong)head_dim;

        float dot = 0.0f;

        for (uint d = 0; d < head_dim; ++d)
            dot += q[qbase + d] * kv_k[kb + d];

        const float score = dot * scale;
        max_score = max(max_score, score);
    }

    const uint cbase =
        h * head_dim;

    for (uint d = 0; d < head_dim; ++d)
        ctx[cbase + d] = 0.0f;

    float denom = 0.0f;

    for (uint t = 0; t <= token_pos; ++t) {
        const ulong kb =
            (ulong)t * (ulong)kv_width +
            (ulong)kvh * (ulong)head_dim;

        float dot = 0.0f;

        for (uint d = 0; d < head_dim; ++d)
            dot += q[qbase + d] * kv_k[kb + d];

        const float e =
            exp(dot * scale - max_score);

        denom += e;

        for (uint d = 0; d < head_dim; ++d)
            ctx[cbase + d] += e * kv_v[kb + d];
    }

    const float inv_denom =
        1.0f / denom;

    for (uint d = 0; d < head_dim; ++d)
        ctx[cbase + d] *= inv_denom;
}

static inline void
qtk_attention_layer(device const uchar *blob,
                    constant QwenTokenKernelParams &p,
                    QwenTokenLayerBlob L,
                    QwenTokenDeviceLayout dl,
                    device uchar *state,
                    device const float *normed,
                    device float *attn,
                    uint layer,
                    uint token_pos,
                    uint tid)
{
    const uint qdim =
        p.n_heads * p.head_dim;

    const uint kvdim =
        p.n_kv_heads * p.head_dim;

    device float *q =
        qtk_state_f32(state, dl.work_off[0]);

    device float *k =
        qtk_state_f32(state, dl.work_off[1]);

    device float *v =
        qtk_state_f32(state, dl.work_off[2]);

    device float *ctx =
        qtk_state_f32(state, dl.work_off[3]);

    qtk_bf16_matvec(blob,
                    L.attn_q,
                    normed,
                    q,
                    qdim,
                    p.hidden,
                    tid);

    qtk_bf16_matvec(blob,
                    L.attn_k,
                    normed,
                    k,
                    kvdim,
                    p.hidden,
                    tid);

    qtk_bf16_matvec(blob,
                    L.attn_v,
                    normed,
                    v,
                    kvdim,
                    p.hidden,
                    tid);

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    qtk_attn_norm_rope(q,
                       k,
                       blob,
                       L.attn_qn,
                       L.attn_kn,
                       p.n_heads,
                       p.n_kv_heads,
                       p.head_dim,
                       p.rotary_dim,
                       token_pos,
                       p.theta,
                       p.eps,
                       tid);

    device float *kv_k =
        qtk_state_f32(state,
                      dl.kv_k_off[layer]);

    device float *kv_v =
        qtk_state_f32(state,
                      dl.kv_v_off[layer]);

    qtk_attn_store_kv(kv_k,
                      kv_v,
                      k,
                      v,
                      token_pos,
                      p.n_kv_heads,
                      p.head_dim,
                      tid);

    if (tid < p.n_heads) {
        qtk_attention_scan(q,
                           kv_k,
                           kv_v,
                           ctx,
                           token_pos,
                           p.n_heads,
                           p.n_kv_heads,
                           p.head_dim,
                           tid);
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    qtk_bf16_matvec(blob,
                    L.attn_o,
                    ctx,
                    attn,
                    p.hidden,
                    qdim,
                    tid);

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * Stage-4 pre-MoE layer.
 *
 * On return:
 *
 *     residual = residual_before_layer + GDN/attention
 *     normed   = postnorm(residual)       <-- exact MoE input
 *
 * The router/MoE half must consume normed and add its result to residual
 * before the next layer begins.
 */
static inline void
qtk_pre_moe_layer(device const uchar *blob,
                  constant QwenTokenKernelParams &p,
                  QwenTokenLayerBlob L,
                  QwenTokenDeviceLayout dl,
                  device uchar *state,
                  uint layer,
                  uint token_pos,
                  uint tid,
                  threadgroup float *tg_scalar)
{
    device float *residual =
        qtk_state_f32(state, dl.residual_off);

    device float *normed =
        qtk_state_f32(state, dl.normed_off);

    /*
     * attn is allowed to reuse work0 only after every projection that reads
     * normed has finished.  GDN/attention helpers therefore write their final
     * hidden-vector output directly to normed: at that point the input
     * normalization is dead.
     */
    qtk_rmsnorm(residual,
                blob,
                L.in_ln,
                normed,
                p.hidden,
                p.eps,
                tid,
                tg_scalar);

    if (L.kind == 1u) {
        qtk_gdn_layer(blob,
                      p,
                      L,
                      dl,
                      state,
                      normed,
                      normed,
                      layer,
                      tid);
    } else {
        qtk_attention_layer(blob,
                            p,
                            L,
                            dl,
                            state,
                            normed,
                            normed,
                            layer,
                            token_pos,
                            tid);
    }

    /*
     * residual += attention_or_gdn_output
     */
    qtk_residual_add(residual,
                     normed,
                     p.hidden,
                     tid);

    /*
     * normed now becomes the persistent device-resident MoE input.
     */
    qtk_rmsnorm(residual,
                blob,
                L.post_ln,
                normed,
                p.hidden,
                p.eps,
                tid,
                tg_scalar);
}

/*
 * Final Stage-4 kernel entry.
 *
 * buffers:
 *   0 = immutable packed weight blob
 *   1 = immutable kernel params / layer blob offsets
 *   2 = mutable device state
 *   3 = token position
 *
 * Exactly one 1024-thread threadgroup is dispatched for one token.
 */
kernel void
qwen_token_whole(device const uchar *weights       [[buffer(0)]],
                 constant QwenTokenKernelParams *p [[buffer(1)]],
                 device uchar *state               [[buffer(2)]],
                 constant uint &token_pos           [[buffer(3)]],
                 constant QwenTokenDeviceLayout *dl_param [[buffer(4)]],
                 uint tid [[thread_position_in_threadgroup]],
                 uint threads [[threads_per_threadgroup]],
                 uint tgpos [[threadgroup_position_in_grid]])
{
    /*
     * Stage-4 is deliberately one threadgroup.  Do not turn this into one
     * threadgroup/layer: recurrent state, residual, and future MoE state are
     * ordered through this single resident invocation.
     */
    if (threads != QTK_THREADS || tgpos != 0u)
        return;

    /*
     * Use the existing layout member/accessor here.  Do not duplicate the ABI.
     */
    const QwenTokenDeviceLayout dl = *dl_param;

    threadgroup float tg_scalar[1];

    for (uint layer = 0u; layer < p->n_layers; ++layer) {
        const QwenTokenLayerBlob L =
            p->layer[layer];

        qtk_pre_moe_layer(weights,
                          *p,
                          L,
                          dl,
                          state,
                          layer,
                          token_pos,
                          tid,
                          tg_scalar);

        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        /*
         * ================================================================
         * STAGE-4 ROUTER / MoE SPLICE POINT
         * ================================================================
         *
         * Current state here:
         *
         *   state + dl.residual_off
         *       = residual after GDN/full-attention
         *
         *   state + dl.normed_off
         *       = post-LN vector, i.e. this layer's exact MoE input
         *
         *   GDN conv/S or full-attention KV
         *       = already committed for token_pos
         *
         * The next Stage-4 relay inserts:
         *
         *   router
         *   -> NEED_EXPERTS miss/resume handling
         *   -> routed experts
         *   -> shared expert
         *   -> residual += MoE output
         *
         * ONLY AFTER that update may this loop advance to layer+1.
         *
         * Do not insert a fake residual update here and do not allow this
         * partial source to run end-to-end before the MoE splice lands.
         * ================================================================
         */

        /* STAGE4_MOE_SPLICE */
    }
}
)METAL";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { if (err&&err_cap) snprintf(err,(size_t)err_cap,"no Metal device"); return 0; }
    NSError *e = nil;
    NSString *src = [NSString stringWithUTF8String:QTK_MSL];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&e];
    if (!lib) {
        if (err && err_cap && e && e.localizedDescription)
            snprintf(err,(size_t)err_cap,"%s", e.localizedDescription.UTF8String);
        else if (err && err_cap) snprintf(err,(size_t)err_cap,"MSL compile failed");
        return 0;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"qwen_token_whole"];
    if (!fn) { if (err&&err_cap) snprintf(err,(size_t)err_cap,"kernel qwen_token_whole not found"); return 0; }
    return 1;
}


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

#define QTK_THREADS 256u

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
        y[i] = x[i] * (1.0f + qtk_f32(blob, w_off, i)) * inv;

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
        /* q buffer packs [q | gate] per head: 2*head_dim per head. */
        const uint base = tid * 2u * head_dim;

        float ss = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            const float v = q[base + d];
            ss += v * v;
        }

        const float inv =
            rsqrt(ss / (float)head_dim + eps);

        for (uint d = 0; d < head_dim; ++d)
            q[base + d] =
                q[base + d] * (1.0f + qtk_f32(blob, qn_off, d)) * inv;

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
                k[base + d] * (1.0f + qtk_f32(blob, kn_off, d)) * inv;

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
     * GQA mapping: rep = v_heads / k_heads value heads share one key head
     * (real model: 32/16 = 2).  kh = h / rep, exactly like the CPU
     * gdn_token (int khd = h / rep).
     */
    const uint rep = v_heads / k_heads;
    const uint kh = h / rep;

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
    const float qscale = rsqrt((float)k_head_dim);

    /*
     * q/k are SHARED between rep value heads and MUST NOT be mutated in
     * place (rep=2 heads would double-apply the norm and race).  The norms
     * are applied at every use site below instead.
     */

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
            kv_mem += decayed * (k[kd] * kinv);
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
                Sh[si] + (k[kd] * kinv) * delta;

            Sh[si] = updated;
            out += updated * ((q[kd] * qinv) * qscale);
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
        h * 2u * head_dim;   /* [q | gate] packing */

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

    const uint qgdim =
        2u * p.n_heads * p.head_dim;   /* [q | gate] per head */

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

    /* attn_q packs [q | gate] per head (2*H*hd rows), matching the CPU. */
    qtk_bf16_matvec(blob,
                    L.attn_q,
                    normed,
                    q,
                    qgdim,
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

    /* Output gate: attn * sigmoid(gate) elementwise (gate = 2nd half of the
     * q|gate block), matching the CPU attention_token. */
    for (uint i = tid; i < qdim; i += QTK_THREADS) {
        const uint h = i / p.head_dim;
        const uint d = i % p.head_dim;
        const float g = q[(2u * h + 1u) * p.head_dim + d];
        ctx[i] *= qtk_sigmoid(g);
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
/* ========================================================================== */
/* Stage 5: resident MoE + miss/resume                                         */
/* ========================================================================== */

static constant uint QTK_STATUS_OK             = 1u; /* = QWEN_TOKEN_STATUS_DONE */
static constant uint QTK_STATUS_NEED_EXPERTS   = 2u; /* = QWEN_TOKEN_STATUS_NEED_EXPERTS */

static constant uint QTK_RESUME_NONE           = 0u;
static constant uint QTK_RESUME_MOE            = 1u;

static constant uint QTK_NO_EXPERT_SLOT        = 0xffffffffu;

/*
 * Canonical E2M1 values.  Apple8 is only a physical 8x32 tiling of the
 * canonical MXFP4 values/scales:
 *
 *   tile[0..127]   = 8 rows x 16 packed nibble bytes
 *   tile[128..135] = one E8M0 scale byte per output row
 *
 * Low nibble is the earlier/even K element.
 */
static constant float APPLE8_MX4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,
     2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f,
    -2.0f, -3.0f, -4.0f, -6.0f
};

static inline device QwenTokenMissRecord *
qtk_miss_record(device uchar *state,
                QwenTokenDeviceLayout dl)
{
    return reinterpret_cast<device QwenTokenMissRecord *>(
        state + dl.miss_off);
}

static inline device float *
qtk_moe_input(device uchar *state,
              QwenTokenDeviceLayout dl)
{
    return qtk_state_f32(state, dl.moe_input_off);
}

static inline void
qtk_copy_f32(device const float *src,
             device float *dst,
             uint n,
             uint tid)
{
    for (uint i = tid; i < n; i += QTK_THREADS)
        dst[i] = src[i];

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * E8M0 scale.  Canonical MX convention:
 *
 *     scale = 2^(e - 127)
 *
 * No E8M0 zero/NaN scale is emitted by the Apple8 packer for real rows.
 */
static inline float
qtk_e8m0(uchar e)
{
    return exp2((float)((int)e - 127));
}

/*
 * Read one value from an Apple8 8x32 / 136-byte tiled matrix.
 *
 * Matrix is logically [rows, cols], row-major.  Physical tile order:
 *
 *     output_tile-major, then K-group-major
 *
 * Edge tiles are zero padded by the packer.
 */
static inline float
qtk_apple8_get(device const uchar *matrix,
               uint row,
               uint col,
               uint cols)
{
    const uint ng =
        (cols + 31u) >> 5;

    const uint otile =
        row >> 3;

    const uint orow =
        row & 7u;

    const uint g =
        col >> 5;

    const uint k =
        col & 31u;

    const ulong tile_index =
        (ulong)otile * (ulong)ng + (ulong)g;

    device const uchar *tile =
        matrix + tile_index * 136ul;

    const uchar packed =
        tile[orow * 16u + (k >> 1)];

    const uint code =
        ((k & 1u) == 0u)
            ? ((uint)packed & 0x0fu)
            : ((uint)packed >> 4);

    const float scale =
        qtk_e8m0(tile[128u + orow]);

    return APPLE8_MX4[code] * scale;
}

/*
 * Deterministic Apple8 matvec.
 *
 * One thread owns an output row.  Its dot product walks input columns in
 * strictly ascending order.
 */
static inline void
qtk_apple8_matvec(device const uchar *matrix,
                  device const float *x,
                  device float *dst,
                  uint rows,
                  uint cols,
                  uint tid)
{
    for (uint r = tid; r < rows; r += QTK_THREADS) {
        float s = 0.0f;

        for (uint c = 0u; c < cols; ++c)
            s += qtk_apple8_get(matrix, r, c, cols) * x[c];

        dst[r] = s;
    }
}

/*
 * Router:
 *
 *   logits[e] = router[e,:] . moe_input
 *   p[e]      = softmax(logits)[e]
 *
 * Top-k is selected from the full softmax probabilities.  The selected
 * probabilities are NOT renormalized.
 *
 * Tie behaviour is deterministic: strict '>' and ascending expert scan means
 * the lower expert id wins exact ties.
 */
static inline void
qtk_router(device const uchar *blob,
           QwenTokenKernelParams p,
           QwenTokenLayerBlob L,
           device const float *moe_input,
           device float *router_scratch,
           device QwenTokenMissRecord *miss,
           uint layer,
           uint tid)
{
    qtk_bf16_matvec(blob,
                    L.router,
                    moe_input,
                    router_scratch,
                    p.n_experts,
                    p.hidden,
                    tid);

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    if (tid == 0u) {
        float mx = -INFINITY;

        for (uint e = 0u; e < p.n_experts; ++e)
            mx = max(mx, router_scratch[e]);

        float sum = 0.0f;

        /*
         * Convert logits to unnormalised probabilities in-place.
         * Reduction order is ascending expert id.
         */
        for (uint e = 0u; e < p.n_experts; ++e) {
            const float v =
                exp(router_scratch[e] - mx);

            router_scratch[e] = v;
            sum += v;
        }

        const float inv_sum =
            1.0f / sum;

        for (uint e = 0u; e < p.n_experts; ++e)
            router_scratch[e] *= inv_sum;

        /*
         * Repeated fixed-order argmax.  No mutation of probabilities, so
         * routed_weight is the original full-softmax probability.
         */
        for (uint r = 0u; r < p.topk; ++r) {
            uint best_id = QTK_NO_EXPERT_SLOT;
            float best_p = -INFINITY;

            for (uint e = 0u; e < p.n_experts; ++e) {
                bool already_selected = false;

                for (uint j = 0u; j < r; ++j) {
                    if (miss->routed_expert[j] == e)
                        already_selected = true;
                }

                const float pe =
                    router_scratch[e];

                if (!already_selected &&
                    (best_id == QTK_NO_EXPERT_SLOT || pe > best_p)) {
                    best_id = e;
                    best_p = pe;
                }
            }

            miss->routed_expert[r] = best_id;
            miss->routed_weight[r] = best_p;
        }

        /* Renormalize top-k weights over the selected set (CPU parity). */
        {
            float wsum = 0.0f;
            for (uint r = 0u; r < p.topk; ++r)
                wsum += miss->routed_weight[r];
            const float inv_wsum = 1.0f / wsum;
            for (uint r = 0u; r < p.topk; ++r)
                miss->routed_weight[r] *= inv_wsum;
        }

        /*
         * Persist the exact layer associated with this route before any
         * possible NEED_EXPERTS return.
         */
        miss->layer = layer;
        miss->missing_count = 0u;
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * Each layer map is an array indexed by bank slot:
 *
 *     map[slot] = resident expert id
 *
 * Thus the index of a matching map entry is the bank slot.
 *
 * Deliberately scans the full bounded map; no early break.
 */
static inline uint
qtk_find_expert_slot(device const uint *map,
                     uint map_slots,
                     uint expert_id)
{
    uint found = QTK_NO_EXPERT_SLOT;

    for (uint slot = 0u; slot < map_slots; ++slot) {
        if (map[slot] == expert_id &&
            found == QTK_NO_EXPERT_SLOT) {
            found = slot;
        }
    }

    return found;
}

/*
 * Validate the complete routed set before launching ANY MoE arithmetic.
 *
 * Missing ids are recorded in routed-rank order.
 *
 * tg_u32[1] is the uniform "must return" flag consumed by the kernel.
 */
static inline void
qtk_check_expert_residency(QwenTokenKernelParams p,
                           QwenTokenDeviceLayout dl,
                           device uchar *state,
                           device QwenTokenMissRecord *miss,
                           uint layer,
                           uint tid,
                           threadgroup uint *tg_u32)
{
    if (tid == 0u) {
        device const uint *map =
            reinterpret_cast<device const uint *>(
                state + dl.expert_map_layer_off[layer]);

        const uint map_slots =
            dl.expert_map_slots[layer];

        uint nmissing = 0u;

        for (uint r = 0u; r < p.topk; ++r) {
            const uint expert_id =
                miss->routed_expert[r];

            const uint slot =
                qtk_find_expert_slot(map,
                                     map_slots,
                                     expert_id);

            if (slot == QTK_NO_EXPERT_SLOT) {
                miss->missing_expert[nmissing] =
                    expert_id;

                ++nmissing;
            }
        }

        miss->missing_count = nmissing;

        if (nmissing != 0u) {
            /*
             * Publish all resume information before the uniform kernel exit.
             */
            miss->layer = layer;
            miss->resume_phase = QTK_RESUME_MOE;
            miss->status = QTK_STATUS_NEED_EXPERTS;

            tg_u32[1] = 1u;
        } else {
            tg_u32[1] = 0u;
        }
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * Routed experts.
 *
 * w0: gate / then SwiGLU intermediate
 * w1: up
 * w2: current expert down output
 * acc: deterministic sum over routed experts
 *
 * Experts execute sequentially in routed-rank order.  Therefore each hidden
 * output's accumulation order is exactly route[0], route[1], ... topk-1.
 */
static inline void
qtk_routed_experts(device const uchar *blob,
                   QwenTokenKernelParams p,
                   QwenTokenLayerBlob L,
                   QwenTokenDeviceLayout dl,
                   device uchar *state,
                   device const float *moe_input,
                   device QwenTokenMissRecord *miss,
                   uint layer,
                   device float *w0,
                   device float *w1,
                   device float *w2,
                   device float *acc,
                   uint tid,
                   threadgroup uint *tg_u32)
{
    for (uint d = tid; d < p.hidden; d += QTK_THREADS)
        acc[d] = 0.0f;

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    device const uint *map =
        reinterpret_cast<device const uint *>(
            state + dl.expert_map_layer_off[layer]);

    const uint map_slots =
        dl.expert_map_slots[layer];

    device const uchar *bank =
        blob + L.expert_bank;

    for (uint r = 0u; r < p.topk; ++r) {
        /*
         * Resolve once per expert and broadcast the slot through TG memory.
         * Residency was already checked, so QTK_NO_EXPERT_SLOT is impossible
         * unless the host violates the publish-before-dispatch contract.
         */
        if (tid == 0u) {
            tg_u32[0] =
                qtk_find_expert_slot(map,
                                     map_slots,
                                     miss->routed_expert[r]);
        }

        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        const uint bank_slot =
            tg_u32[0];

        device const uchar *slot_base =
            bank +
            (ulong)bank_slot *
            (ulong)L.expert_slot_stride;

        device const uchar *gate_w =
            slot_base + p.expert_gate_rel;

        device const uchar *up_w =
            slot_base + p.expert_up_rel;

        device const uchar *down_w =
            slot_base + p.expert_down_rel;

        /*
         * [moe_inter, hidden]
         */
        qtk_apple8_matvec(gate_w,
                          moe_input,
                          w0,
                          p.moe_inter,
                          p.hidden,
                          tid);

        qtk_apple8_matvec(up_w,
                          moe_input,
                          w1,
                          p.moe_inter,
                          p.hidden,
                          tid);

        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        /*
         * SwiGLU expert activation.
         */
        for (uint j = tid;
             j < p.moe_inter;
             j += QTK_THREADS) {
            w0[j] =
                qtk_silu(w0[j]) * w1[j];
        }

        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        /*
         * [hidden, moe_inter]
         */
        qtk_apple8_matvec(down_w,
                          w0,
                          w2,
                          p.hidden,
                          p.moe_inter,
                          tid);

        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        const float routed_weight =
            miss->routed_weight[r];

        /*
         * Crucially: this is the full-softmax probability.  There is no
         * top-k renormalisation here.
         */
        for (uint d = tid;
             d < p.hidden;
             d += QTK_THREADS) {
            acc[d] +=
                routed_weight * w2[d];
        }

        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    }
}

/*
 * Shared expert:
 *
 *   h = silu(se_gate*x) * (se_up*x)
 *   sy = se_down*h
 *   g = sigmoid(se_g*x)
 *   sy *= g
 *
 * se_g is a BF16 [1, hidden] row.
 */
static inline void
qtk_shared_expert(device const uchar *blob,
                  QwenTokenKernelParams p,
                  QwenTokenLayerBlob L,
                  device const float *moe_input,
                  device float *w0,
                  device float *w1,
                  device float *sy,
                  uint tid,
                  threadgroup float *tg_scalar)
{
    qtk_bf16_matvec(blob,
                    L.se_gate,
                    moe_input,
                    w0,
                    p.shared_inter,
                    p.hidden,
                    tid);

    qtk_bf16_matvec(blob,
                    L.se_up,
                    moe_input,
                    w1,
                    p.shared_inter,
                    p.hidden,
                    tid);

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    for (uint j = tid;
         j < p.shared_inter;
         j += QTK_THREADS) {
        w0[j] =
            qtk_silu(w0[j]) * w1[j];
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    qtk_bf16_matvec(blob,
                    L.se_down,
                    w0,
                    sy,
                    p.hidden,
                    p.shared_inter,
                    tid);

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    /*
     * Keep this scalar projection completely serial so its FP order matches
     * the deterministic CPU definition.
     */
    if (tid == 0u) {
        float g = 0.0f;

        for (uint d = 0u; d < p.hidden; ++d)
            g += qtk_bf16(blob, L.se_g, d) *
                 moe_input[d];

        tg_scalar[0] =
            qtk_sigmoid(g);
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    const float sg =
        tg_scalar[0];

    for (uint d = tid;
         d < p.hidden;
         d += QTK_THREADS) {
        sy[d] *= sg;
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

/*
 * Complete post-router MoE half.
 *
 * Layout of the four existing Stage-4 work buffers:
 *
 *   w0 = expert/shared gate activation
 *   w1 = expert/shared up activation
 *   w2 = current expert down / final shared sy
 *   w3 = routed MoE accumulator
 */
static inline void
qtk_moe_execute(device const uchar *blob,
                QwenTokenKernelParams p,
                QwenTokenLayerBlob L,
                QwenTokenDeviceLayout dl,
                device uchar *state,
                device const float *moe_input,
                device QwenTokenMissRecord *miss,
                uint layer,
                uint tid,
                threadgroup float *tg_scalar,
                threadgroup uint *tg_u32)
{
    device float *w0 =
        qtk_state_f32(state, dl.work_off[0]);

    device float *w1 =
        qtk_state_f32(state, dl.work_off[1]);

    device float *w2 =
        qtk_state_f32(state, dl.work_off[2]);

    device float *moe_acc =
        qtk_state_f32(state, dl.work_off[3]);

    qtk_routed_experts(blob,
                       p,
                       L,
                       dl,
                       state,
                       moe_input,
                       miss,
                       layer,
                       w0,
                       w1,
                       w2,
                       moe_acc,
                       tid,
                       tg_u32);

    /*
     * w0/w1/w2 are now free.  moe_acc in work3 must survive shared-expert
     * execution.
     */
    qtk_shared_expert(blob,
                      p,
                      L,
                      moe_input,
                      w0,
                      w1,
                      w2,
                      tid,
                      tg_scalar);

    device float *residual =
        qtk_state_f32(state, dl.residual_off);

    /*
     * Exact model order:
     *
     *     residual += routed_moe + shared_expert
     */
    for (uint d = tid;
         d < p.hidden;
         d += QTK_THREADS) {
        residual[d] +=
            moe_acc[d] + w2[d];
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
}

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

        device QwenTokenMissRecord *miss =
        qtk_miss_record(state, dl);

    /*
     * Additional TG state used by Stage 5:
     *
     *   [0] current resolved expert-bank slot
     *   [1] uniform NEED_EXPERTS return flag
     */
    threadgroup uint tg_u32[2];

    /*
     * A re-dispatch after NEED_EXPERTS begins at the exact layer that missed.
     * resume_phase==MOE is the authority: the GDN/attention half for that
     * layer has already committed recurrent/KV state and MUST NOT execute
     * again.
     */
    uint first_layer = 0u;
    bool resume_moe = false;

    if (miss->resume_phase == QTK_RESUME_MOE) {
        first_layer = miss->layer;
        resume_moe = true;
    }

    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    for (uint layer = first_layer;
         layer < p->n_layers;
         ++layer) {
        const QwenTokenLayerBlob L =
            p->layer[layer];

        const bool this_is_moe_resume =
            resume_moe && (layer == first_layer);

        device float *moe_input =
            qtk_moe_input(state, dl);

        if (!this_is_moe_resume) {
            /*
             * Fresh layer:
             *
             *   RMS in
             *   GDN or full attention
             *   residual += attn
             *   postnorm -> dl.normed_off
             */
            qtk_pre_moe_layer(weights,
                              *p,
                              L,
                              dl,
                              state,
                              layer,
                              token_pos,
                              tid,
                              tg_scalar);

            /*
             * The postnorm vector must survive a NEED_EXPERTS return.
             * Persist it BEFORE routing/residency checking.
             */
            device const float *normed =
                qtk_state_cf32(state,
                               dl.normed_off);

            qtk_copy_f32(normed,
                         moe_input,
                         p->hidden,
                         tid);

            /*
             * Route exactly once.  The selected ids and original softmax
             * probabilities are persisted in QwenTokenMissRecord.
             *
             * work0 is only router scratch here.  It is free once top-k has
             * been persisted.
             */
            device float *router_scratch =
                qtk_state_f32(state,
                              dl.work_off[0]);

            qtk_router(weights,
                       *p,
                       L,
                       moe_input,
                       router_scratch,
                       miss,
                       layer,
                       tid);
        } else {
            /*
             * Resume path:
             *
             *   - DO NOT rerun in-norm
             *   - DO NOT rerun GDN recurrence
             *   - DO NOT rewrite attention KV
             *   - DO NOT residual-add attention again
             *   - DO NOT rerun postnorm
             *   - DO NOT rerun router/top-k
             *
             * moe_input + routed ids/weights are the persisted values from
             * the miss-producing dispatch.
             */
            threadgroup_barrier(
                mem_flags::mem_device_and_threadgroup);
        }

        /*
         * Both fresh and resumed paths converge here.
         *
         * Check the COMPLETE routed set before doing any routed/shared MoE
         * compute.  If even one selected expert is absent, publish all missing
         * ids and exit the entire threadgroup uniformly.
         */
        qtk_check_expert_residency(*p,
                                   dl,
                                   state,
                                   miss,
                                   layer,
                                   tid,
                                   tg_u32);

        if (tg_u32[1] != 0u) {
            /*
             * Uniform condition written by tid 0 and observed only after a
             * threadgroup/device barrier.  Safe whole-kernel return.
             *
             * Host contract:
             *   status       = NEED_EXPERTS
             *   resume_phase = MOE
             *   layer        = current layer
             *   moe_input    = persisted
             *   route        = persisted
             */
            return;
        }

        /*
         * Every routed expert is resident.  Execute routed experts in fixed
         * top-k order, execute shared expert, then update residual.
         */
        qtk_moe_execute(weights,
                        *p,
                        L,
                        dl,
                        state,
                        moe_input,
                        miss,
                        layer,
                        tid,
                        tg_scalar,
                        tg_u32);

        /*
         * Only AFTER the residual update is committed is the resume record
         * cleared.  From this point layer+1 may run normally.
         */
        if (tid == 0u) {
            miss->missing_count = 0u;
            miss->resume_phase = QTK_RESUME_NONE;
            miss->status = QTK_STATUS_OK;
        }

        threadgroup_barrier(
            mem_flags::mem_device_and_threadgroup);

        /*
         * A resumed layer has now fully caught up with the normal forward
         * path.  Every later layer in this dispatch is fresh.
         */
        resume_moe = false;
    }
}
)METAL";

/* ---- Whole-token dispatch (host seam) ---- */
static id<MTLComputePipelineState>
qtk_whole_pipeline(id<MTLDevice> dev, char *why, size_t why_n)
{
    static id<MTLDevice> cached_dev = nil;
    static id<MTLComputePipelineState> cached_pso = nil;

    if (!dev) {
        if (why && why_n) snprintf(why, why_n, "no Metal device");
        return nil;
    }

    if (cached_pso && cached_dev == dev)
        return cached_pso;

    NSError *err = nil;

    NSString *src = [NSString stringWithUTF8String:QTK_MSL];
    if (!src) {
        if (why && why_n) snprintf(why, why_n, "missing qwen token MSL source");
        return nil;
    }

    id<MTLLibrary> lib =
        [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        if (why && why_n)
            snprintf(why, why_n, "MSL compile: %s",
                     err ? [[err localizedDescription] UTF8String]
                         : "unknown error");
        return nil;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"qwen_token_whole"];
    if (!fn) {
        if (why && why_n)
            snprintf(why, why_n, "qwen_token_whole not found");
        return nil;
    }

    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        if (why && why_n)
            snprintf(why, why_n, "pipeline compile: %s",
                     err ? [[err localizedDescription] UTF8String]
                         : "unknown error");
        return nil;
    }

    cached_dev = dev;
    cached_pso = pso;
    return cached_pso;
}

/*
 * Cached constant buffers for params + layout.
 * setBytes() caps at 4KB; QwenTokenKernelParams (~13KB) and
 * QwenTokenDeviceLayout (~12KB) exceed it, so they must ride real MTLBuffers.
 */
static id<MTLBuffer>
qtk_const_buffer(id<MTLDevice> dev, const void *bytes, size_t n,
                 const void **cache_key, id<MTLBuffer> *cache_buf)
{
    if (*cache_buf && *cache_key == bytes)
        return *cache_buf;

    id<MTLBuffer> b =
        [dev newBufferWithBytes:bytes
                        length:n
                       options:MTLResourceStorageModeShared];
    if (b) {
        *cache_key = bytes;
        *cache_buf = b;
    }
    return b;
}

extern "C" int
qwen_token_kernel_dispatch_once(
    QwenTokenWeightBlob *blob,
    QwenTokenDeviceState *state,
    const QwenTokenKernelParams *params,
    int token_pos,
    char *why,
    size_t why_n)
{
    @autoreleasepool {
        if (why && why_n) why[0] = '\0';

        if (!blob || !state || !params) {
            if (why && why_n) snprintf(why, why_n, "null blob/state/params");
            return 0;
        }

        id<MTLBuffer> blob_buf  = blob->buffer;
        id<MTLBuffer> state_buf = state->buffer;

        if (!blob_buf || !state_buf) {
            if (why && why_n) snprintf(why, why_n, "missing Metal buffer");
            return 0;
        }

        id<MTLDevice> dev = [state_buf device];
        if (!dev || [blob_buf device] != dev) {
            if (why && why_n) snprintf(why, why_n, "blob/state device mismatch");
            return 0;
        }

        id<MTLComputePipelineState> pso =
            qtk_whole_pipeline(dev, why, why_n);
        if (!pso)
            return 0;

        static const void *params_key = NULL;
        static id<MTLBuffer> params_buf = nil;
        static const void *layout_key = NULL;
        static id<MTLBuffer> layout_buf = nil;

        id<MTLBuffer> pbuf =
            qtk_const_buffer(dev, params, sizeof(*params),
                             &params_key, &params_buf);
        if (!pbuf) {
            if (why && why_n) snprintf(why, why_n, "params buffer alloc failed");
            return 0;
        }

        const QwenTokenDeviceLayout *dl =
            qwen_token_device_state_layout(state);
        if (!dl) {
            if (why && why_n) snprintf(why, why_n, "missing device layout");
            return 0;
        }

        id<MTLBuffer> lbuf =
            qtk_const_buffer(dev, dl, sizeof(*dl),
                             &layout_key, &layout_buf);
        if (!lbuf) {
            if (why && why_n) snprintf(why, why_n, "layout buffer alloc failed");
            return 0;
        }

        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) {
            if (why && why_n) snprintf(why, why_n, "newCommandQueue failed");
            return 0;
        }

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (!cb) {
            if (why && why_n) snprintf(why, why_n, "commandBuffer failed");
            return 0;
        }

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) {
            if (why && why_n) snprintf(why, why_n, "compute encoder failed");
            return 0;
        }

        [enc setComputePipelineState:pso];

        [enc setBuffer:blob_buf offset:0 atIndex:0];
        [enc setBuffer:pbuf offset:0 atIndex:1];
        [enc setBuffer:state_buf offset:0 atIndex:2];
        [enc setBytes:&token_pos length:sizeof(token_pos) atIndex:3];
        [enc setBuffer:lbuf offset:0 atIndex:4];

        const NSUInteger threads = 256; /* QTK_THREADS; M2 cap is 384. */
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        if ([cb status] != MTLCommandBufferStatusCompleted) {
            NSError *err = [cb error];
            if (why && why_n)
                snprintf(why, why_n, "command buffer: %s",
                         err ? [[err localizedDescription] UTF8String]
                             : "did not complete");
            return 0;
        }

        return 1;
    }
}






// ---- Stage-5 runtime compile selfcheck (Hermes) ----
extern "C" int qwen_token_kernel_selfcheck(char *err, uint64_t err_cap) {

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

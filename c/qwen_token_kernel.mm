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
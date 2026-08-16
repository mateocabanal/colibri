/* qwen_moe.c — Qwen3.5 / Qwen3.6 / Qwen3.7 MoE inference engine (CPU, C11).
 *
 * Architecture (from transformers Qwen3_5MoeForCausalLM, verified against the
 * reference source on 2026-08-13 — see docs/qwen-moe-config-census.md):
 *
 *   hybrid decoder: per-layer `layer_types`:
 *     "linear_attention" -> Gated DeltaNet block (Mamba-style: causal depthwise
 *                           conv1d + delta-rule recurrence + gated RMSNorm)
 *     "full_attention"   -> GQA attention (QK-norm, partial RoPE, output gate)
 *   every layer ends with a fine-grained MoE FFN: softmax top-k router with
 *   renormalised weights, per-expert SwiGLU, plus a sigmoid-gated shared expert.
 *
 * Weight layout (engine snapshot produced by tools/convert_qwen_moe.py or
 * tools/make_qwen_moe_tiny.py):
 *   - dense tensors resident in RAM as f32 (embed, norms, attn, router,
 *     shared expert, lm_head)
 *   - routed experts STREAMED FROM DISK: per-expert tensors, either
 *       f32:   layers.N.mlp.experts.E.gate_up_proj [2*I, H] + .down_proj [H, I]
 *     or int8 (merged, olmoe byte layout):
 *       layers.N.mlp.experts.E.merged_weight (int8 g|u|d packed) + .qs (f32
 *       row scales) — the low-RAM path: 1 byte/param instead of 4.
 *     or packed (merged, ~half the int8 bytes; converted with
 *     tools/convert_qwen_moe.py --bits 4|3):
 *       layers.N.mlp.experts.E.merged_i4  — i4-grouped (fmt=4): 2 vals/byte,
 *         one f32 scale per 64-input group per row
 *       layers.N.mlp.experts.E.merged_i3  — int3-g64 (fmt=5): 24B per 64-input
 *         group (16B low plane + 8B high plane), one f32 scale per group
 *       both carry .qs per-group scales; the loader probes merged_weight,
 *       then merged_i4, then merged_i3, then the f32 pair, with exact-size
 *       rejection on every path.
 *   - per-layer LRU cache of resident experts, HOT pinning via route_trace.h
 *     usage heatmaps, COLI_USAGE history. This is what lets a low-spec box
 *     run a 397B checkpoint: only ~1.8GB dense + cache-sized experts resident.
 *
 * Modes:
 *   QWENMOE_MODE=teacher  QWENMOE_TEACHER="ids"  -> prints PRED <argmax> per
 *                          teacher-forced position (oracle logit check)
 *   QWENMOE_MODE=greedy   QWENMOE_PROMPT_IDS="..." QWENMOE_MAX_NEW=n
 *                          -> prints ID <token> per generated token
 *   CHAT=1                -> interactive chat (prompt on stdin, decoded output)
 *   (default)             -> self-test: runs every ref.json case in the model
 *                            dir and reports PASS/FAIL per case
 *
 * Env: SNAP (or argv[1]) model dir; CACHE (or argv[2]) experts resident per
 * layer; HOT, WARMUP pinning; EXPERT_DROP=1 fadvise(DONTNEED) after expert
 * reads; QWEN_PREFETCH=1 layer-lookahead expert prefetch (opt-in, measured);
 * CTX; MAX_NEW; COLI_TEMP/NUCLEUS; COLI_USAGE; ROUTE_TRACE.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <pthread.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#include "st.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "omp_tune.h"
#include "route_trace.h"
#ifdef COLI_METALIO
#include "metalio.h"
#endif

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/(1024.0*1024.0*1024.0); }
#else
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/(1024.0*1024.0); }
#endif
static int size_mul_ok(size_t a, size_t b, size_t *out){
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}
static int size_add_ok(size_t a, size_t b, size_t *out){
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}
static int i64_mul_ok(int64_t a, int64_t b, int64_t *out){
    if (a < 0 || b < 0 || (a && b > INT64_MAX / a)) return 0;
    *out = a * b;
    return 1;
}
static int i64_add_ok(int64_t a, int64_t b, int64_t *out){
    if (a < 0 || b < 0 || b > INT64_MAX - a) return 0;
    *out = a + b;
    return 1;
}
static int tensor_numel_ok(int64_t got, int64_t want){
    return got >= 0 && want >= 0 && got == want;
}
static size_t size_mul_or_die(size_t a, size_t b, const char *what){
    size_t out;
    if (!size_mul_ok(a, b, &out)) {
        fprintf(stderr, "%s: size overflow (%zu * %zu)\n", what, a, b);
        exit(1);
    }
    return out;
}
static void *calloc_checked(size_t n, size_t size, const char *what){
    size_t bytes = size_mul_or_die(n, size, what);
    void *p = calloc(1, bytes);
    if (!p && bytes) { fprintf(stderr, "OOM %s (%zu bytes)\n", what, bytes); exit(1); }
    return p;
}
static float *falloc(int64_t n){
    size_t bytes;
    if (n <= 0 || (uint64_t)n > SIZE_MAX ||
        !size_mul_ok((size_t)n, sizeof(float), &bytes)) {
        fprintf(stderr, "invalid float allocation count %lld\n", (long long)n);
        exit(1);
    }
    float *p = malloc(bytes);
    if (!p) { fprintf(stderr, "OOM %lld\n", (long long)n); exit(1); }
    return p;
}

#ifdef COLI_METAL
#include "backend_metal.h"
/* QWEN_METAL_COMPUTE=1: route routed-expert gate/up/down matmuls to the
 * Apple-GPU (Metal) backend's batched moe_block kernel. Slabs must be
 * page-aligned so the backend can wrap them zero-copy (16K pages); any
 * failure falls back to the CPU kernels per call (backend returns 0). */
static int g_metal_compute = 0;
#else
#define g_metal_compute 0
#endif
/* page-aligned when Metal zero-copy is active, plain malloc otherwise */
static void *moe_slab_alloc(size_t n){
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM expert\n"); exit(1); }
    if (g_metal_compute) {
        void *q = NULL;
        if (!posix_memalign(&q, 16384, n)) p = q;
        else free(p);   /* fall back to malloc'd; register will copy instead of wrap */
    }
    return p;
}

static float g_temp = 0.0f, g_nuc = 0.0f;       /* sample.h contract; 0 = greedy */
static int g_expert_drop = 0;
static int g_dense_drop = 0;    /* RAM-tight: DONTNEED dense file pages after the
                                 * resident f32 copy is made (page cache would
                                 * otherwise duplicate the whole snapshot) */

/* ---------- config ---------- */

typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, rotary_dim;
    int n_experts, topk, moe_inter, shared_inter;
    int lin_k_heads, lin_k_dim, lin_v_heads, lin_v_dim, conv_kernel;
    int vocab, max_pos;
    float theta, eps;
    int eos;
    int n_stop, stop_ids[8];        /* sample.h stop contract (empty = none) */
    int8_t *layer_is_gdn;           /* 1 = linear_attention (GDN), 0 = full */
} Cfg;

#include "tok.h"
#include "sample.h"                 /* needs Cfg (stops_arm_tok) + falloc/g_temp */

static int cfg_validate(const Cfg *c, char *why, size_t why_n){
#define CFG_BAD(...) do { snprintf(why, why_n, __VA_ARGS__); return 0; } while (0)
    if (c->hidden < 1 || c->hidden > (1 << 20) ||
        c->n_layers < 1 || c->n_layers > 4096 ||
        c->n_heads < 1 || c->n_heads > (1 << 16) ||
        c->n_kv_heads < 1 || c->n_kv_heads > c->n_heads ||
        c->head_dim < 1 || c->head_dim > (1 << 20) ||
        c->n_experts < 1 || c->n_experts > (1 << 20) ||
        c->topk < 1 || c->topk > c->n_experts ||
        c->moe_inter < 1 || c->moe_inter > (1 << 24) ||
        c->shared_inter < 1 || c->shared_inter > (1 << 24) ||
        c->lin_k_heads < 1 || c->lin_k_heads > (1 << 16) ||
        c->lin_k_dim < 1 || c->lin_k_dim > (1 << 20) ||
        c->lin_v_heads < 1 || c->lin_v_heads > (1 << 16) ||
        c->lin_v_dim < 1 || c->lin_v_dim > (1 << 20) ||
        c->conv_kernel < 1 || c->conv_kernel > 16 ||
        c->vocab < 1 || c->vocab > (1 << 24) ||
        c->max_pos < 1 || c->max_pos > (1 << 30))
        CFG_BAD("dimension out of range");
    if (!isfinite(c->theta) || c->theta <= 0.f ||
        !isfinite(c->eps) || c->eps <= 0.f)
        CFG_BAD("theta/epsilon must be finite and positive");
    if (c->eos < 0 || c->eos >= c->vocab)
        CFG_BAD("eos_token_id %d outside vocab_size %d", c->eos, c->vocab);
    if (c->hidden % c->n_heads)
        CFG_BAD("hidden_size %d is not divisible by attention heads %d",
                c->hidden, c->n_heads);
    if (c->n_heads % c->n_kv_heads)
        CFG_BAD("attention heads %d are not divisible by KV heads %d",
                c->n_heads, c->n_kv_heads);
    if (c->rotary_dim < 2 || c->rotary_dim > c->head_dim || (c->rotary_dim & 1))
        CFG_BAD("rotary_dim %d must be even and within [2,head_dim]", c->rotary_dim);
    if (c->lin_v_heads < c->lin_k_heads || c->lin_v_heads % c->lin_k_heads)
        CFG_BAD("GDN value heads %d must be a multiple of key heads %d",
                c->lin_v_heads, c->lin_k_heads);

    size_t qdim, kdim, vdim, conv, tmp, expert, bytes;
    if (!size_mul_ok((size_t)c->n_heads, c->head_dim, &qdim) ||
        qdim > (size_t)INT_MAX / 2)
        CFG_BAD("attention query projection rows overflow int");
    if (!size_mul_ok((size_t)c->n_kv_heads, c->head_dim, &tmp) || tmp > INT_MAX)
        CFG_BAD("attention KV projection rows overflow int");
    if (!size_mul_ok((size_t)c->lin_k_heads, c->lin_k_dim, &kdim) ||
        !size_mul_ok((size_t)c->lin_v_heads, c->lin_v_dim, &vdim) ||
        !size_mul_ok(kdim, 2, &tmp) || !size_add_ok(tmp, vdim, &conv) ||
        conv > INT_MAX || vdim > INT_MAX)
        CFG_BAD("GDN projection rows overflow int");
    if (!size_mul_ok((size_t)c->lin_v_heads, c->lin_k_dim, &tmp) ||
        !size_mul_ok(tmp, c->lin_v_dim, &tmp) ||
        !size_mul_ok(tmp, sizeof(float), &bytes))
        CFG_BAD("GDN recurrence allocation overflows size_t");
    if (!size_mul_ok(conv, (size_t)(c->conv_kernel - 1), &tmp) ||
        !size_mul_ok(tmp, sizeof(float), &bytes))
        CFG_BAD("GDN convolution allocation overflows size_t");
    if (!size_mul_ok((size_t)c->moe_inter, c->hidden, &expert) ||
        !size_mul_ok(expert, 3, &tmp) || !size_mul_ok(tmp, sizeof(float), &bytes))
        CFG_BAD("expert allocation overflows size_t");
    if (!size_mul_ok((size_t)c->vocab, c->hidden, &tmp) ||
        !size_mul_ok(tmp, sizeof(float), &bytes))
        CFG_BAD("embedding allocation overflows size_t");
    if (!size_mul_ok((size_t)c->n_layers, c->n_experts, &tmp))
        CFG_BAD("route-state allocation overflows size_t");
#undef CFG_BAD
    if (why_n) why[0] = 0;
    return 1;
}

/* ---------- dense per-layer weights ---------- */

/* dense matmul weight: f32 resident, or row-int8 (q + per-row f32 scales)
 * when the snapshot carries <name>_q8/<name>_qs — the low-RAM path used for
 * the real checkpoints (4x smaller resident dense, same scheme as experts). */
typedef struct { float *f; int8_t *q; float *s; } WT;

typedef struct {
    float *in_ln, *post_ln;
    /* full attention */
    WT q, k, v, o; float *qn, *kn;
    /* gated deltanet (linear_attn.*) */
    float *A_log, *dt_bias, *conv1d;    /* conv1d [C, 1, k] = [C, k] flat */
    WT in_a, in_b, in_qkv, in_z, gdn_out; float *gdn_norm;
    /* moe */
    WT router, se_gate, se_up, se_down, se_g;  /* shared expert */
} Layer;

/* ---------- expert cache (STREAMED from disk) ---------- */

typedef struct {
    int eid;                       /* -1 = slot in flight */
    int loading_eid;               /* expert id being loaded into this slot (-1 when idle) */
    int pinned;
    int fmt;                       /* 0=f32, 8=int8 merged, 4=i4-grouped, 5=int3-g64 */
    int mio;                       /* 1 = weight bytes live in a MetalIO shared buffer */
    int mio_slot;                  /* metalio slot id, -1 = not backed yet */
    int mio_resident;              /* 1 = current pointers point INTO the mio buffer */
    int64_t mio_event;             /* event value of the slot's latest load */
    int64_t mio_waited;            /* highest mio event already waited (pending if < mio_event) */
    float *gu, *d;                 /* f32: [2I,H] gate|up, [H,I] down */
    int8_t *g, *u, *dd;            /* q8: int8 blocks */
    float *gs, *us, *ds;           /* q8: row scales */
    uint8_t *g4, *u4, *d4;         /* packed (fmt 4/5): g|u|d blocks, packed layout */
    float *g4s, *u4s, *d4s;        /* packed: per-64-input-group f32 scales */
    uint64_t used;
} Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    Tok *tok;                       /* tokenizer owns the valid-vocabulary map */
    WT embed, lm_head; float *final_norm;
    Layer *L;
    LCache *cache;                 /* [n_layers] */
    uint64_t clock, hits, miss;
    float **K, **V;                /* [n_layers][kv_heads*max_t*head_dim] (f32 KV) */
    uint16_t **K16, **V16;         /* same layout as halves (QWEN_KV_F16) */
    int kv_len, max_t;
    float **gdn_S;                 /* [n_layers][v_heads*k_dim*v_dim] state */
    float **gdn_conv;              /* [n_layers][conv_dim*(k-1)] conv state */
    uint32_t **freq;               /* route_trace heatmap alias */
    int hot_pinned, hot_n, warmup_tokens, token_count;
    uint8_t *is_pinned;            /* [n_layers*n_experts] */
    int last_route[64];            /* most recent top-k (for lookahead prefetch) */
    int last_route_k;
    uint64_t prefetch_misses;      /* loads triggered by lookahead prefetch */
    double t_attn, t_gdn, t_moe, t_expio;   /* per-request phase timings (PROF) */
    double dense_load_s;
} Model;

static pthread_mutex_t g_mx = PTHREAD_MUTEX_INITIALIZER;
static int g_prefetch = 0;         /* QWEN_PREFETCH: layer-lookahead expert prefetch */
static int g_prefetch_pipe = 0;    /* QWEN_PREFETCH_PIPE: early l+1 issue from moe_token */

#ifdef COLI_METALIO
/* QWEN_METAL_IO=1: stream expert WEIGHT tensors through MTLIOCommandQueue
 * into persistent shared-storage MTLBuffers; the CPU kernels read the buffer
 * contents in place (no copy, byte-identical by construction). Scales stay on
 * the pread path (tiny). Any failure falls back to pread; never mandatory. */
static int g_metal_io = 0;
static int g_mio_prefetching = 0;   /* 1 = speculative expert_get: enqueue, no wait */
static int g_mio_async_issue = 0;   /* 1 = exact-demand issue: enqueue, no wait */
static int g_mio_fd[64], g_mio_fid[64], g_mio_n;
static int mio_file_for(int fd){
    for (int i = 0; i < g_mio_n; i++) if (g_mio_fd[i] == fd) return g_mio_fid[i];
    return -1;
}
#endif

static size_t gdn_state_count(const Cfg *c){
    size_t n = size_mul_or_die((size_t)c->lin_v_heads, c->lin_k_dim, "GDN recurrence");
    return size_mul_or_die(n, c->lin_v_dim, "GDN recurrence");
}
static size_t gdn_conv_count(const Cfg *c){
    size_t k = size_mul_or_die((size_t)c->lin_k_heads, c->lin_k_dim, "GDN convolution");
    size_t v = size_mul_or_die((size_t)c->lin_v_heads, c->lin_v_dim, "GDN convolution");
    size_t twice, channels;
    if (!size_mul_ok(k, 2, &twice) || !size_add_ok(twice, v, &channels)) {
        fprintf(stderr, "GDN convolution channel count overflow\n"); exit(1);
    }
    return size_mul_or_die(channels, (size_t)(c->conv_kernel - 1), "GDN convolution");
}
static size_t kv_state_count(const Cfg *c, int max_t){
    size_t n = size_mul_or_die((size_t)c->n_kv_heads, (size_t)max_t, "KV cache");
    return size_mul_or_die(n, c->head_dim, "KV cache");
}

/* ---------- fp16 KV cache (QWEN_KV_F16, default on) ---------- */

/* IEEE half, round-to-nearest-even (bit-exact vs the usual softfloat path;
 * st.h already carries the f16_to_f32 decoder). */
static uint16_t f32_to_f16(float x){
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t e = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFFu;
    if (e >= 0x1F) return (uint16_t)(sign | 0x7C00u);          /* inf/nan */
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;                    /* underflow */
        m |= 0x800000u;                                        /* subnormal */
        int shift = 14 - e;
        uint32_t half = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = (m >> 13);
    uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) half++;   /* RNE */
    return (uint16_t)(sign | ((uint32_t)e << 10) | half);
}

static int g_kv_f16 = 1;             /* QWEN_KV_F16=0 disables (f32 KV) */

static void kv_store_row(Model *m, int layer, int g, int pos, const float *row, int hd){
    int64_t off = ((int64_t)g * m->max_t + pos) * hd;
    if (g_kv_f16) {
        uint16_t *dst = m->K16[layer] + off;
        for (int d = 0; d < hd; d++) dst[d] = f32_to_f16(row[d]);
    } else {
        memcpy(m->K[layer] + off, row, (size_t)hd * sizeof(float));
    }
}
static void kv_store_row_v(Model *m, int layer, int g, int pos, const float *row, int hd){
    int64_t off = ((int64_t)g * m->max_t + pos) * hd;
    if (g_kv_f16) {
        uint16_t *dst = m->V16[layer] + off;
        for (int d = 0; d < hd; d++) dst[d] = f32_to_f16(row[d]);
    } else {
        memcpy(m->V[layer] + off, row, (size_t)hd * sizeof(float));
    }
}
static void kv_load_row(Model *m, int layer, int g, int pos, float *out, int hd){
    int64_t off = ((int64_t)g * m->max_t + pos) * hd;
    if (g_kv_f16) {
        const uint16_t *src = m->K16[layer] + off;
        for (int d = 0; d < hd; d++) out[d] = f16_to_f32(src[d]);
    } else {
        memcpy(out, m->K[layer] + off, (size_t)hd * sizeof(float));
    }
}
static void kv_load_row_v(Model *m, int layer, int g, int pos, float *out, int hd){
    int64_t off = ((int64_t)g * m->max_t + pos) * hd;
    if (g_kv_f16) {
        const uint16_t *src = m->V16[layer] + off;
        for (int d = 0; d < hd; d++) out[d] = f16_to_f32(src[d]);
    } else {
        memcpy(out, m->V[layer] + off, (size_t)hd * sizeof(float));
    }
}

/* All output-token selection comes through this helper.  Checkpoints often
 * pad lm_head/embed to an accelerator-friendly row count beyond the actual
 * tokenizer vocabulary; those rows have no Tok.id2str entry and must never be
 * sampled.  Added stop/control tokens do have entries and remain eligible. */
static int qwen_pick_token(Model *m, const float *logits, int ban){
    int V = m->c.vocab, eligible = 0;
    float *masked = falloc(V);
    for (int id = 0; id < V; id++) {
        int valid = m->tok && id < m->tok->n_ids && m->tok->id2str[id];
        if (valid && id != ban) { masked[id] = logits[id]; eligible++; }
        else masked[id] = -INFINITY;
    }
    if (!eligible) {
        fprintf(stderr, "tokenizer.json: no selectable token ids inside vocab_size=%d\n", V);
        free(masked);
        exit(1);
    }
    int token = pick_tok(masked, V, -1);
    free(masked);
    if (token < 0 || token >= V || token >= m->tok->n_ids || !m->tok->id2str[token]) {
        fprintf(stderr, "token selection produced invalid id %d\n", token);
        exit(1);
    }
    return token;
}

static int ids_in_vocab(const Cfg *c, const int *ids, int n, const char *source){
    if (!ids || n < 1) {
        fprintf(stderr, "%s: empty token sequence\n", source);
        return 0;
    }
    for (int i = 0; i < n; i++) if (ids[i] < 0 || ids[i] >= c->vocab) {
        fprintf(stderr, "%s: token id %d at position %d outside [0,%d)\n",
                source, ids[i], i, c->vocab);
        return 0;
    }
    return 1;
}

static void logits_free(float **p){
    if (!p) return;
    free(*p);
    *p = NULL;
}

/* ---------- config.json ---------- */

/* config.json arrives from an untrusted mirror: require each dimension to be
 * present and numeric (same discipline as olmoe.c / SEC-9). */
static double req_num(jval *r, const char *k){
    jval *v = json_get(r, k);
    if (!v || v->t != J_NUM || !isfinite(v->num)) {
        fprintf(stderr, "config.json: missing, non-numeric, or non-finite \"%s\"\n", k);
        exit(1);
    }
    return v->num;
}
static double opt_num(jval *r, const char *k, double dflt){
    jval *v = json_get(r, k);
    if (!v) return dflt;
    if (v->t != J_NUM || !isfinite(v->num)) {
        fprintf(stderr, "config.json: non-numeric or non-finite \"%s\"\n", k);
        exit(1);
    }
    return v->num;
}
static int num_to_int(double v, const char *k){
    if (!isfinite(v) || v < INT_MIN || v > INT_MAX || trunc(v) != v) {
        fprintf(stderr, "config.json: \"%s\" must be an integer in int range\n", k);
        exit(1);
    }
    return (int)v;
}
static int req_int(jval *r, const char *k){
    return num_to_int(req_num(r, k), k);
}
static int opt_int(jval *r, const char *k, int dflt){
    jval *v = json_get(r, k);
    return v ? num_to_int(opt_num(r, k, dflt), k) : dflt;
}

/* Qwen3.5/3.6 checkpoints carry the text config inside "text_config"
 * (vision-language wrapper). The tiny fixtures write it flat. Handle both. */
static jval *cfg_root(jval *r){
    jval *tc = json_get(r, "text_config");
    return (tc && tc->t == J_OBJ) ? tc : r;
}

static int parse_layer_types(jval *r, int n_layers, int8_t *out){
    jval *lt = json_get(r, "layer_types");
    if (!lt || lt->t != J_ARR || lt->len != n_layers) {
        fprintf(stderr, "config.json: layer_types must be an array of %d entries "
                        "({\"linear_attention\"|\"full_attention\"})\n", n_layers);
        return -1;
    }
    for (int i = 0; i < n_layers; i++) {
        jval *e = lt->kids[i];
        if (e->t != J_STR) return -1;
        if (!strcmp(e->str, "linear_attention")) out[i] = 1;
        else if (!strcmp(e->str, "full_attention")) out[i] = 0;
        else { fprintf(stderr, "config.json: unknown layer type \"%s\"\n", e->str); return -1; }
    }
    return 0;
}

static void load_cfg(Cfg *c, const char *snap){
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0 || n > (256L << 20)) { fprintf(stderr, "%s: config.json missing or larger than 256 MB\n", path); exit(1); }
    char *buf = malloc((size_t)n + 1); if (!buf) { fprintf(stderr, "OOM reading %s\n", path); exit(1); }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "%s: short read\n", path); exit(1); }
    buf[n] = 0; fclose(f);
    char *arena = NULL; jval *r = json_parse(buf, &arena);
    jval *root = cfg_root(r);

    c->hidden    = req_int(root, "hidden_size");
    c->n_layers  = req_int(root, "num_hidden_layers");
    c->n_heads   = req_int(root, "num_attention_heads");
    c->n_kv_heads= req_int(root, "num_key_value_heads");
    c->head_dim  = opt_int(root, "head_dim",
                           c->n_heads ? c->hidden / c->n_heads : 0);
    c->n_experts = req_int(root, "num_experts");
    c->topk      = req_int(root, "num_experts_per_tok");
    c->moe_inter = req_int(root, "moe_intermediate_size");
    c->shared_inter = req_int(root, "shared_expert_intermediate_size");
    c->lin_k_heads = req_int(root, "linear_num_key_heads");
    c->lin_k_dim   = req_int(root, "linear_key_head_dim");
    c->lin_v_heads = req_int(root, "linear_num_value_heads");
    c->lin_v_dim   = req_int(root, "linear_value_head_dim");
    c->conv_kernel = req_int(root, "linear_conv_kernel_dim");
    c->vocab     = req_int(root, "vocab_size");
    c->max_pos   = opt_int(root, "max_position_embeddings", 262144);
    c->eps       = (float)opt_num(root, "rms_norm_eps", 1e-6);
    c->eos       = opt_int(root, "eos_token_id", 248044);

    jval *rp = json_get(root, "rope_parameters");
    c->theta = (rp && rp->t == J_OBJ) ? (float)opt_num(rp, "rope_theta", 10000000.0)
                                      : (float)opt_num(root, "rope_theta", 10000000.0);
    double prf = (rp && rp->t == J_OBJ) ? opt_num(rp, "partial_rotary_factor", 1.0)
                                        : opt_num(root, "partial_rotary_factor", 1.0);
    double rotary = c->head_dim * prf;
    if (!isfinite(prf) || !isfinite(rotary) || rotary < INT_MIN || rotary > INT_MAX ||
        trunc(rotary) != rotary) {
        fprintf(stderr, "config.json: partial rotary dimension must be an integer\n");
        exit(1);
    }
    c->rotary_dim = (int)rotary;

    char why[256];
    if (!cfg_validate(c, why, sizeof(why))) {
        fprintf(stderr, "config.json: %s\n", why);
        exit(1);
    }
    c->n_stop = 0;
    c->layer_is_gdn = malloc((size_t)c->n_layers);
    if (!c->layer_is_gdn) { fprintf(stderr, "OOM layer_types\n"); exit(1); }
    if (parse_layer_types(root, c->n_layers, c->layer_is_gdn) != 0) exit(1);
    free(buf); free(arena);
}

/* ---------- tensor loading ---------- */

static char g_prefix[64] = "";      /* "model.language_model." or "" (probed) */

static float *load_t(Model *m, const char *name, int64_t want){
    char nm[512]; snprintf(nm, sizeof(nm), "%s%s", g_prefix, name);
    int64_t n = st_numel(&m->S, nm);
    if (n < 0) { fprintf(stderr, "missing %s\n", nm); exit(1); }
    if (!tensor_numel_ok(n, want)) {
        fprintf(stderr, "%s: %lld elems — expected %lld, refusing\n",
                nm, (long long)n, (long long)want);
        exit(1);
    }
    float *p = falloc(n);
    st_read_f32(&m->S, nm, p, g_dense_drop);
    return p;
}

/* tensor present in f32 or q8 form (q8-only snapshots have no f32 copy) */
static int st_have(Model *m, const char *name){
    char nm[512]; snprintf(nm, sizeof(nm), "%s_q8", name);
    return st_has(&m->S, name) || st_has(&m->S, nm);
}

/* dense matmul weight: probes <base>_q8 + <base>_qs (row-int8, exact-size
 * checked — hostile-container guard), falls back to f32 <base>. */
static WT load_wt_named(Model *m, const char *base, int64_t O, int64_t I){
    WT w = {0};
    int64_t want;
    if (!i64_mul_ok(O, I, &want) || (uint64_t)want > SIZE_MAX) {
        fprintf(stderr, "%s: tensor dimensions overflow (%lld x %lld)\n",
                base, (long long)O, (long long)I);
        exit(1);
    }
    char nm[512]; snprintf(nm, sizeof(nm), "%s_q8", base);
    st_tensor *tq = st_find(&m->S, nm);
    if (tq) {
        if (!tensor_numel_ok(tq->nbytes, want)) {
            fprintf(stderr, "%s: %lld bytes — expected %lld, refusing\n",
                    nm, (long long)tq->nbytes, (long long)want); exit(1);
        }
        char qsnm[512]; snprintf(qsnm, sizeof(qsnm), "%s_qs", base);
        st_tensor *ts = st_find(&m->S, qsnm);
        if (!ts || ts->numel != O) {
            fprintf(stderr, "%s: scale array is %lld elems — expected %lld, refusing\n",
                    qsnm, (long long)(ts ? ts->numel : -1), (long long)O); exit(1);
        }
        w.q = malloc((size_t)want); if (!w.q) { fprintf(stderr, "OOM %s\n", nm); exit(1); }
        w.s = falloc(O);
        st_read_raw(&m->S, nm, w.q, g_dense_drop);
        st_read_f32(&m->S, qsnm, w.s, g_dense_drop);
        return w;
    }
    int64_t n = st_numel(&m->S, base);
    if (n < 0) { fprintf(stderr, "missing %s\n", base); exit(1); }
    if (!tensor_numel_ok(n, want)) {
        fprintf(stderr, "%s: %lld elems — expected %lld, refusing\n",
                base, (long long)n, (long long)want); exit(1);
    }
    w.f = falloc(n);
    st_read_f32(&m->S, base, w.f, g_dense_drop);
    return w;
}
static WT load_wt(Model *m, const char *name, int64_t O, int64_t I){
    char base[512]; snprintf(base, sizeof(base), "%s%s", g_prefix, name);
    return load_wt_named(m, base, O, I);
}

/* ---------- kernels ---------- */

static void matmul(float *y, const float *x, const float *w, int S, int O, int I){
    #pragma omp parallel for schedule(static)
    for (int so = 0; so < S * O; so++) {
        int s = so / O, o = so % O;
        const float *xs = x + (int64_t)s * I;
        const float *wr = w + (int64_t)o * I;
        float acc = 0.f;
        #pragma omp simd reduction(+:acc)
        for (int i = 0; i < I; i++) acc += xs[i] * wr[i];
        y[(int64_t)so] = acc;
    }
}

/* int8 weights + per-row f32 scales (dequant on use) */
static void matmul_q8(float *y, const float *x, const int8_t *q, const float *sc,
                      int S, int O, int I){
    #pragma omp parallel for schedule(static)
    for (int so = 0; so < S * O; so++) {
        int s = so / O, o = so % O;
        const float *xs = x + (int64_t)s * I;
        const int8_t *wr = q + (int64_t)o * I;
        float acc = 0;
        #pragma omp simd reduction(+:acc)
        for (int i = 0; i < I; i++) acc += xs[i] * (float)wr[i];
        y[(int64_t)so] = acc * sc[o];
    }
}

/* WT dispatch: row-int8 when the snapshot supplied it, f32 otherwise */
static void wt_mul(float *y, const float *x, WT *w, int S, int O, int I){
    if (w->q) matmul_q8(y, x, w->q, w->s, S, O, I);
    else      matmul(y, x, w->f, S, O, I);
}

/* ---------- packed expert kernels (copied from quant.h per the standalone-
 * engine convention; the layouts are part of the snapshot format) ---------- */

#ifdef __AVX2__
#include <immintrin.h>
static inline float qm_hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* i4 grouped (fmt=4): 2 values/byte, per-64-input-group f32 scales.
 * y[S,O] = x[S,I] @ W^T; values stored v+8 in [-8,7]. */
static void matmul_i4_grouped(float *y, const float *x, const uint8_t *q4, const float *scale,
                              int S, int I, int O, int gs){
    int rb=(I+1)/2; int ng=(I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0; g*gs<I; g++){
                int base=g*gs; int glen=gs; if(base+glen>I) glen=I-base;
                float sc=scl[g];
                int i=base;
#ifdef __AVX2__
                const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
                __m256 acc=_mm256_setzero_ps();
                for(; i+16<=base+glen; i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                    __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
                a+=qm_hsum256(acc)*sc;
#elif defined(__ARM_NEON)
                /* group partial accumulates in f32 vectors, scaled once by the
                 * group scale at the end — same structure as the AVX2 arm. */
                const uint8x8_t m4v=vdup_n_u8(0x0F); const int8x8_t b8v=vdup_n_s8(8);
                float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
                for(; i+16<=base+glen; i+=16){
                    uint8x8_t by=vld1_u8(w+(i>>1));
                    uint8x8x2_t z=vzip_u8(vand_u8(by,m4v), vshr_n_u8(by,4));
                    int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]),b8v));
                    int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]),b8v));
                    ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                    ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                    ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                    ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
                }
                a += vaddvq_f32(vaddq_f32(ac0,ac1)) * sc;
#endif
                for(; i<base+glen; i+=2){
                    if(i+1<base+glen){ uint8_t byte=w[i>>1];
                        a+=(xs[i]*(float)((int)(byte&0xF)-8)+xs[i+1]*(float)((int)(byte>>4)-8))*sc; }
                    else { uint8_t byte=w[i>>1]; a+=xs[i]*(float)((int)(byte&0xF)-8)*sc; }
                }
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* QWEN_I4_ROWREDUCE=1 (ChatGPT perf pass): fold each 64-group's f32 scale
 * into the SIMD lanes and horizontally reduce once per output row instead
 * of per group. FP order changes (scale now multiplies lanes BEFORE the
 * final horizontal add) — hence opt-in; the default path is untouched. */
static int qwen_i4_rowreduce_enabled(void){
    static int initialized = 0, enabled = 0;
    if (!initialized) {
        const char *s = getenv("QWEN_I4_ROWREDUCE");
        enabled = s && s[0] && strcmp(s, "0") != 0;
        initialized = 1;
    }
    return enabled;
}

/* Fused gate+up i4-grouped pair (S==1): one activation load serves BOTH
 * matrices — the decode path computes gate(x) and up(x) on the same x, so
 * this halves the x memory traffic and the loop overhead. Accumulation
 * order per group is identical to two separate matmul_i4_grouped calls
 * (unpack -> fma per 16, group partial scaled once), so results are
 * bit-exact against the reference. Only the NEON arm is needed here; the
 * scalar tail handles the remainder exactly like the single kernel. */
static void matmul_i4_grouped_pair(float *yg, float *yu,
                                   const float *x,
                                   const uint8_t *qg, const float *sg,
                                   const uint8_t *qu, const float *su,
                                   int I, int O, int gs){
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
#ifdef __ARM_NEON
    /* QWEN_I4_ROWREDUCE=1: one horizontal reduce per output row. Only
     * legal when every group is a whole number of 16-value SIMD chunks
     * (no scalar tail inside any group). S==1 decode caller. */
    if (qwen_i4_rowreduce_enabled() && (I % 16) == 0 && (gs % 16) == 0) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const float *scg = sg + (int64_t)o * ng;
            const float *scu = su + (int64_t)o * ng;
            const uint8x8_t m4v = vdup_n_u8(0x0F), b8v = vdup_n_s8(8);
            float32x4_t acg0 = vdupq_n_f32(0), acg1 = vdupq_n_f32(0);
            float32x4_t acu0 = vdupq_n_f32(0), acu1 = vdupq_n_f32(0);
            for (int g = 0; g * gs < I; g++) {
                int base = g * gs, end = base + gs;
                float32x4_t sg4 = vdupq_n_f32(scg[g]), su4 = vdupq_n_f32(scu[g]);
                for (int i = base; i + 16 <= end; i += 16) {
                    uint8x8_t byg = vld1_u8(qg + (int64_t)o * rb + (i >> 1));
                    uint8x8_t byu = vld1_u8(qu + (int64_t)o * rb + (i >> 1));
                    uint8x8x2_t zg = vzip_u8(vand_u8(byg, m4v), vshr_n_u8(byg, 4));
                    uint8x8x2_t zu = vzip_u8(vand_u8(byu, m4v), vshr_n_u8(byu, 4));
                    int16x8_t wg0 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(zg.val[0]), b8v));
                    int16x8_t wg1 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(zg.val[1]), b8v));
                    int16x8_t wu0 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(zu.val[0]), b8v));
                    int16x8_t wu1 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(zu.val[1]), b8v));
                    float32x4_t x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
                    float32x4_t x2 = vld1q_f32(x + i + 8), x3 = vld1q_f32(x + i + 12);
                    acg0 = vmlaq_f32(acg0, sg4, vmulq_f32(x0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(wg0)))));
                    acg1 = vmlaq_f32(acg1, sg4, vmulq_f32(x1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(wg0)))));
                    acg0 = vmlaq_f32(acg0, sg4, vmulq_f32(x2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(wg1)))));
                    acg1 = vmlaq_f32(acg1, sg4, vmulq_f32(x3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(wg1)))));
                    acu0 = vmlaq_f32(acu0, su4, vmulq_f32(x0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(wu0)))));
                    acu1 = vmlaq_f32(acu1, su4, vmulq_f32(x1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(wu0)))));
                    acu0 = vmlaq_f32(acu0, su4, vmulq_f32(x2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(wu1)))));
                    acu1 = vmlaq_f32(acu1, su4, vmulq_f32(x3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(wu1)))));
                }
            }
            yg[o] = vaddvq_f32(vaddq_f32(acg0, acg1));
            yu[o] = vaddvq_f32(vaddq_f32(acu0, acu1));
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *wg = qg + (int64_t)o * rb;
        const uint8_t *wu = qu + (int64_t)o * rb;
        const float *scg = sg + (int64_t)o * ng;
        const float *scu = su + (int64_t)o * ng;
        const float *xs = x;
        float ag = 0, au = 0;
        for (int g = 0; g * gs < I; g++) {
            int base = g * gs, glen = gs; if (base + glen > I) glen = I - base;
            float scgv = scg[g], scuv = scu[g];
            int i = base;
#ifdef __ARM_NEON
            const uint8x8_t m4v = vdup_n_u8(0x0F); const int8x8_t b8v = vdup_n_s8(8);
            float32x4_t acg0 = vdupq_n_f32(0), acg1 = vdupq_n_f32(0);
            float32x4_t acu0 = vdupq_n_f32(0), acu1 = vdupq_n_f32(0);
            for (; i + 16 <= base + glen; i += 16) {
                uint8x8_t byg = vld1_u8(wg + (i >> 1));
                uint8x8_t byu = vld1_u8(wu + (i >> 1));
                uint8x8x2_t zg = vzip_u8(vand_u8(byg, m4v), vshr_n_u8(byg, 4));
                uint8x8x2_t zu = vzip_u8(vand_u8(byu, m4v), vshr_n_u8(byu, 4));
                int16x8_t wg0 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(zg.val[0]), b8v));
                int16x8_t wg1 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(zg.val[1]), b8v));
                int16x8_t wu0 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(zu.val[0]), b8v));
                int16x8_t wu1 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(zu.val[1]), b8v));
                float32x4_t x0 = vld1q_f32(xs + i), x1 = vld1q_f32(xs + i + 4);
                float32x4_t x2 = vld1q_f32(xs + i + 8), x3 = vld1q_f32(xs + i + 12);
                acg0 = vfmaq_f32(acg0, x0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(wg0))));
                acg1 = vfmaq_f32(acg1, x1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(wg0))));
                acg0 = vfmaq_f32(acg0, x2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(wg1))));
                acg1 = vfmaq_f32(acg1, x3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(wg1))));
                acu0 = vfmaq_f32(acu0, x0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(wu0))));
                acu1 = vfmaq_f32(acu1, x1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(wu0))));
                acu0 = vfmaq_f32(acu0, x2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(wu1))));
                acu1 = vfmaq_f32(acu1, x3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(wu1))));
            }
            ag += vaddvq_f32(vaddq_f32(acg0, acg1)) * scgv;
            au += vaddvq_f32(vaddq_f32(acu0, acu1)) * scuv;
#endif
            for (; i < base + glen; i += 2) {
                if (i + 1 < base + glen) {
                    uint8_t bg = wg[i >> 1], bu = wu[i >> 1];
                    ag += (xs[i] * (float)((int)(bg & 0xF) - 8) + xs[i + 1] * (float)((int)(bg >> 4) - 8)) * scgv;
                    au += (xs[i] * (float)((int)(bu & 0xF) - 8) + xs[i + 1] * (float)((int)(bu >> 4) - 8)) * scuv;
                } else {
                    uint8_t bg = wg[i >> 1], bu = wu[i >> 1];
                    ag += xs[i] * (float)((int)(bg & 0xF) - 8) * scgv;
                    au += xs[i] * (float)((int)(bu & 0xF) - 8) * scuv;
                }
            }
        }
        yg[o] = ag;
        yu[o] = au;
    }
}

/* int3-g64 (fmt=5): 3-bit weights, values [-4,3] stored v+4, ONE f32 scale per
 * 64-input group. Per group: 16B low plane (2 bits/val) + 8B high plane
 * (1 bit/val) = 24B. 3.5 bits/weight effective. */
#define QM_I3_GROUP 64
#define QM_I3_GBYTES 24
static inline int64_t qm_i3_groups(int I){ return ((int64_t)I + QM_I3_GROUP - 1) / QM_I3_GROUP; }
static inline int64_t qm_i3_rowbytes(int I){ return qm_i3_groups(I) * QM_I3_GBYTES; }
static void matmul_i3(float *y, const float *x, const uint8_t *q3, const float *scale, int S, int I, int O){
    int64_t ng=qm_i3_groups(I), rb=qm_i3_rowbytes(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *wrow=q3+(int64_t)o*rb;
        const float *srow=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            float acc=0;
            for(int64_t g=0; g<ng; g++){
                const uint8_t *lo=wrow+g*QM_I3_GBYTES, *hi=lo+16;
                int base=(int)(g*QM_I3_GROUP), n = I-base < QM_I3_GROUP ? I-base : QM_I3_GROUP;
                float a=0; int k=0;
#ifdef __AVX2__
                if(n==QM_I3_GROUP){
                    const __m128i m2=_mm_set1_epi8(0x03);
                    const __m128i bsel=_mm_set_epi8(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
                    const __m128i bitm=_mm_set_epi8((char)128,64,32,16,8,4,2,1,(char)128,64,32,16,8,4,2,1);
                    const __m128i four8=_mm_set1_epi8(4);
                    const __m256i b4=_mm256_set1_epi32(4);
                    __m256 ac0=_mm256_setzero_ps(), ac1=_mm256_setzero_ps();
                    for(;k+16<=QM_I3_GROUP;k+=16){
                        __m128i by=_mm_cvtsi32_si128(*(const int*)(lo+(k>>2)));
                        __m128i p0=_mm_and_si128(by,m2), p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
                        __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2), p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
                        __m128i l01=_mm_unpacklo_epi8(p0,p1), h23=_mm_unpacklo_epi8(p2,p3);
                        __m128i lov=_mm_unpacklo_epi16(l01,h23);
                        __m128i hv=_mm_shuffle_epi8(_mm_cvtsi32_si128(hi[k>>3]|(hi[(k>>3)+1]<<8)),bsel);
                        __m128i hb=_mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hv,bitm),bitm),four8);
                        __m128i u=_mm_add_epi8(lov,hb);
                        __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(u),b4));
                        __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(u,8)),b4));
                        ac0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+base+k),  w0,ac0);
                        ac1=_mm256_fmadd_ps(_mm256_loadu_ps(xs+base+k+8),w1,ac1);
                    }
                    a=qm_hsum256(_mm256_add_ps(ac0,ac1));
                }
#elif defined(__ARM_NEON)
                if(n==QM_I3_GROUP){
                    const uint8x8_t m2v=vdup_n_u8(3); const int8x16_t b4q=vdupq_n_s8(4);
                    const uint8x16_t bitm={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
                    const uint8x16_t fourq=vdupq_n_u8(4);
                    float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
                    for(;k+16<=QM_I3_GROUP;k+=16){
                        uint32_t wd; memcpy(&wd, lo+(k>>2), 4);
                        uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                        uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v),              vand_u8(vshr_n_u8(by,2),m2v));
                        uint8x8x2_t z23=vzip_u8(vand_u8(vshr_n_u8(by,4),m2v), vshr_n_u8(by,6));
                        uint16x4x2_t zz=vzip_u16(vreinterpret_u16_u8(z01.val[0]), vreinterpret_u16_u8(z23.val[0]));
                        uint8x16_t lov=vcombine_u8(vreinterpret_u8_u16(zz.val[0]), vreinterpret_u8_u16(zz.val[1]));
                        uint8x16_t hv=vcombine_u8(vdup_n_u8(hi[k>>3]), vdup_n_u8(hi[(k>>3)+1]));
                        uint8x16_t hb=vandq_u8(vtstq_u8(hv,bitm), fourq);
                        int8x16_t wq=vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(lov,hb)), b4q);
                        int16x8_t w0=vmovl_s8(vget_low_s8(wq)), w1=vmovl_s8(vget_high_s8(wq));
                        ac0=vfmaq_f32(ac0, vld1q_f32(xs+base+k),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                        ac1=vfmaq_f32(ac1, vld1q_f32(xs+base+k+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                        ac0=vfmaq_f32(ac0, vld1q_f32(xs+base+k+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                        ac1=vfmaq_f32(ac1, vld1q_f32(xs+base+k+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
                    }
                    a=vaddvq_f32(vaddq_f32(ac0,ac1));
                }
#endif
                for(;k<n;k++){
                    unsigned u=((lo[k>>2]>>((k&3)*2))&3) | (((hi[k>>3]>>(k&7))&1)<<2);
                    a += xs[base+k]*(float)((int)u-4);
                }
                acc += a*srow[g];
            }
            y[(int64_t)s*O+o]=acc;
        }
    }
}

/* Qwen3.5 RMSNorm: weight is ZERO-initialised and applied as (1 + w). */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps){
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * (1.f + w[i]);
}

static float silu(float x){ return x / (1.f + expf(-x)); }

/* RMSNormGated (GDN output norm): w * rmsnorm(x) * silu(gate) */
static void rmsnorm_gated_row(float *out, const float *x, const float *z,
                              const float *w, int D, float eps){
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = w[i] * (x[i] * r) * silu(z[i]);
}

static void softmax_row(float *x, int n){
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- expert cache (disk streaming) ---------- */

/* Slot buffers are allocated per-format, ONLY what the load actually uses:
 * the f32 path needs the gate|up|down f32 block; the int8 path needs the
 * int8 merged block + f32 row scales. Never both (that was a 12 MB/slot
 * waste on int8 snapshots: at 8 slots x 40 layers = ~4 GB of dead RAM). */
static void slot_alloc_f32(Slot *s, int64_t ng, int64_t nd){
    if (s->gu) return;
    int64_t twice, total;
    if (!i64_mul_ok(ng, 2, &twice) || !i64_add_ok(twice, nd, &total)) {
        fprintf(stderr, "expert f32 allocation overflows\n"); exit(1);
    }
    s->gu = falloc(total);                    /* gate|up [2I,H] + down [H,I] */
    s->d = s->gu + 2 * ng;
    s->fmt = 0;
}
static void slot_alloc_q8(Model *m, Slot *s){
    if (s->g) return;
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->moe_inter * c->hidden;
    int64_t nd = (int64_t)c->hidden * c->moe_inter;
    int64_t twice, total;
    if (!i64_mul_ok(ng, 2, &twice) || !i64_add_ok(twice, nd, &total) ||
        (uint64_t)total > SIZE_MAX) {
        fprintf(stderr, "expert q8 allocation overflows\n"); exit(1);
    }
    s->g = moe_slab_alloc((size_t)total); if (!s->g) { fprintf(stderr, "OOM expert\n"); exit(1); }
    s->u = s->g + ng; s->dd = s->g + ng + ng;
    int64_t scale_n = (int64_t)c->moe_inter * 2 + c->hidden;
    float *sb = falloc(scale_n);
    s->gs = sb; s->us = sb + c->moe_inter; s->ds = sb + c->moe_inter + c->moe_inter;
    s->pinned = 0;
    s->fmt = 8;
#ifdef COLI_METAL
    if (g_metal_compute) { coli_metal_register(s->g, (size_t)total); coli_metal_register(sb, (size_t)scale_n * 4); }
#endif
}
/* Packed experts (fmt 4 = i4-grouped, fmt 5 = int3-g64): weights packed with
 * 2 values/byte (i4) or 24B/64-input group (i3), scales one f32 per 64-input
 * group per output row — HALF the int8 byte count, same merged g|u|d shape,
 * so the disk-streaming LRU is unchanged. */
static void slot_alloc_packed(Model *m, Slot *s, int fmt){
    if (s->g4) return;
    Cfg *c = &m->c;
    int64_t I = c->moe_inter, H = c->hidden;
    int64_t rbH = (fmt == 5) ? qm_i3_rowbytes((int)H) : (H + 1) / 2;
    int64_t rbI = (fmt == 5) ? qm_i3_rowbytes((int)I) : (I + 1) / 2;
    int64_t ngH = (H + 63) / 64, ngI = (I + 63) / 64;
    int64_t gu_bytes, total;
    if (!i64_mul_ok(rbH, I, &gu_bytes) || !i64_mul_ok(gu_bytes, 2, &gu_bytes) ||
        !i64_mul_ok(rbI, H, &total) || !i64_add_ok(gu_bytes, total, &total) ||
        (uint64_t)total > SIZE_MAX) {
        fprintf(stderr, "expert packed allocation overflows\n"); exit(1);
    }
    s->g4 = moe_slab_alloc((size_t)total); if (!s->g4) { fprintf(stderr, "OOM expert\n"); exit(1); }
    s->u4 = s->g4 + rbH * I; s->d4 = s->g4 + 2 * rbH * I;
    int64_t scale_n = I * ngH * 2 + H * ngI;
    float *sb = falloc(scale_n);
    s->g4s = sb; s->u4s = sb + I * ngH; s->d4s = sb + 2 * I * ngH;
    s->pinned = 0;
    s->fmt = fmt;
#ifdef COLI_METAL
    if (g_metal_compute && fmt == 4) { coli_metal_register(s->g4, (size_t)total); coli_metal_register(sb, (size_t)scale_n * 4); }
#endif
}

static void load_expert(Model *m, int layer, int eid, Slot *s){
    char nm[512], qsnm[512];
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->moe_inter * cc->hidden;
    int64_t nd = (int64_t)cc->hidden * cc->moe_inter;

#ifdef COLI_METALIO
    /* MetalIO path (QWEN_METAL_IO=1): the merged expert WEIGHT tensor streams
     * through MTLIOCommandQueue into a persistent shared-storage MTLBuffer;
     * the CPU kernels read the buffer contents IN PLACE (no copy, byte-
     * identical by construction). Scales are small — keep the pread path.
     * Any failure falls through to the pread paths below. */
    if (s->mio && g_metal_io && metalio_active()) {
        static const struct { const char *tag; int fmt; } probes[] = {
            { "merged_weight", 8 }, { "merged_i4", 4 }, { "merged_i3", 5 },
        };
        for (int pi = 0; pi < 3; pi++) {
            snprintf(nm, sizeof(nm), "%slayers.%d.mlp.experts.%d.%s",
                     g_prefix, layer, eid, probes[pi].tag);
            st_tensor *tw = st_find(&m->S, nm);
            if (!tw) continue;
            int64_t want_w = (probes[pi].fmt == 8) ? (2 * ng + nd)
                          : (probes[pi].fmt == 4) ? (2 * ((cc->hidden + 1) / 2) * cc->moe_inter
                                                      + ((cc->moe_inter + 1) / 2) * cc->hidden)
                          : (2 * qm_i3_rowbytes(cc->hidden) * cc->moe_inter
                             + qm_i3_rowbytes((int)cc->moe_inter) * cc->hidden);
            if (want_w < 0 || !tensor_numel_ok(tw->nbytes, want_w)) continue;
            /* scales ride the SAME event: raw-F32 qs tail in the same buffer.
             * Non-F32 scales keep the pread path (st_read_f32 converts). */
            snprintf(qsnm, sizeof(qsnm), "%slayers.%d.mlp.experts.%d.qs", g_prefix, layer, eid);
            st_tensor *ts = st_find(&m->S, qsnm);
            int64_t ngH = ((int64_t)cc->hidden + 63) / 64, ngI = ((int64_t)cc->moe_inter + 63) / 64;
            int64_t want_s = (probes[pi].fmt == 8) ? (cc->moe_inter + cc->moe_inter + cc->hidden)
                          : (cc->moe_inter * ngH * 2 + cc->hidden * ngI);
            if (!ts || ts->numel != want_s || ts->dtype != 2 ||           /* 2 = F32 */
                (uint64_t)ts->nbytes != (uint64_t)want_s * 4) continue;
            int mf = mio_file_for(tw->fd), mq = mio_file_for(ts->fd);
            if (mf < 0 || mq < 0) continue;
            size_t scale_off = ((size_t)tw->nbytes + 15u) & ~(size_t)15u;
            size_t slot_bytes = scale_off + (size_t)ts->nbytes;
            if (s->mio_slot < 0) s->mio_slot = metalio_slot_alloc(slot_bytes);
            if (s->mio_slot < 0) continue;
            ColiMetalioRegion regions[2] = {
                { mf, (uint64_t)tw->off, (size_t)tw->nbytes, 0 },
                { mq, (uint64_t)ts->off, (size_t)ts->nbytes, scale_off },
            };
            int64_t ev = metalio_loadv(s->mio_slot, regions, 2,
                                       g_mio_prefetching ? MIO_LOAD_SPEC
                                     : g_mio_async_issue ? MIO_LOAD_ASYNC
                                     : MIO_LOAD_DEMAND);
            if (ev <= 0) continue;
            s->mio_event = ev;
            if (!g_mio_prefetching && !g_mio_async_issue) {
                /* demand: block until the bytes are in the buffer */
                if (metalio_wait(ev) != 0) continue;
                s->mio_waited = ev;
            }
            /* prefetch (g_mio_prefetching): leave the load PENDING — the
             * demand path's hit-wait in expert_get drains it later; the
             * I/O overlaps compute meanwhile. */
            const unsigned char *base = (const unsigned char *)metalio_slot_ptr(s->mio_slot);
            if (!base) continue;
            float *sbase = (float *)(base + scale_off);    /* F32 scale tail */
            if (probes[pi].fmt == 8) {
                s->g = (int8_t *)base; s->u = s->g + ng; s->dd = s->g + ng + ng;
                s->gs = sbase; s->us = sbase + cc->moe_inter;
                s->ds = sbase + cc->moe_inter + cc->moe_inter;
                s->fmt = 8;
            } else {
                /* packed formats carry per-64-group scales (g4s/u4s/d4s) */
                int64_t rbH = (probes[pi].fmt == 5) ? qm_i3_rowbytes(cc->hidden)
                                                    : (cc->hidden + 1) / 2;
                s->g4 = (uint8_t *)base;
                s->u4 = s->g4 + rbH * cc->moe_inter;
                s->d4 = s->g4 + 2 * rbH * cc->moe_inter;
                s->g4s = sbase; s->u4s = sbase + cc->moe_inter * ngH;
                s->d4s = sbase + 2 * cc->moe_inter * ngH;
                s->fmt = probes[pi].fmt;
            }
            s->pinned = 0;
            s->mio_resident = 1;
            return;
        }
        /* fall through: any failure keeps the pread path */
    }
#endif

    /* int8 merged format (converter output, olmoe byte layout)? */
    snprintf(nm, sizeof(nm), "%slayers.%d.mlp.experts.%d.merged_weight", g_prefix, layer, eid);
    st_tensor *tw = st_find(&m->S, nm);
    if (tw) {
        int64_t twice, want_w;
        if (!i64_mul_ok(ng, 2, &twice) || !i64_add_ok(twice, nd, &want_w)) {
            fprintf(stderr, "%s: expected size overflows\n", nm); exit(1);
        }
        if (!tensor_numel_ok(tw->nbytes, want_w)) {
            fprintf(stderr, "%s: expert weight is %lld bytes — expected %lld, refusing\n",
                    nm, (long long)tw->nbytes, (long long)want_w); exit(1);
        }
        snprintf(qsnm, sizeof(qsnm), "%slayers.%d.mlp.experts.%d.qs", g_prefix, layer, eid);
        st_tensor *ts = st_find(&m->S, qsnm);
        int64_t want_s = cc->moe_inter + cc->moe_inter + cc->hidden;
        if (!ts || ts->numel != want_s) {
            fprintf(stderr, "%s: scale array is %lld elems — expected %lld, refusing\n",
                    qsnm, (long long)(ts ? ts->numel : -1), (long long)want_s); exit(1);
        }
        slot_alloc_q8(m, s);
        s->fmt = 8;
        st_read_raw(&m->S, nm, s->g, g_expert_drop);
        st_read_f32(&m->S, qsnm, s->gs, g_expert_drop);
        return;
    }
    /* i4-grouped packed (fmt=4): merged_i4 (2 values/byte) + qs (one f32 per
     * 64-input group per row) — the same merged g|u|d shape at ~half the bytes. */
    snprintf(nm, sizeof(nm), "%slayers.%d.mlp.experts.%d.merged_i4", g_prefix, layer, eid);
    tw = st_find(&m->S, nm);
    if (tw) {
        int64_t rbH = (cc->hidden + 1) / 2, rbI = (cc->moe_inter + 1) / 2;
        int64_t want_w = 2 * rbH * cc->moe_inter + rbI * cc->hidden;
        if (want_w < 0) { fprintf(stderr, "%s: expected size overflows\n", nm); exit(1); }
        if (!tensor_numel_ok(tw->nbytes, want_w)) {
            fprintf(stderr, "%s: expert weight is %lld bytes — expected %lld, refusing\n",
                    nm, (long long)tw->nbytes, (long long)want_w); exit(1);
        }
        snprintf(qsnm, sizeof(qsnm), "%slayers.%d.mlp.experts.%d.qs", g_prefix, layer, eid);
        st_tensor *ts = st_find(&m->S, qsnm);
        int64_t ngH = ((int64_t)cc->hidden + 63) / 64, ngI = ((int64_t)cc->moe_inter + 63) / 64;
        int64_t want_s = cc->moe_inter * ngH * 2 + cc->hidden * ngI;
        if (!ts || ts->numel != want_s) {
            fprintf(stderr, "%s: scale array is %lld elems — expected %lld, refusing\n",
                    qsnm, (long long)(ts ? ts->numel : -1), (long long)want_s); exit(1);
        }
        slot_alloc_packed(m, s, 4);
        st_read_raw(&m->S, nm, s->g4, g_expert_drop);
        st_read_f32(&m->S, qsnm, s->g4s, g_expert_drop);
        return;
    }
    /* int3-g64 packed (fmt=5): merged_i3 (24B per 64-input group) + qs. */
    snprintf(nm, sizeof(nm), "%slayers.%d.mlp.experts.%d.merged_i3", g_prefix, layer, eid);
    tw = st_find(&m->S, nm);
    if (tw) {
        int64_t rbH = qm_i3_rowbytes(cc->hidden), rbI = qm_i3_rowbytes((int)cc->moe_inter);
        int64_t want_w = 2 * rbH * cc->moe_inter + rbI * cc->hidden;
        if (want_w < 0) { fprintf(stderr, "%s: expected size overflows\n", nm); exit(1); }
        if (!tensor_numel_ok(tw->nbytes, want_w)) {
            fprintf(stderr, "%s: expert weight is %lld bytes — expected %lld, refusing\n",
                    nm, (long long)tw->nbytes, (long long)want_w); exit(1);
        }
        snprintf(qsnm, sizeof(qsnm), "%slayers.%d.mlp.experts.%d.qs", g_prefix, layer, eid);
        st_tensor *ts = st_find(&m->S, qsnm);
        int64_t ngH = ((int64_t)cc->hidden + 63) / 64, ngI = ((int64_t)cc->moe_inter + 63) / 64;
        int64_t want_s = cc->moe_inter * ngH * 2 + cc->hidden * ngI;
        if (!ts || ts->numel != want_s) {
            fprintf(stderr, "%s: scale array is %lld elems — expected %lld, refusing\n",
                    qsnm, (long long)(ts ? ts->numel : -1), (long long)want_s); exit(1);
        }
        slot_alloc_packed(m, s, 5);
        st_read_raw(&m->S, nm, s->g4, g_expert_drop);
        st_read_f32(&m->S, qsnm, s->g4s, g_expert_drop);
        return;
    }
    /* f32 per-expert format (fixture / --bits 32 converter output) */
    snprintf(nm, sizeof(nm), "%slayers.%d.mlp.experts.%d.gate_up_proj", g_prefix, layer, eid);
    st_tensor *tgu = st_find(&m->S, nm);
    if (!tgu) { fprintf(stderr, "missing %s\n", nm); exit(1); }
    int64_t twice_ng, want_gu, want_d;
    if (!i64_mul_ok(ng, 2, &twice_ng) || !i64_mul_ok(twice_ng, 4, &want_gu) ||
        !i64_mul_ok(nd, 4, &want_d)) {
        fprintf(stderr, "%s: expected size overflows\n", nm); exit(1);
    }
    if (!tensor_numel_ok(tgu->nbytes, want_gu)) {
        fprintf(stderr, "%s: %lld bytes — expected %lld, refusing\n",
                nm, (long long)tgu->nbytes, (long long)want_gu); exit(1);
    }
    snprintf(qsnm, sizeof(qsnm), "%slayers.%d.mlp.experts.%d.down_proj", g_prefix, layer, eid);
    st_tensor *td = st_find(&m->S, qsnm);
    if (!td || !tensor_numel_ok(td->nbytes, want_d)) {
        fprintf(stderr, "%s: %lld bytes — expected %lld, refusing\n",
                qsnm, (long long)(td ? td->nbytes : -1), (long long)want_d); exit(1);
    }
    slot_alloc_f32(s, ng, nd);
    s->fmt = 0;
    st_read_f32(&m->S, nm, s->gu, g_expert_drop);
    st_read_f32(&m->S, qsnm, s->d, g_expert_drop);
}

static void expert_get(Model *m, int layer, int eid, Slot **out){
    LCache *lc = &m->cache[layer];
    Cfg *c = &m->c;
    pthread_mutex_lock(&g_mx);
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        Slot *w = &lc->slots[i];
#ifdef COLI_METALIO
        /* MetalIO async prefetch: a resident slot may still have its load in
         * flight (mio_event > mio_waited). Wait only when genuinely needed —
         * for a successfully prefetched expert this returns ~immediately. */
        if (w->mio && w->mio_event > w->mio_waited) {
            int64_t ev = w->mio_event;
            pthread_mutex_unlock(&g_mx);
            metalio_wait(ev);
            pthread_mutex_lock(&g_mx);
            if (w->mio_waited < ev) w->mio_waited = ev;
        }
#endif
        m->hits++; w->used = ++m->clock; *out = w;
        pthread_mutex_unlock(&g_mx); return;
    }
    /* in-flight dedup: another request is already loading this expert — wait
     * for the publish instead of starting a second pread (the olmoe reload
     * storm: repeated picks of the same expert double-loaded under cache
     * pressure and small CACHE). */
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].loading_eid == eid) {
        Slot *w = &lc->slots[i];
        while (w->eid != eid) {
            pthread_mutex_unlock(&g_mx);
            struct timespec ts = {0, 1000000L}; nanosleep(&ts, NULL);
            pthread_mutex_lock(&g_mx);
        }
        m->hits++; w->used = ++m->clock; *out = w;
        pthread_mutex_unlock(&g_mx); return;
    }
    m->miss++;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];        /* buffers allocated by load_expert */
    } else {
        /* LRU eviction — skip pinned and in-flight (eid==-1) slots */
        int lru = -1;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
        }
        if (lru < 0)
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid < 0) continue;
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
        while (lru < 0) {   /* every slot in flight: wait for a publish */
            pthread_mutex_unlock(&g_mx);
            struct timespec ts = {0, 1000000L}; nanosleep(&ts, NULL);
            pthread_mutex_lock(&g_mx);
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid < 0) continue;
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
        }
        s = &lc->slots[lru];
        s->pinned = 0;
    }
    s->eid = -1;
    s->loading_eid = eid;
    pthread_mutex_unlock(&g_mx);

    double t0 = now_s();
    load_expert(m, layer, eid, s);          /* disk I/O outside the lock */
    m->t_expio += now_s() - t0;

    pthread_mutex_lock(&g_mx);
    s->eid = eid;
    s->loading_eid = -1;
    s->pinned = m->is_pinned[(size_t)layer * c->n_experts + eid];
    s->used = ++m->clock;
    pthread_mutex_unlock(&g_mx);
    *out = s;
}

/* ---------- HOT pinning (usage heatmap -> never-evict set) ---------- */

static void pin_hot_experts(Model *m){
    Cfg *c = &m->c;
    if (m->hot_n <= 0 || m->hot_pinned) return;
    m->hot_pinned = 1;
    int pinned_total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        uint32_t *freq_l = m->freq[l];
        if (!freq_l) continue;
        uint64_t layer_total = 0;
        for (int e = 0; e < c->n_experts; e++) layer_total += freq_l[e];
        if (layer_total == 0) continue;
        int hn = m->hot_n < c->n_experts ? m->hot_n : c->n_experts;
        if (hn > 256) hn = 256;
        int hot_eids[256], actual = 0;
        for (int k = 0; k < hn; k++) {
            int best = -1; uint32_t bv = 0;
            for (int e = 0; e < c->n_experts; e++) {
                int already = 0;
                for (int j = 0; j < k; j++) if (hot_eids[j] == e) { already = 1; break; }
                if (!already && freq_l[e] > bv) { bv = freq_l[e]; best = e; }
            }
            if (best < 0 || bv == 0) break;
            hot_eids[k] = best; actual++;
        }
        for (int k = 0; k < actual; k++) {
            int eid = hot_eids[k];
            m->is_pinned[(size_t)l * c->n_experts + eid] = 1;
            pthread_mutex_lock(&g_mx);
            LCache *lc = &m->cache[l];
            for (int i = 0; i < lc->n; i++)
                if (lc->slots[i].eid == eid) lc->slots[i].pinned = 1;
            pthread_mutex_unlock(&g_mx);
            pinned_total++;
        }
    }
    printf("[HOT] Pinned %d experts (top-%d/layer) after %d warmup tokens\n",
           pinned_total, m->hot_n, m->token_count);
}

/* ---------- full attention (GQA + QK-norm + partial RoPE + output gate) ---------- */

/* RoPE per-thread cache (opt-in QWEN_ROPE_CACHE=1, ChatGPT perf pass).
 *
 * The original rope_partial() recomputed powf(theta, -2j/rd) AND cosf/sinf
 * for every Q/K head of every layer of every token. The frequencies depend
 * only on geometry, and the trig row depends only on the token position, so
 * they are cached: 128 powf calls per token become 128 once, and the trig
 * calls drop from 18 heads x 128 per layer to 128 per token.
 *
 * Layout of one allocation (n = rd/2):
 *   buf[0..n)      inv[j] = theta^(-2j/rd)          — geometry key
 *   buf[n..2n)     cos(pos * inv[j])              — position key
 *   buf[2n..3n)    sin(pos * inv[j])
 *
 * Thread-local: decode runs on the main thread here, but __thread keeps the
 * cache safe if multiple host threads/modes are ever driven concurrently.
 */
typedef struct {
    float *buf;
    int cap;         /* allocated n capacity */
    int rd;          /* geometry used to build inv[] */
    float theta;
    int pos;         /* position used to build cos/sin */
    int pos_valid;
} RopeCache;

#if defined(__GNUC__) || defined(__clang__)
static __thread RopeCache g_rope_cache = { NULL, 0, -1, 0.0f, 0, 0 };
#else
static RopeCache g_rope_cache = { NULL, 0, -1, 0.0f, 0, 0 };
#endif

static int rope_cache_enabled(void){
    static int initialized = 0, enabled = 0;
    if (!initialized) {
        const char *s = getenv("QWEN_ROPE_CACHE");
        enabled = s && s[0] && strcmp(s, "0") != 0;
        initialized = 1;
    }
    return enabled;
}

/* Make sure inv[] and the per-position trig row are ready. Returns 1 on
 * success; 0 on allocation failure -> rope_partial falls back to the
 * original scalar implementation. */
static int rope_cache_prepare(RopeCache *rc, int pos, const Cfg *c){
    int rd = c->rotary_dim, n = rd / 2;
    if (n <= 0) return 0;
    if (rc->cap < n) {
        float *p = (float *)realloc(rc->buf, (size_t)3 * n * sizeof(float));
        if (!p) return 0;
        rc->buf = p; rc->cap = n;
        rc->rd = -1; rc->pos_valid = 0;   /* offsets changed: rebuild */
    }
    float *inv = rc->buf, *cs = rc->buf + n, *sn = rc->buf + 2 * n;
    if (rc->rd != rd || rc->theta != c->theta) {   /* geometry: once */
        for (int j = 0; j < n; j++)
            inv[j] = powf(c->theta, -2.0f * (float)j / (float)rd);
        rc->rd = rd; rc->theta = c->theta; rc->pos_valid = 0;
    }
    if (!rc->pos_valid || rc->pos != pos) {        /* position: once */
        float fp = (float)pos;
        for (int j = 0; j < n; j++) {
            float ang = fp * inv[j];
            cs[j] = cosf(ang); sn[j] = sinf(ang);
        }
        rc->pos = pos; rc->pos_valid = 1;
    }
    return 1;
}

/* Explicit per-thread cleanup; optional at process exit. */
static void rope_cache_free_current_thread(void){
    RopeCache *rc = &g_rope_cache;
    free(rc->buf);
    rc->buf = NULL; rc->cap = 0; rc->rd = -1; rc->theta = 0.0f;
    rc->pos = 0; rc->pos_valid = 0;
}

/* Drop-in replacement for the original rope_partial(). Default (env unset or
 * 0) is the exact original scalar path; QWEN_ROPE_CACHE=1 uses the cached
 * frequencies + per-position trig row, with NEON/AVX2 rotation where
 * available. Numerically identical per element (same cs/sn values, same
 * mul/sub ordering), so token output is unchanged. */
static void rope_partial(float *v, int pos, const Cfg *c){
    int rd = c->rotary_dim, n = rd / 2;
    if (n <= 0) return;
    if (rope_cache_enabled()) {
        RopeCache *rc = &g_rope_cache;
        if (rope_cache_prepare(rc, pos, c)) {
            const float *cs = rc->buf + n, *sn = rc->buf + 2 * n;
#ifdef __ARM_NEON
            int j = 0;
            for (; j + 4 <= n; j += 4) {   /* a=lo half, b=hi half */
                float32x4_t a = vld1q_f32(v + j), b = vld1q_f32(v + n + j);
                float32x4_t cv = vld1q_f32(cs + j), sv = vld1q_f32(sn + j);
                float32x4_t r0 = vfmsq_f32(vmulq_f32(a, cv), b, sv);
                float32x4_t r1 = vfmaq_f32(vmulq_f32(b, cv), a, sv);
                vst1q_f32(v + j, r0); vst1q_f32(v + n + j, r1);
            }
            for (; j < n; j++) {
                float a = v[j], b = v[j + n];
                v[j] = a * cs[j] - b * sn[j]; v[j + n] = b * cs[j] + a * sn[j];
            }
#elif defined(__AVX2__)
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 a = _mm256_loadu_ps(v + j), b = _mm256_loadu_ps(v + n + j);
                __m256 cv = _mm256_loadu_ps(cs + j), sv = _mm256_loadu_ps(sn + j);
#if defined(__FMA__)
                __m256 r0 = _mm256_fnmadd_ps(b, sv, _mm256_mul_ps(a, cv));
                __m256 r1 = _mm256_fmadd_ps(a, sv, _mm256_mul_ps(b, cv));
#else
                __m256 r0 = _mm256_sub_ps(_mm256_mul_ps(a, cv), _mm256_mul_ps(b, sv));
                __m256 r1 = _mm256_add_ps(_mm256_mul_ps(b, cv), _mm256_mul_ps(a, sv));
#endif
                _mm256_storeu_ps(v + j, r0); _mm256_storeu_ps(v + n + j, r1);
            }
            for (; j < n; j++) {
                float a = v[j], b = v[j + n];
                v[j] = a * cs[j] - b * sn[j]; v[j + n] = b * cs[j] + a * sn[j];
            }
#else
            for (int j = 0; j < n; j++) {
                float a = v[j], b = v[j + n];
                v[j] = a * cs[j] - b * sn[j]; v[j + n] = b * cs[j] + a * sn[j];
            }
#endif
            return;
        }
    }
    /* exact original scalar path (default / cache allocation failure) */
    for (int j = 0; j < n; j++) {
        float inv = powf(c->theta, -2.0f * j / rd);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = v[j], b = v[j + rd / 2];
        v[j] = a * cs - b * sn; v[j + rd / 2] = b * cs + a * sn;
    }
}

/* one token through one full_attention layer; K/V stored at pos BEFORE scoring
 * (causal attention includes the token's own key, matching transformers). */
static void attention_token(Model *m, Layer *l, int layer, const float *x, int pos, float *out){
    Cfg *c = &m->c; int H = c->n_heads, hd = c->head_dim, D = c->hidden;
    int kv = c->n_kv_heads, groups = H / kv;
    float *qg = falloc((int64_t)2 * H * hd);        /* per head: [q hd | gate hd] */
    float *k = falloc((int64_t)kv * hd), *vv = falloc((int64_t)kv * hd);
    wt_mul(qg, x, &l->q, 1, 2 * H * hd, D);
    wt_mul(k, x, &l->k, 1, kv * hd, D);
    wt_mul(vv, x, &l->v, 1, kv * hd, D);
    for (int h = 0; h < H; h++) {
        float *qh = qg + (int64_t)h * 2 * hd;       /* head h's q block */
        rmsnorm_row(qh, qh, l->qn, hd, c->eps);     /* QK-norm per head */
    }
    for (int g = 0; g < kv; g++)
        rmsnorm_row(k + (int64_t)g * hd, k + (int64_t)g * hd, l->kn, hd, c->eps);
    /* partial RoPE: rotate each q head block and each kv head ONCE (the old
     * per-h call rotated shared kv heads `groups` times, compounding the
     * angle — pos*groups rad instead of pos). */
    for (int h = 0; h < H; h++)
        rope_partial(qg + (int64_t)h * 2 * hd, pos, c);
    for (int g = 0; g < kv; g++)
        rope_partial(k + (int64_t)g * hd, pos, c);

    /* store K/V at pos (self-attention sees its own key) */
    for (int g = 0; g < kv; g++) {
        kv_store_row(m, layer, g, pos, k + (int64_t)g * hd, hd);
        kv_store_row_v(m, layer, g, pos, vv + (int64_t)g * hd, hd);
    }

    float *scores = falloc((int64_t)pos + 1);
    float *attn_out = falloc((int64_t)H * hd);
    float krow[512], vrow[512];
    if (hd > 512) { fprintf(stderr, "head_dim %d > 512\n", hd); exit(1); }
    float scale = 1.f / sqrtf((float)hd);
    for (int h = 0; h < H; h++) {
        const float *qh = qg + (int64_t)h * 2 * hd;
        /* keys/values come from the KV CACHE (past positions), not from the
         * current token's k/vv buffers — those hold only this token's head. */
        int hg = h / groups;
        float mx = -1e30f;
        for (int p = 0; p <= pos; p++) {
            kv_load_row(m, layer, hg, p, krow, hd);
            float acc = 0;
            for (int d = 0; d < hd; d++) acc += qh[d] * krow[d];
            scores[p] = acc * scale;
            if (scores[p] > mx) mx = scores[p];
        }
        float ssum = 0;
        for (int p = 0; p <= pos; p++) { scores[p] = expf(scores[p] - mx); ssum += scores[p]; }
        float *oh = attn_out + (int64_t)h * hd;
        for (int d = 0; d < hd; d++) oh[d] = 0;
        for (int p = 0; p <= pos; p++) {
            kv_load_row_v(m, layer, hg, p, vrow, hd);
            float w = scores[p] / ssum;
            for (int d = 0; d < hd; d++) oh[d] += w * vrow[d];
        }
        /* output gate: attn * sigmoid(gate) elementwise over the head dim
         * (gate is a full head_dim vector per head, second half of the block) */
        const float *gh = qg + (int64_t)(2 * h + 1) * hd;
        for (int d = 0; d < hd; d++) oh[d] *= 1.f / (1.f + expf(-gh[d]));
    }
    wt_mul(out, attn_out, &l->o, 1, D, H * hd);
    free(qg); free(k); free(vv); free(scores); free(attn_out);
}

/* ---------- Gated DeltaNet (linear_attention) ---------- */

static void l2norm(float *x, int D){
    double s = 0; for (int i = 0; i < D; i++) s += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)s + 1e-6f);
    for (int i = 0; i < D; i++) x[i] *= r;
}

/* one token through one GDN layer; updates conv + recurrence state */
static void gdn_token(Model *m, Layer *l, int layer, const float *x, float *out){
    Cfg *c = &m->c;
    int kd = c->lin_k_dim, kheads = c->lin_k_heads;
    int vd = c->lin_v_dim, vheads = c->lin_v_heads;
    int kdim = kd * kheads, vdim = vd * vheads, C = kdim * 2 + vdim, D = c->hidden;
    int kk = c->conv_kernel;

    float *qkv = falloc(C);
    wt_mul(qkv, x, &l->in_qkv, 1, C, D);
    float *a = falloc((int64_t)vheads), *b = falloc((int64_t)vheads);
    float *z = falloc(vdim);
    wt_mul(a, x, &l->in_a, 1, vheads, D);
    wt_mul(b, x, &l->in_b, 1, vheads, D);
    wt_mul(z, x, &l->in_z, 1, vdim, D);

    /* causal depthwise conv1d over channels with silu; state = last kk-1 */
    float *y = falloc(C);
    if (kk > 1) {
        float *conv_st = m->gdn_conv[layer];
        for (int ch = 0; ch < C; ch++) {
            float acc = 0;
            for (int j = 0; j < kk; j++) {
                /* HF mamba-style causal conv (F.conv1d + left pad): the
                 * kernel is applied REVERSED — w[kk-1] taps the current
                 * input (lag 0), w[j] taps state[j] (lag kk-1-j).
                 * state[idx] = x_{t-(kk-2)+idx} after each update. */
                float vv = (j == kk - 1) ? qkv[ch]
                                         : conv_st[ch * (kk - 1) + j];
                acc += l->conv1d[ch * kk + j] * vv;
            }
            y[ch] = silu(acc);
        }
        /* per-channel shift: state[ch][s] = x_{t-(kk-2)+s} (oldest first);
         * drop slot 0, append the current qkv at slot kk-2. */
        for (int ch = 0; ch < C; ch++) {
            for (int s = 0; s < kk - 2; s++)
                conv_st[ch * (kk - 1) + s] = conv_st[ch * (kk - 1) + s + 1];
            conv_st[ch * (kk - 1) + (kk - 2)] = qkv[ch];
        }
    } else {
        for (int ch = 0; ch < C; ch++) y[ch] = silu(l->conv1d[ch] * qkv[ch]);
    }

    const float *q_ = y, *k_ = y + kdim, *v_ = y + kdim * 2;
    int rep = vheads / kheads;
    if (rep < 1 || vheads % kheads) { fprintf(stderr, "GDN head ratio invalid\n"); exit(1); }

    float *S = m->gdn_S[layer];                    /* [vheads, kdim, vdim] */
    float *qh = falloc((int64_t)vheads * kd), *kh = falloc((int64_t)vheads * kd);
    float *vh = falloc((int64_t)vheads * vd);
    for (int h = 0; h < vheads; h++) {
        int khd = h / rep;
        for (int d = 0; d < kd; d++) { qh[(int64_t)h * kd + d] = q_[khd * kd + d]; kh[(int64_t)h * kd + d] = k_[khd * kd + d]; }
        for (int d = 0; d < vd; d++) vh[(int64_t)h * vd + d] = v_[h * vd + d];
        l2norm(qh + (int64_t)h * kd, kd);
        l2norm(kh + (int64_t)h * kd, kd);
        float sc = 1.f / sqrtf((float)kd);
        for (int d = 0; d < kd; d++) qh[(int64_t)h * kd + d] *= sc;
    }

    float *Snew = falloc((int64_t)vheads * kd * vd);
    float *kv_mem = falloc(vd);
    for (int h = 0; h < vheads; h++) {
        float ga = -expf(l->A_log[h]) * logf(1.f + expf(a[h] + l->dt_bias[h]));
        float gt = expf(ga);
        float bt = 1.f / (1.f + expf(-b[h]));
        const float *Sh = S + (int64_t)h * kd * vd;
        float *Sn = Snew + (int64_t)h * kd * vd;
        const float *qhh = qh + (int64_t)h * kd, *khh = kh + (int64_t)h * kd;
        const float *vhh = vh + (int64_t)h * vd;
        for (int d = 0; d < vd; d++) kv_mem[d] = 0;
        for (int kk2 = 0; kk2 < kd; kk2++) {
            const float *Srow = Sh + (int64_t)kk2 * vd;
            for (int d = 0; d < vd; d++) {
                float s = Srow[d] * gt;             /* decay */
                Sn[kk2 * vd + d] = s;
                kv_mem[d] += s * khh[kk2];
            }
        }
        for (int d = 0; d < vd; d++) {
            float delta = (vhh[d] - kv_mem[d]) * bt;
            for (int kk2 = 0; kk2 < kd; kk2++)
                Sn[kk2 * vd + d] += khh[kk2] * delta;
        }
        for (int d = 0; d < vd; d++) {
            float acc = 0;
            for (int kk2 = 0; kk2 < kd; kk2++) acc += Sn[kk2 * vd + d] * qhh[kk2];
            kv_mem[d] = acc;                        /* reuse: out_h */
        }
        /* write out_h for this head */
        for (int d = 0; d < vd; d++) vh[(int64_t)h * vd + d] = kv_mem[d];
    }
    memcpy(S, Snew, (size_t)vheads * kd * vd * sizeof(float));

    /* RMSNormGated: per-head over head_v_dim (weight is [head_v_dim]),
     * w * rmsnorm(o) * silu(z); then out_proj over the flattened heads. */
    float *normed = falloc(vdim);
    for (int h = 0; h < vheads; h++)
        rmsnorm_gated_row(normed + (int64_t)h * vd, vh + (int64_t)h * vd,
                          z + (int64_t)h * vd, l->gdn_norm, vd, c->eps);
    wt_mul(out, normed, &l->gdn_out, 1, D, vdim);

    free(qkv); free(a); free(b); free(z); free(y);
    free(qh); free(kh); free(vh); free(Snew); free(kv_mem); free(normed);
}

/* ---------- MoE ---------- */

/* Drain a slot's pending MetalIO load at use time (exact-async issue and
 * speculative prefetch both publish with the load in flight). No-op when
 * the data is already waited, pread-loaded, or MetalIO is off. */
static void expert_wait_ready(Model *m, Slot *s){
    (void)m;
#ifdef COLI_METALIO
    if (s->mio && s->mio_event > s->mio_waited) {
        metalio_wait(s->mio_event);
        s->mio_waited = s->mio_event;
    }
#endif
}

/* one expert applied to one token, result written into acc (caller zeroes) */
static void expert_apply(Model *m, Slot *s, const float *x, float *acc){
    Cfg *c = &m->c; int I = c->moe_inter, D = c->hidden;
    if (s->fmt == 4 || s->fmt == 5) {
        /* packed experts: i4-grouped (2 vals/byte, 64-group scales) or
         * int3-g64 (24B/64-group). Both stream from disk exactly like int8. */
        float *gate = falloc(I), *up = falloc(I);
        if (s->fmt == 4) {
            /* fused pair: one activation pass feeds both matrices (decode) */
            matmul_i4_grouped_pair(gate, up, x, s->g4, s->g4s, s->u4, s->u4s,
                                   D, I, 64);
        } else {
            matmul_i3(gate, x, s->g4, s->g4s, 1, D, I);
            matmul_i3(up,   x, s->u4, s->u4s, 1, D, I);
        }
        float *h = falloc(I);
        for (int i = 0; i < I; i++) h[i] = silu(gate[i]) * up[i];
        float *y = falloc(D);
        if (s->fmt == 4) matmul_i4_grouped(y, h, s->d4, s->d4s, 1, I, D, 64);
        else             matmul_i3(y, h, s->d4, s->d4s, 1, I, D);
        for (int d = 0; d < D; d++) acc[d] += y[d];
        free(gate); free(up); free(h); free(y);
    } else if (s->fmt == 8) {
        float *gate = falloc(I), *up = falloc(I);
        matmul_q8(gate, x, s->g, s->gs, 1, I, D);
        matmul_q8(up,   x, s->u, s->us, 1, I, D);
        float *h = falloc(I);
        for (int i = 0; i < I; i++) h[i] = silu(gate[i]) * up[i];
        float *y = falloc(D);
        matmul_q8(y, h, s->dd, s->ds, 1, D, I);
        for (int d = 0; d < D; d++) acc[d] += y[d];
        free(gate); free(up); free(h); free(y);
    } else {
        float *gu = falloc(2 * I);
        matmul(gu, x, s->gu, 1, 2 * I, D);
        float *h = falloc(I);
        for (int i = 0; i < I; i++) h[i] = silu(gu[i]) * gu[I + i];
        float *y = falloc(D);
        matmul(y, h, s->d, 1, D, I);
        for (int d = 0; d < D; d++) acc[d] += y[d];
        free(gu); free(h); free(y);
    }
}

/* ---------- MoE token loop ---------- */

/* Early next-layer expert I/O pipeline (ChatGPT perf pass, opt-in).
 * QWEN_PREFETCH=1 QWEN_PREFETCH_PIPE=1: issue layer l+1's predicted expert
 * loads right after layer l's router top-k is known — BEFORE layer l's own
 * expert GEMVs — so disk I/O overlaps current-layer compute. Only acts when
 * MetalIO async is live; the plain pread path falls through to the existing
 * post-layer lookahead in forward_token (early issue would add latency, not
 * hide it). Predictor unchanged: l's top-k predicts l+1's (routing is
 * correlated). Wrong guesses cost one pread, never residency: LRU evicts
 * them like any cold expert. */
static void expert_prefetch_next_early(Model *m, int layer, int nr){
    if (!g_prefetch || !g_prefetch_pipe) return;
    if (layer + 1 >= m->c.n_layers || nr <= 0 || nr > 64) return;
#ifdef COLI_METALIO
    if (!(g_metal_io && metalio_active())) return;   /* pread: stall, not overlap */
    int old = g_mio_prefetching;
    g_mio_prefetching = 1;                 /* enqueue loads, DO NOT wait */
    for (int i = 0; i < nr; i++) {
        Slot *ps;
        expert_get(m, layer + 1, m->last_route[i], &ps);
    }
    g_mio_prefetching = old;
#else
    (void)layer; (void)nr;
#endif
}

static void moe_token(Model *m, Layer *l, int layer, const float *x, float *out){
    Cfg *c = &m->c; int E = c->n_experts, K = c->topk, D = c->hidden;
    float *logits = falloc(E);
    wt_mul(logits, x, &l->router, 1, E, D);
    softmax_row(logits, E);

    /* top-k by probability (torch.topk on probs), deterministic tie-break:
     * lower index wins on exact ties. */
    int *idx = malloc(size_mul_or_die((size_t)E, sizeof(int), "router indices"));
    float *val = malloc(size_mul_or_die((size_t)E, sizeof(float), "router values"));
    if (!idx || !val) { fprintf(stderr, "OOM router selection\n"); exit(1); }
    for (int i = 0; i < E; i++) { idx[i] = i; val[i] = logits[i]; }
    for (int i = 0; i < K; i++) {
        int best = i;
        for (int j = i + 1; j < E; j++)
            if (val[j] > val[best] || (val[j] == val[best] && idx[j] < idx[best])) best = j;
        int ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
        float tv = val[i]; val[i] = val[best]; val[best] = tv;
    }
    float wsum = 0; for (int i = 0; i < K; i++) wsum += val[i];

    float *acc = calloc((size_t)D, sizeof(float));
    if (!acc) { fprintf(stderr, "OOM\n"); exit(1); }
    float *w = malloc(size_mul_or_die((size_t)K, sizeof(float), "router top-k weights"));
    if (!w) { fprintf(stderr, "OOM router weights\n"); exit(1); }
    for (int i = 0; i < K; i++) w[i] = val[i] / wsum;
    /* Exact-demand async issue: the router has already produced the EXACT
     * top-k for this layer — issue all K misses WITHOUT waiting (MetalIO
     * path: loads stay pending; pread path is inherently synchronous), then
     * drain each slot's event at apply time. Later loads overlap earlier
     * expert compute and the earlier waits — no prediction involved. */
    Slot **slots = malloc(size_mul_or_die((size_t)K, sizeof(Slot *), "router slots"));
    if (!slots) { fprintf(stderr, "OOM router slots\n"); exit(1); }
#ifdef COLI_METALIO
    g_mio_async_issue = g_metal_io && metalio_active();
#endif
    for (int i = 0; i < K; i++) expert_get(m, layer, idx[i], &slots[i]);
#ifdef COLI_METALIO
    g_mio_async_issue = 0;
#endif
    if (K <= 64) { memcpy(m->last_route, idx, (size_t)K * sizeof(int)); m->last_route_k = K; }
    /* Early pipeline: layer l+1's guesses go in flight BEFORE the layer-l
     * expert GEMVs below — overlap I/O with compute (MetalIO only). The
     * old post-layer lookahead in forward_token is disabled in pipe mode. */
    expert_prefetch_next_early(m, layer, K);
    int gpu_ok = 0;
#ifdef COLI_METAL
    /* QWEN_METAL_COMPUTE=1: all K routed experts in ONE Metal block
     * (gate/up/down per expert, then weighted scatter-add). Falls back to
     * the per-expert CPU loop below when the backend declines (unresolved
     * slab / unsupported fmt / GPU fault) — identical semantics either way. */
    if (g_metal_compute) {
        int fmt0 = slots[0]->fmt, unif = 1;
        for (int i = 1; i < K; i++) if (slots[i]->fmt != fmt0) { unif = 0; break; }
        if (unif && (fmt0 == 4 || fmt0 == 8)) {
            for (int i = 0; i < K; i++) expert_wait_ready(m, slots[i]);   /* MIO async loads must land before GPU reads them */
            const void *gp[64], *up[64], *dp[64];
            const float *gsp[64], *usp[64], *dsp[64];
            int xoff[64], nr[64], rows[64];
            float rw[64];
            if (K <= 64) {
                float *xg = falloc((int64_t)K * D);
                for (int i = 0; i < K; i++) {
                    Slot *s = slots[i];
                    if (fmt0 == 4) {
                        gp[i] = s->g4; up[i] = s->u4; dp[i] = s->d4;
                        gsp[i] = s->g4s; usp[i] = s->u4s; dsp[i] = s->d4s;
                    } else {
                        gp[i] = s->g; up[i] = s->u; dp[i] = s->dd;
                        gsp[i] = s->gs; usp[i] = s->us; dsp[i] = s->ds;
                    }
                    memcpy(xg + (int64_t)i * D, x, (size_t)D * sizeof(float));
                    xoff[i] = i; nr[i] = 1; rows[i] = 0;
                    rw[i] = w[i];
                }
                gpu_ok = coli_metal_moe_block(K, D, c->moe_inter, fmt0 == 4 ? 4 : 1, 64,
                                              gp, up, dp, gsp, usp, dsp,
                                              xg, xoff, nr, rows, rw, acc, 1);
                free(xg);
            }
        }
    }
#endif
    if (!gpu_ok) for (int i = 0; i < K; i++) {
        expert_wait_ready(m, slots[i]);
        float *y = calloc((size_t)D, sizeof(float));    /* expert_apply ACCUMULATES */
        if (!y) { fprintf(stderr, "OOM\n"); exit(1); }
        expert_apply(m, slots[i], x, y);
        for (int d = 0; d < D; d++) acc[d] += y[d] * w[i];
        free(y);
    }
    free(slots);
    /* routing telemetry: counts (HOT/COLI_USAGE) + ROUTE_TRACE stream */
    rt_route(layer, 0, idx, w, K);

    /* shared expert: silu(gate(x))*up(x) -> down; * sigmoid(se_g(x)) */
    float *sg = falloc(1);
    wt_mul(sg, x, &l->se_g, 1, 1, D);
    float gs = 1.f / (1.f + expf(-sg[0]));
    float *h = falloc(c->shared_inter);
    float *gv = falloc(c->shared_inter);
    wt_mul(gv, x, &l->se_gate, 1, c->shared_inter, D);
    wt_mul(h, x, &l->se_up, 1, c->shared_inter, D);
    for (int i = 0; i < c->shared_inter; i++) h[i] = silu(gv[i]) * h[i];
    float *sy = falloc(D);
    wt_mul(sy, h, &l->se_down, 1, D, c->shared_inter);
    for (int d = 0; d < D; d++) acc[d] += sy[d] * gs;
    memcpy(out, acc, (size_t)D * sizeof(float));

    free(logits); free(idx); free(val); free(acc); free(w);
    free(sg); free(h); free(gv); free(sy);
}

/* ---------- batched prefill (chunked layer-major) ---------- */

/* The numerics contract (see the batched-prefill ADR): matmul/matmul_q8 and
 * the packed kernels parallelize over OUTPUT ROWS with per-row accumulation
 * order untouched, so batching S>1 rows is bit-identical per row to S=1.
 * GDN recurrence, conv state and K/V writes stay strictly in token order.
 * Decode (n<=1) keeps the original per-token functions verbatim. */

#define QWEN_CHUNK_MAX 256
static int g_chunk = 64;             /* QWENMOE_CHUNK, clamped 1..QWEN_CHUNK_MAX */

static void embed_row(Model *m, int token, float *out){
    Cfg *c = &m->c; int D = c->hidden;
    if (m->embed.q) {
        const int8_t *row = m->embed.q + (int64_t)token * D;
        float sc = m->embed.s[token];
        for (int d = 0; d < D; d++) out[d] = (float)row[d] * sc;
    } else {
        memcpy(out, m->embed.f + (int64_t)token * D, (size_t)D * sizeof(float));
    }
}

/* batched full attention over a chunk: projections with S=C, per-token
 * QK-norm/RoPE/KV-store/scores/gate in token order, o_proj with S=C. */
static void attention_batch(Model *m, Layer *l, int layer, const float *xs, int C,
                            int pos0, float *out){
    Cfg *c = &m->c; int H = c->n_heads, hd = c->head_dim, D = c->hidden;
    int kv = c->n_kv_heads, groups = H / kv;
    float *qg = falloc((int64_t)C * 2 * H * hd);
    float *k = falloc((int64_t)C * kv * hd), *vv = falloc((int64_t)C * kv * hd);
    double t0 = now_s();
    wt_mul(qg, xs, &l->q, C, 2 * H * hd, D);
    wt_mul(k,  xs, &l->k, C, kv * hd, D);
    wt_mul(vv, xs, &l->v, C, kv * hd, D);
    float *attn_all = falloc((int64_t)C * H * hd);
    float scale = 1.f / sqrtf((float)hd);
    for (int g = 0; g < C; g++) {
        int pos = pos0 + g;
        float *qg_row = qg + (int64_t)g * 2 * H * hd;
        float *k_row = k + (int64_t)g * kv * hd;
        float *vv_row = vv + (int64_t)g * kv * hd;
        for (int h = 0; h < H; h++)
            rmsnorm_row(qg_row + (int64_t)h * 2 * hd, qg_row + (int64_t)h * 2 * hd, l->qn, hd, c->eps);
        for (int h = 0; h < kv; h++)
            rmsnorm_row(k_row + (int64_t)h * hd, k_row + (int64_t)h * hd, l->kn, hd, c->eps);
        for (int h = 0; h < H; h++) rope_partial(qg_row + (int64_t)h * 2 * hd, pos, c);
        for (int h = 0; h < kv; h++) rope_partial(k_row + (int64_t)h * hd, pos, c);
        for (int h = 0; h < kv; h++) {
            kv_store_row(m, layer, h, pos, k_row + (int64_t)h * hd, hd);
            kv_store_row_v(m, layer, h, pos, vv_row + (int64_t)h * hd, hd);
        }
        /* scores + weighted sum + gate, exactly as attention_token's row path */
        float *scores = falloc((int64_t)pos + 1);
        float *oh = attn_all + (int64_t)g * H * hd;
        float krow[512], vrow[512];
        for (int h = 0; h < H; h++) {
            const float *qh = qg_row + (int64_t)h * 2 * hd;
            int hg = h / groups;
            float mx = -1e30f;
            for (int p = 0; p <= pos; p++) {
                kv_load_row(m, layer, hg, p, krow, hd);
                float acc = 0;
                for (int d = 0; d < hd; d++) acc += qh[d] * krow[d];
                scores[p] = acc * scale;
                if (scores[p] > mx) mx = scores[p];
            }
            float ssum = 0;
            for (int p = 0; p <= pos; p++) { scores[p] = expf(scores[p] - mx); ssum += scores[p]; }
            float *ohh = oh + (int64_t)h * hd;
            for (int d = 0; d < hd; d++) ohh[d] = 0;
            for (int p = 0; p <= pos; p++) {
                kv_load_row_v(m, layer, hg, p, vrow, hd);
                float w = scores[p] / ssum;
                for (int d = 0; d < hd; d++) ohh[d] += w * vrow[d];
            }
            const float *gh = qg_row + (int64_t)(2 * h + 1) * hd;
            for (int d = 0; d < hd; d++) ohh[d] *= 1.f / (1.f + expf(-gh[d]));
        }
        free(scores);
    }
    float *outs = falloc((int64_t)C * D);
    wt_mul(outs, attn_all, &l->o, C, D, H * hd);
    for (int g = 0; g < C; g++)
        for (int d = 0; d < D; d++) out[(int64_t)g * D + d] += outs[(int64_t)g * D + d];
    m->t_attn += now_s() - t0;
    free(qg); free(k); free(vv); free(attn_all); free(outs);
}

/* GDN per-token core (conv + delta-rule recurrence + RMSNormGated), shared by
 * gdn_token and gdn_batch. Reads/writes conv + recurrence state in order. */
static void gdn_token_core(Model *m, Layer *l, int layer,
                           const float *qkv_row, float a, float b,
                           const float *z_row, float *y_out){
    Cfg *c = &m->c;
    int kd = c->lin_k_dim, kheads = c->lin_k_heads;
    int vd = c->lin_v_dim, vheads = c->lin_v_heads;
    int kdim = kd * kheads, vdim = vd * vheads, C = kdim * 2 + vdim;
    int kk = c->conv_kernel;

    /* causal depthwise conv1d over channels with silu; state = last kk-1 */
    float *y = falloc(C);
    if (kk > 1) {
        float *conv_st = m->gdn_conv[layer];
        for (int ch = 0; ch < C; ch++) {
            float acc = 0;
            for (int j = 0; j < kk; j++) {
                float vv = (j == kk - 1) ? qkv_row[ch] : conv_st[ch * (kk - 1) + j];
                acc += l->conv1d[ch * kk + j] * vv;
            }
            y[ch] = silu(acc);
        }
        for (int ch = 0; ch < C; ch++) {
            for (int s = 0; s < kk - 2; s++)
                conv_st[ch * (kk - 1) + s] = conv_st[ch * (kk - 1) + s + 1];
            conv_st[ch * (kk - 1) + (kk - 2)] = qkv_row[ch];
        }
    } else {
        for (int ch = 0; ch < C; ch++) y[ch] = silu(l->conv1d[ch] * qkv_row[ch]);
    }

    const float *q_ = y, *k_ = y + kdim, *v_ = y + kdim * 2;
    int rep = vheads / kheads;
    if (rep < 1 || vheads % kheads) { fprintf(stderr, "GDN head ratio invalid\n"); exit(1); }

    float *S = m->gdn_S[layer];                    /* [vheads, kdim, vdim] */
    float *qh = falloc((int64_t)vheads * kd), *kh = falloc((int64_t)vheads * kd);
    float *vh = falloc((int64_t)vheads * vd);
    for (int h = 0; h < vheads; h++) {
        int khd = h / rep;
        for (int d = 0; d < kd; d++) { qh[(int64_t)h * kd + d] = q_[khd * kd + d]; kh[(int64_t)h * kd + d] = k_[khd * kd + d]; }
        for (int d = 0; d < vd; d++) vh[(int64_t)h * vd + d] = v_[h * vd + d];
        l2norm(qh + (int64_t)h * kd, kd);
        l2norm(kh + (int64_t)h * kd, kd);
        float sc = 1.f / sqrtf((float)kd);
        for (int d = 0; d < kd; d++) qh[(int64_t)h * kd + d] *= sc;
    }

    float *Snew = falloc((int64_t)vheads * kd * vd);
    float *kv_mem = falloc(vd);
    for (int h = 0; h < vheads; h++) {
        float ga = -expf(l->A_log[h]) * logf(1.f + expf(a + l->dt_bias[h]));
        float gt = expf(ga);
        float bt = 1.f / (1.f + expf(-b));
        const float *Sh = S + (int64_t)h * kd * vd;
        float *Sn = Snew + (int64_t)h * kd * vd;
        const float *qhh = qh + (int64_t)h * kd, *khh = kh + (int64_t)h * kd;
        const float *vhh = vh + (int64_t)h * vd;
        for (int d = 0; d < vd; d++) kv_mem[d] = 0;
        for (int kk2 = 0; kk2 < kd; kk2++) {
            const float *Srow = Sh + (int64_t)kk2 * vd;
            for (int d = 0; d < vd; d++) {
                float s = Srow[d] * gt;
                Sn[kk2 * vd + d] = s;
                kv_mem[d] += s * khh[kk2];
            }
        }
        for (int d = 0; d < vd; d++) {
            float delta = (vhh[d] - kv_mem[d]) * bt;
            for (int kk2 = 0; kk2 < kd; kk2++)
                Sn[kk2 * vd + d] += khh[kk2] * delta;
        }
        for (int d = 0; d < vd; d++) {
            float acc = 0;
            for (int kk2 = 0; kk2 < kd; kk2++) acc += Sn[kk2 * vd + d] * qhh[kk2];
            kv_mem[d] = acc;
        }
        for (int d = 0; d < vd; d++) vh[(int64_t)h * vd + d] = kv_mem[d];
    }
    memcpy(S, Snew, (size_t)vheads * kd * vd * sizeof(float));

    for (int h = 0; h < vheads; h++)
        rmsnorm_gated_row(y_out + (int64_t)h * vd, vh + (int64_t)h * vd,
                          z_row + (int64_t)h * vd, l->gdn_norm, vd, c->eps);
    free(y); free(qh); free(kh); free(vh); free(Snew); free(kv_mem);
}

/* batched GDN layer: projections with S=C, per-token conv+recurrence in order,
 * out_proj with S=C. */
static void gdn_batch(Model *m, Layer *l, int layer, const float *xs, int C, float *out){
    Cfg *c = &m->c;
    int kd = c->lin_k_dim, kheads = c->lin_k_heads;
    int vd = c->lin_v_dim, vheads = c->lin_v_heads;
    int kdim = kd * kheads, vdim = vd * vheads, Cdim = kdim * 2 + vdim, D = c->hidden;
    double t0 = now_s();
    float *qkv = falloc((int64_t)C * Cdim);
    float *a = falloc((int64_t)C * vheads), *b = falloc((int64_t)C * vheads);
    float *z = falloc((int64_t)C * vdim);
    wt_mul(qkv, xs, &l->in_qkv, C, Cdim, D);
    wt_mul(a, xs, &l->in_a, C, vheads, D);
    wt_mul(b, xs, &l->in_b, C, vheads, D);
    wt_mul(z, xs, &l->in_z, C, vdim, D);
    float *y = falloc((int64_t)C * vdim);
    for (int g = 0; g < C; g++)
        gdn_token_core(m, l, layer, qkv + (int64_t)g * Cdim, a[g], b[g],
                       z + (int64_t)g * vdim, y + (int64_t)g * vdim);
    float *outs = falloc((int64_t)C * D);
    wt_mul(outs, y, &l->gdn_out, C, D, vdim);
    for (int g = 0; g < C; g++)
        for (int d = 0; d < D; d++) out[(int64_t)g * D + d] += outs[(int64_t)g * D + d];
    m->t_gdn += now_s() - t0;
    free(qkv); free(a); free(b); free(z); free(y); free(outs);
}

/* Prefill expert arena: per-chunk, per-layer scratch slots so a chunk's
 * experts load ONCE regardless of how many tokens route to them (the LRU's
 * 78% miss storm was re-reading the same experts every few tokens).
 * ponytail: arena capped at 64 slots/layer (evicting LRU-within-arena);
 * 64 * 3.1MB = ~200MB transient at 35B geometry — sized from RAM_GB if the
 * cap ever needs to grow past ~128. */
#define QWEN_ARENA_CAP 64             /* default wave/pool size */
#define QWEN_ARENA_CAP_MAX 256
static int g_arena_wave = QWEN_ARENA_CAP;   /* QWEN_ARENA_WAVE, clamped 8..256 */
typedef struct { int eid; int order; Slot s; } ArenaSlot;

/* Distinct routed expert set, first-appearance order: the arena builds the
 * FULL set before loading anything, so a layer routing more than
 * QWEN_ARENA_CAP distinct experts is processed in waves instead of evicting
 * (evict-during-build silently dropped routed contributions). */
static int qwen_arena_plan(const int *picks, int C, int K, int *uniq, int cap){
    int n = 0;
    for (int j = 0; j < C; j++) for (int k = 0; k < K; k++) {
        int e = picks[j * K + k], dup = 0;
        for (int i = 0; i < n; i++) if (uniq[i] == e) { dup = 1; break; }
        if (!dup && n < cap) uniq[n++] = e;
    }
    return n;
}

static void arena_free(ArenaSlot *a, int n){
    for (int i = 0; i < n; i++) {
        Slot *s = &a[i].s;
#ifdef COLI_METAL
        if (g_metal_compute) { coli_metal_unregister(s->g); coli_metal_unregister(s->g4); coli_metal_unregister(s->gs); coli_metal_unregister(s->g4s); }
#endif
        free(s->gu); free(s->g); free(s->g4); free(s->g4s);
        s->gu = NULL; s->g = NULL; s->g4 = NULL; s->g4s = NULL;
    }
}

static void moe_batch(Model *m, Layer *l, int layer, const float *xs, int C, float *out){
    Cfg *c = &m->c; int E = c->n_experts, K = c->topk, D = c->hidden;
    double t0 = now_s();
    /* router over the whole chunk */
    float *rlogits = falloc((int64_t)C * E);
    wt_mul(rlogits, xs, &l->router, C, E, D);
    int *picks = malloc(size_mul_or_die((size_t)C * K, sizeof(int), "moe picks"));
    float *w = malloc(size_mul_or_die((size_t)C * K, sizeof(float), "moe weights"));
    if (!picks || !w) { fprintf(stderr, "OOM moe picks\n"); exit(1); }
    for (int j = 0; j < C; j++) {
        float *row = rlogits + (int64_t)j * E;
        softmax_row(row, E);
        int *idx = malloc(size_mul_or_die((size_t)E, sizeof(int), "router indices"));
        if (!idx) { fprintf(stderr, "OOM router indices\n"); exit(1); }
        for (int i = 0; i < E; i++) idx[i] = i;
        for (int i = 0; i < K; i++) {
            int best = i;
            for (int jj = i + 1; jj < E; jj++)
                if (row[jj] > row[best] || (row[jj] == row[best] && idx[jj] < idx[best])) best = jj;
            int ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
            float tv = row[i]; row[i] = row[best]; row[best] = tv;
        }
        float wsum = 0; for (int i = 0; i < K; i++) wsum += row[i];
        for (int i = 0; i < K; i++) { picks[j * K + i] = idx[i]; w[j * K + i] = row[i] / wsum; }
        free(idx);
    }
    free(rlogits);
    if (K <= 64) { memcpy(m->last_route, picks, (size_t)K * sizeof(int)); m->last_route_k = K; }

    /* plan: the FULL distinct routed set first — never evict during build
     * (the old cap-64 arena silently dropped experts evicted before their
     * routed rows ran on layers routing >64 distinct experts). */
    int nuniq_cap = C * K < E ? C * K : E;   /* distinct <= picks <= E */
    int *uniq = malloc(size_mul_or_die((size_t)nuniq_cap, sizeof(int), "arena distinct"));
    int *eid_slot = malloc(size_mul_or_die((size_t)E, sizeof(int), "arena index"));
    if (!uniq || !eid_slot) { fprintf(stderr, "OOM arena plan\n"); exit(1); }
    for (int i = 0; i < E; i++) eid_slot[i] = -1;
    int nuniq = qwen_arena_plan(picks, C, K, uniq, nuniq_cap);
    for (int i = 0; i < nuniq; i++) eid_slot[uniq[i]] = i;

    /* per-(token, topk-position) scratch: every routed contribution is
     * computed exactly once, then accumulated token-major in k-order —
     * bit-exact against moe_token's reference order. */
    float *yrow = calloc(size_mul_or_die((size_t)C * K * D, sizeof(float), "arena y"), sizeof(float));
    if (!yrow) { fprintf(stderr, "OOM arena y scratch\n"); exit(1); }
    int I = c->moe_inter;
    float *xscratch = falloc((int64_t)C * D);
    float *yscratch = falloc((int64_t)C * D);
    int WAVE = g_arena_wave;
    if (WAVE < 8 || WAVE > QWEN_ARENA_CAP_MAX) WAVE = QWEN_ARENA_CAP;
#ifdef COLI_METALIO
    /* Persistent mio-backed arena pool: the same g_arena_wave physical
     * buffers are reused across every wave/layer/chunk, so long prefills
     * never churn MTLBuffers (reusable ids keep the active set bounded). */
    static Slot *pool = NULL;
    if (g_metal_io && metalio_active() && !pool) {
        pool = calloc((size_t)WAVE, sizeof(Slot));
        for (int i = 0; i < WAVE; i++) { pool[i].mio = 1; pool[i].mio_slot = -1; }
    }
#endif
    Slot *wave = calloc((size_t)QWEN_ARENA_CAP_MAX, sizeof(Slot));
    if (!wave) { fprintf(stderr, "OOM arena wave slots\n"); exit(1); }

    /* bounded waves: load up to g_arena_wave experts at a time; every expert
     * in a wave is applied before its slot is reused — no eviction, no drops.
     * MetalIO: the whole wave is issued without waiting and the NEXT wave's
     * load is enqueued into each slot right after its apply, so expert I/O
     * overlaps the current wave's compute. */
    for (int wb = 0; wb < nuniq; wb += WAVE) {
        int wn = nuniq - wb < WAVE ? nuniq - wb : WAVE;
        /* fresh per-wave descriptors ALWAYS: the previous wave's free loop
         * released heap buffers, and a reused Slot would make load_expert
         * treat the dangling pointer as already-allocated (use-after-free). */
        memset(wave, 0, (size_t)wn * sizeof(Slot));
#ifdef COLI_METALIO
        if (pool) g_mio_async_issue = 1;
#endif
        for (int a = 0; a < wn; a++) {
            Slot *s = &wave[a];
            double t0 = now_s();
#ifdef COLI_METALIO
            if (pool) {
                if (wb >= QWEN_ARENA_CAP) {
                    /* this expert was already pipelined into the pool slot
                     * during the previous wave's apply loop — the data
                     * pointers live in pool[a] (set by that pipeline);
                     * re-enqueueing would double-read the expert. */
                    *s = pool[a];
                    m->t_expio += now_s() - t0;
                    continue;
                }
                s->mio = 1; s->mio_slot = pool[a].mio_slot;
            }
#endif
            load_expert(m, layer, uniq[wb + a], s);
            m->t_expio += now_s() - t0;
#ifdef COLI_METALIO
            if (pool) pool[a].mio_slot = s->mio_slot;      /* persist for reuse */
#endif
        }
#ifdef COLI_METALIO
        if (pool) g_mio_async_issue = 0;
#endif
        for (int a = 0; a < wn; a++) {
            Slot *s = &wave[a];
            expert_wait_ready(m, s);
            int e = uniq[wb + a];
            int st = 0;
            for (int j = 0; j < C; j++) for (int k = 0; k < K; k++)
                if (picks[j * K + k] == e) {
                    memcpy(xscratch + (int64_t)st * D, xs + (int64_t)j * D, (size_t)D * sizeof(float));
                    st++;
                }
            if (s->fmt == 4 || s->fmt == 5) {
                float *gate = falloc((int64_t)st * I), *up = falloc((int64_t)st * I);
                if (s->fmt == 4) {
                    matmul_i4_grouped(gate, xscratch, s->g4, s->g4s, st, D, I, 64);
                    matmul_i4_grouped(up,   xscratch, s->u4, s->u4s, st, D, I, 64);
                } else {
                    matmul_i3(gate, xscratch, s->g4, s->g4s, st, D, I);
                    matmul_i3(up,   xscratch, s->u4, s->u4s, st, D, I);
                }
                float *h = falloc((int64_t)st * I);
                for (int r = 0; r < st; r++)
                    for (int i = 0; i < I; i++) h[(int64_t)r * I + i] = silu(gate[(int64_t)r * I + i]) * up[(int64_t)r * I + i];
                if (s->fmt == 4) matmul_i4_grouped(yscratch, h, s->d4, s->d4s, st, I, D, 64);
                else             matmul_i3(yscratch, h, s->d4, s->d4s, st, I, D);
                free(gate); free(up); free(h);
            } else if (s->fmt == 8) {
                float *gate = falloc((int64_t)st * I), *up = falloc((int64_t)st * I);
                matmul_q8(gate, xscratch, s->g, s->gs, st, I, D);
                matmul_q8(up,   xscratch, s->u, s->us, st, I, D);
                float *h = falloc((int64_t)st * I);
                for (int r = 0; r < st; r++)
                    for (int i = 0; i < I; i++) h[(int64_t)r * I + i] = silu(gate[(int64_t)r * I + i]) * up[(int64_t)r * I + i];
                matmul_q8(yscratch, h, s->dd, s->ds, st, D, I);
                free(gate); free(up); free(h);
            } else {
                float *gu = falloc((int64_t)st * 2 * I);
                matmul(gu, xscratch, s->gu, st, 2 * I, D);
                float *h = falloc((int64_t)st * I);
                for (int r = 0; r < st; r++)
                    for (int i = 0; i < I; i++) h[(int64_t)r * I + i] = silu(gu[(int64_t)r * 2 * I + i]) * gu[(int64_t)r * 2 * I + I + i];
                matmul(yscratch, h, s->d, st, D, I);
                free(gu); free(h);
            }
            /* store this expert's routed rows at their (token, topk) slots */
            int si = 0;
            for (int j = 0; j < C; j++) for (int k = 0; k < K; k++)
                if (picks[j * K + k] == e) {
                    memcpy(yrow + ((int64_t)j * K + k) * D, yscratch + (int64_t)si * D, (size_t)D * sizeof(float));
                    si++;
                }
            /* pipeline: the apply finished reading this slot — enqueue the
             * next wave's expert into it NOW, so its I/O overlaps the
             * remaining applies of this wave. */
#ifdef COLI_METALIO
            if (pool && wb + QWEN_ARENA_CAP < nuniq) {
                /* pipeline the next wave's expert into the SAME buffer right
                 * after the apply finished reading it; persist the full
                 * descriptor so the next wave's skip path finds the
                 * pointers + mio_resident state. */
                int ne = uniq[wb + QWEN_ARENA_CAP + a];
                Slot tmp; memset(&tmp, 0, sizeof(tmp));
                tmp.mio = 1; tmp.mio_slot = s->mio_slot;
                double t0 = now_s();
                g_mio_async_issue = 1;
                load_expert(m, layer, ne, &tmp);
                g_mio_async_issue = 0;
                m->t_expio += now_s() - t0;
                pool[a] = tmp;
            }
#endif
        }
        /* free heap buffers from any fallback (pread) wave slots; mio-RESIDENT
         * slots point into the persistent pool and have nothing to free */
        for (int a = 0; a < wn; a++) {
            Slot *s = &wave[a];
            if (s->mio_resident) continue;
#ifdef COLI_METAL
            if (g_metal_compute) { coli_metal_unregister(s->g4); coli_metal_unregister(s->g4s); }
#endif
            free(s->gu); free(s->g); free(s->gs); free(s->g4); free(s->g4s);
        }
    }
    free(uniq); free(eid_slot); free(wave); free(xscratch); free(yscratch);

    /* routed accumulation: token-major, k = 0..K-1 — moe_token's order */
    for (int j = 0; j < C; j++) {
        float *accrow = out + (int64_t)j * D;
        for (int k = 0; k < K; k++) {
            const float *y = yrow + ((int64_t)j * K + k) * D;
            float ww = w[j * K + k];
            for (int d = 0; d < D; d++) accrow[d] += y[d] * ww;
        }
    }
    free(yrow);

    /* shared expert AFTER routed experts (moe_token reference order) */
    float *sg_all = falloc(C), *h_all = falloc((int64_t)C * c->shared_inter);
    float *gv_all = falloc((int64_t)C * c->shared_inter);
    wt_mul(sg_all, xs, &l->se_g, C, 1, D);
    wt_mul(gv_all, xs, &l->se_gate, C, c->shared_inter, D);
    wt_mul(h_all, xs, &l->se_up, C, c->shared_inter, D);
    for (int j = 0; j < C; j++)
        for (int i = 0; i < c->shared_inter; i++)
            h_all[(int64_t)j * c->shared_inter + i] =
                silu(gv_all[(int64_t)j * c->shared_inter + i]) *
                h_all[(int64_t)j * c->shared_inter + i];
    float *sy_all = falloc((int64_t)C * D);
    wt_mul(sy_all, h_all, &l->se_down, C, D, c->shared_inter);
    for (int j = 0; j < C; j++) {
        float gs = 1.f / (1.f + expf(-sg_all[j]));
        for (int d = 0; d < D; d++) out[(int64_t)j * D + d] += sy_all[(int64_t)j * D + d] * gs;
    }
    free(sg_all); free(h_all); free(gv_all); free(sy_all);

    /* routing telemetry: one row per token (route_trace expects one record
     * per routed batch row, not the flattened C*K chunk). */
    for (int j = 0; j < C; j++)
        rt_route(layer, j, picks + j * K, w + j * K, K);
    m->t_moe += now_s() - t0;
    free(picks); free(w);
}

/* chunked layer-major prefill: returns logits for the last token */
static float *step_batched(Model *m, const int *ids, int n, int pos_base){
    Cfg *c = &m->c; int D = c->hidden;
    int C = g_chunk;
    if (C < 1) C = 1;
    if (C > QWEN_CHUNK_MAX) C = QWEN_CHUNK_MAX;
    float *hbuf = falloc((int64_t)C * D);
    float *normed = falloc((int64_t)C * D);
    int last_cj = 0;
    for (int j0 = 0; j0 < n; j0 += C) {
        int cj = n - j0 < C ? n - j0 : C;
        last_cj = cj;
        for (int j = 0; j < cj; j++) embed_row(m, ids[j0 + j], hbuf + (int64_t)j * D);
        for (int l = 0; l < c->n_layers; l++) {
            Layer *L = &m->L[l];
            for (int j = 0; j < cj; j++)
                rmsnorm_row(normed + (int64_t)j * D, hbuf + (int64_t)j * D, L->in_ln, D, c->eps);
            if (c->layer_is_gdn[l]) gdn_batch(m, L, l, normed, cj, hbuf);
            else                    attention_batch(m, L, l, normed, cj, j0, hbuf);
            for (int j = 0; j < cj; j++)
                rmsnorm_row(normed + (int64_t)j * D, hbuf + (int64_t)j * D, L->post_ln, D, c->eps);
            moe_batch(m, L, l, normed, cj, hbuf);
        }
        if (m->hot_n > 0) {
            m->token_count += cj;
            if (m->token_count >= m->warmup_tokens) pin_hot_experts(m);
        }
    }
    rmsnorm_row(normed, hbuf + (int64_t)(last_cj - 1) * D, m->final_norm, D, c->eps);
    float *logits = falloc(c->vocab);
    wt_mul(logits, normed, &m->lm_head, 1, c->vocab, D);
    free(hbuf); free(normed);
    return logits;
}

/* ---------- forward ---------- */

/* process one token at position `pos`, return output hidden state in out */
static void forward_token(Model *m, int token, int pos, float *out){
    Cfg *c = &m->c; int D = c->hidden;
    if (token < 0 || token >= c->vocab) {
        fprintf(stderr, "token id %d outside [0,%d)\n", token, c->vocab);
        exit(1);
    }
    if (pos < 0) { fprintf(stderr, "negative position %d\n", pos); exit(1); }
    if (pos >= m->max_t) {          /* K/V would overflow; refuse before corrupting */
        fprintf(stderr, "position %d >= CTX %d — raise CTX or shorten the prompt\n", pos, m->max_t);
        exit(1);
    }
    if (m->embed.q) {   /* row-int8 embed: dequant one row per token */
        const int8_t *row = m->embed.q + (int64_t)token * D;
        float sc = m->embed.s[token];
        for (int d = 0; d < D; d++) out[d] = (float)row[d] * sc;
    } else {
        memcpy(out, m->embed.f + (int64_t)token * D, (size_t)D * sizeof(float));
    }
    float *h = falloc(D);
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->L[l];
        float *normed = falloc(D);
        float *attn = falloc(D);
        double t0 = now_s();
        rmsnorm_row(normed, out, L->in_ln, D, c->eps);
        if (c->layer_is_gdn[l]) gdn_token(m, L, l, normed, attn);
        else                    attention_token(m, L, l, normed, pos, attn);
        for (int d = 0; d < D; d++) out[d] += attn[d];
        if (c->layer_is_gdn[l]) m->t_gdn += now_s() - t0;
        else                    m->t_attn += now_s() - t0;
        t0 = now_s();
        rmsnorm_row(normed, out, L->post_ln, D, c->eps);
        moe_token(m, L, l, normed, h);
        m->t_moe += now_s() - t0;
        for (int d = 0; d < D; d++) out[d] += h[d];
        /* ponytail: layer-lookahead prefetch — layer l's top-k predicts layer
         * l+1's (routing is correlated); QWEN_PREFETCH=1 opt-in, measured
         * via prefetch_misses. Wrong guesses cost one pread, never residency:
         * the LRU evicts them like any cold expert. */
        if (g_prefetch && !g_prefetch_pipe && l + 1 < c->n_layers && m->last_route_k > 0) {
            uint64_t before = m->miss;
#ifdef COLI_METALIO
            /* MetalIO async prefetch: enqueue the loads WITHOUT waiting; the
             * demand path's hit-wait drains the events (already signaled for
             * correct guesses -> ~0 wait). I/O overlaps compute meanwhile. */
            g_mio_prefetching = g_metal_io && metalio_active();
#endif
            for (int i = 0; i < m->last_route_k; i++) {
                Slot *ps;
                expert_get(m, l + 1, m->last_route[i], &ps);
            }
#ifdef COLI_METALIO
            g_mio_prefetching = 0;
#endif
            m->prefetch_misses += m->miss - before;
        }
        free(normed); free(attn);
    }
    free(h);
    if (m->hot_n > 0) {
        m->token_count++;
        if (m->token_count == m->warmup_tokens) pin_hot_experts(m);
    }
}

/* process n ids (prefill or decode), return logits for the next token */
static float *step(Model *m, const int *ids, int n, int pos_base){
    Cfg *c = &m->c; int D = c->hidden;
    if (!ids_in_vocab(c, ids, n, "input")) exit(1);
    if (pos_base < 0 || n > m->max_t - pos_base) {
        fprintf(stderr, "token positions [%d,%d) outside CTX %d\n",
                pos_base, pos_base + n, m->max_t);
        exit(1);
    }
    if (n > 1 && g_chunk > 1) return step_batched(m, ids, n, pos_base);
    float *h = falloc(D);
    for (int i = 0; i < n; i++) forward_token(m, ids[i], pos_base + i, h);
    float *normed = falloc(D);
    rmsnorm_row(normed, h, m->final_norm, D, c->eps);
    float *logits = falloc(c->vocab);
    wt_mul(logits, normed, &m->lm_head, 1, c->vocab, D);
    free(h); free(normed);
    return logits;
}

/* ---------- model init ---------- */

static void model_init(Model *m, const char *snap, int cap){
    memset(m, 0, sizeof(*m));
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();

    /* weight prefix probe: real checkpoints nest text weights under
     * model.language_model.*; the tiny fixtures use model.*; lm_head.weight
     * is always top-level. */
    g_prefix[0] = 0;
    {
        char probe[512];
        snprintf(probe, sizeof(probe), "model.language_model.embed_tokens.weight");
        if (st_have(m, probe))
            snprintf(g_prefix, sizeof(g_prefix), "model.language_model.");
        else {
            snprintf(probe, sizeof(probe), "model.embed_tokens.weight");
            if (st_have(m, probe))
                snprintf(g_prefix, sizeof(g_prefix), "model.");
        }
    }

    m->embed      = load_wt(m, "embed_tokens.weight", c->vocab, c->hidden);
    {   /* lm_head: top-level in the checkpoints; prefixed only in synthetic ones */
        char nm[512]; snprintf(nm, sizeof(nm), "lm_head.weight");
        if (!st_have(m, nm)) snprintf(nm, sizeof(nm), "%slm_head.weight", g_prefix);
        m->lm_head = load_wt_named(m, nm, c->vocab, c->hidden);
    }
    m->final_norm = load_t(m, "norm.weight", c->hidden);
    m->L = calloc(c->n_layers, sizeof(Layer));
    if (!m->L) { fprintf(stderr, "OOM layers\n"); exit(1); }
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        char nm[512];
        int D = c->hidden, H = c->n_heads, hd = c->head_dim, kv = c->n_kv_heads;
        int C = c->lin_k_dim * c->lin_k_heads * 2 + c->lin_v_dim * c->lin_v_heads;
        int vdim = c->lin_v_dim * c->lin_v_heads;
        /* load_t prepends g_prefix and reads f32; LDW probes <name>_q8/_qs */
        #define LD(field, suffix, N) snprintf(nm, sizeof(nm), "layers.%d." suffix, i); l->field = load_t(m, nm, N)
        #define LDW(field, suffix, O, I) snprintf(nm, sizeof(nm), "layers.%d." suffix, i); l->field = load_wt(m, nm, O, I)
        LD(in_ln, "input_layernorm.weight", D);
        LD(post_ln, "post_attention_layernorm.weight", D);
        if (c->layer_is_gdn[i]) {
            LD(A_log, "linear_attn.A_log", c->lin_v_heads);
            LD(dt_bias, "linear_attn.dt_bias", c->lin_v_heads);
            LD(conv1d, "linear_attn.conv1d.weight", (int64_t)C * c->conv_kernel);
            LDW(in_a, "linear_attn.in_proj_a.weight", c->lin_v_heads, D);
            LDW(in_b, "linear_attn.in_proj_b.weight", c->lin_v_heads, D);
            LDW(in_qkv, "linear_attn.in_proj_qkv.weight", C, D);
            LDW(in_z, "linear_attn.in_proj_z.weight", vdim, D);
            LD(gdn_norm, "linear_attn.norm.weight", c->lin_v_dim);
            LDW(gdn_out, "linear_attn.out_proj.weight", D, vdim);
        } else {
            LDW(q, "self_attn.q_proj.weight", 2 * H * hd, D);
            LDW(k, "self_attn.k_proj.weight", kv * hd, D);
            LDW(v, "self_attn.v_proj.weight", kv * hd, D);
            LDW(o, "self_attn.o_proj.weight", D, H * hd);
            LD(qn, "self_attn.q_norm.weight", hd);
            LD(kn, "self_attn.k_norm.weight", hd);
        }
        LDW(router, "mlp.gate.weight", c->n_experts, D);
        LDW(se_gate, "mlp.shared_expert.gate_proj.weight", c->shared_inter, D);
        LDW(se_up, "mlp.shared_expert.up_proj.weight", c->shared_inter, D);
        LDW(se_down, "mlp.shared_expert.down_proj.weight", D, c->shared_inter);
        LDW(se_g, "mlp.shared_expert_gate.weight", 1, D);
        #undef LD
        #undef LDW
    }
    m->cache = calloc_checked((size_t)c->n_layers, sizeof(LCache), "expert cache rows");
    for (int i = 0; i < c->n_layers; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc_checked((size_t)cap, sizeof(Slot), "expert cache slots");
        for (int s = 0; s < cap; s++) m->cache[i].slots[s].eid = -2;   /* empty */
    }
    rt_init("qwen3_moe", c->n_layers, c->n_experts);
    rt_drop_row(c->n_layers);                    /* every layer routes; no MTP row */
    m->freq = rt_counts_all();
    { const char *up = getenv("COLI_USAGE");
      if (up && *up) { int64_t h = rt_load(up);
        if (h > 0) fprintf(stderr, "[USAGE] expert history: %lld selections (%s)\n", (long long)h, up); } }
    m->hot_n         = getenv("HOT")    ? atoi(getenv("HOT"))    : 0;
    m->warmup_tokens = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    size_t pin_count = size_mul_or_die((size_t)c->n_layers, c->n_experts, "expert pin map");
    m->is_pinned = calloc_checked(pin_count, 1, "expert pin map");
    char pinpath[512]; snprintf(pinpath, sizeof(pinpath), "%s/hot_pinned.bin", snap);
    FILE *pinf = fopen(pinpath, "rb");
    if (pinf) {
        size_t want = pin_count;
        if (fread(m->is_pinned, 1, want, pinf) == want) {
            m->hot_pinned = 1;
            printf("[HOT] Loaded persistent pinning from %s\n", pinpath);
        }
        fclose(pinf);
    }
    m->dense_load_s = now_s() - t0;
}

/* ---------- modes ---------- */

static int *parse_ids(const char *s, int *n_out){
    int cap = 64, n = 0;
    int *ids = malloc(size_mul_or_die((size_t)cap, sizeof(int), "token-id input"));
    if (!ids) { fprintf(stderr, "OOM token-id input\n"); exit(1); }
    const char *p = s;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        char *end;
        errno = 0;
        long v = strtol(p, &end, 10);
        if (end == p || errno == ERANGE || v < INT_MIN || v > INT_MAX) {
            fprintf(stderr, "invalid token id near \"%.24s\"\n", p);
            free(ids); *n_out = -1; return NULL;
        }
        if (*end && *end != ' ') {
            fprintf(stderr, "invalid token-id separator near \"%.24s\"\n", end);
            free(ids); *n_out = -1; return NULL;
        }
        if (n == cap) {
            if (cap > INT_MAX / 2) { fprintf(stderr, "too many token ids\n"); free(ids); *n_out = -1; return NULL; }
            cap *= 2;
            int *grown = realloc(ids, size_mul_or_die((size_t)cap, sizeof(int), "token-id input"));
            if (!grown) { fprintf(stderr, "OOM token-id input\n"); free(ids); exit(1); }
            ids = grown;
        }
        ids[n++] = (int)v;
        p = end;
    }
    *n_out = n;
    return ids;
}

static void save_usage(Model *m, const char *snap){
    const char *up = getenv("COLI_USAGE");
    if (up && *up) rt_save(up, 0);
    if (m->hot_pinned) {
        char pinpath[512]; snprintf(pinpath, sizeof(pinpath), "%s/hot_pinned.bin", snap);
        FILE *f = fopen(pinpath, "wb");
        if (f) {
            size_t n = size_mul_or_die((size_t)m->c.n_layers, m->c.n_experts, "expert pin map");
            fwrite(m->is_pinned, 1, n, f);
            fclose(f);
        }
    }
}

/* QWENMOE_MODE=teacher: teacher-forced argmax per position */
static int mode_teacher(Model *m){
    const char *s = getenv("QWENMOE_TEACHER");
    if (!s || !*s) { fprintf(stderr, "QWENMOE_TEACHER required\n"); return 1; }
    int n; int *ids = parse_ids(s, &n);
    if (n < 1 || !ids_in_vocab(&m->c, ids, n, "QWENMOE_TEACHER")) {
        fprintf(stderr, "invalid QWENMOE_TEACHER\n"); free(ids); return 1;
    }
    for (int i = 0; i < n; i++) {
        float *logits = step(m, ids + i, 1, i);     /* prefix grows via KV */
        int best = qwen_pick_token(m, logits, -1);
        printf("PRED %d\n", best);
        logits_free(&logits);
    }
    free(ids);
    return 0;
}

/* QWENMOE_MODE=greedy: generate from prompt ids */
static int mode_greedy(Model *m){
    const char *ps = getenv("QWENMOE_PROMPT_IDS");
    if (!ps || !*ps) { fprintf(stderr, "QWENMOE_PROMPT_IDS required\n"); return 1; }
    int max_new = getenv("QWENMOE_MAX_NEW") ? atoi(getenv("QWENMOE_MAX_NEW")) : 8;
    if (max_new < 1 || max_new > 4096) { fprintf(stderr, "QWENMOE_MAX_NEW out of range\n"); return 1; }
    int np; int *ids = parse_ids(ps, &np);
    if (np < 1 || !ids_in_vocab(&m->c, ids, np, "QWENMOE_PROMPT_IDS")) {
        fprintf(stderr, "invalid QWENMOE_PROMPT_IDS\n"); free(ids); return 1;
    }
    float *logits = step(m, ids, np, 0);
    for (int s = 0; s < max_new; s++) {
        int best = qwen_pick_token(m, logits, -1);
        logits_free(&logits);
        if (best == m->c.eos) break;
        printf("ID %d\n", best);
        logits = step(m, &best, 1, np + s);
    }
    logits_free(&logits);
    free(ids);
    return 0;
}

/* Special-token id by content: added tokens are atomic in tok.h's encode
 * (sp[] sorted by length desc), so <|im_start|> etc. map to their ids.
 * Fallback to the vocab map for tokenizers that keep them there only. */
static int tok_sp_id(Tok *T, const char *s){
    int n = (int)strlen(s);
    for (int i = 0; i < T->nsp; i++)
        if (T->sp[i].len == n && !memcmp(T->sp[i].str, s, n)) return T->sp[i].id;
    return hm_get(&T->vocab, s, n);
}

/* Independent requests always restart sequence-dependent GDN state.  KV is
 * intentionally retained: a position-zero prefill overwrites every position
 * that causal attention can read for the new request. */
static void request_state_reset(Model *m){
    for (int i = 0; i < m->c.n_layers; i++) {
        if (m->gdn_S && m->gdn_S[i])
            memset(m->gdn_S[i], 0, size_mul_or_die(gdn_state_count(&m->c), sizeof(float), "GDN recurrence reset"));
        if (m->gdn_conv && m->gdn_conv[i])
            memset(m->gdn_conv[i], 0, size_mul_or_die(gdn_conv_count(&m->c), sizeof(float), "GDN convolution reset"));
    }
}

/* Full session/test reset also clears KV, unlike the narrow serve reset. */
static void state_reset(Model *m){
    request_state_reset(m);
    size_t n = size_mul_or_die(kv_state_count(&m->c, m->max_t), sizeof(float), "KV cache reset");
    for (int i = 0; i < m->c.n_layers; i++) {
        if (g_kv_f16) {
            if (m->K16 && m->K16[i]) memset(m->K16[i], 0, n / 2);
            if (m->V16 && m->V16[i]) memset(m->V16[i], 0, n / 2);
        } else {
            if (m->K && m->K[i]) memset(m->K[i], 0, n);
            if (m->V && m->V[i]) memset(m->V[i], 0, n);
        }
    }
}

static void chat_feed_stop(Model *m, int token, int *hist, int *hpos){
    int pos = *hpos;
    hist[pos] = token;
    *hpos = pos + 1;
    float *discard = step(m, &token, 1, pos);
    logits_free(&discard);
}

/* CHAT=1: interactive multi-turn chat (prompt on stdin, decoded stream).
 * Each turn is framed with the Qwen ChatML template (system turn once per
 * session, <|im_start|>user/assistant, <think> prefix), and history stays
 * in the KV cache so turns continue each other. Env:
 *   QWEN_THINK=0  -> disable the <think> reasoning prefix (answer only)
 *   QWEN_SYSTEM   -> system prompt text; empty string disables the turn
 *   COLI_TEMP/NUCLEUS -> sampling (defaults 0.7 / 0.9; chat, not greedy) */
static int mode_chat(Model *m, Tok *T){
    stops_arm_tok(&m->c, m->c.eos, T);
    int max_new = getenv("MAX_NEW") ? atoi(getenv("MAX_NEW")) : 1024;
    const char *ct = getenv("COLI_TEMP");
    if (ct) g_temp = (float)atof(ct);
    const char *nc = getenv("NUCLEUS");
    if (nc) g_nuc = (float)atof(nc);
    if (!ct) g_temp = 0.7f;
    if (!nc) g_nuc = 0.9f;
    int thinking = getenv("QWEN_THINK") ? atoi(getenv("QWEN_THINK")) : 1;
    const char *sys = getenv("QWEN_SYSTEM");
    if (!sys) sys = "You are Qwen, a helpful AI assistant.";
    int i_im_start = tok_sp_id(T, "<|im_start|>");
    int i_im_end   = tok_sp_id(T, "<|im_end|>");
    if (i_im_start < 0 || i_im_end < 0) {
        fprintf(stderr, "tokenizer.json: missing <|im_start|>/<|im_end|> added tokens\n");
        return 1;
    }
    int *hist = malloc(size_mul_or_die((size_t)m->max_t, sizeof(int), "chat history"));
    if (!hist) { fprintf(stderr, "OOM\n"); return 1; }
    int hpos = 0;
    int sys_ids[4096]; int sys_n = 0;
    if (sys && *sys) {
        char sseg[8192];
        int sl = snprintf(sseg, sizeof(sseg), "<|im_start|>system\n%s<|im_end|>\n", sys);
        sys_n = tok_encode(T, sseg, sl, sys_ids, 4096);
        if (sys_n > 0) { memcpy(hist, sys_ids, (size_t)sys_n * sizeof(int)); hpos = sys_n; }
    }
    printf("resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m->dense_load_s, rss_gb());
    printf("Qwen chat ready (temp %.2f nuc %.2f think %d; Ctrl-D to quit)\n", g_temp, g_nuc, thinking);
    char line[8192];
    while (fgets(line, sizeof(line), stdin)) {
        int n = (int)strlen(line);
        while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
        if (!n) continue;
        char seg[16384]; int sl = 0;
        sl += snprintf(seg + sl, sizeof(seg) - sl, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", line);
        sl += snprintf(seg + sl, sizeof(seg) - sl,
                       thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
        int ids[16384];
        int np = tok_encode(T, seg, sl, ids, 16384);
        if (np <= 0) { printf("(empty prompt)\n"); continue; }
        if (hpos + np > m->max_t) {
            state_reset(m);
            hpos = 0;
            printf("[context full: session restarted]\n");
            if (sys_n > 0) { step(m, sys_ids, sys_n, 0); hpos = sys_n; }
        }
        float *logits = step(m, ids, np, hpos);
        memcpy(hist + hpos, ids, (size_t)np * sizeof(int));
        hpos += np;
        char buf[1024];
        for (int s = 0; s < max_new; s++) {
            if (hpos >= m->max_t) { printf("\n[context full]\n"); break; }
            int nt = qwen_pick_token(m, logits, -1);
            logits_free(&logits);
            if (is_stop(nt)) {
                /* Keep the turn boundary in context: feed the stop token
                 * (the model chose it; for <|im_end|> it IS the template's
                 * turn terminator). Without this the assistant turn stays
                 * open in the KV and the next turn opens incoherently. */
                chat_feed_stop(m, nt, hist, &hpos);
                break;
            }
            int pos = hpos;
            hist[hpos++] = nt;
            int nb = tok_decode(T, &nt, 1, buf, sizeof(buf) - 1);
            buf[nb] = 0;
            fwrite(buf, 1, (size_t)nb, stdout);
            fflush(stdout);
            logits = step(m, &nt, 1, pos);
        }
        logits_free(&logits);
        printf("\n");
    }
    free(hist);
    return 0;
}

/* ---------- serve mode: openai_server.py engine protocol ----------
 * stdin:  SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>\n<payload>\n
 *         CANCEL <id>\n
 * stdout: READY sentinel once loaded, then per request a stream of
 *         DATA <id> <size>\n<bytes>\n frames and a final
 *         DONE <id> STAT <tok> <tps> <hit%> <rss> <prompt_tok> <len_limited>\n
 * Byte-identical to colibri.c's serve protocol (inkling.c documents it in
 * full above its own SUBMIT handling) so the shared openai_server.py gateway
 * drives this engine unchanged.
 *
 * v1 scope, same as olmoe: one request in flight, full re-prefill every turn,
 * no cross-request KV reuse. The payload arrives already rendered by
 * openai_server.py's render_chat_qwen — this engine tokenizes it as-is.
 * Expert-cache contents, route counts, and immutable weights persist.  Before
 * each prefill we clear GDN recurrence/conv state; KV storage need not be
 * cleared because position-zero prefill overwrites every position attention
 * reads for the new request. */
typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen; } SReq;
#define SRV_QMAX 16
static SReq g_q[SRV_QMAX]; static int g_qn = 0;

/* read one control line (+ payload for SUBMIT). cur_id: request in flight;
 * returns 1 if that request was cancelled, 0 otherwise, -1 on stdin EOF. */
static int serve_read_cmd(const char *cur_id) {
    char ln[512];
    if (!fgets(ln, sizeof(ln), stdin)) return -1;
    char cmd[16], id[64];
    if (sscanf(ln, "%15s %63s", cmd, id) < 2) return 0;
    if (!strcmp(cmd, "CANCEL")) return cur_id && !strcmp(id, cur_id);
    if (!strcmp(cmd, "SUBMIT")) {
        int slot, plen, max_tok; float temp, top_p;
        int nf = sscanf(ln, "%*s %*s %d %d %d %f %f", &slot, &plen, &max_tok, &temp, &top_p);
        if (nf < 5 || plen < 0 || plen > (1<<22) || max_tok < 1 || max_tok > (1<<20)) {
            printf("ERROR %s bad submit header\n", id); fflush(stdout); return 0; }
        (void)slot;
        char *pl = malloc((size_t)plen + 1);
        if (fread(pl, 1, (size_t)plen, stdin) != (size_t)plen) { free(pl); return -1; }
        pl[plen] = 0;
        int nl = fgetc(stdin); (void)nl;
        if (g_qn < SRV_QMAX) {
            SReq *q = &g_q[g_qn++];
            snprintf(q->id, sizeof(q->id), "%s", id);
            q->max_tok = max_tok; q->temp = temp; q->top_p = top_p;
            q->payload = pl; q->plen = plen;
        } else { printf("ERROR %s queue full\n", id); fflush(stdout); free(pl); }
    }
    return 0;
}

static void serve_one(Model *m, Tok *T, SReq *q, int ctx_cap) {
    int cap = q->plen + 16;
    int *ids = malloc(size_mul_or_die((size_t)cap, sizeof(int), "serve prompt ids"));
    if (!ids) { fprintf(stderr, "OOM serve prompt ids\n"); exit(1); }
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { printf("ERROR %s empty prompt\n", q->id); fflush(stdout); free(ids); return; }
    if (np + q->max_tok > ctx_cap) {
        printf("ERROR %s context exceeds CTX (%d + %d > %d)\n", q->id, np, q->max_tok, ctx_cap);
        fflush(stdout); free(ids); return;
    }
    g_temp = q->temp; g_nuc = q->top_p;
    m->t_attn = m->t_gdn = m->t_moe = m->t_expio = 0;
    double t0 = now_s();
    uint64_t h0 = m->hits, m0 = m->miss;
    /* ACCEPT before prefill: the server commits the streaming 200 here and,
     * critically, stops polling `cancelled()` (connected flips True), so a
     * long prefill no longer looks like a dead client and draws a spurious
     * CANCEL that would arrive mid-turn and truncate the reply. */
    printf("ACCEPT %s %d\n", q->id, np); fflush(stdout);
    request_state_reset(m);
    float *logit = step(m, ids, np, 0);
    int hist_len = np, gen = 0, limited = 1, cancelled = 0;
    char buf[512];
    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        int nt = qwen_pick_token(m, logit, -1);
        logits_free(&logit);
        if (is_stop(nt)) { limited = 0; break; }
        int nb = tok_decode(T, &nt, 1, buf, sizeof(buf)-1);
        printf("DATA %s %d\n", q->id, nb);
        fwrite(buf, 1, (size_t)nb, stdout);
        fputc('\n', stdout); fflush(stdout);
        gen++; hist_len++;
        while (coli_stdin_readable()) {
            int r = serve_read_cmd(q->id);
            if (r < 0) { free(ids); return; }
            if (r > 0) { cancelled = 1; limited = 0; }
        }
        if (cancelled || s == q->max_tok - 1 || hist_len >= ctx_cap) break;
        logit = step(m, &nt, 1, hist_len - 1);
    }
    logits_free(&logit);
    double dt = now_s() - t0;
    double tot = (double)(m->hits - h0 + m->miss - m0);
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", q->id, gen,
           dt > 0 ? gen/dt : 0.0, tot ? 100.0*(m->hits-h0)/tot : 0.0, rss_gb(), np, limited);
    /* PROF: per-turn phase timings for the dashboard (field order matches
     * colibri.c / openai_server.py's PROF parser: wall, prompt, completion,
     * expert_disk, expert_wait, expert_matmul, attention, other, forwards). */
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %d\n",
           dt, np, gen, m->t_expio, 0.0, m->t_moe, m->t_attn, m->t_gdn, gen + 1);
    fflush(stdout);
    free(ids);
}

/* dashboard HWINFO/TIERS/EMAP: same lines the other serve-capable engines
 * emit for the web dashboard's hardware panel and Brain page (CPU-only, so
 * the GPU fields are always empty). */
static void serve_hwinfo(Model *m) {
    (void)m;
    char cpu[256] = ""; int cores = 0; double rt = 0, ra = 0;
    FILE *ci = fopen("/proc/cpuinfo", "r");
    if (ci) { char ln[256];
        while (fgets(ln, sizeof(ln), ci)) if (!strncmp(ln, "model name", 10)) {
            char *p = strchr(ln, ':'); if (p) { p++; while (*p == ' ') p++;
            int n = (int)strlen(p); if (n > 0 && p[n-1] == '\n') p[--n] = 0;
            snprintf(cpu, sizeof(cpu), "%s", p); } break; }
        fclose(ci); }
#ifdef _SC_NPROCESSORS_ONLN
    cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    FILE *mi = fopen("/proc/meminfo", "r");
    if (mi) { char ln[256]; double v = 0;
        while (fgets(ln, sizeof(ln), mi)) {
            if (sscanf(ln, "MemTotal: %lf", &v) == 1) rt = v/1e6;
            if (sscanf(ln, "MemAvailable: %lf", &v) == 1) ra = v/1e6;
        } fclose(mi); }
    printf("HWINFO %d %.1f %.1f 0 0.0 %s|\n", cores, rt, ra, cpu[0] ? cpu : "unknown");
    fflush(stdout);
}

static void serve_tiers_emap(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts;
    int filled = 0;
    for (int i = 0; i < c->n_layers; i++) filled += m->cache[i].n;
    int64_t I = c->moe_inter, D = c->hidden;
    /* per-expert resident bytes: int8 gate/up/down + one f32 scale per row */
    int64_t slotb = 3*I*D + (2*I+D)*4;
    int64_t total_experts = (int64_t)c->n_layers * E;
    printf("TIERS 0 %d %lld 0.00 %.2f\n", filled,
           (long long)(total_experts - filled), filled*(double)slotb/1e9);
    /* EMAP: 1 byte/expert hex — tier(2b: 0=disk 1=RAM)<<6 | heat(6b: log2 usage) */
    size_t hex_n = size_mul_or_die((size_t)c->n_layers, E, "expert map");
    hex_n = size_mul_or_die(hex_n, 2, "expert map hex");
    if (!size_add_ok(hex_n, 1, &hex_n)) { fprintf(stderr, "expert map size overflow\n"); exit(1); }
    char *hex = malloc(hex_n); size_t w = 0;
    if (!hex) { fprintf(stderr, "OOM expert map\n"); exit(1); }
    for (int i = 0; i < c->n_layers; i++) {
        LCache *lc = &m->cache[i];
        for (int e = 0; e < E; e++) {
            int tier = 0;
            for (int z = 0; z < lc->n; z++) if (lc->slots[z].eid == e) { tier = 1; break; }
            uint32_t u = m->freq[i] ? m->freq[i][e] : 0;
            int heat = 0; while (u) { heat++; u >>= 1; } if (heat > 63) heat = 63;
            int b = (tier << 6) | heat;
            hex[w++] = "0123456789abcdef"[b >> 4];
            hex[w++] = "0123456789abcdef"[b & 15];
        }
    }
    hex[w] = 0;
    printf("EMAP %d %d %s\n", c->n_layers, E, hex);
    fflush(stdout); free(hex);
}

static void serve_loop(Model *m, Tok *T, int ctx_cap) {
    coli_serve_binary_mode();
    setvbuf(stdin, NULL, _IONBF, 0);
    stops_arm_tok(&m->c, m->c.eos, T);
    /* Batched-serve stop filtering keeps ONLY eos (#401 tool-call safety);
     * Qwen turns end with <|im_end|>, so re-arm it or every response runs
     * into the next hallucinated turn. */
    int im_end = tok_sp_id(T, "<|im_end|>");
    if (im_end >= 0 && !is_stop(im_end) && g_nstop < 64) g_stop[g_nstop++] = im_end;
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", rss_gb());
    fflush(stdout);
    serve_hwinfo(m);
    serve_tiers_emap(m);
    for (;;) {
        while (!g_qn) if (serve_read_cmd(NULL) < 0) return;
        SReq q = g_q[0];
        memmove(g_q, g_q + 1, (size_t)(--g_qn) * sizeof(SReq));
        serve_one(m, T, &q, ctx_cap);
        free(q.payload);
    }
}

static int mode_serve(Model *m, Tok *T){
    serve_loop(m, T, m->max_t);
    return 0;
}

static int json_token_array_ok(jval *a, int vocab, const char *name, int ci){
    if (!a || a->t != J_ARR) {
        fprintf(stderr, "ref.json: case %d %s must be an array\n", ci, name);
        return 0;
    }
    for (int i = 0; i < a->len; i++) {
        jval *v = a->kids[i];
        if (!v || v->t != J_NUM || !isfinite(v->num) || trunc(v->num) != v->num ||
            v->num < 0 || v->num >= vocab) {
            fprintf(stderr, "ref.json: case %d %s[%d] is not an id in [0,%d)\n",
                    ci, name, i, vocab);
            return 0;
        }
    }
    return 1;
}

/* default: self-test every ref.json case in the model dir */
static int mode_selftest(Model *m, const char *snap){
    char refpath[2048]; snprintf(refpath, sizeof(refpath), "%s/ref.json", snap);
    FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1); if (!buf) { fprintf(stderr, "OOM\n"); return 1; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 1; }
    buf[n] = 0; fclose(f);
    char *arena = NULL; jval *ref = json_parse(buf, &arena);
    jval *cases = json_get(ref, "cases");
    if (!cases || (cases->t != J_OBJ && cases->t != J_ARR)) { fprintf(stderr, "ref.json: no cases\n"); return 1; }
    int all_ok = 1;
    for (int ci = 0; ci < cases->len; ci++) {
        jval *case_ = cases->kids[ci];
        jval *teacher = json_get(case_, "teacher_forcing_ids");
        jval *greedy = json_get(case_, "greedy_new_ids");
        jval *prompt = json_get(case_, "prompt_ids");
        jval *maxnew = json_get(case_, "max_new_tokens");
        if (!teacher || !greedy || !prompt || !maxnew) continue;
        int ok = 1;
        if (!json_token_array_ok(teacher, m->c.vocab, "teacher_forcing_ids", ci) ||
            !json_token_array_ok(greedy, m->c.vocab, "greedy_new_ids", ci) ||
            !json_token_array_ok(prompt, m->c.vocab, "prompt_ids", ci) ||
            prompt->len < 1 || maxnew->t != J_NUM || !isfinite(maxnew->num) ||
            trunc(maxnew->num) != maxnew->num || maxnew->num < 0 ||
            maxnew->num > greedy->len || prompt->len > m->max_t ||
            greedy->len > m->max_t - prompt->len ||
            teacher->len > prompt->len + greedy->len) {
            fprintf(stderr, "ref.json: malformed bounds/lengths in case %d\n", ci);
            all_ok = 0;
            continue;
        }
        int max_new = (int)maxnew->num;
        /* fresh states for THIS case: the previous case's KV/GDN state must
         * not leak into the next teacher pass */
        state_reset(m);
        /* teacher-forced logits: feed the ACTUAL sequence (prompt + greedy
         * continuation = greedy_full_ids); each position's argmax must equal
         * teacher_forcing_ids[i] (the oracle's prediction at that position). */
        int np = prompt->len;
        int *full = malloc(size_mul_or_die((size_t)(np + greedy->len), sizeof(int), "selftest teacher ids"));
        if (!full) { fprintf(stderr, "OOM selftest teacher ids\n"); exit(1); }
        for (int i = 0; i < np; i++) full[i] = (int)prompt->kids[i]->num;
        for (int i = 0; i < greedy->len; i++) full[np + i] = (int)greedy->kids[i]->num;
        for (int i = 0; i < teacher->len; i++) {
            float *logits = step(m, full + i, 1, i);
            int best = qwen_pick_token(m, logits, -1);
            if (best != (int)teacher->kids[i]->num) {
                printf("  FAIL case %d teacher pos %d: engine=%d oracle=%d\n",
                       ci, i, best, (int)teacher->kids[i]->num);
                ok = 0; logits_free(&logits); break;
            }
            logits_free(&logits);
        }
        free(full);
        printf("  %s case %d teacher-forced (%d positions)\n", ok ? "ok  " : "FAIL", ci, teacher->len);
        if (ok) {
            /* greedy decode: re-prefill from prompt (fresh KV via fresh states) */
            state_reset(m);
            int np = prompt->len;
            int *ids = malloc(size_mul_or_die((size_t)np, sizeof(int), "selftest prompt ids"));
            if (!ids) { fprintf(stderr, "OOM selftest prompt ids\n"); exit(1); }
            for (int i = 0; i < np; i++) ids[i] = (int)prompt->kids[i]->num;
            float *logits = step(m, ids, np, 0);
            for (int s = 0; s < max_new; s++) {
                int best = qwen_pick_token(m, logits, -1);
                logits_free(&logits);
                if (best != (int)greedy->kids[s]->num) {
                    printf("  FAIL case %d greedy token %d: engine=%d oracle=%d\n",
                           ci, s, best, (int)greedy->kids[s]->num);
                    ok = 0; break;
                }
                logits = step(m, &best, 1, np + s);
            }
            logits_free(&logits); free(ids);
            printf("  %s case %d greedy (%d tokens)\n", ok ? "ok  " : "FAIL", ci, max_new);
        }
        all_ok &= ok;
    }
    free(buf); free(arena);
    printf(all_ok ? "SELFTEST PASS\n" : "SELFTEST FAIL\n");
    return all_ok ? 0 : 1;
}

int main(int argc, char **argv){
    coli_omp_tune_threads("qwen3_moe");
    rt_trace_open();
    const char *snap;
    int cap;
    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        /* serve protocol (openai_server.py): argv[1] is the cap sentinel and
         * SNAP env carries the model dir — same convention as olmoe.c. */
        snap = getenv("SNAP");
        cap = argc > 1 ? atoi(argv[1]) : (getenv("CACHE") ? atoi(getenv("CACHE")) : 0);
    } else {
        snap = argc > 1 ? argv[1] : getenv("SNAP");
        cap = argc > 2 ? atoi(argv[2]) : (getenv("CACHE") ? atoi(getenv("CACHE")) : 0);
    }
    if (!snap || !*snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    if (cap == 0) {
        /* Default: ONLY the experts currently in use stay resident — one
         * cache slot per selected expert per layer (topk). Everything else
         * is streamed from disk on demand. Raise CACHE/RAM_GB for more. */
        if (getenv("RAM_GB") && atoi(getenv("RAM_GB")) > 0) {
            cap = atoi(getenv("RAM_GB")) / 2;
        } else {
            Cfg cfg0; load_cfg(&cfg0, snap);   /* config-only pre-pass for topk */
            cap = cfg0.topk;
            free(cfg0.layer_is_gdn);
        }
    }
    if (cap < 1 || cap > 4096) { fprintf(stderr, "CACHE must be 1..4096 (got %d)\n", cap); return 1; }
    /* page-cache discipline: DONTNEED after every file read (experts and
     * dense). Default ON — a 35B+ snapshot otherwise floods the page cache
     * (on a 16 GB box that reads as "too much RAM" in Activity Monitor).
     * EXPERT_DROP=0 / DENSE_KEEP_PAGES=1 opt back out for repeated reads. */
    g_expert_drop = getenv("EXPERT_DROP") ? atoi(getenv("EXPERT_DROP")) : 1;
    g_prefetch = getenv("QWEN_PREFETCH") ? atoi(getenv("QWEN_PREFETCH")) : 0;
    g_prefetch_pipe = getenv("QWEN_PREFETCH_PIPE") ? atoi(getenv("QWEN_PREFETCH_PIPE")) : 0;
    g_chunk = getenv("QWENMOE_CHUNK") ? atoi(getenv("QWENMOE_CHUNK")) : 64;
    if (g_chunk < 1) g_chunk = 1;
    if (g_chunk > QWEN_CHUNK_MAX) g_chunk = QWEN_CHUNK_MAX;
    g_kv_f16 = getenv("QWEN_KV_F16") ? atoi(getenv("QWEN_KV_F16")) : 1;
    g_arena_wave = getenv("QWEN_ARENA_WAVE") ? atoi(getenv("QWEN_ARENA_WAVE")) : QWEN_ARENA_CAP;
    if (g_arena_wave < 8 || g_arena_wave > QWEN_ARENA_CAP_MAX) g_arena_wave = QWEN_ARENA_CAP;
    g_dense_drop  = getenv("DENSE_KEEP_PAGES") ? 0 : 1;

    Model m; model_init(&m, snap, cap);
    Tok T;
    char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
    tok_load(&T, tokpath);
    m.tok = &T;
    int ctx_cap = getenv("CTX") ? atoi(getenv("CTX")) : 65536;
    if (ctx_cap < 1 || ctx_cap > (1 << 20)) { fprintf(stderr, "CTX out of range\n"); return 1; }
    m.max_t = ctx_cap;
    m.K = calloc_checked((size_t)m.c.n_layers, sizeof(float*), "K cache rows");
    m.V = calloc_checked((size_t)m.c.n_layers, sizeof(float*), "V cache rows");
    m.K16 = calloc_checked((size_t)m.c.n_layers, sizeof(uint16_t*), "K16 cache rows");
    m.V16 = calloc_checked((size_t)m.c.n_layers, sizeof(uint16_t*), "V16 cache rows");
    m.gdn_S = calloc_checked((size_t)m.c.n_layers, sizeof(float*), "GDN recurrence rows");
    m.gdn_conv = calloc_checked((size_t)m.c.n_layers, sizeof(float*), "GDN convolution rows");
    for (int i = 0; i < m.c.n_layers; i++) {
        if (!m.c.layer_is_gdn[i]) {     /* only full_attention layers use K/V */
            size_t n = kv_state_count(&m.c, m.max_t);
            if (g_kv_f16) {
                m.K16[i] = calloc_checked(n, sizeof(uint16_t), "K cache (f16)");
                m.V16[i] = calloc_checked(n, sizeof(uint16_t), "V cache (f16)");
            } else {
                m.K[i] = calloc_checked(n, sizeof(float), "K cache");
                m.V[i] = calloc_checked(n, sizeof(float), "V cache");
            }
        }
        if (m.c.layer_is_gdn[i]) {
            m.gdn_S[i] = calloc_checked(gdn_state_count(&m.c), sizeof(float), "GDN recurrence");
            m.gdn_conv[i] = calloc_checked(gdn_conv_count(&m.c), sizeof(float), "GDN convolution");
        }
    }

    int rc = 0;
#ifdef COLI_METAL
    /* QWEN_METAL_COMPUTE=1: Apple-GPU batched MoE matmuls (opt-in; CPU
     * kernels stay the default and the fallback). Init must happen AFTER
     * the Qwen model is parsed (backend_metal resolves expert slabs lazily)
     * but before any expert load allocates/registers slabs. */
    g_metal_compute = getenv("QWEN_METAL_COMPUTE") ? atoi(getenv("QWEN_METAL_COMPUTE")) : 0;
    if (g_metal_compute && !coli_metal_init()) {
        fprintf(stderr, "qwen_moe: Metal unavailable — QWEN_METAL_COMPUTE=1 ignored (CPU MoE)\n");
        g_metal_compute = 0;
    }
#endif
#ifdef COLI_METALIO
    /* MetalIO init AFTER the shards are open (file handles need S.paths);
     * cache slots get a persistent shared-storage MTLBuffer each (lazily
     * allocated on first use). Any failure disables the path — pread stays
     * the fallback. */
    g_metal_io = getenv("QWEN_METAL_IO") ? atoi(getenv("QWEN_METAL_IO")) : 0;
    if (g_metal_io) {
        if (metalio_init()) {
            metalio_verbose(getenv("QWEN_METAL_IO_VERBOSE") ? 1 : 0);
            for (int fi = 0; fi < m.S.nfd && g_mio_n < 64; fi++) {
                int fid = metalio_file_add(m.S.paths[fi]);
                if (fid >= 0) { g_mio_fd[g_mio_n] = m.S.fds[fi]; g_mio_fid[g_mio_n] = fid; g_mio_n++; }
            }
            if (g_mio_n == 0) { metalio_shutdown(); g_metal_io = 0; }
        } else {
            g_metal_io = 0;
        }
        if (g_metal_io) {
            for (int li = 0; li < m.c.n_layers; li++)
                for (int si = 0; si < m.cache[li].cap; si++) {
                    m.cache[li].slots[si].mio = 1;
                    m.cache[li].slots[si].mio_slot = -1;
                }
            fprintf(stderr, "[metalio] expert streaming via MTLIO active (%d shards)\n", g_mio_n);
        }
    }
#endif
    const char *mode = getenv("QWENMOE_MODE");
    if (mode && !strcmp(mode, "teacher"))      rc = mode_teacher(&m);
    else if (mode && !strcmp(mode, "greedy"))  rc = mode_greedy(&m);
    else if (getenv("SERVE") && getenv("SERVE")[0] == '1') rc = mode_serve(&m, &T);
    else if (getenv("CHAT"))                   rc = mode_chat(&m, &T);
    else                                       rc = mode_selftest(&m, snap);

    save_usage(&m, snap);
#ifdef COLI_METALIO
    if (g_metal_io) {
        ColiMetalioStats st;
        metalio_stats(&st);
        fprintf(stderr, "[metalio] loads=%llu bytes=%llu waits=%llu fails=%llu "
                        "outstanding=%llu peak=%llu avg_lat_ms=%.2f\n",
                (unsigned long long)st.loads, (unsigned long long)st.bytes,
                (unsigned long long)st.waits, (unsigned long long)st.fails,
                (unsigned long long)st.outstanding, (unsigned long long)st.peak_outstanding,
                st.latency_samples ? st.total_latency_s * 1000.0 / (double)st.latency_samples : 0.0);
        metalio_shutdown();
    }
#endif
    tok_free(&T);
    return rc;
}

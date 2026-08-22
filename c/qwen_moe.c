/* qwen_moe.c — Milestone B GDN + Milestone C1 router wiring wrapper.
 *
 * Keep the verified Milestone-A engine body byte-for-byte in
 * qwen_moe_base.inc.  The one-shot symbol interceptors below only rename the
 * original definitions to *_reference and publish ordinary prototypes; after
 * each first definition the macro is popped, so all later call sites resolve
 * to the normal wrapper functions defined after the include.  No function-like
 * macro shares the real function's parameter list.
 */
#if defined(__APPLE__) && defined(COLI_METALIO)
#pragma push_macro("calloc_checked")
#define calloc_checked \
    calloc_checked_reference(size_t, size_t, const char *); \
    static void *calloc_checked(size_t, size_t, const char *); \
    _Pragma("pop_macro(\"calloc_checked\")") \
    static void *calloc_checked_reference

#pragma push_macro("coli_wt")
#define coli_wt \
    coli_wt_reference(Model *, const char *, int64_t, int64_t); \
    static WT coli_wt(Model *, const char *, int64_t, int64_t); \
    _Pragma("pop_macro(\"coli_wt\")") \
    static WT coli_wt_reference

#pragma push_macro("gdn_token")
#define gdn_token \
    gdn_token_reference(Model *, Layer *, int, const float *, float *); \
    static void gdn_token(Model *, Layer *, int, const float *, float *); \
    _Pragma("pop_macro(\"gdn_token\")") \
    static void gdn_token_reference

#pragma push_macro("moe_token")
#define moe_token \
    moe_token_reference(Model *, Layer *, int, const float *, float *); \
    static void moe_token(Model *, Layer *, int, const float *, float *); \
    _Pragma("pop_macro(\"moe_token\")") \
    static void moe_token_reference
#endif

#include "qwen_moe_base.inc"

#if defined(__APPLE__) && defined(COLI_METALIO)
/* GDN recurrent/conv state backs no-copy Shared MTLBuffers.  Only the two
 * allocations carrying the explicit GDN diagnostic labels are page-aligned;
 * all other calloc_checked callers preserve the Milestone-A allocator. */
static void *calloc_checked(size_t n, size_t size, const char *what){
    if (!what || strncmp(what, "GDN ", 4) != 0)
        return calloc_checked_reference(n, size, what);
    size_t bytes = size_mul_or_die(n, size, what);
    if (bytes > SIZE_MAX - 16383u) {
        fprintf(stderr, "%s: UMA allocation size overflow (%zu bytes)\n", what, bytes);
        exit(1);
    }
    size_t alloc_bytes = (bytes + 16383u) & ~(size_t)16383u;
    void *p = NULL;
    if (alloc_bytes && posix_memalign(&p, 16384, alloc_bytes) == 0 && p) {
        memset(p, 0, alloc_bytes);
        return p;
    }
    return calloc_checked_reference(n, size, what);
}

/* Exact COLI BF16 GDN matrices live for the model lifetime.  Re-home only
 * linear_attn matrices into page-rounded 16 KiB storage so Metal can wrap
 * them zero-copy.  Allocation failure deliberately keeps the original
 * pointer, causing the Metal entry point to decline before submission and
 * the CPU reference path to run. */
static WT coli_wt(Model *m, const char *cn, int64_t O, int64_t I){
    WT w = coli_wt_reference(m, cn, O, I);
    if (w.bf16 && cn && strstr(cn, "linear_attn.")) {
        size_t elems = size_mul_or_die((size_t)O, (size_t)I, cn);
        size_t bytes = size_mul_or_die(elems, sizeof(uint16_t), cn);
        if (bytes <= SIZE_MAX - 16383u) {
            size_t alloc_bytes = (bytes + 16383u) & ~(size_t)16383u;
            void *p = NULL;
            if (alloc_bytes && posix_memalign(&p, 16384, alloc_bytes) == 0 && p) {
                memcpy(p, w.bf16, bytes);
                free(w.bf16);
                w.bf16 = (uint16_t *)p;
            }
        }
    }
    return w;
}

extern int coli_apple8_metalio_gdn_token(
    int layer, const float *x, float *out,
    const uint16_t *wqkv, const uint16_t *wz,
    const uint16_t *wa, const uint16_t *wb, const uint16_t *wout,
    const float *A_log, const float *dt_bias,
    const float *conv_w, const float *norm_w,
    float *state, float *conv_state,
    int D, int kheads, int kd, int vheads, int vd, int kk, float eps);

static int qwen_gdn_metal_enabled(void){
    static int initialized = 0, enabled = 1;
    if (!initialized) {
        const char *v = getenv("QWEN_GDN_METAL");
        if (v && v[0] && strcmp(v, "0") == 0) enabled = 0;
        initialized = 1;
    }
    return g_apple8_direct && enabled;
}

/* Decode-only Metal seam.  QWEN_GDN_METAL defaults ON when the direct
 * Apple8 path is active; QWEN_GDN_METAL=0 forces the exact CPU reference.
 * rc==0 means Metal declined before commit and CPU fallback is safe.  A
 * negative rc is post-submit failure: recurrent state may have changed, so
 * falling through would double-advance it and is therefore fatal. */
static void gdn_token(Model *m, Layer *l, int layer, const float *x, float *out){
    Cfg *c = &m->c;
    if (qwen_gdn_metal_enabled() &&
        l->in_qkv.bf16 && l->in_z.bf16 && l->in_a.bf16 &&
        l->in_b.bf16 && l->gdn_out.bf16) {
        int rc = coli_apple8_metalio_gdn_token(
            layer, x, out,
            l->in_qkv.bf16, l->in_z.bf16, l->in_a.bf16,
            l->in_b.bf16, l->gdn_out.bf16,
            l->A_log, l->dt_bias, l->conv1d, l->gdn_norm,
            m->gdn_S[layer], m->gdn_conv[layer],
            c->hidden, c->lin_k_heads, c->lin_k_dim,
            c->lin_v_heads, c->lin_v_dim, c->conv_kernel, c->eps);
        if (rc > 0) return;
        if (rc < 0) {
            fprintf(stderr, "qwen: Metal GDN decode failed after submission\n");
            exit(1);
        }
    }
    gdn_token_reference(m, l, layer, x, out);
}

extern int coli_apple8_metalio_router_topk_bf16(
    const uint16_t *router, const float *x,
    int E, int D, int K,
    int *idx_out, float *weights_out, void **route_out);
extern void coli_apple8_metalio_router_route_free(void *route);
extern int coli_apple8_metalio_moe_topk_begin_routed(
    const ColiApple8MetalioExpert *experts, void *route,
    int expert_count, const float *x, int hidden, int intermediate,
    void **pending_out);

static int qwen_router_metal_enabled(void){
    static int initialized = 0, enabled = 1;
    if (!initialized) {
        const char *v = getenv("QWEN_ROUTER_METAL");
        if (v && v[0] && strcmp(v, "0") == 0) enabled = 0;
        initialized = 1;
    }
    return g_apple8_direct && enabled;
}

/* Decode MoE wrapper for Milestone C1.  When the direct Apple8 path is live,
 * QWEN_ROUTER_METAL defaults ON and replaces the CPU router GEMV + softmax +
 * top-k selection with the BF16 Metal kernel.  The selected ids/weights still
 * come back for expert-cache scheduling and telemetry; when the fused Apple8
 * routed path is used, its reduction consumes the Metal-owned weight buffer
 * directly, avoiding the old CPU route-weight handoff.  QWEN_ROUTER_METAL=0,
 * missing BF16 router weights, or a pre-dispatch Metal decline falls back to
 * the exact Milestone-A reference function. */
static void moe_token(Model *m, Layer *l, int layer, const float *x, float *out){
    Cfg *c = &m->c; int E = c->n_experts, K = c->topk, D = c->hidden;

    if (!qwen_router_metal_enabled() || !l->router.bf16 || K > 64) {
        moe_token_reference(m, l, layer, x, out);
        return;
    }

    int *idx = malloc(size_mul_or_die((size_t)K, sizeof(int), "router indices"));
    float *w = malloc(size_mul_or_die((size_t)K, sizeof(float), "router top-k weights"));
    void *metal_route = NULL;
    if (!idx || !w) { fprintf(stderr, "OOM router selection\n"); exit(1); }
    if (!coli_apple8_metalio_router_topk_bf16(
            l->router.bf16, x, E, D, K, idx, w, &metal_route)) {
        free(idx); free(w);
        if (metal_route) coli_apple8_metalio_router_route_free(metal_route);
        moe_token_reference(m, l, layer, x, out);
        return;
    }

    m->prof_expert_requests += (uint64_t)K;
    float *acc = calloc((size_t)D, sizeof(float));
    if (!acc) { fprintf(stderr, "OOM\n"); exit(1); }
    void *apple8_pending = NULL;

    /* A Slot* returned by expert_get() is not leased: a later miss may LRU-
     * reuse that same object. Gathering K pointers is therefore invalid when
     * the resident cache cannot simultaneously hold all K routed experts. */
    if (m->cache[layer].cap < K) {
        static int warned_alias_safe = 0;
        if (!warned_alias_safe) {
            fprintf(stderr,
                    "[QWEN-DECODE] resident cap=%d < topk=%d; using alias-safe sequential routed experts\n",
                    m->cache[layer].cap, K);
            warned_alias_safe = 1;
        }
        if (K <= 64) {
            memcpy(m->last_route, idx, (size_t)K * sizeof(int));
            m->last_route_k = K;
        }
        for (int i = 0; i < K; i++) {
            Slot *s = NULL;
            expert_get(m, layer, idx[i], &s);
            expert_wait_ready(m, s);
            float *y = calloc((size_t)D, sizeof(float));
            if (!y) { fprintf(stderr, "OOM\n"); exit(1); }
            expert_apply(m, s, x, y);
            for (int d = 0; d < D; d++) acc[d] += y[d] * w[i];
            free(y);
        }
        rt_route(layer, 0, idx, w, K);
        coli_apple8_metalio_router_route_free(metal_route);
        metal_route = NULL;
        goto routed_experts_done;
    }

    Slot **slots = malloc(size_mul_or_die((size_t)K, sizeof(Slot *), "router slots"));
    if (!slots) { fprintf(stderr, "OOM router slots\n"); exit(1); }
    g_mio_async_issue = g_metal_io && metalio_active();
    for (int i = 0; i < K; i++) expert_get(m, layer, idx[i], &slots[i]);
    g_mio_async_issue = 0;
    if (K <= 64) {
        memcpy(m->last_route, idx, (size_t)K * sizeof(int));
        m->last_route_k = K;
    }
    expert_prefetch_next_early(m, layer, K);

    int gpu_ok = 0;
#ifdef COLI_METAL
    if (g_metal_compute) {
        int fmt0 = slots[0]->fmt, unif = 1;
        for (int i = 1; i < K; i++) if (slots[i]->fmt != fmt0) { unif = 0; break; }

        if (unif && fmt0 == 17 && K <= 64) {
            int fused_topk = 1;
            const char *fused_env = getenv("QWEN_APPLE8_FUSED_TOPK");
            if (fused_env && fused_env[0] && strcmp(fused_env, "0") == 0)
                fused_topk = 0;
            if (fused_topk) {
                ColiApple8MetalioExpert ex[64];
                for (int i = 0; i < K; i++) {
                    Slot *s = slots[i];
                    expert_wait_ready(m, s);
                    if (!s->apple8_direct) {
                        fprintf(stderr, "qwen: fused direct Apple8 slot is not executable\n");
                        exit(1);
                    }
                    ex[i] = (ColiApple8MetalioExpert){
                        .slot = s->mio_slot,
                        .gate_offset = s->apple8_gate_off,
                        .gate_bytes = s->apple8_gate_bytes,
                        .up_offset = s->apple8_up_off,
                        .up_bytes = s->apple8_up_bytes,
                        .down_offset = s->apple8_down_off,
                        .down_bytes = s->apple8_down_bytes,
                    };
                }
                if (!coli_apple8_metalio_moe_topk_begin_routed(
                        ex, metal_route, K, x, D, c->moe_inter,
                        &apple8_pending)) {
                    fprintf(stderr, "qwen: fused direct Apple8 routed decode submit failed\n");
                    exit(1);
                }
                metal_route = NULL; /* consumed by begin_routed() */
                if (!qwen_apple8_overlap_enabled()) {
                    if (!coli_apple8_metalio_moe_topk_finish(apple8_pending, acc)) {
                        fprintf(stderr, "qwen: fused direct Apple8 routed decode completion failed\n");
                        exit(1);
                    }
                    apple8_pending = NULL;
                }
                m->prof_apple8_direct_blocks++;
                m->prof_apple8_direct_experts += (uint64_t)K;
            } else {
                float *yd = falloc(D);
                for (int i = 0; i < K; i++) {
                    Slot *s = slots[i];
                    expert_wait_ready(m, s);
                    if (!s->apple8_direct ||
                        !coli_apple8_metalio_swiglu_slot(s->mio_slot,
                            s->apple8_gate_off, s->apple8_gate_bytes,
                            s->apple8_up_off, s->apple8_up_bytes,
                            s->apple8_down_off, s->apple8_down_bytes,
                            x, yd, 1, D, c->moe_inter)) {
                        fprintf(stderr, "qwen: direct Apple8 decode dispatch failed\n");
                        exit(1);
                    }
                    for (int d = 0; d < D; d++) acc[d] += yd[d] * w[i];
                    m->prof_apple8_direct_blocks++;
                    m->prof_apple8_direct_experts++;
                }
                free(yd);
            }
            gpu_ok = 1;
        } else if (unif && fmt0 == 7 && K <= 64) {
            for (int i = 0; i < K; i++) expert_wait_ready(m, slots[i]);
            const void *gp[64], *up[64], *dp[64];
            const uint8_t *gsp[64], *usp[64], *dsp[64];
            int xoff[64], nr[64], rows[64]; float rw[64];
            float *xg = falloc((int64_t)K * D);
            for (int i = 0; i < K; i++) {
                Slot *s = slots[i];
                gp[i] = s->mxg; up[i] = s->mxu; dp[i] = s->mxd;
                gsp[i] = s->mxgs; usp[i] = s->mxus; dsp[i] = s->mxds;
                memcpy(xg + (int64_t)i * D, x, (size_t)D * sizeof(float));
                xoff[i] = i; nr[i] = 1; rows[i] = 0; rw[i] = w[i];
            }
            gpu_ok = coli_metal_moe_block_mxfp4(K, D, c->moe_inter,
                                                 gp, up, dp, gsp, usp, dsp,
                                                 xg, xoff, nr, rows, rw, acc, 1);
            free(xg);
        } else if (unif && (fmt0 == 4 || fmt0 == 8)) {
            for (int i = 0; i < K; i++) expert_wait_ready(m, slots[i]);
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
                    xoff[i] = i; nr[i] = 1; rows[i] = 0; rw[i] = w[i];
                }
                gpu_ok = coli_metal_moe_block(K, D, c->moe_inter,
                                              fmt0 == 4 ? 4 : 1, 64,
                                              gp, up, dp, gsp, usp, dsp,
                                              xg, xoff, nr, rows, rw, acc, 1);
                free(xg);
            }
        }
    }
#endif

    if (!gpu_ok) for (int i = 0; i < K; i++) {
        expert_wait_ready(m, slots[i]);
        float *y = calloc((size_t)D, sizeof(float));
        if (!y) { fprintf(stderr, "OOM\n"); exit(1); }
        expert_apply(m, slots[i], x, y);
        for (int d = 0; d < D; d++) acc[d] += y[d] * w[i];
        free(y);
    }
    free(slots);
    rt_route(layer, 0, idx, w, K);
    if (metal_route) {
        coli_apple8_metalio_router_route_free(metal_route);
        metal_route = NULL;
    }

routed_experts_done:
    ;
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

    if (apple8_pending) {
        if (!coli_apple8_metalio_moe_topk_finish(apple8_pending, acc)) {
            fprintf(stderr, "qwen: fused direct Apple8 routed decode completion failed\n");
            exit(1);
        }
        apple8_pending = NULL;
    }
    for (int d = 0; d < D; d++) acc[d] += sy[d] * gs;
    memcpy(out, acc, (size_t)D * sizeof(float));

    if (metal_route) coli_apple8_metalio_router_route_free(metal_route);
    free(idx); free(w); free(acc);
    free(sg); free(h); free(gv); free(sy);
}
#endif

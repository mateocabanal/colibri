/* qwen_moe.c — Milestone B GDN wiring wrapper.
 *
 * Keep the verified Milestone-A engine body byte-for-byte in
 * qwen_moe_base.inc.  The one-shot symbol interceptors below only rename the
 * three original definitions to *_reference and publish ordinary prototypes;
 * after each first definition the macro is popped, so all later call sites
 * resolve to the normal wrapper functions defined after the include.  No
 * function-like macro shares the real function's parameter list.
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

/* Blocker: deterministic Metal GDN reduction is still >100 ms/tok; needs a faster deterministic block reduction plus overlapped recurrence before default-on. */
static int qwen_gdn_metal_enabled(void){
    static int initialized = 0, enabled = 1;
    if (!initialized) {
        const char *v = getenv("QWEN_GDN_METAL");
        if (v && v[0] && strcmp(v, "0") == 0) enabled = 0;
        initialized = 1;
    }
    return g_apple8_direct && enabled;
}

/* Decode-only Metal seam.  QWEN_GDN_METAL defaults ON (GPU reference); QWEN_GDN_METAL=0
 * opts into the deterministic Metal path when direct Apple8 is active.
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
#endif

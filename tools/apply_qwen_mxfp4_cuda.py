#!/usr/bin/env python3
from pathlib import Path


def replace_once(text, old, new, label):
    if old not in text:
        raise SystemExit(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)

# backend API
p = Path("c/backend_cuda.h")
s = p.read_text()
if "coli_cuda_expert_mlp_mxfp4" not in s:
    anchor = '''COLI_CUDA_DLLEXPORT int coli_cuda_matmul_mxfp4(float *y, const float *x,
                                               const unsigned char *q4,
                                               const unsigned char *e8s,
                                               int S, int I, int O);
'''
    add = anchor + '''
/* Stateless streamed MXFP4 SwiGLU expert. Unlike three calls to the matmul
 * helper, activations cross PCIe once: gate/up/down weights+E8M0 scales upload
 * into reusable device staging, gate/up + SiLU + down stay on-device, and only
 * the final [S,D] result returns to the host. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_mlp_mxfp4(
        float *y, const float *x,
        const unsigned char *gate, const unsigned char *gate_e8,
        const unsigned char *up, const unsigned char *up_e8,
        const unsigned char *down, const unsigned char *down_e8,
        int S, int D, int I);
'''
    s = replace_once(s, anchor, add, "CUDA MXFP4 expert API")
p.write_text(s)

# backend implementation
p = Path("c/backend_cuda.cu")
s = p.read_text()
if "mx_weights" not in s:
    s = replace_once(
        s,
        '    uint8_t *qx; float *qscale;\n    size_t qx_cap, qscale_cap;\n',
        '    uint8_t *qx; float *qscale;\n    size_t qx_cap, qscale_cap;\n    void *mx_weights, *mx_scales;\n    size_t mx_weights_cap, mx_scales_cap;\n',
        "CUDA MXFP4 staging fields")
    s = replace_once(
        s,
        '        if (ctx->qscale) cudaFree(ctx->qscale);\n',
        '        if (ctx->qscale) cudaFree(ctx->qscale);\n        if (ctx->mx_weights) cudaFree(ctx->mx_weights);\n        if (ctx->mx_scales) cudaFree(ctx->mx_scales);\n',
        "CUDA MXFP4 staging free")
    s = replace_once(
        s,
        '        ctx->qx=nullptr; ctx->qscale=nullptr;\n',
        '        ctx->qx=nullptr; ctx->qscale=nullptr;\n        ctx->mx_weights=ctx->mx_scales=nullptr;\n',
        "CUDA MXFP4 staging reset pointers")
    s = replace_once(
        s,
        '        ctx->qx_cap=ctx->qscale_cap=0;\n',
        '        ctx->qx_cap=ctx->qscale_cap=0;\n        ctx->mx_weights_cap=ctx->mx_scales_cap=0;\n',
        "CUDA MXFP4 staging reset caps")

if "extern \"C\" int coli_cuda_expert_mlp_mxfp4" not in s:
    anchor = '''extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
'''
    fn = r'''extern "C" int coli_cuda_expert_mlp_mxfp4(
        float *y, const float *x,
        const uint8_t *gate, const uint8_t *gate_e8,
        const uint8_t *up, const uint8_t *up_e8,
        const uint8_t *down, const uint8_t *down_e8,
        int S, int D, int I) {
    if (fault_injected()) return 0;
    if (!y || !x || !gate || !gate_e8 || !up || !up_e8 || !down || !down_e8 ||
        S < 1 || D < 1 || I < 1 || g_nctx < 1) return 0;
    DeviceContext *ctx = &g_ctx[0];
    if (!select_ctx(ctx)) return 0;

    const size_t grb = (size_t)(D + 1) / 2;
    const size_t drb = (size_t)(I + 1) / 2;
    const size_t gng = (size_t)(D + 31) / 32;
    const size_t dng = (size_t)(I + 31) / 32;
    const size_t gwb = (size_t)I * grb;
    const size_t dwb = (size_t)D * drb;
    const size_t gsb = (size_t)I * gng;
    const size_t dsb = (size_t)D * dng;
    if (gwb > SIZE_MAX - gwb || 2 * gwb > SIZE_MAX - dwb ||
        gsb > SIZE_MAX - gsb || 2 * gsb > SIZE_MAX - dsb) return 0;
    const size_t weights_bytes = 2 * gwb + dwb;
    const size_t scales_bytes = 2 * gsb + dsb;
    const size_t xb = (size_t)S * D * sizeof(float);
    const size_t ib = (size_t)S * I * sizeof(float);
    const size_t yb = (size_t)S * D * sizeof(float);

    if (!reserve_bytes(&ctx->mx_weights, &ctx->mx_weights_cap, weights_bytes) ||
        !reserve_bytes(&ctx->mx_scales, &ctx->mx_scales_cap, scales_bytes) ||
        !reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->gate, &ctx->gate_cap, ib) ||
        !reserve(&ctx->up, &ctx->up_cap, ib) ||
        !reserve(&ctx->y, &ctx->y_cap, yb) ||
        !reserve_pinned(&ctx->host_y, &ctx->host_y_cap, yb)) return 0;

    uint8_t *wg = static_cast<uint8_t *>(ctx->mx_weights);
    uint8_t *wu = wg + gwb;
    uint8_t *wd = wu + gwb;
    uint8_t *sg = static_cast<uint8_t *>(ctx->mx_scales);
    uint8_t *su = sg + gsb;
    uint8_t *sd = su + gsb;

#define MXCPY(dst, src, bytes, label) \
    do { if (!cuda_ok(cudaMemcpyAsync((dst), (src), (bytes), cudaMemcpyHostToDevice, ctx->stream), (label))) return 0; } while (0)
    MXCPY(wg, gate, gwb, "MXFP4 gate upload");
    MXCPY(wu, up, gwb, "MXFP4 up upload");
    MXCPY(wd, down, dwb, "MXFP4 down upload");
    MXCPY(sg, gate_e8, gsb, "MXFP4 gate scales upload");
    MXCPY(su, up_e8, gsb, "MXFP4 up scales upload");
    MXCPY(sd, down_e8, dsb, "MXFP4 down scales upload");
    MXCPY(ctx->x, x, xb, "MXFP4 expert input upload");
#undef MXCPY

    dim3 hidden_grid((unsigned)I, (unsigned)S);
    dim3 output_grid((unsigned)D, (unsigned)S);
    quant_matmul<<<hidden_grid, 256, 0, ctx->stream>>>(
        ctx->gate, ctx->x, wg, reinterpret_cast<const float *>(sg),
        7, S, D, I, grb, 32, (int)gng);
    if (!cuda_ok(cudaGetLastError(), "MXFP4 gate launch")) return 0;
    quant_matmul<<<hidden_grid, 256, 0, ctx->stream>>>(
        ctx->up, ctx->x, wu, reinterpret_cast<const float *>(su),
        7, S, D, I, grb, 32, (int)gng);
    if (!cuda_ok(cudaGetLastError(), "MXFP4 up launch")) return 0;
    const size_t n = (size_t)S * I;
    silu_mul<<<(unsigned)((n + 255) / 256), 256, 0, ctx->stream>>>(ctx->gate, ctx->up, n);
    if (!cuda_ok(cudaGetLastError(), "MXFP4 SwiGLU launch")) return 0;
    quant_matmul<<<output_grid, 256, 0, ctx->stream>>>(
        ctx->y, ctx->gate, wd, reinterpret_cast<const float *>(sd),
        7, S, I, D, drb, 32, (int)dng);
    if (!cuda_ok(cudaGetLastError(), "MXFP4 down launch") ||
        !cuda_ok(cudaMemcpyAsync(ctx->host_y, ctx->y, yb, cudaMemcpyDeviceToHost, ctx->stream),
                 "MXFP4 expert output download") ||
        !cuda_ok(cudaStreamSynchronize(ctx->stream), "MXFP4 expert synchronize")) return 0;
    std::memcpy(y, ctx->host_y, yb);
    return 1;
}

'''
    s = replace_once(s, anchor, fn + anchor, "CUDA fused MXFP4 expert implementation")
p.write_text(s)

# Qwen runtime
p = Path("c/qwen_moe.c")
s = p.read_text()
if '"backend_cuda.h"' not in s:
    s = replace_once(
        s,
        '#include "mxfp4_runtime.h"\n',
        '#include "mxfp4_runtime.h"\n#ifdef COLI_CUDA\n#include "backend_cuda.h"\n#endif\n',
        "Qwen CUDA include")
if "g_cuda_compute" not in s:
    anchor = '''#ifdef COLI_METAL
#include "backend_metal.h"
'''
    prefix = '''#ifdef COLI_CUDA
static int g_cuda_compute = 0;
#else
#define g_cuda_compute 0
#endif

'''
    s = replace_once(s, anchor, prefix + anchor, "Qwen CUDA state")

# Decode: prefer CUDA fused expert when compiled/active, otherwise CPU fmt7.
if "coli_cuda_expert_mlp_mxfp4(acc, x" not in s:
    old = '''    if (s->fmt == 7) {
        float *gate = falloc(I), *up = falloc(I), *h = falloc(I), *y = falloc(D);
        coli_mxfp4_swiglu_expert(acc, x,
'''
    new = '''    if (s->fmt == 7) {
#ifdef COLI_CUDA
        if (g_cuda_compute && coli_cuda_expert_mlp_mxfp4(acc, x,
                s->mxg, s->mxgs, s->mxu, s->mxus, s->mxd, s->mxds,
                1, D, I)) return;
#endif
        float *gate = falloc(I), *up = falloc(I), *h = falloc(I), *y = falloc(D);
        coli_mxfp4_swiglu_expert(acc, x,
'''
    s = replace_once(s, old, new, "Qwen CUDA decode dispatch")

# Prefill: one fused GPU call per distinct expert; CPU remains exact fallback.
if "coli_cuda_expert_mlp_mxfp4(yscratch, xscratch" not in s:
    batch = s.index("static void moe_batch(")
    old = '''            if (s->fmt == 7) {
                float *gate = falloc((int64_t)st * I), *up = falloc((int64_t)st * I);
'''
    pos = s.index(old, batch)
    new = '''            if (s->fmt == 7) {
#ifdef COLI_CUDA
                if (g_cuda_compute && coli_cuda_expert_mlp_mxfp4(yscratch, xscratch,
                        s->mxg, s->mxgs, s->mxu, s->mxus, s->mxd, s->mxds,
                        st, D, I)) goto qwen_mxfp4_batch_done;
#endif
                float *gate = falloc((int64_t)st * I), *up = falloc((int64_t)st * I);
'''
    s = s[:pos] + new + s[pos + len(old):]
    # label just before storing routed rows, reached by GPU success and CPU path.
    marker = '''            /* store this expert's routed rows at their (token, topk) slots */
'''
    mpos = s.index(marker, batch)
    s = s[:mpos] + '''#ifdef COLI_CUDA
qwen_mxfp4_batch_done:
#endif
''' + s[mpos:]

# CUDA init: direct CUDA/HIP builds use GPU by default; env can disable/select device.
if "QWEN_CUDA_COMPUTE" not in s:
    anchor = '''    int rc = 0;
#ifdef COLI_METAL
'''
    add = '''    int rc = 0;
#ifdef COLI_CUDA
    /* A direct CUDA/HIP build is expected to use its GPU. Unlike Metal, which
     * is an optional compute experiment in the default macOS binary, `make
     * qwen_moe CUDA=1` opts into this backend explicitly. Set
     * QWEN_CUDA_COMPUTE=0 for the CPU fallback or QWEN_CUDA_DEVICE=N to pick
     * a device. Only routed MXFP4 experts are offloaded in this first slice. */
    g_cuda_compute = getenv("QWEN_CUDA_COMPUTE") ? atoi(getenv("QWEN_CUDA_COMPUTE")) : 1;
    if (g_cuda_compute) {
        int dev = getenv("QWEN_CUDA_DEVICE") ? atoi(getenv("QWEN_CUDA_DEVICE")) : 0;
        if (!coli_cuda_init(&dev, 1)) {
            fprintf(stderr, "qwen_moe: CUDA/HIP unavailable — using CPU MoE\\n");
            g_cuda_compute = 0;
        } else {
            fprintf(stderr, "[qwen] MXFP4 routed experts on GPU device %d\\n", dev);
        }
    }
#endif
#ifdef COLI_METAL
'''
    s = replace_once(s, anchor, add, "Qwen CUDA initialization")
    anchor2 = '''    tok_free(&T);
    return rc;
}'''
    repl2 = '''#ifdef COLI_CUDA
    if (g_cuda_compute) coli_cuda_shutdown();
#endif
    tok_free(&T);
    return rc;
}'''
    s = replace_once(s, anchor2, repl2, "Qwen CUDA shutdown")
p.write_text(s)

# Makefile: direct Linux CUDA/HIP joins Qwen; Windows DLL path stays CPU-only for now.
p = Path("c/Makefile")
s = p.read_text()
if "QMOE_GPU_OBJ" not in s:
    old = '''QMOE_COLI_OBJS = coli_executor.o coli_format.o mxfp4_expert.o mxfp4_runtime.o
qwen_moe$(EXE): qwen_moe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h mxfp4_expert.h mxfp4_runtime.h quant.h $(QMOE_COLI_OBJS) $(MIO_OBJ) $(METAL_OBJ)
\t$(CC) $(NOCUDA_CFLAGS) $(MIO_CFLAGS) qwen_moe.c $(QMOE_COLI_OBJS) $(MIO_OBJ) $(METAL_OBJ) -o qwen_moe$(EXE) $(NOCUDA_LDFLAGS) $(MIO_LDFLAGS) $(filter -framework Metal -framework Foundation -lc++,$(LDFLAGS))
'''
    new = '''QMOE_COLI_OBJS = coli_executor.o coli_format.o mxfp4_expert.o mxfp4_runtime.o
# Qwen's first CUDA/HIP slice is the streamed MXFP4 routed-expert path. Direct
# Linux GPU builds link backend_cuda.o; CPU builds and Windows runtime-DLL builds
# keep the old no-CUDA surface until the loader ABI grows this new entry point.
QMOE_GPU_ENABLED = $(if $(filter 1,$(CUDA) $(HIP)),1,)
QMOE_GPU_OBJ = $(if $(QMOE_GPU_ENABLED),$(CUDA_OBJ),)
QMOE_CFLAGS = $(if $(QMOE_GPU_ENABLED),$(CFLAGS),$(NOCUDA_CFLAGS))
QMOE_LDFLAGS = $(if $(QMOE_GPU_ENABLED),$(LDFLAGS),$(NOCUDA_LDFLAGS))
qwen_moe$(EXE): qwen_moe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h mxfp4_expert.h mxfp4_runtime.h quant.h backend_cuda.h $(QMOE_COLI_OBJS) $(QMOE_GPU_OBJ) $(MIO_OBJ) $(METAL_OBJ)
\t$(CC) $(QMOE_CFLAGS) $(MIO_CFLAGS) qwen_moe.c $(QMOE_COLI_OBJS) $(QMOE_GPU_OBJ) $(MIO_OBJ) $(METAL_OBJ) -o qwen_moe$(EXE) $(QMOE_LDFLAGS) $(MIO_LDFLAGS) $(filter -framework Metal -framework Foundation -lc++,$(LDFLAGS))
'''
    s = replace_once(s, old, new, "Qwen GPU Makefile target")
p.write_text(s)

# CUDA differential test for the fused expert primitive.
p = Path("c/tests/test_mxfp4_cuda.cu")
s = p.read_text()
if "fused_expert_case" not in s:
    test = r'''
static void fused_expert_case(void) {
    const int S=3,D=64,I=32;
    const int grb=(D+1)/2, drb=(I+1)/2, gng=(D+31)/32, dng=(I+31)/32;
    uint8_t *g=(uint8_t*)malloc((size_t)I*grb), *u=(uint8_t*)malloc((size_t)I*grb);
    uint8_t *d=(uint8_t*)malloc((size_t)D*drb);
    uint8_t *gs=(uint8_t*)malloc((size_t)I*gng), *us=(uint8_t*)malloc((size_t)I*gng);
    uint8_t *ds=(uint8_t*)malloc((size_t)D*dng);
    float *x=(float*)malloc((size_t)S*D*sizeof(float));
    float *gate=(float*)malloc((size_t)S*I*sizeof(float));
    float *upv=(float*)malloc((size_t)S*I*sizeof(float));
    float *h=(float*)malloc((size_t)S*I*sizeof(float));
    float *cpu=(float*)malloc((size_t)S*D*sizeof(float));
    float *gpu=(float*)malloc((size_t)S*D*sizeof(float));
    if(!g||!u||!d||!gs||!us||!ds||!x||!gate||!upv||!h||!cpu||!gpu){printf("  FAIL fused expert oom\n");fails++;goto done;}
    for(int n=0;n<I*grb;n++){g[n]=(uint8_t)rnd();u[n]=(uint8_t)rnd();}
    for(int n=0;n<D*drb;n++)d[n]=(uint8_t)rnd();
    for(int n=0;n<I*gng;n++){gs[n]=(uint8_t)(122+rnd()%11);us[n]=(uint8_t)(122+rnd()%11);}
    for(int n=0;n<D*dng;n++)ds[n]=(uint8_t)(122+rnd()%11);
    for(int n=0;n<S*D;n++)x[n]=((float)(rnd()%2001)-1000.f)/2000.f;
    mxfp4_ref(gate,x,g,gs,S,D,I); mxfp4_ref(upv,x,u,us,S,D,I);
    for(int n=0;n<S*I;n++)h[n]=(gate[n]/(1.f+expf(-gate[n])))*upv[n];
    mxfp4_ref(cpu,h,d,ds,S,I,D);
    if(!coli_cuda_expert_mlp_mxfp4(gpu,x,g,gs,u,us,d,ds,S,D,I)){
        printf("  FAIL fused MXFP4 expert CUDA path refused\n");fails++;goto done;
    }
    { double worst=0; int bad=0; for(int n=0;n<S*D;n++){double den=fabs(cpu[n])>1e-6?fabs(cpu[n]):1e-6;double r=fabs((double)cpu[n]-gpu[n])/den;if(r>worst)worst=r;if(r>3e-4)bad++;}
      if(bad){printf("  FAIL fused MXFP4 expert %d/%d differ worst rel %.3e\n",bad,S*D,worst);fails++;}
      else printf("  ok   fused MXFP4 SwiGLU expert      S=%d D=%d I=%d worst rel %.2e\n",S,D,I,worst); }
done:
    free(g);free(u);free(d);free(gs);free(us);free(ds);free(x);free(gate);free(upv);free(h);free(cpu);free(gpu);
}

'''
    s = replace_once(s, 'int main(void) {\n', test + 'int main(void) {\n', "CUDA fused test definition")
    s = replace_once(s, '    all_codes();\n', '    all_codes();\n    fused_expert_case();\n', "CUDA fused test invocation")
p.write_text(s)

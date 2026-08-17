#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)


# ---- backend_metal.h: add a correctly-typed MXFP4 fused-MoE entry point ----
hp = Path("c/backend_metal.h")
h = hp.read_text()
if "coli_metal_moe_block_mxfp4" not in h:
    anchor = '''int coli_metal_moe_block(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw,
                         float *out, int S);
'''
    addition = anchor + '''
/* MXFP4 specialization of the fused routed-expert block. The generic legacy
 * entry point above keeps float scale pointers for fmt 1/2/4/5/6 callers;
 * MXFP4's E8M0 scales are raw bytes, so exposing them as float* would encode a
 * false ABI contract. Internally both paths use format-neutral GPU addresses. */
int coli_metal_moe_block_mxfp4(int nb, int D, int Iinter,
                         const void *const *g, const void *const *u, const void *const *d,
                         const uint8_t *const *gs, const uint8_t *const *us,
                         const uint8_t *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw,
                         float *out, int S);
'''
    h = replace_once(h, anchor, addition, "MXFP4 Metal header API")
hp.write_text(h)


# ---- backend_metal.mm: fmt=7 shader + format-neutral internal scale addresses ----
mp = Path("c/backend_metal.mm")
m = mp.read_text()

if "fmt == 7) {                            // MXFP4" not in m:
    anchor = '''  } else if (fmt == 4) {                            // grouped int4: per-expert scale [O][ng]
    int rb=(K+1)/2, ng=(K+qgs-1)/qgs; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const float* sr=sc+(long)o*ng;           // grouped scales for this output row
    device const uchar4* w4=(device const uchar4*)w;
    for(int c=slane;c<K8;c+=32){ uchar4 b=w4[c];
      float4 w0=float4(float(int(b.x&0xF)-8),float(int(b.x>>4)-8),float(int(b.y&0xF)-8),float(int(b.y>>4)-8));
      float4 w1=float4(float(int(b.z&0xF)-8),float(int(b.z>>4)-8),float(int(b.w&0xF)-8),float(int(b.w>>4)-8));
      int g0=(8*c+0)/qgs,g1=(8*c+1)/qgs,g2=(8*c+2)/qgs,g3=(8*c+3)/qgs;
      int g4=(8*c+4)/qgs,g5=(8*c+5)/qgs,g6=(8*c+6)/qgs,g7=(8*c+7)/qgs;
      acc+=dot(w0*float4(sr[g0],sr[g1],sr[g2],sr[g3]),x4[2*c])
          +dot(w1*float4(sr[g4],sr[g5],sr[g6],sr[g7]),x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]*sr[i/qgs]; }
  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;
'''
    replacement = '''  } else if (fmt == 4) {                            // grouped int4: per-expert scale [O][ng]
    int rb=(K+1)/2, ng=(K+qgs-1)/qgs; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const float* sr=sc+(long)o*ng;           // grouped scales for this output row
    device const uchar4* w4=(device const uchar4*)w;
    for(int c=slane;c<K8;c+=32){ uchar4 b=w4[c];
      float4 w0=float4(float(int(b.x&0xF)-8),float(int(b.x>>4)-8),float(int(b.y&0xF)-8),float(int(b.y>>4)-8));
      float4 w1=float4(float(int(b.z&0xF)-8),float(int(b.z>>4)-8),float(int(b.w&0xF)-8),float(int(b.w>>4)-8));
      int g0=(8*c+0)/qgs,g1=(8*c+1)/qgs,g2=(8*c+2)/qgs,g3=(8*c+3)/qgs;
      int g4=(8*c+4)/qgs,g5=(8*c+5)/qgs,g6=(8*c+6)/qgs,g7=(8*c+7)/qgs;
      acc+=dot(w0*float4(sr[g0],sr[g1],sr[g2],sr[g3]),x4[2*c])
          +dot(w1*float4(sr[g4],sr[g5],sr[g6],sr[g7]),x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]*sr[i/qgs]; }
  } else if (fmt == 7) {                            // MXFP4 E2M1 + raw E8M0/32 columns
    int rb=(K+1)/2, ng=(K+31)/32;
    device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const uchar* sr=(device const uchar*)(saddr[e])+(long)o*ng;
    const float mx4[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
    // One lane owns one adjacent pair. Pairs start at even columns, so a pair
    // never straddles the 32-column E8M0 block boundary.
    for(int i=slane*2;i<K;i+=64){
      uchar b=w[i>>1]; float sv=as_type<float>((uint)sr[i/32]<<23);
      acc += mx4[b&0xFu]*xr[i]*sv;
      if(i+1<K) acc += mx4[b>>4]*xr[i+1]*sv;
    }
  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;
'''
    m = replace_once(m, anchor, replacement, "moe_gemv fmt7 branch")

m = m.replace(
    'if(slane==0) yout[row] = (fmt==4 || fmt==5 || fmt==6) ? acc : acc*sc[o];   // fmt4 grouped / fmt6 in-block scales folded; fmt5 none',
    'if(slane==0) yout[row] = (fmt==4 || fmt==5 || fmt==6 || fmt==7) ? acc : acc*sc[o]; // grouped/in-block/MXFP4 scales folded; fmt5 none',
    1,
)

if "const void *const *gs, const void *const *us, const void *const *ds," not in m:
    old = '''static id<MTLCommandBuffer> moe_submit(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
'''
    new = '''static id<MTLCommandBuffer> moe_submit(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const void *const *gs, const void *const *us, const void *const *ds,
'''
    m = replace_once(m, old, new, "format-neutral internal scales")

m = m.replace(
    'if (!g_dev || (fmt != 1 && fmt != 2 && fmt != 4 && fmt != 5 && fmt != 6)) return nil;',
    'if (!g_dev || (fmt != 1 && fmt != 2 && fmt != 4 && fmt != 5 && fmt != 6 && fmt != 7)) return nil;',
    1,
)

# Existing public callers keep their float-scale source ABI. Cast only at the
# internal address-only boundary; no scale is dereferenced on the host.
m = m.replace(
    'id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,gs,us,ds,xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);',
    'id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,\n'
    '        reinterpret_cast<const void *const *>(gs), reinterpret_cast<const void *const *>(us),\n'
    '        reinterpret_cast<const void *const *>(ds), xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);',
    1,
)

if "extern \"C\" int coli_metal_moe_block_mxfp4" not in m:
    anchor = '''extern "C" int coli_metal_moe_block(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw, float *out, int S) {
  (void)S;
  @autoreleasepool {
    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0) return 1;
    g_xg = ensure(g_xg,&g_xg_cap,(size_t)R*D*4);
    g_gg = ensure(g_gg,&g_gg_cap,(size_t)R*Iinter*4);
    g_uu = ensure(g_uu,&g_uu_cap,(size_t)R*Iinter*4);
    g_hh = ensure(g_hh,&g_hh_cap,(size_t)R*D*4);
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,
        reinterpret_cast<const void *const *>(gs), reinterpret_cast<const void *const *>(us),
        reinterpret_cast<const void *const *>(ds), xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);
    if (!cb) return 0;
    return moe_finish(cb,g_hh,nb,R,D,rows,rw,out);
  }
}
'''
    addition = anchor + '''

extern "C" int coli_metal_moe_block_mxfp4(int nb, int D, int Iinter,
                         const void *const *g, const void *const *u, const void *const *d,
                         const uint8_t *const *gs, const uint8_t *const *us,
                         const uint8_t *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw, float *out, int S) {
  (void)S;
  @autoreleasepool {
    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0) return 1;
    g_xg = ensure(g_xg,&g_xg_cap,(size_t)R*D*4);
    g_gg = ensure(g_gg,&g_gg_cap,(size_t)R*Iinter*4);
    g_uu = ensure(g_uu,&g_uu_cap,(size_t)R*Iinter*4);
    g_hh = ensure(g_hh,&g_hh_cap,(size_t)R*D*4);
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,7,0,g,u,d,
        reinterpret_cast<const void *const *>(gs), reinterpret_cast<const void *const *>(us),
        reinterpret_cast<const void *const *>(ds), xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);
    if (!cb) return 0;
    return moe_finish(cb,g_hh,nb,R,D,rows,rw,out);
  }
}
'''
    m = replace_once(m, anchor, addition, "MXFP4 fused public wrapper")

# Async legacy path also crosses the internal type-neutral boundary.
m = m.replace(
    'id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,gs,us,ds,xg,xoff,nr,R,bxg,bgg,buu,bhh);',
    'id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,\n'
    '        reinterpret_cast<const void *const *>(gs), reinterpret_cast<const void *const *>(us),\n'
    '        reinterpret_cast<const void *const *>(ds), xg,xoff,nr,R,bxg,bgg,buu,bhh);',
    1,
)
mp.write_text(m)


# ---- qwen_moe.c: Metal-friendly MXFP4 cache slabs + fused decode dispatch ----
qp = Path("c/qwen_moe.c")
q = qp.read_text()

old_alloc = '''static void *moe_slab_alloc(size_t n){
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM expert\\n"); exit(1); }
    if (g_metal_compute) {
        void *q = NULL;
        if (!posix_memalign(&q, 16384, n)) p = q;
        else free(p);   /* fall back to malloc'd; register will copy instead of wrap */
    }
    return p;
}
'''
new_alloc = '''static size_t moe_slab_bytes(size_t n){
#ifdef COLI_METAL
    if (g_metal_compute) {
        if (n > SIZE_MAX - 16383u) { fprintf(stderr, "expert slab size overflow\\n"); exit(1); }
        return (n + 16383u) & ~(size_t)16383u;
    }
#endif
    return n;
}
static void *moe_slab_alloc(size_t n){
    void *p = NULL;
#ifdef COLI_METAL
    if (g_metal_compute) {
        if (posix_memalign(&p, 16384, n) != 0) p = NULL;
    } else
#endif
    {
        p = malloc(n);
    }
    if (!p && n) { fprintf(stderr, "OOM expert (%zu bytes)\\n", n); exit(1); }
    return p;
}
'''
if "static size_t moe_slab_bytes" not in q:
    q = replace_once(q, old_alloc, new_alloc, "Metal slab allocator")

old_mxalloc = '''    s->mxg = malloc(weight_bytes);
    s->mxgs = malloc(scale_bytes);
    if ((!s->mxg && weight_bytes) || (!s->mxgs && scale_bytes)) {
        fprintf(stderr, "OOM MXFP4 expert (%zu weight + %zu scale bytes)\\n",
                weight_bytes, scale_bytes);
        exit(1);
    }
'''
new_mxalloc = '''    size_t weight_alloc = moe_slab_bytes(weight_bytes);
    size_t scale_alloc = moe_slab_bytes(scale_bytes);
    s->mxg = moe_slab_alloc(weight_alloc);
    s->mxgs = moe_slab_alloc(scale_alloc);
'''
if "weight_alloc = moe_slab_bytes(weight_bytes)" not in q:
    q = replace_once(q, old_mxalloc, new_mxalloc, "MXFP4 slab allocation")
    q = replace_once(
        q,
        '''    s->pinned = 0;
    s->fmt = 7;
}

static void slot_alloc_q8''',
        '''    s->pinned = 0;
    s->fmt = 7;
#ifdef COLI_METAL
    if (g_metal_compute) {
        coli_metal_register(s->mxg, weight_alloc);
        coli_metal_register(s->mxgs, scale_alloc);
    }
#endif
}

static void slot_alloc_q8''',
        "MXFP4 slab registration",
    )

# Unregister transient MXFP4 slabs before free.
q = q.replace(
    'if (g_metal_compute) { coli_metal_unregister(s->g); coli_metal_unregister(s->g4); coli_metal_unregister(s->gs); coli_metal_unregister(s->g4s); }',
    'if (g_metal_compute) { coli_metal_unregister(s->g); coli_metal_unregister(s->g4); coli_metal_unregister(s->gs); coli_metal_unregister(s->g4s); coli_metal_unregister(s->mxg); coli_metal_unregister(s->mxgs); }',
    1,
)
q = q.replace(
    'if (g_metal_compute) { coli_metal_unregister(s->g4); coli_metal_unregister(s->g4s); }',
    'if (g_metal_compute) { coli_metal_unregister(s->g4); coli_metal_unregister(s->g4s); coli_metal_unregister(s->mxg); coli_metal_unregister(s->mxgs); }',
    1,
)

if "coli_metal_moe_block_mxfp4" not in q:
    anchor = '''        if (unif && (fmt0 == 4 || fmt0 == 8)) {
            for (int i = 0; i < K; i++) expert_wait_ready(m, slots[i]);   /* MIO async loads must land before GPU reads them */
            const void *gp[64], *up[64], *dp[64];
            const float *gsp[64], *usp[64], *dsp[64];
'''
    replacement = '''        if (unif && fmt0 == 7 && K <= 64) {
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
            for (int i = 0; i < K; i++) expert_wait_ready(m, slots[i]);   /* MIO async loads must land before GPU reads them */
            const void *gp[64], *up[64], *dp[64];
            const float *gsp[64], *usp[64], *dsp[64];
'''
    q = replace_once(q, anchor, replacement, "Qwen Metal MXFP4 dispatch")
qp.write_text(q)


# ---- Metal oracle: fused MXFP4 SwiGLU block, not only standalone GEMV ----
tp = Path("c/tests/test_backend_metal.mm")
t = tp.read_text()
if "run_moe_mxfp4" not in t:
    marker = '''// ---- fmt=6 (E8/IQ3) moe_block vs the engine's own scalar decoder ----
'''
    test = r'''// ---- fmt=7 (MXFP4) fused moe_block vs an independent scalar reference ----
static float mx4_ref(uint8_t nib){
  static const float lut[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
  return lut[nib&15];
}
static float mxscale_ref(uint8_t s){ uint32_t u=(uint32_t)s<<23; float f; memcpy(&f,&u,4); return f; }
static double mxrow_ref(const uint8_t* w,const uint8_t* s,const float* x,int K){
  double a=0; for(int k=0;k<K;k++){ uint8_t b=w[k>>1]; uint8_t n=(k&1)?(b>>4):(b&15); a+=(double)mx4_ref(n)*x[k]*mxscale_ref(s[k/32]); } return a;
}
static int run_moe_mxfp4(const std::vector<int>& nrv, const char* name) {
  const int D=128, I=64; int nb=(int)nrv.size();
  int rbG=(D+1)/2, rbD=(I+1)/2, ngG=(D+31)/32, ngD=(I+31)/32;
  int R=0; std::vector<int>xoff(nb),nr(nrv); for(int e=0;e<nb;e++){xoff[e]=R;R+=nr[e];}
  size_t wlogical=(size_t)I*rbG*2+(size_t)D*rbD;
  size_t slogical=(size_t)I*ngG*2+(size_t)D*ngD;
  size_t wlen=roundpg(wlogical), slen=roundpg(slogical);
  std::vector<void*> ws(nb), ss(nb); std::vector<const void*> g(nb),u(nb),d(nb);
  std::vector<const uint8_t*> gs(nb),us(nb),ds(nb);
  srand(7707+nb+R);
  for(int e=0;e<nb;e++){
    posix_memalign(&ws[e],16384,wlen); posix_memalign(&ss[e],16384,slen);
    uint8_t* wp=(uint8_t*)ws[e]; uint8_t* sp=(uint8_t*)ss[e];
    for(size_t j=0;j<wlogical;j++) wp[j]=(uint8_t)(rand()&0xff);
    for(size_t j=0;j<slogical;j++) sp[j]=(uint8_t)(124+(rand()%7)); // 2^-3 .. 2^3
    g[e]=wp; u[e]=wp+(size_t)I*rbG; d[e]=wp+(size_t)I*rbG*2;
    gs[e]=sp; us[e]=sp+(size_t)I*ngG; ds[e]=sp+(size_t)I*ngG*2;
    coli_metal_register(ws[e],wlen); coli_metal_register(ss[e],slen);
  }
  std::vector<float>xg((size_t)R*D); for(auto&v:xg)v=((rand()%2001)-1000)/5000.f;
  std::vector<int>rows(R); std::vector<float>rw(R); for(int r=0;r<R;r++){rows[r]=r%2;rw[r]=0.1f+(rand()%50)/100.f;}
  const int S=2; std::vector<double>ref((size_t)S*D,0.0), mag((size_t)S*D,0.0);
  std::vector<double> gate(I), upv(I), hid(I);
  for(int e=0;e<nb;e++) for(int rr=0;rr<nr[e];rr++){
    int gr=xoff[e]+rr; const float*x=&xg[(size_t)gr*D];
    for(int o=0;o<I;o++){
      gate[o]=mxrow_ref((const uint8_t*)g[e]+(size_t)o*rbG,gs[e]+(size_t)o*ngG,x,D);
      upv[o]=mxrow_ref((const uint8_t*)u[e]+(size_t)o*rbG,us[e]+(size_t)o*ngG,x,D);
      hid[o]=(gate[o]/(1.0+exp(-gate[o])))*upv[o];
    }
    int row=rows[gr];
    for(int o=0;o<D;o++){
      double y=0, ym=0; const uint8_t*wr=(const uint8_t*)d[e]+(size_t)o*rbD; const uint8_t*sr=ds[e]+(size_t)o*ngD;
      for(int k=0;k<I;k++){uint8_t b=wr[k>>1],n=(k&1)?(b>>4):(b&15);double term=mx4_ref(n)*hid[k]*mxscale_ref(sr[k/32]);y+=term;ym+=fabs(term);}
      ref[(size_t)row*D+o]+=rw[gr]*y; mag[(size_t)row*D+o]+=fabs((double)rw[gr])*ym;
    }
  }
  std::vector<float>got((size_t)S*D,0.f);
  int ok=coli_metal_moe_block_mxfp4(nb,D,I,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
                                     xg.data(),xoff.data(),nr.data(),rows.data(),rw.data(),got.data(),S);
  double worst=0; int bad=0; for(size_t i=0;i<got.size();i++){double er=fabs((double)got[i]-ref[i]);double rel=mag[i]>1e-30?er/mag[i]:er;if(rel>worst)worst=rel;if(rel>2e-4)bad++;}
  int pass=ok&&bad==0; printf("  %-30s R=%d worst_rel=%.2e  %s\n",name,R,worst,pass?"ok":"*** MISMATCH");
  for(int e=0;e<nb;e++){coli_metal_unregister(ws[e]);coli_metal_unregister(ss[e]);free(ws[e]);free(ss[e]);}
  return pass?0:1;
}

'''
    t = replace_once(t, marker, test + marker, "MXFP4 fused Metal oracle")
    call_marker = '''  printf("Metal fmt=6 (E8/IQ3) moe_block tests:\\n");
'''
    calls = '''  printf("Metal MXFP4 moe_block tests:\\n");
  fail |= run_moe_mxfp4({1,1,1,1}, "mxfp4 decode nb=4");
  fail |= run_moe_mxfp4({3,1,2},     "mxfp4 ragged nb=3");
'''
    t = replace_once(t, call_marker, calls + call_marker, "MXFP4 Metal oracle calls")
tp.write_text(t)

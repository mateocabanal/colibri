#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend_metal.h"
#include "backend_metal_tile.h"
#include "mxfp4_apple8_tile.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <mach/mach_time.h>

extern "C" int coli_metal_matmul_row(ColiMetalTensor **tensor,
    float *y, const float *x, const void *weights, const float *scales,
    int fmt, int S, int I, int O, int gs);
extern "C" int coli_metal_moe_block_mxfp4_row(int nb, int D, int Iinter,
    const void *const *g, const void *const *u, const void *const *d,
    const uint8_t *const *gs, const uint8_t *const *us,
    const uint8_t *const *ds,
    const float *xg, const int *xoff, const int *nr,
    const int *rows, const float *rw, float *out, int S);

static const char *TILE_SHADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float MX4_TILE[16] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
 -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

kernel void mx_tile_single(device const uchar *rec [[buffer(0)]],
                           device const float *x [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant int& S [[buffer(3)]],
                           constant int& I [[buffer(4)]],
                           constant int& O [[buffer(5)]],
                           constant int& NT [[buffer(6)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row=(long)tg*4+sgid; if(row>=NT) return;
  int o=int(row%O), si=int(row/O), ng=(I+31)/32;
  int orow=o&7, otile=o>>3;
  device const float *xr=x+(long)si*I;
  float acc=0.0f;
  for(int i=int(lane)*2;i<I;i+=64){
    int kg=i/32, kk=i&31;
    device const uchar *tile=rec+((long)otile*ng+kg)*136;
    uchar b=tile[orow*16+(kk>>1)];
    float sv=as_type<float>((uint)tile[128+orow]<<23);
    acc += MX4_TILE[b&15u]*xr[i]*sv;
    if(i+1<I) acc += MX4_TILE[b>>4]*xr[i+1]*sv;
  }
  acc=simd_sum(acc);
  if(lane==0) y[row]=acc;
}

kernel void mx_tile_moe(device const ulong *addr [[buffer(0)]],
                        device const int *erow [[buffer(1)]],
                        device const float *xin [[buffer(2)]],
                        device float *yout [[buffer(3)]],
                        constant int& O [[buffer(4)]],
                        constant int& K [[buffer(5)]],
                        constant int& Kin [[buffer(6)]],
                        constant int& NT [[buffer(7)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row=(long)tg*4+sgid; if(row>=NT) return;
  int gr=int(row/O), o=int(row%O), expert=erow[gr];
  device const uchar *rec=(device const uchar *)(addr[expert]);
  device const float *xr=xin+(long)gr*Kin;
  int ng=(K+31)/32, orow=o&7, otile=o>>3;
  float acc=0.0f;
  for(int i=int(lane)*2;i<K;i+=64){
    int kg=i/32, kk=i&31;
    device const uchar *tile=rec+((long)otile*ng+kg)*136;
    uchar b=tile[orow*16+(kk>>1)];
    float sv=as_type<float>((uint)tile[128+orow]<<23);
    acc += MX4_TILE[b&15u]*xr[i]*sv;
    if(i+1<K) acc += MX4_TILE[b>>4]*xr[i+1]*sv;
  }
  acc=simd_sum(acc);
  if(lane==0) yout[row]=acc;
}

kernel void mx_tile_silu(device float *g [[buffer(0)]],
                         device const float *u [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
  float v=g[i]; g[i]=(v/(1.0f+exp(-v)))*u[i];
}
)METAL";

static uint64_t tile_now_ns(void) {
    static mach_timebase_info_data_t tb={0,0};
    if (!tb.denom) mach_timebase_info(&tb);
    return (uint64_t)((double)mach_absolute_time()*(double)tb.numer/(double)tb.denom);
}

static std::once_flag g_tile_once;
static id<MTLDevice> g_tile_dev;
static id<MTLCommandQueue> g_tile_queue;
static id<MTLComputePipelineState> g_tile_single;
static id<MTLComputePipelineState> g_tile_moe;
static id<MTLComputePipelineState> g_tile_silu;
static bool g_tile_init_ok;

static void tile_init_once(void) {
    @autoreleasepool {
        g_tile_dev=MTLCreateSystemDefaultDevice();
        if(!g_tile_dev) return;
        g_tile_queue=[g_tile_dev newCommandQueue];
        NSError *err=nil;
        id<MTLLibrary> lib=[g_tile_dev newLibraryWithSource:
            [NSString stringWithUTF8String:TILE_SHADER] options:nil error:&err];
        if(!lib){
            fprintf(stderr,"[metal-tile] shader compile failed: %s\n",
                    err?[[err localizedDescription]UTF8String]:"?");
            g_tile_dev=nil; return;
        }
        auto P=[&](NSString *name){
            return [g_tile_dev newComputePipelineStateWithFunction:
                    [lib newFunctionWithName:name] error:&err];
        };
        g_tile_single=P(@"mx_tile_single");
        g_tile_moe=P(@"mx_tile_moe");
        g_tile_silu=P(@"mx_tile_silu");
        if(!g_tile_queue||!g_tile_single||!g_tile_moe||!g_tile_silu){
            fprintf(stderr,"[metal-tile] pipeline creation failed\n");
            g_tile_dev=nil; return;
        }
        g_tile_init_ok=true;
    }
}

static bool tile_init(void) {
    std::call_once(g_tile_once,tile_init_once);
    return g_tile_init_ok;
}

extern "C" int coli_metal_tile_enabled(void) {
    static const int enabled=[](){
        const char *v=getenv("V4_METAL_TILE");
        return (v&&*v&&atoi(v)!=0)?1:0;
    }();
    return enabled;
}

struct TileEntry {
    const void *weights=nullptr;
    const void *scales=nullptr;
    int rows=0;
    int columns=0;
    uint64_t generation=0;
    uint64_t used=0;
    size_t bytes=0;
    id<MTLBuffer> buffer=nil;
};

static std::mutex g_cache_mutex;
static std::vector<TileEntry> g_cache;
static uint64_t g_cache_clock;

static size_t tile_cache_capacity(void) {
    static size_t cap=0;
    if(!cap){
        long value=32;
        const char *v=getenv("V4_METAL_TILE_CACHE_TENSORS");
        if(v&&*v) value=strtol(v,nullptr,10);
        if(value<18) value=18;
        if(value>128) value=128;
        cap=(size_t)value;
    }
    return cap;
}

static std::atomic<uint64_t> st_repack_count{0},st_repack_bytes{0},st_repack_ns{0};
static std::atomic<uint64_t> st_single_calls{0},st_moe_calls{0},st_fallback{0};
static std::atomic<uint64_t> st_experts{0},st_wall_ns{0},st_kernel_ns{0},st_scatter_ns{0};
static std::atomic<uint64_t> st_row_mxfp4_calls{0},st_row_mxfp4_wall_ns{0};

extern "C" void coli_metal_tile_stats(ColiMetalTileStats *stats) {
    if(!stats) return;
    stats->repack_count=st_repack_count.load();
    stats->repack_bytes=st_repack_bytes.load();
    stats->repack_ns=st_repack_ns.load();
    stats->single_calls=st_single_calls.load();
    stats->moe_calls=st_moe_calls.load();
    stats->fallback_calls=st_fallback.load();
    stats->experts=st_experts.load();
    stats->wall_ns=st_wall_ns.load();
    stats->kernel_ns=st_kernel_ns.load();
    stats->scatter_ns=st_scatter_ns.load();
    stats->row_mxfp4_calls=st_row_mxfp4_calls.load();
    stats->row_mxfp4_wall_ns=st_row_mxfp4_wall_ns.load();
}

static id<MTLBuffer> tile_find(const void *weights,const void *scales,
                               int rows,int columns) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    for(auto &entry:g_cache){
        if(entry.weights==weights&&entry.scales==scales&&entry.rows==rows&&
           entry.columns==columns&&entry.buffer){
            entry.used=++g_cache_clock;
            return entry.buffer;
        }
    }
    return nil;
}

extern "C" int coli_metal_tile_prepare_matrix(const void *weights,const void *scales,
                                               int rows,int columns,
                                               uint64_t source_generation) {
    if(!coli_metal_tile_enabled()||!weights||!scales||rows<=0||columns<=0||
       !source_generation||!tile_init()) return 0;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for(auto &entry:g_cache)
            if(entry.weights==weights&&entry.scales==scales&&entry.rows==rows&&
               entry.columns==columns&&entry.generation==source_generation&&entry.buffer){
                entry.used=++g_cache_clock;
                return 1;
            }
    }

    size_t bytes=coli_mxfp4_apple8_tile_bytes((uint64_t)rows,(uint64_t)columns);
    size_t wbytes=(size_t)rows*((size_t)columns+1u)/2u;
    size_t sbytes=(size_t)rows*((size_t)columns+31u)/32u;
    if(!bytes) return 0;
    uint64_t began=tile_now_ns();
    id<MTLBuffer> buffer=[g_tile_dev newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
    if(!buffer||coli_mxfp4_apple8_tile_repack([buffer contents],bytes,
            weights,wbytes,scales,sbytes,(uint64_t)rows,(uint64_t)columns))
        return 0;
    uint64_t elapsed=tile_now_ns()-began;

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for(auto &entry:g_cache){
            if(entry.weights==weights&&entry.scales==scales&&entry.rows==rows&&
               entry.columns==columns){
                entry.generation=source_generation;
                entry.used=++g_cache_clock;
                entry.bytes=bytes;
                entry.buffer=buffer;
                st_repack_count.fetch_add(1);
                st_repack_bytes.fetch_add(bytes);
                st_repack_ns.fetch_add(elapsed);
                return 1;
            }
        }
        TileEntry fresh;
        fresh.weights=weights; fresh.scales=scales; fresh.rows=rows;
        fresh.columns=columns; fresh.generation=source_generation;
        fresh.used=++g_cache_clock; fresh.bytes=bytes; fresh.buffer=buffer;
        if(g_cache.size()<tile_cache_capacity()) g_cache.push_back(fresh);
        else {
            auto victim=std::min_element(g_cache.begin(),g_cache.end(),
                [](const TileEntry&a,const TileEntry&b){return a.used<b.used;});
            *victim=fresh;
        }
    }
    st_repack_count.fetch_add(1);
    st_repack_bytes.fetch_add(bytes);
    st_repack_ns.fetch_add(elapsed);
    return 1;
}

static int tile_single_run(float *y,const float *x,id<MTLBuffer> rec,
                           int S,int I,int O) {
    if(!tile_init()||!rec||!y||!x||S<1||I<1||O<1) return 0;
    uint64_t began=tile_now_ns();
    @autoreleasepool {
        id<MTLBuffer> bx=[g_tile_dev newBufferWithBytes:x
            length:(size_t)S*I*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> by=[g_tile_dev newBufferWithLength:(size_t)S*O*sizeof(float)
            options:MTLResourceStorageModeShared];
        if(!bx||!by) return 0;
        id<MTLCommandBuffer> cb=[g_tile_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e useResource:rec usage:MTLResourceUsageRead];
        [e setComputePipelineState:g_tile_single];
        [e setBuffer:rec offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1];
        [e setBuffer:by offset:0 atIndex:2];
        int NT=S*O;
        [e setBytes:&S length:4 atIndex:3]; [e setBytes:&I length:4 atIndex:4];
        [e setBytes:&O length:4 atIndex:5]; [e setBytes:&NT length:4 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1)
            threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if(cb.status==MTLCommandBufferStatusError) return 0;
        memcpy(y,[by contents],(size_t)S*O*sizeof(float));
        double ks=[cb GPUEndTime]-[cb GPUStartTime];
        if(ks>0) st_kernel_ns.fetch_add((uint64_t)(ks*1.0e9));
    }
    st_single_calls.fetch_add(1);
    st_wall_ns.fetch_add(tile_now_ns()-began);
    return 1;
}

extern "C" int coli_metal_matmul(ColiMetalTensor **tensor,
                                  float *y,const float *x,
                                  const void *weights,const float *scales,
                                  int fmt,int S,int I,int O,int gs) {
    if(coli_metal_tile_enabled()&&fmt==7){
        id<MTLBuffer> rec=tile_find(weights,scales,O,I);
        if(rec&&tile_single_run(y,x,rec,S,I,O)) return 1;
        st_fallback.fetch_add(1);
    }
    uint64_t began=fmt==7?tile_now_ns():0;
    int rc=coli_metal_matmul_row(tensor,y,x,weights,scales,fmt,S,I,O,gs);
    if(began){
        st_row_mxfp4_calls.fetch_add(1);
        st_row_mxfp4_wall_ns.fetch_add(tile_now_ns()-began);
    }
    return rc;
}

static std::mutex g_exec_mutex;
static id<MTLBuffer> g_xg,g_gg,g_uu,g_hh;
static size_t g_xg_cap,g_gg_cap,g_uu_cap,g_hh_cap;

static id<MTLBuffer> ensure_buffer(id<MTLBuffer> current,size_t *cap,size_t need) {
    if(current&&*cap>=need) return current;
    *cap=need;
    return [g_tile_dev newBufferWithLength:need options:MTLResourceStorageModeShared];
}

static int tile_moe_run(int nb,int D,int Iinter,
                        const void *const *g,const void *const *u,const void *const *d,
                        const uint8_t *const *gs,const uint8_t *const *us,
                        const uint8_t *const *ds,
                        const float *xg,const int *xoff,const int *nr,
                        const int *rows,const float *rw,float *out) {
    if(!tile_init()||nb<1||D<1||Iinter<1||!g||!u||!d||!gs||!us||!ds||
       !xg||!xoff||!nr||!rows||!rw||!out) return 0;
    int R=0; for(int i=0;i<nb;i++){ if(nr[i]<0) return 0; R+=nr[i]; }
    if(R==0) return 1;

    std::vector<id<MTLBuffer>> gb(nb),ub(nb),db(nb);
    for(int i=0;i<nb;i++){
        gb[i]=tile_find(g[i],gs[i],Iinter,D);
        ub[i]=tile_find(u[i],us[i],Iinter,D);
        db[i]=tile_find(d[i],ds[i],D,Iinter);
        if(!gb[i]||!ub[i]||!db[i]) return 0;
    }

    std::lock_guard<std::mutex> execution(g_exec_mutex);
    uint64_t began=tile_now_ns();
    @autoreleasepool {
        g_xg=ensure_buffer(g_xg,&g_xg_cap,(size_t)R*D*4u);
        g_gg=ensure_buffer(g_gg,&g_gg_cap,(size_t)R*Iinter*4u);
        g_uu=ensure_buffer(g_uu,&g_uu_cap,(size_t)R*Iinter*4u);
        g_hh=ensure_buffer(g_hh,&g_hh_cap,(size_t)R*D*4u);
        if(!g_xg||!g_gg||!g_uu||!g_hh) return 0;
        memcpy([g_xg contents],xg,(size_t)R*D*4u);

        std::vector<uint64_t> ga(nb),ua(nb),da(nb);
        for(int i=0;i<nb;i++){
            ga[i]=(uint64_t)[gb[i] gpuAddress];
            ua[i]=(uint64_t)[ub[i] gpuAddress];
            da[i]=(uint64_t)[db[i] gpuAddress];
        }
        std::vector<int> erow((size_t)R,0);
        for(int i=0;i<nb;i++)
            for(int r=0;r<nr[i];r++){
                int index=xoff[i]+r;
                if(index<0||index>=R) return 0;
                erow[(size_t)index]=i;
            }
        auto make=[&](const void *p,size_t n){
            return [g_tile_dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> bga=make(ga.data(),ga.size()*sizeof(uint64_t));
        id<MTLBuffer> bua=make(ua.data(),ua.size()*sizeof(uint64_t));
        id<MTLBuffer> bda=make(da.data(),da.size()*sizeof(uint64_t));
        id<MTLBuffer> berow=make(erow.data(),erow.size()*sizeof(int));
        if(!bga||!bua||!bda||!berow) return 0;

        id<MTLCommandBuffer> cb=[g_tile_queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        for(int i=0;i<nb;i++){
            [e useResource:gb[i] usage:MTLResourceUsageRead];
            [e useResource:ub[i] usage:MTLResourceUsageRead];
            [e useResource:db[i] usage:MTLResourceUsageRead];
        }
        auto gemv=[&](id<MTLBuffer> addresses,id<MTLBuffer> input,
                      id<MTLBuffer> output,int O,int K,int Kin){
            int NT=R*O;
            [e setComputePipelineState:g_tile_moe];
            [e setBuffer:addresses offset:0 atIndex:0];
            [e setBuffer:berow offset:0 atIndex:1];
            [e setBuffer:input offset:0 atIndex:2];
            [e setBuffer:output offset:0 atIndex:3];
            [e setBytes:&O length:4 atIndex:4]; [e setBytes:&K length:4 atIndex:5];
            [e setBytes:&Kin length:4 atIndex:6]; [e setBytes:&NT length:4 atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        };
        gemv(bga,g_xg,g_gg,Iinter,D,D);
        gemv(bua,g_xg,g_uu,Iinter,D,D);
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:g_tile_silu];
        [e setBuffer:g_gg offset:0 atIndex:0]; [e setBuffer:g_uu offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake((size_t)R*Iinter,1,1)
            threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        gemv(bda,g_gg,g_hh,D,Iinter,Iinter);
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if(cb.status==MTLCommandBufferStatusError){
            fprintf(stderr,"[metal-tile] moe command buffer failed: %s\n",
                cb.error?[[cb.error localizedDescription]UTF8String]:"?");
            return 0;
        }
        double ks=[cb GPUEndTime]-[cb GPUStartTime];
        if(ks>0) st_kernel_ns.fetch_add((uint64_t)(ks*1.0e9));
        uint64_t scatter_began=tile_now_ns();
        const float *hh=(const float *)[g_hh contents];
        for(int gr=0;gr<R;gr++){
            float *os=out+(size_t)rows[gr]*D;
            const float *hr=hh+(size_t)gr*D;
            float weight=rw[gr];
            for(int j=0;j<D;j++) os[j]+=weight*hr[j];
        }
        st_scatter_ns.fetch_add(tile_now_ns()-scatter_began);
    }
    st_moe_calls.fetch_add(1);
    st_experts.fetch_add((uint64_t)nb);
    st_wall_ns.fetch_add(tile_now_ns()-began);
    return 1;
}

extern "C" int coli_metal_moe_block_mxfp4(int nb,int D,int Iinter,
                         const void *const *g,const void *const *u,const void *const *d,
                         const uint8_t *const *gs,const uint8_t *const *us,
                         const uint8_t *const *ds,
                         const float *xg,const int *xoff,const int *nr,
                         const int *rows,const float *rw,float *out,int S) {
    if(coli_metal_tile_enabled()){
        if(tile_moe_run(nb,D,Iinter,g,u,d,gs,us,ds,xg,xoff,nr,rows,rw,out))
            return 1;
        st_fallback.fetch_add(1);
    }
    return coli_metal_moe_block_mxfp4_row(nb,D,Iinter,g,u,d,gs,us,ds,
                                           xg,xoff,nr,rows,rw,out,S);
}

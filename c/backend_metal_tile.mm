#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend_metal.h"
#include "backend_metal_tile.h"
#include "mxfp4_apple8_tile.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
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
            g_tile_dev=nil;
            return;
        }
        id<MTLFunction> fn=[lib newFunctionWithName:@"mx_tile_single"];
        g_tile_single=[g_tile_dev newComputePipelineStateWithFunction:fn error:&err];
        if(!g_tile_queue||!g_tile_single||[g_tile_single threadExecutionWidth]!=32){
            fprintf(stderr,"[metal-tile] pipeline unavailable or SIMD width != 32\n");
            g_tile_single=nil;
            g_tile_queue=nil;
            g_tile_dev=nil;
            return;
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
    size_t capacity=0;
    unsigned pins=0;
    id<MTLBuffer> buffer=nil;
};

struct TileLease {
    size_t slot=std::numeric_limits<size_t>::max();
    id<MTLBuffer> buffer=nil;
};

static std::mutex g_cache_mutex;
static std::condition_variable g_cache_cv;
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
static std::atomic<uint64_t> st_cached_install_count{0},st_cached_install_bytes{0};
static std::atomic<uint64_t> st_cached_install_ns{0};
static std::atomic<uint64_t> st_single_calls{0},st_moe_calls{0},st_fallback{0};
static std::atomic<uint64_t> st_experts{0},st_wall_ns{0},st_kernel_ns{0},st_scatter_ns{0};
static std::atomic<uint64_t> st_row_mxfp4_calls{0},st_row_mxfp4_wall_ns{0};

extern "C" void coli_metal_tile_stats(ColiMetalTileStats *stats) {
    if(!stats) return;
    stats->repack_count=st_repack_count.load();
    stats->repack_bytes=st_repack_bytes.load();
    stats->repack_ns=st_repack_ns.load();
    stats->cached_install_count=st_cached_install_count.load();
    stats->cached_install_bytes=st_cached_install_bytes.load();
    stats->cached_install_ns=st_cached_install_ns.load();
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

static int same_key(const TileEntry& e,const void *weights,const void *scales,
                    int rows,int columns) {
    return e.weights==weights&&e.scales==scales&&e.rows==rows&&
           e.columns==columns;
}

static TileLease tile_acquire(const void *weights,const void *scales,
                              int rows,int columns) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    for(size_t i=0;i<g_cache.size();++i){
        TileEntry &entry=g_cache[i];
        if(same_key(entry,weights,scales,rows,columns)&&entry.buffer){
            entry.used=++g_cache_clock;
            entry.pins++;
            return TileLease{i,entry.buffer};
        }
    }
    return TileLease{};
}

static void tile_release(TileLease *lease) {
    if(!lease||lease->slot==std::numeric_limits<size_t>::max()) return;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if(lease->slot<g_cache.size()&&g_cache[lease->slot].pins)
            g_cache[lease->slot].pins--;
    }
    lease->slot=std::numeric_limits<size_t>::max();
    lease->buffer=nil;
    g_cache_cv.notify_all();
}

static int tile_store_packed(const void *weights,const void *scales,
                             int rows,int columns,uint64_t source_generation,
                             const void *packed,size_t bytes,int from_cache,
                             uint64_t began) {
    if(!packed||!bytes) return 0;
    std::unique_lock<std::mutex> lock(g_cache_mutex);
    for(;;){
        for(auto &entry:g_cache)
            if(same_key(entry,weights,scales,rows,columns)&&
               entry.generation==source_generation&&entry.buffer){
                entry.used=++g_cache_clock;
                uint64_t elapsed=tile_now_ns()-began;
                if(from_cache){
                    st_cached_install_count.fetch_add(1);
                    st_cached_install_bytes.fetch_add(bytes);
                    st_cached_install_ns.fetch_add(elapsed);
                } else {
                    st_repack_count.fetch_add(1);
                    st_repack_bytes.fetch_add(bytes);
                    st_repack_ns.fetch_add(elapsed);
                }
                return 1;
            }

        size_t slot=std::numeric_limits<size_t>::max();
        bool retry=false;
        for(size_t i=0;i<g_cache.size();++i){
            if(same_key(g_cache[i],weights,scales,rows,columns)){
                if(g_cache[i].pins==0) slot=i;
                else {
                    g_cache_cv.wait(lock);
                    retry=true;
                }
                break;
            }
        }
        if(retry) continue;
        if(slot==std::numeric_limits<size_t>::max()){
            if(g_cache.size()<tile_cache_capacity()){
                g_cache.emplace_back();
                slot=g_cache.size()-1;
            } else {
                uint64_t oldest=UINT64_MAX;
                for(size_t i=0;i<g_cache.size();++i){
                    if(!g_cache[i].pins&&g_cache[i].used<oldest){
                        oldest=g_cache[i].used;
                        slot=i;
                    }
                }
                if(slot==std::numeric_limits<size_t>::max()){
                    g_cache_cv.wait(lock);
                    continue;
                }
            }
        }
        if(slot==std::numeric_limits<size_t>::max()) continue;

        TileEntry &entry=g_cache[slot];
        if(!entry.buffer||entry.capacity<bytes){
            entry.buffer=[g_tile_dev newBufferWithLength:bytes
                options:MTLResourceStorageModeShared];
            if(!entry.buffer) return 0;
            entry.capacity=bytes;
        }
        memcpy([entry.buffer contents],packed,bytes);
        entry.weights=weights;
        entry.scales=scales;
        entry.rows=rows;
        entry.columns=columns;
        entry.generation=source_generation;
        entry.used=++g_cache_clock;
        entry.bytes=bytes;

        uint64_t elapsed=tile_now_ns()-began;
        if(from_cache){
            st_cached_install_count.fetch_add(1);
            st_cached_install_bytes.fetch_add(bytes);
            st_cached_install_ns.fetch_add(elapsed);
        } else {
            st_repack_count.fetch_add(1);
            st_repack_bytes.fetch_add(bytes);
            st_repack_ns.fetch_add(elapsed);
        }
        return 1;
    }
}

extern "C" int coli_metal_tile_prepare_matrix(const void *weights,const void *scales,
                                               int rows,int columns,
                                               uint64_t source_generation) {
    if(!coli_metal_tile_enabled()||!weights||!scales||rows<=0||columns<=0||
       !source_generation||!tile_init()) return 0;

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for(auto &entry:g_cache)
            if(same_key(entry,weights,scales,rows,columns)&&
               entry.generation==source_generation&&entry.buffer){
                entry.used=++g_cache_clock;
                return 1;
            }
    }

    size_t bytes=coli_mxfp4_apple8_tile_bytes((uint64_t)rows,(uint64_t)columns);
    size_t wbytes=(size_t)rows*((size_t)columns+1u)/2u;
    size_t sbytes=(size_t)rows*((size_t)columns+31u)/32u;
    if(!bytes) return 0;

    uint64_t began=tile_now_ns();
    thread_local std::vector<uint8_t> packed;
    packed.resize(bytes);
    if(coli_mxfp4_apple8_tile_repack(packed.data(),bytes,
            weights,wbytes,scales,sbytes,(uint64_t)rows,(uint64_t)columns))
        return 0;
    return tile_store_packed(weights,scales,rows,columns,source_generation,
                             packed.data(),bytes,0,began);
}

extern "C" int coli_metal_tile_prepare_packed_matrix(
        const void *weights,const void *scales,
        int rows,int columns,uint64_t source_generation,
        const void *tile_bytes,size_t tile_byte_count) {
    if(!coli_metal_tile_enabled()||!weights||!scales||rows<=0||columns<=0||
       !source_generation||!tile_init()||!tile_bytes) return 0;
    size_t expected=coli_mxfp4_apple8_tile_bytes(
        (uint64_t)rows,(uint64_t)columns);
    if(!expected||tile_byte_count!=expected) return 0;
    uint64_t began=tile_now_ns();
    return tile_store_packed(weights,scales,rows,columns,source_generation,
                             tile_bytes,tile_byte_count,1,began);
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
        if(!cb||!e) return 0;
        [e useResource:rec usage:MTLResourceUsageRead];
        [e setComputePipelineState:g_tile_single];
        [e setBuffer:rec offset:0 atIndex:0];
        [e setBuffer:bx offset:0 atIndex:1];
        [e setBuffer:by offset:0 atIndex:2];
        int NT=S*O;
        [e setBytes:&S length:4 atIndex:3];
        [e setBytes:&I length:4 atIndex:4];
        [e setBytes:&O length:4 atIndex:5];
        [e setBytes:&NT length:4 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1)
            threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if(cb.status==MTLCommandBufferStatusError){
            fprintf(stderr,"[metal-tile] single command buffer failed: %s\n",
                cb.error?[[cb.error localizedDescription]UTF8String]:"?");
            return 0;
        }
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
        TileLease lease=tile_acquire(weights,scales,O,I);
        if(lease.buffer){
            int ok=tile_single_run(y,x,lease.buffer,S,I,O);
            tile_release(&lease);
            if(ok) return 1;
        } else {
            tile_release(&lease);
        }
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

/* This milestone still leaves the fused routed-MoE wrapper on the proven row
 * implementation. The Apple8 path exercised by V4 is the per-matrix fmt7 shim;
 * milestone-2 adds persistence, not a second GPU lifetime model. */
extern "C" int coli_metal_moe_block_mxfp4(int nb,int D,int Iinter,
                         const void *const *g,const void *const *u,const void *const *d,
                         const uint8_t *const *gs,const uint8_t *const *us,
                         const uint8_t *const *ds,
                         const float *xg,const int *xoff,const int *nr,
                         const int *rows,const float *rw,float *out,int S) {
    if(coli_metal_tile_enabled()) st_fallback.fetch_add(1);
    return coli_metal_moe_block_mxfp4_row(nb,D,Iinter,g,u,d,gs,us,ds,
                                           xg,xoff,nr,rows,rw,out,S);
}

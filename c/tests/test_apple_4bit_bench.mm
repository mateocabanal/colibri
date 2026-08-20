#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mach/mach_time.h>

/*
 * Issue #131 research harness. This is deliberately NOT a production format ABI.
 * It is opt-in via V4_METAL_APPLE4_BENCH=1 and uses deterministic benchmark-only
 * physical layouts. The existing backend_metal.mm fmt==7 path is untouched.
 */

#include "apple4_bench_shader.h"
#include "apple4_bench_pack.h"

struct MetalBench {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> mxrow = nil, mxtile = nil, a16 = nil, a32 = nil, a64 = nil;

  bool init() {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;
    queue = [dev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:APPLE4_SHADER]
                                             options:nil error:&err];
    if (!lib) {
      fprintf(stderr, "APPLE4 shader compile failed: %s\n",
              err ? [[err localizedDescription] UTF8String] : "?");
      return false;
    }
    auto P = [&](NSString *n) -> id<MTLComputePipelineState> {
      NSError *pe = nil;
      id<MTLFunction> fn = [lib newFunctionWithName:n];
      id<MTLComputePipelineState> p = fn ? [dev newComputePipelineStateWithFunction:fn error:&pe] : nil;
      if (!p) fprintf(stderr, "APPLE4 pipeline %s failed: %s\n", [n UTF8String], pe ? [[pe localizedDescription] UTF8String] : "?");
      return p;
    };
    mxrow=P(@"bench_mx_row"); mxtile=P(@"bench_mx_tile");
    a16=P(@"bench_a4s16"); a32=P(@"bench_a4s32"); a64=P(@"bench_a4s64");
    if (!mxrow || !mxtile || !a16 || !a32 || !a64) return false;
    if ([mxrow threadExecutionWidth] != 32 || [mxtile threadExecutionWidth] != 32 ||
        [a16 threadExecutionWidth] != 32 || [a32 threadExecutionWidth] != 32 ||
        [a64 threadExecutionWidth] != 32) {
      fprintf(stderr, "APPLE4 requires 32-lane simdgroups for this Apple8 study\n");
      return false;
    }
    return true;
  }

  id<MTLComputePipelineState> pipe(Format f) const {
    switch (f) {
      case Format::MX_ROW: return mxrow;
      case Format::MX_TILE: return mxtile;
      case Format::A4S16: return a16;
      case Format::A4S32: return a32;
      case Format::A4S64: return a64;
    }
    return nil;
  }

  RunResult run(const Packed& p, const std::vector<float>& x,
                const std::vector<double>& ref, const std::vector<double>& mag,
                int S, int I, int O, int iters) {
    RunResult rr;
    id<MTLComputePipelineState> ps = pipe(p.fmt);
    if (!ps) return rr;
    id<MTLBuffer> br = [dev newBufferWithLength:std::max<size_t>(1,p.record.size()) options:MTLResourceStorageModeShared];
    memcpy([br contents], p.record.data(), p.record.size());
    id<MTLBuffer> bs = nil;
    if (p.fmt == Format::MX_ROW) {
      bs = [dev newBufferWithLength:std::max<size_t>(1,p.scales.size()) options:MTLResourceStorageModeShared];
      memcpy([bs contents], p.scales.data(), p.scales.size());
    }
    id<MTLBuffer> bx = [dev newBufferWithBytes:x.data() length:x.size()*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [dev newBufferWithLength:(size_t)S*O*sizeof(float) options:MTLResourceStorageModeShared];
    int NT = S*O;
    auto dispatch = [&](bool timed, double *gpu_ms, double *wall_ms) -> bool {
      double w0 = timed ? now_ms() : 0.0;
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:ps];
      if (p.fmt == Format::MX_ROW) {
        [e setBuffer:br offset:0 atIndex:0]; [e setBuffer:bs offset:0 atIndex:1];
        [e setBuffer:bx offset:0 atIndex:2]; [e setBuffer:by offset:0 atIndex:3];
        [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5];
        [e setBytes:&O length:4 atIndex:6]; [e setBytes:&NT length:4 atIndex:7];
      } else {
        [e setBuffer:br offset:0 atIndex:0]; [e setBuffer:bx offset:0 atIndex:1]; [e setBuffer:by offset:0 atIndex:2];
        [e setBytes:&S length:4 atIndex:3]; [e setBytes:&I length:4 atIndex:4];
        [e setBytes:&O length:4 atIndex:5]; [e setBytes:&NT length:4 atIndex:6];
      }
      [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1)
                        threadsPerThreadgroup:MTLSizeMake(128,1,1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if ([cb status] != MTLCommandBufferStatusCompleted) return false;
      if (timed) {
        double g0 = [cb GPUStartTime], g1 = [cb GPUEndTime];
        if (gpu_ms) *gpu_ms = (g1 > g0) ? (g1-g0)*1.0e3 : 0.0;
        if (wall_ms) *wall_ms = now_ms() - w0;
      }
      return true;
    };
    for (int i=0;i<5;i++) if (!dispatch(false,nullptr,nullptr)) return rr;
    std::vector<double> gpu, wall;
    for (int i=0;i<iters;i++) {
      double gm=0, wm=0;
      if (!dispatch(true,&gm,&wm)) return rr;
      if (gm > 0) gpu.push_back(gm);
      wall.push_back(wm);
    }
    std::vector<float> got((size_t)S*O);
    memcpy(got.data(), [by contents], got.size()*sizeof(float));
    double worst=0.0;
    for (size_t i=0;i<got.size();++i) {
      double d=std::fabs((double)got[i]-ref[i]);
      double rel=mag[i]>1e-30?d/mag[i]:d;
      worst=std::max(worst,rel);
    }
    rr.kernel_ms = gpu.empty() ? median(wall) : median(gpu);
    rr.wall_ms = median(wall);
    rr.max_rel = worst;
    double moved = (double)p.resident_bytes * S + (double)x.size()*sizeof(float) + (double)S*O*sizeof(float);
    rr.bytes_moved = (uint64_t)moved;
    rr.gbps = rr.kernel_ms > 0 ? moved / (rr.kernel_ms * 1.0e6) : 0.0;
    rr.ok = worst < 2e-4;
    return rr;
  }
};

static int env_iters() {
  const char *e = getenv("V4_METAL_APPLE4_ITERS");
  if (!e) return 25;
  int n = atoi(e);
  return std::max(5, std::min(200, n));
}

int main(void) {
  const char *gate = getenv("V4_METAL_APPLE4_BENCH");
  if (!gate || atoi(gate) == 0) {
    fprintf(stderr, "APPLE4 benchmark disabled; set V4_METAL_APPLE4_BENCH=1\n");
    return 0;
  }
  @autoreleasepool {
    MetalBench mb;
    if (!mb.init()) return 2;
    int iters = env_iters();
    printf("APPLE4 device=%s iters=%d\n", [[mb.dev name] UTF8String], iters);
    printf("shape,format,O,I,S,stored_bytes,resident_bytes,resident_bpw,mse,cosine,kernel_ms,wall_ms,bytes_moved,GBps,max_rel,status\n");
    const Shape shapes[] = {
      {"v4-gate-up",2048,4096,1,0},
      {"v4-down",4096,2048,1,0},
      {"v4-gate-up-S4",2048,4096,4,0},
      {"nonmult32",6,200,2,0},
      {"outlier-row",5,512,3,1},
    };
    double agg[5] = {0,0,0,0,0};
    uint64_t agg_bytes[5] = {0,0,0,0,0};
    int failures = 0;
    for (const Shape& sh : shapes) {
      std::vector<float> w = make_weights(sh.O,sh.I);
      std::vector<float> x = make_x(sh);
      Packed mxr = pack_mx_row(w,sh.O,sh.I);
      Packed mxt = pack_mx_tile(mxr,sh.O,sh.I);
      Packed a64 = pack_a4(w,sh.O,sh.I,64);
      Packed a32 = pack_a4(w,sh.O,sh.I,32);
      Packed a16 = pack_a4(w,sh.O,sh.I,16);
      Packed *all[] = {&mxr,&mxt,&a64,&a32,&a16};
      for (int fi=0;fi<5;fi++) {
        Packed& p=*all[fi];
        Metric qm=quality(w,p.dequant);
        std::vector<double> ref,mag;
        cpu_ref(p.dequant,x,ref,mag,sh.S,sh.I,sh.O);
        RunResult rr=mb.run(p,x,ref,mag,sh.S,sh.I,sh.O,iters);
        double bpw=(double)p.resident_bytes*8.0/((double)sh.O*sh.I);
        printf("%s,%s,%d,%d,%d,%zu,%zu,%.5f,%.9g,%.9f,%.6f,%.6f,%llu,%.3f,%.3e,%s\n",
               sh.name,format_name(p.fmt),sh.O,sh.I,sh.S,p.resident_bytes,p.resident_bytes,bpw,
               qm.mse,qm.cosine,rr.kernel_ms,rr.wall_ms,(unsigned long long)rr.bytes_moved,
               rr.gbps,rr.max_rel,rr.ok?"ok":"FAIL");
        failures += !rr.ok;
        if (sh.S==1 && (strcmp(sh.name,"v4-gate-up")==0 || strcmp(sh.name,"v4-down")==0)) {
          int idx = p.fmt==Format::MX_ROW?0:p.fmt==Format::MX_TILE?1:p.fmt==Format::A4S64?2:p.fmt==Format::A4S32?3:4;
          double mult = strcmp(sh.name,"v4-gate-up")==0 ? 2.0 : 1.0;
          agg[idx] += mult * rr.kernel_ms;
          agg_bytes[idx] += (uint64_t)mult * (uint64_t)p.resident_bytes;
        }
      }
    }
    const Format af[5]={Format::MX_ROW,Format::MX_TILE,Format::A4S64,Format::A4S32,Format::A4S16};
    printf("complete_expert,format,kernel_ms_gate_plus_up_plus_down,resident_bytes_gate_plus_up_plus_down\n");
    for(int i=0;i<5;i++) printf("complete_expert,%s,%.6f,%llu\n",format_name(af[i]),agg[i],(unsigned long long)agg_bytes[i]);
    printf("APPLE4 note: MXFP4-row-fmt7 mirrors backend_metal.mm fmt==7 semantics; Apple8 tile is exact byte-semantic repack. A4* are lossy experimental fixture layouts, not production #26 IDs.\n");
    return failures ? 3 : 0;
  }
}

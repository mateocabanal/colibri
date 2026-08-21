#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "apple8_metalio_direct.h"
#include "apple8_contract.h"
#include "metalio.h"

#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*
 * First direct-execution milestone for Apple8.
 *
 * MetalIO owns the persistent shared-storage allocation. Its public C API
 * intentionally exposes only the CPU-visible pointer, not id<MTLBuffer>, so
 * this bridge creates a cached no-copy Metal view over the exact same pages.
 * There is no byte copy and, importantly, no Apple8 -> canonical detile.
 * A later MetalIO ABI can expose the native MTLBuffer directly and remove even
 * this view construction without changing the kernel/layout contract below.
 */

static const char *APPLE8_SHADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float APPLE8_MX4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float apple8_ue8m0(uchar e) {
    /* Frozen MXFP4 contract: raw UE8M0 is an IEEE-754 exponent byte. */
    return as_type<float>((uint)e << 23);
}

kernel void apple8_mxfp4_matmul(
    device const uchar *tiles [[buffer(0)]],
    device const float *x     [[buffer(1)]],
    device float *y           [[buffer(2)]],
    constant int &S           [[buffer(3)]],
    constant int &I           [[buffer(4)]],
    constant int &O           [[buffer(5)]],
    uint tg                    [[threadgroup_position_in_grid]],
    uint lane                  [[thread_index_in_simdgroup]])
{
    const uint nt = (uint)(S * O);
    if (tg >= nt || lane >= 32) return;

    const int o = (int)(tg % (uint)O);
    const int s = (int)(tg / (uint)O);
    const int groups = (I + 31) / 32;
    const int output_tile = o >> 3;
    const int tile_row = o & 7;
    device const float *xr = x + (long)s * I;

    float acc = 0.0f;
    for (int g = 0; g < groups; ++g) {
        const int k = g * 32 + (int)lane;
        if (k >= I) continue;
        const long tile_index = (long)output_tile * groups + g;
        device const uchar *tile = tiles + tile_index * 136;
        const uchar packed = tile[tile_row * 16 + ((int)lane >> 1)];
        const uchar code = ((lane & 1u) != 0u) ? (packed >> 4) : (packed & 15u);
        const float scale = apple8_ue8m0(tile[128 + tile_row]);
        acc += APPLE8_MX4[code] * scale * xr[k];
    }

    acc = simd_sum(acc);
    if (lane == 0) y[(long)s * O + o] = acc;
}
)METAL";

struct SlotView {
    void *ptr;
    size_t bytes;
    id<MTLBuffer> buffer;
};

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipeline = nil;
static SlotView g_slot_views[256] = {};
static std::mutex g_lock;

static void clear_views_locked(void) {
    for (SlotView &view : g_slot_views) {
        view.buffer = nil;
        view.ptr = nullptr;
        view.bytes = 0;
    }
}

extern "C" int coli_apple8_metalio_direct_init(void) {
    std::lock_guard<std::mutex> guard(g_lock);
    if (g_pipeline && g_queue && g_device) return 1;
    if (!metalio_active()) return 0;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return 0;
    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        g_device = nil;
        return 0;
    }

    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:APPLE8_SHADER];
    id<MTLLibrary> library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        fprintf(stderr, "[apple8-metalio] shader compile failed: %s\n",
                error ? error.localizedDescription.UTF8String : "unknown");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"apple8_mxfp4_matmul"];
    if (!function) {
        fprintf(stderr, "[apple8-metalio] missing apple8_mxfp4_matmul shader\n");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    g_pipeline = [g_device newComputePipelineStateWithFunction:function error:&error];
    if (!g_pipeline) {
        fprintf(stderr, "[apple8-metalio] pipeline creation failed: %s\n",
                error ? error.localizedDescription.UTF8String : "unknown");
        g_queue = nil;
        g_device = nil;
        return 0;
    }
    clear_views_locked();
    return 1;
}

extern "C" void coli_apple8_metalio_direct_shutdown(void) {
    std::lock_guard<std::mutex> guard(g_lock);
    clear_views_locked();
    g_pipeline = nil;
    g_queue = nil;
    g_device = nil;
}

static id<MTLBuffer> slot_buffer_locked(int slot, size_t *slot_bytes_out) {
    if (slot < 0 || slot >= 256) return nil;
    void *ptr = metalio_slot_ptr(slot);
    size_t bytes = metalio_slot_bytes(slot);
    if (!ptr || !bytes) return nil;

    SlotView &view = g_slot_views[slot];
    if (!view.buffer || view.ptr != ptr || view.bytes != bytes) {
        /* Apple Metal's bytesNoCopy contract requires page-aligned storage.
         * MetalIO slots are 16-KiB aligned and sized by construction. */
        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:ptr
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (!buffer) return nil;
        view.ptr = ptr;
        view.bytes = bytes;
        view.buffer = buffer;
    }
    if (slot_bytes_out) *slot_bytes_out = bytes;
    return view.buffer;
}

extern "C" int coli_apple8_metalio_matmul_slot(int slot,
                                                 size_t slot_offset,
                                                 size_t matrix_bytes,
                                                 const float *x,
                                                 float *y,
                                                 int S,
                                                 int I,
                                                 int O) {
    if (!x || !y || S <= 0 || I <= 0 || O <= 0) return 0;
    uint64_t expected_u64 = 0;
    if (coli_apple8_tile_matrix_bytes((uint64_t)O, (uint64_t)I, &expected_u64) != 0 ||
        expected_u64 > SIZE_MAX || matrix_bytes != (size_t)expected_u64)
        return 0;

    std::lock_guard<std::mutex> guard(g_lock);
    if (!g_pipeline || !g_queue || !g_device) return 0;

    size_t slot_bytes = 0;
    id<MTLBuffer> weights = slot_buffer_locked(slot, &slot_bytes);
    if (!weights || slot_offset > slot_bytes || matrix_bytes > slot_bytes - slot_offset)
        return 0;

    size_t x_count = (size_t)S * (size_t)I;
    size_t y_count = (size_t)S * (size_t)O;
    if (I != 0 && x_count / (size_t)I != (size_t)S) return 0;
    if (O != 0 && y_count / (size_t)O != (size_t)S) return 0;
    if (x_count > SIZE_MAX / sizeof(float) || y_count > SIZE_MAX / sizeof(float)) return 0;
    size_t x_bytes = x_count * sizeof(float);
    size_t y_bytes = y_count * sizeof(float);

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x
                                             length:x_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [g_device newBufferWithLength:y_bytes
                                             options:MTLResourceStorageModeShared];
    if (!xb || !yb) return 0;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) return 0;
    [enc setComputePipelineState:g_pipeline];
    [enc setBuffer:weights offset:slot_offset atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:yb offset:0 atIndex:2];
    [enc setBytes:&S length:sizeof(S) atIndex:3];
    [enc setBytes:&I length:sizeof(I) atIndex:4];
    [enc setBytes:&O length:sizeof(O) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)S * (NSUInteger)O, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[apple8-metalio] GPU command failed: %s\n",
                cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
        return 0;
    }
    memcpy(y, yb.contents, y_bytes);
    metalio_slot_consumed(slot);
    return 1;
}

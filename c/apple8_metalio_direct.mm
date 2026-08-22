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
 * Direct Apple8 execution over the actual persistent MTLBuffer owned by
 * MetalIO. No Apple8 -> canonical MXFP4 detile/repack and no second Metal
 * buffer view are created.
 */

static const char *APPLE8_SHADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float APPLE8_MX4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float apple8_ue8m0(uchar e) {
    return as_type<float>((uint)e << 23);
}

/* One SIMD lane consumes one value from each 32-column MXFP4 group. */
inline float apple8_dot_partial(device const uchar *tiles,
                                device const float *x,
                                int I, int o, uint lane) {
    const int groups = (I + 31) / 32;
    const int output_tile = o >> 3;
    const int tile_row = o & 7;
    float acc = 0.0f;
    for (int g = 0; g < groups; ++g) {
        const int k = g * 32 + (int)lane;
        if (k >= I) continue;
        const long tile_index = (long)output_tile * groups + g;
        device const uchar *tile = tiles + tile_index * 136;
        const uchar packed = tile[tile_row * 16 + ((int)lane >> 1)];
        const uchar code = ((lane & 1u) != 0u) ? (packed >> 4) : (packed & 15u);
        acc += APPLE8_MX4[code] * apple8_ue8m0(tile[128 + tile_row]) * x[k];
    }
    return acc;
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
    device const float *xr = x + (long)s * I;
    float acc = simd_sum(apple8_dot_partial(tiles, xr, I, o, lane));
    if (lane == 0) y[(long)s * O + o] = acc;
}

/* gate/up are [M,H]. One SIMDgroup computes both rows and writes one
 * SwiGLU intermediate element. */
kernel void apple8_swiglu_gu(
    device const uchar *gate [[buffer(0)]],
    device const uchar *up   [[buffer(1)]],
    device const float *x    [[buffer(2)]],
    device float *mid        [[buffer(3)]],
    constant int &S          [[buffer(4)]],
    constant int &H          [[buffer(5)]],
    constant int &M          [[buffer(6)]],
    uint tg                   [[threadgroup_position_in_grid]],
    uint lane                 [[thread_index_in_simdgroup]])
{
    const uint nt = (uint)(S * M);
    if (tg >= nt || lane >= 32) return;
    const int m = (int)(tg % (uint)M);
    const int s = (int)(tg / (uint)M);
    device const float *xr = x + (long)s * H;
    float gv = simd_sum(apple8_dot_partial(gate, xr, H, m, lane));
    float uv = simd_sum(apple8_dot_partial(up, xr, H, m, lane));
    if (lane == 0) {
        const float silu = gv / (1.0f + exp(-gv));
        mid[(long)s * M + m] = silu * uv;
    }
}

/* down is [H,M]. */
kernel void apple8_swiglu_down(
    device const uchar *down [[buffer(0)]],
    device const float *mid  [[buffer(1)]],
    device float *y          [[buffer(2)]],
    constant int &S          [[buffer(3)]],
    constant int &H          [[buffer(4)]],
    constant int &M          [[buffer(5)]],
    uint tg                   [[threadgroup_position_in_grid]],
    uint lane                 [[thread_index_in_simdgroup]])
{
    const uint nt = (uint)(S * H);
    if (tg >= nt || lane >= 32) return;
    const int h = (int)(tg % (uint)H);
    const int s = (int)(tg / (uint)H);
    device const float *mr = mid + (long)s * M;
    float acc = simd_sum(apple8_dot_partial(down, mr, M, h, lane));
    if (lane == 0) y[(long)s * H + h] = acc;
}
)METAL";

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_matmul_pipeline = nil;
static id<MTLComputePipelineState> g_gu_pipeline = nil;
static id<MTLComputePipelineState> g_down_pipeline = nil;
static std::mutex g_lock;

static void clear_locked(void) {
    g_matmul_pipeline = nil;
    g_gu_pipeline = nil;
    g_down_pipeline = nil;
    g_queue = nil;
    g_device = nil;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLLibrary> library,
                                                 NSString *name,
                                                 NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) return nil;
    return [g_device newComputePipelineStateWithFunction:function error:error];
}

extern "C" int coli_apple8_metalio_direct_init(void) {
    std::lock_guard<std::mutex> guard(g_lock);
    if (g_matmul_pipeline && g_gu_pipeline && g_down_pipeline && g_queue && g_device)
        return 1;
    if (!metalio_active()) return 0;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return 0;
    g_queue = [g_device newCommandQueue];
    if (!g_queue) { clear_locked(); return 0; }

    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:APPLE8_SHADER];
    id<MTLLibrary> library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        fprintf(stderr, "[apple8-metalio] shader compile failed: %s\n",
                error ? error.localizedDescription.UTF8String : "unknown");
        clear_locked();
        return 0;
    }

    g_matmul_pipeline = make_pipeline(library, @"apple8_mxfp4_matmul", &error);
    if (!g_matmul_pipeline) goto pipeline_fail;
    g_gu_pipeline = make_pipeline(library, @"apple8_swiglu_gu", &error);
    if (!g_gu_pipeline) goto pipeline_fail;
    g_down_pipeline = make_pipeline(library, @"apple8_swiglu_down", &error);
    if (!g_down_pipeline) goto pipeline_fail;
    return 1;

pipeline_fail:
    fprintf(stderr, "[apple8-metalio] pipeline creation failed: %s\n",
            error ? error.localizedDescription.UTF8String : "missing function");
    clear_locked();
    return 0;
}

extern "C" void coli_apple8_metalio_direct_shutdown(void) {
    std::lock_guard<std::mutex> guard(g_lock);
    clear_locked();
}

static id<MTLBuffer> slot_buffer_locked(int slot, size_t *slot_bytes_out) {
    void *opaque = metalio_slot_native_buffer(slot);
    if (!opaque) return nil;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)opaque;
    if (!buffer) return nil;
    if (buffer.device != g_device && ![buffer.device isEqual:g_device]) return nil;
    if (slot_bytes_out) *slot_bytes_out = metalio_slot_bytes(slot);
    return buffer;
}

static int matrix_fits(size_t slot_bytes, size_t offset, size_t bytes,
                       int rows, int columns) {
    uint64_t expected = 0;
    if (rows <= 0 || columns <= 0 || (offset & 15u) != 0) return 0;
    if (coli_apple8_tile_matrix_bytes((uint64_t)rows, (uint64_t)columns, &expected) != 0 ||
        expected > SIZE_MAX || bytes != (size_t)expected)
        return 0;
    return offset <= slot_bytes && bytes <= slot_bytes - offset;
}

static int float_buffer_sizes(int S, int I, int O,
                              size_t *x_bytes, size_t *y_bytes) {
    if (S <= 0 || I <= 0 || O <= 0) return 0;
    size_t xs = (size_t)S, is = (size_t)I, os = (size_t)O;
    if (xs > SIZE_MAX / is || xs * is > SIZE_MAX / sizeof(float)) return 0;
    if (xs > SIZE_MAX / os || xs * os > SIZE_MAX / sizeof(float)) return 0;
    *x_bytes = xs * is * sizeof(float);
    *y_bytes = xs * os * sizeof(float);
    return 1;
}

extern "C" int coli_apple8_metalio_matmul_slot(int slot,
                                                 size_t slot_offset,
                                                 size_t matrix_bytes,
                                                 const float *x,
                                                 float *y,
                                                 int S,
                                                 int I,
                                                 int O) {
    if (!x || !y) return 0;
    std::lock_guard<std::mutex> guard(g_lock);
    if (!g_matmul_pipeline || !g_queue || !g_device) return 0;

    size_t slot_bytes = 0, x_bytes = 0, y_bytes = 0;
    id<MTLBuffer> weights = slot_buffer_locked(slot, &slot_bytes);
    if (!weights || !matrix_fits(slot_bytes, slot_offset, matrix_bytes, O, I) ||
        !float_buffer_sizes(S, I, O, &x_bytes, &y_bytes))
        return 0;

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x length:x_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [g_device newBufferWithLength:y_bytes
                                             options:MTLResourceStorageModeShared];
    if (!xb || !yb) return 0;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) return 0;
    [enc setComputePipelineState:g_matmul_pipeline];
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

extern "C" int coli_apple8_metalio_swiglu_slot(int slot,
                                                 size_t gate_offset,
                                                 size_t gate_bytes,
                                                 size_t up_offset,
                                                 size_t up_bytes,
                                                 size_t down_offset,
                                                 size_t down_bytes,
                                                 const float *x,
                                                 float *y,
                                                 int S,
                                                 int hidden,
                                                 int intermediate) {
    if (!x || !y || S <= 0 || hidden <= 0 || intermediate <= 0) return 0;
    std::lock_guard<std::mutex> guard(g_lock);
    if (!g_gu_pipeline || !g_down_pipeline || !g_queue || !g_device) return 0;

    size_t slot_bytes = 0, x_bytes = 0, y_bytes = 0;
    id<MTLBuffer> weights = slot_buffer_locked(slot, &slot_bytes);
    if (!weights ||
        !matrix_fits(slot_bytes, gate_offset, gate_bytes, intermediate, hidden) ||
        !matrix_fits(slot_bytes, up_offset, up_bytes, intermediate, hidden) ||
        !matrix_fits(slot_bytes, down_offset, down_bytes, hidden, intermediate) ||
        !float_buffer_sizes(S, hidden, hidden, &x_bytes, &y_bytes))
        return 0;

    size_t mid_count = (size_t)S * (size_t)intermediate;
    if ((size_t)S > SIZE_MAX / (size_t)intermediate ||
        mid_count > SIZE_MAX / sizeof(float))
        return 0;
    size_t mid_bytes = mid_count * sizeof(float);

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x length:x_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> mid = [g_device newBufferWithLength:mid_bytes
                                              options:MTLResourceStorageModePrivate];
    id<MTLBuffer> yb = [g_device newBufferWithLength:y_bytes
                                             options:MTLResourceStorageModeShared];
    if (!xb || !mid || !yb) return 0;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return 0;

    id<MTLComputeCommandEncoder> gu = [cb computeCommandEncoder];
    if (!gu) return 0;
    [gu setComputePipelineState:g_gu_pipeline];
    [gu setBuffer:weights offset:gate_offset atIndex:0];
    [gu setBuffer:weights offset:up_offset atIndex:1];
    [gu setBuffer:xb offset:0 atIndex:2];
    [gu setBuffer:mid offset:0 atIndex:3];
    [gu setBytes:&S length:sizeof(S) atIndex:4];
    [gu setBytes:&hidden length:sizeof(hidden) atIndex:5];
    [gu setBytes:&intermediate length:sizeof(intermediate) atIndex:6];
    [gu dispatchThreadgroups:MTLSizeMake((NSUInteger)S * (NSUInteger)intermediate, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [gu endEncoding];

    id<MTLComputeCommandEncoder> down = [cb computeCommandEncoder];
    if (!down) return 0;
    [down setComputePipelineState:g_down_pipeline];
    [down setBuffer:weights offset:down_offset atIndex:0];
    [down setBuffer:mid offset:0 atIndex:1];
    [down setBuffer:yb offset:0 atIndex:2];
    [down setBytes:&S length:sizeof(S) atIndex:3];
    [down setBytes:&hidden length:sizeof(hidden) atIndex:4];
    [down setBytes:&intermediate length:sizeof(intermediate) atIndex:5];
    [down dispatchThreadgroups:MTLSizeMake((NSUInteger)S * (NSUInteger)hidden, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [down endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[apple8-metalio] SwiGLU command failed: %s\n",
                cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
        return 0;
    }
    memcpy(y, yb.contents, y_bytes);
    metalio_slot_consumed(slot);
    return 1;
}

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "apple8_metalio_direct.h"
#include "apple8_contract.h"
#include "metalio.h"

#include <chrono>
#include <mutex>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

/* Keep the routed-expert reduction order identical to the host's top-k loop:
 * expert 0, then 1, ... K-1 for every hidden element. */
kernel void apple8_moe_reduce(
    device const float *expert_y [[buffer(0)]],
    device const float *weights  [[buffer(1)]],
    device float *y              [[buffer(2)]],
    constant int &K              [[buffer(3)]],
    constant int &H              [[buffer(4)]],
    uint h                        [[thread_position_in_grid]])
{
    if (h >= (uint)H) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i)
        acc += expert_y[(long)i * H + (long)h] * weights[i];
    y[h] = acc;
}
)METAL";

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_matmul_pipeline = nil;
static id<MTLComputePipelineState> g_gu_pipeline = nil;
static id<MTLComputePipelineState> g_down_pipeline = nil;
static id<MTLComputePipelineState> g_reduce_pipeline = nil;
static std::mutex g_lock;

static struct {
    uint64_t encode_ns, submit_ns, wait_ns, kernel_ns;
    uint64_t command_buffers, fused_calls, fused_experts;
} g_prof;

/* Split-phase fused MoE handle. The Objective-C object fields are retained by
 * ARC, so the command buffer and shared output remain alive while the host
 * executes independent CPU work between begin() and finish(). */
struct Apple8MoePending {
    id<MTLCommandBuffer> cb = nil;
    id<MTLBuffer> yb = nil;
    int slots[64] = {};
    int expert_count = 0;
    size_t y_bytes = 0;
    uint64_t encode_ns = 0;
    uint64_t submit_ns = 0;
};

static uint64_t direct_now_ns(void) {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

static void profile_completed_locked(id<MTLCommandBuffer> cb,
                                     uint64_t encode_ns,
                                     uint64_t submit_ns,
                                     uint64_t wait_ns,
                                     int fused_experts) {
    g_prof.encode_ns += encode_ns;
    g_prof.submit_ns += submit_ns;
    g_prof.wait_ns += wait_ns;
    g_prof.command_buffers++;
    if (cb.GPUEndTime > cb.GPUStartTime && cb.GPUStartTime > 0.0)
        g_prof.kernel_ns += (uint64_t)((cb.GPUEndTime - cb.GPUStartTime) * 1.0e9);
    if (fused_experts > 0) {
        g_prof.fused_calls++;
        g_prof.fused_experts += (uint64_t)fused_experts;
    }
}

static void clear_locked(void) {
    g_matmul_pipeline = nil;
    g_gu_pipeline = nil;
    g_down_pipeline = nil;
    g_reduce_pipeline = nil;
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
    if (g_matmul_pipeline && g_gu_pipeline && g_down_pipeline && g_reduce_pipeline &&
        g_queue && g_device)
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
    g_reduce_pipeline = make_pipeline(library, @"apple8_moe_reduce", &error);
    if (!g_reduce_pipeline) goto pipeline_fail;
    memset(&g_prof, 0, sizeof(g_prof));
    return 1;

pipeline_fail:
    fprintf(stderr, "[apple8-metalio] pipeline creation failed: %s\n",
            error ? error.localizedDescription.UTF8String : "missing function");
    clear_locked();
    return 0;
}

extern "C" void coli_apple8_metalio_direct_shutdown(void) {
    std::lock_guard<std::mutex> guard(g_lock);
    const char *profile = getenv("QWEN_PROFILE");
    if (profile && profile[0] && strcmp(profile, "0") != 0 && g_prof.command_buffers) {
        fprintf(stderr,
                "[apple8-metalio-profile] command_buffers=%llu fused_layers=%llu "
                "fused_experts=%llu metal_encode_ms=%.3f metal_submit_ms=%.3f "
                "metal_wait_ms=%.3f metal_kernel_ms=%.3f\n",
                (unsigned long long)g_prof.command_buffers,
                (unsigned long long)g_prof.fused_calls,
                (unsigned long long)g_prof.fused_experts,
                (double)g_prof.encode_ns / 1.0e6,
                (double)g_prof.submit_ns / 1.0e6,
                (double)g_prof.wait_ns / 1.0e6,
                (double)g_prof.kernel_ns / 1.0e6);
    }
    clear_locked();
}

extern "C" void coli_apple8_metalio_profile_get(uint64_t *encode_ns,
                                                  uint64_t *submit_ns,
                                                  uint64_t *wait_ns,
                                                  uint64_t *kernel_ns,
                                                  uint64_t *fused_calls,
                                                  uint64_t *fused_experts) {
    std::lock_guard<std::mutex> guard(g_lock);
    if (encode_ns) *encode_ns = g_prof.encode_ns;
    if (submit_ns) *submit_ns = g_prof.submit_ns;
    if (wait_ns) *wait_ns = g_prof.wait_ns;
    if (kernel_ns) *kernel_ns = g_prof.kernel_ns;
    if (fused_calls) *fused_calls = g_prof.fused_calls;
    if (fused_experts) *fused_experts = g_prof.fused_experts;
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

    uint64_t encode_begin = direct_now_ns();
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
    uint64_t encode_ns = direct_now_ns() - encode_begin;
    uint64_t submit_begin = direct_now_ns();
    [cb commit];
    uint64_t submit_ns = direct_now_ns() - submit_begin;
    uint64_t wait_begin = direct_now_ns();
    [cb waitUntilCompleted];
    uint64_t wait_ns = direct_now_ns() - wait_begin;
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[apple8-metalio] GPU command failed: %s\n",
                cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
        return 0;
    }
    profile_completed_locked(cb, encode_ns, submit_ns, wait_ns, 0);
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

    uint64_t encode_begin = direct_now_ns();
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

    uint64_t encode_ns = direct_now_ns() - encode_begin;
    uint64_t submit_begin = direct_now_ns();
    [cb commit];
    uint64_t submit_ns = direct_now_ns() - submit_begin;
    uint64_t wait_begin = direct_now_ns();
    [cb waitUntilCompleted];
    uint64_t wait_ns = direct_now_ns() - wait_begin;
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[apple8-metalio] SwiGLU command failed: %s\n",
                cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
        return 0;
    }
    profile_completed_locked(cb, encode_ns, submit_ns, wait_ns, 0);
    memcpy(y, yb.contents, y_bytes);
    metalio_slot_consumed(slot);
    return 1;
}

/* Split-phase fused routed MoE. begin() performs the exact same validation,
 * encoding and submission as the synchronous entry point, but deliberately
 * leaves the command buffer in flight. finish() is the first host-side
 * synchronization point and accounts only the residual time the CPU actually
 * blocked after doing useful independent work. */
extern "C" int coli_apple8_metalio_moe_topk_begin(
    const ColiApple8MetalioExpert *experts,
    const float *route_weights,
    int expert_count,
    const float *x,
    int hidden,
    int intermediate,
    void **pending_out) {
    if (pending_out) *pending_out = nullptr;
    if (!experts || !route_weights || !x || !pending_out || expert_count <= 0 ||
        expert_count > 64 || hidden <= 0 || intermediate <= 0)
        return 0;
    std::lock_guard<std::mutex> guard(g_lock);
    if (!g_gu_pipeline || !g_down_pipeline || !g_reduce_pipeline ||
        !g_queue || !g_device)
        return 0;

    const size_t H = (size_t)hidden, M = (size_t)intermediate, K = (size_t)expert_count;
    if (H > SIZE_MAX / sizeof(float) || M > SIZE_MAX / sizeof(float) ||
        K > SIZE_MAX / M || K * M > SIZE_MAX / sizeof(float) ||
        K > SIZE_MAX / H || K * H > SIZE_MAX / sizeof(float))
        return 0;
    const size_t x_bytes = H * sizeof(float);
    const size_t y_bytes = H * sizeof(float);
    const size_t mid_stride = M * sizeof(float);
    const size_t out_stride = H * sizeof(float);
    const size_t mid_bytes = K * mid_stride;
    const size_t expert_y_bytes = K * out_stride;

    id<MTLBuffer> weight_buffers[64] = {};
    for (int i = 0; i < expert_count; ++i) {
        size_t slot_bytes = 0;
        weight_buffers[i] = slot_buffer_locked(experts[i].slot, &slot_bytes);
        if (!weight_buffers[i] ||
            !matrix_fits(slot_bytes, experts[i].gate_offset, experts[i].gate_bytes,
                         intermediate, hidden) ||
            !matrix_fits(slot_bytes, experts[i].up_offset, experts[i].up_bytes,
                         intermediate, hidden) ||
            !matrix_fits(slot_bytes, experts[i].down_offset, experts[i].down_bytes,
                         hidden, intermediate))
            return 0;
    }

    id<MTLBuffer> xb = [g_device newBufferWithBytes:x length:x_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> mid = [g_device newBufferWithLength:mid_bytes
                                              options:MTLResourceStorageModePrivate];
    id<MTLBuffer> expert_y = [g_device newBufferWithLength:expert_y_bytes
                                                   options:MTLResourceStorageModePrivate];
    id<MTLBuffer> rw = [g_device newBufferWithBytes:route_weights
                                             length:K * sizeof(float)
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [g_device newBufferWithLength:y_bytes
                                             options:MTLResourceStorageModeShared];
    if (!xb || !mid || !expert_y || !rw || !yb) return 0;

    Apple8MoePending *pending = new (std::nothrow) Apple8MoePending();
    if (!pending) return 0;

    const int S = 1;
    uint64_t encode_begin = direct_now_ns();
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) { delete pending; return 0; }

    /* Stage 1: every gate+up/SwiGLU dispatch. No host wait and no command-buffer
     * boundary between experts. */
    id<MTLComputeCommandEncoder> gu = [cb computeCommandEncoder];
    if (!gu) { delete pending; return 0; }
    [gu setComputePipelineState:g_gu_pipeline];
    for (int i = 0; i < expert_count; ++i) {
        [gu setBuffer:weight_buffers[i] offset:experts[i].gate_offset atIndex:0];
        [gu setBuffer:weight_buffers[i] offset:experts[i].up_offset atIndex:1];
        [gu setBuffer:xb offset:0 atIndex:2];
        [gu setBuffer:mid offset:(NSUInteger)i * mid_stride atIndex:3];
        [gu setBytes:&S length:sizeof(S) atIndex:4];
        [gu setBytes:&hidden length:sizeof(hidden) atIndex:5];
        [gu setBytes:&intermediate length:sizeof(intermediate) atIndex:6];
        [gu dispatchThreadgroups:MTLSizeMake((NSUInteger)intermediate, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [gu endEncoding];

    /* Stage 2: all downs consume the private intermediates produced above.
     * Encoder ordering in one command buffer establishes the dependency. */
    id<MTLComputeCommandEncoder> down = [cb computeCommandEncoder];
    if (!down) { delete pending; return 0; }
    [down setComputePipelineState:g_down_pipeline];
    for (int i = 0; i < expert_count; ++i) {
        [down setBuffer:weight_buffers[i] offset:experts[i].down_offset atIndex:0];
        [down setBuffer:mid offset:(NSUInteger)i * mid_stride atIndex:1];
        [down setBuffer:expert_y offset:(NSUInteger)i * out_stride atIndex:2];
        [down setBytes:&S length:sizeof(S) atIndex:3];
        [down setBytes:&hidden length:sizeof(hidden) atIndex:4];
        [down setBytes:&intermediate length:sizeof(intermediate) atIndex:5];
        [down dispatchThreadgroups:MTLSizeMake((NSUInteger)hidden, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [down endEncoding];

    /* Stage 3: one deterministic top-k reduction, still in the same command
     * buffer. This replaces K host readbacks + K CPU scatter-adds. */
    id<MTLComputeCommandEncoder> reduce = [cb computeCommandEncoder];
    if (!reduce) { delete pending; return 0; }
    [reduce setComputePipelineState:g_reduce_pipeline];
    [reduce setBuffer:expert_y offset:0 atIndex:0];
    [reduce setBuffer:rw offset:0 atIndex:1];
    [reduce setBuffer:yb offset:0 atIndex:2];
    [reduce setBytes:&expert_count length:sizeof(expert_count) atIndex:3];
    [reduce setBytes:&hidden length:sizeof(hidden) atIndex:4];
    NSUInteger threads = g_reduce_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256) threads = 256;
    if (threads < 1) { delete pending; return 0; }
    NSUInteger groups = ((NSUInteger)hidden + threads - 1) / threads;
    [reduce dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [reduce endEncoding];

    uint64_t encode_ns = direct_now_ns() - encode_begin;
    uint64_t submit_begin = direct_now_ns();
    [cb commit];
    uint64_t submit_ns = direct_now_ns() - submit_begin;

    pending->cb = cb;
    pending->yb = yb;
    pending->expert_count = expert_count;
    pending->y_bytes = y_bytes;
    pending->encode_ns = encode_ns;
    pending->submit_ns = submit_ns;
    for (int i = 0; i < expert_count; ++i) pending->slots[i] = experts[i].slot;
    *pending_out = pending;
    return 1;
}

extern "C" int coli_apple8_metalio_moe_topk_finish(void *opaque, float *y) {
    if (!opaque || !y) return 0;
    Apple8MoePending *pending = static_cast<Apple8MoePending *>(opaque);
    uint64_t wait_begin = direct_now_ns();
    [pending->cb waitUntilCompleted];
    uint64_t wait_ns = direct_now_ns() - wait_begin;
    const int ok = pending->cb.status == MTLCommandBufferStatusCompleted;
    if (ok) memcpy(y, pending->yb.contents, pending->y_bytes);

    {
        std::lock_guard<std::mutex> guard(g_lock);
        if (ok)
            profile_completed_locked(pending->cb, pending->encode_ns,
                                     pending->submit_ns, wait_ns,
                                     pending->expert_count);
        for (int i = 0; i < pending->expert_count; ++i)
            metalio_slot_consumed(pending->slots[i]);
    }
    if (!ok) {
        fprintf(stderr, "[apple8-metalio] fused top-k command failed: %s\n",
                pending->cb.error ? pending->cb.error.localizedDescription.UTF8String : "unknown");
    }
    delete pending;
    return ok;
}

extern "C" int coli_apple8_metalio_moe_topk(const ColiApple8MetalioExpert *experts,
                                              const float *route_weights,
                                              int expert_count,
                                              const float *x,
                                              float *y,
                                              int hidden,
                                              int intermediate) {
    if (!y) return 0;
    void *pending = nullptr;
    if (!coli_apple8_metalio_moe_topk_begin(experts, route_weights, expert_count,
                                             x, hidden, intermediate, &pending))
        return 0;
    return coli_apple8_metalio_moe_topk_finish(pending, y);
}

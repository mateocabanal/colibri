#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "apple8_metalio_direct.h"
#include "apple8_contract.h"
#include "metalio.h"

#include <chrono>
#include <limits.h>
#include <mutex>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

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

/*
 * One SIMD lane consumes one value from each 32-column MXFP4 group.
 *
 * The old helper recomputed output_tile*groups+g, the 136-byte tile address,
 * and a tail predicate for every group. Decode dimensions are overwhelmingly
 * full 32-column groups, so walk the physical tile stream directly and handle
 * at most one tail group separately. This keeps the exact accumulation order
 * while removing integer/address work from the hot loop.
 */
inline float apple8_dot_partial(device const uchar *tiles,
                                device const float *x,
                                int I, int o, uint lane) {
    const int groups = (I + 31) >> 5;
    const int full_groups = I >> 5;
    const int tail = I & 31;
    const int output_tile = o >> 3;
    const int tile_row = o & 7;
    const int packed_off = tile_row * 16 + ((int)lane >> 1);
    const int scale_off = 128 + tile_row;
    device const uchar *tile = tiles + (long)output_tile * groups * 136;
    float acc = 0.0f;

    int g = 0;
    for (; g + 3 < full_groups; g += 4) {
        device const uchar *t0 = tile;
        device const uchar *t1 = t0 + 136;
        device const uchar *t2 = t1 + 136;
        device const uchar *t3 = t2 + 136;
        const uchar p0 = t0[packed_off];
        const uchar p1 = t1[packed_off];
        const uchar p2 = t2[packed_off];
        const uchar p3 = t3[packed_off];
        const uchar c0 = ((lane & 1u) != 0u) ? (p0 >> 4) : (p0 & 15u);
        const uchar c1 = ((lane & 1u) != 0u) ? (p1 >> 4) : (p1 & 15u);
        const uchar c2 = ((lane & 1u) != 0u) ? (p2 >> 4) : (p2 & 15u);
        const uchar c3 = ((lane & 1u) != 0u) ? (p3 >> 4) : (p3 & 15u);
        acc += APPLE8_MX4[c0] * apple8_ue8m0(t0[scale_off]) * x[((g + 0) << 5) + (int)lane];
        acc += APPLE8_MX4[c1] * apple8_ue8m0(t1[scale_off]) * x[((g + 1) << 5) + (int)lane];
        acc += APPLE8_MX4[c2] * apple8_ue8m0(t2[scale_off]) * x[((g + 2) << 5) + (int)lane];
        acc += APPLE8_MX4[c3] * apple8_ue8m0(t3[scale_off]) * x[((g + 3) << 5) + (int)lane];
        tile += 4 * 136;
    }
    for (; g < full_groups; ++g, tile += 136) {
        const uchar packed = tile[packed_off];
        const uchar code = ((lane & 1u) != 0u) ? (packed >> 4) : (packed & 15u);
        acc += APPLE8_MX4[code] * apple8_ue8m0(tile[scale_off]) * x[(g << 5) + (int)lane];
    }
    if (tail && (int)lane < tail) {
        const uchar packed = tile[packed_off];
        const uchar code = ((lane & 1u) != 0u) ? (packed >> 4) : (packed & 15u);
        acc += APPLE8_MX4[code] * apple8_ue8m0(tile[scale_off]) * x[(full_groups << 5) + (int)lane];
    }
    return acc;
}

/* Gate/up have identical geometry. Traverse the two tile streams together so
 * x is fetched once per group and both projections have independent work in
 * flight. Per-projection accumulation order is unchanged. */
inline float2 apple8_dot_pair_partial(device const uchar *gate_tiles,
                                      device const uchar *up_tiles,
                                      device const float *x,
                                      int I, int o, uint lane) {
    const int groups = (I + 31) >> 5;
    const int full_groups = I >> 5;
    const int tail = I & 31;
    const int output_tile = o >> 3;
    const int tile_row = o & 7;
    const int packed_off = tile_row * 16 + ((int)lane >> 1);
    const int scale_off = 128 + tile_row;
    device const uchar *gt = gate_tiles + (long)output_tile * groups * 136;
    device const uchar *ut = up_tiles + (long)output_tile * groups * 136;
    float ga = 0.0f, ua = 0.0f;

    int g = 0;
    for (; g + 3 < full_groups; g += 4) {
        for (int j = 0; j < 4; ++j) {
            device const uchar *gtt = gt + (long)j * 136;
            device const uchar *utt = ut + (long)j * 136;
            const float xv = x[((g + j) << 5) + (int)lane];
            const uchar gp = gtt[packed_off];
            const uchar up = utt[packed_off];
            const uchar gc = ((lane & 1u) != 0u) ? (gp >> 4) : (gp & 15u);
            const uchar uc = ((lane & 1u) != 0u) ? (up >> 4) : (up & 15u);
            ga += APPLE8_MX4[gc] * apple8_ue8m0(gtt[scale_off]) * xv;
            ua += APPLE8_MX4[uc] * apple8_ue8m0(utt[scale_off]) * xv;
        }
        gt += 4 * 136;
        ut += 4 * 136;
    }
    for (; g < full_groups; ++g, gt += 136, ut += 136) {
        const float xv = x[(g << 5) + (int)lane];
        const uchar gp = gt[packed_off];
        const uchar up = ut[packed_off];
        const uchar gc = ((lane & 1u) != 0u) ? (gp >> 4) : (gp & 15u);
        const uchar uc = ((lane & 1u) != 0u) ? (up >> 4) : (up & 15u);
        ga += APPLE8_MX4[gc] * apple8_ue8m0(gt[scale_off]) * xv;
        ua += APPLE8_MX4[uc] * apple8_ue8m0(ut[scale_off]) * xv;
    }
    if (tail && (int)lane < tail) {
        const float xv = x[(full_groups << 5) + (int)lane];
        const uchar gp = gt[packed_off];
        const uchar up = ut[packed_off];
        const uchar gc = ((lane & 1u) != 0u) ? (gp >> 4) : (gp & 15u);
        const uchar uc = ((lane & 1u) != 0u) ? (up >> 4) : (up & 15u);
        ga += APPLE8_MX4[gc] * apple8_ue8m0(gt[scale_off]) * xv;
        ua += APPLE8_MX4[uc] * apple8_ue8m0(ut[scale_off]) * xv;
    }
    return float2(ga, ua);
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
    const float2 gu = apple8_dot_pair_partial(gate, up, xr, H, m, lane);
    const float gv = simd_sum(gu.x);
    const float uv = simd_sum(gu.y);
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

/* Dense BF16 projection. Eight rows share a 256-thread group (one SIMDgroup
 * per row). The critical difference from the regressed contiguous-span path
 * is that lane N always reads element N+32*g, so every memory instruction
 * touches a contiguous 128B x segment and 64B BF16 weight segment. The
 * per-lane accumulation sequence and fixed lane-0 0..31 reduction are kept
 * deterministic. */
constant uint QWEN_HEAD_ROWS_PER_TG = 8u;
constant uint QWEN_HEAD_DOT_LANES = 32u;
constant uint QWEN_HEAD_DOT_THREADS = QWEN_HEAD_ROWS_PER_TG * QWEN_HEAD_DOT_LANES;

inline float qwen_bf16(device const ushort *p, long i) {
    return as_type<float>((uint)p[i] << 16);
}

inline float qwen_bf16_dot_lane(device const ushort *wr,
                                device const float *xr,
                                int I,
                                uint lane) {
    float acc = 0.0f;
    int i = (int)lane;
    /* Explicit 4-way unroll keeps the same per-lane summation order while
     * giving the compiler four coalesced memory instructions per loop body. */
    for (; i + 96 < I; i += 128) {
        acc += xr[i + 0]  * qwen_bf16(wr, i + 0);
        acc += xr[i + 32] * qwen_bf16(wr, i + 32);
        acc += xr[i + 64] * qwen_bf16(wr, i + 64);
        acc += xr[i + 96] * qwen_bf16(wr, i + 96);
    }
    for (; i < I; i += 32)
        acc += xr[i] * qwen_bf16(wr, i);
    return acc;
}

kernel void qwen_bf16_matmul(
    device const ushort *w [[buffer(0)]],
    device const float *x  [[buffer(1)]],
    device float *y        [[buffer(2)]],
    constant int &S        [[buffer(3)]],
    constant int &O        [[buffer(4)]],
    constant int &I        [[buffer(5)]],
    uint tg                [[threadgroup_position_in_grid]],
    uint tid               [[thread_index_in_threadgroup]])
{
    threadgroup float partial[QWEN_HEAD_DOT_THREADS];
    const uint row_slot = tid / QWEN_HEAD_DOT_LANES;
    const uint lane = tid - row_slot * QWEN_HEAD_DOT_LANES;
    const uint row = tg * QWEN_HEAD_ROWS_PER_TG + row_slot;
    const uint total = (uint)(S * O);
    float acc = 0.0f;
    if (row < total) {
        const int s = (int)(row / (uint)O);
        const int o = (int)(row % (uint)O);
        device const float *xr = x + (long)s * I;
        device const ushort *wr = w + (long)o * I;
        acc = qwen_bf16_dot_lane(wr, xr, I, lane);
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row < total && lane == 0) {
        float sum = 0.0f;
        const uint pb = row_slot * QWEN_HEAD_DOT_LANES;
        for (uint p = 0; p < QWEN_HEAD_DOT_LANES; ++p) sum += partial[pb + p];
        y[row] = sum;
    }
}

/* Merged top-k MoE layer. One 256-thread group now maps exactly to one
 * physical Apple8 8-row output tile: SIMDgroup r computes tile row r. This
 * reduces threadgroup scheduling by 8x versus the old one-row/32-thread grid
 * and keeps the eight row consumers of a 136-byte tile adjacent. */
kernel void apple8_moe_gu8(
    device const uchar *e0 [[buffer(0)]],
    device const uchar *e1 [[buffer(1)]],
    device const uchar *e2 [[buffer(2)]],
    device const uchar *e3 [[buffer(3)]],
    device const uchar *e4 [[buffer(4)]],
    device const uchar *e5 [[buffer(5)]],
    device const uchar *e6 [[buffer(6)]],
    device const uchar *e7 [[buffer(7)]],
    device const float *x  [[buffer(8)]],
    device float *mid      [[buffer(9)]],
    constant int &S        [[buffer(10)]],
    constant int &H        [[buffer(11)]],
    constant int &M        [[buffer(12)]],
    constant int &K        [[buffer(13)]],
    constant int *gate_off [[buffer(14)]],
    constant int *up_off   [[buffer(15)]],
    constant int *down_off [[buffer(16)]],
    uint tg                [[threadgroup_position_in_grid]],
    uint simd_row          [[simdgroup_index_in_threadgroup]],
    uint lane              [[thread_index_in_simdgroup]])
{
    const int mtiles = (M + 7) >> 3;
    const int mt = (int)(tg % (uint)mtiles);
    const int s = (int)((tg / (uint)mtiles) % (uint)S);
    const int e = (int)(tg / ((uint)mtiles * (uint)S));
    const int m = (mt << 3) + (int)simd_row;
    if (e >= K || simd_row >= 8u || m >= M || lane >= 32u) return;
    device const uchar *base = e0;
    if (e == 1) base = e1;
    else if (e == 2) base = e2;
    else if (e == 3) base = e3;
    else if (e == 4) base = e4;
    else if (e == 5) base = e5;
    else if (e == 6) base = e6;
    else if (e == 7) base = e7;
    device const float *xr = x + (long)s * H;
    const float2 gu = apple8_dot_pair_partial(base + gate_off[e], base + up_off[e], xr, H, m, lane);
    const float gv = simd_sum(gu.x);
    const float uv = simd_sum(gu.y);
    if (lane == 0) {
        const float silu = gv / (1.0f + exp(-gv));
        mid[(long)(e * S + s) * M + m] = silu * uv;
    }
}

kernel void apple8_moe_down8(
    device const uchar *e0 [[buffer(0)]],
    device const uchar *e1 [[buffer(1)]],
    device const uchar *e2 [[buffer(2)]],
    device const uchar *e3 [[buffer(3)]],
    device const uchar *e4 [[buffer(4)]],
    device const uchar *e5 [[buffer(5)]],
    device const uchar *e6 [[buffer(6)]],
    device const uchar *e7 [[buffer(7)]],
    device const float *mid      [[buffer(8)]],
    device float *expert_y       [[buffer(9)]],
    constant int &S              [[buffer(10)]],
    constant int &H              [[buffer(11)]],
    constant int &M              [[buffer(12)]],
    constant int &K              [[buffer(13)]],
    constant int *gate_off       [[buffer(14)]],
    constant int *up_off         [[buffer(15)]],
    constant int *down_off       [[buffer(16)]],
    uint tg                      [[threadgroup_position_in_grid]],
    uint simd_row                [[simdgroup_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]])
{
    const int htiles = (H + 7) >> 3;
    const int ht = (int)(tg % (uint)htiles);
    const int s = (int)((tg / (uint)htiles) % (uint)S);
    const int e = (int)(tg / ((uint)htiles * (uint)S));
    const int h = (ht << 3) + (int)simd_row;
    if (e >= K || simd_row >= 8u || h >= H || lane >= 32u) return;
    device const uchar *base = e0;
    if (e == 1) base = e1;
    else if (e == 2) base = e2;
    else if (e == 3) base = e3;
    else if (e == 4) base = e4;
    else if (e == 5) base = e5;
    else if (e == 6) base = e6;
    else if (e == 7) base = e7;
    device const float *mr = mid + (long)(e * S + s) * M;
    float acc = simd_sum(apple8_dot_partial(base + down_off[e], mr, M, h, lane));
    if (lane == 0) expert_y[(long)(e * S + s) * H + h] = acc;
}
)METAL";

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_matmul_pipeline = nil;
static id<MTLComputePipelineState> g_gu_pipeline = nil;
static id<MTLComputePipelineState> g_down_pipeline = nil;
static id<MTLComputePipelineState> g_reduce_pipeline = nil;
static id<MTLComputePipelineState> g_bf16_matmul_pipeline = nil;
static id<MTLComputePipelineState> g_gu8_pipeline = nil;
static id<MTLComputePipelineState> g_down8_pipeline = nil;
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

/* Dense BF16 weight buffer cache: the lm_head pointer is stable for the
 * model lifetime, so the 622 MB copy happens once and every later token
 * reuses the buffer. */
static const uint16_t *g_bf16_w = NULL;
static id<MTLBuffer> g_bf16_wbuf = nil;
static int g_bf16_O = 0, g_bf16_I = 0;

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

static int qwen_gdn_init_locked(void);
static void qwen_gdn_clear_locked(void);

static void clear_locked(void) {
    qwen_gdn_clear_locked();
    g_matmul_pipeline = nil;
    g_gu_pipeline = nil;
    g_down_pipeline = nil;
    g_reduce_pipeline = nil;
    g_bf16_matmul_pipeline = nil;
    g_bf16_wbuf = nil;
    g_bf16_w = NULL;
    g_bf16_O = g_bf16_I = 0;
    g_gu8_pipeline = nil;
    g_down8_pipeline = nil;
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

/* Decode-only Qwen Gated DeltaNet path. It shares the direct Apple8 device,
 * queue, lock and profiler. Dense BF16 weights plus recurrent/conv state are
 * page-aligned by qwen_moe.c and wrapped zero-copy as Shared buffers, so CPU
 * prefill/reset/prefix-cache and Metal decode see one authoritative UMA state. */
static const char *QWEN_GDN_SHADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint QWEN_GDN_ROWS_PER_TG = 8u;
constant uint QWEN_GDN_DOT_LANES = 32u;
constant uint QWEN_GDN_DOT_THREADS = QWEN_GDN_ROWS_PER_TG * QWEN_GDN_DOT_LANES;

inline float qwen_bf16(device const ushort *p, long i) {
    return as_type<float>((uint)p[i] << 16);
}

inline float qwen_gdn_bf16_dot_lane(device const ushort *wr,
                                    device const float *x,
                                    int I,
                                    uint lane) {
    float acc = 0.0f;
    int i = (int)lane;
    for (; i + 96 < I; i += 128) {
        acc += x[i + 0]  * qwen_bf16(wr, i + 0);
        acc += x[i + 32] * qwen_bf16(wr, i + 32);
        acc += x[i + 64] * qwen_bf16(wr, i + 64);
        acc += x[i + 96] * qwen_bf16(wr, i + 96);
    }
    for (; i < I; i += 32)
        acc += x[i] * qwen_bf16(wr, i);
    return acc;
}

/* Deterministic packed BF16 row dots. Lane N reads N+32*g so each GPU memory
 * instruction is contiguous across the SIMDgroup. This restores the proven
 * CCBPLAN-B access pattern while retaining the fixed 0..31 final reduction. */
kernel void qwen_gdn_input_bf16(
    device const ushort *wqkv [[buffer(0)]],
    device const ushort *wz   [[buffer(1)]],
    device const ushort *wa   [[buffer(2)]],
    device const ushort *wb   [[buffer(3)]],
    device const float *x     [[buffer(4)]],
    device float *qkv         [[buffer(5)]],
    device float *z           [[buffer(6)]],
    device float *a           [[buffer(7)]],
    device float *b           [[buffer(8)]],
    constant int &D           [[buffer(9)]],
    constant int &C           [[buffer(10)]],
    constant int &vdim        [[buffer(11)]],
    constant int &vheads      [[buffer(12)]],
    uint tg                   [[threadgroup_position_in_grid]],
    uint tid                  [[thread_index_in_threadgroup]])
{
    threadgroup float partial[QWEN_GDN_DOT_THREADS];
    const uint row_slot = tid / QWEN_GDN_DOT_LANES;
    const uint lane = tid - row_slot * QWEN_GDN_DOT_LANES;
    const uint row = tg * QWEN_GDN_ROWS_PER_TG + row_slot;
    const uint total = (uint)(C + vdim + 2 * vheads);

    device const ushort *w = wqkv;
    device float *dst = qkv;
    int o = 0;
    float acc = 0.0f;
    if (row < total) {
        o = (int)row;
        if (o >= C) {
            o -= C;
            w = wz; dst = z;
            if (o >= vdim) {
                o -= vdim;
                w = wa; dst = a;
                if (o >= vheads) {
                    o -= vheads;
                    w = wb; dst = b;
                }
            }
        }
        acc = qwen_gdn_bf16_dot_lane(w + (long)o * D, x, D, lane);
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row < total && lane == 0) {
        float sum = 0.0f;
        const uint pb = row_slot * QWEN_GDN_DOT_LANES;
        for (uint p = 0; p < QWEN_GDN_DOT_LANES; ++p) sum += partial[pb + p];
        dst[o] = sum;
    }
}

inline float qwen_gdn_conv_one(
    device const float *qkv,
    device const float *weights,
    device float *conv_state,
    int ch,
    int kk)
{
    float acc = 0.0f;
    if (kk > 1) {
        const long sb = (long)ch * (kk - 1);
        const long wb = (long)ch * kk;
        for (int j = 0; j < kk; ++j) {
            const float v = (j == kk - 1) ? qkv[ch] : conv_state[sb + j];
            acc += weights[wb + j] * v;
        }
        for (int s = 0; s < kk - 2; ++s)
            conv_state[sb + s] = conv_state[sb + s + 1];
        conv_state[sb + (kk - 2)] = qkv[ch];
    } else {
        acc = weights[ch] * qkv[ch];
    }
    return acc / (1.0f + exp(-acc));
}

/* Fuse causal convolution into the recurrent kernel. One threadgroup owns one
 * key head and all of its replicated value heads, so every q/k/v convolution
 * channel is advanced exactly once without a cross-threadgroup state race.
 * q/k RMS sums, recurrence loops, and per-head RMS combine all retain the old
 * ascending scalar order. */
kernel void qwen_gdn_conv_recur_norm(
    device const float *qkv       [[buffer(0)]],
    device const float *conv_w    [[buffer(1)]],
    device float *conv_state      [[buffer(2)]],
    device const float *a         [[buffer(3)]],
    device const float *b         [[buffer(4)]],
    device const float *z         [[buffer(5)]],
    device const float *A_log     [[buffer(6)]],
    device const float *dt_bias   [[buffer(7)]],
    device const float *norm_w    [[buffer(8)]],
    device float *state           [[buffer(9)]],
    device float *normed          [[buffer(10)]],
    constant int &kheads          [[buffer(11)]],
    constant int &kd              [[buffer(12)]],
    constant int &vheads          [[buffer(13)]],
    constant int &vd              [[buffer(14)]],
    constant int &kk              [[buffer(15)]],
    constant float &eps           [[buffer(16)]],
    threadgroup float *scratch    [[threadgroup(0)]],
    uint kh_u                     [[threadgroup_position_in_grid]],
    uint t                        [[thread_index_in_threadgroup]])
{
    const int kh = (int)kh_u;
    const int rep = vheads / kheads;
    const int threads = rep * vd;
    const int local_head = (int)t / vd;
    const int d = (int)t - local_head * vd;
    const int h = kh * rep + local_head;
    const int kdim = kheads * kd;

    threadgroup float *qv = scratch;
    threadgroup float *kv = qv + kd;
    threadgroup float *head_out = kv + kd;
    threadgroup float *norm_inv = head_out + rep * vd;
    threadgroup float *common = norm_inv + rep;
    threadgroup float *decay = common + 3;
    threadgroup float *beta = decay + rep;

    for (int qi = (int)t; qi < 2 * kd; qi += threads) {
        if (qi < kd) {
            const int ch = kh * kd + qi;
            qv[qi] = qwen_gdn_conv_one(qkv, conv_w, conv_state, ch, kk);
        } else {
            const int i = qi - kd;
            const int ch = kdim + kh * kd + i;
            kv[i] = qwen_gdn_conv_one(qkv, conv_w, conv_state, ch, kk);
        }
    }

    const int vch = 2 * kdim + h * vd + d;
    const float vv = qwen_gdn_conv_one(qkv, conv_w, conv_state, vch, kk);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t == 0) {
        float qs = 0.0f, ks = 0.0f;
        for (int i = 0; i < kd; ++i) {
            const float q = qv[i];
            const float k = kv[i];
            qs += q * q;
            ks += k * k;
        }
        common[0] = 1.0f / sqrt(qs + 1.0e-6f);
        common[1] = 1.0f / sqrt(ks + 1.0e-6f);
        common[2] = 1.0f / sqrt((float)kd);
    }
    if (d == 0) {
        const float ga = -exp(A_log[h]) * log(1.0f + exp(a[h] + dt_bias[h]));
        decay[local_head] = exp(ga);
        beta[local_head] = 1.0f / (1.0f + exp(-b[h]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qinv = common[0];
    const float kinv = common[1];
    const float qscale = common[2];
    const float decay_h = decay[local_head];
    const float beta_h = beta[local_head];
    const long hs = (long)h * kd * vd;
    float kv_mem = 0.0f;
    for (int kk2 = 0; kk2 < kd; ++kk2) {
        const float khh = kv[kk2] * kinv;
        const long si = hs + (long)kk2 * vd + d;
        const float s = state[si] * decay_h;
        state[si] = s;
        kv_mem += s * khh;
    }
    const float delta = (vv - kv_mem) * beta_h;
    for (int kk2 = 0; kk2 < kd; ++kk2) {
        const float khh = kv[kk2] * kinv;
        const long si = hs + (long)kk2 * vd + d;
        state[si] += khh * delta;
    }
    float outv = 0.0f;
    for (int kk2 = 0; kk2 < kd; ++kk2) {
        const float qhh = (qv[kk2] * qinv) * qscale;
        const long si = hs + (long)kk2 * vd + d;
        outv += state[si] * qhh;
    }
    head_out[local_head * vd + d] = outv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d == 0) {
        float ms = 0.0f;
        const int hb = local_head * vd;
        for (int i = 0; i < vd; ++i) {
            const float ov = head_out[hb + i];
            ms += ov * ov;
        }
        norm_inv[local_head] = 1.0f / sqrt(ms / (float)vd + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float zv = z[(long)h * vd + d];
    const float silu_z = zv / (1.0f + exp(-zv));
    normed[(long)h * vd + d] = norm_w[d] * (outv * norm_inv[local_head]) * silu_z;
}

/* Same deterministic coalesced projection as the input side. */
kernel void qwen_gdn_output_bf16(
    device const ushort *w       [[buffer(0)]],
    device const float *x        [[buffer(1)]],
    device float *out            [[buffer(2)]],
    constant int &I              [[buffer(3)]],
    constant int &O              [[buffer(4)]],
    uint tg                      [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]])
{
    threadgroup float partial[QWEN_GDN_DOT_THREADS];
    const uint row_slot = tid / QWEN_GDN_DOT_LANES;
    const uint lane = tid - row_slot * QWEN_GDN_DOT_LANES;
    const uint o = tg * QWEN_GDN_ROWS_PER_TG + row_slot;
    float acc = 0.0f;
    if (o < (uint)O)
        acc = qwen_gdn_bf16_dot_lane(w + (long)o * I, x, I, lane);
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (o < (uint)O && lane == 0) {
        float sum = 0.0f;
        const uint pb = row_slot * QWEN_GDN_DOT_LANES;
        for (uint p = 0; p < QWEN_GDN_DOT_LANES; ++p) sum += partial[pb + p];
        out[o] = sum;
    }
}
)METAL";

static id<MTLComputePipelineState> g_gdn_input_pipeline = nil;
static id<MTLComputePipelineState> g_gdn_recur_pipeline = nil;
static id<MTLComputePipelineState> g_gdn_output_pipeline = nil;

struct QwenGdnMetalLayer {
    id<MTLBuffer> wqkv = nil, wz = nil, wa = nil, wb = nil, wout = nil;
    id<MTLBuffer> A_log = nil, dt_bias = nil, conv_w = nil, norm_w = nil;
    id<MTLBuffer> state = nil, conv_state = nil;
    id<MTLBuffer> xb = nil, outb = nil;
    id<MTLBuffer> qkv = nil, z = nil, a = nil, b = nil, normed = nil;
    int D = 0, kheads = 0, kd = 0, vheads = 0, vd = 0, kk = 0;
};
static std::vector<QwenGdnMetalLayer *> g_gdn_layers;

static size_t qwen_gdn_round_page(size_t bytes) {
    if (!bytes || bytes > SIZE_MAX - 16383u) return 0;
    return (bytes + 16383u) & ~(size_t)16383u;
}

static id<MTLBuffer> qwen_gdn_wrap_nocopy_locked(const void *ptr, size_t bytes) {
    const size_t rounded = qwen_gdn_round_page(bytes);
    if (!g_device || !ptr || !rounded || (((uintptr_t)ptr) & 16383u) != 0) return nil;
    return [g_device newBufferWithBytesNoCopy:(void *)ptr
                                       length:rounded
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

static int qwen_gdn_init_locked(void) {
    if (g_gdn_input_pipeline && g_gdn_recur_pipeline && g_gdn_output_pipeline)
        return 1;
    if (!g_device || !g_queue) return 0;
    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:QWEN_GDN_SHADER];
    id<MTLLibrary> library = [g_device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        fprintf(stderr, "[qwen-gdn-metal] shader compile failed: %s\n",
                error ? error.localizedDescription.UTF8String : "unknown");
        return 0;
    }
    g_gdn_input_pipeline = make_pipeline(library, @"qwen_gdn_input_bf16", &error);
    g_gdn_recur_pipeline = make_pipeline(library, @"qwen_gdn_conv_recur_norm", &error);
    g_gdn_output_pipeline = make_pipeline(library, @"qwen_gdn_output_bf16", &error);
    if (!g_gdn_input_pipeline || !g_gdn_recur_pipeline || !g_gdn_output_pipeline) {
        fprintf(stderr, "[qwen-gdn-metal] pipeline creation failed: %s\n",
                error ? error.localizedDescription.UTF8String : "missing function");
        g_gdn_input_pipeline = nil;
        g_gdn_recur_pipeline = nil;
        g_gdn_output_pipeline = nil;
        return 0;
    }
    return 1;
}

static void qwen_gdn_clear_locked(void) {
    for (QwenGdnMetalLayer *ctx : g_gdn_layers) delete ctx;
    g_gdn_layers.clear();
    g_gdn_input_pipeline = nil;
    g_gdn_recur_pipeline = nil;
    g_gdn_output_pipeline = nil;
}

static int qwen_gdn_mul3_size(size_t a, size_t b, size_t c, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    size_t ab = a * b;
    if (ab && c > SIZE_MAX / ab) return 0;
    *out = ab * c;
    return 1;
}

static QwenGdnMetalLayer *qwen_gdn_layer_locked(
    int layer,
    const uint16_t *wqkv, const uint16_t *wz,
    const uint16_t *wa, const uint16_t *wb, const uint16_t *wout,
    const float *A_log, const float *dt_bias,
    const float *conv_w, const float *norm_w,
    float *state, float *conv_state,
    int D, int kheads, int kd, int vheads, int vd, int kk)
{
    if (layer < 0 || D <= 0 || kheads <= 0 || kd <= 0 || vheads <= 0 || vd <= 0 ||
        kk <= 0 || vheads < kheads || vheads % kheads ||
        !wqkv || !wz || !wa || !wb || !wout || !A_log || !dt_bias ||
        !conv_w || !norm_w || !state || (kk > 1 && !conv_state))
        return nullptr;
    if (!qwen_gdn_init_locked()) return nullptr;

    const int rep = vheads / kheads;
    const size_t recur_threads = (size_t)rep * (size_t)vd;
    const size_t scratch_floats = 2u * (size_t)kd + recur_threads +
                                  3u * (size_t)rep + 3u;
    if (g_gdn_input_pipeline.maxTotalThreadsPerThreadgroup < 256 ||
        g_gdn_output_pipeline.maxTotalThreadsPerThreadgroup < 256 ||
        recur_threads > (size_t)g_gdn_recur_pipeline.maxTotalThreadsPerThreadgroup ||
        scratch_floats > SIZE_MAX / sizeof(float) ||
        scratch_floats * sizeof(float) > (size_t)g_device.maxThreadgroupMemoryLength)
        return nullptr;

    if ((size_t)layer >= g_gdn_layers.size())
        g_gdn_layers.resize((size_t)layer + 1, nullptr);
    if (g_gdn_layers[(size_t)layer]) return g_gdn_layers[(size_t)layer];

    const size_t kdim = (size_t)kheads * (size_t)kd;
    const size_t vdim = (size_t)vheads * (size_t)vd;
    if (kdim > (size_t)INT_MAX || vdim > (size_t)INT_MAX || kdim > (SIZE_MAX - vdim) / 2)
        return nullptr;
    const size_t C = 2 * kdim + vdim;
    if (C > (size_t)INT_MAX) return nullptr;

    size_t wqkv_b = 0, wz_b = 0, wa_b = 0, wb_b = 0, wout_b = 0;
    size_t state_b = 0, conv_state_b = 0, conv_w_b = 0;
    if (!qwen_gdn_mul3_size(C, (size_t)D, sizeof(uint16_t), &wqkv_b) ||
        !qwen_gdn_mul3_size(vdim, (size_t)D, sizeof(uint16_t), &wz_b) ||
        !qwen_gdn_mul3_size((size_t)vheads, (size_t)D, sizeof(uint16_t), &wa_b) ||
        !qwen_gdn_mul3_size((size_t)vheads, (size_t)D, sizeof(uint16_t), &wb_b) ||
        !qwen_gdn_mul3_size((size_t)D, vdim, sizeof(uint16_t), &wout_b) ||
        !qwen_gdn_mul3_size((size_t)vheads * (size_t)kd, (size_t)vd, sizeof(float), &state_b) ||
        !qwen_gdn_mul3_size(C, (size_t)(kk > 1 ? kk - 1 : 1), sizeof(float), &conv_state_b) ||
        !qwen_gdn_mul3_size(C, (size_t)kk, sizeof(float), &conv_w_b))
        return nullptr;

    QwenGdnMetalLayer *ctx = new (std::nothrow) QwenGdnMetalLayer();
    if (!ctx) return nullptr;
    ctx->D = D; ctx->kheads = kheads; ctx->kd = kd;
    ctx->vheads = vheads; ctx->vd = vd; ctx->kk = kk;
    ctx->wqkv = qwen_gdn_wrap_nocopy_locked(wqkv, wqkv_b);
    ctx->wz = qwen_gdn_wrap_nocopy_locked(wz, wz_b);
    ctx->wa = qwen_gdn_wrap_nocopy_locked(wa, wa_b);
    ctx->wb = qwen_gdn_wrap_nocopy_locked(wb, wb_b);
    ctx->wout = qwen_gdn_wrap_nocopy_locked(wout, wout_b);
    ctx->state = qwen_gdn_wrap_nocopy_locked(state, state_b);
    if (kk > 1) ctx->conv_state = qwen_gdn_wrap_nocopy_locked(conv_state, conv_state_b);
    else ctx->conv_state = [g_device newBufferWithLength:sizeof(float)
                                                options:MTLResourceStorageModeShared];
    ctx->A_log = [g_device newBufferWithBytes:A_log
                                        length:(size_t)vheads * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    ctx->dt_bias = [g_device newBufferWithBytes:dt_bias
                                          length:(size_t)vheads * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    ctx->conv_w = [g_device newBufferWithBytes:conv_w length:conv_w_b
                                         options:MTLResourceStorageModeShared];
    ctx->norm_w = [g_device newBufferWithBytes:norm_w
                                         length:(size_t)vd * sizeof(float)
                                        options:MTLResourceStorageModeShared];
    ctx->xb = [g_device newBufferWithLength:(size_t)D * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    ctx->outb = [g_device newBufferWithLength:(size_t)D * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    ctx->qkv = [g_device newBufferWithLength:C * sizeof(float)
                                      options:MTLResourceStorageModePrivate];
    ctx->z = [g_device newBufferWithLength:vdim * sizeof(float)
                                    options:MTLResourceStorageModePrivate];
    ctx->a = [g_device newBufferWithLength:(size_t)vheads * sizeof(float)
                                    options:MTLResourceStorageModePrivate];
    ctx->b = [g_device newBufferWithLength:(size_t)vheads * sizeof(float)
                                    options:MTLResourceStorageModePrivate];
    ctx->normed = [g_device newBufferWithLength:vdim * sizeof(float)
                                         options:MTLResourceStorageModePrivate];
    if (!ctx->wqkv || !ctx->wz || !ctx->wa || !ctx->wb || !ctx->wout ||
        !ctx->A_log || !ctx->dt_bias || !ctx->conv_w || !ctx->norm_w ||
        !ctx->state || !ctx->conv_state || !ctx->xb || !ctx->outb ||
        !ctx->qkv || !ctx->z || !ctx->a || !ctx->b || !ctx->normed) {
        delete ctx;
        return nullptr;
    }
    g_gdn_layers[(size_t)layer] = ctx;
    return ctx;
}

extern "C" int coli_apple8_metalio_gdn_token(
    int layer, const float *x, float *out,
    const uint16_t *wqkv, const uint16_t *wz,
    const uint16_t *wa, const uint16_t *wb, const uint16_t *wout,
    const float *A_log, const float *dt_bias,
    const float *conv_w, const float *norm_w,
    float *state, float *conv_state,
    int D, int kheads, int kd, int vheads, int vd, int kk, float eps)
{
    if (!x || !out || !(eps > 0.0f)) return 0;
    std::lock_guard<std::mutex> guard(g_lock);
    QwenGdnMetalLayer *ctx = qwen_gdn_layer_locked(
        layer, wqkv, wz, wa, wb, wout, A_log, dt_bias, conv_w, norm_w,
        state, conv_state, D, kheads, kd, vheads, vd, kk);
    if (!ctx || !g_queue || !g_device) return 0;

    const int kdim = kheads * kd;
    const int vdim = vheads * vd;
    const int C = 2 * kdim + vdim;
    const int rep = vheads / kheads;
    const NSUInteger recur_threads = (NSUInteger)rep * (NSUInteger)vd;
    const NSUInteger scratch_floats = 2u * (NSUInteger)kd + recur_threads +
                                      3u * (NSUInteger)rep + 3u;
    memcpy(ctx->xb.contents, x, (size_t)D * sizeof(float));

    uint64_t encode_begin = direct_now_ns();
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return 0;

    id<MTLComputeCommandEncoder> inp = [cb computeCommandEncoder];
    if (!inp) return 0;
    [inp setComputePipelineState:g_gdn_input_pipeline];
    [inp setBuffer:ctx->wqkv offset:0 atIndex:0];
    [inp setBuffer:ctx->wz offset:0 atIndex:1];
    [inp setBuffer:ctx->wa offset:0 atIndex:2];
    [inp setBuffer:ctx->wb offset:0 atIndex:3];
    [inp setBuffer:ctx->xb offset:0 atIndex:4];
    [inp setBuffer:ctx->qkv offset:0 atIndex:5];
    [inp setBuffer:ctx->z offset:0 atIndex:6];
    [inp setBuffer:ctx->a offset:0 atIndex:7];
    [inp setBuffer:ctx->b offset:0 atIndex:8];
    [inp setBytes:&D length:sizeof(D) atIndex:9];
    [inp setBytes:&C length:sizeof(C) atIndex:10];
    [inp setBytes:&vdim length:sizeof(vdim) atIndex:11];
    [inp setBytes:&vheads length:sizeof(vheads) atIndex:12];
    const NSUInteger input_rows = (NSUInteger)C + (NSUInteger)vdim +
                                  2u * (NSUInteger)vheads;
    [inp dispatchThreadgroups:MTLSizeMake((input_rows + 7u) / 8u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [inp endEncoding];

    id<MTLComputeCommandEncoder> rec = [cb computeCommandEncoder];
    if (!rec) return 0;
    [rec setComputePipelineState:g_gdn_recur_pipeline];
    [rec setBuffer:ctx->qkv offset:0 atIndex:0];
    [rec setBuffer:ctx->conv_w offset:0 atIndex:1];
    [rec setBuffer:ctx->conv_state offset:0 atIndex:2];
    [rec setBuffer:ctx->a offset:0 atIndex:3];
    [rec setBuffer:ctx->b offset:0 atIndex:4];
    [rec setBuffer:ctx->z offset:0 atIndex:5];
    [rec setBuffer:ctx->A_log offset:0 atIndex:6];
    [rec setBuffer:ctx->dt_bias offset:0 atIndex:7];
    [rec setBuffer:ctx->norm_w offset:0 atIndex:8];
    [rec setBuffer:ctx->state offset:0 atIndex:9];
    [rec setBuffer:ctx->normed offset:0 atIndex:10];
    [rec setBytes:&kheads length:sizeof(kheads) atIndex:11];
    [rec setBytes:&kd length:sizeof(kd) atIndex:12];
    [rec setBytes:&vheads length:sizeof(vheads) atIndex:13];
    [rec setBytes:&vd length:sizeof(vd) atIndex:14];
    [rec setBytes:&kk length:sizeof(kk) atIndex:15];
    [rec setBytes:&eps length:sizeof(eps) atIndex:16];
    [rec setThreadgroupMemoryLength:scratch_floats * sizeof(float) atIndex:0];
    [rec dispatchThreadgroups:MTLSizeMake((NSUInteger)kheads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(recur_threads, 1, 1)];
    [rec endEncoding];

    id<MTLComputeCommandEncoder> op = [cb computeCommandEncoder];
    if (!op) return 0;
    [op setComputePipelineState:g_gdn_output_pipeline];
    [op setBuffer:ctx->wout offset:0 atIndex:0];
    [op setBuffer:ctx->normed offset:0 atIndex:1];
    [op setBuffer:ctx->outb offset:0 atIndex:2];
    [op setBytes:&vdim length:sizeof(vdim) atIndex:3];
    [op setBytes:&D length:sizeof(D) atIndex:4];
    [op dispatchThreadgroups:MTLSizeMake(((NSUInteger)D + 7u) / 8u, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [op endEncoding];

    const uint64_t encode_ns = direct_now_ns() - encode_begin;
    const uint64_t submit_begin = direct_now_ns();
    [cb commit];
    const uint64_t submit_ns = direct_now_ns() - submit_begin;
    const uint64_t wait_begin = direct_now_ns();
    [cb waitUntilCompleted];
    const uint64_t wait_ns = direct_now_ns() - wait_begin;
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[qwen-gdn-metal] command failed after submission: %s\n",
                cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
        return -1;
    }
    profile_completed_locked(cb, encode_ns, submit_ns, wait_ns, 0);
    memcpy(out, ctx->outb.contents, (size_t)D * sizeof(float));
    return 1;
}

extern "C" int coli_apple8_metalio_direct_init(void) {
    std::lock_guard<std::mutex> guard(g_lock);
    if (g_matmul_pipeline && g_gu_pipeline && g_down_pipeline && g_reduce_pipeline &&
        g_bf16_matmul_pipeline && g_gu8_pipeline && g_down8_pipeline && g_queue && g_device)
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
    g_bf16_matmul_pipeline = make_pipeline(library, @"qwen_bf16_matmul", &error);
    if (!g_bf16_matmul_pipeline) goto pipeline_fail;
    g_gu8_pipeline = make_pipeline(library, @"apple8_moe_gu8", &error);
    if (!g_gu8_pipeline) goto pipeline_fail;
    g_down8_pipeline = make_pipeline(library, @"apple8_moe_down8", &error);
    if (!g_down8_pipeline) goto pipeline_fail;
    (void)qwen_gdn_init_locked();
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

extern "C" int coli_apple8_metalio_bf16_matmul(
    const uint16_t *w, const float *x, float *y, int S, int O, int I)
{
    if (!w || !x || !y || S <= 0 || O <= 0 || I <= 0) return 0;
    std::lock_guard<std::mutex> guard(g_lock);
    if (!g_bf16_matmul_pipeline || !g_queue || !g_device) return 0;

    if (w != g_bf16_w || O != g_bf16_O || I != g_bf16_I) {
        g_bf16_wbuf = [g_device newBufferWithBytes:w
                        length:(size_t)O * (size_t)I * sizeof(uint16_t)
                       options:MTLResourceStorageModeShared];
        if (!g_bf16_wbuf) return 0;
        g_bf16_w = w;
        g_bf16_O = O;
        g_bf16_I = I;
    }

    const size_t x_bytes = (size_t)S * (size_t)I * sizeof(float);
    const size_t y_bytes = (size_t)S * (size_t)O * sizeof(float);
    id<MTLBuffer> xb = [g_device newBufferWithBytes:x length:x_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> yb = [g_device newBufferWithLength:y_bytes
                                             options:MTLResourceStorageModeShared];
    if (!xb || !yb) return 0;

    uint64_t encode_begin = direct_now_ns();
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return 0;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return 0;
    [enc setComputePipelineState:g_bf16_matmul_pipeline];
    [enc setBuffer:g_bf16_wbuf offset:0 atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:yb offset:0 atIndex:2];
    [enc setBytes:&S length:sizeof(S) atIndex:3];
    [enc setBytes:&O length:sizeof(O) atIndex:4];
    [enc setBytes:&I length:sizeof(I) atIndex:5];
    const NSUInteger total_rows = (NSUInteger)S * (NSUInteger)O;
    [enc dispatchThreadgroups:MTLSizeMake((total_rows + 7u) / 8u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    uint64_t encode_ns = direct_now_ns() - encode_begin;
    uint64_t submit_begin = direct_now_ns();
    [cb commit];
    uint64_t submit_ns = direct_now_ns() - submit_begin;
    uint64_t wait_begin = direct_now_ns();
    [cb waitUntilCompleted];
    uint64_t wait_ns = direct_now_ns() - wait_begin;
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[apple8-metalio] BF16 matmul failed: %s\n",
                cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
        return -1;
    }
    profile_completed_locked(cb, encode_ns, submit_ns, wait_ns, 0);
    memcpy(y, yb.contents, y_bytes);
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

    int gate_off[8] = {}, up_off[8] = {}, down_off[8] = {};
    for (int i = 0; i < expert_count && i < 8; ++i) {
        gate_off[i] = (int)experts[i].gate_offset;
        up_off[i] = (int)experts[i].up_offset;
        down_off[i] = (int)experts[i].down_offset;
    }

    id<MTLComputeCommandEncoder> gu = [cb computeCommandEncoder];
    if (!gu) { delete pending; return 0; }
    if (expert_count <= 8) {
        /* Merged tile dispatch: all experts in one grid and eight physical
         * output rows per 256-thread group. */
        [gu setComputePipelineState:g_gu8_pipeline];
        for (int i = 0; i < 8; ++i)
            [gu setBuffer:weight_buffers[i < expert_count ? i : 0] offset:0 atIndex:i];
        [gu setBuffer:xb offset:0 atIndex:8];
        [gu setBuffer:mid offset:0 atIndex:9];
        [gu setBytes:&S length:sizeof(S) atIndex:10];
        [gu setBytes:&hidden length:sizeof(hidden) atIndex:11];
        [gu setBytes:&intermediate length:sizeof(intermediate) atIndex:12];
        [gu setBytes:&expert_count length:sizeof(expert_count) atIndex:13];
        [gu setBytes:gate_off length:(NSUInteger)expert_count * sizeof(int) atIndex:14];
        [gu setBytes:up_off length:(NSUInteger)expert_count * sizeof(int) atIndex:15];
        [gu setBytes:down_off length:(NSUInteger)expert_count * sizeof(int) atIndex:16];
        const NSUInteger mtiles = ((NSUInteger)intermediate + 7u) / 8u;
        [gu dispatchThreadgroups:MTLSizeMake((NSUInteger)expert_count * (NSUInteger)S * mtiles, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
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
    }
    [gu endEncoding];

    id<MTLComputeCommandEncoder> down = [cb computeCommandEncoder];
    if (!down) { delete pending; return 0; }
    if (expert_count <= 8) {
        [down setComputePipelineState:g_down8_pipeline];
        for (int i = 0; i < 8; ++i)
            [down setBuffer:weight_buffers[i < expert_count ? i : 0] offset:0 atIndex:i];
        [down setBuffer:mid offset:0 atIndex:8];
        [down setBuffer:expert_y offset:0 atIndex:9];
        [down setBytes:&S length:sizeof(S) atIndex:10];
        [down setBytes:&hidden length:sizeof(hidden) atIndex:11];
        [down setBytes:&intermediate length:sizeof(intermediate) atIndex:12];
        [down setBytes:&expert_count length:sizeof(expert_count) atIndex:13];
        [down setBytes:gate_off length:(NSUInteger)expert_count * sizeof(int) atIndex:14];
        [down setBytes:up_off length:(NSUInteger)expert_count * sizeof(int) atIndex:15];
        [down setBytes:down_off length:(NSUInteger)expert_count * sizeof(int) atIndex:16];
        const NSUInteger htiles = ((NSUInteger)hidden + 7u) / 8u;
        [down dispatchThreadgroups:MTLSizeMake((NSUInteger)expert_count * (NSUInteger)S * htiles, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
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
    }
    [down endEncoding];

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

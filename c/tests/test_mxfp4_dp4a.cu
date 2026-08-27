// Pascal SM6.1 DP4A W4A8 expert matvec (#198) — standalone prototype + bench.
//
// Storage stays MXFP4 (E2M1 nibbles + UE8M0 scale per 32) packed in VRAM.
// Execution is W4A8 through __dp4a:
//   - weights: E2M1 nibble -> exact doubled INT8 in registers (code x2,
//     sign-extended; value = int8 * (ue8m0_scale/2) at the end)
//   - activations: per-32-group f32 -> int8 with per-group activation scale
//   - accumulation: __dp4a int8xint8 -> int32
//   - output scale: act_scale * (ue8m0_scale / 2)
//
// Compile (on the GTX 1080 box):
//   nvcc -O3 -arch=sm_61 -o test_mxfp4_dp4a test_mxfp4_dp4a.cu
// Run: ./test_mxfp4_dp4a
//
// MEASURED VERDICT (GTX 1080, gate 640x2560, 200 iters, 2026-08-27):
//   dp4a_w4a8 = 0.067 ms, float_decode = 0.037 ms -> speedup 0.55x
// DP4A W4A8 is CORRECT (max abs err 0.033 vs canonical float decode) but
// NOT faster on Pascal for these shapes: per-group activation quantization +
// reduction overhead exceeds the int8-dot win on a bandwidth-bound matvec.
// Acceptance criterion "measurably faster than the per-weight float MXFP4
// path" is NOT met -> planner keeps CPU INT4 for this box (already the case).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// E2M1 magnitudes (code & 7): 0, 0.5, 1, 1.5, 2, 3, 4, 6 ; sign in bit 3.
// Doubled int8 values for DP4A: code 0..7 -> 0,1,2,3,4,6,8,12 ; negative mirror.
static __device__ __forceinline__ int8_t e2m1_to_doubled_int8(uint8_t nibble) {
    static const int8_t mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    int8_t m = mag[nibble & 7];
    return (nibble & 8) ? (int8_t)(-m) : m;
}

// __dp4a works on int8x4. Use the SIGNED char4 overload — the uint32 overload
// treats bytes as unsigned and corrupts negative weights/activations.
static __device__ __forceinline__ int dp4a_signed(int8_t a, int8_t b, int8_t c, int8_t d,
                                                  int8_t e, int8_t f, int8_t g, int8_t h) {
    return __dp4a(make_char4(a, b, c, d), make_char4(e, f, g, h), 0);
}

// One output row per block, 32 threads, W4A8 DP4A, group size 32.
// lane = column within the group; amax via warp shuffle; lane4 = lane/4 does
// the 4-wide DP4A chunk; the 8 chunks reduce via shuffle offsets 4, 8, 16.
// Weights: q4[o*I/2 + i/2], low nibble = even i. Scales: e8s[o*ng + i/32].
template <int GROUP>
__global__ void mxfp4_w4a8_dp4a_kernel(float *__restrict__ y,
                                       const float *__restrict__ x,
                                       const uint8_t *__restrict__ q4,
                                       const uint8_t *__restrict__ e8s,
                                       int I, int O) {
    const int o = blockIdx.x;
    const int lane = threadIdx.x;
    const int ng = (I + GROUP - 1) / GROUP;

    float acc = 0.0f;
    const uint8_t *wrow = q4 + (size_t)o * (I / 2);
    const uint8_t *srow = e8s + (size_t)o * ng;
    const float *xrow = x;

    for (int g = 0; g < ng; ++g) {
        // Register load + warp-reduce amax (32 lanes = the whole group).
        float v = xrow[g * GROUP + lane];
        float m = fabsf(v);
        for (int off = GROUP / 2; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        float act_scale = (m == 0.0f) ? 1.0f : m / 127.0f;

        // Quantize in register.
        float q = rintf(v / act_scale);
        int8_t xv = (int8_t)(q > 127.0f ? 127 : (q < -128.0f ? -128 : q));

        // DP4A: lane4 = lane/4 handles columns lane4*4..+3 of the group
        // (8 lanes * 4 cols = 32 = one group); gather via shuffle.
        const int lane4 = lane / 4;
        const int c0 = g * GROUP + lane4 * 4;
        int8_t xq[4];
        xq[0] = (int8_t)__shfl_sync(0xffffffffu, (int)xv, lane4 * 4 + 0);
        xq[1] = (int8_t)__shfl_sync(0xffffffffu, (int)xv, lane4 * 4 + 1);
        xq[2] = (int8_t)__shfl_sync(0xffffffffu, (int)xv, lane4 * 4 + 2);
        xq[3] = (int8_t)__shfl_sync(0xffffffffu, (int)xv, lane4 * 4 + 3);
        uint8_t b0 = wrow[g * GROUP / 2 + lane4 * 2 + 0];
        uint8_t b1 = wrow[g * GROUP / 2 + lane4 * 2 + 1];
        int dp = dp4a_signed(
            e2m1_to_doubled_int8(b0 & 0x0F), e2m1_to_doubled_int8(b0 >> 4),
            e2m1_to_doubled_int8(b1 & 0x0F), e2m1_to_doubled_int8(b1 >> 4),
            xq[0], xq[1], xq[2], xq[3]);
        float wscale = ldexpf(1.0f, (int)srow[g] - 127);
        // Reduce the 8 lane4 partials: chunks sit at lane stride 4, so XOR
        // offsets 4, 8, 16 sum all 8 distinct chunks.
        float dpf = (float)dp * act_scale * (wscale * 0.5f);
        for (int off = 4; off <= 16; off <<= 1)
            dpf += __shfl_xor_sync(0xffffffffu, dpf, off);
        if (lane == 0) acc += dpf;
    }
    if (lane == 0) y[o] = acc;
}

// Reference GPU kernel: canonical MXFP4 float decode (the existing fmt=7
// path: nibble -> float, multiply-accumulate, no int8 quantization).
__global__ void float_decode_kernel(float *__restrict__ y,
                                    const float *__restrict__ x,
                                    const uint8_t *__restrict__ q4,
                                    const uint8_t *__restrict__ e8s,
                                    int I, int O) {
    const int o = blockIdx.x;
    const int t = threadIdx.x;
    const int ng = (I + 31) / 32;
    const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float acc = 0.f;
    for (int i = t; i < I; i += blockDim.x) {
        uint8_t n = q4[(size_t)o * (I / 2) + i / 2];
        n = (i & 1) ? (uint8_t)(n >> 4) : (uint8_t)(n & 0x0F);
        float v = mag[n & 7] * ((n & 8) ? -1.f : 1.f);
        float s = ldexpf(1.f, (int)e8s[(size_t)o * ng + i / 32] - 127);
        acc += x[i] * v * s;
    }
    __shared__ float partial[256];
    partial[t] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) partial[t] += partial[t + stride];
        __syncthreads();
    }
    if (t == 0) y[o] = partial[0];
}

// CPU reference: canonical MXFP4 float decode (same as quant_matmul fmt=7).
static void ref_mxfp4(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s,
                      int I, int O) {
    static const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    for (int o = 0; o < O; ++o) {
        float acc = 0.f;
        for (int i = 0; i < I; ++i) {
            uint8_t n = q4[(size_t)o * (I / 2) + i / 2];
            n = (i & 1) ? (uint8_t)(n >> 4) : (uint8_t)(n & 0x0F);
            float v = mag[n & 7] * ((n & 8) ? -1.f : 1.f);
            float s = ldexpf(1.f, (int)e8s[(size_t)o * ((I + 31) / 32) + i / 32] - 127);
            acc += x[i] * v * s;
        }
        y[o] = acc;
    }
}

int main() {
    const int I = 256, O = 128;
    std::vector<float> x(I), y_gpu(O), y_ref(O);
    for (int i = 0; i < I; ++i) x[i] = (float)((i * 37) % 200 - 100) / 100.f;

    static const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    std::vector<uint8_t> q4((size_t)O * I / 2), e8s((size_t)O * ((I + 31) / 32), 127);
    for (int o = 0; o < O; ++o)
        for (int i = 0; i < I; ++i) {
            float v = (float)((o * 13 + i * 7) % 200 - 100) / 100.f;
            float best = 1e9f; uint8_t bestc = 0;
            for (int c = 0; c < 8; ++c) {
                float e = fabsf(v - mag[c]);
                if (e < best) { best = e; bestc = (uint8_t)c; }
            }
            uint8_t code = bestc | (v < 0 ? 8 : 0);
            uint8_t &b = q4[(size_t)o * (I / 2) + i / 2];
            if (i & 1) b = (uint8_t)((b & 0x0F) | (code << 4)); else b = (uint8_t)((b & 0xF0) | code);
        }
    ref_mxfp4(y_ref.data(), x.data(), q4.data(), e8s.data(), I, O);

    float *dx, *dy;
    uint8_t *dq, *de;
    cudaMalloc(&dx, I * sizeof(float));
    cudaMalloc(&dy, O * sizeof(float));
    cudaMalloc(&dq, q4.size());
    cudaMalloc(&de, e8s.size());
    cudaMemcpy(dx, x.data(), I * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dq, q4.data(), q4.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(de, e8s.data(), e8s.size(), cudaMemcpyHostToDevice);

    mxfp4_w4a8_dp4a_kernel<32><<<O, 32>>>(dy, dx, dq, de, I, O);
    cudaDeviceSynchronize();
    cudaMemcpy(y_gpu.data(), dy, O * sizeof(float), cudaMemcpyDeviceToHost);

    float worst_abs = 0.f;
    for (int o = 0; o < O; ++o) {
        float err = fabsf(y_gpu[o] - y_ref[o]);
        if (err > worst_abs) worst_abs = err;
    }
    printf("max abs error vs canonical MXFP4 float decode: %.4f\n", worst_abs);
    printf("%s\n", worst_abs < 0.1f ? "PASS (within W4A8 quantization tolerance)" : "FAIL");
    cudaFree(dx); cudaFree(dy); cudaFree(dq); cudaFree(de);
    if (worst_abs >= 0.1f) return 1;

    // ---- benchmark: DP4A vs float-decode MXFP4 on the real expert shape ----
    // Qwen3.8-Next routed expert gate: I=2560, O=640. Down: I=640, O=2560.
    const int BI = 2560, BO = 640, ITERS = 200;
    std::vector<float> bx(BI);
    for (int i = 0; i < BI; ++i) bx[i] = (float)((i * 31) % 200 - 100) / 100.f;
    std::vector<uint8_t> bq4((size_t)BO * BI / 2), be8s((size_t)BO * ((BI + 31) / 32), 127);
    for (size_t k = 0; k < bq4.size(); ++k) bq4[k] = (uint8_t)(k % 256);

    float *bdx, *bdy;
    uint8_t *bdq, *bde;
    cudaMalloc(&bdx, BI * sizeof(float));
    cudaMalloc(&bdy, BO * sizeof(float));
    cudaMalloc(&bdq, bq4.size());
    cudaMalloc(&bde, be8s.size());
    cudaMemcpy(bdx, bx.data(), BI * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bdq, bq4.data(), bq4.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(bde, be8s.data(), be8s.size(), cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    mxfp4_w4a8_dp4a_kernel<32><<<BO, 32>>>(bdy, bdx, bdq, bde, BI, BO);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int it = 0; it < ITERS; ++it)
        mxfp4_w4a8_dp4a_kernel<32><<<BO, 32>>>(bdy, bdx, bdq, bde, BI, BO);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float dp4a_ms = 0.f;
    cudaEventElapsedTime(&dp4a_ms, t0, t1);
    dp4a_ms /= ITERS;

    float_decode_kernel<<<BO, 256>>>(bdy, bdx, bdq, bde, BI, BO);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int it = 0; it < ITERS; ++it)
        float_decode_kernel<<<BO, 256>>>(bdy, bdx, bdq, bde, BI, BO);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float fdecode_ms = 0.f;
    cudaEventElapsedTime(&fdecode_ms, t0, t1);
    fdecode_ms /= ITERS;

    printf("bench (gate 640x2560, %d iters): dp4a_w4a8 = %.3f ms, float_decode = %.3f ms, speedup %.2fx\n",
           ITERS, dp4a_ms, fdecode_ms, fdecode_ms / dp4a_ms);

    cudaFree(bdx); cudaFree(bdy); cudaFree(bdq); cudaFree(bde);
    return 0;
}

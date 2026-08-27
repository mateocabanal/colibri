// Pascal SM6.1 DP4A W4A8 expert matvec (#198) — standalone prototype.
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

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// E2M1 magnitudes (code & 7): 0, 0.5, 1, 1.5, 2, 3, 4, 6 ; sign in bit 3.
// Doubled int8 values for DP4A: code 0..7 -> 0,1,2,3,4,6,8,12 ; negative mirror.
static __device__ __forceinline__ int8_t e2m1_to_doubled_int8(uint8_t nibble) {
    static const int8_t mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    int8_t m = mag[nibble & 7];
    return (nibble & 8) ? (int8_t)(-m) : m;
}

// __dp4a works on int8x4 in a uint32. Pack 4 doubled weights (a in low byte).
static __device__ __forceinline__ uint32_t pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
           ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

// One output row: y[o] = sum_i x[i] * w[o][i], W4A8 DP4A, group size 32.
// Weights: q4[o*I/2 + i/2], low nibble = even i. Scales: e8s[o*ng + i/32].
// Threads: 32 per block; per group, lanes 0..7 each DP4A a 4-wide chunk
// (8 lanes * 4 cols = 32 = GROUP), partials reduced by lane 0 in shared.
template <int GROUP>
__global__ void mxfp4_w4a8_dp4a_kernel(float *__restrict__ y,
                                       const float *__restrict__ x,
                                       const uint8_t *__restrict__ q4,
                                       const uint8_t *__restrict__ e8s,
                                       int I, int O) {
    const int o = blockIdx.x;
    const int lane = threadIdx.x;
    const int ng = (I + GROUP - 1) / GROUP;
    constexpr int LANES_PER_GROUP = GROUP / 4;  // 8

    __shared__ float xg[GROUP];
    __shared__ float act_scale_s;
    __shared__ float s_partial[LANES_PER_GROUP];

    float acc = 0.0f;
    const uint8_t *wrow = q4 + (size_t)o * (I / 2);
    const uint8_t *srow = e8s + (size_t)o * ng;
    const float *xrow = x;

    for (int g = 0; g < ng; ++g) {
        // Load this group's 32 activations (one per lane), then group-quantize.
        xg[lane] = xrow[g * GROUP + lane];
        __syncthreads();
        float amax = 0.0f;
        for (int k = 0; k < GROUP; ++k) amax = fmaxf(amax, fabsf(xg[k]));
        float act_scale = (amax == 0.0f) ? 1.0f : amax / 127.0f;
        act_scale_s = act_scale;
        __syncthreads();

        // Each of the 8 active lanes quantizes + dots its 4 columns.
        if (lane < LANES_PER_GROUP) {
            int8_t xq[4];
            for (int c = 0; c < 4; ++c) {
                float q = rintf(xg[lane * 4 + c] / act_scale_s);
                xq[c] = (int8_t)(q > 127.0f ? 127 : (q < -128.0f ? -128 : q));
            }
            // Nibble pair: byte g*GROUP/2 + lane*2 covers columns lane*4..+3.
            uint8_t b0 = wrow[g * GROUP / 2 + lane * 2 + 0];
            uint8_t b1 = wrow[g * GROUP / 2 + lane * 2 + 1];
            uint32_t wv = pack4(
                e2m1_to_doubled_int8(b0 & 0x0F),
                e2m1_to_doubled_int8(b0 >> 4),
                e2m1_to_doubled_int8(b1 & 0x0F),
                e2m1_to_doubled_int8(b1 >> 4));
            uint32_t xv = pack4(xq[0], xq[1], xq[2], xq[3]);
            s_partial[lane] = (float)__dp4a(wv, xv, 0);
        }
        __syncthreads();

        if (lane == 0) {
            float sum = 0.0f;
            for (int k = 0; k < LANES_PER_GROUP; ++k) sum += s_partial[k];
            uint8_t es = srow[g];
            float wscale = ldexpf(1.0f, (int)es - 127);
            acc += sum * act_scale_s * (wscale * 0.5f);
        }
        __syncthreads();  // protect xg reuse on the next group
    }
    if (lane == 0) y[o] = acc;
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

    // Weights: encode value v as e2m1 with scale 1 (exponent 127).
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

    float *dx, *dy, *dq, *de;
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

    float worst = 0.f;
    for (int o = 0; o < O; ++o) {
        float err = fabsf(y_gpu[o] - y_ref[o]);
        float rel = err / (fabsf(y_ref[o]) + 1e-6f);
        if (rel > worst) worst = rel;
        if (o < 8) printf("o=%3d gpu=%9.4f ref=%9.4f rel=%8.4f\n", o, y_gpu[o], y_ref[o], rel);
    }
    printf("worst relative error vs canonical MXFP4 float decode: %.4f\n", worst);
    printf("%s\n", worst < 0.05f ? "PASS (within W4A8 quantization tolerance)" : "FAIL");

    cudaFree(dx); cudaFree(dy); cudaFree(dq); cudaFree(de);
    return worst < 0.05f ? 0 : 1;
}

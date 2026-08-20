#ifndef COLIBRI_TEST_APPLE4_BENCH_SHADER_H
#define COLIBRI_TEST_APPLE4_BENCH_SHADER_H

static const char *APPLE4_SHADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float MX4_LUT[16] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
 -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

inline float s4(uchar n) {
  int v = int(n & 15u);
  return float(v >= 8 ? v - 16 : v);
}

kernel void bench_mx_row(device const uchar *w [[buffer(0)]],
                         device const uchar *sc [[buffer(1)]],
                         device const float *x [[buffer(2)]],
                         device float *y [[buffer(3)]],
                         constant int& S [[buffer(4)]],
                         constant int& I [[buffer(5)]],
                         constant int& O [[buffer(6)]],
                         constant int& NT [[buffer(7)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row = (long)tg * 4 + sgid;
  if (row >= NT) return;
  int o = int(row % O), si = int(row / O);
  int rb = (I + 1) / 2, ng = (I + 31) / 32;
  device const uchar *wr = w + (long)o * rb;
  device const uchar *sr = sc + (long)o * ng;
  device const float *xr = x + (long)si * I;
  float acc = 0.0f;
  for (int pair = int(lane); pair < (I + 1) / 2; pair += 32) {
    int k = pair * 2;
    uchar b = wr[pair];
    float scale = as_type<float>((uint)sr[k / 32] << 23);
    acc += MX4_LUT[b & 15u] * xr[k] * scale;
    if (k + 1 < I) acc += MX4_LUT[b >> 4] * xr[k + 1] * scale;
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}

/* Apple8 benchmark layout: 8 output rows x 32 K per tile.
 * 128B E2M1 payload + 8 colocated E8M0 scales = 136B/tile. */
kernel void bench_mx_tile(device const uchar *rec [[buffer(0)]],
                          device const float *x [[buffer(1)]],
                          device float *y [[buffer(2)]],
                          constant int& S [[buffer(3)]],
                          constant int& I [[buffer(4)]],
                          constant int& O [[buffer(5)]],
                          constant int& NT [[buffer(6)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row = (long)tg * 4 + sgid;
  if (row >= NT) return;
  int o = int(row % O), si = int(row / O), ng = (I + 31) / 32;
  int orow = o & 7, otile = o >> 3;
  device const float *xr = x + (long)si * I;
  float acc = 0.0f;
  for (int pair = int(lane); pair < (I + 1) / 2; pair += 32) {
    int k = pair * 2, kg = k / 32, kk = k & 31;
    device const uchar *tile = rec + ((long)otile * ng + kg) * 136;
    uchar b = tile[orow * 16 + (kk >> 1)];
    float scale = as_type<float>((uint)tile[128 + orow] << 23);
    acc += MX4_LUT[b & 15u] * xr[k] * scale;
    if (k + 1 < I) acc += MX4_LUT[b >> 4] * xr[k + 1] * scale;
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}

/* A4S32: 128B payload + 8xFP16 scales = 144B per 8x32 tile. */
kernel void bench_a4s32(device const uchar *rec [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant int& S [[buffer(3)]],
                        constant int& I [[buffer(4)]],
                        constant int& O [[buffer(5)]],
                        constant int& NT [[buffer(6)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row = (long)tg * 4 + sgid;
  if (row >= NT) return;
  int o = int(row % O), si = int(row / O), ng = (I + 31) / 32;
  int orow = o & 7, otile = o >> 3;
  device const float *xr = x + (long)si * I;
  float acc = 0.0f;
  for (int pair = int(lane); pair < (I + 1) / 2; pair += 32) {
    int k = pair * 2, kg = k / 32, kk = k & 31;
    device const uchar *tile = rec + ((long)otile * ng + kg) * 144;
    uchar b = tile[orow * 16 + (kk >> 1)];
    device const half *hs = (device const half *)(tile + 128 + orow * 2);
    float scale = float(*hs);
    acc += s4(b & 15u) * xr[k] * scale;
    if (k + 1 < I) acc += s4(b >> 4) * xr[k + 1] * scale;
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}

/* A4S16: 128B payload + 8 rows x 2 FP16 scales = 160B per 8x32 tile. */
kernel void bench_a4s16(device const uchar *rec [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant int& S [[buffer(3)]],
                        constant int& I [[buffer(4)]],
                        constant int& O [[buffer(5)]],
                        constant int& NT [[buffer(6)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row = (long)tg * 4 + sgid;
  if (row >= NT) return;
  int o = int(row % O), si = int(row / O), ng = (I + 31) / 32;
  int orow = o & 7, otile = o >> 3;
  device const float *xr = x + (long)si * I;
  float acc = 0.0f;
  for (int pair = int(lane); pair < (I + 1) / 2; pair += 32) {
    int k = pair * 2, kg = k / 32, kk = k & 31;
    device const uchar *tile = rec + ((long)otile * ng + kg) * 160;
    uchar b = tile[orow * 16 + (kk >> 1)];
    int sub = kk >> 4;
    device const half *hs = (device const half *)(tile + 128 + orow * 4 + sub * 2);
    float scale = float(*hs);
    acc += s4(b & 15u) * xr[k] * scale;
    if (k + 1 < I) acc += s4(b >> 4) * xr[k + 1] * scale;
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}

/* A4S64: 256B payload + 8xFP16 scales = 272B per 8x64 tile. */
kernel void bench_a4s64(device const uchar *rec [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant int& S [[buffer(3)]],
                        constant int& I [[buffer(4)]],
                        constant int& O [[buffer(5)]],
                        constant int& NT [[buffer(6)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row = (long)tg * 4 + sgid;
  if (row >= NT) return;
  int o = int(row % O), si = int(row / O), ng = (I + 63) / 64;
  int orow = o & 7, otile = o >> 3;
  device const float *xr = x + (long)si * I;
  float acc = 0.0f;
  for (int pair = int(lane); pair < (I + 1) / 2; pair += 32) {
    int k = pair * 2, kg = k / 64, kk = k & 63;
    device const uchar *tile = rec + ((long)otile * ng + kg) * 272;
    uchar b = tile[orow * 32 + (kk >> 1)];
    device const half *hs = (device const half *)(tile + 256 + orow * 2);
    float scale = float(*hs);
    acc += s4(b & 15u) * xr[k] * scale;
    if (k + 1 < I) acc += s4(b >> 4) * xr[k + 1] * scale;
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc;
}
)METAL";

#endif

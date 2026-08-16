/* Ponytail self-check: V4 route SIMD-dot path vs scalar reference.
 * Builds on top of the amalgamated ROUTE_BF16 unit, invokes coli_v4_route_bf16
 * with V4_SIMD_ROUTE on/off, and asserts the top-k set matches (exact-match;
 * an fp-reorder in the dot must not change which experts are routed on this
 * fixed seed). Usage: cc -O2 -I. tests/test_v4_route_selftest.c COLI_V4_UNIT_ROUTE_BF16.o -lm
 * (unit object built with -DCOLI_V4_UNIT_ROUTE_BF16). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* from deepseek_v4_internal.h */
int coli_v4_route_bf16(float *weights, int *indices, const float *hidden,
                       const uint16_t *gate, const float *bias,
                       const int *forced_indices, int experts, int dimension,
                       int topk, float route_scale);

static uint16_t f32_to_bf16(float x) {
    uint32_t bits; memcpy(&bits, &x, 4);
    return (uint16_t)(bits >> 16);
}

int main(void) {
    const int experts = 8, dim = 64, topk = 3;
    float hidden[64], weights_a[3], weights_b[3];
    uint16_t gate[8 * 64];
    int idx_a[3], idx_b[3];
    srand(42);
    for (int i = 0; i < 64; i++) hidden[i] = (float)(rand() % 2000 - 1000) / 100.0f;
    for (int i = 0; i < 8 * 64; i++) gate[i] = f32_to_bf16((float)(rand() % 2000 - 1000) / 100.0f);
    memset(weights_a, 0, sizeof weights_a); memset(weights_b, 0, sizeof weights_b);
    memset(idx_a, 0, sizeof idx_a); memset(idx_b, 0, sizeof idx_b);

    setenv("V4_SIMD_ROUTE", "0", 1);
    if (coli_v4_route_bf16(weights_a, idx_a, hidden, gate, NULL, NULL,
                           experts, dim, topk, 1.0f) != 0) return 1;
    setenv("V4_SIMD_ROUTE", "1", 1);
    if (coli_v4_route_bf16(weights_b, idx_b, hidden, gate, NULL, NULL,
                            experts, dim, topk, 1.0f) != 0) return 1;

    int ok = !memcmp(idx_a, idx_b, sizeof idx_a);
    float tol = 1e-4f;
    for (int i = 0; i < topk; i++)
        if (weights_a[i] < weights_b[i] - tol || weights_a[i] > weights_b[i] + tol) ok = 0;
    printf(ok ? "V4 route SIMD test: ok\n" : "V4 route SIMD test: FAIL\n");
    return ok ? 0 : 2;
}
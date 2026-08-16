#ifndef COLIBRI_NATIVE_QUANT_H
#define COLIBRI_NATIVE_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

float coli_e8m0_decode(uint8_t value);
float coli_e2m1_decode(uint8_t nibble);
float coli_e4m3fn_decode(uint8_t value);
uint8_t coli_e4m3fn_encode(float value);
float coli_bf16_round(float value);
float coli_bf16_decode(uint16_t value);
void coli_bf16_round_array(float *values, size_t count);

/* Simulates the official dynamic E4M3 activation quantization with one E8M0
 * power-of-two scale per block. Output contains the dequantized FP32 values. */
int coli_fp8_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size);
int coli_fp4_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size);
int coli_hadamard_bf16_ref(float *values, size_t length);

/* Correctness-first FP4 matvec. The input is dynamically quantized to E4M3 in
 * blocks of 128, weights are native E2M1 with one E8M0 scale per 32 K, and
 * accumulation is FP32. */
int coli_fp4_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input);

/* Correctness-first FP8 matvec for native 128x128 E4M3 weight blocks with
 * UE8M0 scales and dynamically quantized E4M3 activations. */
int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input);

#ifdef COLI_METAL
#include <stdlib.h>
#include "backend_metal.h"
/* V4 routed-expert matvec on the Apple GPU (fmt=7 MXFP4), env opt-in
 * (V4_METAL_EXPERTS=1). The Metal kernel is numerically equivalent to
 * matmul_mxfp4's scalar path (same e2m1 LUT + e8m0 byte scales, same lane
 * accumulation order per group); the test suite proves parity. Returns 1 on
 * success, 0 to fall back to the CPU kernel. Handles are cached keyed by
 * weight pointer (experts are few and reused); a hit skips re-wrapping. */
static inline int coli_v4_metal_mxfp4_matvec(
    float *output, const float *input, const void *weights, const void *scales,
    int rows, int columns) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *s = getenv("V4_METAL_EXPERTS");
        enabled = s && *s && atoi(s) != 0;
    }
    if (!enabled) return 0;
    static ColiMetalTensor *handles[16];
    static const void *keys[16];
    static int n = 0;
    ColiMetalTensor **h = NULL;
    for (int i = 0; i < n; i++)
        if (keys[i] == weights) { h = &handles[i]; break; }
    if (!h) {
        if (n < 16) { keys[n] = weights; h = &handles[n]; n++; }
        else { h = &handles[0]; keys[0] = weights; }  /* ponytail: 16-slot cache, oldest slot reused */
    }
    return coli_metal_matmul(h, output, input, weights, scales, 7, 1,
                             columns, rows, 0);
}
#endif /* COLI_METAL */

#ifdef __cplusplus
}
#endif

#endif

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
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "backend_metal.h"
#ifdef COLI_METAL_DEBUG
/* Debug-only reference for V4_METAL_DEBUG: quant.h's matmul_mxfp4 is static,
 * so mirror the scalar path here (same e2m1 LUT + e8m0 byte scales as the
 * engine's CPU path; used ONLY to verify the GPU result, never shipped). */
static void metal_debug_ref_mxfp4(float *y, const float *x, const uint8_t *q4,
                                  const uint8_t *e8s, int S, int I, int O) {
  static const float lut[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
  for (int o = 0; o < O; o++) {
    const uint8_t *w = q4 + (size_t)o * ((size_t)(I + 1) / 2);
    for (int s = 0; s < S; s++) {
      const float *xs = x + (size_t)s * I; float a = 0;
      for (int i = 0; i < I; i++) {
        uint8_t b = w[i >> 1]; int nib = (i & 1) ? (int)(b >> 4) : (int)(b & 0xF);
        unsigned sbits = (unsigned)e8s[(size_t)o * ((size_t)(I + 31) / 32) + (unsigned)(i / 32)] << 23;
        float sc; memcpy(&sc, &sbits, 4);
        a += xs[i] * lut[nib] * sc;
      }
      y[(size_t)s * O + o] = a;
    }
  }
}
#endif
/* V4 routed-expert matvec on the Apple GPU (fmt=7 MXFP4). DEFAULT ON when
 * built with COLI_METAL: the Metal kernel is numerically equivalent to
 * matmul_mxfp4's scalar path (same e2m1 LUT + e8m0 byte scales, same lane
 * accumulation order per group); the test suite proves parity. V4_METAL_EXPERTS
 * still controls it: unset = ON, 0 = off, nonzero = ON. Returns 1 on success,
 * 0 to fall back to the CPU kernel. Handles are cached keyed by weight
 * pointer (experts are few and reused); a hit skips re-wrapping.
 * Initialization is lazy on first use: the V4 engine never calls
 * coli_metal_init() itself (colibri.c/inkling/qwen do), so without this the
 * GPU path is dead even when enabled. coli_metal_init is idempotent. */
static inline int coli_v4_metal_mxfp4_matvec(
    float *output, const float *input, const void *weights, const void *scales,
    int rows, int columns) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *s = getenv("V4_METAL_EXPERTS");
        enabled = (!s || !*s) ? 1 : (atoi(s) != 0);   /* default ON */
    }
    if (!enabled) return 0;
    if (!coli_metal_init()) return 0;                /* lazy, idempotent */
    /* Coarse lock around the whole cache lifecycle + matmul: expert matvecs
     * are serial on the Metal queue anyway, so contention is negligible, and
     * it makes the arrays race-free for any future concurrent lanes. */
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&lock);
    static ColiMetalTensor *handles[16];
    static const void *keys[16];
    static int key_rows[16], key_cols[16], key_fmt[16], key_reg[16];
    static int n = 0;
    int slot = -1;
    for (int i = 0; i < n; i++) {
        if (keys[i] == weights && key_rows[i] == rows &&
            key_cols[i] == columns && key_fmt[i] == 7) {
            /* A previously-REGISTERED key whose slab was unregistered (engine
             * teardown) must not serve the stale wrapper: treat as a miss so
             * the handle is rebuilt against (possibly re-registered) memory.
             * Unregistered keys keep the wrap() contract (stable at address). */
            if (key_reg[i] && !coli_metal_ptr_registered(weights)) {
                if (handles[i]) coli_metal_tensor_free(handles[i]);
                handles[i] = NULL; key_reg[i] = 0;
                break;
            }
            slot = i; break;
        }
    }
    if (slot < 0) {
        /* find any dead slot first (freed on teardown) */
        for (int i = 0; i < n; i++)
            if (key_reg[i] && !coli_metal_ptr_registered(keys[i])) {
                if (handles[i]) coli_metal_tensor_free(handles[i]);
                handles[i] = NULL; key_reg[i] = 0; slot = i; break;
            }
        if (slot < 0 && n < 16) slot = n++;
        if (slot < 0) {        /* evict oldest (slot 0) — free the old tensor */
            slot = 0;
            if (handles[0]) coli_metal_tensor_free(handles[0]);
            handles[0] = NULL;
        }
        keys[slot] = weights;
        key_rows[slot] = rows; key_cols[slot] = columns; key_fmt[slot] = 7;
        key_reg[slot] = coli_metal_ptr_registered(weights);
    }
    int rc = coli_metal_matmul(&handles[slot], output, input, weights, scales, 7, 1,
                               columns, rows, 0);
    pthread_mutex_unlock(&lock);
#ifdef COLI_METAL_DEBUG
    /* V4_METAL_DEBUG=1: run the CPU reference on the same inputs and compare
     * the first row of output. Prints a relative error; the GPU result is
     * kept (matches prod behavior). Do not enable by default. */
    {
        static int done = 0;
        if (!done && getenv("V4_METAL_DEBUG") && rc) {
            done = 1;
            float *ref = calloc((size_t)rows, sizeof(float));
            if (ref) {
                metal_debug_ref_mxfp4(ref, input, weights, scales, 1, columns, rows);
                double worst = 0, mag = 0;
                for (int i = 0; i < rows; i++) {
                    double d = fabs((double)output[i] - (double)ref[i]);
                    if (fabs((double)ref[i]) > mag) mag = fabs((double)ref[i]);
                    if (d > worst) worst = d;
                }
                fprintf(stderr, "[metal-debug] rows=%d cols=%d rc=%d worst=%.3g mag=%.3g out0=%.4g ref0=%.4g\n",
                        rows, columns, rc, worst, mag, output[0], ref[0]);
                free(ref);
            }
        }
    }
#endif
    return rc;
}
#endif /* COLI_METAL */

#ifdef __cplusplus
}
#endif

#endif

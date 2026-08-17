#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../mxfp4_runtime.h"

static int closef(float a, float b) {
    float den = 1.0f + fabsf(b);
    return fabsf(a - b) / den < 1e-6f;
}

int main(void) {
    /* 2x2 identity in MXFP4: code 2 == +1.0, code 0 == 0, E8M0 127 == 1. */
    const uint8_t identity[] = {0x02, 0x20};
    const uint8_t scales[] = {127, 127};
    const float x[] = {1.0f, 2.0f};

    float mm[2] = {0};
    coli_mxfp4_matmul(mm, x, identity, scales, 1, 2, 2);
    if (!closef(mm[0], 1.0f) || !closef(mm[1], 2.0f)) {
        fprintf(stderr, "MXFP4 identity matmul mismatch: %.9g %.9g\n", mm[0], mm[1]);
        return 1;
    }

    float output[2] = {0};
    float gate[2], up[2], h[2], y[2];
    coli_mxfp4_swiglu_expert(output, x,
                             identity, scales,
                             identity, scales,
                             identity, scales,
                             1, 2, 2, 0.5f,
                             gate, up, h, y);

    const float want0 = 0.5f * (1.0f / (1.0f + expf(-1.0f)));
    const float want1 = 0.5f * (2.0f / (1.0f + expf(-2.0f)) * 2.0f);
    if (!closef(output[0], want0) || !closef(output[1], want1)) {
        fprintf(stderr,
                "MXFP4 SwiGLU expert mismatch: got %.9g %.9g want %.9g %.9g\n",
                output[0], output[1], want0, want1);
        return 1;
    }

    puts("test_mxfp4_runtime: ok");
    return 0;
}

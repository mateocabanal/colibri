#include "../mxfp4_runtime.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(c, ...) do { if (!(c)) { \
    ++failures; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); \
} } while (0)

static size_t canonical_weight_bytes(int O, int I) {
    return (size_t)O * (size_t)((I + 1) / 2);
}
static size_t canonical_scale_bytes(int O, int I) {
    return (size_t)O * (size_t)((I + 31) / 32);
}
static size_t apple8_bytes(int O, int I) {
    return (size_t)((O + 7) / 8) * (size_t)((I + 31) / 32) *
           COLI_APPLE8_MXFP4_TILE_BYTES;
}

static void make_canonical(uint8_t *w, uint8_t *sc, int O, int I, int seed) {
    const int rb = (I + 1) / 2, ng = (I + 31) / 32;
    memset(w, 0, canonical_weight_bytes(O, I));
    for (int o = 0; o < O; ++o) {
        for (int i = 0; i < I; ++i) {
            uint8_t code = (uint8_t)((seed + o * 5 + i * 3) & 15);
            uint8_t *b = &w[(size_t)o * rb + (size_t)(i >> 1)];
            if (i & 1) *b = (uint8_t)((*b & 0x0f) | (code << 4));
            else       *b = (uint8_t)((*b & 0xf0) | code);
        }
        for (int g = 0; g < ng; ++g)
            sc[(size_t)o * ng + g] = (uint8_t)(125 + ((seed + o + g) % 5));
    }
}

static void pack_apple8(uint8_t *dst, const uint8_t *w, const uint8_t *sc,
                        int O, int I) {
    const int rb = (I + 1) / 2, ng = (I + 31) / 32;
    memset(dst, 0, apple8_bytes(O, I));
    for (int o = 0; o < O; ++o) {
        for (int g = 0; g < ng; ++g) {
            uint8_t *tile = dst +
                ((size_t)(o / 8) * ng + (size_t)g) *
                COLI_APPLE8_MXFP4_TILE_BYTES;
            uint8_t *row = tile + (size_t)(o & 7) *
                COLI_APPLE8_MXFP4_WEIGHT_ROW_BYTES;
            const int first_byte = g * 16;
            int bytes = rb - first_byte;
            if (bytes > 16) bytes = 16;
            if (bytes > 0)
                memcpy(row, w + (size_t)o * rb + first_byte, (size_t)bytes);
            tile[COLI_APPLE8_MXFP4_WEIGHT_BYTES + (o & 7)] =
                sc[(size_t)o * ng + g];
        }
    }
}

static int same_float(float a, float b) {
    float tol = 2e-6f * (1.0f + fabsf(a));
    return fabsf(a - b) <= tol;
}

static void test_matmul_shape(int S, int I, int O) {
    size_t wb = canonical_weight_bytes(O, I), sb = canonical_scale_bytes(O, I);
    size_t ab = apple8_bytes(O, I);
    uint8_t *w = malloc(wb), *sc = malloc(sb), *a8 = malloc(ab);
    float *x = malloc((size_t)S * I * sizeof(float));
    float *raw = malloc((size_t)S * O * sizeof(float));
    float *tile = malloc((size_t)S * O * sizeof(float));
    ColiMxfp4RuntimeTag tag;
    CHECK(w && sc && a8 && x && raw && tile, "allocation");
    if (!w || !sc || !a8 || !x || !raw || !tile) goto done;
    make_canonical(w, sc, O, I, 3);
    pack_apple8(a8, w, sc, O, I);
    for (int i = 0; i < S * I; ++i)
        x[i] = (float)(((i * 17 + 5) % 37) - 18) / 11.0f;
    coli_mxfp4_matmul(raw, x, w, sc, S, I, O);
    CHECK(!coli_mxfp4_matmul_layout(tile, x, a8, NULL,
                                    COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1,
                                    S, I, O), "Apple8 explicit matmul refused");
    for (int i = 0; i < S * O; ++i)
        CHECK(same_float(raw[i], tile[i]),
              "matmul %dx%d row %d canonical %.9g Apple8 %.9g",
              O, I, i, raw[i], tile[i]);
    coli_mxfp4_runtime_tag_init(&tag, COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
    memset(tile, 0, (size_t)S * O * sizeof(float));
    coli_mxfp4_matmul_auto(tile, x, a8, (const uint8_t *)&tag, S, I, O);
    for (int i = 0; i < S * O; ++i)
        CHECK(same_float(raw[i], tile[i]), "auto dispatch mismatch at %d", i);
done:
    free(w); free(sc); free(a8); free(x); free(raw); free(tile);
}

static void test_swiglu(void) {
    enum { S = 2, D = 33, I = 9 };
    uint8_t *gw = malloc(canonical_weight_bytes(I, D));
    uint8_t *gs = malloc(canonical_scale_bytes(I, D));
    uint8_t *uw = malloc(canonical_weight_bytes(I, D));
    uint8_t *us = malloc(canonical_scale_bytes(I, D));
    uint8_t *dw = malloc(canonical_weight_bytes(D, I));
    uint8_t *ds = malloc(canonical_scale_bytes(D, I));
    uint8_t *ga = malloc(apple8_bytes(I, D));
    uint8_t *ua = malloc(apple8_bytes(I, D));
    uint8_t *da = malloc(apple8_bytes(D, I));
    float x[S * D], out_c[S * D], out_a[S * D];
    float cg[S * I], cu[S * I], ch[S * I], cy[S * D];
    float ag[S * I], au[S * I], ah[S * I], ay[S * D];
    ColiMxfp4RuntimeTag gt, ut, dt;
    CHECK(gw && gs && uw && us && dw && ds && ga && ua && da, "swiglu allocation");
    if (!gw || !gs || !uw || !us || !dw || !ds || !ga || !ua || !da) goto done;
    make_canonical(gw, gs, I, D, 1);
    make_canonical(uw, us, I, D, 7);
    make_canonical(dw, ds, D, I, 11);
    pack_apple8(ga, gw, gs, I, D);
    pack_apple8(ua, uw, us, I, D);
    pack_apple8(da, dw, ds, D, I);
    for (int n = 0; n < S * D; ++n) x[n] = (float)((n * 13) % 29 - 14) / 9.0f;
    memset(out_c, 0, sizeof(out_c)); memset(out_a, 0, sizeof(out_a));
    coli_mxfp4_swiglu_expert(out_c, x, gw, gs, uw, us, dw, ds,
                              S, D, I, 0.625f, cg, cu, ch, cy);
    coli_mxfp4_runtime_tag_init(&gt, COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
    coli_mxfp4_runtime_tag_init(&ut, COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
    coli_mxfp4_runtime_tag_init(&dt, COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
    coli_mxfp4_swiglu_expert_auto(out_a, x,
                                   ga, (const uint8_t *)&gt,
                                   ua, (const uint8_t *)&ut,
                                   da, (const uint8_t *)&dt,
                                   S, D, I, 0.625f, ag, au, ah, ay);
    for (int n = 0; n < S * D; ++n)
        CHECK(same_float(out_c[n], out_a[n]),
              "swiglu %d canonical %.9g Apple8 %.9g", n, out_c[n], out_a[n]);
done:
    free(gw); free(gs); free(uw); free(us); free(dw); free(ds);
    free(ga); free(ua); free(da);
}

int main(void) {
    test_matmul_shape(2, 1, 1);
    test_matmul_shape(2, 31, 7);
    test_matmul_shape(2, 32, 8);
    test_matmul_shape(2, 33, 9);
    test_swiglu();
    if (failures) {
        fprintf(stderr, "MXFP4_APPLE8_RUNTIME failures=%d\n", failures);
        return 1;
    }
    puts("MXFP4_APPLE8_RUNTIME ok");
    return 0;
}

#include "../apple8_contract.h"
#include "../apple8_metalio_direct.h"
#include "../metalio.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static const float MX4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static float ue8m0(uint8_t e) {
    uint32_t bits = (uint32_t)e << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint8_t code_for(int row, int col) {
    return (uint8_t)((row * 7 + col * 5 + 3) & 15);
}

static uint8_t scale_for(int row, int group) {
    /* Keep scales around 1.0 so the oracle remains comfortably finite. */
    return (uint8_t)(124 + ((row * 3 + group * 5) % 7));
}

static uint8_t *make_apple8(int O, int I, size_t *bytes_out) {
    uint64_t bytes_u64 = 0;
    if (coli_apple8_tile_matrix_bytes((uint64_t)O, (uint64_t)I, &bytes_u64) != 0 ||
        bytes_u64 > SIZE_MAX)
        return NULL;
    uint8_t *tiles = (uint8_t *)calloc((size_t)bytes_u64, 1);
    if (!tiles) return NULL;
    const int groups = (I + 31) / 32;
    for (int o = 0; o < O; ++o) {
        const int output_tile = o / 8;
        const int tile_row = o % 8;
        for (int g = 0; g < groups; ++g) {
            uint8_t *tile = tiles + ((size_t)output_tile * (size_t)groups + (size_t)g) * 136u;
            for (int lane = 0; lane < 32; ++lane) {
                const int k = g * 32 + lane;
                if (k >= I) break;
                const uint8_t code = code_for(o, k);
                uint8_t *p = &tile[tile_row * 16 + lane / 2];
                if (lane & 1) *p = (uint8_t)((*p & 0x0f) | (code << 4));
                else *p = (uint8_t)((*p & 0xf0) | code);
            }
            tile[128 + tile_row] = scale_for(o, g);
        }
    }
    *bytes_out = (size_t)bytes_u64;
    return tiles;
}

static void cpu_reference(const float *x, float *y, int S, int I, int O) {
    for (int s = 0; s < S; ++s) {
        for (int o = 0; o < O; ++o) {
            float acc = 0.0f;
            for (int k = 0; k < I; ++k) {
                const int g = k / 32;
                acc += MX4[code_for(o, k)] * ue8m0(scale_for(o, g)) * x[(size_t)s * I + k];
            }
            y[(size_t)s * O + o] = acc;
        }
    }
}

static int write_all(int fd, const uint8_t *p, size_t n) {
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return 0;
        p += (size_t)w;
        n -= (size_t)w;
    }
    return 1;
}

int main(void) {
    const int S = 3, I = 65, O = 17;
    const size_t x_count = (size_t)S * (size_t)I;
    const size_t y_count = (size_t)S * (size_t)O;
    size_t apple8_bytes = 0;
    uint8_t *apple8 = make_apple8(O, I, &apple8_bytes);
    float *x = (float *)malloc(x_count * sizeof(float));
    float *ref = (float *)calloc(y_count, sizeof(float));
    float *got = (float *)calloc(y_count, sizeof(float));
    int file = -1;
    int slot = -1;
    int metalio_started = 0;
    char path[] = "/tmp/colibri-apple8-metalio-XXXXXX";

    CHECK(apple8 && x && ref && got, "fixture allocation failed");
    if (!apple8 || !x || !ref || !got) goto cleanup;

    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp failed");
    if (fd < 0) goto cleanup;
    CHECK(write_all(fd, apple8, apple8_bytes), "fixture write failed");
    close(fd);

    if (!metalio_init()) {
        fprintf(stderr, "SKIP apple8-metalio-direct: MetalIO unavailable\n");
        unlink(path);
        free(apple8); free(x); free(ref); free(got);
        return 0;
    }
    metalio_started = 1;
    file = metalio_file_add(path);
    CHECK(file >= 0, "metalio_file_add failed");
    slot = metalio_slot_alloc(apple8_bytes);
    CHECK(slot >= 0, "metalio_slot_alloc failed");
    if (file < 0 || slot < 0) goto cleanup;

    {
        int64_t ev = metalio_load(slot, file, 0, apple8_bytes);
        CHECK(ev > 0, "metalio_load failed");
        CHECK(ev > 0 && metalio_wait(ev) == 0, "metalio_wait failed");
        if (ev <= 0) goto cleanup;
    }
    {
        const void *slot_bytes = metalio_slot_ptr(slot);
        CHECK(slot_bytes != NULL, "slot pointer missing");
        CHECK(slot_bytes && memcmp(slot_bytes, apple8, apple8_bytes) == 0,
              "MetalIO slot bytes differ from raw Apple8 file bytes");
    }

    for (size_t i = 0; i < x_count; ++i)
        x[i] = ((int)(i % 23) - 11) * 0.03125f;
    cpu_reference(x, ref, S, I, O);

    CHECK(coli_apple8_metalio_direct_init(), "direct Apple8 Metal init failed");
    CHECK(coli_apple8_metalio_matmul_slot(slot, 0, apple8_bytes,
                                          x, got, S, I, O),
          "direct Apple8 Metal matmul failed");
    for (size_t i = 0; i < y_count; ++i) {
        float tol = 1e-4f * (1.0f + fabsf(ref[i]));
        CHECK(fabsf(got[i] - ref[i]) <= tol,
              "output[%zu] got=%g ref=%g tol=%g", i, got[i], ref[i], tol);
    }

    {
        ColiMetalioStats stats = {};
        metalio_stats(&stats);
        CHECK(stats.loads >= 1, "MetalIO load counter did not advance");
        CHECK(stats.bytes >= apple8_bytes,
              "MetalIO byte counter %llu < fixture %zu",
              (unsigned long long)stats.bytes, apple8_bytes);
    }

cleanup:
    coli_apple8_metalio_direct_shutdown();
    if (slot >= 0) metalio_slot_free(slot);
    if (metalio_started) metalio_shutdown();
    unlink(path);
    free(apple8); free(x); free(ref); free(got);
    if (failures) return 1;
    puts("APPLE8_METALIO_DIRECT PASS");
    return 0;
}

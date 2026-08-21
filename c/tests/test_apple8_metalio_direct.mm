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

static size_t align16(size_t n) {
    return (n + 15u) & ~(size_t)15u;
}

static uint8_t *make_apple8(int O, int I, int seed, size_t *bytes_out) {
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
                const uint8_t code = (uint8_t)((o * 7 + k * 5 + seed * 3 + 3) & 15);
                uint8_t *p = &tile[tile_row * 16 + lane / 2];
                if (lane & 1) *p = (uint8_t)((*p & 0x0f) | (code << 4));
                else *p = (uint8_t)((*p & 0xf0) | code);
            }
            /* 2^(e-127), deliberately small enough to keep SwiGLU finite. */
            tile[128 + tile_row] = (uint8_t)(122 + ((o * 3 + g * 5 + seed) % 5));
        }
    }
    *bytes_out = (size_t)bytes_u64;
    return tiles;
}

static void cpu_matmul_tiles(const uint8_t *tiles,
                             const float *x, float *y,
                             int S, int I, int O) {
    const int groups = (I + 31) / 32;
    for (int s = 0; s < S; ++s) {
        for (int o = 0; o < O; ++o) {
            const int output_tile = o / 8;
            const int tile_row = o % 8;
            float acc = 0.0f;
            for (int k = 0; k < I; ++k) {
                const int g = k / 32;
                const int lane = k & 31;
                const uint8_t *tile = tiles +
                    ((size_t)output_tile * (size_t)groups + (size_t)g) * 136u;
                const uint8_t packed = tile[tile_row * 16 + lane / 2];
                const uint8_t code = (lane & 1) ? (packed >> 4) : (packed & 15u);
                acc += MX4[code] * ue8m0(tile[128 + tile_row]) *
                       x[(size_t)s * (size_t)I + (size_t)k];
            }
            y[(size_t)s * (size_t)O + (size_t)o] = acc;
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
    const int S = 3, H = 65, M = 17;
    const size_t x_count = (size_t)S * (size_t)H;
    const size_t mid_count = (size_t)S * (size_t)M;
    const size_t y_count = (size_t)S * (size_t)H;

    size_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
    uint8_t *gate = make_apple8(M, H, 1, &gate_bytes);
    uint8_t *up = make_apple8(M, H, 2, &up_bytes);
    uint8_t *down = make_apple8(H, M, 3, &down_bytes);
    size_t gate_off = 0;
    size_t up_off = align16(gate_off + gate_bytes);
    size_t down_off = align16(up_off + up_bytes);
    size_t file_bytes = down_off + down_bytes;
    uint8_t *file_image = (uint8_t *)calloc(file_bytes, 1);

    float *x = (float *)malloc(x_count * sizeof(float));
    float *gate_ref = (float *)calloc(mid_count, sizeof(float));
    float *up_ref = (float *)calloc(mid_count, sizeof(float));
    float *mid_ref = (float *)calloc(mid_count, sizeof(float));
    float *matmul_got = (float *)calloc(mid_count, sizeof(float));
    float *ref = (float *)calloc(y_count, sizeof(float));
    float *got = (float *)calloc(y_count, sizeof(float));

    int fd = -1, file = -1, slot = -1, metalio_started = 0;
    char path[] = "/tmp/colibri-apple8-metalio-XXXXXX";

    CHECK(gate && up && down && file_image && x && gate_ref && up_ref && mid_ref &&
          matmul_got && ref && got, "fixture allocation failed");
    if (!gate || !up || !down || !file_image || !x || !gate_ref || !up_ref ||
        !mid_ref || !matmul_got || !ref || !got)
        goto cleanup;

    memcpy(file_image + gate_off, gate, gate_bytes);
    memcpy(file_image + up_off, up, up_bytes);
    memcpy(file_image + down_off, down, down_bytes);

    fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp failed");
    if (fd < 0) goto cleanup;
    CHECK(write_all(fd, file_image, file_bytes), "fixture write failed");
    close(fd); fd = -1;

    if (!metalio_init()) {
        fprintf(stderr, "SKIP apple8-metalio-direct: MetalIO unavailable\n");
        goto cleanup;
    }
    metalio_started = 1;
    file = metalio_file_add(path);
    CHECK(file >= 0, "metalio_file_add failed");
    slot = metalio_slot_alloc(file_bytes);
    CHECK(slot >= 0, "metalio_slot_alloc failed");
    if (file < 0 || slot < 0) goto cleanup;

    {
        ColiMetalioRegion regions[3] = {
            { file, (uint64_t)gate_off, gate_bytes, (uint64_t)gate_off },
            { file, (uint64_t)up_off, up_bytes, (uint64_t)up_off },
            { file, (uint64_t)down_off, down_bytes, (uint64_t)down_off },
        };
        int64_t ev = metalio_loadv(slot, regions, 3, MIO_LOAD_DEMAND);
        CHECK(ev > 0, "metalio_loadv failed");
        CHECK(ev > 0 && metalio_wait(ev) == 0, "metalio_wait failed");
        if (ev <= 0) goto cleanup;
    }

    {
        const uint8_t *slot_bytes = (const uint8_t *)metalio_slot_ptr(slot);
        CHECK(slot_bytes != NULL, "slot pointer missing");
        CHECK(metalio_slot_native_buffer(slot) != NULL, "native MTLBuffer handle missing");
        CHECK(slot_bytes && memcmp(slot_bytes + gate_off, gate, gate_bytes) == 0,
              "gate bytes differ after MetalIO load");
        CHECK(slot_bytes && memcmp(slot_bytes + up_off, up, up_bytes) == 0,
              "up bytes differ after MetalIO load");
        CHECK(slot_bytes && memcmp(slot_bytes + down_off, down, down_bytes) == 0,
              "down bytes differ after MetalIO load");
    }

    for (size_t i = 0; i < x_count; ++i)
        x[i] = ((int)(i % 23) - 11) * 0.03125f;

    CHECK(coli_apple8_metalio_direct_init(), "direct Apple8 Metal init failed");

    cpu_matmul_tiles(gate, x, gate_ref, S, H, M);
    CHECK(coli_apple8_metalio_matmul_slot(slot, gate_off, gate_bytes,
                                          x, matmul_got, S, H, M),
          "direct Apple8 Metal matmul failed");
    for (size_t i = 0; i < mid_count; ++i) {
        float tol = 1e-4f * (1.0f + fabsf(gate_ref[i]));
        CHECK(fabsf(matmul_got[i] - gate_ref[i]) <= tol,
              "matmul[%zu] got=%g ref=%g tol=%g",
              i, matmul_got[i], gate_ref[i], tol);
    }

    cpu_matmul_tiles(up, x, up_ref, S, H, M);
    for (size_t i = 0; i < mid_count; ++i)
        mid_ref[i] = (gate_ref[i] / (1.0f + expf(-gate_ref[i]))) * up_ref[i];
    cpu_matmul_tiles(down, mid_ref, ref, S, M, H);

    CHECK(coli_apple8_metalio_swiglu_slot(slot,
                                           gate_off, gate_bytes,
                                           up_off, up_bytes,
                                           down_off, down_bytes,
                                           x, got, S, H, M),
          "direct Apple8 Metal SwiGLU failed");
    for (size_t i = 0; i < y_count; ++i) {
        float tol = 5e-4f * (1.0f + fabsf(ref[i]));
        CHECK(fabsf(got[i] - ref[i]) <= tol,
              "swiglu[%zu] got=%g ref=%g tol=%g", i, got[i], ref[i], tol);
    }

    {
        ColiMetalioStats stats = {};
        metalio_stats(&stats);
        uint64_t logical_bytes = (uint64_t)gate_bytes + (uint64_t)up_bytes + (uint64_t)down_bytes;
        CHECK(stats.loads >= 1, "MetalIO load counter did not advance");
        CHECK(stats.bytes >= logical_bytes,
              "MetalIO byte counter %llu < payload %llu",
              (unsigned long long)stats.bytes,
              (unsigned long long)logical_bytes);
    }

cleanup:
    coli_apple8_metalio_direct_shutdown();
    if (fd >= 0) close(fd);
    if (slot >= 0) metalio_slot_free(slot);
    if (metalio_started) metalio_shutdown();
    unlink(path);
    free(gate); free(up); free(down); free(file_image);
    free(x); free(gate_ref); free(up_ref); free(mid_ref);
    free(matmul_got); free(ref); free(got);
    if (failures) return 1;
    puts("APPLE8_METALIO_DIRECT PASS");
    return 0;
}

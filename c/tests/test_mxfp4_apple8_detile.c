#include "../mxfp4_expert.h"
#include "../apple8_contract.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static size_t canonical_weight_bytes(uint64_t rows, uint64_t columns) {
    return (size_t)(rows * (columns / 2u + (columns & 1u)));
}
static size_t canonical_scale_bytes(uint64_t rows, uint64_t columns) {
    return (size_t)(rows * (columns / 32u + (columns % 32u != 0)));
}

static void make_canonical(uint8_t *weights, uint8_t *scales,
                           uint64_t rows, uint64_t columns) {
    uint64_t row_bytes = columns / 2u + (columns & 1u);
    uint64_t groups = columns / 32u + (columns % 32u != 0);
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t column = 0; column < columns; ++column) {
            uint8_t code = (uint8_t)((row * 7u + column * 5u + 3u) & 0x0fu);
            uint8_t *byte = &weights[row * row_bytes + column / 2u];
            if (column & 1u) *byte |= (uint8_t)(code << 4);
            else *byte = (uint8_t)((*byte & 0xf0u) | code);
        }
        /* Frozen canonical contract requires the unused odd-K high nibble zero. */
        if (columns & 1u)
            weights[row * row_bytes + row_bytes - 1u] &= 0x0fu;
        for (uint64_t group = 0; group < groups; ++group)
            scales[row * groups + group] =
                (uint8_t)(120u + ((row * 3u + group * 11u) % 17u));
    }
}

static uint8_t *pack_apple8(const uint8_t *weights, const uint8_t *scales,
                            uint64_t rows, uint64_t columns, size_t *bytes_out) {
    uint64_t tile_bytes = 0;
    uint64_t groups = columns / 32u + (columns % 32u != 0);
    uint64_t row_bytes = columns / 2u + (columns & 1u);
    uint8_t *tiles;
    CHECK(coli_apple8_tile_matrix_bytes(rows, columns, &tile_bytes) == 0,
          "tile size rejected");
    if (tile_bytes > SIZE_MAX) return NULL;
    tiles = (uint8_t *)calloc((size_t)tile_bytes, 1);
    if (!tiles) return NULL;
    for (uint64_t row = 0; row < rows; ++row) {
        uint64_t output_tile = row / COLI_APPLE8_MXFP4_TILE_ROWS;
        uint64_t tile_row = row % COLI_APPLE8_MXFP4_TILE_ROWS;
        for (uint64_t group = 0; group < groups; ++group) {
            uint64_t tile_index = output_tile * groups + group;
            uint8_t *tile = tiles + tile_index * COLI_APPLE8_MXFP4_TILE_BYTES;
            uint64_t base_column = group * COLI_APPLE8_MXFP4_TILE_COLUMNS;
            uint64_t used_columns = columns - base_column;
            size_t used_bytes;
            if (used_columns > COLI_APPLE8_MXFP4_TILE_COLUMNS)
                used_columns = COLI_APPLE8_MXFP4_TILE_COLUMNS;
            used_bytes = (size_t)(used_columns / 2u + (used_columns & 1u));
            memcpy(tile + tile_row * 16u,
                   weights + row * row_bytes + group * 16u,
                   used_bytes);
            tile[128u + tile_row] = scales[row * groups + group];
        }
    }
    *bytes_out = (size_t)tile_bytes;
    return tiles;
}

static void run_shape(uint64_t rows, uint64_t columns) {
    size_t wb = canonical_weight_bytes(rows, columns);
    size_t sb = canonical_scale_bytes(rows, columns);
    uint8_t *weights = (uint8_t *)calloc(wb, 1);
    uint8_t *scales = (uint8_t *)calloc(sb, 1);
    uint8_t *got_w = (uint8_t *)malloc(wb);
    uint8_t *got_s = (uint8_t *)malloc(sb);
    uint8_t *tiles;
    size_t tile_bytes = 0;
    char error[256] = {0};
    CHECK(weights && scales && got_w && got_s, "allocation failed");
    if (!weights || !scales || !got_w || !got_s) goto done;
    make_canonical(weights, scales, rows, columns);
    tiles = pack_apple8(weights, scales, rows, columns, &tile_bytes);
    CHECK(tiles != NULL, "pack failed for %llux%llu",
          (unsigned long long)rows, (unsigned long long)columns);
    if (!tiles) goto done;
    memset(got_w, 0xa5, wb);
    memset(got_s, 0xa5, sb);
    CHECK(coli_mxfp4_apple8_detile_matrix(
              tiles, tile_bytes, rows, columns,
              got_w, wb, got_s, sb, error, sizeof(error)) == 0,
          "detile %llux%llu: %s",
          (unsigned long long)rows, (unsigned long long)columns, error);
    CHECK(memcmp(weights, got_w, wb) == 0,
          "weight mismatch for %llux%llu",
          (unsigned long long)rows, (unsigned long long)columns);
    CHECK(memcmp(scales, got_s, sb) == 0,
          "scale mismatch for %llux%llu",
          (unsigned long long)rows, (unsigned long long)columns);

    if (columns & 1u) {
        uint64_t groups = columns / 32u + (columns % 32u != 0);
        uint64_t output_tile = 0;
        uint64_t group = groups - 1u;
        uint64_t tile_index = output_tile * groups + group;
        uint8_t *dirty = tiles + tile_index * COLI_APPLE8_MXFP4_TILE_BYTES;
        size_t used = (size_t)((columns - group * 32u + 1u) / 2u);
        dirty[used - 1u] |= 0xf0u;
        error[0] = 0;
        CHECK(coli_mxfp4_apple8_detile_matrix(
                  tiles, tile_bytes, rows, columns,
                  got_w, wb, got_s, sb, error, sizeof(error)) != 0,
              "dirty odd-K high nibble accepted for %llux%llu",
              (unsigned long long)rows, (unsigned long long)columns);
    }
    free(tiles);
done:
    free(weights);
    free(scales);
    free(got_w);
    free(got_s);
}

int main(void) {
    static const uint64_t shapes[][2] = {
        {1,1}, {1,31}, {1,32}, {1,33}, {7,32},
        {8,32}, {9,32}, {8,31}, {8,33}, {9,33},
        {2048,4096}, {4096,2048},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i)
        run_shape(shapes[i][0], shapes[i][1]);
    if (failures) return 1;
    puts("MXFP4_APPLE8_DETILE PASS");
    return 0;
}

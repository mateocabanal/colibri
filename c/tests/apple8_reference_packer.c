#include "../generated/coli_target_registry.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int mul_size(size_t a, size_t b, size_t *out) {
    if (!out || (a && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static int parse_u64(const char *text, uint64_t *out) {
    char *end = NULL;
    unsigned long long value;
    if (!text || !*text || !out) return -1;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno || !end || *end || !value) return -1;
    *out = (uint64_t)value;
    return 0;
}

static unsigned char *read_exact(const char *path, size_t bytes) {
    FILE *file = fopen(path, "rb");
    unsigned char *data;
    int extra;
    if (!file) return NULL;
    data = (unsigned char *)malloc(bytes ? bytes : 1);
    if (!data) { fclose(file); return NULL; }
    if (bytes && fread(data, 1, bytes, file) != bytes) {
        free(data); fclose(file); return NULL;
    }
    extra = fgetc(file);
    if (extra != EOF) { free(data); fclose(file); return NULL; }
    if (fclose(file)) { free(data); return NULL; }
    return data;
}

static int write_exact(const char *path, const void *data, size_t bytes) {
    FILE *file = fopen(path, "wb");
    if (!file) return -1;
    if (bytes && fwrite(data, 1, bytes, file) != bytes) {
        fclose(file); return -1;
    }
    return fclose(file) ? -1 : 0;
}

int main(int argc, char **argv) {
    uint64_t rows64, columns64;
    size_t rows, columns, row_bytes, groups, weight_bytes, scale_bytes;
    size_t out_tiles, k_tiles, tile_count, output_bytes;
    unsigned char *weights = NULL, *scales = NULL, *output = NULL;

    if (argc != 6 || parse_u64(argv[1], &rows64) ||
        parse_u64(argv[2], &columns64) ||
        rows64 > SIZE_MAX || columns64 > SIZE_MAX) {
        fprintf(stderr, "usage: %s ROWS COLUMNS WEIGHTS SCALES OUTPUT\n", argv[0]);
        return 2;
    }
    rows = (size_t)rows64;
    columns = (size_t)columns64;
    if (columns > SIZE_MAX - 1 || columns > SIZE_MAX - 31 || rows > SIZE_MAX - 7)
        return 2;
    row_bytes = (columns + 1) / 2;
    groups = (columns + 31) / 32;
    out_tiles = (rows + 7) / 8;
    k_tiles = groups;
    if (mul_size(rows, row_bytes, &weight_bytes) ||
        mul_size(rows, groups, &scale_bytes) ||
        mul_size(out_tiles, k_tiles, &tile_count) ||
        mul_size(tile_count, COLI_APPLE8_MXFP4_TILE_BYTES, &output_bytes))
        return 2;

    weights = read_exact(argv[3], weight_bytes);
    scales = read_exact(argv[4], scale_bytes);
    output = (unsigned char *)calloc(output_bytes ? output_bytes : 1, 1);
    if (!weights || !scales || !output) goto fail;

    for (size_t row = 0; row < rows; ++row) {
        const unsigned char *weight_row = weights + row * row_bytes;
        const unsigned char *scale_row = scales + row * groups;
        size_t output_tile = row / COLI_APPLE8_MXFP4_TILE_ROWS;
        size_t output_row = row % COLI_APPLE8_MXFP4_TILE_ROWS;
        for (size_t group = 0; group < groups; ++group) {
            size_t tile_index = output_tile * groups + group;
            unsigned char *tile = output + tile_index * COLI_APPLE8_MXFP4_TILE_BYTES;
            size_t column = group * COLI_APPLE8_MXFP4_TILE_COLUMNS;
            size_t remaining = columns - column;
            size_t logical_columns = remaining < COLI_APPLE8_MXFP4_TILE_COLUMNS
                ? remaining : COLI_APPLE8_MXFP4_TILE_COLUMNS;
            size_t copy_bytes = (logical_columns + 1) / 2;
            memcpy(tile + output_row * COLI_APPLE8_MXFP4_WEIGHT_ROW_BYTES,
                   weight_row + column / 2, copy_bytes);
            tile[COLI_APPLE8_MXFP4_WEIGHT_BYTES + output_row] = scale_row[group];
        }
    }

    if (write_exact(argv[5], output, output_bytes)) goto fail;
    free(output); free(scales); free(weights);
    return 0;

fail:
    free(output); free(scales); free(weights);
    return 1;
}

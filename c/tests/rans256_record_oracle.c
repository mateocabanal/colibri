#include "../rans.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_BYTES 160u
#define SCALE_BITS 14u
#define NSTREAMS 256u

static uint16_t rd16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t rd32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static unsigned char *read_file(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb");
    unsigned char *data = NULL;
    long n;
    if (!f || fseek(f, 0, SEEK_END) || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) {
        if (f) fclose(f);
        return NULL;
    }
    data = (unsigned char *)malloc((size_t)n ? (size_t)n : 1u);
    if (!data || ((size_t)n && fread(data, 1, (size_t)n, f) != (size_t)n)) {
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *bytes = (size_t)n;
    return data;
}

static int write_file(const char *path, const unsigned char *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    int rc;
    if (!f) return -1;
    rc = bytes && fwrite(data, 1, bytes, f) != bytes;
    if (fclose(f)) rc = 1;
    return rc ? -1 : 0;
}

int main(int argc, char **argv) {
    static const unsigned char magic[8] = {'C','O','L','I','R','N','0','1'};
    unsigned char *blob = NULL, *input = NULL, *nibbles = NULL, *encoded = NULL;
    uint32_t freq[16], start[16];
    uint16_t *slots = NULL;
    rans_table table;
    uint64_t n_symbols, bound, encoded_bytes = 0;
    size_t blob_bytes = 0, input_bytes = 0;
    uint32_t cursor = 0;
    unsigned s;
    rans_err rc;
    int status = 1;

    if (argc != 4) {
        fprintf(stderr, "usage: %s TABLE_BLOB INPUT OUTPUT\n", argv[0]);
        return 2;
    }
    blob = read_file(argv[1], &blob_bytes);
    input = read_file(argv[2], &input_bytes);
    if (!blob || !input || blob_bytes != TABLE_BYTES || input_bytes == 0 ||
        memcmp(blob, magic, 8) || rd16(blob + 8) != 1 ||
        rd16(blob + 10) != SCALE_BITS || rd32(blob + 12) != NSTREAMS) {
        fprintf(stderr, "invalid fixture input\n");
        goto done;
    }
    for (s = 0; s < 16; ++s) {
        freq[s] = rd32(blob + 32 + s * 4u);
        start[s] = rd32(blob + 96 + s * 4u);
        if (start[s] != cursor || freq[s] > (1u << SCALE_BITS) - cursor) {
            fprintf(stderr, "invalid freq/start\n");
            goto done;
        }
        cursor += freq[s];
    }
    if (cursor != (1u << SCALE_BITS)) goto done;
    slots = (uint16_t *)malloc((1u << SCALE_BITS) * sizeof(*slots));
    if (!slots) goto done;
    cursor = 0;
    for (s = 0; s < 16; ++s) {
        uint32_t k;
        for (k = 0; k < freq[s]; ++k) slots[cursor + k] = (uint16_t)s;
        cursor += freq[s];
    }
    rc = rans_table_init(&table, SCALE_BITS, freq, start, slots);
    if (rc != RANS_OK) {
        fprintf(stderr, "table: %s\n", rans_err_name(rc));
        goto done;
    }
    if (input_bytes > UINT64_MAX / 2u) goto free_table;
    n_symbols = (uint64_t)input_bytes * 2u;
    nibbles = (unsigned char *)malloc((size_t)n_symbols);
    if (!nibbles) goto free_table;
    for (size_t i = 0; i < input_bytes; ++i) {
        nibbles[2u * i] = input[i] & 0x0fu;
        nibbles[2u * i + 1u] = input[i] >> 4;
    }
    bound = rans_record_bound(n_symbols, NSTREAMS);
    if (!bound || bound > SIZE_MAX) goto free_table;
    encoded = (unsigned char *)malloc((size_t)bound);
    if (!encoded) goto free_table;
    rc = rans_record_encode(nibbles, n_symbols, &table, NSTREAMS,
                            encoded, bound, &encoded_bytes);
    if (rc != RANS_OK) {
        fprintf(stderr, "encode: %s\n", rans_err_name(rc));
        goto free_table;
    }
    if (encoded_bytes > SIZE_MAX || write_file(argv[3], encoded, (size_t)encoded_bytes))
        goto free_table;
    status = 0;

free_table:
    rans_table_free(&table);
done:
    free(encoded);
    free(nibbles);
    free(slots);
    free(input);
    free(blob);
    return status;
}

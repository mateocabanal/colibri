#ifndef COLI_SPECULATIVE_CACHE_DISK_H
#define COLI_SPECULATIVE_CACHE_DISK_H

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "speculative_cache.h"

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
/* POSIX fileno() may be hidden under strict -std=c11 feature profiles. */
extern int fileno(FILE *stream);
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_SPEC_DISK_VERSION 1u
#define COLI_SPEC_DISK_HEADER_BYTES 80u
#define COLI_SPEC_DISK_FINGERPRINT_BYTES 32u

typedef struct ColiSpecDiskIdentity {
    uint8_t model_fingerprint[COLI_SPEC_DISK_FINGERPRINT_BYTES];
    uint64_t tokenizer_fingerprint;
    uint32_t engine_id;
    uint32_t token_abi;
} ColiSpecDiskIdentity;

static inline void coli_spec_disk_put_u32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static inline uint32_t coli_spec_disk_get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline void coli_spec_disk_put_u64(uint8_t *p, uint64_t value) {
    coli_spec_disk_put_u32(p, (uint32_t)value);
    coli_spec_disk_put_u32(p + 4, (uint32_t)(value >> 32));
}

static inline uint64_t coli_spec_disk_get_u64(const uint8_t *p) {
    return (uint64_t)coli_spec_disk_get_u32(p) |
           ((uint64_t)coli_spec_disk_get_u32(p + 4) << 32);
}

static inline uint32_t coli_spec_disk_crc32c_update(uint32_t crc,
                                                    const uint8_t *data,
                                                    size_t bytes) {
    size_t i;
    crc = ~crc;
    for (i = 0; i < bytes; ++i) {
        uint32_t x = (crc ^ data[i]) & 0xffu;
        int bit;
        for (bit = 0; bit < 8; ++bit) {
            x = (x >> 1) ^ (0x82f63b78u & (uint32_t)-(int32_t)(x & 1u));
        }
        crc = (crc >> 8) ^ x;
    }
    return ~crc;
}

static inline int coli_spec_disk_identity_equal(const ColiSpecDiskIdentity *a,
                                                const ColiSpecDiskIdentity *b) {
    return a != NULL && b != NULL && a->engine_id == b->engine_id &&
           a->token_abi == b->token_abi &&
           a->tokenizer_fingerprint == b->tokenizer_fingerprint &&
           memcmp(a->model_fingerprint, b->model_fingerprint,
                  COLI_SPEC_DISK_FINGERPRINT_BYTES) == 0;
}

static inline void coli_spec_disk_build_header(uint8_t header[COLI_SPEC_DISK_HEADER_BYTES],
                                               const ColiSpecDiskIdentity *identity,
                                               uint64_t token_count,
                                               uint32_t payload_crc) {
    static const uint8_t magic[8] = {'C','O','L','I','S','P','C','1'};
    uint32_t header_crc;
    memset(header, 0, COLI_SPEC_DISK_HEADER_BYTES);
    memcpy(header, magic, sizeof(magic));
    coli_spec_disk_put_u32(header + 8, COLI_SPEC_DISK_VERSION);
    coli_spec_disk_put_u32(header + 12, COLI_SPEC_DISK_HEADER_BYTES);
    coli_spec_disk_put_u32(header + 16, identity->engine_id);
    coli_spec_disk_put_u32(header + 20, identity->token_abi);
    coli_spec_disk_put_u64(header + 24, token_count);
    coli_spec_disk_put_u64(header + 32, identity->tokenizer_fingerprint);
    memcpy(header + 40, identity->model_fingerprint, COLI_SPEC_DISK_FINGERPRINT_BYTES);
    coli_spec_disk_put_u32(header + 72, payload_crc);
    header_crc = coli_spec_disk_crc32c_update(0, header, 76);
    coli_spec_disk_put_u32(header + 76, header_crc);
}

static inline int coli_spec_disk_parse_header(const uint8_t header[COLI_SPEC_DISK_HEADER_BYTES],
                                              ColiSpecDiskIdentity *identity,
                                              uint64_t *token_count,
                                              uint32_t *payload_crc) {
    static const uint8_t magic[8] = {'C','O','L','I','S','P','C','1'};
    uint32_t expected_header_crc;
    if (memcmp(header, magic, sizeof(magic)) != 0 ||
        coli_spec_disk_get_u32(header + 8) != COLI_SPEC_DISK_VERSION ||
        coli_spec_disk_get_u32(header + 12) != COLI_SPEC_DISK_HEADER_BYTES) {
        return 0;
    }
    expected_header_crc = coli_spec_disk_crc32c_update(0, header, 76);
    if (expected_header_crc != coli_spec_disk_get_u32(header + 76)) {
        return 0;
    }
    if (identity != NULL) {
        identity->engine_id = coli_spec_disk_get_u32(header + 16);
        identity->token_abi = coli_spec_disk_get_u32(header + 20);
        identity->tokenizer_fingerprint = coli_spec_disk_get_u64(header + 32);
        memcpy(identity->model_fingerprint, header + 40, COLI_SPEC_DISK_FINGERPRINT_BYTES);
    }
    if (token_count != NULL) {
        *token_count = coli_spec_disk_get_u64(header + 24);
    }
    if (payload_crc != NULL) {
        *payload_crc = coli_spec_disk_get_u32(header + 72);
    }
    return 1;
}

static inline int coli_spec_disk_sync_file(FILE *file) {
    if (fflush(file) != 0) {
        return 0;
    }
#ifdef _WIN32
    return _commit(_fileno(file)) == 0;
#else
    return fsync(fileno(file)) == 0;
#endif
}

static inline int coli_spec_disk_replace(const char *tmp_path, const char *path) {
#ifdef _WIN32
    return MoveFileExA(tmp_path, path,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(tmp_path, path) == 0;
#endif
}

/* Persist one complete continuation snapshot. The caller owns serialization
 * between writers; publication itself is temp-file + durable flush + replace. */
static inline int coli_spec_disk_store_atomic(const char *path,
                                              const ColiSpecDiskIdentity *identity,
                                              const int *tokens,
                                              size_t token_count,
                                              int separator) {
    uint8_t header[COLI_SPEC_DISK_HEADER_BYTES];
    uint8_t word[4];
    uint32_t payload_crc = 0;
    char *tmp_path;
    size_t path_len;
    size_t i;
    FILE *file;
    int ok = 0;

    if (path == NULL || identity == NULL || separator != -1 ||
        (token_count != 0 && tokens == NULL) || token_count > UINT64_MAX) {
        return 0;
    }
    path_len = strlen(path);
    if (path_len > SIZE_MAX - 5) {
        return 0;
    }
    tmp_path = (char *)malloc(path_len + 5);
    if (tmp_path == NULL) {
        return 0;
    }
    memcpy(tmp_path, path, path_len);
    memcpy(tmp_path + path_len, ".tmp", 5);

    for (i = 0; i < token_count; ++i) {
        uint32_t encoded;
        if (tokens[i] < 0 && tokens[i] != separator) {
            free(tmp_path);
            return 0;
        }
        encoded = tokens[i] == separator ? UINT32_MAX : (uint32_t)tokens[i];
        coli_spec_disk_put_u32(word, encoded);
        payload_crc = coli_spec_disk_crc32c_update(payload_crc, word, sizeof(word));
    }
    coli_spec_disk_build_header(header, identity, (uint64_t)token_count, payload_crc);

    file = fopen(tmp_path, "wb");
    if (file == NULL) {
        free(tmp_path);
        return 0;
    }
    if (fwrite(header, 1, sizeof(header), file) != sizeof(header)) {
        goto done;
    }
    for (i = 0; i < token_count; ++i) {
        uint32_t encoded = tokens[i] == separator ? UINT32_MAX : (uint32_t)tokens[i];
        coli_spec_disk_put_u32(word, encoded);
        if (fwrite(word, 1, sizeof(word), file) != sizeof(word)) {
            goto done;
        }
    }
    if (!coli_spec_disk_sync_file(file)) {
        goto done;
    }
    if (fclose(file) != 0) {
        file = NULL;
        goto after_close;
    }
    file = NULL;
    if (!coli_spec_disk_replace(tmp_path, path)) {
        goto after_close;
    }
    ok = 1;

after_close:
    if (!ok) {
        (void)remove(tmp_path);
    }
    free(tmp_path);
    return ok;

done:
    (void)fclose(file);
    file = NULL;
    goto after_close;
}

/* Load into an existing caller-owned cache. Any mismatch/corruption/short read
 * is a safe miss and leaves the live cache untouched. */
static inline int coli_spec_disk_load(const char *path,
                                      const ColiSpecDiskIdentity *expected_identity,
                                      ColiSpecContinuationCache *cache) {
    uint8_t header[COLI_SPEC_DISK_HEADER_BYTES];
    ColiSpecDiskIdentity actual_identity;
    uint64_t token_count_u64;
    uint32_t expected_payload_crc;
    uint32_t actual_payload_crc = 0;
    int *tmp = NULL;
    FILE *file;
    size_t token_count;
    size_t i;
    long file_bytes;

    if (path == NULL || expected_identity == NULL || !coli_spec_cache_valid(cache) ||
        cache->separator != -1) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    if (fread(header, 1, sizeof(header), file) != sizeof(header) ||
        !coli_spec_disk_parse_header(header, &actual_identity, &token_count_u64,
                                     &expected_payload_crc) ||
        !coli_spec_disk_identity_equal(&actual_identity, expected_identity) ||
        token_count_u64 > (uint64_t)SIZE_MAX || token_count_u64 > cache->capacity) {
        fclose(file);
        return 0;
    }
    token_count = (size_t)token_count_u64;
    if (token_count > (SIZE_MAX - COLI_SPEC_DISK_HEADER_BYTES) / 4u) {
        fclose(file);
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 0;
    }
    file_bytes = ftell(file);
    if (file_bytes < 0 || (uint64_t)file_bytes !=
        (uint64_t)COLI_SPEC_DISK_HEADER_BYTES + token_count_u64 * 4u ||
        fseek(file, COLI_SPEC_DISK_HEADER_BYTES, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    if (token_count != 0) {
        tmp = (int *)malloc(token_count * sizeof(*tmp));
        if (tmp == NULL) {
            fclose(file);
            return 0;
        }
    }
    for (i = 0; i < token_count; ++i) {
        uint8_t word[4];
        uint32_t encoded;
        if (fread(word, 1, sizeof(word), file) != sizeof(word)) {
            free(tmp);
            fclose(file);
            return 0;
        }
        actual_payload_crc = coli_spec_disk_crc32c_update(actual_payload_crc, word, sizeof(word));
        encoded = coli_spec_disk_get_u32(word);
        if (encoded == UINT32_MAX) {
            tmp[i] = cache->separator;
        } else if (encoded <= (uint32_t)INT_MAX) {
            tmp[i] = (int)encoded;
        } else {
            free(tmp);
            fclose(file);
            return 0;
        }
    }
    if (fclose(file) != 0 || actual_payload_crc != expected_payload_crc) {
        free(tmp);
        return 0;
    }
    if (token_count != 0) {
        memcpy(cache->tokens, tmp, token_count * sizeof(*tmp));
    }
    cache->count = token_count;
    free(tmp);
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* COLI_SPECULATIVE_CACHE_DISK_H */

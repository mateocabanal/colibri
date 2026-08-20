#ifndef COLIBRI_CACHE_IO_H
#define COLIBRI_CACHE_IO_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
/* POSIX fileno() may be hidden under strict -std=c11 feature profiles. */
extern int fileno(FILE *stream);
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline void coli_cache_put_u16(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static inline uint16_t coli_cache_get_u16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline void coli_cache_put_u32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static inline uint32_t coli_cache_get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline void coli_cache_put_u64(uint8_t *p, uint64_t value) {
    coli_cache_put_u32(p, (uint32_t)value);
    coli_cache_put_u32(p + 4, (uint32_t)(value >> 32));
}

static inline uint64_t coli_cache_get_u64(const uint8_t *p) {
    return (uint64_t)coli_cache_get_u32(p) |
           ((uint64_t)coli_cache_get_u32(p + 4) << 32);
}

/* Castagnoli CRC32C, matching the existing speculative persistent-cache
 * framing. Cache files are untrusted/disposable; CRC failure is always a miss. */
static inline uint32_t coli_cache_crc32c_update(uint32_t crc,
                                                const uint8_t *data,
                                                size_t bytes) {
    size_t i;
    crc = ~crc;
    for (i = 0; i < bytes; ++i) {
        uint32_t x = (crc ^ data[i]) & 0xffu;
        int bit;
        for (bit = 0; bit < 8; ++bit)
            x = (x >> 1) ^
                (0x82f63b78u & (uint32_t)-(int32_t)(x & 1u));
        crc = (crc >> 8) ^ x;
    }
    return ~crc;
}

static inline int coli_cache_sync_file(FILE *file) {
    if (!file || fflush(file) != 0) return 0;
#ifdef _WIN32
    return _commit(_fileno(file)) == 0;
#else
    return fsync(fileno(file)) == 0;
#endif
}

static inline int coli_cache_replace_file(const char *tmp_path,
                                           const char *path) {
    if (!tmp_path || !path) return 0;
#ifdef _WIN32
    return MoveFileExA(tmp_path, path,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(tmp_path, path) == 0;
#endif
}

static inline uint64_t coli_cache_process_id(void) {
#ifdef _WIN32
    return (uint64_t)(unsigned)_getpid();
#else
    return (uint64_t)(unsigned long)getpid();
#endif
}

/* The caller supplies a per-process serial. Unique temporary names let
 * independent writers race safely; only complete files are renamed to final. */
static inline char *coli_cache_make_temp_path(const char *path,
                                              uint64_t serial) {
    size_t path_len;
    size_t capacity;
    char *tmp;
    int written;
    if (!path || !serial) return NULL;
    path_len = strlen(path);
    if (path_len > SIZE_MAX - 64u) return NULL;
    capacity = path_len + 64u;
    tmp = (char *)malloc(capacity);
    if (!tmp) return NULL;
    written = snprintf(tmp, capacity, "%s.tmp.%llu.%llu", path,
                       (unsigned long long)coli_cache_process_id(),
                       (unsigned long long)serial);
    if (written < 0 || (size_t)written >= capacity) {
        free(tmp);
        return NULL;
    }
    return tmp;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_CACHE_IO_H */

#include "expert_derived_cache.h"
#include "cache_io.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dirent.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

#define COLI_DERIVED_CACHE_VERSION 1u
#define COLI_DERIVED_CACHE_HEADER_BYTES 344u
#define COLI_DERIVED_CACHE_IDENTITY_OFFSET 88u
#define COLI_DERIVED_CACHE_CODEC_RAW 0u
#define COLI_DERIVED_CACHE_IO_CHUNK 65536u
#define COLI_DERIVED_CACHE_PREFIX "jit-"
#define COLI_DERIVED_CACHE_SUFFIX ".cdj"
#define COLI_DERIVED_CACHE_NAME_BYTES 72u

_Static_assert(COLI_DERIVED_CACHE_IDENTITY_BYTES == 256u,
               "derived cache identity framing changed");
_Static_assert(COLI_DERIVED_CACHE_HEADER_BYTES ==
                   COLI_DERIVED_CACHE_IDENTITY_OFFSET +
                       COLI_DERIVED_CACHE_IDENTITY_BYTES,
               "derived cache header framing changed");

typedef struct {
    uint64_t payload_bytes;
    uint64_t resident_bytes;
    uint64_t allocation_bytes;
    uint64_t alignment;
    uint64_t prepare_ns_avoided;
    uint32_t payload_crc;
    uint32_t transform_class;
} ColiDerivedHeaderInfo;

typedef struct {
    uint64_t bytes;
    uint64_t objects;
    uint64_t oldest_bytes;
    uint64_t oldest_stamp;
    char *oldest_path;
} ColiDerivedDirectoryScan;

static uint64_t coli_derived_now_ns(const ColiDerivedCache *cache) {
    return cache && cache->clock.now_ns
        ? cache->clock.now_ns(cache->clock.context) : 0;
}

static int coli_derived_fingerprint_nonzero(const uint8_t *fp) {
    if (!fp) return 0;
    for (size_t i = 0; i < COLI_DERIVED_CACHE_FINGERPRINT_BYTES; ++i)
        if (fp[i]) return 1;
    return 0;
}

static int coli_derived_alignment_valid(uint64_t alignment) {
    return alignment && (alignment & (alignment - 1u)) == 0;
}

static void coli_derived_encode_representation(
        uint8_t out[32], const ColiRepresentationId *rep) {
    memset(out, 0, 32);
    coli_cache_put_u16(out + 0, rep->math_format);
    coli_cache_put_u16(out + 2, rep->scale_format);
    coli_cache_put_u16(out + 4, rep->execution_layout);
    coli_cache_put_u16(out + 6, rep->execution_layout_abi);
    coli_cache_put_u16(out + 8, rep->kernel_abi);
    coli_cache_put_u16(out + 10, rep->reserved);
    coli_cache_put_u32(out + 12, rep->target_class);
    coli_cache_put_u32(out + 16, rep->group_size);
    coli_cache_put_u32(out + 20, rep->scale_block_rows);
    coli_cache_put_u32(out + 24, rep->scale_block_columns);
    coli_cache_put_u32(out + 28, rep->flags);
}

static void coli_derived_encode_identity(
        uint8_t out[COLI_DERIVED_CACHE_IDENTITY_BYTES],
        const ColiDerivedCacheIdentity *identity) {
    memset(out, 0, COLI_DERIVED_CACHE_IDENTITY_BYTES);
    memcpy(out + 0, identity->artifact_fingerprint,
           COLI_DERIVED_CACHE_FINGERPRINT_BYTES);
    coli_cache_put_u32(out + 32, (uint32_t)identity->logical_expert.layer);
    coli_cache_put_u32(out + 36, (uint32_t)identity->logical_expert.expert);
    coli_cache_put_u64(out + 40, identity->logical_record_id);
    memcpy(out + 48, identity->source_record_fingerprint,
           COLI_DERIVED_CACHE_FINGERPRINT_BYTES);
    coli_cache_put_u64(out + 80, identity->source_record_bytes);
    coli_cache_put_u32(out + 88, identity->source_record_crc32c);
    coli_cache_put_u32(out + 92, identity->source_record_abi);
    coli_derived_encode_representation(out + 96,
                                       &identity->source_representation);
    coli_derived_encode_representation(out + 128,
                                       &identity->target_representation);
    coli_cache_put_u32(out + 160, identity->transform_abi);
    coli_cache_put_u32(out + 164, (uint32_t)identity->transform_class);
    coli_cache_put_u32(out + 168, identity->target_hardware_class);
    coli_cache_put_u32(out + 172, identity->target_kernel_abi);
    coli_cache_put_u32(out + 176, identity->target_layout_abi);
    coli_cache_put_u32(out + 180, identity->runtime_byte_abi);
    coli_cache_put_u32(out + 184, identity->compiler_byte_abi);
    coli_cache_put_u64(out + 192, identity->quant_profile_id);
    coli_cache_put_u64(out + 200, identity->calibration_profile_id);
    coli_cache_put_u32(out + 208, identity->quality_policy_abi);
    coli_cache_put_u32(out + 212, identity->algorithm_abi);
    memcpy(out + 216, identity->calibration_fingerprint,
           COLI_DERIVED_CACHE_FINGERPRINT_BYTES);
}

int coli_derived_cache_identity_valid(const ColiDerivedCacheIdentity *identity) {
    if (!identity || identity->logical_expert.layer < 0 ||
        identity->logical_expert.expert < 0 ||
        !coli_derived_fingerprint_nonzero(identity->artifact_fingerprint) ||
        !coli_derived_fingerprint_nonzero(identity->source_record_fingerprint) ||
        !identity->source_record_bytes || !identity->source_record_abi ||
        !identity->transform_abi ||
        !coli_representation_known(&identity->source_representation) ||
        !coli_representation_known(&identity->target_representation) ||
        coli_representation_equal(&identity->source_representation,
                                  &identity->target_representation) ||
        identity->target_hardware_class !=
            identity->target_representation.target_class ||
        identity->target_kernel_abi !=
            (uint32_t)identity->target_representation.kernel_abi ||
        identity->target_layout_abi !=
            (uint32_t)identity->target_representation.execution_layout_abi)
        return 0;

    if (identity->transform_class == COLI_JIT_TRANSFORM_EXACT) {
        if (!coli_representation_exact_math_compatible(
                &identity->source_representation,
                &identity->target_representation) ||
            identity->quant_profile_id || identity->calibration_profile_id ||
            identity->quality_policy_abi || identity->algorithm_abi ||
            coli_derived_fingerprint_nonzero(
                identity->calibration_fingerprint))
            return 0;
    } else if (identity->transform_class == COLI_JIT_TRANSFORM_LOSSY) {
        if (!identity->quant_profile_id || !identity->calibration_profile_id ||
            !identity->quality_policy_abi || !identity->algorithm_abi ||
            !coli_derived_fingerprint_nonzero(
                identity->calibration_fingerprint))
            return 0;
    } else {
        return 0;
    }
    return 1;
}

int coli_derived_cache_identity_equal(const ColiDerivedCacheIdentity *a,
                                      const ColiDerivedCacheIdentity *b) {
    uint8_t ea[COLI_DERIVED_CACHE_IDENTITY_BYTES];
    uint8_t eb[COLI_DERIVED_CACHE_IDENTITY_BYTES];
    if (!coli_derived_cache_identity_valid(a) ||
        !coli_derived_cache_identity_valid(b))
        return 0;
    coli_derived_encode_identity(ea, a);
    coli_derived_encode_identity(eb, b);
    return memcmp(ea, eb, sizeof(ea)) == 0;
}

static void coli_derived_telemetry_init(ColiDerivedCacheTelemetry *telemetry) {
    atomic_init(&telemetry->lookup, 0);
    atomic_init(&telemetry->hit, 0);
    atomic_init(&telemetry->miss, 0);
    atomic_init(&telemetry->stale, 0);
    atomic_init(&telemetry->corrupt, 0);
    atomic_init(&telemetry->read_bytes, 0);
    atomic_init(&telemetry->read_ns, 0);
    atomic_init(&telemetry->write_bytes, 0);
    atomic_init(&telemetry->write_ns, 0);
    atomic_init(&telemetry->write_dropped, 0);
    atomic_init(&telemetry->objects, 0);
    atomic_init(&telemetry->disk_bytes, 0);
    atomic_init(&telemetry->pruned_bytes, 0);
    atomic_init(&telemetry->prepare_ns_avoided, 0);
}

static int coli_derived_directory_is_dir(const struct stat *st) {
    if (!st) return 0;
#ifdef _WIN32
    return (st->st_mode & _S_IFDIR) != 0;
#else
    return S_ISDIR(st->st_mode) != 0;
#endif
}

static int coli_derived_directory_ready(const char *path) {
    struct stat st;
    if (!path || !path[0]) return 0;
    if (stat(path, &st) == 0) return coli_derived_directory_is_dir(&st);
    if (errno != ENOENT) return 0;
#ifdef _WIN32
    if (_mkdir(path) != 0 && errno != EEXIST) return 0;
#else
    if (mkdir(path, 0700) != 0 && errno != EEXIST) return 0;
#endif
    return stat(path, &st) == 0 && coli_derived_directory_is_dir(&st);
}

int coli_derived_cache_init(
        ColiDerivedCache *cache,
        const ColiDerivedCacheConfig *config,
        const ColiJitTransformMemoryOps *memory,
        const ColiDerivedCachePayloadOps *payload,
        const ColiJitTransformClockOps *clock) {
    if (!cache || !config || !config->directory ||
        !config->directory[0] || !config->max_disk_bytes ||
        !config->max_object_bytes || !memory || !memory->allocate ||
        !memory->free || !payload || !payload->export_bytes ||
        !payload->import_bytes ||
        config->max_object_bytes > config->max_disk_bytes ||
        !coli_derived_directory_ready(config->directory))
        return -1;
    memset(cache, 0, sizeof(*cache));
    cache->config = *config;
    cache->config.enabled = config->enabled ? 1 : 0;
    cache->memory = *memory;
    cache->payload = *payload;
    if (clock) cache->clock = *clock;
    atomic_init(&cache->temp_serial, 0);
    coli_derived_telemetry_init(&cache->telemetry);
    return 0;
}

void coli_derived_cache_set_enabled(ColiDerivedCache *cache, int enabled) {
    if (cache) cache->config.enabled = enabled ? 1 : 0;
}

static uint64_t coli_derived_hash64(const uint8_t *data, size_t bytes,
                                    uint64_t seed) {
    uint64_t hash = seed;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int coli_derived_cache_object_path(
        const ColiDerivedCache *cache,
        const ColiDerivedCacheIdentity *identity,
        char *path,
        size_t path_capacity) {
    uint8_t encoded[COLI_DERIVED_CACHE_IDENTITY_BYTES];
    uint64_t h0, h1, h2, h3;
    const char *separator;
    int written;
    size_t directory_bytes;
    if (!cache || !path || !path_capacity ||
        !coli_derived_cache_identity_valid(identity) ||
        !cache->config.directory)
        return 0;
    coli_derived_encode_identity(encoded, identity);
    h0 = coli_derived_hash64(encoded, sizeof(encoded),
                             UINT64_C(1469598103934665603));
    h1 = coli_derived_hash64(encoded, sizeof(encoded),
                             UINT64_C(1099511628211));
    h2 = coli_derived_hash64(encoded, sizeof(encoded),
                             UINT64_C(0x9e3779b97f4a7c15));
    h3 = coli_derived_hash64(encoded, sizeof(encoded),
                             UINT64_C(0xd6e8feb86659fd93));
    directory_bytes = strlen(cache->config.directory);
    separator = directory_bytes &&
        (cache->config.directory[directory_bytes - 1] == '/' ||
         cache->config.directory[directory_bytes - 1] == '\\') ? "" : "/";
    written = snprintf(path, path_capacity,
        "%s%s%s%016llx%016llx%016llx%016llx%s",
        cache->config.directory, separator, COLI_DERIVED_CACHE_PREFIX,
        (unsigned long long)h0, (unsigned long long)h1,
        (unsigned long long)h2, (unsigned long long)h3,
        COLI_DERIVED_CACHE_SUFFIX);
    return written >= 0 && (size_t)written < path_capacity;
}

static char *coli_derived_object_path_alloc(
        const ColiDerivedCache *cache,
        const ColiDerivedCacheIdentity *identity) {
    size_t directory_bytes;
    size_t capacity;
    char *path;
    if (!cache || !cache->config.directory) return NULL;
    directory_bytes = strlen(cache->config.directory);
    if (directory_bytes > SIZE_MAX - 96u) return NULL;
    capacity = directory_bytes + 96u;
    path = (char *)malloc(capacity);
    if (!path) return NULL;
    if (!coli_derived_cache_object_path(cache, identity, path, capacity)) {
        free(path);
        return NULL;
    }
    return path;
}

static uint32_t coli_derived_header_crc(
        const uint8_t header[COLI_DERIVED_CACHE_HEADER_BYTES]) {
    static const uint8_t zero_crc[4] = {0, 0, 0, 0};
    uint32_t crc = coli_cache_crc32c_update(0, header, 72);
    crc = coli_cache_crc32c_update(crc, zero_crc, sizeof(zero_crc));
    return coli_cache_crc32c_update(
        crc, header + 76, COLI_DERIVED_CACHE_HEADER_BYTES - 76u);
}

static void coli_derived_build_header(
        uint8_t header[COLI_DERIVED_CACHE_HEADER_BYTES],
        const ColiDerivedCacheIdentity *identity,
        uint64_t resident_bytes,
        uint64_t allocation_bytes,
        uint64_t alignment,
        uint64_t prepare_ns_avoided,
        uint32_t payload_crc) {
    static const uint8_t magic[8] = {'C','O','L','I','J','I','T','1'};
    uint8_t encoded[COLI_DERIVED_CACHE_IDENTITY_BYTES];
    uint32_t identity_crc;
    memset(header, 0, COLI_DERIVED_CACHE_HEADER_BYTES);
    coli_derived_encode_identity(encoded, identity);
    identity_crc = coli_cache_crc32c_update(0, encoded, sizeof(encoded));
    memcpy(header + 0, magic, sizeof(magic));
    coli_cache_put_u32(header + 8, COLI_DERIVED_CACHE_VERSION);
    coli_cache_put_u32(header + 12, COLI_DERIVED_CACHE_HEADER_BYTES);
    coli_cache_put_u32(header + 16, COLI_DERIVED_CACHE_IDENTITY_BYTES);
    coli_cache_put_u32(header + 20, COLI_DERIVED_CACHE_CODEC_RAW);
    coli_cache_put_u64(header + 24, resident_bytes);
    coli_cache_put_u64(header + 32, resident_bytes);
    coli_cache_put_u64(header + 40, allocation_bytes);
    coli_cache_put_u64(header + 48, alignment);
    coli_cache_put_u64(header + 56, prepare_ns_avoided);
    coli_cache_put_u32(header + 64, identity_crc);
    coli_cache_put_u32(header + 68, payload_crc);
    coli_cache_put_u32(header + 76, (uint32_t)identity->transform_class);
    memcpy(header + COLI_DERIVED_CACHE_IDENTITY_OFFSET,
           encoded, sizeof(encoded));
    coli_cache_put_u32(header + 72, coli_derived_header_crc(header));
}

static int coli_derived_parse_header(
        const uint8_t header[COLI_DERIVED_CACHE_HEADER_BYTES],
        const ColiDerivedCacheIdentity *expected_identity,
        ColiDerivedHeaderInfo *info) {
    static const uint8_t magic[8] = {'C','O','L','I','J','I','T','1'};
    uint8_t encoded[COLI_DERIVED_CACHE_IDENTITY_BYTES];
    uint32_t identity_crc;
    uint32_t stored_identity_crc;
    uint32_t stored_header_crc;
    uint64_t payload_bytes;
    uint64_t resident_bytes;
    uint64_t allocation_bytes;
    uint64_t alignment;
    uint32_t transform_class;

    if (!header || !expected_identity || !info ||
        memcmp(header, magic, sizeof(magic)) != 0 ||
        coli_cache_get_u32(header + 8) != COLI_DERIVED_CACHE_VERSION ||
        coli_cache_get_u32(header + 12) != COLI_DERIVED_CACHE_HEADER_BYTES ||
        coli_cache_get_u32(header + 16) != COLI_DERIVED_CACHE_IDENTITY_BYTES ||
        coli_cache_get_u32(header + 20) != COLI_DERIVED_CACHE_CODEC_RAW)
        return -2;
    stored_header_crc = coli_cache_get_u32(header + 72);
    if (stored_header_crc != coli_derived_header_crc(header)) return -2;

    coli_derived_encode_identity(encoded, expected_identity);
    stored_identity_crc = coli_cache_get_u32(header + 64);
    identity_crc = coli_cache_crc32c_update(
        0, header + COLI_DERIVED_CACHE_IDENTITY_OFFSET,
        COLI_DERIVED_CACHE_IDENTITY_BYTES);
    if (stored_identity_crc != identity_crc) return -2;
    if (memcmp(header + COLI_DERIVED_CACHE_IDENTITY_OFFSET,
               encoded, sizeof(encoded)) != 0)
        return -1;

    payload_bytes = coli_cache_get_u64(header + 24);
    resident_bytes = coli_cache_get_u64(header + 32);
    allocation_bytes = coli_cache_get_u64(header + 40);
    alignment = coli_cache_get_u64(header + 48);
    transform_class = coli_cache_get_u32(header + 76);
    if (!payload_bytes || payload_bytes != resident_bytes ||
        allocation_bytes < resident_bytes ||
        !coli_derived_alignment_valid(alignment) ||
        transform_class != (uint32_t)expected_identity->transform_class)
        return -2;
    info->payload_bytes = payload_bytes;
    info->resident_bytes = resident_bytes;
    info->allocation_bytes = allocation_bytes;
    info->alignment = alignment;
    info->prepare_ns_avoided = coli_cache_get_u64(header + 56);
    info->payload_crc = coli_cache_get_u32(header + 68);
    info->transform_class = transform_class;
    return 1;
}

static int coli_derived_name_is_object(const char *name) {
    size_t length;
    if (!name) return 0;
    length = strlen(name);
    if (length != COLI_DERIVED_CACHE_NAME_BYTES ||
        memcmp(name, COLI_DERIVED_CACHE_PREFIX, 4) != 0 ||
        memcmp(name + length - 4, COLI_DERIVED_CACHE_SUFFIX, 4) != 0)
        return 0;
    for (size_t i = 4; i < length - 4; ++i) {
        char c = name[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return 0;
    }
    return 1;
}

static char *coli_derived_join_path(const char *directory, const char *name) {
    size_t d, n, extra;
    char *path;
    if (!directory || !name) return NULL;
    d = strlen(directory);
    n = strlen(name);
    extra = d && (directory[d - 1] == '/' || directory[d - 1] == '\\') ? 1u : 2u;
    if (d > SIZE_MAX - n - extra) return NULL;
    path = (char *)malloc(d + n + extra);
    if (!path) return NULL;
    memcpy(path, directory, d);
    if (extra == 2u) path[d++] = '/';
    memcpy(path + d, name, n + 1u);
    return path;
}

static void coli_derived_scan_clear(ColiDerivedDirectoryScan *scan) {
    if (!scan) return;
    free(scan->oldest_path);
    memset(scan, 0, sizeof(*scan));
}

static uint64_t coli_derived_sat_add(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

#ifdef _WIN32
static uint64_t coli_derived_filetime_u64(FILETIME value) {
    return ((uint64_t)value.dwHighDateTime << 32) | value.dwLowDateTime;
}

static int coli_derived_scan_directory(const ColiDerivedCache *cache,
                                       ColiDerivedDirectoryScan *scan) {
    char *pattern;
    size_t d;
    WIN32_FIND_DATAA data;
    HANDLE handle;
    if (!cache || !scan) return 0;
    memset(scan, 0, sizeof(*scan));
    d = strlen(cache->config.directory);
    if (d > SIZE_MAX - 8u) return 0;
    pattern = (char *)malloc(d + 8u);
    if (!pattern) return 0;
    snprintf(pattern, d + 8u, "%s%s*", cache->config.directory,
             d && (cache->config.directory[d - 1] == '/' ||
                   cache->config.directory[d - 1] == '\\') ? "" : "/");
    handle = FindFirstFileA(pattern, &data);
    free(pattern);
    if (handle == INVALID_HANDLE_VALUE) return 1;
    do {
        uint64_t size, stamp;
        char *path;
        if (!coli_derived_name_is_object(data.cFileName) ||
            (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            continue;
        size = ((uint64_t)data.nFileSizeHigh << 32) | data.nFileSizeLow;
        stamp = coli_derived_filetime_u64(data.ftLastWriteTime);
        scan->bytes = coli_derived_sat_add(scan->bytes, size);
        scan->objects = coli_derived_sat_add(scan->objects, 1);
        if (scan->oldest_path && stamp > scan->oldest_stamp) continue;
        path = coli_derived_join_path(cache->config.directory, data.cFileName);
        if (!path) { FindClose(handle); coli_derived_scan_clear(scan); return 0; }
        if (!scan->oldest_path || stamp < scan->oldest_stamp ||
            strcmp(path, scan->oldest_path) < 0) {
            free(scan->oldest_path);
            scan->oldest_path = path;
            scan->oldest_stamp = stamp;
            scan->oldest_bytes = size;
        } else {
            free(path);
        }
    } while (FindNextFileA(handle, &data));
    FindClose(handle);
    return 1;
}

static int coli_derived_free_bytes(const char *directory, uint64_t *bytes) {
    ULARGE_INTEGER available;
    if (!directory || !bytes ||
        !GetDiskFreeSpaceExA(directory, &available, NULL, NULL))
        return 0;
    *bytes = (uint64_t)available.QuadPart;
    return 1;
}
#else
static int coli_derived_scan_directory(const ColiDerivedCache *cache,
                                       ColiDerivedDirectoryScan *scan) {
    DIR *dir;
    struct dirent *entry;
    if (!cache || !scan) return 0;
    memset(scan, 0, sizeof(*scan));
    dir = opendir(cache->config.directory);
    if (!dir) return 0;
    while ((entry = readdir(dir)) != NULL) {
        struct stat st;
        uint64_t size, stamp;
        char *path;
        if (!coli_derived_name_is_object(entry->d_name)) continue;
        path = coli_derived_join_path(cache->config.directory, entry->d_name);
        if (!path) { closedir(dir); coli_derived_scan_clear(scan); return 0; }
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
            free(path);
            continue;
        }
        size = st.st_size < 0 ? 0 : (uint64_t)st.st_size;
        stamp = st.st_mtime < 0 ? 0 : (uint64_t)st.st_mtime;
        scan->bytes = coli_derived_sat_add(scan->bytes, size);
        scan->objects = coli_derived_sat_add(scan->objects, 1);
        if (!scan->oldest_path || stamp < scan->oldest_stamp ||
            (stamp == scan->oldest_stamp &&
             strcmp(path, scan->oldest_path) < 0)) {
            free(scan->oldest_path);
            scan->oldest_path = path;
            scan->oldest_stamp = stamp;
            scan->oldest_bytes = size;
        } else {
            free(path);
        }
    }
    closedir(dir);
    return 1;
}

static int coli_derived_free_bytes(const char *directory, uint64_t *bytes) {
    struct statvfs vfs;
    uint64_t blocks;
    if (!directory || !bytes || statvfs(directory, &vfs) != 0) return 0;
    blocks = (uint64_t)vfs.f_bavail;
    if (vfs.f_frsize && blocks > UINT64_MAX / (uint64_t)vfs.f_frsize)
        *bytes = UINT64_MAX;
    else
        *bytes = blocks * (uint64_t)vfs.f_frsize;
    return 1;
}
#endif

int coli_derived_cache_prune(ColiDerivedCache *cache, uint64_t bytes_needed) {
    if (!cache || !cache->config.directory ||
        bytes_needed > cache->config.max_disk_bytes)
        return 0;
    for (;;) {
        ColiDerivedDirectoryScan scan;
        uint64_t free_bytes = 0;
        int budget_ok;
        int free_ok = 1;
        if (!coli_derived_scan_directory(cache, &scan)) return 0;
        atomic_store_explicit(&cache->telemetry.objects, scan.objects,
                              memory_order_release);
        atomic_store_explicit(&cache->telemetry.disk_bytes, scan.bytes,
                              memory_order_release);
        budget_ok = scan.bytes <= cache->config.max_disk_bytes &&
            bytes_needed <= cache->config.max_disk_bytes - scan.bytes;
        if (cache->config.min_free_bytes) {
            free_ok = coli_derived_free_bytes(cache->config.directory,
                                              &free_bytes) &&
                free_bytes >= cache->config.min_free_bytes &&
                bytes_needed <= free_bytes - cache->config.min_free_bytes;
        }
        if (budget_ok && free_ok) {
            coli_derived_scan_clear(&scan);
            return 1;
        }
        if (!scan.oldest_path) {
            coli_derived_scan_clear(&scan);
            return 0;
        }
        if (remove(scan.oldest_path) != 0) {
            coli_derived_scan_clear(&scan);
            return 0;
        }
        atomic_fetch_add_explicit(&cache->telemetry.pruned_bytes,
                                  scan.oldest_bytes, memory_order_acq_rel);
        coli_derived_scan_clear(&scan);
    }
}

static int coli_derived_verify_payload(FILE *file, uint64_t payload_bytes,
                                       uint32_t expected_crc) {
    uint8_t *chunk;
    uint32_t crc = 0;
    uint64_t offset = 0;
    int trailing;
    if (!file || !payload_bytes) return 0;
    chunk = (uint8_t *)malloc(COLI_DERIVED_CACHE_IO_CHUNK);
    if (!chunk) return 0;
    while (offset < payload_bytes) {
        uint64_t remaining = payload_bytes - offset;
        size_t bytes = remaining < COLI_DERIVED_CACHE_IO_CHUNK
            ? (size_t)remaining : (size_t)COLI_DERIVED_CACHE_IO_CHUNK;
        if (fread(chunk, 1, bytes, file) != bytes) {
            free(chunk);
            return 0;
        }
        crc = coli_cache_crc32c_update(crc, chunk, bytes);
        offset += bytes;
    }
    trailing = fgetc(file);
    free(chunk);
    return trailing == EOF && !ferror(file) && crc == expected_crc;
}

/* 1 valid, 0 missing/I/O, -1 stale identity, -2 corrupt/framing. */
static int coli_derived_verify_file(
        const char *path,
        const ColiDerivedCacheIdentity *expected_identity,
        uint64_t max_object_bytes,
        ColiDerivedHeaderInfo *info_out) {
    uint8_t header[COLI_DERIVED_CACHE_HEADER_BYTES];
    ColiDerivedHeaderInfo info;
    FILE *file;
    int parsed;
    if (!path || !expected_identity) return 0;
    file = fopen(path, "rb");
    if (!file) return 0;
    if (fread(header, 1, sizeof(header), file) != sizeof(header)) {
        fclose(file);
        return -2;
    }
    parsed = coli_derived_parse_header(header, expected_identity, &info);
    if (parsed != 1) {
        fclose(file);
        return parsed;
    }
    if (info.payload_bytes > max_object_bytes ||
        info.allocation_bytes > max_object_bytes ||
        !coli_derived_verify_payload(file, info.payload_bytes,
                                     info.payload_crc)) {
        fclose(file);
        return -2;
    }
    if (fclose(file) != 0) return -2;
    if (info_out) *info_out = info;
    return 1;
}

static void coli_derived_note_miss(ColiDerivedCache *cache, int status) {
    atomic_fetch_add_explicit(&cache->telemetry.miss, 1,
                              memory_order_acq_rel);
    if (status == -1)
        atomic_fetch_add_explicit(&cache->telemetry.stale, 1,
                                  memory_order_acq_rel);
    else if (status == -2)
        atomic_fetch_add_explicit(&cache->telemetry.corrupt, 1,
                                  memory_order_acq_rel);
}

int coli_derived_cache_store(
        ColiDerivedCache *cache,
        const ColiDerivedCacheIdentity *identity,
        const ColiExpertResidencyLease *target_lease,
        uint64_t resident_alignment,
        uint64_t prepare_ns_avoided) {
    uint8_t header[COLI_DERIVED_CACHE_HEADER_BYTES];
    uint8_t *chunk = NULL;
    char *path = NULL;
    char *tmp_path = NULL;
    FILE *file = NULL;
    ColiExpertResidentView view;
    uint32_t payload_crc = 0;
    uint64_t offset = 0;
    uint64_t object_bytes;
    uint64_t started;
    uint64_t finished;
    uint64_t serial;
    int ok = 0;

    if (!cache || !cache->config.enabled ||
        !coli_derived_cache_identity_valid(identity) || !target_lease ||
        !coli_expert_residency_lease_valid(target_lease) ||
        !coli_expert_key_equal(target_lease->key, identity->logical_expert) ||
        !coli_derived_alignment_valid(resident_alignment) ||
        coli_expert_residency_lease_view(target_lease, &view) != 0 ||
        !view.physical || !view.resident_bytes ||
        view.allocation_bytes < view.resident_bytes ||
        view.allocation_bytes > cache->config.max_object_bytes ||
        !coli_representation_equal(&view.representation,
                                   &identity->target_representation) ||
        coli_expert_residency_preferred_variant(target_lease->entry) !=
            (int)target_lease->variant_id ||
        view.resident_bytes > cache->config.max_object_bytes ||
        view.resident_bytes > UINT64_MAX - COLI_DERIVED_CACHE_HEADER_BYTES) {
        if (cache)
            atomic_fetch_add_explicit(&cache->telemetry.write_dropped, 1,
                                      memory_order_acq_rel);
        return 0;
    }
    object_bytes = view.resident_bytes + COLI_DERIVED_CACHE_HEADER_BYTES;
    if (object_bytes > cache->config.max_disk_bytes ||
        !coli_derived_cache_prune(cache, object_bytes)) {
        atomic_fetch_add_explicit(&cache->telemetry.write_dropped, 1,
                                  memory_order_acq_rel);
        return 0;
    }

    path = coli_derived_object_path_alloc(cache, identity);
    if (!path) goto done;
    serial = atomic_fetch_add_explicit(&cache->temp_serial, 1,
                                       memory_order_acq_rel) + 1;
    if (!serial) goto done;
    tmp_path = coli_cache_make_temp_path(path, serial);
    if (!tmp_path) goto done;
    chunk = (uint8_t *)malloc(COLI_DERIVED_CACHE_IO_CHUNK);
    if (!chunk) goto done;

    started = coli_derived_now_ns(cache);
    file = fopen(tmp_path, "wb");
    if (!file) goto done;
    coli_derived_build_header(header, identity, view.resident_bytes,
                              view.allocation_bytes, resident_alignment,
                              prepare_ns_avoided, 0);
    if (fwrite(header, 1, sizeof(header), file) != sizeof(header)) goto done;

    while (offset < view.resident_bytes) {
        uint64_t remaining = view.resident_bytes - offset;
        size_t bytes = remaining < COLI_DERIVED_CACHE_IO_CHUNK
            ? (size_t)remaining : (size_t)COLI_DERIVED_CACHE_IO_CHUNK;
        if (cache->payload.export_bytes(cache->payload.context, &view,
                                        offset, chunk, bytes) != 0 ||
            fwrite(chunk, 1, bytes, file) != bytes)
            goto done;
        payload_crc = coli_cache_crc32c_update(payload_crc, chunk, bytes);
        offset += bytes;
    }

    coli_derived_build_header(header, identity, view.resident_bytes,
                              view.allocation_bytes, resident_alignment,
                              prepare_ns_avoided, payload_crc);
    if (fseek(file, 0, SEEK_SET) != 0 ||
        fwrite(header, 1, sizeof(header), file) != sizeof(header) ||
        !coli_cache_sync_file(file))
        goto done;
    if (fclose(file) != 0) { file = NULL; goto done; }
    file = NULL;

    if (coli_derived_verify_file(tmp_path, identity,
                                 cache->config.max_object_bytes, NULL) != 1 ||
        !coli_cache_replace_file(tmp_path, path))
        goto done;

    ok = 1;
    finished = coli_derived_now_ns(cache);
    atomic_fetch_add_explicit(&cache->telemetry.write_bytes, object_bytes,
                              memory_order_acq_rel);
    if (started && finished >= started)
        atomic_fetch_add_explicit(&cache->telemetry.write_ns,
                                  finished - started, memory_order_acq_rel);
    (void)coli_derived_cache_prune(cache, 0);

done:
    if (file) (void)fclose(file);
    if (!ok && tmp_path) (void)remove(tmp_path);
    if (!ok)
        atomic_fetch_add_explicit(&cache->telemetry.write_dropped, 1,
                                  memory_order_acq_rel);
    free(chunk);
    free(tmp_path);
    free(path);
    return ok;
}

static int coli_derived_fill_existing_info(
        ColiExpertResidencyEntry *entry, uint32_t variant_id,
        ColiDerivedCacheLoadInfo *info) {
    ColiExpertResidentVariantInfo resident;
    if (!entry || !info ||
        coli_expert_residency_query_variant(entry, variant_id, &resident) != 0 ||
        resident.state != COLI_EXPERT_RESIDENCY_RESIDENT)
        return 0;
    info->variant_id = variant_id;
    info->generation = resident.generation;
    info->resident_bytes = resident.resident_bytes;
    info->allocation_bytes = resident.allocation_bytes;
    return 1;
}

int coli_derived_cache_load_variant(
        ColiDerivedCache *cache,
        const ColiDerivedCacheIdentity *expected_identity,
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *resident_budget,
        unsigned target_tier_mask,
        ColiDerivedCacheLoadInfo *info) {
    uint8_t header[COLI_DERIVED_CACHE_HEADER_BYTES];
    uint8_t *chunk = NULL;
    char *path = NULL;
    FILE *file = NULL;
    ColiDerivedHeaderInfo header_info;
    uint32_t variant_id = COLI_EXPERT_VARIANT_NONE;
    uint64_t generation = 0;
    uint32_t crc = 0;
    uint64_t offset = 0;
    void *output = NULL;
    uint64_t started;
    uint64_t finished;
    int parsed;
    int result = 0;
    int reservation_owned = 0;

    if (info) memset(info, 0, sizeof(*info));
    if (!cache || !cache->config.enabled ||
        !coli_derived_cache_identity_valid(expected_identity) || !entry ||
        !coli_expert_key_equal(entry->key, expected_identity->logical_expert) ||
        !resident_budget || !target_tier_mask || !info)
        return 0;

    atomic_fetch_add_explicit(&cache->telemetry.lookup, 1,
                              memory_order_acq_rel);
    started = coli_derived_now_ns(cache);
    path = coli_derived_object_path_alloc(cache, expected_identity);
    if (!path) { coli_derived_note_miss(cache, 0); goto done; }
    file = fopen(path, "rb");
    if (!file) { coli_derived_note_miss(cache, 0); goto done; }
    if (fread(header, 1, sizeof(header), file) != sizeof(header)) {
        coli_derived_note_miss(cache, -2);
        goto done;
    }
    parsed = coli_derived_parse_header(header, expected_identity, &header_info);
    if (parsed != 1 || header_info.payload_bytes > cache->config.max_object_bytes ||
        header_info.allocation_bytes > cache->config.max_object_bytes) {
        coli_derived_note_miss(cache, parsed == -1 ? -1 : -2);
        goto done;
    }

    ColiExpertRequestResult reserved = coli_expert_residency_reserve_variant(
        entry, resident_budget, &expected_identity->target_representation,
        target_tier_mask, header_info.allocation_bytes,
        &variant_id, &generation);
    if (reserved == COLI_EXPERT_REQUEST_HIT) {
        if (!coli_derived_verify_payload(file, header_info.payload_bytes,
                                         header_info.payload_crc) ||
            !coli_derived_fill_existing_info(entry, variant_id, info)) {
            coli_derived_note_miss(cache, -2);
            goto done;
        }
        info->resident_alignment = header_info.alignment;
        result = 1;
        goto hit;
    }
    if (reserved != COLI_EXPERT_REQUEST_LOAD_OWNER) {
        coli_derived_note_miss(cache, 0);
        goto done;
    }
    reservation_owned = 1;

    output = cache->memory.allocate(
        cache->memory.context, COLI_JIT_MEMORY_OUTPUT,
        header_info.allocation_bytes, header_info.alignment);
    if (!output) { coli_derived_note_miss(cache, 0); goto done; }
    chunk = (uint8_t *)malloc(COLI_DERIVED_CACHE_IO_CHUNK);
    if (!chunk) { coli_derived_note_miss(cache, 0); goto done; }

    while (offset < header_info.payload_bytes) {
        uint64_t remaining = header_info.payload_bytes - offset;
        size_t bytes = remaining < COLI_DERIVED_CACHE_IO_CHUNK
            ? (size_t)remaining : (size_t)COLI_DERIVED_CACHE_IO_CHUNK;
        if (fread(chunk, 1, bytes, file) != bytes ||
            cache->payload.import_bytes(cache->payload.context, output,
                                        offset, chunk, bytes) != 0) {
            coli_derived_note_miss(cache, -2);
            goto done;
        }
        crc = coli_cache_crc32c_update(crc, chunk, bytes);
        offset += bytes;
    }
    if (fgetc(file) != EOF || ferror(file) || crc != header_info.payload_crc) {
        coli_derived_note_miss(cache, -2);
        goto done;
    }
    if (fclose(file) != 0) {
        file = NULL;
        coli_derived_note_miss(cache, -2);
        goto done;
    }
    file = NULL;

    if (coli_expert_residency_publish_variant(
            entry, resident_budget, variant_id, generation,
            header_info.resident_bytes, output) != 0) {
        coli_derived_note_miss(cache, 0);
        goto done;
    }
    output = NULL;
    reservation_owned = 0;
    info->variant_id = variant_id;
    info->generation = generation;
    info->resident_bytes = header_info.resident_bytes;
    info->allocation_bytes = header_info.allocation_bytes;
    info->resident_alignment = header_info.alignment;
    result = 1;

hit:
    atomic_fetch_add_explicit(&cache->telemetry.hit, 1,
                              memory_order_acq_rel);
    atomic_fetch_add_explicit(&cache->telemetry.read_bytes,
                              COLI_DERIVED_CACHE_HEADER_BYTES +
                                  header_info.payload_bytes,
                              memory_order_acq_rel);
    atomic_fetch_add_explicit(&cache->telemetry.prepare_ns_avoided,
                              header_info.prepare_ns_avoided,
                              memory_order_acq_rel);
    finished = coli_derived_now_ns(cache);
    if (started && finished >= started)
        atomic_fetch_add_explicit(&cache->telemetry.read_ns,
                                  finished - started, memory_order_acq_rel);

done:
    if (file) (void)fclose(file);
    if (!result && output)
        cache->memory.free(cache->memory.context, COLI_JIT_MEMORY_OUTPUT,
                           output, header_info.allocation_bytes);
    if (!result && reservation_owned)
        (void)coli_expert_residency_fail_variant(
            entry, resident_budget, variant_id, generation);
    free(chunk);
    free(path);
    return result;
}

void coli_derived_cache_telemetry_snapshot(
        const ColiDerivedCache *cache,
        ColiDerivedCacheTelemetrySnapshot *snapshot) {
    if (!cache || !snapshot) return;
    snapshot->lookup = atomic_load_explicit(&cache->telemetry.lookup,
                                            memory_order_acquire);
    snapshot->hit = atomic_load_explicit(&cache->telemetry.hit,
                                         memory_order_acquire);
    snapshot->miss = atomic_load_explicit(&cache->telemetry.miss,
                                          memory_order_acquire);
    snapshot->stale = atomic_load_explicit(&cache->telemetry.stale,
                                           memory_order_acquire);
    snapshot->corrupt = atomic_load_explicit(&cache->telemetry.corrupt,
                                             memory_order_acquire);
    snapshot->read_bytes = atomic_load_explicit(&cache->telemetry.read_bytes,
                                                memory_order_acquire);
    snapshot->read_ns = atomic_load_explicit(&cache->telemetry.read_ns,
                                             memory_order_acquire);
    snapshot->write_bytes = atomic_load_explicit(&cache->telemetry.write_bytes,
                                                 memory_order_acquire);
    snapshot->write_ns = atomic_load_explicit(&cache->telemetry.write_ns,
                                              memory_order_acquire);
    snapshot->write_dropped = atomic_load_explicit(
        &cache->telemetry.write_dropped, memory_order_acquire);
    snapshot->objects = atomic_load_explicit(&cache->telemetry.objects,
                                             memory_order_acquire);
    snapshot->disk_bytes = atomic_load_explicit(&cache->telemetry.disk_bytes,
                                                memory_order_acquire);
    snapshot->pruned_bytes = atomic_load_explicit(
        &cache->telemetry.pruned_bytes, memory_order_acquire);
    snapshot->prepare_ns_avoided = atomic_load_explicit(
        &cache->telemetry.prepare_ns_avoided, memory_order_acquire);
}

#include <stddef.h>
#include <stdint.h>
#define COLI_PREFIX_DISK_FINGERPRINT_BYTES 32
#define COLI_PREFIX_DISK_PATH_MAX 2048
#define COLI_PREFIX_DISK_VERSION 1u
typedef struct ColiPrefixDiskCache {
    char directory[COLI_PREFIX_DISK_PATH_MAX];
    uint8_t model_fingerprint[COLI_PREFIX_DISK_FINGERPRINT_BYTES];
    uint64_t layout_fingerprint;
    uint32_t engine_id;
    uint32_t state_abi;
    size_t budget_bytes;
    size_t min_free_bytes;
    int min_tokens;
    int enabled;
    int log;
    uint64_t lookups, hits, stores, evictions, corruptions, read_bytes, write_bytes, resident_bytes;
} ColiPrefixDiskCache;
typedef struct ColiPrefixDiskHit { int *tokens; int token_count; unsigned char *payload; size_t payload_bytes; } ColiPrefixDiskHit;
typedef struct ColiPrefixDiskStats {
    uint64_t lookups, hits, stores, evictions, corruptions, read_bytes, write_bytes, resident_bytes;
    size_t budget_bytes, min_free_bytes;
} ColiPrefixDiskStats;
typedef struct ColiPrefixDiskPart { const void *data; size_t bytes; } ColiPrefixDiskPart;
typedef int (*ColiPrefixDiskSinkFn)(void *sink_data,const void *data,size_t bytes);

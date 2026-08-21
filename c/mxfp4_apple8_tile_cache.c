#include "mxfp4_apple8_tile_cache.h"

#include "backend_metal_tile.h"
#include "expert_derived_cache.h"
#include "mxfp4_apple8_tile.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define APPLE8_DERIVED_DEFAULT_DISK_GIB UINT64_C(96)
#define APPLE8_DERIVED_DEFAULT_MIN_FREE_GIB UINT64_C(4)
#define APPLE8_DERIVED_MAX_OBJECT_BYTES (UINT64_C(32) * 1024u * 1024u)
#define APPLE8_SOURCE_RECORD_ABI \
    (((uint32_t)COLI_CSF_VERSION_MAJOR << 16) | (uint32_t)COLI_CSF_VERSION_MINOR)
#define APPLE8_RUNTIME_BYTE_ABI ((uint32_t)COLI_MXFP4_APPLE8_TILE_EXPERT_VERSION)

#define APPLE8_ARENA_MAX_LAYERS 256
#define APPLE8_ARENA_INDEX_BYTES 80u
#define APPLE8_ARENA_INDEX_MAGIC UINT32_C(0x58493841) /* "A8IX" LE */
#define APPLE8_ARENA_INDEX_VERSION 1u
#define APPLE8_ARENA_INDEX_COMMITTED 1u

typedef struct {
    int32_t expert;
    uint8_t identity_key[COLI_DERIVED_CACHE_FINGERPRINT_BYTES];
    uint64_t offset;
    uint64_t bytes;
    uint64_t prepare_ns;
    uint32_t payload_crc;
    int verified;
} Apple8ArenaEntry;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int loaded;
    int data_fd;
    int index_fd;
    int wrote_in_process;
    uint64_t data_size;
    void *mapping;
    size_t mapping_bytes;
    unsigned mapping_pins;
    Apple8ArenaEntry *entries;
    size_t entry_count;
    size_t entry_capacity;
    char data_path[PATH_MAX];
    char index_path[PATH_MAX];
} Apple8ArenaLayer;

static pthread_once_t g_cache_once = PTHREAD_ONCE_INIT;
static pthread_once_t g_arena_once = PTHREAD_ONCE_INIT;
static ColiDerivedCache g_cache;
static int g_cache_enabled;
static int g_cache_ready;
static char g_cache_directory[PATH_MAX];
static Apple8ArenaLayer g_arena_layers[APPLE8_ARENA_MAX_LAYERS];

static atomic_uint_fast64_t g_lookup = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_hit = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_miss = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_stale = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_corrupt = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_read_bytes = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_read_ns = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_write_bytes = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_write_ns = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_write_dropped = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_prepare_ns_avoided = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_cold_prepares = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_cold_prepare_ns = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_installs = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_install_failures = ATOMIC_VAR_INIT(0);

static uint64_t now_ns(void *opaque) {
    (void)opaque;
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static void *cache_allocate(void *opaque, ColiJitMemoryPurpose purpose,
                            uint64_t bytes, uint64_t alignment) {
    (void)opaque;
    (void)purpose;
    if (!bytes || bytes > SIZE_MAX) return NULL;
    size_t a = alignment ? (size_t)alignment : sizeof(void *);
    if (a < sizeof(void *)) a = sizeof(void *);
    if ((a & (a - 1u)) != 0) return NULL;
    void *memory = NULL;
    if (posix_memalign(&memory, a, (size_t)bytes) != 0) return NULL;
    return memory;
}

static void cache_free(void *opaque, ColiJitMemoryPurpose purpose,
                       void *memory, uint64_t bytes) {
    (void)opaque;
    (void)purpose;
    (void)bytes;
    free(memory);
}

static int cache_export(void *opaque, const ColiExpertResidentView *source,
                        uint64_t offset, void *host_bytes, size_t bytes) {
    (void)opaque;
    if (!source || !source->physical || !host_bytes ||
        offset > source->resident_bytes ||
        bytes > source->resident_bytes - offset)
        return -1;
    memcpy(host_bytes, (const uint8_t *)source->physical + offset, bytes);
    return 0;
}

static int cache_import(void *opaque, void *destination_physical,
                        uint64_t offset, const void *host_bytes, size_t bytes) {
    (void)opaque;
    if (!destination_physical || !host_bytes) return -1;
    memcpy((uint8_t *)destination_physical + offset, host_bytes, bytes);
    return 0;
}

static uint64_t env_gib(const char *name, uint64_t fallback,
                        uint64_t minimum, uint64_t maximum) {
    const char *text = getenv(name);
    if (!text || !*text) return fallback;
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || !end || *end) return fallback;
    uint64_t value = (uint64_t)parsed;
    if (value < minimum) value = minimum;
    if (value > maximum) value = maximum;
    return value;
}

static void arena_init_once(void) {
    for (int i = 0; i < APPLE8_ARENA_MAX_LAYERS; ++i) {
        Apple8ArenaLayer *layer = &g_arena_layers[i];
        (void)pthread_mutex_init(&layer->mutex, NULL);
        (void)pthread_cond_init(&layer->cond, NULL);
        layer->data_fd = -1;
        layer->index_fd = -1;
    }
}

static void cache_init_once(void) {
    const char *enabled = getenv("V4_METAL_TILE_DERIVED_CACHE");
    g_cache_enabled = enabled && *enabled && atoi(enabled) != 0;
    if (!g_cache_enabled) return;

    const char *directory = getenv("V4_METAL_TILE_DERIVED_CACHE_DIR");
    if (!directory || !*directory) directory = "./.colibri-derived-apple8";
    int written = snprintf(g_cache_directory, sizeof(g_cache_directory),
                           "%s", directory);
    if (written < 0 || (size_t)written >= sizeof(g_cache_directory)) return;

    uint64_t disk_gib = env_gib("V4_METAL_TILE_DERIVED_CACHE_MAX_GB",
                                APPLE8_DERIVED_DEFAULT_DISK_GIB, 1, 1024);
    uint64_t free_gib = env_gib("V4_METAL_TILE_DERIVED_CACHE_MIN_FREE_GB",
                                APPLE8_DERIVED_DEFAULT_MIN_FREE_GIB, 0, 1024);
    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    if (disk_gib > UINT64_MAX / gib || free_gib > UINT64_MAX / gib) return;

    ColiDerivedCacheConfig config = {
        .directory = g_cache_directory,
        .max_disk_bytes = disk_gib * gib,
        .max_object_bytes = APPLE8_DERIVED_MAX_OBJECT_BYTES,
        .min_free_bytes = free_gib * gib,
        .enabled = 1,
    };
    ColiJitTransformMemoryOps memory = {
        .context = NULL,
        .allocate = cache_allocate,
        .free = cache_free,
    };
    ColiDerivedCachePayloadOps payload = {
        .context = NULL,
        .export_bytes = cache_export,
        .import_bytes = cache_import,
    };
    ColiJitTransformClockOps clock = {
        .context = NULL,
        .now_ns = now_ns,
    };

    /* #137 remains authoritative for correctness identity. The arena backend
     * uses coli_derived_cache_object_path() as the normalized 256-bit identity
     * key, while physical payloads are raw concatenated per-layer records. */
    g_cache_ready =
        coli_derived_cache_init(&g_cache, &config, &memory, &payload, &clock) == 0;
    if (g_cache_ready) pthread_once(&g_arena_once, arena_init_once);
    if (!g_cache_ready)
        fprintf(stderr,
                "v4_metal tile_derived_cache=disabled reason=init-failed dir=%s\n",
                g_cache_directory);
}

int coli_mxfp4_apple8_derived_cache_enabled(void) {
    pthread_once(&g_cache_once, cache_init_once);
    return g_cache_enabled && g_cache_ready;
}

static uint16_t get_u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t get_u64(const uint8_t *p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= (uint64_t)p[i] << (i * 8u);
    return value;
}

static void put_u16(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t *p, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        p[i] = (uint8_t)value;
        value >>= 8;
    }
}

static void put_u64(uint8_t *p, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        p[i] = (uint8_t)value;
        value >>= 8;
    }
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int identity_key(const ColiDerivedCacheIdentity *identity,
                        uint8_t key[COLI_DERIVED_CACHE_FINGERPRINT_BYTES]) {
    char path[PATH_MAX];
    if (!identity || !key ||
        !coli_derived_cache_object_path(&g_cache, identity,
                                        path, sizeof(path)))
        return 0;
    const char *name = strrchr(path, '/');
    name = name ? name + 1 : path;
    if (strncmp(name, "jit-", 4) != 0) return 0;
    name += 4;
    for (size_t i = 0; i < COLI_DERIVED_CACHE_FINGERPRINT_BYTES; ++i) {
        int hi = hex_value(name[i * 2u]);
        int lo = hex_value(name[i * 2u + 1u]);
        if (hi < 0 || lo < 0) return 0;
        key[i] = (uint8_t)((hi << 4) | lo);
    }
    return strncmp(name + COLI_DERIVED_CACHE_FINGERPRINT_BYTES * 2u,
                   ".cdj", 4) == 0;
}

static int pread_all(int fd, void *buffer, size_t bytes, uint64_t offset) {
    uint8_t *p = (uint8_t *)buffer;
    while (bytes) {
        ssize_t got = pread(fd, p, bytes, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 0;
        p += (size_t)got;
        bytes -= (size_t)got;
        offset += (uint64_t)got;
    }
    return 1;
}

static int pwrite_all(int fd, const void *buffer, size_t bytes, uint64_t offset) {
    const uint8_t *p = (const uint8_t *)buffer;
    while (bytes) {
        ssize_t put = pwrite(fd, p, bytes, (off_t)offset);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) return 0;
        p += (size_t)put;
        bytes -= (size_t)put;
        offset += (uint64_t)put;
    }
    return 1;
}

static Apple8ArenaEntry *arena_find_entry(Apple8ArenaLayer *layer,
                                          int32_t expert) {
    if (!layer) return NULL;
    for (size_t i = 0; i < layer->entry_count; ++i)
        if (layer->entries[i].expert == expert) return &layer->entries[i];
    return NULL;
}

static Apple8ArenaEntry *arena_upsert_entry(Apple8ArenaLayer *layer,
                                            const Apple8ArenaEntry *entry) {
    if (!layer || !entry) return NULL;
    Apple8ArenaEntry *existing = arena_find_entry(layer, entry->expert);
    if (existing) {
        *existing = *entry;
        return existing;
    }
    if (layer->entry_count == layer->entry_capacity) {
        size_t next = layer->entry_capacity ? layer->entry_capacity * 2u : 64u;
        if (next < layer->entry_capacity ||
            next > SIZE_MAX / sizeof(*layer->entries))
            return NULL;
        void *memory = realloc(layer->entries, next * sizeof(*layer->entries));
        if (!memory) return NULL;
        layer->entries = (Apple8ArenaEntry *)memory;
        layer->entry_capacity = next;
    }
    layer->entries[layer->entry_count] = *entry;
    return &layer->entries[layer->entry_count++];
}

static int arena_decode_index(const uint8_t bytes[APPLE8_ARENA_INDEX_BYTES],
                              Apple8ArenaEntry *entry) {
    if (!bytes || !entry ||
        get_u32(bytes + 0) != APPLE8_ARENA_INDEX_MAGIC ||
        get_u16(bytes + 4) != APPLE8_ARENA_INDEX_VERSION ||
        get_u16(bytes + 6) != APPLE8_ARENA_INDEX_COMMITTED ||
        get_u32(bytes + 12) != 0 ||
        coli_crc32c(bytes, APPLE8_ARENA_INDEX_BYTES - 4u) !=
            get_u32(bytes + APPLE8_ARENA_INDEX_BYTES - 4u))
        return 0;
    memset(entry, 0, sizeof(*entry));
    entry->expert = (int32_t)get_u32(bytes + 8);
    entry->offset = get_u64(bytes + 16);
    entry->bytes = get_u64(bytes + 24);
    entry->prepare_ns = get_u64(bytes + 32);
    entry->payload_crc = get_u32(bytes + 40);
    memcpy(entry->identity_key, bytes + 44,
           COLI_DERIVED_CACHE_FINGERPRINT_BYTES);
    return entry->expert >= 0 && entry->bytes != 0;
}

static void arena_encode_index(uint8_t bytes[APPLE8_ARENA_INDEX_BYTES],
                               const Apple8ArenaEntry *entry) {
    memset(bytes, 0, APPLE8_ARENA_INDEX_BYTES);
    put_u32(bytes + 0, APPLE8_ARENA_INDEX_MAGIC);
    put_u16(bytes + 4, APPLE8_ARENA_INDEX_VERSION);
    put_u16(bytes + 6, APPLE8_ARENA_INDEX_COMMITTED);
    put_u32(bytes + 8, (uint32_t)entry->expert);
    put_u64(bytes + 16, entry->offset);
    put_u64(bytes + 24, entry->bytes);
    put_u64(bytes + 32, entry->prepare_ns);
    put_u32(bytes + 40, entry->payload_crc);
    memcpy(bytes + 44, entry->identity_key,
           COLI_DERIVED_CACHE_FINGERPRINT_BYTES);
    put_u32(bytes + APPLE8_ARENA_INDEX_BYTES - 4u,
            coli_crc32c(bytes, APPLE8_ARENA_INDEX_BYTES - 4u));
}

static int arena_map_existing(Apple8ArenaLayer *layer) {
    if (!layer || layer->mapping || layer->wrote_in_process ||
        !layer->data_size || layer->data_size > SIZE_MAX)
        return layer && (layer->mapping || !layer->data_size ||
                         layer->wrote_in_process);
    void *mapping = mmap(NULL, (size_t)layer->data_size, PROT_READ,
                         MAP_SHARED, layer->data_fd, 0);
    if (mapping == MAP_FAILED) return 0;
#ifdef MADV_SEQUENTIAL
    (void)madvise(mapping, (size_t)layer->data_size, MADV_SEQUENTIAL);
#endif
    layer->mapping = mapping;
    layer->mapping_bytes = (size_t)layer->data_size;
    return 1;
}

static int arena_load_layer_locked(Apple8ArenaLayer *layer, int32_t layer_id) {
    if (!layer || layer_id < 0 || layer_id >= APPLE8_ARENA_MAX_LAYERS) return 0;
    if (layer->loaded) return 1;

    int dw = snprintf(layer->data_path, sizeof(layer->data_path),
                      "%s/apple8-layer-%03d.arena",
                      g_cache_directory, (int)layer_id);
    int iw = snprintf(layer->index_path, sizeof(layer->index_path),
                      "%s/apple8-layer-%03d.idx",
                      g_cache_directory, (int)layer_id);
    if (dw < 0 || iw < 0 ||
        (size_t)dw >= sizeof(layer->data_path) ||
        (size_t)iw >= sizeof(layer->index_path))
        return 0;

    layer->data_fd = open(layer->data_path, O_RDWR | O_CREAT, 0600);
    if (layer->data_fd < 0) return 0;
    layer->index_fd = open(layer->index_path, O_RDWR | O_CREAT, 0600);
    if (layer->index_fd < 0) {
        close(layer->data_fd);
        layer->data_fd = -1;
        return 0;
    }

    struct stat data_st;
    struct stat index_st;
    if (fstat(layer->data_fd, &data_st) != 0 ||
        fstat(layer->index_fd, &index_st) != 0 ||
        data_st.st_size < 0 || index_st.st_size < 0) {
        close(layer->index_fd);
        close(layer->data_fd);
        layer->data_fd = layer->index_fd = -1;
        return 0;
    }
    layer->data_size = (uint64_t)data_st.st_size;

    uint64_t index_bytes = (uint64_t)index_st.st_size;
    uint64_t cursor = 0;
    uint8_t raw[APPLE8_ARENA_INDEX_BYTES];
    while (cursor + APPLE8_ARENA_INDEX_BYTES <= index_bytes) {
        Apple8ArenaEntry entry;
        if (!pread_all(layer->index_fd, raw, sizeof(raw), cursor)) break;
        cursor += APPLE8_ARENA_INDEX_BYTES;
        if (!arena_decode_index(raw, &entry)) continue;
        if (entry.offset > layer->data_size ||
            entry.bytes > layer->data_size - entry.offset)
            continue;
        if (!arena_upsert_entry(layer, &entry)) {
            close(layer->index_fd);
            close(layer->data_fd);
            layer->data_fd = layer->index_fd = -1;
            return 0;
        }
    }

    layer->loaded = 1;
    (void)arena_map_existing(layer);
    return 1;
}

static int arena_load_blob(const ColiDerivedCacheIdentity *identity,
                           int32_t layer_id, int32_t expert,
                           const void **blob_out, uint64_t *bytes_out,
                           uint64_t *prepare_ns_out, int *mapped_out) {
    if (!identity || !blob_out || !bytes_out || !prepare_ns_out || !mapped_out ||
        layer_id < 0 || layer_id >= APPLE8_ARENA_MAX_LAYERS)
        return 0;
    *blob_out = NULL;
    *bytes_out = 0;
    *prepare_ns_out = 0;
    *mapped_out = 0;

    atomic_fetch_add_explicit(&g_lookup, 1, memory_order_acq_rel);
    uint8_t key[COLI_DERIVED_CACHE_FINGERPRINT_BYTES];
    if (!identity_key(identity, key)) {
        atomic_fetch_add_explicit(&g_miss, 1, memory_order_acq_rel);
        return 0;
    }

    Apple8ArenaLayer *layer = &g_arena_layers[layer_id];
    uint64_t began = now_ns(NULL);
    pthread_mutex_lock(&layer->mutex);
    if (!arena_load_layer_locked(layer, layer_id)) {
        pthread_mutex_unlock(&layer->mutex);
        atomic_fetch_add_explicit(&g_miss, 1, memory_order_acq_rel);
        return 0;
    }

    Apple8ArenaEntry *entry = arena_find_entry(layer, expert);
    if (!entry) {
        pthread_mutex_unlock(&layer->mutex);
        atomic_fetch_add_explicit(&g_miss, 1, memory_order_acq_rel);
        return 0;
    }
    if (memcmp(entry->identity_key, key, sizeof(key)) != 0) {
        pthread_mutex_unlock(&layer->mutex);
        atomic_fetch_add_explicit(&g_miss, 1, memory_order_acq_rel);
        atomic_fetch_add_explicit(&g_stale, 1, memory_order_acq_rel);
        return 0;
    }
    if (!entry->bytes || entry->bytes > g_cache.config.max_object_bytes ||
        entry->offset > layer->data_size ||
        entry->bytes > layer->data_size - entry->offset ||
        entry->bytes > SIZE_MAX) {
        pthread_mutex_unlock(&layer->mutex);
        atomic_fetch_add_explicit(&g_miss, 1, memory_order_acq_rel);
        atomic_fetch_add_explicit(&g_corrupt, 1, memory_order_acq_rel);
        return 0;
    }

    const void *blob = NULL;
    int mapped = 0;
    if (layer->mapping &&
        entry->offset <= layer->mapping_bytes &&
        entry->bytes <= layer->mapping_bytes - entry->offset) {
        blob = (const uint8_t *)layer->mapping + entry->offset;
        mapped = 1;
    } else {
        void *copy = malloc((size_t)entry->bytes);
        if (!copy ||
            !pread_all(layer->data_fd, copy, (size_t)entry->bytes,
                       entry->offset)) {
            free(copy);
            pthread_mutex_unlock(&layer->mutex);
            atomic_fetch_add_explicit(&g_miss, 1, memory_order_acq_rel);
            return 0;
        }
        blob = copy;
    }

    if (!entry->verified &&
        coli_crc32c(blob, (size_t)entry->bytes) != entry->payload_crc) {
        if (!mapped) free((void *)blob);
        pthread_mutex_unlock(&layer->mutex);
        atomic_fetch_add_explicit(&g_miss, 1, memory_order_acq_rel);
        atomic_fetch_add_explicit(&g_corrupt, 1, memory_order_acq_rel);
        return 0;
    }
    entry->verified = 1;
    if (mapped) layer->mapping_pins++;
    *blob_out = blob;
    *bytes_out = entry->bytes;
    *prepare_ns_out = entry->prepare_ns;
    *mapped_out = mapped;
    pthread_mutex_unlock(&layer->mutex);

    uint64_t ended = now_ns(NULL);
    atomic_fetch_add_explicit(&g_hit, 1, memory_order_acq_rel);
    atomic_fetch_add_explicit(&g_read_bytes, *bytes_out, memory_order_acq_rel);
    atomic_fetch_add_explicit(&g_prepare_ns_avoided, *prepare_ns_out,
                              memory_order_acq_rel);
    if (began && ended >= began)
        atomic_fetch_add_explicit(&g_read_ns, ended - began,
                                  memory_order_acq_rel);
    return 1;
}

static void arena_release_blob(int32_t layer_id, const void *blob, int mapped) {
    if (!blob) return;
    if (!mapped) {
        free((void *)blob);
        return;
    }
    if (layer_id < 0 || layer_id >= APPLE8_ARENA_MAX_LAYERS) return;
    Apple8ArenaLayer *layer = &g_arena_layers[layer_id];
    pthread_mutex_lock(&layer->mutex);
    if (layer->mapping_pins) layer->mapping_pins--;
    pthread_mutex_unlock(&layer->mutex);
    pthread_cond_broadcast(&layer->cond);
}

static int arena_store_blob(const ColiDerivedCacheIdentity *identity,
                            int32_t layer_id, int32_t expert,
                            const void *blob, uint64_t bytes,
                            uint64_t prepare_ns) {
    if (!identity || !blob || !bytes ||
        bytes > g_cache.config.max_object_bytes || bytes > SIZE_MAX ||
        layer_id < 0 || layer_id >= APPLE8_ARENA_MAX_LAYERS) {
        atomic_fetch_add_explicit(&g_write_dropped, 1, memory_order_acq_rel);
        return 0;
    }

    uint8_t key[COLI_DERIVED_CACHE_FINGERPRINT_BYTES];
    if (!identity_key(identity, key)) {
        atomic_fetch_add_explicit(&g_write_dropped, 1, memory_order_acq_rel);
        return 0;
    }

    Apple8ArenaLayer *layer = &g_arena_layers[layer_id];
    uint64_t began = now_ns(NULL);
    pthread_mutex_lock(&layer->mutex);
    if (!arena_load_layer_locked(layer, layer_id)) {
        pthread_mutex_unlock(&layer->mutex);
        atomic_fetch_add_explicit(&g_write_dropped, 1, memory_order_acq_rel);
        return 0;
    }

    Apple8ArenaEntry *old = arena_find_entry(layer, expert);
    if (old && old->bytes == bytes &&
        memcmp(old->identity_key, key, sizeof(key)) == 0) {
        pthread_mutex_unlock(&layer->mutex);
        return 1;
    }

    while (layer->mapping && layer->mapping_pins)
        pthread_cond_wait(&layer->cond, &layer->mutex);
    if (layer->mapping) {
        munmap(layer->mapping, layer->mapping_bytes);
        layer->mapping = NULL;
        layer->mapping_bytes = 0;
    }
    layer->wrote_in_process = 1;

    Apple8ArenaEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.expert = expert;
    memcpy(entry.identity_key, key, sizeof(key));
    entry.offset = layer->data_size;
    entry.bytes = bytes;
    entry.prepare_ns = prepare_ns;
    entry.payload_crc = coli_crc32c(blob, (size_t)bytes);
    entry.verified = 1;

    /* Commit protocol: payload is appended first, then a checksummed fixed-size
     * index record. Torn index tails are ignored; a visible record with bad or
     * non-durable payload fails CRC and falls back to authoritative .coli. */
    if (!pwrite_all(layer->data_fd, blob, (size_t)bytes, entry.offset)) {
        pthread_mutex_unlock(&layer->mutex);
        atomic_fetch_add_explicit(&g_write_dropped, 1, memory_order_acq_rel);
        return 0;
    }
    layer->data_size += bytes;

    struct stat index_st;
    uint8_t raw[APPLE8_ARENA_INDEX_BYTES];
    arena_encode_index(raw, &entry);
    if (fstat(layer->index_fd, &index_st) != 0 || index_st.st_size < 0 ||
        !pwrite_all(layer->index_fd, raw, sizeof(raw),
                    (uint64_t)index_st.st_size)) {
        pthread_mutex_unlock(&layer->mutex);
        atomic_fetch_add_explicit(&g_write_dropped, 1, memory_order_acq_rel);
        return 0;
    }

    if (!arena_upsert_entry(layer, &entry)) {
        pthread_mutex_unlock(&layer->mutex);
        atomic_fetch_add_explicit(&g_write_dropped, 1, memory_order_acq_rel);
        return 0;
    }
    pthread_mutex_unlock(&layer->mutex);

    uint64_t ended = now_ns(NULL);
    atomic_fetch_add_explicit(&g_write_bytes, bytes, memory_order_acq_rel);
    if (began && ended >= began)
        atomic_fetch_add_explicit(&g_write_ns, ended - began,
                                  memory_order_acq_rel);
    return 1;
}

static int matrix_eligible(const ColiExpertMatrixInfo *m, size_t resident_bytes) {
    if (!m || m->math_format != COLI_CSF_MATH_MXFP4_E2M1 ||
        m->scale_format != COLI_CSF_SCALE_UE8M0 ||
        m->layout != COLI_CSF_LAYOUT_CANONICAL ||
        m->weight_codec != COLI_CSF_CODEC_NONE ||
        m->scale_codec != COLI_CSF_CODEC_NONE ||
        !m->role || !m->rows || !m->columns ||
        m->rows > INT32_MAX || m->columns > INT32_MAX)
        return 0;

    uint64_t row_bytes = (m->columns + 1u) / 2u;
    uint64_t groups = (m->columns + 31u) / 32u;
    if (!row_bytes || !groups ||
        m->rows > UINT64_MAX / row_bytes ||
        m->rows > UINT64_MAX / groups)
        return 0;
    uint64_t weight_need = m->rows * row_bytes;
    uint64_t scale_need = m->rows * groups;
    if (weight_need > m->weight_stored_bytes ||
        scale_need > m->scale_stored_bytes ||
        m->weight_offset > resident_bytes ||
        weight_need > resident_bytes - m->weight_offset ||
        m->scale_offset > resident_bytes ||
        scale_need > resident_bytes - m->scale_offset)
        return 0;
    return 1;
}

static int build_source(const ColiExpertInfo *info,
                        const void *resident_slot, size_t resident_bytes,
                        ColiMxfp4Apple8SourceExpert *source) {
    if (!info || !resident_slot || !source) return 0;
    memset(source, 0, sizeof(*source));
    source->matrix_count = 3;
    const uint8_t *base = (const uint8_t *)resident_slot;
    for (int i = 0; i < 3; ++i) {
        const ColiExpertMatrixInfo *m = &info->matrices[i];
        if (!matrix_eligible(m, resident_bytes)) return 0;
        uint64_t weight_need = m->rows * ((m->columns + 1u) / 2u);
        uint64_t scale_need = m->rows * ((m->columns + 31u) / 32u);
        source->matrices[i] = (ColiMxfp4Apple8RowMatrix){
            .role = m->role,
            .rows = m->rows,
            .columns = m->columns,
            .weights = base + m->weight_offset,
            .weight_bytes = weight_need,
            .scales = base + m->scale_offset,
            .scale_bytes = scale_need,
        };
    }
    return 1;
}

static uint64_t fp_mix_byte(uint64_t hash, uint8_t byte) {
    hash ^= byte;
    return hash * UINT64_C(1099511628211);
}

static uint64_t fp_mix_u64(uint64_t hash, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        hash = fp_mix_byte(hash, (uint8_t)(value & 0xffu));
        value >>= 8;
    }
    return hash;
}

static uint64_t fp_mix_bytes(uint64_t hash, const uint8_t *bytes, size_t count) {
    for (size_t i = 0; i < count; ++i) hash = fp_mix_byte(hash, bytes[i]);
    return hash;
}

static void fp_store_u64(uint8_t *out, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        out[i] = (uint8_t)(value & 0xffu);
        value >>= 8;
    }
}

static void source_record_fingerprint(
        uint8_t out[COLI_DERIVED_CACHE_FINGERPRINT_BYTES],
        const uint8_t artifact[COLI_DERIVED_CACHE_FINGERPRINT_BYTES],
        const ColiRecordInfo *record, const ColiExpertInfo *info) {
    static const uint64_t seeds[4] = {
        UINT64_C(1469598103934665603),
        UINT64_C(0x9e3779b97f4a7c15),
        UINT64_C(0xd6e8feb86659fd93),
        UINT64_C(0xa0761d6478bd642f),
    };
    for (unsigned lane = 0; lane < 4; ++lane) {
        uint64_t h = fp_mix_bytes(seeds[lane], artifact,
                                  COLI_DERIVED_CACHE_FINGERPRINT_BYTES);
        h = fp_mix_u64(h, record->record_id);
        h = fp_mix_u64(h, record->stored_bytes);
        h = fp_mix_u64(h, record->decoded_bytes);
        h = fp_mix_u64(h, record->stored_crc32c);
        h = fp_mix_u64(h, record->logical_crc32c);
        h = fp_mix_u64(h, (uint32_t)record->kind);
        h = fp_mix_u64(h, (uint32_t)record->codec);
        h = fp_mix_u64(h, (uint32_t)record->layout);
        h = fp_mix_u64(h, info->logical_bytes);
        for (int i = 0; i < 3; ++i) {
            const ColiExpertMatrixInfo *m = &info->matrices[i];
            h = fp_mix_u64(h, m->role);
            h = fp_mix_u64(h, m->math_format);
            h = fp_mix_u64(h, m->scale_format);
            h = fp_mix_u64(h, m->rows);
            h = fp_mix_u64(h, m->columns);
            h = fp_mix_u64(h, m->weight_offset);
            h = fp_mix_u64(h, m->weight_stored_bytes);
            h = fp_mix_u64(h, m->scale_offset);
            h = fp_mix_u64(h, m->scale_stored_bytes);
            h = fp_mix_u64(h, m->logical_crc32c);
        }
        fp_store_u64(out + lane * 8u, h);
    }
}

static int build_identity(const ColiExecutor *executor,
                          int32_t layer, int32_t expert,
                          const ColiRecordInfo *record,
                          const ColiExpertInfo *info,
                          ColiDerivedCacheIdentity *identity) {
    if (!executor || !record || !info || !identity) return 0;
    const ColiPackage *package = coli_executor_package(executor);
    const uint8_t *artifact =
        package ? coli_package_source_fingerprint(package) : NULL;
    if (!artifact) return 0;

    memset(identity, 0, sizeof(*identity));
    memcpy(identity->artifact_fingerprint, artifact,
           COLI_DERIVED_CACHE_FINGERPRINT_BYTES);
    identity->logical_expert = (ColiExpertKey){layer, expert};
    identity->logical_record_id = record->record_id;
    source_record_fingerprint(identity->source_record_fingerprint,
                              artifact, record, info);
    identity->source_record_bytes = record->stored_bytes;
    identity->source_record_crc32c = record->stored_crc32c;
    identity->source_record_abi = APPLE8_SOURCE_RECORD_ABI;
    coli_mxfp4_apple8_source_fixture_representation(
        &identity->source_representation);
    coli_mxfp4_apple8_target_fixture_representation(
        &identity->target_representation);
    identity->transform_abi = COLI_MXFP4_APPLE8_FIXTURE_TRANSFORM_ABI;
    identity->transform_class = COLI_JIT_TRANSFORM_EXACT;
    identity->target_hardware_class =
        identity->target_representation.target_class;
    identity->target_kernel_abi =
        identity->target_representation.kernel_abi;
    identity->target_layout_abi =
        identity->target_representation.execution_layout_abi;
    identity->runtime_byte_abi = APPLE8_RUNTIME_BYTE_ABI;
    identity->compiler_byte_abi = 0;
    return coli_derived_cache_identity_valid(identity);
}

static int install_blob(const ColiMxfp4Apple8SourceExpert *source,
                        const void *blob, uint64_t bytes,
                        uint64_t source_generation) {
    if (!source || !blob || bytes < sizeof(ColiMxfp4Apple8TileExpertHeader))
        return 0;
    const ColiMxfp4Apple8TileExpertHeader *header =
        (const ColiMxfp4Apple8TileExpertHeader *)blob;
    if (header->magic != COLI_MXFP4_APPLE8_TILE_EXPERT_MAGIC ||
        header->version != COLI_MXFP4_APPLE8_TILE_EXPERT_VERSION ||
        header->matrix_count != 3 ||
        header->header_bytes < sizeof(*header) ||
        header->header_bytes > bytes ||
        header->reserved != 0)
        return 0;

    for (uint32_t i = 0; i < 3; ++i) {
        const ColiMxfp4Apple8TileMatrix *tile = &header->matrices[i];
        const ColiMxfp4Apple8RowMatrix *row = &source->matrices[i];
        size_t expected = coli_mxfp4_apple8_tile_bytes(row->rows, row->columns);
        if (!expected || tile->role != row->role ||
            tile->reserved != 0 ||
            tile->rows != row->rows || tile->columns != row->columns ||
            tile->bytes != expected ||
            tile->offset > bytes || tile->bytes > bytes - tile->offset)
            return 0;
        if (!coli_metal_tile_prepare_packed_matrix(
                row->weights, row->scales,
                (int)row->rows, (int)row->columns, source_generation,
                (const uint8_t *)blob + tile->offset, (size_t)tile->bytes))
            return 0;
    }
    return 1;
}

int coli_mxfp4_apple8_derived_prepare_expert(
        const ColiExecutor *executor,
        int32_t layer, int32_t expert,
        const void *resident_slot, size_t resident_bytes,
        uint64_t source_generation) {
    if (!coli_mxfp4_apple8_derived_cache_enabled() ||
        !executor || !resident_slot || !resident_bytes || !source_generation)
        return 0;

    const ColiRecordInfo *record = coli_executor_expert(executor, layer, expert);
    if (!record || record->kind != COLI_CSF_REC_EXPERT ||
        !record->stored_bytes || record->stored_bytes > resident_bytes)
        return 0;

    ColiExpertInfo info;
    char ignored[128] = {0};
    if (coli_executor_expert_info(executor, layer, expert, &info,
                                  ignored, sizeof(ignored)) != 0)
        return 0;

    ColiMxfp4Apple8SourceExpert source;
    if (!build_source(&info, resident_slot, resident_bytes, &source)) return 0;

    ColiDerivedCacheIdentity identity;
    if (!build_identity(executor, layer, expert, record, &info, &identity))
        return 0;

    const void *cached_blob = NULL;
    uint64_t cached_bytes = 0;
    uint64_t prepare_avoided_ns = 0;
    int cached_mapped = 0;
    if (arena_load_blob(&identity, layer, expert,
                        &cached_blob, &cached_bytes, &prepare_avoided_ns,
                        &cached_mapped)) {
        int installed = install_blob(&source, cached_blob, cached_bytes,
                                     source_generation);
        arena_release_blob(layer, cached_blob, cached_mapped);
        if (installed) {
            atomic_fetch_add_explicit(&g_installs, 1, memory_order_acq_rel);
            return 1;
        }
        atomic_fetch_add_explicit(&g_install_failures, 1,
                                  memory_order_acq_rel);
        return 0;
    }

    ColiRepresentationTransformOps ops;
    if (coli_mxfp4_apple8_transform_ops(&ops) != 0) return 0;

    ColiExpertResidentView source_view = {
        .key = {layer, expert},
        .representation = identity.source_representation,
        .generation = source_generation,
        .tier_mask = COLI_EXPERT_TIER_UMA,
        .resident_bytes = record->stored_bytes,
        .allocation_bytes = record->stored_bytes,
        .physical = &source,
    };
    ColiJitTransformEstimate estimate;
    if (ops.estimate(ops.context, &source_view,
                     &identity.target_representation, &estimate) != 0 ||
        !estimate.resident_bytes || !estimate.allocation_bytes ||
        estimate.allocation_bytes > g_cache.config.max_object_bytes)
        return 0;

    void *output = cache_allocate(
        NULL, COLI_JIT_MEMORY_OUTPUT,
        estimate.allocation_bytes, estimate.output_alignment);
    if (!output) return 0;

    uint64_t began = now_ns(NULL);
    int prepared = ops.prepare(
        ops.context, &source_view, &identity.target_representation,
        output, estimate.allocation_bytes,
        NULL, 0, NULL, 0);
    uint64_t ended = now_ns(NULL);
    uint64_t prepare_ns = began && ended >= began ? ended - began : 0;
    if (prepared != 0 ||
        ops.validate(ops.context, &source_view,
                     &identity.target_representation,
                     output, estimate.resident_bytes) != 0) {
        cache_free(NULL, COLI_JIT_MEMORY_OUTPUT,
                   output, estimate.allocation_bytes);
        return 0;
    }

    atomic_fetch_add_explicit(&g_cold_prepares, 1, memory_order_acq_rel);
    atomic_fetch_add_explicit(&g_cold_prepare_ns, prepare_ns,
                              memory_order_acq_rel);

    int installed = install_blob(&source, output, estimate.resident_bytes,
                                 source_generation);
    if (installed) {
        (void)arena_store_blob(&identity, layer, expert, output,
                               estimate.resident_bytes, prepare_ns);
        atomic_fetch_add_explicit(&g_installs, 1, memory_order_acq_rel);
    } else {
        atomic_fetch_add_explicit(&g_install_failures, 1,
                                  memory_order_acq_rel);
    }
    cache_free(NULL, COLI_JIT_MEMORY_OUTPUT,
               output, estimate.allocation_bytes);
    return installed;
}

void coli_mxfp4_apple8_derived_cache_stats(
        ColiMxfp4Apple8DerivedCacheStats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    pthread_once(&g_cache_once, cache_init_once);
    stats->lookup = atomic_load_explicit(&g_lookup, memory_order_acquire);
    stats->hit = atomic_load_explicit(&g_hit, memory_order_acquire);
    stats->miss = atomic_load_explicit(&g_miss, memory_order_acquire);
    stats->stale = atomic_load_explicit(&g_stale, memory_order_acquire);
    stats->corrupt = atomic_load_explicit(&g_corrupt, memory_order_acquire);
    stats->read_bytes = atomic_load_explicit(&g_read_bytes, memory_order_acquire);
    stats->read_ns = atomic_load_explicit(&g_read_ns, memory_order_acquire);
    stats->write_bytes =
        atomic_load_explicit(&g_write_bytes, memory_order_acquire);
    stats->write_ns = atomic_load_explicit(&g_write_ns, memory_order_acquire);
    stats->write_dropped =
        atomic_load_explicit(&g_write_dropped, memory_order_acquire);
    stats->prepare_ns_avoided =
        atomic_load_explicit(&g_prepare_ns_avoided, memory_order_acquire);
    stats->cold_prepares =
        atomic_load_explicit(&g_cold_prepares, memory_order_acquire);
    stats->cold_prepare_ns =
        atomic_load_explicit(&g_cold_prepare_ns, memory_order_acquire);
    stats->installs =
        atomic_load_explicit(&g_installs, memory_order_acquire);
    stats->install_failures =
        atomic_load_explicit(&g_install_failures, memory_order_acquire);
}

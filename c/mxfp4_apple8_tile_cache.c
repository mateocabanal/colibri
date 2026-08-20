#include "mxfp4_apple8_tile_cache.h"

#include "backend_metal_tile.h"
#include "expert_derived_cache.h"
#include "mxfp4_apple8_tile.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define APPLE8_DERIVED_DEFAULT_DISK_GIB UINT64_C(96)
#define APPLE8_DERIVED_DEFAULT_MIN_FREE_GIB UINT64_C(4)
#define APPLE8_DERIVED_MAX_OBJECT_BYTES (UINT64_C(32) * 1024u * 1024u)
#define APPLE8_SOURCE_RECORD_ABI \
    (((uint32_t)COLI_CSF_VERSION_MAJOR << 16) | (uint32_t)COLI_CSF_VERSION_MINOR)
#define APPLE8_RUNTIME_BYTE_ABI ((uint32_t)COLI_MXFP4_APPLE8_TILE_EXPERT_VERSION)

static pthread_once_t g_cache_once = PTHREAD_ONCE_INIT;
static ColiDerivedCache g_cache;
static int g_cache_enabled;
static int g_cache_ready;
static char g_cache_directory[PATH_MAX];

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
    g_cache_ready =
        coli_derived_cache_init(&g_cache, &config, &memory, &payload, &clock) == 0;
    if (!g_cache_ready)
        fprintf(stderr, "v4_metal tile_derived_cache=disabled reason=init-failed dir=%s\n",
                g_cache_directory);
}

int coli_mxfp4_apple8_derived_cache_enabled(void) {
    pthread_once(&g_cache_once, cache_init_once);
    return g_cache_enabled && g_cache_ready;
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
    const uint8_t *artifact = package ? coli_package_source_fingerprint(package) : NULL;
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

static void dispose_variant(ColiExpertResidencyEntry *entry,
                            ColiExpertResidencyBudget *budget,
                            uint32_t variant_id, void *physical,
                            uint64_t allocation_bytes) {
    if (!entry || !budget || variant_id == COLI_EXPERT_VARIANT_NONE ||
        !physical || !allocation_bytes)
        return;
    if (coli_expert_residency_begin_variant_evict(entry, variant_id) == 1 &&
        coli_expert_residency_finish_variant_evict(
            entry, budget, variant_id) == 0)
        cache_free(NULL, COLI_JIT_MEMORY_OUTPUT, physical, allocation_bytes);
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

    ColiExpertResidencyEntry entry;
    if (coli_expert_residency_entry_init(
            &entry, (ColiExpertKey){layer, expert}) != 0)
        return 0;
    ColiExpertResidencyBudget budget;
    coli_expert_residency_budget_init(
        &budget, g_cache.config.max_object_bytes);

    ColiDerivedCacheLoadInfo load;
    if (coli_derived_cache_load_variant(
            &g_cache, &identity, &entry, &budget,
            COLI_EXPERT_TIER_UMA, &load)) {
        ColiExpertResidencyLease lease;
        int acquired = coli_expert_residency_acquire_variant(
            &entry, load.variant_id, &lease);
        if (acquired == 1) {
            void *physical = lease.physical;
            uint64_t allocation_bytes = lease.allocation_bytes;
            int installed = install_blob(
                &source, lease.physical, lease.resident_bytes,
                source_generation);
            (void)coli_expert_residency_release(&lease);
            dispose_variant(&entry, &budget, load.variant_id,
                            physical, allocation_bytes);
            if (installed) {
                atomic_fetch_add_explicit(
                    &g_installs, 1, memory_order_acq_rel);
                return 1;
            }
        }
        atomic_fetch_add_explicit(
            &g_install_failures, 1, memory_order_acq_rel);
        return 0;
    }

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
    atomic_fetch_add_explicit(
        &g_cold_prepare_ns, prepare_ns, memory_order_acq_rel);

    uint32_t variant_id = COLI_EXPERT_VARIANT_NONE;
    uint64_t generation = 0;
    ColiExpertRequestResult reserved =
        coli_expert_residency_reserve_variant(
            &entry, &budget, &identity.target_representation,
            COLI_EXPERT_TIER_UMA, estimate.allocation_bytes,
            &variant_id, &generation);
    if (reserved != COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_mark_variant_preparing(
            &entry, variant_id, generation) != 0 ||
        coli_expert_residency_publish_variant(
            &entry, &budget, variant_id, generation,
            estimate.resident_bytes, output) != 0) {
        if (reserved == COLI_EXPERT_REQUEST_LOAD_OWNER)
            (void)coli_expert_residency_fail_variant(
                &entry, &budget, variant_id, generation);
        cache_free(NULL, COLI_JIT_MEMORY_OUTPUT,
                   output, estimate.allocation_bytes);
        return 0;
    }

    ColiExpertResidencyLease lease;
    if (coli_expert_residency_acquire_variant(
            &entry, variant_id, &lease) != 1) {
        dispose_variant(&entry, &budget, variant_id,
                        output, estimate.allocation_bytes);
        return 0;
    }

    int installed = install_blob(
        &source, lease.physical, lease.resident_bytes,
        source_generation);
    if (installed) {
        (void)coli_derived_cache_store(
            &g_cache, &identity, &lease,
            estimate.output_alignment, prepare_ns);
        atomic_fetch_add_explicit(
            &g_installs, 1, memory_order_acq_rel);
    } else {
        atomic_fetch_add_explicit(
            &g_install_failures, 1, memory_order_acq_rel);
    }

    void *physical = lease.physical;
    uint64_t allocation_bytes = lease.allocation_bytes;
    (void)coli_expert_residency_release(&lease);
    dispose_variant(&entry, &budget, variant_id,
                    physical, allocation_bytes);
    return installed;
}

void coli_mxfp4_apple8_derived_cache_stats(
        ColiMxfp4Apple8DerivedCacheStats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    pthread_once(&g_cache_once, cache_init_once);
    if (g_cache_ready) {
        ColiDerivedCacheTelemetrySnapshot snapshot;
        memset(&snapshot, 0, sizeof(snapshot));
        coli_derived_cache_telemetry_snapshot(&g_cache, &snapshot);
        stats->lookup = snapshot.lookup;
        stats->hit = snapshot.hit;
        stats->miss = snapshot.miss;
        stats->stale = snapshot.stale;
        stats->corrupt = snapshot.corrupt;
        stats->read_bytes = snapshot.read_bytes;
        stats->read_ns = snapshot.read_ns;
        stats->write_bytes = snapshot.write_bytes;
        stats->write_ns = snapshot.write_ns;
        stats->write_dropped = snapshot.write_dropped;
        stats->prepare_ns_avoided = snapshot.prepare_ns_avoided;
    }
    stats->cold_prepares = atomic_load_explicit(
        &g_cold_prepares, memory_order_acquire);
    stats->cold_prepare_ns = atomic_load_explicit(
        &g_cold_prepare_ns, memory_order_acquire);
    stats->installs = atomic_load_explicit(
        &g_installs, memory_order_acquire);
    stats->install_failures = atomic_load_explicit(
        &g_install_failures, memory_order_acquire);
}

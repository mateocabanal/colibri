#ifndef COLIBRI_EXPERT_DERIVED_CACHE_H
#define COLIBRI_EXPERT_DERIVED_CACHE_H

#include "expert_transform.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_DERIVED_CACHE_FINGERPRINT_BYTES 32u
#define COLI_DERIVED_CACHE_IDENTITY_BYTES 256u

typedef struct {
    uint8_t artifact_fingerprint[COLI_DERIVED_CACHE_FINGERPRINT_BYTES];
    ColiExpertKey logical_expert;
    uint64_t logical_record_id;

    uint8_t source_record_fingerprint[COLI_DERIVED_CACHE_FINGERPRINT_BYTES];
    uint64_t source_record_bytes;
    uint32_t source_record_crc32c;
    uint32_t source_record_abi;

    ColiRepresentationId source_representation;
    ColiRepresentationId target_representation;
    uint32_t transform_abi;
    ColiJitTransformClass transform_class;

    /* Explicit duplicates make cross-process compatibility requirements
     * obvious in the cache namespace. Identity validation requires these to
     * match the target representation's shared #133 fields. */
    uint32_t target_hardware_class;
    uint32_t target_kernel_abi;
    uint32_t target_layout_abi;

    /* Zero means the corresponding producer/runtime version does not affect
     * bytes for this transform. Nonzero values participate in identity. */
    uint32_t runtime_byte_abi;
    uint32_t compiler_byte_abi;

    /* Exact transforms require all lossy fields to be zero. Lossy objects are
     * representable but require all explicit quality/calibration identities. */
    uint64_t quant_profile_id;
    uint64_t calibration_profile_id;
    uint32_t quality_policy_abi;
    uint32_t algorithm_abi;
    uint8_t calibration_fingerprint[COLI_DERIVED_CACHE_FINGERPRINT_BYTES];
} ColiDerivedCacheIdentity;

typedef int (*ColiDerivedCacheExportFn)(
    void *context,
    const ColiExpertResidentView *source,
    uint64_t offset,
    void *host_bytes,
    size_t bytes);

typedef int (*ColiDerivedCacheImportFn)(
    void *context,
    void *destination_physical,
    uint64_t offset,
    const void *host_bytes,
    size_t bytes);

typedef struct {
    void *context;
    /* Callbacks are synchronous: return only when the copied bytes are safe to
     * reuse. This keeps the cache core independent of Metal/CUDA object types. */
    ColiDerivedCacheExportFn export_bytes;
    ColiDerivedCacheImportFn import_bytes;
} ColiDerivedCachePayloadOps;

typedef struct {
    const char *directory;
    uint64_t max_disk_bytes;
    uint64_t max_object_bytes;
    uint64_t min_free_bytes;
    int enabled;
} ColiDerivedCacheConfig;

typedef struct {
    uint64_t lookup;
    uint64_t hit;
    uint64_t miss;
    uint64_t stale;
    uint64_t corrupt;
    uint64_t read_bytes;
    uint64_t read_ns;
    uint64_t write_bytes;
    uint64_t write_ns;
    uint64_t write_dropped;
    uint64_t objects;
    uint64_t disk_bytes;
    uint64_t pruned_bytes;
    uint64_t prepare_ns_avoided;
} ColiDerivedCacheTelemetrySnapshot;

typedef struct {
    atomic_uint_fast64_t lookup;
    atomic_uint_fast64_t hit;
    atomic_uint_fast64_t miss;
    atomic_uint_fast64_t stale;
    atomic_uint_fast64_t corrupt;
    atomic_uint_fast64_t read_bytes;
    atomic_uint_fast64_t read_ns;
    atomic_uint_fast64_t write_bytes;
    atomic_uint_fast64_t write_ns;
    atomic_uint_fast64_t write_dropped;
    atomic_uint_fast64_t objects;
    atomic_uint_fast64_t disk_bytes;
    atomic_uint_fast64_t pruned_bytes;
    atomic_uint_fast64_t prepare_ns_avoided;
} ColiDerivedCacheTelemetry;

typedef struct {
    ColiDerivedCacheConfig config;
    ColiJitTransformMemoryOps memory;
    ColiDerivedCachePayloadOps payload;
    ColiJitTransformClockOps clock;
    atomic_uint_fast64_t temp_serial;
    ColiDerivedCacheTelemetry telemetry;
} ColiDerivedCache;

typedef struct {
    uint32_t variant_id;
    uint64_t generation;
    uint64_t resident_bytes;
    uint64_t allocation_bytes;
    uint64_t resident_alignment;
} ColiDerivedCacheLoadInfo;

int coli_derived_cache_identity_valid(const ColiDerivedCacheIdentity *identity);
int coli_derived_cache_identity_equal(const ColiDerivedCacheIdentity *a,
                                      const ColiDerivedCacheIdentity *b);

int coli_derived_cache_init(
    ColiDerivedCache *cache,
    const ColiDerivedCacheConfig *config,
    const ColiJitTransformMemoryOps *memory,
    const ColiDerivedCachePayloadOps *payload,
    const ColiJitTransformClockOps *clock);
void coli_derived_cache_set_enabled(ColiDerivedCache *cache, int enabled);

/* Deterministic object path derived only from persistent correctness identity;
 * no in-process residency generation participates in this namespace. */
int coli_derived_cache_object_path(
    const ColiDerivedCache *cache,
    const ColiDerivedCacheIdentity *identity,
    char *path,
    size_t path_capacity);

/* Explicit write-back: policy (#136) decides whether a validated/promoted
 * resident variant is worth persisting. The authoritative .coli is not passed
 * to this API and therefore cannot be modified by it. */
int coli_derived_cache_store(
    ColiDerivedCache *cache,
    const ColiDerivedCacheIdentity *identity,
    const ColiExpertResidencyLease *target_lease,
    uint64_t resident_alignment,
    uint64_t prepare_ns_avoided);

/* Safe read semantics: returns 1 only for a fully validated cache hit that was
 * published as a #134 resident variant. Missing, stale, corrupt, incompatible,
 * no-budget, allocation and I/O failures all return 0 and leave baseline
 * execution available. The caller must validate authoritative .coli target
 * compatibility before consulting this optional cache. */
int coli_derived_cache_load_variant(
    ColiDerivedCache *cache,
    const ColiDerivedCacheIdentity *expected_identity,
    ColiExpertResidencyEntry *entry,
    ColiExpertResidencyBudget *resident_budget,
    unsigned target_tier_mask,
    ColiDerivedCacheLoadInfo *info);

/* Bounded disposable storage. bytes_needed reserves room for an imminent new
 * object. Only files in this cache's derived-object namespace are removed. */
int coli_derived_cache_prune(ColiDerivedCache *cache, uint64_t bytes_needed);

void coli_derived_cache_telemetry_snapshot(
    const ColiDerivedCache *cache,
    ColiDerivedCacheTelemetrySnapshot *snapshot);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_EXPERT_DERIVED_CACHE_H */

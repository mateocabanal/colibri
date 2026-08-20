#include "../expert_derived_cache.h"
#include "../cache_io.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

typedef struct {
    uint8_t e2m1[8];
    uint8_t scales[4];
} FixtureBlob;

typedef struct {
    unsigned prepare_calls;
    unsigned validate_calls;
    unsigned export_calls;
    unsigned import_calls;
    unsigned alloc_calls[4];
    unsigned free_calls[4];
    uint64_t clock_ns;
} FixtureContext;

static ColiRepresentationId fixture_rep(uint16_t layout) {
    ColiRepresentationId rep = {
        .math_format = COLI_CSF_MATH_MXFP4_E2M1,
        .scale_format = COLI_CSF_SCALE_UE8M0,
        .execution_layout = layout,
        .execution_layout_abi = 0x0137,
        .kernel_abi = 0x0137,
        .target_class = 0x01370001u,
        .group_size = 32,
        .scale_block_rows = 1,
        .scale_block_columns = 32,
    };
    return rep;
}

static uint8_t get_nibble(const uint8_t *bytes, unsigned index) {
    uint8_t byte = bytes[index >> 1];
    return (index & 1u) ? (uint8_t)(byte >> 4) :
                          (uint8_t)(byte & 0x0fu);
}

static void set_nibble(uint8_t *bytes, unsigned index, uint8_t value) {
    unsigned byte_index = index >> 1;
    unsigned shift = (index & 1u) ? 4u : 0u;
    uint8_t mask = (uint8_t)(0x0fu << shift);
    bytes[byte_index] = (uint8_t)(
        (bytes[byte_index] & (uint8_t)~mask) |
        (uint8_t)((value & 0x0fu) << shift));
}

/* Same deterministic test-only exact Apple-layout proof as #135: IDs are
 * sentinels pending #26/#131/#34 and are not production Apple identities. */
static void fixture_repack(const FixtureBlob *source, FixtureBlob *target,
                           uint8_t *scratch16) {
    memset(target, 0, sizeof(*target));
    for (unsigned i = 0; i < 16; ++i)
        scratch16[i] = get_nibble(source->e2m1, i);
    for (unsigned row = 0; row < 4; ++row)
        for (unsigned col = 0; col < 4; ++col)
            set_nibble(target->e2m1, col * 4 + row,
                       scratch16[row * 4 + col]);
    target->scales[0] = source->scales[0];
    target->scales[1] = source->scales[2];
    target->scales[2] = source->scales[1];
    target->scales[3] = source->scales[3];
}

static int fixture_estimate(
        void *opaque,
        const ColiExpertResidentView *source,
        const ColiRepresentationId *target,
        ColiJitTransformEstimate *estimate) {
    (void)opaque;
    (void)target;
    if (!source || !estimate || source->resident_bytes != sizeof(FixtureBlob))
        return -1;
    *estimate = (ColiJitTransformEstimate){
        .resident_bytes = sizeof(FixtureBlob),
        .allocation_bytes = sizeof(FixtureBlob),
        .scratch_bytes = 16,
        .staging_bytes = 4,
        .output_alignment = 8,
        .scratch_alignment = 8,
        .staging_alignment = 4,
    };
    return 0;
}

static int fixture_prepare(
        void *opaque,
        const ColiExpertResidentView *source,
        const ColiRepresentationId *target,
        void *output,
        uint64_t output_bytes,
        void *scratch,
        uint64_t scratch_bytes,
        void *staging,
        uint64_t staging_bytes) {
    FixtureContext *ctx = (FixtureContext *)opaque;
    (void)target;
    if (!ctx || !source || !source->physical || !output ||
        output_bytes < sizeof(FixtureBlob) || !scratch || scratch_bytes < 16 ||
        !staging || staging_bytes < 4)
        return -1;
    ++ctx->prepare_calls;
    memcpy(staging, ((const FixtureBlob *)source->physical)->scales, 4);
    fixture_repack((const FixtureBlob *)source->physical,
                   (FixtureBlob *)output, (uint8_t *)scratch);
    return 0;
}

static int fixture_validate(
        void *opaque,
        const ColiExpertResidentView *source,
        const ColiRepresentationId *target,
        const void *output,
        uint64_t resident_bytes) {
    FixtureContext *ctx = (FixtureContext *)opaque;
    uint8_t scratch[16];
    FixtureBlob roundtrip;
    (void)target;
    if (!ctx || !source || !source->physical || !output ||
        resident_bytes != sizeof(FixtureBlob))
        return -1;
    ++ctx->validate_calls;
    fixture_repack((const FixtureBlob *)output, &roundtrip, scratch);
    return memcmp(&roundtrip, source->physical, sizeof(roundtrip)) == 0
        ? 0 : -1;
}

static void *fixture_allocate(
        void *opaque,
        ColiJitMemoryPurpose purpose,
        uint64_t bytes,
        uint64_t alignment) {
    FixtureContext *ctx = (FixtureContext *)opaque;
    (void)alignment;
    if (!ctx || purpose < COLI_JIT_MEMORY_OUTPUT ||
        purpose > COLI_JIT_MEMORY_STAGING || !bytes || bytes > SIZE_MAX)
        return NULL;
    ++ctx->alloc_calls[purpose];
    return malloc((size_t)bytes);
}

static void fixture_free(
        void *opaque,
        ColiJitMemoryPurpose purpose,
        void *memory,
        uint64_t bytes) {
    FixtureContext *ctx = (FixtureContext *)opaque;
    (void)bytes;
    if (ctx && purpose >= COLI_JIT_MEMORY_OUTPUT &&
        purpose <= COLI_JIT_MEMORY_STAGING)
        ++ctx->free_calls[purpose];
    free(memory);
}

static int fixture_export(
        void *opaque,
        const ColiExpertResidentView *source,
        uint64_t offset,
        void *host_bytes,
        size_t bytes) {
    FixtureContext *ctx = (FixtureContext *)opaque;
    if (!ctx || !source || !source->physical || !host_bytes ||
        offset > source->resident_bytes ||
        bytes > source->resident_bytes - offset)
        return -1;
    ++ctx->export_calls;
    memcpy(host_bytes, (const uint8_t *)source->physical + offset, bytes);
    return 0;
}

static int fixture_import(
        void *opaque,
        void *destination_physical,
        uint64_t offset,
        const void *host_bytes,
        size_t bytes) {
    FixtureContext *ctx = (FixtureContext *)opaque;
    if (!ctx || !destination_physical || !host_bytes) return -1;
    ++ctx->import_calls;
    memcpy((uint8_t *)destination_physical + offset, host_bytes, bytes);
    return 0;
}

static uint64_t fixture_clock(void *opaque) {
    FixtureContext *ctx = (FixtureContext *)opaque;
    ctx->clock_ns += 100;
    return ctx->clock_ns;
}

static int publish_source(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        const ColiRepresentationId *rep,
        FixtureBlob *physical,
        uint32_t *variant_id,
        uint64_t *generation) {
    ColiExpertRequestResult result = coli_expert_residency_reserve_variant(
        entry, budget, rep, COLI_EXPERT_TIER_UMA,
        sizeof(*physical), variant_id, generation);
    if (result != COLI_EXPERT_REQUEST_LOAD_OWNER) return -1;
    if (coli_expert_residency_mark_variant_preparing(
            entry, *variant_id, *generation) != 0)
        return -1;
    return coli_expert_residency_publish_variant(
        entry, budget, *variant_id, *generation,
        sizeof(*physical), physical);
}

static void fill_fingerprint(uint8_t *fp, uint8_t seed) {
    for (unsigned i = 0; i < COLI_DERIVED_CACHE_FINGERPRINT_BYTES; ++i)
        fp[i] = (uint8_t)(seed + i * 7u + 1u);
}

static ColiDerivedCacheIdentity fixture_identity(
        ColiExpertKey key,
        const ColiRepresentationId *source,
        const ColiRepresentationId *target) {
    ColiDerivedCacheIdentity identity;
    memset(&identity, 0, sizeof(identity));
    fill_fingerprint(identity.artifact_fingerprint, 0x11);
    fill_fingerprint(identity.source_record_fingerprint, 0x41);
    identity.logical_expert = key;
    identity.logical_record_id = UINT64_C(0x1122334455667788);
    identity.source_record_bytes = sizeof(FixtureBlob);
    identity.source_record_crc32c = UINT32_C(0x89abcdef);
    identity.source_record_abi = 7;
    identity.source_representation = *source;
    identity.target_representation = *target;
    identity.transform_abi = 1;
    identity.transform_class = COLI_JIT_TRANSFORM_EXACT;
    identity.target_hardware_class = target->target_class;
    identity.target_kernel_abi = target->kernel_abi;
    identity.target_layout_abi = target->execution_layout_abi;
    identity.runtime_byte_abi = 3;
    identity.compiler_byte_abi = 5;
    return identity;
}

static int make_directory(char *path, size_t capacity, const char *tag) {
    int written = snprintf(path, capacity, "./.test-derived-cache-%s-%llu",
                           tag, (unsigned long long)coli_cache_process_id());
    if (written < 0 || (size_t)written >= capacity) return 0;
#ifdef _WIN32
    if (_mkdir(path) != 0 && errno != EEXIST) return 0;
#else
    if (mkdir(path, 0700) != 0 && errno != EEXIST) return 0;
#endif
    return 1;
}

static void remove_directory(const char *path) {
#ifdef _WIN32
    (void)_rmdir(path);
#else
    (void)rmdir(path);
#endif
}

static int copy_file(const char *source, const char *target) {
    uint8_t buffer[1024];
    FILE *in = fopen(source, "rb");
    FILE *out;
    if (!in) return 0;
    out = fopen(target, "wb");
    if (!out) { fclose(in); return 0; }
    for (;;) {
        size_t got = fread(buffer, 1, sizeof(buffer), in);
        if (got && fwrite(buffer, 1, got, out) != got) {
            fclose(in); fclose(out); return 0;
        }
        if (got < sizeof(buffer)) {
            if (ferror(in)) { fclose(in); fclose(out); return 0; }
            break;
        }
    }
    if (!coli_cache_sync_file(out)) { fclose(in); fclose(out); return 0; }
    return fclose(in) == 0 && fclose(out) == 0;
}

static int corrupt_last_byte(const char *path) {
    FILE *file = fopen(path, "r+b");
    int value;
    if (!file || fseek(file, -1, SEEK_END) != 0) {
        if (file) fclose(file);
        return 0;
    }
    value = fgetc(file);
    if (value == EOF || fseek(file, -1, SEEK_END) != 0 ||
        fputc(value ^ 0x5a, file) == EOF || !coli_cache_sync_file(file)) {
        fclose(file);
        return 0;
    }
    return fclose(file) == 0;
}

static int truncate_file(const char *path) {
    static const uint8_t junk[5] = {1, 2, 3, 4, 5};
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    if (fwrite(junk, 1, sizeof(junk), file) != sizeof(junk) ||
        !coli_cache_sync_file(file)) {
        fclose(file);
        return 0;
    }
    return fclose(file) == 0;
}

#define CHECK(condition, code) do { if (!(condition)) return (code); } while (0)

int main(void) {
    FixtureContext ctx = {0};
    ColiRepresentationId rep_a = fixture_rep(0x7301);
    ColiRepresentationId rep_b = fixture_rep(0x7302);
    ColiExpertKey key = {.layer = 8, .expert = 23};
    FixtureBlob source = {
        .e2m1 = {0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe},
        .scales = {3, 7, 11, 19},
    };
    CHECK(coli_representation_exact_math_compatible(&rep_a, &rep_b), 1);

    ColiJitTransformMemoryOps memory = {
        .context = &ctx,
        .allocate = fixture_allocate,
        .free = fixture_free,
    };
    ColiDerivedCachePayloadOps payload = {
        .context = &ctx,
        .export_bytes = fixture_export,
        .import_bytes = fixture_import,
    };
    ColiJitTransformClockOps clock = {
        .context = &ctx,
        .now_ns = fixture_clock,
    };

    /* Produce the first derived object through the real #135 service. */
    ColiRepresentationTransformRegistry registry;
    coli_jit_transform_registry_init(&registry);
    ColiRepresentationTransformOps transform = {
        .source = rep_a,
        .target = rep_b,
        .transform_abi = 1,
        .transform_class = COLI_JIT_TRANSFORM_EXACT,
        .target_tier_mask = COLI_EXPERT_TIER_UMA,
        .backend_tag = 0x1371,
        .context = &ctx,
        .estimate = fixture_estimate,
        .prepare = fixture_prepare,
        .validate = fixture_validate,
    };
    CHECK(coli_jit_transform_registry_register(&registry, &transform) == 1, 2);

    ColiExpertResidencyEntry source_entry;
    ColiExpertResidencyBudget source_budget;
    coli_expert_residency_budget_init(&source_budget, 4096);
    CHECK(coli_expert_residency_entry_init(&source_entry, key) == 0, 3);
    uint32_t source_variant = COLI_EXPERT_VARIANT_NONE;
    uint64_t source_generation = 0;
    CHECK(publish_source(&source_entry, &source_budget, &rep_a, &source,
                         &source_variant, &source_generation) == 0, 4);

    ColiJitTransformTempBudget temp_budget;
    ColiJitTransformService service;
    coli_jit_transform_temp_budget_init(&temp_budget, 128);
    CHECK(coli_jit_transform_service_init(
              &service, &registry, &temp_budget, &memory, &clock) == 0, 5);
    ColiJitTransformTicket ticket;
    CHECK(coli_jit_transform_request(
              &service, &source_entry, source_variant, source_generation,
              &source_budget, &rep_b, 1, COLI_JIT_PRIORITY_BACKGROUND,
              &ticket) == COLI_JIT_REQUEST_OWNER, 6);
    CHECK(coli_jit_transform_run_one(&service) == 1 &&
          ctx.prepare_calls == 1 && ctx.validate_calls == 1, 7);
    ColiJitTransformAttemptInfo attempt;
    CHECK(coli_jit_transform_query(&service, ticket, &attempt) == 0 &&
          attempt.state == COLI_JIT_ATTEMPT_PUBLISHED, 8);
    CHECK(coli_expert_residency_set_preferred(
              &source_entry, attempt.destination_variant_id) == 0, 9);
    ColiExpertResidencyLease derived_lease;
    CHECK(coli_expert_residency_acquire_variant(
              &source_entry, attempt.destination_variant_id,
              &derived_lease) == 1 &&
          derived_lease.generation == attempt.destination_generation, 10);
    FixtureBlob expected_prepared = *(const FixtureBlob *)derived_lease.physical;

    ColiDerivedCacheIdentity identity = fixture_identity(key, &rep_a, &rep_b);
    CHECK(coli_derived_cache_identity_valid(&identity), 11);
    ColiDerivedCacheIdentity lossy = identity;
    lossy.transform_class = COLI_JIT_TRANSFORM_LOSSY;
    lossy.quant_profile_id = 10;
    lossy.calibration_profile_id = 11;
    lossy.quality_policy_abi = 12;
    lossy.algorithm_abi = 13;
    fill_fingerprint(lossy.calibration_fingerprint, 0x81);
    CHECK(coli_derived_cache_identity_valid(&lossy), 12);
    ColiDerivedCacheIdentity lossy_other = lossy;
    lossy_other.algorithm_abi++;
    CHECK(!coli_derived_cache_identity_equal(&lossy, &lossy_other), 13);

    char cache_dir[192];
    CHECK(make_directory(cache_dir, sizeof(cache_dir), "main"), 14);
    ColiDerivedCacheConfig config = {
        .directory = cache_dir,
        .max_disk_bytes = 65536,
        .max_object_bytes = 4096,
        .min_free_bytes = 0,
        .enabled = 1,
    };
    ColiDerivedCache cache;
    CHECK(coli_derived_cache_init(
              &cache, &config, &memory, &payload, &clock) == 0, 15);
    CHECK(coli_derived_cache_store(
              &cache, &identity, &derived_lease, 8, 777) == 1, 16);
    char good_path[320];
    CHECK(coli_derived_cache_object_path(
              &cache, &identity, good_path, sizeof(good_path)), 17);

    /* Restart proof: populate rep B directly from cache without rerunning #135. */
    ColiExpertResidencyEntry restart_entry;
    ColiExpertResidencyBudget restart_budget;
    coli_expert_residency_budget_init(&restart_budget, 4096);
    CHECK(coli_expert_residency_entry_init(&restart_entry, key) == 0, 18);
    ColiDerivedCacheLoadInfo loaded;
    unsigned prepare_before_hit = ctx.prepare_calls;
    CHECK(coli_derived_cache_load_variant(
              &cache, &identity, &restart_entry, &restart_budget,
              COLI_EXPERT_TIER_UMA, &loaded) == 1 &&
          ctx.prepare_calls == prepare_before_hit, 19);
    ColiExpertResidencyLease restart_lease;
    CHECK(coli_expert_residency_acquire_variant(
              &restart_entry, loaded.variant_id, &restart_lease) == 1 &&
          restart_lease.generation == loaded.generation &&
          coli_representation_equal(&restart_lease.representation, &rep_b) &&
          memcmp(restart_lease.physical, &expected_prepared,
                 sizeof(expected_prepared)) == 0, 20);

    /* The live resident allocation is independent of cache-file lifetime. */
    CHECK(remove(good_path) == 0 &&
          memcmp(restart_lease.physical, &expected_prepared,
                 sizeof(expected_prepared)) == 0 &&
          coli_expert_residency_release(&restart_lease) == 0, 21);
    CHECK(coli_derived_cache_store(
              &cache, &identity, &derived_lease, 8, 777) == 1, 22);

    /* Artifact/source identity mismatch is a miss. Copying a valid object into
     * the wrong key path proves embedded identity is checked, not just name. */
    ColiDerivedCacheIdentity stale_identity = identity;
    stale_identity.artifact_fingerprint[0] ^= 0x55;
    CHECK(coli_derived_cache_identity_valid(&stale_identity), 23);
    char stale_path[320];
    CHECK(coli_derived_cache_object_path(
              &cache, &stale_identity, stale_path, sizeof(stale_path)) &&
          copy_file(good_path, stale_path), 24);
    ColiExpertResidencyEntry stale_entry;
    ColiExpertResidencyBudget stale_budget;
    coli_expert_residency_budget_init(&stale_budget, 4096);
    CHECK(coli_expert_residency_entry_init(&stale_entry, key) == 0 &&
          coli_derived_cache_load_variant(
              &cache, &stale_identity, &stale_entry, &stale_budget,
              COLI_EXPERT_TIER_UMA, &loaded) == 0 &&
          atomic_load(&stale_budget.committed_bytes) == 0, 25);
    CHECK(remove(stale_path) == 0, 26);

    /* Kernel/layout compatibility and transform ABI are persistent key inputs. */
    ColiDerivedCacheIdentity kernel_mismatch = identity;
    kernel_mismatch.target_representation.kernel_abi++;
    kernel_mismatch.target_kernel_abi =
        kernel_mismatch.target_representation.kernel_abi;
    CHECK(coli_derived_cache_identity_valid(&kernel_mismatch), 27);
    char mismatch_path[320];
    CHECK(coli_derived_cache_object_path(
              &cache, &kernel_mismatch, mismatch_path, sizeof(mismatch_path)) &&
          copy_file(good_path, mismatch_path), 28);
    ColiExpertResidencyEntry mismatch_entry;
    ColiExpertResidencyBudget mismatch_budget;
    coli_expert_residency_budget_init(&mismatch_budget, 4096);
    CHECK(coli_expert_residency_entry_init(&mismatch_entry, key) == 0 &&
          coli_derived_cache_load_variant(
              &cache, &kernel_mismatch, &mismatch_entry, &mismatch_budget,
              COLI_EXPERT_TIER_UMA, &loaded) == 0, 29);
    CHECK(remove(mismatch_path) == 0, 30);

    ColiDerivedCacheIdentity abi_mismatch = identity;
    abi_mismatch.transform_abi++;
    CHECK(coli_derived_cache_object_path(
              &cache, &abi_mismatch, mismatch_path, sizeof(mismatch_path)) &&
          copy_file(good_path, mismatch_path), 31);
    CHECK(coli_expert_residency_entry_init(&mismatch_entry, key) == 0 &&
          coli_derived_cache_load_variant(
              &cache, &abi_mismatch, &mismatch_entry, &mismatch_budget,
              COLI_EXPERT_TIER_UMA, &loaded) == 0, 32);
    CHECK(remove(mismatch_path) == 0, 33);

    /* Corrupt payload and truncation are safe misses; baseline remains usable. */
    CHECK(corrupt_last_byte(good_path), 34);
    ColiExpertResidencyEntry fallback_entry;
    ColiExpertResidencyBudget fallback_budget;
    FixtureBlob fallback_source = source;
    uint32_t fallback_variant = COLI_EXPERT_VARIANT_NONE;
    uint64_t fallback_generation = 0;
    coli_expert_residency_budget_init(&fallback_budget, 4096);
    CHECK(coli_expert_residency_entry_init(&fallback_entry, key) == 0 &&
          publish_source(&fallback_entry, &fallback_budget, &rep_a,
                         &fallback_source, &fallback_variant,
                         &fallback_generation) == 0, 35);
    CHECK(coli_derived_cache_load_variant(
              &cache, &identity, &fallback_entry, &fallback_budget,
              COLI_EXPERT_TIER_UMA, &loaded) == 0 &&
          coli_expert_residency_preferred_variant(&fallback_entry) ==
              (int)fallback_variant, 36);
    ColiExpertResidencyLease fallback_lease;
    CHECK(coli_expert_residency_acquire_variant(
              &fallback_entry, fallback_variant, &fallback_lease) == 1 &&
          fallback_lease.generation == fallback_generation &&
          coli_expert_residency_release(&fallback_lease) == 0, 37);
    CHECK(coli_derived_cache_store(
              &cache, &identity, &derived_lease, 8, 777) == 1 &&
          truncate_file(good_path), 38);
    CHECK(coli_expert_residency_entry_init(&mismatch_entry, key) == 0 &&
          coli_derived_cache_load_variant(
              &cache, &identity, &mismatch_entry, &mismatch_budget,
              COLI_EXPERT_TIER_UMA, &loaded) == 0, 39);
    CHECK(coli_derived_cache_store(
              &cache, &identity, &derived_lease, 8, 777) == 1, 40);

    /* Disabled cache mutates neither residency nor transform state. */
    coli_derived_cache_set_enabled(&cache, 0);
    ColiExpertResidencyEntry disabled_entry;
    ColiExpertResidencyBudget disabled_budget;
    coli_expert_residency_budget_init(&disabled_budget, 4096);
    CHECK(coli_expert_residency_entry_init(&disabled_entry, key) == 0 &&
          coli_derived_cache_load_variant(
              &cache, &identity, &disabled_entry, &disabled_budget,
              COLI_EXPERT_TIER_UMA, &loaded) == 0 &&
          atomic_load(&disabled_budget.committed_bytes) == 0, 41);
    coli_derived_cache_set_enabled(&cache, 1);

    /* Cross-process duplicate creation uses unique temps and converges on one
     * valid final object. POSIX exercises a real race; Windows repeats publish. */
#ifndef _WIN32
    pid_t child1 = fork();
    if (child1 == 0)
        _exit(coli_derived_cache_store(
            &cache, &identity, &derived_lease, 8, 777) ? 0 : 1);
    CHECK(child1 > 0, 42);
    pid_t child2 = fork();
    if (child2 == 0)
        _exit(coli_derived_cache_store(
            &cache, &identity, &derived_lease, 8, 777) ? 0 : 1);
    CHECK(child2 > 0, 43);
    int status1 = 0, status2 = 0;
    CHECK(waitpid(child1, &status1, 0) == child1 &&
          waitpid(child2, &status2, 0) == child2 &&
          WIFEXITED(status1) && WEXITSTATUS(status1) == 0 &&
          WIFEXITED(status2) && WEXITSTATUS(status2) == 0, 44);
#else
    CHECK(coli_derived_cache_store(
              &cache, &identity, &derived_lease, 8, 777) == 1 &&
          coli_derived_cache_store(
              &cache, &identity, &derived_lease, 8, 777) == 1, 44);
#endif
    ColiExpertResidencyEntry race_entry;
    ColiExpertResidencyBudget race_budget;
    coli_expert_residency_budget_init(&race_budget, 4096);
    CHECK(coli_expert_residency_entry_init(&race_entry, key) == 0 &&
          coli_derived_cache_load_variant(
              &cache, &identity, &race_entry, &race_budget,
              COLI_EXPERT_TIER_UMA, &loaded) == 1, 45);

    /* Hard disk budget prunes derived objects only. */
    char prune_dir[192];
    CHECK(make_directory(prune_dir, sizeof(prune_dir), "prune"), 46);
    ColiDerivedCacheConfig prune_config = config;
    prune_config.directory = prune_dir;
    prune_config.max_disk_bytes = 500;
    prune_config.max_object_bytes = 64;
    ColiDerivedCache prune_cache;
    CHECK(coli_derived_cache_init(
              &prune_cache, &prune_config, &memory, &payload, &clock) == 0, 47);
    ColiDerivedCacheIdentity identity2 = identity;
    identity2.logical_record_id++;
    identity2.source_record_fingerprint[3] ^= 0x22;
    CHECK(coli_derived_cache_store(
              &prune_cache, &identity, &derived_lease, 8, 777) == 1, 48);
    char prune_path1[320], prune_path2[320];
    CHECK(coli_derived_cache_object_path(
              &prune_cache, &identity, prune_path1, sizeof(prune_path1)) &&
          coli_derived_cache_object_path(
              &prune_cache, &identity2, prune_path2, sizeof(prune_path2)), 49);
    CHECK(coli_derived_cache_store(
              &prune_cache, &identity2, &derived_lease, 8, 777) == 1, 50);
    FILE *old_file = fopen(prune_path1, "rb");
    if (old_file) fclose(old_file);
    FILE *new_file = fopen(prune_path2, "rb");
    CHECK(old_file == NULL && new_file != NULL, 51);
    fclose(new_file);
    ColiDerivedCacheTelemetrySnapshot prune_stats;
    coli_derived_cache_telemetry_snapshot(&prune_cache, &prune_stats);
    CHECK(prune_stats.objects == 1 && prune_stats.disk_bytes <= 500 &&
          prune_stats.pruned_bytes > 0, 52);

    ColiDerivedCacheTelemetrySnapshot stats;
    coli_derived_cache_telemetry_snapshot(&cache, &stats);
    CHECK(stats.hit >= 2 && stats.miss >= 5 && stats.stale >= 3 &&
          stats.corrupt >= 2 && stats.write_bytes > 0 &&
          stats.prepare_ns_avoided >= 777 && ctx.prepare_calls == 1 &&
          ctx.export_calls > 0 && ctx.import_calls > 0, 53);

    CHECK(coli_expert_residency_release(&derived_lease) == 0, 54);
    (void)remove(good_path);
    (void)remove(prune_path1);
    (void)remove(prune_path2);
    remove_directory(cache_dir);
    remove_directory(prune_dir);

    puts("PASS persistent derived JIT cache + restart-safe exact MXFP4 reload");
    return 0;
}

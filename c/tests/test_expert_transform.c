#include "../expert_transform.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t e2m1[8];
    uint8_t scales[4];
} FixtureBlob;

typedef struct {
    unsigned prepare_calls;
    unsigned validate_calls;
    unsigned alloc_calls[4];
    unsigned free_calls[4];
    int fail_validate;
    uint64_t clock_ns;
} FixtureContext;

static ColiRepresentationId fixture_rep(uint16_t layout) {
    ColiRepresentationId rep = {
        .math_format = COLI_CSF_MATH_MXFP4_E2M1,
        .scale_format = COLI_CSF_SCALE_UE8M0,
        .execution_layout = layout,
        .execution_layout_abi = 0x0135,
        .kernel_abi = 0x0135,
        .target_class = 0x01350001u,
        .group_size = 32,
        .scale_block_rows = 1,
        .scale_block_columns = 32,
    };
    return rep;
}

static uint8_t get_nibble(const uint8_t *bytes, unsigned index) {
    uint8_t byte = bytes[index >> 1];
    return (index & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0fu);
}

static void set_nibble(uint8_t *bytes, unsigned index, uint8_t value) {
    unsigned byte_index = index >> 1;
    unsigned shift = (index & 1u) ? 4u : 0u;
    uint8_t mask = (uint8_t)(0x0fu << shift);
    bytes[byte_index] = (uint8_t)(
        (bytes[byte_index] & (uint8_t)~mask) |
        (uint8_t)((value & 0x0fu) << shift));
}

/* Test-only Apple-layout fixture: a 4x4 transpose of packed E2M1 nibbles and
 * a 2x2 transpose of E8M0 scales. The transform is its own inverse and never
 * dequantizes/requantizes. IDs are sentinels, not production #26/#131 IDs. */
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
    if (!source || !target || !estimate ||
        source->resident_bytes != sizeof(FixtureBlob))
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
    if (!ctx || !source || !target || !source->physical || !output ||
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
    if (!ctx || !source || !target || !source->physical || !output ||
        resident_bytes != sizeof(FixtureBlob))
        return -1;
    ++ctx->validate_calls;
    if (ctx->fail_validate) return -1;
    uint8_t scratch[16];
    FixtureBlob roundtrip;
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
        purpose > COLI_JIT_MEMORY_STAGING || !bytes)
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

static int blob_roundtrip_equal(
        const FixtureBlob *source, const FixtureBlob *prepared) {
    uint8_t scratch[16];
    FixtureBlob roundtrip;
    fixture_repack(prepared, &roundtrip, scratch);
    return memcmp(source, &roundtrip, sizeof(roundtrip)) == 0;
}

#define CHECK(condition, code) do { if (!(condition)) return (code); } while (0)

int main(void) {
    FixtureContext ctx = {0};
    ColiRepresentationId rep_a = fixture_rep(0x7101);
    ColiRepresentationId rep_b = fixture_rep(0x7102);
    ColiRepresentationId rep_c = fixture_rep(0x7103);
    CHECK(coli_representation_exact_math_compatible(&rep_a, &rep_b), 1);

    ColiRepresentationTransformRegistry registry;
    coli_jit_transform_registry_init(&registry);
    ColiRepresentationTransformOps a_to_b = {
        .source = rep_a,
        .target = rep_b,
        .transform_abi = 1,
        .transform_class = COLI_JIT_TRANSFORM_EXACT,
        .target_tier_mask = COLI_EXPERT_TIER_UMA,
        .backend_tag = 0x1351,
        .context = &ctx,
        .estimate = fixture_estimate,
        .prepare = fixture_prepare,
        .validate = fixture_validate,
    };
    ColiRepresentationTransformOps a_to_c = a_to_b;
    a_to_c.target = rep_c;
    a_to_c.transform_abi = 2;
    a_to_c.backend_tag = 0x1352;
    ColiRepresentationTransformOps same_target_other_abi = a_to_b;
    same_target_other_abi.transform_abi = 3;
    ColiRepresentationTransformOps lossy = a_to_b;
    lossy.target = rep_c;
    lossy.transform_abi = 99;
    lossy.transform_class = COLI_JIT_TRANSFORM_LOSSY;

    CHECK(coli_jit_transform_registry_register(&registry, &a_to_b) == 1, 2);
    CHECK(coli_jit_transform_registry_register(&registry, &a_to_c) == 1, 3);
    CHECK(coli_jit_transform_registry_register(
              &registry, &same_target_other_abi) == 1, 4);
    CHECK(coli_jit_transform_registry_register(&registry, &lossy) == -2, 5);

    ColiExpertResidencyEntry entry;
    ColiExpertResidencyBudget resident_budget;
    ColiJitTransformTempBudget temp_budget;
    ColiExpertKey key = {7, 23};
    coli_expert_residency_budget_init(&resident_budget, 64);
    coli_jit_transform_temp_budget_init(&temp_budget, 64);
    CHECK(coli_expert_residency_entry_init(&entry, key) == 0, 6);

    FixtureBlob source = {
        .e2m1 = {0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe},
        .scales = {0x81, 0x82, 0x83, 0x84},
    };
    uint32_t source_variant = COLI_EXPERT_VARIANT_NONE;
    uint64_t source_generation = 0;
    CHECK(publish_source(
              &entry, &resident_budget, &rep_a, &source,
              &source_variant, &source_generation) == 0, 7);
    CHECK(source_variant == 0 &&
          coli_expert_residency_preferred_variant(&entry) == 0, 8);

    ColiJitTransformMemoryOps memory = {
        .context = &ctx,
        .allocate = fixture_allocate,
        .free = fixture_free,
    };
    ColiJitTransformClockOps clock = {
        .context = &ctx,
        .now_ns = fixture_clock,
    };
    ColiJitTransformService service;
    CHECK(coli_jit_transform_service_init(
              &service, &registry, &temp_budget, &memory, &clock) == 0, 9);

    ColiJitTransformTicket owner_ticket;
    CHECK(coli_jit_transform_request(
              &service, &entry, source_variant, source_generation,
              &resident_budget, &rep_b, 1, COLI_JIT_PRIORITY_BACKGROUND,
              &owner_ticket) == COLI_JIT_REQUEST_OWNER, 10);
    ColiJitTransformAttemptInfo owner_info;
    CHECK(coli_jit_transform_query(
              &service, owner_ticket, &owner_info) == 0 &&
          owner_info.state == COLI_JIT_ATTEMPT_QUEUED &&
          owner_info.destination_variant_id != source_variant, 11);

    /* Request is asynchronous: all final+temporary bytes are hard-reserved, but
     * no physical transform/allocation has begun yet. The source has one
     * service-owned lease and remains the preferred executable variant. */
    CHECK(ctx.prepare_calls == 0 &&
          ctx.alloc_calls[COLI_JIT_MEMORY_OUTPUT] == 0 &&
          atomic_load(&entry.variants[source_variant].refs) == 1 &&
          atomic_load(&resident_budget.committed_bytes) == 24 &&
          atomic_load(&resident_budget.resident_bytes) == 12 &&
          atomic_load(&resident_budget.reserved_bytes) == 12 &&
          coli_jit_transform_temp_budget_committed(&temp_budget) == 20 &&
          atomic_load(&temp_budget.scratch_bytes) == 16 &&
          atomic_load(&temp_budget.staging_bytes) == 4 &&
          coli_expert_residency_preferred_variant(&entry) ==
              (int)source_variant, 12);

    ColiJitTransformTicket join_ticket;
    CHECK(coli_jit_transform_request(
              &service, &entry, source_variant, source_generation,
              &resident_budget, &rep_b, 1, COLI_JIT_PRIORITY_WARMUP,
              &join_ticket) == COLI_JIT_REQUEST_JOIN &&
          join_ticket.slot == owner_ticket.slot &&
          join_ticket.serial == owner_ticket.serial &&
          atomic_load(&entry.variants[source_variant].refs) == 1, 13);

    /* Same destination residency is NOT sufficient to join different transform
     * work: a different ABI sees the destination as busy, not as its own job. */
    ColiJitTransformTicket different_abi_ticket;
    CHECK(coli_jit_transform_request(
              &service, &entry, source_variant, source_generation,
              &resident_budget, &rep_b, 3, COLI_JIT_PRIORITY_BACKGROUND,
              &different_abi_ticket) == COLI_JIT_REQUEST_DESTINATION_BUSY &&
          atomic_load(&entry.variants[source_variant].refs) == 1, 14);

    ColiExpertResidencyLease baseline;
    CHECK(coli_expert_residency_acquire_preferred(&entry, &baseline) == 1 &&
          baseline.variant_id == source_variant &&
          baseline.generation == source_generation &&
          baseline.physical == &source, 15);
    CHECK(coli_expert_residency_begin_variant_evict(
              &entry, source_variant) == 0, 16);
    CHECK(coli_expert_residency_release(&baseline) == 0 &&
          coli_expert_residency_begin_variant_evict(
              &entry, source_variant) == 0, 17);

    CHECK(coli_jit_transform_run_one(&service) == 1, 18);
    CHECK(coli_jit_transform_query(
              &service, owner_ticket, &owner_info) == 0 &&
          owner_info.state == COLI_JIT_ATTEMPT_PUBLISHED &&
          owner_info.error == COLI_JIT_ERROR_NONE, 19);
    CHECK(ctx.prepare_calls == 1 && ctx.validate_calls == 1 &&
          atomic_load(&entry.variants[source_variant].refs) == 0 &&
          coli_jit_transform_temp_budget_committed(&temp_budget) == 0 &&
          atomic_load(&temp_budget.scratch_bytes) == 0 &&
          atomic_load(&temp_budget.staging_bytes) == 0 &&
          coli_expert_residency_preferred_variant(&entry) ==
              (int)source_variant &&
          coli_expert_residency_variant_state(
              &entry, owner_info.destination_variant_id) ==
              COLI_EXPERT_RESIDENCY_RESIDENT, 20);

    ColiExpertResidencyLease prepared;
    CHECK(coli_expert_residency_acquire_compatible(
              &entry, &rep_b, &prepared) == 1 &&
          prepared.variant_id == owner_info.destination_variant_id &&
          prepared.generation == owner_info.destination_generation &&
          prepared.physical != NULL &&
          blob_roundtrip_equal(&source, (const FixtureBlob *)prepared.physical),
          21);
    void *published_output = prepared.physical;
    CHECK(coli_expert_residency_release(&prepared) == 0, 22);

    ColiJitTransformTelemetrySnapshot stats;
    memset(&stats, 0, sizeof(stats));
    coli_jit_transform_telemetry_snapshot(&service, &stats);
    CHECK(stats.requested == 3 && stats.owner == 1 && stats.join == 1 &&
          stats.started == 1 && stats.completed == 1 && stats.failed == 0 &&
          stats.exact == 1 && stats.lossy == 0 &&
          stats.input_bytes == sizeof(FixtureBlob) &&
          stats.output_bytes == sizeof(FixtureBlob) &&
          stats.scratch_bytes == 16 && stats.staging_bytes == 4 &&
          stats.queue_ns > 0 && stats.prepare_ns > 0 &&
          stats.validate_ns > 0 && stats.last_backend_tag == 0x1351, 23);

    CHECK(ctx.alloc_calls[COLI_JIT_MEMORY_OUTPUT] == 1 &&
          ctx.alloc_calls[COLI_JIT_MEMORY_SCRATCH] == 1 &&
          ctx.alloc_calls[COLI_JIT_MEMORY_STAGING] == 1 &&
          ctx.free_calls[COLI_JIT_MEMORY_OUTPUT] == 0 &&
          ctx.free_calls[COLI_JIT_MEMORY_SCRATCH] == 1 &&
          ctx.free_calls[COLI_JIT_MEMORY_STAGING] == 1, 24);

    CHECK(coli_expert_residency_begin_variant_evict(
              &entry, owner_info.destination_variant_id) == 1 &&
          coli_expert_residency_finish_variant_evict(
              &entry, &resident_budget,
              owner_info.destination_variant_id) == 0, 25);
    fixture_free(
        &ctx, COLI_JIT_MEMORY_OUTPUT, published_output, sizeof(FixtureBlob));

    /* Validation failure never publishes partial output and leaves A serving. */
    ctx.fail_validate = 1;
    ColiJitTransformTicket failed_ticket;
    CHECK(coli_jit_transform_request(
              &service, &entry, source_variant, source_generation,
              &resident_budget, &rep_c, 2, COLI_JIT_PRIORITY_WARMUP,
              &failed_ticket) == COLI_JIT_REQUEST_OWNER, 26);
    ColiJitTransformAttemptInfo failed_info;
    CHECK(coli_jit_transform_query(
              &service, failed_ticket, &failed_info) == 0 &&
          failed_info.state == COLI_JIT_ATTEMPT_QUEUED, 27);
    CHECK(coli_jit_transform_run_one(&service) == -1, 28);
    CHECK(coli_jit_transform_query(
              &service, failed_ticket, &failed_info) == 0 &&
          failed_info.state == COLI_JIT_ATTEMPT_FAILED &&
          failed_info.error == COLI_JIT_ERROR_VALIDATE &&
          coli_expert_residency_variant_state(
              &entry, failed_info.destination_variant_id) ==
              COLI_EXPERT_RESIDENCY_COLD &&
          coli_jit_transform_temp_budget_committed(&temp_budget) == 0 &&
          atomic_load(&entry.variants[source_variant].refs) == 0 &&
          coli_expert_residency_preferred_variant(&entry) ==
              (int)source_variant, 29);
    ColiExpertResidencyLease still_baseline;
    CHECK(coli_expert_residency_acquire_preferred(
              &entry, &still_baseline) == 1 &&
          still_baseline.variant_id == source_variant &&
          still_baseline.generation == source_generation, 30);
    CHECK(coli_expert_residency_release(&still_baseline) == 0, 31);
    ctx.fail_validate = 0;

    /* Separate scratch/staging budget rejects before any transform work and
     * rolls the destination reservation back to the baseline-only footprint. */
    ColiJitTransformTempBudget tiny_temp;
    ColiJitTransformService tiny_service;
    coli_jit_transform_temp_budget_init(&tiny_temp, 19);
    CHECK(coli_jit_transform_service_init(
              &tiny_service, &registry, &tiny_temp, &memory, &clock) == 0, 32);
    unsigned prepare_before_no_budget = ctx.prepare_calls;
    ColiJitTransformTicket no_budget_ticket;
    CHECK(coli_jit_transform_request(
              &tiny_service, &entry, source_variant, source_generation,
              &resident_budget, &rep_b, 1, COLI_JIT_PRIORITY_BACKGROUND,
              &no_budget_ticket) == COLI_JIT_REQUEST_NO_BUDGET &&
          ctx.prepare_calls == prepare_before_no_budget &&
          coli_jit_transform_temp_budget_committed(&tiny_temp) == 0 &&
          atomic_load(&resident_budget.committed_bytes) == sizeof(FixtureBlob) &&
          atomic_load(&resident_budget.reserved_bytes) == 0 &&
          atomic_load(&entry.variants[source_variant].refs) == 0, 33);

    /* Queued cancellation releases destination, temp budget and source lease. */
    ColiJitTransformTicket cancelled_ticket;
    CHECK(coli_jit_transform_request(
              &service, &entry, source_variant, source_generation,
              &resident_budget, &rep_b, 1, COLI_JIT_PRIORITY_BACKGROUND,
              &cancelled_ticket) == COLI_JIT_REQUEST_OWNER, 34);
    ColiJitTransformAttemptInfo cancelled_info;
    CHECK(coli_jit_transform_query(
              &service, cancelled_ticket, &cancelled_info) == 0 &&
          atomic_load(&entry.variants[source_variant].refs) == 1 &&
          coli_jit_transform_temp_budget_committed(&temp_budget) == 20, 35);
    CHECK(coli_jit_transform_cancel(&service, cancelled_ticket) == 1 &&
          coli_jit_transform_query(
              &service, cancelled_ticket, &cancelled_info) == 0 &&
          cancelled_info.state == COLI_JIT_ATTEMPT_CANCELLED &&
          cancelled_info.error == COLI_JIT_ERROR_CANCELLED &&
          coli_expert_residency_variant_state(
              &entry, cancelled_info.destination_variant_id) ==
              COLI_EXPERT_RESIDENCY_COLD &&
          atomic_load(&entry.variants[source_variant].refs) == 0 &&
          coli_jit_transform_temp_budget_committed(&temp_budget) == 0 &&
          coli_jit_transform_run_one(&service) == 0, 36);

    /* Disabled service performs no residency or transform mutation. */
    coli_jit_transform_service_set_enabled(&service, 0);
    unsigned prepare_before_disabled = ctx.prepare_calls;
    ColiJitTransformTicket disabled_ticket;
    CHECK(coli_jit_transform_request(
              &service, &entry, source_variant, source_generation,
              &resident_budget, &rep_b, 1, COLI_JIT_PRIORITY_BACKGROUND,
              &disabled_ticket) == COLI_JIT_REQUEST_DISABLED &&
          ctx.prepare_calls == prepare_before_disabled &&
          atomic_load(&entry.variants[source_variant].refs) == 0 &&
          coli_jit_transform_temp_budget_committed(&temp_budget) == 0, 37);
    coli_jit_transform_service_set_enabled(&service, 1);

    /* Reuse the physical source slot with a newer family generation. Requests
     * against the old source generation are rejected before target reservation. */
    uint64_t old_generation = source_generation;
    CHECK(coli_expert_residency_begin_variant_evict(
              &entry, source_variant) == 1 &&
          coli_expert_residency_finish_variant_evict(
              &entry, &resident_budget, source_variant) == 0, 38);
    uint32_t reused_source_variant = COLI_EXPERT_VARIANT_NONE;
    uint64_t reused_source_generation = 0;
    CHECK(publish_source(
              &entry, &resident_budget, &rep_a, &source,
              &reused_source_variant, &reused_source_generation) == 0 &&
          reused_source_variant == source_variant &&
          reused_source_generation > old_generation, 39);
    ColiJitTransformTicket stale_source_ticket;
    CHECK(coli_jit_transform_request(
              &service, &entry, reused_source_variant, old_generation,
              &resident_budget, &rep_b, 1, COLI_JIT_PRIORITY_BACKGROUND,
              &stale_source_ticket) == COLI_JIT_REQUEST_STALE_SOURCE &&
          atomic_load(&entry.variants[reused_source_variant].refs) == 0 &&
          atomic_load(&resident_budget.reserved_bytes) == 0, 40);

    /* A stale ticket cannot alias a reused service-attempt slot/generation. */
    ColiJitTransformTicket fresh_ticket;
    CHECK(coli_jit_transform_request(
              &service, &entry, reused_source_variant, reused_source_generation,
              &resident_budget, &rep_b, 1, COLI_JIT_PRIORITY_BACKGROUND,
              &fresh_ticket) == COLI_JIT_REQUEST_OWNER &&
          fresh_ticket.serial != cancelled_ticket.serial, 41);
    CHECK(coli_jit_transform_cancel(&service, cancelled_ticket) == -1 &&
          coli_jit_transform_cancel(&service, fresh_ticket) == 1 &&
          atomic_load(&entry.variants[reused_source_variant].refs) == 0 &&
          coli_jit_transform_temp_budget_committed(&temp_budget) == 0, 42);

    coli_jit_transform_telemetry_snapshot(&service, &stats);
    CHECK(stats.failed == 1 && stats.cancelled == 2 &&
          stats.reject_no_budget == 0 &&
          atomic_load(&tiny_service.telemetry.reject_no_budget) == 1 &&
          atomic_load(&temp_budget.peak_committed_bytes) <=
              temp_budget.capacity_bytes &&
          atomic_load(&tiny_temp.peak_committed_bytes) <=
              tiny_temp.capacity_bytes &&
          atomic_load(&resident_budget.peak_committed_bytes) <=
              resident_budget.capacity_bytes, 43);

    puts("PASS async JIT transform service + exact MXFP4 fixture repack");
    return 0;
}

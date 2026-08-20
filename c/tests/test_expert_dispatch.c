#include "../expert_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr, code) do { if (!(expr)) return (code); } while (0)

/* Fixture-only execution layouts. These deliberately are not production Apple
 * layout IDs; #131/#26/#34 own the eventual target-profile vocabulary. */
enum {
    FIXTURE_LAYOUT_A = 0x5101,
    FIXTURE_LAYOUT_B = 0x5102,
    FIXTURE_LAYOUT_UNSUPPORTED = 0x51ff,
};

typedef struct {
    unsigned char bytes[32];
} FixtureBlob;

static ColiRepresentationId fixture_rep(uint16_t layout, uint16_t kernel_abi) {
    ColiRepresentationId rep = {
        .math_format = COLI_CSF_MATH_MXFP4_E2M1,
        .scale_format = COLI_CSF_SCALE_UE8M0,
        .execution_layout = layout,
        .execution_layout_abi = 1,
        .kernel_abi = kernel_abi,
        .target_class = 0x5136,
        .group_size = 32,
        .scale_block_rows = 1,
        .scale_block_columns = 32,
    };
    return rep;
}

static int publish_variant(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        const ColiRepresentationId *rep,
        unsigned tier,
        void *physical,
        uint32_t *variant_id,
        uint64_t *generation) {
    ColiExpertRequestResult requested = coli_expert_residency_reserve_variant(
        entry, budget, rep, tier, sizeof(FixtureBlob), variant_id, generation);
    if (requested != COLI_EXPERT_REQUEST_LOAD_OWNER) return -1;
    return coli_expert_residency_publish_variant(
        entry, budget, *variant_id, *generation,
        sizeof(FixtureBlob), physical);
}

static int classify_fixture(
        void *context,
        const ColiExpertResidentView *view,
        uint32_t activation_contract,
        uint32_t geometry_class,
        uint32_t *backend_class,
        uint32_t *kernel_class) {
    (void)context;
    (void)activation_contract;
    (void)geometry_class;
    if (!view || !backend_class || !kernel_class) return -1;
    if (view->representation.execution_layout == FIXTURE_LAYOUT_A) {
        *backend_class = 1;
        *kernel_class = 11;
        return 1;
    }
    if (view->representation.execution_layout == FIXTURE_LAYOUT_B) {
        *backend_class = 1;
        *kernel_class = 22;
        return 1;
    }
    return 0;
}

static int executable_fixture(void *context, const ColiExpertResidentView *view) {
    uint32_t backend = 0, kernel = 0;
    return classify_fixture(context, view, 1, 1, &backend, &kernel);
}

typedef struct {
    unsigned prepare_calls;
    unsigned validate_calls;
} TransformContext;

static int estimate_transform(
        void *context,
        const ColiExpertResidentView *source,
        const ColiRepresentationId *target,
        ColiJitTransformEstimate *estimate) {
    (void)context;
    if (!source || !target || !estimate ||
        source->resident_bytes != sizeof(FixtureBlob))
        return -1;
    *estimate = (ColiJitTransformEstimate){
        .resident_bytes = sizeof(FixtureBlob),
        .allocation_bytes = sizeof(FixtureBlob),
        .scratch_bytes = 8,
        .staging_bytes = 4,
        .output_alignment = 1,
        .scratch_alignment = 1,
        .staging_alignment = 1,
    };
    return 0;
}

static int prepare_transform(
        void *context,
        const ColiExpertResidentView *source,
        const ColiRepresentationId *target,
        void *output,
        uint64_t output_bytes,
        void *scratch,
        uint64_t scratch_bytes,
        void *staging,
        uint64_t staging_bytes) {
    TransformContext *ctx = (TransformContext *)context;
    (void)target;
    if (!ctx || !source || !source->physical || !output ||
        output_bytes < sizeof(FixtureBlob) || !scratch || scratch_bytes < 8 ||
        !staging || staging_bytes < 4)
        return -1;
    ctx->prepare_calls++;
    const FixtureBlob *src = (const FixtureBlob *)source->physical;
    FixtureBlob *dst = (FixtureBlob *)output;
    for (size_t i = 0; i < sizeof(dst->bytes); ++i)
        dst->bytes[i] = src->bytes[sizeof(dst->bytes) - 1 - i];
    return 0;
}

static int validate_transform(
        void *context,
        const ColiExpertResidentView *source,
        const ColiRepresentationId *target,
        const void *output,
        uint64_t resident_bytes) {
    TransformContext *ctx = (TransformContext *)context;
    (void)target;
    if (!ctx || !source || !source->physical || !output ||
        resident_bytes != sizeof(FixtureBlob))
        return -1;
    ctx->validate_calls++;
    const FixtureBlob *src = (const FixtureBlob *)source->physical;
    const FixtureBlob *dst = (const FixtureBlob *)output;
    for (size_t i = 0; i < sizeof(dst->bytes); ++i) {
        if (dst->bytes[i] != src->bytes[sizeof(dst->bytes) - 1 - i])
            return -1;
    }
    return 0;
}

static void *fixture_alloc(
        void *context, ColiJitMemoryPurpose purpose,
        uint64_t bytes, uint64_t alignment) {
    (void)context;
    (void)purpose;
    (void)alignment;
    return malloc((size_t)bytes);
}

static void fixture_free(
        void *context, ColiJitMemoryPurpose purpose,
        void *memory, uint64_t bytes) {
    (void)context;
    (void)purpose;
    (void)bytes;
    free(memory);
}

static int find_batch_for_entry(
        const ColiHeterogeneousDispatchPlan *plan,
        const ColiExpertResidencyEntry *entry) {
    for (uint32_t i = 0; i < plan->batch_count; ++i) {
        if (plan->batches[i].entry == entry) return (int)i;
    }
    return -1;
}

int main(void) {
    ColiRepresentationId rep_a = fixture_rep(FIXTURE_LAYOUT_A, 1);
    ColiRepresentationId rep_b = fixture_rep(FIXTURE_LAYOUT_B, 2);
    ColiRepresentationId rep_bad = fixture_rep(FIXTURE_LAYOUT_UNSUPPORTED, 9);
    CHECK(coli_representation_known(&rep_a) &&
          coli_representation_known(&rep_b) &&
          coli_representation_exact_math_compatible(&rep_a, &rep_b), 1);

    /* Heterogeneous dispatch: class first, then expert-major rows inside class. */
    ColiExpertResidencyBudget dispatch_budget;
    coli_expert_residency_budget_init(&dispatch_budget, 2048);
    ColiExpertResidencyEntry e0, e1, e2, e3, e4;
    CHECK(coli_expert_residency_entry_init(&e0, (ColiExpertKey){0, 7}) == 0 &&
          coli_expert_residency_entry_init(&e1, (ColiExpertKey){0, 15}) == 0 &&
          coli_expert_residency_entry_init(&e2, (ColiExpertKey){0, 20}) == 0 &&
          coli_expert_residency_entry_init(&e3, (ColiExpertKey){0, 81}) == 0 &&
          coli_expert_residency_entry_init(&e4, (ColiExpertKey){0, 99}) == 0, 2);
    FixtureBlob blobs[7] = {0};
    uint32_t v0, v1, v2, v3_bad, v3_a, v4;
    uint64_t g0, g1, g2, g3_bad, g3_a, g4;
    CHECK(publish_variant(&e0, &dispatch_budget, &rep_a, COLI_EXPERT_TIER_UMA,
                          &blobs[0], &v0, &g0) == 0 &&
          publish_variant(&e1, &dispatch_budget, &rep_b, COLI_EXPERT_TIER_UMA,
                          &blobs[1], &v1, &g1) == 0 &&
          publish_variant(&e2, &dispatch_budget, &rep_a, COLI_EXPERT_TIER_UMA,
                          &blobs[2], &v2, &g2) == 0 &&
          coli_expert_residency_set_preferred(&e0, v0) == 0 &&
          coli_expert_residency_set_preferred(&e1, v1) == 0 &&
          coli_expert_residency_set_preferred(&e2, v2) == 0, 3);

    /* e3 prefers an unsupported representation but has a safe resident A. */
    CHECK(publish_variant(&e3, &dispatch_budget, &rep_bad, COLI_EXPERT_TIER_UMA,
                          &blobs[3], &v3_bad, &g3_bad) == 0 &&
          coli_expert_residency_set_preferred(&e3, v3_bad) == 0 &&
          publish_variant(&e3, &dispatch_budget, &rep_a, COLI_EXPERT_TIER_UMA,
                          &blobs[4], &v3_a, &g3_a) == 0 &&
          coli_expert_residency_preferred_variant(&e3) == (int)v3_bad, 4);
    CHECK(publish_variant(&e4, &dispatch_budget, &rep_bad, COLI_EXPERT_TIER_UMA,
                          &blobs[5], &v4, &g4) == 0 &&
          coli_expert_residency_set_preferred(&e4, v4) == 0, 5);

    ColiExpertRoutedWork work[] = {
        {&e0, 0, 0, 65536, 1, 1},
        {&e1, 0, 1, 32768, 1, 1},
        {&e0, 1, 0, 49152, 1, 1},
        {&e2, 0, 0, 65536, 1, 1},
        {&e1, 1, 1, 16384, 1, 1},
        {&e3, 2, 0, 65536, 1, 1},
    };
    ColiHeterogeneousDispatchPlan plan;
    CHECK(coli_expert_dispatch_plan_build(
              work, sizeof(work) / sizeof(work[0]), classify_fixture, NULL,
              COLI_HET_DISPATCH_STRICT, &plan) == COLI_HET_DISPATCH_READY &&
          plan.group_count == 2 && plan.batch_count == 4 &&
          plan.executable_route_count == 6 && plan.unresolved_route_count == 0, 6);

    int b0 = find_batch_for_entry(&plan, &e0);
    int b1 = find_batch_for_entry(&plan, &e1);
    int b2 = find_batch_for_entry(&plan, &e2);
    int b3 = find_batch_for_entry(&plan, &e3);
    CHECK(b0 >= 0 && b1 >= 0 && b2 >= 0 && b3 >= 0 &&
          plan.batches[b0].route_count == 2 &&
          plan.batches[b1].route_count == 2 &&
          plan.batches[b2].route_count == 1 &&
          plan.batches[b3].route_count == 1 &&
          plan.batches[b0].group_index == plan.batches[b2].group_index &&
          plan.batches[b0].group_index == plan.batches[b3].group_index &&
          plan.batches[b0].group_index != plan.batches[b1].group_index, 7);
    CHECK(coli_representation_equal(
              &plan.batches[b3].lease.representation, &rep_a) &&
          coli_expert_residency_preferred_variant(&e3) == (int)v3_bad, 8);

    /* Every expert's rows are contiguous in route_order and still point back to
     * the original weighted/scatter association rather than a rewritten row. */
    for (uint32_t b = 0; b < plan.batch_count; ++b) {
        const ColiExpertDispatchBatch *batch = &plan.batches[b];
        for (uint32_t j = 0; j < batch->route_count; ++j) {
            uint32_t route_index = plan.route_order[batch->route_offset + j];
            CHECK(route_index < sizeof(work) / sizeof(work[0]) &&
                  work[route_index].entry == batch->entry &&
                  work[route_index].route_weight_q16 != 0, 9);
        }
    }
    ColiHeterogeneousDispatchTelemetry dispatch_stats = {0};
    coli_expert_dispatch_telemetry_record(&dispatch_stats, &plan);
    CHECK(dispatch_stats.heterogeneous_routed_blocks == 1 &&
          dispatch_stats.representation_groups == 2 &&
          dispatch_stats.routed_rows == 6 &&
          dispatch_stats.expert_batches == 4 &&
          dispatch_stats.kernel_invocations == 2, 10);
    coli_expert_dispatch_plan_release(&plan);
    CHECK(atomic_load(&e0.variants[v0].refs) == 0 &&
          atomic_load(&e1.variants[v1].refs) == 0 &&
          atomic_load(&e3.variants[v3_a].refs) == 0, 11);

    ColiExpertRoutedWork unsupported[] = {{&e4, 9, 0, 65536, 1, 1}};
    CHECK(coli_expert_dispatch_plan_build(
              unsupported, 1, classify_fixture, NULL,
              COLI_HET_DISPATCH_STRICT, &plan) ==
              COLI_HET_DISPATCH_STRICT_UNSUPPORTED &&
          atomic_load(&e4.variants[v4].refs) == 0, 12);
    CHECK(coli_expert_dispatch_plan_build(
              unsupported, 1, classify_fixture, NULL,
              COLI_HET_DISPATCH_ALLOW_BASELINE_FALLBACK, &plan) ==
              COLI_HET_DISPATCH_NEEDS_FALLBACK &&
          plan.unresolved_route_count == 1 && plan.batch_count == 0, 13);
    coli_expert_dispatch_plan_release(&plan);

    /* Promotion economics consume bounded reuse only; no disk miss value enters
     * the A->B incremental score. */
    ColiExpertActivationEntry activity_storage[8];
    ColiExpertActivationTracker tracker;
    CHECK(coli_expert_activation_init(&tracker, activity_storage, 8) == 0, 14);
    ColiExpertKey hot_key = {3, 5};
    for (uint64_t epoch = 1; epoch <= 8; ++epoch) {
        CHECK(coli_expert_activation_observe(
                  &tracker, (ColiExpertActivationSample){
                      .key = hot_key,
                      .phase = COLI_EXPERT_PHASE_DECODE,
                      .multiplicity = 8,
                      .epoch = epoch,
                  }) >= 0, 15);
    }
    const ColiExpertActivationEntry *hot =
        coli_expert_activation_find_const(&tracker, hot_key);
    CHECK(hot != NULL, 16);
    ColiExpertResidencyPolicyConfig reuse_cfg =
        coli_expert_residency_policy_default();
    ColiExpertPromotionPolicyConfig promotion_cfg =
        coli_expert_promotion_policy_default();
    ColiExpertPromotionHistory history;
    coli_expert_promotion_history_init(&history);
    ColiExpertPromotionInputs inputs = {
        .activation = hot,
        .current_epoch = 8,
        .current_exec_ns = 100,
        .target_exec_ns = 60,
        .transform_prepare_ns = 2000,
        .expected_extra_dispatch_ns = 200,
        .incremental_resident_bytes = 64,
        .shadow_headroom_bytes = 256,
        .memory_pressure_percent = 20,
    };
    ColiExpertPromotionEvaluation evaluation = coli_expert_promotion_evaluate(
        &inputs, &history, &reuse_cfg, &promotion_cfg);
    CHECK(evaluation.decision == COLI_PROMOTION_SELECTED &&
          evaluation.expected_future_uses > 0 &&
          evaluation.per_use_saved_ns == 40 &&
          evaluation.gross_time_gain_ns ==
              evaluation.expected_future_uses * 40 &&
          evaluation.net_time_gain_ns ==
              evaluation.gross_time_gain_ns - 2200 &&
          evaluation.promotion_value_per_byte ==
              evaluation.net_time_gain_ns / 64, 17);
    ColiResourceCandidate promotion_resource;
    CHECK(coli_expert_promotion_resource_candidate(
              &evaluation, 77, &promotion_resource) == 0 &&
          promotion_resource.kind == COLI_RESOURCE_OTHER &&
          promotion_resource.expected_bytes_avoided == 0 &&
          promotion_resource.expected_exposed_ns_avoided ==
              evaluation.net_time_gain_ns, 18);

    ColiExpertPromotionInputs rejected = inputs;
    rejected.target_exec_ns = 95;
    CHECK(coli_expert_promotion_evaluate(
              &rejected, &history, &reuse_cfg, &promotion_cfg).decision ==
              COLI_PROMOTION_REJECT_HYSTERESIS, 19);
    rejected = inputs;
    rejected.shadow_headroom_bytes = 32;
    CHECK(coli_expert_promotion_evaluate(
              &rejected, &history, &reuse_cfg, &promotion_cfg).decision ==
              COLI_PROMOTION_REJECT_NO_BUDGET, 20);
    rejected = inputs;
    rejected.memory_pressure_percent = 95;
    CHECK(coli_expert_promotion_evaluate(
              &rejected, &history, &reuse_cfg, &promotion_cfg).decision ==
              COLI_PROMOTION_REJECT_MEMORY_PRESSURE, 21);
    history.last_switch_epoch = 4;
    CHECK(coli_expert_promotion_evaluate(
              &inputs, &history, &reuse_cfg, &promotion_cfg).decision ==
              COLI_PROMOTION_REJECT_AGE, 22);
    history.last_switch_epoch = 0;
    history.last_transform_prepare_ns = 5000;
    history.realized_saved_ns = 100;
    CHECK(coli_expert_promotion_evaluate(
              &inputs, &history, &reuse_cfg, &promotion_cfg).decision ==
              COLI_PROMOTION_REJECT_UNAMORTIZED, 23);
    coli_expert_promotion_history_init(&history);

    ColiExpertActivationEntry cold_storage[2];
    ColiExpertActivationTracker cold_tracker;
    ColiExpertKey cold_key = {3, 6};
    CHECK(coli_expert_activation_init(&cold_tracker, cold_storage, 2) == 0 &&
          coli_expert_activation_observe(
              &cold_tracker, (ColiExpertActivationSample){
                  .key = cold_key,
                  .phase = COLI_EXPERT_PHASE_DECODE,
                  .multiplicity = 1,
                  .epoch = 1,
              }) >= 0, 24);
    ColiExpertPromotionInputs cold_inputs = inputs;
    cold_inputs.activation = coli_expert_activation_find_const(
        &cold_tracker, cold_key);
    cold_inputs.current_epoch = 1;
    CHECK(coli_expert_promotion_evaluate(
              &cold_inputs, &history, &reuse_cfg, &promotion_cfg).decision ==
              COLI_PROMOTION_REJECT_NO_REUSE, 25);

    /* End-to-end zero-downtime promotion through #135. Transform publication
     * does not switch preferred; #136 verifies destination residency identity. */
    ColiExpertResidencyBudget promotion_budget;
    coli_expert_residency_budget_init(&promotion_budget, 512);
    ColiExpertResidencyEntry promoted;
    CHECK(coli_expert_residency_entry_init(&promoted, hot_key) == 0, 26);
    FixtureBlob source;
    for (size_t i = 0; i < sizeof(source.bytes); ++i)
        source.bytes[i] = (unsigned char)(i * 3u + 1u);
    uint32_t source_variant;
    uint64_t source_generation;
    CHECK(publish_variant(
              &promoted, &promotion_budget, &rep_a, COLI_EXPERT_TIER_UMA,
              &source, &source_variant, &source_generation) == 0 &&
          coli_expert_residency_set_preferred(
              &promoted, source_variant) == 0, 27);

    ColiRepresentationTransformRegistry registry;
    ColiJitTransformTempBudget temp_budget;
    ColiJitTransformService service;
    TransformContext transform_ctx = {0};
    coli_jit_transform_registry_init(&registry);
    ColiRepresentationTransformOps ops = {
        .source = rep_a,
        .target = rep_b,
        .transform_abi = 7,
        .transform_class = COLI_JIT_TRANSFORM_EXACT,
        .target_tier_mask = COLI_EXPERT_TIER_UMA,
        .backend_tag = 136,
        .context = &transform_ctx,
        .estimate = estimate_transform,
        .prepare = prepare_transform,
        .validate = validate_transform,
    };
    CHECK(coli_jit_transform_registry_register(&registry, &ops) == 1, 28);
    coli_jit_transform_temp_budget_init(&temp_budget, 64);
    ColiJitTransformMemoryOps memory = {
        .context = NULL,
        .allocate = fixture_alloc,
        .free = fixture_free,
    };
    CHECK(coli_jit_transform_service_init(
              &service, &registry, &temp_budget, &memory, NULL) == 0, 29);

    ColiExpertResidencyLease old_lease;
    CHECK(coli_expert_residency_acquire_variant(
              &promoted, source_variant, &old_lease) == 1, 30);
    ColiExpertPromotionPending pending;
    CHECK(coli_expert_promotion_request(
              &service, &promoted, source_variant, source_generation,
              &promotion_budget, &rep_b, 7, COLI_JIT_PRIORITY_BACKGROUND,
              &evaluation, &pending) == COLI_JIT_REQUEST_OWNER &&
          coli_expert_residency_preferred_variant(&promoted) ==
              (int)source_variant &&
          old_lease.generation == source_generation &&
          coli_representation_equal(&old_lease.representation, &rep_a), 31);
    CHECK(coli_jit_transform_run_one(&service) == 1 &&
          transform_ctx.prepare_calls == 1 && transform_ctx.validate_calls == 1 &&
          coli_expert_residency_preferred_variant(&promoted) ==
              (int)source_variant, 32);

    ColiExpertPromotionTelemetry promotion_stats = {0};
    CHECK(coli_expert_promotion_finalize(
              &service, &pending, 100, 2000,
              executable_fixture, NULL, &history, &promotion_stats) == 1, 33);
    int promoted_variant = coli_expert_residency_preferred_variant(&promoted);
    CHECK(promoted_variant >= 0 && promoted_variant != (int)source_variant &&
          old_lease.variant_id == source_variant &&
          old_lease.generation == source_generation &&
          coli_representation_equal(&old_lease.representation, &rep_a) &&
          atomic_load(&promoted.variants[source_variant].refs) == 1 &&
          promotion_stats.preferred_variant_switches == 1, 34);

    ColiExpertResidencyLease new_lease;
    CHECK(coli_expert_residency_acquire_preferred(&promoted, &new_lease) == 1 &&
          new_lease.variant_id == (uint32_t)promoted_variant &&
          coli_representation_equal(&new_lease.representation, &rep_b), 35);
    void *published_output = new_lease.physical;
    uint64_t promoted_generation = new_lease.generation;
    CHECK(published_output != NULL &&
          coli_expert_residency_release(&new_lease) == 0 &&
          coli_expert_residency_release(&old_lease) == 0, 36);

    /* The same decayed reuse signal eventually permits demotion; hard pressure
     * may bypass the minimum-age hold. Demotion is a preferred switch only. */
    CHECK(coli_expert_promotion_should_demote(
              hot, 4096, 0, &history, &reuse_cfg, &promotion_cfg) == 1, 37);
    CHECK(coli_expert_promotion_demote_verified(
              &promoted, source_variant, source_generation, &rep_a,
              4096, executable_fixture, NULL, &history, &promotion_stats) == 1 &&
          coli_expert_residency_preferred_variant(&promoted) ==
              (int)source_variant && promotion_stats.variant_demotions == 1, 38);

    /* Known #135/#57 ownership gap: capture the fixture allocation before the
     * generic eviction clears physical, then free it explicitly in this test. */
    CHECK(coli_expert_residency_begin_variant_evict(
              &promoted, (uint32_t)promoted_variant) == 1 &&
          coli_expert_residency_finish_variant_evict(
              &promoted, &promotion_budget,
              (uint32_t)promoted_variant) == 0, 39);
    free(published_output);
    CHECK(atomic_load(&promotion_budget.resident_bytes) == sizeof(FixtureBlob) &&
          promoted_generation > source_generation, 40);

    puts("PASS heterogeneous dispatch + profile-guided hot-variant promotion");
    return 0;
}

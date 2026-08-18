#include "../resource_planner.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MIB UINT64_C(1048576)

static ColiResourceCandidate candidate(ColiResourceKind kind, uint32_t id,
                                       uint64_t resident_mib,
                                       uint64_t reuse_weight,
                                       uint64_t miss_mib,
                                       uint64_t miss_ms) {
    ColiResourceBenefitEstimate estimate = {
        .kind = kind,
        .id = id,
        .resident_bytes = resident_mib * MIB,
        .reuse_weight = reuse_weight,
        .bytes_per_miss = miss_mib * MIB,
        .exposed_ns_per_miss = miss_ms * UINT64_C(1000000),
    };
    ColiResourceCandidate out;
    assert(coli_resource_candidate_from_benefit(&estimate, &out) == 0);
    return out;
}

static void test_benefit_bridge_and_saturation(void) {
    ColiResourceBenefitEstimate estimate = {
        .kind = COLI_RESOURCE_PERSISTENT_EXPERT,
        .id = 7,
        .resident_bytes = 13 * MIB,
        .reuse_weight = 6,
        .bytes_per_miss = 13 * MIB,
        .exposed_ns_per_miss = UINT64_C(12000000),
    };
    ColiResourceCandidate out;
    assert(coli_resource_candidate_from_benefit(&estimate, &out) == 0);
    assert(out.kind == COLI_RESOURCE_PERSISTENT_EXPERT);
    assert(out.id == 7);
    assert(out.resident_bytes == 13 * MIB);
    assert(out.expected_bytes_avoided == 78 * MIB);
    assert(out.expected_exposed_ns_avoided == UINT64_C(72000000));

    estimate.reuse_weight = UINT64_MAX;
    estimate.bytes_per_miss = 2;
    estimate.exposed_ns_per_miss = 2;
    assert(coli_resource_candidate_from_benefit(&estimate, &out) == 0);
    assert(out.expected_bytes_avoided == UINT64_MAX);
    assert(out.expected_exposed_ns_avoided == UINT64_MAX);

    estimate.resident_bytes = 0;
    assert(coli_resource_candidate_from_benefit(&estimate, &out) != 0);
}

static void test_mixed_uma_competition(void) {
    ColiResourcePlan plan;
    coli_resource_plan_init(&plan);

    /* 54 MiB UMA total; 9 MiB is mandatory sequence/scratch/I/O state. The
     * optional 45 MiB must be shared by dense and expert candidates rather than
     * pre-partitioned by engine-local cache knobs. */
    assert(coli_resource_plan_set_budget(
        &plan, COLI_RESOURCE_TIER_UMA, 54 * MIB) == 0);
    assert(coli_resource_plan_reserve_mandatory(
        &plan, COLI_RESOURCE_TIER_UMA, 9 * MIB) == 0);

    ColiResourceCandidate candidates[] = {
        /* Dense tensor: useful every horizon, but lower exposed-time/byte than
         * the hot expert below. */
        candidate(COLI_RESOURCE_DENSE_TENSOR, 1, 32, 4, 32, 8),
        /* Hot routed expert: six weighted future uses over the same horizon. */
        candidate(COLI_RESOURCE_PERSISTENT_EXPERT, 2, 13, 6, 13, 12),
        /* Cold expert: valid, but lower benefit/byte than dense and should lose
         * once the hot expert + dense tensor consume the 45 MiB envelope. */
        candidate(COLI_RESOURCE_PERSISTENT_EXPERT, 3, 13, 1, 13, 2),
    };
    unsigned char selected[3];
    ColiResourceSelection selection;

    assert(coli_resource_plan_select_optional(
        &plan, COLI_RESOURCE_TIER_UMA,
        candidates, 3, COLI_RESOURCE_VALUE_EXPOSED_NS,
        selected, &selection) == 0);
    assert(selected[0] == 1);
    assert(selected[1] == 1);
    assert(selected[2] == 0);
    assert(selection.selected_count == 2);
    assert(selection.selected_resident_bytes == 45 * MIB);
    assert(plan.tier[COLI_RESOURCE_TIER_UMA].mandatory_bytes == 9 * MIB);
    assert(plan.tier[COLI_RESOURCE_TIER_UMA].selected_optional_bytes == 45 * MIB);
    assert(plan.tier[COLI_RESOURCE_TIER_UMA].free_bytes == 0);
    assert(coli_resource_plan_committed(
        &plan, COLI_RESOURCE_TIER_UMA) == 54 * MIB);

    assert(coli_resource_plan_release_optional(
        &plan, COLI_RESOURCE_TIER_UMA, 45 * MIB) == 0);
    assert(plan.tier[COLI_RESOURCE_TIER_UMA].free_bytes == 45 * MIB);
    assert(coli_resource_plan_committed(
        &plan, COLI_RESOURCE_TIER_UMA) == 9 * MIB);
}

static void test_optional_selection_fail_clean(void) {
    ColiResourcePlan plan;
    coli_resource_plan_init(&plan);
    assert(coli_resource_plan_set_budget(
        &plan, COLI_RESOURCE_TIER_UMA, 32 * MIB) == 0);
    assert(coli_resource_plan_reserve_mandatory(
        &plan, COLI_RESOURCE_TIER_UMA, 8 * MIB) == 0);

    ColiResourceCandidate item =
        candidate(COLI_RESOURCE_DENSE_TENSOR, 4, 16, 1, 16, 1);
    unsigned char selected = 0xff;
    ColiResourceSelection selection;
    memset(&selection, 0xff, sizeof(selection));

    uint64_t before = coli_resource_plan_committed(
        &plan, COLI_RESOURCE_TIER_UMA);
    assert(coli_resource_plan_select_optional(
        &plan, COLI_RESOURCE_TIER_UMA, &item, 1,
        (ColiResourceValueMode)99, &selected, &selection) != 0);
    assert(selected == 0);
    assert(selection.selected_count == 0);
    assert(selection.selected_resident_bytes == 0);
    assert(coli_resource_plan_committed(
        &plan, COLI_RESOURCE_TIER_UMA) == before);
    assert(plan.tier[COLI_RESOURCE_TIER_UMA].selected_optional_bytes == 0);
    assert(plan.tier[COLI_RESOURCE_TIER_UMA].free_bytes == 24 * MIB);
}

static void test_bytes_mode_and_zero_reuse(void) {
    ColiResourceCandidate candidates[] = {
        candidate(COLI_RESOURCE_DENSE_TENSOR, 1, 16, 2, 16, 1),
        candidate(COLI_RESOURCE_PREFIX_HOT, 2, 16, 0, 64, 20),
    };
    unsigned char selected[2] = {9, 9};
    ColiResourceSelection selection;
    assert(coli_resource_select(candidates, 2, 16 * MIB,
                                COLI_RESOURCE_VALUE_BYTES,
                                selected, &selection) == 0);
    assert(selected[0] == 1);
    assert(selected[1] == 0);
    assert(selection.selected_count == 1);
}

static void test_cuda_tiers_stay_independent(void) {
    ColiResourcePlan plan;
    coli_resource_plan_init(&plan);
    assert(coli_resource_plan_set_budget(
        &plan, COLI_RESOURCE_TIER_PINNED_HOST, 64 * MIB) == 0);
    assert(coli_resource_plan_set_budget(
        &plan, COLI_RESOURCE_TIER_DEVICE, 96 * MIB) == 0);
    assert(coli_resource_plan_reserve_mandatory(
        &plan, COLI_RESOURCE_TIER_PINNED_HOST, 16 * MIB) == 0);
    assert(coli_resource_plan_reserve_mandatory(
        &plan, COLI_RESOURCE_TIER_DEVICE, 32 * MIB) == 0);
    assert(coli_resource_plan_commit_optional(
        &plan, COLI_RESOURCE_TIER_DEVICE, 48 * MIB) == 0);

    assert(coli_resource_plan_committed(
        &plan, COLI_RESOURCE_TIER_PINNED_HOST) == 16 * MIB);
    assert(coli_resource_plan_committed(
        &plan, COLI_RESOURCE_TIER_DEVICE) == 80 * MIB);
}

int main(void) {
    test_benefit_bridge_and_saturation();
    test_mixed_uma_competition();
    test_optional_selection_fail_clean();
    test_bytes_mode_and_zero_reuse();
    test_cuda_tiers_stay_independent();
    puts("test_resource_planner: ok");
    return 0;
}

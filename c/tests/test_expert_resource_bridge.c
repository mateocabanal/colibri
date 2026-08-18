#include "../expert_residency_policy.h"

#include <assert.h>
#include <stdio.h>

#define MIB UINT64_C(1048576)

int main(void) {
    ColiExpertActivationEntry entries[16];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, entries, 16) == 0);

    ColiExpertActivationSample sample = {
        .key = {2, 7},
        .phase = COLI_EXPERT_PHASE_DECODE,
        .multiplicity = 8,
        .epoch = 8,
    };
    assert(coli_expert_activation_observe(&tracker, sample) >= 0);
    const ColiExpertActivationEntry *entry =
        coli_expert_activation_find_const(&tracker, sample.key);
    assert(entry);

    ColiExpertResidencyPolicyConfig policy =
        coli_expert_residency_policy_default();
    ColiExpertResidencyCandidate expert =
        coli_expert_residency_policy_candidate(
            entry, 8, 13 * MIB, 12000, 0, &policy);
    assert(expert.reuse_weight > 1);

    ColiResourceBenefitEstimate estimate;
    assert(coli_expert_residency_policy_resource_estimate(
        &expert, 23, 12 * MIB, UINT64_C(9000000), &estimate) == 0);
    assert(estimate.kind == COLI_RESOURCE_PERSISTENT_EXPERT);
    assert(estimate.id == 23);
    assert(estimate.resident_bytes == 13 * MIB);
    assert(estimate.reuse_weight == expert.reuse_weight);
    assert(estimate.bytes_per_miss == 12 * MIB);
    assert(estimate.exposed_ns_per_miss == UINT64_C(9000000));

    ColiResourceCandidate routed;
    assert(coli_resource_candidate_from_benefit(&estimate, &routed) == 0);
    assert(routed.expected_bytes_avoided ==
           expert.reuse_weight * 12 * MIB);
    assert(routed.expected_exposed_ns_avoided ==
           expert.reuse_weight * UINT64_C(9000000));

    /* A ratio-1 dense baseline must lose to this sufficiently hot expert in
     * bytes mode, which is the V4 adapter's current apples-to-apples metric. */
    ColiResourceBenefitEstimate dense_estimate = {
        .kind = COLI_RESOURCE_DENSE_TENSOR,
        .id = 0,
        .resident_bytes = 13 * MIB,
        .reuse_weight = 1,
        .bytes_per_miss = 13 * MIB,
    };
    ColiResourceCandidate candidates[2];
    assert(coli_resource_candidate_from_benefit(
        &dense_estimate, &candidates[0]) == 0);
    candidates[1] = routed;
    unsigned char selected[2] = {0, 0};
    ColiResourceSelection selection;
    assert(coli_resource_select(
        candidates, 2, 13 * MIB, COLI_RESOURCE_VALUE_BYTES,
        selected, &selection) == 0);
    assert(selected[0] == 0);
    assert(selected[1] == 1);
    assert(selection.selected_resident_bytes == 13 * MIB);

    puts("test_expert_resource_bridge: ok");
    return 0;
}

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../moe_adaptive_residency.h"

static int never_resident(void *context, ColiExpertKey key) {
    (void)context;
    (void)key;
    return 0;
}

static void test_global_selection_and_warmup(void) {
    ColiMoeAdaptiveResidency state;
    assert(coli_moe_adaptive_init(&state, 2, 4) == 0);

    for (uint64_t epoch = 1; epoch <= 8; epoch++) {
        int a[] = {1};
        int b[] = {2};
        assert(coli_moe_adaptive_observe_routes(
            &state, 0, a, 1, COLI_EXPERT_PHASE_DECODE, epoch) == 0);
        assert(coli_moe_adaptive_observe_routes(
            &state, 1, b, 1, COLI_EXPERT_PHASE_DECODE, epoch) == 0);
    }
    assert(coli_moe_adaptive_should_replan(&state));

    ColiResourceSelection selection;
    assert(coli_moe_adaptive_select_experts(
        &state, 2 * UINT64_C(4096), UINT64_C(4096), UINT64_C(4096),
        0, COLI_RESOURCE_VALUE_BYTES, never_resident, NULL, &selection) == 1);
    assert(selection.selected_count == 2);
    assert(coli_moe_adaptive_selected(&state, (ColiExpertKey){0, 1}));
    assert(coli_moe_adaptive_selected(&state, (ColiExpertKey){1, 2}));
    assert(state.replan_count == 1);

    /* Warmup cadence is generic and token-based: one new token can immediately
     * update the selection without waiting a 64-token decay quantum. */
    int c[] = {3};
    assert(coli_moe_adaptive_observe_routes(
        &state, 0, c, 1, COLI_EXPERT_PHASE_DECODE, 9) == 0);
    assert(coli_moe_adaptive_should_replan(&state));

    coli_moe_adaptive_destroy(&state);
}

static void test_hotness_fills_common_budget(void) {
    ColiMoeAdaptiveResidency state;
    assert(coli_moe_adaptive_init(&state, 1, 4) == 0);

    /* Expert 1 has more unweighted reuse and therefore wins the common-value
     * allocation ordering. Expert 2 has fewer observations but they are decode
     * observations, so decode_weight=4 makes it the hotter WHICH-expert choice.
     * The expert-only convenience path must use common reuse for HOW MANY bytes
     * and hotness for WHICH expert fills those bytes. */
    for (uint64_t epoch = 1; epoch <= 8; epoch++) {
        int prefill[] = {1};
        assert(coli_moe_adaptive_observe_routes(
            &state, 0, prefill, 1, COLI_EXPERT_PHASE_PREFILL, epoch) == 0);
    }
    for (uint64_t epoch = 6; epoch <= 8; epoch++) {
        int decode[] = {2};
        assert(coli_moe_adaptive_observe_routes(
            &state, 0, decode, 1, COLI_EXPERT_PHASE_DECODE, epoch) == 0);
    }

    const ColiExpertActivationEntry *prefill_entry =
        coli_expert_activation_find_const(&state.tracker, (ColiExpertKey){0, 1});
    const ColiExpertActivationEntry *decode_entry =
        coli_expert_activation_find_const(&state.tracker, (ColiExpertKey){0, 2});
    assert(prefill_entry && decode_entry);
    assert(coli_expert_residency_policy_reuse_weight(
               prefill_entry, 8, &state.policy) >
           coli_expert_residency_policy_reuse_weight(
               decode_entry, 8, &state.policy));
    assert(coli_expert_residency_policy_hotness(
               decode_entry, 8, 0, &state.policy) >
           coli_expert_residency_policy_hotness(
               prefill_entry, 8, 0, &state.policy));

    ColiResourceSelection selection;
    assert(coli_moe_adaptive_select_experts(
        &state, UINT64_C(4096), UINT64_C(4096), UINT64_C(4096),
        0, COLI_RESOURCE_VALUE_BYTES, never_resident, NULL, &selection) == 1);
    assert(selection.selected_count == 1);
    assert(!coli_moe_adaptive_selected(&state, (ColiExpertKey){0, 1}));
    assert(coli_moe_adaptive_selected(&state, (ColiExpertKey){0, 2}));

    coli_moe_adaptive_destroy(&state);
}

static void test_prefill_union_equivalence(void) {
    ColiMoeAdaptiveResidency per_token, unioned;
    assert(coli_moe_adaptive_init(&per_token, 1, 8) == 0);
    assert(coli_moe_adaptive_init(&unioned, 1, 8) == 0);

    /* Qwen-style: publish the logical top-k rows before its physical expert
     * arena unions them. */
    for (uint64_t epoch = 1; epoch <= 4; epoch++) {
        int routes[] = {2, 5};
        assert(coli_moe_adaptive_observe_routes(
            &per_token, 0, routes, 2, COLI_EXPERT_PHASE_PREFILL, epoch) == 0);
    }

    /* A V4-style union adapter may aggregate multiplicity, provided it reports
     * the same logical count at the batch-end token epoch. */
    ColiExpertActivationSample samples[] = {
        {{0, 2}, COLI_EXPERT_PHASE_PREFILL, 4, 4},
        {{0, 5}, COLI_EXPERT_PHASE_PREFILL, 4, 4},
    };
    assert(coli_moe_adaptive_observe_samples(&unioned, samples, 2) == 0);

    const ColiExpertActivationEntry *a2 = coli_expert_activation_find_const(
        &per_token.tracker, (ColiExpertKey){0, 2});
    const ColiExpertActivationEntry *b2 = coli_expert_activation_find_const(
        &unioned.tracker, (ColiExpertKey){0, 2});
    const ColiExpertActivationEntry *a5 = coli_expert_activation_find_const(
        &per_token.tracker, (ColiExpertKey){0, 5});
    const ColiExpertActivationEntry *b5 = coli_expert_activation_find_const(
        &unioned.tracker, (ColiExpertKey){0, 5});
    assert(a2 && b2 && a5 && b5);
    assert(a2->logical_activations == b2->logical_activations);
    assert(a5->logical_activations == b5->logical_activations);
    assert(coli_expert_residency_policy_reuse_weight(
        a2, 4, &per_token.policy) ==
        coli_expert_residency_policy_reuse_weight(b2, 4, &unioned.policy));

    coli_moe_adaptive_destroy(&unioned);
    coli_moe_adaptive_destroy(&per_token);
}

static void test_mixed_global_plan_apply(void) {
    ColiMoeAdaptiveResidency state;
    assert(coli_moe_adaptive_init(&state, 1, 4) == 0);
    int routes[] = {1};
    for (uint64_t epoch = 1; epoch <= 8; epoch++)
        assert(coli_moe_adaptive_observe_routes(
            &state, 0, routes, 1, COLI_EXPERT_PHASE_DECODE, epoch) == 0);

    ColiResourceCandidate candidates[4] = {0};
    size_t count = 0;
    candidates[count++] = (ColiResourceCandidate){
        .kind = COLI_RESOURCE_DENSE_TENSOR,
        .id = 77,
        .resident_bytes = 4096,
        .expected_bytes_avoided = 4096 * 100,
    };
    count = coli_moe_adaptive_append_candidates(
        &state, candidates, count, 4, 4096, 4096, 0,
        never_resident, NULL);
    assert(count == 2);

    unsigned char selected[4] = {1, 1, 0, 0};
    coli_moe_adaptive_apply_selection(&state, candidates, selected, count);
    /* Exact external-selection mode ignores dense candidates and projects only
     * the explicitly selected expert id into the shared selected map. */
    assert(coli_moe_adaptive_selected(&state, (ColiExpertKey){0, 1}));

    coli_moe_adaptive_destroy(&state);
}

int main(void) {
    test_global_selection_and_warmup();
    test_hotness_fills_common_budget();
    test_prefill_union_equivalence();
    test_mixed_global_plan_apply();
    puts("generic MoE adaptive residency: ok");
    return 0;
}

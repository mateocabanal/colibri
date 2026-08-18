#include "../expert_residency_policy.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    ColiExpertActivationTracker tracker;
    ColiExpertActivationEntry entries[16];
} FakeState;

static void fake_observe(ColiExpertStore *store,
                         const struct ColiExpertActivationSample *samples,
                         size_t count) {
    FakeState *state = (FakeState *)store->state;
    coli_expert_activation_observe_many(
        &state->tracker, (const ColiExpertActivationSample *)samples, count);
}

static const ColiExpertStoreOps fake_ops = {
    .observe_activations = fake_observe,
};

static ColiExpertActivationEntry *observe(
    ColiExpertActivationTracker *tracker,
    int layer, int expert, ColiExpertPhase phase,
    uint64_t multiplicity, uint64_t epoch) {
    ColiExpertActivationSample sample = {
        .key = {layer, expert},
        .phase = phase,
        .multiplicity = multiplicity,
        .epoch = epoch,
    };
    assert(coli_expert_activation_observe(tracker, sample) >= 0);
    ColiExpertActivationEntry *entry = coli_expert_activation_find(
        tracker, sample.key);
    assert(entry);
    return entry;
}

static void test_store_observer_contract(void) {
    FakeState state;
    memset(&state, 0, sizeof(state));
    assert(coli_expert_activation_init(
        &state.tracker, state.entries, 16) == 0);
    ColiExpertStore store = {.ops = &fake_ops, .state = &state};
    ColiExpertActivationSample samples[] = {
        {{3, 7}, COLI_EXPERT_PHASE_PREFILL, 11, 1},
        {{3, 7}, COLI_EXPERT_PHASE_DECODE, 2, 2},
    };
    coli_expert_observe_activations(&store, samples, 2);
    const ColiExpertActivationEntry *entry = coli_expert_activation_find_const(
        &state.tracker, (ColiExpertKey){3, 7});
    assert(entry);
    assert(entry->prefill_activations == 11);
    assert(entry->decode_activations == 2);
    assert(entry->logical_activations == 13);

    /* Unsupported stores ignore policy signals rather than affecting inference. */
    ColiExpertStore no_observer = {0};
    coli_expert_observe_activations(&no_observer, samples, 2);
}

static void test_frequency_beats_pure_recency(void) {
    ColiExpertActivationEntry entries[16];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, entries, 16) == 0);
    ColiExpertActivationEntry *frequent = observe(
        &tracker, 1, 10, COLI_EXPERT_PHASE_DECODE, 1000, 1);
    ColiExpertActivationEntry *recent = observe(
        &tracker, 2, 20, COLI_EXPERT_PHASE_DECODE, 1, 129);
    ColiExpertResidencyPolicyConfig config =
        coli_expert_residency_policy_default();
    ColiExpertResidencyCandidate a = coli_expert_residency_policy_candidate(
        frequent, 129, 13 * 1024 * 1024, 16000, 0, &config);
    ColiExpertResidencyCandidate b = coli_expert_residency_policy_candidate(
        recent, 129, 13 * 1024 * 1024, 16000, 0, &config);
    assert(a.score > b.score);
    assert(coli_expert_residency_policy_compare(&a, &b) > 0);
}

static void test_stale_frequency_decays(void) {
    ColiExpertActivationEntry entries[16];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, entries, 16) == 0);
    ColiExpertActivationEntry *stale = observe(
        &tracker, 4, 1, COLI_EXPERT_PHASE_DECODE, 1000, 1);
    ColiExpertActivationEntry *recent = observe(
        &tracker, 4, 2, COLI_EXPERT_PHASE_DECODE, 1, 4097);
    ColiExpertResidencyPolicyConfig config =
        coli_expert_residency_policy_default();
    uint64_t old_hotness = coli_expert_residency_policy_hotness(
        stale, 4097, 0, &config);
    uint64_t new_hotness = coli_expert_residency_policy_hotness(
        recent, 4097, 0, &config);
    assert(old_hotness < new_hotness);
    assert(coli_expert_residency_policy_reuse_weight(
        stale, 4097, &config) == 0);
}

static void test_decode_weight_and_hysteresis(void) {
    ColiExpertActivationEntry entries[16];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, entries, 16) == 0);
    ColiExpertActivationEntry *prefill = observe(
        &tracker, 5, 1, COLI_EXPERT_PHASE_PREFILL, 8, 10);
    ColiExpertActivationEntry *decode = observe(
        &tracker, 5, 2, COLI_EXPERT_PHASE_DECODE, 8, 10);
    ColiExpertResidencyPolicyConfig config =
        coli_expert_residency_policy_default();
    assert(coli_expert_residency_policy_hotness(decode, 10, 0, &config) >
           coli_expert_residency_policy_hotness(prefill, 10, 0, &config));
    uint64_t cold = coli_expert_residency_policy_hotness(decode, 10, 0, &config);
    uint64_t resident = coli_expert_residency_policy_hotness(decode, 10, 1, &config);
    assert(resident > cold);

    /* Planner reuse is expected route frequency, not local residency bias. */
    assert(coli_expert_residency_policy_reuse_weight(
               prefill, 10, &config) ==
           coli_expert_residency_policy_reuse_weight(
               decode, 10, &config));
}

static void test_bounded_common_horizon(void) {
    ColiExpertActivationEntry entries[16];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, entries, 16) == 0);

    ColiExpertActivationEntry *entry = NULL;
    for (uint64_t epoch = 1; epoch <= 4096; epoch++)
        entry = observe(&tracker, 8, 42, COLI_EXPERT_PHASE_DECODE, 1, epoch);
    assert(entry);

    /* Lifetime telemetry keeps the exact process history while decision mass
     * converges to a bounded recent-rate state instead of growing to 4096. */
    assert(entry->logical_activations == 4096);
    assert(entry->decode_activations == 4096);
    uint64_t unknown = 0, prefill = 0, decode = 0;
    coli_expert_activation_recent_at(
        entry, 4096, &unknown, &prefill, &decode);
    assert(unknown == 0 && prefill == 0);
    assert(decode <= 128 && decode >= 120);

    ColiExpertResidencyPolicyConfig config =
        coli_expert_residency_policy_default();
    /* One logical use per epoch should project to roughly one use for every
     * epoch in the shared 256-epoch planning horizon. */
    uint64_t reuse = coli_expert_residency_policy_reuse_weight(
        entry, 4096, &config);
    assert(reuse >= 240 && reuse <= 256);

    ColiExpertResidencyCandidate candidate =
        coli_expert_residency_policy_candidate(
            entry, 4096, 13 * 1024 * 1024, 12000, 0, &config);
    assert(candidate.reuse_weight == reuse);
}

static void test_burst_fades_out(void) {
    ColiExpertActivationEntry entries[16];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, entries, 16) == 0);
    ColiExpertActivationEntry *entry = observe(
        &tracker, 9, 5, COLI_EXPERT_PHASE_PREFILL, 512, 1);
    ColiExpertResidencyPolicyConfig config =
        coli_expert_residency_policy_default();
    uint64_t far_epoch = 1 +
        10 * COLI_EXPERT_ACTIVITY_DECAY_QUANTUM_EPOCHS;
    assert(coli_expert_residency_policy_hotness(
        entry, far_epoch, 0, &config) == 0);
    assert(coli_expert_residency_policy_reuse_weight(
        entry, far_epoch, &config) == 0);
}

static void test_benefit_per_byte(void) {
    ColiExpertActivationEntry entries[16];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, entries, 16) == 0);
    ColiExpertActivationEntry *entry = observe(
        &tracker, 7, 3, COLI_EXPERT_PHASE_DECODE, 100, 20);
    ColiExpertResidencyPolicyConfig config =
        coli_expert_residency_policy_default();
    ColiExpertResidencyCandidate small = coli_expert_residency_policy_candidate(
        entry, 20, 8 * 1024 * 1024, 12000, 0, &config);
    ColiExpertResidencyCandidate large = coli_expert_residency_policy_candidate(
        entry, 20, 16 * 1024 * 1024, 12000, 0, &config);
    ColiExpertResidencyCandidate slower_miss = coli_expert_residency_policy_candidate(
        entry, 20, 8 * 1024 * 1024, 24000, 0, &config);
    assert(small.score > large.score);
    assert(slower_miss.score > small.score);
}

static void test_deterministic_tie_break(void) {
    ColiExpertResidencyCandidate a = {
        .key = {2, 9}, .score = 10, .hotness = 4, .last_epoch = 8,
    };
    ColiExpertResidencyCandidate b = {
        .key = {3, 1}, .score = 10, .hotness = 4, .last_epoch = 8,
    };
    assert(coli_expert_residency_policy_compare(&a, &b) > 0);
    assert(coli_expert_residency_policy_compare(&b, &a) < 0);
}

int main(void) {
    test_store_observer_contract();
    test_frequency_beats_pure_recency();
    test_stale_frequency_decays();
    test_decode_weight_and_hysteresis();
    test_bounded_common_horizon();
    test_burst_fades_out();
    test_benefit_per_byte();
    test_deterministic_tie_break();
    puts("test_expert_residency_policy: ok");
    return 0;
}

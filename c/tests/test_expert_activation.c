#include "../expert_activation.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static ColiExpertActivationSample sample(int layer, int expert,
                                         ColiExpertPhase phase,
                                         uint64_t multiplicity,
                                         uint64_t epoch) {
    ColiExpertActivationSample value;
    value.key.layer = layer;
    value.key.expert = expert;
    value.phase = phase;
    value.multiplicity = multiplicity;
    value.epoch = epoch;
    return value;
}

static void test_multiplicity_and_phase(void) {
    ColiExpertActivationEntry storage[8];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, storage, 8) == 0);

    assert(coli_expert_activation_observe(
               &tracker, sample(3, 17, COLI_EXPERT_PHASE_PREFILL, 7, 10)) == 1);
    assert(coli_expert_activation_observe(
               &tracker, sample(3, 17, COLI_EXPERT_PHASE_DECODE, 2, 11)) == 0);

    const ColiExpertActivationEntry *entry =
        coli_expert_activation_find_const(&tracker, (ColiExpertKey){3, 17});
    assert(entry);
    assert(entry->logical_activations == 9);
    assert(entry->prefill_activations == 7);
    assert(entry->decode_activations == 2);
    assert(entry->observations == 2);
    assert(entry->last_epoch == 11);
    assert(tracker.used == 1);
    assert(tracker.total_logical_activations == 9);
    assert(tracker.total_observations == 2);
}

static void test_epoch_is_monotonic(void) {
    ColiExpertActivationEntry storage[4];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, storage, 4) == 0);
    assert(coli_expert_activation_observe(
               &tracker, sample(1, 2, COLI_EXPERT_PHASE_DECODE, 1, 9)) == 1);
    assert(coli_expert_activation_observe(
               &tracker, sample(1, 2, COLI_EXPERT_PHASE_DECODE, 1, 3)) == 0);
    const ColiExpertActivationEntry *entry =
        coli_expert_activation_find_const(&tracker, (ColiExpertKey){1, 2});
    assert(entry && entry->last_epoch == 9);
}

static void test_bounded_full_table_preserves_known_keys(void) {
    ColiExpertActivationEntry storage[4];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, storage, 4) == 0);

    for (int i = 0; i < 4; i++)
        assert(coli_expert_activation_observe(
                   &tracker, sample(i, i + 10, COLI_EXPERT_PHASE_PREFILL,
                                    1, (uint64_t)i + 1)) == 1);
    assert(tracker.used == 4);
    assert(coli_expert_activation_observe(
               &tracker, sample(99, 99, COLI_EXPERT_PHASE_DECODE, 1, 5)) == -2);
    assert(tracker.dropped_new_keys == 1);

    /* A full table must still update an already-known expert. */
    assert(coli_expert_activation_observe(
               &tracker, sample(2, 12, COLI_EXPERT_PHASE_DECODE, 4, 6)) == 0);
    const ColiExpertActivationEntry *entry =
        coli_expert_activation_find_const(&tracker, (ColiExpertKey){2, 12});
    assert(entry);
    assert(entry->logical_activations == 5);
    assert(entry->decode_activations == 4);
}

static void test_saturating_counters(void) {
    ColiExpertActivationEntry storage[2];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, storage, 2) == 0);
    assert(coli_expert_activation_observe(
               &tracker, sample(0, 0, COLI_EXPERT_PHASE_DECODE,
                                UINT64_MAX - 1, 1)) == 1);
    assert(coli_expert_activation_observe(
               &tracker, sample(0, 0, COLI_EXPERT_PHASE_DECODE, 8, 2)) == 0);
    const ColiExpertActivationEntry *entry =
        coli_expert_activation_find_const(&tracker, (ColiExpertKey){0, 0});
    assert(entry && entry->logical_activations == UINT64_MAX);
    assert(entry->decode_activations == UINT64_MAX);
    assert(tracker.total_logical_activations == UINT64_MAX);
}

static void test_validation(void) {
    ColiExpertActivationEntry storage[3];
    ColiExpertActivationTracker tracker;
    assert(coli_expert_activation_init(&tracker, storage, 3) == -1);

    ColiExpertActivationEntry valid_storage[4];
    assert(coli_expert_activation_init(&tracker, valid_storage, 4) == 0);
    assert(coli_expert_activation_observe(
               &tracker, sample(-1, 1, COLI_EXPERT_PHASE_DECODE, 1, 1)) == -1);
    assert(coli_expert_activation_observe(
               &tracker, sample(1, 1, COLI_EXPERT_PHASE_DECODE, 0, 1)) == -1);
}

int main(void) {
    test_multiplicity_and_phase();
    test_epoch_is_monotonic();
    test_bounded_full_table_preserves_known_keys();
    test_saturating_counters();
    test_validation();
    puts("test_expert_activation: PASS");
    return 0;
}

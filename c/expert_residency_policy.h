#ifndef COLIBRI_EXPERT_RESIDENCY_POLICY_H
#define COLIBRI_EXPERT_RESIDENCY_POLICY_H

#include "expert_activation.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Model-neutral first-pass policy signal for adaptive expert residency.
 *
 * This file deliberately ranks candidates only; it does not own bytes, slots,
 * I/O, or eviction state. The global resource planner decides how many bytes
 * expert residency receives, and the residency manager owns generations and
 * leases. This policy answers only: among eligible experts, which bytes are
 * likely to repay themselves best?
 *
 * Frequency is preferred over pure recency because long layer-to-layer reuse
 * distances can make a small global LRU churn without hits. Recency still
 * matters through lazy age decay, and currently resident candidates receive a
 * small hysteresis bonus to avoid replacement oscillation near the cutoff.
 */
typedef struct {
    uint32_t prefill_weight;
    uint32_t decode_weight;
    uint32_t resident_hysteresis_percent;
    uint64_t recency_quantum_epochs;
} ColiExpertResidencyPolicyConfig;

typedef struct {
    ColiExpertKey key;
    uint64_t score;
    uint64_t hotness;
    uint64_t resident_bytes;
    uint64_t expected_miss_cost_us;
    uint64_t last_epoch;
    int currently_resident;
} ColiExpertResidencyCandidate;

static inline ColiExpertResidencyPolicyConfig
coli_expert_residency_policy_default(void) {
    ColiExpertResidencyPolicyConfig config;
    config.prefill_weight = 1;
    config.decode_weight = 4;
    config.resident_hysteresis_percent = 20;
    config.recency_quantum_epochs = 64;
    return config;
}

static inline uint64_t coli_expert_residency_sat_mul(uint64_t a, uint64_t b) {
    return a && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

static inline uint64_t coli_expert_residency_policy_hotness(
    const ColiExpertActivationEntry *entry,
    uint64_t current_epoch,
    int currently_resident,
    const ColiExpertResidencyPolicyConfig *config) {
    if (!entry || !entry->hash_tag || !config ||
        !config->recency_quantum_epochs)
        return 0;

    uint64_t prefill = coli_expert_residency_sat_mul(
        entry->prefill_activations, config->prefill_weight);
    uint64_t decode = coli_expert_residency_sat_mul(
        entry->decode_activations, config->decode_weight);
    uint64_t weighted = coli_expert_activation_sat_add(prefill, decode);

    /* UNKNOWN-phase observations still carry useful frequency. Count them at
     * the conservative prefill weight rather than discarding the signal. */
    uint64_t known = coli_expert_activation_sat_add(
        entry->prefill_activations, entry->decode_activations);
    uint64_t unknown = entry->logical_activations > known
        ? entry->logical_activations - known : 0;
    weighted = coli_expert_activation_sat_add(
        weighted,
        coli_expert_residency_sat_mul(unknown, config->prefill_weight));

    uint64_t age = current_epoch > entry->last_epoch
        ? current_epoch - entry->last_epoch : 0;
    uint64_t buckets = age / config->recency_quantum_epochs;
    unsigned shift = buckets > 63 ? 63u : (unsigned)buckets;
    uint64_t hotness = weighted >> shift;

    if (currently_resident && hotness && config->resident_hysteresis_percent) {
        uint64_t bonus = coli_expert_residency_sat_mul(
            hotness, config->resident_hysteresis_percent) / 100u;
        hotness = coli_expert_activation_sat_add(hotness, bonus);
    }
    return hotness;
}

static inline ColiExpertResidencyCandidate
coli_expert_residency_policy_candidate(
    const ColiExpertActivationEntry *entry,
    uint64_t current_epoch,
    uint64_t resident_bytes,
    uint64_t expected_miss_cost_us,
    int currently_resident,
    const ColiExpertResidencyPolicyConfig *config) {
    ColiExpertResidencyCandidate candidate = {0};
    if (!entry) return candidate;
    candidate.key = entry->key;
    candidate.hotness = coli_expert_residency_policy_hotness(
        entry, current_epoch, currently_resident, config);
    candidate.resident_bytes = resident_bytes;
    candidate.expected_miss_cost_us = expected_miss_cost_us;
    candidate.last_epoch = entry->last_epoch;
    candidate.currently_resident = currently_resident != 0;
    if (!resident_bytes || !candidate.hotness || !expected_miss_cost_us)
        return candidate;

    /* Normalize to 4 KiB allocation quanta so differently-sized expert
     * records can compete without floating point. Saturation is intentional:
     * an overwhelmingly valuable candidate remains tied at the ceiling and the
     * deterministic tie breakers below decide ordering. */
    uint64_t quanta = resident_bytes / UINT64_C(4096);
    if (resident_bytes % UINT64_C(4096)) quanta++;
    if (!quanta) quanta = 1;
    candidate.score = coli_expert_residency_sat_mul(
        candidate.hotness, expected_miss_cost_us) / quanta;
    return candidate;
}

/* Positive means a is preferred, negative means b is preferred. Deterministic
 * key ordering keeps policy decisions reproducible when scores tie. */
static inline int coli_expert_residency_policy_compare(
    const ColiExpertResidencyCandidate *a,
    const ColiExpertResidencyCandidate *b) {
    if (!a || !b) return a ? 1 : b ? -1 : 0;
    if (a->score != b->score) return a->score > b->score ? 1 : -1;
    if (a->hotness != b->hotness) return a->hotness > b->hotness ? 1 : -1;
    if (a->currently_resident != b->currently_resident)
        return a->currently_resident ? 1 : -1;
    if (a->last_epoch != b->last_epoch)
        return a->last_epoch > b->last_epoch ? 1 : -1;
    if (a->key.layer != b->key.layer)
        return a->key.layer < b->key.layer ? 1 : -1;
    if (a->key.expert != b->key.expert)
        return a->key.expert < b->key.expert ? 1 : -1;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_EXPERT_RESIDENCY_POLICY_H */

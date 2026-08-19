#ifndef COLIBRI_EXPERT_RESIDENCY_POLICY_H
#define COLIBRI_EXPERT_RESIDENCY_POLICY_H

#include "expert_activation.h"
#include "resource_planner.h"
#include "reuse_projection.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Model-neutral adaptive expert-residency policy.
 *
 * This layer ranks candidates only; it does not own bytes, slots, I/O, or
 * eviction state. The global resource planner decides how many bytes expert
 * residency receives, and the residency manager owns generations and leases.
 *
 * Two related signals are intentionally separate:
 *   - hotness: phase-weighted recent activity for choosing WHICH experts stay;
 *   - reuse_weight: unweighted recent route rate projected onto one bounded
 *     common horizon for deciding HOW MANY expert bytes compete with dense,
 *     prefix and other optional resources in the global planner.
 *
 * The projection math lives in reuse_projection.h so non-expert resources use
 * the exact same horizon/confidence scale instead of inventing local ratios.
 */
typedef struct {
    uint32_t prefill_weight;
    uint32_t decode_weight;
    uint32_t resident_hysteresis_percent;
    uint64_t recency_quantum_epochs;
    uint64_t planning_horizon_epochs;
    uint64_t planner_confidence_mass;
    uint64_t max_reuse_weight;
} ColiExpertResidencyPolicyConfig;

typedef struct {
    ColiExpertKey key;
    uint64_t score;
    uint64_t hotness;
    /* Planner-ready expected logical route uses over planning_horizon_epochs.
     * Resident hysteresis and decode preference are deliberately excluded so
     * this stays comparable with other resource classes. */
    uint64_t reuse_weight;
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
    config.recency_quantum_epochs = COLI_EXPERT_ACTIVITY_DECAY_QUANTUM_EPOCHS;
    config.planning_horizon_epochs = 256;
    /* Require a small amount of recent evidence before optional planner bytes
     * move away from established dense/prefix resources. Large prefill routing
     * multiplicity can satisfy this immediately; one-off decode routes cannot. */
    config.planner_confidence_mass = 8;
    config.max_reuse_weight = UINT64_C(1) << 32;
    return config;
}

static inline uint64_t coli_expert_residency_sat_mul(uint64_t a, uint64_t b) {
    return a && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

static inline int coli_expert_residency_policy_config_valid(
    const ColiExpertResidencyPolicyConfig *config) {
    return config && config->prefill_weight && config->decode_weight &&
        config->recency_quantum_epochs ==
            COLI_EXPERT_ACTIVITY_DECAY_QUANTUM_EPOCHS &&
        config->planning_horizon_epochs && config->planner_confidence_mass &&
        config->max_reuse_weight;
}

static inline ColiReuseProjectionConfig
coli_expert_residency_policy_projection_config(
    const ColiExpertResidencyPolicyConfig *config) {
    ColiReuseProjectionConfig projection = {0};
    if (!config) return projection;
    projection.decay_quantum_epochs = config->recency_quantum_epochs;
    projection.planning_horizon_epochs = config->planning_horizon_epochs;
    projection.confidence_mass = config->planner_confidence_mass;
    projection.max_reuse_weight = config->max_reuse_weight;
    return projection;
}

static inline uint64_t coli_expert_residency_policy_recent_mass(
    const ColiExpertActivationEntry *entry, uint64_t current_epoch,
    const ColiExpertResidencyPolicyConfig *config,
    int apply_phase_weights) {
    if (!entry || !entry->hash_tag ||
        !coli_expert_residency_policy_config_valid(config))
        return 0;

    uint64_t unknown = 0, prefill = 0, decode = 0;
    coli_expert_activation_recent_at(
        entry, current_epoch, &unknown, &prefill, &decode);

    if (!apply_phase_weights) {
        uint64_t mass = coli_expert_activation_sat_add(unknown, prefill);
        return coli_expert_activation_sat_add(mass, decode);
    }

    uint64_t weighted_prefill = coli_expert_residency_sat_mul(
        prefill, config->prefill_weight);
    uint64_t weighted_decode = coli_expert_residency_sat_mul(
        decode, config->decode_weight);
    uint64_t weighted_unknown = coli_expert_residency_sat_mul(
        unknown, config->prefill_weight);
    uint64_t mass = coli_expert_activation_sat_add(
        weighted_unknown, weighted_prefill);
    return coli_expert_activation_sat_add(mass, weighted_decode);
}

/* Project recent unweighted route mass through the generic optional-resource
 * projection. This keeps expert, dense, prefix and future cache classes on the
 * same fixed future horizon and cold-start confidence scale. */
static inline uint64_t coli_expert_residency_policy_reuse_weight(
    const ColiExpertActivationEntry *entry,
    uint64_t current_epoch,
    const ColiExpertResidencyPolicyConfig *config) {
    if (!coli_expert_residency_policy_config_valid(config)) return 0;
    uint64_t mass = coli_expert_residency_policy_recent_mass(
        entry, current_epoch, config, 0);
    ColiReuseProjectionConfig projection =
        coli_expert_residency_policy_projection_config(config);
    return coli_reuse_project_recent_mass(mass, &projection);
}

static inline uint64_t coli_expert_residency_policy_hotness(
    const ColiExpertActivationEntry *entry,
    uint64_t current_epoch,
    int currently_resident,
    const ColiExpertResidencyPolicyConfig *config) {
    uint64_t hotness = coli_expert_residency_policy_recent_mass(
        entry, current_epoch, config, 1);
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
    candidate.reuse_weight = coli_expert_residency_policy_reuse_weight(
        entry, current_epoch, config);
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

/* Convert the model-neutral expert policy output directly into the generic
 * planner bridge. Local hotness/hysteresis remains a WHICH-expert signal;
 * only the bounded common-horizon reuse weight crosses this boundary. */
static inline int coli_expert_residency_policy_resource_estimate(
    const ColiExpertResidencyCandidate *expert,
    uint32_t resource_id,
    uint64_t bytes_per_miss,
    uint64_t exposed_ns_per_miss,
    ColiResourceBenefitEstimate *out) {
    if (!expert || !out || !expert->resident_bytes || !bytes_per_miss)
        return -1;
    *out = (ColiResourceBenefitEstimate){
        .kind = COLI_RESOURCE_PERSISTENT_EXPERT,
        .id = resource_id,
        .resident_bytes = expert->resident_bytes,
        .reuse_weight = expert->reuse_weight,
        .bytes_per_miss = bytes_per_miss,
        .exposed_ns_per_miss = exposed_ns_per_miss,
    };
    return 0;
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

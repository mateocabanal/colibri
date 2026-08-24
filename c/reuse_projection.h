#ifndef COLIBRI_REUSE_PROJECTION_H
#define COLIBRI_REUSE_PROJECTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generic bridge from a recent-use signal into the global resource planner.
 *
 * Runtime subsystems may measure different things (logical expert routes,
 * dense-tensor hits, prefix reuse, sequence-state restores), but optional
 * resources are comparable only when their reuse weights use the same timebase
 * and future horizon. This helper owns that normalization; it knows nothing
 * about model names or resource kinds.
 *
 * `recent_mass` is the current value of a lazy half-life accumulator whose
 * value halves every `decay_quantum_epochs`. Such an accumulator has an
 * effective history window of roughly 2 * decay_quantum_epochs. We project its
 * observed rate onto `planning_horizon_epochs` and apply a bounded cold-start
 * confidence ramp so one observation cannot immediately seize global memory.
 */
typedef struct {
    uint64_t decay_quantum_epochs;
    uint64_t planning_horizon_epochs;
    uint64_t confidence_mass;
    uint64_t max_reuse_weight;
} ColiReuseProjectionConfig;

static inline uint64_t coli_reuse_sat_add(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static inline uint64_t coli_reuse_sat_mul(uint64_t a, uint64_t b) {
    return a && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

static inline uint64_t coli_reuse_scale_div(
    uint64_t value, uint64_t numerator, uint64_t denominator) {
    if (!value || !numerator || !denominator) return 0;
    uint64_t quotient = value / denominator;
    uint64_t remainder = value % denominator;
    uint64_t scaled = coli_reuse_sat_mul(quotient, numerator);
    uint64_t tail_product = coli_reuse_sat_mul(remainder, numerator);
    uint64_t tail = tail_product / denominator;
    return coli_reuse_sat_add(scaled, tail);
}

static inline int coli_reuse_projection_config_valid(
    const ColiReuseProjectionConfig *config) {
    return config && config->decay_quantum_epochs &&
        config->decay_quantum_epochs <= UINT64_MAX / 2 &&
        config->planning_horizon_epochs && config->confidence_mass &&
        config->max_reuse_weight;
}

static inline uint64_t coli_reuse_project_recent_mass(
    uint64_t recent_mass, const ColiReuseProjectionConfig *config) {
    if (!recent_mass || !coli_reuse_projection_config_valid(config)) return 0;

    uint64_t effective_window = config->decay_quantum_epochs * 2;
    uint64_t projected = coli_reuse_scale_div(
        recent_mass, config->planning_horizon_epochs, effective_window);

    uint64_t confidence = recent_mass < config->confidence_mass
        ? recent_mass : config->confidence_mass;
    projected = coli_reuse_scale_div(
        projected, confidence, config->confidence_mass);

    if (projected > config->max_reuse_weight)
        projected = config->max_reuse_weight;
    return projected;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_REUSE_PROJECTION_H */

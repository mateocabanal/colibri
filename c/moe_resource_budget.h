#ifndef COLIBRI_MOE_RESOURCE_BUDGET_H
#define COLIBRI_MOE_RESOURCE_BUDGET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_MOE_MIB UINT64_C(1048576)

typedef struct {
    /* Current OS-available memory after the model/runtime has been created. */
    uint64_t available_bytes;
    /* Current process RSS. Needed only to enforce an explicit total budget. */
    uint64_t process_resident_bytes;
    /* Optional total process/host budget. Zero means automatic. */
    uint64_t explicit_total_bytes;
    /* Optional resources not yet necessarily materialized (e.g. prefix RAM). */
    uint64_t other_optional_reserve_bytes;
    /* Physical bytes one persistent expert consumes in this memory tier. */
    uint64_t resident_bytes_per_expert;
    /* Execution-only slots that must remain possible independent of locality. */
    uint64_t transient_slots;
    /* Optional explicit persistent-slot ceiling. Zero means no user ceiling. */
    uint64_t max_persistent_slots;
} ColiMoeResourceBudgetInputs;

typedef struct {
    uint64_t system_reserve_bytes;
    uint64_t transient_bytes;
    uint64_t persistent_budget_bytes;
    uint64_t persistent_slots;
    int used_explicit_total;
} ColiMoeResourceBudget;

static inline uint64_t coli_moe_budget_sat_mul(uint64_t a, uint64_t b) {
    return a && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

static inline int coli_moe_resource_budget_compute(
    ColiMoeResourceBudget *out, const ColiMoeResourceBudgetInputs *in) {
    if (!out || !in || !in->available_bytes ||
        !in->resident_bytes_per_expert)
        return -1;
    *out = (ColiMoeResourceBudget){0};

    uint64_t reserve = in->available_bytes / 8;
    if (reserve < 512 * COLI_MOE_MIB) reserve = 512 * COLI_MOE_MIB;
    if (reserve > 4096 * COLI_MOE_MIB) reserve = 4096 * COLI_MOE_MIB;
    if (reserve >= in->available_bytes) reserve = in->available_bytes;
    out->system_reserve_bytes = reserve;

    uint64_t allocatable = in->available_bytes > reserve
        ? in->available_bytes - reserve : 0;

    /* RAM_GB-style controls are total process budgets, not expert budgets.
     * Only convert them into additional bytes when current RSS is known. If an
     * adapter cannot measure RSS it should preserve its explicit legacy cap via
     * max_persistent_slots rather than guessing how much of the total is free. */
    if (in->explicit_total_bytes && in->process_resident_bytes) {
        out->used_explicit_total = 1;
        uint64_t under_total = in->explicit_total_bytes > in->process_resident_bytes
            ? in->explicit_total_bytes - in->process_resident_bytes : 0;
        if (under_total < allocatable) allocatable = under_total;
    }

    if (in->other_optional_reserve_bytes >= allocatable)
        allocatable = 0;
    else
        allocatable -= in->other_optional_reserve_bytes;

    out->transient_bytes = coli_moe_budget_sat_mul(
        in->transient_slots, in->resident_bytes_per_expert);
    if (out->transient_bytes >= allocatable)
        allocatable = 0;
    else
        allocatable -= out->transient_bytes;

    uint64_t slots = allocatable / in->resident_bytes_per_expert;
    if (in->max_persistent_slots && slots > in->max_persistent_slots)
        slots = in->max_persistent_slots;
    out->persistent_slots = slots;
    out->persistent_budget_bytes = coli_moe_budget_sat_mul(
        slots, in->resident_bytes_per_expert);
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_MOE_RESOURCE_BUDGET_H */

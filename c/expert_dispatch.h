#ifndef COLIBRI_EXPERT_DISPATCH_H
#define COLIBRI_EXPERT_DISPATCH_H

#include "expert_residency_policy.h"
#include "expert_transform.h"
#include "resource_planner.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    COLI_HET_DISPATCH_MAX_GROUPS = 16,
    COLI_HET_DISPATCH_MAX_EXPERT_BATCHES = 64,
    COLI_HET_DISPATCH_MAX_ROUTES = 256,
};
#define COLI_HET_DISPATCH_NONE UINT32_MAX

typedef enum {
    COLI_HET_DISPATCH_ALLOW_BASELINE_FALLBACK = 0,
    COLI_HET_DISPATCH_STRICT = 1,
} ColiHeterogeneousDispatchMode;

typedef enum {
    COLI_HET_DISPATCH_INVALID = -1,
    COLI_HET_DISPATCH_STRICT_UNSUPPORTED = -2,
    COLI_HET_DISPATCH_CAPACITY = -3,
    COLI_HET_DISPATCH_READY = 1,
    COLI_HET_DISPATCH_NEEDS_FALLBACK = 2,
} ColiHeterogeneousDispatchResult;

/* A class identifies one specialized executable path. target/device semantics,
 * math/scale/layout/kernel ABI live in representation; backend_class and
 * kernel_class are provider-owned stable values, never model names. */
typedef struct {
    ColiRepresentationId representation;
    uint32_t backend_class;
    uint32_t kernel_class;
    uint32_t activation_contract;
    uint32_t geometry_class;
} ColiExpertDispatchClassKey;

typedef struct {
    ColiExpertResidencyEntry *entry;
    uint32_t routed_row;
    uint32_t route_ordinal;
    int32_t route_weight_q16;
    uint32_t activation_contract;
    uint32_t geometry_class;
} ColiExpertRoutedWork;

/* Return 1 when the exact resident view is executable and provide one
 * specialized backend/kernel class. Return 0 when unsupported, <0 on error. */
typedef int (*ColiExpertDispatchClassifyFn)(
    void *context,
    const ColiExpertResidentView *view,
    uint32_t activation_contract,
    uint32_t geometry_class,
    uint32_t *backend_class,
    uint32_t *kernel_class);

typedef struct {
    ColiExpertDispatchClassKey key;
    uint32_t expert_batch_count;
    uint32_t routed_work_count;
} ColiExpertDispatchGroup;

typedef struct {
    ColiExpertResidencyEntry *entry;
    ColiExpertResidencyLease lease;
    uint32_t group_index;
    uint32_t route_offset;
    uint32_t route_count;
} ColiExpertDispatchBatch;

typedef struct {
    uint32_t group_count;
    uint32_t batch_count;
    uint32_t route_count;
    uint32_t executable_route_count;
    uint32_t unresolved_route_count;
    ColiExpertDispatchGroup groups[COLI_HET_DISPATCH_MAX_GROUPS];
    ColiExpertDispatchBatch batches[COLI_HET_DISPATCH_MAX_EXPERT_BATCHES];
    uint32_t route_order[COLI_HET_DISPATCH_MAX_ROUTES];
    uint32_t route_batch[COLI_HET_DISPATCH_MAX_ROUTES];
} ColiHeterogeneousDispatchPlan;

typedef struct {
    uint64_t heterogeneous_routed_blocks;
    uint64_t representation_groups;
    uint64_t routed_rows;
    uint64_t expert_batches;
    uint64_t fallback_rows;
    uint64_t kernel_invocations;
} ColiHeterogeneousDispatchTelemetry;

static inline int coli_expert_dispatch_class_equal(
        const ColiExpertDispatchClassKey *a,
        const ColiExpertDispatchClassKey *b) {
    return a && b && a->backend_class == b->backend_class &&
        a->kernel_class == b->kernel_class &&
        a->activation_contract == b->activation_contract &&
        a->geometry_class == b->geometry_class &&
        coli_representation_equal(&a->representation, &b->representation);
}

static inline void coli_expert_dispatch_plan_release(
        ColiHeterogeneousDispatchPlan *plan) {
    if (!plan) return;
    for (uint32_t i = 0; i < plan->batch_count; ++i) {
        if (plan->batches[i].lease.entry)
            (void)coli_expert_residency_release(&plan->batches[i].lease);
    }
    memset(plan, 0, sizeof(*plan));
}

static inline int coli_expert_dispatch_try_variant(
        const ColiExpertRoutedWork *work,
        uint32_t variant_id,
        ColiExpertDispatchClassifyFn classify,
        void *classify_context,
        ColiExpertResidencyLease *lease_out,
        ColiExpertDispatchClassKey *class_out) {
    ColiExpertResidencyLease lease;
    int acquired = coli_expert_residency_acquire_variant(
        work->entry, variant_id, &lease);
    if (acquired <= 0) return acquired;

    ColiExpertResidentView view;
    uint32_t backend_class = 0;
    uint32_t kernel_class = 0;
    if (coli_expert_residency_lease_view(&lease, &view) != 0) {
        (void)coli_expert_residency_release(&lease);
        return -1;
    }
    int supported = classify(
        classify_context, &view, work->activation_contract,
        work->geometry_class, &backend_class, &kernel_class);
    if (supported <= 0) {
        (void)coli_expert_residency_release(&lease);
        return supported;
    }
    if (!backend_class || !kernel_class) {
        (void)coli_expert_residency_release(&lease);
        return -1;
    }

    *class_out = (ColiExpertDispatchClassKey){
        .representation = view.representation,
        .backend_class = backend_class,
        .kernel_class = kernel_class,
        .activation_contract = work->activation_contract,
        .geometry_class = work->geometry_class,
    };
    *lease_out = lease;
    return 1;
}

static inline int coli_expert_dispatch_select_variant(
        const ColiExpertRoutedWork *work,
        ColiExpertDispatchClassifyFn classify,
        void *classify_context,
        ColiExpertResidencyLease *lease_out,
        ColiExpertDispatchClassKey *class_out) {
    int preferred = coli_expert_residency_preferred_variant(work->entry);
    if (preferred >= 0 && preferred < COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY) {
        int selected = coli_expert_dispatch_try_variant(
            work, (uint32_t)preferred, classify, classify_context,
            lease_out, class_out);
        if (selected != 0) return selected;
    }
    for (uint32_t i = 0; i < COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY; ++i) {
        if ((int)i == preferred) continue;
        int selected = coli_expert_dispatch_try_variant(
            work, i, classify, classify_context, lease_out, class_out);
        if (selected != 0) return selected;
    }
    return 0;
}

static inline int coli_expert_dispatch_plan_build(
        const ColiExpertRoutedWork *work,
        size_t work_count,
        ColiExpertDispatchClassifyFn classify,
        void *classify_context,
        ColiHeterogeneousDispatchMode mode,
        ColiHeterogeneousDispatchPlan *plan) {
    if ((!work && work_count) || !classify || !plan ||
        work_count > COLI_HET_DISPATCH_MAX_ROUTES ||
        (mode != COLI_HET_DISPATCH_ALLOW_BASELINE_FALLBACK &&
         mode != COLI_HET_DISPATCH_STRICT))
        return COLI_HET_DISPATCH_INVALID;

    memset(plan, 0, sizeof(*plan));
    plan->route_count = (uint32_t)work_count;
    for (size_t i = 0; i < work_count; ++i)
        plan->route_batch[i] = COLI_HET_DISPATCH_NONE;

    for (uint32_t route = 0; route < (uint32_t)work_count; ++route) {
        if (!work[route].entry) {
            coli_expert_dispatch_plan_release(plan);
            return COLI_HET_DISPATCH_INVALID;
        }

        ColiExpertResidencyLease lease;
        ColiExpertDispatchClassKey class_key;
        int selected = coli_expert_dispatch_select_variant(
            &work[route], classify, classify_context, &lease, &class_key);
        if (selected < 0) {
            coli_expert_dispatch_plan_release(plan);
            return COLI_HET_DISPATCH_INVALID;
        }
        if (!selected) {
            if (mode == COLI_HET_DISPATCH_STRICT) {
                coli_expert_dispatch_plan_release(plan);
                return COLI_HET_DISPATCH_STRICT_UNSUPPORTED;
            }
            plan->unresolved_route_count++;
            continue;
        }

        uint32_t group = COLI_HET_DISPATCH_NONE;
        for (uint32_t g = 0; g < plan->group_count; ++g) {
            if (coli_expert_dispatch_class_equal(
                    &plan->groups[g].key, &class_key)) {
                group = g;
                break;
            }
        }
        if (group == COLI_HET_DISPATCH_NONE) {
            if (plan->group_count >= COLI_HET_DISPATCH_MAX_GROUPS) {
                (void)coli_expert_residency_release(&lease);
                coli_expert_dispatch_plan_release(plan);
                return COLI_HET_DISPATCH_CAPACITY;
            }
            group = plan->group_count++;
            plan->groups[group].key = class_key;
        }

        uint32_t batch = COLI_HET_DISPATCH_NONE;
        for (uint32_t b = 0; b < plan->batch_count; ++b) {
            ColiExpertDispatchBatch *existing = &plan->batches[b];
            if (existing->group_index == group &&
                existing->entry == work[route].entry &&
                existing->lease.variant_id == lease.variant_id &&
                existing->lease.generation == lease.generation) {
                batch = b;
                break;
            }
        }
        if (batch != COLI_HET_DISPATCH_NONE) {
            (void)coli_expert_residency_release(&lease);
        } else {
            if (plan->batch_count >= COLI_HET_DISPATCH_MAX_EXPERT_BATCHES) {
                (void)coli_expert_residency_release(&lease);
                coli_expert_dispatch_plan_release(plan);
                return COLI_HET_DISPATCH_CAPACITY;
            }
            batch = plan->batch_count++;
            plan->batches[batch].entry = work[route].entry;
            plan->batches[batch].lease = lease;
            plan->batches[batch].group_index = group;
            plan->groups[group].expert_batch_count++;
        }

        plan->route_batch[route] = batch;
        plan->batches[batch].route_count++;
        plan->groups[group].routed_work_count++;
        plan->executable_route_count++;
    }

    uint32_t cursor = 0;
    for (uint32_t group = 0; group < plan->group_count; ++group) {
        for (uint32_t batch = 0; batch < plan->batch_count; ++batch) {
            if (plan->batches[batch].group_index != group) continue;
            plan->batches[batch].route_offset = cursor;
            uint32_t emitted = 0;
            for (uint32_t route = 0; route < (uint32_t)work_count; ++route) {
                if (plan->route_batch[route] != batch) continue;
                plan->route_order[cursor++] = route;
                emitted++;
            }
            if (emitted != plan->batches[batch].route_count) {
                coli_expert_dispatch_plan_release(plan);
                return COLI_HET_DISPATCH_INVALID;
            }
        }
    }
    if (cursor != plan->executable_route_count) {
        coli_expert_dispatch_plan_release(plan);
        return COLI_HET_DISPATCH_INVALID;
    }

    return plan->unresolved_route_count
        ? COLI_HET_DISPATCH_NEEDS_FALLBACK : COLI_HET_DISPATCH_READY;
}

static inline void coli_expert_dispatch_telemetry_record(
        ColiHeterogeneousDispatchTelemetry *telemetry,
        const ColiHeterogeneousDispatchPlan *plan) {
    if (!telemetry || !plan) return;
    telemetry->heterogeneous_routed_blocks++;
    telemetry->representation_groups += plan->group_count;
    telemetry->routed_rows += plan->route_count;
    telemetry->expert_batches += plan->batch_count;
    telemetry->fallback_rows += plan->unresolved_route_count;
    telemetry->kernel_invocations += plan->group_count;
}

/* ------------------------------------------------------------------------- */
/* Profile-guided representation promotion.                                  */
/* ------------------------------------------------------------------------- */

typedef enum {
    COLI_PROMOTION_SELECTED = 1,
    COLI_PROMOTION_REJECT_NO_REUSE = 2,
    COLI_PROMOTION_REJECT_NO_SPEEDUP = 3,
    COLI_PROMOTION_REJECT_HYSTERESIS = 4,
    COLI_PROMOTION_REJECT_AGE = 5,
    COLI_PROMOTION_REJECT_UNAMORTIZED = 6,
    COLI_PROMOTION_REJECT_NO_BUDGET = 7,
    COLI_PROMOTION_REJECT_MEMORY_PRESSURE = 8,
    COLI_PROMOTION_REJECT_NO_VALUE = 9,
} ColiExpertPromotionDecision;

typedef struct {
    uint32_t speedup_margin_percent;
    uint32_t max_auto_memory_pressure_percent;
    uint64_t minimum_useful_age_epochs;
    uint64_t minimum_reuse_weight;
    uint64_t minimum_gain_ns_per_kib;
} ColiExpertPromotionPolicyConfig;

typedef struct {
    uint64_t last_switch_epoch;
    uint64_t last_transform_prepare_ns;
    uint64_t realized_saved_ns;
    uint64_t realized_uses;
} ColiExpertPromotionHistory;

typedef struct {
    const ColiExpertActivationEntry *activation;
    uint64_t current_epoch;
    uint64_t current_exec_ns;
    uint64_t target_exec_ns;
    uint64_t transform_prepare_ns;
    uint64_t expected_extra_dispatch_ns;
    uint64_t incremental_resident_bytes;
    uint64_t shadow_headroom_bytes;
    uint32_t memory_pressure_percent;
} ColiExpertPromotionInputs;

typedef struct {
    ColiExpertPromotionDecision decision;
    uint64_t expected_future_uses;
    uint64_t per_use_saved_ns;
    uint64_t gross_time_gain_ns;
    uint64_t net_time_gain_ns;
    uint64_t promotion_value_per_byte;
    uint64_t break_even_uses;
    uint64_t incremental_resident_bytes;
} ColiExpertPromotionEvaluation;

typedef struct {
    uint64_t promotion_candidates;
    uint64_t promotion_selected;
    uint64_t promotion_rejected_no_speedup;
    uint64_t promotion_rejected_no_reuse;
    uint64_t promotion_rejected_no_budget;
    uint64_t promotion_rejected_fragmentation;
    uint64_t promotion_rejected_hysteresis;
    uint64_t preferred_variant_switches;
    uint64_t variant_demotions;
    uint64_t estimated_gain_ns;
    uint64_t realized_kernel_ns_saved;
    uint64_t extra_dispatch_ns;
} ColiExpertPromotionTelemetry;

typedef int (*ColiExpertPromotionExecutableFn)(
    void *context, const ColiExpertResidentView *view);

typedef struct {
    int active;
    ColiJitTransformTicket ticket;
    ColiExpertResidencyEntry *entry;
    ColiRepresentationId target;
    uint32_t transform_abi;
    uint64_t estimated_prepare_ns;
} ColiExpertPromotionPending;

static inline ColiExpertPromotionPolicyConfig
coli_expert_promotion_policy_default(void) {
    ColiExpertPromotionPolicyConfig config;
    config.speedup_margin_percent = 10;
    config.max_auto_memory_pressure_percent = 95;
    config.minimum_useful_age_epochs = COLI_EXPERT_ACTIVITY_DECAY_QUANTUM_EPOCHS;
    config.minimum_reuse_weight = 2;
    config.minimum_gain_ns_per_kib = 1;
    return config;
}

static inline void coli_expert_promotion_history_init(
        ColiExpertPromotionHistory *history) {
    if (history) memset(history, 0, sizeof(*history));
}

static inline void coli_expert_promotion_note_use(
        ColiExpertPromotionHistory *history, uint64_t realized_saved_ns) {
    if (!history) return;
    history->realized_uses = coli_expert_activation_sat_add(
        history->realized_uses, 1);
    history->realized_saved_ns = coli_expert_activation_sat_add(
        history->realized_saved_ns, realized_saved_ns);
}

static inline uint64_t coli_expert_promotion_ceil_div(
        uint64_t numerator, uint64_t denominator) {
    if (!denominator) return UINT64_MAX;
    uint64_t quotient = numerator / denominator;
    return quotient + ((numerator % denominator) != 0);
}

static inline ColiExpertPromotionEvaluation coli_expert_promotion_evaluate(
        const ColiExpertPromotionInputs *inputs,
        const ColiExpertPromotionHistory *history,
        const ColiExpertResidencyPolicyConfig *reuse_config,
        const ColiExpertPromotionPolicyConfig *promotion_config) {
    ColiExpertPromotionEvaluation out;
    memset(&out, 0, sizeof(out));
    out.decision = COLI_PROMOTION_REJECT_NO_VALUE;
    if (!inputs || !inputs->activation || !reuse_config || !promotion_config ||
        !inputs->current_exec_ns || !inputs->target_exec_ns ||
        !inputs->incremental_resident_bytes ||
        promotion_config->speedup_margin_percent > 100 ||
        promotion_config->max_auto_memory_pressure_percent > 100 ||
        !promotion_config->minimum_reuse_weight)
        return out;

    out.incremental_resident_bytes = inputs->incremental_resident_bytes;
    out.expected_future_uses = coli_expert_residency_policy_reuse_weight(
        inputs->activation, inputs->current_epoch, reuse_config);
    if (out.expected_future_uses < promotion_config->minimum_reuse_weight) {
        out.decision = COLI_PROMOTION_REJECT_NO_REUSE;
        return out;
    }
    if (inputs->target_exec_ns >= inputs->current_exec_ns) {
        out.decision = COLI_PROMOTION_REJECT_NO_SPEEDUP;
        return out;
    }

    out.per_use_saved_ns = inputs->current_exec_ns - inputs->target_exec_ns;
    if (promotion_config->speedup_margin_percent &&
        coli_resource_ratio_compare(
            out.per_use_saved_ns, inputs->current_exec_ns,
            promotion_config->speedup_margin_percent, 100) < 0) {
        out.decision = COLI_PROMOTION_REJECT_HYSTERESIS;
        return out;
    }

    if (history && history->last_switch_epoch &&
        inputs->current_epoch >= history->last_switch_epoch &&
        inputs->current_epoch - history->last_switch_epoch <
            promotion_config->minimum_useful_age_epochs) {
        out.decision = COLI_PROMOTION_REJECT_AGE;
        return out;
    }
    if (history && history->last_transform_prepare_ns &&
        history->realized_saved_ns < history->last_transform_prepare_ns) {
        out.decision = COLI_PROMOTION_REJECT_UNAMORTIZED;
        return out;
    }
    if (inputs->incremental_resident_bytes > inputs->shadow_headroom_bytes) {
        out.decision = COLI_PROMOTION_REJECT_NO_BUDGET;
        return out;
    }
    if (inputs->memory_pressure_percent >=
        promotion_config->max_auto_memory_pressure_percent) {
        out.decision = COLI_PROMOTION_REJECT_MEMORY_PRESSURE;
        return out;
    }

    out.gross_time_gain_ns = coli_resource_saturating_mul(
        out.expected_future_uses, out.per_use_saved_ns);
    uint64_t one_time_cost = coli_resource_saturating_add(
        inputs->transform_prepare_ns, inputs->expected_extra_dispatch_ns);
    if (out.gross_time_gain_ns <= one_time_cost) {
        out.break_even_uses = coli_expert_promotion_ceil_div(
            one_time_cost, out.per_use_saved_ns);
        out.decision = COLI_PROMOTION_REJECT_NO_VALUE;
        return out;
    }
    out.net_time_gain_ns = out.gross_time_gain_ns - one_time_cost;
    out.promotion_value_per_byte =
        out.net_time_gain_ns / inputs->incremental_resident_bytes;
    out.break_even_uses = coli_expert_promotion_ceil_div(
        one_time_cost, out.per_use_saved_ns);

    if (promotion_config->minimum_gain_ns_per_kib) {
        uint64_t pressure_scale = 100u + inputs->memory_pressure_percent;
        uint64_t required_num = coli_resource_saturating_mul(
            promotion_config->minimum_gain_ns_per_kib, pressure_scale);
        uint64_t required_den = UINT64_C(1024) * UINT64_C(100);
        if (coli_resource_ratio_compare(
                out.net_time_gain_ns, inputs->incremental_resident_bytes,
                required_num, required_den) < 0) {
            out.decision = COLI_PROMOTION_REJECT_NO_VALUE;
            return out;
        }
    }

    out.decision = COLI_PROMOTION_SELECTED;
    return out;
}

/* Optional promoted representation value is already incremental A->B value.
 * It intentionally enters the global economy as OTHER instead of pretending
 * to avoid a cold expert miss. The one-time transform and fragmentation costs
 * have already been subtracted from expected_exposed_ns_avoided. */
static inline int coli_expert_promotion_resource_candidate(
        const ColiExpertPromotionEvaluation *evaluation,
        uint32_t resource_id,
        ColiResourceCandidate *candidate) {
    if (!evaluation || !candidate ||
        evaluation->decision != COLI_PROMOTION_SELECTED ||
        !evaluation->incremental_resident_bytes ||
        !evaluation->net_time_gain_ns)
        return -1;
    *candidate = (ColiResourceCandidate){
        .kind = COLI_RESOURCE_OTHER,
        .id = resource_id,
        .resident_bytes = evaluation->incremental_resident_bytes,
        .expected_bytes_avoided = 0,
        .expected_exposed_ns_avoided = evaluation->net_time_gain_ns,
    };
    return 0;
}

static inline void coli_expert_promotion_telemetry_record_evaluation(
        ColiExpertPromotionTelemetry *telemetry,
        const ColiExpertPromotionEvaluation *evaluation,
        uint64_t extra_dispatch_ns) {
    if (!telemetry || !evaluation) return;
    telemetry->promotion_candidates++;
    telemetry->extra_dispatch_ns = coli_expert_activation_sat_add(
        telemetry->extra_dispatch_ns, extra_dispatch_ns);
    switch (evaluation->decision) {
        case COLI_PROMOTION_SELECTED:
            telemetry->promotion_selected++;
            telemetry->estimated_gain_ns = coli_expert_activation_sat_add(
                telemetry->estimated_gain_ns, evaluation->net_time_gain_ns);
            break;
        case COLI_PROMOTION_REJECT_NO_SPEEDUP:
            telemetry->promotion_rejected_no_speedup++;
            break;
        case COLI_PROMOTION_REJECT_NO_REUSE:
            telemetry->promotion_rejected_no_reuse++;
            break;
        case COLI_PROMOTION_REJECT_NO_BUDGET:
        case COLI_PROMOTION_REJECT_MEMORY_PRESSURE:
            telemetry->promotion_rejected_no_budget++;
            break;
        case COLI_PROMOTION_REJECT_HYSTERESIS:
        case COLI_PROMOTION_REJECT_AGE:
        case COLI_PROMOTION_REJECT_UNAMORTIZED:
            telemetry->promotion_rejected_hysteresis++;
            break;
        case COLI_PROMOTION_REJECT_NO_VALUE:
            if (extra_dispatch_ns)
                telemetry->promotion_rejected_fragmentation++;
            break;
    }
}

/* Reacquire and verify one durable residency identity before changing policy.
 * Holding this lease prevents eviction/reuse across the preferred publication. */
static inline int coli_expert_promotion_set_preferred_verified(
        ColiExpertResidencyEntry *entry,
        uint32_t variant_id,
        uint64_t generation,
        const ColiRepresentationId *target,
        ColiExpertPromotionExecutableFn executable,
        void *executable_context) {
    if (!entry || !target || !executable ||
        variant_id >= COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY || !generation)
        return -1;
    ColiExpertResidencyLease lease;
    int acquired = coli_expert_residency_acquire_variant(
        entry, variant_id, &lease);
    if (acquired <= 0) return acquired;
    ColiExpertResidentView view;
    int valid = lease.generation == generation &&
        coli_representation_equal(&lease.representation, target) &&
        coli_expert_residency_lease_view(&lease, &view) == 0 &&
        executable(executable_context, &view) > 0;
    if (!valid) {
        (void)coli_expert_residency_release(&lease);
        return 0;
    }
    int switched = coli_expert_residency_set_preferred(entry, variant_id);
    int released = coli_expert_residency_release(&lease);
    return switched == 0 && released == 0 ? 1 : -1;
}

static inline ColiJitTransformRequestResult coli_expert_promotion_request(
        ColiJitTransformService *service,
        ColiExpertResidencyEntry *entry,
        uint32_t source_variant_id,
        uint64_t source_generation,
        ColiExpertResidencyBudget *destination_budget,
        const ColiRepresentationId *target,
        uint32_t transform_abi,
        ColiJitTransformPriority priority,
        const ColiExpertPromotionEvaluation *evaluation,
        ColiExpertPromotionPending *pending) {
    if (!pending || !evaluation ||
        evaluation->decision != COLI_PROMOTION_SELECTED)
        return COLI_JIT_REQUEST_INVALID;
    memset(pending, 0, sizeof(*pending));
    ColiJitTransformTicket ticket;
    ColiJitTransformRequestResult result = coli_jit_transform_request(
        service, entry, source_variant_id, source_generation,
        destination_budget, target, transform_abi, priority, &ticket);
    if (result == COLI_JIT_REQUEST_OWNER || result == COLI_JIT_REQUEST_JOIN) {
        pending->active = 1;
        pending->ticket = ticket;
        pending->entry = entry;
        pending->target = *target;
        pending->transform_abi = transform_abi;
    }
    return result;
}

static inline int coli_expert_promotion_finalize(
        ColiJitTransformService *service,
        ColiExpertPromotionPending *pending,
        uint64_t current_epoch,
        uint64_t measured_prepare_ns,
        ColiExpertPromotionExecutableFn executable,
        void *executable_context,
        ColiExpertPromotionHistory *history,
        ColiExpertPromotionTelemetry *telemetry) {
    if (!service || !pending || !pending->active || !pending->entry ||
        !executable)
        return -1;
    ColiJitTransformAttemptInfo info;
    if (coli_jit_transform_query(service, pending->ticket, &info) != 0) {
        pending->active = 0;
        return -1;
    }
    if (info.state == COLI_JIT_ATTEMPT_QUEUED ||
        info.state == COLI_JIT_ATTEMPT_PREPARING ||
        info.state == COLI_JIT_ATTEMPT_VALIDATING)
        return 0;
    if (info.state != COLI_JIT_ATTEMPT_PUBLISHED) {
        pending->active = 0;
        return -1;
    }

    int switched = coli_expert_promotion_set_preferred_verified(
        pending->entry, info.destination_variant_id,
        info.destination_generation, &pending->target,
        executable, executable_context);
    pending->active = 0;
    if (switched != 1) return switched;
    if (history) {
        history->last_switch_epoch = current_epoch;
        history->last_transform_prepare_ns = measured_prepare_ns;
        history->realized_saved_ns = 0;
        history->realized_uses = 0;
    }
    if (telemetry) telemetry->preferred_variant_switches++;
    return 1;
}

static inline int coli_expert_promotion_should_demote(
        const ColiExpertActivationEntry *activation,
        uint64_t current_epoch,
        int hard_memory_pressure,
        const ColiExpertPromotionHistory *history,
        const ColiExpertResidencyPolicyConfig *reuse_config,
        const ColiExpertPromotionPolicyConfig *promotion_config) {
    if (!activation || !reuse_config || !promotion_config) return -1;
    if (hard_memory_pressure) return 1;
    if (history && history->last_switch_epoch &&
        current_epoch >= history->last_switch_epoch &&
        current_epoch - history->last_switch_epoch <
            promotion_config->minimum_useful_age_epochs)
        return 0;
    return coli_expert_residency_policy_reuse_weight(
        activation, current_epoch, reuse_config) <
        promotion_config->minimum_reuse_weight;
}

static inline int coli_expert_promotion_demote_verified(
        ColiExpertResidencyEntry *entry,
        uint32_t baseline_variant_id,
        uint64_t baseline_generation,
        const ColiRepresentationId *baseline_representation,
        uint64_t current_epoch,
        ColiExpertPromotionExecutableFn executable,
        void *executable_context,
        ColiExpertPromotionHistory *history,
        ColiExpertPromotionTelemetry *telemetry) {
    int switched = coli_expert_promotion_set_preferred_verified(
        entry, baseline_variant_id, baseline_generation,
        baseline_representation, executable, executable_context);
    if (switched != 1) return switched;
    if (history) {
        history->last_switch_epoch = current_epoch;
        history->last_transform_prepare_ns = 0;
        history->realized_saved_ns = 0;
        history->realized_uses = 0;
    }
    if (telemetry) telemetry->variant_demotions++;
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_EXPERT_DISPATCH_H */

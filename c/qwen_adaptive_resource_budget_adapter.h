#ifndef COLIBRI_QWEN_ADAPTIVE_RESOURCE_BUDGET_ADAPTER_H
#define COLIBRI_QWEN_ADAPTIVE_RESOURCE_BUDGET_ADAPTER_H

#include "moe_resource_budget.h"
#include "runtime_memory.h"

/*
 * Qwen mapping into the shared MoE byte-budget mechanism.
 *
 * The generic planner knows only bytes, transient concurrency, process/OS
 * memory and an optional total budget. Qwen contributes the already-measured
 * physical expert size and its prefix-cache reserve. No Qwen model geometry or
 * guessed bytes-per-slot enters the budget decision.
 *
 * Qwen's legacy physical cache does not yet expose an exact global count of
 * reclaimable resident expert bytes. Therefore compute the optional byte
 * envelope once, when the first real expert size is known, and keep it stable
 * for that model lifetime. Re-probing OS headroom every policy replan would
 * charge already-resident experts twice. Future shared residency-manager leases
 * can feed `current_optional_resident_bytes` and make the envelope dynamic.
 */
typedef struct {
    void *model;
    uint64_t resident_bytes_per_expert;
    uint64_t persistent_budget_bytes;
    uint64_t system_reserve_bytes;
    int used_explicit_total;
    int valid;
} ColiQwenAdaptiveBudgetCache;

static ColiQwenAdaptiveBudgetCache g_qwen_adaptive_budget;

static int qwen_adaptive_resource_budget_resolve(
    void *model, uint64_t resident_bytes_per_expert,
    uint64_t transient_slots, ColiMoeResourceBudget *out) {
    if (!model || !resident_bytes_per_expert || !out) return -1;

    /* A freshly initialized controller has replan_count==0. Do not reuse a
     * process-global cached envelope in that state even if malloc happened to
     * recycle the same Model address and the new model has the same expert
     * format/size. Once this controller has made its first plan, the cached
     * envelope is intentionally stable for the rest of that model lifetime. */
    if (g_qwen_adaptive_budget.valid &&
        g_qwen_adaptive.adaptive.replan_count > 0 &&
        g_qwen_adaptive_budget.model == model &&
        g_qwen_adaptive_budget.resident_bytes_per_expert ==
            resident_bytes_per_expert) {
        *out = (ColiMoeResourceBudget){
            .system_reserve_bytes = g_qwen_adaptive_budget.system_reserve_bytes,
            .persistent_budget_bytes =
                g_qwen_adaptive_budget.persistent_budget_bytes,
            .persistent_slots = g_qwen_adaptive_budget.persistent_budget_bytes /
                resident_bytes_per_expert,
            .used_explicit_total = g_qwen_adaptive_budget.used_explicit_total,
        };
        return 0;
    }

    uint64_t available = coli_runtime_available_memory_bytes();
    if (!available) return -1;

    uint64_t explicit_total = coli_runtime_parse_decimal_gb(getenv("RAM_GB"));
    uint64_t rss = coli_runtime_process_resident_bytes();
    uint64_t max_slots = 0;

    /* A positional/legacy CACHE larger than top-k is an explicit benchmark/
     * user ceiling when RAM_GB is absent. Preserve it as a ceiling, not as the
     * automatic sizing policy. The ordinary top-k bootstrap is not a ceiling. */
    if (!explicit_total && g_qwen_adaptive.legacy_cap > g_qwen_adaptive.topk) {
        uint64_t total = coli_moe_budget_sat_mul(
            (uint64_t)g_qwen_adaptive.legacy_cap,
            (uint64_t)g_qwen_adaptive.layers);
        max_slots = total > transient_slots ? total - transient_slots : 0;
    }

    uint64_t key_count = (uint64_t)g_qwen_adaptive.layers *
                         (uint64_t)g_qwen_adaptive.experts;
    if (!max_slots || max_slots > key_count) max_slots = key_count;

    /* Reserve the normal serve-prefix allowance even before the first snapshot
     * materializes. On non-serve runs this is conservative by a small fixed
     * amount and prevents experts from consuming memory the prefix tier may
     * need later. */
    uint64_t prefix_reserve = (uint64_t)qwen_prefix_cache_budget_for_serve();

    ColiMoeResourceBudgetInputs inputs = {
        .available_bytes = available,
        .process_resident_bytes = rss,
        .explicit_total_bytes = explicit_total,
        .current_optional_resident_bytes = 0,
        .other_optional_reserve_bytes = prefix_reserve,
        .resident_bytes_per_expert = resident_bytes_per_expert,
        .transient_slots = transient_slots,
        .max_persistent_slots = max_slots,
    };
    if (coli_moe_resource_budget_compute(out, &inputs) != 0) return -1;

    g_qwen_adaptive_budget = (ColiQwenAdaptiveBudgetCache){
        .model = model,
        .resident_bytes_per_expert = resident_bytes_per_expert,
        .persistent_budget_bytes = out->persistent_budget_bytes,
        .system_reserve_bytes = out->system_reserve_bytes,
        .used_explicit_total = out->used_explicit_total,
        .valid = 1,
    };
    return 0;
}

static int qwen_adaptive_resource_budget_plan(
    void *model, uint64_t resident_bytes_per_expert,
    uint64_t transient_slots) {
    if (!model || !resident_bytes_per_expert) return 0;

    ColiMoeResourceBudget budget;
    if (qwen_adaptive_resource_budget_resolve(
            model, resident_bytes_per_expert, transient_slots, &budget) != 0) {
        /* If the platform cannot report a safe byte envelope, retain Qwen's
         * existing bounded physical behavior rather than inventing one. */
        return (qwen_adaptive_adapter_maybe_plan)(
            model, resident_bytes_per_expert, transient_slots);
    }

    int changed = 0;
    pthread_mutex_lock(&g_qwen_adaptive.mutex);
    if (g_qwen_adaptive.enabled && model == g_qwen_adaptive.model) {
        ColiResourceSelection selection = {0};
        int rc = coli_moe_adaptive_select_experts(
            &g_qwen_adaptive.adaptive,
            budget.persistent_budget_bytes,
            resident_bytes_per_expert,
            resident_bytes_per_expert,
            0,
            COLI_RESOURCE_VALUE_BYTES,
            qwen_adaptive_selected_before,
            &g_qwen_adaptive,
            &selection);
        if (rc > 0) {
            g_qwen_adaptive.selected_count = selection.selected_count;
            g_qwen_adaptive.selected_bytes = selection.selected_resident_bytes;
            g_qwen_adaptive.resident_bytes_per_expert = resident_bytes_per_expert;
            fprintf(stderr,
                    "qwen_resource_planner status=replan count=%llu epoch=%llu "
                    "selected=%llu expert_budget=%.2fGiB transient_slots=%llu "
                    "system_reserve=%.2fGiB source=%s\n",
                    (unsigned long long)g_qwen_adaptive.adaptive.replan_count,
                    (unsigned long long)g_qwen_adaptive.adaptive.current_epoch,
                    (unsigned long long)selection.selected_count,
                    budget.persistent_budget_bytes /
                        (1024.0 * 1024.0 * 1024.0),
                    (unsigned long long)transient_slots,
                    budget.system_reserve_bytes /
                        (1024.0 * 1024.0 * 1024.0),
                    budget.used_explicit_total ? "explicit-total" : "os-available");
            changed = 1;
        }
    }
    pthread_mutex_unlock(&g_qwen_adaptive.mutex);
    return changed;
}

/* The existing QWEN_ADAPTIVE_RECONCILE macro expands this token only when it is
 * invoked later in qwen_moe.c. Redirect that call without duplicating/replacing
 * the ownership-sensitive reconciliation macro itself. Parenthesized calls in
 * this header still reach the original static function. */
#define qwen_adaptive_adapter_maybe_plan(model, resident_bytes, transient_slots) \
    qwen_adaptive_resource_budget_plan((model), (resident_bytes), (transient_slots))

#endif /* COLIBRI_QWEN_ADAPTIVE_RESOURCE_BUDGET_ADAPTER_H */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * #12 memory-policy overlay for the standalone expert-store planner unit.
 *
 * `V4_PREFIX_CACHE_MB` is an explicit resident snapshot budget. The #3 planner
 * otherwise uses the entire process/OS envelope for runtime + dense/expert
 * residency, so simply allocating the prefix cache afterwards would violate
 * `--memory-gb` by exactly the cache size.
 *
 * Keep the planner implementation unchanged and lower the budget it sees:
 *   - automatic mode: subtract the prefix reserve from MemAvailable;
 *   - explicit --memory-gb mode: subtract it from both the OS bound and the
 *     user process bound, which is equivalent to planning against
 *     min(os_available, user_limit) - prefix_reserve.
 *
 * The cache itself has the same hard byte cap, so planner + cache remains
 * inside the original envelope. Feature-off (`reserve == 0`) is identical to
 * the existing planner.
 */
#include "deepseek_v4_internal.h"
#include "coli_v4_expert_store.h"
#include "coli_v4_prefix_cache.h"
#include "expert_activation_store.h"

#include <stdio.h>

static uint64_t coli_v4_prefix_reserved_available_memory(void) {
    uint64_t available = coli_v4_os_available_memory();
    uint64_t reserve = (uint64_t)coli_v4_prefix_cache_budget_bytes();
    return reserve < available ? available - reserve : 0;
}

/* The standalone planner computes the right final resource plan but historically
 * returned it only through diagnostics; engine->summary retained just the
 * dense/head/expert-cache fields. Capture the plan pointer when the resource
 * calculation starts, then copy its final (post-tier/post-head) projection at
 * the actual expert-store open call while the planner stack frame is still
 * alive. This adds no second inventory pass and gives #1/#12 an authoritative
 * machine-readable projected-memory value. Thread-local state keeps concurrent
 * engine opens independent. */
static _Thread_local ColiDeepSeekV4ResourcePlan *coli_v4_prefix_active_plan;
static _Thread_local ColiDeepSeekV4ResourcePlan coli_v4_prefix_final_plan;
static _Thread_local int coli_v4_prefix_final_plan_valid;

static int coli_v4_prefix_capture_resource_plan(
    ColiDeepSeekV4ResourcePlan *plan,
    const ColiDeepSeekV4ResourceInputs *inputs,
    char *error, size_t error_size) {
    int result = coli_v4_resource_plan_compute(plan, inputs, error, error_size);
    coli_v4_prefix_active_plan = result ? NULL : plan;
    return result;
}

static void coli_v4_prefix_capture_final_plan(int result) {
    if (!result && coli_v4_prefix_active_plan) {
        coli_v4_prefix_final_plan = *coli_v4_prefix_active_plan;
        coli_v4_prefix_final_plan_valid = 1;
    }
}

static int coli_v4_prefix_capture_coli_store_open(
    const ColiV4ColiExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    int result = coli_v4_coli_expert_store_open(
        options, output, error, error_size);
    coli_v4_prefix_capture_final_plan(result);
    return result;
}

static int coli_v4_prefix_capture_legacy_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    int result = coli_deepseek_v4_expert_store_open(
        options, output, error, error_size);
    coli_v4_prefix_capture_final_plan(result);
    return result;
}

#define coli_v4_os_available_memory coli_v4_prefix_reserved_available_memory
#define coli_v4_resource_plan_compute coli_v4_prefix_capture_resource_plan
#define coli_v4_coli_expert_store_open coli_v4_prefix_capture_coli_store_open
#define coli_deepseek_v4_expert_store_open coli_v4_prefix_capture_legacy_store_open
#define coli_v4_expert_store_open_planned \
    coli_v4_expert_store_open_planned_without_prefix_reserve
#include "coli_v4_expert_store_auto.c"
#undef coli_v4_expert_store_open_planned
#undef coli_deepseek_v4_expert_store_open
#undef coli_v4_coli_expert_store_open
#undef coli_v4_resource_plan_compute
#undef coli_v4_os_available_memory

int coli_v4_expert_store_open_planned(
    ColiV4Engine *engine,
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    if (!engine) return -1;
    uint64_t reserve = (uint64_t)coli_v4_prefix_cache_budget_bytes();
    uint64_t original_limit = engine->runtime.memory_limit_bytes;

    coli_v4_prefix_active_plan = NULL;
    coli_v4_prefix_final_plan_valid = 0;
    if (reserve && original_limit) {
        if (reserve >= original_limit) {
            if (error && error_size)
                snprintf(error, error_size,
                         "V4 prefix-cache reserve %.2f GiB exhausts %.2f GiB memory limit",
                         reserve / 1073741824.0,
                         original_limit / 1073741824.0);
            return -1;
        }
        engine->runtime.memory_limit_bytes = original_limit - reserve;
    }

    int result = coli_v4_expert_store_open_planned_without_prefix_reserve(
        engine, options, output, error, error_size);
    engine->runtime.memory_limit_bytes = original_limit;

    if (!result && coli_v4_prefix_final_plan_valid)
        engine->summary.projected_bytes = coli_v4_prefix_final_plan.projected_bytes;
    coli_v4_prefix_active_plan = NULL;
    coli_v4_prefix_final_plan_valid = 0;

    /* Package-mode COLI stores already own the shared tracker/scorer through
     * coli_v4_expert_store_adaptive.c. Keep the generic best-effort wrapper only
     * for the legacy safetensors store until that path migrates into #95. This
     * avoids double tracking and keeps package admission decisions at the same
     * physical residency boundary that owns the slots. */
    if (!result && !engine->coli_static && output && *output && options &&
        options->layers > 0 && options->experts_per_layer > 0 &&
        (size_t)options->layers <= SIZE_MAX / (size_t)options->experts_per_layer) {
        size_t key_hint = (size_t)options->layers *
                          (size_t)options->experts_per_layer;
        ColiExpertStore *wrapped = NULL;
        if (coli_expert_activation_store_wrap(*output, key_hint, &wrapped) == 1)
            *output = wrapped;
    }

    if (!result && reserve && getenv("V4_PREFIX_LOG"))
        fprintf(stderr,
                "[PREFIX-CACHE] planner_reserve=%.2fMiB memory_envelope=accounted\n",
                reserve / (1024.0 * 1024.0));
    return result;
}

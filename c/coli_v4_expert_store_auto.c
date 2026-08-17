#include "deepseek_v4_internal.h"
#include "coli_v4_expert_store.h"
#include "coli_v4_residency.h"
#include "coli_executor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIB UINT64_C(1048576)
#define GIB UINT64_C(1073741824)

static int multiply_u64_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a && b > UINT64_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static uint64_t expert_record_bytes(const ColiSafetensorsIndex *index) {
    static const char *parts[] = {
        "layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
        "layers.0.ffn.experts.0.w2.weight", "layers.0.ffn.experts.0.w2.scale",
        "layers.0.ffn.experts.0.w3.weight", "layers.0.ffn.experts.0.w3.scale",
    };
    uint64_t total = 0;
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); i++) {
        const ColiSafetensorsTensor *tensor = coli_st_find(index, parts[i]);
        if (!tensor || tensor->nbytes < 0) return 0;
        uint64_t bytes = (uint64_t)tensor->nbytes;
        if (UINT64_MAX - total < bytes) return 0;
        total += bytes;
    }
    return total;
}

static uint64_t context_bytes(const ColiDeepSeekV4Config *config, int context) {
    uint64_t total = (uint64_t)config->num_hidden_layers *
        config->sliding_window * config->head_dim * sizeof(float);
    for (int layer = 0; layer < config->num_hidden_layers; layer++) {
        int ratio = config->compress_ratios[layer];
        if (!ratio) continue;
        uint64_t compressed = ((uint64_t)context + (uint64_t)ratio - 1) /
                              (uint64_t)ratio;
        total += compressed * config->head_dim * sizeof(float);
        if (ratio == 4)
            total += compressed * config->index_head_dim * sizeof(float);
    }
    return total;
}

/* Keep the planner's cache floor synchronized with the bounded asynchronous
 * loader pool. The same V4_LOADER_LANES contract is consumed later by the
 * expert pipeline. One extra slot is retained by the current consumer while
 * all loader lanes may be filling replacements. */
static int coli_v4_loader_lane_budget(void) {
    const char *value = getenv("V4_LOADER_LANES");
    int lanes = value ? atoi(value) : COLI_V4_EXPERT_LOADER_COUNT;
    if (lanes < 1) lanes = COLI_V4_EXPERT_LOADER_COUNT;
    if (lanes > 16) lanes = 16;
    return lanes;
}

static int coli_v4_cache_slot_cap(void) {
    const char *value = getenv("V4_COLI_CACHE_SLOTS");
    if (!value || !*value) return 0;
    int cap = atoi(value);
    return cap > 0 ? cap : 0;
}

static int build_runtime_plan(ColiV4Engine *engine,
                              const ColiDeepSeekV4ExpertStoreOptions *options,
                              ColiDeepSeekV4ResourcePlan *plan,
                              char *error, size_t error_size) {
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    if (!engine) {
        snprintf(error, error_size, "V4 runtime requires an engine");
        return -1;
    }
    if (coli_v4_config_load(&config, options->model_dir, error, error_size))
        return -1;
    if (!engine->coli_static &&
        coli_st_index_open(&index, options->model_dir, error, error_size))
        return -1;
    uint64_t maximum_layer = 0;
    for (int layer = 0; layer < config.num_hidden_layers; layer++) {
        uint64_t layer_bytes = 0;
        if (engine->coli_static
            ? coli_v4_coli_layer_bytes(engine->coli_static, &config, layer,
                                       &layer_bytes, error, error_size)
            : ({ ColiDeepSeekV4LayerPlan p; ColiDeepSeekV4LayerStats s;
                 int rc = coli_v4_layer_plan(&p, &config, layer, error, error_size) ||
                          coli_v4_layer_validate(&p, index, &s, error, error_size);
                 if (!rc) layer_bytes = s.total_bytes; rc; })) {
            if (index) coli_st_index_close(index);
            return -1;
        }
        if (layer_bytes > maximum_layer) maximum_layer = layer_bytes;
    }
    uint64_t record = engine->coli_static
        ? (coli_executor_expert(engine->coli_static, 0, 0)
           ? coli_executor_expert(engine->coli_static, 0, 0)->stored_bytes : 0)
        : expert_record_bytes(index);
    if (index) coli_st_index_close(index);
    if (!record) {
        snprintf(error, error_size, "cannot determine V4 expert record size");
        return -1;
    }
    ColiDeepSeekV4RuntimeOptions *runtime = &engine->runtime;
    int context = runtime->context_tokens;
    if (context > config.max_position_embeddings)
        context = config.max_position_embeddings;
    uint64_t hidden = (uint64_t)64 * config.hc_mult * config.hidden_size *
                      sizeof(float) * 2;
    uint64_t scratch = 512 * MIB;
    uint64_t runtime_other = context_bytes(&config, context) + hidden + scratch;
    if (UINT64_MAX - runtime_other < runtime->dspark_reserve_bytes) {
        snprintf(error, error_size, "V4 DSpark reserve overflow");
        return -1;
    }
    runtime_other += runtime->dspark_reserve_bytes;
    uint64_t available = coli_v4_os_available_memory();
    if (!available) {
        snprintf(error, error_size, "cannot determine OS available memory");
        return -1;
    }

    const int package_mode = engine->coli_static != NULL;
    ColiDeepSeekV4ResourceInputs inputs = {
        available, runtime->memory_limit_bytes, maximum_layer,
        runtime_other, record,
        /* Package execution has one global transient expert pool. The old
         * planner multiplied its floor by every sparse layer, which #64 proved
         * is not a physical requirement. Safetensors keeps its legacy geometry. */
        package_mode ? 1 : config.num_hidden_layers,
        package_mode ? coli_v4_loader_lane_budget() + 1
                     : config.num_experts_per_tok,
        config.n_routed_experts,
    };
    return coli_v4_resource_plan_compute(plan, &inputs, error, error_size);
}

static int v5_dense_inventory(ColiV4Engine *engine, const char *model_dir,
                              uint64_t *bytes,
                              char *error, size_t error_size) {
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    if (coli_v4_config_load(&config, model_dir, error, error_size) ||
        (!engine->coli_static &&
         coli_st_index_open(&index, model_dir, error, error_size)))
        return -1;
    uint64_t total = 0;
    for (int layer = 0; layer < config.num_hidden_layers; layer++) {
        uint64_t layer_bytes = 0;
        if (engine->coli_static
            ? coli_v4_coli_layer_bytes(engine->coli_static, &config, layer,
                                       &layer_bytes, error, error_size)
            : ({ ColiDeepSeekV4LayerPlan p; ColiDeepSeekV4LayerStats s;
                 int rc = coli_v4_layer_plan(&p, &config, layer, error, error_size) ||
                          coli_v4_layer_validate(&p, index, &s, error, error_size);
                 if (!rc) layer_bytes = s.total_bytes; rc; })) {
            if (index) coli_st_index_close(index);
            return -1;
        }
        total += layer_bytes;
    }
    if (index) coli_st_index_close(index);
    *bytes = total;
    return 0;
}

/* Compute package-mode residency geometry in actual allocated bytes. The COLI
 * store rounds each final UMA-visible expert slot to 16 KiB, and its transient
 * pool is global rather than `layers * slots`. */
static int package_residency_geometry(
    ColiV4Engine *engine, const ColiDeepSeekV4ExpertStoreOptions *options,
    uint64_t *record_bytes, uint64_t *slot_bytes, int *minimum_slots,
    uint64_t *minimum_expert_bytes, uint64_t *legacy_slot_quantum,
    char *error, size_t error_size) {
    if (!engine || !engine->coli_static || !options || !record_bytes ||
        !slot_bytes || !minimum_slots || !minimum_expert_bytes ||
        !legacy_slot_quantum)
        return -1;
    const ColiRecordInfo *record = coli_executor_expert(engine->coli_static, 0, 0);
    if (!record || !record->stored_bytes ||
        coli_v4_expert_slot_bytes(record->stored_bytes, slot_bytes)) {
        snprintf(error, error_size, "cannot determine aligned COLI expert slot size");
        return -1;
    }
    *record_bytes = record->stored_bytes;
    *minimum_slots = coli_v4_loader_lane_budget() + 1;
    if (multiply_u64_checked(*slot_bytes, (uint64_t)*minimum_slots,
                             minimum_expert_bytes) ||
        multiply_u64_checked(*record_bytes, (uint64_t)options->layers,
                             legacy_slot_quantum)) {
        snprintf(error, error_size, "COLI expert residency geometry overflow");
        return -1;
    }
    return 0;
}

int coli_v4_expert_store_open_planned(
    ColiV4Engine *engine,
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    if (!options || !engine) return -1;
    const int package_mode = engine->coli_static != NULL;
    ColiDeepSeekV4ResourcePlan plan;
    ColiDeepSeekV4RuntimeOptions *runtime = &engine->runtime;
    if (build_runtime_plan(engine, options, &plan, error, error_size)) return -1;

    uint64_t package_record_bytes = 0, package_slot_bytes = 0;
    uint64_t package_minimum_expert_bytes = 0, legacy_slot_quantum = 0;
    int package_minimum_slots = 0;
    if (package_mode && package_residency_geometry(
            engine, options, &package_record_bytes, &package_slot_bytes,
            &package_minimum_slots, &package_minimum_expert_bytes,
            &legacy_slot_quantum, error, error_size))
        return -1;

    uint64_t per_slot = 0;
    if (!package_mode) {
        if (plan.slots_per_layer < 1) {
            snprintf(error, error_size, "invalid V4 expert slot plan");
            return -1;
        }
        per_slot = plan.expert_cache_bytes / (uint64_t)plan.slots_per_layer;
        if (!per_slot) {
            snprintf(error, error_size, "invalid V4 expert slot size");
            return -1;
        }
    }

    uint64_t head_bytes = 0, dense_bytes = 0;
    if (coli_v4_head_cache_probe(engine, options->model_dir, &head_bytes,
                                 error, error_size) ||
        v5_dense_inventory(engine, options->model_dir, &dense_bytes,
                           error, error_size))
        return -1;

    uint64_t fixed = plan.system_reserve_bytes + plan.runtime_reserve_bytes;
    if (fixed > plan.planner_available_bytes) {
        snprintf(error, error_size, "resident V4 fixed tiers exceed available RAM");
        return -1;
    }
    const uint64_t minimum_expert_bytes = package_mode
        ? package_minimum_expert_bytes : plan.minimum_expert_bytes;

    ColiDeepSeekV4ResidentTierPlan tiers;
    ColiDeepSeekV4ResidentTierInputs tier_inputs = {
        plan.planner_available_bytes, fixed, dense_bytes,
        minimum_expert_bytes,
    };
    if (coli_v4_resident_tier_plan(&tiers, &tier_inputs,
                                   error, error_size))
        return -1;
    dense_bytes = tiers.dense_bytes;
    runtime->dense_resident = tiers.dense_resident;
    if (dense_bytes > plan.planner_available_bytes - fixed) {
        snprintf(error, error_size, "resident V4 tiers exceed available RAM");
        return -1;
    }
    uint64_t safe_payload = plan.planner_available_bytes - fixed - dense_bytes;
    int requested_head = -1;
    int resident_head = safe_payload >= minimum_expert_bytes +
                                      head_bytes + 256 * MIB;
    if (requested_head == 0) resident_head = 0;
    if (requested_head == 1 && !resident_head) {
        snprintf(error, error_size, "resident BF16 head does not fit RAM plan");
        return -1;
    }
    uint64_t cache_limit = safe_payload - (resident_head ? head_bytes : 0);

    if (cache_limit < minimum_expert_bytes) {
        snprintf(error, error_size, "resident tiers leave too little target cache");
        return -1;
    }

    int target_slots = 0;
    int requested_cap = package_mode ? coli_v4_cache_slot_cap() : 0;
    if (package_mode) {
        /* #64's store consumes an arbitrary byte pool: it first reserves
         * `loader_lanes + 1` aligned global slots, then gives every remaining
         * byte to persistent/dense residency. Do not quantize that pool through
         * the obsolete one-slot-per-layer geometry here. */
        if (requested_cap && requested_cap < package_minimum_slots) {
            snprintf(error, error_size,
                     "V4_COLI_CACHE_SLOTS=%d is below the %d-slot loader pipeline floor",
                     requested_cap, package_minimum_slots);
            return -1;
        }
        if (requested_cap) {
            uint64_t capped_bytes;
            if (multiply_u64_checked(legacy_slot_quantum,
                                     (uint64_t)requested_cap, &capped_bytes)) {
                snprintf(error, error_size, "V4_COLI_CACHE_SLOTS budget overflow");
                return -1;
            }
            if (cache_limit > capped_bytes) cache_limit = capped_bytes;
            if (cache_limit < minimum_expert_bytes) {
                snprintf(error, error_size,
                         "V4_COLI_CACHE_SLOTS leaves too little transient capacity");
                return -1;
            }
        }
        plan.expert_cache_bytes = cache_limit;
        target_slots = legacy_slot_quantum
            ? (int)(cache_limit / legacy_slot_quantum) : 0;
    } else {
        int slots = (int)(cache_limit / per_slot);
        if (slots > plan.slots_per_layer) slots = plan.slots_per_layer;
        int minimum_slots = 6;
        if (slots < options->experts_per_layer && slots < minimum_slots)
            slots = minimum_slots;
        plan.expert_cache_bytes = (uint64_t)slots * per_slot;
        target_slots = slots;
    }

    runtime->target_expert_cache_bytes = plan.expert_cache_bytes;
    if (UINT64_MAX - fixed < dense_bytes ||
        UINT64_MAX - (fixed + dense_bytes) < plan.expert_cache_bytes ||
        UINT64_MAX - (fixed + dense_bytes + plan.expert_cache_bytes) <
            (resident_head ? head_bytes : 0)) {
        snprintf(error, error_size, "V4 projected memory overflow");
        return -1;
    }
    plan.projected_bytes = fixed + dense_bytes + plan.expert_cache_bytes +
        (resident_head ? head_bytes : 0);
    if (plan.projected_bytes > plan.planner_available_bytes) {
        snprintf(error, error_size, "V4 plan exceeds available RAM");
        return -1;
    }
    if (resident_head &&
        coli_v4_head_cache_load(engine, options->model_dir, error, error_size))
        return -1;

    if (package_mode) {
        fprintf(stderr,
            "v4_planner package=1 transient_slots=%d transient_bytes=%.3fGiB "
            "optional_pool=%.2fGiB legacy_quantum=%.2fGiB cap=%d\n",
            package_minimum_slots,
            minimum_expert_bytes / (double)GIB,
            plan.expert_cache_bytes / (double)GIB,
            legacy_slot_quantum / (double)GIB,
            requested_cap);
    }
    fprintf(stderr,
        "ram_tiers available=%.2fGiB dense=%s(%.2fGiB) "
        "target_slots=%d target_cache=%.2fGiB head=%s projected=%.2fGiB\n",
        plan.planner_available_bytes / (double)GIB,
        tiers.dense_resident ? "resident" : "streamed",
        dense_bytes / (double)GIB,
        target_slots,
        plan.expert_cache_bytes / (double)GIB,
        resident_head ? "resident-bf16" : "streamed-bf16",
        plan.projected_bytes / (double)GIB);

    ColiDeepSeekV4ExpertStoreOptions automatic = *options;
    automatic.cache_bytes = plan.expert_cache_bytes;
    automatic.pin_slots_per_layer = runtime->pin_slots_per_layer;
    automatic.repin_interval = runtime->repin_interval;
    if (runtime->coli_model_dir) {
        return coli_v4_coli_expert_store_open(
            &(ColiV4ColiExpertStoreOptions){
                runtime->coli_model_dir,
                "macos-arm64-metal-apple8-v1",
                automatic.layers, automatic.experts_per_layer,
                automatic.cache_bytes
            }, output, error, error_size);
    }
    return coli_deepseek_v4_expert_store_open(
        &automatic, output, error, error_size);
}

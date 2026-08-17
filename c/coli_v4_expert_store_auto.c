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
 * loader pool. The same V4_LOADER_LANES contract is consumed later by
 * dual_loader_lanes(); exposing it here prevents a low-memory plan from
 * opening fewer resident slots than outstanding reads. */
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
    ColiDeepSeekV4ResourceInputs inputs = {
        available, runtime->memory_limit_bytes, maximum_layer,
        runtime_other, record, config.num_hidden_layers,
        /* The pipeline queues replacement N+lanes before releasing current
         * N, so its bounded resident floor is loader lanes plus one. It still
         * never needs every top-k expert resident. */
        engine->coli_static
            ? coli_v4_loader_lane_budget() + 1
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

int coli_v4_expert_store_open_planned(
    ColiV4Engine *engine,
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    if (!options || !engine) return -1;
    ColiDeepSeekV4ResourcePlan plan;
    ColiDeepSeekV4RuntimeOptions *runtime = &engine->runtime;
    if (build_runtime_plan(engine, options, &plan, error, error_size)) return -1;
    uint64_t per_slot = plan.expert_cache_bytes /
                        (uint64_t)plan.slots_per_layer;
    uint64_t head_bytes = 0, dense_bytes = 0;
    if (coli_v4_head_cache_probe(engine, options->model_dir, &head_bytes,
                                 error, error_size) ||
        v5_dense_inventory(engine, options->model_dir, &dense_bytes,
                           error, error_size))
        return -1;

    uint64_t fixed = plan.system_reserve_bytes + plan.runtime_reserve_bytes;
    ColiDeepSeekV4ResidentTierPlan tiers;
    ColiDeepSeekV4ResidentTierInputs tier_inputs = {
        plan.planner_available_bytes, fixed, dense_bytes,
        plan.minimum_expert_bytes,
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
    int resident_head = safe_payload >= plan.minimum_expert_bytes +
                                      head_bytes + 256 * MIB;
    if (requested_head == 0) resident_head = 0;
    if (requested_head == 1 && !resident_head) {
        snprintf(error, error_size, "resident BF16 head does not fit RAM plan");
        return -1;
    }
    uint64_t cache_limit = safe_payload - (resident_head ? head_bytes : 0);

    if (cache_limit < plan.minimum_expert_bytes) {
        snprintf(error, error_size, "resident tiers leave too little target cache");
        return -1;
    }
    int slots = (int)(cache_limit / per_slot);
    if (slots > plan.slots_per_layer) slots = plan.slots_per_layer;
    int minimum_slots = engine->coli_static ? coli_v4_loader_lane_budget() + 1 : 6;
    int requested_cap = engine->coli_static ? coli_v4_cache_slot_cap() : 0;
    if (requested_cap && slots > requested_cap) slots = requested_cap;
    if (slots < options->experts_per_layer && slots < minimum_slots)
        slots = minimum_slots;
    if (requested_cap && requested_cap < minimum_slots) {
        snprintf(error, error_size,
                 "V4_COLI_CACHE_SLOTS=%d is below the %d-slot loader pipeline floor",
                 requested_cap, minimum_slots);
        return -1;
    }
    plan.expert_cache_bytes = (uint64_t)slots * per_slot;
    runtime->target_expert_cache_bytes = plan.expert_cache_bytes;
    plan.projected_bytes = fixed + dense_bytes +
        plan.expert_cache_bytes + (resident_head ? head_bytes : 0);
    if (resident_head &&
        coli_v4_head_cache_load(engine, options->model_dir, error, error_size))
        return -1;
    fprintf(stderr,
        "ram_tiers available=%.2fGiB dense=%s(%.2fGiB) "
        "target_slots=%d target_cache=%.2fGiB head=%s projected=%.2fGiB\n",
        plan.planner_available_bytes / (double)GIB,
        tiers.dense_resident ? "resident" : "streamed",
        dense_bytes / (double)GIB,
        slots,
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

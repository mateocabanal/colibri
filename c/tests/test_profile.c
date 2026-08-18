#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "../profile.h"
#include "../metal_policy.h"
#include "../metal_runtime.h"
#include "../metal_region.h"
#include "../expert_residency.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const ColiProfilePhaseDef phases[] = {
    {"compute", COLI_PROFILE_ACCOUNTED},
    {"disk", COLI_PROFILE_IO_WAIT},
    {"gpu_kernel", 0},
};
static const ColiProfileCounterDef counters[] = {
    {"bytes"}, {"hits"},
};

static int fake_metal_init_result;
static int fake_metal_init_calls;
static int fake_metal_shutdown_calls;
static int fake_metal_init(void) {
    fake_metal_init_calls++;
    return fake_metal_init_result;
}
static void fake_metal_shutdown(void) {
    fake_metal_shutdown_calls++;
}

static int test_metal_policy(void) {
    ColiMetalPolicyResolution r;
    ColiMetalRuntime runtime;
    char error[160];

    if (coli_metal_policy_resolve_values(
            "strict", "QWEN_METAL_COMPUTE", "0", COLI_METAL_POLICY_OFF,
            &r, error, sizeof(error)) != 0 ||
        r.policy != COLI_METAL_POLICY_STRICT ||
        r.source != COLI_METAL_POLICY_SOURCE_GLOBAL ||
        strcmp(r.source_name, "COLI_METAL") != 0)
        return 20;

    if (coli_metal_policy_resolve_values(
            NULL, "QWEN_METAL_COMPUTE", "1", COLI_METAL_POLICY_OFF,
            &r, error, sizeof(error)) != 0 ||
        r.policy != COLI_METAL_POLICY_ON ||
        r.source != COLI_METAL_POLICY_SOURCE_LEGACY)
        return 21;

    if (coli_metal_policy_resolve_values(
            NULL, "V4_METAL_EXPERTS", "0", COLI_METAL_POLICY_AUTO,
            &r, error, sizeof(error)) != 0 ||
        r.policy != COLI_METAL_POLICY_OFF ||
        r.source != COLI_METAL_POLICY_SOURCE_LEGACY)
        return 22;

    if (coli_metal_policy_resolve_values(
            NULL, NULL, NULL, COLI_METAL_POLICY_AUTO,
            &r, error, sizeof(error)) != 0 ||
        r.policy != COLI_METAL_POLICY_AUTO ||
        r.source != COLI_METAL_POLICY_SOURCE_DEFAULT)
        return 23;

    if (coli_metal_policy_resolve_values(
            "OFF", "QWEN_METAL_COMPUTE", "1", COLI_METAL_POLICY_AUTO,
            &r, error, sizeof(error)) != 0 ||
        r.policy != COLI_METAL_POLICY_OFF)
        return 24;

    memset(error, 0, sizeof(error));
    if (coli_metal_policy_resolve_values(
            "sometimes", NULL, NULL, COLI_METAL_POLICY_OFF,
            &r, error, sizeof(error)) == 0 ||
        !strstr(error, "invalid COLI_METAL"))
        return 25;

    memset(error, 0, sizeof(error));
    if (coli_metal_policy_resolve_values(
            NULL, "QWEN_METAL_COMPUTE", "strict", COLI_METAL_POLICY_OFF,
            &r, error, sizeof(error)) == 0 ||
        !strstr(error, "legacy Metal flags accept 0|1"))
        return 26;

    if (strcmp(coli_metal_policy_name(COLI_METAL_POLICY_AUTO), "auto") != 0 ||
        !coli_metal_policy_should_init(COLI_METAL_POLICY_AUTO) ||
        coli_metal_policy_should_init(COLI_METAL_POLICY_OFF) ||
        !coli_metal_policy_is_strict(COLI_METAL_POLICY_STRICT))
        return 27;

    fake_metal_init_result = 1;
    fake_metal_init_calls = fake_metal_shutdown_calls = 0;
    if (coli_metal_runtime_open_values(
            &runtime, 1, "auto", NULL, NULL, COLI_METAL_POLICY_OFF,
            fake_metal_init, error, sizeof(error)) != 0 ||
        !coli_metal_runtime_enabled(&runtime) || fake_metal_init_calls != 1)
        return 28;
    coli_metal_runtime_close(&runtime, fake_metal_shutdown);
    if (fake_metal_shutdown_calls != 1 || coli_metal_runtime_enabled(&runtime))
        return 29;

    fake_metal_init_result = 0;
    fake_metal_init_calls = 0;
    if (coli_metal_runtime_open_values(
            &runtime, 1, "auto", NULL, NULL, COLI_METAL_POLICY_OFF,
            fake_metal_init, error, sizeof(error)) != 0 ||
        coli_metal_runtime_enabled(&runtime) || fake_metal_init_calls != 1)
        return 30;

    memset(error, 0, sizeof(error));
    if (coli_metal_runtime_open_values(
            &runtime, 1, "strict", NULL, NULL, COLI_METAL_POLICY_OFF,
            fake_metal_init, error, sizeof(error)) == 0 ||
        !strstr(error, "backend is unavailable"))
        return 31;

    fake_metal_init_calls = 0;
    if (coli_metal_runtime_open_values(
            &runtime, 1, "0", NULL, NULL, COLI_METAL_POLICY_AUTO,
            fake_metal_init, error, sizeof(error)) != 0 ||
        fake_metal_init_calls != 0 || coli_metal_runtime_enabled(&runtime))
        return 32;

    return 0;
}

static int test_metal_region(void) {
    unsigned char storage[256];
    ColiMetalRegion region;
    ColiMetalRegionRef old_ref, new_ref;

    if (coli_metal_region_init(&region, storage, sizeof(storage), 7) != 0)
        return 40;
    if (coli_metal_region_generation(&region) != 0 ||
        coli_metal_region_inflight(&region) != 0 ||
        coli_metal_region_can_overwrite(&region) == 0)
        return 41;
    if (coli_metal_region_ref(&region, 1, 0, 16, &old_ref) == 0)
        return 42;

    if (coli_metal_region_publish(&region, 1) != 0 ||
        coli_metal_region_generation(&region) != 1)
        return 43;
    if (coli_metal_region_ref(&region, 1, 32, 64, &old_ref) != 0 ||
        old_ref.region_id != 7 || old_ref.generation != 1 ||
        old_ref.offset != 32 || old_ref.bytes != 64 ||
        !coli_metal_region_ref_matches(&region, &old_ref))
        return 44;
    if (coli_metal_region_ref(&region, 1, 240, 32, &new_ref) == 0)
        return 45;

    if (coli_metal_region_retain(&region, 1) != 0 ||
        coli_metal_region_inflight(&region) != 1)
        return 46;

    coli_metal_region_begin_overwrite(&region);
    if (coli_metal_region_ref_matches(&region, &old_ref) ||
        coli_metal_region_retain(&region, 1) == 0 ||
        coli_metal_region_can_overwrite(&region))
        return 47;

    if (coli_metal_region_release(&region, 1) != 0 ||
        coli_metal_region_inflight(&region) != 0 ||
        !coli_metal_region_can_overwrite(&region))
        return 48;
    if (coli_metal_region_release(&region, 1) == 0)
        return 49;

    if (coli_metal_region_publish(&region, 2) != 0 ||
        coli_metal_region_generation(&region) != 2 ||
        coli_metal_region_ref_matches(&region, &old_ref))
        return 50;
    if (coli_metal_region_ref(&region, 2, 32, 64, &new_ref) != 0 ||
        !coli_metal_region_ref_matches(&region, &new_ref) ||
        coli_metal_region_retain(&region, 1) == 0 ||
        coli_metal_region_retain(&region, 2) != 0)
        return 51;
    if (coli_metal_region_release(&region, 2) != 0)
        return 52;

    coli_metal_region_begin_overwrite(&region);
    if (!coli_metal_region_can_overwrite(&region) ||
        coli_metal_region_publish(&region, 3) != 0 ||
        coli_metal_region_generation(&region) != 3)
        return 53;

    return 0;
}

static int test_expert_residency(void) {
    ColiExpertResidencyBudget budget;
    ColiExpertResidencyEntry a, b;
    ColiExpertResidencyLease lease;
    ColiExpertKey ka = {3, 11}, kb = {3, 12};

    coli_expert_residency_budget_init(&budget, 100);
    if (coli_expert_residency_entry_init(&a, ka) != 0 ||
        coli_expert_residency_entry_init(&b, kb) != 0)
        return 60;

    if (coli_expert_residency_request(&a, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_budget_committed(&budget) != 60 ||
        atomic_load(&budget.reserved_bytes) != 60 ||
        coli_expert_residency_state(&a) != COLI_EXPERT_RESIDENCY_RESERVED)
        return 61;

    if (coli_expert_residency_request(&a, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_JOIN_INFLIGHT ||
        coli_expert_residency_budget_committed(&budget) != 60)
        return 62;

    if (coli_expert_residency_request(&b, &budget, 50, &lease) !=
            COLI_EXPERT_REQUEST_NO_BUDGET ||
        coli_expert_residency_state(&b) != COLI_EXPERT_RESIDENCY_COLD ||
        coli_expert_residency_budget_committed(&budget) != 60)
        return 63;

    if (coli_expert_residency_mark_loading(&a) != 0 ||
        coli_expert_residency_mark_preparing(&a) != 0 ||
        coli_expert_residency_publish(&a, &budget, 1, COLI_EXPERT_TIER_UMA) != 0 ||
        atomic_load(&budget.reserved_bytes) != 0 ||
        atomic_load(&budget.resident_bytes) != 60 ||
        coli_expert_residency_budget_committed(&budget) != 60)
        return 64;

    if (coli_expert_residency_request(&a, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_HIT ||
        lease.generation != 1 || lease.tier_mask != COLI_EXPERT_TIER_UMA ||
        !coli_expert_key_equal(lease.key, ka) || atomic_load(&a.refs) != 1)
        return 65;

    if (coli_expert_residency_begin_evict(&a) != 0 ||
        coli_expert_residency_release(&lease) != 0 ||
        atomic_load(&a.refs) != 0 ||
        coli_expert_residency_begin_evict(&a) != 1 ||
        coli_expert_residency_finish_evict(&a, &budget) != 0 ||
        coli_expert_residency_state(&a) != COLI_EXPERT_RESIDENCY_COLD ||
        coli_expert_residency_budget_committed(&budget) != 0)
        return 66;

    if (coli_expert_residency_request(&a, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_mark_loading(&a) != 0 ||
        coli_expert_residency_fail_load(&a, &budget) != 0 ||
        coli_expert_residency_state(&a) != COLI_EXPERT_RESIDENCY_COLD ||
        coli_expert_residency_budget_committed(&budget) != 0)
        return 67;

    if (coli_expert_residency_request(&b, &budget, 50, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish(&b, &budget, 9,
            COLI_EXPERT_TIER_PINNED_HOST | COLI_EXPERT_TIER_DEVICE) != 0 ||
        coli_expert_residency_request(&b, &budget, 50, &lease) !=
            COLI_EXPERT_REQUEST_HIT || lease.generation != 9 ||
        (lease.tier_mask & COLI_EXPERT_TIER_DEVICE) == 0)
        return 68;
    if (coli_expert_residency_release(&lease) != 0 ||
        atomic_load(&budget.peak_committed_bytes) > budget.capacity_bytes ||
        atomic_load(&budget.peak_committed_bytes) != 60)
        return 69;

    return 0;
}

int main(void) {
    int rc = test_metal_policy();
    if (rc) return rc;
    rc = test_metal_region();
    if (rc) return rc;
    rc = test_expert_residency();
    if (rc) return rc;

#ifdef _WIN32
    _putenv_s("COLI_PROFILE", "1");
#else
    setenv("COLI_PROFILE", "1", 1);
#endif
    ColiProfileConfig config = {
        .engine = "test",
        .env_name = "TEST_PROFILE",
        .line_prefix = "coli_profile",
        .include_engine = 1,
        .phases = phases,
        .phase_count = sizeof(phases) / sizeof(phases[0]),
        .counters = counters,
        .counter_count = sizeof(counters) / sizeof(counters[0]),
    };
    ColiProfile p;
    coli_profile_reset(&p, &config);
    if (!coli_profile_enabled(&p)) return 1;
    coli_profile_mark(&p, 0);
    coli_profile_mark_tokens(&p, 0, 10);
    coli_profile_add(&p, 0, 7000000);
    coli_profile_add(&p, 1, 2000000);
    coli_profile_add(&p, 2, 9000000);
    coli_profile_counter_add(&p, 0, 4096);
    coli_profile_counter_add(&p, 1, 3);
    coli_profile_mark(&p, 1);
    coli_profile_mark_tokens(&p, 1, 12);

    FILE *f = tmpfile();
    if (!f) return 2;
    coli_profile_emit_scope(f, &p, "run", 0, 1);
    rewind(f);
    char line[2048] = {0};
    if (!fgets(line, sizeof(line), f)) return 3;
    fclose(f);
    if (!strstr(line, "coli_profile engine=test scope=run tokens=2") ||
        !strstr(line, "compute_ms=7.000") ||
        !strstr(line, "disk_ms=2.000") ||
        !strstr(line, "gpu_kernel_ms=9.000") ||
        !strstr(line, "bytes=4096") || !strstr(line, "hits=3")) return 4;

#ifdef _WIN32
    _putenv_s("COLI_PROFILE", "0");
#else
    setenv("COLI_PROFILE", "0", 1);
#endif
    coli_profile_reset(&p, &config);
    if (coli_profile_enabled(&p)) return 5;
    puts("PASS shared runtime policy/regions/residency/profile");
    return 0;
}

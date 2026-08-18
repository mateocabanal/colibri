#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "../profile.h"
#include "../metal_policy.h"

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

static int test_metal_policy(void) {
    ColiMetalPolicyResolution r;
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

    return 0;
}

int main(void) {
    int metal_rc = test_metal_policy();
    if (metal_rc) return metal_rc;

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
    puts("PASS generic profile + metal policy");
    return 0;
}

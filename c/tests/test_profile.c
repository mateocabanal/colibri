#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "../profile.h"

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

int main(void) {
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
    puts("PASS generic profile");
    return 0;
}

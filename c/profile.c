#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "profile.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

uint64_t coli_profile_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int env_enabled(const ColiProfileConfig *config) {
    const char *value = NULL;
    if (config && config->env_name && config->env_name[0])
        value = getenv(config->env_name);
    if (!value || !value[0]) value = getenv("COLI_PROFILE");
    return value && value[0] && atoi(value) != 0;
}

void coli_profile_reset(ColiProfile *profile, const ColiProfileConfig *config) {
    if (!profile) return;
    memset(profile, 0, sizeof(*profile));
    if (!config || !config->phases || config->phase_count > COLI_PROFILE_MAX_PHASES ||
        config->counter_count > COLI_PROFILE_MAX_COUNTERS)
        return;
    profile->config = *config;
    profile->enabled = env_enabled(config);
    profile->wall_base_ns = profile->enabled ? coli_profile_now_ns() : 0;
}

int coli_profile_enabled(const ColiProfile *profile) {
    return profile && profile->enabled;
}

void coli_profile_add(ColiProfile *profile, size_t phase, uint64_t ns) {
    if (!profile || !profile->enabled || phase >= profile->config.phase_count) return;
    profile->current.phase_ns[phase] += ns;
}

void coli_profile_phase_set(ColiProfile *profile, size_t phase, uint64_t ns) {
    if (!profile || !profile->enabled || phase >= profile->config.phase_count) return;
    profile->current.phase_ns[phase] = ns;
}

void coli_profile_counter_add(ColiProfile *profile, size_t counter, uint64_t value) {
    if (!profile || !profile->enabled || counter >= profile->config.counter_count) return;
    profile->current.counters[counter] += value;
}

void coli_profile_counter_set(ColiProfile *profile, size_t counter, uint64_t value) {
    if (!profile || !profile->enabled || counter >= profile->config.counter_count) return;
    profile->current.counters[counter] = value;
}

ColiProfileSnapshot coli_profile_get(ColiProfile *profile) {
    ColiProfileSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    if (!profile || !profile->enabled) return snapshot;
    snapshot = profile->current;
    snapshot.wall_ns = coli_profile_now_ns() - profile->wall_base_ns;
    return snapshot;
}

void coli_profile_mark(ColiProfile *profile, size_t slot) {
    if (!profile || !profile->enabled || slot >= COLI_PROFILE_MAX_MARKS) return;
    profile->marks[slot] = coli_profile_get(profile);
    profile->mark_valid[slot] = 1;
}

void coli_profile_mark_tokens(ColiProfile *profile, size_t slot, uint64_t tokens) {
    if (!profile || !profile->enabled || slot >= COLI_PROFILE_MAX_MARKS) return;
    if (!profile->mark_valid[slot]) coli_profile_mark(profile, slot);
    profile->marks[slot].tokens = tokens;
}

void coli_profile_mark_counter_set(ColiProfile *profile, size_t slot,
                                   size_t counter, uint64_t value) {
    if (!profile || !profile->enabled || slot >= COLI_PROFILE_MAX_MARKS ||
        counter >= profile->config.counter_count) return;
    if (!profile->mark_valid[slot]) coli_profile_mark(profile, slot);
    profile->marks[slot].counters[counter] = value;
}

ColiProfileSnapshot coli_profile_mark_get(const ColiProfile *profile, size_t slot) {
    ColiProfileSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    if (!profile || !profile->enabled || slot >= COLI_PROFILE_MAX_MARKS ||
        !profile->mark_valid[slot]) return snapshot;
    return profile->marks[slot];
}

static uint64_t delta_u64(uint64_t end, uint64_t start) {
    return end >= start ? end - start : 0;
}

void coli_profile_emit_scope(FILE *stream, const ColiProfile *profile,
                             const char *scope, size_t start_slot,
                             size_t end_slot) {
    if (!stream || !profile || !profile->enabled || !scope ||
        start_slot >= COLI_PROFILE_MAX_MARKS || end_slot >= COLI_PROFILE_MAX_MARKS ||
        !profile->mark_valid[start_slot] || !profile->mark_valid[end_slot]) return;

    const ColiProfileSnapshot *start = &profile->marks[start_slot];
    const ColiProfileSnapshot *end = &profile->marks[end_slot];
    uint64_t wall = delta_u64(end->wall_ns, start->wall_ns);
    uint64_t accounted = 0;
    uint64_t io_wait = 0;
    for (size_t i = 0; i < profile->config.phase_count; i++) {
        uint64_t ns = delta_u64(end->phase_ns[i], start->phase_ns[i]);
        if (profile->config.phases[i].flags & COLI_PROFILE_ACCOUNTED)
            accounted += ns;
        if (profile->config.phases[i].flags & COLI_PROFILE_IO_WAIT)
            io_wait += ns;
    }
    int64_t unaccounted = (int64_t)wall - (int64_t)accounted;
    uint64_t cpu_compute = accounted > io_wait ? accounted - io_wait : 0;
    const char *prefix = profile->config.line_prefix && profile->config.line_prefix[0]
        ? profile->config.line_prefix : "coli_profile";
    fprintf(stream, "%s", prefix);
    if (profile->config.include_engine && profile->config.engine)
        fprintf(stream, " engine=%s", profile->config.engine);
    fprintf(stream,
            " scope=%s tokens=%llu wall_ms=%.3f accounted_ms=%.3f "
            "unaccounted_ms=%+.3f cpu_compute_ms=%.3f io_wait_ms=%.3f",
            scope,
            (unsigned long long)delta_u64(end->tokens, start->tokens),
            wall / 1e6, accounted / 1e6, unaccounted / 1e6,
            cpu_compute / 1e6, io_wait / 1e6);
    for (size_t i = 0; i < profile->config.phase_count; i++) {
        uint64_t ns = delta_u64(end->phase_ns[i], start->phase_ns[i]);
        fprintf(stream, " %s_ms=%.3f", profile->config.phases[i].name, ns / 1e6);
    }
    for (size_t i = 0; i < profile->config.counter_count; i++) {
        uint64_t value = delta_u64(end->counters[i], start->counters[i]);
        fprintf(stream, " %s=%llu", profile->config.counters[i].name,
                (unsigned long long)value);
    }
    fputc('\n', stream);
}

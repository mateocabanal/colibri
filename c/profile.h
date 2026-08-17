#ifndef COLIBRI_PROFILE_H
#define COLIBRI_PROFILE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_PROFILE_MAX_PHASES 32
#define COLI_PROFILE_MAX_COUNTERS 24
#define COLI_PROFILE_MAX_MARKS 8

/* ACCOUNTED contributes to owner-thread wall reconciliation.
 * IO_WAIT is blocking I/O time and therefore also CPU-wait time.
 * CPU_WAIT marks other synchronous waits where the owner thread is blocked
 * rather than computing (for example waiting on a GPU command buffer).
 * Diagnostic phases may omit ACCOUNTED while still carrying either wait flag
 * when they are nested inside an accounted owner span. */
enum {
    COLI_PROFILE_ACCOUNTED = 1u << 0,
    COLI_PROFILE_IO_WAIT   = 1u << 1,
    COLI_PROFILE_CPU_WAIT  = 1u << 2,
};

typedef struct {
    const char *name;
    unsigned flags;
} ColiProfilePhaseDef;

typedef struct {
    const char *name;
} ColiProfileCounterDef;

typedef struct {
    const char *engine;
    const char *env_name;       /* engine-specific opt-in, e.g. V4_PROFILE */
    const char *line_prefix;    /* e.g. coli_profile or v4_phases */
    int include_engine;         /* include engine=<name> in emitted rows */
    const ColiProfilePhaseDef *phases;
    size_t phase_count;
    const ColiProfileCounterDef *counters;
    size_t counter_count;
} ColiProfileConfig;

typedef struct {
    uint64_t phase_ns[COLI_PROFILE_MAX_PHASES];
    uint64_t counters[COLI_PROFILE_MAX_COUNTERS];
    uint64_t wall_ns;
    uint64_t tokens;
} ColiProfileSnapshot;

typedef struct {
    ColiProfileConfig config;
    ColiProfileSnapshot current;
    ColiProfileSnapshot marks[COLI_PROFILE_MAX_MARKS];
    uint8_t mark_valid[COLI_PROFILE_MAX_MARKS];
    uint64_t wall_base_ns;
    int enabled;
} ColiProfile;

uint64_t coli_profile_now_ns(void);
void coli_profile_reset(ColiProfile *profile, const ColiProfileConfig *config);
int coli_profile_enabled(const ColiProfile *profile);
void coli_profile_add(ColiProfile *profile, size_t phase, uint64_t ns);
void coli_profile_phase_set(ColiProfile *profile, size_t phase, uint64_t ns);
void coli_profile_counter_add(ColiProfile *profile, size_t counter, uint64_t value);
void coli_profile_counter_set(ColiProfile *profile, size_t counter, uint64_t value);
ColiProfileSnapshot coli_profile_get(ColiProfile *profile);
void coli_profile_mark(ColiProfile *profile, size_t slot);
void coli_profile_mark_tokens(ColiProfile *profile, size_t slot, uint64_t tokens);
void coli_profile_mark_counter_set(ColiProfile *profile, size_t slot,
                                   size_t counter, uint64_t value);
ColiProfileSnapshot coli_profile_mark_get(const ColiProfile *profile, size_t slot);
void coli_profile_emit_scope(FILE *stream, const ColiProfile *profile,
                             const char *scope, size_t start_slot,
                             size_t end_slot);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_PROFILE_H */

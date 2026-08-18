#ifndef COLIBRI_METAL_RUNTIME_H
#define COLIBRI_METAL_RUNTIME_H

#include "metal_policy.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*ColiMetalInitFn)(void);
typedef void (*ColiMetalShutdownFn)(void);

typedef struct {
    ColiMetalPolicyResolution resolution;
    int compiled;
    int initialized;
    int available;
} ColiMetalRuntime;

static inline void coli_metal_runtime_reset(ColiMetalRuntime *runtime) {
    if (runtime) memset(runtime, 0, sizeof(*runtime));
}

/* Common engine lifecycle entry point. `compiled` is the engine/build's Metal
 * capability, not a device probe. In auto/on mode an unavailable device is a
 * safe CPU fallback; strict turns the same condition into an error. */
static inline int coli_metal_runtime_open_values(
    ColiMetalRuntime *runtime,
    int compiled,
    const char *global_value,
    const char *legacy_name,
    const char *legacy_value,
    ColiMetalPolicy default_policy,
    ColiMetalInitFn init_fn,
    char *error,
    size_t error_size) {
    if (!runtime) return -1;
    coli_metal_runtime_reset(runtime);
    runtime->compiled = compiled ? 1 : 0;
    if (coli_metal_policy_resolve_values(
            global_value, legacy_name, legacy_value, default_policy,
            &runtime->resolution, error, error_size) != 0)
        return -1;

    if (runtime->resolution.policy == COLI_METAL_POLICY_OFF)
        return 0;

    if (!runtime->compiled || !init_fn || !init_fn()) {
        if (coli_metal_policy_is_strict(runtime->resolution.policy)) {
            if (error && error_size)
                snprintf(error, error_size,
                         "COLI_METAL=strict requested Metal but the backend is unavailable");
            return -1;
        }
        return 0;
    }

    runtime->initialized = 1;
    runtime->available = 1;
    return 0;
}

static inline int coli_metal_runtime_open(
    ColiMetalRuntime *runtime,
    int compiled,
    const char *legacy_name,
    ColiMetalInitFn init_fn,
    char *error,
    size_t error_size) {
    const char *legacy_value = legacy_name ? getenv(legacy_name) : NULL;
    return coli_metal_runtime_open_values(
        runtime, compiled, getenv("COLI_METAL"), legacy_name, legacy_value,
        coli_metal_platform_default_policy(), init_fn, error, error_size);
}

static inline void coli_metal_runtime_close(ColiMetalRuntime *runtime,
                                            ColiMetalShutdownFn shutdown_fn) {
    if (!runtime) return;
    if (runtime->initialized && shutdown_fn) shutdown_fn();
    runtime->initialized = 0;
    runtime->available = 0;
}

static inline int coli_metal_runtime_enabled(const ColiMetalRuntime *runtime) {
    return runtime && runtime->initialized && runtime->available;
}

static inline int coli_metal_runtime_strict(const ColiMetalRuntime *runtime) {
    return runtime && coli_metal_policy_is_strict(runtime->resolution.policy);
}

#endif

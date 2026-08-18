#ifndef COLIBRI_METAL_POLICY_H
#define COLIBRI_METAL_POLICY_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    COLI_METAL_POLICY_OFF = 0,
    COLI_METAL_POLICY_AUTO = 1,
    COLI_METAL_POLICY_ON = 2,
    COLI_METAL_POLICY_STRICT = 3,
} ColiMetalPolicy;

typedef enum {
    COLI_METAL_POLICY_SOURCE_DEFAULT = 0,
    COLI_METAL_POLICY_SOURCE_GLOBAL = 1,
    COLI_METAL_POLICY_SOURCE_LEGACY = 2,
} ColiMetalPolicySource;

typedef struct {
    ColiMetalPolicy policy;
    ColiMetalPolicySource source;
    const char *source_name;
} ColiMetalPolicyResolution;

static inline int coli_metal_ascii_eq(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return *a == 0 && *b == 0;
}

static inline const char *coli_metal_policy_name(ColiMetalPolicy policy) {
    switch (policy) {
        case COLI_METAL_POLICY_OFF: return "off";
        case COLI_METAL_POLICY_AUTO: return "auto";
        case COLI_METAL_POLICY_ON: return "on";
        case COLI_METAL_POLICY_STRICT: return "strict";
        default: return "invalid";
    }
}

static inline ColiMetalPolicy coli_metal_platform_default_policy(void) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    return COLI_METAL_POLICY_AUTO;
#else
    return COLI_METAL_POLICY_OFF;
#endif
}

static inline int coli_metal_parse_global_value(const char *value,
                                                 ColiMetalPolicy *policy) {
    if (!value || !*value || !policy) return -1;
    if (coli_metal_ascii_eq(value, "0") || coli_metal_ascii_eq(value, "off") ||
        coli_metal_ascii_eq(value, "false") || coli_metal_ascii_eq(value, "no")) {
        *policy = COLI_METAL_POLICY_OFF;
        return 0;
    }
    if (coli_metal_ascii_eq(value, "auto")) {
        *policy = COLI_METAL_POLICY_AUTO;
        return 0;
    }
    if (coli_metal_ascii_eq(value, "1") || coli_metal_ascii_eq(value, "on") ||
        coli_metal_ascii_eq(value, "true") || coli_metal_ascii_eq(value, "yes")) {
        *policy = COLI_METAL_POLICY_ON;
        return 0;
    }
    if (coli_metal_ascii_eq(value, "strict")) {
        *policy = COLI_METAL_POLICY_STRICT;
        return 0;
    }
    return -1;
}

static inline int coli_metal_parse_legacy_value(const char *value,
                                                 ColiMetalPolicy *policy) {
    if (!value || !*value || !policy) return -1;
    if (coli_metal_ascii_eq(value, "0") || coli_metal_ascii_eq(value, "off") ||
        coli_metal_ascii_eq(value, "false") || coli_metal_ascii_eq(value, "no")) {
        *policy = COLI_METAL_POLICY_OFF;
        return 0;
    }
    if (coli_metal_ascii_eq(value, "1") || coli_metal_ascii_eq(value, "on") ||
        coli_metal_ascii_eq(value, "true") || coli_metal_ascii_eq(value, "yes")) {
        *policy = COLI_METAL_POLICY_ON;
        return 0;
    }
    return -1;
}

/* Resolve policy without reading process state. This is the unit-testable core
 * and is also useful to launchers that already have an environment/config view.
 * Explicit COLI_METAL always wins over an engine-specific compatibility alias. */
static inline int coli_metal_policy_resolve_values(
    const char *global_value,
    const char *legacy_name,
    const char *legacy_value,
    ColiMetalPolicy default_policy,
    ColiMetalPolicyResolution *out,
    char *error,
    size_t error_size) {
    if (!out) return -1;
    if (global_value && *global_value) {
        ColiMetalPolicy policy;
        if (coli_metal_parse_global_value(global_value, &policy) != 0) {
            if (error && error_size)
                snprintf(error, error_size,
                         "invalid COLI_METAL='%s' (expected auto|0|1|strict)",
                         global_value);
            return -1;
        }
        out->policy = policy;
        out->source = COLI_METAL_POLICY_SOURCE_GLOBAL;
        out->source_name = "COLI_METAL";
        return 0;
    }
    if (legacy_value && *legacy_value) {
        ColiMetalPolicy policy;
        if (coli_metal_parse_legacy_value(legacy_value, &policy) != 0) {
            if (error && error_size)
                snprintf(error, error_size,
                         "invalid %s='%s' (legacy Metal flags accept 0|1)",
                         legacy_name ? legacy_name : "Metal legacy flag", legacy_value);
            return -1;
        }
        out->policy = policy;
        out->source = COLI_METAL_POLICY_SOURCE_LEGACY;
        out->source_name = legacy_name ? legacy_name : "legacy";
        return 0;
    }
    out->policy = default_policy;
    out->source = COLI_METAL_POLICY_SOURCE_DEFAULT;
    out->source_name = "platform-default";
    return 0;
}

static inline int coli_metal_policy_resolve_env(const char *legacy_name,
                                                 ColiMetalPolicyResolution *out,
                                                 char *error,
                                                 size_t error_size) {
    const char *legacy_value = legacy_name ? getenv(legacy_name) : NULL;
    return coli_metal_policy_resolve_values(
        getenv("COLI_METAL"), legacy_name, legacy_value,
        coli_metal_platform_default_policy(), out, error, error_size);
}

static inline int coli_metal_policy_should_init(ColiMetalPolicy policy) {
    return policy != COLI_METAL_POLICY_OFF;
}

static inline int coli_metal_policy_is_strict(ColiMetalPolicy policy) {
    return policy == COLI_METAL_POLICY_STRICT;
}

#endif

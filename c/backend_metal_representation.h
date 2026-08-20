#ifndef COLIBRI_BACKEND_METAL_REPRESENTATION_H
#define COLIBRI_BACKEND_METAL_REPRESENTATION_H

#include "expert_residency.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Translation only: this is not another representation namespace.
 *
 * #33's descriptor API is not on main yet, while backend_metal.h still selects
 * its historical routed-expert entry points with fmt integers plus a dedicated
 * MXFP4 function. This adapter keeps that temporary selector behind the common
 * ColiRepresentationId so model code never has to infer a Metal format.
 */
typedef struct {
    ColiRepresentationId representation;
    int legacy_fmt;          /* 1/2/4 for current generic Metal entry point */
    int qgroup_size;         /* current fmt=4 group-size argument */
    int uses_mxfp4_entrypoint;
} ColiMetalRepresentationBinding;

typedef struct {
    ColiMetalRepresentationBinding execution;
    uint64_t generation;
    unsigned tier_mask;
    uint64_t resident_bytes;
    uint64_t allocation_bytes;
    const void *physical_handle;
} ColiMetalResidentBinding;

/*
 * Returns 1 for a representation the current Metal Apple8-v1 API can express,
 * 0 for a fail-closed fallback. No model-name or backend-private format
 * inference is permitted.
 */
static inline int coli_metal_representation_binding(
        const ColiRepresentationId *representation,
        ColiMetalRepresentationBinding *out) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));

    /*
     * The main-branch Metal routed-expert ABI is the frozen Apple8 v1 target.
     * Keep this gate next to the legacy selector so an Apple profile with a
     * different layout/kernel ABI cannot be silently reinterpreted. #26's
     * shared runtime-target registry can replace the literal ABI values when
     * that registry lands on main.
     */
    if (!coli_representation_backend_can_execute(
            representation, COLI_CSF_PROFILE_MACOS_ARM64_METAL_APPLE8_V1,
            1u, 1u))
        return 0;

    /*
     * The current backend consumes canonical expert planes. ROWS16 and future
     * #26 layout IDs must not be silently interpreted as canonical bytes.
     */
    if (representation->execution_layout != COLI_CSF_LAYOUT_CANONICAL)
        return 0;

    out->representation = *representation;
    if (representation->math_format == COLI_CSF_MATH_I8 &&
        representation->scale_format == COLI_CSF_SCALE_F32) {
        out->legacy_fmt = 1;
        return 1;
    }
    if (representation->math_format == COLI_CSF_MATH_INT4_PACKED &&
        representation->scale_format == COLI_CSF_SCALE_F32 &&
        representation->group_size == 0) {
        out->legacy_fmt = 2;
        return 1;
    }
    if (representation->math_format == COLI_CSF_MATH_INT4_GROUPED &&
        representation->scale_format == COLI_CSF_SCALE_F32 &&
        representation->group_size != 0 &&
        representation->group_size <= (uint32_t)INT_MAX) {
        out->legacy_fmt = 4;
        out->qgroup_size = (int)representation->group_size;
        return 1;
    }
    if (representation->math_format == COLI_CSF_MATH_MXFP4_E2M1 &&
        representation->scale_format == COLI_CSF_SCALE_UE8M0) {
        out->uses_mxfp4_entrypoint = 1;
        return 1;
    }

    memset(out, 0, sizeof(*out));
    return 0;
}

/*
 * Translate the backend-neutral resident view into the current Metal binding.
 * The generation and opaque physical identity are preserved verbatim; neither
 * is derived from logical expert identity or representation identity.
 */
static inline int coli_metal_resident_binding(
        const ColiExpertResidentView *view,
        ColiMetalResidentBinding *out) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!view || !view->generation || !view->tier_mask ||
        !view->resident_bytes || !view->allocation_bytes ||
        view->resident_bytes > view->allocation_bytes ||
        !view->physical_handle ||
        !coli_metal_representation_binding(&view->representation,
                                           &out->execution))
        return 0;
    out->generation = view->generation;
    out->tier_mask = view->tier_mask;
    out->resident_bytes = view->resident_bytes;
    out->allocation_bytes = view->allocation_bytes;
    out->physical_handle = view->physical_handle;
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_BACKEND_METAL_REPRESENTATION_H */

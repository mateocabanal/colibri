#ifndef COLI_TARGET_H
#define COLI_TARGET_H

#include "apple8_contract.h"
#include "expert_representation.h"
#include "generated/coli_target_registry.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    COLI_TARGET_OS_INVALID = 0,
    COLI_TARGET_OS_MACOS = 1,
    COLI_TARGET_OS_LINUX = 2
};
enum {
    COLI_TARGET_ARCH_INVALID = 0,
    COLI_TARGET_ARCH_ARM64 = 1,
    COLI_TARGET_ARCH_X86_64 = 2
};
enum {
    COLI_TARGET_BACKEND_INVALID = 0,
    COLI_TARGET_BACKEND_CPU = 1,
    COLI_TARGET_BACKEND_METAL = 2,
    COLI_TARGET_BACKEND_CUDA = 3
};
enum {
    COLI_TARGET_GPU_NONE = 0,
    COLI_TARGET_GPU_APPLE_FAMILY = 1,
    COLI_TARGET_GPU_CUDA_SM = 2
};
enum {
    COLI_TARGET_CPU_ARM64_ASIMD = UINT64_C(1) << 0,
    COLI_TARGET_CPU_X86_AVX2 = UINT64_C(1) << 1
};
enum {
    COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY = UINT64_C(1) << 0,
    COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE = UINT64_C(1) << 1,
    COLI_TARGET_RUNTIME_CUDA = UINT64_C(1) << 2
};

typedef struct ColiRuntimeTarget {
    const char *profile_name;
    uint16_t target_os;
    uint16_t target_arch;
    uint16_t backend;
    uint16_t gpu_kind;
    uint64_t cpu_feature_mask;
    uint32_t gpu_family;
    uint64_t runtime_features;
    uint32_t target_profile_abi;
    uint32_t execution_layout_abi;
    uint32_t kernel_abi;
    uint32_t target_class;
    uint32_t max_record_alignment;
    uint32_t max_io_granularity;
    uint32_t max_resident_alignment;
} ColiRuntimeTarget;

/* coli_format.c and coli_executor.c are embedded by several standalone engine
 * build graphs. Keep these small contract helpers header-resolved so freezing
 * target compatibility does not silently add a new link dependency to every
 * legacy engine. */
static inline void coli_target_error(char *error, size_t error_size,
                                     const char *fmt, ...) {
    va_list ap;
    if (!error || !error_size) return;
    va_start(ap, fmt);
    (void)vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static inline int coli_target_layout_registered(uint16_t layout) {
    return layout == COLI_CSF_LAYOUT_CANONICAL ||
           layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1;
}

static inline int coli_target_profile_accepts_layout(const char *profile,
                                                     uint16_t layout) {
    if (!profile || !coli_target_layout_registered(layout)) return 0;
    if (!strcmp(profile, COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1))
        return layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1;
    if (!strcmp(profile, COLI_TARGET_PROFILE_PORTABLE_V1) ||
        !strcmp(profile, COLI_TARGET_PROFILE_LINUX_X86_64_AVX2_V1))
        return layout == COLI_CSF_LAYOUT_CANONICAL;
    return 0;
}

/* Resolve a parsed expert matrix to the exact resident representation promised
 * by PROFILE. No canonicalization or target-layout substitution is performed. */
static inline int coli_target_resolve_matrix_representation(
        const char *profile, const ColiExpertMatrixInfo *matrix,
        ColiRepresentationId *out, char *error, size_t error_size) {
    if (!profile || !matrix || !out ||
        !coli_target_profile_accepts_layout(profile, matrix->layout)) {
        coli_target_error(error, error_size,
                          "matrix layout is not allowed by target profile");
        return -1;
    }
    if (!strcmp(profile, COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1)) {
        if (!coli_apple8_matrix_descriptor_valid(matrix, NULL)) {
            coli_target_error(error, error_size,
                              "matrix violates Apple8 production descriptor");
            return -1;
        }
        return coli_representation_from_csf_matrix(
            out, matrix,
            COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1,
            COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1,
            COLI_TARGET_CLASS_APPLE8_METAL_V1, 0);
    }
    return coli_representation_from_csf_matrix(out, matrix, 0, 0, 0, 0);
}

static inline int coli_target_require_u32(uint32_t got, uint32_t want,
                                          const char *label,
                                          char *error, size_t error_size) {
    if (got == want) return 0;
    coli_target_error(error, error_size,
                      "Apple8 runtime %s mismatch: need %u got %u",
                      label, want, got);
    return -1;
}

/* Compare the frozen profile requirements with one concrete runtime path.
 * PACKAGE_RECORD_ALIGNMENT is the alignment declared by the opened package. */
static inline int coli_target_check_compatibility(
        const char *required_profile, const ColiRuntimeTarget *runtime,
        uint32_t package_record_alignment, char *error, size_t error_size) {
    if (!required_profile || !runtime || !runtime->profile_name) {
        coli_target_error(error, error_size,
                          "target compatibility requires a concrete runtime descriptor");
        return -1;
    }
    if (strcmp(required_profile, runtime->profile_name)) {
        coli_target_error(error, error_size,
                          "runtime profile %s does not match required %s",
                          runtime->profile_name, required_profile);
        return -1;
    }
    if (strcmp(required_profile,
               COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1)) {
        coli_target_error(error, error_size,
                          "runtime compatibility is not frozen for profile %s",
                          required_profile);
        return -1;
    }
    if (coli_target_require_u32(runtime->target_profile_abi,
                                COLI_TARGET_PROFILE_ABI_APPLE8_V1,
                                "target-profile ABI", error, error_size) ||
        coli_target_require_u32(runtime->execution_layout_abi,
                                COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1,
                                "execution-layout ABI", error, error_size) ||
        coli_target_require_u32(runtime->kernel_abi,
                                COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1,
                                "kernel ABI", error, error_size) ||
        coli_target_require_u32(runtime->target_class,
                                COLI_TARGET_CLASS_APPLE8_METAL_V1,
                                "target class", error, error_size) ||
        coli_target_require_u32(runtime->target_os, COLI_APPLE8_TARGET_OS,
                                "OS", error, error_size) ||
        coli_target_require_u32(runtime->target_arch, COLI_APPLE8_TARGET_ARCH,
                                "architecture", error, error_size) ||
        coli_target_require_u32(runtime->backend, COLI_APPLE8_TARGET_BACKEND,
                                "backend", error, error_size) ||
        coli_target_require_u32(runtime->gpu_kind, COLI_APPLE8_GPU_KIND,
                                "GPU kind", error, error_size))
        return -1;
    if ((runtime->cpu_feature_mask & COLI_APPLE8_CPU_FEATURE_MASK) !=
        COLI_APPLE8_CPU_FEATURE_MASK) {
        coli_target_error(error, error_size,
                          "Apple8 runtime is missing required ARM64 CPU features");
        return -1;
    }
    if (runtime->gpu_family < COLI_APPLE8_GPU_FAMILY_MIN) {
        coli_target_error(error, error_size,
                          "Apple8 runtime GPU family %u is below required family %u",
                          runtime->gpu_family, COLI_APPLE8_GPU_FAMILY_MIN);
        return -1;
    }
    if ((runtime->runtime_features & COLI_APPLE8_REQUIRED_RUNTIME_FEATURES) !=
        COLI_APPLE8_REQUIRED_RUNTIME_FEATURES) {
        coli_target_error(error, error_size,
                          "Apple8 runtime is missing required unified-memory/Metal features");
        return -1;
    }
    if (runtime->max_record_alignment < COLI_APPLE8_RECORD_ALIGNMENT ||
        runtime->max_io_granularity < COLI_APPLE8_IO_GRANULARITY ||
        runtime->max_resident_alignment < COLI_APPLE8_RESIDENT_ALIGNMENT ||
        package_record_alignment != COLI_APPLE8_RECORD_ALIGNMENT) {
        coli_target_error(error, error_size,
                          "Apple8 runtime/package alignment contract mismatch");
        return -1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif

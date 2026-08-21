#ifndef COLI_TARGET_H
#define COLI_TARGET_H

#include "expert_representation.h"
#include "generated/coli_target_registry.h"

#include <stddef.h>
#include <stdint.h>

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

int coli_target_layout_registered(uint16_t layout);
int coli_target_profile_accepts_layout(const char *profile, uint16_t layout);

/* Resolve a parsed expert matrix to the exact resident representation promised
 * by PROFILE. No canonicalization or target-layout substitution is performed. */
int coli_target_resolve_matrix_representation(const char *profile,
                                              const ColiExpertMatrixInfo *matrix,
                                              ColiRepresentationId *out,
                                              char *error, size_t error_size);

/* Compare the frozen profile requirements with one concrete runtime path.
 * PACKAGE_RECORD_ALIGNMENT is the alignment declared by the opened package. */
int coli_target_check_compatibility(const char *required_profile,
                                    const ColiRuntimeTarget *runtime,
                                    uint32_t package_record_alignment,
                                    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif

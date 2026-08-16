#ifndef COLI_TARGET_H
#define COLI_TARGET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CSF_TARGET_VERSION_MINOR 1u
#define COLI_CSF_TARGET_DESC_BYTES 256u

/* Target descriptor identity registries from CSF v1.1. */
enum {
    COLI_TARGET_OS_INVALID = 0,
    COLI_TARGET_OS_MACOS = 1,
    COLI_TARGET_OS_LINUX = 2,
    COLI_TARGET_OS_WINDOWS = 3
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
    COLI_TARGET_BACKEND_CUDA = 3,
    COLI_TARGET_BACKEND_HYBRID = 4
};

enum {
    COLI_TARGET_GPU_NONE = 0,
    COLI_TARGET_GPU_APPLE_FAMILY = 1,
    COLI_TARGET_GPU_CUDA_SM = 2
};

enum {
    COLI_TARGET_CPU_ARM64_ASIMD = UINT64_C(1) << 0,
    COLI_TARGET_CPU_X86_AVX2 = UINT64_C(1) << 1,
    COLI_TARGET_CPU_X86_FMA = UINT64_C(1) << 2,
    COLI_TARGET_CPU_X86_AVX512F = UINT64_C(1) << 3,
    COLI_TARGET_CPU_X86_AVX512_BF16 = UINT64_C(1) << 4
};

enum {
    COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY = UINT64_C(1) << 0,
    COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE = UINT64_C(1) << 1,
    COLI_TARGET_RUNTIME_CUDA = UINT64_C(1) << 2,
    COLI_TARGET_RUNTIME_CUDA_ASYNC_COPY = UINT64_C(1) << 3,
    COLI_TARGET_RUNTIME_HOST_PINNED_STAGING = UINT64_C(1) << 4
};

enum {
    COLI_TARGET_F_TUNING_FINGERPRINT_VALID = 1u << 0,
    COLI_TARGET_F_ACCELERATOR_REQUIRED = 1u << 1,
    COLI_TARGET_F_RESIDENT_CPU_READABLE = 1u << 2,
    COLI_TARGET_F_RESIDENT_GPU_READABLE = 1u << 3,
    COLI_TARGET_F_STAGING_REQUIRED = 1u << 4
};

/* Parsed immutable requirements from one v1.1 manifest. Strings/profile_data
 * are heap-owned by this struct and released by coli_target_info_free(). */
typedef struct ColiTargetInfo {
    uint16_t target_os;
    uint16_t target_arch;
    uint16_t backend;
    uint16_t gpu_kind;
    uint32_t flags;
    uint64_t cpu_feature_mask;
    uint32_t gpu_family_min;
    uint32_t gpu_family_max;
    uint32_t gpu_capability_min;
    uint32_t gpu_capability_max;
    uint32_t target_profile_abi;
    uint32_t execution_layout_abi;
    uint32_t kernel_abi_min;
    uint32_t kernel_abi_max;
    uint32_t record_alignment;
    uint32_t io_granularity;
    uint32_t resident_alignment;
    uint64_t required_runtime_features;

    char *profile_name;
    char *quant_profile;
    char *storage_profile;
    char *optimization_profile;
    char *kernel_profile;
    char *target_triple;
    char *semantic_abi;
    char *compiler;

    uint8_t source_fingerprint[32];
    uint8_t artifact_fingerprint[32];
    uint8_t tuning_fingerprint[32];
    uint8_t *profile_data;
    size_t profile_data_bytes;
} ColiTargetInfo;

/* Runtime/back-end capability descriptor. Platform code discovers this; the
 * generic CSF parser only compares it with ColiTargetInfo. Strings describe the
 * one concrete semantic/profile stack the caller is attempting to execute. */
typedef struct ColiRuntimeTarget {
    uint16_t target_os;
    uint16_t target_arch;
    uint16_t backend;
    uint16_t gpu_kind;
    uint64_t cpu_feature_mask;
    uint32_t gpu_family;
    uint32_t gpu_capability;
    uint64_t runtime_features;

    const char *semantic_abi;
    const char *profile_name;
    const char *quant_profile;
    const char *storage_profile;
    uint32_t target_profile_abi;
    uint32_t execution_layout_abi;
    uint32_t kernel_abi;

    /* Maximum alignments/granularities that this runtime path can honor. */
    uint32_t max_record_alignment;
    uint32_t max_io_granularity;
    uint32_t max_resident_alignment;
} ColiRuntimeTarget;

/* Reads and strictly validates only the v1.1 manifest target identity layer:
 * manifest framing/CRC, strings, target descriptor/CRC, profile-data span/CRC,
 * and canonical artifact-fingerprint recomputation. No shard or model payload
 * is opened. This is suitable for compatibility checks before expensive I/O. */
int coli_target_read_package(const char *package_path, ColiTargetInfo *out,
                             char *error, size_t error_size);
void coli_target_info_free(ColiTargetInfo *info);

/* Returns 0 when RUNTIME can execute REQUIRED exactly as declared. On failure,
 * ERROR names the first required-vs-provided incompatibility. */
int coli_target_check_compatibility(const ColiTargetInfo *required,
                                    const ColiRuntimeTarget *runtime,
                                    char *error, size_t error_size);

/* Recomputes the CSF v1.1 artifact fingerprint from parsed target identity.
 * Exposed for compiler/tooling tests; normal callers use read_package(). */
int coli_target_artifact_fingerprint(const ColiTargetInfo *info,
                                     uint8_t out_digest[32],
                                     char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* COLI_TARGET_H */

#ifndef COLIBRI_QWEN_COLI_COMPAT_H
#define COLIBRI_QWEN_COLI_COMPAT_H

#include <string.h>

#include "coli_executor.h"
#include "coli_target.h"

/* qwen_moe historically selected the native COLI profile by platform but did
 * not provide the concrete runtime descriptor required by the frozen Apple8
 * contract. Keep that source surface stable: this Qwen-only compile wrapper
 * fills the descriptor on arm64 macOS and delegates to the real executor.
 *
 * The descriptor expresses host/profile capability. Qwen may subsequently
 * detile Apple8 execution bytes into its existing canonical MXFP4 cache; that
 * compatibility conversion is explicit in mxfp4_expert.c and never changes
 * the package's validated target representation. */
static inline int qwen_coli_executor_open(
    ColiExecutor **out, const char *package_path,
    const ColiExecutorOpenOptions *options,
    char *error, size_t error_size) {
    ColiExecutorOpenOptions resolved;
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    ColiRuntimeTarget apple8;
#endif

    if (!options) {
        return coli_executor_open(out, package_path, options, error, error_size);
    }
    resolved = *options;
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    if (!resolved.runtime_target && resolved.required_profile &&
        !strcmp(resolved.required_profile,
                COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1)) {
        memset(&apple8, 0, sizeof(apple8));
        apple8.profile_name = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
        apple8.target_os = COLI_TARGET_OS_MACOS;
        apple8.target_arch = COLI_TARGET_ARCH_ARM64;
        apple8.backend = COLI_TARGET_BACKEND_METAL;
        apple8.gpu_kind = COLI_TARGET_GPU_APPLE_FAMILY;
        apple8.cpu_feature_mask = COLI_TARGET_CPU_ARM64_ASIMD;
        apple8.gpu_family = COLI_APPLE8_GPU_FAMILY_MIN;
        apple8.runtime_features = COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY |
                                  COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE;
        apple8.target_profile_abi = COLI_TARGET_PROFILE_ABI_APPLE8_V1;
        apple8.execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1;
        apple8.kernel_abi = COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1;
        apple8.target_class = COLI_TARGET_CLASS_APPLE8_METAL_V1;
        apple8.max_record_alignment = COLI_APPLE8_RECORD_ALIGNMENT;
        apple8.max_io_granularity = COLI_APPLE8_IO_GRANULARITY;
        apple8.max_resident_alignment = COLI_APPLE8_RESIDENT_ALIGNMENT;
        resolved.runtime_target = &apple8;
    }
#endif
    return coli_executor_open(out, package_path, &resolved, error, error_size);
}

#define coli_executor_open qwen_coli_executor_open

#endif /* COLIBRI_QWEN_COLI_COMPAT_H */

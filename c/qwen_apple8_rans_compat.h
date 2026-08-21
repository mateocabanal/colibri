#ifndef COLIBRI_QWEN_APPLE8_RANS_COMPAT_H
#define COLIBRI_QWEN_APPLE8_RANS_COMPAT_H

/* Qwen-specific composition seam for the first Apple8+rANS runtime slice.
 * Included before qwen_moe.c by Makefile.qwen-rans-runtime so the large engine
 * stays untouched while the shared CSF/MXFP4 contracts are proven locally.
 *
 * - supplies the concrete native Apple8 runtime descriptor required by
 *   ColiExecutorOpenOptions;
 * - refuses canonical-only CUDA/Metal MXFP4 kernels for Apple8 sidecars, so
 *   qwen falls through to the layout-aware CPU path;
 * - does not enable MetalIO or introduce any storage-pipeline composition. */

#include <string.h>

#include "coli_executor.h"
#include "mxfp4_runtime.h"

#if defined(COLI_CUDA)
#include "backend_cuda.h"
#endif
#if defined(COLI_METAL)
#include "backend_metal.h"
#endif

static inline void qwen_apple8_runtime_target(ColiRuntimeTarget *r) {
    memset(r, 0, sizeof(*r));
    r->profile_name = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    r->target_os = COLI_APPLE8_TARGET_OS;
    r->target_arch = COLI_APPLE8_TARGET_ARCH;
    r->backend = COLI_APPLE8_TARGET_BACKEND;
    r->gpu_kind = COLI_APPLE8_GPU_KIND;
    r->cpu_feature_mask = COLI_APPLE8_CPU_FEATURE_MASK;
    r->gpu_family = COLI_APPLE8_GPU_FAMILY_MIN;
    r->runtime_features = COLI_APPLE8_REQUIRED_RUNTIME_FEATURES;
    r->target_profile_abi = COLI_TARGET_PROFILE_ABI_APPLE8_V1;
    r->execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1;
    r->kernel_abi = COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1;
    r->target_class = COLI_TARGET_CLASS_APPLE8_METAL_V1;
    r->max_record_alignment = COLI_APPLE8_RECORD_ALIGNMENT;
    r->max_io_granularity = COLI_APPLE8_IO_GRANULARITY;
    r->max_resident_alignment = COLI_APPLE8_RESIDENT_ALIGNMENT;
}

static inline int qwen_coli_executor_open(
        ColiExecutor **out, const char *package_path,
        const ColiExecutorOpenOptions *options,
        char *error, size_t error_size) {
    ColiExecutorOpenOptions local;
    ColiRuntimeTarget apple8;
    if (!options) {
        return coli_executor_open(out, package_path, options, error, error_size);
    }
    local = *options;
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    if (!local.runtime_target && local.required_profile &&
        !strcmp(local.required_profile,
                COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1)) {
        qwen_apple8_runtime_target(&apple8);
        local.runtime_target = &apple8;
    }
#endif
    return coli_executor_open(out, package_path, &local, error, error_size);
}

/* Define after qwen_coli_executor_open so its call above resolves the real
 * executor symbol rather than recursively expanding this composition macro. */
#define coli_executor_open qwen_coli_executor_open

#if defined(COLI_CUDA)
static inline int qwen_cuda_expert_mlp_mxfp4(
        float *y, const float *x,
        const unsigned char *gate, const unsigned char *gate_e8,
        const unsigned char *up, const unsigned char *up_e8,
        const unsigned char *down, const unsigned char *down_e8,
        int S, int D, int I) {
    if (coli_mxfp4_runtime_is_apple8(gate_e8) ||
        coli_mxfp4_runtime_is_apple8(up_e8) ||
        coli_mxfp4_runtime_is_apple8(down_e8))
        return 0;
    return coli_cuda_expert_mlp_mxfp4(y, x,
                                      gate, gate_e8, up, up_e8,
                                      down, down_e8, S, D, I);
}
#define coli_cuda_expert_mlp_mxfp4 qwen_cuda_expert_mlp_mxfp4
#endif

#if defined(COLI_METAL)
static inline int qwen_metal_moe_block_mxfp4(
        int nb, int D, int Iinter,
        const void *const *g, const void *const *u, const void *const *d,
        const uint8_t *const *gs, const uint8_t *const *us,
        const uint8_t *const *ds,
        const float *xg, const int *xoff, const int *nr,
        const int *rows, const float *rw,
        float *out, int S) {
    for (int i = 0; i < nb; ++i) {
        if (coli_mxfp4_runtime_is_apple8(gs[i]) ||
            coli_mxfp4_runtime_is_apple8(us[i]) ||
            coli_mxfp4_runtime_is_apple8(ds[i]))
            return 0;
    }
    return coli_metal_moe_block_mxfp4(nb, D, Iinter,
                                      g, u, d, gs, us, ds,
                                      xg, xoff, nr, rows, rw, out, S);
}
#define coli_metal_moe_block_mxfp4 qwen_metal_moe_block_mxfp4
#endif

#endif

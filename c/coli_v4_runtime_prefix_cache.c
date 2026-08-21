#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Engine-lifetime hooks for the process-local and persistent #80 prefix tiers.
 * Remap the existing public open/destroy functions, then layer only cache
 * registration/retirement around current-main runtime behavior.
 *
 * Package-only V4 also exposes named COLITENS through the legacy tensor API
 * when a speculative source actually needs that compatibility surface. Target-
 * only package execution deliberately keeps target_index == NULL and leaves the
 * adapter unbound so disabling speculation is behaviorally identical to the
 * pre-MTP runtime.
 */
#define coli_v4_engine_open coli_v4_engine_open_uncached
#define coli_v4_engine_destroy coli_v4_engine_destroy_uncached
#include "deepseek_v4_internal.h"
#undef coli_v4_engine_destroy
#undef coli_v4_engine_open

#include "coli_v4_prefix_cache.h"
#include "coli_v4_prefix_disk.h"
#include "coli_v4_package_tensor_source.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* coli_executor_open() is intercepted inside the renamed core engine-open call,
 * before v4_dspark_full_wanted(options) runs. Carry only the caller's explicit
 * speculation request across that boundary. Thread-local storage avoids making
 * concurrent engine opens share a transient capability bit. */
static _Thread_local int g_coli_v4_package_bridge_wanted;

static int coli_v4_package_mtp_probe(const ColiExecutor *executor) {
    static const char *required[] = {
        "mtp.0.main_proj.weight",
        "mtp.0.main_proj.scale",
        "mtp.1.attn.wq_a.weight",
        "mtp.1.attn.wq_a.scale",
        "mtp.2.attn.wq_a.weight",
        "mtp.2.attn.wq_a.scale",
        "mtp.2.confidence_head.proj.weight",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
    };
    if (!executor) return 0;
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
        const ColiRecordInfo *record =
            coli_executor_record_by_name(executor, required[i]);
        if (!record || record->kind != COLI_CSF_REC_TENSOR)
            return 0;
    }
    return 1;
}

static void coli_v4_package_identity_log(const ColiExecutor *executor) {
    const ColiPackage *package = executor ? coli_executor_package(executor) : NULL;
    if (!package) return;
    const char *profile = coli_package_profile(package);
    const char *compiler = coli_package_compiler(package);
    const uint8_t *fingerprint = coli_package_source_fingerprint(package);
    fprintf(stderr, "v4_coli package_profile=%s compiler=%s source_fingerprint=",
            profile ? profile : "unknown", compiler ? compiler : "unknown");
    if (fingerprint) {
        for (size_t i = 0; i < 32; i++) fprintf(stderr, "%02x", fingerprint[i]);
    } else {
        fputs("unknown", stderr);
    }
    fputc('\n', stderr);
}

static void coli_v4_fill_apple8_runtime(ColiRuntimeTarget *runtime) {
    if (!runtime) return;
    memset(runtime, 0, sizeof(*runtime));
    runtime->profile_name = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    runtime->target_os = COLI_TARGET_OS_MACOS;
    runtime->target_arch = COLI_TARGET_ARCH_ARM64;
    runtime->backend = COLI_TARGET_BACKEND_METAL;
    runtime->gpu_kind = COLI_TARGET_GPU_APPLE_FAMILY;
    runtime->cpu_feature_mask = COLI_TARGET_CPU_ARM64_ASIMD;
    runtime->gpu_family = COLI_APPLE8_GPU_FAMILY_MIN;
    runtime->runtime_features = COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY |
                                COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE;
    runtime->target_profile_abi = COLI_TARGET_PROFILE_ABI_APPLE8_V1;
    runtime->execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1;
    runtime->kernel_abi = COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1;
    runtime->target_class = COLI_TARGET_CLASS_APPLE8_METAL_V1;
    runtime->max_record_alignment = COLI_APPLE8_RECORD_ALIGNMENT;
    runtime->max_io_granularity = COLI_APPLE8_IO_GRANULARITY;
    runtime->max_resident_alignment = COLI_APPLE8_RESIDENT_ALIGNMENT;
}

/* deepseek_v4.c predates the executor's frozen target contract and used to
 * hard-code Apple8 with a NULL runtime descriptor. Resolve the package's exact
 * declared profile first: canonical exact packages remain canonical, while an
 * Apple8 package receives the same concrete descriptor used by the contract
 * tests/Qwen bridge. package_open_ex only validates manifest/index metadata; it
 * does not scan expert payloads, so this preflight does not duplicate model I/O. */
static int coli_v4_resolve_executor_options(
        const char *package_path, const ColiExecutorOpenOptions *options,
        ColiExecutorOpenOptions *resolved, ColiRuntimeTarget *apple8,
        char *error, size_t error_size) {
    if (!package_path || !options || !resolved || !apple8) return -1;
    ColiPackage *probe = NULL;
    if (coli_package_open_ex(&probe, package_path, options->checksum_policy,
                             error, error_size))
        return -1;
    const char *actual = coli_package_profile(probe);
    const char *required = NULL;
    if (actual && !strcmp(actual, COLI_CSF_PROFILE_PORTABLE_V1))
        required = COLI_CSF_PROFILE_PORTABLE_V1;
    else if (actual && !strcmp(actual,
                              COLI_CSF_PROFILE_MACOS_ARM64_METAL_APPLE8_V1))
        required = COLI_CSF_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    else if (actual && !strcmp(actual, COLI_CSF_PROFILE_LINUX_X86_64_AVX2_V1))
        required = COLI_CSF_PROFILE_LINUX_X86_64_AVX2_V1;
    if (!required) {
        if (error && error_size)
            snprintf(error, error_size, "unsupported V4 package target profile: %s",
                     actual ? actual : "unknown");
        coli_package_close(probe);
        return -1;
    }
    coli_package_close(probe);

    *resolved = *options;
    resolved->required_profile = required;
    resolved->runtime_target = NULL;
    if (!strcmp(required, COLI_CSF_PROFILE_MACOS_ARM64_METAL_APPLE8_V1)) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
        coli_v4_fill_apple8_runtime(apple8);
        resolved->runtime_target = apple8;
#else
        if (error && error_size)
            snprintf(error, error_size,
                     "Apple8 V4 package requires arm64 macOS Metal runtime");
        return -1;
#endif
    }
    return 0;
}

/* Intercept the executor acquire at the exact point engine->coli_static becomes
 * available. A package-only engine gets the non-owning sentinel only when full
 * MTP/DSpark or the explicit Markov proposer was requested. The sentinel exists
 * solely to let legacy speculative probing reach the named-tensor bridge; it is
 * not a package-mode target capability and must never leak into target-only
 * execution. Hybrid opens already own a real safetensors index and need no
 * package compatibility adapter. */
static int coli_v4_package_executor_open_bridge(
        ColiExecutor **out, const char *package_path,
        const ColiExecutorOpenOptions *options,
        char *error, size_t error_size) {
    ColiExecutorOpenOptions resolved;
    ColiRuntimeTarget apple8;
    if (coli_v4_resolve_executor_options(package_path, options, &resolved,
                                         &apple8, error, error_size))
        return -1;
    int result = coli_executor_open(out, package_path, &resolved,
                                    error, error_size);
    if (result || !out || !*out) return result;
    coli_v4_package_identity_log(*out);

    ColiV4Engine *engine = (ColiV4Engine *)((char *)out -
        offsetof(ColiV4Engine, coli_static));
    if (!g_coli_v4_package_bridge_wanted || engine->target_index)
        return 0;

    if (coli_v4_package_source_bind(*out)) {
        coli_executor_close(*out);
        *out = NULL;
        if (error && error_size)
            snprintf(error, error_size,
                     "cannot initialize package named-tensor source");
        return -1;
    }

    engine->target_index = coli_v4_package_source_sentinel();
    const char *mtp = getenv("V4_MTP");
    const char *draft = getenv("V4_DRAFT");
    if (mtp && atoi(mtp) != 0 && draft && atoi(draft) > 0) {
        int profile = coli_v4_package_mtp_probe(*out);
        fprintf(stderr,
                "v4_coli dspark_source=package-named-tensors mtp_probe=%s "
                "validation=%s\n",
                profile ? "present" : "missing",
                profile ? "deferred-to-dspark-loader" : "target-only");
    }
    return 0;
}

#define coli_v4_engine_open coli_v4_engine_open_uncached
#define coli_v4_engine_destroy coli_v4_engine_destroy_uncached
#define coli_executor_open coli_v4_package_executor_open_bridge
#define coli_st_find coli_v4_package_source_find
#define coli_st_read_tensor coli_v4_package_source_read_tensor
#define coli_st_tensor_shard coli_v4_package_source_tensor_shard
#define coli_st_read_at coli_v4_package_source_read_at
#define coli_st_read_at_streaming coli_v4_package_source_read_at_streaming
#define st_read_scale_f32 coli_v4_package_read_scale_f32
#define coli_tensor_load_f32 coli_v4_package_tensor_load_f32
#include "deepseek_v4.c"
#undef coli_tensor_load_f32
#undef st_read_scale_f32
#undef coli_st_read_at_streaming
#undef coli_st_read_at
#undef coli_st_tensor_shard
#undef coli_st_read_tensor
#undef coli_st_find
#undef coli_executor_open
#undef coli_v4_engine_destroy
#undef coli_v4_engine_open

static int coli_v4_package_speculation_requested(
        const ColiV4EngineOpenOptions *options) {
    if (!options || options->no_dspark) return 0;
    const char *mtp = getenv("V4_MTP");
    const char *draft = getenv("V4_DRAFT");
    if (mtp && atoi(mtp) != 0 && draft && atoi(draft) > 0)
        return 1;
    const char *markov = getenv("COLI_V4_MARKOV_SPEC");
    return markov && atoi(markov) != 0;
}

int coli_v4_engine_open(ColiV4Engine **engine,
                        const ColiV4EngineOpenOptions *options,
                        char *error, size_t error_size) {
    int previous_bridge_wanted = g_coli_v4_package_bridge_wanted;
    g_coli_v4_package_bridge_wanted =
        coli_v4_package_speculation_requested(options);
    int result = coli_v4_engine_open_uncached(engine, options, error, error_size);
    g_coli_v4_package_bridge_wanted = previous_bridge_wanted;
    if (!result && engine && *engine)
        coli_v4_prefix_disk_register_engine(*engine);
    return result;
}

void coli_v4_engine_destroy(ColiV4Engine *engine) {
    coli_v4_prefix_cache_forget_engine(engine);
    coli_v4_prefix_disk_forget_engine(engine);
    if (engine &&
        g_coli_v4_package_tensor_source.executor == engine->coli_static)
        coli_v4_package_source_reset();
    coli_v4_engine_destroy_uncached(engine);
}

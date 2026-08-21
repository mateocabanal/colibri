#include "../coli_executor.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

static ColiRuntimeTarget good_runtime(void) {
    ColiRuntimeTarget r;
    memset(&r, 0, sizeof(r));
    r.profile_name = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    r.target_os = COLI_TARGET_OS_MACOS;
    r.target_arch = COLI_TARGET_ARCH_ARM64;
    r.backend = COLI_TARGET_BACKEND_METAL;
    r.gpu_kind = COLI_TARGET_GPU_APPLE_FAMILY;
    r.cpu_feature_mask = COLI_TARGET_CPU_ARM64_ASIMD;
    r.gpu_family = COLI_APPLE8_GPU_FAMILY_MIN;
    r.runtime_features = COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY |
                         COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE;
    r.target_profile_abi = COLI_TARGET_PROFILE_ABI_APPLE8_V1;
    r.execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1;
    r.kernel_abi = COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1;
    r.target_class = COLI_TARGET_CLASS_APPLE8_METAL_V1;
    r.max_record_alignment = COLI_APPLE8_RECORD_ALIGNMENT;
    r.max_io_granularity = COLI_APPLE8_IO_GRANULARITY;
    r.max_resident_alignment = COLI_APPLE8_RESIDENT_ALIGNMENT;
    return r;
}

static ColiExecutorOpenOptions good_options(ColiRuntimeTarget *runtime) {
    ColiExecutorOpenOptions o;
    memset(&o, 0, sizeof(o));
    o.required_profile = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    o.checksum_policy = COLI_CSF_CHECKSUM_MANIFEST_ONLY;
    o.runtime_target = runtime;
    return o;
}

static int target_contract_rejects(ColiRuntimeTarget runtime) {
    ColiExecutor *executor = NULL;
    ColiExecutorOpenOptions options = good_options(&runtime);
    char error[256] = {0};
    CHECK(coli_executor_open(&executor, "/definitely/not/a/coli/package", &options,
                             error, sizeof(error)) != 0);
    CHECK(executor == NULL);
    return strstr(error, "Apple8 runtime") != NULL ||
           strstr(error, "runtime profile") != NULL ||
           strstr(error, "compatibility") != NULL;
}

int main(void) {
    ColiRuntimeTarget r = good_runtime();
    ColiExecutor *executor = NULL;
    ColiExecutorOpenOptions options = good_options(&r);
    char error[256] = {0};

    /* A valid runtime reaches package I/O; it must not fail the target gate. */
    CHECK(coli_executor_open(&executor, "/definitely/not/a/coli/package", &options,
                             error, sizeof(error)) != 0);
    CHECK(executor == NULL);
    CHECK(strstr(error, "Apple8 runtime") == NULL);

    r = good_runtime(); r.target_profile_abi++; CHECK(target_contract_rejects(r));
    r = good_runtime(); r.execution_layout_abi++; CHECK(target_contract_rejects(r));
    r = good_runtime(); r.kernel_abi++; CHECK(target_contract_rejects(r));
    r = good_runtime(); r.target_class++; CHECK(target_contract_rejects(r));
    r = good_runtime(); r.gpu_family--; CHECK(target_contract_rejects(r));
    r = good_runtime(); r.runtime_features &= ~COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE; CHECK(target_contract_rejects(r));
    r = good_runtime(); r.profile_name = COLI_TARGET_PROFILE_PORTABLE_V1; CHECK(target_contract_rejects(r));

    options = good_options(NULL);
    memset(error, 0, sizeof(error));
    CHECK(coli_executor_open(&executor, "/definitely/not/a/coli/package", &options,
                             error, sizeof(error)) != 0);
    CHECK(executor == NULL);
    puts("test_apple8_executor_contract: ok");
    return 0;
}

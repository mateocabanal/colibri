#include "../coli_executor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ColiRuntimeTarget runtime_target(void) {
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

static int write_file(const char *path, const void *bytes, size_t n) {
    FILE *f = fopen(path, "wb");
    int rc = 0;
    if (!f) return -1;
    if (n && fwrite(bytes, 1, n, f) != n) rc = -1;
    if (fclose(f)) rc = -1;
    return rc;
}

int main(int argc, char **argv) {
    ColiRuntimeTarget runtime;
    ColiExecutorOpenOptions options;
    ColiExecutor *executor = NULL;
    const ColiRecordInfo *record;
    const ColiPackage *package;
    uint64_t resident_bytes = 0;
    unsigned char *resident = NULL;
    size_t written = 0;
    char error[512] = {0};
    long layer, expert;
    char *end = NULL;
    int rc = 1;

    if (argc != 5) {
        fprintf(stderr, "usage: %s PACKAGE LAYER EXPERT OUTPUT\n", argv[0]);
        return 2;
    }
    layer = strtol(argv[2], &end, 10);
    if (!end || *end || layer < INT32_MIN || layer > INT32_MAX) return 2;
    expert = strtol(argv[3], &end, 10);
    if (!end || *end || expert < INT32_MIN || expert > INT32_MAX) return 2;

    runtime = runtime_target();
    memset(&options, 0, sizeof(options));
    options.required_profile = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    options.runtime_target = &runtime;
    options.checksum_policy = COLI_CSF_CHECKSUM_MANIFEST_ONLY;
    if (coli_executor_open(&executor, argv[1], &options, error, sizeof(error))) {
        fprintf(stderr, "open: %s\n", error);
        goto done;
    }
    record = coli_executor_expert(executor, (int32_t)layer, (int32_t)expert);
    package = coli_executor_package(executor);
    if (!record || !package ||
        coli_package_expert_resident_bytes(package, record, &resident_bytes,
                                           error, sizeof(error))) {
        fprintf(stderr, "resident-size: %s\n", error);
        goto done;
    }
    if (resident_bytes == 0 || resident_bytes > SIZE_MAX) {
        fprintf(stderr, "invalid resident size\n");
        goto done;
    }
    resident = (unsigned char *)malloc((size_t)resident_bytes);
    if (!resident) goto done;
    if (coli_executor_load_expert(executor, (int32_t)layer, (int32_t)expert,
                                  resident, (size_t)resident_bytes,
                                  error, sizeof(error))) {
        fprintf(stderr, "load: %s\n", error);
        goto done;
    }
    /* The public decode API and executor load must agree on the exact resident
     * record size; call it explicitly once to pin the byte-count contract. */
    {
        unsigned char *second = (unsigned char *)malloc((size_t)resident_bytes);
        if (!second) goto done;
        if (coli_package_decode_expert_record(package, record, second,
                                              (size_t)resident_bytes, &written,
                                              error, sizeof(error)) ||
            written != (size_t)resident_bytes ||
            memcmp(resident, second, written)) {
            free(second);
            fprintf(stderr, "decode-api: %s\n", error);
            goto done;
        }
        free(second);
    }
    if (write_file(argv[4], resident, (size_t)resident_bytes)) goto done;
    printf("APPLE8_DECODE layer=%ld expert=%ld stored=%llu resident=%llu\n",
           layer, expert,
           (unsigned long long)record->stored_bytes,
           (unsigned long long)resident_bytes);
    rc = 0;

done:
    free(resident);
    coli_executor_close(executor);
    return rc;
}

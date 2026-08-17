#include "../coli_exec_format.h"
#include "../coli_target_profiles.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mxfp4_ref(float *y, const float *x, const unsigned char *q4,
               const unsigned char *e8s, int S, int I, int O);

static int fail(const char *what, const char *detail) {
    fprintf(stderr, "colic-e2e: %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    return 1;
}

static ColiRuntimeTarget apple_runtime(void) {
    ColiRuntimeTarget runtime;
    memset(&runtime, 0, sizeof(runtime));
    runtime.target_os = COLI_TARGET_OS_MACOS;
    runtime.target_arch = COLI_TARGET_ARCH_ARM64;
    runtime.backend = COLI_TARGET_BACKEND_METAL;
    runtime.gpu_kind = COLI_TARGET_GPU_APPLE_FAMILY;
    runtime.cpu_feature_mask = COLI_TARGET_CPU_ARM64_ASIMD;
    runtime.gpu_family = 8;
    runtime.runtime_features = COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY |
                               COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE;
    runtime.semantic_abi = "deepseek-v4-exec-v1";
    runtime.profile_name = COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    runtime.quant_profile = "exact";
    runtime.storage_profile = "none";
    runtime.target_profile_abi = COLI_TARGET_PROFILE_ABI_V1;
    runtime.execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_V1;
    runtime.kernel_abi = COLI_KERNEL_ABI_V1;
    runtime.max_record_alignment = 16 * 1024;
    runtime.max_io_granularity = 16 * 1024;
    runtime.max_resident_alignment = 16 * 1024;
    return runtime;
}

static int close_enough(float a, float b) {
    float tolerance = 1e-6f * (1.0f + fabsf(b));
    return fabsf(a - b) <= tolerance;
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static int execute_expert(const char *package_path) {
    char error[512] = {0};
    ColiRuntimeTarget runtime = apple_runtime();
    ColiRuntimeTarget incompatible = runtime;
    ColiExecPackage *package = NULL;
    const ColiExecRecordInfo *record;
    const ColiExecRecordInfo *second;
    ColiExecExpertInfo info;
    unsigned char *bytes = NULL;
    float x[2] = {1.0f, 2.0f};
    float gate[3], up[3], hidden[3], output[2];
    float expected[2] = {silu(1.0f), 2.0f * silu(2.0f)};
    static const unsigned char gate_expected[3] = {0x02, 0x20, 0x22};
    static const unsigned char down_expected[4] = {0x02, 0x00, 0x20, 0x00};
    size_t i;

    incompatible.gpu_family = 7;
    if (coli_exec_package_open(&package, package_path, &incompatible, error, sizeof(error)) == 0) {
        coli_exec_package_close(package);
        return fail("accepted incompatible Apple GPU family", NULL);
    }
    package = NULL;
    error[0] = '\0';

    if (coli_exec_package_open_ex(&package, package_path, &runtime,
                                  COLI_EXEC_CHECKSUM_RECORD_ON_READ,
                                  error, sizeof(error)))
        return fail("strict package open failed", error);
    if (coli_exec_package_verify_all(package, error, sizeof(error))) {
        coli_exec_package_close(package);
        return fail("full package verification failed", error);
    }

    const ColiTargetInfo *target = coli_exec_package_target(package);
    if (!target || strcmp(target->profile_name, COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1) ||
        strcmp(target->semantic_abi, "deepseek-v4-exec-v1") ||
        strcmp(target->quant_profile, "exact") || strcmp(target->storage_profile, "none")) {
        coli_exec_package_close(package);
        return fail("target identity mismatch", NULL);
    }

    record = coli_exec_package_expert(package, 0, 0);
    second = coli_exec_package_expert(package, 0, 1);
    if (!record || !second || record->kind != COLI_CSF_REC_EXPERT ||
        record->math_format != COLI_CSF_MATH_MIXED ||
        record->scale_format != COLI_CSF_SCALE_MIXED ||
        record->layout != COLI_EXEC_LAYOUT_MIXED) {
        coli_exec_package_close(package);
        return fail("expert index/outer descriptor mismatch", NULL);
    }
    if (coli_exec_package_expert_info(package, record, &info, error, sizeof(error))) {
        coli_exec_package_close(package);
        return fail("expert envelope parse failed", error);
    }
    if (info.layer != 0 || info.expert != 0) {
        coli_exec_package_close(package);
        return fail("expert identity mismatch", NULL);
    }

    for (i = 0; i < 3; ++i) {
        const ColiExecMatrixInfo *m = &info.matrices[i];
        if (m->role != i + 1 || m->math_format != COLI_CSF_MATH_MXFP4_E2M1 ||
            m->scale_format != COLI_CSF_SCALE_UE8M0 ||
            m->layout != COLI_LAYOUT_APPLE_MXFP4_ROW32_V1 ||
            m->scale_block_rows != 1 || m->scale_block_columns != 32 ||
            !coli_target_profile_accepts_layout(target->profile_name, m->layout) ||
            !coli_target_layout_accepts_format(m->layout, m->math_format, m->scale_format,
                                               m->scale_block_rows, m->scale_block_columns,
                                               m->group_size)) {
            coli_exec_package_close(package);
            return fail("target matrix ABI mismatch", NULL);
        }
    }

    if (record->stored_bytes > SIZE_MAX) {
        coli_exec_package_close(package);
        return fail("expert record is too large", NULL);
    }
    bytes = (unsigned char *)malloc((size_t)record->stored_bytes);
    if (!bytes) {
        coli_exec_package_close(package);
        return fail("out of memory", NULL);
    }
    if (coli_exec_package_read_record(package, record, bytes, (size_t)record->stored_bytes,
                                      error, sizeof(error))) {
        free(bytes);
        coli_exec_package_close(package);
        return fail("expert read failed", error);
    }

    if (info.matrices[0].weight_stored_bytes != sizeof(gate_expected) ||
        memcmp(bytes + info.matrices[0].weight_offset, gate_expected, sizeof(gate_expected)) ||
        info.matrices[2].weight_stored_bytes != sizeof(down_expected) ||
        memcmp(bytes + info.matrices[2].weight_offset, down_expected, sizeof(down_expected))) {
        free(bytes);
        coli_exec_package_close(package);
        return fail("compiled expert differs from exact source semantics", NULL);
    }

    mxfp4_ref(gate, x,
              bytes + info.matrices[0].weight_offset,
              bytes + info.matrices[0].scale_offset,
              1, (int)info.matrices[0].columns, (int)info.matrices[0].rows);
    mxfp4_ref(up, x,
              bytes + info.matrices[1].weight_offset,
              bytes + info.matrices[1].scale_offset,
              1, (int)info.matrices[1].columns, (int)info.matrices[1].rows);
    for (i = 0; i < 3; ++i) hidden[i] = silu(gate[i]) * up[i];
    mxfp4_ref(output, hidden,
              bytes + info.matrices[2].weight_offset,
              bytes + info.matrices[2].scale_offset,
              1, (int)info.matrices[2].columns, (int)info.matrices[2].rows);

    free(bytes);
    coli_exec_package_close(package);
    if (!close_enough(output[0], expected[0]) || !close_enough(output[1], expected[1])) {
        fprintf(stderr, "colic-e2e: kernel output [%.9g, %.9g], expected [%.9g, %.9g]\n",
                output[0], output[1], expected[0], expected[1]);
        return 1;
    }
    printf("colic-e2e: strict loader + MXFP4 kernel parity ok\n");
    return 0;
}

static int corrupt_and_expect_failure(const char *package_path) {
    char error[512] = {0};
    char shard_path[1024];
    ColiRuntimeTarget runtime = apple_runtime();
    ColiExecPackage *package = NULL;
    const ColiExecRecordInfo *record;
    FILE *file;
    long offset;
    int byte;

    if (coli_exec_package_open_ex(&package, package_path, &runtime,
                                  COLI_EXEC_CHECKSUM_MANIFEST_ONLY,
                                  error, sizeof(error)))
        return fail("cannot open package before corruption", error);
    record = coli_exec_package_expert(package, 0, 0);
    if (!record || record->shard_id > 99999 || record->payload_offset > (uint64_t)LONG_MAX - 448) {
        coli_exec_package_close(package);
        return fail("cannot locate expert for corruption test", NULL);
    }
    snprintf(shard_path, sizeof(shard_path), "%s/data-%05u.coli", package_path, record->shard_id);
    offset = (long)(record->payload_offset + 448);
    coli_exec_package_close(package);

    file = fopen(shard_path, "r+b");
    if (!file || fseek(file, offset, SEEK_SET)) {
        if (file) fclose(file);
        return fail("cannot open/seek shard for corruption test", shard_path);
    }
    byte = fgetc(file);
    if (byte == EOF || fseek(file, offset, SEEK_SET) || fputc(byte ^ 0x01, file) == EOF) {
        fclose(file);
        return fail("cannot mutate shard for corruption test", shard_path);
    }
    fclose(file);

    package = NULL;
    error[0] = '\0';
    if (coli_exec_package_open_ex(&package, package_path, &runtime,
                                  COLI_EXEC_CHECKSUM_MANIFEST_ONLY,
                                  error, sizeof(error)))
        return fail("manifest-only reopen unexpectedly failed", error);
    if (coli_exec_package_verify_all(package, error, sizeof(error)) == 0) {
        coli_exec_package_close(package);
        return fail("corrupt record passed full verification", NULL);
    }
    coli_exec_package_close(package);
    printf("colic-e2e: corrupt record rejected as expected\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2) return execute_expert(argv[1]);
    if (argc == 3 && !strcmp(argv[1], "--corrupt-and-expect-fail"))
        return corrupt_and_expect_failure(argv[2]);
    fprintf(stderr, "usage: %s [--corrupt-and-expect-fail] PACKAGE\n", argv[0]);
    return 2;
}

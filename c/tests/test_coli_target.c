#include "../coli_target.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

static int hexbyte(char a, char b) {
    int x = (a >= '0' && a <= '9') ? a - '0' : a - 'a' + 10;
    int y = (b >= '0' && b <= '9') ? b - '0' : b - 'a' + 10;
    return (x << 4) | y;
}

static int digest_eq_hex(const unsigned char d[32], const char *hex) {
    int i;
    if (!hex || strlen(hex) != 64) return 0;
    for (i = 0; i < 32; ++i)
        if (d[i] != (unsigned char)hexbyte(hex[i * 2], hex[i * 2 + 1])) return 0;
    return 1;
}

static ColiRuntimeTarget apple_runtime(void) {
    ColiRuntimeTarget r;
    memset(&r, 0, sizeof(r));
    r.target_os = COLI_TARGET_OS_MACOS;
    r.target_arch = COLI_TARGET_ARCH_ARM64;
    r.backend = COLI_TARGET_BACKEND_METAL;
    r.gpu_kind = COLI_TARGET_GPU_APPLE_FAMILY;
    r.cpu_feature_mask = COLI_TARGET_CPU_ARM64_ASIMD;
    r.gpu_family = 8;
    r.runtime_features = COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY |
                         COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE;
    r.semantic_abi = "fixture-blob-v1";
    r.profile_name = "macos-arm64-metal-apple8-v1";
    r.quant_profile = "exact";
    r.storage_profile = "none";
    r.target_profile_abi = 1;
    r.execution_layout_abi = 1;
    r.kernel_abi = 1;
    r.max_record_alignment = 16384;
    r.max_io_granularity = 16384;
    r.max_resident_alignment = 16384;
    return r;
}

static ColiRuntimeTarget cuda_runtime(void) {
    ColiRuntimeTarget r;
    memset(&r, 0, sizeof(r));
    r.target_os = COLI_TARGET_OS_LINUX;
    r.target_arch = COLI_TARGET_ARCH_X86_64;
    r.backend = COLI_TARGET_BACKEND_CUDA;
    r.gpu_kind = COLI_TARGET_GPU_CUDA_SM;
    r.cpu_feature_mask = COLI_TARGET_CPU_X86_AVX2 | COLI_TARGET_CPU_X86_FMA;
    r.gpu_capability = 89;
    r.runtime_features = COLI_TARGET_RUNTIME_CUDA |
                         COLI_TARGET_RUNTIME_CUDA_ASYNC_COPY |
                         COLI_TARGET_RUNTIME_HOST_PINNED_STAGING;
    r.semantic_abi = "fixture-blob-v1";
    r.profile_name = "linux-x86_64-cuda-sm89-v1";
    r.quant_profile = "exact";
    r.storage_profile = "none";
    r.target_profile_abi = 1;
    r.execution_layout_abi = 1;
    r.kernel_abi = 1;
    r.max_record_alignment = 4096;
    r.max_io_granularity = 4096;
    r.max_resident_alignment = 256;
    return r;
}

static int test_apple(const char *path) {
    ColiTargetInfo info;
    ColiRuntimeTarget runtime;
    unsigned char digest[32];
    char error[512];

    CHECK(coli_target_read_package(path, &info, error, sizeof(error)) == 0);
    CHECK(info.target_os == COLI_TARGET_OS_MACOS);
    CHECK(info.target_arch == COLI_TARGET_ARCH_ARM64);
    CHECK(info.backend == COLI_TARGET_BACKEND_METAL);
    CHECK(info.gpu_kind == COLI_TARGET_GPU_APPLE_FAMILY);
    CHECK(info.gpu_family_min == 8 && info.gpu_family_max == 0);
    CHECK(info.cpu_feature_mask == COLI_TARGET_CPU_ARM64_ASIMD);
    CHECK(info.record_alignment == 16384);
    CHECK(info.io_granularity == 16384);
    CHECK(info.resident_alignment == 16384);
    CHECK(!strcmp(info.semantic_abi, "fixture-blob-v1"));
    CHECK(!strcmp(info.target_triple, "arm64-apple-macos"));
    CHECK(digest_eq_hex(info.source_fingerprint,
          "34da718f420aec269094374dc41e9df5d2593fccb8da435c1788e07fda3a0853"));
    CHECK(digest_eq_hex(info.artifact_fingerprint,
          "afb713f6fa817b96755a49189a77893ccd3255b599b9d41abe9c5593aa3fe771"));
    CHECK(coli_target_artifact_fingerprint(&info, digest, error, sizeof(error)) == 0);
    CHECK(!memcmp(digest, info.artifact_fingerprint, 32));

    runtime = apple_runtime();
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) == 0);

    runtime.gpu_family = 7;
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "Apple GPU family") != NULL);
    runtime = apple_runtime(); runtime.target_os = COLI_TARGET_OS_LINUX;
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "target OS mismatch") != NULL);
    runtime = apple_runtime(); runtime.semantic_abi = "deepseek-v4-exec-v1";
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "semantic ABI mismatch") != NULL);
    runtime = apple_runtime(); runtime.cpu_feature_mask = 0;
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "CPU features") != NULL);
    runtime = apple_runtime(); runtime.max_record_alignment = 4096;
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "record alignment") != NULL);

    /* Prove target flags are load-bearing artifact identity. */
    info.flags ^= COLI_TARGET_F_STAGING_REQUIRED;
    CHECK(coli_target_artifact_fingerprint(&info, digest, error, sizeof(error)) == 0);
    CHECK(memcmp(digest, info.artifact_fingerprint, 32) != 0);
    info.flags ^= COLI_TARGET_F_STAGING_REQUIRED;

    coli_target_info_free(&info);
    return 0;
}

static int test_cuda(const char *path) {
    ColiTargetInfo info;
    ColiRuntimeTarget runtime;
    unsigned char digest[32];
    char error[512];

    CHECK(coli_target_read_package(path, &info, error, sizeof(error)) == 0);
    CHECK(info.target_os == COLI_TARGET_OS_LINUX);
    CHECK(info.target_arch == COLI_TARGET_ARCH_X86_64);
    CHECK(info.backend == COLI_TARGET_BACKEND_CUDA);
    CHECK(info.gpu_kind == COLI_TARGET_GPU_CUDA_SM);
    CHECK(info.gpu_capability_min == 89 && info.gpu_capability_max == 89);
    CHECK(info.cpu_feature_mask == (COLI_TARGET_CPU_X86_AVX2 | COLI_TARGET_CPU_X86_FMA));
    CHECK(!strcmp(info.profile_name, "linux-x86_64-cuda-sm89-v1"));
    CHECK(!strcmp(info.target_triple, "x86_64-linux-gnu-cuda-sm89"));
    CHECK(digest_eq_hex(info.artifact_fingerprint,
          "2bb6edc7fa23bbe96fcebb21d585c8d92030f0060b825b3a1c2e6ca41ea8fa75"));
    CHECK(coli_target_artifact_fingerprint(&info, digest, error, sizeof(error)) == 0);
    CHECK(!memcmp(digest, info.artifact_fingerprint, 32));

    runtime = cuda_runtime();
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) == 0);
    runtime.gpu_capability = 80;
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "CUDA capability") != NULL);
    runtime = cuda_runtime(); runtime.cpu_feature_mask = COLI_TARGET_CPU_X86_AVX2;
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "CPU features") != NULL);
    runtime = cuda_runtime(); runtime.runtime_features &= ~COLI_TARGET_RUNTIME_HOST_PINNED_STAGING;
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "runtime features") != NULL);
    runtime = cuda_runtime(); runtime.kernel_abi = 2;
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "kernel ABI") != NULL);
    runtime = cuda_runtime(); runtime.profile_name = "linux-x86_64-cuda-v2";
    CHECK(coli_target_check_compatibility(&info, &runtime, error, sizeof(error)) != 0);
    CHECK(strstr(error, "target profile mismatch") != NULL);

    coli_target_info_free(&info);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s APPLE_FIXTURE CUDA_FIXTURE\n", argv[0]);
        return 2;
    }
    CHECK(test_apple(argv[1]) == 0);
    CHECK(test_cuda(argv[2]) == 0);
    puts("test_coli_target: ok");
    return 0;
}

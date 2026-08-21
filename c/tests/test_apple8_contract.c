#include "../apple8_contract.h"
#include "../coli_target.h"
#include "../expert_representation.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

static ColiExpertMatrixInfo good_matrix(uint64_t rows, uint64_t columns) {
    ColiExpertMatrixInfo m;
    uint64_t bytes = 0;
    memset(&m, 0, sizeof(m));
    (void)coli_apple8_tile_matrix_bytes(rows, columns, &bytes);
    m.role = 1;
    m.math_format = COLI_APPLE8_MXFP4_MATH_FORMAT;
    m.scale_format = COLI_APPLE8_MXFP4_SCALE_FORMAT;
    m.weight_codec = COLI_CSF_CODEC_NONE;
    m.scale_codec = COLI_CSF_CODEC_NONE;
    m.layout = COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1;
    m.rows = rows;
    m.columns = columns;
    m.scale_block_rows = COLI_APPLE8_MXFP4_SCALE_BLOCK_ROWS;
    m.scale_block_columns = COLI_APPLE8_MXFP4_SCALE_BLOCK_COLUMNS;
    m.group_size = COLI_APPLE8_MXFP4_GROUP_SIZE;
    m.weight_offset = 448;
    m.weight_stored_bytes = bytes;
    m.weight_decoded_bytes = bytes;
    return m;
}

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

static int test_sizes(void) {
    const struct { uint64_t r, c, bytes; } cases[] = {
        {1,1,136}, {1,31,136}, {1,32,136}, {1,33,272},
        {7,32,136}, {8,32,136}, {9,32,272}, {8,31,136},
        {8,33,272}, {9,33,544},
    };
    size_t i;
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        uint64_t bytes = 0;
        CHECK(coli_apple8_tile_matrix_bytes(cases[i].r, cases[i].c, &bytes) == 0);
        CHECK(bytes == cases[i].bytes);
    }
    {
        uint64_t bytes = 0;
        CHECK(coli_apple8_tile_matrix_bytes(0, 32, &bytes) != 0);
        CHECK(coli_apple8_tile_matrix_bytes(UINT64_MAX, UINT64_MAX, &bytes) != 0);
    }
    return 0;
}

static int test_descriptor(void) {
    ColiExpertMatrixInfo m = good_matrix(9, 33);
    uint64_t expected = 0;
    CHECK(coli_apple8_matrix_descriptor_valid(&m, &expected));
    CHECK(expected == 544);

    m.layout = 0x7131; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.layout = COLI_CSF_LAYOUT_CANONICAL; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.math_format = COLI_CSF_MATH_INT4_PACKED; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_format = COLI_CSF_SCALE_F16; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_block_rows = 2; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_block_columns = 16; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.group_size = 32; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.weight_decoded_bytes--; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.weight_stored_bytes--; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_offset = 1024; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_stored_bytes = 1; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_decoded_bytes = 1; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_codec = COLI_CSF_CODEC_RANS256_G0_NIBBLE; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_codec_table_id = 7; CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    return 0;
}

static int test_identity(void) {
    ColiExpertMatrixInfo m = good_matrix(8, 32);
    ColiRepresentationId resident, supported, wrong;
    char error[256] = {0};
    CHECK(coli_target_resolve_matrix_representation(
        COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, &m,
        &resident, error, sizeof(error)) == 0);
    CHECK(resident.math_format == COLI_CSF_MATH_MXFP4_E2M1);
    CHECK(resident.scale_format == COLI_CSF_SCALE_UE8M0);
    CHECK(resident.execution_layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
    CHECK(resident.execution_layout_abi == COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1);
    CHECK(resident.kernel_abi == COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1);
    CHECK(resident.target_class == COLI_TARGET_CLASS_APPLE8_METAL_V1);
    CHECK(resident.group_size == 0);
    CHECK(resident.scale_block_rows == 1);
    CHECK(resident.scale_block_columns == 32);

    supported = resident;
    CHECK(coli_representation_backend_can_execute(&resident, &supported));
    wrong = supported; wrong.kernel_abi++; CHECK(!coli_representation_backend_can_execute(&resident, &wrong));
    wrong = supported; wrong.target_class++; CHECK(!coli_representation_backend_can_execute(&resident, &wrong));
    wrong = supported; wrong.execution_layout = COLI_CSF_LAYOUT_CANONICAL; CHECK(!coli_representation_backend_can_execute(&resident, &wrong));

    CHECK(!coli_target_profile_accepts_layout(COLI_TARGET_PROFILE_PORTABLE_V1, COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1));
    CHECK(!coli_target_profile_accepts_layout(COLI_TARGET_PROFILE_LINUX_X86_64_AVX2_V1, COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1));
    CHECK(!coli_target_profile_accepts_layout("linux-x86_64-cuda-v1", COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1));
    CHECK(!coli_target_profile_accepts_layout(COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, COLI_CSF_LAYOUT_CANONICAL));
    CHECK(!coli_target_profile_accepts_layout(COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, 0x7131));
    return 0;
}

static int test_runtime_contract(void) {
    ColiRuntimeTarget r = good_runtime();
    char error[256] = {0};
    CHECK(coli_target_check_compatibility(COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1,
                                         &r, COLI_APPLE8_RECORD_ALIGNMENT,
                                         error, sizeof(error)) == 0);
    r = good_runtime(); r.target_profile_abi++; CHECK(coli_target_check_compatibility(COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, &r, COLI_APPLE8_RECORD_ALIGNMENT, error, sizeof(error)) != 0);
    r = good_runtime(); r.execution_layout_abi++; CHECK(coli_target_check_compatibility(COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, &r, COLI_APPLE8_RECORD_ALIGNMENT, error, sizeof(error)) != 0);
    r = good_runtime(); r.kernel_abi++; CHECK(coli_target_check_compatibility(COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, &r, COLI_APPLE8_RECORD_ALIGNMENT, error, sizeof(error)) != 0);
    r = good_runtime(); r.target_class++; CHECK(coli_target_check_compatibility(COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, &r, COLI_APPLE8_RECORD_ALIGNMENT, error, sizeof(error)) != 0);
    r = good_runtime(); r.gpu_family = COLI_APPLE8_GPU_FAMILY_MIN - 1; CHECK(coli_target_check_compatibility(COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, &r, COLI_APPLE8_RECORD_ALIGNMENT, error, sizeof(error)) != 0);
    r = good_runtime(); r.runtime_features &= ~COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE; CHECK(coli_target_check_compatibility(COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, &r, COLI_APPLE8_RECORD_ALIGNMENT, error, sizeof(error)) != 0);
    return 0;
}

int main(void) {
    CHECK(test_sizes() == 0);
    CHECK(test_descriptor() == 0);
    CHECK(test_identity() == 0);
    CHECK(test_runtime_contract() == 0);
    puts("test_apple8_contract: ok");
    return 0;
}

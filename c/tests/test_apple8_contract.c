#include "../apple8_contract.h"
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
    m.math_format = COLI_CSF_MATH_MXFP4_E2M1;
    m.scale_format = COLI_CSF_SCALE_UE8M0;
    m.weight_codec = COLI_CSF_CODEC_NONE;
    m.scale_codec = COLI_CSF_CODEC_NONE;
    m.layout = COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1;
    m.rows = rows;
    m.columns = columns;
    m.scale_block_rows = 1;
    m.scale_block_columns = 32;
    m.weight_offset = 448;
    m.weight_stored_bytes = bytes;
    m.weight_decoded_bytes = bytes;
    return m;
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

    m.layout = 0x7131;
    CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.math_format = COLI_CSF_MATH_INT4_PACKED;
    CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_format = COLI_CSF_SCALE_F16;
    CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_block_columns = 16;
    CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.weight_decoded_bytes--;
    CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_offset = 1024; m.scale_stored_bytes = 1; m.scale_decoded_bytes = 1;
    CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    m = good_matrix(9, 33); m.scale_codec_table_id = 7;
    CHECK(!coli_apple8_matrix_descriptor_valid(&m, NULL));
    return 0;
}

static int test_identity(void) {
    ColiExpertMatrixInfo m = good_matrix(8, 32);
    ColiRepresentationId resident, supported, wrong;
    CHECK(coli_representation_from_csf_matrix(
        &resident, &m,
        COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1,
        COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1,
        COLI_TARGET_CLASS_APPLE8_METAL_V1, 0) == 0);
    supported = resident;
    CHECK(coli_representation_backend_can_execute(&resident, &supported));
    wrong = supported; wrong.kernel_abi++;
    CHECK(!coli_representation_backend_can_execute(&resident, &wrong));
    wrong = supported; wrong.target_class++;
    CHECK(!coli_representation_backend_can_execute(&resident, &wrong));
    wrong = supported; wrong.execution_layout = COLI_CSF_LAYOUT_CANONICAL;
    CHECK(!coli_representation_backend_can_execute(&resident, &wrong));

    CHECK(COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1 == 0x0103u);
    CHECK(COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1 == 1u);
    CHECK(COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1 == 1u);
    CHECK(COLI_TARGET_CLASS_APPLE8_METAL_V1 == 0x01000001u);
    CHECK(coli_apple8_target_contract_compatible(
        COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, 1, 1, 0x01000001u));
    CHECK(!coli_apple8_target_contract_compatible(
        COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, 2, 1, 0x01000001u));
    CHECK(!coli_apple8_target_contract_compatible(
        COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, 1, 2, 0x01000001u));
    CHECK(!coli_apple8_target_contract_compatible(
        COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, 1, 1, 2));
    CHECK(!coli_apple8_target_contract_compatible("portable-v1", 1, 1, 0x01000001u));
    return 0;
}

int main(void) {
    CHECK(test_sizes() == 0);
    CHECK(test_descriptor() == 0);
    CHECK(test_identity() == 0);
    puts("test_apple8_contract: ok");
    return 0;
}

#include <stdio.h>
#include <string.h>

#include "../mxfp4_expert.h"

static ColiExpertMatrixInfo matrix(uint16_t role,
                                   uint64_t rows,
                                   uint64_t columns) {
    ColiExpertMatrixInfo m;
    memset(&m, 0, sizeof(m));
    m.role = role;
    m.math_format = COLI_CSF_MATH_MXFP4_E2M1;
    m.scale_format = COLI_CSF_SCALE_UE8M0;
    m.weight_codec = COLI_CSF_CODEC_NONE;
    m.scale_codec = COLI_CSF_CODEC_NONE;
    m.layout = COLI_CSF_LAYOUT_CANONICAL;
    m.rows = rows;
    m.columns = columns;
    m.scale_block_rows = 1;
    m.scale_block_columns = 32;
    m.weight_stored_bytes = m.weight_decoded_bytes = rows * ((columns + 1) / 2);
    m.scale_stored_bytes = m.scale_decoded_bytes = rows * ((columns + 31) / 32);
    return m;
}

static int expect_valid(void) {
    ColiExpertInfo info;
    memset(&info, 0, sizeof(info));
    /* Intentionally shuffle descriptor order: role IDs, not descriptor index,
     * define the semantic matrix. */
    info.matrices[0] = matrix(COLI_MXFP4_EXPERT_ROLE_DOWN, 64, 32);
    info.matrices[1] = matrix(COLI_MXFP4_EXPERT_ROLE_GATE, 32, 64);
    info.matrices[2] = matrix(COLI_MXFP4_EXPERT_ROLE_UP, 32, 64);
    /* Compiler order is gate W/S, up W/S, down W/S; descriptor order is
     * intentionally shuffled above to prove roles, not indices, drive it. */
    info.matrices[1].weight_offset = 448;
    info.matrices[1].scale_offset = 1472;
    info.matrices[2].weight_offset = 1536;
    info.matrices[2].scale_offset = 2560;
    info.matrices[0].weight_offset = 2624;
    info.matrices[0].scale_offset = 3648;
    info.logical_bytes = 3264;

    ColiMxfp4ExpertLayout layout;
    char error[256] = {0};
    if (coli_mxfp4_expert_validate_info(&info, 64, 32, &layout,
                                        error, sizeof(error)) != 0) {
        fprintf(stderr, "valid MXFP4 expert rejected: %s\n", error);
        return 1;
    }
    if (layout.gate_weight_bytes != 1024 || layout.gate_scale_bytes != 64 ||
        layout.up_weight_bytes != 1024 || layout.up_scale_bytes != 64 ||
        layout.down_weight_bytes != 1024 || layout.down_scale_bytes != 64 ||
        layout.resident_bytes != 3264 ||
        layout.record_span_offset != 448 || layout.record_span_bytes != 3264 ||
        layout.gate_weight_span_offset != 0 ||
        layout.gate_scale_span_offset != 1024 ||
        layout.up_weight_span_offset != 1088 ||
        layout.up_scale_span_offset != 2112 ||
        layout.down_weight_span_offset != 2176 ||
        layout.down_scale_span_offset != 3200) {
        fprintf(stderr, "unexpected MXFP4 layout size: %zu\n",
                layout.resident_bytes);
        return 1;
    }
    return 0;
}

static int expect_bad_block_geometry(void) {
    ColiExpertInfo info;
    memset(&info, 0, sizeof(info));
    info.matrices[0] = matrix(COLI_MXFP4_EXPERT_ROLE_GATE, 32, 64);
    info.matrices[1] = matrix(COLI_MXFP4_EXPERT_ROLE_UP, 32, 64);
    info.matrices[2] = matrix(COLI_MXFP4_EXPERT_ROLE_DOWN, 64, 32);
    info.matrices[1].scale_block_columns = 64;
    info.logical_bytes = 3264;

    ColiMxfp4ExpertLayout layout;
    char error[256] = {0};
    if (coli_mxfp4_expert_validate_info(&info, 64, 32, &layout,
                                        error, sizeof(error)) == 0) {
        fprintf(stderr, "invalid MXFP4 scale geometry was accepted\n");
        return 1;
    }
    if (strstr(error, "1x32") == NULL) {
        fprintf(stderr, "unexpected geometry error: %s\n", error);
        return 1;
    }
    return 0;
}

static int expect_bad_span_size(void) {
    ColiExpertInfo info;
    memset(&info, 0, sizeof(info));
    info.matrices[0] = matrix(COLI_MXFP4_EXPERT_ROLE_GATE, 32, 64);
    info.matrices[1] = matrix(COLI_MXFP4_EXPERT_ROLE_UP, 32, 64);
    info.matrices[2] = matrix(COLI_MXFP4_EXPERT_ROLE_DOWN, 64, 32);
    info.matrices[2].weight_decoded_bytes--;
    info.logical_bytes = 3264;

    ColiMxfp4ExpertLayout layout;
    char error[256] = {0};
    if (coli_mxfp4_expert_validate_info(&info, 64, 32, &layout,
                                        error, sizeof(error)) == 0) {
        fprintf(stderr, "invalid MXFP4 weight span was accepted\n");
        return 1;
    }
    if (strstr(error, "weight span") == NULL) {
        fprintf(stderr, "unexpected span error: %s\n", error);
        return 1;
    }
    return 0;
}

int main(void) {
    if (expect_valid() != 0) return 1;
    if (expect_bad_block_geometry() != 0) return 1;
    if (expect_bad_span_size() != 0) return 1;
    puts("test_mxfp4_expert: ok");
    return 0;
}

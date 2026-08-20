#include "../expert_representation.h"

#include <stdio.h>

static ColiExpertMatrixInfo mxfp4_matrix(uint16_t codec) {
    ColiExpertMatrixInfo m = {0};
    m.math_format = COLI_CSF_MATH_MXFP4_E2M1;
    m.scale_format = COLI_CSF_SCALE_UE8M0;
    m.weight_codec = codec;
    m.scale_codec = COLI_CSF_CODEC_NONE;
    m.layout = COLI_CSF_LAYOUT_CANONICAL;
    m.group_size = 32;
    m.scale_block_rows = 1;
    m.scale_block_columns = 32;
    return m;
}

int main(void) {
    ColiExpertMatrixInfo raw = mxfp4_matrix(COLI_CSF_CODEC_NONE);
    ColiExpertMatrixInfo compressed = mxfp4_matrix(COLI_CSF_CODEC_RANS256_G0_NIBBLE);
    ColiRepresentationId baseline;
    ColiRepresentationId same_from_compressed;

    if (coli_representation_from_csf_matrix(&baseline, &raw, 1, 1, 0, 0) != 0 ||
        coli_representation_from_csf_matrix(
            &same_from_compressed, &compressed, 1, 1, 0, 0) != 0 ||
        !coli_representation_equal(&baseline, &same_from_compressed))
        return 1;

    /* Exact Apple8 repack: identical quantized values/scale geometry, distinct
     * physical layout, target class and kernel ABI. Numeric IDs are test
     * sentinels until #26's registry lands; no parallel production enum exists. */
    ColiRepresentationId apple8 = baseline;
    apple8.execution_layout = 0x0102;
    apple8.execution_layout_abi = 1;
    apple8.kernel_abi = 3;
    apple8.target_class = 0x0000a008u;
    if (coli_representation_equal(&baseline, &apple8) ||
        !coli_representation_exact_math_compatible(&baseline, &apple8) ||
        !coli_representation_transform_may_exist(&baseline, &apple8))
        return 2;

    ColiRepresentationId apple8_kernel4 = apple8;
    apple8_kernel4.kernel_abi = 4;
    if (coli_representation_equal(&apple8, &apple8_kernel4) ||
        coli_representation_backend_can_execute(&apple8, &apple8_kernel4) ||
        !coli_representation_exact_math_compatible(&apple8, &apple8_kernel4))
        return 3;

    ColiRepresentationId lossy = apple8;
    lossy.math_format = COLI_CSF_MATH_INT4_PACKED;
    lossy.scale_format = COLI_CSF_SCALE_F16;
    if (coli_representation_exact_math_compatible(&apple8, &lossy) ||
        coli_representation_transform_may_exist(&apple8, &lossy))
        return 4;

    ColiRepresentationId different_scale_geometry = apple8;
    different_scale_geometry.scale_block_columns = 64;
    if (coli_representation_exact_math_compatible(
            &apple8, &different_scale_geometry))
        return 5;

    /* Same high-level type can identify CPU/CUDA-native representations; there
     * are deliberately no Metal handles/enums in the shared contract. */
    ColiRepresentationId cpu = apple8;
    cpu.execution_layout = 0x0202;
    cpu.target_class = 0x0000c001u;
    ColiRepresentationId cuda = apple8;
    cuda.execution_layout = 0x0302;
    cuda.target_class = 0x0000c0dau;
    if (!coli_representation_known(&cpu) || !coli_representation_known(&cuda) ||
        coli_representation_backend_can_execute(&cpu, &cuda) ||
        !coli_representation_backend_can_execute(&apple8, &apple8))
        return 6;

    ColiRepresentationId unsupported = apple8;
    unsupported.execution_layout = COLI_CSF_LAYOUT_INVALID;
    if (coli_representation_known(&unsupported) ||
        coli_representation_backend_can_execute(&unsupported, &apple8))
        return 7;

    puts("PASS expert representation identity");
    return 0;
}

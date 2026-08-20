#include "../backend_metal_representation.h"
#include "../expert_representation.h"

#include <stdio.h>
#include <string.h>

static ColiExpertInfo homogeneous_expert(uint16_t math, uint16_t scale,
                                         uint16_t layout,
                                         uint32_t block_rows,
                                         uint32_t block_columns,
                                         uint32_t group_size) {
    ColiExpertInfo info;
    memset(&info, 0, sizeof(info));
    info.layer = 4;
    info.expert = 17;
    for (int i = 0; i < 3; i++) {
        info.matrices[i].role = (uint16_t)(i + 1);
        info.matrices[i].math_format = math;
        info.matrices[i].scale_format = scale;
        info.matrices[i].layout = layout;
        info.matrices[i].scale_block_rows = block_rows;
        info.matrices[i].scale_block_columns = block_columns;
        info.matrices[i].group_size = group_size;
    }
    return info;
}

int main(void) {
    const char *apple = COLI_CSF_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    const char *cpu = COLI_CSF_PROFILE_LINUX_X86_64_AVX2_V1;
    const char *cuda = "linux-x86_64-cuda-v1";

    ColiRepresentationId a;
    if (coli_representation_init(
            &a, COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0,
            COLI_CSF_LAYOUT_CANONICAL, 1, 1, 1, 32, 32,
            COLI_REPRESENTATION_F_NONE, apple) != 0 ||
        !coli_representation_valid(&a))
        return 1;

    ColiRepresentationId same = a;
    if (!coli_representation_equal(&a, &same) ||
        !coli_representation_exact_math_compatible(&a, &same) ||
        coli_representation_transform_may_exist(&a, &same))
        return 2;

    /* Same quantized math, different physical layout: distinct identity but an
     * exact physical transform is semantically possible. */
    ColiRepresentationId rows16 = a;
    rows16.execution_layout = COLI_CSF_LAYOUT_ROWS16;
    if (coli_representation_equal(&a, &rows16) ||
        !coli_representation_exact_math_compatible(&a, &rows16) ||
        !coli_representation_transform_may_exist(&a, &rows16))
        return 3;

    /* Same bytes/layout class but a different kernel ABI is not executable by
     * an ABI-1 backend and remains a distinct representation identity. */
    ColiRepresentationId kernel2 = a;
    kernel2.kernel_abi = 2;
    if (coli_representation_equal(&a, &kernel2) ||
        coli_representation_backend_can_execute(&kernel2, apple, 1, 1) ||
        !coli_representation_backend_can_execute(&kernel2, apple, 1, 2))
        return 4;

    ColiRepresentationId int4 = a;
    int4.math_format = COLI_CSF_MATH_INT4_GROUPED;
    int4.scale_format = COLI_CSF_SCALE_F32;
    if (coli_representation_exact_math_compatible(&a, &int4) ||
        coli_representation_transform_may_exist(&a, &int4))
        return 5;

    /* CPU and CUDA target-native identities use the same high-level contract
     * without Metal-only fields. Unsupported target classes fail closed. */
    ColiRepresentationId cpu_rep;
    ColiRepresentationId cuda_rep;
    if (coli_representation_init(
            &cpu_rep, COLI_CSF_MATH_INT4_GROUPED, COLI_CSF_SCALE_F32,
            COLI_CSF_LAYOUT_CANONICAL, 1, 1, 0, 0, 32,
            COLI_REPRESENTATION_F_NONE, cpu) != 0 ||
        coli_representation_init(
            &cuda_rep, COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0,
            COLI_CSF_LAYOUT_CANONICAL, 1, 3, 1, 32, 32,
            COLI_REPRESENTATION_F_NONE, cuda) != 0 ||
        coli_representation_backend_can_execute(&cpu_rep, apple, 1, 1) ||
        coli_representation_backend_can_execute(&cuda_rep, apple, 1, 1) ||
        !coli_representation_backend_can_execute(&cpu_rep, cpu, 1, 1) ||
        !coli_representation_backend_can_execute(&cuda_rep, cuda, 1, 3))
        return 6;

    /* Stored codec identity is deliberately not part of the resident identity. */
    ColiExpertInfo stored_a = homogeneous_expert(
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0,
        COLI_CSF_LAYOUT_CANONICAL, 1, 32, 32);
    ColiExpertInfo stored_b = stored_a;
    for (int i = 0; i < 3; i++) {
        stored_a.matrices[i].weight_codec = COLI_CSF_CODEC_NONE;
        stored_a.matrices[i].scale_codec = COLI_CSF_CODEC_NONE;
        stored_b.matrices[i].weight_codec = COLI_CSF_CODEC_RANS256_G0_NIBBLE;
        stored_b.matrices[i].scale_codec = COLI_CSF_CODEC_RANS256_G0_U8;
    }
    ColiRepresentationId from_a;
    ColiRepresentationId from_b;
    if (coli_representation_from_expert_info(
            &stored_a, apple, 1, 1, COLI_REPRESENTATION_F_NONE,
            &from_a) != 0 ||
        coli_representation_from_expert_info(
            &stored_b, apple, 1, 1, COLI_REPRESENTATION_F_NONE,
            &from_b) != 0 ||
        !coli_representation_equal(&from_a, &from_b))
        return 7;

    /* A mixed per-matrix execution representation cannot be collapsed by
     * guessing; fail closed until a compound descriptor exists. */
    stored_b.matrices[2].layout = COLI_CSF_LAYOUT_ROWS16;
    if (coli_representation_from_expert_info(
            &stored_b, apple, 1, 1, COLI_REPRESENTATION_F_NONE,
            &from_b) != -2)
        return 8;

    /* Current Metal translation consumes the common ID. It accepts canonical
     * MXFP4 but refuses to reinterpret ROWS16 as canonical bytes. */
    ColiMetalRepresentationBinding metal;
    if (!coli_representation_backend_can_execute(&a, apple, 1, 1) ||
        !coli_metal_representation_binding(&a, &metal) ||
        !metal.uses_mxfp4_entrypoint || metal.legacy_fmt != 0 ||
        !coli_representation_equal(&metal.representation, &a) ||
        coli_metal_representation_binding(&rows16, &metal) ||
        coli_metal_representation_binding(&kernel2, &metal) ||
        coli_metal_representation_binding(&cpu_rep, &metal) ||
        coli_metal_representation_binding(&cuda_rep, &metal))
        return 9;

    ColiRepresentationId metal_i4;
    if (coli_representation_init(
            &metal_i4, COLI_CSF_MATH_INT4_GROUPED, COLI_CSF_SCALE_F32,
            COLI_CSF_LAYOUT_CANONICAL, 1, 1, 0, 0, 64,
            COLI_REPRESENTATION_F_NONE, apple) != 0 ||
        !coli_metal_representation_binding(&metal_i4, &metal) ||
        metal.legacy_fmt != 4 || metal.qgroup_size != 64 ||
        metal.uses_mxfp4_entrypoint)
        return 10;

    int physical;
    ColiExpertResidentView view = {
        .key = {4, 17},
        .representation = a,
        .generation = 9,
        .tier_mask = COLI_EXPERT_TIER_UMA,
        .resident_bytes = 55,
        .allocation_bytes = 64,
        .physical_handle = &physical,
    };
    ColiMetalResidentBinding resident_binding;
    if (!coli_metal_resident_binding(&view, &resident_binding) ||
        resident_binding.generation != 9 ||
        resident_binding.tier_mask != COLI_EXPERT_TIER_UMA ||
        resident_binding.resident_bytes != 55 ||
        resident_binding.allocation_bytes != 64 ||
        resident_binding.physical_handle != &physical ||
        !resident_binding.execution.uses_mxfp4_entrypoint)
        return 11;
    view.generation = 0;
    if (coli_metal_resident_binding(&view, &resident_binding))
        return 12;

    ColiRepresentationId zero = {0};
    if (coli_representation_valid(&zero) ||
        coli_metal_representation_binding(&zero, &metal))
        return 13;

    puts("PASS expert representation identity");
    return 0;
}

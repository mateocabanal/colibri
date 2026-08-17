#include "../coli_target_profiles.h"

#include <stdint.h>
#include <stdio.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

static int test_registry(void) {
    CHECK(coli_target_layout_registered(COLI_LAYOUT_APPLE_LINEAR_ROW_MAJOR_V1));
    CHECK(coli_target_layout_registered(COLI_LAYOUT_APPLE_MXFP4_ROW32_V1));
    CHECK(coli_target_layout_registered(COLI_LAYOUT_X86_LINEAR_ROW_MAJOR_V1));
    CHECK(coli_target_layout_registered(COLI_LAYOUT_X86_MXFP4_ROW32_V1));
    CHECK(coli_target_layout_registered(COLI_LAYOUT_CUDA_LINEAR_ROW_MAJOR_V1));
    CHECK(coli_target_layout_registered(COLI_LAYOUT_CUDA_MXFP4_ROW32_V1));
    CHECK(!coli_target_layout_registered(0x0100));
    CHECK(!coli_target_layout_registered(0x0103));
    CHECK(!coli_target_layout_registered(0x0203));
    CHECK(!coli_target_layout_registered(0x0303));

    CHECK(coli_target_profile_accepts_layout(
        COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1,
        COLI_LAYOUT_APPLE_MXFP4_ROW32_V1));
    CHECK(!coli_target_profile_accepts_layout(
        COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1,
        COLI_LAYOUT_X86_MXFP4_ROW32_V1));
    CHECK(coli_target_profile_accepts_layout(
        COLI_PROFILE_LINUX_X86_64_CPU_AVX2_V1,
        COLI_LAYOUT_X86_LINEAR_ROW_MAJOR_V1));
    CHECK(!coli_target_profile_accepts_layout(
        COLI_PROFILE_LINUX_X86_64_CPU_AVX2_V1,
        COLI_LAYOUT_CUDA_LINEAR_ROW_MAJOR_V1));
    CHECK(coli_target_profile_accepts_layout(
        COLI_PROFILE_LINUX_X86_64_CUDA_V1,
        COLI_LAYOUT_CUDA_MXFP4_ROW32_V1));
    CHECK(!coli_target_profile_accepts_layout("unknown-profile-v1",
                                              COLI_LAYOUT_CUDA_MXFP4_ROW32_V1));
    return 0;
}

static int test_formats(void) {
    CHECK(coli_target_layout_accepts_format(
        COLI_LAYOUT_APPLE_LINEAR_ROW_MAJOR_V1,
        COLI_CSF_MATH_BF16, COLI_CSF_SCALE_NONE, 0, 0, 0));
    CHECK(!coli_target_layout_accepts_format(
        COLI_LAYOUT_APPLE_LINEAR_ROW_MAJOR_V1,
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0, 1, 32, 0));
    CHECK(coli_target_layout_accepts_format(
        COLI_LAYOUT_APPLE_MXFP4_ROW32_V1,
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0, 1, 32, 0));
    CHECK(!coli_target_layout_accepts_format(
        COLI_LAYOUT_APPLE_MXFP4_ROW32_V1,
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0, 1, 16, 0));
    CHECK(!coli_target_layout_accepts_format(
        COLI_LAYOUT_CUDA_MXFP4_ROW32_V1,
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_F32, 1, 32, 0));
    return 0;
}

static int test_sizes(void) {
    uint64_t w = 0, s = 0;
    CHECK(coli_target_layout_resident_bytes(
        COLI_LAYOUT_APPLE_MXFP4_ROW32_V1,
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0,
        3, 32, &w, &s) == 0);
    CHECK(w == 48);
    CHECK(s == 3);

    CHECK(coli_target_layout_resident_bytes(
        COLI_LAYOUT_X86_MXFP4_ROW32_V1,
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0,
        2, 33, &w, &s) == 0);
    CHECK(w == 34); /* 17 bytes/row */
    CHECK(s == 4);  /* 2 scales/row */

    CHECK(coli_target_layout_resident_bytes(
        COLI_LAYOUT_CUDA_MXFP4_ROW32_V1,
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0,
        1, 1, &w, &s) == 0);
    CHECK(w == 1 && s == 1);

    CHECK(coli_target_layout_resident_bytes(
        COLI_LAYOUT_APPLE_LINEAR_ROW_MAJOR_V1,
        COLI_CSF_MATH_BF16, COLI_CSF_SCALE_NONE,
        5, 7, &w, &s) == 0);
    CHECK(w == 70 && s == 0);

    CHECK(coli_target_layout_resident_bytes(
        COLI_LAYOUT_X86_LINEAR_ROW_MAJOR_V1,
        COLI_CSF_MATH_F32, COLI_CSF_SCALE_NONE,
        5, 7, &w, &s) == 0);
    CHECK(w == 140 && s == 0);

    CHECK(coli_target_layout_resident_bytes(
        COLI_LAYOUT_CUDA_LINEAR_ROW_MAJOR_V1,
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_NONE,
        1, 32, &w, &s) != 0);
    CHECK(coli_target_layout_resident_bytes(
        COLI_LAYOUT_CUDA_MXFP4_ROW32_V1,
        COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0,
        0, 32, &w, &s) != 0);
    CHECK(coli_target_layout_resident_bytes(
        COLI_LAYOUT_APPLE_LINEAR_ROW_MAJOR_V1,
        COLI_CSF_MATH_F32, COLI_CSF_SCALE_NONE,
        UINT64_MAX, 2, &w, &s) != 0);
    return 0;
}

int main(void) {
    CHECK(test_registry() == 0);
    CHECK(test_formats() == 0);
    CHECK(test_sizes() == 0);
    puts("test_coli_target_profiles: ok");
    return 0;
}

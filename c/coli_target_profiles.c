#include "coli_target_profiles.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

static int is_linear(uint16_t layout) {
    return layout == COLI_LAYOUT_APPLE_LINEAR_ROW_MAJOR_V1 ||
           layout == COLI_LAYOUT_X86_LINEAR_ROW_MAJOR_V1 ||
           layout == COLI_LAYOUT_CUDA_LINEAR_ROW_MAJOR_V1;
}

static int is_mxfp4(uint16_t layout) {
    return layout == COLI_LAYOUT_APPLE_MXFP4_ROW32_V1 ||
           layout == COLI_LAYOUT_X86_MXFP4_ROW32_V1 ||
           layout == COLI_LAYOUT_CUDA_MXFP4_ROW32_V1;
}

static int mul64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a && b > UINT64_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int linear_element_bytes(uint16_t math, uint64_t *bytes) {
    switch (math) {
    case COLI_CSF_MATH_F32:
    case COLI_CSF_MATH_I32:
    case COLI_CSF_MATH_U32:
        *bytes = 4; return 0;
    case COLI_CSF_MATH_F16:
    case COLI_CSF_MATH_BF16:
    case COLI_CSF_MATH_I16:
    case COLI_CSF_MATH_U16:
        *bytes = 2; return 0;
    case COLI_CSF_MATH_I64:
    case COLI_CSF_MATH_U64:
        *bytes = 8; return 0;
    case COLI_CSF_MATH_I8:
    case COLI_CSF_MATH_U8:
    case COLI_CSF_MATH_BOOL:
    case COLI_CSF_MATH_FP8_E4M3FN:
    case COLI_CSF_MATH_FP8_E5M2:
        *bytes = 1; return 0;
    default:
        return -1;
    }
}

int coli_target_layout_registered(uint16_t layout) {
    return is_linear(layout) || is_mxfp4(layout);
}

int coli_target_profile_accepts_layout(const char *profile, uint16_t layout) {
    if (!profile || !coli_target_layout_registered(layout)) return 0;
    if (!strcmp(profile, COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1))
        return layout == COLI_LAYOUT_APPLE_LINEAR_ROW_MAJOR_V1 ||
               layout == COLI_LAYOUT_APPLE_MXFP4_ROW32_V1;
    if (!strcmp(profile, COLI_PROFILE_LINUX_X86_64_CPU_AVX2_V1))
        return layout == COLI_LAYOUT_X86_LINEAR_ROW_MAJOR_V1 ||
               layout == COLI_LAYOUT_X86_MXFP4_ROW32_V1;
    if (!strcmp(profile, COLI_PROFILE_LINUX_X86_64_CUDA_V1))
        return layout == COLI_LAYOUT_CUDA_LINEAR_ROW_MAJOR_V1 ||
               layout == COLI_LAYOUT_CUDA_MXFP4_ROW32_V1;
    return 0;
}

int coli_target_layout_accepts_format(uint16_t layout,
                                      uint16_t math_format,
                                      uint16_t scale_format,
                                      uint32_t scale_block_rows,
                                      uint32_t scale_block_columns,
                                      uint32_t group_size) {
    uint64_t unused;
    if (is_linear(layout)) {
        if (scale_format != COLI_CSF_SCALE_NONE || scale_block_rows ||
            scale_block_columns || group_size)
            return 0;
        return linear_element_bytes(math_format, &unused) == 0;
    }
    if (is_mxfp4(layout)) {
        return math_format == COLI_CSF_MATH_MXFP4_E2M1 &&
               scale_format == COLI_CSF_SCALE_UE8M0 &&
               scale_block_rows == 1 && scale_block_columns == 32 &&
               group_size == 0;
    }
    return 0;
}

int coli_target_layout_resident_bytes(uint16_t layout,
                                      uint16_t math_format,
                                      uint16_t scale_format,
                                      uint64_t rows,
                                      uint64_t columns,
                                      uint64_t *weight_bytes,
                                      uint64_t *scale_bytes) {
    uint64_t elem, row_bytes, scales_per_row;
    if (!weight_bytes || !scale_bytes || !rows || !columns) return -1;
    *weight_bytes = 0;
    *scale_bytes = 0;
    if (is_linear(layout)) {
        if (scale_format != COLI_CSF_SCALE_NONE ||
            linear_element_bytes(math_format, &elem))
            return -1;
        if (mul64(rows, columns, &row_bytes) ||
            mul64(row_bytes, elem, weight_bytes))
            return -1;
        return 0;
    }
    if (is_mxfp4(layout)) {
        if (math_format != COLI_CSF_MATH_MXFP4_E2M1 ||
            scale_format != COLI_CSF_SCALE_UE8M0)
            return -1;
        row_bytes = columns / 2 + (columns & 1);
        scales_per_row = columns / 32 + ((columns & 31) != 0);
        if (mul64(rows, row_bytes, weight_bytes) ||
            mul64(rows, scales_per_row, scale_bytes))
            return -1;
        return 0;
    }
    return -1;
}

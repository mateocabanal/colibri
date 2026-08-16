#ifndef COLI_TARGET_PROFILES_H
#define COLI_TARGET_PROFILES_H

#include <stdint.h>

#include "coli_format.h"
#include "coli_target.h"

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_TARGET_PROFILE_ABI_V1 1u
#define COLI_EXECUTION_LAYOUT_ABI_V1 1u
#define COLI_KERNEL_ABI_V1 1u

/* Concrete CSF execution-layout IDs. */
enum {
    COLI_LAYOUT_APPLE_LINEAR_ROW_MAJOR_V1 = 0x0101,
    COLI_LAYOUT_APPLE_MXFP4_ROW32_V1 = 0x0102,

    COLI_LAYOUT_X86_LINEAR_ROW_MAJOR_V1 = 0x0201,
    COLI_LAYOUT_X86_MXFP4_ROW32_V1 = 0x0202,

    COLI_LAYOUT_CUDA_LINEAR_ROW_MAJOR_V1 = 0x0301,
    COLI_LAYOUT_CUDA_MXFP4_ROW32_V1 = 0x0302
};

#define COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1 "macos-arm64-metal-apple8-v1"
#define COLI_PROFILE_LINUX_X86_64_CPU_AVX2_V1 "linux-x86_64-cpu-avx2-v1"
#define COLI_PROFILE_LINUX_X86_64_CUDA_V1 "linux-x86_64-cuda-v1"

/* Returns nonzero only for a concrete registered layout ID. */
int coli_target_layout_registered(uint16_t layout);

/* Returns nonzero when LAYOUT belongs to PROFILE's target namespace and is one
 * of the concrete v1 IDs above. BLOB/NONE and compound MIXED are intentionally
 * not accepted here; callers handle those record-kind invariants separately. */
int coli_target_profile_accepts_layout(const char *profile, uint16_t layout);

/* Validates semantic format/scale geometry for a concrete layout. This is the
 * shared compiler/loader gate; it does not validate record offsets/lengths. */
int coli_target_layout_accepts_format(uint16_t layout,
                                      uint16_t math_format,
                                      uint16_t scale_format,
                                      uint32_t scale_block_rows,
                                      uint32_t scale_block_columns,
                                      uint32_t group_size);

/* Exact target resident-byte formulas. Returns 0 on success, -1 when the
 * layout/format/shape is invalid or arithmetic overflows. For linear layouts,
 * SCALE_BYTES is zero. For MXFP4 ROW32, WEIGHT_BYTES and SCALE_BYTES are the
 * separate resident planes. */
int coli_target_layout_resident_bytes(uint16_t layout,
                                      uint16_t math_format,
                                      uint16_t scale_format,
                                      uint64_t rows,
                                      uint64_t columns,
                                      uint64_t *weight_bytes,
                                      uint64_t *scale_bytes);

#ifdef __cplusplus
}
#endif

#endif /* COLI_TARGET_PROFILES_H */

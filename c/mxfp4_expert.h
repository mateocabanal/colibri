#ifndef COLIBRI_MXFP4_EXPERT_H
#define COLIBRI_MXFP4_EXPERT_H

#include <stddef.h>
#include <stdint.h>

#include "coli_format.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    COLI_MXFP4_EXPERT_ROLE_GATE = 1,
    COLI_MXFP4_EXPERT_ROLE_UP = 2,
    COLI_MXFP4_EXPERT_ROLE_DOWN = 3,
};

typedef struct ColiMxfp4ExpertLayout {
    /* Execution representation bound into the cache. Canonical keeps separate
     * E2M1 + UE8M0 spans. Apple8 stores the exact combined 8x32 tile stream in
     * each *_weight buffer; *_scale_bytes are runtime-layout sidecars, not
     * package scale payloads (Apple8 has no separate scale span). */
    uint16_t execution_layout;
    uint16_t reserved0;
    uint32_t reserved1;

    size_t gate_weight_bytes;
    size_t gate_scale_bytes;
    size_t up_weight_bytes;
    size_t up_scale_bytes;
    size_t down_weight_bytes;
    size_t down_scale_bytes;
    size_t resident_bytes;

    /* Canonical-only coalesced physical read view. Apple8 is decoded through
     * coli_package_decode_expert_record(), so record_span_bytes is zero and the
     * caller must use the generic loader rather than reading stored spans. */
    uint64_t record_span_offset;
    size_t record_span_bytes;
    size_t gate_weight_span_offset;
    size_t gate_scale_span_offset;
    size_t up_weight_span_offset;
    size_t up_scale_span_offset;
    size_t down_weight_span_offset;
    size_t down_scale_span_offset;
} ColiMxfp4ExpertLayout;

typedef struct ColiMxfp4ExpertBuffers {
    uint8_t *gate_weights;
    size_t gate_weight_capacity;
    uint8_t *gate_scales;
    size_t gate_scale_capacity;

    uint8_t *up_weights;
    size_t up_weight_capacity;
    uint8_t *up_scales;
    size_t up_scale_capacity;

    uint8_t *down_weights;
    size_t down_weight_capacity;
    uint8_t *down_scales;
    size_t down_scale_capacity;
} ColiMxfp4ExpertBuffers;

/* Validate gate/up/down geometry and return cache requirements for either
 * canonical MXFP4 or production Apple8 0x0103. All three matrices must use the
 * same execution layout. Apple8 matrix storage may be raw or rANS-compressed;
 * validation is against decoded execution bytes, never compressed bytes. */
int coli_mxfp4_expert_validate_info(const ColiExpertInfo *info,
                                    int hidden, int intermediate,
                                    ColiMxfp4ExpertLayout *layout,
                                    char *error, size_t error_size);

/* Canonical: reads six executable spans directly.
 * Apple8: synchronously reconstructs the raw target expert record (including
 * rANS decode when present), then copies the three exact combined tile streams
 * plus tiny runtime-layout sidecars into caller-owned cache storage.
 * No Apple8->canonical repack occurs. */
int coli_mxfp4_expert_load_ex(const ColiPackage *package,
                              const ColiRecordInfo *record,
                              int hidden, int intermediate,
                              const ColiMxfp4ExpertBuffers *buffers,
                              ColiMxfp4ExpertLayout *layout,
                              uint32_t read_flags,
                              char *error, size_t error_size);
int coli_mxfp4_expert_load(const ColiPackage *package,
                           const ColiRecordInfo *record,
                           int hidden, int intermediate,
                           const ColiMxfp4ExpertBuffers *buffers,
                           ColiMxfp4ExpertLayout *layout,
                           char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif

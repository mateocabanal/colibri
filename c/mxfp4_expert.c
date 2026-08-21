#include "mxfp4_expert.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "apple8_contract.h"
#include "mxfp4_runtime.h"

static int fail(char *error, size_t error_size, const char *format, ...) {
    if (error && error_size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, error_size, format, args);
        va_end(args);
    }
    return -1;
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int u64_to_size(uint64_t value, size_t *out) {
    if (value > (uint64_t)SIZE_MAX) return 0;
    *out = (size_t)value;
    return 1;
}

static uint16_t rd16(const unsigned char *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

static uint64_t rd64(const unsigned char *p) {
    return (uint64_t)p[0] |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

static const ColiExpertMatrixInfo *find_role(const ColiExpertInfo *info,
                                              uint16_t role) {
    const ColiExpertMatrixInfo *found = NULL;
    for (size_t i = 0; i < 3; ++i) {
        if (info->matrices[i].role != role) continue;
        if (found) return NULL;
        found = &info->matrices[i];
    }
    return found;
}

static int canonical_matrix_layout(const ColiExpertMatrixInfo *matrix,
                                   uint16_t role,
                                   uint64_t expected_rows,
                                   uint64_t expected_columns,
                                   size_t *weight_bytes,
                                   size_t *scale_bytes,
                                   char *error, size_t error_size) {
    const uint64_t packed_columns = (expected_columns + 1u) / 2u;
    const uint64_t scale_columns = (expected_columns + 31u) / 32u;
    uint64_t weights_u64 = 0, scales_u64 = 0;
    if (!matrix)
        return fail(error, error_size, "MXFP4 expert is missing matrix role %u",
                    (unsigned)role);
    if (matrix->math_format != COLI_CSF_MATH_MXFP4_E2M1 ||
        matrix->scale_format != COLI_CSF_SCALE_UE8M0 ||
        matrix->weight_codec != COLI_CSF_CODEC_NONE ||
        matrix->scale_codec != COLI_CSF_CODEC_NONE ||
        matrix->layout != COLI_CSF_LAYOUT_CANONICAL)
        return fail(error, error_size,
                    "MXFP4 role %u is not canonical codec-none E2M1/UE8M0",
                    (unsigned)role);
    if (matrix->rows != expected_rows || matrix->columns != expected_columns)
        return fail(error, error_size,
                    "MXFP4 role %u has shape %llux%llu, expected %llux%llu",
                    (unsigned)role,
                    (unsigned long long)matrix->rows,
                    (unsigned long long)matrix->columns,
                    (unsigned long long)expected_rows,
                    (unsigned long long)expected_columns);
    if (matrix->scale_block_rows != 1 || matrix->scale_block_columns != 32 ||
        matrix->group_size != 0)
        return fail(error, error_size,
                    "MXFP4 role %u has invalid canonical scale/group geometry",
                    (unsigned)role);
    if (!checked_mul_u64(expected_rows, packed_columns, &weights_u64) ||
        !checked_mul_u64(expected_rows, scale_columns, &scales_u64))
        return fail(error, error_size, "MXFP4 role %u byte size overflows u64",
                    (unsigned)role);
    if (matrix->weight_stored_bytes != weights_u64 ||
        matrix->weight_decoded_bytes != weights_u64 ||
        matrix->scale_stored_bytes != scales_u64 ||
        matrix->scale_decoded_bytes != scales_u64)
        return fail(error, error_size,
                    "MXFP4 role %u canonical span sizes disagree with geometry",
                    (unsigned)role);
    if (!u64_to_size(weights_u64, weight_bytes) ||
        !u64_to_size(scales_u64, scale_bytes))
        return fail(error, error_size,
                    "MXFP4 role %u does not fit this host address space",
                    (unsigned)role);
    return 0;
}

static int apple8_matrix_layout(const ColiExpertMatrixInfo *matrix,
                                uint16_t role,
                                uint64_t expected_rows,
                                uint64_t expected_columns,
                                size_t *weight_bytes,
                                size_t *sidecar_bytes,
                                char *error, size_t error_size) {
    uint64_t decoded = 0;
    if (!matrix)
        return fail(error, error_size, "MXFP4 expert is missing matrix role %u",
                    (unsigned)role);
    if (matrix->rows != expected_rows || matrix->columns != expected_columns)
        return fail(error, error_size,
                    "Apple8 MXFP4 role %u has shape %llux%llu, expected %llux%llu",
                    (unsigned)role,
                    (unsigned long long)matrix->rows,
                    (unsigned long long)matrix->columns,
                    (unsigned long long)expected_rows,
                    (unsigned long long)expected_columns);
    if (!coli_apple8_matrix_descriptor_valid(matrix, &decoded))
        return fail(error, error_size,
                    "Apple8 MXFP4 role %u violates production Design-A descriptor",
                    (unsigned)role);
    if (!u64_to_size(decoded, weight_bytes))
        return fail(error, error_size,
                    "Apple8 MXFP4 role %u decoded bytes exceed host address space",
                    (unsigned)role);
    *sidecar_bytes = sizeof(ColiMxfp4RuntimeTag);
    return 0;
}

int coli_mxfp4_expert_validate_info(const ColiExpertInfo *info,
                                    int hidden, int intermediate,
                                    ColiMxfp4ExpertLayout *layout,
                                    char *error, size_t error_size) {
    const ColiExpertMatrixInfo *gate, *up, *down;
    uint16_t execution_layout;
    size_t total = 0;
    uint64_t logical_execution_bytes = 0;
    if (!info || !layout)
        return fail(error, error_size, "MXFP4 expert validation received null input");
    if (hidden <= 0 || intermediate <= 0)
        return fail(error, error_size,
                    "MXFP4 expert dimensions must be positive (hidden=%d intermediate=%d)",
                    hidden, intermediate);

    memset(layout, 0, sizeof(*layout));
    gate = find_role(info, COLI_MXFP4_EXPERT_ROLE_GATE);
    up = find_role(info, COLI_MXFP4_EXPERT_ROLE_UP);
    down = find_role(info, COLI_MXFP4_EXPERT_ROLE_DOWN);
    if (!gate || !up || !down)
        return fail(error, error_size, "MXFP4 expert is missing or duplicates gate/up/down roles");
    if (gate->layout != up->layout || gate->layout != down->layout)
        return fail(error, error_size, "MXFP4 expert mixes execution layouts");
    execution_layout = gate->layout;
    layout->execution_layout = execution_layout;

    if (execution_layout == COLI_CSF_LAYOUT_CANONICAL) {
        if (canonical_matrix_layout(gate, COLI_MXFP4_EXPERT_ROLE_GATE,
                                    (uint64_t)intermediate, (uint64_t)hidden,
                                    &layout->gate_weight_bytes,
                                    &layout->gate_scale_bytes,
                                    error, error_size) ||
            canonical_matrix_layout(up, COLI_MXFP4_EXPERT_ROLE_UP,
                                    (uint64_t)intermediate, (uint64_t)hidden,
                                    &layout->up_weight_bytes,
                                    &layout->up_scale_bytes,
                                    error, error_size) ||
            canonical_matrix_layout(down, COLI_MXFP4_EXPERT_ROLE_DOWN,
                                    (uint64_t)hidden, (uint64_t)intermediate,
                                    &layout->down_weight_bytes,
                                    &layout->down_scale_bytes,
                                    error, error_size))
            return -1;
        logical_execution_bytes =
            (uint64_t)layout->gate_weight_bytes + layout->gate_scale_bytes +
            layout->up_weight_bytes + layout->up_scale_bytes +
            layout->down_weight_bytes + layout->down_scale_bytes;
    } else if (execution_layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1) {
        if (apple8_matrix_layout(gate, COLI_MXFP4_EXPERT_ROLE_GATE,
                                 (uint64_t)intermediate, (uint64_t)hidden,
                                 &layout->gate_weight_bytes,
                                 &layout->gate_scale_bytes,
                                 error, error_size) ||
            apple8_matrix_layout(up, COLI_MXFP4_EXPERT_ROLE_UP,
                                 (uint64_t)intermediate, (uint64_t)hidden,
                                 &layout->up_weight_bytes,
                                 &layout->up_scale_bytes,
                                 error, error_size) ||
            apple8_matrix_layout(down, COLI_MXFP4_EXPERT_ROLE_DOWN,
                                 (uint64_t)hidden, (uint64_t)intermediate,
                                 &layout->down_weight_bytes,
                                 &layout->down_scale_bytes,
                                 error, error_size))
            return -1;
        logical_execution_bytes =
            (uint64_t)layout->gate_weight_bytes +
            layout->up_weight_bytes +
            layout->down_weight_bytes;
    } else {
        return fail(error, error_size,
                    "MXFP4 expert layout 0x%04x is unsupported",
                    (unsigned)execution_layout);
    }

#define ADD_FIELD(field) \
    do { \
        if (layout->field > SIZE_MAX - total) \
            return fail(error, error_size, "MXFP4 resident byte size overflows size_t"); \
        total += layout->field; \
    } while (0)
    ADD_FIELD(gate_weight_bytes);
    ADD_FIELD(gate_scale_bytes);
    ADD_FIELD(up_weight_bytes);
    ADD_FIELD(up_scale_bytes);
    ADD_FIELD(down_weight_bytes);
    ADD_FIELD(down_scale_bytes);
#undef ADD_FIELD
    layout->resident_bytes = total;

    if (info->logical_bytes != logical_execution_bytes)
        return fail(error, error_size,
                    "MXFP4 expert logical_bytes=%llu, expected %llu from execution matrices",
                    (unsigned long long)info->logical_bytes,
                    (unsigned long long)logical_execution_bytes);

    if (execution_layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1) {
        /* Stored offsets may name compressed rANS records and are intentionally
         * unusable after decode. Force the caller onto the generic loader. */
        layout->record_span_offset = 0;
        layout->record_span_bytes = 0;
        return 0;
    }

    /* Canonical-only coalesced span view. */
    {
        const uint64_t span_offsets[6] = {
            gate->weight_offset, gate->scale_offset, up->weight_offset,
            up->scale_offset, down->weight_offset, down->scale_offset,
        };
        const size_t span_sizes[6] = {
            layout->gate_weight_bytes, layout->gate_scale_bytes,
            layout->up_weight_bytes, layout->up_scale_bytes,
            layout->down_weight_bytes, layout->down_scale_bytes,
        };
        uint64_t span_lo = UINT64_MAX, span_hi = 0;
        for (size_t i = 0; i < 6; ++i) {
            uint64_t end;
            if ((uint64_t)span_sizes[i] > UINT64_MAX - span_offsets[i])
                return fail(error, error_size, "MXFP4 executable span overflows u64");
            end = span_offsets[i] + (uint64_t)span_sizes[i];
            if (span_offsets[i] < span_lo) span_lo = span_offsets[i];
            if (end > span_hi) span_hi = end;
        }
        if (span_hi < span_lo || span_hi - span_lo > (uint64_t)SIZE_MAX)
            return fail(error, error_size, "MXFP4 coalesced span does not fit size_t");
        layout->record_span_offset = span_lo;
        layout->record_span_bytes = (size_t)(span_hi - span_lo);
#define SET_REL(field, absolute) \
        do { \
            uint64_t delta = (absolute) - span_lo; \
            if (delta > (uint64_t)SIZE_MAX) \
                return fail(error, error_size, "MXFP4 span offset does not fit size_t"); \
            layout->field = (size_t)delta; \
        } while (0)
        SET_REL(gate_weight_span_offset, gate->weight_offset);
        SET_REL(gate_scale_span_offset, gate->scale_offset);
        SET_REL(up_weight_span_offset, up->weight_offset);
        SET_REL(up_scale_span_offset, up->scale_offset);
        SET_REL(down_weight_span_offset, down->weight_offset);
        SET_REL(down_scale_span_offset, down->scale_offset);
#undef SET_REL
    }
    return 0;
}

static int check_buffer(const char *name, uint8_t *pointer, size_t capacity,
                        size_t required, char *error, size_t error_size) {
    if (required && !pointer)
        return fail(error, error_size, "MXFP4 %s buffer is null", name);
    if (capacity < required)
        return fail(error, error_size,
                    "MXFP4 %s buffer is %llu bytes, need %llu",
                    name, (unsigned long long)capacity,
                    (unsigned long long)required);
    return 0;
}

static int read_span(const ColiPackage *package, const ColiRecordInfo *record,
                     const ColiExpertMatrixInfo *matrix, int scale,
                     uint8_t *destination, size_t bytes, uint32_t read_flags,
                     char *error, size_t error_size) {
    const uint64_t offset = scale ? matrix->scale_offset : matrix->weight_offset;
    if (coli_package_read_range_ex(package, record, offset, destination, bytes,
                                   read_flags, error, error_size) != 0)
        return -1;
    return 0;
}

static int decoded_role_span(const unsigned char *record, size_t record_bytes,
                             uint16_t wanted_role,
                             const unsigned char **payload, size_t *payload_bytes,
                             char *error, size_t error_size) {
    if (!record || record_bytes < 448 || memcmp(record, "COLIEXPT", 8) ||
        rd16(record + 24) != 3)
        return fail(error, error_size, "decoded Apple8 expert envelope is malformed");
    for (size_t i = 0; i < 3; ++i) {
        const unsigned char *d = record + 64 + i * 128;
        uint64_t offset, stored, decoded;
        if (rd16(d) != wanted_role) continue;
        if (rd16(d + 8) != COLI_CSF_CODEC_NONE || rd16(d + 12) !=
            COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1)
            return fail(error, error_size,
                        "decoded Apple8 role %u did not reconstruct raw target bytes",
                        (unsigned)wanted_role);
        offset = rd64(d + 48);
        stored = rd64(d + 56);
        decoded = rd64(d + 64);
        if (stored != decoded || offset > record_bytes || stored > record_bytes - offset ||
            stored > SIZE_MAX)
            return fail(error, error_size,
                        "decoded Apple8 role %u span is invalid",
                        (unsigned)wanted_role);
        *payload = record + (size_t)offset;
        *payload_bytes = (size_t)stored;
        return 0;
    }
    return fail(error, error_size, "decoded Apple8 expert is missing role %u",
                (unsigned)wanted_role);
}

static int load_apple8(const ColiPackage *package,
                       const ColiRecordInfo *record,
                       const ColiMxfp4ExpertBuffers *buffers,
                       const ColiMxfp4ExpertLayout *layout,
                       char *error, size_t error_size) {
    uint64_t resident_u64 = 0;
    size_t written = 0;
    unsigned char *resident = NULL;
    const unsigned char *gate = NULL, *up = NULL, *down = NULL;
    size_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
    if (coli_package_expert_resident_bytes(package, record, &resident_u64,
                                           error, error_size) ||
        resident_u64 > SIZE_MAX)
        return fail(error, error_size, "Apple8 expert resident record exceeds host limits");
    resident = (unsigned char *)malloc((size_t)resident_u64);
    if (!resident)
        return fail(error, error_size, "out of memory decoding Apple8 expert");
    if (coli_package_decode_expert_record(package, record, resident,
                                          (size_t)resident_u64, &written,
                                          error, error_size) ||
        written != (size_t)resident_u64) {
        free(resident);
        return -1;
    }
    if (decoded_role_span(resident, written, COLI_MXFP4_EXPERT_ROLE_GATE,
                          &gate, &gate_bytes, error, error_size) ||
        decoded_role_span(resident, written, COLI_MXFP4_EXPERT_ROLE_UP,
                          &up, &up_bytes, error, error_size) ||
        decoded_role_span(resident, written, COLI_MXFP4_EXPERT_ROLE_DOWN,
                          &down, &down_bytes, error, error_size) ||
        gate_bytes != layout->gate_weight_bytes ||
        up_bytes != layout->up_weight_bytes ||
        down_bytes != layout->down_weight_bytes) {
        free(resident);
        if (error && error_size && !error[0])
            (void)snprintf(error, error_size,
                           "decoded Apple8 expert geometry changed during reconstruction");
        return -1;
    }
    memcpy(buffers->gate_weights, gate, gate_bytes);
    memcpy(buffers->up_weights, up, up_bytes);
    memcpy(buffers->down_weights, down, down_bytes);
    coli_mxfp4_runtime_tag_init(buffers->gate_scales,
                                COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
    coli_mxfp4_runtime_tag_init(buffers->up_scales,
                                COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
    coli_mxfp4_runtime_tag_init(buffers->down_scales,
                                COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
    free(resident);
    return 0;
}

int coli_mxfp4_expert_load_ex(const ColiPackage *package,
                              const ColiRecordInfo *record,
                              int hidden, int intermediate,
                              const ColiMxfp4ExpertBuffers *buffers,
                              ColiMxfp4ExpertLayout *layout,
                              uint32_t read_flags,
                              char *error, size_t error_size) {
    ColiExpertInfo info;
    const ColiExpertMatrixInfo *gate, *up, *down;
    if (!package || !record || !buffers || !layout)
        return fail(error, error_size, "MXFP4 expert load received null input");
    if (record->kind != COLI_CSF_REC_EXPERT)
        return fail(error, error_size, "record %llu is not an EXPERT record",
                    (unsigned long long)record->record_id);
    if (record->codec != COLI_CSF_CODEC_NONE)
        return fail(error, error_size,
                    "MXFP4 expert record %llu uses outer codec 0x%04x",
                    (unsigned long long)record->record_id,
                    (unsigned)record->codec);
    if (coli_package_expert_info(package, record, &info, error, error_size) != 0 ||
        coli_mxfp4_expert_validate_info(&info, hidden, intermediate, layout,
                                        error, error_size) != 0)
        return -1;

    if (check_buffer("gate weights", buffers->gate_weights,
                     buffers->gate_weight_capacity, layout->gate_weight_bytes,
                     error, error_size) ||
        check_buffer("gate scales/tag", buffers->gate_scales,
                     buffers->gate_scale_capacity, layout->gate_scale_bytes,
                     error, error_size) ||
        check_buffer("up weights", buffers->up_weights,
                     buffers->up_weight_capacity, layout->up_weight_bytes,
                     error, error_size) ||
        check_buffer("up scales/tag", buffers->up_scales,
                     buffers->up_scale_capacity, layout->up_scale_bytes,
                     error, error_size) ||
        check_buffer("down weights", buffers->down_weights,
                     buffers->down_weight_capacity, layout->down_weight_bytes,
                     error, error_size) ||
        check_buffer("down scales/tag", buffers->down_scales,
                     buffers->down_scale_capacity, layout->down_scale_bytes,
                     error, error_size))
        return -1;

    if (layout->execution_layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1) {
        (void)read_flags; /* synchronous decoder owns its package reads in PR3 */
        return load_apple8(package, record, buffers, layout, error, error_size);
    }

    gate = find_role(&info, COLI_MXFP4_EXPERT_ROLE_GATE);
    up = find_role(&info, COLI_MXFP4_EXPERT_ROLE_UP);
    down = find_role(&info, COLI_MXFP4_EXPERT_ROLE_DOWN);
    if (read_span(package, record, gate, 0, buffers->gate_weights,
                  layout->gate_weight_bytes, read_flags, error, error_size) ||
        read_span(package, record, gate, 1, buffers->gate_scales,
                  layout->gate_scale_bytes, read_flags, error, error_size) ||
        read_span(package, record, up, 0, buffers->up_weights,
                  layout->up_weight_bytes, read_flags, error, error_size) ||
        read_span(package, record, up, 1, buffers->up_scales,
                  layout->up_scale_bytes, read_flags, error, error_size) ||
        read_span(package, record, down, 0, buffers->down_weights,
                  layout->down_weight_bytes, read_flags, error, error_size) ||
        read_span(package, record, down, 1, buffers->down_scales,
                  layout->down_scale_bytes, read_flags, error, error_size))
        return -1;
    return 0;
}

int coli_mxfp4_expert_load(const ColiPackage *package,
                           const ColiRecordInfo *record,
                           int hidden, int intermediate,
                           const ColiMxfp4ExpertBuffers *buffers,
                           ColiMxfp4ExpertLayout *layout,
                           char *error, size_t error_size) {
    return coli_mxfp4_expert_load_ex(package, record, hidden, intermediate,
                                     buffers, layout, COLI_CSF_READ_DEFAULT,
                                     error, error_size);
}

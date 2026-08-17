#include "mxfp4_expert.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

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

static const ColiExpertMatrixInfo *find_role(const ColiExpertInfo *info,
                                              uint16_t role) {
    const ColiExpertMatrixInfo *found = NULL;
    for (size_t i = 0; i < 3; ++i) {
        if (info->matrices[i].role != role) continue;
        if (found) return NULL; /* duplicate role is invalid */
        found = &info->matrices[i];
    }
    return found;
}

static int matrix_layout(const ColiExpertMatrixInfo *matrix,
                         uint16_t role,
                         uint64_t expected_rows,
                         uint64_t expected_columns,
                         size_t *weight_bytes,
                         size_t *scale_bytes,
                         char *error, size_t error_size) {
    if (!matrix)
        return fail(error, error_size, "MXFP4 expert is missing matrix role %u",
                    (unsigned)role);
    if (matrix->math_format != COLI_CSF_MATH_MXFP4_E2M1)
        return fail(error, error_size,
                    "MXFP4 role %u has math format 0x%04x, expected 0x%04x",
                    (unsigned)role, (unsigned)matrix->math_format,
                    (unsigned)COLI_CSF_MATH_MXFP4_E2M1);
    if (matrix->scale_format != COLI_CSF_SCALE_UE8M0)
        return fail(error, error_size,
                    "MXFP4 role %u has scale format 0x%04x, expected UE8M0",
                    (unsigned)role, (unsigned)matrix->scale_format);
    if (matrix->weight_codec != COLI_CSF_CODEC_NONE ||
        matrix->scale_codec != COLI_CSF_CODEC_NONE)
        return fail(error, error_size,
                    "MXFP4 role %u uses a storage codec; direct cache loading requires codec none",
                    (unsigned)role);
    if (matrix->layout != COLI_CSF_LAYOUT_CANONICAL)
        return fail(error, error_size,
                    "MXFP4 role %u has layout 0x%04x, expected canonical",
                    (unsigned)role, (unsigned)matrix->layout);
    if (matrix->rows != expected_rows || matrix->columns != expected_columns)
        return fail(error, error_size,
                    "MXFP4 role %u has shape %llux%llu, expected %llux%llu",
                    (unsigned)role,
                    (unsigned long long)matrix->rows,
                    (unsigned long long)matrix->columns,
                    (unsigned long long)expected_rows,
                    (unsigned long long)expected_columns);
    if (matrix->scale_block_rows != 1 || matrix->scale_block_columns != 32)
        return fail(error, error_size,
                    "MXFP4 role %u has scale block %ux%u, expected 1x32",
                    (unsigned)role, matrix->scale_block_rows,
                    matrix->scale_block_columns);
    if (matrix->group_size != 0)
        return fail(error, error_size,
                    "MXFP4 role %u has group_size=%u, expected 0",
                    (unsigned)role, matrix->group_size);

    const uint64_t packed_columns = (expected_columns + 1u) / 2u;
    const uint64_t scale_columns = (expected_columns + 31u) / 32u;
    uint64_t weights_u64 = 0, scales_u64 = 0;
    if (!checked_mul_u64(expected_rows, packed_columns, &weights_u64) ||
        !checked_mul_u64(expected_rows, scale_columns, &scales_u64))
        return fail(error, error_size, "MXFP4 role %u byte size overflows u64",
                    (unsigned)role);
    if (matrix->weight_stored_bytes != weights_u64 ||
        matrix->weight_decoded_bytes != weights_u64)
        return fail(error, error_size,
                    "MXFP4 role %u weight span is %llu/%llu bytes, expected %llu",
                    (unsigned)role,
                    (unsigned long long)matrix->weight_stored_bytes,
                    (unsigned long long)matrix->weight_decoded_bytes,
                    (unsigned long long)weights_u64);
    if (matrix->scale_stored_bytes != scales_u64 ||
        matrix->scale_decoded_bytes != scales_u64)
        return fail(error, error_size,
                    "MXFP4 role %u scale span is %llu/%llu bytes, expected %llu",
                    (unsigned)role,
                    (unsigned long long)matrix->scale_stored_bytes,
                    (unsigned long long)matrix->scale_decoded_bytes,
                    (unsigned long long)scales_u64);
    if (!u64_to_size(weights_u64, weight_bytes) ||
        !u64_to_size(scales_u64, scale_bytes))
        return fail(error, error_size,
                    "MXFP4 role %u does not fit this host address space",
                    (unsigned)role);
    return 0;
}

int coli_mxfp4_expert_validate_info(const ColiExpertInfo *info,
                                    int hidden, int intermediate,
                                    ColiMxfp4ExpertLayout *layout,
                                    char *error, size_t error_size) {
    if (!info || !layout)
        return fail(error, error_size, "MXFP4 expert validation received null input");
    if (hidden <= 0 || intermediate <= 0)
        return fail(error, error_size,
                    "MXFP4 expert dimensions must be positive (hidden=%d intermediate=%d)",
                    hidden, intermediate);

    memset(layout, 0, sizeof(*layout));
    const ColiExpertMatrixInfo *gate = find_role(info, COLI_MXFP4_EXPERT_ROLE_GATE);
    const ColiExpertMatrixInfo *up = find_role(info, COLI_MXFP4_EXPERT_ROLE_UP);
    const ColiExpertMatrixInfo *down = find_role(info, COLI_MXFP4_EXPERT_ROLE_DOWN);

    if (matrix_layout(gate, COLI_MXFP4_EXPERT_ROLE_GATE,
                      (uint64_t)intermediate, (uint64_t)hidden,
                      &layout->gate_weight_bytes, &layout->gate_scale_bytes,
                      error, error_size) != 0 ||
        matrix_layout(up, COLI_MXFP4_EXPERT_ROLE_UP,
                      (uint64_t)intermediate, (uint64_t)hidden,
                      &layout->up_weight_bytes, &layout->up_scale_bytes,
                      error, error_size) != 0 ||
        matrix_layout(down, COLI_MXFP4_EXPERT_ROLE_DOWN,
                      (uint64_t)hidden, (uint64_t)intermediate,
                      &layout->down_weight_bytes, &layout->down_scale_bytes,
                      error, error_size) != 0)
        return -1;

    size_t total = 0;
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

    if (info->logical_bytes != (uint64_t)total)
        return fail(error, error_size,
                    "MXFP4 expert logical_bytes=%llu, expected %llu from matrix spans",
                    (unsigned long long)info->logical_bytes,
                    (unsigned long long)total);
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

int coli_mxfp4_expert_load_ex(const ColiPackage *package,
                              const ColiRecordInfo *record,
                              int hidden, int intermediate,
                              const ColiMxfp4ExpertBuffers *buffers,
                              ColiMxfp4ExpertLayout *layout,
                              uint32_t read_flags,
                              char *error, size_t error_size) {
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

    ColiExpertInfo info;
    if (coli_package_expert_info(package, record, &info, error, error_size) != 0)
        return -1;
    if (coli_mxfp4_expert_validate_info(&info, hidden, intermediate, layout,
                                        error, error_size) != 0)
        return -1;

    if (check_buffer("gate weights", buffers->gate_weights,
                     buffers->gate_weight_capacity, layout->gate_weight_bytes,
                     error, error_size) != 0 ||
        check_buffer("gate scales", buffers->gate_scales,
                     buffers->gate_scale_capacity, layout->gate_scale_bytes,
                     error, error_size) != 0 ||
        check_buffer("up weights", buffers->up_weights,
                     buffers->up_weight_capacity, layout->up_weight_bytes,
                     error, error_size) != 0 ||
        check_buffer("up scales", buffers->up_scales,
                     buffers->up_scale_capacity, layout->up_scale_bytes,
                     error, error_size) != 0 ||
        check_buffer("down weights", buffers->down_weights,
                     buffers->down_weight_capacity, layout->down_weight_bytes,
                     error, error_size) != 0 ||
        check_buffer("down scales", buffers->down_scales,
                     buffers->down_scale_capacity, layout->down_scale_bytes,
                     error, error_size) != 0)
        return -1;

    const ColiExpertMatrixInfo *gate = find_role(&info, COLI_MXFP4_EXPERT_ROLE_GATE);
    const ColiExpertMatrixInfo *up = find_role(&info, COLI_MXFP4_EXPERT_ROLE_UP);
    const ColiExpertMatrixInfo *down = find_role(&info, COLI_MXFP4_EXPERT_ROLE_DOWN);

    if (read_span(package, record, gate, 0, buffers->gate_weights,
                  layout->gate_weight_bytes, read_flags, error, error_size) != 0 ||
        read_span(package, record, gate, 1, buffers->gate_scales,
                  layout->gate_scale_bytes, read_flags, error, error_size) != 0 ||
        read_span(package, record, up, 0, buffers->up_weights,
                  layout->up_weight_bytes, read_flags, error, error_size) != 0 ||
        read_span(package, record, up, 1, buffers->up_scales,
                  layout->up_scale_bytes, read_flags, error, error_size) != 0 ||
        read_span(package, record, down, 0, buffers->down_weights,
                  layout->down_weight_bytes, read_flags, error, error_size) != 0 ||
        read_span(package, record, down, 1, buffers->down_scales,
                  layout->down_scale_bytes, read_flags, error, error_size) != 0)
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

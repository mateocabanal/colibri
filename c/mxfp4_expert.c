#include "mxfp4_expert.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "apple8_contract.h"

#define COLI_EXPERT_PREFIX_BYTES (64u + 3u * 128u)

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
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
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

static int canonical_sizes(uint64_t rows, uint64_t columns,
                           size_t *weight_bytes, size_t *scale_bytes,
                           char *error, size_t error_size) {
    uint64_t weights_u64 = 0, scales_u64 = 0;
    if (!rows || !columns ||
        !checked_mul_u64(rows, columns / 2u + (columns & 1u), &weights_u64) ||
        !checked_mul_u64(rows, columns / 32u + (columns % 32u != 0), &scales_u64))
        return fail(error, error_size, "MXFP4 canonical byte size overflows u64");
    if (!u64_to_size(weights_u64, weight_bytes) ||
        !u64_to_size(scales_u64, scale_bytes))
        return fail(error, error_size, "MXFP4 canonical bytes exceed host address space");
    return 0;
}

static int matrix_layout(const ColiExpertMatrixInfo *matrix,
                         uint16_t role,
                         uint64_t expected_rows,
                         uint64_t expected_columns,
                         size_t *weight_bytes,
                         size_t *scale_bytes,
                         uint16_t *source_layout,
                         uint64_t *source_logical_bytes,
                         char *error, size_t error_size) {
    uint64_t apple_bytes = 0;
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
    if (canonical_sizes(expected_rows, expected_columns,
                        weight_bytes, scale_bytes, error, error_size))
        return -1;

    if (matrix->layout == COLI_CSF_LAYOUT_CANONICAL) {
        if (matrix->weight_codec != COLI_CSF_CODEC_NONE ||
            matrix->scale_codec != COLI_CSF_CODEC_NONE ||
            matrix->weight_codec_table_id != 0 ||
            matrix->scale_codec_table_id != 0 ||
            matrix->weight_stored_bytes != *weight_bytes ||
            matrix->weight_decoded_bytes != *weight_bytes ||
            matrix->scale_stored_bytes != *scale_bytes ||
            matrix->scale_decoded_bytes != *scale_bytes)
            return fail(error, error_size,
                        "MXFP4 role %u canonical spans/codecs are invalid",
                        (unsigned)role);
        *source_layout = COLI_CSF_LAYOUT_CANONICAL;
        *source_logical_bytes = (uint64_t)*weight_bytes + (uint64_t)*scale_bytes;
        return 0;
    }

    if (matrix->layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1) {
        if (!coli_apple8_matrix_descriptor_valid(matrix, &apple_bytes))
            return fail(error, error_size,
                        "MXFP4 role %u violates Apple8 Design-A descriptor contract",
                        (unsigned)role);
        *source_layout = COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1;
        *source_logical_bytes = apple_bytes;
        return 0;
    }

    return fail(error, error_size,
                "MXFP4 role %u has unsupported execution layout 0x%04x",
                (unsigned)role, (unsigned)matrix->layout);
}

int coli_mxfp4_expert_validate_info(const ColiExpertInfo *info,
                                    int hidden, int intermediate,
                                    ColiMxfp4ExpertLayout *layout,
                                    char *error, size_t error_size) {
    const ColiExpertMatrixInfo *gate, *up, *down;
    uint16_t gate_layout = 0, up_layout = 0, down_layout = 0;
    uint64_t gate_logical = 0, up_logical = 0, down_logical = 0;
    uint64_t source_total = 0;
    size_t total = 0;

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

    if (matrix_layout(gate, COLI_MXFP4_EXPERT_ROLE_GATE,
                      (uint64_t)intermediate, (uint64_t)hidden,
                      &layout->gate_weight_bytes, &layout->gate_scale_bytes,
                      &gate_layout, &gate_logical, error, error_size) ||
        matrix_layout(up, COLI_MXFP4_EXPERT_ROLE_UP,
                      (uint64_t)intermediate, (uint64_t)hidden,
                      &layout->up_weight_bytes, &layout->up_scale_bytes,
                      &up_layout, &up_logical, error, error_size) ||
        matrix_layout(down, COLI_MXFP4_EXPERT_ROLE_DOWN,
                      (uint64_t)hidden, (uint64_t)intermediate,
                      &layout->down_weight_bytes, &layout->down_scale_bytes,
                      &down_layout, &down_logical, error, error_size))
        return -1;

    if (gate_layout != up_layout || gate_layout != down_layout)
        return fail(error, error_size,
                    "MXFP4 expert mixes canonical and Apple8 matrix layouts");
    layout->source_layout = gate_layout;

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

    if (gate_logical > UINT64_MAX - source_total) return fail(error, error_size, "MXFP4 logical bytes overflow");
    source_total += gate_logical;
    if (up_logical > UINT64_MAX - source_total) return fail(error, error_size, "MXFP4 logical bytes overflow");
    source_total += up_logical;
    if (down_logical > UINT64_MAX - source_total) return fail(error, error_size, "MXFP4 logical bytes overflow");
    source_total += down_logical;
    if (info->logical_bytes != source_total)
        return fail(error, error_size,
                    "MXFP4 expert logical_bytes=%llu, expected %llu from source layout",
                    (unsigned long long)info->logical_bytes,
                    (unsigned long long)source_total);

    if (layout->source_layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1) {
        /* Apple8 combines scales inside each tile and may be rANS-compressed.
         * There is no physical six-span fast path; load_ex will decode+detile. */
        layout->record_span_offset = 0;
        layout->record_span_bytes = 0;
        return 0;
    }

    /* Canonical source: retain the existing one-read coalesced-span fast path. */
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

int coli_mxfp4_apple8_detile_matrix(const uint8_t *tiles, size_t tile_bytes,
                                    uint64_t rows, uint64_t columns,
                                    uint8_t *weights, size_t weight_capacity,
                                    uint8_t *scales, size_t scale_capacity,
                                    char *error, size_t error_size) {
    uint64_t expected_tiles_u64 = 0;
    size_t weight_bytes = 0, scale_bytes = 0;
    uint64_t groups;

    if (!tiles || !weights || !scales || !rows || !columns)
        return fail(error, error_size, "Apple8 detile received invalid input");
    if (coli_apple8_tile_matrix_bytes(rows, columns, &expected_tiles_u64) ||
        expected_tiles_u64 > SIZE_MAX || tile_bytes != (size_t)expected_tiles_u64)
        return fail(error, error_size, "Apple8 detile tile byte count is invalid");
    if (canonical_sizes(rows, columns, &weight_bytes, &scale_bytes,
                        error, error_size))
        return -1;
    if (weight_capacity < weight_bytes || scale_capacity < scale_bytes)
        return fail(error, error_size,
                    "Apple8 detile output buffers are too small");

    memset(weights, 0, weight_bytes);
    memset(scales, 0, scale_bytes);
    groups = columns / COLI_APPLE8_MXFP4_TILE_COLUMNS +
             (columns % COLI_APPLE8_MXFP4_TILE_COLUMNS != 0);
    for (uint64_t o = 0; o < rows; ++o) {
        const uint64_t output_tile = o / COLI_APPLE8_MXFP4_TILE_ROWS;
        const uint64_t tile_row = o % COLI_APPLE8_MXFP4_TILE_ROWS;
        const uint64_t canonical_row_bytes = columns / 2u + (columns & 1u);
        for (uint64_t g = 0; g < groups; ++g) {
            const uint64_t tile_index = output_tile * groups + g;
            const uint8_t *tile = tiles + (size_t)(tile_index * COLI_APPLE8_MXFP4_TILE_BYTES);
            const uint8_t *source = tile + (size_t)tile_row * 16u;
            const uint64_t base_column = g * COLI_APPLE8_MXFP4_TILE_COLUMNS;
            uint64_t used_columns = columns - base_column;
            size_t used_bytes;
            uint8_t *destination;
            if (used_columns > COLI_APPLE8_MXFP4_TILE_COLUMNS)
                used_columns = COLI_APPLE8_MXFP4_TILE_COLUMNS;
            used_bytes = (size_t)(used_columns / 2u + (used_columns & 1u));
            destination = weights + (size_t)(o * canonical_row_bytes + g * 16u);
            memcpy(destination, source, used_bytes);
            if ((used_columns & 1u) && (destination[used_bytes - 1u] & 0xf0u))
                return fail(error, error_size,
                            "Apple8 detile found nonzero odd-K padding nibble");
            scales[(size_t)(o * groups + g)] = tile[128u + (size_t)tile_row];
        }
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
    return coli_package_read_range_ex(package, record, offset, destination, bytes,
                                      read_flags, error, error_size);
}

static int decoded_apple8_role(const unsigned char *resident,
                               size_t resident_bytes, uint16_t wanted_role,
                               const uint8_t **tiles, size_t *tile_bytes,
                               uint64_t *rows, uint64_t *columns,
                               char *error, size_t error_size) {
    if (!resident || resident_bytes < COLI_EXPERT_PREFIX_BYTES)
        return fail(error, error_size, "decoded Apple8 expert record is truncated");
    for (size_t i = 0; i < 3; ++i) {
        const unsigned char *d = resident + 64u + i * 128u;
        uint16_t role = rd16(d);
        uint16_t codec = rd16(d + 8);
        uint16_t source_layout = rd16(d + 12);
        uint64_t offset = rd64(d + 48);
        uint64_t stored = rd64(d + 56);
        uint64_t decoded = rd64(d + 64);
        if (role != wanted_role) continue;
        if (codec != COLI_CSF_CODEC_NONE ||
            source_layout != COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1 ||
            stored != decoded || decoded > SIZE_MAX ||
            offset > resident_bytes || stored > resident_bytes - (size_t)offset)
            return fail(error, error_size,
                        "decoded Apple8 role %u descriptor is invalid",
                        (unsigned)wanted_role);
        *rows = rd64(d + 16);
        *columns = rd64(d + 24);
        *tiles = resident + (size_t)offset;
        *tile_bytes = (size_t)decoded;
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
    size_t resident_bytes = 0, written = 0;
    unsigned char *resident = NULL;
    int rc = -1;
    struct Target {
        uint16_t role;
        uint8_t *weights;
        size_t weight_bytes;
        uint8_t *scales;
        size_t scale_bytes;
    } targets[3] = {
        {COLI_MXFP4_EXPERT_ROLE_GATE, buffers->gate_weights,
         layout->gate_weight_bytes, buffers->gate_scales, layout->gate_scale_bytes},
        {COLI_MXFP4_EXPERT_ROLE_UP, buffers->up_weights,
         layout->up_weight_bytes, buffers->up_scales, layout->up_scale_bytes},
        {COLI_MXFP4_EXPERT_ROLE_DOWN, buffers->down_weights,
         layout->down_weight_bytes, buffers->down_scales, layout->down_scale_bytes},
    };

    if (coli_package_expert_resident_bytes(package, record, &resident_u64,
                                           error, error_size) ||
        !u64_to_size(resident_u64, &resident_bytes) ||
        resident_bytes < COLI_EXPERT_PREFIX_BYTES)
        return fail(error, error_size, "Apple8 decoded expert size is invalid");
    resident = (unsigned char *)malloc(resident_bytes);
    if (!resident)
        return fail(error, error_size, "out of memory allocating decoded Apple8 expert");
    if (coli_package_decode_expert_record(package, record,
                                          resident, resident_bytes, &written,
                                          error, error_size) ||
        written != resident_bytes)
        goto done;

    for (size_t i = 0; i < 3; ++i) {
        const uint8_t *tiles = NULL;
        size_t tile_bytes = 0;
        uint64_t rows = 0, columns = 0;
        if (decoded_apple8_role(resident, resident_bytes, targets[i].role,
                                &tiles, &tile_bytes, &rows, &columns,
                                error, error_size) ||
            coli_mxfp4_apple8_detile_matrix(
                tiles, tile_bytes, rows, columns,
                targets[i].weights, targets[i].weight_bytes,
                targets[i].scales, targets[i].scale_bytes,
                error, error_size))
            goto done;
    }
    rc = 0;
done:
    free(resident);
    return rc;
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

    if (coli_package_expert_info(package, record, &info, error, error_size) ||
        coli_mxfp4_expert_validate_info(&info, hidden, intermediate, layout,
                                        error, error_size))
        return -1;

    if (check_buffer("gate weights", buffers->gate_weights,
                     buffers->gate_weight_capacity, layout->gate_weight_bytes,
                     error, error_size) ||
        check_buffer("gate scales", buffers->gate_scales,
                     buffers->gate_scale_capacity, layout->gate_scale_bytes,
                     error, error_size) ||
        check_buffer("up weights", buffers->up_weights,
                     buffers->up_weight_capacity, layout->up_weight_bytes,
                     error, error_size) ||
        check_buffer("up scales", buffers->up_scales,
                     buffers->up_scale_capacity, layout->up_scale_bytes,
                     error, error_size) ||
        check_buffer("down weights", buffers->down_weights,
                     buffers->down_weight_capacity, layout->down_weight_bytes,
                     error, error_size) ||
        check_buffer("down scales", buffers->down_scales,
                     buffers->down_scale_capacity, layout->down_scale_bytes,
                     error, error_size))
        return -1;

    if (layout->source_layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1) {
        (void)read_flags; /* synchronous CSF decode is the intentional PR3 path */
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

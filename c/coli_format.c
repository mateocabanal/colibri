#define coli_package_expert_info coli_package_expert_info_legacy
#define coli_package_validate_record coli_package_validate_record_legacy
#define coli_package_verify_all coli_package_verify_all_legacy
#include "coli_format_base.c"
#undef coli_package_expert_info
#undef coli_package_validate_record
#undef coli_package_verify_all

#include "apple8_contract.h"
#include "coli_target.h"

static int apple8_profile(const ColiPackage *p) {
    return p && coli_apple8_profile_is_v1(p->profile);
}

static int apple8_edge_padding_valid(const ColiPackage *p,
                                     const ColiRecordInfo *r,
                                     const ColiExpertMatrixInfo *m,
                                     char *error, size_t error_size) {
    uint64_t row_tiles, groups, ot, g;
    unsigned char tile[COLI_APPLE8_MXFP4_TILE_BYTES];
    const uint64_t row_rem = m->rows % COLI_APPLE8_MXFP4_TILE_ROWS;
    const uint64_t col_rem = m->columns % COLI_APPLE8_MXFP4_TILE_COLUMNS;
    row_tiles = m->rows / COLI_APPLE8_MXFP4_TILE_ROWS + (row_rem != 0);
    groups = m->columns / COLI_APPLE8_MXFP4_TILE_COLUMNS + (col_rem != 0);
    for (ot = 0; ot < row_tiles; ++ot) {
        for (g = 0; g < groups; ++g) {
            uint64_t tile_index, rel;
            unsigned rr;
            const int row_edge = row_rem && ot + 1 == row_tiles;
            const int col_edge = col_rem && g + 1 == groups;
            if (!row_edge && !col_edge) continue;
            if (checked_mul_u64(ot, groups, &tile_index) ||
                checked_add_u64(tile_index, g, &tile_index) ||
                checked_mul_u64(tile_index, COLI_APPLE8_MXFP4_TILE_BYTES, &rel) ||
                checked_add_u64(m->weight_offset, rel, &rel) ||
                coli_package_read_range(p, r, rel, tile, sizeof(tile), error, error_size))
                return -1;
            for (rr = 0; rr < COLI_APPLE8_MXFP4_TILE_ROWS; ++rr) {
                unsigned char *row = tile + rr * 16u;
                const int logical_row = !row_edge || rr < row_rem;
                if (!logical_row) {
                    if (!all_zero(row, 16u) || tile[128u + rr] != 0) {
                        csf_error(error, error_size,
                                  "Apple8 matrix edge row padding is nonzero");
                        return -1;
                    }
                    continue;
                }
                if (col_edge) {
                    size_t used = (size_t)((col_rem + 1u) / 2u);
                    if (used < 16u && !all_zero(row + used, 16u - used)) {
                        csf_error(error, error_size,
                                  "Apple8 matrix K padding bytes are nonzero");
                        return -1;
                    }
                    if ((col_rem & 1u) && (row[used - 1u] & 0xf0u)) {
                        csf_error(error, error_size,
                                  "Apple8 matrix odd-K padding nibble is nonzero");
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}

int coli_package_expert_info(const ColiPackage *p, const ColiRecordInfo *r,
                             ColiExpertInfo *out, char *error, size_t error_size) {
    unsigned char h[CSF_EXPERT_HEADER_BYTES + 3u * CSF_EXPERT_MATRIX_DESC_BYTES];
    Span spans[3];
    size_t span_count = 0;
    uint64_t logical_sum = 0, data_start;
    unsigned i;
    if (!apple8_profile(p))
        return coli_package_expert_info_legacy(p, r, out, error, error_size);
    if (!p || !r || !out || r->kind != COLI_CSF_REC_EXPERT) {
        csf_error(error, error_size, "expert_info requires an EXPERT record");
        return -1;
    }
    if (read_record_header(p, r, h, sizeof(h), error, error_size)) return -1;
    if (memcmp(h, k_expert_magic, 8) || rd16(h + 8) != 1 || rd16(h + 10) > 0 ||
        rd32(h + 12) != CSF_EXPERT_HEADER_BYTES || rd16(h + 24) != 3 || rd16(h + 26) ||
        rd32(h + 28) != CSF_EXPERT_MATRIX_DESC_BYTES ||
        rd64(h + 32) != CSF_EXPERT_HEADER_BYTES || rd64(h + 56)) {
        csf_error(error, error_size, "record %llu has invalid expert header",
                  (unsigned long long)r->record_id);
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->layer = rdi32(h + 16);
    out->expert = rdi32(h + 20);
    data_start = rd64(h + 40);
    out->logical_bytes = rd64(h + 48);
    if (out->layer != r->layer || out->expert != r->expert ||
        data_start < sizeof(h) || data_start % CSF_INTERNAL_ALIGNMENT ||
        data_start > r->stored_bytes) {
        csf_error(error, error_size, "expert identity/data offset mismatch");
        return -1;
    }
    for (i = 0; i < 3; ++i) {
        const unsigned char *d = h + CSF_EXPERT_HEADER_BYTES + i * CSF_EXPERT_MATRIX_DESC_BYTES;
        ColiExpertMatrixInfo *m = &out->matrices[i];
        uint64_t b, e, expected;
        m->role = rd16(d);
        m->math_format = rd16(d + 4);
        m->scale_format = rd16(d + 6);
        m->weight_codec = rd16(d + 8);
        m->scale_codec = rd16(d + 10);
        m->layout = rd16(d + 12);
        m->rows = rd64(d + 16);
        m->columns = rd64(d + 24);
        m->scale_block_rows = rd32(d + 32);
        m->scale_block_columns = rd32(d + 36);
        m->weight_codec_table_id = rd32(d + 40);
        m->scale_codec_table_id = rd32(d + 44);
        m->weight_offset = rd64(d + 48);
        m->weight_stored_bytes = rd64(d + 56);
        m->weight_decoded_bytes = rd64(d + 64);
        m->scale_offset = rd64(d + 72);
        m->scale_stored_bytes = rd64(d + 80);
        m->scale_decoded_bytes = rd64(d + 88);
        m->logical_crc32c = rd32(d + 96);
        m->group_size = rd32(d + 104);
        if (m->role != i + 1 || rd16(d + 2) || !matrix_reserved_zero(d) ||
            !coli_target_profile_accepts_layout(p->profile, m->layout) ||
            !coli_apple8_matrix_descriptor_valid(m, &expected)) {
            csf_error(error, error_size,
                      "Apple8 expert matrix %u violates MXFP4 tile8x32 descriptor contract", i);
            return -1;
        }
        if (matrix_span(m->weight_offset, m->weight_stored_bytes,
                        m->weight_codec, data_start, r->stored_bytes, &b, &e)) {
            csf_error(error, error_size, "Apple8 expert matrix %u combined span is invalid", i);
            return -1;
        }
        spans[span_count].begin = b;
        spans[span_count].end = e;
        spans[span_count++].shard = 0;
        if (checked_add_u64(logical_sum, expected, &logical_sum)) {
            csf_error(error, error_size, "Apple8 expert logical byte count overflow");
            return -1;
        }
    }
    qsort(spans, span_count, sizeof(*spans), span_cmp);
    for (i = 1; i < span_count; ++i) {
        if (spans[i].begin < spans[i - 1].end) {
            csf_error(error, error_size, "Apple8 expert matrix combined spans overlap");
            return -1;
        }
    }
    if (logical_sum != out->logical_bytes || logical_sum != r->decoded_bytes) {
        csf_error(error, error_size,
                  "Apple8 expert resident byte count disagrees with descriptor");
        return -1;
    }
    return 0;
}

int coli_package_validate_record(const ColiPackage *p, const ColiRecordInfo *r,
                                 int verify_stored_crc_flag,
                                 char *error, size_t error_size) {
    ColiExpertInfo info;
    unsigned i;
    if (!p || !r) {
        csf_error(error, error_size, "invalid package/record");
        return -1;
    }
    if (!apple8_profile(p) || r->kind != COLI_CSF_REC_EXPERT)
        return coli_package_validate_record_legacy(
            p, r, verify_stored_crc_flag, error, error_size);
    if (verify_stored_crc_flag && verify_record_crc(p, r, error, error_size)) return -1;
    if (coli_package_expert_info(p, r, &info, error, error_size)) return -1;
    for (i = 0; i < 3; ++i) {
        const ColiExpertMatrixInfo *m = &info.matrices[i];
        uint32_t crc;
        uint64_t source_offset;
        if (checked_add_u64(r->payload_offset, m->weight_offset, &source_offset) ||
            crc32c_fd(p->shards[r->shard_id].fd, source_offset,
                      m->weight_decoded_bytes, &crc, error, error_size))
            return -1;
        if (crc != m->logical_crc32c) {
            csf_error(error, error_size,
                      "Apple8 expert matrix %u logical CRC mismatch", i);
            return -1;
        }
        if (apple8_edge_padding_valid(p, r, m, error, error_size)) return -1;
    }
    return 0;
}

int coli_package_verify_all(const ColiPackage *p, char *error, size_t error_size) {
    size_t i;
    if (!p) {
        csf_error(error, error_size, "invalid package");
        return -1;
    }
    for (i = 0; i < p->record_count; ++i) {
        if (coli_package_validate_record(p, &p->records[i], 1, error, error_size))
            return -1;
    }
    return 0;
}

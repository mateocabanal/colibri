#define coli_package_expert_info coli_package_expert_info_legacy
#define coli_package_validate_record coli_package_validate_record_legacy
#define coli_package_verify_all coli_package_verify_all_legacy
#include "coli_format_base.c"
#undef coli_package_expert_info
#undef coli_package_validate_record
#undef coli_package_verify_all

#include "apple8_contract.h"
#include "coli_target.h"
#include "rans.h"

#define COLI_RANS_TABLE_BLOB_BYTES 160u
#define COLI_RANS_TABLE_VERSION 1u
#define COLI_RANS_TABLE_SCALE_BITS 14u
#define COLI_RANS_TABLE_STREAMS 256u
#define COLI_RANS_AUTO_MIN_SAVINGS_BPS 500u
#define COLI_RANS_AUTO_MIN_SAVINGS_BYTES 256u

static const unsigned char k_coli_rans_table_magic[8] = {
    'C','O','L','I','R','N','0','1'
};

static int apple8_profile(const ColiPackage *p) {
    return p && coli_apple8_profile_is_v1(p->profile);
}

static void wr16(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
}

static void wr32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
    p[2] = (unsigned char)((v >> 16) & 0xffu);
    p[3] = (unsigned char)((v >> 24) & 0xffu);
}

static void wr64(unsigned char *p, uint64_t v) {
    wr32(p, (uint32_t)(v & UINT32_MAX));
    wr32(p + 4, (uint32_t)(v >> 32));
}

static int align16_checked(uint64_t value, uint64_t *out) {
    if (value > UINT64_MAX - 15u) return -1;
    *out = (value + 15u) & ~(uint64_t)15u;
    return 0;
}

static int profile_accepts_expert(const ColiPackage *p,
                                  const ColiExpertInfo *info,
                                  char *error, size_t error_size) {
    unsigned i;
    if (!p || !info) return -1;
    for (i = 0; i < 3; ++i) {
        if (!coli_target_profile_accepts_layout(p->profile, info->matrices[i].layout)) {
            csf_error(error, error_size,
                      "expert matrix %u layout 0x%04x is not permitted by target profile %s",
                      i, info->matrices[i].layout,
                      p->profile ? p->profile : "<missing>");
            return -1;
        }
    }
    return 0;
}

static int apple8_edge_padding_bytes_valid(const unsigned char *decoded,
                                           size_t decoded_bytes,
                                           const ColiExpertMatrixInfo *m,
                                           char *error, size_t error_size) {
    uint64_t row_tiles, groups, ot, g;
    const uint64_t row_rem = m->rows % COLI_APPLE8_MXFP4_TILE_ROWS;
    const uint64_t col_rem = m->columns % COLI_APPLE8_MXFP4_TILE_COLUMNS;
    row_tiles = m->rows / COLI_APPLE8_MXFP4_TILE_ROWS + (row_rem != 0);
    groups = m->columns / COLI_APPLE8_MXFP4_TILE_COLUMNS + (col_rem != 0);
    for (ot = 0; ot < row_tiles; ++ot) {
        for (g = 0; g < groups; ++g) {
            uint64_t tile_index, rel;
            const unsigned char *tile;
            unsigned rr;
            const int row_edge = row_rem && ot + 1 == row_tiles;
            const int col_edge = col_rem && g + 1 == groups;
            if (!row_edge && !col_edge) continue;
            if (checked_mul_u64(ot, groups, &tile_index) ||
                checked_add_u64(tile_index, g, &tile_index) ||
                checked_mul_u64(tile_index, COLI_APPLE8_MXFP4_TILE_BYTES, &rel) ||
                rel > decoded_bytes ||
                decoded_bytes - (size_t)rel < COLI_APPLE8_MXFP4_TILE_BYTES) {
                csf_error(error, error_size,
                          "Apple8 decoded matrix tile lies outside payload");
                return -1;
            }
            tile = decoded + (size_t)rel;
            for (rr = 0; rr < COLI_APPLE8_MXFP4_TILE_ROWS; ++rr) {
                const unsigned char *row = tile + rr * 16u;
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

static int apple8_rans_table_init(const ColiPackage *p, uint32_t table_id,
                                  rans_table *table, uint16_t **slots_out,
                                  char *error, size_t error_size) {
    const ColiCsfCodecTable *meta;
    const unsigned char *blob;
    uint64_t region_offset, region_bytes, data_end;
    uint32_t freq[RANS_ALPHABET], start[RANS_ALPHABET];
    uint16_t *slots = NULL;
    uint32_t cursor = 0;
    unsigned s;
    rans_err rc;
    if (table) memset(table, 0, sizeof(*table));
    if (slots_out) *slots_out = NULL;
    if (!p || !table || !slots_out || !table_id) {
        csf_error(error, error_size, "invalid rANS codec table request");
        return -1;
    }
    meta = codec_table(p, table_id);
    if (!meta || meta->codec != COLI_CSF_CODEC_RANS256_G0_NIBBLE ||
        meta->shard_id != -1 || meta->data_bytes != COLI_RANS_TABLE_BLOB_BYTES) {
        csf_error(error, error_size,
                  "rANS codec table %u is absent or not artifact-wide nibble g0",
                  table_id);
        return -1;
    }
    region_offset = rd64(p->manifest + 168);
    region_bytes = rd64(p->manifest + 176);
    if (checked_add_u64(meta->data_offset, meta->data_bytes, &data_end) ||
        data_end > region_bytes ||
        checked_add_u64(region_offset, meta->data_offset, &data_end) ||
        data_end > p->manifest_bytes ||
        p->manifest_bytes - (size_t)data_end < meta->data_bytes) {
        csf_error(error, error_size, "rANS codec table %u data lies outside manifest", table_id);
        return -1;
    }
    blob = p->manifest + (size_t)data_end;
    if (memcmp(blob, k_coli_rans_table_magic, sizeof(k_coli_rans_table_magic)) ||
        rd16(blob + 8) != COLI_RANS_TABLE_VERSION ||
        rd16(blob + 10) != COLI_RANS_TABLE_SCALE_BITS ||
        rd32(blob + 12) != COLI_RANS_TABLE_STREAMS ||
        rd32(blob + 16) != COLI_RANS_AUTO_MIN_SAVINGS_BPS ||
        rd32(blob + 20) != COLI_RANS_AUTO_MIN_SAVINGS_BYTES ||
        !all_zero(blob + 24, 8)) {
        csf_error(error, error_size, "rANS codec table %u has invalid frozen header", table_id);
        return -1;
    }
    for (s = 0; s < RANS_ALPHABET; ++s) {
        freq[s] = rd32(blob + 32 + s * 4u);
        start[s] = rd32(blob + 96 + s * 4u);
    }
    slots = (uint16_t *)malloc((size_t)(1u << COLI_RANS_TABLE_SCALE_BITS) * sizeof(*slots));
    if (!slots) {
        csf_error(error, error_size, "out of memory allocating rANS slot table");
        return -1;
    }
    for (s = 0; s < RANS_ALPHABET; ++s) {
        uint32_t k;
        if (start[s] != cursor || freq[s] > (1u << COLI_RANS_TABLE_SCALE_BITS) - cursor) {
            free(slots);
            csf_error(error, error_size, "rANS codec table %u freq/start mismatch", table_id);
            return -1;
        }
        for (k = 0; k < freq[s]; ++k) slots[cursor + k] = (uint16_t)s;
        cursor += freq[s];
    }
    if (cursor != (1u << COLI_RANS_TABLE_SCALE_BITS)) {
        free(slots);
        csf_error(error, error_size, "rANS codec table %u frequency sum mismatch", table_id);
        return -1;
    }
    rc = rans_table_init(table, COLI_RANS_TABLE_SCALE_BITS, freq, start, slots);
    if (rc != RANS_OK) {
        free(slots);
        csf_error(error, error_size, "rANS codec table %u rejected: %s",
                  table_id, rans_err_name(rc));
        return -1;
    }
    *slots_out = slots;
    return 0;
}

static void apple8_rans_table_free(rans_table *table, uint16_t *slots) {
    rans_table_free(table);
    free(slots);
}

static uint64_t apple8_stream_symbol_count(uint64_t total, uint32_t stream) {
    if ((uint64_t)stream >= total) return 0;
    return (total - 1u - stream) / COLI_RANS_TABLE_STREAMS + 1u;
}

static int apple8_decode_matrix(const ColiPackage *p, const ColiRecordInfo *r,
                                const ColiExpertMatrixInfo *m,
                                unsigned char *destination, size_t destination_bytes,
                                char *error, size_t error_size) {
    uint64_t stored = m->weight_stored_bytes;
    uint64_t decoded = m->weight_decoded_bytes;
    if (decoded > SIZE_MAX || destination_bytes < (size_t)decoded) {
        csf_error(error, error_size, "Apple8 decoded matrix destination is too small");
        return -1;
    }
    if (m->weight_codec == COLI_CSF_CODEC_NONE) {
        if (stored != decoded ||
            coli_package_read_range(p, r, m->weight_offset, destination,
                                    (size_t)decoded, error, error_size))
            return -1;
        return 0;
    }
    if (m->weight_codec == COLI_CSF_CODEC_RANS256_G0_NIBBLE) {
        unsigned char *record = NULL, *scratch = NULL;
        rans_record parsed;
        rans_table table;
        uint16_t *slots = NULL;
        uint64_t expected_symbols;
        uint64_t max_stream_symbols;
        uint32_t stream;
        rans_err rc;
        if (stored > SIZE_MAX ||
            checked_mul_u64(decoded, 2u, &expected_symbols)) {
            csf_error(error, error_size, "Apple8 rANS matrix size exceeds host limits");
            return -1;
        }
        record = (unsigned char *)calloc((size_t)stored + RANS_SLACK, 1);
        if (!record) {
            csf_error(error, error_size, "out of memory reading Apple8 rANS matrix");
            return -1;
        }
        if (coli_package_read_range(p, r, m->weight_offset, record,
                                    (size_t)stored, error, error_size)) {
            free(record);
            return -1;
        }
        {
            unsigned char slack[RANS_SLACK];
            uint64_t slack_offset;
            if (checked_add_u64(m->weight_offset, stored, &slack_offset) ||
                coli_package_read_range(p, r, slack_offset, slack, sizeof(slack),
                                        error, error_size)) {
                free(record);
                return -1;
            }
            if (!all_zero(slack, sizeof(slack))) {
                free(record);
                csf_error(error, error_size, "Apple8 rANS readable slack is nonzero");
                return -1;
            }
        }
        rc = rans_record_parse(record, stored, COLI_RANS_TABLE_STREAMS, &parsed);
        if (rc != RANS_OK) {
            free(record);
            csf_error(error, error_size, "Apple8 rANS record rejected: %s", rans_err_name(rc));
            return -1;
        }
        if (parsed.n_symbols != expected_symbols || parsed.packed_bytes != decoded) {
            free(record);
            csf_error(error, error_size,
                      "Apple8 rANS decoded length disagrees with matrix descriptor");
            return -1;
        }
        if (apple8_rans_table_init(p, m->weight_codec_table_id, &table, &slots,
                                   error, error_size)) {
            free(record);
            return -1;
        }
        max_stream_symbols = expected_symbols / COLI_RANS_TABLE_STREAMS + 1u;
        if (max_stream_symbols > SIZE_MAX) {
            apple8_rans_table_free(&table, slots);
            free(record);
            csf_error(error, error_size, "Apple8 rANS stream scratch exceeds host limits");
            return -1;
        }
        scratch = (unsigned char *)malloc((size_t)max_stream_symbols);
        if (!scratch) {
            apple8_rans_table_free(&table, slots);
            free(record);
            csf_error(error, error_size, "out of memory decoding Apple8 rANS stream");
            return -1;
        }
        memset(destination, 0, (size_t)decoded);
        for (stream = 0; stream < COLI_RANS_TABLE_STREAMS; ++stream) {
            uint64_t count = apple8_stream_symbol_count(expected_symbols, stream);
            uint32_t begin = parsed.stream_offsets[stream];
            uint32_t end = parsed.stream_offsets[stream + 1u];
            uint64_t k;
            rc = rans_decode_stream_checked(parsed.payload + begin,
                                            (size_t)(end - begin),
                                            (size_t)count,
                                            table.freq, table.start,
                                            table.slot_to_symbol,
                                            table.scale_bits,
                                            scratch);
            if (rc != RANS_OK) {
                free(scratch);
                apple8_rans_table_free(&table, slots);
                free(record);
                csf_error(error, error_size,
                          "Apple8 rANS stream %u rejected: %s", stream, rans_err_name(rc));
                return -1;
            }
            for (k = 0; k < count; ++k) {
                uint64_t logical = (uint64_t)stream + k * COLI_RANS_TABLE_STREAMS;
                size_t byte_index = (size_t)(logical >> 1);
                unsigned char symbol = scratch[k];
                if ((logical & 1u) == 0)
                    destination[byte_index] =
                        (unsigned char)((destination[byte_index] & 0xf0u) | symbol);
                else
                    destination[byte_index] =
                        (unsigned char)((destination[byte_index] & 0x0fu) | (symbol << 4));
            }
        }
        free(scratch);
        apple8_rans_table_free(&table, slots);
        free(record);
        return 0;
    }
    csf_error(error, error_size, "unsupported Apple8 matrix codec 0x%04x", m->weight_codec);
    return -1;
}

int coli_package_expert_info(const ColiPackage *p, const ColiRecordInfo *r,
                             ColiExpertInfo *out, char *error, size_t error_size) {
    unsigned char h[CSF_EXPERT_HEADER_BYTES + 3u * CSF_EXPERT_MATRIX_DESC_BYTES];
    Span spans[3];
    size_t span_count = 0;
    uint64_t logical_sum = 0, data_start;
    unsigned i;
    if (!apple8_profile(p)) {
        int rc = coli_package_expert_info_legacy(p, r, out, error, error_size);
        if (rc) return rc;
        return profile_accepts_expert(p, out, error, error_size);
    }
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
            validate_codec_ref(p, m->weight_codec, m->weight_codec_table_id,
                               r->shard_id, error, error_size) ||
            !coli_apple8_matrix_descriptor_valid(m, &expected)) {
            if (!error || !error_size || !error[0])
                csf_error(error, error_size,
                          "Apple8 expert matrix %u violates MXFP4 tile8x32 descriptor contract", i);
            return -1;
        }
        if (matrix_span(m->weight_offset, m->weight_stored_bytes,
                        m->weight_codec, data_start, r->stored_bytes, &b, &e)) {
            csf_error(error, error_size, "Apple8 expert matrix %u combined span is invalid", i);
            return -1;
        }
        if (m->weight_codec != COLI_CSF_CODEC_NONE &&
            zero_slack(p, r, m->weight_offset + m->weight_stored_bytes,
                       error, error_size))
            return -1;
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

int coli_package_expert_resident_bytes(const ColiPackage *p,
                                       const ColiRecordInfo *r,
                                       uint64_t *out_bytes,
                                       char *error, size_t error_size) {
    ColiExpertInfo info;
    uint64_t cursor;
    unsigned i;
    if (!p || !r || !out_bytes || r->kind != COLI_CSF_REC_EXPERT) {
        csf_error(error, error_size, "resident byte query requires an EXPERT record");
        return -1;
    }
    if (!apple8_profile(p)) {
        *out_bytes = r->stored_bytes;
        return 0;
    }
    if (coli_package_expert_info(p, r, &info, error, error_size)) return -1;
    cursor = CSF_EXPERT_HEADER_BYTES + 3u * CSF_EXPERT_MATRIX_DESC_BYTES;
    for (i = 0; i < 3; ++i) {
        if (align16_checked(cursor, &cursor) ||
            checked_add_u64(cursor, info.matrices[i].weight_decoded_bytes, &cursor)) {
            csf_error(error, error_size, "Apple8 resident expert size overflows u64");
            return -1;
        }
    }
    *out_bytes = cursor;
    return 0;
}

int coli_package_decode_expert_record(const ColiPackage *p,
                                      const ColiRecordInfo *r,
                                      void *destination,
                                      size_t destination_bytes,
                                      size_t *written_bytes,
                                      char *error, size_t error_size) {
    ColiExpertInfo info;
    unsigned char prefix[CSF_EXPERT_HEADER_BYTES + 3u * CSF_EXPERT_MATRIX_DESC_BYTES];
    unsigned char *out = (unsigned char *)destination;
    uint64_t required, cursor;
    unsigned i;
    if (written_bytes) *written_bytes = 0;
    if (!p || !r || !destination || r->kind != COLI_CSF_REC_EXPERT) {
        csf_error(error, error_size, "decode_expert_record requires an EXPERT record");
        return -1;
    }
    if (!apple8_profile(p)) {
        if (r->stored_bytes > SIZE_MAX || destination_bytes < (size_t)r->stored_bytes) {
            csf_error(error, error_size, "expert resident destination is too small");
            return -1;
        }
        if (coli_package_read_record(p, r, destination, destination_bytes, error, error_size))
            return -1;
        if (written_bytes) *written_bytes = (size_t)r->stored_bytes;
        return 0;
    }
    /* Stored corruption is rejected before any decompression work. */
    if (verify_record_crc(p, r, error, error_size)) return -1;
    if (coli_package_expert_info(p, r, &info, error, error_size) ||
        coli_package_expert_resident_bytes(p, r, &required, error, error_size))
        return -1;
    if (required > SIZE_MAX || destination_bytes < (size_t)required) {
        csf_error(error, error_size,
                  "expert resident destination is too small: need %llu bytes",
                  (unsigned long long)required);
        return -1;
    }
    if (coli_package_read_range(p, r, 0, prefix, sizeof(prefix), error, error_size))
        return -1;
    memset(out, 0, (size_t)required);
    memcpy(out, prefix, sizeof(prefix));
    cursor = sizeof(prefix);
    for (i = 0; i < 3; ++i) {
        ColiExpertMatrixInfo *m = &info.matrices[i];
        unsigned char *d = out + CSF_EXPERT_HEADER_BYTES + i * CSF_EXPERT_MATRIX_DESC_BYTES;
        size_t decoded_bytes;
        if (align16_checked(cursor, &cursor) ||
            m->weight_decoded_bytes > SIZE_MAX ||
            cursor > required || m->weight_decoded_bytes > required - cursor) {
            csf_error(error, error_size, "Apple8 resident matrix placement overflows");
            return -1;
        }
        decoded_bytes = (size_t)m->weight_decoded_bytes;
        if (apple8_decode_matrix(p, r, m, out + (size_t)cursor,
                                 decoded_bytes, error, error_size))
            return -1;
        if (coli_crc32c(out + (size_t)cursor, decoded_bytes) != m->logical_crc32c) {
            csf_error(error, error_size,
                      "Apple8 expert matrix %u decoded CRC mismatch", i);
            return -1;
        }
        if (apple8_edge_padding_bytes_valid(out + (size_t)cursor, decoded_bytes,
                                            m, error, error_size))
            return -1;
        wr16(d + 8, COLI_CSF_CODEC_NONE);
        wr32(d + 40, 0);
        wr64(d + 48, cursor);
        wr64(d + 56, m->weight_decoded_bytes);
        wr64(d + 64, m->weight_decoded_bytes);
        cursor += m->weight_decoded_bytes;
    }
    if (cursor != required) {
        csf_error(error, error_size, "Apple8 resident expert byte count mismatch");
        return -1;
    }
    if (written_bytes) *written_bytes = (size_t)required;
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
    if (!apple8_profile(p) || r->kind != COLI_CSF_REC_EXPERT) {
        int rc = coli_package_validate_record_legacy(
            p, r, verify_stored_crc_flag, error, error_size);
        if (rc || r->kind != COLI_CSF_REC_EXPERT) return rc;
        return coli_package_expert_info(p, r, &info, error, error_size);
    }
    if (verify_stored_crc_flag && verify_record_crc(p, r, error, error_size)) return -1;
    if (coli_package_expert_info(p, r, &info, error, error_size)) return -1;
    for (i = 0; i < 3; ++i) {
        const ColiExpertMatrixInfo *m = &info.matrices[i];
        unsigned char *decoded;
        if (m->weight_decoded_bytes > SIZE_MAX) {
            csf_error(error, error_size, "Apple8 matrix decoded size exceeds host limits");
            return -1;
        }
        decoded = (unsigned char *)malloc((size_t)m->weight_decoded_bytes);
        if (!decoded) {
            csf_error(error, error_size, "out of memory validating Apple8 matrix");
            return -1;
        }
        if (apple8_decode_matrix(p, r, m, decoded,
                                 (size_t)m->weight_decoded_bytes,
                                 error, error_size)) {
            free(decoded);
            return -1;
        }
        if (coli_crc32c(decoded, (size_t)m->weight_decoded_bytes) != m->logical_crc32c) {
            free(decoded);
            csf_error(error, error_size,
                      "Apple8 expert matrix %u logical CRC mismatch", i);
            return -1;
        }
        if (apple8_edge_padding_bytes_valid(decoded,
                                            (size_t)m->weight_decoded_bytes,
                                            m, error, error_size)) {
            free(decoded);
            return -1;
        }
        free(decoded);
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

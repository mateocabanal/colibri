#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "coli_format.h"
#include "compat.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CSF_MANIFEST_HEADER_BYTES 256u
#define CSF_SHARD_DESC_BYTES 64u
#define CSF_RECORD_DESC_BYTES 96u
#define CSF_STRING_DESC_BYTES 16u
#define CSF_CODEC_TABLE_DESC_BYTES 64u
#define CSF_DATA_HEADER_BYTES 128u
#define CSF_TENSOR_HEADER_BYTES 128u
#define CSF_EXPERT_HEADER_BYTES 64u
#define CSF_EXPERT_MATRIX_DESC_BYTES 128u
#define CSF_INTERNAL_ALIGNMENT 16u
#define CSF_RANS_READABLE_SLACK 64u

#define CSF_MAX_MANIFEST_BYTES (1ULL << 30)
#define CSF_MAX_RECORDS (16ULL << 20)
#define CSF_MAX_STRINGS (16u << 20)
#define CSF_MAX_SHARDS (1u << 16)
#define CSF_MAX_CODEC_TABLES (1u << 20)
#define CSF_MAX_DECODED_RECORD_BYTES (1ULL << 48)
#define CSF_MAX_CODEC_TABLE_BLOB (64ULL << 20)

#define CSF_MANIFEST_F_SOURCE_FINGERPRINT_VALID (1u << 0)
#define CSF_RECORD_REQUIRED_FLAG_MASK 0xff00u
#define CSF_MANIFEST_REQUIRED_FLAG_MASK 0xffff0000u
#define CSF_NO_STRING UINT32_MAX

static const unsigned char k_manifest_magic[8] = {
    0x43, 0x4f, 0x4c, 0x49, 0x0d, 0x0a, 0x1a, 0x0a
};
static const unsigned char k_data_magic[8] = {
    0x43, 0x4f, 0x4c, 0x49, 0x44, 0x41, 0x54, 0x00
};
static const unsigned char k_tensor_magic[8] = {
    'C','O','L','I','T','E','N','S'
};
static const unsigned char k_expert_magic[8] = {
    'C','O','L','I','E','X','P','T'
};

typedef struct ColiCsfShard {
    uint32_t id;
    uint64_t bytes;
    uint32_t header_crc32c;
    const char *file_name;
    char *path;
    int fd;
    int direct_fd;
} ColiCsfShard;

typedef struct ColiCsfCodecTable {
    uint32_t id;
    uint16_t codec;
    int32_t shard_id;
    uint64_t data_offset;
    uint64_t data_bytes;
    uint32_t crc32c;
} ColiCsfCodecTable;

typedef struct U64Index {
    uint64_t *keys;
    uint32_t *values; /* record/table index + 1; 0 = empty */
    size_t capacity;
} U64Index;

typedef struct NameIndex {
    const char **keys;
    uint32_t *values; /* record index + 1 */
    size_t capacity;
} NameIndex;

typedef struct Span {
    uint32_t shard;
    uint64_t begin;
    uint64_t end;
    size_t record_index;
} Span;

typedef struct ManifestRegion {
    uint64_t offset;
    uint64_t bytes;
    const char *name;
} ManifestRegion;

struct ColiPackage {
    char *root;
    unsigned char *manifest;
    size_t manifest_bytes;
    uint16_t version_minor;
    uint32_t flags;
    uint32_t record_alignment;
    uint8_t source_fingerprint[32];
    char **strings;
    uint32_t string_count;
    const char *profile;
    const char *compiler;

    ColiCsfShard *shards;
    uint32_t shard_count;
    ColiRecordInfo *records;
    size_t record_count;
    ColiCsfCodecTable *codec_tables;
    uint32_t codec_table_count;

    U64Index id_index;
    U64Index expert_index;
    U64Index layer_index;
    U64Index codec_index;
    NameIndex name_index;

    ColiCsfChecksumPolicy checksum_policy;
};

static void csf_error(char *error, size_t error_size, const char *fmt, ...) {
    if (!error || !error_size) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
    error[error_size - 1] = '\0';
}

static uint16_t rd16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t rd32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int32_t rdi32(const unsigned char *p) {
    uint32_t u = rd32(p);
    int32_t s;
    memcpy(&s, &u, sizeof(s));
    return s;
}

static uint64_t rd64(const unsigned char *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static int all_zero(const unsigned char *p, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (p[i]) return 0;
    return 1;
}

static int checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (UINT64_MAX - a < b) return -1;
    *out = a + b;
    return 0;
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a && b > UINT64_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int is_power_of_two_u32(uint32_t x) {
    return x && !(x & (x - 1u));
}

static uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uint64_t hash_u64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

static uint64_t hash_name(const char *s) {
    uint64_t h = UINT64_C(1469598103934665603);
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static size_t index_capacity(size_t count) {
    size_t cap = 8;
    if (count > (SIZE_MAX / 2)) return 0;
    while (cap < count * 2 + 1) {
        if (cap > SIZE_MAX / 2) return 0;
        cap <<= 1;
    }
    return cap;
}

static int u64_index_init(U64Index *idx, size_t count) {
    size_t cap = index_capacity(count);
    if (!cap) return -1;
    idx->keys = (uint64_t *)calloc(cap, sizeof(*idx->keys));
    idx->values = (uint32_t *)calloc(cap, sizeof(*idx->values));
    if (!idx->keys || !idx->values) {
        free(idx->keys); free(idx->values);
        memset(idx, 0, sizeof(*idx));
        return -1;
    }
    idx->capacity = cap;
    return 0;
}

static void u64_index_free(U64Index *idx) {
    free(idx->keys); free(idx->values);
    memset(idx, 0, sizeof(*idx));
}

static int u64_index_put(U64Index *idx, uint64_t key, uint32_t value) {
    size_t mask = idx->capacity - 1;
    size_t pos = (size_t)hash_u64(key) & mask;
    size_t start = pos;
    do {
        if (!idx->values[pos]) {
            idx->keys[pos] = key;
            idx->values[pos] = value;
            return 0;
        }
        if (idx->keys[pos] == key) return 1;
        pos = (pos + 1) & mask;
    } while (pos != start);
    return -1;
}

static uint32_t u64_index_get(const U64Index *idx, uint64_t key) {
    size_t mask, pos, start;
    if (!idx || !idx->capacity) return 0;
    mask = idx->capacity - 1;
    pos = (size_t)hash_u64(key) & mask;
    start = pos;
    do {
        if (!idx->values[pos]) return 0;
        if (idx->keys[pos] == key) return idx->values[pos];
        pos = (pos + 1) & mask;
    } while (pos != start);
    return 0;
}

static int name_index_init(NameIndex *idx, size_t count) {
    size_t cap = index_capacity(count);
    if (!cap) return -1;
    idx->keys = (const char **)calloc(cap, sizeof(*idx->keys));
    idx->values = (uint32_t *)calloc(cap, sizeof(*idx->values));
    if (!idx->keys || !idx->values) {
        free(idx->keys); free(idx->values);
        memset(idx, 0, sizeof(*idx));
        return -1;
    }
    idx->capacity = cap;
    return 0;
}

static void name_index_free(NameIndex *idx) {
    free(idx->keys); free(idx->values);
    memset(idx, 0, sizeof(*idx));
}

static int name_index_put(NameIndex *idx, const char *key, uint32_t value) {
    size_t mask = idx->capacity - 1;
    size_t pos = (size_t)hash_name(key) & mask;
    size_t start = pos;
    do {
        if (!idx->values[pos]) {
            idx->keys[pos] = key;
            idx->values[pos] = value;
            return 0;
        }
        if (!strcmp(idx->keys[pos], key)) return 1;
        pos = (pos + 1) & mask;
    } while (pos != start);
    return -1;
}

static uint32_t name_index_get(const NameIndex *idx, const char *key) {
    size_t mask, pos, start;
    if (!idx || !idx->capacity || !key) return 0;
    mask = idx->capacity - 1;
    pos = (size_t)hash_name(key) & mask;
    start = pos;
    do {
        if (!idx->values[pos]) return 0;
        if (!strcmp(idx->keys[pos], key)) return idx->values[pos];
        pos = (pos + 1) & mask;
    } while (pos != start);
    return 0;
}

static int utf8_valid(const unsigned char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned c = s[i++];
        unsigned need, min2 = 0x80, max2 = 0xbf;
        if (c < 0x80) continue;
        if (c >= 0xc2 && c <= 0xdf) need = 1;
        else if (c >= 0xe0 && c <= 0xef) {
            need = 2;
            if (c == 0xe0) min2 = 0xa0;
            if (c == 0xed) max2 = 0x9f;
        } else if (c >= 0xf0 && c <= 0xf4) {
            need = 3;
            if (c == 0xf0) min2 = 0x90;
            if (c == 0xf4) max2 = 0x8f;
        } else return 0;
        if (i + need > n) return 0;
        if (s[i] < min2 || s[i] > max2) return 0;
        ++i;
        while (--need) {
            if (s[i] < 0x80 || s[i] > 0xbf) return 0;
            ++i;
        }
    }
    return 1;
}

static int portable_filename(const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    if (!s || !*s || !strcmp(s, ".") || !strcmp(s, "..")) return 0;
    for (; *p; ++p) {
        if ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') ||
            (*p >= '0' && *p <= '9') || *p == '.' || *p == '_' || *p == '-')
            continue;
        return 0;
    }
    return 1;
}

static char *path_join(const char *root, const char *leaf) {
    size_t a = strlen(root), b = strlen(leaf);
    int slash = a && root[a - 1] != '/' && root[a - 1] != '\\';
    char *p;
    if (a > SIZE_MAX - b - 2) return NULL;
    p = (char *)malloc(a + b + (size_t)slash + 1);
    if (!p) return NULL;
    memcpy(p, root, a);
    if (slash) p[a++] = '/';
    memcpy(p + a, leaf, b + 1);
    return p;
}

static int pread_full(int fd, void *dst, size_t bytes, uint64_t offset,
                      char *error, size_t error_size) {
    unsigned char *p = (unsigned char *)dst;
    size_t done = 0;
    while (done < bytes) {
        size_t chunk = bytes - done;
        ssize_t got;
        if (chunk > (size_t)0x40000000u) chunk = (size_t)0x40000000u;
        if (offset > (uint64_t)INT64_MAX - done) {
            csf_error(error, error_size, "pread offset exceeds signed 64-bit range");
            return -1;
        }
        do {
            got = pread(fd, p + done, chunk, (off_t)(offset + done));
        } while (got < 0 && errno == EINTR);
        if (got < 0) {
            csf_error(error, error_size, "pread failed at offset %llu: %s",
                      (unsigned long long)(offset + done), strerror(errno));
            return -1;
        }
        if (!got) {
            csf_error(error, error_size, "short read at offset %llu",
                      (unsigned long long)(offset + done));
            return -1;
        }
        done += (size_t)got;
    }
    return 0;
}

static uint32_t crc32c_extend(uint32_t state, const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    size_t i;
    uint32_t crc = state;
    for (i = 0; i < bytes; ++i) {
        unsigned k;
        crc ^= p[i];
        for (k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ (0x82f63b78u & (uint32_t)-(int32_t)(crc & 1u));
    }
    return crc;
}

uint32_t coli_crc32c(const void *data, size_t bytes) {
    return crc32c_extend(0xffffffffu, data, bytes) ^ 0xffffffffu;
}

static int crc32c_fd(int fd, uint64_t offset, uint64_t bytes, uint32_t *out,
                     char *error, size_t error_size) {
    unsigned char buffer[64 * 1024];
    uint32_t crc = 0xffffffffu;
    uint64_t done = 0;
    while (done < bytes) {
        size_t n = sizeof(buffer);
        if ((uint64_t)n > bytes - done) n = (size_t)(bytes - done);
        if (pread_full(fd, buffer, n, offset + done, error, error_size)) return -1;
        crc = crc32c_extend(crc, buffer, n);
        done += n;
    }
    *out = crc ^ 0xffffffffu;
    return 0;
}

static int known_codec(uint16_t codec) {
    return codec == COLI_CSF_CODEC_NONE ||
           codec == COLI_CSF_CODEC_RANS256_G0_NIBBLE ||
           codec == COLI_CSF_CODEC_RANS256_G0_U8;
}

static int known_math(uint16_t v) {
    return (v >= COLI_CSF_MATH_F32 && v <= COLI_CSF_MATH_BOOL) ||
           v == COLI_CSF_MATH_NONE || v == COLI_CSF_MATH_FP8_E4M3FN ||
           v == COLI_CSF_MATH_FP8_E5M2 || v == COLI_CSF_MATH_MXFP4_E2M1 ||
           v == COLI_CSF_MATH_INT4_PACKED || v == COLI_CSF_MATH_INT4_GROUPED ||
           v == COLI_CSF_MATH_MIXED;
}

static int known_scale(uint16_t v) {
    return v <= COLI_CSF_SCALE_UE8M0 || v == COLI_CSF_SCALE_MIXED;
}

static int known_layout(uint16_t v) {
    return v == COLI_CSF_LAYOUT_CANONICAL || v == COLI_CSF_LAYOUT_ROWS16 ||
           (v >= 0x0100 && v <= 0x03ff) || v == COLI_CSF_LAYOUT_MIXED;
}

static int manifest_region(ManifestRegion *out, uint64_t offset, uint64_t bytes,
                           const char *name, uint64_t file_bytes,
                           char *error, size_t error_size) {
    uint64_t end;
    if (!bytes) {
        if (offset) {
            csf_error(error, error_size, "%s has zero bytes but nonzero offset", name);
            return -1;
        }
        out->offset = out->bytes = 0; out->name = name;
        return 0;
    }
    if (!offset || offset % CSF_INTERNAL_ALIGNMENT) {
        csf_error(error, error_size, "%s is not 16-byte aligned", name);
        return -1;
    }
    if (checked_add_u64(offset, bytes, &end) || end > file_bytes || offset < CSF_MANIFEST_HEADER_BYTES) {
        csf_error(error, error_size, "%s lies outside manifest", name);
        return -1;
    }
    out->offset = offset; out->bytes = bytes; out->name = name;
    return 0;
}

static int regions_nonoverlap(ManifestRegion *r, size_t n,
                              char *error, size_t error_size) {
    size_t i, j;
    for (i = 0; i < n; ++i) {
        uint64_t a_end;
        if (!r[i].bytes) continue;
        a_end = r[i].offset + r[i].bytes;
        for (j = i + 1; j < n; ++j) {
            uint64_t b_end;
            if (!r[j].bytes) continue;
            b_end = r[j].offset + r[j].bytes;
            if (r[i].offset < b_end && r[j].offset < a_end) {
                csf_error(error, error_size, "manifest regions %s and %s overlap",
                          r[i].name, r[j].name);
                return -1;
            }
        }
    }
    return 0;
}

static int span_cmp(const void *aa, const void *bb) {
    const Span *a = (const Span *)aa, *b = (const Span *)bb;
    if (a->shard != b->shard) return a->shard < b->shard ? -1 : 1;
    if (a->begin != b->begin) return a->begin < b->begin ? -1 : 1;
    if (a->end != b->end) return a->end < b->end ? -1 : 1;
    return 0;
}

static const ColiCsfCodecTable *codec_table(const ColiPackage *p, uint32_t id) {
    uint32_t v;
    if (!id) return NULL;
    v = u64_index_get(&p->codec_index, id);
    if (!v) return NULL;
    return &p->codec_tables[v - 1];
}

static int validate_codec_ref(const ColiPackage *p, uint16_t codec, uint32_t table_id,
                              uint32_t shard_id, char *error, size_t error_size) {
    const ColiCsfCodecTable *t;
    if (codec == COLI_CSF_CODEC_NONE) {
        if (table_id) {
            csf_error(error, error_size, "codec NONE has nonzero table id %u", table_id);
            return -1;
        }
        return 0;
    }
    if (!known_codec(codec)) {
        csf_error(error, error_size, "unsupported codec 0x%04x", codec);
        return -1;
    }
    if (!table_id || !(t = codec_table(p, table_id))) {
        csf_error(error, error_size, "codec 0x%04x references missing table %u", codec, table_id);
        return -1;
    }
    if (t->codec != codec) {
        csf_error(error, error_size, "codec table %u has mismatched codec", table_id);
        return -1;
    }
    if (t->shard_id >= 0 && (uint32_t)t->shard_id != shard_id) {
        csf_error(error, error_size, "shard-local codec table %u used by another shard", table_id);
        return -1;
    }
    return 0;
}

static int parse_strings(ColiPackage *p, const unsigned char *base, uint64_t bytes,
                         uint32_t count, char *error, size_t error_size) {
    uint64_t desc_bytes;
    uint32_t i;
    if (count > CSF_MAX_STRINGS || checked_mul_u64(count, CSF_STRING_DESC_BYTES, &desc_bytes) ||
        desc_bytes > bytes) {
        csf_error(error, error_size, "invalid string table size/count");
        return -1;
    }
    p->strings = (char **)calloc(count ? count : 1, sizeof(*p->strings));
    if (!p->strings) { csf_error(error, error_size, "out of memory for string table"); return -1; }
    p->string_count = count;
    for (i = 0; i < count; ++i) {
        const unsigned char *d = base + (size_t)i * CSF_STRING_DESC_BYTES;
        uint64_t off = rd64(d), end;
        uint32_t len = rd32(d + 8), flags = rd32(d + 12);
        const unsigned char *s;
        if (flags) { csf_error(error, error_size, "string %u has nonzero flags", i); return -1; }
        if (off < desc_bytes || checked_add_u64(off, len, &end) || end > bytes) {
            csf_error(error, error_size, "string %u lies outside string table", i); return -1;
        }
        s = base + (size_t)off;
        if (memchr(s, 0, len)) { csf_error(error, error_size, "string %u contains NUL", i); return -1; }
        if (!utf8_valid(s, len)) { csf_error(error, error_size, "string %u is invalid UTF-8", i); return -1; }
        p->strings[i] = (char *)malloc((size_t)len + 1);
        if (!p->strings[i]) { csf_error(error, error_size, "out of memory for string %u", i); return -1; }
        memcpy(p->strings[i], s, len); p->strings[i][len] = '\0';
    }
    return 0;
}

static int parse_codec_tables(ColiPackage *p, const unsigned char *base, uint64_t bytes,
                              uint32_t count, char *error, size_t error_size) {
    uint64_t desc_bytes;
    Span *spans = NULL;
    uint32_t i;
    if (!count) return 0;
    if (count > CSF_MAX_CODEC_TABLES || checked_mul_u64(count, CSF_CODEC_TABLE_DESC_BYTES, &desc_bytes) ||
        desc_bytes > bytes) {
        csf_error(error, error_size, "invalid codec table size/count"); return -1;
    }
    p->codec_tables = (ColiCsfCodecTable *)calloc(count, sizeof(*p->codec_tables));
    spans = (Span *)calloc(count, sizeof(*spans));
    if (!p->codec_tables || !spans || u64_index_init(&p->codec_index, count)) {
        free(spans); csf_error(error, error_size, "out of memory for codec tables"); return -1;
    }
    p->codec_table_count = count;
    for (i = 0; i < count; ++i) {
        const unsigned char *d = base + (size_t)i * CSF_CODEC_TABLE_DESC_BYTES;
        ColiCsfCodecTable *t = &p->codec_tables[i];
        uint64_t end;
        t->id = rd32(d);
        t->codec = rd16(d + 4);
        t->shard_id = rdi32(d + 8);
        t->data_offset = rd64(d + 16);
        t->data_bytes = rd64(d + 24);
        t->crc32c = rd32(d + 32);
        if (!t->id || u64_index_put(&p->codec_index, t->id, i + 1) != 0) {
            csf_error(error, error_size, "zero/duplicate codec table id %u", t->id); free(spans); return -1;
        }
        if (!known_codec(t->codec) || t->codec == COLI_CSF_CODEC_NONE) {
            csf_error(error, error_size, "codec table %u uses unsupported codec", t->id); free(spans); return -1;
        }
        if (rd16(d + 6) || rd32(d + 12) || rd32(d + 36) || !all_zero(d + 40, 24)) {
            csf_error(error, error_size, "codec table %u has nonzero reserved fields", t->id); free(spans); return -1;
        }
        if (t->shard_id < -1 || (t->shard_id >= 0 && (uint32_t)t->shard_id >= p->shard_count)) {
            csf_error(error, error_size, "codec table %u has invalid shard id", t->id); free(spans); return -1;
        }
        if (!t->data_bytes || t->data_bytes > CSF_MAX_CODEC_TABLE_BLOB ||
            t->data_offset < desc_bytes || t->data_offset % CSF_INTERNAL_ALIGNMENT ||
            checked_add_u64(t->data_offset, t->data_bytes, &end) || end > bytes) {
            csf_error(error, error_size, "codec table %u data span is invalid", t->id); free(spans); return -1;
        }
        if (coli_crc32c(base + (size_t)t->data_offset, (size_t)t->data_bytes) != t->crc32c) {
            csf_error(error, error_size, "codec table %u CRC mismatch", t->id); free(spans); return -1;
        }
        spans[i].begin = t->data_offset; spans[i].end = end; spans[i].shard = 0;
    }
    qsort(spans, count, sizeof(*spans), span_cmp);
    for (i = 1; i < count; ++i) if (spans[i].begin < spans[i - 1].end) {
        csf_error(error, error_size, "codec table blobs overlap"); free(spans); return -1;
    }
    free(spans);
    return 0;
}

static int validate_shard_header(ColiPackage *p, ColiCsfShard *s,
                                 char *error, size_t error_size) {
    unsigned char h[CSF_DATA_HEADER_BYTES], tmp[CSF_DATA_HEADER_BYTES];
    uint32_t crc;
    if (pread_full(s->fd, h, sizeof(h), 0, error, error_size)) return -1;
    if (memcmp(h, k_data_magic, 8)) { csf_error(error, error_size, "shard %u: bad magic", s->id); return -1; }
    if (rd16(h + 8) != COLI_CSF_VERSION_MAJOR || rd16(h + 10) > COLI_CSF_VERSION_MINOR ||
        rd32(h + 12) != CSF_DATA_HEADER_BYTES) {
        csf_error(error, error_size, "shard %u: unsupported header version/size", s->id); return -1;
    }
    if (rd32(h + 16) || rd32(h + 20) != s->id || rd32(h + 24) != p->record_alignment || rd32(h + 28)) {
        csf_error(error, error_size, "shard %u: invalid flags/id/alignment", s->id); return -1;
    }
    if (rd64(h + 32) != s->bytes || memcmp(h + 40, p->source_fingerprint, 32) || !all_zero(h + 76, 52)) {
        csf_error(error, error_size, "shard %u: size/fingerprint/reserved mismatch", s->id); return -1;
    }
    memcpy(tmp, h, sizeof(tmp)); memset(tmp + 72, 0, 4);
    crc = coli_crc32c(tmp, sizeof(tmp));
    if (crc != rd32(h + 72) || crc != s->header_crc32c) {
        csf_error(error, error_size, "shard %u: header CRC mismatch", s->id); return -1;
    }
    return 0;
}

static int open_shards(ColiPackage *p, const unsigned char *base, uint32_t count,
                       char *error, size_t error_size) {
    uint32_t i;
    p->shards = (ColiCsfShard *)calloc(count ? count : 1, sizeof(*p->shards));
    if (!p->shards) { csf_error(error, error_size, "out of memory for shards"); return -1; }
    p->shard_count = count;
    for (i = 0; i < count; ++i) { p->shards[i].fd = -1; p->shards[i].direct_fd = -1; }
    for (i = 0; i < count; ++i) {
        const unsigned char *d = base + (size_t)i * CSF_SHARD_DESC_BYTES;
        uint32_t id = rd32(d), name_id = rd32(d + 8);
        ColiCsfShard *s;
        struct stat st;
        if (id >= count || p->shards[id].file_name) {
            csf_error(error, error_size, "duplicate/out-of-range shard id %u", id); return -1;
        }
        s = &p->shards[id]; s->id = id;
        if (rd32(d + 4) || rd32(d + 12) || rd32(d + 28) || !all_zero(d + 32, 32)) {
            csf_error(error, error_size, "shard %u has nonzero reserved fields", id); return -1;
        }
        if (name_id >= p->string_count || !portable_filename(p->strings[name_id])) {
            csf_error(error, error_size, "shard %u has unsafe filename", id); return -1;
        }
        s->file_name = p->strings[name_id];
        s->bytes = rd64(d + 16); s->header_crc32c = rd32(d + 24);
        if (s->bytes < CSF_DATA_HEADER_BYTES) { csf_error(error, error_size, "shard %u is too small", id); return -1; }
        s->path = path_join(p->root, s->file_name);
        if (!s->path) { csf_error(error, error_size, "out of memory for shard path"); return -1; }
        s->fd = open(s->path, COMPAT_O_RDONLY);
        if (s->fd < 0) { csf_error(error, error_size, "cannot open shard %s: %s", s->file_name, strerror(errno)); return -1; }
        if (fstat(s->fd, &st) || st.st_size < 0 || (uint64_t)st.st_size != s->bytes) {
            csf_error(error, error_size, "shard %s size mismatch", s->file_name); return -1;
        }
#ifdef O_DIRECT
        s->direct_fd = open(s->path, COMPAT_O_RDONLY | O_DIRECT);
#elif defined(__APPLE__) || defined(_WIN32)
        s->direct_fd = compat_open_direct(s->path);
#endif
        if (validate_shard_header(p, s, error, error_size)) return -1;
    }
    return 0;
}

static uint64_t expert_key(int32_t layer, int32_t expert) {
    return ((uint64_t)(uint32_t)layer << 32) | (uint32_t)expert;
}

static int record_generic_valid(ColiPackage *p, ColiRecordInfo *r,
                                uint32_t name_id, char *error, size_t error_size) {
    uint64_t end, min_payload;
    int optional = !!(r->flags & COLI_CSF_RECORD_F_OPTIONAL);
    if (!r->record_id) { csf_error(error, error_size, "record id 0 is invalid"); return -1; }
    if (r->flags & CSF_RECORD_REQUIRED_FLAG_MASK) {
        csf_error(error, error_size, "record %llu uses unknown required flags",
                  (unsigned long long)r->record_id); return -1;
    }
    if (r->shard_id >= p->shard_count) { csf_error(error, error_size, "record %llu has invalid shard", (unsigned long long)r->record_id); return -1; }
    if (name_id != CSF_NO_STRING) {
        if (name_id >= p->string_count) { csf_error(error, error_size, "record %llu has invalid name id", (unsigned long long)r->record_id); return -1; }
        r->name = p->strings[name_id];
    }
    if (r->payload_offset % p->record_alignment ||
        checked_add_u64(r->payload_offset, r->stored_bytes, &end) ||
        end > p->shards[r->shard_id].bytes) {
        csf_error(error, error_size, "record %llu lies outside/alignment of shard", (unsigned long long)r->record_id); return -1;
    }
    min_payload = align_up_u64(CSF_DATA_HEADER_BYTES, p->record_alignment);
    if (r->payload_offset < min_payload || r->decoded_bytes > CSF_MAX_DECODED_RECORD_BYTES) {
        csf_error(error, error_size, "record %llu has invalid payload/decoded size", (unsigned long long)r->record_id); return -1;
    }
    if (r->kind != COLI_CSF_REC_TENSOR && r->kind != COLI_CSF_REC_EXPERT &&
        r->kind != COLI_CSF_REC_LAYER_PACK_RESERVED && r->kind != COLI_CSF_REC_BLOB) {
        if (!optional) { csf_error(error, error_size, "record %llu has unknown required kind 0x%04x", (unsigned long long)r->record_id, r->kind); return -1; }
        return 0;
    }
    if (r->kind == COLI_CSF_REC_LAYER_PACK_RESERVED) {
        if (!optional) { csf_error(error, error_size, "required LAYER_PACK is unsupported in CSF v1.0"); return -1; }
        return 0;
    }
    if (!known_codec(r->codec) || !known_math(r->math_format) ||
        !known_scale(r->scale_format) || !known_layout(r->layout)) {
        csf_error(error, error_size, "record %llu uses unsupported enum", (unsigned long long)r->record_id); return -1;
    }
    if (!strcmp(p->profile, "portable-v1") &&
        r->kind != COLI_CSF_REC_EXPERT && r->layout != COLI_CSF_LAYOUT_CANONICAL) {
        csf_error(error, error_size, "portable-v1 record %llu is not canonical", (unsigned long long)r->record_id); return -1;
    }
    switch (r->kind) {
    case COLI_CSF_REC_TENSOR:
        if (!r->name || r->expert != -1 || r->math_format == COLI_CSF_MATH_MIXED ||
            r->scale_format != COLI_CSF_SCALE_NONE || r->layout == COLI_CSF_LAYOUT_MIXED ||
            r->stored_bytes < CSF_TENSOR_HEADER_BYTES) {
            csf_error(error, error_size, "record %llu violates TENSOR invariants", (unsigned long long)r->record_id); return -1;
        }
        if (r->layer < -1) { csf_error(error, error_size, "tensor has invalid layer"); return -1; }
        break;
    case COLI_CSF_REC_EXPERT:
        if (r->codec != COLI_CSF_CODEC_NONE || r->codec_table_id ||
            r->math_format != COLI_CSF_MATH_MIXED || r->scale_format != COLI_CSF_SCALE_MIXED ||
            r->layout != COLI_CSF_LAYOUT_MIXED || r->layer < 0 || r->expert < 0 ||
            r->stored_bytes < CSF_EXPERT_HEADER_BYTES + 3u * CSF_EXPERT_MATRIX_DESC_BYTES) {
            csf_error(error, error_size, "record %llu violates EXPERT invariants", (unsigned long long)r->record_id); return -1;
        }
        break;
    case COLI_CSF_REC_BLOB:
        if (r->math_format != COLI_CSF_MATH_NONE || r->scale_format != COLI_CSF_SCALE_NONE ||
            r->layout != COLI_CSF_LAYOUT_CANONICAL || r->layer != -1 || r->expert != -1 ||
            r->codec != COLI_CSF_CODEC_NONE || r->codec_table_id) {
            csf_error(error, error_size, "record %llu violates BLOB invariants", (unsigned long long)r->record_id); return -1;
        }
        break;
    default: break;
    }
    if (r->kind != COLI_CSF_REC_EXPERT &&
        validate_codec_ref(p, r->codec, r->codec_table_id, r->shard_id, error, error_size)) return -1;
    return 0;
}

static int parse_records(ColiPackage *p, const unsigned char *base, uint64_t count,
                         char *error, size_t error_size) {
    Span *spans;
    size_t i;
    if (count > CSF_MAX_RECORDS || count > SIZE_MAX / sizeof(*p->records)) {
        csf_error(error, error_size, "record count exceeds parser limit"); return -1;
    }
    p->record_count = (size_t)count;
    p->records = (ColiRecordInfo *)calloc(p->record_count ? p->record_count : 1, sizeof(*p->records));
    spans = (Span *)calloc(p->record_count ? p->record_count : 1, sizeof(*spans));
    if (!p->records || !spans || u64_index_init(&p->id_index, p->record_count) ||
        u64_index_init(&p->expert_index, p->record_count) ||
        u64_index_init(&p->layer_index, p->record_count) ||
        name_index_init(&p->name_index, p->record_count)) {
        free(spans); csf_error(error, error_size, "out of memory for record indexes"); return -1;
    }
    for (i = 0; i < p->record_count; ++i) {
        const unsigned char *d = base + i * CSF_RECORD_DESC_BYTES;
        ColiRecordInfo *r = &p->records[i];
        uint32_t name_id = rd32(d + 24);
        int put;
        r->record_id = rd64(d);
        r->kind = rd16(d + 8); r->codec = rd16(d + 10);
        r->math_format = rd16(d + 12); r->scale_format = rd16(d + 14);
        r->layout = rd16(d + 16); r->flags = rd16(d + 18);
        r->shard_id = rd32(d + 20); r->layer = rdi32(d + 28); r->expert = rdi32(d + 32);
        r->payload_offset = rd64(d + 40); r->stored_bytes = rd64(d + 48); r->decoded_bytes = rd64(d + 56);
        r->stored_crc32c = rd32(d + 64); r->logical_crc32c = rd32(d + 68); r->codec_table_id = rd32(d + 72);
        if (rd32(d + 36) || rd32(d + 76) || !all_zero(d + 80, 16)) {
            csf_error(error, error_size, "record %llu has nonzero reserved fields", (unsigned long long)r->record_id); free(spans); return -1;
        }
        if (record_generic_valid(p, r, name_id, error, error_size)) { free(spans); return -1; }
        put = u64_index_put(&p->id_index, r->record_id, (uint32_t)i + 1);
        if (put != 0) { csf_error(error, error_size, "duplicate record id %llu", (unsigned long long)r->record_id); free(spans); return -1; }
        if (r->name && name_index_put(&p->name_index, r->name, (uint32_t)i + 1) != 0) {
            csf_error(error, error_size, "duplicate record name %s", r->name); free(spans); return -1;
        }
        if (r->kind == COLI_CSF_REC_EXPERT) {
            if (u64_index_put(&p->expert_index, expert_key(r->layer, r->expert), (uint32_t)i + 1) != 0) {
                csf_error(error, error_size, "duplicate expert (%d,%d)", r->layer, r->expert); free(spans); return -1;
            }
        } else if (r->kind == COLI_CSF_REC_LAYER_PACK_RESERVED && r->layer >= 0) {
            if (u64_index_put(&p->layer_index, (uint32_t)r->layer, (uint32_t)i + 1) != 0) {
                csf_error(error, error_size, "duplicate layer pack %d", r->layer); free(spans); return -1;
            }
        }
        spans[i].shard = r->shard_id; spans[i].begin = r->payload_offset;
        spans[i].end = r->payload_offset + r->stored_bytes; spans[i].record_index = i;
    }
    qsort(spans, p->record_count, sizeof(*spans), span_cmp);
    for (i = 1; i < p->record_count; ++i) {
        if (spans[i].shard == spans[i - 1].shard && spans[i].begin < spans[i - 1].end) {
            csf_error(error, error_size, "records %llu and %llu overlap in shard %u",
                      (unsigned long long)p->records[spans[i - 1].record_index].record_id,
                      (unsigned long long)p->records[spans[i].record_index].record_id,
                      spans[i].shard);
            free(spans); return -1;
        }
    }
    free(spans);
    return 0;
}

static int verify_record_crc(const ColiPackage *p, const ColiRecordInfo *r,
                             char *error, size_t error_size) {
    uint32_t got;
    if (crc32c_fd(p->shards[r->shard_id].fd, r->payload_offset, r->stored_bytes,
                  &got, error, error_size)) return -1;
    if (got != r->stored_crc32c) {
        csf_error(error, error_size, "record %llu stored CRC mismatch: expected %08x got %08x",
                  (unsigned long long)r->record_id, r->stored_crc32c, got);
        return -1;
    }
    return 0;
}

static int read_record_header(const ColiPackage *p, const ColiRecordInfo *r,
                              void *dst, size_t bytes, char *error, size_t error_size) {
    if (r->stored_bytes < bytes) { csf_error(error, error_size, "record %llu is truncated", (unsigned long long)r->record_id); return -1; }
    return pread_full(p->shards[r->shard_id].fd, dst, bytes, r->payload_offset, error, error_size);
}

static int zero_slack(const ColiPackage *p, const ColiRecordInfo *r, uint64_t off,
                      char *error, size_t error_size) {
    unsigned char slack[CSF_RANS_READABLE_SLACK];
    uint64_t end;
    if (checked_add_u64(off, CSF_RANS_READABLE_SLACK, &end) || end > r->stored_bytes) {
        csf_error(error, error_size, "record %llu lacks rANS readable slack", (unsigned long long)r->record_id); return -1;
    }
    if (coli_package_read_range(p, r, off, slack, sizeof(slack), error, error_size)) return -1;
    if (!all_zero(slack, sizeof(slack))) {
        csf_error(error, error_size, "record %llu has nonzero rANS slack", (unsigned long long)r->record_id); return -1;
    }
    return 0;
}

int coli_package_tensor_info(const ColiPackage *p, const ColiRecordInfo *r,
                             ColiTensorInfo *out, char *error, size_t error_size) {
    unsigned char h[CSF_TENSOR_HEADER_BYTES];
    uint64_t end;
    unsigned i;
    if (!p || !r || !out || r->kind != COLI_CSF_REC_TENSOR) {
        csf_error(error, error_size, "tensor_info requires a TENSOR record"); return -1;
    }
    if (read_record_header(p, r, h, sizeof(h), error, error_size)) return -1;
    if (memcmp(h, k_tensor_magic, 8) || rd16(h + 8) != 1 || rd16(h + 10) > 0 || rd32(h + 12) != CSF_TENSOR_HEADER_BYTES) {
        csf_error(error, error_size, "record %llu has invalid tensor header", (unsigned long long)r->record_id); return -1;
    }
    memset(out, 0, sizeof(*out));
    out->rank = rd16(h + 16);
    if (out->rank > COLI_CSF_MAX_RANK || rd16(h + 18) || rd32(h + 124)) {
        csf_error(error, error_size, "record %llu has invalid tensor rank/flags/reserved", (unsigned long long)r->record_id); return -1;
    }
    out->scale_block_rows = rd32(h + 20); out->scale_block_columns = rd32(h + 24); out->group_size = rd32(h + 28);
    for (i = 0; i < COLI_CSF_MAX_RANK; ++i) {
        out->dims[i] = rd64(h + 32 + i * 8);
        if (i < out->rank && out->dims[i] == 0) { csf_error(error, error_size, "tensor has zero dimension"); return -1; }
        if (i >= out->rank && out->dims[i] != 0) { csf_error(error, error_size, "tensor has nonzero dimension beyond rank"); return -1; }
    }
    out->data_offset = rd64(h + 96); out->data_stored_bytes = rd64(h + 104); out->data_decoded_bytes = rd64(h + 112); out->logical_crc32c = rd32(h + 120);
    if (out->data_offset < CSF_TENSOR_HEADER_BYTES || out->data_offset % CSF_INTERNAL_ALIGNMENT ||
        checked_add_u64(out->data_offset, out->data_stored_bytes, &end) || end > r->stored_bytes ||
        out->data_decoded_bytes != r->decoded_bytes) {
        csf_error(error, error_size, "tensor data span/decoded length is invalid"); return -1;
    }
    if ((r->flags & COLI_CSF_RECORD_F_HAS_LOGICAL_CRC32C) ? out->logical_crc32c != r->logical_crc32c : out->logical_crc32c != 0) {
        csf_error(error, error_size, "tensor logical CRC field disagrees with record descriptor"); return -1;
    }
    if (r->codec == COLI_CSF_CODEC_NONE) {
        if (out->data_stored_bytes != out->data_decoded_bytes) {
            csf_error(error, error_size, "uncompressed tensor stored/decoded bytes differ"); return -1;
        }
    } else {
        if (zero_slack(p, r, end, error, error_size)) return -1;
    }
    return 0;
}

static int matrix_reserved_zero(const unsigned char *d) {
    return rd16(d + 14) == 0 && rd32(d + 100) == 0 && rd32(d + 108) == 0 && all_zero(d + 112, 16);
}

static int matrix_span(uint64_t off, uint64_t stored, uint16_t codec,
                       uint64_t data_start, uint64_t record_bytes,
                       uint64_t *begin, uint64_t *end) {
    uint64_t e;
    if (!off || off < data_start || off % CSF_INTERNAL_ALIGNMENT || checked_add_u64(off, stored, &e)) return -1;
    if (codec != COLI_CSF_CODEC_NONE && checked_add_u64(e, CSF_RANS_READABLE_SLACK, &e)) return -1;
    if (e > record_bytes) return -1;
    *begin = off; *end = e; return 0;
}

int coli_package_expert_info(const ColiPackage *p, const ColiRecordInfo *r,
                             ColiExpertInfo *out, char *error, size_t error_size) {
    unsigned char h[CSF_EXPERT_HEADER_BYTES + 3u * CSF_EXPERT_MATRIX_DESC_BYTES];
    Span spans[6];
    size_t span_count = 0;
    uint64_t logical_sum = 0, data_start;
    unsigned i;
    if (!p || !r || !out || r->kind != COLI_CSF_REC_EXPERT) {
        csf_error(error, error_size, "expert_info requires an EXPERT record"); return -1;
    }
    if (read_record_header(p, r, h, sizeof(h), error, error_size)) return -1;
    if (memcmp(h, k_expert_magic, 8) || rd16(h + 8) != 1 || rd16(h + 10) > 0 ||
        rd32(h + 12) != CSF_EXPERT_HEADER_BYTES || rd16(h + 24) != 3 || rd16(h + 26) ||
        rd32(h + 28) != CSF_EXPERT_MATRIX_DESC_BYTES || rd64(h + 32) != CSF_EXPERT_HEADER_BYTES || rd64(h + 56)) {
        csf_error(error, error_size, "record %llu has invalid expert header", (unsigned long long)r->record_id); return -1;
    }
    memset(out, 0, sizeof(*out));
    out->layer = rdi32(h + 16); out->expert = rdi32(h + 20); data_start = rd64(h + 40); out->logical_bytes = rd64(h + 48);
    if (out->layer != r->layer || out->expert != r->expert || data_start < sizeof(h) || data_start % CSF_INTERNAL_ALIGNMENT || data_start > r->stored_bytes) {
        csf_error(error, error_size, "expert identity/data offset mismatch"); return -1;
    }
    for (i = 0; i < 3; ++i) {
        const unsigned char *d = h + CSF_EXPERT_HEADER_BYTES + i * CSF_EXPERT_MATRIX_DESC_BYTES;
        ColiExpertMatrixInfo *m = &out->matrices[i];
        uint64_t b, e, add;
        m->role = rd16(d); m->math_format = rd16(d + 4); m->scale_format = rd16(d + 6);
        m->weight_codec = rd16(d + 8); m->scale_codec = rd16(d + 10); m->layout = rd16(d + 12);
        m->rows = rd64(d + 16); m->columns = rd64(d + 24);
        m->scale_block_rows = rd32(d + 32); m->scale_block_columns = rd32(d + 36);
        m->weight_codec_table_id = rd32(d + 40); m->scale_codec_table_id = rd32(d + 44);
        m->weight_offset = rd64(d + 48); m->weight_stored_bytes = rd64(d + 56); m->weight_decoded_bytes = rd64(d + 64);
        m->scale_offset = rd64(d + 72); m->scale_stored_bytes = rd64(d + 80); m->scale_decoded_bytes = rd64(d + 88);
        m->logical_crc32c = rd32(d + 96); m->group_size = rd32(d + 104);
        if (m->role != i + 1 || rd16(d + 2) || !matrix_reserved_zero(d) || !m->rows || !m->columns ||
            !known_math(m->math_format) || m->math_format == COLI_CSF_MATH_NONE || m->math_format == COLI_CSF_MATH_MIXED ||
            !known_scale(m->scale_format) || m->scale_format == COLI_CSF_SCALE_MIXED ||
            !known_codec(m->weight_codec) || !known_codec(m->scale_codec) ||
            !known_layout(m->layout) || m->layout == COLI_CSF_LAYOUT_MIXED) {
            csf_error(error, error_size, "expert matrix %u has invalid enum/framing", i); return -1;
        }
        if (!strcmp(p->profile, "portable-v1") && m->layout != COLI_CSF_LAYOUT_CANONICAL) {
            csf_error(error, error_size, "portable-v1 expert matrix %u is not canonical", i); return -1;
        }
        if (validate_codec_ref(p, m->weight_codec, m->weight_codec_table_id, r->shard_id, error, error_size)) return -1;
        if (matrix_span(m->weight_offset, m->weight_stored_bytes, m->weight_codec, data_start, r->stored_bytes, &b, &e)) {
            csf_error(error, error_size, "expert matrix %u weight span is invalid", i); return -1;
        }
        spans[span_count].begin = b; spans[span_count].end = e; spans[span_count++].shard = 0;
        if (m->weight_codec != COLI_CSF_CODEC_NONE && zero_slack(p, r, m->weight_offset + m->weight_stored_bytes, error, error_size)) return -1;
        if (m->scale_format == COLI_CSF_SCALE_NONE) {
            if (m->scale_codec != COLI_CSF_CODEC_NONE || m->scale_codec_table_id || m->scale_offset ||
                m->scale_stored_bytes || m->scale_decoded_bytes || m->scale_block_rows || m->scale_block_columns) {
                csf_error(error, error_size, "expert matrix %u absent scale fields are nonzero", i); return -1;
            }
        } else {
            if (validate_codec_ref(p, m->scale_codec, m->scale_codec_table_id, r->shard_id, error, error_size)) return -1;
            if (matrix_span(m->scale_offset, m->scale_stored_bytes, m->scale_codec, data_start, r->stored_bytes, &b, &e)) {
                csf_error(error, error_size, "expert matrix %u scale span is invalid", i); return -1;
            }
            spans[span_count].begin = b; spans[span_count].end = e; spans[span_count++].shard = 0;
            if (m->scale_codec != COLI_CSF_CODEC_NONE && zero_slack(p, r, m->scale_offset + m->scale_stored_bytes, error, error_size)) return -1;
        }
        if (m->math_format == COLI_CSF_MATH_MXFP4_E2M1) {
            uint64_t wbpr = (m->columns + 1) / 2, sbpr = (m->columns + 31) / 32, expect_w, expect_s;
            if (m->scale_format != COLI_CSF_SCALE_UE8M0 || m->scale_block_rows != 1 || m->scale_block_columns != 32 || m->group_size != 0 ||
                checked_mul_u64(m->rows, wbpr, &expect_w) || checked_mul_u64(m->rows, sbpr, &expect_s) ||
                m->weight_decoded_bytes != expect_w || m->scale_decoded_bytes != expect_s) {
                csf_error(error, error_size, "expert matrix %u violates canonical MXFP4 geometry", i); return -1;
            }
        }
        if (m->weight_codec == COLI_CSF_CODEC_NONE && m->weight_stored_bytes != m->weight_decoded_bytes) {
            csf_error(error, error_size, "expert matrix %u uncompressed weight size mismatch", i); return -1;
        }
        if (m->scale_codec == COLI_CSF_CODEC_NONE && m->scale_stored_bytes != m->scale_decoded_bytes) {
            csf_error(error, error_size, "expert matrix %u uncompressed scale size mismatch", i); return -1;
        }
        if (checked_add_u64(m->weight_decoded_bytes, m->scale_decoded_bytes, &add) || checked_add_u64(logical_sum, add, &logical_sum)) {
            csf_error(error, error_size, "expert logical byte count overflow"); return -1;
        }
    }
    qsort(spans, span_count, sizeof(*spans), span_cmp);
    for (i = 1; i < span_count; ++i) if (spans[i].begin < spans[i - 1].end) {
        csf_error(error, error_size, "expert matrix subranges overlap"); return -1;
    }
    if (logical_sum != out->logical_bytes || logical_sum != r->decoded_bytes) {
        csf_error(error, error_size, "expert logical byte count disagrees with descriptor"); return -1;
    }
    return 0;
}

int coli_package_validate_record(const ColiPackage *p, const ColiRecordInfo *r,
                                 int verify_stored_crc_flag, char *error, size_t error_size) {
    if (!p || !r) { csf_error(error, error_size, "invalid package/record"); return -1; }
    if (verify_stored_crc_flag && verify_record_crc(p, r, error, error_size)) return -1;
    if (r->kind == COLI_CSF_REC_TENSOR) {
        ColiTensorInfo ti;
        return coli_package_tensor_info(p, r, &ti, error, error_size);
    }
    if (r->kind == COLI_CSF_REC_EXPERT) {
        ColiExpertInfo ei;
        return coli_package_expert_info(p, r, &ei, error, error_size);
    }
    return 0;
}

static int parse_manifest(ColiPackage *p, char *error, size_t error_size) {
    const unsigned char *h = p->manifest;
    uint32_t string_count, shard_count, codec_count, profile_id, compiler_id;
    uint64_t record_count, expected;
    ManifestRegion regions[5];
    unsigned char *tmp;
    uint32_t want_crc, got_crc;
    if (p->manifest_bytes < CSF_MANIFEST_HEADER_BYTES || memcmp(h, k_manifest_magic, 8)) {
        csf_error(error, error_size, "bad/truncated CSF manifest magic/header"); return -1;
    }
    if (rd16(h + 8) != COLI_CSF_VERSION_MAJOR) { csf_error(error, error_size, "unsupported CSF major %u", rd16(h + 8)); return -1; }
    if (rd16(h + 10) > COLI_CSF_VERSION_MINOR) { csf_error(error, error_size, "unsupported CSF minor %u", rd16(h + 10)); return -1; }
    p->version_minor = rd16(h + 10);
    if (rd32(h + 12) != CSF_MANIFEST_HEADER_BYTES || rd32(h + 20) != 0x01020304u) {
        csf_error(error, error_size, "invalid manifest header size/byte-order tag"); return -1;
    }
    p->flags = rd32(h + 16);
    if (p->flags & CSF_MANIFEST_REQUIRED_FLAG_MASK) { csf_error(error, error_size, "unsupported required manifest feature flags"); return -1; }
    p->record_alignment = rd32(h + 24);
    if (!is_power_of_two_u32(p->record_alignment) || p->record_alignment < 4096u || p->record_alignment > 1048576u) {
        csf_error(error, error_size, "invalid record alignment %u", p->record_alignment); return -1;
    }
    string_count = rd32(h + 28); record_count = rd64(h + 32); shard_count = rd32(h + 40); codec_count = rd32(h + 160);
    if (shard_count > CSF_MAX_SHARDS || record_count > CSF_MAX_RECORDS || string_count > CSF_MAX_STRINGS || codec_count > CSF_MAX_CODEC_TABLES) {
        csf_error(error, error_size, "manifest count exceeds parser limit"); return -1;
    }
    if (rd32(h + 44) || rd32(h + 156) || rd32(h + 164) || !all_zero(h + 184, 72)) {
        csf_error(error, error_size, "manifest has nonzero v1.0 reserved fields"); return -1;
    }
    memcpy(p->source_fingerprint, h + 112, 32);
    if (!(p->flags & CSF_MANIFEST_F_SOURCE_FINGERPRINT_VALID) && !all_zero(p->source_fingerprint, 32)) {
        csf_error(error, error_size, "source fingerprint bytes set without valid flag"); return -1;
    }
    if (checked_mul_u64(shard_count, CSF_SHARD_DESC_BYTES, &expected) || expected != rd64(h + 56)) {
        csf_error(error, error_size, "shard table byte count mismatch"); return -1;
    }
    if (checked_mul_u64(record_count, CSF_RECORD_DESC_BYTES, &expected) || expected != rd64(h + 72)) {
        csf_error(error, error_size, "record table byte count mismatch"); return -1;
    }
    if ((codec_count == 0) != (rd64(h + 176) == 0)) { csf_error(error, error_size, "codec table empty-region mismatch"); return -1; }
    if (manifest_region(&regions[0], rd64(h + 48), rd64(h + 56), "shard table", p->manifest_bytes, error, error_size) ||
        manifest_region(&regions[1], rd64(h + 64), rd64(h + 72), "record table", p->manifest_bytes, error, error_size) ||
        manifest_region(&regions[2], rd64(h + 80), rd64(h + 88), "string table", p->manifest_bytes, error, error_size) ||
        manifest_region(&regions[3], rd64(h + 96), rd64(h + 104), "metadata", p->manifest_bytes, error, error_size) ||
        manifest_region(&regions[4], rd64(h + 168), rd64(h + 176), "codec table", p->manifest_bytes, error, error_size) ||
        regions_nonoverlap(regions, 5, error, error_size)) return -1;
    /* Metadata is intentionally opaque/reserved in v1.0: only bounds/CRC cover it. */
    want_crc = rd32(h + 144);
    tmp = (unsigned char *)malloc(p->manifest_bytes);
    if (!tmp) { csf_error(error, error_size, "out of memory verifying manifest CRC"); return -1; }
    memcpy(tmp, p->manifest, p->manifest_bytes); memset(tmp + 144, 0, 4);
    got_crc = coli_crc32c(tmp, p->manifest_bytes); free(tmp);
    if (got_crc != want_crc) { csf_error(error, error_size, "manifest CRC mismatch: expected %08x got %08x", want_crc, got_crc); return -1; }
    if (parse_strings(p, h + (size_t)regions[2].offset, regions[2].bytes, string_count, error, error_size)) return -1;
    profile_id = rd32(h + 148); compiler_id = rd32(h + 152);
    if (profile_id >= p->string_count || compiler_id >= p->string_count) { csf_error(error, error_size, "profile/compiler string id out of range"); return -1; }
    p->profile = p->strings[profile_id]; p->compiler = p->strings[compiler_id];
    if (strcmp(p->profile, "portable-v1")) { csf_error(error, error_size, "unsupported CSF profile %s", p->profile); return -1; }
    if (open_shards(p, h + (size_t)regions[0].offset, shard_count, error, error_size)) return -1;
    if (parse_codec_tables(p, regions[4].bytes ? h + (size_t)regions[4].offset : NULL, regions[4].bytes, codec_count, error, error_size)) return -1;
    if (parse_records(p, h + (size_t)regions[1].offset, record_count, error, error_size)) return -1;
    return 0;
}

int coli_package_open_ex(ColiPackage **out, const char *path,
                         ColiCsfChecksumPolicy checksum_policy,
                         char *error, size_t error_size) {
    ColiPackage *p = NULL;
    char *manifest_path = NULL;
    struct stat st;
    int fd = -1;
    if (out) *out = NULL;
    if (!out || !path || !*path || (checksum_policy != COLI_CSF_CHECKSUM_MANIFEST_ONLY && checksum_policy != COLI_CSF_CHECKSUM_RECORD_ON_READ)) {
        csf_error(error, error_size, "invalid coli_package_open arguments"); return -1;
    }
    p = (ColiPackage *)calloc(1, sizeof(*p));
    if (!p) { csf_error(error, error_size, "out of memory allocating package"); return -1; }
    p->root = strdup(path); p->checksum_policy = checksum_policy;
    if (!p->root || !(manifest_path = path_join(path, "manifest.coli"))) {
        csf_error(error, error_size, "out of memory building manifest path"); goto fail;
    }
    fd = open(manifest_path, COMPAT_O_RDONLY);
    if (fd < 0) { csf_error(error, error_size, "cannot open %s: %s", manifest_path, strerror(errno)); goto fail; }
    if (fstat(fd, &st) || st.st_size < (off_t)CSF_MANIFEST_HEADER_BYTES || (uint64_t)st.st_size > CSF_MAX_MANIFEST_BYTES) {
        csf_error(error, error_size, "manifest has invalid size"); goto fail;
    }
    p->manifest_bytes = (size_t)st.st_size;
    p->manifest = (unsigned char *)malloc(p->manifest_bytes);
    if (!p->manifest) { csf_error(error, error_size, "out of memory reading manifest"); goto fail; }
    if (pread_full(fd, p->manifest, p->manifest_bytes, 0, error, error_size)) goto fail;
    close(fd); fd = -1; free(manifest_path); manifest_path = NULL;
    if (parse_manifest(p, error, error_size)) goto fail;
    *out = p;
    return 0;
fail:
    if (fd >= 0) close(fd);
    free(manifest_path);
    coli_package_close(p);
    return -1;
}

int coli_package_open(ColiPackage **out, const char *path, char *error, size_t error_size) {
    return coli_package_open_ex(out, path, COLI_CSF_CHECKSUM_MANIFEST_ONLY, error, error_size);
}

void coli_package_close(ColiPackage *p) {
    uint32_t i;
    if (!p) return;
    if (p->shards) for (i = 0; i < p->shard_count; ++i) {
        if (p->shards[i].fd >= 0) close(p->shards[i].fd);
        if (p->shards[i].direct_fd >= 0 && p->shards[i].direct_fd != p->shards[i].fd) close(p->shards[i].direct_fd);
        free(p->shards[i].path);
    }
    if (p->strings) for (i = 0; i < p->string_count; ++i) free(p->strings[i]);
    free(p->strings); free(p->shards); free(p->records); free(p->codec_tables);
    u64_index_free(&p->id_index); u64_index_free(&p->expert_index); u64_index_free(&p->layer_index); u64_index_free(&p->codec_index);
    name_index_free(&p->name_index);
    free(p->manifest); free(p->root); free(p);
}

size_t coli_package_record_count(const ColiPackage *p) { return p ? p->record_count : 0; }
const ColiRecordInfo *coli_package_record_at(const ColiPackage *p, size_t index) { return p && index < p->record_count ? &p->records[index] : NULL; }
const ColiRecordInfo *coli_package_record_by_id(const ColiPackage *p, uint64_t id) {
    uint32_t v = p ? u64_index_get(&p->id_index, id) : 0; return v ? &p->records[v - 1] : NULL;
}
const ColiRecordInfo *coli_package_record_by_name(const ColiPackage *p, const char *name) {
    uint32_t v = p ? name_index_get(&p->name_index, name) : 0; return v ? &p->records[v - 1] : NULL;
}
const ColiRecordInfo *coli_package_expert(const ColiPackage *p, int32_t layer, int32_t expert) {
    uint32_t v;
    if (!p || layer < 0 || expert < 0) return NULL;
    v = u64_index_get(&p->expert_index, expert_key(layer, expert)); return v ? &p->records[v - 1] : NULL;
}
const ColiRecordInfo *coli_package_layer_pack(const ColiPackage *p, int32_t layer) {
    uint32_t v;
    if (!p || layer < 0) return NULL;
    v = u64_index_get(&p->layer_index, (uint32_t)layer); return v ? &p->records[v - 1] : NULL;
}
const char *coli_package_profile(const ColiPackage *p) { return p ? p->profile : NULL; }
const char *coli_package_compiler(const ColiPackage *p) { return p ? p->compiler : NULL; }
const uint8_t *coli_package_source_fingerprint(const ColiPackage *p) { return p ? p->source_fingerprint : NULL; }
uint32_t coli_package_record_alignment(const ColiPackage *p) { return p ? p->record_alignment : 0; }

int coli_package_read_range(const ColiPackage *p, const ColiRecordInfo *r,
                            uint64_t record_offset, void *destination, size_t bytes,
                            char *error, size_t error_size) {
    uint64_t end;
    if (!p || !r || (!destination && bytes) || checked_add_u64(record_offset, bytes, &end) || end > r->stored_bytes || r->shard_id >= p->shard_count) {
        csf_error(error, error_size, "record range is invalid"); return -1;
    }
    if (!bytes) return 0;
    return pread_full(p->shards[r->shard_id].fd, destination, bytes, r->payload_offset + record_offset, error, error_size);
}

int coli_package_read_record(const ColiPackage *p, const ColiRecordInfo *r,
                             void *destination, size_t destination_bytes,
                             char *error, size_t error_size) {
    if (!p || !r || r->stored_bytes > SIZE_MAX || destination_bytes < (size_t)r->stored_bytes) {
        csf_error(error, error_size, "destination is too small for record"); return -1;
    }
    if (coli_package_read_range(p, r, 0, destination, (size_t)r->stored_bytes, error, error_size)) return -1;
    if (p->checksum_policy == COLI_CSF_CHECKSUM_RECORD_ON_READ &&
        coli_crc32c(destination, (size_t)r->stored_bytes) != r->stored_crc32c) {
        csf_error(error, error_size, "record %llu stored CRC mismatch", (unsigned long long)r->record_id); return -1;
    }
    return 0;
}

int coli_package_verify_all(const ColiPackage *p, char *error, size_t error_size) {
    size_t i;
    if (!p) { csf_error(error, error_size, "invalid package"); return -1; }
    for (i = 0; i < p->record_count; ++i)
        if (coli_package_validate_record(p, &p->records[i], 1, error, error_size)) return -1;
    return 0;
}

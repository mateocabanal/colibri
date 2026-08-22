#ifndef COLI_FORMAT_H
#define COLI_FORMAT_H

#include <stddef.h>
#include <stdint.h>
/* compat.h's Windows branch exposes stdio-based helpers. Keeping stdio visible
 * before any CSF implementation includes compat.h makes this public header's
 * normal include order portable under MinGW/UCRT64 too. */
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CSF_VERSION_MAJOR 1u
#define COLI_CSF_VERSION_MINOR 0u
#define COLI_CSF_MAX_RANK 8u

/* Known v1.0 physical-layout profiles. A package reader accepts only these
 * profiles; a target executor must still select one explicitly at open. */
#define COLI_CSF_PROFILE_PORTABLE_V1 "portable-v1"
#define COLI_CSF_PROFILE_MACOS_ARM64_METAL_APPLE8_V1 \
    "macos-arm64-metal-apple8-v1"
#define COLI_CSF_PROFILE_LINUX_X86_64_AVX2_V1 \
    "linux-x86_64-avx2-v1"

/* Container record kinds. These are CSF IDs, not QT.fmt ordinals. */
enum {
    COLI_CSF_REC_INVALID = 0x0000,
    COLI_CSF_REC_TENSOR = 0x0001,
    COLI_CSF_REC_EXPERT = 0x0002,
    COLI_CSF_REC_LAYER_PACK_RESERVED = 0x0003,
    COLI_CSF_REC_BLOB = 0x0004
};

enum {
    COLI_CSF_CODEC_NONE = 0x0000,
    COLI_CSF_CODEC_RANS256_G0_NIBBLE = 0x0001,
    COLI_CSF_CODEC_RANS256_G0_U8 = 0x0002
};

enum {
    COLI_CSF_MATH_NONE = 0x0000,
    COLI_CSF_MATH_F32 = 0x0001,
    COLI_CSF_MATH_F16 = 0x0002,
    COLI_CSF_MATH_BF16 = 0x0003,
    COLI_CSF_MATH_I8 = 0x0004,
    COLI_CSF_MATH_U8 = 0x0005,
    COLI_CSF_MATH_I16 = 0x0006,
    COLI_CSF_MATH_U16 = 0x0007,
    COLI_CSF_MATH_I32 = 0x0008,
    COLI_CSF_MATH_U32 = 0x0009,
    COLI_CSF_MATH_I64 = 0x000a,
    COLI_CSF_MATH_U64 = 0x000b,
    COLI_CSF_MATH_BOOL = 0x000c,
    COLI_CSF_MATH_FP8_E4M3FN = 0x0010,
    COLI_CSF_MATH_FP8_E5M2 = 0x0011,
    COLI_CSF_MATH_MXFP4_E2M1 = 0x0020,
    COLI_CSF_MATH_INT4_PACKED = 0x0021,
    COLI_CSF_MATH_INT4_GROUPED = 0x0022,
    COLI_CSF_MATH_MIXED = 0xfffe,
    COLI_CSF_MATH_INVALID = 0xffff
};

enum {
    COLI_CSF_SCALE_NONE = 0x0000,
    COLI_CSF_SCALE_F32 = 0x0001,
    COLI_CSF_SCALE_F16 = 0x0002,
    COLI_CSF_SCALE_BF16 = 0x0003,
    COLI_CSF_SCALE_UE8M0 = 0x0004,
    COLI_CSF_SCALE_MIXED = 0xfffe,
    COLI_CSF_SCALE_INVALID = 0xffff
};

enum {
    COLI_CSF_LAYOUT_CANONICAL = 0x0000,
    COLI_CSF_LAYOUT_ROWS16 = 0x0001,
    COLI_CSF_LAYOUT_MIXED = 0xfffe,
    COLI_CSF_LAYOUT_INVALID = 0xffff
};

enum {
    COLI_CSF_RECORD_F_OPTIONAL = 1u << 0,
    COLI_CSF_RECORD_F_HAS_LOGICAL_CRC32C = 1u << 1
};

typedef enum ColiCsfChecksumPolicy {
    /* Always checks manifest and data-shard headers. Record payload CRCs are
     * deferred unless explicitly requested with coli_package_validate_record(). */
    COLI_CSF_CHECKSUM_MANIFEST_ONLY = 0,
    /* Additionally verifies stored record CRCs in read_record(). It still does
     * not scan every payload at open; validate_record() follows its explicit
     * verify_stored_crc argument. */
    COLI_CSF_CHECKSUM_RECORD_ON_READ = 1
} ColiCsfChecksumPolicy;

typedef struct ColiPackage ColiPackage;

typedef struct ColiRecordInfo {
    uint64_t record_id;
    uint16_t kind;
    uint16_t codec;
    uint16_t math_format;
    uint16_t scale_format;
    uint16_t layout;
    uint16_t flags;
    uint32_t shard_id;
    int32_t layer;
    int32_t expert;
    uint64_t payload_offset;
    uint64_t stored_bytes;
    uint64_t decoded_bytes;
    uint32_t stored_crc32c;
    uint32_t logical_crc32c;
    uint32_t codec_table_id;
    const char *name; /* package-owned; NULL for an unnamed record */
} ColiRecordInfo;

typedef struct ColiTensorInfo {
    uint16_t rank;
    uint64_t dims[COLI_CSF_MAX_RANK];
    uint32_t scale_block_rows;
    uint32_t scale_block_columns;
    uint32_t group_size;
    uint64_t data_offset;
    uint64_t data_stored_bytes;
    uint64_t data_decoded_bytes;
    uint32_t logical_crc32c;
} ColiTensorInfo;

typedef struct ColiExpertMatrixInfo {
    uint16_t role;
    uint16_t math_format;
    uint16_t scale_format;
    uint16_t weight_codec;
    uint16_t scale_codec;
    uint16_t layout;
    uint64_t rows;
    uint64_t columns;
    uint32_t scale_block_rows;
    uint32_t scale_block_columns;
    uint32_t group_size;
    uint32_t weight_codec_table_id;
    uint32_t scale_codec_table_id;
    uint64_t weight_offset;
    uint64_t weight_stored_bytes;
    uint64_t weight_decoded_bytes;
    uint64_t scale_offset;
    uint64_t scale_stored_bytes;
    uint64_t scale_decoded_bytes;
    uint32_t logical_crc32c;
} ColiExpertMatrixInfo;

typedef struct ColiExpertInfo {
    int32_t layer;
    int32_t expert;
    uint64_t logical_bytes;
    ColiExpertMatrixInfo matrices[3];
} ColiExpertInfo;

/* Opens PATH/manifest.coli and validates all metadata/index framing, manifest
 * CRC, shard headers/sizes, record spans, duplicate keys and codec-table
 * references. It intentionally does not scan record payloads. */
int coli_package_open(ColiPackage **out, const char *path,
                      char *error, size_t error_size);
int coli_package_open_ex(ColiPackage **out, const char *path,
                         ColiCsfChecksumPolicy checksum_policy,
                         char *error, size_t error_size);
void coli_package_close(ColiPackage *package);

size_t coli_package_record_count(const ColiPackage *package);
const ColiRecordInfo *coli_package_record_at(const ColiPackage *package,
                                             size_t index);
const ColiRecordInfo *coli_package_record_by_id(const ColiPackage *package,
                                                uint64_t record_id);
const ColiRecordInfo *coli_package_record_by_name(const ColiPackage *package,
                                                  const char *name);
/* O(1) expected-time lookup; no allocation and no string hashing. */
const ColiRecordInfo *coli_package_expert(const ColiPackage *package,
                                          int32_t layer, int32_t expert);
const ColiRecordInfo *coli_package_layer_pack(const ColiPackage *package,
                                              int32_t layer);

const char *coli_package_profile(const ColiPackage *package);
const char *coli_package_compiler(const ColiPackage *package);
const uint8_t *coli_package_source_fingerprint(const ColiPackage *package);
uint32_t coli_package_record_alignment(const ColiPackage *package);
/* Package-owned absolute shard path for backend-native async I/O. */
const char *coli_package_shard_path(const ColiPackage *package, uint32_t shard_id);

enum {
    COLI_CSF_READ_DEFAULT = 0,
    /* Best effort: consume the requested source bytes without retaining them in
     * the host file cache. On macOS this uses the shard's F_NOCACHE descriptor;
     * on POSIX systems without a safe unaligned direct-I/O path it reads
     * normally then advises DONTNEED. Callers must not depend on the hint. */
    COLI_CSF_READ_UNCACHED = 1u << 0,
};

/* Reads an exact byte range relative to the top-level record. Range reads are
 * thread-safe: they use pread/compat_pread and no shared seek position. */
int coli_package_read_range_ex(const ColiPackage *package,
                               const ColiRecordInfo *record,
                               uint64_t record_offset, void *destination,
                               size_t bytes, uint32_t read_flags,
                               char *error, size_t error_size);
int coli_package_read_range(const ColiPackage *package,
                            const ColiRecordInfo *record,
                            uint64_t record_offset, void *destination,
                            size_t bytes, char *error, size_t error_size);

/* Reads the whole stored top-level record. destination_bytes must be at least
 * record->stored_bytes. With RECORD_ON_READ policy the stored CRC is checked. */
int coli_package_read_record(const ColiPackage *package,
                             const ColiRecordInfo *record,
                             void *destination, size_t destination_bytes,
                             char *error, size_t error_size);

/* Validates the typed envelope of TENSOR/EXPERT without model-specific code.
 * If verify_stored_crc is nonzero, also scans this record and verifies its
 * stored CRC. Optional/opaque record kinds only receive generic span checks. */
int coli_package_validate_record(const ColiPackage *package,
                                 const ColiRecordInfo *record,
                                 int verify_stored_crc,
                                 char *error, size_t error_size);
int coli_package_tensor_info(const ColiPackage *package,
                             const ColiRecordInfo *record,
                             ColiTensorInfo *out,
                             char *error, size_t error_size);
int coli_package_expert_info(const ColiPackage *package,
                             const ColiRecordInfo *record,
                             ColiExpertInfo *out,
                             char *error, size_t error_size);

/* Synchronous codec path used before MetalIO pipeline composition lands.
 * The returned resident record is the exact uncompressed expert envelope and
 * target execution bytes that a --codec none package would contain. Stored CRC
 * is checked before any decode; every decoded matrix must match its logical
 * CRC before the function succeeds. */
int coli_package_expert_resident_bytes(const ColiPackage *package,
                                       const ColiRecordInfo *record,
                                       uint64_t *out_bytes,
                                       char *error, size_t error_size);
int coli_package_decode_expert_record(const ColiPackage *package,
                                      const ColiRecordInfo *record,
                                      void *destination,
                                      size_t destination_bytes,
                                      size_t *written_bytes,
                                      char *error, size_t error_size);

/* Explicit tooling path: validates every typed envelope and every stored CRC.
 * This can scan the full package and is never called by package_open(). */
int coli_package_verify_all(const ColiPackage *package,
                            char *error, size_t error_size);

uint32_t coli_crc32c(const void *data, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* COLI_FORMAT_H */

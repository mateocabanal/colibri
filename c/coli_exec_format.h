#ifndef COLI_EXEC_FORMAT_H
#define COLI_EXEC_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#include "coli_format.h" /* shared record/math/scale/codec numeric registry */
#include "coli_target.h"

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_EXEC_MAX_RANK 8u

enum {
    COLI_EXEC_LAYOUT_NONE = 0x0000,
    COLI_EXEC_LAYOUT_APPLE_MIN = 0x0100,
    COLI_EXEC_LAYOUT_APPLE_MAX = 0x01ff,
    COLI_EXEC_LAYOUT_X86_MIN = 0x0200,
    COLI_EXEC_LAYOUT_X86_MAX = 0x02ff,
    COLI_EXEC_LAYOUT_CUDA_MIN = 0x0300,
    COLI_EXEC_LAYOUT_CUDA_MAX = 0x03ff,
    COLI_EXEC_LAYOUT_MIXED = 0xfffe,
    COLI_EXEC_LAYOUT_INVALID = 0xffff
};

typedef enum ColiExecChecksumPolicy {
    COLI_EXEC_CHECKSUM_MANIFEST_ONLY = 0,
    COLI_EXEC_CHECKSUM_RECORD_ON_READ = 1
} ColiExecChecksumPolicy;

typedef struct ColiExecPackage ColiExecPackage;

typedef struct ColiExecRecordInfo {
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
    uint64_t resident_bytes;
    uint32_t stored_crc32c;
    uint32_t logical_crc32c;
    uint32_t codec_table_id;
    const char *name; /* package-owned, NULL when unnamed */
} ColiExecRecordInfo;

typedef struct ColiExecTensorInfo {
    uint16_t rank;
    uint64_t dims[COLI_EXEC_MAX_RANK]; /* logical model geometry */
    uint32_t scale_block_rows;
    uint32_t scale_block_columns;
    uint32_t group_size;
    uint64_t data_offset;
    uint64_t data_stored_bytes;
    uint64_t data_resident_bytes;
    uint32_t logical_crc32c;
} ColiExecTensorInfo;

typedef struct ColiExecMatrixInfo {
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
    uint64_t weight_resident_bytes;
    uint64_t scale_offset;
    uint64_t scale_stored_bytes;
    uint64_t scale_resident_bytes;
    uint32_t logical_crc32c;
} ColiExecMatrixInfo;

typedef struct ColiExecExpertInfo {
    int32_t layer;
    int32_t expert;
    uint64_t logical_bytes; /* tooling semantic count from envelope */
    ColiExecMatrixInfo matrices[3];
} ColiExecExpertInfo;

/* Production target-open path. Compatibility is checked before shard files or
 * execution records are published. The runtime descriptor is injected by the
 * platform/backend layer; the generic parser never guesses Metal/CUDA support. */
int coli_exec_package_open(ColiExecPackage **out, const char *path,
                           const ColiRuntimeTarget *runtime,
                           char *error, size_t error_size);
int coli_exec_package_open_ex(ColiExecPackage **out, const char *path,
                              const ColiRuntimeTarget *runtime,
                              ColiExecChecksumPolicy checksum_policy,
                              char *error, size_t error_size);
void coli_exec_package_close(ColiExecPackage *package);

const ColiTargetInfo *coli_exec_package_target(const ColiExecPackage *package);
size_t coli_exec_package_record_count(const ColiExecPackage *package);
const ColiExecRecordInfo *coli_exec_package_record_at(const ColiExecPackage *package,
                                                      size_t index);
const ColiExecRecordInfo *coli_exec_package_record_by_id(const ColiExecPackage *package,
                                                         uint64_t record_id);
const ColiExecRecordInfo *coli_exec_package_record_by_name(const ColiExecPackage *package,
                                                           const char *name);
/* Expected O(1), allocation-free hot lookup. */
const ColiExecRecordInfo *coli_exec_package_expert(const ColiExecPackage *package,
                                                   int32_t layer, int32_t expert);
const ColiExecRecordInfo *coli_exec_package_layer_pack(const ColiExecPackage *package,
                                                       int32_t layer);

int coli_exec_package_read_range(const ColiExecPackage *package,
                                 const ColiExecRecordInfo *record,
                                 uint64_t record_offset,
                                 void *destination, size_t bytes,
                                 char *error, size_t error_size);
int coli_exec_package_read_record(const ColiExecPackage *package,
                                  const ColiExecRecordInfo *record,
                                  void *destination, size_t destination_bytes,
                                  char *error, size_t error_size);

/* Generic envelope/framing checks only. Target layout interpretation is owned by
 * the target-profile ABI/#26 kernels; this parser verifies spans/codecs/layout
 * namespace without reconstructing canonical/source tensor bytes. */
int coli_exec_package_validate_record(const ColiExecPackage *package,
                                      const ColiExecRecordInfo *record,
                                      int verify_stored_crc,
                                      char *error, size_t error_size);
int coli_exec_package_tensor_info(const ColiExecPackage *package,
                                  const ColiExecRecordInfo *record,
                                  ColiExecTensorInfo *out,
                                  char *error, size_t error_size);
int coli_exec_package_expert_info(const ColiExecPackage *package,
                                  const ColiExecRecordInfo *record,
                                  ColiExecExpertInfo *out,
                                  char *error, size_t error_size);

/* Full stored-byte corruption scan. Logical inverse-transform verification is a
 * profile/tooling concern and is intentionally not performed by generic runtime
 * open/load. */
int coli_exec_package_verify_all(const ColiExecPackage *package,
                                 char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* COLI_EXEC_FORMAT_H */

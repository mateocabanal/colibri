#ifndef COLI_EXECUTOR_H
#define COLI_EXECUTOR_H

#include "coli_format.h"
#include "coli_target.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiExecutor ColiExecutor;

typedef struct ColiExecutorOpenOptions {
    /* Exact physical-layout profile. There is deliberately no native fallback. */
    const char *required_profile;
    ColiCsfChecksumPolicy checksum_policy;
    /* Required for frozen execution profiles such as Apple8-v1. The runtime
     * descriptor is compared against the generated target registry before any
     * execution record is published. */
    const ColiRuntimeTarget *runtime_target;
    /* Zero is unlimited. */
    uint64_t max_resident_record_bytes;
} ColiExecutorOpenOptions;

int coli_executor_open(ColiExecutor **out, const char *package_path,
                       const ColiExecutorOpenOptions *options,
                       char *error, size_t error_size);
void coli_executor_close(ColiExecutor *executor);

const ColiRecordInfo *coli_executor_expert(const ColiExecutor *executor,
                                           int32_t layer, int32_t expert);
const ColiRecordInfo *coli_executor_record_by_name(const ColiExecutor *executor,
                                                   const char *name);
int coli_executor_load_record(const ColiExecutor *executor,
                              const ColiRecordInfo *record,
                              void *resident_slot, size_t resident_bytes,
                              char *error, size_t error_size);
int coli_executor_expert_info(const ColiExecutor *executor,
                              int32_t layer, int32_t expert,
                              ColiExpertInfo *out,
                              char *error, size_t error_size);
int coli_executor_load_expert(const ColiExecutor *executor,
                              int32_t layer, int32_t expert,
                              void *resident_slot, size_t resident_bytes,
                              char *error, size_t error_size);

const ColiPackage *coli_executor_package(const ColiExecutor *executor);

#ifdef __cplusplus
}
#endif

#endif

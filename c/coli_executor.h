#ifndef COLI_EXECUTOR_H
#define COLI_EXECUTOR_H

#include "coli_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Target-selected package access for the runtime hot path. It owns no tensor
 * name mapping and never repacks a record: callers provide final resident-slot
 * storage and select experts by (layer, expert). */
typedef struct ColiExecutor ColiExecutor;

typedef struct ColiExecutorOpenOptions {
    /* Required exact physical-layout profile. There is deliberately no native
     * fallback: accepting another target layout would be unsafe. */
    const char *required_profile;
    ColiCsfChecksumPolicy checksum_policy;
    /* For a production Apple8-v1 profile these must exactly match the shared
     * target registry. Zero is not a wildcard for that profile. Other legacy
     * profiles ignore these fields until they gain a frozen target contract. */
    uint32_t required_execution_layout_abi;
    uint32_t required_kernel_abi;
    uint32_t required_target_class;
    /* Zero is unlimited. Otherwise reject records larger than this resident
     * slot contract when loading them. */
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

#endif /* COLI_EXECUTOR_H */

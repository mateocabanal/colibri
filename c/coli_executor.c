#include "coli_executor.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ColiExecutor {
    ColiPackage *package;
    uint64_t max_resident_record_bytes;
};

static void executor_error(char *error, size_t error_size, const char *fmt, ...) {
    va_list ap;
    if (!error || !error_size) return;
    va_start(ap, fmt);
    (void)vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

int coli_executor_open(ColiExecutor **out, const char *package_path,
                       const ColiExecutorOpenOptions *options,
                       char *error, size_t error_size) {
    ColiExecutor *executor;
    ColiPackage *package = NULL;
    ColiCsfChecksumPolicy policy;
    const char *actual;
    if (out) *out = NULL;
    if (!out || !package_path || !*package_path || !options ||
        !options->required_profile || !*options->required_profile) {
        executor_error(error, error_size, "executor requires package path and exact target profile");
        return -1;
    }
    policy = options->checksum_policy;
    if (policy != COLI_CSF_CHECKSUM_MANIFEST_ONLY &&
        policy != COLI_CSF_CHECKSUM_RECORD_ON_READ) {
        executor_error(error, error_size, "invalid executor checksum policy");
        return -1;
    }
    if (coli_package_open_ex(&package, package_path, policy, error, error_size)) return -1;
    actual = coli_package_profile(package);
    if (!actual || strcmp(actual, options->required_profile)) {
        executor_error(error, error_size, "package target profile %s does not match required %s",
                       actual ? actual : "<missing>", options->required_profile);
        coli_package_close(package);
        return -1;
    }
    executor = (ColiExecutor *)calloc(1, sizeof(*executor));
    if (!executor) {
        executor_error(error, error_size, "out of memory allocating COLI executor");
        coli_package_close(package);
        return -1;
    }
    executor->package = package;
    executor->max_resident_record_bytes = options->max_resident_record_bytes;
    *out = executor;
    return 0;
}

void coli_executor_close(ColiExecutor *executor) {
    if (!executor) return;
    coli_package_close(executor->package);
    free(executor);
}

const ColiRecordInfo *coli_executor_expert(const ColiExecutor *executor,
                                           int32_t layer, int32_t expert) {
    return executor ? coli_package_expert(executor->package, layer, expert) : NULL;
}

const ColiRecordInfo *coli_executor_record_by_name(const ColiExecutor *executor,
                                                   const char *name) {
    return executor ? coli_package_record_by_name(executor->package, name) : NULL;
}

int coli_executor_load_record(const ColiExecutor *executor,
                              const ColiRecordInfo *record,
                              void *resident_slot, size_t resident_bytes,
                              char *error, size_t error_size) {
    if (!executor || !record || !resident_slot || record->stored_bytes > SIZE_MAX ||
        resident_bytes < (size_t)record->stored_bytes) {
        executor_error(error, error_size, "invalid COLI record resident-slot load");
        return -1;
    }
    return coli_package_read_record(executor->package, record, resident_slot,
                                    resident_bytes, error, error_size);
}

int coli_executor_expert_info(const ColiExecutor *executor,
                              int32_t layer, int32_t expert,
                              ColiExpertInfo *out,
                              char *error, size_t error_size) {
    const ColiRecordInfo *record = coli_executor_expert(executor, layer, expert);
    if (!record) {
        executor_error(error, error_size, "expert (%d,%d) is absent", layer, expert);
        return -1;
    }
    return coli_package_expert_info(executor->package, record, out, error, error_size);
}

int coli_executor_load_expert(const ColiExecutor *executor,
                              int32_t layer, int32_t expert,
                              void *resident_slot, size_t resident_bytes,
                              char *error, size_t error_size) {
    const ColiRecordInfo *record;
    if (!executor || !resident_slot) {
        executor_error(error, error_size, "invalid executor resident-slot load");
        return -1;
    }
    record = coli_executor_expert(executor, layer, expert);
    if (!record) {
        executor_error(error, error_size, "expert (%d,%d) is absent", layer, expert);
        return -1;
    }
    if (executor->max_resident_record_bytes &&
        record->stored_bytes > executor->max_resident_record_bytes) {
        executor_error(error, error_size, "expert (%d,%d) exceeds resident-slot contract", layer, expert);
        return -1;
    }
    if (record->stored_bytes > SIZE_MAX || resident_bytes < (size_t)record->stored_bytes) {
        executor_error(error, error_size, "resident slot too small for expert (%d,%d): need %llu bytes",
                       layer, expert, (unsigned long long)record->stored_bytes);
        return -1;
    }
    return coli_package_read_record(executor->package, record, resident_slot,
                                    resident_bytes, error, error_size);
}

const ColiPackage *coli_executor_package(const ColiExecutor *executor) {
    return executor ? executor->package : NULL;
}

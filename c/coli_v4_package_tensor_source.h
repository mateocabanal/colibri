#ifndef COLI_V4_PACKAGE_TENSOR_SOURCE_H
#define COLI_V4_PACKAGE_TENSOR_SOURCE_H

/*
 * Package-only V4 compatibility bridge for code that still consumes the
 * safetensors-shaped named-tensor API.
 *
 * COLI already stores ancillary tensors (including mtp.*) as independently
 * named COLITENS records.  Rather than duplicate DSpark's shape/dtype/security
 * validation, expose those records through stable synthetic st_tensor views and
 * redirect the actual byte reads to ColiPackage.  Real safetensors indexes pass
 * straight through unchanged.
 *
 * This is intentionally a migration seam.  Target experts keep using native
 * COLI expert records; only name-addressed ancillary tensors come through here.
 */

#include "deepseek_v4_internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COLI_V4_PACKAGE_TENSOR_CAP 8192u
#define COLI_V4_PACKAGE_HASH_CAP   16384u
#define COLI_V4_PACKAGE_FD_BASE    1000000

typedef struct ColiV4PackageTensorAdapter {
    ColiSafetensorsTensor tensor;
    const ColiRecordInfo *record;
    ColiTensorInfo info;
} ColiV4PackageTensorAdapter;

typedef struct ColiV4PackageTensorSourceState {
    const ColiExecutor *executor;
    ColiV4PackageTensorAdapter *entries;
    int *hash_slots;
    size_t count;
} ColiV4PackageTensorSourceState;

/* DSpark itself is process-global today, so this compatibility state follows
 * the same lifetime/concurrency contract instead of pretending to be safer
 * than its consumer.  A future multi-engine DSpark refactor should move both
 * into engine-owned state together. */
static ColiV4PackageTensorSourceState g_coli_v4_package_tensor_source;

static inline uint64_t coli_v4_package_name_hash(const char *name) {
    uint64_t hash = UINT64_C(1469598103934665603);
    while (name && *name) {
        hash ^= (unsigned char)*name++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static inline void coli_v4_package_source_reset(void) {
    free(g_coli_v4_package_tensor_source.hash_slots);
    free(g_coli_v4_package_tensor_source.entries);
    memset(&g_coli_v4_package_tensor_source, 0,
           sizeof(g_coli_v4_package_tensor_source));
}

static inline int coli_v4_package_source_bind(const ColiExecutor *executor) {
    ColiV4PackageTensorSourceState *state = &g_coli_v4_package_tensor_source;
    if (state->executor == executor &&
        (!executor || (state->entries && state->hash_slots)))
        return 0;
    coli_v4_package_source_reset();
    if (!executor) return 0;
    state->entries = calloc(COLI_V4_PACKAGE_TENSOR_CAP,
                            sizeof(*state->entries));
    state->hash_slots = malloc(COLI_V4_PACKAGE_HASH_CAP *
                               sizeof(*state->hash_slots));
    if (!state->entries || !state->hash_slots) {
        coli_v4_package_source_reset();
        return -1;
    }
    for (size_t i = 0; i < COLI_V4_PACKAGE_HASH_CAP; i++)
        state->hash_slots[i] = -1;
    state->executor = executor;
    return 0;
}

static inline ColiSafetensorsIndex *coli_v4_package_source_sentinel(void) {
    /* A zeroed shards object is never a valid opened safetensors index.  It is
     * only a non-NULL capability marker so legacy V4 code does not reject the
     * package before the named-tensor bridge gets a chance to resolve records. */
    static ColiSafetensorsIndex sentinel;
    return &sentinel;
}

static inline int coli_v4_package_source_active(
        const ColiSafetensorsIndex *index) {
    return g_coli_v4_package_tensor_source.executor != NULL &&
           (!index || (index->n == 0 && index->nfd == 0));
}

static inline int coli_v4_package_st_dtype(const ColiRecordInfo *record,
                                           const char *name) {
    if (!record) return -1;
    switch (record->math_format) {
        case COLI_CSF_MATH_BF16: return COLI_ST_BF16;
        case COLI_CSF_MATH_F16: return COLI_ST_F16;
        case COLI_CSF_MATH_F32: return COLI_ST_F32;
        case COLI_CSF_MATH_FP8_E4M3FN: return COLI_ST_F8_E4M3;
        case COLI_CSF_MATH_I64:
        case COLI_CSF_MATH_U64: return COLI_ST_I64;
        case COLI_CSF_MATH_U8:
            /* Exact COLI keeps standalone UE8M0 sidecars byte-identical and
             * registers them as U8 because UE8M0 is a scale format, not a math
             * format. MTP *.scale records are therefore the one place where
             * the semantic dtype must be recovered from the role name. */
            if (name && !strncmp(name, "mtp.", 4)) {
                size_t length = strlen(name);
                if (length >= 6 && !strcmp(name + length - 6, ".scale"))
                    return COLI_ST_F8_E8M0;
            }
            return COLI_ST_U8;
        case COLI_CSF_MATH_MXFP4_E2M1:
        case COLI_CSF_MATH_I8:
        case COLI_CSF_MATH_INT4_PACKED:
        case COLI_CSF_MATH_INT4_GROUPED:
            return COLI_ST_U8;
        default: return -1;
    }
}

static inline int coli_v4_package_tensor_numel(const ColiTensorInfo *info,
                                               int64_t *out) {
    if (!info || !out || info->rank > ST_MAX_RANK) return -1;
    uint64_t value = 1;
    for (uint16_t i = 0; i < info->rank; i++) {
        if (info->dims[i] != 0 && value > (uint64_t)INT64_MAX / info->dims[i])
            return -1;
        value *= info->dims[i];
    }
    *out = (int64_t)value;
    return 0;
}

static inline ColiV4PackageTensorAdapter *
coli_v4_package_adapter_from_tensor(const ColiSafetensorsTensor *tensor) {
    if (!tensor || tensor->fd > -COLI_V4_PACKAGE_FD_BASE) return NULL;
    int64_t index = -(int64_t)tensor->fd - COLI_V4_PACKAGE_FD_BASE;
    if (index < 0 || (size_t)index >= g_coli_v4_package_tensor_source.count)
        return NULL;
    ColiV4PackageTensorAdapter *entry =
        &g_coli_v4_package_tensor_source.entries[index];
    return &entry->tensor == tensor ? entry : NULL;
}

static inline ColiV4PackageTensorAdapter *
coli_v4_package_adapter_from_shard(int shard) {
    if (shard <= 0) return NULL;
    size_t index = (size_t)(shard - 1);
    if (index >= g_coli_v4_package_tensor_source.count) return NULL;
    return &g_coli_v4_package_tensor_source.entries[index];
}

static inline const ColiSafetensorsTensor *
coli_v4_package_source_find(const ColiSafetensorsIndex *index,
                            const char *name) {
    if (!coli_v4_package_source_active(index))
        return coli_st_find(index, name);
    if (!name) return NULL;

    ColiV4PackageTensorSourceState *state = &g_coli_v4_package_tensor_source;
    uint64_t hash = coli_v4_package_name_hash(name);
    size_t slot = (size_t)hash & (COLI_V4_PACKAGE_HASH_CAP - 1u);
    for (size_t probe = 0; probe < COLI_V4_PACKAGE_HASH_CAP; probe++) {
        int entry_index = state->hash_slots[slot];
        if (entry_index < 0) break;
        ColiV4PackageTensorAdapter *entry = &state->entries[entry_index];
        if (entry->tensor.name && !strcmp(entry->tensor.name, name))
            return &entry->tensor;
        slot = (slot + 1u) & (COLI_V4_PACKAGE_HASH_CAP - 1u);
    }

    const ColiRecordInfo *record =
        coli_executor_record_by_name(state->executor, name);
    if (!record || record->kind != COLI_CSF_REC_TENSOR ||
        record->codec != COLI_CSF_CODEC_NONE ||
        state->count >= COLI_V4_PACKAGE_TENSOR_CAP)
        return NULL;
    ColiTensorInfo info;
    const ColiPackage *package = coli_executor_package(state->executor);
    if (!package || coli_package_tensor_info(package, record, &info, NULL, 0) ||
        info.rank > ST_MAX_RANK ||
        info.data_stored_bytes != info.data_decoded_bytes ||
        info.data_stored_bytes > INT64_MAX)
        return NULL;
    int dtype = coli_v4_package_st_dtype(record, name);
    int64_t numel = 0;
    if (dtype < 0 || coli_v4_package_tensor_numel(&info, &numel)) return NULL;

    size_t index_value = state->count++;
    ColiV4PackageTensorAdapter *entry = &state->entries[index_value];
    memset(entry, 0, sizeof(*entry));
    entry->record = record;
    entry->info = info;
    entry->tensor.name = (char *)record->name;
    entry->tensor.fd = -(COLI_V4_PACKAGE_FD_BASE + (int)index_value);
    entry->tensor.off = (int64_t)info.data_offset;
    entry->tensor.nbytes = (int64_t)info.data_stored_bytes;
    entry->tensor.dtype = dtype;
    entry->tensor.numel = numel;
    entry->tensor.rank = (int)info.rank;
    for (uint16_t i = 0; i < info.rank; i++) {
        if (info.dims[i] > INT64_MAX) {
            state->count--;
            memset(entry, 0, sizeof(*entry));
            return NULL;
        }
        entry->tensor.shape[i] = (int64_t)info.dims[i];
    }

    slot = (size_t)hash & (COLI_V4_PACKAGE_HASH_CAP - 1u);
    for (size_t probe = 0; probe < COLI_V4_PACKAGE_HASH_CAP; probe++) {
        if (state->hash_slots[slot] < 0) {
            state->hash_slots[slot] = (int)index_value;
            return &entry->tensor;
        }
        slot = (slot + 1u) & (COLI_V4_PACKAGE_HASH_CAP - 1u);
    }
    state->count--;
    memset(entry, 0, sizeof(*entry));
    return NULL;
}

static inline int coli_v4_package_source_read_tensor(
        const ColiSafetensorsIndex *index,
        const ColiSafetensorsTensor *tensor,
        void *destination) {
    ColiV4PackageTensorAdapter *entry =
        coli_v4_package_adapter_from_tensor(tensor);
    if (!entry) return coli_st_read_tensor(index, tensor, destination);
    if (!destination || !g_coli_v4_package_tensor_source.executor) return -1;
    return coli_package_read_range(
        coli_executor_package(g_coli_v4_package_tensor_source.executor),
        entry->record, entry->info.data_offset, destination,
        (size_t)entry->info.data_stored_bytes, NULL, 0);
}

static inline int coli_v4_package_source_tensor_shard(
        const ColiSafetensorsIndex *index,
        const ColiSafetensorsTensor *tensor) {
    ColiV4PackageTensorAdapter *entry =
        coli_v4_package_adapter_from_tensor(tensor);
    if (!entry) return coli_st_tensor_shard(index, tensor);
    /* Package head residency uses the established synthetic shard -2/offset 0
     * contract. Returning it here makes the existing batched-head fast path
     * reuse the resident BF16 COLI head without another special case. */
    if (entry->tensor.name && !strcmp(entry->tensor.name, "head.weight"))
        return -2;
    return (int)(entry - g_coli_v4_package_tensor_source.entries) + 1;
}

static inline int coli_v4_package_source_read_at(
        const ColiSafetensorsIndex *index, int shard,
        uint64_t offset, size_t length, void *destination) {
    if (!coli_v4_package_source_active(index))
        return coli_st_read_at(index, shard, offset, length, destination);
    ColiV4PackageTensorAdapter *entry =
        coli_v4_package_adapter_from_shard(shard);
    if (!entry || !destination ||
        offset < entry->info.data_offset ||
        offset - entry->info.data_offset > entry->info.data_stored_bytes ||
        length > entry->info.data_stored_bytes -
                     (offset - entry->info.data_offset))
        return -1;
    return coli_package_read_range(
        coli_executor_package(g_coli_v4_package_tensor_source.executor),
        entry->record, offset, destination, length, NULL, 0);
}

static inline int coli_v4_package_source_read_at_streaming(
        const ColiSafetensorsIndex *index, int shard,
        uint64_t offset, size_t length, void *destination) {
    if (!coli_v4_package_source_active(index))
        return coli_st_read_at_streaming(index, shard, offset, length,
                                         destination);
    ColiV4PackageTensorAdapter *entry =
        coli_v4_package_adapter_from_shard(shard);
    if (!entry || !destination ||
        offset < entry->info.data_offset ||
        offset - entry->info.data_offset > entry->info.data_stored_bytes ||
        length > entry->info.data_stored_bytes -
                     (offset - entry->info.data_offset))
        return -1;
    return coli_package_read_range_ex(
        coli_executor_package(g_coli_v4_package_tensor_source.executor),
        entry->record, offset, destination, length, COLI_CSF_READ_UNCACHED,
        NULL, 0);
}

static inline int64_t coli_v4_package_read_scale_f32(
        ColiSafetensorsIndex *index, const char *name,
        float *out, int64_t cap, int drop) {
    if (!coli_v4_package_source_active(index))
        return st_read_scale_f32(index, name, out, cap, drop);
    const ColiSafetensorsTensor *tensor =
        coli_v4_package_source_find(index, name);
    if (!tensor || !out || tensor->numel < 0 || tensor->numel > cap ||
        tensor->nbytes < 0)
        return -1;
    size_t bytes = (size_t)tensor->nbytes;
    void *raw = malloc(bytes);
    if (!raw || coli_v4_package_source_read_tensor(index, tensor, raw)) {
        free(raw);
        return -1;
    }
    if (tensor->dtype == COLI_ST_F32) {
        if (bytes != (size_t)tensor->numel * sizeof(float)) {
            free(raw); return -1;
        }
        memcpy(out, raw, bytes);
    } else if (tensor->dtype == COLI_ST_BF16 || tensor->dtype == COLI_ST_F16) {
        if (bytes != (size_t)tensor->numel * sizeof(uint16_t)) {
            free(raw); return -1;
        }
        const uint16_t *values = raw;
        for (int64_t i = 0; i < tensor->numel; i++)
            out[i] = tensor->dtype == COLI_ST_BF16
                ? bf16_to_f32(values[i]) : f16_to_f32(values[i]);
    } else if (tensor->dtype == COLI_ST_F8_E8M0) {
        if (bytes != (size_t)tensor->numel) { free(raw); return -1; }
        const uint8_t *values = raw;
        for (int64_t i = 0; i < tensor->numel; i++)
            out[i] = ue8m0_to_f32(values[i]);
    } else {
        free(raw); return -1;
    }
    free(raw);
    return tensor->numel;
}

static inline int coli_v4_package_tensor_load_f32(
        ColiFloatTensor *output, const ColiSafetensorsIndex *index,
        const char *name, char *error, size_t error_size) {
    if (!coli_v4_package_source_active(index))
        return coli_tensor_load_f32(output, index, name, error, error_size);
    if (!output || !name) return -1;
    memset(output, 0, sizeof(*output));
    const ColiSafetensorsTensor *tensor =
        coli_v4_package_source_find(index, name);
    if (!tensor || (tensor->dtype != COLI_ST_F32 &&
                    tensor->dtype != COLI_ST_BF16) ||
        tensor->numel < 0 || tensor->nbytes < 0) {
        if (error && error_size)
            snprintf(error, error_size, "missing BF16/F32 COLI tensor: %s", name);
        return -1;
    }
    void *raw = malloc((size_t)tensor->nbytes);
    output->data = malloc((size_t)tensor->numel * sizeof(*output->data));
    if (!raw || !output->data ||
        coli_v4_package_source_read_tensor(index, tensor, raw)) {
        free(raw);
        coli_float_tensor_free(output);
        if (error && error_size)
            snprintf(error, error_size, "cannot read COLI tensor: %s", name);
        return -1;
    }
    if (tensor->dtype == COLI_ST_F32) {
        if ((size_t)tensor->nbytes !=
            (size_t)tensor->numel * sizeof(float)) {
            free(raw); coli_float_tensor_free(output); return -1;
        }
        memcpy(output->data, raw, (size_t)tensor->nbytes);
    } else {
        if ((size_t)tensor->nbytes !=
            (size_t)tensor->numel * sizeof(uint16_t)) {
            free(raw); coli_float_tensor_free(output); return -1;
        }
        const uint16_t *values = raw;
        for (int64_t i = 0; i < tensor->numel; i++)
            output->data[i] = coli_bf16_decode(values[i]);
    }
    free(raw);
    output->count = (uint64_t)tensor->numel;
    output->rank = tensor->rank;
    for (int i = 0; i < tensor->rank && i < ST_MAX_RANK; i++)
        output->shape[i] = tensor->shape[i];
    return 0;
}

#endif /* COLI_V4_PACKAGE_TENSOR_SOURCE_H */

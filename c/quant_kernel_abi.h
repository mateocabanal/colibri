#ifndef COLIBRI_QUANT_KERNEL_ABI_H
#define COLIBRI_QUANT_KERNEL_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_QUANT_KERNEL_DESCRIPTOR_VERSION 1u
#define COLI_QUANT_KERNEL_ABI_VERSION 1u
#define COLI_EXEC_LAYOUT_ABI_VERSION 1u

/* Semantic math format. This does not imply one physical target layout. */
typedef enum {
    COLI_QUANT_FORMAT_INVALID = 0,
    COLI_QUANT_FORMAT_INT8 = 1,
    COLI_QUANT_FORMAT_INT4_PACKED = 2,
    COLI_QUANT_FORMAT_INT4_GROUPED = 3,
    COLI_QUANT_FORMAT_MXFP4_E2M1 = 4,
} ColiQuantFormatId;

typedef enum {
    COLI_SCALE_FORMAT_INVALID = 0,
    COLI_SCALE_FORMAT_NONE = 1,
    COLI_SCALE_FORMAT_F32_PER_ROW = 2,
    COLI_SCALE_FORMAT_F32_PER_GROUP = 3,
    COLI_SCALE_FORMAT_E8M0_PER_GROUP = 4,
} ColiScaleFormatId;

typedef enum {
    COLI_ACTIVATION_INVALID = 0,
    COLI_ACTIVATION_SWIGLU = 1,
    COLI_ACTIVATION_SILU = 2,
    COLI_ACTIVATION_GELU = 3,
} ColiActivationKind;

/* Physical execution-layout IDs. Adding a target-native layout requires a new
 * ID; callers must never infer layout compatibility from a model name. */
typedef enum {
    COLI_EXEC_LAYOUT_INVALID = 0,
    COLI_EXEC_LAYOUT_INT8_ROW_MAJOR = 1,
    COLI_EXEC_LAYOUT_INT4_PACKED_ROW_MAJOR = 2,
    COLI_EXEC_LAYOUT_INT4_GROUPED_ROW_MAJOR = 3,
    COLI_EXEC_LAYOUT_MXFP4_E2M1_E8M0_32 = 4,
} ColiExecutionLayoutAbiId;

/* Kernel ABI identifies the high-level execution unit, independently of the
 * concrete backend implementation selected for that ABI. */
typedef enum {
    COLI_KERNEL_ABI_INVALID = 0,
    COLI_KERNEL_ABI_QUANT_GEMV = 1,
    COLI_KERNEL_ABI_QUANT_GEMM = 2,
    COLI_KERNEL_ABI_FUSED_SWIGLU_EXPERT = 3,
    COLI_KERNEL_ABI_ROUTED_EXPERT_BLOCK = 4,
} ColiQuantKernelAbiId;

enum {
    COLI_BACKEND_CAP_QUANT_GEMV = UINT64_C(1) << 0,
    COLI_BACKEND_CAP_QUANT_GEMM = UINT64_C(1) << 1,
    COLI_BACKEND_CAP_DUAL_GATE_UP = UINT64_C(1) << 2,
    COLI_BACKEND_CAP_FUSED_EXPERT = UINT64_C(1) << 3,
    COLI_BACKEND_CAP_ROUTED_EXPERT_BLOCK = UINT64_C(1) << 4,
    COLI_BACKEND_CAP_BATCHED_EXPERT_ROWS = UINT64_C(1) << 5,
};

#define COLI_ACTIVATION_BIT(kind) \
    ((uint64_t)1u << (unsigned)(kind))

typedef struct {
    uint32_t descriptor_version;
    uint32_t quant_format;
    uint32_t scale_format;
    uint32_t activation;

    uint32_t layout_abi_id;
    uint32_t layout_abi_version;
    uint32_t kernel_abi_id;
    uint32_t kernel_abi_version;

    uint32_t input_columns;
    uint32_t output_rows;
    uint32_t weight_block_rows;
    uint32_t weight_block_columns;
    uint32_t scale_block_rows;
    uint32_t scale_block_columns;
    uint32_t group_size;
    uint32_t required_alignment;

    uint32_t min_rows;
    uint32_t max_rows;
    uint64_t scratch_bytes_per_row;
    uint64_t required_backend_caps;
} ColiQuantKernelRecordDesc;

/* A backend/executor asks for one exact execution contract. Keeping this
 * separate from the record descriptor prevents "format is close enough"
 * dispatch and makes ABI mismatch fail closed. */
typedef struct {
    uint32_t descriptor_version;
    uint32_t quant_format;
    uint32_t scale_format;
    uint32_t activation;

    uint32_t layout_abi_id;
    uint32_t layout_abi_version;
    uint32_t kernel_abi_id;
    uint32_t kernel_abi_version;

    uint32_t input_columns;
    uint32_t output_rows;
    uint32_t weight_block_rows;
    uint32_t weight_block_columns;
    uint32_t scale_block_rows;
    uint32_t scale_block_columns;
    uint32_t group_size;

    uint32_t rows;
    uint32_t record_alignment;
} ColiQuantKernelRequestDesc;

typedef struct {
    uint64_t capability_bits;
    uint64_t supported_activation_bits;
    uint64_t scratch_bytes_available;
    uint32_t max_rows;
    uint32_t max_supported_alignment;
} ColiQuantBackendCaps;

typedef enum {
    COLI_QUANT_KERNEL_COMPAT_OK = 0,
    COLI_QUANT_KERNEL_COMPAT_INVALID_RECORD,
    COLI_QUANT_KERNEL_COMPAT_INVALID_REQUEST,
    COLI_QUANT_KERNEL_COMPAT_INVALID_BACKEND,
    COLI_QUANT_KERNEL_COMPAT_QUANT_MISMATCH,
    COLI_QUANT_KERNEL_COMPAT_SCALE_MISMATCH,
    COLI_QUANT_KERNEL_COMPAT_ACTIVATION_MISMATCH,
    COLI_QUANT_KERNEL_COMPAT_LAYOUT_ABI_MISMATCH,
    COLI_QUANT_KERNEL_COMPAT_KERNEL_ABI_MISMATCH,
    COLI_QUANT_KERNEL_COMPAT_GEOMETRY_MISMATCH,
    COLI_QUANT_KERNEL_COMPAT_ALIGNMENT_MISMATCH,
    COLI_QUANT_KERNEL_COMPAT_ROW_LIMIT,
    COLI_QUANT_KERNEL_COMPAT_CAPABILITY_MISSING,
    COLI_QUANT_KERNEL_COMPAT_SCRATCH_LIMIT,
} ColiQuantKernelCompat;

static inline int coli_quant_u64_mul_ok(uint64_t a, uint64_t b,
                                        uint64_t *out) {
    if (!out || (a && b > UINT64_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static inline int coli_quant_power_of_two_u32(uint32_t value) {
    return value && (value & (value - 1u)) == 0;
}

static inline int coli_quant_format_known(uint32_t value) {
    switch (value) {
        case COLI_QUANT_FORMAT_INT8:
        case COLI_QUANT_FORMAT_INT4_PACKED:
        case COLI_QUANT_FORMAT_INT4_GROUPED:
        case COLI_QUANT_FORMAT_MXFP4_E2M1:
            return 1;
        default:
            return 0;
    }
}

static inline int coli_scale_format_known(uint32_t value) {
    switch (value) {
        case COLI_SCALE_FORMAT_NONE:
        case COLI_SCALE_FORMAT_F32_PER_ROW:
        case COLI_SCALE_FORMAT_F32_PER_GROUP:
        case COLI_SCALE_FORMAT_E8M0_PER_GROUP:
            return 1;
        default:
            return 0;
    }
}

static inline int coli_activation_known(uint32_t value) {
    switch (value) {
        case COLI_ACTIVATION_SWIGLU:
        case COLI_ACTIVATION_SILU:
        case COLI_ACTIVATION_GELU:
            return 1;
        default:
            return 0;
    }
}

static inline int coli_exec_layout_known(uint32_t value) {
    switch (value) {
        case COLI_EXEC_LAYOUT_INT8_ROW_MAJOR:
        case COLI_EXEC_LAYOUT_INT4_PACKED_ROW_MAJOR:
        case COLI_EXEC_LAYOUT_INT4_GROUPED_ROW_MAJOR:
        case COLI_EXEC_LAYOUT_MXFP4_E2M1_E8M0_32:
            return 1;
        default:
            return 0;
    }
}

static inline int coli_kernel_abi_known(uint32_t value) {
    switch (value) {
        case COLI_KERNEL_ABI_QUANT_GEMV:
        case COLI_KERNEL_ABI_QUANT_GEMM:
        case COLI_KERNEL_ABI_FUSED_SWIGLU_EXPERT:
        case COLI_KERNEL_ABI_ROUTED_EXPERT_BLOCK:
            return 1;
        default:
            return 0;
    }
}

static inline int coli_quant_kernel_geometry_valid(
        uint32_t input_columns, uint32_t output_rows,
        uint32_t weight_block_rows, uint32_t weight_block_columns,
        uint32_t scale_block_rows, uint32_t scale_block_columns,
        uint32_t group_size, uint64_t scratch_bytes_per_row,
        uint32_t max_rows) {
    if (!input_columns || !output_rows || !weight_block_rows ||
        !weight_block_columns || !scale_block_rows || !scale_block_columns ||
        !max_rows)
        return 0;
    uint64_t ignored;
    if (!coli_quant_u64_mul_ok(input_columns, output_rows, &ignored) ||
        !coli_quant_u64_mul_ok(weight_block_rows, weight_block_columns, &ignored) ||
        !coli_quant_u64_mul_ok(scale_block_rows, scale_block_columns, &ignored) ||
        !coli_quant_u64_mul_ok(scratch_bytes_per_row, max_rows, &ignored))
        return 0;
    if (group_size > input_columns) return 0;
    return 1;
}

static inline int coli_quant_kernel_record_valid(
        const ColiQuantKernelRecordDesc *record) {
    if (!record || record->descriptor_version != COLI_QUANT_KERNEL_DESCRIPTOR_VERSION ||
        !coli_quant_format_known(record->quant_format) ||
        !coli_scale_format_known(record->scale_format) ||
        !coli_activation_known(record->activation) ||
        !coli_exec_layout_known(record->layout_abi_id) ||
        !coli_kernel_abi_known(record->kernel_abi_id) ||
        record->layout_abi_version != COLI_EXEC_LAYOUT_ABI_VERSION ||
        record->kernel_abi_version != COLI_QUANT_KERNEL_ABI_VERSION ||
        !coli_quant_power_of_two_u32(record->required_alignment) ||
        !record->min_rows || record->max_rows < record->min_rows ||
        !coli_quant_kernel_geometry_valid(
            record->input_columns, record->output_rows,
            record->weight_block_rows, record->weight_block_columns,
            record->scale_block_rows, record->scale_block_columns,
            record->group_size, record->scratch_bytes_per_row,
            record->max_rows))
        return 0;

    /* Pin only the format contracts already present in-tree. Physical packing
     * still lives in layout_abi_id, so future target layouts get new IDs rather
     * than silently reusing these assumptions. */
    switch (record->quant_format) {
        case COLI_QUANT_FORMAT_MXFP4_E2M1:
            return record->scale_format == COLI_SCALE_FORMAT_E8M0_PER_GROUP &&
                record->group_size == 32u &&
                record->layout_abi_id == COLI_EXEC_LAYOUT_MXFP4_E2M1_E8M0_32;
        case COLI_QUANT_FORMAT_INT8:
            return record->scale_format == COLI_SCALE_FORMAT_F32_PER_ROW &&
                record->layout_abi_id == COLI_EXEC_LAYOUT_INT8_ROW_MAJOR;
        case COLI_QUANT_FORMAT_INT4_PACKED:
            return record->scale_format == COLI_SCALE_FORMAT_F32_PER_ROW &&
                record->layout_abi_id == COLI_EXEC_LAYOUT_INT4_PACKED_ROW_MAJOR;
        case COLI_QUANT_FORMAT_INT4_GROUPED:
            return record->scale_format == COLI_SCALE_FORMAT_F32_PER_GROUP &&
                record->group_size != 0u &&
                record->layout_abi_id == COLI_EXEC_LAYOUT_INT4_GROUPED_ROW_MAJOR;
        default:
            return 0;
    }
}

static inline int coli_quant_kernel_request_valid(
        const ColiQuantKernelRequestDesc *request) {
    if (!request || request->descriptor_version != COLI_QUANT_KERNEL_DESCRIPTOR_VERSION ||
        !coli_quant_format_known(request->quant_format) ||
        !coli_scale_format_known(request->scale_format) ||
        !coli_activation_known(request->activation) ||
        !coli_exec_layout_known(request->layout_abi_id) ||
        !coli_kernel_abi_known(request->kernel_abi_id) ||
        request->layout_abi_version != COLI_EXEC_LAYOUT_ABI_VERSION ||
        request->kernel_abi_version != COLI_QUANT_KERNEL_ABI_VERSION ||
        !request->rows || !coli_quant_power_of_two_u32(request->record_alignment) ||
        !coli_quant_kernel_geometry_valid(
            request->input_columns, request->output_rows,
            request->weight_block_rows, request->weight_block_columns,
            request->scale_block_rows, request->scale_block_columns,
            request->group_size, 0, request->rows))
        return 0;
    return 1;
}

static inline int coli_quant_backend_caps_valid(const ColiQuantBackendCaps *caps) {
    if (!caps || !caps->max_rows ||
        !coli_quant_power_of_two_u32(caps->max_supported_alignment))
        return 0;
    return 1;
}

static inline ColiQuantKernelCompat coli_quant_kernel_compatible(
        const ColiQuantKernelRecordDesc *record,
        const ColiQuantBackendCaps *backend,
        const ColiQuantKernelRequestDesc *request) {
    if (!coli_quant_kernel_record_valid(record))
        return COLI_QUANT_KERNEL_COMPAT_INVALID_RECORD;
    if (!coli_quant_kernel_request_valid(request))
        return COLI_QUANT_KERNEL_COMPAT_INVALID_REQUEST;
    if (!coli_quant_backend_caps_valid(backend))
        return COLI_QUANT_KERNEL_COMPAT_INVALID_BACKEND;

    if (record->quant_format != request->quant_format)
        return COLI_QUANT_KERNEL_COMPAT_QUANT_MISMATCH;
    if (record->scale_format != request->scale_format)
        return COLI_QUANT_KERNEL_COMPAT_SCALE_MISMATCH;
    if (record->activation != request->activation)
        return COLI_QUANT_KERNEL_COMPAT_ACTIVATION_MISMATCH;
    if (record->layout_abi_id != request->layout_abi_id ||
        record->layout_abi_version != request->layout_abi_version)
        return COLI_QUANT_KERNEL_COMPAT_LAYOUT_ABI_MISMATCH;
    if (record->kernel_abi_id != request->kernel_abi_id ||
        record->kernel_abi_version != request->kernel_abi_version)
        return COLI_QUANT_KERNEL_COMPAT_KERNEL_ABI_MISMATCH;

    if (record->input_columns != request->input_columns ||
        record->output_rows != request->output_rows ||
        record->weight_block_rows != request->weight_block_rows ||
        record->weight_block_columns != request->weight_block_columns ||
        record->scale_block_rows != request->scale_block_rows ||
        record->scale_block_columns != request->scale_block_columns ||
        record->group_size != request->group_size)
        return COLI_QUANT_KERNEL_COMPAT_GEOMETRY_MISMATCH;

    if (request->record_alignment < record->required_alignment ||
        backend->max_supported_alignment < record->required_alignment)
        return COLI_QUANT_KERNEL_COMPAT_ALIGNMENT_MISMATCH;
    if (request->rows < record->min_rows || request->rows > record->max_rows ||
        request->rows > backend->max_rows)
        return COLI_QUANT_KERNEL_COMPAT_ROW_LIMIT;
    if ((backend->capability_bits & record->required_backend_caps) !=
            record->required_backend_caps ||
        !(backend->supported_activation_bits & COLI_ACTIVATION_BIT(record->activation)))
        return COLI_QUANT_KERNEL_COMPAT_CAPABILITY_MISSING;

    uint64_t scratch;
    if (!coli_quant_u64_mul_ok(record->scratch_bytes_per_row, request->rows,
                               &scratch) ||
        scratch > backend->scratch_bytes_available)
        return COLI_QUANT_KERNEL_COMPAT_SCRATCH_LIMIT;
    return COLI_QUANT_KERNEL_COMPAT_OK;
}

static inline const char *coli_quant_kernel_compat_name(ColiQuantKernelCompat result) {
    switch (result) {
        case COLI_QUANT_KERNEL_COMPAT_OK: return "ok";
        case COLI_QUANT_KERNEL_COMPAT_INVALID_RECORD: return "invalid-record";
        case COLI_QUANT_KERNEL_COMPAT_INVALID_REQUEST: return "invalid-request";
        case COLI_QUANT_KERNEL_COMPAT_INVALID_BACKEND: return "invalid-backend";
        case COLI_QUANT_KERNEL_COMPAT_QUANT_MISMATCH: return "quant-mismatch";
        case COLI_QUANT_KERNEL_COMPAT_SCALE_MISMATCH: return "scale-mismatch";
        case COLI_QUANT_KERNEL_COMPAT_ACTIVATION_MISMATCH: return "activation-mismatch";
        case COLI_QUANT_KERNEL_COMPAT_LAYOUT_ABI_MISMATCH: return "layout-abi-mismatch";
        case COLI_QUANT_KERNEL_COMPAT_KERNEL_ABI_MISMATCH: return "kernel-abi-mismatch";
        case COLI_QUANT_KERNEL_COMPAT_GEOMETRY_MISMATCH: return "geometry-mismatch";
        case COLI_QUANT_KERNEL_COMPAT_ALIGNMENT_MISMATCH: return "alignment-mismatch";
        case COLI_QUANT_KERNEL_COMPAT_ROW_LIMIT: return "row-limit";
        case COLI_QUANT_KERNEL_COMPAT_CAPABILITY_MISSING: return "capability-missing";
        case COLI_QUANT_KERNEL_COMPAT_SCRATCH_LIMIT: return "scratch-limit";
        default: return "unknown";
    }
}

#ifdef __cplusplus
}
#endif

#endif

#include "../quant_kernel_abi.h"

#include <stdio.h>
#include <string.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static ColiQuantKernelRecordDesc base_record(void) {
    ColiQuantKernelRecordDesc d;
    memset(&d, 0, sizeof(d));
    d.descriptor_version = COLI_QUANT_KERNEL_DESCRIPTOR_VERSION;
    d.quant_format = COLI_QUANT_FORMAT_MXFP4_E2M1;
    d.scale_format = COLI_SCALE_FORMAT_E8M0_PER_GROUP;
    d.activation = COLI_ACTIVATION_SWIGLU;
    d.layout_abi_id = COLI_EXEC_LAYOUT_MXFP4_E2M1_E8M0_32;
    d.layout_abi_version = COLI_EXEC_LAYOUT_ABI_VERSION;
    d.kernel_abi_id = COLI_KERNEL_ABI_FUSED_SWIGLU_EXPERT;
    d.kernel_abi_version = COLI_QUANT_KERNEL_ABI_VERSION;
    d.input_columns = 128;
    d.output_rows = 64;
    d.weight_block_rows = 1;
    d.weight_block_columns = 2;
    d.scale_block_rows = 1;
    d.scale_block_columns = 32;
    d.group_size = 32;
    d.required_alignment = 64;
    d.min_rows = 1;
    d.max_rows = 8;
    d.scratch_bytes_per_row = 4096;
    d.required_backend_caps = COLI_BACKEND_CAP_QUANT_GEMV |
                              COLI_BACKEND_CAP_DUAL_GATE_UP |
                              COLI_BACKEND_CAP_FUSED_EXPERT;
    return d;
}

static ColiQuantKernelRequestDesc base_request(void) {
    ColiQuantKernelRequestDesc q;
    memset(&q, 0, sizeof(q));
    q.descriptor_version = COLI_QUANT_KERNEL_DESCRIPTOR_VERSION;
    q.quant_format = COLI_QUANT_FORMAT_MXFP4_E2M1;
    q.scale_format = COLI_SCALE_FORMAT_E8M0_PER_GROUP;
    q.activation = COLI_ACTIVATION_SWIGLU;
    q.layout_abi_id = COLI_EXEC_LAYOUT_MXFP4_E2M1_E8M0_32;
    q.layout_abi_version = COLI_EXEC_LAYOUT_ABI_VERSION;
    q.kernel_abi_id = COLI_KERNEL_ABI_FUSED_SWIGLU_EXPERT;
    q.kernel_abi_version = COLI_QUANT_KERNEL_ABI_VERSION;
    q.input_columns = 128;
    q.output_rows = 64;
    q.weight_block_rows = 1;
    q.weight_block_columns = 2;
    q.scale_block_rows = 1;
    q.scale_block_columns = 32;
    q.group_size = 32;
    q.rows = 2;
    q.record_alignment = 16384;
    return q;
}

static ColiQuantBackendCaps base_backend(void) {
    ColiQuantBackendCaps caps;
    memset(&caps, 0, sizeof(caps));
    caps.capability_bits = COLI_BACKEND_CAP_QUANT_GEMV |
                           COLI_BACKEND_CAP_QUANT_GEMM |
                           COLI_BACKEND_CAP_DUAL_GATE_UP |
                           COLI_BACKEND_CAP_FUSED_EXPERT |
                           COLI_BACKEND_CAP_ROUTED_EXPERT_BLOCK |
                           COLI_BACKEND_CAP_BATCHED_EXPERT_ROWS;
    caps.supported_activation_bits = COLI_ACTIVATION_BIT(COLI_ACTIVATION_SWIGLU) |
                                     COLI_ACTIVATION_BIT(COLI_ACTIVATION_SILU);
    caps.scratch_bytes_available = 1u << 20;
    caps.max_rows = 16;
    caps.max_supported_alignment = 16384;
    return caps;
}

static int test_known_format_contracts(void) {
    ColiQuantKernelRecordDesc d = base_record();
    CHECK(coli_quant_kernel_record_valid(&d));

    d.quant_format = COLI_QUANT_FORMAT_INT8;
    d.scale_format = COLI_SCALE_FORMAT_F32_PER_ROW;
    d.layout_abi_id = COLI_EXEC_LAYOUT_INT8_ROW_MAJOR;
    d.group_size = 0;
    CHECK(coli_quant_kernel_record_valid(&d));

    d.quant_format = COLI_QUANT_FORMAT_INT4_PACKED;
    d.layout_abi_id = COLI_EXEC_LAYOUT_INT4_PACKED_ROW_MAJOR;
    CHECK(coli_quant_kernel_record_valid(&d));

    d.quant_format = COLI_QUANT_FORMAT_INT4_GROUPED;
    d.scale_format = COLI_SCALE_FORMAT_F32_PER_GROUP;
    d.layout_abi_id = COLI_EXEC_LAYOUT_INT4_GROUPED_ROW_MAJOR;
    d.group_size = 64;
    CHECK(coli_quant_kernel_record_valid(&d));
    return 0;
}

int main(void) {
    ColiQuantKernelRecordDesc record = base_record();
    ColiQuantKernelRequestDesc request = base_request();
    ColiQuantBackendCaps backend = base_backend();

    CHECK(test_known_format_contracts() == 0);
    CHECK(coli_quant_kernel_compatible(&record, &backend, &request) ==
          COLI_QUANT_KERNEL_COMPAT_OK);

    ColiQuantKernelRequestDesc q = request;
    q.quant_format = COLI_QUANT_FORMAT_INT8;
    CHECK(coli_quant_kernel_compatible(&record, &backend, &q) ==
          COLI_QUANT_KERNEL_COMPAT_QUANT_MISMATCH);

    q = request;
    q.scale_format = COLI_SCALE_FORMAT_F32_PER_ROW;
    CHECK(coli_quant_kernel_compatible(&record, &backend, &q) ==
          COLI_QUANT_KERNEL_COMPAT_SCALE_MISMATCH);

    q = request;
    q.layout_abi_id = COLI_EXEC_LAYOUT_INT4_GROUPED_ROW_MAJOR;
    CHECK(coli_quant_kernel_compatible(&record, &backend, &q) ==
          COLI_QUANT_KERNEL_COMPAT_LAYOUT_ABI_MISMATCH);

    q = request;
    q.kernel_abi_id = COLI_KERNEL_ABI_ROUTED_EXPERT_BLOCK;
    CHECK(coli_quant_kernel_compatible(&record, &backend, &q) ==
          COLI_QUANT_KERNEL_COMPAT_KERNEL_ABI_MISMATCH);

    /* Version 2 is not registered yet: fail closed before dispatch. */
    q = request;
    q.kernel_abi_version = COLI_QUANT_KERNEL_ABI_VERSION + 1;
    CHECK(coli_quant_kernel_compatible(&record, &backend, &q) ==
          COLI_QUANT_KERNEL_COMPAT_INVALID_REQUEST);

    q = request;
    q.input_columns++;
    CHECK(coli_quant_kernel_compatible(&record, &backend, &q) ==
          COLI_QUANT_KERNEL_COMPAT_GEOMETRY_MISMATCH);

    q = request;
    q.record_alignment = 32;
    CHECK(coli_quant_kernel_compatible(&record, &backend, &q) ==
          COLI_QUANT_KERNEL_COMPAT_ALIGNMENT_MISMATCH);

    q = request;
    q.rows = record.max_rows + 1;
    CHECK(coli_quant_kernel_compatible(&record, &backend, &q) ==
          COLI_QUANT_KERNEL_COMPAT_ROW_LIMIT);

    ColiQuantBackendCaps caps = backend;
    caps.capability_bits &= ~COLI_BACKEND_CAP_DUAL_GATE_UP;
    CHECK(coli_quant_kernel_compatible(&record, &caps, &request) ==
          COLI_QUANT_KERNEL_COMPAT_CAPABILITY_MISSING);

    caps = backend;
    caps.supported_activation_bits = COLI_ACTIVATION_BIT(COLI_ACTIVATION_GELU);
    CHECK(coli_quant_kernel_compatible(&record, &caps, &request) ==
          COLI_QUANT_KERNEL_COMPAT_CAPABILITY_MISSING);

    caps = backend;
    caps.scratch_bytes_available = record.scratch_bytes_per_row * request.rows - 1;
    CHECK(coli_quant_kernel_compatible(&record, &caps, &request) ==
          COLI_QUANT_KERNEL_COMPAT_SCRATCH_LIMIT);

    q = request;
    q.layout_abi_id = 9999;
    CHECK(coli_quant_kernel_compatible(&record, &backend, &q) ==
          COLI_QUANT_KERNEL_COMPAT_INVALID_REQUEST);

    ColiQuantKernelRecordDesc bad = record;
    bad.quant_format = 9999;
    CHECK(coli_quant_kernel_compatible(&bad, &backend, &request) ==
          COLI_QUANT_KERNEL_COMPAT_INVALID_RECORD);

    bad = record;
    bad.required_alignment = 48;
    CHECK(!coli_quant_kernel_record_valid(&bad));

    bad = record;
    bad.group_size = bad.input_columns + 1;
    CHECK(!coli_quant_kernel_record_valid(&bad));

    bad = record;
    bad.scratch_bytes_per_row = UINT64_MAX;
    bad.max_rows = 2;
    CHECK(!coli_quant_kernel_record_valid(&bad));

    q = request;
    q.record_alignment = 3;
    CHECK(!coli_quant_kernel_request_valid(&q));

    CHECK(strcmp(coli_quant_kernel_compat_name(COLI_QUANT_KERNEL_COMPAT_OK), "ok") == 0);
    CHECK(strcmp(coli_quant_kernel_compat_name(COLI_QUANT_KERNEL_COMPAT_CAPABILITY_MISSING),
                 "capability-missing") == 0);

    puts("PASS quant/kernel ABI descriptor compatibility");
    return 0;
}

#include "../apple8_contract.h"
#include "../coli_executor.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

static ColiExecutorOpenOptions good_options(void) {
    ColiExecutorOpenOptions o;
    memset(&o, 0, sizeof(o));
    o.required_profile = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    o.checksum_policy = COLI_CSF_CHECKSUM_MANIFEST_ONLY;
    o.required_execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1;
    o.required_kernel_abi = COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1;
    o.required_target_class = COLI_TARGET_CLASS_APPLE8_METAL_V1;
    return o;
}

static int rejects(ColiExecutorOpenOptions o) {
    ColiExecutor *executor = NULL;
    char error[256] = {0};
    CHECK(coli_executor_open(&executor, "/definitely/not/a/coli/package", &o,
                             error, sizeof(error)) != 0);
    CHECK(executor == NULL);
    return strstr(error, "target contract mismatch") != NULL;
}

int main(void) {
    ColiExecutorOpenOptions o = good_options();
    CHECK(!rejects(o));
    o = good_options(); o.required_execution_layout_abi++;
    CHECK(rejects(o));
    o = good_options(); o.required_kernel_abi++;
    CHECK(rejects(o));
    o = good_options(); o.required_target_class++;
    CHECK(rejects(o));
    o = good_options(); o.required_execution_layout_abi = 0;
    o.required_kernel_abi = 0; o.required_target_class = 0;
    CHECK(rejects(o));
    puts("test_apple8_executor_contract: ok");
    return 0;
}

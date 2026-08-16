#!/usr/bin/env python3
from pathlib import Path


def rep(path: str, old: str, new: str, count: int = 1) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"missing anchor in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, count))


# ---------------------------------------------------------------------------
# Legacy zero-copy/data-access reader: v1.1 target compatibility is checked by
# ColiExecutor before this parser runs, but the parser still has to understand
# the v1.1 container/envelope framing used by production compiler output.
# ---------------------------------------------------------------------------
rep("c/coli_format.h", "#define COLI_CSF_VERSION_MINOR 0u", "#define COLI_CSF_VERSION_MINOR 1u")

p = Path("c/coli_format.c")
t = p.read_text()
t = t.replace(
    "#define CSF_MANIFEST_REQUIRED_FLAG_MASK UINT32_C(0xffff0000)",
    "#define CSF_MANIFEST_REQUIRED_FLAG_MASK UINT32_C(0xffff0000)\n#define CSF_MANIFEST_F_TARGET_DESC_VALID (UINT32_C(1) << 16)",
    1,
)
t = t.replace(
    "memcmp(h, k_manifest_magic, 8) || rd16(h + 8) != 1 || rd16(h + 10) > 0 ||",
    "memcmp(h, k_manifest_magic, 8) || rd16(h + 8) != 1 || rd16(h + 10) > COLI_CSF_VERSION_MINOR ||",
    1,
)
t = t.replace(
'''    flags = rd32(h + 16);
    if (flags & CSF_MANIFEST_REQUIRED_FLAG_MASK) {
        csf_error(error, error_size, "manifest contains unknown required feature flags"); return -1;
    }''',
'''    flags = rd32(h + 16);
    if (rd16(h + 10) == 0) {
        if (flags & CSF_MANIFEST_REQUIRED_FLAG_MASK) {
            csf_error(error, error_size, "manifest contains unknown required feature flags"); return -1;
        }
    } else {
        uint32_t required = flags & CSF_MANIFEST_REQUIRED_FLAG_MASK;
        if ((required & ~CSF_MANIFEST_F_TARGET_DESC_VALID) ||
            !(required & CSF_MANIFEST_F_TARGET_DESC_VALID)) {
            csf_error(error, error_size, "v1.1 manifest target feature flags are invalid"); return -1;
        }
    }''',
    1,
)
t = t.replace(
'''    if (!all_zero(h + 184, 72)) {
        csf_error(error, error_size, "manifest reserved extension fields are nonzero"); return -1;
    }''',
'''    if (rd16(h + 10) == 0) {
        if (!all_zero(h + 184, 72)) {
            csf_error(error, error_size, "manifest reserved extension fields are nonzero"); return -1;
        }
    } else if (!all_zero(h + 236, 20)) {
        csf_error(error, error_size, "v1.1 manifest reserved extension fields are nonzero"); return -1;
    }''',
    1,
)
# Tensor/expert typed envelopes must be coherent with the package minor.
t = t.replace("rd16(h + 10) > 0", "rd16(h + 10) != p->version_minor", 2)
p.write_text(t)

# ---------------------------------------------------------------------------
# Production executor: strict target identity/capability preflight before the
# older package reader opens any shard/model payload.
# ---------------------------------------------------------------------------
rep("c/coli_executor.h", '#include "coli_format.h"', '#include "coli_format.h"\n#include "coli_target.h"')
rep(
    "c/coli_executor.h",
'''typedef struct ColiExecutorOpenOptions {
    const char *required_profile;
    ColiCsfChecksumPolicy checksum_policy;
    uint64_t max_resident_record_bytes;
} ColiExecutorOpenOptions;''',
'''typedef struct ColiExecutorOpenOptions {
    const char *required_profile;
    const ColiRuntimeTarget *runtime_target;
    ColiCsfChecksumPolicy checksum_policy;
    uint64_t max_resident_record_bytes;
} ColiExecutorOpenOptions;''',
)
rep("c/coli_executor.c", '#include "coli_executor.h"', '#include "coli_executor.h"\n#include "coli_target.h"')
rep(
    "c/coli_executor.c",
'''    policy = options->checksum_policy;
    if (policy != COLI_CSF_CHECKSUM_MANIFEST_ONLY &&''',
'''    if (options->runtime_target) {
        ColiTargetInfo required_target;
        memset(&required_target, 0, sizeof(required_target));
        if (coli_target_read_package(package_path, &required_target, error, error_size))
            return -1;
        if (options->required_profile &&
            strcmp(required_target.profile_name, options->required_profile)) {
            executor_error(error, error_size,
                           "package target profile %s does not match required %s",
                           required_target.profile_name, options->required_profile);
            coli_target_info_free(&required_target);
            return -1;
        }
        if (coli_target_check_compatibility(&required_target, options->runtime_target,
                                            error, error_size)) {
            coli_target_info_free(&required_target);
            return -1;
        }
        coli_target_info_free(&required_target);
    }
    policy = options->checksum_policy;
    if (policy != COLI_CSF_CHECKSUM_MANIFEST_ONLY &&''',
)

# ---------------------------------------------------------------------------
# Metal capability discovery used by the production V4 store. We only need a
# lower-bound test because Apple8 target compatibility is monotonic.
# ---------------------------------------------------------------------------
rep("c/backend_metal.h", "int  coli_metal_init(void);", "int  coli_metal_init(void);\nint  coli_metal_supports_apple8(void);")
p = Path("c/backend_metal.mm")
t = p.read_text()
anchor = '''extern "C" int coli_metal_register(void *ptr, size_t bytes) {'''
if anchor not in t:
    raise SystemExit("backend_metal register anchor missing")
t = t.replace(anchor, '''extern "C" int coli_metal_supports_apple8(void) {
    if (!coli_metal_init() || !g_dev) return 0;
    return [g_dev supportsFamily:MTLGPUFamilyApple8] ? 1 : 0;
}

extern "C" int coli_metal_register(void *ptr, size_t bytes) {''', 1)
p.write_text(t)

# ---------------------------------------------------------------------------
# V4 expert store: construct actual host capability descriptor, feed it to the
# executor, and accept the concrete target-native MXFP4 row32 layouts.
# ---------------------------------------------------------------------------
rep(
    "c/coli_v4_expert_store.c",
    '#include "coli_executor.h"',
    '#include "coli_executor.h"\n#include "coli_target.h"\n#include "coli_target_profiles.h"',
)
rep(
    "c/coli_v4_expert_store.c",
'''static Slot *slots_for(State*s,int layer) { return s->slots+(size_t)layer*s->slots_per_layer; }
static int tensor_format''',
'''static Slot *slots_for(State*s,int layer) { return s->slots+(size_t)layer*s->slots_per_layer; }

static int runtime_target_for_profile(const char *profile, ColiRuntimeTarget *runtime,
                                      char *error, size_t error_size) {
    if (!profile || !runtime) return fail(error, error_size, "invalid runtime target request");
    memset(runtime, 0, sizeof(*runtime));
    if (!strcmp(profile, COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1)) {
#if defined(__APPLE__) && defined(COLI_METAL)
        if (!coli_metal_supports_apple8())
            return fail(error, error_size, "Apple8 Metal target requires an Apple8-or-newer Metal GPU");
        runtime->target_os = COLI_TARGET_OS_MACOS;
        runtime->target_arch = COLI_TARGET_ARCH_ARM64;
        runtime->backend = COLI_TARGET_BACKEND_METAL;
        runtime->gpu_kind = COLI_TARGET_GPU_APPLE_FAMILY;
        runtime->cpu_feature_mask = COLI_TARGET_CPU_ARM64_ASIMD;
        runtime->gpu_family = 8;
        runtime->runtime_features = COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY |
                                    COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE;
        runtime->semantic_abi = "deepseek-v4-exec-v1";
        runtime->profile_name = COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
        runtime->quant_profile = "exact";
        runtime->storage_profile = "none";
        runtime->target_profile_abi = COLI_TARGET_PROFILE_ABI_V1;
        runtime->execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_V1;
        runtime->kernel_abi = COLI_KERNEL_ABI_V1;
        runtime->max_record_alignment = 16 * 1024;
        runtime->max_io_granularity = 16 * 1024;
        runtime->max_resident_alignment = 16 * 1024;
        return 0;
#else
        return fail(error, error_size, "Apple8 Metal target is unavailable in this build");
#endif
    }
    if (!strcmp(profile, COLI_PROFILE_LINUX_X86_64_CPU_AVX2_V1)) {
#if defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))
            return fail(error, error_size, "x86_64 CPU target requires AVX2+FMA");
        runtime->target_os = COLI_TARGET_OS_LINUX;
        runtime->target_arch = COLI_TARGET_ARCH_X86_64;
        runtime->backend = COLI_TARGET_BACKEND_CPU;
        runtime->gpu_kind = COLI_TARGET_GPU_NONE;
        runtime->cpu_feature_mask = COLI_TARGET_CPU_X86_AVX2 | COLI_TARGET_CPU_X86_FMA;
        runtime->semantic_abi = "deepseek-v4-exec-v1";
        runtime->profile_name = COLI_PROFILE_LINUX_X86_64_CPU_AVX2_V1;
        runtime->quant_profile = "exact";
        runtime->storage_profile = "none";
        runtime->target_profile_abi = COLI_TARGET_PROFILE_ABI_V1;
        runtime->execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_V1;
        runtime->kernel_abi = COLI_KERNEL_ABI_V1;
        runtime->max_record_alignment = 4096;
        runtime->max_io_granularity = 4096;
        runtime->max_resident_alignment = 64;
        return 0;
#else
        return fail(error, error_size, "linux x86_64 AVX2 target is unavailable on this host");
#endif
    }
    return fail(error, error_size, "unsupported production COLI target profile: %s", profile);
}

static int tensor_format''',
)
rep(
    "c/coli_v4_expert_store.c",
'''    if(m->math_format!=COLI_CSF_MATH_MXFP4_E2M1 || m->scale_format!=COLI_CSF_SCALE_UE8M0 ||
       m->layout!=COLI_CSF_LAYOUT_CANONICAL || m->weight_codec!=COLI_CSF_CODEC_NONE ||''',
'''    if(m->math_format!=COLI_CSF_MATH_MXFP4_E2M1 || m->scale_format!=COLI_CSF_SCALE_UE8M0 ||
       (m->layout!=COLI_LAYOUT_APPLE_MXFP4_ROW32_V1 &&
        m->layout!=COLI_LAYOUT_X86_MXFP4_ROW32_V1) ||
       !coli_target_layout_accepts_format(m->layout,m->math_format,m->scale_format,
                                          m->scale_block_rows,m->scale_block_columns,m->group_size) ||
       m->weight_codec!=COLI_CSF_CODEC_NONE ||''',
)
rep(
    "c/coli_v4_expert_store.c",
'''int coli_v4_coli_expert_store_open(const ColiV4ColiExpertStoreOptions*o,ColiExpertStore**out,char*e,size_t n) {
    static const ColiExpertStoreOps ops={lookup,release,prefetch,stats,destroy}; ColiExpertStore*store=NULL;State*s=NULL;ColiExecutorOpenOptions xo={0};''',
'''int coli_v4_coli_expert_store_open(const ColiV4ColiExpertStoreOptions*o,ColiExpertStore**out,char*e,size_t n) {
    static const ColiExpertStoreOps ops={lookup,release,prefetch,stats,destroy}; ColiExpertStore*store=NULL;State*s=NULL;ColiExecutorOpenOptions xo={0}; ColiRuntimeTarget runtime;''',
)
rep(
    "c/coli_v4_expert_store.c",
'''    xo.required_profile=o->required_profile;
    xo.checksum_policy=getenv("COLI_VERIFY_RECORDS")''',
'''    if(runtime_target_for_profile(o->required_profile,&runtime,e,n))goto bad;
    xo.required_profile=o->required_profile;
    xo.runtime_target=&runtime;
    xo.checksum_policy=getenv("COLI_VERIFY_RECORDS")''',
)

# ---------------------------------------------------------------------------
# Build surfaces: production executor/store now depend on target validation.
# ---------------------------------------------------------------------------
rep(
    "c/Makefile.deepseek-v4",
    "V4_COLI_OBJS = coli_v4_expert_store.o coli_v4_static.o coli_executor.o coli_format.o",
    "V4_COLI_OBJS = coli_v4_expert_store.o coli_v4_static.o coli_executor.o coli_format.o coli_target.o coli_target_profiles.o",
)
rep(
    "c/Makefile.deepseek-v4",
'''coli_v4_expert_store.o: coli_v4_expert_store.c coli_v4_expert_store.h coli_executor.h coli_format.h expert_store.h tensor.h
	$(CC) $(CFLAGS) -c coli_v4_expert_store.c -o $@''',
'''coli_v4_expert_store.o: coli_v4_expert_store.c coli_v4_expert_store.h coli_executor.h coli_format.h coli_target.h coli_target_profiles.h expert_store.h tensor.h
	$(CC) $(CFLAGS) -c coli_v4_expert_store.c -o $@''',
)
rep(
    "c/Makefile.deepseek-v4",
'''coli_executor.o: coli_executor.c coli_executor.h coli_format.h
	$(CC) $(CFLAGS) -c coli_executor.c -o $@
coli_format.o: coli_format.c coli_format.h compat.h
	$(CC) $(CFLAGS) -c coli_format.c -o $@''',
'''coli_executor.o: coli_executor.c coli_executor.h coli_format.h coli_target.h
	$(CC) $(CFLAGS) -c coli_executor.c -o $@
coli_format.o: coli_format.c coli_format.h compat.h
	$(CC) $(CFLAGS) -c coli_format.c -o $@
coli_target.o: coli_target.c coli_target.h
	$(CC) $(CFLAGS) -c coli_target.c -o $@
coli_target_profiles.o: coli_target_profiles.c coli_target_profiles.h coli_target.h coli_format.h
	$(CC) $(CFLAGS) -c coli_target_profiles.c -o $@''',
)

# CSF standalone test target links the executor's new target preflight dependency.
p = Path("c/Makefile.csf")
t = p.read_text()
t = t.replace(
    "tests/test_coli_format.c coli_format.c coli_executor.c coli_format.h coli_executor.h compat.h",
    "tests/test_coli_format.c coli_format.c coli_executor.c coli_target.c coli_format.h coli_executor.h coli_target.h compat.h",
)
t = t.replace(
    "tests/test_coli_format.c coli_format.c coli_executor.c -o $@ -pthread",
    "tests/test_coli_format.c coli_format.c coli_executor.c coli_target.c -o $@ -pthread",
)
p.write_text(t)

# ---------------------------------------------------------------------------
# #53: prove the same v1.1 package opens through the production ColiExecutor,
# including capability mismatch rejection, before the independent strict reader
# and kernel oracle run.
# ---------------------------------------------------------------------------
p = Path("c/tests/test_colic_e2e.c")
t = p.read_text()
t = t.replace('#include "../coli_exec_format.h"', '#include "../coli_exec_format.h"\n#include "../coli_executor.h"', 1)
anchor = '''    incompatible.gpu_family = 7;
    if (coli_exec_package_open(&package, package_path, &incompatible, error, sizeof(error)) == 0) {'''
if anchor not in t:
    raise SystemExit("e2e incompatible target anchor missing")
insert = '''    incompatible.gpu_family = 7;
    {
        ColiExecutor *executor = NULL;
        ColiExecutorOpenOptions options;
        memset(&options, 0, sizeof(options));
        options.required_profile = COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
        options.runtime_target = &incompatible;
        options.checksum_policy = COLI_CSF_CHECKSUM_MANIFEST_ONLY;
        if (coli_executor_open(&executor, package_path, &options, error, sizeof(error)) == 0) {
            coli_executor_close(executor);
            return fail("production executor accepted incompatible Apple GPU family", NULL);
        }
    }
    error[0] = '\\0';
    if (coli_exec_package_open(&package, package_path, &incompatible, error, sizeof(error)) == 0) {'''
t = t.replace(anchor, insert, 1)
anchor2 = '''    package = NULL;
    error[0] = '\\0';

    if (coli_exec_package_open_ex(&package, package_path, &runtime,'''
if anchor2 not in t:
    raise SystemExit("e2e compatible open anchor missing")
insert2 = '''    package = NULL;
    error[0] = '\\0';

    {
        ColiExecutor *executor = NULL;
        ColiExecutorOpenOptions options;
        ColiExpertInfo production_info;
        const ColiRecordInfo *production_record;
        unsigned char *production_bytes;
        memset(&options, 0, sizeof(options));
        options.required_profile = COLI_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
        options.runtime_target = &runtime;
        options.checksum_policy = COLI_CSF_CHECKSUM_RECORD_ON_READ;
        if (coli_executor_open(&executor, package_path, &options, error, sizeof(error)))
            return fail("production executor strict open failed", error);
        production_record = coli_executor_expert(executor, 0, 0);
        if (!production_record ||
            coli_executor_expert_info(executor, 0, 0, &production_info, error, sizeof(error))) {
            coli_executor_close(executor);
            return fail("production executor expert index/parse failed", error);
        }
        if (production_info.matrices[0].layout != COLI_LAYOUT_APPLE_MXFP4_ROW32_V1 ||
            production_info.matrices[1].layout != COLI_LAYOUT_APPLE_MXFP4_ROW32_V1 ||
            production_info.matrices[2].layout != COLI_LAYOUT_APPLE_MXFP4_ROW32_V1) {
            coli_executor_close(executor);
            return fail("production executor lost target-native matrix layout", NULL);
        }
        if (production_record->stored_bytes > SIZE_MAX) {
            coli_executor_close(executor);
            return fail("production executor record too large", NULL);
        }
        production_bytes = (unsigned char *)malloc((size_t)production_record->stored_bytes);
        if (!production_bytes || coli_executor_load_expert(
                executor, 0, 0, production_bytes, (size_t)production_record->stored_bytes,
                error, sizeof(error))) {
            free(production_bytes);
            coli_executor_close(executor);
            return fail("production executor expert load failed", error);
        }
        free(production_bytes);
        coli_executor_close(executor);
    }
    error[0] = '\\0';

    if (coli_exec_package_open_ex(&package, package_path, &runtime,'''
t = t.replace(anchor2, insert2, 1)
p.write_text(t)

# E2E build now includes production executor/data-access reader.
p = Path("tools/colic_e2e.sh")
t = p.read_text()
t = t.replace(
'''  c/coli_exec_format.c \\
  c/coli_target.c \\
  c/coli_target_profiles.c \\''',
'''  c/coli_exec_format.c \\
  c/coli_executor.c \\
  c/coli_format.c \\
  c/coli_target.c \\
  c/coli_target_profiles.c \\''',
)
p.write_text(t)

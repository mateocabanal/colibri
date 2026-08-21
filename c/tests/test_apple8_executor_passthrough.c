#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "../apple8_contract.h"
#include "../coli_executor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
#define ALIGNMENT 16384u
#define RECORD_BYTES 872u
#define RECORD_OFFSET ALIGNMENT

static void wr16(unsigned char *p, uint16_t v) { p[0]=(unsigned char)v; p[1]=(unsigned char)(v>>8); }
static void wr32(unsigned char *p, uint32_t v) { p[0]=(unsigned char)v; p[1]=(unsigned char)(v>>8); p[2]=(unsigned char)(v>>16); p[3]=(unsigned char)(v>>24); }
static void wr64(unsigned char *p, uint64_t v) { wr32(p,(uint32_t)v); wr32(p+4,(uint32_t)(v>>32)); }

static int write_file(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (bytes && fwrite(data, 1, bytes, f) != bytes) { fclose(f); return -1; }
    return fclose(f) ? -1 : 0;
}

static char *join2(const char *a, const char *b) {
    size_t na = strlen(a), nb = strlen(b);
    char *p = (char *)malloc(na + nb + 2);
    if (!p) return NULL;
    memcpy(p, a, na); p[na]='/'; memcpy(p+na+1, b, nb+1);
    return p;
}

static void rewrite_crc(unsigned char *bytes, size_t n, size_t field) {
    uint32_t crc;
    memset(bytes + field, 0, 4);
    crc = coli_crc32c(bytes, n);
    wr32(bytes + field, crc);
}

static int build_package(const char *dir) {
    const size_t data_bytes = RECORD_OFFSET + RECORD_BYTES;
    const size_t string_off = 416, string_bytes = 112, manifest_bytes = 528;
    unsigned char *data = (unsigned char *)calloc(1, data_bytes);
    unsigned char *manifest = (unsigned char *)calloc(1, manifest_bytes);
    unsigned char *record;
    char *mp = NULL, *dp = NULL;
    uint32_t shard_crc;
    int i, rr, b;
    if (!data || !manifest) goto fail;

    memcpy(data, "COLIDAT\0", 8);
    wr16(data+8, 1); wr16(data+10, 0); wr32(data+12, 128);
    wr32(data+20, 0); wr32(data+24, ALIGNMENT); wr64(data+32, data_bytes);

    record = data + RECORD_OFFSET;
    memcpy(record, "COLIEXPT", 8);
    wr16(record+8, 1); wr16(record+10, 0); wr32(record+12, 64);
    wr32(record+16, 2); wr32(record+20, 7); wr16(record+24, 3);
    wr32(record+28, 128); wr64(record+32, 64); wr64(record+40, 448); wr64(record+48, 408);

    for (i = 0; i < 3; ++i) {
        const uint64_t payload_off = 448u + (uint64_t)i * 144u;
        unsigned char *d = record + 64 + (size_t)i * 128;
        unsigned char *tile = record + payload_off;
        wr16(d, (uint16_t)(i + 1));
        wr16(d+4, COLI_CSF_MATH_MXFP4_E2M1);
        wr16(d+6, COLI_CSF_SCALE_UE8M0);
        wr16(d+8, COLI_CSF_CODEC_NONE);
        wr16(d+10, COLI_CSF_CODEC_NONE);
        wr16(d+12, COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
        wr64(d+16, 8); wr64(d+24, 32); wr32(d+32, 1); wr32(d+36, 32);
        wr64(d+48, payload_off); wr64(d+56, 136); wr64(d+64, 136);
        for (rr = 0; rr < 8; ++rr) {
            for (b = 0; b < 16; ++b)
                tile[rr * 16 + b] = (unsigned char)(1 + i * 37 + rr * 3 + b);
            tile[128 + rr] = (unsigned char)(0x70 + i * 8 + rr);
        }
        wr32(d+96, coli_crc32c(tile, 136));
    }

    rewrite_crc(data, 128, 72);
    shard_crc = (uint32_t)data[72] | ((uint32_t)data[73] << 8) |
                ((uint32_t)data[74] << 16) | ((uint32_t)data[75] << 24);

    memcpy(manifest, "COLI\r\n\x1a\n", 8);
    wr16(manifest+8, 1); wr16(manifest+10, 0); wr32(manifest+12, 256);
    wr32(manifest+20, 0x01020304u); wr32(manifest+24, ALIGNMENT);
    wr32(manifest+28, 3); wr64(manifest+32, 1); wr32(manifest+40, 1);
    wr64(manifest+48, 256); wr64(manifest+56, 64);
    wr64(manifest+64, 320); wr64(manifest+72, 96);
    wr64(manifest+80, string_off); wr64(manifest+88, string_bytes);
    wr32(manifest+148, 1); wr32(manifest+152, 2);

    wr32(manifest+256, 0); wr32(manifest+264, 0);
    wr64(manifest+272, data_bytes); wr32(manifest+280, shard_crc);

    wr64(manifest+320, 1); wr16(manifest+328, COLI_CSF_REC_EXPERT);
    wr16(manifest+330, COLI_CSF_CODEC_NONE); wr16(manifest+332, COLI_CSF_MATH_MIXED);
    wr16(manifest+334, COLI_CSF_SCALE_MIXED); wr16(manifest+336, COLI_CSF_LAYOUT_MIXED);
    wr32(manifest+340, 0); wr32(manifest+344, UINT32_MAX);
    wr32(manifest+348, 2); wr32(manifest+352, 7);
    wr64(manifest+360, RECORD_OFFSET); wr64(manifest+368, RECORD_BYTES); wr64(manifest+376, 408);
    wr32(manifest+384, coli_crc32c(record, RECORD_BYTES));

    {
        unsigned char *s = manifest + string_off;
        const char *names[3] = {"data-00000.coli", COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1, "test-executor-passthrough"};
        size_t cursor = 48;
        for (i = 0; i < 3; ++i) {
            size_t n = strlen(names[i]);
            wr64(s + i * 16, cursor); wr32(s + i * 16 + 8, (uint32_t)n);
            memcpy(s + cursor, names[i], n); cursor += n;
        }
    }
    rewrite_crc(manifest, manifest_bytes, 144);

    mp = join2(dir, "manifest.coli"); dp = join2(dir, "data-00000.coli");
    if (!mp || !dp || write_file(mp, manifest, manifest_bytes) || write_file(dp, data, data_bytes)) goto fail;
    free(mp); free(dp); free(manifest); free(data); return 0;
fail:
    free(mp); free(dp); free(manifest); free(data); return -1;
}

static ColiRuntimeTarget good_runtime(void) {
    ColiRuntimeTarget r;
    memset(&r, 0, sizeof(r));
    r.profile_name = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    r.target_os = COLI_TARGET_OS_MACOS;
    r.target_arch = COLI_TARGET_ARCH_ARM64;
    r.backend = COLI_TARGET_BACKEND_METAL;
    r.gpu_kind = COLI_TARGET_GPU_APPLE_FAMILY;
    r.cpu_feature_mask = COLI_TARGET_CPU_ARM64_ASIMD;
    r.gpu_family = COLI_APPLE8_GPU_FAMILY_MIN;
    r.runtime_features = COLI_TARGET_RUNTIME_APPLE_UNIFIED_MEMORY |
                         COLI_TARGET_RUNTIME_METAL_SHARED_STORAGE;
    r.target_profile_abi = COLI_TARGET_PROFILE_ABI_APPLE8_V1;
    r.execution_layout_abi = COLI_EXECUTION_LAYOUT_ABI_APPLE8_V1;
    r.kernel_abi = COLI_KERNEL_ABI_APPLE8_MXFP4_TILE_V1;
    r.target_class = COLI_TARGET_CLASS_APPLE8_METAL_V1;
    r.max_record_alignment = COLI_APPLE8_RECORD_ALIGNMENT;
    r.max_io_granularity = COLI_APPLE8_IO_GRANULARITY;
    r.max_resident_alignment = COLI_APPLE8_RESIDENT_ALIGNMENT;
    return r;
}

static void cleanup(const char *dir) {
    char *p = join2(dir, "manifest.coli"); if (p) { unlink(p); free(p); }
    p = join2(dir, "data-00000.coli"); if (p) { unlink(p); free(p); }
    rmdir(dir);
}

int main(void) {
    char dir[] = "apple8_exec_passthrough_XXXXXX";
    char error[256] = {0};
    ColiRuntimeTarget runtime = good_runtime();
    ColiExecutorOpenOptions options;
    ColiExecutor *executor = NULL;
    ColiExpertInfo info;
    const ColiRecordInfo *record;
    unsigned char resident[RECORD_BYTES];
    unsigned char physical[RECORD_BYTES];
    char *data_path = NULL;
    FILE *data_file = NULL;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(build_package(dir) == 0);
    memset(&options, 0, sizeof(options));
    options.required_profile = COLI_TARGET_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    options.checksum_policy = COLI_CSF_CHECKSUM_RECORD_ON_READ;
    options.runtime_target = &runtime;
    options.max_resident_record_bytes = RECORD_BYTES;
    CHECK(coli_executor_open(&executor, dir, &options, error, sizeof(error)) == 0);
    CHECK(executor != NULL);
    record = coli_executor_expert(executor, 2, 7);
    CHECK(record != NULL);
    CHECK(record->stored_bytes == RECORD_BYTES);
    CHECK(coli_executor_expert_info(executor, 2, 7, &info, error, sizeof(error)) == 0);
    for (int i = 0; i < 3; ++i) {
        CHECK(info.matrices[i].layout == COLI_LAYOUT_APPLE_MXFP4_TILE8X32_V1);
        CHECK(info.matrices[i].scale_offset == 0);
        CHECK(info.matrices[i].scale_stored_bytes == 0);
        CHECK(info.matrices[i].scale_decoded_bytes == 0);
    }
    memset(resident, 0xa5, sizeof(resident));
    CHECK(coli_executor_load_expert(executor, 2, 7, resident, sizeof(resident), error, sizeof(error)) == 0);

    data_path = join2(dir, "data-00000.coli");
    CHECK(data_path != NULL);
    data_file = fopen(data_path, "rb");
    CHECK(data_file != NULL);
    CHECK(fseeko(data_file, (off_t)record->payload_offset, SEEK_SET) == 0);
    CHECK(fread(physical, 1, sizeof(physical), data_file) == sizeof(physical));
    CHECK(fclose(data_file) == 0); data_file = NULL;
    CHECK(memcmp(resident, physical, sizeof(resident)) == 0);

    printf("APPLE8_EXECUTOR_PASSTHROUGH layout=0x%04x transforms=0 bytes=%u\n",
           info.matrices[0].layout, (unsigned)RECORD_BYTES);
    free(data_path);
    coli_executor_close(executor);
    cleanup(dir);
    return 0;
}

#import <Foundation/Foundation.h>

#include "../apple8_contract.h"
#include "../apple8_metalio_direct.h"
#include "../coli_format.h"
#include "../metalio.h"
#include "../mxfp4_expert.h"
#include "../mxfp4_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

static size_t align16(size_t n) { return (n + 15u) & ~(size_t)15u; }

static const ColiExpertMatrixInfo *role(const ColiExpertInfo &info, uint16_t wanted) {
    for (const auto &m : info.matrices) if (m.role == wanted) return &m;
    return nullptr;
}

static bool write_all(int fd, const uint8_t *p, size_t n) {
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return false;
        p += (size_t)w;
        n -= (size_t)w;
    }
    return true;
}

static void fill_input(std::vector<float> &x, int hidden) {
    for (size_t i = 0; i < x.size(); ++i) {
        int row = (int)(i / (size_t)hidden);
        int col = (int)(i % (size_t)hidden);
        int v = ((col * 17 + row * 29 + 11) % 257) - 128;
        x[i] = (float)v * (1.0f / 256.0f);
    }
}

static int compare_batch(int slot,
                         size_t gate_off, size_t gate_bytes,
                         size_t up_off, size_t up_bytes,
                         size_t down_off, size_t down_bytes,
                         const ColiMxfp4ExpertBuffers &b,
                         int hidden, int intermediate, int rows) {
    std::vector<float> x((size_t)rows * (size_t)hidden);
    std::vector<float> ref((size_t)rows * (size_t)hidden, 0.0f);
    std::vector<float> got((size_t)rows * (size_t)hidden, 0.0f);
    std::vector<float> gate((size_t)rows * (size_t)intermediate);
    std::vector<float> up((size_t)rows * (size_t)intermediate);
    std::vector<float> h((size_t)rows * (size_t)intermediate);
    std::vector<float> y((size_t)rows * (size_t)hidden);
    fill_input(x, hidden);

    coli_mxfp4_swiglu_expert(ref.data(), x.data(),
                             b.gate_weights, b.gate_scales,
                             b.up_weights, b.up_scales,
                             b.down_weights, b.down_scales,
                             rows, hidden, intermediate, 1.0f,
                             gate.data(), up.data(), h.data(), y.data());

    if (!coli_apple8_metalio_swiglu_slot(slot,
                                          gate_off, gate_bytes,
                                          up_off, up_bytes,
                                          down_off, down_bytes,
                                          x.data(), got.data(), rows,
                                          hidden, intermediate)) {
        std::fprintf(stderr, "direct Apple8 dispatch failed for rows=%d\n", rows);
        return 1;
    }

    double se = 0.0, sr = 0.0;
    float max_abs = 0.0f, max_rel = 0.0f;
    size_t max_i = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
            std::fprintf(stderr, "non-finite result rows=%d index=%zu ref=%g got=%g\n",
                         rows, i, ref[i], got[i]);
            return 1;
        }
        float d = std::fabs(got[i] - ref[i]);
        float rel = d / (1.0f + std::fabs(ref[i]));
        if (d > max_abs) { max_abs = d; max_i = i; }
        if (rel > max_rel) max_rel = rel;
        se += (double)d * (double)d;
        sr += (double)ref[i] * (double)ref[i];
    }
    double nrmse = std::sqrt(se / (sr + 1e-30));
    std::printf("APPLE8_REAL_PARITY rows=%d max_abs=%.9g max_rel=%.9g nrmse=%.9g worst=%zu ref=%.9g got=%.9g\n",
                rows, max_abs, max_rel, nrmse, max_i, ref[max_i], got[max_i]);

    /* This is a diagnostic gate, not a bit-exact CPU/GPU requirement.  A direct
     * MXFP4 kernel may sum in a different order, but an error at the percent
     * scale is large enough to alter router trajectories and is not acceptable. */
    return (max_rel > 1.0e-2f || nrmse > 5.0e-3) ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s PACKAGE LAYER EXPERT\n", argv[0]);
        return 2;
    }
    int layer = std::atoi(argv[2]);
    int expert = std::atoi(argv[3]);
    char error[512] = {0};
    ColiPackage *package = nullptr;
    if (coli_package_open_ex(&package, argv[1], COLI_CSF_CHECKSUM_MANIFEST_ONLY,
                             error, sizeof(error))) {
        std::fprintf(stderr, "open: %s\n", error);
        return 1;
    }

    const ColiRecordInfo *record = coli_package_expert(package, layer, expert);
    ColiExpertInfo info{};
    if (!record || coli_package_expert_info(package, record, &info, error, sizeof(error))) {
        std::fprintf(stderr, "expert-info: %s\n", error);
        coli_package_close(package);
        return 1;
    }
    const ColiExpertMatrixInfo *gm = role(info, COLI_MXFP4_EXPERT_ROLE_GATE);
    const ColiExpertMatrixInfo *um = role(info, COLI_MXFP4_EXPERT_ROLE_UP);
    const ColiExpertMatrixInfo *dm = role(info, COLI_MXFP4_EXPERT_ROLE_DOWN);
    if (!gm || !um || !dm || gm->columns > INT32_MAX || gm->rows > INT32_MAX ||
        dm->rows != gm->columns || dm->columns != gm->rows) {
        std::fprintf(stderr, "unexpected expert geometry\n");
        coli_package_close(package);
        return 1;
    }
    int hidden = (int)gm->columns;
    int intermediate = (int)gm->rows;

    uint64_t gb64 = 0, ub64 = 0, db64 = 0;
    if (!coli_apple8_matrix_descriptor_valid(gm, &gb64) ||
        !coli_apple8_matrix_descriptor_valid(um, &ub64) ||
        !coli_apple8_matrix_descriptor_valid(dm, &db64) ||
        gm->weight_codec != COLI_CSF_CODEC_NONE ||
        um->weight_codec != COLI_CSF_CODEC_NONE ||
        dm->weight_codec != COLI_CSF_CODEC_NONE ||
        gb64 > SIZE_MAX || ub64 > SIZE_MAX || db64 > SIZE_MAX) {
        std::fprintf(stderr, "expert is not raw codec-none Apple8\n");
        coli_package_close(package);
        return 1;
    }
    size_t gate_bytes = (size_t)gb64, up_bytes = (size_t)ub64, down_bytes = (size_t)db64;

    ColiMxfp4ExpertLayout layout{};
    if (coli_mxfp4_expert_validate_info(&info, hidden, intermediate, &layout,
                                        error, sizeof(error))) {
        std::fprintf(stderr, "validate: %s\n", error);
        coli_package_close(package);
        return 1;
    }

    std::vector<uint8_t> gw(layout.gate_weight_bytes), gs(layout.gate_scale_bytes);
    std::vector<uint8_t> uw(layout.up_weight_bytes), us(layout.up_scale_bytes);
    std::vector<uint8_t> dw(layout.down_weight_bytes), ds(layout.down_scale_bytes);
    ColiMxfp4ExpertBuffers b{};
    b.gate_weights = gw.data(); b.gate_weight_capacity = gw.size();
    b.gate_scales = gs.data(); b.gate_scale_capacity = gs.size();
    b.up_weights = uw.data(); b.up_weight_capacity = uw.size();
    b.up_scales = us.data(); b.up_scale_capacity = us.size();
    b.down_weights = dw.data(); b.down_weight_capacity = dw.size();
    b.down_scales = ds.data(); b.down_scale_capacity = ds.size();
    if (coli_mxfp4_expert_load(package, record, hidden, intermediate, &b, &layout,
                               error, sizeof(error))) {
        std::fprintf(stderr, "canonical-load: %s\n", error);
        coli_package_close(package);
        return 1;
    }

    size_t gate_off = 0;
    size_t up_off = align16(gate_bytes);
    size_t down_off = align16(up_off + up_bytes);
    size_t file_bytes = down_off + down_bytes;
    std::vector<uint8_t> image(file_bytes, 0);
    if (coli_package_read_range(package, record, gm->weight_offset,
                                image.data() + gate_off, gate_bytes, error, sizeof(error)) ||
        coli_package_read_range(package, record, um->weight_offset,
                                image.data() + up_off, up_bytes, error, sizeof(error)) ||
        coli_package_read_range(package, record, dm->weight_offset,
                                image.data() + down_off, down_bytes, error, sizeof(error))) {
        std::fprintf(stderr, "raw-read: %s\n", error);
        coli_package_close(package);
        return 1;
    }

    char path[] = "/tmp/colibri-apple8-real-parity-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0 || !write_all(fd, image.data(), image.size())) {
        std::perror("fixture-file");
        if (fd >= 0) close(fd);
        unlink(path);
        coli_package_close(package);
        return 1;
    }
    close(fd);

    int rc = 1, file = -1, slot = -1;
    if (!metalio_init() || (file = metalio_file_add(path)) < 0 ||
        (slot = metalio_slot_alloc(file_bytes)) < 0 ||
        !coli_apple8_metalio_direct_init()) {
        std::fprintf(stderr, "MetalIO/direct init failed\n");
        goto done;
    }
    {
        ColiMetalioRegion regions[3] = {
            { file, (uint64_t)gate_off, gate_bytes, gate_off },
            { file, (uint64_t)up_off, up_bytes, up_off },
            { file, (uint64_t)down_off, down_bytes, down_off },
        };
        int64_t ev = metalio_loadv(slot, regions, 3, MIO_LOAD_DEMAND);
        if (ev <= 0 || metalio_wait(ev)) {
            std::fprintf(stderr, "MetalIO load failed\n");
            goto done;
        }
    }

    rc = 0;
    for (int rows : {1, 3, 15})
        rc |= compare_batch(slot, gate_off, gate_bytes, up_off, up_bytes,
                            down_off, down_bytes, b, hidden, intermediate, rows);
    if (!rc) std::puts("APPLE8_REAL_EXPERT_PARITY PASS");

done:
    coli_apple8_metalio_direct_shutdown();
    if (slot >= 0) metalio_slot_free(slot);
    metalio_shutdown();
    unlink(path);
    coli_package_close(package);
    return rc;
}

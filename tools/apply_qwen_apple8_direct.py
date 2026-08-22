#!/usr/bin/env python3
"""Apply the first Qwen raw-Apple8 + MetalIO direct-execution integration.

Temporary development helper for PR #145.  It patches the checked-out branch in
place so the M2 hardware path can be validated before the large qwen_moe.c diff
is folded into the PR.  Every replacement is anchor-checked and idempotent.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, marker: str | None = None) -> None:
    text = path.read_text()
    if marker and marker in text:
        print(f"[apple8-qwen-patch] already patched: {path.relative_to(ROOT)}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path.relative_to(ROOT)}: expected exactly one anchor, found {count}\n"
            f"anchor:\n{old[:500]}"
        )
    path.write_text(text.replace(old, new, 1))
    print(f"[apple8-qwen-patch] patched: {path.relative_to(ROOT)}")


def patch_format_header() -> None:
    path = ROOT / "c" / "coli_format.h"
    old = "uint32_t coli_package_record_alignment(const ColiPackage *package);\n"
    new = old + "/* Package-owned absolute shard path for backend-native async I/O. */\n" \
        "const char *coli_package_shard_path(const ColiPackage *package, uint32_t shard_id);\n"
    replace_once(path, old, new, "coli_package_shard_path")


def patch_format_base() -> None:
    path = ROOT / "c" / "coli_format_base.c"
    old = "uint32_t coli_package_record_alignment(const ColiPackage *p) { return p ? p->record_alignment : 0; }\n"
    new = old + "const char *coli_package_shard_path(const ColiPackage *p, uint32_t shard_id) {\n" \
        "    if (!p || shard_id >= p->shard_count) return NULL;\n" \
        "    return p->shards[shard_id].path;\n" \
        "}\n"
    replace_once(path, old, new, "coli_package_shard_path(const ColiPackage")


def patch_makefile() -> None:
    path = ROOT / "c" / "Makefile"
    old = "METAL_OBJ = backend_metal.o\nMIO_OBJ   = metalio.o\n"
    new = "METAL_OBJ = backend_metal.o\nMIO_OBJ   = metalio.o apple8_metalio_direct.o\n"
    replace_once(path, old, new, "MIO_OBJ   = metalio.o apple8_metalio_direct.o")

    old = "metalio.o: metalio.mm metalio.h\n\t$(METALXX) -c metalio.mm -o $@\n\n"
    new = old + "apple8_metalio_direct.o: apple8_metalio_direct.mm apple8_metalio_direct.h apple8_contract.h metalio.h\n" \
        "\t$(METALXX) -c apple8_metalio_direct.mm -o $@\n\n"
    replace_once(path, old, new, "apple8_metalio_direct.o:")


def patch_qwen() -> None:
    path = ROOT / "c" / "qwen_moe.c"

    old = '#ifdef COLI_METALIO\n#include "metalio.h"\n#endif\n'
    new = '#ifdef COLI_METALIO\n#include "metalio.h"\n#include "apple8_contract.h"\n#include "apple8_metalio_direct.h"\n#endif\n'
    replace_once(path, old, new, 'apple8_metalio_direct.h')

    old = "    int fmt;                       /* 0=f32, 16=BF16, 8=int8, 7=MXFP4, 4=i4-grouped, 5=int3-g64 */\n"
    new = "    int fmt;                       /* 0=f32, 16=BF16, 8=int8, 7=canonical MXFP4, 17=direct Apple8, 4=i4, 5=i3 */\n"
    replace_once(path, old, new, "17=direct Apple8")

    old = "    int64_t mio_waited;            /* highest mio event already waited (pending if < mio_event) */\n"
    new = old + \
        "    int apple8_direct;             /* raw Apple8 payload is executable in mio_slot */\n" \
        "    size_t apple8_gate_off, apple8_gate_bytes;\n" \
        "    size_t apple8_up_off, apple8_up_bytes;\n" \
        "    size_t apple8_down_off, apple8_down_bytes;\n"
    replace_once(path, old, new, "apple8_gate_off")

    old = "    uint64_t prof_expert_loads;    /* physical expert loads from storage */\n"
    new = old + \
        "    uint64_t prof_apple8_direct_blocks;\n" \
        "    uint64_t prof_apple8_direct_experts;\n" \
        "    uint64_t prof_apple8_direct_bytes;\n"
    replace_once(path, old, new, "prof_apple8_direct_blocks")

    old = "    QPC_METAL_MOE_EXPERTS,\n    QPC_COUNT\n};\n"
    new = "    QPC_METAL_MOE_EXPERTS,\n" \
        "    QPC_APPLE8_DIRECT_BLOCKS,\n" \
        "    QPC_APPLE8_DIRECT_EXPERTS,\n" \
        "    QPC_APPLE8_DIRECT_BYTES,\n" \
        "    QPC_COUNT\n};\n"
    replace_once(path, old, new, "QPC_APPLE8_DIRECT_BLOCKS")

    old = "    [QPC_METAL_MOE_EXPERTS] = {\"metal_moe_experts\"},\n};\n"
    new = "    [QPC_METAL_MOE_EXPERTS] = {\"metal_moe_experts\"},\n" \
        "    [QPC_APPLE8_DIRECT_BLOCKS] = {\"apple8_direct_blocks\"},\n" \
        "    [QPC_APPLE8_DIRECT_EXPERTS] = {\"apple8_direct_experts\"},\n" \
        "    [QPC_APPLE8_DIRECT_BYTES] = {\"apple8_direct_bytes\"},\n" \
        "};\n"
    replace_once(path, old, new, "apple8_direct_blocks\"}")

    old = "    coli_profile_counter_set(&g_qprof, QPC_PREFETCH_MISSES, m->prefetch_misses);\n"
    new = old + \
        "    coli_profile_counter_set(&g_qprof, QPC_APPLE8_DIRECT_BLOCKS, m->prof_apple8_direct_blocks);\n" \
        "    coli_profile_counter_set(&g_qprof, QPC_APPLE8_DIRECT_EXPERTS, m->prof_apple8_direct_experts);\n" \
        "    coli_profile_counter_set(&g_qprof, QPC_APPLE8_DIRECT_BYTES, m->prof_apple8_direct_bytes);\n"
    replace_once(path, old, new, "QPC_APPLE8_DIRECT_BYTES, m->prof_apple8_direct_bytes")

    old = "static int g_mio_fd[64], g_mio_fid[64], g_mio_n;\n" \
          "static int mio_file_for(int fd){\n" \
          "    for (int i = 0; i < g_mio_n; i++) if (g_mio_fd[i] == fd) return g_mio_fid[i];\n" \
          "    return -1;\n" \
          "}\n"
    new = old + r'''static int g_apple8_direct = 0;
static const ColiPackage *g_coli_mio_pkg[64];
static uint32_t g_coli_mio_shard[64];
static int g_coli_mio_fid[64], g_coli_mio_n;

static int qwen_coli_mio_file(const ColiPackage *pkg, uint32_t shard_id){
    for (int i = 0; i < g_coli_mio_n; i++)
        if (g_coli_mio_pkg[i] == pkg && g_coli_mio_shard[i] == shard_id)
            return g_coli_mio_fid[i];
    if (!pkg || g_coli_mio_n >= 64) return -1;
    const char *path = coli_package_shard_path(pkg, shard_id);
    if (!path) return -1;
    int fid = metalio_file_add(path);
    if (fid < 0) return -1;
    g_coli_mio_pkg[g_coli_mio_n] = pkg;
    g_coli_mio_shard[g_coli_mio_n] = shard_id;
    g_coli_mio_fid[g_coli_mio_n] = fid;
    g_coli_mio_n++;
    return fid;
}

static int qwen_apple8_raw_matrix(const ColiExpertMatrixInfo *mi,
                                  uint64_t rows, uint64_t cols,
                                  size_t *bytes_out){
    uint64_t bytes = 0;
    if (!mi || mi->rows != rows || mi->columns != cols ||
        !coli_apple8_matrix_descriptor_valid(mi, &bytes) ||
        mi->weight_codec != COLI_CSF_CODEC_NONE || bytes > SIZE_MAX)
        return 0;
    if (bytes_out) *bytes_out = (size_t)bytes;
    return 1;
}

static size_t qwen_align16(size_t n){
    if (n > SIZE_MAX - 15u) return 0;
    return (n + 15u) & ~(size_t)15u;
}
'''
    replace_once(path, old, new, "qwen_coli_mio_file")

    old = "    if (gi < 0 || ui < 0 || di < 0) {\n" \
          "        fprintf(stderr, \"qwen coli: expert (%d,%d) is missing gate/up/down roles\\n\", layer, eid); exit(1);\n" \
          "    }\n\n" \
          "    /* Keep MXFP4 compressed in the streamed cache. The generic loader validates\n"
    direct = r'''    if (gi < 0 || ui < 0 || di < 0) {
        fprintf(stderr, "qwen coli: expert (%d,%d) is missing gate/up/down roles\n", layer, eid); exit(1);
    }

#ifdef COLI_METALIO
    /* Raw Apple8 fast path: the compiler's target-native bytes land directly
     * in the persistent MetalIO slot and remain in Apple8 tile order.  No
     * canonical MXFP4 allocation, detile, or repack is performed.  rANS is
     * deliberately excluded: compressed records still use the validated
     * synchronous decode+detile fallback below. */
    if (g_apple8_direct && g_metal_io && g_metal_compute && metalio_active()) {
        const ColiExpertMatrixInfo *gm = &ei.matrices[gi];
        const ColiExpertMatrixInfo *um = &ei.matrices[ui];
        const ColiExpertMatrixInfo *dm = &ei.matrices[di];
        size_t gb = 0, ub = 0, db = 0;
        if (qwen_apple8_raw_matrix(gm, (uint64_t)cc->moe_inter, (uint64_t)cc->hidden, &gb) &&
            qwen_apple8_raw_matrix(um, (uint64_t)cc->moe_inter, (uint64_t)cc->hidden, &ub) &&
            qwen_apple8_raw_matrix(dm, (uint64_t)cc->hidden, (uint64_t)cc->moe_inter, &db)) {
            const ColiPackage *pkg = coli_executor_package(m->coli);
            int fid = qwen_coli_mio_file(pkg, erec->shard_id);
            size_t uoff = qwen_align16(gb);
            size_t doff = uoff ? qwen_align16(uoff + ub) : 0;
            size_t total = doff && db <= SIZE_MAX - doff ? doff + db : 0;
            if (fid >= 0 && total &&
                gm->weight_offset <= UINT64_MAX - erec->payload_offset &&
                um->weight_offset <= UINT64_MAX - erec->payload_offset &&
                dm->weight_offset <= UINT64_MAX - erec->payload_offset) {
                if (!s->mio) { s->mio = 1; s->mio_slot = -1; }
                if (s->mio_slot >= 0 && metalio_slot_bytes(s->mio_slot) < total) {
                    metalio_slot_free(s->mio_slot);
                    s->mio_slot = -1;
                }
                if (s->mio_slot < 0) s->mio_slot = metalio_slot_alloc(total);
                if (s->mio_slot >= 0) {
                    ColiMetalioRegion regions[3] = {
                        { fid, erec->payload_offset + gm->weight_offset, gb, 0 },
                        { fid, erec->payload_offset + um->weight_offset, ub, uoff },
                        { fid, erec->payload_offset + dm->weight_offset, db, doff },
                    };
                    int64_t ev = metalio_loadv(s->mio_slot, regions, 3,
                                               g_mio_prefetching ? MIO_LOAD_SPEC
                                             : g_mio_async_issue ? MIO_LOAD_ASYNC
                                             : MIO_LOAD_DEMAND);
                    if (ev > 0) {
                        s->mio_event = ev;
                        if (!g_mio_prefetching && !g_mio_async_issue) {
                            if (metalio_wait(ev) == 0) s->mio_waited = ev;
                            else ev = -1;
                        }
                        if (ev > 0) {
                            s->fmt = 17;
                            s->apple8_direct = 1;
                            s->mio_resident = 1;
                            s->apple8_gate_off = 0; s->apple8_gate_bytes = gb;
                            s->apple8_up_off = uoff; s->apple8_up_bytes = ub;
                            s->apple8_down_off = doff; s->apple8_down_bytes = db;
                            s->pinned = 0;
                            m->prof_apple8_direct_bytes += (uint64_t)gb + ub + db;
                            return;
                        }
                    }
                }
            }
        }
    }
    /* A slot can move from direct raw Apple8 back to the canonical fallback
     * (for example a mixed/codec package). Keep its MetalIO allocation for
     * future reuse but stop treating heap MXFP4 buffers as mio-resident. */
    s->apple8_direct = 0;
    if (s->fmt == 17) s->mio_resident = 0;
#endif

    /* Keep MXFP4 compressed in the streamed cache. The generic loader validates
'''
    replace_once(path, old, direct, "Raw Apple8 fast path")

    old = "        if (unif && fmt0 == 7 && K <= 64) {\n"
    new = r'''#ifdef COLI_METALIO
        if (unif && fmt0 == 17 && K <= 64) {
            float *yd = falloc(D);
            for (int i = 0; i < K; i++) {
                Slot *s = slots[i];
                expert_wait_ready(m, s);
                if (!s->apple8_direct ||
                    !coli_apple8_metalio_swiglu_slot(s->mio_slot,
                        s->apple8_gate_off, s->apple8_gate_bytes,
                        s->apple8_up_off, s->apple8_up_bytes,
                        s->apple8_down_off, s->apple8_down_bytes,
                        x, yd, 1, D, c->moe_inter)) {
                    fprintf(stderr, "qwen: direct Apple8 decode dispatch failed\n");
                    exit(1);
                }
                for (int d = 0; d < D; d++) acc[d] += yd[d] * w[i];
                m->prof_apple8_direct_blocks++;
                m->prof_apple8_direct_experts++;
            }
            free(yd);
            gpu_ok = 1;
        } else
#endif
        if (unif && fmt0 == 7 && K <= 64) {
'''
    replace_once(path, old, new, "direct Apple8 decode dispatch failed")

    old = "            if (s->fmt == 7) {\n"
    new = r'''#ifdef COLI_METALIO
            if (s->fmt == 17) {
                if (!s->apple8_direct ||
                    !coli_apple8_metalio_swiglu_slot(s->mio_slot,
                        s->apple8_gate_off, s->apple8_gate_bytes,
                        s->apple8_up_off, s->apple8_up_bytes,
                        s->apple8_down_off, s->apple8_down_bytes,
                        xscratch, yscratch, st, D, I)) {
                    fprintf(stderr, "qwen: direct Apple8 batched-prefill dispatch failed\n");
                    exit(1);
                }
                m->prof_apple8_direct_blocks++;
                m->prof_apple8_direct_experts++;
            } else
#endif
            if (s->fmt == 7) {
'''
    replace_once(path, old, new, "direct Apple8 batched-prefill dispatch failed")

    old = "            if (g_mio_n == 0) { metalio_shutdown(); g_metal_io = 0; }\n"
    new = "            if (g_mio_n == 0 && !coli_mode) { metalio_shutdown(); g_metal_io = 0; }\n"
    replace_once(path, old, new, "g_mio_n == 0 && !coli_mode")

    old = "            fprintf(stderr, \"[metalio] expert streaming via MTLIO active (%d shards)\\n\", g_mio_n);\n" \
          "        }\n" \
          "    }\n" \
          "#endif\n" \
          "    qprof_startup_end(&m);\n"
    new = "            fprintf(stderr, \"[metalio] expert streaming via MTLIO active (%d snapshot shards; COLI shards lazy)\\n\", g_mio_n);\n" \
          "        }\n" \
          "    }\n" \
          "    g_apple8_direct = getenv(\"QWEN_APPLE8_DIRECT\") ? atoi(getenv(\"QWEN_APPLE8_DIRECT\")) : 0;\n" \
          "    if (g_apple8_direct) {\n" \
          "        if (!(g_metal_io && g_metal_compute && metalio_active()) ||\n" \
          "            !coli_apple8_metalio_direct_init()) {\n" \
          "            fprintf(stderr, \"[QWEN-APPLE8] direct path unavailable; using canonical fallback\\n\");\n" \
          "            g_apple8_direct = 0;\n" \
          "        } else {\n" \
          "            fprintf(stderr, \"[QWEN-APPLE8] direct raw Apple8 + MetalIO execution enabled\\n\");\n" \
          "        }\n" \
          "    }\n" \
          "#endif\n" \
          "    qprof_startup_end(&m);\n"
    replace_once(path, old, new, "direct raw Apple8 + MetalIO execution enabled")

    old = "    if (g_metal_io) {\n        ColiMetalioStats st;\n"
    new = "    if (g_metal_io) {\n        if (g_apple8_direct) coli_apple8_metalio_direct_shutdown();\n        ColiMetalioStats st;\n"
    replace_once(path, old, new, "g_apple8_direct) coli_apple8_metalio_direct_shutdown")


def main() -> None:
    patch_format_header()
    patch_format_base()
    patch_makefile()
    patch_qwen()
    print("[apple8-qwen-patch] done")
    print("[apple8-qwen-patch] next: cd c && make qwen_moe METAL=1")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"apple8-qwen-patch: {exc}", file=sys.stderr)
        raise

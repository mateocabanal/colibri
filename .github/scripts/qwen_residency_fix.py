from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    s = p.read_text()
    if old not in s:
        raise SystemExit(f"{path}: expected text not found: {old[:100]!r}")
    if s.count(old) != 1:
        raise SystemExit(f"{path}: expected exactly one match, got {s.count(old)}")
    p.write_text(s.replace(old, new, 1))

# 1) Generic COLI best-effort uncached range API. Buffered remains the default.
replace_once(
    "c/coli_format.h",
    '''/* Reads an exact byte range relative to the top-level record. Range reads are\n * thread-safe: they use pread/compat_pread and no shared seek position. */\nint coli_package_read_range(const ColiPackage *package,\n                            const ColiRecordInfo *record,\n                            uint64_t record_offset, void *destination,\n                            size_t bytes, char *error, size_t error_size);\n''',
    '''enum {\n    COLI_CSF_READ_DEFAULT = 0,\n    /* Best effort: consume the requested source bytes without retaining them in\n     * the host file cache. On macOS this uses the shard's F_NOCACHE descriptor;\n     * on POSIX systems without a safe unaligned direct-I/O path it reads\n     * normally then advises DONTNEED. Callers must not depend on the hint. */\n    COLI_CSF_READ_UNCACHED = 1u << 0,\n};\n\n/* Reads an exact byte range relative to the top-level record. Range reads are\n * thread-safe: they use pread/compat_pread and no shared seek position. */\nint coli_package_read_range_ex(const ColiPackage *package,\n                               const ColiRecordInfo *record,\n                               uint64_t record_offset, void *destination,\n                               size_t bytes, uint32_t read_flags,\n                               char *error, size_t error_size);\nint coli_package_read_range(const ColiPackage *package,\n                            const ColiRecordInfo *record,\n                            uint64_t record_offset, void *destination,\n                            size_t bytes, char *error, size_t error_size);\n''')

replace_once(
    "c/coli_format.c",
    '''int coli_package_read_range(const ColiPackage *p, const ColiRecordInfo *r,\n                            uint64_t record_offset, void *destination, size_t bytes,\n                            char *error, size_t error_size) {\n    uint64_t end;\n    if (!p || !r || (!destination && bytes) || checked_add_u64(record_offset, bytes, &end) || end > r->stored_bytes || r->shard_id >= p->shard_count) {\n        csf_error(error, error_size, "record range is invalid"); return -1;\n    }\n    if (!bytes) return 0;\n    return pread_full(p->shards[r->shard_id].fd, destination, bytes, r->payload_offset + record_offset, error, error_size);\n}\n''',
    '''int coli_package_read_range_ex(const ColiPackage *p, const ColiRecordInfo *r,\n                               uint64_t record_offset, void *destination, size_t bytes,\n                               uint32_t read_flags, char *error, size_t error_size) {\n    uint64_t end, source_offset;\n    ColiCsfShard *shard;\n    int fd, rc;\n    if (read_flags & ~COLI_CSF_READ_UNCACHED) {\n        csf_error(error, error_size, "unsupported COLI read flags 0x%x", read_flags);\n        return -1;\n    }\n    if (!p || !r || (!destination && bytes) ||\n        checked_add_u64(record_offset, bytes, &end) || end > r->stored_bytes ||\n        r->shard_id >= p->shard_count ||\n        checked_add_u64(r->payload_offset, record_offset, &source_offset)) {\n        csf_error(error, error_size, "record range is invalid"); return -1;\n    }\n    if (!bytes) return 0;\n    shard = &p->shards[r->shard_id];\n    fd = shard->fd;\n#ifdef __APPLE__\n    /* F_NOCACHE has no alignment requirement, unlike Linux O_DIRECT, so the\n     * compiler's naturally ragged MXFP4 matrix spans can use it directly. */\n    if ((read_flags & COLI_CSF_READ_UNCACHED) && shard->direct_fd >= 0)\n        fd = shard->direct_fd;\n#endif\n    rc = pread_full(fd, destination, bytes, source_offset, error, error_size);\n#if !defined(__APPLE__) && !defined(_WIN32)\n    /* Arbitrary matrix subspans are not guaranteed to satisfy O_DIRECT's\n     * address/offset/length alignment. Keep correctness first and make the\n     * retention policy best-effort through DONTNEED instead. */\n    if (rc == 0 && (read_flags & COLI_CSF_READ_UNCACHED) &&\n        source_offset <= (uint64_t)INT64_MAX && bytes <= (size_t)(INT64_MAX - source_offset))\n        (void)posix_fadvise(shard->fd, (off_t)source_offset, (off_t)bytes, POSIX_FADV_DONTNEED);\n#else\n    (void)shard;\n#endif\n    return rc;\n}\n\nint coli_package_read_range(const ColiPackage *p, const ColiRecordInfo *r,\n                            uint64_t record_offset, void *destination, size_t bytes,\n                            char *error, size_t error_size) {\n    return coli_package_read_range_ex(p, r, record_offset, destination, bytes,\n                                      COLI_CSF_READ_DEFAULT, error, error_size);\n}\n''')

# 2) Let reusable MXFP4 expert loader select the range retention policy.
replace_once(
    "c/mxfp4_expert.h",
    '''int coli_mxfp4_expert_load(const ColiPackage *package,\n                           const ColiRecordInfo *record,\n                           int hidden, int intermediate,\n                           const ColiMxfp4ExpertBuffers *buffers,\n                           ColiMxfp4ExpertLayout *layout,\n                           char *error, size_t error_size);\n''',
    '''int coli_mxfp4_expert_load_ex(const ColiPackage *package,\n                              const ColiRecordInfo *record,\n                              int hidden, int intermediate,\n                              const ColiMxfp4ExpertBuffers *buffers,\n                              ColiMxfp4ExpertLayout *layout,\n                              uint32_t read_flags,\n                              char *error, size_t error_size);\nint coli_mxfp4_expert_load(const ColiPackage *package,\n                           const ColiRecordInfo *record,\n                           int hidden, int intermediate,\n                           const ColiMxfp4ExpertBuffers *buffers,\n                           ColiMxfp4ExpertLayout *layout,\n                           char *error, size_t error_size);\n''')

replace_once(
    "c/mxfp4_expert.c",
    '''static int read_span(const ColiPackage *package, const ColiRecordInfo *record,\n                     const ColiExpertMatrixInfo *matrix, int scale,\n                     uint8_t *destination, size_t bytes,\n                     char *error, size_t error_size) {\n    const uint64_t offset = scale ? matrix->scale_offset : matrix->weight_offset;\n    if (coli_package_read_range(package, record, offset, destination, bytes,\n                                error, error_size) != 0)\n        return -1;\n    return 0;\n}\n\nint coli_mxfp4_expert_load(const ColiPackage *package,\n                           const ColiRecordInfo *record,\n                           int hidden, int intermediate,\n                           const ColiMxfp4ExpertBuffers *buffers,\n                           ColiMxfp4ExpertLayout *layout,\n                           char *error, size_t error_size) {\n''',
    '''static int read_span(const ColiPackage *package, const ColiRecordInfo *record,\n                     const ColiExpertMatrixInfo *matrix, int scale,\n                     uint8_t *destination, size_t bytes, uint32_t read_flags,\n                     char *error, size_t error_size) {\n    const uint64_t offset = scale ? matrix->scale_offset : matrix->weight_offset;\n    if (coli_package_read_range_ex(package, record, offset, destination, bytes,\n                                   read_flags, error, error_size) != 0)\n        return -1;\n    return 0;\n}\n\nint coli_mxfp4_expert_load_ex(const ColiPackage *package,\n                              const ColiRecordInfo *record,\n                              int hidden, int intermediate,\n                              const ColiMxfp4ExpertBuffers *buffers,\n                              ColiMxfp4ExpertLayout *layout,\n                              uint32_t read_flags,\n                              char *error, size_t error_size) {\n''')

p = Path("c/mxfp4_expert.c")
s = p.read_text()
s = s.replace('''if (read_span(package, record, gate, 0, buffers->gate_weights,\n                  layout->gate_weight_bytes, error, error_size) != 0 ||\n        read_span(package, record, gate, 1, buffers->gate_scales,\n                  layout->gate_scale_bytes, error, error_size) != 0 ||\n        read_span(package, record, up, 0, buffers->up_weights,\n                  layout->up_weight_bytes, error, error_size) != 0 ||\n        read_span(package, record, up, 1, buffers->up_scales,\n                  layout->up_scale_bytes, error, error_size) != 0 ||\n        read_span(package, record, down, 0, buffers->down_weights,\n                  layout->down_weight_bytes, error, error_size) != 0 ||\n        read_span(package, record, down, 1, buffers->down_scales,\n                  layout->down_scale_bytes, error, error_size) != 0)\n        return -1;\n    return 0;\n}\n''', '''if (read_span(package, record, gate, 0, buffers->gate_weights,\n                  layout->gate_weight_bytes, read_flags, error, error_size) != 0 ||\n        read_span(package, record, gate, 1, buffers->gate_scales,\n                  layout->gate_scale_bytes, read_flags, error, error_size) != 0 ||\n        read_span(package, record, up, 0, buffers->up_weights,\n                  layout->up_weight_bytes, read_flags, error, error_size) != 0 ||\n        read_span(package, record, up, 1, buffers->up_scales,\n                  layout->up_scale_bytes, read_flags, error, error_size) != 0 ||\n        read_span(package, record, down, 0, buffers->down_weights,\n                  layout->down_weight_bytes, read_flags, error, error_size) != 0 ||\n        read_span(package, record, down, 1, buffers->down_scales,\n                  layout->down_scale_bytes, read_flags, error, error_size) != 0)\n        return -1;\n    return 0;\n}\n\nint coli_mxfp4_expert_load(const ColiPackage *package,\n                           const ColiRecordInfo *record,\n                           int hidden, int intermediate,\n                           const ColiMxfp4ExpertBuffers *buffers,\n                           ColiMxfp4ExpertLayout *layout,\n                           char *error, size_t error_size) {\n    return coli_mxfp4_expert_load_ex(package, record, hidden, intermediate,\n                                     buffers, layout, COLI_CSF_READ_DEFAULT,\n                                     error, error_size);\n}\n''')
if 'read_span(package, record, gate, 0, buffers->gate_weights' not in s or 'read_flags, error, error_size' not in s:
    raise SystemExit("c/mxfp4_expert.c: failed to patch span reads")
p.write_text(s)

# 3) Qwen: streamed experts use uncached policy by default; dense/static stays buffered.
replace_once(
    "c/qwen_moe.c",
    '''        if (coli_mxfp4_expert_load(pkg, erec, cc->hidden, cc->moe_inter,\n                                   &buffers, &layout, err, sizeof(err))) {\n''',
    '''        uint32_t read_flags = g_expert_drop ? COLI_CSF_READ_UNCACHED : COLI_CSF_READ_DEFAULT;\n        if (coli_mxfp4_expert_load_ex(pkg, erec, cc->hidden, cc->moe_inter,\n                                      &buffers, &layout, read_flags, err, sizeof(err))) {\n''')
replace_once(
    "c/qwen_moe.c",
    '''    const ColiPackage *pkg = coli_executor_package(m->coli);\n    if (coli_package_read_range(pkg, erec, ei.matrices[gi].weight_offset,\n                                s->bgu, (size_t)ng * 2, NULL, 0) ||\n        coli_package_read_range(pkg, erec, ei.matrices[ui].weight_offset,\n                                s->bgu + ng, (size_t)ng * 2, NULL, 0) ||\n        coli_package_read_range(pkg, erec, ei.matrices[di].weight_offset,\n                                s->bd, (size_t)nd * 2, NULL, 0)) {\n''',
    '''    const ColiPackage *pkg = coli_executor_package(m->coli);\n    uint32_t read_flags = g_expert_drop ? COLI_CSF_READ_UNCACHED : COLI_CSF_READ_DEFAULT;\n    if (coli_package_read_range_ex(pkg, erec, ei.matrices[gi].weight_offset,\n                                   s->bgu, (size_t)ng * 2, read_flags, NULL, 0) ||\n        coli_package_read_range_ex(pkg, erec, ei.matrices[ui].weight_offset,\n                                   s->bgu + ng, (size_t)ng * 2, read_flags, NULL, 0) ||\n        coli_package_read_range_ex(pkg, erec, ei.matrices[di].weight_offset,\n                                   s->bd, (size_t)nd * 2, read_flags, NULL, 0)) {\n''')

# 4) Metal: ordered interval registry instead of O(N) vector scans.
replace_once("c/backend_metal.mm", '#include <vector>\n', '#include <vector>\n#include <map>\n')
replace_once(
    "c/backend_metal.mm",
    '''struct Slab { void *base; size_t len; id<MTLBuffer> buf; };\nstatic std::vector<Slab> g_slabs;\nstatic std::mutex g_slab_mtx;   // expert_load registers slabs from parallel OpenMP threads\n''',
    '''struct Slab { void *base; size_t len; id<MTLBuffer> buf; };\n/* Ordered by base address: exact register/unregister and predecessor interval\n * lookup are O(log N), rather than scanning every resident expert slab. */\nstatic std::map<uintptr_t, Slab> g_slabs;\nstatic std::mutex g_slab_mtx;   // expert_load registers slabs from parallel OpenMP threads\n\nstatic id<MTLBuffer> slab_resolve_locked(uintptr_t u, uint64_t *addr) {\n  auto it = g_slabs.upper_bound(u);\n  if (it == g_slabs.begin()) return nil;\n  --it;\n  uintptr_t base = it->first;\n  const Slab &s = it->second;\n  if ((u - base) >= s.len) return nil;\n  if (addr) *addr = (uint64_t)[s.buf gpuAddress] + (u - base);\n  return s.buf;\n}\n''')
replace_once(
    "c/backend_metal.mm",
    '''  {\n    std::lock_guard<std::mutex> lk(g_slab_mtx);   // called from parallel expert_load threads\n    bool found = false;\n    for (auto &s : g_slabs) if (s.base == base) { old = s.buf; s.len = len; s.buf = b; found = true; break; }\n    if (!found) g_slabs.push_back({base, len, b});\n  }\n''',
    '''  {\n    std::lock_guard<std::mutex> lk(g_slab_mtx);   // called from parallel expert_load threads\n    uintptr_t key = (uintptr_t)base;\n    auto it = g_slabs.find(key);\n    if (it != g_slabs.end()) { old = it->second.buf; it->second = {base, len, b}; }\n    else g_slabs.emplace(key, Slab{base, len, b});\n  }\n''')
replace_once(
    "c/backend_metal.mm",
    '''  {\n    std::lock_guard<std::mutex> lk(g_slab_mtx);\n    for (size_t i=0;i<g_slabs.size();i++) if (g_slabs[i].base==base) {\n      b = g_slabs[i].buf; g_slabs[i].buf=nil; g_slabs.erase(g_slabs.begin()+i); break;\n    }\n  }\n''',
    '''  {\n    std::lock_guard<std::mutex> lk(g_slab_mtx);\n    auto it = g_slabs.find((uintptr_t)base);\n    if (it != g_slabs.end()) { b = it->second.buf; it->second.buf=nil; g_slabs.erase(it); }\n  }\n''')
replace_once(
    "c/backend_metal.mm",
    '''static id<MTLBuffer> resolve(const void *p, uint64_t *addr) {\n  std::lock_guard<std::mutex> lk(g_slab_mtx);\n  uintptr_t u=(uintptr_t)p;\n  for (auto &s : g_slabs) { uintptr_t b=(uintptr_t)s.base;\n    if (u>=b && u<b+s.len) { *addr = (uint64_t)[s.buf gpuAddress] + (u-b); return s.buf; } }\n  return nil;\n}\n''',
    '''static id<MTLBuffer> resolve(const void *p, uint64_t *addr) {\n  std::lock_guard<std::mutex> lk(g_slab_mtx);\n  return slab_resolve_locked((uintptr_t)p, addr);\n}\n''')
replace_once(
    "c/backend_metal.mm",
    '''extern "C" int coli_metal_ptr_registered(const void *p) {\n  std::lock_guard<std::mutex> lk(g_slab_mtx);\n  uintptr_t u=(uintptr_t)p;\n  for (auto &s : g_slabs)\n    if (u>=(uintptr_t)s.base && u<(uintptr_t)s.base+s.len) return 1;\n  return 0;\n}\n''',
    '''extern "C" int coli_metal_ptr_registered(const void *p) {\n  std::lock_guard<std::mutex> lk(g_slab_mtx);\n  return slab_resolve_locked((uintptr_t)p, NULL) != nil;\n}\n''')

# 5) Regression: generic uncached range is byte-identical and rejects unknown flags.
replace_once(
    "c/tests/test_coli_format.c",
    '''r=coli_package_record_by_name(p,"tiny.weight");CHECK(r&&r==coli_package_record_by_id(p,1));CHECK(coli_package_tensor_info(p,r,&ti,err,sizeof(err))==0);CHECK(ti.rank==2&&ti.dims[0]==1&&ti.dims[1]==1&&ti.data_offset==128);CHECK(coli_package_read_range(p,r,128,&b,1,err,sizeof(err))==0&&b==0x2a);CHECK(coli_package_verify_all(p,err,sizeof(err))==0);\n''',
    '''r=coli_package_record_by_name(p,"tiny.weight");CHECK(r&&r==coli_package_record_by_id(p,1));CHECK(coli_package_tensor_info(p,r,&ti,err,sizeof(err))==0);CHECK(ti.rank==2&&ti.dims[0]==1&&ti.dims[1]==1&&ti.data_offset==128);CHECK(coli_package_read_range(p,r,128,&b,1,err,sizeof(err))==0&&b==0x2a);b=0;CHECK(coli_package_read_range_ex(p,r,128,&b,1,COLI_CSF_READ_UNCACHED,err,sizeof(err))==0&&b==0x2a);CHECK(coli_package_read_range_ex(p,r,128,&b,1,0x80000000u,err,sizeof(err))!=0);CHECK(coli_package_verify_all(p,err,sizeof(err))==0);\n''')

# Sanity assertions to keep this staging transform from silently half-applying.
checks = {
    "c/coli_format.c": ["coli_package_read_range_ex", "COLI_CSF_READ_UNCACHED"],
    "c/mxfp4_expert.c": ["coli_mxfp4_expert_load_ex", "read_flags"],
    "c/qwen_moe.c": ["coli_mxfp4_expert_load_ex", "COLI_CSF_READ_UNCACHED"],
    "c/backend_metal.mm": ["std::map<uintptr_t, Slab>", "slab_resolve_locked"],
}
for path, needles in checks.items():
    text = Path(path).read_text()
    for needle in needles:
        if needle not in text:
            raise SystemExit(f"{path}: missing postcondition {needle}")
print("qwen residency fix staged")

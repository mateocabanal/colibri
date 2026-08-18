/* test_kv_prefix — token-identity reuse plus the global exact-prefix service. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../kv_prefix.h"
#include "../prefix_cache.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

typedef struct {
    unsigned char bytes[16];
    uint64_t position;
    int reset_count;
} PrefixFakeState;

static int pfx_describe(void *opaque, uint64_t position,
                        ColiSequenceStateInfo *info,
                        ColiSequenceSegmentDesc *segments, size_t capacity) {
    PrefixFakeState *s = (PrefixFakeState *)opaque;
    if (!s || !info || position != s->position) return -1;
    memset(info, 0, sizeof(*info));
    info->state_abi = 11;
    info->absolute_position = position;
    info->logical_bytes = sizeof(s->bytes);
    info->resident_bytes = sizeof(*s);
    info->segment_count = 1;
    if (!segments) return 0;
    if (capacity < 1) return -1;
    segments[0] = (ColiSequenceSegmentDesc){
        .segment_id = 1,
        .kind = COLI_SEQUENCE_SEGMENT_ENGINE_NATIVE,
        .element_bytes = 1,
        .visibility = COLI_SEQUENCE_VIS_CPU,
        .logical_rows = 1,
        .row_bytes = sizeof(s->bytes),
        .snapshot_bytes = sizeof(s->bytes),
        .layout_abi = 4,
    };
    return 0;
}

static int pfx_read(void *opaque, uint64_t position, uint32_t segment_id,
                    uint64_t offset, void *dst, size_t bytes) {
    PrefixFakeState *s = (PrefixFakeState *)opaque;
    if (!s || !dst || segment_id != 1 || position != s->position ||
        offset > sizeof(s->bytes) || bytes > sizeof(s->bytes) - (size_t)offset)
        return -1;
    memcpy(dst, s->bytes + offset, bytes);
    return 0;
}

static int pfx_write(void *opaque, uint64_t position, uint32_t segment_id,
                     uint64_t offset, const void *src, size_t bytes) {
    PrefixFakeState *s = (PrefixFakeState *)opaque;
    if (!s || !src || segment_id != 1 || offset > sizeof(s->bytes) ||
        bytes > sizeof(s->bytes) - (size_t)offset)
        return -1;
    memcpy(s->bytes + offset, src, bytes);
    s->position = position;
    return 0;
}

static int pfx_reset(void *opaque) {
    PrefixFakeState *s = (PrefixFakeState *)opaque;
    if (!s) return -1;
    memset(s->bytes, 0, sizeof(s->bytes));
    s->position = 0;
    s->reset_count++;
    return 0;
}

static int pfx_finish(void *opaque, uint64_t position) {
    PrefixFakeState *s = (PrefixFakeState *)opaque;
    if (!s) return -1;
    s->position = position;
    return 0;
}

static const ColiSequenceStateOps pfx_ops = {
    pfx_describe, pfx_read, pfx_write, pfx_reset, pfx_finish
};

static void make_ns(ColiPrefixNamespace *ns) {
    memset(ns, 0, sizeof(*ns));
    ns->state_abi = 11;
    for (size_t i = 0; i < sizeof(ns->fingerprint); ++i)
        ns->fingerprint[i] = (unsigned char)(i * 3u + 1u);
}

static void test_global_prefix_cache(void) {
    char path[256];
    snprintf(path, sizeof(path), ".test_coli_prefix_%ld.bin", (long)getpid());
    remove(path);
    char temp[264]; snprintf(temp, sizeof(temp), "%s.tmp", path); remove(temp);

    ColiPrefixNamespace ns;
    make_ns(&ns);
    const int prefix[] = {10, 20, 30};
    const int extended[] = {10, 20, 30, 40, 50};
    const int divergent[] = {10, 20, 99, 40};

    PrefixFakeState source = {0}, warm = {0}, restart = {0};
    source.position = 3;
    for (size_t i = 0; i < sizeof(source.bytes); ++i)
        source.bytes[i] = (unsigned char)(0x40u + i);
    ColiSequenceStateAdapter src = {&source, &pfx_ops};
    ColiSequenceStateAdapter warm_adapter = {&warm, &pfx_ops};
    ColiSequenceStateAdapter restart_adapter = {&restart, &pfx_ops};

    ColiPrefixCache cache;
    CHECK(coli_prefix_cache_init(&cache, COLI_PREFIX_CACHE_AUTO,
                                 1024 * 1024, 1024 * 1024, path) == 0,
          "global cache init failed");
    CHECK(coli_prefix_cache_store(&cache, &ns, prefix, 3, 3, &src) == 1,
          "global cache store failed");
    int matched = 0;
    CHECK(coli_prefix_cache_restore(&cache, &ns, extended, 5,
                                    &warm_adapter, &matched) == 1,
          "RAM restore failed");
    CHECK(matched == 3, "RAM matched=%d want 3", matched);
    CHECK(warm.position == 3 && !memcmp(warm.bytes, source.bytes, sizeof(warm.bytes)),
          "RAM-restored state differs");
    matched = -1;
    CHECK(coli_prefix_cache_restore(&cache, &ns, divergent, 4,
                                    &warm_adapter, &matched) == 0 && matched == 0,
          "divergent prompt must miss");
    ColiPrefixCacheStats stats;
    coli_prefix_cache_get_stats(&cache, &stats);
    CHECK(stats.hits_ram == 1 && stats.stores == 1,
          "RAM stats hits=%llu stores=%llu",
          (unsigned long long)stats.hits_ram,
          (unsigned long long)stats.stores);
    CHECK(stats.ram_resident_bytes <= stats.ram_budget_bytes,
          "RAM budget exceeded: %zu > %zu",
          stats.ram_resident_bytes, stats.ram_budget_bytes);
    CHECK(stats.disk_live_bytes <= stats.disk_budget_bytes,
          "disk budget exceeded: %llu > %llu",
          (unsigned long long)stats.disk_live_bytes,
          (unsigned long long)stats.disk_budget_bytes);
    coli_prefix_cache_close(&cache);

    /* New process-equivalent object loads only metadata. Snapshot bytes remain
     * SSD-cold until this exact hit. */
    CHECK(coli_prefix_cache_init(&cache, COLI_PREFIX_CACHE_AUTO,
                                 1024 * 1024, 1024 * 1024, path) == 0,
          "restart cache init failed");
    CHECK(cache.count == 1 && cache.entries[0]->snapshot == NULL,
          "restart should load metadata only");
    matched = 0;
    CHECK(coli_prefix_cache_restore(&cache, &ns, extended, 5,
                                    &restart_adapter, &matched) == 1,
          "SSD restart restore failed");
    CHECK(matched == 3 && restart.position == 3 &&
          !memcmp(restart.bytes, source.bytes, sizeof(restart.bytes)),
          "SSD-restored state differs");
    coli_prefix_cache_get_stats(&cache, &stats);
    CHECK(stats.hits_ssd == 1, "SSD hit counter=%llu want 1",
          (unsigned long long)stats.hits_ssd);
    coli_prefix_cache_close(&cache);

    /* Corrupt the snapshot byte, then restart. Index parsing may succeed, but
     * restore must verify CRC and become a safe miss. SSD still needs RAM for
     * exact-token/segment metadata; it just does not keep the payload hot. */
    FILE *f = fopen(path, "r+b");
    CHECK(f != NULL, "open persisted cache for corruption failed");
    if (f) {
        CHECK(coli_prefix_file_header_read(f) == 0, "persisted header invalid");
        ColiPrefixRecordHeader h;
        CHECK(fread(&h, sizeof(h), 1, f) == 1, "persisted record missing");
        long long off = (long long)sizeof(ColiPrefixFileHeader) +
            (long long)sizeof(ColiPrefixRecordHeader) +
            (long long)h.token_count * sizeof(int) +
            (long long)h.segment_count * sizeof(ColiSequenceSegmentDesc);
        CHECK(coli_prefix_fseek(f, off, SEEK_SET) == 0, "seek corrupt byte failed");
        int byte = fgetc(f);
        CHECK(byte != EOF, "read corrupt byte failed");
        CHECK(coli_prefix_fseek(f, off, SEEK_SET) == 0, "seek rewrite byte failed");
        if (byte != EOF) fputc(byte ^ 0x5a, f);
        fclose(f);
    }
    memset(&restart, 0, sizeof(restart));
    CHECK(coli_prefix_cache_init(&cache, COLI_PREFIX_CACHE_SSD,
                                 4096, 1024 * 1024, path) == 0,
          "corrupt restart index init failed");
    matched = -1;
    CHECK(coli_prefix_cache_restore(&cache, &ns, extended, 5,
                                    &restart_adapter, &matched) == 0 && matched == 0,
          "corrupt SSD state must miss");
    coli_prefix_cache_get_stats(&cache, &stats);
    CHECK(stats.corrupt_entries >= 1, "corruption counter not incremented");
    coli_prefix_cache_close(&cache);
    remove(path); remove(temp);
}

static void test_prefix_cache_budgets(void) {
    char path[256];
    snprintf(path, sizeof(path), ".test_coli_prefix_budget_%ld.bin", (long)getpid());
    remove(path);
    char temp[264]; snprintf(temp, sizeof(temp), "%s.tmp", path); remove(temp);

    ColiPrefixNamespace ns;
    make_ns(&ns);
    PrefixFakeState source = {0};
    source.position = 3;
    memset(source.bytes, 0x6d, sizeof(source.bytes));
    ColiSequenceStateAdapter src = {&source, &pfx_ops};
    const int a[] = {1, 2, 3};
    const int b[] = {1, 2, 3, 4};
    size_t metadata = sizeof(ColiPrefixCacheEntry) +
        sizeof(a) + sizeof(ColiSequenceSegmentDesc);

    /* RAM-only admission that cannot fit its own payload must be rejected, not
     * retained above the cap. */
    ColiPrefixCache cache;
    CHECK(coli_prefix_cache_init(&cache, COLI_PREFIX_CACHE_RAM,
                                 metadata + 4, 0, NULL) == 0,
          "RAM-only budget cache init failed");
    CHECK(coli_prefix_cache_store(&cache, &ns, a, 3, 3, &src) == 0,
          "oversized RAM-only admission should be rejected");
    CHECK(cache.count == 0 && cache.ram_resident_bytes <= cache.ram_budget_bytes,
          "RAM-only hard cap violated");
    coli_prefix_cache_close(&cache);

    /* SSD can own the payload while RAM owns metadata only. */
    remove(path);
    CHECK(coli_prefix_cache_init(&cache, COLI_PREFIX_CACHE_SSD,
                                 metadata + 64, 4096, path) == 0,
          "SSD-only budget cache init failed");
    CHECK(coli_prefix_cache_store(&cache, &ns, a, 3, 3, &src) == 1,
          "SSD-only admission failed");
    CHECK(cache.count == 1 && cache.entries[0]->snapshot == NULL &&
          cache.entries[0]->disk_record_bytes != 0 &&
          cache.ram_resident_bytes <= cache.ram_budget_bytes &&
          cache.disk_live_bytes <= cache.disk_budget_bytes,
          "SSD-only hot/cold accounting wrong");
    coli_prefix_cache_close(&cache);

    /* A record too large for the current SSD budget stays RAM-only. Raising the
     * budget later and compacting another entry must not accidentally persist
     * that first RAM-only record. */
    remove(path);
    CHECK(coli_prefix_cache_init(&cache, COLI_PREFIX_CACHE_AUTO,
                                 1024 * 1024, 1, path) == 0,
          "mixed budget cache init failed");
    CHECK(coli_prefix_cache_store(&cache, &ns, a, 3, 3, &src) == 1,
          "RAM fallback admission failed");
    CHECK(cache.count == 1 && cache.entries[0]->disk_record_bytes == 0 &&
          cache.entries[0]->snapshot != NULL,
          "too-large disk record should remain RAM-only");
    cache.disk_budget_bytes = 4096;
    source.position = 4;
    source.bytes[0] ^= 0x11;
    CHECK(coli_prefix_cache_store(&cache, &ns, b, 4, 4, &src) == 1,
          "second persisted admission failed");
    size_t ia = coli_prefix_find_exact_index(&cache, &ns, a, 3);
    size_t ib = coli_prefix_find_exact_index(&cache, &ns, b, 4);
    CHECK(ia < cache.count && ib < cache.count,
          "expected both RAM and persisted entries");
    if (ia < cache.count && ib < cache.count) {
        CHECK(cache.entries[ia]->disk_record_bytes == 0 &&
              cache.entries[ia]->disk_snapshot_offset == 0,
              "disk rewrite persisted a RAM-only entry");
        CHECK(cache.entries[ib]->disk_record_bytes != 0 &&
              cache.entries[ib]->disk_snapshot_offset != 0,
              "new entry was not persisted");
    }
    CHECK(cache.ram_resident_bytes <= cache.ram_budget_bytes &&
          cache.disk_live_bytes <= cache.disk_budget_bytes,
          "mixed cache exceeded a hard budget");
    coli_prefix_cache_close(&cache);
    remove(path); remove(temp);
}

int main(void) {
    kv_prefix p = {0};
    CHECK(kv_prefix_alloc(&p, 16), "alloc failed");

    const int turn1[] = {5, 6, 7};
    const int turn2[] = {5, 6, 7, 8, 9};
    const int other[] = {5, 6, 99, 8, 9};

    CHECK(kv_prefix_reuse(&p, turn2, 5) == 0, "empty record must not be reused");

    kv_prefix_record(&p, turn1, 0, 3);
    CHECK(p.len == 3, "len=%d, want 3", p.len);
    CHECK(kv_prefix_reuse(&p, turn2, 5) == 3, "an extending prompt reuses all 3");
    CHECK(kv_prefix_reuse(&p, other, 5) == 0, "a diverging prompt reuses nothing");
    CHECK(kv_prefix_reuse(&p, turn1, 3) == 0, "equal-length prompt must not reuse");

    const int shorter[] = {5, 6};
    CHECK(kv_prefix_reuse(&p, shorter, 2) == 0, "shorter prompt must not reuse");
    const int first[] = {1, 6, 7, 8};
    CHECK(kv_prefix_reuse(&p, first, 4) == 0, "divergence at 0 reuses nothing");
    const int last[] = {5, 6, 70, 8};
    CHECK(kv_prefix_reuse(&p, last, 4) == 0, "divergence at len-1 reuses nothing");

    const int gen[] = {8, 9};
    kv_prefix_record(&p, gen, 3, 2);
    CHECK(p.len == 5, "len=%d after append, want 5", p.len);
    const int turn3[] = {5, 6, 7, 8, 9, 10};
    CHECK(kv_prefix_reuse(&p, turn3, 6) == 5, "reuse must cover prompt+generated");

    kv_prefix_taint(&p);
    CHECK(kv_prefix_reuse(&p, turn3, 6) == 0, "a tainted state is never reused");
    kv_prefix_clear(&p);
    CHECK(p.tainted == 0 && p.len == 0, "clear must drop both len and taint");

    CHECK(kv_prefix_alloc(&p, 4), "realloc failed");
    kv_prefix_record(&p, turn1, 0, 3);
    const int spill[] = {1, 2, 3};
    kv_prefix_record(&p, spill, 3, 3);
    CHECK(p.len == 0, "overflowing write must drop the record, len=%d", p.len);
    CHECK(kv_prefix_reuse(&p, turn2, 5) == 0, "dropped record must not be reused");

    CHECK(kv_prefix_alloc(&p, 32), "alloc failed");
    CHECK(p.len == 0 && p.tainted == 0, "alloc must reset the record");
    kv_prefix_record(&p, turn1, 0, 3);
    kv_prefix_record(&p, gen, 3, 2);
    CHECK(kv_prefix_grow(&p, 64, 5), "grow failed");
    CHECK(p.cap == 64, "cap=%d after grow, want 64", p.cap);
    CHECK(p.len == 5, "len=%d after grow, want 5 preserved", p.len);
    CHECK(kv_prefix_reuse(&p, turn3, 6) == 5,
          "a preserved record must still be reusable after growing");
    CHECK(kv_prefix_grow(&p, 128, 999), "grow failed");
    CHECK(p.len == 5, "keep must clamp to len, got %d", p.len);
    CHECK(kv_prefix_grow(&p, 256, 0), "grow failed");
    CHECK(p.len == 0, "keep=0 must leave the record empty, got %d", p.len);
    CHECK(kv_prefix_reuse(&p, turn3, 6) == 0, "an emptied record reuses nothing");
    kv_prefix_record(&p, turn1, 0, 3);
    CHECK(kv_prefix_grow(&p, 2, 3), "grow failed");
    CHECK(p.len <= 2 && p.cap == 2, "len=%d cap=%d must stay within cap", p.len, p.cap);

    kv_prefix nul = {0};
    CHECK(kv_prefix_reuse(&nul, turn2, 5) == 0, "NULL record reuses nothing");
    kv_prefix_record(&nul, turn1, 0, 3);
    kv_prefix_clear(&nul);
    kv_prefix_free(&nul);

    kv_prefix_free(&p);
    CHECK(p.fed == NULL && p.cap == 0, "free must clear the record");

    test_global_prefix_cache();
    test_prefix_cache_budgets();

    if (failures) { fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    puts("kv_prefix + global RAM/SSD prefix cache: ok");
    return 0;
}

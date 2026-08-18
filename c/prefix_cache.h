#ifndef COLIBRI_PREFIX_CACHE_H
#define COLIBRI_PREFIX_CACHE_H

#include "compat.h"
#include "sequence_state.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#define coli_prefix_fseek _fseeki64
#define coli_prefix_ftell _ftelli64
#else
#include <unistd.h>
#define coli_prefix_fseek fseeko
#define coli_prefix_ftell ftello
#endif

#define COLI_PREFIX_CACHE_MAX_ENTRIES 64
#define COLI_PREFIX_CACHE_FILE_VERSION 1u
#define COLI_PREFIX_CACHE_RECORD_MAGIC UINT32_C(0x58465043) /* CPFX */
#define COLI_PREFIX_CACHE_MAX_TOKENS (1u << 24)
#define COLI_PREFIX_CACHE_MAX_SEGMENTS 4096u
#define COLI_PREFIX_CACHE_PATH_MAX 1024

typedef enum {
    COLI_PREFIX_CACHE_OFF = 0,
    COLI_PREFIX_CACHE_RAM = 1u << 0,
    COLI_PREFIX_CACHE_SSD = 1u << 1,
    COLI_PREFIX_CACHE_AUTO = (1u << 0) | (1u << 1),
} ColiPrefixCachePolicy;

typedef struct {
    unsigned char fingerprint[32];
    uint64_t state_abi;
} ColiPrefixNamespace;

typedef struct {
    uint64_t lookups;
    uint64_t hits_ram;
    uint64_t hits_ssd;
    uint64_t misses;
    uint64_t stores;
    uint64_t evictions_ram;
    uint64_t evictions_ssd;
    uint64_t matched_tokens;
    uint64_t restore_bytes;
    uint64_t write_bytes;
    uint64_t corrupt_entries;
    size_t entries;
    size_t ram_resident_bytes;
    size_t ram_budget_bytes;
    uint64_t disk_live_bytes;
    uint64_t disk_budget_bytes;
} ColiPrefixCacheStats;

typedef struct {
    ColiPrefixNamespace ns;
    int *tokens;
    uint32_t token_count;
    uint64_t absolute_position;
    ColiSequenceSegmentDesc *segments;
    uint32_t segment_count;
    unsigned char *snapshot;       /* NULL means SSD-cold metadata only */
    size_t snapshot_bytes;
    size_t metadata_bytes;
    uint64_t last_used;
    uint64_t disk_snapshot_offset;
    uint64_t disk_record_bytes;
    uint32_t payload_crc32c;
} ColiPrefixCacheEntry;

typedef struct {
    ColiPrefixCacheEntry *entries[COLI_PREFIX_CACHE_MAX_ENTRIES];
    size_t count;
    size_t ram_resident_bytes;
    size_t ram_budget_bytes;
    uint64_t disk_live_bytes;
    uint64_t disk_budget_bytes;
    ColiPrefixCachePolicy policy;
    char disk_path[COLI_PREFIX_CACHE_PATH_MAX];
    uint64_t clock;
    ColiPrefixCacheStats stats;
} ColiPrefixCache;

typedef struct {
    char magic[8];               /* COLIPFX1 */
    uint32_t version;
    uint32_t reserved;
} ColiPrefixFileHeader;

typedef struct {
    uint32_t magic;
    uint32_t token_count;
    uint32_t segment_count;
    uint32_t reserved;
    unsigned char fingerprint[32];
    uint64_t state_abi;
    uint64_t absolute_position;
    uint64_t snapshot_bytes;
    uint64_t record_bytes;
    uint32_t payload_crc32c;
    uint32_t reserved2;
} ColiPrefixRecordHeader;

static inline int coli_prefix_size_add(size_t a, size_t b, size_t *out) {
    if (!out || SIZE_MAX - a < b) return -1;
    *out = a + b;
    return 0;
}

static inline int coli_prefix_size_mul(size_t a, size_t b, size_t *out) {
    if (!out || (a && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static inline int coli_prefix_ns_equal(const ColiPrefixNamespace *a,
                                       const ColiPrefixNamespace *b) {
    return a && b && a->state_abi == b->state_abi &&
        memcmp(a->fingerprint, b->fingerprint, sizeof(a->fingerprint)) == 0;
}

static inline int coli_prefix_ascii_eq(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return *a == 0 && *b == 0;
}

static inline ColiPrefixCachePolicy coli_prefix_cache_policy_parse(
        const char *value, ColiPrefixCachePolicy fallback) {
    if (!value || !*value) return fallback;
    if (coli_prefix_ascii_eq(value, "off") || coli_prefix_ascii_eq(value, "0"))
        return COLI_PREFIX_CACHE_OFF;
    if (coli_prefix_ascii_eq(value, "ram")) return COLI_PREFIX_CACHE_RAM;
    if (coli_prefix_ascii_eq(value, "ssd")) return COLI_PREFIX_CACHE_SSD;
    if (coli_prefix_ascii_eq(value, "auto")) return COLI_PREFIX_CACHE_AUTO;
    return COLI_PREFIX_CACHE_OFF;
}

static inline uint32_t coli_prefix_crc32c_update(uint32_t crc,
                                                  const void *data,
                                                  size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    crc = ~crc;
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (UINT32_C(0x82f63b78) &
                                (uint32_t)-(int32_t)(crc & 1));
    }
    return ~crc;
}

static inline uint32_t coli_prefix_entry_crc(const ColiPrefixCacheEntry *entry,
                                              const unsigned char *snapshot) {
    if (!entry || (!snapshot && entry->snapshot_bytes)) return 0;
    uint32_t crc = 0;
    crc = coli_prefix_crc32c_update(crc, entry->tokens,
        (size_t)entry->token_count * sizeof(int));
    crc = coli_prefix_crc32c_update(crc, entry->segments,
        (size_t)entry->segment_count * sizeof(ColiSequenceSegmentDesc));
    crc = coli_prefix_crc32c_update(crc, snapshot, entry->snapshot_bytes);
    return crc;
}

static inline size_t coli_prefix_entry_ram_bytes(const ColiPrefixCacheEntry *e) {
    return e ? e->metadata_bytes + (e->snapshot ? e->snapshot_bytes : 0) : 0;
}

static inline void coli_prefix_entry_free(ColiPrefixCacheEntry *e) {
    if (!e) return;
    free(e->tokens);
    free(e->segments);
    free(e->snapshot);
    free(e);
}

static inline uint64_t coli_prefix_record_bytes(uint32_t token_count,
                                                 uint32_t segment_count,
                                                 uint64_t snapshot_bytes) {
    uint64_t bytes = sizeof(ColiPrefixRecordHeader);
    uint64_t token_bytes = (uint64_t)token_count * sizeof(int);
    uint64_t segment_bytes = (uint64_t)segment_count * sizeof(ColiSequenceSegmentDesc);
    if (UINT64_MAX - bytes < token_bytes) return UINT64_MAX;
    bytes += token_bytes;
    if (UINT64_MAX - bytes < segment_bytes) return UINT64_MAX;
    bytes += segment_bytes;
    if (UINT64_MAX - bytes < snapshot_bytes) return UINT64_MAX;
    return bytes + snapshot_bytes;
}

static inline int coli_prefix_file_header_write(FILE *f) {
    ColiPrefixFileHeader h = {{'C','O','L','I','P','F','X','1'},
                              COLI_PREFIX_CACHE_FILE_VERSION, 0};
    return f && fwrite(&h, sizeof(h), 1, f) == 1 ? 0 : -1;
}

static inline int coli_prefix_file_header_read(FILE *f) {
    ColiPrefixFileHeader h;
    static const char magic[8] = {'C','O','L','I','P','F','X','1'};
    if (!f || fread(&h, sizeof(h), 1, f) != 1 ||
        memcmp(h.magic, magic, sizeof(magic)) ||
        h.version != COLI_PREFIX_CACHE_FILE_VERSION)
        return -1;
    return 0;
}

static inline int coli_prefix_read_snapshot_from_disk(
        const ColiPrefixCache *cache, const ColiPrefixCacheEntry *entry,
        unsigned char **out) {
    if (!cache || !entry || !out || !entry->disk_snapshot_offset ||
        !entry->snapshot_bytes || !cache->disk_path[0])
        return -1;
    if (entry->snapshot_bytes > SIZE_MAX) return -1;
    FILE *f = fopen(cache->disk_path, "rb");
    if (!f) return -1;
    if (coli_prefix_fseek(f, (long long)entry->disk_snapshot_offset, SEEK_SET) != 0) {
        fclose(f); return -1;
    }
    unsigned char *blob = (unsigned char *)malloc(entry->snapshot_bytes);
    if (!blob) { fclose(f); return -1; }
    if (fread(blob, 1, entry->snapshot_bytes, f) != entry->snapshot_bytes) {
        free(blob); fclose(f); return -1;
    }
    fclose(f);
    if (coli_prefix_entry_crc(entry, blob) != entry->payload_crc32c) {
        free(blob); return -2;
    }
    *out = blob;
    return 0;
}

static inline void coli_prefix_cache_drop_index(ColiPrefixCache *cache,
                                                 size_t index,
                                                 int disk_eviction) {
    ColiPrefixCacheEntry *e = cache->entries[index];
    size_t ram = coli_prefix_entry_ram_bytes(e);
    cache->ram_resident_bytes = ram <= cache->ram_resident_bytes
        ? cache->ram_resident_bytes - ram : 0;
    if (e->disk_record_bytes <= cache->disk_live_bytes)
        cache->disk_live_bytes -= e->disk_record_bytes;
    else cache->disk_live_bytes = 0;
    for (size_t i = index + 1; i < cache->count; ++i)
        cache->entries[i - 1] = cache->entries[i];
    cache->entries[--cache->count] = NULL;
    if (disk_eviction) cache->stats.evictions_ssd++;
    coli_prefix_entry_free(e);
}

static inline size_t coli_prefix_oldest_index(const ColiPrefixCache *cache,
                                               const ColiPrefixCacheEntry *protect,
                                               int require_snapshot) {
    size_t best = cache ? cache->count : 0;
    uint64_t oldest = UINT64_MAX;
    if (!cache) return best;
    for (size_t i = 0; i < cache->count; ++i) {
        ColiPrefixCacheEntry *e = cache->entries[i];
        if (e == protect || (require_snapshot && !e->snapshot)) continue;
        if (e->last_used < oldest) { oldest = e->last_used; best = i; }
    }
    return best;
}

static inline void coli_prefix_enforce_ram_budget(ColiPrefixCache *cache,
                                                   ColiPrefixCacheEntry *protect) {
    if (!cache) return;
    while (cache->ram_resident_bytes > cache->ram_budget_bytes) {
        size_t victim = coli_prefix_oldest_index(cache, protect, 1);
        if (victim == cache->count) break;
        ColiPrefixCacheEntry *e = cache->entries[victim];
        if (!e->disk_snapshot_offset) {
            /* RAM-only entry cannot be demoted; evict it entirely. */
            coli_prefix_cache_drop_index(cache, victim, 0);
            cache->stats.evictions_ram++;
            continue;
        }
        free(e->snapshot);
        e->snapshot = NULL;
        cache->ram_resident_bytes -= e->snapshot_bytes;
        cache->stats.evictions_ram++;
    }
}

static inline int coli_prefix_cache_rewrite_disk(ColiPrefixCache *cache) {
    if (!cache || !(cache->policy & COLI_PREFIX_CACHE_SSD) || !cache->disk_path[0])
        return 0;
    char temp[COLI_PREFIX_CACHE_PATH_MAX + 8];
    if (snprintf(temp, sizeof(temp), "%s.tmp", cache->disk_path) >= (int)sizeof(temp))
        return -1;
    FILE *out = fopen(temp, "wb");
    if (!out) return -1;
    if (coli_prefix_file_header_write(out) != 0) { fclose(out); remove(temp); return -1; }

    uint64_t new_snapshot_offsets[COLI_PREFIX_CACHE_MAX_ENTRIES] = {0};
    uint64_t new_record_bytes[COLI_PREFIX_CACHE_MAX_ENTRIES] = {0};
    uint64_t live = sizeof(ColiPrefixFileHeader);

    for (size_t i = 0; i < cache->count; ++i) {
        ColiPrefixCacheEntry *e = cache->entries[i];
        unsigned char *owned = NULL;
        const unsigned char *blob = e->snapshot;
        if (!blob) {
            int rr = coli_prefix_read_snapshot_from_disk(cache, e, &owned);
            if (rr != 0) { fclose(out); remove(temp); free(owned); return rr; }
            blob = owned;
        }
        uint64_t rec_bytes = coli_prefix_record_bytes(
            e->token_count, e->segment_count, e->snapshot_bytes);
        if (rec_bytes == UINT64_MAX || UINT64_MAX - live < rec_bytes) {
            free(owned); fclose(out); remove(temp); return -1;
        }
        ColiPrefixRecordHeader h;
        memset(&h, 0, sizeof(h));
        h.magic = COLI_PREFIX_CACHE_RECORD_MAGIC;
        h.token_count = e->token_count;
        h.segment_count = e->segment_count;
        memcpy(h.fingerprint, e->ns.fingerprint, sizeof(h.fingerprint));
        h.state_abi = e->ns.state_abi;
        h.absolute_position = e->absolute_position;
        h.snapshot_bytes = e->snapshot_bytes;
        h.record_bytes = rec_bytes;
        h.payload_crc32c = coli_prefix_entry_crc(e, blob);
        if (fwrite(&h, sizeof(h), 1, out) != 1 ||
            fwrite(e->tokens, sizeof(int), e->token_count, out) != e->token_count ||
            fwrite(e->segments, sizeof(ColiSequenceSegmentDesc), e->segment_count, out)
                != e->segment_count) {
            free(owned); fclose(out); remove(temp); return -1;
        }
        long long snapshot_pos = (long long)coli_prefix_ftell(out);
        if (snapshot_pos <= 0 ||
            fwrite(blob, 1, e->snapshot_bytes, out) != e->snapshot_bytes) {
            free(owned); fclose(out); remove(temp); return -1;
        }
        free(owned);
        new_snapshot_offsets[i] = (uint64_t)snapshot_pos;
        new_record_bytes[i] = rec_bytes;
        live += rec_bytes;
    }
    if (fflush(out) != 0 || fclose(out) != 0) { remove(temp); return -1; }
    if (rename(temp, cache->disk_path) != 0) { remove(temp); return -1; }
    cache->disk_live_bytes = live;
    for (size_t i = 0; i < cache->count; ++i) {
        cache->entries[i]->disk_snapshot_offset = new_snapshot_offsets[i];
        cache->entries[i]->disk_record_bytes = new_record_bytes[i];
        cache->entries[i]->payload_crc32c = cache->entries[i]->snapshot
            ? coli_prefix_entry_crc(cache->entries[i], cache->entries[i]->snapshot)
            : cache->entries[i]->payload_crc32c;
    }
    cache->stats.write_bytes += live;
    return 0;
}

static inline void coli_prefix_enforce_disk_budget(ColiPrefixCache *cache,
                                                    ColiPrefixCacheEntry *protect) {
    if (!cache || !(cache->policy & COLI_PREFIX_CACHE_SSD)) return;
    while (cache->disk_budget_bytes &&
           cache->disk_live_bytes > cache->disk_budget_bytes) {
        size_t victim = coli_prefix_oldest_index(cache, protect, 0);
        if (victim == cache->count) break;
        coli_prefix_cache_drop_index(cache, victim, 1);
    }
}

static inline int coli_prefix_cache_load_index(ColiPrefixCache *cache) {
    if (!cache || !(cache->policy & COLI_PREFIX_CACHE_SSD) || !cache->disk_path[0])
        return 0;
    FILE *f = fopen(cache->disk_path, "rb");
    if (!f) return errno == ENOENT ? 0 : -1;
    if (coli_prefix_file_header_read(f) != 0) { fclose(f); return -1; }
    cache->disk_live_bytes = sizeof(ColiPrefixFileHeader);
    while (cache->count < COLI_PREFIX_CACHE_MAX_ENTRIES) {
        ColiPrefixRecordHeader h;
        size_t got = fread(&h, 1, sizeof(h), f);
        if (!got) break;
        if (got != sizeof(h) || h.magic != COLI_PREFIX_CACHE_RECORD_MAGIC ||
            !h.token_count || h.token_count > COLI_PREFIX_CACHE_MAX_TOKENS ||
            !h.segment_count || h.segment_count > COLI_PREFIX_CACHE_MAX_SEGMENTS ||
            !h.state_abi || !h.snapshot_bytes || h.snapshot_bytes > SIZE_MAX)
            { cache->stats.corrupt_entries++; break; }
        uint64_t expected = coli_prefix_record_bytes(
            h.token_count, h.segment_count, h.snapshot_bytes);
        if (expected == UINT64_MAX || expected != h.record_bytes)
            { cache->stats.corrupt_entries++; break; }

        ColiPrefixCacheEntry *e = (ColiPrefixCacheEntry *)calloc(1, sizeof(*e));
        if (!e) { fclose(f); return -1; }
        e->token_count = h.token_count;
        e->segment_count = h.segment_count;
        e->absolute_position = h.absolute_position;
        e->snapshot_bytes = (size_t)h.snapshot_bytes;
        e->payload_crc32c = h.payload_crc32c;
        e->ns.state_abi = h.state_abi;
        memcpy(e->ns.fingerprint, h.fingerprint, sizeof(h.fingerprint));
        size_t token_bytes, segment_bytes, meta = sizeof(*e);
        if (coli_prefix_size_mul(e->token_count, sizeof(int), &token_bytes) != 0 ||
            coli_prefix_size_mul(e->segment_count, sizeof(ColiSequenceSegmentDesc),
                                 &segment_bytes) != 0 ||
            coli_prefix_size_add(meta, token_bytes, &meta) != 0 ||
            coli_prefix_size_add(meta, segment_bytes, &meta) != 0) {
            coli_prefix_entry_free(e); fclose(f); return -1;
        }
        e->metadata_bytes = meta;
        e->tokens = (int *)malloc(token_bytes);
        e->segments = (ColiSequenceSegmentDesc *)malloc(segment_bytes);
        if (!e->tokens || !e->segments ||
            fread(e->tokens, 1, token_bytes, f) != token_bytes ||
            fread(e->segments, 1, segment_bytes, f) != segment_bytes) {
            coli_prefix_entry_free(e); cache->stats.corrupt_entries++; break;
        }
        for (uint32_t s = 0; s < e->segment_count; ++s)
            if (!coli_sequence_segment_valid(&e->segments[s])) {
                coli_prefix_entry_free(e); cache->stats.corrupt_entries++;
                fclose(f); return -1;
            }
        long long pos = (long long)coli_prefix_ftell(f);
        if (pos <= 0 || coli_prefix_fseek(f, (long long)e->snapshot_bytes, SEEK_CUR) != 0) {
            coli_prefix_entry_free(e); cache->stats.corrupt_entries++; break;
        }
        e->disk_snapshot_offset = (uint64_t)pos;
        e->disk_record_bytes = h.record_bytes;
        e->last_used = ++cache->clock;
        cache->entries[cache->count++] = e;
        cache->ram_resident_bytes += e->metadata_bytes;
        cache->disk_live_bytes += e->disk_record_bytes;
    }
    fclose(f);
    coli_prefix_enforce_ram_budget(cache, NULL);
    return 0;
}

static inline int coli_prefix_cache_init(ColiPrefixCache *cache,
                                         ColiPrefixCachePolicy policy,
                                         size_t ram_budget_bytes,
                                         uint64_t disk_budget_bytes,
                                         const char *disk_path) {
    if (!cache) return -1;
    memset(cache, 0, sizeof(*cache));
    cache->policy = policy;
    cache->ram_budget_bytes = ram_budget_bytes;
    cache->disk_budget_bytes = disk_budget_bytes;
    if (disk_path && *disk_path) {
        size_t n = strlen(disk_path);
        if (n >= sizeof(cache->disk_path)) return -1;
        memcpy(cache->disk_path, disk_path, n + 1);
    }
    if ((policy & COLI_PREFIX_CACHE_SSD) && !cache->disk_path[0])
        cache->policy = (ColiPrefixCachePolicy)(policy & ~COLI_PREFIX_CACHE_SSD);
    return coli_prefix_cache_load_index(cache);
}

static inline void coli_prefix_cache_close(ColiPrefixCache *cache) {
    if (!cache) return;
    for (size_t i = 0; i < cache->count; ++i)
        coli_prefix_entry_free(cache->entries[i]);
    memset(cache, 0, sizeof(*cache));
}

static inline ColiPrefixCacheEntry *coli_prefix_find_longest(
        ColiPrefixCache *cache, const ColiPrefixNamespace *ns,
        const int *tokens, int token_count) {
    if (!cache || !ns || !tokens || token_count <= 1) return NULL;
    ColiPrefixCacheEntry *best = NULL;
    for (size_t i = 0; i < cache->count; ++i) {
        ColiPrefixCacheEntry *e = cache->entries[i];
        if (!coli_prefix_ns_equal(&e->ns, ns) ||
            e->token_count >= (uint32_t)token_count ||
            (best && e->token_count <= best->token_count))
            continue;
        if (!memcmp(e->tokens, tokens, (size_t)e->token_count * sizeof(int)))
            best = e;
    }
    return best;
}

static inline int coli_prefix_cache_promote_blob(ColiPrefixCache *cache,
                                                  ColiPrefixCacheEntry *entry,
                                                  unsigned char *blob) {
    if (!cache || !entry || !blob) return -1;
    if (!(cache->policy & COLI_PREFIX_CACHE_RAM) ||
        entry->metadata_bytes + entry->snapshot_bytes > cache->ram_budget_bytes)
        return 0;
    entry->snapshot = blob;
    cache->ram_resident_bytes += entry->snapshot_bytes;
    coli_prefix_enforce_ram_budget(cache, entry);
    return entry->snapshot == blob ? 1 : 0;
}

static inline int coli_prefix_cache_restore(
        ColiPrefixCache *cache, const ColiPrefixNamespace *ns,
        const int *tokens, int token_count,
        const ColiSequenceStateAdapter *adapter,
        int *matched_tokens) {
    if (matched_tokens) *matched_tokens = 0;
    if (!cache || !ns || !tokens || !adapter || cache->policy == COLI_PREFIX_CACHE_OFF)
        return 0;
    cache->stats.lookups++;
    ColiPrefixCacheEntry *e = coli_prefix_find_longest(cache, ns, tokens, token_count);
    if (!e) { cache->stats.misses++; return 0; }
    e->last_used = ++cache->clock;

    unsigned char *owned = NULL;
    const unsigned char *blob = e->snapshot;
    int from_ram = blob != NULL;
    if (!blob) {
        int rr = coli_prefix_read_snapshot_from_disk(cache, e, &owned);
        if (rr != 0) {
            cache->stats.corrupt_entries++;
            cache->stats.misses++;
            return 0;
        }
        blob = owned;
    }

    ColiSequenceSnapshotLayout layout;
    memset(&layout, 0, sizeof(layout));
    layout.info.state_abi = e->ns.state_abi;
    layout.info.absolute_position = e->absolute_position;
    layout.info.segment_count = e->segment_count;
    layout.segments = e->segments;
    layout.segment_capacity = e->segment_count;
    if (coli_sequence_snapshot_restore_all(adapter, &layout, blob,
                                           e->snapshot_bytes) != 0) {
        free(owned); cache->stats.misses++; return 0;
    }
    if (from_ram) cache->stats.hits_ram++;
    else cache->stats.hits_ssd++;
    cache->stats.matched_tokens += e->token_count;
    cache->stats.restore_bytes += e->snapshot_bytes;
    if (matched_tokens) *matched_tokens = (int)e->token_count;

    if (owned) {
        int promoted = coli_prefix_cache_promote_blob(cache, e, owned);
        if (promoted != 1) free(owned);
    }
    return 1;
}

static inline size_t coli_prefix_find_exact_index(
        const ColiPrefixCache *cache, const ColiPrefixNamespace *ns,
        const int *tokens, uint32_t token_count) {
    if (!cache || !ns || !tokens) return cache ? cache->count : 0;
    for (size_t i = 0; i < cache->count; ++i) {
        const ColiPrefixCacheEntry *e = cache->entries[i];
        if (e->token_count == token_count && coli_prefix_ns_equal(&e->ns, ns) &&
            !memcmp(e->tokens, tokens, (size_t)token_count * sizeof(int)))
            return i;
    }
    return cache->count;
}

static inline int coli_prefix_cache_store(
        ColiPrefixCache *cache, const ColiPrefixNamespace *ns,
        const int *tokens, uint32_t token_count, uint64_t absolute_position,
        const ColiSequenceStateAdapter *adapter) {
    if (!cache || !ns || !tokens || !token_count || !adapter ||
        cache->policy == COLI_PREFIX_CACHE_OFF || token_count > COLI_PREFIX_CACHE_MAX_TOKENS)
        return 0;
    if (coli_prefix_find_exact_index(cache, ns, tokens, token_count) != cache->count)
        return 0;

    ColiSequenceStateInfo info;
    memset(&info, 0, sizeof(info));
    if (!adapter->ops || !adapter->ops->describe ||
        adapter->ops->describe(adapter->ctx, absolute_position, &info, NULL, 0) != 0 ||
        info.state_abi != ns->state_abi || !info.segment_count ||
        info.segment_count > COLI_PREFIX_CACHE_MAX_SEGMENTS)
        return -1;

    ColiPrefixCacheEntry *e = (ColiPrefixCacheEntry *)calloc(1, sizeof(*e));
    if (!e) return -1;
    e->ns = *ns;
    e->token_count = token_count;
    e->absolute_position = absolute_position;
    e->segment_count = (uint32_t)info.segment_count;
    size_t token_bytes, segment_bytes, meta = sizeof(*e);
    if (coli_prefix_size_mul(token_count, sizeof(int), &token_bytes) != 0 ||
        coli_prefix_size_mul(e->segment_count, sizeof(ColiSequenceSegmentDesc),
                             &segment_bytes) != 0 ||
        coli_prefix_size_add(meta, token_bytes, &meta) != 0 ||
        coli_prefix_size_add(meta, segment_bytes, &meta) != 0) {
        coli_prefix_entry_free(e); return -1;
    }
    e->metadata_bytes = meta;
    e->tokens = (int *)malloc(token_bytes);
    e->segments = (ColiSequenceSegmentDesc *)malloc(segment_bytes);
    if (!e->tokens || !e->segments) { coli_prefix_entry_free(e); return -1; }
    memcpy(e->tokens, tokens, token_bytes);

    ColiSequenceSnapshotLayout layout;
    memset(&layout, 0, sizeof(layout));
    layout.segments = e->segments;
    layout.segment_capacity = e->segment_count;
    if (coli_sequence_snapshot_layout(adapter, absolute_position, &layout) != 0) {
        coli_prefix_entry_free(e); return -1;
    }
    uint64_t blob_bytes = coli_sequence_snapshot_bytes(&layout);
    if (!blob_bytes || blob_bytes == UINT64_MAX || blob_bytes > SIZE_MAX) {
        coli_prefix_entry_free(e); return -1;
    }
    e->snapshot_bytes = (size_t)blob_bytes;
    e->snapshot = (unsigned char *)malloc(e->snapshot_bytes);
    if (!e->snapshot ||
        coli_sequence_snapshot_read_all(adapter, &layout, e->snapshot,
                                        e->snapshot_bytes) != 0) {
        coli_prefix_entry_free(e); return -1;
    }
    e->payload_crc32c = coli_prefix_entry_crc(e, e->snapshot);
    e->disk_record_bytes = coli_prefix_record_bytes(
        e->token_count, e->segment_count, e->snapshot_bytes);
    e->last_used = ++cache->clock;

    if (cache->count == COLI_PREFIX_CACHE_MAX_ENTRIES) {
        size_t victim = coli_prefix_oldest_index(cache, NULL, 0);
        coli_prefix_cache_drop_index(cache, victim, 1);
    }
    cache->entries[cache->count++] = e;
    cache->ram_resident_bytes += coli_prefix_entry_ram_bytes(e);
    cache->disk_live_bytes += e->disk_record_bytes;

    if ((cache->policy & COLI_PREFIX_CACHE_SSD) && cache->disk_budget_bytes &&
        e->disk_record_bytes + sizeof(ColiPrefixFileHeader) > cache->disk_budget_bytes) {
        /* Too large for the declared cold tier. Keep only if RAM policy can own it. */
        cache->disk_live_bytes -= e->disk_record_bytes;
        e->disk_record_bytes = 0;
        if (!(cache->policy & COLI_PREFIX_CACHE_RAM)) {
            coli_prefix_cache_drop_index(cache, cache->count - 1, 0);
            return 0;
        }
    }

    if (e->disk_record_bytes) {
        coli_prefix_enforce_disk_budget(cache, e);
        if (coli_prefix_cache_rewrite_disk(cache) != 0) {
            /* Persistence failure must not corrupt inference. The entry remains
             * a RAM optimization if RAM is enabled, otherwise discard it. */
            if (!(cache->policy & COLI_PREFIX_CACHE_RAM)) {
                size_t idx = coli_prefix_find_exact_index(cache, ns, tokens, token_count);
                if (idx < cache->count) coli_prefix_cache_drop_index(cache, idx, 0);
                return 0;
            }
            e->disk_snapshot_offset = 0;
            e->disk_record_bytes = 0;
        }
    }
    coli_prefix_enforce_ram_budget(cache, e);
    cache->stats.stores++;
    return 1;
}

static inline void coli_prefix_cache_get_stats(const ColiPrefixCache *cache,
                                                ColiPrefixCacheStats *out) {
    if (!cache || !out) return;
    *out = cache->stats;
    out->entries = cache->count;
    out->ram_resident_bytes = cache->ram_resident_bytes;
    out->ram_budget_bytes = cache->ram_budget_bytes;
    out->disk_live_bytes = cache->disk_live_bytes;
    out->disk_budget_bytes = cache->disk_budget_bytes;
}

#endif

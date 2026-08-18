#ifndef COLIBRI_V4_PREFIX_CACHE_H
#define COLIBRI_V4_PREFIX_CACHE_H

#include <stddef.h>
#include <stdint.h>

typedef struct ColiV4Engine ColiV4Engine;
typedef struct ColiV4Session ColiV4Session;
typedef struct ColiV4AttentionSnapshot ColiV4AttentionSnapshot;
typedef struct ColiV4CompressorSnapshot ColiV4CompressorSnapshot;
typedef struct ColiV4IndexerSnapshot ColiV4IndexerSnapshot;
struct ColiDeepSeekV4WindowAttentionState;
struct ColiDeepSeekV4CompressorState;
struct ColiDeepSeekV4Indexer;

typedef struct {
    uint64_t lookups;
    uint64_t hits;
    uint64_t stores;
    uint64_t evictions;
    uint64_t matched_tokens;
    uint64_t restore_bytes;
    uint64_t restore_ns;
    uint64_t store_bytes;
    uint64_t store_ns;
    size_t entries;
    size_t resident_bytes;
    size_t budget_bytes;
} ColiV4PrefixCacheStats;

size_t coli_v4_prefix_cache_budget_bytes(void);
int coli_v4_prefix_cache_restore(ColiV4Session *session,
                                 const int *prompt_ids,
                                 int prompt_tokens);
void coli_v4_prefix_cache_store(ColiV4Session *session);
void coli_v4_prefix_cache_forget_engine(ColiV4Engine *engine);
void coli_v4_prefix_cache_stats(ColiV4PrefixCacheStats *stats);

/* Borrow the exact end-of-prefill RAM entry for SESSION's retained prompt.
 * The callback runs outside the cache mutex while the entry is ref-pinned, so
 * an SSD adapter can stream the immutable native snapshots without making a
 * second full snapshot. A zero return means no exact entry was available;
 * otherwise the visitor's return value is propagated. */
typedef int (*ColiV4PrefixCacheVisitFn)(
    const int *tokens, int token_count,
    ColiV4AttentionSnapshot *const *attention, int layer_count,
    void *user_data);
int coli_v4_prefix_cache_visit_exact(ColiV4Session *session,
                                     ColiV4PrefixCacheVisitFn visitor,
                                     void *user_data);

size_t coli_v4_compressor_snapshot_bytes(const ColiV4CompressorSnapshot *snapshot);
size_t coli_v4_indexer_snapshot_bytes(const ColiV4IndexerSnapshot *snapshot);
size_t coli_v4_attention_snapshot_bytes(const ColiV4AttentionSnapshot *snapshot);
size_t coli_v4_compressor_state_snapshot_bytes(
    const struct ColiDeepSeekV4CompressorState *state);
size_t coli_v4_indexer_state_snapshot_bytes(
    const struct ColiDeepSeekV4Indexer *state);
size_t coli_v4_attention_state_snapshot_bytes(
    const struct ColiDeepSeekV4WindowAttentionState *state);

/* Native persistent-state wire codec used by the global #80 SSD framing.
 * Scalars are copied with memcpy in host byte order; the outer cache object
 * carries the endian marker, model/layout fingerprint and state ABI, so these
 * bytes are intentionally local-runtime state rather than a portable model
 * format. Every reader is length-bounded and allocation geometry is validated
 * before allocation.
 *
 * The emit functions produce exactly the same byte sequence as wire_write(),
 * but borrow the snapshot's native arrays. This is the production write path:
 * the generic SSD store can checksum/write a ref-pinned RAM entry with O(1)
 * scratch memory instead of allocating another full snapshot-sized buffer. */
typedef int (*ColiV4PrefixWireSink)(void *user_data,
                                    const void *data, size_t bytes);

size_t coli_v4_compressor_snapshot_wire_bytes(
    const ColiV4CompressorSnapshot *snapshot);
int coli_v4_compressor_snapshot_wire_emit(
    const ColiV4CompressorSnapshot *snapshot,
    ColiV4PrefixWireSink sink, void *user_data);
int coli_v4_compressor_snapshot_wire_write(
    const ColiV4CompressorSnapshot *snapshot,
    unsigned char *output, size_t output_size, size_t *written);
int coli_v4_compressor_snapshot_wire_read(
    ColiV4CompressorSnapshot **output,
    const unsigned char *input, size_t input_size, size_t *consumed);

size_t coli_v4_indexer_snapshot_wire_bytes(
    const ColiV4IndexerSnapshot *snapshot);
int coli_v4_indexer_snapshot_wire_emit(
    const ColiV4IndexerSnapshot *snapshot,
    ColiV4PrefixWireSink sink, void *user_data);
int coli_v4_indexer_snapshot_wire_write(
    const ColiV4IndexerSnapshot *snapshot,
    unsigned char *output, size_t output_size, size_t *written);
int coli_v4_indexer_snapshot_wire_read(
    ColiV4IndexerSnapshot **output,
    const unsigned char *input, size_t input_size, size_t *consumed);

size_t coli_v4_attention_snapshot_wire_bytes(
    const ColiV4AttentionSnapshot *snapshot);
int coli_v4_attention_snapshot_wire_emit(
    const ColiV4AttentionSnapshot *snapshot,
    ColiV4PrefixWireSink sink, void *user_data);
int coli_v4_attention_snapshot_wire_write(
    const ColiV4AttentionSnapshot *snapshot,
    unsigned char *output, size_t output_size, size_t *written);
int coli_v4_attention_snapshot_wire_read(
    ColiV4AttentionSnapshot **output,
    const unsigned char *input, size_t input_size, size_t *consumed);

#endif /* COLIBRI_V4_PREFIX_CACHE_H */

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

/*
 * Process-local exact prefix cache for issue #12.
 *
 * The first implementation is deliberately opt-in: V4_PREFIX_CACHE_MB must be
 * set to a positive value. The expert-store planner reserves this byte budget
 * before assigning optional RAM to dense/expert residency, so enabling prompt
 * snapshots cannot silently exceed the engine's requested memory envelope.
 *
 * A hit restores all per-layer target attention state and returns the number
 * of prompt tokens already represented by that state. Only strict prefixes are
 * returned (matched < prompt_tokens), preserving the existing generation
 * contract that always executes at least one fresh prompt token.
 */
size_t coli_v4_prefix_cache_budget_bytes(void);
int coli_v4_prefix_cache_restore(ColiV4Session *session,
                                 const int *prompt_ids,
                                 int prompt_tokens);
void coli_v4_prefix_cache_store(ColiV4Session *session);
void coli_v4_prefix_cache_forget_engine(ColiV4Engine *engine);
void coli_v4_prefix_cache_stats(ColiV4PrefixCacheStats *stats);

/* Exact allocated payload accounting for the existing V4 transaction
 * snapshots. The state variants are allocation-free preflight estimators used
 * to reserve/evict cache capacity before cloning a snapshot; they must stay
 * byte-identical to the corresponding snapshot_create allocation geometry. */
size_t coli_v4_compressor_snapshot_bytes(const ColiV4CompressorSnapshot *snapshot);
size_t coli_v4_indexer_snapshot_bytes(const ColiV4IndexerSnapshot *snapshot);
size_t coli_v4_attention_snapshot_bytes(const ColiV4AttentionSnapshot *snapshot);
size_t coli_v4_compressor_state_snapshot_bytes(
    const struct ColiDeepSeekV4CompressorState *state);
size_t coli_v4_indexer_state_snapshot_bytes(
    const struct ColiDeepSeekV4Indexer *state);
size_t coli_v4_attention_state_snapshot_bytes(
    const struct ColiDeepSeekV4WindowAttentionState *state);

#endif /* COLIBRI_V4_PREFIX_CACHE_H */

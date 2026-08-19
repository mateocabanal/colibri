#ifndef COLI_SPECULATIVE_CACHE_H
#define COLI_SPECULATIVE_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ColiSpecProposalSource {
    COLI_SPEC_SOURCE_NONE = 0,
    COLI_SPEC_SOURCE_NGRAM = 1,
    COLI_SPEC_SOURCE_OUTPUT_CACHE = 2,
    COLI_SPEC_SOURCE_MTP = 3,
    COLI_SPEC_SOURCE_DSPARK = 4,
    COLI_SPEC_SOURCE_EXTERNAL = 5,
} ColiSpecProposalSource;

typedef size_t (*ColiSpecProposeFn)(void *ctx,
                                    const int *history,
                                    size_t history_count,
                                    int *out_tokens,
                                    size_t max_tokens);

typedef struct ColiSpecProposer {
    ColiSpecProposalSource source;
    void *ctx;
    ColiSpecProposeFn propose;
} ColiSpecProposer;

typedef struct ColiSpecProposal {
    ColiSpecProposalSource source;
    size_t token_count;
} ColiSpecProposal;

static inline ColiSpecProposal coli_spec_propose(const ColiSpecProposer *proposer,
                                                 const int *history,
                                                 size_t history_count,
                                                 int *out_tokens,
                                                 size_t max_tokens) {
    ColiSpecProposal result;
    result.source = COLI_SPEC_SOURCE_NONE;
    result.token_count = 0;
    if (proposer == NULL || proposer->propose == NULL || out_tokens == NULL || max_tokens == 0) {
        return result;
    }
    result.token_count = proposer->propose(proposer->ctx, history, history_count, out_tokens, max_tokens);
    if (result.token_count > max_tokens) {
        result.token_count = 0;
        return result;
    }
    if (result.token_count != 0) {
        result.source = proposer->source;
    }
    return result;
}

typedef struct ColiSpecContinuationCache {
    int *tokens;
    size_t count;
    size_t capacity;
    int separator;
} ColiSpecContinuationCache;

static inline int coli_spec_cache_valid(const ColiSpecContinuationCache *cache) {
    return cache != NULL && cache->separator < 0 &&
           (cache->capacity == 0 || cache->tokens != NULL) &&
           cache->count <= cache->capacity;
}

static inline int coli_spec_cache_append_span(ColiSpecContinuationCache *cache,
                                              const int *tokens,
                                              size_t token_count) {
    size_t required;
    size_t i;
    int needs_separator;

    if (!coli_spec_cache_valid(cache) || tokens == NULL || token_count == 0) {
        return 0;
    }
    for (i = 0; i < token_count; ++i) {
        if (tokens[i] < 0) {
            return 0;
        }
    }

    needs_separator = cache->count != 0 && cache->tokens[cache->count - 1] != cache->separator;
    if (token_count > SIZE_MAX - (size_t)needs_separator) {
        return 0;
    }
    required = token_count + (size_t)needs_separator;
    if (required > cache->capacity - cache->count) {
        return 0;
    }

    if (needs_separator) {
        cache->tokens[cache->count++] = cache->separator;
    }
    for (i = 0; i < token_count; ++i) {
        cache->tokens[cache->count++] = tokens[i];
    }
    return 1;
}

static inline size_t coli_spec_cache_propose(const int *corpus,
                                             size_t corpus_count,
                                             int separator,
                                             const int *history,
                                             size_t history_count,
                                             int *out_tokens,
                                             size_t max_tokens,
                                             size_t min_match,
                                             size_t max_match) {
    size_t match_len;

    if (corpus == NULL || corpus_count == 0 || separator >= 0 || history == NULL ||
        history_count == 0 || out_tokens == NULL || max_tokens == 0 ||
        min_match == 0 || max_match < min_match) {
        return 0;
    }
    if (max_match > history_count) {
        max_match = history_count;
    }
    if (max_match < min_match) {
        return 0;
    }

    /* Longest suffix wins. For equal-length matches, scan from the end so the
     * most recently appended occurrence supplies the continuation. */
    match_len = max_match;
    for (;;) {
        size_t start;
        if (corpus_count > match_len) {
            start = corpus_count - match_len - 1;
            for (;;) {
                size_t j;
                int match = 1;
                size_t continuation;
                size_t produced = 0;

                /* A matched suffix may not cross a span separator. */
                for (j = 0; j < match_len; ++j) {
                    int token = corpus[start + j];
                    if (token == separator || token != history[history_count - match_len + j]) {
                        match = 0;
                        break;
                    }
                }
                continuation = start + match_len;
                if (match && continuation < corpus_count && corpus[continuation] != separator) {
                    while (continuation < corpus_count && produced < max_tokens) {
                        int token = corpus[continuation++];
                        if (token == separator) {
                            break;
                        }
                        out_tokens[produced++] = token;
                    }
                    if (produced != 0) {
                        return produced;
                    }
                }
                if (start == 0) {
                    break;
                }
                --start;
            }
        }
        if (match_len == min_match) {
            break;
        }
        --match_len;
    }
    return 0;
}

typedef struct ColiSpecCacheProposerCtx {
    const int *corpus;
    size_t corpus_count;
    int separator;
    size_t min_match;
    size_t max_match;
} ColiSpecCacheProposerCtx;

static inline size_t coli_spec_cache_proposer(void *opaque,
                                              const int *history,
                                              size_t history_count,
                                              int *out_tokens,
                                              size_t max_tokens) {
    const ColiSpecCacheProposerCtx *ctx = (const ColiSpecCacheProposerCtx *)opaque;
    if (ctx == NULL) {
        return 0;
    }
    return coli_spec_cache_propose(ctx->corpus,
                                   ctx->corpus_count,
                                   ctx->separator,
                                   history,
                                   history_count,
                                   out_tokens,
                                   max_tokens,
                                   ctx->min_match,
                                   ctx->max_match);
}

#ifdef __cplusplus
}
#endif

#endif /* COLI_SPECULATIVE_CACHE_H */

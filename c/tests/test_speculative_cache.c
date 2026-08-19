#include <stdio.h>
#include <string.h>

#include "../speculative_cache.h"

static int failures = 0;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

int main(void) {
    int out[8];

    {
        static const int corpus[] = {90, 91, 13, 77, 78, -1, 10, 11, 12, 13, 14, 15, 16, 17};
        static const int history[] = {1, 2, 3, 10, 11, 12, 13};
        size_t n = coli_spec_cache_propose(corpus, sizeof(corpus) / sizeof(corpus[0]), -1,
                                           history, sizeof(history) / sizeof(history[0]),
                                           out, 4, 3, 8);
        check(n == 4, "longest suffix proposes four tokens");
        check(n == 4 && out[0] == 14 && out[1] == 15 && out[2] == 16 && out[3] == 17,
              "longest suffix beats shorter decoy");
    }

    {
        static const int corpus[] = {7, 8, 9, 100, 101, -1, 7, 8, 9, 200, 201};
        static const int history[] = {0, 7, 8, 9};
        size_t n = coli_spec_cache_propose(corpus, sizeof(corpus) / sizeof(corpus[0]), -1,
                                           history, sizeof(history) / sizeof(history[0]),
                                           out, 2, 3, 8);
        check(n == 2, "recency case proposes two tokens");
        check(n == 2 && out[0] == 200 && out[1] == 201,
              "most recent equal-length occurrence wins");
    }

    {
        static const int corpus[] = {5, 6, 7, 42, -1, 99, 98};
        static const int history[] = {4, 5, 6, 7};
        size_t n = coli_spec_cache_propose(corpus, sizeof(corpus) / sizeof(corpus[0]), -1,
                                           history, sizeof(history) / sizeof(history[0]),
                                           out, 8, 3, 8);
        check(n == 1 && out[0] == 42, "proposal stops at span separator");
    }

    {
        int storage[16];
        ColiSpecContinuationCache cache = {storage, 0, sizeof(storage) / sizeof(storage[0]), -1};
        static const int first[] = {1, 2, 3, 4, 5};
        static const int second[] = {9, 2, 3, 7, 8};
        static const int history[] = {0, 2, 3};
        check(coli_spec_cache_append_span(&cache, first, sizeof(first) / sizeof(first[0])) == 1,
              "first output span appended");
        check(coli_spec_cache_append_span(&cache, second, sizeof(second) / sizeof(second[0])) == 1,
              "second output span appended");
        check(cache.count == 11 && storage[5] == -1, "spans separated automatically");
        {
            size_t n = coli_spec_cache_propose(cache.tokens, cache.count, cache.separator,
                                               history, sizeof(history) / sizeof(history[0]),
                                               out, 2, 2, 4);
            check(n == 2 && out[0] == 7 && out[1] == 8,
                  "mutable output cache feeds most recent continuation");
        }
    }

    {
        int storage[4] = {1, 2, 3, 4};
        ColiSpecContinuationCache cache = {storage, 4, 4, -1};
        static const int extra[] = {5, 6};
        int before[4];
        memcpy(before, storage, sizeof(before));
        check(coli_spec_cache_append_span(&cache, extra, 2) == 0,
              "append rejects insufficient capacity");
        check(cache.count == 4 && memcmp(before, storage, sizeof(before)) == 0,
              "failed append is fail-clean");
    }

    {
        static const int corpus[] = {3, 4, 5, 6, 7};
        static const int history[] = {3, 4, 5};
        ColiSpecCacheProposerCtx ctx = {
            corpus, sizeof(corpus) / sizeof(corpus[0]), -1, 3, 8
        };
        ColiSpecProposer proposer = {
            COLI_SPEC_SOURCE_OUTPUT_CACHE, &ctx, coli_spec_cache_proposer
        };
        ColiSpecProposalRequest request = {
            history, sizeof(history) / sizeof(history[0]), 1234
        };
        ColiSpecProposal proposal = coli_spec_propose(&proposer, &request, out, 2);
        check(proposal.source == COLI_SPEC_SOURCE_OUTPUT_CACHE,
              "proposal retains source identity");
        check(proposal.token_count == 2 && out[0] == 6 && out[1] == 7,
              "generic proposer dispatches continuation cache");
    }

    if (failures != 0) {
        fprintf(stderr, "speculative_cache: %d failure(s)\n", failures);
        return 1;
    }
    printf("speculative_cache: proposer, suffix lookup, recency, append, caps ok\n");
    return 0;
}

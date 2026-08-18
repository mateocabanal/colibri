#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../deepseek_v4_internal.h"
#include "../coli_v4_prefix_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int decode_token_string(const ColiV4Session *session,
                               const int *ids, int count,
                               char **output) {
    char *text = NULL;
    size_t length = 0, capacity = 0;
    for (int index = 0; index < count; index++) {
        char piece[512];
        int bytes = tok_decode((Tok *)&session->tokenizer, &ids[index], 1,
                               piece, (int)sizeof(piece));
        if (bytes < 0 || length > SIZE_MAX - (size_t)bytes - 1) {
            free(text);
            return -1;
        }
        size_t needed = length + (size_t)bytes + 1;
        if (needed > capacity) {
            size_t next = capacity ? capacity : 256;
            while (next < needed) {
                if (next > SIZE_MAX / 2) { next = needed; break; }
                next *= 2;
            }
            char *grown = realloc(text, next);
            if (!grown) { free(text); return -1; }
            text = grown;
            capacity = next;
        }
        memcpy(text + length, piece, (size_t)bytes);
        length += (size_t)bytes;
        text[length] = '\0';
    }
    if (!text) {
        text = calloc(1, 1);
        if (!text) return -1;
    }
    *output = text;
    return 0;
}

static char *append_text(const char *prefix, const char *suffix) {
    size_t prefix_len = strlen(prefix);
    size_t suffix_len = strlen(suffix);
    if (prefix_len > SIZE_MAX - suffix_len - 1) return NULL;
    char *output = malloc(prefix_len + suffix_len + 1);
    if (!output) return NULL;
    memcpy(output, prefix, prefix_len);
    memcpy(output + prefix_len, suffix, suffix_len + 1);
    return output;
}

static int generated_text(ColiV4Session *session, char **output) {
    size_t capacity = (size_t)session->text_length + 1;
    char *text = malloc(capacity);
    if (!text) return -1;
    size_t length = 0;
    if (coli_v4_session_generated_text(session, text, capacity, &length) != 0) {
        free(text);
        return -1;
    }
    *output = text;
    return 0;
}

static int open_engine_with_limit(ColiV4Engine **engine, const char *model,
                                  uint64_t memory_limit_bytes,
                                  char *error, size_t error_size) {
    ColiV4EngineOpenOptions options = {
        .target_model_dir = model,
        .memory_limit_bytes = memory_limit_bytes,
        .context_tokens = 128,
        .pin_slots_per_layer = -1,
        .no_dspark = 1,
    };
    return coli_v4_engine_open(engine, &options, error, error_size);
}

static int open_engine(ColiV4Engine **engine, const char *model,
                       char *error, size_t error_size) {
    return open_engine_with_limit(engine, model, 0, error, error_size);
}

static int open_session(ColiV4Session **session, ColiV4Engine *engine,
                        char *error, size_t error_size) {
    ColiV4SessionCreateOptions options = {
        .max_prompt_tokens = 128,
        .max_new_tokens_cap = 8,
    };
    return coli_v4_session_create(session, engine, &options, error, error_size);
}

static int run_generation(ColiV4Session *session, const char *prompt,
                          char *error, size_t error_size) {
    ColiV4SessionGenerateOptions options = {
        .max_new_tokens = 4,
        .no_dspark = 1,
    };
    ColiV4SessionGenerateStats stats = {0};
    return coli_v4_session_generate(session, prompt, strlen(prompt), &options,
                                    NULL, NULL, &stats, error, error_size);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR INITIAL_PROMPT\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *initial_prompt = argv[2];
    char error[1024] = {0};
    ColiV4Engine *shared_engine = NULL, *cold_engine = NULL, *rejected_engine = NULL;
    ColiV4Session *first = NULL, *equal = NULL, *extended = NULL;
    ColiV4Session *longest = NULL, *cold = NULL;
    char *extension_token = NULL, *prompt_plus_one = NULL, *prompt_plus_two = NULL;
    char *longest_text = NULL, *cold_text = NULL;
    uint64_t reserve = (uint64_t)coli_v4_prefix_cache_budget_bytes();
    const uint64_t explicit_limit = 3ULL * 1024ULL * 1024ULL * 1024ULL;
    int status = 1;

    if (!reserve) {
        fprintf(stderr, "V4_PREFIX_CACHE_MB must be positive for this test\n");
        goto cleanup;
    }

    if (open_engine(&shared_engine, model, error, sizeof(error)) ||
        open_session(&first, shared_engine, error, sizeof(error)) ||
        open_session(&equal, shared_engine, error, sizeof(error)) ||
        open_session(&extended, shared_engine, error, sizeof(error)) ||
        open_session(&longest, shared_engine, error, sizeof(error))) {
        fprintf(stderr, "shared engine/session open failed: %s\n", error);
        goto cleanup;
    }

    if (run_generation(first, initial_prompt, error, sizeof(error))) {
        fprintf(stderr, "first generation failed: %s\n", error);
        goto cleanup;
    }
    if (!first->prompt_ids || first->prompt_count <= 0) {
        fprintf(stderr, "first session did not retain tokenized prompt\n");
        goto cleanup;
    }
    int base_tokens = first->prompt_count;

    ColiV4PrefixCacheStats after_first = {0};
    coli_v4_prefix_cache_stats(&after_first);
    if (!after_first.stores || !after_first.entries) {
        fprintf(stderr,
                "first prefill was not admitted: stores=%llu entries=%zu\n",
                (unsigned long long)after_first.stores, after_first.entries);
        goto cleanup;
    }

    /* Exact equality is deliberately not a hit: the generation contract always
     * executes at least one fresh prompt token, so only strict prefixes qualify. */
    if (run_generation(equal, initial_prompt, error, sizeof(error))) {
        fprintf(stderr, "equal-prompt generation failed: %s\n", error);
        goto cleanup;
    }
    if (equal->prefix_reused != 0) {
        fprintf(stderr, "equal prompt unexpectedly reused %d tokens\n",
                equal->prefix_reused);
        goto cleanup;
    }

    /* The cache snapshot is taken immediately after prefill, so it represents
     * exactly the request prompt. Append a known round-trippable tiny token to
     * create P+X and require a fresh session to restore P. */
    if (decode_token_string(first, first->prompt_ids, 1, &extension_token)) {
        fprintf(stderr, "failed to decode extension token\n");
        goto cleanup;
    }
    prompt_plus_one = append_text(initial_prompt, extension_token);
    if (!prompt_plus_one) {
        fprintf(stderr, "out of memory building first extension\n");
        goto cleanup;
    }
    if (run_generation(extended, prompt_plus_one, error, sizeof(error))) {
        fprintf(stderr, "first extension generation failed: %s\n", error);
        goto cleanup;
    }
    if (extended->prefix_reused != base_tokens) {
        fprintf(stderr,
                "cross-session cache did not restore base prefix: got=%d expected=%d\n",
                extended->prefix_reused, base_tokens);
        goto cleanup;
    }
    if (!extended->prompt_ids || extended->prompt_count <= base_tokens) {
        fprintf(stderr, "extended request did not grow the prompt token prefix\n");
        goto cleanup;
    }
    int extended_tokens = extended->prompt_count;

    ColiV4PrefixCacheStats after_extended = {0};
    coli_v4_prefix_cache_stats(&after_extended);
    if (after_extended.entries < 2 || after_extended.stores < 2) {
        fprintf(stderr,
                "second prefix was not retained: stores=%llu entries=%zu\n",
                (unsigned long long)after_extended.stores,
                after_extended.entries);
        goto cleanup;
    }

    /* Now P and P+X are both resident. P+X+Y must restore the longer P+X
     * snapshot, proving selection is longest-prefix rather than first-match. */
    prompt_plus_two = append_text(prompt_plus_one, extension_token);
    if (!prompt_plus_two) {
        fprintf(stderr, "out of memory building second extension\n");
        goto cleanup;
    }
    if (run_generation(longest, prompt_plus_two, error, sizeof(error))) {
        fprintf(stderr, "longest-prefix generation failed: %s\n", error);
        goto cleanup;
    }
    if (longest->prefix_reused != extended_tokens) {
        fprintf(stderr,
                "longest-prefix selection failed: got=%d expected=%d base=%d\n",
                longest->prefix_reused, extended_tokens, base_tokens);
        goto cleanup;
    }
    if (generated_text(longest, &longest_text)) {
        fprintf(stderr, "cannot decode longest-prefix output\n");
        goto cleanup;
    }

    /* Use an explicit API memory envelope for the cold correctness oracle and
     * assert the planner leaves the resident prefix budget inside that same
     * envelope. This is the Apple-UMA safety contract: cache bytes are not a
     * hidden allocation on top of --memory-gb. */
    if (open_engine_with_limit(&cold_engine, model, explicit_limit,
                               error, sizeof(error)) ||
        open_session(&cold, cold_engine, error, sizeof(error))) {
        fprintf(stderr, "cold engine/session open failed: %s\n", error);
        goto cleanup;
    }
    ColiV4EngineMemorySummary memory = {0};
    coli_v4_engine_memory_summary(cold_engine, &memory);
    if (!memory.projected_bytes || reserve > explicit_limit ||
        memory.projected_bytes > explicit_limit - reserve) {
        fprintf(stderr,
                "prefix-cache memory envelope invalid: projected=%llu reserve=%llu limit=%llu\n",
                (unsigned long long)memory.projected_bytes,
                (unsigned long long)reserve,
                (unsigned long long)explicit_limit);
        goto cleanup;
    }

    if (run_generation(cold, prompt_plus_two, error, sizeof(error))) {
        fprintf(stderr, "cold generation failed: %s\n", error);
        goto cleanup;
    }
    if (cold->prefix_reused != 0) {
        fprintf(stderr, "different engine unexpectedly reused %d tokens\n",
                cold->prefix_reused);
        goto cleanup;
    }
    if (generated_text(cold, &cold_text)) {
        fprintf(stderr, "cannot decode cold output\n");
        goto cleanup;
    }
    if (strcmp(longest_text, cold_text)) {
        fprintf(stderr,
                "longest-prefix restored output diverged from cold prefill:\nrestored=%s\ncold=%s\n",
                longest_text, cold_text);
        goto cleanup;
    }

    /* A user limit no larger than the explicit cache reserve must fail for the
     * cache-reserve reason, rather than allowing planning to proceed and exceed
     * the requested process envelope later. */
    memset(error, 0, sizeof(error));
    if (!open_engine_with_limit(&rejected_engine, model, reserve,
                                error, sizeof(error))) {
        fprintf(stderr, "cache-exhausted memory limit unexpectedly opened\n");
        goto cleanup;
    }
    if (!strstr(error, "prefix-cache reserve")) {
        fprintf(stderr,
                "cache-exhausted limit failed for wrong reason: %s\n", error);
        goto cleanup;
    }

    ColiV4PrefixCacheStats cache = {0};
    coli_v4_prefix_cache_stats(&cache);
    uint64_t expected_matched = (uint64_t)base_tokens + (uint64_t)extended_tokens;
    if (cache.hits < 2 || cache.matched_tokens < expected_matched ||
        !cache.restore_bytes || !cache.restore_ns) {
        fprintf(stderr,
                "prefix-cache telemetry incomplete: hits=%llu matched=%llu expected_matched=%llu bytes=%llu ns=%llu\n",
                (unsigned long long)cache.hits,
                (unsigned long long)cache.matched_tokens,
                (unsigned long long)expected_matched,
                (unsigned long long)cache.restore_bytes,
                (unsigned long long)cache.restore_ns);
        goto cleanup;
    }

    printf("PASS cross-session prefix cache: base=%d extended=%d longest=%d prompt tokens, %.3f MiB restored in %.3f ms, output identical to cold prefill; memory projected=%.3fGiB reserve=%.3fGiB limit=%.3fGiB\n",
           base_tokens, extended_tokens, longest->prefix_reused,
           cache.restore_bytes / (1024.0 * 1024.0),
           cache.restore_ns / 1.0e6,
           memory.projected_bytes / 1073741824.0,
           reserve / 1073741824.0,
           explicit_limit / 1073741824.0);
    status = 0;

cleanup:
    free(extension_token);
    free(prompt_plus_one);
    free(prompt_plus_two);
    free(longest_text);
    free(cold_text);
    if (first) coli_v4_session_destroy(first);
    if (equal) coli_v4_session_destroy(equal);
    if (extended) coli_v4_session_destroy(extended);
    if (longest) coli_v4_session_destroy(longest);
    if (cold) coli_v4_session_destroy(cold);
    if (shared_engine) coli_v4_engine_destroy(shared_engine);
    if (cold_engine) coli_v4_engine_destroy(cold_engine);
    if (rejected_engine) coli_v4_engine_destroy(rejected_engine);
    return status;
}

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../deepseek_v4_internal.h"
#include "../coli_v4_prefix_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int coli_v4_attention_snapshot_restore_fresh(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiV4AttentionSnapshot *snapshot,
    const ColiDeepSeekV4Config *config, int layer);

typedef struct {
    int ids[16];
    float logits[16];
    int count;
} TokenCapture;

static int capture_token(void *user_data, int token, float logit,
                         int position, int ordinal) {
    (void)position;
    (void)ordinal;
    TokenCapture *capture = (TokenCapture *)user_data;
    if (!capture || capture->count >= (int)(sizeof(capture->ids) /
                                            sizeof(capture->ids[0])))
        return 1;
    capture->ids[capture->count] = token;
    capture->logits[capture->count] = logit;
    capture->count++;
    return 0;
}

static int same_tokens(const TokenCapture *a, const TokenCapture *b) {
    return a && b && a->count == b->count && a->count > 0 &&
           memcmp(a->ids, b->ids, (size_t)a->count * sizeof(a->ids[0])) == 0;
}

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
                          TokenCapture *capture,
                          ColiV4SessionGenerateStats *stats,
                          char *error, size_t error_size) {
    ColiV4SessionGenerateOptions options = {
        .max_new_tokens = 4,
        .no_dspark = 1,
    };
    if (capture) memset(capture, 0, sizeof(*capture));
    if (stats) memset(stats, 0, sizeof(*stats));
    return coli_v4_session_generate(session, prompt, strlen(prompt), &options,
                                    capture_token, capture, stats,
                                    error, error_size);
}

/* The persistent codec must preserve every byte that the existing native
 * snapshot/restore API considers sequence state. Restore each serialized layer
 * into a virgin session, snapshot it again, and require a byte-identical wire
 * encoding. This catches omitted compressor/indexer/recurrent fields without
 * relying on a production-size model. */
static int check_wire_roundtrip(ColiV4Session *source,
                                ColiV4Session *destination) {
    if (!source || !destination || !source->attention || !destination->attention)
        return -1;
    for (int layer = 0; layer < source->config.num_hidden_layers; layer++) {
        ColiV4AttentionSnapshot *before = NULL, *decoded = NULL, *after = NULL;
        unsigned char *wire = NULL, *wire_after = NULL;
        size_t need = 0, wrote = 0, consumed = 0, need_after = 0, wrote_after = 0;
        int ok = 0;
        if (coli_v4_attention_snapshot_create(source->attention[layer], &before))
            goto layer_done;
        need = coli_v4_attention_snapshot_wire_bytes(before);
        if (!need || need == SIZE_MAX) goto layer_done;
        wire = malloc(need);
        if (!wire || coli_v4_attention_snapshot_wire_write(
                before, wire, need, &wrote) || wrote != need)
            goto layer_done;
        if (coli_v4_attention_snapshot_wire_read(
                &decoded, wire, need, &consumed) || consumed != need)
            goto layer_done;
        if (coli_v4_attention_snapshot_restore_fresh(
                destination->attention[layer], decoded,
                &destination->config, layer))
            goto layer_done;
        if (coli_v4_attention_snapshot_create(destination->attention[layer], &after))
            goto layer_done;
        need_after = coli_v4_attention_snapshot_wire_bytes(after);
        if (need_after != need) goto layer_done;
        wire_after = malloc(need_after);
        if (!wire_after || coli_v4_attention_snapshot_wire_write(
                after, wire_after, need_after, &wrote_after) ||
            wrote_after != need_after || memcmp(wire, wire_after, need))
            goto layer_done;
        ok = 1;
layer_done:
        free(wire_after);
        free(wire);
        coli_v4_attention_snapshot_destroy(after);
        coli_v4_attention_snapshot_destroy(decoded);
        coli_v4_attention_snapshot_destroy(before);
        if (!ok) {
            fprintf(stderr, "V4 prefix wire round-trip failed at layer %d\n", layer);
            return -1;
        }
    }
    return 0;
}

static int check_disabled_mode(const char *model, const char *prompt,
                               char *error, size_t error_size) {
    ColiV4Engine *engine = NULL;
    ColiV4Session *session = NULL;
    TokenCapture first = {0}, second = {0};
    ColiV4SessionGenerateStats first_stats = {0}, second_stats = {0};
    int status = 1;

    if (open_engine(&engine, model, error, error_size) ||
        open_session(&session, engine, error, error_size)) {
        fprintf(stderr, "disabled-mode open failed: %s\n", error);
        goto cleanup;
    }
    if (run_generation(session, prompt, &first, &first_stats,
                       error, error_size) ||
        run_generation(session, prompt, &second, &second_stats,
                       error, error_size)) {
        fprintf(stderr, "disabled-mode generation failed: %s\n", error);
        goto cleanup;
    }
    if (session->prefix_reused || first_stats.prefix_reused_tokens ||
        second_stats.prefix_reused_tokens || first_stats.prefix_ram_hits ||
        second_stats.prefix_ram_hits || first_stats.prefix_ram_restore_bytes ||
        second_stats.prefix_ram_restore_bytes) {
        fprintf(stderr,
                "disabled cache reused state: session=%d first=%d second=%d hits=%llu/%llu bytes=%llu/%llu\n",
                session->prefix_reused,
                first_stats.prefix_reused_tokens,
                second_stats.prefix_reused_tokens,
                (unsigned long long)first_stats.prefix_ram_hits,
                (unsigned long long)second_stats.prefix_ram_hits,
                (unsigned long long)first_stats.prefix_ram_restore_bytes,
                (unsigned long long)second_stats.prefix_ram_restore_bytes);
        goto cleanup;
    }
    if (!same_tokens(&first, &second)) {
        fprintf(stderr, "disabled cache changed deterministic token ids\n");
        goto cleanup;
    }
    ColiV4PrefixCacheStats cache = {0};
    coli_v4_prefix_cache_stats(&cache);
    if (cache.budget_bytes || cache.hits || cache.stores || cache.entries ||
        cache.resident_bytes) {
        fprintf(stderr,
                "disabled cache retained resources: budget=%zu hits=%llu stores=%llu entries=%zu resident=%zu\n",
                cache.budget_bytes, (unsigned long long)cache.hits,
                (unsigned long long)cache.stores, cache.entries,
                cache.resident_bytes);
        goto cleanup;
    }
    printf("PASS cache-disabled parity: first_token=%d generated=%d reuse=0 budget=0\n",
           first.ids[0], first.count);
    status = 0;

cleanup:
    if (session) coli_v4_session_destroy(session);
    if (engine) coli_v4_engine_destroy(engine);
    return status;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR INITIAL_PROMPT\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *initial_prompt = argv[2];
    char error[1024] = {0};
    uint64_t reserve = (uint64_t)coli_v4_prefix_cache_budget_bytes();
    if (!reserve) {
        const char *mode = getenv("COLI_PREFIX_CACHE");
        if (mode && !strcmp(mode, "off"))
            return check_disabled_mode(model, initial_prompt,
                                       error, sizeof(error));
        fprintf(stderr,
                "set V4_PREFIX_CACHE_MB positive for enabled test, or COLI_PREFIX_CACHE=off for disabled test\n");
        return 2;
    }

    ColiV4Engine *shared_engine = NULL, *cold_engine = NULL;
    ColiV4Engine *rejected_engine = NULL;
    ColiV4Session *first = NULL, *equal = NULL, *extended = NULL, *codec = NULL;
    ColiV4Session *longest = NULL, *cold = NULL, *miss = NULL;
    char *extension_token = NULL, *prompt_plus_one = NULL, *prompt_plus_two = NULL;
    char *longest_text = NULL, *cold_text = NULL;
    int *divergent_ids = NULL;
    TokenCapture first_tokens = {0}, equal_tokens = {0};
    TokenCapture extended_capture = {0}, warm_tokens = {0}, cold_tokens = {0};
    ColiV4SessionGenerateStats first_stats = {0}, equal_stats = {0};
    ColiV4SessionGenerateStats extended_stats = {0};
    ColiV4SessionGenerateStats warm_stats = {0}, cold_stats = {0};
    const uint64_t explicit_limit = 3ULL * 1024ULL * 1024ULL * 1024ULL;
    int status = 1;

    if (open_engine(&shared_engine, model, error, sizeof(error)) ||
        open_session(&first, shared_engine, error, sizeof(error)) ||
        open_session(&equal, shared_engine, error, sizeof(error)) ||
        open_session(&extended, shared_engine, error, sizeof(error)) ||
        open_session(&codec, shared_engine, error, sizeof(error)) ||
        open_session(&longest, shared_engine, error, sizeof(error)) ||
        open_session(&miss, shared_engine, error, sizeof(error))) {
        fprintf(stderr, "shared engine/session open failed: %s\n", error);
        goto cleanup;
    }

    if (run_generation(first, initial_prompt, &first_tokens, &first_stats,
                       error, sizeof(error))) {
        fprintf(stderr, "first generation failed: %s\n", error);
        goto cleanup;
    }
    if (!first->prompt_ids || first->prompt_count < 2 ||
        first_stats.prefix_reused_tokens || first_stats.prefix_ram_hits) {
        fprintf(stderr,
                "first session invalid: prompt=%d reused=%d hits=%llu\n",
                first->prompt_count, first_stats.prefix_reused_tokens,
                (unsigned long long)first_stats.prefix_ram_hits);
        goto cleanup;
    }
    if (check_wire_roundtrip(first, codec)) goto cleanup;
    int base_tokens = first->prompt_count;

    ColiV4PrefixCacheStats after_first = {0};
    coli_v4_prefix_cache_stats(&after_first);
    if (!after_first.stores || !after_first.entries) {
        fprintf(stderr,
                "first prefill was not admitted: stores=%llu entries=%zu\n",
                (unsigned long long)after_first.stores, after_first.entries);
        goto cleanup;
    }

    /* Equal length cannot restore an end-of-prefill attention snapshot because
     * generation still needs the final prompt hidden row. It must cold-prefill
     * and remain token-identical. The exact same request is tested warm below by
     * restoring a strict cached prefix and comparing its next argmax to cold. */
    if (run_generation(equal, initial_prompt, &equal_tokens, &equal_stats,
                       error, sizeof(error))) {
        fprintf(stderr, "equal-prompt generation failed: %s\n", error);
        goto cleanup;
    }
    if (equal->prefix_reused != 0 || equal_stats.prefix_reused_tokens != 0 ||
        equal_stats.prefix_ram_hits != 0 ||
        !same_tokens(&first_tokens, &equal_tokens)) {
        fprintf(stderr,
                "equal prompt fallback failed: reused=%d stats=%d hits=%llu token_identity=%d\n",
                equal->prefix_reused, equal_stats.prefix_reused_tokens,
                (unsigned long long)equal_stats.prefix_ram_hits,
                same_tokens(&first_tokens, &equal_tokens));
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
    if (run_generation(extended, prompt_plus_one,
                       &extended_capture, &extended_stats,
                       error, sizeof(error))) {
        fprintf(stderr, "first extension generation failed: %s\n", error);
        goto cleanup;
    }
    if (extended->prefix_reused != base_tokens ||
        extended_stats.prefix_reused_tokens != base_tokens ||
        extended_stats.prefix_ram_hits != 1 ||
        !extended_stats.prefix_ram_restore_bytes ||
        !(extended_stats.prefix_ram_restore_sec > 0.0)) {
        fprintf(stderr,
                "cross-session restore stats invalid: session=%d stats=%d hits=%llu bytes=%llu sec=%.9f expected=%d\n",
                extended->prefix_reused, extended_stats.prefix_reused_tokens,
                (unsigned long long)extended_stats.prefix_ram_hits,
                (unsigned long long)extended_stats.prefix_ram_restore_bytes,
                extended_stats.prefix_ram_restore_sec, base_tokens);
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
    if (run_generation(longest, prompt_plus_two, &warm_tokens, &warm_stats,
                       error, sizeof(error))) {
        fprintf(stderr, "longest-prefix generation failed: %s\n", error);
        goto cleanup;
    }
    if (longest->prefix_reused != extended_tokens ||
        warm_stats.prefix_reused_tokens != extended_tokens ||
        warm_stats.prefix_ram_hits != 1) {
        fprintf(stderr,
                "longest-prefix selection failed: session=%d stats=%d hits=%llu expected=%d base=%d\n",
                longest->prefix_reused, warm_stats.prefix_reused_tokens,
                (unsigned long long)warm_stats.prefix_ram_hits,
                extended_tokens, base_tokens);
        goto cleanup;
    }
    if (generated_text(longest, &longest_text)) {
        fprintf(stderr, "cannot decode longest-prefix output\n");
        goto cleanup;
    }

    /* Mismatch and rewind are fail-closed misses. Use the exact token arrays so
     * tokenizer round-tripping cannot weaken either matching assertion. */
    divergent_ids = malloc((size_t)longest->prompt_count * sizeof(*divergent_ids));
    if (!divergent_ids) {
        fprintf(stderr, "out of memory building divergent token probe\n");
        goto cleanup;
    }
    memcpy(divergent_ids, longest->prompt_ids,
           (size_t)longest->prompt_count * sizeof(*divergent_ids));
    int pivot = base_tokens / 2;
    int replacement = divergent_ids[pivot] + 1;
    if (replacement >= longest->config.vocab_size) replacement = 0;
    if (replacement == divergent_ids[pivot] && longest->config.vocab_size > 1)
        replacement = 1;
    divergent_ids[pivot] = replacement;
    if (coli_v4_prefix_cache_restore(miss, divergent_ids,
                                     longest->prompt_count) != 0) {
        fprintf(stderr, "one-token divergence unexpectedly restored a prefix\n");
        goto cleanup;
    }
    if (coli_v4_prefix_cache_restore(miss, first->prompt_ids,
                                     base_tokens - 1) != 0) {
        fprintf(stderr, "shorter/rewind request unexpectedly restored a prefix\n");
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

    /* Same exact request prompt on the warm and cold sessions. The warm run has
     * restored P+X and prefilled only Y; its very first generated token is the
     * next-token argmax acceptance gate, then every generated token must match. */
    if (run_generation(cold, prompt_plus_two, &cold_tokens, &cold_stats,
                       error, sizeof(error))) {
        fprintf(stderr, "cold generation failed: %s\n", error);
        goto cleanup;
    }
    if (cold->prefix_reused != 0 || cold_stats.prefix_reused_tokens != 0 ||
        cold_stats.prefix_ram_hits != 0) {
        fprintf(stderr,
                "different engine unexpectedly reused: session=%d stats=%d hits=%llu\n",
                cold->prefix_reused, cold_stats.prefix_reused_tokens,
                (unsigned long long)cold_stats.prefix_ram_hits);
        goto cleanup;
    }
    if (!same_tokens(&warm_tokens, &cold_tokens) ||
        warm_tokens.ids[0] != cold_tokens.ids[0]) {
        fprintf(stderr,
                "warm/cold token identity failed: warm_first=%d cold_first=%d warm_count=%d cold_count=%d\n",
                warm_tokens.count ? warm_tokens.ids[0] : -1,
                cold_tokens.count ? cold_tokens.ids[0] : -1,
                warm_tokens.count, cold_tokens.count);
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
        !cache.restore_bytes || !cache.restore_ns || cache.resident_bytes > reserve) {
        fprintf(stderr,
                "prefix-cache telemetry incomplete: hits=%llu matched=%llu expected=%llu bytes=%llu ns=%llu resident=%zu reserve=%llu\n",
                (unsigned long long)cache.hits,
                (unsigned long long)cache.matched_tokens,
                (unsigned long long)expected_matched,
                (unsigned long long)cache.restore_bytes,
                (unsigned long long)cache.restore_ns,
                cache.resident_bytes, (unsigned long long)reserve);
        goto cleanup;
    }

    printf("PASS Stage1 V4 prefix cache: base=%d extended=%d warm_reuse=%d first_argmax=%d token_identity=exact divergence=miss rewind=miss restore=%.3fMiB/%.3fms memory=%.3fGiB+%.3fGiB<=%.3fGiB\n",
           base_tokens, extended_tokens, longest->prefix_reused,
           warm_tokens.ids[0],
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
    free(divergent_ids);
    if (first) coli_v4_session_destroy(first);
    if (equal) coli_v4_session_destroy(equal);
    if (extended) coli_v4_session_destroy(extended);
    if (codec) coli_v4_session_destroy(codec);
    if (longest) coli_v4_session_destroy(longest);
    if (cold) coli_v4_session_destroy(cold);
    if (miss) coli_v4_session_destroy(miss);
    if (shared_engine) coli_v4_engine_destroy(shared_engine);
    if (cold_engine) coli_v4_engine_destroy(cold_engine);
    if (rejected_engine) coli_v4_engine_destroy(rejected_engine);
    return status;
}

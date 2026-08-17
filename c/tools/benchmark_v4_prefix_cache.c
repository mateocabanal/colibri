#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Controlled process-local prefix-cache performance probe.
 *
 * One process performs the same two-request workload in either cache-off or
 * cache-on mode:
 *   1. fresh session A prefills a supplied prefix P, warming the engine;
 *   2. fresh session B runs a tokenizer-verified strict extension P+X.
 *
 * The Python A/B driver runs this probe in separate cache-off/cache-on
 * processes under the same total memory limit and compares only request 2.
 * That avoids crediting ordinary same-engine warmup to the prefix cache, while
 * still charging cache-on for the RAM it reserves away from dense/expert
 * residency. The first request is deliberately reported for diagnostics only.
 */
#include "../deepseek_v4_internal.h"
#include "../coli_v4_prefix_cache.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_text_file(const char *path, size_t *length) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return NULL; }
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) { fclose(file); return NULL; }
    if ((unsigned long)size > SIZE_MAX - 1) { fclose(file); return NULL; }
    char *text = malloc((size_t)size + 1);
    if (!text) { fclose(file); return NULL; }
    size_t got = fread(text, 1, (size_t)size, file);
    if (got != (size_t)size && ferror(file)) {
        free(text); fclose(file); return NULL;
    }
    fclose(file);
    text[got] = '\0';
    if (length) *length = got;
    return text;
}

static char *append_text(const char *prefix, size_t prefix_length,
                         const char *suffix, size_t suffix_length) {
    if (!prefix || !suffix || prefix_length > SIZE_MAX - suffix_length - 1)
        return NULL;
    char *text = malloc(prefix_length + suffix_length + 1);
    if (!text) return NULL;
    memcpy(text, prefix, prefix_length);
    memcpy(text + prefix_length, suffix, suffix_length);
    text[prefix_length + suffix_length] = '\0';
    return text;
}

static int candidate_is_strict_extension(Tok *tokenizer,
                                         const int *base_ids, int base_count,
                                         const char *candidate,
                                         int capacity, int *candidate_count) {
    int *ids = malloc((size_t)capacity * sizeof(*ids));
    if (!ids) return 0;
    size_t bytes = strlen(candidate);
    if (bytes > INT_MAX) { free(ids); return 0; }
    int count = tok_encode(tokenizer, candidate, (int)bytes, ids, capacity);
    int valid = count > base_count && count < capacity &&
        memcmp(ids, base_ids, (size_t)base_count * sizeof(*ids)) == 0;
    if (valid && candidate_count) *candidate_count = count;
    free(ids);
    return valid;
}

static char *verified_strict_extension(ColiV4Session *session,
                                       const char *prefix, size_t prefix_length,
                                       int *prefix_tokens,
                                       int *extended_tokens) {
    if (!session || !prefix || prefix_length > INT_MAX || session->fed.cap < 2)
        return NULL;
    int capacity = session->fed.cap;
    int *base = malloc((size_t)capacity * sizeof(*base));
    if (!base) return NULL;
    int base_count = tok_encode(&session->tokenizer, prefix, (int)prefix_length,
                                base, capacity);
    if (base_count <= 0 || base_count >= capacity) {
        free(base);
        return NULL;
    }

    static const char *suffixes[] = {"\n", "\n\n", " x", ".", "!", " 0"};
    for (size_t index = 0; index < sizeof(suffixes) / sizeof(suffixes[0]); index++) {
        const char *suffix = suffixes[index];
        char *candidate = append_text(prefix, prefix_length,
                                      suffix, strlen(suffix));
        int count = 0;
        if (candidate && candidate_is_strict_extension(
                &session->tokenizer, base, base_count, candidate,
                capacity, &count)) {
            if (prefix_tokens) *prefix_tokens = base_count;
            if (extended_tokens) *extended_tokens = count;
            free(base);
            return candidate;
        }
        free(candidate);
    }

    /* Added tokens force a tokenizer chunk boundary and are a robust fallback
     * when ordinary punctuation merges across the exact supplied prefix. */
    for (int pass = 0; pass < 2; pass++) {
        for (int index = 0; index < session->tokenizer.nsp; index++) {
            Special *special = &session->tokenizer.sp[index];
            if (!special->str || special->len <= 0 || special->id < 0 ||
                special->id >= session->tokenizer.n_ids)
                continue;
            int is_control = session->tokenizer.id_special
                ? session->tokenizer.id_special[special->id] != 0 : 0;
            if ((pass == 0 && is_control) || (pass == 1 && !is_control))
                continue;
            char *candidate = append_text(prefix, prefix_length,
                                          special->str, (size_t)special->len);
            int count = 0;
            if (candidate && candidate_is_strict_extension(
                    &session->tokenizer, base, base_count, candidate,
                    capacity, &count)) {
                if (prefix_tokens) *prefix_tokens = base_count;
                if (extended_tokens) *extended_tokens = count;
                free(base);
                return candidate;
            }
            free(candidate);
        }
    }
    free(base);
    return NULL;
}

static int parse_positive_int(const char *text, int *output) {
    if (!text || !*text || !output) return -1;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end || value <= 0 || value > INT_MAX) return -1;
    *output = (int)value;
    return 0;
}

static int parse_memory_gb(const char *text, uint64_t *output) {
    if (!text || !*text || !output) return -1;
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text || *end || value <= 0.0) return -1;
    long double bytes = (long double)value * 1073741824.0L;
    if (bytes >= (long double)UINT64_MAX) return -1;
    *output = (uint64_t)bytes;
    return 0;
}

static int run_request(ColiV4Session *session, const char *prompt,
                       size_t prompt_length, int max_new_tokens,
                       ColiV4SessionGenerateStats *stats,
                       char *error, size_t error_size) {
    ColiV4SessionGenerateOptions options = {
        .max_new_tokens = max_new_tokens,
        .no_dspark = 1,
    };
    memset(stats, 0, sizeof(*stats));
    return coli_v4_session_generate(session, prompt, prompt_length, &options,
                                    NULL, NULL, stats, error, error_size);
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s MODEL_DIR PREFIX_FILE [--memory-gb N] [--context N] "
            "[--max-new N] [--coli-model DIR]\n", program);
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 2; }
    const char *model = argv[1];
    const char *prefix_path = argv[2];
    const char *coli_model = getenv("COLI_MODEL");
    int context = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
    if (context <= 0) context = 4096;
    int max_new = 1;
    uint64_t memory_limit = 0;
    if (getenv("RAM_GB") && parse_memory_gb(getenv("RAM_GB"), &memory_limit)) {
        fprintf(stderr, "invalid RAM_GB=%s\n", getenv("RAM_GB"));
        return 2;
    }

    for (int index = 3; index < argc; index++) {
        if (!strcmp(argv[index], "--memory-gb") && index + 1 < argc) {
            if (parse_memory_gb(argv[++index], &memory_limit)) {
                fprintf(stderr, "invalid --memory-gb\n"); return 2;
            }
        } else if (!strcmp(argv[index], "--context") && index + 1 < argc) {
            if (parse_positive_int(argv[++index], &context)) {
                fprintf(stderr, "invalid --context\n"); return 2;
            }
        } else if (!strcmp(argv[index], "--max-new") && index + 1 < argc) {
            if (parse_positive_int(argv[++index], &max_new)) {
                fprintf(stderr, "invalid --max-new\n"); return 2;
            }
        } else if (!strcmp(argv[index], "--coli-model") && index + 1 < argc) {
            coli_model = argv[++index];
        } else {
            usage(argv[0]); return 2;
        }
    }

    size_t prefix_length = 0;
    char *prefix = read_text_file(prefix_path, &prefix_length);
    if (!prefix || !prefix_length) {
        fprintf(stderr, "cannot read non-empty prefix file %s: %s\n",
                prefix_path, strerror(errno));
        free(prefix);
        return 2;
    }

    char error[1024] = {0};
    ColiV4Engine *engine = NULL;
    ColiV4Session *source = NULL, *second = NULL;
    char *extended = NULL;
    int status = 1;

    ColiV4EngineOpenOptions open_options = {
        .target_model_dir = model,
        .coli_model_dir = coli_model,
        .memory_limit_bytes = memory_limit,
        .context_tokens = context,
        .pin_slots_per_layer = -1,
        .no_dspark = 1,
    };
    if (coli_v4_engine_open(&engine, &open_options, error, sizeof(error))) {
        fprintf(stderr, "engine open failed: %s\n", error);
        goto cleanup;
    }

    ColiV4SessionCreateOptions session_options = {
        .max_prompt_tokens = context,
        .max_new_tokens_cap = max_new,
    };
    if (coli_v4_session_create(&source, engine, &session_options,
                               error, sizeof(error)) ||
        coli_v4_session_create(&second, engine, &session_options,
                               error, sizeof(error))) {
        fprintf(stderr, "session create failed: %s\n", error);
        goto cleanup;
    }

    int prefix_tokens = 0, extended_tokens = 0;
    extended = verified_strict_extension(source, prefix, prefix_length,
                                         &prefix_tokens, &extended_tokens);
    if (!extended) {
        fprintf(stderr,
                "could not construct a tokenizer-verified strict extension; "
                "increase --context if the prefix fills the session\n");
        goto cleanup;
    }

    ColiV4PrefixCacheStats cache0 = {0}, cache1 = {0}, cache2 = {0};
    coli_v4_prefix_cache_stats(&cache0);
    int cache_enabled = cache0.budget_bytes != 0;

    ColiV4EngineMemorySummary memory = {0};
    coli_v4_engine_memory_summary(engine, &memory);
    if (memory_limit && (!memory.projected_bytes ||
        cache0.budget_bytes > memory_limit ||
        memory.projected_bytes > memory_limit - cache0.budget_bytes)) {
        fprintf(stderr,
                "probe memory envelope invalid: projected=%llu reserve=%zu limit=%llu\n",
                (unsigned long long)memory.projected_bytes,
                cache0.budget_bytes, (unsigned long long)memory_limit);
        goto cleanup;
    }

    ColiV4SessionGenerateStats first_stats = {0}, second_stats = {0};
    if (run_request(source, prefix, prefix_length, max_new,
                    &first_stats, error, sizeof(error))) {
        fprintf(stderr, "first prefix request failed: %s\n", error);
        goto cleanup;
    }
    if (source->prompt_count != prefix_tokens) {
        fprintf(stderr,
                "prefix tokenizer count changed: preflight=%d runtime=%d\n",
                prefix_tokens, source->prompt_count);
        goto cleanup;
    }
    coli_v4_prefix_cache_stats(&cache1);
    if (cache_enabled && cache1.stores <= cache0.stores) {
        fprintf(stderr,
                "cache-on first request was not admitted (tokens=%d); lower "
                "V4_PREFIX_CACHE_MIN_TOKENS or increase prefix length\n",
                prefix_tokens);
        goto cleanup;
    }
    if (!cache_enabled && cache1.stores != cache0.stores) {
        fprintf(stderr, "cache-off probe unexpectedly stored a prefix\n");
        goto cleanup;
    }

    size_t extended_length = strlen(extended);
    if (run_request(second, extended, extended_length, max_new,
                    &second_stats, error, sizeof(error))) {
        fprintf(stderr, "second extension request failed: %s\n", error);
        goto cleanup;
    }
    if (second->prompt_count != extended_tokens) {
        fprintf(stderr,
                "second tokenizer count changed: preflight=%d runtime=%d\n",
                extended_tokens, second->prompt_count);
        goto cleanup;
    }
    coli_v4_prefix_cache_stats(&cache2);
    if (cache_enabled) {
        if (second->prefix_reused != prefix_tokens || cache2.hits <= cache1.hits) {
            fprintf(stderr,
                    "cache-on second request missed: reused=%d/%d hits=%llu/%llu\n",
                    second->prefix_reused, prefix_tokens,
                    (unsigned long long)cache2.hits,
                    (unsigned long long)cache1.hits);
            goto cleanup;
        }
    } else if (second->prefix_reused != 0 || cache2.hits != cache1.hits) {
        fprintf(stderr,
                "cache-off second request unexpectedly reused state: reused=%d hits=%llu/%llu\n",
                second->prefix_reused,
                (unsigned long long)cache2.hits,
                (unsigned long long)cache1.hits);
        goto cleanup;
    }

    uint64_t hit_delta = cache2.hits >= cache1.hits
        ? cache2.hits - cache1.hits : 0;
    uint64_t store_delta = cache1.stores >= cache0.stores
        ? cache1.stores - cache0.stores : 0;
    uint64_t restore_bytes = cache2.restore_bytes >= cache1.restore_bytes
        ? cache2.restore_bytes - cache1.restore_bytes : 0;
    uint64_t restore_ns = cache2.restore_ns >= cache1.restore_ns
        ? cache2.restore_ns - cache1.restore_ns : 0;

    printf("{\"schema\":\"colibri.v4.prefix_cache_probe.v1\","
           "\"mode\":\"%s\",\"prefix_tokens\":%d,"
           "\"second_prompt_tokens\":%d,\"second_prefix_reused\":%d,"
           "\"first_ttft_sec\":%.9f,\"second_ttft_sec\":%.9f,"
           "\"cache_hits_delta\":%llu,\"cache_stores_delta\":%llu,"
           "\"restore_bytes\":%llu,\"restore_ms\":%.6f,"
           "\"cache_resident_bytes\":%zu,\"cache_budget_bytes\":%zu,"
           "\"memory_projected_bytes\":%llu,\"memory_limit_bytes\":%llu}\n",
           cache_enabled ? "cache_on" : "cache_off",
           prefix_tokens, second->prompt_count, second->prefix_reused,
           first_stats.time_to_first_token_sec, second_stats.time_to_first_token_sec,
           (unsigned long long)hit_delta, (unsigned long long)store_delta,
           (unsigned long long)restore_bytes, restore_ns / 1.0e6,
           cache2.resident_bytes, cache2.budget_bytes,
           (unsigned long long)memory.projected_bytes,
           (unsigned long long)memory_limit);
    status = 0;

cleanup:
    free(extended);
    free(prefix);
    if (source) coli_v4_session_destroy(source);
    if (second) coli_v4_session_destroy(second);
    if (engine) coli_v4_engine_destroy(engine);
    return status;
}

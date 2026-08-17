#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Real process-local prefix-cache performance probe.
 *
 * One engine, two independent sessions:
 *   1. cold-prefill a supplied long prefix P, which admits P;
 *   2. build a tokenizer-verified strict extension P+X and run it in a fresh
 *      session, which must restore P and prefill only X.
 *
 * Comparing P cold against the strictly-longer P+X warm request is conservative:
 * absent caching the second request has at least as much prompt work. Keeping one
 * engine also reflects the intended long-lived agent/server workload instead of
 * conflating prefix reuse with a second engine's residency/page-cache state.
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

    /* Added tokens split the preceding text into its own tokenizer chunk, so
     * they are a robust fallback when ordinary punctuation merges across the
     * user's exact prefix boundary. Prefer a non-control added token when the
     * tokenizer exposes one, but correctness only depends on strict token-ID
     * verification below. */
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

    const char *cache_mb = getenv("V4_PREFIX_CACHE_MB");
    if (!cache_mb || !*cache_mb || strtod(cache_mb, NULL) <= 0.0) {
        fprintf(stderr,
                "V4_PREFIX_CACHE_MB must be set to a positive cache budget\n");
        return 2;
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
    ColiV4Session *source = NULL, *warm = NULL;
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
        coli_v4_session_create(&warm, engine, &session_options,
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

    ColiV4SessionGenerateStats cold_stats = {0}, warm_stats = {0};
    ColiV4PrefixCacheStats cache0 = {0}, cache1 = {0}, cache2 = {0};
    coli_v4_prefix_cache_stats(&cache0);
    if (run_request(source, prefix, prefix_length, max_new,
                    &cold_stats, error, sizeof(error))) {
        fprintf(stderr, "cold prefix request failed: %s\n", error);
        goto cleanup;
    }
    if (source->prompt_count != prefix_tokens) {
        fprintf(stderr,
                "prefix tokenizer count changed: preflight=%d runtime=%d\n",
                prefix_tokens, source->prompt_count);
        goto cleanup;
    }
    coli_v4_prefix_cache_stats(&cache1);
    if (cache1.stores <= cache0.stores) {
        fprintf(stderr,
                "cold prefix was not admitted (tokens=%d); lower "
                "V4_PREFIX_CACHE_MIN_TOKENS or increase prefix length\n",
                prefix_tokens);
        goto cleanup;
    }

    size_t extended_length = strlen(extended);
    if (run_request(warm, extended, extended_length, max_new,
                    &warm_stats, error, sizeof(error))) {
        fprintf(stderr, "warm extension request failed: %s\n", error);
        goto cleanup;
    }
    if (warm->prompt_count != extended_tokens || warm->prefix_reused != prefix_tokens) {
        fprintf(stderr,
                "warm request did not restore full prefix: prompt=%d/%d reused=%d/%d\n",
                warm->prompt_count, extended_tokens, warm->prefix_reused, prefix_tokens);
        goto cleanup;
    }
    coli_v4_prefix_cache_stats(&cache2);
    if (cache2.hits <= cache1.hits) {
        fprintf(stderr, "warm request produced no prefix-cache hit\n");
        goto cleanup;
    }

    ColiV4EngineMemorySummary memory = {0};
    coli_v4_engine_memory_summary(engine, &memory);
    uint64_t restore_bytes = cache2.restore_bytes >= cache1.restore_bytes
        ? cache2.restore_bytes - cache1.restore_bytes : 0;
    uint64_t restore_ns = cache2.restore_ns >= cache1.restore_ns
        ? cache2.restore_ns - cache1.restore_ns : 0;
    double cold_ttft = cold_stats.time_to_first_token_sec;
    double warm_ttft = warm_stats.time_to_first_token_sec;
    double saved = cold_ttft - warm_ttft;
    double speedup = warm_ttft > 0.0 ? cold_ttft / warm_ttft : 0.0;

    printf("{\"schema\":\"colibri.v4.prefix_cache_bench.v1\","
           "\"prefix_tokens\":%d,\"warm_prompt_tokens\":%d,"
           "\"matched_tokens\":%d,\"cold_prefix_ttft_sec\":%.9f,"
           "\"warm_extension_ttft_sec\":%.9f,\"ttft_saved_sec\":%.9f,"
           "\"ttft_speedup\":%.6f,\"restore_bytes\":%llu,"
           "\"restore_ms\":%.6f,\"cache_resident_bytes\":%zu,"
           "\"cache_budget_bytes\":%zu,\"memory_projected_bytes\":%llu,"
           "\"memory_limit_bytes\":%llu}\n",
           prefix_tokens, warm->prompt_count, warm->prefix_reused,
           cold_ttft, warm_ttft, saved, speedup,
           (unsigned long long)restore_bytes, restore_ns / 1.0e6,
           cache2.resident_bytes, cache2.budget_bytes,
           (unsigned long long)memory.projected_bytes,
           (unsigned long long)memory_limit);
    status = 0;

cleanup:
    free(extended);
    free(prefix);
    if (source) coli_v4_session_destroy(source);
    if (warm) coli_v4_session_destroy(warm);
    if (engine) coli_v4_engine_destroy(engine);
    return status;
}

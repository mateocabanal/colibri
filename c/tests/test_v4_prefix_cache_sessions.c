#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../deepseek_v4_internal.h"
#include "../coli_v4_prefix_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int append_bytes(char **buffer, size_t *length, size_t *capacity,
                        const char *bytes, size_t count) {
    if (*length > SIZE_MAX - count - 1) return -1;
    size_t needed = *length + count + 1;
    if (needed > *capacity) {
        size_t next = *capacity ? *capacity : 256;
        while (next < needed) {
            if (next > SIZE_MAX / 2) { next = needed; break; }
            next *= 2;
        }
        char *grown = realloc(*buffer, next);
        if (!grown) return -1;
        *buffer = grown;
        *capacity = next;
    }
    memcpy(*buffer + *length, bytes, count);
    *length += count;
    (*buffer)[*length] = '\0';
    return 0;
}

static int decode_token_string(const ColiV4Session *session,
                               const int *ids, int count,
                               char **output) {
    char *text = NULL;
    size_t length = 0, capacity = 0;
    for (int index = 0; index < count; index++) {
        char piece[512];
        int bytes = tok_decode((tok *)&session->tokenizer, &ids[index], 1,
                               piece, (int)sizeof(piece));
        if (bytes < 0 || append_bytes(&text, &length, &capacity,
                                     piece, (size_t)bytes)) {
            free(text);
            return -1;
        }
    }
    if (!text) {
        text = calloc(1, 1);
        if (!text) return -1;
    }
    *output = text;
    return 0;
}

static int generated_text(ColiV4Session *session, char **output) {
    size_t length = 0;
    if (coli_v4_session_generated_text(session, NULL, 0, &length) != 0)
        return -1;
    char *text = malloc(length + 1);
    if (!text) return -1;
    if (coli_v4_session_generated_text(session, text, length + 1, &length) != 0) {
        free(text);
        return -1;
    }
    *output = text;
    return 0;
}

static int open_engine(ColiV4Engine **engine, const char *model,
                       char *error, size_t error_size) {
    ColiV4EngineOpenOptions options = {
        .target_model_dir = model,
        .context_tokens = 128,
        .pin_slots_per_layer = -1,
        .no_dspark = 1,
    };
    return coli_v4_engine_open(engine, &options, error, error_size);
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
    ColiV4Engine *shared_engine = NULL, *cold_engine = NULL;
    ColiV4Session *first = NULL, *restored = NULL, *cold = NULL;
    char *extension = NULL, *restored_text = NULL, *cold_text = NULL;
    int status = 1;

    if (!getenv("V4_PREFIX_CACHE_MB") || !atoi(getenv("V4_PREFIX_CACHE_MB"))) {
        fprintf(stderr, "V4_PREFIX_CACHE_MB must be positive for this test\n");
        goto cleanup;
    }

    if (open_engine(&shared_engine, model, error, sizeof(error)) ||
        open_session(&first, shared_engine, error, sizeof(error)) ||
        open_session(&restored, shared_engine, error, sizeof(error))) {
        fprintf(stderr, "shared engine/session open failed: %s\n", error);
        goto cleanup;
    }

    if (run_generation(first, initial_prompt, error, sizeof(error))) {
        fprintf(stderr, "first generation failed: %s\n", error);
        goto cleanup;
    }
    if (!first->fed.fed || first->fed.len <= 0 || first->fed.tainted) {
        fprintf(stderr, "first session did not leave reusable exact state\n");
        goto cleanup;
    }
    int cached_tokens = first->fed.len;

    /* Decode the exact token sequence represented by the captured state, then
     * append one known token. The synthetic tiny tokenizer is round-trippable;
     * this creates a strict prefix extension without guessing which final
     * generated token has or has not already entered target state. */
    if (decode_token_string(first, first->fed.fed, first->fed.len, &extension)) {
        fprintf(stderr, "failed to decode cached prefix\n");
        goto cleanup;
    }
    char *extra = NULL;
    if (decode_token_string(first, first->fed.fed, 1, &extra)) {
        fprintf(stderr, "failed to decode extension token\n");
        goto cleanup;
    }
    size_t ext_len = strlen(extension), extra_len = strlen(extra);
    char *grown = realloc(extension, ext_len + extra_len + 1);
    if (!grown) {
        free(extra);
        fprintf(stderr, "out of memory extending prompt\n");
        goto cleanup;
    }
    extension = grown;
    memcpy(extension + ext_len, extra, extra_len + 1);
    free(extra);

    if (run_generation(restored, extension, error, sizeof(error))) {
        fprintf(stderr, "restored generation failed: %s\n", error);
        goto cleanup;
    }
    if (restored->prefix_reused != cached_tokens) {
        fprintf(stderr,
                "cross-session cache did not restore exact prefix: got=%d expected=%d\n",
                restored->prefix_reused, cached_tokens);
        goto cleanup;
    }
    if (generated_text(restored, &restored_text)) {
        fprintf(stderr, "cannot decode restored output\n");
        goto cleanup;
    }

    /* A different engine pointer is a strict cache namespace, so its first
     * request is the cold-prefill oracle even though the process cache remains
     * enabled globally. */
    if (open_engine(&cold_engine, model, error, sizeof(error)) ||
        open_session(&cold, cold_engine, error, sizeof(error))) {
        fprintf(stderr, "cold engine/session open failed: %s\n", error);
        goto cleanup;
    }
    if (run_generation(cold, extension, error, sizeof(error))) {
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
    if (strcmp(restored_text, cold_text)) {
        fprintf(stderr,
                "restored output diverged from cold prefill:\nrestored=%s\ncold=%s\n",
                restored_text, cold_text);
        goto cleanup;
    }

    ColiV4PrefixCacheStats cache = {0};
    coli_v4_prefix_cache_stats(&cache);
    if (!cache.hits || cache.matched_tokens < (uint64_t)cached_tokens ||
        !cache.restore_bytes || !cache.restore_ns) {
        fprintf(stderr,
                "prefix-cache telemetry incomplete: hits=%llu matched=%llu bytes=%llu ns=%llu\n",
                (unsigned long long)cache.hits,
                (unsigned long long)cache.matched_tokens,
                (unsigned long long)cache.restore_bytes,
                (unsigned long long)cache.restore_ns);
        goto cleanup;
    }

    printf("PASS cross-session prefix cache: %d tokens restored, %.3f MiB copied in %.3f ms, output identical to cold prefill\n",
           cached_tokens, cache.restore_bytes / (1024.0 * 1024.0),
           cache.restore_ns / 1.0e6);
    status = 0;

cleanup:
    free(extension);
    free(restored_text);
    free(cold_text);
    if (first) coli_v4_session_destroy(first);
    if (restored) coli_v4_session_destroy(restored);
    if (cold) coli_v4_session_destroy(cold);
    if (shared_engine) coli_v4_engine_destroy(shared_engine);
    if (cold_engine) coli_v4_engine_destroy(cold_engine);
    return status;
}

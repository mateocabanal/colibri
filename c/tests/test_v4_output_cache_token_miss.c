#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../compat.h"
#include "../deepseek_v4_internal.h"
#include "../coli_v4_output_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int tokens[8];
    int count;
} Capture;

static int capture_token(void *user_data, int token, float logit,
                         int position, int ordinal) {
    (void)logit;
    (void)position;
    (void)ordinal;
    Capture *capture = (Capture *)user_data;
    if (!capture || capture->count >= 8) return 1;
    capture->tokens[capture->count++] = token;
    return 0;
}

static int run(ColiV4Session *session, const char *prompt, Capture *capture,
               char *error, size_t error_size) {
    ColiV4SessionGenerateOptions options = {
        .max_new_tokens = 4,
        .no_dspark = 1,
    };
    memset(capture, 0, sizeof(*capture));
    return coli_v4_session_generate(session, prompt, strlen(prompt), &options,
                                    capture_token, capture, NULL,
                                    error, error_size);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    static const char prompt_a[] =
        "<t005><t007><t009><t011><t013><t017><t019><t023>";
    static const char prompt_b[] =
        "<t005><t007><t009><t012><t013><t017><t019><t023>";

    char directory[1024];
    snprintf(directory, sizeof(directory),
             "/tmp/coli-v4-output-token-miss-%lu-%llu",
             (unsigned long)getpid(), (unsigned long long)time(NULL));
#ifdef _WIN32
    _putenv_s("COLI_OUTPUT_CACHE", "on");
    _putenv_s("COLI_OUTPUT_CACHE_DIR", directory);
    _putenv_s("COLI_OUTPUT_CACHE_GB", "0.125");
    _putenv_s("COLI_PREFIX_CACHE", "off");
    _putenv_s("COLI_PREFIX_CACHE_MIN_FREE_GB", "0");
    _putenv_s("V4_PREFIX_CACHE_MB", "off");
#else
    setenv("COLI_OUTPUT_CACHE", "on", 1);
    setenv("COLI_OUTPUT_CACHE_DIR", directory, 1);
    setenv("COLI_OUTPUT_CACHE_GB", "0.125", 1);
    setenv("COLI_PREFIX_CACHE", "off", 1);
    setenv("COLI_PREFIX_CACHE_MIN_FREE_GB", "0", 1);
    setenv("V4_PREFIX_CACHE_MB", "off", 1);
#endif

    char error[1024] = {0};
    ColiV4Engine *engine = NULL;
    ColiV4Session *session = NULL;
    ColiV4EngineOpenOptions engine_options = {
        .target_model_dir = argv[1],
        .context_tokens = 128,
        .pin_slots_per_layer = -1,
        .no_dspark = 1,
    };
    ColiV4SessionCreateOptions session_options = {
        .max_prompt_tokens = 128,
        .max_new_tokens_cap = 8,
    };
    if (coli_v4_engine_open(&engine, &engine_options, error, sizeof(error))) {
        fprintf(stderr, "engine open failed: %s\n", error);
        return 1;
    }

    uint8_t model_fingerprint[32], tokenizer_fingerprint[32];
    for (int i = 0; i < 32; i++) {
        model_fingerprint[i] = (uint8_t)(0x41 + i * 3);
        tokenizer_fingerprint[i] = (uint8_t)(0x67 + i * 5);
    }
    coli_v4_output_cache_test_identity(engine, model_fingerprint,
                                       tokenizer_fingerprint);
    if (coli_v4_session_create(&session, engine, &session_options,
                               error, sizeof(error))) {
        fprintf(stderr, "session open failed: %s\n", error);
        coli_v4_engine_destroy(engine);
        return 1;
    }

    Capture first = {0}, changed = {0};
    ColiV4OutputCacheStats before = {0}, after = {0};
    int first_ids[128];
    int first_count = 0;
    int status = 1;

    if (run(session, prompt_a, &first, error, sizeof(error))) {
        fprintf(stderr, "baseline generation failed: %s\n", error);
        goto cleanup;
    }
    first_count = session->prompt_count;
    if (first_count < 1 || first_count > 128) goto cleanup;
    memcpy(first_ids, session->prompt_ids,
           (size_t)first_count * sizeof(*first_ids));
    coli_v4_output_cache_stats(&before);
    if (before.hits != 0 || before.stores != 1 || first.count != 4) {
        fprintf(stderr, "baseline was not a cold stored request\n");
        goto cleanup;
    }

    if (run(session, prompt_b, &changed, error, sizeof(error))) {
        fprintf(stderr, "changed-token generation failed: %s\n", error);
        goto cleanup;
    }
    coli_v4_output_cache_stats(&after);

    int differences = 0;
    if (session->prompt_count != first_count) {
        fprintf(stderr, "changed prompt token count changed unexpectedly\n");
        goto cleanup;
    }
    for (int i = 0; i < first_count; i++)
        differences += first_ids[i] != session->prompt_ids[i];
    if (differences != 1 || after.hits != before.hits ||
        after.stores != before.stores + 1 || changed.count != 4) {
        fprintf(stderr,
                "one-token request did not miss: differences=%d hits=%llu->%llu stores=%llu->%llu generated=%d\n",
                differences,
                (unsigned long long)before.hits,
                (unsigned long long)after.hits,
                (unsigned long long)before.stores,
                (unsigned long long)after.stores,
                changed.count);
        goto cleanup;
    }

    puts("PASS Stage3 V4 output cache one-token divergence: token_delta=1 cache=miss");
    status = 0;

cleanup:
    coli_v4_session_destroy(session);
    coli_v4_engine_destroy(engine);
    return status;
}

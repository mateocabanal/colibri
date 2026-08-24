#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../compat.h"
#include "../deepseek_v4_internal.h"
#include "../coli_v4_output_cache.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int ids[16];
    float logits[16];
    int positions[16];
    int ordinals[16];
    int count;
    int stop_after_first;
} Capture;

static int capture_token(void *user_data, int token, float logit,
                         int position, int ordinal) {
    Capture *capture = (Capture *)user_data;
    if (!capture || capture->count >= 16) return 1;
    int at = capture->count++;
    capture->ids[at] = token;
    capture->logits[at] = logit;
    capture->positions[at] = position;
    capture->ordinals[at] = ordinal;
    return capture->stop_after_first && capture->count == 1;
}

/* Cached replay must reproduce the complete callback stream byte-for-byte,
 * including logits and position metadata. Independent cold regeneration has a
 * weaker semantic contract: greedy token IDs must be identical, while floating
 * logits may differ in their low bits across fresh engine instances. */
static int same_capture(const Capture *a, const Capture *b) {
    return a && b && a->count == b->count && a->count > 0 &&
        !memcmp(a->ids, b->ids, (size_t)a->count * sizeof(a->ids[0])) &&
        !memcmp(a->logits, b->logits,
                (size_t)a->count * sizeof(a->logits[0])) &&
        !memcmp(a->positions, b->positions,
                (size_t)a->count * sizeof(a->positions[0])) &&
        !memcmp(a->ordinals, b->ordinals,
                (size_t)a->count * sizeof(a->ordinals[0]));
}

static int same_tokens(const Capture *a, const Capture *b) {
    return a && b && a->count == b->count && a->count > 0 &&
        !memcmp(a->ids, b->ids, (size_t)a->count * sizeof(a->ids[0]));
}

static int prefix_capture(const Capture *full, const Capture *shorter) {
    return full && shorter && shorter->count > 0 &&
        shorter->count <= full->count &&
        !memcmp(full->ids, shorter->ids,
                (size_t)shorter->count * sizeof(full->ids[0]));
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
    return coli_v4_session_create(session, engine, &options,
                                  error, error_size);
}

static int run_generation(ColiV4Session *session, const char *prompt,
                          int max_new, Capture *capture,
                          char *error, size_t error_size) {
    ColiV4SessionGenerateOptions options = {
        .max_new_tokens = max_new,
        .no_dspark = 1,
    };
    if (capture) {
        int stop = capture->stop_after_first;
        memset(capture, 0, sizeof(*capture));
        capture->stop_after_first = stop;
    }
    ColiV4SessionGenerateStats stats = {0};
    return coli_v4_session_generate(session, prompt, strlen(prompt), &options,
                                    capture_token, capture, &stats,
                                    error, error_size);
}

static void install_identity(ColiV4Engine *engine, int alternate) {
    uint8_t model[32], tokenizer[32];
    for (int i = 0; i < 32; i++) {
        model[i] = (uint8_t)(0x31 + i * 3);
        tokenizer[i] = (uint8_t)((alternate ? 0xa7 : 0x73) + i * 5);
    }
    coli_v4_output_cache_test_identity(engine, model, tokenizer);
}

static int open_pair(ColiV4Engine **engine, ColiV4Session **session,
                     const char *model, int alternate_identity,
                     char *error, size_t error_size) {
    if (open_engine(engine, model, error, error_size)) return -1;
    install_identity(*engine, alternate_identity);
    if (open_session(session, *engine, error, error_size)) {
        coli_v4_engine_destroy(*engine);
        *engine = NULL;
        return -1;
    }
    return 0;
}

static int corrupt_only_object(const char *directory) {
    DIR *dir = opendir(directory);
    if (!dir) return -1;
    char path[2048] = {0};
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        size_t n = strlen(entry->d_name);
        if (strncmp(entry->d_name, "cpfx-", 5) || n < 5 ||
            strcmp(entry->d_name + n - 4, ".bin"))
            continue;
        int wrote = snprintf(path, sizeof(path), "%s/%s",
                             directory, entry->d_name);
        if (wrote <= 0 || (size_t)wrote >= sizeof(path)) {
            closedir(dir); return -1;
        }
        count++;
    }
    closedir(dir);
    if (count != 1) return -1;

    FILE *fp = fopen(path, "r+b");
    if (!fp) return -1;
    if (fseek(fp, -1, SEEK_END)) { fclose(fp); return -1; }
    int value = fgetc(fp);
    if (value == EOF || fseek(fp, -1, SEEK_CUR)) {
        fclose(fp); return -1;
    }
    unsigned char flipped = (unsigned char)value ^ 0x5a;
    int ok = fwrite(&flipped, 1, 1, fp) == 1 && fflush(fp) == 0;
    fclose(fp);
    return ok ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR INITIAL_PROMPT\n", argv[0]);
        return 2;
    }
    const char *model = argv[1], *prompt = argv[2];
    char error[1024] = {0};
    char directory[1024];
    snprintf(directory, sizeof(directory),
             "/tmp/coli-v4-output-cache-%lu-%llu",
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

    ColiV4Engine *engine = NULL;
    ColiV4Session *session = NULL;
    Capture baseline = {0}, restart = {0}, recovered = {0};
    Capture settings = {0}, identity = {0}, abort = {0}, post_abort = {0};
    ColiV4OutputCacheStats stats0 = {0}, stats1 = {0}, stats2 = {0};
    ColiV4OutputCacheStats stats3 = {0}, stats4 = {0}, stats5 = {0};
    int status = 1;

    if (open_pair(&engine, &session, model, 0, error, sizeof(error)) ||
        run_generation(session, prompt, 4, &baseline, error, sizeof(error))) {
        fprintf(stderr, "baseline output-cache generation failed: %s\n", error);
        goto cleanup;
    }
    coli_v4_output_cache_stats(&stats0);
    if (stats0.hits || stats0.stores != 1 || baseline.count != 4) {
        fprintf(stderr,
                "first request cache accounting invalid: hits=%llu stores=%llu generated=%d\n",
                (unsigned long long)stats0.hits,
                (unsigned long long)stats0.stores, baseline.count);
        goto cleanup;
    }
    coli_v4_session_destroy(session); session = NULL;
    coli_v4_engine_destroy(engine); engine = NULL;

    if (open_pair(&engine, &session, model, 0, error, sizeof(error)) ||
        run_generation(session, prompt, 4, &restart, error, sizeof(error))) {
        fprintf(stderr, "restart output-cache generation failed: %s\n", error);
        goto cleanup;
    }
    coli_v4_output_cache_stats(&stats1);
    if (stats1.hits != stats0.hits + 1 || !same_capture(&baseline, &restart) ||
        session->prefix_reused != 0 || !session->fed.tainted) {
        fprintf(stderr,
                "restart hit invalid: hits=%llu->%llu identity=%d prefix=%d tainted=%d\n",
                (unsigned long long)stats0.hits,
                (unsigned long long)stats1.hits,
                same_capture(&baseline, &restart), session->prefix_reused,
                session->fed.tainted);
        goto cleanup;
    }
    coli_v4_session_destroy(session); session = NULL;
    coli_v4_engine_destroy(engine); engine = NULL;

    if (corrupt_only_object(directory)) {
        fprintf(stderr, "could not corrupt the single output-cache object\n");
        goto cleanup;
    }
    if (open_pair(&engine, &session, model, 0, error, sizeof(error)) ||
        run_generation(session, prompt, 4, &recovered, error, sizeof(error))) {
        fprintf(stderr, "corruption fallback generation failed: %s\n", error);
        goto cleanup;
    }
    coli_v4_output_cache_stats(&stats2);
    if (stats2.hits != stats1.hits ||
        stats2.corruptions != stats1.corruptions + 1 ||
        !same_tokens(&baseline, &recovered)) {
        fprintf(stderr,
                "corruption did not fail closed: hits=%llu->%llu corruptions=%llu->%llu token_identity=%d\n",
                (unsigned long long)stats1.hits,
                (unsigned long long)stats2.hits,
                (unsigned long long)stats1.corruptions,
                (unsigned long long)stats2.corruptions,
                same_tokens(&baseline, &recovered));
        goto cleanup;
    }
    coli_v4_session_destroy(session); session = NULL;
    coli_v4_engine_destroy(engine); engine = NULL;

    if (open_pair(&engine, &session, model, 0, error, sizeof(error)) ||
        run_generation(session, prompt, 3, &settings, error, sizeof(error))) {
        fprintf(stderr, "settings-key generation failed: %s\n", error);
        goto cleanup;
    }
    coli_v4_output_cache_stats(&stats3);
    if (stats3.hits != stats2.hits || !prefix_capture(&baseline, &settings)) {
        fprintf(stderr,
                "generation settings unexpectedly hit: hits=%llu->%llu prefix_identity=%d\n",
                (unsigned long long)stats2.hits,
                (unsigned long long)stats3.hits,
                prefix_capture(&baseline, &settings));
        goto cleanup;
    }
    coli_v4_session_destroy(session); session = NULL;
    coli_v4_engine_destroy(engine); engine = NULL;

    if (open_pair(&engine, &session, model, 1, error, sizeof(error)) ||
        run_generation(session, prompt, 4, &identity, error, sizeof(error))) {
        fprintf(stderr, "identity-key generation failed: %s\n", error);
        goto cleanup;
    }
    coli_v4_output_cache_stats(&stats4);
    if (stats4.hits != stats3.hits || !same_tokens(&baseline, &identity)) {
        fprintf(stderr,
                "tokenizer/template identity gate failed: hits=%llu->%llu token_identity=%d\n",
                (unsigned long long)stats3.hits,
                (unsigned long long)stats4.hits,
                same_tokens(&baseline, &identity));
        goto cleanup;
    }
    coli_v4_session_destroy(session); session = NULL;
    coli_v4_engine_destroy(engine); engine = NULL;

    /* A caller-controlled early stop is not a model completion. Use a distinct
     * prompt composed only of known fixture tokens so the aborted run cannot
     * accidentally test tokenizer fallback instead of cache admission. */
    size_t prompt_len = strlen(prompt);
    if (prompt_len > (SIZE_MAX - 1) / 2) goto cleanup;
    char *abort_prompt = malloc(prompt_len * 2 + 1);
    if (!abort_prompt) goto cleanup;
    memcpy(abort_prompt, prompt, prompt_len);
    memcpy(abort_prompt + prompt_len, prompt, prompt_len + 1);
    abort.stop_after_first = 1;
    if (open_pair(&engine, &session, model, 0, error, sizeof(error)) ||
        run_generation(session, abort_prompt, 4, &abort, error, sizeof(error))) {
        fprintf(stderr, "callback-abort generation failed: %s\n", error);
        free(abort_prompt);
        goto cleanup;
    }
    coli_v4_output_cache_stats(&stats5);
    uint64_t stores_before_retry = stats5.stores;
    coli_v4_session_destroy(session); session = NULL;
    coli_v4_engine_destroy(engine); engine = NULL;
    if (open_pair(&engine, &session, model, 0, error, sizeof(error)) ||
        run_generation(session, abort_prompt, 4, &post_abort,
                       error, sizeof(error))) {
        fprintf(stderr, "post-abort generation failed: %s\n", error);
        free(abort_prompt);
        goto cleanup;
    }
    free(abort_prompt);
    ColiV4OutputCacheStats final_stats = {0};
    coli_v4_output_cache_stats(&final_stats);
    if (final_stats.hits != stats5.hits ||
        final_stats.stores != stores_before_retry + 1 ||
        abort.count != 1 || post_abort.count != 4) {
        fprintf(stderr,
                "callback abort was admitted: hits=%llu->%llu stores=%llu->%llu generated=%d/%d\n",
                (unsigned long long)stats5.hits,
                (unsigned long long)final_stats.hits,
                (unsigned long long)stores_before_retry,
                (unsigned long long)final_stats.stores,
                abort.count, post_abort.count);
        goto cleanup;
    }

    printf("PASS Stage3 V4 output cache: restart_hit=1 prompt=%d generated=%d token_identity=exact settings=miss tokenizer_template=miss callback_abort=not-admitted corruptions=%llu read=%.3fMiB write=%.3fMiB\n",
           session->prompt_count, baseline.count,
           (unsigned long long)final_stats.corruptions,
           final_stats.read_bytes / (1024.0 * 1024.0),
           final_stats.write_bytes / (1024.0 * 1024.0));
    status = 0;

cleanup:
    if (session) coli_v4_session_destroy(session);
    if (engine) coli_v4_engine_destroy(engine);
    return status;
}

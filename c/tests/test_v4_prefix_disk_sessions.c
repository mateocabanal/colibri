#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../compat.h"
#include "../deepseek_v4_internal.h"
#include "../coli_v4_prefix_cache.h"
#include "../coli_v4_prefix_disk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
            free(text); return -1;
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
            text = grown; capacity = next;
        }
        memcpy(text + length, piece, (size_t)bytes);
        length += (size_t)bytes;
        text[length] = 0;
    }
    if (!text) text = calloc(1, 1);
    if (!text) return -1;
    *output = text;
    return 0;
}

static char *append_text(const char *a, const char *b) {
    size_t na = strlen(a), nb = strlen(b);
    if (na > SIZE_MAX - nb - 1) return NULL;
    char *out = malloc(na + nb + 1);
    if (!out) return NULL;
    memcpy(out, a, na);
    memcpy(out + na, b, nb + 1);
    return out;
}

static int generated_text(ColiV4Session *session, char **output) {
    size_t capacity = (size_t)session->text_length + 1;
    char *text = malloc(capacity ? capacity : 1);
    if (!text) return -1;
    size_t length = 0;
    if (coli_v4_session_generated_text(session, text, capacity, &length)) {
        free(text); return -1;
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
    return coli_v4_session_create(session, engine, &options,
                                  error, error_size);
}

static int generate(ColiV4Session *session, const char *prompt,
                    char *error, size_t error_size) {
    ColiV4SessionGenerateOptions options = {
        .max_new_tokens = 4,
        .no_dspark = 1,
    };
    ColiV4SessionGenerateStats stats = {0};
    return coli_v4_session_generate(session, prompt, strlen(prompt), &options,
                                    NULL, NULL, &stats, error, error_size);
}

static void configure_cache(void) {
    char dir[1024];
#ifdef _WIN32
    const char *base = getenv("TEMP");
    if (!base || !*base) base = ".";
    snprintf(dir, sizeof(dir), "%s/coli-v4-prefix-disk-%lu-%llu", base,
             (unsigned long)getpid(), (unsigned long long)time(NULL));
    _putenv_s("COLI_PREFIX_CACHE", "auto");
    _putenv_s("COLI_PREFIX_CACHE_DIR", dir);
    _putenv_s("COLI_PREFIX_CACHE_DISK_GB", "0.125");
    _putenv_s("COLI_PREFIX_CACHE_MIN_FREE_GB", "0");
    _putenv_s("V4_PREFIX_CACHE_MB", "64");
    _putenv_s("V4_PREFIX_CACHE_MIN_TOKENS", "1");
#else
    snprintf(dir, sizeof(dir), "/tmp/coli-v4-prefix-disk-%lu-%llu",
             (unsigned long)getpid(), (unsigned long long)time(NULL));
    setenv("COLI_PREFIX_CACHE", "auto", 1);
    setenv("COLI_PREFIX_CACHE_DIR", dir, 1);
    setenv("COLI_PREFIX_CACHE_DISK_GB", "0.125", 1);
    setenv("COLI_PREFIX_CACHE_MIN_FREE_GB", "0", 1);
    setenv("V4_PREFIX_CACHE_MB", "64", 1);
    setenv("V4_PREFIX_CACHE_MIN_TOKENS", "1", 1);
#endif
}

static void disk_off_for_cold_engine(void) {
#ifdef _WIN32
    _putenv_s("COLI_PREFIX_CACHE", "ram");
#else
    setenv("COLI_PREFIX_CACHE", "ram", 1);
#endif
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s PACKAGE_DIR INITIAL_PROMPT\n", argv[0]);
        return 2;
    }
    const char *model = argv[1], *prefix = argv[2];
    char error[1024] = {0};
    ColiV4Engine *a_engine = NULL, *b_engine = NULL, *cold_engine = NULL;
    ColiV4Session *a = NULL, *b = NULL, *cold = NULL;
    char *extension = NULL, *extended = NULL;
    char *warm_text = NULL, *cold_text = NULL;
    int base_tokens = 0, status = 1;

    configure_cache();
    if (open_engine(&a_engine, model, error, sizeof(error)) ||
        open_session(&a, a_engine, error, sizeof(error)) ||
        generate(a, prefix, error, sizeof(error))) {
        fprintf(stderr, "writer generation failed: %s\n", error);
        goto cleanup;
    }
    if (!a->prompt_ids || a->prompt_count <= 0 ||
        decode_token_string(a, a->prompt_ids, 1, &extension)) {
        fprintf(stderr, "writer did not retain a usable prompt\n");
        goto cleanup;
    }
    base_tokens = a->prompt_count;
    extended = append_text(prefix, extension);
    if (!extended) goto cleanup;

    /* Destroying the engine retires all process-local entries. The immutable
     * SSD object must survive this boundary. */
    coli_v4_session_destroy(a); a = NULL;
    coli_v4_engine_destroy(a_engine); a_engine = NULL;

    memset(error, 0, sizeof(error));
    if (open_engine(&b_engine, model, error, sizeof(error)) ||
        open_session(&b, b_engine, error, sizeof(error)) ||
        generate(b, extended, error, sizeof(error))) {
        fprintf(stderr, "restart generation failed: %s\n", error);
        goto cleanup;
    }
    if (b->prefix_reused != base_tokens) {
        fprintf(stderr, "SSD restore reused %d tokens, expected %d\n",
                b->prefix_reused, base_tokens);
        goto cleanup;
    }
    if (generated_text(b, &warm_text)) goto cleanup;

    coli_v4_session_destroy(b); b = NULL;
    coli_v4_engine_destroy(b_engine); b_engine = NULL;

    /* Same process, new engine, but no SSD registration. Process RAM entries
     * were retired with engine B, so this is a genuine cold oracle. */
    disk_off_for_cold_engine();
    memset(error, 0, sizeof(error));
    if (open_engine(&cold_engine, model, error, sizeof(error)) ||
        open_session(&cold, cold_engine, error, sizeof(error)) ||
        generate(cold, extended, error, sizeof(error))) {
        fprintf(stderr, "cold generation failed: %s\n", error);
        goto cleanup;
    }
    if (cold->prefix_reused != 0 || generated_text(cold, &cold_text)) {
        fprintf(stderr, "cold oracle unexpectedly reused state\n");
        goto cleanup;
    }
    if (strcmp(warm_text, cold_text)) {
        fprintf(stderr,
                "SSD-restored output diverged from cold execution:\nrestored=%s\ncold=%s\n",
                warm_text, cold_text);
        goto cleanup;
    }

    printf("PASS V4 persistent prefix cache: restored %d tokens after engine restart; 4-token output identical to cold execution\n",
           base_tokens);
    status = 0;

cleanup:
    free(extension);
    free(extended);
    free(warm_text);
    free(cold_text);
    if (a) coli_v4_session_destroy(a);
    if (b) coli_v4_session_destroy(b);
    if (cold) coli_v4_session_destroy(cold);
    if (a_engine) coli_v4_engine_destroy(a_engine);
    if (b_engine) coli_v4_engine_destroy(b_engine);
    if (cold_engine) coli_v4_engine_destroy(cold_engine);
    return status;
}

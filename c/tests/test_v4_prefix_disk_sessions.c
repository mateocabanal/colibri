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

static const uint8_t k_test_model_fingerprint[32] = {
    0x31, 0x38, 0x3f, 0x46, 0x4d, 0x54, 0x5b, 0x62,
    0x69, 0x70, 0x77, 0x7e, 0x85, 0x8c, 0x93, 0x9a,
    0xa1, 0xa8, 0xaf, 0xb6, 0xbd, 0xc4, 0xcb, 0xd2,
    0xd9, 0xe0, 0xe7, 0xee, 0xf5, 0xfc, 0x03, 0x0a,
};
#define TEST_LAYOUT_FINGERPRINT UINT64_C(0x5353445354414745) /* SSDSTAGE */

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

static int register_test_namespace(ColiV4Engine *engine) {
    return coli_v4_prefix_disk_register_test_namespace(
        engine, k_test_model_fingerprint, TEST_LAYOUT_FINGERPRINT);
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
    _putenv_s("COLI_PREFIX_CACHE", "ssd");
    _putenv_s("COLI_PREFIX_CACHE_DIR", dir);
    _putenv_s("COLI_PREFIX_CACHE_DISK_GB", "0.125");
    _putenv_s("COLI_PREFIX_CACHE_MIN_FREE_GB", "0");
    _putenv_s("V4_PREFIX_CACHE_MB", "off");
    _putenv_s("V4_PREFIX_CACHE_MIN_TOKENS", "1");
#else
    snprintf(dir, sizeof(dir), "/tmp/coli-v4-prefix-disk-%lu-%llu",
             (unsigned long)getpid(), (unsigned long long)time(NULL));
    setenv("COLI_PREFIX_CACHE", "ssd", 1);
    setenv("COLI_PREFIX_CACHE_DIR", dir, 1);
    setenv("COLI_PREFIX_CACHE_DISK_GB", "0.125", 1);
    setenv("COLI_PREFIX_CACHE_MIN_FREE_GB", "0", 1);
    setenv("V4_PREFIX_CACHE_MB", "off", 1);
    setenv("V4_PREFIX_CACHE_MIN_TOKENS", "1", 1);
#endif
}

static void cache_off_for_cold_engine(void) {
#ifdef _WIN32
    _putenv_s("COLI_PREFIX_CACHE", "off");
#else
    setenv("COLI_PREFIX_CACHE", "off", 1);
#endif
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR INITIAL_PROMPT\n", argv[0]);
        return 2;
    }
    const char *model = argv[1], *prefix = argv[2];
    char error[1024] = {0};
    ColiV4Engine *a_engine = NULL, *b_engine = NULL, *cold_engine = NULL;
    ColiV4Session *a = NULL, *b = NULL, *cold = NULL;
    char *extension = NULL, *extended = NULL;
    char *warm_text = NULL, *cold_text = NULL;
    int base_tokens = 0, status = 1;
    ColiV4PrefixDiskStats writer_disk = {0}, reader_disk = {0};

    configure_cache();
    if (open_engine(&a_engine, model, error, sizeof(error)) ||
        !register_test_namespace(a_engine) ||
        open_session(&a, a_engine, error, sizeof(error)) ||
        generate(a, prefix, error, sizeof(error))) {
        fprintf(stderr, "SSD-only writer generation failed: %s\n", error);
        goto cleanup;
    }

    ColiV4PrefixCacheStats ram_stats = {0};
    coli_v4_prefix_cache_stats(&ram_stats);
    if (ram_stats.budget_bytes || ram_stats.entries || ram_stats.resident_bytes) {
        fprintf(stderr,
                "SSD-only writer unexpectedly retained RAM prefix state: budget=%zu entries=%zu resident=%zu\n",
                ram_stats.budget_bytes, ram_stats.entries, ram_stats.resident_bytes);
        goto cleanup;
    }
    coli_v4_prefix_disk_stats(a_engine, &writer_disk);
    if (!writer_disk.enabled || writer_disk.stores < 1 ||
        writer_disk.write_bytes == 0 || writer_disk.hits != 0) {
        fprintf(stderr,
                "SSD writer telemetry invalid: enabled=%d hits=%llu stores=%llu write=%llu\n",
                writer_disk.enabled,
                (unsigned long long)writer_disk.hits,
                (unsigned long long)writer_disk.stores,
                (unsigned long long)writer_disk.write_bytes);
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

    /* No RAM cache entry exists in SSD-only mode. Destroying engine A retires
     * the in-process namespace; only the atomically-published SSD object remains. */
    coli_v4_session_destroy(a); a = NULL;
    coli_v4_engine_destroy(a_engine); a_engine = NULL;

    memset(error, 0, sizeof(error));
    if (open_engine(&b_engine, model, error, sizeof(error)) ||
        !register_test_namespace(b_engine) ||
        open_session(&b, b_engine, error, sizeof(error)) ||
        generate(b, extended, error, sizeof(error))) {
        fprintf(stderr, "SSD-only restart generation failed: %s\n", error);
        goto cleanup;
    }
    if (b->prefix_reused != base_tokens) {
        fprintf(stderr, "SSD restore reused %d tokens, expected %d\n",
                b->prefix_reused, base_tokens);
        goto cleanup;
    }
    coli_v4_prefix_disk_stats(b_engine, &reader_disk);
    if (!reader_disk.enabled || reader_disk.hits < 1 ||
        reader_disk.read_bytes == 0 || reader_disk.corruptions != 0) {
        fprintf(stderr,
                "SSD reader telemetry invalid: enabled=%d hits=%llu read=%llu corruptions=%llu\n",
                reader_disk.enabled,
                (unsigned long long)reader_disk.hits,
                (unsigned long long)reader_disk.read_bytes,
                (unsigned long long)reader_disk.corruptions);
        goto cleanup;
    }
    if (generated_text(b, &warm_text)) goto cleanup;

    coli_v4_session_destroy(b); b = NULL;
    coli_v4_engine_destroy(b_engine); b_engine = NULL;

    /* Disable both tiers before opening the oracle engine. The process RAM
     * cache has a zero budget, so this is a genuine cold execution. */
    cache_off_for_cold_engine();
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

    printf("PASS Stage2 V4 SSD prefix cache: ram=0 restart_reuse=%d write=%.3fMiB read=%.3fMiB token_identity=exact corruptions=0\n",
           base_tokens,
           (double)writer_disk.write_bytes / (1024.0 * 1024.0),
           (double)reader_disk.read_bytes / (1024.0 * 1024.0));
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

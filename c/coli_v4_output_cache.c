#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "coli_v4_output_cache.h"
#include "deepseek_v4_internal.h"
#include "prefix_cache_disk.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef COLI_V4_GIT_SHA
#define COLI_V4_GIT_SHA "unknown"
#endif

#define COLI_V4_OUTPUT_ENGINE_ID UINT32_C(0x56344f43) /* V4OC */
#define COLI_V4_OUTPUT_STATE_ABI 1u
#define COLI_V4_OUTPUT_PAYLOAD_VERSION 1u
#define COLI_V4_OUTPUT_DEFAULT_GB 2u
#define COLI_V4_OUTPUT_KEY_WORDS 8
#define COLI_V4_OUTPUT_KEY_SENTINEL INT32_C(0x4f434b31) /* OCK1 */

static const char g_v4_output_template_id[] =
    "v4-session-api-preformatted-greedy-v1";
static const char g_v4_output_payload_magic[8] =
    {'V','4','O','U','T','0','1','\0'};

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint8_t tokenizer_template_fingerprint[32];
    uint64_t execution_fingerprint;
    uint32_t prompt_count;
    uint32_t generated_count;
    int32_t max_new_tokens;
    int32_t stop_at_sentence;
    int32_t no_dspark;
    uint32_t reserved;
} V4OutputPayloadHeader;

typedef struct {
    int32_t token;
    float logit;
    int32_t position;
    int32_t ordinal;
} V4OutputTokenRecord;

typedef struct {
    uint8_t model[32];
    uint8_t tokenizer_template[32];
    uint64_t execution;
} V4OutputIdentity;

static pthread_mutex_t g_v4_output_stats_mutex = PTHREAD_MUTEX_INITIALIZER;
static ColiV4OutputCacheStats g_v4_output_stats;

#ifdef COLI_V4_OUTPUT_CACHE_TESTING
static pthread_mutex_t g_v4_output_test_mutex = PTHREAD_MUTEX_INITIALIZER;
static ColiV4Engine *g_v4_output_test_engine;
static uint8_t g_v4_output_test_model[32];
static uint8_t g_v4_output_test_tokenizer[32];

void coli_v4_output_cache_test_identity(
    ColiV4Engine *engine, const uint8_t model_fingerprint[32],
    const uint8_t tokenizer_template_fingerprint[32]) {
    pthread_mutex_lock(&g_v4_output_test_mutex);
    g_v4_output_test_engine = engine;
    if (model_fingerprint)
        memcpy(g_v4_output_test_model, model_fingerprint, 32);
    else
        memset(g_v4_output_test_model, 0, 32);
    if (tokenizer_template_fingerprint)
        memcpy(g_v4_output_test_tokenizer,
               tokenizer_template_fingerprint, 32);
    else
        memset(g_v4_output_test_tokenizer, 0, 32);
    pthread_mutex_unlock(&g_v4_output_test_mutex);
}
#endif

static uint64_t output_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) +
           (uint64_t)ts.tv_nsec;
}

static int output_ascii_ieq(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return *a == 0 && *b == 0;
}

static int output_policy_enabled(void) {
    const char *mode = getenv("COLI_OUTPUT_CACHE");
    if (mode && *mode) {
        if (output_ascii_ieq(mode, "on") || output_ascii_ieq(mode, "1"))
            return 1;
        if (output_ascii_ieq(mode, "auto")) {
            const char *serve = getenv("SERVE");
            return serve && serve[0] == '1';
        }
        return 0;
    }
    const char *serve = getenv("SERVE");
    return serve && serve[0] == '1';
}

static size_t output_budget_bytes(void) {
    const char *value = getenv("COLI_OUTPUT_CACHE_GB");
    if (!value || !*value)
        return (size_t)COLI_V4_OUTPUT_DEFAULT_GB * 1024u * 1024u * 1024u;
    char *end = NULL;
    long double gib = strtold(value, &end);
    if (end == value || !(gib > 0.0L)) return 0;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')
        end++;
    if (*end) return 0;
    long double bytes = gib * 1024.0L * 1024.0L * 1024.0L;
    if (bytes >= (long double)SIZE_MAX) return SIZE_MAX;
    return (size_t)bytes;
}

static int output_directory(char *out, size_t cap) {
    if (!out || !cap) return 0;
    const char *explicit_dir = getenv("COLI_OUTPUT_CACHE_DIR");
    if (explicit_dir && *explicit_dir) {
        int wrote = snprintf(out, cap, "%s", explicit_dir);
        return wrote > 0 && (size_t)wrote < cap;
    }
    char base[COLI_PREFIX_DISK_PATH_MAX];
    if (!coli_prefix_disk_default_directory(base, sizeof(base))) return 0;
    int wrote = snprintf(out, cap, "%s/output-v4", base);
    return wrote > 0 && (size_t)wrote < cap;
}

static int output_hash_file(uint64_t lanes[4], const char *directory,
                            const char *name) {
    char path[COLI_PREFIX_DISK_PATH_MAX];
    int wrote = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (wrote <= 0 || (size_t)wrote >= sizeof(path)) return 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    for (int lane = 0; lane < 4; lane++)
        lanes[lane] = coli_prefix_hash64(name, strlen(name) + 1, lanes[lane]);
    unsigned char buffer[64 * 1024];
    for (;;) {
        size_t got = fread(buffer, 1, sizeof(buffer), fp);
        if (got) {
            for (int lane = 0; lane < 4; lane++)
                lanes[lane] = coli_prefix_hash64(buffer, got, lanes[lane]);
        }
        if (got < sizeof(buffer)) {
            int ok = !ferror(fp);
            fclose(fp);
            return ok;
        }
    }
}

static void output_hash_init(uint64_t h[4]) {
    h[0] = UINT64_C(1469598103934665603);
    h[1] = UINT64_C(1099511628211) ^ UINT64_C(0x9e3779b97f4a7c15);
    h[2] = UINT64_C(0x6a09e667f3bcc909);
    h[3] = UINT64_C(0xbb67ae8584caa73b);
}

static int output_model_fingerprint(const ColiV4Engine *engine,
                                    uint8_t out[32]) {
    if (!engine || !out || !engine->coli_static || engine->target_index ||
        !engine->runtime.target_model_dir || !*engine->runtime.target_model_dir)
        return 0;
    const ColiPackage *package = coli_executor_package(engine->coli_static);
    if (!package) return 0;
    const uint8_t *source = coli_package_source_fingerprint(package);
    if (!source) return 0;

    uint64_t h[4];
    output_hash_init(h);
    const char *profile = coli_package_profile(package);
    const char *compiler = coli_package_compiler(package);
    for (int lane = 0; lane < 4; lane++) {
        h[lane] = coli_prefix_hash64(source, 32, h[lane]);
        if (profile)
            h[lane] = coli_prefix_hash64(profile, strlen(profile) + 1, h[lane]);
        if (compiler)
            h[lane] = coli_prefix_hash64(compiler, strlen(compiler) + 1, h[lane]);
    }
    size_t records = coli_package_record_count(package);
    for (size_t i = 0; i < records; i++) {
        const ColiRecordInfo *r = coli_package_record_at(package, i);
        if (!r) return 0;
        uint64_t meta[] = {
            r->record_id,
            ((uint64_t)r->kind << 48) | ((uint64_t)r->codec << 32) |
                ((uint64_t)r->math_format << 16) | r->scale_format,
            ((uint64_t)r->layout << 48) | ((uint64_t)r->flags << 32) |
                r->shard_id,
            ((uint64_t)(uint32_t)r->layer << 32) | (uint32_t)r->expert,
            r->stored_bytes,
            r->decoded_bytes,
            ((uint64_t)r->stored_crc32c << 32) | r->logical_crc32c,
        };
        for (int lane = 0; lane < 4; lane++) {
            h[lane] = coli_prefix_hash64(meta, sizeof(meta), h[lane]);
            if (r->name)
                h[lane] = coli_prefix_hash64(r->name,
                                             strlen(r->name) + 1, h[lane]);
        }
    }
    if (!output_hash_file(h, engine->runtime.target_model_dir, "config.json") ||
        !output_hash_file(h, engine->runtime.target_model_dir, "tokenizer.json"))
        return 0;
    memcpy(out, h, 32);
    return 1;
}

static int output_tokenizer_template_fingerprint(const ColiV4Engine *engine,
                                                 uint8_t out[32]) {
    if (!engine || !out || !engine->runtime.target_model_dir ||
        !*engine->runtime.target_model_dir)
        return 0;
    uint64_t h[4];
    output_hash_init(h);
    for (int lane = 0; lane < 4; lane++)
        h[lane] = coli_prefix_hash64(g_v4_output_template_id,
                                     sizeof(g_v4_output_template_id), h[lane]);
    if (!output_hash_file(h, engine->runtime.target_model_dir, "tokenizer.json"))
        return 0;
    memcpy(out, h, 32);
    return 1;
}

static uint64_t output_execution_fingerprint(const ColiV4Engine *engine,
                                             const uint8_t tokenizer[32]) {
    uint64_t h = UINT64_C(1469598103934665603);
    const char build[] = COLI_V4_GIT_SHA;
    h = coli_prefix_hash64(build, sizeof(build), h);
    h = coli_prefix_hash64(g_v4_output_template_id,
                           sizeof(g_v4_output_template_id), h);
    h = coli_prefix_hash64(tokenizer, 32, h);
    uint64_t runtime[] = {
        COLI_V4_OUTPUT_STATE_ABI,
        COLI_V4_OUTPUT_PAYLOAD_VERSION,
        engine ? (uint64_t)(uint32_t)engine->runtime.context_tokens : 0,
        (uint64_t)sizeof(float),
#ifdef COLI_METAL
        UINT64_C(1),
#else
        UINT64_C(0),
#endif
    };
    return coli_prefix_hash64(runtime, sizeof(runtime), h);
}

static int output_identity(ColiV4Engine *engine, V4OutputIdentity *identity) {
    if (!engine || !identity) return 0;
#ifdef COLI_V4_OUTPUT_CACHE_TESTING
    pthread_mutex_lock(&g_v4_output_test_mutex);
    if (g_v4_output_test_engine == engine) {
        memcpy(identity->model, g_v4_output_test_model, 32);
        memcpy(identity->tokenizer_template, g_v4_output_test_tokenizer, 32);
        pthread_mutex_unlock(&g_v4_output_test_mutex);
        identity->execution = output_execution_fingerprint(
            engine, identity->tokenizer_template);
        return 1;
    }
    pthread_mutex_unlock(&g_v4_output_test_mutex);
#endif
    if (!output_model_fingerprint(engine, identity->model) ||
        !output_tokenizer_template_fingerprint(
            engine, identity->tokenizer_template))
        return 0;
    identity->execution = output_execution_fingerprint(
        engine, identity->tokenizer_template);
    return identity->execution != 0;
}

static int output_cache_init(ColiV4Engine *engine, ColiPrefixDiskCache *cache,
                             V4OutputIdentity *identity) {
    if (!engine || !cache || !identity || !output_policy_enabled()) return 0;
    if (!output_identity(engine, identity)) {
        if (getenv("V4_OUTPUT_CACHE_LOG"))
            fprintf(stderr,
                    "[OUTPUT-CACHE] disabled: strong package/tokenizer identity unavailable\n");
        return 0;
    }
    char directory[COLI_PREFIX_DISK_PATH_MAX];
    size_t budget = output_budget_bytes();
    if (!budget || !output_directory(directory, sizeof(directory))) return 0;
    return coli_prefix_disk_init(cache, directory, budget, 1,
                                 COLI_V4_OUTPUT_ENGINE_ID,
                                 COLI_V4_OUTPUT_STATE_ABI,
                                 identity->model, identity->execution, 0) &&
           cache->enabled;
}

static int output_effective_settings(const ColiV4Session *session,
                                     const ColiV4SessionGenerateOptions *options,
                                     int *max_new, int *stop_sentence,
                                     int *no_dspark) {
    if (!session || !options || options->max_new_tokens < 1 ||
        !options->no_dspark)
        return 0;
    int limit = options->max_new_tokens;
    if (limit > session->max_new_tokens_cap) limit = session->max_new_tokens_cap;
    if (limit < 1) return 0;
    if (max_new) *max_new = limit;
    if (stop_sentence) *stop_sentence = !!options->stop_at_sentence;
    if (no_dspark) *no_dspark = 1;
    return 1;
}

static void output_request_digest(uint64_t digest[4],
                                  const V4OutputIdentity *identity,
                                  const int *prompt_ids, int prompt_count,
                                  int max_new, int stop_sentence,
                                  int no_dspark) {
    output_hash_init(digest);
    int32_t settings[4] = {
        (int32_t)prompt_count,
        (int32_t)max_new,
        (int32_t)stop_sentence,
        (int32_t)no_dspark,
    };
    for (int lane = 0; lane < 4; lane++) {
        digest[lane] = coli_prefix_hash64(identity->model, 32, digest[lane]);
        digest[lane] = coli_prefix_hash64(identity->tokenizer_template, 32,
                                           digest[lane]);
        digest[lane] = coli_prefix_hash64(&identity->execution,
                                           sizeof(identity->execution),
                                           digest[lane]);
        digest[lane] = coli_prefix_hash64(settings, sizeof(settings),
                                           digest[lane]);
        digest[lane] = coli_prefix_hash64(prompt_ids,
                                           (size_t)prompt_count * sizeof(int),
                                           digest[lane]);
    }
}

static int *output_build_key(const V4OutputIdentity *identity,
                             const int *prompt_ids, int prompt_count,
                             int max_new, int stop_sentence, int no_dspark,
                             int *key_count) {
    if (!identity || !prompt_ids || prompt_count < 1 || !key_count ||
        prompt_count > INT_MAX - COLI_V4_OUTPUT_KEY_WORDS - 3)
        return NULL;
    int count = COLI_V4_OUTPUT_KEY_WORDS + prompt_count + 3;
    int *key = malloc((size_t)count * sizeof(*key));
    if (!key) return NULL;
    uint64_t digest[4];
    output_request_digest(digest, identity, prompt_ids, prompt_count,
                          max_new, stop_sentence, no_dspark);
    for (int lane = 0; lane < 4; lane++) {
        key[lane * 2] = (int)(uint32_t)(digest[lane] & UINT32_MAX);
        key[lane * 2 + 1] = (int)(uint32_t)(digest[lane] >> 32);
    }
    memcpy(key + COLI_V4_OUTPUT_KEY_WORDS, prompt_ids,
           (size_t)prompt_count * sizeof(*key));
    int at = COLI_V4_OUTPUT_KEY_WORDS + prompt_count;
    key[at++] = max_new;
    key[at++] = stop_sentence;
    key[at++] = no_dspark;
    *key_count = count;
    return key;
}

static int output_payload_decode(const unsigned char *payload,
                                 size_t payload_bytes,
                                 const V4OutputIdentity *identity,
                                 const int *prompt_ids, int prompt_count,
                                 int max_new, int stop_sentence, int no_dspark,
                                 ColiV4OutputCacheHit *hit) {
    if (!payload || payload_bytes < sizeof(V4OutputPayloadHeader) ||
        !identity || !prompt_ids || !hit)
        return 0;
    V4OutputPayloadHeader header;
    memcpy(&header, payload, sizeof(header));
    if (memcmp(header.magic, g_v4_output_payload_magic, 8) ||
        header.version != COLI_V4_OUTPUT_PAYLOAD_VERSION ||
        header.header_bytes != sizeof(header) ||
        memcmp(header.tokenizer_template_fingerprint,
               identity->tokenizer_template, 32) ||
        header.execution_fingerprint != identity->execution ||
        header.prompt_count != (uint32_t)prompt_count ||
        header.generated_count < 1 || header.generated_count > (uint32_t)max_new ||
        header.max_new_tokens != max_new ||
        header.stop_at_sentence != stop_sentence ||
        header.no_dspark != no_dspark)
        return 0;

    size_t prompt_bytes = (size_t)prompt_count * sizeof(int32_t);
    size_t generated = (size_t)header.generated_count;
    if (generated > SIZE_MAX / sizeof(V4OutputTokenRecord)) return 0;
    size_t record_bytes = generated * sizeof(V4OutputTokenRecord);
    if (sizeof(header) > SIZE_MAX - prompt_bytes ||
        sizeof(header) + prompt_bytes > SIZE_MAX - record_bytes ||
        payload_bytes != sizeof(header) + prompt_bytes + record_bytes)
        return 0;

    const unsigned char *cursor = payload + sizeof(header);
    for (int i = 0; i < prompt_count; i++) {
        int32_t stored;
        memcpy(&stored, cursor + (size_t)i * sizeof(stored), sizeof(stored));
        if (stored != (int32_t)prompt_ids[i]) return 0;
    }
    cursor += prompt_bytes;

    int count = (int)header.generated_count;
    hit->prompt_ids = malloc((size_t)prompt_count * sizeof(*hit->prompt_ids));
    hit->tokens = malloc((size_t)count * sizeof(*hit->tokens));
    hit->logits = malloc((size_t)count * sizeof(*hit->logits));
    hit->positions = malloc((size_t)count * sizeof(*hit->positions));
    hit->ordinals = malloc((size_t)count * sizeof(*hit->ordinals));
    if (!hit->prompt_ids || !hit->tokens || !hit->logits ||
        !hit->positions || !hit->ordinals) {
        coli_v4_output_cache_hit_free(hit);
        return 0;
    }
    memcpy(hit->prompt_ids, prompt_ids,
           (size_t)prompt_count * sizeof(*hit->prompt_ids));
    hit->prompt_count = prompt_count;
    hit->generated_count = count;
    for (int i = 0; i < count; i++) {
        V4OutputTokenRecord record;
        memcpy(&record, cursor + (size_t)i * sizeof(record), sizeof(record));
        hit->tokens[i] = (int)record.token;
        hit->logits[i] = record.logit;
        hit->positions[i] = (int)record.position;
        hit->ordinals[i] = (int)record.ordinal;
        if (record.ordinal != i + 1) {
            coli_v4_output_cache_hit_free(hit);
            return 0;
        }
    }
    return 1;
}

static void output_stats_add(const ColiV4OutputCacheStats *delta) {
    if (!delta) return;
    pthread_mutex_lock(&g_v4_output_stats_mutex);
    g_v4_output_stats.lookups += delta->lookups;
    g_v4_output_stats.hits += delta->hits;
    g_v4_output_stats.stores += delta->stores;
    g_v4_output_stats.bypasses += delta->bypasses;
    g_v4_output_stats.corruptions += delta->corruptions;
    g_v4_output_stats.identity_rejects += delta->identity_rejects;
    g_v4_output_stats.read_bytes += delta->read_bytes;
    g_v4_output_stats.write_bytes += delta->write_bytes;
    pthread_mutex_unlock(&g_v4_output_stats_mutex);
}

int coli_v4_output_cache_lookup(ColiV4Session *session,
                                const char *prompt, size_t prompt_length,
                                const ColiV4SessionGenerateOptions *options,
                                ColiV4OutputCacheHit *hit) {
    if (hit) memset(hit, 0, sizeof(*hit));
    ColiV4OutputCacheStats delta = {0};
    if (!session || !session->engine || !prompt || !options || !hit ||
        !output_policy_enabled()) {
        delta.bypasses = 1;
        output_stats_add(&delta);
        return 0;
    }
    int max_new = 0, stop_sentence = 0, no_dspark = 0;
    if (!output_effective_settings(session, options, &max_new,
                                   &stop_sentence, &no_dspark)) {
        delta.bypasses = 1;
        output_stats_add(&delta);
        return 0;
    }

    int capacity = session->max_prompt_tokens + 16;
    int *prompt_ids = malloc((size_t)capacity * sizeof(*prompt_ids));
    if (!prompt_ids) return 0;
    int prompt_count = tok_encode(&session->tokenizer, prompt, prompt_length,
                                  prompt_ids, capacity);
    if (prompt_count < 1 || prompt_count > session->max_prompt_tokens) {
        free(prompt_ids);
        return 0;
    }

    V4OutputIdentity identity;
    ColiPrefixDiskCache cache;
    if (!output_cache_init(session->engine, &cache, &identity)) {
        free(prompt_ids);
        delta.bypasses = 1;
        output_stats_add(&delta);
        return 0;
    }
    int key_count = 0;
    int *key = output_build_key(&identity, prompt_ids, prompt_count,
                                max_new, stop_sentence, no_dspark, &key_count);
    if (!key) {
        free(prompt_ids);
        return 0;
    }
    int *query = malloc((size_t)(key_count + 1) * sizeof(*query));
    if (!query) {
        free(key); free(prompt_ids); return 0;
    }
    memcpy(query, key, (size_t)key_count * sizeof(*query));
    query[key_count] = (int)COLI_V4_OUTPUT_KEY_SENTINEL;

    delta.lookups = 1;
    uint64_t began = output_now_ns();
    ColiPrefixDiskHit raw_hit = {0};
    int matched = coli_prefix_disk_restore_longest(
        &cache, query, key_count + 1, &raw_hit);
    uint64_t ended = output_now_ns();
    ColiPrefixDiskStats raw_stats;
    coli_prefix_disk_stats(&cache, &raw_stats);
    delta.corruptions = raw_stats.corruptions;
    delta.read_bytes = raw_stats.read_bytes;

    int accepted = matched == key_count && raw_hit.token_count == key_count &&
        output_payload_decode(raw_hit.payload, raw_hit.payload_bytes,
                              &identity, prompt_ids, prompt_count,
                              max_new, stop_sentence, no_dspark, hit);
    if (matched && !accepted) delta.identity_rejects = 1;
    if (accepted) {
        delta.hits = 1;
        hit->read_bytes = raw_stats.read_bytes;
        hit->restore_sec = ended >= began ? (ended - began) * 1.0e-9 : 0.0;
        if (getenv("V4_OUTPUT_CACHE_LOG"))
            fprintf(stderr,
                    "[OUTPUT-CACHE] hit prompt=%d generated=%d read=%.3fMiB restore=%.3fms\n",
                    prompt_count, hit->generated_count,
                    hit->read_bytes / (1024.0 * 1024.0),
                    hit->restore_sec * 1000.0);
    }

    coli_prefix_disk_hit_free(&raw_hit);
    free(query);
    free(key);
    free(prompt_ids);
    output_stats_add(&delta);
    return accepted;
}

int coli_v4_output_cache_store(ColiV4Session *session,
                               const ColiV4SessionGenerateOptions *options,
                               const int *tokens, const float *logits,
                               const int *positions, const int *ordinals,
                               int generated_count) {
    if (!session || !session->engine || !options || !tokens || !logits ||
        !positions || !ordinals || generated_count < 1 ||
        !session->prompt_ids || session->prompt_count < 1 ||
        !output_policy_enabled())
        return 0;
    int max_new = 0, stop_sentence = 0, no_dspark = 0;
    if (!output_effective_settings(session, options, &max_new,
                                   &stop_sentence, &no_dspark) ||
        generated_count > max_new)
        return 0;

    V4OutputIdentity identity;
    ColiPrefixDiskCache cache;
    if (!output_cache_init(session->engine, &cache, &identity)) return 0;
    int key_count = 0;
    int *key = output_build_key(&identity, session->prompt_ids,
                                session->prompt_count, max_new,
                                stop_sentence, no_dspark, &key_count);
    if (!key) return 0;

    size_t prompt_bytes = (size_t)session->prompt_count * sizeof(int32_t);
    if ((size_t)generated_count > SIZE_MAX / sizeof(V4OutputTokenRecord)) {
        free(key); return 0;
    }
    size_t record_bytes = (size_t)generated_count * sizeof(V4OutputTokenRecord);
    if (sizeof(V4OutputPayloadHeader) > SIZE_MAX - prompt_bytes ||
        sizeof(V4OutputPayloadHeader) + prompt_bytes > SIZE_MAX - record_bytes) {
        free(key); return 0;
    }
    size_t payload_bytes = sizeof(V4OutputPayloadHeader) + prompt_bytes + record_bytes;
    unsigned char *payload = malloc(payload_bytes);
    if (!payload) { free(key); return 0; }

    V4OutputPayloadHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, g_v4_output_payload_magic, 8);
    header.version = COLI_V4_OUTPUT_PAYLOAD_VERSION;
    header.header_bytes = (uint32_t)sizeof(header);
    memcpy(header.tokenizer_template_fingerprint,
           identity.tokenizer_template, 32);
    header.execution_fingerprint = identity.execution;
    header.prompt_count = (uint32_t)session->prompt_count;
    header.generated_count = (uint32_t)generated_count;
    header.max_new_tokens = max_new;
    header.stop_at_sentence = stop_sentence;
    header.no_dspark = no_dspark;
    memcpy(payload, &header, sizeof(header));

    unsigned char *cursor = payload + sizeof(header);
    for (int i = 0; i < session->prompt_count; i++) {
        int32_t value = (int32_t)session->prompt_ids[i];
        memcpy(cursor + (size_t)i * sizeof(value), &value, sizeof(value));
    }
    cursor += prompt_bytes;
    for (int i = 0; i < generated_count; i++) {
        V4OutputTokenRecord record = {
            (int32_t)tokens[i], logits[i],
            (int32_t)positions[i], (int32_t)ordinals[i]
        };
        memcpy(cursor + (size_t)i * sizeof(record), &record, sizeof(record));
    }

    int stored = coli_prefix_disk_store(&cache, key, key_count,
                                        payload, payload_bytes);
    ColiPrefixDiskStats raw_stats;
    coli_prefix_disk_stats(&cache, &raw_stats);
    ColiV4OutputCacheStats delta = {0};
    delta.stores = raw_stats.stores;
    delta.write_bytes = raw_stats.write_bytes;
    output_stats_add(&delta);
    if (stored && raw_stats.stores && getenv("V4_OUTPUT_CACHE_LOG"))
        fprintf(stderr,
                "[OUTPUT-CACHE] store prompt=%d generated=%d write=%.3fMiB\n",
                session->prompt_count, generated_count,
                raw_stats.write_bytes / (1024.0 * 1024.0));

    free(payload);
    free(key);
    return stored;
}

void coli_v4_output_cache_hit_free(ColiV4OutputCacheHit *hit) {
    if (!hit) return;
    free(hit->prompt_ids);
    free(hit->tokens);
    free(hit->logits);
    free(hit->positions);
    free(hit->ordinals);
    memset(hit, 0, sizeof(*hit));
}

void coli_v4_output_cache_stats(ColiV4OutputCacheStats *stats) {
    if (!stats) return;
    pthread_mutex_lock(&g_v4_output_stats_mutex);
    *stats = g_v4_output_stats;
    pthread_mutex_unlock(&g_v4_output_stats_mutex);
}

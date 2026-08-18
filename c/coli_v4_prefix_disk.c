#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "deepseek_v4_internal.h"
#include "coli_v4_prefix_cache.h"
#include "coli_v4_prefix_disk.h"
#include "prefix_cache_disk.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef COLI_V4_GIT_SHA
#define COLI_V4_GIT_SHA "unknown"
#endif

#define COLI_V4_PREFIX_DISK_ENGINE_ID UINT32_C(0x56345046) /* V4PF */
#define COLI_V4_PREFIX_DISK_STATE_ABI 1u
#define COLI_V4_PREFIX_DISK_PAYLOAD_VERSION 1u
#define COLI_V4_PREFIX_DISK_DEFAULT_GB 8u
#define COLI_V4_PREFIX_DISK_MAX_ENGINES 16

typedef struct {
    ColiV4Engine *engine;
    ColiPrefixDiskCache disk;
} V4PrefixDiskNamespace;

static V4PrefixDiskNamespace g_v4_disk[COLI_V4_PREFIX_DISK_MAX_ENGINES];
static pthread_mutex_t g_v4_disk_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    uint32_t version;
    uint32_t layer_count;
} V4PrefixPayloadHeader;

int coli_v4_attention_snapshot_restore_fresh(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiV4AttentionSnapshot *snapshot,
    const ColiDeepSeekV4Config *config, int layer);

static int ascii_ieq(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++, cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return *a == 0 && *b == 0;
}

static int v4_compat_cache_disabled(void) {
    const char *value = getenv("V4_PREFIX_CACHE_MB");
    if (!value) return 0;
    if (!*value || ascii_ieq(value, "off")) return 1;
    char *end = NULL;
    long double mib = strtold(value, &end);
    if (end == value || !(mib > 0.0L)) return 1;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    return *end != 0;
}

static int v4_disk_policy_wanted(void) {
    if (v4_compat_cache_disabled() || !coli_prefix_disk_mode_allows_ssd())
        return 0;
    /* Cache-by-default is a serving policy. Explicit common policy may opt a
     * non-serve engine API user into persistence for tests/tools. */
    const char *mode = getenv("COLI_PREFIX_CACHE");
    if (mode && *mode) return 1;
    const char *serve = getenv("SERVE");
    return serve && serve[0] == '1';
}

static int v4_min_tokens(void) {
    const char *value = getenv("V4_PREFIX_CACHE_MIN_TOKENS");
    if (!value || !*value) value = getenv("COLI_PREFIX_CACHE_MIN_TOKENS");
    if (!value || !*value) return 256;
    char *end = NULL;
    long n = strtol(value, &end, 10);
    if (end == value || n < 1) return 256;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    if (*end) return 256;
    return n > INT_MAX ? INT_MAX : (int)n;
}

static int hash_file(uint64_t lanes[4], const char *directory,
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
        if (got) for (int lane = 0; lane < 4; lane++)
            lanes[lane] = coli_prefix_hash64(buffer, got, lanes[lane]);
        if (got < sizeof(buffer)) {
            int ok = !ferror(fp);
            fclose(fp);
            return ok;
        }
    }
}

/* Strict identity without scanning model payloads: source fingerprint + target
 * compiler/profile + every manifest record's physical format/CRC metadata +
 * the external config/tokenizer bytes that the V4 session actually consumes. */
static int v4_model_fingerprint(const ColiV4Engine *engine, uint8_t out[32]) {
    if (!engine || !out || !engine->coli_static || engine->target_index ||
        !engine->runtime.target_model_dir || !*engine->runtime.target_model_dir)
        return 0; /* package-only first; hybrid source+COLI fails closed */
    const ColiPackage *package = coli_executor_package(engine->coli_static);
    if (!package) return 0;
    const uint8_t *source = coli_package_source_fingerprint(package);
    if (!source) return 0;
    uint64_t h[4] = {
        UINT64_C(1469598103934665603),
        UINT64_C(1099511628211) ^ UINT64_C(0x9e3779b97f4a7c15),
        UINT64_C(0x6a09e667f3bcc909),
        UINT64_C(0xbb67ae8584caa73b),
    };
    const char *profile = coli_package_profile(package);
    const char *compiler = coli_package_compiler(package);
    for (int lane = 0; lane < 4; lane++) {
        h[lane] = coli_prefix_hash64(source, 32, h[lane]);
        if (profile) h[lane] = coli_prefix_hash64(profile, strlen(profile) + 1, h[lane]);
        if (compiler) h[lane] = coli_prefix_hash64(compiler, strlen(compiler) + 1, h[lane]);
    }
    size_t records = coli_package_record_count(package);
    for (size_t i = 0; i < records; i++) {
        const ColiRecordInfo *r = coli_package_record_at(package, i);
        if (!r) return 0;
        uint64_t meta[] = {
            r->record_id,
            ((uint64_t)r->kind << 48) | ((uint64_t)r->codec << 32) |
                ((uint64_t)r->math_format << 16) | r->scale_format,
            ((uint64_t)r->layout << 48) | ((uint64_t)r->flags << 32) | r->shard_id,
            ((uint64_t)(uint32_t)r->layer << 32) | (uint32_t)r->expert,
            r->stored_bytes, r->decoded_bytes,
            ((uint64_t)r->stored_crc32c << 32) | r->logical_crc32c,
        };
        for (int lane = 0; lane < 4; lane++) {
            h[lane] = coli_prefix_hash64(meta, sizeof(meta), h[lane]);
            if (r->name) h[lane] = coli_prefix_hash64(r->name, strlen(r->name) + 1, h[lane]);
        }
    }
    if (!hash_file(h, engine->runtime.target_model_dir, "config.json") ||
        !hash_file(h, engine->runtime.target_model_dir, "tokenizer.json"))
        return 0;
    memcpy(out, h, sizeof(h));
    return 1;
}

static uint64_t v4_layout_fingerprint(const ColiV4Engine *engine) {
    uint64_t h = UINT64_C(1469598103934665603);
    const char build[] = COLI_V4_GIT_SHA;
    h = coli_prefix_hash64(build, sizeof(build), h);
    uint64_t runtime[] = {
        COLI_V4_PREFIX_DISK_STATE_ABI,
        COLI_V4_PREFIX_DISK_PAYLOAD_VERSION,
        (uint64_t)(uint32_t)engine->runtime.context_tokens,
        (uint64_t)sizeof(float),
#ifdef COLI_METAL
        UINT64_C(1),
#else
        UINT64_C(0),
#endif
    };
    return coli_prefix_hash64(runtime, sizeof(runtime), h);
}

static ColiPrefixDiskCache *v4_disk_for_engine(ColiV4Engine *engine) {
    if (!engine) return NULL;
    ColiPrefixDiskCache *result = NULL;
    pthread_mutex_lock(&g_v4_disk_mutex);
    for (int i = 0; i < COLI_V4_PREFIX_DISK_MAX_ENGINES; i++)
        if (g_v4_disk[i].engine == engine) {
            result = &g_v4_disk[i].disk;
            break;
        }
    pthread_mutex_unlock(&g_v4_disk_mutex);
    return result;
}

void coli_v4_prefix_disk_register_engine(ColiV4Engine *engine) {
    if (!engine || !v4_disk_policy_wanted()) return;
    uint8_t fingerprint[COLI_PREFIX_DISK_FINGERPRINT_BYTES];
    char directory[COLI_PREFIX_DISK_PATH_MAX];
    if (!v4_model_fingerprint(engine, fingerprint) ||
        !coli_prefix_disk_default_directory(directory, sizeof(directory))) {
        if (getenv("V4_PREFIX_LOG"))
            fprintf(stderr,
                    "[PREFIX-SSD] V4 disabled: requires package-only COLI plus fingerprintable config/tokenizer\n");
        return;
    }
    ColiPrefixDiskCache initialized;
    size_t budget = coli_prefix_disk_budget_from_env(
        (size_t)COLI_V4_PREFIX_DISK_DEFAULT_GB * 1024u * 1024u * 1024u);
    if (!coli_prefix_disk_init(&initialized, directory, budget, v4_min_tokens(),
                               COLI_V4_PREFIX_DISK_ENGINE_ID,
                               COLI_V4_PREFIX_DISK_STATE_ABI,
                               fingerprint, v4_layout_fingerprint(engine),
                               getenv("V4_PREFIX_LOG") != NULL) ||
        !initialized.enabled)
        return;

    pthread_mutex_lock(&g_v4_disk_mutex);
    int slot = -1;
    for (int i = 0; i < COLI_V4_PREFIX_DISK_MAX_ENGINES; i++) {
        if (g_v4_disk[i].engine == engine) { slot = i; break; }
        if (slot < 0 && !g_v4_disk[i].engine) slot = i;
    }
    if (slot >= 0) {
        g_v4_disk[slot].engine = engine;
        g_v4_disk[slot].disk = initialized;
    }
    pthread_mutex_unlock(&g_v4_disk_mutex);
}

void coli_v4_prefix_disk_forget_engine(ColiV4Engine *engine) {
    if (!engine) return;
    pthread_mutex_lock(&g_v4_disk_mutex);
    for (int i = 0; i < COLI_V4_PREFIX_DISK_MAX_ENGINES; i++)
        if (g_v4_disk[i].engine == engine) {
            memset(&g_v4_disk[i], 0, sizeof(g_v4_disk[i]));
            break;
        }
    pthread_mutex_unlock(&g_v4_disk_mutex);
}

static size_t v4_payload_bytes(ColiV4AttentionSnapshot *const *attention,
                               int layer_count) {
    if (!attention || layer_count <= 0) return 0;
    size_t total = sizeof(V4PrefixPayloadHeader);
    for (int layer = 0; layer < layer_count; layer++) {
        size_t bytes = coli_v4_attention_snapshot_wire_bytes(attention[layer]);
        if (!bytes || bytes == SIZE_MAX || total > SIZE_MAX - sizeof(uint64_t) ||
            total + sizeof(uint64_t) > SIZE_MAX - bytes)
            return 0;
        total += sizeof(uint64_t) + bytes;
    }
    return total;
}

typedef struct {
    ColiV4AttentionSnapshot *const *attention;
    int layer_count;
} V4EmitContext;

typedef struct {
    ColiPrefixDiskSinkFn sink;
    void *sink_data;
} V4SinkAdapter;

static int v4_wire_sink(void *user_data, const void *data, size_t bytes) {
    V4SinkAdapter *adapter = (V4SinkAdapter *)user_data;
    return !adapter || !adapter->sink
        ? -1 : adapter->sink(adapter->sink_data, data, bytes);
}

static int v4_emit_payload(void *user_data, ColiPrefixDiskSinkFn sink,
                           void *sink_data) {
    V4EmitContext *ctx = (V4EmitContext *)user_data;
    if (!ctx || !sink || !ctx->attention || ctx->layer_count <= 0) return -1;
    V4PrefixPayloadHeader header = {
        COLI_V4_PREFIX_DISK_PAYLOAD_VERSION, (uint32_t)ctx->layer_count
    };
    if (sink(sink_data, &header, sizeof(header))) return -1;
    V4SinkAdapter adapter = { sink, sink_data };
    for (int layer = 0; layer < ctx->layer_count; layer++) {
        size_t size = coli_v4_attention_snapshot_wire_bytes(ctx->attention[layer]);
        if (!size || size == SIZE_MAX || size > UINT64_MAX) return -1;
        uint64_t wire_bytes = (uint64_t)size;
        if (sink(sink_data, &wire_bytes, sizeof(wire_bytes)) ||
            coli_v4_attention_snapshot_wire_emit(ctx->attention[layer],
                                                 v4_wire_sink, &adapter))
            return -1;
    }
    return 0;
}

typedef struct {
    ColiPrefixDiskCache *disk;
} V4PublishContext;

static int v4_publish_entry(const int *tokens, int token_count,
                            ColiV4AttentionSnapshot *const *attention,
                            int layer_count, void *user_data) {
    V4PublishContext *ctx = (V4PublishContext *)user_data;
    if (!ctx || !ctx->disk) return -1;
    size_t payload_bytes = v4_payload_bytes(attention, layer_count);
    if (!payload_bytes) return -1;
    V4EmitContext emit = { attention, layer_count };
    return coli_prefix_disk_store_stream(ctx->disk, tokens, token_count,
                                         payload_bytes, v4_emit_payload, &emit)
        ? 1 : -1;
}

void coli_v4_prefix_disk_publish_session(ColiV4Session *session) {
    if (!session || !session->engine) return;
    ColiPrefixDiskCache *disk = v4_disk_for_engine(session->engine);
    if (!disk || !disk->enabled) return;
    V4PublishContext context = { disk };
    (void)coli_v4_prefix_cache_visit_exact(session, v4_publish_entry, &context);
}

static void reset_failed_restore(ColiV4Session *session) {
    if (!session) return;
    if (session->attention)
        for (int layer = 0; layer < session->config.num_hidden_layers; layer++)
            coli_v4_window_attention_reset(session->attention[layer]);
    kv_prefix_clear(&session->fed);
}

int coli_v4_prefix_disk_restore(ColiV4Session *session,
                                const int *prompt_ids,
                                int prompt_tokens) {
    if (!session || !session->engine || !prompt_ids || prompt_tokens <= 1 ||
        !session->attention || !session->fed.fed)
        return 0;
    ColiPrefixDiskCache *disk = v4_disk_for_engine(session->engine);
    if (!disk || !disk->enabled) return 0;

    ColiPrefixDiskHit hit;
    int matched = coli_prefix_disk_restore_longest(disk, prompt_ids,
                                                   prompt_tokens, &hit);
    if (matched <= 0) return 0;
    int ok = matched <= session->fed.cap &&
             hit.payload_bytes >= sizeof(V4PrefixPayloadHeader);
    size_t offset = 0;
    V4PrefixPayloadHeader header = {0};
    if (ok) {
        memcpy(&header, hit.payload, sizeof(header));
        offset = sizeof(header);
        ok = header.version == COLI_V4_PREFIX_DISK_PAYLOAD_VERSION &&
             header.layer_count == (uint32_t)session->config.num_hidden_layers;
    }
    for (int layer = 0; ok && layer < session->config.num_hidden_layers; layer++) {
        if (offset > hit.payload_bytes ||
            hit.payload_bytes - offset < sizeof(uint64_t)) {
            ok = 0; break;
        }
        uint64_t raw_bytes = 0;
        memcpy(&raw_bytes, hit.payload + offset, sizeof(raw_bytes));
        offset += sizeof(raw_bytes);
        if (raw_bytes > SIZE_MAX || (size_t)raw_bytes > hit.payload_bytes - offset) {
            ok = 0; break;
        }
        size_t layer_bytes = (size_t)raw_bytes, consumed = 0;
        ColiV4AttentionSnapshot *snapshot = NULL;
        if (coli_v4_attention_snapshot_wire_read(&snapshot, hit.payload + offset,
                                                 layer_bytes, &consumed) ||
            consumed != layer_bytes ||
            coli_v4_attention_snapshot_restore_fresh(
                session->attention[layer], snapshot, &session->config, layer))
            ok = 0;
        coli_v4_attention_snapshot_destroy(snapshot);
        offset += layer_bytes;
    }
    if (ok && offset != hit.payload_bytes) ok = 0;
    if (ok) {
        memcpy(session->fed.fed, hit.tokens,
               (size_t)matched * sizeof(*session->fed.fed));
        session->fed.len = matched;
        session->fed.tainted = 0;
    } else {
        reset_failed_restore(session);
    }
    coli_prefix_disk_hit_free(&hit);
    if (!ok) return 0;

    /* Promote only after the disk payload is gone, avoiding payload + full RAM
     * snapshot + per-layer decode allocations at the same instant. */
    coli_v4_prefix_cache_store(session);
    if (getenv("V4_PREFIX_LOG"))
        fprintf(stderr, "[PREFIX-SSD] V4 hit matched=%d prompt=%d\n",
                matched, prompt_tokens);
    return matched;
}

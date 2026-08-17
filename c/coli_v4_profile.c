#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "deepseek_v4_internal.h"
#include "profile.h"

#include <string.h>

/* Compatibility surface for the existing V4 instrumentation sites. The timing
 * engine itself is generic (profile.c); V4 only supplies its phase taxonomy,
 * expert-store counters and legacy v4_phases output prefix. */
static const ColiProfilePhaseDef v4_phases[COLI_V4_PROF_COUNT] = {
    [COLI_V4_PROF_DENSE_READ]         = {"dense_read", COLI_PROFILE_ACCOUNTED | COLI_PROFILE_IO_WAIT},
    [COLI_V4_PROF_HC_PRE_ATTN]        = {"hc_pre_attn", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_ATTN_PROJ]          = {"attn_proj", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_COMPRESSOR]         = {"compressor", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_INDEXER]            = {"indexer", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_SPARSE_ATTN]        = {"sparse_attn", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_HC_POST_ATTN]       = {"hc_post_attn", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_HC_PRE_FFN]         = {"hc_pre_ffn", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_ROUTER]             = {"router", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_EXPERT_LOOKUP]      = {"expert_lookup", 0},
    [COLI_V4_PROF_EXPERT_READ_WORK]   = {"expert_read_work", 0},
    [COLI_V4_PROF_EXPERT_PACK_WORK]   = {"expert_pack_work", 0},
    [COLI_V4_PROF_EXPERT_LOADER_WAIT] = {"expert_wait", COLI_PROFILE_ACCOUNTED | COLI_PROFILE_IO_WAIT},
    [COLI_V4_PROF_EXPERT_COMPUTE]     = {"expert_compute", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_SHARED_EXPERT]      = {"shared_expert", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_HC_POST_FFN]        = {"hc_post_ffn", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_HEAD_READ]          = {"head_read", COLI_PROFILE_ACCOUNTED | COLI_PROFILE_IO_WAIT},
    [COLI_V4_PROF_HEAD_COMPUTE]       = {"head_compute", COLI_PROFILE_ACCOUNTED},
    [COLI_V4_PROF_METAL_ENCODE]       = {"metal_encode", 0},
    [COLI_V4_PROF_METAL_SUBMIT]       = {"metal_submit", 0},
    [COLI_V4_PROF_METAL_WAIT]         = {"metal_wait", 0},
    [COLI_V4_PROF_METAL_KERNEL]       = {"metal_kernel", 0},
};

enum {
    V4_COUNTER_DENSE_READ_BYTES = 0,
    V4_COUNTER_EXPERT_READ_BYTES,
    V4_COUNTER_HEAD_READ_BYTES,
    V4_COUNTER_EXPERT_REQUESTS,
    V4_COUNTER_EXPERT_HITS,
    V4_COUNTER_EXPERT_MISSES,
    V4_COUNTER_EXPERT_RESIDENT_BYTES,
    V4_COUNTER_EXPERT_CAPACITY_BYTES,
    V4_COUNTER_COUNT
};

static const ColiProfileCounterDef v4_counters[V4_COUNTER_COUNT] = {
    [V4_COUNTER_DENSE_READ_BYTES]       = {"dense_read_bytes"},
    [V4_COUNTER_EXPERT_READ_BYTES]      = {"expert_read_bytes"},
    [V4_COUNTER_HEAD_READ_BYTES]        = {"head_read_bytes"},
    [V4_COUNTER_EXPERT_REQUESTS]        = {"expert_requests"},
    [V4_COUNTER_EXPERT_HITS]            = {"expert_hits"},
    [V4_COUNTER_EXPERT_MISSES]          = {"expert_misses"},
    [V4_COUNTER_EXPERT_RESIDENT_BYTES]  = {"expert_resident_bytes"},
    [V4_COUNTER_EXPERT_CAPACITY_BYTES]  = {"expert_capacity_bytes"},
};

static ColiProfile g_v4_profile;
int g_coli_v4_profile_on = 0;

static void sync_metal_profile(void) {
#ifdef COLI_METAL
    if (!g_coli_v4_profile_on) return;
    uint64_t encode = 0, submit = 0, wait = 0, kernel = 0;
    coli_metal_profile_get(&encode, &submit, &wait, &kernel);
    coli_profile_phase_set(&g_v4_profile, COLI_V4_PROF_METAL_ENCODE, encode);
    coli_profile_phase_set(&g_v4_profile, COLI_V4_PROF_METAL_SUBMIT, submit);
    coli_profile_phase_set(&g_v4_profile, COLI_V4_PROF_METAL_WAIT, wait);
    coli_profile_phase_set(&g_v4_profile, COLI_V4_PROF_METAL_KERNEL, kernel);
#endif
}

static ColiV4Profile v4_snapshot(ColiProfileSnapshot snapshot) {
    ColiV4Profile out;
    memset(&out, 0, sizeof(out));
    for (int i = 0; i < COLI_V4_PROF_COUNT; i++) out.ns[i] = snapshot.phase_ns[i];
    out.dense_read_bytes = snapshot.counters[V4_COUNTER_DENSE_READ_BYTES];
    out.expert_read_bytes = snapshot.counters[V4_COUNTER_EXPERT_READ_BYTES];
    out.head_read_bytes = snapshot.counters[V4_COUNTER_HEAD_READ_BYTES];
    out.wall_ns = snapshot.wall_ns;
    out.tokens = snapshot.tokens;
    return out;
}

void coli_v4_profile_reset(void) {
    static const ColiProfileConfig config = {
        .engine = "deepseek_v4",
        .env_name = "V4_PROFILE",
        .line_prefix = "v4_phases",
        .include_engine = 0,
        .phases = v4_phases,
        .phase_count = COLI_V4_PROF_COUNT,
        .counters = v4_counters,
        .counter_count = V4_COUNTER_COUNT,
    };
    coli_profile_reset(&g_v4_profile, &config);
    g_coli_v4_profile_on = coli_profile_enabled(&g_v4_profile);
#ifdef COLI_METAL
    coli_metal_profile_set_on(g_coli_v4_profile_on);
    if (g_coli_v4_profile_on) coli_metal_profile_reset();
#endif
    /* Private zero-time baseline for startup. Existing public slots stay 0..4. */
    if (g_coli_v4_profile_on) coli_profile_mark(&g_v4_profile, 7);
}

void coli_v4_profile_add(int kind, uint64_t ns) {
    if (kind < 0 || kind >= COLI_V4_PROF_COUNT) return;
    coli_profile_add(&g_v4_profile, (size_t)kind, ns);
}

void coli_v4_profile_add_bytes(ColiV4ProfileKind kind, uint64_t bytes) {
    if (kind == COLI_V4_PROF_DENSE_READ)
        coli_profile_counter_add(&g_v4_profile, V4_COUNTER_DENSE_READ_BYTES, bytes);
    else if (kind == COLI_V4_PROF_HEAD_READ)
        coli_profile_counter_add(&g_v4_profile, V4_COUNTER_HEAD_READ_BYTES, bytes);
}

ColiV4Profile coli_v4_profile_get(void) {
    sync_metal_profile();
    return v4_snapshot(coli_profile_get(&g_v4_profile));
}

void coli_v4_profile_mark(int slot) {
    if (slot < 0 || slot >= 5) return;
    sync_metal_profile();
    coli_profile_mark(&g_v4_profile, (size_t)slot);
}

void coli_v4_profile_tokens(int slot, uint64_t tokens) {
    if (slot < 0 || slot >= 5) return;
    coli_profile_mark_tokens(&g_v4_profile, (size_t)slot, tokens);
}

ColiV4Profile coli_v4_profile_mark_get(int slot) {
    if (slot < 0 || slot >= 5) {
        ColiV4Profile empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }
    return v4_snapshot(coli_profile_mark_get(&g_v4_profile, (size_t)slot));
}

void coli_v4_profile_mark_stats(int slot, ColiExpertStore *experts) {
    if (slot < 0 || slot >= 5 || !g_coli_v4_profile_on) return;
    ColiExpertStoreStats stats;
    memset(&stats, 0, sizeof(stats));
    if (experts && experts->ops && experts->ops->stats)
        experts->ops->stats(experts, &stats);
    coli_profile_mark_counter_set(&g_v4_profile, (size_t)slot,
                                  V4_COUNTER_EXPERT_READ_BYTES, stats.bytes_read);
    coli_profile_mark_counter_set(&g_v4_profile, (size_t)slot,
                                  V4_COUNTER_EXPERT_REQUESTS, stats.requests);
    coli_profile_mark_counter_set(&g_v4_profile, (size_t)slot,
                                  V4_COUNTER_EXPERT_HITS, stats.hits);
    coli_profile_mark_counter_set(&g_v4_profile, (size_t)slot,
                                  V4_COUNTER_EXPERT_MISSES, stats.misses);
    coli_profile_mark_counter_set(&g_v4_profile, (size_t)slot,
                                  V4_COUNTER_EXPERT_RESIDENT_BYTES, stats.resident_bytes);
    coli_profile_mark_counter_set(&g_v4_profile, (size_t)slot,
                                  V4_COUNTER_EXPERT_CAPACITY_BYTES, stats.capacity_bytes);
}

void coli_v4_profile_emit(FILE *stream) {
    if (!stream || !g_coli_v4_profile_on) return;
    coli_profile_emit_scope(stream, &g_v4_profile, "startup", 7, 0);
    coli_profile_emit_scope(stream, &g_v4_profile, "run", 1, 4);
    coli_profile_emit_scope(stream, &g_v4_profile, "prompt", 2, 3);
    coli_profile_emit_scope(stream, &g_v4_profile, "decode", 3, 4);
}

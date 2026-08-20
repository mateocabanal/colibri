#include "../deepseek_v4.h"
#include "../backend_metal.h"
#include "../backend_metal_tile.h"
#include "../mxfp4_apple8_tile_cache.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *trace;
    int count;
} TokenSink;

typedef struct {
    uint64_t ok, fallback, experts;
    double setup, gpu, scatter, kernel;
} RowStats;

static int sink_token(void *opaque, int token, float logit,
                      int position, int ordinal) {
    (void)logit;
    (void)position;
    (void)ordinal;
    TokenSink *sink = (TokenSink *)opaque;
    if (!sink) return 0;
    sink->count++;
    if (sink->trace) fprintf(sink->trace, "%d\n", token);
    return 0;
}

static void row_stats(RowStats *s) {
    memset(s, 0, sizeof(*s));
    coli_metal_moe_counts(&s->ok, &s->fallback, &s->experts);
    coli_metal_moe_times(&s->setup, &s->gpu, &s->scatter);
    s->kernel = coli_metal_moe_kernel_time();
}

static uint64_t du64(uint64_t a, uint64_t b) { return a >= b ? a - b : 0; }
static double dd(double a, double b) { return a >= b ? a - b : 0.0; }

static int positive(const char *text, int *out) {
    char *end = NULL;
    errno = 0;
    long v = strtol(text, &end, 10);
    if (errno || !end || *end || v < 1 || v > INT32_MAX) return -1;
    *out = (int)v;
    return 0;
}

static void usage(const char *program) {
    fprintf(stderr,
        "usage: %s MODEL PROMPT [--warm N] [--tokens N] [--memory-gb N] [--trace PATH]\n",
        program);
}

static int run_generation(ColiV4Engine *engine, const char *prompt, int tokens,
                          FILE *trace, ColiV4SessionGenerateStats *stats,
                          char *error, size_t error_size) {
    ColiV4Session *session = NULL;
    ColiV4SessionCreateOptions create = {
        .max_prompt_tokens = 512,
        .max_new_tokens_cap = tokens,
    };
    if (coli_v4_session_create(&session, engine, &create, error, error_size))
        return -1;
    TokenSink sink = {.trace = trace, .count = 0};
    ColiV4SessionGenerateOptions generate = {
        .max_new_tokens = tokens,
        .stop_at_sentence = 0,
        .no_dspark = 1,
    };
    int rc = coli_v4_session_generate(
        session, prompt, strlen(prompt), &generate,
        sink_token, &sink, stats, error, error_size);
    coli_v4_session_destroy(session);
    return rc;
}

static void print_derived_delta(
        const char *tag,
        const ColiMxfp4Apple8DerivedCacheStats *before,
        const ColiMxfp4Apple8DerivedCacheStats *after) {
    fprintf(stdout,
        "WIRE_TILE_DERIVED_%s enabled=%d lookup=%llu hit=%llu miss=%llu stale=%llu corrupt=%llu "
        "read_mib=%.3f read_ms=%.3f write_mib=%.3f write_ms=%.3f write_dropped=%llu "
        "prepare_avoided_ms=%.3f cold_prepares=%llu cold_prepare_ms=%.3f "
        "installs=%llu install_failures=%llu\n",
        tag,
        coli_mxfp4_apple8_derived_cache_enabled(),
        (unsigned long long)du64(after->lookup, before->lookup),
        (unsigned long long)du64(after->hit, before->hit),
        (unsigned long long)du64(after->miss, before->miss),
        (unsigned long long)du64(after->stale, before->stale),
        (unsigned long long)du64(after->corrupt, before->corrupt),
        (double)du64(after->read_bytes, before->read_bytes) / (1024.0 * 1024.0),
        (double)du64(after->read_ns, before->read_ns) / 1.0e6,
        (double)du64(after->write_bytes, before->write_bytes) / (1024.0 * 1024.0),
        (double)du64(after->write_ns, before->write_ns) / 1.0e6,
        (unsigned long long)du64(after->write_dropped, before->write_dropped),
        (double)du64(after->prepare_ns_avoided,
                     before->prepare_ns_avoided) / 1.0e6,
        (unsigned long long)du64(after->cold_prepares, before->cold_prepares),
        (double)du64(after->cold_prepare_ns, before->cold_prepare_ns) / 1.0e6,
        (unsigned long long)du64(after->installs, before->installs),
        (unsigned long long)du64(after->install_failures,
                                 before->install_failures));
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 2; }
    const char *model = argv[1];
    const char *prompt = argv[2];
    const char *trace_path = NULL;
    int warm = 12, tokens = 32, memory_gb = 10;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--warm") && i + 1 < argc) {
            if (positive(argv[++i], &warm)) { usage(argv[0]); return 2; }
        } else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) {
            if (positive(argv[++i], &tokens)) { usage(argv[0]); return 2; }
        } else if (!strcmp(argv[i], "--memory-gb") && i + 1 < argc) {
            if (positive(argv[++i], &memory_gb)) { usage(argv[0]); return 2; }
        } else if (!strcmp(argv[i], "--trace") && i + 1 < argc) {
            trace_path = argv[++i];
        } else {
            usage(argv[0]); return 2;
        }
    }

    char error[512] = {0};
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions open = {
        .target_model_dir = model,
        .memory_limit_bytes = (uint64_t)memory_gb * UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024),
        .context_tokens = 4096,
        .pin_slots_per_layer = -1,
        .repin_interval = 0,
        .no_dspark = 1,
        .coli_model_dir = getenv("COLI_MODEL"),
    };
    if (coli_v4_engine_open(&engine, &open, error, sizeof(error))) {
        fprintf(stderr, "WIRE_TILE open failed: %s\n", error);
        return 1;
    }

    ColiMxfp4Apple8DerivedCacheStats derived_before_warm;
    ColiMxfp4Apple8DerivedCacheStats derived_after_warm;
    ColiMxfp4Apple8DerivedCacheStats derived_after_timed;
    coli_mxfp4_apple8_derived_cache_stats(&derived_before_warm);

    ColiV4SessionGenerateStats warm_stats;
    memset(&warm_stats, 0, sizeof(warm_stats));
    if (run_generation(engine, prompt, warm, NULL, &warm_stats,
                       error, sizeof(error))) {
        fprintf(stderr, "WIRE_TILE warm failed: %s\n", error);
        coli_v4_engine_destroy(engine);
        return 1;
    }
    coli_mxfp4_apple8_derived_cache_stats(&derived_after_warm);
    double cold_tok_s = warm_stats.decode_sec > 0.0
        ? (double)warm_stats.generated_tokens / warm_stats.decode_sec : 0.0;
    fprintf(stdout,
        "WIRE_TILE_COLD_RESULT mode=%s generated=%d decode_sec=%.6f tok_s=%.6f ttft=%.6f eos=%d\n",
        coli_metal_tile_enabled() ? "tile" : "row",
        warm_stats.generated_tokens, warm_stats.decode_sec, cold_tok_s,
        warm_stats.time_to_first_token_sec, warm_stats.eos_stopped);
    print_derived_delta("COLD", &derived_before_warm, &derived_after_warm);
    fflush(stdout);

    RowStats row_before, row_after;
    ColiMetalTileStats tile_before, tile_after;
    row_stats(&row_before);
    coli_metal_tile_stats(&tile_before);
    coli_metal_profile_reset();
    coli_metal_profile_set_on(1);

    FILE *trace = NULL;
    if (trace_path) {
        trace = fopen(trace_path, "w");
        if (!trace) {
            fprintf(stderr, "WIRE_TILE cannot open trace: %s\n", trace_path);
            coli_metal_profile_set_on(0);
            coli_v4_engine_destroy(engine);
            return 1;
        }
    }

    ColiV4SessionGenerateStats stats;
    memset(&stats, 0, sizeof(stats));
    int rc = run_generation(engine, prompt, tokens, trace, &stats,
                            error, sizeof(error));
    if (trace) {
        if (fflush(trace) || fclose(trace)) rc = -1;
    }

    uint64_t profile_encode_ns = 0, profile_submit_ns = 0;
    uint64_t profile_wait_ns = 0, profile_kernel_ns = 0;
    coli_metal_profile_get(&profile_encode_ns, &profile_submit_ns,
                           &profile_wait_ns, &profile_kernel_ns);
    coli_metal_profile_set_on(0);

    if (rc) {
        fprintf(stderr, "WIRE_TILE timed failed: %s\n", error[0] ? error : "trace write");
        coli_v4_engine_destroy(engine);
        return 1;
    }

    row_stats(&row_after);
    coli_metal_tile_stats(&tile_after);
    coli_mxfp4_apple8_derived_cache_stats(&derived_after_timed);
    const char *mode = coli_metal_tile_enabled() ? "tile" : "row";
    double tok_s = stats.decode_sec > 0.0
        ? (double)stats.generated_tokens / stats.decode_sec : 0.0;

    double row_setup = dd(row_after.setup, row_before.setup);
    double row_gpu = dd(row_after.gpu, row_before.gpu);
    double row_scatter = dd(row_after.scatter, row_before.scatter);
    double row_kernel = dd(row_after.kernel, row_before.kernel);
    uint64_t tile_repack_ns = du64(tile_after.repack_ns, tile_before.repack_ns);
    uint64_t tile_cached_install_ns =
        du64(tile_after.cached_install_ns, tile_before.cached_install_ns);
    uint64_t tile_wall_ns = du64(tile_after.wall_ns, tile_before.wall_ns);
    uint64_t tile_kernel_ns = du64(tile_after.kernel_ns, tile_before.kernel_ns);
    uint64_t tile_scatter_ns = du64(tile_after.scatter_ns, tile_before.scatter_ns);
    uint64_t row_mxfp4_calls = du64(tile_after.row_mxfp4_calls,
                                    tile_before.row_mxfp4_calls);
    uint64_t row_mxfp4_wall_ns = du64(tile_after.row_mxfp4_wall_ns,
                                      tile_before.row_mxfp4_wall_ns);
    double expert_seconds =
        (double)row_mxfp4_wall_ns / 1.0e9 +
        (double)tile_repack_ns / 1.0e9 +
        (double)tile_cached_install_ns / 1.0e9 +
        (double)tile_wall_ns / 1.0e9;
    double expert_pct = stats.decode_sec > 0.0
        ? 100.0 * expert_seconds / stats.decode_sec : 0.0;

    fprintf(stdout,
        "WIRE_TILE_RESULT mode=%s generated=%d decode_sec=%.6f tok_s=%.6f ttft=%.6f eos=%d\n",
        mode, stats.generated_tokens, stats.decode_sec, tok_s,
        stats.time_to_first_token_sec, stats.eos_stopped);
    fprintf(stdout,
        "WIRE_TILE_ROW blocks=%llu fallback=%llu experts=%llu setup_ms=%.3f gpu_ms=%.3f kernel_ms=%.3f scatter_ms=%.3f\n",
        (unsigned long long)du64(row_after.ok, row_before.ok),
        (unsigned long long)du64(row_after.fallback, row_before.fallback),
        (unsigned long long)du64(row_after.experts, row_before.experts),
        row_setup * 1.0e3, row_gpu * 1.0e3, row_kernel * 1.0e3,
        row_scatter * 1.0e3);
    fprintf(stdout,
        "WIRE_TILE_ROW_MXFP4 calls=%llu wall_ms=%.3f\n",
        (unsigned long long)row_mxfp4_calls,
        (double)row_mxfp4_wall_ns / 1.0e6);
    fprintf(stdout,
        "WIRE_TILE_PROFILE encode_ms=%.3f submit_ms=%.3f wait_ms=%.3f kernel_ms=%.3f\n",
        (double)profile_encode_ns / 1.0e6,
        (double)profile_submit_ns / 1.0e6,
        (double)profile_wait_ns / 1.0e6,
        (double)profile_kernel_ns / 1.0e6);
    fprintf(stdout,
        "WIRE_TILE_APPLE8 repacks=%llu repack_mib=%.3f repack_ms=%.3f "
        "cached_installs=%llu cached_install_mib=%.3f cached_install_ms=%.3f "
        "single=%llu blocks=%llu fallback=%llu experts=%llu wall_ms=%.3f kernel_ms=%.3f scatter_ms=%.3f\n",
        (unsigned long long)du64(tile_after.repack_count, tile_before.repack_count),
        (double)du64(tile_after.repack_bytes, tile_before.repack_bytes) / (1024.0 * 1024.0),
        (double)tile_repack_ns / 1.0e6,
        (unsigned long long)du64(tile_after.cached_install_count,
                                tile_before.cached_install_count),
        (double)du64(tile_after.cached_install_bytes,
                     tile_before.cached_install_bytes) / (1024.0 * 1024.0),
        (double)tile_cached_install_ns / 1.0e6,
        (unsigned long long)du64(tile_after.single_calls, tile_before.single_calls),
        (unsigned long long)du64(tile_after.moe_calls, tile_before.moe_calls),
        (unsigned long long)du64(tile_after.fallback_calls, tile_before.fallback_calls),
        (unsigned long long)du64(tile_after.experts, tile_before.experts),
        (double)tile_wall_ns / 1.0e6,
        (double)tile_kernel_ns / 1.0e6,
        (double)tile_scatter_ns / 1.0e6);
    print_derived_delta("TIMED", &derived_after_warm, &derived_after_timed);
    fprintf(stdout,
        "WIRE_TILE_EXPERT_SHARE mode=%s measured_expert_sec=%.6f decode_pct=%.3f trace=%s\n",
        mode, expert_seconds, expert_pct, trace_path ? trace_path : "none");

    coli_v4_engine_destroy(engine);
    return 0;
}

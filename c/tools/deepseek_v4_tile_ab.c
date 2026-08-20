#include "../deepseek_v4.h"
#include "../backend_metal.h"
#include "../backend_metal_tile.h"

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
    TokenSink *sink = (TokenSink *)opaque;
    if (!sink) return 0;
    sink->count++;
    if (sink->trace) fprintf(sink->trace, "%d\n", token);
    (void)ordinal;
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

    ColiV4SessionGenerateStats warm_stats;
    memset(&warm_stats, 0, sizeof(warm_stats));
    if (run_generation(engine, prompt, warm, NULL, &warm_stats,
                       error, sizeof(error))) {
        fprintf(stderr, "WIRE_TILE warm failed: %s\n", error);
        coli_v4_engine_destroy(engine);
        return 1;
    }
    fprintf(stdout,
        "WIRE_TILE_WARM mode=%s generated=%d decode_sec=%.6f\n",
        coli_metal_tile_enabled() ? "tile" : "row",
        warm_stats.generated_tokens, warm_stats.decode_sec);
    fflush(stdout);

    RowStats row_before, row_after;
    ColiMetalTileStats tile_before, tile_after;
    row_stats(&row_before);
    coli_metal_tile_stats(&tile_before);

    FILE *trace = NULL;
    if (trace_path) {
        trace = fopen(trace_path, "w");
        if (!trace) {
            fprintf(stderr, "WIRE_TILE cannot open trace: %s\n", trace_path);
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
    if (rc) {
        fprintf(stderr, "WIRE_TILE timed failed: %s\n", error[0] ? error : "trace write");
        coli_v4_engine_destroy(engine);
        return 1;
    }

    row_stats(&row_after);
    coli_metal_tile_stats(&tile_after);
    const char *mode = coli_metal_tile_enabled() ? "tile" : "row";
    double tok_s = stats.decode_sec > 0.0
        ? (double)stats.generated_tokens / stats.decode_sec : 0.0;

    double row_setup = dd(row_after.setup, row_before.setup);
    double row_gpu = dd(row_after.gpu, row_before.gpu);
    double row_scatter = dd(row_after.scatter, row_before.scatter);
    double row_kernel = dd(row_after.kernel, row_before.kernel);
    uint64_t tile_repack_ns = du64(tile_after.repack_ns, tile_before.repack_ns);
    uint64_t tile_wall_ns = du64(tile_after.wall_ns, tile_before.wall_ns);
    uint64_t tile_kernel_ns = du64(tile_after.kernel_ns, tile_before.kernel_ns);
    uint64_t tile_scatter_ns = du64(tile_after.scatter_ns, tile_before.scatter_ns);
    double expert_seconds = row_setup + row_gpu + row_scatter +
        (double)tile_repack_ns / 1.0e9 + (double)tile_wall_ns / 1.0e9;
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
        "WIRE_TILE_APPLE8 repacks=%llu repack_mib=%.3f repack_ms=%.3f single=%llu blocks=%llu fallback=%llu experts=%llu wall_ms=%.3f kernel_ms=%.3f scatter_ms=%.3f\n",
        (unsigned long long)du64(tile_after.repack_count, tile_before.repack_count),
        (double)du64(tile_after.repack_bytes, tile_before.repack_bytes) / (1024.0 * 1024.0),
        (double)tile_repack_ns / 1.0e6,
        (unsigned long long)du64(tile_after.single_calls, tile_before.single_calls),
        (unsigned long long)du64(tile_after.moe_calls, tile_before.moe_calls),
        (unsigned long long)du64(tile_after.fallback_calls, tile_before.fallback_calls),
        (unsigned long long)du64(tile_after.experts, tile_before.experts),
        (double)tile_wall_ns / 1.0e6,
        (double)tile_kernel_ns / 1.0e6,
        (double)tile_scatter_ns / 1.0e6);
    fprintf(stdout,
        "WIRE_TILE_EXPERT_SHARE mode=%s measured_expert_sec=%.6f decode_pct=%.3f trace=%s\n",
        mode, expert_seconds, expert_pct, trace_path ? trace_path : "none");

    coli_v4_engine_destroy(engine);
    return 0;
}

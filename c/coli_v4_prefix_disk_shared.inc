/* Compile the shared persistent-store helpers and the V4 SSD adapter exactly
 * once in this composition TU. Both the process-RAM cache and the shared disk
 * header define a private `ascii_ieq`, while coli_v4_prefix_disk.c defines its
 * own helper with the same spelling. Give each included implementation a
 * distinct preprocessor namespace instead of collapsing both helpers onto one
 * symbol. The prefix_cache_disk.h include guard then keeps the V4 adapter's
 * own include from expanding the shared implementation a second time. */
#define ascii_ieq coli_v4_prefix_store_ascii_ieq
#include "prefix_cache_disk.h"
#undef ascii_ieq

#define ascii_ieq coli_v4_disk_ascii_ieq
#define coli_v4_prefix_disk_publish_session \
    coli_v4_prefix_disk_publish_session_from_ram
#include "coli_v4_prefix_disk.c"
#undef coli_v4_prefix_disk_publish_session
#undef ascii_ieq

static int v4_prefix_has_exact_ram_entry(
    const int *tokens, int token_count,
    ColiV4AttentionSnapshot *const *attention, int layer_count,
    void *user_data) {
    (void)tokens;
    (void)token_count;
    (void)attention;
    (void)layer_count;
    (void)user_data;
    return 1;
}

typedef struct {
    ColiV4Session *session;
    int layer_count;
    size_t layer_bytes[COLI_V4_MAX_LAYERS];
} V4LivePrefixEmitContext;

/* Size the persistent wire representation without retaining a second complete
 * prefix snapshot. Each temporary layer snapshot is destroyed before the next
 * one is created. The emitter below repeats the capture while inference is
 * still stopped at the same canonical end-of-prefill boundary. */
static size_t v4_live_prefix_payload_layout(
    ColiV4Session *session, V4LivePrefixEmitContext *ctx) {
    if (!session || !ctx || !session->attention) return 0;
    int layers = session->config.num_hidden_layers;
    if (layers <= 0 || layers > COLI_V4_MAX_LAYERS) return 0;

    memset(ctx, 0, sizeof(*ctx));
    ctx->session = session;
    ctx->layer_count = layers;
    size_t total = sizeof(V4PrefixPayloadHeader);
    for (int layer = 0; layer < layers; layer++) {
        ColiV4AttentionSnapshot *snapshot = NULL;
        if (coli_v4_attention_snapshot_create(session->attention[layer], &snapshot))
            return 0;
        size_t bytes = coli_v4_attention_snapshot_wire_bytes(snapshot);
        coli_v4_attention_snapshot_destroy(snapshot);
        if (!bytes || bytes == SIZE_MAX ||
            total > SIZE_MAX - sizeof(uint64_t) ||
            total + sizeof(uint64_t) > SIZE_MAX - bytes)
            return 0;
        ctx->layer_bytes[layer] = bytes;
        total += sizeof(uint64_t) + bytes;
    }
    return total;
}

static int v4_emit_live_prefix_payload(void *user_data,
                                       ColiPrefixDiskSinkFn sink,
                                       void *sink_data) {
    V4LivePrefixEmitContext *ctx = (V4LivePrefixEmitContext *)user_data;
    if (!ctx || !ctx->session || !sink || ctx->layer_count <= 0) return -1;

    V4PrefixPayloadHeader header = {
        COLI_V4_PREFIX_DISK_PAYLOAD_VERSION, (uint32_t)ctx->layer_count
    };
    if (sink(sink_data, &header, sizeof(header))) return -1;

    V4SinkAdapter adapter = { sink, sink_data };
    for (int layer = 0; layer < ctx->layer_count; layer++) {
        ColiV4AttentionSnapshot *snapshot = NULL;
        if (coli_v4_attention_snapshot_create(
                ctx->session->attention[layer], &snapshot))
            return -1;
        size_t bytes = coli_v4_attention_snapshot_wire_bytes(snapshot);
        int failed = !bytes || bytes == SIZE_MAX ||
                     bytes != ctx->layer_bytes[layer] || bytes > UINT64_MAX;
        uint64_t wire_bytes = failed ? 0 : (uint64_t)bytes;
        if (!failed &&
            (sink(sink_data, &wire_bytes, sizeof(wire_bytes)) ||
             coli_v4_attention_snapshot_wire_emit(snapshot,
                                                  v4_wire_sink, &adapter)))
            failed = 1;
        coli_v4_attention_snapshot_destroy(snapshot);
        if (failed) return -1;
    }
    return 0;
}

void coli_v4_prefix_disk_publish_session(ColiV4Session *session) {
    coli_v4_prefix_disk_publish_session_from_ram(session);
}

void coli_v4_prefix_disk_publish_live_prefix(ColiV4Session *session) {
    if (!session || !session->engine || !session->prompt_ids ||
        session->prompt_count <= 0 || !session->fed.fed || session->fed.tainted ||
        session->fed.len != session->prompt_count)
        return;

    ColiPrefixDiskCache *disk = v4_disk_for_engine(session->engine);
    if (!disk || !disk->enabled) return;

    /* Normal auto mode already captured this exact immutable boundary in RAM;
     * leave its SSD write on the post-generation path so TTFT stays unchanged.
     * If RAM admission was disabled/rejected, persist the live boundary now,
     * before decode can mutate it. */
    if (coli_v4_prefix_cache_visit_exact(
            session, v4_prefix_has_exact_ram_entry, NULL))
        return;

    V4LivePrefixEmitContext ctx;
    size_t payload_bytes = v4_live_prefix_payload_layout(session, &ctx);
    if (!payload_bytes) return;
    if (!coli_prefix_disk_store_stream(
            disk, session->prompt_ids, session->prompt_count,
            payload_bytes, v4_emit_live_prefix_payload, &ctx) &&
        getenv("V4_PREFIX_LOG"))
        fprintf(stderr,
                "[PREFIX-SSD] V4 live end-of-prefill publication failed\n");
}

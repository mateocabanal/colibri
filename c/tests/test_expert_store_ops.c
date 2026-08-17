#include "../expert_store.h"
#include "../coli_v4_residency.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    int held;
    unsigned active_leases;
    unsigned context_calls;
    ColiExpertRequestContext last_context;
    ColiExpertStoreStats stats;
} MockState;

static int mock_lookup(ColiExpertStore *store, ColiExpertKey key,
                       ColiExpertView *view) {
    static const unsigned char weights[4] = {1, 2, 3, 4};
    MockState *state = (MockState *)store->state;
    if (state->held || !view) return -1;
    memset(view, 0, sizeof(*view));
    view->key = key;
    view->gate.format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->gate.scale_format = COLI_SCALE_UE8M0;
    view->gate.data = weights;
    view->gate.data_bytes = sizeof(weights);
    view->lease = state;
    state->held = 1;
    state->active_leases++;
    state->stats.requests++;
    state->stats.misses++;
    state->stats.bytes_read += sizeof(weights);
    return 0;
}

static int mock_lookup_context(ColiExpertStore *store, ColiExpertKey key,
                               const ColiExpertRequestContext *context,
                               ColiExpertView *view) {
    MockState *state = (MockState *)store->state;
    if (!context) return -1;
    state->context_calls++;
    state->last_context = *context;
    return mock_lookup(store, key, view);
}

static void mock_release(ColiExpertStore *store, ColiExpertView *view) {
    MockState *state = (MockState *)store->state;
    if (view && view->lease == state) {
        state->held = 0;
        if (state->active_leases) state->active_leases--;
        view->lease = NULL;
    }
}

static int mock_prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                         size_t count) {
    MockState *state = (MockState *)store->state;
    (void)keys;
    state->stats.prefetched += count;
    return (int)count;
}

static void mock_stats(const ColiExpertStore *store,
                       ColiExpertStoreStats *stats) {
    *stats = ((const MockState *)store->state)->stats;
}

static void mock_destroy(ColiExpertStore *store) {
    MockState *state = (MockState *)store->state;
    if (state->active_leases != 0)
        fprintf(stderr, "destroy with active leases=%u\n", state->active_leases);
}

static int view_is_cleared(const ColiExpertView *view) {
    static const ColiExpertView zero;
    return memcmp(view, &zero, sizeof(*view)) == 0;
}

static int test_residency_value_selector(void) {
    uint64_t slot_bytes = 0;
    if (coli_v4_expert_slot_bytes(1, &slot_bytes) || slot_bytes != 16384)
        return 1;
    if (coli_v4_expert_slot_bytes(16384, &slot_bytes) || slot_bytes != 16384)
        return 1;
    if (coli_v4_expert_slot_bytes(16385, &slot_bytes) || slot_bytes != 32768)
        return 1;
    if (coli_v4_expert_slot_bytes(0, &slot_bytes) == 0 ||
        coli_v4_expert_slot_bytes(UINT64_MAX, &slot_bytes) == 0)
        return 1;

    /* The ratio comparator must not rely on uint64 cross multiplication. */
    if (coli_v4_residency_ratio_compare(6, 3, 4, 2) != 0) return 1;
    if (coli_v4_residency_ratio_compare(
            UINT64_MAX, UINT64_MAX - 1,
            UINT64_MAX - 1, UINT64_MAX) <= 0)
        return 1;

    static const ColiV4ResidencyCandidate candidates[] = {
        /* Dense A: best exposed-time value. */
        {COLI_V4_RESIDENCY_DENSE_TENSOR, 10, 100, 500, 1000},
        /* Expert B: best raw byte value. */
        {COLI_V4_RESIDENCY_PERSISTENT_EXPERT, 20, 100, 700, 200},
        /* Dense C fits in the 50-byte tail after either 100-byte winner. */
        {COLI_V4_RESIDENCY_DENSE_TENSOR, 11, 50, 100, 100},
        /* Optional residency with no predicted benefit is never admitted. */
        {COLI_V4_RESIDENCY_OTHER, 1, 20, 0, 0},
    };
    unsigned char selected[sizeof(candidates) / sizeof(candidates[0])];
    ColiV4ResidencySelection plan;

    if (coli_v4_residency_select(
            candidates, 4, 100, COLI_V4_RESIDENCY_VALUE_BYTES,
            selected, &plan) != 0)
        return 1;
    if (!selected[1] || selected[0] || selected[2] || selected[3]) return 1;
    if (plan.selected_count != 1 || plan.selected_resident_bytes != 100 ||
        plan.expected_bytes_avoided != 700 ||
        plan.expected_exposed_ns_avoided != 200)
        return 1;

    if (coli_v4_residency_select(
            candidates, 4, 100, COLI_V4_RESIDENCY_VALUE_EXPOSED_NS,
            selected, &plan) != 0)
        return 1;
    if (!selected[0] || selected[1] || selected[2] || selected[3]) return 1;
    if (plan.selected_count != 1 || plan.selected_resident_bytes != 100 ||
        plan.expected_bytes_avoided != 500 ||
        plan.expected_exposed_ns_avoided != 1000)
        return 1;

    /* Greedy allocation may fill a smaller tail, but can never exceed budget. */
    if (coli_v4_residency_select(
            candidates, 4, 150, COLI_V4_RESIDENCY_VALUE_EXPOSED_NS,
            selected, &plan) != 0)
        return 1;
    if (!selected[0] || selected[1] || !selected[2] || selected[3]) return 1;
    if (plan.selected_count != 2 || plan.selected_resident_bytes != 150 ||
        plan.selected_resident_bytes > plan.budget_bytes)
        return 1;

    /* Equal ratios are deterministic: lower kind, then id, then position. */
    static const ColiV4ResidencyCandidate tied[] = {
        {COLI_V4_RESIDENCY_PERSISTENT_EXPERT, 1, 50, 100, 100},
        {COLI_V4_RESIDENCY_DENSE_TENSOR, 9, 50, 100, 100},
    };
    unsigned char tied_selected[2];
    if (coli_v4_residency_select(
            tied, 2, 50, COLI_V4_RESIDENCY_VALUE_BYTES,
            tied_selected, &plan) != 0)
        return 1;
    if (tied_selected[0] || !tied_selected[1]) return 1;

    /* Predicted-benefit totals saturate rather than wrapping. */
    static const ColiV4ResidencyCandidate saturating[] = {
        {COLI_V4_RESIDENCY_DENSE_TENSOR, 1, 1, UINT64_MAX, UINT64_MAX},
        {COLI_V4_RESIDENCY_DENSE_TENSOR, 2, 1, 2, 2},
    };
    unsigned char saturating_selected[2];
    if (coli_v4_residency_select(
            saturating, 2, 2, COLI_V4_RESIDENCY_VALUE_BYTES,
            saturating_selected, &plan) != 0)
        return 1;
    if (plan.expected_bytes_avoided != UINT64_MAX ||
        plan.expected_exposed_ns_avoided != UINT64_MAX)
        return 1;

    if (coli_v4_residency_select(
            NULL, 1, 50, COLI_V4_RESIDENCY_VALUE_BYTES,
            tied_selected, &plan) == 0)
        return 1;
    return 0;
}

int main(void) {
    static const ColiExpertStoreOps legacy_ops = {
        mock_lookup, mock_release, mock_prefetch, mock_stats, mock_destroy
    };
    static const ColiExpertStoreOps context_ops = {
        mock_lookup, mock_release, mock_prefetch, mock_stats, mock_destroy,
        mock_lookup_context
    };
    MockState state = {0};
    ColiExpertStore store = {&legacy_ops, &state};
    ColiExpertView view;
    ColiExpertView other;
    ColiExpertKey key = {7, 19};
    ColiExpertStoreStats stats;

    if (test_residency_value_selector()) return 1;

    if (coli_expert_lookup(&store, key, &view) != 0) return 1;
    if (view.key.layer != 7 || view.key.expert != 19) return 1;
    if (view.gate.format != COLI_TENSOR_FP4_NATIVE_BLOCK) return 1;
    if (view.gate.scale_format != COLI_SCALE_UE8M0) return 1;

    /* A second lookup while the store is busy must fail and clear *other*. */
    memset(&other, 0x5a, sizeof(other));
    if (coli_expert_lookup(&store, key, &other) == 0) return 1;
    if (!view_is_cleared(&other)) return 1;
    if (view.lease != &state || !state.held) return 1;

    coli_expert_release(&store, &view);
    if (state.held || state.active_leases) return 1;
    if (!view_is_cleared(&view)) return 1;

    /* Legacy stores must accept context-aware calls through the old lookup.
     * The context callback is absent, so observational metadata is ignored. */
    ColiExpertRequestContext context = {
        .request_id = 42,
        .token_position = 17,
        .route_rank = 3,
        .route_weight = 0.625f,
        .phase = COLI_EXPERT_PHASE_DECODE,
    };
    if (coli_expert_lookup_context(&store, key, &context, &view) != 0) return 1;
    if (state.context_calls != 0) return 1;
    if (view.key.layer != key.layer || view.key.expert != key.expert) return 1;
    coli_expert_release(&store, &view);

    /* Context-aware stores receive an exact by-value snapshot before lookup,
     * which is the contract async loader jobs will rely on. */
    store.ops = &context_ops;
    if (coli_expert_lookup_context(&store, key, &context, &view) != 0) return 1;
    if (state.context_calls != 1) return 1;
    if (state.last_context.request_id != 42 ||
        state.last_context.token_position != 17 ||
        state.last_context.route_rank != 3 ||
        state.last_context.route_weight != 0.625f ||
        state.last_context.phase != COLI_EXPERT_PHASE_DECODE)
        return 1;
    coli_expert_release(&store, &view);

    /* Failure still clears the caller's output view on the context path. */
    state.held = 1;
    memset(&other, 0x5a, sizeof(other));
    if (coli_expert_lookup_context(&store, key, &context, &other) == 0) return 1;
    if (!view_is_cleared(&other)) return 1;
    state.held = 0;

    /* Double release and release of a zero view are no-ops. */
    coli_expert_release(&store, &view);
    memset(&other, 0, sizeof(other));
    coli_expert_release(&store, &other);
    if (!view_is_cleared(&other)) return 1;

    if (store.ops->prefetch(&store, &key, 1) != 1) return 1;
    store.ops->stats(&store, &stats);
    if (stats.requests != 3 || stats.misses != 3 ||
        stats.prefetched != 1 || stats.bytes_read != 12) return 1;

    store.ops->destroy(&store);
    if (state.active_leases != 0) return 1;
    puts("expert store ops + V4 residency selector tests: ok");
    return 0;
}

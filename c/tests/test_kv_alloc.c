/* kv_alloc must survive re-allocation on the same KVState: every free path is
 * guarded by if(k->Lc) precisely so callers (context resize, slot re-init) can
 * call it again. A stale duplicate free block frees every Lc[i]/Rc[i] and both
 * arrays twice on the second call -> allocator abort. No model file needed:
 * the CPU path of kv_alloc only reads c->n_layers/kv_lora/qk_rope. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include "../sequence_state.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    unsigned char kv[24];
    unsigned char recurrent[8];
    uint64_t position;
    int reset_calls;
    int finish_calls;
    int unstable_describe;
    int write_fail_segment;
    int finish_fail;
} FakeSequenceState;

static int fake_describe(void *opaque, uint64_t position,
                         ColiSequenceStateInfo *info,
                         ColiSequenceSegmentDesc *segments, size_t cap) {
    FakeSequenceState *state = (FakeSequenceState *)opaque;
    if (!state || !info || position != state->position) return -1;
    memset(info, 0, sizeof(*info));
    info->state_abi = 7;
    info->absolute_position = position;
    info->logical_bytes = sizeof(state->kv) + sizeof(state->recurrent);
    info->resident_bytes = sizeof(*state);
    info->segment_count = 2;
    if (!segments) return 0;
    if (cap < 2) return -1;
    if (state->unstable_describe) info->state_abi = 8;
    segments[0] = (ColiSequenceSegmentDesc){
        .segment_id = 1,
        .kind = COLI_SEQUENCE_SEGMENT_PAGED_APPEND,
        .element_bytes = 1,
        .visibility = COLI_SEQUENCE_VIS_CPU | COLI_SEQUENCE_VIS_ACCELERATOR,
        .logical_rows = 3,
        .row_bytes = 8,
        .snapshot_bytes = sizeof(state->kv),
        .page_rows = 2,
        .layout_abi = 1,
    };
    segments[1] = (ColiSequenceSegmentDesc){
        .segment_id = 2,
        .kind = COLI_SEQUENCE_SEGMENT_RECURRENT_FIXED,
        .element_bytes = 1,
        .visibility = COLI_SEQUENCE_VIS_CPU,
        .logical_rows = 1,
        .row_bytes = sizeof(state->recurrent),
        .snapshot_bytes = sizeof(state->recurrent),
        .layout_abi = 2,
    };
    return 0;
}

static int fake_read(void *opaque, uint64_t position, uint32_t segment_id,
                     uint64_t offset, void *dst, size_t bytes) {
    FakeSequenceState *state = (FakeSequenceState *)opaque;
    const unsigned char *src = NULL;
    size_t size = 0;
    if (!state || position != state->position || !dst) return -1;
    if (segment_id == 1) { src = state->kv; size = sizeof(state->kv); }
    else if (segment_id == 2) { src = state->recurrent; size = sizeof(state->recurrent); }
    else return -1;
    if (offset > size || bytes > size - (size_t)offset) return -1;
    memcpy(dst, src + offset, bytes);
    return 0;
}

static int fake_write(void *opaque, uint64_t position, uint32_t segment_id,
                      uint64_t offset, const void *src, size_t bytes) {
    FakeSequenceState *state = (FakeSequenceState *)opaque;
    unsigned char *dst = NULL;
    size_t size = 0;
    if (!state || !src || state->write_fail_segment == (int)segment_id) return -1;
    if (segment_id == 1) { dst = state->kv; size = sizeof(state->kv); }
    else if (segment_id == 2) { dst = state->recurrent; size = sizeof(state->recurrent); }
    else return -1;
    if (offset > size || bytes > size - (size_t)offset) return -1;
    memcpy(dst + offset, src, bytes);
    state->position = position;
    return 0;
}

static int fake_reset(void *opaque) {
    FakeSequenceState *state = (FakeSequenceState *)opaque;
    if (!state) return -1;
    memset(state->kv, 0, sizeof(state->kv));
    memset(state->recurrent, 0, sizeof(state->recurrent));
    state->position = 0;
    state->reset_calls++;
    return 0;
}

static int fake_finish(void *opaque, uint64_t position) {
    FakeSequenceState *state = (FakeSequenceState *)opaque;
    if (!state || state->finish_fail) return -1;
    state->position = position;
    state->finish_calls++;
    return 0;
}

static int test_sequence_state(void) {
    FakeSequenceState source = {0}, restored = {0};
    source.position = 3;
    for (size_t i = 0; i < sizeof(source.kv); ++i) source.kv[i] = (unsigned char)(i + 1);
    for (size_t i = 0; i < sizeof(source.recurrent); ++i)
        source.recurrent[i] = (unsigned char)(0xa0u + i);

    static const ColiSequenceStateOps ops = {
        fake_describe, fake_read, fake_write, fake_reset, fake_finish
    };
    ColiSequenceStateAdapter src = {&source, &ops};
    ColiSequenceStateAdapter dst = {&restored, &ops};
    ColiSequenceSegmentDesc segments[2];
    ColiSequenceSnapshotLayout layout = {0};
    layout.segments = segments;
    layout.segment_capacity = 2;

    if (coli_sequence_snapshot_layout(&src, 3, &layout) != 0 ||
        layout.info.state_abi != 7 || layout.info.segment_count != 2 ||
        coli_sequence_snapshot_bytes(&layout) != 32)
        return 1;

    unsigned char blob[32];
    if (coli_sequence_snapshot_read_all(&src, &layout, blob, sizeof(blob)) != 0)
        return 2;
    if (coli_sequence_snapshot_restore_all(&dst, &layout, blob, sizeof(blob)) != 0 ||
        restored.position != 3 || restored.reset_calls != 1 || restored.finish_calls != 1 ||
        memcmp(source.kv, restored.kv, sizeof(source.kv)) ||
        memcmp(source.recurrent, restored.recurrent, sizeof(source.recurrent)))
        return 3;

    /* A boundary beyond the live state is rejected by the adapter before any
     * state bytes are read. */
    if (coli_sequence_snapshot_layout(&src, 4, &layout) == 0)
        return 4;

    /* The two describe phases must describe exactly the same boundary. */
    source.unstable_describe = 1;
    if (coli_sequence_snapshot_layout(&src, 3, &layout) == 0)
        return 5;
    source.unstable_describe = 0;
    if (coli_sequence_snapshot_layout(&src, 3, &layout) != 0)
        return 6;

    /* Framing is preflighted before reset: truncated input cannot destroy the
     * live destination state on what should simply be a rejected restore. */
    unsigned char before_kv[sizeof(restored.kv)];
    memcpy(before_kv, restored.kv, sizeof(before_kv));
    int resets_before = restored.reset_calls;
    if (coli_sequence_snapshot_restore_all(&dst, &layout, blob, sizeof(blob) - 1) == 0 ||
        restored.reset_calls != resets_before ||
        memcmp(before_kv, restored.kv, sizeof(before_kv)))
        return 7;

    /* A mid-restore adapter failure must leave an empty state, not a plausible
     * mixture of newly restored and stale segments. */
    restored.write_fail_segment = 2;
    resets_before = restored.reset_calls;
    if (coli_sequence_snapshot_restore_all(&dst, &layout, blob, sizeof(blob)) == 0 ||
        restored.reset_calls != resets_before + 2 || restored.position != 0)
        return 8;
    for (size_t i = 0; i < sizeof(restored.kv); ++i)
        if (restored.kv[i]) return 9;
    for (size_t i = 0; i < sizeof(restored.recurrent); ++i)
        if (restored.recurrent[i]) return 10;
    restored.write_fail_segment = 0;

    /* Finalization failure receives the same fail-empty treatment. */
    restored.finish_fail = 1;
    resets_before = restored.reset_calls;
    if (coli_sequence_snapshot_restore_all(&dst, &layout, blob, sizeof(blob)) == 0 ||
        restored.reset_calls != resets_before + 2 || restored.position != 0)
        return 11;
    restored.finish_fail = 0;

    return 0;
}

int main(void){
    static Model m;
    m.c.n_layers=2; m.c.kv_lora=8; m.c.qk_rope=4;
    m.kv=calloc(1,sizeof(KVState));
    kv_alloc(&m,16);
    for(int i=0;i<m.c.n_layers+1;i++){ m.Lc[i][0]=1.0f; m.Rc[i][0]=1.0f; }
    kv_alloc(&m,32);                       /* the re-allocation path under test */
    for(int i=0;i<m.c.n_layers+1;i++){
        m.Lc[i][(int64_t)32*m.c.kv_lora-1]=2.0f;
        m.Rc[i][(int64_t)32*m.c.qk_rope-1]=2.0f;
    }
    int rc = test_sequence_state();
    if (rc) return 20 + rc;
    printf("OK kv_alloc re-allocation + sequence-state adapter\n");
    return 0;
}

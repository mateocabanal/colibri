#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Faithful field surface used by qwen_adaptive_backing_pool_adapter.h. The
 * test intentionally does not include qwen_moe.c: it validates that the Qwen
 * physical adapter can circulate an opaque expert allocation without freeing
 * it or multiplying the global transient slot count by layers. */
typedef struct {
    int eid, loading_eid, pinned, fmt;
    int mio, mio_slot, mio_resident;
    int64_t mio_event, mio_waited;
    float *gu, *d;
    uint16_t *bgu, *bd;
    int8_t *g, *u, *dd;
    float *gs, *us, *ds;
    uint8_t *g4, *u4, *d4;
    float *g4s, *u4s, *d4s;
    uint8_t *mxg, *mxu, *mxd;
    uint8_t *mxgs, *mxus, *mxds;
    uint8_t *mxbase;
    uint64_t used;
} Slot;

typedef struct { Slot *slots; int n, cap; } LCache;

#define QWEN_ADAPTIVE_METALIO_REINIT(_s) ((void)(_s))
#define QWEN_ADAPTIVE_RELEASE_SLOT(_s) do { \
    free((_s)->mxbase); \
    memset((_s), 0, sizeof(*(_s))); \
    (_s)->eid = -2; \
    (_s)->loading_eid = -1; \
    (_s)->mio_slot = -1; \
} while (0)

#include "../qwen_adaptive_backing_pool_adapter.h"

static void reset_test_pool(void) {
    if (g_qwen_adaptive_backing.initialized) {
        for (size_t i = 0; i < g_qwen_adaptive_backing.pool.slot_count; i++) {
            ColiExpertBackingSlot *ps = &g_qwen_adaptive_backing.pool.slots[i];
            Slot *shadow = (Slot *)ps->payload;
            if (!shadow) continue;
            if (atomic_load_explicit(&ps->state, memory_order_acquire) ==
                    COLI_EXPERT_BACKING_AVAILABLE)
                QWEN_ADAPTIVE_RELEASE_SLOT(shadow);
            free(shadow);
        }
        coli_expert_backing_pool_destroy(&g_qwen_adaptive_backing.pool);
    }
    memset(&g_qwen_adaptive_backing, 0, sizeof(g_qwen_adaptive_backing));
    g_qwen_adaptive_backing.last_layer = -1;
}

static void test_circulate_and_promote(void) {
    reset_test_pool();
    assert(coli_expert_backing_pool_init(
        &g_qwen_adaptive_backing.pool, 2) == 0);
    g_qwen_adaptive_backing.initialized = 1;
    g_qwen_adaptive_backing.slot_count = 2;
    g_qwen_adaptive_backing.model = (void *)(uintptr_t)1;

    Slot source = {0};
    source.eid = 3;
    source.loading_eid = -1;
    source.mio_slot = -1;
    source.fmt = 7;
    source.mxbase = (uint8_t *)malloc(4096);
    assert(source.mxbase);
    uint8_t *physical = source.mxbase;

    int deposited = 0;
    QWEN_ADAPTIVE_DEPOSIT_TRANSIENT(&source, 4096, deposited);
    assert(deposited == 1);
    assert(QWEN_ADAPTIVE_BACKING_ID(&source) == NULL);

    ColiExpertBackingPoolStats stats =
        coli_expert_backing_pool_stats(&g_qwen_adaptive_backing.pool);
    assert(stats.slots_with_payload == 1);
    assert(stats.leased_slots == 0);
    assert(stats.allocated_bytes == 4096);

    Slot storage[2];
    memset(storage, 0, sizeof(storage));
    for (int i = 0; i < 2; i++) {
        storage[i].eid = -2;
        storage[i].loading_eid = -1;
        storage[i].mio_slot = -1;
    }
    LCache cache = {.slots = storage, .n = 0, .cap = 2};

    QWEN_ADAPTIVE_ATTACH_TRANSIENT(&cache, 4096);
    assert(cache.n == 0); /* logical activation still belongs to expert_get */
    assert(storage[0].mxbase == physical);
    assert(storage[0].eid == -2);
    stats = coli_expert_backing_pool_stats(&g_qwen_adaptive_backing.pool);
    assert(stats.leased_slots == 1);

    /* Simulate expert_get/load_expert re-binding the borrowed physical slot. */
    cache.n = 1;
    storage[0].eid = 9;
    storage[0].used = 42;

    int returned = 0;
    QWEN_ADAPTIVE_RETURN_BORROWED(&storage[0], returned);
    assert(returned == 1);
    assert(QWEN_ADAPTIVE_BACKING_ID(&storage[0]) == NULL);
    stats = coli_expert_backing_pool_stats(&g_qwen_adaptive_backing.pool);
    assert(stats.leased_slots == 0);

    cache.n = 0;
    QWEN_ADAPTIVE_ATTACH_TRANSIENT(&cache, 4096);
    assert(storage[0].mxbase == physical);
    assert(storage[0].eid == -2);

    /* Selecting the currently borrowed expert promotes the same allocation to
     * persistent ownership and empties the transient metadata slot. */
    storage[0].eid = 11;
    storage[0].pinned = 1;
    int promoted = 0;
    QWEN_ADAPTIVE_PROMOTE_BORROWED(&storage[0], promoted);
    assert(promoted == 1);
    assert(storage[0].mxbase == physical);
    stats = coli_expert_backing_pool_stats(&g_qwen_adaptive_backing.pool);
    assert(stats.leased_slots == 0);
    assert(stats.slots_with_payload == 0);

    free(storage[0].mxbase); /* persistent owner after promotion */
    storage[0].mxbase = NULL;
    reset_test_pool();
}

static void test_global_slot_bound(void) {
    reset_test_pool();
    assert(coli_expert_backing_pool_init(
        &g_qwen_adaptive_backing.pool, 1) == 0);
    g_qwen_adaptive_backing.initialized = 1;
    g_qwen_adaptive_backing.slot_count = 1;
    g_qwen_adaptive_backing.model = (void *)(uintptr_t)1;

    Slot a = {.eid = 1, .loading_eid = -1, .mio_slot = -1};
    Slot b = {.eid = 2, .loading_eid = -1, .mio_slot = -1};
    a.mxbase = (uint8_t *)malloc(1024);
    b.mxbase = (uint8_t *)malloc(1024);
    assert(a.mxbase && b.mxbase);

    int deposited_a = 0, deposited_b = 0;
    QWEN_ADAPTIVE_DEPOSIT_TRANSIENT(&a, 1024, deposited_a);
    QWEN_ADAPTIVE_DEPOSIT_TRANSIENT(&b, 1024, deposited_b);
    assert(deposited_a == 1);
    assert(deposited_b == 0); /* second backing is outside the global budget */
    assert(b.mxbase != NULL);

    free(b.mxbase);
    b.mxbase = NULL;
    reset_test_pool();
}

int main(void) {
    test_circulate_and_promote();
    test_global_slot_bound();
    puts("qwen adaptive backing pool: ok");
    return 0;
}

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "../expert_backing_pool.h"

static void test_reuses_compatible_backing(void) {
    ColiExpertBackingPool pool;
    assert(coli_expert_backing_pool_init(&pool, 2) == 0);

    int payload_a = 1;
    ColiExpertBackingLease first;
    assert(coli_expert_backing_pool_acquire(
        &pool, 7, 1, 32u * 1024u * 1024u, &first) == 1);
    assert(!first.reused);
    uint64_t first_generation = first.generation;
    assert(coli_expert_backing_lease_set_payload(
        &first, &payload_a, 32u * 1024u * 1024u, 7, 1, NULL) == 0);
    assert(coli_expert_backing_pool_release(&first) == 0);

    ColiExpertBackingLease second;
    assert(coli_expert_backing_pool_acquire(
        &pool, 7, 1, 16u * 1024u * 1024u, &second) == 1);
    assert(second.reused);
    assert(second.generation != first_generation);
    ColiExpertBackingSlot *slot = coli_expert_backing_lease_slot(&second);
    assert(slot && slot->payload == &payload_a);
    assert(coli_expert_backing_pool_release(&second) == 0);

    ColiExpertBackingPoolStats stats = coli_expert_backing_pool_stats(&pool);
    assert(stats.acquires == 2);
    assert(stats.reuses == 1);
    assert(stats.releases == 2);
    assert(stats.slots_with_payload == 1);
    assert(stats.allocated_bytes == 32u * 1024u * 1024u);
    assert(coli_expert_backing_pool_can_destroy(&pool));
    coli_expert_backing_pool_destroy(&pool);
}

static void test_global_bound_and_stale_generation(void) {
    ColiExpertBackingPool pool;
    assert(coli_expert_backing_pool_init(&pool, 2) == 0);

    ColiExpertBackingLease a, b, blocked;
    assert(coli_expert_backing_pool_acquire(&pool, 1, 1, 1, &a) == 1);
    assert(coli_expert_backing_pool_acquire(&pool, 1, 1, 1, &b) == 1);
    assert(coli_expert_backing_pool_acquire(&pool, 1, 1, 1, &blocked) == 0);
    assert(!coli_expert_backing_pool_can_destroy(&pool));

    size_t old_index = a.index;
    uint64_t old_generation = a.generation;
    assert(coli_expert_backing_pool_release(&a) == 0);

    ColiExpertBackingLease next;
    assert(coli_expert_backing_pool_acquire(&pool, 1, 1, 1, &next) == 1);
    assert(next.index == old_index);
    assert(next.generation != old_generation);
    assert(coli_expert_backing_pool_release_generation(
        &pool, old_index, old_generation) == -1);

    assert(coli_expert_backing_pool_release(&next) == 0);
    assert(coli_expert_backing_pool_release(&b) == 0);
    ColiExpertBackingPoolStats stats = coli_expert_backing_pool_stats(&pool);
    assert(stats.waits == 1);
    assert(stats.leased_slots == 0);
    coli_expert_backing_pool_destroy(&pool);
}

static void test_empty_before_incompatible_repurpose(void) {
    ColiExpertBackingPool pool;
    assert(coli_expert_backing_pool_init(&pool, 2) == 0);

    int payload = 7;
    ColiExpertBackingLease a;
    assert(coli_expert_backing_pool_acquire(&pool, 7, 1, 64, &a) == 1);
    assert(coli_expert_backing_lease_set_payload(
        &a, &payload, 64, 7, 1, NULL) == 0);
    assert(coli_expert_backing_pool_release(&a) == 0);

    ColiExpertBackingLease b;
    assert(coli_expert_backing_pool_acquire(&pool, 8, 1, 64, &b) == 1);
    assert(!b.reused);
    ColiExpertBackingSlot *slot = coli_expert_backing_lease_slot(&b);
    assert(slot && slot->payload == NULL);
    assert(coli_expert_backing_pool_release(&b) == 0);

    coli_expert_backing_pool_destroy(&pool);
}

int main(void) {
    test_reuses_compatible_backing();
    test_global_bound_and_stale_generation();
    test_empty_before_incompatible_repurpose();
    puts("generic expert backing pool: ok");
    return 0;
}

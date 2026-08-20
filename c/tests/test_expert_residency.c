#include "../expert_residency.h"

#include <stdatomic.h>
#include <stdio.h>

static int expect_cold_clean(ColiExpertResidencyEntry *entry,
                             ColiExpertResidencyBudget *budget,
                             uint64_t generation) {
    return coli_expert_residency_state(entry) == COLI_EXPERT_RESIDENCY_COLD &&
        atomic_load(&entry->generation) == generation &&
        atomic_load(&entry->refs) == 0 &&
        atomic_load(&budget->committed_bytes) == 0 &&
        atomic_load(&budget->reserved_bytes) == 0 &&
        atomic_load(&budget->resident_bytes) == 0 &&
        entry->resident_bytes == 0 && entry->physical == NULL &&
        !coli_representation_known(&entry->representation);
}

int main(void) {
    ColiExpertResidencyBudget budget;
    ColiExpertResidencyEntry entry;
    ColiExpertResidencyLease lease;
    ColiExpertKey key = {4, 17};

    coli_expert_residency_budget_init(&budget, 100);
    if (coli_expert_residency_entry_init(&entry, key) != 0) return 1;

    /* Existing callers keep identical lifetime/generation behavior. They now
     * expose UNKNOWN representation instead of silently inferring one. */
    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish(&entry, &budget, 7,
                                      COLI_EXPERT_TIER_HOST) != 0)
        return 2;
    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_HIT || lease.generation != 7 ||
        lease.resident_bytes != 60 || lease.allocation_bytes != 60 ||
        coli_representation_known(&lease.representation) ||
        atomic_load(&entry.refs) != 1)
        return 3;
    if (coli_expert_residency_release(&lease) != 0 ||
        coli_expert_residency_begin_evict(&entry) != 1 ||
        coli_expert_residency_finish_evict(&entry, &budget) != 0 ||
        !expect_cold_clean(&entry, &budget, 7))
        return 4;

    /* Reusing the exact old generation after physical-slot reuse would make a
     * stale backend descriptor valid again. Publication must fail without
     * consuming or reclassifying the reservation. */
    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish(&entry, &budget, 7,
                                      COLI_EXPERT_TIER_HOST) == 0 ||
        coli_expert_residency_state(&entry) != COLI_EXPERT_RESIDENCY_RESERVED ||
        atomic_load(&budget.committed_bytes) != 60 ||
        atomic_load(&budget.reserved_bytes) != 60 ||
        atomic_load(&budget.resident_bytes) != 0)
        return 5;
    if (coli_expert_residency_fail_load(&entry, &budget) != 0 ||
        !expect_cold_clean(&entry, &budget, 7))
        return 6;

    /* Older generations are equally invalid. */
    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish(&entry, &budget, 6,
                                      COLI_EXPERT_TIER_HOST) == 0 ||
        coli_expert_residency_fail_load(&entry, &budget) != 0 ||
        !expect_cold_clean(&entry, &budget, 7))
        return 7;

    ColiRepresentationId rep = {
        .math_format = COLI_CSF_MATH_MXFP4_E2M1,
        .scale_format = COLI_CSF_SCALE_UE8M0,
        .execution_layout = COLI_CSF_LAYOUT_CANONICAL,
        .execution_layout_abi = 1,
        .kernel_abi = 1,
        .target_class = 0,
        .group_size = 32,
        .scale_block_rows = 1,
        .scale_block_columns = 32,
    };
    int physical_slot = 42;

    /* A strictly newer generation is publishable and leases pin the exact
     * representation/physical generation that was actually published. */
    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish_representation(
            &entry, &budget, 8,
            COLI_EXPERT_TIER_PINNED_HOST | COLI_EXPERT_TIER_DEVICE,
            &rep, 56, &physical_slot) != 0 ||
        coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_HIT || lease.generation != 8 ||
        !coli_representation_equal(&lease.representation, &rep) ||
        lease.resident_bytes != 56 || lease.allocation_bytes != 60 ||
        lease.physical != &physical_slot || atomic_load(&entry.refs) != 1)
        return 8;

    ColiExpertResidentView view;
    if (coli_expert_residency_lease_view(&lease, &view) != 0 ||
        !coli_expert_key_equal(view.key, key) || view.generation != 8 ||
        !coli_representation_equal(&view.representation, &rep) ||
        view.resident_bytes != 56 || view.allocation_bytes != 60 ||
        view.physical != &physical_slot)
        return 9;

    /* Even if client code copied an old lease before releasing it, that stale
     * identity must not be able to release the ref belonging to generation 8. */
    ColiExpertResidencyLease stale = {
        .entry = &entry,
        .key = key,
        .generation = 7,
        .tier_mask = COLI_EXPERT_TIER_HOST,
    };
    if (coli_expert_residency_release(&stale) == 0 ||
        atomic_load(&entry.refs) != 1)
        return 10;

    if (coli_expert_residency_release(&lease) != 0 ||
        coli_expert_residency_begin_evict(&entry) != 1 ||
        coli_expert_residency_finish_evict(&entry, &budget) != 0 ||
        !expect_cold_clean(&entry, &budget, 8) ||
        atomic_load(&budget.peak_committed_bytes) > budget.capacity_bytes)
        return 11;

    puts("PASS expert residency representation + monotonic generation reuse");
    return 0;
}

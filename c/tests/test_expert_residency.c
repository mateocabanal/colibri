#include "../expert_residency.h"

#include <stdatomic.h>
#include <stdio.h>

static int expect_cold_clean(ColiExpertResidencyEntry *entry,
                             ColiExpertResidencyBudget *budget,
                             uint64_t generation) {
    return coli_expert_residency_state(entry) == COLI_EXPERT_RESIDENCY_COLD &&
        atomic_load(&entry->generation) == generation &&
        atomic_load(&entry->refs) == 0 &&
        entry->resident_bytes == 0 &&
        !coli_representation_valid(&entry->representation) &&
        entry->physical_handle == NULL &&
        atomic_load(&budget->committed_bytes) == 0 &&
        atomic_load(&budget->reserved_bytes) == 0 &&
        atomic_load(&budget->resident_bytes) == 0;
}

static int make_rep(ColiRepresentationId *rep, uint16_t layout) {
    return coli_representation_init(
        rep, COLI_CSF_MATH_MXFP4_E2M1, COLI_CSF_SCALE_UE8M0,
        layout, 1, 1, 1, 32, 32, COLI_REPRESENTATION_F_NONE,
        COLI_CSF_PROFILE_MACOS_ARM64_METAL_APPLE8_V1);
}

int main(void) {
    ColiExpertResidencyBudget budget;
    ColiExpertResidencyEntry entry;
    ColiExpertResidencyLease lease;
    ColiExpertResidentView resident;
    ColiExpertKey key = {4, 17};
    ColiRepresentationId canonical;
    ColiRepresentationId rows16;
    int physical_a;
    int physical_b;

    if (make_rep(&canonical, COLI_CSF_LAYOUT_CANONICAL) != 0 ||
        make_rep(&rows16, COLI_CSF_LAYOUT_ROWS16) != 0)
        return 1;

    coli_expert_residency_budget_init(&budget, 100);
    if (coli_expert_residency_entry_init(&entry, key) != 0) return 2;

    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish_representation(
            &entry, &budget, 7, COLI_EXPERT_TIER_HOST, 55,
            &canonical, &physical_a) != 0)
        return 3;
    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_HIT || lease.generation != 7 ||
        atomic_load(&entry.refs) != 1 ||
        coli_expert_residency_lease_view(&lease, &resident) != 0 ||
        resident.generation != 7 ||
        resident.tier_mask != COLI_EXPERT_TIER_HOST ||
        resident.resident_bytes != 55 ||
        resident.allocation_bytes != 60 ||
        resident.physical_handle != &physical_a ||
        !coli_representation_equal(&resident.representation, &canonical))
        return 4;
    if (coli_expert_residency_release(&lease) != 0 ||
        coli_expert_residency_begin_evict(&entry) != 1 ||
        coli_expert_residency_finish_evict(&entry, &budget) != 0 ||
        !expect_cold_clean(&entry, &budget, 7))
        return 5;

    /* Reusing the exact old generation after physical-slot reuse would make a
     * stale backend descriptor valid again. Publication must fail without
     * consuming or reclassifying the reservation. */
    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish_representation(
            &entry, &budget, 7, COLI_EXPERT_TIER_HOST, 55,
            &canonical, &physical_a) == 0 ||
        coli_expert_residency_state(&entry) != COLI_EXPERT_RESIDENCY_RESERVED ||
        atomic_load(&budget.committed_bytes) != 60 ||
        atomic_load(&budget.reserved_bytes) != 60 ||
        atomic_load(&budget.resident_bytes) != 0)
        return 6;
    if (coli_expert_residency_fail_load(&entry, &budget) != 0 ||
        !expect_cold_clean(&entry, &budget, 7))
        return 7;

    /* Older generations are equally invalid. */
    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish_representation(
            &entry, &budget, 6, COLI_EXPERT_TIER_HOST, 55,
            &canonical, &physical_a) == 0 ||
        coli_expert_residency_fail_load(&entry, &budget) != 0 ||
        !expect_cold_clean(&entry, &budget, 7))
        return 8;

    /*
     * A newer publication may carry a different representation identity while
     * the generation rules remain unchanged. This is sequential reuse only;
     * simultaneous variants belong to #134.
     */
    if (coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish_representation(
            &entry, &budget, 8,
            COLI_EXPERT_TIER_PINNED_HOST | COLI_EXPERT_TIER_DEVICE,
            58, &rows16, &physical_b) != 0 ||
        coli_expert_residency_request(&entry, &budget, 60, &lease) !=
            COLI_EXPERT_REQUEST_HIT || lease.generation != 8 ||
        atomic_load(&entry.refs) != 1 ||
        coli_expert_residency_lease_view(&lease, &resident) != 0 ||
        resident.physical_handle != &physical_b ||
        resident.resident_bytes != 58 ||
        !coli_representation_equal(&resident.representation, &rows16))
        return 9;

    /* Even if client code copied an old lease before releasing it, that stale
     * identity must not be able to release the ref belonging to generation 8. */
    ColiExpertResidencyLease stale = {
        .entry = &entry,
        .key = key,
        .generation = 7,
        .tier_mask = COLI_EXPERT_TIER_HOST,
        .resident = {
            .key = key,
            .representation = canonical,
            .generation = 7,
            .tier_mask = COLI_EXPERT_TIER_HOST,
            .resident_bytes = 55,
            .allocation_bytes = 60,
            .physical_handle = &physical_a,
        },
    };
    if (coli_expert_residency_release(&stale) == 0 ||
        coli_expert_residency_lease_view(&stale, &resident) == 0 ||
        atomic_load(&entry.refs) != 1)
        return 10;

    if (coli_expert_residency_release(&lease) != 0 ||
        coli_expert_residency_begin_evict(&entry) != 1 ||
        coli_expert_residency_finish_evict(&entry, &budget) != 0 ||
        !expect_cold_clean(&entry, &budget, 8) ||
        atomic_load(&budget.peak_committed_bytes) > budget.capacity_bytes)
        return 11;

    /*
     * Legacy publication remains lifetime-compatible for callers that have not
     * migrated yet, but its representation is explicitly unknown. New dispatch
     * must fail closed instead of inferring bytes from model/backend context.
     */
    if (coli_expert_residency_request(&entry, &budget, 40, &lease) !=
            COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish(
            &entry, &budget, 9, COLI_EXPERT_TIER_HOST) != 0 ||
        coli_expert_residency_request(&entry, &budget, 40, &lease) !=
            COLI_EXPERT_REQUEST_HIT ||
        coli_expert_residency_lease_view(&lease, &resident) != 0 ||
        coli_representation_valid(&resident.representation) ||
        resident.resident_bytes != 40 ||
        resident.allocation_bytes != 40)
        return 12;
    if (coli_expert_residency_release(&lease) != 0 ||
        coli_expert_residency_begin_evict(&entry) != 1 ||
        coli_expert_residency_finish_evict(&entry, &budget) != 0 ||
        !expect_cold_clean(&entry, &budget, 9))
        return 13;

    puts("PASS expert residency representation + monotonic generation reuse");
    return 0;
}

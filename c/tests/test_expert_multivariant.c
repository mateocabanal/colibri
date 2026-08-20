#include "../expert_residency.h"

#include <stdatomic.h>
#include <stdio.h>

static ColiRepresentationId rep(uint16_t layout, uint32_t target_class) {
    ColiRepresentationId r = {
        .math_format = COLI_CSF_MATH_MXFP4_E2M1,
        .scale_format = COLI_CSF_SCALE_UE8M0,
        .execution_layout = layout,
        .execution_layout_abi = 1,
        .kernel_abi = 1,
        .target_class = target_class,
        .group_size = 32,
        .scale_block_rows = 1,
        .scale_block_columns = 32,
    };
    return r;
}

static int refs(ColiExpertResidencyEntry *entry, uint32_t variant_id) {
    return (int)atomic_load(&entry->variants[variant_id].refs);
}

int main(void) {
    ColiExpertResidencyBudget budget;
    ColiExpertResidencyEntry entry;
    ColiExpertKey key = {9, 23};
    ColiRepresentationId a = rep(COLI_CSF_LAYOUT_CANONICAL, 0);
    ColiRepresentationId b = rep(0x0102, 0x0000a008u);
    ColiRepresentationId c = rep(0x0202, 0x0000c001u);
    ColiRepresentationId d = rep(0x0302, 0x0000c0dau);
    int physical_a = 1, physical_b = 2, physical_c = 3, physical_d = 4;

    coli_expert_residency_budget_init(&budget, 150);
    if (coli_expert_residency_entry_init(&entry, key) != 0) return 1;

    uint32_t a_id, b_id, c_id, d_id;
    uint64_t a_gen, b_gen, c_gen, d_gen;

    if (coli_expert_residency_reserve_variant(
            &entry, &budget, &a, COLI_EXPERT_TIER_UMA, 60,
            &a_id, &a_gen) != COLI_EXPERT_REQUEST_LOAD_OWNER ||
        a_id != 0 || a_gen != 1 ||
        coli_expert_residency_publish_variant(
            &entry, &budget, a_id, a_gen, 56, &physical_a) != 0 ||
        coli_expert_residency_preferred_variant(&entry) != (int)a_id)
        return 2;

    ColiExpertResidencyLease old_lease;
    if (coli_expert_residency_acquire_preferred(&entry, &old_lease) != 1 ||
        old_lease.variant_id != a_id || old_lease.generation != a_gen ||
        !coli_representation_equal(&old_lease.representation, &a))
        return 3;

    if (coli_expert_residency_reserve_variant(
            &entry, &budget, &b, COLI_EXPERT_TIER_UMA, 60,
            &b_id, &b_gen) != COLI_EXPERT_REQUEST_LOAD_OWNER ||
        b_id == a_id || b_gen <= a_gen ||
        coli_expert_residency_mark_variant_preparing(
            &entry, b_id, b_gen) != 0 ||
        coli_expert_residency_publish_variant(
            &entry, &budget, b_id, b_gen, 56, &physical_b) != 0 ||
        atomic_load(&budget.committed_bytes) != 120 ||
        coli_expert_residency_set_preferred(&entry, b_id) != 0)
        return 4;

    ColiExpertResidencyLease new_lease;
    if (coli_expert_residency_acquire_preferred(&entry, &new_lease) != 1 ||
        new_lease.variant_id != b_id || new_lease.generation != b_gen ||
        !coli_representation_equal(&new_lease.representation, &b) ||
        old_lease.variant_id != a_id || old_lease.generation != a_gen ||
        !coli_representation_equal(&old_lease.representation, &a) ||
        refs(&entry, a_id) != 1 || refs(&entry, b_id) != 1)
        return 5;

    if (coli_expert_residency_begin_variant_evict(&entry, a_id) != 0 ||
        coli_expert_residency_release(&old_lease) != 0 ||
        coli_expert_residency_begin_variant_evict(&entry, a_id) != 1 ||
        coli_expert_residency_finish_variant_evict(
            &entry, &budget, a_id) != 0 ||
        coli_expert_residency_variant_state(&entry, a_id) !=
            COLI_EXPERT_RESIDENCY_COLD ||
        !coli_expert_residency_lease_valid(&new_lease) ||
        coli_expert_residency_preferred_variant(&entry) != (int)b_id ||
        atomic_load(&budget.committed_bytes) != 60)
        return 6;

    if (coli_expert_residency_reserve_variant(
            &entry, &budget, &c, COLI_EXPERT_TIER_HOST, 50,
            &c_id, &c_gen) != COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_mark_variant_preparing(
            &entry, c_id, c_gen) != 0 ||
        coli_expert_residency_fail_variant(
            &entry, &budget, c_id, c_gen) != 0 ||
        coli_expert_residency_preferred_variant(&entry) != (int)b_id ||
        atomic_load(&budget.committed_bytes) != 60)
        return 7;

    if (coli_expert_residency_reserve_variant(
            &entry, &budget, &c, COLI_EXPERT_TIER_HOST, 100,
            &c_id, &c_gen) != COLI_EXPERT_REQUEST_NO_BUDGET ||
        atomic_load(&budget.committed_bytes) != 60 ||
        atomic_load(&budget.peak_committed_bytes) > budget.capacity_bytes)
        return 8;

    uint32_t c2_id;
    uint64_t c2_gen;
    if (coli_expert_residency_reserve_variant(
            &entry, &budget, &c, COLI_EXPERT_TIER_HOST, 40,
            &c_id, &c_gen) != COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_reserve_variant(
            &entry, &budget, &c, COLI_EXPERT_TIER_HOST, 40,
            &c2_id, &c2_gen) != COLI_EXPERT_REQUEST_JOIN_INFLIGHT ||
        c2_id != c_id || c2_gen != c_gen ||
        coli_expert_residency_fail_variant(
            &entry, &budget, c_id, c_gen) != 0)
        return 9;

    if (coli_expert_residency_reserve_variant(
            &entry, &budget, &c, COLI_EXPERT_TIER_HOST, 40,
            &c_id, &c_gen) != COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish_variant(
            &entry, &budget, c_id, c_gen, 36, &physical_c) != 0 ||
        coli_expert_residency_reserve_variant(
            &entry, &budget, &c, COLI_EXPERT_TIER_DEVICE, 40,
            &d_id, &d_gen) != COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_publish_variant(
            &entry, &budget, d_id, d_gen, 36, &physical_d) != 0 ||
        c_id == d_id ||
        coli_expert_residency_resident_variant_count(&entry) != 3 ||
        atomic_load(&budget.committed_bytes) != 140)
        return 10;

    ColiExpertResidencyLease compatible;
    if (coli_expert_residency_acquire_compatible(
            &entry, &c, &compatible) != 1 ||
        !coli_representation_equal(&compatible.representation, &c) ||
        coli_expert_residency_release(&compatible) != 0)
        return 11;

    if (coli_expert_residency_begin_variant_evict(&entry, c_id) != 1 ||
        coli_expert_residency_finish_variant_evict(
            &entry, &budget, c_id) != 0 ||
        coli_expert_residency_begin_variant_evict(&entry, d_id) != 1 ||
        coli_expert_residency_finish_variant_evict(
            &entry, &budget, d_id) != 0 ||
        atomic_load(&budget.committed_bytes) != 60)
        return 12;

    ColiExpertResidencyLease source;
    if (coli_expert_residency_acquire_variant(&entry, b_id, &source) != 1)
        return 13;
    ColiExpertResidencyLease stale_source = source;

    if (coli_expert_residency_reserve_variant(
            &entry, &budget, &a, COLI_EXPERT_TIER_HOST, 40,
            &a_id, &a_gen) != COLI_EXPERT_REQUEST_LOAD_OWNER ||
        coli_expert_residency_mark_variant_preparing(
            &entry, a_id, a_gen) != 0 ||
        coli_expert_residency_release(&source) != 0 ||
        coli_expert_residency_release(&new_lease) != 0 ||
        coli_expert_residency_begin_variant_evict(&entry, b_id) != 1 ||
        coli_expert_residency_finish_variant_evict(
            &entry, &budget, b_id) != 0)
        return 14;

    if (coli_expert_residency_reserve_variant(
            &entry, &budget, &d, COLI_EXPERT_TIER_UMA, 60,
            &d_id, &d_gen) != COLI_EXPERT_REQUEST_LOAD_OWNER ||
        d_id != b_id || d_gen <= b_gen ||
        coli_expert_residency_publish_variant(
            &entry, &budget, d_id, d_gen, 56, &physical_d) != 0 ||
        coli_expert_residency_publish_variant_from_source(
            &entry, &budget, a_id, a_gen, 36, &physical_a,
            &stale_source) == 0 ||
        coli_expert_residency_variant_state(&entry, a_id) !=
            COLI_EXPERT_RESIDENCY_PREPARING)
        return 15;

    ColiExpertResidencyLease replacement;
    if (coli_expert_residency_acquire_variant(
            &entry, d_id, &replacement) != 1 ||
        coli_expert_residency_release(&stale_source) == 0 ||
        refs(&entry, d_id) != 1 ||
        coli_expert_residency_release(&replacement) != 0 ||
        coli_expert_residency_fail_variant(
            &entry, &budget, a_id, a_gen) != 0)
        return 16;

    ColiExpertResidentVariantInfo info;
    if (coli_expert_residency_query_variant(&entry, d_id, &info) != 0 ||
        info.state != COLI_EXPERT_RESIDENCY_RESIDENT ||
        info.generation != d_gen ||
        !coli_representation_equal(&info.representation, &d) ||
        !info.preferred ||
        atomic_load(&entry.generation_allocator) != d_gen ||
        atomic_load(&budget.peak_committed_bytes) > budget.capacity_bytes)
        return 17;

    puts("PASS multi-variant residency promotion + generation-safe leases");
    return 0;
}

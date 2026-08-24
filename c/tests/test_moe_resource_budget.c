#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "../moe_resource_budget.h"

static void test_automatic_bytes_not_model_constants(void) {
    ColiMoeResourceBudgetInputs in = {
        .available_bytes = 8 * UINT64_C(1073741824),
        .resident_bytes_per_expert = 16 * COLI_MOE_MIB,
        .transient_slots = 8,
    };
    ColiMoeResourceBudget out;
    assert(coli_moe_resource_budget_compute(&out, &in) == 0);
    assert(out.system_reserve_bytes == UINT64_C(1073741824));
    assert(out.transient_bytes == 128 * COLI_MOE_MIB);
    assert(out.persistent_slots > 0);
    assert(out.persistent_budget_bytes ==
           out.persistent_slots * in.resident_bytes_per_expert);
}

static void test_explicit_total_is_total_not_expert_budget(void) {
    ColiMoeResourceBudgetInputs in = {
        .available_bytes = 12 * UINT64_C(1073741824),
        .process_resident_bytes = 9 * UINT64_C(1073741824),
        .explicit_total_bytes = 10 * UINT64_C(1073741824),
        .other_optional_reserve_bytes = 256 * COLI_MOE_MIB,
        .resident_bytes_per_expert = 16 * COLI_MOE_MIB,
        .transient_slots = 4,
    };
    ColiMoeResourceBudget out;
    assert(coli_moe_resource_budget_compute(&out, &in) == 0);
    assert(out.used_explicit_total == 1);
    uint64_t ceiling = UINT64_C(1073741824) - 256 * COLI_MOE_MIB -
                       4 * 16 * COLI_MOE_MIB;
    assert(out.persistent_budget_bytes <= ceiling);
}

static void test_replan_adds_back_current_optional_residency(void) {
    const uint64_t expert = 16 * COLI_MOE_MIB;
    ColiMoeResourceBudgetInputs cold = {
        .available_bytes = 8 * UINT64_C(1073741824),
        .process_resident_bytes = 6 * UINT64_C(1073741824),
        .explicit_total_bytes = 10 * UINT64_C(1073741824),
        .resident_bytes_per_expert = expert,
        .transient_slots = 4,
    };
    ColiMoeResourceBudget first;
    assert(coli_moe_resource_budget_compute(&first, &cold) == 0);
    assert(first.persistent_budget_bytes > 0);

    /* Simulate 1 GiB of the selected optional pool becoming resident. RSS rises
     * and OS-available falls by the same amount. The target pool must not lose
     * that 1 GiB merely because it is already resident. */
    ColiMoeResourceBudgetInputs warm = cold;
    warm.available_bytes -= UINT64_C(1073741824);
    warm.process_resident_bytes += UINT64_C(1073741824);
    warm.current_optional_resident_bytes = UINT64_C(1073741824);
    ColiMoeResourceBudget second;
    assert(coli_moe_resource_budget_compute(&second, &warm) == 0);
    assert(second.persistent_budget_bytes == first.persistent_budget_bytes);
}

static void test_automatic_replan_keeps_system_reserve(void) {
    const uint64_t expert = 16 * COLI_MOE_MIB;
    ColiMoeResourceBudgetInputs cold = {
        .available_bytes = 8 * UINT64_C(1073741824),
        .resident_bytes_per_expert = expert,
        .transient_slots = 4,
    };
    ColiMoeResourceBudget first;
    assert(coli_moe_resource_budget_compute(&first, &cold) == 0);
    assert(first.system_reserve_bytes == UINT64_C(1073741824));

    /* Moving one GiB from OS-available into planner-owned optional residency
     * does not change the effective memory pool, so neither the safety reserve
     * nor the target persistent envelope may grow or shrink. */
    ColiMoeResourceBudgetInputs warm = cold;
    warm.available_bytes -= UINT64_C(1073741824);
    warm.current_optional_resident_bytes = UINT64_C(1073741824);
    ColiMoeResourceBudget second;
    assert(coli_moe_resource_budget_compute(&second, &warm) == 0);
    assert(second.system_reserve_bytes == first.system_reserve_bytes);
    assert(second.persistent_budget_bytes == first.persistent_budget_bytes);
}

static void test_explicit_slot_ceiling(void) {
    ColiMoeResourceBudgetInputs in = {
        .available_bytes = 64 * UINT64_C(1073741824),
        .resident_bytes_per_expert = 8 * COLI_MOE_MIB,
        .transient_slots = 2,
        .max_persistent_slots = 7,
    };
    ColiMoeResourceBudget out;
    assert(coli_moe_resource_budget_compute(&out, &in) == 0);
    assert(out.persistent_slots == 7);
    assert(out.persistent_budget_bytes == 56 * COLI_MOE_MIB);
}

static void test_no_rss_does_not_fake_total_conversion(void) {
    ColiMoeResourceBudgetInputs in = {
        .available_bytes = 4 * UINT64_C(1073741824),
        .process_resident_bytes = 0,
        .explicit_total_bytes = 1 * UINT64_C(1073741824),
        .resident_bytes_per_expert = 8 * COLI_MOE_MIB,
        .transient_slots = 2,
        .max_persistent_slots = 5,
    };
    ColiMoeResourceBudget out;
    assert(coli_moe_resource_budget_compute(&out, &in) == 0);
    assert(out.used_explicit_total == 0);
    assert(out.persistent_slots == 5);
}

int main(void) {
    test_automatic_bytes_not_model_constants();
    test_explicit_total_is_total_not_expert_budget();
    test_replan_adds_back_current_optional_residency();
    test_automatic_replan_keeps_system_reserve();
    test_explicit_slot_ceiling();
    test_no_rss_does_not_fake_total_conversion();
    puts("generic MoE resource budget: ok");
    return 0;
}

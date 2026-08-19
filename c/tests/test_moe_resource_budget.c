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
    /* Only the 1 GiB remaining under the total process cap can be considered,
     * then prefix + transient reserves are removed. */
    uint64_t ceiling = UINT64_C(1073741824) - 256 * COLI_MOE_MIB -
                       4 * 16 * COLI_MOE_MIB;
    assert(out.persistent_budget_bytes <= ceiling);
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
    test_explicit_slot_ceiling();
    test_no_rss_does_not_fake_total_conversion();
    puts("generic MoE resource budget: ok");
    return 0;
}

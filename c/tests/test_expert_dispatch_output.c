#include "../expert_dispatch.h"

#include <stdio.h>

#define CHECK(expr, code) do { if (!(expr)) return (code); } while (0)

enum {
    OUT_LAYOUT_A = 0x5201,
    OUT_LAYOUT_B = 0x5202,
};

typedef struct {
    int gain;
} FakeExpert;

static ColiRepresentationId output_rep(uint16_t layout, uint16_t kernel_abi) {
    ColiRepresentationId rep = {
        .math_format = COLI_CSF_MATH_MXFP4_E2M1,
        .scale_format = COLI_CSF_SCALE_UE8M0,
        .execution_layout = layout,
        .execution_layout_abi = 1,
        .kernel_abi = kernel_abi,
        .target_class = 0x5236,
        .group_size = 32,
        .scale_block_rows = 1,
        .scale_block_columns = 32,
    };
    return rep;
}

static int publish(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        const ColiRepresentationId *rep,
        FakeExpert *expert,
        uint32_t *variant_id) {
    uint64_t generation = 0;
    ColiExpertRequestResult requested = coli_expert_residency_reserve_variant(
        entry, budget, rep, COLI_EXPERT_TIER_UMA,
        sizeof(*expert), variant_id, &generation);
    if (requested != COLI_EXPERT_REQUEST_LOAD_OWNER) return -1;
    if (coli_expert_residency_publish_variant(
            entry, budget, *variant_id, generation,
            sizeof(*expert), expert) != 0)
        return -1;
    return coli_expert_residency_set_preferred(entry, *variant_id);
}

static int classify(
        void *context,
        const ColiExpertResidentView *view,
        uint32_t activation_contract,
        uint32_t geometry_class,
        uint32_t *backend_class,
        uint32_t *kernel_class) {
    (void)context;
    if (!view || activation_contract != 7 || geometry_class != 3) return 0;
    *backend_class = 1;
    if (view->representation.execution_layout == OUT_LAYOUT_A) {
        *kernel_class = 101;
        return 1;
    }
    if (view->representation.execution_layout == OUT_LAYOUT_B) {
        *kernel_class = 202;
        return 1;
    }
    return 0;
}

int main(void) {
    ColiRepresentationId rep_a = output_rep(OUT_LAYOUT_A, 1);
    ColiRepresentationId rep_b = output_rep(OUT_LAYOUT_B, 2);
    ColiExpertResidencyBudget budget;
    coli_expert_residency_budget_init(&budget, 256);

    ColiExpertResidencyEntry a0, a1, b0;
    CHECK(coli_expert_residency_entry_init(&a0, (ColiExpertKey){1, 7}) == 0 &&
          coli_expert_residency_entry_init(&a1, (ColiExpertKey){1, 20}) == 0 &&
          coli_expert_residency_entry_init(&b0, (ColiExpertKey){1, 15}) == 0, 1);
    FakeExpert experts[] = {{2}, {5}, {3}};
    uint32_t va0, va1, vb0;
    CHECK(publish(&a0, &budget, &rep_a, &experts[0], &va0) == 0 &&
          publish(&a1, &budget, &rep_a, &experts[1], &va1) == 0 &&
          publish(&b0, &budget, &rep_b, &experts[2], &vb0) == 0, 2);

    /* Two routed rows hit a0, two hit b0, and one hits a1. Route order is
     * intentionally interleaved so expert-major grouping must reorder work. */
    ColiExpertRoutedWork work[] = {
        {&a0, 0, 0, 65536, 7, 3},
        {&b0, 0, 1, 32768, 7, 3},
        {&a0, 1, 0, 32768, 7, 3},
        {&a1, 0, 2, 65536, 7, 3},
        {&b0, 1, 1, 65536, 7, 3},
    };
    int input[] = {10, 20};
    int reference[] = {0, 0};
    for (uint32_t r = 0; r < sizeof(work) / sizeof(work[0]); ++r) {
        const FakeExpert *expert = NULL;
        if (work[r].entry == &a0) expert = &experts[0];
        else if (work[r].entry == &a1) expert = &experts[1];
        else expert = &experts[2];
        reference[work[r].routed_row] +=
            (input[work[r].routed_row] * expert->gain *
             work[r].route_weight_q16) / 65536;
    }

    ColiHeterogeneousDispatchPlan plan;
    CHECK(coli_expert_dispatch_plan_build(
              work, sizeof(work) / sizeof(work[0]), classify, NULL,
              COLI_HET_DISPATCH_STRICT, &plan) == COLI_HET_DISPATCH_READY &&
          plan.group_count == 2 && plan.batch_count == 3, 3);

    int actual[] = {0, 0};
    uint32_t specialized_invocations = 0;
    for (uint32_t group = 0; group < plan.group_count; ++group) {
        uint32_t kernel = plan.groups[group].key.kernel_class;
        CHECK(kernel == 101 || kernel == 202, 4);
        specialized_invocations++;
        for (uint32_t batch_index = 0; batch_index < plan.batch_count;
             ++batch_index) {
            ColiExpertDispatchBatch *batch = &plan.batches[batch_index];
            if (batch->group_index != group) continue;
            const FakeExpert *expert = (const FakeExpert *)batch->lease.physical;
            CHECK(expert != NULL, 5);
            if (kernel == 101)
                CHECK(batch->lease.representation.execution_layout ==
                          OUT_LAYOUT_A, 6);
            else
                CHECK(batch->lease.representation.execution_layout ==
                          OUT_LAYOUT_B, 7);
            for (uint32_t j = 0; j < batch->route_count; ++j) {
                uint32_t r = plan.route_order[batch->route_offset + j];
                actual[work[r].routed_row] +=
                    (input[work[r].routed_row] * expert->gain *
                     work[r].route_weight_q16) / 65536;
            }
        }
    }
    CHECK(specialized_invocations == 2 &&
          actual[0] == reference[0] && actual[1] == reference[1], 8);

    coli_expert_dispatch_plan_release(&plan);
    CHECK(atomic_load(&a0.variants[va0].refs) == 0 &&
          atomic_load(&a1.variants[va1].refs) == 0 &&
          atomic_load(&b0.variants[vb0].refs) == 0, 9);

    puts("PASS heterogeneous dispatch reference-equivalent grouped execution");
    return 0;
}

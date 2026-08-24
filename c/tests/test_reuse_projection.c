#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "../reuse_projection.h"

static ColiReuseProjectionConfig cfg(void) {
    return (ColiReuseProjectionConfig){
        .decay_quantum_epochs = 64,
        .planning_horizon_epochs = 256,
        .confidence_mass = 8,
        .max_reuse_weight = UINT64_C(1) << 32,
    };
}

static void test_common_horizon(void) {
    ColiReuseProjectionConfig c = cfg();
    /* steady recent mass 16 over effective window 128 => 32 future uses */
    assert(coli_reuse_project_recent_mass(16, &c) == 32);
    /* Same recent rate expressed with twice the mass/window stays comparable. */
    ColiReuseProjectionConfig wider = c;
    wider.decay_quantum_epochs = 128;
    assert(coli_reuse_project_recent_mass(32, &wider) == 32);
}

static void test_confidence_ramp(void) {
    ColiReuseProjectionConfig c = cfg();
    uint64_t one = coli_reuse_project_recent_mass(1, &c);
    uint64_t four = coli_reuse_project_recent_mass(4, &c);
    uint64_t eight = coli_reuse_project_recent_mass(8, &c);
    assert(one <= four);
    assert(four <= eight);
    assert(eight > 0);
}

static void test_bounds_and_invalid(void) {
    ColiReuseProjectionConfig c = cfg();
    c.max_reuse_weight = 7;
    assert(coli_reuse_project_recent_mass(UINT64_MAX, &c) == 7);
    assert(coli_reuse_project_recent_mass(0, &c) == 0);
    c.decay_quantum_epochs = 0;
    assert(coli_reuse_project_recent_mass(100, &c) == 0);
}

int main(void) {
    test_common_horizon();
    test_confidence_ramp();
    test_bounds_and_invalid();
    puts("generic reuse projection: ok");
    return 0;
}

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../qwen_prefix_cache.h"

int main(void) {
    int8_t kinds[2] = {1, 0};
    QwenPrefixStateView view = {
        .layer_count = 2,
        .layer_is_gdn = kinds,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_t = 64,
        .kv_f16 = 1,
        .gdn_state_elems = 16,
        .gdn_conv_elems = 8,
    };

#ifdef COLI_METAL
    g_metal_compute = 0;
    uint64_t cpu = qwen_prefix_disk_layout_fingerprint(&view);
    assert(cpu != 0);

    /* Same state geometry under a different resolved compute backend must map
     * to a different persistent object namespace. */
    g_metal_compute = 1;
    uint64_t metal = qwen_prefix_disk_layout_fingerprint(&view);
    assert(metal != 0);
    assert(metal != cpu);

    /* Returning to CPU must recover the exact original identity. */
    g_metal_compute = 0;
    assert(qwen_prefix_disk_layout_fingerprint(&view) == cpu);
#else
#error "compile this test with -DCOLI_METAL"
#endif

    /* Existing numerics-affecting chunk identity remains part of the key. */
    assert(setenv("QWENMOE_CHUNK", "32", 1) == 0);
    uint64_t chunk32 = qwen_prefix_disk_layout_fingerprint(&view);
    assert(chunk32 != cpu);
    assert(unsetenv("QWENMOE_CHUNK") == 0);
    assert(qwen_prefix_disk_layout_fingerprint(&view) == cpu);

    puts("qwen prefix backend fingerprint: ok");
    return 0;
}

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../qwen_prefix_cache.h"

#define LAYERS 4
#define KVH 2
#define HD 3
#define MAXT 8
#define GDN_S 5
#define GDN_C 4

typedef struct {
    int8_t kinds[LAYERS];
    float *K[LAYERS], *V[LAYERS];
    uint16_t *K16[LAYERS], *V16[LAYERS];
    float *S[LAYERS], *C[LAYERS];
    QwenPrefixStateView view;
} Fixture;

static void fixture_init(Fixture *f, int kv_f16) {
    memset(f, 0, sizeof(*f));
    f->kinds[0] = 1; f->kinds[1] = 0; f->kinds[2] = 1; f->kinds[3] = 0;
    const size_t kvn = KVH * MAXT * HD;
    for (int l = 0; l < LAYERS; l++) {
        if (f->kinds[l]) {
            f->S[l] = calloc(GDN_S, sizeof(float));
            f->C[l] = calloc(GDN_C, sizeof(float));
            assert(f->S[l] && f->C[l]);
        } else if (kv_f16) {
            f->K16[l] = calloc(kvn, sizeof(uint16_t));
            f->V16[l] = calloc(kvn, sizeof(uint16_t));
            assert(f->K16[l] && f->V16[l]);
        } else {
            f->K[l] = calloc(kvn, sizeof(float));
            f->V[l] = calloc(kvn, sizeof(float));
            assert(f->K[l] && f->V[l]);
        }
    }
    f->view = (QwenPrefixStateView){
        .layer_count = LAYERS,
        .layer_is_gdn = f->kinds,
        .n_kv_heads = KVH,
        .head_dim = HD,
        .max_t = MAXT,
        .kv_f16 = kv_f16,
        .K = f->K, .V = f->V, .K16 = f->K16, .V16 = f->V16,
        .gdn_S = f->S, .gdn_conv = f->C,
        .gdn_state_elems = GDN_S,
        .gdn_conv_elems = GDN_C,
    };
}

static void fixture_free(Fixture *f) {
    for (int l = 0; l < LAYERS; l++) {
        free(f->K[l]); free(f->V[l]); free(f->K16[l]); free(f->V16[l]);
        free(f->S[l]); free(f->C[l]);
    }
}

static void fill_state(Fixture *f, int seed) {
    const size_t kvn = KVH * MAXT * HD;
    for (int l = 0; l < LAYERS; l++) {
        if (f->kinds[l]) {
            for (size_t i = 0; i < GDN_S; i++) f->S[l][i] = (float)(seed * 1000 + l * 100 + (int)i);
            for (size_t i = 0; i < GDN_C; i++) f->C[l][i] = (float)(seed * 2000 + l * 100 + (int)i);
        } else if (f->view.kv_f16) {
            for (size_t i = 0; i < kvn; i++) {
                f->K16[l][i] = (uint16_t)(seed * 100 + l * 10 + (int)i);
                f->V16[l][i] = (uint16_t)(seed * 200 + l * 10 + (int)i);
            }
        } else {
            for (size_t i = 0; i < kvn; i++) {
                f->K[l][i] = (float)(seed * 100 + l * 10 + (int)i);
                f->V[l][i] = (float)(seed * 200 + l * 10 + (int)i);
            }
        }
    }
}

static void assert_prefix_seed(const Fixture *f, int seed, int prefix) {
    for (int l = 0; l < LAYERS; l++) {
        if (f->kinds[l]) {
            for (size_t i = 0; i < GDN_S; i++)
                assert(f->S[l][i] == (float)(seed * 1000 + l * 100 + (int)i));
            for (size_t i = 0; i < GDN_C; i++)
                assert(f->C[l][i] == (float)(seed * 2000 + l * 100 + (int)i));
            continue;
        }
        for (int g = 0; g < KVH; g++) {
            for (int p = 0; p < prefix; p++) {
                for (int d = 0; d < HD; d++) {
                    size_t i = ((size_t)g * MAXT + (size_t)p) * HD + (size_t)d;
                    if (f->view.kv_f16) {
                        assert(f->K16[l][i] == (uint16_t)(seed * 100 + l * 10 + (int)i));
                        assert(f->V16[l][i] == (uint16_t)(seed * 200 + l * 10 + (int)i));
                    } else {
                        assert(f->K[l][i] == (float)(seed * 100 + l * 10 + (int)i));
                        assert(f->V[l][i] == (float)(seed * 200 + l * 10 + (int)i));
                    }
                }
            }
        }
    }
}

static void assert_tail_seed(const Fixture *f, int seed, int prefix) {
    for (int l = 0; l < LAYERS; l++) {
        if (f->kinds[l]) continue;
        for (int g = 0; g < KVH; g++) {
            for (int p = prefix; p < MAXT; p++) {
                for (int d = 0; d < HD; d++) {
                    size_t i = ((size_t)g * MAXT + (size_t)p) * HD + (size_t)d;
                    if (f->view.kv_f16) {
                        assert(f->K16[l][i] == (uint16_t)(seed * 100 + l * 10 + (int)i));
                        assert(f->V16[l][i] == (uint16_t)(seed * 200 + l * 10 + (int)i));
                    } else {
                        assert(f->K[l][i] == (float)(seed * 100 + l * 10 + (int)i));
                        assert(f->V[l][i] == (float)(seed * 200 + l * 10 + (int)i));
                    }
                }
            }
        }
    }
}

static void test_restore(int kv_f16) {
    Fixture f;
    fixture_init(&f, kv_f16);
    QwenPrefixCache c = {0};
    qwen_prefix_cache_init(&c, 1024 * 1024, 1, 0);

    int p3[] = {10, 11, 12};
    int p4[] = {10, 11, 12, 13};
    int p5[] = {10, 11, 12, 13, 14};

    fill_state(&f, 1);
    qwen_prefix_cache_store(&c, &f.view, p3, 3);
    assert(c.count == 1 && c.stores == 1);

    /* Restore must replace only used KV positions, while restoring the whole
     * recurrent GDN state that represents the same three-token prefix. */
    fill_state(&f, 9);
    assert(qwen_prefix_cache_restore(&c, &f.view, p5, 5) == 3);
    assert_prefix_seed(&f, 1, 3);
    assert_tail_seed(&f, 9, 3);

    /* A later/longer exact snapshot wins over the shorter matching entry. */
    fill_state(&f, 2);
    qwen_prefix_cache_store(&c, &f.view, p4, 4);
    fill_state(&f, 8);
    assert(qwen_prefix_cache_restore(&c, &f.view, p5, 5) == 4);
    assert_prefix_seed(&f, 2, 4);
    assert_tail_seed(&f, 8, 4);

    /* Equal prompts are not reusable: step() still needs at least one new token
     * to produce the final logits. Divergent prefixes also miss exactly. */
    assert(qwen_prefix_cache_restore(&c, &f.view, p4, 4) == 3);
    {
        int diverged[] = {10, 11, 99, 13, 14};
        assert(qwen_prefix_cache_restore(&c, &f.view, diverged, 5) == 0);
    }

    QwenPrefixCacheStats s;
    qwen_prefix_cache_stats(&c, &s);
    assert(s.hits == 3);
    assert(s.matched_tokens == 10);
    assert(s.resident_bytes <= s.budget_bytes);
    qwen_prefix_cache_clear(&c);
    fixture_free(&f);
}

static void test_ram_cap_reservation(void) {
    int valid = 0;
    assert(qwen_prefix_cache_ram_cap(NULL, 0, &valid) == 0 && !valid);
    assert(qwen_prefix_cache_ram_cap("garbage", 0, &valid) == 0 && !valid);
    assert(qwen_prefix_cache_ram_cap("4", 0, &valid) == 2 && valid);
    assert(qwen_prefix_cache_ram_cap("4", 1, &valid) == 1 && valid);
    assert(qwen_prefix_cache_ram_cap("4.5", 500000000, &valid) == 2 && valid);
    assert(qwen_prefix_cache_ram_cap("1", 2ULL * 1000000000ULL, &valid) == 0 && valid);
}

static void test_budget_policy(void) {
    const size_t mib = 1024u * 1024u;
    const size_t fallback = 256u * mib;
    assert(qwen_prefix_cache_budget_parse(NULL, fallback) == fallback);
    assert(qwen_prefix_cache_budget_parse("", fallback) == fallback);
    assert(qwen_prefix_cache_budget_parse("0", fallback) == 0);
    assert(qwen_prefix_cache_budget_parse("off", fallback) == 0);
    assert(qwen_prefix_cache_budget_parse("OFF", fallback) == 0);
    assert(qwen_prefix_cache_budget_parse("64", fallback) == 64u * mib);
    assert(qwen_prefix_cache_budget_parse("garbage", fallback) == 0);
    assert(qwen_prefix_cache_budget_parse("64garbage", fallback) == 0);
}

static void test_hard_budget(void) {
    Fixture f;
    fixture_init(&f, 1);
    int a[] = {1, 2, 3};
    int b[] = {4, 5, 6};
    size_t one, kvb, gde;
    assert(qwen_prefix_cache_entry_bytes(&f.view, 3, &one, &kvb, &gde));

    QwenPrefixCache c = {0};
    qwen_prefix_cache_init(&c, one, 1, 0);
    fill_state(&f, 3);
    qwen_prefix_cache_store(&c, &f.view, a, 3);
    assert(c.count == 1 && c.resident_bytes == one);
    fill_state(&f, 4);
    qwen_prefix_cache_store(&c, &f.view, b, 3);
    assert(c.count == 1);
    assert(c.resident_bytes == one);
    assert(c.evictions == 1);
    assert(qwen_prefix_cache_restore(&c, &f.view, (int[]){1,2,3,9}, 4) == 0);
    assert(qwen_prefix_cache_restore(&c, &f.view, (int[]){4,5,6,9}, 4) == 3);

    qwen_prefix_cache_clear(&c);
    fixture_free(&f);
}

int main(void) {
    test_restore(1);
    test_restore(0);
    test_ram_cap_reservation();
    test_budget_policy();
    test_hard_budget();
    puts("qwen prefix cache: ok");
    return 0;
}

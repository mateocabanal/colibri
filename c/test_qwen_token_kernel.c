#include "qwen_token_kernel.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    QwenTokenKernelParams p;
    memset(&p, 0, sizeof(p));
    p.abi_version = QWEN_TOKEN_KERNEL_ABI;
    p.n_layers = 40;
    p.hidden = 2048;
    p.max_t = 4096;
    p.n_heads = 8; p.n_kv_heads = 8; p.head_dim = 128; p.rotary_dim = 64;
    p.lin_k_heads = 8; p.lin_k_dim = 128;
    p.lin_v_heads = 8; p.lin_v_dim = 128;
    p.conv_kernel = 4;
    p.n_experts = 256; p.topk = 8;
    p.moe_inter = 1120; p.shared_inter = 1024;
    p.eps = 1e-6f; p.theta = 10000.0f;

    /* 40 layers: 3 GDN, 1 ATTN pattern, 16 resident slots/layer (matches engine) */
    int gdn = 0;
    for (int i = 0; i < 40; i++) {
        p.layer[i].kind = (i % 4 == 3) ? QWEN_TOKEN_LAYER_ATTN : QWEN_TOKEN_LAYER_GDN;
        if (p.layer[i].kind == QWEN_TOKEN_LAYER_GDN) gdn++;
        p.layer[i].expert_slots = 16;
    }

    QwenTokenDeviceLayout l;
    char err[256];
    if (!qwen_token_device_layout_init(&p, &l, err, sizeof(err))) {
        fprintf(stderr, "LAYOUT FAIL: %s\n", err);
        return 1;
    }

    int nattn = 40 - gdn;
    printf("G=GDN layers=%d  A=ATTN layers=%d\n", gdn, nattn);
    printf("conv bytes/layer= expect 36864  got %llu/layer\n",
           (unsigned long long)(l.gdn_conv_bytes / (gdn?gdn:1)));
    printf("S bytes/layer    expect 524288 got %llu\n",
           (unsigned long long)(l.gdn_s_bytes / (gdn?gdn:1)));
    printf("kv bytes/layer   expect 2*max_t*8*128*4 got %llu\n",
           (unsigned long long)(l.kv_bytes / (nattn?nattn:1)));
    printf("expert_map_bytes %llu (expect %d)\n",
           (unsigned long long)l.expert_map_bytes, 40*16*4);

    /* invariants */
    int fail = 0;
    for (int i = 0; i < 40; i++) {
        if (l.gdn_s_layer_off[i] != QWEN_TOKEN_OFF_NONE) {
            /* S region must never overlap KV region */
            if (l.gdn_s_layer_off[i] >= l.kv_off && l.gdn_s_layer_off[i] < l.kv_off + l.kv_bytes) {
                fprintf(stderr, "FAIL: S aliases KV at layer %d\n", i); fail = 1;
            }
        }
    }
    /* region disjointness: S fully after kv_off */
    if (l.gdn_s_off < l.kv_off + l.kv_bytes) { fprintf(stderr, "FAIL: S region inside KV\n"); fail = 1; }
    if (fail) return 1;

    printf("PASS: layout sane, total=%llu bytes (~%.2f MiB)\n",
           (unsigned long long)l.total_bytes, l.total_bytes/1048576.0);
    return 0;
}
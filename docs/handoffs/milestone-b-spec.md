# Milestone B spec: Metal GDN decode (Qwen3.6-35B-A3B, Apple8 direct path)

Captured 2026-08-22 from the ChatGPT MILESTONEB design (truncated before the
commit could be pushed). Branch: `feat/qwen-decode-milestone-a` (tip 52975f8,
PR #157 open, main untouched). Implement exactly this design, then push.

## Design (as authored)

- Fused BF16 projections (in_proj_a, in_proj_b, in_proj_qkv, in_proj_z).
- Causal-conv state update (conv1d, state in gdn_conv).
- Serial-kd delta recurrence PRESERVING CPU recurrence order (gdn_S update).
- Gated normalization.
- BF16 output projection.
- One Metal command buffer per GDN layer.
- Shared UMA recurrent state (gdn_S + gdn_conv GPU-visible).
- Pre-submit-only CPU fallback.

## Wiring

- Env gate `QWEN_GDN_METAL`, default ON when the direct Apple8 path is active;
  `QWEN_GDN_METAL=0` falls back to the CPU `gdn_token` path.
- Call site: c/qwen_moe.c line ~3371 (`c->layer_is_gdn[l] ? gdn_token(...) : full attention`).
- CPU reference functions to match exactly (float accumulation order):
  `gdn_token` (~2303) and `gdn_token_core` (~2879).

## Geometry (from the loader, qwen_moe.c ~1026)

- A_log, dt_bias, conv1d.weight [C x conv_kernel]
- in_proj_a.weight [v_heads x D], in_proj_b.weight [v_heads x D]
- in_proj_qkv.weight [C x D], in_proj_z.weight [vdim x D]
- gdn_S [layers][v_heads*k_dim*v_dim]; gdn_conv [layers][channels*(conv_kernel-1)]

## Acceptance

- All GDN layers combined < 15 ms/token (today: ~76 ms/token CPU).
- Token parity 40/40 vs the CPU reference (MILESTONE A committed at 52975f8
  is the parity baseline: `ID` output of QWENMOE_MODE=greedy prompt 271, 40
  tokens = `ID 59 2720 ... 3036`).
- Expected when complete: GDN ~8-12 ms/token; decode wall ~6.4-6.6 s / 40 tok;
  ~6.1-6.3 tok/s.

## Bench command (Hermes runs it)

```
cd ~/CODE/colibri && git fetch origin feat/qwen-decode-milestone-a && git switch feat/qwen-decode-milestone-a && git reset --hard origin/feat/qwen-decode-milestone-a && make -C c clean qwen_moe METAL=1 METALIO=1 && (cd c && env RAM_GB=8 QWEN_GDN_METAL=1 QWEN_PROFILE=1 QWENMOE_MODE=greedy QWENMOE_PROMPT_IDS=271 QWENMOE_MAX_NEW=40 QWEN_METAL_COMPUTE=1 QWEN_METAL_IO=1 QWEN_APPLE8_DIRECT=1 ./qwen_moe ~/models/Qwen3.6-35B-A3B.Apple8.raw.coli)
```

## Pending

- Commit: Metal GDN kernel file + qwen_moe.c wiring -> push DIRECTLY to
  feat/qwen-decode-milestone-a (no Actions workflow).
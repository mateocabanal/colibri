# QwenMoE (qwen_moe) — Qwen3.5 / 3.6 / 3.7 MoE engine

CPU engine for the Qwen3.5/3.6/3.7 MoE family (Apache 2.0 open weights):
hybrid **Gated DeltaNet** linear attention + **Gated Attention** (GQA) layers,
with a fine-grained MoE FFN (softmax top-k router, sigmoid-gated shared
expert) after every layer. See `docs/qwen-moe-config-census.md` for the
ground-truth config/weight census (retrieved from HF, 2026-08-13).

## Building

    make -C c qwen_moe

CPU-only (no CUDA/Vulkan/Metal), same build class as `olmoe` (#783).

## Preparing a checkpoint

    python c/tools/convert_qwen_moe.py <hf_model_dir> --out <snap> [--bits 8|32]

`--bits 8` (default) writes the **disk-streaming** layout: dense weights
(embed, norms, attention/GDN, router, shared expert, lm_head) as f32, and
each routed expert as an int8-packed `merged_weight` + f32 row scales
(1 byte/param). `--bits 32` writes f32 experts for oracle-diff runs.
`c/tools/make_qwen_moe_tiny.py` generates a tiny hybrid-layout fixture with a
transformers oracle (`ref.json`) — the token-exact regression harness.

## Running

    c/qwen_moe <snap>                 # self-test every ref.json case
    coli chat --model <snap>          # via the launcher (needs tokenizer.json)

Environments:

| Var | Meaning | Default |
|---|---|---|
| `CACHE` / argv[2] | experts resident per layer (LRU) | 16 (or `RAM_GB/2`) |
| `RAM_GB` | RAM budget; sizes the expert cache (`coli --ram`) | — |
| `HOT`, `WARMUP` | pin top-N hot experts per layer after WARMUP tokens | 0 / 5 |
| `EXPERT_DROP=1` | fadvise(DONTNEED) after expert reads (RAM-tight boxes) | 0 |
| `CTX` | KV context capacity | 4096 |
| `COLI_USAGE` | expert history file (`.coli_usage` format) | — |
| `ROUTE_TRACE` | routing trace stream | — |
| `QWENMOE_MODE` | `teacher` / `greedy` (oracle harness modes) | — |

## How expert streaming works

Routed experts are **never fully resident**. `expert_get()` looks up a
per-layer LRU cache; on a miss it `pread`s one per-expert tensor pair from
disk outside the cache lock (exact-size checked — hostile-container guard),
dequantizes on use in the matmul (`int8 + per-row f32 scale`), and evicts
least-recently-used (never pinned/in-flight) slots. HOT pinning ranks experts
from the `route_trace.h` heatmap and makes them never-evict; `COLI_USAGE`
persists the ranking between sessions. A 397B checkpoint therefore needs only
dense (~2 GB) + cache-sized expert memory.

## Math sources of truth

All formulas were ported from `transformers.Qwen3_5MoeForCausalLM` (5.15.0)
and validated **token-exact** against it on the tiny fixture (teacher-forced
logits AND greedy decode, f32 and int8-expert snapshots). Notable details:

- RMSNorm uses `(1 + weight)` with zero-initialized weights.
- Full attention: `q_proj` emits 2×head_dim per head (`q | gate` blocks);
  `attn_output * sigmoid(gate)` elementwise over head_dim; QK-norm per head;
  partial RoPE over `partial_rotary_factor * head_dim` dims.
- GDN: causal depthwise conv1d (silu) → `in_proj_qkv` split → per-value-head
  delta rule `S = γS + k(v − kᵀS)β`, `out = Sᵀq`, per-head gated RMSNorm
  (weight is `[head_v_dim]`), with QK l2norm (eps 1e-6) and `1/√k_dim` scale.
- MoE: router softmax → top-k by probability → renormalize; fused
  `gate_up_proj` (gate = first half); shared expert gated by
  `sigmoid(shared_expert_gate)`.

## Status

- Qwen3.5-35B-A3B / Qwen3.6-35B-A3B / Qwen3.5-397B-A17B supported via
  config-driven `layer_types`; Qwen3.7 (same family) awaits open weights.
- Qwen3 original MoE (128E/top8, no shared expert) is a config subset.
- Not yet: chunked GDN prefill (per-token recurrence is correct but slower),
  YaRN long-context tuning, MTP-native speculation, GPU backends.

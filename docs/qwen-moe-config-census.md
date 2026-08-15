# QwenMoE Config & Weight Census

Ground truth for the `qwen_moe` engine. Retrieved 2026-08-13 from Hugging Face
`resolve/main` (config.json + model.safetensors.index.json) for:
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.5-397B-A17B`

All three use `model_type: "qwen3_5_moe"` (even Qwen3.6!) with architecture
`Qwen3_5MoeForConditionalGeneration` — a vision-language wrapper whose text
backbone config lives in `text_config`. `transformers_version`: 4.57.0.dev0 /
4.57.1 → **oracle requires transformers >= 4.57.1**.

## Shared text_config facts (all three)

| Key | 3.5-35B-A3B | 3.6-35B-A3B | 3.5-397B-A17B |
|---|---|---|---|
| hidden_size | 2048 | 2048 | 4096 |
| num_hidden_layers | 40 | 40 | 60 |
| num_attention_heads | 16 | 16 | 32 |
| num_key_value_heads | 2 | 2 | 2 |
| head_dim | 256 | 256 | 256 |
| vocab_size | 248320 | 248320 | 248320 |
| max_position_embeddings | 262144 | 262144 | 262144 |
| num_experts | 256 | 256 | 512 |
| num_experts_per_tok | 8 | 8 | 10 |
| moe_intermediate_size | 512 | 512 | 1024 |
| shared_expert_intermediate_size | 512 | 512 | 1024 |
| linear_num_key_heads | 16 | 16 | 16 |
| linear_key_head_dim | 128 | 128 | 128 |
| linear_num_value_heads | 32 | 32 | 64 |
| linear_value_head_dim | 128 | 128 | 128 |
| linear_conv_kernel_dim | 4 | 4 | 4 |
| rms_norm_eps | 1e-6 | 1e-6 | 1e-6 |
| hidden_act | silu | silu | silu |
| attn_output_gate | true | true | true |
| mamba_ssm_dtype | float32 | float32 | float32 |
| mtp_num_hidden_layers | 1 | 1 | 1 |
| mtp_use_dedicated_embeddings | false | false | false |
| tie_word_embeddings | false | false | false |
| full_attention_interval | 4 | 4 | 4 |
| mlp_only_layers | [] (3.5 only key) | absent | [] (3.5 only key) |
| dtype | bfloat16 | bfloat16 | bfloat16 |
| attention_bias | false | false | false |

### Layer layout: `layer_types` array (explicit)

Every layer listed individually. Pattern for 40 layers (35B):
`[linear, linear, linear, full]` × 10. For 60 layers (397B): `[linear × 3, full]` × 15.
- `"linear_attention"` → Gated DeltaNet block (Mamba-style, see below)
- `"full_attention"` → GQA attention block (self_attn)

`full_attention_interval: 4` is the repetition period; `layer_types` is the
authoritative per-layer list. Engine derives `layer_is_gdn[l]` from it.

### RoPE (mRoPE, in `rope_parameters`)

- `rope_theta: 10000000`, `rope_type: "default"` (no YaRN scaling in these checkpoints)
- `partial_rotary_factor: 0.25` → rotary dims = 0.25 × head_dim = **64** of 256
- `mrope_interleaved: true`, `mrope_section: [11, 11, 10]`
- Text-only implication (to confirm against transformers source): text tokens
  get mRoPE positions; engine must replicate the text-only branch exactly.

### Special ids (35B; 397B same shape)

bos = eos = 248044; image_token 248056; video_token 248057; vision_start 248053;
vision_end 248054; pad = null. Vocab 248320 is o200k base + padding (base ~151k
+ added specials; ids 151643..248319 must decode but never be sampled).

## Weight names (from index.json weight_map)

Prefix: **`model.language_model.`** for the text backbone. Exception:
**`lm_head.weight` has no prefix.** Vision weights live under `model.visual.*`
(ignore entirely). MTP under top-level `mtp.*` (skip in v1).

### Global tensors
- `model.language_model.embed_tokens.weight` [vocab, hidden] bf16
- `model.language_model.norm.weight` [hidden] (final RMSNorm)
- `lm_head.weight` [vocab, hidden] bf16

### Per layer N — linear_attention (GDN) layer (`linear_attn.*`)
- `model.language_model.layers.N.input_layernorm.weight`
- `model.language_model.layers.N.linear_attn.A_log`
- `model.language_model.layers.N.linear_attn.dt_bias`
- `model.language_model.layers.N.linear_attn.conv1d.weight`  (kernel dim 4)
- `model.language_model.layers.N.linear_attn.in_proj_a.weight`
- `model.language_model.layers.N.linear_attn.in_proj_b.weight`
- `model.language_model.layers.N.linear_attn.in_proj_qkv.weight`
- `model.language_model.layers.N.linear_attn.in_proj_z.weight`
- `model.language_model.layers.N.linear_attn.norm.weight`
- `model.language_model.layers.N.linear_attn.out_proj.weight`
- `model.language_model.layers.N.post_attention_layernorm.weight`

(Mamba-family naming: A_log/dt_bias/conv1d + in_proj_a/b/qkv/z + norm + out_proj.
This is a Mamba-2-style Gated DeltaNet. The transformers reference is
`Qwen3_5GatedDeltaNet`; the exact recurrence must be ported from it, not guessed.)

### Per layer N — full_attention layer (`self_attn.*`)
- `model.language_model.layers.N.input_layernorm.weight`
- `model.language_model.layers.N.self_attn.q_proj.weight` [hidden, hidden]
- `model.language_model.layers.N.self_attn.k_proj.weight` [kv*hd, hidden]
- `model.language_model.layers.N.self_attn.v_proj.weight` [kv*hd, hidden]
- `model.language_model.layers.N.self_attn.o_proj.weight` [hidden, hidden]
- `model.language_model.layers.N.self_attn.q_norm.weight` (QK-norm ✓, no v_norm)
- `model.language_model.layers.N.self_attn.k_norm.weight`
- `model.language_model.layers.N.post_attention_layernorm.weight`

`attn_output_gate: true` → the attention output is additionally gated; the
gating parameter name and formula must be read from the transformers source.

### MoE FFN (every layer, both types)
- `model.language_model.layers.N.mlp.gate.weight` — router [num_experts, hidden]
- `model.language_model.layers.N.mlp.experts.gate_up_proj` — **fused**, bf16,
  shape [num_experts, 2*moe_inter, hidden] (NO `.weight` suffix in the key)
- `model.language_model.layers.N.mlp.experts.down_proj` — **fused**, bf16,
  shape [num_experts, hidden, moe_inter] (NO `.weight` suffix)
- `model.language_model.layers.N.mlp.shared_expert.gate_proj.weight` [shared_inter, hidden]
- `model.language_model.layers.N.mlp.shared_expert.up_proj.weight`   [shared_inter, hidden]
- `model.language_model.layers.N.mlp.shared_expert.down_proj.weight` [hidden, shared_inter]
- `model.language_model.layers.N.mlp.shared_expert_gate.weight` — **shared expert is gated**;
  formula from transformers source (expect sigmoid gate, exact form to verify)

Converter implication: Qwen ships per-layer fused expert tensors; the colibri
snapshot must split them into per-expert `experts.E.merged_weight` (int8
gate|up|down packed, olmoe byte layout) + `.qs` scales.

### MTP (skipped in v1, names recorded)
- `mtp.fc.weight`, `mtp.layers.N.mlp.*` (same MoE structure as main layers),
  `mtp.layers.N.input_layernorm.weight`. Qwen3.6-35B MTP experts also fused
  (gate_up_proj without `.weight`); Qwen3.5-35B MTP has per-expert
  `gate_proj/up_proj/down_proj.weight`.

## Open items for the transformers source (Phase 2/4 reads)
1. `attn_output_gate` formula and parameter name (self_attn has only 4 projs + 2 norms — gate may reuse a norm or be part of o_proj layout; verify).
2. `shared_expert_gate` application (sigmoid? multiply before/after residual).
3. GatedDeltaNet exact recurrence + conv handling + `norm` placement.
4. mRoPE text-only position/frequency application.
5. Router: softmax + top-k + `norm_topk_prob` (no such key here — confirm default off in Qwen3.5).
6. Expert gate_up_proj: is it [gate; up] or [up; gate] row order? (converter must match.)

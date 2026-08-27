#!/usr/bin/env python3
"""Generate the deterministic tiny Qwen4 (qwen4_exp hybrid MoE) fixture.

This generator does NOT use transformers: the Qwen4 class is not in any
stable transformers release yet, so the fixture weights are drawn with a
fixed torch seed and an HF-style init, and the reference tokens are produced
by tools/qwen4_oracle.py — a pure numpy/torch implementation of the pinned
M1 formulas (transformers 5.57.1 qwen4_exp modular source). f32 everywhere;
the generated safetensors are never committed.

Usage: python tools/make_qwen4_tiny.py [--output c/qwen4_moe_tiny] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

SEED = 42
SCHEMA_VERSION = 1
GENERATOR_VERSION = "1"

# ---- tiny geometry (Qwen4 text_config shapes, shrunk) ----
VOCAB = 500
HIDDEN = 64
LAYERS = 2
LAYER_TYPES = ["linear_attention", "full_attention"]
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
EXPERTS = 8
TOPK = 2
MOE_INTER = 64
SHARED_INTER = 64
LIN_K_HEADS = 2
LIN_K_DIM = 8
LIN_V_HEADS = 4
LIN_V_DIM = 8
CONV_KERNEL = 3
HC_COUNT = 2
HC_LOWRANK = 8
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 2
NGRAM_VOCAB_BASE = 1000
NGRAM_DIVISOR = 8
PLE_EMBED_DIM = 64
PLE_CONV_KERNEL = 3
INDEX_N_HEADS = 2
INDEX_KV_HEADS = 1
INDEX_HEAD_DIM = 8
INDEX_BUDGET = 64
INDEX_RATIO = 4
PARTIAL_ROTARY = 0.5          # rope dim = int(head_dim * partial) = 8
ROPE_THETA = 10000.0
MAX_POS = 4096
BOS = 1
EOS = 499
RMS_EPS = 1e-6
INIT_RANGE = 0.02


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, math.isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


def ngram_geometry():
    ngram_heads = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
    sizes = [nth_prime_after(NGRAM_VOCAB_BASE - 1, h + 1) for h in range(ngram_heads)]
    total = sum(sizes)
    padded = math.ceil(total / NGRAM_DIVISOR) * NGRAM_DIVISOR
    return ngram_heads, sizes, total, padded


def build_multipliers(vocab: int) -> list[int]:
    max_long = (1 << 63) - 1
    mult_max = max_long // max(vocab, 1)
    half_bound = max(1, mult_max // 2)
    base_seed = SEED + 10007 * 0
    gamma = 0x9E3779B97F4A7C15
    mask64 = (1 << 64) - 1
    m1 = 0xBF58476D1CE4E5B9
    m2 = 0x94D049BB133111EB

    def splitmix64(value: int) -> int:
        value = (value + gamma) & mask64
        value = ((value ^ (value >> 30)) * m1) & mask64
        value = ((value ^ (value >> 27)) * m2) & mask64
        return (value ^ (value >> 31)) & mask64

    out = []
    for i in range(NGRAM_SIZE):
        value = (base_seed + gamma * (i + 1)) & mask64
        out.append(2 * (splitmix64(value) % half_bound) + 1)
    return out


def make_state() -> dict[str, torch.Tensor]:
    """HF-name state dict built with torch.manual_seed(SEED) in fixed order.

    HF-style init: Linear/Embedding -> normal(0, initializer_range),
    RMSNorm weight -> zeros (formula is (1+w)), RMSNormGated weight -> ones,
    GDN dt_bias -> ones, A_log -> log(uniform(0.01,16)), PLE conv -> zeros.
    """
    import torch
    torch.manual_seed(SEED)
    D, HCD = HIDDEN, HC_COUNT * HIDDEN
    V = VOCAB
    kdim = LIN_K_DIM * LIN_K_HEADS
    vdim = LIN_V_DIM * LIN_V_HEADS
    C = kdim * 2 + vdim
    st: dict[str, torch.Tensor] = {}

    def lin(name, out_f, in_f):
        st[name] = torch.randn(out_f, in_f) * INIT_RANGE

    def rms(name, dim):
        st[name] = torch.zeros(dim)

    def rmsg(name, dim):
        st[name] = torch.ones(dim)

    # embedding
    st["model.embed_tokens.weight"] = torch.randn(V, D) * INIT_RANGE
    # --- layer 0: linear_attention (GDN) ---
    st["model.layers.0.linear_attn.in_proj_qkv.weight"] = torch.randn(C, D) * INIT_RANGE
    st["model.layers.0.linear_attn.in_proj_z.weight"] = torch.randn(vdim, D) * INIT_RANGE
    st["model.layers.0.linear_attn.in_proj_a.weight"] = torch.randn(LIN_V_HEADS, D) * INIT_RANGE
    st["model.layers.0.linear_attn.in_proj_b.weight"] = torch.randn(LIN_V_HEADS, D) * INIT_RANGE
    st["model.layers.0.linear_attn.conv1d.weight"] = torch.randn(C, 1, CONV_KERNEL) * INIT_RANGE
    st["model.layers.0.linear_attn.A_log"] = torch.log(
        torch.empty(LIN_V_HEADS).uniform_(0.01, 16.0))
    st["model.layers.0.linear_attn.dt_bias"] = torch.ones(LIN_V_HEADS)
    rmsg("model.layers.0.linear_attn.norm.weight", LIN_V_DIM)
    st["model.layers.0.linear_attn.out_proj.weight"] = torch.randn(D, vdim) * INIT_RANGE
    # --- layer 1: full attention + QSA indexer ---
    st["model.layers.1.self_attn.q_proj.weight"] = torch.randn(HEADS * HEAD_DIM * 2, D) * INIT_RANGE
    st["model.layers.1.self_attn.k_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, D) * INIT_RANGE
    st["model.layers.1.self_attn.v_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, D) * INIT_RANGE
    st["model.layers.1.self_attn.o_proj.weight"] = torch.randn(HEADS * HEAD_DIM, D) * INIT_RANGE
    rmsg("model.layers.1.self_attn.q_norm.weight", HEAD_DIM)
    rmsg("model.layers.1.self_attn.k_norm.weight", HEAD_DIM)
    st["model.layers.1.self_attn.indexer.index_qk_proj.weight"] = torch.randn(
        (INDEX_N_HEADS + INDEX_KV_HEADS) * INDEX_HEAD_DIM, D) * INIT_RANGE
    rmsg("model.layers.1.self_attn.indexer.q_layernorm.weight", INDEX_HEAD_DIM)
    rmsg("model.layers.1.self_attn.indexer.k_layernorm.weight", INDEX_HEAD_DIM)

    for layer in range(LAYERS):
        for tag in ("attn_hyper_connection", "mlp_hyper_connection"):
            p = f"model.layers.{layer}.{tag}"
            rms(p + ".hc_norm.weight", HCD)
            lin(p + ".input_mix_weight_down.weight", HC_LOWRANK, HCD)
            lin(p + ".input_mix_weight_up.weight", HCD, HC_LOWRANK)
            lin(p + ".block_inject_weight.weight", HC_COUNT, HCD)
        lin(f"model.layers.{layer}.mlp.gate.weight", EXPERTS, D)
        for e in range(EXPERTS):
            lin(f"model.layers.{layer}.mlp.experts.{e}.gate_up_proj", 2 * MOE_INTER, D)
            lin(f"model.layers.{layer}.mlp.experts.{e}.down_proj", D, MOE_INTER)
        lin(f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight", SHARED_INTER, D)
        lin(f"model.layers.{layer}.mlp.shared_expert.up_proj.weight", SHARED_INTER, D)
        lin(f"model.layers.{layer}.mlp.shared_expert.down_proj.weight", D, SHARED_INTER)
        lin(f"model.layers.{layer}.mlp.shared_expert_gate.weight", 1, D)
        # deterministic per-layer sinusoid router (wide margins, fixed top-2)
        vals = 0.025 * torch.sin(torch.arange(EXPERTS, dtype=torch.float32) * 0.017 + layer)
        st[f"model.layers.{layer}.mlp.gate.weight"].copy_(vals.reshape(EXPERTS, 1).expand(EXPERTS, D).clone())

    # --- PLE (single layer, ple_layer_index 0) ---
    ngram_heads, sizes, total, padded = ngram_geometry()
    head_dim_per_ngram = PLE_EMBED_DIM // ngram_heads
    st["model.ple.ple_embedding.ngram_embedding.weight"] = torch.randn(padded, head_dim_per_ngram) * INIT_RANGE
    lin("model.ple.key_proj.weight", HCD, PLE_EMBED_DIM)
    lin("model.ple.value_proj.weight", D, PLE_EMBED_DIM)
    rms("model.ple.norm_key.weight", HCD)
    rms("model.ple.norm_query.weight", HCD)
    rms("model.ple.norm_conv.weight", HCD)
    st["model.ple.conv1d.weight"] = torch.zeros(HCD, 1, PLE_CONV_KERNEL)

    # --- global mixer (use_combine=False: NO block_inject_weight) ---
    rms("model.hyper_connection_mixer.hc_norm.weight", HCD)
    lin("model.hyper_connection_mixer.input_mix_weight_down.weight", HC_LOWRANK, HCD)
    lin("model.hyper_connection_mixer.input_mix_weight_up.weight", HCD, HC_LOWRANK)

    # --- lm_head: cyclic shift of embeddings, EOS zeroed (wide margins) ---
    st["lm_head.weight"] = torch.zeros(V, D)
    for t in range(V):
        st["lm_head.weight"][(t + 1) % V] = st["model.embed_tokens.weight"][t].clone()
    st["lm_head.weight"][EOS].zero_()
    return st


def runtime_config() -> dict:
    return {
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp_text",
        "torch_dtype": "float32",
        "transformers_version": "5.57.1",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "layer_types": LAYER_TYPES,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_experts": EXPERTS,
        "num_experts_per_tok": TOPK,
        "moe_intermediate_size": MOE_INTER,
        "shared_expert_intermediate_size": SHARED_INTER,
        "norm_topk_prob": True,
        "linear_num_key_heads": LIN_K_HEADS,
        "linear_key_head_dim": LIN_K_DIM,
        "linear_num_value_heads": LIN_V_HEADS,
        "linear_value_head_dim": LIN_V_DIM,
        "linear_conv_kernel_dim": CONV_KERNEL,
        "hc_count": HC_COUNT,
        "hc_lowrank": HC_LOWRANK,
        "ngram_size": NGRAM_SIZE,
        "heads_per_ngram": HEADS_PER_NGRAM,
        "ngram_vocab_size_base": NGRAM_VOCAB_BASE,
        "make_ngram_vocab_size_divisible_by": NGRAM_DIVISOR,
        "ple_embed_dim": PLE_EMBED_DIM,
        "ple_conv_kernel_size": PLE_CONV_KERNEL,
        "ple_layer_ids": [1],
        "indexer_n_heads": INDEX_N_HEADS,
        "indexer_kv_heads": INDEX_KV_HEADS,
        "indexer_head_dim": INDEX_HEAD_DIM,
        "indexer_budget": INDEX_BUDGET,
        "indexer_compress_ratio": INDEX_RATIO,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rms_norm_eps": RMS_EPS,
        "initializer_range": INIT_RANGE,
        "max_position_embeddings": MAX_POS,
        "rope_parameters": {
            "rope_theta": ROPE_THETA,
            "rope_type": "default",
            "partial_rotary_factor": PARTIAL_ROTARY,
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 2],
        },
        "bos_token_id": BOS,
        "eos_token_id": EOS,
        "tie_word_embeddings": False,
        "output_router_logits": False,
        "seed": SEED,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    }


def make_tokenizer() -> dict:
    added = [
        {
            "id": token,
            "content": f"<t{token:03d}>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": token == EOS,
        }
        for token in range(VOCAB)
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": {"x": VOCAB - 1},
            "merges": [],
        },
    }


def emit_engine_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split fused per-expert 3D tensors into per-expert 2D tensors (the
    streamable unit the qwen_moe engine reads)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        m = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_up_proj|down_proj)$", k)
        if m:
            out[f"model.layers.{m.group(1)}.mlp.experts.{m.group(2)}.{m.group(3)}"] = v
        else:
            out[k] = v
    return out


def make_reference() -> dict:
    """Case skeletons: the oracle fills in expected ids (ref.json is the
    contract; oracle recomputes and verifies it)."""
    prompts = {
        "short": [5, 7, 9, 11, 13, 17, 19, 23],
        "mixed": [5, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        "long": [5 + (i * 11) % 97 for i in range(48)],
    }
    max_new = {"short": 8, "mixed": 6, "long": 4}
    cases = {}
    for name, prompt in prompts.items():
        cases[name] = {
            "prompt_ids": prompt,
            "max_new_tokens": max_new[name],
            "teacher_forcing_ids": [],
            "greedy_full_ids": [],
            "greedy_new_ids": [],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "source": "qwen4_oracle.py (pinned M1 formulas, transformers 5.57.1 qwen4_exp)",
        "dtype": "f32 params, f32 oracle math",
        "config_summary": {
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "layer_types": LAYER_TYPES,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "n_experts": EXPERTS,
            "num_experts_per_tok": TOPK,
            "moe_intermediate_size": MOE_INTER,
            "shared_expert_intermediate_size": SHARED_INTER,
        },
        "prompt_ids_short": prompts["short"],
        "prompt_ids_mixed": prompts["mixed"],
        "prompt_ids_long": prompts["long"],
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "qwen4_moe_tiny"
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    import torch  # noqa: F401  (import check)

    output = args.output.resolve() if args.output else default
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists (use --force): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    state = emit_engine_tensors(make_state())
    (output / "config.json").write_text(
        json.dumps(runtime_config(), indent=2) + "\n", encoding="utf-8")
    (output / "tokenizer.json").write_text(
        json.dumps(make_tokenizer(), separators=(",", ":")) + "\n", encoding="utf-8")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in state.items()},
        str(output / "model.safetensors"))
    # case skeletons; expected ids are produced by tools/qwen4_oracle.py
    (output / "ref.json").write_text(
        json.dumps(make_reference(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"  tensors: {len(state)}")
    print("  run tools/qwen4_oracle.py qwen4_moe_tiny to fill ref.json + fixture_oracle_ids.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

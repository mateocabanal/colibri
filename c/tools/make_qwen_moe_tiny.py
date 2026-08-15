#!/usr/bin/env python3
"""Generate the deterministic tiny QwenMoE (Qwen3.5/3.6 hybrid MoE) target oracle fixture.

The tiny fixture exercises the full hybrid architecture the qwen_moe engine
must support: Gated DeltaNet (linear_attention) layers + Gated Attention
(full_attention) layers + fine-grained MoE with a gated shared expert, fused
gate_up_proj expert storage, QK-norm, partial rotary embedding and a padded
o200k-style tokenizer.

Reference tokens always come from transformers' Qwen3_5MoeForCausalLM
(requires transformers >= 4.57.1). The generated safetensors are never
committed. Weight dtypes are kept native (bf16/f32 as the checkpoint ships
them) so a f32-reading C engine sees bit-identical values.

Usage: python tools/make_qwen_moe_tiny.py [--output c/qwen_moe_tiny] [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SEED = 20260813
SCHEMA_VERSION = 1
GENERATOR_VERSION = "1"

# ---- tiny geometry (mirrors Qwen3.6-35B-A3B text_config shapes, shrunk) ----
VOCAB = 320
HIDDEN = 64
LAYERS = 4                 # layer_types below exercises both attention kinds
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16              # 64/4; partial_rotary 0.25 -> 4 rotary dims
EXPERTS = 8
TOPK = 2
MOE_INTER = 32
SHARED_INTER = 32
LIN_QK_HEADS = 2
LIN_QK_DIM = 8
LIN_V_HEADS = 4
LIN_V_DIM = 8
CONV_KERNEL = 3
MAX_POS = 4096
BOS = 0
EOS = 1
LAYER_TYPES = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]


def require_dependencies():
    try:
        import torch
        import transformers
        from transformers import Qwen3_5MoeConfig
        from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
    except Exception as exc:  # pragma: no cover - regeneration-only diagnostic
        raise SystemExit(
            "QwenMoE tiny generation requires PyTorch and transformers >= 4.57.1 "
            "with Qwen3_5MoeForCausalLM"
        ) from exc
    if not hasattr(transformers, "Qwen3_5MoeConfig"):
        raise SystemExit(
            f"Transformers {transformers.__version__} has no Qwen3.5 MoE support "
            "(need >= 4.57.1)"
        )
    return torch, transformers, Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM


def tiny_config_dict() -> dict:
    """text_config-shaped kwargs, shrunk; keys copied from the real
    Qwen/Qwen3.6-35B-A3B config.json text_config (see docs/qwen-moe-config-census.md)."""
    return dict(
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        bos_token_id=BOS,
        eos_token_id=EOS,
        head_dim=HEAD_DIM,
        hidden_act="silu",
        hidden_size=HIDDEN,
        initializer_range=0.02,
        layer_types=LAYER_TYPES,
        linear_conv_kernel_dim=CONV_KERNEL,
        linear_key_head_dim=LIN_QK_DIM,
        linear_num_key_heads=LIN_QK_HEADS,
        linear_num_value_heads=LIN_V_HEADS,
        linear_value_head_dim=LIN_V_DIM,
        mamba_ssm_dtype="float32",
        max_position_embeddings=MAX_POS,
        moe_intermediate_size=MOE_INTER,
        mtp_num_hidden_layers=0,
        mtp_use_dedicated_embeddings=False,
        num_attention_heads=HEADS,
        num_experts=EXPERTS,
        num_experts_per_tok=TOPK,
        num_hidden_layers=LAYERS,
        num_key_value_heads=KV_HEADS,
        output_router_logits=False,
        pad_token_id=None,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-06,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 2],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
        router_aux_loss_coef=0.001,
        shared_expert_intermediate_size=SHARED_INTER,
        tie_word_embeddings=False,
        use_cache=True,
        vocab_size=VOCAB,
    )


def make_hf_config(Qwen3_5MoeConfig):
    return Qwen3_5MoeConfig.from_dict(tiny_config_dict())


def make_runtime_config(transformers_version: str) -> dict:
    """The config.json the C engine parses: the same flat text_config shape the
    real checkpoints carry inside text_config (the engine descends into
    text_config when present)."""
    d = tiny_config_dict()
    d.update({
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe_text",
        "transformers_version": transformers_version,
        "torch_dtype": "bfloat16",
    })
    return d


def initialize_deterministic(torch, model) -> None:
    """Wide, deterministic margins so a f32 C engine reproduces the bf16 oracle:
    lm_head rows follow a cyclic shift of the embeddings, router gates are a
    per-layer sinusoid, and EOS is deliberately unattractive (a fixed-length
    regression must detect early truncation instead of accepting a prefix)."""
    with torch.no_grad():
        embeddings = model.model.embed_tokens.weight.detach().clone()
        for token in range(VOCAB):
            model.lm_head.weight[(token + 1) % VOCAB].copy_(embeddings[token])
        model.lm_head.weight[EOS].zero_()
        for layer_id, layer in enumerate(model.model.layers):
            gate = layer.mlp.gate
            values = torch.arange(gate.weight.numel(), dtype=torch.float32)
            gate.weight.copy_((0.025 * torch.sin(values * 0.017 + layer_id)).reshape_as(gate.weight))
        # bf16 round-trip every parameter so C loads exactly what the oracle
        # computed with (bf16 mantissa -> f32 is exact).
        for param in model.parameters():
            param.copy_(param.detach().to(torch.bfloat16).float())


def greedy_reference(torch, model, prompt: list[int], max_new: int):
    sequence = list(prompt)
    with torch.no_grad():
        for _ in range(max_new):
            inputs = torch.tensor([sequence], dtype=torch.long)
            logits = model(input_ids=inputs, use_cache=False).logits[0, -1]
            sequence.append(int(logits.argmax()))
        full = torch.tensor([sequence], dtype=torch.long)
        teacher = model(input_ids=full, use_cache=False).logits[0].argmax(-1).tolist()
    generated = sequence[len(prompt):]
    if len(generated) != max_new or EOS in generated:
        raise RuntimeError(
            f"reference generation truncated or produced EOS: prompt={prompt} generated={generated}"
        )
    return sequence, teacher


def make_reference(torch, transformers, model) -> dict:
    prompts = {
        "short": [5, 7, 9, 11, 13, 17, 19, 23],
        "mixed": [5, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        "long": [5 + (index * 11) % 97 for index in range(48)],
    }
    max_new = {"short": 8, "mixed": 6, "long": 4}
    cases = {}
    for name, prompt in prompts.items():
        full, teacher = greedy_reference(torch, model, prompt, max_new[name])
        cases[name] = {
            "prompt_ids": prompt,
            "teacher_forcing_ids": teacher,
            "greedy_full_ids": full,
            "greedy_new_ids": full[len(prompt):],
            "max_new_tokens": max_new[name],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "source": "transformers",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": "native bf16/f32 params, bf16 round-tripped",
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


def make_tokenizer() -> dict:
    added = [
        {
            "id": token,
            "content": f"<t{token:03d}>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
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


def emit_engine_tensors(torch, state):
    """Engine-layout snapshot: dense tensors under their engine names, and
    PER-EXPERT tensors (the streamable unit the qwen_moe LRU cache reads).
    The real-checkpoint fused `mlp.experts.gate_up_proj` / `down_proj` are split
    here exactly like tools/convert_qwen_moe.py will for production weights.
    f32 values already bf16-round-tripped by initialize_deterministic."""
    from collections import OrderedDict

    out: OrderedDict[str, object] = OrderedDict()
    out["model.embed_tokens.weight"] = state["model.embed_tokens.weight"]
    out["model.norm.weight"] = state["model.norm.weight"]
    out["lm_head.weight"] = state["lm_head.weight"]
    for layer_id in range(LAYERS):
        hf = f"model.layers.{layer_id}"
        out[f"{hf}.input_layernorm.weight"] = state[f"{hf}.input_layernorm.weight"]
        if LAYER_TYPES[layer_id] == "full_attention":
            for key in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                out[f"{hf}.self_attn.{key}.weight"] = state[f"{hf}.self_attn.{key}.weight"]
        else:
            for key in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "norm", "out_proj"):
                out[f"{hf}.linear_attn.{key}.weight"] = state[f"{hf}.linear_attn.{key}.weight"]
            # A_log / dt_bias are bare Parameters in the checkpoint (no .weight)
            out[f"{hf}.linear_attn.A_log"] = state[f"{hf}.linear_attn.A_log"]
            out[f"{hf}.linear_attn.dt_bias"] = state[f"{hf}.linear_attn.dt_bias"]
            out[f"{hf}.linear_attn.conv1d.weight"] = state[f"{hf}.linear_attn.conv1d.weight"]
        out[f"{hf}.post_attention_layernorm.weight"] = state[f"{hf}.post_attention_layernorm.weight"]
        out[f"{hf}.mlp.gate.weight"] = state[f"{hf}.mlp.gate.weight"]
        gate_up = state[f"{hf}.mlp.experts.gate_up_proj"]
        down = state[f"{hf}.mlp.experts.down_proj"]
        for expert in range(EXPERTS):
            out[f"{hf}.mlp.experts.{expert}.gate_up_proj"] = gate_up[expert]
            out[f"{hf}.mlp.experts.{expert}.down_proj"] = down[expert]
        for key in ("gate_proj", "up_proj", "down_proj"):
            out[f"{hf}.mlp.shared_expert.{key}.weight"] = state[f"{hf}.mlp.shared_expert.{key}.weight"]
        out[f"{hf}.mlp.shared_expert_gate.weight"] = state[f"{hf}.mlp.shared_expert_gate.weight"]
    return out


def write_safetensors(torch, path: Path, state) -> None:
    from safetensors.torch import save_file

    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
    }
    save_file(tensors, str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "qwen_moe_tiny"
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch, transformers, Config, Model = require_dependencies()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    config = make_hf_config(Config)
    model = Model(config).eval()
    initialize_deterministic(torch, model)
    reference = make_reference(torch, transformers, model)

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists (use --force): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(make_runtime_config(transformers.__version__), indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "tokenizer.json").write_text(
        json.dumps(make_tokenizer(), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    write_safetensors(torch, output / "model.safetensors", emit_engine_tensors(torch, model.state_dict()))
    (output / "ref.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output} (transformers={transformers.__version__})")
    print(f"  tensors: {len(emit_engine_tensors(torch, model.state_dict()))}")
    for name, case in reference["cases"].items():
        print(f"  case {name}: prompt={len(case['prompt_ids'])} "
              f"max_new={case['max_new_tokens']} greedy={case['greedy_new_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

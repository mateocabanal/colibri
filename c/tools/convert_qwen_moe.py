#!/usr/bin/env python3
"""Convert a Qwen3.5/3.6 MoE checkpoint (HF format) into a colibri qwen_moe
snapshot with DISK-STREAMED int8 experts.

The engine keeps dense tensors resident (embed, norms, attention, GDN,
router, shared expert, lm_head) and reads routed experts on demand from disk,
one per-expert tensor at a time, through a per-layer LRU cache. This
converter produces exactly that layout:

  snapshot/config.json      flat text_config (engine's load_cfg shape)
  snapshot/tokenizer.json   copied from the checkpoint
  snapshot/global.safetensors     embed, norm, lm_head (f32)
  snapshot/layers.N.safetensors   layer N dense weights (f32) + per-expert:
      layers.N.mlp.experts.E.merged_weight  int8 g|u|d packed, olmoe byte layout
      layers.N.mlp.experts.E.qs             f32 row scales [I + I + H]

  --bits 32 instead writes per-expert f32 tensors (gate_up_proj/down_proj)
  for oracle-diff runs. The engine probes merged_weight first, then f32.

Weight-name census (see docs/qwen-moe-config-census.md): text weights live
under model.language_model.* (lm_head.weight is top-level); experts ship
FUSED per layer (mlp.experts.gate_up_proj [E, 2I, H], mlp.experts.down_proj
[E, H, I], bf16, no .weight suffix); the shared expert has its own gate
(mlp.shared_expert_gate.weight); GDN A_log/dt_bias are bare Parameters.

Streaming: one layer at a time in memory (a 397B checkpoint's per-layer
expert block is ~4-9 GB bf16 — nothing else is ever fully resident).

Usage:
  python tools/convert_qwen_moe.py Qwen/Qwen3.6-35B-A3B-snap --out c/qwen_moe_merged
  python tools/convert_qwen_moe.py <hf_dir> --out <snap> --bits 32
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch


def require_deps():
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        sys.exit(f"Missing deps: {exc}. Run: pip install safetensors torch")


def load_tensor(paths: list[Path], name: str) -> torch.Tensor:
    for p in paths:
        try:
            with safe_open(str(p), framework="pt") as f:
                if name in f.keys():
                    return f.get_tensor(name).float().contiguous()
        except Exception:
            continue
    raise KeyError(f"tensor {name} not found in {[str(p) for p in paths]}")


def quantize_rowwise(t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-row int8 + f32 scales: q = round(w / s), s = max|w|/127."""
    w = t.detach().float().numpy()
    rows = w.shape[0]
    scales = np.abs(w).max(axis=1) / 127.0
    scales = np.maximum(scales, 1e-12)
    q = np.clip(np.round(w / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def q8_pair(t: torch.Tensor, name: str) -> dict:
    """Dense-weight low-RAM form: <name>_q8 (row-int8) + <name>_qs (f32 row
    scales). The engine probes this pair first and falls back to f32."""
    q, s = quantize_rowwise(t)
    return {f"{name}_q8": torch.from_numpy(q), f"{name}_qs": torch.from_numpy(s)}


def expert_merged(layer: int, expert: int, gate_up: torch.Tensor,
                  down: torch.Tensor, inter: int, hidden: int) -> dict:
    """Split the fused [2I, H] gate|up into int8 g|u|d + scales, olmoe layout:
    merged_weight = int8 [I*H | I*H | H*I], qs = f32 [I | I | H]."""
    gate, up = gate_up[:inter], gate_up[inter:]
    qg, sg = quantize_rowwise(gate)
    qu, su = quantize_rowwise(up)
    qd, sd = quantize_rowwise(down)
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    merged = np.concatenate([qg.ravel(), qu.ravel(), qd.ravel()])
    qs = np.concatenate([sg, su, sd]).astype(np.float32)
    return {
        f"{prefix}.merged_weight": torch.from_numpy(merged),
        f"{prefix}.qs": torch.from_numpy(qs),
    }


def expert_f32(layer: int, expert: int, gate_up: torch.Tensor,
               down: torch.Tensor) -> dict:
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    return {
        f"{prefix}.gate_up_proj": gate_up.contiguous(),
        f"{prefix}.down_proj": down.contiguous(),
    }


def build_config(hf_cfg: dict) -> dict:
    tc = hf_cfg.get("text_config") or hf_cfg
    out = dict(tc)
    out["architectures"] = ["Qwen3_5MoeForCausalLM"]
    out["model_type"] = tc.get("model_type", "qwen3_5_moe_text")
    out["torch_dtype"] = hf_cfg.get("dtype", "bfloat16")
    return out


def main() -> int:
    require_deps()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path, help="HF checkpoint directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bits", type=int, default=8, choices=(8, 32))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = args.model_dir.resolve()
    out = args.out.resolve()
    if not src.is_dir():
        sys.exit(f"model dir not found: {src}")
    if out.exists():
        if not args.force:
            sys.exit(f"output exists (use --force): {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    hf_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    tc = hf_cfg.get("text_config") or hf_cfg
    hidden = int(tc["hidden_size"])
    layers = int(tc["num_hidden_layers"])
    experts = int(tc["num_experts"])
    inter = int(tc["moe_intermediate_size"])
    layer_types = tc["layer_types"]
    vocab = int(tc["vocab_size"])

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors shards in {src}")

    # ---- global tensors: embed + lm_head row-int8 (4 GB f32 -> 1 GB), norm f32 ----
    global_t = {}
    embed = load_tensor(shards, "model.language_model.embed_tokens.weight")
    global_t.update(q8_pair(embed, "model.embed_tokens.weight"))
    lm_head = load_tensor(shards, "lm_head.weight")
    global_t.update(q8_pair(lm_head, "lm_head.weight"))
    global_t["model.norm.weight"] = load_tensor(shards, "model.language_model.norm.weight")
    save_file(global_t, str(out / "global.safetensors"))

    # ---- tokenizer ----
    tok = src / "tokenizer.json"
    if tok.exists():
        shutil.copyfile(tok, out / "tokenizer.json")
    else:
        print("warning: no tokenizer.json in checkpoint (chat mode needs one)")

    # ---- per-layer: dense + streamed experts ----
    for l in range(layers):
        lt = layer_types[l]
        dense: dict[str, torch.Tensor] = {}
        pre = f"model.language_model.layers.{l}"
        dst = f"model.layers.{l}"

        dense[f"{dst}.input_layernorm.weight"] = load_tensor(shards, f"{pre}.input_layernorm.weight")
        if lt == "full_attention":
            for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
                dense.update(q8_pair(load_tensor(shards, f"{pre}.self_attn.{k}.weight"),
                                     f"{dst}.self_attn.{k}.weight"))
            for k in ("q_norm", "k_norm"):
                dense[f"{dst}.self_attn.{k}.weight"] = load_tensor(shards, f"{pre}.self_attn.{k}.weight")
        else:
            for k in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"):
                dense.update(q8_pair(load_tensor(shards, f"{pre}.linear_attn.{k}.weight"),
                                     f"{dst}.linear_attn.{k}.weight"))
            dense[f"{dst}.linear_attn.norm.weight"] = load_tensor(shards, f"{pre}.linear_attn.norm.weight")
            dense[f"{dst}.linear_attn.A_log"] = load_tensor(shards, f"{pre}.linear_attn.A_log")
            dense[f"{dst}.linear_attn.dt_bias"] = load_tensor(shards, f"{pre}.linear_attn.dt_bias")
            dense[f"{dst}.linear_attn.conv1d.weight"] = load_tensor(shards, f"{pre}.linear_attn.conv1d.weight")
        dense[f"{dst}.post_attention_layernorm.weight"] = load_tensor(shards, f"{pre}.post_attention_layernorm.weight")
        dense.update(q8_pair(load_tensor(shards, f"{pre}.mlp.gate.weight"), f"{dst}.mlp.gate.weight"))
        for k in ("gate_proj", "up_proj", "down_proj"):
            dense.update(q8_pair(load_tensor(shards, f"{pre}.mlp.shared_expert.{k}.weight"),
                                 f"{dst}.mlp.shared_expert.{k}.weight"))
        dense.update(q8_pair(load_tensor(shards, f"{pre}.mlp.shared_expert_gate.weight"),
                             f"{dst}.mlp.shared_expert_gate.weight"))

        gate_up = load_tensor(shards, f"{pre}.mlp.experts.gate_up_proj")
        down = load_tensor(shards, f"{pre}.mlp.experts.down_proj")
        if gate_up.shape != (experts, 2 * inter, hidden) or down.shape != (experts, hidden, inter):
            sys.exit(f"layer {l}: unexpected expert shapes {gate_up.shape} / {down.shape}")

        for e in range(experts):
            if args.bits == 8:
                dense.update(expert_merged(l, e, gate_up[e], down[e], inter, hidden))
            else:
                dense.update(expert_f32(l, e, gate_up[e], down[e]))
        save_file(dense, str(out / f"layers.{l}.safetensors"))
        print(f"layer {l:3d}/{layers}: {lt:16s} experts {experts:4d} x int{args.bits} "
              f"({len(dense)} tensors)")

    (out / "config.json").write_text(
        json.dumps(build_config(hf_cfg), indent=2) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    print(f"wrote {out} ({total/1e9:.2f} GB, {vocab}-vocab, {layers} layers)")
    print("next: make -C c qwen_moe && c/qwen_moe <snap>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

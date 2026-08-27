#!/usr/bin/env python3
"""Generate a colic-shaped tiny Qwen4-Exp fixture (real-checkpoint layout).

Mirrors Qwen3.8-Flash-Next naming/dtypes so colic's Qwen4Exp frontend can be
tested without the 172 GB checkpoint: model.language_model.* prefix, separate
F8_E4M3 gate/up/down experts with BF16 weight_scale_inv block scales, BF16
dense. Output: fixture dir with config.json + model.safetensors.
"""
import json
import struct
import sys
from pathlib import Path

LAYERS = 2
LAYER_TYPES = ["linear_attention", "full_attention"]
EXPERTS = 4
HIDDEN = 64
INTER = 64
VOCAB = 500
HEADS = 4
HEAD_DIM = 16
KV_HEADS = 2
LIN_NUM_VALUE_HEADS = 4
LIN_KEY_HEAD_DIM = 16
LIN_VALUE_HEAD_DIM = 16
MAX_POS = 262144
EXPERT_BLOCK = 128


def bf16(value: float) -> bytes:
    # f32 -> bf16 round-to-nearest-even
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    lsb = (bits >> 16) & 1
    return struct.pack("<H", (bits + 0x7FFF + lsb) >> 16)


def e4m3(value: float) -> int:
    # nearest E4M3FN code for small test values (handles 0..448)
    best, best_err = 0, 1e9
    for code in range(256):
        exp = (code >> 3) & 0x0F
        mant = code & 0x07
        if exp == 15 and mant == 7:
            continue
        if exp == 0:
            v = (mant / 8) * (2 ** -6) if mant else 0.0
        else:
            v = (1 + mant / 8) * (2 ** (exp - 7))
        err = abs(v - value)
        if err < best_err:
            best, best_err = code, err
    return best


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "full_attention_interval": 2,
            "head_dim": HEAD_DIM,
            "hidden_size": HIDDEN,
            "indexer_budget": 16,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 16,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 1,
            "layer_types": LAYER_TYPES,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": LIN_KEY_HEAD_DIM,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": LIN_NUM_VALUE_HEADS,
            "linear_value_head_dim": LIN_VALUE_HEAD_DIM,
            "max_position_embeddings": MAX_POS,
            "moe_intermediate_size": INTER,
            "num_attention_heads": HEADS,
            "num_experts": EXPERTS,
            "num_experts_per_tok": 2,
            "num_hidden_layers": LAYERS,
            "num_key_value_heads": KV_HEADS,
            "rms_norm_eps": 1e-06,
            "vocab_size": VOCAB,
        },
    }
    (out / "config.json").write_text(json.dumps(config))

    tensors = {}
    payload = bytearray()

    def add(name, dtype, shape, data: bytes):
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)

    # embed + lm_head (BF16)
    embed = b"".join(bf16(float((i * 7 + 3) % 97) / 97.0) for i in range(VOCAB * HIDDEN))
    add("model.language_model.embed_tokens.weight", "BF16", [VOCAB, HIDDEN], embed)
    head = b"".join(bf16(float((i * 3 + 11) % 89) / 89.0) for i in range(VOCAB * HIDDEN))
    add("lm_head.weight", "BF16", [VOCAB, HIDDEN], head)

    # layer statics (bucket-only for the frontend; values just need valid dtypes)
    for layer, ltype in enumerate(LAYER_TYPES):
        lp = f"model.language_model.layers.{layer}"
        add(f"{lp}.input_layernorm.weight", "BF16", [HIDDEN], b"".join(bf16(1.0) for _ in range(HIDDEN)))
        add(f"{lp}.post_attention_layernorm.weight", "BF16", [HIDDEN], b"".join(bf16(1.0) for _ in range(HIDDEN)))
        add(f"{lp}.mlp.gate.weight", "BF16", [EXPERTS, HIDDEN], b"".join(bf16(0.5) for _ in range(EXPERTS * HIDDEN)))
        for role in ("gate_proj", "up_proj", "down_proj"):
            rows, cols = (INTER, HIDDEN) if role != "down_proj" else (HIDDEN, INTER)
            add(f"{lp}.mlp.shared_expert.{role}.weight", "BF16", [rows, cols],
                b"".join(bf16(0.25) for _ in range(rows * cols)))
        add(f"{lp}.mlp.shared_expert_gate.weight", "BF16", [HIDDEN], b"".join(bf16(0.5) for _ in range(HIDDEN)))
        if ltype == "full_attention":
            for role, rows in (("q_proj", HEADS * HEAD_DIM), ("k_proj", KV_HEADS * HEAD_DIM),
                               ("v_proj", KV_HEADS * HEAD_DIM), ("o_proj", HEADS * HEAD_DIM)):
                add(f"{lp}.self_attn.{role}.weight", "BF16", [rows, HIDDEN],
                    b"".join(bf16(0.1) for _ in range(rows * HIDDEN)))
            for role in ("q_norm", "k_norm"):
                add(f"{lp}.self_attn.{role}.weight", "BF16", [HEAD_DIM], b"".join(bf16(1.0) for _ in range(HEAD_DIM)))
            add(f"{lp}.self_attn.indexer.index_qk_proj.weight", "BF16", [HEADS * HEAD_DIM, HIDDEN],
                b"".join(bf16(0.1) for _ in range(HEADS * HEAD_DIM * HIDDEN)))
            for role in ("q_layernorm", "k_layernorm"):
                add(f"{lp}.self_attn.indexer.{role}.weight", "BF16", [HEAD_DIM], b"".join(bf16(1.0) for _ in range(HEAD_DIM)))
        else:
            for role, rows in (("in_proj_a", LIN_NUM_VALUE_HEADS), ("in_proj_b", LIN_NUM_VALUE_HEADS),
                               ("in_proj_z", LIN_NUM_VALUE_HEADS * LIN_VALUE_HEAD_DIM)):
                add(f"{lp}.linear_attn.{role}.weight", "BF16", [rows, HIDDEN],
                    b"".join(bf16(0.1) for _ in range(rows * HIDDEN)))
            qkv_rows = 2 * 2 * LIN_KEY_HEAD_DIM + LIN_NUM_VALUE_HEADS * LIN_VALUE_HEAD_DIM
            add(f"{lp}.linear_attn.in_proj_qkv.weight", "BF16", [qkv_rows, HIDDEN],
                b"".join(bf16(0.1) for _ in range(qkv_rows * HIDDEN)))
            add(f"{lp}.linear_attn.out_proj.weight", "BF16", [HIDDEN, LIN_NUM_VALUE_HEADS * LIN_VALUE_HEAD_DIM],
                b"".join(bf16(0.1) for _ in range(HIDDEN * LIN_NUM_VALUE_HEADS * LIN_VALUE_HEAD_DIM)))
            add(f"{lp}.linear_attn.norm.weight", "BF16", [LIN_VALUE_HEAD_DIM], b"".join(bf16(1.0) for _ in range(LIN_VALUE_HEAD_DIM)))
            add(f"{lp}.linear_attn.A_log", "BF16", [LIN_NUM_VALUE_HEADS], b"".join(bf16(1.0) for _ in range(LIN_NUM_VALUE_HEADS)))
            add(f"{lp}.linear_attn.dt_bias", "BF16", [LIN_NUM_VALUE_HEADS], b"".join(bf16(1.0) for _ in range(LIN_NUM_VALUE_HEADS)))
            add(f"{lp}.linear_attn.conv1d.weight", "BF16", [qkv_rows, 1, 4],
                b"".join(bf16(0.1) for _ in range(qkv_rows * 4)))

    # experts: separate F8 gate/up/down + BF16 weight_scale_inv
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            for role, rows, cols in (("gate", INTER, HIDDEN), ("up", INTER, HIDDEN), ("down", HIDDEN, INTER)):
                vals = [((layer + 1) * (expert + 1) * (r + 1)) % 8 / 8.0 for r in range(rows) for c in range(cols)]
                # scale: per 128x128 block; value * scale_inv = stored code magnitude; use scale 1.0
                add(f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}_proj.weight",
                    "F8_E4M3", [rows, cols], bytes(e4m3(v * 4) for v in vals))
                br = (rows + EXPERT_BLOCK - 1) // EXPERT_BLOCK
                bc = (cols + EXPERT_BLOCK - 1) // EXPERT_BLOCK
                add(f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}_proj.weight_scale_inv",
                    "BF16", [br, bc], b"".join(bf16(1.0) for _ in range(br * bc)))

    header = json.dumps(tensors, separators=(",", ":")).encode()
    with open(out / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(payload)
    print(f"fixture written to {out}: {len(tensors)} tensors, {len(payload)} payload bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("qwen4_coli_tiny"))
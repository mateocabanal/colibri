#!/usr/bin/env python3
"""Construct a tiny legal DeepSeek-V4-style safetensors source fixture.

The routed experts use native packed MXFP4 + UE8M0 so the fixture exercises the
actual Apple target ABI rather than an FP8 placeholder. Static tensors are zeroed:
the #53 execution oracle intentionally targets one compiler-emitted expert.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
    "I64": 8,
    "I8": 1,
    "F8_E4M3FN": 1,
    "F8_E8M0": 1,
}


def numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def pack_mxfp4(rows: list[list[int]]) -> bytes:
    """Pack e2m1 nibble codes; values are already the 0..15 wire codes."""
    out = bytearray()
    for row in rows:
        for i in range(0, len(row), 2):
            lo = row[i] & 0xF
            hi = (row[i + 1] & 0xF) if i + 1 < len(row) else 0
            out.append(lo | (hi << 4))
    return bytes(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)

    config = {
        "model_type": "deepseek_v4",
        "hidden_size": 2,
        "num_hidden_layers": 1,
        "n_routed_experts": 2,
        "moe_intermediate_size": 3,
        "vocab_size": 4,
        "hc_mult": 2,
        "num_hash_layers": 0,
        "num_experts_per_tok": 1,
        "num_attention_heads": 1,
        "head_dim": 2,
        "q_lora_rank": 1,
        "o_groups": 1,
        "o_lora_rank": 1,
        "index_n_heads": 1,
        "index_head_dim": 1,
        "compress_ratios": [0],
    }
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    tensors: dict[str, tuple[str, list[int], bytes]] = {}

    def add(name: str, dtype: str, shape: list[int], data: bytes | None = None) -> None:
        size = numel(shape) * DTYPE_BYTES[dtype]
        if data is None:
            data = bytes(size)
        if len(data) != size:
            raise ValueError(f"{name}: got {len(data)} bytes, expected {size}")
        tensors[name] = (dtype, shape, data)

    # Global/static semantic roles required by the strict V4 frontend.
    for name, dtype, shape in [
        ("embed.weight", "BF16", [4, 2]),
        ("head.weight", "BF16", [4, 2]),
        ("norm.weight", "BF16", [2]),
        ("hc_head_base", "F32", [2]),
        ("hc_head_fn", "F32", [2, 4]),
        ("hc_head_scale", "F32", [1]),
        ("layers.0.attn.attn_sink", "F32", [1]),
        ("layers.0.attn.kv_norm.weight", "BF16", [2]),
        ("layers.0.attn.q_norm.weight", "BF16", [1]),
        ("layers.0.attn.wkv.weight", "F8_E4M3FN", [2, 2]),
        ("layers.0.attn.wkv.scale", "F8_E8M0", [1, 1]),
        ("layers.0.attn.wo_a.weight", "F8_E4M3FN", [1, 2]),
        ("layers.0.attn.wo_a.scale", "F8_E8M0", [1, 1]),
        ("layers.0.attn.wo_b.weight", "F8_E4M3FN", [2, 1]),
        ("layers.0.attn.wo_b.scale", "F8_E8M0", [1, 1]),
        ("layers.0.attn.wq_a.weight", "F8_E4M3FN", [1, 2]),
        ("layers.0.attn.wq_a.scale", "F8_E8M0", [1, 1]),
        ("layers.0.attn.wq_b.weight", "F8_E4M3FN", [2, 1]),
        ("layers.0.attn.wq_b.scale", "F8_E8M0", [1, 1]),
        ("layers.0.attn_norm.weight", "BF16", [2]),
        ("layers.0.hc_attn_base", "F32", [8]),
        ("layers.0.hc_attn_fn", "F32", [8, 4]),
        ("layers.0.hc_attn_scale", "F32", [3]),
        ("layers.0.hc_ffn_base", "F32", [8]),
        ("layers.0.hc_ffn_fn", "F32", [8, 4]),
        ("layers.0.hc_ffn_scale", "F32", [3]),
        ("layers.0.ffn.gate.weight", "BF16", [2, 2]),
        ("layers.0.ffn.gate.bias", "F32", [2]),
        ("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3FN", [3, 2]),
        ("layers.0.ffn.shared_experts.w1.scale", "F8_E8M0", [1, 1]),
        ("layers.0.ffn.shared_experts.w2.weight", "F8_E4M3FN", [2, 3]),
        ("layers.0.ffn.shared_experts.w2.scale", "F8_E8M0", [1, 1]),
        ("layers.0.ffn.shared_experts.w3.weight", "F8_E4M3FN", [3, 2]),
        ("layers.0.ffn.shared_experts.w3.scale", "F8_E8M0", [1, 1]),
        ("layers.0.ffn_norm.weight", "BF16", [2]),
    ]:
        add(name, dtype, shape)

    # Optional draft tensors are valid V4 semantic records. Four bounded records
    # make the package exceed a 1 MiB test shard without creating an oversized
    # single record, so #52 can prove real multi-shard compiler/loader behavior.
    for index in range(4):
        add(f"mtp.fixture_padding.{index}", "BF16", [200_000])

    # E2M1 code 2 is +1.0 and code 0 is +0.0. Scale exponent 127 is 1.0.
    # Expert 0 gives gate/up rows [x0, x1, x0+x1], then down selects h0,h1.
    expert0 = {
        "w1": ([[2, 0], [0, 2], [2, 2]], 3, 2),
        "w2": ([[2, 0, 0], [0, 2, 0]], 2, 3),
        "w3": ([[2, 0], [0, 2], [2, 2]], 3, 2),
    }
    # Expert 1 is a legal zero expert; it exists to exercise complete indexing.
    expert1 = {
        "w1": ([[0, 0], [0, 0], [0, 0]], 3, 2),
        "w2": ([[0, 0, 0], [0, 0, 0]], 2, 3),
        "w3": ([[0, 0], [0, 0], [0, 0]], 3, 2),
    }
    for expert, matrices in enumerate([expert0, expert1]):
        for role, (rows, nrows, columns) in matrices.items():
            add(
                f"layers.0.ffn.experts.{expert}.{role}.weight",
                "I8",
                [nrows, (columns + 1) // 2],
                pack_mxfp4(rows),
            )
            add(
                f"layers.0.ffn.experts.{expert}.{role}.scale",
                "F8_E8M0",
                [nrows, (columns + 31) // 32],
                bytes([127]) * nrows * ((columns + 31) // 32),
            )

    # Safetensors headers are deterministic when keys are sorted and separators fixed.
    header: dict[str, dict[str, object]] = {}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with (root / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        f.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

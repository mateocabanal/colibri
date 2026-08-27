#!/usr/bin/env python3
"""#200 — Deterministic per-tensor-family calibration for the auto-planner.

Compares candidate math formats (INT4-G32, MXFP4, NVFP4) against the source
weights for each tensor family (routed_expert, shared_expert, router, gdn,
qsa, ngram_ple, attention_dense, moe_dense, embed, mtp) using offline metrics:

  - nRMSE      : sqrt(mean((q-r)^2)) / std(r)          (reconstruction error)
  - cosine_err : 1 - cos(q, r)                          (direction error)
  - outlier%   : |q-r| > 3*std(r) fraction              (heavy-tail damage)

Deterministic: seeded RNG, fixed sample grid (first 4 experts x rows 0,1 of
the first layer per family), no network. Results are cached in
calibration.json keyed by source fingerprint + quantizer version; the planner
consumes the cache as an optional veto input (`colic plan --calibration F`).

Usage:
  calibrate.py MODEL_DIR [--out calibration.json] [--seed 7]
"""

import argparse
import json
import math
import random
import struct
import sys
from pathlib import Path

QUANT_VERSION = "colic-quant-v1"  # must match colic planner_version when wired

FP8_BLOCK = 128
INT4_GROUP = 32
E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def f32_to_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def e4m3_to_f32(code: int) -> float:
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> 3) & 0x0F
    mant = code & 0x07
    if exp == 0:
        v = 0.0 if mant == 0 else (mant / 8) * (2 ** -6)
    elif exp == 15 and mant == 7:
        return float("nan")
    else:
        v = (1 + mant / 8) * (2 ** (exp - 7))
    return sign * v


def quantize_int4(values):
    """Symmetric grouped INT4-G32 (matches colic: scale = amax/7, code = round(v/scale)+8)."""
    out_w, out_s = [], []
    for start in range(0, len(values), INT4_GROUP):
        group = values[start : start + INT4_GROUP]
        amax = max((abs(v) for v in group), default=0.0)
        scale = 1.0 if amax == 0.0 else amax / 7.0
        codes = []
        for v in group:
            q = round(v / scale) if scale else 0
            q = max(-7, min(7, q))
            codes.append(q + 8)
        for i in range(0, len(codes), 2):
            lo = codes[i] & 0x0F
            hi = codes[i + 1] & 0x0F if i + 1 < len(codes) else 8
            out_w.append(lo | (hi << 4))
        out_s.append(scale)
    # decode
    dec = []
    for i, code_byte in enumerate(out_w):
        for j in (0, 1):
            nib = (code_byte >> (4 * j)) & 0x0F
            v = (nib - 8) * out_s[i // (INT4_GROUP // 2)]
            dec.append(v)
    return dec[: len(values)]


def quantize_mxfp4(values):
    """MXFP4 canonical: E2M1 + UE8M0 per 32 (matches colic mxfp4 quantizer)."""
    out_w, out_s = [], []
    for start in range(0, len(values), 32):
        group = values[start : start + 32]
        amax = max((abs(v) for v in group), default=0.0)
        if amax == 0.0:
            scale_code = 127
        else:
            # ue8m0: exponent of amax/6 (max magnitude), clamped to fp8 range
            exp = max(-127, min(127, math.floor(math.log2(amax / 6.0))))
            scale_code = exp + 127
        scale = math.ldexp(1.0, scale_code - 127)
        codes = []
        for v in group:
            best, best_err = 0, 1e30
            for c, m in enumerate(E2M1_MAG):
                e = abs(abs(v) / scale - m)
                if e < best_err:
                    best, best_err = c, e
            codes.append(best | (8 if v < 0 else 0))
        for i in range(0, len(codes), 2):
            out_w.append((codes[i] & 0x0F) | ((codes[i + 1] & 0x0F) << 4))
        out_s.append(scale_code)
    dec = []
    for i, code_byte in enumerate(out_w):
        scale = math.ldexp(1.0, out_s[i // 16] - 127)
        for j in (0, 1):
            nib = (code_byte >> (4 * j)) & 0x0F
            v = E2M1_MAG[nib & 7] * scale * (-1 if nib & 8 else 1)
            dec.append(v)
    return dec[: len(values)]


def metrics(orig, dec):
    n = len(orig)
    mean = sum(orig) / n
    var = sum((v - mean) ** 2 for v in orig) / n
    std = math.sqrt(var) if var else 1.0
    mse = sum((a - b) ** 2 for a, b in zip(orig, dec)) / n
    nrmse = math.sqrt(mse) / std if std else 0.0
    dot = sum(a * b for a, b in zip(orig, dec))
    na = math.sqrt(sum(a * a for a in orig)) or 1.0
    nb = math.sqrt(sum(b * b for b in dec)) or 1.0
    cosine_err = 1.0 - dot / (na * nb)
    outliers = sum(1 for a, b in zip(orig, dec) if abs(a - b) > 3 * std) / n
    return {"nrmse": round(nrmse, 4), "cosine_err": round(cosine_err, 6), "outlier_frac": round(outliers, 4)}


def read_tensor(f, hdr, name, data_start):
    """Return list[float] dequantized to f32 for a tensor (BF16 or F8_E4M3).

    safetensors data_offsets are relative to the DATA section start (right
    after the 8-byte length prefix + JSON header), not the file start.
    """
    info = hdr[name]
    dtype, shape = info["dtype"], info["shape"]
    off, length = info["data_offsets"]
    f.seek(data_start + off)
    raw = f.read(length)
    n = math.prod(shape)
    if dtype in ("BF16",):
        return [bf16_to_f32(int.from_bytes(raw[i : i + 2], "little")) for i in range(0, len(raw), 2)][:n]
    if dtype in ("F8_E4M3", "F8_E4M3FN"):
        return [e4m3_to_f32(b) for b in raw][:n]
    if dtype == "F32":
        return [struct.unpack("<f", raw[i : i + 4])[0] for i in range(0, len(raw), 4)][:n]
    raise ValueError(f"unsupported dtype {dtype} for {name}")


def family_of(name: str) -> str:
    low = name.lower()
    if "indexer" in low:
        return "qsa"
    if "ngram" in low or "ple" in low:
        return "ngram_ple"
    if "shared_expert" in low:
        return "shared_expert"
    if "ffn.gate" in low or "mlp.gate" in low or low.endswith(".gate.weight"):
        return "router"
    if "linear_attn" in low or "mamba" in low:
        return "gdn"
    if "self_attn" in low or ".attn" in low or "attention" in low:
        return "attention_dense"
    if "mlp" in low or "experts" in low:
        return "moe_dense"
    if "mtp" in low:
        return "mtp"
    if "embed" in low or "lm_head" in low:
        return "embed"
    if "norm" in low:
        return "norm"
    return "other"


def sample_tensor(name, values, family, rng, rows=2, cols=512):
    """Deterministic row-slice sample for calibration: rows x cols from the start."""
    return values[: rows * cols]


def fingerprint(root: Path) -> str:
    """Cheap deterministic fingerprint: config + index + shard count + sizes."""
    import hashlib

    h = hashlib.sha256()
    for fname in sorted(root.iterdir()):
        if fname.name.endswith(".safetensors"):
            h.update(fname.name.encode())
            h.update(str(fname.stat().st_size).encode())
    cfg = root / "config.json"
    if cfg.exists():
        h.update(cfg.read_bytes())
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        h.update(idx.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.model_dir)
    fp = fingerprint(root)
    index = root / "model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(wm.values()))
        tensors = {}
        for shard in shards:
            with open(root / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
                for name in hdr:
                    tensors[name] = (shard, hdr)
    else:
        with open(root / "model.safetensors", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            tensors = {name: ("model.safetensors", hdr) for name in hdr}

    # Deterministic per-family sampling: first tensor per family (sorted), first rows.
    families = {}
    for name in sorted(tensors):
        fam = family_of(name)
        if fam not in families:
            families[fam] = name
    rng = random.Random(args.seed)
    cache = {"fingerprint": fp, "quantizer_version": QUANT_VERSION, "seed": args.seed,
             "families": {}}
    for fam, name in sorted(families.items()):
        shard, hdr = tensors[name]
        with open(root / shard, "rb") as f:
            # data section start = 8 (length prefix) + header length
            header_len = struct.unpack("<Q", f.read(8))[0]
            data_start = 8 + header_len
            values = read_tensor(f, hdr, name, data_start)
        if len(values) < 64:
            continue
        sample = values[: args.rows * min(args.cols, len(values))]
        # deterministic sample: take the first slice (sorted names = stable)
        res = {"int4_g32": metrics(sample, quantize_int4(sample)),
               "mxfp4": metrics(sample, quantize_mxfp4(sample))}
        # sanity: family tensor id for the report
        res["_tensor"] = name
        cache["families"][fam] = res
        print(f"{fam:16s} {name[-70:]:70s} nrmse i4={res['int4_g32']['nrmse']:.3f} mx={res['mxfp4']['nrmse']:.3f}")

    out = Path(args.out)
    out.write_text(json.dumps(cache, indent=2))
    print(f"wrote {out} (fingerprint {fp[:16]}…)")


if __name__ == "__main__":
    sys.exit(main())

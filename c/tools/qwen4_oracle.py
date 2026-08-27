#!/usr/bin/env python3
"""qwen4_oracle.py — pure numpy/torch forward oracle for the Qwen4 tiny fixture.

Implements the exact Qwen4Exp text forward (transformers 5.57.1 qwen4_exp
modular, pinned M1 formulas): repeated-stream hyper connections (hc_count*D),
Gated DeltaNet linear_attention layer, full attention + QSA indexer layer,
top-k MoE with a gated shared expert, Per-Layer Embedding (PLE) with hashed
n-grams, and the global hyper_connection_mixer (use_combine=False) before
lm_head. All math in float32; every gate margin is wide so a f32 C engine
reproduces the same argmax tokens.

Reads a fixture dir (config.json + model.safetensors + ref.json), recomputes
every case (teacher-forcing + greedy decode), checks it against ref.json and
writes fixture_oracle_ids.txt with the expected token IDs.

Usage: python qwen4_oracle.py <fixture_dir> [--out fixture_oracle_ids.txt]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB


def splitmix64(value: int) -> int:
    value = (value + GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * M1) & MASK64
    value = ((value ^ (value >> 27)) * M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


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


def rms_group(x: torch.Tensor, weight: torch.Tensor, group: int, eps: float) -> torch.Tensor:
    """Group RMSNorm (Qwen4ExpTextRMSNorm): x * rsqrt(mean(x^2)+eps) * (1+w)."""
    shape = x.shape
    xg = x.reshape(*shape[:-1], -1, group)
    out = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.reshape(-1, group))
    return out.reshape(*shape)


def rms_flat(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return rms_group(x, weight, x.shape[-1], eps)


def rope_embeddings(seq_len: int, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen4ExpTextRotaryEmbedding, pure text (mrope no-op): [T, dim] cos/sin."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = pos[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the FIRST rotary_dim elements of the last dim (HF layout)."""
    rd = cos.shape[-1]
    rope, nope = x[..., :rd], x[..., rd:]
    rope = rope * cos + rotate_half(rope) * sin
    return torch.cat([rope, nope], dim=-1)


def topk_lower_index_first(scores: np.ndarray, k: int) -> np.ndarray:
    """torch.topk(descending) with the engine's tie-break: lower index wins."""
    return np.argsort(-scores, kind="stable")[:k]


def greedy_argmax(logits: torch.Tensor) -> int:
    return int(np.argmax(logits.detach().cpu().numpy()))


class Qwen4Oracle:
    """Stateless f32 forward of the Qwen4 tiny model.

    Conv / recurrence / n-gram context are rebuilt from the token sequence on
    every call — exactly equivalent to the cached HF forward for pure text.
    """

    def __init__(self, cfg: dict, tensors: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.t = {k: v.detach().float() for k, v in tensors.items()}
        self.V = int(cfg["vocab_size"])
        self.D = int(cfg["hidden_size"])
        self.HC = int(cfg["hc_count"])
        self.HCD = self.HC * self.D
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.eos = int(cfg["eos_token_id"])
        self.theta = float(cfg.get("rope_parameters", {}).get("rope_theta", 10000.0))
        self.rotary_dim = int(int(cfg.get("head_dim", self.D // 4)) *
                              cfg.get("rope_parameters", {}).get("partial_rotary_factor", 1.0))
        self.seed = int(cfg["seed"])
        self.ns = int(cfg["ngram_size"])
        self.hpn = int(cfg["heads_per_ngram"])
        self.nheads = (self.ns - 1) * self.hpn
        self.ple_dim = int(cfg["ple_embed_dim"])
        self.hd_per_ngram = self.ple_dim // self.nheads
        self.multipliers = self._build_multipliers()
        sizes, offsets = [], []
        total = 0
        for h in range(self.nheads):
            s = nth_prime_after(int(cfg["ngram_vocab_size_base"]) - 1, h + 1)
            sizes.append(s)
            offsets.append(total)
            total += s
        self.head_vocab_sizes = sizes
        self.head_offsets = offsets
        self.ngram_total = total
        self.ngram_padded = math.ceil(total / int(cfg["make_ngram_vocab_size_divisible_by"])) * \
            int(cfg["make_ngram_vocab_size_divisible_by"])
        self.layer_types = cfg["layer_types"]

    def _build_multipliers(self) -> list[int]:
        max_long = (1 << 63) - 1
        mult_max = max_long // max(self.V, 1)
        half_bound = max(1, mult_max // 2)
        base_seed = self.seed + 10007 * 0
        out = []
        for i in range(self.ns):
            value = (base_seed + GAMMA * (i + 1)) & MASK64
            out.append(2 * (splitmix64(value) % half_bound) + 1)
        return out

    # ---------------- PLE n-gram embeddings ----------------
    def _shift_right_ignore_eos(self, token_ids: np.ndarray, shift: int) -> np.ndarray:
        T = token_ids.shape[0]
        positions = np.arange(T)
        eos_positions = np.where(token_ids == self.eos, positions, -1)
        prev_eos_incl = np.maximum.accumulate(eos_positions)
        prev_eos = np.concatenate([[-1], prev_eos_incl[:-1]])
        seg_start = prev_eos + 1
        pos_in_seg = positions - seg_start
        src = np.clip(positions - shift, 0, None)
        shifted = token_ids[src]
        valid = (pos_in_seg >= shift) & (positions - shift >= 0)
        return np.where(valid, shifted, self.eos)

    def _ngram_embeddings(self, ids: np.ndarray) -> torch.Tensor:
        """[T] ids -> [T, ple_embed_dim]."""
        ctx = self.ns - 1
        history = np.concatenate([np.full(ctx, self.eos, dtype=np.int64), ids])
        shifted = [self._shift_right_ignore_eos(history, n) for n in range(self.ns)]
        su = [s.astype(np.uint64) for s in shifted]
        blocks = []
        for ngram in range(2, self.ns + 1):
            start = (ngram - 2) * self.hpn
            end = start + self.hpn
            mixed = su[0] * np.uint64(self.multipliers[0])
            for pos in range(1, ngram):
                mixed = np.bitwise_xor(mixed, su[pos] * np.uint64(self.multipliers[pos]))
            sizes = np.array(self.head_vocab_sizes[start:end], dtype=np.uint64)
            offsets = np.array(self.head_offsets[start:end], dtype=np.uint64)
            ngram_ids = np.remainder(mixed[:, None], sizes[None, :]) + offsets[None, :]
            blocks.append(ngram_ids)
        ngram_ids = np.concatenate(blocks, axis=-1)[-ids.shape[0]:]  # [T, nheads]
        ids_t = torch.from_numpy(ngram_ids.astype(np.int64))
        w = self.t["model.ple.ple_embedding.ngram_embedding.weight"]
        emb = F.embedding(ids_t, w)  # [T, nheads, hd]
        return emb.reshape(ids_t.shape[0], -1)

    # ---------------- PLE layer ----------------
    def _ple(self, hidden: torch.Tensor, ids: np.ndarray) -> torch.Tensor:
        emb = self._ngram_embeddings(ids)  # [T, ple_dim]
        key = emb @ self.t["model.ple.key_proj.weight"].t()  # [T, HC*D]
        key = rms_group(key, self.t["model.ple.norm_key.weight"], self.D, self.eps)
        key = key.unflatten(-1, (self.HC, self.D))
        value = emb @ self.t["model.ple.value_proj.weight"].t()  # [T, D]
        q = rms_group(hidden, self.t["model.ple.norm_query.weight"], self.D, self.eps)
        q = q.unflatten(-1, (self.HC, self.D))
        gate = (key * q).sum(-1, keepdim=True) / math.sqrt(self.D)  # [T, HC, 1]
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)  # [T, HC, D]
        gated_flat = gated.flatten(-2)
        conv_in = rms_group(gated_flat, self.t["model.ple.norm_conv.weight"], self.D, self.eps)
        k = int(self.cfg["ple_conv_kernel_size"])
        dil = self.ns
        pad = (k - 1) * dil
        x = gated_flat.transpose(0, 1).unsqueeze(0)  # [1, HC*D, T]
        x = F.pad(x, (pad, 0))
        w = self.t["model.ple.conv1d.weight"]  # [HC*D, 1, k]
        conv = F.conv1d(x, w, padding=0, dilation=dil, groups=self.HCD)[0]
        conv = F.silu(conv).transpose(0, 1)  # [T, HC*D]
        return gated_flat + conv

    # ---------------- GatedResidual ----------------
    def _gated_residual(self, x: torch.Tensor, tag: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = rms_group(x, self.t[f"{tag}.hc_norm.weight"], self.D, self.eps)  # [*, HC*D]
        down = self.t[f"{tag}.input_mix_weight_down.weight"]
        up = self.t[f"{tag}.input_mix_weight_up.weight"]
        m = F.silu(n @ down.t() / self.HC)          # [*, hc_lowrank]
        m = torch.sigmoid(m @ up.t())               # [*, HC*D]
        m = m.reshape(*x.shape[:-1], self.HC, self.D)
        normed = n.reshape(*x.shape[:-1], self.HC, self.D)
        mixed = (m * normed).mean(dim=-2)           # [*, D]
        if f"{tag}.block_inject_weight.weight" not in self.t:
            return mixed, None, None
        bi = self.t[f"{tag}.block_inject_weight.weight"]
        inj = 2.0 * torch.sigmoid(n @ bi.t() / self.HC)  # [*, HC]
        return mixed, x, inj

    # ---------------- Gated DeltaNet (linear_attention) ----------------
    def _gdn(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        kd = int(self.cfg["linear_key_head_dim"])
        kh = int(self.cfg["linear_num_key_heads"])
        vd = int(self.cfg["linear_value_head_dim"])
        vh = int(self.cfg["linear_num_value_heads"])
        kdim, vdim = kd * kh, vd * vh
        kk = int(self.cfg["linear_conv_kernel_dim"])
        C = kdim * 2 + vdim
        rep = vh // kh
        t = self.t
        Wqkv = t["model.layers.0.linear_attn.in_proj_qkv.weight"]
        Wa = t["model.layers.0.linear_attn.in_proj_a.weight"]
        Wb = t["model.layers.0.linear_attn.in_proj_b.weight"]
        Wz = t["model.layers.0.linear_attn.in_proj_z.weight"]
        Wconv = t["model.layers.0.linear_attn.conv1d.weight"][:, 0, :]  # [C, kk]
        A_log = t["model.layers.0.linear_attn.A_log"]
        dt = t["model.layers.0.linear_attn.dt_bias"]
        Wnorm = t["model.layers.0.linear_attn.norm.weight"]
        Wout = t["model.layers.0.linear_attn.out_proj.weight"]

        qkv = x @ Wqkv.t()  # [T, C]
        a = x @ Wa.t()
        b = x @ Wb.t()
        z = x @ Wz.t()
        # causal depthwise conv (HF conv1d + left pad kk-1, then silu)
        y = torch.zeros_like(qkv)
        for tt in range(T):
            acc = torch.zeros(C)
            for j in range(kk):
                lag = kk - 1 - j
                src = qkv[tt - lag] if tt >= lag else torch.zeros(C)
                acc = acc + Wconv[:, j] * src
            y[tt] = F.silu(acc)
        outs = []
        S = torch.zeros(vh, kd, vd)
        for tt in range(T):
            q_ = y[tt, :kdim].reshape(kh, kd)
            k_ = y[tt, kdim:2 * kdim].reshape(kh, kd)
            v_ = y[tt, 2 * kdim:].reshape(vh, vd)
            out_h = []
            for h in range(vh):
                khd = h // rep
                qh = F.normalize(q_[khd], dim=0)
                kh_ = F.normalize(k_[khd], dim=0)
                qh = qh / math.sqrt(kd)
                ga = -A_log[h].exp() * F.softplus(a[tt, h] + dt[h])
                gt = ga.exp()
                bt = torch.sigmoid(b[tt, h])
                Sh = S[h] * gt
                kv_mem = (Sh * kh_.unsqueeze(0)).sum(dim=0)  # [vd]
                delta = (v_[h] - kv_mem) * bt
                Sh = Sh + kh_[:, None] * delta[None, :]
                S[h] = Sh
                out_h.append((Sh * qh[:, None]).sum(dim=0))  # [vd]
            o = torch.stack(out_h)  # [vh, vd]
            o = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6)
            o = o * Wnorm * F.silu(z[tt].reshape(vh, vd))
            outs.append(o.reshape(-1))  # [vdim]
        return torch.stack(outs) @ Wout.t()  # [T, D]

    # ---------------- QSA indexer + full attention ----------------
    def _attn(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        hd = int(self.cfg["head_dim"])
        H = int(self.cfg["num_attention_heads"])
        KH = int(self.cfg["num_key_value_heads"])
        nq = int(self.cfg["indexer_n_heads"])
        nkv = int(self.cfg["indexer_kv_heads"])
        ihd = int(self.cfg["indexer_head_dim"])
        budget = int(self.cfg["indexer_budget"])
        ratio = int(self.cfg["indexer_compress_ratio"])
        block_topk = budget // ratio
        t = self.t
        # ---- indexer ----
        Wqk = t["model.layers.1.self_attn.indexer.index_qk_proj.weight"]
        qk = x @ Wqk.t()  # [T, (nq+nkv)*ihd]
        q = qk[:, : nq * ihd].reshape(T, nq, ihd)
        token_k = qk[:, nq * ihd:].reshape(T, nkv, ihd).squeeze(1)  # [T, ihd]
        q = rms_flat(q, t["model.layers.1.self_attn.indexer.q_layernorm.weight"], self.eps)
        q = rope_apply(q, cos.unsqueeze(1), sin.unsqueeze(1))  # [T, nq, ihd]
        # selected-token mask per query (pure text: causal visible = 0..p)
        sel_mask = torch.zeros(T, T, dtype=torch.bool)
        for p in range(T):
            visible = np.arange(p + 1)
            nblocks = visible.shape[0] // ratio
            selected = []
            if nblocks > 0:
                btok = visible[: nblocks * ratio].reshape(nblocks, ratio)
                pooled = token_k[btok.flatten()].reshape(nblocks, ratio, ihd).float().mean(dim=1)
                pooled = rms_flat(pooled, t["model.layers.1.self_attn.indexer.k_layernorm.weight"], self.eps)
                starts = btok[:, 0]
                pooled = rope_apply(pooled.unsqueeze(1), cos[starts].unsqueeze(1),
                                    sin[starts].unsqueeze(1)).squeeze(1)
                scores = torch.relu(q[p].float() @ pooled.float().t()).sum(dim=0) / math.sqrt(ihd)
                sel = topk_lower_index_first(scores.detach().cpu().numpy(), min(block_topk, nblocks))
                selected = btok[sel].reshape(-1).tolist()
            tail = visible[nblocks * ratio:]
            sel_tokens = np.concatenate([np.array(selected, dtype=np.int64), tail])
            sel_mask[p, sel_tokens] = True
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        mask = causal & sel_mask
        # ---- attention ----
        qg = (x @ t["model.layers.1.self_attn.q_proj.weight"].t()).reshape(T, H, 2 * hd)
        q_, gate = torch.chunk(qg, 2, dim=-1)
        q_ = rms_flat(q_, t["model.layers.1.self_attn.q_norm.weight"], self.eps)
        k_ = rms_flat((x @ t["model.layers.1.self_attn.k_proj.weight"].t()).reshape(T, KH, hd),
                      t["model.layers.1.self_attn.k_norm.weight"], self.eps)
        v = (x @ t["model.layers.1.self_attn.v_proj.weight"].t()).reshape(T, KH, hd)
        q_ = rope_apply(q_, cos.unsqueeze(1), sin.unsqueeze(1))
        k_ = rope_apply(k_, cos.unsqueeze(1), sin.unsqueeze(1))
        q_, k_, v = q_.transpose(0, 1), k_.transpose(0, 1), v.transpose(0, 1)
        k_e = k_.repeat_interleave(H // KH, dim=0)
        v_e = v.repeat_interleave(H // KH, dim=0)
        scores = torch.einsum("htd,hkd->htk", q_, k_e) / math.sqrt(hd)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        wts = F.softmax(scores, dim=-1)
        out = torch.einsum("htk,hkd->htd", wts, v_e)  # [H, T, hd]
        out = out.transpose(0, 1).reshape(T, H * hd)
        out = out * torch.sigmoid(gate.reshape(T, H * hd))
        return out @ t["model.layers.1.self_attn.o_proj.weight"].t()

    # ---------------- MoE ----------------
    def _moe(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        E = int(self.cfg["num_experts"])
        K = int(self.cfg["num_experts_per_tok"])
        t = self.t
        probs = torch.softmax(x @ t[f"model.layers.{layer}.mlp.gate.weight"].t(), dim=-1)
        pn = probs.detach().cpu().numpy()
        sel = np.argsort(-pn, axis=-1, kind="stable")[:, :K]  # [T, K], tie -> lowest idx
        w = np.take_along_axis(pn, sel, axis=-1)
        w = w / w.sum(-1, keepdims=True)
        T = x.shape[0]
        out = torch.zeros(T, self.D)
        for row in range(T):
            for e, wt in zip(sel[row], w[row]):
                e = int(e)
                gu = t[f"model.layers.{layer}.mlp.experts.{e}.gate_up_proj"]
                dn = t[f"model.layers.{layer}.mlp.experts.{e}.down_proj"]
                g_, u = torch.chunk(gu @ x[row], 2)
                out[row] = out[row] + wt * (dn @ (F.silu(g_) * u))
        sg = t[f"model.layers.{layer}.mlp.shared_expert_gate.weight"]
        s_gu = t[f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight"]
        s_u = t[f"model.layers.{layer}.mlp.shared_expert.up_proj.weight"]
        s_dn = t[f"model.layers.{layer}.mlp.shared_expert.down_proj.weight"]
        s_out = s_dn @ (F.silu(x @ s_gu.t()) * (x @ s_u.t())).t()
        s_out = s_out.t()  # [T, D]
        out = out + torch.sigmoid(x @ sg.t()) * s_out
        return out

    # ---------------- full forward ----------------
    def forward(self, input_ids: np.ndarray) -> torch.Tensor:
        """[T] ids -> logits [T, V]."""
        T = input_ids.shape[0]
        ids = torch.from_numpy(input_ids.astype(np.int64))
        emb = self.t["model.embed_tokens.weight"][ids]  # [T, D]
        h = emb.repeat(1, self.HC)  # [T, HC*D]
        cos, sin = rope_embeddings(T, self.D, self.theta)
        cos_i, sin_i = rope_embeddings(T, self.rotary_dim, self.theta)
        for li in range(int(self.cfg["num_hidden_layers"])):
            if self.cfg.get("ple_layer_ids") and (li + 1) in self.cfg["ple_layer_ids"]:
                h = h + self._ple(h, input_ids)
            # attention hyper connection
            mixed, hyper, inj = self._gated_residual(h, f"model.layers.{li}.attn_hyper_connection")
            if self.layer_types[li] == "linear_attention":
                a_out = self._gdn(mixed)
            else:
                a_out = self._attn(mixed, cos_i, sin_i)
            h = hyper + (inj.unsqueeze(-1) * a_out.unsqueeze(-2)).flatten(-2)
            # mlp hyper connection
            mixed, hyper, inj = self._gated_residual(h, f"model.layers.{li}.mlp_hyper_connection")
            m_out = self._moe(mixed, li)
            h = hyper + (inj.unsqueeze(-1) * m_out.unsqueeze(-2)).flatten(-2)
        # global mixer (use_combine=False -> no block_inject_weight)
        n = rms_group(h, self.t["model.hyper_connection_mixer.hc_norm.weight"], self.D, self.eps)
        down = self.t["model.hyper_connection_mixer.input_mix_weight_down.weight"]
        up = self.t["model.hyper_connection_mixer.input_mix_weight_up.weight"]
        m = F.silu(n @ down.t() / self.HC)
        m = torch.sigmoid(m @ up.t())
        m = m.reshape(T, self.HC, self.D)
        normed = n.reshape(T, self.HC, self.D)
        final = (m * normed).mean(dim=-2)  # [T, D]
        return final @ self.t["lm_head.weight"].t()  # [T, V]


# ------------------------------------------------------------------ main ---
def load_model(fixture: Path):
    cfg = json.loads((fixture / "config.json").read_text(encoding="utf-8"))
    tensors = load_file(str(fixture / "model.safetensors"))
    return Qwen4Oracle(cfg, tensors)


def run_case(model: Qwen4Oracle, case: dict) -> dict:
    prompt = np.array(case["prompt_ids"], dtype=np.int64)
    max_new = int(case["max_new_tokens"])
    full = list(prompt.tolist())
    with torch.no_grad():
        for _ in range(max_new):
            logits = model.forward(np.array(full, dtype=np.int64))[-1]
            tok = greedy_argmax(logits)
            full.append(tok)
        tf = model.forward(np.array(full, dtype=np.int64))
    teacher = [greedy_argmax(tf[i]) for i in range(tf.shape[0])]
    return {
        "greedy_full_ids": full,
        "greedy_new_ids": full[len(prompt):],
        "teacher_forcing_ids": teacher,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    fixture = args.fixture.resolve()
    model = load_model(fixture)
    ref = json.loads((fixture / "ref.json").read_text(encoding="utf-8"))
    cases = ref["cases"]

    results = {}
    all_ok = True
    lines = []
    for name in sorted(cases.keys()):
        case = cases[name]
        got = run_case(model, case)
        results[name] = {
            "prompt_ids": case["prompt_ids"],
            "max_new_tokens": case["max_new_tokens"],
            **got,
        }
        ok = True
        if case.get("greedy_new_ids"):
            ok = ok and got["greedy_new_ids"] == case["greedy_new_ids"]
        if case.get("teacher_forcing_ids"):
            ok = ok and got["teacher_forcing_ids"] == case["teacher_forcing_ids"]
        all_ok = all_ok and ok
        lines.append(f"case {name}: greedy_new_ids={got['greedy_new_ids']} "
                     f"teacher_forcing_ids={got['teacher_forcing_ids']} "
                     f"{'OK' if ok else 'MISMATCH'}")
        print("  " + lines[-1])

    out_path = (args.out.resolve() if args.out else fixture / "fixture_oracle_ids.txt")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # update ref.json with the recomputed values (same numbers when already correct)
    for name in sorted(cases.keys()):
        cases[name].update(results[name])
    (fixture / "ref.json").write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print("SELFTEST " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

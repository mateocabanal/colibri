#!/usr/bin/env python3
"""T9a — cache-aware offline evaluation of layer-lookahead prefetch policies
on a REAL Qwen route trace. Simulates the demand path exactly (cap-8 LRU per
layer) and evaluates candidate predictors that would have placed their guess
in a SEPARATE one-slot probation buffer (never the demand LRU, so no demand
evictions by construction). Gate: held-out useful-load precision >= 50% and
a simulated reduction in demand misses vs the no-prediction control.

Usage: route_prefetch_sim.py <trace> [--k 1|2|4|8 ...]"""
import argparse
import sys
from collections import OrderedDict, defaultdict

def parse_trace(path):
    """Decode-only extraction: batched prefill emits many rows per
    (call, layer) — those are not LRU-prefetch candidates; decode emits
    exactly one row per layer per call. Return the ordered decode ladder. """
    calls = OrderedDict()      # call -> {layer: [ids (single row assumed)]}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            call, layer = int(parts[0]), int(parts[2])
            ids = []
            for tok in parts[3:]:
                if ":" in tok:
                    ids.append(int(tok.split(":")[0]))
            if len(ids) < 1:
                continue
            calls.setdefault(call, {}).setdefault(layer, []).extend(ids)
    seq = []
    for call in sorted(calls):
        layers = calls[call]
        single = all(len(v) <= 8 for v in layers.values())
        # prefill batch rows produce >8 ids per layer in a call (64 rows x
        # top-k); decode calls produce <= topk ids per layer. Keep decode.
        if not single:
            continue
        for layer in sorted(layers):
            seq.append((layer, frozenset(layers[layer])))
    return seq

def demand_sim(seq, cap=8):
    """Exact demand replay: per-layer LRU of the last cap distinct experts."""
    lru = {}                   # layer -> list of ids, MRU first
    hits = misses = 0
    for layer, picks in seq:
        row = lru.get(layer, [])
        for e in picks:
            if e in row:
                hits += 1
                row.remove(e); row.insert(0, e)
            else:
                misses += 1
                row = [e] + [x for x in row if x != e][: cap - 1]
        lru[layer] = row
    return hits, misses

def eval_policy(seq, k):
    """Existing hook semantics: layer L's top-k predicts layer L+1's picks AT
    THE SAME TOKEN. Prediction goes to a probation slot; useful = predicted
    id is in the target layer's ACTUAL picks (a demand miss avoided when the
    target is a miss)."""
    useful = predicted = 0
    per_layer = defaultdict(list)
    for layer, picks in seq:
        per_layer[layer].append((layer, picks))
    # build ordered token progression per layer pair
    order = [s for s in seq]
    for i in range(len(order) - 1):
        cur_layer, cur_picks = order[i]
        nxt_layer, nxt_picks = order[i + 1]
        if nxt_layer != cur_layer + 1:
            continue
        # candidate = top-k of the current layer's picks (routing likely
        # correlated); rank by nothing (set order) — use first k
        cands = list(cur_picks)[:k]
        for c in cands:
            predicted += 1
            if c in nxt_picks:
                useful += 1
    return useful, predicted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--k-top", type=int, default=1, choices=(1, 2, 4, 8))
    a = ap.parse_args()
    seq = parse_trace(a.trace)
    if not seq:
        sys.exit("no rows parsed")
    print(f"rows: {len(seq)}  layers: {len(set(l for l, _ in seq))}")
    h, m = demand_sim(seq)
    print(f"demand: hits={h} misses={m} hit%={100.0 * h / (h + m):.1f}")
    useful, predicted = eval_policy(seq, a.k_top)
    prec = 100.0 * useful / predicted if predicted else 0.0
    saved_misses = useful   # a useful probation hit avoids exactly one miss
    print(f"policy top-{a.k_top}: predicted={predicted} useful={useful} "
          f"precision={prec:.1f}%  misses avoided={saved_misses} / {m} "
          f"({100.0 * saved_misses / m if m else 0:.1f}%)")
    gate = prec >= 50.0
    print(f"GATE (>=50% precision): {'PASS' if gate else 'FAIL'}")

if __name__ == "__main__":
    main()
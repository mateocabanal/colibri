#!/usr/bin/env python3
"""Analyze V4 routed-expert JSONL traces without third-party dependencies.

The runtime trace is intentionally cheap: logical expert requests plus residency
transitions. This tool turns the request stream into the policy inputs needed by
#3/#56/#57: activation skew, LRU stack/reuse distance, hypothetical cache hit
curves, physical load counts, and persistent/transient hit attribution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class Fenwick:
    """Fenwick tree for O(log n) exact LRU stack-distance computation."""

    def __init__(self, size: int) -> None:
        self.tree = [0] * (size + 2)

    def add(self, index: int, delta: int) -> None:
        i = index + 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & -i

    def prefix(self, index: int) -> int:
        if index < 0:
            return 0
        total = 0
        i = index + 1
        while i:
            total += self.tree[i]
            i -= i & -i
        return total


@dataclass
class TraceAnalysis:
    requests: list[tuple[int, int]]
    event_counts: Counter[str]
    tier_hits: Counter[str]
    layer_frequency: dict[int, Counter[int]]
    reuse_distances: list[int]
    cold_requests: int
    record_bytes: int
    dropped: int


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield item


def exact_reuse_distances(requests: list[tuple[int, int]]) -> tuple[list[int], int]:
    """Return exact LRU stack distance for each repeated expert request.

    Distance is the number of distinct experts referenced more recently than the
    prior reference to this same logical `(layer, expert)` key. Therefore an LRU
    cache of capacity C would hit exactly when distance < C. First references
    are cold and excluded from the distance list.
    """

    bit = Fenwick(len(requests) + 1)
    last: dict[tuple[int, int], int] = {}
    distances: list[int] = []
    cold = 0

    for position, key in enumerate(requests):
        previous = last.get(key)
        if previous is None:
            cold += 1
        else:
            active = bit.prefix(position - 1)
            not_older_than_previous = bit.prefix(previous)
            distances.append(active - not_older_than_previous)
            bit.add(previous, -1)
        bit.add(position, 1)
        last[key] = position
    return distances, cold


def analyze(path: Path) -> TraceAnalysis:
    requests: list[tuple[int, int]] = []
    event_counts: Counter[str] = Counter()
    tier_hits: Counter[str] = Counter()
    layer_frequency: dict[int, Counter[int]] = defaultdict(Counter)
    record_bytes = 0
    dropped = 0

    for item in read_jsonl(path):
        if item.get("schema") == "colibri.v4.expert_trace.v1":
            record_bytes = int(item.get("record_bytes", 0) or 0)
            dropped = int(item.get("dropped", 0) or 0)
            continue
        event = item.get("event")
        if not isinstance(event, str):
            continue
        event_counts[event] += 1
        if event == "request":
            try:
                key = (int(item["layer"]), int(item["expert"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("request event missing integer layer/expert") from exc
            requests.append(key)
            layer_frequency[key[0]][key[1]] += 1
        elif event == "hit":
            tier = item.get("tier")
            tier_hits[str(tier) if tier is not None else "unknown"] += 1

    distances, cold = exact_reuse_distances(requests)
    return TraceAnalysis(
        requests=requests,
        event_counts=event_counts,
        tier_hits=tier_hits,
        layer_frequency=dict(layer_frequency),
        reuse_distances=distances,
        cold_requests=cold,
        record_bytes=record_bytes,
        dropped=dropped,
    )


def percentile(values: list[int], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def capacity_curve(analysis: TraceAnalysis, capacities: list[int]) -> list[dict]:
    """Hypothetical global LRU over logical `(layer, expert)` identities."""
    total = len(analysis.requests)
    rows = []
    for capacity in capacities:
        hits = sum(distance < capacity for distance in analysis.reuse_distances)
        rows.append(
            {
                "capacity": capacity,
                "hits": hits,
                "misses": total - hits,
                "hit_rate": (hits / total) if total else 0.0,
                "bytes_avoided": hits * analysis.record_bytes,
            }
        )
    return rows


def _layer_request_streams(analysis: TraceAnalysis) -> dict[int, list[tuple[int, int]]]:
    streams: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for layer, expert in analysis.requests:
        streams[layer].append((layer, expert))
    return dict(streams)


def per_layer_lru_curve(analysis: TraceAnalysis, capacities: list[int]) -> list[dict]:
    """Model N independent persistent LRU slots in every observed layer.

    This matches the RAM geometry of `V4_PERSISTENT_EXPERT_SLOTS_PER_LAYER`
    much better than the global LRU curve. The value ratio is measured over the
    complete trace: recurring expert bytes avoided divided by persistent bytes
    reserved. It can be compared directly with the same trace-level ratio for
    deterministic dense residency.
    """

    streams = _layer_request_streams(analysis)
    layer_distances: dict[int, list[int]] = {}
    for layer, requests in streams.items():
        layer_distances[layer], _ = exact_reuse_distances(requests)

    total = len(analysis.requests)
    layers = len(streams)
    rows = []
    for capacity in capacities:
        hits = sum(
            sum(distance < capacity for distance in distances)
            for distances in layer_distances.values()
        )
        resident_slots = layers * capacity
        resident_bytes = resident_slots * analysis.record_bytes
        bytes_avoided = hits * analysis.record_bytes
        rows.append(
            {
                "slots_per_layer": capacity,
                "layers": layers,
                "resident_slots": resident_slots,
                "resident_bytes": resident_bytes,
                "hits": hits,
                "misses": total - hits,
                "hit_rate": (hits / total) if total else 0.0,
                "bytes_avoided": bytes_avoided,
                "trace_value_per_resident_byte": (
                    bytes_avoided / resident_bytes if resident_bytes else 0.0
                ),
            }
        )
    return rows


def per_layer_frequency_oracle_curve(
    analysis: TraceAnalysis, capacities: list[int]
) -> list[dict]:
    """Upper bound for perfectly chosen hot experts in each layer.

    For each layer, choose the N most frequent experts using knowledge of the
    entire trace. A selected expert still misses on its first request and only
    subsequent requests count as avoided loads. The real online hysteresis
    policy cannot beat this curve on the same request sequence, so this is a
    useful rejection test: if even the oracle's bytes-saved/resident-byte is
    below dense residency, persistent expert RAM is not competitive.
    """

    total = len(analysis.requests)
    layers = len(analysis.layer_frequency)
    rows = []
    for capacity in capacities:
        hits = 0
        selected_experts = 0
        for frequency in analysis.layer_frequency.values():
            selected = frequency.most_common(capacity)
            selected_experts += len(selected)
            hits += sum(max(0, count - 1) for _, count in selected)
        resident_slots = layers * capacity
        resident_bytes = resident_slots * analysis.record_bytes
        bytes_avoided = hits * analysis.record_bytes
        rows.append(
            {
                "slots_per_layer": capacity,
                "layers": layers,
                "resident_slots": resident_slots,
                "selected_experts": selected_experts,
                "resident_bytes": resident_bytes,
                "hits": hits,
                "misses": total - hits,
                "hit_rate": (hits / total) if total else 0.0,
                "bytes_avoided": bytes_avoided,
                "trace_value_per_resident_byte": (
                    bytes_avoided / resident_bytes if resident_bytes else 0.0
                ),
            }
        )
    return rows


def summary_dict(
    analysis: TraceAnalysis,
    capacities: list[int],
    top_n: int,
    persistent_capacities: list[int] | None = None,
) -> dict:
    if persistent_capacities is None:
        persistent_capacities = [1, 2, 4, 8]

    top_layers = {}
    for layer in sorted(analysis.layer_frequency):
        total = sum(analysis.layer_frequency[layer].values())
        top = analysis.layer_frequency[layer].most_common(top_n)
        top_layers[str(layer)] = {
            "requests": total,
            "top_experts": [
                {
                    "expert": expert,
                    "requests": count,
                    "share": count / total if total else 0.0,
                }
                for expert, count in top
            ],
        }

    return {
        "requests": len(analysis.requests),
        "unique_experts": len(set(analysis.requests)),
        "cold_requests": analysis.cold_requests,
        "physical_loads": analysis.event_counts.get("load_complete", 0),
        "inflight_joins": analysis.event_counts.get("inflight_join", 0),
        "evictions": analysis.event_counts.get("evict", 0),
        "slot_waits": analysis.event_counts.get("slot_wait", 0),
        "runtime_hits": analysis.event_counts.get("hit", 0),
        "persistent_hits": analysis.tier_hits.get("persistent", 0),
        "transient_hits": analysis.tier_hits.get("transient", 0),
        "record_bytes": analysis.record_bytes,
        "dropped_events": analysis.dropped,
        "reuse_distance": {
            "samples": len(analysis.reuse_distances),
            "p50": percentile(analysis.reuse_distances, 0.50),
            "p90": percentile(analysis.reuse_distances, 0.90),
            "p95": percentile(analysis.reuse_distances, 0.95),
            "p99": percentile(analysis.reuse_distances, 0.99),
        },
        "lru_capacity_curve": capacity_curve(analysis, capacities),
        "persistent_per_layer_lru_curve": per_layer_lru_curve(
            analysis, persistent_capacities
        ),
        "persistent_per_layer_frequency_oracle_curve": (
            per_layer_frequency_oracle_curve(analysis, persistent_capacities)
        ),
        "layers": top_layers,
    }


def parse_capacities(raw: str) -> list[int]:
    capacities = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError("cache capacities must be positive")
        capacities.append(value)
    if not capacities:
        raise ValueError("at least one cache capacity is required")
    return sorted(set(capacities))


def _print_persistent_curve(title: str, rows: list[dict]) -> None:
    print(f"\n{title}:")
    print("  slots/layer   hit-rate     hits   resident   avoided   value/byte")
    for row in rows:
        resident_gib = row["resident_bytes"] / (1024**3)
        avoided_gib = row["bytes_avoided"] / (1024**3)
        print(
            f"  {row['slots_per_layer']:11d}   {100.0 * row['hit_rate']:6.2f}%  "
            f"{row['hits']:7d}   {resident_gib:7.2f}G   {avoided_gib:7.2f}G   "
            f"{row['trace_value_per_resident_byte']:8.3f}x"
        )


def print_human(summary: dict, top_n: int) -> None:
    print(
        "v4 expert trace: "
        f"requests={summary['requests']} unique={summary['unique_experts']} "
        f"loads={summary['physical_loads']} hits={summary['runtime_hits']} "
        f"joins={summary['inflight_joins']} evictions={summary['evictions']} "
        f"slot_waits={summary['slot_waits']} dropped={summary['dropped_events']}"
    )
    rd = summary["reuse_distance"]
    if rd["samples"]:
        print(
            "reuse distance (distinct logical experts since prior use): "
            f"p50={rd['p50']:.0f} p90={rd['p90']:.0f} "
            f"p95={rd['p95']:.0f} p99={rd['p99']:.0f}"
        )
    else:
        print("reuse distance: no repeated expert requests")

    print("\nGlobal LRU working-set curve:")
    print("  slots      hit-rate        hits     avoided")
    for row in summary["lru_capacity_curve"]:
        gib = row["bytes_avoided"] / (1024**3)
        print(
            f"  {row['capacity']:5d}      {100.0 * row['hit_rate']:6.2f}%  "
            f"{row['hits']:10d}   {gib:8.2f} GiB"
        )

    _print_persistent_curve(
        "Per-layer persistent LRU value",
        summary["persistent_per_layer_lru_curve"],
    )
    _print_persistent_curve(
        "Per-layer perfect-hot oracle (upper bound)",
        summary["persistent_per_layer_frequency_oracle_curve"],
    )

    print(f"\nTop {top_n} experts per layer:")
    for layer, info in summary["layers"].items():
        hot = ", ".join(
            f"e{item['expert']}={item['requests']} ({100.0 * item['share']:.1f}%)"
            for item in info["top_experts"]
        )
        print(f"  layer {int(layer):2d}: {hot}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="V4_EXPERT_TRACE JSONL file")
    parser.add_argument(
        "--capacities",
        default="1,2,4,8,16,32,64,128,256,512",
        help="comma-separated hypothetical global LRU capacities",
    )
    parser.add_argument(
        "--persistent-capacities",
        default="1,2,4,8",
        help="comma-separated persistent expert slots to model per layer",
    )
    parser.add_argument("--top", type=int, default=5, help="experts shown per layer")
    parser.add_argument("--json", action="store_true", help="emit machine-readable summary")
    args = parser.parse_args(argv)

    if args.top < 1:
        parser.error("--top must be positive")
    try:
        capacities = parse_capacities(args.capacities)
        persistent_capacities = parse_capacities(args.persistent_capacities)
        analysis = analyze(args.trace)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    summary = summary_dict(
        analysis, capacities, args.top, persistent_capacities
    )
    if args.json:
        json.dump(summary, sys.stdout, sort_keys=True, indent=2)
        sys.stdout.write("\n")
    else:
        print_human(summary, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

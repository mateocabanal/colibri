#!/usr/bin/env python3
"""Analyze cross-request routed-expert locality from a V4 schema-v2 trace.

The main expert-trace analyzer focuses on token/layer locality. This companion
view answers the serving question in #56 without an O(requests^2) sweep: how
much of request N's logical expert working set is reused by request N+1?

Only logical route selections are consumed. Physical lookup fanout does not
change the request working set and is deliberately ignored here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


Expert = tuple[int, int]


def _as_nonnegative_int(value: Any, name: str, line: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"line {line}: {name} must be a non-negative integer")
    return value


def load_requests(path: Path) -> tuple[dict[str, Any], list[tuple[int, set[Expert]]], dict[int, dict[int, set[int]]]]:
    """Return header, ordered request working sets, and per-request per-layer sets."""
    header: dict[str, Any] | None = None
    order: list[int] = []
    seen: set[int] = set()
    experts: dict[int, set[Expert]] = defaultdict(set)
    by_layer: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            if header is None:
                header = row
                if header.get("schema") != "colibri.v4.expert_trace.v2":
                    raise ValueError(
                        "request-overlap analysis requires colibri.v4.expert_trace.v2"
                    )
                continue
            if row.get("event") not in ("request", "route_selected"):
                continue
            request_id = _as_nonnegative_int(row.get("request_id"), "request_id", line_number)
            layer = _as_nonnegative_int(row.get("layer"), "layer", line_number)
            expert = _as_nonnegative_int(row.get("expert"), "expert", line_number)
            if request_id not in seen:
                seen.add(request_id)
                order.append(request_id)
            experts[request_id].add((layer, expert))
            by_layer[request_id][layer].add(expert)

    if header is None:
        raise ValueError("empty trace")
    ordered = [(request_id, experts[request_id]) for request_id in order]
    return header, ordered, by_layer


def _pair_overlap(left: set[Any], right: set[Any]) -> dict[str, float | int]:
    shared = len(left & right)
    union = len(left | right)
    return {
        "left": len(left),
        "right": len(right),
        "shared": shared,
        "union": union,
        "jaccard": shared / union if union else 1.0,
        "left_retained": shared / len(left) if left else 1.0,
        "right_reused": shared / len(right) if right else 1.0,
    }


def analyze_request_overlap(path: Path) -> dict[str, Any]:
    header, requests, by_layer = load_requests(path)
    pairs: list[dict[str, Any]] = []
    layer_acc: dict[int, list[dict[str, float | int]]] = defaultdict(list)

    for index in range(1, len(requests)):
        left_id, left = requests[index - 1]
        right_id, right = requests[index]
        metrics = _pair_overlap(left, right)
        pairs.append({"left_request": left_id, "right_request": right_id, **metrics})

        layers = set(by_layer[left_id]) | set(by_layer[right_id])
        for layer in layers:
            layer_acc[layer].append(
                _pair_overlap(
                    by_layer[left_id].get(layer, set()),
                    by_layer[right_id].get(layer, set()),
                )
            )

    if pairs:
        mean_shared = statistics.fmean(float(pair["shared"]) for pair in pairs)
        mean_jaccard = statistics.fmean(float(pair["jaccard"]) for pair in pairs)
        mean_left_retained = statistics.fmean(
            float(pair["left_retained"]) for pair in pairs
        )
        mean_right_reused = statistics.fmean(
            float(pair["right_reused"]) for pair in pairs
        )
        aggregate_shared = sum(int(pair["shared"]) for pair in pairs)
        aggregate_union = sum(int(pair["union"]) for pair in pairs)
    else:
        mean_shared = mean_jaccard = mean_left_retained = mean_right_reused = 0.0
        aggregate_shared = aggregate_union = 0

    layers_out: list[dict[str, Any]] = []
    for layer in sorted(layer_acc):
        values = layer_acc[layer]
        shared = sum(int(value["shared"]) for value in values)
        union = sum(int(value["union"]) for value in values)
        layers_out.append(
            {
                "layer": layer,
                "pairs": len(values),
                "mean_shared": statistics.fmean(
                    float(value["shared"]) for value in values
                ),
                "mean_jaccard": statistics.fmean(
                    float(value["jaccard"]) for value in values
                ),
                "aggregate_jaccard": shared / union if union else 1.0,
            }
        )

    return {
        "schema": header.get("schema"),
        "requests": len(requests),
        "adjacent_pairs": len(pairs),
        "request_ids": [request_id for request_id, _ in requests],
        "mean_shared": mean_shared,
        "mean_jaccard": mean_jaccard,
        "mean_left_retained": mean_left_retained,
        "mean_right_reused": mean_right_reused,
        "aggregate_shared": aggregate_shared,
        "aggregate_union": aggregate_union,
        "aggregate_jaccard": (
            aggregate_shared / aggregate_union if aggregate_union else 0.0
        ),
        "pairs": pairs,
        "layers": layers_out,
    }


def emit_human(result: dict[str, Any], top_layers: int) -> None:
    print(
        "cross_request "
        f"requests={result['requests']} adjacent_pairs={result['adjacent_pairs']} "
        f"mean_shared={result['mean_shared']:.3f} "
        f"mean_jaccard={result['mean_jaccard']:.4f} "
        f"aggregate_jaccard={result['aggregate_jaccard']:.4f} "
        f"prev_retained={result['mean_left_retained']:.4f} "
        f"next_reused={result['mean_right_reused']:.4f}"
    )
    for pair in result["pairs"]:
        print(
            "request_pair "
            f"left={pair['left_request']} right={pair['right_request']} "
            f"shared={pair['shared']} union={pair['union']} "
            f"jaccard={pair['jaccard']:.4f} "
            f"prev_retained={pair['left_retained']:.4f} "
            f"next_reused={pair['right_reused']:.4f}"
        )
    ranked = sorted(
        result["layers"],
        key=lambda row: (-row["aggregate_jaccard"], row["layer"]),
    )
    for row in ranked[:top_layers]:
        print(
            "request_layer "
            f"layer={row['layer']} pairs={row['pairs']} "
            f"mean_shared={row['mean_shared']:.3f} "
            f"mean_jaccard={row['mean_jaccard']:.4f} "
            f"aggregate_jaccard={row['aggregate_jaccard']:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze adjacent-request expert working-set overlap from V4 v2 traces."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top-layers", type=int, default=10)
    args = parser.parse_args()
    if args.top_layers < 0:
        parser.error("--top-layers must be >= 0")
    try:
        result = analyze_request_overlap(args.trace)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(result, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        emit_human(result, args.top_layers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

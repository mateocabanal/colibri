#!/usr/bin/env python3
"""Compare V4 dense and expert residency by exposed-I/O value per resident byte.

This is the policy reference for #3. It intentionally stays offline until the
inputs and scoring are stable enough to wire into startup planning.

The tool combines:
- a V4 expert trace (for perfect-hot capacity upper bounds),
- measured exposed expert wait/read traffic,
- one or more dense residency benchmark points.

A dense point is:
    resident_gib,avoided_gib,physical_read_gib,exposed_read_ms

With two or more *same-workload* dense points, the tool also reports finite-
difference marginal value between adjacent resident budgets. That is the metric
needed to decide whether the next chunk of RAM belongs to deterministic tensors
or a persistent expert tier.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.analyze_v4_expert_trace import (
    analyze,
    global_frequency_oracle_curve,
    parse_capacities,
)


@dataclass(frozen=True)
class DensePoint:
    resident_gib: float
    avoided_gib: float
    physical_read_gib: float
    exposed_read_ms: float


def parse_dense_point(raw: str) -> DensePoint:
    pieces = [piece.strip() for piece in raw.split(",")]
    if len(pieces) != 4:
        raise ValueError(
            "dense point must be resident_gib,avoided_gib,physical_read_gib,exposed_read_ms"
        )
    try:
        values = [float(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError("dense point fields must be numeric") from exc
    if any(value < 0.0 for value in values):
        raise ValueError("dense point fields must be non-negative")
    if values[0] <= 0.0:
        raise ValueError("dense resident_gib must be positive")
    return DensePoint(*values)


def dense_point_value(point: DensePoint) -> dict:
    ms_per_read_gib = (
        point.exposed_read_ms / point.physical_read_gib
        if point.physical_read_gib
        else 0.0
    )
    estimated_avoided_ms = point.avoided_gib * ms_per_read_gib
    return {
        "resident_gib": point.resident_gib,
        "avoided_gib": point.avoided_gib,
        "physical_read_gib": point.physical_read_gib,
        "exposed_read_ms": point.exposed_read_ms,
        "bytes_value": point.avoided_gib / point.resident_gib,
        "exposed_ms_per_physical_read_gib": ms_per_read_gib,
        "estimated_exposed_ms_avoided": estimated_avoided_ms,
        "estimated_exposed_ms_avoided_per_resident_gib": (
            estimated_avoided_ms / point.resident_gib
        ),
    }


def dense_marginal_frontier(points: list[DensePoint]) -> list[dict]:
    """Finite-difference marginal value for adjacent same-workload A/B points."""

    ordered = sorted(points, key=lambda point: point.resident_gib)
    rows = []
    for low, high in zip(ordered, ordered[1:]):
        delta_resident = high.resident_gib - low.resident_gib
        if delta_resident <= 0.0:
            continue
        delta_avoided = high.avoided_gib - low.avoided_gib
        # More residency should reduce exposed read time. Negative savings are
        # preserved rather than clamped so noisy/non-comparable A/Bs are obvious.
        exposed_ms_saved = low.exposed_read_ms - high.exposed_read_ms
        rows.append(
            {
                "resident_from_gib": low.resident_gib,
                "resident_to_gib": high.resident_gib,
                "delta_resident_gib": delta_resident,
                "delta_avoided_gib": delta_avoided,
                "marginal_bytes_value": delta_avoided / delta_resident,
                "measured_exposed_ms_saved": exposed_ms_saved,
                "measured_exposed_ms_saved_per_resident_gib": (
                    exposed_ms_saved / delta_resident
                ),
            }
        )
    return rows


def expert_value_curve(
    trace: Path,
    capacities: list[int],
    expert_read_gib: float,
    expert_wait_ms: float,
) -> list[dict]:
    analysis = analyze(trace)
    curve = global_frequency_oracle_curve(analysis, capacities)
    ms_per_read_gib = expert_wait_ms / expert_read_gib if expert_read_gib else 0.0
    rows = []
    for row in curve:
        resident_gib = row["resident_bytes"] / (1024**3)
        avoided_gib = row["bytes_avoided"] / (1024**3)
        estimated_ms = avoided_gib * ms_per_read_gib
        rows.append(
            {
                **row,
                "resident_gib": resident_gib,
                "avoided_gib": avoided_gib,
                "exposed_ms_per_physical_read_gib": ms_per_read_gib,
                "estimated_exposed_ms_avoided": estimated_ms,
                "estimated_exposed_ms_avoided_per_resident_gib": (
                    estimated_ms / resident_gib if resident_gib else 0.0
                ),
            }
        )
    return rows


def build_summary(
    trace: Path,
    dense_points: list[DensePoint],
    expert_capacities: list[int],
    expert_read_gib: float,
    expert_wait_ms: float,
) -> dict:
    dense_values = [dense_point_value(point) for point in dense_points]
    expert_values = expert_value_curve(
        trace, expert_capacities, expert_read_gib, expert_wait_ms
    )
    return {
        "metric": "estimated_exposed_io_ms_avoided_per_resident_gib",
        "dense_points": dense_values,
        "dense_marginal_frontier": dense_marginal_frontier(dense_points),
        "expert_global_perfect_hot_upper_bound": expert_values,
        "notes": [
            "expert curve is a perfect-frequency oracle upper bound, not an online-policy prediction",
            "dense marginal rows are valid only when adjacent points use the same workload",
            "estimated avoided time assumes observed exposed-ms/read-GiB remains locally representative",
        ],
    }


def print_human(summary: dict) -> None:
    print("V4 residency value model")
    print(f"metric={summary['metric']}")

    print("\nDense points:")
    print(" resident  avoided  read_GiB  exposed_ms  bytes/value  est_ms/GiB_res")
    for row in summary["dense_points"]:
        print(
            f" {row['resident_gib']:8.2f} {row['avoided_gib']:8.2f} "
            f"{row['physical_read_gib']:9.2f} {row['exposed_read_ms']:11.1f} "
            f"{row['bytes_value']:11.3f}x "
            f"{row['estimated_exposed_ms_avoided_per_resident_gib']:14.1f}"
        )

    frontier = summary["dense_marginal_frontier"]
    if frontier:
        print("\nDense marginal frontier (same-workload points only):")
        print(" from->to GiB  delta_GiB  delta_avoided  bytes/value  measured_ms/GiB_res")
        for row in frontier:
            print(
                f" {row['resident_from_gib']:.2f}->{row['resident_to_gib']:.2f} "
                f"{row['delta_resident_gib']:9.2f} "
                f"{row['delta_avoided_gib']:13.2f} "
                f"{row['marginal_bytes_value']:11.3f}x "
                f"{row['measured_exposed_ms_saved_per_resident_gib']:19.1f}"
            )

    print("\nExpert global perfect-hot upper bound:")
    print(" slots  resident  avoided  bytes/value  est_ms/GiB_res")
    for row in summary["expert_global_perfect_hot_upper_bound"]:
        print(
            f" {row['capacity']:5d} {row['resident_gib']:9.2f} "
            f"{row['avoided_gib']:8.2f} "
            f"{row['trace_value_per_resident_byte']:11.3f}x "
            f"{row['estimated_exposed_ms_avoided_per_resident_gib']:14.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="V4_EXPERT_TRACE JSONL")
    parser.add_argument(
        "--dense-point",
        action="append",
        default=[],
        metavar="R,A,READ,MS",
        help=(
            "dense benchmark point: resident_gib,avoided_gib,"
            "physical_read_gib,exposed_read_ms; repeat for marginal A/B"
        ),
    )
    parser.add_argument(
        "--expert-capacities",
        default="43,86,172,344",
        help="global persistent expert slot capacities",
    )
    parser.add_argument("--expert-read-gib", type=float, required=True)
    parser.add_argument("--expert-wait-ms", type=float, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.expert_read_gib < 0.0 or args.expert_wait_ms < 0.0:
        parser.error("expert read/wait values must be non-negative")
    try:
        dense_points = [parse_dense_point(raw) for raw in args.dense_point]
        if not dense_points:
            raise ValueError("at least one --dense-point is required")
        capacities = parse_capacities(args.expert_capacities)
        summary = build_summary(
            args.trace,
            dense_points,
            capacities,
            args.expert_read_gib,
            args.expert_wait_ms,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        json.dump(summary, sys.stdout, sort_keys=True, indent=2)
        sys.stdout.write("\n")
    else:
        print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

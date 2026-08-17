#!/usr/bin/env python3
"""Analyze how much expert lookup work becomes exposed owner-thread wait.

Inputs are the three #56 streams:
  * physical v1 residency lifecycle (`V4_EXPERT_TRACE`)
  * logical v2 route/lookup sidecar
  * compile-gated expert execution trace (`V4_TRACE_EXEC=1`)

The logical stream groups every routed selection by the physical `lookup_id`
that served it. The execution stream is ordered and carries the same stable
`(layer, expert, lease_generation)` identity plus owner-thread loader-wait deltas.
For each identity, this tool consumes execution records in lookup-group fanout
order. That reconstructs one owner-wait total per physical lookup without
requiring the runtime to duplicate request/token identity into a second stream.

A lookup group is classified as `cold` only when it is the first group for a
lease generation that has a physical v1 `load_complete`. Later lookups reusing
that same generation are resident/hit groups. `lookup_ns` is worker-side lookup
latency; `owner_wait_ns` is time the owner actually blocked. Their ratio is a
useful overlap diagnostic, not a proof that the difference was exclusively
hidden by compute -- scheduling and other concurrent work can contribute too.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


Identity = tuple[int, int, int]


@dataclass
class LookupGroup:
    lookup_id: int
    identity: Identity
    routes: int
    lookup_ns: int
    phase_counts: Counter[str]
    request_ids: set[int]
    owner_wait_ns: int = 0
    execute_ns: int = 0
    cold: bool = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty trace")
    return rows


def nonnegative_int(value: Any, field: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: {field} must be a non-negative integer")
    return value


def identity(row: dict[str, Any], generation_field: str, where: str) -> Identity:
    return (
        nonnegative_int(row.get("layer"), "layer", where),
        nonnegative_int(row.get("expert"), "expert", where),
        nonnegative_int(row.get(generation_field), generation_field, where),
    )


def check_header(row: dict[str, Any], schema: str, label: str) -> None:
    if row.get("schema") != schema:
        raise ValueError(f"{label} trace must use {schema}")
    dropped = nonnegative_int(row.get("dropped", 0), "dropped", f"{label} header")
    if dropped:
        raise ValueError(f"{label} trace is incomplete: dropped={dropped}")


def loaded_identities(rows: list[dict[str, Any]]) -> set[Identity]:
    check_header(rows[0], "colibri.v4.expert_trace.v1", "physical")
    loaded: set[Identity] = set()
    for line_number, row in enumerate(rows[1:], 2):
        if row.get("event") != "load_complete":
            continue
        key = identity(row, "generation", f"physical line {line_number}")
        if key[2] == 0:
            raise ValueError(f"physical line {line_number}: load_complete has zero generation")
        loaded.add(key)
    return loaded


def logical_groups(rows: list[dict[str, Any]]) -> list[LookupGroup]:
    check_header(rows[0], "colibri.v4.expert_trace.v2", "logical")
    for field in ("correlation_misses", "uncorrelated_routes"):
        value = nonnegative_int(rows[0].get(field, 0), field, "logical header")
        if value:
            raise ValueError(f"logical trace is incomplete: {field}={value}")

    raw_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    order: list[int] = []
    for line_number, row in enumerate(rows[1:], 2):
        if row.get("event") not in ("request", "route_selected"):
            continue
        lookup_id = nonnegative_int(row.get("lookup_id"), "lookup_id", f"logical line {line_number}")
        if not lookup_id:
            raise ValueError(f"logical line {line_number}: lookup_id must be non-zero")
        if lookup_id not in raw_groups:
            order.append(lookup_id)
        raw_groups[lookup_id].append(row)
    if not order:
        raise ValueError("logical trace contains no routed lookup groups")

    groups: list[LookupGroup] = []
    for lookup_id in order:
        routes = raw_groups[lookup_id]
        identities = {
            identity(row, "lease_generation", f"lookup {lookup_id}") for row in routes
        }
        if len(identities) != 1:
            raise ValueError(f"lookup {lookup_id}: mixed lease identities {sorted(identities)}")
        key = next(iter(identities))
        if key[2] == 0:
            raise ValueError(f"lookup {lookup_id}: exact-COLI lease generation is zero")
        lookup_ns_values = {
            nonnegative_int(row.get("lookup_ns"), "lookup_ns", f"lookup {lookup_id}")
            for row in routes
        }
        if len(lookup_ns_values) != 1:
            raise ValueError(f"lookup {lookup_id}: inconsistent lookup_ns values")
        results = {row.get("lookup_result") for row in routes}
        if results != {0}:
            raise ValueError(f"lookup {lookup_id}: lookup_result values={sorted(results, key=str)}")
        fanout = len(routes)
        if any(row.get("lookup_routes") != fanout for row in routes):
            raise ValueError(f"lookup {lookup_id}: lookup_routes does not match fanout={fanout}")
        phases: Counter[str] = Counter()
        requests: set[int] = set()
        for row in routes:
            phase = row.get("phase")
            if phase not in ("prefill", "decode"):
                raise ValueError(f"lookup {lookup_id}: invalid phase {phase!r}")
            phases[phase] += 1
            requests.add(nonnegative_int(row.get("request_id"), "request_id", f"lookup {lookup_id}"))
        groups.append(
            LookupGroup(
                lookup_id=lookup_id,
                identity=key,
                routes=fanout,
                lookup_ns=next(iter(lookup_ns_values)),
                phase_counts=phases,
                request_ids=requests,
            )
        )
    expected = rows[0].get("physical_lookups")
    if isinstance(expected, int) and expected != len(groups):
        raise ValueError(f"logical header physical_lookups={expected}, observed={len(groups)}")
    return groups


def execution_queues(rows: list[dict[str, Any]]) -> dict[Identity, deque[dict[str, Any]]]:
    check_header(rows[0], "colibri.v4.expert_execute_trace.v1", "execution")
    queues: dict[Identity, deque[dict[str, Any]]] = defaultdict(deque)
    measured = 0
    total_wait = 0
    total_execute = 0
    events = 0
    for line_number, row in enumerate(rows[1:], 2):
        if row.get("event") != "execute":
            raise ValueError(f"execution line {line_number}: expected execute event")
        where = f"execution line {line_number}"
        key = identity(row, "generation", where)
        if key[2] == 0:
            raise ValueError(f"{where}: exact-COLI generation is zero")
        if row.get("result") != 0:
            raise ValueError(f"{where}: result={row.get('result')!r}")
        wait_measured = row.get("owner_wait_measured")
        if wait_measured is not True:
            raise ValueError(
                f"{where}: owner wait is not measured; rerun with V4_PROFILE=1"
            )
        wait_ns = nonnegative_int(row.get("owner_wait_ns"), "owner_wait_ns", where)
        execute_ns = nonnegative_int(row.get("execute_ns"), "execute_ns", where)
        queues[key].append(row)
        measured += 1
        total_wait += wait_ns
        total_execute += execute_ns
        events += 1
    if not events:
        raise ValueError("execution trace contains no execute events")
    header = rows[0]
    for field, observed in (
        ("events", events),
        ("owner_wait_measured_events", measured),
        ("total_owner_wait_ns", total_wait),
        ("total_execute_ns", total_execute),
    ):
        expected = header.get(field)
        if isinstance(expected, int) and expected != observed:
            raise ValueError(f"execution header {field}={expected}, observed={observed}")
    return queues


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def percentage(value: float | None) -> float | None:
    return value * 100.0 if value is not None else None


def summarize(
    physical: Path,
    logical: Path,
    execution: Path,
    stall_threshold_ms: float = 0.0,
) -> dict[str, Any]:
    physical_rows = read_jsonl(physical)
    logical_rows = read_jsonl(logical)
    execution_rows = read_jsonl(execution)
    builds = {
        row.get("build")
        for row in (physical_rows[0], logical_rows[0], execution_rows[0])
        if row.get("build")
    }
    if len(builds) > 1:
        raise ValueError(f"trace build mismatch: {sorted(builds)}")

    loaded = loaded_identities(physical_rows)
    groups = logical_groups(logical_rows)
    queues = execution_queues(execution_rows)
    seen_loaded: set[Identity] = set()
    threshold_ns = int(max(stall_threshold_ms, 0.0) * 1e6)

    for group in groups:
        queue = queues.get(group.identity)
        if queue is None or len(queue) < group.routes:
            available = 0 if queue is None else len(queue)
            raise ValueError(
                f"lookup {group.lookup_id}: needs {group.routes} executions for "
                f"{group.identity}, only {available} remain"
            )
        for _ in range(group.routes):
            row = queue.popleft()
            group.owner_wait_ns += nonnegative_int(
                row.get("owner_wait_ns"), "owner_wait_ns", f"lookup {group.lookup_id} execution"
            )
            group.execute_ns += nonnegative_int(
                row.get("execute_ns"), "execute_ns", f"lookup {group.lookup_id} execution"
            )
        if group.identity in loaded and group.identity not in seen_loaded:
            group.cold = True
            seen_loaded.add(group.identity)

    leftovers = sum(len(queue) for queue in queues.values())
    if leftovers:
        examples = [
            f"L{key[0]}/E{key[1]}/G{key[2]}:{len(queue)}"
            for key, queue in queues.items() if queue
        ][:8]
        raise ValueError(
            f"execution trace has {leftovers} unconsumed events; examples: {', '.join(examples)}"
        )

    cold = [group for group in groups if group.cold]
    resident = [group for group in groups if not group.cold]
    stalled_cold = [group for group in cold if group.owner_wait_ns > threshold_ns]

    def totals(items: Iterable[LookupGroup]) -> dict[str, int]:
        selected = list(items)
        return {
            "groups": len(selected),
            "routes": sum(group.routes for group in selected),
            "lookup_ns": sum(group.lookup_ns for group in selected),
            "owner_wait_ns": sum(group.owner_wait_ns for group in selected),
            "execute_ns": sum(group.execute_ns for group in selected),
        }

    all_totals = totals(groups)
    cold_totals = totals(cold)
    resident_totals = totals(resident)

    per_expert: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {
            "groups": 0,
            "cold_groups": 0,
            "routes": 0,
            "lookup_ns": 0,
            "owner_wait_ns": 0,
            "execute_ns": 0,
        }
    )
    per_layer: dict[int, dict[str, int]] = defaultdict(
        lambda: {
            "groups": 0,
            "cold_groups": 0,
            "routes": 0,
            "lookup_ns": 0,
            "owner_wait_ns": 0,
            "execute_ns": 0,
        }
    )
    phase_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"routes": 0, "owner_wait_ns": 0, "lookup_groups": 0}
    )

    for group in groups:
        layer, expert, _ = group.identity
        for bucket in (per_expert[(layer, expert)], per_layer[layer]):
            bucket["groups"] += 1
            bucket["cold_groups"] += int(group.cold)
            bucket["routes"] += group.routes
            bucket["lookup_ns"] += group.lookup_ns
            bucket["owner_wait_ns"] += group.owner_wait_ns
            bucket["execute_ns"] += group.execute_ns
        # A batch-union lookup can span phases only if caller construction is
        # changed in the future. Split route counts exactly; assign physical
        # lookup-group owner wait to the dominant/current phase only when all
        # routes agree, otherwise leave it in an explicit mixed bucket.
        phase_name = next(iter(group.phase_counts)) if len(group.phase_counts) == 1 else "mixed"
        phase_totals[phase_name]["routes"] += group.routes
        phase_totals[phase_name]["owner_wait_ns"] += group.owner_wait_ns
        phase_totals[phase_name]["lookup_groups"] += 1

    expert_rows = [
        {"layer": layer, "expert": expert, **values}
        for (layer, expert), values in per_expert.items()
    ]
    expert_rows.sort(key=lambda row: (-row["owner_wait_ns"], row["layer"], row["expert"]))
    layer_rows = [
        {"layer": layer, **values} for layer, values in per_layer.items()
    ]
    layer_rows.sort(key=lambda row: (-row["owner_wait_ns"], row["layer"]))

    cold_exposure = ratio(cold_totals["owner_wait_ns"], cold_totals["lookup_ns"])
    result: dict[str, Any] = {
        "build": next(iter(builds)) if builds else None,
        "stall_threshold_ms": stall_threshold_ms,
        "lookup_groups": all_totals["groups"],
        "logical_routes": all_totals["routes"],
        "worker_lookup_ns": all_totals["lookup_ns"],
        "owner_wait_ns": all_totals["owner_wait_ns"],
        "execute_ns": all_totals["execute_ns"],
        "cold_lookup_groups": cold_totals["groups"],
        "cold_routes": cold_totals["routes"],
        "cold_worker_lookup_ns": cold_totals["lookup_ns"],
        "cold_owner_wait_ns": cold_totals["owner_wait_ns"],
        "cold_execute_ns": cold_totals["execute_ns"],
        "resident_lookup_groups": resident_totals["groups"],
        "resident_owner_wait_ns": resident_totals["owner_wait_ns"],
        "stalled_cold_groups": len(stalled_cold),
        "stalled_cold_fraction": ratio(len(stalled_cold), len(cold)),
        "cold_lookup_exposure_ratio": cold_exposure,
        "cold_lookup_hidden_fraction_estimate": (
            max(0.0, 1.0 - cold_exposure) if cold_exposure is not None else None
        ),
        "phase_totals": dict(sorted(phase_totals.items())),
        "top_experts": expert_rows,
        "top_layers": layer_rows,
    }
    return result


def ms(ns: int) -> float:
    return ns / 1e6


def fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100.0:.2f}%"


def emit_human(result: dict[str, Any], top: int) -> None:
    print(
        "expert_stalls "
        f"build={result['build']} groups={result['lookup_groups']} "
        f"routes={result['logical_routes']} cold_groups={result['cold_lookup_groups']} "
        f"resident_groups={result['resident_lookup_groups']}"
    )
    print(
        "latency "
        f"worker_lookup_ms={ms(result['worker_lookup_ns']):.3f} "
        f"owner_wait_ms={ms(result['owner_wait_ns']):.3f} "
        f"expert_execute_ms={ms(result['execute_ns']):.3f}"
    )
    print(
        "cold "
        f"worker_lookup_ms={ms(result['cold_worker_lookup_ns']):.3f} "
        f"owner_wait_ms={ms(result['cold_owner_wait_ns']):.3f} "
        f"exposure={fmt_percent(result['cold_lookup_exposure_ratio'])} "
        f"hidden_estimate={fmt_percent(result['cold_lookup_hidden_fraction_estimate'])} "
        f"stalled_groups={result['stalled_cold_groups']}/{result['cold_lookup_groups']} "
        f"threshold_ms={result['stall_threshold_ms']:.3f}"
    )
    if result["resident_lookup_groups"]:
        print(
            "resident "
            f"owner_wait_ms={ms(result['resident_owner_wait_ns']):.3f}"
        )
    if result["phase_totals"]:
        print("phases")
        for phase, values in result["phase_totals"].items():
            print(
                f"  {phase}: groups={values['lookup_groups']} routes={values['routes']} "
                f"owner_wait_ms={ms(values['owner_wait_ns']):.3f}"
            )
    if top > 0 and result["top_experts"]:
        print("top_experts_by_owner_wait")
        for row in result["top_experts"][:top]:
            print(
                f"  L{row['layer']} E{row['expert']}: groups={row['groups']} "
                f"cold={row['cold_groups']} routes={row['routes']} "
                f"lookup_ms={ms(row['lookup_ns']):.3f} "
                f"owner_wait_ms={ms(row['owner_wait_ns']):.3f}"
            )
    if top > 0 and result["top_layers"]:
        print("top_layers_by_owner_wait")
        for row in result["top_layers"][:top]:
            print(
                f"  L{row['layer']}: groups={row['groups']} cold={row['cold_groups']} "
                f"routes={row['routes']} lookup_ms={ms(row['lookup_ns']):.3f} "
                f"owner_wait_ms={ms(row['owner_wait_ns']):.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report worker expert lookup time versus exposed owner-thread wait."
    )
    parser.add_argument("physical", type=Path, help="physical expert_trace.v1 JSONL")
    parser.add_argument("logical", type=Path, help="logical expert_trace.v2 JSONL")
    parser.add_argument("execution", type=Path, help="expert_execute_trace.v1 JSONL")
    parser.add_argument(
        "--stall-threshold-ms",
        type=float,
        default=0.0,
        help="count a cold lookup group as stalled only above this owner-wait threshold",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.stall_threshold_ms) or args.stall_threshold_ms < 0:
        parser.error("--stall-threshold-ms must be a finite non-negative number")
    if args.top < 0:
        parser.error("--top must be non-negative")
    try:
        result = summarize(
            args.physical,
            args.logical,
            args.execution,
            stall_threshold_ms=args.stall_threshold_ms,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(result, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        emit_human(result, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

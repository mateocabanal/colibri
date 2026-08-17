#!/usr/bin/env python3
"""Validate V4 physical-v1, logical-v2, and optional execution traces.

This is the exact-COLI closure gate for #56. The logical sidecar records one
route selection per `(request, token, layer, rank)`, while the physical store
records residency/load lifecycle events. Successful exact-COLI leases carry a
non-zero generation. `(layer, expert, generation)` is therefore the stable join
key between the streams even when one physical lookup serves many logical routes
during batch-union prefill.

When a V4_TRACE_EXEC=1 build also supplies an execution trace, the validator
requires one successful expert-forward execution for every logical route
selection, grouped by that same stable lease identity. With V4_PROFILE=1 the
execution stream also carries the canonical owner-thread loader-wait delta
sampled immediately before each expert executes. This remains separate from
worker `lookup_ns`; the two measurements answer different questions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


Identity = tuple[int, int, int]


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


def as_nonnegative_int(value: Any, field: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: {field} must be a non-negative integer")
    return value


def identity(row: dict[str, Any], where: str, generation_field: str = "generation") -> Identity:
    return (
        as_nonnegative_int(row.get("layer"), "layer", where),
        as_nonnegative_int(row.get("expert"), "expert", where),
        as_nonnegative_int(row.get(generation_field), generation_field, where),
    )


def validate_execution_trace(
    execution_path: Path,
    logical_events: list[dict[str, Any]],
    expected_build: Any,
    loaded_identities: set[Identity],
) -> dict[str, Any]:
    rows = read_jsonl(execution_path)
    header = rows[0]
    if header.get("schema") != "colibri.v4.expert_execute_trace.v1":
        raise ValueError(
            "execution trace must use colibri.v4.expert_execute_trace.v1"
        )
    if header.get("source") not in (None, "expert_execute"):
        raise ValueError(f"execution trace has unexpected source={header.get('source')!r}")
    build = header.get("build")
    if expected_build and build and expected_build != build:
        raise ValueError(
            f"execution build mismatch: expected={expected_build} execution={build}"
        )
    dropped = as_nonnegative_int(header.get("dropped", 0), "dropped", "execution header")
    if dropped:
        raise ValueError(f"execution trace is incomplete: dropped={dropped}")

    logical_counts: Counter[Identity] = Counter(
        identity(row, "logical route", "lease_generation")
        for row in logical_events
    )
    execution_counts: Counter[Identity] = Counter()
    first_execution_seen: set[Identity] = set()
    total_execute_ns = 0
    total_owner_wait_ns = 0
    measured_owner_wait_events = 0
    cold_first_owner_wait_ns = 0
    cold_first_owner_wait_measured = 0
    execution_rows = 0
    resident_bytes: set[int] = set()
    for line_number, row in enumerate(rows[1:], 2):
        if row.get("event") != "execute":
            raise ValueError(
                f"execution line {line_number}: expected event='execute'"
            )
        where = f"execution line {line_number}"
        key = identity(row, where)
        if key[2] == 0:
            raise ValueError(
                f"{where}: exact-COLI execution has zero generation for "
                f"layer={key[0]} expert={key[1]}"
            )
        if row.get("result") != 0:
            raise ValueError(f"{where}: result={row.get('result')!r}")
        elapsed = as_nonnegative_int(row.get("execute_ns"), "execute_ns", where)
        bytes_here = as_nonnegative_int(
            row.get("resident_bytes"), "resident_bytes", where
        )
        if bytes_here == 0:
            raise ValueError(f"{where}: resident_bytes must be non-zero")
        for field in ("gate_format", "down_format", "up_format"):
            as_nonnegative_int(row.get(field), field, where)

        owner_wait_measured = row.get("owner_wait_measured", False)
        if not isinstance(owner_wait_measured, bool):
            raise ValueError(f"{where}: owner_wait_measured must be boolean")
        owner_wait_ns = as_nonnegative_int(
            row.get("owner_wait_ns", 0), "owner_wait_ns", where
        )
        if not owner_wait_measured and owner_wait_ns:
            raise ValueError(
                f"{where}: owner_wait_ns={owner_wait_ns} while measurement is disabled"
            )
        if owner_wait_measured:
            measured_owner_wait_events += 1
            total_owner_wait_ns += owner_wait_ns

        is_first = key not in first_execution_seen
        first_execution_seen.add(key)
        if is_first and key in loaded_identities and owner_wait_measured:
            cold_first_owner_wait_measured += 1
            cold_first_owner_wait_ns += owner_wait_ns

        execution_counts[key] += 1
        total_execute_ns += elapsed
        resident_bytes.add(bytes_here)
        execution_rows += 1

    header_events = header.get("events")
    if isinstance(header_events, int) and header_events != execution_rows:
        raise ValueError(
            f"execution header events={header_events}, observed={execution_rows}"
        )
    header_total = header.get("total_execute_ns")
    if isinstance(header_total, int) and header_total != total_execute_ns:
        raise ValueError(
            f"execution header total_execute_ns={header_total}, "
            f"observed={total_execute_ns}"
        )
    header_measured = header.get("owner_wait_measured_events")
    if isinstance(header_measured, int) and header_measured != measured_owner_wait_events:
        raise ValueError(
            f"execution header owner_wait_measured_events={header_measured}, "
            f"observed={measured_owner_wait_events}"
        )
    header_wait = header.get("total_owner_wait_ns")
    if isinstance(header_wait, int) and header_wait != total_owner_wait_ns:
        raise ValueError(
            f"execution header total_owner_wait_ns={header_wait}, "
            f"observed={total_owner_wait_ns}"
        )

    if execution_counts != logical_counts:
        missing = logical_counts - execution_counts
        extra = execution_counts - logical_counts
        details: list[str] = []
        if missing:
            details.append(
                "missing="
                + ",".join(
                    f"L{layer}/E{expert}/G{generation}x{count}"
                    for (layer, expert, generation), count in missing.most_common(8)
                )
            )
        if extra:
            details.append(
                "extra="
                + ",".join(
                    f"L{layer}/E{expert}/G{generation}x{count}"
                    for (layer, expert, generation), count in extra.most_common(8)
                )
            )
        raise ValueError(
            "expert execution count does not match logical routes by lease identity: "
            + " ".join(details)
        )

    return {
        "execution_events": execution_rows,
        "execution_total_ns": total_execute_ns,
        "execution_unique_lease_identities": len(execution_counts),
        "execution_resident_byte_sizes": sorted(resident_bytes),
        "owner_wait_measured_events": measured_owner_wait_events,
        "owner_wait_total_ns": total_owner_wait_ns,
        "cold_loaded_generations": len(loaded_identities & set(execution_counts)),
        "cold_first_execute_owner_wait_measured": cold_first_owner_wait_measured,
        "cold_first_execute_owner_wait_ns": cold_first_owner_wait_ns,
    }


def validate_pair(
    physical_path: Path,
    logical_path: Path,
    execution_path: Path | None = None,
) -> dict[str, Any]:
    physical_rows = read_jsonl(physical_path)
    logical_rows = read_jsonl(logical_path)
    physical_header = physical_rows[0]
    logical_header = logical_rows[0]

    if physical_header.get("schema") != "colibri.v4.expert_trace.v1":
        raise ValueError("physical trace must use colibri.v4.expert_trace.v1")
    if logical_header.get("schema") != "colibri.v4.expert_trace.v2":
        raise ValueError("logical trace must use colibri.v4.expert_trace.v2")

    physical_build = physical_header.get("build")
    logical_build = logical_header.get("build")
    if physical_build and logical_build and physical_build != logical_build:
        raise ValueError(
            f"trace build mismatch: physical={physical_build} logical={logical_build}"
        )

    for field in ("dropped",):
        value = as_nonnegative_int(physical_header.get(field, 0), field, "physical header")
        if value:
            raise ValueError(f"physical trace is incomplete: {field}={value}")
    for field in ("dropped", "correlation_misses", "uncorrelated_routes"):
        value = as_nonnegative_int(logical_header.get(field, 0), field, "logical header")
        if value:
            raise ValueError(f"logical trace is incomplete: {field}={value}")

    lifecycle_names = {
        "hit",
        "inflight_join",
        "load_begin",
        "load_complete",
        "load_failed",
        "evict",
        "release",
        "slot_wait",
    }
    physical_identities: set[Identity] = set()
    loaded_identities: set[Identity] = set()
    physical_events: Counter[str] = Counter()
    for index, row in enumerate(physical_rows[1:], 2):
        event = row.get("event")
        if event == "layer_summary":
            continue
        if not isinstance(event, str):
            raise ValueError(f"physical line {index}: missing event name")
        physical_events[event] += 1
        generation = as_nonnegative_int(
            row.get("generation", 0), "generation", f"physical line {index}"
        )
        if generation and event in lifecycle_names:
            key = identity(row, f"physical line {index}")
            physical_identities.add(key)
            if event == "load_complete":
                loaded_identities.add(key)

    logical_events = [
        row for row in logical_rows[1:]
        if row.get("event") in ("request", "route_selected")
    ]
    if not logical_events:
        raise ValueError("logical trace contains no route selections")

    missing: Counter[Identity] = Counter()
    lookup_groups: dict[int, list[dict[str, Any]]] = {}
    request_ids: set[int] = set()
    phases: Counter[str] = Counter()
    for index, row in enumerate(logical_events, 2):
        where = f"logical route {index - 1}"
        layer = as_nonnegative_int(row.get("layer"), "layer", where)
        expert = as_nonnegative_int(row.get("expert"), "expert", where)
        generation = as_nonnegative_int(
            row.get("lease_generation"), "lease_generation", where
        )
        if generation == 0:
            raise ValueError(
                f"{where}: exact-COLI route has zero lease_generation "
                f"for layer={layer} expert={expert}"
            )
        key = (layer, expert, generation)
        if key not in physical_identities:
            missing[key] += 1

        lookup_id = as_nonnegative_int(row.get("lookup_id"), "lookup_id", where)
        if lookup_id == 0:
            raise ValueError(f"{where}: lookup_id must be non-zero")
        lookup_result = row.get("lookup_result")
        if lookup_result != 0:
            raise ValueError(f"{where}: lookup_result={lookup_result!r}")
        lookup_groups.setdefault(lookup_id, []).append(row)

        request_ids.add(
            as_nonnegative_int(row.get("request_id"), "request_id", where)
        )
        phase = row.get("phase")
        if phase not in ("prefill", "decode"):
            raise ValueError(f"{where}: invalid/unknown phase {phase!r}")
        phases[phase] += 1

    if missing:
        examples = ", ".join(
            f"L{layer}/E{expert}/G{generation} x{count}"
            for (layer, expert, generation), count in missing.most_common(8)
        )
        raise ValueError(
            f"{sum(missing.values())} logical routes do not join to the physical "
            f"lifecycle; examples: {examples}"
        )

    for lookup_id, rows in lookup_groups.items():
        expected = len(rows)
        identities = {
            (row["layer"], row["expert"], row["lease_generation"])
            for row in rows
        }
        if len(identities) != 1:
            raise ValueError(
                f"lookup {lookup_id}: mixed physical identities {sorted(identities)}"
            )
        for row in rows:
            if row.get("lookup_routes") != expected:
                raise ValueError(
                    f"lookup {lookup_id}: fanout says {row.get('lookup_routes')}, "
                    f"actual logical routes={expected}"
                )

    header_lookups = logical_header.get("physical_lookups")
    if isinstance(header_lookups, int) and header_lookups != len(lookup_groups):
        raise ValueError(
            f"logical header physical_lookups={header_lookups}, "
            f"observed lookup groups={len(lookup_groups)}"
        )

    result: dict[str, Any] = {
        "build": logical_build or physical_build,
        "logical_routes": len(logical_events),
        "physical_events": sum(physical_events.values()),
        "physical_lookup_groups": len(lookup_groups),
        "requests": len(request_ids),
        "request_ids": sorted(request_ids),
        "prefill_routes": phases["prefill"],
        "decode_routes": phases["decode"],
        "unique_lease_identities": len(
            {
                (row["layer"], row["expert"], row["lease_generation"])
                for row in logical_events
            }
        ),
        "physical_event_counts": dict(sorted(physical_events.items())),
        "loaded_lease_identities": len(loaded_identities),
        "execution_validated": execution_path is not None,
    }
    if execution_path is not None:
        result.update(
            validate_execution_trace(
                execution_path,
                logical_events,
                logical_build or physical_build,
                loaded_identities,
            )
        )
    return result


def emit_human(result: dict[str, Any]) -> None:
    line = (
        "trace_pair status=ok "
        f"build={result['build']} requests={result['requests']} "
        f"routes={result['logical_routes']} lookups={result['physical_lookup_groups']} "
        f"lease_identities={result['unique_lease_identities']} "
        f"prefill_routes={result['prefill_routes']} "
        f"decode_routes={result['decode_routes']} "
        f"physical_events={result['physical_events']}"
    )
    if result.get("execution_validated"):
        line += (
            f" executions={result['execution_events']}"
            f" execute_ms={result['execution_total_ns'] / 1e6:.3f}"
            f" owner_wait_measured={result['owner_wait_measured_events']}"
            f" owner_wait_ms={result['owner_wait_total_ns'] / 1e6:.3f}"
            f" cold_first_wait_ms={result['cold_first_execute_owner_wait_ns'] / 1e6:.3f}"
        )
    print(line)
    counts = result["physical_event_counts"]
    if counts:
        print(
            "physical_events "
            + " ".join(f"{name}={count}" for name, count in counts.items())
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate V4 physical v1 and logical v2 trace correlation, with "
            "optional expert execution/owner-wait lifecycle validation."
        )
    )
    parser.add_argument("physical", type=Path)
    parser.add_argument("logical", type=Path)
    parser.add_argument(
        "--execution",
        type=Path,
        help="optional colibri.v4.expert_execute_trace.v1 file",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_pair(args.physical, args.logical, args.execution)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(result, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        emit_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

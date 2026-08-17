#!/usr/bin/env python3
"""Validate a V4 physical-v1 + logical-v2 trace pair.

This is the exact-COLI closure gate for #56. The logical sidecar records one
route selection per `(request, token, layer, rank)`, while the physical store
records residency/load lifecycle events. Successful exact-COLI leases carry a
non-zero generation. `(layer, expert, generation)` is therefore the stable join
key between the two streams even when one physical lookup serves many logical
routes during batch-union prefill.

The validator intentionally does not infer cold-vs-hit from generation alone:
a resident expert can be hit repeatedly under the same generation. It proves
identity/correlation, not a lifecycle state the physical trace did not attach
to the logical lookup id.
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


def identity(row: dict[str, Any], where: str) -> Identity:
    return (
        as_nonnegative_int(row.get("layer"), "layer", where),
        as_nonnegative_int(row.get("expert"), "expert", where),
        as_nonnegative_int(row.get("generation"), "generation", where),
    )


def validate_pair(physical_path: Path, logical_path: Path) -> dict[str, Any]:
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
            physical_identities.add(identity(row, f"physical line {index}"))

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

    return {
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
    }


def emit_human(result: dict[str, Any]) -> None:
    print(
        "trace_pair status=ok "
        f"build={result['build']} requests={result['requests']} "
        f"routes={result['logical_routes']} lookups={result['physical_lookup_groups']} "
        f"lease_identities={result['unique_lease_identities']} "
        f"prefill_routes={result['prefill_routes']} "
        f"decode_routes={result['decode_routes']} "
        f"physical_events={result['physical_events']}"
    )
    counts = result["physical_event_counts"]
    if counts:
        print(
            "physical_events "
            + " ".join(f"{name}={count}" for name, count in counts.items())
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate V4 physical v1 and logical v2 trace correlation."
    )
    parser.add_argument("physical", type=Path)
    parser.add_argument("logical", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_pair(args.physical, args.logical)
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

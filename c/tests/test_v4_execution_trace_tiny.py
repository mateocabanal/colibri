#!/usr/bin/env python3
"""Exercise the compile-gated V4 expert execution trace on the tiny target."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


def token_prompt(ids: list[int]) -> str:
    return "".join(f"<t{token:03d}>" for token in ids)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path.name}:{number}: malformed JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AssertionError(f"{path.name}:{number}: expected JSON object")
        rows.append(row)
    if not rows:
        raise AssertionError(f"{path.name}: empty trace")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    fixture = args.fixture.resolve()
    reference = json.loads((fixture / "ref.json").read_text(encoding="utf-8"))
    case = reference["cases"]["short"]
    max_new = min(2, int(case["max_new_tokens"]))

    with tempfile.TemporaryDirectory(prefix="colibri-v4-exec-") as directory:
        temporary = Path(directory)
        routes = temporary / "routes.jsonl"
        executions = temporary / "executions.jsonl"
        record = temporary / "record.json"
        env = dict(
            os.environ,
            V4_ROUTE_TRACE=routes.as_posix(),
            V4_ROUTE_TRACE_CAP="65536",
            V4_ROUTE_REQUEST_ID="901",
            V4_EXEC_TRACE=executions.as_posix(),
            V4_EXEC_TRACE_CAP="65536",
            V4_PROFILE="1",
        )
        result = subprocess.run(
            [
                binary.as_posix(),
                fixture.as_posix(),
                token_prompt(case["prompt_ids"]),
                "--raw-prompt",
                "--max-tokens",
                str(max_new),
                "--no-dspark",
                "--record-oracle",
                record.as_posix(),
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            env=env,
        )
        if result.returncode:
            raise AssertionError(
                f"execution trace run failed ({result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if "v4_execute_trace status=written" not in result.stderr:
            raise AssertionError(f"execution trace was not flushed:\n{result.stderr}")
        if "owner_wait_measured=" not in result.stderr:
            raise AssertionError(f"execution trace did not report owner wait:\n{result.stderr}")

        actual = json.loads(record.read_text(encoding="utf-8"))
        expected_full = list(case["prompt_ids"]) + list(case["greedy_new_ids"][:max_new])
        if actual.get("full_ids") != expected_full:
            raise AssertionError(
                "V4_PROFILE + execution tracing changed model output: "
                f"expected {expected_full}, got {actual.get('full_ids')}"
            )

        route_rows = read_jsonl(routes)
        execute_rows = read_jsonl(executions)
        route_header, execute_header = route_rows[0], execute_rows[0]
        if route_header.get("schema") != "colibri.v4.expert_trace.v2":
            raise AssertionError(f"bad route header: {route_header}")
        if execute_header.get("schema") != "colibri.v4.expert_execute_trace.v1":
            raise AssertionError(f"bad execution header: {execute_header}")
        if execute_header.get("dropped") != 0:
            raise AssertionError(f"execution trace dropped events: {execute_header}")

        routes_data = [row for row in route_rows[1:] if row.get("event") == "request"]
        executions_data = [
            row for row in execute_rows[1:] if row.get("event") == "execute"
        ]
        if not routes_data or not executions_data:
            raise AssertionError("route/execution trace contains no expert work")
        if execute_header.get("events") != len(executions_data):
            raise AssertionError(
                f"execution event count mismatch: {execute_header} vs {len(executions_data)}"
            )

        route_counts = Counter(
            (row.get("layer"), row.get("expert"), row.get("lease_generation"))
            for row in routes_data
        )
        execute_counts = Counter(
            (row.get("layer"), row.get("expert"), row.get("generation"))
            for row in executions_data
        )
        if route_counts != execute_counts:
            raise AssertionError(
                "one expert execution was not recorded per logical route: "
                f"routes={route_counts} executions={execute_counts}"
            )

        execute_total = 0
        owner_wait_total = 0
        measured = 0
        for row in executions_data:
            execute_ns = row.get("execute_ns")
            owner_wait_ns = row.get("owner_wait_ns")
            owner_wait_measured = row.get("owner_wait_measured")
            if not isinstance(execute_ns, int) or execute_ns < 0:
                raise AssertionError(f"bad execute_ns: {row}")
            if not isinstance(owner_wait_ns, int) or owner_wait_ns < 0:
                raise AssertionError(f"bad owner_wait_ns: {row}")
            if owner_wait_measured is not True:
                raise AssertionError(f"V4_PROFILE=1 did not measure owner wait: {row}")
            if row.get("result") != 0:
                raise AssertionError(f"expert execution failed: {row}")
            if not isinstance(row.get("resident_bytes"), int) or row["resident_bytes"] <= 0:
                raise AssertionError(f"execution is missing resident bytes: {row}")
            execute_total += execute_ns
            owner_wait_total += owner_wait_ns
            measured += 1

        if execute_header.get("total_execute_ns") != execute_total:
            raise AssertionError(
                f"execution duration total mismatch: {execute_header} vs {execute_total}"
            )
        if execute_header.get("owner_wait_measured_events") != measured:
            raise AssertionError(
                f"owner wait measurement count mismatch: {execute_header} vs {measured}"
            )
        if execute_header.get("total_owner_wait_ns") != owner_wait_total:
            raise AssertionError(
                f"owner wait total mismatch: {execute_header} vs {owner_wait_total}"
            )

        print(
            "PASS target execution trace: token-exact + one execution per route + "
            f"owner wait measured ({len(executions_data)} executions, "
            f"{owner_wait_total / 1e6:.3f} ms owner wait)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

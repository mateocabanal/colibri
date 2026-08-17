#!/usr/bin/env python3
"""Run the generated tiny V4 as a native exact-COLI package and validate #56.

Unlike the safetensors tiny oracle, this exercises `coli_v4_expert_store.c`: the
physical lifecycle has real reusable slot generations, so the full v1 <-> v2 <->
execution join can be checked. Intended for the arm64 macOS CI runner after
`colic compile --target native --quant exact --codec none`.

Do not use the CLI's `--record-oracle` helper here. That diagnostic currently
rebuilds its teacher-forcing state with safetensors-only `load_embedding()` and
`final_hidden()` calls, so a package-only engine (`target_index == NULL`) exits 1
after otherwise successful generation. The normal runtime path is package-aware;
this gate compares its emitted `generated_text=` against the fixture's exact
expected token text instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_module("validate_v4_trace_pair_ci", TOOLS / "validate_v4_trace_pair.py")
STALLS = load_module("analyze_v4_expert_stalls_ci", TOOLS / "analyze_v4_expert_stalls.py")


def token_prompt(ids: list[int]) -> str:
    return "".join(f"<t{token:03d}>" for token in ids)


def emitted_generated_text(stderr: str) -> str:
    for line in stderr.splitlines():
        if line.startswith("generated_text="):
            return line.removeprefix("generated_text=")
    raise AssertionError(f"runtime did not emit generated_text=:\n{stderr}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    package = args.package.resolve()
    reference = json.loads((package / "ref.json").read_text(encoding="utf-8"))
    case = reference["cases"]["short"]
    max_new = min(2, int(case["max_new_tokens"]))
    expected_ids = list(case["greedy_new_ids"][:max_new])
    expected_text = token_prompt([token for token in expected_ids if token != 1])

    with tempfile.TemporaryDirectory(prefix="colibri-v4-coli-trace-") as directory:
        temporary = Path(directory)
        physical = temporary / "physical.jsonl"
        execution = temporary / "execution.jsonl"
        logical = Path(str(physical) + ".routes.jsonl")
        env = dict(
            os.environ,
            V4_EXPERT_TRACE=physical.as_posix(),
            V4_EXPERT_TRACE_CAP="65536",
            V4_EXEC_TRACE=execution.as_posix(),
            V4_EXEC_TRACE_CAP="65536",
            V4_PROFILE="1",
            V4_PROGRESS="0",
        )
        result = subprocess.run(
            [
                binary.as_posix(),
                package.as_posix(),
                token_prompt(case["prompt_ids"]),
                "--raw-prompt",
                "--max-tokens",
                str(max_new),
                "--no-dspark",
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
                f"exact-COLI trace run failed ({result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        for marker in (
            "v4_coli mode=package-only",
            "v4_expert_trace status=written",
            "v4_route_trace status=written",
            "v4_execute_trace status=written",
        ):
            if marker not in result.stderr:
                raise AssertionError(f"missing runtime marker {marker!r}:\n{result.stderr}")

        actual_text = emitted_generated_text(result.stderr)
        if actual_text != expected_text:
            raise AssertionError(
                "exact COLI + detailed tracing changed model output: "
                f"expected ids={expected_ids} text={expected_text!r}, "
                f"got text={actual_text!r}"
            )

        pair = VALIDATOR.validate_pair(physical, logical, execution)
        if pair["logical_routes"] <= 0 or pair["execution_events"] != pair["logical_routes"]:
            raise AssertionError(f"route/execution coverage mismatch: {pair}")
        if pair["loaded_lease_identities"] <= 0:
            raise AssertionError(f"COLI trace observed no loaded expert generations: {pair}")
        if pair["cold_loaded_generations"] <= 0:
            raise AssertionError(f"execution trace joined no cold COLI generation: {pair}")
        if pair["owner_wait_measured_events"] != pair["execution_events"]:
            raise AssertionError(f"owner wait is not measured for every execution: {pair}")

        stalls = STALLS.summarize(physical, logical, execution)
        if stalls["cold_lookup_groups"] <= 0:
            raise AssertionError(f"stall analysis found no cold lookup groups: {stalls}")
        if stalls["worker_lookup_ns"] <= 0:
            raise AssertionError(f"stall analysis found no lookup timing: {stalls}")
        if stalls["logical_routes"] != pair["logical_routes"]:
            raise AssertionError(
                f"stall/validator route counts disagree: stalls={stalls} pair={pair}"
            )

        exposure = stalls["cold_lookup_exposure_ratio"]
        exposure_text = "n/a" if exposure is None else f"{100.0 * exposure:.2f}%"
        print(
            "PASS exact COLI trace: token-exact + physical/logical/execution join + "
            f"owner wait ({pair['logical_routes']} routes, "
            f"{pair['loaded_lease_identities']} loaded generations, "
            f"{pair['owner_wait_total_ns'] / 1e6:.3f} ms owner wait, "
            f"cold exposure={exposure_text})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent.parent / "tools"
SPEC = importlib.util.spec_from_file_location(
    "analyze_v4_expert_stalls", TOOLS / "analyze_v4_expert_stalls.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExpertStallTests(unittest.TestCase):
    def write_trace(self, rows: list[dict[str, object]]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".jsonl", delete=False
        )
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        path = Path(handle.name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def physical(self) -> Path:
        return self.write_trace(
            [
                {"schema": "colibri.v4.expert_trace.v1", "build": "abc",
                 "events": 6, "dropped": 0},
                {"event": "load_begin", "layer": 0, "expert": 2,
                 "tier": "transient", "generation": 7, "bytes": 100},
                {"event": "load_complete", "layer": 0, "expert": 2,
                 "tier": "transient", "generation": 7, "bytes": 100},
                {"event": "release", "layer": 0, "expert": 2,
                 "tier": "transient", "generation": 7, "bytes": 0},
                {"event": "load_begin", "layer": 1, "expert": 1,
                 "tier": "transient", "generation": 9, "bytes": 100},
                {"event": "load_complete", "layer": 1, "expert": 1,
                 "tier": "transient", "generation": 9, "bytes": 100},
                {"event": "release", "layer": 1, "expert": 1,
                 "tier": "transient", "generation": 9, "bytes": 0},
            ]
        )

    def logical(self) -> Path:
        # Lookup 10 is cold with two logical routes; lookup 11 reuses the same
        # generation; lookup 12 is another cold generation in decode.
        return self.write_trace(
            [
                {"schema": "colibri.v4.expert_trace.v2", "build": "abc",
                 "events": 4, "dropped": 0, "physical_lookups": 3,
                 "correlation_misses": 0, "uncorrelated_routes": 0},
                {"event": "request", "layer": 0, "expert": 2,
                 "lease_generation": 7, "lookup_id": 10, "lookup_ns": 10_000_000,
                 "lookup_routes": 2, "lookup_result": 0, "request_id": 1,
                 "token_position": 0, "phase": "prefill", "route_rank": 0,
                 "route_weight": 0.7},
                {"event": "request", "layer": 0, "expert": 2,
                 "lease_generation": 7, "lookup_id": 10, "lookup_ns": 10_000_000,
                 "lookup_routes": 2, "lookup_result": 0, "request_id": 1,
                 "token_position": 1, "phase": "prefill", "route_rank": 1,
                 "route_weight": 0.2},
                {"event": "request", "layer": 0, "expert": 2,
                 "lease_generation": 7, "lookup_id": 11, "lookup_ns": 1_000_000,
                 "lookup_routes": 1, "lookup_result": 0, "request_id": 1,
                 "token_position": 2, "phase": "decode", "route_rank": 0,
                 "route_weight": 0.8},
                {"event": "request", "layer": 1, "expert": 1,
                 "lease_generation": 9, "lookup_id": 12, "lookup_ns": 20_000_000,
                 "lookup_routes": 1, "lookup_result": 0, "request_id": 2,
                 "token_position": 0, "phase": "decode", "route_rank": 0,
                 "route_weight": 0.9},
            ]
        )

    def execution(self) -> Path:
        return self.write_trace(
            [
                {"schema": "colibri.v4.expert_execute_trace.v1", "build": "abc",
                 "source": "expert_execute", "events": 4, "dropped": 0,
                 "total_execute_ns": 6_000_000,
                 "owner_wait_measured_events": 4,
                 "total_owner_wait_ns": 9_000_000},
                {"event": "execute", "layer": 0, "expert": 2, "generation": 7,
                 "execute_ns": 1_000_000, "owner_wait_measured": True,
                 "owner_wait_ns": 5_000_000, "result": 0},
                {"event": "execute", "layer": 0, "expert": 2, "generation": 7,
                 "execute_ns": 1_000_000, "owner_wait_measured": True,
                 "owner_wait_ns": 0, "result": 0},
                {"event": "execute", "layer": 0, "expert": 2, "generation": 7,
                 "execute_ns": 1_000_000, "owner_wait_measured": True,
                 "owner_wait_ns": 1_000_000, "result": 0},
                {"event": "execute", "layer": 1, "expert": 1, "generation": 9,
                 "execute_ns": 3_000_000, "owner_wait_measured": True,
                 "owner_wait_ns": 3_000_000, "result": 0},
            ]
        )

    def test_cold_and_resident_groups_are_separated(self) -> None:
        result = MODULE.summarize(
            self.physical(), self.logical(), self.execution(), stall_threshold_ms=4.0
        )
        self.assertEqual(result["lookup_groups"], 3)
        self.assertEqual(result["logical_routes"], 4)
        self.assertEqual(result["cold_lookup_groups"], 2)
        self.assertEqual(result["resident_lookup_groups"], 1)
        self.assertEqual(result["cold_worker_lookup_ns"], 30_000_000)
        self.assertEqual(result["cold_owner_wait_ns"], 8_000_000)
        self.assertEqual(result["resident_owner_wait_ns"], 1_000_000)
        self.assertAlmostEqual(result["cold_lookup_exposure_ratio"], 8 / 30)
        self.assertAlmostEqual(result["cold_lookup_hidden_fraction_estimate"], 22 / 30)
        self.assertEqual(result["stalled_cold_groups"], 1)
        self.assertAlmostEqual(result["stalled_cold_fraction"], 0.5)

    def test_phase_and_hotspot_rollups(self) -> None:
        result = MODULE.summarize(self.physical(), self.logical(), self.execution())
        self.assertEqual(result["phase_totals"]["prefill"]["routes"], 2)
        self.assertEqual(result["phase_totals"]["prefill"]["owner_wait_ns"], 5_000_000)
        self.assertEqual(result["phase_totals"]["decode"]["routes"], 2)
        self.assertEqual(result["phase_totals"]["decode"]["owner_wait_ns"], 4_000_000)
        self.assertEqual(result["top_experts"][0]["layer"], 0)
        self.assertEqual(result["top_experts"][0]["expert"], 2)
        self.assertEqual(result["top_experts"][0]["owner_wait_ns"], 6_000_000)
        self.assertEqual(result["top_layers"][0]["layer"], 0)

    def test_unmeasured_owner_wait_is_rejected(self) -> None:
        rows = [json.loads(line) for line in self.execution().read_text().splitlines()]
        rows[1]["owner_wait_measured"] = False
        rows[1]["owner_wait_ns"] = 0
        rows[0]["owner_wait_measured_events"] = 3
        rows[0]["total_owner_wait_ns"] = 4_000_000
        execution = self.write_trace(rows)
        with self.assertRaisesRegex(ValueError, "rerun with V4_PROFILE=1"):
            MODULE.summarize(self.physical(), self.logical(), execution)

    def test_fanout_mismatch_is_rejected(self) -> None:
        rows = [json.loads(line) for line in self.logical().read_text().splitlines()]
        rows[1]["lookup_routes"] = 1
        logical = self.write_trace(rows)
        with self.assertRaisesRegex(ValueError, "fanout"):
            MODULE.summarize(self.physical(), logical, self.execution())

    def test_execution_leftovers_are_rejected(self) -> None:
        rows = [json.loads(line) for line in self.execution().read_text().splitlines()]
        rows.append({"event": "execute", "layer": 2, "expert": 3, "generation": 4,
                     "execute_ns": 1, "owner_wait_measured": True,
                     "owner_wait_ns": 0, "result": 0})
        rows[0]["events"] = 5
        rows[0]["total_execute_ns"] = 6_000_001
        rows[0]["owner_wait_measured_events"] = 5
        execution = self.write_trace(rows)
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            MODULE.summarize(self.physical(), self.logical(), execution)


if __name__ == "__main__":
    unittest.main()

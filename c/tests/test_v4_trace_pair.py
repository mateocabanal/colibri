#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent.parent / "tools"
SPEC = importlib.util.spec_from_file_location(
    "validate_v4_trace_pair", TOOLS / "validate_v4_trace_pair.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TracePairTests(unittest.TestCase):
    def write_trace(self, rows: list[dict[str, object]]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".jsonl", delete=False
        )
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def physical(self) -> Path:
        return self.write_trace(
            [
                {
                    "schema": "colibri.v4.expert_trace.v1",
                    "build": "abc123",
                    "record_bytes": 100,
                    "events": 8,
                    "dropped": 0,
                },
                {"seq": 1, "event": "request", "layer": 0, "expert": 2,
                 "tier": None, "generation": 0, "bytes": 100},
                {"seq": 2, "event": "load_begin", "layer": 0, "expert": 2,
                 "tier": "transient", "generation": 7, "bytes": 100},
                {"seq": 3, "event": "load_complete", "layer": 0, "expert": 2,
                 "tier": "transient", "generation": 7, "bytes": 100},
                {"seq": 4, "event": "release", "layer": 0, "expert": 2,
                 "tier": "transient", "generation": 7, "bytes": 0},
                {"seq": 5, "event": "request", "layer": 0, "expert": 4,
                 "tier": None, "generation": 0, "bytes": 100},
                {"seq": 6, "event": "load_begin", "layer": 0, "expert": 4,
                 "tier": "transient", "generation": 8, "bytes": 100},
                {"seq": 7, "event": "load_complete", "layer": 0, "expert": 4,
                 "tier": "transient", "generation": 8, "bytes": 100},
                {"seq": 8, "event": "release", "layer": 0, "expert": 4,
                 "tier": "transient", "generation": 8, "bytes": 0},
                {"event": "layer_summary", "layer": 0, "requests": 2,
                 "hot_expert": 2, "hot_requests": 1},
            ]
        )

    def logical(self) -> Path:
        return self.write_trace(
            [
                {
                    "schema": "colibri.v4.expert_trace.v2",
                    "build": "abc123",
                    "events": 3,
                    "dropped": 0,
                    "requests_started": 1,
                    "physical_lookups": 2,
                    "correlation_misses": 0,
                    "uncorrelated_routes": 0,
                },
                {"seq": 1, "event": "request", "layer": 0, "expert": 2,
                 "request_id": 11, "token_position": 0, "phase": "prefill",
                 "route_rank": 0, "route_weight": 0.7, "lookup_id": 1,
                 "lookup_ns": 10, "lookup_routes": 2, "lookup_result": 0,
                 "lease_generation": 7},
                {"seq": 2, "event": "request", "layer": 0, "expert": 2,
                 "request_id": 11, "token_position": 1, "phase": "prefill",
                 "route_rank": 1, "route_weight": 0.2, "lookup_id": 1,
                 "lookup_ns": 10, "lookup_routes": 2, "lookup_result": 0,
                 "lease_generation": 7},
                {"seq": 3, "event": "request", "layer": 0, "expert": 4,
                 "request_id": 11, "token_position": 2, "phase": "decode",
                 "route_rank": 0, "route_weight": 0.8, "lookup_id": 2,
                 "lookup_ns": 12, "lookup_routes": 1, "lookup_result": 0,
                 "lease_generation": 8},
            ]
        )

    def test_valid_pair(self) -> None:
        result = MODULE.validate_pair(self.physical(), self.logical())
        self.assertEqual(result["build"], "abc123")
        self.assertEqual(result["logical_routes"], 3)
        self.assertEqual(result["physical_lookup_groups"], 2)
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["request_ids"], [11])
        self.assertEqual(result["prefill_routes"], 2)
        self.assertEqual(result["decode_routes"], 1)
        self.assertEqual(result["unique_lease_identities"], 2)
        self.assertEqual(result["physical_event_counts"]["load_complete"], 2)

    def test_missing_physical_identity_is_rejected(self) -> None:
        logical_rows = [json.loads(line) for line in self.logical().read_text().splitlines()]
        logical_rows[-1]["lease_generation"] = 99
        logical = self.write_trace(logical_rows)
        with self.assertRaisesRegex(ValueError, "do not join"):
            MODULE.validate_pair(self.physical(), logical)

    def test_zero_exact_coli_generation_is_rejected(self) -> None:
        logical_rows = [json.loads(line) for line in self.logical().read_text().splitlines()]
        logical_rows[1]["lease_generation"] = 0
        logical = self.write_trace(logical_rows)
        with self.assertRaisesRegex(ValueError, "zero lease_generation"):
            MODULE.validate_pair(self.physical(), logical)

    def test_dropped_or_uncorrelated_trace_is_rejected(self) -> None:
        logical_rows = [json.loads(line) for line in self.logical().read_text().splitlines()]
        logical_rows[0]["uncorrelated_routes"] = 1
        logical = self.write_trace(logical_rows)
        with self.assertRaisesRegex(ValueError, "uncorrelated_routes=1"):
            MODULE.validate_pair(self.physical(), logical)

    def test_build_mismatch_is_rejected(self) -> None:
        logical_rows = [json.loads(line) for line in self.logical().read_text().splitlines()]
        logical_rows[0]["build"] = "different"
        logical = self.write_trace(logical_rows)
        with self.assertRaisesRegex(ValueError, "build mismatch"):
            MODULE.validate_pair(self.physical(), logical)

    def test_bad_lookup_fanout_is_rejected(self) -> None:
        logical_rows = [json.loads(line) for line in self.logical().read_text().splitlines()]
        logical_rows[1]["lookup_routes"] = 1
        logical = self.write_trace(logical_rows)
        with self.assertRaisesRegex(ValueError, "fanout says"):
            MODULE.validate_pair(self.physical(), logical)


if __name__ == "__main__":
    unittest.main()

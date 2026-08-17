#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent.parent / "tools"
SPEC = importlib.util.spec_from_file_location(
    "analyze_v4_request_overlap", TOOLS / "analyze_v4_request_overlap.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RequestOverlapTests(unittest.TestCase):
    def write_trace(self, rows: list[dict[str, object]]) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".jsonl", delete=False
        )
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def test_adjacent_request_overlap_and_layers(self) -> None:
        path = self.write_trace(
            [
                {"schema": "colibri.v4.expert_trace.v2", "events": 8},
                {"event": "request", "request_id": 10, "layer": 0, "expert": 1},
                {"event": "request", "request_id": 10, "layer": 0, "expert": 2},
                {"event": "request", "request_id": 11, "layer": 0, "expert": 2},
                {"event": "request", "request_id": 10, "layer": 1, "expert": 3},
                {"event": "request", "request_id": 11, "layer": 0, "expert": 4},
                {"event": "request", "request_id": 11, "layer": 1, "expert": 3},
                {"event": "request", "request_id": 12, "layer": 0, "expert": 4},
                {"event": "request", "request_id": 12, "layer": 1, "expert": 5},
            ]
        )
        result = MODULE.analyze_request_overlap(path)
        self.assertEqual(result["request_ids"], [10, 11, 12])
        self.assertEqual(result["requests"], 3)
        self.assertEqual(result["adjacent_pairs"], 2)
        self.assertAlmostEqual(result["mean_shared"], 1.5)
        self.assertAlmostEqual(result["mean_jaccard"], 0.375)
        self.assertAlmostEqual(result["aggregate_jaccard"], 0.375)
        self.assertAlmostEqual(result["mean_left_retained"], 0.5)
        self.assertAlmostEqual(result["mean_right_reused"], 7.0 / 12.0)

        first, second = result["pairs"]
        self.assertEqual((first["left_request"], first["right_request"]), (10, 11))
        self.assertEqual((first["shared"], first["union"]), (2, 4))
        self.assertAlmostEqual(first["jaccard"], 0.5)
        self.assertEqual((second["left_request"], second["right_request"]), (11, 12))
        self.assertEqual((second["shared"], second["union"]), (1, 4))
        self.assertAlmostEqual(second["jaccard"], 0.25)

        layers = {row["layer"]: row for row in result["layers"]}
        self.assertAlmostEqual(layers[0]["aggregate_jaccard"], 0.4)
        self.assertAlmostEqual(layers[1]["aggregate_jaccard"], 1.0 / 3.0)

    def test_duplicate_routes_do_not_inflate_working_set(self) -> None:
        path = self.write_trace(
            [
                {"schema": "colibri.v4.expert_trace.v2"},
                {"event": "request", "request_id": 1, "layer": 2, "expert": 7},
                {"event": "request", "request_id": 1, "layer": 2, "expert": 7},
                {"event": "request", "request_id": 2, "layer": 2, "expert": 7},
            ]
        )
        result = MODULE.analyze_request_overlap(path)
        pair = result["pairs"][0]
        self.assertEqual(pair["left"], 1)
        self.assertEqual(pair["right"], 1)
        self.assertEqual(pair["shared"], 1)
        self.assertEqual(pair["union"], 1)
        self.assertAlmostEqual(pair["jaccard"], 1.0)

    def test_route_selected_alias_is_supported(self) -> None:
        path = self.write_trace(
            [
                {"schema": "colibri.v4.expert_trace.v2"},
                {"event": "route_selected", "request_id": 3, "layer": 0, "expert": 1},
                {"event": "route_selected", "request_id": 4, "layer": 0, "expert": 2},
            ]
        )
        result = MODULE.analyze_request_overlap(path)
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["pairs"][0]["shared"], 0)

    def test_v1_trace_is_rejected(self) -> None:
        path = self.write_trace(
            [{"schema": "colibri.v4.expert_trace.v1", "record_bytes": 123}]
        )
        with self.assertRaisesRegex(ValueError, "requires colibri.v4.expert_trace.v2"):
            MODULE.analyze_request_overlap(path)


if __name__ == "__main__":
    unittest.main()

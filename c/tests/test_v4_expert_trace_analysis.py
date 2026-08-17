import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_v4_expert_trace import (
    analyze,
    capacity_curve,
    global_frequency_oracle_curve,
    per_layer_frequency_oracle_curve,
    per_layer_lru_curve,
    summary_dict,
)


class V4ExpertTraceAnalysisTest(unittest.TestCase):
    def write_trace(self, rows):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "trace.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "schema": "colibri.v4.expert_trace.v1",
                        "build": "test",
                        "record_bytes": 1024,
                        "events": len(rows),
                        "dropped": 2,
                    }
                )
                + "\n"
            )
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return temp, path

    def test_exact_reuse_distance_and_lru_curve(self):
        # Logical request stream: A B A C A B.
        rows = [
            {"event": "request", "layer": 0, "expert": 1},
            {"event": "request", "layer": 0, "expert": 2},
            {"event": "request", "layer": 0, "expert": 1},
            {"event": "request", "layer": 1, "expert": 1},
            {"event": "request", "layer": 0, "expert": 1},
            {"event": "request", "layer": 0, "expert": 2},
            {"event": "hit", "layer": 0, "expert": 1, "tier": "persistent"},
            {"event": "hit", "layer": 0, "expert": 2, "tier": "transient"},
            {"event": "load_complete", "layer": 1, "expert": 1, "bytes": 1024},
            {"event": "inflight_join", "layer": 0, "expert": 1},
            {"event": "evict", "layer": 0, "expert": 9},
        ]
        temp, path = self.write_trace(rows)
        self.addCleanup(temp.cleanup)

        result = analyze(path)
        self.assertEqual(result.cold_requests, 3)
        self.assertEqual(result.reuse_distances, [1, 1, 2])
        self.assertEqual(result.dropped, 2)

        curve = capacity_curve(result, [1, 2, 3])
        self.assertEqual([row["hits"] for row in curve], [0, 2, 3])
        self.assertEqual(curve[1]["bytes_avoided"], 2 * 1024)

        summary = summary_dict(result, [2], 2)
        self.assertEqual(summary["requests"], 6)
        self.assertEqual(summary["unique_experts"], 3)
        self.assertEqual(summary["physical_loads"], 1)
        self.assertEqual(summary["inflight_joins"], 1)
        self.assertEqual(summary["evictions"], 1)
        self.assertEqual(summary["persistent_hits"], 1)
        self.assertEqual(summary["transient_hits"], 1)
        self.assertEqual(summary["layers"]["0"]["top_experts"][0]["expert"], 1)
        self.assertEqual(summary["layers"]["0"]["top_experts"][0]["requests"], 3)

    def test_per_layer_persistent_curves(self):
        # Layer 0: A B A A B => one-slot LRU hits only the adjacent A, while a
        # perfect one-hot choice of A avoids two loads after A's first request.
        # Layer 1: C C => both policies avoid C's second load.
        rows = [
            {"event": "request", "layer": 0, "expert": 1},
            {"event": "request", "layer": 0, "expert": 2},
            {"event": "request", "layer": 0, "expert": 1},
            {"event": "request", "layer": 0, "expert": 1},
            {"event": "request", "layer": 0, "expert": 2},
            {"event": "request", "layer": 1, "expert": 7},
            {"event": "request", "layer": 1, "expert": 7},
        ]
        temp, path = self.write_trace(rows)
        self.addCleanup(temp.cleanup)
        result = analyze(path)

        lru = per_layer_lru_curve(result, [1, 2])
        self.assertEqual(lru[0]["layers"], 2)
        self.assertEqual(lru[0]["resident_slots"], 2)
        self.assertEqual(lru[0]["hits"], 2)
        self.assertEqual(lru[0]["resident_bytes"], 2 * 1024)
        self.assertEqual(lru[0]["bytes_avoided"], 2 * 1024)
        self.assertEqual(lru[0]["trace_value_per_resident_byte"], 1.0)
        self.assertEqual(lru[1]["hits"], 4)

        oracle = per_layer_frequency_oracle_curve(result, [1, 2])
        self.assertEqual(oracle[0]["hits"], 3)
        self.assertEqual(oracle[0]["bytes_avoided"], 3 * 1024)
        self.assertEqual(oracle[0]["trace_value_per_resident_byte"], 1.5)
        self.assertEqual(oracle[1]["hits"], 4)

        global_oracle = global_frequency_oracle_curve(result, [2, 4])
        self.assertEqual(global_oracle[0]["resident_slots"], 2)
        self.assertEqual(global_oracle[0]["hits"], 3)
        self.assertEqual(global_oracle[0]["trace_value_per_resident_byte"], 1.5)
        self.assertEqual(global_oracle[1]["resident_slots"], 3)
        self.assertEqual(global_oracle[1]["hits"], 4)
        self.assertAlmostEqual(
            global_oracle[1]["trace_value_per_resident_byte"], 4.0 / 3.0
        )

        summary = summary_dict(result, [2], 2, [1, 2])
        self.assertEqual(
            summary["persistent_per_layer_lru_curve"][0]["hits"], 2
        )
        self.assertEqual(
            summary["persistent_per_layer_frequency_oracle_curve"][0]["hits"], 3
        )
        self.assertEqual(
            summary["persistent_global_frequency_oracle_curve"][0]["capacity"], 2
        )
        self.assertEqual(
            summary["persistent_global_frequency_oracle_curve"][0]["hits"], 3
        )

    def test_global_hot_oracle_avoids_diffuse_layer_waste(self):
        # Equal per-layer allocation wastes one of two slots on layer 0, whose
        # requests are all cold. A global two-slot budget can spend both slots
        # on layer 1's two repeatedly used experts instead.
        rows = [
            {"event": "request", "layer": 0, "expert": 1},
            {"event": "request", "layer": 0, "expert": 2},
            {"event": "request", "layer": 0, "expert": 3},
            {"event": "request", "layer": 1, "expert": 7},
            {"event": "request", "layer": 1, "expert": 8},
            {"event": "request", "layer": 1, "expert": 7},
            {"event": "request", "layer": 1, "expert": 8},
            {"event": "request", "layer": 1, "expert": 7},
            {"event": "request", "layer": 1, "expert": 8},
        ]
        temp, path = self.write_trace(rows)
        self.addCleanup(temp.cleanup)
        result = analyze(path)

        per_layer = per_layer_frequency_oracle_curve(result, [1])[0]
        global_hot = global_frequency_oracle_curve(result, [2])[0]
        self.assertEqual(per_layer["resident_slots"], global_hot["resident_slots"])
        self.assertEqual(per_layer["hits"], 2)
        self.assertEqual(global_hot["hits"], 4)
        self.assertGreater(
            global_hot["trace_value_per_resident_byte"],
            per_layer["trace_value_per_resident_byte"],
        )

    def test_invalid_request_event_is_rejected(self):
        temp, path = self.write_trace([{"event": "request", "layer": "x"}])
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "layer/expert"):
            analyze(path)


if __name__ == "__main__":
    unittest.main()
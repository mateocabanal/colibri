import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_v4_expert_trace import analyze, capacity_curve, summary_dict


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

    def test_invalid_request_event_is_rejected(self):
        temp, path = self.write_trace([{"event": "request", "layer": "x"}])
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "layer/expert"):
            analyze(path)


if __name__ == "__main__":
    unittest.main()

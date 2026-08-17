import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_v4_residency_value import (
    DensePoint,
    build_summary,
    dense_marginal_frontier,
    dense_point_value,
    parse_dense_point,
)


class V4ResidencyValueAnalysisTest(unittest.TestCase):
    def write_trace(self, requests, record_bytes=1024):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "trace.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "schema": "colibri.v4.expert_trace.v1",
                        "build": "test",
                        "record_bytes": record_bytes,
                        "events": len(requests),
                        "dropped": 0,
                    }
                )
                + "\n"
            )
            for layer, expert in requests:
                handle.write(
                    json.dumps(
                        {"event": "request", "layer": layer, "expert": expert}
                    )
                    + "\n"
                )
        return temp, path

    def test_parse_dense_point(self):
        point = parse_dense_point("3.0,12.0,8.0,4000")
        self.assertEqual(point, DensePoint(3.0, 12.0, 8.0, 4000.0))
        with self.assertRaisesRegex(ValueError, "must be resident_gib"):
            parse_dense_point("1,2,3")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            parse_dense_point("0,2,3,4")

    def test_dense_average_value_uses_exposed_read_cost(self):
        row = dense_point_value(DensePoint(2.0, 8.0, 4.0, 2000.0))
        self.assertEqual(row["bytes_value"], 4.0)
        self.assertEqual(row["exposed_ms_per_physical_read_gib"], 500.0)
        self.assertEqual(row["estimated_exposed_ms_avoided"], 4000.0)
        self.assertEqual(
            row["estimated_exposed_ms_avoided_per_resident_gib"], 2000.0
        )

    def test_dense_marginal_frontier_uses_same_workload_deltas(self):
        rows = dense_marginal_frontier(
            [
                DensePoint(2.0, 10.0, 20.0, 10000.0),
                DensePoint(3.0, 16.0, 14.0, 7000.0),
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["delta_resident_gib"], 1.0)
        self.assertEqual(rows[0]["delta_avoided_gib"], 6.0)
        self.assertEqual(rows[0]["marginal_bytes_value"], 6.0)
        self.assertEqual(rows[0]["measured_exposed_ms_saved"], 3000.0)
        self.assertEqual(
            rows[0]["measured_exposed_ms_saved_per_resident_gib"], 3000.0
        )

    def test_summary_compares_dense_with_global_hot_oracle(self):
        # A repeats four times, B three times, C once. A perfect global two-slot
        # tier credits (4-1) + (3-1) = 5 avoided expert reads.
        requests = [
            (0, 1),
            (0, 2),
            (1, 3),
            (0, 1),
            (0, 2),
            (0, 1),
            (0, 2),
            (0, 1),
        ]
        temp, path = self.write_trace(requests, record_bytes=1024**3)
        self.addCleanup(temp.cleanup)

        summary = build_summary(
            path,
            [DensePoint(2.0, 8.0, 4.0, 2000.0)],
            [1, 2],
            expert_read_gib=10.0,
            expert_wait_ms=1000.0,
        )
        self.assertEqual(
            summary["metric"],
            "estimated_exposed_io_ms_avoided_per_resident_gib",
        )
        expert = summary["expert_global_perfect_hot_upper_bound"]
        self.assertEqual(expert[0]["hits"], 3)
        self.assertEqual(expert[1]["hits"], 5)
        self.assertEqual(expert[1]["resident_gib"], 2.0)
        self.assertEqual(expert[1]["avoided_gib"], 5.0)
        self.assertEqual(
            expert[1]["estimated_exposed_ms_avoided_per_resident_gib"], 250.0
        )


if __name__ == "__main__":
    unittest.main()

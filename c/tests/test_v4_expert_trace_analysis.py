import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_v4_expert_trace import (
    analyze,
    capacity_curve,
    explicit_route_groups,
    global_frequency_oracle_curve,
    infer_route_groups,
    per_layer_frequency_oracle_curve,
    per_layer_lru_curve,
    routing_locality_summary,
    summary_dict,
)


class V4ExpertTraceAnalysisTest(unittest.TestCase):
    def write_trace(self, rows, schema="colibri.v4.expert_trace.v1"):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "trace.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "schema": schema,
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
        self.assertEqual(result.schema, "colibri.v4.expert_trace.v1")
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
        self.assertEqual(summary["persistent_per_layer_lru_curve"][0]["hits"], 2)
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

    def test_inferred_token_groups_corouting_and_adjacent_overlap(self):
        rows = [
            {"event": "request", "layer": 0, "expert": 1},
            {"event": "request", "layer": 0, "expert": 2},
            {"event": "request", "layer": 1, "expert": 3},
            {"event": "request", "layer": 1, "expert": 4},
            {"event": "request", "layer": 0, "expert": 1},
            {"event": "request", "layer": 0, "expert": 5},
            {"event": "request", "layer": 1, "expert": 3},
            {"event": "request", "layer": 1, "expert": 6},
        ]
        temp, path = self.write_trace(rows)
        self.addCleanup(temp.cleanup)
        result = analyze(path)

        groups = infer_route_groups(result.requests)
        self.assertEqual(len(groups), 4)
        self.assertEqual([group.token for group in groups], [0, 0, 1, 1])
        self.assertEqual(groups[0].experts, (1, 2))
        self.assertEqual(groups[2].experts, (1, 5))

        routing = routing_locality_summary(result, 10, prompt_tokens=1)
        self.assertEqual(routing["token_ids"], "inferred-from-layer-wraps")
        self.assertEqual(routing["token_count"], 2)
        self.assertEqual(routing["route_groups"], 4)
        self.assertEqual(routing["unique_logical_experts_per_token"]["mean"], 4.0)
        self.assertEqual(routing["adjacent_token_overlap"]["samples"], 2)
        self.assertEqual(routing["adjacent_token_overlap"]["mean_shared"], 1.0)
        self.assertAlmostEqual(
            routing["adjacent_token_overlap"]["mean_jaccard"], 1.0 / 3.0
        )
        self.assertEqual(routing["phase"]["source"], "inferred")
        self.assertEqual(routing["phase"]["prefill_requests"], 4)
        self.assertEqual(routing["phase"]["decode_requests"], 4)
        self.assertIn(
            {"layer": 0, "experts": [1, 2], "co_routes": 1},
            routing["top_co_routing_pairs"],
        )

    def test_v2_explicit_context_wins_over_stream_order(self):
        # Deliberately interleave request IDs. Explicit identity must group and
        # compare tokens within each request rather than infer boundaries from
        # global layer ordering.
        rows = [
            {
                "event": "request", "layer": 0, "expert": 1,
                "request_id": 10, "token_position": 7, "phase": "prefill",
                "route_rank": 0, "route_weight": 0.7,
            },
            {
                "event": "request", "layer": 0, "expert": 2,
                "request_id": 10, "token_position": 7, "phase": "prefill",
                "route_rank": 1, "route_weight": 0.3,
            },
            {
                "event": "request", "layer": 0, "expert": 9,
                "request_id": 22, "token_position": 3, "phase": "decode",
                "route_rank": 0, "route_weight": 1.0,
            },
            {
                "event": "request", "layer": 0, "expert": 1,
                "request_id": 10, "token_position": 8, "phase": "decode",
                "route_rank": 0, "route_weight": 0.6,
            },
            {
                "event": "request", "layer": 0, "expert": 3,
                "request_id": 10, "token_position": 8, "phase": "decode",
                "route_rank": 1, "route_weight": 0.4,
            },
        ]
        temp, path = self.write_trace(rows, "colibri.v4.expert_trace.v2")
        self.addCleanup(temp.cleanup)
        result = analyze(path)
        self.assertEqual(result.schema, "colibri.v4.expert_trace.v2")
        groups = explicit_route_groups(result)
        self.assertIsNotNone(groups)
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0].request_id, 10)
        self.assertEqual(groups[0].token, 7)
        self.assertEqual(groups[0].experts, (1, 2))

        routing = routing_locality_summary(result, 5)
        self.assertEqual(routing["token_ids"], "explicit")
        self.assertEqual(routing["request_count"], 2)
        self.assertEqual(routing["token_count"], 3)
        self.assertEqual(routing["explicit_context_requests"], 5)
        self.assertEqual(routing["route_rank_coverage"], 5)
        self.assertEqual(routing["route_weight_coverage"], 5)
        self.assertEqual(routing["adjacent_token_overlap"]["samples"], 1)
        self.assertEqual(routing["phase"]["source"], "explicit")
        self.assertEqual(routing["phase"]["prefill_requests"], 2)
        self.assertEqual(routing["phase"]["decode_requests"], 3)
        self.assertEqual(routing["phase"]["prefill_tokens"], 1)
        self.assertEqual(routing["phase"]["decode_tokens"], 2)

    def test_invalid_request_event_is_rejected(self):
        temp, path = self.write_trace([{"event": "request", "layer": "x"}])
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "layer/expert"):
            analyze(path)


if __name__ == "__main__":
    unittest.main()

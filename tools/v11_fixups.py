#!/usr/bin/env python3
from pathlib import Path

# The end-to-end source fixture should exercise the actual Apple v1 routed-expert
# contract: packed MXFP4 nibbles + one UE8M0 scale byte per 32 columns per row.
p = Path("colic/src/pipeline.rs")
t = p.read_text()
old = '''        for expert in 0..2 {
            for (role, shape) in [("w1", vec![3, 2]), ("w2", vec![2, 3]), ("w3", vec![3, 2])] {
                add(
                    format!("layers.0.ffn.experts.{expert}.{role}.weight"),
                    "F8_E4M3FN",
                    shape,
                );
                add(
                    format!("layers.0.ffn.experts.{expert}.{role}.scale"),
                    "F8_E8M0",
                    vec![1, 1],
                );
            }
        }'''
new = '''        for expert in 0..2 {
            for (role, rows, columns) in [("w1", 3_u64, 2_u64), ("w2", 2, 3), ("w3", 3, 2)] {
                add(
                    format!("layers.0.ffn.experts.{expert}.{role}.weight"),
                    "I8",
                    vec![rows, columns.div_ceil(2)],
                );
                add(
                    format!("layers.0.ffn.experts.{expert}.{role}.scale"),
                    "F8_E8M0",
                    vec![rows, columns.div_ceil(32)],
                );
            }
        }'''
if old not in t:
    raise SystemExit("synthetic expert fixture anchor missing")
t = t.replace(old, new, 1)
t = t.replace(
    '                "U8" | "F8_E4M3FN" | "F8_E8M0" => 1,',
    '                "U8" | "I8" | "F8_E4M3FN" | "F8_E8M0" => 1,',
    1,
)
p.write_text(t)

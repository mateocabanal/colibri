#!/usr/bin/env bash
set -euo pipefail

work=${1:-"${RUNNER_TEMP:-/tmp}/colic-e2e"}
rm -rf "$work"
mkdir -p "$work"

python3 colic/tests/fixtures/make_v4_tiny.py --output "$work/v4-source"

cargo run --locked --manifest-path colic/Cargo.toml -- compile \
  "$work/v4-source" \
  --target macos-arm64-metal-apple8-v1 \
  --quant exact --codec none --shard-size-mib 1 --verify \
  -o "$work/model-a.coli"

cargo run --locked --manifest-path colic/Cargo.toml -- compile \
  "$work/v4-source" \
  --target macos-arm64-metal-apple8-v1 \
  --quant exact --codec none --shard-size-mib 1 --verify \
  -o "$work/model-b.coli"

test -f "$work/model-a.coli/data-00001.coli"
test -f "$work/model-b.coli/data-00001.coli"
diff -r "$work/model-a.coli" "$work/model-b.coli"

cc -O2 -Wall -Wextra -I c \
  c/tests/test_colic_e2e.c \
  c/tests/mxfp4_ref.c \
  c/coli_exec_format.c \
  c/coli_target.c \
  c/coli_target_profiles.c \
  -lm -pthread \
  -o "$work/test_colic_e2e"

rm -rf "$work/v4-source"
test ! -e "$work/v4-source"
"$work/test_colic_e2e" "$work/model-a.coli"
"$work/test_colic_e2e" --corrupt-and-expect-fail "$work/model-b.coli"

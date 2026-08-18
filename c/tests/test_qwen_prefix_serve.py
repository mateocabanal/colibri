#!/usr/bin/env python3
"""End-to-end prompt-prefix correctness gate for the hybrid Qwen serve path.

Qwen3.5/3.6/3.7 mixes full attention with Gated DeltaNet. A cache hit is only
correct if it restores BOTH attention K/V and the GDN recurrence/conv history.
The unit test for qwen_prefix_cache.h checks the copy geometry; this test checks
the property users actually care about: a warm reused prefix must emit exactly
the same bytes as a cold full prefill of the same prompt.

The generated tiny fixture already contains a useful pair:
  short = [5,7,9,11,13,17,19,23]
  mixed = short + [29,31,37,41,43,47]
Its tokenizer maps every id to an added token named <tNNN>, so serve mode can
exercise the real tokenizer and protocol without a production checkpoint.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

READY = b"\x01\x01READY\x01\x01"


def token_prompt(ids: list[int]) -> bytes:
    return "".join(f"<t{token:03d}>" for token in ids).encode("utf-8")


class Serve:
    """One persistent qwen_moe SERVE=1 process."""

    def __init__(self, binary: Path, fixture: Path, cache_mb: str = "8") -> None:
        env = dict(
            os.environ,
            SERVE="1",
            SNAP=str(fixture),
            CTX="128",
            QWEN_PREFIX_CACHE_MB=cache_mb,
            QWEN_PREFIX_CACHE_MIN_TOKENS="1",
            QWEN_PREFIX_LOG="1",
            OMP_NUM_THREADS="1",
            COLI_NO_OMP_TUNE="1",
        )
        # argv[1] is the per-layer resident expert cap in serve mode. Two is the
        # fixture's TOPK and avoids making this cache test about expert residency.
        self.process = subprocess.Popen(
            [str(binary), "2"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        self.counter = 0
        self._stderr: list[str] = []
        self._pump = threading.Thread(target=self._drain_stderr, daemon=True)
        self._pump.start()

        assert self.process.stdout is not None
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                raise AssertionError(
                    "qwen_moe exited before READY:\n" + "".join(self._stderr)[-4000:]
                )
            if READY in line:
                return
        raise AssertionError("qwen_moe did not report READY")

    def submit(self, prompt: bytes, max_tokens: int = 4) -> bytes:
        self.counter += 1
        request_id = f"r{self.counter}"
        header = (
            f"SUBMIT {request_id} 0 {len(prompt)} {max_tokens} 0.0 1.0\n"
        ).encode("ascii")
        assert self.process.stdin is not None
        self.process.stdin.write(header + prompt + b"\n")
        self.process.stdin.flush()

        pieces: list[bytes] = []
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise AssertionError(
                    f"qwen_moe closed during {request_id}:\n"
                    + "".join(self._stderr)[-4000:]
                )
            fields = line.decode("latin-1").split()
            if not fields:
                continue
            if fields[0] == "ERROR":
                raise AssertionError(f"engine ERROR: {line!r}")
            if fields[0] == "DATA" and len(fields) == 3:
                size = int(fields[2])
                pieces.append(self.process.stdout.read(size))
                self.process.stdout.read(1)  # DATA payload trailing newline
            elif fields[0] == "DONE":
                return b"".join(pieces)
            # ACCEPT/PROF/STAT/HWINFO/TIERS/EMAP are protocol metadata.

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in iter(self.process.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", "replace"))

    def close(self) -> str:
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=10)
        self._pump.join(timeout=10)
        return "".join(self._stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    args = parser.parse_args()

    import json

    reference = json.loads((args.fixture / "ref.json").read_text("utf-8"))
    short_case = reference["cases"]["short"]
    mixed_case = reference["cases"]["mixed"]
    short = list(short_case["prompt_ids"])
    mixed = list(mixed_case["prompt_ids"])
    if mixed[: len(short)] != short or len(mixed) <= len(short):
        raise AssertionError("tiny fixture lost the strict short -> mixed prefix pair")

    short_prompt = token_prompt(short)
    mixed_prompt = token_prompt(mixed)
    expected_short = token_prompt(list(short_case["greedy_new_ids"])[:4])
    expected_mixed = token_prompt(list(mixed_case["greedy_new_ids"])[:4])
    if not expected_short or not expected_mixed:
        raise AssertionError("tiny oracle must contain non-empty generated tokens")

    warm = Serve(args.binary, args.fixture)
    try:
        first = warm.submit(short_prompt)
        # Equal-length entries are deliberately not restorable: step() needs at
        # least one unmatched prompt token to produce the final prefill logits.
        repeated = warm.submit(short_prompt)
        reused = warm.submit(mixed_prompt)
    finally:
        warm_log = warm.close()

    cold = Serve(args.binary, args.fixture)
    try:
        cold_mixed = cold.submit(mixed_prompt)
    finally:
        cold_log = cold.close()

    cold_repeat = Serve(args.binary, args.fixture)
    try:
        cold_short = cold_repeat.submit(short_prompt)
    finally:
        cold_repeat.close()

    if first != expected_short or repeated != expected_short or cold_short != expected_short:
        raise AssertionError(
            "short serve output did not match the Transformers-generated token oracle; "
            f"expected={expected_short!r} first={first!r} repeated={repeated!r} cold={cold_short!r}"
        )
    expected_hit = f"[QWEN-PREFIX] hit matched={len(short)} prompt={len(mixed)}"
    hits = [line for line in warm_log.splitlines() if "[QWEN-PREFIX] hit" in line]
    if hits != [next((line for line in hits if expected_hit in line), "")]:
        raise AssertionError(
            "expected exactly one Qwen prefix hit, on short -> mixed; got:\n"
            + ("\n".join(hits) or "(none)")
            + "\nfull stderr:\n"
            + warm_log[-6000:]
        )
    if expected_hit not in warm_log:
        raise AssertionError(
            f"the warm mixed request did not reuse all {len(short)} short tokens:\n"
            + warm_log[-6000:]
        )
    if "[QWEN-PREFIX] hit" in cold_log:
        raise AssertionError("fresh engine unexpectedly reported a prefix hit")
    if reused != expected_mixed or cold_mixed != expected_mixed:
        raise AssertionError(
            "restoring the hybrid prompt state changed generation or diverged from "
            "the Transformers-generated token oracle:\n"
            f"  expected={expected_mixed!r}\n  warm={reused!r}\n  cold={cold_mixed!r}"
        )

    print(
        f"PASS Qwen hybrid prefix reuse: matched={len(short)} prompt={len(mixed)}, "
        "warm output is byte-identical to cold prefill"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

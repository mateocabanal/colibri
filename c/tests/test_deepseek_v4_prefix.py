#!/usr/bin/env python3
"""KV/prompt prefix reuse on the DeepSeek V4 serve path.

The property under test is not "it is faster" -- it is that reusing or
restoring attention state produces the bytes a cold prefill would have
produced. Same-session cases exercise the existing `kv_prefix` path. The #12
case uses two distinct session slots inside one engine process so only the
process-local multi-prefix cache can provide the second request's state.

Speaking the SUBMIT/DATA/DONE protocol directly rather than through
openai_server.Engine keeps the test on the engine's own contract, including the
trailing reuse field of the DONE frame.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def token_prompt(ids: list[int]) -> str:
    return "".join(f"<t{token:03d}>" for token in ids)


def parse_tokens(text: str) -> list[int]:
    """The tiny fixture's vocabulary is <tNNN>, so a reply decodes back to ids."""
    return [int(match) for match in re.findall(r"<t(\d+)>", text)]


class Serve:
    """One persistent `SERVE=1` engine process with multiple session slots."""

    def __init__(self, binary: Path, model: Path, ctx: str = "128",
                 prefix_cache_mb: int | None = None) -> None:
        env = dict(os.environ, SERVE="1", SNAP=str(model), CTX=ctx,
                   V4_PREFIX_LOG="1")
        if prefix_cache_mb is not None:
            env["V4_PREFIX_CACHE_MB"] = str(prefix_cache_mb)
            # Tiny fixture prompts are intentionally short; production keeps
            # the much more conservative default admission threshold.
            env["V4_PREFIX_CACHE_MIN_TOKENS"] = "1"
        else:
            env.pop("V4_PREFIX_CACHE_MB", None)
            env.pop("V4_PREFIX_CACHE_MIN_TOKENS", None)
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        self.counter = 0

    def submit(self, prompt: str, max_tokens: int,
               slot: int = 0) -> tuple[str, int]:
        """Returns generated text and the DONE frame's reused-prefix count."""
        self.counter += 1
        request_id = f"r{self.counter}"
        payload = prompt.encode("utf-8")
        header = (f"SUBMIT {request_id} {slot} {len(payload)} {max_tokens} "
                  f"0.0 1.0 0\n").encode("ascii")
        assert self.process.stdin is not None
        self.process.stdin.write(header + payload + b"\n")
        self.process.stdin.flush()

        pieces: list[bytes] = []
        reuse = -1
        stream = self.process.stdout
        assert stream is not None
        while True:
            line = stream.readline()
            if not line:
                raise AssertionError(
                    f"engine closed stdout during {request_id}; "
                    f"stderr:\n{self._drain_stderr()}"
                )
            fields = line.decode("utf-8", "replace").split()
            if not fields:
                continue
            if fields[0] == "ERROR":
                raise AssertionError(f"engine ERROR: {line!r}")
            if fields[0] == "DATA" and len(fields) == 3:
                size = int(fields[2])
                pieces.append(stream.read(size))
                stream.read(1)                     # trailing newline
            elif fields[0] == "DONE":
                # DONE <id> STAT <completion> <tok/s> <hit%> <rss> <prompt> <cap> [<reuse>]
                if len(fields) >= 10:
                    reuse = int(fields[9])
                break
        return b"".join(pieces).decode("utf-8", "replace"), reuse

    def _drain_stderr(self) -> str:
        assert self.process.stderr is not None
        try:
            return self.process.stderr.read(8192).decode("utf-8", "replace")
        except Exception:                          # pragma: no cover
            return "<unavailable>"

    def close(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:                          # pragma: no cover
            self.process.kill()


def extended_prompt(first_ids: list[int], first_text: str) -> list[int]:
    """Build a strict extension of exactly the target state left by a turn."""
    reply = parse_tokens(first_text)
    if len(reply) < 2:
        raise AssertionError(f"first turn produced too little: {first_text!r}")
    # The final emitted token has not yet been fed through the target. The state
    # therefore represents prompt + reply[:-1]. This mirrors a continuation at
    # the exact reusable boundary rather than pretending an unseen token exists
    # in KV/recurrent state.
    return first_ids + reply[:-1] + [first_ids[0]]


def check_growing_conversation(binary: Path, model: Path,
                               case: dict[str, object]) -> None:
    """Turn 2 extends turn 1: same-session reuse stays token-exact."""
    first_ids = list(case["prompt_ids"])            # type: ignore[arg-type]
    max_new = 4

    warm = Serve(binary, model)
    try:
        first_text, first_reuse = warm.submit(token_prompt(first_ids), max_new)
        if first_reuse != 0:
            raise AssertionError(
                f"first turn of a fresh session reused {first_reuse} tokens; "
                "there was nothing to reuse"
            )
        second_ids = extended_prompt(first_ids, first_text)
        second_text, second_reuse = warm.submit(token_prompt(second_ids), max_new)
    finally:
        warm.close()

    cold = Serve(binary, model)
    try:
        cold_text, cold_reuse = cold.submit(token_prompt(second_ids), max_new)
    finally:
        cold.close()

    if cold_reuse != 0:
        raise AssertionError(f"cold engine reported reuse={cold_reuse}")
    if second_text != cold_text:
        raise AssertionError(
            "warm continuation diverged from a cold prefill of the same prompt:\n"
            f"  warm: {second_text!r}\n  cold: {cold_text!r}"
        )
    if second_reuse <= 0:
        raise AssertionError(
            "prefix reuse never fired on a prompt that extends the previous "
            f"turn (reuse={second_reuse}); the optimisation is inert"
        )
    if second_reuse < len(first_ids):
        raise AssertionError(
            f"reused only {second_reuse} tokens, expected at least the "
            f"{len(first_ids)} prompt tokens of the previous turn"
        )
    print(f"PASS same-session prefix reuse: {second_reuse} tokens reused, "
          f"output identical to a cold prefill")


def check_cross_session_cache(binary: Path, model: Path,
                              case: dict[str, object]) -> None:
    """A distinct session restores the longest exact prefix from #12 cache."""
    first_ids = list(case["prompt_ids"])            # type: ignore[arg-type]
    max_new = 4

    shared = Serve(binary, model, prefix_cache_mb=64)
    try:
        first_text, first_reuse = shared.submit(
            token_prompt(first_ids), max_new, slot=0)
        if first_reuse != 0:
            raise AssertionError(
                f"first cached turn unexpectedly reported reuse={first_reuse}"
            )
        second_ids = extended_prompt(first_ids, first_text)
        # Slot 1 has never run. A positive reuse count here cannot come from its
        # own kv_prefix; it must be a process-local cross-session cache hit.
        restored_text, restored_reuse = shared.submit(
            token_prompt(second_ids), max_new, slot=1)
    finally:
        shared.close()

    cold = Serve(binary, model)
    try:
        cold_text, cold_reuse = cold.submit(token_prompt(second_ids), max_new)
    finally:
        cold.close()

    if cold_reuse != 0:
        raise AssertionError(f"cold comparison engine reported reuse={cold_reuse}")
    if restored_reuse < len(first_ids):
        raise AssertionError(
            "cross-session cache did not restore the first request state: "
            f"reuse={restored_reuse}, first_prompt={len(first_ids)}"
        )
    if restored_text != cold_text:
        raise AssertionError(
            "cross-session restored state diverged from cold prefill:\n"
            f"  restored: {restored_text!r}\n  cold: {cold_text!r}"
        )
    print(f"PASS cross-session prefix cache: {restored_reuse} tokens restored, "
          "output identical to a cold prefill")


def check_divergent_prompt_resets(binary: Path, model: Path,
                                  case: dict[str, object]) -> None:
    """A prompt that is not an extension must fall back to a full prefill."""
    first_ids = list(case["prompt_ids"])            # type: ignore[arg-type]
    other_ids = [token + 1 for token in first_ids]
    max_new = 4

    warm = Serve(binary, model)
    try:
        warm.submit(token_prompt(first_ids), max_new)
        divergent_text, reuse = warm.submit(token_prompt(other_ids), max_new)
    finally:
        warm.close()

    cold = Serve(binary, model)
    try:
        cold_text, _ = cold.submit(token_prompt(other_ids), max_new)
    finally:
        cold.close()

    if reuse != 0:
        raise AssertionError(
            f"reused {reuse} tokens from a prompt that shares no full prefix"
        )
    if divergent_text != cold_text:
        raise AssertionError(
            "divergent prompt did not reset the state:\n"
            f"  warm: {divergent_text!r}\n  cold: {cold_text!r}"
        )
    print("PASS prefix reset: divergent prompt re-prefills and matches cold")


def check_repeated_prompt(binary: Path, model: Path,
                          case: dict[str, object]) -> None:
    """An identical prompt is not a strict extension: it must re-prefill."""
    ids = list(case["prompt_ids"])                  # type: ignore[arg-type]
    max_new = 4

    warm = Serve(binary, model)
    try:
        first_text, _ = warm.submit(token_prompt(ids), max_new)
        second_text, reuse = warm.submit(token_prompt(ids), max_new)
    finally:
        warm.close()

    if reuse != 0:
        raise AssertionError(f"identical prompt reported reuse={reuse}")
    if first_text != second_text:
        raise AssertionError(
            f"repeated prompt changed answer: {first_text!r} vs {second_text!r}"
        )
    print("PASS prefix repeat: identical prompt re-prefills, answer unchanged")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    arguments = parser.parse_args()

    reference = json.loads((arguments.fixture / "ref.json").read_text("utf-8"))
    case = reference["cases"]["short"]

    check_growing_conversation(arguments.binary, arguments.fixture, case)
    check_cross_session_cache(arguments.binary, arguments.fixture, case)
    check_repeated_prompt(arguments.binary, arguments.fixture, case)
    check_divergent_prompt_resets(arguments.binary, arguments.fixture, case)
    print("PASS DeepSeek V4 prefix reuse/cache: all checks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

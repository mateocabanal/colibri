#!/usr/bin/env python3
"""KV/prompt prefix reuse on the DeepSeek V4 runtime.

Same-session cases speak the existing single-session SERVE protocol directly.
The #12 cross-session case deliberately does not extend that protocol: it builds
and runs a tiny C integration binary that creates two real ColiV4Session
instances sharing one ColiV4Engine, plus a second engine as the cold oracle.
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
    """One persistent `SERVE=1` engine process (the protocol owns one session)."""

    def __init__(self, binary: Path, model: Path, ctx: str = "128") -> None:
        env = dict(os.environ, SERVE="1", SNAP=str(model), CTX=ctx)
        env.pop("V4_PREFIX_CACHE_MB", None)
        env.pop("V4_PREFIX_CACHE_MIN_TOKENS", None)
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        self.counter = 0

    def submit(self, prompt: str, max_tokens: int) -> tuple[str, int]:
        """Returns generated text and the DONE frame's reused-prefix count."""
        self.counter += 1
        request_id = f"r{self.counter}"
        payload = prompt.encode("utf-8")
        # The second field is currently a reserved protocol value and must be 0.
        header = (f"SUBMIT {request_id} 0 {len(payload)} {max_tokens} "
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
    """Two actual sessions share one engine/cache; a second engine is cold."""
    c_dir = binary.resolve().parent
    make = os.environ.get("MAKE", "make")
    build = subprocess.run(
        [make, "-f", "Makefile.deepseek-v4", "deepseek-v4-prefix-cache-test"],
        cwd=c_dir,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    if build.returncode:
        raise AssertionError(
            "cross-session helper build failed:\n"
            f"stdout:\n{build.stdout}\nstderr:\n{build.stderr}"
        )
    helper = c_dir / (
        "test_v4_prefix_cache_sessions.exe"
        if os.name == "nt" else "test_v4_prefix_cache_sessions"
    )
    env = dict(os.environ,
               V4_PREFIX_CACHE_MB="64",
               V4_PREFIX_CACHE_MIN_TOKENS="1",
               V4_PREFIX_LOG="1")
    result = subprocess.run(
        [helper.as_posix(), model.resolve().as_posix(),
         token_prompt(list(case["prompt_ids"]))],  # type: ignore[arg-type]
        cwd=c_dir,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        env=env,
    )
    if result.returncode:
        raise AssertionError(
            "cross-session prefix-cache oracle failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if "PASS cross-session prefix cache:" not in result.stdout:
        raise AssertionError(
            f"cross-session helper did not report success: {result.stdout!r}"
        )
    print(result.stdout.strip())


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

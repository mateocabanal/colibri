#!/usr/bin/env python3
"""Token-exact oracle check for the qwen_moe engine against ref.json.

Usage:
    python tools/test_qwen_moe_real.py --model c/qwen_moe_tiny
    python tools/test_qwen_moe_real.py --model c/qwen_moe_tiny --engine c/qwen_moe

Engine interface exercised (implemented by qwen_moe.c, env-driven so the
harness needs no CLI coupling):

  QWENMOE_MODE=teacher
      QWENMOE_TEACHER="<space separated full ids>"
      -> prints one "PRED <argmax_id>" per input position (logits over the
         full vocab, teacher-forced), then exits 0.
  QWENMOE_MODE=greedy
      QWENMOE_PROMPT_IDS="<space separated prompt ids>"
      QWENMOE_MAX_NEW=<n>
      -> prints one "ID <generated_id>" per generated token, stops at EOS,
         then exits 0 (0 < printed <= n).

Checks against ref.json:
  teacher_forcing_ids == engine PRED lines (exact)
  greedy_new_ids == engine ID lines (exact)

Exit code 0 = token-exact; 1 = mismatch (first divergence printed); 2 = setup
error (missing fixture/engine).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def run_engine(engine: Path, model_dir: Path, env: dict) -> list[str]:
    full_env = os.environ.copy()
    for name in (
        "QWENMOE_MODE", "QWENMOE_TEACHER", "QWENMOE_PROMPT_IDS",
        "QWENMOE_MAX_NEW", "SERVE", "SERVE_BATCH", "SNAP", "CHAT",
        "COLI_TEMP", "NUCLEUS",
    ):
        full_env.pop(name, None)
    full_env.update(env)
    if os.environ.get("HARNESS_DEBUG"):
        print(f"[harness] env: { {k: v for k, v in env.items()} }")
    proc = subprocess.run(
        [str(engine), str(model_dir)],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if os.environ.get("HARNESS_DEBUG"):
        print(f"[harness] rc={proc.returncode} stdout={proc.stdout!r}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"engine exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.splitlines()


@dataclass(frozen=True)
class ServeResult:
    frames: tuple[bytes, ...]
    generated: int
    prompt_tokens: int
    length_limited: int

    @property
    def stable_done(self) -> tuple[int, int, int]:
        """DONE fields that are deterministic (exclude TPS/cache-hit/RSS)."""
        return self.generated, self.prompt_tokens, self.length_limited


class ServeClient:
    """Small dependency-free reader for the engine's framed byte protocol."""

    def __init__(self, engine: Path, model_dir: Path, timeout: float = 600.0):
        env = os.environ.copy()
        for name in (
            "QWENMOE_MODE", "QWENMOE_TEACHER", "QWENMOE_PROMPT_IDS",
            "QWENMOE_MAX_NEW", "CHAT", "COLI_TEMP", "NUCLEUS",
        ):
            env.pop(name, None)
        env.update({
            "SERVE": "1",
            "SERVE_BATCH": "1",
            "SNAP": str(model_dir),
        })
        self.timeout = timeout
        self.events: queue.Queue[tuple] = queue.Queue()
        self.stderr_chunks: list[bytes] = []
        self.proc = subprocess.Popen(
            [str(engine)],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.err_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self.reader.start()
        self.err_reader.start()
        deadline = time.monotonic() + timeout
        while True:
            event = self._next(deadline)
            if event[0] == "LINE" and b"READY" in event[1]:
                break
            if event[0] == "EOF":
                raise RuntimeError(self._failure("serve engine exited before READY"))

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        try:
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    self.events.put(("EOF",))
                    return
                stripped = line.rstrip(b"\r\n")
                if stripped.startswith(b"DATA "):
                    parts = stripped.split()
                    if len(parts) != 3:
                        self.events.put(("ERROR", f"malformed DATA header: {stripped!r}"))
                        return
                    size = int(parts[2])
                    payload = self.proc.stdout.read(size)
                    terminator = self.proc.stdout.read(1)
                    if len(payload) != size or terminator != b"\n":
                        self.events.put(("ERROR", "truncated DATA frame"))
                        return
                    self.events.put(("DATA", parts[1].decode("ascii"), payload))
                else:
                    self.events.put(("LINE", stripped))
        except Exception as exc:  # pragma: no cover - diagnostic path
            self.events.put(("ERROR", f"protocol reader failed: {exc}"))

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        while True:
            chunk = self.proc.stderr.read(4096)
            if not chunk:
                return
            self.stderr_chunks.append(chunk)

    def _failure(self, message: str) -> str:
        stderr = b"".join(self.stderr_chunks).decode("utf-8", errors="replace")
        return f"{message}\nserve stderr:\n{stderr}"

    def _next(self, deadline: float) -> tuple:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(self._failure("timed out waiting for serve protocol"))
        try:
            return self.events.get(timeout=remaining)
        except queue.Empty as exc:
            raise RuntimeError(self._failure("timed out waiting for serve protocol")) from exc

    def request(
        self,
        request_id: str,
        payload: bytes,
        max_tokens: int,
        *,
        cancel_after_accept: bool = False,
    ) -> ServeResult:
        assert self.proc.stdin is not None
        header = (
            f"SUBMIT {request_id} 0 {len(payload)} {max_tokens} 0 0\n"
        ).encode("ascii")
        self.proc.stdin.write(header + payload + b"\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + self.timeout
        accepted = False
        frames: list[bytes] = []
        while True:
            event = self._next(deadline)
            if event[0] == "ERROR":
                raise RuntimeError(self._failure(str(event[1])))
            if event[0] == "EOF":
                raise RuntimeError(self._failure(f"serve engine exited during {request_id}"))
            if event[0] == "DATA":
                _, event_id, data = event
                if event_id == request_id:
                    if not accepted:
                        raise RuntimeError("DATA arrived before ACCEPT")
                    frames.append(data)
                continue
            line = event[1]
            parts = line.split()
            if len(parts) >= 2 and parts[0] == b"ERROR" and parts[1].decode() == request_id:
                raise RuntimeError(self._failure(line.decode("utf-8", errors="replace")))
            if len(parts) == 3 and parts[0] == b"ACCEPT" and parts[1].decode() == request_id:
                accepted = True
                if cancel_after_accept:
                    self.proc.stdin.write(f"CANCEL {request_id}\n".encode("ascii"))
                    self.proc.stdin.flush()
                continue
            if len(parts) >= 9 and parts[0] == b"DONE" and parts[1].decode() == request_id:
                if not accepted:
                    raise RuntimeError("DONE arrived before ACCEPT")
                return ServeResult(
                    frames=tuple(frames),
                    generated=int(parts[3]),
                    prompt_tokens=int(parts[7]),
                    length_limited=int(parts[8]),
                )

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def __enter__(self) -> "ServeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def token_text_map(model_dir: Path) -> dict[int, str]:
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    by_id = {
        int(token_id): text
        for text, token_id in tokenizer.get("model", {}).get("vocab", {}).items()
    }
    for token in tokenizer.get("added_tokens", []):
        by_id[int(token["id"])] = token["content"]
    return by_id


def check_serve_isolation(engine: Path, model_dir: Path, case: dict) -> ServeResult:
    by_id = token_text_map(model_dir)
    try:
        payload = "".join(by_id[token] for token in case["prompt_ids"]).encode("utf-8")
        expected = "".join(by_id[token] for token in case["greedy_new_ids"]).encode("utf-8")
    except KeyError as exc:
        raise RuntimeError(f"fixture tokenizer has no text for token id {exc.args[0]}") from exc
    max_tokens = case["max_new_tokens"]
    with ServeClient(engine, model_dir) as same_process:
        first = same_process.request("isolation-a", payload, max_tokens)
        second = same_process.request("isolation-b", payload, max_tokens)
    with ServeClient(engine, model_dir) as fresh_process:
        fresh = fresh_process.request("isolation-fresh", payload, max_tokens)
        cancelled = fresh_process.request(
            "isolation-cancel", payload, max_tokens, cancel_after_accept=True
        )
    if first != second or first != fresh:
        raise RuntimeError(
            "serve isolation mismatch:\n"
            f"  first={first}\n  second={second}\n  fresh={fresh}"
        )
    if b"".join(first.frames) != expected:
        raise RuntimeError(
            "serve output differs from greedy oracle: "
            f"engine={b''.join(first.frames)!r} oracle={expected!r}"
        )
    if first.stable_done != (max_tokens, len(case["prompt_ids"]), 1):
        raise RuntimeError(
            f"unexpected stable DONE fields {first.stable_done}; "
            f"want {(max_tokens, len(case['prompt_ids']), 1)}"
        )
    if cancelled.generated != 1 or cancelled.prompt_tokens != len(case["prompt_ids"]) or cancelled.length_limited != 0:
        raise RuntimeError(
            f"CANCEL/DONE framing produced {cancelled.stable_done}; "
            f"want {(1, len(case['prompt_ids']), 0)}"
        )
    return first


def parse_lines(lines: list[str], prefix: str) -> list[int]:
    out = []
    for line in lines:
        line = line.strip()
        if line.startswith(prefix):
            out.append(int(line[len(prefix):].strip()))
    return out


def check_case(name: str, case: dict, engine: Path, model_dir: Path, verbose: bool) -> bool:
    ok = True

    # 1) teacher-forced logits: feed the ACTUAL sequence (greedy_full_ids =
    #    prompt + greedy continuation); each position's argmax must equal
    #    teacher_forcing_ids[i] (the oracle's prediction at that position).
    actual = case.get("greedy_full_ids") or (case["prompt_ids"] + case["greedy_new_ids"])
    teacher_env = {
        "QWENMOE_MODE": "teacher",
        "QWENMOE_TEACHER": " ".join(str(i) for i in actual),
    }
    preds = parse_lines(run_engine(engine, model_dir, teacher_env), "PRED")
    expected = case["teacher_forcing_ids"]
    if preds != expected:
        ok = False
        first = next(
            (i for i, (a, b) in enumerate(zip(preds, expected)) if a != b),
            min(len(preds), len(expected)),
        )
        print(f"  FAIL {name}: teacher-forced divergence at position {first}: "
              f"engine={preds[first] if first < len(preds) else 'EOF'} "
              f"oracle={expected[first] if first < len(expected) else 'EOF'} "
              f"(engine {len(preds)}/{len(expected)} preds)")
    elif verbose:
        print(f"  ok   {name}: teacher-forced {len(preds)}/{len(expected)} positions match")

    # 2) greedy decode
    greedy_env = {
        "QWENMOE_MODE": "greedy",
        "QWENMOE_PROMPT_IDS": " ".join(str(i) for i in case["prompt_ids"]),
        "QWENMOE_MAX_NEW": str(case["max_new_tokens"]),
    }
    ids = parse_lines(run_engine(engine, model_dir, greedy_env), "ID")
    expected_new = case["greedy_new_ids"]
    if ids != expected_new:
        ok = False
        first = next(
            (i for i, (a, b) in enumerate(zip(ids, expected_new)) if a != b),
            min(len(ids), len(expected_new)),
        )
        print(f"  FAIL {name}: greedy divergence at token {first}: "
              f"engine={ids[first] if first < len(ids) else 'EOS'} "
              f"oracle={expected_new[first] if first < len(expected_new) else 'EOS'} "
              f"(engine {len(ids)}/{len(expected_new)} tokens)")
    elif verbose:
        print(f"  ok   {name}: greedy {len(ids)}/{len(expected_new)} tokens match")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "qwen_moe_tiny"
    parser.add_argument("--model", type=Path, default=default)
    parser.add_argument("--engine", type=Path, default=Path(__file__).resolve().parents[1] / "qwen_moe")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    engine = args.engine.resolve()
    model_dir = args.model.resolve()
    if not engine.exists():
        print(f"engine not built: {engine} (run: make -C c qwen_moe)")
        return 2
    ref_path = model_dir / "ref.json"
    if not ref_path.exists():
        print(f"missing fixture: {ref_path} (run: c/mio_env/bin/python "
              f"c/tools/make_qwen_moe_tiny.py --output {model_dir} --force)")
        return 2

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    print(f"ref: {ref_path} (schema {ref.get('schema_version')}, "
          f"transformers {ref.get('transformers_version')})")

    ok = True
    for name, case in ref["cases"].items():
        ok = check_case(name, case, engine, model_dir, args.verbose) and ok

    if ok:
        serve_case = next(iter(ref["cases"].values()))
        try:
            serve = check_serve_isolation(engine, model_dir, serve_case)
            print("  ok   serve isolation: same-process A/B == fresh process; "
                  f"DATA frames={len(serve.frames)} DONE={serve.stable_done}; "
                  "ACCEPT-before-prefill and CANCEL/DONE ok")
        except RuntimeError as exc:
            ok = False
            print(f"  FAIL serve isolation: {exc}")

    if ok:
        print("PASS: qwen_moe is token-exact vs the transformers oracle")
        return 0
    print("FAIL: qwen_moe diverges from the transformers oracle")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

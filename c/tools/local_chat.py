"""Direct, no-network interactive chat for the public ``coli`` launcher.

This module deliberately consumes the same ``openai_server.Engine`` process
session and model-native prompt renderers that back ``coli serve``.  The Engine
class is stdin/stdout only; HTTP remains a separate transport owned exclusively
by the serve/web commands.
"""
import os
import sys
import threading
import time


def install(core):
    """Replace the legacy chat command in an executed coli CLI namespace."""
    C = core["C"]

    def model_id_for(arch):
        return {
            "glm": "glm-5.2-colibri",
            "inkling": "inkling-colibri",
            "kimi": "kimi-k3-colibri",
            "deepseek_v4": "deepseek-v4-colibri",
            "olmoe": "olmoe-colibri",
            "qwen3_moe": "qwen3-moe-colibri",
        }.get(arch, arch + "-colibri")

    class ReasoningWriter:
        def __init__(self, visible=True):
            self.visible = visible
            self.started = False
            self.line_start = True
            self.parts = []

        def feed(self, text):
            if not text:
                return
            self.parts.append(text)
            if not self.visible:
                return
            if not self.started:
                self.started = True
                sys.stdout.write(f"  {C.dgray}┌ thinking{C.r}\n")
            chunks = text.splitlines(True)
            for chunk in chunks:
                if self.line_start:
                    sys.stdout.write(f"  {C.dgray}│{C.r} {C.dim}")
                sys.stdout.write(chunk + C.r)
                self.line_start = chunk.endswith("\n")
            sys.stdout.flush()

        def close(self):
            if not self.visible or not self.started:
                return
            if not self.line_start:
                sys.stdout.write("\n")
            sys.stdout.write(f"  {C.dgray}└─{C.r}\n")
            sys.stdout.flush()
            self.started = False
            self.line_start = True

        @property
        def text(self):
            return "".join(self.parts)

    def cmd_chat(a):
        # `chat` is intentionally a local-process command now.  Even an explicit
        # legacy --attach must fail before probing localhost so this command has a
        # hard no-socket contract.  `serve` remains the network/API entry point.
        if getattr(a, "attach", None):
            sys.exit("--attach is no longer supported by `coli chat`: chat is local-only and "
                     "opens no socket. Use `coli serve` for API/network clients.")

        arch = core["model_arch"](a.model)
        engine = core["engine_for"](a.model)
        core["need_model"](a.model, engine)
        maximum = core["ngen_for"](a, interactive=True)
        env = core["env_for_engine"](a, arch) if arch != "glm" else core["env_for"](a)
        # The request carries its own max token count, but keep engine-level
        # ceilings consistent too so no family silently clamps an interactive
        # session to a one-shot default.
        env["NGEN"] = str(maximum)
        if arch in ("qwen3_moe", "olmoe"):
            env["MAX_NEW"] = str(maximum)

        import openai_server as gateway
        gateway.ARCH = arch
        model_id = model_id_for(arch)
        core["banner"](f"chat · {model_id} · local process", model=a.model)
        print(f"  {C.dim}engine {os.path.basename(engine)} · stdin/stdout · no network socket{C.r}")
        print(f"  {C.dim}type and press Enter · Ctrl-C stops the answer · :reset clears the conversation · :q exits{C.r}\n")

        runtime = None
        spinner = core["Spinner"](f"loading {model_id}…")
        spinner.start()
        try:
            runtime = gateway.Engine(
                engine, os.path.abspath(a.model), cap=a.cap,
                max_tokens=maximum, env=env, kv_slots=1)
        except Exception as error:
            spinner.stop()
            sys.exit(f"{C.yel}could not start {model_id}:{C.r} {error}\n"
                     f"  engine: {engine}\n  model: {a.model}")
        spinner.stop()
        print(f"  {C.grn}✓ ready{C.r} {C.dim}· local engine process started{C.r}\n")

        messages = []
        width = core["term_w"]() - 4
        try:
            while True:
                if core["TTY"]:
                    print(f"  {C.dgray}╭{'─'*width}╮{C.r}")
                    try:
                        message = core["read_prompt"](
                            f"  {C.dgray}│{C.r} {C.teal}{C.b}›{C.r} ")
                    except EOFError:
                        print()
                        break
                    try:
                        message = core["redraw_prompt_box"](message, width)
                    except Exception:
                        print(f"  {C.dgray}╰{'─'*width}╯{C.r}")
                else:
                    try:
                        message = input()
                    except EOFError:
                        break

                message = message.strip("\r\n")
                if message in (":q", ":quit", "exit"):
                    break
                if not message.strip():
                    continue
                if message == ":reset":
                    messages.clear()
                    print(f"  {C.dim}✦ conversation cleared{C.r}\n")
                    continue
                if message in (":more", ":continua", ":piu", ":più"):
                    print(f"  {C.yel}:more is not needed by the shared chat protocol; ask the model to continue.{C.r}\n")
                    continue

                messages.append({"role": "user", "content": message})
                body = {"messages": messages, "max_tokens": maximum}
                if a.temp is not None:
                    body["temperature"] = a.temp

                reasoning_effort = "high" if os.environ.get("COLI_THINK", "0") == "1" else None
                enable_thinking = reasoning_effort is not None
                if arch == "olmoe":
                    enable_thinking = False
                    reasoning_effort = None

                try:
                    prompt = gateway.render_chat_for_arch(
                        messages, enable_thinking, reasoning_effort, None, None)
                    max_tokens, temperature, top_p, grammar, _ = gateway.generation_options(
                        body, maximum)
                    stop_sequences, ignore_leading = gateway.stop_policy(body, True)
                except gateway.APIError as error:
                    messages.pop()
                    print(f"  {C.yel}{error.message}{C.r}\n")
                    continue

                print(f"\n  {C.teal}◆ colibri{C.r}")
                answer = []
                raw_answer = os.environ.get("COLI_RAW") == "1"
                reasoning = ReasoningWriter(os.environ.get("COLI_SHOW_THINK", "1") != "0")
                md = core["MDStream"]("  ")
                response_started = [False]
                thinking_spinner = core["Spinner"]("thinking…")
                thinking_spinner.start()

                def begin_output():
                    if not response_started[0]:
                        thinking_spinner.stop()
                        response_started[0] = True

                def emit_answer(text):
                    if not text:
                        return
                    reasoning.close()
                    begin_output()
                    answer.append(text)
                    if raw_answer:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    else:
                        md.feed(text)

                def emit_reasoning(text):
                    if not text:
                        return
                    # Hidden thinking deliberately leaves the spinner running until
                    # visible answer text arrives, matching the API/TUI behavior.
                    if reasoning.visible:
                        begin_output()
                    reasoning.feed(text)

                if arch == "inkling":
                    splitter = gateway.InklingStreamSplit(
                        emit_answer, emit_reasoning if enable_thinking else None,
                        reasoning.close if enable_thinking else None)
                elif enable_thinking:
                    splitter = gateway.ThinkingStreamSplit(
                        emit_reasoning, emit_answer, reasoning.close,
                        initial_thinking=True)
                else:
                    splitter = None

                def routed_text(text):
                    (splitter.feed if splitter is not None else emit_answer)(text)

                stop_filter = gateway.StopFilter(
                    stop_sequences, routed_text, ignore_leading_stop)
                cancelled = threading.Event()
                result = {}

                def generate():
                    try:
                        result["stats"] = runtime.generate(
                            prompt, max_tokens, temperature, top_p,
                            stop_filter.feed, 0, cancelled.is_set,
                            grammar=grammar, stopped=stop_filter.stopped)
                    except BaseException as error:
                        result["error"] = error
                    finally:
                        try:
                            stop_filter.finish()
                            if splitter is not None:
                                splitter.close()
                        except BaseException as error:
                            result.setdefault("error", error)

                worker = threading.Thread(target=generate, name="coli-local-chat-turn", daemon=True)
                started = time.time()
                worker.start()
                interrupted = False
                hard_interrupt = False
                while worker.is_alive():
                    try:
                        worker.join(0.10)
                    except KeyboardInterrupt:
                        if interrupted:
                            hard_interrupt = True
                            break
                        interrupted = True
                        cancelled.set()
                        thinking_spinner.stop()
                        print(f"\n  {C.yel}⏹ stopping… (Ctrl-C again to quit){C.r}", flush=True)

                if hard_interrupt:
                    raise KeyboardInterrupt
                # A session is serialized by design. If a cancelled request does
                # not reach DONE/CANCEL promptly, do not accept another prompt on
                # the same pipe and risk overlapping generations; close the child.
                if worker.is_alive():
                    worker.join(2.0)
                    if worker.is_alive():
                        thinking_spinner.stop()
                        reasoning.close()
                        if not raw_answer:
                            md.close()
                        messages.pop()
                        print(f"\n  {C.yel}[engine did not acknowledge cancellation; closing this session]{C.r}")
                        break

                thinking_spinner.stop()
                reasoning.close()
                if not raw_answer:
                    md.close()
                elif answer:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                error = result.get("error")
                if error is not None and not isinstance(error, gateway.ClientCancelled):
                    messages.pop()
                    print(f"  {C.yel}[engine request failed: {error}]{C.r}\n")
                    if runtime.process.poll() is not None:
                        break
                    continue

                reply = "".join(answer)
                if reply or reasoning.text:
                    assistant = {"role": "assistant", "content": reply}
                    if reasoning.text:
                        assistant["reasoning_content"] = reasoning.text
                    messages.append(assistant)
                else:
                    messages.pop()

                stats = result.get("stats") or {}
                elapsed = time.time() - started
                if stats:
                    note = " · ⏹ interrupted" if interrupted else ""
                    print(f"  {C.dgray}└─ {stats.get('completion_tokens', 0)} tok · "
                          f"{stats.get('tokens_per_second', 0.0):.2f} tok/s · "
                          f"hit {stats.get('cache_hit_percent', 0.0):.0f}% · "
                          f"RSS {stats.get('rss_gb', 0.0):.1f} GB · {elapsed:.1f}s{note}{C.r}\n")
                elif interrupted:
                    print(f"  {C.dgray}└─ interrupted · {elapsed:.1f}s{C.r}\n")
        except KeyboardInterrupt:
            print(f"\n  {C.dim}interrupted{C.r}")
        finally:
            if runtime is not None:
                runtime.close()

        print(f"  {C.teal}goodbye{C.r} {C.dim}— engine stopped, RAM released{C.r} 🐦\n")

    core["cmd_chat"] = cmd_chat
    return cmd_chat

import argparse
import builtins
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("coli_local_chat", ROOT / "tools" / "local_chat.py")
LOCAL_CHAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOCAL_CHAT)


class _C:
    teal = b = r = dim = dgray = yel = grn = ""


class _Spinner:
    def __init__(self, *args, **kwargs):
        self.started = False
    def start(self):
        self.started = True
    def stop(self):
        self.started = False


class _MD:
    def __init__(self, *args):
        self.text = []
    def feed(self, text):
        self.text.append(text)
    def close(self):
        pass


class _StopFilter:
    def __init__(self, sequences, emit, ignore_leading=False):
        self.emit = emit
    def feed(self, text):
        self.emit(text)
    def finish(self):
        pass
    def stopped(self):
        return False


class _Engine:
    started = 0
    closed = 0
    prompts = []

    def __init__(self, executable, model, cap=None, max_tokens=0, env=None, kv_slots=1):
        type(self).started += 1
        self.process = types.SimpleNamespace(poll=lambda: None)

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, **kwargs):
        type(self).prompts.append(prompt)
        on_text("hello from qwen")
        return {"completion_tokens": 3, "tokens_per_second": 2.0,
                "cache_hit_percent": 50.0, "rss_gb": 4.0,
                "prompt_tokens": 2, "length_limited": False}

    def close(self):
        type(self).closed += 1


class APIError(Exception):
    def __init__(self, message):
        self.message = message


class ClientCancelled(Exception):
    pass


def _gateway():
    module = types.ModuleType("openai_server")
    module.ARCH = None
    module.Engine = _Engine
    module.APIError = APIError
    module.ClientCancelled = ClientCancelled
    module.StopFilter = _StopFilter
    module.render_chat_for_arch = lambda messages, *args, **kwargs: "QWEN_NATIVE_PROMPT"
    module.generation_options = lambda body, limit: (limit, 0.7, 0.9, None, ())
    module.stop_policy = lambda body, chat: ((), False)
    # Not used with thinking disabled / Qwen in this test, but present so a
    # future refactor can select them without changing the fake gateway shape.
    module.ThinkingStreamSplit = object
    module.InklingStreamSplit = object
    return module


def _core():
    def forbidden_probe(*args, **kwargs):
        raise AssertionError("coli chat attempted the legacy HTTP/server probe")

    return {
        "C": _C,
        "model_arch": lambda model: "qwen3_moe",
        "engine_for": lambda model: "/fake/qwen_moe",
        "need_model": lambda model, engine=None: None,
        "ngen_for": lambda args, interactive=False: 8,
        "env_for_engine": lambda args, arch: {},
        "env_for": lambda args: {},
        "banner": lambda *args, **kwargs: None,
        "Spinner": _Spinner,
        "term_w": lambda: 80,
        "TTY": False,
        "MDStream": _MD,
        "server_probe": forbidden_probe,
        "chat_attached": forbidden_probe,
    }


def _args(attach=None):
    return argparse.Namespace(model="/fake/model", cap=None, temp=None, ram=0,
                              ctx=0, ngen=None, attach=attach, no_attach=False,
                              api_key=None)


class LocalChatTest(unittest.TestCase):
    def setUp(self):
        _Engine.started = _Engine.closed = 0
        _Engine.prompts = []
        self.old_gateway = sys.modules.get("openai_server")
        sys.modules["openai_server"] = _gateway()

    def tearDown(self):
        if self.old_gateway is None:
            sys.modules.pop("openai_server", None)
        else:
            sys.modules["openai_server"] = self.old_gateway

    def test_qwen_turn_uses_local_engine_session_only(self):
        core = _core()
        command = LOCAL_CHAT.install(core)
        with mock.patch.object(builtins, "input", side_effect=["hello", ":q"]), \
             mock.patch.dict(os.environ, {"COLI_THINK": "0"}, clear=False):
            command(_args())
        self.assertEqual(_Engine.started, 1)
        self.assertEqual(_Engine.closed, 1)
        self.assertEqual(_Engine.prompts, ["QWEN_NATIVE_PROMPT"])

    def test_legacy_attach_fails_before_engine_or_probe(self):
        core = _core()
        command = LOCAL_CHAT.install(core)
        with self.assertRaises(SystemExit) as raised:
            command(_args("http://127.0.0.1:8000"))
        self.assertIn("local-only", str(raised.exception))
        self.assertEqual(_Engine.started, 0)


if __name__ == "__main__":
    unittest.main()

"""Every model type the launcher NAMES, the launcher must also DISPATCH.

#879: `coli` printed "OLMoE · 7B" in its banner and then handed the request to
the GLM engine, because `_BANNER_MODELS` knew about `olmoe` and `model_arch()`
did not. The engine binary shipped in the release archive the whole time; the
launcher structurally could not select it.

The release check that was supposed to catch this (#868, "coli would not
resolve these next to itself") asserted the binary was PRESENT next to `coli`.
It passed. Presence is not dispatch, and a green test that proves the weaker
property is worse than no test, because it is read as proving the stronger one.

So the invariant is stated once, here, over the launcher's own table:

    for every model_type in _BANNER_MODELS:
        model_arch() must classify it, and engine_for() must resolve it to a
        binary name that is not merely the GLM fallback by accident

It is self-updating. Adding a family to the banner roster without teaching the
dispatcher about it fails this file, rather than shipping and being found by
someone who downloaded a release.
"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_dispatch_test", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class LauncherDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def _model_dir(self, model_type):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "config.json").write_text(
            json.dumps({"model_type": model_type}), encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        return str(root)

    def test_every_named_model_type_is_classified(self):
        """A model_type in the banner roster must not fall through to the
        `return "glm"` default -- except glm itself, whose default that is."""
        for model_type, label, _ in self.cli._BANNER_MODELS:
            with self.subTest(model_type=model_type):
                arch = self.cli.model_arch(self._model_dir(model_type))
                if model_type == "glm":
                    self.assertEqual(arch, "glm")
                    continue
                self.assertNotEqual(
                    arch, "glm",
                    f"the banner prints {label!r} for model_type {model_type!r} "
                    f"and model_arch() returns 'glm' -- the launcher names a "
                    f"family it cannot dispatch (#879)")

    def test_every_named_model_type_resolves_to_a_distinct_engine(self):
        """Two families must not resolve to the same binary. That is the shape
        the bug took: everything unrecognised quietly became the GLM engine.
        Versioned families (qwen3_5_moe, qwen3_6_moe, ...) share ONE engine by
        design, so the distinctness key is the family (name minus version)."""
        def family(model_type):
            parts = model_type.split("_")
            if len(parts) > 2 and parts[-2].isdigit():
                return parts[0] + "_" + parts[-1]
            return model_type

        seen = {}       # family -> (engine, label)
        engines = {}    # engine -> family
        for model_type, label, _ in self.cli._BANNER_MODELS:
            with self.subTest(model_type=model_type):
                engine = os.path.basename(
                    self.cli.engine_for(self._model_dir(model_type)))
                self.assertTrue(engine, f"{label}: engine_for returned nothing")
                key = family(model_type)
                if key in seen:
                    # versioned siblings must share their family's engine
                    self.assertEqual(
                        seen[key][0], engine,
                        f"{label} ({model_type}) resolves to {engine!r} but the "
                        f"{key} family resolves to {seen[key][0]!r}")
                else:
                    seen[key] = (engine, label)
                previous = engines.get(engine)
                self.assertIsNone(
                    None if previous == key else previous,
                    f"{label} ({model_type}) and family {previous} both dispatch "
                    f"to {engine!r} -- one of them is being misrouted (#879)")
                engines[engine] = key

    def test_engine_for_has_a_branch_for_every_arch_model_arch_can_return(self):
        """engine_for() indexes a dict with model_arch()'s return value. A value
        model_arch can produce and the dict lacks is a KeyError at runtime, on
        the user's machine, after the banner has already printed."""
        for model_type, label, _ in self.cli._BANNER_MODELS:
            with self.subTest(model_type=model_type):
                try:
                    self.cli.engine_for(self._model_dir(model_type))
                except KeyError as error:
                    self.fail(f"{label}: engine_for raised KeyError({error}) -- "
                              f"model_arch() classifies this family and the "
                              f"engine dict has no entry for it")

    def test_the_roster_is_not_empty(self):
        """Guards the whole file: if _BANNER_MODELS is ever renamed or emptied,
        the loops above pass vacuously and this suite proves nothing."""
        self.assertGreaterEqual(len(self.cli._BANNER_MODELS), 5)


class QwenDirectLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name)
        (self.model / "config.json").write_text(
            json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8")

    def _launch(self, cap=None, ram=None, ctx=None, ambient=None):
        args = types.SimpleNamespace(
            model=str(self.model), prompt=["hello", "qwen"], cap=cap,
            ram=ram, ctx=ctx, ngen=64, temp=None, policy="quality", topp=0,
            topk=0, repin=0, auto_tier=False, gpu=None)
        completed = types.SimpleNamespace(returncode=0)
        with mock.patch.dict(os.environ, ambient or {}, clear=True), \
             mock.patch("resource_plan.physical_cpu_count", return_value=6), \
             mock.patch.object(self.cli, "need_model"), \
             mock.patch.object(self.cli, "banner"), \
             mock.patch.object(self.cli, "engine_for", return_value="/stub/qwen_moe"), \
             mock.patch.object(self.cli.subprocess, "call", return_value=0), \
             mock.patch.object(self.cli.subprocess, "run", return_value=completed) as run:
            with self.assertRaises(SystemExit) as stopped:
                self.cli.cmd_run(args)
        self.assertEqual(stopped.exception.code, 0)
        return run.call_args

    def test_direct_qwen_resource_precedence_matrix(self):
        cases = (
            ("absent delegates to native defaults", None, None, None, {}, None, {}),
            ("explicit cap is argv", 13, 32, 8192, {"CACHE": "7"}, "13",
             {"CACHE": "7", "RAM_GB": "32", "CTX": "8192"}),
            ("CACHE wins over RAM", None, 32, None, {"CACHE": "7"}, None,
             {"CACHE": "7", "RAM_GB": "32"}),
            ("RAM derives cache and CTX forwards", None, 32, 8192, {}, None,
             {"CACHE": "16", "RAM_GB": "32", "CTX": "8192"}),
        )
        for label, cap, ram, ctx, ambient, expected_cap, expected_env in cases:
            with self.subTest(label=label):
                call = self._launch(cap, ram, ctx, ambient)
                self.assertIsNotNone(call)
                cmd = call.args[0]
                expected = ["/stub/qwen_moe", str(self.model)]
                if expected_cap is not None:
                    expected.append(expected_cap)
                self.assertEqual(cmd, expected)
                self.assertEqual(call.kwargs["input"], "hello qwen\n")
                for key, value in expected_env.items():
                    self.assertEqual(call.kwargs["env"][key], value)
                for key in {"CACHE", "RAM_GB", "CTX"} - expected_env.keys():
                    self.assertNotIn(key, call.kwargs["env"])


class ProjectPythonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_bench_and_convert_use_windows_project_python(self):
        python = self.root / "mio_env" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")
        data = self.root / "data"
        data.mkdir()
        (data / "hellaswag.jsonl").write_text("", encoding="utf-8")

        bench = types.SimpleNamespace(
            model="model", tasks=["hellaswag"], data=str(data), limit=1, ram=None)
        convert = types.SimpleNamespace(
            model="output", repo="repo", ebits=4, io_bits=4,
            group_size=128, xbits=None, no_mtp=True)

        with mock.patch.object(self.cli, "HERE", str(self.root)), \
             mock.patch.object(self.cli.sys, "platform", "win32"), \
             mock.patch.object(self.cli, "need_model"), \
             mock.patch.object(self.cli, "banner"), \
             mock.patch.object(self.cli, "env_for", return_value={}), \
             mock.patch.object(self.cli.subprocess, "call", return_value=0) as call:
            with self.assertRaisesRegex(SystemExit, "0"):
                self.cli.cmd_bench(bench)
            self.assertEqual(call.call_args.args[0][0], str(python))

            call.reset_mock()
            with self.assertRaisesRegex(SystemExit, "0"):
                self.cli.cmd_convert(convert)
            self.assertEqual(call.call_args.args[0][0], str(python))

    def test_posix_project_python_is_unchanged(self):
        python = self.root / "mio_env" / "bin" / "python3"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")
        with mock.patch.object(self.cli, "HERE", str(self.root)), \
             mock.patch.object(self.cli.sys, "platform", "linux"):
            self.assertEqual(self.cli.project_python(), str(python))

    def test_missing_project_python_falls_back_to_current_interpreter(self):
        with mock.patch.object(self.cli, "HERE", str(self.root)), \
             mock.patch.object(self.cli.sys, "platform", "win32"):
            self.assertEqual(self.cli.project_python(), sys.executable)


if __name__ == "__main__":
    unittest.main()

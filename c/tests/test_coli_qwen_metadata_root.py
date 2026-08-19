import argparse
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class QwenColiMetadataRootTest(unittest.TestCase):
    def test_package_owns_config_even_when_legacy_env_is_set(self):
        namespace = runpy.run_path(str(ROOT / "coli"), run_name="coli_bootstrap_test")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "manifest.coli").write_bytes(b"test")
            args = argparse.Namespace(
                model=str(model), ram=0, temp=None, ctx=0, ngen=None,
                cap=None, topp=0, topk=0, repin=0, auto_tier=False,
                no_tune_profile=False, gpu=None, vram=0, policy="quality",
            )
            with mock.patch.dict(os.environ, {"COLI_CONFIG": "/external/hf"}, clear=False):
                env = namespace["env_for_engine"](args, "qwen3_moe")
            self.assertEqual(env["COLI_CONFIG"], str(model.resolve()))

    def test_non_package_does_not_invent_legacy_config(self):
        namespace = runpy.run_path(str(ROOT / "coli"), run_name="coli_bootstrap_test")
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                model=directory, ram=0, temp=None, ctx=0, ngen=None,
                cap=None, topp=0, topk=0, repin=0, auto_tier=False,
                no_tune_profile=False, gpu=None, vram=0, policy="quality",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                env = namespace["env_for_engine"](args, "qwen3_moe")
            self.assertNotIn("COLI_CONFIG", env)


if __name__ == "__main__":
    unittest.main()

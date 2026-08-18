import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tests" / "test_qwen_prefix_cache.c"


class QwenPrefixCacheCTest(unittest.TestCase):
    def test_hybrid_snapshot_cache(self):
        cc = os.environ.get("CC", "cc")
        with tempfile.TemporaryDirectory() as td:
            exe = pathlib.Path(td) / ("qwen_prefix_cache_test.exe" if os.name == "nt" else "qwen_prefix_cache_test")
            build = subprocess.run(
                [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(SRC), "-o", str(exe)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(build.returncode, 0, build.stdout)
            run = subprocess.run(
                [str(exe)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(run.returncode, 0, run.stdout)
            self.assertIn("qwen prefix cache: ok", run.stdout)


if __name__ == "__main__":
    unittest.main()

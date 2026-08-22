import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "tests" / "test_qwen_prefix_backend_fingerprint.c"


@unittest.skipIf(os.name == "nt", "synthetic COLI_METAL header test is POSIX-only")
class QwenPrefixBackendFingerprintCTest(unittest.TestCase):
    def test_resolved_backend_partitions_persistent_identity(self):
        cc = os.environ.get("CC", "cc")
        with tempfile.TemporaryDirectory() as td:
            exe = pathlib.Path(td) / "qwen_prefix_backend_fingerprint"
            build = subprocess.run(
                [
                    cc,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-Wno-misleading-indentation",
                    "-DCOLI_METAL",
                    str(SRC),
                    "-o",
                    str(exe),
                ],
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
            self.assertIn("qwen prefix backend fingerprint: ok", run.stdout)


if __name__ == "__main__":
    unittest.main()

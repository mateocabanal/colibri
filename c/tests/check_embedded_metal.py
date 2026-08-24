#!/usr/bin/env python3
"""Compile embedded MSL strings so runtime-only shader errors fail CI.

The Apple8 direct backend builds its Metal libraries from Objective-C++ raw
strings at runtime. A normal `clang++` build therefore cannot validate the MSL.
This script extracts every `R"METAL(... )METAL"` payload from the backend and
runs Apple's offline Metal compiler on it.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "c" / "apple8_metalio_direct.mm"
PATTERN = re.compile(
    r"static\s+const\s+char\s*\*\s*(\w+)\s*=\s*R\"METAL\((.*?)\)METAL\";",
    re.DOTALL,
)


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    shaders = PATTERN.findall(text)
    if not shaders:
        print(f"no embedded Metal shaders found in {SOURCE}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="colibri-msl-") as td:
        tmp = pathlib.Path(td)
        for name, source in shaders:
            metal = tmp / f"{name}.metal"
            air = tmp / f"{name}.air"
            metal.write_text(source, encoding="utf-8")
            cmd = ["xcrun", "-sdk", "macosx", "metal", "-c", str(metal), "-o", str(air)]
            proc = subprocess.run(cmd, text=True, capture_output=True)
            if proc.returncode != 0:
                print(f"embedded Metal shader {name} failed to compile", file=sys.stderr)
                if proc.stdout:
                    print(proc.stdout, file=sys.stderr)
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr)
                return proc.returncode
            print(f"ok: {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

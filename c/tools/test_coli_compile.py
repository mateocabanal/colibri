#!/usr/bin/env python3
"""Dependency-free tests for coli_compile.py.

The safetensors fixtures below are real format-conformant files (8-byte header
length, padded JSON header, payload offsets relative to the data section), but
small enough that CI can inspect every byte.  The generated CSF package is also
opened through the independent C reader from issue #23 when tools/coli_check is
available, preventing writer and verifier from accidentally agreeing on the
same Python bug.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import coli_compile as cc


DTYPE_BYTES = cc.DTYPE_BYTES


def payload(dtype: str, shape: tuple[int, ...], seed: int) -> bytes:
    n = 1
    for d in shape:
        n *= d
    size = n * DTYPE_BYTES[dtype]
    return bytes(((seed + i * 29) & 0xFF) for i in range(size))


def write_safetensors(path: Path,
                      tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
                      metadata: dict[str, str] | None = None) -> None:
    cursor = 0
    header: dict[str, object] = {}
    data = bytearray()
    for name in sorted(tensors, key=lambda s: s.encode("utf-8")):
        dtype, shape, raw = tensors[name]
        expected = DTYPE_BYTES[dtype]
        for d in shape:
            expected *= d
        if len(raw) != expected:
            raise AssertionError((name, len(raw), expected))
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        data.extend(raw)
        cursor += len(raw)
    if metadata:
        header["__metadata__"] = dict(metadata)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def make_model(root: Path, *, incomplete: bool = False,
               bad_dtype: bool = False, bad_geometry: bool = False,
               conflicting_stamp: bool = False) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"architectures": ["DeepSeekV4ForCausalLM"], "hidden_size": 32},
                   sort_keys=True, separators=(",", ":")), encoding="utf-8")
    (model / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {"type": "WordLevel", "vocab": {"x": 0}}},
                   sort_keys=True, separators=(",", ":")), encoding="utf-8")

    expert_prefix = "layers.1.ffn.experts.2"
    stamps = {
        f"{expert_prefix}.w1.weight": "bad-format" if conflicting_stamp else "mxfp4",
        f"{expert_prefix}.w2.weight": "mxfp4",
        f"{expert_prefix}.w3.weight": "mxfp4",
    }
    metadata = {"colibri.fmt": json.dumps(stamps, sort_keys=True, separators=(",", ":"))}

    weight_dtype = "U8" if bad_dtype else "I8"
    scale_shape = (1, 2) if bad_geometry else (1, 1)
    shard0: dict[str, tuple[str, tuple[int, ...], bytes]] = {
        "embed.weight": ("BF16", (2,), payload("BF16", (2,), 1)),
        f"{expert_prefix}.w1.weight": (weight_dtype, (1, 16), payload(weight_dtype, (1, 16), 7)),
        f"{expert_prefix}.w1.scale": ("F8_E8M0", scale_shape, payload("F8_E8M0", scale_shape, 11)),
        f"{expert_prefix}.w2.weight": ("I8", (1, 16), payload("I8", (1, 16), 13)),
        f"{expert_prefix}.w2.scale": ("F8_E8M0", (1, 1), payload("F8_E8M0", (1, 1), 17)),
    }
    shard1: dict[str, tuple[str, tuple[int, ...], bytes]] = {
        "layers.0.norm.weight": ("F32", (1,), payload("F32", (1,), 23)),
        f"{expert_prefix}.w3.weight": ("I8", (1, 16), payload("I8", (1, 16), 31)),
    }
    if not incomplete:
        shard1[f"{expert_prefix}.w3.scale"] = (
            "F8_E8M0", (1, 1), payload("F8_E8M0", (1, 1), 37))

    write_safetensors(model / "model-00001-of-00002.safetensors", shard0, metadata)
    write_safetensors(model / "model-00002-of-00002.safetensors", shard1)
    weight_map = {name: "model-00001-of-00002.safetensors" for name in shard0}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in shard1})
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(len(v[2]) for v in shard0.values()) +
                                  sum(len(v[2]) for v in shard1.values())},
                    "weight_map": weight_map},
                   sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return model


def package_bytes(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for entry in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix().encode()):
        if entry.is_file():
            result[entry.relative_to(path).as_posix()] = entry.read_bytes()
    return result


def tree_digest(path: Path) -> str:
    h = hashlib.sha256()
    for rel, data in package_bytes(path).items():
        raw = rel.encode("utf-8")
        h.update(struct.pack("<I", len(raw)))
        h.update(raw)
        h.update(struct.pack("<Q", len(data)))
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


def c_checker() -> Path | None:
    suffix = ".exe" if os.name == "nt" else ""
    candidate = Path(__file__).resolve().with_name("coli_check" + suffix)
    return candidate if candidate.is_file() else None


def check_with_c_reader(package: Path) -> None:
    checker = c_checker()
    if checker is None:
        return
    proc = subprocess.run([str(checker), str(package)], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False)
    if proc.returncode:
        raise AssertionError(
            f"C reader rejected compiler output ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    if "coli_check: ok" not in proc.stdout:
        raise AssertionError(proc.stdout)


class CompilerTests(unittest.TestCase):
    def test_reproducible_multishard_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="coli-compile-") as td:
            root = Path(td)
            model = make_model(root)
            source_before = tree_digest(model)
            out_a, out_b = root / "a.coli", root / "b.coli"
            plan_a = cc.compile_model(model, out_a, shard_size_bytes=4500,
                                      verify=True, codec="none")
            plan_b = cc.compile_model(model, out_b, shard_size_bytes=4500,
                                      verify=True, codec="auto")
            self.assertGreaterEqual(len(plan_a.shards), 2)
            self.assertEqual(len(plan_a.records), 3)  # 2 ordinary + 1 compound expert
            experts = [r for r in plan_a.records if r.kind == cc.REC_EXPERT]
            self.assertEqual([(r.layer, r.expert) for r in experts], [(1, 2)])
            self.assertEqual([m.role for m in experts[0].matrices],
                             [cc.ROLE_GATE, cc.ROLE_UP, cc.ROLE_DOWN])
            self.assertEqual(experts[0].matrices[0].weight.name,
                             "layers.1.ffn.experts.2.w1.weight")
            self.assertEqual(experts[0].matrices[1].weight.name,
                             "layers.1.ffn.experts.2.w3.weight")
            self.assertEqual(experts[0].matrices[2].weight.name,
                             "layers.1.ffn.experts.2.w2.weight")
            self.assertEqual(package_bytes(out_a), package_bytes(out_b))
            self.assertEqual(tree_digest(model), source_before)
            check_with_c_reader(out_a)
            check_with_c_reader(out_b)

    def test_fingerprint_stable_and_sensitive_to_sidecar(self) -> None:
        with tempfile.TemporaryDirectory(prefix="coli-fingerprint-") as td:
            root = Path(td)
            model = make_model(root)
            out_a, out_b = root / "a.coli", root / "b.coli"
            cc.compile_model(model, out_a, shard_size_bytes=1 << 20, verify=True)
            fp_a = (out_a / "manifest.coli").read_bytes()[112:144]
            # Reformat config without changing its parsed meaning: provenance is
            # byte identity, so this must change the source fingerprint.
            config = model / "config.json"
            obj = json.loads(config.read_text(encoding="utf-8"))
            config.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
            cc.compile_model(model, out_b, shard_size_bytes=1 << 20, verify=True)
            fp_b = (out_b / "manifest.coli").read_bytes()[112:144]
            self.assertNotEqual(fp_a, fp_b)
            check_with_c_reader(out_b)

    def test_incomplete_expert_is_hard_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="coli-missing-") as td:
            root = Path(td)
            model = make_model(root, incomplete=True)
            with self.assertRaisesRegex(cc.CompileError, "incomplete expert"):
                cc.compile_model(model, root / "bad.coli", shard_size_bytes=1 << 20)
            self.assertFalse((root / "bad.coli").exists())

    def test_bad_expert_dtype_and_geometry_are_hard_errors(self) -> None:
        for keyword in ("bad_dtype", "bad_geometry"):
            with self.subTest(keyword=keyword), tempfile.TemporaryDirectory(prefix="coli-shape-") as td:
                root = Path(td)
                model = make_model(root, **{keyword: True})
                with self.assertRaisesRegex(cc.CompileError, "invalid MXFP4"):
                    cc.compile_model(model, root / "bad.coli", shard_size_bytes=1 << 20)
                self.assertFalse((root / "bad.coli").exists())

    def test_conflicting_source_format_stamp_refuses(self) -> None:
        with tempfile.TemporaryDirectory(prefix="coli-stamp-") as td:
            root = Path(td)
            model = make_model(root, conflicting_stamp=True)
            with self.assertRaisesRegex(cc.CompileError, "conflicting format stamp"):
                cc.compile_model(model, root / "bad.coli", shard_size_bytes=1 << 20)

    def test_dry_run_and_rans_refusal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="coli-dry-") as td:
            root = Path(td)
            model = make_model(root)
            out = root / "out.coli"
            plan = cc.compile_model(model, out, shard_size_bytes=4500, dry_run=True)
            self.assertEqual(len(plan.records), 3)
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(cc.CompileError, "requires issue #6"):
                cc.compile_model(model, out, shard_size_bytes=4500,
                                 codec="mxfp4-rans256-g0")
            self.assertFalse(out.exists())

    def test_force_failure_preserves_previous_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="coli-force-") as td:
            root = Path(td)
            model = make_model(root)
            out = root / "out.coli"
            cc.compile_model(model, out, shard_size_bytes=1 << 20, verify=True)
            before = package_bytes(out)
            # Mutate the source inventory into an invalid expert. Planning fails
            # before installation; --force must leave the old final package intact.
            shard = model / "model-00002-of-00002.safetensors"
            tensors, _metadata = cc.read_safetensors_header(
                model, "model-00002-of-00002.safetensors")
            # Rebuild just this shard without w3.scale and keep the index stale;
            # either inventory mismatch or incomplete expert is a valid refusal.
            rebuilt: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
            for name, tensor in tensors.items():
                if name.endswith("w3.scale"):
                    continue
                with shard.open("rb") as f:
                    f.seek(tensor.offset)
                    raw = f.read(tensor.nbytes)
                rebuilt[name] = (tensor.dtype, tensor.shape, raw)
            write_safetensors(shard, rebuilt)
            with self.assertRaises(cc.CompileError):
                cc.compile_model(model, out, shard_size_bytes=1 << 20,
                                 verify=True, force=True)
            self.assertEqual(package_bytes(out), before)
            self.assertFalse(Path(str(out) + f".tmp.{os.getpid()}").exists())
            check_with_c_reader(out)

    def test_existing_output_requires_force(self) -> None:
        with tempfile.TemporaryDirectory(prefix="coli-existing-") as td:
            root = Path(td)
            model = make_model(root)
            out = root / "out.coli"
            cc.compile_model(model, out, shard_size_bytes=1 << 20, verify=True)
            before = package_bytes(out)
            with self.assertRaisesRegex(cc.CompileError, "output already exists"):
                cc.compile_model(model, out, shard_size_bytes=1 << 20, verify=True)
            self.assertEqual(package_bytes(out), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)

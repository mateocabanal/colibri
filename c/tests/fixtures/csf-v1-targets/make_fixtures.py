#!/usr/bin/env python3
"""Independently construct the CSF v1.1 target-compatibility fixtures.

This script intentionally does not import colic/compiler code. It writes fields
from the v1.1 spec + target-identity amendment directly so #23 has an oracle
that cannot merely repeat writer implementation bugs.
"""
from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

POLY = 0x82F63B78


def crc32c(data: bytes | bytearray) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (POLY if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def source_fingerprint(source_bytes: bytes) -> bytes:
    path = b"fixture.source"
    h = hashlib.sha256()
    h.update(b"COLI-SOURCE-V1\0")
    h.update(struct.pack("<I", 1))
    h.update(struct.pack("<BI", 3, len(path)))
    h.update(path)
    h.update(struct.pack("<Q", len(source_bytes)))
    h.update(hashlib.sha256(source_bytes).digest())
    return h.digest()


def put_string(h: "hashlib._Hash", value: str) -> None:
    raw = value.encode("utf-8")
    h.update(struct.pack("<I", len(raw)))
    h.update(raw)


def artifact_fingerprint(*, source: bytes, compiler: str, semantic_abi: str,
                         profile: str, quant: str, storage: str,
                         optimization: str, kernel: str, triple: str,
                         target_flags: int, target_os: int, target_arch: int,
                         backend: int, gpu_kind: int, cpu_features: int,
                         family_min: int, family_max: int,
                         capability_min: int, capability_max: int,
                         profile_abi: int, layout_abi: int,
                         kernel_min: int, kernel_max: int,
                         record_alignment: int, io_granularity: int,
                         resident_alignment: int,
                         runtime_features: int) -> bytes:
    h = hashlib.sha256()
    h.update(b"COLI-ARTIFACT-V1\0")
    h.update(source)
    for value in (compiler, semantic_abi, profile, quant, storage,
                  optimization, kernel, triple):
        put_string(h, value)
    h.update(struct.pack("<I", target_flags))
    h.update(struct.pack("<HHHHQ", target_os, target_arch, backend, gpu_kind,
                         cpu_features))
    h.update(struct.pack("<IIII", family_min, family_max,
                         capability_min, capability_max))
    h.update(struct.pack("<IIIIIII", profile_abi, layout_abi,
                         kernel_min, kernel_max, record_alignment,
                         io_granularity, resident_alignment))
    h.update(struct.pack("<Q", runtime_features))
    h.update(b"\0")                      # tuning_valid
    h.update(b"\0" * 32)                 # tuning_fingerprint
    h.update(struct.pack("<Q", 0))       # profile_data_bytes
    h.update(hashlib.sha256(b"").digest())
    return h.digest()


def string_table(strings: list[str]) -> bytes:
    desc_bytes = 16 * len(strings)
    out = bytearray(desc_bytes)
    cursor = desc_bytes
    for i, text in enumerate(strings):
        raw = text.encode("utf-8")
        struct.pack_into("<QII", out, i * 16, cursor, len(raw), 0)
        out.extend(raw)
        cursor += len(raw)
    out.extend(b"\0" * (align_up(len(out), 16) - len(out)))
    return bytes(out)


def build(kind: str) -> tuple[bytes, bytes, dict[str, str | int]]:
    if kind == "apple":
        profile = "macos-arm64-metal-apple8-v1"
        triple = "arm64-apple-macos"
        target_os, target_arch, backend, gpu_kind = 1, 1, 2, 1
        cpu_features = 1 << 0
        family_min, family_max = 8, 0
        capability_min = capability_max = 0
        record_alignment = io_granularity = resident_alignment = 16384
        runtime_features = (1 << 0) | (1 << 1)
        target_flags = (1 << 1) | (1 << 2) | (1 << 3)
        source_bytes = b"apple-target-fixture-v1"
        payload = b"apple8-target-fixture"
    elif kind == "linux-cuda-sm89":
        profile = "linux-x86_64-cuda-sm89-v1"
        triple = "x86_64-linux-gnu-cuda-sm89"
        target_os, target_arch, backend, gpu_kind = 2, 2, 3, 2
        cpu_features = (1 << 1) | (1 << 2)
        family_min = family_max = 0
        capability_min = capability_max = 89
        record_alignment = io_granularity = 4096
        resident_alignment = 256
        runtime_features = (1 << 2) | (1 << 3) | (1 << 4)
        target_flags = (1 << 1) | (1 << 3) | (1 << 4)
        source_bytes = b"linux-cuda-sm89-target-fixture-v1"
        payload = b"cuda-sm89-target-fixture"
    else:
        raise ValueError(kind)

    compiler = "hand-target-fixture/1"
    semantic_abi = "fixture-blob-v1"
    quant = "exact"
    storage = "none"
    optimization = "default"
    kernel = "fixture-kernel-v1"
    strings = ["data-00000.coli", "fixture.blob", profile, compiler,
               quant, storage, optimization, kernel, triple, semantic_abi]
    strings_blob = string_table(strings)

    source_fp = source_fingerprint(source_bytes)
    artifact_fp = artifact_fingerprint(
        source=source_fp, compiler=compiler, semantic_abi=semantic_abi,
        profile=profile, quant=quant, storage=storage,
        optimization=optimization, kernel=kernel, triple=triple,
        target_flags=target_flags,
        target_os=target_os, target_arch=target_arch, backend=backend,
        gpu_kind=gpu_kind, cpu_features=cpu_features,
        family_min=family_min, family_max=family_max,
        capability_min=capability_min, capability_max=capability_max,
        profile_abi=1, layout_abi=1, kernel_min=1, kernel_max=1,
        record_alignment=record_alignment, io_granularity=io_granularity,
        resident_alignment=resident_alignment,
        runtime_features=runtime_features)

    data_size = record_alignment + len(payload)
    data_header = bytearray(128)
    data_header[:8] = b"COLIDAT\0"
    struct.pack_into("<HHI", data_header, 8, 1, 1, 128)
    struct.pack_into("<II", data_header, 16, 0, 0)
    struct.pack_into("<I", data_header, 24, record_alignment)
    struct.pack_into("<Q", data_header, 32, data_size)
    data_header[40:72] = source_fp
    struct.pack_into("<I", data_header, 72, 0)
    data_crc = crc32c(data_header)
    struct.pack_into("<I", data_header, 72, data_crc)
    data = bytearray(data_size)
    data[:128] = data_header
    data[record_alignment:] = payload

    target = bytearray(256)
    target[:8] = b"COLITGT\0"
    struct.pack_into("<HHI", target, 8, 1, 0, 256)
    struct.pack_into("<I", target, 16, target_flags)
    struct.pack_into("<HHHH", target, 20,
                     target_os, target_arch, backend, gpu_kind)
    struct.pack_into("<Q", target, 28, cpu_features)
    struct.pack_into("<IIII", target, 36, family_min, family_max,
                     capability_min, capability_max)
    struct.pack_into("<IIII", target, 52, 1, 1, 1, 1)
    struct.pack_into("<III", target, 68, record_alignment,
                     io_granularity, resident_alignment)
    struct.pack_into("<Q", target, 80, runtime_features)
    struct.pack_into("<IIIIII", target, 88, 2, 4, 5, 6, 7, 8)
    struct.pack_into("<I", target, 164, 9)  # semantic_abi_string_id
    target_crc = crc32c(target)

    target_off = 256
    shard_off = 512
    record_off = 576
    string_off = 672
    manifest = bytearray(string_off + len(strings_blob))
    manifest[:8] = b"COLI\r\n\x1a\n"
    struct.pack_into("<HHI", manifest, 8, 1, 1, 256)
    struct.pack_into("<I", manifest, 16,
                     (1 << 0) | (1 << 1) | (1 << 16))
    struct.pack_into("<I", manifest, 20, 0x01020304)
    struct.pack_into("<II", manifest, 24, record_alignment, len(strings))
    struct.pack_into("<Q", manifest, 32, 1)
    struct.pack_into("<I", manifest, 40, 1)
    struct.pack_into("<QQ", manifest, 48, shard_off, 64)
    struct.pack_into("<QQ", manifest, 64, record_off, 96)
    struct.pack_into("<QQ", manifest, 80, string_off, len(strings_blob))
    manifest[112:144] = source_fp
    struct.pack_into("<II", manifest, 148, 2, 3)
    struct.pack_into("<QQ", manifest, 184, target_off, 256)
    manifest[200:232] = artifact_fp
    struct.pack_into("<I", manifest, 232, target_crc)
    manifest[target_off:target_off + 256] = target

    struct.pack_into("<III", manifest, shard_off, 0, 0, 0)
    struct.pack_into("<Q", manifest, shard_off + 16, data_size)
    struct.pack_into("<I", manifest, shard_off + 24, data_crc)

    record_crc = crc32c(payload)
    struct.pack_into("<QHHHHHHIIiiI", manifest, record_off,
                     1, 4, 0, 0, 0, 0, 1 << 1, 0, 1, -1, -1, 0)
    struct.pack_into("<QQQII", manifest, record_off + 40,
                     record_alignment, len(payload), len(payload),
                     record_crc, record_crc)
    struct.pack_into("<I", manifest, record_off + 72, 0)
    manifest[string_off:string_off + len(strings_blob)] = strings_blob
    struct.pack_into("<I", manifest, 144, 0)
    manifest_crc = crc32c(manifest)
    struct.pack_into("<I", manifest, 144, manifest_crc)

    info: dict[str, str | int] = {
        "source_fingerprint": source_fp.hex(),
        "artifact_fingerprint": artifact_fp.hex(),
        "target_desc_crc32c": target_crc,
        "manifest_crc32c": manifest_crc,
        "data_header_crc32c": data_crc,
        "record_crc32c": record_crc,
    }
    return bytes(manifest), bytes(data), info


EXPECTED = {
    "apple": {
        "manifest_bytes": 976,
        "manifest_sha256": "4f2577193c1b897ffbb76ee03de1b1cc0eb1a02dcf8606b49d7ab098f5c19320",
        "data_bytes": 16405,
        "data_sha256": "19f8516349e50d732079ab88a93ad54a6e3bb4aa722f0c496a072ecd77ec7efe",
        "artifact_fingerprint": "afb713f6fa817b96755a49189a77893ccd3255b599b9d41abe9c5593aa3fe771",
        "target_desc_crc32c": 0xEE552DF5,
        "manifest_crc32c": 0x9E19076D,
    },
    "linux-cuda-sm89": {
        "manifest_bytes": 992,
        "manifest_sha256": "f835a7b281886fd05bbb435e813219f439d89b184f6fca151a06b9799c862911",
        "data_bytes": 4120,
        "data_sha256": "5f813c80e80259a95189c248ea13bdd0900f8687a921fbd7f3d7466420bf50ea",
        "artifact_fingerprint": "2bb6edc7fa23bbe96fcebb21d585c8d92030f0060b825b3a1c2e6ca41ea8fa75",
        "target_desc_crc32c": 0xB529C066,
        "manifest_crc32c": 0x3CDF0FAA,
    },
}


def write_fixture(root: Path, kind: str) -> None:
    manifest, data, info = build(kind)
    expected = EXPECTED[kind]
    assert len(manifest) == expected["manifest_bytes"]
    assert hashlib.sha256(manifest).hexdigest() == expected["manifest_sha256"]
    assert len(data) == expected["data_bytes"]
    assert hashlib.sha256(data).hexdigest() == expected["data_sha256"]
    assert info["artifact_fingerprint"] == expected["artifact_fingerprint"]
    assert info["target_desc_crc32c"] == expected["target_desc_crc32c"]
    assert info["manifest_crc32c"] == expected["manifest_crc32c"]
    out = root / kind
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.coli").write_bytes(manifest)
    (out / "data-00000.coli").write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for kind in EXPECTED:
        write_fixture(args.output, kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

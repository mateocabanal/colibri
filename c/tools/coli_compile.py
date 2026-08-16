#!/usr/bin/env python3
"""Compile Hugging Face safetensors into Colibri Serving Format v1.

v1 intentionally starts with portable-v1 / codec=none.  The planner freezes
record grouping, record IDs, shard placement and source provenance before any
large output is written.  Source safetensors shards are then streamed exactly
once: the same sequential pass feeds SHA-256 and scatters tensor payload bytes
into their pre-planned .coli destinations.

The optional rANS storage codec is owned by issue #6.  `--codec auto` currently
selects none; requesting mxfp4-rans256-g0 explicitly is a named refusal rather
than silently producing a different format.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Iterable, Optional

VERSION = "colibri-csf-compiler/1"
PROFILE = "portable-v1"
ALIGNMENT = 4096
INTERNAL_ALIGN = 16
MANIFEST_HEADER = 256
SHARD_DESC = 64
RECORD_DESC = 96
STRING_DESC = 16
DATA_HEADER = 128
TENSOR_HEADER = 128
EXPERT_HEADER = 64
EXPERT_MATRIX_DESC = 128
EXPERT_DATA_OFFSET = EXPERT_HEADER + 3 * EXPERT_MATRIX_DESC
CHUNK = 4 * 1024 * 1024

MANIFEST_MAGIC = b"COLI\r\n\x1a\n"
DATA_MAGIC = b"COLIDAT\0"
TENSOR_MAGIC = b"COLITENS"
EXPERT_MAGIC = b"COLIEXPT"
SOURCE_FINGERPRINT_TAG = b"COLI-SOURCE-V1\0"

F_SOURCE_VALID = 1 << 0
R_OPTIONAL = 1 << 0
R_LOGICAL_CRC = 1 << 1

REC_TENSOR = 0x0001
REC_EXPERT = 0x0002

CODEC_NONE = 0x0000

MATH_NONE = 0x0000
MATH_F32 = 0x0001
MATH_F16 = 0x0002
MATH_BF16 = 0x0003
MATH_I8 = 0x0004
MATH_U8 = 0x0005
MATH_I16 = 0x0006
MATH_U16 = 0x0007
MATH_I32 = 0x0008
MATH_U32 = 0x0009
MATH_I64 = 0x000A
MATH_U64 = 0x000B
MATH_BOOL = 0x000C
MATH_FP8_E4M3FN = 0x0010
MATH_FP8_E5M2 = 0x0011
MATH_MXFP4_E2M1 = 0x0020
MATH_MIXED = 0xFFFE

SCALE_NONE = 0x0000
SCALE_UE8M0 = 0x0004
SCALE_MIXED = 0xFFFE

LAYOUT_CANONICAL = 0x0000
LAYOUT_MIXED = 0xFFFE

ROLE_GATE = 1
ROLE_UP = 2
ROLE_DOWN = 3

DTYPE_BYTES = {
    "F32": 4, "F16": 2, "BF16": 2,
    "I8": 1, "U8": 1, "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    "BOOL": 1, "F8_E4M3": 1, "F8_E4M3FN": 1,
    "F8_E5M2": 1, "F8_E8M0": 1,
}

DTYPE_MATH = {
    "F32": MATH_F32, "F16": MATH_F16, "BF16": MATH_BF16,
    "I8": MATH_I8, "U8": MATH_U8, "I16": MATH_I16,
    "U16": MATH_U16, "I32": MATH_I32, "U32": MATH_U32,
    "I64": MATH_I64, "U64": MATH_U64, "BOOL": MATH_BOOL,
    "F8_E4M3": MATH_FP8_E4M3FN, "F8_E4M3FN": MATH_FP8_E4M3FN,
    "F8_E5M2": MATH_FP8_E5M2,
    # CSF v1 has UE8M0 as a scale format rather than a standalone math type.
    # A named standalone UE8M0 source tensor therefore preserves its exact byte
    # representation as U8; compound expert scales use SCALE_UE8M0 below.
    "F8_E8M0": MATH_U8,
}

EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$"
)
LAYER_RE = re.compile(r"^layers\.(\d+)\.")
ROLE_SOURCE = ((ROLE_GATE, "w1"), (ROLE_UP, "w3"), (ROLE_DOWN, "w2"))

SIDECARS = (
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "generation_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt",
)


class CompileError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard_rel: str
    shard_path: Path
    offset: int
    nbytes: int


@dataclasses.dataclass
class Expert:
    layer: int
    expert: int
    parts: dict[str, Tensor]


@dataclasses.dataclass
class DestSpan:
    source: Tensor
    output_shard: int
    output_offset: int
    nbytes: int
    copied: int = 0

    @property
    def source_end(self) -> int:
        return self.source.offset + self.nbytes


@dataclasses.dataclass
class MatrixPlan:
    role: int
    weight: Tensor
    scale: Tensor
    rows: int
    columns: int
    weight_offset: int = 0
    scale_offset: int = 0
    logical_crc32c: int = 0


@dataclasses.dataclass
class RecordPlan:
    record_id: int
    kind: int
    name: Optional[str]
    layer: int
    expert: int
    source_tensor: Optional[Tensor] = None
    matrices: list[MatrixPlan] = dataclasses.field(default_factory=list)
    stored_bytes: int = 0
    decoded_bytes: int = 0
    shard: int = -1
    payload_offset: int = 0
    stored_crc32c: int = 0
    logical_crc32c: int = 0


@dataclasses.dataclass
class ShardPlan:
    shard_id: int
    records: list[RecordPlan] = dataclasses.field(default_factory=list)
    file_bytes: int = DATA_HEADER
    header_crc32c: int = 0

    @property
    def name(self) -> str:
        return f"data-{self.shard_id:05d}.coli"


@dataclasses.dataclass(frozen=True)
class InventoryHash:
    kind: int
    rel: str
    size: int
    digest: bytes


@dataclasses.dataclass
class Plan:
    model_dir: Path
    records: list[RecordPlan]
    shards: list[ShardPlan]
    source_shards: list[str]
    source_paths: dict[str, Path]
    spans_by_source: dict[str, list[DestSpan]]
    sidecars: list[str]
    index_rel: Optional[str]
    fmt_stamps: dict[str, str]
    source_payload_bytes: int
    projected_bytes: int
    padding_bytes: int


# CRC-32C / Castagnoli.  A table implementation keeps the stdlib-only fallback
# usable for large sequential chunks; a future #6 helper may replace this hot
# loop without changing any container bytes.
def _crc_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
        table.append(c & 0xFFFFFFFF)
    return tuple(table)


CRC_TABLE = _crc_table()


def crc32c_update(state: int, data: bytes | bytearray | memoryview) -> int:
    c = state
    for b in data:
        c = CRC_TABLE[(c ^ b) & 0xFF] ^ (c >> 8)
    return c & 0xFFFFFFFF


def crc32c(data: bytes | bytearray | memoryview) -> int:
    return crc32c_update(0xFFFFFFFF, data) ^ 0xFFFFFFFF


def crc32c_file_ranges(f: BinaryIO, ranges: Iterable[tuple[int, int]]) -> int:
    state = 0xFFFFFFFF
    for offset, size in ranges:
        f.seek(offset)
        left = size
        while left:
            chunk = f.read(min(CHUNK, left))
            if not chunk:
                raise CompileError("short read while computing output CRC")
            state = crc32c_update(state, chunk)
            left -= len(chunk)
    return state ^ 0xFFFFFFFF


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise CompileError(f"invalid alignment {alignment}")
    return (value + alignment - 1) & ~(alignment - 1)


def product(shape: Iterable[int]) -> int:
    n = 1
    for d in shape:
        if not isinstance(d, int) or d < 0:
            raise CompileError(f"invalid tensor dimension {d!r}")
        n *= d
    return n


def canonical_rel(root: Path, path: Path) -> str:
    root_r = root.resolve()
    path_r = path.resolve()
    try:
        rel = path_r.relative_to(root_r)
    except ValueError as exc:
        raise CompileError(f"source path escapes model root: {path}") from exc
    text = rel.as_posix()
    if not text or any(part in ("", ".", "..") for part in text.split("/")):
        raise CompileError(f"invalid canonical source path {text!r}")
    text.encode("utf-8")
    return text


def read_safetensors_header(root: Path, rel: str) -> tuple[dict[str, Tensor], dict[str, str]]:
    path = (root / rel).resolve()
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise CompileError(f"truncated safetensors prefix: {rel}")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len == 0 or header_len > 512 * 1024 * 1024:
            raise CompileError(f"invalid safetensors header length in {rel}: {header_len}")
        raw = f.read(header_len)
        if len(raw) != header_len:
            raise CompileError(f"truncated safetensors header: {rel}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise CompileError(f"invalid safetensors JSON in {rel}: {exc}") from exc
    if not isinstance(header, dict):
        raise CompileError(f"safetensors header is not an object: {rel}")
    data_base = 8 + header_len
    file_size = path.stat().st_size
    tensors: dict[str, Tensor] = {}
    metadata: dict[str, str] = {}
    for name, desc in header.items():
        if name == "__metadata__":
            if desc is None:
                continue
            if not isinstance(desc, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in desc.items()):
                raise CompileError(f"invalid __metadata__ in {rel}")
            metadata.update(desc)
            continue
        if not isinstance(name, str) or not isinstance(desc, dict):
            raise CompileError(f"invalid tensor descriptor in {rel}")
        dtype = desc.get("dtype")
        shape_raw = desc.get("shape")
        offsets = desc.get("data_offsets")
        if dtype not in DTYPE_BYTES or not isinstance(shape_raw, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise CompileError(f"unsupported/malformed tensor {name!r} in {rel}")
        shape = tuple(shape_raw)
        if len(shape) > 8:
            raise CompileError(f"tensor rank > 8 is not representable in CSF v1: {name}")
        numel = product(shape)
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise CompileError(f"invalid data offsets for {name}")
        nbytes = end - start
        expected = numel * DTYPE_BYTES[dtype]
        if nbytes != expected:
            raise CompileError(f"tensor byte size mismatch for {name}: {nbytes} != {expected}")
        absolute = data_base + start
        if absolute < data_base or absolute + nbytes > file_size:
            raise CompileError(f"tensor {name} lies outside {rel}")
        tensors[name] = Tensor(name, dtype, shape, rel, path, absolute, nbytes)
    return tensors, metadata


def discover_source(model_dir: Path) -> tuple[dict[str, Tensor], list[str], Optional[str], dict[str, str]]:
    index_candidates = sorted(model_dir.glob("*.safetensors.index.json"), key=lambda p: p.name.encode())
    if len(index_candidates) > 1:
        raise CompileError("multiple safetensors index files found")
    index_rel: Optional[str] = None
    weight_map: Optional[dict[str, str]] = None
    if index_candidates:
        index_path = index_candidates[0]
        index_rel = canonical_rel(model_dir, index_path)
        try:
            obj = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CompileError(f"invalid safetensors index: {exc}") from exc
        wm = obj.get("weight_map") if isinstance(obj, dict) else None
        if not isinstance(wm, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in wm.items()):
            raise CompileError("safetensors index lacks a valid weight_map")
        weight_map = wm
        shard_rels = sorted(set(wm.values()), key=lambda s: s.encode("utf-8"))
    else:
        paths = sorted(model_dir.glob("*.safetensors"), key=lambda p: p.name.encode())
        shard_rels = [canonical_rel(model_dir, p) for p in paths]
    if not shard_rels:
        raise CompileError("no safetensors shards found")

    all_tensors: dict[str, Tensor] = {}
    metadata_maps: list[dict[str, str]] = []
    for rel in shard_rels:
        rel_path = (model_dir / rel).resolve()
        canonical_rel(model_dir, rel_path)  # traversal check
        if not rel_path.is_file():
            raise CompileError(f"missing safetensors shard: {rel}")
        tensors, metadata = read_safetensors_header(model_dir, rel)
        for name, tensor in tensors.items():
            if name in all_tensors:
                raise CompileError(f"duplicate tensor name across shards: {name}")
            all_tensors[name] = tensor
        metadata_maps.append(metadata)
    if weight_map is not None:
        if set(weight_map) != set(all_tensors):
            missing = sorted(set(weight_map) - set(all_tensors))[:3]
            extra = sorted(set(all_tensors) - set(weight_map))[:3]
            raise CompileError(f"index/header tensor inventory mismatch; missing={missing}, extra={extra}")
        for name, rel in weight_map.items():
            if all_tensors[name].shard_rel != rel:
                raise CompileError(f"index maps {name} to {rel}, header found it in {all_tensors[name].shard_rel}")

    fmt_stamps: dict[str, str] = {}
    for metadata in metadata_maps:
        raw = metadata.get("colibri.fmt")
        if raw is None:
            continue
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise CompileError(f"invalid colibri.fmt metadata: {exc}") from exc
        if not isinstance(parsed, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
            raise CompileError("colibri.fmt must be a JSON object mapping tensor names to format strings")
        for name, value in parsed.items():
            if name in fmt_stamps and fmt_stamps[name] != value:
                raise CompileError(f"conflicting colibri.fmt stamps for {name}")
            fmt_stamps[name] = value
    for name in fmt_stamps:
        if name not in all_tensors:
            raise CompileError(f"colibri.fmt stamp names missing tensor: {name}")
        if fmt_stamps[name] == "int4-rans256-g0":
            raise CompileError("source checkpoint is already int4-rans256-g0 encoded; decode/import support belongs to #6")
    return all_tensors, shard_rels, index_rel, fmt_stamps


def classify(tensors: dict[str, Tensor], fmt_stamps: dict[str, str]) -> tuple[list[Tensor], list[Expert]]:
    expert_parts: dict[tuple[int, int], dict[str, Tensor]] = {}
    consumed: set[str] = set()
    for name, tensor in tensors.items():
        match = EXPERT_RE.match(name)
        if not match:
            continue
        layer, expert, matrix, suffix = match.groups()
        key = (int(layer), int(expert))
        part_key = f"{matrix}.{suffix}"
        parts = expert_parts.setdefault(key, {})
        if part_key in parts:
            raise CompileError(f"duplicate expert member {name}")
        parts[part_key] = tensor
        consumed.add(name)

    experts: list[Expert] = []
    required = {f"{m}.{s}" for m in ("w1", "w2", "w3") for s in ("weight", "scale")}
    for (layer, expert), parts in sorted(expert_parts.items()):
        if set(parts) != required:
            missing = sorted(required - set(parts))
            extra = sorted(set(parts) - required)
            raise CompileError(f"incomplete expert ({layer},{expert}); missing={missing}, extra={extra}")
        for _role, matrix in ROLE_SOURCE:
            weight = parts[f"{matrix}.weight"]
            scale = parts[f"{matrix}.scale"]
            if weight.dtype != "I8" or scale.dtype != "F8_E8M0" or len(weight.shape) != 2 or len(scale.shape) != 2:
                raise CompileError(f"invalid MXFP4 dtype/rank for {weight.name}")
            if not weight.shape[0] or not weight.shape[1] or weight.shape[0] != scale.shape[0] or scale.shape[1] * 32 != weight.shape[1] * 2:
                raise CompileError(f"invalid MXFP4 geometry for {weight.name}")
            stamp = fmt_stamps.get(weight.name)
            if stamp is not None and stamp not in ("mxfp4", "mxfp4-e2m1", "mxfp4-e2m1-ue8m0"):
                raise CompileError(f"expert weight {weight.name} has conflicting format stamp {stamp!r}")
        experts.append(Expert(layer, expert, parts))

    ordinary = [t for name, t in tensors.items() if name not in consumed]
    for tensor in ordinary:
        if tensor.dtype not in DTYPE_MATH:
            raise CompileError(f"tensor dtype is not representable in portable-v1: {tensor.name} ({tensor.dtype})")
    return ordinary, experts


def tensor_sort_key(t: Tensor) -> tuple[int, int, bytes]:
    m = LAYER_RE.match(t.name)
    if m:
        return (1, int(m.group(1)), t.name.encode("utf-8"))
    return (0, -1, t.name.encode("utf-8"))


def build_records(ordinary: list[Tensor], experts: list[Expert]) -> list[RecordPlan]:
    records: list[RecordPlan] = []
    rid = 1
    for tensor in sorted(ordinary, key=tensor_sort_key):
        m = LAYER_RE.match(tensor.name)
        layer = int(m.group(1)) if m else -1
        records.append(RecordPlan(
            rid, REC_TENSOR, tensor.name, layer, -1,
            source_tensor=tensor,
            stored_bytes=TENSOR_HEADER + tensor.nbytes,
            decoded_bytes=tensor.nbytes,
        ))
        rid += 1
    for expert in sorted(experts, key=lambda e: (e.layer, e.expert)):
        matrices: list[MatrixPlan] = []
        cursor = EXPERT_DATA_OFFSET
        decoded = 0
        for role, source_name in ROLE_SOURCE:
            weight = expert.parts[f"{source_name}.weight"]
            scale = expert.parts[f"{source_name}.scale"]
            cursor = align_up(cursor, INTERNAL_ALIGN)
            weight_offset = cursor
            cursor += weight.nbytes
            cursor = align_up(cursor, INTERNAL_ALIGN)
            scale_offset = cursor
            cursor += scale.nbytes
            matrix = MatrixPlan(role, weight, scale, weight.shape[0], weight.shape[1] * 2,
                                weight_offset=weight_offset, scale_offset=scale_offset)
            matrices.append(matrix)
            decoded += weight.nbytes + scale.nbytes
        records.append(RecordPlan(
            rid, REC_EXPERT, None, expert.layer, expert.expert,
            matrices=matrices, stored_bytes=cursor, decoded_bytes=decoded,
        ))
        rid += 1
    return records


def plan_shards(records: list[RecordPlan], shard_target: int) -> list[ShardPlan]:
    if shard_target < ALIGNMENT + 1:
        raise CompileError(f"shard target is too small: {shard_target}")
    shards: list[ShardPlan] = []
    current = ShardPlan(0)
    cursor = align_up(DATA_HEADER, ALIGNMENT)
    padding = 0
    for record in records:
        offset = align_up(cursor, ALIGNMENT)
        if current.records and offset + record.stored_bytes > shard_target:
            current.file_bytes = cursor
            shards.append(current)
            current = ShardPlan(len(shards))
            cursor = align_up(DATA_HEADER, ALIGNMENT)
            offset = cursor
        padding += offset - cursor
        record.shard = current.shard_id
        record.payload_offset = offset
        current.records.append(record)
        cursor = offset + record.stored_bytes
    if current.records:
        current.file_bytes = cursor
        shards.append(current)
    if not shards:
        raise CompileError("source checkpoint contains no records")
    return shards


def sidecar_inventory(model_dir: Path) -> list[str]:
    return [name for name in SIDECARS if (model_dir / name).is_file()]


def build_plan(model_dir: Path, shard_target: int) -> Plan:
    tensors, source_shards, index_rel, stamps = discover_source(model_dir)
    ordinary, experts = classify(tensors, stamps)
    records = build_records(ordinary, experts)
    shards = plan_shards(records, shard_target)
    spans_by_source: dict[str, list[DestSpan]] = {rel: [] for rel in source_shards}
    for record in records:
        if record.kind == REC_TENSOR:
            assert record.source_tensor is not None
            t = record.source_tensor
            spans_by_source[t.shard_rel].append(DestSpan(t, record.shard, record.payload_offset + TENSOR_HEADER, t.nbytes))
        else:
            for matrix in record.matrices:
                spans_by_source[matrix.weight.shard_rel].append(
                    DestSpan(matrix.weight, record.shard, record.payload_offset + matrix.weight_offset, matrix.weight.nbytes))
                spans_by_source[matrix.scale.shard_rel].append(
                    DestSpan(matrix.scale, record.shard, record.payload_offset + matrix.scale_offset, matrix.scale.nbytes))
    for spans in spans_by_source.values():
        spans.sort(key=lambda s: (s.source.offset, s.output_shard, s.output_offset))
    source_paths = {rel: (model_dir / rel).resolve() for rel in source_shards}
    source_payload = sum(t.nbytes for t in tensors.values())
    projected = sum(s.file_bytes for s in shards)
    used_record = sum(r.stored_bytes for r in records)
    padding = projected - len(shards) * DATA_HEADER - used_record
    return Plan(model_dir, records, shards, source_shards, source_paths,
                spans_by_source, sidecar_inventory(model_dir), index_rel, stamps,
                source_payload, projected, padding)


def write_zeros(f: BinaryIO, size: int) -> None:
    if size <= 0:
        return
    f.seek(size - 1)
    f.write(b"\0")


def tensor_header(record: RecordPlan, logical_crc: int = 0) -> bytes:
    t = record.source_tensor
    assert t is not None
    h = bytearray(TENSOR_HEADER)
    h[:8] = TENSOR_MAGIC
    struct.pack_into("<HHI", h, 8, 1, 0, TENSOR_HEADER)
    struct.pack_into("<HH", h, 16, len(t.shape), 0)
    for i, d in enumerate(t.shape):
        struct.pack_into("<Q", h, 32 + i * 8, d)
    struct.pack_into("<QQQ", h, 96, TENSOR_HEADER, t.nbytes, t.nbytes)
    struct.pack_into("<I", h, 120, logical_crc)
    return bytes(h)


def expert_header(record: RecordPlan) -> bytes:
    h = bytearray(EXPERT_DATA_OFFSET)
    h[:8] = EXPERT_MAGIC
    struct.pack_into("<HHI", h, 8, 1, 0, EXPERT_HEADER)
    struct.pack_into("<iiHHI", h, 16, record.layer, record.expert, 3, 0, EXPERT_MATRIX_DESC)
    struct.pack_into("<QQQ", h, 32, EXPERT_HEADER, EXPERT_DATA_OFFSET, record.decoded_bytes)
    for i, matrix in enumerate(record.matrices):
        off = EXPERT_HEADER + i * EXPERT_MATRIX_DESC
        struct.pack_into("<HHHHHHHH", h, off,
                         matrix.role, 0, MATH_MXFP4_E2M1, SCALE_UE8M0,
                         CODEC_NONE, CODEC_NONE, LAYOUT_CANONICAL, 0)
        struct.pack_into("<QQ", h, off + 16, matrix.rows, matrix.columns)
        struct.pack_into("<II", h, off + 32, 1, 32)
        struct.pack_into("<II", h, off + 40, 0, 0)
        struct.pack_into("<QQQ", h, off + 48,
                         matrix.weight_offset, matrix.weight.nbytes, matrix.weight.nbytes)
        struct.pack_into("<QQQ", h, off + 72,
                         matrix.scale_offset, matrix.scale.nbytes, matrix.scale.nbytes)
        struct.pack_into("<I", h, off + 96, matrix.logical_crc32c)
        struct.pack_into("<I", h, off + 104, 0)
    return bytes(h)


def open_output_shards(temp_dir: Path, plan: Plan) -> list[BinaryIO]:
    handles: list[BinaryIO] = []
    try:
        for shard in plan.shards:
            path = temp_dir / shard.name
            f = path.open("w+b")
            write_zeros(f, shard.file_bytes)
            handles.append(f)
        for record in plan.records:
            f = handles[record.shard]
            f.seek(record.payload_offset)
            f.write(tensor_header(record) if record.kind == REC_TENSOR else expert_header(record))
        return handles
    except Exception:
        for f in handles:
            f.close()
        raise


def stream_source_shards(plan: Plan, outputs: list[BinaryIO]) -> list[InventoryHash]:
    hashes: list[InventoryHash] = []
    for rel in plan.source_shards:
        path = plan.source_paths[rel]
        spans = plan.spans_by_source[rel]
        hasher = hashlib.sha256()
        size = path.stat().st_size
        index = 0
        pos = 0
        with path.open("rb") as source:
            while True:
                chunk = source.read(CHUNK)
                if not chunk:
                    break
                hasher.update(chunk)
                chunk_start, chunk_end = pos, pos + len(chunk)
                while index < len(spans) and spans[index].source_end <= chunk_start:
                    if spans[index].copied != spans[index].nbytes:
                        raise CompileError(f"failed to copy tensor {spans[index].source.name}")
                    index += 1
                j = index
                while j < len(spans) and spans[j].source.offset < chunk_end:
                    span = spans[j]
                    ov_start = max(chunk_start, span.source.offset)
                    ov_end = min(chunk_end, span.source_end)
                    if ov_start < ov_end:
                        src_at = ov_start - chunk_start
                        dst_at = span.output_offset + (ov_start - span.source.offset)
                        outputs[span.output_shard].seek(dst_at)
                        outputs[span.output_shard].write(chunk[src_at:src_at + (ov_end - ov_start)])
                        span.copied += ov_end - ov_start
                    j += 1
                pos = chunk_end
        if pos != size:
            raise CompileError(f"source shard changed size while compiling: {rel}")
        while index < len(spans):
            if spans[index].copied != spans[index].nbytes:
                raise CompileError(f"failed to copy tensor {spans[index].source.name}")
            index += 1
        hashes.append(InventoryHash(1, rel, size, hasher.digest()))
    return hashes


def hash_file(path: Path, kind: int, rel: str) -> InventoryHash:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return InventoryHash(kind, rel, size, h.digest())


def source_fingerprint(entries: list[InventoryHash]) -> bytes:
    ordered = sorted(entries, key=lambda e: e.rel.encode("utf-8"))
    h = hashlib.sha256()
    h.update(SOURCE_FINGERPRINT_TAG)
    h.update(struct.pack("<I", len(ordered)))
    seen: set[bytes] = set()
    for entry in ordered:
        raw = entry.rel.encode("utf-8")
        if raw in seen:
            raise CompileError(f"duplicate source fingerprint path: {entry.rel}")
        seen.add(raw)
        h.update(struct.pack("<BI", entry.kind, len(raw)))
        h.update(raw)
        h.update(struct.pack("<Q", entry.size))
        h.update(entry.digest)
    return h.digest()


def copy_and_hash_sidecars(temp_dir: Path, plan: Plan) -> list[InventoryHash]:
    entries: list[InventoryHash] = []
    for rel in plan.sidecars:
        src = plan.model_dir / rel
        dst = temp_dir / rel
        h = hashlib.sha256()
        size = 0
        with src.open("rb") as inp, dst.open("wb") as out:
            while True:
                chunk = inp.read(CHUNK)
                if not chunk:
                    break
                h.update(chunk)
                out.write(chunk)
                size += len(chunk)
            out.flush()
            os.fsync(out.fileno())
        entries.append(InventoryHash(3, rel, size, h.digest()))
    if plan.index_rel:
        path = plan.model_dir / plan.index_rel
        entries.append(hash_file(path, 2, plan.index_rel))
    return entries


def finalize_records(plan: Plan, outputs: list[BinaryIO]) -> None:
    for record in plan.records:
        f = outputs[record.shard]
        base = record.payload_offset
        if record.kind == REC_TENSOR:
            record.logical_crc32c = crc32c_file_ranges(f, [(base + TENSOR_HEADER, record.decoded_bytes)])
            f.seek(base + 120)
            f.write(struct.pack("<I", record.logical_crc32c))
        else:
            outer_ranges: list[tuple[int, int]] = []
            for i, matrix in enumerate(record.matrices):
                ranges = [
                    (base + matrix.weight_offset, matrix.weight.nbytes),
                    (base + matrix.scale_offset, matrix.scale.nbytes),
                ]
                matrix.logical_crc32c = crc32c_file_ranges(f, ranges)
                f.seek(base + EXPERT_HEADER + i * EXPERT_MATRIX_DESC + 96)
                f.write(struct.pack("<I", matrix.logical_crc32c))
                outer_ranges.extend(ranges)
            record.logical_crc32c = crc32c_file_ranges(f, outer_ranges)
        f.flush()
        record.stored_crc32c = crc32c_file_ranges(f, [(base, record.stored_bytes)])


def data_header(shard: ShardPlan, fingerprint: bytes) -> bytes:
    h = bytearray(DATA_HEADER)
    h[:8] = DATA_MAGIC
    struct.pack_into("<HHI", h, 8, 1, 0, DATA_HEADER)
    struct.pack_into("<II", h, 16, 0, shard.shard_id)
    struct.pack_into("<I", h, 24, ALIGNMENT)
    struct.pack_into("<Q", h, 32, shard.file_bytes)
    h[40:72] = fingerprint
    struct.pack_into("<I", h, 72, 0)
    crc = crc32c(h)
    struct.pack_into("<I", h, 72, crc)
    shard.header_crc32c = crc
    return bytes(h)


def string_table(strings: list[str]) -> bytes:
    raw_strings = [s.encode("utf-8") for s in strings]
    desc_bytes = len(strings) * STRING_DESC
    cursor = desc_bytes
    out = bytearray(desc_bytes + sum(len(s) for s in raw_strings))
    for i, raw in enumerate(raw_strings):
        if b"\0" in raw:
            raise CompileError("manifest string contains NUL")
        struct.pack_into("<QII", out, i * STRING_DESC, cursor, len(raw), 0)
        out[cursor:cursor + len(raw)] = raw
        cursor += len(raw)
    out.extend(b"\0" * (align_up(len(out), INTERNAL_ALIGN) - len(out)))
    return bytes(out)


def make_manifest(plan: Plan, fingerprint: bytes) -> bytes:
    strings: list[str] = []
    string_id: dict[str, int] = {}

    def intern(s: str) -> int:
        if s not in string_id:
            string_id[s] = len(strings)
            strings.append(s)
        return string_id[s]

    for shard in plan.shards:
        intern(shard.name)
    for record in plan.records:
        if record.name is not None:
            intern(record.name)
    profile_id = intern(PROFILE)
    compiler_id = intern(VERSION)

    shard_off = MANIFEST_HEADER
    shard_bytes = len(plan.shards) * SHARD_DESC
    record_off = align_up(shard_off + shard_bytes, INTERNAL_ALIGN)
    record_bytes = len(plan.records) * RECORD_DESC
    strings_blob = string_table(strings)
    string_off = align_up(record_off + record_bytes, INTERNAL_ALIGN)
    manifest_bytes = string_off + len(strings_blob)
    out = bytearray(manifest_bytes)
    out[:8] = MANIFEST_MAGIC
    struct.pack_into("<HHI", out, 8, 1, 0, MANIFEST_HEADER)
    struct.pack_into("<I", out, 16, F_SOURCE_VALID)
    struct.pack_into("<I", out, 20, 0x01020304)
    struct.pack_into("<II", out, 24, ALIGNMENT, len(strings))
    struct.pack_into("<Q", out, 32, len(plan.records))
    struct.pack_into("<I", out, 40, len(plan.shards))
    struct.pack_into("<QQ", out, 48, shard_off, shard_bytes)
    struct.pack_into("<QQ", out, 64, record_off, record_bytes)
    struct.pack_into("<QQ", out, 80, string_off, len(strings_blob))
    out[112:144] = fingerprint
    struct.pack_into("<II", out, 148, profile_id, compiler_id)

    for i, shard in enumerate(plan.shards):
        off = shard_off + i * SHARD_DESC
        struct.pack_into("<III", out, off, shard.shard_id, 0, intern(shard.name))
        struct.pack_into("<Q", out, off + 16, shard.file_bytes)
        struct.pack_into("<I", out, off + 24, shard.header_crc32c)

    for i, record in enumerate(plan.records):
        off = record_off + i * RECORD_DESC
        if record.kind == REC_TENSOR:
            assert record.source_tensor is not None
            math_format = DTYPE_MATH[record.source_tensor.dtype]
            scale_format = SCALE_NONE
            layout = LAYOUT_CANONICAL
            name_id = intern(record.name or "")
        else:
            math_format = MATH_MIXED
            scale_format = SCALE_MIXED
            layout = LAYOUT_MIXED
            name_id = 0xFFFFFFFF
        struct.pack_into("<QHHHHHHIIiiI", out, off,
                         record.record_id, record.kind, CODEC_NONE,
                         math_format, scale_format, layout, R_LOGICAL_CRC,
                         record.shard, name_id, record.layer, record.expert, 0)
        struct.pack_into("<QQQII", out, off + 40,
                         record.payload_offset, record.stored_bytes, record.decoded_bytes,
                         record.stored_crc32c, record.logical_crc32c)
        struct.pack_into("<I", out, off + 72, 0)
    out[string_off:string_off + len(strings_blob)] = strings_blob
    struct.pack_into("<I", out, 144, 0)
    struct.pack_into("<I", out, 144, crc32c(out))
    return bytes(out)


def fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def finalize_output(temp_dir: Path, output: Path, force: bool) -> None:
    parent = output.parent
    backup = parent / f".{output.name}.old.{os.getpid()}"
    if output.exists():
        if not force:
            raise CompileError(f"output already exists (use --force): {output}")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(output, backup)
        try:
            os.replace(temp_dir, output)
        except Exception:
            os.replace(backup, output)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(temp_dir, output)
    fsync_dir(parent)


def report(plan: Plan, *, fingerprint: Optional[bytes] = None, stored_bytes: Optional[int] = None) -> None:
    expert_count = sum(r.kind == REC_EXPERT for r in plan.records)
    tensor_count = sum(r.kind == REC_TENSOR for r in plan.records)
    print(f"profile={PROFILE}")
    print(f"source_bytes={sum(p.stat().st_size for p in plan.source_paths.values())}")
    print(f"source_tensor_payload_bytes={plan.source_payload_bytes}")
    print(f"tensor_records={tensor_count}")
    print(f"expert_records={expert_count}")
    print(f"record_count={len(plan.records)}")
    print(f"shard_count={len(plan.shards)}")
    print(f"record_alignment={ALIGNMENT}")
    print(f"projected_data_bytes={plan.projected_bytes}")
    print(f"padding_bytes={plan.padding_bytes}")
    print(f"codec=none")
    if stored_bytes is not None:
        print(f"stored_package_data_bytes={stored_bytes}")
    if fingerprint is not None:
        print(f"source_fingerprint={fingerprint.hex()}")


def compile_model(model_dir: Path, output: Path, *, shard_size_bytes: int,
                  codec: str = "none", verify: bool = False,
                  force: bool = False, dry_run: bool = False) -> Plan:
    model_dir = model_dir.resolve()
    output = output.resolve()
    if not model_dir.is_dir():
        raise CompileError(f"model directory does not exist: {model_dir}")
    if output == model_dir or model_dir in output.parents:
        raise CompileError("output must not be inside the source model directory")
    if codec not in ("none", "auto", "mxfp4-rans256-g0"):
        raise CompileError(f"unsupported codec option {codec!r}")
    if codec == "mxfp4-rans256-g0":
        raise CompileError("mxfp4-rans256-g0 compiler integration requires issue #6")
    plan = build_plan(model_dir, shard_size_bytes)
    if dry_run:
        report(plan)
        return plan

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(str(output) + f".tmp.{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    outputs: list[BinaryIO] = []
    installed = False
    try:
        outputs = open_output_shards(temp_dir, plan)
        source_hashes = stream_source_shards(plan, outputs)
        auxiliary_hashes = copy_and_hash_sidecars(temp_dir, plan)
        fingerprint = source_fingerprint(source_hashes + auxiliary_hashes)
        finalize_records(plan, outputs)
        for shard, f in zip(plan.shards, outputs):
            f.seek(0)
            f.write(data_header(shard, fingerprint))
            f.flush()
            os.fsync(f.fileno())
            f.close()
        outputs.clear()
        manifest = make_manifest(plan, fingerprint)
        manifest_path = temp_dir / "manifest.coli"
        with manifest_path.open("wb") as f:
            f.write(manifest)
            f.flush()
            os.fsync(f.fileno())
        fsync_dir(temp_dir)
        if verify:
            verify_package(temp_dir, plan, fingerprint)
        finalize_output(temp_dir, output, force)
        installed = True
        report(plan, fingerprint=fingerprint,
               stored_bytes=sum(s.file_bytes for s in plan.shards) + len(manifest))
        return plan
    finally:
        for f in outputs:
            try:
                f.close()
            except Exception:
                pass
        if not installed and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_strings(manifest: bytes, offset: int, size: int, count: int) -> list[str]:
    table = memoryview(manifest)[offset:offset + size]
    desc_bytes = count * STRING_DESC
    if len(table) != size or desc_bytes > size:
        raise CompileError("verification: invalid string table")
    strings = []
    for i in range(count):
        data_off, data_len, flags = struct.unpack_from("<QII", table, i * STRING_DESC)
        if flags or data_off < desc_bytes or data_off + data_len > size:
            raise CompileError("verification: invalid string descriptor")
        strings.append(bytes(table[data_off:data_off + data_len]).decode("utf-8"))
    return strings


def compare_range(source: Tensor, package_path: Path, package_offset: int) -> None:
    with source.shard_path.open("rb") as src, package_path.open("rb") as dst:
        src.seek(source.offset)
        dst.seek(package_offset)
        left = source.nbytes
        while left:
            n = min(CHUNK, left)
            a, b = src.read(n), dst.read(n)
            if len(a) != n or a != b:
                raise CompileError(f"verification: logical bytes differ for {source.name}")
            left -= n


def verify_package(package: Path, plan: Plan, expected_fingerprint: bytes) -> None:
    manifest = (package / "manifest.coli").read_bytes()
    if len(manifest) < MANIFEST_HEADER or manifest[:8] != MANIFEST_MAGIC:
        raise CompileError("verification: bad manifest")
    copy = bytearray(manifest)
    expected_crc = struct.unpack_from("<I", copy, 144)[0]
    struct.pack_into("<I", copy, 144, 0)
    if crc32c(copy) != expected_crc:
        raise CompileError("verification: manifest CRC mismatch")
    if manifest[112:144] != expected_fingerprint:
        raise CompileError("verification: source fingerprint mismatch")
    string_count = struct.unpack_from("<I", manifest, 28)[0]
    record_count = struct.unpack_from("<Q", manifest, 32)[0]
    shard_count = struct.unpack_from("<I", manifest, 40)[0]
    shard_off, shard_bytes = struct.unpack_from("<QQ", manifest, 48)
    record_off, record_bytes = struct.unpack_from("<QQ", manifest, 64)
    string_off, string_bytes = struct.unpack_from("<QQ", manifest, 80)
    if record_count != len(plan.records) or shard_count != len(plan.shards) or shard_bytes != shard_count * SHARD_DESC or record_bytes != record_count * RECORD_DESC:
        raise CompileError("verification: manifest counts disagree with plan")
    strings = parse_strings(manifest, string_off, string_bytes, string_count)
    for i, shard in enumerate(plan.shards):
        desc = shard_off + i * SHARD_DESC
        shard_id, _flags, name_id = struct.unpack_from("<III", manifest, desc)
        file_bytes = struct.unpack_from("<Q", manifest, desc + 16)[0]
        path = package / strings[name_id]
        raw_header = path.read_bytes()[:DATA_HEADER]
        if shard_id != i or file_bytes != path.stat().st_size or raw_header[:8] != DATA_MAGIC or raw_header[40:72] != expected_fingerprint:
            raise CompileError("verification: shard header/descriptor mismatch")
        header_copy = bytearray(raw_header)
        expected = struct.unpack_from("<I", header_copy, 72)[0]
        struct.pack_into("<I", header_copy, 72, 0)
        if crc32c(header_copy) != expected:
            raise CompileError("verification: shard header CRC mismatch")
    for i, record in enumerate(plan.records):
        desc = record_off + i * RECORD_DESC
        rid, kind = struct.unpack_from("<QH", manifest, desc)
        shard = struct.unpack_from("<I", manifest, desc + 20)[0]
        payload, stored, decoded = struct.unpack_from("<QQQ", manifest, desc + 40)
        stored_crc, logical_crc = struct.unpack_from("<II", manifest, desc + 64)
        if rid != record.record_id or kind != record.kind or shard != record.shard or payload != record.payload_offset or stored != record.stored_bytes or decoded != record.decoded_bytes:
            raise CompileError("verification: record descriptor differs from plan")
        shard_path = package / plan.shards[shard].name
        with shard_path.open("rb") as f:
            if crc32c_file_ranges(f, [(payload, stored)]) != stored_crc:
                raise CompileError(f"verification: stored CRC mismatch for record {rid}")
        if kind == REC_TENSOR:
            t = record.source_tensor
            assert t is not None
            compare_range(t, shard_path, payload + TENSOR_HEADER)
            with shard_path.open("rb") as f:
                if crc32c_file_ranges(f, [(payload + TENSOR_HEADER, t.nbytes)]) != logical_crc:
                    raise CompileError(f"verification: tensor logical CRC mismatch for {t.name}")
        else:
            ranges = []
            for matrix in record.matrices:
                compare_range(matrix.weight, shard_path, payload + matrix.weight_offset)
                compare_range(matrix.scale, shard_path, payload + matrix.scale_offset)
                ranges.extend(((payload + matrix.weight_offset, matrix.weight.nbytes),
                               (payload + matrix.scale_offset, matrix.scale.nbytes)))
            with shard_path.open("rb") as f:
                if crc32c_file_ranges(f, ranges) != logical_crc:
                    raise CompileError(f"verification: expert logical CRC mismatch for ({record.layer},{record.expert})")


def parse_size_gb(value: str) -> int:
    try:
        f = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not (f > 0):
        raise argparse.ArgumentTypeError("must be > 0")
    size = int(f * (1024 ** 3))
    if size < ALIGNMENT + 1:
        raise argparse.ArgumentTypeError("shard size is too small")
    return size


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", default=PROFILE, choices=[PROFILE])
    parser.add_argument("--shard-size-gb", type=parse_size_gb, default=4 * 1024 ** 3,
                        metavar="N", help="target data-shard size in GiB (default: 4)")
    parser.add_argument("--codec", choices=["none", "auto", "mxfp4-rans256-g0"], default="none")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or Path(str(args.model_dir) + ".coli")
    try:
        compile_model(args.model_dir, output, shard_size_bytes=args.shard_size_gb,
                      codec=args.codec, verify=args.verify, force=args.force,
                      dry_run=args.dry_run)
    except CompileError as exc:
        print(f"coli_compile: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Umbrella #191 — automatic hardware-aware mixed-format compiler (sprint results, 2026-08-27)

Branch: `feat/colic-auto-planner` (based on PR #189 tip + main). Development on the GPD,
builds/tests on Yigal's PC (Ryzen 5 5600X / 32 GB / GTX 1080 8 GB, Windows MSVC toolchain,
worktree `C:\colibri\colic-sprint`). ChatGPT (chatgpt-web-20260827094212-5724b1) advised
the design (Q1-Q4: state-as-resources, per-rep capability descriptors, replay pins, DP4A
fits-VRAM rule).

## What shipped (all committed + pushed to feat/colic-auto-planner)

| Issue | Module | Status |
|---|---|---|
| #192 machine probe | `plan/machine.rs` + `colic probe [--json]` + `--machine-profile` override | DONE — verified live on the 1080 box: AVX2, 32 GB via GlobalMemoryStatusEx FFI, GTX 1080 = cc6.1/dp4a/tensor=false |
| #193 physical IR | `plan/ir.rs` (MathFormat/PhysicalLayout/BackendKind/Placement/TensorPlan/Decision/PhysicalPlan) | DONE — mixed CPU+CUDA plans valid; unsupported combos fail at plan time |
| #194 memory planner | `plan/memory.rs` (StateGeometry from real config: hybrid KV only for full-attn layers, GDN fixed state, QSA indexer capped at context) | DONE — real Qwen3.8-Next config drives it |
| #195 cost model | `plan/cost.rs` (objectives quality|balanced|throughput|latency|minimum-size, rule tables v1, chosen+rejected+reason) | DONE — DP4A gated on expert-set-fits-VRAM (measured 2243 vs 1931 ms/tok on this box) |
| #199 placement | `plan/placement.rs` (PLE resident first, VRAM cache for CUDA reps, RAM cache, pageable backing) | DONE — bug found on real model: pageable PLE must not consume the RAM budget |
| #201 plan CLI + replay | `plan/plan_cli.rs` + `plan/planner.rs` + `--plan` replay validation (fingerprint pin + expert-math match) | DONE — plan/compile/verify verified end-to-end on a colic-shaped tiny fixture on the box |
| Qwen4-Exp frontend | `model/qwen4_exp.rs` (bare + text_config configs; F8_E4M3 separate gate/up/down + BF16 weight_scale_inv block scales; final norm optional — hyper-connection mixer replaces it) | DONE — real checkpoint plans cleanly (26,649 records) |
| FP8 -> INT4 requant | `quant/fp8.rs` (E4M3 LUT, const bit-exact pow2) + `quant/int4.rs` FP8 block-scale path | DONE — tiny-fixture roundtrip test + real-shape math |
| Stack fix | `main.rs` 64 MiB worker thread (Windows 1 MiB main stack) | DONE — replaces editbin |
| #200 calibration | `c/tools/calibrate.py` (deterministic nRMSE/cosine/outlier per family, fingerprint cache) + `plan/calibration.rs` veto (quality floors, fallback candidate) | DONE — floors: quality 0.02, balanced 0.15, throughput 0.30, size 0.40 |
| #198 Pascal DP4A | `c/tests/test_mxfp4_dp4a.cu` (W4A8 DP4A prototype + float-decode reference + bench) | VERDICT: CORRECT (max abs err 0.033 vs canonical) but 0.52x SLOWER (0.071 vs 0.037 ms, gate 640x2560, 200 iters) — perf gate NOT met; planner keeps CPU INT4 |

## The 4-bit deliverable (final ask)

`colic plan Qwen3.8-Flash-Next-FP8 --objective balanced --context 32768` on the box
chooses: routed experts = INT4-G32 canonical CPU (30.51 GiB, 25,088 records), everything
else exact BF16/F8 (PLE n-gram 47.75 GiB pageable — does not fit the 32 GB RAM cache),
total projected package 88.41 GiB vs 160.8 GiB source. `colic compile --plan` replays the
fingerprint-pinned plan and emits the package; `--verify` checks every record CRC.

## Known gaps / follow-ups (next sprints)

- **C-runtime consumption of INT4-G32 expert records in qwen_moe/qwen4 engine**: the
  compiler emits the exact COLIEXPT descriptor contract (MATH_FORMAT_INT4_GROUPED,
  SCALE_FORMAT_F32, group 32) the C kernel `matmul_i4_grouped` consumes, and colic's
  verify accepts it — but the engine loader wiring is on PR #190's branch (qwen4
  engine), not this branch. Engine decode smoke = follow-up on #190.
- **`--target auto` inline resolution**: wired (maps planner outcome -> profile+quant);
  end-to-end compile under `--target auto` untested on a real model (compile used the
  explicit `--plan` replay path).
- **Rows16**: deliberately NOT chosen in v1 — canonical INT4-G32 is what the runtime
  kernel consumes today; ROWS16 needs the #197 canonical-vs-rows16 benchmark first.
- **Multi-GPU VRAM split**: v1 uses the first CUDA GPU; per-GPU splitting is a later
  planner iteration.
- **PLE FP8 (validated NVFP4)**: calibrator supports the family; NVFP4 requant of the
  n-gram table is a follow-up once the runtime has an FP8 PLE loader (engine-side).

## Determinism + reproducibility

- BTreeMap ordering everywhere; plan JSON round-trips; `source_fingerprint` (SHA-256 of
  all shards + config) pins replay; planner/cost-model/schema versions in the manifest.
- Replay validates, never re-plans; mismatched fingerprint or expert math fails BEFORE
  emission (verified: replay with wrong --quant rejected pre-emission on the box).

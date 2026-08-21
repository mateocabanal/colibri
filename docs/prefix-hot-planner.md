# Prefix RAM-hot residency and the global planner

Issue: #124

This migration deliberately separates **cache-state correctness** from
**resource valuation**.

## Landed/proven engine state

Qwen and DeepSeek V4 already know how to capture and restore complete resumable
prefix state. Those native snapshot representations stay engine-owned:

- Qwen: full-attention KV + Gated DeltaNet recurrence + causal-conv state;
- V4: window/compressed attention + compressor/indexer state.

The global resource layer must not duplicate or reinterpret those bytes.

## Planner boundary

`prefix_hot_resource.h` defines the common inventory/benefit vocabulary.
Engine adapters expose resident entries as:

```text
id
resident bytes
token count
observed hit count
recency epoch
active references
```

A policy converts those observations plus measured avoided work into the same
`ColiResourceCandidate` representation used by dense tensors and persistent
experts.

The allocator remains model-neutral and does not invent an exposed-time score.
Candidates competing in one selection must use a common reuse horizon.

## Budget application

A planner-selected prefix budget is an **effective sub-budget**, not a second
reservation on top of the user's memory envelope.

- Qwen can contract/re-expand its request-local process cache within the resolved
  policy cap.
- V4 can reclaim only unreferenced entries. Ref-pinned restores and in-flight
  snapshot reservations form a temporary mandatory floor for that replan.

Cache admission failure always degrades to SSD/cold-prefill behavior.

## Remaining handoff

The current engines still reserve/partition prefix RAM before enough information
exists to value a prospective prefix:

- V4 subtracts its configured process-prefix cap before the dense/expert planner
  runs.
- Qwen subtracts its serving prefix budget before deriving the legacy expert
  cache cap.

Removing either partition without a replacement would be wrong: it would either
make prefix memory additive to the user's envelope or set the prefix budget to
zero before any entry can exist and therefore prevent the planner from learning
about useful prefixes.

The next slice must add **prospective admission** at the canonical end-of-prefill
boundary:

1. compute exact snapshot resident bytes before allocation;
2. attach request-observed avoided-work telemetry (prefer actual exposed time or
   actual streamed bytes; do not synthesize nanoseconds);
3. ask the common planner to re-evaluate dense/expert/prefix optional residency;
4. lower competing future-admission budgets safely;
5. capture/retain the prefix only when the resulting plan admits it;
6. keep referenced/borrowed resources mandatory until their leases end.

Only after this path exists should the V4 startup prefix reserve and Qwen's
`RAM_GB - prefix_budget` partition be removed.

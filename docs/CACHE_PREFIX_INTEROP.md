# Prefix Cache Interoperability Notes

This note defines how the existing cache implementations should coexist while the V4 cache work is added.

## Do not unify state bytes

The repository has multiple correct cache representations because the engines have different runtime state:

- `kv_prefix.h`: token record describing live in-place KV/recurrent state;
- Qwen prefix cache: Qwen-specific prompt/KV reuse;
- V4 RAM prefix cache: native attention/compressor/indexer snapshots;
- V4 prefix disk cache: persisted V4 snapshot state;
- `kv_persist.h`: colibri.c-specific per-position `.coli_kv` persistence.

These should not be converted into one universal serialized KV layout. Sharing state bytes across engines would couple unrelated internal ABIs and create silent-staleness risk.

## What can be shared

The useful common contract is around **identity and lifecycle**:

1. Cache lookup is an optimization. Any validation or I/O failure becomes a miss.
2. Exact token identity is required; no fuzzy prefix reuse.
3. Model/package identity and state ABI must be part of persistent-cache validation.
4. State and its token record must be updated/cleared together.
5. A cache entry may only claim positions actually consumed by the state it stores.
6. Reuse that requires rewinding state is a miss unless the engine has an explicit rewind/snapshot mechanism.
7. Non-token inputs must either participate in the cache key or taint/disable reuse.
8. Persistent writes must become visible only after the complete entry is durable and validated.
9. Memory/disk budgets are hard resource envelopes, not best-effort accounting after allocation.

A later cleanup may extract small shared helpers for token digests, model fingerprints, cache-key versioning, checksums, and stats. That is only worthwhile after the V4 integration proves the exact common surface.

## V4 integration rule

For V4, `coli_v4_prefix_cache` remains the source of truth for reusable RAM snapshots and `coli_v4_prefix_disk` remains the source of truth for persisted snapshots.

`kv_prefix.h` is optional bookkeeping only. Use it if the V4 request loop needs a compact exact record of tokens actually fed to the live session; do not use it to bypass or duplicate V4 snapshot validation.

Lookup order should be:

1. exact reusable live/session state, if the V4 session owns one and can prove token identity;
2. V4 process-RAM prefix snapshot;
3. V4 SSD prefix snapshot;
4. cold reset + full prefill.

Only tiers that actually exist in the implementation should be enabled; the lookup order is not a requirement to add an extra live-state tier.

## Qwen rule

The Qwen prefix cache is already substantially integrated. Do not rewrite it as part of the V4 workstream.

When output caching is added later, Qwen and V4 may share the higher-level deterministic request key/output-entry layer because output tokens are architecture-neutral once model/tokenizer/generation identity is pinned. Their underlying prefix/KV caches remain engine-specific.

## Persistent storage rule

Reuse `kv_persist.h`'s durability ideas, not necessarily its file bytes. In particular:

- validate model/geometry/version before restore;
- write payload first and commit metadata last, or retain an equally strong existing V4 atomic framing rule;
- reject truncation and stale state as a miss;
- never partially restore an entry.

If `coli_v4_prefix_disk` already provides these guarantees, harden/test that implementation instead of creating another V4 `.coli_kv` format.

## Output-cache relationship

Output caching sits above prefix caching:

- output hit: no prefill and no decode;
- output miss + prefix hit: skip matched prefill, then decode normally;
- both miss: cold/full prefill and decode.

Output cache entries store deterministic completion token IDs and request identity. They do not store or expose engine KV state.

## Acceptance principle

Every reuse path must have a cold-path oracle test. The core invariant is simple:

> A cache hit may reduce work, but it must not change the token sequence the uncached engine would have produced for the same deterministic request.

Performance decides whether a passing optimization becomes default-on; correctness decides whether it is allowed to exist at all.

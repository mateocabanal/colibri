# Cache Workstream Plan

This document defines the staged plan for prompt/prefix reuse, persistent SSD KV state, and deterministic output caching in colibri. It is a plan only; this branch contains no engine behavior changes.

## Goals

1. Make repeated chat turns cheap when the client resends an unchanged transcript prefix.
2. Let a conversation reopen warm after process restart without re-prefilling the saved prefix.
3. Re-serve an already generated completion when the entire deterministic request is identical.
4. Reuse existing cache machinery instead of creating parallel state formats.
5. Keep the default execution path unchanged until current-main measurements show a win and correctness gates are green.

## Existing pieces to preserve

The repository already has three different cache shapes because their runtime state is different:

- `qwen_prefix_cache.*` already provides substantial Qwen prompt-prefix reuse.
- `coli_v4_prefix_cache.*` owns V4 RAM snapshots, statistics, exact-prefix lookup, and snapshot wire codecs for attention/compressor/indexer state.
- `coli_v4_prefix_disk.*` is the natural persistent adapter for those V4 snapshots and already has engine registration, restore, and publish boundaries.
- `kv_prefix.h` is a tiny live-state token record used by engines whose KV state can simply remain in place between turns.
- `kv_persist.h` provides a useful durability precedent for `.coli_kv`: model-shape validation and commit metadata written last so an interrupted append does not advertise incomplete records.

The workstream should unify identity, safety, and observability contracts where useful, not force these engines to share one state representation.

## Stage 1: V4 prompt/prefix reuse

### Objective

Wire the existing V4 RAM prefix cache into the real `deepseek_v4` request path so a later chat request whose tokenized prompt begins with an already cached prefix restores that prefix and prefills only the new tail.

### Integration points

The implementation should use the existing public V4 cache API rather than introducing another snapshot container:

- engine lifetime: retain the existing registration/retirement wrappers;
- before prefill: attempt `coli_v4_prefix_cache_restore(session, prompt_ids, prompt_tokens)`;
- prefill: start at the returned matched-token boundary and process only the unmatched tail;
- end-of-prefill / successful generation boundary: publish a reusable exact prompt state with `coli_v4_prefix_cache_store(session)`;
- persistent adapter hooks stay outside token streaming where possible, using the existing ref-pinned snapshot visitor;
- CLI/chat/serve loops must preserve the exact token transcript used to build the state so a resent conversation can hit the cache.

`kv_prefix.h` should be used only as a small bookkeeping helper if the V4 session needs an exact record of tokens actually fed. It must not become a second V4 snapshot cache.

### Correctness contract

A V4 prefix hit is legal only when all of the following hold:

- cached token IDs exactly equal the leading token IDs of the new request;
- the restored state belongs to the same model/package identity and compatible V4 state ABI/layout;
- the request has an unmatched tail to prefill; no implicit rewind is allowed;
- any input not fully described by token IDs is either included in the identity or makes the entry ineligible;
- restore failure is an optimization miss, never a generation failure;
- feature-off behavior is byte-for-byte/current-path equivalent.

The restored native state must include every V4 component needed to make the next token identical to a full prefill: attention/window state, compressor state, indexer state, positions/counters, and any other recurrent state discovered during the implementation audit.

### Admission and memory

Keep the existing V4 prefix-cache budget accounting and planner reserve. Do not silently allocate cache memory outside `--memory-gb` / the automatic memory envelope.

The current default remains unchanged until current-main benchmarks demonstrate a useful TTFT/prefill reduction without a material decode regression or memory-policy violation. Existing measurements documented elsewhere are evidence that the idea is valuable, but this integration gets its own acceptance numbers.

### Stage 1 acceptance

Minimum local gates:

- exact same prompt, cold vs restored: identical next-token logits/argmax or deterministic token sequence;
- transcript extension: restored run skips the matched prefix and produces the same tokens as a cold full-prefill run;
- one-token divergence inside the prefix: miss and cold-prefill behavior;
- shorter/equal prompt requiring rewind: miss;
- cache disabled: current-main behavior;
- memory budget: planner + cache remain inside the original envelope;
- stats expose lookups, hits, matched tokens, restore bytes/time, and stores;
- deterministic short real-model chat test comparing cold and warm token IDs;
- benchmark TTFT/prefill wall time at several reuse ratios.

## Stage 2: persistent SSD KV / V4 prefix state

### Objective

Allow a V4 conversation prefix to survive engine/process restart and restore without model prefill.

### Representation decision

Use the existing V4 prefix-disk adapter and V4 native snapshot wire codecs as the V4 persisted payload. Do **not** write V4 state using `kv_persist.h`'s raw per-position layout: V4 already has explicit serializers for the state it actually needs.

Reuse the durability principles from `kv_persist.h` where they fit:

- validate magic/version plus model/package/state ABI identity before trusting bytes;
- publish commit metadata only after the full payload and checksum are durable;
- a crash during append/store must leave the previous committed entry readable or make the new entry invisible;
- malformed, truncated, stale, or foreign-model files fail closed to a cache miss;
- cache corruption must never change generation semantics.

If the existing V4 disk framing already provides stronger atomicity, keep that framing rather than adding a second `.coli_kv` format.

### Namespace and identity

Persistent entries need an explicit namespace so unrelated conversations do not collide. The storage key should cover at least:

- model/package source fingerprint;
- V4 state/wire ABI and target-relevant layout identity;
- tokenizer/chat-template identity when tokenization is performed outside the cache layer;
- exact prefix token sequence (or a collision-resistant digest plus exact-token verification);
- conversation/cache namespace supplied by the caller when persistent chat continuity is desired.

Path names are not identity. The header/key must make stale files self-rejecting after model or ABI changes.

### SSD policy

Persistent caching is an optimization tier, not a second source of truth.

- RAM hit wins over SSD.
- SSD restore happens before cold prefill.
- writes should avoid blocking TTFT/token streaming; publish after the prompt boundary/generation when the existing adapter can borrow immutable snapshots.
- bound total bytes and provide an eviction/retention policy before default-on persistence.
- use checksums for both metadata and state payloads where the current disk format supports them.

### Stage 2 acceptance

- create cache, terminate process, restart, exact transcript restores and matches cold token output;
- process killed during write: prior entry remains usable or incomplete entry is ignored;
- truncate/corrupt payload/header/checksum: miss, no crash, no partial restore;
- wrong model/package/state ABI: miss;
- exact token mismatch: miss;
- restore-time and bytes-read metrics;
- compare restart TTFT against cold prefill on a real conversation prefix;
- bounded disk usage/eviction test.

## Stage 3: deterministic output cache

### Objective

For requests whose generation is fully deterministic, return a previously produced completion without running model decode again.

This is separate from prefix/KV caching. Prefix caching avoids recomputing prompt state but still generates new output; output caching may skip the whole model request.

### V1 eligibility

V1 is deliberately narrow: deterministic greedy generation only.

Do not cache or serve sampled generation unless a future contract includes every RNG/seed/state input required to make it reproducible. If temperature/top-p/top-k or another sampling path can affect token selection, the request is ineligible unless it resolves to the engine's exact deterministic greedy mode.

### Cache key

The key must identify the effective request, not just user text. It should include:

- model/package source fingerprint and execution/state ABI relevant to token identity;
- tokenizer identity/version and chat-template/system-prompt identity;
- exact tokenized input transcript, including any supplied assistant/generated prefix tokens;
- generation mode and every setting that can change observable output: greedy/sampling mode, max-new limit, EOS/stop-token behavior, stop sequences, reasoning/thinking mode, speculative verification mode when it can change externally visible termination, and other engine-specific switches discovered in the implementation audit;
- output-cache schema version.

Prefer a collision-resistant digest for lookup, but keep enough metadata/token data to verify a candidate hit before serving it.

### Stored value

Store token IDs as the canonical completion. Text is derived output and may optionally be stored only as an acceleration after tokenizer identity is pinned.

An entry should also store:

- completion token count;
- termination reason / EOS or stop condition;
- checksum over stored token IDs and metadata;
- key/version/model fingerprints needed for validation.

### Verification semantics

"Verify token identity before re-serving" means fail closed on identity, not rerun inference on every cache hit.

Before serving, verify the complete cache key/fingerprints and the stored completion checksum/token framing. Acceptance tests must independently replay cached requests through live greedy generation and prove the stored token IDs are identical. An optional debug/oracle mode may recompute and compare, but production hits must not erase the benefit by decoding the completion again.

### Output-cache invalidation

Any model/package, tokenizer/template, generation-setting, schema, or semantics change represented in the key naturally creates a miss. Manual global invalidation remains useful for development but must not be required for correctness.

### Stage 3 acceptance

- identical deterministic request: cache hit, byte/token-identical completion;
- one input token changed: miss;
- model/package fingerprint changed: miss;
- max-new/stop/reasoning setting changed: miss;
- sampled request: ineligible in V1;
- corrupt/truncated entry: miss;
- live greedy oracle vs stored completion: exact token identity;
- hit latency measured against normal prefix-hit generation;
- bounded RAM/disk policy before default-on.

## Cross-stage observability

Use separate counters so we can tell where the win came from:

- prefix RAM lookup/hit/matched tokens/restore time;
- prefix SSD lookup/hit/read bytes/restore time/write bytes/write time;
- output lookup/hit/tokens bypassed/bytes served;
- invalid/stale/corrupt entry rejections.

Logs should be opt-in and concise. Tests should consume machine-readable stats where possible instead of parsing prose.

## Rollout order

1. Land this plan only.
2. Implement V4 RAM prefix integration behind the existing cache controls; measure cold vs warm correctness and TTFT.
3. If the measured win is strong and resource accounting is clean, decide whether the V4 RAM cache becomes default-on and at what budget.
4. Connect/harden persistent V4 SSD restore/store and crash-safety; keep it separately controllable until restart benchmarks and corruption tests pass.
5. Add deterministic output cache as a separate module/API, initially opt-in.
6. Only after each stage has measured wins and stable gates, consider sharing common key/fingerprint helpers with Qwen and other engines.

## Non-goals for the first implementation

- sharing raw KV/snapshot bytes between different model architectures;
- semantic/fuzzy prompt matching;
- cache reuse across different model weights;
- speculative or sampled output-cache reuse without complete determinism inputs;
- remote/distributed cache service;
- changing model math or token selection to make cache hits easier.

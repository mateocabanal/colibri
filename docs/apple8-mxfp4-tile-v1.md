# Apple8 MXFP4 tile execution contract v1

Status: production execution-layout contract for `macos-arm64-metal-apple8-v1`.

## Identity

The persisted production identity is generated from `abi/coli-target-registry.json`:

- execution layout: `APPLE_MXFP4_TILE8X32_V1` = `0x0103`
- execution-layout ABI: `1`
- Metal kernel ABI: `1`
- target compatibility class: `0x01000001`
- target profile: `macos-arm64-metal-apple8-v1`

The milestone fixture values `0x7131`, layout/kernel ABI `0x0131`, and target class
`0x01310008` are not production IDs and must be rejected as persisted Apple8-v1
execution layouts.

## Matrix bytes

Required semantic geometry:

- math format: MXFP4 E2M1
- scale format: unsigned E8M0
- scale block: 1 output row x 32 K columns
- group size: 0

One physical tile covers 8 output rows x 32 K values and contains exactly 136 bytes:

- bytes 0..127: eight 16-byte packed E2M1 row fragments
- bytes 128..135: one E8M0 scale byte for each of the eight output rows

Tile order is output-row tile major, then K-group:

`tile_index = (output_row / 8) * ceil(columns / 32) + (k / 32)`

The resident matrix byte count is exactly:

`ceil(rows / 8) * ceil(columns / 32) * 136`

Unused output-row slots, unused packed weight bytes in a partial final K group,
and the unused high nibble for an odd final K column are zero in deterministic
compiled output.

## COLIEXPT descriptor: combined payload Design A

Apple8 co-locates scales with weights. It therefore has one physical matrix
payload, not a weight plane plus a fake separate scale plane.

For layout `0x0103`, the historical `weight_*` fields name the primary combined
physical matrix payload:

- `weight_offset`: start of the combined 136-byte tile stream
- `weight_stored_bytes`: exact stored size; equals resident size for codec NONE
- `weight_decoded_bytes`: exact combined resident size
- `weight_codec`: NONE in PR 1
- `weight_codec_table_id`: zero in PR 1

The semantic `scale_format` remains UE8M0, but the independent physical scale
fields are all zero:

- `scale_offset = 0`
- `scale_stored_bytes = 0`
- `scale_decoded_bytes = 0`
- `scale_codec = NONE`
- `scale_codec_table_id = 0`

This exception is specific to `APPLE_MXFP4_TILE8X32_V1`. Canonical/ROW32 MXFP4
continues to use separate weight and scale spans.

The per-matrix logical CRC covers the exact decoded combined Apple8 target byte
stream, including deterministic edge padding. A later lossless storage codec
must decode byte-for-byte to that same stream and does not change representation
identity.

## Compatibility and refusal

The runtime must treat representation identity as exact. A package or resident
variant is not Apple8-v1 executable unless profile, layout, execution-layout ABI,
kernel ABI, target class, math/scale semantics and scale geometry all match the
registered contract.

No runtime path may reinterpret a canonical, ROW32, x86, CUDA, or fixture layout
as Apple8-v1 merely because the mathematical MXFP4 values could be transformed.
An explicit exact JIT transform may remain a compatibility path for an artifact
that honestly declares its source representation, but wrong-target artifacts
fail closed rather than being silently canonicalized.

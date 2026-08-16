# CSF v1.1 target compatibility fixtures

These fixtures are independent parser oracles for the target-compiled CSF
contract in `docs/coli-serving-format-v1.md`.

They are **not emitted by `colic`**. `make_fixtures.py` writes the byte fields
directly from the v1.1 offsets and independently implements CRC32C, source
fingerprinting and artifact fingerprinting.

Each package contains one layout-neutral `BLOB` record. That is intentional:
these fixtures freeze target compatibility/framing without inventing #26's
future matrix packing or kernel layout IDs.

## Reconstruct

```sh
python3 make_fixtures.py --output /tmp/csf-v1-targets
```

The script asserts the identities below before writing anything.

## Apple / M2 / Metal fixture

Directory: `apple/`

```text
profile                    macos-arm64-metal-apple8-v1
OS                         macOS
arch                       arm64
backend                    Metal
GPU namespace              Apple GPU family
minimum Apple family       8
CPU requirements           ARM64_ASIMD
record alignment           16384
I/O granularity            16384
resident alignment         16384
runtime features           APPLE_UNIFIED_MEMORY | METAL_SHARED_STORAGE
```

Expected binary identities:

```text
manifest.coli
  bytes    944
  sha256   8c3ce7c8532d3db1abae0498d517502004b07b0d2d31d5d555d3a388c37eb4e8

data-00000.coli
  bytes    16405
  sha256   19f8516349e50d732079ab88a93ad54a6e3bb4aa722f0c496a072ecd77ec7efe
```

Canonical fingerprints/CRCs:

```text
source_fingerprint       34da718f420aec269094374dc41e9df5d2593fccb8da435c1788e07fda3a0853
artifact_fingerprint     5fa5bc6abb988815bc3627b551e663c1a55d347535a71712d31d1aab2ad96182
target_desc_crc32c       0x43f5cc3e
manifest_crc32c          0xb409f497
data_header_crc32c       0xe44d1442
record/logical_crc32c    0x6a7b4ecb
```

The virtual source inventory used only to make the fixture's source fingerprint
well-defined is one kind-3 entry named `fixture.source` whose bytes are
`apple-target-fixture-v1`.

## Linux x86_64 / CUDA sm_89 fixture

Directory: `linux-cuda-sm89/`

```text
profile                    linux-x86_64-cuda-sm89-v1
OS                         Linux
arch                       x86_64
backend                    CUDA
GPU namespace              CUDA SM
compute capability         exactly 89 for this fixture
CPU requirements           AVX2 | FMA
record alignment           4096
I/O granularity            4096
resident alignment         256
runtime features           CUDA_RUNTIME | CUDA_ASYNC_COPY | HOST_PINNED_STAGING
```

Expected binary identities:

```text
manifest.coli
  bytes    960
  sha256   7d628399f4f0b7847585fb57238f81708b1bc929b4f45ada0adb21e3ee416b97

data-00000.coli
  bytes    4120
  sha256   5f813c80e80259a95189c248ea13bdd0900f8687a921fbd7f3d7466420bf50ea
```

Canonical fingerprints/CRCs:

```text
source_fingerprint       be9ded9dd47a909b00c7cf71ef2c6209dd8a0ac9cc81da23cb07684a2897601b
artifact_fingerprint     c5ddb89fc599c2eefb57c6bd00571426eb233f9913ac5f8f5787747c79a68b7b
target_desc_crc32c       0x188921ad
manifest_crc32c          0xc772fa5a
data_header_crc32c       0xe6ada26e
record/logical_crc32c    0x6348e237
```

Its virtual source inventory contains one `fixture.source` entry with bytes
`linux-cuda-sm89-target-fixture-v1`.

## Compatibility tests expected in #23

The reader should use these packages to prove at least:

- Apple fixture accepts a macOS/arm64/Metal runtime supporting Apple family 8,
  ASIMD, unified memory, shared Metal storage, profile/layout ABI 1 and kernel
  ABI 1.
- Apple fixture rejects Linux/CUDA and rejects an Apple-family-7-only runtime.
- CUDA fixture accepts Linux/x86_64/CUDA sm_89 with AVX2+FMA and the declared
  runtime features.
- CUDA fixture rejects sm_80, macOS/Metal, missing AVX2/FMA, and wrong profile or
  kernel ABI.
- v1.0-only readers reject both fixtures before record publication because
  container minor 1 and required target feature bit 16 are set.

# Colibri Serving Format (`.coli`)

This is the stable entry point for the current COLI serving/deployment contract.

## Current deployment contract

**v1.1 target-compiled:** [`coli-serving-format-v1.1-target.md`](./coli-serving-format-v1.1-target.md)

The production architecture is:

```text
source checkpoint
      |
      v
    colic
      |
      v
target-compiled .coli
      |
      v
Colibri runtime
```

Safetensors/Hugging Face checkpoints are compiler and verification inputs. They are not required inside an installed `.coli` artifact or by the normal inference runtime.

A production `.coli` declares the target/profile ABI it was lowered for. Incompatible artifacts are recompiled rather than converted/repacked on the inference hot path.

## Legacy framing reference

**v1.0 portable framing:** [`coli-serving-format-v1.md`](./coli-serving-format-v1.md)

The v1.0 document and tiny fixture are retained because they define useful framing, corruption handling, and parser regression material already merged in #39/#42. `portable-v1` is now a legacy/parser contract, not the production deployment target.

Do not implement new compiler/runtime work from the v1.0 portability policy. New work should follow the v1.1 target-compiled contract and issues #22–#28.

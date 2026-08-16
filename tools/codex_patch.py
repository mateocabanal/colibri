#!/usr/bin/env python3
from pathlib import Path


def replace(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"patch anchor missing in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))


# #49: old unit fixtures need the newly explicit semantic fields.
replace(
    "colic/src/pipeline.rs",
    """    use crate::{\n        ir::{Architecture, Matrix, ModelGeometry, RoutedExpert},\n        source::TensorRef,\n    };""",
    """    use crate::{\n        ir::{\n            Activation, Architecture, MathFormat, Matrix, ModelAssets, ModelGeometry,\n            Quantization, RoutedExpert, ScaleFormat, SourceRepresentation,\n        },\n        source::TensorRef,\n    };""",
)
replace(
    "colic/src/pipeline.rs",
    """        let matrix = Matrix {\n            source: tensor(1),\n            rows: 1,\n            columns: 1,\n            scale: None,\n        };""",
    """        let matrix = Matrix {\n            source: tensor(1),\n            rows: 1,\n            columns: 1,\n            scale: None,\n            quantization: Quantization {\n                math_format: MathFormat::MxFp4E2M1,\n                source_representation: SourceRepresentation::PackedMxFp4Nibbles,\n                scale_format: ScaleFormat::Ue8m0,\n                scale_block_rows: 1,\n                scale_block_columns: 32,\n            },\n        };""",
)
replace(
    "colic/src/pipeline.rs",
    """                gate: matrix.clone(),\n                up: matrix.clone(),\n                down: matrix,\n            },""",
    """                gate: matrix.clone(),\n                up: matrix.clone(),\n                down: matrix,\n                activation: Activation::SwiGlu,\n            },""",
)
replace(
    "colic/src/pipeline.rs",
    """            resident_tensors: BTreeMap::new(),\n        };""",
    """            resident_tensors: BTreeMap::new(),\n            assets: ModelAssets::default(),\n        };""",
)

replace(
    "colic/src/target/mod.rs",
    """    use crate::{ir::Matrix, source::TensorRef};""",
    """    use crate::{\n        ir::{Activation, MathFormat, Matrix, Quantization, ScaleFormat, SourceRepresentation},\n        source::TensorRef,\n    };""",
)
replace(
    "colic/src/target/mod.rs",
    """            scale: Some(TensorRef {\n                source: path.clone(),\n                offset: offset + 2,\n                len: 2,\n                dtype: \"F8_E8M0\".into(),\n                shape: vec![1, 1],\n            }),\n        };""",
    """            scale: Some(TensorRef {\n                source: path.clone(),\n                offset: offset + 2,\n                len: 2,\n                dtype: \"F8_E8M0\".into(),\n                shape: vec![1, 1],\n            }),\n            quantization: Quantization {\n                math_format: MathFormat::Fp8E4M3,\n                source_representation: SourceRepresentation::NativeFp8,\n                scale_format: ScaleFormat::Ue8m0,\n                scale_block_rows: 128,\n                scale_block_columns: 128,\n            },\n        };""",
)
replace(
    "colic/src/target/mod.rs",
    """            gate: matrix(0, \"F8_E4M3FN\"),\n            up: matrix(4, \"F8_E4M3FN\"),\n            down: matrix(8, \"F8_E4M3FN\"),\n        };""",
    """            gate: matrix(0, \"F8_E4M3FN\"),\n            up: matrix(4, \"F8_E4M3FN\"),\n            down: matrix(8, \"F8_E4M3FN\"),\n            activation: Activation::SwiGlu,\n        };""",
)
replace(
    "colic/src/target/mod.rs",
    """            scale: Some(TensorRef {\n                source: path.clone(),\n                offset: offset + 1,\n                len: 1,\n                dtype: \"F8_E8M0\".into(),\n                shape: vec![1, 1],\n            }),\n        };""",
    """            scale: Some(TensorRef {\n                source: path.clone(),\n                offset: offset + 1,\n                len: 1,\n                dtype: \"F8_E8M0\".into(),\n                shape: vec![1, 1],\n            }),\n            quantization: Quantization {\n                math_format: MathFormat::MxFp4E2M1,\n                source_representation: SourceRepresentation::PackedMxFp4Nibbles,\n                scale_format: ScaleFormat::Ue8m0,\n                scale_block_rows: 1,\n                scale_block_columns: 32,\n            },\n        };""",
)
replace(
    "colic/src/target/mod.rs",
    """            gate: matrix(0),\n            up: matrix(2),\n            down: matrix(4),\n        };""",
    """            gate: matrix(0),\n            up: matrix(2),\n            down: matrix(4),\n            activation: Activation::SwiGlu,\n        };""",
)

# #52: verify the temporary package before either publication path can touch
# the final artifact. A failed --force --verify therefore preserves the old one.
replace(
    "colic/src/pipeline.rs",
    """    if request.force {\n        storage::replace_package(&temporary, output)\n    } else {\n        storage::publish_package(&temporary, output)\n    }?;\n    if request.verify {\n        progress.stage(Stage::Verification);\n        let _summary = verify::verify_package(output)?;\n    }\n    Ok(())""",
    """    if request.verify {\n        progress.stage(Stage::Verification);\n        if let Err(error) = verify::verify_package(&temporary) {\n            let _ = fs::remove_dir_all(&temporary);\n            return Err(error);\n        }\n    }\n    if request.force {\n        storage::replace_package(&temporary, output)\n    } else {\n        storage::publish_package(&temporary, output)\n    }?;\n    Ok(())""",
)

# Keep non-test builds warning-clean.
p = Path("colic/src/model/deepseek_v4_semantic.rs")
text = p.read_text().replace("    path::{Path, PathBuf},\n", "    path::Path,\n")
text = text.replace("    fn root() -> PathBuf {", "    fn root() -> std::path::PathBuf {")
p.write_text(text)

# Merge the existing v1 reader gate with the target identity/profile suites.
Path("c/Makefile.csf").write_text(r'''# Standalone CSF reader / target ABI gate.
# Kept separate from the engine build so persisted-format compatibility remains
# cheap to test across Linux, macOS, and Windows.

CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra
EXTRA_CFLAGS ?=
EXTRA_LDFLAGS ?=
EXE ?=

ifeq ($(OS),Windows_NT)
CFLAGS += -D_FILE_OFFSET_BITS=64
EXE := .exe
endif

.PHONY: test test-profiles test-target test-asan test-profiles-asan test-target-asan clean

tests/test_coli_format$(EXE): tests/test_coli_format.c coli_format.c coli_executor.c coli_format.h coli_executor.h compat.h
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) tests/test_coli_format.c coli_format.c coli_executor.c -o $@ -pthread $(EXTRA_LDFLAGS)

tests/test_coli_target_profiles$(EXE): tests/test_coli_target_profiles.c coli_target_profiles.c coli_target_profiles.h coli_target.h coli_format.h
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) tests/test_coli_target_profiles.c coli_target_profiles.c -o $@ $(EXTRA_LDFLAGS)

tests/test_coli_target$(EXE): tests/test_coli_target.c coli_target.c coli_target.h coli_format.c coli_format.h compat.h
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) tests/test_coli_target.c coli_target.c coli_format.c -o $@ -pthread $(EXTRA_LDFLAGS)

test: tests/test_coli_format$(EXE) tests/test_coli_target_profiles$(EXE)
	./tests/test_coli_format$(EXE)
	./tests/test_coli_target_profiles$(EXE)

test-profiles: tests/test_coli_target_profiles$(EXE)
	./tests/test_coli_target_profiles$(EXE)

test-target: tests/test_coli_target$(EXE)
	@test -n "$(APPLE_FIXTURE)" || (echo "APPLE_FIXTURE is required" >&2; exit 2)
	@test -n "$(CUDA_FIXTURE)" || (echo "CUDA_FIXTURE is required" >&2; exit 2)
	./tests/test_coli_target$(EXE) "$(APPLE_FIXTURE)" "$(CUDA_FIXTURE)"

test-asan:
	$(MAKE) -f Makefile.csf clean
	$(MAKE) -f Makefile.csf test EXTRA_CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" EXTRA_LDFLAGS="-fsanitize=address,undefined"

test-profiles-asan:
	$(MAKE) -f Makefile.csf clean
	$(MAKE) -f Makefile.csf test-profiles EXTRA_CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" EXTRA_LDFLAGS="-fsanitize=address,undefined"

test-target-asan:
	$(MAKE) -f Makefile.csf clean
	$(MAKE) -f Makefile.csf tests/test_coli_target$(EXE) EXTRA_CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" EXTRA_LDFLAGS="-fsanitize=address,undefined"
	$(MAKE) -f Makefile.csf test-target APPLE_FIXTURE="$(APPLE_FIXTURE)" CUDA_FIXTURE="$(CUDA_FIXTURE)" EXTRA_CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" EXTRA_LDFLAGS="-fsanitize=address,undefined"

clean:
	rm -f tests/test_coli_format tests/test_coli_format.exe \
	      tests/test_coli_target tests/test_coli_target.exe \
	      tests/test_coli_target_profiles tests/test_coli_target_profiles.exe
''')

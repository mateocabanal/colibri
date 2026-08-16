#!/usr/bin/env python3
from pathlib import Path


def replace(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"patch anchor missing in {path}: {old[:80]!r}")
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
# Both target Matrix fixture constructors get explicit semantics.
needle = """            scale: Some(TensorRef {\n                source: path.clone(),\n                offset: offset + 2,\n                len: 2,\n                dtype: \"F8_E8M0\".into(),\n                shape: vec![1, 1],\n            }),\n        };"""
replacement = """            scale: Some(TensorRef {\n                source: path.clone(),\n                offset: offset + 2,\n                len: 2,\n                dtype: \"F8_E8M0\".into(),\n                shape: vec![1, 1],\n            }),\n            quantization: Quantization {\n                math_format: MathFormat::Fp8E4M3,\n                source_representation: SourceRepresentation::NativeFp8,\n                scale_format: ScaleFormat::Ue8m0,\n                scale_block_rows: 128,\n                scale_block_columns: 128,\n            },\n        };"""
replace("colic/src/target/mod.rs", needle, replacement)
# The second target test uses the same field shape but an I8 MXFP4 source.
needle2 = """            scale: Some(TensorRef {\n                source: path.clone(),\n                offset: offset + 2,\n                len: 1,\n                dtype: \"F8_E8M0\".into(),\n                shape: vec![1, 1],\n            }),\n        };"""
replacement2 = """            scale: Some(TensorRef {\n                source: path.clone(),\n                offset: offset + 2,\n                len: 1,\n                dtype: \"F8_E8M0\".into(),\n                shape: vec![1, 1],\n            }),\n            quantization: Quantization {\n                math_format: MathFormat::MxFp4E2M1,\n                source_representation: SourceRepresentation::PackedMxFp4Nibbles,\n                scale_format: ScaleFormat::Ue8m0,\n                scale_block_rows: 1,\n                scale_block_columns: 32,\n            },\n        };"""
replace("colic/src/target/mod.rs", needle2, replacement2)
# There are two RoutedExpert literals in target tests.
for _ in range(2):
    replace(
        "colic/src/target/mod.rs",
        """            gate: matrix(0, \"F8_E4M3FN\"),\n            up: matrix(4, \"F8_E4M3FN\"),\n            down: matrix(8, \"F8_E4M3FN\"),\n        };""",
        """            gate: matrix(0, \"F8_E4M3FN\"),\n            up: matrix(4, \"F8_E4M3FN\"),\n            down: matrix(8, \"F8_E4M3FN\"),\n            activation: Activation::SwiGlu,\n        };""",
    ) if "gate: matrix(0, \"F8_E4M3FN\")" in Path("colic/src/target/mod.rs").read_text() else None
# The MXFP4 test uses a closure without dtype.
if "gate: matrix(0)," in Path("colic/src/target/mod.rs").read_text():
    replace(
        "colic/src/target/mod.rs",
        """            gate: matrix(0),\n            up: matrix(3),\n            down: matrix(6),\n        };""",
        """            gate: matrix(0),\n            up: matrix(3),\n            down: matrix(6),\n            activation: Activation::SwiGlu,\n        };""",
    )

# #52: expensive verification must happen against the temporary package before
# either first publication or --force replacement can touch the final output.
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

//! Target registry data lives in the shared `colibri-abi` crate (generated
//! from `abi/coli-target-registry.toml` by `tools/gen_target_registry.py`).
//! This module keeps the historical `colic::target_registry` path working.
pub use colibri_abi::generated::target_registry::*;

//! `colic` is the offline compiler for target-compiled COLI artifacts.
//!
//! It deliberately has no link-time relationship with the C inference runtime.

pub mod cli;
pub mod codec;
pub mod error;
pub mod format;
pub mod generated;
pub mod ir;
pub mod model;
pub mod passes;
pub mod pipeline;
pub mod quant;
pub mod source;
pub mod storage;
pub mod target;
pub mod target_registry;
pub mod verify;
pub mod verify_target;

pub use error::{ColicError, Result};

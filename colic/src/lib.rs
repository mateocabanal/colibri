//! `colic` is the offline compiler for target-compiled COLI artifacts.
//!
//! It deliberately has no link-time relationship with the C inference runtime.

pub mod cli;
pub mod error;
pub mod format;
pub mod ir;
pub mod model;
pub mod passes;
pub mod pipeline;
pub mod quant;
pub mod source;
pub mod storage;
pub mod target;
pub mod verify;

pub use error::{ColicError, Result};

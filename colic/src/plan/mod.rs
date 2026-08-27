//! #191 — automatic hardware-aware planning (umbrella module).
//!
//! `machine` = capability probe (#192), `ir` = physical plan IR (#193),
//! `memory` = context-aware budgets (#194), `cost` = objectives + reasons
//! (#195), `placement` = RAM/VRAM/cache/pageable policy (#199),
//! `manifest` = reproducible plan replay (#201).

pub mod ir;
pub mod machine;
pub mod cost;
pub mod memory;
pub mod placement;
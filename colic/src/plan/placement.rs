//! #199 — Placement planner: turn memory budgets into per-tensor placement.
//!
//! Mandatory state (dense/static) first, then PLE resident (tiny random
//! gathers hate paging), then expert caches (VRAM for the CUDA representation
//! when present, RAM otherwise); pageable storage is the backing tier. The
//! planner never relies on swap/pagefile as intended residency.

use crate::plan::ir::{Placement, TensorPlan, TensorRole};
use crate::plan::memory::MemoryBudgets;

/// Expert popularity share covered by the caches (v1: uniform hot set; the
/// runtime's activation telemetry can later refine per-expert ranking).
/// # ponytail: uniform hot-set fraction, per-expert telemetry when #132-style
/// activation data is wired into the compiler.
pub const HOT_SET_FRACTION: f64 = 0.0;

#[derive(Debug, Clone, PartialEq)]
pub struct PlacementOutcome {
    /// Bytes of expert data that fit the VRAM cache (0 when no CUDA rep).
    pub vram_cached_bytes: u64,
    pub ram_cached_bytes: u64,
    pub pageable_bytes: u64,
    pub ple_resident: bool,
    pub notes: Vec<String>,
}

/// Assign placement for every tensor plan in place (deterministic order =
/// input order; callers pass BTreeMap-ordered tensors).
pub fn place(tensors: &mut [TensorPlan], budgets: &MemoryBudgets) -> PlacementOutcome {
    let mut outcome = PlacementOutcome {
        vram_cached_bytes: 0,
        ram_cached_bytes: 0,
        pageable_bytes: 0,
        ple_resident: false,
        notes: Vec::new(),
    };

    // Pass 1: dense/static families are mandatory-resident on host RAM.
    for tensor in tensors.iter_mut() {
        if !matches!(
            tensor.role,
            TensorRole::RoutedExpert | TensorRole::NgramPle
        ) {
            tensor.placement = Placement::ResidentRam;
        }
    }

    // Pass 2: PLE n-gram — resident in RAM when it fits the expert-cache
    // budget (it is tiny compared to experts but random-access heavy).
    let ple_bytes: u64 = tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::NgramPle)
        .map(|tensor| tensor.stored_bytes)
        .sum();
    if ple_bytes > 0 {
        if ple_bytes <= budgets.ram_expert_cache_bytes {
            for tensor in tensors.iter_mut().filter(|t| t.role == TensorRole::NgramPle) {
                tensor.placement = Placement::ResidentRam;
            }
            outcome.ple_resident = true;
            outcome.notes.push(format!(
                "ple resident in ram ({ple_bytes} B <= ram cache budget)"
            ));
        } else {
            for tensor in tensors.iter_mut().filter(|t| t.role == TensorRole::NgramPle) {
                tensor.placement = Placement::Pageable;
            }
            outcome.notes.push(format!(
                "ple does NOT fit ram cache budget ({ple_bytes} B) — pageable"
            ));
        }
    }

    // Pass 3: routed experts — VRAM cache first (when a CUDA rep exists),
    // then RAM cache, remainder pageable.
    let has_cuda_rep = tensors
        .iter()
        .any(|tensor| tensor.role == TensorRole::RoutedExpert && tensor.backend == crate::plan::ir::BackendKind::Cuda);
    let mut vram_budget = if has_cuda_rep {
        budgets.vram_expert_cache_bytes
    } else {
        0
    };
    let mut ram_budget = budgets.ram_expert_cache_bytes.saturating_sub(ple_bytes);
    for tensor in tensors.iter_mut().filter(|t| t.role == TensorRole::RoutedExpert) {
        if vram_budget >= tensor.stored_bytes {
            tensor.placement = Placement::VramCache;
            vram_budget -= tensor.stored_bytes;
            outcome.vram_cached_bytes += tensor.stored_bytes;
        } else if ram_budget >= tensor.stored_bytes {
            tensor.placement = Placement::RamCache;
            ram_budget -= tensor.stored_bytes;
            outcome.ram_cached_bytes += tensor.stored_bytes;
        } else {
            tensor.placement = Placement::Pageable;
            outcome.pageable_bytes += tensor.stored_bytes;
        }
    }
    if has_cuda_rep && outcome.vram_cached_bytes > 0 {
        outcome
            .notes
            .push(format!("vram cache holds {} B of experts", outcome.vram_cached_bytes));
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::ir::{BackendKind, MathFormat, PhysicalLayout};

    fn expert(key: &str, backend: BackendKind, bytes: u64) -> TensorPlan {
        TensorPlan {
            semantic_key: key.into(),
            role: TensorRole::RoutedExpert,
            math: match backend {
                BackendKind::Cuda => MathFormat::Mxfp4,
                _ => MathFormat::Int4G32,
            },
            layout: match backend {
                BackendKind::Cuda => PhysicalLayout::PascalDp4aTile,
                _ => PhysicalLayout::Rows16,
            },
            backend,
            placement: Placement::Pageable,
            decoded_bytes: bytes * 3,
            stored_bytes: bytes,
            lowering_id: "test_v1".into(),
        }
    }

    fn dense(key: &str, role: TensorRole, bytes: u64) -> TensorPlan {
        TensorPlan {
            semantic_key: key.into(),
            role,
            math: MathFormat::Bf16,
            layout: PhysicalLayout::Canonical,
            backend: BackendKind::Cpu,
            placement: Placement::Pageable,
            decoded_bytes: bytes,
            stored_bytes: bytes,
            lowering_id: "exact_v1".into(),
        }
    }

    fn budgets(ram_cache: u64, vram_cache: u64) -> MemoryBudgets {
        MemoryBudgets {
            context_tokens: 8192,
            batch: 1,
            kv_bytes: 100,
            gdn_state_bytes: 100,
            qsa_state_bytes: 100,
            dense_resident_bytes: 100,
            runtime_reserve_bytes: 0,
            scratch_bytes: 0,
            ram_expert_cache_bytes: ram_cache,
            vram_expert_cache_bytes: vram_cache,
            headroom_pct: 10,
            fits: true,
            failures: Vec::new(),
        }
    }

    #[test]
    fn dense_becomes_resident_ram() {
        let mut tensors = vec![dense("ffn.gate.weight", TensorRole::Router, 1000)];
        let outcome = place(&mut tensors, &budgets(0, 0));
        assert_eq!(tensors[0].placement, Placement::ResidentRam);
        assert_eq!(outcome.pageable_bytes, 0);
    }

    #[test]
    fn ple_prefers_resident_ram_and_fits() {
        let mut tensors = vec![
            dense("ple.ngram.weight", TensorRole::NgramPle, 512),
            expert("e0", BackendKind::Cpu, 10_000),
        ];
        let outcome = place(&mut tensors, &budgets(12_000, 0));
        assert!(outcome.ple_resident);
        assert_eq!(tensors[0].placement, Placement::ResidentRam);
        // Expert still gets the remaining RAM cache.
        assert_eq!(tensors[1].placement, Placement::RamCache);
    }

    #[test]
    fn reducing_ram_pushes_ple_pageable_predictably() {
        let mut tensors = vec![
            dense("ple.ngram.weight", TensorRole::NgramPle, 512),
            expert("e0", BackendKind::Cpu, 10_000),
        ];
        let outcome = place(&mut tensors, &budgets(400, 0));
        assert!(!outcome.ple_resident);
        assert_eq!(tensors[0].placement, Placement::Pageable);
    }

    #[test]
    fn vram_cache_fills_first_when_cuda_rep_exists() {
        let mut tensors = vec![
            expert("e0", BackendKind::Cuda, 3_000),
            expert("e1", BackendKind::Cuda, 3_000),
            expert("e2", BackendKind::Cuda, 3_000),
            expert("e3", BackendKind::Cuda, 3_000),
        ];
        let outcome = place(&mut tensors, &budgets(0, 7_000));
        assert_eq!(outcome.vram_cached_bytes, 6_000);
        assert_eq!(tensors[0].placement, Placement::VramCache);
        assert_eq!(tensors[1].placement, Placement::VramCache);
        // e2: no vram, no ram -> pageable; e3 same.
        assert_eq!(tensors[2].placement, Placement::Pageable);
        assert_eq!(outcome.pageable_bytes, 6_000);
    }

    #[test]
    fn no_cuda_rep_means_no_vram_usage() {
        let mut tensors = vec![expert("e0", BackendKind::Cpu, 3_000)];
        let outcome = place(&mut tensors, &budgets(10_000, 10_000));
        assert_eq!(outcome.vram_cached_bytes, 0);
        assert_eq!(tensors[0].placement, Placement::RamCache);
    }
}
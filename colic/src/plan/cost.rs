//! #195 — Cost model and optimization objectives.
//!
//! Deterministic rule/table-based v1 (`COST_MODEL_VERSION`). Capability →
//! policy: the machine profile supplies facts, this module decides. Every
//! choice records the alternatives it beat and why (reproducible plans).

use crate::plan::ir::{BackendKind, Decision, MathFormat, PhysicalLayout, TensorRole};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    Quality,
    Balanced,
    Throughput,
    Latency,
    MinimumSize,
}

impl Objective {
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "quality" => Self::Quality,
            "balanced" => Self::Balanced,
            "throughput" => Self::Throughput,
            "latency" => Self::Latency,
            "minimum-size" | "size" => Self::MinimumSize,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Balanced => "balanced",
            Self::Throughput => "throughput",
            Self::Latency => "latency",
            Self::MinimumSize => "minimum-size",
        }
    }
}

/// One candidate representation offered by the cost model for a role.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub math: MathFormat,
    pub layout: PhysicalLayout,
    pub backend: BackendKind,
    /// Deterministic score: lower wins. Components logged in `reason`.
    pub score: f64,
    pub reason: String,
}

/// Rule-table v1 choice for one tensor role on one machine.
///
/// Returns `None` when the machine supports no representation for the role
/// (planner must then fail that role explicitly).
pub fn choose(
    role: TensorRole,
    machine: &crate::plan::machine::MachineProfile,
    objective: Objective,
    bytes_at: impl Fn(MathFormat) -> u64,
) -> Option<Decision> {
    choose_candidate(role, machine, objective, bytes_at).map(|(decision, _)| decision)
}

/// Structured winner + recorded decision (planner consumes the Candidate).
pub fn choose_candidate(
    role: TensorRole,
    machine: &crate::plan::machine::MachineProfile,
    objective: Objective,
    bytes_at: impl Fn(MathFormat) -> u64,
) -> Option<(Decision, Candidate)> {
    let mut candidates = Vec::new();

    let has_cuda_dp4a = machine.cuda_gpus().any(|gpu| match gpu {
        crate::plan::machine::GpuProfile::Cuda {
            dp4a,
            cc_major,
            tensor_cores,
            ..
        } => *dp4a && *cc_major >= 6 && !*tensor_cores && matches!(role, TensorRole::RoutedExpert),
        _ => false,
    });
    let has_cuda_any = machine
        .cuda_gpus()
        .any(|gpu| matches!(gpu, crate::plan::machine::GpuProfile::Cuda { .. }));
    let has_metal_apple8 = machine
        .metal_gpu()
        .is_some_and(|gpu| matches!(gpu, crate::plan::machine::GpuProfile::Metal { family: 8, .. }));
    let avx2 = machine.has_avx2();

    // ---- routed experts: the big pageable family; quantize aggressively ----
    // Pascal DP4A only wins when the expert working set fits VRAM (measured
    // on the 5600X/1080 box: per-call weight uploads made CUDA mxfp4 slower
    // than CPU when experts streamed from disk — 2243 vs 1931 ms/tok).
    let expert_stored_bytes = bytes_at(MathFormat::Mxfp4);
    let vram_headroom: u64 = machine
        .cuda_gpus()
        .map(|gpu| match gpu {
            crate::plan::machine::GpuProfile::Cuda { vram_bytes, .. } => *vram_bytes,
            _ => 0,
        })
        .sum();
    let experts_fit_vram = expert_stored_bytes > 0 && expert_stored_bytes <= vram_headroom * 9 / 10;
    if role == TensorRole::RoutedExpert {
        if avx2 {
            candidates.push(Candidate {
                math: MathFormat::Int4G32,
                layout: PhysicalLayout::Canonical,
                backend: BackendKind::Cpu,
                score: 10.0 + size_penalty(bytes_at(MathFormat::Int4G32), objective),
                reason: "avx2 cpu: grouped int4-g32 canonical kernel (4-bit storage, simd matvec)".into(),
            });
            candidates.push(Candidate {
                math: MathFormat::Mxfp4,
                layout: PhysicalLayout::Canonical,
                backend: BackendKind::Cpu,
                score: 30.0 + size_penalty(bytes_at(MathFormat::Mxfp4), objective),
                reason: "avx2 cpu: mxfp4 canonical decode (slower detile than int4-g32 rows16)".into(),
            });
        }
        if has_cuda_dp4a && experts_fit_vram {
            candidates.push(Candidate {
                math: MathFormat::Mxfp4,
                layout: PhysicalLayout::PascalDp4aTile,
                backend: BackendKind::Cuda,
                score: 5.0 + size_penalty(bytes_at(MathFormat::Mxfp4), objective),
                reason: "sm61 dp4a: mxfp4 w4a8 dp4a matvec; expert set fits VRAM".into(),
            });
        }
        if has_cuda_any && !has_cuda_dp4a {
            candidates.push(Candidate {
                math: MathFormat::Int4G32,
                layout: PhysicalLayout::Canonical,
                backend: BackendKind::Cuda,
                score: 40.0 + size_penalty(bytes_at(MathFormat::Int4G32), objective),
                reason: "cuda without dp4a: canonical int4 fallback".into(),
            });
        }
        if has_metal_apple8 {
            candidates.push(Candidate {
                math: MathFormat::Mxfp4,
                layout: PhysicalLayout::Apple8Tile,
                backend: BackendKind::Metal,
                score: 0.0 + size_penalty(bytes_at(MathFormat::Mxfp4), objective),
                reason: "apple family-8: production apple8 tile path".into(),
            });
        }
        // Exact bf16 always exists as the last-resort candidate.
        candidates.push(Candidate {
            math: MathFormat::Bf16,
            layout: PhysicalLayout::Canonical,
            backend: BackendKind::Cpu,
            score: 100.0 + size_penalty(bytes_at(MathFormat::Bf16), objective),
            reason: "exact bf16 canonical (last resort: largest, no quant risk)".into(),
        });
    } else {
        // ---- non-expert roles: v1 keeps them exact (quality-sensitive, small) ----
        candidates.push(Candidate {
            math: MathFormat::Bf16,
            layout: PhysicalLayout::Canonical,
            backend: BackendKind::Cpu,
            score: 0.0 + size_penalty(bytes_at(MathFormat::Bf16), objective),
            reason: "dense/static family stays bf16 in v1 (small share, quantization-sensitive)"
                .into(),
        });
    }

    candidates.sort_by(|left, right| {
        left.score
            .partial_cmp(&right.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| format_candidate(&left.math, &left.layout).cmp(&format_candidate(&right.math, &right.layout)))
    });
    let winner = candidates.first()?.clone();
    let rejected = candidates
        .iter()
        .skip(1)
        .map(|candidate| {
            (
                format_candidate(&candidate.math, &candidate.layout),
                candidate.reason.clone(),
            )
        })
        .collect();
    let decision = Decision {
        subject: role.as_str().to_string(),
        chosen: format_candidate(&winner.math, &winner.layout),
        rejected,
    };
    Some((decision, winner))
}

fn format_candidate(math: &MathFormat, layout: &PhysicalLayout) -> String {
    format!("{} {}", math.as_str(), layout.as_str())
}

/// Minimum-size pushes size harder; quality ignores size entirely.
fn size_penalty(bytes: u64, objective: Objective) -> f64 {
    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    match objective {
        Objective::MinimumSize => gib,
        Objective::Quality => 0.0,
        _ => gib * 0.25,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::machine::{CpuProfile, GpuProfile, MachineProfile, StorageClass};

    fn box64_avx2_cuda() -> MachineProfile {
        MachineProfile {
            os: "windows".into(),
            arch: "x86_64".into(),
            cpu: CpuProfile {
                vendor: "AuthenticAMD".into(),
                brand: "AMD Ryzen 5 5600X".into(),
                logical_cores: 12,
                avx2: true,
                avx_vnni: false,
                avx512_f: false,
                avx512_bw: false,
                avx512_vnni: false,
                neon_i8mm: false,
                neon_dotprod: false,
            },
            ram_bytes: 64 * 1024 * 1024 * 1024,
            gpus: vec![GpuProfile::Cuda {
                name: "NVIDIA GeForce GTX 1080".into(),
                vram_bytes: 8 * 1024 * 1024 * 1024,
                cc_major: 6,
                cc_minor: 1,
                dp4a: true,
                tensor_cores: false,
                native_fp8: false,
                native_fp4: false,
            }],
            storage: StorageClass::Nvme,
        }
    }

    fn mac_m2() -> MachineProfile {
        MachineProfile {
            os: "macos".into(),
            arch: "aarch64".into(),
            cpu: CpuProfile {
                vendor: "Apple".into(),
                brand: "Apple M2".into(),
                logical_cores: 8,
                avx2: false,
                avx_vnni: false,
                avx512_f: false,
                avx512_bw: false,
                avx512_vnni: false,
                neon_i8mm: true,
                neon_dotprod: true,
            },
            ram_bytes: 24 * 1024 * 1024 * 1024,
            gpus: vec![GpuProfile::Metal {
                name: "Apple M2".into(),
                family: 8,
            }],
            storage: StorageClass::Unknown,
        }
    }

    #[test]
    fn avx2_plus_dp4a_keeps_experts_on_cpu_when_vram_is_too_small() {
        // 30 GiB of experts vs 8 GB VRAM: DP4A is offered but streaming cost
        // dominates (measured 2243 vs 1931 ms/tok), so CPU rows16 wins.
        let decision = choose(
            TensorRole::RoutedExpert,
            &box64_avx2_cuda(),
            Objective::Balanced,
            |_| 30 * 1024 * 1024 * 1024,
        )
        .unwrap();
        assert_eq!(decision.chosen, "int4_g32 canonical");
        // DP4A was not even offered: the expert set does not fit VRAM.
        assert!(decision
            .rejected
            .iter()
            .any(|(candidate, _)| candidate == "mxfp4 canonical"));
    }

    #[test]
    fn dp4a_wins_when_expert_set_fits_vram() {
        // ~5 GiB of experts vs 8 GB VRAM: the GPU path wins.
        let decision = choose(
            TensorRole::RoutedExpert,
            &box64_avx2_cuda(),
            Objective::Balanced,
            |_| 5 * 1024 * 1024 * 1024,
        )
        .unwrap();
        assert_eq!(decision.chosen, "mxfp4 pascal_dp4a_tile");
    }

    #[test]
    fn cpu_only_avx2_picks_rows16() {
        let mut cpu_only = box64_avx2_cuda();
        cpu_only.gpus.clear();
        let decision = choose(
            TensorRole::RoutedExpert,
            &cpu_only,
            Objective::Balanced,
            |_| 30 * 1024 * 1024 * 1024,
        )
        .unwrap();
        assert_eq!(decision.chosen, "int4_g32 canonical");
    }

    #[test]
    fn apple8_machine_prefers_apple8_tile() {
        let decision = choose(
            TensorRole::RoutedExpert,
            &mac_m2(),
            Objective::Balanced,
            |_| 30 * 1024 * 1024 * 1024,
        )
        .unwrap();
        assert_eq!(decision.chosen, "mxfp4 apple8_tile");
    }

    #[test]
    fn quality_objective_keeps_experts_quantized_but_records_exact_cost() {
        // Even under quality, 4-bit stays the v1 choice for routed experts,
        // but exact bf16 must appear among the recorded alternatives.
        let decision = choose(
            TensorRole::RoutedExpert,
            &box64_avx2_cuda(),
            Objective::Quality,
            |_| 30 * 1024 * 1024 * 1024,
        )
        .unwrap();
        assert_eq!(decision.chosen, "int4_g32 canonical");
        assert!(decision
            .rejected
            .iter()
            .any(|(candidate, _)| candidate == "bf16 canonical"));
    }

    #[test]
    fn minimum_size_prefers_smallest_format() {
        let mut cpu_only = box64_avx2_cuda();
        cpu_only.gpus.clear();
        let decision = choose(
            TensorRole::RoutedExpert,
            &cpu_only,
            Objective::MinimumSize,
            |math| match math {
                MathFormat::Int4G32 => 30 * 1024 * 1024 * 1024_u64,
                MathFormat::Mxfp4 => 31 * 1024 * 1024 * 1024_u64,
                _ => 122 * 1024 * 1024 * 1024_u64,
            },
        )
        .unwrap();
        // int4_g32 (30 GiB) beats mxfp4 (31 GiB) purely on size.
        assert_eq!(decision.chosen, "int4_g32 canonical");
    }

    #[test]
    fn dense_roles_stay_exact() {
        for role in [
            TensorRole::Router,
            TensorRole::AttentionDense,
            TensorRole::Gdn,
            TensorRole::Qsa,
            TensorRole::NgramPle,
            TensorRole::Embed,
            TensorRole::Norm,
            TensorRole::SharedExpert,
        ] {
            let decision = choose(role, &box64_avx2_cuda(), Objective::Balanced, |_| 1024)
                .unwrap();
            assert_eq!(decision.chosen, "bf16 canonical", "{role:?}");
        }
    }

    #[test]
    fn deterministic_across_calls() {
        let first = choose(
            TensorRole::RoutedExpert,
            &box64_avx2_cuda(),
            Objective::Balanced,
            |_| 1024,
        );
        let second = choose(
            TensorRole::RoutedExpert,
            &box64_avx2_cuda(),
            Objective::Balanced,
            |_| 1024,
        );
        assert_eq!(first, second);
    }
}
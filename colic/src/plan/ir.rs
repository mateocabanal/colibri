//! #193 — Physical Model IR: per-tensor format/layout/backend/placement.
//!
//! One `TensorPlan` per semantic tensor family describes HOW it executes.
//! Math format and physical layout stay distinct; a package may carry several
//! representations of the same semantic tensor for different backends.
//! Unsupported (math, layout, backend) combinations fail during planning via
//! `validate()`, never at runtime after emission.
//!
//! GDN/QSA/PLE runtime *state* is deliberately NOT a tensor role — state is
//! accounted by `plan::memory` as planner resources, not placed like weights.

use serde_json::{Value, json};

/// Numerical math format (WHAT the numbers are).
///
/// IDs mirror the COLI manifest math-format identifiers already emitted by
/// the quantizers (exact bf16 tensors = 0x0002, MXFP4 = 0x0020, canonical
/// INT4-G32 = 0x0040, NVFP4 = 0x0041; see quant/*_record.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MathFormat {
    Bf16,
    Fp8E4m3,
    Mxfp4,
    Int4G32,
    Nvfp4,
}

impl MathFormat {
    pub const fn format_id(self) -> u16 {
        match self {
            Self::Bf16 => 0x0002,
            Self::Fp8E4m3 => 0x0010,
            Self::Mxfp4 => 0x0020,
            Self::Int4G32 => 0x0040,
            Self::Nvfp4 => 0x0041,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8_e4m3",
            Self::Mxfp4 => "mxfp4",
            Self::Int4G32 => "int4_g32",
            Self::Nvfp4 => "nvfp4",
        }
    }

    /// Approximate bits per weight including scale overhead, for byte estimates.
    pub const fn bits_per_weight(self) -> f64 {
        match self {
            Self::Bf16 => 16.0,
            Self::Fp8E4m3 => 8.0,
            Self::Mxfp4 => 4.25,
            Self::Int4G32 => 4.25, // 4 bits + f32 scale per 32
            Self::Nvfp4 => 4.5,    // 4 bits + e4m3 scale per 16
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "bf16" => Self::Bf16,
            "fp8_e4m3" => Self::Fp8E4m3,
            "mxfp4" => Self::Mxfp4,
            "int4_g32" => Self::Int4G32,
            "nvfp4" => Self::Nvfp4,
            _ => return None,
        })
    }
}

/// Physical execution layout (HOW the numbers are arranged for a kernel).
///
/// Canonical (0x0000) is the portable row-major record layout. Apple8 tile is
/// the existing 0x0103. Rows16 and the Pascal DP4A tile are new versioned
/// layouts registered alongside the existing ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PhysicalLayout {
    Canonical,
    Apple8Tile,
    Rows16,
    PascalDp4aTile,
}

impl PhysicalLayout {
    pub const fn layout_id(self) -> u16 {
        match self {
            Self::Canonical => 0x0000,
            Self::Apple8Tile => 0x0103,
            Self::Rows16 => 0x0200,
            Self::PascalDp4aTile => 0x0300,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Apple8Tile => "apple8_tile",
            Self::Rows16 => "rows16",
            Self::PascalDp4aTile => "pascal_dp4a_tile",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "canonical" => Self::Canonical,
            "apple8_tile" => Self::Apple8Tile,
            "rows16" => Self::Rows16,
            "pascal_dp4a_tile" => Self::PascalDp4aTile,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
}

impl BackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "cpu" => Self::Cpu,
            "metal" => Self::Metal,
            "cuda" => Self::Cuda,
            _ => return None,
        })
    }
}

/// #199 placement classes. Pageable storage is always the backing tier of the
/// cache classes; `Pageable` means "streamed from disk on every use".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Placement {
    ResidentRam,
    ResidentVram,
    RamCache,
    VramCache,
    Pageable,
}

impl Placement {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ResidentRam => "resident_ram",
            Self::ResidentVram => "resident_vram",
            Self::RamCache => "ram_cache",
            Self::VramCache => "vram_cache",
            Self::Pageable => "pageable",
        }
    }
}

/// Semantic tensor family. Dense is split attention-vs-MLP so the planner can
/// treat them differently (attention projections are more quantization-
/// sensitive than router/MLP dense). Expert layers past the main stack (MTP)
/// stay RoutedExpert but carry their own semantic key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorRole {
    RoutedExpert,
    SharedExpert,
    Router,
    AttentionDense,
    MoeDense,
    Gdn,
    Qsa,
    NgramPle,
    Mtp,
    Norm,
    Embed,
    Other,
}

impl TensorRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RoutedExpert => "routed_expert",
            Self::SharedExpert => "shared_expert",
            Self::Router => "router",
            Self::AttentionDense => "attention_dense",
            Self::MoeDense => "moe_dense",
            Self::Gdn => "gdn",
            Self::Qsa => "qsa",
            Self::NgramPle => "ngram_ple",
            Self::Mtp => "mtp",
            Self::Norm => "norm",
            Self::Embed => "embed",
            Self::Other => "other",
        }
    }

    /// Classify a semantic tensor name (canonical or resident spelling).
    /// Order matters: the most specific patterns first.
    pub fn classify(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        if lower.contains("indexer") {
            return Self::Qsa;
        }
        if lower.contains("ngram") || lower.contains("ple") {
            return Self::NgramPle;
        }
        if lower.contains("shared_expert") {
            return Self::SharedExpert;
        }
        if lower.contains("ffn.gate")
            || lower.contains("mlp.gate")
            || lower.ends_with(".gate.weight")
        {
            return Self::Router;
        }
        if lower.contains("linear_attn") || lower.contains("mamba") {
            return Self::Gdn;
        }
        if lower.contains("norm") {
            return Self::Norm;
        }
        if lower.contains("self_attn") || lower.contains(".attn") || lower.contains("attention") {
            return Self::AttentionDense;
        }
        if lower.contains("mlp") || lower.contains("experts") {
            return Self::MoeDense;
        }
        if lower.contains("mtp") {
            return Self::Mtp;
        }
        if lower.contains("embed") || lower.contains("lm_head") {
            return Self::Embed;
        }
        if lower.contains("norm") {
            return Self::Norm;
        }
        Self::Other
    }
}

/// One physical decision for one semantic tensor (family instance).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlan {
    /// Stable semantic identity, e.g. "layers.3.ffn.gate.weight" or the
    /// canonical global name; experts use "layers.N.experts.E".
    pub semantic_key: String,
    pub role: TensorRole,
    pub math: MathFormat,
    pub layout: PhysicalLayout,
    pub backend: BackendKind,
    pub placement: Placement,
    /// Decoded (runtime) bytes and stored (package) bytes.
    pub decoded_bytes: u64,
    pub stored_bytes: u64,
    /// Lowering implementation identifier + version, e.g. "int4_g32_rows16_v1".
    pub lowering_id: String,
}

impl TensorPlan {
    pub fn to_json(&self) -> Value {
        json!({
            "semantic_key": self.semantic_key,
            "role": self.role.as_str(),
            "math": self.math.as_str(),
            "layout": self.layout.as_str(),
            "backend": self.backend.as_str(),
            "placement": self.placement.as_str(),
            "decoded_bytes": self.decoded_bytes,
            "stored_bytes": self.stored_bytes,
            "lowering_id": self.lowering_id,
        })
    }
}

/// A recorded planner decision: what was chosen, what lost, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    /// Role or semantic key the decision applies to.
    pub subject: String,
    pub chosen: String,
    pub rejected: Vec<(String, String)>, // (candidate, reason)
}

impl Decision {
    pub fn to_json(&self) -> Value {
        json!({
            "subject": self.subject,
            "chosen": self.chosen,
            "rejected": self.rejected.iter().map(|(candidate, reason)| json!({
                "candidate": candidate,
                "reason": reason,
            })).collect::<Vec<_>>(),
        })
    }
}

/// The complete physical plan for one compile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalPlan {
    pub plan_schema_version: u64,
    pub tensors: Vec<TensorPlan>,
    pub decisions: Vec<Decision>,
    pub machine: crate::plan::machine::MachineProfile,
    pub objective: String,
    pub context_tokens: u32,
    /// Planner + cost-model versions for reproducibility (#201).
    pub planner_version: u64,
    pub cost_model_version: u64,
    /// Source checkpoint fingerprint the plan was built from (replay pin).
    pub source_fingerprint: Option<String>,
}

pub const PLAN_SCHEMA_VERSION: u64 = 1;
pub const PLANNER_VERSION: u64 = 1;
pub const COST_MODEL_VERSION: u64 = 1;

impl PhysicalPlan {
    /// Fail planning (not runtime) on unsupported combinations.
    pub fn validate(&self) -> Result<(), String> {
        for tensor in &self.tensors {
            let ok = match (tensor.math, tensor.layout, tensor.backend) {
                (MathFormat::Bf16, PhysicalLayout::Canonical, BackendKind::Cpu)
                | (MathFormat::Bf16, PhysicalLayout::Canonical, BackendKind::Metal)
                | (MathFormat::Bf16, PhysicalLayout::Canonical, BackendKind::Cuda) => true,
                (MathFormat::Mxfp4, PhysicalLayout::Apple8Tile, BackendKind::Metal) => true,
                (MathFormat::Mxfp4, PhysicalLayout::Canonical, BackendKind::Cpu) => true,
                (MathFormat::Int4G32, PhysicalLayout::Canonical, BackendKind::Cpu) => true,
                (MathFormat::Int4G32, PhysicalLayout::Rows16, BackendKind::Cpu) => true,
                (MathFormat::Nvfp4, PhysicalLayout::Canonical, BackendKind::Cpu) => true,
                (MathFormat::Mxfp4, PhysicalLayout::PascalDp4aTile, BackendKind::Cuda) => true,
                (MathFormat::Int4G32, PhysicalLayout::Canonical, BackendKind::Cuda) => true,
                (MathFormat::Fp8E4m3, PhysicalLayout::Canonical, BackendKind::Cpu) => true,
                _ => false,
            };
            if !ok {
                return Err(format!(
                    "unsupported physical combination for `{}`: math={} layout={} backend={}",
                    tensor.semantic_key,
                    tensor.math.as_str(),
                    tensor.layout.as_str(),
                    tensor.backend.as_str()
                ));
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "plan_schema_version": self.plan_schema_version,
            "planner_version": self.planner_version,
            "cost_model_version": self.cost_model_version,
            "source_fingerprint": self.source_fingerprint,
            "objective": self.objective,
            "context_tokens": self.context_tokens,
            "machine": self.machine.to_json(),
            "tensors": self.tensors.iter().map(TensorPlan::to_json).collect::<Vec<_>>(),
            "decisions": self.decisions.iter().map(Decision::to_json).collect::<Vec<_>>(),
        })
    }

    /// Aggregate byte totals per backend + placement for the dry-run report.
    pub fn stored_bytes_by_role(&self) -> Vec<(TensorRole, u64)> {
        let mut totals: Vec<(TensorRole, u64)> = Vec::new();
        for tensor in &self.tensors {
            if let Some(entry) = totals.iter_mut().find(|(role, _)| *role == tensor.role) {
                entry.1 += tensor.stored_bytes;
            } else {
                totals.push((tensor.role, tensor.stored_bytes));
            }
        }
        totals.sort_by_key(|(role, _)| *role);
        totals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::machine::{CpuProfile, GpuProfile, MachineProfile, StorageClass};

    fn sample_plan() -> PhysicalPlan {
        PhysicalPlan {
            plan_schema_version: PLAN_SCHEMA_VERSION,
            tensors: vec![
                TensorPlan {
                    semantic_key: "layers.0.experts.7".into(),
                    role: TensorRole::RoutedExpert,
                    math: MathFormat::Int4G32,
                    layout: PhysicalLayout::Rows16,
                    backend: BackendKind::Cpu,
                    placement: Placement::Pageable,
                    decoded_bytes: 1_048_576,
                    stored_bytes: 294_912,
                    lowering_id: "int4_g32_rows16_v1".into(),
                },
                TensorPlan {
                    semantic_key: "layers.0.experts.7".into(),
                    role: TensorRole::RoutedExpert,
                    math: MathFormat::Mxfp4,
                    layout: PhysicalLayout::PascalDp4aTile,
                    backend: BackendKind::Cuda,
                    placement: Placement::VramCache,
                    decoded_bytes: 1_048_576,
                    stored_bytes: 288_512,
                    lowering_id: "pascal_dp4a_v1".into(),
                },
                TensorPlan {
                    semantic_key: "model.embed_tokens.weight".into(),
                    role: TensorRole::Embed,
                    math: MathFormat::Bf16,
                    layout: PhysicalLayout::Canonical,
                    backend: BackendKind::Cpu,
                    placement: Placement::ResidentRam,
                    decoded_bytes: 1_270_419_456,
                    stored_bytes: 1_270_419_456,
                    lowering_id: "exact_v1".into(),
                },
            ],
            decisions: vec![Decision {
                subject: "routed_expert".into(),
                chosen: "cuda mxfp4 pascal_dp4a_tile (hot set)".into(),
                rejected: vec![(
                    "cpu int4_g32 rows16".into(),
                    "per-token PCIe transfer exceeds DP4A compute win".into(),
                )],
            }],
            machine: MachineProfile {
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
            },
            objective: "balanced".into(),
            context_tokens: 32768,
            planner_version: PLANNER_VERSION,
            cost_model_version: COST_MODEL_VERSION,
            source_fingerprint: None,
        }
    }

    #[test]
    fn mixed_cpu_gpu_plan_is_expressible_and_valid() {
        let plan = sample_plan();
        plan.validate().expect("mixed CPU/CUDA plan must validate");
        // Same semantic key, two representations, two backends.
        assert_eq!(plan.tensors[0].semantic_key, plan.tensors[1].semantic_key);
        assert_ne!(plan.tensors[0].backend, plan.tensors[1].backend);
        assert_ne!(plan.tensors[0].math, plan.tensors[1].math);
    }

    #[test]
    fn unsupported_combination_fails_at_planning() {
        let mut plan = sample_plan();
        plan.tensors[2].layout = PhysicalLayout::Apple8Tile;
        let error = plan.validate().unwrap_err();
        assert!(
            error.contains("unsupported physical combination"),
            "{error}"
        );
    }

    #[test]
    fn role_classification_covers_qwen_names() {
        assert_eq!(TensorRole::classify("ffn.gate.weight"), TensorRole::Router);
        assert_eq!(
            TensorRole::classify("ffn.shared_experts.up.weight"),
            TensorRole::SharedExpert
        );
        assert_eq!(
            TensorRole::classify("model.language_model.layers.3.linear_attn.in_proj_qkv.weight"),
            TensorRole::Gdn
        );
        assert_eq!(
            TensorRole::classify("layers.5.self_attn.indexer.index_qk_proj.weight"),
            TensorRole::Qsa
        );
        assert_eq!(
            TensorRole::classify("layers.2.ple.ple_embedding.ngram_embedders.7.weight"),
            TensorRole::NgramPle
        );
        assert_eq!(
            TensorRole::classify("layers.4.mlp.experts.9.down_proj.weight"),
            TensorRole::MoeDense
        );
        assert_eq!(
            TensorRole::classify(
                "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight"
            ),
            TensorRole::Norm
        );
        assert_eq!(
            TensorRole::classify("model.language_model.embed_tokens.weight"),
            TensorRole::Embed
        );
    }

    #[test]
    fn plan_json_round_trips_through_machine_profile() {
        let plan = sample_plan();
        let json = plan.to_json();
        // Machine profile subtree must independently round-trip.
        let machine = MachineProfile::from_json(&json["machine"]).unwrap();
        assert_eq!(machine, plan.machine);
        assert_eq!(json["plan_schema_version"], PLAN_SCHEMA_VERSION);
        assert_eq!(json["cost_model_version"], COST_MODEL_VERSION);
        assert_eq!(json["tensors"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn bytes_roll_up_by_role() {
        let plan = sample_plan();
        let totals = plan.stored_bytes_by_role();
        let experts = totals
            .iter()
            .find(|(role, _)| *role == TensorRole::RoutedExpert)
            .unwrap();
        // Both expert representations count toward the routed_expert total.
        assert_eq!(experts.1, 294_912 + 288_512);
    }
}

//! Planner orchestration: semantic model + machine profile + objective +
//! context -> deterministic PhysicalPlan (#192/#193/#194/#195/#199/#201).
//!
//! Byte estimation reuses the quantizers' own byte math via a shape-driven
//! estimate so planning never reads checkpoint data (fast, dry-run friendly).

use std::collections::BTreeMap;

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    ir::SemanticModel,
    plan::{
        cost::{self, Objective},
        ir::{
            BackendKind, Decision, MathFormat, PhysicalLayout, PhysicalPlan, Placement,
            PLAN_SCHEMA_VERSION, PLANNER_VERSION, COST_MODEL_VERSION, TensorPlan, TensorRole,
        },
        machine::MachineProfile,
        memory::{plan_memory, MemoryPlannerConfig, StateGeometry},
        placement,
    },
};

#[derive(Debug, Clone)]
pub struct PlanRequest {
    pub objective: Objective,
    pub context_tokens: u32,
    pub batch: u32,
    /// `--target auto` may be pinned to an explicit profile name.
    pub target_profile: Option<String>,
}

/// Plan for one semantic tensor family instance.
///
/// `bytes_at` estimates stored bytes for a math format from the tensor's
/// decoded byte size (element count × bits + scale overhead), without
/// touching source files.
fn estimate_stored_bytes(decoded_bytes: u64, math: MathFormat) -> u64 {
    ((decoded_bytes as f64) * math.bits_per_weight() / 16.0).ceil() as u64
}

/// Build the full physical plan. Deterministic: BTreeMap iteration order +
/// rule-table costs; identical inputs produce byte-identical plans.
pub fn build_plan(
    model: &SemanticModel,
    machine: &MachineProfile,
    config: Option<&Value>,
    request: &PlanRequest,
) -> Result<PhysicalPlan> {
    let objective = request.objective;

    // ---- state geometry: from real config when available ----
    let state = match config {
        Some(config) => StateGeometry::from_config_value(config)?,
        None => synthetic_state(model),
    };

    // ---- classify every semantic tensor into families with byte counts ----
    // (role, decoded_bytes) per stable semantic key, in deterministic order.
    let mut families: BTreeMap<String, (TensorRole, u64)> = BTreeMap::new();
    let mut add = |key: String, role: TensorRole, decoded_bytes: u64, families: &mut BTreeMap<String, (TensorRole, u64)>| {
        let entry = families.entry(key).or_insert((role, 0));
        entry.1 += decoded_bytes;
    };
    for (name, tensor) in &model.global_tensors {
        add(
            name.clone(),
            TensorRole::classify(name),
            tensor.len,
            &mut families,
        );
    }
    for (layer, tensors) in &model.layer_static_tensors {
        for (role_name, tensor) in tensors {
            add(
                format!("layers.{layer}.{role_name}"),
                TensorRole::classify(role_name),
                tensor.len,
                &mut families,
            );
        }
    }
    for ((layer, expert), routed) in &model.routed_experts {
        let decoded = routed.gate.source.len + routed.up.source.len + routed.down.source.len;
        add(
            format!("layers.{layer}.experts.{expert}"),
            TensorRole::RoutedExpert,
            decoded,
            &mut families,
        );
    }
    for (name, tensor) in &model.resident_tensors {
        add(
            name.clone(),
            TensorRole::classify(name),
            tensor.len,
            &mut families,
        );
    }

    // ---- aggregate byte helper per role for the cost model ----
    let mut role_bytes: BTreeMap<TensorRole, u64> = BTreeMap::new();
    for (role, bytes) in families.values() {
        *role_bytes.entry(*role).or_default() += bytes;
    }

    // ---- run the cost model once per role ----
    let mut decisions = Vec::new();
    let mut role_choice: BTreeMap<TensorRole, (MathFormat, PhysicalLayout, BackendKind)> =
        BTreeMap::new();
    for (role, bytes) in &role_bytes {
        let decoded = *bytes;
        let (decision, winner) = cost::choose_candidate(*role, machine, objective, |math| {
            estimate_stored_bytes(decoded, math)
        })
        .ok_or_else(|| {
            ColicError::unsupported(
                "target planning",
                format!("no supported representation for role {role:?} on this machine"),
            )
        })?;
        role_choice.insert(*role, (winner.math, winner.layout, winner.backend));
        decisions.push(decision);
    }

    // ---- build tensor plans (one record per family entry) ----
    let mut tensors = Vec::new();
    for (key, (role, decoded_bytes)) in &families {
        let Some((math, layout, backend)) = role_choice.get(role) else {
            continue;
        };
        let stored = estimate_stored_bytes(*decoded_bytes, *math);
        tensors.push(TensorPlan {
            semantic_key: key.clone(),
            role: *role,
            math: *math,
            layout: *layout,
            backend: *backend,
            placement: Placement::Pageable, // placement pass assigns below
            decoded_bytes: *decoded_bytes,
            stored_bytes: stored,
            lowering_id: lowering_id(*math, *layout),
        });
    }

    // ---- memory budgets ----
    // Mandatory-resident = everything except routed experts and the PLE
    // n-gram table (PLE competes for cache capacity in placement: it prefers
    // RAM residency but degrades to pageable rather than failing the plan).
    let dense_resident_bytes: u64 = tensors
        .iter()
        .filter(|tensor| {
            !matches!(tensor.role, TensorRole::RoutedExpert | TensorRole::NgramPle)
        })
        .map(|tensor| tensor.stored_bytes)
        .sum();
    let memory_config = MemoryPlannerConfig {
        context_tokens: request.context_tokens,
        batch: request.batch,
        ..MemoryPlannerConfig::default()
    };
    let budgets = plan_memory(machine, &state, dense_resident_bytes, memory_config);
    if !budgets.fits {
        return Err(ColicError::unsupported(
            "memory planning",
            budgets.failures.join("; "),
        ));
    }

    // ---- placement ----
    let placement_outcome = placement::place(&mut tensors, &budgets);
    decisions.push(Decision {
        subject: "placement".into(),
        chosen: format!(
            "vram_cache={}B ram_cache={}B pageable={}B ple_resident={}",
            placement_outcome.vram_cached_bytes,
            placement_outcome.ram_cached_bytes,
            placement_outcome.pageable_bytes,
            placement_outcome.ple_resident
        ),
        rejected: placement_outcome
            .notes
            .iter()
            .map(|note| (String::from("note"), note.clone()))
            .collect(),
    });

    Ok(PhysicalPlan {
        plan_schema_version: PLAN_SCHEMA_VERSION,
        tensors,
        decisions,
        machine: machine.clone(),
        objective: objective.as_str().to_string(),
        context_tokens: budgets.context_tokens,
        planner_version: PLANNER_VERSION,
        cost_model_version: COST_MODEL_VERSION,
    })
}

fn lowering_id(math: MathFormat, layout: PhysicalLayout) -> String {
    format!("{}_{}_v1", math.as_str(), layout.as_str())
}

/// Synthetic state geometry when no real config is available (tests /
/// configless planning): treat every layer as full attention with the
/// semantic model's head geometry.
fn synthetic_state(model: &SemanticModel) -> StateGeometry {
    StateGeometry {
        layers: model.geometry.layers,
        full_attention_layers: model.geometry.layers,
        kv_heads: model.geometry.num_key_value_heads.max(1),
        head_dim: model.geometry.head_dim,
        kv_dtype_bytes: 2,
        gdn_layers: 0,
        gdn_state_bytes_per_layer: 0,
        qsa_layers: 0,
        qsa_state_bytes_per_layer_at_max_pos: 0,
        max_position_embeddings: 1 << 20,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::machine::{CpuProfile, GpuProfile, StorageClass};

    fn box64() -> MachineProfile {
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

    fn tiny_model() -> SemanticModel {
        use crate::ir::{Matrix, ModelGeometry, RoutedExpert};
        use crate::source::TensorRef;
        let tensor = |len: u64| TensorRef {
            source: "fixture.safetensors".into(),
            offset: 0,
            len,
            dtype: "BF16".into(),
            shape: vec![len],
        };
        let matrix = |rows: u32, cols: u32| Matrix {
            source: tensor(u64::from(rows) * u64::from(cols) * 2),
            rows,
            columns: cols,
            scale: None,
        };
        let mut experts = BTreeMap::new();
        for layer in 0..2 {
            for expert in 0..4 {
                experts.insert(
                    (layer, expert),
                    RoutedExpert {
                        layer,
                        expert,
                        gate: matrix(64, 32),
                        up: matrix(64, 32),
                        down: matrix(32, 64),
                    },
                );
            }
        }
        let mut globals = BTreeMap::new();
        globals.insert("model.embed_tokens.weight".into(), tensor(1000 * 32 * 2));
        let mut layer0 = BTreeMap::new();
        layer0.insert("ffn.gate.weight".into(), tensor(32 * 32 * 2));
        let mut layers = BTreeMap::new();
        layers.insert(0, layer0);
        SemanticModel {
            architecture: crate::ir::Architecture::Qwen3_5MoeMoE,
            geometry: ModelGeometry {
                hidden_size: 32,
                layers: 2,
                routed_experts_per_layer: 4,
                moe_intermediate_size: 64,
                vocab_size: 1000,
                hc_mult: 0,
                num_hash_layers: 0,
                experts_per_token: 2,
                attention_heads: 1,
                head_dim: 32,
                num_key_value_heads: 1,
                linear_key_head_dim: 0,
                q_lora_rank: 0,
                o_groups: 0,
                o_lora_rank: 0,
                index_heads: 0,
                index_head_dim: 0,
                compression_ratios: Vec::new(),
            },
            routed_experts: experts,
            global_tensors: globals,
            layer_static_tensors: layers,
            resident_tensors: BTreeMap::new(),
        }
    }

    #[test]
    fn plan_is_deterministic_and_validates() {
        let model = tiny_model();
        let request = PlanRequest {
            objective: Objective::Balanced,
            context_tokens: 4096,
            batch: 1,
            target_profile: None,
        };
        let first = build_plan(&model, &box64(), None, &request).unwrap();
        let second = build_plan(&model, &box64(), None, &request).unwrap();
        assert_eq!(first, second);
        first.validate().expect("plan must validate");
    }

    #[test]
    fn experts_get_quantized_dense_stays_exact() {
        let model = tiny_model();
        let request = PlanRequest {
            objective: Objective::Balanced,
            context_tokens: 4096,
            batch: 1,
            target_profile: None,
        };
        let plan = build_plan(&model, &box64(), None, &request).unwrap();
        let expert_math: Vec<_> = plan
            .tensors
            .iter()
            .filter(|t| t.role == TensorRole::RoutedExpert)
            .map(|t| t.math)
            .collect();
        assert!(expert_math.iter().all(|math| *math != MathFormat::Bf16));
        let dense: Vec<_> = plan
            .tensors
            .iter()
            .filter(|t| t.role == TensorRole::Embed)
            .map(|t| t.math)
            .collect();
        assert!(dense.iter().all(|math| *math == MathFormat::Bf16));
    }

    #[test]
    fn context_flag_moves_cache_budget() {
        let model = tiny_model();
        let small = build_plan(
            &model,
            &box64(),
            None,
            &PlanRequest {
                objective: Objective::Balanced,
                context_tokens: 2048,
                batch: 1,
                target_profile: None,
            },
        )
        .unwrap();
        // Synthetic state treats all layers as full attention; larger context
        // must shrink (or keep equal) the RAM cache via bigger KV.
        let large = build_plan(
            &model,
            &box64(),
            None,
            &PlanRequest {
                objective: Objective::Balanced,
                context_tokens: 65536,
                batch: 1,
                target_profile: None,
            },
        )
        .unwrap();
        let cache_of = |plan: &PhysicalPlan| {
            plan.decisions
                .iter()
                .find(|d| d.subject == "placement")
                .map(|d| d.chosen.clone())
                .unwrap_or_default()
        };
        let small_ram = cache_of(&small);
        let large_ram = cache_of(&large);
        // Parse "ram_cache=NB".
        let parse_ram = |text: &str| -> u64 {
            text.split_whitespace()
                .find(|token| token.starts_with("ram_cache="))
                .and_then(|token| token.trim_start_matches("ram_cache=").trim_end_matches('B').parse().ok())
                .unwrap_or(0)
        };
        assert!(parse_ram(&large_ram) <= parse_ram(&small_ram));
    }
}
//! #194 — Context-aware RAM/VRAM/KV/state memory planner.
//!
//! Requested context is a first-class compile input. Only layers that actually
//! maintain growing KV are charged full-attention KV; fixed-state GDN layers
//! and QSA indexer buffers are charged their own (context-capped) state.
//! Output budgets are consumed by `plan::placement`, never informational.

use serde_json::{Value, json};

use crate::{
    error::{ColicError, Result},
    plan::machine::MachineProfile,
};

/// Runtime state geometry of the model being compiled (state = planner
/// resource, not tensor roles). Built from the real config via
/// `from_config_value`, or hand-built for synthetic tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateGeometry {
    pub layers: u32,
    pub full_attention_layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Bytes per KV element (bf16 = 2).
    pub kv_dtype_bytes: u32,
    pub gdn_layers: u32,
    /// Fixed (context-independent) recurrent state per GDN layer:
    /// value_heads * key_head_dim * value_head_dim * 4 (fp32 ssm dtype).
    pub gdn_state_bytes_per_layer: u64,
    pub qsa_layers: u32,
    /// QSA indexer key cache per layer at max_pos: idx_kv * idx_dim * 4.
    pub qsa_state_bytes_per_layer_at_max_pos: u64,
    /// Model max position (caps indexer state at plan time).
    pub max_position_embeddings: u32,
}

impl StateGeometry {
    /// Derive state geometry from a Qwen-family `config.json` (text_config
    /// aware). Hybrid split comes from `layer_types` when present, else
    /// `full_attention_interval` (every Nth layer is full attention).
    pub fn from_config_value(config: &Value) -> Result<Self> {
        let text = config.get("text_config").unwrap_or(config);
        let get_u32 = |key: &str| -> Result<u32> {
            let value = text
                .get(key)
                .and_then(Value::as_u64)
                .ok_or_else(|| ColicError::Usage(format!("config missing `{key}`")))?;
            u32::try_from(value).map_err(|_| ColicError::Usage(format!("{key} exceeds u32")))
        };
        let layers = get_u32("num_hidden_layers")?;
        let full_attention_layers = if let Some(types) = text.get("layer_types").and_then(Value::as_array) {
            types
                .iter()
                .filter(|value| {
                    value.as_str().is_some_and(|kind| {
                        kind.contains("full_attention") && !kind.contains("linear")
                    })
                })
                .count() as u32
        } else if let Some(interval) = text.get("full_attention_interval").and_then(Value::as_u64) {
            layers.div_ceil(interval as u32)
        } else {
            // Pure attention model: every layer keeps KV.
            layers
        };
        let gdn_layers = layers.saturating_sub(full_attention_layers);
        let kv_heads = get_u32("num_key_value_heads")?;
        let head_dim = get_u32("head_dim")?;
        let gdn_state_bytes_per_layer = match (
            text.get("linear_num_value_heads").and_then(Value::as_u64),
            text.get("linear_key_head_dim").and_then(Value::as_u64),
            text.get("linear_value_head_dim").and_then(Value::as_u64),
        ) {
            (Some(value_heads), Some(key_dim), Some(value_dim)) => {
                value_heads * key_dim * value_dim * 4
            }
            _ => 0,
        };
        let qsa_state_bytes_per_layer_at_max_pos = match (
            text.get("indexer_kv_heads").and_then(Value::as_u64),
            text.get("indexer_head_dim").and_then(Value::as_u64),
            get_u32("max_position_embeddings")?,
        ) {
            (Some(idx_kv), Some(idx_dim), max_pos) => idx_kv * idx_dim * u64::from(max_pos) * 4,
            _ => 0,
        };
        let max_position_embeddings = get_u32("max_position_embeddings")?;
        Ok(Self {
            layers,
            full_attention_layers,
            kv_heads,
            head_dim,
            kv_dtype_bytes: 2,
            gdn_layers,
            gdn_state_bytes_per_layer,
            qsa_layers: full_attention_layers,
            qsa_state_bytes_per_layer_at_max_pos,
            max_position_embeddings,
        })
    }

    /// Full-attention KV bytes at `context` for `batch` sequences.
    pub fn kv_bytes(&self, context: u32, batch: u32) -> u64 {
        u64::from(self.full_attention_layers)
            * 2 // K and V
            * u64::from(self.kv_heads)
            * u64::from(self.head_dim)
            * u64::from(self.kv_dtype_bytes)
            * u64::from(context)
            * u64::from(batch)
    }

    /// GDN fixed state (context-independent).
    pub fn gdn_state_bytes(&self) -> u64 {
        u64::from(self.gdn_layers) * self.gdn_state_bytes_per_layer
    }

    /// QSA indexer state, capped at the planned context (never the model max).
    pub fn qsa_state_bytes(&self, context: u32) -> u64 {
        let positions = context.min(self.max_position_embeddings) as u64;
        let max_pos = self.max_position_embeddings as u64;
        if max_pos == 0 {
            return 0;
        }
        u64::from(self.qsa_layers) * self.qsa_state_bytes_per_layer_at_max_pos * positions / max_pos
    }
}

/// Deterministic memory budgets consumed by placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBudgets {
    pub context_tokens: u32,
    pub batch: u32,
    pub kv_bytes: u64,
    pub gdn_state_bytes: u64,
    pub qsa_state_bytes: u64,
    pub dense_resident_bytes: u64,
    pub runtime_reserve_bytes: u64,
    pub scratch_bytes: u64,
    pub ram_expert_cache_bytes: u64,
    pub vram_expert_cache_bytes: u64,
    pub headroom_pct: u32,
    /// False when a mandatory component cannot fit — planner fails early.
    pub fits: bool,
    pub failures: Vec<String>,
}

impl MemoryBudgets {
    pub fn to_json(&self) -> Value {
        json!({
            "context_tokens": self.context_tokens,
            "batch": self.batch,
            "kv_bytes": self.kv_bytes,
            "gdn_state_bytes": self.gdn_state_bytes,
            "qsa_state_bytes": self.qsa_state_bytes,
            "dense_resident_bytes": self.dense_resident_bytes,
            "runtime_reserve_bytes": self.runtime_reserve_bytes,
            "scratch_bytes": self.scratch_bytes,
            "ram_expert_cache_bytes": self.ram_expert_cache_bytes,
            "vram_expert_cache_bytes": self.vram_expert_cache_bytes,
            "headroom_pct": self.headroom_pct,
            "fits": self.fits,
            "failures": self.failures,
        })
    }
}

pub const DEFAULT_HEADROOM_PCT: u32 = 10;
/// Runtime reserve per device: activation scratch + loader buffers.
pub const DEFAULT_RUNTIME_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
pub const DEFAULT_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlannerConfig {
    pub context_tokens: u32,
    pub batch: u32,
    pub headroom_pct: u32,
    pub runtime_reserve_bytes: u64,
    pub scratch_bytes: u64,
}

impl Default for MemoryPlannerConfig {
    fn default() -> Self {
        Self {
            context_tokens: 8192,
            batch: 1,
            headroom_pct: DEFAULT_HEADROOM_PCT,
            runtime_reserve_bytes: DEFAULT_RUNTIME_RESERVE_BYTES,
            scratch_bytes: DEFAULT_SCRATCH_BYTES,
        }
    }
}

/// Compute budgets: mandatory state first, then expert-cache capacity from
/// what remains under (1 - headroom) of RAM / primary-GPU VRAM.
pub fn plan_memory(
    machine: &MachineProfile,
    state: &StateGeometry,
    dense_resident_bytes: u64,
    config: MemoryPlannerConfig,
) -> MemoryBudgets {
    let mut failures = Vec::new();
    let context = config.context_tokens.min(state.max_position_embeddings.max(1));
    let kv_bytes = state.kv_bytes(context, config.batch);
    let gdn_state_bytes = state.gdn_state_bytes();
    let qsa_state_bytes = state.qsa_state_bytes(context);

    let host_usable = usable(machine.ram_bytes, config.headroom_pct);
    let mandatory_host = dense_resident_bytes
        + kv_bytes
        + gdn_state_bytes
        + qsa_state_bytes
        + config.runtime_reserve_bytes
        + config.scratch_bytes;
    if mandatory_host > machine.ram_bytes {
        failures.push(format!(
            "host mandatory bytes {mandatory_host} exceed physical RAM {}",
            machine.ram_bytes
        ));
    }
    let ram_expert_cache_bytes = host_usable.saturating_sub(mandatory_host);

    // Primary CUDA GPU (v1: first one; multiple-GPU split is a later planner).
    let vram_expert_cache_bytes = machine
        .cuda_gpus()
        .next()
        .map(|gpu| {
            let Some(crate::plan::machine::GpuProfile::Cuda {
                vram_bytes, ..
            }) = Some(gpu)
            else {
                unreachable!()
            };
            let vram_usable = usable(*vram_bytes, config.headroom_pct);
            let mandatory_vram = config.runtime_reserve_bytes / 2 + config.scratch_bytes / 2;
            vram_usable.saturating_sub(mandatory_vram)
        })
        .unwrap_or(0);

    MemoryBudgets {
        context_tokens: context,
        batch: config.batch,
        kv_bytes,
        gdn_state_bytes,
        qsa_state_bytes,
        dense_resident_bytes,
        runtime_reserve_bytes: config.runtime_reserve_bytes,
        scratch_bytes: config.scratch_bytes,
        ram_expert_cache_bytes,
        vram_expert_cache_bytes,
        headroom_pct: config.headroom_pct,
        fits: failures.is_empty(),
        failures,
    }
}

fn usable(total_bytes: u64, headroom_pct: u32) -> u64 {
    total_bytes.saturating_mul(100 - headroom_pct.min(100) as u64) / 100
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::machine::{CpuProfile, GpuProfile, StorageClass};

    /// Qwen3.8-Flash-Next text geometry (real config, verified on the box).
    fn qwen38_state() -> StateGeometry {
        StateGeometry {
            layers: 48,
            full_attention_layers: 12,
            kv_heads: 2,
            head_dim: 256,
            kv_dtype_bytes: 2,
            gdn_layers: 36,
            gdn_state_bytes_per_layer: 48 * 128 * 128 * 4,
            qsa_layers: 12,
            qsa_state_bytes_per_layer_at_max_pos: 1 * 128 * 262_144 * 4,
            max_position_embeddings: 262_144,
        }
    }

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

    #[test]
    fn qwen_hybrid_kv_only_charges_full_attention_layers() {
        let state = qwen38_state();
        // 12 full-attn layers * 2(k,v) * 2 kv_heads * 256 dim * 2B = 24 KiB/token.
        assert_eq!(state.kv_bytes(1000, 1), 12 * 2 * 2 * 256 * 2 * 1000);
        // GDN state is context-independent: 36 * 48*128*128*4.
        assert_eq!(state.gdn_state_bytes(), 36 * 48 * 128 * 128_u64 * 4);
    }

    #[test]
    fn changing_context_changes_expert_cache_budget() {
        let state = qwen38_state();
        let small = plan_memory(&box64(), &state, 4 * 1024 * 1024 * 1024, MemoryPlannerConfig {
            context_tokens: 8192,
            ..MemoryPlannerConfig::default()
        });
        let large = plan_memory(&box64(), &state, 4 * 1024 * 1024 * 1024, MemoryPlannerConfig {
            context_tokens: 131_072,
            ..MemoryPlannerConfig::default()
        });
        assert!(large.kv_bytes > small.kv_bytes);
        assert!(large.ram_expert_cache_bytes < small.ram_expert_cache_bytes);
        assert!(small.fits && large.fits);
    }

    #[test]
    fn impossible_context_fails_early() {
        let state = qwen38_state();
        // 32 GB RAM box, max context KV alone: 24 KiB/token * 262144 = 6.3 GB
        // fits; instead force failure with a tiny RAM machine.
        let mut tiny = box64();
        tiny.ram_bytes = 1024 * 1024 * 1024;
        let budgets = plan_memory(&tiny, &state, 900 * 1024 * 1024, MemoryPlannerConfig {
            context_tokens: 262_144,
            ..MemoryPlannerConfig::default()
        });
        assert!(!budgets.fits);
        assert!(!budgets.failures.is_empty());
    }

    #[test]
    fn qsa_state_scales_with_context_not_model_max() {
        let state = qwen38_state();
        let small = state.qsa_state_bytes(8192);
        let large = state.qsa_state_bytes(262_144);
        assert!(small < large);
        assert_eq!(large, state.qsa_state_bytes(1_000_000)); // capped at max_pos
    }

    #[test]
    fn config_derivation_matches_qwen38_next() {
        let config: Value = serde_json::from_str(
            r#"{
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
                "num_key_value_heads": 2,
                "head_dim": 256,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "max_position_embeddings": 262144
            }"#,
        )
        .unwrap();
        let state = StateGeometry::from_config_value(&config).unwrap();
        assert_eq!(state.layers, 4);
        // 4-layer sample scaled: 1 of 4 layers is full attention.
        assert_eq!(state.full_attention_layers, 1);
        assert_eq!(state.gdn_layers, 3);
        assert_eq!(state.kv_bytes(1, 1), 2 * 2 * 256 * 2);
        assert_eq!(state.gdn_state_bytes(), 3 * 48 * 128 * 128 * 4);
    }

    #[test]
    fn gpu_budget_accounts_headroom_and_reserve() {
        let budgets = plan_memory(
            &box64(),
            &qwen38_state(),
            4 * 1024 * 1024 * 1024,
            MemoryPlannerConfig::default(),
        );
        // 8 GB * 90% - (256 + 128) MiB reserve ≈ 6.9 GiB of VRAM expert cache.
        assert!(budgets.vram_expert_cache_bytes > 6 * 1024 * 1024 * 1024);
        assert!(budgets.vram_expert_cache_bytes < 8 * 1024 * 1024 * 1024);
    }
}
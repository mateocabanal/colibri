//! Target-independent semantic model representation.

use std::{collections::BTreeMap, path::PathBuf};

use crate::source::TensorRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    DeepSeekV4Flash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathFormat {
    Fp8E4M3,
    MxFp4E2M1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceRepresentation {
    NativeFp8,
    PackedMxFp4Nibbles,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleFormat {
    Ue8m0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// DeepSeek routed experts compute SiLU(gate) * up before the down projection.
    SwiGlu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelGeometry {
    pub hidden_size: u32,
    pub layers: u32,
    pub routed_experts_per_layer: u32,
    pub moe_intermediate_size: u32,
    pub vocab_size: u32,
    pub hc_mult: u32,
    pub num_hash_layers: u32,
    pub experts_per_token: u32,
    pub attention_heads: u32,
    pub head_dim: u32,
    pub q_lora_rank: u32,
    pub o_groups: u32,
    pub o_lora_rank: u32,
    pub index_heads: u32,
    pub index_head_dim: u32,
    pub compression_ratios: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixScale {
    pub source: TensorRef,
    pub format: ScaleFormat,
    pub block_rows: u32,
    pub block_columns: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix {
    pub source: TensorRef,
    pub rows: u32,
    pub columns: u32,
    pub math_format: MathFormat,
    pub source_representation: SourceRepresentation,
    pub scale: Option<MatrixScale>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedExpert {
    pub layer: u32,
    pub expert: u32,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
    pub activation: Activation,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModelAssets {
    /// Source assets are compiler inputs, not runtime tensor records. Keeping the
    /// classified paths here makes the semantic inventory explicit without
    /// introducing payload ownership or physical target layout.
    pub config: Option<PathBuf>,
    pub tokenizer: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticModel {
    pub architecture: Architecture,
    pub geometry: ModelGeometry,
    pub routed_experts: BTreeMap<(u32, u32), RoutedExpert>,
    /// Globally scoped execution roles, keyed by stable semantic names.
    pub global_tensors: BTreeMap<String, TensorRef>,
    /// Per-layer static execution roles, keyed by layer then canonical role name.
    /// Routed experts are stored separately because they are independently pageable.
    pub layer_static_tensors: BTreeMap<u32, BTreeMap<String, TensorRef>>,
    pub assets: ModelAssets,
}

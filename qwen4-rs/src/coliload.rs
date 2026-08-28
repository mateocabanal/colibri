//! Model loading from a `.coli` package via `ColiSource`.
//!
//! Dense matrices stay BF16 bytes (decoded in matmul); small vectors decode
//! to f32 at load (they're tiny); experts and the PLE ngram table are fetched
//! on demand through `Model::coli` (never resident as a whole).

use crate::{
    colisource::{bf16_to_f32, ColiSource},
    Cfg, HcGlobal, Layer, Model, Wt,
};

fn load_wt(src: &ColiSource, name: &str, o: usize, i: usize) -> Result<Wt, String> {
    let m = src.wt(name, o, i)?;
    Ok(Wt {
        f: vec![],
        bytes: Some(m.bytes),
        o: m.o,
        i: m.i,
    })
}

fn vec_f32(src: &ColiSource, name: &str, want: usize) -> Result<Vec<f32>, String> {
    let bytes = src.vec(name, want)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
        .collect())
}

impl Model {
    /// Loads from a `.coli` package. Dense matrices resident as BF16 bytes;
    /// experts + ngram fetched on demand (16 GB M2 budget).
    pub fn load_coli(src: &ColiSource, cfg: &Cfg) -> Result<Model, String> {
        let mut cfg = cfg.clone();
        // Bring up the Metal backend once (experts GEMV via FFI).
        crate::ffi::metal_init();
        // Package is ground truth for the PLE layer (frontend resolution can
        // differ from config ple_layer_ids; live: config says 2, package has
        // layers.1.ple.*).
        if let Some(pl) = src.ple_layer() {
            cfg.ple_layer = pl as i64;
        }
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let lp = format!("layers.{l}");
            let is_gdn = cfg.gdn_layers[l];
            let is_qsa = cfg.qsa_layers[l];
            let cdim = cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads;
            let vdim = cfg.lin_v_dim * cfg.lin_v_heads;
            let hd = cfg.head_dim;
            let hcd = cfg.hc_count * cfg.hidden;
            let in_ln: Vec<f32> = if cfg.hc_count > 0 {
                vec![]
            } else {
                vec_f32(src, &format!("{lp}.input_layernorm.weight"), cfg.hidden)?
            };
            let empty = || Wt { f: vec![], bytes: None, o: 0, i: 0 };
            let layer = Layer {
                in_ln,
                is_gdn,
                is_qsa,
                gdn_a_log: if is_gdn { vec_f32(src, &format!("{lp}.linear_attn.A_log"), cfg.lin_v_heads)? } else { vec![] },
                gdn_dt_bias: if is_gdn { vec_f32(src, &format!("{lp}.linear_attn.dt_bias"), cfg.lin_v_heads)? } else { vec![] },
                gdn_conv1d: if is_gdn { vec_f32(src, &format!("{lp}.linear_attn.conv1d.weight"), cdim * cfg.conv_kernel)? } else { vec![] },
                gdn_in_a: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.in_proj_a.weight"), cfg.lin_v_heads, cfg.hidden)? } else { empty() },
                gdn_in_b: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.in_proj_b.weight"), cfg.lin_v_heads, cfg.hidden)? } else { empty() },
                gdn_in_qkv: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.in_proj_qkv.weight"), cdim, cfg.hidden)? } else { empty() },
                gdn_in_z: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.in_proj_z.weight"), vdim, cfg.hidden)? } else { empty() },
                gdn_norm: if is_gdn { vec_f32(src, &format!("{lp}.linear_attn.norm.weight"), cfg.lin_v_dim)? } else { vec![] },
                gdn_out: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.out_proj.weight"), cfg.hidden, vdim)? } else { empty() },
                attn_q: if !is_gdn { load_wt(src, &format!("{lp}.self_attn.q_proj.weight"), 2 * cfg.heads * hd, cfg.hidden)? } else { empty() },
                attn_k: if !is_gdn { load_wt(src, &format!("{lp}.self_attn.k_proj.weight"), cfg.kv_heads * hd, cfg.hidden)? } else { empty() },
                attn_v: if !is_gdn { load_wt(src, &format!("{lp}.self_attn.v_proj.weight"), cfg.kv_heads * hd, cfg.hidden)? } else { empty() },
                attn_o: if !is_gdn { load_wt(src, &format!("{lp}.self_attn.o_proj.weight"), cfg.hidden, cfg.heads * hd)? } else { empty() },
                attn_qn: if !is_gdn { vec_f32(src, &format!("{lp}.self_attn.q_norm.weight"), hd)? } else { vec![] },
                attn_kn: if !is_gdn { vec_f32(src, &format!("{lp}.self_attn.k_norm.weight"), hd)? } else { vec![] },
                index_qk: if is_qsa {
                    load_wt(src, &format!("{lp}.self_attn.indexer.index_qk_proj.weight"), cfg.idx_n_heads * cfg.idx_head_dim + cfg.idx_kv_heads * cfg.idx_head_dim, cfg.hidden)?
                } else {
                    empty()
                },
                idx_qn: if is_qsa { vec_f32(src, &format!("{lp}.self_attn.indexer.q_layernorm.weight"), cfg.idx_head_dim)? } else { vec![] },
                idx_kn: if is_qsa { vec_f32(src, &format!("{lp}.self_attn.indexer.k_layernorm.weight"), cfg.idx_head_dim)? } else { vec![] },
                hc_norm: vec_f32(src, &format!("{lp}.attn_hyper_connection.hc_norm.weight"), hcd)?,
                hc_mix_down: load_wt(src, &format!("{lp}.attn_hyper_connection.input_mix_weight_down.weight"), cfg.hc_lowrank, hcd)?,
                hc_mix_up: load_wt(src, &format!("{lp}.attn_hyper_connection.input_mix_weight_up.weight"), hcd, cfg.hc_lowrank)?,
                hc_inject: load_wt(src, &format!("{lp}.attn_hyper_connection.block_inject_weight.weight"), cfg.hc_count, hcd)?,
                hc_mlp_norm: vec_f32(src, &format!("{lp}.mlp_hyper_connection.hc_norm.weight"), hcd)?,
                hc_mlp_mix_down: load_wt(src, &format!("{lp}.mlp_hyper_connection.input_mix_weight_down.weight"), cfg.hc_lowrank, hcd)?,
                hc_mlp_mix_up: load_wt(src, &format!("{lp}.mlp_hyper_connection.input_mix_weight_up.weight"), hcd, cfg.hc_lowrank)?,
                hc_mlp_inject: load_wt(src, &format!("{lp}.mlp_hyper_connection.block_inject_weight.weight"), cfg.hc_count, hcd)?,
                router: load_wt(src, &format!("{lp}.mlp.gate.weight"), cfg.experts, cfg.hidden)?,
                se_gate: load_wt(src, &format!("{lp}.mlp.shared_expert.gate_proj.weight"), cfg.shared_inter, cfg.hidden)?,
                se_up: load_wt(src, &format!("{lp}.mlp.shared_expert.up_proj.weight"), cfg.shared_inter, cfg.hidden)?,
                se_down: load_wt(src, &format!("{lp}.mlp.shared_expert.down_proj.weight"), cfg.hidden, cfg.shared_inter)?,
                se_g: load_wt(src, &format!("{lp}.mlp.shared_expert_gate.weight"), 1, cfg.hidden)?,
            };
            layers.push(layer);
        }

        // PLE geometry: the package carries ground-truth vocab_sizes/offsets/
        // multipliers (i64 records). The config-derived prime math diverges
        // on the real model (row 173M vs computed 160M), so .coli mode reads
        // the package values; safetensors mode keeps the prime computation.
        let (mut ple_sizes, mut ple_offsets, mut ple_mult) = if cfg.ple_layer >= 0 {
            let (sizes, offsets, mult) = src.ple_metadata(cfg.ple_layer as i32)?;
            (sizes, offsets, mult)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        if ple_sizes.is_empty() && cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let mut total = 0_i64;
            for h in 0..cfg.ngram_heads {
                let size = crate::nth_prime_after(cfg.ngram_vocab_base - 1, h as i64 + 1);
                ple_sizes.push(size);
                ple_offsets.push(total);
                total += size;
            }
        }
        let hcd = cfg.hc_count * cfg.hidden;
        let ple_key_proj = if cfg.ple_layer >= 0 {
            load_wt(src, &format!("layers.{}.ple.key_proj.weight", cfg.ple_layer), cfg.hc_count * cfg.hidden, cfg.ple_embed_dim)?
        } else {
            Wt { f: vec![], bytes: None, o: 0, i: 0 }
        };
        let ple_value_proj = if cfg.ple_layer >= 0 {
            load_wt(src, &format!("layers.{}.ple.value_proj.weight", cfg.ple_layer), cfg.hidden, cfg.ple_embed_dim)?
        } else {
            Wt { f: vec![], bytes: None, o: 0, i: 0 }
        };
        let ple_norm_key = if cfg.ple_layer >= 0 { vec_f32(src, &format!("layers.{}.ple.norm_key.weight", cfg.ple_layer), hcd)? } else { vec![] };
        let ple_norm_query = if cfg.ple_layer >= 0 { vec_f32(src, &format!("layers.{}.ple.norm_query.weight", cfg.ple_layer), hcd)? } else { vec![] };
        let ple_norm_conv = if cfg.ple_layer >= 0 { vec_f32(src, &format!("layers.{}.ple.norm_conv.weight", cfg.ple_layer), hcd)? } else { vec![] };
        let ple_conv1d = if cfg.ple_layer >= 0 { vec_f32(src, &format!("layers.{}.ple.conv1d.weight", cfg.ple_layer), hcd * cfg.ple_conv_kernel)? } else { vec![] };

        let final_norm = if cfg.hc_count > 0 && src.rec("norm.weight").is_none() {
            vec![]
        } else {
            vec_f32(src, "norm.weight", cfg.hidden)?
        };

        // C-style one-line status (QWEN-APPLE8 / metalio parity with the C
        // engine's startup banner): makes silent-fallback visible.
        if crate::ffi::direct_available() {
            eprintln!(
                "[qwen4-rs] direct raw Apple8 + MetalIO execution enabled{}",
                if crate::ffi::metal_available() { "" } else { " (compute backend unavailable)" }
            );
        } else {
            eprintln!("[qwen4-rs] direct path unavailable; using canonical fallback");
        }
        Ok(Model {
            cfg: cfg.clone(),
            coli: Some(src.clone()),
            embed: load_wt(src, "embed.weight", cfg.vocab, cfg.hidden)?,
            lm_head: load_wt(src, "head.weight", cfg.vocab, cfg.hidden)?,
            final_norm,
            layers,
            experts: Vec::new(), // fetched on demand via coli
            hc_global: HcGlobal {
                norm: vec_f32(src, "hyper_connection_mixer.hc_norm.weight", hcd)?,
                mix_down: load_wt(src, "hyper_connection_mixer.input_mix_weight_down.weight", cfg.hc_lowrank, hcd)?,
                mix_up: load_wt(src, "hyper_connection_mixer.input_mix_weight_up.weight", hcd, cfg.hc_lowrank)?,
            },
            ple_ngram: Wt { f: vec![], bytes: None, o: 0, i: 0 }, // rows fetched via coli
            ple_key_proj,
            ple_value_proj,
            ple_norm_key,
            ple_norm_query,
            ple_norm_conv,
            ple_conv1d,
            ple_offsets,
            ple_sizes,
            ple_mult,
            gdn_conv: vec![
                vec![0.0; (cdim_total(&cfg)) * cfg.conv_kernel.saturating_sub(1)];
                cfg.layers
            ],
            gdn_s: vec![vec![0.0; cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim]; cfg.layers],
            kv_k: vec![0.0; cfg.layers * cfg.kv_heads * cfg.max_t * cfg.head_dim],
            kv_v: vec![0.0; cfg.layers * cfg.kv_heads * cfg.max_t * cfg.head_dim],
            idx_cache: vec![vec![0.0; cfg.max_t * cfg.idx_kv_heads * cfg.idx_head_dim]; cfg.layers],
            ple_ring: vec![cfg.eos; cfg.ngram_size.max(1)],
            ple_conv_state: vec![
                0.0;
                hcd * ((cfg.ple_conv_kernel - 1) * cfg.ngram_size + 1).max(1)
            ],
            expert_cache: std::collections::VecDeque::new(),
            expert_cache_cap: 256,
            metal_direct: crate::ffi::direct_init()
                && std::env::var("QWEN_APPLE8_DIRECT")
                    .map(|v| v != "0")
                    .unwrap_or(true),
            metal_overlap: std::env::var("QWEN_APPLE8_OVERLAP")
                .map(|v| v != "0")
                .unwrap_or(true),
            gdn_metal: (0..cfg.layers).map(|_| None).collect(),
        })
    }
}

fn cdim_total(cfg: &Cfg) -> usize {
    cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads
}

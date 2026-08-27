//! Model loading from a `.coli` package via `ColiSource` (Apple8/MXFP4
//! experts, BF16 dense). Mirrors `Model::load` (safetensors) exactly — same
//! struct, same geometry, only the weight source differs.

use crate::{
    colisource::{ColiSource, ColiWt},
    Cfg, HcGlobal, Layer, Model, Wt,
};

fn load_wt(src: &ColiSource, name: &str, o: usize, i: usize) -> Result<Wt, String> {
    let ColiWt { f, o: _, i: _ } = src.wt(name, o, i)?;
    Ok(Wt { f, o, i })
}

impl Model {
    /// Loads from a `.coli` package (manifest.coli + data shards). Dense and
    /// layer-static tensors resolve by canonical name (HF-prefix fallback,
    /// like the C runtime); experts decode Apple8 MXFP4 tiles or BF16.
    pub fn load_coli(src: &ColiSource, cfg: &Cfg) -> Result<Model, String> {
        let mut experts = Vec::new();
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let lp = format!("layers.{l}");
            let is_gdn = cfg.gdn_layers[l];
            let is_qsa = cfg.qsa_layers[l];
            let cdim = cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads;
            let vdim = cfg.lin_v_dim * cfg.lin_v_heads;
            let hd = cfg.head_dim;
            let hcd = cfg.hc_count * cfg.hidden;
            // qwen4 hc path: no per-layer input/post norms (hc_mix normalizes)
            let in_ln: Vec<f32> = if cfg.hc_count > 0 {
                vec![]
            } else {
                src.vec(&format!("{lp}.input_layernorm.weight"), cfg.hidden)?
            };
            let layer = Layer {
                in_ln,
                is_gdn,
                is_qsa,
                gdn_a_log: if is_gdn { src.vec(&format!("{lp}.linear_attn.A_log"), cfg.lin_v_heads)? } else { vec![] },
                gdn_dt_bias: if is_gdn { src.vec(&format!("{lp}.linear_attn.dt_bias"), cfg.lin_v_heads)? } else { vec![] },
                gdn_conv1d: if is_gdn { src.vec(&format!("{lp}.linear_attn.conv1d.weight"), cdim * cfg.conv_kernel)? } else { vec![] },
                gdn_in_a: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.in_proj_a.weight"), cfg.lin_v_heads, cfg.hidden)? } else { Wt { f: vec![], o: 0, i: 0 } },
                gdn_in_b: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.in_proj_b.weight"), cfg.lin_v_heads, cfg.hidden)? } else { Wt { f: vec![], o: 0, i: 0 } },
                gdn_in_qkv: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.in_proj_qkv.weight"), cdim, cfg.hidden)? } else { Wt { f: vec![], o: 0, i: 0 } },
                gdn_in_z: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.in_proj_z.weight"), vdim, cfg.hidden)? } else { Wt { f: vec![], o: 0, i: 0 } },
                gdn_norm: if is_gdn { src.vec(&format!("{lp}.linear_attn.norm.weight"), cfg.lin_v_dim)? } else { vec![] },
                gdn_out: if is_gdn { load_wt(src, &format!("{lp}.linear_attn.out_proj.weight"), cfg.hidden, vdim)? } else { Wt { f: vec![], o: 0, i: 0 } },
                attn_q: if !is_gdn { load_wt(src, &format!("{lp}.attn.q_proj.weight"), 2 * cfg.heads * hd, cfg.hidden)? } else { Wt { f: vec![], o: 0, i: 0 } },
                attn_k: if !is_gdn { load_wt(src, &format!("{lp}.attn.k_proj.weight"), cfg.kv_heads * hd, cfg.hidden)? } else { Wt { f: vec![], o: 0, i: 0 } },
                attn_v: if !is_gdn { load_wt(src, &format!("{lp}.attn.v_proj.weight"), cfg.kv_heads * hd, cfg.hidden)? } else { Wt { f: vec![], o: 0, i: 0 } },
                attn_o: if !is_gdn { load_wt(src, &format!("{lp}.attn.o_proj.weight"), cfg.hidden, cfg.heads * hd)? } else { Wt { f: vec![], o: 0, i: 0 } },
                attn_qn: if !is_gdn { src.vec(&format!("{lp}.attn.q_norm.weight"), hd)? } else { vec![] },
                attn_kn: if !is_gdn { src.vec(&format!("{lp}.attn.k_norm.weight"), hd)? } else { vec![] },
                index_qk: if is_qsa {
                    load_wt(src, &format!("{lp}.attn.indexer.index_qk_proj.weight"), cfg.idx_n_heads * cfg.idx_head_dim + cfg.idx_kv_heads * cfg.idx_head_dim, cfg.hidden)?
                } else {
                    Wt { f: vec![], o: 0, i: 0 }
                },
                idx_qn: if is_qsa { src.vec(&format!("{lp}.attn.indexer.q_layernorm.weight"), cfg.idx_head_dim)? } else { vec![] },
                idx_kn: if is_qsa { src.vec(&format!("{lp}.attn.indexer.k_layernorm.weight"), cfg.idx_head_dim)? } else { vec![] },
                hc_norm: src.vec(&format!("{lp}.attn_hyper_connection.hc_norm.weight"), hcd)?,
                hc_mix_down: load_wt(src, &format!("{lp}.attn_hyper_connection.input_mix_weight_down.weight"), cfg.hc_lowrank, hcd)?,
                hc_mix_up: load_wt(src, &format!("{lp}.attn_hyper_connection.input_mix_weight_up.weight"), hcd, cfg.hc_lowrank)?,
                hc_inject: load_wt(src, &format!("{lp}.attn_hyper_connection.block_inject_weight.weight"), cfg.hc_count, hcd)?,
                hc_mlp_norm: src.vec(&format!("{lp}.mlp_hyper_connection.hc_norm.weight"), hcd)?,
                hc_mlp_mix_down: load_wt(src, &format!("{lp}.mlp_hyper_connection.input_mix_weight_down.weight"), cfg.hc_lowrank, hcd)?,
                hc_mlp_mix_up: load_wt(src, &format!("{lp}.mlp_hyper_connection.input_mix_weight_up.weight"), hcd, cfg.hc_lowrank)?,
                hc_mlp_inject: load_wt(src, &format!("{lp}.mlp_hyper_connection.block_inject_weight.weight"), cfg.hc_count, hcd)?,
                router: load_wt(src, &format!("{lp}.ffn.gate.weight"), cfg.experts, cfg.hidden)?,
                se_gate: load_wt(src, &format!("{lp}.ffn.shared_experts.gate_proj.weight"), cfg.shared_inter, cfg.hidden)?,
                se_up: load_wt(src, &format!("{lp}.ffn.shared_experts.up.weight"), cfg.shared_inter, cfg.hidden)?,
                se_down: load_wt(src, &format!("{lp}.ffn.shared_experts.down.weight"), cfg.hidden, cfg.shared_inter)?,
                se_g: load_wt(src, &format!("{lp}.ffn.shared_experts.gate.weight"), 1, cfg.hidden)?,
            };
            let mut layer_experts = Vec::new();
            for e in 0..cfg.experts {
                let mats = src.expert_matrices(l as i32, e as i32)?;
                layer_experts.push(mats.map(|m| Wt { f: m.f, o: m.o, i: m.i }));
            }
            experts.push(layer_experts);
            layers.push(layer);
        }

        // PLE geometry (same as safetensors load)
        let mut ple_offsets = Vec::new();
        let mut ple_sizes = Vec::new();
        if cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let mut total = 0_i64;
            for h in 0..cfg.ngram_heads {
                let size = crate::nth_prime_after(cfg.ngram_vocab_base - 1, h as i64 + 1);
                ple_sizes.push(size);
                ple_offsets.push(total);
                total += size;
            }
        }
        let hcd = cfg.hc_count * cfg.hidden;
        let ple_embed: Wt = if cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let total: i64 = ple_sizes.iter().sum();
            let padded = (total + cfg.ngram_div - 1) / cfg.ngram_div * cfg.ngram_div;
            let hd_per = cfg.ple_embed_dim / cfg.ngram_heads;
            load_wt(src, "ple.ple_embedding.ngram_embedding.weight", padded as usize, hd_per)?
        } else {
            Wt { f: vec![], o: 0, i: 0 }
        };
        let ple_key_proj = if cfg.ple_layer >= 0 {
            load_wt(src, "ple.key_proj.weight", cfg.hc_count * cfg.hidden, cfg.ple_embed_dim)?
        } else {
            Wt { f: vec![], o: 0, i: 0 }
        };
        let ple_value_proj = if cfg.ple_layer >= 0 {
            load_wt(src, "ple.value_proj.weight", cfg.hidden, cfg.ple_embed_dim)?
        } else {
            Wt { f: vec![], o: 0, i: 0 }
        };
        let ple_norm_key = if cfg.ple_layer >= 0 { src.vec("ple.norm_key.weight", hcd)? } else { vec![] };
        let ple_norm_query = if cfg.ple_layer >= 0 { src.vec("ple.norm_query.weight", hcd)? } else { vec![] };
        let ple_norm_conv = if cfg.ple_layer >= 0 { src.vec("ple.norm_conv.weight", hcd)? } else { vec![] };
        let ple_conv1d = if cfg.ple_layer >= 0 {
            src.vec("ple.conv1d.weight", hcd * cfg.ple_conv_kernel)?
        } else {
            vec![]
        };
        let mut ple_mult = Vec::new();
        if cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let max_long = i64::MAX;
            let mult_max = max_long / (cfg.vocab.max(1) as i64);
            let half = (mult_max / 2).max(1);
            let base = cfg.seed as u64 + 10007_u64.wrapping_mul(cfg.ple_layer as u64);
            for i in 0..cfg.ngram_size {
                let v = crate::ple_splitmix64(base.wrapping_add(crate::PLE_GAMMA.wrapping_mul((i + 1) as u64)));
                ple_mult.push(2 * (v % half as u64) + 1);
            }
        }

        let final_norm = match src.vec("norm.weight", cfg.hidden) {
            Ok(v) => v,
            Err(_) if cfg.hc_count > 0 => vec![],
            Err(e) => return Err(e),
        };

        Ok(Model {
            cfg: cfg.clone(),
            embed: load_wt(src, "embed.weight", cfg.vocab, cfg.hidden)?,
            lm_head: load_wt(src, "head.weight", cfg.vocab, cfg.hidden)?,
            final_norm,
            layers,
            experts,
            hc_global: HcGlobal {
                norm: src.vec("hyper_connection_mixer.hc_norm.weight", hcd)?,
                mix_down: load_wt(src, "hyper_connection_mixer.input_mix_weight_down.weight", cfg.hc_lowrank, hcd)?,
                mix_up: load_wt(src, "hyper_connection_mixer.input_mix_weight_up.weight", hcd, cfg.hc_lowrank)?,
            },
            ple_ngram: ple_embed,
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
                vec![0.0; (cdim_total(cfg)) * cfg.conv_kernel.saturating_sub(1)];
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
        })
    }
}

fn cdim_total(cfg: &Cfg) -> usize {
    cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads
}

//! Tiny Qwen MoE scalar reference (Rust rewrite walking skeleton, RW-041).
//!
//! Verbatim math port of the C engine's per-token path (`c/qwen_moe_base.inc`):
//! embed -> per-layer [rmsnorm -> (GDN | full-attn) -> residual] ->
//! [rmsnorm -> MoE (router top-k + routed experts + shared expert) -> residual]
//! -> final rmsnorm -> lm_head logits. Greedy decode then picks argmax.
//!
//! Numerics are deliberately C-identical: f32 accumulators, `1/sqrtf`,
//! `expf`, `powf`, exact reduction order, top-k by probability with
//! lower-index tie-break and renormalized weights. Token-identity gate is
//! `ref.json` `greedy_new_ids` (plan §12.2).

use std::path::Path;

// ---------------------------------------------------------------------------
// tiny safetensors reader (RW-015 legacy-safetensors adapter, minimal form)
// ---------------------------------------------------------------------------

pub struct StFile {
    data: Vec<u8>,
    tensors: std::collections::HashMap<String, (Vec<u64>, usize, usize)>, // name -> (shape, offset, len)
}

impl StFile {
    pub fn open(path: &Path) -> Result<StFile, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + n as usize]).map_err(|e| e.to_string())?;
        let obj = header.as_object().unwrap();
        let data_start = 8 + n as usize;
        let mut tensors = std::collections::HashMap::new();
        for (name, spec) in obj {
            let dtype = spec["dtype"].as_str().unwrap().to_string();
            let shape: Vec<u64> = spec["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            let offs = spec["data_offsets"].as_array().unwrap();
            let offset = offs[0].as_u64().unwrap() as usize;
            let len = offs[1].as_u64().unwrap() as usize - offset;
            if dtype != "F32" {
                return Err(format!("{name}: only F32 supported, got {dtype}"));
            }
            tensors.insert(name.clone(), (shape, data_start + offset, len));
        }
        Ok(StFile {
            data: bytes,
            tensors,
        })
    }

    fn f32(&self, name: &str, expect: &[u64]) -> Result<Vec<f32>, String> {
        let (shape, offset, len) = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let want: u64 = expect.iter().product();
        let have: u64 = shape.iter().product();
        if have != want || *len != want as usize * 4 {
            return Err(format!(
                "{name}: shape {shape:?} len {} != expected {expect:?}",
                *len
            ));
        }
        Ok(self.data[*offset..*offset + *len]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Cfg {
    pub hidden: usize,
    pub layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    pub experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    lin_k_heads: usize,
    lin_k_dim: usize,
    lin_v_heads: usize,
    lin_v_dim: usize,
    conv_kernel: usize,
    max_t: usize,
    pub vocab: usize,
    pub eps: f32,
    gdn_layers: Vec<bool>,
}

pub fn load_cfg(path: &Path) -> Result<Cfg, String> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?).unwrap();
    let get = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let num = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
    let rope = v.get("rope_parameters");
    let theta = rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or_else(|| num("rope_theta").max(10000000.0));
    let prf = rope
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(1.0);
    let head_dim = get("head_dim").max(get("hidden_size") / get("num_attention_heads").max(1));
    let layer_types = v
        .get("layer_types")
        .and_then(|x| x.as_array())
        .cloned()
        .unwrap_or_default();
    let gdn_layers: Vec<bool> = layer_types
        .iter()
        .map(|t| t.as_str() == Some("linear_attention"))
        .collect();
    let cfg = Cfg {
        hidden: get("hidden_size"),
        layers: get("num_hidden_layers"),
        heads: get("num_attention_heads"),
        kv_heads: get("num_key_value_heads"),
        head_dim,
        rotary_dim: (head_dim as f32 * prf) as usize,
        theta,
        experts: get("num_experts"),
        topk: get("num_experts_per_tok"),
        moe_inter: get("moe_intermediate_size"),
        shared_inter: get("shared_expert_intermediate_size"),
        lin_k_heads: get("linear_num_key_heads"),
        lin_k_dim: get("linear_key_head_dim"),
        lin_v_heads: get("linear_num_value_heads"),
        lin_v_dim: get("linear_value_head_dim"),
        conv_kernel: get("linear_conv_kernel_dim"),
        max_t: get("max_position_embeddings"),
        vocab: get("vocab_size"),
        eps: num("rms_norm_eps").max(1e-6),
        gdn_layers,
    };
    if cfg.layers != cfg.gdn_layers.len() {
        return Err(format!(
            "layer_types {} != num_hidden_layers {}",
            cfg.gdn_layers.len(),
            cfg.layers
        ));
    }
    Ok(cfg)
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Wt {
    f: Vec<f32>,
    o: usize,
    i: usize,
}

#[derive(Clone)]
struct Layer {
    in_ln: Vec<f32>,
    post_ln: Vec<f32>,
    is_gdn: bool,
    // GDN
    gdn_a_log: Vec<f32>,
    gdn_dt_bias: Vec<f32>,
    gdn_conv1d: Vec<f32>,
    gdn_in_a: Wt,
    gdn_in_b: Wt,
    gdn_in_qkv: Wt,
    gdn_in_z: Wt,
    gdn_norm: Vec<f32>,
    gdn_out: Wt,
    // full attention
    attn_q: Wt,
    attn_k: Wt,
    attn_v: Wt,
    attn_o: Wt,
    attn_qn: Vec<f32>,
    attn_kn: Vec<f32>,
    // MoE
    router: Wt,
    se_gate: Wt,
    se_up: Wt,
    se_down: Wt,
    se_g: Wt,
}

pub struct Model {
    cfg: Cfg,
    embed: Wt,
    lm_head: Wt,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    experts: Vec<Vec<[Wt; 3]>>, // [layer][expert] -> [gate, up, down]
    // state
    gdn_conv: Vec<Vec<f32>>, // [layer][C*(kk-1)]
    gdn_s: Vec<Vec<f32>>,    // [layer][vheads*kdim*vdim]
    kv_k: Vec<f32>,          // flat [layer][kv_heads][max_t][hd]
    kv_v: Vec<f32>,
}

// ---------------------------------------------------------------------------
// C-identical math helpers
// ---------------------------------------------------------------------------

fn matmul(y: &mut [f32], x: &[f32], w: &Wt) {
    // S=1 only here. C: for o in 0..O: acc = sum_i x[i]*w[o*I+i]
    let (o, i) = (w.o, w.i);
    for oo in 0..o {
        let mut acc = 0.0_f32;
        for ii in 0..i {
            acc += x[ii] * w.f[oo * i + ii];
        }
        y[oo] = acc;
    }
}

fn rmsnorm_row(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let d = x.len();
    let mut ms = 0.0_f64;
    for i in 0..d {
        ms += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (ms as f32 / d as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = x[i] * r * (1.0 + w[i]);
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn rmsnorm_gated_row(out: &mut [f32], x: &[f32], z: &[f32], w: &[f32], eps: f32) {
    let d = x.len();
    let mut ms = 0.0_f64;
    for i in 0..d {
        ms += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (ms as f32 / d as f32 + eps).sqrt();
    for i in 0..d {
        out[i] = w[i] * (x[i] * r) * silu(z[i]);
    }
}

fn softmax_row(x: &mut [f32]) {
    let n = x.len();
    let mut m = -1e30_f32;
    for i in 0..n {
        if x[i] > m {
            m = x[i];
        }
    }
    let mut s = 0.0_f32;
    for i in 0..n {
        x[i] = (x[i] - m).exp();
        s += x[i];
    }
    for i in 0..n {
        x[i] /= s;
    }
}

fn l2norm(x: &mut [f32]) {
    let d = x.len();
    let mut s = 0.0_f64;
    for i in 0..d {
        s += x[i] as f64 * x[i] as f64;
    }
    let r = 1.0 / (s as f32 + 1e-6).sqrt();
    for i in 0..d {
        x[i] *= r;
    }
}

fn rope_partial(v: &mut [f32], pos: usize, cfg: &Cfg) {
    let rd = cfg.rotary_dim;
    let n = rd / 2;
    if n == 0 {
        return;
    }
    for j in 0..n {
        let inv = cfg.theta.powf(-2.0 * j as f32 / rd as f32);
        let ang = pos as f32 * inv;
        let (cs, sn) = (ang.cos(), ang.sin());
        let a = v[j];
        let b = v[j + rd / 2];
        v[j] = a * cs - b * sn;
        v[j + rd / 2] = b * cs + a * sn;
    }
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

impl Model {
    fn gdn_token(&mut self, layer: &Layer, li: usize, x: &[f32], out: &mut [f32]) {
        let c = self.cfg.clone();
        let kd = c.lin_k_dim;
        let kheads = c.lin_k_heads;
        let vd = c.lin_v_dim;
        let vheads = c.lin_v_heads;
        let kdim = kd * kheads;
        let vdim = vd * vheads;
        let cdim = kdim * 2 + vdim;
        let kk = c.conv_kernel;

        let mut qkv = vec![0.0; cdim];
        matmul(&mut qkv, x, &layer.gdn_in_qkv);
        let mut a = vec![0.0; vheads];
        let mut b = vec![0.0; vheads];
        let mut z = vec![0.0; vdim];
        matmul(&mut a, x, &layer.gdn_in_a);
        matmul(&mut b, x, &layer.gdn_in_b);
        matmul(&mut z, x, &layer.gdn_in_z);

        let mut y = vec![0.0; cdim];
        if kk > 1 {
            let conv_st = &mut self.gdn_conv[li];
            for ch in 0..cdim {
                let mut acc = 0.0_f32;
                for j in 0..kk {
                    let vv = if j == kk - 1 {
                        qkv[ch]
                    } else {
                        conv_st[ch * (kk - 1) + j]
                    };
                    acc += layer.gdn_conv1d[ch * kk + j] * vv;
                }
                y[ch] = silu(acc);
            }
            for ch in 0..cdim {
                for s in 0..kk - 2 {
                    conv_st[ch * (kk - 1) + s] = conv_st[ch * (kk - 1) + s + 1];
                }
                conv_st[ch * (kk - 1) + (kk - 2)] = qkv[ch];
            }
        } else {
            for ch in 0..cdim {
                y[ch] = silu(layer.gdn_conv1d[ch] * qkv[ch]);
            }
        }

        let q_ = &y[..kdim];
        let k_ = &y[kdim..kdim * 2];
        let v_ = &y[kdim * 2..];
        let rep = vheads / kheads;
        assert!(rep >= 1 && vheads % kheads == 0);

        let s = &mut self.gdn_s[li];
        let mut qh = vec![0.0; vheads * kd];
        let mut kh = vec![0.0; vheads * kd];
        let mut vh = vec![0.0; vheads * vd];
        for h in 0..vheads {
            let khd = h / rep;
            for d in 0..kd {
                qh[h * kd + d] = q_[khd * kd + d];
                kh[h * kd + d] = k_[khd * kd + d];
            }
            for d in 0..vd {
                vh[h * vd + d] = v_[h * vd + d];
            }
            l2norm(&mut qh[h * kd..h * kd + kd]);
            l2norm(&mut kh[h * kd..h * kd + kd]);
            let sc = 1.0 / (kd as f32).sqrt();
            for d in 0..kd {
                qh[h * kd + d] *= sc;
            }
        }

        let mut snew = vec![0.0; vheads * kd * vd];
        let mut kv_mem = vec![0.0; vd];
        for h in 0..vheads {
            let ga = -layer.gdn_a_log[h].exp() * (1.0 + (a[h] + layer.gdn_dt_bias[h]).exp()).ln();
            let gt = ga.exp();
            let bt = 1.0 / (1.0 + (-b[h]).exp());
            let sh = &s[h * kd * vd..(h + 1) * kd * vd];
            let sn = &mut snew[h * kd * vd..(h + 1) * kd * vd];
            let qhh = &qh[h * kd..(h + 1) * kd];
            let khh = &kh[h * kd..(h + 1) * kd];
            let vhh = &vh[h * vd..(h + 1) * vd];
            for d in 0..vd {
                kv_mem[d] = 0.0;
            }
            for kk2 in 0..kd {
                let srow = &sh[kk2 * vd..(kk2 + 1) * vd];
                for d in 0..vd {
                    let sv = srow[d] * gt;
                    sn[kk2 * vd + d] = sv;
                    kv_mem[d] += sv * khh[kk2];
                }
            }
            for d in 0..vd {
                let delta = (vhh[d] - kv_mem[d]) * bt;
                for kk2 in 0..kd {
                    sn[kk2 * vd + d] += khh[kk2] * delta;
                }
            }
            for d in 0..vd {
                let mut acc = 0.0_f32;
                for kk2 in 0..kd {
                    acc += sn[kk2 * vd + d] * qhh[kk2];
                }
                kv_mem[d] = acc;
            }
            for d in 0..vd {
                vh[h * vd + d] = kv_mem[d];
            }
        }
        self.gdn_s[li].copy_from_slice(&snew);

        let mut normed = vec![0.0; vdim];
        for h in 0..vheads {
            rmsnorm_gated_row(
                &mut normed[h * vd..h * vd + vd],
                &vh[h * vd..h * vd + vd],
                &z[h * vd..h * vd + vd],
                &layer.gdn_norm,
                c.eps,
            );
        }
        matmul(out, &normed, &layer.gdn_out);
    }

    fn attention_token(
        &mut self,
        layer: &Layer,
        li: usize,
        x: &[f32],
        pos: usize,
        out: &mut [f32],
    ) {
        let c = self.cfg.clone();
        let h = c.heads;
        let hd = c.head_dim;
        let kv = c.kv_heads;
        let groups = h / kv;

        let mut qg = vec![0.0; 2 * h * hd];
        let mut k = vec![0.0; kv * hd];
        let mut vv = vec![0.0; kv * hd];
        matmul(&mut qg, x, &layer.attn_q);
        matmul(&mut k, x, &layer.attn_k);
        matmul(&mut vv, x, &layer.attn_v);
        // C calls rmsnorm_row in place (out == x); Rust forbids the alias, so
        // snapshot the row before normalizing.
        let qg_snap = qg.clone();
        for hh in 0..h {
            rmsnorm_row(
                &mut qg[hh * 2 * hd..hh * 2 * hd + hd],
                &qg_snap[hh * 2 * hd..hh * 2 * hd + hd],
                &layer.attn_qn,
                c.eps,
            );
        }
        let k_snap = k.clone();
        for g in 0..kv {
            rmsnorm_row(
                &mut k[g * hd..g * hd + hd],
                &k_snap[g * hd..g * hd + hd],
                &layer.attn_kn,
                c.eps,
            );
        }
        for hh in 0..h {
            rope_partial(&mut qg[hh * 2 * hd..hh * 2 * hd + 2 * hd], pos, &c);
        }
        for g in 0..kv {
            rope_partial(&mut k[g * hd..g * hd + hd], pos, &c);
        }
        // store K/V at pos (self-attention sees its own key)
        for g in 0..kv {
            let base = (li * kv + g) * c.max_t * hd + pos * hd;
            self.kv_k[base..base + hd].copy_from_slice(&k[g * hd..g * hd + hd]);
            self.kv_v[base..base + hd].copy_from_slice(&vv[g * hd..g * hd + hd]);
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0; pos + 1];
        let mut attn_out = vec![0.0; h * hd];
        for hh in 0..h {
            let qh = &qg[hh * 2 * hd..hh * 2 * hd + hd];
            let hg = hh / groups;
            let mut mx = -1e30_f32;
            for p in 0..=pos {
                let base = (li * kv + hg) * c.max_t * hd + p * hd;
                let mut acc = 0.0_f32;
                for dd in 0..hd {
                    acc += qh[dd] * self.kv_k[base + dd];
                }
                scores[p] = acc * scale;
                if scores[p] > mx {
                    mx = scores[p];
                }
            }
            let mut ssum = 0.0_f32;
            for p in 0..=pos {
                scores[p] = (scores[p] - mx).exp();
                ssum += scores[p];
            }
            let oh = &mut attn_out[hh * hd..hh * hd + hd];
            for dd in 0..hd {
                oh[dd] = 0.0;
            }
            for p in 0..=pos {
                let base = (li * kv + hg) * c.max_t * hd + p * hd;
                let w = scores[p] / ssum;
                for dd in 0..hd {
                    oh[dd] += w * self.kv_v[base + dd];
                }
            }
            let gh = &qg[(2 * hh + 1) * hd..(2 * hh + 2) * hd];
            for dd in 0..hd {
                oh[dd] *= 1.0 / (1.0 + (-gh[dd]).exp());
            }
        }
        matmul(out, &attn_out, &layer.attn_o);
    }

    fn moe_token(&mut self, layer: &Layer, li: usize, x: &[f32], out: &mut [f32]) {
        let c = self.cfg.clone();
        let e = c.experts;
        let k = c.topk;
        let d = c.hidden;

        let mut logits = vec![0.0; e];
        matmul(&mut logits, x, &layer.router);
        softmax_row(&mut logits);

        // top-k by probability, lower-index tie-break (torch.topk on probs)
        let mut idx: Vec<usize> = (0..e).collect();
        let mut val = logits.clone();
        for i in 0..k {
            let mut best = i;
            for j in i + 1..e {
                if val[j] > val[best] || (val[j] == val[best] && idx[j] < idx[best]) {
                    best = j;
                }
            }
            idx.swap(i, best);
            val.swap(i, best);
        }
        let wsum: f32 = val[..k].iter().sum();
        let mut acc = vec![0.0; d];
        for i in 0..k {
            let w = val[i] / wsum;
            let mats = &self.experts[li][idx[i]];
            let mut gate = vec![0.0; c.moe_inter];
            let mut up = vec![0.0; c.moe_inter];
            matmul(&mut gate, x, &mats[0]);
            matmul(&mut up, x, &mats[1]);
            let mut h = vec![0.0; c.moe_inter];
            for ii in 0..c.moe_inter {
                h[ii] = silu(gate[ii]) * up[ii];
            }
            let mut y = vec![0.0; d];
            matmul(&mut y, &h, &mats[2]);
            for dd in 0..d {
                acc[dd] += y[dd] * w;
            }
        }
        // shared expert: gs = sigmoid(se_g·x); y = se_down·(silu(se_gate·x)*se_up·x); out = acc + y*gs
        let mut sg = vec![0.0; 1];
        matmul(&mut sg, x, &layer.se_g);
        let gs = 1.0 / (1.0 + (-sg[0]).exp());
        let mut gv = vec![0.0; c.shared_inter];
        let mut h = vec![0.0; c.shared_inter];
        matmul(&mut gv, x, &layer.se_gate);
        matmul(&mut h, x, &layer.se_up);
        for i in 0..c.shared_inter {
            h[i] = silu(gv[i]) * h[i];
        }
        let mut sy = vec![0.0; d];
        matmul(&mut sy, &h, &layer.se_down);
        for dd in 0..d {
            out[dd] = acc[dd] + sy[dd] * gs;
        }
    }

    pub fn forward_token(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let c = self.cfg.clone();
        let d = c.hidden;
        let mut out = vec![0.0; d];
        out.copy_from_slice(&self.embed.f[token * d..(token + 1) * d]);
        for l in 0..c.layers {
            // clone the immutable layer weights to release the &self borrow
            // before the &mut self stage calls (ponytail: tiny fixture; on the
            // real model, split weights into an Arc/arena instead of cloning)
            let layer = self.layers[l].clone();
            let mut normed = vec![0.0; d];
            let mut attn = vec![0.0; d];
            rmsnorm_row(&mut normed, &out, &layer.in_ln, c.eps);
            if layer.is_gdn {
                self.gdn_token(&layer, l, &normed, &mut attn);
            } else {
                self.attention_token(&layer, l, &normed, pos, &mut attn);
            }
            for dd in 0..d {
                out[dd] += attn[dd];
            }
            rmsnorm_row(&mut normed, &out, &layer.post_ln, c.eps);
            let mut h = vec![0.0; d];
            self.moe_token(&layer, l, &normed, &mut h);
            for dd in 0..d {
                out[dd] += h[dd];
            }
        }
        let mut normed = vec![0.0; d];
        rmsnorm_row(&mut normed, &out, &self.final_norm, c.eps);
        let mut logits = vec![0.0; c.vocab];
        matmul(&mut logits, &normed, &self.lm_head);
        logits
    }
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

fn load_wt(st: &StFile, name: &str, o: usize, i: usize) -> Result<Wt, String> {
    Ok(Wt {
        f: st.f32(name, &[o as u64, i as u64])?,
        o,
        i,
    })
}

impl Model {
    pub fn load(st: &StFile, cfg: &Cfg) -> Result<Model, String> {
        let mut experts = Vec::new();
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let lp = format!("model.layers.{l}");
            let is_gdn = cfg.gdn_layers[l];
            let cdim = cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads;
            let vdim = cfg.lin_v_dim * cfg.lin_v_heads;
            let hd = cfg.head_dim;
            let layer = if is_gdn {
                Layer {
                    in_ln: st.f32(
                        &format!("{lp}.input_layernorm.weight"),
                        &[cfg.hidden as u64],
                    )?,
                    post_ln: st.f32(
                        &format!("{lp}.post_attention_layernorm.weight"),
                        &[cfg.hidden as u64],
                    )?,
                    is_gdn: true,
                    gdn_a_log: st.f32(
                        &format!("{lp}.linear_attn.A_log"),
                        &[cfg.lin_v_heads as u64],
                    )?,
                    gdn_dt_bias: st.f32(
                        &format!("{lp}.linear_attn.dt_bias"),
                        &[cfg.lin_v_heads as u64],
                    )?,
                    gdn_conv1d: st.f32(
                        &format!("{lp}.linear_attn.conv1d.weight"),
                        &[(cdim * cfg.conv_kernel) as u64],
                    )?,
                    gdn_in_a: load_wt(
                        st,
                        &format!("{lp}.linear_attn.in_proj_a.weight"),
                        cfg.lin_v_heads,
                        cfg.hidden,
                    )?,
                    gdn_in_b: load_wt(
                        st,
                        &format!("{lp}.linear_attn.in_proj_b.weight"),
                        cfg.lin_v_heads,
                        cfg.hidden,
                    )?,
                    gdn_in_qkv: load_wt(
                        st,
                        &format!("{lp}.linear_attn.in_proj_qkv.weight"),
                        cdim,
                        cfg.hidden,
                    )?,
                    gdn_in_z: load_wt(
                        st,
                        &format!("{lp}.linear_attn.in_proj_z.weight"),
                        vdim,
                        cfg.hidden,
                    )?,
                    gdn_norm: st.f32(
                        &format!("{lp}.linear_attn.norm.weight"),
                        &[cfg.lin_v_dim as u64],
                    )?,
                    gdn_out: load_wt(
                        st,
                        &format!("{lp}.linear_attn.out_proj.weight"),
                        cfg.hidden,
                        vdim,
                    )?,
                    attn_q: Wt {
                        f: vec![],
                        o: 0,
                        i: 0,
                    },
                    attn_k: Wt {
                        f: vec![],
                        o: 0,
                        i: 0,
                    },
                    attn_v: Wt {
                        f: vec![],
                        o: 0,
                        i: 0,
                    },
                    attn_o: Wt {
                        f: vec![],
                        o: 0,
                        i: 0,
                    },
                    attn_qn: vec![],
                    attn_kn: vec![],
                    router: load_wt(
                        st,
                        &format!("{lp}.mlp.gate.weight"),
                        cfg.experts,
                        cfg.hidden,
                    )?,
                    se_gate: load_wt(
                        st,
                        &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
                        cfg.shared_inter,
                        cfg.hidden,
                    )?,
                    se_up: load_wt(
                        st,
                        &format!("{lp}.mlp.shared_expert.up_proj.weight"),
                        cfg.shared_inter,
                        cfg.hidden,
                    )?,
                    se_down: load_wt(
                        st,
                        &format!("{lp}.mlp.shared_expert.down_proj.weight"),
                        cfg.hidden,
                        cfg.shared_inter,
                    )?,
                    se_g: load_wt(
                        st,
                        &format!("{lp}.mlp.shared_expert_gate.weight"),
                        1,
                        cfg.hidden,
                    )?,
                }
            } else {
                Layer {
                    in_ln: st.f32(
                        &format!("{lp}.input_layernorm.weight"),
                        &[cfg.hidden as u64],
                    )?,
                    post_ln: st.f32(
                        &format!("{lp}.post_attention_layernorm.weight"),
                        &[cfg.hidden as u64],
                    )?,
                    is_gdn: false,
                    gdn_a_log: vec![],
                    gdn_dt_bias: vec![],
                    gdn_conv1d: vec![],
                    gdn_in_a: Wt {
                        f: vec![],
                        o: 0,
                        i: 0,
                    },
                    gdn_in_b: Wt {
                        f: vec![],
                        o: 0,
                        i: 0,
                    },
                    gdn_in_qkv: Wt {
                        f: vec![],
                        o: 0,
                        i: 0,
                    },
                    gdn_in_z: Wt {
                        f: vec![],
                        o: 0,
                        i: 0,
                    },
                    gdn_norm: vec![],
                    gdn_out: Wt {
                        f: vec![],
                        o: 0,
                        i: 0,
                    },
                    attn_q: load_wt(
                        st,
                        &format!("{lp}.self_attn.q_proj.weight"),
                        2 * cfg.heads * hd,
                        cfg.hidden,
                    )?,
                    attn_k: load_wt(
                        st,
                        &format!("{lp}.self_attn.k_proj.weight"),
                        cfg.kv_heads * hd,
                        cfg.hidden,
                    )?,
                    attn_v: load_wt(
                        st,
                        &format!("{lp}.self_attn.v_proj.weight"),
                        cfg.kv_heads * hd,
                        cfg.hidden,
                    )?,
                    attn_o: load_wt(
                        st,
                        &format!("{lp}.self_attn.o_proj.weight"),
                        cfg.hidden,
                        cfg.heads * hd,
                    )?,
                    attn_qn: st.f32(&format!("{lp}.self_attn.q_norm.weight"), &[hd as u64])?,
                    attn_kn: st.f32(&format!("{lp}.self_attn.k_norm.weight"), &[hd as u64])?,
                    router: load_wt(
                        st,
                        &format!("{lp}.mlp.gate.weight"),
                        cfg.experts,
                        cfg.hidden,
                    )?,
                    se_gate: load_wt(
                        st,
                        &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
                        cfg.shared_inter,
                        cfg.hidden,
                    )?,
                    se_up: load_wt(
                        st,
                        &format!("{lp}.mlp.shared_expert.up_proj.weight"),
                        cfg.shared_inter,
                        cfg.hidden,
                    )?,
                    se_down: load_wt(
                        st,
                        &format!("{lp}.mlp.shared_expert.down_proj.weight"),
                        cfg.hidden,
                        cfg.shared_inter,
                    )?,
                    se_g: load_wt(
                        st,
                        &format!("{lp}.mlp.shared_expert_gate.weight"),
                        1,
                        cfg.hidden,
                    )?,
                }
            };
            // per-expert matrices: gate_up_proj [2I,H] fused -> gate [I,H], up [I,H]; down [H,I]
            let mut layer_experts = Vec::new();
            for e in 0..cfg.experts {
                let elp = format!("{lp}.mlp.experts.{e}");
                let gu = st.f32(
                    &format!("{elp}.gate_up_proj"),
                    &[(2 * cfg.moe_inter) as u64, cfg.hidden as u64],
                )?;
                let dn = st.f32(
                    &format!("{elp}.down_proj"),
                    &[cfg.hidden as u64, cfg.moe_inter as u64],
                )?;
                let half = cfg.moe_inter * cfg.hidden;
                let gate = Wt {
                    f: gu[..half].to_vec(),
                    o: cfg.moe_inter,
                    i: cfg.hidden,
                };
                let up = Wt {
                    f: gu[half..].to_vec(),
                    o: cfg.moe_inter,
                    i: cfg.hidden,
                };
                let down = Wt {
                    f: dn,
                    o: cfg.hidden,
                    i: cfg.moe_inter,
                };
                layer_experts.push([gate, up, down]);
            }
            experts.push(layer_experts);
            layers.push(layer);
        }
        Ok(Model {
            cfg: cfg.clone(),
            embed: load_wt(st, "model.embed_tokens.weight", cfg.vocab, cfg.hidden)?,
            lm_head: load_wt(st, "lm_head.weight", cfg.vocab, cfg.hidden)?,
            final_norm: st.f32("model.norm.weight", &[cfg.hidden as u64])?,
            layers,
            experts,
            gdn_conv: vec![
                vec![0.0; (cdim_total(cfg)) * cfg.conv_kernel.saturating_sub(1)];
                cfg.layers
            ],
            gdn_s: vec![vec![0.0; cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim]; cfg.layers],
            kv_k: vec![0.0; cfg.layers * cfg.kv_heads * cfg.max_t * cfg.head_dim],
            kv_v: vec![0.0; cfg.layers * cfg.kv_heads * cfg.max_t * cfg.head_dim],
        })
    }
}

fn cdim_total(cfg: &Cfg) -> usize {
    cfg.lin_k_dim * cfg.lin_k_heads * 2 + cfg.lin_v_dim * cfg.lin_v_heads
}

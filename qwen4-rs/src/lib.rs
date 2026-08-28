//! Tiny Qwen4 (Qwen3.8-Flash-Next / Qwen4Exp) scalar reference in Rust.
//!
//! Extends the qwen-rs port with the qwen4 additions: hyper connections
//! (GatedResidual mixer), QSA indexer sparse attention, and the PLE n-gram
//! layer. GDN/full-attention/MoE math is unchanged from qwen-rs (C-identical
//! numerics, f32 accumulators, exact reduction order).
//!
//! Gate: `ref.json` greedy_new_ids (token identity).

use std::path::Path;

pub mod coliload;
pub mod colisource;
pub mod ffi;

// ---------------------------------------------------------------------------
// safetensors reader (same minimal F32 adapter as qwen-rs)
// ---------------------------------------------------------------------------

pub struct StFile {
    data: Vec<u8>,
    tensors: std::collections::HashMap<String, (Vec<u64>, usize, usize)>,
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
        Ok(StFile { data: bytes, tensors })
    }

    pub fn f32(&self, name: &str, expect: &[u64]) -> Result<Vec<f32>, String> {
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
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub theta: f32,
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
    qsa_layers: Vec<bool>,
    // qwen4 hyper connections
    pub hc_count: usize,
    pub hc_lowrank: usize,
    // qwen4 QSA indexer
    pub idx_n_heads: usize,
    pub idx_kv_heads: usize,
    pub idx_head_dim: usize,
    pub idx_budget: usize,
    pub idx_ratio: usize,
    // qwen4 PLE
    pub ple_layer: i64, // -1 = off
    pub ple_embed_dim: usize,
    pub ple_conv_kernel: usize,
    pub ngram_size: usize,
    pub ngram_heads: usize,
    pub ngram_vocab_base: i64,
    pub ngram_div: i64,
    pub seed: u64,
    pub eos: i64,
}

pub fn load_cfg(path: &Path) -> Result<Cfg, String> {
    let mut v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?).unwrap();
    // Real checkpoints wrap the text backbone under `text_config`; the tiny
    // fixture has it top-level. Read from text_config when present.
    if let Some(tc) = v.get("text_config").and_then(|x| x.as_object()) {
        v = serde_json::Value::Object(tc.clone());
    }
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

    // qwen4 keys are TOP-LEVEL in config.json (C engine reads from root)
    let idx_n_heads = get("indexer_n_heads");
    let idx_kv_heads = get("indexer_kv_heads");
    let idx_head_dim = get("indexer_head_dim");
    let idx_budget = get("indexer_budget");
    let idx_ratio = get("indexer_compress_ratio");
    // QSA layers = full_attention layers when the indexer is configured
    let qsa_layers: Vec<bool> = layer_types
        .iter()
        .map(|t| t.as_str() == Some("full_attention") && idx_n_heads > 0)
        .collect();

    let hc_count = get("hc_count");
    let hc_lowrank = get("hc_lowrank");

    let ple_layer = v
        .get("ple_layer_ids")
        .and_then(|x| x.as_array())
        .and_then(|a| a.first())
        .and_then(|x| x.as_i64())
        .unwrap_or(-1);
    let ngram_size = get("ngram_size");
    let heads_per = get("heads_per_ngram");
    let ngram_heads = if ngram_size > 1 { heads_per * (ngram_size - 1) } else { 0 };
    let ple_embed_dim = get("ple_embed_dim");
    let ple_conv_kernel = get("ple_conv_kernel_size");
    let ngram_vocab_base = v.get("ngram_vocab_size_base").and_then(|x| x.as_i64()).unwrap_or(0);
    let ngram_div = v.get("make_ngram_vocab_size_divisible_by").and_then(|x| x.as_i64()).unwrap_or(1);

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
        qsa_layers,
        hc_count,
        hc_lowrank,
        idx_n_heads,
        idx_kv_heads,
        idx_head_dim,
        idx_budget,
        idx_ratio,
        ple_layer,
        ple_embed_dim,
        ple_conv_kernel,
        ngram_size,
        ngram_heads,
        ngram_vocab_base,
        ngram_div,
        seed: v.get("seed").and_then(|x| x.as_u64()).unwrap_or(0),
        eos: v.get("eos_token_id").and_then(|x| x.as_i64()).unwrap_or(-1),
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
    /// BF16 bytes when loaded from a .coli package (decoded per-row in
    /// matmul to keep resident memory at package size).
    bytes: Option<Vec<u8>>,
    o: usize,
    i: usize,
}

#[derive(Clone)]
struct Layer {
    in_ln: Vec<f32>,
    is_gdn: bool,
    is_qsa: bool,
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
    // QSA indexer
    index_qk: Wt,
    idx_qn: Vec<f32>,
    idx_kn: Vec<f32>,
    // hyper connections (attn + mlp sides)
    hc_norm: Vec<f32>,
    hc_mix_down: Wt,
    hc_mix_up: Wt,
    hc_inject: Wt,
    hc_mlp_norm: Vec<f32>,
    hc_mlp_mix_down: Wt,
    hc_mlp_mix_up: Wt,
    hc_mlp_inject: Wt,
    // MoE
    router: Wt,
    se_gate: Wt,
    se_up: Wt,
    se_down: Wt,
    se_g: Wt,
}

struct HcGlobal {
    norm: Vec<f32>,
    mix_down: Wt,
    mix_up: Wt,
}

impl Layer {
    /// Placeholder for the per-token mem::replace swap in forward_token
    /// (zero-sized weights; never used for compute).
    fn empty() -> Layer {
        Layer {
            in_ln: vec![],
            is_gdn: false,
            is_qsa: false,
            gdn_a_log: vec![],
            gdn_dt_bias: vec![],
            gdn_conv1d: vec![],
            gdn_in_a: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            gdn_in_b: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            gdn_in_qkv: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            gdn_in_z: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            gdn_norm: vec![],
            gdn_out: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            attn_q: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            attn_k: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            attn_v: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            attn_o: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            attn_qn: vec![],
            attn_kn: vec![],
            index_qk: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            idx_qn: vec![],
            idx_kn: vec![],
            hc_norm: vec![],
            hc_mix_down: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            hc_mix_up: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            hc_inject: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            hc_mlp_norm: vec![],
            hc_mlp_mix_down: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            hc_mlp_mix_up: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            hc_mlp_inject: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            router: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            se_gate: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            se_up: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            se_down: Wt { f: vec![], bytes: None, o: 0, i: 0 },
            se_g: Wt { f: vec![], bytes: None, o: 0, i: 0 },
        }
    }
}

pub struct Model {
    cfg: Cfg,
    /// .coli package for on-demand expert/ngram fetches (None in safetensors
    /// mode). ponytail: no cache yet — each fetch re-reads the record; add a
    /// per-layer FIFO when disk shows in profiles.
    coli: Option<colisource::ColiSource>,
    embed: Wt,
    lm_head: Wt,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    experts: Vec<Vec<[Wt; 3]>>,
    hc_global: HcGlobal,
    // PLE (present when cfg.ple_layer >= 0)
    ple_ngram: Wt,
    ple_key_proj: Wt,
    ple_value_proj: Wt,
    ple_norm_key: Vec<f32>,
    ple_norm_query: Vec<f32>,
    ple_norm_conv: Vec<f32>,
    ple_conv1d: Vec<f32>,
    ple_offsets: Vec<i64>,
    ple_sizes: Vec<i64>,
    ple_mult: Vec<u64>,
    // state
    gdn_conv: Vec<Vec<f32>>,
    gdn_s: Vec<Vec<f32>>,
    kv_k: Vec<f32>,
    kv_v: Vec<f32>,
    idx_cache: Vec<Vec<f32>>, // [layer][pos*nk]
    ple_ring: Vec<i64>,
    ple_conv_state: Vec<f32>,
    // ponytail: FIFO expert cache (the C engine's CACHE 0->256 win was
    // 280->173 ms/tok; LRU upgrade if hit-rate plateaus low). Entries own
    // their Metal tensor handles — the C backend keys handles by weight
    // pointer, so stale handles would serve wrong weights.
    expert_cache: std::collections::VecDeque<((i32, i32), crate::colisource::SlotExpert)>,
    expert_cache_cap: usize,
    /// Metal direct path (fused Apple8 moe_topk + coalesced GDN kernels).
    /// Brought up lazily on the first decode token; failures leave it off and
    /// every caller falls back to the CPU reference (C contract).
    metal_direct: bool,
    /// C QWEN_APPLE8_OVERLAP: split-phase moe_topk (submit -> CPU shared
    /// expert -> wait). Default ON (measured +2.5% loss only when OFF).
    metal_overlap: bool,
    /// Per-GDN-layer 16 KiB page-aligned re-home of the BF16 weights the
    /// Metal GDN kernels wrap zero-copy (C qwen_moe.c contract: wqkv/wz/wa/
    /// wb/wout + recurrent state + conv state must be page-aligned, weights
    /// live for the model lifetime, state is MUTATED BY THE GPU). Backed by
    /// one aligned alloc per layer; CPU fallback reads the same memory, so
    /// there is exactly ONE copy of the GDN weights (moved, not duplicated —
    /// the 16 GB M2 budget).
    gdn_metal: Vec<Option<GdnMetalLayer>>,
}

/// 16 KiB page-aligned allocation (Metal zero-copy wrap contract).
struct AlignedBuf {
    ptr: *mut u8,
    len: usize, // allocated (page-rounded) length
}

// SAFETY: the raw allocation is owned exclusively by this struct; it is
// created and freed on the host thread that owns the Model (decode is a
// single-threaded per-token pipeline; the GPU reads the memory but Metal
// shared-storage buffers are explicitly designed for host+device access).
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn zeroed(len: usize) -> Option<AlignedBuf> {
        if len == 0 {
            return None;
        }
        let rounded = (len + 16383) & !16383usize;
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut ptr as *mut *mut u8 as *mut *mut libc::c_void, 16384, rounded) };
        if rc != 0 || ptr.is_null() {
            return None;
        }
        unsafe { std::ptr::write_bytes(ptr, 0, rounded) };
        Some(AlignedBuf { ptr, len: rounded })
    }
    fn as_mut_f32(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.len / 4) }
    }
    fn as_mut_u8(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr as *mut libc::c_void) };
    }
}

/// Page-aligned GDN weights + state for one layer (Metal zero-copy source of
/// truth). The CPU scalar path reads the same memory, so both paths see
/// identical state; layouts are [O, I] row-major to match the Rust `Wt`.
struct GdnMetalLayer {
    /// [cdim, hidden] BF16 in_proj_qkv (16 KiB-aligned, zero-copy wrapped)
    wqkv: *mut u8,
    /// [vdim, hidden] BF16 in_proj_z
    wz: *mut u8,
    /// [vheads, hidden] BF16 in_proj_a
    wa: *mut u8,
    /// [vheads, hidden] BF16 in_proj_b
    wb: *mut u8,
    /// [hidden, vdim] BF16 out_proj
    wout: *mut u8,
    /// [vheads * kd * vd] recurrent state (f32, GPU-mutated in place)
    state: *mut f32,
    /// [cdim * (kk-1)] conv state (f32, GPU-mutated in place)
    conv_state: *mut f32,
    /// Keeps every allocation alive for the model lifetime.
    _bufs: Vec<AlignedBuf>,
}

/// Per-token scratch, allocated once at load (hidden/vocab sized).
struct TokScratch {
    mixed: Vec<f32>,
    attn: Vec<f32>,
    m2: Vec<f32>,
    moe: Vec<f32>,
    inj: Vec<f32>,
    inj2: Vec<f32>,
}

// ---------------------------------------------------------------------------
// math helpers (C-identical, same as qwen-rs)
// ---------------------------------------------------------------------------

/// NEON BF16 dot: y[o] = x[.] · w[o,.], weights BF16 (u16<<16 = f32).
/// 4-lane fma; fp-order differs from scalar (the gate decides).
#[cfg(target_arch = "aarch64")]
fn matmul_bf16_neon(y: &mut [f32], x: &[f32], w: &[u8], o: usize, i: usize) {
    use std::arch::aarch64::*;
    for oo in 0..o {
        let wr = &w[oo * i * 2..(oo + 1) * i * 2];
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let mut ii = 0;
        while ii + 8 <= i {
            // 8 bf16 -> 4 f32 (u16<<16), 4 x-f32
            unsafe {
                let wv = vld1q_u16(wr[ii * 2..].as_ptr() as *const u16);
                let w0 = vshlq_n_u32(vmovl_u16(vget_low_u16(wv)), 16);
                let w1 = vshlq_n_u32(vmovl_u16(vget_high_u16(wv)), 16);
                let wf0 = vreinterpretq_f32_u32(w0);
                let wf1 = vreinterpretq_f32_u32(w1);
                let x0 = vld1q_f32(x[ii..].as_ptr());
                let x1 = vld1q_f32(x[ii + 4..].as_ptr());
                acc = vfmaq_f32(acc, wf0, x0);
                acc = vfmaq_f32(acc, wf1, x1);
            }
            ii += 8;
        }
        let mut a = unsafe { vaddvq_f32(acc) };
        while ii < i {
            let u = u16::from_le_bytes([wr[ii * 2], wr[ii * 2 + 1]]);
            a += x[ii] * f32::from_bits((u as u32) << 16);
            ii += 1;
        }
        y[oo] = a;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn matmul_bf16_neon(_y: &mut [f32], _x: &[f32], _w: &[u8], _o: usize, _i: usize) {
    unreachable!();
}

fn matmul(y: &mut [f32], x: &[f32], w: &Wt) {
    let (o, i) = (w.o, w.i);
    // ponytail: thread::scope per call costs ~50-100us of spawn; only
    // parallelize matmuls big enough to amortize it (>= 16M MACs ≈ 2ms of
    // work at ~8 GFLOPs scalar). A persistent pool (rayon) would lower this
    // ceiling — add when the dense path is the measured bottleneck again.
    let parallel = o * i >= 16_000_000 && std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) > 1;
    if let Some(bytes) = &w.bytes {
        // NEON BF16: 4 f32 lanes, bf16 weights widened by (u16<<16).
        // fp-order differs from scalar (grouped fma) — the token-identity
        // gate decides; QWEN_NEON_BF16=0 opts out.
        let neon = std::env::var("QWEN_NEON_BF16").map(|v| v != "0").unwrap_or(true);
        #[cfg(target_arch = "aarch64")]
        let neon = neon && o * i >= 1 << 18;
        #[cfg(not(target_arch = "aarch64"))]
        let neon = false;
        if neon {
            matmul_bf16_neon(y, x, bytes, o, i);
            return;
        }
        if parallel {
            std::thread::scope(|s| {
                let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
                let chunk = o.div_ceil(nthreads);
                for (c, yslice) in y.chunks_mut(chunk).enumerate() {
                    let rows = c * chunk;
                    let (x, bytes) = (&*x, &*bytes);
                    s.spawn(move || {
                        for (oo, yv) in yslice.iter_mut().enumerate() {
                            let oo = rows + oo;
                            let mut acc = 0.0_f32;
                            for ii in 0..i {
                                let u = u16::from_le_bytes([
                                    bytes[(oo * i + ii) * 2],
                                    bytes[(oo * i + ii) * 2 + 1],
                                ]);
                                acc += x[ii] * f32::from_bits((u as u32) << 16);
                            }
                            *yv = acc;
                        }
                    });
                }
            });
        } else {
            for oo in 0..o {
                let mut acc = 0.0_f32;
                for ii in 0..i {
                    let u = u16::from_le_bytes([
                        bytes[(oo * i + ii) * 2],
                        bytes[(oo * i + ii) * 2 + 1],
                    ]);
                    acc += x[ii] * f32::from_bits((u as u32) << 16);
                }
                y[oo] = acc;
            }
        }
        return;
    }
    if parallel {
        std::thread::scope(|s| {
            let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            let chunk = o.div_ceil(nthreads);
            for (c, yslice) in y.chunks_mut(chunk).enumerate() {
                let rows = c * chunk;
                let (x, w) = (&*x, &*w);
                s.spawn(move || {
                    for (oo, yv) in yslice.iter_mut().enumerate() {
                        let oo = rows + oo;
                        let mut acc = 0.0_f32;
                        for ii in 0..i {
                            acc += x[ii] * w.f[oo * i + ii];
                        }
                        *yv = acc;
                    }
                });
            }
        });
    } else {
        for oo in 0..o {
            let mut acc = 0.0_f32;
            for ii in 0..i {
                acc += x[ii] * w.f[oo * i + ii];
            }
            y[oo] = acc;
        }
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

fn rmsnorm_grouped(out: &mut [f32], x: &[f32], w: &[f32], hc: usize, d: usize, eps: f32) {
    for g in 0..hc {
        rmsnorm_row(
            &mut out[g * d..g * d + d],
            &x[g * d..g * d + d],
            &w[g * d..g * d + d],
            eps,
        );
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

// qwen4 PLE helpers (C-identical)
const PLE_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const PLE_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const PLE_M2: u64 = 0x94D0_49BB_1331_11EB;

fn ple_splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(PLE_GAMMA);
    v = (v ^ (v >> 30)).wrapping_mul(PLE_M1);
    v = (v ^ (v >> 27)).wrapping_mul(PLE_M2);
    v ^ (v >> 31)
}

fn nth_prime_after(mut p: i64, count: i64) -> i64 {
    for _ in 0..count {
        p += 1;
        loop {
            let mut prime = p >= 2;
            if prime && p % 2 == 0 {
                prime = p == 2;
            }
            if prime {
                let mut d = 3_i64;
                while d * d <= p && d <= 46340 {
                    if p % d == 0 {
                        prime = false;
                        break;
                    }
                    d += 2;
                }
            }
            if prime {
                break;
            }
            p += 1;
        }
    }
    p
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

impl Model {
    fn hc_mix(
        &self,
        hc_norm: &[f32],
        hc_down: &Wt,
        hc_up: &Wt,
        hc_inj: Option<&Wt>,
        hx: &[f32],
        out: &mut [f32],
        inject: Option<&mut [f32]>,
    ) {
        let c = &self.cfg;
        let d = c.hidden;
        let hc = c.hc_count;
        let lr = c.hc_lowrank;
        let hcd = hc * d;
        let mut normed = vec![0.0; hcd];
        rmsnorm_grouped(&mut normed, hx, hc_norm, hc, d, c.eps);
        let mut lo = vec![0.0; lr];
        matmul(&mut lo, &normed, hc_down);
        for i in 0..lr {
            lo[i] = silu(lo[i] / hc as f32);
        }
        let mut hi = vec![0.0; hcd];
        matmul(&mut hi, &lo, hc_up);
        for i in 0..hcd {
            hi[i] = 1.0 / (1.0 + (-hi[i]).exp());
        }
        for i in 0..hcd {
            out[i % d] += (hi[i] * normed[i]) * (1.0 / hc as f32);
        }
        if let Some(inj) = inject {
            let bi_w = hc_inj.unwrap();
            let mut bi = vec![0.0; hc];
            matmul(&mut bi, &normed, bi_w);
            for g in 0..hc {
                inj[g] = 2.0 / (1.0 + (-bi[g] / hc as f32).exp());
            }
        }
    }

    /// Move one GDN layer's BF16 weights into page-aligned buffers (C
    /// contract: newBufferWithBytesNoCopy requires 16 KiB-aligned pointers,
    /// length page-rounded). Returns None if any alloc fails (Metal GDN then
    /// stays off for that layer; CPU path unaffected).
    fn build_gdn_metal(layer: &Layer, cfg: &Cfg) -> Option<GdnMetalLayer> {
        if !layer.is_gdn || !crate::ffi::direct_available() {
            return None;
        }
        let kd = cfg.lin_k_dim;
        let kheads = cfg.lin_k_heads;
        let vd = cfg.lin_v_dim;
        let vheads = cfg.lin_v_heads;
        let cdim = kd * kheads * 2 + vd * vheads;
        let kk = cfg.conv_kernel;
        let move_bf16 = |w: &Wt| -> Option<AlignedBuf> {
            let bytes = w.bytes.as_ref()?;
            let mut buf = AlignedBuf::zeroed(bytes.len())?;
            buf.as_mut_u8()[..bytes.len()].copy_from_slice(bytes);
            Some(buf)
        };
        let state_elems = vheads * kd * vd;
        let conv_elems = cdim * kk.saturating_sub(1);
        let state = AlignedBuf::zeroed(state_elems * 4)?;
        let conv_state = AlignedBuf::zeroed(conv_elems * 4)?;
        let wqkv = move_bf16(&layer.gdn_in_qkv)?;
        let wz = move_bf16(&layer.gdn_in_z)?;
        let wa = move_bf16(&layer.gdn_in_a)?;
        let wb = move_bf16(&layer.gdn_in_b)?;
        let wout = move_bf16(&layer.gdn_out)?;
        Some(GdnMetalLayer {
            wqkv: wqkv.ptr,
            wz: wz.ptr,
            wa: wa.ptr,
            wb: wb.ptr,
            wout: wout.ptr,
            state: state.ptr as *mut f32,
            conv_state: conv_state.ptr as *mut f32,
            _bufs: vec![wqkv, wz, wa, wb, wout, state, conv_state],
        })
    }

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

        // One-time aligned re-home (C calloc_checked/coli_wt intercept): move
        // the five BF16 GDN matrices into 16 KiB-aligned buffers so Metal can
        // wrap them zero-copy. State lives in the same aligned blocks; the
        // CPU fallback reads the SAME memory (single source of truth).
        if self.gdn_metal[li].is_none() {
            self.gdn_metal[li] = Self::build_gdn_metal(layer, &self.cfg);
        }
        // Metal direct path (C QWEN_GDN_METAL default ON): the coalesced
        // kernels consume the page-aligned re-home above; rc semantics per
        // C contract (0=decline pre-submit, <0 = fatal post-submit).
        let gdn_enabled = std::env::var("QWEN_GDN_METAL")
            .map(|v| v != "0")
            .unwrap_or(true);
        if let Some(gm) = &self.gdn_metal[li] {
            if gdn_enabled {
                // SAFETY: exact-length views over the layer's aligned blocks
                // (kept alive by gm._bufs for the model lifetime). The GPU
                // mutates state/conv_state in place; the CPU fallback syncs
                // (below) so both paths share one source of truth.
                let (n_state, n_conv) = (vheads * kd * vd, cdim * (kk - 1));
                let rc = unsafe {
                    crate::ffi::gdn_token(
                        li,
                        x,
                        out,
                        std::slice::from_raw_parts(gm.wqkv, cdim * c.hidden * 2),
                        std::slice::from_raw_parts(gm.wz, vdim * c.hidden * 2),
                        std::slice::from_raw_parts(gm.wa, vheads * c.hidden * 2),
                        std::slice::from_raw_parts(gm.wb, vheads * c.hidden * 2),
                        std::slice::from_raw_parts(gm.wout, c.hidden * vdim * 2),
                        &layer.gdn_a_log,
                        &layer.gdn_dt_bias,
                        &layer.gdn_conv1d,
                        &layer.gdn_norm,
                        std::slice::from_raw_parts_mut(gm.state, n_state),
                        std::slice::from_raw_parts_mut(gm.conv_state, n_conv),
                        c.hidden,
                        kheads,
                        kd,
                        vheads,
                        vd,
                        kk,
                        c.eps,
                    )
                };
                if rc > 0 {
                    return;
                }
                if rc < 0 {
                    eprintln!("qwen4-rs: Metal GDN failed after submission (layer {li})");
                    std::process::exit(1);
                }
                // rc == 0: declined pre-submit. GPU state is the truth from
                // any earlier Metal tokens -> pull it into the CPU state
                // before the scalar reference below runs.
                let n_state = vheads * kd * vd;
                let n_conv = cdim * (kk - 1);
                unsafe {
                    std::ptr::copy_nonoverlapping(gm.state, self.gdn_s[li].as_mut_ptr(), n_state);
                    std::ptr::copy_nonoverlapping(
                        gm.conv_state,
                        self.gdn_conv[li].as_mut_ptr(),
                        n_conv,
                    );
                }
            }
        }

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
        // Keep the aligned (GPU-visible) state in sync after a CPU token so
        // the next Metal token starts from the same recurrence (C has ONE
        // state; here we sync on the CPU path only).
        if let Some(gm) = &self.gdn_metal[li] {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.gdn_s[li].as_ptr(),
                    gm.state,
                    vheads * kd * vd,
                );
                std::ptr::copy_nonoverlapping(
                    self.gdn_conv[li].as_ptr(),
                    gm.conv_state,
                    cdim * (kk - 1),
                );
            }
        }

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

    fn attention_common(
        &mut self,
        layer: &Layer,
        li: usize,
        x: &[f32],
        pos: usize,
        selected: Option<&[usize]>,
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
        for g in 0..kv {
            let base = (li * kv + g) * c.max_t * hd + pos * hd;
            self.kv_k[base..base + hd].copy_from_slice(&k[g * hd..g * hd + hd]);
            self.kv_v[base..base + hd].copy_from_slice(&vv[g * hd..g * hd + hd]);
        }

        let positions: Vec<usize> = match selected {
            Some(sel) => sel.to_vec(),
            None => (0..=pos).collect(),
        };
        let nsel = positions.len();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = vec![0.0; nsel];
        let mut attn_out = vec![0.0; h * hd];
        for hh in 0..h {
            let qh = &qg[hh * 2 * hd..hh * 2 * hd + hd];
            let hg = hh / groups;
            let mut mx = -1e30_f32;
            for (s, &p) in positions.iter().enumerate() {
                let base = (li * kv + hg) * c.max_t * hd + p * hd;
                let mut acc = 0.0_f32;
                for dd in 0..hd {
                    acc += qh[dd] * self.kv_k[base + dd];
                }
                scores[s] = acc * scale;
                if scores[s] > mx {
                    mx = scores[s];
                }
            }
            let mut ssum = 0.0_f32;
            for s in 0..nsel {
                scores[s] = (scores[s] - mx).exp();
                ssum += scores[s];
            }
            let oh = &mut attn_out[hh * hd..hh * hd + hd];
            for dd in 0..hd {
                oh[dd] = 0.0;
            }
            for (s, &p) in positions.iter().enumerate() {
                let base = (li * kv + hg) * c.max_t * hd + p * hd;
                let w = scores[s] / ssum;
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

    fn attention_token(&mut self, layer: &Layer, li: usize, x: &[f32], pos: usize, out: &mut [f32]) {
        self.attention_common(layer, li, x, pos, None, out);
    }

    fn qsa_select(&mut self, layer: &Layer, li: usize, x: &[f32], pos: usize) -> Vec<usize> {
        let c = self.cfg.clone();
        let ih = c.idx_head_dim;
        let in_ = c.idx_n_heads;
        let ik = c.idx_kv_heads;
        let ratio = c.idx_ratio;
        let budget = c.idx_budget;
        let nq = ih * in_;
        let nk = ih * ik;

        let mut qk = vec![0.0; nq + nk];
        matmul(&mut qk, x, &layer.index_qk);
        let mut q = qk[..nq].to_vec();
        let q_snap = q.clone();
        for hh in 0..in_ {
            rmsnorm_row(
                &mut q[hh * ih..hh * ih + ih],
                &q_snap[hh * ih..hh * ih + ih],
                &layer.idx_qn,
                c.eps,
            );
        }
        for hh in 0..in_ {
            rope_partial(&mut q[hh * ih..hh * ih + ih], pos, &c);
        }
        // store raw indexer k for this position
        let cached = &mut self.idx_cache[li];
        cached[pos * nk..pos * nk + nk].copy_from_slice(&qk[nq..nq + nk]);
        let len = pos + 1;
        let nblk = len / ratio;
        let mut sel: Vec<usize> = Vec::new();
        if nblk > 0 {
            let mut pool = vec![0.0; nblk * ih];
            let mut starts = vec![0_usize; nblk];
            for b in 0..nblk {
                starts[b] = b * ratio;
                for d in 0..ih {
                    let mut acc = 0.0_f64;
                    for r in 0..ratio {
                        acc += cached[(b * ratio + r) * nk + d] as f64;
                    }
                    pool[b * ih + d] = (acc / ratio as f64) as f32;
                }
            }
            let pool_snap = pool.clone();
            for b in 0..nblk {
                rmsnorm_row(
                    &mut pool[b * ih..b * ih + ih],
                    &pool_snap[b * ih..b * ih + ih],
                    &layer.idx_kn,
                    c.eps,
                );
            }
            let pool2 = pool.clone();
            for b in 0..nblk {
                let mut row = pool2[b * ih..b * ih + ih].to_vec();
                rope_partial(&mut row, starts[b], &c);
                pool[b * ih..b * ih + ih].copy_from_slice(&row);
            }
            let mut topk = budget / ratio;
            if topk > nblk {
                topk = nblk;
            }
            let mut sc = vec![0.0; nblk];
            let mut ord: Vec<usize> = (0..nblk).collect();
            for b in 0..nblk {
                let mut acc = 0.0_f32;
                for hh in 0..in_ {
                    let qh = &q[hh * ih..hh * ih + ih];
                    let kb = &pool[b * ih..b * ih + ih];
                    let mut dot = 0.0_f32;
                    for d in 0..ih {
                        dot += qh[d] * kb[d];
                    }
                    acc += if dot > 0.0 { dot } else { 0.0 };
                }
                sc[b] = acc / (ih as f32).sqrt();
            }
            // selection sort desc, lower index wins ties
            for i in 0..topk {
                let mut best = i;
                for j in i + 1..nblk {
                    if sc[j] > sc[best] || (sc[j] == sc[best] && ord[j] < ord[best]) {
                        best = j;
                    }
                }
                ord.swap(i, best);
                sc.swap(i, best);
            }
            for i in 0..topk {
                for r in 0..ratio {
                    sel.push(starts[ord[i]] + r);
                }
            }
        }
        for p in nblk * ratio..len {
            sel.push(p);
        }
        sel
    }

    fn sparse_attn_token(&mut self, layer: &Layer, li: usize, x: &[f32], pos: usize, out: &mut [f32]) {
        let sel = self.qsa_select(layer, li, x, pos);
        self.attention_common(layer, li, x, pos, Some(&sel), out);
    }

    /// FIFO expert cache. On miss, stream the expert's three raw Apple8
    /// matrices async (MetalIO) into ONE slot with the C engine's layout
    /// (gate at 0, up at align16, down at align16(up_end)); the slot IS the
    /// cache unit — the fused moe_topk consumes it in native tile order, the
    /// CPU fallback preads its shared-storage bytes. One MTLIOFileHandle per
    /// shard (cached process-wide; the C table caps at 64). Eviction frees
    /// the slot (its Drop waits + releases).
    fn cached_expert(
        &mut self,
        li: i32,
        ei: i32,
    ) -> Option<std::rc::Rc<crate::colisource::SlotRef>> {
        for (k, v) in self.expert_cache.iter() {
            if *k == (li, ei) {
                return Some(std::rc::Rc::new(v.ref_view()));
            }
        }
        let coli = self.coli.as_ref()?;
        let se: Option<crate::colisource::SlotExpert> = (|| {
            let recs = coli.pkg_ref().expert_records(li, ei);
            let rec = recs.first()?;
            let shard = coli.pkg_ref().shard_path(rec.shard_id)?;
            let (regions, dims) = coli.pkg_ref().expert_matrix_regions(rec)?;
            let fid = crate::ffi::mio_file(&shard)?;
            let (slot, ev) = crate::ffi::mio_load_expert(fid, &regions)?;
            if unsafe { crate::ffi::metalio_wait(ev) } != 0 {
                unsafe { crate::ffi::metalio_slot_free(slot) };
                return None;
            }
            let ptr = unsafe { crate::ffi::metalio_slot_ptr(slot) } as *mut u8;
            if ptr.is_null() {
                unsafe { crate::ffi::metalio_slot_free(slot) };
                return None;
            }
            let gb = regions[0].1;
            let ub = regions[1].1;
            let db = regions[2].1;
            let up_off = (gb + 15) & !15usize;
            let down_off = (up_off + ub + 15) & !15usize;
            Some(crate::colisource::SlotExpert {
                slot,
                gate_bytes: gb,
                up_offset: up_off,
                up_bytes: ub,
                down_offset: down_off,
                down_bytes: db,
                ptr,
                bf16_cache: std::cell::RefCell::new(None),
                rows: [dims[0].0, dims[1].0, dims[2].0],
                cols: [dims[0].1, dims[1].1, dims[2].1],
            })
        })();
        let se = se?;
        let entry = se;
        if self.expert_cache.len() >= self.expert_cache_cap {
            if let Some((_, old)) = self.expert_cache.pop_front() {
                drop(old); // SlotExpert::drop frees the MetalIO slot
            }
        }
        self.expert_cache.push_back(((li, ei), entry));
        for (k, v) in self.expert_cache.iter() {
            if *k == (li, ei) {
                return Some(std::rc::Rc::new(v.ref_view()));
            }
        }
        None
    }

    /// Direct-path expert descriptor for the fused moe_topk (slot + offsets).
    fn slot_descriptor(se: &crate::colisource::SlotRef) -> crate::ffi::ColiApple8MetalioExpert {
        crate::ffi::ColiApple8MetalioExpert {
            slot: se.slot,
            gate_offset: 0,
            gate_bytes: se.gate_bytes,
            up_offset: se.up_offset,
            up_bytes: se.up_bytes,
            down_offset: se.down_offset,
            down_bytes: se.down_bytes,
        }
    }

    /// CPU fallback for one slot-resident expert: decode the shared-storage
    /// tile bytes (lazily cached per expert) and run the BF16 matmuls.
    fn slot_expert_cpu(
        se: &crate::colisource::SlotRef,
        x: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        y: &mut [f32],
    ) {
        // SAFETY: slot bytes are CPU-visible shared storage, valid while the
        // cache entry owns the slot (ref_view borrows a live entry; eviction
        // only happens inside cached_expert while no ref is outstanding).
        let total = se.down_offset + se.down_bytes;
        let raw = unsafe { std::slice::from_raw_parts(se.ptr, total) };
        let parts = [
            &raw[0..se.gate_bytes],
            &raw[se.up_offset..se.up_offset + se.up_bytes],
            &raw[se.down_offset..se.down_offset + se.down_bytes],
        ];
        let mut cb: [Vec<u8>; 3] = Default::default();
        for (pi, p) in parts.iter().enumerate() {
            let f = colibri_format::codecs::apple8_mxfp4_decode(
                p,
                se.rows[pi] as u64,
                se.cols[pi] as u64,
            )
            .unwrap_or_default();
            cb[pi] = f.into_iter().flat_map(crate::colisource::bf16_bytes).collect();
        }
        let [gb, ub, db] = cb;
        let g = Wt { f: vec![], bytes: Some(gb), o: se.rows[0], i: se.cols[0] };
        let u = Wt { f: vec![], bytes: Some(ub), o: se.rows[1], i: se.cols[1] };
        let dw = Wt { f: vec![], bytes: Some(db), o: se.rows[2], i: se.cols[2] };
        matmul(gate, x, &g);
        matmul(up, x, &u);
        let h: Vec<f32> = (0..gate.len()).map(|ii| silu(gate[ii]) * up[ii]).collect();
        matmul(y, &h, &dw);
    }

    fn moe_token(&mut self, layer: &Layer, li: usize, x: &[f32], out: &mut [f32]) {
        let c = self.cfg.clone();
        let e = c.experts;
        let k = c.topk;
        let d = c.hidden;

        let mut logits = vec![0.0; e];
        matmul(&mut logits, x, &layer.router);
        softmax_row(&mut logits);

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

        // Direct fused path (C QWEN_APPLE8_DIRECT): all K slot-resident
        // experts in ONE command buffer (gate+up+swiglu -> down -> weighted
        // reduce), consumed in top-k order with pre-renormalized weights.
        // Split-phase by default (C QWEN_APPLE8_OVERLAP=1): submit first, run
        // the CPU shared expert while the GPU works, then wait.
        let direct = self.metal_direct
            && crate::ffi::direct_available()
            && k <= 64
            && self.coli.is_some();
        let mut pending: Option<*mut std::ffi::c_void> = None;
        let mut pending_acc: Option<Vec<f32>> = None;
        if direct {
            let mut ex: Vec<crate::ffi::ColiApple8MetalioExpert> = Vec::with_capacity(k);
            let mut ws: Vec<f32> = Vec::with_capacity(k);
            let mut all_ok = true;
            for i in 0..k {
                match self.cached_expert(li as i32, idx[i] as i32) {
                    Some(ce) => {
                        ex.push(Self::slot_descriptor(&ce));
                        ws.push(val[i] / wsum);
                    }
                    None => {
                        all_ok = false;
                        break;
                    }
                }
            }
            if all_ok {
                if self.metal_overlap {
                    pending = crate::ffi::moe_topk_begin(&ex, &ws, x, d, c.moe_inter);
                } else if !crate::ffi::moe_topk(&ex, &ws, x, &mut acc, d, c.moe_inter) {
                    pending = None; // decline -> CPU per-expert loop below
                }
            }
            if pending.is_none() && self.metal_overlap && all_ok {
                // begin() declined mid-run: fall through to the CPU loop
                // (weights unchanged; nothing was submitted).
            }
        }

        if pending.is_some() {
            // CPU shared expert overlaps the routed-GPU wait (C order: shared
            // expert runs BETWEEN submit and finish).
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
            let p = pending.unwrap();
            if !crate::ffi::moe_topk_finish(p, &mut acc, d) {
                // GPU fault AFTER submit: C contract = redo those experts on
                // CPU. acc was scratch for the GPU result; recompute routed
                // experts on CPU into a fresh accumulator.
                acc = vec![0.0; d];
                for i in 0..k {
                    let w = val[i] / wsum;
                    if let Some(ce) = self.cached_expert(li as i32, idx[i] as i32) {
                        let mut gate = vec![0.0; c.moe_inter];
                        let mut up = vec![0.0; c.moe_inter];
                        let mut y = vec![0.0; d];
                        Self::slot_expert_cpu(&ce, x, &mut gate, &mut up, &mut y);
                        for dd in 0..d {
                            acc[dd] += y[dd] * w;
                        }
                    }
                }
            }
            for dd in 0..d {
                out[dd] = acc[dd] + sy[dd] * gs;
            }
            return;
        }

        for i in 0..k {
            let w = val[i] / wsum;
            // .coli mode: slot-resident expert. The Metal fused path ran
            // above (returned early on success); this loop is the CPU
            // fallback (decodes the slot's shared-storage tiles lazily) or
            // the canonical decode path for non-Apple8 packages.
            // safetensors mode: preloaded experts.
            let mats: [Wt; 3] = if self.coli.is_some() {
                match self.cached_expert(li as i32, idx[i] as i32) {
                    Some(ce) => {
                        let mut gate = vec![0.0; c.moe_inter];
                        let mut up = vec![0.0; c.moe_inter];
                        let mut y = vec![0.0; d];
                        Self::slot_expert_cpu(&ce, x, &mut gate, &mut up, &mut y);
                        for dd in 0..d {
                            acc[dd] += y[dd] * w;
                        }
                        continue; // acc updated; skip the common tail
                    }
                    None => {
                        // BF16/INT4 canonical experts: decode path
                        let m = self
                            .coli
                            .as_ref()
                            .unwrap()
                            .expert_matrices(li as i32, idx[i] as i32)
                            .unwrap_or_else(|e| {
                                panic!("expert ({li},{}) fetch failed: {e}", idx[i])
                            });
                        [
                            Wt { f: vec![], bytes: Some(m[0].bytes.clone()), o: m[0].o, i: m[0].i },
                            Wt { f: vec![], bytes: Some(m[1].bytes.clone()), o: m[1].o, i: m[1].i },
                            Wt { f: vec![], bytes: Some(m[2].bytes.clone()), o: m[2].o, i: m[2].i },
                        ]
                    }
                }
            } else {
                self.experts[li][idx[i]].clone()
            };
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

    fn ple_forward(&mut self, stream: &mut [f32]) {
        let c = self.cfg.clone();
        if c.ple_layer < 0 {
            return;
        }
        let d = c.hidden;
        let hc = c.hc_count;
        let hcd = hc * d;
        let ns = c.ngram_size;
        let hpn = c.ngram_heads / (ns - 1);
        let heads = c.ngram_heads;
        let hd_per = c.ple_embed_dim / heads;
        let kk = c.ple_conv_kernel;
        let dil = ns;
        let pad = (kk - 1) * dil;
        let ctx = ns - 1;
        let t = ctx + 1;

        let mut hist = [0_i64; 16];
        for i in 0..ctx {
            hist[i] = self.ple_ring[i];
        }
        hist[ctx] = self.ple_ring[ctx];

        let mut shifted = [[0_i64; 16]; 8];
        for s in 0..ns {
            let mut seg = 0_usize;
            for p in 0..t {
                let mut v = c.eos;
                let src = p as i64 - s as i64;
                if src >= 0 && p - seg >= s {
                    v = hist[src as usize];
                }
                shifted[s][p] = v;
                if hist[p] == c.eos {
                    seg = p + 1;
                }
            }
        }

        let mut rows = [0_u64; 16];
        let mut n = 0;
        for ng in 2..=ns {
            for kk2 in 0..hpn {
                let h = (ng - 2) * hpn + kk2;
                let mut mixed = 0_u64;
                for j in 0..ng {
                    mixed ^= (shifted[j][t - 1] as u64).wrapping_mul(self.ple_mult[j]);
                }
                rows[n] = (mixed % self.ple_sizes[h] as u64) + self.ple_offsets[h] as u64;
                n += 1;
            }
        }

        // ponytail: sized from config — the tiny fixture's 256 hides this;
        // the real model's ple_embed_dim is 20480.
        let mut emb = vec![0.0_f32; c.ple_embed_dim.max(256)];
        for h in 0..heads {
            let r = rows[h] as usize;
            // .coli mode: fetch the ngram row on demand (F8 E4M3 shards, one
            // pread per row — the 51 GB table is never resident).
            // safetensors mode: read the resident table.
            if let Some(coli) = &self.coli {
                let row_bytes = coli
                    .ple_ngram_row_f8(c.ple_layer as i32, r as u64, hd_per)
                    .unwrap_or_else(|e| panic!("ple ngram row {r} fetch failed: {e}"));
                let scale = coli
                    .ple_ngram_scale(c.ple_layer as i32)
                    .unwrap_or(1.0);
                for d in 0..hd_per {
                    emb[h * hd_per + d] = colisource::ColiSource::e4m3_decode(row_bytes[d]) * scale;
                }
            } else {
                let row = &self.ple_ngram.f[r * hd_per..(r + 1) * hd_per];
                emb[h * hd_per..(h + 1) * hd_per].copy_from_slice(row);
            }
        }

        let mut key = vec![0.0; hcd];
        let mut value = vec![0.0; d];
        matmul(&mut key, &emb, &self.ple_key_proj);
        matmul(&mut value, &emb, &self.ple_value_proj);
        let key_snap = key.clone();
        rmsnorm_grouped(&mut key, &key_snap, &self.ple_norm_key, hc, d, c.eps);
        let mut qn = vec![0.0; hcd];
        rmsnorm_grouped(&mut qn, stream, &self.ple_norm_query, hc, d, c.eps);
        let mut gated = vec![0.0; hcd];
        let r_d = 1.0 / (d as f32).sqrt();
        for g in 0..hc {
            let mut acc = 0.0_f32;
            for dd in 0..d {
                acc += key[g * d + dd] * qn[g * d + dd];
            }
            let tt = acc * r_d;
            let mut mag = tt.abs();
            if mag < 1e-6 {
                mag = 1e-6;
            }
            let sg = if tt >= 0.0 { 1.0 } else { -1.0 } * mag.sqrt();
            let sig = 1.0 / (1.0 + (-sg).exp());
            for dd in 0..d {
                gated[g * d + dd] = sig * value[dd];
            }
        }
        let mut conv_in = vec![0.0; hcd];
        rmsnorm_grouped(&mut conv_in, &gated, &self.ple_norm_conv, hc, d, c.eps);
        let st = pad + 1;
        for i in 0..hcd {
            let w = &self.ple_conv1d[i * kk..(i + 1) * kk];
            let mut acc = 0.0_f32;
            for j in 0..kk {
                let lag = j * dil;
                let xj = if lag == 0 {
                    conv_in[i]
                } else {
                    self.ple_conv_state[i * st + (lag - 1)]
                };
                acc += w[j] * xj;
            }
            for s in (1..st).rev() {
                self.ple_conv_state[i * st + s] = self.ple_conv_state[i * st + s - 1];
            }
            self.ple_conv_state[i * st] = conv_in[i];
            stream[i] += gated[i] + silu(acc);
        }
    }

    pub fn forward_token(&mut self, token: usize, pos: usize) -> Vec<f32> {
        let c = self.cfg.clone();
        let d = c.hidden;
        let hc = c.hc_count;
        let hcd = hc * d;

        // embed row repeated hc times (BF16 bytes in .coli mode, f32 in
        // safetensors mode)
        let mut stream = vec![0.0; hcd];
        let row: Vec<f32> = if let Some(eb) = &self.embed.bytes {
            (0..d)
                .map(|j| {
                    let u = u16::from_le_bytes([eb[(token * d + j) * 2], eb[(token * d + j) * 2 + 1]]);
                    f32::from_bits((u as u32) << 16)
                })
                .collect()
        } else {
            self.embed.f[token * d..(token + 1) * d].to_vec()
        };
        for g in 0..hc {
            stream[g * d..(g + 1) * d].copy_from_slice(&row);
        }

        // PLE ring push
        for i in 0..c.ngram_size - 1 {
            self.ple_ring[i] = self.ple_ring[i + 1];
        }
        self.ple_ring[c.ngram_size - 1] = token as i64;

        for l in 0..c.layers {
            // Share-on-read: `self.layers[l].clone()` deep-copied EVERY weight
            // per token (~3.7 GB memcpy/token on the real model — it would
            // dominate any Metal win). Split-borrow: methods take &Layer, so
            // pull the layer out, run both sub-phases, put it back.
            let mut layer = std::mem::replace(&mut self.layers[l], Layer::empty());
            let mut mixed = vec![0.0; d];
            let mut attn = vec![0.0; d];
            if c.ple_layer == l as i64 {
                self.ple_forward(&mut stream);
            }
            let mut inj = vec![0.0; hc];
            self.hc_mix(
                &layer.hc_norm,
                &layer.hc_mix_down,
                &layer.hc_mix_up,
                Some(&layer.hc_inject),
                &stream,
                &mut mixed,
                Some(&mut inj),
            );
            if layer.is_gdn {
                self.gdn_token(&layer, l, &mixed, &mut attn);
            } else if layer.is_qsa {
                self.sparse_attn_token(&layer, l, &mixed, pos, &mut attn);
            } else {
                self.attention_token(&layer, l, &mixed, pos, &mut attn);
            }
            for g in 0..hc {
                for dd in 0..d {
                    stream[g * d + dd] += inj[g] * attn[dd];
                }
            }
            let mut m2 = vec![0.0; d];
            let mut moe = vec![0.0; d];
            self.hc_mix(
                &layer.hc_mlp_norm,
                &layer.hc_mlp_mix_down,
                &layer.hc_mlp_mix_up,
                Some(&layer.hc_mlp_inject),
                &stream,
                &mut m2,
                Some(&mut inj),
            );
            self.moe_token(&layer, l, &m2, &mut moe);
            for g in 0..hc {
                for dd in 0..d {
                    stream[g * d + dd] += inj[g] * moe[dd];
                }
            }
            self.layers[l] = layer;
        }
        // final global hc_mix (no inject)
        let mut out = vec![0.0; d];
        self.hc_mix(
            &self.hc_global.norm,
            &self.hc_global.mix_down,
            &self.hc_global.mix_up,
            None,
            &stream,
            &mut out,
            None,
        );
        // qwen4_no_final_norm: when hc is active and norm.weight is absent,
        // the mixer output goes straight to the head (C: qwen4_no_final_norm
        // = hc_count > 0 && !st_have(norm.weight)).
        let mut logits = vec![0.0; c.vocab];
        if !self.final_norm.is_empty() {
            let mut normed = vec![0.0; d];
            rmsnorm_row(&mut normed, &out, &self.final_norm, c.eps);
            matmul(&mut logits, &normed, &self.lm_head);
        } else {
            matmul(&mut logits, &out, &self.lm_head);
        }
        logits
    }
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

fn load_wt(st: &StFile, name: &str, o: usize, i: usize) -> Result<Wt, String> {
    Ok(Wt { f: st.f32(name, &[o as u64, i as u64])?, bytes: None, o, i })
}

impl Model {
    pub fn load(st: &StFile, cfg: &Cfg) -> Result<Model, String> {
        let mut experts = Vec::new();
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let lp = format!("model.layers.{l}");
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
                st.f32(&format!("{lp}.input_layernorm.weight"), &[cfg.hidden as u64])?
            };
            let layer = Layer {
                in_ln,
                is_gdn,
                is_qsa,
                gdn_a_log: if is_gdn { st.f32(&format!("{lp}.linear_attn.A_log"), &[cfg.lin_v_heads as u64])? } else { vec![] },
                gdn_dt_bias: if is_gdn { st.f32(&format!("{lp}.linear_attn.dt_bias"), &[cfg.lin_v_heads as u64])? } else { vec![] },
                gdn_conv1d: if is_gdn { st.f32(&format!("{lp}.linear_attn.conv1d.weight"), &[(cdim * cfg.conv_kernel) as u64])? } else { vec![] },
                gdn_in_a: if is_gdn { load_wt(st, &format!("{lp}.linear_attn.in_proj_a.weight"), cfg.lin_v_heads, cfg.hidden)? } else { Wt { f: vec![], bytes: None, o: 0, i: 0  } },
                gdn_in_b: if is_gdn { load_wt(st, &format!("{lp}.linear_attn.in_proj_b.weight"), cfg.lin_v_heads, cfg.hidden)? } else { Wt { f: vec![], bytes: None, o: 0, i: 0  } },
                gdn_in_qkv: if is_gdn { load_wt(st, &format!("{lp}.linear_attn.in_proj_qkv.weight"), cdim, cfg.hidden)? } else { Wt { f: vec![], bytes: None, o: 0, i: 0  } },
                gdn_in_z: if is_gdn { load_wt(st, &format!("{lp}.linear_attn.in_proj_z.weight"), vdim, cfg.hidden)? } else { Wt { f: vec![], bytes: None, o: 0, i: 0  } },
                gdn_norm: if is_gdn { st.f32(&format!("{lp}.linear_attn.norm.weight"), &[cfg.lin_v_dim as u64])? } else { vec![] },
                gdn_out: if is_gdn { load_wt(st, &format!("{lp}.linear_attn.out_proj.weight"), cfg.hidden, vdim)? } else { Wt { f: vec![], bytes: None, o: 0, i: 0  } },
                attn_q: if !is_gdn { load_wt(st, &format!("{lp}.self_attn.q_proj.weight"), 2 * cfg.heads * hd, cfg.hidden)? } else { Wt { f: vec![], bytes: None, o: 0, i: 0  } },
                attn_k: if !is_gdn { load_wt(st, &format!("{lp}.self_attn.k_proj.weight"), cfg.kv_heads * hd, cfg.hidden)? } else { Wt { f: vec![], bytes: None, o: 0, i: 0  } },
                attn_v: if !is_gdn { load_wt(st, &format!("{lp}.self_attn.v_proj.weight"), cfg.kv_heads * hd, cfg.hidden)? } else { Wt { f: vec![], bytes: None, o: 0, i: 0  } },
                attn_o: if !is_gdn { load_wt(st, &format!("{lp}.self_attn.o_proj.weight"), cfg.hidden, cfg.heads * hd)? } else { Wt { f: vec![], bytes: None, o: 0, i: 0  } },
                attn_qn: if !is_gdn { st.f32(&format!("{lp}.self_attn.q_norm.weight"), &[hd as u64])? } else { vec![] },
                attn_kn: if !is_gdn { st.f32(&format!("{lp}.self_attn.k_norm.weight"), &[hd as u64])? } else { vec![] },
                index_qk: if is_qsa {
                    load_wt(st, &format!("{lp}.self_attn.indexer.index_qk_proj.weight"), cfg.idx_n_heads * cfg.idx_head_dim + cfg.idx_kv_heads * cfg.idx_head_dim, cfg.hidden)?
                } else {
                    Wt { f: vec![], bytes: None, o: 0, i: 0  }
                },
                idx_qn: if is_qsa { st.f32(&format!("{lp}.self_attn.indexer.q_layernorm.weight"), &[cfg.idx_head_dim as u64])? } else { vec![] },
                idx_kn: if is_qsa { st.f32(&format!("{lp}.self_attn.indexer.k_layernorm.weight"), &[cfg.idx_head_dim as u64])? } else { vec![] },
                hc_norm: st.f32(&format!("{lp}.attn_hyper_connection.hc_norm.weight"), &[hcd as u64])?,
                hc_mix_down: load_wt(st, &format!("{lp}.attn_hyper_connection.input_mix_weight_down.weight"), cfg.hc_lowrank, hcd)?,
                hc_mix_up: load_wt(st, &format!("{lp}.attn_hyper_connection.input_mix_weight_up.weight"), hcd, cfg.hc_lowrank)?,
                hc_inject: load_wt(st, &format!("{lp}.attn_hyper_connection.block_inject_weight.weight"), cfg.hc_count, hcd)?,
                hc_mlp_norm: st.f32(&format!("{lp}.mlp_hyper_connection.hc_norm.weight"), &[hcd as u64])?,
                hc_mlp_mix_down: load_wt(st, &format!("{lp}.mlp_hyper_connection.input_mix_weight_down.weight"), cfg.hc_lowrank, hcd)?,
                hc_mlp_mix_up: load_wt(st, &format!("{lp}.mlp_hyper_connection.input_mix_weight_up.weight"), hcd, cfg.hc_lowrank)?,
                hc_mlp_inject: load_wt(st, &format!("{lp}.mlp_hyper_connection.block_inject_weight.weight"), cfg.hc_count, hcd)?,
                router: load_wt(st, &format!("{lp}.mlp.gate.weight"), cfg.experts, cfg.hidden)?,
                se_gate: load_wt(st, &format!("{lp}.mlp.shared_expert.gate_proj.weight"), cfg.shared_inter, cfg.hidden)?,
                se_up: load_wt(st, &format!("{lp}.mlp.shared_expert.up_proj.weight"), cfg.shared_inter, cfg.hidden)?,
                se_down: load_wt(st, &format!("{lp}.mlp.shared_expert.down_proj.weight"), cfg.hidden, cfg.shared_inter)?,
                se_g: load_wt(st, &format!("{lp}.mlp.shared_expert_gate.weight"), 1, cfg.hidden)?,
            };
            let mut layer_experts = Vec::new();
            for e in 0..cfg.experts {
                let elp = format!("{lp}.mlp.experts.{e}");
                let gu = st.f32(&format!("{elp}.gate_up_proj"), &[(2 * cfg.moe_inter) as u64, cfg.hidden as u64])?;
                let dn = st.f32(&format!("{elp}.down_proj"), &[cfg.hidden as u64, cfg.moe_inter as u64])?;
                let half = cfg.moe_inter * cfg.hidden;
                let gate = Wt { f: gu[..half].to_vec(), bytes: None, o: cfg.moe_inter, i: cfg.hidden };
                let up = Wt { f: gu[half..].to_vec(), bytes: None, o: cfg.moe_inter, i: cfg.hidden };
                let down = Wt { f: dn, bytes: None, o: cfg.hidden, i: cfg.moe_inter };
                layer_experts.push([gate, up, down]);
            }
            experts.push(layer_experts);
            layers.push(layer);
        }

        // PLE geometry
        let mut ple_offsets = Vec::new();
        let mut ple_sizes = Vec::new();
        if cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let mut total = 0_i64;
            for h in 0..cfg.ngram_heads {
                let size = nth_prime_after(cfg.ngram_vocab_base - 1, h as i64 + 1);
                ple_sizes.push(size);
                ple_offsets.push(total);
                total += size;
            }
        }
        let ple_embed: Wt = if cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let total: i64 = ple_sizes.iter().sum();
            let padded = (total + cfg.ngram_div - 1) / cfg.ngram_div * cfg.ngram_div;
            let hd_per = cfg.ple_embed_dim / cfg.ngram_heads;
            load_wt(st, "model.ple.ple_embedding.ngram_embedding.weight", padded as usize, hd_per)?
        } else {
            Wt { f: vec![], bytes: None, o: 0, i: 0  }
        };
        let ple_key_proj = if cfg.ple_layer >= 0 {
            load_wt(st, "model.ple.key_proj.weight", cfg.hc_count * cfg.hidden, cfg.ple_embed_dim)?
        } else {
            Wt { f: vec![], bytes: None, o: 0, i: 0  }
        };
        let ple_value_proj = if cfg.ple_layer >= 0 {
            load_wt(st, "model.ple.value_proj.weight", cfg.hidden, cfg.ple_embed_dim)?
        } else {
            Wt { f: vec![], bytes: None, o: 0, i: 0  }
        };
        let hcd = cfg.hc_count * cfg.hidden;
        let ple_norm_key = if cfg.ple_layer >= 0 { st.f32("model.ple.norm_key.weight", &[hcd as u64])? } else { vec![] };
        let ple_norm_query = if cfg.ple_layer >= 0 { st.f32("model.ple.norm_query.weight", &[hcd as u64])? } else { vec![] };
        let ple_norm_conv = if cfg.ple_layer >= 0 { st.f32("model.ple.norm_conv.weight", &[hcd as u64])? } else { vec![] };
        let ple_conv1d = if cfg.ple_layer >= 0 {
            st.f32("model.ple.conv1d.weight", &[(hcd * cfg.ple_conv_kernel) as u64])?
        } else {
            vec![]
        };
        // ple_mult: odd multipliers from splitmix64
        let mut ple_mult = Vec::new();
        if cfg.ple_layer >= 0 && cfg.ngram_heads > 0 {
            let max_long = i64::MAX;
            let mult_max = max_long / (cfg.vocab.max(1) as i64);
            let half = (mult_max / 2).max(1);
            let base = cfg.seed as u64 + 10007_u64.wrapping_mul(cfg.ple_layer as u64);
            for i in 0..cfg.ngram_size {
                let v = ple_splitmix64(base.wrapping_add(PLE_GAMMA.wrapping_mul((i + 1) as u64)));
                ple_mult.push(2 * (v % half as u64) + 1);
            }
        }

        Ok(Model {
            cfg: cfg.clone(),
            coli: None,
            embed: load_wt(st, "model.embed_tokens.weight", cfg.vocab, cfg.hidden)?,
            lm_head: load_wt(st, "lm_head.weight", cfg.vocab, cfg.hidden)?,
            // qwen4 drops norm.weight when hyper connections are active
            final_norm: match st.f32("model.norm.weight", &[cfg.hidden as u64]) {
                Ok(v) => v,
                Err(_) if cfg.hc_count > 0 => vec![],
                Err(e) => return Err(e),
            },
            layers,
            experts,
            hc_global: HcGlobal {
                norm: st.f32("model.hyper_connection_mixer.hc_norm.weight", &[hcd as u64])?,
                mix_down: load_wt(st, "model.hyper_connection_mixer.input_mix_weight_down.weight", cfg.hc_lowrank, hcd)?,
                mix_up: load_wt(st, "model.hyper_connection_mixer.input_mix_weight_up.weight", hcd, cfg.hc_lowrank)?,
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

/// Standalone runner for `coli run`: loads a .coli package (or safetensors
/// fixture dir) and greedy-decodes `max_new` tokens from `prompt`.
/// Returns the generated token ids (prompt excluded).
pub fn run_greedy(package_dir: &std::path::Path, prompt: &[u32], max_new: usize) -> Result<Vec<u32>, String> {
    let cfg = load_cfg(&package_dir.join("config.json"))?;
    let model = if package_dir.join("model.safetensors").exists() {
        let st = StFile::open(&package_dir.join("model.safetensors"))?;
        Model::load(&st, &cfg)?
    } else {
        let src = colisource::ColiSource::open(package_dir)?;
        Model::load_coli(&src, &cfg)?
    };
    Ok(run_greedy_with(model, cfg, prompt, max_new))
}

/// Greedy decode against an already-loaded model.
pub fn run_greedy_with(mut model: Model, cfg: Cfg, prompt: &[u32], max_new: usize) -> Vec<u32> {
    for (i, &t) in prompt.iter().enumerate() {
        model.forward_token(t as usize, i);
    }
    let mut out = Vec::with_capacity(max_new);
    let mut last = *prompt.last().unwrap_or(&0);
    let start = prompt.len();
    for pos in start..start + max_new {
        let logits = model.forward_token(last as usize, pos);
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        out.push(next);
        last = next;
    }
    out
}


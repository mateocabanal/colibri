//! .coli package weight source for qwen4-rs (M2 real-model path).
//!
//! Mirrors the C runtime's dual-probe (HF-prefixed then canonical) and keeps
//! everything in its resident form:
//! - dense tensors: BF16 bytes (decode in matmul)
//! - experts: Apple8 MXFP4 tiles (rANS or raw) -> BF16 bytes, read on demand
//! - PLE ngram: per-shard record, rows read on demand (51 GB never resident)

use std::path::Path;

use colibri_format::codecs::{
    apple8_mxfp4_decode, RansTable, INT4_MATH_FORMAT, INT4_SCALE_FORMAT, RANS_CODEC_ID,
};
use colibri_format::package::{Package, RecordInfo};

const HF_PREFIX: &str = "model.language_model.";

#[derive(Clone)]
pub struct ColiSource {
    pkg: Package,
}

/// Resident weight: BF16 bytes + shape. matmul decodes on the fly.
pub struct ColiWt {
    pub bytes: Vec<u8>, // BF16 little-endian
    pub o: usize,
    pub i: usize,
}

impl ColiSource {
    pub fn open(dir: &Path) -> Result<ColiSource, String> {
        Ok(ColiSource {
            pkg: Package::open(dir).map_err(|e| e.to_string())?,
        })
    }

    /// Dual-probe record lookup: prefixed (resident HF) then bare canonical.
    pub(crate) fn rec(&self, name: &str) -> Option<&RecordInfo> {
        let pref = format!("{HF_PREFIX}{name}");
        self.pkg
            .record_by_name(&pref)
            .or_else(|| self.pkg.record_by_name(name))
    }

    /// Dense vector tensor -> BF16 bytes.
    pub fn vec(&self, name: &str, want: usize) -> Result<Vec<u8>, String> {
        let rec = self.rec(name).ok_or_else(|| format!("missing dense tensor {name}"))?;
        let payload = self.pkg.read_tensor_payload(rec).map_err(|e| e.to_string())?;
        if payload.len() != want * 2 {
            return Err(format!(
                "{name}: payload {} bytes != expected {}",
                payload.len(),
                want * 2
            ));
        }
        Ok(payload)
    }

    /// Dense matrix -> BF16 bytes, rows x cols.
    pub fn wt(&self, name: &str, o: usize, i: usize) -> Result<ColiWt, String> {
        let rec = self.rec(name).ok_or_else(|| format!("missing dense matrix {name}"))?;
        let payload = self.pkg.read_tensor_payload(rec).map_err(|e| e.to_string())?;
        let want = o * i * 2; // BF16
        if payload.len() != want {
            return Err(format!(
                "{name}: payload {} bytes != expected {want} ({o}x{i})",
                payload.len()
            ));
        }
        Ok(ColiWt {
            bytes: payload,
            o,
            i,
        })
    }

    /// Routed expert (layer, expert) -> [gate, up, down] BF16 bytes,
    /// decoding Apple8 MXFP4 tiles or reading BF16 canonical payloads.
    /// Each call reads the record fresh (ponytail: no LRU yet; add when
    /// profiling shows disk-bound decode).
    pub fn expert_matrices(&self, layer: i32, expert: i32) -> Result<[ColiWt; 3], String> {
        let recs = self.pkg.expert_records(layer, expert);
        let rec = recs
            .first()
            .ok_or_else(|| format!("missing expert ({layer},{expert})"))?;
        let raw = self.pkg.read_record(rec).map_err(|e| e.to_string())?;
        assert_eq!(&raw[..8], b"COLIEXPT");
        let desc_size = u32::from_le_bytes(raw[28..32].try_into().unwrap()) as usize;
        let mut out: Vec<ColiWt> = Vec::with_capacity(3);
        for i in 0..3 {
            let d = 64 + i * desc_size;
            let role = u16::from_le_bytes(raw[d..d + 2].try_into().unwrap());
            let math = u16::from_le_bytes(raw[d + 4..d + 6].try_into().unwrap());
            let scale = u16::from_le_bytes(raw[d + 6..d + 8].try_into().unwrap());
            let wc = u16::from_le_bytes(raw[d + 8..d + 10].try_into().unwrap());
            let wt = u32::from_le_bytes(raw[d + 40..d + 44].try_into().unwrap());
            let rows = u64::from_le_bytes(raw[d + 16..d + 24].try_into().unwrap());
            let cols = u64::from_le_bytes(raw[d + 24..d + 32].try_into().unwrap());
            let w_off = u64::from_le_bytes(raw[d + 48..d + 56].try_into().unwrap());
            let w_stored = u64::from_le_bytes(raw[d + 56..d + 64].try_into().unwrap());
            let w_decoded = u64::from_le_bytes(raw[d + 64..d + 72].try_into().unwrap());
            let s_off = u64::from_le_bytes(raw[d + 72..d + 80].try_into().unwrap());
            let s_stored = u64::from_le_bytes(raw[d + 80..d + 88].try_into().unwrap());
            let w = &raw[w_off as usize..(w_off + w_stored) as usize];

            // All expert representations decode to BF16 bytes.
            let bytes: Vec<u8> = match (math, scale) {
                // BF16 canonical: raw bytes already
                (0x0003, 0x0000) => w.to_vec(),
                // INT4-G32: dequant to BF16
                (INT4_MATH_FORMAT, INT4_SCALE_FORMAT) => {
                    let s = &raw[s_off as usize..(s_off + s_stored) as usize];
                    let f =
                        colibri_format::codecs::int4_grouped_decode(w, s, rows as usize, cols as usize)
                            .map_err(|e| e.to_string())?;
                    f.into_iter().flat_map(bf16_bytes).collect()
                }
                // Apple8 MXFP4: rANS or raw tiles -> f32 -> BF16
                (0x0020, 0x0004) => {
                    let tiles: Vec<u8> = if wc == RANS_CODEC_ID {
                        let table = RansTable::from_manifest(&self.pkg.manifest_ref(), wt, wc)
                            .map_err(|e| e.to_string())?;
                        colibri_format::codecs::apple8_decode(w, &table, rows, cols)
                            .map_err(|e| e.to_string())?
                    } else {
                        if w.len() as u64 != w_decoded {
                            return Err(format!("expert {layer}/{expert} m{i} raw tile size"));
                        }
                        w.to_vec()
                    };
                    let f = apple8_mxfp4_decode(&tiles, rows, cols).map_err(|e| e.to_string())?;
                    f.into_iter().flat_map(bf16_bytes).collect()
                }
                _ => {
                    return Err(format!(
                        "expert {layer}/{expert} m{i} unsupported math=0x{math:04x} scale=0x{scale:04x}"
                    ))
                }
            };
            out.push(ColiWt {
                bytes,
                o: rows as usize,
                i: cols as usize,
            });
        }
        Ok([out.remove(0), out.remove(0), out.remove(0)])
    }

    /// The layer that actually carries PLE tensors in this package (the
    /// frontend's resolved ple_layer, which can differ from config's
    /// ple_layer_ids). Ground truth = first `layers.N.ple.` record.
    pub fn ple_layer(&self) -> Option<i32> {
        self.pkg.records().iter().find_map(|r| {
            let nm = r.name.as_deref()?;
            let rest = nm.strip_prefix("layers.")?;
            let dot = rest.find('.')?;
            let layer: i32 = rest[..dot].parse().ok()?;
            if nm.contains(".ple.") {
                Some(layer)
            } else {
                None
            }
        })
    }

    /// PLE metadata from the package (ground truth, i64 records): per-head
    /// vocab sizes, cumulative offsets, and the layer multipliers. The
    /// config-derived prime math diverges on the real model (row 173M vs
    /// computed 160M capacity), so .coli mode reads these instead.
    pub fn ple_metadata(&self, layer: i32) -> Result<(Vec<i64>, Vec<i64>, Vec<u64>), String> {
        let sizes = self.i64_tensor(&format!("layers.{layer}.ple.ple_embedding.ngram_heads_vocab_sizes"))?;
        let offsets = self.i64_tensor(&format!("layers.{layer}.ple.ple_embedding.ngram_heads_offsets"))?;
        let mult = self.i64_tensor(&format!("layers.{layer}.ple.ple_embedding.layer_multipliers"))?;
        Ok((sizes, offsets, mult.into_iter().map(|m| m as u64).collect()))
    }

    fn i64_tensor(&self, name: &str) -> Result<Vec<i64>, String> {
        let rec = self.rec(name).ok_or_else(|| format!("missing PLE metadata {name}"))?;
        let payload = self.pkg.read_tensor_payload(rec).map_err(|e| e.to_string())?;
        // i64 records (8 bytes/elem: 16 heads x 8 = 128 bytes for
        // vocab_sizes/offsets; 3 x 8 = 24 for multipliers).
        if payload.len() % 8 != 0 {
            return Err(format!("{name}: payload {} not 8-aligned", payload.len()));
        }
        Ok(payload
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// PLE n-gram row fetch (F8 E4M3 shards): row `r` of the ngram table.
    /// The real package stores it as F8 shards (1 byte/elem, math=0x10) with
    /// a global BF16 scale; the C engine preads hd_per bytes per row, never
    /// loading the 400 MB shard. The tiny fixture's BF16 single-tensor form
    /// is handled by the safetensors loader (not this path).
    pub fn ple_ngram_row_f8(&self, layer: i32, r: u64, hd_per: usize) -> Result<Vec<u8>, String> {
        let prefix = format!("layers.{layer}.ple.ple_embedding.ngram_embedding.shard_");
        let recs: Vec<&RecordInfo> = self
            .pkg
            .records()
            .iter()
            .filter(|rec| {
                rec.kind == 1
                    && rec
                        .name
                        .as_deref()
                        .map(|n| n.starts_with(&prefix))
                        .unwrap_or(false)
            })
            .collect();
        if recs.is_empty() {
            return Err(format!("no ngram shards for layer {layer}"));
        }
        let rps = recs[0].decoded as u64 / hd_per as u64; // rows per shard (F8: 1 byte/row-elem)
        let shard_idx = (r / rps) as usize;
        let rec = recs
            .get(shard_idx)
            .ok_or_else(|| format!("ngram row {r} out of range ({} shards)", recs.len()))?;
        let within = (r % rps) as u64;
        // pread only this row (F8: hd_per bytes) — never the whole shard
        self.pkg
            .read_payload_range(rec, within * hd_per as u64, hd_per)
            .map_err(|e| e.to_string())
    }

    /// Global BF16 scale for the F8 ngram table.
    pub fn ple_ngram_scale(&self, layer: i32) -> Result<f32, String> {
        let rec = self
            .rec(&format!("layers.{layer}.ple.ple_embedding.ngram_embedding.weight_scale"))
            .ok_or_else(|| format!("missing ngram weight_scale"))?;
        let payload = self.pkg.read_tensor_payload(rec).map_err(|e| e.to_string())?;
        if payload.len() != 2 {
            return Err(format!("weight_scale payload {} != 2", payload.len()));
        }
        let u = u16::from_le_bytes(payload.try_into().unwrap());
        Ok(f32::from_bits((u as u32) << 16))
    }

    /// F8 E4M3 decode (bit-exact with the C engine's E4M3_LUT).
    pub fn e4m3_decode(b: u8) -> f32 {
        let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let exp = ((b >> 3) & 0x0f) as i32;
        let mant = (b & 0x07) as f32;
        match exp {
            0 => sign * mant * 0.001953125,   // subnormal: 2^-9
            0x0f => sign * f32::INFINITY,     // NaN/Inf
            e => sign * (1.0 + mant * 0.125) * 2f32.powi(e - 7),
        }
    }

}

/// f32 -> BF16 (top 16 bits, round-to-nearest-even), as 2 LE bytes.
pub fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

pub fn bf16_bytes(f: f32) -> [u8; 2] {
    f32_to_bf16(f).to_le_bytes()
}

pub fn bf16_to_f32(u: u16) -> f32 {
    f32::from_bits((u as u32) << 16)
}

//! .coli package weight source for qwen4-rs.
//!
//! Mirrors the C runtime's `coli_vec`/`coli_wt` dual-probe: dense/layer-static
//! tensors live under the resident HF prefix `model.language_model.` in the
//! package, expert matrices are canonical per-(layer, expert) records. This
//! loader resolves either spelling and decodes the record payload into f32.
//!
//! Apple8 experts: resident payload = rANS-decoded tile execution bytes
//! (E2M1 nibbles + E8M0 scales) -> f32 via apple8_mxfp4_decode.
//! BF16 experts (exact): raw f32 bytes.
//! Dense BF16 tensors: raw f32 bytes.

use std::path::Path;

use colibri_format::codecs::{
    apple8_mxfp4_decode, RansTable, INT4_MATH_FORMAT, INT4_SCALE_FORMAT, RANS_CODEC_ID,
};
use colibri_format::package::{Package, RecordInfo};

const HF_PREFIX: &str = "model.language_model.";

pub struct ColiSource {
    pkg: Package,
}

pub struct ColiWt {
    pub f: Vec<f32>,
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
    fn rec(&self, name: &str) -> Option<&RecordInfo> {
        let pref = format!("{HF_PREFIX}{name}");
        self.pkg
            .record_by_name(&pref)
            .or_else(|| self.pkg.record_by_name(name))
    }

    /// Dense/vector tensor -> f32 (BF16 canonical payload).
    pub fn vec(&self, name: &str, want: usize) -> Result<Vec<f32>, String> {
        let rec = self.rec(name).ok_or_else(|| format!("missing dense tensor {name}"))?;
        let payload = self.pkg.read_tensor_payload(rec).map_err(|e| e.to_string())?;
        if payload.len() != want * 4 {
            return Err(format!(
                "{name}: payload {} bytes != expected {}",
                payload.len(),
                want * 4
            ));
        }
        Ok(payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// Dense matrix -> f32 (BF16 canonical payload), rows x cols.
    pub fn wt(&self, name: &str, o: usize, i: usize) -> Result<ColiWt, String> {
        let rec = self.rec(name).ok_or_else(|| format!("missing dense matrix {name}"))?;
        let payload = self.pkg.read_tensor_payload(rec).map_err(|e| e.to_string())?;
        let want = o * i * 4;
        if payload.len() != want {
            return Err(format!(
                "{name}: payload {} bytes != expected {want} ({o}x{i})",
                payload.len()
            ));
        }
        Ok(ColiWt {
            f: payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            o,
            i,
        })
    }

    /// Routed expert matrices for (layer, expert): returns [gate, up, down]
    /// as f32, decoding Apple8 (MXFP4 tiles) or BF16 canonical payloads.
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

            let f: Vec<f32> = match (math, scale) {
                // BF16 canonical: raw f32 bytes
                (0x0003, 0x0000) => {
                    if w_stored != w_decoded || w.len() != rows as usize * cols as usize * 2 {
                        return Err(format!("expert {layer}/{expert} m{i} BF16 size mismatch"));
                    }
                    // BF16 -> f32 (bfloat16: top 16 bits)
                    w.chunks_exact(2)
                        .map(|c| {
                            let u = u16::from_le_bytes(c.try_into().unwrap());
                            f32::from_bits((u as u32) << 16)
                        })
                        .collect()
                }
                // INT4-G32: packed weights ++ LE f32 scales
                (INT4_MATH_FORMAT, INT4_SCALE_FORMAT) => {
                    let s = &raw[s_off as usize..(s_off + s_stored) as usize];
                    colibri_format::codecs::int4_grouped_decode(
                        w,
                        s,
                        rows as usize,
                        cols as usize,
                    )
                    .map_err(|e| e.to_string())?
                }
                // Apple8 MXFP4: rANS-decoded tile execution bytes -> f32
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
                    let mut f = apple8_mxfp4_decode(&tiles, rows, cols).map_err(|e| e.to_string())?;
                    // gate/up/down roles: role 1 = gate, 2 = up, 3 = down
                    let _ = &mut f;
                    f
                }
                _ => {
                    return Err(format!(
                        "expert {layer}/{expert} m{i} unsupported math=0x{math:04x} scale=0x{scale:04x}"
                    ))
                }
            };
            let (o, i) = match role {
                1 | 2 => (rows as usize, cols as usize), // gate/up: [I, H]
                _ => (rows as usize, cols as usize),      // down: [H, I]
            };
            out.push(ColiWt { f, o, i });
        }
        Ok([out.remove(0), out.remove(0), out.remove(0)])
    }
}

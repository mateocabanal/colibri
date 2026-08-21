use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use crate::{
    error::{ColicError, Result},
    target_registry::{
        layout_registered, profile_allows_layout, profile_by_name, APPLE8_MXFP4_GROUP_SIZE,
        APPLE8_MXFP4_MATH_FORMAT, APPLE8_MXFP4_SCALE_BLOCK_COLUMNS,
        APPLE8_MXFP4_SCALE_BLOCK_ROWS, APPLE8_MXFP4_SCALE_FORMAT,
        APPLE8_MXFP4_TILE_BYTES, APPLE8_MXFP4_TILE_COLUMNS, APPLE8_MXFP4_TILE_LAYOUT,
        APPLE8_MXFP4_TILE_ROWS, APPLE8_PROFILE_NAME,
    },
};

const CODEC_NONE: u16 = 0;
const REC_EXPERT: u16 = 2;
const PREFIX: usize = 64 + 3 * 128;

#[derive(Clone, Copy)]
struct Matrix {
    role: u16,
    math: u16,
    scale: u16,
    wc: u16,
    sc: u16,
    layout: u16,
    rows: u64,
    cols: u64,
    sr: u32,
    sk: u32,
    wt: u32,
    st: u32,
    wo: u64,
    ws: u64,
    wd: u64,
    so: u64,
    ss: u64,
    sd: u64,
    crc: u32,
    group: u32,
}

fn bad(s: impl Into<String>) -> ColicError {
    ColicError::Usage(format!("invalid target-layout package: {}", s.into()))
}
fn u16a(b: &[u8], o: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        b.get(o..o + 2)
            .ok_or_else(|| bad("truncated u16"))?
            .try_into()
            .unwrap(),
    ))
}
fn u32a(b: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        b.get(o..o + 4)
            .ok_or_else(|| bad("truncated u32"))?
            .try_into()
            .unwrap(),
    ))
}
fn i32a(b: &[u8], o: usize) -> Result<i32> {
    Ok(i32::from_le_bytes(
        b.get(o..o + 4)
            .ok_or_else(|| bad("truncated i32"))?
            .try_into()
            .unwrap(),
    ))
}
fn u64a(b: &[u8], o: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        b.get(o..o + 8)
            .ok_or_else(|| bad("truncated u64"))?
            .try_into()
            .unwrap(),
    ))
}

fn string_at(m: &[u8], id: u32) -> Result<&str> {
    let count = u32a(m, 28)?;
    if id >= count {
        return Err(bad("string id out of range"));
    }
    let table = usize::try_from(u64a(m, 80)?).map_err(|_| bad("string table offset"))?;
    let d = table
        .checked_add(id as usize * 16)
        .ok_or_else(|| bad("string descriptor overflow"))?;
    let rel = usize::try_from(u64a(m, d)?).map_err(|_| bad("string offset"))?;
    let len = u32a(m, d + 8)? as usize;
    let s = table
        .checked_add(rel)
        .ok_or_else(|| bad("string offset overflow"))?;
    let e = s
        .checked_add(len)
        .ok_or_else(|| bad("string length overflow"))?;
    std::str::from_utf8(m.get(s..e).ok_or_else(|| bad("string outside manifest"))?)
        .map_err(|_| bad("invalid UTF-8"))
}

fn tile_bytes(rows: u64, cols: u64) -> Result<u64> {
    if rows == 0 || cols == 0 {
        return Err(bad("zero matrix dimension"));
    }
    let rt = rows / APPLE8_MXFP4_TILE_ROWS
        + u64::from(!rows.is_multiple_of(APPLE8_MXFP4_TILE_ROWS));
    let kg = cols / APPLE8_MXFP4_TILE_COLUMNS
        + u64::from(!cols.is_multiple_of(APPLE8_MXFP4_TILE_COLUMNS));
    rt.checked_mul(kg)
        .and_then(|v| v.checked_mul(APPLE8_MXFP4_TILE_BYTES))
        .ok_or_else(|| bad("matrix bytes overflow"))
}

fn matrix(p: &[u8], i: usize) -> Result<Matrix> {
    let d = 64 + i * 128;
    if u16a(p, d + 2)? != 0
        || u16a(p, d + 14)? != 0
        || u32a(p, d + 100)? != 0
        || u32a(p, d + 108)? != 0
        || p.get(d + 112..d + 128)
            .is_none_or(|x| x.iter().any(|v| *v != 0))
    {
        return Err(bad(format!("matrix {i} reserved fields")));
    }
    Ok(Matrix {
        role: u16a(p, d)?,
        math: u16a(p, d + 4)?,
        scale: u16a(p, d + 6)?,
        wc: u16a(p, d + 8)?,
        sc: u16a(p, d + 10)?,
        layout: u16a(p, d + 12)?,
        rows: u64a(p, d + 16)?,
        cols: u64a(p, d + 24)?,
        sr: u32a(p, d + 32)?,
        sk: u32a(p, d + 36)?,
        wt: u32a(p, d + 40)?,
        st: u32a(p, d + 44)?,
        wo: u64a(p, d + 48)?,
        ws: u64a(p, d + 56)?,
        wd: u64a(p, d + 64)?,
        so: u64a(p, d + 72)?,
        ss: u64a(p, d + 80)?,
        sd: u64a(p, d + 88)?,
        crc: u32a(p, d + 96)?,
        group: u32a(p, d + 104)?,
    })
}

fn valid_apple_matrix(i: usize, m: Matrix) -> Result<u64> {
    let n = tile_bytes(m.rows, m.cols)?;
    if m.role != (i + 1) as u16
        || m.layout != APPLE8_MXFP4_TILE_LAYOUT
        || m.math != APPLE8_MXFP4_MATH_FORMAT
        || m.scale != APPLE8_MXFP4_SCALE_FORMAT
        || m.sr != APPLE8_MXFP4_SCALE_BLOCK_ROWS
        || m.sk != APPLE8_MXFP4_SCALE_BLOCK_COLUMNS
        || m.group != APPLE8_MXFP4_GROUP_SIZE
        || m.wc != CODEC_NONE
        || m.wt != 0
        || m.wo == 0
        || !m.wo.is_multiple_of(16)
        || m.ws != n
        || m.wd != n
        || m.sc != CODEC_NONE
        || m.st != 0
        || m.so != 0
        || m.ss != 0
        || m.sd != 0
    {
        return Err(bad(format!(
            "matrix {i} violates Apple8 Design-A descriptor"
        )));
    }
    Ok(n)
}

fn crc_range(f: &mut File, off: u64, bytes: u64) -> Result<u32> {
    f.seek(SeekFrom::Start(off))
        .map_err(|source| ColicError::Io {
            path: PathBuf::from("<target shard>"),
            source,
        })?;
    let mut left = bytes;
    let mut state = !0u32;
    let mut buf = [0u8; 65536];
    while left != 0 {
        let n = left.min(buf.len() as u64) as usize;
        f.read_exact(&mut buf[..n])
            .map_err(|source| ColicError::Io {
                path: PathBuf::from("<target shard>"),
                source,
            })?;
        for x in &buf[..n] {
            state ^= *x as u32;
            for _ in 0..8 {
                state = (state >> 1) ^ (0x82f6_3b78 & (0u32.wrapping_sub(state & 1)));
            }
        }
        left -= n as u64;
    }
    Ok(!state)
}

fn padding(f: &mut File, base: u64, m: Matrix) -> Result<()> {
    let rr = m.rows % APPLE8_MXFP4_TILE_ROWS;
    let cr = m.cols % APPLE8_MXFP4_TILE_COLUMNS;
    let rts = m.rows.div_ceil(APPLE8_MXFP4_TILE_ROWS);
    let gs = m.cols.div_ceil(APPLE8_MXFP4_TILE_COLUMNS);
    let mut t = [0u8; APPLE8_MXFP4_TILE_BYTES as usize];
    for ot in 0..rts {
        for g in 0..gs {
            let re = rr != 0 && ot + 1 == rts;
            let ce = cr != 0 && g + 1 == gs;
            if !re && !ce {
                continue;
            }
            let idx = ot
                .checked_mul(gs)
                .and_then(|v| v.checked_add(g))
                .ok_or_else(|| bad("tile index overflow"))?;
            let off = base
                .checked_add(m.wo)
                .and_then(|v| v.checked_add(idx.checked_mul(APPLE8_MXFP4_TILE_BYTES)?))
                .ok_or_else(|| bad("tile offset overflow"))?;
            f.seek(SeekFrom::Start(off))
                .map_err(|source| ColicError::Io {
                    path: PathBuf::from("<target shard>"),
                    source,
                })?;
            f.read_exact(&mut t).map_err(|source| ColicError::Io {
                path: PathBuf::from("<target shard>"),
                source,
            })?;
            for r in 0..APPLE8_MXFP4_TILE_ROWS as usize {
                let row = &t[r * 16..r * 16 + 16];
                let logical = !re || r < rr as usize;
                if !logical {
                    if row.iter().any(|v| *v != 0) || t[128 + r] != 0 {
                        return Err(bad("nonzero output-row padding"));
                    }
                } else if ce {
                    let used = cr.div_ceil(2) as usize;
                    if row[used..].iter().any(|v| *v != 0)
                        || (!cr.is_multiple_of(2) && row[used - 1] & 0xf0 != 0)
                    {
                        return Err(bad("nonzero K padding"));
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn verify_target_layouts(package: &Path) -> Result<()> {
    let mp = package.join("manifest.coli");
    let m = fs::read(&mp).map_err(|source| ColicError::Io {
        path: mp,
        source,
    })?;
    let profile_name = string_at(&m, u32a(&m, 148)?)?;
    let profile = profile_by_name(profile_name)
        .ok_or_else(|| bad(format!("unknown target profile `{profile_name}`")))?;

    let ns = u32a(&m, 40)? as usize;
    let st = usize::try_from(u64a(&m, 48)?).map_err(|_| bad("shard table offset"))?;
    let mut paths = Vec::with_capacity(ns);
    for i in 0..ns {
        let d = st
            .checked_add(i * 64)
            .ok_or_else(|| bad("shard table overflow"))?;
        paths.push(package.join(string_at(&m, u32a(&m, d + 8)?)?));
    }
    let mut files = paths
        .iter()
        .map(|p| {
            File::open(p).map_err(|source| ColicError::Io {
                path: p.clone(),
                source,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let nr = usize::try_from(u64a(&m, 32)?).map_err(|_| bad("record count"))?;
    let rt = usize::try_from(u64a(&m, 64)?).map_err(|_| bad("record table offset"))?;
    for ri in 0..nr {
        let d = rt
            .checked_add(ri * 96)
            .ok_or_else(|| bad("record table overflow"))?;
        if u16a(&m, d + 8)? != REC_EXPERT {
            continue;
        }
        let si = u32a(&m, d + 20)? as usize;
        let ro = u64a(&m, d + 40)?;
        let stored = u64a(&m, d + 48)?;
        let decoded = u64a(&m, d + 56)?;
        let layer = i32a(&m, d + 28)?;
        let expert = i32a(&m, d + 32)?;
        let f = files.get_mut(si).ok_or_else(|| bad("invalid shard"))?;
        let mut p = [0u8; PREFIX];
        f.seek(SeekFrom::Start(ro))
            .map_err(|source| ColicError::Io {
                path: paths[si].clone(),
                source,
            })?;
        f.read_exact(&mut p).map_err(|source| ColicError::Io {
            path: paths[si].clone(),
            source,
        })?;
        if &p[..8] != b"COLIEXPT"
            || u16a(&p, 8)? != 1
            || u32a(&p, 12)? != 64
            || i32a(&p, 16)? != layer
            || i32a(&p, 20)? != expert
            || u16a(&p, 24)? != 3
            || u32a(&p, 28)? != 128
            || u64a(&p, 32)? != 64
            || u64a(&p, 40)? != PREFIX as u64
        {
            return Err(bad(format!("expert {ri} envelope")));
        }

        let mut matrices = [matrix(&p, 0)?, matrix(&p, 1)?, matrix(&p, 2)?];
        for (mi, x) in matrices.iter().enumerate() {
            if !layout_registered(x.layout) {
                return Err(bad(format!(
                    "expert {layer}/{expert} matrix {mi} uses unknown layout 0x{:04x}",
                    x.layout
                )));
            }
            if !profile_allows_layout(profile, x.layout) {
                return Err(bad(format!(
                    "expert {layer}/{expert} matrix {mi} layout 0x{:04x} is outside profile `{profile_name}`",
                    x.layout
                )));
            }
        }

        if profile_name != APPLE8_PROFILE_NAME {
            continue;
        }
        if u16a(&m, d + 10)? != CODEC_NONE {
            return Err(bad("Apple8 expert outer codec must be NONE in PR1"));
        }
        let mut total = 0u64;
        let mut spans = Vec::with_capacity(3);
        for (mi, x) in matrices.iter_mut().enumerate() {
            let n = valid_apple_matrix(mi, *x)?;
            let end = x
                .wo
                .checked_add(x.ws)
                .ok_or_else(|| bad("matrix span overflow"))?;
            if x.wo < PREFIX as u64 || end > stored {
                return Err(bad("matrix outside record"));
            }
            spans.push((x.wo, end));
            total = total
                .checked_add(n)
                .ok_or_else(|| bad("resident bytes overflow"))?;
            let abs = ro
                .checked_add(x.wo)
                .ok_or_else(|| bad("file offset overflow"))?;
            if crc_range(f, abs, x.wd)? != x.crc {
                return Err(bad(format!(
                    "expert {layer}/{expert} matrix {mi} logical CRC"
                )));
            }
            padding(f, ro, *x)?;
        }
        spans.sort_unstable();
        if spans.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(bad("matrix spans overlap"));
        }
        if total != decoded || u64a(&p, 48)? != decoded {
            return Err(bad("resident byte total"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good(r: u64, c: u64) -> Matrix {
        let n = tile_bytes(r, c).unwrap();
        Matrix {
            role: 1,
            math: APPLE8_MXFP4_MATH_FORMAT,
            scale: APPLE8_MXFP4_SCALE_FORMAT,
            wc: 0,
            sc: 0,
            layout: APPLE8_MXFP4_TILE_LAYOUT,
            rows: r,
            cols: c,
            sr: APPLE8_MXFP4_SCALE_BLOCK_ROWS,
            sk: APPLE8_MXFP4_SCALE_BLOCK_COLUMNS,
            wt: 0,
            st: 0,
            wo: 448,
            ws: n,
            wd: n,
            so: 0,
            ss: 0,
            sd: 0,
            crc: 1,
            group: APPLE8_MXFP4_GROUP_SIZE,
        }
    }

    #[test]
    fn sizes() {
        for (r, c, n) in [
            (1, 1, 136),
            (1, 31, 136),
            (1, 32, 136),
            (1, 33, 272),
            (7, 32, 136),
            (8, 32, 136),
            (9, 32, 272),
            (8, 31, 136),
            (8, 33, 272),
            (9, 33, 544),
        ] {
            assert_eq!(tile_bytes(r, c).unwrap(), n);
        }
        assert!(tile_bytes(0, 32).is_err());
        assert!(tile_bytes(u64::MAX, u64::MAX).is_err());
    }

    #[test]
    fn refusal() {
        assert_eq!(valid_apple_matrix(0, good(9, 33)).unwrap(), 544);
        let mut x = good(9, 33);
        x.layout = 0x7131;
        assert!(valid_apple_matrix(0, x).is_err());
        let mut x = good(9, 33);
        x.so = 16;
        assert!(valid_apple_matrix(0, x).is_err());
        let mut x = good(9, 33);
        x.wd -= 1;
        assert!(valid_apple_matrix(0, x).is_err());
        let mut x = good(9, 33);
        x.sk = 16;
        assert!(valid_apple_matrix(0, x).is_err());
    }
}

use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use crate::{
    codec::rans256,
    error::{ColicError, Result},
    storage,
    target_registry::{
        APPLE8_MXFP4_GROUP_SIZE, APPLE8_MXFP4_MATH_FORMAT, APPLE8_MXFP4_SCALE_BLOCK_COLUMNS,
        APPLE8_MXFP4_SCALE_BLOCK_ROWS, APPLE8_MXFP4_SCALE_FORMAT, APPLE8_MXFP4_TILE_BYTES,
        APPLE8_MXFP4_TILE_COLUMNS, APPLE8_MXFP4_TILE_LAYOUT, APPLE8_MXFP4_TILE_ROWS,
        APPLE8_PROFILE_NAME, layout_registered, profile_allows_layout, profile_by_name,
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

fn bad(message: impl Into<String>) -> ColicError {
    ColicError::Usage(format!("invalid target-layout package: {}", message.into()))
}

fn u16a(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or_else(|| bad("truncated u16"))?
            .try_into()
            .unwrap(),
    ))
}
fn u32a(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| bad("truncated u32"))?
            .try_into()
            .unwrap(),
    ))
}
fn i32a(bytes: &[u8], offset: usize) -> Result<i32> {
    Ok(i32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or_else(|| bad("truncated i32"))?
            .try_into()
            .unwrap(),
    ))
}
fn u64a(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| bad("truncated u64"))?
            .try_into()
            .unwrap(),
    ))
}

fn string_at(manifest: &[u8], id: u32) -> Result<&str> {
    let count = u32a(manifest, 28)?;
    if id >= count {
        return Err(bad("string id out of range"));
    }
    let table = usize::try_from(u64a(manifest, 80)?).map_err(|_| bad("string table offset"))?;
    let descriptor = table
        .checked_add(id as usize * 16)
        .ok_or_else(|| bad("string descriptor overflow"))?;
    let relative =
        usize::try_from(u64a(manifest, descriptor)?).map_err(|_| bad("string offset"))?;
    let length = u32a(manifest, descriptor + 8)? as usize;
    let start = table
        .checked_add(relative)
        .ok_or_else(|| bad("string offset overflow"))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| bad("string length overflow"))?;
    std::str::from_utf8(
        manifest
            .get(start..end)
            .ok_or_else(|| bad("string outside manifest"))?,
    )
    .map_err(|_| bad("invalid UTF-8"))
}

fn tile_bytes(rows: u64, columns: u64) -> Result<u64> {
    if rows == 0 || columns == 0 {
        return Err(bad("zero matrix dimension"));
    }
    rows.div_ceil(APPLE8_MXFP4_TILE_ROWS)
        .checked_mul(columns.div_ceil(APPLE8_MXFP4_TILE_COLUMNS))
        .and_then(|tiles| tiles.checked_mul(APPLE8_MXFP4_TILE_BYTES))
        .ok_or_else(|| bad("matrix bytes overflow"))
}

fn matrix(prefix: &[u8], index: usize) -> Result<Matrix> {
    let descriptor = 64 + index * 128;
    if u16a(prefix, descriptor + 2)? != 0
        || u16a(prefix, descriptor + 14)? != 0
        || u32a(prefix, descriptor + 100)? != 0
        || u32a(prefix, descriptor + 108)? != 0
        || prefix
            .get(descriptor + 112..descriptor + 128)
            .is_none_or(|reserved| reserved.iter().any(|value| *value != 0))
    {
        return Err(bad(format!("matrix {index} reserved fields")));
    }
    Ok(Matrix {
        role: u16a(prefix, descriptor)?,
        math: u16a(prefix, descriptor + 4)?,
        scale: u16a(prefix, descriptor + 6)?,
        wc: u16a(prefix, descriptor + 8)?,
        sc: u16a(prefix, descriptor + 10)?,
        layout: u16a(prefix, descriptor + 12)?,
        rows: u64a(prefix, descriptor + 16)?,
        cols: u64a(prefix, descriptor + 24)?,
        sr: u32a(prefix, descriptor + 32)?,
        sk: u32a(prefix, descriptor + 36)?,
        wt: u32a(prefix, descriptor + 40)?,
        st: u32a(prefix, descriptor + 44)?,
        wo: u64a(prefix, descriptor + 48)?,
        ws: u64a(prefix, descriptor + 56)?,
        wd: u64a(prefix, descriptor + 64)?,
        so: u64a(prefix, descriptor + 72)?,
        ss: u64a(prefix, descriptor + 80)?,
        sd: u64a(prefix, descriptor + 88)?,
        crc: u32a(prefix, descriptor + 96)?,
        group: u32a(prefix, descriptor + 104)?,
    })
}

fn valid_apple_matrix(index: usize, matrix: Matrix) -> Result<u64> {
    let decoded = tile_bytes(matrix.rows, matrix.cols)?;
    let weight_codec_valid = match matrix.wc {
        CODEC_NONE => matrix.wt == 0 && matrix.ws == decoded,
        rans256::CODEC_ID => matrix.wt == rans256::TABLE_ID && matrix.ws != 0,
        _ => false,
    };
    if matrix.role != (index + 1) as u16
        || matrix.layout != APPLE8_MXFP4_TILE_LAYOUT
        || matrix.math != APPLE8_MXFP4_MATH_FORMAT
        || matrix.scale != APPLE8_MXFP4_SCALE_FORMAT
        || matrix.sr != APPLE8_MXFP4_SCALE_BLOCK_ROWS
        || matrix.sk != APPLE8_MXFP4_SCALE_BLOCK_COLUMNS
        || matrix.group != APPLE8_MXFP4_GROUP_SIZE
        || !weight_codec_valid
        || matrix.wo == 0
        || !matrix.wo.is_multiple_of(16)
        || matrix.wd != decoded
        || matrix.sc != CODEC_NONE
        || matrix.st != 0
        || matrix.so != 0
        || matrix.ss != 0
        || matrix.sd != 0
    {
        return Err(bad(format!(
            "matrix {index} violates Apple8 Design-A descriptor"
        )));
    }
    Ok(decoded)
}

fn read_range(file: &mut File, offset: u64, bytes: usize) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| ColicError::Io {
            path: PathBuf::from("<target shard>"),
            source,
        })?;
    let mut result = vec![0_u8; bytes];
    file.read_exact(&mut result)
        .map_err(|source| ColicError::Io {
            path: PathBuf::from("<target shard>"),
            source,
        })?;
    Ok(result)
}

fn decoded_matrix(
    manifest: &[u8],
    file: &mut File,
    record_offset: u64,
    matrix: Matrix,
) -> Result<Vec<u8>> {
    let source_offset = record_offset
        .checked_add(matrix.wo)
        .ok_or_else(|| bad("matrix file offset overflow"))?;
    let stored = usize::try_from(matrix.ws).map_err(|_| bad("matrix stored bytes exceed usize"))?;
    let decoded =
        usize::try_from(matrix.wd).map_err(|_| bad("matrix decoded bytes exceed usize"))?;
    if matrix.wc == CODEC_NONE {
        return read_range(file, source_offset, stored);
    }
    let table = rans256::table_from_manifest(manifest, matrix.wt, matrix.wc)?;
    let source = read_range(file, source_offset, stored)?;
    let slack_offset = source_offset
        .checked_add(matrix.ws)
        .ok_or_else(|| bad("rANS slack offset overflow"))?;
    let slack = read_range(file, slack_offset, rans256::READABLE_SLACK)?;
    if slack.iter().any(|byte| *byte != 0) {
        return Err(bad("rANS readable slack is nonzero"));
    }
    rans256::decode_bytes(&source, &table, decoded)
}

fn padding(decoded: &[u8], matrix: Matrix) -> Result<()> {
    let row_remainder = matrix.rows % APPLE8_MXFP4_TILE_ROWS;
    let column_remainder = matrix.cols % APPLE8_MXFP4_TILE_COLUMNS;
    let row_tiles = matrix.rows.div_ceil(APPLE8_MXFP4_TILE_ROWS);
    let groups = matrix.cols.div_ceil(APPLE8_MXFP4_TILE_COLUMNS);
    for output_tile in 0..row_tiles {
        for group in 0..groups {
            let row_edge = row_remainder != 0 && output_tile + 1 == row_tiles;
            let column_edge = column_remainder != 0 && group + 1 == groups;
            if !row_edge && !column_edge {
                continue;
            }
            let tile_index = output_tile
                .checked_mul(groups)
                .and_then(|value| value.checked_add(group))
                .ok_or_else(|| bad("tile index overflow"))?;
            let tile_offset = usize::try_from(
                tile_index
                    .checked_mul(APPLE8_MXFP4_TILE_BYTES)
                    .ok_or_else(|| bad("tile byte offset overflow"))?,
            )
            .map_err(|_| bad("tile byte offset exceeds usize"))?;
            let tile = decoded
                .get(tile_offset..tile_offset + APPLE8_MXFP4_TILE_BYTES as usize)
                .ok_or_else(|| bad("decoded matrix tile is truncated"))?;
            for row_index in 0..APPLE8_MXFP4_TILE_ROWS as usize {
                let row = &tile[row_index * 16..row_index * 16 + 16];
                let logical_row = !row_edge || row_index < row_remainder as usize;
                if !logical_row {
                    if row.iter().any(|value| *value != 0) || tile[128 + row_index] != 0 {
                        return Err(bad("nonzero output-row padding"));
                    }
                } else if column_edge {
                    let used = column_remainder.div_ceil(2) as usize;
                    if row[used..].iter().any(|value| *value != 0)
                        || (!column_remainder.is_multiple_of(2) && row[used - 1] & 0xf0 != 0)
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
    let manifest_path = package.join("manifest.coli");
    let manifest = fs::read(&manifest_path).map_err(|source| ColicError::Io {
        path: manifest_path,
        source,
    })?;
    let profile_name = string_at(&manifest, u32a(&manifest, 148)?)?;
    let profile = profile_by_name(profile_name)
        .ok_or_else(|| bad(format!("unknown target profile `{profile_name}`")))?;

    let shard_count = u32a(&manifest, 40)? as usize;
    let shard_table =
        usize::try_from(u64a(&manifest, 48)?).map_err(|_| bad("shard table offset"))?;
    let mut paths = Vec::with_capacity(shard_count);
    for index in 0..shard_count {
        let descriptor = shard_table
            .checked_add(index * 64)
            .ok_or_else(|| bad("shard table overflow"))?;
        paths.push(package.join(string_at(&manifest, u32a(&manifest, descriptor + 8)?)?));
    }
    let mut files = paths
        .iter()
        .map(|path| {
            File::open(path).map_err(|source| ColicError::Io {
                path: path.clone(),
                source,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let record_count = usize::try_from(u64a(&manifest, 32)?).map_err(|_| bad("record count"))?;
    let record_table =
        usize::try_from(u64a(&manifest, 64)?).map_err(|_| bad("record table offset"))?;
    for record_index in 0..record_count {
        let descriptor = record_table
            .checked_add(record_index * 96)
            .ok_or_else(|| bad("record table overflow"))?;
        if u16a(&manifest, descriptor + 8)? != REC_EXPERT {
            continue;
        }
        let shard_index = u32a(&manifest, descriptor + 20)? as usize;
        let record_offset = u64a(&manifest, descriptor + 40)?;
        let stored = u64a(&manifest, descriptor + 48)?;
        let decoded = u64a(&manifest, descriptor + 56)?;
        let layer = i32a(&manifest, descriptor + 28)?;
        let expert = i32a(&manifest, descriptor + 32)?;
        let file = files
            .get_mut(shard_index)
            .ok_or_else(|| bad("invalid shard"))?;
        let prefix = read_range(file, record_offset, PREFIX)?;
        if prefix.get(..8) != Some(b"COLIEXPT".as_slice())
            || u16a(&prefix, 8)? != 1
            || u32a(&prefix, 12)? != 64
            || i32a(&prefix, 16)? != layer
            || i32a(&prefix, 20)? != expert
            || u16a(&prefix, 24)? != 3
            || u32a(&prefix, 28)? != 128
            || u64a(&prefix, 32)? != 64
            || u64a(&prefix, 40)? != PREFIX as u64
        {
            return Err(bad(format!("expert {record_index} envelope")));
        }

        let matrices = [
            matrix(&prefix, 0)?,
            matrix(&prefix, 1)?,
            matrix(&prefix, 2)?,
        ];
        for (matrix_index, matrix) in matrices.iter().enumerate() {
            if !layout_registered(matrix.layout) {
                return Err(bad(format!(
                    "expert {layer}/{expert} matrix {matrix_index} uses unknown layout 0x{:04x}",
                    matrix.layout
                )));
            }
            if !profile_allows_layout(profile, matrix.layout) {
                return Err(bad(format!(
                    "expert {layer}/{expert} matrix {matrix_index} layout 0x{:04x} is outside profile `{profile_name}`",
                    matrix.layout
                )));
            }
        }
        if profile_name != APPLE8_PROFILE_NAME {
            continue;
        }
        if u16a(&manifest, descriptor + 10)? != CODEC_NONE {
            return Err(bad("Apple8 expert outer codec must be NONE"));
        }

        let mut total = 0_u64;
        let mut spans = Vec::with_capacity(3);
        for (matrix_index, matrix) in matrices.iter().copied().enumerate() {
            let expected = valid_apple_matrix(matrix_index, matrix)?;
            let physical_end = matrix
                .wo
                .checked_add(matrix.ws)
                .and_then(|end| {
                    if matrix.wc == rans256::CODEC_ID {
                        end.checked_add(rans256::READABLE_SLACK as u64)
                    } else {
                        Some(end)
                    }
                })
                .ok_or_else(|| bad("matrix span overflow"))?;
            if matrix.wo < PREFIX as u64 || physical_end > stored {
                return Err(bad("matrix outside record"));
            }
            spans.push((matrix.wo, physical_end));
            total = total
                .checked_add(expected)
                .ok_or_else(|| bad("resident bytes overflow"))?;
            let bytes = decoded_matrix(&manifest, file, record_offset, matrix)?;
            if storage::crc32c(&bytes) != matrix.crc {
                return Err(bad(format!(
                    "expert {layer}/{expert} matrix {matrix_index} logical CRC"
                )));
            }
            padding(&bytes, matrix)?;
        }
        spans.sort_unstable();
        if spans.windows(2).any(|window| window[1].0 < window[0].1) {
            return Err(bad("matrix spans overlap"));
        }
        if total != decoded || u64a(&prefix, 48)? != decoded {
            return Err(bad("resident byte total"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good(rows: u64, columns: u64) -> Matrix {
        let bytes = tile_bytes(rows, columns).unwrap();
        Matrix {
            role: 1,
            math: APPLE8_MXFP4_MATH_FORMAT,
            scale: APPLE8_MXFP4_SCALE_FORMAT,
            wc: 0,
            sc: 0,
            layout: APPLE8_MXFP4_TILE_LAYOUT,
            rows,
            cols: columns,
            sr: APPLE8_MXFP4_SCALE_BLOCK_ROWS,
            sk: APPLE8_MXFP4_SCALE_BLOCK_COLUMNS,
            wt: 0,
            st: 0,
            wo: 448,
            ws: bytes,
            wd: bytes,
            so: 0,
            ss: 0,
            sd: 0,
            crc: 1,
            group: APPLE8_MXFP4_GROUP_SIZE,
        }
    }

    #[test]
    fn sizes() {
        for (rows, columns, bytes) in [
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
            assert_eq!(tile_bytes(rows, columns).unwrap(), bytes);
        }
        assert!(tile_bytes(0, 32).is_err());
        assert!(tile_bytes(u64::MAX, u64::MAX).is_err());
    }

    #[test]
    fn descriptor_accepts_raw_and_rans_design_a() {
        let raw = good(8, 33);
        assert_eq!(valid_apple_matrix(0, raw).unwrap(), 272);
        let mut compressed = raw;
        compressed.wc = rans256::CODEC_ID;
        compressed.wt = rans256::TABLE_ID;
        compressed.ws = 160;
        assert_eq!(valid_apple_matrix(0, compressed).unwrap(), 272);
        compressed.sc = rans256::CODEC_ID;
        assert!(valid_apple_matrix(0, compressed).is_err());
    }
}

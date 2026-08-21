const APPLE8_EXPERT_HEADER_BYTES: usize = 64;
const APPLE8_EXPERT_DESC_BYTES: usize = 128;
const APPLE8_EXPERT_MATRIX_COUNT: usize = 3;
const APPLE8_EXPERT_DATA_OFFSET: usize =
    APPLE8_EXPERT_HEADER_BYTES + APPLE8_EXPERT_DESC_BYTES * APPLE8_EXPERT_MATRIX_COUNT;

pub fn apple8_tile_bytes(rows: u32, columns: u32) -> Result<u64> {
    if rows == 0 || columns == 0 {
        return Err(ColicError::Usage(
            "Apple8 MXFP4 matrices must have non-zero dimensions".into(),
        ));
    }
    let row_tiles = u64::from(rows).div_ceil(target_registry::APPLE8_MXFP4_TILE_ROWS);
    let k_tiles = u64::from(columns).div_ceil(target_registry::APPLE8_MXFP4_TILE_COLUMNS);
    row_tiles
        .checked_mul(k_tiles)
        .and_then(|tiles| tiles.checked_mul(target_registry::APPLE8_MXFP4_TILE_BYTES))
        .ok_or_else(|| ColicError::Usage("Apple8 MXFP4 matrix size overflows u64".into()))
}

pub fn apple8_expert_stored_bytes(expert: &RoutedExpert) -> Result<u64> {
    let mut cursor = APPLE8_EXPERT_DATA_OFFSET as u64;
    for matrix in [&expert.gate, &expert.up, &expert.down] {
        cursor = align_up(cursor, target_registry::APPLE8_MXFP4_MATRIX_ALIGNMENT)?;
        cursor = cursor
            .checked_add(apple8_tile_bytes(matrix.rows, matrix.columns)?)
            .ok_or_else(|| ColicError::Usage("Apple8 expert stored size overflows u64".into()))?;
    }
    Ok(cursor)
}

pub fn apple8_expert_decoded_bytes(expert: &RoutedExpert) -> Result<u64> {
    [&expert.gate, &expert.up, &expert.down]
        .into_iter()
        .try_fold(0_u64, |total, matrix| {
            total
                .checked_add(apple8_tile_bytes(matrix.rows, matrix.columns)?)
                .ok_or_else(|| ColicError::Usage("Apple8 expert resident size overflows u64".into()))
        })
}

pub fn validate_apple8_exact_mxfp4_expert(expert: &RoutedExpert) -> Result<()> {
    for matrix in [&expert.gate, &expert.up, &expert.down] {
        validate_canonical_mxfp4_matrix(matrix)?;
    }
    Ok(())
}

pub fn validate_apple8_quantized_mxfp4_expert(expert: &RoutedExpert) -> Result<()> {
    for matrix in [&expert.gate, &expert.up, &expert.down] {
        if matrix.source.dtype != "BF16" || matrix.scale.is_some() {
            return Err(ColicError::unsupported(
                "Apple8 target lowering",
                format!(
                    "compiler MXFP4 lowering requires unscaled BF16 source matrices; got `{}`",
                    matrix.source.dtype
                ),
            ));
        }
        let expected = u64::from(matrix.rows)
            .checked_mul(u64::from(matrix.columns))
            .and_then(|values| values.checked_mul(2))
            .ok_or_else(|| ColicError::Usage("Apple8 BF16 source size overflows u64".into()))?;
        if matrix.source.len != expected
            || matrix.source.shape != [u64::from(matrix.rows), u64::from(matrix.columns)]
        {
            return Err(ColicError::InvalidSource {
                path: matrix.source.source.clone(),
                detail: format!(
                    "BF16 expert matrix has len/shape {}/{:?}; expected {expected}/[{}, {}]",
                    matrix.source.len, matrix.source.shape, matrix.rows, matrix.columns
                ),
            });
        }
    }
    Ok(())
}

pub fn lower_apple8_exact_mxfp4_expert(expert: &RoutedExpert) -> Result<Vec<u8>> {
    validate_apple8_exact_mxfp4_expert(expert)?;
    let gate = canonical_mxfp4_matrix(&expert.gate)?;
    let up = canonical_mxfp4_matrix(&expert.up)?;
    let down = canonical_mxfp4_matrix(&expert.down)?;
    lower_apple8_packed_expert(expert.layer, expert.expert, [&gate, &up, &down])
}

pub fn lower_apple8_quantized_mxfp4_expert(expert: &RoutedExpert) -> Result<Vec<u8>> {
    validate_apple8_quantized_mxfp4_expert(expert)?;
    let gate = crate::quant::mxfp4::quantize_matrix(&expert.gate)?;
    let up = crate::quant::mxfp4::quantize_matrix(&expert.up)?;
    let down = crate::quant::mxfp4::quantize_matrix(&expert.down)?;
    lower_apple8_packed_expert(expert.layer, expert.expert, [&gate, &up, &down])
}

fn validate_canonical_mxfp4_matrix(matrix: &crate::ir::Matrix) -> Result<()> {
    let Some(scale) = matrix.scale.as_ref() else {
        return Err(ColicError::unsupported(
            "Apple8 target lowering",
            "exact Apple8 lowering requires canonical MXFP4 source scales",
        ));
    };
    let row_bytes = u64::from(matrix.columns).div_ceil(2);
    let groups = u64::from(matrix.columns).div_ceil(32);
    let weight_bytes = u64::from(matrix.rows)
        .checked_mul(row_bytes)
        .ok_or_else(|| ColicError::Usage("canonical MXFP4 weight size overflows u64".into()))?;
    let scale_bytes = u64::from(matrix.rows)
        .checked_mul(groups)
        .ok_or_else(|| ColicError::Usage("canonical MXFP4 scale size overflows u64".into()))?;
    if matrix.source.dtype != "I8"
        || !matches!(scale.dtype.as_str(), "F8_E8M0" | "F8_E8M0FNU")
        || matrix.source.len != weight_bytes
        || scale.len != scale_bytes
        || matrix.source.shape != [u64::from(matrix.rows), row_bytes]
        || scale.shape != [u64::from(matrix.rows), groups]
    {
        return Err(ColicError::unsupported(
            "Apple8 target lowering",
            format!(
                "exact Apple8 lowering requires canonical MXFP4 I8/[{}, {}] + UE8M0/[{}, {}] source; got {}/{:?} + {}/{:?}",
                matrix.rows,
                row_bytes,
                matrix.rows,
                groups,
                matrix.source.dtype,
                matrix.source.shape,
                scale.dtype,
                scale.shape
            ),
        ));
    }
    Ok(())
}

fn canonical_mxfp4_matrix(matrix: &crate::ir::Matrix) -> Result<crate::quant::mxfp4::PackedMatrix> {
    validate_canonical_mxfp4_matrix(matrix)?;
    let scale = matrix.scale.as_ref().unwrap();
    Ok(crate::quant::mxfp4::PackedMatrix {
        rows: matrix.rows,
        columns: matrix.columns,
        weights: read_tensor(&matrix.source)?,
        scales: read_tensor(scale)?,
    })
}

fn lower_apple8_packed_expert(
    layer: u32,
    expert: u32,
    matrices: [&crate::quant::mxfp4::PackedMatrix; APPLE8_EXPERT_MATRIX_COUNT],
) -> Result<Vec<u8>> {
    let mut payload = vec![0_u8; APPLE8_EXPERT_DATA_OFFSET];
    payload[..8].copy_from_slice(b"COLIEXPT");
    put_u16(&mut payload, 8, 1);
    put_u32(&mut payload, 12, APPLE8_EXPERT_HEADER_BYTES as u32);
    put_i32(
        &mut payload,
        16,
        i32::try_from(layer)
            .map_err(|_| ColicError::Usage("Apple8 expert layer exceeds i32".into()))?,
    );
    put_i32(
        &mut payload,
        20,
        i32::try_from(expert)
            .map_err(|_| ColicError::Usage("Apple8 expert id exceeds i32".into()))?,
    );
    put_u16(&mut payload, 24, APPLE8_EXPERT_MATRIX_COUNT as u16);
    put_u32(&mut payload, 28, APPLE8_EXPERT_DESC_BYTES as u32);
    put_u64(&mut payload, 32, APPLE8_EXPERT_HEADER_BYTES as u64);
    put_u64(&mut payload, 40, APPLE8_EXPERT_DATA_OFFSET as u64);

    let mut decoded = 0_u64;
    for (index, matrix) in matrices.into_iter().enumerate() {
        let tiles = repack_apple8_matrix(matrix)?;
        let offset = append_aligned(&mut payload, &tiles)?;
        let bytes = tiles.len() as u64;
        decoded = decoded
            .checked_add(bytes)
            .ok_or_else(|| ColicError::Usage("Apple8 decoded size overflows u64".into()))?;

        let desc = APPLE8_EXPERT_HEADER_BYTES + index * APPLE8_EXPERT_DESC_BYTES;
        put_u16(&mut payload, desc, (index + 1) as u16);
        put_u16(
            &mut payload,
            desc + 4,
            target_registry::APPLE8_MXFP4_MATH_FORMAT,
        );
        put_u16(
            &mut payload,
            desc + 6,
            target_registry::APPLE8_MXFP4_SCALE_FORMAT,
        );
        put_u16(
            &mut payload,
            desc + 12,
            target_registry::APPLE8_MXFP4_TILE_LAYOUT,
        );
        put_u64(&mut payload, desc + 16, u64::from(matrix.rows));
        put_u64(&mut payload, desc + 24, u64::from(matrix.columns));
        put_u32(
            &mut payload,
            desc + 32,
            target_registry::APPLE8_MXFP4_SCALE_BLOCK_ROWS,
        );
        put_u32(
            &mut payload,
            desc + 36,
            target_registry::APPLE8_MXFP4_SCALE_BLOCK_COLUMNS,
        );
        put_u64(&mut payload, desc + 48, offset);
        put_u64(&mut payload, desc + 56, bytes);
        put_u64(&mut payload, desc + 64, bytes);
        put_u32(&mut payload, desc + 96, crc32c(&tiles));
        put_u32(
            &mut payload,
            desc + 104,
            target_registry::APPLE8_MXFP4_GROUP_SIZE,
        );
    }
    put_u64(&mut payload, 48, decoded);

    let expected = apple8_packed_stored_bytes(matrices)?;
    if payload.len() as u64 != expected {
        return Err(ColicError::Usage(
            "Apple8 lowerer output does not match raw storage plan".into(),
        ));
    }
    Ok(payload)
}

fn apple8_packed_stored_bytes(
    matrices: [&crate::quant::mxfp4::PackedMatrix; APPLE8_EXPERT_MATRIX_COUNT],
) -> Result<u64> {
    let mut cursor = APPLE8_EXPERT_DATA_OFFSET as u64;
    for matrix in matrices {
        cursor = align_up(cursor, target_registry::APPLE8_MXFP4_MATRIX_ALIGNMENT)?;
        cursor = cursor
            .checked_add(apple8_tile_bytes(matrix.rows, matrix.columns)?)
            .ok_or_else(|| ColicError::Usage("Apple8 packed expert size overflows u64".into()))?;
    }
    Ok(cursor)
}

fn repack_apple8_matrix(matrix: &crate::quant::mxfp4::PackedMatrix) -> Result<Vec<u8>> {
    let row_bytes = (matrix.columns as usize).div_ceil(2);
    let groups = (matrix.columns as usize).div_ceil(32);
    let expected_weights = (matrix.rows as usize)
        .checked_mul(row_bytes)
        .ok_or_else(|| ColicError::Usage("MXFP4 canonical weight size overflows usize".into()))?;
    let expected_scales = (matrix.rows as usize)
        .checked_mul(groups)
        .ok_or_else(|| ColicError::Usage("MXFP4 canonical scale size overflows usize".into()))?;
    if matrix.weights.len() != expected_weights || matrix.scales.len() != expected_scales {
        return Err(ColicError::Usage(
            "canonical MXFP4 buffers do not match matrix shape".into(),
        ));
    }

    let output_bytes = usize::try_from(apple8_tile_bytes(matrix.rows, matrix.columns)?)
        .map_err(|_| ColicError::Usage("Apple8 matrix exceeds usize".into()))?;
    let mut output = vec![0_u8; output_bytes];
    for row in 0..matrix.rows as usize {
        let weight_row = row * row_bytes;
        let scale_row = row * groups;
        let output_tile = row / target_registry::APPLE8_MXFP4_TILE_ROWS as usize;
        let output_row = row % target_registry::APPLE8_MXFP4_TILE_ROWS as usize;
        for group in 0..groups {
            let tile_index = output_tile
                .checked_mul(groups)
                .and_then(|index| index.checked_add(group))
                .ok_or_else(|| ColicError::Usage("Apple8 tile index overflows usize".into()))?;
            let tile = tile_index
                .checked_mul(target_registry::APPLE8_MXFP4_TILE_BYTES as usize)
                .ok_or_else(|| ColicError::Usage("Apple8 tile offset overflows usize".into()))?;
            let column = group * target_registry::APPLE8_MXFP4_TILE_COLUMNS as usize;
            let remaining = matrix.columns as usize - column;
            let logical_columns = remaining.min(target_registry::APPLE8_MXFP4_TILE_COLUMNS as usize);
            let copy_bytes = logical_columns.div_ceil(2);
            let source = weight_row + column / 2;
            let destination = tile + output_row * target_registry::APPLE8_MXFP4_WEIGHT_ROW_BYTES as usize;
            output[destination..destination + copy_bytes]
                .copy_from_slice(&matrix.weights[source..source + copy_bytes]);
            output[tile + target_registry::APPLE8_MXFP4_WEIGHT_BYTES as usize + output_row] =
                matrix.scales[scale_row + group];
        }
    }
    Ok(output)
}

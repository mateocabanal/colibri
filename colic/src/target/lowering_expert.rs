pub fn lower_exact_expert(expert: &RoutedExpert) -> Result<Vec<u8>> {
    const HEADER_BYTES: usize = 64;
    const DESC_BYTES: usize = 128;
    const TABLE_BYTES: usize = DESC_BYTES * 3;
    const DATA_OFFSET: usize = HEADER_BYTES + TABLE_BYTES;
    let matrices = [
        (&expert.gate, 1_u16),
        (&expert.up, 2_u16),
        (&expert.down, 3_u16),
    ];
    let mut payload = vec![0_u8; DATA_OFFSET];
    payload[0..8].copy_from_slice(b"COLIEXPT");
    put_u16(&mut payload, 8, 1);
    put_u16(&mut payload, 10, 0);
    put_u32(&mut payload, 12, HEADER_BYTES as u32);
    put_i32(&mut payload, 16, expert.layer as i32);
    put_i32(&mut payload, 20, expert.expert as i32);
    put_u16(&mut payload, 24, 3);
    put_u32(&mut payload, 28, DESC_BYTES as u32);
    put_u64(&mut payload, 32, HEADER_BYTES as u64);
    put_u64(&mut payload, 40, DATA_OFFSET as u64);
    let mut logical = Vec::new();
    for (index, (matrix, role)) in matrices.into_iter().enumerate() {
        let weight = read_tensor(&matrix.source)?;
        let (scale, scale_format) = match &matrix.scale {
            Some(scale) => (read_tensor(scale)?, scale_format(&scale.dtype)?),
            None => (Vec::new(), 0),
        };
        let weight_offset = append_aligned(&mut payload, &weight)?;
        let scale_offset = if scale.is_empty() {
            0
        } else {
            append_aligned(&mut payload, &scale)?
        };
        let desc = HEADER_BYTES + index * DESC_BYTES;
        put_u16(&mut payload, desc, role);
        put_u16(&mut payload, desc + 4, expert_math_format(&matrix.source.dtype)?);
        put_u16(&mut payload, desc + 6, scale_format);
        put_u64(&mut payload, desc + 16, matrix.rows as u64);
        put_u64(&mut payload, desc + 24, matrix.columns as u64);
        if matrix.source.dtype == "I8" {
            put_u32(&mut payload, desc + 32, 1);
            put_u32(&mut payload, desc + 36, 32);
        }
        put_u64(&mut payload, desc + 48, weight_offset);
        put_u64(&mut payload, desc + 56, weight.len() as u64);
        put_u64(&mut payload, desc + 64, weight.len() as u64);
        put_u64(&mut payload, desc + 72, scale_offset);
        put_u64(&mut payload, desc + 80, scale.len() as u64);
        put_u64(&mut payload, desc + 88, scale.len() as u64);
        let mut matrix_logical = weight.clone();
        matrix_logical.extend_from_slice(&scale);
        put_u32(&mut payload, desc + 96, crc32c(&matrix_logical));
        logical.extend_from_slice(&matrix_logical);
    }
    put_u64(&mut payload, 48, logical.len() as u64);
    Ok(payload)
}

pub fn exact_expert_stored_bytes(expert: &RoutedExpert) -> Result<u64> {
    let mut bytes = 64_u64 + 128 * 3;
    for matrix in [&expert.gate, &expert.up, &expert.down] {
        bytes = align_up(bytes, 16)?
            .checked_add(matrix.source.len)
            .ok_or_else(|| ColicError::Usage("projected expert payload size overflows u64".into()))?;
        if let Some(scale) = &matrix.scale {
            bytes = align_up(bytes, 16)?
                .checked_add(scale.len)
                .ok_or_else(|| ColicError::Usage("projected expert payload size overflows u64".into()))?;
        }
    }
    Ok(bytes)
}

pub fn exact_expert_decoded_bytes(expert: &RoutedExpert) -> Result<u64> {
    [&expert.gate, &expert.up, &expert.down]
        .into_iter()
        .try_fold(0_u64, |total, matrix| {
            let scale_bytes = matrix.scale.as_ref().map_or(0, |scale| scale.len);
            total
                .checked_add(matrix.source.len)
                .and_then(|bytes| bytes.checked_add(scale_bytes))
                .ok_or_else(|| ColicError::Usage("expert logical byte count overflows u64".into()))
        })
}

pub fn stream_exact_expert<W: Write + Seek>(expert: &RoutedExpert, output: &mut W) -> Result<u32> {
    const HEADER_BYTES: u64 = 64;
    const DESC_BYTES: u64 = 128;
    const DATA_OFFSET: u64 = HEADER_BYTES + DESC_BYTES * 3;
    let record_start = output.stream_position().map_err(|source| ColicError::Io {
        path: expert.gate.source.source.clone(), source,
    })?;
    let stored_bytes = exact_expert_stored_bytes(expert)?;
    let record_end = record_start.checked_add(stored_bytes)
        .ok_or_else(|| ColicError::Usage("expert output offset overflows u64".into()))?;
    let mut header = [0_u8; DATA_OFFSET as usize];
    header[..8].copy_from_slice(b"COLIEXPT");
    put_u16(&mut header, 8, 1);
    put_u32(&mut header, 12, HEADER_BYTES as u32);
    put_i32(&mut header, 16, expert.layer as i32);
    put_i32(&mut header, 20, expert.expert as i32);
    put_u16(&mut header, 24, 3);
    put_u32(&mut header, 28, DESC_BYTES as u32);
    put_u64(&mut header, 32, HEADER_BYTES);
    put_u64(&mut header, 40, DATA_OFFSET);
    put_u64(&mut header, 48, exact_expert_decoded_bytes(expert)?);
    output.write_all(&header).map_err(|source| ColicError::Io {
        path: expert.gate.source.source.clone(), source,
    })?;
    let matrices = [
        (&expert.gate, 1_u16),
        (&expert.up, 2_u16),
        (&expert.down, 3_u16),
    ];
    let mut cursor = DATA_OFFSET;
    let mut data_state = !0_u32;
    for (index, (matrix, role)) in matrices.into_iter().enumerate() {
        let desc = HEADER_BYTES as usize + index * DESC_BYTES as usize;
        let weight_offset = align_up(cursor, 16)?;
        write_padding(output, weight_offset - cursor, &mut data_state, &matrix.source.source)?;
        cursor = weight_offset;
        let mut logical_state = !0_u32;
        copy_tensor_stream(&matrix.source, output, &mut data_state, Some(&mut logical_state))?;
        cursor = cursor.checked_add(matrix.source.len)
            .ok_or_else(|| ColicError::Usage("expert source size overflows u64".into()))?;
        let (scale_offset, scale_len, scale_id) = if let Some(scale) = &matrix.scale {
            let offset = align_up(cursor, 16)?;
            write_padding(output, offset - cursor, &mut data_state, &scale.source)?;
            cursor = offset;
            copy_tensor_stream(scale, output, &mut data_state, Some(&mut logical_state))?;
            cursor = cursor.checked_add(scale.len)
                .ok_or_else(|| ColicError::Usage("expert scale size overflows u64".into()))?;
            (offset, scale.len, scale_format(&scale.dtype)?)
        } else {
            (0, 0, 0)
        };
        put_u16(&mut header, desc, role);
        put_u16(&mut header, desc + 4, expert_math_format(&matrix.source.dtype)?);
        put_u16(&mut header, desc + 6, scale_id);
        put_u64(&mut header, desc + 16, matrix.rows as u64);
        put_u64(&mut header, desc + 24, matrix.columns as u64);
        if matrix.source.dtype == "I8" {
            put_u32(&mut header, desc + 32, 1);
            put_u32(&mut header, desc + 36, 32);
        }
        put_u64(&mut header, desc + 48, weight_offset);
        put_u64(&mut header, desc + 56, matrix.source.len);
        put_u64(&mut header, desc + 64, matrix.source.len);
        put_u64(&mut header, desc + 72, scale_offset);
        put_u64(&mut header, desc + 80, scale_len);
        put_u64(&mut header, desc + 88, scale_len);
        put_u32(&mut header, desc + 96, !logical_state);
    }
    if cursor != stored_bytes {
        return Err(ColicError::Usage("expert stream does not match its planned stored size".into()));
    }
    output.seek(SeekFrom::Start(record_start)).map_err(|source| ColicError::Io {
        path: expert.gate.source.source.clone(), source,
    })?;
    output.write_all(&header).map_err(|source| ColicError::Io {
        path: expert.gate.source.source.clone(), source,
    })?;
    output.seek(SeekFrom::Start(record_end)).map_err(|source| ColicError::Io {
        path: expert.gate.source.source.clone(), source,
    })?;
    Ok(crc32c_combine(crc32c(&header), !data_state, stored_bytes - DATA_OFFSET))
}

pub fn lower_exact_tensor(tensor: &source::TensorRef) -> Result<Vec<u8>> {
    if tensor.shape.len() > 8 {
        return Err(ColicError::unsupported(
            "exact tensor lowering",
            format!("rank {} exceeds the COLI v1 limit", tensor.shape.len()),
        ));
    }
    let _ = math_format_for_dtype(&tensor.dtype)?;
    let data = read_tensor(tensor)?;
    let mut payload = vec![0_u8; TENSOR_HEADER_BYTES];
    payload[0..8].copy_from_slice(b"COLITENS");
    put_u16(&mut payload, 8, 1);
    put_u32(&mut payload, 12, TENSOR_HEADER_BYTES as u32);
    put_u16(&mut payload, 16, tensor.shape.len() as u16);
    for (index, dimension) in tensor.shape.iter().enumerate() {
        put_u64(&mut payload, 32 + index * 8, *dimension);
    }
    put_u64(&mut payload, 96, TENSOR_HEADER_BYTES as u64);
    put_u64(&mut payload, 104, data.len() as u64);
    put_u64(&mut payload, 112, data.len() as u64);
    put_u32(&mut payload, 120, crc32c(&data));
    payload.extend_from_slice(&data);
    Ok(payload)
}

pub fn exact_tensor_stored_bytes(tensor: &source::TensorRef) -> Result<u64> {
    if tensor.shape.len() > 8 {
        return Err(ColicError::unsupported(
            "exact tensor lowering",
            format!("rank {} exceeds the COLI v1 limit", tensor.shape.len()),
        ));
    }
    let _ = math_format_for_dtype(&tensor.dtype)?;
    (TENSOR_HEADER_BYTES as u64)
        .checked_add(tensor.len)
        .ok_or_else(|| ColicError::Usage("projected tensor payload size overflows u64".into()))
}

pub fn stream_exact_tensor<W: Write + Seek>(
    tensor: &source::TensorRef,
    output: &mut W,
) -> Result<(u32, u32)> {
    exact_tensor_stored_bytes(tensor)?;
    let record_start = output.stream_position().map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    let mut header = [0_u8; TENSOR_HEADER_BYTES];
    header[0..8].copy_from_slice(b"COLITENS");
    put_u16(&mut header, 8, 1);
    put_u32(&mut header, 12, TENSOR_HEADER_BYTES as u32);
    put_u16(&mut header, 16, tensor.shape.len() as u16);
    for (index, dimension) in tensor.shape.iter().enumerate() {
        put_u64(&mut header, 32 + index * 8, *dimension);
    }
    put_u64(&mut header, 96, TENSOR_HEADER_BYTES as u64);
    put_u64(&mut header, 104, tensor.len);
    put_u64(&mut header, 112, tensor.len);
    output.write_all(&header).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    let mut logical_state = !0_u32;
    let mut payload_state = !0_u32;
    copy_tensor_stream(tensor, output, &mut payload_state, Some(&mut logical_state))?;
    let logical_crc32c = !logical_state;
    put_u32(&mut header, 120, logical_crc32c);
    output.seek(SeekFrom::Start(record_start)).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    output.write_all(&header).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    output.seek(SeekFrom::Start(
        record_start
            .checked_add(TENSOR_HEADER_BYTES as u64)
            .and_then(|offset| offset.checked_add(tensor.len))
            .ok_or_else(|| ColicError::Usage("tensor output offset overflows u64".into()))?,
    )).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    Ok((logical_crc32c, crc32c_combine(crc32c(&header), !payload_state, tensor.len)))
}

fn crc32c_state(mut crc: u32, bytes: &[u8]) -> u32 {
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0x82f6_3b78 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    crc
}

fn write_padding<W: Write>(
    output: &mut W,
    mut bytes: u64,
    state: &mut u32,
    path: &std::path::Path,
) -> Result<()> {
    const ZEROES: [u8; 16] = [0; 16];
    while bytes != 0 {
        let count = bytes.min(ZEROES.len() as u64) as usize;
        output.write_all(&ZEROES[..count]).map_err(|source| ColicError::Io {
            path: path.to_owned(), source,
        })?;
        *state = crc32c_state(*state, &ZEROES[..count]);
        bytes -= count as u64;
    }
    Ok(())
}

fn copy_tensor_stream<W: Write>(
    tensor: &source::TensorRef,
    output: &mut W,
    output_state: &mut u32,
    logical_state: Option<&mut u32>,
) -> Result<()> {
    let mut input = File::open(&tensor.source).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    input.seek(SeekFrom::Start(tensor.offset)).map_err(|source| ColicError::Io {
        path: tensor.source.clone(), source,
    })?;
    let mut remaining = tensor.len;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    let mut logical_state = logical_state;
    while remaining != 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        input.read_exact(&mut buffer[..count]).map_err(|source| ColicError::Io {
            path: tensor.source.clone(), source,
        })?;
        output.write_all(&buffer[..count]).map_err(|source| ColicError::Io {
            path: tensor.source.clone(), source,
        })?;
        *output_state = crc32c_state(*output_state, &buffer[..count]);
        if let Some(state) = logical_state.as_deref_mut() {
            *state = crc32c_state(*state, &buffer[..count]);
        }
        remaining -= count as u64;
    }
    Ok(())
}

pub fn crc32c_combine(mut left: u32, right: u32, mut right_len: u64) -> u32 {
    if right_len == 0 {
        return left;
    }
    const POLY: u32 = 0x82f6_3b78;
    let mut odd = [0_u32; 32];
    odd[0] = POLY;
    let mut row = 1_u32;
    for item in odd.iter_mut().skip(1) {
        *item = row;
        row <<= 1;
    }
    let mut even = gf2_matrix_square(&odd);
    odd = gf2_matrix_square(&even);
    loop {
        even = gf2_matrix_square(&odd);
        if right_len & 1 != 0 {
            left = gf2_matrix_times(&even, left);
        }
        right_len >>= 1;
        if right_len == 0 { break; }
        odd = gf2_matrix_square(&even);
        if right_len & 1 != 0 {
            left = gf2_matrix_times(&odd, left);
        }
        right_len >>= 1;
        if right_len == 0 { break; }
    }
    left ^ right
}

fn gf2_matrix_times(matrix: &[u32; 32], mut vector: u32) -> u32 {
    let mut sum = 0_u32;
    let mut index = 0;
    while vector != 0 {
        if vector & 1 != 0 { sum ^= matrix[index]; }
        vector >>= 1;
        index += 1;
    }
    sum
}

fn gf2_matrix_square(matrix: &[u32; 32]) -> [u32; 32] {
    std::array::from_fn(|index| gf2_matrix_times(matrix, matrix[index]))
}

fn append_aligned(output: &mut Vec<u8>, bytes: &[u8]) -> Result<u64> {
    let offset = align_up(output.len() as u64, 16)?;
    output.resize(offset as usize, 0);
    output.extend_from_slice(bytes);
    Ok(offset)
}

fn read_tensor(tensor: &source::TensorRef) -> Result<Vec<u8>> {
    let mut bytes = vec![0; tensor.len.try_into().map_err(|_| ColicError::Usage(
        "tensor is too large for the current record-lowering address space".into()
    ))?];
    source::read_range(tensor, 0..tensor.len, &mut bytes)?;
    Ok(bytes)
}

pub fn math_format_for_dtype(dtype: &str) -> Result<u16> {
    match dtype {
        "F32" => Ok(1),
        "F16" => Ok(2),
        "BF16" => Ok(3),
        "U8" => Ok(5),
        "F8_E8M0" | "F8_E8M0FNU" => Ok(5),
        "I64" => Ok(0x0a),
        "I8" => Ok(0x20),
        "F8_E4M3" | "F8_E4M3FN" => Ok(0x10),
        "F8_E5M2" => Ok(0x11),
        _ => Err(ColicError::unsupported(
            "exact expert lowering",
            format!("unsupported matrix dtype `{dtype}`"),
        )),
    }
}

fn expert_math_format(dtype: &str) -> Result<u16> {
    match dtype {
        "I8" => Ok(0x20),
        other => math_format_for_dtype(other),
    }
}

fn scale_format(dtype: &str) -> Result<u16> {
    match dtype {
        "F32" => Ok(1),
        "F16" => Ok(2),
        "BF16" => Ok(3),
        "U8" | "F8_E8M0" | "F8_E8M0FNU" => Ok(4),
        _ => Err(ColicError::unsupported(
            "exact expert lowering",
            format!("unsupported scale dtype `{dtype}`"),
        )),
    }
}

fn put_u16(buffer: &mut [u8], offset: usize, value: u16) {
    buffer[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(buffer: &mut [u8], offset: usize, value: u64) {
    buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn put_i32(buffer: &mut [u8], offset: usize, value: i32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

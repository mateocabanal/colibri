//! FFI to the proven C Metal backend (backend_metal.mm / backend_metal.h).
//! Only the ops we measured as wins: quantized expert GEMV (fmt 7 = MXFP4,
//! byte-compatible with Apple8 tiles) + the small dense ops.
//!
//! Unsafe at the boundary only; every call checks the return code and
//! falls back to CPU on failure (Metal unavailable, invalid fmt, ...).

use std::ffi::c_void;

#[repr(C)]
pub struct ColiMetalTensor {
    _private: [u8; 0],
}

extern "C" {
    pub fn coli_metal_init() -> i32;
    pub fn coli_metal_available() -> i32;
    pub fn coli_metal_matmul(
        tensor: *mut *mut ColiMetalTensor,
        y: *mut f32,
        x: *const f32,
        weights: *const c_void,
        scales: *const f32,
        fmt: i32,
        s: i32,
        i: i32,
        o: i32,
        gs: i32,
    ) -> i32;
    pub fn coli_metal_rmsnorm(x: *mut f32, w: *const f32, n: i32, nrows: i32, eps: f32) -> i32;
    pub fn coli_metal_add(y: *mut f32, a: *const f32, n: i32) -> i32;
    pub fn coli_metal_silu_mul(g: *mut f32, u: *const f32, n: i32) -> i32;
    pub fn coli_metal_tensor_free(tensor: *mut ColiMetalTensor);
    pub fn coli_metal_shutdown();
}

/// Lazily-initialized Metal availability. Returns true once init() succeeded.
static INIT: std::sync::Once = std::sync::Once::new();
static mut AVAILABLE: bool = false;

pub fn metal_init() -> bool {
    INIT.call_once(|| {
        let ok = unsafe { coli_metal_init() } == 1;
        unsafe { AVAILABLE = ok };
    });
    metal_available()
}

pub fn metal_available() -> bool {
    unsafe { AVAILABLE && coli_metal_available() == 1 }
}

/// y[O] = x[I] @ W^T for one token. `fmt` 7 = MXFP4 (Apple8 tiles),
/// weights = O*((I+1)/2) nibble bytes, scales = O*ceil(I/32) raw E8M0 bytes.
/// Returns true if Metal ran the matmul.
pub fn metal_matmul(
    tensor: &mut *mut ColiMetalTensor,
    y: &mut [f32],
    x: &[f32],
    weights: &[u8],
    scales: &[u8],
    fmt: i32,
    i: usize,
    o: usize,
) -> bool {
    if !metal_available() {
        return false;
    }
    if weights.len() < o * ((i + 1) / 2) || scales.len() < o * ((i + 31) / 32) {
        return false;
    }
    let rc = unsafe {
        coli_metal_matmul(
            tensor,
            y.as_mut_ptr(),
            x.as_ptr(),
            weights.as_ptr() as *const c_void,
            scales.as_ptr() as *const f32,
            fmt,
            1,
            i as i32,
            o as i32,
            0,
        )
    };
    rc == 1
}

/// In-place rmsnorm over nrows rows of n. Returns true on GPU success.
pub fn metal_rmsnorm(x: &mut [f32], w: &[f32], n: usize, nrows: usize, eps: f32) -> bool {
    if !metal_available() || x.len() < n * nrows || w.len() < n {
        return false;
    }
    unsafe { coli_metal_rmsnorm(x.as_mut_ptr(), w.as_ptr(), n as i32, nrows as i32, eps) == 1 }
}

/// y += a. Returns true on GPU success.
pub fn metal_add(y: &mut [f32], a: &[f32]) -> bool {
    if !metal_available() || y.len() != a.len() {
        return false;
    }
    unsafe { coli_metal_add(y.as_mut_ptr(), a.as_ptr(), y.len() as i32) == 1 }
}

/// g *= silu(u), in place. Returns true on GPU success.
pub fn metal_silu_mul(g: &mut [f32], u: &[f32]) -> bool {
    if !metal_available() || g.len() != u.len() {
        return false;
    }
    unsafe { coli_metal_silu_mul(g.as_mut_ptr(), u.as_ptr(), g.len() as i32) == 1 }
}

// ---------------------------------------------------------------------------
// MetalIO: async NVMe -> MTLBuffer expert streaming (from metalio.mm)
// Never mandatory: every fn returns an error/0 and the caller falls back to
// the pread path.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ColiMetalioRegion {
    pub file: i32,
    pub src_off: u64,
    pub bytes: usize,
    pub dst_off: u64,
}

extern "C" {
    pub fn metalio_init() -> i32;
    pub fn metalio_active() -> i32;
    pub fn metalio_shutdown();
    pub fn metalio_file_add(path: *const std::os::raw::c_char) -> i32;
    pub fn metalio_slot_alloc(max_bytes: usize) -> i32;
    pub fn metalio_slot_free(slot: i32);
    pub fn metalio_slot_ptr(slot: i32) -> *mut std::os::raw::c_void;
    pub fn metalio_slot_bytes(slot: i32) -> usize;
    pub fn metalio_loadv(
        slot: i32,
        regions: *const ColiMetalioRegion,
        count: i32,
        kind: i32,
    ) -> i64;
    pub fn metalio_wait(event_value: i64) -> i32;
    pub fn metalio_slot_consumed(slot: i32);
}

pub fn mio_init() -> bool {
    static INIT: std::sync::Once = std::sync::Once::new();
    static mut ACTIVE: bool = false;
    INIT.call_once(|| {
        let ok = unsafe { metalio_init() } == 1;
        unsafe { ACTIVE = ok };
    });
    mio_active()
}

pub fn mio_active() -> bool {
    unsafe { metalio_active() == 1 }
}

/// One region per matrix: stream all 3 matrices of an expert into the slot.
/// Returns (slot, event) on success, None if MetalIO is unavailable or any
/// step fails (caller falls back to pread + decode).
pub fn mio_load_expert(
    shard_path: &str,
    regions: &[(u64, usize)], // (file_offset, bytes) per matrix
) -> Option<(i32, i64)> {
    if !mio_init() || regions.is_empty() {
        return None;
    }
    let cpath = std::ffi::CString::new(shard_path).ok()?;
    let file = unsafe { metalio_file_add(cpath.as_ptr()) };
    if file < 0 {
        return None;
    }
    let total: usize = regions.iter().map(|r| r.1).sum();
    let slot = unsafe { metalio_slot_alloc(total) };
    if slot < 0 {
        return None;
    }
    let mut dst = 0usize;
    let mut cr: Vec<ColiMetalioRegion> = regions
        .iter()
        .map(|(off, len)| {
            let r = ColiMetalioRegion {
                file,
                src_off: *off,
                bytes: *len,
                dst_off: dst as u64,
            };
            dst += len;
            r
        })
        .collect();
    let ev = unsafe {
        metalio_loadv(
            slot,
            cr.as_mut_ptr(),
            cr.len() as i32,
            1, // MIO_LOAD_ASYNC
        )
    };
    if ev < 0 {
        unsafe { metalio_slot_free(slot) };
        return None;
    }
    Some((slot, ev))
}

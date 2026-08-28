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

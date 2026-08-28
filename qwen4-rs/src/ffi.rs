//! FFI to the proven C Metal backend (backend_metal.mm / backend_metal.h).
//! Only the ops we measured as wins: quantized expert GEMV (fmt 7 = MXFP4,
//! byte-compatible with Apple8 tiles) + the small dense ops.
//!
//! ALSO the direct Apple8 execution seam (apple8_metalio_direct.mm):
//! slot-resident expert GEMV/SwiGLU, the fused one-command-buffer moe_topk
//! (with begin/finish split phase so CPU work overlaps the routed-GPU wait),
//! and the coalesced Metal GDN kernels. Every entry point returns 0/None on
//! decline; the caller falls back to the CPU reference path (Metal is never
//! mandatory, matching the C engine's fallback contract).
//!
//! Unsafe at the boundary only; every call checks the return code and
//! falls back to CPU on failure (Metal unavailable, invalid fmt, ...).

#[cfg(target_os = "macos")]
mod imp {
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

    // -------------------------------------------------------------------------
    // MetalIO: async NVMe -> MTLBuffer expert streaming (from metalio.mm)
    // Never mandatory: every fn returns an error/0 and the caller falls back to
    // the pread path.
    // -------------------------------------------------------------------------

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

    /// The crate keeps ONE MTLIOFileHandle per shard file for the process
    /// lifetime (the C table hard-caps at METALIO_MAX_FILES=64; re-adding per
    /// miss would exhaust it and every load after the 64th would fall back to
    /// pread forever).
    static MIO_FILES: std::sync::Mutex<Option<std::collections::HashMap<String, i32>>> =
        std::sync::Mutex::new(None);

    /// Register a shard file once; returns its MetalIO file id (cached).
    /// None = MetalIO unavailable or the handle failed (caller falls back).
    pub fn mio_file(path: &str) -> Option<i32> {
        if !mio_init() {
            return None;
        }
        let mut guard = MIO_FILES.lock().unwrap();
        let map = guard.get_or_insert_with(std::collections::HashMap::new);
        if let Some(&fid) = map.get(path) {
            return Some(fid);
        }
        let cpath = std::ffi::CString::new(path).ok()?;
        let fid = unsafe { metalio_file_add(cpath.as_ptr()) };
        if fid < 0 {
            return None;
        }
        map.insert(path.to_string(), fid);
        Some(fid)
    }

    /// Stream (offset, bytes) regions of one expert into a fresh slot, packed
    /// contiguously (dst offsets 0..total) so a single `moe_topk`/`swiglu`
    /// submission can consume the expert. Returns (slot, event) on success.
    /// The caller owns the slot until it frees it (or drops it into a cache).
    pub fn mio_load_expert(
        fid: i32,
        regions: &[(u64, usize)], // (file_offset, bytes) per matrix
    ) -> Option<(i32, i64)> {
        if !mio_init() || regions.is_empty() {
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
                    file: fid,
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

    // -------------------------------------------------------------------------
    // Direct Apple8 execution seam (apple8_metalio_direct.mm) — kernels run on
    // tile bytes already resident in a MetalIO slot, native tile order, no
    // host decode, no repack.
    // -------------------------------------------------------------------------

    /// One expert's three matrices inside one MetalIO slot (contiguous packing
    /// as produced by `mio_load_expert`).
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ColiApple8MetalioExpert {
        pub slot: i32,
        pub gate_offset: usize,
        pub gate_bytes: usize,
        pub up_offset: usize,
        pub up_bytes: usize,
        pub down_offset: usize,
        pub down_bytes: usize,
    }

    extern "C" {
        pub fn coli_apple8_metalio_direct_init() -> i32;
        pub fn coli_apple8_metalio_direct_shutdown();
        pub fn coli_apple8_metalio_matmul_slot(
            slot: i32,
            slot_offset: usize,
            matrix_bytes: usize,
            x: *const f32,
            y: *mut f32,
            s: i32,
            i: i32,
            o: i32,
        ) -> i32;
        pub fn coli_apple8_metalio_swiglu_slot(
            slot: i32,
            gate_offset: usize,
            gate_bytes: usize,
            up_offset: usize,
            up_bytes: usize,
            down_offset: usize,
            down_bytes: usize,
            x: *const f32,
            y: *mut f32,
            s: i32,
            hidden: i32,
            intermediate: i32,
        ) -> i32;
        pub fn coli_apple8_metalio_moe_topk(
            experts: *const ColiApple8MetalioExpert,
            route_weights: *const f32,
            expert_count: i32,
            x: *const f32,
            y: *mut f32,
            hidden: i32,
            intermediate: i32,
        ) -> i32;
        pub fn coli_apple8_metalio_moe_topk_begin(
            experts: *const ColiApple8MetalioExpert,
            route_weights: *const f32,
            expert_count: i32,
            x: *const f32,
            hidden: i32,
            intermediate: i32,
            pending_out: *mut *mut c_void,
        ) -> i32;
        pub fn coli_apple8_metalio_moe_topk_finish(pending: *mut c_void, y: *mut f32) -> i32;
        pub fn coli_apple8_metalio_profile_get(
            encode_ns: *mut u64,
            submit_ns: *mut u64,
            wait_ns: *mut u64,
            kernel_ns: *mut u64,
            fused_calls: *mut u64,
            fused_experts: *mut u64,
        );
        // GDN (coalesced Metal kernels, qwen_moe.c seam contract)
        pub fn coli_apple8_metalio_gdn_token(
            layer: i32,
            x: *const f32,
            out: *mut f32,
            wqkv: *const u16,
            wz: *const u16,
            wa: *const u16,
            wb: *const u16,
            wout: *const u16,
            a_log: *const f32,
            dt_bias: *const f32,
            conv_w: *const f32,
            norm_w: *const f32,
            state: *mut f32,
            conv_state: *mut f32,
            d: i32,
            kheads: i32,
            kd: i32,
            vheads: i32,
            vd: i32,
            kk: i32,
            eps: f32,
        ) -> i32;
    }

    /// Bring up the direct path (Metal device + command queue + pipelines).
    /// Requires MetalIO active (the C contract: slots are the weight source).
    static DIRECT_INIT: std::sync::Once = std::sync::Once::new();
    static mut DIRECT_OK: bool = false;

    pub fn direct_init() -> bool {
        if !mio_init() {
            return false;
        }
        DIRECT_INIT.call_once(|| {
            let ok = unsafe { coli_apple8_metalio_direct_init() } == 1;
            unsafe { DIRECT_OK = ok };
        });
        direct_available()
    }

    pub fn direct_available() -> bool {
        unsafe { DIRECT_OK }
    }

    /// Fused decode-only routed layer: for K experts submits ONE command
    /// buffer (gate+up+swiglu -> down -> deterministic K-order weighted
    /// reduce) and waits once. `experts` in top-k order; `weights[i]` is the
    /// pre-renormalized route weight for experts[i] (C contract: consumed in
    /// caller order). K <= 64. Returns false -> caller runs the CPU path.
    pub fn moe_topk(
        experts: &[ColiApple8MetalioExpert],
        weights: &[f32],
        x: &[f32],
        y: &mut [f32],
        hidden: usize,
        intermediate: usize,
    ) -> bool {
        if !direct_available()
            || experts.is_empty()
            || experts.len() > 64
            || experts.len() != weights.len()
            || x.len() < hidden
            || y.len() < hidden
        {
            return false;
        }
        unsafe {
            coli_apple8_metalio_moe_topk(
                experts.as_ptr(),
                weights.as_ptr(),
                experts.len() as i32,
                x.as_ptr(),
                y.as_mut_ptr(),
                hidden as i32,
                intermediate as i32,
            ) == 1
        }
    }

    /// Split-phase begin: encode+commit the fused block, return a pending
    /// handle, do NOT wait — the CPU can run the shared expert while the GPU
    /// works (the C engine's QWEN_APPLE8_OVERLAP=1 default path).
    pub fn moe_topk_begin(
        experts: &[ColiApple8MetalioExpert],
        weights: &[f32],
        x: &[f32],
        hidden: usize,
        intermediate: usize,
    ) -> Option<*mut std::ffi::c_void> {
        if !direct_available()
            || experts.is_empty()
            || experts.len() > 64
            || experts.len() != weights.len()
            || x.len() < hidden
        {
            return None;
        }
        let mut pending: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe {
            coli_apple8_metalio_moe_topk_begin(
                experts.as_ptr(),
                weights.as_ptr(),
                experts.len() as i32,
                x.as_ptr(),
                hidden as i32,
                intermediate as i32,
                &mut pending,
            )
        };
        if rc == 1 && !pending.is_null() {
            Some(pending)
        } else {
            None
        }
    }

    /// Wait for the pending block, scatter-add into y, free the handle.
    /// false = GPU fault: the caller must redo those experts on CPU (C
    /// contract: end returning 0 means the caller redoes them).
    pub fn moe_topk_finish(pending: *mut std::ffi::c_void, y: &mut [f32], hidden: usize) -> bool {
        if pending.is_null() {
            return false;
        }
        unsafe { coli_apple8_metalio_moe_topk_finish(pending, y.as_mut_ptr()) == 1 }
    }

    /// Decode-only Metal GDN token, byte-exact C seam contract:
    /// - the five BF16 weight matrices, the recurrent state, and the conv
    ///   state MUST live in 16 KiB page-aligned host memory (the .mm wraps
    ///   them zero-copy via newBufferWithBytesNoCopy);
    /// - x/out are copied;
    /// - rc > 0: done; rc == 0: declined BEFORE commit (CPU fallback safe);
    ///   rc < 0: failed AFTER submit — recurrent state may have advanced, the
    ///   C engine treats that as fatal (do not fall through).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_token(
        layer: usize,
        x: &[f32],
        out: &mut [f32],
        wqkv: &[u8],
        wz: &[u8],
        wa: &[u8],
        wb: &[u8],
        wout: &[u8],
        a_log: &[f32],
        dt_bias: &[f32],
        conv_w: &[f32],
        norm_w: &[f32],
        state: &mut [f32],
        conv_state: &mut [f32],
        d: usize,
        kheads: usize,
        kd: usize,
        vheads: usize,
        vd: usize,
        kk: usize,
        eps: f32,
    ) -> i32 {
        if !direct_available() || !metal_available() {
            return 0;
        }
        unsafe {
            coli_apple8_metalio_gdn_token(
                layer as i32,
                x.as_ptr(),
                out.as_mut_ptr(),
                wqkv.as_ptr() as *const u16,
                wz.as_ptr() as *const u16,
                wa.as_ptr() as *const u16,
                wb.as_ptr() as *const u16,
                wout.as_ptr() as *const u16,
                a_log.as_ptr(),
                dt_bias.as_ptr(),
                conv_w.as_ptr(),
                norm_w.as_ptr(),
                state.as_mut_ptr(),
                conv_state.as_mut_ptr(),
                d as i32,
                kheads as i32,
                kd as i32,
                vheads as i32,
                vd as i32,
                kk as i32,
                eps,
            )
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    // Non-macOS (CI/Linux): Metal is unavailable by definition. Every entry
    // point declines so callers take the CPU path — same contract as the C
    // engine's non-Darwin build.
    #[repr(C)]
    pub struct ColiMetalTensor {
        _private: [u8; 0],
    }

    pub fn metal_init() -> bool {
        false
    }
    pub fn metal_available() -> bool {
        false
    }
    pub fn metal_matmul(
        _tensor: &mut *mut ColiMetalTensor,
        _y: &mut [f32],
        _x: &[f32],
        _weights: &[u8],
        _scales: &[u8],
        _fmt: i32,
        _i: usize,
        _o: usize,
    ) -> bool {
        false
    }
    pub fn metal_rmsnorm(_x: &mut [f32], _w: &[f32], _n: usize, _nrows: usize, _eps: f32) -> bool {
        false
    }
    pub fn metal_add(_y: &mut [f32], _a: &[f32]) -> bool {
        false
    }
    pub fn metal_silu_mul(_g: &mut [f32], _u: &[f32]) -> bool {
        false
    }

    pub struct ColiApple8MetalioExpert {
        pub slot: i32,
        pub gate_offset: usize,
        pub gate_bytes: usize,
        pub up_offset: usize,
        pub up_bytes: usize,
        pub down_offset: usize,
        pub down_bytes: usize,
    }

    pub fn mio_init() -> bool {
        false
    }
    pub fn mio_active() -> bool {
        false
    }
    pub fn mio_file(_path: &str) -> Option<i32> {
        None
    }
    pub fn mio_load_expert(_fid: i32, _regions: &[(u64, usize)]) -> Option<(i32, i64)> {
        None
    }
    pub fn metalio_wait(_ev: i64) -> i32 {
        -1
    }
    pub fn metalio_slot_free(_slot: i32) {}
    pub fn metalio_slot_ptr(_slot: i32) -> *mut std::os::raw::c_void {
        std::ptr::null_mut()
    }

    pub fn direct_init() -> bool {
        false
    }
    pub fn direct_available() -> bool {
        false
    }
    pub fn moe_topk(
        _experts: &[ColiApple8MetalioExpert],
        _weights: &[f32],
        _x: &[f32],
        _y: &mut [f32],
        _hidden: usize,
        _inter: usize,
    ) -> bool {
        false
    }
    pub fn moe_topk_begin(
        _experts: &[ColiApple8MetalioExpert],
        _weights: &[f32],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
    ) -> Option<*mut std::ffi::c_void> {
        None
    }
    pub fn moe_topk_finish(
        _pending: *mut std::ffi::c_void,
        _y: &mut [f32],
        _hidden: usize,
    ) -> bool {
        false
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_token(
        _layer: usize,
        _x: &[f32],
        _out: &mut [f32],
        _wqkv: &[u8],
        _wz: &[u8],
        _wa: &[u8],
        _wb: &[u8],
        _wout: &[u8],
        _a_log: &[f32],
        _dt_bias: &[f32],
        _conv_w: &[f32],
        _norm_w: &[f32],
        _state: &mut [f32],
        _conv_state: &mut [f32],
        _d: usize,
        _kheads: usize,
        _kd: usize,
        _vheads: usize,
        _vd: usize,
        _kk: usize,
        _eps: f32,
    ) -> i32 {
        0
    }
}

pub use imp::*;

//! #192 — Machine capability probe and profile.
//!
//! `MachineProfile` records hardware *capabilities* only. Choosing formats,
//! layouts, backends, or placement from these facts is the planner's job
//! (`plan::cost`), never the probe's. Profiles serialize to stable JSON so
//! plans are reproducible and tests can run on synthetic hardware.

use serde_json::{Value, json};

use crate::error::{ColicError, Result};

/// Stable schema version for serialized machine profiles.
pub const MACHINE_PROFILE_VERSION: u64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuProfile {
    pub vendor: String,
    pub brand: String,
    pub logical_cores: u32,
    pub avx2: bool,
    pub avx_vnni: bool,
    pub avx512_f: bool,
    pub avx512_bw: bool,
    pub avx512_vnni: bool,
    /// ARM SIMD capabilities (all false on x86_64).
    pub neon_i8mm: bool,
    pub neon_dotprod: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuProfile {
    Cuda {
        name: String,
        vram_bytes: u64,
        cc_major: u32,
        cc_minor: u32,
        dp4a: bool,
        tensor_cores: bool,
        native_fp8: bool,
        native_fp4: bool,
    },
    Metal {
        name: String,
        /// Metal GPU family (Apple8 = family 8).
        family: u32,
    },
}

/// Storage class of the device holding the package/checkpoint.
///
/// # ponytail: v1 cannot classify without root/WMI; defaults to Unknown and
/// planners must treat Unknown as "pageable streaming disk". Upgrade path:
/// rotational bit from /sys/block on Linux, MSFT_PhysicalDisk on Windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    Nvme,
    Ssd,
    Hdd,
    Network,
    Unknown,
}

impl StorageClass {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "nvme" => Ok(Self::Nvme),
            "ssd" => Ok(Self::Ssd),
            "hdd" => Ok(Self::Hdd),
            "network" => Ok(Self::Network),
            "unknown" => Ok(Self::Unknown),
            other => Err(ColicError::Usage(format!(
                "unknown storage class `{other}` (expected nvme|ssd|hdd|network|unknown)"
            ))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nvme => "nvme",
            Self::Ssd => "ssd",
            Self::Hdd => "hdd",
            Self::Network => "network",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineProfile {
    pub os: String,
    pub arch: String,
    pub cpu: CpuProfile,
    pub ram_bytes: u64,
    pub gpus: Vec<GpuProfile>,
    pub storage: StorageClass,
}

impl MachineProfile {
    /// Probe the machine colic is running on.
    pub fn probe() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu: probe_cpu(),
            ram_bytes: probe_ram().unwrap_or(0),
            gpus: probe_gpus(),
            storage: StorageClass::Unknown,
        }
    }

    /// Parse a machine profile from its stable JSON form (schema v1).
    pub fn from_json(value: &Value) -> Result<Self> {
        let version = value
            .get("machine_profile_version")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        if version != MACHINE_PROFILE_VERSION {
            return Err(ColicError::Usage(format!(
                "machine profile schema version {version} unsupported (expected {MACHINE_PROFILE_VERSION})"
            )));
        }
        let cpu_value = value
            .get("cpu")
            .ok_or_else(|| ColicError::Usage("machine profile missing `cpu`".into()))?;
        let cpu = CpuProfile {
            vendor: string_field(cpu_value, "vendor"),
            brand: string_field(cpu_value, "brand"),
            logical_cores: cpu_value
                .get("logical_cores")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32,
            avx2: bool_field(cpu_value, "avx2"),
            avx_vnni: bool_field(cpu_value, "avx_vnni"),
            avx512_f: bool_field(cpu_value, "avx512_f"),
            avx512_bw: bool_field(cpu_value, "avx512_bw"),
            avx512_vnni: bool_field(cpu_value, "avx512_vnni"),
            neon_i8mm: bool_field(cpu_value, "neon_i8mm"),
            neon_dotprod: bool_field(cpu_value, "neon_dotprod"),
        };
        let mut gpus = Vec::new();
        if let Some(list) = value.get("gpus").and_then(Value::as_array) {
            for gpu in list {
                let backend = string_field(gpu, "backend");
                match backend.as_str() {
                    "cuda" => {
                        let cc_major =
                            gpu.get("cc_major").and_then(Value::as_u64).unwrap_or(0) as u32;
                        let cc_minor =
                            gpu.get("cc_minor").and_then(Value::as_u64).unwrap_or(0) as u32;
                        gpus.push(GpuProfile::Cuda {
                            name: string_field(gpu, "name"),
                            vram_bytes: gpu.get("vram_bytes").and_then(Value::as_u64).unwrap_or(0),
                            cc_major,
                            cc_minor,
                            dp4a: bool_field(gpu, "dp4a"),
                            tensor_cores: bool_field(gpu, "tensor_cores"),
                            native_fp8: bool_field(gpu, "native_fp8"),
                            native_fp4: bool_field(gpu, "native_fp4"),
                        });
                    }
                    "metal" => gpus.push(GpuProfile::Metal {
                        name: string_field(gpu, "name"),
                        family: gpu.get("family").and_then(Value::as_u64).unwrap_or(0) as u32,
                    }),
                    other => {
                        return Err(ColicError::Usage(format!(
                            "machine profile gpu has unknown backend `{other}`"
                        )));
                    }
                }
            }
        }
        let storage = value
            .get("storage_class")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        Ok(Self {
            os: string_field(value, "os"),
            arch: string_field(value, "arch"),
            cpu,
            ram_bytes: value.get("ram_bytes").and_then(Value::as_u64).unwrap_or(0),
            gpus,
            storage: StorageClass::parse(storage)?,
        })
    }

    /// Stable JSON serialization (schema v1).
    pub fn to_json(&self) -> Value {
        json!({
            "machine_profile_version": MACHINE_PROFILE_VERSION,
            "os": self.os,
            "arch": self.arch,
            "cpu": {
                "vendor": self.cpu.vendor,
                "brand": self.cpu.brand,
                "logical_cores": self.cpu.logical_cores,
                "avx2": self.cpu.avx2,
                "avx_vnni": self.cpu.avx_vnni,
                "avx512_f": self.cpu.avx512_f,
                "avx512_bw": self.cpu.avx512_bw,
                "avx512_vnni": self.cpu.avx512_vnni,
                "neon_i8mm": self.cpu.neon_i8mm,
                "neon_dotprod": self.cpu.neon_dotprod,
            },
            "ram_bytes": self.ram_bytes,
            "gpus": self.gpus.iter().map(gpu_to_json).collect::<Vec<_>>(),
            "storage_class": self.storage.as_str(),
        })
    }

    pub fn has_avx2(&self) -> bool {
        self.arch == "x86_64" && self.cpu.avx2
    }

    pub fn cuda_gpus(&self) -> impl Iterator<Item = &GpuProfile> {
        self.gpus
            .iter()
            .filter(|gpu| matches!(gpu, GpuProfile::Cuda { .. }))
    }

    pub fn metal_gpu(&self) -> Option<&GpuProfile> {
        self.gpus
            .iter()
            .find(|gpu| matches!(gpu, GpuProfile::Metal { .. }))
    }

    /// Human-readable summary for `colic probe`.
    pub fn summary_text(&self) -> String {
        let mut text = String::new();
        text.push_str(&format!(
            "machine: {}-{} ram={}B storage={}\n",
            self.os,
            self.arch,
            self.ram_bytes,
            self.storage.as_str()
        ));
        text.push_str(&format!(
            "cpu: {} {} ({} threads) avx2={} vnni={} avx512={}/{}/{}\n",
            self.cpu.vendor,
            self.cpu.brand,
            self.cpu.logical_cores,
            self.cpu.avx2,
            self.cpu.avx_vnni,
            self.cpu.avx512_f,
            self.cpu.avx512_bw,
            self.cpu.avx512_vnni
        ));
        if self.gpus.is_empty() {
            text.push_str("gpus: none detected\n");
        }
        for gpu in &self.gpus {
            match gpu {
                GpuProfile::Cuda {
                    name,
                    vram_bytes,
                    cc_major,
                    cc_minor,
                    dp4a,
                    tensor_cores,
                    native_fp8,
                    native_fp4,
                } => text.push_str(&format!(
                    "gpu: cuda {name} vram={vram_bytes}B cc={cc_major}.{cc_minor} dp4a={dp4a} tensor={tensor_cores} fp8={native_fp8} fp4={native_fp4}\n"
                )),
                GpuProfile::Metal { name, family } => {
                    text.push_str(&format!("gpu: metal {name} family={family}\n"))
                }
            }
        }
        text
    }
}

fn gpu_to_json(gpu: &GpuProfile) -> Value {
    match gpu {
        GpuProfile::Cuda {
            name,
            vram_bytes,
            cc_major,
            cc_minor,
            dp4a,
            tensor_cores,
            native_fp8,
            native_fp4,
        } => json!({
            "backend": "cuda",
            "name": name,
            "vram_bytes": vram_bytes,
            "cc_major": cc_major,
            "cc_minor": cc_minor,
            "dp4a": dp4a,
            "tensor_cores": tensor_cores,
            "native_fp8": native_fp8,
            "native_fp4": native_fp4,
        }),
        GpuProfile::Metal { name, family } => {
            json!({ "backend": "metal", "name": name, "family": family })
        }
    }
}

fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn bool_field(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn probe_cpu() -> CpuProfile {
    #[cfg(target_arch = "x86_64")]
    let (avx2, avx_vnni, avx512_f, avx512_bw, avx512_vnni) = (
        std::arch::is_x86_feature_detected!("avx2"),
        std::arch::is_x86_feature_detected!("avxvnni"),
        std::arch::is_x86_feature_detected!("avx512f"),
        std::arch::is_x86_feature_detected!("avx512bw"),
        std::arch::is_x86_feature_detected!("avx512vnni"),
    );
    #[cfg(not(target_arch = "x86_64"))]
    let (avx2, avx_vnni, avx512_f, avx512_bw, avx512_vnni) = (false, false, false, false, false);
    #[cfg(target_arch = "aarch64")]
    let (neon_i8mm, neon_dotprod) = (
        std::arch::is_aarch64_feature_detected!("i8mm"),
        std::arch::is_aarch64_feature_detected!("dotprod"),
    );
    #[cfg(not(target_arch = "aarch64"))]
    let (neon_i8mm, neon_dotprod) = (false, false);
    let (vendor, brand) = probe_cpu_identity();
    CpuProfile {
        vendor,
        brand,
        logical_cores: std::thread::available_parallelism()
            .map(|cores| cores.get() as u32)
            .unwrap_or(1),
        avx2,
        avx_vnni,
        avx512_f,
        avx512_bw,
        avx512_vnni,
        neon_i8mm,
        neon_dotprod,
    }
}

/// CPU vendor + brand strings, best-effort per OS. Empty when unavailable.
fn probe_cpu_identity() -> (String, String) {
    match std::env::consts::OS {
        "linux" => {
            let info = read_command("cat", &["/proc/cpuinfo"]).unwrap_or_default();
            let vendor = info
                .lines()
                .find(|line| line.starts_with("vendor_id"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().to_string())
                .unwrap_or_default();
            let brand = info
                .lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().to_string())
                .unwrap_or_default();
            (vendor, brand)
        }
        "windows" => {
            let key = r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0";
            let vendor = query_registry(key, "VendorIdentifier");
            let brand = query_registry(key, "ProcessorNameString");
            (vendor, brand)
        }
        "macos" => (
            String::new(),
            read_command("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default(),
        ),
        _ => (String::new(), String::new()),
    }
}

/// Read one REG_SZ value through `reg query` (no new dependencies).
fn query_registry(key: &str, value_name: &str) -> String {
    let Some(output) = read_command("reg", &["query", key, "/v", value_name]) else {
        return String::new();
    };
    for line in output.lines() {
        if line.contains(value_name) {
            if let Some(pos) = line.find("REG_SZ") {
                return line[pos + "REG_SZ".len()..].trim().to_string();
            }
        }
    }
    String::new()
}

#[cfg(target_os = "windows")]
fn probe_ram() -> Option<u64> {
    #[repr(C)]
    #[allow(non_snake_case)]
    struct MemoryStatusEx {
        dwLength: u32,
        dwMemoryLoad: u32,
        ullTotalPhys: u64,
        ullAvailPhys: u64,
        ullTotalPageFile: u64,
        ullAvailPageFile: u64,
        ullTotalVirtual: u64,
        ullAvailVirtual: u64,
        ullAvailExtendedVirtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(lpBuffer: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        dwLength: std::mem::size_of::<MemoryStatusEx>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: status is a fully initialized MemoryStatusEx whose dwLength
    // field is set per the GlobalMemoryStatusEx contract; the call only
    // writes through the single pointer we pass.
    if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 {
        Some(status.ullTotalPhys)
    } else {
        None
    }
}

#[cfg(not(target_os = "windows"))]
fn probe_ram() -> Option<u64> {
    match std::env::consts::OS {
        "linux" => {
            let info = read_command("cat", &["/proc/meminfo"])?;
            for line in info.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kib: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
                    return Some(kib * 1024);
                }
            }
            None
        }
        "macos" => {
            let out = read_command("sysctl", &["-n", "hw.memsize"])?;
            out.trim().parse().ok()
        }
        _ => None,
    }
}

/// CUDA GPUs via nvidia-smi; empty when nvidia-smi is absent or fails.
fn probe_cuda_gpus() -> Vec<GpuProfile> {
    let Some(output) = read_command(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    ) else {
        return Vec::new();
    };
    output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let mut fields = line.split(',');
            let name = fields.next()?.trim().to_string();
            let mib: u64 = fields.next()?.trim().parse().ok()?;
            let capability = fields.next()?.trim();
            let (cc_major, cc_minor) = capability.split_once('.')?;
            let cc_major: u32 = cc_major.trim().parse().ok()?;
            let cc_minor: u32 = cc_minor.trim().parse().ok()?;
            Some(GpuProfile::Cuda {
                name,
                vram_bytes: mib * 1024 * 1024,
                cc_major,
                cc_minor,
                dp4a: cc_major > 6 || (cc_major == 6 && cc_minor >= 1),
                tensor_cores: cc_major >= 7,
                native_fp8: cc_major >= 8 && (cc_major > 8 || cc_minor >= 9),
                native_fp4: cc_major >= 12,
            })
        })
        .collect()
}

fn probe_gpus() -> Vec<GpuProfile> {
    let mut gpus = probe_cuda_gpus();
    if std::env::consts::OS == "macos" && std::env::consts::ARCH == "aarch64" {
        gpus.push(GpuProfile::Metal {
            name: read_command("sysctl", &["-n", "hw.model"]).unwrap_or_default(),
            family: 8,
        });
    }
    gpus
}

fn read_command(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic 64 GB AVX2 box + GTX 1080 (the umbrella success case).
    fn gtx1080_box() -> MachineProfile {
        MachineProfile {
            os: "windows".into(),
            arch: "x86_64".into(),
            cpu: CpuProfile {
                vendor: "AuthenticAMD".into(),
                brand: "AMD Ryzen 5 5600X".into(),
                logical_cores: 12,
                avx2: true,
                avx_vnni: false,
                avx512_f: false,
                avx512_bw: false,
                avx512_vnni: false,
                neon_i8mm: false,
                neon_dotprod: false,
            },
            ram_bytes: 64 * 1024 * 1024 * 1024,
            gpus: vec![GpuProfile::Cuda {
                name: "NVIDIA GeForce GTX 1080".into(),
                vram_bytes: 8 * 1024 * 1024 * 1024,
                cc_major: 6,
                cc_minor: 1,
                dp4a: true,
                tensor_cores: false,
                native_fp8: false,
                native_fp4: false,
            }],
            storage: StorageClass::Nvme,
        }
    }

    #[test]
    fn profile_json_round_trips() {
        let profile = gtx1080_box();
        let parsed = MachineProfile::from_json(&profile.to_json()).unwrap();
        assert_eq!(profile, parsed);
    }

    #[test]
    fn gtx1080_is_dp4a_capable_but_not_tensor_or_fp4() {
        let profile = gtx1080_box();
        let Some(GpuProfile::Cuda {
            dp4a,
            tensor_cores,
            native_fp8,
            native_fp4,
            ..
        }) = profile.gpus.first()
        else {
            panic!("expected cuda gpu");
        };
        assert!((*dp4a, *tensor_cores, *native_fp8, *native_fp4) == (true, false, false, false));
    }

    #[test]
    fn avx2_capability_is_independent_from_cuda() {
        let mut cpu_only = gtx1080_box();
        cpu_only.gpus.clear();
        assert!(cpu_only.has_avx2());
        assert_eq!(cpu_only.cuda_gpus().count(), 0);
    }

    #[test]
    fn multiple_accelerators_are_representable() {
        let mut profile = gtx1080_box();
        profile.gpus.push(GpuProfile::Metal {
            name: "Mac14,2".into(),
            family: 8,
        });
        let parsed = MachineProfile::from_json(&profile.to_json()).unwrap();
        assert_eq!(parsed.gpus.len(), 2);
    }

    #[test]
    fn capability_facts_stay_separate_from_policy_strings() {
        // The profile must not contain any planner decision fields.
        let profile = gtx1080_box();
        let json = profile.to_json();
        let text = json.to_string();
        for banned in ["plan", "objective", "placement", "layout"] {
            assert!(
                !text.contains(banned),
                "profile leaked policy field {banned}"
            );
        }
    }

    #[test]
    fn unknown_storage_class_is_the_default() {
        let profile = MachineProfile::probe();
        // probe() never panics; storage stays Unknown until classified.
        assert_eq!(profile.storage, StorageClass::Unknown);
        assert!(MachineProfile::from_json(&profile.to_json()).is_ok());
    }
}

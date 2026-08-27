//! #200 — Calibration cache + planner veto.
//!
//! `colic plan --calibration FILE` loads a deterministic per-family
//! calibration cache (produced by `c/tools/calibrate.py`) and lets the
//! objective's quality floor VETO a lower-bit format the rule table chose.
//! Calibration is optional: without the flag the rule-based planner is
//! unchanged (explicit per #200 milestone).

use serde_json::Value;

use crate::{
    error::{ColicError, Result},
    plan::{
        cost::Objective,
        ir::{MathFormat, TensorRole},
    },
};

/// Quality floors: maximum per-family nRMSE a format may exhibit before the
/// planner refuses it (per objective). nRMSE is normalized reconstruction
/// error; 0.1 ≈ 10% of the tensor's own spread.
pub fn nrmse_floor(objective: Objective) -> f64 {
    match objective {
        Objective::Quality => 0.02,
        Objective::Balanced => 0.15,
        Objective::Throughput | Objective::Latency => 0.30,
        Objective::MinimumSize => 0.40,
    }
}

/// Parsed calibration cache (subset of the calibrate.py output we consume).
#[derive(Debug, Clone, Default)]
pub struct Calibration {
    pub fingerprint: Option<String>,
    pub quantizer_version: Option<String>,
    /// family -> format -> nrmse
    pub nrmse: std::collections::BTreeMap<String, std::collections::BTreeMap<String, f64>>,
}

impl Calibration {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| ColicError::Io {
            path: path.to_path_buf(),
            source: error,
        })?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
            ColicError::Usage(format!(
                "calibration file {} is not valid JSON: {error}",
                path.display()
            ))
        })?;
        let mut calibration = Calibration {
            fingerprint: value
                .get("fingerprint")
                .and_then(Value::as_str)
                .map(str::to_owned),
            quantizer_version: value
                .get("quantizer_version")
                .and_then(Value::as_str)
                .map(str::to_owned),
            nrmse: Default::default(),
        };
        let families = value
            .get("families")
            .and_then(Value::as_object)
            .ok_or_else(|| ColicError::Usage("calibration file missing `families`".into()))?;
        for (family, entry) in families {
            let Some(entry) = entry.as_object() else {
                continue;
            };
            let mut formats = std::collections::BTreeMap::new();
            for (format, metrics) in entry {
                if format.starts_with('_') {
                    continue;
                }
                if let Some(nrmse) = metrics.get("nrmse").and_then(Value::as_f64) {
                    formats.insert(format.clone(), nrmse);
                }
            }
            calibration.nrmse.insert(family.clone(), formats);
        }
        Ok(calibration)
    }

    /// nRMSE for a role's chosen format, when the cache covers it.
    pub fn nrmse_for(&self, role: TensorRole, math: MathFormat) -> Option<f64> {
        let family = role.as_str();
        self.nrmse
            .get(family)
            .and_then(|formats| formats.get(format_name(math)))
            .copied()
    }
}

/// Map a MathFormat to the calibrator's format key.
pub fn format_name(math: MathFormat) -> &'static str {
    match math {
        MathFormat::Bf16 => "bf16",
        MathFormat::Fp8E4m3 => "fp8",
        MathFormat::Mxfp4 => "mxfp4",
        MathFormat::Int4G32 => "int4_g32",
        MathFormat::Nvfp4 => "nvfp4",
    }
}

/// Veto check: does calibration data forbid `math` for `role` under this
/// objective? Returns the reason string when vetoed.
pub fn vetoed(
    calibration: &Calibration,
    role: TensorRole,
    math: MathFormat,
    objective: Objective,
) -> Option<String> {
    let nrmse = calibration.nrmse_for(role, math)?;
    let floor = nrmse_floor(objective);
    if nrmse > floor {
        Some(format!(
            "calibration nRMSE {nrmse:.3} for {} exceeds the {} floor {floor:.2}",
            role.as_str(),
            objective.as_str()
        ))
    } else {
        None
    }
}

/// Apply the calibration veto to one role decision: returns the fallback
/// math to use instead of `chosen` when vetoed (next-best among candidates
/// ordered by the caller), or None when the choice stands.
pub fn apply_veto(
    calibration: &Calibration,
    role: TensorRole,
    chosen: MathFormat,
    candidates: &[(MathFormat, f64)], // (format, size-ish score) ascending
    objective: Objective,
) -> Result<MathFormat> {
    if calibration.nrmse.is_empty() {
        return Ok(chosen);
    }
    if let Some(reason) = vetoed(calibration, role, chosen, objective) {
        for (candidate, _) in candidates {
            if *candidate == chosen {
                continue;
            }
            if vetoed(calibration, role, *candidate, objective).is_none() {
                return Ok(*candidate);
            }
        }
        return Err(ColicError::unsupported(
            "calibration",
            format!("{reason}; no alternative format passes the floor"),
        ));
    }
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn calibration_with(nrmse: f64) -> Calibration {
        let json = serde_json::json!({
            "fingerprint": "f".repeat(64),
            "quantizer_version": "colic-quant-v1",
            "families": {
                "routed_expert": {
                    "int4_g32": {"nrmse": nrmse},
                    "mxfp4": {"nrmse": 0.01},
                }
            }
        });
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "colic-calib-{}-{unique}.json",
            std::process::id()
        ));
        std::fs::write(&path, json.to_string()).unwrap();
        let calibration = Calibration::from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        calibration
    }

    #[test]
    fn parses_calibration_cache() {
        let calibration = calibration_with(0.05);
        let expected = "f".repeat(64);
        assert_eq!(calibration.fingerprint.as_deref(), Some(expected.as_str()));
        assert_eq!(
            calibration.nrmse_for(TensorRole::RoutedExpert, MathFormat::Int4G32),
            Some(0.05)
        );
        assert_eq!(
            calibration.nrmse_for(TensorRole::Router, MathFormat::Int4G32),
            None
        );
    }

    #[test]
    fn veto_blocks_high_error_format_and_falls_back() {
        // INT4 at 0.5 nRMSE exceeds the balanced floor (0.15) -> fall to mxfp4.
        let calibration = calibration_with(0.5);
        let chosen = apply_veto(
            &calibration,
            TensorRole::RoutedExpert,
            MathFormat::Int4G32,
            &[(MathFormat::Int4G32, 10.0), (MathFormat::Mxfp4, 30.0)],
            Objective::Balanced,
        )
        .unwrap();
        assert_eq!(chosen, MathFormat::Mxfp4);
    }

    #[test]
    fn no_veto_when_error_is_within_floor() {
        let calibration = calibration_with(0.05);
        let chosen = apply_veto(
            &calibration,
            TensorRole::RoutedExpert,
            MathFormat::Int4G32,
            &[(MathFormat::Int4G32, 10.0), (MathFormat::Mxfp4, 30.0)],
            Objective::Balanced,
        )
        .unwrap();
        assert_eq!(chosen, MathFormat::Int4G32);
    }

    #[test]
    fn no_calibration_means_no_veto() {
        let calibration = Calibration::default();
        let chosen = apply_veto(
            &calibration,
            TensorRole::RoutedExpert,
            MathFormat::Int4G32,
            &[(MathFormat::Int4G32, 10.0)],
            Objective::Quality,
        )
        .unwrap();
        assert_eq!(chosen, MathFormat::Int4G32);
    }

    #[test]
    fn quality_floor_is_stricter() {
        let calibration = calibration_with(0.05);
        // 0.05 passes balanced (0.15) but fails quality (0.02).
        assert!(vetoed(&calibration, TensorRole::RoutedExpert, MathFormat::Int4G32, Objective::Balanced).is_none());
        assert!(vetoed(&calibration, TensorRole::RoutedExpert, MathFormat::Int4G32, Objective::Quality).is_some());
    }
}
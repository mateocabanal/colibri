#[path = "verify_base.rs"]
mod base;

pub use base::{VerificationProgress, VerificationSummary};

use std::path::Path;

use crate::Result;

pub fn verify_package(package: &Path) -> Result<VerificationSummary> {
    let summary = base::verify_package(package)?;
    crate::verify_target::verify_target_layouts(package)?;
    Ok(summary)
}

pub fn verify_package_with_progress(
    package: &Path,
    progress: &mut dyn FnMut(VerificationProgress),
) -> Result<VerificationSummary> {
    let summary = base::verify_package_with_progress(package, progress)?;
    crate::verify_target::verify_target_layouts(package)?;
    Ok(summary)
}

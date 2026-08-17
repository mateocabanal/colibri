#!/usr/bin/env python3
from pathlib import Path


def replace(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"missing patch anchor in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))


replace(
    "colic/src/verify.rs",
    '''pub fn verify_package(package: &Path) -> Result<VerificationSummary> {
    verify_package_with_progress(package, &mut |_| {})
}

/// Validates final package bytes and reports bounded-rate progress. The
/// verifier deliberately reports after integrity checks, not merely after
/// metadata parsing, so reported bytes correspond to completed validation.
pub fn verify_package_with_progress(
    package: &Path,
    progress: &mut dyn FnMut(VerificationProgress),
) -> Result<VerificationSummary> {
    let manifest_path = package.join("manifest.coli");''',
    '''pub fn verify_package(package: &Path) -> Result<VerificationSummary> {
    verify_package_with_progress(package, &mut |_| {})
}

/// Performs the mandatory pre-publication structural reopen without rereading
/// every record payload. This validates framing, checksums for manifest/shard
/// headers, string/descriptor tables, shard sizes and every record range.
pub fn verify_package_structure(package: &Path) -> Result<VerificationSummary> {
    verify_package_impl(package, &mut |_| {}, false)
}

/// Validates final package bytes and reports bounded-rate progress. The
/// verifier deliberately reports after integrity checks, not merely after
/// metadata parsing, so reported bytes correspond to completed validation.
pub fn verify_package_with_progress(
    package: &Path,
    progress: &mut dyn FnMut(VerificationProgress),
) -> Result<VerificationSummary> {
    verify_package_impl(package, progress, true)
}

fn verify_package_impl(
    package: &Path,
    progress: &mut dyn FnMut(VerificationProgress),
    verify_payloads: bool,
) -> Result<VerificationSummary> {
    let manifest_path = package.join("manifest.coli");''',
)

replace(
    "colic/src/verify.rs",
    '''        if crc32c_file_range(&shard_paths[shard_id], offset, stored)? != stored_crc {
            return invalid("record stored CRC32C does not match");
        }
        match kind {
            1 => verify_tensor_record(
                &shard_paths[shard_id],
                offset,
                stored,
                resident,
                codec,
                flags,
                logical_crc,
            )?,
            2 => verify_expert_record(
                &shard_paths[shard_id],
                offset,
                stored,
                resident,
                layer,
                expert,
                codec,
            )?,
            _ => {}
        }
        verified_bytes = verified_bytes
            .checked_add(stored)
            .ok_or_else(|| usage("verified byte total overflows"))?;
        let completed_records = index + 1;
        if completed_records % 64 == 0 || completed_records == records {
            progress(VerificationProgress {
                completed_records,
                total_records: records,
                verified_bytes,
                current_shard: shard_id as u32,
                total_shards: shards,
            });
        }''',
    '''        if verify_payloads {
            if crc32c_file_range(&shard_paths[shard_id], offset, stored)? != stored_crc {
                return invalid("record stored CRC32C does not match");
            }
            match kind {
                1 => verify_tensor_record(
                    &shard_paths[shard_id],
                    offset,
                    stored,
                    resident,
                    codec,
                    flags,
                    logical_crc,
                )?,
                2 => verify_expert_record(
                    &shard_paths[shard_id],
                    offset,
                    stored,
                    resident,
                    layer,
                    expert,
                    codec,
                )?,
                _ => {}
            }
            verified_bytes = verified_bytes
                .checked_add(stored)
                .ok_or_else(|| usage("verified byte total overflows"))?;
            let completed_records = index + 1;
            if completed_records % 64 == 0 || completed_records == records {
                progress(VerificationProgress {
                    completed_records,
                    total_records: records,
                    verified_bytes,
                    current_shard: shard_id as u32,
                    total_shards: shards,
                });
            }
        }''',
)

replace(
    "colic/src/pipeline.rs",
    '''    if request.verify {
        progress.stage(Stage::Verification);
        if let Err(error) = verify::verify_package(&temporary) {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
    }
    if request.force {''',
    '''    if let Err(error) = verify::verify_package_structure(&temporary) {
        let _ = fs::remove_dir_all(&temporary);
        return Err(error);
    }
    if request.verify {
        progress.stage(Stage::Verification);
        if let Err(error) = verify::verify_package(&temporary) {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
    }
    if request.force {''',
)

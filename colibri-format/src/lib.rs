//! COLI CSF artifact framing shared by the `colic` compiler and the future Rust
//! inference runtime: checksums, manifest/data-shard constants, and the
//! package reader. No unsafe code, no dependency on compiler internals.
#![forbid(unsafe_code)]

pub mod codecs;
pub mod package;
pub mod verify;

pub use verify::{FormatError, Result};

/// COLI data shard file magic.
pub const DATA_MAGIC: &[u8; 8] = b"COLIDAT\0";
/// COLI manifest file magic.
pub const MANIFEST_MAGIC: &[u8; 8] = b"COLI\r\n\x1a\n";
/// Size of a data shard header.
pub const DATA_SHARD_HEADER_BYTES: u64 = 128;
/// Size of a manifest header.
pub const MANIFEST_HEADER_BYTES: usize = 256;

/// Rounds a value up to a power-of-two alignment.
pub fn align_up(value: u64, alignment: u64) -> crate::verify::Result<u64> {
    crate::verify::align_up_impl(value, alignment)
}

/// CRC-32C (Castagnoli) checksum over a byte slice.
pub fn crc32c(bytes: &[u8]) -> u32 {
    crate::verify::crc32c_impl(bytes)
}

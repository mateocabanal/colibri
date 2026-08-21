use colic::{target, target_registry};

#[test]
fn apple8_profile_uses_generated_production_identity() {
    let profile = target::MACOS_ARM64_METAL_APPLE8_V1;
    assert_eq!(profile.id, target_registry::APPLE8_PROFILE_ID);
    assert_eq!(profile.name, target_registry::APPLE8_PROFILE_NAME);
    assert_eq!(profile.execution_layout_abi, target_registry::APPLE8_EXECUTION_LAYOUT_ABI);
    assert_eq!(profile.kernel_abi, target_registry::APPLE8_KERNEL_ABI);
    assert_eq!(target_registry::APPLE8_TARGET_CLASS, 0x0100_0001);
    assert_eq!(target_registry::APPLE8_MXFP4_TILE_LAYOUT, 0x0103);
    assert_eq!(target_registry::APPLE8_MXFP4_TILE_ROWS, 8);
    assert_eq!(target_registry::APPLE8_MXFP4_TILE_COLUMNS, 32);
    assert_eq!(target_registry::APPLE8_MXFP4_TILE_BYTES, 136);
}

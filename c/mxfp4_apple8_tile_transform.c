#include "mxfp4_apple8_tile.h"

#include <limits.h>
#include <string.h>

static uint64_t align64(uint64_t value) {
    if (value > UINT64_MAX - 63u) return 0;
    return (value + 63u) & ~UINT64_C(63);
}

static int source_descriptor_valid(const ColiMxfp4Apple8SourceExpert *source) {
    if (!source || source->matrix_count != 3) return 0;
    for (uint32_t i = 0; i < source->matrix_count; ++i) {
        const ColiMxfp4Apple8RowMatrix *m = &source->matrices[i];
        if (!m->role || !m->rows || !m->columns || !m->weights || !m->scales)
            return 0;
        uint64_t row_bytes = (m->columns + 1u) / 2u;
        uint64_t groups = (m->columns + 31u) / 32u;
        if (m->rows > UINT64_MAX / row_bytes ||
            m->rows * row_bytes > m->weight_bytes ||
            m->rows > UINT64_MAX / groups ||
            m->rows * groups > m->scale_bytes)
            return 0;
    }
    return 1;
}

static int layout_plan(const ColiMxfp4Apple8SourceExpert *source,
                       uint64_t *resident_bytes,
                       ColiMxfp4Apple8TileMatrix entries[3]) {
    if (!source_descriptor_valid(source) || !resident_bytes || !entries)
        return -1;
    uint64_t cursor = align64(sizeof(ColiMxfp4Apple8TileExpertHeader));
    if (!cursor) return -1;
    for (uint32_t i = 0; i < 3; ++i) {
        const ColiMxfp4Apple8RowMatrix *src = &source->matrices[i];
        size_t bytes = coli_mxfp4_apple8_tile_bytes(src->rows, src->columns);
        if (!bytes) return -1;
        cursor = align64(cursor);
        if (!cursor || cursor > UINT64_MAX - (uint64_t)bytes) return -1;
        entries[i] = (ColiMxfp4Apple8TileMatrix){
            .role = src->role,
            .reserved = 0,
            .rows = src->rows,
            .columns = src->columns,
            .offset = cursor,
            .bytes = (uint64_t)bytes,
        };
        cursor += (uint64_t)bytes;
    }
    *resident_bytes = cursor;
    return 0;
}

static int apple8_estimate(void *context,
                           const ColiExpertResidentView *source_view,
                           const ColiRepresentationId *target,
                           ColiJitTransformEstimate *estimate) {
    (void)context;
    ColiRepresentationId expected;
    coli_mxfp4_apple8_target_fixture_representation(&expected);
    if (!source_view || !target || !estimate || !source_view->physical ||
        !coli_representation_equal(target, &expected))
        return -1;
    ColiMxfp4Apple8TileMatrix entries[3];
    uint64_t bytes = 0;
    if (layout_plan((const ColiMxfp4Apple8SourceExpert *)source_view->physical,
                    &bytes, entries))
        return -1;
    *estimate = (ColiJitTransformEstimate){
        .resident_bytes = bytes,
        .allocation_bytes = bytes,
        .scratch_bytes = 0,
        .staging_bytes = 0,
        .output_alignment = 64,
        .scratch_alignment = 0,
        .staging_alignment = 0,
    };
    return 0;
}

static int apple8_prepare(void *context,
                          const ColiExpertResidentView *source_view,
                          const ColiRepresentationId *target,
                          void *output, uint64_t output_bytes,
                          void *scratch, uint64_t scratch_bytes,
                          void *staging, uint64_t staging_bytes) {
    (void)context;
    (void)scratch;
    (void)scratch_bytes;
    (void)staging;
    (void)staging_bytes;
    ColiRepresentationId expected;
    coli_mxfp4_apple8_target_fixture_representation(&expected);
    if (!source_view || !source_view->physical || !target || !output ||
        !coli_representation_equal(target, &expected))
        return -1;

    const ColiMxfp4Apple8SourceExpert *source =
        (const ColiMxfp4Apple8SourceExpert *)source_view->physical;
    ColiMxfp4Apple8TileMatrix entries[3];
    uint64_t needed = 0;
    if (layout_plan(source, &needed, entries) || output_bytes < needed ||
        needed > SIZE_MAX)
        return -1;

    memset(output, 0, (size_t)needed);
    ColiMxfp4Apple8TileExpertHeader *header =
        (ColiMxfp4Apple8TileExpertHeader *)output;
    header->magic = COLI_MXFP4_APPLE8_TILE_EXPERT_MAGIC;
    header->version = COLI_MXFP4_APPLE8_TILE_EXPERT_VERSION;
    header->matrix_count = 3;
    header->header_bytes = (uint32_t)align64(sizeof(*header));
    header->reserved = 0;
    memcpy(header->matrices, entries, sizeof(entries));

    for (uint32_t i = 0; i < 3; ++i) {
        const ColiMxfp4Apple8RowMatrix *src = &source->matrices[i];
        if (coli_mxfp4_apple8_tile_repack(
                (uint8_t *)output + entries[i].offset,
                (size_t)entries[i].bytes,
                src->weights, (size_t)src->weight_bytes,
                src->scales, (size_t)src->scale_bytes,
                src->rows, src->columns))
            return -1;
    }
    return 0;
}

static int apple8_validate(void *context,
                           const ColiExpertResidentView *source_view,
                           const ColiRepresentationId *target,
                           const void *output, uint64_t resident_bytes) {
    (void)context;
    ColiRepresentationId expected;
    coli_mxfp4_apple8_target_fixture_representation(&expected);
    if (!source_view || !source_view->physical || !target || !output ||
        !coli_representation_equal(target, &expected) ||
        resident_bytes < sizeof(ColiMxfp4Apple8TileExpertHeader))
        return -1;

    const ColiMxfp4Apple8SourceExpert *source =
        (const ColiMxfp4Apple8SourceExpert *)source_view->physical;
    ColiMxfp4Apple8TileMatrix expected_entries[3];
    uint64_t expected_bytes = 0;
    if (layout_plan(source, &expected_bytes, expected_entries) ||
        resident_bytes != expected_bytes)
        return -1;

    const ColiMxfp4Apple8TileExpertHeader *header =
        (const ColiMxfp4Apple8TileExpertHeader *)output;
    if (header->magic != COLI_MXFP4_APPLE8_TILE_EXPERT_MAGIC ||
        header->version != COLI_MXFP4_APPLE8_TILE_EXPERT_VERSION ||
        header->matrix_count != 3 ||
        header->header_bytes != (uint32_t)align64(sizeof(*header)) ||
        header->reserved != 0)
        return -1;

    for (uint32_t i = 0; i < 3; ++i) {
        const ColiMxfp4Apple8TileMatrix *actual = &header->matrices[i];
        const ColiMxfp4Apple8TileMatrix *wanted = &expected_entries[i];
        const ColiMxfp4Apple8RowMatrix *src = &source->matrices[i];
        if (memcmp(actual, wanted, sizeof(*actual)) != 0 ||
            actual->offset > resident_bytes ||
            actual->bytes > resident_bytes - actual->offset ||
            coli_mxfp4_apple8_tile_validate(
                (const uint8_t *)output + actual->offset,
                (size_t)actual->bytes,
                src->weights, (size_t)src->weight_bytes,
                src->scales, (size_t)src->scale_bytes,
                src->rows, src->columns))
            return -1;
    }
    return 0;
}

int coli_mxfp4_apple8_transform_ops(ColiRepresentationTransformOps *ops) {
    if (!ops) return -1;
    ColiRepresentationId source, target;
    coli_mxfp4_apple8_source_fixture_representation(&source);
    coli_mxfp4_apple8_target_fixture_representation(&target);
    *ops = (ColiRepresentationTransformOps){
        .source = source,
        .target = target,
        .transform_abi = COLI_MXFP4_APPLE8_FIXTURE_TRANSFORM_ABI,
        .transform_class = COLI_JIT_TRANSFORM_EXACT,
        .target_tier_mask = COLI_EXPERT_TIER_UMA,
        /* Diagnostic fixture tag only; not a production backend/layout ID. */
        .backend_tag = UINT32_C(0x41383131),
        .context = NULL,
        .estimate = apple8_estimate,
        .prepare = apple8_prepare,
        .validate = apple8_validate,
    };
    return 0;
}

int coli_mxfp4_apple8_register_transform(
        ColiRepresentationTransformRegistry *registry) {
    ColiRepresentationTransformOps ops;
    if (!registry || coli_mxfp4_apple8_transform_ops(&ops)) return -1;
    return coli_jit_transform_registry_register(registry, &ops);
}

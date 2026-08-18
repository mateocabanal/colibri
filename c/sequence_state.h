#ifndef COLIBRI_SEQUENCE_STATE_H
#define COLIBRI_SEQUENCE_STATE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef enum {
    COLI_SEQUENCE_SEGMENT_PAGED_APPEND = 0,
    COLI_SEQUENCE_SEGMENT_RING = 1,
    COLI_SEQUENCE_SEGMENT_RECURRENT_FIXED = 2,
    COLI_SEQUENCE_SEGMENT_SMALL_INDEX = 3,
    COLI_SEQUENCE_SEGMENT_ENGINE_NATIVE = 4,
} ColiSequenceSegmentKind;

typedef enum {
    COLI_SEQUENCE_VIS_CPU = 1u << 0,
    COLI_SEQUENCE_VIS_ACCELERATOR = 1u << 1,
} ColiSequenceVisibility;

typedef struct {
    /* Stable within one state ABI. The global cache persists this identity, not
     * a model-private struct address. */
    uint32_t segment_id;
    ColiSequenceSegmentKind kind;
    uint32_t element_bytes;
    uint32_t visibility;
    uint64_t logical_rows;
    uint64_t row_bytes;
    uint64_t snapshot_bytes;
    uint64_t page_rows;      /* >0 only for page-addressable append state */
    uint64_t layout_abi;
} ColiSequenceSegmentDesc;

typedef struct {
    uint64_t state_abi;
    uint64_t absolute_position;
    uint64_t logical_bytes;
    uint64_t resident_bytes;
    size_t segment_count;
} ColiSequenceStateInfo;

typedef struct ColiSequenceStateAdapter ColiSequenceStateAdapter;

typedef struct {
    /* Populate stable descriptors for the exact resumable boundary. Passing
     * segments=NULL asks only for `info`/segment_count. */
    int (*describe)(void *ctx, uint64_t absolute_position,
                    ColiSequenceStateInfo *info,
                    ColiSequenceSegmentDesc *segments, size_t segment_capacity);

    /* Copy a slice of one snapshot segment to/from caller-owned storage. The
     * adapter owns semantic packing. Common cache/persistence owns indexing,
     * budgets, framing and checksums. */
    int (*read_segment)(void *ctx, uint64_t absolute_position,
                        uint32_t segment_id, uint64_t offset,
                        void *dst, size_t bytes);
    int (*write_segment)(void *ctx, uint64_t absolute_position,
                         uint32_t segment_id, uint64_t offset,
                         const void *src, size_t bytes);

    int (*reset)(void *ctx);
    int (*finish_restore)(void *ctx, uint64_t absolute_position);
} ColiSequenceStateOps;

struct ColiSequenceStateAdapter {
    void *ctx;
    const ColiSequenceStateOps *ops;
};

typedef struct {
    ColiSequenceStateInfo info;
    ColiSequenceSegmentDesc *segments;
    size_t segment_capacity;
} ColiSequenceSnapshotLayout;

static inline int coli_sequence_u64_mul(uint64_t a, uint64_t b,
                                        uint64_t *out) {
    if (!out || (a && b > UINT64_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static inline int coli_sequence_u64_add(uint64_t a, uint64_t b,
                                        uint64_t *out) {
    if (!out || UINT64_MAX - a < b) return -1;
    *out = a + b;
    return 0;
}

static inline int coli_sequence_segment_valid(
        const ColiSequenceSegmentDesc *segment) {
    if (!segment || !segment->segment_id || !segment->snapshot_bytes ||
        segment->kind < COLI_SEQUENCE_SEGMENT_PAGED_APPEND ||
        segment->kind > COLI_SEQUENCE_SEGMENT_ENGINE_NATIVE ||
        !segment->visibility)
        return 0;
    if (segment->kind == COLI_SEQUENCE_SEGMENT_PAGED_APPEND) {
        if (!segment->page_rows || !segment->logical_rows || !segment->row_bytes)
            return 0;
        uint64_t logical;
        if (coli_sequence_u64_mul(segment->logical_rows, segment->row_bytes,
                                  &logical) != 0 ||
            logical != segment->snapshot_bytes)
            return 0;
    }
    if (segment->kind == COLI_SEQUENCE_SEGMENT_RING &&
        (!segment->logical_rows || !segment->row_bytes))
        return 0;
    return 1;
}

/* Validate the adapter's state boundary before any bytes are copied. Segment
 * ids must be unique and aggregate snapshot size must be overflow-safe. */
static inline int coli_sequence_snapshot_layout(
        const ColiSequenceStateAdapter *adapter,
        uint64_t absolute_position,
        ColiSequenceSnapshotLayout *layout) {
    if (!adapter || !adapter->ctx || !adapter->ops || !adapter->ops->describe ||
        !layout)
        return -1;
    ColiSequenceStateInfo info;
    memset(&info, 0, sizeof(info));
    if (adapter->ops->describe(adapter->ctx, absolute_position, &info, NULL, 0) != 0 ||
        !info.state_abi || info.absolute_position != absolute_position ||
        !info.segment_count || info.segment_count > layout->segment_capacity ||
        !layout->segments)
        return -1;

    if (adapter->ops->describe(adapter->ctx, absolute_position, &info,
                               layout->segments, layout->segment_capacity) != 0)
        return -1;

    uint64_t total = 0;
    for (size_t i = 0; i < info.segment_count; ++i) {
        if (!coli_sequence_segment_valid(&layout->segments[i])) return -1;
        for (size_t j = 0; j < i; ++j)
            if (layout->segments[j].segment_id == layout->segments[i].segment_id)
                return -1;
        if (coli_sequence_u64_add(total, layout->segments[i].snapshot_bytes,
                                  &total) != 0)
            return -1;
    }
    if (info.logical_bytes && info.logical_bytes > info.resident_bytes &&
        !info.resident_bytes)
        return -1;
    layout->info = info;
    return 0;
}

static inline int coli_sequence_snapshot_read_all(
        const ColiSequenceStateAdapter *adapter,
        const ColiSequenceSnapshotLayout *layout,
        void *dst, size_t dst_bytes) {
    if (!adapter || !adapter->ops || !adapter->ops->read_segment || !layout || !dst)
        return -1;
    uint64_t total = 0;
    for (size_t i = 0; i < layout->info.segment_count; ++i) {
        uint64_t next;
        if (coli_sequence_u64_add(total, layout->segments[i].snapshot_bytes, &next) != 0 ||
            next > dst_bytes || layout->segments[i].snapshot_bytes > SIZE_MAX)
            return -1;
        if (adapter->ops->read_segment(
                adapter->ctx, layout->info.absolute_position,
                layout->segments[i].segment_id, 0,
                (unsigned char *)dst + (size_t)total,
                (size_t)layout->segments[i].snapshot_bytes) != 0)
            return -1;
        total = next;
    }
    return 0;
}

static inline int coli_sequence_snapshot_restore_all(
        const ColiSequenceStateAdapter *adapter,
        const ColiSequenceSnapshotLayout *layout,
        const void *src, size_t src_bytes) {
    if (!adapter || !adapter->ops || !adapter->ops->write_segment || !layout || !src)
        return -1;
    if (adapter->ops->reset && adapter->ops->reset(adapter->ctx) != 0)
        return -1;
    uint64_t total = 0;
    for (size_t i = 0; i < layout->info.segment_count; ++i) {
        uint64_t next;
        if (coli_sequence_u64_add(total, layout->segments[i].snapshot_bytes, &next) != 0 ||
            next > src_bytes || layout->segments[i].snapshot_bytes > SIZE_MAX)
            return -1;
        if (adapter->ops->write_segment(
                adapter->ctx, layout->info.absolute_position,
                layout->segments[i].segment_id, 0,
                (const unsigned char *)src + (size_t)total,
                (size_t)layout->segments[i].snapshot_bytes) != 0)
            return -1;
        total = next;
    }
    if (adapter->ops->finish_restore &&
        adapter->ops->finish_restore(adapter->ctx,
                                     layout->info.absolute_position) != 0)
        return -1;
    return 0;
}

static inline uint64_t coli_sequence_snapshot_bytes(
        const ColiSequenceSnapshotLayout *layout) {
    if (!layout) return 0;
    uint64_t total = 0;
    for (size_t i = 0; i < layout->info.segment_count; ++i)
        if (coli_sequence_u64_add(total, layout->segments[i].snapshot_bytes,
                                  &total) != 0)
            return UINT64_MAX;
    return total;
}

#endif

#ifndef COLIBRI_EXPERT_REPRESENTATION_H
#define COLIBRI_EXPERT_REPRESENTATION_H

#include "coli_format.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Resident representation identity.
 *
 * This deliberately reuses CSF's math/scale/layout numeric registries rather
 * than defining runtime- or backend-private format ordinals. target_class is
 * the exact target/profile compatibility class that owns the execution ABI
 * (for example macos-arm64-metal-apple8-v1). It is correctness identity, not
 * a model name and not a tuning fingerprint.
 *
 * Storage codec/framing is intentionally absent: different lossless stored
 * encodings may decode to the same resident representation.
 */
#define COLI_REPRESENTATION_TARGET_CLASS_BYTES 64u

typedef struct {
    uint16_t math_format;          /* COLI_CSF_MATH_* */
    uint16_t scale_format;         /* COLI_CSF_SCALE_* */
    uint16_t execution_layout;     /* COLI_CSF_LAYOUT_* / #26 registry */
    uint16_t reserved0;
    uint32_t execution_layout_abi;
    uint32_t kernel_abi;
    uint32_t scale_block_rows;
    uint32_t scale_block_columns;
    uint32_t group_size;
    uint32_t flags;
    char target_class[COLI_REPRESENTATION_TARGET_CLASS_BYTES];
} ColiRepresentationId;

enum {
    COLI_REPRESENTATION_F_NONE = 0u
};

static inline size_t coli_representation_bounded_strlen(
        const char *value, size_t capacity) {
    if (!value) return capacity;
    size_t n = 0;
    while (n < capacity && value[n]) n++;
    return n;
}

static inline int coli_representation_target_class_valid(const char *value) {
    if (!value || !value[0]) return 0;
    return coli_representation_bounded_strlen(
               value, COLI_REPRESENTATION_TARGET_CLASS_BYTES) <
           COLI_REPRESENTATION_TARGET_CLASS_BYTES;
}

static inline int coli_representation_valid(const ColiRepresentationId *rep) {
    if (!rep || rep->reserved0 != 0 ||
        !coli_representation_target_class_valid(rep->target_class) ||
        !rep->execution_layout_abi || !rep->kernel_abi)
        return 0;
    if (rep->math_format == COLI_CSF_MATH_NONE ||
        rep->math_format == COLI_CSF_MATH_MIXED ||
        rep->math_format == COLI_CSF_MATH_INVALID)
        return 0;
    if (rep->scale_format == COLI_CSF_SCALE_MIXED ||
        rep->scale_format == COLI_CSF_SCALE_INVALID)
        return 0;
    if (rep->execution_layout == COLI_CSF_LAYOUT_MIXED ||
        rep->execution_layout == COLI_CSF_LAYOUT_INVALID)
        return 0;
    return 1;
}

static inline int coli_representation_init(
        ColiRepresentationId *out,
        uint16_t math_format,
        uint16_t scale_format,
        uint16_t execution_layout,
        uint32_t execution_layout_abi,
        uint32_t kernel_abi,
        uint32_t scale_block_rows,
        uint32_t scale_block_columns,
        uint32_t group_size,
        uint32_t flags,
        const char *target_class) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    size_t target_bytes = coli_representation_bounded_strlen(
        target_class, COLI_REPRESENTATION_TARGET_CLASS_BYTES);
    if (!target_class || !target_bytes ||
        target_bytes >= COLI_REPRESENTATION_TARGET_CLASS_BYTES)
        return -1;

    out->math_format = math_format;
    out->scale_format = scale_format;
    out->execution_layout = execution_layout;
    out->execution_layout_abi = execution_layout_abi;
    out->kernel_abi = kernel_abi;
    out->scale_block_rows = scale_block_rows;
    out->scale_block_columns = scale_block_columns;
    out->group_size = group_size;
    out->flags = flags;
    memcpy(out->target_class, target_class, target_bytes);
    out->target_class[target_bytes] = '\0';
    if (!coli_representation_valid(out)) {
        memset(out, 0, sizeof(*out));
        return -1;
    }
    return 0;
}

static inline int coli_representation_equal(
        const ColiRepresentationId *a, const ColiRepresentationId *b) {
    if (!coli_representation_valid(a) || !coli_representation_valid(b))
        return 0;
    return a->math_format == b->math_format &&
        a->scale_format == b->scale_format &&
        a->execution_layout == b->execution_layout &&
        a->reserved0 == b->reserved0 &&
        a->execution_layout_abi == b->execution_layout_abi &&
        a->kernel_abi == b->kernel_abi &&
        a->scale_block_rows == b->scale_block_rows &&
        a->scale_block_columns == b->scale_block_columns &&
        a->group_size == b->group_size &&
        a->flags == b->flags &&
        strncmp(a->target_class, b->target_class,
                COLI_REPRESENTATION_TARGET_CLASS_BYTES) == 0;
}

/*
 * True only when the declared quantized mathematical interpretation is the
 * same. Physical layout, target class and kernel ABI may differ, which is the
 * exact-layout-repack case #135 will use. This does not claim that a transform
 * implementation exists and does not authorize lossy requantization.
 */
static inline int coli_representation_exact_math_compatible(
        const ColiRepresentationId *a, const ColiRepresentationId *b) {
    if (!coli_representation_valid(a) || !coli_representation_valid(b))
        return 0;
    return a->math_format == b->math_format &&
        a->scale_format == b->scale_format &&
        a->scale_block_rows == b->scale_block_rows &&
        a->scale_block_columns == b->scale_block_columns &&
        a->group_size == b->group_size &&
        a->flags == b->flags;
}

/*
 * Fail-closed execution compatibility for an already-selected backend target.
 * Target/profile discovery remains owned by the target loader/backend layer.
 */
static inline int coli_representation_backend_can_execute(
        const ColiRepresentationId *rep,
        const char *target_class,
        uint32_t execution_layout_abi,
        uint32_t kernel_abi) {
    if (!coli_representation_valid(rep) ||
        !coli_representation_target_class_valid(target_class) ||
        !execution_layout_abi || !kernel_abi)
        return 0;
    return rep->execution_layout_abi == execution_layout_abi &&
        rep->kernel_abi == kernel_abi &&
        strncmp(rep->target_class, target_class,
                COLI_REPRESENTATION_TARGET_CLASS_BYTES) == 0;
}

/*
 * Conservative mechanism-level predicate for the exact-transform workstream:
 * distinct physical representations with identical declared quant semantics
 * may have a transform. Registry/capability lookup in #135 decides whether one
 * actually exists.
 */
static inline int coli_representation_transform_may_exist(
        const ColiRepresentationId *source,
        const ColiRepresentationId *target) {
    return coli_representation_exact_math_compatible(source, target) &&
        !coli_representation_equal(source, target);
}

static inline int coli_representation_matrix_semantics_equal(
        const ColiExpertMatrixInfo *a, const ColiExpertMatrixInfo *b) {
    if (!a || !b) return 0;
    return a->math_format == b->math_format &&
        a->scale_format == b->scale_format &&
        a->layout == b->layout &&
        a->scale_block_rows == b->scale_block_rows &&
        a->scale_block_columns == b->scale_block_columns &&
        a->group_size == b->group_size;
}

/*
 * Resolve a homogeneous compiler-emitted expert envelope into the resident
 * representation vocabulary. Codec fields are deliberately ignored.
 *
 * Returns -2 when the three matrices do not share one representation; such a
 * record requires a future compound descriptor rather than lossy inference.
 */
static inline int coli_representation_from_expert_info(
        const ColiExpertInfo *expert,
        const char *target_class,
        uint32_t execution_layout_abi,
        uint32_t kernel_abi,
        uint32_t flags,
        ColiRepresentationId *out) {
    if (!expert || !out) return -1;
    const ColiExpertMatrixInfo *first = &expert->matrices[0];
    for (size_t i = 1; i < 3; i++) {
        if (!coli_representation_matrix_semantics_equal(
                first, &expert->matrices[i])) {
            memset(out, 0, sizeof(*out));
            return -2;
        }
    }
    return coli_representation_init(
        out, first->math_format, first->scale_format, first->layout,
        execution_layout_abi, kernel_abi,
        first->scale_block_rows, first->scale_block_columns,
        first->group_size, flags, target_class);
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_EXPERT_REPRESENTATION_H */

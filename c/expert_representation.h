#ifndef COLIBRI_EXPERT_REPRESENTATION_H
#define COLIBRI_EXPERT_REPRESENTATION_H

#include "coli_format.h"

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Backend- and model-neutral identity of bytes that are ready for execution.
 *
 * math_format / scale_format / execution_layout are the stable COLI_CSF_*
 * semantic IDs. execution_layout_abi and kernel_abi are populated from the
 * target-profile ABI registry (#26); this header intentionally does not create
 * a competing target/backend enum. target_class is likewise an opaque numeric
 * compatibility class owned by that registry. A value of zero means that no
 * target-specific compatibility class is required by this representation.
 *
 * Storage codecs are deliberately absent: two differently framed/compressed
 * records may decode to the same resident representation.
 */
typedef struct {
    uint16_t math_format;
    uint16_t scale_format;
    uint16_t execution_layout;
    uint16_t execution_layout_abi;
    uint16_t kernel_abi;
    uint16_t reserved;
    uint32_t target_class;
    uint32_t group_size;
    uint32_t scale_block_rows;
    uint32_t scale_block_columns;
    uint32_t flags;
} ColiRepresentationId;

static inline void coli_representation_clear(ColiRepresentationId *rep) {
    if (rep) memset(rep, 0, sizeof(*rep));
}

static inline int coli_representation_known(const ColiRepresentationId *rep) {
    return rep && rep->math_format != COLI_CSF_MATH_NONE &&
        rep->math_format != COLI_CSF_MATH_MIXED &&
        rep->math_format != COLI_CSF_MATH_INVALID &&
        rep->scale_format != COLI_CSF_SCALE_MIXED &&
        rep->scale_format != COLI_CSF_SCALE_INVALID &&
        rep->execution_layout != COLI_CSF_LAYOUT_MIXED &&
        rep->execution_layout != COLI_CSF_LAYOUT_INVALID &&
        rep->reserved == 0;
}

static inline int coli_representation_equal(
        const ColiRepresentationId *a, const ColiRepresentationId *b) {
    if (!a || !b) return 0;
    return a->math_format == b->math_format &&
        a->scale_format == b->scale_format &&
        a->execution_layout == b->execution_layout &&
        a->execution_layout_abi == b->execution_layout_abi &&
        a->kernel_abi == b->kernel_abi &&
        a->target_class == b->target_class &&
        a->group_size == b->group_size &&
        a->scale_block_rows == b->scale_block_rows &&
        a->scale_block_columns == b->scale_block_columns &&
        a->flags == b->flags;
}

/* Exact mathematical compatibility intentionally ignores physical layout,
 * kernel ABI and target class, but it does include scale geometry. This is the
 * predicate an exact physical repack may use; it is not a permission to
 * requantize into a different mathematical format. */
static inline int coli_representation_exact_math_compatible(
        const ColiRepresentationId *a, const ColiRepresentationId *b) {
    if (!coli_representation_known(a) || !coli_representation_known(b)) return 0;
    return a->math_format == b->math_format &&
        a->scale_format == b->scale_format &&
        a->group_size == b->group_size &&
        a->scale_block_rows == b->scale_block_rows &&
        a->scale_block_columns == b->scale_block_columns;
}

/* Backends advertise exact representations they can execute. Matching is
 * deliberately strict/fail-closed; broader capability tables can be layered on
 * top without weakening the identity contract. */
static inline int coli_representation_backend_can_execute(
        const ColiRepresentationId *resident,
        const ColiRepresentationId *supported) {
    return coli_representation_known(resident) &&
        coli_representation_known(supported) &&
        coli_representation_equal(resident, supported);
}

/* #133 only admits exact-semantics transform candidates. #135 will decide
 * whether a concrete transform implementation actually exists. */
static inline int coli_representation_transform_may_exist(
        const ColiRepresentationId *source,
        const ColiRepresentationId *target) {
    return coli_representation_exact_math_compatible(source, target) &&
        !coli_representation_equal(source, target);
}

/* Resolve a compiled .coli matrix descriptor to the same resident vocabulary.
 * Codec/framing fields are intentionally not consulted. Target/layout/kernel
 * ABI values are supplied by the package target-profile resolver. */
static inline int coli_representation_from_csf_matrix(
        ColiRepresentationId *out,
        const ColiExpertMatrixInfo *matrix,
        uint16_t execution_layout_abi,
        uint16_t kernel_abi,
        uint32_t target_class,
        uint32_t flags) {
    if (!out || !matrix) return -1;
    ColiRepresentationId rep = {
        .math_format = matrix->math_format,
        .scale_format = matrix->scale_format,
        .execution_layout = matrix->layout,
        .execution_layout_abi = execution_layout_abi,
        .kernel_abi = kernel_abi,
        .reserved = 0,
        .target_class = target_class,
        .group_size = matrix->group_size,
        .scale_block_rows = matrix->scale_block_rows,
        .scale_block_columns = matrix->scale_block_columns,
        .flags = flags,
    };
    if (!coli_representation_known(&rep)) return -1;
    *out = rep;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif

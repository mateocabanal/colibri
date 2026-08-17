#include "coli_v4_static.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int bad(char *e, size_t n, const char *f, const char *s) {
    if (e && n) snprintf(e, n, f, s);
    return -1;
}

static uint16_t fmt(ColiSafetensorsDType d) {
    switch (d) {
    case COLI_ST_BF16: return COLI_CSF_MATH_BF16;
    case COLI_ST_F32: return COLI_CSF_MATH_F32;
    case COLI_ST_F8_E4M3: return COLI_CSF_MATH_FP8_E4M3FN;
    /* E8M0 is an exponent byte, not a floating-point tensor payload. */
    case COLI_ST_F8_E8M0: return COLI_CSF_MATH_U8;
    case COLI_ST_I64: return COLI_CSF_MATH_I64;
    default: return COLI_CSF_MATH_INVALID;
    }
}

int coli_v4_coli_layer_load(ColiExecutor *x, ColiDeepSeekV4LayerWeights *w,
                            const ColiDeepSeekV4Config *c, int layer,
                            char *e, size_t n) {
    const int profiling = g_coli_v4_profile_on;
    const uint64_t began = profiling ? coli_v4_profile_now() : 0;
    uint64_t stored_bytes_read = 0;

    if (!x || !w || coli_v4_layer_plan(&w->plan, c, layer, e, n)) return -1;
    memset(w->data, 0, sizeof(w->data));
    memset(&w->stats, 0, sizeof(w->stats));

    for (size_t i = 0; i < w->plan.tensor_count; i++) {
        ColiDeepSeekV4TensorSpec *s = &w->plan.tensors[i];
        const ColiRecordInfo *r = coli_executor_record_by_name(x, s->name);
        ColiTensorInfo t;
        unsigned char *b;

        if (!r || coli_package_tensor_info(coli_executor_package(x), r, &t, e, n) ||
            t.rank != (uint16_t)s->rank || r->math_format != fmt(s->dtype))
            goto fail;
        for (int d = 0; d < s->rank; d++)
            if (t.dims[d] != (uint64_t)s->shape[d]) {
                bad(e, n, "COLI static shape mismatch: %s", s->name);
                goto fail;
            }
        if (r->stored_bytes > SIZE_MAX || t.data_offset > r->stored_bytes ||
            t.data_stored_bytes > r->stored_bytes - t.data_offset) {
            bad(e, n, "COLI static span invalid: %s", s->name);
            goto fail;
        }

        b = malloc((size_t)r->stored_bytes);
        if (!b) {
            bad(e, n, "COLI static allocation failed: %s", s->name);
            goto fail;
        }
        if (coli_executor_load_record(x, r, b, (size_t)r->stored_bytes, e, n)) {
            free(b);
            goto fail;
        }
        stored_bytes_read += r->stored_bytes;

        if (s->dtype == COLI_ST_F8_E8M0) {
            size_t k = (size_t)t.data_stored_bytes;
            float *out = malloc(k * sizeof(*out));
            if (!out) {
                free(b);
                goto fail;
            }
            for (size_t q = 0; q < k; q++)
                out[q] = b[t.data_offset + q] == 255
                    ? NAN : ldexpf(1.f, (int)b[t.data_offset + q] - 127);
            free(b);
            w->data[i] = out;
            w->stats.fp8_scale_bytes += k * sizeof(*out);
            w->stats.total_bytes += k * sizeof(*out);
        } else {
            memmove(b, b + t.data_offset, (size_t)t.data_stored_bytes);
            w->data[i] = b;
            w->stats.total_bytes += t.data_stored_bytes;
        }
        w->stats.tensor_count++;
    }

    if (profiling) {
        coli_v4_profile_add(COLI_V4_PROF_DENSE_READ,
                            coli_v4_profile_now() - began);
        coli_v4_profile_add_bytes(COLI_V4_PROF_DENSE_READ,
                                  stored_bytes_read);
    }
    return 0;

fail:
    if (profiling) {
        coli_v4_profile_add(COLI_V4_PROF_DENSE_READ,
                            coli_v4_profile_now() - began);
        coli_v4_profile_add_bytes(COLI_V4_PROF_DENSE_READ,
                                  stored_bytes_read);
    }
    coli_v4_layer_free(NULL, w);
    return -1;
}

int coli_v4_coli_layer_bytes(ColiExecutor *x, const ColiDeepSeekV4Config *c,
                             int layer, uint64_t *bytes, char *e, size_t n) {
    ColiDeepSeekV4LayerPlan p;
    uint64_t total = 0;
    if (!x || !bytes || coli_v4_layer_plan(&p, c, layer, e, n)) return -1;
    for (size_t i = 0; i < p.tensor_count; i++) {
        const ColiRecordInfo *r = coli_executor_record_by_name(x, p.tensors[i].name);
        ColiTensorInfo t;
        uint64_t resident;
        if (!r || coli_package_tensor_info(coli_executor_package(x), r, &t, e, n))
            return bad(e, n, "COLI planner missing tensor: %s", p.tensors[i].name);
        resident = t.data_stored_bytes;
        if (p.tensors[i].dtype == COLI_ST_F8_E8M0) {
            if (resident > UINT64_MAX / sizeof(float))
                return bad(e, n, "COLI planner scale overflow: %s", p.tensors[i].name);
            resident *= sizeof(float);
        }
        if (UINT64_MAX - total < resident)
            return bad(e, n, "COLI planner byte overflow: %s", p.tensors[i].name);
        total += resident;
    }
    *bytes = total;
    return 0;
}

int coli_v4_coli_tensor_load_f32(ColiExecutor *x, ColiFloatTensor *out,
                                 const char *name, char *e, size_t n) {
    const ColiRecordInfo *r;
    ColiTensorInfo t;
    unsigned char *raw = NULL;
    if (!x || !out || !name)
        return bad(e, n, "invalid COLI float tensor: %s", name ? name : "(null)");
    memset(out, 0, sizeof(*out));
    r = coli_executor_record_by_name(x, name);
    if (!r || (r->math_format != COLI_CSF_MATH_F32 &&
               r->math_format != COLI_CSF_MATH_BF16) ||
        coli_package_tensor_info(coli_executor_package(x), r, &t, e, n) ||
        t.rank > COLI_ST_MAX_RANK || !t.data_stored_bytes ||
        t.data_stored_bytes > SIZE_MAX || t.data_decoded_bytes > SIZE_MAX)
        return bad(e, n, "invalid COLI float tensor: %s", name);

    uint64_t count = 1;
    for (uint16_t i = 0; i < t.rank; i++) {
        if (!t.dims[i] || count > UINT64_MAX / t.dims[i])
            return bad(e, n, "COLI tensor shape overflow: %s", name);
        count *= t.dims[i];
        out->shape[i] = (int64_t)t.dims[i];
    }
    if ((r->math_format == COLI_CSF_MATH_F32 &&
         t.data_stored_bytes != count * sizeof(float)) ||
        (r->math_format == COLI_CSF_MATH_BF16 &&
         t.data_stored_bytes != count * sizeof(uint16_t)) ||
        count > SIZE_MAX / sizeof(float))
        return bad(e, n, "COLI tensor size mismatch: %s", name);

    raw = malloc((size_t)t.data_stored_bytes);
    out->data = malloc((size_t)count * sizeof(float));
    if (!raw || !out->data ||
        coli_package_read_range(coli_executor_package(x), r, t.data_offset,
                                raw, (size_t)t.data_stored_bytes, e, n)) {
        free(raw);
        coli_float_tensor_free(out);
        return -1;
    }
    if (r->math_format == COLI_CSF_MATH_F32) {
        memcpy(out->data, raw, (size_t)t.data_stored_bytes);
    } else {
        for (uint64_t i = 0; i < count; i++) {
            uint16_t v;
            uint32_t bits;
            memcpy(&v, raw + i * 2, 2);
            bits = (uint32_t)v << 16;
            memcpy(out->data + i, &bits, 4);
        }
    }
    free(raw);
    out->count = count;
    out->rank = t.rank;
    return 0;
}

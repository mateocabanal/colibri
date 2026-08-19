#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Engine-lifetime hooks for the process-local and persistent #80 prefix tiers.
 * Remap the existing public open/destroy functions, then layer only cache
 * registration/retirement around current-main runtime behavior.
 *
 * Package-only V4 also binds the named COLITENS compatibility source here.
 * The current exact compiler already preserves ancillary mtp.* tensors by
 * name; exposing them through the legacy tensor API lets the existing DSpark
 * validator/loader consume them without a second model-format implementation.
 */
#define coli_v4_engine_open coli_v4_engine_open_uncached
#define coli_v4_engine_destroy coli_v4_engine_destroy_uncached
#include "deepseek_v4_internal.h"
#undef coli_v4_engine_destroy
#undef coli_v4_engine_open

#include "coli_v4_prefix_cache.h"
#include "coli_v4_prefix_disk.h"
#include "coli_v4_package_tensor_source.h"

#include <stddef.h>

/* Intercept the executor acquire at the exact point engine->coli_static becomes
 * available.  For package-only opens, install a non-owning sentinel index so
 * the existing DSpark control flow reaches its normal profile/shape checks;
 * all named-tensor operations below are redirected to the COLI package.  A
 * hybrid open already has a real safetensors index and is left untouched. */
static int coli_v4_package_executor_open_bridge(
        ColiExecutor **out, const char *package_path,
        const ColiExecutorOpenOptions *options,
        char *error, size_t error_size) {
    int result = coli_executor_open(out, package_path, options,
                                    error, error_size);
    if (result || !out || !*out) return result;
    if (coli_v4_package_source_bind(*out)) {
        coli_executor_close(*out);
        *out = NULL;
        if (error && error_size)
            snprintf(error, error_size,
                     "cannot initialize package named-tensor source");
        return -1;
    }
    ColiV4Engine *engine = (ColiV4Engine *)((char *)out -
        offsetof(ColiV4Engine, coli_static));
    if (!engine->target_index)
        engine->target_index = coli_v4_package_source_sentinel();
    return 0;
}

#define coli_v4_engine_open coli_v4_engine_open_uncached
#define coli_v4_engine_destroy coli_v4_engine_destroy_uncached
#define coli_executor_open coli_v4_package_executor_open_bridge
#define coli_st_find coli_v4_package_source_find
#define coli_st_read_tensor coli_v4_package_source_read_tensor
#define coli_st_tensor_shard coli_v4_package_source_tensor_shard
#define coli_st_read_at coli_v4_package_source_read_at
#define coli_st_read_at_streaming coli_v4_package_source_read_at_streaming
#define st_read_scale_f32 coli_v4_package_read_scale_f32
#define coli_tensor_load_f32 coli_v4_package_tensor_load_f32
#include "deepseek_v4.c"
#undef coli_tensor_load_f32
#undef st_read_scale_f32
#undef coli_st_read_at_streaming
#undef coli_st_read_at
#undef coli_st_tensor_shard
#undef coli_st_read_tensor
#undef coli_st_find
#undef coli_executor_open
#undef coli_v4_engine_destroy
#undef coli_v4_engine_open

int coli_v4_engine_open(ColiV4Engine **engine,
                        const ColiV4EngineOpenOptions *options,
                        char *error, size_t error_size) {
    int result = coli_v4_engine_open_uncached(engine, options, error, error_size);
    if (!result && engine && *engine)
        coli_v4_prefix_disk_register_engine(*engine);
    return result;
}

void coli_v4_engine_destroy(ColiV4Engine *engine) {
    coli_v4_prefix_cache_forget_engine(engine);
    coli_v4_prefix_disk_forget_engine(engine);
    if (engine &&
        g_coli_v4_package_tensor_source.executor == engine->coli_static)
        coli_v4_package_source_reset();
    coli_v4_engine_destroy_uncached(engine);
}

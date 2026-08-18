
#include "prefix_cache_disk.h"

#define qwen_prefix_cache_budget_from_env qpc_ram_budget_from_env
#define qwen_prefix_cache_budget_for_serve qpc_ram_budget_for_serve
#define qwen_prefix_cache_min_tokens_from_env qpc_ram_min_tokens_from_env
#define qwen_prefix_cache_store qpc_ram_store
#define qwen_prefix_cache_restore qpc_ram_restore
#include "qwen_prefix_cache_impl.h"
#undef qwen_prefix_cache_restore
#undef qwen_prefix_cache_store
#undef qwen_prefix_cache_min_tokens_from_env
#undef qwen_prefix_cache_budget_for_serve
#undef qwen_prefix_cache_budget_from_env

#define QWEN_PREFIX_DISK_ENGINE_ID UINT32_C(0x5157454e) /* QWEN */
#define QWEN_PREFIX_DISK_STATE_ABI 2u
#define QWEN_PREFIX_DISK_DEFAULT_GB 8u
#define QWEN_PREFIX_DISK_PAYLOAD_VERSION 1u

typedef struct {
    uint32_t version;
    uint32_t reserved;
    uint64_t gdn_elems;
    uint64_t kv_bytes;
} QwenPrefixDiskPayloadHeader;

static inline int qpc_common_mode_allows_ram(void) {
    return coli_prefix_disk_mode_allows_ram();
}

static inline size_t qwen_prefix_cache_budget_from_env(void) {
    if (!qpc_common_mode_allows_ram()) return 0;
    const char *legacy = getenv("QWEN_PREFIX_CACHE_MB");
    if (legacy && *legacy) return qwen_prefix_cache_budget_parse(legacy, 0);
    return coli_prefix_ram_budget_from_env(0);
}

static inline size_t qwen_prefix_cache_budget_for_serve(void) {
    if (!qpc_common_mode_allows_ram()) return 0;
    const size_t fallback =
        (size_t)QWEN_PREFIX_CACHE_DEFAULT_SERVE_MB * 1024u * 1024u;
    const char *legacy = getenv("QWEN_PREFIX_CACHE_MB");
    if (legacy && *legacy)
        return qwen_prefix_cache_budget_parse(legacy, fallback);

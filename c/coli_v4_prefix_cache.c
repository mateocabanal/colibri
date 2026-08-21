/* Compose the validated process-RAM cache, V4 persistent prefix adapter, and
 * Stage-3 deterministic output store into one already-linked translation unit.
 * Keeping the cache tiers together avoids a second engine-lifetime registry and
 * leaves the normal V4 link graph unchanged. */
#include "coli_v4_prefix_cache_impl.inc"
#include "coli_v4_prefix_disk_shared.inc"

/*
 * Stage-3's output payload reserves its final header word for the completion
 * termination reason. Keep that contract at the persistent-store boundary so
 * the payload CRC covers it and a hit can reject inconsistent metadata before
 * any cached token is re-served. The output-cache implementation itself remains
 * deliberately independent of the generic prefix-disk framing.
 */
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint8_t tokenizer_template_fingerprint[32];
    uint64_t execution_fingerprint;
    uint32_t prompt_count;
    uint32_t generated_count;
    int32_t max_new_tokens;
    int32_t stop_at_sentence;
    int32_t no_dspark;
    uint32_t termination_reason;
} ColiV4OutputContractHeader;

typedef struct {
    int32_t token;
    float logit;
    int32_t position;
    int32_t ordinal;
} ColiV4OutputContractToken;

enum {
    COLI_V4_OUTPUT_TERM_EOS = 1u,
    COLI_V4_OUTPUT_TERM_MAX_NEW = 2u,
    COLI_V4_OUTPUT_TERM_STOP_SENTENCE = 3u,
};

_Static_assert(sizeof(ColiV4OutputContractHeader) == 80,
               "Stage-3 output header ABI changed");
_Static_assert(sizeof(ColiV4OutputContractToken) == 16,
               "Stage-3 output token ABI changed");

static int coli_v4_output_contract_reason(
    const unsigned char *payload, size_t payload_bytes,
    uint32_t *reason_out) {
    static const char magic[8] = {'V','4','O','U','T','0','1','\0'};
    if (!payload || !reason_out ||
        payload_bytes < sizeof(ColiV4OutputContractHeader))
        return 0;

    ColiV4OutputContractHeader header;
    memcpy(&header, payload, sizeof(header));
    if (memcmp(header.magic, magic, sizeof(magic)) ||
        header.version != 1u ||
        header.header_bytes != sizeof(header) ||
        header.prompt_count > (uint32_t)INT_MAX ||
        header.generated_count < 1 ||
        header.generated_count > (uint32_t)INT_MAX ||
        header.max_new_tokens < 1 ||
        header.generated_count > (uint32_t)header.max_new_tokens ||
        header.no_dspark != 1)
        return 0;

    size_t prompt_count = (size_t)header.prompt_count;
    size_t generated_count = (size_t)header.generated_count;
    if (prompt_count > SIZE_MAX / sizeof(int32_t) ||
        generated_count > SIZE_MAX / sizeof(ColiV4OutputContractToken))
        return 0;
    size_t prompt_bytes = prompt_count * sizeof(int32_t);
    size_t token_bytes = generated_count * sizeof(ColiV4OutputContractToken);
    if (sizeof(header) > SIZE_MAX - prompt_bytes ||
        sizeof(header) + prompt_bytes > SIZE_MAX - token_bytes ||
        payload_bytes != sizeof(header) + prompt_bytes + token_bytes)
        return 0;

    size_t last_offset = sizeof(header) + prompt_bytes +
        (generated_count - 1) * sizeof(ColiV4OutputContractToken);
    ColiV4OutputContractToken last;
    memcpy(&last, payload + last_offset, sizeof(last));

    uint32_t reason = 0;
    if (last.token == 1)
        reason = COLI_V4_OUTPUT_TERM_EOS;
    else if (header.generated_count == (uint32_t)header.max_new_tokens)
        reason = COLI_V4_OUTPUT_TERM_MAX_NEW;
    else if (header.stop_at_sentence)
        reason = COLI_V4_OUTPUT_TERM_STOP_SENTENCE;
    else
        return 0;

    *reason_out = reason;
    return 1;
}

static int coli_v4_output_contract_store(
    ColiPrefixDiskCache *cache, const int *key, int key_count,
    const void *payload, size_t payload_bytes) {
    uint32_t reason = 0;
    if (!coli_v4_output_contract_reason(
            (const unsigned char *)payload, payload_bytes, &reason))
        return 0;
    unsigned char *copy = malloc(payload_bytes);
    if (!copy) return 0;
    memcpy(copy, payload, payload_bytes);
    memcpy(copy + offsetof(ColiV4OutputContractHeader, termination_reason),
           &reason, sizeof(reason));
    int stored = coli_prefix_disk_store(cache, key, key_count,
                                        copy, payload_bytes);
    free(copy);
    return stored;
}

static int coli_v4_output_contract_restore(
    ColiPrefixDiskCache *cache, const int *key, int key_count,
    ColiPrefixDiskHit *hit) {
    int matched = coli_prefix_disk_restore_longest(
        cache, key, key_count, hit);
    if (!matched) return 0;

    uint32_t expected = 0, stored = 0;
    int valid = hit &&
        coli_v4_output_contract_reason(hit->payload, hit->payload_bytes,
                                       &expected);
    if (valid)
        memcpy(&stored,
               hit->payload +
                   offsetof(ColiV4OutputContractHeader, termination_reason),
               sizeof(stored));
    if (!valid || stored != expected) {
        if (cache) cache->corruptions++;
        coli_prefix_disk_hit_free(hit);
        return 0;
    }
    return matched;
}

#define coli_prefix_disk_store coli_v4_output_contract_store
#define coli_prefix_disk_restore_longest coli_v4_output_contract_restore
#include "coli_v4_output_cache.c"
#undef coli_prefix_disk_restore_longest
#undef coli_prefix_disk_store

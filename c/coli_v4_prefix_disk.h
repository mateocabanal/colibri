#ifndef COLIBRI_V4_PREFIX_DISK_H
#define COLIBRI_V4_PREFIX_DISK_H

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4.h"

/* One process-wide V4 adapter owns persistent-cache namespaces. Engine open and
 * destroy register/retire exactly once; request paths only look up that shared
 * registration and never create translation-unit-local registries. */
void coli_v4_prefix_disk_register_engine(ColiV4Engine *engine);
void coli_v4_prefix_disk_forget_engine(ColiV4Engine *engine);

typedef struct {
    uint64_t lookups;
    uint64_t hits;
    uint64_t stores;
    uint64_t evictions;
    uint64_t corruptions;
    uint64_t read_bytes;
    uint64_t write_bytes;
    uint64_t resident_bytes;
    size_t budget_bytes;
    size_t min_free_bytes;
    int enabled;
} ColiV4PrefixDiskStats;

void coli_v4_prefix_disk_stats(ColiV4Engine *engine,
                               ColiV4PrefixDiskStats *stats);

/* Internal oracle seam. Production registration always derives a strong model
 * fingerprint from the COLI package. The tiny V4 restart test has no package,
 * so it installs an isolated namespace with an explicit synthetic fingerprint.
 * This is not part of deepseek_v4.h and is never called by runtime code. */
int coli_v4_prefix_disk_register_test_namespace(
    ColiV4Engine *engine, const uint8_t model_fingerprint[32],
    uint64_t layout_fingerprint);

/* Post-generation publication borrows the immutable process-RAM prefix entry,
 * so the normal auto (RAM+SSD) path keeps disk I/O outside TTFT/token streaming. */
void coli_v4_prefix_disk_publish_session(ColiV4Session *session);

/* End-of-prefill fallback used when no RAM entry exists (notably explicit SSD
 * mode or RAM-admission failure). It snapshots/emits one layer at a time while
 * the live state still represents the exact prompt boundary. */
void coli_v4_prefix_disk_publish_live_prefix(ColiV4Session *session);

int coli_v4_prefix_disk_restore(ColiV4Session *session,
                                const int *prompt_ids,
                                int prompt_tokens);

#endif /* COLIBRI_V4_PREFIX_DISK_H */

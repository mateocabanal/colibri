#ifndef COLIBRI_V4_PREFIX_DISK_H
#define COLIBRI_V4_PREFIX_DISK_H

#include "deepseek_v4.h"

/* One process-wide V4 adapter owns persistent-cache namespaces. Engine open and
 * destroy register/retire exactly once; request paths only look up that shared
 * registration and never create translation-unit-local registries. */
void coli_v4_prefix_disk_register_engine(ColiV4Engine *engine);
void coli_v4_prefix_disk_forget_engine(ColiV4Engine *engine);

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

#ifndef COLIBRI_V4_PREFIX_DISK_H
#define COLIBRI_V4_PREFIX_DISK_H

#include <limits.h>
#include <stdint.h>

#include "deepseek_v4.h"

/* Shared #80 SSD adapter. Registration is best-effort and fail-closed: engines
 * whose compiled artifact/runtime identity cannot be fingerprinted simply keep
 * using the process-local RAM cache. */
void coli_v4_prefix_disk_register_engine(ColiV4Engine *engine);
void coli_v4_prefix_disk_forget_engine(ColiV4Engine *engine);

/* Restore the longest strict exact token prefix from SSD into SESSION. Returns
 * the number of restored prompt tokens, or zero on miss/incompatibility/error.
 * A successful restore promotes the live state into the process-local RAM tier
 * when that tier has budget. */
int coli_v4_prefix_disk_restore(ColiV4Session *session,
                                const int *prompt_ids,
                                int prompt_tokens);

/* Publish SESSION's exact end-of-prefill process-cache entry. The entry is
 * ref-pinned and streamed directly into the common atomic/checksummed object
 * writer, so no second full snapshot allocation is needed. */
void coli_v4_prefix_disk_publish_session(ColiV4Session *session);

#endif /* COLIBRI_V4_PREFIX_DISK_H */

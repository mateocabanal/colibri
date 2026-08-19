/* Compose the validated process-RAM cache and the V4 persistent adapter into
 * one already-linked translation unit. The RAM implementation remains a
 * byte-for-byte copy of the previously reviewed source. */
#include "coli_v4_prefix_cache_impl.inc"
#include "coli_v4_prefix_hot_planner.inc"
#include "coli_v4_prefix_disk_shared.inc"

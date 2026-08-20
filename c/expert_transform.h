#ifndef COLIBRI_EXPERT_TRANSFORM_H
#define COLIBRI_EXPERT_TRANSFORM_H

#include "expert_residency.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    COLI_JIT_TRANSFORM_REGISTRY_CAPACITY = 8,
    COLI_JIT_TRANSFORM_ATTEMPT_CAPACITY = 8,
};

typedef enum {
    COLI_JIT_TRANSFORM_EXACT = 1,
    COLI_JIT_TRANSFORM_LOSSY = 2,
} ColiJitTransformClass;

typedef enum {
    COLI_JIT_PRIORITY_BACKGROUND = 0,
    COLI_JIT_PRIORITY_WARMUP = 1,
    COLI_JIT_PRIORITY_BLOCKING = 2,
} ColiJitTransformPriority;

typedef enum {
    COLI_JIT_ATTEMPT_EMPTY = 0,
    COLI_JIT_ATTEMPT_QUEUED = 1,
    COLI_JIT_ATTEMPT_PREPARING = 2,
    COLI_JIT_ATTEMPT_VALIDATING = 3,
    COLI_JIT_ATTEMPT_PUBLISHED = 4,
    COLI_JIT_ATTEMPT_FAILED = 5,
    COLI_JIT_ATTEMPT_CANCELLING = 6,
    COLI_JIT_ATTEMPT_CANCELLED = 7,
} ColiJitTransformAttemptState;

typedef enum {
    COLI_JIT_REQUEST_INVALID = -1,
    COLI_JIT_REQUEST_DISABLED = 0,
    COLI_JIT_REQUEST_OWNER = 1,
    COLI_JIT_REQUEST_JOIN = 2,
    COLI_JIT_REQUEST_READY = 3,
    COLI_JIT_REQUEST_NO_BUDGET = 4,
    COLI_JIT_REQUEST_UNSUPPORTED = 5,
    COLI_JIT_REQUEST_DESTINATION_BUSY = 6,
    COLI_JIT_REQUEST_NO_SLOT = 7,
    COLI_JIT_REQUEST_STALE_SOURCE = 8,
} ColiJitTransformRequestResult;

typedef enum {
    COLI_JIT_ERROR_NONE = 0,
    COLI_JIT_ERROR_STALE_SOURCE = 1,
    COLI_JIT_ERROR_DESTINATION_STALE = 2,
    COLI_JIT_ERROR_ALLOCATION = 3,
    COLI_JIT_ERROR_PREPARE = 4,
    COLI_JIT_ERROR_VALIDATE = 5,
    COLI_JIT_ERROR_PUBLISH = 6,
    COLI_JIT_ERROR_CANCELLED = 7,
} ColiJitTransformError;

typedef enum {
    COLI_JIT_MEMORY_OUTPUT = 1,
    COLI_JIT_MEMORY_SCRATCH = 2,
    COLI_JIT_MEMORY_STAGING = 3,
} ColiJitMemoryPurpose;

typedef struct {
    uint64_t resident_bytes;
    uint64_t allocation_bytes;
    uint64_t scratch_bytes;
    uint64_t staging_bytes;
    uint64_t output_alignment;
    uint64_t scratch_alignment;
    uint64_t staging_alignment;
} ColiJitTransformEstimate;

typedef int (*ColiJitTransformEstimateFn)(
    void *context,
    const ColiExpertResidentView *source,
    const ColiRepresentationId *target,
    ColiJitTransformEstimate *estimate);

typedef int (*ColiJitTransformPrepareFn)(
    void *context,
    const ColiExpertResidentView *source,
    const ColiRepresentationId *target,
    void *output,
    uint64_t output_bytes,
    void *scratch,
    uint64_t scratch_bytes,
    void *staging,
    uint64_t staging_bytes);

typedef int (*ColiJitTransformValidateFn)(
    void *context,
    const ColiExpertResidentView *source,
    const ColiRepresentationId *target,
    const void *output,
    uint64_t resident_bytes);

typedef struct {
    ColiRepresentationId source;
    ColiRepresentationId target;
    uint32_t transform_abi;
    ColiJitTransformClass transform_class;
    unsigned target_tier_mask;
    uint32_t backend_tag;
    void *context;
    ColiJitTransformEstimateFn estimate;
    ColiJitTransformPrepareFn prepare;
    ColiJitTransformValidateFn validate;
} ColiRepresentationTransformOps;

typedef struct {
    atomic_int lock;
    uint32_t count;
    ColiRepresentationTransformOps ops[COLI_JIT_TRANSFORM_REGISTRY_CAPACITY];
} ColiRepresentationTransformRegistry;

typedef struct {
    uint64_t capacity_bytes;
    atomic_uint_fast64_t committed_bytes;
    atomic_uint_fast64_t scratch_bytes;
    atomic_uint_fast64_t staging_bytes;
    atomic_uint_fast64_t peak_committed_bytes;
} ColiJitTransformTempBudget;

typedef void *(*ColiJitMemoryAllocateFn)(
    void *context,
    ColiJitMemoryPurpose purpose,
    uint64_t bytes,
    uint64_t alignment);

typedef void (*ColiJitMemoryFreeFn)(
    void *context,
    ColiJitMemoryPurpose purpose,
    void *memory,
    uint64_t bytes);

typedef struct {
    void *context;
    ColiJitMemoryAllocateFn allocate;
    ColiJitMemoryFreeFn free;
} ColiJitTransformMemoryOps;

typedef uint64_t (*ColiJitClockNowNsFn)(void *context);

typedef struct {
    void *context;
    ColiJitClockNowNsFn now_ns;
} ColiJitTransformClockOps;

typedef struct {
    ColiExpertResidencyEntry *entry;
    ColiExpertKey key;
    uint32_t source_variant_id;
    uint64_t source_generation;
    ColiRepresentationId target;
    uint32_t transform_abi;
} ColiJitTransformRequestKey;

typedef struct {
    uint32_t slot;
    uint64_t serial;
} ColiJitTransformTicket;

typedef struct {
    ColiJitTransformAttemptState state;
    ColiJitTransformError error;
    ColiJitTransformPriority priority;
    ColiJitTransformRequestKey key;
    uint32_t destination_variant_id;
    uint64_t destination_generation;
    uint64_t resident_bytes;
    uint64_t allocation_bytes;
    uint64_t scratch_bytes;
    uint64_t staging_bytes;
    uint32_t backend_tag;
    ColiJitTransformClass transform_class;
} ColiJitTransformAttemptInfo;

typedef struct {
    uint64_t requested;
    uint64_t owner;
    uint64_t join;
    uint64_t started;
    uint64_t completed;
    uint64_t failed;
    uint64_t cancelled;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t scratch_bytes;
    uint64_t staging_bytes;
    uint64_t queue_ns;
    uint64_t prepare_ns;
    uint64_t validate_ns;
    uint64_t exact;
    uint64_t lossy;
    uint64_t reject_no_budget;
    uint64_t reject_unsupported;
    uint32_t last_backend_tag;
} ColiJitTransformTelemetrySnapshot;

typedef struct {
    uint64_t serial;
    ColiJitTransformAttemptState state;
    ColiJitTransformError error;
    ColiJitTransformPriority priority;
    ColiJitTransformRequestKey key;
    ColiRepresentationTransformOps ops;
    ColiExpertResidencyLease source_lease;
    ColiExpertResidencyBudget *destination_budget;
    uint32_t destination_variant_id;
    uint64_t destination_generation;
    ColiJitTransformEstimate estimate;
    int temp_reserved;
    void *output;
    void *scratch;
    void *staging;
    uint64_t queued_ns;
} ColiJitTransformAttempt;

typedef struct {
    atomic_uint_fast64_t requested;
    atomic_uint_fast64_t owner;
    atomic_uint_fast64_t join;
    atomic_uint_fast64_t started;
    atomic_uint_fast64_t completed;
    atomic_uint_fast64_t failed;
    atomic_uint_fast64_t cancelled;
    atomic_uint_fast64_t input_bytes;
    atomic_uint_fast64_t output_bytes;
    atomic_uint_fast64_t scratch_bytes;
    atomic_uint_fast64_t staging_bytes;
    atomic_uint_fast64_t queue_ns;
    atomic_uint_fast64_t prepare_ns;
    atomic_uint_fast64_t validate_ns;
    atomic_uint_fast64_t exact;
    atomic_uint_fast64_t lossy;
    atomic_uint_fast64_t reject_no_budget;
    atomic_uint_fast64_t reject_unsupported;
    atomic_uint last_backend_tag;
} ColiJitTransformTelemetry;

typedef struct {
    ColiRepresentationTransformRegistry *registry;
    ColiJitTransformTempBudget *temp_budget;
    ColiJitTransformMemoryOps memory;
    ColiJitTransformClockOps clock;
    atomic_int lock;
    atomic_uint_fast64_t serial_allocator;
    int enabled;
    ColiJitTransformAttempt attempts[COLI_JIT_TRANSFORM_ATTEMPT_CAPACITY];
    ColiJitTransformTelemetry telemetry;
} ColiJitTransformService;

void coli_jit_transform_registry_init(ColiRepresentationTransformRegistry *registry);
int coli_jit_transform_registry_register(
    ColiRepresentationTransformRegistry *registry,
    const ColiRepresentationTransformOps *ops);
int coli_jit_transform_registry_find(
    ColiRepresentationTransformRegistry *registry,
    const ColiRepresentationId *source,
    const ColiRepresentationId *target,
    uint32_t transform_abi,
    ColiRepresentationTransformOps *ops_out);

void coli_jit_transform_temp_budget_init(
    ColiJitTransformTempBudget *budget, uint64_t capacity_bytes);
uint64_t coli_jit_transform_temp_budget_committed(
    const ColiJitTransformTempBudget *budget);

int coli_jit_transform_service_init(
    ColiJitTransformService *service,
    ColiRepresentationTransformRegistry *registry,
    ColiJitTransformTempBudget *temp_budget,
    const ColiJitTransformMemoryOps *memory,
    const ColiJitTransformClockOps *clock);
void coli_jit_transform_service_set_enabled(
    ColiJitTransformService *service, int enabled);

ColiJitTransformRequestResult coli_jit_transform_request(
    ColiJitTransformService *service,
    ColiExpertResidencyEntry *entry,
    uint32_t source_variant_id,
    uint64_t source_generation,
    ColiExpertResidencyBudget *destination_budget,
    const ColiRepresentationId *target,
    uint32_t transform_abi,
    ColiJitTransformPriority priority,
    ColiJitTransformTicket *ticket_out);

/* Executes at most one queued attempt. It is intentionally caller-driven:
 * inference/request threads enqueue without performing the transform; a runtime
 * worker invokes this function from its own execution context. */
int coli_jit_transform_run_one(ColiJitTransformService *service);

int coli_jit_transform_cancel(
    ColiJitTransformService *service, ColiJitTransformTicket ticket);
int coli_jit_transform_query(
    ColiJitTransformService *service,
    ColiJitTransformTicket ticket,
    ColiJitTransformAttemptInfo *info);
void coli_jit_transform_telemetry_snapshot(
    const ColiJitTransformService *service,
    ColiJitTransformTelemetrySnapshot *snapshot);

#ifdef __cplusplus
}
#endif

#endif

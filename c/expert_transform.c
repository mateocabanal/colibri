#include "expert_transform.h"

#include <string.h>

static void coli_jit_spin_lock(atomic_int *lock) {
    int expected = 0;
    while (!atomic_compare_exchange_weak_explicit(
            lock, &expected, 1, memory_order_acq_rel, memory_order_acquire))
        expected = 0;
}

static void coli_jit_spin_unlock(atomic_int *lock) {
    atomic_store_explicit(lock, 0, memory_order_release);
}

static uint64_t coli_jit_now_ns(const ColiJitTransformService *service) {
    return service && service->clock.now_ns
        ? service->clock.now_ns(service->clock.context) : 0;
}

static void coli_jit_update_peak(
        atomic_uint_fast64_t *peak_value, uint64_t candidate) {
    uint64_t peak = atomic_load_explicit(peak_value, memory_order_acquire);
    while (candidate > peak &&
           !atomic_compare_exchange_weak_explicit(
               peak_value, &peak, candidate,
               memory_order_acq_rel, memory_order_acquire)) {}
}

static int coli_jit_atomic_sub(
        atomic_uint_fast64_t *value, uint64_t amount) {
    if (!amount) return 0;
    uint64_t current = atomic_load_explicit(value, memory_order_acquire);
    for (;;) {
        if (current < amount) return -1;
        if (atomic_compare_exchange_weak_explicit(
                value, &current, current - amount,
                memory_order_acq_rel, memory_order_acquire))
            return 0;
    }
}

static int coli_jit_alignment_valid(uint64_t alignment) {
    return alignment == 0 || (alignment & (alignment - 1)) == 0;
}

static uint64_t coli_jit_alignment_or_one(uint64_t alignment) {
    return alignment ? alignment : 1;
}

void coli_jit_transform_registry_init(
        ColiRepresentationTransformRegistry *registry) {
    if (!registry) return;
    memset(registry, 0, sizeof(*registry));
    atomic_init(&registry->lock, 0);
}

int coli_jit_transform_registry_register(
        ColiRepresentationTransformRegistry *registry,
        const ColiRepresentationTransformOps *ops) {
    if (!registry || !ops || !ops->transform_abi || !ops->target_tier_mask ||
        !ops->estimate || !ops->prepare || !ops->validate ||
        !coli_representation_known(&ops->source) ||
        !coli_representation_known(&ops->target) ||
        coli_representation_equal(&ops->source, &ops->target))
        return -1;

    if (ops->transform_class == COLI_JIT_TRANSFORM_EXACT) {
        if (!coli_representation_exact_math_compatible(
                &ops->source, &ops->target))
            return -1;
    } else if (ops->transform_class == COLI_JIT_TRANSFORM_LOSSY) {
        /* Lossy execution is deliberately gated until a quality/profile
         * identity is part of the request contract. #135 proves exact repacks. */
        return -2;
    } else {
        return -1;
    }

    coli_jit_spin_lock(&registry->lock);
    for (uint32_t i = 0; i < registry->count; ++i) {
        const ColiRepresentationTransformOps *existing = &registry->ops[i];
        if (existing->transform_abi == ops->transform_abi &&
            coli_representation_equal(&existing->source, &ops->source) &&
            coli_representation_equal(&existing->target, &ops->target)) {
            coli_jit_spin_unlock(&registry->lock);
            return 0;
        }
    }
    if (registry->count >= COLI_JIT_TRANSFORM_REGISTRY_CAPACITY) {
        coli_jit_spin_unlock(&registry->lock);
        return -1;
    }
    registry->ops[registry->count++] = *ops;
    coli_jit_spin_unlock(&registry->lock);
    return 1;
}

int coli_jit_transform_registry_find(
        ColiRepresentationTransformRegistry *registry,
        const ColiRepresentationId *source,
        const ColiRepresentationId *target,
        uint32_t transform_abi,
        ColiRepresentationTransformOps *ops_out) {
    if (!registry || !source || !target || !transform_abi || !ops_out)
        return -1;
    coli_jit_spin_lock(&registry->lock);
    for (uint32_t i = 0; i < registry->count; ++i) {
        const ColiRepresentationTransformOps *candidate = &registry->ops[i];
        if (candidate->transform_abi == transform_abi &&
            coli_representation_equal(&candidate->source, source) &&
            coli_representation_equal(&candidate->target, target)) {
            *ops_out = *candidate;
            coli_jit_spin_unlock(&registry->lock);
            return 1;
        }
    }
    coli_jit_spin_unlock(&registry->lock);
    return 0;
}

void coli_jit_transform_temp_budget_init(
        ColiJitTransformTempBudget *budget, uint64_t capacity_bytes) {
    if (!budget) return;
    memset(budget, 0, sizeof(*budget));
    budget->capacity_bytes = capacity_bytes;
    atomic_init(&budget->committed_bytes, 0);
    atomic_init(&budget->scratch_bytes, 0);
    atomic_init(&budget->staging_bytes, 0);
    atomic_init(&budget->peak_committed_bytes, 0);
}

uint64_t coli_jit_transform_temp_budget_committed(
        const ColiJitTransformTempBudget *budget) {
    return budget ? atomic_load_explicit(
        &budget->committed_bytes, memory_order_acquire) : 0;
}

static int coli_jit_temp_budget_reserve(
        ColiJitTransformTempBudget *budget,
        uint64_t scratch_bytes,
        uint64_t staging_bytes) {
    if (!budget || scratch_bytes > UINT64_MAX - staging_bytes) return -1;
    uint64_t total = scratch_bytes + staging_bytes;
    if (!total) return 1;
    if (total > budget->capacity_bytes) return 0;

    uint64_t committed = atomic_load_explicit(
        &budget->committed_bytes, memory_order_acquire);
    for (;;) {
        if (committed > budget->capacity_bytes ||
            total > budget->capacity_bytes - committed)
            return 0;
        uint64_t desired = committed + total;
        if (atomic_compare_exchange_weak_explicit(
                &budget->committed_bytes, &committed, desired,
                memory_order_acq_rel, memory_order_acquire)) {
            if (scratch_bytes)
                atomic_fetch_add_explicit(
                    &budget->scratch_bytes, scratch_bytes,
                    memory_order_acq_rel);
            if (staging_bytes)
                atomic_fetch_add_explicit(
                    &budget->staging_bytes, staging_bytes,
                    memory_order_acq_rel);
            coli_jit_update_peak(&budget->peak_committed_bytes, desired);
            return 1;
        }
    }
}

static int coli_jit_temp_budget_release(
        ColiJitTransformTempBudget *budget,
        uint64_t scratch_bytes,
        uint64_t staging_bytes) {
    if (!budget || scratch_bytes > UINT64_MAX - staging_bytes) return -1;
    uint64_t total = scratch_bytes + staging_bytes;
    if (scratch_bytes &&
        coli_jit_atomic_sub(&budget->scratch_bytes, scratch_bytes) != 0)
        return -1;
    if (staging_bytes &&
        coli_jit_atomic_sub(&budget->staging_bytes, staging_bytes) != 0)
        return -1;
    return coli_jit_atomic_sub(&budget->committed_bytes, total);
}

static void coli_jit_telemetry_init(ColiJitTransformTelemetry *telemetry) {
    atomic_init(&telemetry->requested, 0);
    atomic_init(&telemetry->owner, 0);
    atomic_init(&telemetry->join, 0);
    atomic_init(&telemetry->started, 0);
    atomic_init(&telemetry->completed, 0);
    atomic_init(&telemetry->failed, 0);
    atomic_init(&telemetry->cancelled, 0);
    atomic_init(&telemetry->input_bytes, 0);
    atomic_init(&telemetry->output_bytes, 0);
    atomic_init(&telemetry->scratch_bytes, 0);
    atomic_init(&telemetry->staging_bytes, 0);
    atomic_init(&telemetry->queue_ns, 0);
    atomic_init(&telemetry->prepare_ns, 0);
    atomic_init(&telemetry->validate_ns, 0);
    atomic_init(&telemetry->exact, 0);
    atomic_init(&telemetry->lossy, 0);
    atomic_init(&telemetry->reject_no_budget, 0);
    atomic_init(&telemetry->reject_unsupported, 0);
    atomic_init(&telemetry->last_backend_tag, 0);
}

int coli_jit_transform_service_init(
        ColiJitTransformService *service,
        ColiRepresentationTransformRegistry *registry,
        ColiJitTransformTempBudget *temp_budget,
        const ColiJitTransformMemoryOps *memory,
        const ColiJitTransformClockOps *clock) {
    if (!service || !registry || !temp_budget || !memory ||
        !memory->allocate || !memory->free)
        return -1;
    memset(service, 0, sizeof(*service));
    service->registry = registry;
    service->temp_budget = temp_budget;
    service->memory = *memory;
    if (clock) service->clock = *clock;
    atomic_init(&service->lock, 0);
    atomic_init(&service->serial_allocator, 0);
    service->enabled = 1;
    for (uint32_t i = 0; i < COLI_JIT_TRANSFORM_ATTEMPT_CAPACITY; ++i)
        service->attempts[i].state = COLI_JIT_ATTEMPT_EMPTY;
    coli_jit_telemetry_init(&service->telemetry);
    return 0;
}

void coli_jit_transform_service_set_enabled(
        ColiJitTransformService *service, int enabled) {
    if (!service) return;
    coli_jit_spin_lock(&service->lock);
    service->enabled = enabled ? 1 : 0;
    coli_jit_spin_unlock(&service->lock);
}

static int coli_jit_request_key_equal(
        const ColiJitTransformRequestKey *a,
        const ColiJitTransformRequestKey *b) {
    return a && b &&
        a->entry == b->entry &&
        coli_expert_key_equal(a->key, b->key) &&
        a->source_variant_id == b->source_variant_id &&
        a->source_generation == b->source_generation &&
        a->transform_abi == b->transform_abi &&
        coli_representation_equal(&a->target, &b->target);
}

static int coli_jit_attempt_active(const ColiJitTransformAttempt *attempt) {
    return attempt &&
        (attempt->state == COLI_JIT_ATTEMPT_QUEUED ||
         attempt->state == COLI_JIT_ATTEMPT_PREPARING ||
         attempt->state == COLI_JIT_ATTEMPT_VALIDATING);
}

static int coli_jit_attempt_reusable(const ColiJitTransformAttempt *attempt) {
    return attempt &&
        (attempt->state == COLI_JIT_ATTEMPT_EMPTY ||
         attempt->state == COLI_JIT_ATTEMPT_PUBLISHED ||
         attempt->state == COLI_JIT_ATTEMPT_FAILED ||
         attempt->state == COLI_JIT_ATTEMPT_CANCELLED);
}

static int coli_jit_ticket_equal(
        const ColiJitTransformAttempt *attempt,
        uint32_t slot,
        ColiJitTransformTicket ticket) {
    return attempt && ticket.slot == slot && ticket.serial &&
        attempt->serial == ticket.serial &&
        attempt->state != COLI_JIT_ATTEMPT_EMPTY;
}

static int coli_jit_estimate_valid(const ColiJitTransformEstimate *estimate) {
    return estimate && estimate->resident_bytes &&
        estimate->allocation_bytes >= estimate->resident_bytes &&
        coli_jit_alignment_valid(estimate->output_alignment) &&
        coli_jit_alignment_valid(estimate->scratch_alignment) &&
        coli_jit_alignment_valid(estimate->staging_alignment);
}

ColiJitTransformRequestResult coli_jit_transform_request(
        ColiJitTransformService *service,
        ColiExpertResidencyEntry *entry,
        uint32_t source_variant_id,
        uint64_t source_generation,
        ColiExpertResidencyBudget *destination_budget,
        const ColiRepresentationId *target,
        uint32_t transform_abi,
        ColiJitTransformPriority priority,
        ColiJitTransformTicket *ticket_out) {
    if (ticket_out) {
        ticket_out->slot = UINT32_MAX;
        ticket_out->serial = 0;
    }
    if (!service || !entry || !destination_budget || !target || !ticket_out ||
        source_variant_id >= COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY ||
        !source_generation || !transform_abi ||
        !coli_representation_known(target) ||
        priority < COLI_JIT_PRIORITY_BACKGROUND ||
        priority > COLI_JIT_PRIORITY_BLOCKING)
        return COLI_JIT_REQUEST_INVALID;

    atomic_fetch_add_explicit(
        &service->telemetry.requested, 1, memory_order_acq_rel);

    ColiJitTransformRequestKey key = {
        .entry = entry,
        .key = entry->key,
        .source_variant_id = source_variant_id,
        .source_generation = source_generation,
        .target = *target,
        .transform_abi = transform_abi,
    };

    coli_jit_spin_lock(&service->lock);
    if (!service->enabled) {
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_DISABLED;
    }

    /* Work dedup is intentionally independent of destination-residency dedup.
     * Only this exact source variant/generation + target + ABI may join. */
    for (uint32_t i = 0; i < COLI_JIT_TRANSFORM_ATTEMPT_CAPACITY; ++i) {
        ColiJitTransformAttempt *attempt = &service->attempts[i];
        if (coli_jit_attempt_active(attempt) &&
            coli_jit_request_key_equal(&attempt->key, &key)) {
            ticket_out->slot = i;
            ticket_out->serial = attempt->serial;
            atomic_fetch_add_explicit(
                &service->telemetry.join, 1, memory_order_acq_rel);
            coli_jit_spin_unlock(&service->lock);
            return COLI_JIT_REQUEST_JOIN;
        }
    }

    uint32_t free_slot = UINT32_MAX;
    for (uint32_t i = 0; i < COLI_JIT_TRANSFORM_ATTEMPT_CAPACITY; ++i) {
        if (coli_jit_attempt_reusable(&service->attempts[i])) {
            free_slot = i;
            break;
        }
    }
    if (free_slot == UINT32_MAX) {
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_NO_SLOT;
    }

    ColiExpertResidencyLease source_lease;
    int acquired = coli_expert_residency_acquire_variant(
        entry, source_variant_id, &source_lease);
    if (acquired <= 0) {
        coli_jit_spin_unlock(&service->lock);
        return acquired < 0 ? COLI_JIT_REQUEST_INVALID
                            : COLI_JIT_REQUEST_STALE_SOURCE;
    }
    if (source_lease.generation != source_generation ||
        !coli_representation_known(&source_lease.representation)) {
        (void)coli_expert_residency_release(&source_lease);
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_STALE_SOURCE;
    }

    ColiRepresentationTransformOps ops;
    int found = coli_jit_transform_registry_find(
        service->registry, &source_lease.representation,
        target, transform_abi, &ops);
    if (found <= 0) {
        (void)coli_expert_residency_release(&source_lease);
        atomic_fetch_add_explicit(
            &service->telemetry.reject_unsupported, 1, memory_order_acq_rel);
        coli_jit_spin_unlock(&service->lock);
        return found < 0 ? COLI_JIT_REQUEST_INVALID
                         : COLI_JIT_REQUEST_UNSUPPORTED;
    }

    ColiExpertResidentView source_view;
    ColiJitTransformEstimate estimate;
    memset(&estimate, 0, sizeof(estimate));
    if (coli_expert_residency_lease_view(&source_lease, &source_view) != 0 ||
        ops.estimate(ops.context, &source_view, target, &estimate) != 0 ||
        !coli_jit_estimate_valid(&estimate)) {
        (void)coli_expert_residency_release(&source_lease);
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_INVALID;
    }

    uint32_t destination_variant_id = COLI_EXPERT_VARIANT_NONE;
    uint64_t destination_generation = 0;
    ColiExpertRequestResult reserved = coli_expert_residency_reserve_variant(
        entry, destination_budget, target, ops.target_tier_mask,
        estimate.allocation_bytes, &destination_variant_id,
        &destination_generation);
    if (reserved == COLI_EXPERT_REQUEST_HIT) {
        (void)coli_expert_residency_release(&source_lease);
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_READY;
    }
    if (reserved == COLI_EXPERT_REQUEST_JOIN_INFLIGHT) {
        /* An in-flight destination with no matching transform key may have a
         * different source generation/ABI or belong to another mechanism. */
        (void)coli_expert_residency_release(&source_lease);
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_DESTINATION_BUSY;
    }
    if (reserved == COLI_EXPERT_REQUEST_NO_BUDGET) {
        (void)coli_expert_residency_release(&source_lease);
        atomic_fetch_add_explicit(
            &service->telemetry.reject_no_budget, 1, memory_order_acq_rel);
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_NO_BUDGET;
    }
    if (reserved == COLI_EXPERT_REQUEST_NO_SLOT) {
        (void)coli_expert_residency_release(&source_lease);
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_NO_SLOT;
    }
    if (reserved != COLI_EXPERT_REQUEST_LOAD_OWNER) {
        (void)coli_expert_residency_release(&source_lease);
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_INVALID;
    }

    int temp_reserved = coli_jit_temp_budget_reserve(
        service->temp_budget, estimate.scratch_bytes, estimate.staging_bytes);
    if (temp_reserved <= 0) {
        (void)coli_expert_residency_fail_variant(
            entry, destination_budget, destination_variant_id,
            destination_generation);
        (void)coli_expert_residency_release(&source_lease);
        if (temp_reserved == 0)
            atomic_fetch_add_explicit(
                &service->telemetry.reject_no_budget, 1,
                memory_order_acq_rel);
        coli_jit_spin_unlock(&service->lock);
        return temp_reserved < 0 ? COLI_JIT_REQUEST_INVALID
                                 : COLI_JIT_REQUEST_NO_BUDGET;
    }

    uint64_t serial = atomic_fetch_add_explicit(
        &service->serial_allocator, 1, memory_order_acq_rel) + 1;
    if (!serial) {
        (void)coli_jit_temp_budget_release(
            service->temp_budget, estimate.scratch_bytes,
            estimate.staging_bytes);
        (void)coli_expert_residency_fail_variant(
            entry, destination_budget, destination_variant_id,
            destination_generation);
        (void)coli_expert_residency_release(&source_lease);
        coli_jit_spin_unlock(&service->lock);
        return COLI_JIT_REQUEST_INVALID;
    }

    ColiJitTransformAttempt *attempt = &service->attempts[free_slot];
    memset(attempt, 0, sizeof(*attempt));
    attempt->serial = serial;
    attempt->state = COLI_JIT_ATTEMPT_QUEUED;
    attempt->error = COLI_JIT_ERROR_NONE;
    attempt->priority = priority;
    attempt->key = key;
    attempt->ops = ops;
    attempt->source_lease = source_lease;
    attempt->destination_budget = destination_budget;
    attempt->destination_variant_id = destination_variant_id;
    attempt->destination_generation = destination_generation;
    attempt->estimate = estimate;
    attempt->temp_reserved = 1;
    attempt->queued_ns = coli_jit_now_ns(service);

    ticket_out->slot = free_slot;
    ticket_out->serial = serial;
    atomic_fetch_add_explicit(
        &service->telemetry.owner, 1, memory_order_acq_rel);
    atomic_fetch_add_explicit(
        &service->telemetry.input_bytes, source_lease.resident_bytes,
        memory_order_acq_rel);
    atomic_fetch_add_explicit(
        &service->telemetry.output_bytes, estimate.resident_bytes,
        memory_order_acq_rel);
    atomic_fetch_add_explicit(
        &service->telemetry.scratch_bytes, estimate.scratch_bytes,
        memory_order_acq_rel);
    atomic_fetch_add_explicit(
        &service->telemetry.staging_bytes, estimate.staging_bytes,
        memory_order_acq_rel);
    if (ops.transform_class == COLI_JIT_TRANSFORM_EXACT)
        atomic_fetch_add_explicit(
            &service->telemetry.exact, 1, memory_order_acq_rel);
    else
        atomic_fetch_add_explicit(
            &service->telemetry.lossy, 1, memory_order_acq_rel);

    coli_jit_spin_unlock(&service->lock);
    return COLI_JIT_REQUEST_OWNER;
}

static int coli_jit_destination_owned(
        const ColiJitTransformAttempt *attempt) {
    if (!attempt || !attempt->key.entry ||
        attempt->destination_variant_id >=
            COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY)
        return 0;
    const ColiExpertResidentVariant *variant =
        coli_expert_residency_variant_const(
            attempt->key.entry, attempt->destination_variant_id);
    if (!variant) return 0;
    int state = atomic_load_explicit(&variant->state, memory_order_acquire);
    return (state == COLI_EXPERT_RESIDENCY_RESERVED ||
            state == COLI_EXPERT_RESIDENCY_LOADING ||
            state == COLI_EXPERT_RESIDENCY_PREPARING) &&
        atomic_load_explicit(
            &variant->pending_generation, memory_order_acquire) ==
            attempt->destination_generation;
}

static void coli_jit_free_memory(
        ColiJitTransformService *service,
        ColiJitTransformAttempt *attempt) {
    if (attempt->output) {
        service->memory.free(
            service->memory.context, COLI_JIT_MEMORY_OUTPUT,
            attempt->output, attempt->estimate.allocation_bytes);
        attempt->output = NULL;
    }
    if (attempt->scratch) {
        service->memory.free(
            service->memory.context, COLI_JIT_MEMORY_SCRATCH,
            attempt->scratch, attempt->estimate.scratch_bytes);
        attempt->scratch = NULL;
    }
    if (attempt->staging) {
        service->memory.free(
            service->memory.context, COLI_JIT_MEMORY_STAGING,
            attempt->staging, attempt->estimate.staging_bytes);
        attempt->staging = NULL;
    }
}

static void coli_jit_release_temp_reservation(
        ColiJitTransformService *service,
        ColiJitTransformAttempt *attempt) {
    if (!attempt->temp_reserved) return;
    (void)coli_jit_temp_budget_release(
        service->temp_budget, attempt->estimate.scratch_bytes,
        attempt->estimate.staging_bytes);
    attempt->temp_reserved = 0;
}

static void coli_jit_cleanup_unpublished(
        ColiJitTransformService *service,
        ColiJitTransformAttempt *attempt) {
    coli_jit_free_memory(service, attempt);
    coli_jit_release_temp_reservation(service, attempt);
    if (coli_jit_destination_owned(attempt)) {
        (void)coli_expert_residency_fail_variant(
            attempt->key.entry, attempt->destination_budget,
            attempt->destination_variant_id,
            attempt->destination_generation);
    }
    if (attempt->source_lease.entry)
        (void)coli_expert_residency_release(&attempt->source_lease);
}

static int coli_jit_fail_attempt(
        ColiJitTransformService *service,
        ColiJitTransformAttempt *attempt,
        ColiJitTransformError error) {
    coli_jit_cleanup_unpublished(service, attempt);
    coli_jit_spin_lock(&service->lock);
    attempt->error = error;
    attempt->state = COLI_JIT_ATTEMPT_FAILED;
    atomic_fetch_add_explicit(
        &service->telemetry.failed, 1, memory_order_acq_rel);
    coli_jit_spin_unlock(&service->lock);
    return -1;
}

static void *coli_jit_allocate(
        ColiJitTransformService *service,
        ColiJitMemoryPurpose purpose,
        uint64_t bytes,
        uint64_t alignment) {
    if (!bytes) return NULL;
    return service->memory.allocate(
        service->memory.context, purpose, bytes,
        coli_jit_alignment_or_one(alignment));
}

int coli_jit_transform_run_one(ColiJitTransformService *service) {
    if (!service) return -1;

    coli_jit_spin_lock(&service->lock);
    uint32_t selected = UINT32_MAX;
    for (uint32_t i = 0; i < COLI_JIT_TRANSFORM_ATTEMPT_CAPACITY; ++i) {
        ColiJitTransformAttempt *candidate = &service->attempts[i];
        if (candidate->state != COLI_JIT_ATTEMPT_QUEUED) continue;
        if (selected == UINT32_MAX ||
            candidate->priority > service->attempts[selected].priority ||
            (candidate->priority == service->attempts[selected].priority &&
             candidate->serial < service->attempts[selected].serial))
            selected = i;
    }
    if (selected == UINT32_MAX) {
        coli_jit_spin_unlock(&service->lock);
        return 0;
    }

    ColiJitTransformAttempt *attempt = &service->attempts[selected];
    attempt->state = COLI_JIT_ATTEMPT_PREPARING;
    atomic_fetch_add_explicit(
        &service->telemetry.started, 1, memory_order_acq_rel);
    uint64_t now = coli_jit_now_ns(service);
    if (now && attempt->queued_ns && now >= attempt->queued_ns)
        atomic_fetch_add_explicit(
            &service->telemetry.queue_ns, now - attempt->queued_ns,
            memory_order_acq_rel);
    coli_jit_spin_unlock(&service->lock);

    if (!coli_expert_residency_lease_valid(&attempt->source_lease))
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_STALE_SOURCE);

    if (coli_expert_residency_mark_variant_preparing(
            attempt->key.entry, attempt->destination_variant_id,
            attempt->destination_generation) != 0)
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_DESTINATION_STALE);

    attempt->output = coli_jit_allocate(
        service, COLI_JIT_MEMORY_OUTPUT,
        attempt->estimate.allocation_bytes,
        attempt->estimate.output_alignment);
    if (!attempt->output)
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_ALLOCATION);

    if (attempt->estimate.scratch_bytes) {
        attempt->scratch = coli_jit_allocate(
            service, COLI_JIT_MEMORY_SCRATCH,
            attempt->estimate.scratch_bytes,
            attempt->estimate.scratch_alignment);
        if (!attempt->scratch)
            return coli_jit_fail_attempt(
                service, attempt, COLI_JIT_ERROR_ALLOCATION);
    }
    if (attempt->estimate.staging_bytes) {
        attempt->staging = coli_jit_allocate(
            service, COLI_JIT_MEMORY_STAGING,
            attempt->estimate.staging_bytes,
            attempt->estimate.staging_alignment);
        if (!attempt->staging)
            return coli_jit_fail_attempt(
                service, attempt, COLI_JIT_ERROR_ALLOCATION);
    }

    ColiExpertResidentView source_view;
    if (coli_expert_residency_lease_view(
            &attempt->source_lease, &source_view) != 0)
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_STALE_SOURCE);

    uint64_t started = coli_jit_now_ns(service);
    int prepared = attempt->ops.prepare(
        attempt->ops.context, &source_view, &attempt->key.target,
        attempt->output, attempt->estimate.allocation_bytes,
        attempt->scratch, attempt->estimate.scratch_bytes,
        attempt->staging, attempt->estimate.staging_bytes);
    uint64_t finished = coli_jit_now_ns(service);
    if (started && finished >= started)
        atomic_fetch_add_explicit(
            &service->telemetry.prepare_ns, finished - started,
            memory_order_acq_rel);
    if (prepared != 0)
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_PREPARE);

    if (!coli_expert_residency_lease_valid(&attempt->source_lease))
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_STALE_SOURCE);
    if (!coli_jit_destination_owned(attempt))
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_DESTINATION_STALE);

    coli_jit_spin_lock(&service->lock);
    attempt->state = COLI_JIT_ATTEMPT_VALIDATING;
    coli_jit_spin_unlock(&service->lock);

    started = coli_jit_now_ns(service);
    int valid = attempt->ops.validate(
        attempt->ops.context, &source_view, &attempt->key.target,
        attempt->output, attempt->estimate.resident_bytes);
    finished = coli_jit_now_ns(service);
    if (started && finished >= started)
        atomic_fetch_add_explicit(
            &service->telemetry.validate_ns, finished - started,
            memory_order_acq_rel);
    if (valid != 0)
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_VALIDATE);

    if (!coli_expert_residency_lease_valid(&attempt->source_lease))
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_STALE_SOURCE);
    if (!coli_jit_destination_owned(attempt))
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_DESTINATION_STALE);

    if (coli_expert_residency_publish_variant_from_source(
            attempt->key.entry, attempt->destination_budget,
            attempt->destination_variant_id,
            attempt->destination_generation,
            attempt->estimate.resident_bytes,
            attempt->output,
            &attempt->source_lease) != 0)
        return coli_jit_fail_attempt(
            service, attempt, COLI_JIT_ERROR_PUBLISH);

    /* Publication transfers ownership of output to the resident variant.
     * Scratch/staging remain transform-owned and are released immediately. */
    attempt->output = NULL;
    if (attempt->scratch) {
        service->memory.free(
            service->memory.context, COLI_JIT_MEMORY_SCRATCH,
            attempt->scratch, attempt->estimate.scratch_bytes);
        attempt->scratch = NULL;
    }
    if (attempt->staging) {
        service->memory.free(
            service->memory.context, COLI_JIT_MEMORY_STAGING,
            attempt->staging, attempt->estimate.staging_bytes);
        attempt->staging = NULL;
    }
    coli_jit_release_temp_reservation(service, attempt);
    (void)coli_expert_residency_release(&attempt->source_lease);

    coli_jit_spin_lock(&service->lock);
    attempt->error = COLI_JIT_ERROR_NONE;
    attempt->state = COLI_JIT_ATTEMPT_PUBLISHED;
    atomic_fetch_add_explicit(
        &service->telemetry.completed, 1, memory_order_acq_rel);
    atomic_store_explicit(
        &service->telemetry.last_backend_tag,
        attempt->ops.backend_tag, memory_order_release);
    coli_jit_spin_unlock(&service->lock);
    return 1;
}

int coli_jit_transform_cancel(
        ColiJitTransformService *service, ColiJitTransformTicket ticket) {
    if (!service || ticket.slot >= COLI_JIT_TRANSFORM_ATTEMPT_CAPACITY ||
        !ticket.serial)
        return -1;
    coli_jit_spin_lock(&service->lock);
    ColiJitTransformAttempt *attempt = &service->attempts[ticket.slot];
    if (!coli_jit_ticket_equal(attempt, ticket.slot, ticket)) {
        coli_jit_spin_unlock(&service->lock);
        return -1;
    }
    if (attempt->state != COLI_JIT_ATTEMPT_QUEUED) {
        coli_jit_spin_unlock(&service->lock);
        return 0;
    }
    attempt->state = COLI_JIT_ATTEMPT_CANCELLING;
    coli_jit_spin_unlock(&service->lock);

    coli_jit_cleanup_unpublished(service, attempt);

    coli_jit_spin_lock(&service->lock);
    if (attempt->serial != ticket.serial ||
        attempt->state != COLI_JIT_ATTEMPT_CANCELLING) {
        coli_jit_spin_unlock(&service->lock);
        return -1;
    }
    attempt->error = COLI_JIT_ERROR_CANCELLED;
    attempt->state = COLI_JIT_ATTEMPT_CANCELLED;
    atomic_fetch_add_explicit(
        &service->telemetry.cancelled, 1, memory_order_acq_rel);
    coli_jit_spin_unlock(&service->lock);
    return 1;
}

int coli_jit_transform_query(
        ColiJitTransformService *service,
        ColiJitTransformTicket ticket,
        ColiJitTransformAttemptInfo *info) {
    if (!service || !info ||
        ticket.slot >= COLI_JIT_TRANSFORM_ATTEMPT_CAPACITY ||
        !ticket.serial)
        return -1;
    coli_jit_spin_lock(&service->lock);
    const ColiJitTransformAttempt *attempt = &service->attempts[ticket.slot];
    if (!coli_jit_ticket_equal(attempt, ticket.slot, ticket)) {
        coli_jit_spin_unlock(&service->lock);
        return -1;
    }
    memset(info, 0, sizeof(*info));
    info->state = attempt->state;
    info->error = attempt->error;
    info->priority = attempt->priority;
    info->key = attempt->key;
    info->destination_variant_id = attempt->destination_variant_id;
    info->destination_generation = attempt->destination_generation;
    info->resident_bytes = attempt->estimate.resident_bytes;
    info->allocation_bytes = attempt->estimate.allocation_bytes;
    info->scratch_bytes = attempt->estimate.scratch_bytes;
    info->staging_bytes = attempt->estimate.staging_bytes;
    info->backend_tag = attempt->ops.backend_tag;
    info->transform_class = attempt->ops.transform_class;
    coli_jit_spin_unlock(&service->lock);
    return 0;
}

void coli_jit_transform_telemetry_snapshot(
        const ColiJitTransformService *service,
        ColiJitTransformTelemetrySnapshot *snapshot) {
    if (!service || !snapshot) return;
    snapshot->requested = atomic_load_explicit(
        &service->telemetry.requested, memory_order_acquire);
    snapshot->owner = atomic_load_explicit(
        &service->telemetry.owner, memory_order_acquire);
    snapshot->join = atomic_load_explicit(
        &service->telemetry.join, memory_order_acquire);
    snapshot->started = atomic_load_explicit(
        &service->telemetry.started, memory_order_acquire);
    snapshot->completed = atomic_load_explicit(
        &service->telemetry.completed, memory_order_acquire);
    snapshot->failed = atomic_load_explicit(
        &service->telemetry.failed, memory_order_acquire);
    snapshot->cancelled = atomic_load_explicit(
        &service->telemetry.cancelled, memory_order_acquire);
    snapshot->input_bytes = atomic_load_explicit(
        &service->telemetry.input_bytes, memory_order_acquire);
    snapshot->output_bytes = atomic_load_explicit(
        &service->telemetry.output_bytes, memory_order_acquire);
    snapshot->scratch_bytes = atomic_load_explicit(
        &service->telemetry.scratch_bytes, memory_order_acquire);
    snapshot->staging_bytes = atomic_load_explicit(
        &service->telemetry.staging_bytes, memory_order_acquire);
    snapshot->queue_ns = atomic_load_explicit(
        &service->telemetry.queue_ns, memory_order_acquire);
    snapshot->prepare_ns = atomic_load_explicit(
        &service->telemetry.prepare_ns, memory_order_acquire);
    snapshot->validate_ns = atomic_load_explicit(
        &service->telemetry.validate_ns, memory_order_acquire);
    snapshot->exact = atomic_load_explicit(
        &service->telemetry.exact, memory_order_acquire);
    snapshot->lossy = atomic_load_explicit(
        &service->telemetry.lossy, memory_order_acquire);
    snapshot->reject_no_budget = atomic_load_explicit(
        &service->telemetry.reject_no_budget, memory_order_acquire);
    snapshot->reject_unsupported = atomic_load_explicit(
        &service->telemetry.reject_unsupported, memory_order_acquire);
    snapshot->last_backend_tag = atomic_load_explicit(
        &service->telemetry.last_backend_tag, memory_order_acquire);
}

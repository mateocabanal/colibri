#ifndef COLIBRI_RUNTIME_MEMORY_H
#define COLIBRI_RUNTIME_MEMORY_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Current host/UMA memory the OS can still satisfy without treating model
 * geometry as a proxy. This is runtime mechanism, not model policy. */
static inline uint64_t coli_runtime_available_memory_bytes(void) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? (uint64_t)status.ullAvailPhys : 0;
#elif defined(__APPLE__)
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm;
    vm_size_t page = 0;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &count) == KERN_SUCCESS &&
        host_page_size(mach_host_self(), &page) == KERN_SUCCESS && page)
        return ((uint64_t)vm.free_count + (uint64_t)vm.inactive_count) *
               (uint64_t)page;
    return 0;
#else
    FILE *stream = fopen("/proc/meminfo", "r");
    if (stream) {
        char line[256];
        unsigned long long kib = 0;
        while (fgets(line, sizeof(line), stream))
            if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
        fclose(stream);
        if (kib) return (uint64_t)kib * UINT64_C(1024);
    }
    long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    return pages > 0 && page > 0 ? (uint64_t)pages * (uint64_t)page : 0;
#endif
}

/* Current process resident bytes. A zero result means unavailable; callers must
 * then preserve an explicit pre-existing cap rather than pretending a total
 * process budget can be converted into free bytes exactly. */
static inline uint64_t coli_runtime_process_resident_bytes(void) {
#ifdef _WIN32
    /* Keep runtime-core and MinGW builds free of an extra Psapi link dependency.
     * Windows callers can still use the OS-available automatic path; explicit
     * total-budget conversion falls back to their existing cap when RSS is 0. */
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return (uint64_t)info.resident_size;
    return 0;
#else
    FILE *stream = fopen("/proc/self/statm", "r");
    if (!stream) return 0;
    unsigned long long total_pages = 0, resident_pages = 0;
    int ok = fscanf(stream, "%llu %llu", &total_pages, &resident_pages) == 2;
    fclose(stream);
    long page = sysconf(_SC_PAGESIZE);
    if (!ok || page <= 0 || resident_pages > UINT64_MAX / (uint64_t)page)
        return 0;
    return (uint64_t)resident_pages * (uint64_t)page;
#endif
}

static inline uint64_t coli_runtime_parse_decimal_gb(const char *value) {
    if (!value || !*value) return 0;
    char *end = NULL;
    long double gb = strtold(value, &end);
    if (end == value || !(gb > 0.0L) || gb > 1000000.0L) return 0;
    long double bytes = gb * 1000000000.0L;
    if (bytes >= (long double)UINT64_MAX) return UINT64_MAX;
    return (uint64_t)bytes;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_RUNTIME_MEMORY_H */

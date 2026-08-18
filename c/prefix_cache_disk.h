/* prefix_cache_disk.h — engine-agnostic persistent prompt-state object store.
 *
 * This layer deliberately knows nothing about attention, GDN, MLA, or any
 * engine-native state layout.  Engines hand it opaque state segments plus exact
 * token IDs.  It owns persistence policy, fail-closed identity, atomic publish,
 * checksums, longest-exact-prefix lookup, and a hard global disk budget.
 *
 * The v1 object format stores a complete engine boundary snapshot.  That is the
 * correctness-first persistence ABI; engines may later decompose opaque state
 * into deduplicated pages without changing the policy/index surface here.
 */
#ifndef COLI_PREFIX_CACHE_DISK_H
#define COLI_PREFIX_CACHE_DISK_H

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <utime.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <sys/statvfs.h>
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define COLI_PREFIX_DISK_VERSION 1u
#define COLI_PREFIX_DISK_DEFAULT_GB 8.0
#define COLI_PREFIX_DISK_DEFAULT_MIN_FREE_GB 1.0
#define COLI_PREFIX_DISK_MAX_TOKENS (1u << 20)
#define COLI_PREFIX_DISK_MAX_SEGMENT_BYTES (64ULL << 30)

#pragma pack(push, 1)
typedef struct {
    char magic[8];                 /* "COLIPF01" */
    uint32_t version;
    uint32_t header_bytes;
    uint64_t namespace_hash;
    uint32_t engine_abi;
    uint32_t flags;
    uint32_t token_count;
    uint32_t reserved;
    uint64_t token_hash;
    uint64_t kv_bytes;
    uint64_t aux_bytes;
    uint64_t payload_bytes;
    uint64_t payload_checksum;
} ColiPrefixDiskHeader;
#pragma pack(pop)

typedef struct {
    int *tokens;
    int token_count;
    unsigned char *kv;
    size_t kv_bytes;
    unsigned char *aux;
    size_t aux_bytes;
    uint64_t namespace_hash;
    uint32_t engine_abi;
} ColiPrefixDiskObject;

typedef struct {
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t writes;
    uint64_t write_bytes;
    uint64_t read_bytes;
    uint64_t corrupt;
    uint64_t evictions;
} ColiPrefixDiskStats;

/* Header-only engine builds each own one process-local counter set. */
static ColiPrefixDiskStats coli_prefix_disk_stats_global;

static inline uint64_t cpd_hash_update(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    if (!h) h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static inline uint64_t cpd_hash_string(uint64_t h, const char *s) {
    if (!s) return cpd_hash_update(h, "", 1);
    return cpd_hash_update(h, s, strlen(s) + 1);
}

static inline int cpd_size_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static inline int cpd_size_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static inline int cpd_eq_ci(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++, cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return *a == 0 && *b == 0;
}

/* 0=off, 1=RAM only, 2=SSD(+hot RAM), 3=auto.  Unset is auto. */
static inline int coli_prefix_cache_mode(void) {
    const char *v = getenv("COLI_PREFIX_CACHE");
    if (!v || !*v || cpd_eq_ci(v, "auto")) return 3;
    if (cpd_eq_ci(v, "off") || !strcmp(v, "0")) return 0;
    if (cpd_eq_ci(v, "ram")) return 1;
    if (cpd_eq_ci(v, "ssd")) return 2;
    return 0; /* unknown policy is fail-closed */
}

static inline uint64_t cpd_env_bytes_gb(const char *name, double def_gb) {
    const char *v = getenv(name);
    double gb = def_gb;
    if (v && *v) {
        char *end = NULL;
        gb = strtod(v, &end);
        if (end == v || !(gb >= 0.0)) return 0;
    }
    long double b = (long double)gb * 1024.0L * 1024.0L * 1024.0L;
    if (b >= (long double)UINT64_MAX) return UINT64_MAX;
    return (uint64_t)b;
}

static inline uint64_t coli_prefix_disk_budget_bytes(void) {
    return cpd_env_bytes_gb("COLI_PREFIX_CACHE_DISK_GB",
                            COLI_PREFIX_DISK_DEFAULT_GB);
}

static inline uint64_t coli_prefix_disk_min_free_bytes(void) {
    return cpd_env_bytes_gb("COLI_PREFIX_CACHE_MIN_FREE_GB",
                            COLI_PREFIX_DISK_DEFAULT_MIN_FREE_GB);
}

static inline int coli_prefix_disk_dir(char *out, size_t cap) {
    const char *explicit_dir = getenv("COLI_PREFIX_CACHE_DIR");
    if (explicit_dir && *explicit_dir) {
        int n = snprintf(out, cap, "%s", explicit_dir);
        return n > 0 && (size_t)n < cap;
    }
#ifdef _WIN32
    const char *base = getenv("LOCALAPPDATA");
    if (!base || !*base) base = getenv("TEMP");
    if (!base || !*base) return 0;
    int n = snprintf(out, cap, "%s/colibri/prefix", base);
#elif defined(__APPLE__)
    const char *home = getenv("HOME");
    if (!home || !*home) return 0;
    int n = snprintf(out, cap, "%s/Library/Caches/colibri/prefix", home);
#else
    const char *base = getenv("XDG_CACHE_HOME");
    const char *home = getenv("HOME");
    int n;
    if (base && *base) n = snprintf(out, cap, "%s/colibri/prefix", base);
    else if (home && *home) n = snprintf(out, cap, "%s/.cache/colibri/prefix", home);
    else return 0;
#endif
    return n > 0 && (size_t)n < cap;
}

static inline int cpd_mkdir_one(const char *path) {
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) return 1;
#else
    if (mkdir(path, 0700) == 0 || errno == EEXIST) return 1;
#endif
    return 0;
}

static inline int cpd_mkdirs(const char *path) {
    if (!path || !*path) return 0;
    char tmp[PATH_MAX];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) return 0;
    memcpy(tmp, path, n + 1);
    for (size_t i = 1; i < n; i++) {
        if (tmp[i] != '/' && tmp[i] != '\\') continue;
#ifdef _WIN32
        if (i == 2 && tmp[1] == ':') continue;
#endif
        char save = tmp[i]; tmp[i] = 0;
        if (*tmp && !cpd_mkdir_one(tmp)) return 0;
        tmp[i] = save;
    }
    return cpd_mkdir_one(tmp);
}

static inline uint64_t cpd_free_bytes(const char *path) {
#ifdef _WIN32
    ULARGE_INTEGER free_bytes;
    if (GetDiskFreeSpaceExA(path, &free_bytes, NULL, NULL))
        return (uint64_t)free_bytes.QuadPart;
    return UINT64_MAX;
#else
    struct statvfs v;
    if (statvfs(path, &v) == 0)
        return (uint64_t)v.f_bavail * (uint64_t)v.f_frsize;
    return UINT64_MAX;
#endif
}

static inline int cpd_sync_file(FILE *f) {
    if (fflush(f) != 0) return 0;
#ifdef _WIN32
    return _commit(_fileno(f)) == 0;
#else
    return fsync(fileno(f)) == 0;
#endif
}

static inline int cpd_seek64(FILE *f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (__int64)off, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t)off, SEEK_SET) == 0;
#endif
}

static inline uint64_t cpd_hash_file_edges(uint64_t h, const char *path,
                                           int require_regular, int *ok) {
    if (ok) *ok = 0;
    if (!path || !*path) return h;
    struct stat st;
    if (stat(path, &st) != 0) return h;
    if (require_regular && !S_ISREG(st.st_mode)) return h;
    h = cpd_hash_string(h, path);
    h = cpd_hash_update(h, &st.st_size, sizeof(st.st_size));
    h = cpd_hash_update(h, &st.st_mtime, sizeof(st.st_mtime));
    if (S_ISREG(st.st_mode)) {
        FILE *f = fopen(path, "rb");
        if (!f) return h;
        unsigned char buf[65536];
        size_t nr = fread(buf, 1, sizeof(buf), f);
        h = cpd_hash_update(h, buf, nr);
        uint64_t size = st.st_size > 0 ? (uint64_t)st.st_size : 0;
        if (size > sizeof(buf) && cpd_seek64(f, size - sizeof(buf))) {
            nr = fread(buf, 1, sizeof(buf), f);
            h = cpd_hash_update(h, buf, nr);
        }
        fclose(f);
    }
    if (ok) *ok = 1;
    return h;
}

/* Persistent native state is enabled only for a regular model artifact.  This
 * intentionally rejects a raw directory namespace: a directory mtime is not a
 * sufficient model fingerprint for unsafe-to-migrate runtime state. */
static inline uint64_t coli_prefix_disk_namespace(const char *engine,
                                                   uint32_t engine_abi,
                                                   const char *model_path,
                                                   const char *config_dir,
                                                   const void *geometry,
                                                   size_t geometry_bytes) {
    int model_ok = 0;
    uint64_t h = 0;
    h = cpd_hash_string(h, engine ? engine : "unknown");
    h = cpd_hash_update(h, &engine_abi, sizeof(engine_abi));
    h = cpd_hash_file_edges(h, model_path, 1, &model_ok);
    if (!model_ok) return 0;
    if (config_dir && *config_dir) {
        char path[PATH_MAX];
        int n = snprintf(path, sizeof(path), "%s/config.json", config_dir);
        if (n > 0 && (size_t)n < sizeof(path)) h = cpd_hash_file_edges(h, path, 0, NULL);
        n = snprintf(path, sizeof(path), "%s/tokenizer.json", config_dir);
        if (n > 0 && (size_t)n < sizeof(path)) h = cpd_hash_file_edges(h, path, 0, NULL);
    }
    if (geometry && geometry_bytes) h = cpd_hash_update(h, geometry, geometry_bytes);
    return h;
}

static inline uint64_t cpd_token_hash(const int *tokens, int n) {
    uint64_t h = 0;
    for (int i = 0; i < n; i++) {
        int32_t t = (int32_t)tokens[i];
        h = cpd_hash_update(h, &t, sizeof(t));
    }
    return h;
}

static inline int cpd_header_basic_valid(const ColiPrefixDiskHeader *h) {
    if (!h || memcmp(h->magic, "COLIPF01", 8) ||
        h->version != COLI_PREFIX_DISK_VERSION ||
        h->header_bytes != sizeof(*h) || h->token_count == 0 ||
        h->token_count > COLI_PREFIX_DISK_MAX_TOKENS ||
        h->kv_bytes > COLI_PREFIX_DISK_MAX_SEGMENT_BYTES ||
        h->aux_bytes > COLI_PREFIX_DISK_MAX_SEGMENT_BYTES)
        return 0;
    uint64_t token_bytes = (uint64_t)h->token_count * sizeof(int32_t);
    if (h->kv_bytes > UINT64_MAX - token_bytes) return 0;
    uint64_t p = token_bytes + h->kv_bytes;
    if (h->aux_bytes > UINT64_MAX - p) return 0;
    return h->payload_bytes == p + h->aux_bytes;
}

static inline int cpd_object_path(char *out, size_t cap, const char *dir,
                                  uint64_t ns, uint32_t abi,
                                  int token_count, uint64_t token_hash) {
    int n = snprintf(out, cap, "%s/%016llx-%08x-%08x-%016llx.cpc",
                     dir, (unsigned long long)ns, abi,
                     (unsigned)token_count, (unsigned long long)token_hash);
    return n > 0 && (size_t)n < cap;
}

static inline int cpd_has_suffix(const char *s, const char *suffix) {
    size_t ns = strlen(s), nx = strlen(suffix);
    return ns >= nx && !strcmp(s + ns - nx, suffix);
}

static inline void cpd_prune_to_budget(const char *dir, uint64_t budget) {
    if (!budget) return;
    for (;;) {
        DIR *d = opendir(dir);
        if (!d) return;
        uint64_t total = 0;
        time_t oldest_time = (time_t)LLONG_MAX;
        char oldest[PATH_MAX] = {0};
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            if (!cpd_has_suffix(de->d_name, ".cpc")) continue;
            char p[PATH_MAX];
            int n = snprintf(p, sizeof(p), "%s/%s", dir, de->d_name);
            if (n <= 0 || (size_t)n >= sizeof(p)) continue;
            struct stat st;
            if (stat(p, &st) != 0 || !S_ISREG(st.st_mode)) continue;
            if (st.st_size > 0) total += (uint64_t)st.st_size;
            if (st.st_mtime < oldest_time) {
                oldest_time = st.st_mtime;
                snprintf(oldest, sizeof(oldest), "%s", p);
            }
        }
        closedir(d);
        if (total <= budget || !*oldest) return;
        if (remove(oldest) != 0) return;
        coli_prefix_disk_stats_global.evictions++;
    }
}

static inline int coli_prefix_disk_enabled(void) {
    int mode = coli_prefix_cache_mode();
    return (mode == 2 || mode == 3) && coli_prefix_disk_budget_bytes() > 0;
}

static inline int coli_prefix_disk_store(uint64_t ns, uint32_t abi,
                                         const int *tokens, int token_count,
                                         const void *kv, size_t kv_bytes,
                                         const void *aux, size_t aux_bytes,
                                         int log) {
    if (!ns || !tokens || token_count <= 0 || !coli_prefix_disk_enabled()) return 0;
    char dir[PATH_MAX];
    if (!coli_prefix_disk_dir(dir, sizeof(dir)) || !cpd_mkdirs(dir)) return 0;
    uint64_t token_hash = cpd_token_hash(tokens, token_count);
    char final_path[PATH_MAX];
    if (!cpd_object_path(final_path, sizeof(final_path), dir, ns, abi,
                         token_count, token_hash)) return 0;

    size_t token_bytes;
    if (!cpd_size_mul((size_t)token_count, sizeof(int32_t), &token_bytes)) return 0;
    size_t payload;
    if (!cpd_size_add(token_bytes, kv_bytes, &payload) ||
        !cpd_size_add(payload, aux_bytes, &payload)) return 0;

    uint64_t reserve = coli_prefix_disk_min_free_bytes();
    uint64_t free_b = cpd_free_bytes(dir);
    if (free_b != UINT64_MAX &&
        ((uint64_t)payload > UINT64_MAX - reserve || free_b < (uint64_t)payload + reserve))
        return 0;

    ColiPrefixDiskHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, "COLIPF01", 8);
    h.version = COLI_PREFIX_DISK_VERSION;
    h.header_bytes = (uint32_t)sizeof(h);
    h.namespace_hash = ns;
    h.engine_abi = abi;
    h.token_count = (uint32_t)token_count;
    h.token_hash = token_hash;
    h.kv_bytes = kv_bytes;
    h.aux_bytes = aux_bytes;
    h.payload_bytes = payload;
    uint64_t checksum = 0;
    for (int i = 0; i < token_count; i++) {
        int32_t t = (int32_t)tokens[i];
        checksum = cpd_hash_update(checksum, &t, sizeof(t));
    }
    if (kv_bytes) checksum = cpd_hash_update(checksum, kv, kv_bytes);
    if (aux_bytes) checksum = cpd_hash_update(checksum, aux, aux_bytes);
    h.payload_checksum = checksum;

    char tmp[PATH_MAX];
#ifdef _WIN32
    int pid = _getpid();
#else
    int pid = (int)getpid();
#endif
    int tn = snprintf(tmp, sizeof(tmp), "%s.tmp-%d-%llu", final_path, pid,
                      (unsigned long long)clock());
    if (tn <= 0 || (size_t)tn >= sizeof(tmp)) return 0;
    FILE *f = fopen(tmp, "wb");
    if (!f) return 0;
    int ok = fwrite(&h, 1, sizeof(h), f) == sizeof(h);
    for (int i = 0; ok && i < token_count; i++) {
        int32_t t = (int32_t)tokens[i];
        ok = fwrite(&t, 1, sizeof(t), f) == sizeof(t);
    }
    if (ok && kv_bytes) ok = fwrite(kv, 1, kv_bytes, f) == kv_bytes;
    if (ok && aux_bytes) ok = fwrite(aux, 1, aux_bytes, f) == aux_bytes;
    if (ok) ok = cpd_sync_file(f);
    if (fclose(f) != 0) ok = 0;
    if (!ok) { remove(tmp); return 0; }
#ifdef _WIN32
    /* rename() cannot replace an existing file under MSVCRT.  Objects are
     * content-addressed; removing the old complete object before publication is
     * safe because a concurrent reader sees either a miss or the new complete file. */
    remove(final_path);
#endif
    if (rename(tmp, final_path) != 0) { remove(tmp); return 0; }
    coli_prefix_disk_stats_global.writes++;
    coli_prefix_disk_stats_global.write_bytes += sizeof(h) + payload;
    cpd_prune_to_budget(dir, coli_prefix_disk_budget_bytes());
    if (log)
        fprintf(stderr, "[PREFIX-DISK] store tier=ssd tokens=%d bytes=%.2fMiB ns=%016llx\n",
                token_count, (double)(sizeof(h) + payload) / (1024.0 * 1024.0),
                (unsigned long long)ns);
    return 1;
}

static inline void coli_prefix_disk_object_free(ColiPrefixDiskObject *o) {
    if (!o) return;
    free(o->tokens); free(o->kv); free(o->aux);
    memset(o, 0, sizeof(*o));
}

static inline int cpd_read_header_tokens(const char *path,
                                         ColiPrefixDiskHeader *h,
                                         int **tokens_out) {
    *tokens_out = NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int ok = fread(h, 1, sizeof(*h), f) == sizeof(*h) && cpd_header_basic_valid(h);
    struct stat st;
    if (ok && stat(path, &st) == 0) {
        uint64_t expected = (uint64_t)sizeof(*h) + h->payload_bytes;
        if (st.st_size < 0 || (uint64_t)st.st_size != expected) ok = 0;
    }
    int *tokens = NULL;
    if (ok) {
        tokens = (int *)malloc((size_t)h->token_count * sizeof(int));
        if (!tokens) ok = 0;
    }
    for (uint32_t i = 0; ok && i < h->token_count; i++) {
        int32_t t;
        if (fread(&t, 1, sizeof(t), f) != sizeof(t)) ok = 0;
        else tokens[i] = (int)t;
    }
    fclose(f);
    if (!ok) { free(tokens); return 0; }
    if (cpd_token_hash(tokens, (int)h->token_count) != h->token_hash) {
        free(tokens); return 0;
    }
    *tokens_out = tokens;
    return 1;
}

static inline int cpd_load_full(const char *path, uint64_t ns, uint32_t abi,
                                ColiPrefixDiskObject *out) {
    ColiPrefixDiskHeader h;
    int *tokens = NULL;
    if (!cpd_read_header_tokens(path, &h, &tokens) ||
        h.namespace_hash != ns || h.engine_abi != abi) {
        free(tokens); return 0;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { free(tokens); return 0; }
    uint64_t off = sizeof(h) + (uint64_t)h.token_count * sizeof(int32_t);
    int ok = cpd_seek64(f, off);
    unsigned char *kv = h.kv_bytes ? (unsigned char *)malloc((size_t)h.kv_bytes) : NULL;
    unsigned char *aux = h.aux_bytes ? (unsigned char *)malloc((size_t)h.aux_bytes) : NULL;
    if ((h.kv_bytes && !kv) || (h.aux_bytes && !aux)) ok = 0;
    if (ok && h.kv_bytes) ok = fread(kv, 1, (size_t)h.kv_bytes, f) == (size_t)h.kv_bytes;
    if (ok && h.aux_bytes) ok = fread(aux, 1, (size_t)h.aux_bytes, f) == (size_t)h.aux_bytes;
    fclose(f);
    uint64_t checksum = 0;
    if (ok) {
        for (uint32_t i = 0; i < h.token_count; i++) {
            int32_t t = (int32_t)tokens[i];
            checksum = cpd_hash_update(checksum, &t, sizeof(t));
        }
        if (h.kv_bytes) checksum = cpd_hash_update(checksum, kv, (size_t)h.kv_bytes);
        if (h.aux_bytes) checksum = cpd_hash_update(checksum, aux, (size_t)h.aux_bytes);
        ok = checksum == h.payload_checksum;
    }
    if (!ok) { free(tokens); free(kv); free(aux); return 0; }
    memset(out, 0, sizeof(*out));
    out->tokens = tokens;
    out->token_count = (int)h.token_count;
    out->kv = kv; out->kv_bytes = (size_t)h.kv_bytes;
    out->aux = aux; out->aux_bytes = (size_t)h.aux_bytes;
    out->namespace_hash = ns; out->engine_abi = abi;
    coli_prefix_disk_stats_global.read_bytes += sizeof(h) + h.payload_bytes;
    (void)utime(path, NULL); /* access heat for deterministic disk LRU */
    return 1;
}

static inline int coli_prefix_disk_load_longest(uint64_t ns, uint32_t abi,
                                                 const int *prompt,
                                                 int prompt_count,
                                                 ColiPrefixDiskObject *out,
                                                 int log) {
    if (out) memset(out, 0, sizeof(*out));
    if (!ns || !prompt || prompt_count <= 1 || !out || !coli_prefix_disk_enabled()) return 0;
    coli_prefix_disk_stats_global.lookups++;
    char dir[PATH_MAX];
    if (!coli_prefix_disk_dir(dir, sizeof(dir))) return 0;
    DIR *d = opendir(dir);
    if (!d) { coli_prefix_disk_stats_global.misses++; return 0; }
    char best[PATH_MAX] = {0};
    int best_tokens = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (!cpd_has_suffix(de->d_name, ".cpc")) continue;
        char path[PATH_MAX];
        int pn = snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        if (pn <= 0 || (size_t)pn >= sizeof(path)) continue;
        ColiPrefixDiskHeader h;
        int *tokens = NULL;
        if (!cpd_read_header_tokens(path, &h, &tokens)) {
            coli_prefix_disk_stats_global.corrupt++;
            continue;
        }
        int n = (int)h.token_count;
        int match = h.namespace_hash == ns && h.engine_abi == abi &&
                    n > best_tokens && n < prompt_count &&
                    !memcmp(tokens, prompt, (size_t)n * sizeof(int));
        free(tokens);
        if (match) {
            best_tokens = n;
            snprintf(best, sizeof(best), "%s", path);
        }
    }
    closedir(d);
    if (!*best || !cpd_load_full(best, ns, abi, out)) {
        if (*best) coli_prefix_disk_stats_global.corrupt++;
        coli_prefix_disk_stats_global.misses++;
        return 0;
    }
    coli_prefix_disk_stats_global.hits++;
    if (log)
        fprintf(stderr, "[PREFIX-DISK] hit tier=ssd matched=%d prompt=%d bytes=%.2fMiB ns=%016llx\n",
                out->token_count, prompt_count,
                (double)(out->kv_bytes + out->aux_bytes) / (1024.0 * 1024.0),
                (unsigned long long)ns);
    return out->token_count;
}

static inline ColiPrefixDiskStats coli_prefix_disk_stats(void) {
    return coli_prefix_disk_stats_global;
}

#endif /* COLI_PREFIX_CACHE_DISK_H */

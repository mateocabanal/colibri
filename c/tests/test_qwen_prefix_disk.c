#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <assert.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../qwen_prefix_cache.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/wait.h>
#endif

#define LAYERS 2
#define MAXT 8
#define KVH 1
#define HD 2
#define GDN_S 4
#define GDN_C 3

typedef struct {
    int8_t kinds[LAYERS];
    uint16_t *K16[LAYERS], *V16[LAYERS];
    float *S[LAYERS], *C[LAYERS];
    QwenPrefixStateView view;
} Fixture;

static void fixture_init(Fixture *f) {
    memset(f, 0, sizeof(*f));
    f->kinds[0] = 1;
    f->S[0] = calloc(GDN_S, sizeof(float));
    f->C[0] = calloc(GDN_C, sizeof(float));
    f->K16[1] = calloc(KVH * MAXT * HD, sizeof(uint16_t));
    f->V16[1] = calloc(KVH * MAXT * HD, sizeof(uint16_t));
    assert(f->S[0] && f->C[0] && f->K16[1] && f->V16[1]);
    f->view = (QwenPrefixStateView){
        .layer_count = LAYERS, .layer_is_gdn = f->kinds,
        .n_kv_heads = KVH, .head_dim = HD, .max_t = MAXT, .kv_f16 = 1,
        .K16 = f->K16, .V16 = f->V16, .gdn_S = f->S, .gdn_conv = f->C,
        .gdn_state_elems = GDN_S, .gdn_conv_elems = GDN_C,
    };
}

static void fixture_free(Fixture *f) {
    free(f->S[0]); free(f->C[0]); free(f->K16[1]); free(f->V16[1]);
}

static void fill(Fixture *f, int seed) {
    for (int i = 0; i < GDN_S; i++) f->S[0][i] = (float)(seed * 100 + i);
    for (int i = 0; i < GDN_C; i++) f->C[0][i] = (float)(seed * 200 + i);
    for (int i = 0; i < KVH * MAXT * HD; i++) {
        f->K16[1][i] = (uint16_t)(seed * 10 + i);
        f->V16[1][i] = (uint16_t)(seed * 20 + i);
    }
}

static void assert_prefix(const Fixture *f, int seed, int prefix) {
    for (int i = 0; i < GDN_S; i++) assert(f->S[0][i] == (float)(seed * 100 + i));
    for (int i = 0; i < GDN_C; i++) assert(f->C[0][i] == (float)(seed * 200 + i));
    for (int p = 0; p < prefix; p++) for (int d = 0; d < HD; d++) {
        size_t i = (size_t)p * HD + (size_t)d;
        assert(f->K16[1][i] == (uint16_t)(seed * 10 + (int)i));
        assert(f->V16[1][i] == (uint16_t)(seed * 20 + (int)i));
    }
}

static int mkdir_portable(const char *path) {
#ifdef _WIN32
    return _mkdir(path) == 0 || errno == EEXIST;
#else
    return mkdir(path, 0700) == 0 || errno == EEXIST;
#endif
}

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    assert(f);
    size_t n = strlen(text);
    assert(fwrite(text, 1, n, f) == n);
    assert(fclose(f) == 0);
}

static void clear_flat_dir(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *de;
    char path[COLI_PREFIX_DISK_PATH_MAX];
    while ((de = readdir(d)) != NULL) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        size_t a = strlen(dir), b = strlen(de->d_name);
        if (a + 1 + b + 1 > sizeof(path)) continue;
        memcpy(path, dir, a); path[a] = '/'; memcpy(path + a + 1, de->d_name, b + 1);
        (void)remove(path);
    }
    closedir(d);
#ifdef _WIN32
    (void)_rmdir(dir);
#else
    (void)rmdir(dir);
#endif
}

static void set_env(const char *name, const char *value) {
#ifdef _WIN32
    assert(_putenv_s(name, value) == 0);
#else
    assert(setenv(name, value, 1) == 0);
#endif
}

static void unset_env(const char *name) {
#ifdef _WIN32
    assert(_putenv_s(name, "") == 0);
#else
    assert(unsetenv(name) == 0);
#endif
}

#ifndef _WIN32
static void test_posix_lock_behavior(const char *dir) {
    PrefixLock owner = {.fd = -1};
    assert(lock_acquire(dir, &owner));

    int blocked_pipe[2], acquired_pipe[2];
    assert(pipe(blocked_pipe) == 0);
    assert(pipe(acquired_pipe) == 0);
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        close(blocked_pipe[0]);
        close(acquired_pipe[0]);

        char lock_path[COLI_PREFIX_DISK_PATH_MAX];
        assert(join_path(lock_path, sizeof(lock_path), dir, ".lock"));
        int probe_fd = open(lock_path, O_RDWR);
        assert(probe_fd >= 0);
        struct flock probe;
        memset(&probe, 0, sizeof(probe));
        probe.l_type = F_WRLCK;
        probe.l_whence = SEEK_SET;
        probe.l_start = 0;
        probe.l_len = 1;
        errno = 0;
        assert(fcntl(probe_fd, F_SETLK, &probe) == -1);
        assert(errno == EACCES || errno == EAGAIN);
        close(probe_fd);
        assert(write(blocked_pipe[1], "B", 1) == 1);

        PrefixLock waiter = {.fd = -1};
        assert(lock_acquire(dir, &waiter));
        assert(write(acquired_pipe[1], "A", 1) == 1);
        lock_release(&waiter);
        close(blocked_pipe[1]);
        close(acquired_pipe[1]);
        _exit(0);
    }

    close(blocked_pipe[1]);
    close(acquired_pipe[1]);
    char marker = 0;
    assert(read(blocked_pipe[0], &marker, 1) == 1 && marker == 'B');

    int flags = fcntl(acquired_pipe[0], F_GETFL);
    assert(flags >= 0);
    assert(fcntl(acquired_pipe[0], F_SETFL, flags | O_NONBLOCK) == 0);
    errno = 0;
    assert(read(acquired_pipe[0], &marker, 1) == -1);
    assert(errno == EAGAIN || errno == EWOULDBLOCK);
    assert(fcntl(acquired_pipe[0], F_SETFL, flags) == 0);

    /* lock_release must issue F_UNLCK; the blocked F_SETLKW then acquires. */
    lock_release(&owner);
    assert(read(acquired_pipe[0], &marker, 1) == 1 && marker == 'A');

    int status = 0;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    close(blocked_pipe[0]);
    close(acquired_pipe[0]);
}
#endif

int main(void) {
    char root[512], package[640], config[640], cache[640], path[1024];
#ifdef _WIN32
    const char *base = getenv("TEMP"); if (!base || !*base) base = ".";
    snprintf(root, sizeof(root), "%s/coli-qwen-prefix-disk-%lu", base, (unsigned long)getpid());
#else
    snprintf(root, sizeof(root), "/tmp/coli-qwen-prefix-disk-%lu", (unsigned long)getpid());
#endif
    snprintf(package, sizeof(package), "%s/model.coli", root);
    snprintf(config, sizeof(config), "%s/config", root);
    snprintf(cache, sizeof(cache), "%s/cache", root);
    assert(mkdir_portable(root)); assert(mkdir_portable(package));
    assert(mkdir_portable(config)); assert(mkdir_portable(cache));
#ifndef _WIN32
    test_posix_lock_behavior(cache);
#endif
    snprintf(path, sizeof(path), "%s/manifest.coli", package); write_file(path, "manifest-v1\n");
    snprintf(path, sizeof(path), "%s/config.json", config); write_file(path, "{\"model_type\":\"qwen-test\"}\n");
    snprintf(path, sizeof(path), "%s/tokenizer.json", config); write_file(path, "{\"version\":\"1\"}\n");

    set_env("SERVE", "1"); set_env("SNAP", package); set_env("COLI_CONFIG", config);
    set_env("COLI_PREFIX_CACHE", "ssd"); set_env("COLI_PREFIX_CACHE_DIR", cache);
    set_env("COLI_PREFIX_CACHE_DISK_GB", "0.01"); set_env("COLI_PREFIX_CACHE_MIN_FREE_GB", "0");
    set_env("COLI_PREFIX_CACHE_MIN_TOKENS", "1"); unset_env("QWEN_PREFIX_CACHE_MB");
#ifdef COLI_METAL
    set_env("QWEN_METAL_COMPUTE", "0");
#endif
#ifdef COLI_CUDA
    set_env("QWEN_CUDA_COMPUTE", "0");
#endif

    int p[] = {7, 8, 9};
    int q[] = {7, 8, 9, 10};
    Fixture f; fixture_init(&f); fill(&f, 3);

    QwenPrefixCache writer = {0};
    qwen_prefix_cache_store(&writer, &f.view, p, 3);
    assert(writer.budget_bytes == 0); /* SSD-only really means no RAM hot tier. */
    fill(&f, 9);
    QwenPrefixCache reader = {0};
    assert(qwen_prefix_cache_restore(&reader, &f.view, q, 4) == 3);
    assert_prefix(&f, 3, 3);

    /* Restart-like second cache object in auto mode: SSD hit must promote to RAM. */
    set_env("COLI_PREFIX_CACHE", "auto");
    fill(&f, 11);
    QwenPrefixCache promoted = {0};
    assert(qwen_prefix_cache_restore(&promoted, &f.view, q, 4) == 3);
    assert_prefix(&f, 3, 3);
    assert(promoted.budget_bytes > 0 && promoted.count == 1);

    /* Changing tokenizer identity must make the persisted object unreachable. */
    snprintf(path, sizeof(path), "%s/tokenizer.json", config); write_file(path, "{\"version\":\"2\"}\n");
    fill(&f, 13);
    QwenPrefixCache mismatch = {0};
    assert(qwen_prefix_cache_restore(&mismatch, &f.view, q, 4) == 0);

    qwen_prefix_cache_clear(&writer); qwen_prefix_cache_clear(&reader);
    qwen_prefix_cache_clear(&promoted); qwen_prefix_cache_clear(&mismatch);
    fixture_free(&f);
    clear_flat_dir(cache);
    snprintf(path, sizeof(path), "%s/manifest.coli", package); (void)remove(path);
    snprintf(path, sizeof(path), "%s/config.json", config); (void)remove(path);
    snprintf(path, sizeof(path), "%s/tokenizer.json", config); (void)remove(path);
#ifdef _WIN32
    (void)_rmdir(package); (void)_rmdir(config); (void)_rmdir(root);
#else
    (void)rmdir(package); (void)rmdir(config); (void)rmdir(root);
#endif
    puts("qwen prefix disk: ok");
    return 0;
}

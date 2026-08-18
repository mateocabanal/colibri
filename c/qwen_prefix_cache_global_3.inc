            return ok;
        }
    }
}

static inline int qpc_join_file(char *out, size_t out_size,
                                const char *directory, const char *name) {
    if (!out || !out_size || !directory || !*directory || !name || !*name)
        return 0;
    int n = snprintf(out, out_size, "%s/%s", directory, name);
    return n > 0 && (size_t)n < out_size;
}

/* Native state is persisted only when the model/config/tokenizer identity can
 * be established without guessing. A compiled .coli directory is identified by
 * its CRC-covered manifest.coli. A regular-file SNAP is accepted for fixtures
 * and legacy compiled artifacts. Raw model directories fail closed. */
static inline int qwen_prefix_disk_model_fingerprint(uint8_t out[32]) {
    if (!out) return 0;
    const char *model = getenv("SNAP");
    const char *config = getenv("COLI_CONFIG");
    if (!model || !*model || !config || !*config) return 0;
    uint64_t h[4] = {
        UINT64_C(1469598103934665603),
        UINT64_C(1099511628211) ^ UINT64_C(0x9e3779b97f4a7c15),
        UINT64_C(0x6a09e667f3bcc909),
        UINT64_C(0xbb67ae8584caa73b),
    };
    struct stat st;
    if (stat(model, &st) != 0) return 0;
    char path[COLI_PREFIX_DISK_PATH_MAX];
    if (S_ISREG(st.st_mode)) {
        if (!qpc_hash_file_full(h, model, "model-artifact")) return 0;
    } else if (S_ISDIR(st.st_mode)) {
        if (!qpc_join_file(path, sizeof(path), model, "manifest.coli") ||
            !qpc_hash_file_full(h, path, "manifest.coli"))
            return 0;
    } else {
        return 0;
    }
    if (!qpc_join_file(path, sizeof(path), config, "config.json") ||
        !qpc_hash_file_full(h, path, "config.json") ||
        !qpc_join_file(path, sizeof(path), config, "tokenizer.json") ||
        !qpc_hash_file_full(h, path, "tokenizer.json"))
        return 0;

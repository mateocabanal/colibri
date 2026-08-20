#include <stdio.h>
#include <string.h>

#include "../speculative_cache_disk.h"

static int failures = 0;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

static ColiSpecDiskIdentity identity(uint8_t seed) {
    ColiSpecDiskIdentity id;
    size_t i;
    memset(&id, 0, sizeof(id));
    for (i = 0; i < sizeof(id.model_fingerprint); ++i) {
        id.model_fingerprint[i] = (uint8_t)(seed + i);
    }
    id.tokenizer_fingerprint = 0x1122334455667788ull + seed;
    id.engine_id = 7u;
    id.token_abi = 1u;
    return id;
}

int main(void) {
    const char *path = "test_speculative_cache.snapshot";
    int source_storage[16];
    int loaded_storage[16] = {77, 77, 77};
    ColiSpecContinuationCache source = {source_storage, 0, 16, -1};
    ColiSpecContinuationCache loaded = {loaded_storage, 3, 16, -1};
    ColiSpecDiskIdentity id = identity(3);
    ColiSpecDiskIdentity wrong = identity(4);
    static const int first[] = {10, 11, 12, 13};
    static const int second[] = {20, 21, 22};

    (void)remove(path);
    (void)remove("test_speculative_cache.snapshot.tmp");

    check(coli_spec_cache_append_span(&source, first, 4) == 1, "append first persisted span");
    check(coli_spec_cache_append_span(&source, second, 3) == 1, "append second persisted span");
    check(coli_spec_disk_store_atomic(path, &id, source.tokens, source.count, source.separator) == 1,
          "atomic continuation snapshot stored");
    check(coli_spec_disk_load(path, &id, &loaded) == 1, "matching identity restores snapshot");
    check(loaded.count == source.count &&
          memcmp(loaded.tokens, source.tokens, source.count * sizeof(int)) == 0,
          "restored continuation tokens are exact");

    loaded.count = 3;
    loaded.tokens[0] = 77;
    loaded.tokens[1] = 77;
    loaded.tokens[2] = 77;
    check(coli_spec_disk_load(path, &wrong, &loaded) == 0, "identity mismatch is safe miss");
    check(loaded.count == 3 && loaded.tokens[0] == 77 && loaded.tokens[1] == 77,
          "identity mismatch leaves live cache untouched");

    {
        FILE *file = fopen(path, "r+b");
        check(file != NULL, "open snapshot for corruption test");
        if (file != NULL) {
            int byte;
            check(fseek(file, COLI_SPEC_DISK_HEADER_BYTES + 4, SEEK_SET) == 0,
                  "seek to payload byte");
            byte = fgetc(file);
            check(byte != EOF, "read payload byte");
            if (byte != EOF) {
                check(fseek(file, -1, SEEK_CUR) == 0, "rewind payload byte");
                check(fputc(byte ^ 0x40, file) != EOF, "flip payload byte");
            }
            fclose(file);
        }
    }
    check(coli_spec_disk_load(path, &id, &loaded) == 0, "payload CRC corruption is safe miss");
    check(loaded.count == 3 && loaded.tokens[0] == 77,
          "corrupt payload leaves live cache untouched");

    check(coli_spec_disk_store_atomic(path, &id, source.tokens, source.count, source.separator) == 1,
          "rewrite valid snapshot after corruption");
    {
        FILE *file = fopen(path, "ab");
        check(file != NULL, "open snapshot for trailing-byte test");
        if (file != NULL) {
            check(fputc(0, file) != EOF, "append trailing byte");
            fclose(file);
        }
    }
    check(coli_spec_disk_load(path, &id, &loaded) == 0, "trailing bytes reject snapshot");

    (void)remove(path);
    (void)remove("test_speculative_cache.snapshot.tmp");
    if (failures != 0) {
        fprintf(stderr, "speculative_cache_disk: %d failure(s)\n", failures);
        return 1;
    }
    printf("speculative_cache_disk: identity, atomic snapshot, CRC, fail-clean restore ok\n");
    return 0;
}


#define PFX_MAGIC "COLIPF1\0"
#define PFX_ENDIAN UINT32_C(0x01020304)
#define PFX_FNV_OFFSET UINT64_C(1469598103934665603)
#define PFX_FNV_PRIME UINT64_C(1099511628211)
#define PFX_DEFAULT_FREE_GIB 1.0L

#pragma pack(push,1)
typedef struct {
    char magic[8];
    uint32_t version, header_bytes, endian, engine_id, state_abi, token_count;
    uint8_t model_fingerprint[32];
    uint64_t layout_fingerprint, token_hash, payload_bytes;
    uint32_t payload_crc32c, reserved;
} PrefixDiskHeader;
#pragma pack(pop)

typedef struct { char path[COLI_PREFIX_DISK_PATH_MAX]; uint64_t bytes; time_t mtime; } PruneEntry;
typedef struct {
#ifdef _WIN32
    HANDLE handle; OVERLAPPED overlapped;
#else
    int fd;
#endif
} PrefixLock;

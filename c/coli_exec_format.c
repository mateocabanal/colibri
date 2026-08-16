#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "coli_exec_format.h"
#include "compat.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MANIFEST_HEADER_BYTES 256u
#define SHARD_DESC_BYTES 64u
#define RECORD_DESC_BYTES 96u
#define STRING_DESC_BYTES 16u
#define CODEC_DESC_BYTES 64u
#define DATA_HEADER_BYTES 128u
#define TENSOR_HEADER_BYTES 128u
#define EXPERT_HEADER_BYTES 64u
#define EXPERT_MATRIX_BYTES 128u
#define EXPERT_TABLE_BYTES (EXPERT_HEADER_BYTES + 3u * EXPERT_MATRIX_BYTES)
#define RANS_SLACK 64u
#define MAX_MANIFEST_BYTES (1ULL << 30)
#define MAX_RECORDS (1u << 24)
#define MAX_SHARDS (1u << 20)
#define MAX_STRINGS (1u << 24)
#define MAX_CODEC_TABLES (1u << 20)

static const unsigned char k_manifest_magic[8] =
    {0x43,0x4f,0x4c,0x49,0x0d,0x0a,0x1a,0x0a};
static const unsigned char k_data_magic[8] =
    {0x43,0x4f,0x49,0x44,0x41,0x54,0x00,0x00}; /* fixed below in comparison */
static const unsigned char k_data_magic_exact[8] =
    {0x43,0x4f,0x4c,0x49,0x44,0x41,0x54,0x00};
static const unsigned char k_tensor_magic[8] =
    {0x43,0x4f,0x4c,0x49,0x54,0x45,0x4e,0x53};
static const unsigned char k_expert_magic[8] =
    {0x43,0x4f,0x4c,0x49,0x45,0x58,0x50,0x54};

typedef struct ExecShard {
    uint32_t id;
    uint64_t file_bytes;
    uint32_t header_crc32c;
    char *name;
    char *path;
    int fd;
} ExecShard;

typedef struct ExecCodecTable {
    uint32_t id;
    uint16_t codec;
    int32_t shard_id;
    uint64_t data_offset;
    uint64_t data_bytes;
    uint32_t data_crc32c;
} ExecCodecTable;

typedef struct IndexSlot {
    uint64_t hash;
    uint32_t index_plus_one;
} IndexSlot;

typedef struct Span {
    uint32_t shard;
    uint64_t begin;
    uint64_t end;
    size_t index;
} Span;

struct ColiExecPackage {
    char *root;
    ColiTargetInfo target;
    ColiExecChecksumPolicy checksum_policy;

    char **strings;
    uint32_t string_count;
    ExecShard *shards;
    uint32_t shard_count;
    ExecCodecTable *codec_tables;
    uint32_t codec_table_count;
    ColiExecRecordInfo *records;
    size_t record_count;

    IndexSlot *id_index;
    size_t id_cap;
    IndexSlot *name_index;
    size_t name_cap;
    IndexSlot *expert_index;
    size_t expert_cap;
    IndexSlot *layer_index;
    size_t layer_cap;
};

static void set_error(char *error, size_t error_size, const char *fmt, ...) {
    va_list ap;
    if (!error || !error_size) return;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
    error[error_size - 1] = '\0';
}
static uint16_t rd16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t rd32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const unsigned char *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}
static int32_t rdi32(const unsigned char *p) { return (int32_t)rd32(p); }
static int checked_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (UINT64_MAX - a < b) return -1;
    *out = a + b;
    return 0;
}
static int all_zero(const unsigned char *p, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (p[i]) return 0;
    return 1;
}
static int power2(uint32_t x) { return x && !(x & (x - 1)); }
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static uint64_t hash_string(const char *s) {
    uint64_t h = UINT64_C(1469598103934665603);
    while (*s) { h ^= (unsigned char)*s++; h *= UINT64_C(1099511628211); }
    return h ? h : 1;
}
static uint64_t hash_pair(int32_t a, int32_t b) {
    return mix64(((uint64_t)(uint32_t)a << 32) | (uint32_t)b) | 1;
}
static char *join_path(const char *root, const char *leaf) {
    size_t a = strlen(root), b = strlen(leaf);
    int slash = a && root[a - 1] != '/' && root[a - 1] != '\\';
    char *out;
    if (a > SIZE_MAX - b - 2) return NULL;
    out = (char *)malloc(a + b + (size_t)slash + 1);
    if (!out) return NULL;
    memcpy(out, root, a);
    if (slash) out[a++] = '/';
    memcpy(out + a, leaf, b + 1);
    return out;
}
static int safe_filename(const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    if (!s[0] || !strcmp(s, ".") || !strcmp(s, "..")) return 0;
    for (; *p; ++p)
        if (!( (*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') ||
               (*p >= '0' && *p <= '9') || *p == '.' || *p == '_' || *p == '-' ))
            return 0;
    return 1;
}
static int pread_full(int fd, void *dst, size_t bytes, uint64_t offset,
                      char *error, size_t error_size) {
    unsigned char *p = (unsigned char *)dst;
    size_t done = 0;
    while (done < bytes) {
        size_t chunk = bytes - done;
        ssize_t n;
        if (chunk > 0x40000000u) chunk = 0x40000000u;
        if (offset > (uint64_t)INT64_MAX - done) {
            set_error(error, error_size, "read offset exceeds signed 64-bit range");
            return -1;
        }
        do { n = pread(fd, p + done, chunk, (off_t)(offset + done)); }
        while (n < 0 && errno == EINTR);
        if (n < 0) {
            set_error(error, error_size, "pread failed: %s", strerror(errno));
            return -1;
        }
        if (!n) {
            set_error(error, error_size, "truncated data shard");
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}
static int read_file(const char *path, unsigned char **out, size_t *out_n,
                     char *error, size_t error_size) {
    int fd = -1;
    struct stat st;
    unsigned char *p = NULL;
    fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) { set_error(error,error_size,"cannot open %s: %s",path,strerror(errno)); return -1; }
    if (fstat(fd,&st) || st.st_size < 0 || (uint64_t)st.st_size > MAX_MANIFEST_BYTES) {
        set_error(error,error_size,"manifest size is invalid"); close(fd); return -1;
    }
    p = (unsigned char *)malloc((size_t)st.st_size ? (size_t)st.st_size : 1);
    if (!p) { set_error(error,error_size,"out of memory reading manifest"); close(fd); return -1; }
    if (pread_full(fd,p,(size_t)st.st_size,0,error,error_size)) { free(p); close(fd); return -1; }
    close(fd); *out=p; *out_n=(size_t)st.st_size; return 0;
}

static int utf8_valid(const unsigned char *s, size_t n) {
    size_t i=0;
    while(i<n){unsigned c=s[i++],need,min2=0x80,max2=0xbf;if(c<0x80)continue;
        if(c>=0xc2&&c<=0xdf)need=1;else if(c>=0xe0&&c<=0xef){need=2;if(c==0xe0)min2=0xa0;if(c==0xed)max2=0x9f;}
        else if(c>=0xf0&&c<=0xf4){need=3;if(c==0xf0)min2=0x90;if(c==0xf4)max2=0x8f;}else return 0;
        if(i+need>n||s[i]<min2||s[i]>max2)return 0;i++;while(--need){if(s[i]<0x80||s[i]>0xbf)return 0;i++;}}
    return 1;
}
static int parse_strings(ColiExecPackage *p, const unsigned char *m,
                         uint64_t off, uint64_t bytes, uint32_t count,
                         char *error, size_t error_size) {
    uint64_t desc = (uint64_t)count * STRING_DESC_BYTES;
    uint32_t i;
    if (count > MAX_STRINGS || desc > bytes) {
        set_error(error,error_size,"invalid/oversized string table"); return -1;
    }
    p->strings = (char **)calloc(count ? count : 1, sizeof(*p->strings));
    if (!p->strings) { set_error(error,error_size,"out of memory for strings"); return -1; }
    p->string_count = count;
    for (i=0;i<count;i++) {
        const unsigned char *d=m+off+(size_t)i*16;
        uint64_t so=rd64(d),end; uint32_t n=rd32(d+8);
        if(rd32(d+12)||so<desc||checked_add(so,n,&end)||end>bytes||
           memchr(m+off+so,0,n)||!utf8_valid(m+off+so,n)) {
            set_error(error,error_size,"invalid string descriptor %u",i); return -1;
        }
        p->strings[i]=(char *)malloc((size_t)n+1);
        if(!p->strings[i]){set_error(error,error_size,"out of memory for string");return -1;}
        memcpy(p->strings[i],m+off+so,n);p->strings[i][n]='\0';
    }
    return 0;
}

static int known_codec(uint16_t c) {
    return c == COLI_CSF_CODEC_NONE || c == COLI_CSF_CODEC_RANS256_G0_NIBBLE ||
           c == COLI_CSF_CODEC_RANS256_G0_U8;
}
static int known_math(uint16_t m) {
    switch(m){
    case COLI_CSF_MATH_NONE:case COLI_CSF_MATH_F32:case COLI_CSF_MATH_F16:case COLI_CSF_MATH_BF16:
    case COLI_CSF_MATH_I8:case COLI_CSF_MATH_U8:case COLI_CSF_MATH_I16:case COLI_CSF_MATH_U16:
    case COLI_CSF_MATH_I32:case COLI_CSF_MATH_U32:case COLI_CSF_MATH_I64:case COLI_CSF_MATH_U64:
    case COLI_CSF_MATH_BOOL:case COLI_CSF_MATH_FP8_E4M3FN:case COLI_CSF_MATH_FP8_E5M2:
    case COLI_CSF_MATH_MXFP4_E2M1:case COLI_CSF_MATH_INT4_PACKED:case COLI_CSF_MATH_INT4_GROUPED:
    case COLI_CSF_MATH_MIXED:return 1;default:return 0;}
}
static int known_scale(uint16_t s) {
    return s==COLI_CSF_SCALE_NONE||s==COLI_CSF_SCALE_F32||s==COLI_CSF_SCALE_F16||
           s==COLI_CSF_SCALE_BF16||s==COLI_CSF_SCALE_UE8M0||s==COLI_CSF_SCALE_MIXED;
}
static int target_layout(uint16_t l) {
    return (l>=COLI_EXEC_LAYOUT_APPLE_MIN&&l<=COLI_EXEC_LAYOUT_APPLE_MAX)||
           (l>=COLI_EXEC_LAYOUT_X86_MIN&&l<=COLI_EXEC_LAYOUT_X86_MAX)||
           (l>=COLI_EXEC_LAYOUT_CUDA_MIN&&l<=COLI_EXEC_LAYOUT_CUDA_MAX);
}
static int optional_record(const ColiExecRecordInfo *r) {
    return (r->flags & COLI_CSF_RECORD_F_OPTIONAL) != 0;
}

static size_t next_pow2(size_t n) {
    size_t p=8; while(p<n && p<=SIZE_MAX/2)p<<=1; return p;
}
static int alloc_index(IndexSlot **out,size_t *cap,size_t n,char *error,size_t error_size){
    size_t c=next_pow2(n*2+1);*out=(IndexSlot *)calloc(c,sizeof(**out));if(!*out){set_error(error,error_size,"out of memory for package index");return -1;}*cap=c;return 0;
}
static int id_insert(ColiExecPackage *p,size_t index,char *error,size_t error_size){
    uint64_t key=p->records[index].record_id,h=mix64(key)|1;size_t pos=(size_t)h&(p->id_cap-1);
    while(p->id_index[pos].index_plus_one){size_t old=p->id_index[pos].index_plus_one-1;if(p->records[old].record_id==key){set_error(error,error_size,"duplicate record id %llu",(unsigned long long)key);return -1;}pos=(pos+1)&(p->id_cap-1);}p->id_index[pos].hash=h;p->id_index[pos].index_plus_one=(uint32_t)index+1;return 0;
}
static int name_insert(ColiExecPackage *p,size_t index,char *error,size_t error_size){
    const char *key=p->records[index].name;uint64_t h=hash_string(key);size_t pos=(size_t)h&(p->name_cap-1);
    while(p->name_index[pos].index_plus_one){size_t old=p->name_index[pos].index_plus_one-1;if(!strcmp(p->records[old].name,key)){set_error(error,error_size,"duplicate record name %s",key);return -1;}pos=(pos+1)&(p->name_cap-1);}p->name_index[pos].hash=h;p->name_index[pos].index_plus_one=(uint32_t)index+1;return 0;
}
static int pair_insert(IndexSlot *slots,size_t cap,const ColiExecRecordInfo *records,size_t index,int layer_only,char *error,size_t error_size){
    int32_t a=records[index].layer,b=layer_only?0:records[index].expert;uint64_t h=hash_pair(a,b);size_t pos=(size_t)h&(cap-1);
    while(slots[pos].index_plus_one){size_t old=slots[pos].index_plus_one-1;if(records[old].layer==a&&(layer_only||records[old].expert==b)){set_error(error,error_size,layer_only?"duplicate layer-pack layer %d":"duplicate expert (%d,%d)",a,b);return -1;}pos=(pos+1)&(cap-1);}slots[pos].hash=h;slots[pos].index_plus_one=(uint32_t)index+1;return 0;
}

static const ExecCodecTable *codec_table(const ColiExecPackage *p,uint32_t id){
    uint32_t i;if(!id)return NULL;for(i=0;i<p->codec_table_count;i++)if(p->codec_tables[i].id==id)return &p->codec_tables[i];return NULL;
}
static int codec_ref_ok(const ColiExecPackage *p,uint16_t codec,uint32_t table_id,uint32_t shard,char *error,size_t error_size){
    const ExecCodecTable *t;if(codec==COLI_CSF_CODEC_NONE){if(table_id){set_error(error,error_size,"NONE codec has table id");return -1;}return 0;}if(!known_codec(codec)){set_error(error,error_size,"unknown codec %u",codec);return -1;}t=codec_table(p,table_id);if(!t||t->codec!=codec||(t->shard_id>=0&&(uint32_t)t->shard_id!=shard)){set_error(error,error_size,"invalid codec table reference %u",table_id);return -1;}return 0;
}

static int parse_codec_tables(ColiExecPackage *p,const unsigned char *m,size_t mn,uint64_t off,uint64_t bytes,uint32_t count,char *error,size_t error_size){
    uint32_t i,j;if(count>MAX_CODEC_TABLES||(uint64_t)count*CODEC_DESC_BYTES>bytes){set_error(error,error_size,"invalid codec-table region");return -1;}p->codec_table_count=count;if(!count){if(off||bytes){set_error(error,error_size,"empty codec table has nonzero region");return -1;}return 0;}p->codec_tables=(ExecCodecTable *)calloc(count,sizeof(*p->codec_tables));if(!p->codec_tables){set_error(error,error_size,"out of memory for codec tables");return -1;}
    for(i=0;i<count;i++){const unsigned char*d=m+off+(size_t)i*64;ExecCodecTable*t=&p->codec_tables[i];uint64_t end;t->id=rd32(d);t->codec=rd16(d+4);t->shard_id=rdi32(d+8);t->data_offset=rd64(d+16);t->data_bytes=rd64(d+24);t->data_crc32c=rd32(d+32);
        if(!t->id||!known_codec(t->codec)||t->codec==COLI_CSF_CODEC_NONE||rd16(d+6)||rd32(d+12)||rd32(d+36)||!all_zero(d+40,24)||t->shard_id< -1||(t->shard_id>=0&&(uint32_t)t->shard_id>=p->shard_count)||!t->data_bytes||t->data_offset%16||t->data_offset<(uint64_t)count*64||checked_add(t->data_offset,t->data_bytes,&end)||end>bytes){set_error(error,error_size,"invalid codec table descriptor %u",i);return -1;}
        for(j=0;j<i;j++)if(p->codec_tables[j].id==t->id){set_error(error,error_size,"duplicate codec table id %u",t->id);return -1;}
        if(coli_crc32c(m+off+t->data_offset,(size_t)t->data_bytes)!=t->data_crc32c){set_error(error,error_size,"bad codec table CRC id %u",t->id);return -1;}
    }
    (void)mn;return 0;
}

static int parse_shards(ColiExecPackage *p,const unsigned char *m,uint64_t off,uint64_t bytes,uint32_t count,char *error,size_t error_size){
    uint32_t i;if(count>MAX_SHARDS||bytes!=(uint64_t)count*SHARD_DESC_BYTES||!count){set_error(error,error_size,"invalid shard table size/count");return -1;}p->shard_count=count;p->shards=(ExecShard *)calloc(count,sizeof(*p->shards));if(!p->shards){set_error(error,error_size,"out of memory for shards");return -1;}for(i=0;i<count;i++)p->shards[i].fd=-1;
    for(i=0;i<count;i++){const unsigned char*d=m+off+(size_t)i*64;ExecShard*s=&p->shards[i];uint32_t name_id=rd32(d+8);struct stat st;unsigned char h[128];uint32_t crc;
        s->id=rd32(d);s->file_bytes=rd64(d+16);s->header_crc32c=rd32(d+24);
        if(s->id!=i||rd32(d+4)||rd32(d+12)||rd32(d+28)||!all_zero(d+32,32)||name_id>=p->string_count||!safe_filename(p->strings[name_id])){set_error(error,error_size,"invalid shard descriptor %u",i);return -1;}
        s->name=strdup(p->strings[name_id]);s->path=join_path(p->root,s->name);if(!s->name||!s->path){set_error(error,error_size,"out of memory for shard path");return -1;}s->fd=open(s->path,COMPAT_O_RDONLY);if(s->fd<0){set_error(error,error_size,"cannot open shard %s: %s",s->name,strerror(errno));return -1;}if(fstat(s->fd,&st)||st.st_size<0||(uint64_t)st.st_size!=s->file_bytes){set_error(error,error_size,"shard size mismatch: %s",s->name);return -1;}if(s->file_bytes<DATA_HEADER_BYTES||pread_full(s->fd,h,sizeof(h),0,error,error_size))return -1;
        if(memcmp(h,k_data_magic_exact,8)||rd16(h+8)!=1||rd16(h+10)!=1||rd32(h+12)!=DATA_HEADER_BYTES||rd32(h+16)||rd32(h+20)!=i||rd32(h+24)!=p->target.record_alignment||rd32(h+28)||rd64(h+32)!=s->file_bytes||memcmp(h+40,p->target.source_fingerprint,32)||!all_zero(h+76,52)){set_error(error,error_size,"invalid data-shard header: %s",s->name);return -1;}crc=rd32(h+72);memset(h+72,0,4);if(coli_crc32c(h,sizeof(h))!=crc||crc!=s->header_crc32c){set_error(error,error_size,"bad data-shard header CRC: %s",s->name);return -1;}
    }return 0;
}

static int record_layout_valid(const ColiExecRecordInfo *r){
    if(r->kind==COLI_CSF_REC_BLOB)return r->layout==COLI_EXEC_LAYOUT_NONE;
    if(r->kind==COLI_CSF_REC_EXPERT)return r->layout==COLI_EXEC_LAYOUT_MIXED;
    if(r->kind==COLI_CSF_REC_TENSOR)return target_layout(r->layout);
    if(r->kind==COLI_CSF_REC_LAYER_PACK_RESERVED)return target_layout(r->layout)||r->layout==COLI_EXEC_LAYOUT_MIXED;
    return 1;
}
static int parse_records(ColiExecPackage *p,const unsigned char*m,uint64_t off,uint64_t bytes,uint64_t count,char *error,size_t error_size){
    size_t i;Span *spans;if(count>MAX_RECORDS||bytes!=count*RECORD_DESC_BYTES){set_error(error,error_size,"invalid record table size/count");return -1;}p->record_count=(size_t)count;p->records=(ColiExecRecordInfo *)calloc(p->record_count?p->record_count:1,sizeof(*p->records));if(!p->records){set_error(error,error_size,"out of memory for records");return -1;}if(alloc_index(&p->id_index,&p->id_cap,p->record_count,error,error_size)||alloc_index(&p->name_index,&p->name_cap,p->record_count,error,error_size)||alloc_index(&p->expert_index,&p->expert_cap,p->record_count,error,error_size)||alloc_index(&p->layer_index,&p->layer_cap,p->record_count,error,error_size))return -1;
    spans=(Span *)calloc(p->record_count?p->record_count:1,sizeof(*spans));if(!spans){set_error(error,error_size,"out of memory for record spans");return -1;}
    for(i=0;i<p->record_count;i++){const unsigned char*d=m+off+i*96;ColiExecRecordInfo*r=&p->records[i];uint32_t name_id=rd32(d+24);uint64_t end;int known_kind;
        r->record_id=rd64(d);r->kind=rd16(d+8);r->codec=rd16(d+10);r->math_format=rd16(d+12);r->scale_format=rd16(d+14);r->layout=rd16(d+16);r->flags=rd16(d+18);r->shard_id=rd32(d+20);r->layer=rdi32(d+28);r->expert=rdi32(d+32);r->payload_offset=rd64(d+40);r->stored_bytes=rd64(d+48);r->resident_bytes=rd64(d+56);r->stored_crc32c=rd32(d+64);r->logical_crc32c=rd32(d+68);r->codec_table_id=rd32(d+72);
        known_kind=r->kind>=COLI_CSF_REC_TENSOR&&r->kind<=COLI_CSF_REC_BLOB;
        if(!r->record_id||r->shard_id>=p->shard_count||rd32(d+36)||rd32(d+76)||!all_zero(d+80,16)||(r->flags&0xff00u)||!r->stored_bytes||!r->resident_bytes||r->payload_offset%p->target.record_alignment||checked_add(r->payload_offset,r->stored_bytes,&end)||end>p->shards[r->shard_id].file_bytes){set_error(error,error_size,"invalid record descriptor %zu",i);free(spans);return -1;}
        if(!known_kind&&!optional_record(r)){set_error(error,error_size,"unknown required record kind %u",r->kind);free(spans);return -1;}
        if(known_kind&&(!known_codec(r->codec)||!known_math(r->math_format)||!known_scale(r->scale_format)||!record_layout_valid(r))){set_error(error,error_size,"invalid enum/layout combination for record %llu",(unsigned long long)r->record_id);free(spans);return -1;}
        if(name_id!=UINT32_MAX){if(name_id>=p->string_count){set_error(error,error_size,"record name id out of range");free(spans);return -1;}r->name=p->strings[name_id];}
        if(r->kind==COLI_CSF_REC_TENSOR){if(!r->name||r->expert!=-1||r->math_format==COLI_CSF_MATH_MIXED||r->scale_format==COLI_CSF_SCALE_MIXED||r->layout==COLI_EXEC_LAYOUT_MIXED){set_error(error,error_size,"invalid TENSOR record invariants");free(spans);return -1;}}
        if(r->kind==COLI_CSF_REC_EXPERT){if(r->layer<0||r->expert<0||r->codec!=COLI_CSF_CODEC_NONE||r->codec_table_id||r->math_format!=COLI_CSF_MATH_MIXED||r->scale_format!=COLI_CSF_SCALE_MIXED||r->layout!=COLI_EXEC_LAYOUT_MIXED){set_error(error,error_size,"invalid EXPERT record invariants");free(spans);return -1;}}
        if(r->kind==COLI_CSF_REC_BLOB){if(r->math_format!=COLI_CSF_MATH_NONE||r->scale_format!=COLI_CSF_SCALE_NONE||r->layout!=COLI_EXEC_LAYOUT_NONE||r->layer!=-1||r->expert!=-1||r->codec!=COLI_CSF_CODEC_NONE||r->codec_table_id||r->stored_bytes!=r->resident_bytes){set_error(error,error_size,"invalid BLOB record invariants");free(spans);return -1;}}
        if(r->kind==COLI_CSF_REC_LAYER_PACK_RESERVED&&r->layer<0){set_error(error,error_size,"invalid LAYER_PACK layer");free(spans);return -1;}
        if(known_kind&&codec_ref_ok(p,r->codec,r->codec_table_id,r->shard_id,error,error_size)){free(spans);return -1;}
        if(id_insert(p,i,error,error_size)||(r->name&&name_insert(p,i,error,error_size))||(r->kind==COLI_CSF_REC_EXPERT&&pair_insert(p->expert_index,p->expert_cap,p->records,i,0,error,error_size))||(r->kind==COLI_CSF_REC_LAYER_PACK_RESERVED&&pair_insert(p->layer_index,p->layer_cap,p->records,i,1,error,error_size))){free(spans);return -1;}
        spans[i].shard=r->shard_id;spans[i].begin=r->payload_offset;spans[i].end=end;spans[i].index=i;
    }
    {int cmp(const void*a,const void*b){const Span*x=(const Span*)a,*y=(const Span*)b;if(x->shard!=y->shard)return x->shard<y->shard?-1:1;if(x->begin!=y->begin)return x->begin<y->begin?-1:1;return 0;}qsort(spans,p->record_count,sizeof(*spans),cmp);}
    for(i=1;i<p->record_count;i++)if(spans[i].shard==spans[i-1].shard&&spans[i].begin<spans[i-1].end){set_error(error,error_size,"overlapping records %llu and %llu",(unsigned long long)p->records[spans[i-1].index].record_id,(unsigned long long)p->records[spans[i].index].record_id);free(spans);return -1;}free(spans);return 0;
}

static int package_record_index(const ColiExecPackage*p,const ColiExecRecordInfo*r,size_t*out){uintptr_t b,e,x,d;if(!p||!r||!p->records)return -1;b=(uintptr_t)p->records;e=b+p->record_count*sizeof(*p->records);x=(uintptr_t)r;if(x<b||x>=e)return -1;d=x-b;if(d%sizeof(*p->records))return -1;*out=(size_t)(d/sizeof(*p->records));return 0;}

int coli_exec_package_open_ex(ColiExecPackage **out,const char*path,const ColiRuntimeTarget*runtime,ColiExecChecksumPolicy policy,char*error,size_t error_size){
    ColiExecPackage*p=NULL;unsigned char*m=NULL;size_t mn=0;char*manifest_path=NULL;uint64_t record_count,shard_off,shard_bytes,record_off,record_bytes,string_off,string_bytes,codec_off,codec_bytes;uint32_t string_count,shard_count,codec_count;uint64_t end;
    if(!out||!path||!runtime){set_error(error,error_size,"invalid execution package open arguments");return -1;}*out=NULL;if(policy!=COLI_EXEC_CHECKSUM_MANIFEST_ONLY&&policy!=COLI_EXEC_CHECKSUM_RECORD_ON_READ){set_error(error,error_size,"invalid checksum policy");return -1;}
    p=(ColiExecPackage *)calloc(1,sizeof(*p));if(!p){set_error(error,error_size,"out of memory for execution package");return -1;}p->checksum_policy=policy;p->root=strdup(path);if(!p->root){set_error(error,error_size,"out of memory for package path");goto fail;}
    if(coli_target_read_package(path,&p->target,error,error_size)||coli_target_check_compatibility(&p->target,runtime,error,error_size))goto fail;
    manifest_path=join_path(path,"manifest.coli");if(!manifest_path){set_error(error,error_size,"out of memory for manifest path");goto fail;}if(read_file(manifest_path,&m,&mn,error,error_size))goto fail;
    if(mn<MANIFEST_HEADER_BYTES||memcmp(m,k_manifest_magic,8)||rd16(m+8)!=1||rd16(m+10)!=1||rd32(m+12)!=MANIFEST_HEADER_BYTES||rd32(m+20)!=0x01020304u||rd32(m+24)!=p->target.record_alignment){set_error(error,error_size,"invalid target manifest framing");goto fail;}
    record_count=rd64(m+32);shard_count=rd32(m+40);string_count=rd32(m+28);codec_count=rd32(m+160);shard_off=rd64(m+48);shard_bytes=rd64(m+56);record_off=rd64(m+64);record_bytes=rd64(m+72);string_off=rd64(m+80);string_bytes=rd64(m+88);codec_off=rd64(m+168);codec_bytes=rd64(m+176);
    if(record_count>MAX_RECORDS||shard_count>MAX_SHARDS||string_count>MAX_STRINGS||codec_count>MAX_CODEC_TABLES){set_error(error,error_size,"manifest count exceeds reader limit");goto fail;}
    if(checked_add(string_off,string_bytes,&end)||end>mn||!string_bytes||string_off%16||checked_add(shard_off,shard_bytes,&end)||end>mn||shard_off%16||checked_add(record_off,record_bytes,&end)||end>mn||record_off%16||(codec_bytes&&(checked_add(codec_off,codec_bytes,&end)||end>mn||codec_off%16))){set_error(error,error_size,"manifest table region outside file");goto fail;}
    if(parse_strings(p,m,string_off,string_bytes,string_count,error,error_size)||parse_shards(p,m,shard_off,shard_bytes,shard_count,error,error_size)||parse_codec_tables(p,m,mn,codec_off,codec_bytes,codec_count,error,error_size)||parse_records(p,m,record_off,record_bytes,record_count,error,error_size))goto fail;
    free(manifest_path);free(m);*out=p;return 0;
fail: free(manifest_path);free(m);coli_exec_package_close(p);return -1;
}
int coli_exec_package_open(ColiExecPackage **out,const char*path,const ColiRuntimeTarget*runtime,char*error,size_t error_size){return coli_exec_package_open_ex(out,path,runtime,COLI_EXEC_CHECKSUM_MANIFEST_ONLY,error,error_size);}

void coli_exec_package_close(ColiExecPackage*p){uint32_t i;if(!p)return;if(p->shards)for(i=0;i<p->shard_count;i++){if(p->shards[i].fd>=0)close(p->shards[i].fd);free(p->shards[i].name);free(p->shards[i].path);}if(p->strings)for(i=0;i<p->string_count;i++)free(p->strings[i]);free(p->strings);free(p->shards);free(p->codec_tables);free(p->records);free(p->id_index);free(p->name_index);free(p->expert_index);free(p->layer_index);coli_target_info_free(&p->target);free(p->root);free(p);}
const ColiTargetInfo *coli_exec_package_target(const ColiExecPackage*p){return p?&p->target:NULL;}
size_t coli_exec_package_record_count(const ColiExecPackage*p){return p?p->record_count:0;}
const ColiExecRecordInfo *coli_exec_package_record_at(const ColiExecPackage*p,size_t i){return p&&i<p->record_count?&p->records[i]:NULL;}
const ColiExecRecordInfo *coli_exec_package_record_by_id(const ColiExecPackage*p,uint64_t key){uint64_t h;size_t pos;if(!p||!key)return NULL;h=mix64(key)|1;pos=(size_t)h&(p->id_cap-1);while(p->id_index[pos].index_plus_one){size_t i=p->id_index[pos].index_plus_one-1;if(p->records[i].record_id==key)return &p->records[i];pos=(pos+1)&(p->id_cap-1);}return NULL;}
const ColiExecRecordInfo *coli_exec_package_record_by_name(const ColiExecPackage*p,const char*key){uint64_t h;size_t pos;if(!p||!key||!p->name_cap)return NULL;h=hash_string(key);pos=(size_t)h&(p->name_cap-1);while(p->name_index[pos].index_plus_one){size_t i=p->name_index[pos].index_plus_one-1;if(!strcmp(p->records[i].name,key))return &p->records[i];pos=(pos+1)&(p->name_cap-1);}return NULL;}
static const ColiExecRecordInfo *pair_lookup(const ColiExecPackage*p,const IndexSlot*slots,size_t cap,int32_t a,int32_t b,int layer_only){uint64_t h;size_t pos;if(!p||!slots||!cap)return NULL;h=hash_pair(a,layer_only?0:b);pos=(size_t)h&(cap-1);while(slots[pos].index_plus_one){size_t i=slots[pos].index_plus_one-1;if(p->records[i].layer==a&&(layer_only||p->records[i].expert==b))return &p->records[i];pos=(pos+1)&(cap-1);}return NULL;}
const ColiExecRecordInfo *coli_exec_package_expert(const ColiExecPackage*p,int32_t l,int32_t e){return pair_lookup(p,p?p->expert_index:NULL,p?p->expert_cap:0,l,e,0);}
const ColiExecRecordInfo *coli_exec_package_layer_pack(const ColiExecPackage*p,int32_t l){return pair_lookup(p,p?p->layer_index:NULL,p?p->layer_cap:0,l,0,1);}

int coli_exec_package_read_range(const ColiExecPackage*p,const ColiExecRecordInfo*r,uint64_t roff,void*dst,size_t bytes,char*error,size_t error_size){size_t idx;uint64_t end,abs;if(!p||!r||(!dst&&bytes)||package_record_index(p,r,&idx)){set_error(error,error_size,"invalid record/range arguments");return -1;}if(checked_add(roff,bytes,&end)||end>r->stored_bytes||checked_add(r->payload_offset,roff,&abs)){set_error(error,error_size,"record range outside stored bytes");return -1;}if(!bytes)return 0;return pread_full(p->shards[r->shard_id].fd,dst,bytes,abs,error,error_size);}
static int verify_crc(const ColiExecPackage*p,const ColiExecRecordInfo*r,char*error,size_t error_size){unsigned char buf[65536];uint64_t left=r->stored_bytes,off=0;uint32_t state=0xffffffffu;/* reuse public CRC by buffering entire chunks cannot combine final CRCs; implement streaming Castagnoli locally */static uint32_t table[256];static int ready=0;unsigned i,j;if(!ready){for(i=0;i<256;i++){uint32_t c=i;for(j=0;j<8;j++)c=(c>>1)^(0x82f63b78u&-(int)(c&1));table[i]=c;}ready=1;}while(left){size_t n=left>sizeof(buf)?sizeof(buf):(size_t)left;if(coli_exec_package_read_range(p,r,off,buf,n,error,error_size))return -1;for(i=0;i<n;i++)state=table[(state^buf[i])&255]^(state>>8);off+=n;left-=n;}state^=0xffffffffu;if(state!=r->stored_crc32c){set_error(error,error_size,"bad stored CRC for record %llu",(unsigned long long)r->record_id);return -1;}return 0;}
int coli_exec_package_read_record(const ColiExecPackage*p,const ColiExecRecordInfo*r,void*dst,size_t dst_bytes,char*error,size_t error_size){if(!r||r->stored_bytes>SIZE_MAX||dst_bytes<(size_t)r->stored_bytes){set_error(error,error_size,"destination too small for stored record");return -1;}if(coli_exec_package_read_range(p,r,0,dst,(size_t)r->stored_bytes,error,error_size))return -1;if(p->checksum_policy==COLI_EXEC_CHECKSUM_RECORD_ON_READ&&verify_crc(p,r,error,error_size))return -1;return 0;}

static int span_ok(uint64_t off,uint64_t stored,uint16_t codec,uint64_t record_bytes,uint64_t min,uint64_t *end){uint64_t e,slack=(codec==COLI_CSF_CODEC_NONE)?0:RANS_SLACK;if(off<min||off%16||!stored||checked_add(off,stored,&e)||checked_add(e,slack,&e)||e>record_bytes)return 0;*end=e;return 1;}
int coli_exec_package_tensor_info(const ColiExecPackage*p,const ColiExecRecordInfo*r,ColiExecTensorInfo*out,char*error,size_t error_size){unsigned char h[128];uint16_t rank;unsigned i;uint64_t end;if(!p||!r||!out||r->kind!=COLI_CSF_REC_TENSOR){set_error(error,error_size,"record is not a target TENSOR");return -1;}if(r->stored_bytes<TENSOR_HEADER_BYTES||coli_exec_package_read_range(p,r,0,h,sizeof(h),error,error_size))return -1;if(memcmp(h,k_tensor_magic,8)||rd16(h+8)!=1||rd16(h+10)!=1||rd32(h+12)!=128||rd16(h+18)||rd32(h+124)){set_error(error,error_size,"invalid target TENSOR envelope");return -1;}rank=rd16(h+16);if(rank>8){set_error(error,error_size,"target tensor rank exceeds 8");return -1;}memset(out,0,sizeof(*out));out->rank=rank;out->scale_block_rows=rd32(h+20);out->scale_block_columns=rd32(h+24);out->group_size=rd32(h+28);for(i=0;i<8;i++){out->dims[i]=rd64(h+32+i*8);if((i<rank&&!out->dims[i])||(i>=rank&&out->dims[i])){set_error(error,error_size,"invalid target tensor dims");return -1;}}out->data_offset=rd64(h+96);out->data_stored_bytes=rd64(h+104);out->data_resident_bytes=rd64(h+112);out->logical_crc32c=rd32(h+120);if(out->data_resident_bytes!=r->resident_bytes||!span_ok(out->data_offset,out->data_stored_bytes,r->codec,r->stored_bytes,TENSOR_HEADER_BYTES,&end)){set_error(error,error_size,"target tensor data span/resident size invalid");return -1;}if(r->codec==COLI_CSF_CODEC_NONE&&out->data_stored_bytes!=out->data_resident_bytes){set_error(error,error_size,"uncompressed target tensor stored/resident sizes differ");return -1;}if((r->flags&COLI_CSF_RECORD_F_HAS_LOGICAL_CRC32C)&&out->logical_crc32c!=r->logical_crc32c){set_error(error,error_size,"target tensor logical CRC fields disagree");return -1;}return 0;}

int coli_exec_package_expert_info(const ColiExecPackage*p,const ColiExecRecordInfo*r,ColiExecExpertInfo*out,char*error,size_t error_size){unsigned char h[EXPERT_TABLE_BYTES];uint64_t spans_b[6],spans_e[6],resident=0;unsigned i,j,nspan=0;if(!p||!r||!out||r->kind!=COLI_CSF_REC_EXPERT){set_error(error,error_size,"record is not target EXPERT");return -1;}if(r->stored_bytes<EXPERT_TABLE_BYTES||coli_exec_package_read_range(p,r,0,h,sizeof(h),error,error_size))return -1;if(memcmp(h,k_expert_magic,8)||rd16(h+8)!=1||rd16(h+10)!=1||rd32(h+12)!=64||rdi32(h+16)!=r->layer||rdi32(h+20)!=r->expert||rd16(h+24)!=3||rd16(h+26)||rd32(h+28)!=128||rd64(h+32)!=64||rd64(h+40)<EXPERT_TABLE_BYTES||rd64(h+40)%16||rd64(h+56)){set_error(error,error_size,"invalid target EXPERT envelope");return -1;}memset(out,0,sizeof(*out));out->layer=r->layer;out->expert=r->expert;out->logical_bytes=rd64(h+48);
    for(i=0;i<3;i++){const unsigned char*d=h+64+i*128;ColiExecMatrixInfo*m=&out->matrices[i];uint64_t end;m->role=rd16(d);m->math_format=rd16(d+4);m->scale_format=rd16(d+6);m->weight_codec=rd16(d+8);m->scale_codec=rd16(d+10);m->layout=rd16(d+12);m->rows=rd64(d+16);m->columns=rd64(d+24);m->scale_block_rows=rd32(d+32);m->scale_block_columns=rd32(d+36);m->weight_codec_table_id=rd32(d+40);m->scale_codec_table_id=rd32(d+44);m->weight_offset=rd64(d+48);m->weight_stored_bytes=rd64(d+56);m->weight_resident_bytes=rd64(d+64);m->scale_offset=rd64(d+72);m->scale_stored_bytes=rd64(d+80);m->scale_resident_bytes=rd64(d+88);m->logical_crc32c=rd32(d+96);m->group_size=rd32(d+104);
        if(m->role!=(uint16_t)(i+1)||rd16(d+2)||rd16(d+14)||rd32(d+100)||rd32(d+108)||!all_zero(d+112,16)||!known_math(m->math_format)||m->math_format==COLI_CSF_MATH_MIXED||!known_scale(m->scale_format)||m->scale_format==COLI_CSF_SCALE_MIXED||!known_codec(m->weight_codec)||!known_codec(m->scale_codec)||!target_layout(m->layout)||!m->rows||!m->columns||!m->weight_resident_bytes||codec_ref_ok(p,m->weight_codec,m->weight_codec_table_id,r->shard_id,error,error_size)){set_error(error,error_size,"invalid target expert matrix %u",i);return -1;}
        if(!span_ok(m->weight_offset,m->weight_stored_bytes,m->weight_codec,r->stored_bytes,rd64(h+40),&end)){set_error(error,error_size,"invalid target expert weight span %u",i);return -1;}spans_b[nspan]=m->weight_offset;spans_e[nspan++]=end;resident+=m->weight_resident_bytes;
        if(m->weight_codec==COLI_CSF_CODEC_NONE&&m->weight_stored_bytes!=m->weight_resident_bytes){set_error(error,error_size,"uncompressed expert weight stored/resident differ");return -1;}
        if(m->scale_format==COLI_CSF_SCALE_NONE){if(m->scale_codec!=COLI_CSF_CODEC_NONE||m->scale_codec_table_id||m->scale_offset||m->scale_stored_bytes||m->scale_resident_bytes||m->scale_block_rows||m->scale_block_columns){set_error(error,error_size,"absent expert scale has nonzero fields");return -1;}}
        else {if(!m->scale_resident_bytes||codec_ref_ok(p,m->scale_codec,m->scale_codec_table_id,r->shard_id,error,error_size)||!span_ok(m->scale_offset,m->scale_stored_bytes,m->scale_codec,r->stored_bytes,rd64(h+40),&end)){set_error(error,error_size,"invalid target expert scale span %u",i);return -1;}spans_b[nspan]=m->scale_offset;spans_e[nspan++]=end;resident+=m->scale_resident_bytes;if(m->scale_codec==COLI_CSF_CODEC_NONE&&m->scale_stored_bytes!=m->scale_resident_bytes){set_error(error,error_size,"uncompressed expert scale stored/resident differ");return -1;}}
    }
    for(i=0;i<nspan;i++)for(j=i+1;j<nspan;j++)if(spans_b[i]<spans_e[j]&&spans_b[j]<spans_e[i]){set_error(error,error_size,"overlapping target expert subranges");return -1;}if(resident!=r->resident_bytes){set_error(error,error_size,"expert resident byte sum mismatch");return -1;}return 0;}

int coli_exec_package_validate_record(const ColiExecPackage*p,const ColiExecRecordInfo*r,int verify_stored,char*error,size_t error_size){ColiExecTensorInfo ti;ColiExecExpertInfo ei;size_t idx;if(!p||!r||package_record_index(p,r,&idx)){set_error(error,error_size,"record does not belong to package");return -1;}if(r->kind==COLI_CSF_REC_TENSOR&&coli_exec_package_tensor_info(p,r,&ti,error,error_size))return -1;if(r->kind==COLI_CSF_REC_EXPERT&&coli_exec_package_expert_info(p,r,&ei,error,error_size))return -1;if(verify_stored&&verify_crc(p,r,error,error_size))return -1;return 0;}
int coli_exec_package_verify_all(const ColiExecPackage*p,char*error,size_t error_size){size_t i;if(!p){set_error(error,error_size,"package is null");return -1;}for(i=0;i<p->record_count;i++)if(coli_exec_package_validate_record(p,&p->records[i],1,error,error_size))return -1;return 0;}

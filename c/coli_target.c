#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "coli_target.h"
#include "coli_format.h"
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
#define TARGET_DESC_BYTES 256u
#define STRING_DESC_BYTES 16u
#define INTERNAL_ALIGNMENT 16u
#define MAX_MANIFEST_BYTES (1ULL << 30)
#define MAX_PROFILE_DATA_BYTES (64ULL << 20)
#define MANIFEST_F_SOURCE_VALID (1u << 0)
#define MANIFEST_F_ARTIFACT_VALID (1u << 1)
#define MANIFEST_F_TARGET_REQUIRED (1u << 16)
#define MANIFEST_REQUIRED_UNKNOWN_MASK 0xfffe0000u
#define TARGET_REQUIRED_UNKNOWN_MASK 0xffff0000u

static const unsigned char k_manifest_magic[8] =
    {0x43,0x4f,0x4c,0x49,0x0d,0x0a,0x1a,0x0a};
static const unsigned char k_target_magic[8] =
    {0x43,0x4f,0x4c,0x49,0x54,0x47,0x54,0x00};

typedef struct Region {
    uint64_t offset;
    uint64_t bytes;
    const char *name;
} Region;

typedef struct Sha256 {
    uint32_t h[8];
    uint64_t total;
    unsigned char block[64];
    size_t used;
} Sha256;

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

static int pread_full(int fd, void *dst, size_t bytes, uint64_t offset,
                      char *error, size_t error_size) {
    unsigned char *p = (unsigned char *)dst;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n;
        size_t chunk = bytes - done;
        if (chunk > 0x40000000u) chunk = 0x40000000u;
        if (offset > (uint64_t)INT64_MAX - done) {
            set_error(error, error_size, "manifest read offset exceeds signed 64-bit range");
            return -1;
        }
        do { n = pread(fd, p + done, chunk, (off_t)(offset + done)); }
        while (n < 0 && errno == EINTR);
        if (n < 0) {
            set_error(error, error_size, "manifest read failed: %s", strerror(errno));
            return -1;
        }
        if (!n) {
            set_error(error, error_size, "truncated manifest");
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

/* Small dependency-free SHA-256 for artifact identity. */
static uint32_t rotr32(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
static const uint32_t sha_k[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};
static void sha_transform(Sha256 *s, const unsigned char block[64]) {
    uint32_t w[64], a,b,c,d,e,f,g,h;
    unsigned i;
    for (i = 0; i < 16; ++i)
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) | block[i*4+3];
    for (i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a=s->h[0]; b=s->h[1]; c=s->h[2]; d=s->h[3];
    e=s->h[4]; f=s->h[5]; g=s->h[6]; h=s->h[7];
    for (i = 0; i < 64; ++i) {
        uint32_t S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t t1=h+S1+ch+sha_k[i]+w[i];
        uint32_t S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}
static void sha_init(Sha256 *s) {
    static const uint32_t init[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    memcpy(s->h,init,sizeof(init)); s->total=0; s->used=0;
}
static void sha_update(Sha256 *s, const void *data, size_t n) {
    const unsigned char *p=(const unsigned char *)data;
    s->total += n;
    while (n) {
        size_t take=64-s->used; if (take>n) take=n;
        memcpy(s->block+s->used,p,take); s->used+=take; p+=take; n-=take;
        if (s->used==64) { sha_transform(s,s->block); s->used=0; }
    }
}
static void sha_final(Sha256 *s, unsigned char out[32]) {
    uint64_t bits=s->total*8; unsigned i;
    s->block[s->used++]=0x80;
    if (s->used>56) { while(s->used<64)s->block[s->used++]=0; sha_transform(s,s->block); s->used=0; }
    while(s->used<56)s->block[s->used++]=0;
    for(i=0;i<8;i++) s->block[63-i]=(unsigned char)(bits>>(i*8));
    sha_transform(s,s->block);
    for(i=0;i<8;i++) { out[i*4]=(unsigned char)(s->h[i]>>24); out[i*4+1]=(unsigned char)(s->h[i]>>16); out[i*4+2]=(unsigned char)(s->h[i]>>8); out[i*4+3]=(unsigned char)s->h[i]; }
}
static void sha256(const void *data, size_t n, unsigned char out[32]) {
    Sha256 s; sha_init(&s); sha_update(&s,data,n); sha_final(&s,out);
}
static void sha_u16(Sha256 *s, uint16_t v) { unsigned char b[2]={(unsigned char)v,(unsigned char)(v>>8)}; sha_update(s,b,2); }
static void sha_u32(Sha256 *s, uint32_t v) { unsigned char b[4]={(unsigned char)v,(unsigned char)(v>>8),(unsigned char)(v>>16),(unsigned char)(v>>24)}; sha_update(s,b,4); }
static void sha_u64(Sha256 *s, uint64_t v) { unsigned char b[8]; unsigned i; for(i=0;i<8;i++)b[i]=(unsigned char)(v>>(i*8)); sha_update(s,b,8); }
static int sha_string(Sha256 *s, const char *v, char *error, size_t error_size) {
    size_t n;
    if (!v) { set_error(error,error_size,"artifact identity string is missing"); return -1; }
    n=strlen(v); if(n>UINT32_MAX){set_error(error,error_size,"artifact identity string is too long");return -1;}
    sha_u32(s,(uint32_t)n); sha_update(s,v,n); return 0;
}

static int region(Region *r, uint64_t offset, uint64_t bytes, const char *name,
                  size_t file_bytes, int required, char *error, size_t error_size) {
    uint64_t end;
    r->offset=offset; r->bytes=bytes; r->name=name;
    if (!bytes) {
        if (required || offset) { set_error(error,error_size,"%s is missing/malformed",name); return -1; }
        return 0;
    }
    if (!offset || offset%INTERNAL_ALIGNMENT || checked_add(offset,bytes,&end) || end>file_bytes || offset<MANIFEST_HEADER_BYTES) {
        set_error(error,error_size,"%s is outside/alignment of manifest",name); return -1;
    }
    return 0;
}
static int no_overlap(const Region *r, size_t n, char *error, size_t error_size) {
    size_t i,j;
    for(i=0;i<n;i++) if(r[i].bytes) {
        uint64_t ae=r[i].offset+r[i].bytes;
        for(j=i+1;j<n;j++) if(r[j].bytes) {
            uint64_t be=r[j].offset+r[j].bytes;
            if(r[i].offset<be && r[j].offset<ae){set_error(error,error_size,"manifest regions %s and %s overlap",r[i].name,r[j].name);return -1;}
        }
    }
    return 0;
}

static int utf8_valid(const unsigned char *s, size_t n) {
    size_t i=0;
    while(i<n){unsigned c=s[i++],need,min2=0x80,max2=0xbf;if(c<0x80)continue;
        if(c>=0xc2&&c<=0xdf)need=1;else if(c>=0xe0&&c<=0xef){need=2;if(c==0xe0)min2=0xa0;if(c==0xed)max2=0x9f;}
        else if(c>=0xf0&&c<=0xf4){need=3;if(c==0xf0)min2=0x90;if(c==0xf4)max2=0x8f;}else return 0;
        if(i+need>n||s[i]<min2||s[i]>max2)return 0;i++;while(--need){if(s[i]<0x80||s[i]>0xbf)return 0;i++;}}
    return 1;
}

static int parse_strings(const unsigned char *m, const Region *sr, uint32_t count,
                         char ***out, char *error, size_t error_size) {
    uint64_t desc=(uint64_t)count*STRING_DESC_BYTES; uint32_t i; char **strings;
    if(desc>sr->bytes){set_error(error,error_size,"string descriptors exceed table");return -1;}
    strings=(char **)calloc(count?count:1,sizeof(*strings)); if(!strings){set_error(error,error_size,"out of memory for strings");return -1;}
    for(i=0;i<count;i++){const unsigned char *d=m+sr->offset+(size_t)i*16;uint64_t off=rd64(d),end;uint32_t len=rd32(d+8);
        if(rd32(d+12)||off<desc||checked_add(off,len,&end)||end>sr->bytes){set_error(error,error_size,"invalid string descriptor %u",i);goto fail;}
        if(memchr(m+sr->offset+off,0,len)||!utf8_valid(m+sr->offset+off,len)){set_error(error,error_size,"invalid UTF-8/NUL in string %u",i);goto fail;}
        strings[i]=(char *)malloc((size_t)len+1);if(!strings[i]){set_error(error,error_size,"out of memory for string");goto fail;}
        memcpy(strings[i],m+sr->offset+off,len);strings[i][len]='\0';
    }
    *out=strings;return 0;
fail: for(i=0;i<count;i++)free(strings[i]);free(strings);return -1;
}
static char *dup_id(char **strings,uint32_t count,uint32_t id,const char *field,char *error,size_t error_size){
    char *p;if(id>=count){set_error(error,error_size,"%s string id out of range",field);return NULL;}p=strdup(strings[id]);if(!p)set_error(error,error_size,"out of memory duplicating %s",field);return p;
}

int coli_target_artifact_fingerprint(const ColiTargetInfo *i, uint8_t out[32], char *error, size_t error_size) {
    Sha256 s; unsigned char pd[32]; unsigned char zero_tuning[32]={0};
    static const char tag[]="COLI-ARTIFACT-V1\0";
    if(!i||!out){set_error(error,error_size,"invalid artifact fingerprint arguments");return -1;}
    sha_init(&s);sha_update(&s,tag,sizeof(tag)-1);sha_update(&s,i->source_fingerprint,32);
    if(sha_string(&s,i->compiler,error,error_size)||sha_string(&s,i->semantic_abi,error,error_size)||
       sha_string(&s,i->profile_name,error,error_size)||sha_string(&s,i->quant_profile,error,error_size)||
       sha_string(&s,i->storage_profile,error,error_size)||sha_string(&s,i->optimization_profile,error,error_size)||
       sha_string(&s,i->kernel_profile,error,error_size)||sha_string(&s,i->target_triple,error,error_size)) return -1;
    sha_u32(&s,i->flags);sha_u16(&s,i->target_os);sha_u16(&s,i->target_arch);sha_u16(&s,i->backend);sha_u16(&s,i->gpu_kind);
    sha_u64(&s,i->cpu_feature_mask);sha_u32(&s,i->gpu_family_min);sha_u32(&s,i->gpu_family_max);sha_u32(&s,i->gpu_capability_min);sha_u32(&s,i->gpu_capability_max);
    sha_u32(&s,i->target_profile_abi);sha_u32(&s,i->execution_layout_abi);sha_u32(&s,i->kernel_abi_min);sha_u32(&s,i->kernel_abi_max);
    sha_u32(&s,i->record_alignment);sha_u32(&s,i->io_granularity);sha_u32(&s,i->resident_alignment);sha_u64(&s,i->required_runtime_features);
    { unsigned char tv=(i->flags&COLI_TARGET_F_TUNING_FINGERPRINT_VALID)?1:0;sha_update(&s,&tv,1); }
    sha_update(&s,(i->flags&COLI_TARGET_F_TUNING_FINGERPRINT_VALID)?i->tuning_fingerprint:zero_tuning,32);
    sha_u64(&s,i->profile_data_bytes);sha256(i->profile_data,i->profile_data_bytes,pd);sha_update(&s,pd,32);sha_final(&s,out);return 0;
}

void coli_target_info_free(ColiTargetInfo *i) {
    if(!i)return;free(i->profile_name);free(i->quant_profile);free(i->storage_profile);free(i->optimization_profile);free(i->kernel_profile);free(i->target_triple);free(i->semantic_abi);free(i->compiler);free(i->profile_data);memset(i,0,sizeof(*i));
}

int coli_target_read_package(const char *package_path, ColiTargetInfo *out, char *error, size_t error_size) {
    char *path=NULL;int fd=-1;struct stat st;unsigned char *m=NULL,*tmp=NULL;size_t n=0;uint32_t flags,string_count,manifest_crc,target_crc;uint64_t t_off,t_bytes,pd_off,pd_bytes;Region regs[7];char **strings=NULL;uint32_t i;const unsigned char *t;unsigned char recomputed[32];int rc=-1;
    if(!package_path||!out){set_error(error,error_size,"invalid target package arguments");return -1;}memset(out,0,sizeof(*out));
    path=join_path(package_path,"manifest.coli");if(!path){set_error(error,error_size,"out of memory building manifest path");goto done;}
    fd=open(path,COMPAT_O_RDONLY);if(fd<0){set_error(error,error_size,"cannot open target manifest: %s",strerror(errno));goto done;}
    if(fstat(fd,&st)||st.st_size<(off_t)MANIFEST_HEADER_BYTES||(uint64_t)st.st_size>MAX_MANIFEST_BYTES){set_error(error,error_size,"target manifest size is invalid");goto done;}n=(size_t)st.st_size;
    m=(unsigned char *)malloc(n);if(!m){set_error(error,error_size,"out of memory reading target manifest");goto done;}if(pread_full(fd,m,n,0,error,error_size))goto done;
    if(memcmp(m,k_manifest_magic,8)||rd16(m+8)!=1||rd16(m+10)!=COLI_CSF_TARGET_VERSION_MINOR||rd32(m+12)!=MANIFEST_HEADER_BYTES||rd32(m+20)!=0x01020304u){set_error(error,error_size,"unsupported/non-target CSF manifest version");goto done;}
    flags=rd32(m+16);if((flags&(MANIFEST_F_SOURCE_VALID|MANIFEST_F_ARTIFACT_VALID|MANIFEST_F_TARGET_REQUIRED))!=(MANIFEST_F_SOURCE_VALID|MANIFEST_F_ARTIFACT_VALID|MANIFEST_F_TARGET_REQUIRED)||(flags&MANIFEST_REQUIRED_UNKNOWN_MASK)){set_error(error,error_size,"missing/unknown required target manifest feature");goto done;}
    if(!power2(rd32(m+24))||rd32(m+44)||rd32(m+156)||rd32(m+164)||rd32(m+236)||!all_zero(m+240,16)){set_error(error,error_size,"invalid target manifest alignment/reserved fields");goto done;}
    if(all_zero(m+112,32)||all_zero(m+200,32)){set_error(error,error_size,"required source/artifact fingerprint is zero");goto done;}
    tmp=(unsigned char *)malloc(n);if(!tmp){set_error(error,error_size,"out of memory verifying manifest CRC");goto done;}memcpy(tmp,m,n);manifest_crc=rd32(m+144);memset(tmp+144,0,4);if(coli_crc32c(tmp,n)!=manifest_crc){set_error(error,error_size,"bad manifest CRC");goto done;}free(tmp);tmp=NULL;
    string_count=rd32(m+28);t_off=rd64(m+184);t_bytes=rd64(m+192);if(t_bytes!=TARGET_DESC_BYTES){set_error(error,error_size,"target descriptor size is not 256");goto done;}
    if(region(&regs[0],rd64(m+48),rd64(m+56),"shard table",n,0,error,error_size)||region(&regs[1],rd64(m+64),rd64(m+72),"record table",n,0,error,error_size)||region(&regs[2],rd64(m+80),rd64(m+88),"string table",n,1,error,error_size)||region(&regs[3],rd64(m+96),rd64(m+104),"metadata",n,0,error,error_size)||region(&regs[4],rd64(m+168),rd64(m+176),"codec table",n,0,error,error_size)||region(&regs[5],t_off,t_bytes,"target descriptor",n,1,error,error_size)||no_overlap(regs,6,error,error_size))goto done;
    if(parse_strings(m,&regs[2],string_count,&strings,error,error_size))goto done;
    t=m+t_off;if(memcmp(t,k_target_magic,8)||rd16(t+8)!=1||rd16(t+10)!=0||rd32(t+12)!=TARGET_DESC_BYTES){set_error(error,error_size,"bad target descriptor magic/version");goto done;}
    out->flags=rd32(t+16);if(out->flags&TARGET_REQUIRED_UNKNOWN_MASK){set_error(error,error_size,"unknown required target feature");goto done;}
    out->target_os=rd16(t+20);out->target_arch=rd16(t+22);out->backend=rd16(t+24);out->gpu_kind=rd16(t+26);out->cpu_feature_mask=rd64(t+28);out->gpu_family_min=rd32(t+36);out->gpu_family_max=rd32(t+40);out->gpu_capability_min=rd32(t+44);out->gpu_capability_max=rd32(t+48);out->target_profile_abi=rd32(t+52);out->execution_layout_abi=rd32(t+56);out->kernel_abi_min=rd32(t+60);out->kernel_abi_max=rd32(t+64);out->record_alignment=rd32(t+68);out->io_granularity=rd32(t+72);out->resident_alignment=rd32(t+76);out->required_runtime_features=rd64(t+80);
    if(out->target_os<1||out->target_os>3||out->target_arch<1||out->target_arch>2||out->backend<1||out->backend>4||out->gpu_kind>2||!power2(out->record_alignment)||!power2(out->io_granularity)||!power2(out->resident_alignment)||out->record_alignment!=rd32(m+24)||!out->target_profile_abi||!out->execution_layout_abi||!out->kernel_abi_min){set_error(error,error_size,"invalid target capability fields");goto done;}
    if((out->flags&COLI_TARGET_F_ACCELERATOR_REQUIRED)&&out->gpu_kind==COLI_TARGET_GPU_NONE){set_error(error,error_size,"accelerator required but gpu kind is NONE");goto done;}
    if(out->gpu_kind==COLI_TARGET_GPU_APPLE_FAMILY&&(!out->gpu_family_min||out->gpu_capability_min||out->gpu_capability_max)){set_error(error,error_size,"invalid Apple GPU-family target range");goto done;}
    if(out->gpu_kind==COLI_TARGET_GPU_CUDA_SM&&(!out->gpu_capability_min||out->gpu_family_min||out->gpu_family_max)){set_error(error,error_size,"invalid CUDA capability target range");goto done;}
    if(out->gpu_family_max&&out->gpu_family_max<out->gpu_family_min){set_error(error,error_size,"invalid GPU family min/max");goto done;}if(out->gpu_capability_max&&out->gpu_capability_max<out->gpu_capability_min){set_error(error,error_size,"invalid GPU capability min/max");goto done;}if(out->kernel_abi_max&&out->kernel_abi_max<out->kernel_abi_min){set_error(error,error_size,"invalid kernel ABI min/max");goto done;}
    if(!(out->flags&COLI_TARGET_F_TUNING_FINGERPRINT_VALID)&&!all_zero(t+112,32)){set_error(error,error_size,"tuning fingerprint set without valid flag");goto done;}memcpy(out->tuning_fingerprint,t+112,32);
    if(!all_zero(t+168,88)){set_error(error,error_size,"target descriptor reserved bytes are nonzero");goto done;}
    target_crc=rd32(m+232);if(coli_crc32c(t,TARGET_DESC_BYTES)!=target_crc){set_error(error,error_size,"bad target descriptor CRC");goto done;}
    out->profile_name=dup_id(strings,string_count,rd32(t+88),"profile",error,error_size);if(!out->profile_name)goto done;
    out->quant_profile=dup_id(strings,string_count,rd32(t+92),"quant profile",error,error_size);if(!out->quant_profile)goto done;
    out->storage_profile=dup_id(strings,string_count,rd32(t+96),"storage profile",error,error_size);if(!out->storage_profile)goto done;
    out->optimization_profile=dup_id(strings,string_count,rd32(t+100),"optimization profile",error,error_size);if(!out->optimization_profile)goto done;
    out->kernel_profile=dup_id(strings,string_count,rd32(t+104),"kernel profile",error,error_size);if(!out->kernel_profile)goto done;
    out->target_triple=dup_id(strings,string_count,rd32(t+108),"target triple",error,error_size);if(!out->target_triple)goto done;
    out->semantic_abi=dup_id(strings,string_count,rd32(t+164),"semantic ABI",error,error_size);if(!out->semantic_abi)goto done;
    out->compiler=dup_id(strings,string_count,rd32(m+152),"compiler",error,error_size);if(!out->compiler)goto done;
    if(rd32(m+148)>=string_count||strcmp(strings[rd32(m+148)],out->profile_name)){set_error(error,error_size,"manifest/target profile name mismatch");goto done;}
    pd_off=rd64(t+144);pd_bytes=rd64(t+152);if(pd_bytes>MAX_PROFILE_DATA_BYTES){set_error(error,error_size,"profile data exceeds parser limit");goto done;}if(!pd_bytes){if(pd_off||rd32(t+160)){set_error(error,error_size,"empty profile data has nonzero offset/CRC");goto done;}}
    else { if(region(&regs[6],pd_off,pd_bytes,"profile data",n,1,error,error_size)||no_overlap(regs,7,error,error_size))goto done;out->profile_data=(uint8_t *)malloc((size_t)pd_bytes);if(!out->profile_data){set_error(error,error_size,"out of memory for profile data");goto done;}memcpy(out->profile_data,m+pd_off,(size_t)pd_bytes);out->profile_data_bytes=(size_t)pd_bytes;if(coli_crc32c(out->profile_data,out->profile_data_bytes)!=rd32(t+160)){set_error(error,error_size,"bad profile-data CRC");goto done;} }
    memcpy(out->source_fingerprint,m+112,32);memcpy(out->artifact_fingerprint,m+200,32);
    if(coli_target_artifact_fingerprint(out,recomputed,error,error_size))goto done;if(memcmp(recomputed,out->artifact_fingerprint,32)){set_error(error,error_size,"bad artifact fingerprint");goto done;}
    rc=0;
done:
    if(strings){for(i=0;i<string_count;i++)free(strings[i]);free(strings);}if(fd>=0)close(fd);free(path);free(m);free(tmp);if(rc)coli_target_info_free(out);return rc;
}

static int same_string(const char *a,const char *b){return a&&b&&!strcmp(a,b);}
int coli_target_check_compatibility(const ColiTargetInfo *r,const ColiRuntimeTarget *v,char *error,size_t error_size){
    if(!r||!v){set_error(error,error_size,"invalid target compatibility arguments");return -1;}
    if(r->target_os!=v->target_os){set_error(error,error_size,"target OS mismatch: required=%u runtime=%u",r->target_os,v->target_os);return -1;}
    if(r->target_arch!=v->target_arch){set_error(error,error_size,"target arch mismatch: required=%u runtime=%u",r->target_arch,v->target_arch);return -1;}
    if(r->backend!=v->backend){set_error(error,error_size,"backend mismatch: required=%u runtime=%u",r->backend,v->backend);return -1;}
    if((r->cpu_feature_mask&v->cpu_feature_mask)!=r->cpu_feature_mask){set_error(error,error_size,"missing required CPU features: required=0x%llx runtime=0x%llx",(unsigned long long)r->cpu_feature_mask,(unsigned long long)v->cpu_feature_mask);return -1;}
    if((r->required_runtime_features&v->runtime_features)!=r->required_runtime_features){set_error(error,error_size,"missing required runtime features: required=0x%llx runtime=0x%llx",(unsigned long long)r->required_runtime_features,(unsigned long long)v->runtime_features);return -1;}
    if((r->flags&COLI_TARGET_F_ACCELERATOR_REQUIRED)&&r->gpu_kind!=v->gpu_kind){set_error(error,error_size,"GPU kind mismatch: required=%u runtime=%u",r->gpu_kind,v->gpu_kind);return -1;}
    if(r->gpu_kind==COLI_TARGET_GPU_APPLE_FAMILY){if(v->gpu_family<r->gpu_family_min||(r->gpu_family_max&&v->gpu_family>r->gpu_family_max)){set_error(error,error_size,"Apple GPU family unsupported: required=%u..%u runtime=%u",r->gpu_family_min,r->gpu_family_max,v->gpu_family);return -1;}}
    if(r->gpu_kind==COLI_TARGET_GPU_CUDA_SM){if(v->gpu_capability<r->gpu_capability_min||(r->gpu_capability_max&&v->gpu_capability>r->gpu_capability_max)){set_error(error,error_size,"CUDA capability unsupported: required=%u..%u runtime=%u",r->gpu_capability_min,r->gpu_capability_max,v->gpu_capability);return -1;}}
    if(!same_string(r->semantic_abi,v->semantic_abi)){set_error(error,error_size,"semantic ABI mismatch: required=%s runtime=%s",r->semantic_abi,v->semantic_abi?v->semantic_abi:"(null)");return -1;}
    if(!same_string(r->profile_name,v->profile_name)){set_error(error,error_size,"target profile mismatch: required=%s runtime=%s",r->profile_name,v->profile_name?v->profile_name:"(null)");return -1;}
    if(!same_string(r->quant_profile,v->quant_profile)){set_error(error,error_size,"quant profile mismatch: required=%s runtime=%s",r->quant_profile,v->quant_profile?v->quant_profile:"(null)");return -1;}
    if(!same_string(r->storage_profile,v->storage_profile)){set_error(error,error_size,"storage profile mismatch: required=%s runtime=%s",r->storage_profile,v->storage_profile?v->storage_profile:"(null)");return -1;}
    if(r->target_profile_abi!=v->target_profile_abi){set_error(error,error_size,"target profile ABI mismatch: required=%u runtime=%u",r->target_profile_abi,v->target_profile_abi);return -1;}
    if(r->execution_layout_abi!=v->execution_layout_abi){set_error(error,error_size,"execution layout ABI mismatch: required=%u runtime=%u",r->execution_layout_abi,v->execution_layout_abi);return -1;}
    if(v->kernel_abi<r->kernel_abi_min||(r->kernel_abi_max&&v->kernel_abi>r->kernel_abi_max)){set_error(error,error_size,"kernel ABI unsupported: required=%u..%u runtime=%u",r->kernel_abi_min,r->kernel_abi_max,v->kernel_abi);return -1;}
    if(v->max_record_alignment<r->record_alignment){set_error(error,error_size,"record alignment unsupported: required=%u runtime-max=%u",r->record_alignment,v->max_record_alignment);return -1;}
    if(v->max_io_granularity<r->io_granularity){set_error(error,error_size,"I/O granularity unsupported: required=%u runtime-max=%u",r->io_granularity,v->max_io_granularity);return -1;}
    if(v->max_resident_alignment<r->resident_alignment){set_error(error,error_size,"resident alignment unsupported: required=%u runtime-max=%u",r->resident_alignment,v->max_resident_alignment);return -1;}
    return 0;
}

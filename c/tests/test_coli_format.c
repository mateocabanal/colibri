#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "../coli_format.h"
#include "../coli_executor.h"
#include "../compat.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
#define ALIGNMENT 4096u

static void wr16(unsigned char *p, uint16_t v) { p[0]=(unsigned char)v; p[1]=(unsigned char)(v>>8); }
static void wr32(unsigned char *p, uint32_t v) { p[0]=(unsigned char)v; p[1]=(unsigned char)(v>>8); p[2]=(unsigned char)(v>>16); p[3]=(unsigned char)(v>>24); }
static void wr64(unsigned char *p, uint64_t v) { wr32(p,(uint32_t)v); wr32(p+4,(uint32_t)(v>>32)); }

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static unsigned char *read_hex(const char *path, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    unsigned char *buf = NULL;
    size_t n = 0, cap = 0;
    int hi = -1, c;
    if (!f) return NULL;
    while ((c = fgetc(f)) != EOF) {
        int v = hexval(c);
        if (v < 0) continue;
        if (hi < 0) hi = v;
        else {
            if (n == cap) {
                size_t nc = cap ? cap * 2 : 256;
                unsigned char *nb = (unsigned char *)realloc(buf, nc);
                if (!nb) { free(buf); fclose(f); return NULL; }
                buf = nb; cap = nc;
            }
            buf[n++] = (unsigned char)((hi << 4) | v); hi = -1;
        }
    }
    fclose(f);
    if (hi >= 0) { free(buf); return NULL; }
    *out_n = n; return buf;
}

static int write_file(const char *path, const void *data, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (n && fwrite(data, 1, n, f) != n) { fclose(f); return -1; }
    return fclose(f) ? -1 : 0;
}

static char *join2(const char *a, const char *b) {
    size_t na=strlen(a), nb=strlen(b);
    char *p=(char*)malloc(na+nb+2);
    if (!p) return NULL;
    memcpy(p,a,na); p[na]='/'; memcpy(p+na+1,b,nb+1); return p;
}

static int make_temp(char out[128]) {
    strcpy(out, "csf_reader_test_XXXXXX");
    return mkdtemp(out) ? 0 : -1;
}

static void cleanup_dir(const char *dir, int nshards) {
    int i; char name[64]; char *p;
    p=join2(dir,"manifest.coli"); if(p){ unlink(p); free(p); }
    for(i=0;i<nshards;i++) { snprintf(name,sizeof(name),"data-%05d.coli",i); p=join2(dir,name); if(p){ unlink(p); free(p); } }
    rmdir(dir);
}

static int install_hand_fixture(const char *dir) {
    size_t mn=0,dn=0; unsigned char *m,*d; char *mp,*dp; int rc;
    m=read_hex("tests/fixtures/csf-v1-tiny/manifest.coli.hex",&mn);
    d=read_hex("tests/fixtures/csf-v1-tiny/data-00000.coli.hex",&dn);
    if(!m||!d){ free(m); free(d); return -1; }
    mp=join2(dir,"manifest.coli"); dp=join2(dir,"data-00000.coli");
    if(!mp||!dp){ free(m);free(d);free(mp);free(dp);return -1; }
    rc=write_file(mp,m,mn)||write_file(dp,d,dn);
    free(m);free(d);free(mp);free(dp); return rc?-1:0;
}

static int rewrite_manifest_crc(unsigned char *m, size_t n) {
    uint32_t crc;
    if(n<148) return -1;
    memset(m+144,0,4); crc=coli_crc32c(m,n); wr32(m+144,crc); return 0;
}

static int rewrite_data_header_crc(unsigned char *d, size_t n) {
    uint32_t crc;
    if(n<128) return -1;
    memset(d+72,0,4); crc=coli_crc32c(d,128); wr32(d+72,crc); return 0;
}

static int write_hand_manifest_mutation(const char *dir, void (*mut)(unsigned char*,size_t), int recalc_crc) {
    size_t n=0; unsigned char *m=read_hex("tests/fixtures/csf-v1-tiny/manifest.coli.hex",&n); char *p; int rc;
    if(!m) return -1; mut(m,n); if(recalc_crc && rewrite_manifest_crc(m,n)){free(m);return -1;}
    p=join2(dir,"manifest.coli"); if(!p){free(m);return -1;} rc=write_file(p,m,n); free(m);free(p); return rc;
}

static void mut_overflow(unsigned char *m,size_t n){(void)n;wr64(m+320+40,UINT64_MAX-7);wr64(m+320+48,16);}
static void mut_outside(unsigned char *m,size_t n){(void)n;wr64(m+320+40,8192);}
static void mut_major(unsigned char *m,size_t n){(void)n;wr16(m+8,2);}
static void mut_bad_manifest_crc(unsigned char *m,size_t n){(void)n;m[144]^=1;}
static void mut_bad_record_crc(unsigned char *m,size_t n){(void)n;wr32(m+320+64,0x12345678u);}
static void mut_path(unsigned char *m,size_t n){const char s[15]="../evil123.coli";(void)n;memcpy(m+416+64,s,15);}

static int build_expert_package(const char *dir, int count) {
    size_t rec_bytes=529, last_off=(size_t)ALIGNMENT*(size_t)count;
    size_t data_bytes=last_off+rec_bytes;
    size_t string_off=320u+(size_t)count*96u, string_bytes=96u, manifest_bytes=string_off+string_bytes;
    unsigned char *data=(unsigned char*)calloc(1,data_bytes), *m=(unsigned char*)calloc(1,manifest_bytes);
    const char *fn="data-00000.coli", *profile="portable-v1", *compiler="test-reader";
    int k,j; char *mp=NULL,*dp=NULL; uint32_t hcrc,mcrc;
    if(!data||!m||count<1||count>2) goto fail;
    memcpy(data,"COLIDAT\0",8); wr16(data+8,1); wr16(data+10,0); wr32(data+12,128); wr32(data+20,0); wr32(data+24,ALIGNMENT); wr64(data+32,data_bytes);
    for(k=0;k<count;k++) {
        size_t off=(size_t)ALIGNMENT*(size_t)(k+1), b=off;
        unsigned char *r=data+off;
        memcpy(r,"COLIEXPT",8); wr16(r+8,1); wr16(r+10,0); wr32(r+12,64); wr32(r+16,2); wr32(r+20,7); wr16(r+24,3); wr32(r+28,128); wr64(r+32,64); wr64(r+40,448); wr64(r+48,51);
        for(j=0;j<3;j++) {
            unsigned char *q=r+64+(size_t)j*128; size_t wo=448+(size_t)j*32, so=wo+16; int z;
            wr16(q,(uint16_t)(j+1)); wr16(q+4,COLI_CSF_MATH_MXFP4_E2M1); wr16(q+6,COLI_CSF_SCALE_UE8M0);
            wr16(q+8,COLI_CSF_CODEC_NONE); wr16(q+10,COLI_CSF_CODEC_NONE); wr16(q+12,COLI_CSF_LAYOUT_CANONICAL);
            wr64(q+16,1); wr64(q+24,32); wr32(q+32,1); wr32(q+36,32);
            wr64(q+48,wo); wr64(q+56,16); wr64(q+64,16); wr64(q+72,so); wr64(q+80,1); wr64(q+88,1);
            for(z=0;z<16;z++) r[wo+(size_t)z]=(unsigned char)(j*16+z+1);
            r[so]=(unsigned char)(0x7f+j); wr32(q+96,coli_crc32c(r+wo,17));
        }
        (void)b;
    }
    rewrite_data_header_crc(data,data_bytes); hcrc=(uint32_t)data[72]|((uint32_t)data[73]<<8)|((uint32_t)data[74]<<16)|((uint32_t)data[75]<<24);
    memcpy(m,"COLI\r\n\x1a\n",8); wr16(m+8,1); wr16(m+10,0); wr32(m+12,256); wr32(m+20,0x01020304u); wr32(m+24,ALIGNMENT); wr32(m+28,3); wr64(m+32,(uint64_t)count); wr32(m+40,1);
    wr64(m+48,256); wr64(m+56,64); wr64(m+64,320); wr64(m+72,(uint64_t)count*96); wr64(m+80,string_off); wr64(m+88,string_bytes); wr32(m+148,1); wr32(m+152,2);
    wr32(m+256,0); wr32(m+256+8,0); wr64(m+256+16,data_bytes); wr32(m+256+24,hcrc);
    for(k=0;k<count;k++) {
        unsigned char *q=m+320+(size_t)k*96; size_t off=(size_t)ALIGNMENT*(size_t)(k+1);
        wr64(q,(uint64_t)k+1); wr16(q+8,COLI_CSF_REC_EXPERT); wr16(q+10,COLI_CSF_CODEC_NONE); wr16(q+12,COLI_CSF_MATH_MIXED); wr16(q+14,COLI_CSF_SCALE_MIXED); wr16(q+16,COLI_CSF_LAYOUT_MIXED); wr32(q+20,0); wr32(q+24,UINT32_MAX); wr32(q+28,2); wr32(q+32,7); wr64(q+40,off); wr64(q+48,rec_bytes); wr64(q+56,51); wr32(q+64,coli_crc32c(data+off,rec_bytes));
    }
    {
        unsigned char *s=m+string_off; size_t desc=48;
        wr64(s,desc);wr32(s+8,15); wr64(s+16,desc+15);wr32(s+24,11); wr64(s+32,desc+26);wr32(s+40,11);
        memcpy(s+desc,fn,15); memcpy(s+desc+15,profile,11); memcpy(s+desc+26,compiler,11);
    }
    rewrite_manifest_crc(m,manifest_bytes); mcrc=(uint32_t)m[144]|((uint32_t)m[145]<<8)|((uint32_t)m[146]<<16)|((uint32_t)m[147]<<24); (void)mcrc;
    mp=join2(dir,"manifest.coli");dp=join2(dir,"data-00000.coli"); if(!mp||!dp)goto fail;
    if(write_file(mp,m,manifest_bytes)||write_file(dp,data,data_bytes))goto fail;
    free(m);free(data);free(mp);free(dp);return 0;
fail:
    free(m);free(data);free(mp);free(dp);return -1;
}

static int build_two_shard_package(const char *dir) {
    const size_t dbytes=4225, mbytes=736, soff=576, sbytes=160;
    unsigned char *d0=(unsigned char*)calloc(1,dbytes),*d1=(unsigned char*)calloc(1,dbytes),*m=(unsigned char*)calloc(1,mbytes);
    const char *strs[6]={"data-00000.coli","data-00001.coli","a","b","portable-v1","test-reader"};
    uint32_t hcrc[2]; int s; char *p=NULL;
    if(!d0||!d1||!m)goto fail;
    for(s=0;s<2;s++) {
        unsigned char *d=s?d1:d0,*r=d+4096; uint32_t crc;
        memcpy(d,"COLIDAT\0",8);wr16(d+8,1);wr32(d+12,128);wr32(d+20,(uint32_t)s);wr32(d+24,ALIGNMENT);wr64(d+32,dbytes);
        memcpy(r,"COLITENS",8);wr16(r+8,1);wr32(r+12,128);wr16(r+16,2);wr64(r+32,1);wr64(r+40,1);wr64(r+96,128);wr64(r+104,1);wr64(r+112,1);r[128]=(unsigned char)(0x2a+s);crc=coli_crc32c(r+128,1);wr32(r+120,crc);
        rewrite_data_header_crc(d,dbytes); hcrc[s]=(uint32_t)d[72]|((uint32_t)d[73]<<8)|((uint32_t)d[74]<<16)|((uint32_t)d[75]<<24);
    }
    memcpy(m,"COLI\r\n\x1a\n",8);wr16(m+8,1);wr32(m+12,256);wr32(m+20,0x01020304u);wr32(m+24,ALIGNMENT);wr32(m+28,6);wr64(m+32,2);wr32(m+40,2);
    wr64(m+48,256);wr64(m+56,128);wr64(m+64,384);wr64(m+72,192);wr64(m+80,soff);wr64(m+88,sbytes);wr32(m+148,4);wr32(m+152,5);
    for(s=0;s<2;s++){unsigned char*q=m+256+s*64;wr32(q,(uint32_t)s);wr32(q+8,(uint32_t)s);wr64(q+16,dbytes);wr32(q+24,hcrc[s]);}
    for(s=0;s<2;s++){unsigned char*q=m+384+s*96;unsigned char*d=s?d1:d0;wr64(q,(uint64_t)s+1);wr16(q+8,COLI_CSF_REC_TENSOR);wr16(q+12,COLI_CSF_MATH_U8);wr32(q+20,(uint32_t)s);wr32(q+24,(uint32_t)s+2);wr32(q+28,UINT32_MAX);wr32(q+32,UINT32_MAX);wr64(q+40,4096);wr64(q+48,129);wr64(q+56,1);wr32(q+64,coli_crc32c(d+4096,129));wr32(q+68,coli_crc32c(d+4224,1));wr16(q+18,COLI_CSF_RECORD_F_HAS_LOGICAL_CRC32C);}
    {
        unsigned char*q=m+soff;size_t off=96;int i;
        for(i=0;i<6;i++){size_t n=strlen(strs[i]);wr64(q+i*16,off);wr32(q+i*16+8,(uint32_t)n);memcpy(q+off,strs[i],n);off+=n;}
    }
    rewrite_manifest_crc(m,mbytes);
    p=join2(dir,"manifest.coli");if(!p||write_file(p,m,mbytes))goto fail;free(p);p=join2(dir,"data-00000.coli");if(!p||write_file(p,d0,dbytes))goto fail;free(p);p=join2(dir,"data-00001.coli");if(!p||write_file(p,d1,dbytes))goto fail;
    free(p);free(d0);free(d1);free(m);return 0;
fail: free(p);free(d0);free(d1);free(m);return -1;
}

typedef struct ReadThread { const ColiPackage *p; const ColiRecordInfo *r; int fail; } ReadThread;
static void *read_worker(void *arg) {
    ReadThread *t=(ReadThread*)arg; int i; char err[128];
    for(i=0;i<2000;i++){unsigned char b=0;if(coli_package_read_range(t->p,t->r,128,&b,1,err,sizeof(err))||b!=0x2a){t->fail=1;break;}}
    return NULL;
}

static int test_hand_fixture(void) {
    char dir[128],err[256];ColiPackage*p=NULL;const ColiRecordInfo*r;ColiTensorInfo ti;unsigned char b=0;pthread_t th[4];ReadThread ctx[4];int i;
    CHECK(make_temp(dir)==0);CHECK(install_hand_fixture(dir)==0);CHECK(coli_package_open(&p,dir,err,sizeof(err))==0);CHECK(p!=NULL);CHECK(coli_package_record_count(p)==1);CHECK(strcmp(coli_package_profile(p),"portable-v1")==0);CHECK(strcmp(coli_package_compiler(p),"hand-fixture")==0);
    r=coli_package_record_by_name(p,"tiny.weight");CHECK(r&&r==coli_package_record_by_id(p,1));CHECK(coli_package_tensor_info(p,r,&ti,err,sizeof(err))==0);CHECK(ti.rank==2&&ti.dims[0]==1&&ti.dims[1]==1&&ti.data_offset==128);CHECK(coli_package_read_range(p,r,128,&b,1,err,sizeof(err))==0&&b==0x2a);b=0;CHECK(coli_package_read_range_ex(p,r,128,&b,1,COLI_CSF_READ_UNCACHED,err,sizeof(err))==0&&b==0x2a);CHECK(coli_package_read_range_ex(p,r,128,&b,1,0x80000000u,err,sizeof(err))!=0);CHECK(coli_package_verify_all(p,err,sizeof(err))==0);
    for(i=0;i<4;i++){ctx[i].p=p;ctx[i].r=r;ctx[i].fail=0;CHECK(pthread_create(&th[i],NULL,read_worker,&ctx[i])==0);}for(i=0;i<4;i++){CHECK(pthread_join(th[i],NULL)==0);CHECK(!ctx[i].fail);}coli_package_close(p);cleanup_dir(dir,1);return 0;
}

static int test_expert_and_duplicate(void) {
    char dir[128],err[256];ColiPackage*p=NULL;const ColiRecordInfo*r;ColiExpertInfo ei;
    CHECK(make_temp(dir)==0);CHECK(build_expert_package(dir,1)==0);CHECK(coli_package_open(&p,dir,err,sizeof(err))==0);r=coli_package_expert(p,2,7);CHECK(r&&r->kind==COLI_CSF_REC_EXPERT);CHECK(coli_package_expert_info(p,r,&ei,err,sizeof(err))==0);CHECK(ei.logical_bytes==51);CHECK(ei.matrices[0].role==1&&ei.matrices[2].role==3);CHECK(ei.matrices[1].weight_decoded_bytes==16&&ei.matrices[1].scale_decoded_bytes==1);CHECK(coli_package_verify_all(p,err,sizeof(err))==0);coli_package_close(p);cleanup_dir(dir,1);
    CHECK(make_temp(dir)==0);CHECK(build_expert_package(dir,2)==0);CHECK(coli_package_open(&p,dir,err,sizeof(err))!=0);CHECK(p==NULL);cleanup_dir(dir,1);return 0;
}

static int test_executor(void) {
    char dir[128],err[256]; ColiExecutor *executor=NULL; ColiExecutorOpenOptions opt;
    const ColiRecordInfo *record; unsigned char *resident;
    CHECK(make_temp(dir)==0); CHECK(build_expert_package(dir,1)==0);
    memset(&opt,0,sizeof(opt)); opt.required_profile=COLI_CSF_PROFILE_PORTABLE_V1;
    opt.checksum_policy=COLI_CSF_CHECKSUM_RECORD_ON_READ;
    CHECK(coli_executor_open(&executor,dir,&opt,err,sizeof(err))==0);
    record=coli_executor_expert(executor,2,7); CHECK(record&&record->stored_bytes==529);
    CHECK(coli_executor_expert(executor,2,8)==NULL);
    resident=(unsigned char*)malloc((size_t)record->stored_bytes); CHECK(resident);
    CHECK(coli_executor_load_expert(executor,2,7,resident,(size_t)record->stored_bytes-1,err,sizeof(err))!=0);
    CHECK(coli_executor_load_expert(executor,2,7,resident,(size_t)record->stored_bytes,err,sizeof(err))==0);
    CHECK(!memcmp(resident,"COLIEXPT",8)); CHECK(resident[448]==1 && resident[464]==0x7f);
    free(resident); coli_executor_close(executor); executor=NULL;
    opt.required_profile=COLI_CSF_PROFILE_MACOS_ARM64_METAL_APPLE8_V1;
    CHECK(coli_executor_open(&executor,dir,&opt,err,sizeof(err))!=0); CHECK(executor==NULL);
    cleanup_dir(dir,1); return 0;
}

static int test_multishard(void) {
    char dir[128],err[256];ColiPackage*p=NULL;unsigned char b;const ColiRecordInfo*r;
    CHECK(make_temp(dir)==0);CHECK(build_two_shard_package(dir)==0);CHECK(coli_package_open(&p,dir,err,sizeof(err))==0);CHECK(coli_package_record_count(p)==2);r=coli_package_record_by_name(p,"b");CHECK(r&&r->shard_id==1);CHECK(coli_package_read_range(p,r,128,&b,1,err,sizeof(err))==0&&b==0x2b);CHECK(coli_package_verify_all(p,err,sizeof(err))==0);coli_package_close(p);cleanup_dir(dir,2);return 0;
}

static int expect_mutation_rejected(void (*mut)(unsigned char*,size_t),int recalc,int truncate) {
    char dir[128],err[256];ColiPackage*p=NULL;char*mp;int rc;
    CHECK(make_temp(dir)==0);CHECK(install_hand_fixture(dir)==0);
    if(truncate){unsigned char x[100]={0};mp=join2(dir,"manifest.coli");CHECK(mp);CHECK(write_file(mp,x,sizeof(x))==0);free(mp);}else CHECK(write_hand_manifest_mutation(dir,mut,recalc)==0);
    rc=coli_package_open(&p,dir,err,sizeof(err));CHECK(rc!=0);CHECK(p==NULL);cleanup_dir(dir,1);return 0;
}

static int test_record_crc_policy(void) {
    char dir[128],err[256];ColiPackage*p=NULL;const ColiRecordInfo*r;
    CHECK(make_temp(dir)==0);CHECK(install_hand_fixture(dir)==0);CHECK(write_hand_manifest_mutation(dir,mut_bad_record_crc,1)==0);CHECK(coli_package_open(&p,dir,err,sizeof(err))==0);r=coli_package_record_by_id(p,1);CHECK(r);CHECK(coli_package_validate_record(p,r,1,err,sizeof(err))!=0);coli_package_close(p);cleanup_dir(dir,1);return 0;
}

static int test_random_corruption(void) {
    size_t mn=0,dn=0;unsigned char *orig=read_hex("tests/fixtures/csf-v1-tiny/manifest.coli.hex",&mn),*d=read_hex("tests/fixtures/csf-v1-tiny/data-00000.coli.hex",&dn);int i;
    CHECK(orig&&d);
    for(i=0;i<128;i++){
        char dir[128],*mp,*dp,err[128];unsigned char *m=(unsigned char*)malloc(mn);ColiPackage*p=NULL;size_t pos=(size_t)((i*1103515245u+12345u)%mn);CHECK(m);memcpy(m,orig,mn);m[pos]^=(unsigned char)(1u<<(i&7));CHECK(make_temp(dir)==0);mp=join2(dir,"manifest.coli");dp=join2(dir,"data-00000.coli");CHECK(mp&&dp);CHECK(write_file(mp,m,mn)==0);CHECK(write_file(dp,d,dn)==0);if(coli_package_open(&p,dir,err,sizeof(err))==0)coli_package_close(p);free(m);free(mp);free(dp);cleanup_dir(dir,1);
    }
    free(orig);free(d);return 0;
}

int main(void) {
    CHECK(coli_crc32c("123456789",9)==0xe3069283u);
    CHECK(test_hand_fixture()==0);
    CHECK(test_expert_and_duplicate()==0);
    CHECK(test_executor()==0);
    CHECK(test_multishard()==0);
    CHECK(expect_mutation_rejected(mut_major,0,0)==0);
    CHECK(expect_mutation_rejected(mut_bad_manifest_crc,0,0)==0);
    CHECK(expect_mutation_rejected(mut_overflow,1,0)==0);
    CHECK(expect_mutation_rejected(mut_outside,1,0)==0);
    CHECK(expect_mutation_rejected(mut_path,1,0)==0);
    CHECK(expect_mutation_rejected(NULL,0,1)==0);
    CHECK(test_record_crc_policy()==0);
    CHECK(test_random_corruption()==0);
    puts("test_coli_format: ok");
    return 0;
}

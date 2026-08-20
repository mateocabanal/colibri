
static int ascii_ieq(const char *a,const char *b){
    if(!a||!b)return 0;
    while(*a&&*b){
        unsigned char x=(unsigned char)*a++,y=(unsigned char)*b++;
        if(x>='A'&&x<='Z')x=(unsigned char)(x+32);
        if(y>='A'&&y<='Z')y=(unsigned char)(y+32);
        if(x!=y)return 0;
    }
    return *a==0&&*b==0;
}
static inline uint64_t coli_prefix_hash64(const void *data,size_t bytes,uint64_t seed){
    const unsigned char *p=(const unsigned char*)data; uint64_t h=seed?seed:PFX_FNV_OFFSET; for(size_t i=0;i<bytes;i++){h^=p[i];h*=PFX_FNV_PRIME;} return h;
}
static uint64_t token_hash_next(uint64_t h,int token){uint32_t u=(uint32_t)token; unsigned char b[4]={(unsigned char)u,(unsigned char)(u>>8),(unsigned char)(u>>16),(unsigned char)(u>>24)}; return coli_prefix_hash64(b,4,h);}
static uint64_t token_hash(const int *t,int n){uint64_t h=PFX_FNV_OFFSET; for(int i=0;i<n;i++)h=token_hash_next(h,t[i]); return h;}
static uint32_t crc32c_update(uint32_t crc,const void *data,size_t n){const unsigned char *p=(const unsigned char*)data; for(size_t i=0;i<n;i++){crc^=p[i]; for(int b=0;b<8;b++)crc=(crc>>1)^(UINT32_C(0x82f63b78)&(uint32_t)-(int32_t)(crc&1));} return crc;}
static int join_path(char *out,size_t cap,const char *dir,const char *name){if(!out||!cap||!dir||!name)return 0; size_t a=strlen(dir),b=strlen(name),sep=(a&&dir[a-1]!='/'&&dir[a-1]!='\\'); if(a>cap-1||b>cap-1-a||sep>cap-1-a-b)return 0; memcpy(out,dir,a); if(sep)out[a++]='/'; memcpy(out+a,name,b+1); return 1;}
static int mkdir_one(const char *p){
#ifdef _WIN32
    return _mkdir(p)==0||errno==EEXIST;
#else
    return mkdir(p,0700)==0||errno==EEXIST;
#endif
}
static int mkdirs(const char *path){if(!path||!*path||strlen(path)>=COLI_PREFIX_DISK_PATH_MAX)return 0; char tmp[COLI_PREFIX_DISK_PATH_MAX]; strcpy(tmp,path); size_t n=strlen(tmp); while(n>1&&(tmp[n-1]=='/'||tmp[n-1]=='\\'))tmp[--n]=0; for(char *p=tmp+1;*p;p++){if(*p!='/'&&*p!='\\')continue;
#ifdef _WIN32
    if(p==tmp+2&&tmp[1]==':')continue;
#endif
    char c=*p;*p=0;if(*tmp&&!mkdir_one(tmp))return 0;*p=c;} return mkdir_one(tmp);}
static int sync_file(FILE *fp){if(!fp||fflush(fp)!=0)return 0;
#ifdef _WIN32
    return _commit(_fileno(fp))==0;
#else
    return fsync(fileno(fp))==0;
#endif

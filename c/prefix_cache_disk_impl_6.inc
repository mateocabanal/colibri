}
static uint64_t free_bytes(const char *dir,int *valid){if(valid)*valid=0;
#ifdef _WIN32
    ULARGE_INTEGER b; if(!GetDiskFreeSpaceExA(dir,&b,NULL,NULL))return 0; if(valid)*valid=1; return (uint64_t)b.QuadPart;
#else
    struct statvfs st; if(statvfs(dir,&st)!=0)return 0; if(valid)*valid=1; if(st.f_bavail&&(uint64_t)st.f_frsize>UINT64_MAX/(uint64_t)st.f_bavail)return UINT64_MAX; return (uint64_t)st.f_bavail*(uint64_t)st.f_frsize;
#endif
}
static void model_tag(const uint8_t fp[32],char out[65]){static const char h[]="0123456789abcdef"; for(int i=0;i<32;i++){out[2*i]=h[fp[i]>>4];out[2*i+1]=h[fp[i]&15];}out[64]=0;}
static int object_path(const ColiPrefixDiskCache *c,int n,uint64_t hash,char *out,size_t cap){char tag[65],name[320];model_tag(c->model_fingerprint,tag);int w=snprintf(name,sizeof(name),"cpfx-%08" PRIx32 "-%08" PRIx32 "-%016" PRIx64 "-%s-%08x-%016" PRIx64 ".bin",c->engine_id,c->state_abi,c->layout_fingerprint,tag,(unsigned)n,hash); return w>0&&(size_t)w<sizeof(name)&&join_path(out,cap,c->directory,name);}
static int is_object_name(const char *n){size_t z;if(!n||strncmp(n,"cpfx-",5))return 0;z=strlen(n);return z>9&&!strcmp(n+z-4,".bin");}
static int prune_cmp(const void *a,const void *b){const PruneEntry*x=(const PruneEntry*)a,*y=(const PruneEntry*)b;if(x->mtime<y->mtime)return -1;if(x->mtime>y->mtime)return 1;return strcmp(x->path,y->path);}
static int prune_for(ColiPrefixDiskCache *c,size_t incoming){if(!c||!c->budget_bytes||incoming>c->budget_bytes)return 0; DIR*d=opendir(c->directory); if(!d){c->resident_bytes=0;return 1;} PruneEntry*items=NULL;size_t count=0,cap=0;uint64_t total=0;struct dirent*de; while((de=readdir(d))){if(!is_object_name(de->d_name))continue;PruneEntry e;memset(&e,0,sizeof(e));if(!join_path(e.path,sizeof(e.path),c->directory,de->d_name))continue;struct stat st;if(stat(e.path,&st)!=0||st.st_size<0)continue;e.bytes=(uint64_t)st.st_size;e.mtime=st.st_mtime;total=UINT64_MAX-total<e.bytes?UINT64_MAX:total+e.bytes;if(count==cap){size_t nc=cap?cap*2:64;PruneEntry*g=(PruneEntry*)realloc(items,nc*sizeof(*items));if(!g){free(items);closedir(d);return 0;}items=g;cap=nc;}items[count++]=e;}closedir(d);c->resident_bytes=total; if(total<=c->budget_bytes&&incoming<=c->budget_bytes-(size_t)total){free(items);return 1;} qsort(items,count,sizeof(*items),prune_cmp);for(size_t i=0;i<count;i++){if(total<=c->budget_bytes&&incoming<=c->budget_bytes-(size_t)total)break;if(remove(items[i].path)==0){total=items[i].bytes<=total?total-items[i].bytes:0;c->resident_bytes=total;c->evictions++;}}free(items);return total<=c->budget_bytes&&incoming<=c->budget_bytes-(size_t)total;}
static void cleanup_temps(const char *dir){DIR*d=opendir(dir);if(!d)return;struct dirent*de;char p[COLI_PREFIX_DISK_PATH_MAX];while((de=readdir(d))){if(!strstr(de->d_name,".tmp."))continue;if(join_path(p,sizeof(p),dir,de->d_name))(void)remove(p);}closedir(d);}
static int fingerprint_nonzero(const uint8_t *fp){unsigned char any=0;for(int i=0;i<32;i++)any|=fp[i];return any!=0;}
static size_t parse_scaled(const char *name,size_t fallback,long double scale,int allow_zero){const char*v=getenv(name);if(!v||!*v)return fallback;char*end=NULL;long double x=strtold(v,&end);if(end==v||x<0.0L||(!allow_zero&&!(x>0.0L)))return 0;while(*end==' '||*end=='\t'||*end=='\r'||*end=='\n')end++;if(*end)return 0;long double bytes=x*scale;if(bytes>=(long double)SIZE_MAX)return SIZE_MAX;return(size_t)bytes;}
static size_t min_free_from_env(void){return parse_scaled("COLI_PREFIX_CACHE_MIN_FREE_GB",(size_t)(PFX_DEFAULT_FREE_GIB*1024.0L*1024.0L*1024.0L),1024.0L*1024.0L*1024.0L,1);}

static inline int coli_prefix_disk_init(ColiPrefixDiskCache*c,const char*dir,size_t budget,int min_tokens,uint32_t engine,uint32_t abi,const uint8_t fp[32],uint64_t layout,int log){if(!c)return 0;memset(c,0,sizeof(*c));c->budget_bytes=budget;c->min_free_bytes=min_free_from_env();c->min_tokens=min_tokens>0?min_tokens:1;c->engine_id=engine;c->state_abi=abi;c->layout_fingerprint=layout;c->log=!!log;if(!budget||!dir||!*dir||!fp||!fingerprint_nonzero(fp))return 1;if(strlen(dir)>=sizeof(c->directory))return 0;memcpy(c->model_fingerprint,fp,32);strcpy(c->directory,dir);if(!mkdirs(c->directory))return 0;c->enabled=1;PrefixLock l;
#ifndef _WIN32
l.fd=-1;
#endif
if(lock_acquire(c->directory,&l)){cleanup_temps(c->directory);(void)prune_for(c,0);lock_release(&l);}if(c->log)fprintf(stderr,"[PREFIX-SSD] dir=%s budget=%.2fGiB min_tokens=%d engine=%u abi=%u\n",c->directory,(double)c->budget_bytes/(1024.0*1024.0*1024.0),c->min_tokens,c->engine_id,c->state_abi);return 1;}

typedef struct{FILE*fp;uint32_t crc;size_t written,expected;int failed;} WriteSink;

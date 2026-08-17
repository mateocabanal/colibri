#include "coli_v4_expert_store.h"
#include "coli_executor.h"
#include "compat.h"
#ifdef COLI_METAL
#include "backend_metal.h"
#endif

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
/* Consumed by the V4-local pread interposition in coli_v4_macos_uncached_io.h.
 * Loader threads set this only while reading a routed-expert record. */
__thread int coli_v4_expert_io_active;
#endif

typedef struct { const ColiRecordInfo *record; ColiExpertInfo info; } Record;
/* A slot remains unavailable from selection until its positioned read and CRC
 * have completed. refs alone is insufficient: loader lanes acquire before a
 * view exists, so two lanes could otherwise write the same resident buffer. */
typedef struct { int expert; unsigned refs, loading; unsigned char *data; } Slot;
typedef struct {
    ColiExecutor *executor; int layers, experts, slots_per_layer;
    uint64_t record_bytes, slot_bytes, clock; Record *records; Slot *slots;
    unsigned active_leases; ColiExpertStoreStats stats; pthread_mutex_t mutex;

    /* End-user progress + I/O diagnostics. These counters are guarded by mutex.
     * They intentionally live in the expert store so they measure actual cache
     * misses / storage work rather than inferred model progress. */
    time_t io_started_at, io_last_report_at;
    uint64_t io_reads, io_bytes;
    unsigned io_inflight, io_peak_inflight;
    int progress_enabled, progress_interval_s;
} State;

static int fail(char *e,size_t n,const char *f,...) { va_list a; if(e&&n){va_start(a,f);vsnprintf(e,n,f,a);va_end(a);} return -1; }
static Record *record_for(State *s, ColiExpertKey k) {
    if(k.layer<0||k.layer>=s->layers||k.expert<0||k.expert>=s->experts) return NULL;
    return &s->records[(size_t)k.layer*s->experts+k.expert];
}
static Slot *slots_for(State*s,int layer) { return s->slots+(size_t)layer*s->slots_per_layer; }

static void print_execution_mode(void) {
#ifdef COLI_METAL
    const char *setting = getenv("V4_METAL_EXPERTS");
    const int metal_requested = (!setting || !*setting) ? 1 : (atoi(setting) != 0);
    const int metal_ready = metal_requested && coli_metal_init() && coli_metal_available();
    if (metal_ready) {
        /* V4's Metal helper accelerates MXFP4 small-batch matvecs. Routing,
         * control flow, unsupported kernels and any failed Metal dispatch still
         * execute on the CPU, so calling this mode "hybrid" is deliberate. */
        fprintf(stderr,
                "v4_execution mode=hybrid cpu=control+unsupported-kernels+fallback "
                "gpu=metal-mxfp4 metal=available fallback=cpu\n");
    } else if (!metal_requested) {
        fprintf(stderr,
                "v4_execution mode=cpu gpu=disabled reason=V4_METAL_EXPERTS=0\n");
    } else {
        fprintf(stderr,
                "v4_execution mode=cpu gpu=unavailable requested=metal fallback=cpu\n");
    }
#else
    fprintf(stderr,
            "v4_execution mode=cpu gpu=not-built reason=COLI_METAL-disabled\n");
#endif
}

/* Called with s->mutex held. Progress is on by default but deliberately waits
 * for the first real expert miss, so fast/cache-hot runs do not gain startup
 * noise. V4_PROGRESS=0 disables it; V4_PROGRESS_INTERVAL controls cadence. */
static void io_begin_locked(State *s, ColiExpertKey key) {
    const time_t now = time(NULL);
    if (!s->io_started_at) {
        s->io_started_at = now;
        s->io_last_report_at = now;
        if (s->progress_enabled) {
            fprintf(stderr,
                    "v4_progress phase=expert-stream status=started "
                    "record=%.2fMiB cache_slots_per_layer=%d layer=%d expert=%d\n",
                    (double)s->record_bytes / (1024.0 * 1024.0),
                    s->slots_per_layer, key.layer, key.expert);
        }
    }
    s->io_inflight++;
    if (s->io_inflight > s->io_peak_inflight) s->io_peak_inflight = s->io_inflight;
}

/* Called with s->mutex held after a storage operation. */
static void io_finish_locked(State *s, ColiExpertKey key, int success) {
    const time_t now = time(NULL);
    if (s->io_inflight) s->io_inflight--;
    if (success) {
        s->io_reads++;
        s->io_bytes += s->record_bytes;
    }
    if (!s->progress_enabled || !s->io_started_at ||
        now - s->io_last_report_at < s->progress_interval_s)
        return;

    double elapsed = difftime(now, s->io_started_at);
    if (elapsed < 1.0) elapsed = 1.0;
    const double gib = (double)s->io_bytes / (1024.0 * 1024.0 * 1024.0);
    const double mib_s = ((double)s->io_bytes / (1024.0 * 1024.0)) / elapsed;
    const double hit_pct = s->stats.requests
        ? 100.0 * (double)s->stats.hits / (double)s->stats.requests : 0.0;
    fprintf(stderr,
            "v4_progress phase=expert-stream elapsed=%.0fs reads=%llu bytes=%.2fGiB "
            "avg=%.1fMiB/s inflight=%u peak_inflight=%u cache_hit=%.1f%% "
            "layer=%d/%d expert=%d\n",
            elapsed, (unsigned long long)s->io_reads, gib, mib_s,
            s->io_inflight, s->io_peak_inflight, hit_pct,
            key.layer + 1, s->layers, key.expert);
    s->io_last_report_at = now;
}

static int tensor_format(const ColiExpertMatrixInfo *m, ColiTensorView *v, const unsigned char *data) {
    if(m->math_format!=COLI_CSF_MATH_MXFP4_E2M1 || m->scale_format!=COLI_CSF_SCALE_UE8M0 ||
       m->layout!=COLI_CSF_LAYOUT_CANONICAL || m->weight_codec!=COLI_CSF_CODEC_NONE ||
       m->scale_codec!=COLI_CSF_CODEC_NONE || m->rows>(uint64_t)INT64_MAX || m->columns>(uint64_t)INT64_MAX ||
       m->weight_stored_bytes>SIZE_MAX || m->scale_stored_bytes>SIZE_MAX) return -1;
    memset(v,0,sizeof(*v)); v->format=COLI_TENSOR_FP4_NATIVE_BLOCK; v->scale_format=COLI_SCALE_UE8M0;
    v->data=data+m->weight_offset; v->scales=data+m->scale_offset;
    v->data_bytes=(size_t)m->weight_stored_bytes; v->scale_bytes=(size_t)m->scale_stored_bytes;
    v->rows=(int64_t)m->rows; v->columns=(int64_t)m->columns;
    v->block_rows=m->scale_block_rows; v->block_columns=m->scale_block_columns; return 0;
}
static int fill(ColiExpertView *v, ColiExpertKey k, const Record *r, const Slot *s) {
    const ColiExpertMatrixInfo *gate=NULL,*up=NULL,*down=NULL;
    for(int i=0;i<3;i++) { const ColiExpertMatrixInfo*m=&r->info.matrices[i]; if(m->role==1)gate=m; else if(m->role==2)up=m; else if(m->role==3)down=m; }
    if(!gate||!up||!down||tensor_format(gate,&v->gate,s->data)||tensor_format(up,&v->up,s->data)||tensor_format(down,&v->down,s->data)) return -1;
    v->key=k; v->lease=(void*)s; return 0;
}
static int lookup(ColiExpertStore *store, ColiExpertKey key, ColiExpertView *view) {
    State*s=store?store->state:NULL; Record*r=s?record_for(s,key):NULL; Slot*slot=NULL;
    if(!s||!r||!view){if(view)memset(view,0,sizeof(*view));return -1;}
    pthread_mutex_lock(&s->mutex); s->stats.requests++; Slot *slots=slots_for(s,key.layer);
    for(int i=0;i<s->slots_per_layer;i++) if(!slots[i].loading&&slots[i].data&&slots[i].expert==key.expert){slot=&slots[i];s->stats.hits++;break;}
    if(!slot) { for(int i=0;i<s->slots_per_layer;i++) if(!slots[i].refs&&!slots[i].loading&&(!slot||!slots[i].data)) slot=&slots[i];
        if(!slot) { pthread_mutex_unlock(&s->mutex); memset(view,0,sizeof(*view)); return -1; }
        if(!slot->data) {
            if(s->slot_bytes>SIZE_MAX||posix_memalign((void**)&slot->data,16384,(size_t)s->slot_bytes)){pthread_mutex_unlock(&s->mutex);memset(view,0,sizeof(*view));return -1;}
#ifdef COLI_METAL
            /* bytesNoCopy requires an Apple-page-aligned base and page-rounded span. */
            if(coli_metal_init())coli_metal_register(slot->data,(size_t)s->slot_bytes);
#endif
            s->stats.resident_bytes+=s->slot_bytes; }
        slot->expert=-1; slot->loading=1; io_begin_locked(s,key); pthread_mutex_unlock(&s->mutex);
        char error[256];
#ifdef __APPLE__
        const char *direct=getenv("COLI_V4_DIRECT");
        coli_v4_expert_io_active=!direct||atoi(direct)!=0;
#endif
        int bad=coli_executor_load_expert(s->executor,key.layer,key.expert,slot->data,(size_t)s->record_bytes,error,sizeof(error));
#ifdef __APPLE__
        coli_v4_expert_io_active=0;
#endif
        pthread_mutex_lock(&s->mutex); slot->loading=0; io_finish_locked(s,key,!bad); if(bad){fprintf(stderr,"v4_coli expert-load failed layer=%d expert=%d: %s\n",key.layer,key.expert,error);pthread_mutex_unlock(&s->mutex);memset(view,0,sizeof(*view));return -1;} slot->expert=key.expert;s->stats.misses++;s->stats.bytes_read+=s->record_bytes;
    }
    slot->refs++; s->active_leases++; int bad=fill(view,key,r,slot); if(bad){fprintf(stderr,"v4_coli expert-view invalid layer=%d expert=%d\n",key.layer,key.expert);slot->refs--;s->active_leases--;} pthread_mutex_unlock(&s->mutex); return bad?-1:0;
}
static void release(ColiExpertStore *store, ColiExpertView *v) { State*s=store?store->state:NULL; Slot*slot=v?v->lease:NULL;if(!s||!slot)return;pthread_mutex_lock(&s->mutex);if(slot->refs)slot->refs--;if(s->active_leases)s->active_leases--;pthread_mutex_unlock(&s->mutex); }
static int prefetch(ColiExpertStore *store,const ColiExpertKey*k,size_t n){(void)store;(void)k;(void)n;return 0;}
static void stats(const ColiExpertStore *store,ColiExpertStoreStats*out){State*s=store?store->state:NULL;if(!s||!out)return;pthread_mutex_lock(&s->mutex);*out=s->stats;pthread_mutex_unlock(&s->mutex);}
static void destroy(ColiExpertStore *store){State*s=store?store->state:NULL;if(s){
    if(s->progress_enabled&&s->io_started_at&&s->io_reads){double elapsed=difftime(time(NULL),s->io_started_at);if(elapsed<1.0)elapsed=1.0;fprintf(stderr,"v4_progress phase=expert-stream status=done elapsed=%.0fs reads=%llu bytes=%.2fGiB avg=%.1fMiB/s peak_inflight=%u\n",elapsed,(unsigned long long)s->io_reads,(double)s->io_bytes/(1024.0*1024.0*1024.0),((double)s->io_bytes/(1024.0*1024.0))/elapsed,s->io_peak_inflight);}
    for(int i=0;i<s->layers*s->slots_per_layer;i++){
#ifdef COLI_METAL
    if(s->slots[i].data)coli_metal_unregister(s->slots[i].data);
#endif
    compat_aligned_free(s->slots[i].data);
}pthread_mutex_destroy(&s->mutex);coli_executor_close(s->executor);free(s->slots);free(s->records);free(s);}free(store);}
int coli_v4_coli_expert_store_open(const ColiV4ColiExpertStoreOptions*o,ColiExpertStore**out,char*e,size_t n) {
    static const ColiExpertStoreOps ops={lookup,release,prefetch,stats,destroy}; ColiExpertStore*store=NULL;State*s=NULL;ColiExecutorOpenOptions xo={0};
    if(out)*out=NULL; if(!o||!out||!o->package_dir||!o->required_profile||o->layers<1||o->experts_per_layer<1||!o->cache_bytes)return fail(e,n,"invalid COLI V4 expert-store options");
    store=calloc(1,sizeof(*store));s=calloc(1,sizeof(*s));if(!store||!s){free(store);free(s);return fail(e,n,"out of memory creating COLI expert store");}pthread_mutex_init(&s->mutex,NULL);
    s->progress_enabled=!getenv("V4_PROGRESS")||atoi(getenv("V4_PROGRESS"))!=0;
    s->progress_interval_s=5;{const char*p=getenv("V4_PROGRESS_INTERVAL");if(p&&*p){int v=atoi(p);if(v>=1&&v<=60)s->progress_interval_s=v;}}
    xo.required_profile=o->required_profile;
    xo.checksum_policy=getenv("COLI_VERIFY_RECORDS")&&atoi(getenv("COLI_VERIFY_RECORDS"))
        ?COLI_CSF_CHECKSUM_RECORD_ON_READ:COLI_CSF_CHECKSUM_MANIFEST_ONLY;
    if(coli_executor_open(&s->executor,o->package_dir,&xo,e,n))goto bad;s->layers=o->layers;s->experts=o->experts_per_layer;
    s->records=calloc((size_t)s->layers*s->experts,sizeof(*s->records));if(!s->records){fail(e,n,"out of memory indexing COLI experts");goto bad;}
    for(int l=0;l<s->layers;l++)for(int x=0;x<s->experts;x++){Record*r=&s->records[(size_t)l*s->experts+x];r->record=coli_executor_expert(s->executor,l,x);if(!r->record||coli_executor_expert_info(s->executor,l,x,&r->info,e,n)){fail(e,n,"COLI package is missing/invalid expert (%d,%d)",l,x);goto bad;}if(!s->record_bytes)s->record_bytes=r->record->stored_bytes;if(r->record->stored_bytes!=s->record_bytes){fail(e,n,"COLI experts have non-uniform stored sizes");goto bad;}}
    if(s->record_bytes>UINT64_MAX-16383u){fail(e,n,"COLI expert slot size overflow");goto bad;}s->slot_bytes=(s->record_bytes+16383u)&~UINT64_C(16383);if(s->slot_bytes>SIZE_MAX){fail(e,n,"COLI expert slot exceeds address space");goto bad;}
    s->slots_per_layer=(int)(o->cache_bytes/((uint64_t)s->layers*s->record_bytes));if(s->slots_per_layer<1){fail(e,n,"COLI cache budget cannot hold one expert per layer");goto bad;}if(s->slots_per_layer>s->experts)s->slots_per_layer=s->experts;
    s->slots=calloc((size_t)s->layers*s->slots_per_layer,sizeof(*s->slots));if(!s->slots){fail(e,n,"out of memory creating COLI expert slots");goto bad;}for(int i=0;i<s->layers*s->slots_per_layer;i++)s->slots[i].expert=-1;s->stats.capacity_bytes=(uint64_t)s->layers*s->slots_per_layer*s->slot_bytes;
    print_execution_mode();
    if(s->progress_enabled)fprintf(stderr,"v4_progress enabled=1 interval=%ds hint=V4_PROGRESS=0-to-disable\n",s->progress_interval_s);
    store->ops=&ops;store->state=s;*out=store;return 0;
bad:destroy(store);return -1;
}

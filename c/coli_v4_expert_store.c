#include "coli_v4_expert_store.h"
#include "coli_executor.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const ColiRecordInfo *record; ColiExpertInfo info; } Record;
typedef struct { int expert; unsigned refs; unsigned char *data; } Slot;
typedef struct {
    ColiExecutor *executor; int layers, experts, slots_per_layer;
    uint64_t record_bytes, clock; Record *records; Slot *slots;
    unsigned active_leases; ColiExpertStoreStats stats; pthread_mutex_t mutex;
} State;

static int fail(char *e,size_t n,const char *f,...) { va_list a; if(e&&n){va_start(a,f);vsnprintf(e,n,f,a);va_end(a);} return -1; }
static Record *record_for(State *s, ColiExpertKey k) {
    if(k.layer<0||k.layer>=s->layers||k.expert<0||k.expert>=s->experts) return NULL;
    return &s->records[(size_t)k.layer*s->experts+k.expert];
}
static Slot *slots_for(State*s,int layer) { return s->slots+(size_t)layer*s->slots_per_layer; }
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
    for(int i=0;i<s->slots_per_layer;i++) if(slots[i].data&&slots[i].expert==key.expert){slot=&slots[i];s->stats.hits++;break;}
    if(!slot) { for(int i=0;i<s->slots_per_layer;i++) if(!slots[i].refs&&(!slot||!slots[i].data)) slot=&slots[i];
        if(!slot) { pthread_mutex_unlock(&s->mutex); memset(view,0,sizeof(*view)); return -1; }
        if(!slot->data) { if(posix_memalign((void**)&slot->data,4096,(size_t)s->record_bytes)){pthread_mutex_unlock(&s->mutex);memset(view,0,sizeof(*view));return -1;} s->stats.resident_bytes+=s->record_bytes; }
        slot->expert=-1; pthread_mutex_unlock(&s->mutex);
        char error[256]; int bad=coli_executor_load_expert(s->executor,key.layer,key.expert,slot->data,(size_t)s->record_bytes,error,sizeof(error));
        pthread_mutex_lock(&s->mutex); if(bad){pthread_mutex_unlock(&s->mutex);memset(view,0,sizeof(*view));return -1;} slot->expert=key.expert;s->stats.misses++;s->stats.bytes_read+=s->record_bytes;
    }
    slot->refs++; s->active_leases++; int bad=fill(view,key,r,slot); if(bad){slot->refs--;s->active_leases--;} pthread_mutex_unlock(&s->mutex); return bad?-1:0;
}
static void release(ColiExpertStore *store, ColiExpertView *v) { State*s=store?store->state:NULL; Slot*slot=v?v->lease:NULL;if(!s||!slot)return;pthread_mutex_lock(&s->mutex);if(slot->refs)slot->refs--;if(s->active_leases)s->active_leases--;pthread_mutex_unlock(&s->mutex); }
static int prefetch(ColiExpertStore *store,const ColiExpertKey*k,size_t n){(void)store;(void)k;(void)n;return 0;}
static void stats(const ColiExpertStore *store,ColiExpertStoreStats*out){State*s=store?store->state:NULL;if(!s||!out)return;pthread_mutex_lock(&s->mutex);*out=s->stats;pthread_mutex_unlock(&s->mutex);}
static void destroy(ColiExpertStore *store){State*s=store?store->state:NULL;if(s){for(int i=0;i<s->layers*s->slots_per_layer;i++)free(s->slots[i].data);pthread_mutex_destroy(&s->mutex);coli_executor_close(s->executor);free(s->slots);free(s->records);free(s);}free(store);}
int coli_v4_coli_expert_store_open(const ColiV4ColiExpertStoreOptions*o,ColiExpertStore**out,char*e,size_t n) {
    static const ColiExpertStoreOps ops={lookup,release,prefetch,stats,destroy}; ColiExpertStore*store=NULL;State*s=NULL;ColiExecutorOpenOptions xo={0};
    if(out)*out=NULL; if(!o||!out||!o->package_dir||!o->required_profile||o->layers<1||o->experts_per_layer<1||!o->cache_bytes)return fail(e,n,"invalid COLI V4 expert-store options");
    store=calloc(1,sizeof(*store));s=calloc(1,sizeof(*s));if(!store||!s){free(store);free(s);return fail(e,n,"out of memory creating COLI expert store");}pthread_mutex_init(&s->mutex,NULL);
    xo.required_profile=o->required_profile;xo.checksum_policy=COLI_CSF_CHECKSUM_RECORD_ON_READ;
    if(coli_executor_open(&s->executor,o->package_dir,&xo,e,n))goto bad;s->layers=o->layers;s->experts=o->experts_per_layer;
    s->records=calloc((size_t)s->layers*s->experts,sizeof(*s->records));if(!s->records){fail(e,n,"out of memory indexing COLI experts");goto bad;}
    for(int l=0;l<s->layers;l++)for(int x=0;x<s->experts;x++){Record*r=&s->records[(size_t)l*s->experts+x];r->record=coli_executor_expert(s->executor,l,x);if(!r->record||coli_executor_expert_info(s->executor,l,x,&r->info,e,n)){fail(e,n,"COLI package is missing/invalid expert (%d,%d)",l,x);goto bad;}if(!s->record_bytes)s->record_bytes=r->record->stored_bytes;if(r->record->stored_bytes!=s->record_bytes){fail(e,n,"COLI experts have non-uniform stored sizes");goto bad;}}
    s->slots_per_layer=(int)(o->cache_bytes/((uint64_t)s->layers*s->record_bytes));if(s->slots_per_layer<1){fail(e,n,"COLI cache budget cannot hold one expert per layer");goto bad;}if(s->slots_per_layer>s->experts)s->slots_per_layer=s->experts;
    s->slots=calloc((size_t)s->layers*s->slots_per_layer,sizeof(*s->slots));if(!s->slots){fail(e,n,"out of memory creating COLI expert slots");goto bad;}for(int i=0;i<s->layers*s->slots_per_layer;i++)s->slots[i].expert=-1;s->stats.capacity_bytes=(uint64_t)s->layers*s->slots_per_layer*s->record_bytes;store->ops=&ops;store->state=s;*out=store;return 0;
bad:destroy(store);return -1;
}

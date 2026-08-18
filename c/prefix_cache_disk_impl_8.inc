static inline size_t coli_prefix_disk_budget_from_env(size_t fallback){return parse_scaled("COLI_PREFIX_CACHE_DISK_GB",fallback,1024.0L*1024.0L*1024.0L,1);}
static inline size_t coli_prefix_ram_budget_from_env(size_t fallback){return parse_scaled("COLI_PREFIX_CACHE_RAM_MB",fallback,1024.0L*1024.0L,1);}
static const char*cache_mode(void){const char*v=getenv("COLI_PREFIX_CACHE");return(!v||!*v)?"auto":v;}
static inline int coli_prefix_disk_mode_allows_ssd(void){const char*m=cache_mode();return ascii_ieq(m,"auto")||ascii_ieq(m,"ssd");}
static inline int coli_prefix_disk_mode_allows_ram(void){const char*m=cache_mode();return ascii_ieq(m,"auto")||ascii_ieq(m,"ram");}
static inline int coli_prefix_disk_default_directory(char*out,size_t cap){if(!out||!cap)return 0;const char*e=getenv("COLI_PREFIX_CACHE_DIR");if(e&&*e){int n=snprintf(out,cap,"%s",e);return n>0&&(size_t)n<cap;}
#ifdef _WIN32
const char*base=getenv("LOCALAPPDATA");if(!base||!*base)base=getenv("TEMP");if(!base||!*base)return 0;int n=snprintf(out,cap,"%s/Colibri/prefix-cache",base);
#elif defined(__APPLE__)
const char*base=getenv("HOME");if(!base||!*base)return 0;int n=snprintf(out,cap,"%s/Library/Caches/colibri/prefix-cache",base);
#else
const char*base=getenv("XDG_CACHE_HOME");int n;if(base&&*base)n=snprintf(out,cap,"%s/colibri/prefix-cache",base);else{base=getenv("HOME");if(!base||!*base)return 0;n=snprintf(out,cap,"%s/.cache/colibri/prefix-cache",base);}
#endif
return n>0&&(size_t)n<cap;}


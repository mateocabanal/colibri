}
static void sync_directory(const char *dir){
#ifndef _WIN32
#ifdef O_DIRECTORY
    int fd=open(dir,O_RDONLY|O_DIRECTORY); if(fd>=0){(void)fsync(fd);close(fd);}
#else
    (void)dir;
#endif
#else
    (void)dir;
#endif
}
static int lock_acquire(const char *dir,PrefixLock *l){char p[COLI_PREFIX_DISK_PATH_MAX]; if(!l||!join_path(p,sizeof(p),dir,".lock"))return 0;
#ifdef _WIN32
    memset(l,0,sizeof(*l)); l->handle=CreateFileA(p,GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,NULL,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,NULL); if(l->handle==INVALID_HANDLE_VALUE)return 0; if(!LockFileEx(l->handle,LOCKFILE_EXCLUSIVE_LOCK,0,1,0,&l->overlapped)){CloseHandle(l->handle);l->handle=INVALID_HANDLE_VALUE;return 0;} return 1;
#else
    l->fd=open(p,O_CREAT|O_RDWR,0600);
    if(l->fd<0)return 0;
    struct flock lock;
    memset(&lock,0,sizeof(lock));
    lock.l_type=F_WRLCK;
    lock.l_whence=SEEK_SET;
    lock.l_start=0;
    lock.l_len=1;
    for(;;){
        if(fcntl(l->fd,F_SETLKW,&lock)==0)return 1;
        if(errno==EINTR)continue;
        close(l->fd);
        l->fd=-1;
        return 0;
    }
#endif
}
static void lock_release(PrefixLock *l){if(!l)return;
#ifdef _WIN32
    if(l->handle&&l->handle!=INVALID_HANDLE_VALUE){UnlockFileEx(l->handle,0,1,0,&l->overlapped);CloseHandle(l->handle);l->handle=INVALID_HANDLE_VALUE;}
#else
    if(l->fd>=0){
        struct flock lock;
        memset(&lock,0,sizeof(lock));
        lock.l_type=F_UNLCK;
        lock.l_whence=SEEK_SET;
        lock.l_start=0;
        lock.l_len=1;
        while(fcntl(l->fd,F_SETLK,&lock)!=0&&errno==EINTR){}
        close(l->fd);
        l->fd=-1;
    }
#endif

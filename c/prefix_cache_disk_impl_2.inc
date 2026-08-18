typedef int (*ColiPrefixDiskStreamFn)(void *user_data,ColiPrefixDiskSinkFn sink,void *sink_data);

#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <utime.h>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <process.h>
#include <windows.h>
#define getpid _getpid
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <unistd.h>
extern int fileno(FILE *);
#endif

#include "../coli_format.h"

#include <stdio.h>

int main(int argc, char **argv) {
    ColiPackage *package = NULL;
    char error[512];
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.coli\n", argc > 0 ? argv[0] : "coli_check");
        return 2;
    }
    if (coli_package_open(&package, argv[1], error, sizeof(error))) {
        fprintf(stderr, "coli_check: open failed: %s\n", error);
        return 1;
    }
    if (coli_package_verify_all(package, error, sizeof(error))) {
        fprintf(stderr, "coli_check: verification failed: %s\n", error);
        coli_package_close(package);
        return 1;
    }
    printf("coli_check: ok records=%zu profile=%s alignment=%u\n",
           coli_package_record_count(package),
           coli_package_profile(package) ? coli_package_profile(package) : "?",
           coli_package_record_alignment(package));
    coli_package_close(package);
    return 0;
}

#include <sys/resource.h>
#include <errno.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    struct rlimit rlim;
    rlim.rlim_cur = RLIM_INFINITY;
    rlim.rlim_max = RLIM_INFINITY;

    if (setrlimit(RLIMIT_MSGQUEUE, &rlim) == -1) {
        perror("setrlimit");
        return 1;
    }
}

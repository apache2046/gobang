#include <sys/wait.h> /* wait */
#include <stdio.h>
#include <stdlib.h>   /* exit functions */
#include <unistd.h>   /* read, write, pipe, _exit */
#include <string.h>

#define ReadEnd  0
#define WriteEnd 1

void report_and_exit(const char* msg) {
  perror(msg);
  exit(-1);    /** failure **/
}

int main() {
  int pipeFDs[2]; /* two file descriptors */
  int buf = 123;  /* 1-byte buffer */
  const char* msg = "Nature's first green is gold\n"; /* bytes to write */
  int ret1;
  int ret2;

  if (pipe(pipeFDs) < 0) report_and_exit("pipeFD");

  ret1 = write(pipeFDs[ReadEnd], &buf, 4);
  buf = 9999;
  ret2 = read(pipeFDs[WriteEnd], &buf, 4);
  printf("%d %d %d\n", buf, ret1, ret2);

  return 0;
}

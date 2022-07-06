#include <sys/wait.h> /* wait */
#include <stdio.h>
#include <stdlib.h>   /* exit functions */
#include <unistd.h>   /* read, write, pipe, _exit */
#include <string.h>
#include <unordered_map>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/epoll.h>//epoll
#include <iostream>
#include <chrono>

#define ReadEnd  0
#define WriteEnd 1
using namespace std;

void report_and_exit(const char* msg) {
  perror(msg);
  exit(-1);    /** failure **/
}
void child_worker(int id){
  char buf[1024];       /* 1-byte buffer */
  sprintf(buf, "/tmp/cpp_fifo/%d_0", id);
  int outfd = open(buf,  O_RDWR);
  sprintf(buf, "/tmp/cpp_fifo/%d_1", id);
  int infd = open(buf,  O_RDWR);

  while(1){
    write(outfd, buf, 200);
    read(infd, buf, 1);
  }
  cout <<" child return " << id << endl;
}
// void server_worker(){

// }

int main() {
  int fds[1024];
  // int fd2s[1024];
  char buf[1024];       /* 1-byte buffer */
  int i = 0;
  const int p_cnt = 100;
  for (i=0; i< p_cnt; i++){
    pid_t cpid = fork();                                /* fork a child process */
    if (cpid < 0) report_and_exit("fork");              /* check for failure */

    if (0 == cpid) {    /*** child ***/                 /* child process */
      child_worker(i);
    }
  }
  std::unordered_map<int, int> m;
  for (i=0; i< p_cnt; i++){
    sprintf(buf, "/tmp/cpp_fifo/%d_0", i);
    fds[i] = open(buf,  O_RDWR);
    sprintf(buf, "/tmp/cpp_fifo/%d_1", i);
    int fd = open(buf,  O_RDWR);
    m[fds[i]] = fd;
  }

  //创建一个epoll,size已经不起作用了,一般填1就好了
  int eFd = epoll_create(1);

  //把socket包装成一个epoll_event对象
  //并添加到epoll中
  for (i=0; i< p_cnt; i++){
    epoll_event epev{};
    epev.events = EPOLLIN;//可以响应的事件,这里只响应可读就可以了
    epev.data.fd = fds[i];
    epoll_ctl(eFd, EPOLL_CTL_ADD, fds[i], &epev);//添加到epoll中
  }

  #define EVENTS_SIZE 1024
  //回调事件的数组,当epoll中有响应事件时,通过这个数组返回
  epoll_event events[EVENTS_SIZE];

  int cnt = 0;
  //整个epoll_wait 处理都要在一个死循环中处理
  auto stime = std::chrono::steady_clock::now();;
  while (true) {
    if (cnt > 100000) {
      cnt = 0;
      std::chrono::duration<double> delta = std::chrono::steady_clock::now() - stime;
      cout << cnt << " " << delta.count() << " pid:" << getpid() << endl;
      stime = std::chrono::steady_clock::now();
      // fflush(stdout);
    }
    //这个函数会阻塞,直到超时或者有响应事件发生
    int eNum = epoll_wait(eFd, events, EVENTS_SIZE, -1);

    if (eNum == -1 || eNum > 200) {
        cout << "epoll_wait" << endl;
        return -1;
    }
    // printf("%d ", eNum);
    //遍历所有的事件
    for (int i = 0; i < eNum; i++) {
        //判断这次是不是socket可读(是不是有新的连接)
        // if (events[i].data.fd == socketFd) {
        //     if (events[i].events & EPOLLIN) {

        // }
      int infd = events[i].data.fd;
      int outfd = m[infd];
      // char buffer[8];
      read(infd, buf, 200);
      write(outfd, buf, 1);
      cnt ++;
    }
  }

  return 0;
}

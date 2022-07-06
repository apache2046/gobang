from queue import Empty, Queue
from model import Policy_Value
from multiprocessing.connection import Listener, Connection
import numpy as np
import multiprocessing as mp
from typing import List
import time
import queue
from multiprocessing import shared_memory
import threading
import select
# import torch
# import pickle
def infer_worker(pipes: List[Connection]):
    # nnet = Policy_Value().to('cuda:0')
    # nnet.eval()
    # id_list = []
    # states = []
    epoll = select.epoll()
    cnt = 0
    stime = time.time()
    # shmdict = {}
    pdict = {}
    for p in pipes:

        epoll.register(p.fileno(), select.EPOLLIN | select.EPOLLET)
        pdict[p.fileno()] = p
    while True:
        if cnt > 100000:
            cnt = 0
            print("complete", cnt, f"{time.time() - stime:.2f}")
            stime = time.time()
        events = epoll.poll()
        for fileno, event in events:
            p = pdict[fileno]
            data = p.recv()
            # ret = data.mean()
            ret = 1
            p.send(ret)
            cnt += 1



def net_worker(wid: int, p: Connection):

    input = np.random.randn(1024)
    while True:
        p.send(1)
        p.recv()


def main():
    mp.set_start_method("spawn")
    # task_q = mp.Queue()
    # manager = mp.Manager()
    # result_qlist = manager.list()



    # p2 = mp.Process(target=infer_worker, args=(task_q, result_qlist))
    # p2.daemon = True
    # p2.start()

    print("prepare listen")
    children = []
    pipes1 = []
    pipes2 = []
    pipes3 = []
    pipes4 = []

    for _ in range(100):
        p1, p2 = mp.Pipe()
        pipes1.append(p1)
        pipes2.append(p2)

    p = mp.Process(target=infer_worker, args=(pipes1,))
    p.daemon = True
    p.start()

    # for _ in range(100):
    #     p3, p4 = mp.Pipe()
    #     pipes3.append(p3)
    #     pipes4.append(p4)

    # p = mp.Process(target=infer_worker, args=(pipes3,))
    # p.daemon = True
    # p.start()

    wid = 0
    for _ in range(100):
        time.sleep(0.1)
        print("1", wid)
        # result_q = mp.Queue()
        # result_q = mp.Manager().Queue()
        # result_qlist.append(result_q)
        # print('2', wid)
        child = mp.Process(target=net_worker, args=(wid, pipes2[wid]))
        # child = threading.Thread(target=net_worker, args=(wid, task_q, result_q))
        child.daemon = True
        child.start()
        children.append(child)
        wid += 1

    # wid = 0
    # for _ in range(100):
    #     time.sleep(0.1)
    #     print("1", wid)
    #     # result_q = mp.Queue()
    #     # result_q = mp.Manager().Queue()
    #     # result_qlist.append(result_q)
    #     # print('2', wid)
    #     child = mp.Process(target=net_worker, args=(wid, pipes4[wid]))
    #     # child = threading.Thread(target=net_worker, args=(wid, task_q, result_q))
    #     child.daemon = True
    #     child.start()
    #     children.append(child)
    #     wid += 1

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

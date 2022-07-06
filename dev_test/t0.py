from queue import Empty, Queue
# from model import Policy_Value
from multiprocessing.connection import Listener, Connection
import numpy as np
import multiprocessing as mp
from typing import List
import time
import queue
from multiprocessing import shared_memory
import threading
# import torch
# import pickle
def infer_worker(task_q: mp.Queue, result_qlist: List[mp.Queue]):
    # nnet = Policy_Value().to('cuda:0')
    # nnet.eval()
    # id_list = []
    # states = []
    cnt = 0
    stime = time.time()
    shmdict = {}
    local_result_qlist = {}
    while True:
        cnt += 1
        if cnt % 10000 == 0:
            print("complete", cnt, f"{time.time() - stime:.2f}")
            stime = time.time()
        wid, shmname = task_q.get()
        if False:
            result_q = result_qlist[wid]
        else:
            if (result_q := local_result_qlist.get(wid)) is None:
                result_q = local_result_qlist[wid] = result_qlist[wid]
        # if (shm := shmdict.get(shmname)) is None:
        #     shm = shared_memory.SharedMemory(shmname)
        #     shmdict[shmname] = shm
        # task = np.ndarray((1024), dtype=np.float32, buffer=shm.buf)
        # result = task.mean()
        result_q.put(1)


def net_worker(wid: int, task_q: mp.Queue, result_q: mp.Queue):
    shm = shared_memory.SharedMemory(create=True, size=1024 * 1024)
    shm_nparray = np.ndarray((1024), dtype=np.float32, buffer=shm.buf)
    while True:
        # task = np.random.randn(1024)
        # shm_nparray = task
        # task_q.put((wid, shm.name))
        task_q.put((wid, 1))
        result = result_q.get()
        # time.sleep(0.001)


def main():
    mp.set_start_method("spawn")
    task_q = mp.Queue()
    manager = mp.Manager()
    result_qlist = manager.list()

    p = mp.Process(target=infer_worker, args=(task_q, result_qlist))
    p.daemon = True
    p.start()

    # p2 = mp.Process(target=infer_worker, args=(task_q, result_qlist))
    # p2.daemon = True
    # p2.start()

    print("prepare listen")
    wid = 0
    children = []
    for _ in range(100):
        time.sleep(0.1)
        print("1", wid)
        # result_q = mp.Queue()
        # result_q = mp.Manager().Queue()
        result_q = manager.Queue()
        result_qlist.append(result_q)
        # print('2', wid)
        child = mp.Process(target=net_worker, args=(wid, task_q, result_q))
        # child = threading.Thread(target=net_worker, args=(wid, task_q, result_q))
        child.daemon = True
        child.start()
        children.append(child)
        wid += 1

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

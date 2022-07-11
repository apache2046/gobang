from queue import Empty, Queue
from model import Policy_Value
from multiprocessing.connection import Listener, Connection
import numpy as np
import multiprocessing as mp
from typing import List
import torch
import pickle
import time

def infer_worker(task_q: mp.Queue, result_qlist: List[mp.Queue]):
    nnet = Policy_Value().to('cuda:0')
    nnet.eval()
    id_list = []
    states = []
    cnt = 0
    stime = time.time()
    while True:
        try:
            task = task_q.get(timeout=0.1)
        except Empty:
            task = None
        if task is not None:
            wid, state = task
            # state = pickle.loads(state)
            id_list.append(wid)
            states.append(torch.tensor(state, dtype=torch.float32))
        cnt += len(id_list)
        if cnt > 10000:
            print(cnt, time.time() - stime)
            stime = time.time()
            cnt = 0
        if len(id_list) >= 8 or task is None and len(id_list) > 0:
            states = torch.stack(states).permute(0, 3, 1, 2).to("cuda:0")
            with torch.no_grad():
                pi, v = nnet(states)
            pi = pi.to('cpu').numpy()
            v = v.to('cpu').numpy()
            for i in range(len(id_list)):
                result_qlist[i].put(pickle.dumps((pi[i], v[i])))
            id_list = []
            states = []


def net_worker(wid: int, conn: Connection, task_q: mp.Queue, result_q: mp.Queue):
    while True:
        task = conn.recv()
        task_q.put((wid, task))
        result = result_q.get()
        conn.send(result)


def main():
    mp.set_start_method('spawn')
    task_q = mp.Queue()
    manager = mp.Manager()
    result_qlist = manager.list()

    p = mp.Process(target=infer_worker, args=(task_q, result_qlist))
    p.daemon = True
    p.start()

    p2 = mp.Process(target=infer_worker, args=(task_q, result_qlist))
    p2.daemon = True
    p2.start()


    address = ("", 6000)
    listener = Listener(address, authkey=b"secret password123")
    print("prepare listen")
    wid = 0
    children = []
    while True:
        conn = listener.accept()
        print("connection accepted from", listener.last_accepted, 'wid:', wid)
        result_q = manager.Queue()
        result_qlist.append(result_q)
        child = mp.Process(target=net_worker, args=(wid, conn, task_q, result_q))
        child.daemon = True
        child.start()
        children.append(child)
        wid += 1


if __name__ == "__main__":
    main()

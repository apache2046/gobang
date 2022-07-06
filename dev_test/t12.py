from queue import Empty, Queue
from unicodedata import name
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
import os
import struct
import pickle
import shutil
import sys
# from line_profiler import LineProfiler

# profiler = LineProfiler()

# def print(*args):
#     s = str(args) + '\n'
#     with open('/tmp/log123_' + str(os.getpid()), 'a') as f:
#         f.write(s)
# import torch
# import pickle
def readall(fd, remain):
    ret = bytes()
    while remain > 0:
        d = os.read(fd, remain)
        remain -= len(d)
        ret += d
    return ret

def read_obj_from_fifo(fd):
    n_buf = os.read(fd, 4)
    n = struct.unpack('I', n_buf)[0]
    # obj_bytes = os.read(fd, n)
    obj_bytes = readall(fd, n)
    # print('read2', n, len(obj_bytes))
    if n != len(obj_bytes):
        raise Exception('hahahaha')
    obj = pickle.loads(obj_bytes)
    return obj

def write_obj_to_fifo(fd, obj):
    obj_bytes = pickle.dumps(obj)
    n = len(obj_bytes)
    n_buf = struct.pack('I', n)
    l = os.write(fd, n_buf)
    if l != len(n_buf):
        raise Exception('hahahaha2', l, len(n_buf))
    l = os.write(fd, obj_bytes)
    if l != len(obj_bytes):
        raise Exception('hahahaha3', l, len(obj_bytes))
    # obj_bytes
    # print('write2', n, len(obj_bytes))

# @profiler
def _infer_worker(q: mp.Queue):
    # nnet = Policy_Value().to('cuda:0')
    # nnet.eval()
    # id_list = []
    # states = []
    # logfile = open(f'log_infer_{os.getpid()}', 'w')
    # sys.stdout = logfile
    epoll = select.epoll()
    cnt = 0
    stime = time.time()
    qcheck_time = time.time()
    # shmdict = {}
    fifolist = {}

    infered = False
    while True:
        # if infered:
        #     cnt += 1
        #     infered = False
        if cnt >= 100000:
            cnt = 0
            print("complete", f"{time.time() - stime:.2f}")
            stime = time.time()
        if time.time() - qcheck_time > 0.1:
            try:
                fifo0, fifo1 = q.get(block=False)
                print('server get', fifo0, fifo1)
                fd0 = os.open(fifo0, os.O_RDONLY)
                fd1 = os.open(fifo1, os.O_WRONLY)
                print('server open', fifo0, fifo1)
                epoll.register(fd0, select.EPOLLIN | select.EPOLLET)
                fifolist[fd0] = fd1
            except Empty:
                pass
            qcheck_time = time.time()

        events = epoll.poll(0.1)
        for fileno, event in events:
            # print('server event')
            read_obj_from_fifo(fileno)
            outfd = fifolist[fileno]
            write_obj_to_fifo(outfd, 11)
            # infered = True
            cnt += 1
def infer_worker(q: mp.Queue):
    return _infer_worker(q)


def net_worker(wid: int, fifo0, fifo1):
    # os.nice(20)
    input = np.random.randn(1024)
    # print('child', wid, fifo0, fifo1)
    fd0 = os.open(fifo0, os.O_WRONLY)
    fd1 = os.open(fifo1, os.O_RDONLY)
    while True:
    # for _ in range(2):
        # print('child', wid, pipename)
        write_obj_to_fifo(fd0, input)
        read_obj_from_fifo(fd1)


def main():
    # mp.set_start_method("spawn")

    print("prepare listen")
    children = []

    q1 = mp.Manager().Queue()

    p = mp.Process(target=infer_worker, args=(q1,))
    p.daemon = True
    p.start()

    shutil.rmtree('/tmp/t12', ignore_errors=True)
    os.mkdir('/tmp/t12')
    wid = 0
    for _ in range(100):
        time.sleep(0.1)
        print("1", wid)
        # result_q = mp.Queue()
        # result_q = mp.Manager().Queue()
        # result_qlist.append(result_q)
        # print('2', wid)
        fifo0 = f'/tmp/t12/fifo_{wid}_0'
        fifo1 = f'/tmp/t12/fifo_{wid}_1'
        os.mkfifo(fifo0)
        os.mkfifo(fifo1)
        q1.put((fifo0, fifo1))
        child = mp.Process(target=net_worker, args=(wid, fifo0, fifo1))
        # child = threading.Thread(target=net_worker, args=(wid, task_q, result_q))
        child.daemon = True
        child.start()
        children.append(child)
        wid += 1

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

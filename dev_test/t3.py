import multiprocessing as mp
import time

def worker(q1, q2):
    cnt = 0
    stime = time.time()
    while True:
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt, time.time()-stime)
            stime = time.time()
        q1.get()
        q2.put(2)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    manager = mp.Manager()
    q1 = manager.Queue()
    q2 = manager.Queue()
    p = mp.Process(target=worker, args=(q1, q2))
    p.daemon = True
    p.start()
    while True:
        q1.put(1)
        q2.get()

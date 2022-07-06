import multiprocessing as mp
import time
def worker(semp1, semp2):
    cnt = 0
    stime = time.time()
    while True:
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt, time.time()-stime)
            stime = time.time()
        semp1.wait()
        semp2.set()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    semp1 = mp.Event()
    semp2 = mp.Event()
    p = mp.Process(target=worker, args=(semp1, semp2))
    p.daemon = True
    p.start()
    while True:
        semp1.set()
        semp2.wait()

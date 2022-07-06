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
        semp1.release()
        semp2.acquire()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    semp1 = mp.Semaphore(0)
    semp2 = mp.Semaphore(0)
    p = mp.Process(target=worker, args=(semp1, semp2))
    p.daemon = True
    p.start()
    while True:
        semp1.acquire()
        semp2.release()

import multiprocessing as mp
import time
import numpy as np
def worker(pipe):
    cnt = 0
    stime = time.time()
    while True:
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt, time.time()-stime)
            stime = time.time()
        state = pipe.recv()
        pipe.send(state.mean())


if __name__ == '__main__':
    mp.set_start_method('spawn')
    p1, p2 = mp.Pipe()
    p = mp.Process(target=worker, args=(p2,))
    p.daemon = True
    p.start()
    input = np.random.randn(10000)
    while True:
        p1.send(input)
        p1.recv()

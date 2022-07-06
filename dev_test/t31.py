from ipcqueue import posixmq
import multiprocessing as mp
import time
import numpy as np

def infer_worker():
    taskq = posixmq.Queue('/foo', maxsize=10, maxmsgsize=1024)
    cnt = 0
    rq_list = {}
    stime = time.time()
    while True:
        cnt += 1
        if cnt % 100000 == 0:
            print(cnt, f'{time.time() - stime:.2f}')
            stime = time.time()

        data = taskq.get()
        # print('got', data, type(data))
        wid, task = data
        if (rq := rq_list.get(wid)) is None:
            rq = posixmq.Queue(f'/result_{wid}')
            rq_list[wid] = rq
        # rq.put(task.mean())
        rq.put(1)


def net_worker(wid):
    taskq = posixmq.Queue('/foo')
    resultq = posixmq.Queue(f'/result_{wid}')
    input = np.random.randn(1)
    while True:
        taskq.put((wid, input))
        resultq.get()

def main():
    mp.set_start_method("spawn")
    print("prepare listen")
    children = []
    taskq = posixmq.Queue('/foo')
    taskq.unlink()
    del taskq

    p = mp.Process(target=infer_worker, args=())
    p.daemon = True
    p.start()

    time.sleep(1)
    wid = 0
    for _ in range(50):
        time.sleep(0.1)
        print("1", wid)
        child = mp.Process(target=net_worker, args=(wid,))
        child.daemon = True
        child.start()
        children.append(child)
        wid += 1

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

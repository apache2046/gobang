from multiprocessing.connection import Client
import numpy as np
from mcts import MCTS
from game import GoBang
import multiprocessing as mp
import time
import pickle

# with Client(address, authkey=b'secret password') as conn:
#     conn.send(np.arange(12, dtype=np.int8).reshape(3,4))
#     print(conn.recv())


def executeEpisode(game, nnet):
    mcts = MCTS(game, c_puct=0.5)
    state = game.start_state()
    samples = []
    cnt = 0
    while True:
        cnt += 1
        print("GHB", cnt)
        for i in range(1000):
            mcts.search(state, nnet)
        pi = mcts.pi(state)
        samples.append([state, pi, None])
        action = np.random.choice(len(pi), p=pi)
        next_state, isend, reward = game.next_state(state, action)
        if isend:
            v = reward
            for j in reversed(range(len(samples))):
                samples[j][2] = v
                v = -v
            return samples
        else:
            state = next_state


class RPC_infer:
    def __init__(self, address, key):
        self.conn = Client(address, authkey=key)

    def infer(self, state):
        # print("before send")
        self.conn.send(state)
        # print("before recv")
        result = self.conn.recv()
        result = pickle.loads(result)
        # print("end recv", result)
        return result


def worker():
    game = GoBang(size=15)

    rpc = RPC_infer(("localhost", 6000), b"secret password123")

    while True:
        result = executeEpisode(game, rpc)
        print("got result", len(result))


def main():
    plist = []
    # mp.set_start_method('spawn')
    for i in range(32):
        p = mp.Process(target=worker, args=())
        p.daemon = True
        p.start()
        plist.append(p)
        time.sleep(0.1)
    for p in plist:
        p.join()


if __name__ == "__main__":
    main()

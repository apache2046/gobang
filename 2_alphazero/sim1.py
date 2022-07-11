from multiprocessing.connection import Client
import numpy as np
from mcts3 import MCTS
from game import GoBang
import multiprocessing as mp
import time
import pickle
import numba

# with Client(address, authkey=b'secret password') as conn:
#     conn.send(np.arange(12, dtype=np.int8).reshape(3,4))
#     print(conn.recv())

def executeEpisode(game, nnet):
    mcts = MCTS(game, c_puct=0.5)
    state = game.start_state()
    samples = []
    cnt = 0
    board_record = np.zeros((game.size, game.size), dtype=np.int8)
    stime = time.time()
    while True:

        # mcts = MCTS(game, c_puct=0.5)
        cnt += 1
        print("GHB", cnt, f'{time.time()-stime:.2f}', len(mcts.ns))
        stime = time.time()
        for i in range(2000):
            mcts.search(state, nnet)
        pi = mcts.pi(state)
        samples.append([state, pi, None])
        action = np.random.choice(len(pi), p=pi)
        next_state, isend, reward = game.next_state(state, action)
        y = action // game.size
        x = action % game.size
        board_record[y, x] = cnt
        if isend:
            v = reward
            for j in reversed(range(len(samples))):
                samples[j][2] = v
                v = -v
            # with open(f"{epid}.txt", "a") as f:
            #     f.write(str(board_record) + " " + str(cnt) + " " + str(reward) + "\n\n")
            print(board_record)
            return samples
        else:
            state = next_state


class RPC_infer:
    def __init__(self, address, key):
        # self.conn = Client(address, authkey=key)
        pass

    def infer(self, state):
        # print(state.shape)
        # h, w = state.shape
        return np.ones(225) / 225, 1


def worker():
    game = GoBang(size=15)

    rpc = RPC_infer(("localhost", 6000), b"secret password123")

    while True:
        result = executeEpisode(game, rpc)
        print("got result", len(result))


def main():
    worker()


if __name__ == "__main__":
    main()

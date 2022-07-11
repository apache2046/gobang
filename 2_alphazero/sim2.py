from multiprocessing.connection import Client
import numpy as np
from mcts2 import MCTS
from game import GoBang
import multiprocessing as mp
import time
import pickle
from model import Policy_Value
import torch

# with Client(address, authkey=b'secret password') as conn:
#     conn.send(np.arange(12, dtype=np.int8).reshape(3,4))
#     print(conn.recv())


def executeEpisode(game, epid):
    mcts = MCTS(game, c_puct=0.5)
    state = game.start_state()
    samples = []
    cnt = 0
    board_record = np.zeros((game.size, game.size), dtype=np.int)
    while True:
        cnt += 1
        print("GHB", cnt)
        for i in range(1000):
            yield from mcts.search(state)
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
            with open(f"{epid}.txt", "a") as f:
                f.write(str(board_record) + " " + str(cnt) + " " + str(reward) + "\n\n")
            return samples
        else:
            state = next_state


def executeEpisodeEndless(epid):
    game = GoBang(size=15)
    while True:
        trajectory = yield from executeEpisode(game, epid)
        print(trajectory)


def main():
    states = []
    g = []
    for i in range(4):
        # np.random.seed(100+i)

        item = executeEpisodeEndless(i)
        g.append(item)
        states.append(torch.tensor(next(item)))

    print(states)

    nnet = Policy_Value().to("cuda:0")
    nnet.eval()
    while True:
        input_tensor = torch.stack(states).permute(0, 3, 1, 2).to("cuda:0").to(torch.float32)
        with torch.no_grad():
            prob, v = nnet(input_tensor)
            prob = prob.cpu().numpy()
            v = v.cpu()
        for i in range(len(g)):
            states[i] = torch.tensor(g[i].send((prob[i], v[i])))


if __name__ == "__main__":
    main()

from multiprocessing.connection import Client
import numpy as np
from mcts2 import MCTS
from game import GoBang
import multiprocessing as mp
import time
import pickle
from model import Policy_Value
import torch
import ray

# ray.init(address='auto', _node_ip_address='192.168.5.7')
ray.init(address='ray://192.168.5.7:10001')

# with Client(address, authkey=b'secret password') as conn:
#     conn.send(np.arange(12, dtype=np.int8).reshape(3,4))
#     print(conn.recv())


def executeEpisode(game, epid):
    mcts = MCTS(game, c_puct=0.5)
    state = game.start_state()
    samples = []
    cnt = 0
    board_record = np.zeros((game.size, game.size), dtype=np.int)
    stime = time.time()
    while True:
        cnt += 1
        print("GHB", cnt, f'{time.time()-stime: .2f}')
        stime = time.time()
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


@ray.remote
def simbatch(infer_service):
    states = []
    g = []
    for i in range(32):
        # np.random.seed(100+i)

        item = executeEpisodeEndless(i)
        g.append(item)
        states.append(torch.tensor(next(item)))

    while True:
        # print('before remote1')
        prob, v = ray.get(infer_service.infer.remote(states))
        # print(prob, v)
        for i in range(len(g)):
            states[i] = torch.tensor(g[i].send((prob[i], v[i])))


@ray.remote(num_cpus=1, num_gpus=1)
class Infer_srv:
    def __init__(self):
        self.nnet = Policy_Value().to("cuda:0")
        self.nnet.eval()

    def infer(self, data):
        # print('in infer1', torch.cuda.is_available())
        input_tensor = torch.stack(data).permute(0, 3, 1, 2).to("cuda:0").to(torch.float32)
        # print('in infer2')
        with torch.no_grad():
            prob, v = self.nnet(input_tensor)
            prob = prob.cpu().numpy()
            v = v.cpu()

        return prob, v


def main():
    infer_service = Infer_srv.remote()
    s = []
    for i in range(48):
        s.append(simbatch.remote(infer_service))
    ray.wait(s)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

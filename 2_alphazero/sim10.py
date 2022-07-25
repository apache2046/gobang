from multiprocessing.connection import Client
import numpy as np
from mcts5 import MCTS
from game2 import GoBang
import multiprocessing as mp
import time
import pickle
from model4 import Policy_Value
import torch
import ray
import io
import random
from collections import deque
import os
import socket
import traceback

print(socket.gethostname(), os.getcwd())
os.environ["RAY_LOG_TO_STDERR"] = "1"
ray.init(address="auto", _node_ip_address="192.168.5.6")
# ray.init(address="auto")
# ray.init(address='ray://192.168.5.7:10001')

# with Client(address, authkey=b'secret password') as conn:
#     conn.send(np.arange(12, dtype=np.int8).reshape(3,4))
#     print(conn.recv())


def executeEpisode(game, epid):
    # mcts = MCTS(game, c_puct=0.5)
    state = game.start_state()
    samples = []
    cnt = 0
    board_record = np.zeros((game.size, game.size), dtype=np.int8)
    stime = time.time()
    tau = 0.8
    while True:
        mcts = MCTS(game, c_puct=5)
        cnt += 1
        if cnt > 2:
            # tau = max(0.05, tau * 0.85)
            tau = max(0.05, tau * 0.9)
        if epid == 0:
            print("GHB", cnt, f"tau:{tau:.2f}, {time.time()-stime: .2f}")
        stime = time.time()
        for i in range(2000):
            yield from mcts.search(state)
        pi = mcts.pi(state, tau)
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
            return samples, board_record
        else:
            state = next_state
     

def executeEpisodeEndless(epid, tainer):
    game = GoBang(size=15)
    while True:
        print("executeEpisodeEndless1", epid)
        trajectory, board_record = yield from executeEpisode(game, epid)
        print(f"{epid} got trajectory", "\n", board_record, "\n", trajectory[-1][2], trajectory[-2][2], trajectory[-3][2] , trajectory[-4][2])
        tainer.push_samples.remote(trajectory)


@ray.remote(num_cpus=1)
def simbatch(infer_service, tainer):
    try: 
        states = []
        g = []
        print("GHB in simbatch")
        for i in range(128):
            # np.random.seed(100+i)

            item = executeEpisodeEndless(i, tainer)
            g.append(item)
            states.append(torch.tensor(next(item)))

        while True:
            # print('before remote1')
            prob, v = ray.get(infer_service.infer.remote(states))
            # print(prob, v)
            for i in range(len(g)):
                # try:
                states[i] = torch.tensor(g[i].send((prob[i], v[i])))
                # except StopIteration:
                #     g[i] = executeEpisodeEndless(i, tainer)
                #     states[i] = next(g[i])
    except Exception:
        print(traceback.format_exc())


@ray.remote(num_cpus=0.1, num_gpus=0.1)
class Infer_srv:
    def __init__(self):
        self.nnet = Policy_Value().to("cuda:0").half()
        self.nnet.eval()

    def infer(self, data):
        try:
            # print('in infer1', torch.cuda.is_available())
            input_tensor = (
                torch.stack(data).permute(0, 3, 1, 2).to("cuda:0").to(torch.float16)
            )
            # print('in infer2')
            with torch.no_grad():
                prob, v = self.nnet(input_tensor)
                prob = prob.cpu().numpy()
                v = v.cpu()
        except Exception:
            print(traceback.format_exc())
        return prob, v

    def load_weight(self, weight):
        # weight = ray.get(weight)
        try:
            self.nnet.load_state_dict(weight)
        except Exception:
            print(traceback.format_exc())


@ray.remote(num_cpus=0.1, num_gpus=0.2)
class Train_srv:
    def __init__(self, infer_services):
        self.nnet = Policy_Value().to("cuda:0")
        self.opt = torch.optim.AdamW(params=self.nnet.parameters(), lr=1e-4)
        self.infer_services = infer_services
        self.batchsize = 1024
        self.mse_loss = torch.nn.MSELoss()
        self.kl_loss = torch.nn.KLDivLoss()
        self.samples = deque(maxlen=50000)
        self.sn = 0
        self.epoch = 0

    def _train(self):
        print("Train11")
        opt = self.opt
        batch = random.choices(self.samples, k=self.batchsize)
        states = []
        pis = []
        vs = []
        cnt = 4
        for item in batch:
            # print(type(item[0]), item[0])
            # sys.exit()
            s = torch.tensor(item[0], dtype=torch.float32)
            pi = torch.tensor(item[1], dtype=torch.float32).reshape(15, 15)
            v = torch.tensor(item[2], dtype=torch.float32)
            if cnt > 0:
                # print("tsample", s, pi, v)
                cnt -= 1
            r = random.choice([-1, 0, 1, 2])
            torch.rot90(s, r)
            torch.rot90(pi, r)
            if random.random() > 0.5:
                s = s.flip(0)
                pi = pi.flip(0)
            if random.random() > 0.5:
                s = s.flip(1)
                pi = pi.flip(1)
            pi = pi.flatten()

            states.append(s)
            pis.append(pi)
            vs.append(v)
        states = torch.stack(states).permute(0, 3, 1, 2).to("cuda:0")
        pis = torch.stack(pis).to("cuda:0")
        vs = torch.vstack(vs).to("cuda:0")

        self.nnet.train()
        pred_pis, pred_vs = self.nnet(states)
        # pi_loss = -torch.mean(
        #    pis.matmul(torch.log(torch.clip(pred_pis, 1e-9, 1 - 1e-9).transpose(0, 1)))
        # )
        pi_loss = -torch.mean(
            (pis * torch.log(torch.clip(pred_pis, 1e-9, 1 - 1e-9))).sum(1)
        )
        v_loss = self.mse_loss(pred_vs, vs)
        print(f"loss: {pi_loss.tolist():.03f}, {v_loss.tolist():.03f}")
        loss = pi_loss + v_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Train12")

    def push_samples(self, samples):
        try:
            self.samples.extend(samples)
            self.sn += 1
            if self.sn == 200:
                self.sn = 0
                self.train()
        except Exception:
            print(traceback.format_exc())

    def train(self):
        print("Train1")
        for i in range(70):
            self._train()
        print("Train2")
        time.sleep(4)
        torch.save(self.nnet.state_dict(), f"models/{self.epoch}.pt")
        print(f"saved {self.epoch}.pt file...")
        weight = self.nnet.state_dict()
        for item in self.infer_services:
            item.load_weight.remote(weight)
        print("Train3")
        self.epoch += 1



def main():
    print("GHB1")
    infer_services = [Infer_srv.remote() for _ in range(4)]
    print("GHB2")
    tainer = Train_srv.remote(infer_services)
    print("GHB3")
    s = []
    for i in range(32):
        s.append(simbatch.remote(infer_services[i % 4], tainer))
    print("GHB4")
    ray.wait(s)
    print("GHB5")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

from multiprocessing.connection import Client
import numpy as np
from mcts5 import MCTS
from game2 import GoBang
import time
from model4 import Policy_Value
import torch
import ray
import random
from collections import deque
import os
import socket
import traceback
from io import BytesIO

print(socket.gethostname(), os.getcwd())
#os.environ["RAY_LOG_TO_STDERR"] = "1"
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
    board_record = np.zeros((game.size, game.size), dtype=np.uint8)
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
        for i in range(1000):
            yield from mcts.search(state)
        pi = mcts.pi(state, tau)
        samples.append([state, pi, None])
        action = np.random.choice(len(pi), p=pi)
        next_state, isend, reward = game.next_state(state, action)
        y = action // game.size
        x = action % game.size
        ####
        if board_record[y, x] != 0:
            print("####GHB Found Replicate move!", y, x, cnt, board_record)
            print(state[0])
            print('sample\n', samples)
            import sys
            sys.exit()
        ####
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
        print(
            f"{epid} got trajectory",
            "\n",
            board_record,
            "\n",
            trajectory[-1][2],
            trajectory[-2][2],
            trajectory[-3][2],
            trajectory[-4][2],
        )
        if len(trajectory) % 2 == 1:
            print('black win!')
        else:
            print('white win!')

        tainer.push_samples.remote(trajectory)


def send_and_recv(addr, data):
    #print('in send_and_recv', addr)
    with Client(addr, authkey=b"secret password123") as conn:
        conn.send(data)
        return conn.recv()


@ray.remote(num_cpus=1)
def simbatch(infer_srv_addr, tainer):
    try:
        states = []
        g = []
        print("GHB in simbatch")
        for i in range(128):
            # np.random.seed(100+i)
            item = executeEpisodeEndless(i, tainer)
            g.append(item)
            states.append(next(item))

        while True:
            #print("before remote1")
            prob, v = send_and_recv(infer_srv_addr, ("infer", states))
            # print(prob, v)
            for i in range(len(g)):
                # try:
                states[i] = g[i].send((prob[i], v[i]))
                # except StopIteration:
                #     g[i] = executeEpisodeEndless(i, tainer)
                #     states[i] = next(g[i])
    except Exception:
        print(traceback.format_exc())


def get_onnx_bytes_from_remote(model):
    with Client(("192.168.5.6", 6000), authkey=b"secret password123") as conn:
        f = BytesIO()
        # model.load_state_dict(torch.load('models/224.pt'))
        model.eval()
        torch.save(model.state_dict(), f, _use_new_zipfile_serialization=False)
        print("get_onnx_bytes len:", len(f.getvalue()))
        conn.send(f.getvalue())
        conn.send(128)
        onnxbytes = conn.recv()
        return onnxbytes


@ray.remote(num_cpus=1, num_gpus=0.2)
class Train_srv:
    def __init__(self):
        print("Train1100")
        pass

    def myinit(self, infer_srv_addresses: None):
        try:
            print("Train110")
            self.nnet = Policy_Value().to("cuda:0")
            self.opt = torch.optim.AdamW(params=self.nnet.parameters(), lr=1e-3)
            self.infer_srv_addresses = infer_srv_addresses
            self.batchsize = 1024
            self.mse_loss = torch.nn.MSELoss()
            self.kl_loss = torch.nn.KLDivLoss()
            self.blackwin_samples = deque(maxlen=30000)
            self.whitewin_samples = deque(maxlen=30000)
            self.sn = 0
            self.epoch = 0
            onnxbytes = get_onnx_bytes_from_remote(self.nnet)
            [send_and_recv(addr, ("load_onnx", onnxbytes)) for addr in self.infer_srv_addresses]
            print("GGG after init")
        except Exception:
            print(traceback.format_exc())

    def _train(self):
        print("Train11")
        opt = self.opt
        batch_blackwin = random.choices(self.blackwin_samples, k=self.batchsize // 2)
        batch_whitewin = random.choices(self.whitewin_samples, k=self.batchsize // 2)
        batch = []
        batch.extend(batch_blackwin)
        batch.extend(batch_whitewin)
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
        # pi_loss = -torch.mean(pis.matmul(torch.log(torch.clip(pred_pis, 1e-9, 1 - 1e-9).transpose(0, 1))))
        pi_loss = -torch.mean((pis * torch.log(torch.clip(pred_pis, 1e-9, 1 - 1e-9))).sum(1))
        v_loss = self.mse_loss(pred_vs, vs)
        print(f"loss: {pi_loss.tolist():.03f}, {v_loss.tolist():.03f}")
        loss = pi_loss + v_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Train12")

    def push_samples(self, samples):
        try:
            if len(samples) % 2 == 1:
                self.blackwin_samples.extend(samples)
            else:
                self.whitewin_samples.extend(samples)
            self.sn += 1
            if self.sn == 200:
                self.sn = 0
                self.train()
        except Exception:
            print(traceback.format_exc())

    def train(self):
        print("Train1")
        for i in range(50):
            self._train()
        print("Train2")
        #time.sleep(4)
        torch.save(self.nnet.state_dict(), f"models/{self.epoch}.pt")
        print(f"saved {self.epoch}.pt file...")
        onnxbytes = get_onnx_bytes_from_remote(self.nnet)
        [send_and_recv(addr, ("load_onnx", onnxbytes)) for addr in self.infer_srv_addresses]
        print("Train3")
        self.epoch += 1


def main():
    print("GHB1")
    infer_srv_cnt = 4
    infer_srv_addresses = [("192.168.5.106", 6001 + i) for i in range(infer_srv_cnt)]
    print("GHB2")
    tainer = Train_srv.remote()
    ray.wait([tainer.myinit.remote(infer_srv_addresses=infer_srv_addresses)])
    print("GHB3")
    s = []
    for i in range(20):
        s.append(simbatch.remote(infer_srv_addresses[i % infer_srv_cnt], tainer))
    print("GHB4")
    ray.wait(s)
    print("GHB5")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

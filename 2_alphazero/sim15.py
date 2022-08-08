from multiprocessing.connection import Client
import numpy as np
from mcts7 import MCTS
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
# import pymongo
import redis

print(socket.gethostname(), os.getcwd())
ray.init(address="auto", _node_ip_address="192.168.5.6")
# dbclient = pymongo.MongoClient("mongodb://root:mongomprc12@localhost:27017/")
# gobang_db = dbclient["gobang"]
# gobang_col1 = gobang_db["kifu2"]
GBOARD_SIZE = 15
TRAIN_BATCHSIZE = 512

def executeEpisode(game, epid, rdb):
    state = game.start_state()
    samples = []
    cnt = 0
    board_record = np.zeros((game.size, game.size), dtype=np.uint8)
    stime = time.time()
    tau = 1.0
    mcts = MCTS(game, rdb, c_puct=5, dirichlet_alpha=0.05, dirichlet_weight=0.25, reward_scale=10.0)
    while True:
        cnt += 1
        if cnt > 12:
            tau = 0.1
        if epid == 0:
            print("GHB", cnt, f"tau:{tau:.2f}, {time.time()-stime: .2f}")
        stime = time.time()
        for i in range(1000):
            yield from mcts.search(state, cnt - 1)
        pi = mcts.pi(state, cnt - 1, tau)
        mcts.clear_satistic(cnt - 1)
        samples.append([state, pi, None])
        action = np.random.choice(len(pi), p=pi)
        next_state, isend, reward = game.next_state(state, action)
        y = action // game.size
        x = action % game.size
        ####
        if board_record[y, x] != 0:
            print("####GHB Found Replicate move!", y, x, cnt, board_record)
            print(state[0])
            print("sample\n", samples)
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


def executeEpisodeEndless(epid, tainer, rdb):
    game = GoBang(size=15)
    traj_cnt = 0
    while True:
        print("executeEpisodeEndless1", epid, traj_cnt)
        trajectory, board_record = yield from executeEpisode(game, epid, rdb)
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
        traj_cnt += 1
        if len(trajectory) % 2 == 1:
            winner = "black"
        else:
            winner = "white"
        # gobang_col1.insert_one(
        #     {"kifu": board_record.tolist(), "winner": winner, "len": len(trajectory), "traj_cnt": traj_cnt}
        # )
        print(f"{winner} win!", traj_cnt, len(trajectory))

        tainer.push_games.remote(trajectory)


def send_and_recv(addr, data):
    # print('in send_and_recv', addr)
    with Client(addr, authkey=b"secret password123") as conn:
        conn.send(data)
        return conn.recv()


@ray.remote(num_cpus=0.5)
def simbatch(infer_srv_addr, tainer):
    try:
        # rdb = redis.Redis(host="192.168.5.6", port=1001, db=2)
        rdb = None
        states = []
        g = []
        print("GHB in simbatch")
        for i in range(128):
            # np.random.seed(100+i)
            item = executeEpisodeEndless(i, tainer, rdb)
            g.append(item)
            states.append(next(item))

        while True:
            # print("before remote1")
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
            # self.rdb = redis.Redis(host="192.168.5.6", port=1001, db=2)
            self.nnet = Policy_Value().to("cuda:0")
            self.opt = torch.optim.AdamW(params=self.nnet.parameters(), lr=1e-4)
            self.infer_srv_addresses = infer_srv_addresses
            self.batchsize = 2048
            self.mse_loss = torch.nn.MSELoss()
            self.kl_loss = torch.nn.KLDivLoss()
            self.blackwin_games = deque(maxlen=50_000)
            self.whitewin_games = deque(maxlen=50_000)
            self.sn = 0
            self.epoch = 0
            onnxbytes = get_onnx_bytes_from_remote(self.nnet)
            [send_and_recv(addr, ("load_onnx", onnxbytes)) for addr in self.infer_srv_addresses]
            print("GGG after init")
        except Exception:
            print(traceback.format_exc())

    def _train(self, states, pis, vs):
        print("Train11")
        opt = self.opt
        idxes = list(range(len(states)))
        random.shuffle(idxes)

        self.nnet.train()
        for i in range(len(idxes) // TRAIN_BATCHSIZE):
            idx = idxes[i * TRAIN_BATCHSIZE : (i + 1) * TRAIN_BATCHSIZE]
            state = states[idx]
            pi = pis[idx]
            v = vs[idx]
            # print("G123", idx, pi.shape, v.shape)

            state = state.permute(0, 3, 1, 2).to("cuda:0")
            pi = pi.to("cuda:0")
            v = v.to("cuda:0")

            pred_pi, pred_v = self.nnet(state)
            pi_loss = -torch.mean((pi * torch.log(torch.clip(pred_pi, 1e-9, 1 - 1e-9))).sum(1))
            v_loss = self.mse_loss(pred_v, v)
            print(f"loss: {pi_loss.tolist():.03f}, {v_loss.tolist():.03f}")
            loss = pi_loss + v_loss * 1.0
            opt.zero_grad()
            loss.backward()
            opt.step()
            print("Train12")

    def push_games(self, trajectory):
        try:
            if len(trajectory) % 2 == 1:
                self.blackwin_games.append(trajectory)
            else:
                self.whitewin_games.append(trajectory)
            self.sn += 1
            if self.sn == 100:
                self.sn = 0
                self.train()
        except Exception:
            print(traceback.format_exc())

    def train(self):
        print("Train1")
        states = []
        pis = []
        vs = []
        game_samples = list(self.blackwin_games)
        game_samples.extend(list(self.whitewin_games))
        for traj in game_samples:
            for step in traj:
                s = torch.tensor(step[0], dtype=torch.float32)
                pi = torch.tensor(step[1], dtype=torch.float32).reshape(GBOARD_SIZE, GBOARD_SIZE)
                v = torch.tensor(step[2], dtype=torch.float32)

                for r in range(4):
                    states.append(s.rot90(r))
                    pis.append(pi.rot90(r).flatten())
                    vs.append(v.clone())

                    states.append(s.rot90(r).flip(0))
                    pis.append(pi.rot90(r).flip(0).flatten())
                    vs.append(v.clone())

        states = torch.stack(states)
        pis = torch.stack(pis)
        vs = torch.vstack(vs)

        for i in range(2):
            self._train(states, pis, vs)
        print("Train2")
        # time.sleep(4)
        torch.save(self.nnet.state_dict(), f"models/{self.epoch}.pt")
        print(f"saved {self.epoch}.pt file...")
        onnxbytes = get_onnx_bytes_from_remote(self.nnet)
        [send_and_recv(addr, ("load_onnx", onnxbytes)) for addr in self.infer_srv_addresses]
        print("Train3")
        # self.rdb.flushdb()
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
    for i in range(8):
        s.append(simbatch.remote(infer_srv_addresses[i % infer_srv_cnt], tainer))
    print("GHB4")
    ray.wait(s)
    print("GHB5")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

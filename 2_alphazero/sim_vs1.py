from multiprocessing.connection import Client
import numpy as np
from mctsvl import MCTS
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

print(socket.gethostname(), os.getcwd())
ray.init(address="auto", _node_ip_address="192.168.5.6")

GBOARD_SIZE = 15
TRAIN_BATCHSIZE = 2048
INFER_BATCHSIZE = 64
INFER_SRV_CNT = 4


@ray.remote(num_cpus=0.5)
def execute1Episode(selfplay, mcts_search_cnt, mcts_parallel_cnt, infer_srv_addr_black, infer_srv_addr_white):
    try:
        game = GoBang(size=GBOARD_SIZE)
        state = game.start_state()
        samples = []
        play_step = 0
        board_record = np.zeros((game.size, game.size), dtype=np.uint8)
        stime = time.time()
        tau = 0.8
        cur_player = 0  # black
        while True:
            mcts = MCTS(
                game, selfplay=selfplay, c_puct=5, dirichlet_alpha=0.05, dirichlet_weight=0.25, reward_scale=10.0
            )
            play_step += 1
            if play_step > 2:
                tau = max(0.05, tau * 0.95)
            print("GHB", play_step, f"tau:{tau:.2f}, {time.time()-stime: .2f}")
            stime = time.time()
            search_cnt = 0
            while search_cnt < mcts_search_cnt:
                gen_arr = []
                infer_req = []
                while len(infer_req) < mcts_parallel_cnt:
                    search_cnt += 1
                    gen = mcts.search(state)
                    try:
                        infer_req.append(next(gen))
                        gen_arr.append(gen)
                    except StopIteration:
                        pass
                infer_srv_addr = infer_srv_addr_black if cur_player == 0 else infer_srv_addr_white
                prob, v = send_and_recv(infer_srv_addr, ("infer", infer_req))
                for i in range(len(gen_arr)):
                    try:
                        gen_arr[i].send((prob[i], v[i]))
                    except StopIteration:
                        pass

            pi = mcts.pi(state, tau)
            samples.append([state, pi, None])
            action = np.random.choice(len(pi), p=pi)
            next_state, isend, reward = game.next_state(state, action)
            cur_player = (cur_player + 1) % 2
            y = action // game.size
            x = action % game.size
            ####
            if board_record[y, x] != 0:
                print("####GHB Found Replicate move!", y, x, play_step, board_record)
                print(state[0])
                print("sample\n", samples)
                import sys

                sys.exit()
            ####
            board_record[y, x] = play_step
            if isend:
                v = reward
                for j in reversed(range(len(samples))):
                    samples[j][2] = v
                    v = -v
                return samples, board_record
            else:
                state = next_state
    except Exception:
        print(traceback.format_exc())


def check_winrate():
    pass


def send_and_recv(addr, data):
    # print('in send_and_recv', addr)
    with Client(addr, authkey=b"secret password123") as conn:
        conn.send(data)
        return conn.recv()


def get_model_bytes(model):
    f = BytesIO()
    model.eval()
    torch.save(model.state_dict(), f, _use_new_zipfile_serialization=False)
    return f.getvalue()


def get_onnx_bytes_from_remote(model_bytes, batch_size):
    with Client(("192.168.5.6", 6000), authkey=b"secret password123") as conn:
        conn.send(model_bytes)
        conn.send(batch_size)
        onnxbytes = conn.recv()
        return onnxbytes


@ray.remote(num_cpus=1, num_gpus=0.2)
class Train_srv:
    def __init__(self):
        print("Train1100")
        pass

    def myinit(self, nnet_bytes):
        try:
            print("Train110")
            nnet = Policy_Value()
            nnet.load_state_dict(torch.load(BytesIO(nnet_bytes), map_location="cpu"))
            self.nnet = nnet.to("cuda:0")
            self.opt = torch.optim.AdamW(params=self.nnet.parameters(), lr=1e-4)
            self.batchsize = TRAIN_BATCHSIZE
            self.mse_loss = torch.nn.MSELoss()
            self.kl_loss = torch.nn.KLDivLoss()
            self.epoch = 0
            print("GGG after init")
        except Exception:
            print(traceback.format_exc())

    def _train(self, states, pis, vs):
        print("Train11")
        opt = self.opt
        idxes = list(range(len(states)))
        random.shuffle(idxes)

        for i in range(len(idxes) // TRAIN_BATCHSIZE):
            idx = idxes[i * TRAIN_BATCHSIZE : (i + 1) * TRAIN_BATCHSIZE]
            state = states[idx]
            pi = pis[idx]
            v = vs[idx]
            print("G123", idx.shape, pi.shape, v.shape)

            state = state.permute(0, 3, 1, 2).to("cuda:0")
            pi = pi.to("cuda:0")
            v = v.to("cuda:0")

            self.nnet.train()
            pred_pi, pred_v = self.nnet(state)
            pi_loss = -torch.mean((pi * torch.log(torch.clip(pred_pi, 1e-9, 1 - 1e-9))).sum(1))
            v_loss = self.mse_loss(pred_v, v)
            print(f"loss: {pi_loss.tolist():.03f}, {v_loss.tolist():.03f}")
            loss = pi_loss + v_loss * 1.0
            opt.zero_grad()
            loss.backward()
            opt.step()
            print("Train12")

    def train(self, game_samples, train_num):
        print("Train1")
        states = []
        pis = []
        vs = []
        for traj in game_samples:
            for step in traj:
                s = torch.tensor(step[0], dtype=torch.float32)
                pi = torch.tensor(step[1], dtype=torch.float32).reshape(GBOARD_SIZE, GBOARD_SIZE)
                v = torch.tensor(step[2], dtype=torch.float32)

                for r in range(4):
                    states.append(s.rot90(r))
                    pis.append(pi.rot90(r).flatten())
                    vs.append(v)

                    states.append(s.rot90(r).flip(0))
                    pis.append(pi.rot90(r).flip(0).flatten())
                    vs.append(v)

        states = torch.stack(states)
        pis = torch.stack(pis)
        vs = torch.vstack(vs)

        for i in range(train_num):
            self._train(states, pis, vs)
        return get_model_bytes(self.nnet)


def main():
    print("GHB1")
    infer_srv_cnt = 4
    infer_srv_addresses = [("192.168.5.106", 6001 + i) for i in range(infer_srv_cnt)]
    print("GHB2")
    nnet = Policy_Value()
    nnet_bytes = get_model_bytes(nnet)
    onnx_bytes = get_onnx_bytes_from_remote(nnet_bytes, INFER_BATCHSIZE)
    [send_and_recv(addr, ("load_onnx", onnx_bytes)) for addr in infer_srv_addresses]
    print("GHB21")
    trainer = Train_srv.remote()
    ray.wait([trainer.myinit.remote(nnet_bytes)])
    print("GHB3")
    whitewin_game_queue = deque(maxlen=50_000)
    blackwin_game_queue = deque(maxlen=50_000)
    epoch = 0
    while True:
        # generate sampels
        s = []
        for i in range(1):
            infer_srv_addr_black = infer_srv_addresses[i % len(infer_srv_addresses)]
            infer_srv_addr_white = infer_srv_addr_black
            s.append(execute1Episode.remote(True, 1000, INFER_BATCHSIZE, infer_srv_addr_black, infer_srv_addr_white))
        print("GHB4")
        samples = ray.get(s)
        print("GHB5", samples)
        for item in samples:
            traj = item[0]
            if len(traj) % 2 == 0:
                whitewin_game_queue.append(traj)
            else:
                blackwin_game_queue.append(traj)

        # train
        game_samples = list(whitewin_game_queue)
        game_samples.extend(list(blackwin_game_queue))
        new_model_bytes = trainer.train(game_samples, 2)
        new_onnx_bytes = get_onnx_bytes_from_remote(new_model_bytes, INFER_BATCHSIZE)
        [send_and_recv(addr, ("load_onnx", new_onnx_bytes)) for addr in infer_srv_addresses[infer_srv_cnt // 2 :]]

        # evaluate new model
        half = infer_srv_cnt // 2
        s = []
        for i in range(400):
            infer_srv_addr_black = infer_srv_addresses[i % len(infer_srv_addresses)]
            infer_srv_addr_white = infer_srv_addresses[(i + half) % len(infer_srv_addresses)]
            s.append(execute1Episode.remote(False, 1000, INFER_BATCHSIZE, infer_srv_addr_black, infer_srv_addr_white))
        samples = ray.get(s)
        white_win_cnt = 0
        black_win_cnt = 0
        for i, sample in enumerate(samples):
            traj, board_record = sample
            if i % len(infer_srv_addresses) < half:
                # new model play white
                if len(traj) % 2 == 0:
                    white_win_cnt += 1
            elif i % len(infer_srv_addresses) >= half:
                # new model play black
                if len(traj) % 2 == 1:
                    black_win_cnt += 1
        print(
            "evalute result, "
            f"white:{black_win_cnt / 200:.3f}, "
            f"black:{white_win_cnt / 200:.3f}, "
            f"total:{(black_win_cnt + white_win_cnt) / 400:.3f}"
        )
        # update or recover infer model
        win_rate = (black_win_cnt + white_win_cnt) / 400
        if win_rate > 0.55:
            [send_and_recv(addr, ("load_onnx", new_onnx_bytes)) for addr in infer_srv_addresses[: infer_srv_cnt // 2]]
            with open(f"models/{epoch}.pt", "wb") as f:
                f.write(new_model_bytes)
            onnx_bytes = new_onnx_bytes
        else:
            [send_and_recv(addr, ("load_onnx", onnx_bytes)) for addr in infer_srv_addresses[infer_srv_cnt // 2 :]]

        epoch += 1


if __name__ == "__main__":
    main()

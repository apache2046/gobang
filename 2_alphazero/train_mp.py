from mcts import MCTS
from game import GoBang
from model import Policy_Value
import numpy as np
import torch
import random
import torch.multiprocessing as mp
import sys

# import multiprocessing as mp
import queue

game = GoBang()
# batchsize = 32
batchsize = 256

mse_loss = torch.nn.MSELoss()
kl_loss = torch.nn.KLDivLoss()
num_processes = 10


def train(samples, nnet, opt):
    batch = random.choices(samples, k=batchsize)
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
            print("tsample", s, pi, v)
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
    vs = torch.stack(vs).to("cuda:0")

    nnet.train()
    pred_pis, pred_vs = nnet(states)
    # print(pis.shape, pred_pis.shape, vs.shape, pred_vs.shape)
    # pi_loss = -torch.mean(pis.matmul(torch.log(pred_pis).transpose(0, 1)))
    # pi_loss = kl_loss(pred_pis, pis)
    pi_loss = -torch.mean(pis.matmul(torch.log(torch.clip(pred_pis, 1e-9, 1 - 1e-9).transpose(0, 1))))
    v_loss = mse_loss(pred_vs, vs)
    print(f"loss: {pi_loss.tolist():.03f}, {v_loss.tolist():.03f}")
    loss = pi_loss + v_loss
    opt.zero_grad()
    loss.backward()
    opt.step()


def gather_and_train(nnet, sample_queue: mp.Queue, train_complete: mp.Condition, sample_complete: mp.Value):
    opt = torch.optim.AdamW(params=nnet.parameters(), lr=1e-4)
    epoch = 0
    while True:
        epoch += 1
        samples = []
        while sample_complete.value < num_processes:
            try:
                data = sample_queue.get(timeout=1)
            except queue.Empty:
                data = None
            if data is not None:
                samples.extend(data)

        sample_complete.acquire()
        sample_complete.value = 0
        sample_complete.release()

        for i in range(200):
            train(samples, nnet, opt)

        torch.save(nnet.state_dict(), f"models/{epoch}.pt")
        print(f"saved {epoch}.pt file...")

        train_complete.acquire()
        train_complete.notify_all()
        train_complete.release()


def executeEpisode(nnet):
    mcts = MCTS(game, c_puct=0.5)
    state = game.start_state()
    samples = []
    cnt = 0
    while True:
        cnt += 1
        # print("GHB", cnt)
        for i in range(2000):
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


def getSample(rank, nnet, sample_queue: mp.Queue, train_complete: mp.Condition, sample_complete: mp.Value):
    np.random.seed(rank)
    while True:
        #for episode in range(20):
        for episode in range(100):
            sample = executeEpisode(nnet)
            sample_queue.put(sample)
            print('episode done', rank, episode)

        sample_complete.acquire()
        sample_complete.value += 1
        sample_complete.release()

        train_complete.acquire()
        train_complete.wait()
        train_complete.release()


def main():
    mp.set_start_method("spawn")

    sample_queue = mp.Queue()
    train_complete = mp.Condition()
    sample_complete = mp.Value("i", 0)

    nnet = Policy_Value().to("cuda")
    nnet.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=getSample, args=(rank, nnet, sample_queue, train_complete, sample_complete))
        p.daemon = True
        p.start()
        processes.append(p)

    gather_and_train(nnet, sample_queue, train_complete, sample_complete)


if __name__ == "__main__":
    main()

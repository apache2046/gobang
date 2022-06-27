from mcts import MCTS
from game import GoBang
from model import Policy_Value
import numpy as np
import torch

nnet = Policy_Value().to('cuda:0')
game = GoBang()
batchsize = 32
opt = torch.optim.AdamW(params=nnet.parameters(), lr=1e-3)

mse_loss = torch.nn.MSELoss()


def train(samples):
    nnet.train()
    for i in range(100):
        batch = np.random.choice(samples, batchsize)
        states = []
        pis = []
        vs = []
        for item in batch:
            states.append(torch.tensor(item[0], dtype=torch.float32))
            pis.append(torch.tensor(item[1], dtype=torch.float32))
            vs.append(torch.tensor(item[2], dtype=torch.float32))
        states = torch.stack(states).permute(0, 3, 1, 2).to('cuda:0')
        pis = torch.stack(pis).to('cuda:0')
        vs = torch.stack(vs).to('cuda:0')

        pred_pis, pred_vs = nnet(states)
        pi_loss = -torch.mean(pis.dot(torch.log(pred_pis)))
        v_loss = mse_loss(pred_vs, vs)
        loss = pi_loss + v_loss
        opt.zero_grad()
        loss.backward()
        opt.step()


def policyIterSP():
    samples = []
    for i in range(100):
        samples.append(*executeEpisode())
    train(samples)


def executeEpisode():
    mcts = MCTS(game)
    state = game.start_state()
    samples = []

    while True:
        for i in range(1000):
            mcts.search(state, nnet)
        pi = mcts.pi(state)
        samples.append([state, pi, None])
        action = np.random.choice(len(pi), p=pi)
        next_state, iswin = game.next_state(state, action)
        if iswin:
            v = 1
            for j in reversed(range(len(samples))):
                samples[j][2] = v
                v = -v
            return samples
        else:
            state = next_state


for i in range(100):
    policyIterSP()

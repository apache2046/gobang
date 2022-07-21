from collections import defaultdict
import game
from math import sqrt
import numpy as np

class MCTS:
    def __init__(self, game: game.GoBang, c_puct=0.001, tau=1):
        # super.__init__(self)
        self.ps = {}
        # self.qsa = {}
        # self.wsa = defaultdict(int)
        # self.nsa = defaultdict(int)
        self.wsa = {}
        self.nsa = {}
        self.ns = defaultdict(int)
        self.c_puct = c_puct
        self.tau = tau
        self.game = game

    def search(self, state, net, level=0):
        sk = self.game.state2key(state)
        if self.ns[sk] == 0:
            self.ns[sk] = 1

            # p, v = yield state
            p, v = net.infer(state)
            w = self.game.size
            self.ps[sk] = p.reshape((w, w))
            self.wsa[sk] = np.zeros_like(self.ps[sk])
            self.nsa[sk] = np.zeros_like(self.ps[sk])

            # print(self.ps[sk], v)
            # self.ps[sk] = np.ones(225) / 225
            # v = 0.01
            return -v

        # max_u, best_a = -float("inf"), -1
        # for a in self.game.valid_positions(state):
        #     qsa = self.wsa[(sk, a)] / (self.nsa[(sk, a)] + 1)
        #     u = qsa + self.c_puct * self.ps[sk][a] * sqrt(self.ns[sk]) / (self.nsa[(sk, a)] + 1)
        #     if u > max_u:
        #         max_u = u
        #         best_a = a
        # a = best_a
        u = self.wsa[sk] / (self.nsa[sk] + 1) + self.c_puct * self.ps[sk] * sqrt(self.ns[sk]) / (self.nsa[sk] + 1)
        invalid_pos = (state[:, :, 0] + state[:, :, 1] > 0)
        # print("u", u, invalid_pos)
        u[invalid_pos] = -10000
        a = u.argmax()
        # print(a)
        # if a == -1:
        #     print(state, max_u, u)
        #     raise Exception("aaaa")
        state_next, isend, reward = self.game.next_state(state, a)
        # print(iswin, level, a // 15, a % 15, self.ps[sk][a], self.wsa[(sk, a)], self.nsa[(sk, a)], self.ns[sk])
        #print(isend, reward, level, a // 15, a % 15)
        if isend:
            v = reward * 100
            # print(reward, level, a // 15, a % 15, state_next)
        else:
            v = self.search(state_next, net, level + 1)

        # self.qsa[(sk, a)] = (N[s][a] * Q[s][a] + v) / (N[s][a] + 1)
        y = a // 15
        x = a % 15
        self.wsa[sk][y, x] += v
        self.nsa[sk][y, x] += 1
        self.ns[sk] += 1
        return -v

    def pi(self, state):
        pi = np.zeros_like(state[:, :, 0], dtype=np.float32).flatten()
        sk = self.game.state2key(state)
        # for a in self.game.valid_positions(state):
        #     pi[a] = (self.nsa[sk, a)] ** (1 / self.tau)) / (self.ns[sk] ** (1 / self.tau))
        # return pi / pi.sum()
        invalid_pos = (state[:, :, 0] + state[:, :, 1] > 0)
        pi = (self.nsa[sk] ** (1 / self.tau)) / (self.ns[sk] ** (1 / self.tau))
        pi[invalid_pos] = 0
        pi = pi / pi.sum()
        return pi.reshape((225,))


if __name__ == "__main__":
    print("OK")
    print(game.GoBang)

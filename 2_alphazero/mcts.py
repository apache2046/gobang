from collections import defaultdict
import game
from math import sqrt
import numpy as np


class MCTS:
    def __init__(self, game: game.GoBang, c_puct=1, tau=1):
        super.__init__(self)
        self.ps = {}
        # self.qsa = {}
        self.wsa = defaultdict(int)
        self.nsa = defaultdict(int)
        self.ns = defaultdict(int)
        self.c_puct = c_puct
        self.tau = tau
        self.game = game

    def search(self, state, net):
        sk = self.game.state2key(state)
        if self.ns[sk] == 0:
            self.ns[sk] = 1
            self.ps[sk], v = net(state)
            return -v

        max_u, best_a = -float("inf"), -1
        for a in self.game.valid_positions(state):
            qsa = self.wsa[(sk, *a)] / (self.nsa[(sk, *a)] + 1)
            u = qsa + self.c_puct * self.ps[sk][a] * sqrt(self.ns[sk]) / (self.nsa[(sk, *a)] + 1)
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a

        state_next, iswin = self.game.next_state(state, a)
        if iswin:
            v = 1
        else:
            v = self.search(state_next, net)

        # self.qsa[(sk, *a)] = (N[s][a] * Q[s][a] + v) / (N[s][a] + 1)
        self.wsa[(sk, *a)] += v
        self.nsa[(sk, *a)] += 1
        self.ns[sk] += 1
        return -v

    def pi(self, state):
        pi = np.zeros_like(state, dtype=np.float32)
        sk = self.game.state2key(state)
        for a in game.valid_positions(state):
            pi[a] = (self.nsa[(sk, *a)] ** (1 / self.tau)) / (self.ns[sk] ** (1 / self.tau))
        return pi.flatten()


if __name__ == "__main__":
    print("OK")
    print(game.GoBang)

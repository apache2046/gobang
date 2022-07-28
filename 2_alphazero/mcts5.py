from collections import defaultdict
import game
from math import sqrt
import numpy as np
import random


class MCTS:
    def __init__(
        self,
        game: game.GoBang,
        selfplay=True,
        c_puct=0.001,
        tau=1,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.2,
        random_dihedral_reflection=True,
    ):
        self.ps = {}
        self.wsa = {}
        self.nsa = {}
        self.ns = defaultdict(int)
        self.c_puct = c_puct
        self.tau = tau
        self.game = game
        self.selfplay = selfplay
        self.random_dihedral_reflection = random_dihedral_reflection
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

    def search(self, state, level=0):
        sk = self.game.state2key(state)
        # print("G1", level)
        w = self.game.size
        if self.ns[sk] == 0:
            # print("G2", level)
            self.ns[sk] = 1
            if self.random_dihedral_reflection:
                rot_degree = random.choice([0, 1, 2, 3])
                flip_lr = random.choice([0, 1])
                flip_ud = random.choice([0, 1])
                if rot_degree > 0:
                    state = np.rot90(state, rot_degree)
                if flip_lr > 0:
                    state = np.fliplr(state)
                if flip_ud > 0:
                    state = np.flipud(state)
            p, v = yield state
            p = p.reshape((w, w)).astype(np.float64)
            if self.random_dihedral_reflection:
                if flip_ud > 0:
                    state = np.flipud(state)
                if flip_lr > 0:
                    state = np.fliplr(state)
                if rot_degree > 0:
                    state = np.rot90(state, -rot_degree)

            # print("G2.1", level)
            self.ps[sk] = p
            if level == 0 and self.selfplay:
                # print("G3", level)
                vps = state[:, :, 0] + state[:, :, 1] == 0
                vps_cnt = np.count_nonzero(vps)
                # alpha = (w * w) / vps_cnt * 0.03
                alpha = (w * w) / vps_cnt * self.dirichlet_alpha
                noise = np.random.dirichlet(alpha * np.ones(vps_cnt))
                self.ps[sk][vps] = (1 - self.dirichlet_weight) * self.ps[sk][vps] + self.dirichlet_weight * noise
            # print("G4", level)
            self.wsa[sk] = np.zeros_like(self.ps[sk]).astype(np.float64)
            self.nsa[sk] = np.zeros_like(self.ps[sk]).astype(np.float64)
            return -v

        u = self.wsa[sk] / (self.nsa[sk] + 1) + self.c_puct * self.ps[sk] * sqrt(self.ns[sk]) / (self.nsa[sk] + 1)
        invalid_pos = state[:, :, 0] + state[:, :, 1] > 0
        # print("u", u, invalid_pos)
        u[invalid_pos] = -10000
        a = u.argmax()
        state_next, isend, reward = self.game.next_state(state, a)
        if isend:
            v = reward
        else:
            v = yield from self.search(state_next, level + 1)

        y = a // 15
        x = a % 15
        self.wsa[sk][y, x] += v
        self.nsa[sk][y, x] += 1
        self.ns[sk] += 1
        return -v

    def pi(self, state, tau=1.0):
        pi = np.zeros_like(state[:, :, 0], dtype=np.float32).flatten()
        sk = self.game.state2key(state)
        # for a in self.game.valid_positions(state):
        #     pi[a] = (self.nsa[sk, a)] ** (1 / self.tau)) / (self.ns[sk] ** (1 / self.tau))
        # return pi / pi.sum()
        invalid_pos = state[:, :, 0] + state[:, :, 1] > 0
        # pi = (self.nsa[sk] ** (1 / tau)) / (self.ns[sk] ** (1 / tau))
        # pi = (self.nsa[sk] / self.ns[sk]) ** (1 / tau)
        pi = self.nsa[sk] ** (1 / tau)
        pi[invalid_pos] = 0
        pi = pi / pi.sum()
        return pi.reshape((225,))


if __name__ == "__main__":
    print("OK")
    print(game.GoBang)

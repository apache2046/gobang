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
        reward_scale=1.0,
        v_discount=1.0,
        vloss=3.0,
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
        self.reward_scale = reward_scale
        self.v_discount = v_discount
        self.vloss = vloss

    def search(self, state, level=0):
        sk = self.game.state2key(state)
        # print("G1", level)
        w = self.game.size
        if self.ns[sk] == 0:
            print("G2", level)
            self.ns[sk] = 1
            if self.random_dihedral_reflection:
                transform_state = state.copy()
                rot_degree = random.choice([0, 1, 2, 3])
                flip_ud = random.choice([0, 1])
                if rot_degree > 0:
                    transform_state = np.rot90(transform_state, rot_degree)
                if flip_ud > 0:
                    transform_state = np.flipud(transform_state)
                print("G2.01")
                p, v = yield transform_state
                print("G2.02")
                p = p.reshape((w, w)).astype(np.float64)
                if flip_ud > 0:
                    p = np.flipud(p)
                if rot_degree > 0:
                    p = np.rot90(p, -rot_degree)
            else:
                print("G2.03")
                p, v = yield state
                print("G2.04")
                p = p.reshape((w, w)).astype(np.float64)

            print("G2.1", level)
            self.ps[sk] = p
            if level == 0 and self.selfplay:
                print("G3", level)
                vps = state[:, :, 0] + state[:, :, 1] == 0
                vps_cnt = np.count_nonzero(vps)
                # alpha = (w * w) / vps_cnt * 0.03
                alpha = (w * w) / vps_cnt * self.dirichlet_alpha
                noise = np.random.dirichlet(alpha * np.ones(vps_cnt))
                self.ps[sk][vps] = (1 - self.dirichlet_weight) * self.ps[sk][vps] + self.dirichlet_weight * noise
            print("G4", level)
            self.wsa[sk] = np.zeros_like(self.ps[sk]).astype(np.float64)
            self.nsa[sk] = np.zeros_like(self.ps[sk]).astype(np.float64)
            return -v * self.v_discount

        u = self.wsa[sk] / (self.nsa[sk] + 1) + self.c_puct * self.ps[sk] * sqrt(self.ns[sk]) / (self.nsa[sk] + 1)
        invalid_pos = state[:, :, 0] + state[:, :, 1] > 0
        # print("u", u, invalid_pos)
        u[invalid_pos] = -10000
        a = u.argmax()
        y = a // 15
        x = a % 15
        self.wsa[sk][y, x] -= self.vloss
        self.nsa[sk][y, x] += self.vloss

        state_next, isend, reward = self.game.next_state(state, a)
        if isend:
            v = reward * self.reward_scale
        else:
            v = yield from self.search(state_next, level + 1)

        self.wsa[sk][y, x] += v + self.vloss
        self.nsa[sk][y, x] += 1 - self.vloss
        self.ns[sk] += 1
        return -v * self.v_discount

    def pi(self, state, tau=1.0):
        pi = np.zeros_like(state[:, :, 0], dtype=np.float32).flatten()
        sk = self.game.state2key(state)
        invalid_pos = state[:, :, 0] + state[:, :, 1] > 0
        pi = self.nsa[sk] ** (1 / tau)
        pi[invalid_pos] = 0
        pi = pi / pi.sum()
        return pi.reshape((225,))


if __name__ == "__main__":
    print("OK")
    print(game.GoBang)

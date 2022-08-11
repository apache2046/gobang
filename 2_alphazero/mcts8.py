from collections import defaultdict
import game3
from math import sqrt
import numpy as np
import random


class MCTS:
    def __init__(
        self,
        game: game3.GoBang,
        selfplay=True,
        c_puct=0.001,
        tau=1,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.2,
        random_dihedral_reflection=True,
        reward_scale=1.0,
        v_discount=1.0,
    ):
        self.psa = [dict() for _ in range(225)]
        self.wsa = [dict() for _ in range(225)]
        self.nsa = [dict() for _ in range(225)]
        self.ns = [defaultdict(int) for _ in range(225)]
        self.c_puct = c_puct
        self.tau = tau
        self.game = game
        self.selfplay = selfplay
        self.random_dihedral_reflection = random_dihedral_reflection
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.reward_scale = reward_scale
        self.v_discount = v_discount

    def search(self, state, stones: int, level=0):
        sk = self.game.state2key(state)
        ns = self.ns[stones]
        psa = self.psa[stones]
        nsa = self.nsa[stones]
        wsa = self.wsa[stones]
        # print("G1", level)
        w = self.game.size
        if ns[sk] == 0:
            # print("G2", level)
            ns[sk] = 1
            if self.random_dihedral_reflection:
                transform_state = state.copy()
                rot_degree = random.choice([0, 1, 2, 3])
                flip_ud = random.choice([0, 1])
                if rot_degree > 0:
                    transform_state = np.rot90(transform_state, rot_degree)
                if flip_ud > 0:
                    transform_state = np.flipud(transform_state)

                p, v = yield transform_state

                p = p.reshape((w, w)).astype(np.float32)
                if flip_ud > 0:
                    p = np.flipud(p)
                if rot_degree > 0:
                    p = np.rot90(p, -rot_degree)
                p = p.flatten()
            else:
                p, v = yield state
                # p = p.reshape((w, w)).astype(np.float64)

            psa[sk] = p
            wsa[sk] = np.zeros_like(psa[sk]).astype(np.float16)
            nsa[sk] = np.zeros_like(psa[sk]).astype(np.float16)
            return -v * self.v_discount

        # add dirichlet noise if cur is S0 node
        if level == 0 and self.selfplay:
            # print("G3", level)
            vps = (state[:, :, 0] + state[:, :, 1] == 0).flatten()
            vps_cnt = np.count_nonzero(vps)
            # alpha = (w * w) / vps_cnt * 0.03
            alpha = (w * w) / vps_cnt * self.dirichlet_alpha
            noise = np.random.dirichlet(alpha * np.ones(vps_cnt))
            noise *= psa[sk][vps].max() / noise.max()
            psa[sk][vps] = (1 - self.dirichlet_weight) * psa[sk][vps] + self.dirichlet_weight * noise

        u = wsa[sk] / (nsa[sk] + 1) + self.c_puct * psa[sk] * sqrt(ns[sk]) / (nsa[sk] + 1)
        invalid_pos = (state[:, :, 0] + state[:, :, 1] > 0).flatten()
        # print("u", u, invalid_pos)
        u[invalid_pos] = -100000
        a = u.argmax()
        state_next, isend, reward = self.game.next_state(state, a)
        if isend:
            v = reward * self.reward_scale
        else:
            v = yield from self.search(state_next, stones + 1, level + 1)

        wsa[sk][a] += v
        nsa[sk][a] += 1
        ns[sk] += 1
        return -v * self.v_discount

    def pi(self, state, stones, tau=1.0):
        sk = self.game.state2key(state)
        invalid_pos = (state[:, :, 0] + state[:, :, 1] > 0).flatten()
        if tau == 1.0:
            pi = self.nsa[stones][sk] ** (1 / tau)
            pi[invalid_pos] = 0
            pi = pi / pi.sum()
            return pi
        else:
            pi = np.zeros_like(self.nsa[stones][sk])
            a = self.nsa[stones][sk].argmax()
            pi[a] = 1.0
            return pi

    def clear_satistic(self, stones):
        self.psa[stones] = None
        self.wsa[stones] = None
        self.nsa[stones] = None
        self.ns[stones] = None


if __name__ == "__main__":
    print("OK")
    print(game3.GoBang)

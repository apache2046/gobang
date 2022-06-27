import numpy as np

# This is for trainning
class GoBang:
    def __init__(self, size=15):
        self.size = size
        self.nextstate_cache = {}

    def start_state(self):
        size = self.size

        # 3 panles, black 0, white 1, empty 2, whois next 3 (1 for balck, -1 for white)
        state = np.zeros((size, size, 4), dtype=np.int8)
        state[:, :, 2] = 1  # all empty
        state[:, :, 3] = 1  # next is black
        return state

    def have_five(self, arr, pos):
        x, y = pos
        w = h = self.size

        cnt = 0  # -
        for i in range(1, 5):
            if x - i < 0 or arr[y][x - i] == 0:
                break
            cnt += 1
        for i in range(0, 5):
            if x + i == w or arr[y][x + i] == 0:
                break
            cnt += 1
        if cnt >= 5:
            return True

        cnt = 0  # |
        for i in range(1, 5):
            if y - i < 0 or arr[y - i][x] == 0:
                break
            cnt += 1
        for i in range(0, 5):
            if y + i == h or arr[y + i][x] == 0:
                break
            cnt += 1
        if cnt >= 5:
            return True

        cnt = 0  # \
        for i in range(1, 5):
            if x - i < 0 or y - i < 0 or arr[y - i][x - i] == 0:
                break
            cnt += 1
        for i in range(0, 5):
            if x + i == w or y + i == h or arr[y + i][x + i] == 0:
                break
            cnt += 1
        if cnt >= 5:
            return True

        cnt = 0  # /
        for i in range(1, 5):
            if x - i < 0 or y + i == h or arr[y + i][x - i] == 0:
                break
            cnt += 1
        for i in range(0, 5):
            if x + i == w or y - i < 0 or arr[y - i][x + i] == 0:
                break
            cnt += 1
        if cnt >= 5:
            return True

        return False

    def next_state(self, state, pos):
        #         sk = bytes(state)
        # if self.nextstate_cache.get((sk, pos)) is not None:
        #     return self.nextstate_cache.get(sk)
        state = state.copy()
        y = pos // self.size
        x = pos % self.size

        state[y, x, 2] = 0
        actor = state[0, 0, 3]
        if actor == 1:
            state[y, x, 0] = 1
            state[:, :, 3] = -1
            state[y]
            win = self.have_five(state[:, :, 0], (x, y))
            # win = False
            return state, win
        else:  # actor == -1
            state[y, x, 1] = 1
            state[:, :, 3] = 1
            win = self.have_five(state[:, :, 1], (x, y))
            # win = False
            return state, win

    def valid_positions(self, state):
        # return state[:, :, 2] == 1
        p = np.transpose(np.where(state[:, :, 2] == 1))
        ret = [y * self.size + x for y, x in p]
        # ret = []
        # for y, x in p:
        #     ret.append(y * self.size + x)
        return ret

    def state2key(self, state):
        return bytes(state)

    def key2state(self, key):
        size = self.size
        state = np.frombuffer(key, dtype=np.int8).reshape(size, size)
        return state

    def key2tensor(self):
        pass

    def state2tensor(self):
        pass

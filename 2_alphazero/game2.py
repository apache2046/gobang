import numpy as np
#from numba import jit

#@jit(nopython=True)
def have_five(arr, x, y):
    # x, y
    w = h = 15

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

class GoBang:
    def __init__(self, size=15):
        self.size = size
        self.nextstate_cache = {}

    def start_state(self):
        size = self.size

        # 3 panles, black 0, white 1, empty 2, whois next 3 (1 for balck, -1 for white)
        state = np.zeros((size, size, 5), dtype=np.int8)
        state[:, :, 4] = 1  # next is black
        return state
    # @jit(forceobj=True)
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

        state[:, :, 2] = state[:, :, 0]
        state[:, :, 3] = state[:, :, 1]
        actor = state[0, 0, 4]
        if actor == 1:
            state[y, x, 0] = 1
            state[:, :, 4] = 0
            win = have_five(state[:, :, 0], x, y)
        else:  # actor == 0
            state[y, x, 1] = 1
            state[:, :, 4] = 1
            win = have_five(state[:, :, 1], x, y)

        end = True if win or np.count_nonzero(state[:, :, 0] + state[:, :, 1] == 0) < 20 else False
        return state, end, int(win)

    def state2key(self, state):
        # print("GGG", state)
        # return bytes(state)
        return np.packbits(state[:, :, :2]).tobytes()

    # def key2state(self, key):
    #     size = self.size
    #     state = np.frombuffer(key, dtype=np.int8).reshape(size, size)
    #     return state

    def key2tensor(self):
        pass

    def state2tensor(self):
        pass

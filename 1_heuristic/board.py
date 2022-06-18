import numpy as np
import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import evaluate as eva


class BoardState:
    def __init__(self, size=15):
        self.setSize(size)

    def setSize(self, size):
        self.size = size
        self.position = np.zeros((size, size), dtype=np.int32)  # 棋子摆放，-1,0,1个取值
        self.score_v = np.zeros((size, 2), dtype=np.int32)  # 垂直| 方向,棋面评估
        self.score_h = np.zeros((size, 2), dtype=np.int32)  # 水平- 方向,棋面评估
        self.score_lu2rb = np.zeros((size * 2 - 1, 2), dtype=np.int32)  # 左上到右下\ 的对角线方向,棋面评估
        self.score_lb2ru = np.zeros((size * 2 - 1, 2), dtype=np.int32)  # 左下到右上/ 的对角线方向,棋面评估
        self.search_point = set()  # 可搜索的点
        self.place_history = []
        self.v_actor = [0, 0]

    def reset(self):
        self.position.fill(0)
        self.score_v.fill(0)
        self.score_h.fill(0)
        self.score_lu2rb.fill(0)
        self.score_lb2ru.fill(0)
        self.search_point.clear()
        self.place_history.clear()
        self.v_actor = [0, 0]

    def revert(steps):
        pass

    def update_search_point(self, nextpos: List[int], wide=False):
        x, y = nextpos
        self.search_point.discard((x, y))
        h, w = self.size, self.size
        if wide:
            low, up = -2, 3
        else:
            low, up = -1, 2
        # low, up = -1, 2
        for i in range(max(0, y + low), min(h, y + up)):
            for j in range(max(0, x + low), min(w, x + up)):
                if self.position[i][j] == 0:
                    self.search_point.add((j, i))

    def place_stone(self, actor: int, pos: List[int]):
        self.place_history.append([*pos, actor])
        x, y = pos
        self.position[y][x] = actor
        self.update_search_point(pos)
        # print("sp", search_point)
        v1, v2, v3, v4 = eva.evaluate_4dir_lines(self.position, x, y)

        old_v1 = self.score_v[x].tolist()
        old_v2 = self.score_h[y].tolist()
        old_v3 = self.score_lu2rb[self.size - 1 + x - y].tolist()
        old_v4 = self.score_lb2ru[x + y].tolist()

        self.score_v[x] = v1
        self.score_h[y] = v2
        self.score_lu2rb[self.size - 1 + x - y] = v3
        self.score_lb2ru[x + y] = v4

        for i in range(2):
            self.v_actor[i] = (
                self.score_v[:, i].sum()
                + self.score_h[:, i].sum()
                + self.score_lu2rb[:, i].sum()
                + self.score_lb2ru[:, i].sum()
            ).tolist()
        return old_v1, old_v2, old_v3, old_v4, v1, v2, v3, v4

    def cur_patterns(self):
        patterns = [[], []]

        def _cur_patterns(score_line):
            for v in score_line:
                for i in range(2):
                    if v[i] >= eva.StonePattern.FIVE.value:
                        patterns[i].append[eva.StonePattern.FIVE.value]

                    elif v[i] >= eva.StonePattern.FOUR.value:
                        patterns[i].append[eva.StonePattern.FOUR.value]

                    elif v[i] >= eva.StonePattern.BLOCKED_FOUR.value:
                        patterns[i].append[eva.StonePattern.BLOCKED_FOUR.value]

                    elif v[i] >= eva.StonePattern.THREE.value:
                        patterns[i].append[eva.StonePattern.THREE.value]

        _cur_patterns(self.score_v)
        _cur_patterns(self.score_h)
        _cur_patterns(self.score_lu2rb)
        _cur_patterns(self.score_lb2ru)
        return patterns

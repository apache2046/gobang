import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from back_end import serv
import numpy as np
import evaluate as eva

board_state = np.zeros((15, 15))
search_point = set()
board_score_v = np.zeros((2, 15), dtype=np.int32)  # 垂直| 方向,棋面评估
board_score_h = np.zeros((2, 15), dtype=np.int32)  # 水平- 方向,棋面评估
board_score_lu2rb = np.zeros((2, 15 * 2 - 9), dtype=np.int32)  # 左上到右下\ 的对角线方向,棋面评估
board_score_lb2ru = np.zeros((2, 15 * 2 - 9), dtype=np.int32)  # 左下到右上/ 的对角线方向,棋面评估


def board_size(size):
    global board_state
    global board_score_v
    global board_score_h
    global board_score_lu2rb
    global board_score_lb2ru

    print("in play board_size", size, id(board_state))
    board_state = np.zeros((size, size), dtype=np.int32)
    board_score_v = np.zeros((2, size), dtype=np.int32)
    board_score_h = np.zeros((2, size), dtype=np.int32)
    board_score_lu2rb = np.zeros((2, size * 2 - 9), dtype=np.int32)
    board_score_lb2ru = np.zeros((2, size * 2 - 9), dtype=np.int32)
    print("in play board_size2", board_state.shape, id(board_state))
    return True


def clearboard():
    print("in play clearboard1", board_state.shape, id(board_state))
    board_state.fill(0)
    board_score_v.fill(0)
    board_score_h.fill(0)
    board_score_lu2rb.fill(0)
    board_score_lb2ru.fill(0)
    search_point.clear()
    print("play clearboard2")
    return True


def play(actor, pos):
    print("in play play", actor, pos)
    x, y = pos

    board_state[y][x] = actor
    update_search_point(board_state, (x, y), search_point)
    return True


def genmove(actor):
    print("in play genmove", board_state.shape)
    if False:
        size = board_state.shape[0]
        time.sleep(2)
        while True:
            pos = np.random.randint(0, size, 2)
            if board_state[pos[1]][pos[0]] == 0:
                return [int(pos[0]), int(pos[1])]
    else:
        pos, v = alpha_beta_search(board_state, search_point, actor, 0, 0, 4)
        return pos


def update_search_point(board, nextpos, sp):
    x, y = nextpos
    sp.discard((x, y))
    h, w = board.shape
    for i in range(max(0, y - 2), min(h, y + 3)):
        for j in range(max(0, x - 2), min(w, x + 3)):
            if board[i][j] == 0:
                sp.add((j, i))
    print(sp)


def alpha_beta_search(board, search_point, actor, alpha, beta, maxlevel):
    for next in search_point:
        new_board = board.copy()
        x, y = next
        new_board[y][x] = actor
        new_sp = search_point.copy()
        update_search_point(new_board, next, new_sp)
        p, v = alpha_beta_search(board_state, search_point, -actor, 0, 0, maxlevel - 1)

    pass


serv(8080, board_size, clearboard, play, genmove)

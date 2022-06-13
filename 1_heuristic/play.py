import sys
import os
import time
import copy
from turtle import pos

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from back_end import serv
import numpy as np
import evaluate as eva

board_state = np.zeros((15, 15))
search_point = set()
board_score_v = np.zeros((15, 2), dtype=np.int32)  # 垂直| 方向,棋面评估
board_score_h = np.zeros((15, 2), dtype=np.int32)  # 水平- 方向,棋面评估
board_score_lu2rb = np.zeros((15 * 2 - 1, 2), dtype=np.int32)  # 左上到右下\ 的对角线方向,棋面评估
board_score_lb2ru = np.zeros((15 * 2 - 1, 2), dtype=np.int32)  # 左下到右上/ 的对角线方向,棋面评估


def board_size(size):
    global board_state
    global board_score_v
    global board_score_h
    global board_score_lu2rb
    global board_score_lb2ru

    print("in play board_size", size, id(board_state))
    board_state = np.zeros((size, size), dtype=np.int32)
    board_score_v = np.zeros((size, 2), dtype=np.int32)
    board_score_h = np.zeros((size, 2), dtype=np.int32)
    board_score_lu2rb = np.zeros((size * 2 - 1, 2), dtype=np.int32)
    board_score_lb2ru = np.zeros((size * 2 - 1, 2), dtype=np.int32)
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
    h, w = board_state.shape
    board_state[y][x] = actor
    update_search_point(board_state, (x, y), search_point)
    print("sp", search_point)
    v1, v2, v3, v4 = eva.evaluate_4dir_lines(board_state, x, y)
    board_score_v[x] = v1
    board_score_h[y] = v2
    board_score_lu2rb[w - 1 + x - y] = v3
    board_score_lb2ru[x + y] = v4

    v_actor = [0, 0]
    # print(type(board_score_v), board_score_v.shape)
    for i in range(2):
        v_actor[i] = (
            board_score_v[:, i].sum()
            + board_score_h[:, i].sum()
            + board_score_lu2rb[:, i].sum()
            + board_score_lb2ru[:, i].sum()
        )
    print("v_actor:", v_actor, len(search_point))
    return True


def genmove(actor):
    print("in genmove", actor, board_state.shape)
    if False:
        size = board_state.shape[0]
        time.sleep(2)
        while True:
            pos = np.random.randint(0, size, 2)
            if board_state[pos[1]][pos[0]] == 0:
                play(actor, pos)
                return [int(pos[0]), int(pos[1])]
    else:
        new_board_state = copy.deepcopy(board_state)
        if actor == -1:
            new_board_state *= -1
        alpha, beta, bestmov = alpha_beta_search(
            board_state,
            board_score_v,
            board_score_h,
            board_score_lu2rb,
            board_score_lb2ru,
            search_point,
            True,
            -1e9,
            1e9,
            4,
        )
        print("end genmove", alpha, beta, bestmov, search_point)
        play(actor, bestmov)
        return bestmov


def update_search_point(board, nextpos, sp):
    x, y = nextpos
    sp.discard((x, y))
    h, w = board.shape
    # for i in range(max(0, y - 2), min(h, y + 3)):
    #     for j in range(max(0, x - 2), min(w, x + 3)):
    for i in range(max(0, y - 1), min(h, y + 2)):
        for j in range(max(0, x - 1), min(w, x + 2)):
            if board[i][j] == 0:
                sp.add((j, i))
    # print("sp", sp)

call_cnt = 0
gtime = 0
def alpha_beta_search(
    board, board_score_v, board_score_h, board_score_lu2rb, board_score_lb2ru, search_point, ismax, alpha, beta, level
):
    global call_cnt
    call_cnt +=1
    # if call_cnt % 10 == 0:
    #     print(call_cnt)
    bestmove = None
    h, w = board.shape
    for next_pos in search_point:
        new_board = copy.deepcopy(board)
        new_board_score_v = copy.deepcopy(board_score_v)
        new_board_score_h = copy.deepcopy(board_score_h)
        new_board_score_lu2rb = copy.deepcopy(board_score_lu2rb)
        new_board_score_lb2ru = copy.deepcopy(board_score_lb2ru)

        x, y = next_pos
        new_board[y][x] = 1 if ismax else -1
        new_sp = search_point.copy()
        update_search_point(new_board, next_pos, new_sp)
        stime = time.time()
        v1, v2, v3, v4 = eva.evaluate_4dir_lines(new_board, x, y)
        # global gtime
        # gtime += time.time() - stime
        # print(gtime, time.ctime())
        new_board_score_v[x] = v1
        new_board_score_h[y] = v2
        new_board_score_lu2rb[w - 1 + x - y] = v3
        new_board_score_lb2ru[x + y] = v4

        v_actor = [0, 0]
        if level == 1:
            for i in range(2):
                v_actor[i] = (
                    board_score_v[:, i].sum()
                    + board_score_h[:, i].sum()
                    + board_score_lu2rb[:, i].sum()
                    + board_score_lb2ru[:, i].sum()
                )
            v = v_actor[0] - v_actor[1]
            if ismax:
                if v > alpha:
                    alpha = v
                    bestmove = next_pos
            else:
                if v < beta:
                    beta = v
                    bestmove = next_pos
        else:
            c_alpha, c_beta, _ = alpha_beta_search(
                new_board,
                new_board_score_v,
                new_board_score_h,
                new_board_score_lu2rb,
                new_board_score_lb2ru,
                new_sp,
                not ismax,
                alpha,
                beta,
                level - 1,
            )
            if ismax:
                # if level == 3:
                #     print(alpha, c_alpha, c_beta)
                if c_beta > alpha:
                    bestmove = next_pos
                alpha = max(alpha, c_beta)
            else:
                if c_alpha < beta:
                    bestmove = next_pos
                beta = min(beta, c_alpha)

            # pruning
            if alpha > beta:
                print("pruned..")
                break
    # print("end ab", level, alpha, beta, bestmove)
    return alpha, beta, bestmove


serv(8080, board_size, clearboard, play, genmove)
# board_size(15)
# play(1, [8,8])
# genmove(-1)

import sys
import os
import time
import copy
import multiprocessing as mp

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
board_history = []


def board_size(size):
    global board_state
    global board_score_v
    global board_score_h
    global board_score_lu2rb
    global board_score_lb2ru

    print("in board_size", size, id(board_state))
    board_state = np.zeros((size, size), dtype=np.int32)
    board_score_v = np.zeros((size, 2), dtype=np.int32)
    board_score_h = np.zeros((size, 2), dtype=np.int32)
    board_score_lu2rb = np.zeros((size * 2 - 1, 2), dtype=np.int32)
    board_score_lb2ru = np.zeros((size * 2 - 1, 2), dtype=np.int32)
    print("in board_size2", board_state.shape, id(board_state))
    return True


def clearboard():
    print("in clearboard1", board_state.shape, id(board_state))
    board_state.fill(0)
    board_score_v.fill(0)
    board_score_h.fill(0)
    board_score_lu2rb.fill(0)
    board_score_lb2ru.fill(0)
    search_point.clear()
    board_history.clear()
    print("in clearboard2")
    return True


def play(actor, pos):
    print("in play", actor, pos)
    board_history.append([*pos, actor])
    x, y = pos
    h, w = board_state.shape
    board_state[y][x] = actor
    update_search_point(board_state, (x, y), search_point)
    # print("sp", search_point)
    v1, v2, v3, v4 = eva.evaluate_4dir_lines(board_state * -1, x, y)
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
    return mp_genmove(actor)
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
            new_board_state,
            board_score_v,
            board_score_h,
            board_score_lu2rb,
            board_score_lb2ru,
            search_point,
            True,
            -1e10,
            1e10,
            6,
        )
        # print('cs', alpha, beta)
        # if alpha < eva.Pattern.FOUR.value:
        #     alpha, beta, bestmov = alpha_beta_search(
        #         new_board_state,
        #         board_score_v,
        #         board_score_h,
        #         board_score_lu2rb,
        #         board_score_lb2ru,
        #         search_point,
        #         True,
        #         -1e10,
        #         1e10,
        #         4,
        #     )
        print("##end genmove", alpha, beta, bestmov, len(search_point))
        play(actor, bestmov[0][:2])
        print(board_history)
        print(board_state, "\n")
        return bestmov[0][:2]


if __name__ == "__main__":
    mp.set_start_method("spawn")
    mp_pool = mp.Pool(mp.cpu_count() // 2)
    # mp_pool = mp.Pool(1)


def mp_genmove(actor):
    new_board_state = copy.deepcopy(board_state)
    if actor == -1:
        new_board_state *= -1
    params = []
    for s in search_point:
        params.append(
            (
                new_board_state,
                board_score_v,
                board_score_h,
                board_score_lu2rb,
                board_score_lb2ru,
                search_point,
                True,
                -1e10,
                1e10,
                6,
                s,
            )
        )
    result = mp_pool.starmap(alpha_beta_search, params)
    # debug = [x for x in result if x[0] >= 1000000]
    result.sort(key=lambda x: x[0], reverse=True)
    debug = [x[0] for x in result]
    alpha, beta, bestmov = result[0]
    print("##end genmove", alpha, beta, bestmov, search_point, debug)
    play(actor, bestmov[0][:2])
    print(board_history, "\n")
    # print(board_state, "\n")
    return bestmov[0][:2]


def update_search_point(board, nextpos, sp, wide=False):
    x, y = nextpos
    sp.discard((x, y))
    h, w = board.shape
    if wide:
        low, up = -2, 3
    else:
        low, up = -1, 2
    # low, up = -1, 2
    for i in range(max(0, y + low), min(h, y + up)):
        for j in range(max(0, x + low), min(w, x + up)):
            if board[i][j] == 0:
                sp.add((j, i))
    # print("sp", sp)


call_cnt = 0
gtime = 0

np.random.seed(1234)
zobrist = np.random.randint(1e8, 1e14, (15, 15, 3))
abs_cache = {}


def alpha_beta_search(
    board,
    board_score_v,
    board_score_h,
    board_score_lu2rb,
    board_score_lb2ru,
    search_point,
    ismax,
    alpha,
    beta,
    level,
    first_point=None,
):
    # cached = abs_cache.get(np.take_along_axis(zobrist, np.expand_dims(board + 1, 2), 2).sum())
    # cached = abs_cache.get("".join([str(x) for x in (board + 1).flatten().tolist()]))
    cache_code = None
    if first_point is None:
        cache_code = "".join([str(x) for x in (board + 1).flatten().tolist()]) + str(level)
        # cache_code = np.bitwise_xor.reduce(np.take_along_axis(zobrist, np.expand_dims(board + 1, 2), 2).flatten())
        cached = abs_cache.get(cache_code)
        if cached is not None:
            alpha, beta, bestmove = cached
            return alpha, beta, bestmove
    # global call_cnt
    # call_cnt += 1
    # if call_cnt % 100 == 0:
    #     print(call_cnt)
    bestmove = None
    h, w = board.shape
    search_q = []
    if first_point is not None:
        search = [first_point]
    else:
        search = search_point
    # print("search",len(search), search)
    for next_pos in search:
        new_board = copy.deepcopy(board)
        new_board_score_v = copy.deepcopy(board_score_v)
        new_board_score_h = copy.deepcopy(board_score_h)
        new_board_score_lu2rb = copy.deepcopy(board_score_lu2rb)
        new_board_score_lb2ru = copy.deepcopy(board_score_lb2ru)

        if bestmove is None:
            bestmove = [[*next_pos, ismax]]
        x, y = next_pos
        new_board[y][x] = 1 if ismax else -1
        new_sp = copy.deepcopy(search_point)
        update_search_point(new_board, next_pos, new_sp, level >= 5)
        v1, v2, v3, v4 = eva.evaluate_4dir_lines(new_board, x, y)
        old_v1 = new_board_score_v[x]
        old_v2 = new_board_score_h[y]
        old_v3 = new_board_score_lu2rb[w - 1 + x - y]
        old_v4 = new_board_score_lb2ru[x + y]
        new_board_score_v[x] = v1
        new_board_score_h[y] = v2
        new_board_score_lu2rb[w - 1 + x - y] = v3
        new_board_score_lb2ru[x + y] = v4

        v_actor = [0, 0]
        for i in range(2):
            v_actor[i] = (
                new_board_score_v[:, i].sum()
                + new_board_score_h[:, i].sum()
                + new_board_score_lu2rb[:, i].sum()
                + new_board_score_lb2ru[:, i].sum()
            )
        v = v_actor[0] - v_actor[1]

        # 无论当前level如何，若出现 活四、连五 棋形，判定为叶子结点，不需要继续深入了
        # 当前动作若消除了对方的活三、冲四，也是叶子结点
        if ismax:
            if (
                old_v1[1] - v1[1] >= eva.Pattern.BLOCKED_FOUR.value
                or old_v2[1] - v2[1] >= eva.Pattern.BLOCKED_FOUR.value
                or old_v3[1] - v3[1] >= eva.Pattern.BLOCKED_FOUR.value
                or old_v4[1] - v4[1] >= eva.Pattern.BLOCKED_FOUR.value
            ):
                alpha = v
                bestmove = [[*next_pos, ismax, v_actor]]
                level = 0
                break

            if (
                v_actor[0] >= eva.Pattern.FIVE.value
                or v_actor[0] >= eva.Pattern.FOUR.value
                and v_actor[1] < eva.Pattern.BLOCKED_FOUR.value
            ):
                alpha = 1000000 + level
                bestmove = [[*next_pos, ismax, v_actor]]
                level = 0
                break

            if v_actor[1] >= eva.Pattern.THREE.value:
                continue
            #     alpha = -1000000 - level
            #     bestmove = [[*next_pos, ismax, v_actor]]
            #     level = 0
            #     break
        else:
            if (
                old_v1[0] - v1[0] >= eva.Pattern.BLOCKED_FOUR.value
                or old_v2[0] - v2[0] >= eva.Pattern.BLOCKED_FOUR.value
                or old_v3[0] - v3[0] >= eva.Pattern.BLOCKED_FOUR.value
                or old_v4[0] - v4[0] >= eva.Pattern.BLOCKED_FOUR.value
            ):
                beta = v
                bestmove = [[*next_pos, ismax, v_actor]]
                level = 0
                break

            if (
                v_actor[1] >= eva.Pattern.FIVE.value
                or v_actor[1] >= eva.Pattern.FOUR.value
                and v_actor[0] < eva.Pattern.BLOCKED_FOUR.value
            ):
                beta = -1000000 - level
                bestmove = [[*next_pos, ismax, v_actor]]
                level = 0
                break

            if v_actor[0] >= eva.Pattern.THREE.value:
                continue
            #     beta = 1000000 + level
            #     bestmove = [[*next_pos, ismax, v_actor]]
            #     level = 0
            #     break

        # if v_actor[0] > eva.Pattern.FOUR.value:
        #     break

        if level == 1:
            # print(next_pos, v_actor)
            # bestmove = [[*next_pos, ismax]]
            if ismax:
                if v > alpha:
                    alpha = v
                    bestmove = [[*next_pos, ismax]]
            else:
                if v < beta:
                    beta = v
                    bestmove = [[*next_pos, ismax]]
        else:
            # len(search_q) == 0 or
            # if len(search_q) == 0 or level >=3 or v_actor[0] > eva.Pattern.BLOCKED_FOUR.value or v_actor[1] > eva.Pattern.BLOCKED_FOUR.value :
            if True:
                search_q.append(
                    [
                        v,
                        new_board,
                        new_board_score_v,
                        new_board_score_h,
                        new_board_score_lu2rb,
                        new_board_score_lb2ru,
                        new_sp,
                        next_pos,
                    ]
                )
    if level > 1:
        if ismax:
            search_q.sort(key=lambda x: x[0], reverse=True)
        else:
            search_q.sort(key=lambda x: x[0], reverse=False)
        for (
            _,
            nnew_board,
            nnew_board_score_v,
            nnew_board_score_h,
            nnew_board_score_lu2rb,
            nnew_board_score_lb2ru,
            nnew_sp,
            nnext_pos,
        ) in search_q:
            c_alpha, c_beta, c_bestmove = alpha_beta_search(
                nnew_board,
                nnew_board_score_v,
                nnew_board_score_h,
                nnew_board_score_lu2rb,
                nnew_board_score_lb2ru,
                nnew_sp,
                not ismax,
                alpha,
                beta,
                level - 1,
            )
            if ismax:
                # if level == 3:
                #     print(alpha, c_alpha, c_beta)
                if c_beta > alpha:
                    c_bestmove.insert(0, [*nnext_pos, ismax])
                    bestmove = c_bestmove
                alpha = max(alpha, c_beta)
            else:
                if c_alpha < beta:
                    c_bestmove.insert(0, [*nnext_pos, ismax])
                    bestmove = c_bestmove
                beta = min(beta, c_alpha)
                # print(level, alpha, beta, c_alpha)

            # pruning
            if alpha >= beta:
                # if alpha > beta:
                #     print("pruned..", alpha, beta)
                break

    # cache_code = np.bitwise_xor.reduce(np.take_along_axis(zobrist, np.expand_dims(board + 1, 2), 2).flatten())
    if cache_code is not None:
        abs_cache[cache_code] = (alpha, beta, bestmove)

    return alpha, beta, bestmove


if __name__ == "__main__":
    serv(8080, board_size, clearboard, play, genmove)
    # time.sleep(1)
    # board_size(15)
    # clearboard()
    # actions = [
    #     [7, 7, 1],
    #     [8, 8, -1],
    #     [7, 9, 1],
    #     [6, 8, -1],
    #     [7, 8, 1],
    #     [7, 10, -1],
    #     [8, 7, 1],
    #     [6, 7, -1],
    #     [6, 9, 1],
    #     [9, 6, -1],
    #     [9, 9, 1],
    #     [8, 9, -1],
    #     [5, 10, 1],
    # ]
    # for x, y, a in actions:
    #     play(a, [x, y])
    # genmove(-1)

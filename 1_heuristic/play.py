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
from board import BoardState

if __name__ == "__main__":
    board_state = BoardState()


def board_size(size):
    print("in board_size", size)
    board_state.setSize(size)
    print("in board_size2", board_state.position.shape)
    return True


def clearboard():
    print("in clearboard1")
    board_state.reset()
    print("in clearboard2")
    return True


def play(actor, pos):
    print("in play", actor, pos)
    board_state.place_stone(actor, pos)
    print("v_actor:", board_state.v_actor, len(board_state.search_point))
    return True


def genmove(actor):
    print("in genmove", actor)
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
        alpha, beta, bestmov = alpha_beta_search(
            actor,
            board_state,
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
        print("##end genmove", alpha, beta, bestmov, len(board_state.search_point))
        board_state.place_stone(actor, bestmov[0][:2])
        print(board_state.place_history)
        print(board_state, "\n")
        return bestmov[0][:2]


if __name__ == "__main__":
    mp.set_start_method("spawn")
    mp_pool = mp.Pool(mp.cpu_count() // 2)
    # mp_pool = mp.Pool(1)
    mp_abs_cache = mp.Manager().dict()


def mp_genmove(actor):
    print("in mp_genmove", actor)
    params = []
    for s in board_state.search_point:
        params.append(
            (
                actor,
                board_state,
                True,
                -1e10,
                1e10,
                6,
                s,
                mp_abs_cache,
            )
        )
    result = mp_pool.starmap(alpha_beta_search, params)
    # debug = [x for x in result if x[0] >= 1000000]
    result.sort(key=lambda x: x[0], reverse=True)
    debug = [x[0] for x in result]
    alpha, beta, bestmov = result[0]
    print("##end genmove", alpha, beta, bestmov, board_state.search_point, debug)
    play(actor, bestmov[0][:2])
    print(board_state.place_history, "\n")
    # print(board_state, "\n")
    return bestmov[0][:2]


call_cnt = 0
gtime = 0

np.random.seed(1234)
zobrist = np.random.randint(1e8, 1e14, (15, 15, 3))
local_abs_cache = {}


def swap(data):
    for item in data:
        item[0], item[1] = item[1], item[0]


def alpha_beta_search(
    actor,
    boardState: BoardState,
    ismax,
    alpha,
    beta,
    level,
    first_point=None,
    mp_abs_cache=None
):
    # salpha = -1e10
    # sbeta = 1e10
    # cached = abs_cache.get(np.take_along_axis(zobrist, np.expand_dims(board + 1, 2), 2).sum())
    # cached = abs_cache.get("".join([str(x) for x in (board + 1).flatten().tolist()]))
    cache_code = None
    abs_cache = mp_abs_cache if mp_abs_cache is not None else local_abs_cache
    if first_point is None:
        cache_code = "".join([str(x) for x in (boardState.position + 1).flatten().tolist()]) + str(level)
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
    deeper_search_q = []
    if first_point is not None:
        search = [first_point]
    else:
        search = boardState.search_point
    # print("search",len(search), search)
    for idx, next_pos in enumerate(search):
        if level == 6:
            print(idx, "/", len(search))
        if bestmove is None:
            bestmove = [[*next_pos, ismax]]

        new_boardState = copy.deepcopy(boardState)

        stone = actor if ismax else -actor
        old_v1, old_v2, old_v3, old_v4, v1, v2, v3, v4 = new_boardState.place_stone(stone, next_pos)
        new_boardState.update_search_point(next_pos, level >= 5)
        v_actor = new_boardState.v_actor

        if actor == -1:
            swap([old_v1, old_v2, old_v3, old_v4, v1, v2, v3, v4, v_actor])

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

        # if ismax:
        #     if v > salpha:
        #         salpha = v
        #         sbestmove = [[*next_pos, ismax]]
        # else:
        #     if v < sbeta:
        #         sbeta = v
        #         sbestmove = [[*next_pos, ismax]]

        if level == 1:

            if ismax:
                if v > alpha:
                    alpha = v
                    bestmove = [[*next_pos, ismax]]
            else:
                if v < beta:
                    beta = v
                    bestmove = [[*next_pos, ismax]]
        else:
            # len(deeper_search_q) == 0 or
            # if level >= 3 or v_actor[0] > eva.Pattern.THREE.value or v_actor[1] > eva.Pattern.THREE.value :
            if True:
                deeper_search_q.append([v, new_boardState, next_pos])
    # if len(deeper_search_q) == 0:
    #     if ismax:
    #         alpha = salpha
    #     else:
    #         beta = sbeta
    #     bestmove = [[*next_pos, ismax]]

    if level > 1: # Start deeper search
        if ismax:
            deeper_search_q.sort(key=lambda x: x[0], reverse=True)
        else:
            deeper_search_q.sort(key=lambda x: x[0], reverse=False)

        for _, nnew_boardState, nnext_pos in deeper_search_q:
            c_alpha, c_beta, c_bestmove = alpha_beta_search(
                actor,
                nnew_boardState,
                not ismax,
                alpha,
                beta,
                level - 1,
            )
            if level == 6:
                print(1, "//", len(deeper_search_q))
            if ismax:
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
    # actions = [[7, 7, 1], [7, 8, -1], [8, 8, 1], [9, 9, -1], [9, 7, 1], [6, 7, -1], [8, 9, 1], [8, 7, -1], [10, 6, 1]]

    # for x, y, a in actions:
    #     play(a, [x, y])
    # genmove(-1)

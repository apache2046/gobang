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
    stime = time.time()
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
                8,
                [],
                s,
                mp_abs_cache,
            )
        )
    result = mp_pool.starmap(alpha_beta_search, params)
    # debug = [x for x in result if x[0] >= 1000000]
    result.sort(key=lambda x: x[0], reverse=True)
    debug = [x[0] for x in result]
    alpha, beta, bestmov = result[0]
    print("##end genmove", f'{time.time()-stime:.1f}S', alpha, beta, bestmov, board_state.search_point, debug)
    play(actor, bestmov[0][:2])
    print(board_state.place_history, "\n")
    # print(board_state, "\n")
    return bestmov[0][:2]


call_cnt = 0
gtime = 0

np.random.seed(1234)
zobrist = np.random.randint(1e8, 1e14, (15, 15, 3))
local_abs_cache = {}

# fmt: off
def swap(data):
    for item in data:
        item[0], item[1] = item[1], item[0]

def will_winnext(data):
    for item in data:
        if item >= eva.Pattern.THREE:  # also include BLOCKED_FOUR
            return True
    else:
        return False

def is_win(me, rival):
    return (
        me >= eva.Pattern.FIVE.value
        or me >= eva.Pattern.FOUR.value and rival < eva.Pattern.BLOCKED_FOUR.value
    )

def is_lose(me, rival):
    return (
        me >= eva.Pattern.FIVE.value
        or me >= eva.Pattern.FOUR.value and rival < eva.Pattern.BLOCKED_FOUR.value
    )

def is_badmove(me, rival):
    # 对方有冲四
    # 对方有活三但我方无冲四
    return (
        rival >= eva.Pattern.BLOCKED_FOUR.value
        or rival >= eva.Pattern.THREE.value and me < eva.Pattern.BLOCKED_FOUR.value
    )

def is_1point_danger_removed(
        rival_old_v1, rival_old_v2, rival_old_v3, rival_old_v4,
        rival_v1, rival_v2, rival_v3, rival_v4):

    return (
        rival_old_v1 - rival_v1 >= eva.Pattern.BLOCKED_FOUR.value
        or rival_old_v2 - rival_v2 >= eva.Pattern.BLOCKED_FOUR.value
        or rival_old_v3 - rival_v3 >= eva.Pattern.BLOCKED_FOUR.value
        or rival_old_v4 - rival_v4 >= eva.Pattern.BLOCKED_FOUR.value
    )
# fmt: on


def alpha_beta_search(actor, boardState: BoardState, ismax, alpha, beta, level, trace, first_point=None, mp_abs_cache=None):
    # cached = abs_cache.get(np.take_along_axis(zobrist, np.expand_dims(board + 1, 2), 2).sum())
    # cached = abs_cache.get("".join([str(x) for x in (board + 1).flatten().tolist()]))
    cache_code = None
    # abs_cache = mp_abs_cache if mp_abs_cache is not None else local_abs_cache
    # if first_point is None:
    #     cache_code = "".join([str(x) for x in (boardState.position + 1).flatten().tolist()]) + str(level)
    #     # cache_code = np.bitwise_xor.reduce(np.take_along_axis(zobrist, np.expand_dims(board + 1, 2), 2).flatten())
    #     cached = abs_cache.get(cache_code)
    #     if cached is not None:
    #         alpha, beta, bestmove = cached
    #         return alpha, beta, bestmove
    # global call_cnt
    # call_cnt += 1
    # if call_cnt % 100 == 0:
    #     print(call_cnt)
    bestmove = []
    deeper_search_q = []
    early_prune = False
    if first_point is not None:
        search = [first_point]
    else:
        search = boardState.search_point
    # print("search",len(search), search)
    for idx, next_pos in enumerate(search):
        # if level == 6:
        #     print(idx, "/", len(search))

        new_boardState = copy.deepcopy(boardState)

        stone = actor if ismax else -actor
        old_v1, old_v2, old_v3, old_v4, v1, v2, v3, v4 = new_boardState.place_stone(stone, next_pos)
        new_boardState.update_search_point(next_pos, level >= 7)
        v_actor = new_boardState.v_actor

        if actor == -1:
            swap([old_v1, old_v2, old_v3, old_v4, v1, v2, v3, v4, v_actor])

        v = v_actor[0] - v_actor[1]
        next_pos_isleaf = False
        # 无论当前level如何，若出现 活四、连五 棋形，判定为叶子结点，不需要继续深入了
        # 当前动作若消除了对方的活三、冲四，也是叶子结点
        if ismax:
            if is_badmove(v_actor[0], v_actor[1]):
                v = -1000000 + v
                next_pos_isleaf = True

            # if is_1point_danger_removed(old_v1[1], old_v2[1], old_v3[1], old_v4[1], v1[1], v2[1], v3[1], v4[1]):
            #     alpha = v
            #     bestmove = [[*next_pos, ismax, v_actor]]
            #     early_prune = True
            #     break

            if is_win(v_actor[0], v_actor[1]):
                alpha = 1000000 + level * 100000 + v_actor[0]
                bestmove = [[*next_pos, ismax, v_actor]]
                early_prune = True
                break

        else:
            if is_badmove(v_actor[1], v_actor[0]):
                v = 2000000 + v
                next_pos_isleaf = True

            # if is_1point_danger_removed(old_v1[0], old_v2[0], old_v3[0], old_v4[0], v1[0], v2[0], v3[0], v4[0]):
            #     beta = v
            #     bestmove = [[*next_pos, ismax, v_actor]]
            #     early_prune = True
            #     break

            if is_win(v_actor[1], v_actor[0]):
                beta = -1000000 - level
                bestmove = [[*next_pos, ismax, v_actor]]
                early_prune = True
                break


        if level == 1 or (level < 5 and (v_actor[0] < eva.Pattern.THREE.value or v_actor[1] < eva.Pattern.THREE.value)):
            next_pos_isleaf = True

        deeper_search_q.append([v, new_boardState, next_pos, next_pos_isleaf])

    if early_prune:
        return alpha, beta, bestmove

    if ismax:
        deeper_search_q.sort(key=lambda x: x[0], reverse=True)
    else:
        deeper_search_q.sort(key=lambda x: x[0], reverse=False)

    for nv, nnew_boardState, nnext_pos, isleaf in deeper_search_q:
        if isleaf:
            c_alpha = c_beta = nv
            c_bestmove = []
        else:
            ntrace = trace.copy()
            ntrace.append(nnext_pos)
            c_alpha, c_beta, c_bestmove = alpha_beta_search(
                actor,
                nnew_boardState,
                not ismax,
                alpha,
                beta,
                level - 1,
                ntrace
            )
        # if level == 6:
        #     print(1, "//", len(deeper_search_q))
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
    # actions = [[7, 7, 1], [8, 8, -1], [7, 9, 1], [8, 7, -1], [6, 8, 1], [8, 9, -1], [8, 6, 1], [9, 5, -1], [6, 7, 1], [9, 8, -1], [5, 7, 1], [4, 6, -1], [4, 7, 1], [3, 7, -1], [8, 10, 1], [9, 11, -1], [6, 9, 1], [6, 6, -1], [5, 9, 1], [4, 10, -1], [4, 9, 1], [3, 9, -1], [5, 8, 1]]
    # actions = [[7, 7, 1], [8, 8, -1], [7, 9, 1], [7, 10, -1], [8, 10, 1], [6, 8, -1], [7, 8, 1], [6, 9, -1], [5, 8, 1], [6, 11, -1], [6, 7, 1]]
    # actions = [[7, 7, 1], [8, 8, -1], [7, 9, 1], [7, 10, -1], [8, 10, 1], [6, 8, -1], [7, 8, 1], [6, 9, -1], [5, 8, 1], [6, 7, -1], [6, 6, 1], [6, 11, -1], [6, 10, 1], [7, 5, -1], [7, 11, 1], [8, 12, -1], [10, 8, 1], [9, 9, -1], [9, 12, 1], [5, 12, -1], [8, 9, 1], [10, 10, -1], [9, 11, 1], [10, 12, -1], [10, 11, 1], [11, 11, -1], [12, 12, 1], [4, 13, -1], [3, 14, 1], [7, 12, -1], [8, 13, 1], [11, 10, -1], [9, 10, 1], [11, 12, -1], [9, 13, 1], [9, 14, -1], [11, 13, 1], [10, 13, -1], [12, 11, 1], [12, 10, -1], [4, 12, 1], [5, 13, -1], [5, 11, 1], [3, 13, -1], [2, 13, 1], [11, 8, -1], [11, 9, 1], [4, 9, -1], [5, 10, 1], [5, 9, -1], [3, 9, 1], [4, 10, -1], [4, 8, 1], [2, 10, -1], [13, 9, 1], [3, 10, -1], [6, 13, 1]]

    # for x, y, a in actions:
    #     play(a, [x, y])
    # genmove(-1)

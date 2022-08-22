import sys
import os
from board import BoardState

from mctsvl2 import MCTS
from game3 import GoBang
from model9 import Policy_Value
import torch
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from back_end import serv

np.set_printoptions(precision=5, linewidth=1500)


if __name__ == "__main__":
    board_state = BoardState()
    nnet = Policy_Value().to("cuda:0")
    nnet.load_state_dict(torch.load("/home/apache/ray_run/models/0.pt"))
    nnet = nnet.to("cuda:0").half()
    nnet.eval()
    game = GoBang()
    mcts = None
    state = None


def board_size(size):
    print("in board_size", size)
    board_state.setSize(size)
    print("in board_size2", board_state.position.shape)
    return True


def clearboard():
    print("in clearboard1")
    board_state.reset()
    global mcts
    global state
    # mcts = MCTS(game, c_puct=0.1, selfplay=False)
    mcts = MCTS(
        game,
        c_puct=0.2,
        dirichlet_alpha=0.05,
        dirichlet_weight=0.25,
        reward_scale=4.0,
        selfplay=False,
        v_discount=1.0,
        vloss=1.0
    )
    state = game.start_state()
    print("in clearboard2")
    return True


def play(actor, pos):
    print("in play", actor, pos)
    board_state.place_stone(actor, pos)
    cur_patterns = board_state.cur_patterns()

    x, y = pos
    global state
    state, _, _ = game.next_state(state, y * 15 + x)

    print("v_actor:", board_state.v_actor, len(board_state.search_point))
    return [actor, pos, board_state.v_actor, cur_patterns]


def genmove(actor):
    print("in genmove", actor)
    ## empty board
    if len(board_state.place_history) == 0:
        p = board_state.size // 2
        p = [p, p]
        return play(actor, p)

    # global mcts
    global state
    mcts = MCTS(
        game,
        c_puct=1.0,
        dirichlet_alpha=0.05,
        dirichlet_weight=0.25,
        reward_scale=4.0,
        selfplay=False,
        v_discount=1.0,
        vloss=1.0
    )
    # print(state)
    # mcts = MCTS(game, rdb, c_puct=5, dirichlet_alpha=0.05, dirichlet_weight=0.25, reward_scale=4.0)
    if True:
        stones = (state[:, :, 0] + state[:, :, 1] > 0).sum()
        # mcts = MCTS(game, c_puct=2, selfplay=False)
        
        infer_data = []
        need_infer = []
        wait_infer = []
        for search_cnt in range(1000):

            for _ in range(128):
                try:
                    g = mcts.search(state)
                    state_infer = next(g)
                    # print("G10", state_infer)
                    if state_infer is None:
                        wait_infer.append(g)
                    else:
                        state_infer = np.ascontiguousarray(state_infer)
                        infer_data.append(torch.tensor(state_infer, dtype=torch.float32))
                        need_infer.append(g)
                except StopIteration:
                    continue
            if len(need_infer) == 0:
                continue
            data = torch.stack(infer_data).permute(0,3,1,2).to('cuda:0').half()
            print(data.shape)
            with torch.no_grad():
                ps, vs = nnet(data)
            ps = ps.to('cpu').numpy()
            vs = vs.to('cpu').numpy()
            new_infer_data = []
            new_need_infer = []
            new_wait_infer = []
            for i in range(len(need_infer)):
                try:
                    need_infer[i].send((ps[i], vs[i][0]))
                except StopIteration:
                    continue

            for i in range(len(wait_infer)):
                try:
                    g = wait_infer[i]
                    # print(ps[i], vs[i])
                    state_infer = g.send(None)
                    if state_infer is None:
                        new_wait_infer.append(g)
                    else:
                        state_infer = np.ascontiguousarray(state_infer)
                        new_infer_data.append(torch.tensor(state_infer, dtype=torch.float32))
                        new_need_infer.append(g)
                except StopIteration:
                    continue
            infer_data = new_infer_data
            need_infer = new_need_infer
            wait_infer = new_wait_infer

            # if search_cnt % 200 == 0 and search_cnt >= 200:
            #     _, pi = mcts.pi(state)
            #     if pi.max() > 0.95:
            #         print('cut!!', search_cnt)
            #         break

            
        nsa, pi = mcts.pi(state)
        print(nsa, '\n', pi.reshape(15,15), pi.max(), mcts.get_depth(state))
        # action = np.random.choice(len(pi), p=pi)
        action = int(np.argmax(pi).tolist())
        x = action % 15
        y = action // 15
    else:
        pi, v = nnet.infer(state)
        # a = game.valid_positions(state)
        # pi[a] *= 1000
        # print(state, '\n', pi.reshape(15,15), v)
        print(pi.reshape(15, 15), v)
        # print('### v', v)
        action = int(np.argmax(pi))
        x = action % 15
        y = action // 15
        print(type(x), type(y), type(actor))
    result = play(actor, (x, y))
    pi, v = nnet.infer(state)
    print(pi.reshape(15, 15), v)
    # r_state = state.copy()
    # if r_state[0, 0, -1] == 0:
    #     r_state[:, :, -1] = 1
    # else:
    #     r_state[:, :, -1] = 0
    # pi, v = nnet.infer(r_state)
    # print(pi.reshape(15, 15), v)

    return result


if __name__ == "__main__":
    serv(8080, board_size, clearboard, play, genmove)

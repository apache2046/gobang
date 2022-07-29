import sys
import os
from board import BoardState

from mcts5 import MCTS
from game2 import GoBang
from model4 import Policy_Value
import torch
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from back_end import serv
np.set_printoptions(precision=1, linewidth=1500)


if __name__ == "__main__":
    board_state = BoardState()
    nnet = Policy_Value().to("cuda:0")
    nnet.load_state_dict(torch.load("/home/apache/ray_run/models.10/28.pt"))
    nnet = nnet.to("cuda:0")
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
    mcts = MCTS(game, c_puct=0.1, selfplay=False)
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

    global mcts
    global state
    # print(state)

    if False:
        mcts = MCTS(game, c_puct=5, selfplay=False)
        for i in range(10000):
           g = mcts.search(state)
           #print("F1", i)
           try:
               state_infer = next(g)
               while True:
                   #print("F2", i)
                   p, v = nnet.infer(state_infer)
                   state_infer = g.send((p, v))
           except StopIteration:
               continue
        pi = mcts.pi(state)
        print(pi)
        #action = np.random.choice(len(pi), p=pi)
        action = int(np.argmax(pi).tolist())
        x = action % 15
        y = action // 15
        return play(actor, (x, y))
    else:
        pi, v = nnet.infer(state)
        #a = game.valid_positions(state)
        # pi[a] *= 1000
        print(state, '\n', pi.reshape(15,15), v)
        #print('### v', v)
        action = int(np.argmax(pi))
        x = action % 15
        y = action // 15
        print(type(x), type(y), type(actor))
        return play(actor, (x, y))

if __name__ == "__main__":
    serv(8080, board_size, clearboard, play, genmove)

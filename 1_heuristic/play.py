import sys
import os
import time
sys.path.append(os.path.dirname(__file__) + "/../")

from back_end import serv
import numpy as np

board_state = np.zeros((15,15))
def board_size(size):
    global board_state
    print("in play board_size", size, id(board_state))
    board_state = np.zeros((size,size), dtype=np.int32)
    print("in play board_size2", board_state.shape, id(board_state))
    return True

def clearboard():
    global board_state
    print("in play clearboard1", board_state.shape, id(board_state))
    board_state = np.zeros_like(board_state, dtype=np.int32)
    print("play clearboard2")
    return True

def play(actor, pos):
    print('in play play', actor, pos)
    x, y = pos
    
    board_state[y][x] = actor
    return True

def genmove(actor):
    print('in play genmove', board_state.shape)
    size = board_state.shape[0]
    time.sleep(2)
    while True:
        pos = np.random.randint(0, size, 2)
        if board_state[pos[1]][pos[0]] == 0:
            return [int(pos[0]), int(pos[1])]


serv(8080, board_size, clearboard, play, genmove)
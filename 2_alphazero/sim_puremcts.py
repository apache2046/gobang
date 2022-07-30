from multiprocessing.connection import Client
import numpy as np
from mcts6 import MCTS
from game2 import GoBang
import time
from model4 import Policy_Value
import torch
import ray
import random
from collections import deque
import os
import socket
import traceback
from io import BytesIO
from uploadkifu import Uploader
import json

print(socket.gethostname(), os.getcwd())
ray.init(address="auto", _node_ip_address="192.168.5.7")
# dbclient = pymongo.MongoClient("mongodb://root:mongomprc12@localhost:27017/")
# gobang_db = dbclient["gobang"]
# gobang_col1 = gobang_db["kifu2"]


def executeEpisode(game, epid, mcts_playout):
    state = game.start_state()
    samples = []
    cnt = 0
    board_record = np.zeros((game.size, game.size), dtype=np.uint8)
    stime = time.time()
    tau = 0.8
    while True:
        mcts = MCTS(game, c_puct=5, dirichlet_alpha=0.05, dirichlet_weight=0.25, reward_scale=10.0, v_discount=0.9)
        cnt += 1
        if cnt > 2:
            # tau = max(0.05, tau * 0.85)
            tau = max(0.05, tau * 0.95)
        if epid == 0:
            print("GHB", mcts_playout, cnt, f"tau:{tau:.2f}, {time.time()-stime: .2f}")
        stime = time.time()
        for i in range(mcts_playout):
            mcts.search(state)
        pi = mcts.pi(state, tau)
        samples.append([state, pi, None])
        action = np.random.choice(len(pi), p=pi)
        next_state, isend, reward = game.next_state(state, action)
        y = action // game.size
        x = action % game.size
        ####
        if board_record[y, x] != 0:
            print("####GHB Found Replicate move!", y, x, cnt, board_record)
            print(state[0])
            print("sample\n", samples)
            import sys

            sys.exit()
        ####
        board_record[y, x] = cnt
        if isend:
            v = reward
            for j in reversed(range(len(samples))):
                samples[j][2] = v
                v = -v
            return samples, board_record
        else:
            state = next_state


@ray.remote
def executeEpisodeEndless():
    try:
        uploader = Uploader()
        game = GoBang(size=15)
        traj_cnt = 0
        # playouts_choices = np.array([400000, 200000, 100000, 50000, 20000, 10000, 5000])
        playouts_choices = np.array([100000, 50000, 20000, 10000])
        playouts_p = playouts_choices / playouts_choices.sum()

        mcts_playout = int(np.random.choice(playouts_choices, p=playouts_p))

        while True:
            print("executeEpisodeEndless1", traj_cnt)
            trajectory, board_record = executeEpisode(game, 0, mcts_playout)
            print(
                "got trajectory",
                mcts_playout,
                "\n",
                board_record,
                "\n",
                trajectory[-1][2],
                trajectory[-2][2],
                trajectory[-3][2],
                trajectory[-4][2],
            )
            traj_cnt += 1
            if len(trajectory) % 2 == 1:
                winner = "black"
            else:
                winner = "white"
            # gobang_col1.insert_one(
            #     {"kifu": board_record.tolist(), "winner": winner, "len": len(trajectory), "traj_cnt": traj_cnt}
            # )
            precord = []
            for item in trajectory:
                precord.append(item[1].round(3).tolist())
            uploader.upload(
                json.dumps(precord),
                json.dumps(board_record.tolist()),
                winner,
                len(precord),
                traj_cnt,
                mcts_playout,
            )
            print(f"{winner} win!", traj_cnt, len(trajectory))

            # tainer.push_samples.remote(trajectory)
    except Exception:
        print(traceback.format_exc())


def main():
    print("GHB1")
    s = []
    for i in range(59):
        s.append(executeEpisodeEndless.remote())
    print("GHB4")
    ray.wait(s)
    print("GHB5")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

import pymongo
import numpy as np
import json
from game2 import GoBang


class Feeder:
    def __init__(self, db="gobang", col="kifu_pure2"):
        dbclient = pymongo.MongoClient("mongodb://root:mongomprc12@192.168.5.6:27017/")
        gobang_db = dbclient[db]
        self.gobang_col1 = gobang_db[col]

    def feed(self, trainer):
        game = GoBang()
        for x in self.gobang_col1.find():
            board_record = np.array(json.loads(x["kifu"])).astype(np.uint8)
            pi = np.array(json.loads(x["samples"])).astype(np.float16)
            # print(board_record)
            board_record = board_record.flatten()
            l = x["len"]
            if l % 2 == 1:  # black win
                v = 1
            else:
                v = -1
            state = game.start_state()
            trajectory = []
            # print(type(state))
            for i in range(1, l + 1):
                trajectory.append([state, pi[i - 1], v])
                v = -v
                pos = np.where(board_record == i)[0][0]
                state, _, _ = game.next_state(state, pos)

            trainer.push_games.remote(trajectory)
            # print(trajectory)
            # return


if __name__ == "__main__":
    feeder = Feeder()
    feeder.feed(None)

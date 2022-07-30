import pymongo


class Uploader:
    def __init__(self, dbname="gobang", colname="kifu_pure2"):
        dbclient = pymongo.MongoClient("mongodb://root:mongomprc12@192.168.5.6:27017/")
        gobang_db = dbclient[dbname]
        self.gobang_col1 = gobang_db[colname]

    def upload(self, samples, board_record, winner, traj_len, traj_cnt, mcts_playout):
        self.gobang_col1.insert_one(
            {
                "samples": samples,
                "kifu": board_record,
                "winner": winner,
                "len": traj_len,
                "traj_cnt": traj_cnt,
                "mcts_playout": mcts_playout,
            }
        )

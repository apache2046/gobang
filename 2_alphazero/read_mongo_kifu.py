import pymongo
import numpy as np
import json
from game3 import GoBang

dbclient = pymongo.MongoClient("mongodb://root:mongomprc12@192.168.5.6:27017/")
gobang_db = dbclient["gobang"]
gobang_col1 = gobang_db["kifu_pure2"]

cnt = 0
samples = []
game = GoBang()
for x in gobang_col1.find():
    board_record = np.array(json.loads(x['kifu'])).astype(np.uint8)
    # print(board_record)
    board_record = board_record.flatten()
    l = x['len']
    if l < 60:
        continue
    state = game.start_state()
    samples.append(state)
    # print(type(state))
    for i in range(1, l + 1):
        # print("G", i)
        pos = np.where(board_record == i)[0][0]
        state, _, _ = game.next_state(state, pos)
        samples.append(state)
    # print(x['kifu'])
    # y = x['kifu']
    # print(type(y)) #, type(y[0]), type(y[0,0]), y[0,0])
    # print(list(y))
    cnt += 1
    if cnt > 20000:
        break
    # break
print(cnt)
samples = np.stack(samples)
np.random.shuffle(samples)
print(samples.shape)
np.savez_compressed("3.npz", a=samples)

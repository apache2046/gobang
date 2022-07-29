import pymongo
import numpy as np

dbclient = pymongo.MongoClient("mongodb://root:mongomprc12@localhost:27017/")
go_db = dbclient['gobang']
go_col1 = go_db['kifu1']

for _ in range(100000):
    kifu = np.random.randint(0, 255, (15, 15), dtype=np.uint8)
    go_col1.insert_one({'kifu': kifu.tolist()})


cnt = 0
for x in go_col1.find():
    x = np.array(x['kifu']).astype(np.uint8)
    print(x)
    cnt += 1

print(cnt)

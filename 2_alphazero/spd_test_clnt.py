from multiprocessing.connection import Client
from io import BytesIO
import numpy as np
import time

address = ('192.168.5.6', 6001)     # family is deduced to be 'AF_INET'
result = np.random.randn(128,15*15)

state = np.random.randint(0, 2, (128,15,15,5)).astype(np.int8)
stime = time.time()
for _ in range(10000):
    with Client(address, authkey=b'secret password123') as conn:
        for _ in range(10):
            conn.send(('infer', state))
            ret = conn.recv()

print(time.time()-stime)
import redis
import numpy as np
import struct
import time

rdb = redis.Redis(host="192.168.5.6", port=1001, db=1)


def toRedis(r, a, n):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    h, w = a.shape
    shape = struct.pack(">II", h, w)
    encoded = shape + a.tobytes()

    # Store encoded data in Redis
    r.set(n, encoded)
    return


def fromRedis(r, n):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(n)
    h, w = struct.unpack(">II", encoded[:8])
    # Add slicing here, or else the array would differ from the original
    a = np.frombuffer(encoded[8:], dtype=np.int8).reshape(h, w)
    return a


stime = time.time()
state = np.random.randint(0, 2, (15, 15, 4), dtype=np.int8)
prob = np.random.randn(15, 15).astype(np.float32)
for _ in range(100000):
    k = bytes(state)
    rdb.set(k, bytes(prob))
    v = rdb.get(k)
print(f"time:{time.time() - stime: .3f}")

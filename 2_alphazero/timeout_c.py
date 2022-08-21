from multiprocessing.connection import Client
import time

stime = time.time()
try:
    address = ("192.168.5.106", 2022)
    data = "hello"
    with Client(address, authkey=b"secret password123") as conn:
        conn.send(data)
        conn.recv()
finally:
    print(time.time() - stime)

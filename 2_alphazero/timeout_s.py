from multiprocessing.connection import Listener
import time

stime = time.time()
try:
    address = ("0.0.0.0", 2022)

    with Listener(address, authkey=b"secret password123", backlog=100) as listener:
        while True:
            time.sleep(120)
            print("G0")
            with listener.accept() as conn:
                print("G1")
                data = conn.recv()
                print("G2")
                time.sleep(3600)
finally:
    print(time.time() - stime)

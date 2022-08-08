import os
import ray
import time
import socket

ray.init(address="auto", _node_ip_address="192.168.5.6")


@ray.remote(num_cpus=1)
def fn(idx):
    time.sleep(5)
    print(idx)
    return socket.gethostname(), os.getcwd()


wl = [fn.remote(i) for i in range(64)]
r = ray.wait(wl)
# print(r)
print(wl)

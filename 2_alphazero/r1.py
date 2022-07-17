import os
import ray
import time
import socket
from cc import ffn

ray.init(address='auto', _node_ip_address='192.168.5.6')
@ray.remote
def fn():
  time.sleep(2)
  return socket.gethostname(), os.getcwd()

@ray.remote
def fn2():
  time.sleep(2)
  return socket.gethostname(), os.getcwd(),ffn()


wl=[fn.remote() for _ in range(48)]
r = ray.get(wl)
print(r)
  
wl=[fn2.remote() for _ in range(48)]
r = ray.get(wl)
print(r)
  

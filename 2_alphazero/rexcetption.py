import os
import ray
import time
import socket
from cc import ffn

ray.init(address='auto', _node_ip_address='192.168.5.6')
@ray.remote
def fn():
  time.sleep(2)
  #x = 1.0 / 0.0
  return socket.gethostname(), os.getcwd()

@ray.remote
def fn2():
  time.sleep(2)
  return socket.gethostname(), os.getcwd(),ffn()

@ray.remote
class AA():
  def __init__(self):
    self.data = [1,2,3]
  def fn3(self, v):
    return v / 0.0


wl=[fn.remote() for _ in range(48)]
r = ray.get(wl)
print(r)
  
wl=[fn2.remote() for _ in range(48)]
r = ray.get(wl)
print(r)

aa = AA.remote() 
wl=[aa.fn3.remote(3.0) for _ in range(48)]
r = ray.get(wl)
print(r)

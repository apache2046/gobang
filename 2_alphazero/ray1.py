import time
import ray
import numpy as np


ray.init(address="auto", _node_ip_address="192.168.5.6")

@ray.remote
def fn(a):
   for i in range(10000):
      for j in range(10000):
          a += 1
   return a+1 

def main():
   fn_feature = []
   for i in range(48):
       fn_feature.append(fn.remote(i*100))
   print('before wait')
   
   ray.wait(fn_feature)


if __name__ == "__main__":
    main()

import time
import os
import ray
ray.init(num_cpus=4)
print("after ray.init")

@ray.remote
def f(x):
    # time.sleep(1)
    #return str(x * x) + ": " + str(os.getpid())
    return 1
@ray.remote
def f2(x):
    # time.sleep(1)
    return x * x

# for _ in range(4):
#     futures = [f.remote(i) for i in range(20)]
#     result = [ray.get(i) for i in futures]
#     for i in result:
#         print(i)
#     print("main:", os.getpid())
#     time.sleep(1)
for _ in range(4):
    stime = time.time()
    futures = [f.remote(i) for i in range(10000)]
    result = [ray.get(i) for i in futures]
    print("time:", time.time() - stime)


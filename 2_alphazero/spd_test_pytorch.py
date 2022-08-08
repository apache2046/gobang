import torch
from model4 import Policy_Value
import time

stime = time.time()
m = Policy_Value().to('cuda').half()

m.eval()
for _ in range(10000):
    data = torch.randn(128,5,15,15).to('cuda').half()
    with torch.no_grad():
        p, v = m(data)

print(time.time() - stime)

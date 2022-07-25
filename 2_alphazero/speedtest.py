import torch
import torchvision
import time
import numpy as np
from model4 import Policy_Value

stime = time.time()
data = torch.randn(256,5,15,15).half().to("cuda")
model = Policy_Value().half().to("cuda")

stime = time.time()
with torch.no_grad():
    for i in range(1000):
       # data = torch.tensor(datab[i], dtype=torch.float32).to("cuda")
        y = model(data)

print(time.time() - stime)


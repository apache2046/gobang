import torch
import torchvision
import time
import numpy as np

stime = time.time()
data = torch.randn(64, 3, 224, 224).to("cuda")
model = torchvision.models.resnet18().to("cuda")

# datab = np.random.randn(1000, 64, 3, 224, 224).astype(np.float32)
datab = np.ones((1000, 64, 3, 224, 224)).astype(np.float32)
stime = time.time()
with torch.no_grad():
    for i in range(1000):
        data = torch.tensor(datab[i], dtype=torch.float32).to("cuda")
        y = model(data)

print(time.time() - stime)

# data = torch.randn(64, 3, 224, 224).to("cuda").half()
# model = torchvision.models.resnet18().to("cuda").half()

# stime = time.time()
# with torch.no_grad():
#     for _ in range(1000):
#         y = model(data)

# print(time.time() - stime)

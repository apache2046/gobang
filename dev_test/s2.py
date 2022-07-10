import onnxruntime
import numpy as np
import time

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
ort = onnxruntime.InferenceSession("resnet18.onnx", sess_options=sess_options, providers=["CUDAExecutionProvider"])

print("prerun begin")
img = np.random.randn(64, 3, 224, 224)
ort.run(["output"], {"input": img.astype(np.float32)})
print("prerun end")
datab = np.ones((1000, 64, 3, 224, 224)).astype(np.float32)

stime = time.time()
for i in range(1000):
    # img = np.random.randn(64, 3, 224, 224)
    img = datab[i]
    y = ort.run(["output"], {"input": img})
print(time.time() - stime)


ort = onnxruntime.InferenceSession("resnet18_fp16.onnx", sess_options=sess_options, providers=["CUDAExecutionProvider"])

print("prerun begin")
img = np.random.randn(64, 3, 224, 224)
ort.run(["output"], {"input": img.astype(np.float16)})
print("prerun end")
datab = np.ones((1000, 64, 3, 224, 224)).astype(np.float16)

stime = time.time()
for i in range(1000):
    # img = np.random.randn(64, 3, 224, 224)
    img = datab[i]
    y = ort.run(["output"], {"input": img})
print(time.time() - stime)

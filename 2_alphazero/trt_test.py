import traceback
import trtcommon
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import model4
import torch

class Infer_srv:
    def __init__(self, model_file):
        # self.nnet = Policy_Value().to("cuda:0").half()
        # self.nnet.eval()
        self.load_onnx(model_file)

    def load_onnx(self, model_file):
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(trtcommon.EXPLICIT_BATCH)
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            parser = trt.OnnxParser(network, TRT_LOGGER)

            config.max_workspace_size = trtcommon.GiB(1)
            # Load the Onnx model and parse it in order to populate the TensorRT network.
            with open(model_file, "rb") as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
                engine = builder.build_engine(network, config)

            self.bindings = []

            for binding in engine:
                size = (
                    trt.volume(engine.get_binding_shape(binding))
                    * engine.max_batch_size
                )
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # Allocate host and device buffers
                # host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(size * 4)
                # Append the device buffer to device bindings.
                self.bindings.append(device_mem)

                # Append to the appropriate list.
                if engine.binding_is_input(binding):
                    print("inputmem", )
                else:
                    print("outputmem")
                print(engine.get_binding_shape(binding), size * 4)

            self.engin = engine
            self.context = engine.create_execution_context()
            self.stream = cuda.Stream()
        except Exception:
            print(traceback.format_exc())

    def infer(self, inputdata):
        try:
            # mem1 = cuda.mem_alloc(115200)
            # b1 = np.ones(115200).astype(np.int8)
            # cuda.memcpy_htod(mem1, b1)
            # cuda.memcpy_dtoh(b1, self.bindings[1])
            # print(b1)
            inputdata = np.stack(inputdata).transpose([0, 3, 1, 2]).astype(np.float32).ravel()
            bs = 128
            output1 = np.empty((bs, 15*15), np.float32)
            output2 = np.empty((bs, 1), np.float32)
            # print(self.bindings[0], type(self.bindings[0]))
            # print(inputdata, type(inputdata), inputdata.dtype)
            cuda.memcpy_htod_async(self.bindings[0], inputdata, self.stream)
            # Run inference.
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(output1, int(self.bindings[1]), self.stream)
            cuda.memcpy_dtoh_async(output2, int(self.bindings[2]), self.stream)
            # Synchronize the stream
            self.stream.synchronize()
            # cuda.memcpy_htod(self.bindings[0], inputdata)
            # Run inference.
            # self.context.execute_v2(bindings=self.bindings)
            # Transfer predictions back from the GPU.
            # print(self.bindings[1], type(self.bindings[1]))
            # print(output1, type(output1), output1.dtype)
            # cuda.memcpy_dtoh(output1, self.bindings[1])
            # cuda.memcpy_dtoh(output2, self.bindings[2])
            prob = output1
            v = output2
             
            return prob, v
        except Exception:
            print(traceback.format_exc())

def main():
    # indata = np.random.randn(256,5,15,15).astype(np.int8)
    indata = np.random.randint(0,2,(128, 15, 15, 5)).astype(np.int8)
    indata[:, :, :, 4] = 1
    infer_srv = Infer_srv("/ray_run/g1.onnx")
    # import time
    # stime = time.time()
    # for _ in range(1000):
    #     infer_srv.infer(indata)
    # print(time.time()-stime)

    m = model4.Policy_Value()
    m.load_state_dict(torch.load('models/224.pt'))
    m.eval()
    with torch.no_grad():
        d = torch.tensor(indata).to(torch.float32)
        d = d.permute(0,3,1,2)
        y11, y12 = m(d)
        y11 = y11.numpy()
        y12 = y12.numpy()
    y21, y22 = infer_srv.infer(indata)
    np.set_printoptions(precision=3, linewidth=1500)
    print(y11.shape, y11[0][:20])
    print(y21.shape, y21[0][:20])

    print(y12.ravel())
    print(y22.ravel())

    big = np.abs(y11-y21).argmax()
    print(big)
    y = big // 225
    x = big % 225
    print(np.abs(y11-y21).max(), y11[y][x], y21[y][x], y11[y][x]-y21[y][x])

if __name__ == '__main__':
    main()

import traceback
import trtcommon
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import model4
import torch
from multiprocessing.connection import Client
from io import BytesIO

class Infer_srv:
    def __init__(self):
        pass

    def load_onnx(self, model_bytes):
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(trtcommon.EXPLICIT_BATCH)
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            parser = trt.OnnxParser(network, TRT_LOGGER)

            config.max_workspace_size = trtcommon.GiB(1)
            # Load the Onnx model and parse it in order to populate the TensorRT network.
            # with open(model_file, "rb") as model:
            if not parser.parse(model_bytes):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
            engine = builder.build_engine(network, config)

            self.bindings = []

            for binding_name in engine:
                size = (
                    trt.volume(engine.get_binding_shape(binding_name))
                    * engine.max_batch_size
                )
                dtype = trt.nptype(engine.get_binding_dtype(binding_name))
                # Allocate host and device buffers
                # host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(size * 4)
                # Append the device buffer to device bindings.
                self.bindings.append(device_mem)

                # Append to the appropriate list.
                if engine.binding_is_input(binding_name):
                    print("inputmem", binding_name, engine.get_binding_shape(binding_name), size * 4)
                else:
                    print("outputmem", binding_name, engine.get_binding_shape(binding_name), size * 4)
                # print(engine.get_binding_shape(binding_name), size * 4)

            self.engin = engine
            self.context = engine.create_execution_context()
            self.stream = cuda.Stream()
        except Exception:
            print(traceback.format_exc())

    def infer(self, inputdata):
        try:

            bs = len(inputdata)
            inputdata = np.stack(inputdata).transpose([0, 3, 1, 2]).astype(np.float32).ravel()
            output1 = np.empty((bs, 15*15), np.float32)
            output2 = np.empty((bs, 1), np.float32)

            cuda.memcpy_htod_async(self.bindings[0], inputdata, self.stream)
            # Run inference.
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(output1, int(self.bindings[1]), self.stream)
            cuda.memcpy_dtoh_async(output2, int(self.bindings[2]), self.stream)
            # Synchronize the stream
            self.stream.synchronize()
            
            prob = output1
            v = output2
             
            return prob, v
        except Exception:
            print(traceback.format_exc())

def get_onnx_bytes_from_remote(model):
    model.eval()
    with Client(('192.168.5.6', 6000) , authkey=b'secret password123') as conn:
        f = BytesIO()
        model.load_state_dict(torch.load('models/224.pt'))
        model.eval()
        torch.save(model.state_dict(), f, _use_new_zipfile_serialization=False)
        conn.send(f.getvalue())
        conn.send(128)
        onnxbytes = conn.recv()
        return onnxbytes

def main():
    # indata = np.random.randn(256,5,15,15).astype(np.int8)
    indata = np.random.randint(0,2,(128, 15, 15, 5)).astype(np.int8)
    indata[:, :, :, 4] = 1
    
    m = model4.Policy_Value()
    m.load_state_dict(torch.load('models/224.pt'))
    onnxbytes = get_onnx_bytes_from_remote(m)

    infer_srv = Infer_srv()

    infer_srv.load_onnx(onnxbytes)

    # import time
    # stime = time.time()
    # for _ in range(1000):
    #     infer_srv.infer(indata)
    # print("time:", time.time()-stime)


    with torch.no_grad():
        d = torch.tensor(indata).to(torch.float32)
        d = d.permute(0,3,1,2)
        y11, y12 = m(d)
        y11 = y11.numpy()
        y12 = y12.numpy()
    y21, y22 = infer_srv.infer(indata)
    np.set_printoptions(precision=2, linewidth=1500)
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

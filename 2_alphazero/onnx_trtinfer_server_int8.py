# Should run in pytorch 1.2
from multiprocessing.connection import Listener
import pycuda.autoinit
import pycuda.driver as cuda
import traceback
import tensorrt as trt
import trtcommon
import numpy as np
import argparse
from datetime import datetime


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None
        x = np.load("1.npz")
        self.data = x["a"]
        self.cnt = 0
        self.batch_allocation = cuda.mem_alloc(4 * 128 * 15 * 15 * 5)

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        return 128

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """

        try:
            batch = self.data[128 * self.cnt : 128 * self.cnt + 128]
            batch = batch.astype(np.float32)
            cuda.memcpy_htod(self.batch_allocation, batch)
            return [int(self.batch_allocation)]
        except StopIteration:
            print("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        # if os.path.exists(self.cache_file):
        #     with open(self.cache_file, "rb") as f:
        #         print("Using calibration cache file: {}".format(self.cache_file))
        #         return f.read()
        print("int read_calibration_cache")

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        # with open(self.cache_file, "wb") as f:
        #     print("Writing calibration cache data to: {}".format(self.cache_file))
        #     f.write(cache)
        print("int write_calibration_cache", cache)


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
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = EngineCalibrator("./calibration.cache")
            parser = trt.OnnxParser(network, TRT_LOGGER)

            config.max_workspace_size = trtcommon.GiB(2)
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
                size = trt.volume(engine.get_binding_shape(binding_name)) * engine.max_batch_size
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
            output1 = np.empty((bs, 15 * 15), np.float32)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    address = ("0.0.0.0", args.port)  # family is deduced to be 'AF_INET'
    print(address)
    infer_srv = Infer_srv()
    with Listener(address, authkey=b"secret password123") as listener:
        while True:
            with listener.accept() as conn:
                command, data = conn.recv()
                if command == "infer":
                    result = infer_srv.infer(data)
                    conn.send(result)
                elif command == "load_onnx":
                    print("load_onnx1", datetime.now().strftime("%D %H:%M:%S"))
                    infer_srv.load_onnx(data)
                    print("load_onnx2\n")
                    conn.send("ok")
                else:
                    print("invalid cmd:", command, data)


if __name__ == "__main__":
    main()

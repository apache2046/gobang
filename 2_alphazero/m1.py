import tensorrt as trt
import trtcommon
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(trtcommon.EXPLICIT_BATCH)
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
parser = trt.OnnxParser(network, TRT_LOGGER)

config.max_workspace_size = trtcommon.GiB(1)
# Load the Onnx model and parse it in order to populate the TensorRT network.
with open("g1.onnx", "rb") as model:
    if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))

engine = builder.build_engine(network, config)

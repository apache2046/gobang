import torch
import torchvision

model = torchvision.models.resnet18()
dummy_input = torch.randn(128, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "resnet1.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    verbose=True,
    opset_version=9,  # the ONNX version to export the model to
    # do_constant_folding=True,  # whether to execute constant folding for optimization
    # input_names=["input"],  # the model's input names
    # output_names=["output"],  # the model's output names
    # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
)

# model = model.half().to('cuda')
# dummy_input = dummy_input.to(torch.float16).to('cuda')
# torch.onnx.export(
#     model,
#     dummy_input,
#     "resnet2.onnx",  # where to save the model (can be a file or file-like object)
#     export_params=True,  # store the trained parameter weights inside the model file
#     opset_version=16,  # the ONNX version to export the model to
#     do_constant_folding=True,  # whether to execute constant folding for optimization
#     input_names=["input"],  # the model's input names
#     output_names=["output"],  # the model's output names
#     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
# )
